#!/usr/bin/env python
# advanced_ensemble.py — Point72-grade ensemble engine
"""
Four pillars of institutional-grade portfolio construction:

  1. Half-Life Decay: Multi-timeframe fusion with exponential decay weighting
  2. XGBoost Ensemble: Nonlinear replacement for linear school voting
  3. Barra Style Neutralization: Size/Momentum/Vol beta exposure penalty
  4. Ledoit-Wolf + L1 Turnover: Convex optimization with transaction cost
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
from scipy.optimize import minimize


# =============================================================================
# 1. Multi-Timeframe Half-Life Decay
# =============================================================================

@dataclass
class HalfLifeDecay:
    """
    Exponential decay weighting for signals at different time scales.

    Signal weight = exp(-ln(2) × lag / half_life)

    Intuition:
      - 5-min order flow: half_life = 12h (decays fast — last ~1 day)
      - Daily technical:    half_life = 5d  (decays slowly — last ~2 weeks)
      - Fundamental/value:  half_life = 60d (stable — lasts months)
    """

    # Half-lives in hours for different time scales
    HALF_LIVES = OrderedDict({
        'intraday':    12,     # 5-min / 30-min signals
        'daily':       120,    # Daily technical indicators (5 trading days)
        'weekly':      600,    # Weekly signals (25 trading days)
        'fundamental': 1440,   # Value/fundamental (60 trading days)
    })

    def weight(self, scale: str, lag_hours: float) -> float:
        """
        Compute decay weight for a signal at given scale and lag.

        Args:
            scale: 'intraday' | 'daily' | 'weekly' | 'fundamental'
            lag_hours: hours since signal was generated

        Returns:
            Weight ∈ (0, 1]
        """
        hl = self.HALF_LIVES.get(scale, 120)
        return float(np.exp(-np.log(2) * lag_hours / hl))

    def decay_weights(self, scale: str, n_signals: int,
                      interval_hours: float = 1.0) -> np.ndarray:
        """
        Generate decay weight vector for n_signals at given interval.

        Returns:
            Array of weights [w_0, w_1, ..., w_{n-1}] where w_0 is newest.
        """
        lags = np.arange(n_signals) * interval_hours
        return np.exp(-np.log(2) * lags / self.HALF_LIVES.get(scale, 120))

    def fuse_multi_scale(self, signals: Dict[str, float],
                         lags: Dict[str, float]) -> float:
        """
        Fuse signals from multiple time scales with decay weighting.

        Args:
            signals: {'daily_macd': 0.7, 'weekly_trend': 0.3, '5min_orderflow': 0.9}
            lags: {'daily_macd': 4.0, 'weekly_trend': 24.0, '5min_orderflow': 0.5}

        Returns:
            Fused signal score
        """
        total_weight = 0.0
        fused_score = 0.0

        scale_map = {
            '5min': 'intraday', '30min': 'intraday', 'orderflow': 'intraday',
            'daily': 'daily', 'macd': 'daily', 'rsi': 'daily',
            'weekly': 'weekly', 'trend': 'weekly',
            'fundamental': 'fundamental', 'value': 'fundamental',
        }

        for name, sig in signals.items():
            lag = lags.get(name, 0.0)
            # Determine scale from name
            scale = 'daily'  # Default
            for keyword, s in scale_map.items():
                if keyword in name.lower():
                    scale = s
                    break
            w = self.weight(scale, lag)
            fused_score += w * sig
            total_weight += w

        return fused_score / max(total_weight, 1e-10)


# =============================================================================
# 2. XGBoost Nonlinear Ensemble
# =============================================================================

@dataclass
class NonLinearEnsemble:
    """
    Replace linear school voting with gradient-boosted trees.

    Features: 15 school direction scores + confidence values (30 features)
    Label: 1 if trade was profitable, 0 if not
    """

    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    _model: Optional[object] = None
    _feature_names: List[str] = field(default_factory=list)

    def build_features(self, school_signals: Dict[str, Dict]) -> np.ndarray:
        """
        Build feature vector from school signals.

        For each school: [direction_score, confidence]
        where direction_score = +score for bullish, -score for bearish, 0 for neutral

        Returns:
            (1, 2N) feature array
        """
        features = []
        names = []
        for name in sorted(school_signals.keys()):
            sig = school_signals[name]
            direction = sig.get('direction', 'neutral')
            score = sig.get('score', 0.0)
            conf = sig.get('confidence', 0.0)

            # Encode direction as signed score
            if direction == 'bullish':
                dir_score = score
            elif direction == 'bearish':
                dir_score = -score
            else:
                dir_score = 0.0

            features.extend([dir_score, conf])
            names.extend([f'{name}_dir', f'{name}_conf'])

        self._feature_names = names
        return np.array(features).reshape(1, -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NonLinearEnsemble":
        """
        Train XGBoost model on historical school signals.

        Args:
            X: (N, 2M) feature matrix — N trades, M schools × 2 features
            y: (N,) binary labels — 1 = profitable, 0 = not
        """
        try:
            import xgboost as xgb
            self._model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=0,
            )
            self._model.fit(X, y)

            # Store feature importance
            self.feature_importance = dict(zip(
                self._feature_names,
                self._model.feature_importances_
            ))
        except ImportError:
            print("[NonLinearEnsemble] XGBoost not installed — falling back to LR")
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(C=1.0, max_iter=1000)
            self._model.fit(X, y)

        return self

    def predict(self, school_signals: Dict[str, Dict]) -> Tuple[str, float]:
        """
        Predict trade direction and probability.

        Returns:
            (direction, probability): 'bullish' | 'bearish' | 'neutral', prob
        """
        if self._model is None:
            return 'neutral', 0.0

        X = self.build_features(school_signals)

        if hasattr(self._model, 'predict_proba'):
            proba = self._model.predict_proba(X)[0]
            bull_prob = proba[1] if len(proba) > 1 else proba[0]
        else:
            bull_prob = float(self._model.predict(X)[0])

        if bull_prob > 0.55:
            return 'bullish', bull_prob
        elif bull_prob < 0.45:
            return 'bearish', 1 - bull_prob
        return 'neutral', 0.5

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return top N most important features."""
        if not hasattr(self, 'feature_importance'):
            return []
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1], reverse=True
        )
        return sorted_features[:n]


# =============================================================================
# 3. Barra Style Factor Neutralization
# =============================================================================

@dataclass
class BarraStyleNeutralizer:
    """
    Compute and penalize style factor exposures.

    Three-factor Barra-lite model:
      1. Size (市值): log(market_cap) — small vs large
      2. Momentum (动量): 12-month return
      3. Volatility (波动率): daily return std

    Target: portfolio exposure to each factor ≤ threshold (in Z-score).
    """

    size_threshold: float = 0.5    # Max size Z-score exposure
    momentum_threshold: float = 0.5
    volatility_threshold: float = 0.5

    def compute_factor_exposures(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute portfolio exposure to each style factor.

        Args:
            weights: (N,) portfolio weights
            factor_loadings: (N, K) factor loadings per asset

        Returns:
            (exposures, z_scores): raw and Z-scored factor exposures
        """
        exposures = factor_loadings.T @ weights  # (K,) = (K, N) × (N,)
        # Z-score relative to equal-weight benchmark
        equal_w = np.ones(len(weights)) / len(weights)
        benchmark_exposures = factor_loadings.T @ equal_w
        z_scores = exposures - benchmark_exposures
        return exposures, z_scores

    def penalty(
        self,
        weights: np.ndarray,
        factor_loadings: np.ndarray,
        penalty_strength: float = 1.0,
    ) -> float:
        """
        Compute style exposure penalty for optimization.

        Penalty = Σ_k max(0, |z_k| - threshold_k)²

        This softly constrains each factor Z-score to be within threshold.
        """
        _, z_scores = self.compute_factor_exposures(weights, factor_loadings)

        thresholds = np.array([
            self.size_threshold,
            self.momentum_threshold,
            self.volatility_threshold,
        ])

        # Only penalize exposures exceeding threshold
        excess = np.maximum(0, np.abs(z_scores[:len(thresholds)]) - thresholds)
        return penalty_strength * np.sum(excess ** 2)

    def estimate_factor_loadings(
        self,
        symbols: List[str],
        daily_data: Dict[str, pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Estimate Barra factor loadings from available data.

        Returns:
            (N, K) loadings matrix, factor_names
        """
        N = len(symbols)
        K = 3  # Size, Momentum, Volatility
        loadings = np.zeros((N, K))

        for i, sym in enumerate(symbols):
            # Size proxy: use 1/sqrt(avg_volume) as inverse size proxy
            # (smaller avg vol → smaller cap)
            if daily_data and sym in daily_data:
                df = daily_data[sym]
                if len(df) >= 20:
                    # Size: inverse of avg dollar volume as proxy
                    avg_amount = (df['volume'] * df['close']).tail(20).mean()
                    loadings[i, 0] = -np.log(max(avg_amount, 1))
                    # Momentum: 20-day return
                    loadings[i, 1] = df['close'].pct_change(20).iloc[-1]
                    # Volatility: 20-day return std
                    loadings[i, 2] = df['close'].pct_change().tail(20).std()
                else:
                    loadings[i] = [0.0, 0.0, 0.01]
            else:
                loadings[i] = [0.0, 0.0, 0.01]

        # Z-score normalize factor loadings
        for k in range(K):
            std = loadings[:, k].std()
            if std > 1e-10:
                loadings[:, k] = (loadings[:, k] - loadings[:, k].mean()) / std

        return loadings, ['Size', 'Momentum', 'Volatility']


# =============================================================================
# 4. Ledoit-Wolf + L1 Turnover Optimization
# =============================================================================

@dataclass
class TurnoverAwareOptimizer:
    """
    Minimum-variance optimization with turnover penalty.

    Objective:
      min  w'Σw  +  λ_turnover × Σ|w - w_prev|  +  λ_style × StylePenalty

    Subject to:
      Σw = 1  (fully invested relative weights)
      w ≥ 0   (long-only)

    The L1 term Σ|w - w_prev| penalizes turnover — each basis point of
    weight change costs transaction cost, discouraging high-frequency churn.
    """

    turnover_penalty: float = 0.50    # λ_turnover — higher = less churn
    style_penalty: float = 0.10       # λ_style — Barra deviation penalty
    max_weight: float = 0.10          # Single asset cap (10%)

    def optimize(
        self,
        returns: np.ndarray,
        prev_weights: np.ndarray,
        factor_loadings: np.ndarray = None,
        neutralizer: BarraStyleNeutralizer = None,
    ) -> np.ndarray:
        """
        Run the full optimization.

        Args:
            returns: (T, N) asset returns
            prev_weights: (N,) previous portfolio weights
            factor_loadings: (N, K) Barra factor loadings (optional)
            neutralizer: BarraStyleNeutralizer instance (optional)

        Returns:
            (N,) optimal weights
        """
        from covariance_shrinkage import robust_covariance

        N = returns.shape[1]
        cov = robust_covariance(returns, method='ledoit_wolf')
        prev_w = np.asarray(prev_weights, dtype=np.float64)

        def objective(w):
            # Risk term
            risk = w @ cov @ w
            # Turnover penalty (L1)
            turnover = self.turnover_penalty * np.sum(np.abs(w - prev_w))
            # Style penalty
            style = 0.0
            if neutralizer is not None and factor_loadings is not None:
                style = neutralizer.penalty(w, factor_loadings,
                                           self.style_penalty)
            return risk + turnover + style

        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, self.max_weight) for _ in range(N)]

        x0 = prev_w if prev_w.sum() > 0 else np.ones(N) / N

        result = minimize(
            objective, x0, method='SLSQP',
            constraints=constraints, bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12},
        )

        if result.success:
            w_opt = result.x
            w_opt /= w_opt.sum()  # Re-normalize
            return w_opt
        else:
            return prev_w  # Fallback: keep current weights

    def turnover_analysis(self, w_new: np.ndarray,
                          w_old: np.ndarray) -> Dict:
        """Analyze turnover between two weight vectors."""
        turnover = np.sum(np.abs(w_new - w_old)) / 2  # Two-way turnover
        changed = np.sum(np.abs(w_new - w_old) > 1e-6)
        return {
            'two_way_turnover': round(turnover, 4),
            'fraction_changed': round(changed / len(w_new), 3),
            'max_delta': round(float(np.max(np.abs(w_new - w_old))), 4),
        }


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    # ---- 1. Half-Life Decay ----
    decay = HalfLifeDecay()
    print("=== Half-Life Decay ===")
    for scale in ['intraday', 'daily', 'weekly', 'fundamental']:
        w24h = decay.weight(scale, 24)
        print(f"  {scale:12}: weight after 24h = {w24h:.3f}")

    # ---- 2. XGBoost Ensemble ----
    print("\n=== XGBoost Ensemble ===")
    nle = NonLinearEnsemble(n_estimators=20, max_depth=3)
    # Simulate training data
    X_train = np.random.randn(500, 28)  # 14 schools × 2 features
    y_train = (X_train[:, 0] + X_train[:, 2] + np.random.randn(500) * 0.5 > 0).astype(int)
    nle.fit(X_train, y_train)
    # Simulate school signals
    schools = {f'school_{i}': {'direction': 'bullish' if np.random.rand() > 0.5 else 'neutral',
                                'score': np.random.rand(), 'confidence': np.random.rand()}
               for i in range(14)}
    direction, prob = nle.predict(schools)
    print(f"  Prediction: {direction} ({prob:.0%})")

    # ---- 3. Barra Neutralization ----
    print("\n=== Barra Style Neutralization ===")
    neutralizer = BarraStyleNeutralizer()
    N = 10
    factor_loadings = np.random.randn(N, 3)
    w = np.ones(N) / N
    exposures, z = neutralizer.compute_factor_exposures(w, factor_loadings)
    penalty = neutralizer.penalty(w, factor_loadings)
    print(f"  Equal-weight: exposures={exposures}, z={z}, penalty={penalty:.4f}")

    # ---- 4. Turnover-Aware Optimization ----
    print("\n=== Turnover-Aware Optimization ===")
    returns = np.random.randn(50, 10) * 0.02
    prev_w = np.ones(10) / 10
    opt = TurnoverAwareOptimizer(turnover_penalty=0.5, max_weight=0.15)
    w_new = opt.optimize(returns, prev_w)
    analysis = opt.turnover_analysis(w_new, prev_w)
    print(f"  Turnover: {analysis['two_way_turnover']:.4f}, "
          f"changed: {analysis['fraction_changed']:.1%}")

    print("\nAll advanced ensemble modules: OK")
