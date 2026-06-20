#!/usr/bin/env python
# ranker_ensemble.py — XGBRanker + Half-Life Decay + SHAP Explainability
"""
Three institutional ML upgrades:

  1. XGBRanker: Cross-sectional ranking (not binary classification)
     - Target: quintile based on 5-day excess return vs CSI 300
     - Objective: rank:pairwise
     - Output: relative strength score per stock (not 0/1)

  2. Half-Life Decay: Multi-frequency feature EMA decay
     - High-freq (orderflow): EMA over past 3 periods, strong decay
     - Low-freq (classical MACD): raw values, no decay

  3. SHAP Explainer: Real-time factor attribution
     - TreeExplainer on trained model
     - Top 3 contributing factors per prediction
     - Injected into signal's resonance_detail
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. XGBRanker — Cross-Sectional Ranking Labels (Vol-Scaled)
# =============================================================================

def compute_vol_scaled_ret(
    df_panel: pd.DataFrame,
    excess_col: str = 'excess_ret',
    atr_col: str = None,
    atr_period: int = 20,
) -> pd.Series:
    """
    Volatility-Scaled Excess Return.

    target_i = excess_ret_i / σ_i    where σ_i = rolling 20-day ATR% or realized vol.

    Why vol-scaling?
      - Raw 5-day excess is dominated by high-beta noise stocks (妖股)
      - Dividing by ATR% normalizes: a 3% excess on a 1% ATR stock ranks higher
        than a 3% excess on a 5% ATR stock
      - Prevents XGBoost from fitting to volatile lottery tickets
    """
    if atr_col and atr_col in df_panel.columns:
        vol_proxy = df_panel[atr_col].clip(lower=0.005)  # Floor at 0.5% ATR
    elif 'atr_pct' in df_panel.columns:
        vol_proxy = df_panel['atr_pct'].clip(lower=0.005)
    else:
        # Fallback: compute from date-stock grouped close if available
        if 'close' in df_panel.columns and 'date' in df_panel.columns:
            df = df_panel.copy()
            df = df.sort_values(['symbol', 'date'])
            df['ret'] = df.groupby('symbol')['close'].pct_change()
            vol_proxy = df.groupby('symbol')['ret'].rolling(
                atr_period, min_periods=5).std().reset_index(level=0, drop=True).clip(lower=0.005)
        else:
            # No price data — use identity
            return df_panel[excess_col]

    raw_excess = df_panel[excess_col]
    vol_scaled = raw_excess / vol_proxy
    # Winsorize at ±5σ to prevent outliers
    vol_scaled = vol_scaled.clip(lower=-5.0, upper=5.0)
    return vol_scaled


def build_ranking_labels(
    df_panel: pd.DataFrame,
    n_bins: int = 5,
    excess_col: str = 'excess_ret',
    date_col: str = 'date',
    vol_scale: bool = True,
) -> pd.Series:
    """
    Convert excess returns into cross-sectional ranking labels.

    When vol_scale=True (default): target = excess_ret / ATR% → ranks
    risk-adjusted alpha, not noisy absolute returns.

    For each trading day:
      1. Compute volatility-scaled excess
      2. Sort all stocks, assign quintile scores: top 20% = 5, bottom 20% = 1
    """
    if vol_scale and excess_col in df_panel.columns:
        target_values = compute_vol_scaled_ret(df_panel, excess_col)
    else:
        target_values = df_panel[excess_col]

    labels = pd.Series(0, index=df_panel.index, dtype=int)
    dates = sorted(df_panel[date_col].unique())

    for d in dates:
        mask = df_panel[date_col] == d
        n_stocks = mask.sum()
        if n_stocks < n_bins * 3:
            continue

        daily_vals = target_values.loc[mask].values
        # Drop NaN/Inf before qcut
        finite_arr = np.isfinite(daily_vals)
        if finite_arr.sum() < n_bins * 3:
            continue

        # Build a sub-index that maps back into the full labels Series
        sub_idx = labels.index[mask][finite_arr]
        try:
            bin_labels = pd.qcut(daily_vals[finite_arr], n_bins,
                                labels=range(1, n_bins + 1), duplicates='drop')
            labels.loc[sub_idx] = bin_labels.astype(int)
        except ValueError:
            bins = np.percentile(daily_vals[finite_arr],
                                np.linspace(0, 100, n_bins + 1))
            bin_labels = np.digitize(daily_vals[finite_arr], bins[1:-1]) + 1
            labels.loc[sub_idx] = bin_labels

    return labels


def train_xgb_ranker(
    df_panel: pd.DataFrame,
    feature_cols: List[str],
    n_bins: int = 5,
    n_estimators: int = 80,
    max_depth: int = 4,
    learning_rate: float = 0.03,
) -> Tuple[Optional[object], Optional[object], List[str]]:
    """
    Train an XGBRanker on cross-sectional ranking data.

    Uses group-based ranking: stocks on the same day form a group.
    The model learns to order stocks within each daily group.

    Returns:
        (ranker, shap_explainer, feature_names)
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("[Ranker] XGBoost not installed")
        return None, None, []

    if len(df_panel) < 100:
        print("[Ranker] Insufficient data")
        return None, None, []

    # Build ranking labels
    df = df_panel.copy()
    df['rank_label'] = build_ranking_labels(df, n_bins=n_bins)
    df = df.dropna(subset=feature_cols + ['rank_label'])

    if len(df) < 100:
        return None, None, []

    # Feature matrix
    X = df[feature_cols].values.astype(np.float32)

    # Sort by date for proper group construction
    df = df.sort_values('date')
    dates = df['date'].values
    unique_dates, group_sizes = np.unique(dates, return_counts=True)

    # Filter out small groups (< 5 stocks)
    valid_idx = np.isin(dates, unique_dates[group_sizes >= 5])
    df = df[valid_idx]
    X = df[feature_cols].values.astype(np.float32)
    dates = df['date'].values
    unique_dates, group_sizes = np.unique(dates, return_counts=True)

    # XGBRanker expects group as array of SIZES (not cumulative IDs)
    group = group_sizes.astype(np.int32)
    y = df['rank_label'].values.astype(int) - 1  # 0-indexed for XGBoost

    # Train XGBRanker
    params = {
        'objective': 'rank:pairwise',
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'n_estimators': n_estimators,
        'subsample': 0.70,
        'colsample_bytree': 0.60,
        'min_child_weight': 10,
        'gamma': 0.5,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'verbosity': 0,
        'random_state': 42,
    }

    ranker = xgb.XGBRanker(**params)
    ranker.fit(X, y, group=group, verbose=False)

    # Build SHAP explainer
    try:
        import shap
        explainer = shap.TreeExplainer(ranker, feature_perturbation='tree_path_dependent')
    except ImportError:
        explainer = None

    print(f"[Ranker] Trained on {len(df)} rows, {len(unique_dates)} daily groups, "
          f"{len(feature_cols)} features")

    return ranker, explainer, feature_cols


# =============================================================================
# 2. Half-Life Decay for Multi-Frequency Features
# =============================================================================

@dataclass
class HalfLifeFeatureProcessor:
    """
    Apply half-life decay to features based on their time scale.

    High-frequency signals (orderflow, intraday) decay fast → use recent EMA.
    Low-frequency signals (value, fundamentals) stay stable → use raw values.

    EMA_weight = exp(-ln(2) × lag / half_life)
    """

    # Half-lives in trading periods
    HALF_LIVES = {
        'intraday':    2,      # ~2 periods = 2 days at 5-day sampling
        'daily':       10,     # ~10 periods = 50 days
        'weekly':      50,     # ~50 periods
        'fundamental': 200,    # ~200 periods = 4 years
    }

    # Feature-to-scale mapping by keyword
    SCALE_MAP = {
        'orderflow': 'intraday',
        'volume_delta': 'intraday',
        'chanlun': 'daily',
        'classical': 'daily',
        'macd': 'daily',
        'rsi': 'daily',
        'risk': 'weekly',
        'gann': 'weekly',
        'wyckoff': 'daily',
        'tang': 'daily',
        'livermore': 'daily',
        'busch': 'daily',
        'harmonic': 'daily',
        'roc': 'daily',
        'volume_profile': 'daily',
        'fusion': 'daily',
        'mean_reversion': 'daily',
        'capital_flow': 'intraday',
        'value': 'fundamental',
    }

    def __init__(self):
        self._ema_cache: Dict[str, float] = {}  # feature_name → last EMA value

    def get_scale(self, feature_name: str) -> str:
        """Determine time scale from feature name keywords."""
        for keyword, scale in self.SCALE_MAP.items():
            if keyword in feature_name:
                return scale
        return 'daily'

    def process(self, feature_name: str, raw_value: float,
                lag: int = 1) -> float:
        """
        Apply half-life decay to a feature value.

        Args:
            feature_name: e.g. 'school_orderflow_dir'
            raw_value: current raw feature value
            lag: periods since last update

        Returns:
            Decayed feature value
        """
        scale = self.get_scale(feature_name)
        hl = self.HALF_LIVES.get(scale, 10)

        # EMA weight
        alpha = np.exp(-np.log(2) * lag / hl)
        alpha = max(0.05, min(0.95, alpha))  # Clamp to avoid degenerate values

        cache_key = feature_name
        prev = self._ema_cache.get(cache_key, raw_value)
        ema_value = alpha * raw_value + (1 - alpha) * prev
        self._ema_cache[cache_key] = ema_value

        return ema_value

    def process_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Process a full feature dict with half-life decay."""
        return {name: self.process(name, val) for name, val in features.items()}


# =============================================================================
# 3. SHAP Explainability Injection
# =============================================================================

@dataclass
class SHAPExplainer:
    """
    Real-time factor attribution for AI predictions.

    Predict → compute SHAP → extract Top 3 contributors → format as string.
    """

    explainer: Optional[object] = None   # shap.TreeExplainer
    feature_names: List[str] = field(default_factory=list)

    def explain(self, features: np.ndarray) -> Dict:
        """
        Compute SHAP values for a single prediction.

        Args:
            features: (1, N) feature array

        Returns:
            Dict with top_3 factors and formatted explanation string
        """
        if self.explainer is None or len(features) == 0:
            return {'top_factors': [], 'explanation': '', 'shap_values': None}

        try:
            shap_values = self.explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Multi-class case

            sv = shap_values[0] if shap_values.ndim > 1 else shap_values

            # Find Top 3 contributing factors by absolute SHAP value
            contributions = [(self.feature_names[i], float(sv[i]))
                           for i in range(min(len(self.feature_names), len(sv)))]
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)

            top_3 = contributions[:3]

            # Format explanation string
            total_abs = sum(abs(c[1]) for c in contributions) + 1e-10
            parts = []
            for name, val in top_3:
                direction = '驱动看多' if val > 0 else '压制看空'
                pct = abs(val) / total_abs * 100
                school_name = name.replace('school_', '').replace('_dir', '').replace('_conf', '')
                parts.append(f'{school_name}({direction},{pct:.0f}%)')

            explanation = '[AI归因] ' + ' + '.join(parts)

            return {
                'top_factors': top_3,
                'explanation': explanation,
                'shap_values': sv,
            }
        except Exception:
            return {'top_factors': [], 'explanation': '', 'shap_values': None}


# =============================================================================
# 4. Unified Ranker Pipeline
# =============================================================================

class InstitutionRanker:
    """
    Production pipeline: XGBRanker + Half-Life Decay + SHAP.

    Usage:
        ranker = InstitutionRanker()
        ranker.train(parquet_path='data/ml_features.parquet')
        score, explanation = ranker.predict(school_signals)
    """

    def __init__(self):
        self._ranker: Optional[object] = None
        self._shap: Optional[SHAPExplainer] = None
        self._hl_processor = HalfLifeFeatureProcessor()
        self._feature_cols: List[str] = []
        self._trained = False

    def train(self, parquet_path: str = None) -> bool:
        """Train XGBRanker from ML panel data."""
        try:
            df = pd.read_parquet(parquet_path) if parquet_path else None
            if df is None or len(df) < 100:
                return False
        except Exception:
            return False

        # Identify feature columns
        feature_cols = sorted([c for c in df.columns
                              if c.endswith('_dir') or c.endswith('_conf')])
        if len(feature_cols) < 5:
            return False

        self._feature_cols = feature_cols

        # Train ranker
        ranker, explainer, cols = train_xgb_ranker(df, feature_cols)
        if ranker is None:
            return False

        self._ranker = ranker
        self._shap = SHAPExplainer(explainer=explainer, feature_names=feature_cols)
        self._trained = True
        return True

    def predict_rank_score(self, school_signals: Dict[str, Dict]) -> float:
        """
        Predict cross-sectional relative strength score.

        Returns:
            Score ∈ [1, 5] — higher = stronger relative to peers
        """
        if not self._trained:
            return 3.0  # Neutral

        features = self._extract_features(school_signals)
        features = self._hl_processor.process_features(features)
        X = np.array([[features.get(c, 0.0) for c in self._feature_cols]])

        try:
            pred = self._ranker.predict(X)
            score = float(pred[0]) if len(pred) > 0 else 3.0
            # Clamp to [1, 5]
            return max(1.0, min(5.0, score))
        except Exception:
            return 3.0

    def predict_with_shap(self, school_signals: Dict[str, Dict]) -> Dict:
        """Predict score + SHAP explanation."""
        score = self.predict_rank_score(school_signals)

        shap_result = {'top_factors': [], 'explanation': '', 'shap_values': None}
        if self._shap is not None and self._trained:
            features = self._extract_features(school_signals)
            features = self._hl_processor.process_features(features)
            X = np.array([[features.get(c, 0.0) for c in self._feature_cols]])
            shap_result = self._shap.explain(X)

        return {
            'rank_score': round(score, 2),
            'shap_explanation': shap_result.get('explanation', ''),
            'top_factors': shap_result.get('top_factors', []),
        }

    def _extract_features(self, school_signals: Dict[str, Dict]) -> Dict[str, float]:
        features = {}
        for name in sorted(school_signals.keys()):
            sig = school_signals[name]
            direction = sig.get('direction', 'neutral')
            score = sig.get('score', 0.0)
            conf = sig.get('confidence', 0.0)
            if direction == 'bullish':
                dir_score = max(0.0, min(1.0, score))
            elif direction == 'bearish':
                dir_score = -max(0.0, min(1.0, score))
            else:
                dir_score = 0.0
            features[f'{name}_dir'] = round(dir_score, 4)
            features[f'{name}_conf'] = round(float(conf), 4)
        return features

    @property
    def is_trained(self) -> bool:
        return self._trained


# =============================================================================
# 5. 集成到 expert_ensemble
# =============================================================================

# Global singleton — trained once, used for all predictions
_global_ranker: Optional[InstitutionRanker] = None


def get_ranker() -> InstitutionRanker:
    """Lazy-load global InstitutionRanker, auto-training from Parquet."""
    global _global_ranker
    if _global_ranker is None:
        _global_ranker = InstitutionRanker()
        import os
        parquet_path = os.path.join(os.path.dirname(__file__) if '__file__' in dir()
                                    else 'G:\\.book_extracts',
                                    'data', 'ml_features.parquet')
        if os.path.exists(parquet_path):
            _global_ranker.train(parquet_path)
    return _global_ranker


def explain_prediction(school_signals: Dict[str, Dict]) -> Dict:
    """Convenience: rank_score + SHAP explanation for current signal."""
    ranker = get_ranker()
    return ranker.predict_with_shap(school_signals)


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    # Simulate training data
    n_dates = 30
    n_stocks = 50
    n_features = 28

    df = pd.DataFrame({
        'date': np.repeat(pd.date_range('2026-01-01', periods=n_dates, freq='5B'), n_stocks),
        'excess_ret': np.random.randn(n_dates * n_stocks) * 0.05,
    })
    for i in range(n_features):
        df[f'feat_{i:02d}_dir'] = np.random.randn(n_dates * n_stocks) * 0.1
        df[f'feat_{i:02d}_conf'] = np.random.rand(n_dates * n_stocks)

    feature_cols = [c for c in df.columns if c.endswith('_dir') or c.endswith('_conf')]

    # Test ranking labels
    labels = build_ranking_labels(df, n_bins=5)
    print(f"=== Ranking Labels ===")
    print(f"Distribution: {labels.value_counts().sort_index().to_dict()}")

    # Test ranker training
    ranker, explainer, cols = train_xgb_ranker(df, feature_cols)
    print(f"Ranker trained: {ranker is not None}")
    print(f"SHAP explainer: {explainer is not None}")

    # Test half-life decay
    hl = HalfLifeFeatureProcessor()
    for scale in ['intraday', 'daily', 'weekly']:
        val = hl.process(f'school_{scale}_dir', 0.8)
        print(f"HalfLife({scale}): {val:.3f}")

    print("\nAll ranker ensemble modules: OK")
