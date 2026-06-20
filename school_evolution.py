# school_evolution.py
"""
School weight auto-evolution system with statistical significance gating.

CRITICAL CHANGE: Weights are NO LONGER updated on single-trade P&L attribution.
Instead, a binomial test (95% confidence) is required:
  - For each school, collect last N=30 trade outcomes (correct/incorrect)
  - H0: win_rate >= expected_win_rate (school is competent)
  - Only if H0 is REJECTED (p < 0.05) does weight decay trigger
  - This prevents neurotic overfitting to isolated stop-losses.

Weight blending formula:
    w_i = (1-α-β) × regime_weight_i + α × learned_weight_i + β × trust_score_i

where α grows with data quantity: α = min(0.60, n_trades / 200)
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
from scipy.stats import binomtest

from trade_logger import (
    save_school_performance,
    save_regime_weights_learned,
    load_regime_weights_learned,
    list_backtest_runs,
    load_backtest_run,
)

# ---------------------------------------------------------------------------
# Binomial significance test for weight decay gating
# ---------------------------------------------------------------------------

def binomial_significance_test(
    n_correct: int,
    n_total: int,
    expected_win_rate: float = 0.50,
    confidence_level: float = 0.95,
) -> Tuple[bool, float]:
    """
    Two-sided binomial test for school win rate significance.

    H0: true win rate = expected_win_rate
    H1: true win rate ≠ expected_win_rate

    Only rejects H0 if p < (1 - confidence_level), i.e., p < 0.05 at 95%.

    Returns:
        (is_significantly_worse, p_value)
        is_significantly_worse=True means the school's observed win rate is
        statistically below expected with 95% confidence → weight decay justified.
    """
    if n_total < 5:
        return False, 1.0  # Insufficient data — cannot reject H0

    result = binomtest(n_correct, n=n_total, p=expected_win_rate, alternative='two-sided')
    p_value = result.pvalue
    alpha = 1.0 - confidence_level
    observed_rate = n_correct / max(n_total, 1)

    # Only flag if worse AND statistically significant
    is_worse = observed_rate < expected_win_rate and p_value < alpha

    return is_worse, float(p_value)

# ---------------------------------------------------------------------------
# School definitions (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------
SCHOOL_NAMES = [
    # 13学派 — 流水线测试配置 (100只 + 13学派, Sharpe -0.743)
    # 9 核心学派
    'school_risk',          # 风险与市场环境
    'school_classical',     # 经典技术分析
    'school_livermore',     # 利弗莫尔战法
    'school_chanlun',       # 缠论学派
    'school_gann',          # 江恩理论
    'school_tang',          # 唐能通体系
    'school_wyckoff',       # 威科夫量价
    'school_busch',         # Busch量化体系
    # 4 策略学派
    'school_roc_breakout',  # ROC动量突破策略
    'school_volume_profile',# 成交量分布策略
    'school_harmonic',      # 谐波形态
    'school_fusion',        # 蒸馏融合策略(Fusion)
    'school_mean_reversion',# 均值回归学派 (BB+RSI+BIAS+KC 多因子)
    'school_capital_flow', # 资金流向学派
    'school_pattern_features', # 经典形态特征因子库 (42维连续因子)
    'school_ml',             # ML模型(XGBoost) — Champion/Challenger shadow
    'school_nd',             # 正态分布学派 — Z-Score均值回归+动量
    'school_vp',             # 成交量分布学派 — Volume Profile POC
    'school_roc_factor',     # ROC动量因子 — AX/BX变体
    'school_beta',           # 历史Beta学派 — 高Beta牛熊判定
    'school_ewm_beta',             # EWM Beta学派 — 指数衰减加权Beta
    'school_brooks_pa',         # Brooks价格行为学派 — 趋势K线/H计数/陷阱/EMA引力
]

REGIMES = ['trending', 'ranging', 'volatile', 'transitional']

# Quality-weighted default regime weights — 13学派配置
_DEFAULT_REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    'trending': {
        'school_risk': 0.5, 'school_classical': 0.8, 'school_livermore': 0.7,
        'school_chanlun': 1.0, 'school_gann': 0.8,
        'school_tang': 0.4, 'school_wyckoff': 0.8, 'school_busch': 0.5,
        'school_roc_breakout': 0.9, 'school_volume_profile': 0.6,
        'school_harmonic': 0.4, 'school_fusion': 0.5,
        'school_mean_reversion': 0.3,
        'school_capital_flow': 0.7,
        'school_pattern_features': 0.7,
    },
    'ranging': {
        'school_risk': 0.6, 'school_classical': 0.5, 'school_livermore': 0.4,
        'school_chanlun': 1.0, 'school_gann': 0.6,
        'school_tang': 1.0, 'school_wyckoff': 1.0, 'school_busch': 0.7,
        'school_roc_breakout': 0.4, 'school_volume_profile': 1.0,
        'school_harmonic': 0.7, 'school_fusion': 0.6,
        'school_mean_reversion': 1.0,
        'school_capital_flow': 0.6,
        'school_pattern_features': 0.8,
    },
    'volatile': {
        'school_risk': 1.0, 'school_classical': 0.3, 'school_livermore': 0.4,
        'school_chanlun': 1.0, 'school_gann': 0.5,
        'school_tang': 0.4, 'school_wyckoff': 0.4, 'school_busch': 0.5,
        'school_roc_breakout': 0.8, 'school_volume_profile': 0.5,
        'school_harmonic': 0.3, 'school_fusion': 0.4,
        'school_mean_reversion': 0.5,
        'school_capital_flow': 0.7,
        'school_pattern_features': 0.4,
    },
    'transitional': {
        'school_risk': 0.6, 'school_classical': 0.6, 'school_livermore': 0.5,
        'school_chanlun': 1.0, 'school_gann': 0.7,
        'school_tang': 0.6, 'school_wyckoff': 0.8, 'school_busch': 0.6,
        'school_roc_breakout': 0.7, 'school_volume_profile': 0.8,
        'school_harmonic': 0.6, 'school_fusion': 0.6,
        'school_mean_reversion': 0.7,
        'school_capital_flow': 0.7,
        'school_pattern_features': 0.7,
    },
}


class SchoolWeightLearner:
    """
    Tracks per-school performance across backtest runs and learns optimal
    ensemble weights that adapt to market regimes.

    Weight decay is GATED by binomial significance test:
      - Per-school sliding window of last 30 trade outcomes
      - Only if observed win rate is statistically below expected (p<0.05)
        does the weight actually decay
      - Prevents neurotic overfitting to isolated stop-losses
    """

    WINDOW_SIZE: int = 30          # Sliding window for binomial test
    EXPECTED_WIN_RATE: float = 0.50
    CONFIDENCE_LEVEL: float = 0.95

    def __init__(self) -> None:
        self.metrics: Dict[str, Dict[str, Dict[str, float]]] = {
            regime: {school: self._empty_metrics() for school in SCHOOL_NAMES}
            for regime in REGIMES
        }
        self.global_metrics: Dict[str, Dict[str, float]] = {
            school: self._empty_metrics() for school in SCHOOL_NAMES
        }
        self._total_trades_attributed: int = 0
        # Per-school sliding outcome windows for binomial test
        self._outcome_windows: Dict[str, deque] = {
            school: deque(maxlen=self.WINDOW_SIZE) for school in SCHOOL_NAMES
        }
        # Per-school binomial test results (updated after each trade batch)
        self._binom_results: Dict[str, Dict] = {
            school: {'is_significantly_worse': False, 'p_value': 1.0,
                     'observed_rate': 0.5, 'n_trials': 0}
            for school in SCHOOL_NAMES
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            'n_signals': 0, 'n_correct': 0, 'n_incorrect': 0,
            'total_pnl_bps': 0.0, 'total_pnl_sq_bps': 0.0,
            'total_win_bps': 0.0, 'total_loss_bps': 0.0,
            'total_confidence': 0.0, 'win_rate': 0.0,
            'avg_pnl_bps': 0.0, 'sharpe_like': 0.0, 'quality_score': 0.0,
            'decay_gated': 0.0,  # 1.0 if weight decay is active (binom significant)
        }

    # ========================================================================
    # Attribution (unchanged core logic)
    # ========================================================================

    def compute_school_attribution(
        self, trade: Dict[str, Any], regime: str = 'transitional'
    ) -> Dict[str, float]:
        net_pnl_bps = trade.get('net_pnl_pct', 0) * 100
        school_votes = trade.get('school_votes_json', {})
        if isinstance(school_votes, str):
            try:
                school_votes = json.loads(school_votes)
            except (json.JSONDecodeError, TypeError):
                school_votes = {}
        if not school_votes:
            return {}

        active_schools: Dict[str, Tuple[str, float]] = {}
        total_conf = 0.0
        for name in SCHOOL_NAMES:
            sv = school_votes.get(name, {})
            if not isinstance(sv, dict):
                continue
            direction = str(sv.get('direction', 'neutral'))
            confidence = float(sv.get('confidence', 0))
            if confidence > 0.01:
                active_schools[name] = (direction, confidence)
                total_conf += confidence

        if total_conf <= 0:
            return {}

        attribution: Dict[str, float] = {}
        for name, (direction, conf) in active_schools.items():
            weight = conf / total_conf
            if direction == 'bullish':
                attribution[name] = net_pnl_bps * weight
            elif direction == 'bearish':
                attribution[name] = -net_pnl_bps * weight
            else:
                attribution[name] = -abs(net_pnl_bps) * weight * 0.5

        return attribution

    # ========================================================================
    # Attribution
    # ========================================================================

    def compute_school_attribution(
        self,
        trade: Dict[str, Any],
        regime: str = 'transitional',
    ) -> Dict[str, float]:
        """
        Attribute a single trade's P&L to individual schools.

        For a long trade:
          - Schools that voted bullish: credited if P&L > 0, debited if P&L < 0
          - Schools that voted bearish: debited if P&L > 0, credited if P&L < 0
          - Neutral schools: small debit (indecision cost)

        Attribution is proportional to each school's confidence at entry time.

        Returns dict of school_name -> attributed_pnl_bps (basis points).
        """
        net_pnl_bps = trade.get('net_pnl_pct', 0) * 100  # convert % to bps
        school_votes = trade.get('school_votes_json', {})

        if isinstance(school_votes, str):
            try:
                school_votes = json.loads(school_votes)
            except (json.JSONDecodeError, TypeError):
                school_votes = {}

        if not school_votes:
            return {}

        # Collect active schools and their directions
        active_schools: Dict[str, Tuple[str, float]] = {}
        total_conf = 0.0

        for name in SCHOOL_NAMES:
            sv = school_votes.get(name, {})
            if not isinstance(sv, dict):
                continue
            direction = str(sv.get('direction', 'neutral'))
            confidence = float(sv.get('confidence', 0))
            if confidence > 0.01:
                active_schools[name] = (direction, confidence)
                total_conf += confidence

        if total_conf <= 0:
            return {}

        # Attribute P&L proportional to confidence-weighted correctness
        attribution: Dict[str, float] = {}
        for name, (direction, conf) in active_schools.items():
            weight = conf / total_conf
            if direction == 'bullish':
                attribution[name] = net_pnl_bps * weight
            elif direction == 'bearish':
                attribution[name] = -net_pnl_bps * weight
            else:
                attribution[name] = -abs(net_pnl_bps) * weight * 0.5

        return attribution

    # ========================================================================
    # Update from backtest
    # ========================================================================

    def _push_outcomes_and_test(self) -> None:
        """
        Push latest trade outcomes into per-school sliding windows,
        then run binomial significance test for each school.
        Only schools with statistically significant underperformance get decay.
        """
        for name in SCHOOL_NAMES:
            m = self.global_metrics[name]
            n = int(m['n_signals'])
            if n < 5:
                continue
            # Reconstruct recent outcomes from cumulative counts
            # (approximate: use the trailing window of size WINDOW_SIZE)
            total_correct = int(m['n_correct'])
            total_incorrect = int(m['n_incorrect'])
            recent_total = min(self.WINDOW_SIZE, total_correct + total_incorrect)
            if recent_total < 10:
                continue

            # Use the most recent WINDOW_SIZE outcomes (approximate from cumulative)
            # For exact tracking, we use the deque
            window = self._outcome_windows[name]
            if len(window) >= 10:
                n_correct = sum(window)
                n_total = len(window)
                is_worse, p_val = binomial_significance_test(
                    n_correct, n_total, self.EXPECTED_WIN_RATE, self.CONFIDENCE_LEVEL)
                self._binom_results[name] = {
                    'is_significantly_worse': is_worse,
                    'p_value': round(p_val, 4),
                    'observed_rate': round(n_correct / max(n_total, 1), 3),
                    'n_trials': n_total,
                }
                self.global_metrics[name]['decay_gated'] = 1.0 if is_worse else 0.0

    def update_from_backtest(
        self,
        trades: List[Dict[str, Any]],
        regime: Optional[str] = None,
        run_id: Optional[int] = None,
    ) -> None:
        """
        Update cumulative school performance from backtest trades.

        CRITICAL: Single-trade attribution still accumulates, but weight DECAY
        is now gated by binomial significance test (updated in batch after
        all trades are processed via _push_outcomes_and_test).
        """
        regime = regime or 'transitional'

        for trade in trades:
            attribution = self.compute_school_attribution(trade, regime)
            net_pnl_bps = trade.get('net_pnl_pct', 0) * 100
            is_winner = net_pnl_bps > 0

            for name in SCHOOL_NAMES:
                attr_bps = attribution.get(name, 0)
                m = self.metrics[regime][name]
                g = self.global_metrics[name]
                m['n_signals'] += 1; g['n_signals'] += 1

                # Track correctness for sliding window
                school_correct = (is_winner and attr_bps > 0) or (not is_winner and attr_bps > 0)
                if school_correct:
                    m['n_correct'] += 1; g['n_correct'] += 1
                else:
                    m['n_incorrect'] += 1; g['n_incorrect'] += 1

                # Push outcome to sliding window
                self._outcome_windows[name].append(1 if school_correct else 0)

                m['total_pnl_bps'] += attr_bps; g['total_pnl_bps'] += attr_bps
                m['total_pnl_sq_bps'] += attr_bps ** 2; g['total_pnl_sq_bps'] += attr_bps ** 2
                if attr_bps > 0:
                    m['total_win_bps'] += attr_bps; g['total_win_bps'] += attr_bps
                else:
                    m['total_loss_bps'] += attr_bps; g['total_loss_bps'] += attr_bps

            self._total_trades_attributed += 1

        # Run binomial tests after batch
        self._push_outcomes_and_test()

        # Recompute derived metrics
        for rn in REGIMES:
            for name in SCHOOL_NAMES:
                self._recompute_derived(self.metrics[rn][name])
        for name in SCHOOL_NAMES:
            self._recompute_derived(self.global_metrics[name])

        if run_id is not None:
            self._persist_to_db(run_id, regime)

    def update_from_backtest_engine(self, engine, run_id: Optional[int] = None) -> None:
        """
        Update from a BacktestEngine instance, auto-detecting per-trade regime.

        Regime detection uses the same logic as expert_ensemble._detect_market_regime
        but applied at trade-entry time using stored school signal data.
        """
        trades = getattr(engine, 'trades', [])
        if not trades:
            print("[SchoolEvolution] No trades in engine — nothing to learn from.")
            return

        # Group trades by approximate regime using available data
        # Since we don't store full indicators at entry time, we use
        # a simple heuristic: ensemble_confidence as a proxy
        # (high confidence → trending, medium → transitional, low → ranging/volatile)
        regime_buckets: Dict[str, List[Dict]] = {r: [] for r in REGIMES}

        for trade in trades:
            conf = trade.get('ensemble_confidence', 0.5)
            # Heuristic regime assignment from confidence level
            if conf > 0.65:
                r = 'trending'
            elif conf > 0.45:
                r = 'transitional'
            elif conf > 0.30:
                r = 'ranging'
            else:
                r = 'volatile'
            regime_buckets[r].append(trade)

        for regime, bucket_trades in regime_buckets.items():
            if bucket_trades:
                self.update_from_backtest(bucket_trades, regime=regime)

        if run_id is not None:
            self._persist_to_db(run_id)

    @staticmethod
    def _recompute_derived(m: Dict[str, float]) -> None:
        n = max(int(m['n_signals']), 1)
        n_correct = int(m['n_correct'])
        n_incorrect = int(m.get('n_incorrect', n - n_correct))
        m['win_rate'] = round(n_correct / n, 4)
        m['avg_pnl_bps'] = round(m['total_pnl_bps'] / n, 2)

        # ---- Expected Value quality score (replaces pseudo-Sharpe) ----
        # EV = Win_Rate * Avg_Win - Loss_Rate * |Avg_Loss|
        # This is mathematically valid for variable-holding-period trades.
        avg_win  = m['total_win_bps'] / max(n_correct, 1)
        avg_loss = m['total_loss_bps'] / max(n_incorrect, 1)
        loss_rate = 1.0 - m['win_rate']
        ev_per_trade = m['win_rate'] * avg_win - loss_rate * abs(avg_loss)

        # sqrt(n) credibility: more trades = more confidence, with diminishing returns
        credibility = min(1.0, np.sqrt(n / 30.0))

        # Normalize EV to ~[0, 1]: 50 bps EV → 0.5, 100 bps → 1.0
        ev_normalized = min(1.0, max(0.0, ev_per_trade / 100.0))
        m['quality_score'] = round(ev_normalized * credibility, 4)

        # Retain sharpe_like for backward compatibility (deprecated, not used in quality)
        mean_bps = m['total_pnl_bps'] / n
        var_bps = m['total_pnl_sq_bps'] / n - mean_bps ** 2
        m['sharpe_like'] = round(mean_bps / (max(var_bps, 1e-10) ** 0.5), 3) if var_bps > 0 else 0.0

    # ========================================================================
    # Weight computation
    # ========================================================================

    def compute_learned_weights(
        self,
        regime: Optional[str] = None,
        temperature: float = 1.5,
    ) -> Dict[str, float]:
        """
        Compute learned weights with binomial significance penalty.

        Schools with statistically significant underperformance (p<0.05 binomial)
        get their quality_score multiplied by 0.30 (70% weight reduction).
        Schools with insufficient data get neutral quality.
        """
        if regime:
            metrics_source = self.metrics.get(regime, self.global_metrics)
        else:
            metrics_source = self.global_metrics

        qualities = []
        for name in SCHOOL_NAMES:
            q = max(0.001, metrics_source[name]['quality_score'])
            # Binomial gate: if significantly underperforming, penalize heavily
            binom = self._binom_results.get(name, {})
            if binom.get('is_significantly_worse', False):
                q *= 0.30  # 70% reduction for statistically-bad schools
            qualities.append(q)

        qualities = np.array(qualities, dtype=np.float64)

        # Softmax
        logits = np.log(qualities + 1e-10)
        logits = logits / max(temperature, 0.1)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        weights_arr = exp_logits / np.sum(exp_logits)

        # Normalize so max weight = 1.0
        weights_arr = weights_arr / max(np.max(weights_arr), 0.01)

        return {
            name: round(float(weights_arr[i]), 3)
            for i, name in enumerate(SCHOOL_NAMES)
        }

    def get_blended_weights(
        self,
        regime: str,
        trust_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute the final blended weights using the three-source formula:

            w_i = (1-α-β) × regime_weight_i + α × learned_weight_i + β × trust_score_i

        where:
            α = min(0.60, n_trades / 200)  — learned weight share
            β = 0.25                         — trust score share (fixed)
        """
        alpha = min(0.70, self._total_trades_attributed / 30.0)
        beta = 0.0  # Trust removed — over-complicated
        regime_share = max(0.0, 1.0 - alpha - beta)

        regime_weights = _DEFAULT_REGIME_WEIGHTS.get(regime, _DEFAULT_REGIME_WEIGHTS['transitional'])
        learned_weights = self.compute_learned_weights(regime=regime)
        trust_scores = trust_scores or {}

        blended: Dict[str, float] = {}
        for name in SCHOOL_NAMES:
            rw = regime_weights.get(name, 0.5)
            lw = learned_weights.get(name, 0.5)
            ts = trust_scores.get(name, 0.5)
            blended[name] = round(
                regime_share * rw + alpha * lw + beta * ts, 3
            )

        return blended

    # ========================================================================
    # Regime weight auto-tuning
    # ========================================================================

    def auto_tune_regime_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Generate fully data-driven regime weights to replace hardcoded table.

        For each regime, computes learned weights. If a regime has insufficient
        data (< 50 signals), falls back to a blend of learned + default.
        """
        tuned: Dict[str, Dict[str, float]] = {}

        for regime in REGIMES:
            total_signals = sum(
                int(self.metrics[regime][s]['n_signals']) for s in SCHOOL_NAMES
            )
            if total_signals >= 50:
                tuned[regime] = self.compute_learned_weights(regime=regime)
            else:
                # Blend 50% learned + 50% default when data is thin
                learned = self.compute_learned_weights(regime=regime)
                default = _DEFAULT_REGIME_WEIGHTS[regime]
                tuned[regime] = {
                    s: round(0.5 * learned.get(s, 0.5) + 0.5 * default.get(s, 0.5), 3)
                    for s in SCHOOL_NAMES
                }

        return tuned

    # ========================================================================
    # Persistence
    # ========================================================================

    def _persist_to_db(self, run_id: int, regime_override: Optional[str] = None) -> None:
        """Save current cumulative metrics to the database."""
        for regime in REGIMES:
            if regime_override and regime != regime_override:
                continue
            school_metrics: Dict[str, Dict] = {}
            for name in SCHOOL_NAMES:
                m = self.metrics[regime][name]
                n = int(m['n_signals'])
                n_c = int(m['n_correct'])
                n_i = int(m['n_incorrect'])
                total = n_c + n_i
                school_metrics[name] = {
                    'window_label': 'cumulative',
                    'n_signals': n,
                    'n_correct': n_c,
                    'win_rate': round(n_c / max(total, 1), 4),
                    'avg_pnl_pct': round(m['avg_pnl_bps'] / 100, 4),
                    'avg_confidence': round(m['total_confidence'] / max(n, 1), 3),
                    'sharpe_like_score': m['sharpe_like'],
                    'direction_bias': self._compute_direction_bias(name, regime),
                    'contribution_weight': round(
                        self.compute_learned_weights(regime=regime).get(name, 0.5), 3
                    ),
                }
            save_school_performance(run_id, school_metrics, regime=regime)

        # Save learned regime weights
        learned_all = self.auto_tune_regime_weights()
        for regime, weights in learned_all.items():
            total = sum(
                int(self.metrics[regime][s]['n_signals']) for s in SCHOOL_NAMES
            )
            save_regime_weights_learned(regime, weights, int(total))

    def _compute_direction_bias(self, name: str, regime: str) -> str:
        """Classify a school's directional bias from its metrics."""
        m = self.metrics[regime][name]
        n = int(m['n_signals'])
        if n < 5:
            return 'insufficient_data'
        win_rate = m['win_rate']
        avg_pnl = m['avg_pnl_bps']
        if win_rate > 0.55 and avg_pnl > 5:
            return 'strong_bullish'
        elif win_rate > 0.50:
            return 'slight_bullish'
        elif win_rate < 0.45 and avg_pnl < -5:
            return 'strong_bearish'
        elif win_rate < 0.50:
            return 'slight_bearish'
        return 'neutral'

    # ========================================================================
    # Load historical data
    # ========================================================================

    def load_from_db(self, run_id: Optional[int] = None) -> int:
        """
        Load cumulative school performance from previous backtest runs.

        If run_id is provided, loads only that run. Otherwise loads all runs.
        Returns the number of trades loaded.
        """
        if run_id is not None:
            run_data, trades = load_backtest_run(run_id)
            if trades:
                self.update_from_backtest(trades)
                return len(trades)
            return 0

        # Load all runs
        runs = list_backtest_runs(limit=50)
        total_trades = 0
        for run in runs:
            rid = run.get('id')
            if rid is None:
                continue
            try:
                _, trades = load_backtest_run(rid)
                if trades:
                    self.update_from_backtest(trades)
                    total_trades += len(trades)
            except Exception as e:
                print(f"  [WARN] Failed to load run {rid}: {e}")

        return total_trades

    def load_regime_weights_from_db(self) -> Optional[Dict[str, Dict[str, float]]]:
        """Load previously learned regime weights from the database."""
        return load_regime_weights_learned()

    # ========================================================================
    # Reporting
    # ========================================================================

    def report(self) -> str:
        """Generate a text report of school performance and learned weights."""
        lines = []
        lines.append("=" * 70)
        lines.append("  SCHOOL EVOLUTION REPORT")
        lines.append(f"  Trades attributed: {self._total_trades_attributed}")
        alpha = min(0.60, self._total_trades_attributed / 200.0)
        lines.append(f"  Learning alpha: {alpha:.3f} (data weight in blending)")
        lines.append("=" * 70)

        for regime in REGIMES:
            total = sum(
                int(self.metrics[regime][s]['n_signals']) for s in SCHOOL_NAMES
            )
            if total == 0:
                continue
            lines.append(f"\n--- {regime.upper()} (n={total}) ---")
            lines.append(f"{'School':<22} {'Win%':>6} {'AvgPnL':>8} {'Sharpe':>7} {'Quality':>8} {'LearnedW':>9} {'DefaultW':>9}")
            lines.append("-" * 70)

            learned = self.compute_learned_weights(regime=regime)
            default = _DEFAULT_REGIME_WEIGHTS[regime]

            sorted_schools = sorted(
                SCHOOL_NAMES,
                key=lambda s: self.metrics[regime][s]['quality_score'],
                reverse=True,
            )
            for name in sorted_schools:
                m = self.metrics[regime][name]
                n = int(m['n_signals'])
                if n == 0:
                    continue
                label = name.replace('school_', '')[:20]
                lines.append(
                    f"{label:<22} {m['win_rate']:>6.2%} {m['avg_pnl_bps']:>7.1f}bps "
                    f"{m['sharpe_like']:>7.3f} {m['quality_score']:>8.3f} "
                    f"{learned.get(name, 0):>9.3f} {default.get(name, 0):>9.3f}"
                )

        return "\n".join(lines)


# =========================================================================
# Convenience function
# =========================================================================

def run_evolution_cycle(
    engine=None,
    run_id: Optional[int] = None,
    verbose: bool = True,
) -> SchoolWeightLearner:
    """
    Run one evolution cycle: load history → process trades → persist → report.

    If engine is provided, its trades are added to the learner.
    If run_id is provided, that DB run is loaded instead.
    """
    learner = SchoolWeightLearner()

    # Load existing learned weights from DB
    existing = learner.load_regime_weights_from_db()
    if existing and verbose:
        print(f"[Evolution] Loaded existing weights for {len(existing)} regimes from DB.")

    # Load historical trade data
    if run_id is not None:
        n = learner.load_from_db(run_id=run_id)
        if verbose:
            print(f"[Evolution] Loaded {n} trades from run_id={run_id}.")
    else:
        n = learner.load_from_db()
        if verbose and n > 0:
            print(f"[Evolution] Loaded {n} trades from {len(list_backtest_runs(limit=50))} historical runs.")

    # Add engine trades if provided
    if engine is not None:
        learner.update_from_backtest_engine(engine)
        if verbose:
            print(f"[Evolution] Added {len(getattr(engine, 'trades', []))} trades from engine.")

    # Persist
    if engine is not None and hasattr(engine, 'save_to_db'):
        pass  # engine.save_to_db already called by caller

    if verbose:
        print(learner.report())

    return learner


# =========================================================================
# __main__ — quick test
# =========================================================================

if __name__ == '__main__':
    learner = run_evolution_cycle(verbose=True)

    print("\n--- Blended weights (trending regime) ---")
    blended = learner.get_blended_weights('trending')
    for name, w in sorted(blended.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {w:.3f}")

    print("\n--- Auto-tuned regime weights ---")
    tuned = learner.auto_tune_regime_weights()
    for regime, weights in tuned.items():
        top3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  {regime}: {', '.join(f'{n}({w:.2f})' for n, w in top3)}")
