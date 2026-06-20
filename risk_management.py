#!/usr/bin/env python
# risk_management.py — CRO-grade risk analytics
"""
Three pillars of institutional risk management:

  1. Block-Bootstrap MC: 10,000 parallel universes → Ruin Probability
  2. Capacity Calculator: ADV-based max AUM with spread constraints
  3. Tail-Risk Hedging: Dynamic Beta + index futures hedge ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats


# =============================================================================
# 1. Block-Bootstrap Monte Carlo Engine
# =============================================================================

@dataclass
class BlockBootstrapMC:
    """
    Block-bootstrap simulation for strategy stress testing.

    Why blocks (not individual daily returns)?
      - Preserves serial correlation (momentum, mean-reversion)
      - Maintains volatility clustering (GARCH effects)
      - Respects regime persistence (bull/bear markets last weeks)

    Block size: 5 days (one trading week) — standard in academic literature
    (Politis & Romano, 1994; Lahiri, 2003).

    Returns:
      - Ruin probability: P(drawdown > ruin_threshold)
      - Expected shortfall at 95% and 99%
      - Distribution of terminal returns
    """

    block_size: int = 5
    n_paths: int = 10000
    ruin_threshold: float = 0.30         # 30% drawdown = ruined
    confidence_levels: Tuple = (0.95, 0.99)

    def bootstrap_returns(self, daily_returns: np.ndarray,
                         path_length: int = 252) -> np.ndarray:
        """
        Generate N bootstrap paths of returns using block resampling.

        Args:
            daily_returns: (T,) array of historical daily returns
            path_length: desired length of each bootstrap path (e.g. 252 = 1 year)

        Returns:
            (n_paths, path_length) bootstrap return matrix
        """
        T = len(daily_returns)
        n_blocks = T // self.block_size

        # Pre-compute blocks
        blocks = []
        for i in range(n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            blocks.append(daily_returns[start:end])

        # Handle remaining days
        if T % self.block_size > 0:
            blocks.append(daily_returns[n_blocks * self.block_size:])

        blocks = np.array([b for b in blocks if len(b) == self.block_size])
        if len(blocks) == 0:
            raise ValueError(f"Not enough data for block_size={self.block_size}")

        # Generate bootstrap paths
        paths_needed = int(np.ceil(path_length / self.block_size))
        bootstrap_paths = np.zeros((self.n_paths, path_length))

        for p in range(self.n_paths):
            sampled_indices = np.random.choice(len(blocks), size=paths_needed, replace=True)
            path_returns = np.concatenate([blocks[i] for i in sampled_indices])
            bootstrap_paths[p] = path_returns[:path_length]

        return bootstrap_paths

    def simulate(self, daily_returns: np.ndarray, initial_capital: float = 1.0,
                 path_length: int = 252) -> Dict:
        """
        Full Monte Carlo simulation.

        Returns comprehensive risk analytics.
        """
        bootstrap = self.bootstrap_returns(daily_returns, path_length)

        # Cumulative returns for all paths
        cumulative = np.cumprod(1 + bootstrap, axis=1) * initial_capital

        # Peak-to-trough drawdown per path
        peak = np.maximum.accumulate(cumulative, axis=1)
        drawdown = (peak - cumulative) / peak

        # Maximum drawdown per path
        max_dd = drawdown.max(axis=1)

        # Terminal values
        terminal_values = cumulative[:, -1]

        # Ruin probability
        ruin_prob = np.mean(max_dd > self.ruin_threshold)

        # Expected shortfall (CVaR)
        losses = 1 - terminal_values / initial_capital

        es_95 = np.mean(losses[losses > np.percentile(losses, 95)])
        es_99 = np.mean(losses[losses > np.percentile(losses, 99)])

        # Value at Risk
        var_95 = np.percentile(losses, 95)
        var_99 = np.percentile(losses, 99)

        # Return distribution statistics
        total_returns = terminal_values / initial_capital - 1

        # Calmar ratio (annual return / max drawdown)
        ann_return = np.median(total_returns)
        ann_maxdd = np.median(max_dd)
        calmar = ann_return / max(ann_maxdd, 0.001)

        return {
            'ruin_probability': round(float(ruin_prob), 4),
            'var_95_pct': round(float(var_95 * 100), 2),
            'var_99_pct': round(float(var_99 * 100), 2),
            'expected_shortfall_95': round(float(es_95 * 100), 2),
            'expected_shortfall_99': round(float(es_99 * 100), 2),
            'median_return_pct': round(float(np.median(total_returns) * 100), 2),
            'median_maxdd_pct': round(float(np.median(max_dd) * 100), 2),
            'calmar_ratio': round(float(calmar), 2),
            'best_case_pct': round(float(np.percentile(total_returns, 95) * 100), 2),
            'worst_case_pct': round(float(np.percentile(total_returns, 5) * 100), 2),
            'n_paths': self.n_paths,
            'n_days_per_path': path_length,
            'block_size': self.block_size,
            'risk_assessment': 'HIGH' if ruin_prob > 0.05 else 'MODERATE' if ruin_prob > 0.01 else 'LOW',
        }

    def stress_test(self, daily_returns: np.ndarray,
                    scenarios: Dict[str, float] = None) -> Dict:
        """
        Conditional stress tests under specific shock scenarios.

        Args:
            scenarios: {'2008_crash': -0.60, '2015_rout': -0.45, 'covid': -0.35}
        """
        if scenarios is None:
            scenarios = {
                'mild_bear': -0.20,
                'severe_bear': -0.40,
                'flash_crash': -0.60,
            }

        results = {}
        base_result = self.simulate(daily_returns)

        for name, shock in scenarios.items():
            # Add shock to first week of bootstrap
            shocked_returns = daily_returns.copy()
            shocked_returns[:5] += shock / 5  # Spread over 5 days
            scenario_result = self.simulate(shocked_returns)
            results[name] = {
                'shock_pct': round(shock * 100, 0),
                'ruin_prob': scenario_result['ruin_probability'],
                'var_99': scenario_result['var_99_pct'],
                'expected_shortfall_99': scenario_result['expected_shortfall_99'],
            }

        base_result['stress_scenarios'] = results
        return base_result


# =============================================================================
# 2. Capacity Calculator (Max AUM Estimation)
# =============================================================================

@dataclass
class CapacityCalculator:
    """
    Estimate maximum AUM given liquidity constraints.

    Formula (inverted from Almgren-Chriss):
      Max_AUM = Σ_i (ADV_i × Max_Participation_i × Price_i) / Turnover_Rate

    Where Max_Participation_i is the largest fraction of ADV that keeps
    impact below the target threshold.

    Additional constraint: Position_i ≤ ADV_i × Max_Holding_Days
      (Must be able to exit position within N days without exceeding impact limit)
    """

    max_impact_bps: float = 25.0       # Max market impact (25bp = 0.25%)
    max_participation: float = 0.02    # Max % of ADV per day
    max_holding_days: int = 10         # Days to fully exit
    spread_cost_bps: float = 5.0       # Bid-ask spread estimate (5bp for liquid)
    annual_turnover: float = 12.0      # Portfolio turnover per year (12x = monthly)

    def max_position_value(self, adv_shares: int, price: float,
                          spread_bps: float = None) -> float:
        """Max position value for a single stock before exceeding impact limit."""
        spread = spread_bps if spread_bps is not None else self.spread_cost_bps

        # From Almgren-Chriss: impact = σ × η × sqrt(Q/ADV)
        # Solve for Q: Q = (impact / (σ × η))² × ADV
        vol_assumed = 0.025  # Assume 2.5% daily vol as baseline
        eta = 0.30  # A-share temp impact coefficient

        max_frac = (self.max_impact_bps / 10000 / (vol_assumed * eta)) ** 2
        max_frac = min(max_frac, self.max_participation)  # Cap at participation limit

        max_qty = int(max_frac * adv_shares)
        max_value = max_qty * price

        # Spread cost adjustment
        spread_adjustment = 1 / (1 + spread / 10000)

        return max_value * spread_adjustment

    def estimate_max_aum(self, positions: List[Dict]) -> Dict:
        """
        Estimate maximum AUM for a given set of candidate positions.

        Args:
            positions: [{'symbol': str, 'price': float, 'adv_shares': int}, ...]

        Returns:
            Capacity analysis dict
        """
        total_capacity = 0.0
        per_stock = []

        for pos in positions:
            max_val = self.max_position_value(
                pos['adv_shares'], pos['price'])
            total_capacity += max_val
            per_stock.append({
                'symbol': pos['symbol'],
                'max_value': round(max_val, 0),
                'max_shares': int(max_val / pos['price'] / 100) * 100,
                'adv_participation_pct': round(
                    (max_val / pos['price']) / pos['adv_shares'] * 100, 2),
            })

        # Adjust for turnover — only a fraction of AUM turns over daily
        max_aum = total_capacity / (self.annual_turnover / 252)

        # Limit per-position concentration
        n_positions = max(len(positions), 1)
        max_aum_per_position = max_aum * 0.10  # Max 10% per position
        max_aum = min(max_aum, n_positions * max_aum_per_position)

        return {
            'max_aum_rmb': round(max_aum, 0),
            'max_aum_wan': round(max_aum / 10000, 0),
            'total_capacity_rmb': round(total_capacity, 0),
            'max_positions': len(positions),
            'avg_per_position_rmb': round(max_aum / n_positions, 0),
            'per_stock': per_stock[:10],  # Top 10
            'constraints': {
                'max_impact_bps': self.max_impact_bps,
                'max_participation': self.max_participation,
                'annual_turnover': self.annual_turnover,
                'exit_days': self.max_holding_days,
            },
        }


# =============================================================================
# 3. Macro Tail-Risk Hedging Trigger
# =============================================================================

@dataclass
class TailRiskHedge:
    """
    Dynamic beta monitor with index futures hedge calculator.

    Monitors:
      1. Portfolio beta to CSI 300 (rolling 60-day)
      2. Market volatility regime (historical vol percentile)
      3. Tail event detection (extreme daily moves)

    When triggered, calculates:
      - Number of IM (中证1000) or IC (中证500) futures contracts to short
      - Hedge ratio to neutralize beta exposure
    """

    # IM futures contract specs (中证1000股指期货)
    IM_MULTIPLIER: float = 200.0       # 200 RMB per index point
    IM_MARGIN_RATE: float = 0.12       # 12% margin

    # IC futures contract specs (中证500股指期货)
    IC_MULTIPLIER: float = 200.0
    IC_MARGIN_RATE: float = 0.12

    # Trigger thresholds
    beta_hedge_threshold: float = 0.70     # Hedge when portfolio beta > this
    vol_percentile_threshold: float = 80   # Hedge when vol > 80th percentile
    tail_day_threshold: float = -0.02      # -2% daily drop triggers emergency

    def __post_init__(self):
        self._vol_history: List[float] = []
        self._beta_history: List[float] = []

    def compute_portfolio_beta(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        window: int = 60,
    ) -> Tuple[float, float]:
        """
        Compute rolling CAPM beta of portfolio vs benchmark.

        Returns:
            (beta, r_squared): regression beta and fit quality
        """
        if len(portfolio_returns) < window or len(benchmark_returns) < window:
            return 1.0, 0.0

        p = portfolio_returns[-window:]
        b = benchmark_returns[-window:]

        # OLS: p = α + β × b + ε
        cov = np.cov(p, b)[0, 1]
        var_b = np.var(b, ddof=1)

        if var_b < 1e-10:
            return 1.0, 0.0

        beta = cov / var_b
        # R-squared
        residuals = p - beta * b
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((p - np.mean(p)) ** 2)
        r_sq = 1 - ss_res / max(ss_tot, 1e-10)

        return float(beta), float(r_sq)

    def volatility_regime(self, benchmark_returns: np.ndarray,
                         window: int = 60) -> Tuple[float, float]:
        """
        Assess current volatility vs historical distribution.

        Returns:
            (current_vol, vol_percentile): e.g. (0.025, 92) = 92nd percentile
        """
        if len(benchmark_returns) < window:
            return 0.02, 50.0

        current_vol = np.std(benchmark_returns[-20:], ddof=1) * np.sqrt(252)

        # Rolling vol history
        vols = []
        for i in range(window, len(benchmark_returns)):
            vols.append(np.std(benchmark_returns[i-window:i], ddof=1) * np.sqrt(252))

        if not vols:
            return float(current_vol), 50.0

        percentile = stats.percentileofscore(vols, current_vol)

        return float(current_vol), float(percentile)

    def tail_event_detected(self, benchmark_return: float) -> bool:
        """Detect extreme single-day moves."""
        return benchmark_return < self.tail_day_threshold

    def hedge_analysis(
        self,
        portfolio_value: float,
        portfolio_beta: float,
        index_level: float,
        contract_multiplier: float = 200.0,
    ) -> Dict:
        """
        Calculate number of futures contracts to short.

        Hedge_Contracts = (Portfolio_Value × Beta) / (Index_Level × Multiplier)

        Example:
          Portfolio = 10M RMB, Beta = 0.85, CSI 500 = 6000
          Contracts = 10,000,000 × 0.85 / (6000 × 200) = 7.08 → Short 7 contracts
        """
        if index_level <= 0:
            return {'contracts': 0, 'hedge_ratio': 0, 'reason': 'invalid_index'}

        notional_per_contract = index_level * contract_multiplier
        hedge_notional = portfolio_value * portfolio_beta
        contracts = int(np.ceil(hedge_notional / notional_per_contract))

        margin_required = contracts * notional_per_contract * self.IM_MARGIN_RATE

        return {
            'contracts_to_short': contracts,
            'hedge_notional_rmb': round(hedge_notional, 0),
            'notional_per_contract': round(notional_per_contract, 0),
            'margin_required_rmb': round(margin_required, 0),
            'hedge_ratio_pct': round(contracts * notional_per_contract / portfolio_value * 100, 1),
            'index_level': index_level,
            'portfolio_beta': round(portfolio_beta, 3),
        }

    def monitor_and_hedge(
        self,
        portfolio_value: float,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        index_level: float,
    ) -> Dict:
        """
        Full tail-risk monitoring cycle.

        Returns:
            Dict with hedge recommendation or 'no_action' status.
        """
        beta, r_sq = self.compute_portfolio_beta(portfolio_returns, benchmark_returns)
        vol, vol_pct = self.volatility_regime(benchmark_returns)
        latest_return = benchmark_returns[-1] if len(benchmark_returns) > 0 else 0
        tail_event = self.tail_event_detected(latest_return)

        # Decision logic
        triggers = []

        if beta > self.beta_hedge_threshold:
            triggers.append(f'High Beta ({beta:.2f})')
        if vol_pct > self.vol_percentile_threshold:
            triggers.append(f'High Vol ({vol_pct:.0f}%ile)')
        if tail_event:
            triggers.append(f'Tail Event ({latest_return:.1%})')

        should_hedge = len(triggers) >= 2 or tail_event  # 2+ triggers or emergency

        hedge_plan = None
        if should_hedge:
            hedge_plan = self.hedge_analysis(portfolio_value, beta, index_level)

        return {
            'timestamp': pd.Timestamp.now().isoformat(),
            'should_hedge': should_hedge,
            'triggers': triggers,
            'beta': round(beta, 3),
            'beta_r_squared': round(r_sq, 3),
            'current_vol_annual': round(vol, 3),
            'vol_percentile': round(vol_pct, 1),
            'tail_event': tail_event,
            'hedge_plan': hedge_plan,
            'action': f"Short {hedge_plan['contracts_to_short']} contracts" if hedge_plan else 'No action',
        }


# =============================================================================
# Benchmark
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    # ---- 1. Block Bootstrap ----
    print("=== Block-Bootstrap Monte Carlo ===")
    # Synthetic returns with negative skew
    T = 500
    returns = np.random.randn(T) * 0.02 - 0.0005  # Mean -5bp, vol 2%
    # Add a crash
    returns[100:105] = -0.03

    mc = BlockBootstrapMC(n_paths=5000, ruin_threshold=0.30)
    result = mc.simulate(returns, path_length=252)
    print(f"  Ruin Prob: {result['ruin_probability']:.2%}")
    print(f"  VaR 95%: {result['var_95_pct']}%")
    print(f"  ES 95%: {result['expected_shortfall_95']}%")
    print(f"  Calmar: {result['calmar_ratio']}")
    print(f"  Risk: {result['risk_assessment']}")

    # Stress test
    stress = mc.stress_test(returns)
    for scenario, r in stress['stress_scenarios'].items():
        print(f"  {scenario}: ruin_prob={r['ruin_prob']:.2%}")

    # ---- 2. Capacity Calculator ----
    print("\n=== Capacity Calculator ===")
    cc = CapacityCalculator(max_impact_bps=25)
    positions = [
        {'symbol': '600519', 'price': 1800, 'adv_shares': 5_000_000},
        {'symbol': '000001', 'price': 12, 'adv_shares': 50_000_000},
        {'symbol': '300750', 'price': 250, 'adv_shares': 8_000_000},
    ]
    cap = cc.estimate_max_aum(positions)
    print(f"  Max AUM: {cap['max_aum_wan']}万 RMB")
    print(f"  Total capacity: {cap['total_capacity_rmb']:,.0f} RMB")

    # ---- 3. Tail Risk Hedge ----
    print("\n=== Tail Risk Hedge ===")
    hedge = TailRiskHedge()
    p_returns = np.random.randn(200) * 0.015 - 0.0003
    b_returns = np.random.randn(200) * 0.018 - 0.0002
    # Simulate high-beta, high-vol day
    b_returns[-1] = -0.025  # Tail day
    b_returns[-60:] *= 1.5  # Higher vol recently

    result = hedge.monitor_and_hedge(10_000_000, p_returns, b_returns, 6000)
    print(f"  Should hedge: {result['should_hedge']}")
    print(f"  Beta: {result['beta']:.3f}")
    print(f"  Vol %ile: {result['vol_percentile']:.0f}%")
    print(f"  Action: {result['action']}")
    if result['hedge_plan']:
        print(f"  Contracts: {result['hedge_plan']['contracts_to_short']}")
        print(f"  Margin: {result['hedge_plan']['margin_required_rmb']:,.0f} RMB")

    print("\nAll risk management modules: OK")
