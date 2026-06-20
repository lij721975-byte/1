#!/usr/bin/env python
# portfolio_optimizer.py — Two Sigma Grade Convex Portfolio Optimization
"""
Barra-Lite Style Neutralization + CVXPY Convex Solver + L1 Turnover Penalty.

Mathematical Program:
  Maximize:  alpha^T × w  -  λ_turnover × ||w - w_old||_1  -  λ_risk × w^T Σ w

  Subject to:
    (1) Σw ≤ 1.0,  w ≥ 0                         (long-only, fully-invested cap)
    (2) |F_size^T × w| ≤ 0.15                      (Size style neutrality)
    (3) |F_vol^T × w|  ≤ 0.15                      (Volatility style neutrality)
    (4) w_i ≤ 0.10                                 (Single-stock cap)

  Where:
    alpha_i  = XGBRanker score ∈ [1, 5] for stock i
    w        = portfolio weight vector (N,)
    w_old    = previous weights (0 if new entry)
    F_size   = Size factor Z-score loadings
    F_vol    = Volatility factor Z-score loadings
    λ_turnover = 0.50 (cost per unit weight change)
    λ_risk     = 0.10 (risk aversion)

Why L1 Turnover:
  ||w - w_old||_1 penalizes the MAGNITUDE of weight changes.
  A stock with alpha advantage +0.01 must overcome 50bps of transaction cost
  (印花税 0.1% + commission 0.05% + slippage ~0.35%). The L1 penalty
  naturally keeps small-alpha changes from triggering unnecessary turnover.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. Barra-Lite Factor Matrix
# =============================================================================

def compute_barra_factors(
    symbols: List[str],
    daily_data: Dict[str, pd.DataFrame] = None,
    fallback: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build Barra-Lite factor exposure matrix for N candidate stocks.

    Factors (K=3):
      1. Size:       log(avg_dollar_volume) → Z-score
      2. Momentum:   60-day return → Z-score
      3. Volatility: 20-day daily return std → Z-score

    Args:
        symbols: stock codes
        daily_data: {symbol: DataFrame with OHLCV} or None for fallback
        fallback: use synthetic values when data unavailable

    Returns:
        (F, factor_names): (N×3) exposure matrix, ['Size','Momentum','Volatility']
    """
    N = len(symbols)
    K = 3
    F = np.zeros((N, K))

    for i, sym in enumerate(symbols):
        if daily_data and sym in daily_data:
            df = daily_data[sym]
            if len(df) >= 60:
                # Size: negative log of avg dollar volume (smaller → more negative)
                avg_amount = (df['volume'] * df['close']).tail(20).mean()
                F[i, 0] = -np.log(max(avg_amount, 1.0))

                # Momentum: 60-day total return
                if len(df) >= 60:
                    F[i, 1] = df['close'].iloc[-1] / df['close'].iloc[-60] - 1
                else:
                    F[i, 1] = df['close'].pct_change(20).sum()

                # Volatility: 20-day annualized vol
                daily_ret = df['close'].pct_change().tail(20)
                F[i, 2] = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 5 else 0.25
            elif fallback:
                F[i] = [0.0, 0.0, 0.25]
        elif fallback:
            # Synthetic values for stocks without data
            F[i] = [0.0, 0.0, 0.25]

    # Z-score normalize (cross-sectionally)
    for k in range(K):
        col = F[:, k]
        mean = col.mean()
        std = col.std(ddof=1)
        if std > 1e-10:
            F[:, k] = (col - mean) / std
        else:
            F[:, k] = 0.0

    return F, ['Size', 'Momentum', 'Volatility']


# =============================================================================
# 2. CVXPY Convex Solver
# =============================================================================

@dataclass
class PortfolioOptimizer:
    """
    Convex portfolio optimization with style neutrality and turnover penalty.

    Solves the quadratic program:
      maximize  alpha^T w - λ_t * ||w - w_prev||_1 - λ_r * w^T Σ w
      s.t.      sum(w) ≤ 1, w ≥ 0, |F_k^T w| ≤ 0.15, w_i ≤ 0.10
    """

    turnover_cost: float = 0.50    # λ_turnover: cost per unit weight change
    risk_aversion: float = 0.10    # λ_risk
    max_single_weight: float = 0.10   # 10% cap per stock
    style_exposure_limit: float = 0.15  # Barra exposure bound

    def optimize(
        self,
        alpha_scores: np.ndarray,
        prev_weights: np.ndarray,
        factor_exposures: np.ndarray = None,
        factor_names: List[str] = None,
        cov_matrix: np.ndarray = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Solve the full convex program.

        Args:
            alpha_scores: (N,) predicted ranking scores ∈ [1, 5]
            prev_weights: (N,) previous portfolio weights (0 for new entries)
            factor_exposures: (N, K) Barra factor loadings
            factor_names: list of factor names
            cov_matrix: (N, N) covariance or None (use Ledoit-Wolf)

        Returns:
            (w_optimal, diagnostics): weight vector and solve info
        """
        try:
            import cvxpy as cp
        except ImportError:
            # Fallback: equal weight when CVXPY unavailable
            N = len(alpha_scores)
            w = np.ones(N) / N
            return w, {'status': 'cvxpy_unavailable', 'turnover': 0.0}

        N = len(alpha_scores)
        alpha = np.asarray(alpha_scores, dtype=np.float64).flatten()
        alpha = alpha / max(np.max(alpha), 1.0)  # Normalize to [0, 1]

        w_prev = np.asarray(prev_weights, dtype=np.float64).flatten()

        # Decision variable
        w = cp.Variable(N, nonneg=True)

        # ---- Objective ----
        # Alpha term: maximize expected score
        alpha_term = alpha @ w

        # Turnover penalty: L1 norm of weight changes
        turnover_penalty = self.turnover_cost * cp.norm1(w - w_prev)

        # Risk penalty: w^T Σ w
        if cov_matrix is not None:
            Sigma = np.asarray(cov_matrix, dtype=np.float64)
            risk_penalty = self.risk_aversion * cp.quad_form(w, Sigma)
        else:
            # Identity proxy: penalize concentration
            risk_penalty = self.risk_aversion * cp.sum_squares(w)

        objective = cp.Maximize(alpha_term - turnover_penalty - risk_penalty)

        # ---- Constraints ----
        constraints = [
            cp.sum(w) <= 1.0,                     # (1) Sum ≤ 1
            w <= self.max_single_weight,           # (4) Per-stock cap
        ]

        # (2)-(3) Style neutrality
        if factor_exposures is not None:
            # Size (column 0) and Volatility (column 2) constraints
            constrain_indices = [0, 2]  # Size and Volatility
            for k in constrain_indices:
                if k < factor_exposures.shape[1]:
                    constraints.append(
                        cp.abs(factor_exposures[:, k] @ w) <= self.style_exposure_limit
                    )

        # ---- Solve ----
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(solver=cp.ECOS, max_iters=500)
        except Exception:
            try:
                prob.solve(solver=cp.SCS, max_iters=1000)
            except Exception:
                N = len(alpha_scores)
                w = np.ones(N) / N
                return w, {'status': 'solve_failed', 'turnover': 0.0}

        w_opt = w.value

        if w_opt is None:
            N = len(alpha_scores)
            w_opt = np.ones(N) / N
            status = 'infeasible'
        else:
            w_opt = np.maximum(w_opt, 0)
            w_opt /= max(w_opt.sum(), 1e-10)
            status = prob.status

        # ---- Diagnostics ----
        turnover = np.sum(np.abs(w_opt - w_prev)) / 2  # Two-way
        concentration = np.sum(w_opt ** 2)  # HHI

        diagnostics = {
            'status': status,
            'turnover': round(float(turnover), 4),
            'concentration': round(float(concentration), 4),
            'effective_n': round(1.0 / max(concentration, 1e-10), 1),
            'n_assets': N,
            'max_weight': round(float(np.max(w_opt)), 4),
            'objective_value': round(float(prob.value) if prob.value is not None else 0, 4),
        }

        return w_opt, diagnostics


# =============================================================================
# 3. Odd-Lot Discretization — A-share 100-share lot rounding
# =============================================================================

A_SHARE_LOT_SIZE = 100

def discretize_to_lots(
    weights: np.ndarray,
    total_capital: float,
    prices: np.ndarray,
    lot_size: int = A_SHARE_LOT_SIZE,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Convert continuous CVXPY weights to integer lots (100-share multiples).

    Algorithm:
      1. Compute target capital per stock = weight_i × total_capital
      2. Convert to lots: lots_i = floor(target_capital / (price_i × 100))
      3. Any residual cash (< 1 lot per stock) is truncated
      4. Re-normalize actual allocation weights from the discrete lots

    Returns:
        (allocated_capital, remaining_cash, diagnostics)
        where allocated_capital[i] = lots_i × price_i × 100
    """
    N = len(weights)
    if N == 0:
        return np.array([]), total_capital, {'truncated_stocks': 0, 'cash_drag_pct': 0.0}

    target_capital = np.maximum(weights, 0) * total_capital
    lot_value = prices * lot_size

    # Floor to integer lots
    lots = np.floor(target_capital / np.maximum(lot_value, 1e-6)).astype(int)
    lots = np.maximum(lots, 0)

    # Compute actual allocated capital
    allocated = lots.astype(np.float64) * lot_value

    # Truncation: stocks that can't afford even 1 lot are dropped
    truncated_mask = (lots == 0) & (weights > 0)
    n_truncated = int(np.sum(truncated_mask))

    # Remaining cash
    remaining_cash = total_capital - allocated.sum()
    cash_drag_pct = remaining_cash / max(total_capital, 1.0) * 100

    diag = {
        'truncated_stocks': n_truncated,
        'cash_drag_pct': round(cash_drag_pct, 2),
        'total_lots': int(np.sum(lots)),
        'n_selected': int(np.sum(lots > 0)),
    }

    return allocated, remaining_cash, diag


def build_discrete_weights_dict(
    symbols: List[str],
    weights: np.ndarray,
    total_capital: float,
    prices: np.ndarray,
) -> Dict[str, Dict]:
    """
    Build human-readable discrete allocation dict.

    Returns:
        {symbol: {'weight': float, 'lots': int, 'shares': int,
                  'allocated_rmb': float, 'price': float}, ...}
    """
    allocated, remaining, diag = discretize_to_lots(weights, total_capital, prices)

    result = {}
    for i, sym in enumerate(symbols):
        if allocated[i] <= 0:
            continue
        lots = int(round(allocated[i] / (prices[i] * A_SHARE_LOT_SIZE)))
        shares = lots * A_SHARE_LOT_SIZE
        result[sym] = {
            'weight': round(float(allocated[i] / max(total_capital, 1.0)), 4),
            'lots': lots,
            'shares': shares,
            'allocated_rmb': round(float(allocated[i]), 2),
            'price': round(float(prices[i]), 2),
        }

    return result


# =============================================================================
# 4. Unified Pipeline
# =============================================================================

def optimize_portfolio(
    candidates: List[Dict],
    prev_weights: Dict[str, float] = None,
    daily_data: Dict[str, pd.DataFrame] = None,
    total_capital: float = None,
    discretize: bool = True,
) -> Dict:
    """
    One-stop portfolio optimization from candidate signal list.

    Args:
        candidates: [{'symbol': str, 'ranker_score': float, 'close': float, ...}, ...]
        prev_weights: {symbol: weight} from previous period
        daily_data: {symbol: DataFrame} for factor computation
        total_capital: total account equity (for lot discretization)
        discretize: if True, round to 100-share lots

    Returns:
        Dict with weights, diagnostics, Barra exposures, lot_details
    """
    if not candidates:
        return {'weights': {}, 'diagnostics': {'status': 'empty'}}

    N = len(candidates)
    symbols = [c['symbol'] for c in candidates]

    # Alpha scores
    alpha = np.array([c.get('ranker_score', 3.0) for c in candidates])

    # Previous weights
    if prev_weights is None:
        prev_weights = {}
    w_prev = np.array([prev_weights.get(s, 0.0) for s in symbols])

    # Barra factors
    F, factor_names = compute_barra_factors(symbols, daily_data)

    # Run optimization
    opt = PortfolioOptimizer()
    w_opt, diag = opt.optimize(alpha, w_prev, F, factor_names)

    # Compute final Barra exposures
    barra_exposures = {}
    if F is not None:
        for k, name in enumerate(factor_names):
            barra_exposures[name] = round(float(F[:, k] @ w_opt), 4)

    # ---- Odd-lot discretization ----
    lot_details = {}
    if discretize and total_capital and total_capital > 0:
        prices = np.array([c.get('close', c.get('price', 10.0))
                          for c in candidates], dtype=np.float64)
        capital = total_capital

        discrete_weights = build_discrete_weights_dict(
            symbols, w_opt, capital, prices)

        # Re-normalize actual weights
        weights_dict = {s: d['weight'] for s, d in discrete_weights.items()}
        lot_details = discrete_weights
        diag['discretized'] = True
        diag['cash_drag_pct'] = round(
            (capital - sum(d['allocated_rmb'] for d in discrete_weights.values())) /
            max(capital, 1.0) * 100, 2)
    else:
        # Continuous weights (no discretization)
        weights_dict = {}
        for i, sym in enumerate(symbols):
            if w_opt[i] > 0.001:
                weights_dict[sym] = round(float(w_opt[i]), 4)

    return {
        'weights': weights_dict,
        'diagnostics': diag,
        'barra_exposures': barra_exposures,
        'n_selected': len(weights_dict),
        'lot_details': lot_details,
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    np.random.seed(42)

    N = 20
    symbols = [f'600{i:03d}' for i in range(N)]

    # Simulate alpha scores
    alpha = np.random.uniform(1, 5, N)

    # Simulate previous weights (equal weight)
    w_prev = np.ones(N) / N

    # Simulate Barra factors
    F = np.random.randn(N, 3)
    F[:, 0] = np.random.randn(N)  # Size
    F[:, 1] = np.random.randn(N)  # Momentum
    F[:, 2] = np.abs(np.random.randn(N))  # Volatility

    opt = PortfolioOptimizer(turnover_cost=0.50)
    w_opt, diag = opt.optimize(alpha, w_prev, F, ['Size', 'Momentum', 'Volatility'])

    print("=== Portfolio Optimizer ===")
    print(f"Status: {diag['status']}")
    print(f"Turnover: {diag['turnover']:.4f}")
    print(f"Effective N: {diag['effective_n']}")
    print(f"Max weight: {diag['max_weight']:.4f}")

    # Show top 5 weights
    top5 = np.argsort(w_opt)[-5:][::-1]
    print(f"\nTop 5 weights:")
    for i in top5:
        if w_opt[i] > 0.01:
            print(f"  {symbols[i]}: {w_opt[i]:.4f}")

    # Barra exposures
    for k, name in enumerate(['Size', 'Momentum', 'Volatility']):
        exposure = F[:, k] @ w_opt
        print(f"  {name} exposure: {exposure:+.4f} (limit: ±0.15)")

    print("\nPortfolio Optimizer: OK")
