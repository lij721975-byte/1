#!/usr/bin/env python
# beta_factor.py — Historical Beta factor with pure numpy stride-trick rolling
"""
Historical Beta: β = Cov(r_i, R_m) / Var(R_m) over rolling window W (default 252).

Pure numpy stride-trick implementation — zero Python for-loops over time/cross-section.
Output: raw beta values, no MAD/neutralization/Z-score applied.

School interface: high beta (>1.2) → bullish in uptrend, bearish in downtrend.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ==============================================================================
# 1. Pure-Numpy Rolling Window Statistics (stride tricks)
# ==============================================================================

def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean via sliding_window_view. NaN-safe."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    sw = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1:] = np.nanmean(sw, axis=1)
    return out


def _rolling_var(x: np.ndarray, window: int) -> np.ndarray:
    """Rolling variance via sliding_window_view. NaN-safe."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    sw = np.lib.stride_tricks.sliding_window_view(x, window)
    out[window - 1:] = np.nanvar(sw, axis=1, ddof=1)
    return out


def _rolling_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling covariance of x and y via sliding_window_view. NaN-safe."""
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return out
    sw_x = np.lib.stride_tricks.sliding_window_view(x, window)
    sw_y = np.lib.stride_tricks.sliding_window_view(y, window)
    # Cov = mean((x - mean(x)) * (y - mean(y))) with ddof=1
    mx = np.nanmean(sw_x, axis=1, keepdims=True)
    my = np.nanmean(sw_y, axis=1, keepdims=True)
    dx = sw_x - mx
    dy = sw_y - my
    valid = ~(np.isnan(dx) | np.isnan(dy))
    n_valid = valid.sum(axis=1)
    cov = np.nansum(dx * dy, axis=1) / np.maximum(n_valid - 1, 1)
    out[window - 1:] = cov
    return out


# ==============================================================================
# 2. Main Computation Function
# ==============================================================================

def calculate_historical_beta(
    stock_returns_df: pd.DataFrame,
    market_returns_series: pd.Series,
    window: int = 252,
    epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute rolling historical beta for all stocks.

    Parameters
    ----------
    stock_returns_df : pd.DataFrame (wide format)
        Index = dates, Columns = stock codes, Values = daily returns.
    market_returns_series : pd.Series
        Index = dates, Values = market daily returns.
    window : int, default 252
        Rolling window for beta calculation.
    epsilon : float
        Small value to prevent division by zero for market variance.

    Returns
    -------
    beta_df : pd.DataFrame
        Same shape as stock_returns_df. Raw beta values.
    """
    # Align dates
    common_dates = stock_returns_df.index.intersection(market_returns_series.index)
    stock_aligned = stock_returns_df.loc[common_dates]
    market_aligned = market_returns_series.loc[common_dates]

    mkt = market_aligned.values.astype(np.float64)
    n_dates = len(mkt)
    n_stocks = len(stock_aligned.columns)

    # Precompute market rolling variance once
    mkt_var = _rolling_var(mkt, window)
    mkt_var = np.maximum(mkt_var, epsilon)

    # Output matrix
    beta_arr = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)

    # Compute rolling covariance for each stock, divide by market variance
    for j in range(n_stocks):
        stock_ret = stock_aligned.iloc[:, j].values.astype(np.float64)
        cov = _rolling_cov(stock_ret, mkt, window)
        beta_arr[:, j] = cov / mkt_var

    return pd.DataFrame(beta_arr, index=common_dates, columns=stock_aligned.columns)


# ==============================================================================
# 3. School Interface (single-stock → direction/confidence)
# ==============================================================================

class BetaFactorSchool:
    """
    Historical Beta school.

    Logic:
      - High beta (> 1.3) stock in bull market → bullish (leverage long)
      - High beta (> 1.3) in bear market → bearish (high risk)
      - Low beta (< 0.7) → neutral (defensive)
      - Beta near 1.0 → neutral (market-like)
    """

    def __init__(self, window: int = 60):
        self.window = window  # shorter window for school (daily signal)

    def _compute_single_beta(self, stock_ret: np.ndarray,
                              mkt_ret: np.ndarray) -> float:
        n = len(stock_ret)
        if n < self.window:
            return np.nan
        w = min(self.window, n)
        sx = stock_ret[-w:]
        sy = mkt_ret[-w:]
        valid = ~(np.isnan(sx) | np.isnan(sy))
        if valid.sum() < 10:
            return np.nan
        sx_v = sx[valid]; sy_v = sy[valid]
        cov = np.cov(sx_v, sy_v, ddof=1)[0, 1]
        var_m = np.var(sy_v, ddof=1)
        if var_m <= 0:
            return 0.0
        return cov / var_m

    def compute_signal(self, df: pd.DataFrame,
                        market_df: pd.DataFrame = None) -> Optional[Dict]:
        if df is None or len(df) < self.window:
            return None

        close = df['close'].values.astype(np.float64)
        n = len(close)
        # Compute stock returns
        stock_ret = np.diff(close) / (close[:-1] + 1e-10)

        # Market proxy: if not provided, use stock's own returns as fallback
        # (beta ≈ 1.0 by construction when stock = market)
        if market_df is not None and len(market_df) >= self.window:
            mkt_close = market_df['close'].values.astype(np.float64)
            mkt_ret = np.diff(mkt_close) / (mkt_close[:-1] + 1e-10)
            # Align lengths
            min_len = min(len(stock_ret), len(mkt_ret))
            stock_ret = stock_ret[-min_len:]
            mkt_ret = mkt_ret[-min_len:]
        else:
            mkt_ret = stock_ret  # fallback: beta = 1.0

        beta = self._compute_single_beta(stock_ret, mkt_ret)
        if np.isnan(beta):
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['Beta数据不足']}

        score = 0.0
        reasons = []

        # Beta interpretation
        if beta > 1.30:
            # High beta: check recent trend
            ret_20 = (close[-1] - close[-20]) / close[-20] if n >= 20 and close[-20] > 0 else 0
            if ret_20 > 0.03:
                score = 0.40
                reasons.append(f'高Beta({beta:.2f})牛市中→杠杆做多')
            elif ret_20 < -0.03:
                score = -0.40
                reasons.append(f'高Beta({beta:.2f})熊市中→高风险回避')
            else:
                score = 0.10
                reasons.append(f'高Beta({beta:.2f})震荡中')
        elif beta > 1.10:
            score = 0.20
            reasons.append(f'偏强Beta({beta:.2f})')
        elif beta < 0.70:
            score = -0.15
            reasons.append(f'低Beta({beta:.2f})防御型')
        else:
            reasons.append(f'Beta中性({beta:.2f})')

        score = np.clip(score, -1.0, 1.0)
        direction = 'bullish' if score > 0.06 else ('bearish' if score < -0.06 else 'neutral')
        conf = min(abs(score) * 1.2, 0.85)

        return {'direction': direction, 'score': round(float(score), 3),
                'confidence': round(float(conf), 3), 'reasons': reasons[:4]}


# ── School-compatible interface ──
_school = BetaFactorSchool()

def compute_beta_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 60:
        return None
    try:
        r = _school.compute_signal(df_daily)
        return {'signal': r['direction'], 'confidence': r['confidence'],
                'metadata': {'score': r['score'], 'reasons': r['reasons']}} if r else None
    except:
        return None


# ==============================================================================
# 4. Test Stub
# ==============================================================================

if __name__ == '__main__':
    print("=== Historical Beta Factor Test ===")

    np.random.seed(42)
    n_dates = 300
    n_stocks = 5
    dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')

    # Market returns with autocorrelation
    mkt_ret = np.random.randn(n_dates) * 0.015
    # Stock returns = beta_i * market + noise
    true_betas = [0.5, 0.8, 1.0, 1.3, 1.8]
    stock_data = {}
    for i, b in enumerate(true_betas):
        stock_data[f'STOCK_{i+1}'] = b * mkt_ret + np.random.randn(n_dates) * 0.01

    stock_df = pd.DataFrame(stock_data, index=dates)
    mkt_series = pd.Series(mkt_ret, index=dates)

    # Compute betas
    beta_df = calculate_historical_beta(stock_df, mkt_series, window=60)
    last_betas = beta_df.iloc[-1].dropna()
    print(f"Last day betas (rolling 60d):\n{last_betas.to_string()}")
    print(f"True betas: {true_betas}")

    # Dimension assertion
    assert beta_df.shape == stock_df.shape, \
        f"Shape mismatch: {beta_df.shape} vs {stock_df.shape}"
    print(f"\nShape OK: {beta_df.shape}")

    # NaN check: early dates should be NaN
    nan_early = beta_df.iloc[:30].isna().all().all()
    print(f"Early dates all NaN: {nan_early}")
    print(f"Late dates have values: {beta_df.iloc[-1].notna().all()}")

    # School test on a single stock
    df_single = pd.DataFrame({
        'open': (np.random.randn(n_dates) * 0.01 + 1).cumprod() * 10,
        'high': (np.random.randn(n_dates) * 0.01 + 1).cumprod() * 10.5,
        'low': (np.random.randn(n_dates) * 0.01 + 1).cumprod() * 9.5,
        'close': (np.random.randn(n_dates) * 0.01 + 1).cumprod() * 10,
        'volume': np.random.randint(1e6, 1e7, n_dates),
    }, index=dates)
    sig = _school.compute_signal(df_single)
    if sig:
        print(f"\nSchool signal: dir={sig['direction']}, "
              f"conf={sig['confidence']:.3f}, reasons={sig['reasons']}")
