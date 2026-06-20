#!/usr/bin/env python
# ewm_beta_factor.py — Exponentially-Weighted Moving Beta (halflife=63, window=252)
"""
EWM Beta: β = Σ(w_t × (r-μ_r) × (R-μ_R)) / Σ(w_t × (R-μ_R)²)

Weights: w_t = 0.5^(t/63),  window=252,  decay from oldest to newest.

Uses pandas ewm(halflife=63).cov() / .var() for vectorized computation.
Raw beta output — no MAD / neutralization / Z-score.

Direction: positive factor — high beta → high expected return (in bull markets).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ==============================================================================
# 1. EWM Beta Computation
# ==============================================================================

def calculate_ewm_beta(
    stock_close_df: pd.DataFrame,
    benchmark_close_series: pd.Series,
    window: int = 252,
    halflife: int = 63,
    min_periods: int = None,
) -> pd.DataFrame:
    """
    Compute EWM Beta for all stocks.

    Parameters
    ----------
    stock_close_df : pd.DataFrame (wide)
        Index = dates, Columns = stock codes, Values = close prices.
    benchmark_close_series : pd.Series
        Index = dates, Values = benchmark close prices.
    window : int
        Rolling lookback (default 252).
    halflife : int
        EWM half-life in bars (default 63).
    min_periods : int or None
        Min observations required. Defaults to window // 3.

    Returns
    -------
    beta_df : pd.DataFrame
        Same shape as input. EWM beta values.
    """
    if min_periods is None:
        min_periods = max(30, window // 3)

    # Align dates
    common = stock_close_df.index.intersection(benchmark_close_series.index)
    stock_aligned = stock_close_df.loc[common]
    bm_aligned = benchmark_close_series.loc[common]

    # Daily returns
    stock_ret = stock_aligned.pct_change()
    bm_ret = bm_aligned.pct_change()

    # EWM rolling cov(stock_i, benchmark) and var(benchmark)
    # pandas ewm().cov() produces pair-wise cov; we use it per-stock
    beta_df = pd.DataFrame(index=common, columns=stock_aligned.columns, dtype=np.float64)

    for col in stock_aligned.columns:
        combined = pd.concat([stock_ret[col], bm_ret], axis=1)
        combined.columns = ['stock', 'bm']
        # Drop NaN rows for this stock
        valid = combined.dropna()
        if len(valid) < min_periods:
            continue

        # EWM rolling covariance matrix
        ewm_cov = valid.ewm(halflife=halflife, min_periods=min_periods).cov(pairwise=True)

        # Extract stock-vs-benchmark covariance at each date
        cov_series = ewm_cov.loc[(slice(None), 'stock'), 'bm']
        cov_series.index = cov_series.index.droplevel(1)

        # EWM rolling variance of benchmark
        var_series = bm_ret.ewm(halflife=halflife, min_periods=min_periods).var()

        # Beta = cov(stock,bm) / var(bm)
        beta_series = cov_series / var_series.replace(0, np.nan)
        beta_df[col] = beta_series.reindex(common)

    return beta_df


# ==============================================================================
# 2. School Interface
# ==============================================================================

class EWMBetaSchool:
    """
    EWM Beta school.

    Direction logic (positive factor):
      β > 1.30 + recent uptrend → bullish (high-beta boost)
      β > 1.30 + recent downtrend → bearish
      β < 0.70 → defensive neutral
      β near 1.0 → market neutral
    """

    def __init__(self, window: int = 60, halflife: int = 21):
        self.window = window
        self.halflife = halflife

    def _compute_single_ewm_beta(
        self, stock_ret: np.ndarray, bm_ret: np.ndarray
    ) -> float:
        n = len(stock_ret)
        if n < 10:
            return np.nan
        w = min(self.window, n)
        sr = pd.Series(stock_ret[-w:])
        br = pd.Series(bm_ret[-w:])

        # EWM
        ewm_sr = sr.ewm(halflife=self.halflife, min_periods=10)
        ewm_br = br.ewm(halflife=self.halflife, min_periods=10)

        cov_val = sr.ewm(halflife=self.halflife, min_periods=10).cov(br).iloc[-1]
        var_val = br.ewm(halflife=self.halflife, min_periods=10).var().iloc[-1]

        if pd.isna(cov_val) or pd.isna(var_val) or var_val <= 0:
            return np.nan
        return cov_val / var_val

    def compute_signal(self, df: pd.DataFrame,
                        benchmark_df: pd.DataFrame = None) -> Optional[Dict]:
        if df is None or len(df) < self.window:
            return None

        close = df['close'].values.astype(np.float64)
        n = len(close)
        stock_ret = np.diff(close) / (close[:-1] + 1e-10)

        if benchmark_df is not None and len(benchmark_df) >= self.window:
            bm_close = benchmark_df['close'].values.astype(np.float64)
            bm_ret = np.diff(bm_close) / (bm_close[:-1] + 1e-10)
        else:
            bm_ret = stock_ret

        min_len = min(len(stock_ret), len(bm_ret))
        stock_ret = stock_ret[-min_len:]
        bm_ret = bm_ret[-min_len:]

        beta = self._compute_single_ewm_beta(stock_ret, bm_ret)
        if np.isnan(beta):
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['EWM Beta数据不足']}

        score = 0.0; reasons = []
        ret_20 = (close[-1] - close[-20]) / close[-20] if n >= 20 and close[-20] > 0 else 0

        if beta > 1.30 and ret_20 > 0.03:
            score = 0.45; reasons.append(f'EWM高Beta({beta:.2f})+牛市→做多')
        elif beta > 1.30 and ret_20 < -0.03:
            score = -0.35; reasons.append(f'EWM高Beta({beta:.2f})+熊市→回避')
        elif beta > 1.10:
            score = 0.20; reasons.append(f'EWM偏强Beta({beta:.2f})')
        elif beta < 0.70:
            score = -0.10; reasons.append(f'EWM低Beta({beta:.2f})')
        else:
            reasons.append(f'EWM Beta中性({beta:.2f})')

        score = np.clip(score, -1.0, 1.0)
        direction = 'bullish' if score > 0.06 else ('bearish' if score < -0.06 else 'neutral')
        conf = min(abs(score) * 1.2, 0.85)

        return {'direction': direction, 'score': round(float(score), 3),
                'confidence': round(float(conf), 3), 'reasons': reasons[:4]}


# ── School-compatible interface ──
_school = EWMBetaSchool()

def compute_ewm_beta_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 60:
        return None
    try:
        r = _school.compute_signal(df_daily)
        return {'signal': r['direction'], 'confidence': r['confidence'],
                'metadata': {'score': r['score'], 'reasons': r['reasons']}} if r else None
    except:
        return None


# ==============================================================================
# 3. Test Stub
# ==============================================================================

if __name__ == '__main__':
    print("=== EWM Beta Factor Test ===")

    np.random.seed(42)
    n_dates = 400
    dates = pd.date_range('2024-06-01', periods=n_dates, freq='B')
    n_stocks = 4

    # Simulate benchmark
    bm_ret = np.random.randn(n_dates) * 0.012
    bm_close = pd.Series((1 + bm_ret).cumprod() * 3000, index=dates,
                         name='benchmark')

    # Simulate stocks with known betas
    true_betas = [0.6, 1.0, 1.4, 1.9]
    stock_data = {}
    for i, b in enumerate(true_betas):
        s_ret = b * bm_ret + np.random.randn(n_dates) * 0.008
        stock_data[f'S{i+1}'] = (1 + s_ret).cumprod() * 10
    stock_df = pd.DataFrame(stock_data, index=dates)

    # Compute EWM betas
    beta_df = calculate_ewm_beta(stock_df, bm_close, window=252, halflife=63)
    last = beta_df.iloc[-1].dropna()
    print(f"Last day EWM betas:\n{last.to_string()}")
    print(f"True betas: {true_betas}")

    assert beta_df.shape == stock_df.shape
    print(f"Shape OK: {beta_df.shape}")

    # Factor direction: positive — high beta → high expected return
    print("\nFactor direction: POSITIVE (high beta → high return, no sign flip needed)")

    # School test
    df_single = pd.DataFrame({
        'open': stock_df['S1'] * 0.99,
        'high': stock_df['S1'] * 1.02,
        'low': stock_df['S1'] * 0.98,
        'close': stock_df['S1'],
        'volume': np.random.randint(1e6, 1e7, n_dates),
    }, index=dates)
    sig = _school.compute_signal(df_single)
    if sig:
        print(f"School: dir={sig['direction']} conf={sig['confidence']:.3f} {sig['reasons']}")
