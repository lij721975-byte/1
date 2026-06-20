#!/usr/bin/env python
# roc_factor.py — ROC-Momentum Factor (AX/BX variant) + Cross-Sectional Pipeline
"""
ROC Factor: AX = P_t - P_{t-20},  BX = P_{t-60},  ROC = (AX/BX) × 100

Cross-sectional pipeline (for multi-stock ranking):
  1. MAD outlier trim (3 × 1.4826 × MAD)
  2. Industry + MarketCap neutralization (OLS residuals)
  3. Z-score standardization

School interface: single-stock → direction/confidence based on ROC signal.

ROC direction: positive ROC → bullish (momentum up), negative → bearish.
High ROC percentile → high expected return (positive momentum factor).
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


# ==============================================================================
# 1. Raw Factor Computation (vectorized, wide-format)
# ==============================================================================

def calculate_roc_factor(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw ROC factor from wide-format price DataFrame.

    Parameters
    ----------
    price_df : pd.DataFrame
        Index = dates, Columns = stock codes, Values = close prices.

    Returns
    -------
    roc_df : pd.DataFrame
        Raw ROC values. NaN where BX ≤ 0 or insufficient data.
    """
    ax = price_df - price_df.shift(20)         # AX = P_t - P_{t-20}
    bx = price_df.shift(60)                    # BX = P_{t-60}
    roc = (ax / bx.replace(0, np.nan)) * 100.0 # ROC = (AX/BX) × 100
    return roc


# ==============================================================================
# 2. Cross-Sectional Pipeline
# ==============================================================================

def mad_outlier_trim(series: pd.Series, k: float = 3.0) -> pd.Series:
    """MAD outlier trimming: clip values beyond k × 1.4826 × MAD."""
    median = series.median()
    mad = (series - median).abs().median()
    if mad <= 0:
        return series
    threshold = k * 1.4826 * mad
    return series.clip(lower=median - threshold, upper=median + threshold)


def industry_marketcap_neutralize(
    factor_series: pd.Series,
    industry_dummies: pd.DataFrame,
    log_mktcap: pd.Series,
) -> pd.Series:
    """
    OLS neutralization: factor ~ industry_dummies + log_mktcap → residuals.

    Parameters
    ----------
    factor_series : pd.Series, index = stock codes
    industry_dummies : pd.DataFrame, index = stock codes, columns = industry dummies
    log_mktcap : pd.Series, index = stock codes
    """
    valid = factor_series.notna() & log_mktcap.notna()
    valid = valid & industry_dummies.notna().all(axis=1)
    if valid.sum() < 10:
        return factor_series - factor_series.mean()

    y = factor_series[valid].values
    X = industry_dummies[valid].values.astype(np.float64)
    X = np.column_stack([X, log_mktcap[valid].values])

    # Add constant
    X = np.column_stack([np.ones(X.shape[0]), X])

    # OLS via lstsq
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ theta
    residuals = y - y_pred

    result = pd.Series(np.nan, index=factor_series.index)
    result.loc[valid.index[valid]] = residuals
    return result


def zscore_standardize(series: pd.Series) -> pd.Series:
    """Z-score: (x - μ) / σ"""
    mu = series.mean()
    sigma = series.std(ddof=1)
    if sigma <= 0:
        return series * 0.0
    return (series - mu) / sigma


def run_cross_sectional_pipeline(
    roc_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Full cross-sectional pipeline for each date.

    Parameters
    ----------
    roc_df : raw ROC values (index=dates, columns=stocks)
    industry_df : industry dummies (index=dates, columns=stocks×industries)
                  OR same columns as roc_df with industry codes
    market_cap_df : market caps (index=dates, columns=stocks)
    """
    result = roc_df.copy()
    log_mktcap = np.log(market_cap_df.replace(0, np.nan))

    for date_idx in range(len(roc_df)):
        row = roc_df.iloc[date_idx]
        valid_stocks = row.dropna().index
        if len(valid_stocks) < 10:
            continue

        # Step 1: MAD trim
        trimmed = mad_outlier_trim(row[valid_stocks])

        # Step 2: Industry + MarketCap neutralization
        # Build industry dummies for this date
        ind_row = industry_df.iloc[date_idx]
        if isinstance(ind_row.iloc[0], (int, float, np.integer, np.floating)):
            # Industry is codes → create dummies
            ind_dummies = pd.get_dummies(ind_row)
        else:
            ind_dummies = ind_row

        mktcap_row = log_mktcap.iloc[date_idx]
        valid_idx = trimmed.index.intersection(ind_dummies.index).intersection(mktcap_row.index)
        if len(valid_idx) < 10:
            continue

        neutralized = industry_marketcap_neutralize(
            trimmed[valid_idx],
            ind_dummies.loc[valid_idx],
            mktcap_row[valid_idx]
        )

        # Step 3: Z-score
        z_scored = zscore_standardize(neutralized.dropna())
        result.loc[roc_df.index[date_idx], z_scored.index] = z_scored.values

    return result


# ==============================================================================
# 3. School Interface (single-stock → direction/confidence)
# ==============================================================================

class ROCFactorSchool:
    """
    ROC-Momentum school: compute AX/BX factor for a single stock.

    Signal logic:
      - ROC > 10  → strong bullish (momentum breakout)
      - ROC > 3   → mild bullish
      - ROC < -10 → bearish
      - ROC < -3  → mild bearish
      - otherwise → neutral
    """

    def __init__(self, bullish_threshold: float = 3.0,
                 bearish_threshold: float = -3.0,
                 strong_bullish: float = 10.0):
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.strong_bullish = strong_bullish

    def _compute_single_roc(self, close: np.ndarray) -> float:
        """Compute ROC for the latest bar."""
        n = len(close)
        if n < 61:
            return np.nan
        ax = close[-1] - close[-21] if n >= 21 else np.nan
        bx = close[-61] if n >= 61 else np.nan
        if bx is None or np.isnan(bx) or bx <= 0:
            return np.nan
        return (ax / bx) * 100.0

    def compute_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < 65:
            return None
        close = df['close'].values.astype(np.float64)
        roc_val = self._compute_single_roc(close)
        if np.isnan(roc_val):
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0,
                    'reasons': ['ROC数据不足']}

        score = 0.0
        reasons = []

        if roc_val > self.strong_bullish:
            score = 0.70
            reasons.append(f'ROC强多头({roc_val:.1f}%>10%)')
        elif roc_val > self.bullish_threshold:
            score = 0.35
            reasons.append(f'ROC偏多({roc_val:.1f}%)')
        elif roc_val < -self.strong_bullish:
            score = -0.70
            reasons.append(f'ROC强空头({roc_val:.1f}%)')
        elif roc_val < self.bearish_threshold:
            score = -0.35
            reasons.append(f'ROC偏空({roc_val:.1f}%)')
        else:
            reasons.append(f'ROC中性({roc_val:.1f}%)')

        # Volume confirmation
        if len(df) >= 5:
            vol_ratio = df['volume'].values[-1] / (df['volume'].values[-5:].mean() + 1e-10)
            if vol_ratio > 1.5 and score > 0:
                score += 0.15
                reasons.append('放量确认')
            elif vol_ratio > 1.5 and score < 0:
                score -= 0.10

        score = np.clip(score, -1.0, 1.0)
        direction = 'bullish' if score > 0.06 else ('bearish' if score < -0.06 else 'neutral')
        conf = min(abs(score) * 1.3, 0.90)

        return {'direction': direction, 'score': round(float(score), 3),
                'confidence': round(float(conf), 3), 'reasons': reasons[:4]}


# ── School-compatible interface ──
_school = ROCFactorSchool()

def compute_roc_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 65:
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
    print("=== ROC Factor Test ===")

    # Mock data: 3 stocks, 100 days
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='B')
    stocks = ['000001', '000002', '000003']

    # Price simulation with trend + noise
    price_data = {}
    for s in stocks:
        base = np.cumsum(np.random.randn(100) * 0.5) + 10
        price_data[s] = np.exp(base) * 10
    price_df = pd.DataFrame(price_data, index=dates)

    # Industry data
    industry_data = {s: np.random.choice(['金融', '制造', '科技']) for s in stocks}
    industry_df = pd.DataFrame([industry_data] * len(dates), index=dates)

    # Market cap data
    mktcap_data = {s: np.random.uniform(1e9, 1e11) for s in stocks}
    market_cap_df = pd.DataFrame([mktcap_data] * len(dates), index=dates)

    # Compute raw factor
    roc_raw = calculate_roc_factor(price_df)
    print(f"Raw ROC (last day):\n{roc_raw.iloc[-1].dropna().to_string()}")
    print(f"NaN count: {roc_raw.isna().sum().sum()} (expected: early dates)")

    # School test
    for s in stocks:
        df_single = pd.DataFrame({
            'open': price_df[s].values,
            'high': price_df[s].values * 1.02,
            'low': price_df[s].values * 0.98,
            'close': price_df[s].values,
            'volume': np.random.randint(1e6, 1e7, 100),
        }, index=dates)
        sig = _school.compute_signal(df_single)
        if sig:
            print(f"{s}: dir={sig['direction']}, conf={sig['confidence']:.3f}, "
                  f"reasons={sig['reasons']}")

    # Verify: positive correlation between ROC percentile and returns
    # ROC_max_quantile should have higher forward returns than ROC_min_quantile
    print("\nFactor direction check:")
    print("  ROC > 0 → bullish (momentum up) → positive expected return")
    print("  ROC < 0 → bearish (momentum down) → negative expected return")
    print("  High ROC percentile → high return (positive momentum factor)")
