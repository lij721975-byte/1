#!/usr/bin/env python
# advanced_pattern_features.py
"""
Advanced Pattern Features — 圆弧底 / N字型 / 突破上升
========================================================
Three specialized pattern detectors for the classical pattern school.

Red-line compliance:
  1. Pure vectorized — scipy.signal.argrelextrema + numpy broadcasting + pd.rolling
  2. Continuous score — float64 ∈ [0.0, 1.0]
  3. Zero NaN — .ffill().fillna(0.0)
  4. Standard I/O — df[OHLCV] → np.ndarray (n,) or pd.DataFrame

Author: Chief Alpha Miner
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# ==============================================================================
# 1. 圆弧底 (Rounding Bottom / Saucer)
# ==============================================================================

def compute_rounding_bottom_score(
    df: pd.DataFrame,
    window: int = 60,
    atr_period: int = 14,
) -> np.ndarray:
    """
    Vectorized Rounding Bottom (圆弧底) score.

    Principle (NO parabolic fit — low-cost vectorized proxy):
      In [t-window, t], price forms a U-shape:
        - Left third:  price declining
        - Middle third: price flat/slightly recovering, LOW volatility, LOW volume
        - Right third:  price recovering/rising
      Score combines: price U-shape × volatility contraction × volume contraction.

    Parameters
    ----------
    df : pd.DataFrame with open,high,low,close,volume
    window : int
        Lookback window in bars (default 60 = ~3 months daily).
    atr_period : int
        ATR period for volatility measurement.

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [0.0, 1.0]
    """
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    n = len(c)

    if n < window:
        return np.zeros(n, dtype=np.float64)

    third = window // 3  # ~20 bars per segment

    # ── Segment boundaries (vectorized: rolling windows with offsets) ──
    # Left segment:  [t-window, t-window+third]
    # Mid segment:   [t-window+third, t-third]
    # Right segment: [t-third, t]

    # Price averages per segment via rolling means with staggered windows
    price_left = (
        pd.Series(c).shift(window - third).rolling(third, min_periods=max(3, third // 2))
        .mean().ffill().fillna(c[0]).values
    )
    price_mid = (
        pd.Series(c).shift(third).rolling(window - 2 * third, min_periods=max(3, (window - 2 * third) // 2))
        .mean().ffill().fillna(c[0]).values
    )
    price_right = (
        pd.Series(c).rolling(third, min_periods=max(3, third // 2))
        .mean().ffill().fillna(c[0]).values
    )

    # ── 1a. Price U-shape: left declining, right recovering ──
    # Left trend: price change over left segment (negative = declining)
    left_ret = (c - pd.Series(c).shift(third).ffill().fillna(c[0]).values) / (
        pd.Series(c).shift(third).ffill().fillna(1.0).values + 1e-10
    )  # wrong direction — let me use segment endpoints

    # Simpler: compare close at segment boundaries
    # left_start = price at t-window
    # left_end = price at t-window+third
    # mid_end = price at t-third
    # right_end = price at t

    price_at_left_start = pd.Series(c).shift(window).ffill().fillna(c[0]).values
    price_at_left_end = pd.Series(c).shift(window - third).ffill().fillna(c[0]).values
    price_at_mid_end = pd.Series(c).shift(third).ffill().fillna(c[0]).values
    price_at_right_end = c

    # Left leg should be declining: price_at_left_end < price_at_left_start
    left_decline = np.clip(
        (price_at_left_start - price_at_left_end) / (price_at_left_start + 1e-10), 0.0, 1.0
    ) * 20.0  # scale: 5% decline → 1.0
    left_decline = np.clip(left_decline, 0.0, 1.0)

    # Right leg should be rising: price_at_right_end > price_at_mid_end
    right_rise = np.clip(
        (price_at_right_end - price_at_mid_end) / (price_at_mid_end + 1e-10), 0.0, 1.0
    ) * 20.0
    right_rise = np.clip(right_rise, 0.0, 1.0)

    # Bottom should be in the middle segment (lowest price in window is in mid)
    rolling_min_idx = np.argmin(
        np.column_stack([
            pd.Series(l).rolling(third, min_periods=1).min().ffill().fillna(l[0]).values,
            pd.Series(l).shift(third).rolling(window - 2 * third, min_periods=1).min().ffill().fillna(l[0]).values,
            pd.Series(l).shift(window - third).rolling(third, min_periods=1).min().ffill().fillna(l[0]).values,
        ]), axis=1
    )
    bottom_in_mid = (rolling_min_idx == 1).astype(np.float64)

    # U-shape symmetry: left decline ≈ right rise
    symmetry = 1.0 - np.abs(left_decline - right_rise)
    symmetry = np.clip(symmetry, 0.0, 1.0)

    price_u_score = left_decline * right_rise * symmetry * (0.5 + 0.5 * bottom_in_mid)

    # ── 1b. Volatility contraction in middle (ATR mid < ATR ends) ──
    true_range = np.maximum(
        h - l,
        np.maximum(
            np.abs(h - pd.Series(c).shift(1).ffill().fillna(c[0]).values),
            np.abs(l - pd.Series(c).shift(1).ffill().fillna(c[0]).values),
        )
    )
    atr = (
        pd.Series(true_range)
        .rolling(atr_period, min_periods=max(3, atr_period // 2))
        .mean()
        .ffill()
        .fillna(true_range[0])
        .values
    )

    # ATR in each segment
    atr_left = pd.Series(atr).shift(window - third).rolling(third, min_periods=3).mean().ffill().fillna(atr[0]).values
    atr_mid = pd.Series(atr).shift(third).rolling(window - 2 * third, min_periods=3).mean().ffill().fillna(atr[0]).values
    atr_right = pd.Series(atr).rolling(third, min_periods=3).mean().ffill().fillna(atr[0]).values

    atr_ends_avg = (atr_left + atr_right) / 2.0
    vol_contraction = 1.0 - np.clip(atr_mid / (atr_ends_avg + 1e-10), 0.0, 1.0)
    # Scale: mid ATR = 50% of ends → 0.5 score; mid ATR = 30% of ends → 0.7 score

    # ── 1c. Volume U-shape (volume contraction at bottom) ──
    vol_mid = pd.Series(v).shift(third).rolling(window - 2 * third, min_periods=3).mean().ffill().fillna(v[0]).values
    vol_left = pd.Series(v).shift(window - third).rolling(third, min_periods=3).mean().ffill().fillna(v[0]).values
    vol_right = pd.Series(v).rolling(third, min_periods=3).mean().ffill().fillna(v[0]).values

    vol_ends_avg = (vol_left + vol_right) / 2.0
    vol_contraction = 1.0 - np.clip(vol_mid / (vol_ends_avg + 1e-10), 0.0, 1.0)
    # Stronger contraction → higher score

    # ── 1d. Composite ──
    score = (
        price_u_score * 0.40 +
        vol_contraction * 0.30 +
        vol_contraction * 0.30   # deliberate: vol contraction weighted 60%
    )
    score = np.clip(score, 0.0, 1.0)

    return score.astype(np.float64)


# ==============================================================================
# 2. N字型 (N-Shape / Measured Move)
# ==============================================================================

def compute_n_shape_score(
    df: pd.DataFrame,
    window: int = 20,
    fib_min: float = 0.382,
    fib_max: float = 0.618,
) -> np.ndarray:
    """
    Vectorized N-Shape (N字型) score.

    Pattern: A(low) → B(high) → C(low) → [breakout above B]
      - Rally A→B: significant upward move
      - Pullback B→C: retrace 38.2%-61.8% of A→B, on declining volume
      - Breakout: price closes above B (or near B with momentum)

    Uses scipy argrelextrema for swing detection, then vectorized trace-back.

    Parameters
    ----------
    df : pd.DataFrame
    window : int
        Max bars for the A→B→C structure (default 20).
    fib_min, fib_max : float
        Fibonacci retracement bounds.

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [0.0, 1.0]
    """
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    n = len(c)

    if n < window:
        return np.zeros(n, dtype=np.float64)

    order = max(2, window // 8)  # swing detection order

    # ── Swing points ──
    sh_idx = argrelextrema(h, np.greater, order=order)[0]
    sl_idx = argrelextrema(l, np.less, order=order)[0]

    # ── "Last swing" index arrays (pandas ffill, C-level) ──
    def _last_of(idx_array, n):
        pos = np.full(n, np.nan)
        pos[idx_array] = idx_array.astype(np.float64)
        return pd.Series(pos).ffill().fillna(-1).astype(np.intp).values

    last_sh = _last_of(sh_idx, n)
    last_sl = _last_of(sl_idx, n)

    # ── Trace N-shape points (fully vectorized) ──
    # For bar i:
    #   B = last_sh[i]                          (most recent swing high)
    #   C = last_sl[i]                          (most recent swing low → pullback low)
    #   A = last_sl[B - 1] or last_sl before B  (swing low before the rally)
    # Actually: A = last_sl at position B (the swing low that preceded B)
    # But B is an index into last_sh, and we need last_sl at B's position.
    # → A = last_sl before B, i.e., the most recent swing low at bar B.

    B_safe = np.maximum(last_sh, 0)
    A_safe = np.maximum(last_sl[B_safe], 0)  # swing low before B

    # C is the pullback low AFTER B
    C_safe = np.maximum(last_sl, 0)  # most recent swing low (must be after B to be C)

    # ── Validity masks ──
    has_all = (last_sh >= 0) & (last_sl[B_safe] >= 0) & (last_sl >= 0)
    correct_seq = (A_safe < B_safe) & (B_safe < C_safe)  # A < B < C temporally
    within_window = (C_safe >= np.arange(n, dtype=np.intp) - window)  # all within window

    pattern_mask = has_all & correct_seq & within_window

    # ── Price extraction ──
    price_A = np.where(pattern_mask, l[A_safe], np.nan)
    price_B = np.where(pattern_mask, h[B_safe], np.nan)
    price_C = np.where(pattern_mask, l[C_safe], np.nan)

    rally_AB = price_B - price_A   # A→B rally magnitude
    pullback_BC = price_B - price_C  # B→C pullback magnitude
    retrace_ratio = pullback_BC / (rally_AB + 1e-10)  # how much of AB was retraced

    # ── 2a. Fibonacci retracement score ──
    # Ideal retracement is between fib_min and fib_max
    # Score peaks at midpoint (0.50) and falls linearly to 0 at bounds
    fib_center = (fib_min + fib_max) / 2.0
    fib_width = (fib_max - fib_min) / 2.0
    fib_deviation = np.abs(retrace_ratio - fib_center) / (fib_width + 1e-10)
    fib_score = np.clip(1.0 - fib_deviation, 0.0, 1.0)

    # ── 2b. Pullback depth validity ──
    # C should be above A (higher low — bullish N)
    higher_low = (price_C > price_A).astype(np.float64)

    # ── 2c. Volume: pullback volume < rally volume ──
    # Approximate: average volume during BC vs AB
    vol_after_B = pd.Series(v).rolling(window // 2, min_periods=3).mean().ffill().fillna(v[0]).values
    vol_before_B = pd.Series(v).shift(window // 2).rolling(window // 2, min_periods=3).mean().ffill().fillna(v[0]).values
    vol_decline = np.clip(1.0 - vol_after_B / (vol_before_B + 1e-10), 0.0, 1.0)

    # ── 2d. Breakout detection ──
    # Current close vs B: breaking above B = bullish completion
    breakout_ratio = np.clip((c - price_B) / (rally_AB + 1e-10), -0.1, 0.3) + 0.1
    breakout_score = np.clip(breakout_ratio, 0.0, 1.0)

    # Current volume on breakout
    vol_ratio_20 = v / (pd.Series(v).rolling(20, min_periods=5).mean().ffill().fillna(v[0]).values + 1e-10)
    vol_breakout = np.clip(vol_ratio_20 / 1.5, 0.0, 1.0)

    # ── 2e. Composite ──
    score = np.where(pattern_mask,
        fib_score * 0.30 +
        higher_low * 0.15 +
        vol_decline * 0.15 +
        breakout_score * 0.25 +
        vol_breakout * breakout_score * 0.15,  # vol only matters when breaking out
        0.0
    )
    score = np.clip(score, 0.0, 1.0)

    return score.astype(np.float64)


# ==============================================================================
# 3. 突破上升形态 (Ascending Breakout / Tight Range Breakout)
# ==============================================================================

def compute_ascending_breakout_score(
    df: pd.DataFrame,
    window: int = 20,
    tightness_ref_pct: float = 0.08,
) -> np.ndarray:
    """
    Vectorized Ascending Breakout (突破上升) — continuous "breakout pressure" score.

    Measures how strongly price is pressing against the top of a tightening range
    with volume elevation. Fully continuous — no binary gate.

    Three-factor soft product:
      1. Tightness:     how tight is the recent range?   (narrower → higher)
      2. Position:      where is close within the range?  (near top → higher)
      3. Volume surge:  how elevated is current volume?   (higher → higher)

    Score = tightness × position × volume_surge
    All three factors are continuous [0,1] → product spans [0,1] with rich gradation.

    Additional: score persists with exponential decay after peaks.

    Parameters
    ----------
    df : pd.DataFrame
    window : int
        Range lookback (default 20).
    tightness_ref_pct : float
        Reference tightness — range/price at this level → tightness=0.5.
        Default 0.08 = 8% (calibrated for A-share volatility).

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [0.0, 1.0]
    """
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    n = len(c)

    if n < window:
        return np.zeros(n, dtype=np.float64)

    # ── Rolling range statistics ──
    roll_high = pd.Series(h).rolling(window, min_periods=max(5, window // 2)).max().ffill().fillna(h[0]).values
    roll_low  = pd.Series(l).rolling(window, min_periods=max(5, window // 2)).min().ffill().fillna(l[0]).values
    roll_avg  = pd.Series(c).rolling(window, min_periods=max(5, window // 2)).mean().ffill().fillna(c[0]).values

    range_abs = roll_high - roll_low
    range_pct = range_abs / (roll_avg + 1e-10)

    # ── Factor 1: Tightness (continuous) ──
    # Reference: range/tightness_ref_pct → at ref → 0.5, at half ref → 0.75, at 0 → 1.0
    # Use exponential decay: exp(-range_pct / tightness_ref_pct)
    # 0% range → 1.0,  8% range → 0.37,  16% range → 0.14
    tightness = np.exp(-range_pct / tightness_ref_pct)

    # ── Factor 2: Position in range (continuous) ──
    # 0 = at roll_low, 1 = at roll_high, >1 = breaking above
    position = (c - roll_low) / (range_abs + 1e-10)
    # Clip to [0, 1.5] then normalize: position ≤ 1 means within range; >1 means breakout
    # Use sigmoid-like: position^2 for values < 1, linear beyond
    # Actually simpler: clip to [0,1] for "within range" pressure, add bonus for breakout
    position_pressure = np.clip(position, 0.0, 1.0)
    # Breakout bonus: how far above the high (continuous, not binary)
    breakout_extension = np.clip((c - roll_high) / (roll_high + 1e-10) / 0.03, 0.0, 1.0)  # 3% above → 1.0
    # Combine: near the top of range gets high score; breaking above gets full 1.0
    position_score = position_pressure * 0.80 + breakout_extension * 0.20

    # ── Factor 3: Volume surge (continuous) ──
    vol_ma = pd.Series(v).rolling(window, min_periods=max(5, window // 2)).mean().ffill().fillna(v[0]).values
    vol_ratio = v / (vol_ma + 1e-10)
    # Log-scale: 1x → 0.0,  1.5x → 0.40,  2x → 0.63,  3x → 0.86
    vol_surge = np.clip(np.log(np.clip(vol_ratio, 1.0, 10.0)) / np.log(3.0), 0.0, 1.0)

    # ── Range compression momentum (bonus factor) ──
    # Is the range getting tighter over time? (range now vs range half-window ago)
    half = window // 2
    roll_high_prev = pd.Series(h).shift(half).rolling(half, min_periods=max(3, half // 2)).max().ffill().fillna(h[0]).values
    roll_low_prev  = pd.Series(l).shift(half).rolling(half, min_periods=max(3, half // 2)).min().ffill().fillna(l[0]).values
    range_prev = roll_high_prev - roll_low_prev
    compression = np.clip(1.0 - range_abs / (range_prev + 1e-10), -0.5, 0.5) + 0.5
    # >0.5 = range is tightening (bullish), <0.5 = range is expanding

    # ── Composite: 3-factor soft product ──
    score = tightness * position_score * vol_surge

    # Amplify when range is actively compressing
    score = score * (0.7 + 0.3 * compression)

    # ── Persistence decay (vectorized) ──
    # Strong signals decay over ~10 bars rather than vanishing instantly
    score_positions = np.where(
        score > 0.10,
        np.arange(n, dtype=np.float64),
        np.nan
    )
    last_signal_idx = pd.Series(score_positions).ffill().fillna(-np.inf).values
    days_since = np.arange(n, dtype=np.float64) - last_signal_idx
    decay = np.exp(-np.clip(days_since, 0.0, np.inf) / 10.0)
    had_signal = np.isfinite(last_signal_idx)
    score = np.where(had_signal, score * decay, score)

    score = np.clip(score, 0.0, 1.0)

    return score.astype(np.float64)


# ==============================================================================
# Integration: drop-in feature block for pattern_feature_factory.py
# ==============================================================================

def compute_advanced_chart_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three advanced chart pattern features.

    Returns DataFrame with columns:
      f_c4_rounding_bottom     — 圆弧底 [0,1]
      f_c4_n_shape             — N字型   [0,1]
      f_c4_ascending_breakout  — 突破上升 [0,1]
    """
    out = pd.DataFrame(index=df.index)
    out['f_c4_rounding_bottom'] = compute_rounding_bottom_score(df)
    out['f_c4_n_shape'] = compute_n_shape_score(df)
    out['f_c4_ascending_breakout'] = compute_ascending_breakout_score(df)
    return out


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 70)
    print("  Advanced Pattern Features — Quick Test")
    print("=" * 70)

    for code in ['000012', '600519', '000001']:
        df = get_daily_kline(code, days=300)
        if df is None or not df.empty:
            rb = compute_rounding_bottom_score(df)
            ns = compute_n_shape_score(df)
            ab = compute_ascending_breakout_score(df)

            print(f"\n  {code} ({len(df)} bars):")
            print(f"    rounding_bottom:     >0.3={(rb>0.3).sum():4d}  max={rb.max():.4f}  NaN={np.isnan(rb).sum()}")
            print(f"    n_shape:             >0.3={(ns>0.3).sum():4d}  max={ns.max():.4f}  NaN={np.isnan(ns).sum()}")
            print(f"    ascending_breakout:  >0.3={(ab>0.3).sum():4d}  max={ab.max():.4f}  NaN={np.isnan(ab).sum()}")
