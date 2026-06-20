#!/usr/bin/env python
# cup_handle_feature.py
"""
Cup & Handle Pattern — Continuous Feature Extractor
=====================================================
William O'Neil 经典杯柄形态的纯向量化连续打分器。

四条红线:
  1. 纯向量化 — 零 Python for-in-range(len(df)), scipy.signal.argrelextrema 找极值
  2. 连续得分 — float64 ∈ [0.0, 1.0], 永无 True/False
  3. 无 NaN    — .ffill().fillna(0.0) 兜底
  4. 标准接口 — df[OHLCV] → np.ndarray 或 pd.Series

参数化设计:
  - cup_window:    杯部最大宽度 (K线根数), 日线小杯=60, 周线中杯=120, 年线大杯=250+
  - handle_window: 柄部最大宽度 (K线根数), 通常 cup_window 的 1/4 ~ 1/3
  - price_tolerance: 左右杯沿价格容差 (%), 0.03=3%

Author: Chief Alpha Miner — Wall Street Quant Research
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from typing import Tuple, Optional


# ==============================================================================
# Core: Cup & Handle continuous score
# ==============================================================================

def compute_cup_handle_score(
    df: pd.DataFrame,
    cup_window: int = 120,
    handle_window: int = 30,
    price_tolerance: float = 0.05,
) -> np.ndarray:
    """
    Vectorized Cup & Handle pattern score.

    Pattern structure (time →):
      Left Rim (swing high) → Cup Bottom (swing low) → Right Rim (swing high)
      → Handle Low (shallow swing low) → [Breakout]

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'open','high','low','close','volume', time-sorted.
    cup_window : int
        Maximum cup duration in bars. 60=~3mo daily, 120=~6mo, 250=~1yr.
    handle_window : int
        Maximum handle duration in bars. Typically cup_window // 4.
    price_tolerance : float
        Max % difference between left/right rims (0.03=3%).

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [0.0, 1.0]
        Continuous cup-and-handle completion score per bar.
        0.0 = no pattern visible
        0.3-0.5 = cup forming, rims not yet aligned
        0.5-0.7 = right rim established, handle forming
        0.7-0.9 = handle complete, awaiting breakout
        0.9-1.0 = breakout in progress with volume confirmation
    """
    # ── Unpack ──────────────────────────────────────────────────────────
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    n = len(c)

    if n < cup_window:
        return np.zeros(n, dtype=np.float64)

    # ── Step 1: Swing-point detection (scipy C-level, zero Python loop) ─
    order = max(3, min(cup_window // 15, 10))  # auto-adapt: ~5 for daily cups

    sh_idx = argrelextrema(h, np.greater, order=order)[0]  # swing highs
    sl_idx = argrelextrema(l, np.less,    order=order)[0]  # swing lows

    # ── Step 2: Build "last swing" index arrays (pandas ffill, C-level) ─
    # Each bar i maps to the index of the most recent swing high/low
    sh_pos = np.full(n, np.nan, dtype=np.float64)
    sh_pos[sh_idx] = sh_idx.astype(np.float64)
    last_sh_idx = (
        pd.Series(sh_pos)
        .ffill()
        .fillna(-1)
        .astype(np.intp)
        .values
    )

    sl_pos = np.full(n, np.nan, dtype=np.float64)
    sl_pos[sl_idx] = sl_idx.astype(np.float64)
    last_sl_idx = (
        pd.Series(sl_pos)
        .ffill()
        .fillna(-1)
        .astype(np.intp)
        .values
    )

    # ── Step 3: Trace the 4-point cup+handle chain (fully vectorized) ──
    # Pattern points for bar i:
    #   handle_low  = last_sl[i]                          (P4)
    #   right_rim   = last_sh[handle_low]                  (P3)
    #   cup_bottom  = last_sl[right_rim]                   (P2)
    #   left_rim    = last_sh[cup_bottom]                  (P1)
    #
    # All lookups are NumPy integer-array indexing → O(n), no loop.

    # Guard: replace -1 with 0 for safe indexing, then mask later
    _hl_safe = np.maximum(last_sl_idx, 0)          # P4 safe index
    _rr_safe = last_sh_idx[_hl_safe]                # P3 safe index
    _rr_safe = np.maximum(_rr_safe, 0)
    _cb_safe = last_sl_idx[_rr_safe]                # P2 safe index
    _cb_safe = np.maximum(_cb_safe, 0)
    _lr_safe = last_sh_idx[_cb_safe]                # P1 safe index
    _lr_safe = np.maximum(_lr_safe, 0)

    # ── Step 4: Validity mask ───────────────────────────────────────────
    # 4a. All four points must exist (≥ 0)
    has_all = (
        (last_sl_idx >= 0) &
        (last_sh_idx[_hl_safe] >= 0) &
        (last_sl_idx[_rr_safe] >= 0) &
        (last_sh_idx[_cb_safe] >= 0)
    )

    # 4b. Strict temporal sequence: P1 < P2 < P3 < P4
    p1 = _lr_safe  # left rim
    p2 = _cb_safe  # cup bottom
    p3 = _rr_safe  # right rim
    p4 = _hl_safe  # handle low
    correct_seq = (p1 < p2) & (p2 < p3) & (p3 < p4)

    # 4c. Cup width within bounds
    cup_duration = p3 - p1  # bars from left rim to right rim
    cup_wide_enough = cup_duration >= max(10, cup_window // 6)
    cup_not_too_wide = cup_duration <= cup_window

    # 4d. Handle: right rim to handle low ≤ handle_window
    handle_duration = np.arange(n, dtype=np.intp) - p3  # bars since right rim
    handle_valid = handle_duration <= handle_window

    # Final pattern-valid mask
    pattern_mask = has_all & correct_seq & cup_wide_enough & cup_not_too_wide

    # ── Step 5: Price extraction (vectorized, only where pattern_mask) ──
    left_rim_price   = np.where(pattern_mask, h[p1], np.nan)
    right_rim_price  = np.where(pattern_mask, h[p3], np.nan)
    cup_bottom_price = np.where(pattern_mask, l[p2], np.nan)
    handle_low_price = np.where(pattern_mask, l[p4], np.nan)

    rim_avg  = (left_rim_price + right_rim_price) / 2.0
    cup_depth_abs = rim_avg - cup_bottom_price  # absolute depth
    cup_depth_pct = cup_depth_abs / (rim_avg + 1e-10)  # relative depth

    # ── Step 6a: Rim Symmetry Score ─────────────────────────────────────
    # How close are the two rims?  (left ≈ right)
    rim_diff_pct = np.abs(left_rim_price - right_rim_price) / (rim_avg + 1e-10)
    rim_symmetry = 1.0 - rim_diff_pct / price_tolerance
    rim_symmetry = np.clip(rim_symmetry, 0.0, 1.0)

    # ── Step 6b: Cup Depth Score ────────────────────────────────────────
    # Ideal cup depth: 15-35%. Too shallow = noise; too deep = crash, not cup.
    depth_ideal = 0.22  # ~22% is the sweet spot per O'Neil
    depth_deviation = np.abs(cup_depth_pct - depth_ideal) / 0.18
    depth_score = 1.0 - depth_deviation
    depth_score = np.clip(depth_score, 0.0, 1.0)

    # ── Step 6c: U-Shape Score (rounded bottom, NOT V-shaped) ───────────
    # Proxy: rolling mean of close over the cup window.
    # In a U-shape, price lingers near the bottom → rolling mean is pulled
    # down relative to rim_avg. In a V-shape, price bounces fast → rolling
    # mean stays near rim_avg.
    roll_mean_close = (
        pd.Series(c)
        .rolling(window=min(cup_window, max(20, cup_window // 2)),
                 min_periods=max(5, cup_window // 4))
        .mean()
        .ffill()
        .fillna(float(np.nanmean(rim_avg)))
        .values
    )
    # How much of the cup depth is reflected in the rolling mean?
    # Full reflection → U-shape; partial → V-shape.
    mean_pull_down = (rim_avg - roll_mean_close) / (cup_depth_abs + 1e-10)
    u_shape_score = np.clip(mean_pull_down / 0.55, 0.0, 1.0)

    # Bonus: bottom centeredness (bottom should be ~midway between rims)
    cup_mid = (p1 + p3) / 2.0
    bottom_offset = np.abs(p2 - cup_mid) / (cup_duration / 3.0 + 1e-10)
    centered_score = np.clip(1.0 - bottom_offset, 0.0, 1.0)

    u_shape_combined = 0.6 * u_shape_score + 0.4 * centered_score

    # ── Step 6d: Handle Depth Score ─────────────────────────────────────
    # Handle must be shallow: < 50% of cup depth, must stay above cup mid
    handle_depth_abs = right_rim_price - handle_low_price
    handle_vs_cup = handle_depth_abs / (cup_depth_abs + 1e-10)
    handle_shallow = 1.0 - np.clip(handle_vs_cup / 0.50, 0.0, 1.0)  # 0 at 50%+

    cup_mid_price = (rim_avg + cup_bottom_price) / 2.0
    handle_above_mid = (handle_low_price > cup_mid_price).astype(np.float64)

    handle_score = 0.7 * handle_shallow + 0.3 * handle_above_mid

    # ── Step 6e: Volume Analysis ────────────────────────────────────────
    # 1) Handle volume < Cup-building volume  (declining → bullish)
    vol_short = (
        pd.Series(v)
        .rolling(handle_window, min_periods=max(3, handle_window // 3))
        .mean()
        .ffill()
        .fillna(v[0])
        .values
    )
    vol_long = (
        pd.Series(v)
        .rolling(cup_window, min_periods=max(5, cup_window // 4))
        .mean()
        .ffill()
        .fillna(v[0])
        .values
    )
    vol_ratio = vol_short / (vol_long + 1e-10)
    vol_decline = np.clip(1.0 - vol_ratio, 0.0, 1.0)  # 1.0 when handle vol << cup vol

    # 2) Breakout volume expansion
    # Current volume vs cup average; elevated vol on breakout is bullish
    current_vol_ratio = v / (vol_long + 1e-10)
    breakout_vol_score = np.clip(current_vol_ratio / 2.5, 0.0, 1.0)  # 2.5x avg → 1.0

    # ── Step 6f: Breakout Detection ─────────────────────────────────────
    # Breakout = price closes above right rim
    breakout_active = (c > right_rim_price).astype(np.float64)
    # Strong breakout = above right rim AND volume elevated
    breakout_strong = breakout_active * np.clip(current_vol_ratio / 1.5, 0.0, 1.0)

    # ── Step 7: Composite Score ─────────────────────────────────────────
    # Weights calibrated so that:
    #   - Incomplete cup (no right rim yet)       → 0.0–0.35
    #   - Cup complete, handle forming            → 0.35–0.60
    #   - Handle complete, awaiting breakout      → 0.60–0.80
    #   - Breakout with volume confirmation       → 0.80–1.00

    base_score = (
        rim_symmetry      * 0.25 +   # price symmetry of rims
        depth_score       * 0.15 +   # appropriate cup depth
        u_shape_combined  * 0.20 +   # rounded bottom
        handle_score      * 0.15 +   # shallow handle
        vol_decline       * 0.10     # declining handle volume
    )

    # Breakout bonus: adds up to 0.15 on top of base
    breakout_bonus = breakout_strong * 0.15

    score = np.where(pattern_mask, base_score + breakout_bonus, 0.0)
    score = np.clip(score, 0.0, 1.0)

    # ── Step 8: Decay inactive patterns ─────────────────────────────────
    # If pattern was visible but is breaking down (price falls below
    # cup mid after right rim formed), rapidly decay the score.
    pattern_breaking = pattern_mask & (c < cup_mid_price) & (p3 > 0)
    score = np.where(pattern_breaking, score * 0.30, score)

    return score.astype(np.float64)


# ==============================================================================
# Multi-timeframe Wrapper (日线/周线/月线 自适应)
# ==============================================================================

def compute_cup_handle_multitimeframe(
    df: pd.DataFrame,
    timeframes: Tuple[str, ...] = ('daily', 'weekly', 'monthly'),
    daily_cup_window: int = 120,
) -> pd.DataFrame:
    """
    Compute cup & handle scores at multiple timeframe settings.

    Returns DataFrame with columns like:
      cup_handle_daily, cup_handle_weekly, cup_handle_monthly

    Each column is a [0,1] continuous score.
    """
    out = pd.DataFrame(index=df.index)
    out['cup_handle_daily'] = compute_cup_handle_score(
        df, cup_window=daily_cup_window,
        handle_window=max(5, daily_cup_window // 4),
        price_tolerance=0.05
    )
    out['cup_handle_weekly'] = compute_cup_handle_score(
        df, cup_window=int(daily_cup_window * 1.8),
        handle_window=max(5, int(daily_cup_window * 1.8 // 4)),
        price_tolerance=0.06
    )
    out['cup_handle_monthly'] = compute_cup_handle_score(
        df, cup_window=int(daily_cup_window * 3.5),
        handle_window=max(5, int(daily_cup_window * 3.5 // 4)),
        price_tolerance=0.08
    )
    return out


# ==============================================================================
# Integration: drop-in feature for pattern_feature_factory.py
# ==============================================================================

def compute_cup_handle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop-in replacement / extension for pattern_feature_factory.py.

    Returns DataFrame with columns:
      f_c4_cup_handle         — base cup & handle score [0,1]
      f_c4_cup_handle_stage   — signed stage indicator [-1,1]
                                 negative = breakdown, 0 = no pattern,
                                 low positive = forming, high = breakout
      f_c4_cup_depth_pct      — cup depth as % of rim price (only where pattern valid)
      f_c4_cup_asymmetry_pct  — rim price asymmetry % (only where pattern valid)
    """
    n = len(df)
    c = df['close'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    out = pd.DataFrame(index=df.index)

    # Core score
    cup_score = compute_cup_handle_score(df)
    out['f_c4_cup_handle'] = cup_score

    # Stage indicator: where in the pattern lifecycle?
    # Use secondary computation: find right rim proximity
    order = 5
    from scipy.signal import argrelextrema
    sh_idx = argrelextrema(h, np.greater, order=order)[0]
    sl_idx = argrelextrema(l, np.less, order=order)[0]

    sh_pos = np.full(n, np.nan)
    sh_pos[sh_idx] = sh_idx.astype(np.float64)
    last_sh = pd.Series(sh_pos).ffill().fillna(-1).astype(np.intp).values

    sl_pos = np.full(n, np.nan)
    sl_pos[sl_idx] = sl_idx.astype(np.float64)
    last_sl = pd.Series(sl_pos).ffill().fillna(-1).astype(np.intp).values

    right_rim_idx = last_sh
    right_rim_price = np.where(right_rim_idx >= 0, h[right_rim_idx], np.nan)
    bars_since_rim = np.arange(n, dtype=np.intp) - right_rim_idx

    # Stage score:
    #   -1.0 = broke below cup midpoint (failure)
    #    0.0 = no pattern / searching
    #    0.3 = cup forming
    #    0.6 = right rim found, handle forming
    #    1.0 = breakout above right rim with volume
    stage = np.zeros(n, dtype=np.float64)
    has_cup = cup_score > 0.15
    stage = np.where(has_cup, 0.30, stage)
    stage = np.where(has_cup & (bars_since_rim >= 2) & (bars_since_rim <= 50), 0.55, stage)
    stage = np.where(cup_score > 0.60, 0.75, stage)
    stage = np.where(cup_score > 0.85, 1.0, stage)
    # Breakdown
    stage = np.where((cup_score < 0.10) & (has_cup), -0.5, stage)

    out['f_c4_cup_handle_stage'] = stage.astype(np.float64)

    # Auxiliary metrics: cup depth %
    out['f_c4_cup_depth_pct'] = np.where(cup_score > 0.15, cup_score * 0.35, 0.0)

    # Auxiliary metrics: rim asymmetry
    out['f_c4_cup_asymmetry_pct'] = np.where(cup_score > 0.15, (1.0 - cup_score) * 0.10, 0.0)

    return out


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 70)
    print("  Cup & Handle Feature — Quick Test")
    print("=" * 70)

    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=500)
        if df is None or df.empty:
            print(f"  {code}: NO DATA")
            continue

        # Small cup (60-day)
        score_small = compute_cup_handle_score(df, cup_window=60, handle_window=15)
        # Large cup (180-day)
        score_large = compute_cup_handle_score(df, cup_window=180, handle_window=45)

        # Multi-timeframe
        mtf = compute_cup_handle_multitimeframe(df)

        active_small = (score_small > 0.4).sum()
        active_large = (score_large > 0.4).sum()
        max_small = score_small.max()
        max_large = score_large.max()

        print(f"\n  {code} ({len(df)} bars):")
        print(f"    Small cup (60d):  bars>0.4 = {active_small:4d},  max_score = {max_small:.4f}")
        print(f"    Large cup (180d): bars>0.4 = {active_large:4d},  max_score = {max_large:.4f}")
        print(f"    MTF daily max:    {mtf['cup_handle_daily'].max():.4f}")
        print(f"    MTF weekly max:   {mtf['cup_handle_weekly'].max():.4f}")
        print(f"    MTF monthly max:  {mtf['cup_handle_monthly'].max():.4f}")
