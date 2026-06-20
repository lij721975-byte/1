#!/usr/bin/env python
# pattern_feature_factory.py
"""
Classical Pattern Feature Factory — 经典形态连续特征因子库
================================================================
将抽象 K 线形态转化为 ML-ready 连续特征值 (tree-model friendly)。

Design principles:
  A. Every pattern → continuous score ∈ [-1,1] or [0,1], never binary
  B. Fully vectorized (numpy/pandas) — zero Python loops over bars in core compute
  C. "Degree of pattern-ness" — fuzzy matching via ratio scores, not exact templates
  D. Multi-timeframe: each family rolled at 5/10/20/50-bar windows
  E. Tree-model friendly: monotonic, normalized, decorrelated across families

Pattern families:
  I.   Single-candle   (8 features)  — body/shadows geometry
  II.  Double-candle   (10 features) — 2-bar relationships
  III. Triple-candle   (8 features)  — 3-bar reversal/continuation
  IV.  Chart patterns  (10 features) — swing-based geometric structures
  V.   Composite       (6 features)  — derived aggregators

Output: pd.DataFrame with same index as input, ~80+ continuous feature columns.
All values are float64, no NaN, zero-filled where insufficient data.

Integration:
  from pattern_feature_factory import compute_pattern_features
  features_df = compute_pattern_features(df)

Author: Chief Alpha Miner — Wall Street Quant Research
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from collections import deque
from scipy.signal import argrelextrema
from cup_handle_feature import compute_cup_handle_score
from advanced_pattern_features import (
    compute_rounding_bottom_score,
    compute_n_shape_score,
    compute_ascending_breakout_score,
)
from rectangle_range_feature import compute_rectangle_range_score
from advanced_reversal_patterns import (
    compute_island_reversal_score,
    compute_diamond_score,
)


# ==============================================================================
# 0. Utility helpers
# ==============================================================================

def _safe_div(a: np.ndarray, b: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Vectorized safe division: a/b, with fill where b ≈ 0."""
    result = np.full_like(a, fill, dtype=np.float64)
    mask = np.abs(b) > 1e-10
    result[mask] = a[mask] / b[mask]
    return result


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _clip_sym(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0)


# ==============================================================================
# I. Single-Candle Features (单K形态 → 8 连续特征)
# ==============================================================================

def compute_single_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-candle geometry features.
    All operations are vectorized numpy — O(n).

    Features:
      f_c1_body_ratio       — body / range, signed (bullish +, bearish -)
      f_c1_doji              — 1 - body/range, continuous [0,1]
      f_c1_marubozu          — body/range, continuous [0,1] (inverse of doji)
      f_c1_hammer            — hammer-like score [0,1], trend-aware
      f_c1_shooting_star     — shooting-star-like score [0,1], trend-aware
      f_c1_lower_shadow_pct  — lower shadow / range [0,1]
      f_c1_upper_shadow_pct  — upper shadow / range [0,1]
      f_c1_body_center       — where the body sits within the range [0,1] (0=at low, 1=at high)
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(o)

    body = np.abs(c - o)
    body_signed = c - o
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l
    candle_range = h - l

    out = pd.DataFrame(index=df.index, dtype=np.float64)

    # --- f_c1_body_ratio: signed body / range [-1, 1] ---
    body_ratio = _safe_div(body, candle_range)
    out['f_c1_body_ratio'] = np.sign(body_signed) * body_ratio

    # --- f_c1_doji: 1 - body_ratio, [0,1] ---
    out['f_c1_doji'] = _clip01(1.0 - body_ratio)

    # --- f_c1_marubozu: body_ratio, [0,1] ---
    out['f_c1_marubozu'] = _clip01(body_ratio)

    # --- f_c1_lower_shadow_pct ---
    out['f_c1_lower_shadow_pct'] = _safe_div(lower_shadow, candle_range)

    # --- f_c1_upper_shadow_pct ---
    out['f_c1_upper_shadow_pct'] = _safe_div(upper_shadow, candle_range)

    # --- f_c1_body_center: where body sits in the range ---
    body_mid = (np.minimum(o, c) + np.maximum(o, c)) / 2.0
    out['f_c1_body_center'] = _clip01(_safe_div(body_mid - l, candle_range, fill=0.5))

    # --- f_c1_hammer: long lower shadow, small body, minimal upper shadow, in downtrend ---
    # Hammer score = lower_shadow_ratio * (1-body_ratio) * (1-upper_shadow_ratio) * downtrend_factor
    lower_ratio = out['f_c1_lower_shadow_pct'].values
    upper_ratio = out['f_c1_upper_shadow_pct'].values
    hammer_raw = lower_ratio * (1.0 - body_ratio) * (1.0 - upper_ratio)

    # Downtrend factor: price below MA10, or recent negative return
    closes = pd.Series(c, index=df.index)
    ma10 = closes.rolling(10, min_periods=5).mean().ffill().fillna(c[0]).values
    ret5 = np.zeros(n, dtype=np.float64)
    ret5[5:] = (c[5:] - c[:-5]) / (np.abs(c[:-5]) + 1e-10)
    downtrend = np.clip(-ret5 / 0.10, 0.0, 1.0)  # -10% ret → 1.0 downtrend

    # Also price < MA10
    below_ma = (c < ma10).astype(np.float64)

    trend_factor = 0.5 * downtrend + 0.5 * below_ma
    out['f_c1_hammer'] = _clip01(hammer_raw * trend_factor * 2.0)  # scale up since trend_factor ≤ 1

    # --- f_c1_shooting_star: long upper shadow, small body, minimal lower shadow, in uptrend ---
    star_raw = upper_ratio * (1.0 - body_ratio) * (1.0 - lower_ratio)
    uptrend = np.clip(ret5 / 0.10, 0.0, 1.0)
    above_ma = (c > ma10).astype(np.float64)
    up_factor = 0.5 * uptrend + 0.5 * above_ma
    out['f_c1_shooting_star'] = _clip01(star_raw * up_factor * 2.0)

    return out


# ==============================================================================
# II. Double-Candle Features (双K形态 → 10 连续特征)
# ==============================================================================

def compute_double_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Two-bar relationship features. All vectorized.

    Features:
      f_c2_engulfing_bull    — bullish engulfing strength [0,1]
      f_c2_engulfing_bear    — bearish engulfing strength [0,1]
      f_c2_harami_bull       — bullish harami (inside bar after bearish) [0,1]
      f_c2_harami_bear       — bearish harami (inside bar after bullish) [0,1]
      f_c2_piercing          — piercing line strength [0,1]
      f_c2_dark_cloud_cover  — dark cloud cover strength [0,1]
      f_c2_gap_up            — gap-up magnitude / ATR [0,1]
      f_c2_gap_down          — gap-down magnitude / ATR [0,1]
      f_c2_tweezer_top       — two similar highs [0,1]
      f_c2_tweezer_bottom    — two similar lows [0,1]
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(o)

    body_curr = np.abs(c - o)
    body_prev = np.roll(body_curr, 1)
    body_prev[0] = 0

    range_curr = h - l
    atr14 = pd.Series(range_curr).rolling(14, min_periods=5).mean().ffill().fillna(range_curr[0]).values

    out = pd.DataFrame(index=df.index, dtype=np.float64)

    # Shifted values (t-1)
    o_prev = np.roll(o, 1); o_prev[0] = o[0]
    c_prev = np.roll(c, 1); c_prev[0] = c[0]
    h_prev = np.roll(h, 1); h_prev[0] = h[0]
    l_prev = np.roll(l, 1); l_prev[0] = l[0]

    body_s_curr = c - o         # signed body current
    body_s_prev = c_prev - o_prev  # signed body previous

    # --- f_c2_engulfing_bull: current bullish body engulfs prior bearish body ---
    # Conditions: prev bearish, curr bullish, curr_open < prev_close, curr_close > prev_open
    prev_bearish = body_s_prev < 0
    curr_bullish = body_s_curr > 0
    engulfs_down = o < c_prev   # opened below prior close
    engulfs_up = c > o_prev     # closed above prior open
    engulf_size = _safe_div(body_curr, body_prev + 1e-10)
    engulf_size_clipped = np.clip(engulf_size, 0.0, 3.0) / 3.0  # normalize, cap at 3x

    bull_engulf_raw = (
        prev_bearish.astype(np.float64) *
        curr_bullish.astype(np.float64) *
        engulfs_down.astype(np.float64) *
        engulfs_up.astype(np.float64) *
        engulf_size_clipped
    )
    out['f_c2_engulfing_bull'] = _clip01(bull_engulf_raw)

    # --- f_c2_engulfing_bear: current bearish body engulfs prior bullish body ---
    prev_bullish = body_s_prev > 0
    curr_bearish = body_s_curr < 0
    engulfs_up2 = o > c_prev
    engulfs_down2 = c < o_prev
    bear_engulf_raw = (
        prev_bullish.astype(np.float64) *
        curr_bearish.astype(np.float64) *
        engulfs_up2.astype(np.float64) *
        engulfs_down2.astype(np.float64) *
        engulf_size_clipped
    )
    out['f_c2_engulfing_bear'] = _clip01(bear_engulf_raw)

    # --- f_c2_piercing: bullish close penetrates >50% into prior bearish body ---
    penetration = _safe_div(c - o_prev, np.abs(body_s_prev) + 1e-10)  # 0=no pen, 1=full pen
    pierce_raw = (
        prev_bearish.astype(np.float64) *
        curr_bullish.astype(np.float64) *
        (o < l_prev).astype(np.float64) *       # opened below prior low
        np.clip(penetration, 0.0, 1.0)            # penetration depth
    )
    out['f_c2_piercing'] = _clip01(pierce_raw)

    # --- f_c2_dark_cloud_cover: bearish close penetrates >50% into prior bullish body ---
    dark_penetration = _safe_div(o_prev - c, np.abs(body_s_prev) + 1e-10)
    dark_raw = (
        prev_bullish.astype(np.float64) *
        curr_bearish.astype(np.float64) *
        (o > h_prev).astype(np.float64) *        # opened above prior high
        np.clip(dark_penetration, 0.0, 1.0)
    )
    out['f_c2_dark_cloud_cover'] = _clip01(dark_raw)

    # --- f_c2_harami_bull: small bullish body contained within prior large bearish body ---
    body_contraction = 1.0 - _clip01(_safe_div(body_curr, body_prev + 1e-10))
    contained = ((h <= h_prev) & (l >= l_prev)).astype(np.float64)
    harami_bull_raw = prev_bearish.astype(np.float64) * curr_bullish.astype(np.float64) * contained * body_contraction
    out['f_c2_harami_bull'] = _clip01(harami_bull_raw)

    # --- f_c2_harami_bear: small bearish body contained within prior large bullish body ---
    harami_bear_raw = prev_bullish.astype(np.float64) * curr_bearish.astype(np.float64) * contained * body_contraction
    out['f_c2_harami_bear'] = _clip01(harami_bear_raw)

    # --- f_c2_gap_up: gap up as fraction of ATR ---
    gap_up = _safe_div(o - h_prev, atr14 + 1e-10)
    out['f_c2_gap_up'] = _clip01(gap_up)

    # --- f_c2_gap_down: gap down as fraction of ATR ---
    gap_down = _safe_div(l_prev - o, atr14 + 1e-10)
    out['f_c2_gap_down'] = _clip01(gap_down)

    # --- f_c2_tweezer_top: two consecutive bars with same high (within 0.1% or 1 tick) ---
    high_diff_pct = _safe_div(np.abs(h - h_prev), h_prev + 1e-10)
    out['f_c2_tweezer_top'] = _clip01(1.0 - high_diff_pct / 0.005)  # 0.5% diff → 0, exact match → 1

    # --- f_c2_tweezer_bottom: two consecutive bars with same low ---
    low_diff_pct = _safe_div(np.abs(l - l_prev), l_prev + 1e-10)
    out['f_c2_tweezer_bottom'] = _clip01(1.0 - low_diff_pct / 0.005)

    return out


# ==============================================================================
# III. Triple-Candle Features (三K形态 → 8 连续特征)
# ==============================================================================

def compute_triple_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-bar pattern features. All vectorized.

    Features:
      f_c3_morning_star       — bullish 3-bar reversal [0,1]
      f_c3_evening_star       — bearish 3-bar reversal [0,1]
      f_c3_three_soldiers     — three advancing white soldiers [0,1]
      f_c3_three_crows        — three black crows [0,1]
      f_c3_abandoned_baby_bull — gap + doji + gap bullish reversal [0,1]
      f_c3_abandoned_baby_bear — gap + doji + gap bearish reversal [0,1]
      f_c3_squeeze            — NR3/NR4 (narrow range) → breakout precursor [0,1]
      f_c3_range_expansion    — range expansion (volatility surge) [0,1]
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(o)

    body = np.abs(c - o)
    body_signed = c - o
    candle_range = h - l

    out = pd.DataFrame(index=df.index, dtype=np.float64)

    # Index at t-1, t-2
    # (we use shift-by-2 indexing: for bar i, the pattern completes at bar i with bars i-2, i-1, i)

    # --- Helper: get values at relative offsets ---
    def _at(arr, offset):
        """Get arr shifted by offset (negative = lag). First |offset| entries filled with first value."""
        if offset == 0:
            return arr
        if offset < 0:
            result = np.empty_like(arr)
            k = -offset
            result[:k] = arr[0]
            result[k:] = arr[:-k]
            return result
        else:
            result = np.empty_like(arr)
            result[:-offset] = arr[offset:]
            result[-offset:] = arr[-1]
            return result

    # t-2, t-1, t
    o2, o1, o0 = _at(o, -2), _at(o, -1), o
    c2, c1, c0 = _at(c, -2), _at(c, -1), c
    h2, h1, h0 = _at(h, -2), _at(h, -1), h
    l2, l1, l0 = _at(l, -2), _at(l, -1), l
    b2, b1, b0 = _at(body, -2), _at(body, -1), body
    r2, r1, r0 = _at(candle_range, -2), _at(candle_range, -1), candle_range
    bs2, bs1, bs0 = _at(body_signed, -2), _at(body_signed, -1), body_signed

    # --- f_c3_morning_star: bearish(t-2) + small body(t-1) + bullish(t-0) + close > midpoint(t-2) ---
    midpoint_t2 = (o2 + c2) / 2.0
    small_body_t1 = 1.0 - _clip01(_safe_div(b1, r1 + 1e-10))  # near-doji at t-1
    ms_raw = (
        (bs2 < 0).astype(np.float64) *      # t-2 bearish
        small_body_t1 *                      # t-1 small body
        (bs0 > 0).astype(np.float64) *      # t-0 bullish
        _clip01(_safe_div(c0 - midpoint_t2, np.abs(bs2) + 1e-10))  # close penetration
    )
    out['f_c3_morning_star'] = _clip01(ms_raw)

    # --- f_c3_evening_star: bullish(t-2) + small body(t-1) + bearish(t-0) + close < midpoint(t-2) ---
    es_raw = (
        (bs2 > 0).astype(np.float64) *
        small_body_t1 *
        (bs0 < 0).astype(np.float64) *
        _clip01(_safe_div(midpoint_t2 - c0, np.abs(bs2) + 1e-10))
    )
    out['f_c3_evening_star'] = _clip01(es_raw)

    # --- f_c3_three_soldiers: three consecutive bullish, each close > prior close ---
    three_bull = (
        (bs2 > 0).astype(np.float64) *
        (bs1 > 0).astype(np.float64) *
        (bs0 > 0).astype(np.float64)
    )
    rising = (
        (c1 > c2).astype(np.float64) *
        (c0 > c1).astype(np.float64)
    )
    big_bodies = _clip01(_safe_div(b2, r2 + 1e-10) * _safe_div(b1, r1 + 1e-10) * _safe_div(b0, r0 + 1e-10) * 3.0)
    out['f_c3_three_soldiers'] = _clip01(three_bull * rising * big_bodies)

    # --- f_c3_three_crows: three consecutive bearish, each close < prior close ---
    three_bear = (
        (bs2 < 0).astype(np.float64) *
        (bs1 < 0).astype(np.float64) *
        (bs0 < 0).astype(np.float64)
    )
    falling = (
        (c1 < c2).astype(np.float64) *
        (c0 < c1).astype(np.float64)
    )
    out['f_c3_three_crows'] = _clip01(three_bear * falling * big_bodies)

    # --- f_c3_abandoned_baby_bull: gap down + doji + gap up reversal ---
    doji_t1 = 1.0 - _clip01(_safe_div(b1, r1 + 1e-10))
    gap_down_before = _safe_div(l1 - h2, r2 + 1e-10)   # t-1 low < t-2 high → gap down
    gap_up_after = _safe_div(o0 - h1, r1 + 1e-10)       # t-0 open > t-1 high → gap up
    ab_bull = (
        (bs2 < 0).astype(np.float64) *
        doji_t1 *
        (bs0 > 0).astype(np.float64) *
        _clip01(gap_down_before) *
        _clip01(gap_up_after)
    )
    out['f_c3_abandoned_baby_bull'] = _clip01(ab_bull)

    # --- f_c3_abandoned_baby_bear: gap up + doji + gap down reversal ---
    gap_up_before = _safe_div(h1 - l2, r2 + 1e-10)
    gap_down_after = _safe_div(l0 - h1, r1 + 1e-10)
    ab_bear = (
        (bs2 > 0).astype(np.float64) *
        doji_t1 *
        (bs0 < 0).astype(np.float64) *
        _clip01(gap_up_before) *
        _clip01(gap_down_after)
    )
    out['f_c3_abandoned_baby_bear'] = _clip01(ab_bear)

    # --- f_c3_squeeze: NR3 (narrowest range in 3 bars) → precursor to expansion ---
    r_min_3 = np.minimum(np.minimum(r2, r1), r0)
    r_max_3 = np.maximum(np.maximum(r2, r1), r0)
    is_nr3 = (r0 <= r_min_3 * 1.01).astype(np.float64)  # current bar has narrowest range
    squeeze_intensity = 1.0 - _clip01(_safe_div(r0, r_max_3 + 1e-10))
    out['f_c3_squeeze'] = _clip01(is_nr3 * squeeze_intensity)

    # --- f_c3_range_expansion: widest range in 3 bars ---
    is_wide = (r0 >= r_max_3 * 0.99).astype(np.float64)
    expansion_intensity = _clip01(_safe_div(r0, r_min_3 + 1e-10) / 3.0)  # 3x min → 1.0
    out['f_c3_range_expansion'] = _clip01(is_wide * expansion_intensity)

    return out


# ==============================================================================
# IV. Chart Pattern Features (图表形态 → 10 连续特征)
# ==============================================================================

def _find_swing_points(highs: np.ndarray, lows: np.ndarray, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find swing highs and swing lows using scipy argrelextrema (C-level, zero Python loop).
    Returns (is_swing_high, is_swing_low) boolean arrays.
    """
    n = len(highs)
    is_sh = np.zeros(n, dtype=bool)
    is_sl = np.zeros(n, dtype=bool)

    if n < 2 * order + 1:
        return is_sh, is_sl

    sh = argrelextrema(highs, np.greater, order=order)[0]
    sl = argrelextrema(lows, np.less, order=order)[0]
    is_sh[sh] = True
    is_sl[sl] = True

    return is_sh, is_sl


def compute_chart_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fully vectorized chart pattern features using rolling-boundary analysis.

    Key design change: NO Python for-loop over bars.
    - Double top/bottom & H&S: precomputed at swing points, forward-filled
    - Triangles & wedges: rolling-max/min boundary slopes, O(N) vectorized
    - Flags: rolling momentum + range compression, O(N) vectorized

    Features:
      f_c4_double_top          — two-peak similarity [0,1]
      f_c4_double_bottom       — two-trough similarity [0,1]
      f_c4_head_shoulders_top  — center peak + lower flanking peaks [0,1]
      f_c4_head_shoulders_bot  — center trough + higher flanking troughs [0,1]
      f_c4_ascending_triangle  — flat resistance + rising support [0,1]
      f_c4_descending_triangle — flat support + falling resistance [0,1]
      f_c4_symmetrical_triangle — converging highs and lows [0,1]
      f_c4_bull_flag           — sharp up move + shallow pullback channel [0,1]
      f_c4_bear_flag           — sharp down move + shallow bounce channel [0,1]
      f_c4_wedge_rising        — converging rising trendlines (bearish reversal) [0,1]
      f_c4_wedge_falling       — converging falling trendlines (bullish reversal) [0,1]
    """
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(c)

    out = pd.DataFrame(index=df.index, dtype=np.float64)

    # ── Shared precomputations (O(N), fully vectorized) ──
    atr = (
        pd.Series(h - l)
        .rolling(14, min_periods=5).mean()
        .ffill().fillna(1.0).values
    )

    # ────────────────────────────────────────────────────────────────
    # A. Double Top / Double Bottom / Head & Shoulders
    #    Precompute at swing points, forward-fill to all bars.
    #    Loop only over swing points (~10-50), not over bars (~300+).
    # ────────────────────────────────────────────────────────────────
    is_sh, is_sl = _find_swing_points(h, l, order=3)
    sh_idx = np.where(is_sh)[0]
    sl_idx = np.where(is_sl)[0]

    # Accumulators: use np.maximum (not +=) to prevent saturation from many swing points
    double_top = np.zeros(n, dtype=np.float64)
    double_bot = np.zeros(n, dtype=np.float64)
    hs_top = np.zeros(n, dtype=np.float64)
    hs_bot = np.zeros(n, dtype=np.float64)
    halflife = 5.0  # short half-life — fast decay prevents swing-point overload
    decay_rate = np.log(2.0) / halflife
    score_min = 0.25

    def _add_decaying(arr, i_event, score_val):
        days = np.arange(n, dtype=np.float64) - i_event
        days = np.maximum(days, 0.0)
        contrib = score_val * np.exp(-days * decay_rate)
        np.maximum(arr, contrib, out=arr)

    # Double Top
    for k in range(1, len(sh_idx)):
        i_curr, i_prev = sh_idx[k], sh_idx[k - 1]
        diff = abs(h[i_curr] - h[i_prev]) / (atr[i_curr] + 1e-10)
        score = float(_clip01(1.0 - diff / 2.0))
        if score > score_min:
            _add_decaying(double_top, i_curr, score)

    # Double Bottom
    for k in range(1, len(sl_idx)):
        i_curr, i_prev = sl_idx[k], sl_idx[k - 1]
        diff = abs(l[i_curr] - l[i_prev]) / (atr[i_curr] + 1e-10)
        score = float(_clip01(1.0 - diff / 2.0))
        if score > score_min:
            _add_decaying(double_bot, i_curr, score)

    # Head & Shoulders Top
    for k in range(2, len(sh_idx)):
        i_l, i_c, i_r = sh_idx[k - 2], sh_idx[k - 1], sh_idx[k]
        h_l, h_c, h_r = h[i_l], h[i_c], h[i_r]
        if h_c > h_l and h_c > h_r:
            neckline = (l[i_l:i_c].min() + l[i_c:i_r].min()) / 2.0
            head_h = (h_c - neckline) / (atr[i_r] + 1e-10)
            sym = 1.0 - abs((h_c - h_l) - (h_c - h_r)) / (atr[i_r] + 1e-10 + abs(h_c - h_l))
            score = float(_clip01(min(head_h / 2.0, 1.0) * _clip01(sym)))
            if score > score_min:
                _add_decaying(hs_top, i_r, score)

    # Inverse Head & Shoulders
    for k in range(2, len(sl_idx)):
        i_l, i_c, i_r = sl_idx[k - 2], sl_idx[k - 1], sl_idx[k]
        l_l, l_c, l_r = l[i_l], l[i_c], l[i_r]
        if l_c < l_l and l_c < l_r:
            neckline = (h[i_l:i_c].max() + h[i_c:i_r].max()) / 2.0
            head_d = (neckline - l_c) / (atr[i_r] + 1e-10)
            sym = 1.0 - abs((l_l - l_c) - (l_r - l_c)) / (atr[i_r] + 1e-10 + abs(l_l - l_c))
            score = float(_clip01(min(head_d / 2.0, 1.0) * _clip01(sym)))
            if score > score_min:
                _add_decaying(hs_bot, i_r, score)

    # Already clipped: np.maximum keeps values ≤ each contribution's score_val ≤ 1.0

    out['f_c4_double_top'] = double_top
    out['f_c4_double_bottom'] = double_bot
    out['f_c4_head_shoulders_top'] = hs_top
    out['f_c4_head_shoulders_bot'] = hs_bot

    # ────────────────────────────────────────────────────────────────
    # B. Triangles & Wedges — pure rolling-boundary slope (O(N))
    #    Key insight: use rolling max of highs / rolling min of lows
    #    as proxies for the upper/lower boundary trendlines.
    #    Slope = (endpoint - startpoint) / duration  — fully vectorized.
    # ────────────────────────────────────────────────────────────────
    W = 40  # triangle lookback window

    # Rolling boundaries: max high and min low over lookback
    roll_h_max = pd.Series(h).rolling(W, min_periods=W // 2).max().ffill().fillna(h[0]).values
    roll_l_min = pd.Series(l).rolling(W, min_periods=W // 2).min().ffill().fillna(l[0]).values

    # Boundary slopes: (now - W/2 ago) / (W/2)  →  normalized by current price
    half = W // 2
    h_slope = (roll_h_max - pd.Series(roll_h_max).shift(half).ffill().fillna(roll_h_max[0]).values) / half
    l_slope = (roll_l_min - pd.Series(roll_l_min).shift(half).ffill().fillna(roll_l_min[0]).values) / half

    # Normalize slopes by average price (make comparable across stocks)
    avg_price = pd.Series(c).rolling(W, min_periods=W // 2).mean().ffill().fillna(c[0]).values
    h_slope_norm = h_slope / (avg_price + 1e-10)
    l_slope_norm = l_slope / (avg_price + 1e-10)

    # --- Ascending Triangle: flat highs + rising lows ---
    # Exponential mapping:  slope=0 → 1.0,  slope=0.0002 → 0.37,  slope=0.0005 → 0.08
    high_flatness = np.exp(-np.abs(h_slope_norm) * 15000.0)
    # Sigmoid-like for direction:  negative → 0,  zero → 0.5,  positive → 1.0
    low_rising = 1.0 / (1.0 + np.exp(-l_slope_norm * 20000.0))
    out['f_c4_ascending_triangle'] = (high_flatness * low_rising).astype(np.float64)

    # --- Descending Triangle: flat lows + falling highs ---
    low_flatness = np.exp(-np.abs(l_slope_norm) * 15000.0)
    high_falling = 1.0 / (1.0 + np.exp(h_slope_norm * 20000.0))  # reversed: neg slope → 1
    out['f_c4_descending_triangle'] = (low_flatness * high_falling).astype(np.float64)

    # --- Symmetrical Triangle: highs falling AND lows rising (convergence) ---
    out['f_c4_symmetrical_triangle'] = (high_falling * low_rising).astype(np.float64)

    # --- Rising Wedge: both slopes up, lows rising faster → convergence ---
    h_up = 1.0 / (1.0 + np.exp(-h_slope_norm * 20000.0))
    l_up = 1.0 / (1.0 + np.exp(-l_slope_norm * 20000.0))
    wedge_rise_conv = np.clip(l_up - h_up, 0.0, 1.0)  # low slope > high slope
    out['f_c4_wedge_rising'] = (h_up * l_up * wedge_rise_conv).astype(np.float64)

    # --- Falling Wedge: both slopes down, highs falling faster → convergence ---
    h_down = 1.0 / (1.0 + np.exp(h_slope_norm * 20000.0))
    l_down = 1.0 / (1.0 + np.exp(l_slope_norm * 20000.0))
    wedge_fall_conv = np.clip(h_down - l_down, 0.0, 1.0)  # high slope more negative
    out['f_c4_wedge_falling'] = (h_down * l_down * wedge_fall_conv).astype(np.float64)

    # ────────────────────────────────────────────────────────────────
    # C. Flags — rolling momentum + consolidation (O(N))
    # ────────────────────────────────────────────────────────────────
    # Prior move (10-bar price change) normalized by ATR
    move_10 = (c - pd.Series(c).shift(10).ffill().fillna(c[0]).values) / (atr + 1e-10)
    # Recent range compression
    recent_range = (
        pd.Series(h).rolling(5, min_periods=3).max().ffill().fillna(h[0]).values -
        pd.Series(l).rolling(5, min_periods=3).min().ffill().fillna(l[0]).values
    ) / (atr + 1e-10)

    # Bull flag: prior sharp up + now tight range
    bull_flag_raw = np.clip(move_10 / 4.0, 0.0, 1.0) * np.clip(1.0 - recent_range / 4.0, 0.0, 1.0)
    out['f_c4_bull_flag'] = bull_flag_raw.astype(np.float64)

    # Bear flag: prior sharp down + now tight range
    bear_flag_raw = np.clip(-move_10 / 4.0, 0.0, 1.0) * np.clip(1.0 - recent_range / 4.0, 0.0, 1.0)
    out['f_c4_bear_flag'] = bear_flag_raw.astype(np.float64)

    # ── Cup & Handle (O'Neil): multi-scale parametrized detection ──
    out['f_c4_cup_handle'] = compute_cup_handle_score(
        df, cup_window=120, handle_window=30, price_tolerance=0.05)
    out['f_c4_cup_handle_small'] = compute_cup_handle_score(
        df, cup_window=60, handle_window=15, price_tolerance=0.04)
    out['f_c4_cup_handle_large'] = compute_cup_handle_score(
        df, cup_window=200, handle_window=50, price_tolerance=0.06)

    # ── Advanced chart patterns ──
    out['f_c4_rounding_bottom'] = compute_rounding_bottom_score(df)
    out['f_c4_n_shape'] = compute_n_shape_score(df)
    out['f_c4_ascending_breakout'] = compute_ascending_breakout_score(df)
    out['f_c4_rectangle_range'] = compute_rectangle_range_score(df)
    out['f_c4_island_reversal'] = compute_island_reversal_score(df)
    out['f_c4_diamond'] = compute_diamond_score(df)

    return out


# ==============================================================================
# V. Composite/Derived Features (合成特征 → 6 连续特征)
# ==============================================================================

def compute_composite_features(df: pd.DataFrame,
                               single: pd.DataFrame,
                               double: pd.DataFrame,
                               triple: pd.DataFrame,
                               chart: pd.DataFrame = None) -> pd.DataFrame:
    """
    Derived aggregators that combine signals across pattern families.

    Features:
      f_c5_fractal_density       — fractal count / window [0,1]
      f_c5_reversal_pressure     — aggregate reversal signal [-1,1]
      f_c5_continuation_pressure — aggregate continuation signal [-1,1]
      f_c5_pattern_entropy       — diversity of active patterns [0,1]
      f_c5_confidence_decay      — time since last strong signal, decayed [0,1]
      f_c5_volume_confirmation   — volume alignment with pattern direction [-1,1]
    """
    n = len(df)
    out = pd.DataFrame(index=df.index, dtype=np.float64)

    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)

    # Build unified feature lookup across all families
    _all_dfs = [single, double, triple]
    if chart is not None:
        _all_dfs.append(chart)
    _all_colmap = {}
    for _df in _all_dfs:
        for _c in _df.columns:
            _all_colmap[_c] = _df[_c].values

    def _add_col(name, dest):
        if name in _all_colmap:
            dest[:] += _all_colmap[name]

    # --- f_c5_fractal_density: fractal count per 20-bar window ---
    sh, sl = _find_swing_points(h, l, order=3)
    fractal_count = (sh.astype(np.float64) + sl.astype(np.float64))
    out['f_c5_fractal_density'] = pd.Series(fractal_count).rolling(20, min_periods=5).sum().fillna(0).values / 20.0

    # --- f_c5_reversal_pressure: aggregate of all reversal-type pattern scores ---
    reversal_bull = np.zeros(n, dtype=np.float64)
    reversal_bear = np.zeros(n, dtype=np.float64)

    # Bullish reversal patterns (single / double / triple / chart)
    for col in ['f_c1_hammer', 'f_c2_engulfing_bull', 'f_c2_piercing', 'f_c2_harami_bull',
                'f_c3_morning_star', 'f_c3_abandoned_baby_bull',
                'f_c4_double_bottom', 'f_c4_head_shoulders_bot', 'f_c4_wedge_falling',
                'f_c4_cup_handle', 'f_c4_cup_handle_small', 'f_c4_cup_handle_large',
                'f_c4_rounding_bottom', 'f_c4_n_shape']:
        _add_col(col, reversal_bull)

    # Bearish reversal patterns
    for col in ['f_c1_shooting_star', 'f_c2_engulfing_bear', 'f_c2_dark_cloud_cover', 'f_c2_harami_bear',
                'f_c3_evening_star', 'f_c3_abandoned_baby_bear',
                'f_c4_double_top', 'f_c4_head_shoulders_top', 'f_c4_wedge_rising']:
        _add_col(col, reversal_bear)

    # Signed reversal features: positive → bull, negative → bear (abs split)
    for col in ['f_c4_island_reversal', 'f_c4_diamond']:
        if col in _all_colmap:
            vals = _all_colmap[col]
            reversal_bull += np.clip(vals, 0.0, 1.0)
            reversal_bear += np.clip(-vals, 0.0, 1.0)

    out['f_c5_reversal_pressure'] = _clip_sym(reversal_bull - reversal_bear)

    # --- f_c5_continuation_pressure: aggregate of continuation-type pattern scores ---
    cont_bull = np.zeros(n, dtype=np.float64)
    cont_bear = np.zeros(n, dtype=np.float64)

    for col in ['f_c3_three_soldiers', 'f_c4_bull_flag', 'f_c4_ascending_triangle',
                'f_c4_ascending_breakout']:
        _add_col(col, cont_bull)

    for col in ['f_c3_three_crows', 'f_c4_bear_flag', 'f_c4_descending_triangle']:
        _add_col(col, cont_bear)

    out['f_c5_continuation_pressure'] = _clip_sym(cont_bull - cont_bear)

    # --- f_c5_pattern_entropy: how many DIFFERENT pattern types are active ---
    all_cols = []
    for _df in _all_dfs:
        all_cols.extend(_df.columns)
    all_features = pd.concat(_all_dfs, axis=1)
    active_count = (all_features[all_cols] > 0.3).astype(np.float64).sum(axis=1).values
    out['f_c5_pattern_entropy'] = _clip01(active_count / max(len(all_cols), 1))

    # --- f_c5_confidence_decay: exponential decay since last strong signal ---
    # Strong signal = any pattern score > 0.6
    strong_signal = ((all_features[all_cols] > 0.6).any(axis=1)).astype(np.float64).values
    decay = np.ones(n, dtype=np.float64)
    days_since = 0
    for i in range(n):
        if strong_signal[i] > 0.5:
            days_since = 0
            decay[i] = 1.0
        else:
            days_since += 1
            decay[i] = np.exp(-days_since / 10.0)  # half-life ~7 days
    out['f_c5_confidence_decay'] = decay

    # --- f_c5_volume_confirmation: volume surge direction vs pattern direction ---
    vol_ratio = _safe_div(v, pd.Series(v).rolling(20, min_periods=5).mean().ffill().fillna(v[0]).values)
    price_direction = np.sign(c - pd.Series(c).shift(1).fillna(c[0]).values)
    vol_confirmed = price_direction * (vol_ratio - 1.0)  # positive = vol confirms price
    out['f_c5_volume_confirmation'] = _clip_sym(vol_confirmed / 3.0)

    return out


# ==============================================================================
# VI. Multi-Timeframe Aggregator (多周期聚合)
# ==============================================================================

def add_multitimeframe_features(features_df: pd.DataFrame,
                                windows: Tuple[int, ...] = (5, 10, 20, 50),
                                aggs: Tuple[str, ...] = ('mean', 'max')) -> pd.DataFrame:
    """
    Add rolling aggregations of base features at multiple timeframes.

    For each base feature f_X, creates:
      f_X_5d_mean, f_X_5d_max, f_X_10d_mean, f_X_10d_max, f_X_20d_mean, f_X_20d_max

    This captures "has there been a pattern recently?" and "how strong was the
    strongest pattern in the last N days?" — critical for tree models.
    """
    base_cols = [c for c in features_df.columns if not any(f'_{w}d_' in c for w in windows)]

    # Build all rolled features in a list, concat once at the end (avoids fragmentation)
    rolled_parts = [features_df[base_cols]]
    for w in windows:
        for agg in aggs:
            # Roll all base columns at once for this (window, agg) combo
            rolled_block = features_df[base_cols].rolling(
                window=w, min_periods=max(3, w // 3)
            ).agg(agg)
            rolled_block = rolled_block.ffill().fillna(0)
            rolled_block.columns = [f'{c}_{w}d_{agg}' for c in base_cols]
            rolled_parts.append(rolled_block)

    return pd.concat(rolled_parts, axis=1)


# ==============================================================================
# VII. Main API
# ==============================================================================

def compute_pattern_features(df: pd.DataFrame,
                             include_multitimeframe: bool = True,
                             mtf_windows: Tuple[int, ...] = (5, 10, 20, 50),
                             verbose: bool = False) -> pd.DataFrame:
    """
    Compute the complete classical pattern feature set from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume. Index = datetime.
    include_multitimeframe : bool
        If True, add rolling aggregations at multiple timeframes.
    mtf_windows : tuple
        Rolling window sizes for multi-timeframe features.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Same index as df. All columns are float64, no NaN.
        ~42 base features, ~80+ with multi-timeframe expansion.
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must have column: {col}")

    if len(df) < 60:
        raise ValueError(f"Need at least 60 bars, got {len(df)}")

    if verbose:
        print(f"[PatternFactory] Computing features for {len(df)} bars...")

    # Compute all four families
    if verbose: print("  [1/5] Single-candle features...")
    single = compute_single_candle_features(df)

    if verbose: print("  [2/5] Double-candle features...")
    double = compute_double_candle_features(df)

    if verbose: print("  [3/5] Triple-candle features...")
    triple = compute_triple_candle_features(df)

    if verbose: print("  [4/5] Chart pattern features...")
    chart = compute_chart_pattern_features(df)

    if verbose: print("  [5/5] Composite features...")
    composite = compute_composite_features(df, single, double, triple, chart)

    # Concatenate all base features
    all_features = pd.concat([single, double, triple, chart, composite], axis=1)

    # Forward-fill + zero-fill to ensure no NaN
    all_features = all_features.ffill().fillna(0.0)

    # Add multi-timeframe expansions
    if include_multitimeframe:
        if verbose:
            n_base = len(all_features.columns)
            n_mtf = n_base * len(mtf_windows) * 2  # mean + max
            print(f"  Expanding {n_base} base features → ~{n_base + n_mtf} multi-timeframe features...")
        all_features = add_multitimeframe_features(all_features, windows=mtf_windows)

    # Final cleanup
    all_features = all_features.ffill().fillna(0.0)

    if verbose:
        print(f"[PatternFactory] Done. {len(all_features.columns)} features, shape={all_features.shape}")

    return all_features


# ==============================================================================
# VIII. School Integration
# ==============================================================================

def compute_school_pattern_signal(df: pd.DataFrame) -> Optional[Dict]:
    """
    School-compatible signal interface for expert_ensemble integration.

    Uses the pattern feature factory to generate a directional signal
    with confidence, suitable for inclusion as a new school.

    Returns dict with keys: signal, confidence, metadata
    """
    if df is None or df.empty or len(df) < 60:
        return None

    try:
        features = compute_pattern_features(df, include_multitimeframe=False, verbose=False)
        last = features.iloc[-1]

        # Aggregate direction from reversal + continuation pressure
        reversal = float(last.get('f_c5_reversal_pressure', 0))
        continuation = float(last.get('f_c5_continuation_pressure', 0))
        decay = float(last.get('f_c5_confidence_decay', 0))

        # Combine: fresh patterns matter more
        direction_score = reversal * decay + continuation * decay * 0.5
        confidence = min(abs(direction_score) * 1.5, 0.95)

        if direction_score > 0.06:
            signal = 'bullish'
        elif direction_score < -0.06:
            signal = 'bearish'
        else:
            signal = 'neutral'
            confidence = 0.1

        # Collect active pattern names for metadata
        pattern_cols = [c for c in features.columns if c.startswith('f_c1_') or c.startswith('f_c2_')
                       or c.startswith('f_c3_') or c.startswith('f_c4_')]
        active_patterns = []
        for col in pattern_cols:
            val = float(last.get(col, 0))
            if val > 0.4:
                active_patterns.append(f"{col.replace('f_', '')}={val:.2f}")

        return {
            'signal': signal,
            'confidence': round(confidence, 3),
            'metadata': {
                'direction_score': round(direction_score, 3),
                'reversal_pressure': round(reversal, 3),
                'continuation_pressure': round(continuation, 3),
                'active_patterns': active_patterns[:5],
                'decay': round(decay, 3),
            }
        }
    except Exception as e:
        return {'signal': 'neutral', 'confidence': 0.0, 'metadata': {'error': str(e)}}


# ==============================================================================
# IX. Quick Test
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 70)
    print("  Pattern Feature Factory — Quick Test")
    print("=" * 70)

    # Test on a single stock
    df = get_daily_kline('000012', days=300)
    if df is not None and not df.empty:
        print(f"\nLoaded {len(df)} bars for 000012")

        features = compute_pattern_features(df, verbose=True)

        print(f"\n--- Base feature columns ({len([c for c in features.columns if '_20d_' not in c and '_50d_' not in c and '_10d_' not in c and '_5d_' not in c])}) ---")
        base_cols = [c for c in features.columns if '_20d_' not in c and '_50d_' not in c
                    and '_10d_' not in c and '_5d_' not in c]
        for col in base_cols:
            vals = features[col].values[-20:]
            print(f"  {col:<35} mean={vals.mean():.4f}  std={vals.std():.4f}  last={vals[-1]:.4f}")

        print(f"\n--- School signal ---")
        sig = compute_school_pattern_signal(df)
        if sig:
            print(f"  signal={sig['signal']}, confidence={sig['confidence']}")
            print(f"  metadata={sig['metadata']}")
    else:
        print("No data loaded — test skipped.")
