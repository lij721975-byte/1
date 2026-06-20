#!/usr/bin/env python
# advanced_reversal_patterns.py
"""
Advanced Reversal Patterns — Island Reversal / Diamond Top & Bottom
=====================================================================
Two high-difficulty pattern detectors for the classical pattern school.

Red-line compliance:
  1. Pure vectorized — minimal loop: k ∈ [2,10] for island width; zero bar-loops
  2. Continuous score — float64 ∈ [-1.0, 1.0]
  3. Zero NaN — .ffill().fillna(0.0)
  4. Standard I/O — df[OHLCV] → np.ndarray (n,)

Author: Chief Alpha Miner
"""

import numpy as np
import pandas as pd


# ==============================================================================
# 1. Island Reversal (岛形反转)
# ==============================================================================

def compute_island_reversal_score(
    df: pd.DataFrame,
    max_island_width: int = 10,
) -> np.ndarray:
    """
    Vectorized Island Reversal (岛形反转) — continuous gap-reversal pressure.

    Key insight: true island reversals are extremely rare (0-1/year/stock).
    Instead of binary detection, we compute a continuous "gap reversal pressure"
    at EVERY bar using:
      - Gap strength (continuous, not boolean — how much did price gap?)
      - ewm-decaying memory of recent gaps
      - Island isolation as a continuous margin

    For k ∈ [2, max_island_width] (tiny loop over pattern widths):
      gap_reversal[t] = gap_down_ewm[t-k] × gap_up[t] × isolation_margin × tightness

    Parameters
    ----------
    df : pd.DataFrame with open,high,low,close
    max_island_width : int
        Maximum island duration in bars (default 10).

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [-1.0, 1.0]
        + = bullish island reversal pressure (bottom island)
        - = bearish island reversal pressure (top island)
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(c)

    # ── Continuous "gap pressure" (soft, NOT boolean) ────────────────
    # Key: even opening near yesterday's extreme creates nonzero pressure.
    # Formula: clip((open - high_prev) / atr + offset, 0, 1)
    #   offset=0.3 → opening AT yesterday's high gives 0.3 pressure
    #   offset=0.3 + slope=1/1.0 → opening 0.7 ATR above → 1.0 pressure
    h_prev = pd.Series(h).shift(1).ffill().fillna(h[0]).values
    l_prev = pd.Series(l).shift(1).ffill().fillna(l[0]).values

    atr = (
        pd.Series(h - l)
        .rolling(14, min_periods=5).mean()
        .ffill().fillna(1.0)
        .values
    )

    # Soft gap pressure: most bars get small nonzero values
    gap_up_strength = np.clip((o - h_prev) / (atr + 1e-10) / 1.2 + 0.25, 0.0, 1.0)
    gap_down_strength = np.clip((l_prev - o) / (atr + 1e-10) / 1.2 + 0.25, 0.0, 1.0)

    # ── Decaying gap memory (ewm, fully vectorized) ──────────────────
    # Recent exhaustion gap memory — stronger for recent, larger gaps
    half_gap = 4.0  # half-life of gap memory in bars
    alpha = 1.0 - np.exp(-np.log(2.0) / half_gap)

    gap_up_memory = (
        pd.Series(gap_up_strength)
        .ewm(alpha=alpha, min_periods=1, adjust=False)
        .mean()
        .values
    )
    gap_down_memory = (
        pd.Series(gap_down_strength)
        .ewm(alpha=alpha, min_periods=1, adjust=False)
        .mean()
        .values
    )

    # ── Accumulators ──────────────────────────────────────────────────
    island_bull = np.zeros(n, dtype=np.float64)
    island_bear = np.zeros(n, dtype=np.float64)

    # ── Scan island widths ────────────────────────────────────────────
    for k in range(2, max_island_width + 1):
        # --- Bottom Island (bullish) ---
        # Recent gap_down memory at t-k  ×  current gap_up × isolation
        gd_memory_k_ago = pd.Series(gap_down_memory).shift(k).ffill().fillna(0.0).values

        # Island high & low during the k-1 bars between gaps
        island_high = (
            pd.Series(h).rolling(k - 1, min_periods=k - 1).max()
            .shift(1).ffill().fillna(h[0]).values
        )
        island_low = (
            pd.Series(l).rolling(k - 1, min_periods=k - 1).min()
            .shift(1).ffill().fillna(l[0]).values
        )

        # Sigmoid isolation margin — centered at ratio=1.0 (island_high == boundary)
        # ratio < 1 → good isolation → margin > 0.5
        # ratio > 1 → poor isolation → margin < 0.5
        pre_gap_low = pd.Series(l).shift(k + 1).ffill().fillna(l[0]).values
        ratio_pre = island_high / (pre_gap_low + 1e-10)
        ratio_post = island_high / (o + 1e-10)
        margin_pre = 1.0 / (1.0 + np.exp((ratio_pre - 1.0) * 12.0))
        margin_post = 1.0 / (1.0 + np.exp((ratio_post - 1.0) * 12.0))
        isolation_margin = margin_pre * margin_post

        # Island tightness
        island_range = island_high - island_low
        tightness = np.exp(-island_range / (atr + 1e-10) / 1.5)

        # Bull score: exhaustion gap memory × current breakaway gap × isolation × tightness
        bull_score = gd_memory_k_ago * gap_up_strength * isolation_margin * tightness
        np.maximum(island_bull, bull_score, out=island_bull)

        # --- Top Island (bearish) ---
        gu_memory_k_ago = pd.Series(gap_up_memory).shift(k).ffill().fillna(0.0).values

        island_high = (
            pd.Series(h).rolling(k - 1, min_periods=k - 1).max()
            .shift(1).ffill().fillna(h[0]).values
        )
        island_low = (
            pd.Series(l).rolling(k - 1, min_periods=k - 1).min()
            .shift(1).ffill().fillna(l[0]).values
        )

        pre_gap_high = pd.Series(h).shift(k + 1).ffill().fillna(h[0]).values
        ratio_pre = pre_gap_high / (island_low + 1e-10)
        ratio_post = o / (island_low + 1e-10)
        margin_pre = 1.0 / (1.0 + np.exp((ratio_pre - 1.0) * 12.0))
        margin_post = 1.0 / (1.0 + np.exp((ratio_post - 1.0) * 12.0))
        isolation_margin = margin_pre * margin_post

        island_range = island_high - island_low
        tightness = np.exp(-island_range / (atr + 1e-10) / 1.5)

        bear_score = gu_memory_k_ago * gap_down_strength * isolation_margin * tightness
        np.maximum(island_bear, bear_score, out=island_bear)

    # ── Combine ───────────────────────────────────────────────────────
    score = island_bull - island_bear

    # ── Persistence decay ─────────────────────────────────────────────
    nonzero_mask = np.abs(score) > 0.02
    positions = np.where(nonzero_mask, np.arange(n, dtype=np.float64), np.nan)
    last_pos = pd.Series(positions).ffill().fillna(-np.inf).values
    days = np.arange(n, dtype=np.float64) - last_pos
    decay = np.exp(-np.clip(days, 0.0, np.inf) / 8.0)
    had_signal = np.isfinite(last_pos)
    score = np.where(had_signal, score * decay, score)

    return np.clip(score, -1.0, 1.0).astype(np.float64)


# ==============================================================================
# 2. Diamond Top / Bottom (菱形顶/底)
# ==============================================================================

def compute_diamond_score(
    df: pd.DataFrame,
    window: int = 40,
    trend_window: int = 20,
) -> np.ndarray:
    """
    Vectorized Diamond Top/Bottom (菱形顶/底) score.

    Core logic:
      - Split window=40 into two halves (20 bars each).
      - First half (t-40 to t-20): TR should EXPAND (megaphone / broadening).
        → Compare TR mean of Q1 (oldest 10 bars) vs Q2 (next 10 bars).
        Q2_TR > Q1_TR ⇒ expanding ✓
      - Second half (t-20 to t): TR should CONTRACT (triangle / squeeze).
        → Compare TR mean of Q3 vs Q4.
        Q4_TR < Q3_TR ⇒ contracting ✓
      - Trend context: 20-day return decides diamond-top (+return) vs
        diamond-bottom (-return).

    Score ∈ [-1, 1]:
      + = diamond bottom (bullish reversal after downtrend)
      - = diamond top   (bearish reversal after uptrend)

    Parameters
    ----------
    df : pd.DataFrame with high,low,close
    window : int
        Total diamond lookback (default 40). Must be divisible by 4.
    trend_window : int
        Lookback for trend context (default 20).

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [-1.0, 1.0]
    """
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(c)

    if n < window:
        return np.zeros(n, dtype=np.float64)

    q = window // 4  # quarter width in bars (10 for window=40)
    half = window // 2

    # ── Daily true range ──────────────────────────────────────────────
    tr = h - l
    atr_base = (
        pd.Series(tr)
        .rolling(14, min_periods=5).mean()
        .ffill().fillna(tr[0])
        .values
    )

    # ── TR averages over each quarter (fully vectorized rolling + shift) ──
    # Q1: bars [t-window, t-window+q]     — oldest quarter
    # Q2: bars [t-window+q, t-half]       — second quarter
    # Q3: bars [t-half, t-half+q]         — third quarter
    # Q4: bars [t-q, t]                   — most recent quarter

    tr_q1 = (
        pd.Series(tr).shift(window - q)
        .rolling(q, min_periods=max(3, q // 2)).mean()
        .ffill().fillna(tr[0]).values
    )
    tr_q2 = (
        pd.Series(tr).shift(half - q)
        .rolling(q, min_periods=max(3, q // 2)).mean()
        .ffill().fillna(tr[0]).values
    )
    tr_q3 = (
        pd.Series(tr).shift(q)
        .rolling(q, min_periods=max(3, q // 2)).mean()
        .ffill().fillna(tr[0]).values
    )
    tr_q4 = (
        pd.Series(tr)
        .rolling(q, min_periods=max(3, q // 2)).mean()
        .ffill().fillna(tr[0]).values
    )

    # ── Expansion score (first half: Q2 > Q1 → megaphone) ─────────────
    # Expansion ratio: Q2_TR / Q1_TR.  >1.0 = expanding (bullish for diamond)
    expansion_ratio = tr_q2 / (tr_q1 + 1e-10)
    # Sigmoid: ratio=1.0 → 0.5,  ratio=1.3 → 0.82,  ratio=1.5 → 0.92
    expansion_score = 1.0 / (1.0 + np.exp(-(expansion_ratio - 1.15) * 8.0))

    # ── Contraction score (second half: Q4 < Q3 → contracting triangle) ──
    contraction_ratio = tr_q4 / (tr_q3 + 1e-10)
    # Inverse sigmoid: ratio=1.0 → 0.5,  ratio=0.7 → 0.82,  ratio=0.5 → 0.92
    contraction_score = 1.0 / (1.0 + np.exp((contraction_ratio - 0.85) * 8.0))

    # ── Diamond shape score = both halves must conform ─────────────────
    diamond_shape = expansion_score * contraction_score  # geometric mean-like

    # ── Trend context ──────────────────────────────────────────────────
    ret_20 = (c - pd.Series(c).shift(trend_window).ffill().fillna(c[0]).values) / (
        pd.Series(c).shift(trend_window).ffill().fillna(c[0]).values + 1e-10
    )

    # In uptrend → diamond top (bearish reversal) → negative score
    # In downtrend → diamond bottom (bullish reversal) → positive score
    trend_strength = np.clip(np.abs(ret_20) / 0.15, 0.0, 1.0)  # 15% move → full trend

    uptrend = np.clip(ret_20 / 0.10, 0.0, 1.0)     # positive return → uptrend
    downtrend = np.clip(-ret_20 / 0.10, 0.0, 1.0)   # negative return → downtrend

    # ── Composite ──────────────────────────────────────────────────────
    # Diamond top: uptrend + diamond shape → negative
    # Diamond bottom: downtrend + diamond shape → positive
    diamond_top_strength = diamond_shape * uptrend * trend_strength
    diamond_bot_strength = diamond_shape * downtrend * trend_strength

    score = diamond_bot_strength - diamond_top_strength

    # ── Persistence decay ──────────────────────────────────────────────
    nonzero_mask = np.abs(score) > 0.05
    positions = np.where(nonzero_mask, np.arange(n, dtype=np.float64), np.nan)
    last_pos = pd.Series(positions).ffill().fillna(-np.inf).values
    days = np.arange(n, dtype=np.float64) - last_pos
    decay = np.exp(-np.clip(days, 0.0, np.inf) / 12.0)
    had_signal = np.isfinite(last_pos)
    score = np.where(had_signal, score * decay, score)

    score = np.clip(score, -1.0, 1.0)
    return score.astype(np.float64)


# ==============================================================================
# Drop-in feature block for pattern_feature_factory.py
# ==============================================================================

def compute_advanced_reversal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      f_c4_island_reversal     — 岛形反转 [-1, 1]  (+bullish, -bearish)
      f_c4_diamond             — 菱形顶/底  [-1, 1]  (+bottom, -top)
    """
    out = pd.DataFrame(index=df.index)
    out['f_c4_island_reversal'] = compute_island_reversal_score(df)
    out['f_c4_diamond'] = compute_diamond_score(df)
    return out


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 70)
    print("  Advanced Reversal Patterns — Quick Test")
    print("=" * 70)

    for code in ['000012', '600519', '000001', '600036']:
        df = get_daily_kline(code, days=500)
        if df is None or not df.empty:
            island = compute_island_reversal_score(df)
            diamond = compute_diamond_score(df)

            for name, arr in [('island_reversal', island), ('diamond', diamond)]:
                uq = len(set(arr.round(4)))
                bull = (arr > 0.2).sum()
                bear = (arr < -0.2).sum()
                print(f"  {code} {name:<20} unique={uq:4d}  NaN={np.isnan(arr).sum()}  "
                      f"range=[{arr.min():.4f},{arr.max():.4f}]  "
                      f">+0.2={bull:4d}  <-0.2={bear:4d}")
