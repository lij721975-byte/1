#!/usr/bin/env python
# rectangle_range_feature.py
"""
Rectangle / Range (矩形箱体) — Continuous Feature Extractor
=============================================================
Fully vectorized box-range consolidation detector.

Red-line compliance:
  1. Pure vectorized — zero Python for-in-range, all numpy/pandas rolling
  2. Continuous score — float64 ∈ [0.0, 1.0]
  3. Zero NaN — .ffill().fillna(0.0)
  4. Standard I/O — df[OHLCV] → np.ndarray (n,)

Quantitative logic (window=40):
  A. Channel width: (rolling_max_high - rolling_min_low) / avg_price
     Ideal range: 5%–15%, peak score at ~10%
  B. Internal stability: rolling_std(close) / channel_width
     Lower ratio → tighter consolidation → higher score
  C. Range persistence: current_width / longer_term_avg_width
     Consistently narrow = genuine box (not random noise)

Score = width_score × stability_score × persistence_score

Author: Chief Alpha Miner
"""

import numpy as np
import pandas as pd


def compute_rectangle_range_score(
    df: pd.DataFrame,
    window: int = 40,
    width_min: float = 0.05,
    width_max: float = 0.15,
) -> np.ndarray:
    """
    Vectorized Rectangle/Range (矩形箱体) continuous score.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'high','low','close'. Time-sorted.
    window : int
        Lookback for range detection (default 40 = ~2 months daily).
    width_min : float
        Min channel width as % of price to qualify (default 0.05 = 5%).
    width_max : float
        Max channel width as % of price to qualify (default 0.15 = 15%).

    Returns
    -------
    score : np.ndarray (n,) float64 ∈ [0.0, 1.0]
        0.0 = no box / trending
        0.3–0.5 = loose consolidation forming
        0.5–0.7 = tight box, low internal volatility
        0.7–1.0 = textbook rectangle — tight, persistent, near boundary
    """
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    n = len(c)

    if n < window:
        return np.zeros(n, dtype=np.float64)

    # ── Shared rolling precomputations (O(N), fully vectorized) ──
    half = max(window // 2, 5)
    min_p = max(window // 4, 5)

    roll_high = (
        pd.Series(h).rolling(window, min_periods=half).max().ffill().fillna(h[0]).values
    )
    roll_low = (
        pd.Series(l).rolling(window, min_periods=half).min().ffill().fillna(l[0]).values
    )
    roll_avg = (
        pd.Series(c).rolling(window, min_periods=half).mean().ffill().fillna(c[0]).values
    )
    roll_std = (
        pd.Series(c).rolling(window, min_periods=half).std().ffill().fillna(0.0).values
    )

    # ── Factor A: Channel width appropriateness ────────────────────────
    channel_width_abs = roll_high - roll_low
    channel_width_pct = channel_width_abs / (roll_avg + 1e-10)

    # Gaussian-like: peak at ideal_center, decay to 0 at width_min and width_max
    ideal_center = (width_min + width_max) / 2.0   # 0.10
    ideal_sigma = (width_max - width_min) / 4.0     # 0.025 → width_min/max ≈ 2σ from center
    width_deviation = (channel_width_pct - ideal_center) / (ideal_sigma + 1e-10)
    # exp(-0.5 * deviation^2):  deviation=0 → 1.0,  deviation=2 → 0.135,  deviation=3 → 0.011
    width_score = np.exp(-0.5 * width_deviation * width_deviation)

    # ── Factor B: Internal stability (low close-volatility within channel) ──
    # Ratio of internal price dispersion to channel width.
    # Low ratio → price is stable within the box (not choppy).
    stability_ratio = roll_std / (channel_width_abs + 1e-10)
    # Exponential decay: ratio=0 → 1.0,  ratio=0.15 → 0.61,  ratio=0.30 → 0.37
    stability_score = np.exp(-stability_ratio / 0.15)

    # ── Factor C: Range persistence (is the box sustained, not fleeting?) ──
    # Compare current channel width to longer-term average width.
    # If the range has been tight for a while, persistence is high.
    roll_width_avg = (
        pd.Series(channel_width_abs)
        .rolling(window * 2, min_periods=window)
        .mean()
        .ffill()
        .fillna(channel_width_abs[0])
        .values
    )
    persistence_ratio = channel_width_abs / (roll_width_avg + 1e-10)
    # ratio ≈ 1.0 → channel width is stable (persistent box)
    # ratio < 1.0 → range is tightening further (compression before breakout)
    # ratio > 1.5 → range is expanding (box breaking down)
    persistence_score = np.exp(-np.abs(persistence_ratio - 1.0) / 0.40)

    # ── Factor D: Boundary proximity (optional, adds richness) ─────────
    # Where is current close within the channel?  (0=support, 1=resistance)
    position_in_range = (c - roll_low) / (channel_width_abs + 1e-10)
    position_in_range = np.clip(position_in_range, 0.0, 1.0)

    # Near-boundary bonus: being near support or resistance adds signal value
    # (it means the box is being tested → potential breakout)
    boundary_proximity = 1.0 - 2.0 * np.abs(position_in_range - 0.5)  # 0 at mid, 1 at edges
    boundary_proximity = np.clip(boundary_proximity, 0.0, 1.0)

    # ── Composite Score ─────────────────────────────────────────────────
    # Geometric mean: preserves AND-logic (all must align) without
    # over-compressing the score range like a plain product would.
    # 0.8 × 0.7 × 0.7 = 0.39 → geom_mean = 0.73 ✓
    score = np.power(width_score * stability_score * persistence_score, 1.0 / 3.0)

    # Boundary proximity amplifies: testing the edges of a good box is
    # more informative than sitting in the middle
    score = score * (0.85 + 0.15 * boundary_proximity)

    # ── Persistence decay: score persists briefly after box ends ────────
    # When score drops below threshold, decay rather than vanish instantly
    score_positions = np.where(
        score > 0.15,
        np.arange(n, dtype=np.float64),
        np.nan
    )
    last_signal_idx = (
        pd.Series(score_positions)
        .ffill()
        .fillna(-np.inf)
        .values
    )
    days_since = np.arange(n, dtype=np.float64) - last_signal_idx
    decay = np.exp(-np.clip(days_since, 0.0, np.inf) / 8.0)  # half-life ~5.5 bars
    had_signal = np.isfinite(last_signal_idx)
    score = np.where(had_signal, np.maximum(score, score * decay), score)

    score = np.clip(score, 0.0, 1.0)
    return score.astype(np.float64)


# ==============================================================================
# Drop-in feature for pattern_feature_factory integration
# ==============================================================================

def compute_rectangle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      f_c4_rectangle_range        — box consolidation score [0,1]
      f_c4_rectangle_position     — where price sits in the box [0,1] (0=low, 1=high)
    """
    out = pd.DataFrame(index=df.index)
    out['f_c4_rectangle_range'] = compute_rectangle_range_score(df)

    # Position: only meaningful when range score is elevated
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    window = 40
    roll_high = pd.Series(h).rolling(window, min_periods=10).max().ffill().fillna(h[0]).values
    roll_low = pd.Series(l).rolling(window, min_periods=10).min().ffill().fillna(l[0]).values
    channel = roll_high - roll_low
    pos = np.clip((c - roll_low) / (channel + 1e-10), 0.0, 1.0)
    # Weight by range score: position only matters when actually in a box
    range_score = out['f_c4_rectangle_range'].values
    out['f_c4_rectangle_position'] = (pos * range_score).astype(np.float64)

    return out


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 70)
    print("  Rectangle Range Feature — Quick Test")
    print("=" * 70)

    for code in ['000012', '600519', '000001', '600036']:
        df = get_daily_kline(code, days=300)
        if df is None or not df.empty:
            score = compute_rectangle_range_score(df)
            uq = len(set(score.round(4)))
            print(f"  {code}: unique={uq:4d}  NaN={np.isnan(score).sum()}  "
                  f"range=[{score.min():.4f},{score.max():.4f}]  "
                  f">0.3={(score>0.3).sum():4d}  >0.5={(score>0.5).sum():4d}  >0.7={(score>0.7).sum():4d}")
