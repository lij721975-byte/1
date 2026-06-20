#!/usr/bin/env python
# brooks_macro_features.py — Al Brooks Price Action Part 1: Macro Foundation
#
# T1: Trend Bar Classification    (趋势K线 vs 盘整K线 — polarity-separated)
# T4: 20 EMA Gap Magnet           (20EMA偏离引力 — direction-NEUTRAL)
# T7: Always In Direction         (始终在场方向 — composite score [-1,1])
#
# Pure vectorized Pandas/NumPy. ZERO for-loops. ZERO shift(-1).
# A-share calibrated: body_pct threshold relaxed to 0.45.
# Limit-up/down special handling: spread≈0 → force score based on direction.

import numpy as np
import pandas as pd

EPS = 1e-8


# ======================================================================
# Main entry point
# ======================================================================

def compute_brooks_macro_features(open_df, high_df, low_df, close_df, volume_df):
    """
    Compute Brooks PA macro foundation features.

    Parameters
    ----------
    open_df, high_df, low_df, close_df, volume_df : pd.DataFrame
        Wide-format (index=datetime, columns=stocks).

    Returns
    -------
    dict[str, pd.DataFrame]
        brooks_trend_bull_score    — T1 Strong/Normal bull trend bar (0-1)
        brooks_trend_bear_score    — T1 Strong/Normal bear trend bar (0-1)
        brooks_ema_gap_magnet      — T4 EMA gap magnet strength (0-1, neutral)
        brooks_ema_gap_direction   — T4 EMA gap direction (+1/-1)
        brooks_always_in_score     — T7 Always In composite [-1, 1]
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

    # ==================================================================
    # SECTION 0 — Shared precomputations
    # ==================================================================

    # ---- K-line geometry ----
    o_arr, h_arr, l_arr, c_arr, v_arr = o.values, h.values, l.values, c.values, v.values
    spread_arr = h_arr - l_arr
    body_arr   = np.abs(c_arr - o_arr)
    up_wick    = h_arr - np.maximum(o_arr, c_arr)
    lo_wick    = np.minimum(o_arr, c_arr) - l_arr

    body_pct   = body_arr / (spread_arr + EPS)
    up_w_pct   = up_wick / (spread_arr + EPS)
    lo_w_pct   = lo_wick / (spread_arr + EPS)
    close_pct  = (c_arr - l_arr) / (spread_arr + EPS)   # 0=low, 1=high

    is_bull_bar = c_arr > o_arr
    is_bear_bar = c_arr < o_arr

    # ---- ATR(14) ----
    tr_vals = np.fmax.reduce([
        h_arr - l_arr,
        np.abs(h_arr - np.roll(c_arr, 1, axis=0)),
        np.abs(l_arr - np.roll(c_arr, 1, axis=0)),
    ])
    tr_vals[0] = h_arr[0] - l_arr[0]
    tr_df = pd.DataFrame(tr_vals, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean().values

    # ---- EMA(20) ----
    ema20 = c.ewm(span=20, adjust=False).mean().values

    # ---- Previous close for limit-up/down detection ----
    c_prev = np.roll(c_arr, 1, axis=0)
    c_prev[0] = c_arr[0]

    # ════════════════════════════════════════════════════════════════
    # T1 — Trend Bar Classification (趋势K线分类)
    # ════════════════════════════════════════════════════════════════

    spread_tiny = spread_arr < atr14 * 0.05  # near-zero spread (limit-up/down)

    # ---- Limit-up special handling ----
    limit_up   = spread_tiny & (c_arr > c_prev) & (v_arr > 0)
    limit_down = spread_tiny & (c_arr < c_prev) & (v_arr > 0)

    # ---- Strong Bull Trend Bar ----
    # body_pct >= 0.45 (A-share relaxed), lo_w_pct <= 0.15, close_pct >= 0.75
    strong_bull = (
        is_bull_bar & ~spread_tiny &
        (body_pct >= 0.45) &
        (lo_w_pct <= 0.15) &
        (close_pct >= 0.75) &
        (spread_arr >= atr14 * 0.50)
    )
    strong_bull_score = np.where(
        strong_bull,
        np.minimum(1.0, (body_pct - 0.30) * 2.0) * np.minimum(1.0, close_pct * 1.2),
        0.0
    )
    # Limit-up gets score 1.0
    strong_bull_score = np.where(limit_up, 1.0, strong_bull_score)

    # ---- Strong Bear Trend Bar ----
    strong_bear = (
        is_bear_bar & ~spread_tiny &
        (body_pct >= 0.45) &
        (up_w_pct <= 0.15) &
        (close_pct <= 0.25) &
        (spread_arr >= atr14 * 0.50)
    )
    strong_bear_score = np.where(
        strong_bear,
        np.minimum(1.0, (body_pct - 0.30) * 2.0) * np.minimum(1.0, (1.0 - close_pct) * 1.2),
        0.0
    )
    strong_bear_score = np.where(limit_down, 1.0, strong_bear_score)

    # ---- Normal Bull Bar ----
    normal_bull = (
        is_bull_bar & ~spread_tiny &
        (body_pct >= 0.35) &
        (close_pct >= 0.55) &
        (spread_arr >= atr14 * 0.30) &
        ~strong_bull  # already captured
    )
    normal_bull_score = np.where(normal_bull, body_pct * close_pct, 0.0)

    # ---- Normal Bear Bar ----
    normal_bear = (
        is_bear_bar & ~spread_tiny &
        (body_pct >= 0.35) &
        (close_pct <= 0.45) &
        (spread_arr >= atr14 * 0.30) &
        ~strong_bear
    )
    normal_bear_score = np.where(normal_bear, body_pct * (1.0 - close_pct), 0.0)

    # ---- Tail Reversal Bar (long lower wick = potential bullish reversal) ----
    tail_bull = (
        is_bull_bar & ~spread_tiny &
        (lo_w_pct >= 0.50) &
        ~strong_bull & ~normal_bull
    )
    tail_bull_score = np.where(tail_bull, lo_w_pct * 0.60, 0.0)

    # ---- Tail Reversal Bar (long upper wick = potential bearish reversal) ----
    tail_bear = (
        is_bear_bar & ~spread_tiny &
        (up_w_pct >= 0.50) &
        ~strong_bear & ~normal_bear
    )
    tail_bear_score = np.where(tail_bear, up_w_pct * 0.60, 0.0)

    # ---- Composite trend scores ----
    trend_bull = np.maximum.reduce([strong_bull_score, normal_bull_score, tail_bull_score])
    trend_bear = np.maximum.reduce([strong_bear_score, normal_bear_score, tail_bear_score])
    trend_bull = np.clip(trend_bull, 0.0, 1.0)
    trend_bear = np.clip(trend_bear, 0.0, 1.0)

    # ════════════════════════════════════════════════════════════════
    # T4 — 20 EMA Gap Magnet (20EMA偏离引力)
    # ════════════════════════════════════════════════════════════════

    gap_ema = (c_arr - ema20) / (ema20 + EPS)           # signed deviation rate
    ema_ratio_for_atr = atr14 / (ema20 + EPS)
    gap_normalized = gap_ema / (ema_ratio_for_atr + EPS) # ATR-normalized gap

    gap_abs = np.abs(gap_normalized)

    # Magnet strength: 0=on EMA, 1=extremely far
    magnet_strength = np.clip(gap_abs / 3.0, 0.0, 1.0)

    # Enhancement M1: gap beyond 1.5x its 60-bar rolling mean
    gap_abs_60_mean = pd.DataFrame(gap_abs, index=idx, columns=cols) \
                        .rolling(60, min_periods=20).mean().values
    m1_boost = gap_abs > gap_abs_60_mean * 1.5
    magnet_strength = np.where(m1_boost, np.minimum(magnet_strength * 1.20, 1.0), magnet_strength)

    # Enhancement M2: gap shrinking over last 3 bars (regression already underway)
    gap_s1 = np.roll(gap_abs, 1, axis=0); gap_s1[0] = gap_abs[0]
    gap_s2 = np.roll(gap_abs, 2, axis=0); gap_s2[:2] = gap_abs[:2]
    m2_shrinking = (gap_abs < gap_s1) & (gap_s1 < gap_s2)
    # If shrinking: magnet is partially "used" → reduce
    magnet_strength = np.where(m2_shrinking, magnet_strength * 0.85, magnet_strength)

    magnet_strength = np.clip(magnet_strength, 0.0, 1.0)

    # Gap direction: +1.0 above EMA, -1.0 below EMA
    gap_direction = np.where(gap_ema > 0, 1.0, np.where(gap_ema < 0, -1.0, 0.0))

    # ════════════════════════════════════════════════════════════════
    # T7 — Always In Direction (始终在场方向)
    # ════════════════════════════════════════════════════════════════

    # ---- Dimension 1: Bar Momentum (10-bar EMA of signed body) ----
    body_sign = np.where(c_arr > o_arr, 1.0, np.where(c_arr < o_arr, -1.0, 0.0))
    signed_body = body_sign * body_pct
    bar_momentum = pd.DataFrame(signed_body, index=idx, columns=cols) \
                     .ewm(span=10, adjust=False).mean().values
    # Scale: typical range ~[-0.5, 0.5], clip to [-1, 1]
    bar_momentum = np.clip(bar_momentum * 2.0, -1.0, 1.0)

    # ---- Dimension 2: EMA Position ----
    ema_position = np.clip(gap_ema * 10.0, -1.0, 1.0)

    # ---- Dimension 3: Swing Structure (Higher High / Higher Low) ----
    # Detect recent swing highs and lows via rolling max/min
    # HH: the 20-bar rolling max is HIGHER than the 20-bar rolling max 10 bars ago
    high_20 = h.rolling(20, min_periods=10).max().values
    low_20  = l.rolling(20, min_periods=10).min().values
    high_20_s10 = pd.DataFrame(high_20, index=idx, columns=cols).shift(10).values
    low_20_s10  = pd.DataFrame(low_20, index=idx, columns=cols).shift(10).values

    higher_high = high_20 > high_20_s10
    higher_low  = low_20 > low_20_s10

    # Nested np.where flattened for clarity
    structure_score = np.where(
        higher_high & higher_low,  0.60,
        np.where(higher_high,      0.30,
        np.where(higher_low,       0.30,
        np.where((high_20 < high_20_s10) & (low_20 < low_20_s10), -0.60, 0.0))))

    # ---- Composite Always In ----
    ai_raw = bar_momentum * 0.30 + ema_position * 0.30 + structure_score * 0.40
    ai_score = np.clip(ai_raw, -1.0, 1.0)

    # ==================================================================
    # Package results
    # ==================================================================

    results = {}

    results['brooks_trend_bull_score']  = pd.DataFrame(trend_bull, index=idx, columns=cols)
    results['brooks_trend_bear_score']  = pd.DataFrame(trend_bear, index=idx, columns=cols)
    results['brooks_ema_gap_magnet']    = pd.DataFrame(magnet_strength, index=idx, columns=cols)
    results['brooks_ema_gap_direction'] = pd.DataFrame(gap_direction, index=idx, columns=cols)
    results['brooks_always_in_score']   = pd.DataFrame(ai_score, index=idx, columns=cols)

    return results


# ======================================================================
# School-compatible split-return interface
# ======================================================================

def compute_brooks_macro_score_split(indicators: dict):
    """
    Polarity-grouped split-return for _compute_school_brooks_pa integration.

    Returns
    -------
    (score_bull, score_bear, reasons_bull, reasons_bear,
     ema_magnet, ema_direction, always_in_score)
    """
    try:
        cp = indicators.get('current_price', 0) or 0
        o_v = indicators.get('open', cp) or cp
        h_v = indicators.get('high', cp) or cp
        l_v = indicators.get('low', cp) or cp
        v_v = indicators.get('volume', 0) or 0
        idx_d = pd.DatetimeIndex([pd.Timestamp.now()])
        o_df = pd.DataFrame({'S': [float(o_v)]}, index=idx_d)
        h_df = pd.DataFrame({'S': [float(h_v)]}, index=idx_d)
        l_df = pd.DataFrame({'S': [float(l_v)]}, index=idx_d)
        c_df = pd.DataFrame({'S': [float(cp)]}, index=idx_d)
        v_df = pd.DataFrame({'S': [float(v_v)]}, index=idx_d)

        feats = compute_brooks_macro_features(o_df, h_df, l_df, c_df, v_df)

        def _read(key):
            df = feats.get(key)
            if df is not None and not df.empty:
                v = df.values
                if v.ndim == 2:
                    return float(v[0, 0]) if v.size > 0 else 0.0
                return float(v[0]) if len(v) > 0 else 0.0
            return 0.0

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        # T1: Trend bar scores
        tb = _read('brooks_trend_bull_score')
        ts = _read('brooks_trend_bear_score')
        if tb > 0:
            score_bull += tb * 0.15
            if tb >= 0.80:
                reasons_bull.append(f'强看涨趋势K线(score={tb:.2f})')
            elif tb >= 0.30:
                reasons_bull.append(f'看涨K线(score={tb:.2f})')
        if ts > 0:
            score_bear += ts * 0.15
            if ts >= 0.80:
                reasons_bear.append(f'强看跌趋势K线(score={ts:.2f})')
            elif ts >= 0.30:
                reasons_bear.append(f'看跌K线(score={ts:.2f})')

        # T4: EMA magnet (neutral)
        magnet = _read('brooks_ema_gap_magnet')
        ema_dir = _read('brooks_ema_gap_direction')
        if magnet > 0.50:
            if ema_dir > 0:
                reasons_bear.append(f'EMA上方强引力(mag={magnet:.2f})→回归风险')
            else:
                reasons_bull.append(f'EMA下方强引力(mag={magnet:.2f})→回归机会')

        # T7: Always In direction
        ai = _read('brooks_always_in_score')
        if ai > 0.15:
            score_bull += abs(ai) * 0.20
            reasons_bull.append(f'Always In看涨(score={ai:.2f})')
        elif ai < -0.15:
            score_bear += abs(ai) * 0.20
            reasons_bear.append(f'Always In看跌(score={ai:.2f})')

        return score_bull, score_bear, reasons_bull, reasons_bear, magnet, ema_dir, ai
    except Exception:
        return 0.0, 0.0, [], [], 0.0, 0.0, 0.0


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Brooks PA Macro Features Test ===")
    for code in ['000651', '300750', '600519']:
        df = get_daily_kline(code, days=300)
        if df is None or df.empty:
            continue
        idx_d = df.index
        def _d(cn):
            return pd.DataFrame({'S': df[cn].values}, index=idx_d)
        feats = compute_brooks_macro_features(
            _d('open'), _d('high'), _d('low'), _d('close'), _d('volume'))

        print(f"\n{code}:")
        tb = feats['brooks_trend_bull_score'].values
        ts = feats['brooks_trend_bear_score'].values
        strong_bull = int((tb >= 0.80).sum())
        strong_bear = int((ts >= 0.80).sum())
        any_bull = int((tb > 0).sum())
        any_bear = int((ts > 0).sum())
        print(f"  T1 Trend: strong_bull={strong_bull} strong_bear={strong_bear} "
              f"any_bull={any_bull} any_bear={any_bear}")

        mag = feats['brooks_ema_gap_magnet'].values
        print(f"  T4 Magnet: mean={mag.mean():.3f} max={mag.max():.3f} "
              f"strong(>0.5)={int((mag > 0.5).sum())}")

        ai = feats['brooks_always_in_score'].values
        bull_ai = int((ai > 0.15).sum())
        bear_ai = int((ai < -0.15).sum())
        print(f"  T7 AlwaysIn: bull={bull_ai} bear={bear_ai} neutral={300-bull_ai-bear_ai}")
