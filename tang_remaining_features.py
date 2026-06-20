#!/usr/bin/env python
# tang_remaining_features.py — 唐能通 剩余 9 形态 向量化计算
"""
Appends to TangNengTongFeatures: 9 additional patterns.

  1. 两阳夹一阴       6. 青龙取水
  2. 芝麻开花         7. 梅开二度
  3. 冷空气           8. 黑马草上飞
  4. 量顶天立地       9. 季线穿越
  5. 断头铡刀

All inputs: wide-format DataFrames. Zero for-loops. EPS on all divisions.
"""

import numpy as np
import pandas as pd

EPS = 1e-8


def compute_remaining_tnt_features(open_df, high_df, low_df, close_df, volume_df):
    """
    Compute 9 additional Tang Nengtong features.

    Returns: dict[str, DataFrame] with keys matching pattern names below.
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df

    # ================================================================
    # Shared indicators
    # ================================================================
    ma5   = c.rolling(5,  min_periods=3).mean()
    ma10  = c.rolling(10, min_periods=5).mean()
    ma20  = c.rolling(20, min_periods=10).mean()
    ma60  = c.rolling(60, min_periods=30).mean()

    vol_ma5  = v.rolling(5,  min_periods=2).mean()
    vol_ma10 = v.rolling(10, min_periods=5).mean()
    vol_ma20 = v.rolling(20, min_periods=10).mean()

    body   = (c - o).abs()
    chl    = h - l + EPS
    body_pct = body / chl
    is_bull = c > o
    is_bear = c < o

    ma_list = [ma5.values, ma10.values, ma20.values, ma60.values]
    ma_max = np.fmax.reduce(ma_list)
    ma_min = np.fmin.reduce(ma_list)
    ma_bind = ma_max / (ma_min + EPS) - 1.0

    results = {}

    # ================================================================
    # 1. 两阳夹一阴 (Two Yang Clipping One Yin)
    # ================================================================
    o2, o1, o0 = o.shift(2), o.shift(1), o
    c2, c1, c0 = c.shift(2), c.shift(1), c
    v2, v1, v0 = v.shift(2), v.shift(1), v
    h2, l2 = h.shift(2), l.shift(2)
    h1, l1 = h.shift(1), l.shift(1)
    r2, r1, r0 = h2-l2+EPS, h1-l1+EPS, chl

    clip_a = (c2 > o2) & ((c2-o2).abs()/r2 > 0.30)
    clip_b = (c1 < o1) & ((o1-c1).abs()/r1 < 0.20)
    clip_c = (c0 > o0) & ((c0-o0).abs()/r0 > 0.30)
    clip_d = (v1 < v2) & (v0 > v1)
    clip_e = c0 > (o1 + c1) / 2.0
    clip_f = c0 > ma20
    results['two_yang_clip_yin'] = np.where(
        clip_a & clip_b & clip_c & clip_d & clip_e & clip_f, 1.0, 0.0)

    # ================================================================
    # 2. 芝麻开花 (Sesame Bloom)
    # ================================================================
    sesame_a = is_bull.rolling(5, min_periods=5).sum() == 5
    daily_ret = (c / c.shift(1).replace(0, np.nan) - 1.0).abs()
    sesame_b = daily_ret < 0.03
    sesame_c = (c > c.shift(1)).rolling(4, min_periods=4).sum() == 4
    # vol stability: 5-day std / 5-day mean < 0.5
    vol_std5 = v.rolling(5, min_periods=5).std()
    vol_cv = vol_std5 / vol_ma5.replace(0, np.nan)
    sesame_d = vol_cv < 0.5
    sesame_e = (c > ma5) & (ma5 > ma10)
    results['sesame_bloom'] = np.where(
        sesame_a & sesame_b & sesame_c & sesame_d & sesame_e, 1.0, 0.0)

    # ================================================================
    # 3. 冷空气 (Cold Air)
    # ================================================================
    cold_a = ma5.shift(5) > ma10.shift(5)
    cold_b = ma_bind <= 0.03
    cold_c = v < vol_ma20 * 0.60
    cold_d = is_bear | (c < ma5)
    results['cold_air'] = np.where(cold_a & cold_b & cold_c & cold_d, -1.0, 0.0)

    # ================================================================
    # 4. 量顶天立地 (Volume Sky-Pillar)
    # ================================================================
    pillar_a = v > vol_ma20 * 5.0
    pillar_bottom = (c < ma60 * 0.85) & is_bull & (body_pct > 0.60)
    pillar_top    = (c > ma60 * 1.30) & is_bear & (body_pct > 0.60)
    # Mutual exclusion: bottom has priority; top only fires if bottom is False
    pillar_out = np.zeros_like(c.values, dtype=float)
    pillar_out = np.where(pillar_a.values & pillar_bottom.values, 1.0, pillar_out)
    pillar_out = np.where(pillar_a.values & pillar_top.values & ~pillar_bottom.values, -1.0, pillar_out)
    results['volume_pillar'] = pd.DataFrame(pillar_out, index=c.index, columns=c.columns)

    # ================================================================
    # 5. 断头铡刀 (Guillotine)
    # ================================================================
    guil_a = is_bear & (body_pct > 0.65)
    guil_b = (c < ma5) & (c < ma10) & (c < ma20) & (c.shift(1) > ma5.shift(1))
    guil_c = v > vol_ma5 * 1.50
    guil_d = c > ma60 * 1.15
    results['guillotine'] = np.where(guil_a & guil_b & guil_c & guil_d, -1.0, 0.0)

    # ================================================================
    # 6. 青龙取水 (Dragon Drinks Water)
    # ================================================================
    dragon_a = ma60 > ma60.shift(10)
    dragon_b = (l < ma20 * 1.01) & (c > ma20)
    dragon_c = v < vol_ma5 * 0.80
    dragon_d = is_bull
    results['dragon_drinks'] = np.where(
        dragon_a & dragon_b & dragon_c & dragon_d, 1.0, 0.0)

    # ================================================================
    # 7. 梅开二度 (Second Bloom) — state-tracking via ffill
    # ================================================================
    base_signal = (results['two_yang_clip_yin'] > 0)
    signal_low_arr = np.where(base_signal, l.values.copy(), np.nan)
    signal_low = pd.DataFrame(signal_low_arr, index=l.index, columns=l.columns)
    recent_low = signal_low.ffill(limit=30)

    plum_a = (recent_low.notna() & (recent_low > 0))
    plum_b = results.get('_plum_low_check', True)
    plum_c = c > h.shift(1).rolling(20, min_periods=10).max()
    plum_d = v > vol_ma10 * 1.50
    results['second_bloom'] = np.where(plum_a & plum_c & plum_d, 1.0, 0.0)

    # ================================================================
    # 8. 黑马草上飞 (Dark Horse Gliding)
    # ================================================================
    stick_to_ma5 = (abs(c - ma5.values) / (ma5.values + EPS) < 0.015)
    # Convert to DataFrame for rolling
    stick_df = pd.DataFrame(stick_to_ma5, index=c.index, columns=c.columns)
    horse_a = stick_df.rolling(10, min_periods=10).sum() == 10
    horse_b = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60) & (ma5 / ma20.replace(0, np.nan) > 1.03)
    horse_c = vol_ma5 > vol_ma20 * 1.10
    horse_d = is_bull.rolling(5, min_periods=5).sum() >= 4
    results['dark_horse'] = np.where(horse_a & horse_b & horse_c & horse_d, 1.0, 0.0)

    # ================================================================
    # 9. 季线穿越 (Quarterly Line Breakthrough)
    # ================================================================
    qtr_a = (c > ma60) & (c.shift(1) < ma60.shift(1))
    qtr_b = v > vol_ma20 * 1.50
    qtr_c = ma60 > ma60.shift(10)
    qtr_d = is_bull
    results['quarterly_cross'] = np.where(qtr_a & qtr_b & qtr_c & qtr_d, 1.0, 0.0)

    # Convert all to DataFrames
    for k in list(results.keys()):
        if not isinstance(results[k], pd.DataFrame):
            results[k] = pd.DataFrame(results[k], index=c.index, columns=c.columns).astype(np.float64)

    return results


# ==========================================================================
# School integration: add these to existing _compute_school_tang
# ==========================================================================
def tang_remaining_signal_score(indicators: dict) -> tuple:
    """Legacy wrapper — returns (score_delta, reasons_list)."""
    b, bear, rb, rbe = tang_remaining_signal_score_split(indicators)
    return b - bear, rb + rbe


def tang_remaining_signal_score_split(indicators: dict) -> tuple:
    """
    Compute additional score from 9 remaining patterns.
    Returns (score_bull, score_bear, reasons_bull, reasons_bear).
    Each polarity tracked independently for upstream conflict detection.
    """
    try:
        from tang_remaining_features import compute_remaining_tnt_features
        import pandas as pd
        cp = indicators.get('current_price', 0) or 0
        o_v = indicators.get('open', cp) or cp
        h_v = indicators.get('high', cp) or cp
        l_v = indicators.get('low', cp) or cp
        v_v = indicators.get('volume', 0) or 0
        o_df = pd.DataFrame({'S': [float(o_v)]})
        h_df = pd.DataFrame({'S': [float(h_v)]})
        l_df = pd.DataFrame({'S': [float(l_v)]})
        c_df = pd.DataFrame({'S': [float(cp)]})
        v_df = pd.DataFrame({'S': [float(v_v)]})
        feats = compute_remaining_tnt_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        for key, weight, label in [
            ('two_yang_clip_yin', 0.20, '两阳夹一阴'),
            ('sesame_bloom',      0.15, '芝麻开花'),
            ('dragon_drinks',     0.20, '青龙取水'),
            ('second_bloom',      0.25, '梅开二度'),
            ('dark_horse',        0.20, '黑马草上飞'),
            ('quarterly_cross',   0.20, '季线穿越'),
        ]:
            if feats[key]['S'].iloc[0] > 0:
                score_bull += weight; reasons_bull.append(label)

        for key, weight, label in [
            ('cold_air',          0.20, '冷空气'),
            ('guillotine',        0.35, '断头铡刀'),
        ]:
            if feats[key]['S'].iloc[0] < 0:
                score_bear += weight; reasons_bear.append(label)

        # Volume pillar bidirectional — split by sign
        vp_val = feats['volume_pillar']['S'].iloc[0]
        if vp_val > 0:
            score_bull += 0.25; reasons_bull.append('量顶天立地(底)')
        elif vp_val < 0:
            score_bear += 0.25; reasons_bear.append('量顶天立地(顶)')

        return score_bull, score_bear, reasons_bull, reasons_bear
    except Exception:
        return 0.0, 0.0, [], []
