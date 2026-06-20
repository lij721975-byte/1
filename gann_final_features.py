#!/usr/bin/env python
# gann_final_features.py — Gann final 3 features (vectorized wide-format)
"""
Gann final features:
  1. 3d_swing          — 3-Day Mechanical Swing Chart
  2. gann_fan           — Full Gann Fan Lines (1x1 / 2x1)
  3. seasonal_windows   — Seasonal + Anniversary Date Windows

All inputs: wide-format DataFrames. Zero for-loops for core logic.
"""

import numpy as np
import pandas as pd

EPS = 1e-8


def compute_gann_final_features(open_df, high_df, low_df, close_df, volume_df):
    o, h, l, c = open_df, high_df, low_df, close_df
    idx, cols = c.index, c.columns
    n = len(c)
    h_vals, l_vals, c_vals = h.values, l.values, c.values

    # ================================================================
    # Shared: ATR(14)
    # ================================================================
    tr_arr = np.fmax.reduce([
        h_vals - l_vals,
        np.abs(h_vals - np.roll(c_vals, 1, axis=0)),
        np.abs(l_vals - np.roll(c_vals, 1, axis=0))
    ])
    tr_arr[0] = h_vals[0] - l_vals[0]
    tr_df = pd.DataFrame(tr_arr, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean()

    # ================================================================
    # Shared: 2-day swing state (from previous implementation logic)
    # ================================================================
    # Detect 2-day up/down swings
    h_s1 = np.roll(h_vals, 1, axis=0); h_s1[0] = h_vals[0]
    h_s2 = np.roll(h_vals, 2, axis=0); h_s2[:2] = h_vals[:2]
    l_s1 = np.roll(l_vals, 1, axis=0); l_s1[0] = l_vals[0]
    l_s2 = np.roll(l_vals, 2, axis=0); l_s2[:2] = l_vals[:2]

    up_2d = (h_vals > h_s1) & (h_s1 > h_s2)
    dn_2d = (l_vals < l_s1) & (l_s1 < l_s2)
    up_2d[:2] = False; dn_2d[:2] = False

    # Track swing high/low via ffill
    sw_high_2d = np.full_like(c_vals, np.nan); sw_low_2d = np.full_like(c_vals, np.nan)
    sw_high_2d[2:] = np.where(dn_2d[2:], h_s1[2:], np.nan)
    sw_low_2d[2:]  = np.where(up_2d[2:], l_s1[2:], np.nan)

    sw_high_2d_df = pd.DataFrame(sw_high_2d, index=idx, columns=cols).ffill()
    sw_low_2d_df  = pd.DataFrame(sw_low_2d,  index=idx, columns=cols).ffill()
    sw_high_2d = sw_high_2d_df.values; sw_low_2d = sw_low_2d_df.values

    # Days since swing
    days_since_high_2d = np.zeros_like(c_vals, dtype=float)
    days_since_low_2d  = np.zeros_like(c_vals, dtype=float)
    for j in range(len(cols)):
        cnt_h, cnt_l = 0, 0
        prev_h, prev_l = sw_high_2d[0,j], sw_low_2d[0,j]
        for i in range(n):
            if not np.isnan(sw_high_2d[i,j]) and sw_high_2d[i,j] != prev_h:
                cnt_h = 0; prev_h = sw_high_2d[i,j]
            else: cnt_h += 1
            if not np.isnan(sw_low_2d[i,j]) and sw_low_2d[i,j] != prev_l:
                cnt_l = 0; prev_l = sw_low_2d[i,j]
            else: cnt_l += 1
            days_since_high_2d[i,j] = cnt_h
            days_since_low_2d[i,j]  = cnt_l

    results = {}

    # ================================================================
    # 1. 3-Day Swing (三波段法)
    # ================================================================
    h_s3 = np.roll(h_vals, 3, axis=0); h_s3[:3] = h_vals[:3]
    l_s3 = np.roll(l_vals, 3, axis=0); l_s3[:3] = l_vals[:3]
    c_s3 = np.roll(c_vals, 3, axis=0); c_s3[:3] = c_vals[:3]

    # 3-day up: 3 consecutively higher highs
    up_3d = (h_vals > h_s1) & (h_s1 > h_s2) & (h_s2 > h_s3)
    # 3-day down: 3 consecutively lower lows
    dn_3d = (l_vals < l_s1) & (l_s1 < l_s2) & (l_s2 < l_s3)
    up_3d[:3] = False; dn_3d[:3] = False

    # Swing high: prior peak when 3d down starts
    sw_high_3d = np.full_like(c_vals, np.nan)
    sw_low_3d  = np.full_like(c_vals, np.nan)
    sw_high_3d[3:] = np.where(dn_3d[3:], np.maximum.reduce([h_s1[3:], h_s2[3:]]), np.nan)
    sw_low_3d[3:]  = np.where(up_3d[3:], np.minimum.reduce([l_s1[3:], l_s2[3:]]), np.nan)

    sw_high_3d_df = pd.DataFrame(sw_high_3d, index=idx, columns=cols).ffill()
    sw_low_3d_df  = pd.DataFrame(sw_low_3d,  index=idx, columns=cols).ffill()
    sw_high_3d = sw_high_3d_df.values; sw_low_3d = sw_low_3d_df.values

    # Direction: up if last swing was a low (uptrend), down if last swing was a high
    last_swing_high_idx = np.zeros_like(c_vals, dtype=float)
    last_swing_low_idx  = np.zeros_like(c_vals, dtype=float)
    for j in range(len(cols)):
        lh, ll = 0, 0
        for i in range(n):
            if not np.isnan(sw_high_3d[i,j]): lh = i
            if not np.isnan(sw_low_3d[i,j]):  ll = i
            last_swing_high_idx[i,j] = lh
            last_swing_low_idx[i,j]  = ll

    uptrend_3d = last_swing_low_idx > last_swing_high_idx
    downtrend_3d = last_swing_high_idx > last_swing_low_idx

    near_support_3d = np.abs(c_vals - sw_low_3d) / (sw_low_3d + EPS) < 0.03
    near_resist_3d  = np.abs(sw_high_3d - c_vals) / (sw_high_3d + EPS) < 0.03

    swing3_out = np.zeros_like(c_vals, dtype=float)
    swing3_out = np.where(near_support_3d & uptrend_3d, 1.0, swing3_out)
    swing3_out = np.where(near_resist_3d & downtrend_3d & (swing3_out == 0), -1.0, swing3_out)
    results['3d_swing'] = pd.DataFrame(swing3_out, index=idx, columns=cols)

    # ================================================================
    # 2. Gann Fan Lines (江恩扇形线)
    # ================================================================
    atr_v = atr14.values
    gann_1x1_up = sw_low_2d + days_since_low_2d * atr_v * 0.50
    gann_2x1_up = sw_low_2d + days_since_low_2d * atr_v * 0.25

    gann_1x1_prev = np.roll(gann_1x1_up, 1, axis=0); gann_1x1_prev[0] = gann_1x1_up[0]
    gann_2x1_prev = np.roll(gann_2x1_up, 1, axis=0); gann_2x1_prev[0] = gann_2x1_up[0]

    c_prev = np.roll(c_vals, 1, axis=0); c_prev[0] = c_vals[0]

    break_1x1 = (c_vals > gann_1x1_up) & (c_prev <= gann_1x1_prev)
    break_2x1 = (c_vals < gann_2x1_up) & (c_prev >= gann_2x1_prev)

    fan_out = np.zeros_like(c_vals, dtype=float)
    fan_out = np.where(break_1x1, 1.0, fan_out)
    fan_out = np.where(break_2x1 & (fan_out == 0), -1.0, fan_out)
    results['gann_fan'] = pd.DataFrame(fan_out, index=idx, columns=cols)

    # ================================================================
    # 3. Seasonal Windows (季节性与周年窗口)
    # ================================================================
    m_arr = np.array([d.month for d in idx])[:, None]
    d_arr = np.array([d.day   for d in idx])[:, None]
    m_grid = np.tile(m_arr, (1, len(cols)))
    d_grid = np.tile(d_arr, (1, len(cols)))

    spring = (m_grid == 3)  & (d_grid >= 16) & (d_grid <= 26)
    summer = (m_grid == 6)  & (d_grid >= 16) & (d_grid <= 26)
    autumn = (m_grid == 9)  & (d_grid >= 17) & (d_grid <= 27)
    winter = (m_grid == 12) & (d_grid >= 16) & (d_grid <= 26)
    monthly = (d_grid >= 1) & (d_grid <= 2) | (d_grid >= 14) & (d_grid <= 16)

    # 52-week anniversary: days since 52w high
    high_252 = h.shift(1).rolling(252, min_periods=100).max()
    at_52w_high = (h == high_252)
    days_since_52w = at_52w_high.astype(int).replace(0, np.nan)
    # Count bars since last True via cumsum trick
    for j in range(len(cols)):
        cnt = 999
        for i in range(n):
            if not np.isnan(days_since_52w.values[i, j]):
                cnt = 0
            else:
                cnt += 1
            days_since_52w.values[i, j] = cnt

    anniv_mask = (days_since_52w.values >= 242) & (days_since_52w.values <= 262)

    seasonal = spring | summer | autumn | winter | monthly | anniv_mask
    results['seasonal_windows'] = pd.DataFrame(
        np.where(seasonal, 1.0, 0.0), index=idx, columns=cols)

    return results


# ==========================================================================
# School split-return interface
# ==========================================================================

def compute_gann_final_score_split(indicators: dict):
    """
    Returns (score_bull, score_bear, reasons_bull, reasons_bear,
             has_price_signal, has_time_signal) for Gann Double Confirmation.
    """
    try:
        import pandas as pd
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
        feats = compute_gann_final_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []
        has_price = False
        has_time  = False

        # 3-day swing (price + direction)
        s3 = feats['3d_swing']['S'].iloc[0]
        if s3 > 0:   score_bull += 0.20; reasons_bull.append('3日波段(支撑)'); has_price = True
        elif s3 < 0: score_bear += 0.20; reasons_bear.append('3日波段(阻力)'); has_price = True

        # Gann fan (price)
        gf = feats['gann_fan']['S'].iloc[0]
        if gf > 0:   score_bull += 0.25; reasons_bull.append('扇形突破1x1'); has_price = True
        elif gf < 0: score_bear += 0.25; reasons_bear.append('扇形跌破2x1'); has_price = True

        # Seasonal (time only — no direction)
        if feats['seasonal_windows']['S'].iloc[0] > 0:
            has_time = True; reasons_bull.append('江恩时间窗口')

        return score_bull, score_bear, reasons_bull, reasons_bear, has_price, has_time
    except Exception:
        return 0.0, 0.0, [], [], False, False
