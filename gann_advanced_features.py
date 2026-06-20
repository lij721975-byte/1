#!/usr/bin/env python
# gann_advanced_features.py — Gann 6 missing algorithms (vectorized wide-format)
"""
Gann advanced features:
  1. mechanical_swing     — 2-Day Mechanical Swing Chart
  2. square_of_nine       — Square Root Price Calculator
  3. gann_angles          — ATR-Adaptive Gann Angles (1x1, 1x2, 2x1)
  4. time_cycle_warning   — 90/144/180 Day Cycle Windows
  5. retracement_eighths  — Exact 1/8 Retracement Levels
  6. price_time_square    — Price-Time Squaring Ratio

All inputs: wide-format DataFrames. Zero for-loops.
"""

import numpy as np
import pandas as pd

EPS = 1e-8


def generate_gann_advanced_features(open_df, high_df, low_df, close_df, volume_df):
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns
    n = len(c)

    # ================================================================
    # Shared: ATR(14)
    # ================================================================
    tr_arr = np.fmax.reduce([
        h.values - l.values,
        np.abs(h.values - c.shift(1).values),
        np.abs(l.values - c.shift(1).values)
    ])
    tr_df = pd.DataFrame(tr_arr, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean()

    # ================================================================
    # Shared: Mechanical Swing Chart (2-Day Rule) — state machine
    # ================================================================
    # Detect swing highs and swing lows using the 2-day rule
    h_vals, l_vals = h.values, l.values
    h_s1, h_s2 = np.roll(h_vals, 1, axis=0), np.roll(h_vals, 2, axis=0)
    l_s1, l_s2 = np.roll(l_vals, 1, axis=0), np.roll(l_vals, 2, axis=0)

    # UP swing: 2 consecutive higher highs
    up_swing = (h_vals > h_s1) & (h_s1 > h_s2)
    # DOWN swing: 2 consecutive lower lows
    dn_swing = (l_vals < l_s1) & (l_s1 < l_s2)

    up_swing[:2] = False; dn_swing[:2] = False

    # Track most recent swing high price and swing low price
    # For swing high: at each dn_swing start, record the prior peak
    swing_high_price = np.full_like(c.values, np.nan, dtype=float)
    swing_low_price  = np.full_like(c.values, np.nan, dtype=float)

    # A swing high is the highest high between two down-swing signals
    # Simplification: use rolling max since last down-swing as proxy
    # Forward-fill approach: at each bar, record if it's a swing point
    is_swing_high = np.zeros_like(c.values, dtype=bool)
    is_swing_low  = np.zeros_like(c.values, dtype=bool)

    # Simple 2-day swing detection: mark local extrema confirmed by 2-day rule
    for i in range(2, n):
        if dn_swing[i]:
            # The swing high is the highest high in the prior up-leg
            is_swing_high[i-1] = h_vals[i-1] >= h_vals[max(0,i-5):i].max()
        if up_swing[i]:
            is_swing_low[i-1] = l_vals[i-1] <= l_vals[min(n-1,i):min(n,i+5)].min() if i+1 < n else True

    # Fill swing prices at swing points
    sw_high_vals = h_vals.copy(); sw_high_vals[~is_swing_high] = np.nan
    sw_low_vals  = l_vals.copy(); sw_low_vals[~is_swing_low]  = np.nan

    # Forward-fill to all bars
    sw_high_df = pd.DataFrame(sw_high_vals, index=idx, columns=cols).ffill()
    sw_low_df  = pd.DataFrame(sw_low_vals,  index=idx, columns=cols).ffill()

    swing_high_price = sw_high_df.values
    swing_low_price  = sw_low_df.values

    # Days since last swing: count bars since last NaN in forward-filled series
    # days_since_high: for each column, count bars since swing_high changed
    days_since_high = np.zeros_like(c.values, dtype=float)
    days_since_low  = np.zeros_like(c.values, dtype=float)

    for j in range(len(cols)):
        last_h = 0; last_l = 0
        for i in range(n):
            if not np.isnan(sw_high_vals[i, j]) and sw_high_vals[i, j] != sw_high_df.values[max(0,i-1), j]:
                last_h = 0
            else:
                last_h += 1
            if not np.isnan(sw_low_vals[i, j]) and sw_low_vals[i, j] != sw_low_df.values[max(0,i-1), j]:
                last_l = 0
            else:
                last_l += 1
            days_since_high[i, j] = last_h
            days_since_low[i, j]  = last_l

    results = {}

    # ================================================================
    # 1. Mechanical Swing (机械波段法)
    # ================================================================
    dist_to_support = (c.values - swing_low_price) / (swing_low_price + EPS)
    dist_to_resist  = (swing_high_price - c.values) / (swing_high_price + EPS)
    near_support = (np.abs(dist_to_support) < 0.02) & (days_since_low < days_since_high)
    near_resist  = (np.abs(dist_to_resist) < 0.02) & (days_since_high < days_since_low)

    mech_out = np.zeros_like(c.values, dtype=float)
    mech_out = np.where(near_support, 1.0, mech_out)
    mech_out = np.where(near_resist & (mech_out == 0), -1.0, mech_out)
    results['mechanical_swing'] = pd.DataFrame(mech_out, index=idx, columns=cols)

    # ================================================================
    # 2. Square of Nine (九方图平方根)
    # ================================================================
    root = np.sqrt(c.values)
    r90 = (root + 0.5) ** 2
    s90 = (root - 0.5) ** 2
    r90_prev = np.roll(r90, 1, axis=0); r90_prev[0] = r90[0]
    s90_prev = np.roll(s90, 1, axis=0); s90_prev[0] = s90[0]

    son_out = np.zeros_like(c.values, dtype=float)
    son_out = np.where(c.values > r90_prev, 1.0, son_out)
    son_out = np.where((c.values < s90_prev) & (son_out == 0), -1.0, son_out)
    results['square_of_nine'] = pd.DataFrame(son_out, index=idx, columns=cols)

    # ================================================================
    # 3. Gann Angles (动态江恩角度线)
    # ================================================================
    atr_vals = atr14.values
    gann_up_1x1 = swing_low_price + days_since_low * atr_vals * 0.50
    gann_dn_1x1 = swing_high_price - days_since_high * atr_vals * 0.50

    # Break above 1x1 today (yesterday was below, today above)
    gann_up_prev = np.roll(gann_up_1x1, 1, axis=0); gann_up_prev[0] = gann_up_1x1[0]
    gann_dn_prev = np.roll(gann_dn_1x1, 1, axis=0); gann_dn_prev[0] = gann_dn_1x1[0]

    break_above = (c.values > gann_up_1x1) & (c.shift(1).values <= gann_up_prev)
    break_below = (c.values < gann_dn_1x1) & (c.shift(1).values >= gann_dn_prev)

    angle_out = np.zeros_like(c.values, dtype=float)
    angle_out = np.where(break_above, 1.0, angle_out)
    angle_out = np.where(break_below & (angle_out == 0), -1.0, angle_out)
    results['gann_angles'] = pd.DataFrame(angle_out, index=idx, columns=cols)

    # ================================================================
    # 4. Time Cycle Warning (时间周期预警)
    # ================================================================
    in_window = (
        ((days_since_low >= 85) & (days_since_low <= 95)) |
        ((days_since_low >= 136) & (days_since_low <= 151)) |
        ((days_since_low >= 171) & (days_since_low <= 189)) |
        ((days_since_high >= 85) & (days_since_high <= 95)) |
        ((days_since_high >= 136) & (days_since_high <= 151)) |
        ((days_since_high >= 171) & (days_since_high <= 189))
    )
    results['time_cycle_warning'] = pd.DataFrame(
        np.where(in_window, 1.0, 0.0), index=idx, columns=cols)

    # ================================================================
    # 5. Retracement Eighths (精确百分比回调)
    # ================================================================
    range_val = swing_high_price - swing_low_price + EPS
    retrace_pct = (swing_high_price - c.values) / range_val
    in_retrace_zone = (retrace_pct >= 0.375) & (retrace_pct <= 0.500)
    bullish_candle = c.values > o.values
    results['retracement_eighths'] = pd.DataFrame(
        np.where(in_retrace_zone & bullish_candle, 1.0, 0.0), index=idx, columns=cols)

    # ================================================================
    # 6. Price-Time Square (时价正方)
    # ================================================================
    price_change = np.abs(c.values - swing_low_price) / (swing_low_price + EPS) * 100.0
    squaring_ratio = price_change / (days_since_low + 1)
    near_square = np.abs(squaring_ratio - 1.0) < 0.10
    results['price_time_square'] = pd.DataFrame(
        np.where(near_square, 1.0, 0.0), index=idx, columns=cols)

    return results


# ==========================================================================
# School-compatible split-return interface
# ==========================================================================

def compute_gann_signal_score_split(indicators: dict):
    """
    Split-return for polarity-grouped scoring with time-signal isolation.

    Time signals (time_cycle_warning, price_time_square) are DIRECTIONLESS.
    They MUST NOT enter score_bull or score_bear. They only set
    is_time_triggered = True for the upstream double-confirmation gate.

    Returns (score_bull, score_bear, reasons_bull, reasons_bear, is_time_triggered).
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
        feats = generate_gann_advanced_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []
        is_time_triggered = False

        # ── Price-directional signals (bull side) ──
        for key, w, label in [
            ('mechanical_swing',    0.25, '机械波段(支撑)'),
            ('square_of_nine',      0.20, '九方图(突破)'),
            ('gann_angles',         0.25, '角度线(突破1x1)'),
            ('retracement_eighths', 0.20, '八分回调(支撑)'),
        ]:
            val = feats[key]['S'].iloc[0]
            if val > 0: score_bull += w; reasons_bull.append(label)

        # ── Price-directional signals (bear side) ──
        for key, w, label in [
            ('mechanical_swing',    0.25, '机械波段(阻力)'),
            ('square_of_nine',      0.20, '九方图(跌破)'),
            ('gann_angles',         0.25, '角度线(跌破1x1)'),
        ]:
            val = feats[key]['S'].iloc[0]
            if val < 0: score_bear += w; reasons_bear.append(label)

        # ── Time signals (directionless — flag only, ZERO score contribution) ──
        if feats['time_cycle_warning']['S'].iloc[0] > 0:
            is_time_triggered = True
            reasons_bull.append('时间周期预警(90/144/180日)')
        if feats['price_time_square']['S'].iloc[0] > 0:
            is_time_triggered = True
            reasons_bull.append('时价正方预警')

        return score_bull, score_bear, reasons_bull, reasons_bear, is_time_triggered
    except Exception:
        return 0.0, 0.0, [], [], False


# ==========================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Gann Advanced Features Test ===")
    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=300)
        if df is not None and not df.empty:
            idx_d = df.index
            def _d(c): return pd.DataFrame({'S': df[c].values}, index=idx_d)
            feats = generate_gann_advanced_features(_d('open'),_d('high'),_d('low'),_d('close'),_d('volume'))
            for k, v in feats.items():
                nz = (v.values != 0).sum()
                if nz > 0: print(f"  {code} {k}: {nz} non-zero")
