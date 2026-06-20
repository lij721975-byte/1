#!/usr/bin/env python
# wyckoff_advanced_features.py — Wyckoff/VSA 11 missing patterns (fully vectorized)
"""
Wyckoff/VSA advanced feature factory. All computations are wide-format,
pure Pandas/NumPy vectorized — ZERO for-loops, zero apply/iterrows,
zero shift(-1) forward-looking bias.

Patterns:
  M1  — Jump Across the Creek (JAC / 跳跃小溪)
  M2  — Ice Breaking (跌破冰层)
  M3  — Shakeout (震仓) — 2-day confirmation, T-1 shakeout + T confirmation
  M4  — Backup to Creek (BUEC / 回踩确认)
  M5  — No Supply Bar (无供应柱)
  M6  — No Demand Bar (无需求柱)
  M7  — Stopping Volume Enhanced (增强停止行为)
  M8  — Absorption (主力吸筹承接)
  M9  — Wyckoff Coil (弹簧压缩)
  M10 — Phase Transition Signal (阶段过渡信号)
  M11 — VDB / VIB / VSB Bar Classification (柱线结构化分类)

Interface:
  compute_wyckoff_advanced_features(open_df, high_df, low_df, close_df, volume_df, phase_df=None)
  → dict of DataFrames
"""

import numpy as np
import pandas as pd

EPS = 1e-8


# ======================================================================
# Main entry point
# ======================================================================

def compute_wyckoff_advanced_features(open_df, high_df, low_df, close_df, volume_df,
                                      phase_df=None):
    """
    Compute all 11 Wyckoff/VSA advanced patterns.

    Parameters
    ----------
    open_df, high_df, low_df, close_df, volume_df : pd.DataFrame
        Wide-format (index=datetime, columns=stocks).
    phase_df : pd.DataFrame, optional
        Wide-format phase labels from wyckoff_phase_detection.
        Values: 'accumulation', 'markup', 'distribution', 'markdown', 'transition'.
        If None, M10 returns zero signals (transition detection skipped).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys like 'wyckoff_jac_signal', 'wyckoff_jac_score', etc.
        Every DataFrame has index=close_df.index, columns=close_df.columns.
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

    # ==================================================================
    # SECTION 0 — Shared precomputations (computed once, consumed by all)
    # ==================================================================

    # ---- ATR(14) ----
    tr_vals = np.fmax.reduce([
        h.values - l.values,
        np.abs(h.values - np.roll(c.values, 1, axis=0)),
        np.abs(l.values - np.roll(c.values, 1, axis=0)),
    ])
    tr_vals[0] = h.values[0] - l.values[0]
    tr_df = pd.DataFrame(tr_vals, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean()
    atr14_arr = atr14.values

    # ---- Trading Range (60-day) — shifted to avoid look-ahead ----
    range_high_60 = h.rolling(60, min_periods=20).max().shift(1)
    range_low_60 = l.rolling(60, min_periods=20).min().shift(1)
    rh60 = range_high_60.values
    rl60 = range_low_60.values
    range_width_60 = (rh60 - rl60) / (rl60 + EPS)

    # ---- Trading Range (20-day) — for position checks ----
    range_high_20 = h.rolling(20, min_periods=10).max().shift(1)
    range_low_20 = l.rolling(20, min_periods=10).min().shift(1)
    rh20 = range_high_20.values
    rl20 = range_low_20.values

    # ---- Volume MAs ----
    vol_ma20 = v.rolling(20, min_periods=10).mean()
    vol_ma60 = v.rolling(60, min_periods=30).mean()
    vm20 = vol_ma20.values
    vm60 = vol_ma60.values

    # ---- Price MAs ----
    ma5 = c.rolling(5, min_periods=3).mean()
    ma20 = c.rolling(20, min_periods=10).mean()
    ma60 = c.rolling(60, min_periods=30).mean()

    # ---- OBV (On-Balance Volume) ----
    close_diff_sign = np.sign(c.values - np.roll(c.values, 1, axis=0))
    close_diff_sign[0] = 0
    obv_vals = np.cumsum(close_diff_sign * v.values, axis=0)
    obv_df = pd.DataFrame(obv_vals, index=idx, columns=cols)

    # ---- K-line geometry (wide-format arrays) ----
    o_arr, h_arr, l_arr, c_arr, v_arr = o.values, h.values, l.values, c.values, v.values
    spread_arr = h_arr - l_arr
    body_arr = np.abs(c_arr - o_arr)
    upper_wick_arr = h_arr - np.maximum(o_arr, c_arr)
    lower_wick_arr = np.minimum(o_arr, c_arr) - l_arr
    close_pos_arr = (c_arr - l_arr) / (spread_arr + EPS)  # 0=low, 1=high

    # ---- Convenience: shift-1 arrays ----
    def _s1(arr):
        """shift(1) wrapper — returns a view shifted by 1 row (row 0 unchanged)."""
        out = np.roll(arr, 1, axis=0)
        out[0] = arr[0]
        return out

    c_s1 = _s1(c_arr); o_s1 = _s1(o_arr); h_s1 = _s1(h_arr); l_s1 = _s1(l_arr)
    v_s1 = _s1(v_arr); spread_s1 = _s1(spread_arr)
    lower_wick_s1 = _s1(lower_wick_arr)
    close_pos_s1 = _s1(close_pos_arr)

    results = {}

    # ════════════════════════════════════════════════════════════════
    # M1 — Jump Across the Creek (JAC / 跳跃小溪)
    # ════════════════════════════════════════════════════════════════
    # A: trading range structure — range_width < 25%
    jac_a = range_width_60 < 0.25
    # B: breakout — close > range_high_60
    jac_b = c_arr > rh60
    # C: volume surge — vol > vol_ma20 * 1.5
    jac_c = v_arr > vm20 * 1.5
    # D: closing strength — bullish candle closing in upper 60%
    jac_d = (c_arr > o_arr) & (close_pos_arr > 0.60)
    # E: prior test — in last 20 bars, close reached >= 95% of range_high
    close_prev_20_high = pd.DataFrame(c_arr, index=idx, columns=cols).shift(1).rolling(20).max().values
    jac_e = close_prev_20_high >= rh60 * 0.95

    jac_score = (
        np.where(jac_a & ~np.isnan(rh60), 0.15, 0.0) +
        np.where(jac_b & ~np.isnan(rh60), 0.30, 0.0) +
        np.where(jac_c, 0.25, 0.0) +
        np.where(jac_d, 0.15, 0.0) +
        np.where(jac_e, 0.15, 0.0)
    )
    vol_bonus_jac = np.minimum(v_arr / (vm20 * 1.5 + EPS), 2.0)
    jac_score *= vol_bonus_jac
    jac_score = np.clip(jac_score, 0.0, 1.0)
    jac_signal = (jac_score >= 0.70).astype(float)

    results['wyckoff_jac_signal'] = pd.DataFrame(jac_signal, index=idx, columns=cols)
    results['wyckoff_jac_score'] = pd.DataFrame(jac_score, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M2 — Ice Breaking (跌破冰层)
    # ════════════════════════════════════════════════════════════════
    ice_a = range_width_60 < 0.25
    ice_b = c_arr < rl60
    ice_c = v_arr > vm20 * 1.5
    ice_d = (h_arr - c_arr) / (spread_arr + EPS) > 0.60  # close in lower 40%
    # E: prior LPSY test — in last 20 bars, low reached <= 105% of range_low
    low_prev_20_low = pd.DataFrame(l_arr, index=idx, columns=cols).shift(1).rolling(20).min().values
    ice_e = low_prev_20_low <= rl60 * 1.05

    ice_score = (
        np.where(ice_a & ~np.isnan(rl60), 0.15, 0.0) +
        np.where(ice_b & ~np.isnan(rl60), 0.30, 0.0) +
        np.where(ice_c, 0.25, 0.0) +
        np.where(ice_d, 0.15, 0.0) +
        np.where(ice_e, 0.15, 0.0)
    )
    vol_bonus_ice = np.minimum(v_arr / (vm20 * 1.5 + EPS), 2.0)
    ice_score *= vol_bonus_ice
    ice_score = np.clip(ice_score, 0.0, 1.0)
    ice_signal = (ice_score >= 0.70).astype(float)

    results['wyckoff_ice_breaking_signal'] = pd.DataFrame(ice_signal, index=idx, columns=cols)
    results['wyckoff_ice_breaking_score'] = pd.DataFrame(ice_score, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M3 — Shakeout (震仓)
    #      Day T-1 = shakeout day, Day T = confirmation day.
    #      ZERO shift(-1): we stand at T and look back at T-1.
    # ════════════════════════════════════════════════════════════════
    # ---- Day T-1 (shakeout day) conditions ----
    shk_a1 = l_s1 < rl60 * 0.97                     # penetration depth >= 3%
    shk_a2 = v_s1 > vm20 * 1.8                       # panic volume
    shk_a3 = c_s1 > l_s1 + spread_s1 * 0.50          # intraday reversal
    shk_a4 = lower_wick_s1 / (spread_s1 + EPS) > 0.50 # lower wick > 50%

    # ---- Day T (confirmation day) conditions ----
    shk_b1 = c_arr > rl60                            # close back inside range
    shk_b2 = v_arr > vm20 * 1.2                       # sustained volume
    shk_b3 = c_arr > o_arr                            # bullish close

    # ---- Additional validations (all pre-computed at T-1) ----
    # C1: pre-accumulation zone (vol_20d > vol_60d * 0.90 before shakeout)
    vol_20d_pre = pd.DataFrame(v_s1, index=idx, columns=cols).rolling(20, min_periods=10).mean().values
    vol_60d_pre = pd.DataFrame(v_s1, index=idx, columns=cols).rolling(60, min_periods=30).mean().values
    shk_c1 = vol_20d_pre > vol_60d_pre * 0.90
    # C2: low volume in 5 days before shakeout
    low_vol_5d_pre = pd.DataFrame(v_s1, index=idx, columns=cols).rolling(5, min_periods=3).min().values
    shk_c2 = low_vol_5d_pre < vm20 * 0.60
    # C3: price in lower half of range before shakeout
    range_mid = (rh60 + rl60) / 2.0
    shk_c3 = c_s1 < range_mid

    shakeout_score = (
        np.where(shk_a1 & ~np.isnan(rl60), 0.15, 0.0) +
        np.where(shk_a2 & ~np.isnan(vm20), 0.20, 0.0) +
        np.where(shk_a3, 0.15, 0.0) +
        np.where(shk_a4, 0.10, 0.0) +
        np.where(shk_b1 & ~np.isnan(rl60), 0.15, 0.0) +
        np.where(shk_b2, 0.10, 0.0) +
        np.where(shk_b3, 0.05, 0.0) +
        np.where(shk_c1 & ~np.isnan(vol_20d_pre), 0.05, 0.0) +
        np.where(shk_c2 & ~np.isnan(vm20), 0.03, 0.0) +
        np.where(shk_c3 & ~np.isnan(range_mid), 0.02, 0.0)
    )
    vol_bonus_shk = np.minimum(v_s1 / (vm20 * 1.8 + EPS), 1.5)
    shakeout_score *= vol_bonus_shk
    shakeout_score = np.clip(shakeout_score, 0.0, 1.0)
    shakeout_signal = (shakeout_score >= 0.75).astype(float)

    results['wyckoff_shakeout_signal'] = pd.DataFrame(shakeout_signal, index=idx, columns=cols)
    results['wyckoff_shakeout_score'] = pd.DataFrame(shakeout_score, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M4 — Backup to Creek (BUEC / 回踩确认)
    #      Requires JAC signal in the window [T-30, T-10].
    # ════════════════════════════════════════════════════════════════
    # A: prior JAC within 10-30 days ago
    jac_df = results['wyckoff_jac_signal']
    jac_10_to_30 = (jac_df.shift(10)
                    .rolling(20, min_periods=1)
                    .max()
                    .values) > 0
    # B: pullback reaches creek level (±2%)
    buec_b = l_arr < rh60 * 1.02
    # C: low volume pullback
    buec_c = v_arr < vm20 * 0.65
    # D: support holds — close recovers
    buec_d = (c_arr > o_arr) | (close_pos_arr > 0.55)
    # E: price still above MA20
    buec_e = c_arr > ma20.values

    buec_score = (
        np.where(jac_10_to_30, 0.25, 0.0) +
        np.where(buec_b & ~np.isnan(rh60), 0.20, 0.0) +
        np.where(buec_c, 0.25, 0.0) +
        np.where(buec_d, 0.15, 0.0) +
        np.where(buec_e & ~np.isnan(ma20.values), 0.15, 0.0)
    )
    # Low-volume intensity bonus: the lower the volume, the higher the confidence
    vol_deficit = np.maximum(0.0, 0.65 - v_arr / (vm20 + EPS))
    buec_score *= (1.0 + vol_deficit * 2.0)
    buec_score = np.clip(buec_score, 0.0, 1.0)
    buec_signal = (buec_score >= 0.65).astype(float)

    results['wyckoff_buec_signal'] = pd.DataFrame(buec_signal, index=idx, columns=cols)
    results['wyckoff_buec_score'] = pd.DataFrame(buec_score, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M5 — No Supply Bar (无供应柱)
    # ════════════════════════════════════════════════════════════════
    ns_a = v_arr < vm20 * 0.55                                # ultra-low volume
    ns_b = spread_arr < atr14_arr * 0.70                       # narrow spread
    ns_c = body_arr / (spread_arr + EPS) < 0.40                # small body
    # D: not a strong bullish bar (close - open < atr * 0.30)
    ns_d = (c_arr - o_arr) < atr14_arr * 0.30
    # E: in correction context (close < MA20 or close < 5-day-ago close)
    c_5d_ago = pd.DataFrame(c_arr, index=idx, columns=cols).shift(5).values
    ns_e = (c_arr < ma20.values) | (c_arr < c_5d_ago)

    ns_score = np.zeros_like(c_arr, dtype=float)
    ns_score = np.where(ns_a & ~np.isnan(vm20), 0.30, ns_score)
    ns_score = np.where(ns_b & ~np.isnan(atr14_arr), 0.25, ns_score)
    ns_score = np.where(ns_c, 0.20, ns_score)
    ns_score = np.where(ns_d, 0.15, ns_score)
    ns_score = np.where(ns_e, 0.10, ns_score)
    ns_score = np.clip(ns_score, 0.0, 1.0)

    ns_strong = (ns_score >= 0.85).astype(float)
    ns_standard = ((ns_score >= 0.65) & (ns_score < 0.85)).astype(float)
    ns_weak = ((ns_score >= 0.45) & (ns_score < 0.65)).astype(float)
    ns_signal = (ns_score >= 0.45).astype(float)

    results['wyckoff_no_supply_signal'] = pd.DataFrame(ns_signal, index=idx, columns=cols)
    results['wyckoff_no_supply_score'] = pd.DataFrame(ns_score, index=idx, columns=cols)
    results['wyckoff_no_supply_grade'] = pd.DataFrame(
        np.where(ns_strong, 3.0, np.where(ns_standard, 2.0, np.where(ns_weak, 1.0, 0.0))),
        index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M6 — No Demand Bar (无需求柱)
    # ════════════════════════════════════════════════════════════════
    nd_a = v_arr < vm20 * 0.55                               # ultra-low volume
    nd_b = spread_arr < atr14_arr * 0.70                      # narrow spread
    nd_c = body_arr / (spread_arr + EPS) < 0.40               # small body
    # D: not a strong bearish bar
    nd_d = (o_arr - c_arr) < atr14_arr * 0.30
    # E: in rally context (close > MA20 or close > 5-day-ago close)
    nd_e = (c_arr > ma20.values) | (c_arr > c_5d_ago)

    nd_score = np.zeros_like(c_arr, dtype=float)
    nd_score = np.where(nd_a & ~np.isnan(vm20), 0.30, nd_score)
    nd_score = np.where(nd_b & ~np.isnan(atr14_arr), 0.25, nd_score)
    nd_score = np.where(nd_c, 0.20, nd_score)
    nd_score = np.where(nd_d, 0.15, nd_score)
    nd_score = np.where(nd_e, 0.10, nd_score)
    nd_score = np.clip(nd_score, 0.0, 1.0)

    nd_strong = (nd_score >= 0.85).astype(float)
    nd_standard = ((nd_score >= 0.65) & (nd_score < 0.85)).astype(float)
    nd_weak = ((nd_score >= 0.45) & (nd_score < 0.65)).astype(float)
    nd_signal = (nd_score >= 0.45).astype(float)

    results['wyckoff_no_demand_signal'] = pd.DataFrame(nd_signal, index=idx, columns=cols)
    results['wyckoff_no_demand_score'] = pd.DataFrame(nd_score, index=idx, columns=cols)
    results['wyckoff_no_demand_grade'] = pd.DataFrame(
        np.where(nd_strong, 3.0, np.where(nd_standard, 2.0, np.where(nd_weak, 1.0, 0.0))),
        index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M7 — Stopping Volume Enhanced (增强停止行为)
    # ════════════════════════════════════════════════════════════════
    sv_a = (v_arr > vm20 * 1.5) & (v_arr > vm60 * 1.3)       # dual-period volume surge
    sv_b = spread_arr > atr14_arr * 1.3                        # wide spread
    sv_c = lower_wick_arr / (spread_arr + EPS) > 0.50          # lower wick dominant
    sv_d = close_pos_arr > 0.50                                # close in upper half
    # E: near low — close below MA20*0.95 OR near 20-day low
    close_20_low = c.rolling(20, min_periods=10).min().values
    sv_e = (c_arr < ma20.values * 0.95) | (c_arr < close_20_low * 1.05)

    # Absorption power
    absorption_power = (v_arr / (vm20 + EPS)) * (lower_wick_arr / (spread_arr + EPS)) * close_pos_arr

    sv_cond_count = (
        sv_a.astype(float) + sv_b.astype(float) + sv_c.astype(float) +
        sv_d.astype(float) + sv_e.astype(float)
    )
    sv_score = np.zeros_like(c_arr, dtype=float)
    sv_score = np.where(sv_cond_count >= 4,
                        np.minimum(0.70 + absorption_power * 0.10, 1.0), sv_score)
    sv_score = np.where((sv_cond_count >= 3) & (sv_score == 0),
                        np.minimum(0.50 + absorption_power * 0.08, 0.85), sv_score)
    sv_signal = (sv_score >= 0.60).astype(float)

    results['wyckoff_stopping_vol_signal'] = pd.DataFrame(sv_signal, index=idx, columns=cols)
    results['wyckoff_stopping_vol_score'] = pd.DataFrame(sv_score, index=idx, columns=cols)
    results['wyckoff_stopping_vol_power'] = pd.DataFrame(absorption_power, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M8 — Absorption (主力吸筹承接) — 10-day rolling window
    # ════════════════════════════════════════════════════════════════
    N = 10
    # A: price stagnation — price change over N days < 3%
    c_n_ago = pd.DataFrame(c_arr, index=idx, columns=cols).shift(N).values
    abs_a = np.abs(c_arr - c_n_ago) / (c_n_ago + EPS) < 0.03
    # B: sustained volume — at least 70% of bars have vol > vm20*0.85
    vol_above_thresh = (v_arr > vm20 * 0.85).astype(float)
    vol_above_count = pd.DataFrame(vol_above_thresh, index=idx, columns=cols).rolling(N, min_periods=N).sum().values
    abs_b = vol_above_count >= N * 0.70
    # C: OBV refuses new low — OBV's N-day min > OBV N-days-ago * 0.95
    obv_n_min = obv_df.rolling(N, min_periods=N).min().values
    obv_n_ago = pd.DataFrame(obv_vals, index=idx, columns=cols).shift(N).values
    abs_c = obv_n_min > obv_n_ago * 0.95
    # D: down-day volume ≈ up-day volume (down-day vol >= up-day vol * 0.90)
    down_mask = c_arr < c_s1
    up_mask = c_arr > c_s1
    down_vol_sum = pd.DataFrame(
        np.where(down_mask, v_arr, 0.0), index=idx, columns=cols
    ).rolling(N, min_periods=N).mean().values
    up_vol_sum = pd.DataFrame(
        np.where(up_mask, v_arr, 0.0), index=idx, columns=cols
    ).rolling(N, min_periods=N).mean().values
    abs_d = down_vol_sum > up_vol_sum * 0.90
    # E: low price context — price was below MA60 at window start
    abs_e = c_n_ago < ma60.values

    abs_score = (
        np.where(abs_a, 0.20, 0.0) +
        np.where(abs_b, 0.30, 0.0) +
        np.where(abs_c, 0.25, 0.0) +
        np.where(abs_d & ~np.isnan(down_vol_sum), 0.15, 0.0) +
        np.where(abs_e & ~np.isnan(ma60.values), 0.10, 0.0)
    )
    # Volume consistency bonus
    vol_consistency = vol_above_count / N
    abs_score *= np.minimum(vol_consistency / (0.70 + EPS), 1.5)
    abs_score = np.clip(abs_score, 0.0, 1.0)
    abs_signal = (abs_score >= 0.65).astype(float)

    results['wyckoff_absorption_signal'] = pd.DataFrame(abs_signal, index=idx, columns=cols)
    results['wyckoff_absorption_score'] = pd.DataFrame(abs_score, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M9 — Wyckoff Coil (弹簧压缩)
    #      W1 = (-20:-10], W2 = (-10:0]
    # ════════════════════════════════════════════════════════════════
    range_width_20 = (rh20 - rl20) / (rl20 + EPS)

    rw_df = pd.DataFrame(range_width_20, index=idx, columns=cols)
    w2_avg = rw_df.rolling(10, min_periods=10).mean().values      # recent 10
    w1_avg = rw_df.shift(10).rolling(10, min_periods=10).mean().values  # older 10

    vol_df = pd.DataFrame(v_arr, index=idx, columns=cols)
    vol_w2_avg = vol_df.rolling(10, min_periods=10).mean().values
    vol_w1_avg = vol_df.shift(10).rolling(10, min_periods=10).mean().values

    coil_a = w2_avg < w1_avg * 0.75                               # range compression
    coil_b = vol_w2_avg < vol_w1_avg * 0.80                        # volume compression
    coil_c = v_arr < vm20 * 0.50                                   # extreme low vol today
    coil_d = spread_arr < atr14_arr * 0.60                          # extreme narrow spread
    coil_e = range_width_20 < 0.20                                  # still in trading range

    coil_score = (
        np.where(coil_a & ~np.isnan(w1_avg), 0.25, 0.0) +
        np.where(coil_b & ~np.isnan(vol_w1_avg), 0.20, 0.0) +
        np.where(coil_c & ~np.isnan(vm20), 0.25, 0.0) +
        np.where(coil_d & ~np.isnan(atr14_arr), 0.20, 0.0) +
        np.where(coil_e & ~np.isnan(rl20), 0.10, 0.0)
    )
    # Ultra-low volume bonus
    ultra_low_vol = v_arr < vm20 * 0.35
    coil_score = np.where(ultra_low_vol & ~np.isnan(vm20), coil_score + 0.10, coil_score)
    coil_score = np.clip(coil_score, 0.0, 1.0)
    coil_signal = (coil_score >= 0.70).astype(float)

    results['wyckoff_coil_signal'] = pd.DataFrame(coil_signal, index=idx, columns=cols)
    results['wyckoff_coil_score'] = pd.DataFrame(coil_score, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M10 — Phase Transition Signal (阶段过渡信号)
    #       Uses external phase_df if provided; otherwise builds a
    #       simplified wide-format phase internally.
    # ════════════════════════════════════════════════════════════════
    if phase_df is not None:
        current_phase = phase_df.values
    else:
        # Simplified internal phase detection (wide-format, vectorized)
        pchg_60 = (c_arr - pd.DataFrame(c_arr, index=idx, columns=cols).shift(60).values) / \
                  (pd.DataFrame(c_arr, index=idx, columns=cols).shift(60).values + EPS)
        vol_ratio_60 = v_arr / (vm60 + EPS)
        rh60_raw = h.rolling(60, min_periods=20).max().values
        rl60_raw = l.rolling(60, min_periods=20).min().values
        rw60_raw = (rh60_raw - rl60_raw) / (rl60_raw + EPS)

        down_vol_avg = pd.DataFrame(
            np.where(c_arr < c_s1, v_arr, 0.0), index=idx, columns=cols
        ).rolling(60, min_periods=20).mean().values
        up_vol_avg = pd.DataFrame(
            np.where(c_arr > c_s1, v_arr, 0.0), index=idx, columns=cols
        ).rolling(60, min_periods=20).mean().values

        acc_mask = (rw60_raw < 0.25) & (down_vol_avg > up_vol_avg * 1.3) & (pchg_60 <= 0.05)
        dist_mask = (rw60_raw < 0.25) & (up_vol_avg > down_vol_avg * 1.3) & (pchg_60 >= -0.05)
        up_mask = pchg_60 > 0.05
        down_mask60 = pchg_60 < -0.05

        current_phase = np.where(up_mask, 'markup',
                         np.where(down_mask60, 'markdown',
                         np.where(acc_mask, 'accumulation',
                         np.where(dist_mask, 'distribution', 'transition'))))

    # Transition: phase changed within last 5 bars
    phase_5d_ago = pd.DataFrame(current_phase, index=idx, columns=cols).shift(5).values
    transition_base = current_phase != phase_5d_ago
    trans_confidence = np.where(transition_base, 0.50, 0.0)

    # Additional confirmation signals (simplified vectorized checks)
    # Accumulation→Transition (B→C): recent Spring or Shakeout
    spring_recent = pd.DataFrame(shakeout_signal, index=idx, columns=cols).rolling(5, min_periods=1).max().values
    prev_acc = (phase_5d_ago == 'accumulation')
    curr_trans = (current_phase == 'transition')
    trans_confidence = np.where(
        prev_acc & curr_trans & (spring_recent > 0), trans_confidence + 0.30, trans_confidence)
    trans_confidence = np.where(
        prev_acc & curr_trans & (v_arr > vm20 * 1.2), trans_confidence + 0.10, trans_confidence)

    # Distribution→Transition (B→C for distribution): recent UTAD or Ice Breaking
    ice_recent = pd.DataFrame(ice_signal, index=idx, columns=cols).rolling(5, min_periods=1).max().values
    prev_dist = (phase_5d_ago == 'distribution')
    trans_confidence = np.where(
        prev_dist & curr_trans & (ice_recent > 0), trans_confidence + 0.30, trans_confidence)

    # Transition→Markup (C→D): recent JAC or SOS
    jac_recent = pd.DataFrame(jac_signal, index=idx, columns=cols).rolling(5, min_periods=1).max().values
    prev_trans = (phase_5d_ago == 'transition')
    curr_up = (current_phase == 'markup')
    trans_confidence = np.where(
        prev_trans & curr_up & (jac_recent > 0), trans_confidence + 0.30, trans_confidence)

    # Transition→Markdown (C→D for distribution)
    curr_down = (current_phase == 'markdown')
    trans_confidence = np.where(
        prev_trans & curr_down & (ice_recent > 0), trans_confidence + 0.30, trans_confidence)

    trans_confidence = np.clip(trans_confidence, 0.0, 1.0)
    trans_signal = (trans_confidence >= 0.60).astype(float)

    results['wyckoff_phase_transition_signal'] = pd.DataFrame(trans_signal, index=idx, columns=cols)
    results['wyckoff_phase_transition_confidence'] = pd.DataFrame(trans_confidence, index=idx, columns=cols)
    results['wyckoff_phase_current'] = pd.DataFrame(current_phase, index=idx, columns=cols)

    # ════════════════════════════════════════════════════════════════
    # M11 — VDB / VIB / VSB Bar Classification (柱线结构化分类)
    #       MUST use np.select — no if/elif chains.
    # ════════════════════════════════════════════════════════════════
    spread_ratio = spread_arr / (atr14_arr + EPS)
    vol_ratio = v_arr / (vm20 + EPS)

    # np.select works on 1D arrays, so we flatten → classify → reshape.
    n_rows, n_cols = c_arr.shape
    sr_flat = spread_ratio.ravel()
    vr_flat = vol_ratio.ravel()
    cp_flat = close_pos_arr.ravel()

    vsb_up_cond = (sr_flat >= 1.2) & (vr_flat >= 1.5) & (cp_flat >= 0.75)
    vsb_down_cond = (sr_flat >= 1.2) & (vr_flat >= 1.5) & (cp_flat <= 0.25)
    vsb_neutral_cond = (sr_flat >= 1.2) & (vr_flat >= 1.5) & (cp_flat > 0.25) & (cp_flat < 0.75)

    vub_cond = (sr_flat >= 1.0) & (sr_flat < 1.2) & (vr_flat >= 1.2) & (cp_flat >= 0.60) | \
               (sr_flat >= 1.0) & (vr_flat >= 1.2) & (vr_flat < 1.5) & (cp_flat >= 0.60)
    # Simplified: VUB = spread>=1.0 & vol>=1.2 & close_pos>=0.60 & NOT VSB
    vub_cond = (sr_flat >= 1.0) & (vr_flat >= 1.2) & (cp_flat >= 0.60) & \
               ~((sr_flat >= 1.2) & (vr_flat >= 1.5))
    vdb_cond = (sr_flat >= 1.0) & (vr_flat >= 1.2) & (cp_flat <= 0.40) & \
               ~((sr_flat >= 1.2) & (vr_flat >= 1.5))

    vib_cond = (sr_flat <= 0.60) & (vr_flat <= 0.60) & (cp_flat >= 0.35) & (cp_flat <= 0.65)
    vib_edge_cond = (sr_flat <= 0.60) & (vr_flat <= 0.60) & \
                    ((cp_flat < 0.35) | (cp_flat > 0.65))

    conditions = [vsb_up_cond, vsb_down_cond, vsb_neutral_cond,
                  vub_cond, vdb_cond,
                  vib_cond, vib_edge_cond]
    choices = ['VSB_UP', 'VSB_DOWN', 'VSB_NEUTRAL',
               'VUB', 'VDB',
               'VIB', 'VIB_EDGE']

    bar_type_flat = np.select(conditions, choices, default='NORMAL')
    bar_type = bar_type_flat.reshape(n_rows, n_cols)

    # Bar power: spread_ratio * vol_ratio * (close_position - 0.5) * 2
    bar_power = spread_ratio * vol_ratio * (close_pos_arr - 0.5) * 2.0
    bar_power = np.clip(bar_power, -2.0, 2.0)

    is_significant_flat = np.isin(bar_type_flat, ['VSB_UP', 'VSB_DOWN', 'VUB', 'VDB'])
    is_significant = is_significant_flat.reshape(n_rows, n_cols)

    results['wyckoff_bar_type'] = pd.DataFrame(bar_type, index=idx, columns=cols)
    results['wyckoff_bar_power'] = pd.DataFrame(bar_power, index=idx, columns=cols)
    results['wyckoff_bar_is_significant'] = pd.DataFrame(is_significant, index=idx, columns=cols)

    return results


# ======================================================================
# School-compatible split-return interface
# ======================================================================

def compute_wyckoff_advanced_score_split(indicators: dict):
    """
    Polarity-grouped split-return for integration into expert_ensemble.py
    _compute_school_wyckoff.

    POLARITY CONTRACT (enforced):
      - M3 Shakeout (震仓): MUST go to score_bull — it is a bullish trap-reversal,
        NOT a bearish signal. The shakeout is the Composite Operator's final
        washout before markup.
      - M9 Wyckoff Coil (弹簧压缩): MUST NOT enter score_bull or score_bear.
        It is a direction-neutral volatility-compression预警. It is returned
        separately as is_coil_active for use as a signal amplifier.

    Parameters
    ----------
    indicators : dict
        Must contain 'current_price', 'open', 'high', 'low', 'volume'.

    Returns
    -------
    (score_bull, score_bear, reasons_bull, reasons_bear, is_coil_active)
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

        feats = compute_wyckoff_advanced_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []
        is_coil_active = False

        def _read(key):
            df = feats.get(key)
            if df is not None and not df.empty:
                return df['S'].iloc[0]
            return 0.0

        # ══════════════════════════════════════════════════════════
        # Bullish signals (accumulation / demand-side)
        # ══════════════════════════════════════════════════════════
        if _read('wyckoff_jac_signal') > 0:
            score_bull += 0.20; reasons_bull.append(f'JAC跳跃小溪(score={_read("wyckoff_jac_score"):.2f})')
        # M3 Shakeout → score_bull ONLY. This is the Composite Operator's
        # final bear-trap washout before markup — it is a BULLISH signal.
        if _read('wyckoff_shakeout_signal') > 0:
            score_bull += 0.22; reasons_bull.append(f'震仓(score={_read("wyckoff_shakeout_score"):.2f})')
        if _read('wyckoff_buec_signal') > 0:
            score_bull += 0.15; reasons_bull.append(f'BUEC回踩确认(score={_read("wyckoff_buec_score"):.2f})')
        if _read('wyckoff_no_supply_signal') > 0:
            ns_grade = _read('wyckoff_no_supply_grade')
            grade_label = {3.0: '强', 2.0: '标准', 1.0: '弱'}.get(ns_grade, '')
            score_bull += 0.12; reasons_bull.append(f'无供应柱({grade_label})')
        if _read('wyckoff_stopping_vol_signal') > 0:
            score_bull += 0.18; reasons_bull.append(f'增强停止行为(power={_read("wyckoff_stopping_vol_power"):.2f})')
        if _read('wyckoff_absorption_signal') > 0:
            score_bull += 0.16; reasons_bull.append(f'主力吸筹(score={_read("wyckoff_absorption_score"):.2f})')

        # ══════════════════════════════════════════════════════════
        # Bearish signals (distribution / supply-side)
        # ══════════════════════════════════════════════════════════
        if _read('wyckoff_ice_breaking_signal') > 0:
            score_bear += 0.20; reasons_bear.append(f'Ice跌破冰层(score={_read("wyckoff_ice_breaking_score"):.2f})')
        if _read('wyckoff_no_demand_signal') > 0:
            nd_grade = _read('wyckoff_no_demand_grade')
            grade_label = {3.0: '强', 2.0: '标准', 1.0: '弱'}.get(nd_grade, '')
            score_bear += 0.12; reasons_bear.append(f'无需求柱({grade_label})')

        # ══════════════════════════════════════════════════════════
        # Direction-NEUTRAL signals (ZERO score contribution)
        # ══════════════════════════════════════════════════════════
        # M9 Coil: directionless compression预警 → flag only, never scored
        if _read('wyckoff_coil_signal') > 0:
            is_coil_active = True
            reasons_bull.append(f'Coil弹簧压缩(score={_read("wyckoff_coil_score"):.2f})—变盘临近')
        # M10 Phase Transition: informational only
        if _read('wyckoff_phase_transition_signal') > 0:
            phase_cur = feats.get('wyckoff_phase_current')
            if phase_cur is not None and not phase_cur.empty:
                reasons_bull.append(f'阶段过渡(conf={_read("wyckoff_phase_transition_confidence"):.2f})')
        # M11 Bar classification: display-only
        bt = feats.get('wyckoff_bar_type')
        if bt is not None and not bt.empty:
            bt_val = bt['S'].iloc[0]
            if bt_val in ('VSB_UP', 'VUB'):
                reasons_bull.append(f'VSA柱线:{bt_val}')
            elif bt_val in ('VSB_DOWN', 'VDB'):
                reasons_bear.append(f'VSA柱线:{bt_val}')

        return score_bull, score_bear, reasons_bull, reasons_bear, is_coil_active
    except Exception:
        return 0.0, 0.0, [], [], False


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Wyckoff Advanced Features Test ===")
    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=300)
        if df is not None and not df.empty:
            idx_d = df.index
            def _d(c_name):
                return pd.DataFrame({'S': df[c_name].values}, index=idx_d)
            feats = compute_wyckoff_advanced_features(
                _d('open'), _d('high'), _d('low'), _d('close'), _d('volume'))
            print(f"\n{code}:")
            for k, v in feats.items():
                if 'score' in k or 'signal' in k:
                    nz = (v.values > 0).sum()
                    if nz > 0:
                        print(f"  {k}: {nz} non-zero")
                elif k == 'wyckoff_bar_type':
                    vc = v.value_counts().to_dict()
                    print(f"  {k}: {vc}")
