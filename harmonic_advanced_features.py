#!/usr/bin/env python
# harmonic_advanced_features.py — Harmonic pattern detection Part 2
# Deep Crab / Cypher / Shark / PRZ Micro-Validation / RSI BAMM
#
# Inherits the overlapping rolling-window pivot extraction and
# _ratio_proximity / _ratio_best scoring architecture from Part 1.
#
# A-share tolerances: ±0.05 base, ±0.08 wide (extreme CD).

import numpy as np
import pandas as pd

EPS = 1e-8

# ======================================================================
# Window parameters (identical to Part 1 — overlapping segments)
# ======================================================================
C_START, C_END = 1, 8
B_START, B_END = 3, 22
A_START, A_END = 10, 48
X_START, X_END = 25, 85

C_WIN = C_END - C_START + 1
B_WIN = B_END - B_START + 1
A_WIN = A_END - A_START + 1
X_WIN = X_END - X_START + 1

# Tolerances
TOL = 0.05
TOL_WIDE = 0.08
MIN_LEG_ATR = 0.50
SCORE_THRESH = 0.50     # calibrated for window-approximated pivots
SCORE_THRESH_DC = 0.40  # Deep Crab: lower bar (extreme CD ratios compressed)


# ======================================================================
# Helper: proximity scoring (identical to Part 1)
# ======================================================================

def _ratio_proximity(arr, ideal, tol):
    """Proximity score: 1.0 at exact hit, decays to 0.0 at 2*tol away."""
    dev = np.abs(arr - ideal)
    return np.clip(1.0 - dev / (tol * 2.0 + EPS), 0.0, 1.0)


def _ratio_best(arr, ideals, tol):
    """Best proximity among multiple ideal Fibonacci values."""
    scores = [_ratio_proximity(arr, i, tol) for i in ideals]
    return np.maximum.reduce(scores)


# ======================================================================
# Main entry point
# ======================================================================

def compute_harmonic_advanced_features(open_df, high_df, low_df, close_df, volume_df):
    """
    Detect Deep Crab, Cypher, Shark, PRZ Micro-Validation, RSI BAMM.

    Parameters
    ----------
    open_df, high_df, low_df, close_df, volume_df : pd.DataFrame
        Wide-format (index=datetime, columns=stocks).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys:
          harmonic_deep_crab_signal / _score
          harmonic_cypher_signal / _score
          harmonic_shark_signal / _score
          harmonic_prz_confirmation
          harmonic_rsi_bamm_active / _score
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

    # ==================================================================
    # SECTION 0 — Shared precomputations
    # ==================================================================

    # ---- ATR(14) ----
    tr_vals = np.fmax.reduce([
        h.values - l.values,
        np.abs(h.values - np.roll(c.values, 1, axis=0)),
        np.abs(l.values - np.roll(c.values, 1, axis=0)),
    ])
    tr_vals[0] = h.values[0] - l.values[0]
    tr_df = pd.DataFrame(tr_vals, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean().values
    min_leg = atr14 * MIN_LEG_ATR

    # ---- Overlapping rolling-window pivot extraction (bullish) ----
    D_bull = c.values
    C_bull = h.shift(C_START).rolling(C_WIN, min_periods=4).max().values
    B_bull = l.shift(B_START).rolling(B_WIN, min_periods=6).min().values
    A_bull = h.shift(A_START).rolling(A_WIN, min_periods=8).max().values
    X_bull = l.shift(X_START).rolling(X_WIN, min_periods=10).min().values

    # ---- Bearish pivots ----
    D_bear = c.values
    C_bear = l.shift(C_START).rolling(C_WIN, min_periods=4).min().values
    B_bear = h.shift(B_START).rolling(B_WIN, min_periods=6).max().values
    A_bear = l.shift(A_START).rolling(A_WIN, min_periods=8).min().values
    X_bear = h.shift(X_START).rolling(X_WIN, min_periods=10).max().values

    # ---- Volume MAs ----
    vol_ma20 = v.rolling(20, min_periods=10).mean().values

    # ---- RSI(5) for BAMM ----
    delta = c.values - np.roll(c.values, 1, axis=0)
    delta[0] = 0
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.DataFrame(gain, index=idx, columns=cols).ewm(span=5, adjust=False).mean().values
    avg_loss = pd.DataFrame(loss, index=idx, columns=cols).ewm(span=5, adjust=False).mean().values
    rs = avg_gain / (avg_loss + EPS)
    rsi5 = 100.0 - (100.0 / (1.0 + rs))

    # RSI(14) as secondary confirmation
    avg_gain14 = pd.DataFrame(gain, index=idx, columns=cols).ewm(span=14, adjust=False).mean().values
    avg_loss14 = pd.DataFrame(loss, index=idx, columns=cols).ewm(span=14, adjust=False).mean().values
    rs14 = avg_gain14 / (avg_loss14 + EPS)
    rsi14 = 100.0 - (100.0 / (1.0 + rs14))

    # ---- RSI at C position (approximated from C-pivot window) ----
    # For bullish: C is a swing HIGH → use rolling MAX of RSI in C-window
    rsi5_c_bull = pd.DataFrame(rsi5, index=idx, columns=cols).shift(C_START) \
                   .rolling(C_WIN, min_periods=4).max().values
    # For bearish: C is a swing LOW → use rolling MIN of RSI in C-window
    rsi5_c_bear = pd.DataFrame(rsi5, index=idx, columns=cols).shift(C_START) \
                   .rolling(C_WIN, min_periods=4).min().values

    # ---- K-line geometry for PRZ ----
    spread_arr = h.values - l.values
    body_arr = np.abs(c.values - o.values)
    lower_wick = np.minimum(o.values, c.values) - l.values
    upper_wick = h.values - np.maximum(o.values, c.values)
    close_pos = (c.values - l.values) / (spread_arr + EPS)

    # ==================================================================
    # SECTION 1 — Bullish & bearish leg lengths
    # ==================================================================
    XA_bull = A_bull - X_bull
    AB_bull = A_bull - B_bull
    BC_bull = C_bull - B_bull
    CD_bull = C_bull - D_bull
    XC_bull = C_bull - X_bull    # X→C total (for Cypher/Shark anchor)

    XA_bear = X_bear - A_bear
    AB_bear = B_bear - A_bear
    BC_bear = B_bear - C_bear
    CD_bear = D_bear - C_bear
    XC_bear = X_bear - C_bear    # X→C total (bearish: C is below X)

    # ATR-gated minimum leg constraint (applies to all patterns)
    legs_ok_bull = (XA_bull > min_leg) & (AB_bull > min_leg) & \
                   (BC_bull > min_leg) & (CD_bull > min_leg)
    legs_ok_bear = (XA_bear > min_leg) & (AB_bear > min_leg) & \
                   (BC_bear > min_leg) & (CD_bear > min_leg)

    # ---- Structural constraints ----
    # Standard (XA-anchored): A > C (Gartley/Bat/Butterfly/Crab/DeepCrab)
    struct_bull_ok = (
        (A_bull > X_bull) & (A_bull > C_bull) &
        (C_bull > B_bull) & (C_bull > D_bull) &
        (B_bull < A_bull) & (B_bull < C_bull)
    )
    struct_bear_ok = (
        (A_bear < X_bear) & (A_bear < C_bear) &
        (C_bear < B_bear) & (C_bear < D_bear) &
        (B_bear > A_bear) & (B_bear > C_bear)
    )
    # XC-anchored (Cypher/Shark): C > A — C breaks beyond A
    struct_xc_bull_ok = (
        (A_bull > X_bull) &           # A is a high above X
        (B_bull < A_bull) &           # B retraces below A
        (C_bull > A_bull) &           # ★ C breaks ABOVE A (genetic marker)
        (C_bull > B_bull) &           # C above B
        (D_bull < C_bull)             # D is the final low
    )
    struct_xc_bear_ok = (
        (A_bear < X_bear) &           # A is a low below X
        (B_bear > A_bear) &           # B rebounds above A
        (C_bear < A_bear) &           # ★ C breaks BELOW A
        (C_bear < B_bear) &           # C below B
        (D_bear > C_bear)             # D is the final high
    )

    # ---- Ratios ----
    rAB_XA_b = AB_bull / (XA_bull + EPS)
    rBC_AB_b = BC_bull / (AB_bull + EPS)
    rCD_BC_b = CD_bull / (BC_bull + EPS)
    rCD_XC_b = CD_bull / (XC_bull + EPS)   # Cypher/Shark anchor
    rXC_XA_b = XC_bull / (XA_bull + EPS)   # Cypher/Shark: total C extension

    rAB_XA_s = AB_bear / (XA_bear + EPS)
    rBC_AB_s = BC_bear / (AB_bear + EPS)
    rCD_BC_s = CD_bear / (BC_bear + EPS)
    rCD_XC_s = CD_bear / (XC_bear + EPS)
    rXC_XA_s = XC_bear / (XA_bear + EPS)

    # XD extension (for Deep Crab)
    D_gt_X_b = D_bull > X_bull
    D_lt_X_s = D_bear < X_bear
    XD_ext_b = np.where(~D_gt_X_b, (X_bull - D_bull) / (XA_bull + EPS), np.nan)
    XD_ext_s = np.where(~D_lt_X_s, (D_bear - X_bear) / (XA_bear + EPS), np.nan)

    # ---- Volume factors ----
    vol_bonus = np.minimum(v.values / (vol_ma20 + EPS), 2.0)
    vol_factor = np.minimum(vol_bonus * 0.5 + 0.75, 1.15)

    # ════════════════════════════════════════════════════════════════
    # PATTERN 1 — Deep Crab (深蟹 — window-calibrated)
    #   AB/XA ≈ {0.382,0.618}  BC/AB ≈ {0.382,0.886}
    #   CD dominates BC (CD > BC * 1.2)  XD/XA ≈ 1.618 (EXTENSION)
    #   ★ Volume gate mandatory: vol > vol_ma20 * 1.2
    #   ★ CD scored against {2.0, 2.618, 3.0} with TOL_DC=0.12
    #     (window pivots compress CD/BC ratio; textbook 2.618-3.618 unreachable)
    # ════════════════════════════════════════════════════════════════
    TOL_DC = 0.12

    dc_vol_gate = v.values > vol_ma20 * 1.2
    dc_base_bull = struct_bull_ok & legs_ok_bull & (C_bull < A_bull) & \
                   (~D_gt_X_b) & (CD_bull > BC_bull * 1.2) & dc_vol_gate
    dc_base_bear = struct_bear_ok & legs_ok_bear & (C_bear > A_bear) & \
                   (~D_lt_X_s) & (CD_bear > BC_bear * 1.2) & dc_vol_gate

    dc_s1_b = _ratio_best(rAB_XA_b, [0.382, 0.618], TOL)
    dc_s2_b = _ratio_best(rBC_AB_b, [0.382, 0.886], TOL)
    dc_s3_b = _ratio_best(rCD_BC_b, [2.000, 2.618, 3.000], TOL_DC)
    dc_s4_b = _ratio_proximity(XD_ext_b, 1.618, TOL)
    dc_raw_bull = np.nan_to_num(dc_s1_b * 0.10 + dc_s2_b * 0.10 +
                                 dc_s3_b * 0.40 + np.nan_to_num(dc_s4_b, nan=0.0) * 0.40,
                                 nan=0.0)

    dc_s1_s = _ratio_best(rAB_XA_s, [0.382, 0.618], TOL)
    dc_s2_s = _ratio_best(rBC_AB_s, [0.382, 0.886], TOL)
    dc_s3_s = _ratio_best(rCD_BC_s, [2.000, 2.618, 3.000], TOL_DC)
    dc_s4_s = _ratio_proximity(XD_ext_s, 1.618, TOL)
    dc_raw_bear = np.nan_to_num(dc_s1_s * 0.10 + dc_s2_s * 0.10 +
                                 dc_s3_s * 0.40 + np.nan_to_num(dc_s4_s, nan=0.0) * 0.40,
                                 nan=0.0)

    deep_crab_score = np.where(dc_base_bull, dc_raw_bull,
                      np.where(dc_base_bear, dc_raw_bear, 0.0))
    deep_crab_score = np.clip(deep_crab_score, 0.0, 1.0)

    deep_crab_signal = np.where(dc_base_bull & (deep_crab_score >= SCORE_THRESH_DC), 1.0,
                       np.where(dc_base_bear & (deep_crab_score >= SCORE_THRESH_DC), -1.0, 0.0))

    # ════════════════════════════════════════════════════════════════
    # PATTERN 2 — Cypher (暗号)
    #   AB/XA ≈ {0.382,0.618}  XC/XA ≈ {1.272,1.414}  CD/XC ≈ 0.786
    #   ★ Core constraint: C > A (bullish) / C < A (bearish)
    #   ★ Anchor: D is a retracement of XC (NOT XA)
    # ════════════════════════════════════════════════════════════════

    # C must cross beyond A — the genetic marker of Cypher
    cy_C_gt_A_bull = C_bull > A_bull   # (redundant with struct_xc, kept for clarity)
    cy_C_lt_A_bear = C_bear < A_bear

    cy_base_bull = struct_xc_bull_ok & legs_ok_bull
    cy_base_bear = struct_xc_bear_ok & legs_ok_bear

    cy_s1_b = _ratio_best(rAB_XA_b, [0.382, 0.618], TOL)
    cy_s2_b = _ratio_best(rXC_XA_b, [1.272, 1.414], TOL)  # XC extension
    cy_s3_b = _ratio_proximity(rCD_XC_b, 0.786, TOL)       # D at 78.6% of XC
    cy_raw_bull = np.nan_to_num(cy_s1_b * 0.15 + cy_s2_b * 0.35 +
                                 np.nan_to_num(cy_s3_b, nan=0.0) * 0.50, nan=0.0)

    cy_s1_s = _ratio_best(rAB_XA_s, [0.382, 0.618], TOL)
    cy_s2_s = _ratio_best(rXC_XA_s, [1.272, 1.414], TOL)
    cy_s3_s = _ratio_proximity(rCD_XC_s, 0.786, TOL)
    cy_raw_bear = np.nan_to_num(cy_s1_s * 0.15 + cy_s2_s * 0.35 +
                                 np.nan_to_num(cy_s3_s, nan=0.0) * 0.50, nan=0.0)

    cypher_score = np.where(cy_base_bull, cy_raw_bull * vol_factor,
                   np.where(cy_base_bear, cy_raw_bear * vol_factor, 0.0))
    cypher_score = np.clip(cypher_score, 0.0, 1.0)

    cypher_signal = np.where(cy_base_bull & (cypher_score >= SCORE_THRESH), 1.0,
                    np.where(cy_base_bear & (cypher_score >= SCORE_THRESH), -1.0, 0.0))

    # ════════════════════════════════════════════════════════════════
    # PATTERN 3 — Shark (鲨鱼)
    #   AB/XA ≈ 0.382 (very shallow)  BC/XA ≈ {1.13,1.618}
    #   CD/XC ≈ {0.886,1.13}
    #   ★ Core: C > A, AB is the shallowest retracement
    # ════════════════════════════════════════════════════════════════

    sh_C_gt_A_bull = C_bull > A_bull   # (redundant with struct_xc, kept for clarity)
    sh_C_lt_A_bear = C_bear < A_bear
    # AB must be shallow — less than 0.50 of XA
    sh_shallow_bull = rAB_XA_b < 0.50
    sh_shallow_bear = rAB_XA_s < 0.50

    sh_base_bull = struct_xc_bull_ok & legs_ok_bull & sh_shallow_bull
    sh_base_bear = struct_xc_bear_ok & legs_ok_bear & sh_shallow_bear

    sh_s1_b = _ratio_proximity(rAB_XA_b, 0.382, TOL)
    sh_s2_b = _ratio_best(rXC_XA_b, [1.130, 1.618], TOL)  # BC extension
    sh_s3_b = _ratio_best(rCD_XC_b, [0.886, 1.130], TOL)   # D as retrace/ext of XC
    sh_raw_bull = np.nan_to_num(sh_s1_b * 0.15 + sh_s2_b * 0.40 +
                                 np.nan_to_num(sh_s3_b, nan=0.0) * 0.45, nan=0.0)

    sh_s1_s = _ratio_proximity(rAB_XA_s, 0.382, TOL)
    sh_s2_s = _ratio_best(rXC_XA_s, [1.130, 1.618], TOL)
    sh_s3_s = _ratio_best(rCD_XC_s, [0.886, 1.130], TOL)
    sh_raw_bear = np.nan_to_num(sh_s1_s * 0.15 + sh_s2_s * 0.40 +
                                 np.nan_to_num(sh_s3_s, nan=0.0) * 0.45, nan=0.0)

    shark_score = np.where(sh_base_bull, sh_raw_bull * vol_factor,
                  np.where(sh_base_bear, sh_raw_bear * vol_factor, 0.0))
    shark_score = np.clip(shark_score, 0.0, 1.0)

    shark_signal = np.where(sh_base_bull & (shark_score >= SCORE_THRESH), 1.0,
                   np.where(sh_base_bear & (shark_score >= SCORE_THRESH), -1.0, 0.0))

    # ════════════════════════════════════════════════════════════════
    # PATTERN 4 — PRZ Micro-Validation (终端价格柱反转确认)
    #   Checks whether the current bar at the D point shows reversal
    #   candlestick characteristics. Direction-agnostic confidence
    #   score (0-1) usable as a multiplier for any harmonic signal.
    # ════════════════════════════════════════════════════════════════

    # T1: Lower wick dominant → bullish reversal potential
    prz_t1_bull = (lower_wick / (spread_arr + EPS) > 0.50) & (close_pos > 0.40)

    # T2: Upper wick dominant → bearish reversal potential
    prz_t1_bear = (upper_wick / (spread_arr + EPS) > 0.50) & (close_pos < 0.60)

    # T3: Bullish engulfing (close > open, body > prev_body * 1.5)
    prev_body = np.abs(np.roll(c.values, 1, axis=0) - np.roll(o.values, 1, axis=0))
    prev_body[0] = body_arr[0]
    prz_t3_bull = (c.values > o.values) & (body_arr > prev_body * 1.3)

    # T4: Bearish engulfing
    prz_t3_bear = (c.values < o.values) & (body_arr > prev_body * 1.3)

    # T5: Volume confirmation at D
    prz_t5 = v.values > vol_ma20 * 1.2

    # Bullish PRZ score
    prz_bull_raw = (
        prz_t1_bull.astype(float) * 0.35 +
        prz_t3_bull.astype(float) * 0.35 +
        prz_t5.astype(float) * 0.30
    )

    # Bearish PRZ score
    prz_bear_raw = (
        prz_t1_bear.astype(float) * 0.35 +
        prz_t3_bear.astype(float) * 0.35 +
        prz_t5.astype(float) * 0.30
    )

    # Combined: whichever side scores higher
    prz_confirmation = np.maximum(prz_bull_raw, prz_bear_raw)
    prz_confirmation = np.clip(prz_confirmation, 0.0, 1.0)

    # ════════════════════════════════════════════════════════════════
    # PATTERN 5 — RSI BAMM (Bat Action Magnet Momentum)
    #   Carney's RSI(5) divergence confirmation at the D point.
    #   Bullish: price_D < price_C, RSI_D > RSI_C, RSI_D < 30
    #   Bearish: price_D > price_C, RSI_D < RSI_C, RSI_D > 70
    #   Direction-NEUTRAL — returns a confidence multiplier (0-1).
    # ════════════════════════════════════════════════════════════════

    rsi5_d = rsi5  # RSI at D (current bar)

    # ---- Bullish BAMM ----
    bamm_b1 = D_bull < C_bull                          # price lower at D than C
    bamm_b2 = rsi5_d > rsi5_c_bull                     # RSI higher at D (positive divergence!)
    bamm_b3 = rsi5_d < 30                              # RSI in oversold zone
    bamm_b4 = rsi14 < 45                               # RSI(14) secondary confirm
    bamm_b5 = rsi5_d > np.roll(rsi5_d, 1, axis=0)      # RSI turning up (1-bar change)
    bamm_b5[0] = False

    bamm_bull = bamm_b1 & bamm_b2 & bamm_b3

    bamm_score_bull = (
        bamm_b1.astype(float) * 0.15 +
        bamm_b2.astype(float) * 0.35 +   # divergence is the soul of BAMM
        bamm_b3.astype(float) * 0.25 +
        bamm_b4.astype(float) * 0.15 +
        bamm_b5.astype(float) * 0.10
    )

    # ---- Bearish BAMM ----
    bamm_s1 = D_bear > C_bear                          # price higher at D than C
    bamm_s2 = rsi5_d < rsi5_c_bear                     # RSI lower at D (negative divergence!)
    bamm_s3 = rsi5_d > 70                              # RSI in overbought zone
    bamm_s4 = rsi14 > 55                               # RSI(14) secondary confirm
    bamm_s5 = rsi5_d < np.roll(rsi5_d, 1, axis=0)      # RSI turning down
    bamm_s5[0] = False

    bamm_bear = bamm_s1 & bamm_s2 & bamm_s3

    bamm_score_bear = (
        bamm_s1.astype(float) * 0.15 +
        bamm_s2.astype(float) * 0.35 +
        bamm_s3.astype(float) * 0.25 +
        bamm_s4.astype(float) * 0.15 +
        bamm_s5.astype(float) * 0.10
    )

    # Unified BAMM score (whichever side fires)
    bamm_score = np.where(bamm_bull, bamm_score_bull,
                 np.where(bamm_bear, bamm_score_bear, 0.0))
    bamm_score = np.clip(bamm_score, 0.0, 1.0)
    bamm_active = (bamm_score >= 0.55).astype(float)

    # ==================================================================
    # SECTION 8 — Package results
    # ==================================================================

    results = {}

    results['harmonic_deep_crab_signal'] = pd.DataFrame(deep_crab_signal, index=idx, columns=cols)
    results['harmonic_deep_crab_score']  = pd.DataFrame(deep_crab_score, index=idx, columns=cols)

    results['harmonic_cypher_signal'] = pd.DataFrame(cypher_signal, index=idx, columns=cols)
    results['harmonic_cypher_score']  = pd.DataFrame(cypher_score, index=idx, columns=cols)

    results['harmonic_shark_signal'] = pd.DataFrame(shark_signal, index=idx, columns=cols)
    results['harmonic_shark_score']  = pd.DataFrame(shark_score, index=idx, columns=cols)

    results['harmonic_prz_confirmation'] = pd.DataFrame(prz_confirmation, index=idx, columns=cols)

    results['harmonic_rsi_bamm_active'] = pd.DataFrame(bamm_active, index=idx, columns=cols)
    results['harmonic_rsi_bamm_score']  = pd.DataFrame(bamm_score, index=idx, columns=cols)

    return results


# ======================================================================
# School-compatible split-return interface
# ======================================================================

def compute_harmonic_advanced_score_split(indicators: dict):
    """
    Polarity-grouped split-return for _compute_school_harmonic integration.

    Returns
    -------
    (score_bull, score_bear, reasons_bull, reasons_bear,
     prz_confidence, bamm_multiplier)
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

        feats = compute_harmonic_advanced_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        def _read(key):
            df = feats.get(key)
            if df is not None and not df.empty:
                return df['S'].iloc[0]
            return 0.0

        # Deep Crab — extreme reversal with volume confirmation
        if _read('harmonic_deep_crab_signal') > 0:
            score_bull += 0.22
            reasons_bull.append(f'DeepCrab看涨(score={_read("harmonic_deep_crab_score"):.2f})')
        elif _read('harmonic_deep_crab_signal') < 0:
            score_bear += 0.22
            reasons_bear.append(f'DeepCrab看跌(score={_read("harmonic_deep_crab_score"):.2f})')

        # Cypher — XC-anchored, C beyond A
        if _read('harmonic_cypher_signal') > 0:
            score_bull += 0.18
            reasons_bull.append(f'Cypher看涨(score={_read("harmonic_cypher_score"):.2f})')
        elif _read('harmonic_cypher_signal') < 0:
            score_bear += 0.18
            reasons_bear.append(f'Cypher看跌(score={_read("harmonic_cypher_score"):.2f})')

        # Shark — aggressive reversal
        if _read('harmonic_shark_signal') > 0:
            score_bull += 0.16
            reasons_bull.append(f'Shark看涨(score={_read("harmonic_shark_score"):.2f})')
        elif _read('harmonic_shark_signal') < 0:
            score_bear += 0.16
            reasons_bear.append(f'Shark看跌(score={_read("harmonic_shark_score"):.2f})')

        prz_conf = _read('harmonic_prz_confirmation')
        if prz_conf > 0.60:
            reasons_bull.append(f'PRZ确认(conf={prz_conf:.2f})')

        bamm_active = _read('harmonic_rsi_bamm_active')
        bamm_score = _read('harmonic_rsi_bamm_score')
        bamm_mult = min(1.0 + bamm_score * 0.30, 1.50) if bamm_active > 0 else 1.0
        if bamm_active > 0:
            reasons_bull.append(f'RSI BAMM激活(score={bamm_score:.2f}, mult={bamm_mult:.2f}x)')

        return score_bull, score_bear, reasons_bull, reasons_bear, prz_conf, bamm_mult
    except Exception:
        return 0.0, 0.0, [], [], 0.0, 1.0


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Harmonic Advanced Features Test ===")
    for code in ['000012', '600519', '000651', '300750']:
        df = get_daily_kline(code, days=300)
        if df is not None and not df.empty:
            idx_d = df.index
            def _d(c_name):
                return pd.DataFrame({'S': df[c_name].values}, index=idx_d)
            feats = compute_harmonic_advanced_features(
                _d('open'), _d('high'), _d('low'), _d('close'), _d('volume'))
            print(f"\n{code}:")
            for k in ['harmonic_deep_crab_signal', 'harmonic_cypher_signal',
                       'harmonic_shark_signal']:
                sv = feats[k].values
                bull = int((sv > 0).sum())
                bear = int((sv < 0).sum())
                score_k = k.replace('_signal', '_score')
                sc_max = feats[score_k].values.max()
                if bull + bear > 0:
                    print(f"  {k}: bull={bull} bear={bear} max_score={sc_max:.2f}")
                else:
                    print(f"  {k}: none")
            prz_nz = int((feats['harmonic_prz_confirmation'].values > 0.50).sum())
            bamm_nz = int((feats['harmonic_rsi_bamm_active'].values > 0).sum())
            print(f"  PRZ confirm (>0.50): {prz_nz}")
            print(f"  RSI BAMM active: {bamm_nz}")
