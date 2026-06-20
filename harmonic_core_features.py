#!/usr/bin/env python
# harmonic_core_features.py — Harmonic pattern detection Part 1
# Gartley / Bat / Butterfly / Crab — 4 core patterns
#
# Architecture: OVERLAPPING segmented rolling-window pivot extraction.
#   Each pivot window overlaps its neighbor. The alternating high/low
#   structural constraint ensures only valid X-A-B-C-D sequences pass.
#
#   D = T (today's close)
#   C window = [T-8,  T-1 ]   rolling max/min (most recent opposite swing)
#   B window = [T-22, T-3 ]   rolling min/max (penultimate swing)
#   A window = [T-48, T-10]   rolling max/min
#   X window = [T-85, T-25]   rolling min/max (earliest swing)
#
#   MIN_LEG = ATR(14) * 0.50  — minimum swing magnitude
#   Tolerances: ±0.05 base, ±0.08 for extreme CD (Crab).

import numpy as np
import pandas as pd

EPS = 1e-8

# ======================================================================
# Overlapping window parameters (bars from T)
# ======================================================================
C_START, C_END = 1, 8       # C: most recent opposite swing
B_START, B_END = 3, 22      # B: penultimate swing (overlaps C)
A_START, A_END = 10, 48     # A: antepenultimate swing (overlaps B)
X_START, X_END = 25, 85     # X: earliest swing (overlaps A)

C_WIN = C_END - C_START + 1
B_WIN = B_END - B_START + 1
A_WIN = A_END - A_START + 1
X_WIN = X_END - X_START + 1

# Tolerances (A-share adjusted)
TOL = 0.05       # base Fibonacci tolerance
TOL_WIDE = 0.08  # CD extreme extensions (Crab, deep Bat)

# Minimum leg size as fraction of ATR
MIN_LEG_ATR = 0.50


# ======================================================================
# Main entry point
# ======================================================================

def compute_harmonic_core_features(open_df, high_df, low_df, close_df, volume_df):
    """
    Detect Gartley, Bat, Butterfly, Crab harmonic patterns.

    Parameters
    ----------
    open_df, high_df, low_df, close_df, volume_df : pd.DataFrame
        Wide-format (index=datetime, columns=stocks).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: harmonic_gartley_signal, harmonic_gartley_score,
              harmonic_bat_signal, harmonic_bat_score,
              harmonic_butterfly_signal, harmonic_butterfly_score,
              harmonic_crab_signal, harmonic_crab_score.
        Signals: +1=bullish, -1=bearish, 0=no pattern.
        Scores: 0-1 quality.
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

    # ==================================================================
    # SECTION 0 — ATR & overlapping rolling-window pivot extraction
    # ==================================================================
    # ---- ATR(14) for minimum-leg constraint ----
    tr_vals = np.fmax.reduce([
        h.values - l.values,
        np.abs(h.values - np.roll(c.values, 1, axis=0)),
        np.abs(l.values - np.roll(c.values, 1, axis=0)),
    ])
    tr_vals[0] = h.values[0] - l.values[0]
    tr_df = pd.DataFrame(tr_vals, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean().values
    min_leg = atr14 * MIN_LEG_ATR

    # ---- Bullish pivots (X=low, A=high, B=low, C=high, D=close) ----
    D_bull = c.values
    C_bull = h.shift(C_START).rolling(C_WIN, min_periods=4).max().values
    B_bull = l.shift(B_START).rolling(B_WIN, min_periods=6).min().values
    A_bull = h.shift(A_START).rolling(A_WIN, min_periods=8).max().values
    X_bull = l.shift(X_START).rolling(X_WIN, min_periods=10).min().values

    # ---- Bearish pivots (X=high, A=low, B=high, C=low, D=close) ----
    D_bear = c.values
    C_bear = l.shift(C_START).rolling(C_WIN, min_periods=4).min().values
    B_bear = h.shift(B_START).rolling(B_WIN, min_periods=6).max().values
    A_bear = l.shift(A_START).rolling(A_WIN, min_periods=8).min().values
    X_bear = h.shift(X_START).rolling(X_WIN, min_periods=10).max().values

    # ---- Volume MA ----
    vol_ma20 = v.rolling(20, min_periods=10).mean().values

    # ==================================================================
    # SECTION 1 — Bullish leg lengths, ATR filter & ratios
    # ==================================================================
    XA_bull = A_bull - X_bull
    AB_bull = A_bull - B_bull
    BC_bull = C_bull - B_bull
    CD_bull = C_bull - D_bull

    # ATR-gated minimum leg size: every leg must have meaningful amplitude
    legs_ok_bull = (XA_bull > min_leg) & (AB_bull > min_leg) & \
                   (BC_bull > min_leg) & (CD_bull > min_leg)

    rAB_XA_b = AB_bull / (XA_bull + EPS)
    rBC_AB_b = BC_bull / (AB_bull + EPS)
    rCD_BC_b = CD_bull / (BC_bull + EPS)

    # XD ratio — two modes for the two pattern families
    D_gt_X_b = D_bull > X_bull
    XD_retrace_b = np.where(D_gt_X_b, (D_bull - X_bull) / (XA_bull + EPS), np.nan)
    XD_ext_b     = np.where(~D_gt_X_b, (X_bull - D_bull) / (XA_bull + EPS), np.nan)

    # Structural: alternating high-low sequence with consistent price ordering
    struct_bull_ok = (
        (A_bull > X_bull) & (A_bull > C_bull) &   # A dominates later high
        (C_bull > B_bull) & (C_bull > D_bull) &   # C dominates B and D
        (B_bull < A_bull) & (B_bull < C_bull)     # B is a retracement low
    )

    # ==================================================================
    # SECTION 2 — Bearish leg lengths, ATR filter & ratios
    # ==================================================================
    XA_bear = X_bear - A_bear
    AB_bear = B_bear - A_bear
    BC_bear = B_bear - C_bear
    CD_bear = D_bear - C_bear

    legs_ok_bear = (XA_bear > min_leg) & (AB_bear > min_leg) & \
                   (BC_bear > min_leg) & (CD_bear > min_leg)

    rAB_XA_s = AB_bear / (XA_bear + EPS)
    rBC_AB_s = BC_bear / (AB_bear + EPS)
    rCD_BC_s = CD_bear / (BC_bear + EPS)

    D_lt_X_s = D_bear < X_bear
    XD_retrace_s = np.where(D_lt_X_s, (X_bear - D_bear) / (XA_bear + EPS), np.nan)
    XD_ext_s     = np.where(~D_lt_X_s, (D_bear - X_bear) / (XA_bear + EPS), np.nan)

    struct_bear_ok = (
        (A_bear < X_bear) & (A_bear < C_bear) &
        (C_bear < B_bear) & (C_bear < D_bear) &
        (B_bear > A_bear) & (B_bear > C_bear)
    )

    # ==================================================================
    # SECTION 3 — Helper: proximity scoring & ratio scorer
    # ==================================================================
    # Harmonic patterns are ZONES (Carney's PRZ), not exact points.
    # Proximity scoring: score=1.0 at perfect Fibonacci hit,
    # decays linearly to 0.0 at 2*tol away.

    def _ratio_proximity(arr, ideal, tol):
        """Proximity score: 1.0 at exact hit, 0.0 at tol*2 away."""
        dev = np.abs(arr - ideal)
        return np.clip(1.0 - dev / (tol * 2.0 + EPS), 0.0, 1.0)

    def _ratio_best(arr, ideals, tol):
        """Best proximity among multiple ideal values."""
        scores = [_ratio_proximity(arr, i, tol) for i in ideals]
        return np.maximum.reduce(scores)

    def _between(arr, lo, hi):
        """Element-wise [lo, hi] check — for structural constraints."""
        return (arr >= lo) & (arr <= hi)

    # ==================================================================
    # SECTION 4 — Pattern matching: GARTLEY  (proximity-scored)
    # ==================================================================
    #   AB/XA ≈ 0.618,  BC/AB ≈ {0.382,0.886},  CD/BC ≈ {1.272,1.618},  XD/XA ≈ 0.786
    #   Constraint: D > X (retracement, not extension)

    gb_base_bull = struct_bull_ok & legs_ok_bull & (C_bull < A_bull)
    gb_base_bear = struct_bear_ok & legs_ok_bear & (C_bear > A_bear)

    # Bullish proximity scores
    gb_s1 = _ratio_proximity(rAB_XA_b, 0.618, TOL)
    gb_s2 = _ratio_best(rBC_AB_b, [0.382, 0.886], TOL)
    gb_s3 = _ratio_best(rCD_BC_b, [1.272, 1.618], TOL)
    gb_s4 = _ratio_proximity(XD_retrace_b, 0.786, TOL)
    gb_raw = np.nan_to_num(gb_s1 * 0.25 + gb_s2 * 0.25 + gb_s3 * 0.20 + np.nan_to_num(gb_s4, nan=0.0) * 0.30, nan=0.0)

    # Bearish proximity scores
    gs_s1 = _ratio_proximity(rAB_XA_s, 0.618, TOL)
    gs_s2 = _ratio_best(rBC_AB_s, [0.382, 0.886], TOL)
    gs_s3 = _ratio_best(rCD_BC_s, [1.272, 1.618], TOL)
    gs_s4 = _ratio_proximity(XD_retrace_s, 0.786, TOL)
    gs_raw = np.nan_to_num(gs_s1 * 0.25 + gs_s2 * 0.25 + gs_s3 * 0.20 + np.nan_to_num(gs_s4, nan=0.0) * 0.30, nan=0.0)

    # Volume bonus
    vol_bonus = np.minimum(v.values / (vol_ma20 + EPS), 2.0)
    vol_factor = np.minimum(vol_bonus * 0.5 + 0.75, 1.15)

    gartley_score = np.where(gb_base_bull, gb_raw * vol_factor,
                    np.where(gb_base_bear, gs_raw * vol_factor, 0.0))
    gartley_score = np.clip(gartley_score, 0.0, 1.0)

    SCORE_THRESH = 0.50  # calibrated for window-approximated pivots
    gartley_signal = np.where(gb_base_bull & (gartley_score >= SCORE_THRESH), 1.0,
                     np.where(gb_base_bear & (gartley_score >= SCORE_THRESH), -1.0, 0.0))

    # ==================================================================
    # SECTION 5 — Pattern matching: BAT  (proximity-scored)
    # ==================================================================
    #   AB/XA ≈ {0.382,0.500}, BC/AB ≈ {0.382,0.886}, CD/BC ≈ {1.618,2.618}, XD/XA ≈ 0.886
    #   Constraint: AB < 0.618*XA (shallower retracement than Gartley)

    bb_base_bull = struct_bull_ok & legs_ok_bull & (C_bull < A_bull) & (rAB_XA_b < 0.618)
    bb_base_bear = struct_bear_ok & legs_ok_bear & (C_bear > A_bear) & (rAB_XA_s < 0.618)

    bb_s1 = _ratio_best(rAB_XA_b, [0.382, 0.500], TOL)
    bb_s2 = _ratio_best(rBC_AB_b, [0.382, 0.886], TOL)
    bb_s3 = _ratio_best(rCD_BC_b, [1.618, 2.618], TOL_WIDE)
    bb_s4 = _ratio_proximity(XD_retrace_b, 0.886, TOL)
    bb_raw = np.nan_to_num(bb_s1 * 0.15 + bb_s2 * 0.20 + bb_s3 * 0.30 + np.nan_to_num(bb_s4, nan=0.0) * 0.35, nan=0.0)

    bs_s1 = _ratio_best(rAB_XA_s, [0.382, 0.500], TOL)
    bs_s2 = _ratio_best(rBC_AB_s, [0.382, 0.886], TOL)
    bs_s3 = _ratio_best(rCD_BC_s, [1.618, 2.618], TOL_WIDE)
    bs_s4 = _ratio_proximity(XD_retrace_s, 0.886, TOL)
    bs_raw = np.nan_to_num(bs_s1 * 0.15 + bs_s2 * 0.20 + bs_s3 * 0.30 + np.nan_to_num(bs_s4, nan=0.0) * 0.35, nan=0.0)

    bat_score = np.where(bb_base_bull, bb_raw * vol_factor,
                np.where(bb_base_bear, bs_raw * vol_factor, 0.0))
    bat_score = np.clip(bat_score, 0.0, 1.0)

    bat_signal = np.where(bb_base_bull & (bat_score >= SCORE_THRESH), 1.0,
                 np.where(bb_base_bear & (bat_score >= SCORE_THRESH), -1.0, 0.0))

    # ==================================================================
    # SECTION 6 — Pattern matching: BUTTERFLY  (proximity-scored)
    # ==================================================================
    #   AB/XA ≈ 0.786, BC/AB ≈ {0.382,0.886}, CD/BC ≈ {1.618,2.618}, XD/XA ≈ 1.272 (EXTENSION)
    #   Constraint: B is deep (>0.618 XA), D extends below X

    bf_base_bull = struct_bull_ok & legs_ok_bull & (C_bull < A_bull) & \
                   (rAB_XA_b > 0.618) & (~D_gt_X_b)
    bf_base_bear = struct_bear_ok & legs_ok_bear & (C_bear > A_bear) & \
                   (rAB_XA_s > 0.618) & (~D_lt_X_s)

    bf_s1 = _ratio_proximity(rAB_XA_b, 0.786, TOL)
    bf_s2 = _ratio_best(rBC_AB_b, [0.382, 0.886], TOL)
    bf_s3 = _ratio_best(rCD_BC_b, [1.618, 2.618], TOL_WIDE)
    bf_s4 = _ratio_proximity(XD_ext_b, 1.272, TOL)
    bf_raw_bull = np.nan_to_num(bf_s1 * 0.25 + bf_s2 * 0.15 + bf_s3 * 0.25 + np.nan_to_num(bf_s4, nan=0.0) * 0.35, nan=0.0)

    bf_s1s = _ratio_proximity(rAB_XA_s, 0.786, TOL)
    bf_s2s = _ratio_best(rBC_AB_s, [0.382, 0.886], TOL)
    bf_s3s = _ratio_best(rCD_BC_s, [1.618, 2.618], TOL_WIDE)
    bf_s4s = _ratio_proximity(XD_ext_s, 1.272, TOL)
    bf_raw_bear = np.nan_to_num(bf_s1s * 0.25 + bf_s2s * 0.15 + bf_s3s * 0.25 + np.nan_to_num(bf_s4s, nan=0.0) * 0.35, nan=0.0)

    butterfly_score = np.where(bf_base_bull, bf_raw_bull * vol_factor,
                      np.where(bf_base_bear, bf_raw_bear * vol_factor, 0.0))
    butterfly_score = np.clip(butterfly_score, 0.0, 1.0)

    butterfly_signal = np.where(bf_base_bull & (butterfly_score >= SCORE_THRESH), 1.0,
                       np.where(bf_base_bear & (butterfly_score >= SCORE_THRESH), -1.0, 0.0))

    # ==================================================================
    # SECTION 7 — Pattern matching: CRAB  (proximity-scored)
    # ==================================================================
    #   AB/XA ≈ {0.382,0.618}, BC/AB ≈ {0.382,0.886},
    #   CD/BC ≈ {2.618,3.000,3.618}, XD/XA ≈ 1.618 (EXTENSION)
    #   Constraint: CD is the longest leg, D extends below X

    cb_base_bull = struct_bull_ok & legs_ok_bull & (C_bull < A_bull) & \
                   (~D_gt_X_b) & (CD_bull > XA_bull) & (CD_bull > AB_bull) & (CD_bull > BC_bull)
    cb_base_bear = struct_bear_ok & legs_ok_bear & (C_bear > A_bear) & \
                   (~D_lt_X_s) & (CD_bear > XA_bear) & (CD_bear > AB_bear) & (CD_bear > BC_bear)

    cb_s1 = _ratio_best(rAB_XA_b, [0.382, 0.618], TOL)
    cb_s2 = _ratio_best(rBC_AB_b, [0.382, 0.886], TOL)
    cb_s3 = _ratio_best(rCD_BC_b, [2.618, 3.000, 3.618], TOL_WIDE)
    cb_s4 = _ratio_proximity(XD_ext_b, 1.618, TOL)
    cb_raw = np.nan_to_num(cb_s1 * 0.10 + cb_s2 * 0.10 + cb_s3 * 0.40 + np.nan_to_num(cb_s4, nan=0.0) * 0.40, nan=0.0)

    cs_s1 = _ratio_best(rAB_XA_s, [0.382, 0.618], TOL)
    cs_s2 = _ratio_best(rBC_AB_s, [0.382, 0.886], TOL)
    cs_s3 = _ratio_best(rCD_BC_s, [2.618, 3.000, 3.618], TOL_WIDE)
    cs_s4 = _ratio_proximity(XD_ext_s, 1.618, TOL)
    cs_raw = np.nan_to_num(cs_s1 * 0.10 + cs_s2 * 0.10 + cs_s3 * 0.40 + np.nan_to_num(cs_s4, nan=0.0) * 0.40, nan=0.0)

    # Crab has stricter volume requirement
    crab_vol_factor = np.where(v.values > vol_ma20 * 1.3, vol_factor, vol_factor * 0.80)

    crab_score = np.where(cb_base_bull, cb_raw * crab_vol_factor,
                 np.where(cb_base_bear, cs_raw * crab_vol_factor, 0.0))
    crab_score = np.clip(crab_score, 0.0, 1.0)

    crab_signal = np.where(cb_base_bull & (crab_score >= SCORE_THRESH), 1.0,
                  np.where(cb_base_bear & (crab_score >= SCORE_THRESH), -1.0, 0.0))

    # ==================================================================
    # SECTION 8 — Package results
    # ==================================================================
    results = {}

    results['harmonic_gartley_signal'] = pd.DataFrame(gartley_signal, index=idx, columns=cols)
    results['harmonic_gartley_score']  = pd.DataFrame(gartley_score, index=idx, columns=cols)

    results['harmonic_bat_signal'] = pd.DataFrame(bat_signal, index=idx, columns=cols)
    results['harmonic_bat_score']  = pd.DataFrame(bat_score, index=idx, columns=cols)

    results['harmonic_butterfly_signal'] = pd.DataFrame(butterfly_signal, index=idx, columns=cols)
    results['harmonic_butterfly_score']  = pd.DataFrame(butterfly_score, index=idx, columns=cols)

    results['harmonic_crab_signal'] = pd.DataFrame(crab_signal, index=idx, columns=cols)
    results['harmonic_crab_score']  = pd.DataFrame(crab_score, index=idx, columns=cols)

    return results


# ======================================================================
# School-compatible split-return interface
# ======================================================================

def compute_harmonic_core_score_split(indicators: dict):
    """
    Polarity-grouped split-return for _compute_school_harmonic integration.

    Returns
    -------
    (score_bull, score_bear, reasons_bull, reasons_bear)
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

        feats = compute_harmonic_core_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        def _read(key):
            df = feats.get(key)
            if df is not None and not df.empty:
                return df['S'].iloc[0]
            return 0.0

        # Gartley
        if _read('harmonic_gartley_signal') > 0:
            score_bull += 0.18; reasons_bull.append(f'Gartley看涨(score={_read("harmonic_gartley_score"):.2f})')
        elif _read('harmonic_gartley_signal') < 0:
            score_bear += 0.18; reasons_bear.append(f'Gartley看跌(score={_read("harmonic_gartley_score"):.2f})')

        # Bat — highest weight (Carney 85% win rate)
        if _read('harmonic_bat_signal') > 0:
            score_bull += 0.22; reasons_bull.append(f'Bat看涨(score={_read("harmonic_bat_score"):.2f})')
        elif _read('harmonic_bat_signal') < 0:
            score_bear += 0.22; reasons_bear.append(f'Bat看跌(score={_read("harmonic_bat_score"):.2f})')

        # Butterfly
        if _read('harmonic_butterfly_signal') > 0:
            score_bull += 0.16; reasons_bull.append(f'Butterfly看涨(score={_read("harmonic_butterfly_score"):.2f})')
        elif _read('harmonic_butterfly_signal') < 0:
            score_bear += 0.16; reasons_bear.append(f'Butterfly看跌(score={_read("harmonic_butterfly_score"):.2f})')

        # Crab — highest RR, but lower base weight due to rarity
        if _read('harmonic_crab_signal') > 0:
            score_bull += 0.20; reasons_bull.append(f'Crab看涨(score={_read("harmonic_crab_score"):.2f})')
        elif _read('harmonic_crab_signal') < 0:
            score_bear += 0.20; reasons_bear.append(f'Crab看跌(score={_read("harmonic_crab_score"):.2f})')

        return score_bull, score_bear, reasons_bull, reasons_bear
    except Exception:
        return 0.0, 0.0, [], []


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Harmonic Core Features Test ===")
    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=300)
        if df is not None and not df.empty:
            idx_d = df.index
            def _d(c_name):
                return pd.DataFrame({'S': df[c_name].values}, index=idx_d)
            feats = compute_harmonic_core_features(
                _d('open'), _d('high'), _d('low'), _d('close'), _d('volume'))
            print(f"\n{code}:")
            for k in ['harmonic_gartley_signal', 'harmonic_bat_signal',
                       'harmonic_butterfly_signal', 'harmonic_crab_signal']:
                sv = feats[k].values
                bull = int((sv > 0).sum())
                bear = int((sv < 0).sum())
                if bull + bear > 0:
                    print(f"  {k}: bull={bull} bear={bear}")
            # Score stats
            for k in ['harmonic_gartley_score', 'harmonic_bat_score',
                       'harmonic_butterfly_score', 'harmonic_crab_score']:
                sc = feats[k].values
                nz = int((sc > 0).sum())
                if nz > 0:
                    print(f"  {k}: nonzero={nz} max={sc.max():.3f}")
