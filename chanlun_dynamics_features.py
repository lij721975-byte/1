#!/usr/bin/env python
# chanlun_dynamics_features.py — Chanlun Dynamics & Probability Features
#
# C2: Trend vs Consolidation Divergence (趋势背驰 vs 盘整背驰)
# C4: Trend Completion Probability    (走势完美概率评分)
# C8: Multi-Dimensional Divergence    (背驰力度多维归一化)
#
# Pure vectorized NumPy/Pandas. ZERO for-loops.
# Polarity-separated: bullish (底背驰) → score_bull, bearish (顶背驰) → score_bear.
# All scores clipped to [0.0, 1.0]. EPS = 1e-8 on all denominators.

import numpy as np
import pandas as pd

EPS = 1e-8


# ======================================================================
# Main entry point
# ======================================================================

def compute_chanlun_dynamics_features(indicators: dict):
    """
    Compute Chanlun dynamics features from pre-computed structural indicators.

    Parameters
    ----------
    indicators : dict
        Must contain DataFrames (aligned index × columns):
          zs_count        — number of same-direction 中枢 in current trend
          bi_count        — number of 笔 since trend origin
          macd_area_seg1  — MACD histogram area of penultimate segment
          macd_area_seg2  — MACD histogram area of last (current) segment
          dif_extreme1    — DIF extreme value of penultimate segment
          dif_extreme2    — DIF extreme value of last segment
          amp_seg1        — price amplitude of penultimate segment
          amp_seg2        — price amplitude of last segment
          zs_zg           — last 中枢 upper bound (ZG)
          zs_zd           — last 中枢 lower bound (ZD)
        Optional:
          macd_hist_max1  — max single MACD histogram bar in seg1
          macd_hist_max2  — max single MACD histogram bar in seg2
          current_price   — current close price (for C4 amplitude exit)

    Returns
    -------
    dict[str, pd.DataFrame]
        chanlun_trend_div_bull_score     — trend divergence bullish  (0-1)
        chanlun_trend_div_bear_score     — trend divergence bearish  (0-1)
        chanlun_consolid_div_bull_score  — consolidation div bullish (0-1)
        chanlun_consolid_div_bear_score  — consolidation div bearish (0-1)
        chanlun_completion_prob          — trend completion probability (0-1)
        chanlun_multi_div_bull_score     — multi-dim divergence bullish (0-1)
        chanlun_multi_div_bear_score     — multi-dim divergence bearish (0-1)
    """
    # ---- Extract inputs with safe defaults ----
    def _get(key, default=None):
        v = indicators.get(key, default)
        if v is None:
            return default
        return v

    zs_count = _get('zs_count')
    bi_count = _get('bi_count')
    area1    = _get('macd_area_seg1')
    area2    = _get('macd_area_seg2')
    dif1     = _get('dif_extreme1')
    dif2     = _get('dif_extreme2')
    amp1     = _get('amp_seg1')
    amp2     = _get('amp_seg2')
    zg       = _get('zs_zg')
    zd       = _get('zs_zd')
    hist1    = _get('macd_hist_max1')
    hist2    = _get('macd_hist_max2')
    cp       = _get('current_price')

    # If critical inputs are missing, return zeros
    if any(v is None for v in [area1, area2, dif1, dif2, amp1, amp2]):
        return _empty_results(indicators)

    # Derive index/columns from any available DataFrame
    ref = area1
    idx, cols = ref.index, ref.columns

    # ---- Convert to numpy arrays ----
    zc = zs_count.values.astype(float) if zs_count is not None else np.zeros_like(ref.values)
    bc = bi_count.values.astype(float) if bi_count is not None else np.zeros_like(ref.values)
    a1 = np.abs(area1.values)  # MACD area is always positive (sum of abs histogram)
    a2 = np.abs(area2.values)
    d1 = dif1.values
    d2 = dif2.values
    p1 = amp1.values   # signed amplitude: + = upward segment, - = downward
    p2 = amp2.values
    zg_v = zg.values if zg is not None else np.full_like(ref.values, np.nan)
    zd_v = zd.values if zd is not None else np.full_like(ref.values, np.nan)
    h1 = np.abs(hist1.values) if hist1 is not None else None
    h2 = np.abs(hist2.values) if hist2 is not None else None
    cp_v = cp.values if cp is not None else np.full_like(ref.values, np.nan)

    # ==================================================================
    # Direction inference from segment amplitudes
    #   Both seg1 and seg2 in same direction (both up or both down)
    #   Upward   → checking for bearish divergence (顶背驰)
    #   Downward → checking for bullish divergence (底背驰)
    # ==================================================================
    upward_trend   = (p1 > 0) & (p2 > 0)
    downward_trend = (p1 < 0) & (p2 < 0)
    valid_pair     = upward_trend | downward_trend

    # ---- Common divergence metrics (direction-agnostic) ----
    area_ratio = a2 / (a1 + EPS)           # < 1 = shrinking MACD area
    amp_ratio  = np.abs(p2) / (np.abs(p1) + EPS)

    # DIF extreme comparison (direction-aware)
    # Upward: DIF peaks should increase; if DIF2_max < DIF1_max → bearish divergence
    # Downward: DIF troughs should get more negative; if DIF2_min > DIF1_min → bullish divergence
    dif_decline_bear = (d2 < d1) & upward_trend    # DIF peak lower → bearish
    dif_improve_bull = (d2 > d1) & downward_trend   # DIF trough higher → bullish

    # Price-energy efficiency: amplitude per unit MACD area
    eff1 = np.abs(p1) / (a1 + EPS)
    eff2 = np.abs(p2) / (a2 + EPS)
    eff_ratio = eff2 / (eff1 + EPS)       # > 1 = more amplitude per MACD unit = divergence
    eff_worse  = eff_ratio > 1.30          # efficiency degraded

    # ════════════════════════════════════════════════════════════════
    # C2 — Trend Divergence vs Consolidation Divergence
    # ════════════════════════════════════════════════════════════════

    # ---- Trend Divergence (趋势背驰): zs_count >= 2 ----
    is_trend = (zc >= 2) & valid_pair

    # TD-A: MACD area divergence (threshold 0.75)
    td_a = area_ratio < 0.75
    # TD-B: DIF extreme divergence
    td_b_bull = dif_improve_bull & is_trend & downward_trend
    td_b_bear = dif_decline_bear & is_trend & upward_trend
    # TD-C: Price-energy efficiency
    td_c = eff_worse & is_trend

    # Bullish trend divergence: TD-A + TD-B + TD-C, at least 2 of 3
    td_cond_bull = is_trend & downward_trend & (
        (td_a.astype(float) + td_b_bull.astype(float) + td_c.astype(float)) >= 2
    )
    # Bearish trend divergence
    td_cond_bear = is_trend & upward_trend & (
        (td_a.astype(float) + td_b_bear.astype(float) + td_c.astype(float)) >= 2
    )

    # Trend divergence strength
    td_str_bull = (
        np.clip(1.0 - area_ratio / 0.75, 0, 1) * 0.50 +
        np.where(dif_improve_bull, (d2 - d1) / (np.abs(d1) + EPS), 0).clip(0, 1) * 0.30 +
        np.clip((eff_ratio - 1.0) / 0.50, 0, 1) * 0.20
    )
    td_str_bear = (
        np.clip(1.0 - area_ratio / 0.75, 0, 1) * 0.50 +
        np.where(dif_decline_bear, (d1 - d2) / (np.abs(d1) + EPS), 0).clip(0, 1) * 0.30 +
        np.clip((eff_ratio - 1.0) / 0.50, 0, 1) * 0.20
    )

    trend_div_bull = np.where(td_cond_bull, np.clip(td_str_bull, 0.0, 1.0), 0.0)
    trend_div_bear = np.where(td_cond_bear, np.clip(td_str_bear, 0.0, 1.0), 0.0)

    # ---- Consolidation Divergence (盘整背驰): zs_count == 1 ----
    is_consolid = (zc >= 1) & (zc < 2) & valid_pair

    # CD-A: MACD area divergence (stricter threshold 0.65)
    cd_a = area_ratio < 0.65
    # CD-B: DIF zero-crossing — approximated by DIF extreme sign comparison
    cd_b_bull = dif_improve_bull & is_consolid & downward_trend
    cd_b_bear = dif_decline_bear & is_consolid & upward_trend
    # CD-C: Price moving away from 中枢
    cd_c_bull = is_consolid & downward_trend & (cp_v < zd_v)  # below ZD, continuing down
    cd_c_bear = is_consolid & upward_trend   & (cp_v > zg_v)  # above ZG, continuing up

    cd_cond_bull = is_consolid & downward_trend & cd_a & (
        cd_b_bull.astype(float) + cd_c_bull.astype(float) >= 1
    )
    cd_cond_bear = is_consolid & upward_trend & cd_a & (
        cd_b_bear.astype(float) + cd_c_bear.astype(float) >= 1
    )

    cd_str_bull = (
        np.clip(1.0 - area_ratio / 0.65, 0, 1) * 0.60 +
        np.where(cd_b_bull, 0.20, 0.0) +
        np.where(cd_c_bull, 0.20, 0.0)
    )
    cd_str_bear = (
        np.clip(1.0 - area_ratio / 0.65, 0, 1) * 0.60 +
        np.where(cd_b_bear, 0.20, 0.0) +
        np.where(cd_c_bear, 0.20, 0.0)
    )

    consolid_div_bull = np.where(cd_cond_bull, np.clip(cd_str_bull, 0.0, 1.0), 0.0)
    consolid_div_bear = np.where(cd_cond_bear, np.clip(cd_str_bear, 0.0, 1.0), 0.0)

    # ════════════════════════════════════════════════════════════════
    # C4 — Trend Completion Probability (走势完美概率)
    # ════════════════════════════════════════════════════════════════

    # Target 中枢 count: 2 for trend, 1 for consolidation
    target_zs = np.where(zc >= 2, 2.0, 1.0)

    # CP-A: 中枢 completion: min(1, zs_count / target)
    cp_a = np.clip(zc / (target_zs + EPS), 0.0, 1.0)

    # CP-B: Divergence presence
    #   Full divergence confirmed → 1.0
    #   Area shrinking (pre-divergence) → 0.3
    #   None → 0.0
    div_confirmed = (trend_div_bull > 0) | (trend_div_bear > 0) | \
                    (consolid_div_bull > 0) | (consolid_div_bear > 0)
    area_shrinking = (area_ratio < 0.90) & valid_pair & ~div_confirmed
    cp_b = np.where(div_confirmed, 1.0,
           np.where(area_shrinking, 0.30, 0.0))

    # CP-C: 笔 saturation (Fibonacci-style): min(1, bi_count / 9)
    cp_c = np.clip(bc / 9.0, 0.0, 1.0)

    # CP-D: Amplitude exit from last 中枢
    #   Bullish (price above ZG): exit_ratio = (close - ZG) / (ZG - ZD)
    #   Bearish (price below ZD): exit_ratio = (ZD - close) / (ZG - ZD)
    zs_range = zg_v - zd_v
    exit_up   = np.where(cp_v > zg_v, (cp_v - zg_v) / (zs_range + EPS), 0.0)
    exit_down = np.where(cp_v < zd_v, (zd_v - cp_v) / (zs_range + EPS), 0.0)
    exit_ratio = np.maximum(exit_up, exit_down)
    cp_d = np.clip(exit_ratio / 0.50, 0.0, 1.0)

    completion_prob = (
        cp_a * 0.35 +
        cp_b * 0.30 +
        cp_c * 0.15 +
        cp_d * 0.20
    )
    completion_prob = np.clip(completion_prob, 0.0, 1.0)

    # ════════════════════════════════════════════════════════════════
    # C8 — Multi-Dimensional Divergence Normalization
    # ════════════════════════════════════════════════════════════════

    # ---- Dimension 1: MACD Area Ratio ----
    # Normalized: area shrinking by 50% → score = 1.0
    dim1 = np.clip((1.0 - area_ratio) / 0.50, 0.0, 1.0)

    # ---- Dimension 2: DIF Extreme Ratio ----
    # Bullish (downward trend): DIF trough improvement → (DIF2 - DIF1) / |DIF1|
    # Bearish (upward trend): DIF peak decline → (DIF1 - DIF2) / |DIF1|
    dim2_bull = np.where(
        downward_trend & (d2 > d1),
        np.clip((d2 - d1) / (np.abs(d1) + EPS) / 0.30, 0.0, 1.0),
        0.0
    )
    dim2_bear = np.where(
        upward_trend & (d2 < d1),
        np.clip((d1 - d2) / (np.abs(d1) + EPS) / 0.30, 0.0, 1.0),
        0.0
    )

    # ---- Dimension 3: MACD Histogram Peak Ratio ----
    # Normalized: peak shrinking by 40% → score = 1.0
    if h1 is not None and h2 is not None:
        hist_ratio = h2 / (h1 + EPS)
        dim3 = np.clip((1.0 - hist_ratio) / 0.40, 0.0, 1.0)
        w3 = 0.15  # weight for dim3
    else:
        dim3 = np.zeros_like(a1)
        w3 = 0.0   # skip this dimension

    # ---- Dimension 4: Price-Energy Efficiency Ratio ----
    # Eff ratio > 1 means more price movement per unit MACD → divergence
    dim4 = np.clip((eff_ratio - 1.0) / 0.40, 0.0, 1.0)

    # ---- Composite weights (redistribute dim3 weight if missing) ----
    if w3 > 0:
        w = [0.35, 0.25, 0.15, 0.25]
    else:
        w = [0.40, 0.30, 0.00, 0.30]

    # ---- Multi-dim divergence: bullish (底背驰) ----
    multi_bull_raw = (
        dim1 * w[0] +
        dim2_bull * w[1] +
        dim3 * w[2] +
        dim4 * w[3]
    )
    multi_bull = np.where(downward_trend & valid_pair, np.clip(multi_bull_raw, 0.0, 1.0), 0.0)

    # ---- Multi-dim divergence: bearish (顶背驰) ----
    multi_bear_raw = (
        dim1 * w[0] +
        dim2_bear * w[1] +
        dim3 * w[2] +
        dim4 * w[3]
    )
    multi_bear = np.where(upward_trend & valid_pair, np.clip(multi_bear_raw, 0.0, 1.0), 0.0)

    # ---- Divergence confidence based on dimension confirmation ----
    # Count how many dimensions are active (dim_score >= 0.30)
    dim_count_bull = (
        (dim1 >= 0.30).astype(float) +
        (dim2_bull >= 0.30).astype(float) +
        ((dim3 >= 0.30) if w3 > 0 else np.zeros_like(dim1)).astype(float) +
        (dim4 >= 0.30).astype(float)
    )
    dim_count_bear = (
        (dim1 >= 0.30).astype(float) +
        (dim2_bear >= 0.30).astype(float) +
        ((dim3 >= 0.30) if w3 > 0 else np.zeros_like(dim1)).astype(float) +
        (dim4 >= 0.30).astype(float)
    )

    conf_boost_bull = np.where(dim_count_bull >= 4, 0.10,
                      np.where(dim_count_bull >= 3, 0.0, -0.25))
    conf_boost_bear = np.where(dim_count_bear >= 4, 0.10,
                      np.where(dim_count_bear >= 3, 0.0, -0.25))

    multi_bull = np.clip(np.where(multi_bull > 0, multi_bull + conf_boost_bull, 0.0), 0.0, 1.0)
    multi_bear = np.clip(np.where(multi_bear > 0, multi_bear + conf_boost_bear, 0.0), 0.0, 1.0)

    # ==================================================================
    # SECTION 8 — Package results
    # ==================================================================

    results = {}

    results['chanlun_trend_div_bull_score']    = pd.DataFrame(trend_div_bull, index=idx, columns=cols)
    results['chanlun_trend_div_bear_score']    = pd.DataFrame(trend_div_bear, index=idx, columns=cols)
    results['chanlun_consolid_div_bull_score'] = pd.DataFrame(consolid_div_bull, index=idx, columns=cols)
    results['chanlun_consolid_div_bear_score'] = pd.DataFrame(consolid_div_bear, index=idx, columns=cols)
    results['chanlun_completion_prob']         = pd.DataFrame(completion_prob, index=idx, columns=cols)
    results['chanlun_multi_div_bull_score']    = pd.DataFrame(multi_bull, index=idx, columns=cols)
    results['chanlun_multi_div_bear_score']    = pd.DataFrame(multi_bear, index=idx, columns=cols)

    return results


# ======================================================================
# Helper: empty results for safety
# ======================================================================

def _empty_results(indicators):
    """Return zero-filled results when critical inputs are missing."""
    ref = next((v for v in indicators.values()
                if isinstance(v, pd.DataFrame)), None)
    if ref is None:
        return {}
    idx, cols = ref.index, ref.columns
    zeros = pd.DataFrame(np.zeros_like(ref.values, dtype=float), index=idx, columns=cols)
    keys = [
        'chanlun_trend_div_bull_score', 'chanlun_trend_div_bear_score',
        'chanlun_consolid_div_bull_score', 'chanlun_consolid_div_bear_score',
        'chanlun_completion_prob',
        'chanlun_multi_div_bull_score', 'chanlun_multi_div_bear_score',
    ]
    return {k: zeros.copy() for k in keys}


# ======================================================================
# School-compatible split-return interface
# ======================================================================

def compute_chanlun_dynamics_score_split(indicators: dict):
    """
    Polarity-grouped split-return for _compute_school_chanlun integration.

    Consumes all 3 Chanlun dynamics features and returns:
      (score_bull, score_bear, reasons_bull, reasons_bear, completion_prob)

    completion_prob is a direction-NEUTRAL multiplier (走势完美 → 待反转).
    """
    try:
        feats = compute_chanlun_dynamics_features(indicators)

        def _read(key):
            df = feats.get(key)
            if df is not None and not df.empty:
                # Assume single-column; take first column's value
                v = df.values
                if v.ndim == 2:
                    return float(v[0, 0]) if v.size > 0 else 0.0
                return float(v[0]) if len(v) > 0 else 0.0
            return 0.0

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        # Trend divergence — highest weight (guaranteed pullback to 中枢)
        td_bull = _read('chanlun_trend_div_bull_score')
        td_bear = _read('chanlun_trend_div_bear_score')
        if td_bull > 0:
            score_bull += td_bull * 0.25
            reasons_bull.append(f'趋势底背驰(score={td_bull:.2f})→回拉中枢')
        if td_bear > 0:
            score_bear += td_bear * 0.25
            reasons_bear.append(f'趋势顶背驰(score={td_bear:.2f})→回拉中枢')

        # Consolidation divergence — lower weight (no guaranteed pullback)
        cd_bull = _read('chanlun_consolid_div_bull_score')
        cd_bear = _read('chanlun_consolid_div_bear_score')
        if cd_bull > 0:
            score_bull += cd_bull * 0.15
            reasons_bull.append(f'盘整底背驰(score={cd_bull:.2f})→动能衰减')
        if cd_bear > 0:
            score_bear += cd_bear * 0.15
            reasons_bear.append(f'盘整顶背驰(score={cd_bear:.2f})→动能衰减')

        # Multi-dimensional divergence — continuous scoring
        md_bull = _read('chanlun_multi_div_bull_score')
        md_bear = _read('chanlun_multi_div_bear_score')
        if md_bull > 0:
            score_bull += md_bull * 0.20
            if md_bull >= 0.75:
                reasons_bull.append(f'强多维底背驰({md_bull:.2f})→4/4维确认')
            elif md_bull >= 0.55:
                reasons_bull.append(f'标准多维底背驰({md_bull:.2f})')
        if md_bear > 0:
            score_bear += md_bear * 0.20
            if md_bear >= 0.75:
                reasons_bear.append(f'强多维顶背驰({md_bear:.2f})→4/4维确认')
            elif md_bear >= 0.55:
                reasons_bear.append(f'标准多维顶背驰({md_bear:.2f})')

        # Completion probability — neutral
        comp = _read('chanlun_completion_prob')

        return score_bull, score_bear, reasons_bull, reasons_bear, comp
    except Exception:
        return 0.0, 0.0, [], [], 0.0


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    from advanced_indicators import chanlun_analysis
    print("=== Chanlun Dynamics Features Test ===")
    for code in ['000012', '600519', '000651']:
        df = get_daily_kline(code, days=300)
        if df is None or df.empty:
            continue
        # Run full chanlun analysis to get structural indicators
        cl = chanlun_analysis(df)
        if cl is None:
            print(f"{code}: chanlun_analysis failed")
            continue
        idx_d = df.index
        cols_d = ['S']

        # Build indicator DataFrames from chanlun_analysis output
        # Extract scalar/series values and broadcast to DataFrames
        n = len(df)
        zc_arr = np.full((n, 1), np.nan)
        bc_arr = np.full((n, 1), np.nan)
        a1_arr = np.full((n, 1), np.nan)
        a2_arr = np.full((n, 1), np.nan)
        d1_arr = np.full((n, 1), np.nan)
        d2_arr = np.full((n, 1), np.nan)
        p1_arr = np.full((n, 1), np.nan)
        p2_arr = np.full((n, 1), np.nan)
        zg_arr = np.full((n, 1), np.nan)
        zd_arr = np.full((n, 1), np.nan)

        # From chanlun_analysis output
        trend_type = cl.get('trend_type', '')
        zs_count_val = cl.get('zhongshu_count', 0)
        bi_count_val = cl.get('bi_count', 0)
        div_detail = cl.get('divergence_detail', {}) or {}

        # Fill last bar's values (simplified test)
        zc_arr[-1, 0] = float(zs_count_val)
        bc_arr[-1, 0] = float(bi_count_val)

        # MACD area from divergence detail
        if isinstance(div_detail, dict):
            a1_arr[-1, 0] = float(div_detail.get('area1', 0))
            a2_arr[-1, 0] = float(div_detail.get('area2', 0))
            d1_arr[-1, 0] = float(div_detail.get('dif1', 0))
            d2_arr[-1, 0] = float(div_detail.get('dif2', 0))
            p1_arr[-1, 0] = float(div_detail.get('amp1', 0))
            p2_arr[-1, 0] = float(div_detail.get('amp2', 0))

        # ZG/ZD from zhongshu_list
        zs_list = cl.get('zhongshu_list', []) or []
        if len(zs_list) > 0:
            last_zs = zs_list[-1]
            if isinstance(last_zs, dict):
                zg_arr[-1, 0] = float(last_zs.get('ZG', np.nan))
                zd_arr[-1, 0] = float(last_zs.get('ZD', np.nan))

        # Build indicators dict
        ind = {
            'zs_count': pd.DataFrame(zc_arr, index=idx_d, columns=cols_d),
            'bi_count': pd.DataFrame(bc_arr, index=idx_d, columns=cols_d),
            'macd_area_seg1': pd.DataFrame(a1_arr, index=idx_d, columns=cols_d),
            'macd_area_seg2': pd.DataFrame(a2_arr, index=idx_d, columns=cols_d),
            'dif_extreme1': pd.DataFrame(d1_arr, index=idx_d, columns=cols_d),
            'dif_extreme2': pd.DataFrame(d2_arr, index=idx_d, columns=cols_d),
            'amp_seg1': pd.DataFrame(p1_arr, index=idx_d, columns=cols_d),
            'amp_seg2': pd.DataFrame(p2_arr, index=idx_d, columns=cols_d),
            'zs_zg': pd.DataFrame(zg_arr, index=idx_d, columns=cols_d),
            'zs_zd': pd.DataFrame(zd_arr, index=idx_d, columns=cols_d),
            'current_price': pd.DataFrame(df['close'].values.reshape(-1, 1), index=idx_d, columns=cols_d),
        }

        feats = compute_chanlun_dynamics_features(ind)
        print(f"\n{code}:")
        for k, v in feats.items():
            nz = int((v.values > 0).sum())
            if nz > 0:
                mx = v.values.max()
                print(f"  {k}: nonzero={nz} max={mx:.3f}")
