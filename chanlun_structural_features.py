#!/usr/bin/env python
# chanlun_structural_features.py — Chanlun Structural & Oscillation Features
#
# C1: Feature Sequence Destruction    (特征序列破坏 — Case 1 proxy)
# C3: Zhongshu Oscillation Buy/Sell   (中枢震荡自动买卖点)
# C5: Third Buy/Sell Retest           (第三类买卖点次级别回试代理)
# C7: Bardo Stage Dynamic Pattern     (中阴阶段动态模式识别)
#
# Pure vectorized NumPy/Pandas. ZERO for-loops. ZERO shift(-1).
# C7 outputs are NEUTRAL (new_zs_prob, old_zs_prob) — not scored to bull/bear.

import numpy as np
import pandas as pd

EPS = 1e-8


# ======================================================================
# Main entry point
# ======================================================================

def compute_chanlun_structural_features(indicators: dict):
    """
    Compute Chanlun structural features from pre-computed indicators.

    Parameters
    ----------
    indicators : dict
        Required DataFrames (aligned index × columns):
          close_df, open_df, high_df, low_df, volume_df  — OHLCV
          zs_zn_df        — Zn monitor values (中枢震荡位置)
          zs_zg_df        — last 中枢 upper bound (ZG)
          zs_zd_df        — last 中枢 lower bound (ZD)
          is_up_bi        — bool/int: current 笔 is upward
          is_down_bi      — bool/int: current 笔 is downward
          bi_start_price  — start price of current 笔
          bi_end_price    — end price of current 笔 (latest confirmed)
          is_up_fractal   — bool/int: top fractal (顶分型) present at current bar
          is_down_fractal — bool/int: bottom fractal (底分型) present
          rsi_df          — RSI(14) values
          vol_ma20_df     — 20-period volume moving average
        Optional:
          bb_upper, bb_mid, bb_lower  — Bollinger Bands (for C7)
          zs_count_df    — number of 中枢 in current trend (for C7 exit check)

    Returns
    -------
    dict[str, pd.DataFrame]
        chanlun_case1_bull_score      — C1 case-1 destruction bullish  (0-1)
        chanlun_case1_bear_score      — C1 case-1 destruction bearish  (0-1)
        chanlun_shock_buy_score       — C3 oscillation buy score      (0-1)
        chanlun_shock_sell_score      — C3 oscillation sell score     (0-1)
        chanlun_buy3_score            — C5 third-buy retest score     (0-1)
        chanlun_sell3_score           — C5 third-sell retest score    (0-1)
        chanlun_new_zs_prob           — C7 new-中枢 forming prob      (0-1)
        chanlun_old_zs_expand_prob    — C7 old-中枢 expanding prob    (0-1)
    """
    # ---- Extract inputs ----
    def _get(key):
        return indicators.get(key)

    close   = _get('close_df')
    open_   = _get('open_df')
    high    = _get('high_df')
    low     = _get('low_df')
    volume  = _get('volume_df')
    zn      = _get('zs_zn_df')
    zg      = _get('zs_zg_df')
    zd      = _get('zs_zd_df')
    up_bi   = _get('is_up_bi')
    dn_bi   = _get('is_down_bi')
    bi_st   = _get('bi_start_price')
    bi_end  = _get('bi_end_price')
    up_fx   = _get('is_up_fractal')
    dn_fx   = _get('is_down_fractal')
    rsi     = _get('rsi_df')
    vol_ma  = _get('vol_ma20_df')
    bb_u    = _get('bb_upper')
    bb_m    = _get('bb_mid')
    bb_l    = _get('bb_lower')
    zs_cnt  = _get('zs_count_df')

    required = [close, open_, high, low, volume]
    if any(v is None for v in required):
        return _empty_structural(indicators)

    idx, cols = close.index, close.columns

    # ---- Convert to numpy ----
    c_arr = close.values; o_arr = open_.values
    h_arr = high.values;  l_arr = low.values
    v_arr = volume.values

    zn_v  = zn.values  if zn  is not None else np.zeros_like(c_arr)
    zg_v  = zg.values  if zg  is not None else np.full_like(c_arr, np.nan)
    zd_v  = zd.values  if zd  is not None else np.full_like(c_arr, np.nan)
    upb   = (up_bi.values.astype(bool)  if up_bi  is not None else np.zeros_like(c_arr, dtype=bool))
    dnb   = (dn_bi.values.astype(bool)  if dn_bi  is not None else np.zeros_like(c_arr, dtype=bool))
    bs_v  = bi_st.values if bi_st is not None else np.full_like(c_arr, np.nan)
    be_v  = bi_end.values if bi_end is not None else np.full_like(c_arr, np.nan)
    ufx   = (up_fx.values.astype(bool) if up_fx is not None else np.zeros_like(c_arr, dtype=bool))
    dfx   = (dn_fx.values.astype(bool) if dn_fx is not None else np.zeros_like(c_arr, dtype=bool))
    rs_v  = rsi.values if rsi is not None else np.full_like(c_arr, 50.0)
    vm_v  = vol_ma.values if vol_ma is not None else np.ones_like(c_arr)

    # Optional BB
    bb_u_v = bb_u.values if bb_u is not None else None
    bb_l_v = bb_l.values if bb_l is not None else None
    bb_m_v = bb_m.values if bb_m is not None else None

    # Optional zs_count
    zsc_v = zs_cnt.values if zs_cnt is not None else np.zeros_like(c_arr)

    # ---- Derived: K-line geometry ----
    spread  = h_arr - l_arr
    up_wick = h_arr - np.maximum(o_arr, c_arr)
    lo_wick = np.minimum(o_arr, c_arr) - l_arr
    body    = np.abs(c_arr - o_arr)
    close_pos = (c_arr - l_arr) / (spread + EPS)

    # ---- Derived: shift-1 helpers (NO shift(-1) anywhere) ----
    def _s1(arr):
        out = np.roll(arr, 1, axis=0)
        out[0] = arr[0]
        return out

    c_s1 = _s1(c_arr); o_s1 = _s1(o_arr)
    h_s1 = _s1(h_arr); l_s1 = _s1(l_arr)
    v_s1 = _s1(v_arr)

    # ════════════════════════════════════════════════════════════════
    # C1 — Feature Sequence Destruction (Case 1 Proxy)
    #   Case 1: reverse force breaks through previous 笔's start point.
    #   Upward 笔: if close < bi_start → the upward structure is violated → bearish
    #   Downward 笔: if close > bi_start → the downward structure is violated → bullish
    # ════════════════════════════════════════════════════════════════

    # Case 1-A: 标准破坏 — 反向笔击穿前一笔起点
    case1_bear_raw = upb & (c_arr < bs_v)   # upward 笔 being destroyed from above
    case1_bull_raw = dnb & (c_arr > bs_v)   # downward 笔 being destroyed from below

    # Penetration depth matters — deeper break = higher confidence
    penetration_bear = np.where(case1_bear_raw,
                               (bs_v - c_arr) / (np.abs(bs_v) + EPS), 0.0)
    penetration_bull = np.where(case1_bull_raw,
                               (c_arr - bs_v) / (np.abs(bs_v) + EPS), 0.0)

    case1_bear_score = np.where(case1_bear_raw,
                                np.clip(penetration_bear * 5.0, 0.30, 0.90), 0.0)
    case1_bull_score = np.where(case1_bull_raw,
                                np.clip(penetration_bull * 5.0, 0.30, 0.90), 0.0)

    # ════════════════════════════════════════════════════════════════
    # C3 — Zhongshu Oscillation Buy/Sell Points (中枢震荡买卖点)
    # ════════════════════════════════════════════════════════════════

    # ---- Shock Buy (震荡买点 — 下沿支撑) ----
    sb_a = (zn_v >= -1.5) & (zn_v <= -0.5)           # Zn in lower zone
    sb_b = dfx                                        # bottom fractal present
    sb_c = rs_v < 35                                  # RSI oversold
    sb_d = v_arr < vm_v * 0.80                        # volume contraction
    # RSI bullish divergence check: price at 5d low but RSI not at 5d low
    c_5d_low = pd.DataFrame(c_arr, index=idx, columns=cols).rolling(5, min_periods=3).min().values
    rsi_5d_low = pd.DataFrame(rs_v, index=idx, columns=cols).rolling(5, min_periods=3).min().values
    sb_e = (c_arr <= c_5d_low * 1.005) & (rs_v > rsi_5d_low * 1.02)  # RSI divergence

    shock_buy_cond_count = (
        sb_a.astype(float) + sb_b.astype(float) + sb_c.astype(float) +
        sb_d.astype(float) + sb_e.astype(float)
    )
    shock_buy_raw = (
        np.where(sb_a, np.clip((-zn_v - 0.5) / 1.0, 0, 1) * 0.25, 0.0) +
        np.where(sb_b, 0.30, 0.0) +
        np.where(sb_c, np.clip((35 - rs_v) / 15, 0, 1) * 0.20, 0.0) +
        np.where(sb_d, np.clip((vm_v * 0.80 - v_arr) / (vm_v * 0.30 + EPS), 0, 1) * 0.15, 0.0) +
        np.where(sb_e, 0.10, 0.0)
    )
    # Penalty: MACD direction opposite — approximated by close position
    weak_close = close_pos < 0.40  # closed near low → momentum still down
    shock_buy_score = np.where(shock_buy_cond_count >= 3,
                               np.where(weak_close, shock_buy_raw * 0.70, shock_buy_raw),
                               0.0)
    shock_buy_score = np.clip(shock_buy_score, 0.0, 1.0)

    # ---- Shock Sell (震荡卖点 — 上沿阻力) ----
    ss_a = (zn_v >= 0.5) & (zn_v <= 1.5)             # Zn in upper zone
    ss_b = ufx                                        # top fractal present
    ss_c = rs_v > 65                                  # RSI overbought
    ss_d = (v_arr > vm_v * 1.2) & (up_wick / (spread + EPS) > 0.40)  # high-vol stagnation
    # RSI bearish divergence
    c_5d_high = pd.DataFrame(c_arr, index=idx, columns=cols).rolling(5, min_periods=3).max().values
    rsi_5d_high = pd.DataFrame(rs_v, index=idx, columns=cols).rolling(5, min_periods=3).max().values
    ss_e = (c_arr >= c_5d_high * 0.995) & (rs_v < rsi_5d_high * 0.98)

    shock_sell_cond_count = (
        ss_a.astype(float) + ss_b.astype(float) + ss_c.astype(float) +
        ss_d.astype(float) + ss_e.astype(float)
    )
    shock_sell_raw = (
        np.where(ss_a, np.clip((zn_v - 0.5) / 1.0, 0, 1) * 0.25, 0.0) +
        np.where(ss_b, 0.30, 0.0) +
        np.where(ss_c, np.clip((rs_v - 65) / 15, 0, 1) * 0.20, 0.0) +
        np.where(ss_d, 0.15, 0.0) +
        np.where(ss_e, 0.10, 0.0)
    )
    strong_close = close_pos > 0.60
    shock_sell_score = np.where(shock_sell_cond_count >= 3,
                                np.where(strong_close, shock_sell_raw * 0.70, shock_sell_raw),
                                0.0)
    shock_sell_score = np.clip(shock_sell_score, 0.0, 1.0)

    # ════════════════════════════════════════════════════════════════
    # C5 — Third Buy/Sell Point with Sub-level Retest Proxy
    #   All logic stands at T (confirmation day), looks back via shift.
    # ════════════════════════════════════════════════════════════════

    # ---- Breakout detection (looking back from T) ----
    # Breakout happened at bar K: close[K] > ZG[K] AND close[K-1] <= ZG[K-1]
    breakout_up = (c_arr > zg_v) & (c_s1 <= _s1(zg_v))
    breakout_dn = (c_arr < zd_v) & (c_s1 >= _s1(zd_v))

    # ---- Third Buy (三买) — break above ZG then retest ZG support ----
    # Retest: low touches ZG ±3% zone
    retest_buy_zone = (l_arr >= zg_v * 0.97) & (l_arr <= zg_v * 1.03)

    # T-1 or T-2 had retest touch (look back up to 3 bars)
    l_s2 = _s1(l_s1)
    retest_buy_zone_s1 = _s1(retest_buy_zone)
    retest_buy_zone_s2 = _s1(retest_buy_zone_s1)
    retest_buy_recent = retest_buy_zone_s1 | retest_buy_zone_s2

    # T-1, T-2, or T-3 had breakout
    breakout_up_s1 = _s1(breakout_up)
    breakout_up_s2 = _s1(breakout_up_s1)
    breakout_up_s3 = _s1(breakout_up_s2)
    breakout_up_recent = breakout_up_s1 | breakout_up_s2 | breakout_up_s3

    # T-day confirmation: bullish close, low volume, price above ZG
    buy3_t_confirm = (c_arr > o_arr) & (v_arr < vm_v * 0.75) & (c_arr > zg_v * 1.01)

    buy3_base = buy3_t_confirm & retest_buy_recent & breakout_up_recent
    # Strength: how clean the retest was
    buy3_strength = (
        np.where(buy3_base, 0.35, 0.0) +                                    # structure met
        np.where(buy3_base & retest_buy_zone_s1, 0.25, 0.0) +               # retest yesterday (fresher)
        np.where(buy3_base & (v_arr < vm_v * 0.50), 0.15, 0.0) +            # ultra-low vol
        np.where(buy3_base & (c_arr - o_arr > body.mean() * 0.5), 0.15, 0.0) + # strong bullish candle
        np.where(buy3_base & (breakout_up_s1 | breakout_up_s2), 0.10, 0.0)  # breakout was recent
    )
    buy3_score = np.clip(buy3_strength, 0.0, 1.0)

    # ---- Third Sell (三卖) — break below ZD then retest ZD resistance ----
    retest_sell_zone = (h_arr >= zd_v * 0.97) & (h_arr <= zd_v * 1.03)
    retest_sell_zone_s1 = _s1(retest_sell_zone)
    retest_sell_zone_s2 = _s1(retest_sell_zone_s1)
    retest_sell_recent = retest_sell_zone_s1 | retest_sell_zone_s2

    breakout_dn_s1 = _s1(breakout_dn)
    breakout_dn_s2 = _s1(breakout_dn_s1)
    breakout_dn_s3 = _s1(breakout_dn_s2)
    breakout_dn_recent = breakout_dn_s1 | breakout_dn_s2 | breakout_dn_s3

    sell3_t_confirm = (c_arr < o_arr) & (v_arr < vm_v * 0.75) & (c_arr < zd_v * 0.99)
    sell3_base = sell3_t_confirm & retest_sell_recent & breakout_dn_recent

    sell3_strength = (
        np.where(sell3_base, 0.35, 0.0) +
        np.where(sell3_base & retest_sell_zone_s1, 0.25, 0.0) +
        np.where(sell3_base & (v_arr < vm_v * 0.50), 0.15, 0.0) +
        np.where(sell3_base & (o_arr - c_arr > body.mean() * 0.5), 0.15, 0.0) +
        np.where(sell3_base & (breakout_dn_s1 | breakout_dn_s2), 0.10, 0.0)
    )
    sell3_score = np.clip(sell3_strength, 0.0, 1.0)

    # ════════════════════════════════════════════════════════════════
    # C7 — Bardo Stage Dynamic Pattern (中阴阶段 — NEUTRAL)
    # ════════════════════════════════════════════════════════════════

    zs_range = zg_v - zd_v  # old 中枢 width
    half_range = zs_range / 2.0

    # Exit from old 中枢
    exited_up   = c_arr > zg_v * 1.02
    exited_down = c_arr < zd_v * 0.98
    exited_zs   = exited_up | exited_down

    # Recent price range (20-bar rolling)
    rng_high = pd.DataFrame(h_arr, index=idx, columns=cols).rolling(20, min_periods=10).max().values
    rng_low  = pd.DataFrame(l_arr, index=idx, columns=cols).rolling(20, min_periods=10).min().values
    recent_range = rng_high - rng_low

    # ---- B1: New 中枢 Forming Probability ----
    # B1-A: exited old 中枢
    b1a = exited_zs
    # B1-B: early sign of new range — recent_range < old_range * 1.20
    b1b = recent_range < zs_range * 1.20
    # B1-C: volatility contracting (BB squeeze proxy via recent_range trend)
    rng_10d_before = pd.DataFrame(h_arr, index=idx, columns=cols).shift(10).rolling(10, min_periods=5).max().values - \
                     pd.DataFrame(l_arr, index=idx, columns=cols).shift(10).rolling(10, min_periods=5).min().values
    b1c = (recent_range < rng_10d_before * 0.85) | (bb_u_v is not None and bb_l_v is not None and bb_m_v is not None and
           ((bb_u_v - bb_l_v) / (bb_m_v + EPS) <
            _s1((bb_u_v - bb_l_v) / (bb_m_v + EPS)) * 0.90))

    new_zs_prob = (
        b1a.astype(float) * 0.25 +
        b1b.astype(float) * 0.40 +
        b1c.astype(float) * 0.35
    )
    new_zs_prob = np.clip(new_zs_prob, 0.0, 1.0)

    # ---- B2: Old 中枢 Expanding Probability ----
    # B2-A: price repeatedly returns to old 中枢 zone
    #   Count entries into [ZD_old, ZG_old] in last 30 bars
    in_old_zs = (l_arr <= zg_v) & (h_arr >= zd_v)
    returns_to_zs = pd.DataFrame(in_old_zs.astype(float), index=idx, columns=cols) \
                      .rolling(30, min_periods=10).sum().values
    b2a = returns_to_zs >= 2.0  # at least 2 returns to old zone

    # B2-B: volatility expanding
    b2b = recent_range > zs_range * 1.50
    # B2-C: no new 中枢 forming (B1-B fails)
    b2c = ~b1b

    old_zs_expand_prob = (
        b2a.astype(float) * 0.35 +
        b2b.astype(float) * 0.40 +
        b2c.astype(float) * 0.25
    )
    old_zs_expand_prob = np.clip(old_zs_expand_prob, 0.0, 1.0)

    # ---- B3: Undetermined — when neither probability is high ----
    # Both probabilities share the same space; the dominant one wins
    # (upstream consumer decides based on both values)

    # ==================================================================
    # Package results
    # ==================================================================

    results = {}

    results['chanlun_case1_bull_score']   = pd.DataFrame(case1_bull_score, index=idx, columns=cols)
    results['chanlun_case1_bear_score']   = pd.DataFrame(case1_bear_score, index=idx, columns=cols)
    results['chanlun_shock_buy_score']    = pd.DataFrame(shock_buy_score, index=idx, columns=cols)
    results['chanlun_shock_sell_score']   = pd.DataFrame(shock_sell_score, index=idx, columns=cols)
    results['chanlun_buy3_score']         = pd.DataFrame(buy3_score, index=idx, columns=cols)
    results['chanlun_sell3_score']        = pd.DataFrame(sell3_score, index=idx, columns=cols)
    results['chanlun_new_zs_prob']        = pd.DataFrame(new_zs_prob, index=idx, columns=cols)
    results['chanlun_old_zs_expand_prob'] = pd.DataFrame(old_zs_expand_prob, index=idx, columns=cols)

    return results


# ======================================================================
# Empty results fallback
# ======================================================================

def _empty_structural(indicators):
    ref = next((v for v in indicators.values()
                if isinstance(v, pd.DataFrame)), None)
    if ref is None:
        return {}
    idx, cols = ref.index, ref.columns
    z = pd.DataFrame(np.zeros_like(ref.values, dtype=float), index=idx, columns=cols)
    keys = [
        'chanlun_case1_bull_score', 'chanlun_case1_bear_score',
        'chanlun_shock_buy_score', 'chanlun_shock_sell_score',
        'chanlun_buy3_score', 'chanlun_sell3_score',
        'chanlun_new_zs_prob', 'chanlun_old_zs_expand_prob',
    ]
    return {k: z.copy() for k in keys}


# ======================================================================
# School-compatible split-return interface (6-tuple)
# ======================================================================

def compute_chanlun_structural_score_split(indicators: dict):
    """
    Polarity-grouped split-return for _compute_school_chanlun integration.

    Returns
    -------
    (score_bull, score_bear, reasons_bull, reasons_bear,
     new_zs_prob, old_zs_expand_prob)

    new_zs_prob and old_zs_expand_prob are DIRECTION-NEUTRAL.
    """
    try:
        feats = compute_chanlun_structural_features(indicators)

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

        # C1: Case 1 destruction
        c1_bull = _read('chanlun_case1_bull_score')
        c1_bear = _read('chanlun_case1_bear_score')
        if c1_bull > 0:
            score_bull += c1_bull * 0.22
            reasons_bull.append(f'Case1标准破坏(向下笔终结, score={c1_bull:.2f})')
        if c1_bear > 0:
            score_bear += c1_bear * 0.22
            reasons_bear.append(f'Case1标准破坏(向上笔终结, score={c1_bear:.2f})')

        # C3: Shock oscillation
        sb_bull = _read('chanlun_shock_buy_score')
        sb_bear = _read('chanlun_shock_sell_score')
        if sb_bull > 0:
            score_bull += sb_bull * 0.15
            reasons_bull.append(f'中枢震荡买点(Zn下沿, score={sb_bull:.2f})')
        if sb_bear > 0:
            score_bear += sb_bear * 0.15
            reasons_bear.append(f'中枢震荡卖点(Zn上沿, score={sb_bear:.2f})')

        # C5: Third buy/sell
        b3 = _read('chanlun_buy3_score')
        s3 = _read('chanlun_sell3_score')
        if b3 > 0:
            score_bull += b3 * 0.20
            reasons_bull.append(f'三买(突破ZG+回试确认, score={b3:.2f})')
        if s3 > 0:
            score_bear += s3 * 0.20
            reasons_bear.append(f'三卖(跌破ZD+回试确认, score={s3:.2f})')

        # C7: Bardo stage (neutral)
        new_zs = _read('chanlun_new_zs_prob')
        old_zs = _read('chanlun_old_zs_expand_prob')
        if new_zs > 0.50:
            reasons_bull.append(f'中阴:新中枢形成中(prob={new_zs:.2f})')
        if old_zs > 0.50:
            reasons_bull.append(f'中阴:原中枢扩展中(prob={old_zs:.2f})')

        return score_bull, score_bear, reasons_bull, reasons_bear, new_zs, old_zs
    except Exception:
        return 0.0, 0.0, [], [], 0.0, 0.0


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    from advanced_indicators import chanlun_analysis
    print("=== Chanlun Structural Features Test ===")
    for code in ['000651', '300750']:
        df = get_daily_kline(code, days=300)
        if df is None or df.empty:
            continue
        cl = chanlun_analysis(df)
        if cl is None:
            print(f"{code}: chanlun_analysis failed")
            continue

        idx_d = df.index; n = len(df)
        # Build indicator DataFrames
        zn_val = cl.get('zn_value', 0)
        zg_val = cl.get('zs_zg', np.nan) or np.nan
        zd_val = cl.get('zs_zd', np.nan) or np.nan

        # Detect 笔 direction from stroke_state
        stroke_dir = cl.get('stroke_direction', 0)
        up_bi_arr = np.zeros((n, 1)); dn_bi_arr = np.zeros((n, 1))
        up_bi_arr[-1, 0] = 1.0 if stroke_dir == 1 else 0.0
        dn_bi_arr[-1, 0] = 1.0 if stroke_dir == -1 else 0.0

        # Fractal from last fractal type
        fractal_type = cl.get('fractal_type', '')
        up_fx_arr = np.zeros((n, 1)); dn_fx_arr = np.zeros((n, 1))
        up_fx_arr[-1, 0] = 1.0 if '顶分型' in str(fractal_type) else 0.0
        dn_fx_arr[-1, 0] = 1.0 if '底分型' in str(fractal_type) else 0.0

        # Simple RSI approx
        delta = df['close'].diff().values
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).ewm(span=14, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=14, adjust=False).mean().values
        rs_arr = avg_gain / (avg_loss + EPS)
        rsi_arr = 100.0 - (100.0 / (1.0 + rs_arr))

        ind = {
            'close_df': pd.DataFrame(df['close'].values.reshape(-1, 1), index=idx_d, columns=['S']),
            'open_df':  pd.DataFrame(df['open'].values.reshape(-1, 1), index=idx_d, columns=['S']),
            'high_df':  pd.DataFrame(df['high'].values.reshape(-1, 1), index=idx_d, columns=['S']),
            'low_df':   pd.DataFrame(df['low'].values.reshape(-1, 1), index=idx_d, columns=['S']),
            'volume_df':pd.DataFrame(df['volume'].values.reshape(-1, 1), index=idx_d, columns=['S']),
            'zs_zn_df': pd.DataFrame(np.full((n, 1), float(zn_val)), index=idx_d, columns=['S']),
            'zs_zg_df': pd.DataFrame(np.full((n, 1), float(zg_val)), index=idx_d, columns=['S']),
            'zs_zd_df': pd.DataFrame(np.full((n, 1), float(zd_val)), index=idx_d, columns=['S']),
            'is_up_bi': pd.DataFrame(up_bi_arr, index=idx_d, columns=['S']),
            'is_down_bi': pd.DataFrame(dn_bi_arr, index=idx_d, columns=['S']),
            'bi_start_price': pd.DataFrame(np.full((n, 1), float(df['close'].iloc[-20])), index=idx_d, columns=['S']),
            'bi_end_price': pd.DataFrame(np.full((n, 1), float(df['close'].iloc[-1])), index=idx_d, columns=['S']),
            'is_up_fractal': pd.DataFrame(up_fx_arr, index=idx_d, columns=['S']),
            'is_down_fractal': pd.DataFrame(dn_fx_arr, index=idx_d, columns=['S']),
            'rsi_df': pd.DataFrame(rsi_arr.reshape(-1, 1), index=idx_d, columns=['S']),
            'vol_ma20_df': pd.DataFrame(df['volume'].rolling(20, min_periods=10).mean().values.reshape(-1, 1),
                                        index=idx_d, columns=['S']),
        }

        feats = compute_chanlun_structural_features(ind)
        print(f"\n{code}:")
        for k, v in feats.items():
            nz = int((v.values > 0).sum())
            if nz > 0:
                mx = v.values.max()
                print(f"  {k}: nonzero={nz}/{n} max={mx:.3f}")
