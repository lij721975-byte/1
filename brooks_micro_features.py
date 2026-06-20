#!/usr/bin/env python
# brooks_micro_features.py — Al Brooks PA Part 2: State Machine & Entry
#
# T2: H1-H4 / L1-L4 Pullback Counting  (cumsum vectorized state machine)
# T5: Signal Bar + Entry Bar            (2-step confirmation)
# T8: Bull/Bear Trap + Failed Breakout  (T3 subsumed → T8 superset)
#
# Pure vectorized. ZERO for-loops. ZERO shift(-1).
# Grouped cumsum via numpy cumsum + ffill baseline trick.

import numpy as np
import pandas as pd

EPS = 1e-8


# ======================================================================
# Main entry point
# ======================================================================

def compute_brooks_micro_features(open_df, high_df, low_df, close_df, volume_df,
                                   trend_bull_df=None, trend_bear_df=None,
                                   always_in_df=None):
    """
    Compute Brooks PA micro state-machine features.

    Parameters
    ----------
    open_df, high_df, low_df, close_df, volume_df : pd.DataFrame
        Wide-format OHLCV.
    trend_bull_df, trend_bear_df : pd.DataFrame, optional
        From Part 1 T1 output (brooks_trend_bull/bear_score).
    always_in_df : pd.DataFrame, optional
        From Part 1 T7 output (brooks_always_in_score).

    Returns
    -------
    dict[str, pd.DataFrame]
        brooks_H1_score .. H4_score   — H1-H4 bullish pullback scores (0-1)
        brooks_L1_score .. L4_score   — L1-L4 bearish pullback scores (0-1)
        brooks_entry_bull_score       — T5 bull entry confirmation (0-1)
        brooks_entry_bear_score       — T5 bear entry confirmation (0-1)
        brooks_bull_trap_score        — T8 bull trap → bearish signal (0-1)
        brooks_bear_trap_score        — T8 bear trap → bullish signal (0-1)
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

    # ---- Use provided or compute inline ----
    if trend_bull_df is not None:
        trend_bull = trend_bull_df.values
    else:
        trend_bull = np.zeros_like(c.values)
    if trend_bear_df is not None:
        trend_bear = trend_bear_df.values
    else:
        trend_bear = np.zeros_like(c.values)
    if always_in_df is not None:
        always_in = always_in_df.values
    else:
        always_in = np.zeros_like(c.values)

    # ---- Arrays ----
    o_arr, h_arr, l_arr, c_arr, v_arr = o.values, h.values, l.values, c.values, v.values
    spread_arr   = h_arr - l_arr
    body_arr     = np.abs(c_arr - o_arr)
    body_pct     = body_arr / (spread_arr + EPS)
    lo_w_pct     = (np.minimum(o_arr, c_arr) - l_arr) / (spread_arr + EPS)
    up_w_pct     = (h_arr - np.maximum(o_arr, c_arr)) / (spread_arr + EPS)

    is_bull_bar  = c_arr > o_arr
    is_bear_bar  = c_arr < o_arr

    # ---- ATR(14) & Volume MA ----
    tr_vals = np.fmax.reduce([
        h_arr - l_arr,
        np.abs(h_arr - np.roll(c_arr, 1, axis=0)),
        np.abs(l_arr - np.roll(c_arr, 1, axis=0)),
    ])
    tr_vals[0] = h_arr[0] - l_arr[0]
    tr_df = pd.DataFrame(tr_vals, index=idx, columns=cols)
    atr14 = tr_df.ewm(span=14, adjust=False).mean().values
    vol_ma20 = v.rolling(20, min_periods=10).mean().values

    # ---- Shift helpers (NO shift(-1)) ----
    def _s1(arr):
        out = np.roll(arr, 1, axis=0); out[0] = arr[0]; return out

    h_s1 = _s1(h_arr); l_s1 = _s1(l_arr); c_s1 = _s1(c_arr)
    o_s1 = _s1(o_arr); v_s1 = _s1(v_arr)

    # ════════════════════════════════════════════════════════════════
    # T2 — H1-H4 / L1-L4 Pullback Counting (cumsum vectorized)
    # ════════════════════════════════════════════════════════════════

    # ---- Common: rolling 20-bar high/low (shifted for anti-lookahead) ----
    high_20 = h.rolling(20, min_periods=10).max().shift(1).values
    low_20  = l.rolling(20, min_periods=10).min().shift(1).values

    # ══ Bullish H-counting (active when always_in > 0.15) ══
    bull_trend_active = always_in > 0.15

    # Reset mask: strong bull trend bar appears → new H-counting cycle
    reset_H = (trend_bull > 0.70) & bull_trend_active
    reset_H_int = reset_H.astype(float)

    # Pullback bar definition (bearish bar during bull trend)
    is_pullback_H = (
        is_bear_bar & bull_trend_active &
        (l_arr < l_s1) & (h_arr < h_s1)
    ).astype(float)

    # ---- Vectorized group-cumsum trick ----
    # H_count = cumsum(pullbacks) - cumsum_at_last_reset
    pullback_cum_H = np.cumsum(is_pullback_H, axis=0)

    # Baseline: cumsum value just BEFORE each reset bar
    baseline_H = _s1(pullback_cum_H) * reset_H_int
    # Forward-fill baseline through each group
    baseline_H_df = pd.DataFrame(baseline_H, index=idx, columns=cols)
    baseline_H_ff = baseline_H_df.replace(0.0, np.nan).ffill().fillna(0.0).values

    H_count = pullback_cum_H - baseline_H_ff
    # H_count at reset bars should be 0 (strong bull trend bars aren't pullbacks)
    H_count = np.where(reset_H, 0.0, H_count)
    H_count = np.maximum(H_count, 0.0)  # safety floor

    # H1-H4 signal scores
    def _h_signal(hc, n, base_w):
        """Score for Hn signal: count == n at a pullback bar."""
        is_nth = (hc == n) & (is_pullback_H > 0) & bull_trend_active
        return np.where(is_nth, base_w, 0.0)

    H1_score = _h_signal(H_count, 1, 0.25)
    H2_score = _h_signal(H_count, 2, 0.20)
    H3_score = _h_signal(H_count, 3, 0.12)
    H4_score = _h_signal(H_count, 4, 0.06)

    # ══ Bearish L-counting (symmetric, active when always_in < -0.15) ══
    bear_trend_active = always_in < -0.15

    reset_L = (trend_bear > 0.70) & bear_trend_active
    reset_L_int = reset_L.astype(float)

    is_rally_L = (
        is_bull_bar & bear_trend_active &
        (h_arr > h_s1) & (l_arr > l_s1)
    ).astype(float)

    rally_cum_L = np.cumsum(is_rally_L, axis=0)
    baseline_L = _s1(rally_cum_L) * reset_L_int
    baseline_L_df = pd.DataFrame(baseline_L, index=idx, columns=cols)
    baseline_L_ff = baseline_L_df.replace(0.0, np.nan).ffill().fillna(0.0).values

    L_count = rally_cum_L - baseline_L_ff
    L_count = np.where(reset_L, 0.0, L_count)
    L_count = np.maximum(L_count, 0.0)

    def _l_signal(lc, n, base_w):
        is_nth = (lc == n) & (is_rally_L > 0) & bear_trend_active
        return np.where(is_nth, base_w, 0.0)

    L1_score = _l_signal(L_count, 1, 0.25)
    L2_score = _l_signal(L_count, 2, 0.20)
    L3_score = _l_signal(L_count, 3, 0.12)
    L4_score = _l_signal(L_count, 4, 0.06)

    # ════════════════════════════════════════════════════════════════
    # T5 — Signal Bar + Entry Bar Confirmation
    # ════════════════════════════════════════════════════════════════

    # ---- Bull Signal Bar (doji with long lower wick at potential support) ----
    is_signal_bull = (
        (body_pct < 0.40) & (lo_w_pct >= 0.25) & (c_arr > l_arr + spread_arr * 0.30)
    )
    # Signal bar high (for entry trigger)
    sig_high_bull = np.where(is_signal_bull, h_arr, np.nan)
    sig_high_df = pd.DataFrame(sig_high_bull, index=idx, columns=cols)
    sig_high_ff = sig_high_df.ffill().values   # most recent signal high
    sig_high_s1 = _s1(sig_high_ff)              # yesterday's active signal high

    # Signal active in last 3 bars
    sig_active_3d = pd.DataFrame(is_signal_bull.astype(float), index=idx, columns=cols) \
                      .rolling(3, min_periods=1).max().shift(1).values > 0

    # Bull entry: close breaks above signal high, with bullish confirmation
    entry_bull = (
        sig_active_3d &
        (c_arr > sig_high_s1) &
        is_bull_bar &
        (body_pct >= 0.35)
    )
    entry_bull_score = np.where(
        entry_bull,
        np.where(c_arr > sig_high_s1, 0.30, 0.0) +
        np.where(is_bull_bar, 0.20, 0.0) +
        np.clip(body_pct, 0.0, 0.15) +
        np.where(c_arr > sig_high_s1 * 1.005, 0.15, 0.0) +
        np.where(v_arr > vol_ma20 * 1.1, 0.20, 0.0),
        0.0
    )
    entry_bull_score = np.clip(entry_bull_score, 0.0, 1.0)

    # ---- Bear Signal Bar (doji with long upper wick) ----
    is_signal_bear = (
        (body_pct < 0.40) & (up_w_pct >= 0.25) & (c_arr < h_arr - spread_arr * 0.30)
    )
    sig_low_bear = np.where(is_signal_bear, l_arr, np.nan)
    sig_low_df = pd.DataFrame(sig_low_bear, index=idx, columns=cols)
    sig_low_ff = sig_low_df.ffill().values
    sig_low_s1 = _s1(sig_low_ff)

    sig_active_3d_bear = pd.DataFrame(is_signal_bear.astype(float), index=idx, columns=cols) \
                           .rolling(3, min_periods=1).max().shift(1).values > 0

    entry_bear = (
        sig_active_3d_bear &
        (c_arr < sig_low_s1) &
        is_bear_bar &
        (body_pct >= 0.35)
    )
    entry_bear_score = np.where(
        entry_bear,
        np.where(c_arr < sig_low_s1, 0.30, 0.0) +
        np.where(is_bear_bar, 0.20, 0.0) +
        np.clip(body_pct, 0.0, 0.15) +
        np.where(c_arr < sig_low_s1 * 0.995, 0.15, 0.0) +
        np.where(v_arr > vol_ma20 * 1.1, 0.20, 0.0),
        0.0
    )
    entry_bear_score = np.clip(entry_bear_score, 0.0, 1.0)

    # ════════════════════════════════════════════════════════════════
    # T8 — Bull/Bear Trap (subsumes T3 Failed Breakout)
    #   Stand at T (confirmation). Look back 1-3 bars for breakout.
    # ════════════════════════════════════════════════════════════════

    # ---- High-volume breakout detection ----
    breakout_up = (c_arr > high_20) & (v_arr > vol_ma20 * 1.3)
    breakout_dn = (c_arr < low_20)  & (v_arr > vol_ma20 * 1.3)

    # Recent breakout (T-1, T-2, or T-3 occurred)
    br_up_s1 = _s1(breakout_up); br_up_s2 = _s1(br_up_s1); br_up_s3 = _s1(br_up_s2)
    breakout_up_recent = br_up_s1 | br_up_s2 | br_up_s3

    br_dn_s1 = _s1(breakout_dn); br_dn_s2 = _s1(br_dn_s1); br_dn_s3 = _s1(br_dn_s2)
    breakout_dn_recent = br_dn_s1 | br_dn_s2 | br_dn_s3

    # ══ Bull Trap (诱多 → 看跌信号) ══
    # T-day confirmation: bearish engulfing + high volume + price back below breakout level
    bull_trap_confirm = (
        breakout_up_recent &
        is_bear_bar &
        (body_pct >= 0.45) &
        (v_arr > vol_ma20 * 1.2) &
        (c_arr < high_20 * 0.99)               # price fell back below 20d high
    )
    # Add: trap bar's low breaks below the breakout bar's low
    # Approx: today's low < yesterday's or day-before's low when those were breakouts
    trap_low_break = (l_arr < l_s1) | (l_arr < _s1(l_s1))
    bull_trap_confirm = bull_trap_confirm & trap_low_break

    bull_trap_score = np.where(
        bull_trap_confirm,
        np.where(breakout_up_recent, 0.20, 0.0) +       # breakout happened
        np.where(is_bear_bar & (body_pct >= 0.45), 0.40, 0.0) +  # strong reversal bar
        np.where(v_arr > vol_ma20 * 1.2, 0.25, 0.0) +          # high vol
        np.where(c_arr < high_20 * 0.99, 0.15, 0.0),            # price back under
        0.0
    )
    bull_trap_score = np.clip(bull_trap_score, 0.0, 1.0)

    # ══ Bear Trap (诱空 → 看涨信号) ══
    bear_trap_confirm = (
        breakout_dn_recent &
        is_bull_bar &
        (body_pct >= 0.45) &
        (v_arr > vol_ma20 * 1.2) &
        (c_arr > low_20 * 1.01)               # price back above 20d low
    )
    trap_high_break = (h_arr > h_s1) | (h_arr > _s1(h_s1))
    bear_trap_confirm = bear_trap_confirm & trap_high_break

    bear_trap_score = np.where(
        bear_trap_confirm,
        np.where(breakout_dn_recent, 0.20, 0.0) +
        np.where(is_bull_bar & (body_pct >= 0.45), 0.40, 0.0) +
        np.where(v_arr > vol_ma20 * 1.2, 0.25, 0.0) +
        np.where(c_arr > low_20 * 1.01, 0.15, 0.0),
        0.0
    )
    bear_trap_score = np.clip(bear_trap_score, 0.0, 1.0)

    # ==================================================================
    # Package results
    # ==================================================================

    results = {}

    # T2: H1-H4
    results['brooks_H1_score'] = pd.DataFrame(H1_score, index=idx, columns=cols)
    results['brooks_H2_score'] = pd.DataFrame(H2_score, index=idx, columns=cols)
    results['brooks_H3_score'] = pd.DataFrame(H3_score, index=idx, columns=cols)
    results['brooks_H4_score'] = pd.DataFrame(H4_score, index=idx, columns=cols)

    # T2: L1-L4
    results['brooks_L1_score'] = pd.DataFrame(L1_score, index=idx, columns=cols)
    results['brooks_L2_score'] = pd.DataFrame(L2_score, index=idx, columns=cols)
    results['brooks_L3_score'] = pd.DataFrame(L3_score, index=idx, columns=cols)
    results['brooks_L4_score'] = pd.DataFrame(L4_score, index=idx, columns=cols)

    # T5
    results['brooks_entry_bull_score'] = pd.DataFrame(entry_bull_score, index=idx, columns=cols)
    results['brooks_entry_bear_score'] = pd.DataFrame(entry_bear_score, index=idx, columns=cols)

    # T8
    results['brooks_bull_trap_score'] = pd.DataFrame(bull_trap_score, index=idx, columns=cols)
    results['brooks_bear_trap_score'] = pd.DataFrame(bear_trap_score, index=idx, columns=cols)

    return results


# ======================================================================
# School-compatible split-return interface (6-tuple)
# ======================================================================

def compute_brooks_micro_score_split(indicators: dict):
    """
    Polarity-grouped split-return for _compute_school_brooks_pa integration.

    Returns
    -------
    (score_bull, score_bear, reasons_bull, reasons_bear,
     bull_trap_active, bear_trap_active)
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

        # Optional Part 1 inputs (may be in indicators dict directly)
        tb_df = indicators.get('brooks_trend_bull_df')
        ts_df = indicators.get('brooks_trend_bear_df')
        ai_df = indicators.get('brooks_always_in_df')

        feats = compute_brooks_micro_features(
            o_df, h_df, l_df, c_df, v_df, tb_df, ts_df, ai_df)

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

        # T2: H1-H4 pullback entries (bullish)
        for n, w, label in [('H1', 0.25, 'H1回调买点'), ('H2', 0.20, 'H2回调买点'),
                             ('H3', 0.12, 'H3回调买点'), ('H4', 0.06, 'H4回调买点')]:
            s = _read(f'brooks_{n}_score')
            if s > 0:
                score_bull += s * w
                reasons_bull.append(f'{label}(score={s:.2f})')

        # T2: L1-L4 rally entries (bearish)
        for n, w, label in [('L1', 0.25, 'L1反弹卖点'), ('L2', 0.20, 'L2反弹卖点'),
                             ('L3', 0.12, 'L3反弹卖点'), ('L4', 0.06, 'L4反弹卖点')]:
            s = _read(f'brooks_{n}_score')
            if s > 0:
                score_bear += s * w
                reasons_bear.append(f'{label}(score={s:.2f})')

        # T5: Entry confirmation
        eb = _read('brooks_entry_bull_score')
        es = _read('brooks_entry_bear_score')
        if eb > 0:
            score_bull += eb * 0.20
            reasons_bull.append(f'入场确认看涨(score={eb:.2f})')
        if es > 0:
            score_bear += es * 0.20
            reasons_bear.append(f'入场确认看跌(score={es:.2f})')

        # T8: Traps (注意极性：bull_trap → bearish, bear_trap → bullish)
        bt = _read('brooks_bull_trap_score')   # 诱多 → 看跌
        brt = _read('brooks_bear_trap_score')  # 诱空 → 看涨
        if bt > 0:
            score_bear += bt * 0.30
            reasons_bear.append(f'Bull Trap诱多陷阱(score={bt:.2f})→看跌')
        if brt > 0:
            score_bull += brt * 0.30
            reasons_bull.append(f'Bear Trap诱空陷阱(score={brt:.2f})→看涨')

        return (score_bull, score_bear, reasons_bull, reasons_bear,
                float(bt > 0), float(brt > 0))
    except Exception:
        return 0.0, 0.0, [], [], 0.0, 0.0


# ======================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    from brooks_macro_features import compute_brooks_macro_features
    print("=== Brooks PA Micro Features Test ===")
    for code in ['000651', '300750']:
        df = get_daily_kline(code, days=300)
        if df is None or df.empty:
            continue
        idx_d = df.index
        def _d(cn):
            return pd.DataFrame({'S': df[cn].values}, index=idx_d)

        # Get Part 1 outputs
        macro = compute_brooks_macro_features(_d('open'), _d('high'), _d('low'), _d('close'), _d('volume'))

        feats = compute_brooks_micro_features(
            _d('open'), _d('high'), _d('low'), _d('close'), _d('volume'),
            trend_bull_df=macro['brooks_trend_bull_score'],
            trend_bear_df=macro['brooks_trend_bear_score'],
            always_in_df=macro['brooks_always_in_score'])

        print(f"\n{code}:")
        for k in sorted(feats.keys()):
            nz = int((feats[k].values > 0).sum())
            if nz > 0:
                mx = feats[k].values.max()
                print(f"  {k}: nonzero={nz}/{len(df)} max={mx:.3f}")
