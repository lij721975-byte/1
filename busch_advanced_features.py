#!/usr/bin/env python
# busch_advanced_features.py — Busch 6 missing morphologies (vectorized wide-format)
"""
Busch advanced features:
  1. 波动率压缩    (Volatility Squeeze)
  2. MACD零轴穿越  (MACD Zero-Line Cross)
  3. 多周期共振    (Multi-TF Resonance)
  4. 量价背离增强  (Enhanced V-P Divergence)
  5. 趋势强度过滤  (Trend Strength Filter)
  6. 回调买入法    (Pullback Entry)

All inputs: wide-format DataFrames. Zero for-loops. EPS on all divisions.
"""

import numpy as np
import pandas as pd

EPS = 1e-8


def generate_busch_advanced_features(open_df, high_df, low_df, close_df, volume_df):
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

    # ================================================================
    # Shared indicators
    # ================================================================
    ma10  = c.rolling(10, min_periods=5).mean()
    ma20  = c.rolling(20, min_periods=10).mean()
    ma25  = c.rolling(25, min_periods=12).mean()
    ma60  = c.rolling(60, min_periods=30).mean()
    ma125 = c.rolling(125, min_periods=60).mean()
    ma300 = c.rolling(300, min_periods=150).mean()

    vol_ma5  = v.rolling(5,  min_periods=2).mean()
    vol_ma20 = v.rolling(20, min_periods=10).mean()

    # BB width
    bb_mid = ma20
    bb_std = c.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    bb_width_ma60 = bb_width.rolling(60, min_periods=30).mean()

    # ── DMI / ADX (14) ──
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = np.fmax.reduce([tr1.values, tr2.values, tr3.values])

    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Wilder smoothing proxy: EMA(14)
    atr14 = pd.DataFrame(tr, index=idx, columns=cols).ewm(span=14, adjust=False).mean()
    plus_di = 100.0 * pd.DataFrame(plus_dm, index=idx, columns=cols).ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100.0 * pd.DataFrame(minus_dm, index=idx, columns=cols).ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx14 = dx.ewm(span=14, adjust=False).mean()

    # ── MACD ──
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    macd_dif = e12 - e26
    macd_dea = macd_dif.ewm(span=9, adjust=False).mean()
    macd_hist = macd_dif - macd_dea

    results = {}

    # ================================================================
    # 1. 波动率压缩 (Volatility Squeeze)
    # ================================================================
    squeeze_a = bb_width < bb_width_ma60 * 0.20
    squeeze_b = adx14 < 18
    ma_bind = (ma10 / ma60.replace(0, np.nan) - 1.0).abs()
    squeeze_c = ma_bind < 0.03
    squeeze_d = v < vol_ma20 * 0.60

    # Direction: price > MA20 → bullish squeeze; price < MA20 → bearish squeeze
    squeeze_mask = squeeze_a & squeeze_b & squeeze_c & squeeze_d
    squeeze_strength = (1.0 - bb_width / bb_width_ma60.replace(0, np.nan)) * (1.0 - adx14 / 18.0)
    squeeze_strength = np.clip(squeeze_strength.values, 0, 1)

    squeeze_out = np.zeros_like(c.values, dtype=float)
    squeeze_out = np.where(squeeze_mask.values & (c.values > ma20.values), squeeze_strength, squeeze_out)
    squeeze_out = np.where(squeeze_mask.values & (c.values < ma20.values) & (squeeze_out == 0), -squeeze_strength, squeeze_out)
    results['volatility_squeeze'] = pd.DataFrame(squeeze_out, index=idx, columns=cols)

    # ================================================================
    # 2. MACD零轴穿越 (MACD Zero-Line Cross)
    # ================================================================
    macd_a = (macd_dif > 0) & (macd_dif.shift(1) < 0)
    macd_b = (macd_hist > 0) & (macd_hist > macd_hist.shift(1))
    macd_c = v > vol_ma20 * 1.20
    macd_d = c > ma60
    macd_mask = macd_a & macd_b & macd_c & macd_d
    macd_strength = np.clip(macd_dif / (macd_dea.abs() + EPS), 0, 3) / 3.0
    results['macd_zero_cross'] = pd.DataFrame(
        np.where(macd_mask.values, macd_strength.values, 0.0), index=idx, columns=cols)

    # ================================================================
    # 3. 多周期共振 (Multi-TF Resonance)
    # ================================================================
    # A: daily MA25 > MA60
    tf_a = ma25 > ma60
    # B: weekly proxy — MA125 cross above MA300
    tf_b = (ma125 > ma300) & (ma125.shift(1) <= ma300.shift(1))
    # C: ADX > 20 & PDI > MDI
    tf_c = (adx14 > 20) & (plus_di > minus_di)

    tf_score = np.zeros_like(c.values, dtype=float)
    tf_score = np.where(tf_a.values & tf_b.values & tf_c.values, 0.80, tf_score)
    tf_score = np.where(tf_a.values & tf_b.values & (tf_score == 0), 0.60, tf_score)
    tf_score = np.where(tf_a.values & tf_c.values & (tf_score == 0), 0.50, tf_score)
    results['multi_tf_resonance'] = pd.DataFrame(tf_score, index=idx, columns=cols)

    # ================================================================
    # 4. 量价背离增强 (Enhanced V-P Divergence)
    # ================================================================
    high_20_vals = h.shift(1).rolling(20, min_periods=10).max().values
    low_20_vals  = l.shift(1).rolling(20, min_periods=10).min().values

    # Top divergence: new high + volume declining
    top_new_high = c.values > high_20_vals
    top_vol_decline = (v.values < v.shift(5).values) & (v.shift(5).values < v.shift(10).values)
    top_low_vol = v.values < vol_ma20.values * 0.80
    top_mask = top_new_high & top_vol_decline & top_low_vol
    top_strength = np.abs(v.values / (vol_ma20.values + EPS) - 1.0)

    # Bottom divergence: new low + volume surge
    bot_new_low = c.values < low_20_vals
    bot_vol_surge = v.values > vol_ma20.values * 1.50
    bot_mask = bot_new_low & bot_vol_surge
    bot_strength = np.abs(v.values / (vol_ma20.values + EPS) - 1.0)

    div_out = np.zeros_like(c.values, dtype=float)
    div_out = np.where(bot_mask, bot_strength, div_out)
    div_out = np.where(top_mask & (div_out == 0), -top_strength, div_out)
    results['enhanced_divergence'] = pd.DataFrame(div_out, index=idx, columns=cols)

    # ================================================================
    # 5. 趋势强度过滤 (Trend Strength Filter)
    # ================================================================
    ts_a = adx14 > 25
    ts_b = adx14 > adx14.shift(5)
    ts_c = (plus_di - minus_di).abs() > 10
    ts_d = vol_ma5 > vol_ma20

    ts_strong = ts_a & ts_b & ts_c & ts_d
    ts_out = np.where(ts_strong.values, 1.0, 0.60)
    results['trend_strength_filter'] = pd.DataFrame(ts_out, index=idx, columns=cols)

    # ================================================================
    # 6. 回调买入法 (Pullback Entry)
    # ================================================================
    pb_a = (ma25 > ma60) & (c > ma60)
    pb_b = (c >= ma25 * 0.97) & (c <= ma25 * 1.03)
    pb_c = (v < vol_ma5 * 0.80) & (c > o)
    pb_d = (adx14 > 20) & (plus_di > minus_di)

    pb_mask = pb_a & pb_b & pb_c & pb_d
    close_dist = (1.0 - (c - ma25).abs() / ma25.replace(0, np.nan) / 0.03)
    vol_light = 1.0 - v / vol_ma5.replace(0, np.nan)
    pb_strength = np.clip(close_dist.values * vol_light.values, 0, 1)
    results['pullback_entry'] = pd.DataFrame(
        np.where(pb_mask.values, pb_strength, 0.0), index=idx, columns=cols)

    return results


# ==========================================================================
# School-compatible split-return interface
# ==========================================================================

def compute_busch_signal_score_split(indicators: dict):
    """
    Split-return for polarity-grouped scoring in _compute_school_busch.
    Returns (score_bull, score_bear, reasons_bull, reasons_bear).
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
        feats = generate_busch_advanced_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        # Squeeze (bidirectional)
        sq = feats['volatility_squeeze']['S'].iloc[0]
        if sq > 0:    score_bull += 0.25; reasons_bull.append('波动率压缩(看涨)')
        elif sq < 0:  score_bear += 0.25; reasons_bear.append('波动率压缩(看跌)')

        # MACD zero cross (bull only)
        mz = feats['macd_zero_cross']['S'].iloc[0]
        if mz > 0: score_bull += 0.25 * mz; reasons_bull.append('MACD零轴穿越')

        # Multi-TF resonance (bull only)
        tf = feats['multi_tf_resonance']['S'].iloc[0]
        if tf > 0: score_bull += tf * 0.30; reasons_bull.append(f'多周期共振({tf:.1f})')

        # Divergence (bidirectional)
        dv = feats['enhanced_divergence']['S'].iloc[0]
        if dv > 0:    score_bull += 0.20; reasons_bull.append('量价底背离')
        elif dv < 0:  score_bear += 0.20; reasons_bear.append('量价顶背离')

        # Trend strength filter is a global multiplier, not scored directly
        ts = feats['trend_strength_filter']['S'].iloc[0]

        # Pullback entry (bull only)
        pb = feats['pullback_entry']['S'].iloc[0]
        if pb > 0: score_bull += 0.25 * pb; reasons_bull.append('回调买入')

        return score_bull, score_bear, reasons_bull, reasons_bear, ts
    except Exception:
        return 0.0, 0.0, [], [], 1.0


# ==========================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Busch Advanced Features Test ===")
    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=300)
        if df is not None and not df.empty:
            idx = df.index
            def _d(c): return pd.DataFrame({'S': df[c].values}, index=idx)
            feats = generate_busch_advanced_features(_d('open'),_d('high'),_d('low'),_d('close'),_d('volume'))
            for k, v in feats.items():
                nz = (v.values != 0).sum()
                if nz > 0:
                    print(f"  {code} {k}: {nz} non-zero")
