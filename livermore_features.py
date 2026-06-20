#!/usr/bin/env python
# livermore_features.py — Livermore 6 missing morphologies (vectorized wide-format)
"""
Livermore advanced features:
  1. 最小阻力线 (Line of Least Resistance)
  2. 单日反转   (One-Day Reversal)
  3. 金字塔加仓 (Pyramiding Probe)
  4. 成交量枯竭 (Volume Drying Up)
  5. 突破失败   (Failed Breakout / Shakeout)
  6. 整数关口   (Round-Number Pivot)

All inputs: wide-format DataFrames. Zero for-loops.
"""

import numpy as np
import pandas as pd

EPS = 1e-8


def generate_livermore_features(open_df, high_df, low_df, close_df, volume_df):
    """
    Compute 6 advanced Livermore features.

    Returns: dict[str, DataFrame]
    """
    o, h, l, c, v = open_df, high_df, low_df, close_df, volume_df
    idx, cols = c.index, c.columns

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
    is_bull = c > o
    is_bear = c < o

    # Anti-leak rolling highs/lows
    high_20 = h.shift(1).rolling(20, min_periods=10).max()
    low_20  = l.shift(1).rolling(20, min_periods=10).min()
    ret_20  = c / c.shift(20).replace(0, np.nan) - 1.0

    results = {}

    # ================================================================
    # 1. 最小阻力线 (Line of Least Resistance)
    # ================================================================
    # Bull: 10 consecutive days close > close 5 days ago + recent new high
    bull_a = (c > c.shift(5)).rolling(10, min_periods=10).sum() == 10
    bull_b = (c > high_20).rolling(3, min_periods=1).sum() >= 1
    bull_c = vol_ma5 > vol_ma20 * 1.1
    bull_mask = bull_a & bull_b & bull_c

    # Bear: 10 consecutive days close < close 5 days ago + recent new low
    bear_a = (c < c.shift(5)).rolling(10, min_periods=10).sum() == 10
    bear_b = (c < low_20).rolling(3, min_periods=1).sum() >= 1
    bear_c = v > vol_ma20  # volume confirms direction
    bear_mask = bear_a & bear_b & bear_c

    resistance = np.zeros_like(c.values, dtype=float)
    resistance = np.where(bull_mask.values, 1.0, resistance)
    resistance = np.where(bear_mask.values & ~bull_mask.values, -1.0, resistance)
    results['line_of_least_resistance'] = pd.DataFrame(resistance, index=idx, columns=cols)

    # ================================================================
    # 2. 单日反转 (One-Day Reversal)
    # ================================================================
    yest_close = c.shift(1)
    vol_ma10_val = vol_ma10

    # Bearish reversal (top): intraday surge >3%, close below yesterday, heavy vol
    top_a = h > yest_close * 1.03
    top_b = c < yest_close
    top_c = v > vol_ma10_val * 1.50
    top_d = c < (h + l) / 2.0
    top_e = ret_20 > 0.10
    top_mask = top_a & top_b & top_c & top_d & top_e

    # Bullish reversal (bottom): intraday plunge >3%, close above yesterday, heavy vol
    bot_a = l < yest_close * 0.97
    bot_b = c > yest_close
    bot_c = v > vol_ma10_val * 1.50
    bot_d = c > (h + l) / 2.0
    bot_e = ret_20 < -0.10
    bot_mask = bot_a & bot_b & bot_c & bot_d & bot_e

    reversal = np.zeros_like(c.values, dtype=float)
    reversal = np.where(bot_mask.values, 1.0, reversal)
    reversal = np.where(top_mask.values & ~bot_mask.values, -1.0, reversal)
    results['one_day_reversal'] = pd.DataFrame(reversal, index=idx, columns=cols)

    # ================================================================
    # 3. 金字塔加仓 (Pyramiding Probe)
    # ================================================================
    # A: First breakout within last 8 days
    breakout_a = (c > high_20) & (v > vol_ma20 * 1.5)
    # Record breakout close price, forward-fill for up to 8 days
    breakout_price = c.copy()
    breakout_price.values[:] = np.where(breakout_a.values, c.values, np.nan)
    # Forward-fill breakout close (limit=8 bars)
    last_breakout_close = breakout_price.ffill(limit=8)

    # B: Pullback holds above 97% of breakout close
    probe_b = l > last_breakout_close * 0.97

    # C: Today volume > 1.3x 10-day avg, bullish candle
    probe_c = (v > vol_ma10 * 1.30) & is_bull

    # D: Close above breakout close (trend resumed)
    probe_d = c > last_breakout_close

    # All conditions + breakout was recent (< 8 days ago reachable by ffill limit)
    probe_mask = last_breakout_close.notna() & probe_b & probe_c & probe_d
    # Strength: how tight the pullback + volume surge
    probe_strength = (l / last_breakout_close.replace(0, np.nan) - 0.97) / 0.03  # 0 at 97%, 1 at 100%
    probe_strength = np.clip(probe_strength.values, 0, 1)
    probe_out = np.where(probe_mask.values, probe_strength, 0.0)
    results['pyramiding_probe'] = pd.DataFrame(probe_out, index=idx, columns=cols)

    # ================================================================
    # 4. 成交量枯竭 (Volume Drying Up)
    # ================================================================
    v_vals = v.values
    vol_decline = (v_vals < v.shift(1).values) & (v.shift(1).values < v.shift(2).values)
    vol_low = v < vol_ma20 * 0.60
    narrow_range = (chl / c.replace(0, np.nan)) < 0.02
    narrow_recent = narrow_range.rolling(5, min_periods=5).sum() >= 4

    ma_vals = [ma5.values, ma10.values, ma20.values, ma60.values]
    ma_converge = np.fmax.reduce(ma_vals) / (np.fmin.reduce(ma_vals) + EPS) - 1.0
    converge_ok = (ma_converge <= 0.04) & (np.abs(ma5.values - ma20.values) / (ma20.values + EPS) <= 0.03)

    near_pivot = (c / high_20.replace(0, np.nan) > 0.97) | (c / low_20.replace(0, np.nan) < 1.03)

    dry_mask = vol_decline & vol_low.values & narrow_recent.values & converge_ok & near_pivot.values
    dry_strength = (1.0 - v_vals / vol_ma20.values) * (1.0 - chl.values / c.values / 0.05)
    dry_out = np.where(dry_mask, np.clip(dry_strength, 0, 1), 0.0)
    results['volume_drying_up'] = pd.DataFrame(dry_out, index=idx, columns=cols)

    # ================================================================
    # 5. 突破失败 (Failed Breakout / Shakeout)
    # ================================================================
    # False breakout up (bull trap): yesterday broke 20d high, today collapses
    fake_up_a = c.shift(1) > high_20.shift(1)
    fake_up_b = c < l.shift(1)
    fake_up_c = v > vol_ma20 * 1.80

    # False breakdown (bear trap): yesterday broke 20d low, today surges
    fake_down_a = c.shift(1) < low_20.shift(1)
    fake_down_b = c > h.shift(1)
    fake_down_c = v > vol_ma20 * 1.80

    fake_out = np.zeros_like(c.values, dtype=float)
    fake_out = np.where(fake_down_a.values & fake_down_b.values & fake_down_c.values, 1.0, fake_out)
    fake_out = np.where(fake_up_a.values & fake_up_b.values & fake_up_c.values & (fake_out == 0), -1.0, fake_out)
    results['failed_breakout'] = pd.DataFrame(fake_out, index=idx, columns=cols)

    # ================================================================
    # 6. 整数关口 (Round-Number Pivot)
    # ================================================================
    round_price = np.round(c.values)
    dist_pct = np.abs(c.values - round_price) / (round_price + EPS)

    # Multiple probes near round number in last 5 days
    near_round = dist_pct <= 0.005
    probe_count = pd.DataFrame(near_round, index=idx, columns=cols).rolling(5, min_periods=1).sum()

    # Breakout above/below round number
    break_up = (c.values > round_price * 1.02) & (probe_count.values >= 3) & (v.values > vol_ma10.values * 1.50)
    break_down = (c.values < round_price * 0.98) & (probe_count.values >= 3) & (v.values > vol_ma10.values * 1.50)

    round_out = np.zeros_like(c.values, dtype=float)
    round_out = np.where(break_up, 1.0, round_out)
    round_out = np.where(break_down & ~break_up, -1.0, round_out)
    results['round_number_pivot'] = pd.DataFrame(round_out, index=idx, columns=cols)

    return results


# ==========================================================================
# School-compatible interface
# ==========================================================================

class LivermoreAdvancedSchool:
    """利弗莫尔高级形态综合评分"""

    def compute_signal(self, df):
        if df is None or len(df) < 70:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}
        idx = df.index
        def _d(col): return pd.DataFrame({'S': df[col].values}, index=idx)
        feats = generate_livermore_features(_d('open'), _d('high'), _d('low'),
                                             _d('close'), _d('volume'))

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        for key, w, label in [
            ('line_of_least_resistance', 0.30, '最小阻力线(升)'),
            ('pyramiding_probe',         0.25, '金字塔加仓'),
            ('volume_drying_up',         0.20, '成交量枯竭'),
        ]:
            val = feats[key]['S'].iloc[-1]
            if val > 0: score_bull += w * (val if key == 'pyramiding_probe' else 1.0); reasons_bull.append(label)

        for key, w, label in [
            ('line_of_least_resistance', 0.30, '最小阻力线(降)'),
        ]:
            val = feats[key]['S'].iloc[-1]
            if val < 0: score_bear += w; reasons_bear.append(label)

        # Bidirectional
        for key, w_bull, w_bear, label_bull, label_bear in [
            ('one_day_reversal',   0.30, 0.30, '单日反转(底)', '单日反转(顶)'),
            ('failed_breakout',    0.30, 0.30, '假突破(震仓)', '假突破(诱多)'),
            ('round_number_pivot', 0.20, 0.20, '整数关口(突破)', '整数关口(跌破)'),
        ]:
            val = feats[key]['S'].iloc[-1]
            if val > 0: score_bull += w_bull; reasons_bull.append(label_bull)
            elif val < 0: score_bear += w_bear; reasons_bear.append(label_bear)

        # Nonlinear squash
        score_bull = 1.0 - np.exp(-score_bull * 1.5)
        score_bear = 1.0 - np.exp(-score_bear * 1.5)

        if score_bull > 0.40 and score_bear > 0.40:
            final_score = 0.0; confidence = 0.10; direction = 'neutral'
            reasons_final = ['多空矛盾→观望']
        else:
            final_score = np.clip(score_bull - score_bear, -1.0, 1.0)
            confidence = min(abs(final_score) * 1.25, 0.90)
            direction = 'bullish' if final_score > 0.06 else ('bearish' if final_score < -0.06 else 'neutral')
            reasons_final = (reasons_bull if final_score > 0 else (reasons_bear if final_score < 0 else reasons_bull + reasons_bear))[:5]

        return {'direction': direction, 'score': round(float(final_score), 3),
                'confidence': round(float(confidence), 3), 'reasons': reasons_final}


_liv_school = LivermoreAdvancedSchool()

def compute_livermore_advanced_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 70: return None
    try:
        r = _liv_school.compute_signal(df_daily)
        return {'signal': r['direction'], 'confidence': r['confidence'],
                'metadata': {'score': r['score'], 'reasons': r['reasons']}}
    except: return None


def compute_livermore_signal_score_split(indicators: dict):
    """
    Split-return for polarity-grouped scoring in _compute_school_livermore.
    Returns (score_bull, score_bear, reasons_bull, reasons_bear).
    """
    try:
        import pandas as pd
        cp = indicators.get('current_price', 0) or 0
        o_v = indicators.get('open', cp) or cp
        h_v = indicators.get('high', cp) or cp
        l_v = indicators.get('low', cp) or cp
        v_v = indicators.get('volume', 0) or 0
        idx = pd.DatetimeIndex([pd.Timestamp.now()])
        o_df = pd.DataFrame({'S': [float(o_v)]}, index=idx)
        h_df = pd.DataFrame({'S': [float(h_v)]}, index=idx)
        l_df = pd.DataFrame({'S': [float(l_v)]}, index=idx)
        c_df = pd.DataFrame({'S': [float(cp)]}, index=idx)
        v_df = pd.DataFrame({'S': [float(v_v)]}, index=idx)
        feats = generate_livermore_features(o_df, h_df, l_df, c_df, v_df)

        score_bull = 0.0; reasons_bull = []
        score_bear = 0.0; reasons_bear = []

        for key, w, label in [
            ('line_of_least_resistance', 0.30, '最小阻力线(升)'),
            ('pyramiding_probe',         0.25, '金字塔加仓'),
            ('volume_drying_up',         0.20, '成交量枯竭'),
        ]:
            val = feats[key]['S'].iloc[0]
            if val > 0: score_bull += w; reasons_bull.append(label)

        for key, w, label in [
            ('line_of_least_resistance', 0.30, '最小阻力线(降)'),
        ]:
            val = feats[key]['S'].iloc[0]
            if val < 0: score_bear += w; reasons_bear.append(label)

        for key, w_bull, w_bear, lb, ls in [
            ('one_day_reversal',   0.30, 0.30, '单日反转(底)', '单日反转(顶)'),
            ('failed_breakout',    0.30, 0.30, '假突破(震仓)', '假突破(诱多)'),
            ('round_number_pivot', 0.20, 0.20, '整数关口(突破)', '整数关口(跌破)'),
        ]:
            val = feats[key]['S'].iloc[0]
            if val > 0:   score_bull += w_bull; reasons_bull.append(lb)
            elif val < 0: score_bear += w_bear; reasons_bear.append(ls)

        return score_bull, score_bear, reasons_bull, reasons_bear
    except Exception:
        return 0.0, 0.0, [], []


# ==========================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Livermore Features Test ===")
    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=300)
        if df is not None and not df.empty:
            s = _liv_school.compute_signal(df)
            print(f"{code}: {s['direction']} conf={s['confidence']:.3f} reasons={s['reasons'][:4]}")
