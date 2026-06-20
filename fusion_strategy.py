#!/usr/bin/env python
# fusion_strategy.py
"""
融合蒸馏策略: 4+5号文件合并

来源A (文件4): ADX趋势判别 + FRVP POC + 缠论MACD背驰 + RVOL量能
来源B (文件5): FRVP滚动50-bin POC/VAH/VAL + 缠论分型 + Delta订单流 + ROC动量

合并后核心:
  Regime识别: ADX+VWAP → Uptrend/Ranging/Downtrend
  缠论结构: 底分型/顶分型 + MACD背驰
  订单流: 5日累计Delta净买卖
  FRVP: 滚动POC+VAH+VAL区间
  多条件共振: ≥4/6 → 信号
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any


# =============================================================================
# FRVP Rolling
# =============================================================================

def compute_frvp_rolling(df: pd.DataFrame, window: int = 60, num_bins: int = 40) -> pd.DataFrame:
    """滚动计算 FRVP: POC, VAH, VAL"""
    df = df.copy()
    df['frvp_poc'] = np.nan
    df['frvp_vah'] = np.nan
    df['frvp_val'] = np.nan

    for idx in range(window, len(df)):
        sub = df.iloc[idx - window:idx]
        price_min, price_max = sub['low'].min(), sub['high'].max()
        if price_max <= price_min:
            continue

        bins = np.linspace(price_min, price_max, num_bins)
        bin_vol = np.zeros(len(bins) - 1)

        for _, row in sub.iterrows():
            matched = (bins[:-1] >= row['low']) & (bins[:-1] < row['high'])
            if matched.sum() > 0:
                bin_vol[matched] += row['volume'] / matched.sum()

        if bin_vol.sum() == 0:
            continue

        poc_idx = np.argmax(bin_vol)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2

        total_v = bin_vol.sum()
        target_v = total_v * 0.70
        cum_v = bin_vol[poc_idx]
        l_idx = r_idx = poc_idx

        while cum_v < target_v and (l_idx > 0 or r_idx < len(bin_vol) - 1):
            vl = bin_vol[l_idx - 1] if l_idx > 0 else 0
            vr = bin_vol[r_idx + 1] if r_idx < len(bin_vol) - 1 else 0
            if vl >= vr:
                cum_v += vl; l_idx -= 1
            else:
                cum_v += vr; r_idx += 1

        df.loc[df.index[idx], 'frvp_poc'] = poc
        df.loc[df.index[idx], 'frvp_vah'] = bins[min(r_idx, len(bins)-2)]
        df.loc[df.index[idx], 'frvp_val'] = bins[max(l_idx, 0)]

    return df


# =============================================================================
# Indicators Bundle
# =============================================================================

def compute_fusion_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """All indicators needed for the fusion strategy."""
    df = df.copy()
    h, l, c, v = df['high'], df['low'], df['close'], df['volume']

    # VWAP
    typical = (h + l + c) / 3
    df['vwap'] = (v * typical).cumsum() / v.cumsum()

    # MACD
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    df['macd'] = e12 - e26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ATR
    tr1 = h - l
    tr2 = abs(h - c.shift(1))
    tr3 = abs(l - c.shift(1))
    df['atr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
    df['atr_pct'] = df['atr'] / c

    # ADX
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = df['atr']
    df['pdi'] = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9))
    df['mdi'] = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-9))
    dx = 100 * abs(df['pdi'] - df['mdi']) / (df['pdi'] + df['mdi'] + 1e-9)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    # Bollinger
    df['bb_mid'] = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std

    # RVOL
    df['rvol'] = v / (v.rolling(20).mean() + 1e-9)

    # ROC
    df['roc'] = c.pct_change(10) * 100

    # Volume Delta (order flow proxy)
    candle_range = h - l + 1e-9
    buying = (c - l) / candle_range
    selling = (h - c) / candle_range
    df['vol_delta'] = v * (buying - selling)
    df['cum_delta_5'] = df['vol_delta'].rolling(5).sum()

    # OBV
    df['obv'] = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    # Chanlun fractal (pivot detection)
    df['pivot_type'] = 0
    for i in range(2, len(df) - 2):
        if (h.iloc[i] > h.iloc[i-1] and h.iloc[i] > h.iloc[i-2] and
            h.iloc[i] > h.iloc[i+1] and h.iloc[i] > h.iloc[i+2]):
            df.loc[df.index[i], 'pivot_type'] = 1  # top fractal
        elif (l.iloc[i] < l.iloc[i-1] and l.iloc[i] < l.iloc[i-2] and
              l.iloc[i] < l.iloc[i+1] and l.iloc[i] < l.iloc[i+2]):
            df.loc[df.index[i], 'pivot_type'] = -1  # bottom fractal

    # Chanlun MACD divergence
    price_low_20 = l.rolling(20).min()
    macdh_low_20 = df['macd_hist'].rolling(20).min()
    df['chan_divergence'] = (
        (l <= price_low_20 * 1.02) &
        (df['macd_hist'] > macdh_low_20) &
        (df['macd_hist'] > df['macd_hist'].shift(1))
    )

    # FRVP
    df = compute_frvp_rolling(df, window=60)

    # Regime
    df['regime'] = 'unknown'
    df.loc[(df['adx'] < 22), 'regime'] = 'ranging'
    df.loc[(df['adx'] >= 22) & (df['pdi'] > df['mdi']) & (c > df['vwap']), 'regime'] = 'uptrend'
    df.loc[(df['adx'] >= 22) & (df['mdi'] > df['pdi']) & (c < df['vwap']), 'regime'] = 'downtrend'

    return df


# =============================================================================
# Signal Generation
# =============================================================================

def generate_fusion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals from fusion strategy."""
    df = compute_fusion_indicators(df)

    signal = pd.Series(0, index=df.index)
    confidence = pd.Series(0.0, index=df.index)

    for i in range(60, len(df)):
        row = df.iloc[i]
        regime = row['regime']
        score = 0
        total = 6

        # Score all 6 conditions regardless of regime
        # Regime penalty applied at end — downtrend needs stronger signal

        # 1. Trend: price > VWAP or strong ADX uptrend
        if row['close'] > row['vwap'] or (regime == 'uptrend' and row['pdi'] > row['mdi'] * 1.2):
            score += 1

        # 2. Chanlun: bottom fractal or MACD divergence
        recent_pivots = df['pivot_type'].iloc[max(0,i-7):i].values
        has_bottom = -1 in recent_pivots
        if has_bottom or row['chan_divergence']:
            score += 1

        # 3. Volume: RVOL > 1.3 or cum_delta positive
        if row['rvol'] > 1.3 or row['cum_delta_5'] > 0:
            score += 1

        # 4. Momentum: RSI 30-65 AND ROC > -2
        if 30 < row['rsi'] < 65 and row['roc'] > -2:
            score += 1

        # 5. FRVP: near POC or above VAL
        if not pd.isna(row['frvp_poc']):
            near_poc = row['frvp_poc'] * 0.97 <= row['close'] <= row['frvp_poc'] * 1.05
            above_val = row['close'] > row['frvp_val']
            if near_poc or above_val:
                score += 1

        # 6. MACD: histogram positive
        if row['macd_hist'] > 0:
            score += 1

        # Signal: >= 4/6 conditions met (regime injected as feature, not gate)
        if score >= 4:
            signal.iloc[i] = 1
            confidence.iloc[i] = score / total

    df['signal'] = signal.astype(int)
    df['confidence'] = confidence
    df['direction'] = signal.map({1: 'bullish', 0: 'neutral'})
    df['score'] = pd.Series(0, index=df.index)

    return df


# =============================================================================
# School-compatible interface
# =============================================================================

def compute_fusion_signal(df_daily, df_hourly=None):
    """School-compatible signal interface for expert_ensemble."""
    if df_daily is None or df_daily.empty or len(df_daily) < 80:
        return None
    try:
        result = generate_fusion_signals(df_daily)
        last = result.iloc[-1]
        regime = last.get('regime', 'unknown')
        score = int((result['signal'].iloc[-60:] == 1).sum()) / 60  # recent bullish ratio

        if last['signal'] == 0:
            return {'signal': 'neutral', 'confidence': 0.0,
                    'metadata': {'regime': regime, 'recent_bullish_pct': round(score, 2)}}

        return {'signal': 'bullish', 'confidence': float(last['confidence']),
                'metadata': {'regime': regime, 'rsi': float(last['rsi']) if not pd.isna(last['rsi']) else 50,
                            'adx': float(last['adx']) if not pd.isna(last['adx']) else 20,
                            'rvol': float(last['rvol']) if not pd.isna(last['rvol']) else 1.0}}
    except Exception:
        return None


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    df = get_daily_kline('000012', days=200)
    if df is not None and not df.empty:
        result = generate_fusion_signals(df)
        last = result.iloc[-1]
        buys = (result['signal'] == 1).sum()
        print(f"Latest: {result.index[-1]}, Signal: {last['signal']}, Regime: {last['regime']}")
        print(f"Confidence: {last['confidence']:.1%}")
        print(f"Bullish: {buys} | Neutral: {(result['signal']==0).sum()}")
