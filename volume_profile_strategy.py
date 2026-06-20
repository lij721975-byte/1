#!/usr/bin/env python
# volume_profile_strategy.py
"""
Fixed Range Volume Profile (FRVP) Strategy.

Core concepts:
  - POC (Point of Control): price level with highest traded volume
  - VA  (Value Area): range containing ~70% of total volume
  - VAH (Value Area High) / VAL (Value Area Low)
  - HVN (High Volume Node): accumulation zone, acts as support/resistance
  - LVN (Low Volume Node): price rejection zone, price moves through quickly

Signals:
  - Bullish: price > VAH (breakout) or price bounces off POC from above
  - Bearish: price < VAL (breakdown) or price rejected at POC from below

Strategy rules from research:
  1. POC牵引效应: price tends to return to POC after breaking it
  2. VA通道突破: break above VAH = bullish, break below VAL = bearish
  3. Price in VA = ranging, Price outside VA = trending
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple


def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 40,
    va_pct: float = 0.70,
    lookback: int = 60,
) -> Dict[str, Any]:
    """
    Compute Fixed Range Volume Profile over the last `lookback` bars.

    Returns dict with:
      profile: dict mapping price to volume
      poc: Point of Control price
      vah: Value Area High
      val: Value Area Low
      total_volume: sum of all volume
    """
    window = df.iloc[-lookback:] if len(df) > lookback else df
    price_min = window['low'].min()
    price_max = window['high'].max()

    if price_max <= price_min:
        price_max = price_min + 0.01

    bin_size = (price_max - price_min) / num_bins

    # Distribute volume using weighted close
    volume_profile = {}
    for _, row in window.iterrows():
        weighted_close = (row['close'] * 2 + row['high'] + row['low']) / 4
        bin_idx = min(int((weighted_close - price_min) / bin_size), num_bins - 1)
        bin_price = round(price_min + bin_idx * bin_size, 2)
        volume_profile[bin_price] = volume_profile.get(bin_price, 0) + row['volume']

    if not volume_profile:
        mid = (price_min + price_max) / 2
        return {'poc': mid, 'vah': mid, 'val': mid, 'profile': {}}

    # POC = price with max volume
    poc = max(volume_profile, key=volume_profile.get)
    total_vol = sum(volume_profile.values())

    # Value Area: expand outward from POC, taking the side with more volume
    sorted_levels = sorted(volume_profile.items())  # (price, volume)
    poc_idx = next(i for i, (p, _) in enumerate(sorted_levels) if p == poc)

    va_vol = sorted_levels[poc_idx][1]
    lo_idx = hi_idx = poc_idx

    while va_vol < total_vol * va_pct:
        lo_vol = sorted_levels[lo_idx - 1][1] if lo_idx > 0 else 0
        hi_vol = sorted_levels[hi_idx + 1][1] if hi_idx < len(sorted_levels) - 1 else 0

        if lo_vol >= hi_vol and lo_idx > 0:
            lo_idx -= 1
            va_vol += sorted_levels[lo_idx][1]
        elif hi_idx < len(sorted_levels) - 1:
            hi_idx += 1
            va_vol += sorted_levels[hi_idx][1]
        elif lo_idx > 0:
            lo_idx -= 1
            va_vol += sorted_levels[lo_idx][1]
        else:
            break

    vah = sorted_levels[hi_idx][0]
    val = sorted_levels[lo_idx][0]

    # Detect HVN/LVN
    avg_vol_per_bin = total_vol / len(volume_profile) if volume_profile else 1
    hvns = [p for p, v in volume_profile.items() if v > avg_vol_per_bin * 1.5]
    lvns = [p for p, v in volume_profile.items() if v < avg_vol_per_bin * 0.3]

    return {
        'poc': round(poc, 2),
        'vah': round(vah, 2),
        'val': round(val, 2),
        'profile': volume_profile,
        'total_volume': total_vol,
        'hvns': hvns,
        'lvns': lvns,
    }


def generate_volume_profile_signals(
    df: pd.DataFrame,
    lookback: int = 60,
    num_bins: int = 40,
    va_pct: float = 0.70,
    rvol_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Generate Volume Profile trading signals.

    Scoring system (0-5):
      1. Price vs VA position (above VAH +, below VAL -)
      2. Price vs POC (above +, below -)
      3. Volume confirmation (rvol > threshold)
      4. Trend momentum (price vs MA20)
      5. POC test (price near POC, potential bounce/rejection)

    Signal >= 3/5 with dominant direction.
    """
    df = df.copy()

    # Compute rolling volume profile
    df['vp_poc'] = np.nan
    df['vp_vah'] = np.nan
    df['vp_val'] = np.nan
    df['vp_va_width'] = np.nan

    for i in range(lookback, len(df)):
        window = df.iloc[max(0, i - lookback):i + 1]
        vp = compute_volume_profile(window, num_bins=num_bins, va_pct=va_pct,
                                    lookback=lookback)
        df.at[df.index[i], 'vp_poc'] = vp['poc']
        df.at[df.index[i], 'vp_vah'] = vp['vah']
        df.at[df.index[i], 'vp_val'] = vp['val']
        df.at[df.index[i], 'vp_va_width'] = (vp['vah'] - vp['val']) / vp['val'] * 100 if vp['val'] > 0 else 0

    # Basic indicators
    df['ma20'] = df['close'].rolling(20).mean()
    df['rvol'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    delta = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.clip(lower=0).rolling(7).mean() /
                                   (-delta.clip(upper=0)).rolling(7).mean() + 1e-10)))

    # ---- Scoring ----
    long_score = pd.Series(0, index=df.index)
    short_score = pd.Series(0, index=df.index)

    # 1. Price vs VA position
    long_score += (df['close'] > df['vp_vah']).astype(int)       # breakout above VA
    short_score += (df['close'] < df['vp_val']).astype(int)      # breakdown below VA
    # Bonus: price in VA middle = neutral (don't add to either)

    # 2. Price vs POC
    long_score += (df['close'] > df['vp_poc']).astype(int)
    short_score += (df['close'] < df['vp_poc']).astype(int)

    # 3. Volume confirmation
    long_score += (df['rvol'] > rvol_threshold).astype(int)
    short_score += (df['rvol'] > rvol_threshold).astype(int)

    # 4. Trend momentum
    long_score += (df['close'] > df['ma20']).astype(int)
    short_score += (df['close'] < df['ma20']).astype(int)

    # 5. POC magnetic effect: price near POC after breaking it
    poc_dist_pct = abs(df['close'] - df['vp_poc']) / (df['vp_poc'] + 1e-10) * 100
    near_poc = poc_dist_pct < 1.0  # within 1% of POC
    long_score += (near_poc & (df['close'] > df['vp_poc'])).astype(int)
    short_score += (near_poc & (df['close'] < df['vp_poc'])).astype(int)

    # Signal: >= 3/5 AND dominant direction
    long_mask = (long_score >= 3) & (long_score > short_score)
    short_mask = (short_score >= 3) & (short_score > long_score)

    signal = pd.Series(0, index=df.index)
    signal[long_mask] = 1
    signal[short_mask] = -1

    confidence = pd.Series(0.0, index=df.index)
    confidence[long_mask] = long_score[long_mask] / 5.0
    confidence[short_mask] = short_score[short_mask] / 5.0

    # VA width bonus: wider VA = stronger signal potential
    wide_va = df['vp_va_width'] > df['vp_va_width'].rolling(50).mean()
    confidence[long_mask & wide_va] = (confidence[long_mask & wide_va] + 0.05).clip(0, 1)
    confidence[short_mask & wide_va] = (confidence[short_mask & wide_va] + 0.05).clip(0, 1)

    df['signal'] = signal.astype(int)
    df['confidence'] = confidence
    df['direction'] = signal.map({1: 'bullish', -1: 'bearish', 0: 'neutral'})

    return df


def compute_volume_profile_signal(
    df_daily: pd.DataFrame,
    df_hourly: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """School-compatible signal interface."""
    if df_daily is None or df_daily.empty or len(df_daily) < 60:
        return None

    try:
        result = generate_volume_profile_signals(df_daily)
        last = result.iloc[-1]

        if last['signal'] == 0:
            return {
                'signal': 'neutral', 'confidence': 0.0,
                'metadata': {
                    'poc': float(last['vp_poc']) if not pd.isna(last['vp_poc']) else 0,
                    'vah': float(last['vp_vah']) if not pd.isna(last['vp_vah']) else 0,
                    'val': float(last['vp_val']) if not pd.isna(last['vp_val']) else 0,
                    'va_width_pct': float(last['vp_va_width']) if not pd.isna(last['vp_va_width']) else 0,
                    'rvol': float(last['rvol']),
                }
            }

        return {
            'signal': 'bullish' if last['signal'] == 1 else 'bearish',
            'confidence': float(last['confidence']),
            'metadata': {
                'poc': float(last['vp_poc']) if not pd.isna(last['vp_poc']) else 0,
                'vah': float(last['vp_vah']) if not pd.isna(last['vp_vah']) else 0,
                'val': float(last['vp_val']) if not pd.isna(last['vp_val']) else 0,
                'va_width_pct': float(last['vp_va_width']) if not pd.isna(last['vp_va_width']) else 0,
                'rvol': float(last['rvol']),
            }
        }
    except Exception:
        return None


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    df = get_daily_kline('000012', days=200)
    if df is not None and not df.empty:
        result = generate_volume_profile_signals(df)
        last = result.iloc[-1]
        print(f"Latest: {result.index[-1]}")
        print(f"Signal: {last['signal']} ({last['direction']})")
        print(f"Confidence: {last['confidence']:.1%}")
        print(f"POC: {last['vp_poc']:.2f} | VAH: {last['vp_vah']:.2f} | VAL: {last['vp_val']:.2f}")
        print(f"VA Width: {last['vp_va_width']:.1f}%")
        print(f"Bullish: {(result['signal']==1).sum()} | Bearish: {(result['signal']==-1).sum()} | Neutral: {(result['signal']==0).sum()}")
