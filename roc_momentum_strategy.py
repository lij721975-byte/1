#!/usr/bin/env python
# roc_momentum_strategy.py
"""
ROC + RSI + ATR + RVOL Momentum Breakout Strategy.

Signals:
  Long:  ROC(12) > 0 AND RSI exiting oversold AND Vol > 1.5x avg AND ATR expanding
  Short: ROC(12) < 0 AND RSI exiting overbought AND Vol > 1.5x avg AND ATR expanding

Stop/Target: ATR-based (1x ATR stop, 2x ATR target)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any


def compute_roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change: (price - price[N]) / price[N] * 100"""
    return (close - close.shift(period)) / close.shift(period) * 100


def compute_rsi(close: pd.Series, period: int = 7) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Relative Volume = current vol / SMA(vol, lookback)"""
    return df['volume'] / (df['volume'].rolling(lookback).mean() + 1e-10)


def generate_roc_signals(
    df: pd.DataFrame,
    roc_period: int = 12,
    rsi_period: int = 7,
    rsi_oversold: int = 35,
    rsi_overbought: int = 65,
    rvol_threshold: float = 1.3,
    atr_period: int = 14,
    atr_expand_factor: float = 1.2,
    stop_atr_mult: float = 1.0,
    target_atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Generate ROC Momentum Breakout signals.

    Scoring (0-5 conditions):
      1. ROC direction (ROC > 0 bullish, < 0 bearish)
      2. RSI context (exiting oversold for long, exiting overbought for short)
      3. Volume confirmation (RVOL > threshold)
      4. ATR expansion (ATR rising = breakout valid)
      5. Price vs MA20 (trend filter)
    Signal at >= 3/5 conditions met.
    """
    df = df.copy()

    # Indicators
    df['roc12'] = compute_roc(df['close'], roc_period)
    df['roc6'] = compute_roc(df['close'], 6)  # short-term ROC for momentum change
    df['rsi'] = compute_rsi(df['close'], rsi_period)
    df['atr14'] = compute_atr(df, atr_period)
    df['atr_pct'] = df['atr14'] / df['close']
    df['rvol'] = compute_rvol(df, lookback=20)
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # ATR expansion: is ATR increasing vs 5 bars ago?
    df['atr_expanding'] = df['atr14'] > df['atr14'].shift(5) * atr_expand_factor

    # RSI context: exiting oversold (RSI was < oversold 1 bar ago, now crossing up)
    # Or entering momentum: RSI between 40-60 and rising
    df['rsi_rising'] = df['rsi'] > df['rsi'].shift(3)

    # ---- Scoring system ----
    long_score = pd.Series(0, index=df.index)
    short_score = pd.Series(0, index=df.index)

    # 1. ROC direction
    long_score += (df['roc12'] > 2).astype(int)   # strong positive momentum
    short_score += (df['roc12'] < -2).astype(int)  # strong negative momentum

    # 2. RSI context
    long_score += ((df['rsi'] < rsi_oversold) |
                   ((df['rsi'] > 40) & (df['rsi'] < 60) & df['rsi_rising'])).astype(int)
    short_score += ((df['rsi'] > rsi_overbought) |
                    ((df['rsi'] > 40) & (df['rsi'] < 60) & ~df['rsi_rising'])).astype(int)

    # 3. Volume confirmation
    long_score += (df['rvol'] > rvol_threshold).astype(int)
    short_score += (df['rvol'] > rvol_threshold).astype(int)

    # 4. ATR expansion (breakout validity)
    long_score += df['atr_expanding'].astype(int)
    short_score += df['atr_expanding'].astype(int)

    # 5. Trend filter
    long_score += (df['close'] > df['ma20']).astype(int)
    short_score += (df['close'] < df['ma20']).astype(int)

    # Signal: >= 3/5 and dominant direction
    long_mask = (long_score >= 3) & (long_score > short_score)
    short_mask = (short_score >= 3) & (short_score > long_score)

    signal = pd.Series(0, index=df.index)
    signal[long_mask] = 1
    signal[short_mask] = -1

    # Confidence = score / 5
    confidence = pd.Series(0.0, index=df.index)
    confidence[long_mask] = long_score[long_mask] / 5.0
    confidence[short_mask] = short_score[short_mask] / 5.0

    # Bonus: ROC acceleration (ROC6 crossing above ROC12)
    roc_accel = (df['roc6'] > df['roc12']) & (df['roc6'].shift(1) <= df['roc12'].shift(1))
    confidence[long_mask & roc_accel] = (confidence[long_mask & roc_accel] + 0.10).clip(0, 1)

    # Stop/Target
    df['signal'] = signal.astype(int)
    df['confidence'] = confidence
    df['stop_loss'] = np.where(
        signal == 1, df['close'] - stop_atr_mult * df['atr14'],
        np.where(signal == -1, df['close'] + stop_atr_mult * df['atr14'], np.nan)
    )
    df['target_price'] = np.where(
        signal == 1, df['close'] + target_atr_mult * df['atr14'],
        np.where(signal == -1, df['close'] - target_atr_mult * df['atr14'], np.nan)
    )
    df['direction'] = signal.map({1: 'bullish', -1: 'bearish', 0: 'neutral'})

    return df


def compute_roc_breakout_signal(
    df_daily: pd.DataFrame,
    df_hourly: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """School-compatible signal interface."""
    if df_daily is None or df_daily.empty or len(df_daily) < 60:
        return None

    try:
        result = generate_roc_signals(df_daily)
        last = result.iloc[-1]

        if last['signal'] == 0:
            return {
                'signal': 'neutral', 'confidence': 0.0,
                'metadata': {
                    'roc12': float(last['roc12']), 'rsi': float(last['rsi']),
                    'rvol': float(last['rvol']), 'atr_pct': float(last['atr_pct']),
                }
            }

        return {
            'signal': 'bullish' if last['signal'] == 1 else 'bearish',
            'confidence': float(last['confidence']),
            'metadata': {
                'roc12': float(last['roc12']), 'rsi': float(last['rsi']),
                'rvol': float(last['rvol']), 'atr_pct': float(last['atr_pct']),
                'stop_loss': float(last['stop_loss']) if not pd.isna(last['stop_loss']) else None,
                'target_price': float(last['target_price']) if not pd.isna(last['target_price']) else None,
            }
        }
    except Exception:
        return None


if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    df = get_daily_kline('000012', days=200)
    if df is not None and not df.empty:
        result = generate_roc_signals(df)
        last = result.iloc[-1]
        print(f"Latest: {result.index[-1]}")
        print(f"Signal: {last['signal']} ({last['direction']})")
        print(f"Confidence: {last['confidence']:.2%}")
        print(f"ROC12: {last['roc12']:.1f}% | RSI: {last['rsi']:.0f} | RVOL: {last['rvol']:.1f}x")
        print(f"Bullish: {(result['signal']==1).sum()} | Bearish: {(result['signal']==-1).sum()} | Neutral: {(result['signal']==0).sum()}")
