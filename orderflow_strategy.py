#!/usr/bin/env python
# orderflow_strategy.py
"""
Order Flow + VWAP + RSI + ATR-based TP/SL strategy.

Components:
  1. VWAP - Volume Weighted Average Price with deviation bands
  2. CVD - Cumulative Volume Delta (proxy from OHLCV)
  3. ORV - Order flow + Relative Volume zones
  4. RSI - Momentum filter
  5. ATR - Dynamic Take Profit / Stop Loss levels

Based on research combining:
  - VWAP-RSI Momentum-Reversion Hybrid Scalping System (FMZ)
  - Relative Volume Zone + Smart Order Flow Dynamic S/R (TradingView)
  - Cumulative Volume Delta with divergence detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import date


# =============================================================================
# Indicator Computation
# =============================================================================

def compute_vwap(df: pd.DataFrame, session_reset: bool = False) -> pd.Series:
    """
    Compute session or cumulative VWAP.

    VWAP = cumsum(volume * typical_price) / cumsum(volume)
    typical_price = (high + low + close) / 3
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    vp = df['volume'] * typical_price
    if session_reset:
        # Reset daily — use groupby on date
        vwap = vp.groupby(df.index.date).cumsum() / df['volume'].groupby(df.index.date).cumsum()
    else:
        vwap = vp.cumsum() / df['volume'].cumsum()
    return vwap


def compute_vwap_bands(df: pd.DataFrame, vwap: pd.Series,
                       num_std: float = 2.0, window: int = 20) -> Dict[str, pd.Series]:
    """Compute VWAP standard deviation bands."""
    dev = (df['close'] - vwap) / vwap  # normalized deviation
    rolling_std = dev.rolling(window).std()
    return {
        'vwap_upper': vwap * (1 + num_std * rolling_std),
        'vwap_lower': vwap * (1 - num_std * rolling_std),
        'vwap_dev': dev,
        'vwap_dev_zscore': (dev - dev.rolling(window).mean()) / (rolling_std + 1e-10),
    }


def compute_cvd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative Volume Delta (proxy from OHLCV).

    Estimates buy/sell volume per bar using close location:
      buy_vol  = volume * (close - low) / (high - low)
      sell_vol = volume - buy_vol
      delta    = buy_vol - sell_vol
      cvd      = cumsum(delta)
    """
    hl_range = df['high'] - df['low'] + 1e-10
    df = df.copy()
    df['buy_vol'] = df['volume'] * (df['close'] - df['low']) / hl_range
    df['sell_vol'] = df['volume'] - df['buy_vol']
    df['delta'] = df['buy_vol'] - df['sell_vol']
    df['cvd'] = df['delta'].cumsum()

    # CVD moving average for crossover signals
    df['cvd_sma'] = df['cvd'].rolling(20).mean()
    df['cvd_zscore'] = (df['cvd'] - df['cvd'].rolling(50).mean()) / \
                       (df['cvd'].rolling(50).std() + 1e-10)
    return df


def compute_order_flow_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Order Flow Imbalance (OFI).

    OFI = (buy_vol - sell_vol) / total_vol  ∈ [-1, 1]
    Positive → buying pressure, Negative → selling pressure
    """
    if 'buy_vol' not in df.columns:
        df = compute_cvd(df)
    return (df['buy_vol'] - df['sell_vol']) / (df['volume'] + 1e-10)


def compute_relative_volume(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Relative Volume = Current Volume / SMA(Volume, lookback).
    > 1.5 → high volume zone (potential accumulation/distribution)
    """
    avg_vol = df['volume'].rolling(lookback).mean()
    return df['volume'] / (avg_vol + 1e-10)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =============================================================================
# Composite Signal Generation
# =============================================================================

def generate_orderflow_signals(
    df: pd.DataFrame,
    rsi_period: int = 7,
    rsi_oversold: float = 35,
    rsi_overbought: float = 70,
    vol_lookback: int = 20,
    vol_multiplier: float = 1.5,
    cvd_window: int = 20,
    atr_period: int = 14,
    stop_atr_mult: float = 1.0,
    target_atr_mult: float = 2.0,
) -> pd.DataFrame:
    """
    Generate composite order flow + VWAP + RSI signals.

    Returns DataFrame with columns:
      signal: 1=bullish, -1=bearish, 0=neutral
      confidence: signal confidence score [0, 1]
      stop_price: suggested stop loss
      target_price: suggested take profit
    """
    df = df.copy()

    # ---- VWAP ----
    df['vwap'] = compute_vwap(df)
    bands = compute_vwap_bands(df, df['vwap'])
    df['vwap_upper'] = bands['vwap_upper']
    df['vwap_lower'] = bands['vwap_lower']
    df['vwap_dev'] = bands['vwap_dev']

    # ---- CVD / Order Flow ----
    df = compute_cvd(df)
    df['ofi'] = compute_order_flow_imbalance(df)

    # ---- Relative Volume ----
    df['rel_vol'] = compute_relative_volume(df, lookback=vol_lookback)

    # ---- RSI ----
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ---- EMA ----
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()

    # ---- ATR ----
    df['atr'] = compute_atr(df, period=atr_period)

    # ---- Composite Signal: Scoring System (≥3/5 conditions → signal) ----
    signal = pd.Series(0, index=df.index)
    confidence = pd.Series(0.0, index=df.index)

    # Long score: count how many bullish conditions are met (0-5)
    long_score = pd.Series(0, index=df.index)
    long_score += (df['rsi'] < rsi_oversold).astype(int)           # oversold bounce
    long_score += (df['close'] > df['vwap']).astype(int)           # above VWAP
    long_score += (df['close'] > df['ema_20']).astype(int)         # above EMA
    long_score += (df['cvd'] > df['cvd_sma']).astype(int)          # order flow buying
    long_score += (df['rel_vol'] > vol_multiplier).astype(int)      # high volume

    # Short score: count how many bearish conditions are met (0-5)
    short_score = pd.Series(0, index=df.index)
    short_score += (df['rsi'] > rsi_overbought).astype(int)
    short_score += (df['close'] < df['vwap']).astype(int)
    short_score += (df['close'] < df['ema_20']).astype(int)
    short_score += (df['cvd'] < df['cvd_sma']).astype(int)
    short_score += (df['rel_vol'] > vol_multiplier).astype(int)

    # Signal when ≥3/5 conditions met AND the opposing side has ≤2
    long_mask = (long_score >= 3) & (short_score <= 2)
    short_mask = (short_score >= 3) & (long_score <= 2)

    signal[long_mask] = 1
    signal[short_mask] = -1

    # Confidence = score / 5 + bonus for divergence
    confidence[long_mask] = long_score[long_mask] / 5.0
    confidence[short_mask] = short_score[short_mask] / 5.0

    # VWAP-CVD divergence bonus (price vs order flow disagreement)
    df['close_low_5'] = df['close'].rolling(5).min()
    df['close_high_5'] = df['close'].rolling(5).max()
    df['cvd_high_5'] = df['cvd'].rolling(5).max()
    df['cvd_low_5'] = df['cvd'].rolling(5).min()
    bull_div = (df['close'] <= df['close_low_5']) & (df['cvd'] > df['cvd_low_5'].shift(5))
    bear_div = (df['close'] >= df['close_high_5']) & (df['cvd'] < df['cvd_high_5'].shift(5))

    confidence[bull_div] = (confidence[bull_div] + 0.1).clip(0, 1)
    confidence[bear_div] = (confidence[bear_div] + 0.1).clip(0, 1)

    # Stop loss & take profit
    df['stop_loss'] = np.where(
        signal == 1,
        df['close'] - stop_atr_mult * df['atr'],
        np.where(signal == -1,
                 df['close'] + stop_atr_mult * df['atr'],
                 np.nan)
    )
    df['target_price'] = np.where(
        signal == 1,
        df['close'] + target_atr_mult * df['atr'],
        np.where(signal == -1,
                 df['close'] - target_atr_mult * df['atr'],
                 np.nan)
    )

    df['signal'] = signal.astype(int)
    df['confidence'] = confidence
    df['direction'] = signal.map({1: 'bullish', -1: 'bearish', 0: 'neutral'})

    return df


# =============================================================================
# School-compatible interface (integrates with expert_ensemble)
# =============================================================================

def compute_orderflow_signal(
    df_daily: pd.DataFrame,
    df_hourly: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    """
    Compute order flow signal for a single stock, compatible with
    the school signal interface used by expert_ensemble.

    Returns dict with keys:
      signal: 'bullish' | 'bearish' | 'neutral'
      confidence: float [0, 1]
      metadata: dict with VWAP, CVD, RSI details
    """
    if df_daily is None or df_daily.empty or len(df_daily) < 30:
        return None

    try:
        result = generate_orderflow_signals(df_daily)
        last = result.iloc[-1]

        if last['signal'] == 0:
            return {
                'signal': 'neutral',
                'confidence': 0.0,
                'metadata': {
                    'vwap': float(last['vwap']),
                    'vwap_dev': float(last['vwap_dev']),
                    'rsi': float(last['rsi']),
                    'ofi': float(last['ofi']),
                    'rel_vol': float(last['rel_vol']),
                    'cvd_zscore': float(last['cvd_zscore']),
                    'stop_loss': float(last['stop_loss']) if not np.isnan(last['stop_loss']) else None,
                    'target_price': float(last['target_price']) if not np.isnan(last['target_price']) else None,
                }
            }

        return {
            'signal': 'bullish' if last['signal'] == 1 else 'bearish',
            'confidence': float(last['confidence']),
            'metadata': {
                'vwap': float(last['vwap']),
                'vwap_dev': float(last['vwap_dev']),
                'rsi': float(last['rsi']),
                'ofi': float(last['ofi']),
                'rel_vol': float(last['rel_vol']),
                'cvd_zscore': float(last['cvd_zscore']),
                'stop_loss': float(last['stop_loss']) if not np.isnan(last['stop_loss']) else None,
                'target_price': float(last['target_price']) if not np.isnan(last['target_price']) else None,
            }
        }
    except Exception as e:
        return None


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    # Test on a single stock
    df = get_daily_kline('000012', days=200)
    if df is not None and not df.empty:
        result = generate_orderflow_signals(df)
        print("=== Order Flow Strategy Signals ===")
        print(f"Latest date: {result.index[-1]}")
        last = result.iloc[-1]
        print(f"  Signal:     {last['signal']} ({last['direction']})")
        print(f"  Confidence: {last['confidence']:.2%}")
        print(f"  RSI:        {last['rsi']:.1f}")
        print(f"  VWAP:       {last['vwap']:.2f}")
        print(f"  VWAP Dev:   {last['vwap_dev']:.2%}")
        print(f"  OFI:        {last['ofi']:.3f}")
        print(f"  Rel Vol:    {last['rel_vol']:.1f}x")
        print(f"  CVD Z:      {last['cvd_zscore']:.2f}")
        print(f"  Stop Loss:  {last['stop_loss']:.2f}")
        print(f"  Target:     {last['target_price']:.2f}")
        print(f"\n  Signal distribution:")
        print(f"    Bullish: {(result['signal']==1).sum()}")
        print(f"    Bearish: {(result['signal']==-1).sum()}")
        print(f"    Neutral: {(result['signal']==0).sum()}")
    else:
        print("No data loaded")
