#!/usr/bin/env python
# capital_flow.py — Zero-HTTP local order flow capital signal
"""
Fully local capital flow signal using CVD + RVOL + VWAP divergence.

ZERO network requests. ZERO pywencai. ZERO cookies.
All analysis based on local TDX K-line data and pre-computed indicators.

Mapping logic:
  CVD (Cumulative Volume Delta) ramp    → "主力资金净流入" proxy
  RVOL > 1.5 combined with VWAP offset  → "机构异动" signal
  Volume bar range analysis             → "大单占比" estimate

Backward compatible: output dict structure unchanged.
"""

import numpy as np
from typing import Dict, Optional, Any


# =============================================================================
# Local Order Flow Analysis (replaces pywencai)
# =============================================================================

def compute_cvd(df_daily) -> np.ndarray:
    """
    Cumulative Volume Delta from OHLC bar analysis.

    CVD[t] = Σ volume[i] × sign(close[i] - open[i]) × position_in_range[i]

    Positive CVD ramp → buying pressure (主力净流入 proxy)
    Negative CVD ramp → selling pressure (主力净流出 proxy)
    """
    candle_range = df_daily['high'] - df_daily['low'] + 1e-9

    # Buy pressure: how much of the bar's volume is "buying" (close near high)
    buy_ratio = (df_daily['close'] - df_daily['low']) / candle_range
    sell_ratio = (df_daily['high'] - df_daily['close']) / candle_range
    vol_delta = df_daily['volume'] * (buy_ratio - sell_ratio)

    cvd = vol_delta.cumsum()
    return cvd.values, vol_delta.values


def compute_rvol(df_daily, lookback: int = 20) -> np.ndarray:
    """Relative Volume = current vol / 20-day avg vol."""
    avg_vol = df_daily['volume'].rolling(lookback).mean()
    return (df_daily['volume'] / (avg_vol + 1e-10)).values


def compute_vwap_offset(df_daily) -> np.ndarray:
    """VWAP deviation: (close - VWAP) / VWAP."""
    typical = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 3
    vwap = (typical * df_daily['volume']).cumsum() / (df_daily['volume'].cumsum() + 1e-10)
    return ((df_daily['close'] - vwap) / (vwap + 1e-10)).values


# =============================================================================
# Signal generators (backward compatible outputs)
# =============================================================================

def get_northbound_signal(symbol: str = None, indicators: Dict = None,
                          df_daily=None) -> Dict[str, Any]:
    """
    Northbound proxy via CVD + VWAP trend alignment.

    (No real northbound data available locally — uses persistent CVD trend
    as a proxy for institutional flow direction.)
    """
    try:
        if df_daily is not None and len(df_daily) >= 10:
            _, vol_delta = compute_cvd(df_daily)
            cvd_5d = vol_delta[-5:].sum() if len(vol_delta) >= 5 else 0
            cvd_10d = vol_delta[-10:].sum() if len(vol_delta) >= 10 else cvd_5d
            avg_vol = float(df_daily['volume'].tail(5).mean())

            # Consecutive positive CVD days as "northbound buy days" proxy
            delta_signs = np.sign(vol_delta[-10:])
            consec_buy = 0
            for s in reversed(delta_signs):
                if s > 0: consec_buy += 1
                else: break

            net_inflow = cvd_5d  # CVD accumulation as "net inflow"

            if consec_buy >= 5:
                direction, conf = 'bullish', min(0.85, 0.55 + consec_buy * 0.05)
            elif consec_buy >= 2:
                direction, conf = 'bullish', 0.40 + consec_buy * 0.05
            elif cvd_10d > avg_vol * 0.5:
                direction, conf = 'bullish', 0.35
            elif cvd_10d < -avg_vol * 0.5:
                direction, conf = 'bearish', 0.30
            else:
                direction, conf = 'neutral', 0.20

            return {
                'direction': direction,
                'consecutive_buy_days': consec_buy,
                'net_inflow': round(float(cvd_10d), 1),
                'confidence': round(conf, 3),
                'source': 'cvd_proxy(local)',
            }
    except Exception:
        pass

    return {'direction': 'neutral', 'consecutive_buy_days': 0,
            'net_inflow': 0, 'confidence': 0.15, 'source': 'fallback'}


def get_main_force_signal(symbol: str = None, indicators: Dict = None,
                          df_daily=None) -> Dict[str, Any]:
    """
    Main force signal via RVOL + VWAP divergence + CVD ramp.

    CVD急剧上升 → "主力净流入"
    RVOL > 1.5 + close > VWAP → "机构放量拉升"
    """
    try:
        if df_daily is not None and len(df_daily) >= 10:
            _, vol_delta = compute_cvd(df_daily)
            cvd_5d = vol_delta[-5:].sum() if len(vol_delta) >= 5 else 0
            rvol_arr = compute_rvol(df_daily)
            rvol_now = float(rvol_arr[-1]) if len(rvol_arr) > 0 else 1.0
            vwap_off = compute_vwap_offset(df_daily)
            vwap_now = float(vwap_off[-1]) if len(vwap_off) > 0 else 0.0

            # Large order ratio: use volume concentration as proxy
            # (vol spike vs average = "institutional activity")
            large_order_ratio = min(0.80, max(0.05, (rvol_now - 1.0) * 0.25))

            avg_vol = float(df_daily['volume'].tail(5).mean())
            net_inflow = cvd_5d  # CVD as "net inflow" in 手*100

            # Scoring
            if cvd_5d > avg_vol * 2.0 and vwap_now > 0:
                direction, conf = 'bullish', 0.70
            elif cvd_5d > 0 and rvol_now > 1.5:
                direction, conf = 'bullish', 0.55
            elif cvd_5d > 0:
                direction, conf = 'bullish', 0.40
            elif cvd_5d < -avg_vol * 2.0 and vwap_now < 0:
                direction, conf = 'bearish', 0.55
            elif cvd_5d < 0 and rvol_now < 0.7:
                direction, conf = 'bearish', 0.40
            elif cvd_5d < 0:
                direction, conf = 'bearish', 0.30
            else:
                direction, conf = 'neutral', 0.20

            return {
                'direction': direction,
                'main_net_inflow': round(float(cvd_5d), 0),
                'large_order_ratio': round(large_order_ratio, 3),
                'confidence': round(conf, 3),
                'source': 'orderflow_proxy(local)',
            }
    except Exception:
        pass

    return {'direction': 'neutral', 'main_net_inflow': 0,
            'large_order_ratio': 0, 'confidence': 0.10, 'source': 'error'}


# =============================================================================
# Composite signal (backward compatible)
# =============================================================================

def get_capital_flow_signal(symbol: str = None, indicators: Dict = None,
                            df_daily=None) -> Dict[str, Any]:
    """
    Composite capital flow signal using local order flow only.

    Args may be:
      - (symbol)       → loads data internally (slow, rarely used)
      - (indicators, df_daily) → uses pre-computed data (fast, preferred)

    Returns (unchanged interface):
      {'direction': 'bullish'|'bearish'|'neutral',
       'score': float,
       'northbound': {...},
       'main_force': {...},
       'resonance': bool,
       'reasons': [...]}
    """
    # Load data if only symbol provided
    if df_daily is None and symbol is not None:
        try:
            from data_loader import get_daily_kline
            df_daily = get_daily_kline(symbol, days=60)
        except Exception:
            df_daily = None

    nb = get_northbound_signal(symbol=symbol, indicators=indicators, df_daily=df_daily)
    mf = get_main_force_signal(symbol=symbol, indicators=indicators, df_daily=df_daily)

    score = 0.0
    reasons = []

    # Northbound scoring (40%)
    if nb['direction'] == 'bullish':
        score += 0.40 * nb['confidence']
        if nb.get('consecutive_buy_days', 0) >= 3:
            reasons.append(f'CVD连续{nb["consecutive_buy_days"]}日净流入(北向代理)')
    elif nb['direction'] == 'bearish':
        score -= 0.40 * nb['confidence']
        reasons.append('CVD持续流出(北向代理)')

    # Main force scoring (40%)
    if mf['direction'] == 'bullish':
        score += 0.40 * mf['confidence']
        if mf.get('large_order_ratio', 0) > 0.25:
            reasons.append(f'机构放量(RVOL+{mf["large_order_ratio"]:.0%})')
        else:
            reasons.append('CVD净流入(主力代理)')
    elif mf['direction'] == 'bearish':
        score -= 0.40 * mf['confidence']
        reasons.append('CVD净流出(主力代理)')

    # Resonance bonus (20%)
    resonance = (nb['direction'] == mf['direction'] and nb['direction'] != 'neutral')
    if resonance:
        score += 0.20 * min(nb['confidence'], mf['confidence'])
        reasons.append('资金共振(CVD+RVOL)')

    if score >= 0.40:
        direction = 'bullish'
    elif score <= -0.30:
        direction = 'bearish'
    else:
        direction = 'neutral'

    return {
        'direction': direction,
        'score': round(max(-1.0, min(1.0, score)), 3),
        'confidence': round(abs(score), 3),
        'northbound': nb,
        'main_force': mf,
        'resonance': resonance,
        'reasons': reasons[:3],
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == '__main__':
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else '000001'
    print(f"=== 资金流向信号 (本地订单流): {sym} ===")

    from data_loader import get_daily_kline
    df = get_daily_kline(sym, days=60)
    if df is not None and not df.empty:
        signal = get_capital_flow_signal(symbol=sym, df_daily=df)
        print(f"  方向: {signal['direction']}")
        print(f"  分数: {signal['score']}")
        print(f"  共振: {signal['resonance']}")
        print(f"  北向代理: {signal['northbound']['direction']} "
              f"(CVD连续{signal['northbound']['consecutive_buy_days']}日, "
              f"conf={signal['northbound']['confidence']:.2f})")
        print(f"  主力代理: {signal['main_force']['direction']} "
              f"(CVD={signal['main_force']['main_net_inflow']:.0f}, "
              f"大单代理={signal['main_force']['large_order_ratio']:.2f}, "
              f"conf={signal['main_force']['confidence']:.2f})")
        print(f"  理由: {signal['reasons']}")
    else:
        print("  无数据")
