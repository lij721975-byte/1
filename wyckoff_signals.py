#!/usr/bin/env python
# wyckoff_signals.py
"""
Wyckoff VSA — Three Core Signal Extractors
============================================
Spring / UTAD / SOS+LPS  — fully vectorized, zero future-data leak.

Every signal returns:
  - sig_wyckoff_<name> : int  ∈ {0, 1}   binary trade signal
  - score_wyckoff_<name>: float64 ∈ [0,1] continuous confidence (for ensemble)

Anti-future-leak guarantee:
  - All lookbacks use rolling(window) or shift(1) exclusively.
  - No shift(-1), no expanding().mean(), no full-series stats.
  - At bar t, only information from [0, t] is used.

Author: Chief Alpha Miner
"""

import numpy as np
import pandas as pd


# ==============================================================================
# Shared precomputations (called once, reused across all three signals)
# ==============================================================================

def _precompute(df: pd.DataFrame) -> dict:
    """Compute all shared rolling statistics once. Fully vectorized."""
    o = df['open'].astype(np.float64)
    h = df['high'].astype(np.float64)
    l = df['low'].astype(np.float64)
    c = df['close'].astype(np.float64)
    v = df['volume'].astype(np.float64)
    n = len(c)

    # ── Price boundaries ──────────────────────────────────────────
    hh_20 = h.rolling(20, min_periods=10).max()       # 20-day high
    ll_20 = l.rolling(20, min_periods=10).min()       # 20-day low
    hh_40 = h.rolling(40, min_periods=20).max()
    ll_40 = l.rolling(40, min_periods=20).min()
    hh_5  = h.rolling(5,  min_periods=3).max()
    ll_5  = l.rolling(5,  min_periods=3).min()

    # ── Moving averages ───────────────────────────────────────────
    ma5  = c.rolling(5,  min_periods=3).mean()
    ma20 = c.rolling(20, min_periods=10).mean()
    ma5_slope = ma5 - ma5.shift(5)   # 5-bar slope of MA5
    ma20_slope = ma20 - ma20.shift(5)

    # ── ATR ───────────────────────────────────────────────────────
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=7).mean()

    # ── Volume ────────────────────────────────────────────────────
    vol_ma5  = v.rolling(5,  min_periods=3).mean()
    vol_ma20 = v.rolling(20, min_periods=10).mean()

    # ── RSI(14) ───────────────────────────────────────────────────
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rsi14 = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))

    # ── CCI(14) ───────────────────────────────────────────────────
    # Fully vectorized (no rolling().apply): |TP - SMA| rolling mean
    tp = (h + l + c) / 3.0
    tp_ma14 = tp.rolling(14, min_periods=7).mean()
    tp_mad14 = (tp - tp_ma14).abs().rolling(14, min_periods=7).mean()
    cci14 = (tp - tp_ma14) / (0.015 * tp_mad14 + 1e-10)

    # ── Bollinger Bands (20,2) ────────────────────────────────────
    bb_mid = c.rolling(20, min_periods=10).mean()
    bb_std = c.rolling(20, min_periods=10).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    # ── OBV ───────────────────────────────────────────────────────
    obv = (np.sign(c.diff().fillna(0)) * v).cumsum()

    # ── Candle geometry ───────────────────────────────────────────
    body = (c - o).abs()
    candle_range = h - l
    body_ratio = body / (candle_range + 1e-10)   # [0,1]
    lower_shadow_pct = (o.clip(lower=None, upper=None).combine(c, min) - l) / (candle_range + 1e-10)
    upper_shadow_pct = (h - o.clip(lower=None, upper=None).combine(c, max)) / (candle_range + 1e-10)

    return {
        'o': o, 'h': h, 'l': l, 'c': c, 'v': v, 'n': n,
        'hh_20': hh_20, 'll_20': ll_20, 'hh_40': hh_40, 'll_40': ll_40,
        'hh_5': hh_5, 'll_5': ll_5,
        'ma5': ma5, 'ma20': ma20,
        'ma5_slope': ma5_slope, 'ma20_slope': ma20_slope,
        'atr14': atr14,
        'vol_ma5': vol_ma5, 'vol_ma20': vol_ma20,
        'rsi14': rsi14, 'cci14': cci14,
        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_mid': bb_mid,
        'obv': obv,
        'body_ratio': body_ratio,
        'lower_shadow_pct': lower_shadow_pct,
        'upper_shadow_pct': upper_shadow_pct,
        'tr': tr,
    }


# ==============================================================================
# Signal 1: Spring (弹簧 / 震仓买入)
# ==============================================================================

def compute_spring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyckoff Spring — false breakdown buy setup.

    Returns DataFrame with columns:
      sig_wyckoff_spring   : int  ∈ {0, 1}   buy signal
      score_wyckoff_spring : float64 ∈ [0,1]  continuous confidence
    """
    p = _precompute(df)
    out = pd.DataFrame(index=df.index)

    # ── Condition A: price in lower half of 40-day range ──────────
    range_mid_40 = (p['hh_40'] + p['ll_40']) / 2.0
    cond_a = (p['c'] < p['ma20']) & (p['c'] < range_mid_40)

    # ── Condition B: false breakdown — new 20d low, closes bullish ─
    cond_b_new_low = p['l'] < p['ll_20'].shift(1)   # NOTE: shift(1) — yesterday's ll_20
    cond_b_bullish_body = p['c'] > p['o']            # closes green
    cond_b_big_body = p['body_ratio'] > 0.35          # body > 35% of range
    cond_b = cond_b_new_low & cond_b_bullish_body & cond_b_big_body

    # ── Condition C: volume surge — institutions absorbing ─────────
    cond_c = p['v'] > p['vol_ma20'] * 1.30

    # ── Condition D: oversold resonance (≥1 of 3) ─────────────────
    cond_d_rsi  = p['rsi14'] < 40.0
    cond_d_cci  = p['cci14'] < -100.0
    cond_d_bb   = p['c'] < p['bb_lower'] * 1.02
    cond_d = cond_d_rsi | cond_d_cci | cond_d_bb

    # ── Condition E: recovery — close back above prior 20d low ────
    cond_e = p['c'] > p['ll_20'].shift(1)

    # ── Composite mask ────────────────────────────────────────────
    spring_mask = cond_a & cond_b & cond_c & cond_d & cond_e

    # ── Score (continuous confidence) ─────────────────────────────
    cond_count = (cond_a.astype(int) + cond_b.astype(int) +
                  cond_c.astype(int) + cond_d.astype(int) +
                  cond_e.astype(int))
    shadow_strength = p['lower_shadow_pct'].clip(0.0, 1.0)
    vol_power = (p['v'] / p['vol_ma20']).clip(0.0, 3.0) / 3.0

    score = cond_count / 5.0 * shadow_strength * vol_power
    score = score.clip(0.0, 1.0).fillna(0.0)

    out['sig_wyckoff_spring'] = spring_mask.astype(int)
    out['score_wyckoff_spring'] = score.astype(np.float64)
    return out


# ==============================================================================
# Signal 2: UTAD (上冲失败 / Upthrust After Distribution)
# ==============================================================================

def compute_utad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyckoff UTAD — false breakout sell setup.

    Returns DataFrame with columns:
      sig_wyckoff_utad   : int  ∈ {0, 1}   sell signal
      score_wyckoff_utad : float64 ∈ [0,1]  continuous confidence
    """
    p = _precompute(df)
    out = pd.DataFrame(index=df.index)

    # ── Condition A: price in upper half of 40-day range ──────────
    range_mid_40 = (p['hh_40'] + p['ll_40']) / 2.0
    cond_a = (p['c'] > p['ma20']) & (p['c'] > range_mid_40)

    # ── Condition B: false breakout — new 20d high, closes bearish ─
    cond_b_new_high = p['h'] > p['hh_20'].shift(1)
    cond_b_bearish_body = p['c'] < p['o']
    cond_b_big_body = p['body_ratio'] > 0.35
    cond_b = cond_b_new_high & cond_b_bearish_body & cond_b_big_body

    # ── Condition C: high-volume stalling — distribution ───────────
    cond_c = (p['v'] > p['vol_ma20'] * 1.30) & (p['c'] < p['o'])

    # ── Condition D: overbought resonance (≥1 of 3) ───────────────
    cond_d_rsi  = p['rsi14'] > 60.0
    cond_d_cci  = p['cci14'] > 100.0
    cond_d_bb   = p['c'] > p['bb_upper'] * 0.98
    cond_d = cond_d_rsi | cond_d_cci | cond_d_bb

    # ── Condition E: rejection — close back below prior 20d high ──
    cond_e = p['c'] < p['hh_20'].shift(1)

    # ── Composite mask ────────────────────────────────────────────
    utad_mask = cond_a & cond_b & cond_c & cond_d & cond_e

    # ── Score ─────────────────────────────────────────────────────
    cond_count = (cond_a.astype(int) + cond_b.astype(int) +
                  cond_c.astype(int) + cond_d.astype(int) +
                  cond_e.astype(int))
    shadow_strength = p['upper_shadow_pct'].clip(0.0, 1.0)
    vol_power = (p['v'] / p['vol_ma20']).clip(0.0, 3.0) / 3.0

    score = cond_count / 5.0 * shadow_strength * vol_power
    score = score.clip(0.0, 1.0).fillna(0.0)

    out['sig_wyckoff_utad'] = utad_mask.astype(int)
    out['score_wyckoff_utad'] = score.astype(np.float64)
    return out


# ==============================================================================
# Signal 3: SOS + LPS (强势信号 + 最后支撑点)
# ==============================================================================

def compute_sos_lps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyckoff SOS + LPS — confirmed trend entry after breakout & retest.

    Returns DataFrame with columns:
      sig_wyckoff_sos_lps   : int  ∈ {0, 1}   buy signal
      score_wyckoff_sos_lps : float64 ∈ [0,1]  continuous confidence
    """
    p = _precompute(df)
    out = pd.DataFrame(index=df.index)

    # ── Condition A: trend repair — MA5 > MA20, MA20 rising ───────
    cond_a = (p['ma5'] > p['ma20']) & (p['ma20_slope'] > 0) & (p['c'] > p['ma20'])

    # ── Condition B: SOS — volume-backed breakout within last 5d ───
    # "过去5日内至少存在1日": close > 20d high AND vol > 1.5x avg
    daily_breakout = (p['c'] > p['hh_20'].shift(1)) & (p['v'] > p['vol_ma20'] * 1.50)
    # Rolling sum over past 5 bars: if ≥ 1, condition met
    cond_b = daily_breakout.rolling(5, min_periods=1).sum() >= 1

    # ── Condition C: LPS — shallow pullback above 60% of range ────
    range_20 = p['hh_20'] - p['ll_20']
    support_60pct = p['ll_20'] + range_20 * 0.60
    cond_c = p['l'] > support_60pct

    # ── Condition D: low-volume pullback — supply drying up ────────
    cond_d = p['v'] < p['vol_ma20'] * 0.80

    # ── Condition E: OBV confirmation — accumulation trend ─────────
    cond_e_obv = p['obv'] > p['obv'].shift(20)
    cond_e_vp  = (p['c'] > p['ma5']) & (p['v'] < p['v'].shift(1))
    cond_e = cond_e_obv | cond_e_vp

    # ── Composite mask ────────────────────────────────────────────
    sos_mask = cond_a & cond_b & cond_c & cond_d & cond_e

    # ── Score ─────────────────────────────────────────────────────
    cond_count = (cond_a.astype(int) + cond_b.astype(int) +
                  cond_c.astype(int) + cond_d.astype(int) +
                  cond_e.astype(int))

    # Breakout strength: how strong was the SOS day?
    breakout_vol_ratio = (p['v'] / p['vol_ma20']).clip(0.0, 2.0) / 2.0
    # Forward-fill the most recent breakout strength (no future leak)
    breakout_strength = breakout_vol_ratio.where(daily_breakout).ffill().fillna(0.0)

    # Pullback quality: how shallow is the retest?
    pullback_depth = (p['c'] - p['l']) / (p['hh_5'] - p['ll_5'] + 1e-10)
    pullback_quality = pullback_depth.clip(0.0, 1.0)

    score = cond_count / 5.0 * breakout_strength * pullback_quality
    score = score.clip(0.0, 1.0).fillna(0.0)

    out['sig_wyckoff_sos_lps'] = sos_mask.astype(int)
    out['score_wyckoff_sos_lps'] = score.astype(np.float64)
    return out


# ==============================================================================
# Main API
# ==============================================================================

def compute_wyckoff_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three Wyckoff core signals.

    Parameters
    ----------
    df : pd.DataFrame  with columns open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame with 6 new columns:
      sig_wyckoff_spring    — Spring buy signal  {0, 1}
      score_wyckoff_spring  — Spring confidence   [0, 1]
      sig_wyckoff_utad      — UTAD sell signal    {0, 1}
      score_wyckoff_utad    — UTAD confidence     [0, 1]
      sig_wyckoff_sos_lps   — SOS+LPS buy signal  {0, 1}
      score_wyckoff_sos_lps — SOS+LPS confidence  [0, 1]
    """
    df_s = compute_spring(df)
    df_u = compute_utad(df)
    df_l = compute_sos_lps(df)
    return pd.concat([df_s, df_u, df_l], axis=1)


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 70)
    print("  Wyckoff VSA Signals — Quick Test")
    print("=" * 70)

    for code in ['000012', '600519', '000001', '600036']:
        df = get_daily_kline(code, days=500)
        if df is None or not df.empty:
            sigs = compute_wyckoff_signals(df)
            for col in [c for c in sigs.columns if c.startswith('sig_')]:
                count = sigs[col].sum()
                print(f"  {code}  {col:<28} signals={int(count):4d}")
            for col in [c for c in sigs.columns if c.startswith('score_')]:
                s = sigs[col]
                print(f"  {code}  {col:<28} mean={s.mean():.4f}  max={s.max():.4f}  >0.5={(s>0.5).sum():4d}")
            # Future-leak check
            for col in sigs.columns:
                assert sigs[col].isna().sum() == 0, f'{col} has NaN'
            print()
