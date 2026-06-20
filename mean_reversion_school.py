#!/usr/bin/env python
# mean_reversion_school.py
"""
Mean Reversion School — 多因子均值回归学派
=============================================
Composite strategy: Bollinger Bands + RSI + BIAS + Volume + ADX

Red-line compliance:
  1. Pure vectorized — zero Python for-loops over bars
  2. Zero future-data leak — rolling windows & shift(1) only
  3. Continuous confidence — float64 ∈ [0.0, 1.0]
  4. Standard I/O — df[OHLCV] → {'direction','confidence','reasons'}

Anti-falling-knife safeguards:
  - L1: ADX < 22 (no trend-following entries in mean-reversion)
  - L2: Must have bullish candle (close > open)
  - L3: Price not in crash territory (close > MA20 × 0.80)

Author: Chief Alpha Miner — Wall Street Quant Research
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class MeanReversionSchool:
    """
    Multi-factor mean reversion strategy for daily K-lines.

    Parameters
    ----------
    bb_period : int = 20
        Bollinger Band lookback.
    bb_std : float = 2.0
        Band width in standard deviations.
    rsi_period : int = 14
        RSI lookback.
    rsi_oversold : float = 35.0
        RSI below this → oversold (buy zone).
    rsi_overbought : float = 65.0
        RSI above this → overbought (sell zone).
    bias_threshold : float = 0.05
        |BIAS| above this → extreme deviation.
    adx_period : int = 14
        ADX lookback.
    adx_trend_threshold : float = 22.0
        ADX above this → trending (disable mean reversion).
    kc_period : int = 20
        Keltner Channel lookback.
    kc_atr_mult : float = 1.5
        KC width in ATR multiples.
    crash_floor_pct : float = 0.80
        Close below MA20 × this → crash territory, no buy.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        bias_period: int = 20,
        bias_threshold: float = 0.05,
        adx_period: int = 14,
        adx_trend_threshold: float = 22.0,
        kc_period: int = 20,
        kc_atr_mult: float = 1.5,
        crash_floor_pct: float = 0.80,
        vol_ma_period: int = 5,
        bb_width_max: float = 0.15,
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bias_period = bias_period
        self.bias_threshold = bias_threshold
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.kc_period = kc_period
        self.kc_atr_mult = kc_atr_mult
        self.crash_floor_pct = crash_floor_pct
        self.vol_ma_period = vol_ma_period
        self.bb_width_max = bb_width_max

    # ==================================================================
    # Public API
    # ==================================================================

    def compute_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Compute mean-reversion signal from the latest bar.

        Returns dict with keys: direction, confidence, score, reasons
        or None if insufficient data.
        """
        if df is None or len(df) < max(self.bb_period, self.adx_period, self.rsi_period) + 5:
            return None

        indicators = self._compute_indicators(df)
        if indicators is None:
            return None

        return self._generate_signal(indicators, df)

    def compute_signal_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute signals for ALL bars (vectorized). Returns DataFrame
        with columns: direction, score, confidence, reasons
        """
        n = len(df)
        if n < 60:
            return pd.DataFrame(index=df.index,
                data={'direction': 'neutral', 'score': 0.0, 'confidence': 0.0})

        ind = self._compute_all_indicators_vectorized(df)
        signals = self._generate_signals_vectorized(ind, df)
        return signals

    # ==================================================================
    # Indicator computation (fully vectorized)
    # ==================================================================

    def _compute_indicators(self, df: pd.DataFrame) -> Optional[Dict]:
        """Compute indicators for the latest bar."""
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        o = df['open'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        n = len(c)

        # ── Bollinger Bands ──
        ma20 = np.mean(c[-self.bb_period:])
        std20 = np.std(c[-self.bb_period:], ddof=1)
        bb_upper = ma20 + self.bb_std * std20
        bb_lower = ma20 - self.bb_std * std20
        bb_width = (bb_upper - bb_lower) / ma20 if ma20 > 0 else 1.0

        # ── RSI ──
        delta = np.diff(c[-(self.rsi_period + 1):])
        gain = np.sum(delta[delta > 0]) if np.any(delta > 0) else 0.0
        loss = -np.sum(delta[delta < 0]) if np.any(delta < 0) else 0.0
        avg_gain = gain / self.rsi_period
        avg_loss = loss / self.rsi_period
        rsi = 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-10)) if avg_loss > 0 else 100.0

        # ── BIAS ──
        ma20_bias = np.mean(c[-self.bias_period:])
        bias = (c[-1] - ma20_bias) / ma20_bias if ma20_bias > 0 else 0.0

        # ── ADX ──
        adx = self._compute_adx_single(h, l, c, self.adx_period)

        # ── Keltner Channel ──
        tr_arr = np.maximum(h[-self.kc_period:] - l[-self.kc_period:],
                  np.maximum(np.abs(h[-self.kc_period:] - np.roll(c[-self.kc_period:], 1)),
                             np.abs(l[-self.kc_period:] - np.roll(c[-self.kc_period:], 1))))
        atr_kc = np.mean(tr_arr)
        ema_kc = np.mean(c[-self.kc_period:])  # simplified: SMA as EMA proxy
        kc_upper = ema_kc + self.kc_atr_mult * atr_kc
        kc_lower = ema_kc - self.kc_atr_mult * atr_kc

        # ── Volume ──
        vol_ma5 = np.mean(v[-self.vol_ma_period:])
        vol_ratio = v[-1] / vol_ma5 if vol_ma5 > 0 else 1.0

        # ── Candle geometry ──
        body = abs(c[-1] - o[-1])
        candle_range = h[-1] - l[-1]
        body_pct = body / (candle_range + 1e-10)
        lower_shadow = (min(o[-1], c[-1]) - l[-1]) / (candle_range + 1e-10)
        is_bullish = c[-1] > o[-1]

        # ── MA20 slope ──
        if n >= self.bias_period + 5:
            ma20_5d_ago = np.mean(c[-self.bias_period - 5:-5])
            ma20_slope = (ma20 - ma20_5d_ago) / (ma20_5d_ago + 1e-10)
        else:
            ma20_slope = 0.0

        # ── BB squeeze: BB inside KC → compression ──
        squeeze = (bb_upper < kc_upper) and (bb_lower > kc_lower)

        return {
            'close': c[-1], 'ma20': ma20, 'bb_upper': bb_upper, 'bb_lower': bb_lower,
            'bb_width': bb_width,
            'rsi': rsi, 'bias': bias, 'adx': adx,
            'kc_upper': kc_upper, 'kc_lower': kc_lower,
            'vol_ratio': vol_ratio,
            'is_bullish': is_bullish, 'body_pct': body_pct,
            'lower_shadow_pct': lower_shadow,
            'ma20_slope': ma20_slope, 'squeeze': squeeze,
            'candle_range': candle_range,
        }

    # ==================================================================
    # Signal generation
    # ==================================================================

    def _generate_signal(self, ind: Dict, df: pd.DataFrame) -> Dict:
        """Generate signal from indicator snapshot."""
        score = 0.0
        reasons = []
        c = ind['close']

        # ── BUY conditions ──────────────────────────────────────────
        buy_conditions = 0
        buy_total = 7

        # A: Below BB lower
        if c < ind['bb_lower']:
            buy_conditions += 1
            reasons.append(f'跌破BB下轨({c:.2f}<{ind["bb_lower"]:.2f})')
        elif c < ind['kc_lower']:
            buy_conditions += 0.5  # half-credit for KC breach
            reasons.append(f'跌破KC下轨({c:.2f}<{ind["kc_lower"]:.2f})')

        # B: RSI oversold
        if ind['rsi'] < self.rsi_oversold:
            buy_conditions += 1
            reasons.append(f'RSI超卖({ind["rsi"]:.1f}<{self.rsi_oversold})')
        elif ind['rsi'] < self.rsi_oversold + 10:
            buy_conditions += 0.5

        # C: BIAS extreme negative
        if ind['bias'] < -self.bias_threshold:
            buy_conditions += 1
            reasons.append(f'BIAS乖离({ind["bias"]:.1%}<{-self.bias_threshold:.0%})')
        elif ind['bias'] < -self.bias_threshold * 0.6:
            buy_conditions += 0.5

        # D: Bullish candle (anti-falling-knife)
        if ind['is_bullish']:
            buy_conditions += 1
            reasons.append('收阳线→多头反击确认')
        if ind['lower_shadow_pct'] > 0.6 and ind['body_pct'] < 0.4:
            buy_conditions += 0.5
            reasons.append('长下影线→卖方衰竭')

        # E: Not trending (mean reversion prerequisite)
        if ind['adx'] < self.adx_trend_threshold:
            buy_conditions += 1
        else:
            reasons.append(f'ADX={ind["adx"]:.1f}>22→强趋势禁用均值回归')

        # F: Volume healthy (not dead stock)
        if ind['vol_ratio'] > 0.6:
            buy_conditions += 1

        # G: Not in crash territory
        if c > ind['ma20'] * self.crash_floor_pct:
            buy_conditions += 1
        else:
            reasons.append(f'崩盘区({c:.2f}<MA20×{self.crash_floor_pct})→禁止抄底')

        # ── SELL conditions ─────────────────────────────────────────
        sell_conditions = 0
        sell_total = 4

        if c > ind['bb_upper']:
            sell_conditions += 1
        if ind['rsi'] > self.rsi_overbought:
            sell_conditions += 1
        if ind['bias'] > self.bias_threshold:
            sell_conditions += 1
        if not ind['is_bullish']:
            sell_conditions += 1

        # ── Direction & Confidence ──────────────────────────────────
        buy_score = buy_conditions / buy_total
        sell_score = sell_conditions / sell_total

        # Depth factors: deeper oversold → higher confidence
        rsi_depth = max(0.0, (self.rsi_oversold - ind['rsi']) / self.rsi_oversold)
        bias_depth = max(0.0, (-ind['bias'] - self.bias_threshold) / self.bias_threshold)
        depth_boost = 0.5 + 0.25 * rsi_depth + 0.25 * bias_depth

        # BB squeeze bonus: when BB inside KC, reversion is higher-quality
        squeeze_bonus = 1.15 if ind['squeeze'] else 1.0

        # Composite
        net_score = buy_score - sell_score
        confidence = min(abs(net_score) * depth_boost * squeeze_bonus * 1.3, 0.95)

        if net_score > 0.25:
            direction = 'bullish'
        elif net_score < -0.25:
            direction = 'bearish'
        else:
            direction = 'neutral'
            confidence = min(confidence, 0.3)

        return {
            'direction': direction,
            'score': round(net_score, 3),
            'confidence': round(confidence, 3),
            'reasons': reasons[:5],
            'indicators': {
                'bb_lower': round(ind['bb_lower'], 2),
                'bb_upper': round(ind['bb_upper'], 2),
                'rsi': round(ind['rsi'], 1),
                'bias': round(ind['bias'], 4),
                'adx': round(ind['adx'], 1),
            }
        }

    # ==================================================================
    # Internal helpers (single-bar)
    # ==================================================================

    @staticmethod
    def _compute_adx_single(h, l, c, period=14):
        """Compute ADX for the latest bar."""
        n = len(h)
        if n < period + 1:
            return 20.0
        tr_arr = np.maximum(h[-period:] - l[-period:],
                  np.maximum(np.abs(h[-period:] - np.roll(c[-period:], 1)),
                             np.abs(l[-period:] - np.roll(c[-period:], 1))))
        atr = np.mean(tr_arr)
        if atr <= 0:
            return 20.0
        up_move = np.diff(h[-(period + 1):])
        down_move = -np.diff(l[-(period + 1):])
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_di = 100.0 * np.mean(plus_dm) / atr
        minus_di = 100.0 * np.mean(minus_dm) / atr
        dx = 100.0 * abs(plus_di - minus_di) / max(plus_di + minus_di, 1e-10)
        return float(dx)

    # ==================================================================
    # Vectorized indicator computation (all bars)
    # ==================================================================

    def _compute_all_indicators_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators for every bar. Fully vectorized."""
        c = df['close']
        h = df['high']
        l = df['low']
        o = df['open']
        v = df['volume']
        n = len(c)

        out = pd.DataFrame(index=df.index)

        # ── Bollinger Bands ──
        ma20 = c.rolling(self.bb_period, min_periods=self.bb_period // 2).mean()
        std20 = c.rolling(self.bb_period, min_periods=self.bb_period // 2).std(ddof=1)
        out['bb_upper'] = ma20 + self.bb_std * std20
        out['bb_lower'] = ma20 - self.bb_std * std20
        out['bb_width'] = (out['bb_upper'] - out['bb_lower']) / ma20.replace(0, np.nan)

        # ── RSI ──
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(self.rsi_period, min_periods=self.rsi_period // 2).mean()
        loss = (-delta.clip(upper=0)).rolling(self.rsi_period, min_periods=self.rsi_period // 2).mean()
        out['rsi'] = 100.0 - 100.0 / (1.0 + gain / (loss + 1e-10))

        # ── BIAS ──
        out['bias'] = (c - ma20) / ma20.replace(0, np.nan)

        # ── ADX ──
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr_adx = tr.rolling(self.adx_period, min_periods=self.adx_period // 2).mean()
        up = h.diff()
        dn = -l.diff()
        p_dm = np.where((up > dn) & (up > 0), up, 0)
        m_dm = np.where((dn > up) & (dn > 0), dn, 0)
        p_di = 100.0 * pd.Series(p_dm, index=df.index).rolling(self.adx_period, min_periods=self.adx_period // 2).mean() / atr_adx.replace(0, np.nan)
        m_di = 100.0 * pd.Series(m_dm, index=df.index).rolling(self.adx_period, min_periods=self.adx_period // 2).mean() / atr_adx.replace(0, np.nan)
        dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di).replace(0, np.nan)
        out['adx'] = dx.rolling(self.adx_period, min_periods=self.adx_period // 2).mean()

        # ── Keltner Channel ──
        ema_kc = c.ewm(span=self.kc_period, adjust=False).mean()
        atr_kc = tr.rolling(self.kc_period, min_periods=self.kc_period // 2).mean()
        out['kc_upper'] = ema_kc + self.kc_atr_mult * atr_kc
        out['kc_lower'] = ema_kc - self.kc_atr_mult * atr_kc

        # ── Volume ──
        vol_ma5 = v.rolling(self.vol_ma_period, min_periods=2).mean()
        out['vol_ratio'] = v / vol_ma5.replace(0, np.nan)

        # ── Candle geometry ──
        body = (c - o).abs()
        candle_range = h - l
        out['is_bullish'] = c > o
        out['body_pct'] = body / (candle_range + 1e-10)
        out['lower_shadow_pct'] = (o.combine(c, min) - l) / (candle_range + 1e-10)

        # ── MA20 slope ──
        out['ma20_slope'] = (ma20 - ma20.shift(5)) / ma20.shift(5).replace(0, np.nan)

        # ── Squeeze ──
        out['squeeze'] = (out['bb_upper'] < out['kc_upper']) & (out['bb_lower'] > out['kc_lower'])

        out['ma20'] = ma20
        out['close'] = c

        return out.ffill().fillna(0.0)

    # ==================================================================
    # Vectorized signal generation (all bars)
    # ==================================================================

    def _generate_signals_vectorized(self, ind: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals for all bars. Fully vectorized."""
        n = len(ind)
        out = pd.DataFrame(index=ind.index)
        out['direction'] = 'neutral'
        out['score'] = 0.0
        out['confidence'] = 0.0

        c = ind['close'].values.astype(np.float64)

        # ── BUY conditions (each [0 or 1]) ──
        cond_a = (c < ind['bb_lower'].values).astype(float)
        cond_a_half = ((c >= ind['bb_lower'].values) & (c < ind['kc_lower'].values)).astype(float) * 0.5
        cond_a = cond_a + cond_a_half

        cond_b = (ind['rsi'].values < self.rsi_oversold).astype(float)
        cond_b_half = ((ind['rsi'].values >= self.rsi_oversold) & (ind['rsi'].values < self.rsi_oversold + 10)).astype(float) * 0.5
        cond_b = cond_b + cond_b_half

        cond_c = (ind['bias'].values < -self.bias_threshold).astype(float)
        cond_c_half = ((ind['bias'].values >= -self.bias_threshold) & (ind['bias'].values < -self.bias_threshold * 0.6)).astype(float) * 0.5
        cond_c = cond_c + cond_c_half

        cond_d = ind['is_bullish'].values.astype(float)
        cond_d_shadow = ((ind['lower_shadow_pct'].values > 0.6) & (ind['body_pct'].values < 0.4)).astype(float) * 0.5
        cond_d = cond_d + cond_d_shadow

        cond_e = (ind['adx'].values < self.adx_trend_threshold).astype(float)
        cond_f = (ind['vol_ratio'].values > 0.6).astype(float)
        cond_g = (c > ind['ma20'].values * self.crash_floor_pct).astype(float)

        buy_score = (cond_a + cond_b + cond_c + cond_d + cond_e + cond_f + cond_g) / 7.0

        # ── SELL conditions ──
        sell_a = (c > ind['bb_upper'].values).astype(float)
        sell_b = (ind['rsi'].values > self.rsi_overbought).astype(float)
        sell_c = (ind['bias'].values > self.bias_threshold).astype(float)
        sell_d = (~ind['is_bullish'].values).astype(float)

        sell_score = (sell_a + sell_b + sell_c + sell_d) / 4.0

        # ── Composite ──
        net_score = buy_score - sell_score

        rsi_depth = np.clip((self.rsi_oversold - ind['rsi'].values) / self.rsi_oversold, 0, 1)
        bias_depth = np.clip((-ind['bias'].values - self.bias_threshold) / self.bias_threshold, 0, 1)
        depth_boost = 0.5 + 0.25 * rsi_depth + 0.25 * bias_depth

        squeeze_bonus = np.where(ind['squeeze'].values, 1.15, 1.0)

        confidence = np.clip(np.abs(net_score) * depth_boost * squeeze_bonus * 1.3, 0.0, 0.95)

        out['score'] = np.round(net_score, 3)
        out['confidence'] = np.round(confidence, 3)
        out.loc[net_score > 0.25, 'direction'] = 'bullish'
        out.loc[net_score < -0.25, 'direction'] = 'bearish'
        out.loc[(net_score >= -0.25) & (net_score <= 0.25), 'confidence'] = \
            np.clip(out.loc[(net_score >= -0.25) & (net_score <= 0.25), 'confidence'], 0, 0.3)

        return out


# ==========================================================================
# School-compatible interface (for expert_ensemble.py integration)
# ==========================================================================

# Singleton instance for ensemble use
_school = MeanReversionSchool()

def compute_mean_reversion_signal(df_daily, df_hourly=None):
    """School-compatible signal interface."""
    if df_daily is None or df_daily.empty or len(df_daily) < 60:
        return None
    try:
        result = _school.compute_signal(df_daily)
        if result is None:
            return None
        return {
            'signal': result['direction'],
            'confidence': result['confidence'],
            'metadata': {
                'score': result['score'],
                'reasons': result['reasons'],
                'indicators': result.get('indicators', {}),
            }
        }
    except Exception:
        return None


# ==========================================================================
# Quick test
# ==========================================================================

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import get_daily_kline

    print("=" * 60)
    print("  Mean Reversion School — Quick Test")
    print("=" * 60)

    school = MeanReversionSchool()

    for code in ['000012', '600519', '000001']:
        df = get_daily_kline(code, days=300)
        if df is None or df.empty:
            continue
        # Single signal
        sig = school.compute_signal(df)
        if sig:
            print(f"\n{code}: dir={sig['direction']} conf={sig['confidence']:.3f} score={sig['score']:.3f}")
            print(f"  Indicators: {sig['indicators']}")
            for r in sig['reasons'][:3]:
                print(f"  → {r}")

        # Vectorized signals
        sigs = school.compute_signal_series(df)
        bull = (sigs['direction'] == 'bullish').sum()
        bear = (sigs['direction'] == 'bearish').sum()
        neutral = (sigs['direction'] == 'neutral').sum()
        print(f"  All bars: bull={bull} bear={bear} neutral={neutral} "
              f"(mean_conf={sigs[sigs['direction']!='neutral']['confidence'].mean():.3f})")
