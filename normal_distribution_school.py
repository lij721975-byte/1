#!/usr/bin/env python
# normal_distribution_school.py — Z-Score based mean reversion + momentum school
"""
Normal Distribution School: log-return Z-Score analysis.

- Oversold (< -2σ) + stabilization → bullish (mean reversion)
- Breakout (> +2σ) + volume confirmation → bullish (momentum)
- Overbought (> +2σ) without volume → bearish

Pure numpy/pandas, fully vectorized. Standard school interface.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class NormalDistributionSchool:
    def __init__(self, window: int = 60, z_extreme: float = 2.0,
                 z_oversold: float = -2.0, z_overbought: float = 2.0):
        self.window = window
        self.z_extreme = z_extreme
        self.z_oversold = z_oversold
        self.z_overbought = z_overbought

    def compute_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < self.window + 5:
            return None
        return self._generate(df)

    def compute_signal_series(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame(index=df.index, columns=['direction','confidence','score'])
        out['direction'] = 'neutral'; out['confidence'] = 0.0; out['score'] = 0.0
        if n < self.window + 5: return out
        for i in range(self.window + 5, n):
            r = self._generate(df.iloc[:i+1])
            if r:
                out.iloc[i, 0] = r['direction']
                out.iloc[i, 1] = r['confidence']
        return out

    def _generate(self, df: pd.DataFrame) -> Dict:
        c = df['close'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)

        # Log returns
        log_ret = np.diff(np.log(c))[-(self.window):]
        mu = np.mean(log_ret)
        sigma = np.std(log_ret, ddof=1)
        if sigma <= 0:
            sigma = 0.001

        # Current Z-Score
        latest_ret = np.log(c[-1] / c[-2])
        z_score = (latest_ret - mu) / sigma

        # Price Z-Score (for mean reversion)
        price_z = (c[-1] - np.mean(c[-self.window:])) / (np.std(c[-self.window:], ddof=1) + 1e-10)

        # Volume conditions
        vol_ma5 = np.mean(v[-5:])
        vol_ratio = v[-1] / (vol_ma5 + 1e-10)

        score = 0.0; reasons = []

        # ── Oversold mean reversion ──
        if price_z < self.z_oversold:
            # Check stabilization: today's low >= yesterday's low (stopped falling)
            stabilized = l[-1] >= l[-2] or c[-1] > c[-2]
            if stabilized:
                score += 0.50; reasons.append(f'超卖Z={price_z:.1f}且企稳→均值回归')
            else:
                score += 0.20; reasons.append(f'超卖Z={price_z:.1f}但未企稳')

        # ── Overbought reversal ──
        if price_z > self.z_overbought:
            if vol_ratio < 0.7:
                score -= 0.30; reasons.append(f'超买Z={price_z:.1f}缩量→衰竭')

        # ── Momentum breakout ──
        if z_score > self.z_extreme and vol_ratio > 1.3:
            score += 0.40; reasons.append(f'动量突破Z={z_score:.1f}放量{vol_ratio:.1f}x')

        # ── Candlestick confirmation ──
        if c[-1] > c[-2]:
            score += 0.10

        # ── MA20 trend context ──
        ma20 = np.mean(c[-20:]) if len(c) >= 20 else c[-1]
        if c[-1] > ma20:
            score += 0.05; reasons.append('价格在MA20上方')
        else:
            score -= 0.05

        score = np.clip(score, -1.0, 1.0)
        direction = 'bullish' if score > 0.08 else ('bearish' if score < -0.08 else 'neutral')
        conf = min(abs(score) * 1.3, 0.90)

        return {'direction': direction, 'score': round(float(score), 3),
                'confidence': round(float(conf), 3), 'reasons': reasons[:4]}


# ── School-compatible interface ──
_school = NormalDistributionSchool()

def compute_nd_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 65: return None
    try:
        r = _school.compute_signal(df_daily)
        return {'signal': r['direction'], 'confidence': r['confidence'],
                'metadata': {'score': r['score'], 'reasons': r['reasons']}} if r else None
    except: return None
