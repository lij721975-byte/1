#!/usr/bin/env python
# volume_profile_school.py — Volume Profile / POC based support-resistance school
"""
Volume Profile School: approximate daily-bar volume distribution to find POC.

- Price breaking above POC → bullish (cleared resistance)
- Price bouncing off POC from above → bullish (support confirmed)
- Price rejected at POC from below → bearish

Pure numpy/pandas, fully vectorized. Standard school interface.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


class VolumeProfileSchool:
    def __init__(self, window: int = 120, num_bins: int = 30):
        self.window = window
        self.num_bins = num_bins

    def _compute_poc(self, df: pd.DataFrame) -> float:
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        c = df['close'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        n = len(h)

        price_min, price_max = np.min(l), np.max(h)
        if price_max <= price_min:
            return c[-1]

        bins = np.linspace(price_min, price_max, self.num_bins + 1)
        vol_profile = np.zeros(self.num_bins, dtype=np.float64)

        for i in range(n):
            typical = (h[i] + l[i] + c[i]) / 3.0
            # Distribute volume across bins the bar spans
            bar_low = min(l[i], typical)
            bar_high = max(h[i], typical)
            if bar_high <= bar_low:
                bar_high = bar_low + 0.01
            # Fraction of each bin covered by this bar
            for j in range(self.num_bins):
                bin_low = bins[j]
                bin_high = bins[j + 1]
                overlap = max(0.0, min(bar_high, bin_high) - max(bar_low, bin_low))
                fraction = overlap / (bar_high - bar_low)
                vol_profile[j] += v[i] * fraction

        poc_idx = np.argmax(vol_profile)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0

        # Value Area (70%)
        total_vol = vol_profile.sum()
        target = total_vol * 0.70
        l_idx = r_idx = poc_idx
        cum = vol_profile[poc_idx]
        while cum < target and (l_idx > 0 or r_idx < self.num_bins - 1):
            vl = vol_profile[l_idx - 1] if l_idx > 0 else 0
            vr = vol_profile[r_idx + 1] if r_idx < self.num_bins - 1 else 0
            if vl >= vr:
                cum += vl; l_idx -= 1
            else:
                cum += vr; r_idx += 1

        return float(poc), float(bins[l_idx]), float(bins[r_idx])

    def compute_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        if df is None or len(df) < self.window:
            return None
        return self._generate(df)

    def compute_signal_series(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        out = pd.DataFrame(index=df.index, columns=['direction','confidence','score'])
        out['direction'] = 'neutral'; out['confidence'] = 0.0; out['score'] = 0.0
        if n < self.window: return out
        for i in range(self.window, n):
            r = self._generate(df.iloc[:i+1])
            if r:
                out.iloc[i, 0] = r['direction']
                out.iloc[i, 1] = r['confidence']
        return out

    def _generate(self, df: pd.DataFrame) -> Dict:
        c = df['close'].values.astype(np.float64)
        h = df['high'].values.astype(np.float64)
        l = df['low'].values.astype(np.float64)
        v = df['volume'].values.astype(np.float64)
        o = df['open'].values.astype(np.float64)
        n = len(c)

        sub = df.iloc[-self.window:]
        poc, val, vah = self._compute_poc(sub)

        score = 0.0; reasons = []
        cp = c[-1]

        # ── Price relative to POC ──
        poc_dist = (cp - poc) / (poc + 1e-10)

        # ── Breakout above POC ──
        if cp > poc and cp > o[-1] and v[-1] > np.mean(v[-5:]):
            score += 0.35; reasons.append(f'放量突破POC({poc:.2f})')
        elif cp > poc and cp > o[-1]:
            score += 0.20; reasons.append(f'突破POC({poc:.2f})')

        # ── Bounce off POC support ──
        yesterday_close = c[-2] if n >= 2 else cp
        if yesterday_close > poc and l[-1] <= poc * 1.01 and c[-1] > poc:
            score += 0.30; reasons.append(f'回踩POC({poc:.2f})获支撑')

        # ── POC rejection ──
        if cp < poc and h[-1] >= poc * 0.99 and c[-1] < o[-1]:
            score -= 0.25; reasons.append(f'POC({poc:.2f})受阻回落')

        # ── Value Area context ──
        if cp > vah:
            score += 0.10; reasons.append('价格在VA上方')
        elif cp < val:
            score -= 0.10; reasons.append('价格在VA下方')
        else:
            reasons.append('价格在VA内部')

        # ── Volume confirmation ──
        vol_ratio = v[-1] / (np.mean(v[-5:]) + 1e-10)
        if vol_ratio > 1.5 and score > 0:
            score += 0.10; reasons.append(f'放量{vol_ratio:.1f}x确认')

        score = np.clip(score, -1.0, 1.0)
        direction = 'bullish' if score > 0.06 else ('bearish' if score < -0.06 else 'neutral')
        conf = min(abs(score) * 1.2, 0.88)

        return {'direction': direction, 'score': round(float(score), 3),
                'confidence': round(float(conf), 3), 'reasons': reasons[:4]}


# ── School-compatible interface ──
_school = VolumeProfileSchool()

def compute_vp_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 120: return None
    try:
        r = _school.compute_signal(df_daily)
        return {'signal': r['direction'], 'confidence': r['confidence'],
                'metadata': {'score': r['score'], 'reasons': r['reasons']}} if r else None
    except: return None
