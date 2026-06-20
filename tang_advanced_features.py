#!/usr/bin/env python
# tang_advanced_features.py — 唐能通学派 5 个新增形态特征（宽表向量化）
"""
新增形态（纯 Pandas/Numpy，零 for 循环）：
  1. 出水芙蓉 (Water Lotus)
  2. 轻松过头 (Easy Breakthrough)
  3. 海底捞月 (Deep-V Moon Fishing)
  4. 东方红大阳升 (Eastern Red Sun Rising)
  5. 毒蜘蛛 (Poison Spider)

接口: generate_tnt_features(open_df, high_df, low_df, close_df, volume_df) -> dict[DataFrame]
"""

import numpy as np
import pandas as pd

EPS = 1e-8


def generate_tnt_features(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
) -> dict:
    """
    Generate 5 advanced Tang Nengtong features.

    All inputs: wide format (index=dates, columns=stocks).
    Returns dict with keys: water_lotus, easy_breakthrough, deep_v,
                              eastern_red_sun, poison_spider.
    """
    # ==================================================================
    # Precompute shared indicators (do once, reuse across all patterns)
    # ==================================================================
    ma5   = close_df.rolling(5,  min_periods=3).mean()
    ma10  = close_df.rolling(10, min_periods=5).mean()
    ma20  = close_df.rolling(20, min_periods=10).mean()
    ma60  = close_df.rolling(60, min_periods=30).mean()

    vol_ma5  = volume_df.rolling(5,  min_periods=2).mean()
    vol_ma20 = volume_df.rolling(20, min_periods=10).mean()

    # Candle geometry (wide-format)
    body   = (close_df - open_df).abs()
    candle_range = high_df - low_df
    upper_shadow = high_df - open_df.combine(close_df, max)
    lower_shadow = open_df.combine(close_df, min) - low_df

    # ==================================================================
    # 1. 出水芙蓉 (Water Lotus)
    # ==================================================================
    # A: 4 MA convergence within ±4%
    ma_max = ma5.combine(ma10, max).combine(ma20, max).combine(ma60, max)
    ma_min = ma5.combine(ma10, min).combine(ma20, min).combine(ma60, min)
    ma_convergence = (ma_max / ma_min.replace(0, np.nan)) - 1.0
    lotus_a = ma_convergence <= 0.04

    # B: close above ALL four MAs
    lotus_b = (close_df > ma5) & (close_df > ma10) & (close_df > ma20) & (close_df > ma60)

    # C: volume > 2x 20-day avg
    lotus_c = volume_df > vol_ma20 * 2.0

    # D: bullish body > 50% of range
    lotus_d = (close_df > open_df) & (body / (candle_range + EPS) > 0.50)

    lotus_mask = lotus_a & lotus_b & lotus_c & lotus_d
    water_lotus = np.where(lotus_mask, volume_df / vol_ma20.replace(0, np.nan), 0.0)
    water_lotus = pd.DataFrame(water_lotus, index=close_df.index, columns=close_df.columns)

    # ==================================================================
    # 2. 轻松过头 (Easy Breakthrough)
    # ==================================================================
    # A: close > 60-day high (use shift(1) to avoid self-reference)
    high_60 = high_df.shift(1).rolling(60, min_periods=30).max()
    easy_a = close_df > high_60

    # B: volume < 70% of 20-day avg (light selling pressure)
    easy_b = volume_df < vol_ma20 * 0.70

    # C: MA5 > MA20 > MA60 (bullish alignment)
    easy_c = (ma5 > ma20) & (ma20 > ma60)

    # D: small upper shadow, bullish close
    easy_d = (close_df > open_df) & (upper_shadow / (candle_range + EPS) < 0.20)

    easy_mask = easy_a & easy_b & easy_c & easy_d
    breakout_pct = (close_df - high_60) / high_60.replace(0, np.nan)
    vol_lightness = 1.0 - volume_df / vol_ma20.replace(0, np.nan)
    easy_breakthrough = np.where(easy_mask, vol_lightness * breakout_pct, 0.0)
    easy_breakthrough = pd.DataFrame(easy_breakthrough, index=close_df.index, columns=close_df.columns)

    # ==================================================================
    # 3. 海底捞月 (Deep-V Moon Fishing)
    # ==================================================================
    # A: bottom zone — close < 70% of 60-day avg
    deep_a = close_df < ma60 * 0.70

    # B: intraday new 20-day low, close bullish
    low_20 = low_df.shift(1).rolling(20, min_periods=10).min()
    deep_b = (low_df < low_20) & (close_df > open_df)

    # C: lower shadow > 50% of range AND lower shadow > 2x body
    deep_c = (lower_shadow / (candle_range + EPS) > 0.50) & (lower_shadow / (body + EPS) > 2.0)

    # D: volume > 1.3x 5-day avg
    deep_d = volume_df > vol_ma5 * 1.30

    # E: close in upper half of candle
    deep_e = close_df > (high_df + low_df) / 2.0 + 0.02 * candle_range

    deep_mask = deep_a & deep_b & deep_c & deep_d & deep_e
    shadow_score = (lower_shadow / (body + EPS)) / 4.0
    deep_v = np.where(deep_mask, shadow_score.clip(upper=1.0), 0.0)
    deep_v = pd.DataFrame(deep_v, index=close_df.index, columns=close_df.columns)

    # ==================================================================
    # 4. 东方红大阳升 (Eastern Red Sun Rising)
    # ==================================================================
    # A: prior 20-day decline > 5%
    ret_20 = close_df / close_df.shift(20).replace(0, np.nan) - 1.0
    sun_a = ret_20 < -0.05

    # B: last 3 days all bullish
    c0, c1, c2 = close_df, close_df.shift(1), close_df.shift(2)
    o0, o1, o2 = open_df, open_df.shift(1), open_df.shift(2)
    sun_b = (c0 > o0) & (c1 > o1) & (c2 > o2)

    # C: each day body > 40% of range
    body0 = (c0 - o0).abs(); body1 = (c1 - o1).abs(); body2 = (c2 - o2).abs()
    r0 = high_df - low_df
    r1 = high_df.shift(1) - low_df.shift(1)
    r2 = high_df.shift(2) - low_df.shift(2)
    sun_c = (body0 / (r0 + EPS) > 0.40) & (body1 / (r1 + EPS) > 0.40) & (body2 / (r2 + EPS) > 0.40)

    # D: volume increasing
    v0, v1, v2 = volume_df, volume_df.shift(1), volume_df.shift(2)
    sun_d = (v0 > v1) & (v1 > v2)

    # E: close rising
    sun_e = (c0 > c1) & (c1 > c2)

    sun_mask = sun_a & sun_b & sun_c & sun_d & sun_e
    eastern_red_sun = pd.DataFrame(
        np.where(sun_mask, 1.0, 0.0), index=close_df.index, columns=close_df.columns)

    # ==================================================================
    # 5. 毒蜘蛛 (Poison Spider)
    # ==================================================================
    # A: high zone — close > 120% of 60-day avg
    spider_a = close_df > ma60 * 1.20

    # B: MA death cross divergence: MA5 < MA10 < MA20 < MA60
    spider_b = (ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)

    # C: just crossed today (yesterday MA5 was above MA10)
    spider_c = ma5.shift(1) > ma10.shift(1)

    # D: volume expansion on down day
    spider_d = (volume_df > vol_ma5) & (close_df < open_df)

    spider_mask = spider_a & spider_b & spider_c & spider_d
    poison_spider = pd.DataFrame(
        np.where(spider_mask, -1.0, 0.0), index=close_df.index, columns=close_df.columns)

    return {
        'water_lotus':       water_lotus,
        'easy_breakthrough': easy_breakthrough,
        'deep_v':            deep_v,
        'eastern_red_sun':   eastern_red_sun,
        'poison_spider':     poison_spider,
    }


# ==========================================================================
# School-compatible interface (single-stock, for expert_ensemble.py)
# ==========================================================================

class TangAdvancedSchool:
    """唐能通高级形态学派 — 5个新增形态的综合评分"""

    def compute_signal(self, df: pd.DataFrame) -> dict:
        """
        Single-stock signal from 5 advanced Tang patterns.
        Input: df with open,high,low,close,volume (time-ordered).
        """
        if df is None or len(df) < 70:
            return {'direction': 'neutral', 'score': 0.0, 'confidence': 0.0, 'reasons': []}

        # Convert to wide format (1 stock)
        idx = df.index
        o = pd.DataFrame({'S': df['open'].values}, index=idx)
        h = pd.DataFrame({'S': df['high'].values}, index=idx)
        l = pd.DataFrame({'S': df['low'].values},  index=idx)
        c = pd.DataFrame({'S': df['close'].values}, index=idx)
        v = pd.DataFrame({'S': df['volume'].values}, index=idx)

        feats = generate_tnt_features(o, h, l, c, v)

        score = 0.0; reasons = []
        last = c.index[-1]

        # Water Lotus
        wl = feats['water_lotus']['S'].iloc[-1]
        if wl > 0:
            score += 0.30; reasons.append(f'出水芙蓉(强度{wl:.1f})')

        # Easy Breakthrough
        eb = feats['easy_breakthrough']['S'].iloc[-1]
        if eb > 0:
            score += 0.25; reasons.append(f'轻松过头(强度{eb:.3f})')

        # Deep-V
        dv = feats['deep_v']['S'].iloc[-1]
        if dv > 0:
            score += 0.35; reasons.append(f'海底捞月(强度{dv:.2f})')

        # Eastern Red Sun
        if feats['eastern_red_sun']['S'].iloc[-1] > 0:
            score += 0.20; reasons.append('东方红大阳升')

        # Poison Spider
        if feats['poison_spider']['S'].iloc[-1] < 0:
            score -= 0.30; reasons.append('毒蜘蛛顶部预警')

        score = np.clip(score, -1.0, 1.0)
        direction = 'bullish' if score > 0.06 else ('bearish' if score < -0.06 else 'neutral')
        conf = min(abs(score) * 1.25, 0.90)

        return {'direction': direction, 'score': round(float(score), 3),
                'confidence': round(float(conf), 3), 'reasons': reasons[:5]}


_school = TangAdvancedSchool()

def compute_tang_advanced_signal(df_daily, df_hourly=None):
    if df_daily is None or len(df_daily) < 70:
        return None
    try:
        r = _school.compute_signal(df_daily)
        return {'signal': r['direction'], 'confidence': r['confidence'],
                'metadata': {'score': r['score'], 'reasons': r['reasons']}}
    except:
        return None


# ==========================================================================
# Quick test
# ==========================================================================
if __name__ == '__main__':
    import sys; sys.path.insert(0, '.')
    from data_loader import get_daily_kline
    print("=== Tang Advanced Features Test ===")
    for code in ['000012', '600519']:
        df = get_daily_kline(code, days=300)
        if df is None or not df.empty:
            sig = _school.compute_signal(df)
            if sig:
                print(f"{code}: dir={sig['direction']} conf={sig['confidence']:.3f} "
                      f"reasons={sig['reasons'][:3]}")
