# advanced_indicators.py
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import talib
from collections import OrderedDict


def _safe(v, default=0.0):
    """安全取数值，处理None和NaN"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


# ===================== 1. 缠论（完整版 - 缠中说禅108课） =====================

def _chan_inclusion_direction(prev_prev, prev):
    """判断包含处理方向:向上=1，向下=-1（前两根K线收盘比较）"""
    if prev['close'] >= prev_prev['close']:
        return 1  # 向上
    return -1  # 向下


def chanlun_process_inclusion(df, max_iterations=5):
    """
    缠论K线包含处理（方向性处理，递归直到无包含关系）
    向上处理:高高(高点max)，低高(低点max)
    向下处理:低低(低点min)，高低(高点min)
    """
    if len(df) < 2:
        return df.copy()
    for _ in range(max_iterations):
        rows = []
        changed = False
        for i in range(len(df)):
            curr = df.iloc[i].copy()
            if not rows:
                rows.append(curr)
                continue
            prev = rows[-1]
            is_contained = ((curr['high'] <= prev['high'] and curr['low'] >= prev['low']) or
                            (curr['high'] >= prev['high'] and curr['low'] <= prev['low']))
            if is_contained:
                changed = True
                if len(rows) >= 2:
                    direction = _chan_inclusion_direction(rows[-2], prev)
                else:
                    direction = 1 if prev['close'] >= prev['open'] else -1
                if direction == 1:
                    prev['high'] = max(prev['high'], curr['high'])
                    prev['low'] = max(prev['low'], curr['low'])
                else:
                    prev['low'] = min(prev['low'], curr['low'])
                    prev['high'] = min(prev['high'], curr['high'])
                prev['close'] = curr['close']
                prev['volume'] = prev.get('volume', 0) + curr.get('volume', 0)
            else:
                rows.append(curr)
        df = pd.DataFrame(rows)
        if not changed or len(df) < 2:
            break
    return df


def _is_up_fractal(highs, i):
    """顶分型:中间K线高点最高"""
    if i < 1 or i >= len(highs) - 1:
        return False
    return highs[i] > highs[i - 1] and highs[i] > highs[i + 1]


def _is_down_fractal(lows, i):
    """底分型:中间K线低点最低"""
    if i < 1 or i >= len(lows) - 1:
        return False
    return lows[i] < lows[i - 1] and lows[i] < lows[i + 1]


def _count_k_between(df, idx1, idx2):
    """计算两个索引之间的K线数量（不含两端）"""
    if idx1 is None or idx2 is None:
        return 0
    return max(0, abs(idx2 - idx1) - 1)


def chanlun_find_fractals(df, min_k_between=3):
    """
    找出所有顶底分型（新笔定义:相邻同向分型间至少min_k_between根K线）
    返回 (顶分型列表, 底分型列表)，每项为 (index, price, date)
    """
    highs = df['high'].values
    lows = df['low'].values
    dates = df.index
    raw_tops, raw_bottoms = [], []
    for i in range(1, len(df) - 1):
        if _is_up_fractal(highs, i):
            raw_tops.append((i, highs[i], dates[i]))
        if _is_down_fractal(lows, i):
            raw_bottoms.append((i, lows[i], dates[i]))

    def _filter(raw_list, is_top):
        if not raw_list:
            return []
        filtered = [raw_list[0]]
        for j in range(1, len(raw_list)):
            last, curr = filtered[-1], raw_list[j]
            if _count_k_between(df, last[0], curr[0]) < min_k_between:
                if (is_top and curr[1] > last[1]) or (not is_top and curr[1] < last[1]):
                    filtered[-1] = curr
            else:
                filtered.append(curr)
        return filtered

    return _filter(raw_tops, is_top=True), _filter(raw_bottoms, is_top=False)


def chanlun_build_bi(df, tops, bottoms):
    """
    从分型构建笔（新笔定义:顶底交替）
    返回 [(date, price, type, idx), ...]  type='top'/'bottom'
    """
    events = [(t[0], 'top', t[2], t[1]) for t in tops] + \
             [(b[0], 'bottom', b[2], b[1]) for b in bottoms]
    events.sort(key=lambda x: x[0])
    if not events:
        return []
    bi_points = [events[0]]
    for i in range(1, len(events)):
        last, curr = bi_points[-1], events[i]
        if curr[1] != last[1]:
            if _count_k_between(df, last[0], curr[0]) >= 2:
                bi_points.append(curr)
            elif (curr[1] == 'top' and curr[3] > last[3]) or \
                 (curr[1] == 'bottom' and curr[3] < last[3]):
                bi_points[-1] = curr
    return bi_points


def chanlun_build_segments(bi_points):
    """
    从笔构建线段（含线段破坏规则 - 缠中说禅第65-72课）

    标准线段破坏:
    - 向上线段被向下笔破坏: 该向下笔的终点必须低于前一笔的终点
      且与该向下笔同向的特征序列出现底分型
    - 向下线段被向上笔破坏: 该向上笔的终点必须高于前一笔的终点
      且与该向上笔同向的特征序列出现顶分型

    特征序列: 向上线段的特征序列是线段内各向下笔
              向下线段的特征序列是线段内各向上笔
    """
    if len(bi_points) < 3:
        return []
    segments = []
    seg_start = 0

    # 特征序列分析:对向上线段，取向下笔为特征序列；反之亦然
    def _feature_sequence(bi_slice, seg_type):
        """提取特征序列笔的端点价格"""
        feats = []
        for k in range(len(bi_slice)):
            bi = bi_slice[k]
            if seg_type == 'up' and bi[1] == 'bottom':
                feats.append((k, bi[3], 'low'))
            elif seg_type == 'down' and bi[1] == 'top':
                feats.append((k, bi[3], 'high'))
        return feats

    def _has_feature_fractal(feats, fractal_type):
        """检测特征序列是否出现分型（3元素形成顶/底分型）"""
        if len(feats) < 3:
            return False
        # 取最近3个特征序列点
        f0, f1, f2 = feats[-3][1], feats[-2][1], feats[-1][1]
        if fractal_type == 'top':
            return f1 > f0 and f1 > f2  # 顶分型
        return f1 < f0 and f1 < f2  # 底分型

    i = 2
    while i < len(bi_points):
        p0, p1, p2 = bi_points[i-2], bi_points[i-1], bi_points[i]

        # 向下线段检测 (top→bottom→top 结构)
        if p0[1] == 'top' and p1[1] == 'bottom' and p2[1] == 'top':
            # 标准线段破坏: p2突破p0高点→向上笔破坏向下线段
            if p2[3] > p0[3]:
                # 检查特征序列确认
                seg_bi = bi_points[seg_start:i]
                feats = _feature_sequence(seg_bi, 'down')
                if _has_feature_fractal(feats, 'top') or len(feats) < 3:
                    segments.append({'type': 'down', 'start': bi_points[seg_start],
                                   'end': p1, 'bi_count': i - seg_start,
                                   'destruction': '标准破坏(新高突破)'})
                    seg_start = i - 1
                    i = seg_start + 2
                    continue
            # 背驰型完成:未新高，但特征序列出现顶分型
            elif p2[3] <= p0[3]:
                seg_bi = bi_points[seg_start:i]
                feats = _feature_sequence(seg_bi, 'down')
                if _has_feature_fractal(feats, 'top') and len(seg_bi) >= 5:
                    segments.append({'type': 'down', 'start': bi_points[seg_start],
                                   'end': p1, 'bi_count': i - seg_start,
                                   'destruction': '背驰完成(特征序列顶分型)'})
                    seg_start = i - 1
                    i = seg_start + 2
                    continue

        # 向上线段检测 (bottom→top→bottom 结构)
        elif p0[1] == 'bottom' and p1[1] == 'top' and p2[1] == 'bottom':
            if p2[3] < p0[3]:  # p2跌破p0低点→向下笔破坏向上线段
                seg_bi = bi_points[seg_start:i]
                feats = _feature_sequence(seg_bi, 'up')
                if _has_feature_fractal(feats, 'bottom') or len(feats) < 3:
                    segments.append({'type': 'up', 'start': bi_points[seg_start],
                                   'end': p1, 'bi_count': i - seg_start,
                                   'destruction': '标准破坏(新低突破)'})
                    seg_start = i - 1
                    i = seg_start + 2
                    continue
            elif p2[3] >= p0[3]:
                seg_bi = bi_points[seg_start:i]
                feats = _feature_sequence(seg_bi, 'up')
                if _has_feature_fractal(feats, 'bottom') and len(seg_bi) >= 5:
                    segments.append({'type': 'up', 'start': bi_points[seg_start],
                                   'end': p1, 'bi_count': i - seg_start,
                                   'destruction': '背驰完成(特征序列底分型)'})
                    seg_start = i - 1
                    i = seg_start + 2
                    continue
        i += 1

    # 处理未完成的最后一段
    if seg_start < len(bi_points) - 1:
        direction = 'up' if bi_points[-1][1] == 'top' else 'down'
        segments.append({'type': direction, 'start': bi_points[seg_start],
                        'end': bi_points[-1], 'bi_count': len(bi_points) - seg_start,
                        'unfinished': True, 'destruction': '未完成'})

    return segments


def chanlun_zhongshu_expansion(zhongshu_list, segments):
    """
    中枢扩展/级别升级（缠中说禅第61-64课）

    当两个同级别中枢发生重叠时（ZG1 > ZD2），中枢级别升级
    返回升级后的中枢列表 + 原始中枢
    """
    if len(zhongshu_list) < 2:
        return {'expanded': False, 'upgraded_zhongshu': None, 'original': zhongshu_list}

    zs1, zs2 = zhongshu_list[-2], zhongshu_list[-1]
    expanded = False
    upgraded = None

    # 中枢重叠检测
    if zs2['ZD'] < zs1['ZG'] and zs1['ZD'] < zs2['ZG']:
        expanded = True
        # 升级中枢取两中枢的ZG和ZD
        new_ZD = max(zs1['ZD'], zs2['ZD'])
        new_ZG = min(zs1['ZG'], zs2['ZG'])
        new_ZZ = (new_ZD + new_ZG) / 2

        # 检查升级后的中枢是否与之前的更高级别中枢重叠
        # 如果重叠，继续升级
        outer_ZD = min(zs1['ZD'], zs2['ZD'])
        outer_ZG = max(zs1['ZG'], zs2['ZG'])

        upgraded = {
            'ZD': new_ZD,
            'ZG': new_ZG,
            'ZZ': new_ZZ,
            'range': new_ZG - new_ZD,
            'outer_ZD': outer_ZD,
            'outer_ZG': outer_ZG,
            'level': '日线次级别→日线级别',
            'merged_from': [zs1, zs2],
        }

    return {
        'expanded': expanded,
        'upgraded_zhongshu': upgraded,
        'original': zhongshu_list,
    }


def chanlun_xiao_zhuan_da(df_daily, df_60min=None):
    """
    小转大检测（缠中说禅 - 小级别反转引发大级别反转）

    检测信号:
    1. 60分钟出现底/顶背驰且背驰力度很强(>0.7)
    2. 日线尚未出现背驰信号（仍处于趋势中）
    3. 60分钟出现反转K线形态（底分型/顶分型）
    → 可能小转大:小级别反转引发大级别转折

    返回小转大预警
    """
    if df_60min is None or len(df_60min) < 60:
        return {'xzd_active': False, 'xzd_direction': None, 'xzd_confidence': 0,
                'xzd_reasons': [], 'xzd_hourly_div': '', 'xzd_hourly_div_strength': 0,
                'xzd_daily_div': '', 'xzd_hourly_stroke': ''}

    # 60分钟分析
    df_60_proc = chanlun_process_inclusion(df_60min)
    tops_60, bottoms_60 = chanlun_find_fractals(df_60_proc)
    bi_60 = chanlun_build_bi(df_60_proc, tops_60, bottoms_60)
    div_60 = chanlun_macd_divergence(df_60_proc, bi_60)
    stroke_60 = chanlun_stroke_state(df_60min)

    # 日线分析
    df_d_proc = chanlun_process_inclusion(df_daily)
    tops_d, bottoms_d = chanlun_find_fractals(df_d_proc)
    bi_d = chanlun_build_bi(df_d_proc, tops_d, bottoms_d)
    div_d = chanlun_macd_divergence(df_d_proc, bi_d)

    xzd_active = False
    xzd_direction = None
    xzd_confidence = 0.0
    reasons = []

    div_type_60 = div_60.get('type', '')
    div_strength_60 = div_60.get('strength', 0)
    div_type_d = div_d.get('type', '')
    stroke_60_state = stroke_60.get('state', '')

    # 小转大条件1: 60分钟强背驰 + 日线无背驰
    if div_type_60 == '底背驰' and div_strength_60 > 0.65 and div_type_d != '底背驰':
        xzd_active = True
        xzd_direction = 'bullish'
        xzd_confidence = min(0.85, div_strength_60 + 0.1)
        reasons.append(f'60分钟强底背驰({div_strength_60:.0%})但日线无背驰→小转大看多')

    elif div_type_60 == '顶背驰' and div_strength_60 > 0.65 and div_type_d != '顶背驰':
        xzd_active = True
        xzd_direction = 'bearish'
        xzd_confidence = min(0.85, div_strength_60 + 0.1)
        reasons.append(f'60分钟强顶背驰({div_strength_60:.0%})但日线无背驰→小转大看空')

    # 小转大条件2: 60分钟反转分型确认 + 日线趋势强势但60分钟已反向
    if stroke_60_state in ['(1,1)'] and div_type_d == '顶背驰':
        if not xzd_active:
            xzd_active = True
            xzd_direction = 'bullish'
            xzd_confidence = 0.6
        reasons.append('60分钟向上笔确认+日线顶背驰→可能小转大反转向上')
    elif stroke_60_state in ['(-1,1)'] and div_type_d == '底背驰':
        if not xzd_active:
            xzd_active = True
            xzd_direction = 'bearish'
            xzd_confidence = 0.6
        reasons.append('60分钟向下笔确认+日线底背驰→可能小转大反转向下')

    return {
        'xzd_active': xzd_active,
        'xzd_direction': xzd_direction,
        'xzd_confidence': round(xzd_confidence, 2),
        'xzd_reasons': reasons,
        'xzd_hourly_div': div_type_60,
        'xzd_hourly_div_strength': round(div_strength_60, 2),
        'xzd_daily_div': div_type_d,
        'xzd_hourly_stroke': stroke_60_state,
    }


def chanlun_find_all_zhongshu(segments):
    """
    从线段中找出所有中枢（至少3段重叠）
    返回 [{'ZD': float, 'ZG': float, 'ZZ': float, 'range': float, 'strength': str}]
    """
    if len(segments) < 3:
        return []
    zhongshu_list = []
    for i in range(len(segments) - 2):
        s0, s1, s2 = segments[i], segments[i+1], segments[i+2]
        all_lows, all_highs = [], []
        for s in [s0, s1, s2]:
            all_lows.append(min(s['start'][3], s['end'][3]))
            all_highs.append(max(s['start'][3], s['end'][3]))
        ZD, ZG = max(all_lows), min(all_highs)
        if ZD < ZG:
            rng = ZG - ZD
            strength = '强' if rng < ZG * 0.03 else ('弱' if rng > ZG * 0.08 else '中')
            zhongshu_list.append({
                'ZD': ZD, 'ZG': ZG, 'ZZ': (ZD + ZG) / 2,
                'range': rng, 'strength': strength,
                'start_seg': i, 'end_seg': i + 2,
            })
    return zhongshu_list


def chanlun_stroke_state(df):
    """
    笔状态机 (1,1)/(1,0)/(-1,1)/(-1,0) — 土匪版定义
    direction: 1=向上笔, -1=向下笔
    state: 1=有效分型确认, 0=无新分型(延伸/包含合并中)
    """
    if len(df) < 3:
        return {'state': '(-1,0)', 'direction': -1, 'last_fractal': None,
                'k_after_fractal': 0, 'state_label': '无方向(数据不足)', 'is_confirmed': False}

    df_proc = chanlun_process_inclusion(df)
    highs = df_proc['high'].values
    lows = df_proc['low'].values

    last_fractal_idx, last_fractal_type = None, None
    for i in range(len(df_proc) - 2, 0, -1):
        if _is_up_fractal(highs, i):
            last_fractal_idx, last_fractal_type = i, 'top'
            break
        if _is_down_fractal(lows, i):
            last_fractal_idx, last_fractal_type = i, 'bottom'
            break

    if last_fractal_type is None:
        recent_close = df_proc['close'].iloc[-1]
        direction = 1 if recent_close >= df_proc['open'].iloc[-1] else -1
        return {'state': f"({direction},0)", 'direction': direction,
                'last_fractal': None, 'last_fractal_date': None,
                'k_after_fractal': 0, 'state_label': f"{'向上' if direction==1 else '向下'}笔方向(无分型)", 'is_confirmed': False}

    k_after = len(df_proc) - last_fractal_idx - 1
    if last_fractal_type == 'bottom':
        final_direction, final_state = 1, (1 if k_after >= 2 else 0)
    else:
        final_direction, final_state = -1, (1 if k_after >= 2 else 0)

    labels = {
        '(1,1)': '向上笔确认(底分型有效)→看多',
        '(1,0)': '向上笔延伸(无新分型)→偏多',
        '(-1,1)': '向下笔确认(顶分型有效)→看空',
        '(-1,0)': '向下笔延伸(无新分型)→偏空',
    }
    state_str = f"({final_direction},{final_state})"
    return {
        'state': state_str, 'direction': final_direction,
        'last_fractal': last_fractal_type,
        'last_fractal_date': str(df_proc.index[last_fractal_idx]),
        'k_after_fractal': k_after,
        'state_label': labels.get(state_str, '未知'),
        'is_confirmed': final_state == 1,
    }


def chanlun_macd_divergence(df, bi_points):
    """
    MACD面积背驰判断（缠论核心:比较相邻同向笔的MACD柱面积）
    返回 {'type': '顶背驰'/'底背驰'/None, 'strength': 0~1, 'details': str}
    """
    if len(bi_points) < 4:
        return {'type': None, 'strength': 0, 'details': '笔数不足'}
    close = df['close'].values
    if len(close) < 26:
        return {'type': None, 'strength': 0, 'details': '数据不足'}
    _, _, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    def _macd_area(start_idx, end_idx):
        if start_idx < 0 or end_idx >= len(macd_hist) or end_idx <= start_idx:
            return 0
        seg = macd_hist[start_idx:end_idx + 1]
        seg = seg[~np.isnan(seg)]
        return np.sum(np.abs(seg))

    recent = bi_points[-4:]
    p0, p1, p2, p3 = recent[0], recent[1], recent[2], recent[3]
    if p2[1] == 'top' and p3[1] == 'bottom':
        area1 = _macd_area(p1[0], p2[0])
        area2 = _macd_area(p2[0], p3[0])
        if p3[3] < p1[3] and area2 < area1 * 0.85:
            strength = min(1.0, (area1 - area2) / (area1 + 1e-10))
            return {'type': '底背驰', 'strength': round(strength, 3),
                    'details': f'价格新低({p1[3]:.2f}→{p3[3]:.2f}) MACD面积缩小({area1:.0f}→{area2:.0f})'}
    elif p2[1] == 'bottom' and p3[1] == 'top':
        area1 = _macd_area(p1[0], p2[0])
        area2 = _macd_area(p2[0], p3[0])
        if p3[3] > p1[3] and area2 < area1 * 0.85:
            strength = min(1.0, (area1 - area2) / (area1 + 1e-10))
            return {'type': '顶背驰', 'strength': round(strength, 3),
                    'details': f'价格新高({p1[3]:.2f}→{p3[3]:.2f}) MACD面积缩小({area1:.0f}→{area2:.0f})'}
    return {'type': None, 'strength': 0, 'details': '无背驰信号'}


def chanlun_center_monitor(df, zhongshu_list):
    """
    中枢震荡监视器Zn
    Z = (ZG + ZD) / 2, Zn = (price - Z) / ((ZG - ZD) / 2)
    返回 {'zn_value': float, 'position': str, 'pattern': str}
    """
    if not zhongshu_list:
        return {'zn_value': 0, 'position': '无中枢', 'zn_series': [], 'pattern': '无'}
    zs = zhongshu_list[-1]
    ZD, ZG, ZZ = zs['ZD'], zs['ZG'], zs['ZZ']
    current_price = df['close'].iloc[-1]
    half_range = (ZG - ZD) / 2
    if half_range < 1e-10:
        half_range = 0.01

    zn_series = []
    for i in range(max(0, len(df) - 20), len(df)):
        zn = (df['close'].iloc[i] - ZZ) / half_range
        zn_series.append({'date': str(df.index[i]), 'zn': round(zn, 3)})
    current_zn = (current_price - ZZ) / half_range

    if current_zn > 2.0:
        position = '中枢上方远离(超买)'
    elif current_zn > 1.0:
        position = '中枢上方'
    elif current_zn > 0:
        position = '中枢中轴上方'
    elif current_zn > -1.0:
        position = '中枢中轴下方'
    elif current_zn > -2.0:
        position = '中枢下方'
    else:
        position = '中枢下方远离(超卖)'

    zn_vals = [z['zn'] for z in zn_series]
    pattern = '震荡'
    if len(zn_vals) >= 5:
        r = zn_vals[-5:]
        if all(z > 1.5 for z in r):
            pattern = '持续偏上(可能三买形成)'
        elif all(z < -1.5 for z in r):
            pattern = '持续偏下(可能三卖形成)'
        elif max(r) - min(r) < 0.5:
            pattern = '窄幅收敛(即将变盘)'
        elif max(r) > 1.5 and min(r) < -1.5:
            pattern = '宽幅震荡'

    return {'zn_value': round(current_zn, 3), 'position': position,
            'zn_series': zn_series[-10:], 'pattern': pattern,
            'center_zz': round(ZZ, 2), 'center_range': round(ZG - ZD, 2)}


def chanlun_wolf_defense(df):
    """
    防狼术:MACD黄白线在0轴以下→空头主导，不入场
    返回 {'signal': 'safe'/'danger', 'macd_position': str, 'days_below_zero': int}
    """
    if len(df) < 26:
        return {'signal': 'unknown', 'macd_position': '数据不足', 'macd_dif': 0,
                'macd_dea': 0, 'days_below_zero': 0, 'warning': ''}
    close = df['close'].values
    macd_line, signal_line, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    current_dif = macd_line[-1]
    current_dea = signal_line[-1]
    below_zero = int(np.sum((macd_line[-20:] < 0) & (~np.isnan(macd_line[-20:]))))

    if current_dif > 0 and current_dea > 0:
        macd_pos, signal = '双线0轴上方(安全区)', 'safe'
    elif current_dif > 0 and current_dea < 0:
        macd_pos, signal = 'DIF上穿0轴(转好)', 'safe'
    elif current_dif < 0 and current_dea > 0:
        macd_pos, signal = 'DIF下穿0轴(转坏)', 'danger'
    else:
        macd_pos, signal = '双线0轴下方(空头主导)', 'danger'

    return {'signal': signal, 'macd_position': macd_pos,
            'macd_dif': round(float(current_dif), 4) if not np.isnan(current_dif) else 0,
            'macd_dea': round(float(current_dea), 4) if not np.isnan(current_dea) else 0,
            'days_below_zero': below_zero,
            'warning': '防狼术:不入场' if signal == 'danger' else ''}


def chanlun_trend_type(bi_points, zhongshu_list):
    """判断走势类型:盘整(1中枢) vs 趋势(2+同向中枢)"""
    n = len(zhongshu_list)
    if n == 0:
        if len(bi_points) >= 2:
            d = 'up' if bi_points[-1][1] == 'top' else 'down'
            return {'type': f'无中枢({ "上涨" if d=="up" else "下跌" }中)', 'center_count': 0, 'direction': d}
        return {'type': '无法判断', 'center_count': 0, 'direction': 'unknown'}
    if n == 1:
        return {'type': '盘整', 'center_count': 1, 'direction': '震荡'}
    z1, z2 = zhongshu_list[-2], zhongshu_list[-1]
    if z2['ZD'] > z1['ZG']:
        return {'type': '上涨趋势', 'center_count': n, 'direction': 'up'}
    elif z2['ZG'] < z1['ZD']:
        return {'type': '下跌趋势', 'center_count': n, 'direction': 'down'}
    return {'type': '中枢扩张', 'center_count': n, 'direction': '震荡'}


def _chanlun_buy_sell_points(current_price, zhongshu_list, bi_points, divergence, df):
    """
    三类买卖点识别:
    一买/一卖:趋势背驰后第一个反向分型
    二买/二卖:回踩/反弹不破前极值
    三买/三卖:突破中枢后回踩确认不破/跌破中枢后反弹确认无力
    """
    buy_points, sell_points = [], []
    if not zhongshu_list:
        return buy_points, sell_points
    zs = zhongshu_list[-1]
    ZD, ZG, ZZ = zs['ZD'], zs['ZG'], zs['ZZ']
    div_type = divergence.get('type') if divergence else None
    div_strength = divergence.get('strength', 0) if divergence else 0

    if len(bi_points) >= 2:
        last_bi = bi_points[-1]
        # 一买:底背驰+价格在中枢下方
        if div_type == '底背驰' and div_strength > 0.3 and current_price < ZD:
            buy_points.append({'type': '一买', 'level': 'strong' if div_strength > 0.6 else 'moderate',
                'desc': f'底背驰(强度{div_strength:.0%})中枢下方一买', 'action': '分批建仓，止损最低点-3%'})
        # 二买:回踩不破前低
        if last_bi[1] == 'bottom' and len(bi_points) >= 4:
            prev_bot = bi_points[-3]
            if prev_bot[1] == 'bottom' and last_bi[3] > prev_bot[3]:
                buy_points.append({'type': '二买', 'level': 'moderate',
                    'desc': f'回踩不破前低({prev_bot[3]:.2f})', 'action': '加仓，止损二买低点'})
        # 三买:突破中枢上沿回踩不破
        if current_price > ZG and df['high'].iloc[-10:].max() > ZG * 1.02:
            buy_points.append({'type': '三买', 'level': 'aggressive',
                'desc': f'突破中枢{ZG:.2f}回踩确认', 'action': '轻仓追涨，止损ZG-2%'})
        # 一卖
        if div_type == '顶背驰' and div_strength > 0.3 and current_price > ZG:
            sell_points.append({'type': '一卖', 'level': 'strong' if div_strength > 0.6 else 'moderate',
                'desc': f'顶背驰(强度{div_strength:.0%})中枢上方一卖', 'action': '减仓或清仓'})
        # 二卖
        if last_bi[1] == 'top' and len(bi_points) >= 4:
            prev_top = bi_points[-3]
            if prev_top[1] == 'top' and last_bi[3] < prev_top[3]:
                sell_points.append({'type': '二卖', 'level': 'moderate',
                    'desc': f'反弹不破前高({prev_top[3]:.2f})', 'action': '继续减仓'})
        # 三卖
        if current_price < ZD and df['low'].iloc[-10:].min() < ZD * 0.98:
            sell_points.append({'type': '三卖', 'level': 'aggressive',
                'desc': f'跌破中枢{ZD:.2f}反弹无力', 'action': '清仓离场'})
    return buy_points, sell_points


def chanlun_multi_timeframe(daily_result, hourly_result=None):
    """
    多级别联立分析（区间套）
    日线定方向，60分钟精确定位买卖点
    """
    if hourly_result is None:
        return {'alignment': '单级别(仅日线)', 'recommendation': '', 'conflicts': [], 'daily_dir': '', 'hourly_state': ''}
    daily_dir = daily_result.get('trend_type', {}).get('direction', 'unknown')
    daily_status = daily_result.get('status', '')
    hourly_state = hourly_result.get('stroke_state', {}).get('state', '')
    daily_bull = daily_dir == 'up' or '上方' in daily_status
    daily_bear = daily_dir == 'down' or '下方' in daily_status
    hourly_bull = hourly_state in ['(1,1)', '(1,0)']
    hourly_bear = hourly_state in ['(-1,1)', '(-1,0)']
    conflicts = []
    if daily_bull and hourly_bull:
        alignment = '多级别共振看多'
        recommendation = '日线+60分钟同步看多，寻找60分钟二买/三买加仓'
    elif daily_bear and hourly_bear:
        alignment = '多级别共振看空'
        recommendation = '日线+60分钟同步看空，减仓观望，不抄底'
    elif daily_bull and hourly_bear:
        alignment = '日线多+60分钟空(回调)'
        recommendation = '日线向上但60分钟回调中，等待60分钟底分型后入场(区间套买点)'
        conflicts.append('日线看多但60分钟看空→等小时级别企稳')
    elif daily_bear and hourly_bull:
        alignment = '日线空+60分钟多(反弹)'
        recommendation = '日线向下但60分钟反弹中，反弹不过中枢上沿则减仓(区间套卖点)'
        conflicts.append('日线看空但60分钟反弹→反弹减仓机会')
    else:
        alignment = '多级别信号不明确'
        recommendation = '等待更明确信号'
        conflicts.append('多级别混乱，建议观望')
    return {'alignment': alignment, 'recommendation': recommendation, 'conflicts': conflicts,
            'daily_dir': daily_dir, 'hourly_state': hourly_state}


def chanlun_analysis(df, df_60min=None):
    """
    缠论综合分析（完整版 - 缠中说禅教你炒股票108课）

    参数:
        df: 日线DataFrame (columns: open,high,low,close,volume)
        df_60min: 60分钟线DataFrame (可选，多级别联立)

    返回:
        status, zhongshu, zhongshu_list, bi_count, segment_count,
        buy_points, sell_points, stroke_state, divergence,
        wolf_defense, trend_type, zn_monitor, fractal_recent, multi_tf
    """
    df_proc = chanlun_process_inclusion(df)
    tops, bottoms = chanlun_find_fractals(df_proc)
    bi_points = chanlun_build_bi(df_proc, tops, bottoms)
    segments = chanlun_build_segments(bi_points)
    zhongshu_list = chanlun_find_all_zhongshu(segments)
    stroke_state = chanlun_stroke_state(df)
    divergence = chanlun_macd_divergence(df_proc, bi_points)
    zn_monitor = chanlun_center_monitor(df_proc, zhongshu_list)
    wolf_defense = chanlun_wolf_defense(df)
    trend_type = chanlun_trend_type(bi_points, zhongshu_list)

    current_price = df['close'].iloc[-1]
    buy_points, sell_points = _chanlun_buy_sell_points(
        current_price, zhongshu_list, bi_points, divergence, df)

    fractal_recent = None
    if tops:
        fractal_recent = {'type': '顶分型', 'price': tops[-1][1], 'date': str(tops[-1][2])}
    if bottoms:
        b = bottoms[-1]
        if fractal_recent is None or b[0] > tops[-1][0]:
            fractal_recent = {'type': '底分型', 'price': b[1], 'date': str(b[2])}

    status = "无中枢"
    zhongshu = None
    if zhongshu_list:
        zs = zhongshu_list[-1]
        zhongshu = {'ZD': zs['ZD'], 'ZG': zs['ZG'], 'ZZ': zs['ZZ'],
                    'range': zs['range'], 'strength': zs['strength']}
        if current_price > zs['ZG']:
            status = "中枢上方运行"
        elif current_price < zs['ZD']:
            status = "中枢下方运行"
        else:
            status = "中枢震荡"

    hourly_analysis = None
    if df_60min is not None and len(df_60min) > 0:
        hourly_analysis = chanlun_analysis(df_60min, df_60min=None)

    multi_tf = chanlun_multi_timeframe(
        {'trend_type': trend_type, 'status': status, 'stroke_state': stroke_state},
        hourly_analysis)

    zhongshu_expansion = chanlun_zhongshu_expansion(zhongshu_list, segments)
    xiao_zhuan_da = chanlun_xiao_zhuan_da(df, df_60min)

    return {
        "status": status,
        "zhongshu": zhongshu,
        "zhongshu_list": zhongshu_list,
        "bi_count": len(bi_points) // 2 + 1 if bi_points else 0,
        "segment_count": len(segments),
        "buy_points": buy_points,
        "sell_points": sell_points,
        "stroke_state": stroke_state,
        "divergence": divergence,
        "wolf_defense": wolf_defense,
        "trend_type": trend_type,
        "zn_monitor": zn_monitor,
        "fractal_recent": fractal_recent,
        "multi_tf": multi_tf,
        "zhongshu_expansion": zhongshu_expansion,
        "xiao_zhuan_da": xiao_zhuan_da,
        "_bi_points": bi_points,
        "_segments": segments,
    }

# ===================== 2. 神奇九转（TD序列） =====================
def td_sequential(df):
    """
    德马克TD序列。计算买入和卖出结构的计数。
    买入结构:连续9天收盘价低于4天前的收盘价。
    卖出结构:连续9天收盘价高于4天前的收盘价。
    计数阶段（13）:完成结构后，当收盘价低于/高于结构内某根K线时计数。
    简化版:只实现9转结构和计数，返回当前计数和方向。
    """
    close = df['close'].values
    # 买入结构
    buy_setup = np.zeros(len(close), dtype=int)
    sell_setup = np.zeros(len(close), dtype=int)
    for i in range(4, len(close)):
        if close[i] < close[i-4]:
            buy_setup[i] = buy_setup[i-1] + 1 if buy_setup[i-1] else 1
        else:
            buy_setup[i] = 0
        if close[i] > close[i-4]:
            sell_setup[i] = sell_setup[i-1] + 1 if sell_setup[i-1] else 1
        else:
            sell_setup[i] = 0

    # 检查最近是否完成9转
    last_idx = len(close) - 1
    buy_complete = buy_setup[last_idx] >= 9
    sell_complete = sell_setup[last_idx] >= 9

    # 简化计数阶段:从第9根开始计数，如果价格相对结构内K线创新低/新高
    count_direction = None
    count = 0
    completed = False

    if buy_complete:
        # 找到最近一次完成9转的起始索引
        start_idx = last_idx - buy_setup[last_idx] + 1
        # 计数阶段:当收盘价低于结构内前一根K线的最低价，计数+1，直到13
        count = 0
        for j in range(start_idx, last_idx + 1):
            if j > start_idx and close[j] < close[j-1]:
                count += 1
        count_direction = 'buy'
        if count >= 13:
            completed = True
    elif sell_complete:
        start_idx = last_idx - sell_setup[last_idx] + 1
        count = 0
        for j in range(start_idx, last_idx + 1):
            if j > start_idx and close[j] > close[j-1]:
                count += 1
        count_direction = 'sell'
        if count >= 13:
            completed = True

    return {
        "buy_setup_count": int(buy_setup[last_idx]),
        "sell_setup_count": int(sell_setup[last_idx]),
        "count": count,
        "direction": count_direction,
        "completed": completed
    }

# ===================== 3. 动态支撑阻力 =====================
def dynamic_support_resistance(df, window=50):
    """
    综合多种方法计算支撑和阻力位，返回(支撑列表，阻力列表)，每个元素(价格, 强度)
    方法:近期高低点、均线(20,60,120)、布林带、斐波那契、成交量密集区（简化）
    """
    df = df.tail(window).copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    current_price = close[-1]

    supports = OrderedDict()
    resistances = OrderedDict()

    # 1. 近期明显高点和低点
    recent_high = np.max(high)
    recent_low = np.min(low)
    supports[round(recent_low, 2)] = 70
    resistances[round(recent_high, 2)] = 70

    # 2. 均线
    for period in [20, 60, 120]:
        if len(close) >= period:
            ma = talib.SMA(close, period)[-1]
            if not np.isnan(ma):
                ma = round(ma, 2)
                if ma < current_price:
                    supports[ma] = 50 + period//4
                else:
                    resistances[ma] = 50 + period//4

    # 3. 布林带
    if len(close) >= 20:
        upper, middle, lower = talib.BBANDS(close, 20, 2, 2)
        upper = round(upper[-1], 2)
        middle = round(middle[-1], 2)
        lower = round(lower[-1], 2)
        resistances[upper] = 65
        supports[lower] = 65
        if middle < current_price:
            supports[middle] = 60
        else:
            resistances[middle] = 60

    # 4. 斐波那契（基于近30日高低点）
    if len(df) >= 30:
        hh = np.max(df['high'].iloc[-30:])
        ll = np.min(df['low'].iloc[-30:])
        diff = hh - ll
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for fib in fib_levels:
            price = round(ll + diff * fib, 2)
            if price < current_price:
                supports[price] = 55
            else:
                resistances[price] = 55

    # 5. 成交量密集区（简化:找到成交量最大的价格区间）
    if len(df) >= 10:
        # 粗略将价格分成10个区间
        price_bins = np.linspace(low.min(), high.max(), 20)
        vol_profile = np.zeros(len(price_bins)-1)
        for i in range(len(price_bins)-1):
            mask = (close >= price_bins[i]) & (close <= price_bins[i+1])
            vol_profile[i] = volume[mask].sum()
        max_bin_idx = np.argmax(vol_profile)
        dense_price = round((price_bins[max_bin_idx] + price_bins[max_bin_idx+1])/2, 2)
        if dense_price < current_price:
            supports[dense_price] = 45
        else:
            resistances[dense_price] = 45

    # 合并去重排序
    supports = sorted([(p, s) for p, s in supports.items()], key=lambda x: x[0], reverse=True)
    resistances = sorted([(p, s) for p, s in resistances.items()], key=lambda x: x[0])

    return supports, resistances

# ===================== 4. 动量指标 =====================
def momentum_indicators(df, periods=(10, 20)):
    """
    计算动量相关指标，返回字典
    df: 日线DataFrame
    periods: ROC和CCI使用的周期元组
    """
    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    
    # ROC (Rate of Change)
    roc_short = talib.ROC(close, timeperiod=periods[0])[-1]
    roc_long = talib.ROC(close, timeperiod=periods[1])[-1]
    
    # CCI (Commodity Channel Index)
    cci_short = talib.CCI(high, low, close, timeperiod=periods[0])[-1]
    cci_long = talib.CCI(high, low, close, timeperiod=periods[1])[-1]
    
    # 动量状态研判
    def judge_momentum(roc, cci):
        if roc > 0 and cci > 100:
            return "强势上涨"
        elif roc > 0 and cci < -100:
            return "超卖反弹"
        elif roc < 0 and cci < -100:
            return "强势下跌"
        elif roc < 0 and cci > 100:
            return "超买回落"
        elif -3 < roc < 3 and -100 < cci < 100:
            return "震荡无趋势"
        else:
            return "方向不确定"
    
    state_short = judge_momentum(roc_short, cci_short)
    state_long = judge_momentum(roc_long, cci_long)
    
    # 多周期共振:如果短周期和长周期方向一致则为共振
    direction_short = "bullish" if roc_short > 0 else "bearish" if roc_short < 0 else "neutral"
    direction_long = "bullish" if roc_long > 0 else "bearish" if roc_long < 0 else "neutral"
    resonance = True if direction_short == direction_long and direction_short != "neutral" else False
    
    return {
        'roc_short': roc_short,
        'roc_long': roc_long,
        'cci_short': cci_short,
        'cci_long': cci_long,
        'momentum_state_short': state_short,
        'momentum_state_long': state_long,
        'momentum_resonance': resonance,
        'momentum_direction': direction_short if resonance else "分化"
    }

# ===================== 5. 量价关系 =====================
def volume_price_analysis(df, short_period=5, long_period=20):
    """
    返回量价关系指标字典
    df: 日线DataFrame (必须包含 open, high, low, close, volume)
    """
    close = df['close'].astype(float)
    volume = df['volume'].astype(float)

    # 量比:短期均量 / 长期均量
    vol_ma_short = volume.rolling(window=short_period).mean().iloc[-1]
    vol_ma_long = volume.rolling(window=long_period).mean().iloc[-1]
    vol_ratio = vol_ma_short / vol_ma_long if vol_ma_long != 0 else 1.0

    # 量价配合状态（简化:今天 vs 昨天）
    if len(close) >= 2:
        price_up = close.iloc[-1] > close.iloc[-2]
        vol_up = volume.iloc[-1] > volume.iloc[-2]
        if price_up and vol_up:
            vol_price_resonance = True   # 价涨量增 → 配合
        elif not price_up and not vol_up:
            vol_price_resonance = True   # 价跌量缩 → 也算配合
        else:
            vol_price_resonance = False  # 价涨量缩 或 价跌量增 → 背离
    else:
        vol_price_resonance = True

    # OBV 趋势:最近5日 OBV 是否高于20日均线
    obv = (volume * ((close.diff() > 0).astype(int) * 2 - 1)).cumsum()
    obv_ma5 = obv.rolling(window=5).mean().iloc[-1]
    obv_ma20 = obv.rolling(window=20).mean().iloc[-1]
    obv_trend = obv_ma5 > obv_ma20  # True 表示 OBV 趋势向上

    # 量能验证支撑/阻力:动态支撑阻力中附带成交量信息的占位
    vol_verified_support = []
    vol_verified_resistance = []

    # ---- 地量/天量检测 ----
    vol_today = volume.iloc[-1]
    vol_ma20_val = vol_ma_long if vol_ma_long != 0 else vol_today
    vol_drought = vol_today < vol_ma20_val * 0.5
    vol_spike = vol_today > vol_ma20_val * 2.0
    vol_drought_ratio = round(vol_today / vol_ma20_val, 2) if vol_ma20_val > 0 else 1.0
    vol_spike_ratio = round(vol_today / vol_ma20_val, 2) if vol_ma20_val > 0 else 1.0

    # 地量级别
    if vol_drought_ratio < 0.3:
        vol_type = "极致地量"
    elif vol_drought_ratio < 0.5:
        vol_type = "地量"
    elif vol_spike_ratio > 2.5:
        vol_type = "巨量"
    elif vol_spike_ratio > 2.0:
        vol_type = "天量"
    elif vol_spike_ratio > 1.5:
        vol_type = "放量"
    elif vol_drought_ratio < 0.7:
        vol_type = "缩量"
    else:
        vol_type = "正常量"

    # ---- 格兰维尔八准则关键信号 ----
    granville_signal = None
    if len(close) >= 5 and len(volume) >= 5:
        price_5d_change = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100
        vol_5d_avg = volume.iloc[-5:].mean()
        # 低位可能: 价格在60日最低20%范围内
        if len(close) >= 60:
            low_60 = close.iloc[-60:].min()
            high_60 = close.iloc[-60:].max()
            price_position = (close.iloc[-1] - low_60) / (high_60 - low_60) if high_60 != low_60 else 0.5

            if price_position < 0.2 and vol_drought:
                granville_signal = "准则1:价稳量缩，底部区域"
            elif price_position < 0.2 and vol_spike and price_5d_change > 0:
                granville_signal = "准则7:低位放量止跌，关注反转"
            elif price_position > 0.8 and vol_spike and abs(price_5d_change) < 2:
                granville_signal = "准则6:高位放量滞涨，警惕出货"
            elif price_position > 0.8 and vol_drought and price_5d_change > 0:
                granville_signal = "准则3:高位价涨量缩，背离警告"

    return {
        'vol_ratio': round(vol_ratio, 2),
        'vol_price_resonance': vol_price_resonance,
        'obv_trend': obv_trend,
        'vol_verified_support': vol_verified_support,
        'vol_verified_resistance': vol_verified_resistance,
        # 新增
        'vol_type': vol_type,
        'vol_drought': vol_drought,
        'vol_spike': vol_spike,
        'vol_drought_ratio': vol_drought_ratio,
        'vol_spike_ratio': vol_spike_ratio,
        'granville_signal': granville_signal
    }


# ===================== 5b. 量价形态增强分析 =====================
def volume_pattern_analysis(df, ma_period=20):
    """
    增强量价形态检测:堆量、放量滞涨、缩量回踩、后量超前量、量价背离。
    """
    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    volume = df['volume'].astype(float).values

    if len(close) < 30:
        return _empty_vol_pattern()

    vol_ma20 = talib.SMA(volume, ma_period)
    vol_ma20_val = vol_ma20[-1] if not np.isnan(vol_ma20[-1]) else volume[-1]
    ma20 = talib.SMA(close, ma_period)
    ma20_val = ma20[-1] if not np.isnan(ma20[-1]) else 0

    current_price = close[-1]
    current_vol = volume[-1]

    # ---- 堆量检测:连续N日成交量递增 ----
    stacking_days = 0
    for i in range(len(volume)-1, max(len(volume)-15, 0), -1):
        if volume[i] > volume[i-1] and volume[i] > vol_ma20_val * 0.8:
            stacking_days += 1
        else:
            break
    vol_stacking = stacking_days >= 3

    # ---- 放量滞涨:近5日价格波动<3% 且 量>1.5倍均量 ----
    recent_5_close = close[-5:]
    price_range_5d = (recent_5_close.max() - recent_5_close.min()) / recent_5_close.mean() * 100
    vol_5d_avg = volume[-5:].mean()
    high_vol_stagnation = (price_range_5d < 3.0 and vol_5d_avg > vol_ma20_val * 1.5)

    # ---- 缩量回踩:价格接近MA20(<3%) 且 量<0.7倍均量 ----
    near_ma20 = ma20_val > 0 and abs(current_price - ma20_val) / ma20_val < 0.03
    low_vol_pullback = near_ma20 and current_vol < vol_ma20_val * 0.7

    # ---- 后量超前量:突破前高时量价比较 ----
    volume_breakout = None
    if len(close) >= 60:
        prev_high_60 = high[-60:-5].max()
        if current_price > prev_high_60:
            # 找前高出现时的成交量
            prev_high_idx = np.argmax(high[-60:-5]) + (len(high) - 60)
            prev_high_vol = volume[prev_high_idx] if 0 <= prev_high_idx < len(volume) else 0
            if prev_high_vol > 0:
                vol_ratio_at_breakout = current_vol / prev_high_vol
                volume_breakout = {
                    'breakout': True,
                    'prev_high': round(prev_high_60, 2),
                    'prev_vol': int(prev_high_vol),
                    'current_vol': int(current_vol),
                    'vol_ratio': round(vol_ratio_at_breakout, 2),
                    'valid': vol_ratio_at_breakout > 0.8
                }

    # ---- 量价背离增强（5日窗口） ----
    vol_divergence_detail = None
    if len(close) >= 10:
        price_half = len(close) // 2
        recent_close_seg = close[-10:]
        recent_vol_seg = volume[-10:]
        # 分成前后5日比较
        price_first = np.mean(recent_close_seg[:5])
        price_second = np.mean(recent_close_seg[5:])
        vol_first = np.mean(recent_vol_seg[:5])
        vol_second = np.mean(recent_vol_seg[5:])

        price_change = price_second - price_first
        vol_change = vol_second - vol_first

        if price_change > 0 and vol_change < 0:
            vol_divergence_detail = f"顶背离(近10日:价格+{price_change/price_first*100:.1f}%,量{vol_change/vol_first*100:.1f}%)"
        elif price_change < 0 and vol_change > 0:
            vol_divergence_detail = f"底背离(近10日:价格{price_change/price_first*100:.1f}%,量+{vol_change/vol_first*100:.1f}%)"
        elif price_change > 0 and vol_change > 0:
            vol_divergence_detail = "量价配合上涨"
        elif price_change < 0 and vol_change < 0:
            vol_divergence_detail = "量价配合下跌"

    # ---- 成交量趋势 ----
    if len(vol_ma20) >= 5:
        vol_trend = "放量" if np.mean(volume[-5:]) > vol_ma20_val * 1.3 else (
            "缩量" if np.mean(volume[-5:]) < vol_ma20_val * 0.7 else "平量")
    else:
        vol_trend = "未知"

    return {
        'vol_stacking': vol_stacking,
        'vol_stacking_days': stacking_days,
        'high_vol_stagnation': high_vol_stagnation,
        'low_vol_pullback': low_vol_pullback,
        'volume_breakout': volume_breakout,
        'vol_divergence_detail': vol_divergence_detail,
        'vol_trend': vol_trend,
        'near_ma20': near_ma20,
        'price_range_5d_pct': round(price_range_5d, 2),
    }


def _empty_vol_pattern():
    return {
        'vol_stacking': False, 'vol_stacking_days': 0,
        'high_vol_stagnation': False, 'low_vol_pullback': False,
        'volume_breakout': None, 'vol_divergence_detail': None,
        'vol_trend': '数据不足', 'near_ma20': False, 'price_range_5d_pct': 0
    }


# ===================== 6. DMI（趋势强度与方向） =====================
def dmi_analysis(df, period=14):
    """
    计算 DMI 指标:ADX, +DI, -DI
    df: 日线DataFrame (需包含 high, low, close)
    """
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    # 使用 TA-Lib 计算
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=period)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=period)
    adx = talib.ADX(high, low, close, timeperiod=period)

    adx_value = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0
    pdi = plus_di.iloc[-1] if not np.isnan(plus_di.iloc[-1]) else 0
    mdi = minus_di.iloc[-1] if not np.isnan(minus_di.iloc[-1]) else 0

    # 趋势强度判断
    if adx_value > 25:
        adx_trend = "强趋势"
    elif adx_value > 20:
        adx_trend = "趋势形成"
    else:
        adx_trend = "震荡"

    # DI 交叉状态
    if pdi > mdi:
        di_direction = "多头"
    elif mdi > pdi:
        di_direction = "空头"
    else:
        di_direction = "均衡"

    return {
        'adx': round(adx_value, 2),
        'adx_trend': adx_trend,
        'pdi': round(pdi, 2),
        'mdi': round(mdi, 2),
        'di_direction': di_direction
    }

# ===================== 7. KMeans走势聚类 =====================
class KMeansPattern:
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, historical_daily_list):
        """
        输入历史日线DataFrame列表，训练聚类模型。
        每个DataFrame提取标准化20日收盘价序列作为特征。
        """
        all_features = []
        for df in historical_daily_list:
            if len(df) < 20:
                continue
            closes = df['close'].values
            # 滑动窗口取每20日标准化序列
            for i in range(len(closes)-20):
                segment = closes[i:i+20]
                segment_norm = (segment - segment.mean()) / segment.std() if segment.std() != 0 else np.zeros(20)
                all_features.append(segment_norm)
        if len(all_features) < self.n_clusters:
            print("警告:训练数据不足，无法聚类")
            return
        X = np.array(all_features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        print(f"KMeans训练完成，样本数:{len(all_features)}")

    def predict(self, df_recent):
        """返回最近20日走势的簇标签，若未训练返回-1"""
        if not self.fitted:
            return -1
        if len(df_recent) < 20:
            return -1
        closes = df_recent['close'].values[-20:]
        segment_norm = (closes - closes.mean()) / closes.std() if closes.std() != 0 else np.zeros(20)
        X = self.scaler.transform([segment_norm])
        return self.model.predict(X)[0]


# ===================== 8. K线形态识别 =====================
def candlestick_patterns(df, lookback=20):
    """
    识别经典K线形态，返回最近lookback根的形态列表和汇总。
    检测:分型、启明星/黄昏之星、吞没、锤子/射击之星、十字星、
          三鸦/三兵、孕线、刺透/乌云盖顶
    """
    if len(df) < 20:
        return _empty_pattern_result()

    open_ = df['open'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    volume = df['volume'].astype(float).values

    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    candle_range = high - low

    patterns_found = []
    pattern_summary = {
        'top_fractal': [], 'bottom_fractal': [],
        'evening_star': [], 'morning_star': [],
        'bullish_engulfing': [], 'bearish_engulfing': [],
        'hammer': [], 'shooting_star': [],
        'doji': [], 'three_crows': [], 'three_soldiers': [],
        'harami_bullish': [], 'harami_bearish': [],
        'piercing': [], 'dark_cloud_cover': []
    }

    idx_end = len(close)
    idx_start = max(0, idx_end - lookback)

    for i in range(idx_start, idx_end):
        # 跳过分型需要边界的情况 (前一根和后一根)
        if i < 1 or i >= idx_end - 1:
            # 十字星等单根形态在最后位置仍可检测
            if i >= idx_end - 1 and candle_range[i] > 0 and body[i] < candle_range[i] * 0.1:
                pattern_summary['doji'].append({
                    'index': i, 'price': round(close[i], 2),
                    'date': str(df.index[i])[:10]
                })
            continue

        # ---- 顶分型 ----
        if high[i] > high[i-1] and high[i] > high[i+1]:
            pattern_summary['top_fractal'].append({
                'index': i, 'price': round(high[i], 2),
                'date': str(df.index[i])[:10]
            })

        # ---- 底分型 ----
        if low[i] < low[i-1] and low[i] < low[i+1]:
            pattern_summary['bottom_fractal'].append({
                'index': i, 'price': round(low[i], 2),
                'date': str(df.index[i])[:10]
            })

        # ---- 十字星 (Doji): body < 10% of range ----
        if candle_range[i] > 0 and body[i] < candle_range[i] * 0.1:
            pattern_summary['doji'].append({
                'index': i, 'price': round(close[i], 2),
                'date': str(df.index[i])[:10]
            })

        # ---- 锤子线 (Hammer): 下影 >= 2*body, 上影 < 0.3*body, 在下跌趋势中 ----
        if i >= 2 and body[i] > 0:
            if lower_shadow[i] >= 2 * body[i] and upper_shadow[i] < 0.3 * body[i]:
                # 确认在下跌趋势中（前两日下跌）
                if close[i-1] < close[i-2] and close[i-1] < open_[i-1]:
                    pattern_summary['hammer'].append({
                        'index': i, 'price': round(close[i], 2),
                        'date': str(df.index[i])[:10]
                    })

        # ---- 射击之星 (Shooting Star): 上影 >= 2*body, 下影 < 0.3*body ----
        if i >= 2 and body[i] > 0:
            if upper_shadow[i] >= 2 * body[i] and lower_shadow[i] < 0.3 * body[i]:
                if close[i-2] < close[i-1]:
                    pattern_summary['shooting_star'].append({
                        'index': i, 'price': round(close[i], 2),
                        'date': str(df.index[i])[:10]
                    })

        # ---- 三根K线形态 ----
        if i >= 2:
            # 启明星 (Morning Star)
            if (_is_bearish(open_, close, i-2) and
                body[i-1] < candle_range[i-1] * 0.3 and
                _is_bullish(open_, close, i) and
                close[i] > (open_[i-2] + close[i-2]) / 2):
                pattern_summary['morning_star'].append({
                    'index': i, 'price': round(close[i], 2),
                    'date': str(df.index[i])[:10]
                })

            # 黄昏之星 (Evening Star)
            if (_is_bullish(open_, close, i-2) and
                body[i-1] < candle_range[i-1] * 0.3 and
                _is_bearish(open_, close, i) and
                close[i] < (open_[i-2] + close[i-2]) / 2):
                pattern_summary['evening_star'].append({
                    'index': i, 'price': round(close[i], 2),
                    'date': str(df.index[i])[:10]
                })

            # 刺透形态 (Piercing)
            if (_is_bearish(open_, close, i-1) and
                _is_bullish(open_, close, i) and
                open_[i] < low[i-1] and
                close[i] > (open_[i-1] + close[i-1]) / 2):
                pattern_summary['piercing'].append({
                    'index': i, 'price': round(close[i], 2),
                    'date': str(df.index[i])[:10]
                })

            # 乌云盖顶 (Dark Cloud Cover)
            if (_is_bullish(open_, close, i-1) and
                _is_bearish(open_, close, i) and
                open_[i] > high[i-1] and
                close[i] < (open_[i-1] + close[i-1]) / 2):
                pattern_summary['dark_cloud_cover'].append({
                    'index': i, 'price': round(close[i], 2),
                    'date': str(df.index[i])[:10]
                })

            # 吞没 (Engulfing)
            open_i, close_i = open_[i], close[i]
            open_p, close_p = open_[i-1], close[i-1]
            # 看涨吞没
            if (close_p < open_p and close_i > open_i and
                open_i < close_p and close_i > open_p):
                pattern_summary['bullish_engulfing'].append({
                    'index': i, 'price': round(close_i, 2),
                    'date': str(df.index[i])[:10]
                })
            # 看跌吞没
            if (close_p > open_p and close_i < open_i and
                open_i > close_p and close_i < open_p):
                pattern_summary['bearish_engulfing'].append({
                    'index': i, 'price': round(close_i, 2),
                    'date': str(df.index[i])[:10]
                })

            # 孕线 (Harami)
            prev_body = abs(open_p - close_p)
            curr_body = abs(open_i - close_i)
            if prev_body > 0 and curr_body < prev_body * 0.5:
                if (high[i] <= high[i-1] and low[i] >= low[i-1]):
                    if close_p < open_p:
                        pattern_summary['harami_bullish'].append({
                            'index': i, 'price': round(close_i, 2),
                            'date': str(df.index[i])[:10]
                        })
                    else:
                        pattern_summary['harami_bearish'].append({
                            'index': i, 'price': round(close_i, 2),
                            'date': str(df.index[i])[:10]
                        })

        # ---- 三只乌鸦/红三兵 (需要i>=3) ----
        if i >= 3:
            # 三只乌鸦
            if all(_is_bearish(open_, close, i-j) for j in range(3)):
                if all(body[i-j] > candle_range[i-j] * 0.4 for j in range(3)):
                    pattern_summary['three_crows'].append({
                        'index': i, 'price': round(close[i], 2),
                        'date': str(df.index[i])[:10]
                    })
            # 红三兵
            if all(_is_bullish(open_, close, i-j) for j in range(3)):
                if all(body[i-j] > candle_range[i-j] * 0.4 for j in range(3)):
                    pattern_summary['three_soldiers'].append({
                        'index': i, 'price': round(close[i], 2),
                        'date': str(df.index[i])[:10]
                    })

    # ---- 汇总最近信号 ----
    recent_signals = []
    for pattern_name, occurrences in pattern_summary.items():
        for occ in occurrences:
            if occ['index'] >= idx_end - 5:
                recent_signals.append(f"{_pattern_label(pattern_name)}({occ['date']}@{occ['price']})")

    # 各形态计数
    counts = {k: len(v) for k, v in pattern_summary.items()}

    return {
        'counts': counts,
        'recent_signals': recent_signals[:10],
        'latest_fractal': _latest_fractal(pattern_summary, close, idx_end),
        'latest_reversal': _latest_reversal(pattern_summary, close, idx_end),
        'dominant_pattern': _dominant_pattern(counts)
    }


def _is_bullish(open_, close, i):
    return close[i] > open_[i]


def _is_bearish(open_, close, i):
    return close[i] < open_[i]


def _pattern_label(name):
    labels = {
        'top_fractal': '顶分型', 'bottom_fractal': '底分型',
        'evening_star': '黄昏之星', 'morning_star': '启明星',
        'bullish_engulfing': '看涨吞没', 'bearish_engulfing': '看跌吞没',
        'hammer': '锤子线', 'shooting_star': '射击之星',
        'doji': '十字星', 'three_crows': '三只乌鸦', 'three_soldiers': '红三兵',
        'harami_bullish': '看涨孕线', 'harami_bearish': '看跌孕线',
        'piercing': '刺透形态', 'dark_cloud_cover': '乌云盖顶'
    }
    return labels.get(name, name)


def _latest_fractal(pattern_summary, close, idx_end):
    """返回最近的一个顶分型或底分型"""
    latest_top = pattern_summary['top_fractal'][-1] if pattern_summary['top_fractal'] else None
    latest_bottom = pattern_summary['bottom_fractal'][-1] if pattern_summary['bottom_fractal'] else None

    result = {'type': None, 'price': None, 'date': None}
    if latest_top and latest_top['index'] >= idx_end - 3:
        result = {'type': '顶分型', 'price': latest_top['price'], 'date': latest_top['date']}
    if latest_bottom and latest_bottom['index'] >= idx_end - 3:
        if latest_top is None or latest_bottom['index'] > latest_top['index']:
            result = {'type': '底分型', 'price': latest_bottom['price'], 'date': latest_bottom['date']}
    return result


def _latest_reversal(pattern_summary, close, idx_end):
    """返回最近的反转信号（启明星/黄昏之星/吞没/刺透/乌云盖顶）"""
    reversal_patterns = ['morning_star', 'evening_star', 'bullish_engulfing',
                         'bearish_engulfing', 'piercing', 'dark_cloud_cover']
    latest = None
    for rp in reversal_patterns:
        if pattern_summary[rp]:
            occ = pattern_summary[rp][-1]
            if latest is None or occ['index'] > latest['index']:
                latest = {**occ, 'pattern': _pattern_label(rp)}
    if latest and latest['index'] >= idx_end - 5:
        return latest
    return None


def _dominant_pattern(counts):
    """返回最近出现最多的形态类型"""
    non_zero = {k: v for k, v in counts.items() if v > 0}
    if not non_zero:
        return '无明显形态'
    # 按重要性排序
    priority = ['evening_star', 'morning_star', 'three_crows', 'three_soldiers',
                'bullish_engulfing', 'bearish_engulfing', 'piercing', 'dark_cloud_cover',
                'hammer', 'shooting_star', 'harami_bullish', 'harami_bearish',
                'top_fractal', 'bottom_fractal', 'doji']
    for p in priority:
        if p in non_zero and non_zero[p] >= 2:
            return f"{_pattern_label(p)}x{non_zero[p]}"
    # 返回出现最多的
    best = max(non_zero, key=non_zero.get)
    return f"{_pattern_label(best)}x{non_zero[best]}"


def _empty_pattern_result():
    return {
        'counts': {}, 'recent_signals': [],
        'latest_fractal': {'type': None, 'price': None, 'date': None},
        'latest_reversal': None, 'dominant_pattern': '数据不足'
    }


# ===================== 9. DMA 指标 =====================
def dma_analysis(df, short=10, long=50, m=10):
    """
    DMA（平均线差）指标。
    DMA = MA(close, short) - MA(close, long)   —— 快慢均线差
    AMA = MA(DMA, m)                           —— DMA的m日均值
    返回当前值、AMA值、金叉/死叉状态、趋势方向。
    """
    close = df['close'].astype(float).values
    if len(close) < max(short, long, m) + 2:
        return {'dma': 0, 'ama': 0, 'dma_diff': 0, 'dma_signal': '数据不足', 'dma_trend': '未知'}

    ma_short = talib.SMA(close, short)
    ma_long = talib.SMA(close, long)

    # DMA 序列
    dma_series = ma_short - ma_long
    ama_series = talib.SMA(dma_series, m)

    dma_now = dma_series[-1]
    ama_now = ama_series[-1]
    dma_prev = dma_series[-2]
    ama_prev = ama_series[-2]

    dma_diff = dma_now - ama_now

    # 金叉/死叉
    if dma_prev <= ama_prev and dma_now > ama_now:
        signal = '金叉'
    elif dma_prev >= ama_prev and dma_now < ama_now:
        signal = '死叉'
    elif dma_now > ama_now:
        signal = '多头排列'
    else:
        signal = '空头排列'

    # DMA趋势方向
    if dma_now > 0 and dma_now > ama_now:
        dma_trend = '强势多头'
    elif dma_now > 0 and dma_now < ama_now:
        dma_trend = '多头回调'
    elif dma_now < 0 and dma_now < ama_now:
        dma_trend = '强势空头'
    elif dma_now < 0 and dma_now > ama_now:
        dma_trend = '空头反弹'
    else:
        dma_trend = '震荡'

    return {
        'dma': round(dma_now, 2),
        'ama': round(ama_now, 2),
        'dma_diff': round(dma_diff, 2),
        'dma_signal': signal,
        'dma_trend': dma_trend
    }


# ===================== 10. 增强OBV分析 =====================
def obv_analysis(df):
    """
    增强版OBV（能量潮）分析。
    返回OBV趋势、OBV与价格的背离检测、OBV突破信号。
    """
    close = df['close'].astype(float).values
    volume = df['volume'].astype(float).values
    if len(close) < 30:
        return {'obv_trend': False, 'obv_divergence': '数据不足', 'obv_signal': '中性'}

    # 计算OBV
    obv = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]

    # OBV均线
    obv_ma5 = talib.SMA(obv, 5)
    obv_ma20 = talib.SMA(obv, 20)

    obv_trend = obv_ma5[-1] > obv_ma20[-1] if not (np.isnan(obv_ma5[-1]) or np.isnan(obv_ma20[-1])) else False

    # OBV与价格背离检测（最近10日）
    recent_10 = min(10, len(close))
    price_slice = close[-recent_10:]
    obv_slice = obv[-recent_10:]

    # 价格趋势
    price_half = len(price_slice) // 2
    price_first_half = np.mean(price_slice[:price_half])
    price_second_half = np.mean(price_slice[price_half:])
    price_direction = "上涨" if price_second_half > price_first_half else "下跌"

    obv_first_half = np.mean(obv_slice[:price_half])
    obv_second_half = np.mean(obv_slice[price_half:])
    obv_direction = "上升" if obv_second_half > obv_first_half else "下降"

    if price_direction == "上涨" and obv_direction == "下降":
        divergence = "顶背离（价涨OBV降，警惕回调）"
    elif price_direction == "下跌" and obv_direction == "上升":
        divergence = "底背离（价跌OBV升，关注反弹）"
    elif price_direction == "上涨" and obv_direction == "上升":
        divergence = "量价配合上涨"
    elif price_direction == "下跌" and obv_direction == "下降":
        divergence = "量价配合下跌"
    else:
        divergence = "无背离"

    # OBV突破信号（OBV创近期新高/新低）
    obv_20_high = np.max(obv[-20:])
    obv_20_low = np.min(obv[-20:])
    if obv[-1] >= obv_20_high * 0.98:
        obv_signal = "OBV突破新高，资金涌入"
    elif obv[-1] <= obv_20_low * 1.02:
        obv_signal = "OBV新低，资金流出"
    else:
        obv_signal = "中性"

    return {
        'obv_trend': obv_trend,
        'obv_divergence': divergence,
        'obv_signal': obv_signal,
        'obv_price_direction': price_direction,
        'obv_direction': obv_direction
    }


# ===================== 11. 艾略特波浪理论 =====================
def find_swing_points(df, min_bars=2, min_pct=0.01):
    """
    ZigZag 枢轴检测:找出所有阶段性高点和低点。
    min_bars: 两个同向枢轴之间最少K线数
    min_pct: 枢轴确认的最小价格变动百分比
    返回 (swing_highs, swing_lows)，每个元素为 (index, price, date)
    """
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values

    swing_highs = []
    swing_lows = []
    last_direction = None

    for i in range(min_bars, len(close) - min_bars):
        # 局部高点检测
        is_local_high = True
        for j in range(1, min_bars + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_local_high = False
                break

        # 局部低点检测
        is_local_low = True
        for j in range(1, min_bars + 1):
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_local_low = False
                break

        if is_local_high and last_direction != 'up':
            swing_highs.append((i, high[i], str(df.index[i])[:10]))
            last_direction = 'up'

        if is_local_low and last_direction != 'down':
            swing_lows.append((i, low[i], str(df.index[i])[:10]))
            last_direction = 'down'

    # 按时间合并排序所有枢轴点
    all_pivots = [(idx, price, date, 'high') for idx, price, date in swing_highs] + \
                 [(idx, price, date, 'low') for idx, price, date in swing_lows]
    all_pivots.sort(key=lambda x: x[0])

    # 确保高低交替
    filtered_pivots = []
    last_type = None
    for p in all_pivots:
        if last_type is None or p[3] != last_type:
            filtered_pivots.append(p)
            last_type = p[3]

    return swing_highs, swing_lows, filtered_pivots


def _get_wave_label(wave_num):
    labels = {
        1: '浪1-初升', 2: '浪2-回调', 3: '浪3-主升',
        4: '浪4-调整', 5: '浪5-末升',
        'A': '浪A-初跌', 'B': '浪B-反弹', 'C': '浪C-主跌'
    }
    return labels.get(wave_num, str(wave_num))


def _calc_fib_retracement(start_price, end_price, level):
    """计算斐波那契回调位"""
    diff = end_price - start_price
    if start_price < end_price:  # 上升趋势
        return round(end_price - diff * level, 2)
    else:  # 下降趋势
        return round(end_price + abs(diff) * level, 2)


def _calc_fib_extension(start_price, end_price, level):
    """计算斐波那契扩展位"""
    diff = end_price - start_price
    return round(end_price + diff * level, 2)


def elliott_wave_analysis(df, trend_direction='auto'):
    """
    艾略特波浪分析主函数。
    检测推动5浪和修正ABC浪，验证规则，计算斐波那契关系。

    返回:
    {
        'wave_pattern': 'impulse_5'/'corrective_abc'/'unknown',
        'current_wave': 当前浪标签,
        'wave_structure': 浪结构描述,
        'rules_valid': 三大铁律是否满足,
        'rule_violations': 违规列表,
        'fib_ratios': 各类斐波那契比率,
        'next_target': 下一目标价,
        'next_support': 下一支撑价,
        'confidence': 0-1置信度,
        'trade_setup': 交易建议
    }
    """
    if len(df) < 60:
        return _empty_elliott_result()

    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values

    # 自动判断主趋势方向
    if trend_direction == 'auto':
        ma20 = talib.SMA(close, 20)
        ma60 = talib.SMA(close, 60)
        trend_direction = 'up' if ma20[-1] > ma60[-1] else 'down'

    # Step 1: 检测枢轴点
    swing_highs, swing_lows, pivots = find_swing_points(df, min_bars=2, min_pct=0.01)

    if len(pivots) < 5:
        return {**_empty_elliott_result(), 'wave_pattern': '枢轴不足',
                'pivot_count': len(pivots)}

    current_price = close[-1]

    # Step 2: 根据趋势方向匹配推动浪(5浪)或修正浪(ABC)
    wave_result = _match_wave_pattern(pivots, swing_highs, swing_lows, current_price, trend_direction)

    # Step 3: 斐波那契验证
    fib_result = _compute_wave_fibonacci(wave_result, current_price, trend_direction)

    # Step 4: 规则验证与置信度评分
    validation = _validate_elliott_rules(wave_result, trend_direction)

    # Step 5: 交易设置
    trade_setup = _generate_elliott_trade_setup(wave_result, fib_result, validation, current_price, trend_direction)

    return {
        'wave_pattern': wave_result['pattern'],
        'current_wave': wave_result['current_wave'],
        'wave_structure': wave_result['structure_desc'],
        'wave_points': wave_result.get('wave_points', {}),
        'rules_valid': validation['valid'],
        'rule_violations': validation['violations'],
        'fib_ratios': fib_result,
        'next_target': fib_result.get('next_target'),
        'next_support': fib_result.get('next_support'),
        'confidence': validation['confidence'],
        'trade_setup': trade_setup,
        'pivot_count': len(pivots),
        'trend_direction': trend_direction
    }


def _match_wave_pattern(pivots, swing_highs, swing_lows, current_price, trend):
    """匹配波浪模式（推动浪或修正浪）"""
    result = {
        'pattern': 'unknown',
        'current_wave': '未知',
        'structure_desc': '未知',
        'wave_points': {}
    }

    # 取最近的枢轴点（最多8个用于分析）
    recent = pivots[-8:] if len(pivots) >= 8 else pivots

    if trend == 'up':
        # 上升趋势找推动浪: 低→高→低→高→低→高→低→高→低(5浪)
        # 最后一个转折:如果最后是high，可能在浪5顶；如果最后是low，可能在浪4底或浪2底
        points = []
        for p in recent:
            points.append({'idx': p[0], 'price': p[1], 'date': p[2], 'type': p[3]})

        if len(points) >= 5:
            # 尝试标记: [0]=浪0(起点低), [1]=浪1顶, [2]=浪2底, [3]=浪3顶, [4]=浪4底, [5]=浪5顶
            if points[0]['type'] == 'low':
                result['wave_points'] = {
                    'wave_0_start': points[0],
                    'wave_1_top': points[1] if len(points) > 1 and points[1]['type'] == 'high' else None,
                    'wave_2_bottom': points[2] if len(points) > 2 and points[2]['type'] == 'low' else None,
                    'wave_3_top': points[3] if len(points) > 3 and points[3]['type'] == 'high' else None,
                    'wave_4_bottom': points[4] if len(points) > 4 and points[4]['type'] == 'low' else None,
                    'wave_5_top': points[5] if len(points) > 5 and points[5]['type'] == 'high' else None,
                }
                wp = result['wave_points']

                # 判断当前处于哪一浪
                if wp['wave_5_top'] is not None:
                    result['pattern'] = 'impulse_5_complete'
                    result['current_wave'] = '浪5完成/修正浪开始'
                    result['structure_desc'] = '5浪推动结构疑似完成，关注ABC回调'
                elif wp['wave_4_bottom'] is not None and wp['wave_5_top'] is None:
                    result['pattern'] = 'impulse_4_complete'
                    result['current_wave'] = '浪5进行中'
                    result['structure_desc'] = '浪4回调完成，可能进入浪5末升段'
                elif wp['wave_3_top'] is not None and wp['wave_4_bottom'] is None:
                    result['pattern'] = 'impulse_3_complete'
                    result['current_wave'] = '浪4调整中'
                    result['structure_desc'] = '浪3主升完成，处于浪4回调'
                elif wp['wave_2_bottom'] is not None and wp['wave_3_top'] is not None:
                    result['pattern'] = 'impulse_in_progress'
                    result['current_wave'] = '浪3进行中'
                    result['structure_desc'] = '浪2回调完成，浪3主升段进行中'
                elif wp['wave_1_top'] is not None and wp['wave_2_bottom'] is not None:
                    result['current_wave'] = '浪2调整中'
                    result['structure_desc'] = '浪1初升完成，处于浪2回调'
                else:
                    result['current_wave'] = '浪1初升'
                    result['structure_desc'] = '可能处于浪1启动阶段'

    else:  # 下降趋势
        points = []
        for p in recent:
            points.append({'idx': p[0], 'price': p[1], 'date': p[2], 'type': p[3]})

        if len(points) >= 5:
            # 下降趋势: 高→低→高→低→高→低(5浪下跌)
            if points[0]['type'] == 'high':
                result['wave_points'] = {
                    'wave_0_start': points[0],
                    'wave_1_bottom': points[1] if len(points) > 1 and points[1]['type'] == 'low' else None,
                    'wave_2_top': points[2] if len(points) > 2 and points[2]['type'] == 'high' else None,
                    'wave_3_bottom': points[3] if len(points) > 3 and points[3]['type'] == 'low' else None,
                    'wave_4_top': points[4] if len(points) > 4 and points[4]['type'] == 'high' else None,
                    'wave_5_bottom': points[5] if len(points) > 5 and points[5]['type'] == 'low' else None,
                }
                wp = result['wave_points']

                if wp['wave_5_bottom'] is not None:
                    result['pattern'] = 'impulse_5_complete'
                    result['current_wave'] = '浪5完成/修正浪开始'
                    result['structure_desc'] = '5浪下跌疑似完成，关注ABC反弹'
                elif wp['wave_4_top'] is not None:
                    result['pattern'] = 'impulse_4_complete'
                    result['current_wave'] = '浪5进行中'
                    result['structure_desc'] = '浪4反弹完成，可能进入浪5末跌段'
                elif wp['wave_3_bottom'] is not None:
                    result['pattern'] = 'impulse_3_complete'
                    result['current_wave'] = '浪4反弹中'
                    result['structure_desc'] = '浪3主跌完成，处于浪4反弹'
                elif wp['wave_2_top'] is not None:
                    result['current_wave'] = '浪3进行中'
                    result['structure_desc'] = '浪2反弹完成，浪3主跌段'
                else:
                    result['current_wave'] = '浪1初跌'
                    result['structure_desc'] = '可能处于浪1下跌启动阶段'

    return result


def _compute_wave_fibonacci(wave_result, current_price, trend):
    """计算波浪斐波那契关系"""
    fib = {
        'next_target': None,
        'next_support': None,
        'w2_retrace_w1': None,
        'w3_extend_w1': None,
        'w4_retrace_w3': None,
        'w5_target_1': None,
        'w5_target_2': None,
        'w5_target_3': None,
    }

    wp = wave_result.get('wave_points', {})
    if not wp:
        return fib

    if trend == 'up':
        w0 = wp.get('wave_0_start')
        w1 = wp.get('wave_1_top')
        w2 = wp.get('wave_2_bottom')
        w3 = wp.get('wave_3_top')
        w4 = wp.get('wave_4_bottom')
        w5 = wp.get('wave_5_top')

        # 浪2回撤浪1
        if w0 and w1 and w2:
            wave1_len = w1['price'] - w0['price']
            wave2_retrace = w1['price'] - w2['price']
            if wave1_len > 0:
                fib['w2_retrace_w1'] = round(wave2_retrace / wave1_len, 2)

        # 浪3延伸
        if w0 and w1 and w3:
            wave1_len = w1['price'] - w0['price']
            wave3_len_from_w0 = w3['price'] - w2['price'] if w2 else w3['price'] - w0['price']
            if wave1_len > 0:
                fib['w3_extend_w1'] = round(wave3_len_from_w0 / wave1_len, 2)

        # 浪4回撤浪3
        if w2 and w3 and w4:
            wave3_len = w3['price'] - w2['price']
            wave4_retrace = w3['price'] - w4['price']
            if wave3_len > 0:
                fib['w4_retrace_w3'] = round(wave4_retrace / wave3_len, 2)

        # 浪5目标（基于浪1长度延伸）
        if w0 and w1 and (w4 or w3):
            wave1_len = w1['price'] - w0['price']
            base = w4['price'] if w4 else w3['price']
            fib['w5_target_1'] = round(base + wave1_len * 1.0, 2)
            fib['w5_target_2'] = round(base + wave1_len * 1.618, 2)
            fib['w5_target_3'] = round(base + wave1_len * 2.618, 2)
            fib['next_target'] = fib['w5_target_2']

        # 下方支撑
        if w4:
            fib['next_support'] = w4['price']
        elif w2:
            fib['next_support'] = w2['price']

    else:  # 下降趋势
        w0 = wp.get('wave_0_start')
        w1 = wp.get('wave_1_bottom')
        w2 = wp.get('wave_2_top')
        w3 = wp.get('wave_3_bottom')
        w4 = wp.get('wave_4_top')
        w5 = wp.get('wave_5_bottom')

        if w0 and w1 and w2:
            wave1_len = w0['price'] - w1['price']
            wave2_retrace = w2['price'] - w1['price']
            if wave1_len > 0:
                fib['w2_retrace_w1'] = round(wave2_retrace / wave1_len, 2)

        if w2 and w3 and w4:
            wave3_len = w2['price'] - w3['price']
            wave4_retrace = w4['price'] - w3['price']
            if wave3_len > 0:
                fib['w4_retrace_w3'] = round(wave4_retrace / wave3_len, 2)

        if w0 and w1 and (w4 or w3):
            wave1_len = w0['price'] - w1['price']
            base = w4['price'] if w4 else w3['price']
            fib['w5_target_1'] = round(base - wave1_len * 1.0, 2)
            fib['w5_target_2'] = round(base - wave1_len * 1.618, 2)
            fib['w5_target_3'] = round(base - wave1_len * 2.618, 2)
            fib['next_target'] = fib['w5_target_2']

        if w4:
            fib['next_support'] = w4['price']

    return fib


def _validate_elliott_rules(wave_result, trend):
    """验证艾略特三大铁律，返回合规性和置信度"""
    violations = []
    confidence = 0.5
    wp = wave_result.get('wave_points', {})

    if trend == 'up':
        w0 = wp.get('wave_0_start')
        w1 = wp.get('wave_1_top')
        w2 = wp.get('wave_2_bottom')
        w3 = wp.get('wave_3_top')
        w4 = wp.get('wave_4_bottom')

        # 铁律1: 浪2不能跌破浪1起点
        if w0 and w1 and w2:
            if w2['price'] <= w0['price']:
                violations.append('违规:浪2跌破浪1起点')
            else:
                confidence += 0.15

        # 铁律2: 浪3不能是最短的推动浪
        if w0 and w1 and w2 and w3:
            wave1_len = w1['price'] - w0['price']
            wave3_len = w3['price'] - w2['price']
            wave5_len = None
            if wave1_len > 0 and wave3_len > 0:
                if wave3_len < wave1_len:
                    violations.append('违规:浪3短于浪1，可能不是推动浪')
                else:
                    confidence += 0.15

        # 铁律3: 浪4不能与浪1重叠
        if w1 and w4:
            if w4['price'] <= w1['price']:
                violations.append('违规:浪4低点低于浪1高点(重叠)')
            else:
                confidence += 0.15

        # 浪2回撤应在0.382~0.786之间
        if w0 and w1 and w2:
            wave1_len = w1['price'] - w0['price']
            retrace = (w1['price'] - w2['price']) / wave1_len if wave1_len > 0 else 0
            if 0.382 <= retrace <= 0.786:
                confidence += 0.1
            elif 0.3 <= retrace <= 0.9:
                confidence += 0.05

    else:  # 下降趋势
        w0 = wp.get('wave_0_start')
        w1 = wp.get('wave_1_bottom')
        w2 = wp.get('wave_2_top')
        w3 = wp.get('wave_3_bottom')
        w4 = wp.get('wave_4_top')

        if w0 and w1 and w2:
            if w2['price'] >= w0['price']:
                violations.append('违规:浪2反弹超过浪1起点')

        if w1 and w2 and w3:
            wave1_len = w0['price'] - w1['price'] if w0 else abs(w2['price'] - w1['price'])
            wave3_len = w2['price'] - w3['price'] if w2 else 0
            wave5_len = None
            if wave3_len > 0 and wave1_len > 0:
                if wave3_len < wave1_len:
                    violations.append('违规:浪3短于浪1')

        if w1 and w4:
            if w4['price'] >= w1['price']:
                violations.append('违规:浪4高点高于浪1低点(重叠)')

    confidence = min(0.95, max(0.1, confidence))
    return {
        'valid': len(violations) == 0,
        'violations': violations,
        'confidence': round(confidence, 2)
    }


def _generate_elliott_trade_setup(wave_result, fib, validation, current_price, trend):
    """根据波浪分析生成交易建议"""
    current_wave = wave_result.get('current_wave', '未知')
    pattern = wave_result.get('wave_pattern', 'unknown')
    conf = validation['confidence']

    setup = {
        'signal': 'neutral',
        'setup_type': '无明确波浪交易机会',
        'entry_zone': '',
        'stop_loss': '',
        'target': '',
        'rationale': ''
    }

    if conf < 0.4:
        setup['rationale'] = '波浪结构置信度不足，建议观望'
        return setup

    wp = wave_result.get('wave_points', {})

    if trend == 'up':
        if current_wave == '浪2调整中' and conf >= 0.5:
            setup['signal'] = 'bullish'
            setup['setup_type'] = '浪2回调买入'
            w1 = wp.get('wave_1_top')
            fib_618 = _calc_fib_retracement(
                wp['wave_0_start']['price'] if wp.get('wave_0_start') else current_price,
                w1['price'] if w1 else current_price, 0.618)
            setup['entry_zone'] = f"{fib_618:.2f}附近(浪1的61.8%回调位)"
            setup['stop_loss'] = f"{wp['wave_0_start']['price']:.2f}(浪1起点下方)" if wp.get('wave_0_start') else '浪1起点'
            setup['target'] = f"{fib.get('next_target', '?')}(浪3预期目标)"
            setup['rationale'] = '浪2回调至0.5-0.618黄金区域，一旦完成回调将启动浪3主升'

        elif current_wave in ['浪3进行中', '浪3主升'] and conf >= 0.5:
            setup['signal'] = 'bullish'
            setup['setup_type'] = '浪3追涨'
            w2 = wp.get('wave_2_bottom')
            w3 = wp.get('wave_3_top')
            setup['entry_zone'] = f"突破{w3['price']:.2f}(浪3高点)确认" if w3 else f"{current_price:.2f}"
            setup['stop_loss'] = f"{w2['price']:.2f}(浪2低点)" if w2 else '浪2低点'
            setup['target'] = f"{fib.get('w5_target_2', '?')}(浪5目标)"

        elif current_wave == '浪4调整中' and conf >= 0.5:
            setup['signal'] = 'bullish'
            setup['setup_type'] = '浪4回调低吸'
            w3 = wp.get('wave_3_top')
            fib_382 = _calc_fib_retracement(
                wp['wave_2_bottom']['price'] if wp.get('wave_2_bottom') else current_price,
                w3['price'] if w3 else current_price, 0.382)
            setup['entry_zone'] = f"{fib_382:.2f}附近(浪3的38.2%回调位)"
            setup['stop_loss'] = f"{wp['wave_1_top']['price']:.2f}(浪1顶/铁律3保护)" if wp.get('wave_1_top') else '浪1顶'
            setup['target'] = f"{fib.get('next_target', '?')}(浪5目标)"
            setup['rationale'] = '浪4回调至浪3的38.2%且不低于浪1顶，符合铁律3'

        elif current_wave in ['浪5完成/修正浪开始', 'impulse_5_complete'] and conf >= 0.5:
            setup['signal'] = 'bearish'
            setup['setup_type'] = '5浪完成/ABC回调'
            setup['entry_zone'] = f"跌破{wp['wave_4_bottom']['price']:.2f}(浪4底)确认回调" if wp.get('wave_4_bottom') else '跌破浪4底'
            setup['stop_loss'] = f"{wp['wave_5_top']['price']:.2f}(浪5高点)" if wp.get('wave_5_top') else '浪5高点'
            setup['target'] = f"{_calc_fib_retracement(wp['wave_0_start']['price'], wp['wave_5_top']['price'], 0.382):.2f}(38.2%回调)" if wp.get('wave_0_start') and wp.get('wave_5_top') else '斐波那契回调'
            setup['rationale'] = '5浪推动完成后通常出现ABC三浪回调，目标至少38.2%回调'

    else:  # 下降趋势
        if current_wave == '浪2反弹中' and conf >= 0.5:
            setup['signal'] = 'bearish'
            setup['setup_type'] = '浪2反弹做空'
            w1 = wp.get('wave_1_bottom')
            if w1 and wp.get('wave_0_start'):
                fib_618 = _calc_fib_retracement(wp['wave_0_start']['price'], w1['price'], 0.618)
                setup['entry_zone'] = f"{fib_618:.2f}附近(浪1的61.8%反弹位)"
                setup['stop_loss'] = f"{wp['wave_0_start']['price']:.2f}(浪1起点上方)"
            setup['target'] = f"{fib.get('next_target', '?')}(浪3目标)"
            setup['rationale'] = '浪2反弹至0.5-0.618区域后可能开启浪3主跌'

        elif current_wave in ['浪3进行中', '浪3主跌'] and conf >= 0.5:
            setup['signal'] = 'bearish'
            setup['setup_type'] = '浪3追空'
            setup['entry_zone'] = f"{current_price:.2f}(当前价追空)"
            setup['stop_loss'] = f"{wp['wave_2_top']['price']:.2f}(浪2高点)" if wp.get('wave_2_top') else '浪2高点'
            setup['target'] = f"{fib.get('w5_target_2', '?')}(浪5目标)"

        elif current_wave in ['浪5完成/修正浪开始', 'impulse_5_complete'] and conf >= 0.5:
            setup['signal'] = 'bullish'
            setup['setup_type'] = '5浪下跌完成/ABC反弹'
            setup['entry_zone'] = f"突破{wp['wave_4_top']['price']:.2f}(浪4高点)确认反弹" if wp.get('wave_4_top') else '突破浪4高点'
            setup['stop_loss'] = f"{wp['wave_5_bottom']['price']:.2f}(浪5低点)" if wp.get('wave_5_bottom') else '浪5低点'
            setup['rationale'] = '5浪下跌完成后通常出现ABC反弹，目标至少38.2%反弹'

    return setup


def _empty_elliott_result():
    return {
        'wave_pattern': '数据不足',
        'current_wave': '未知',
        'wave_structure': '需要更多K线数据(≥60根)',
        'wave_points': {},
        'rules_valid': False,
        'rule_violations': [],
        'fib_ratios': {},
        'next_target': None,
        'next_support': None,
        'confidence': 0.0,
        'trade_setup': {'signal': 'neutral', 'setup_type': '数据不足', 'entry_zone': '', 'stop_loss': '', 'target': '', 'rationale': ''},
        'pivot_count': 0,
        'trend_direction': 'unknown'
    }


# ===================== 12. 已实现波动率估计 (Realized Volatility Estimators) =====================
# 基于 Andersen & Bollerslev (2003) 和 Corsi (2009)
# 从日线OHLC计算range-based波动率，无需高频tick数据

def parkinson_volatility(high, low, window=20):
    """
    Parkinson (1980) 波动率估计器
    仅用最高/最低价，效率是日收益平方的~5倍
    """
    ln_hl = np.log(high / low)
    pk = (1.0 / (4.0 * np.log(2.0))) * (ln_hl ** 2)
    return pd.Series(pk, index=high.index if hasattr(high, 'index') else None).rolling(window).mean() * np.sqrt(252)


def garman_klass_volatility(open_, high, low, close, window=20):
    """
    Garman-Klass (1980) 波动率估计器
    用OHLC四个价格，效率是日收益平方的~7.4倍
    """
    ln_hl = np.log(high / low)
    ln_co = np.log(close / open_)
    gk = 0.5 * ln_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * ln_co ** 2
    return pd.Series(gk, index=close.index if hasattr(close, 'index') else None).rolling(window).mean() * np.sqrt(252)


def rogers_satchell_volatility(open_, high, low, close, window=20):
    """
    Rogers-Satchell (1991) 波动率估计器
    允许漂移（非零均值），对趋势市更准确
    """
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    return pd.Series(rs, index=close.index if hasattr(close, 'index') else None).rolling(window).mean() * np.sqrt(252)


def yang_zhang_volatility(open_, high, low, close, window=20):
    """
    Yang-Zhang (2000) 波动率估计器
    无偏且独立于漂移和开盘跳空，处理隔夜缺口
    对A股跳空频繁的市场尤其适用
    """
    ln_co = np.log(close / open_)
    ln_oc = np.log(open_ / np.roll(close, 1))
    ln_oc.iloc[0] = 0

    # 隔夜波动率
    vo = np.var(ln_oc, ddof=1)
    # 开盘波动率
    vc = np.var(ln_co, ddof=1)
    # Parkinson分量
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    ln_hl = np.log(high / low)
    vrs = np.mean(ln_hl * (ln_hl - ln_co))

    yz = vo + k * vc + (1 - k) * vrs
    return pd.Series(np.full(len(close), yz), index=close.index if hasattr(close, 'index') else None) * np.sqrt(252)


def compute_realized_volatility(df, window=20):
    """
    综合已实现波动率分析
    返回多种range-based波动率估计及汇总
    """
    if len(df) < window + 1:
        return _empty_rv_result()

    open_ = df['open'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    # 各估计器序列
    pk_series = parkinson_volatility(high, low, window)
    gk_series = garman_klass_volatility(open_, high, low, close, window)
    rs_series = rogers_satchell_volatility(open_, high, low, close, window)
    yz_series = yang_zhang_volatility(open_, high, low, close, window)

    # 当前值
    pk_val = pk_series.iloc[-1] if not pd.isna(pk_series.iloc[-1]) else 0
    gk_val = gk_series.iloc[-1] if not pd.isna(gk_series.iloc[-1]) else 0
    yz_val = yz_series.iloc[-1] if not pd.isna(yz_series.iloc[-1]) else 0

    # 综合波动率估算（取PK和GK均值，稳健）
    rv_composite = (pk_val + gk_val) / 2.0

    # 近期趋势
    if len(pk_series) >= 10:
        pk_trend = "扩张" if pk_series.iloc[-1] > pk_series.iloc[-10] else "收缩"
    else:
        pk_trend = "未知"

    # 波动率水平
    if len(pk_series) >= 60:
        pk_percentile = (pk_series.iloc[-60:].values < pk_val).mean()
    else:
        pk_percentile = 0.5

    if pk_percentile > 0.85:
        vol_level = "极高波动"
    elif pk_percentile > 0.7:
        vol_level = "高波动"
    elif pk_percentile > 0.3:
        vol_level = "正常波动"
    elif pk_percentile > 0.15:
        vol_level = "低波动"
    else:
        vol_level = "极低波动"

    return {
        'rv_parkinson': round(pk_val, 4),
        'rv_garman_klass': round(gk_val, 4),
        'rv_yang_zhang': round(yz_val, 4),
        'rv_composite': round(rv_composite, 4),
        'rv_trend': pk_trend,
        'rv_percentile': round(pk_percentile, 3),
        'rv_level': vol_level
    }


# ===================== 13. HAR 异质自回归波动率预测 =====================
# 基于 Corsi (2009) "A Simple Approximate Long-Memory Model of Realized Volatility"
# HAR-RV: RV_{t+1} = β₀ + βd·RV_d + βw·RV_w + βm·RV_m

def _daily_rv_from_gk(open_, high, low, close):
    """从Garman-Klass估计器提取日频已实现波动率序列"""
    ln_hl = np.log(high / low)
    ln_co = np.log(close / open_)
    return 0.5 * ln_hl ** 2 - (2.0 * np.log(2.0) - 1.0) * ln_co ** 2


def har_model(df):
    """
    HAR-RV模型:用日/周/月三尺度波动率预测次日波动率
    返回预测值、方向和置信度
    """
    if len(df) < 60:
        return _empty_har_result()

    close = df['close'].astype(float).values
    open_ = df['open'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values

    # 日频已实现波动率序列
    rv_daily = _daily_rv_from_gk(open_, high, low, close) * 10000  # 放大以便回归

    # HAR三成分
    rv_d = rv_daily[-1]                    # 日度RV
    rv_w = np.mean(rv_daily[-5:])          # 周度RV（5日均值）
    rv_m = np.mean(rv_daily[-22:])         # 月度RV（22日均值）

    # 用OLS做滚动窗口回归（最近60天）
    X = []
    y = []
    for i in range(22, len(rv_daily) - 1):
        X.append([rv_daily[i], np.mean(rv_daily[i-4:i+1]), np.mean(rv_daily[i-21:i+1])])
        y.append(rv_daily[i + 1])

    if len(X) < 20:
        return _empty_har_result()

    X = np.array(X)
    y = np.array(y)
    try:
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]
        # 预测
        rv_forecast = beta[0] + beta[1] * rv_d + beta[2] * rv_w + beta[3] * rv_m
    except np.linalg.LinAlgError:
        rv_forecast = rv_d

    # 方向预测
    current_rv = rv_daily[-1]
    rv_direction = "扩张" if rv_forecast > current_rv else "收缩"

    # HAR R²评估
    y_pred = beta[0] + np.dot(X, beta[1:])
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = max(0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0

    # 衰减模式分析
    if beta[1] > beta[2] > beta[3] and all(b > 0 for b in beta[1:]):
        decay = "典型长记忆衰减(d>w>m)"
    elif beta[2] > beta[1]:
        decay = "中期波动主导"
    elif beta[3] > max(beta[1], beta[2]):
        decay = "长期波动主导"
    else:
        decay = "非标准衰减"

    return {
        'har_rv_daily': round(rv_d, 4),
        'har_rv_weekly': round(rv_w, 4),
        'har_rv_monthly': round(rv_m, 4),
        'har_forecast': round(rv_forecast, 4),
        'har_direction': rv_direction,
        'har_r_squared': round(r_squared, 3),
        'har_decay': decay,
        'har_beta_d': round(float(beta[1]), 4),
        'har_beta_w': round(float(beta[2]), 4),
        'har_beta_m': round(float(beta[3]), 4),
    }


# ===================== 14. 波动率交易信号 (Volatility Regime & Trading Signals) =====================
# 综合 RV估计 + HAR预测 + 波动率分位数 → 生成仓位/止损建议

def volatility_trading_signals(df, rv_result=None, har_result=None, bebe_result=None):
    """
    基于波动率的交易信号生成
    整合:波动率水平 + HAR预测方向 + BEGE好坏环境分解 + 波动率扩张/收缩 → 仓位管理和风控建议
    """
    if rv_result is None:
        rv_result = compute_realized_volatility(df)
    if har_result is None:
        har_result = har_model(df)
    if bebe_result is None:
        bebe_result = bebe_volatility_decomposition(df)

    if len(df) < 30:
        return _empty_vol_signal_result()

    close = df['close'].astype(float).values
    current_price = close[-1]

    rv_level = rv_result['rv_level']
    rv_composite = rv_result['rv_composite']
    rv_trend = rv_result['rv_trend']
    har_dir = har_result.get('har_direction', '未知')

    # ---- 波动率择时 ----
    # 波动率从低位开始扩张 → 趋势启动，适合趋势策略
    # 波动率从高位开始收缩 → 震荡蓄势，适合反转策略
    # 极高波动 → 风险规避，降低仓位
    # 极低波动 → 可能酝酿突破

    if rv_level in ["极高波动"] and rv_trend == "扩张":
        vol_signal = "risk_off"
        vol_advice = "极高波动扩张→避险模式"
    elif rv_level in ["极高波动"] and rv_trend == "收缩":
        vol_signal = "cautious_long"
        vol_advice = "极高波动收缩→恐慌消退，试探建仓"
    elif rv_level in ["低波动", "极低波动"] and rv_trend == "扩张":
        vol_signal = "breakout_ready"
        vol_advice = "低波动开始扩张→突破行情酝酿"
    elif rv_level in ["低波动", "极低波动"] and rv_trend == "收缩":
        vol_signal = "range_bound"
        vol_advice = "低波动持续收缩→窄幅震荡"
    elif rv_level in ["高波动"] and rv_trend == "扩张":
        vol_signal = "trend_follow"
        vol_advice = "高波动扩张→趋势跟踪"
    elif rv_level in ["高波动"] and rv_trend == "收缩":
        vol_signal = "trend_fade"
        vol_advice = "高波动收缩→趋势减弱"
    elif rv_level in ["正常波动"] and rv_trend == "扩张":
        vol_signal = "normal_trend"
        vol_advice = "正常波动扩张→常规趋势"
    else:
        vol_signal = "normal"
        vol_advice = "正常波动率环境"

    # ---- 动态仓位建议 ----
    # 基于波动率分位数调整仓位
    rv_pct = rv_result['rv_percentile']
    if rv_pct > 0.9:
        position_mult = 0.3
        position_advice = "低仓位(<30%)，波动率极高"
    elif rv_pct > 0.75:
        position_mult = 0.5
        position_advice = "半仓(50%)，波动率偏高"
    elif rv_pct > 0.4:
        position_mult = 0.7
        position_advice = "中等仓位(70%)，波动率正常"
    elif rv_pct > 0.2:
        position_mult = 0.85
        position_advice = "较高仓位(85%)，波动率偏低"
    else:
        position_mult = 1.0
        position_advice = "满仓(100%)，波动率极低"

    # HAR方向修正
    if har_dir == "扩张":
        position_mult *= 0.9  # 即将扩张波动，略减仓
    else:
        position_mult *= 1.05  # 即将收缩波动，可稍加仓

    # BEGE好坏环境修正
    bebe_regime = bebe_result.get('bebe_regime', 'unknown')
    bebe_vrp = bebe_result.get('bebe_vrp_signal', 'neutral')
    bebe_asym = bebe_result.get('bebe_vol_asymmetry', 0.5)

    if bebe_regime == "bad_environment":
        position_mult *= 0.75  # 坏环境主导→显著减仓
    elif bebe_regime == "good_environment":
        position_mult *= 1.05  # 好环境主导→可略加仓

    if bebe_vrp == "high_premium":
        position_mult *= 0.85  # VRP高→市场定价恐慌，谨慎
    elif bebe_vrp == "negative_premium":
        position_mult *= 0.8   # VRP负→异常信号，减仓观望

    # ---- 自适应止损 ----
    # 用ATR或波动率确定止损距离
    atr_arr = talib.ATR(df['high'].astype(float).values, df['low'].astype(float).values, close, 14)
    atr14 = float(atr_arr[-1]) if len(atr_arr) > 0 else current_price * 0.02
    atr_pct = atr14 / current_price if current_price > 0 else 0.02

    if rv_pct > 0.85:
        stop_mult = 2.5
    elif rv_pct > 0.6:
        stop_mult = 2.0
    elif rv_pct > 0.3:
        stop_mult = 1.5
    else:
        stop_mult = 1.2

    adaptive_stop_pct = round(atr_pct * stop_mult * 100, 2)
    adaptive_stop_price = round(current_price * (1 - adaptive_stop_pct / 100), 2)

    # ---- 波动率-趋势共振 ----
    # 低波动+波动即将扩张 + 趋势明确 = 最佳入场时机
    if rv_level in ["低波动", "极低波动"] and har_dir == "扩张":
        entry_quality = "高质量入场窗口"
    elif rv_level == "极高波动":
        entry_quality = "高风险区，不建议新开仓"
    elif rv_level in ["正常波动"] and rv_trend == "扩张":
        entry_quality = "可入场，注意仓位"
    else:
        entry_quality = "中性"

    # ---- BEGE增强:好坏环境对入场质量的影响 ----
    if bebe_regime == "bad_environment":
        entry_quality += "⚠坏环境主导"
    elif bebe_regime == "good_environment":
        entry_quality += "✓好环境主导"
    if bebe_vrp == "high_premium":
        entry_quality += "⚠高方差风险溢价"

    return {
        'vol_signal': vol_signal,
        'vol_advice': vol_advice,
        'vol_position_mult': round(min(1.0, position_mult), 2),
        'vol_position_advice': position_advice,
        'vol_adaptive_stop_pct': adaptive_stop_pct,
        'vol_adaptive_stop_price': adaptive_stop_price,
        'vol_entry_quality': entry_quality,
        'vol_atr14': round(float(atr14), 2),
        'vol_atr_pct': round(float(atr_pct * 100), 2),
        'vol_bebe_regime': bebe_regime,
        'vol_bebe_vrp': bebe_vrp,
        'vol_bebe_asymmetry': round(float(bebe_asym), 3),
    }


def _empty_rv_result():
    return {
        'rv_parkinson': 0, 'rv_garman_klass': 0, 'rv_yang_zhang': 0,
        'rv_composite': 0, 'rv_trend': '未知', 'rv_percentile': 0.5, 'rv_level': '未知'
    }


def _empty_har_result():
    return {
        'har_rv_daily': 0, 'har_rv_weekly': 0, 'har_rv_monthly': 0,
        'har_forecast': 0, 'har_direction': '未知', 'har_r_squared': 0,
        'har_decay': '未知', 'har_beta_d': 0, 'har_beta_w': 0, 'har_beta_m': 0
    }


# ===================== 13B. BEGE 好坏环境波动率分解 =====================
# 基于 NBER Working Paper w27108 — BEGE (Bad Environment, Good Environment) 模型
# 将总波动率分解为好环境（上行Gamma）和坏环境（下行Gamma）两个成分
# VRP (Variance Risk Premium) = 风险中性方差 - 物理方差
# 核心发现:坏波动率驱动VRP的程度超过物理方差；好波动率降低VRP

def bebe_volatility_decomposition(df, window=20):
    """
    BEGE波动率分解:上行半方差 vs 下行半方差
    - good_vol: 上行已实现半波动率（正收益贡献）
    - bad_vol:  下行已实现半波动率（负收益贡献）
    - VRP代理:  坏波动率与好波动率的相对增速差

    增强:融合σ-LSTM特征（Rodikov & Antulov-Fantulin, 2022）
    当df充足时自动计算LSTM遗忘门/细胞状态，增强环境判定
    """
    if len(df) < window + 5:
        result = _empty_bebe_result()
        result['bebe_sigma_lstm'] = _empty_sigma_lstm_result()
        return result

    close = df['close'].astype(float).values
    returns = np.diff(np.log(close))

    good_vols, bad_vols = [], []
    for i in range(window, len(returns) + 1):
        wr = returns[max(0, i - window):i]
        pos_r = wr[wr > 0]
        neg_r = wr[wr < 0]
        gv = np.sqrt(np.mean(pos_r ** 2) * 252) if len(pos_r) > 0 else 0.0
        bv = np.sqrt(np.mean(neg_r ** 2) * 252) if len(neg_r) > 0 else 0.0
        good_vols.append(gv)
        bad_vols.append(bv)

    cur_good = good_vols[-1]
    cur_bad = bad_vols[-1]
    ratio = cur_good / cur_bad if cur_bad > 1e-10 else 1.0
    total_vol = np.sqrt(cur_good ** 2 + cur_bad ** 2)

    # VRP代理:坏波动率增速 - 好波动率增速
    if len(good_vols) >= 6:
        gv_change = cur_good / max(np.mean(good_vols[-6:-1]), 1e-10) - 1.0
        bv_change = cur_bad / max(np.mean(bad_vols[-6:-1]), 1e-10) - 1.0
        vrp_proxy = bv_change - gv_change
    else:
        vrp_proxy = 0.0

    # σ-LSTM增强（Gelman BDA3 + Rodikov 2022）
    sigma_lstm = compute_sigma_lstm_volatility_features(df)

    # 环境判定（融合BEGE + LSTM）
    lstm_r = sigma_lstm.get('sigma_lstm_regime', 'no_data')
    if ratio > 1.3:
        base_regime = "good_environment"
    elif ratio < 0.7:
        base_regime = "bad_environment"
    else:
        base_regime = "balanced"

    if lstm_r in ('volatile_shock', 'regime_shift'):
        regime = f"{base_regime}_LSTM:{lstm_r}"
    elif lstm_r in ('stable_low_vol', 'stable_persistent'):
        regime = f"{base_regime}_LSTM:stable"
    else:
        regime = base_regime

    # 连续偏度度量
    bege_skew = np.log(ratio) if ratio > 0 else 0.0

    # VRP信号
    if vrp_proxy > 0.3:
        vrp_signal = "high_premium"
    elif vrp_proxy > 0.1:
        vrp_signal = "moderate_premium"
    elif vrp_proxy > -0.1:
        vrp_signal = "neutral"
    elif vrp_proxy > -0.3:
        vrp_signal = "low_premium"
    else:
        vrp_signal = "negative_premium"

    # 趋势
    if len(good_vols) >= 10:
        gv_trend = "rising" if cur_good > np.mean(good_vols[-6:-1]) else "falling"
        bv_trend = "rising" if cur_bad > np.mean(bad_vols[-6:-1]) else "falling"
        gv_trend_l = "rising" if np.mean(good_vols[-5:]) > np.mean(good_vols[-10:-5]) else "falling"
        bv_trend_l = "rising" if np.mean(bad_vols[-5:]) > np.mean(bad_vols[-10:-5]) else "falling"
    else:
        gv_trend = bv_trend = gv_trend_l = bv_trend_l = "stable"

    # 波动率不对称性
    asym = (cur_bad ** 2) / (cur_good ** 2 + cur_bad ** 2) if (cur_good ** 2 + cur_bad ** 2) > 0 else 0.5

    result = {
        'bebe_good_vol': round(cur_good, 4),
        'bebe_bad_vol': round(cur_bad, 4),
        'bebe_total_vol': round(total_vol, 4),
        'bebe_good_bad_ratio': round(ratio, 3),
        'bebe_regime': regime,
        'bebe_skew': round(bege_skew, 4),
        'bebe_vrp_proxy': round(vrp_proxy, 4),
        'bebe_vrp_signal': vrp_signal,
        'bebe_good_vol_trend': gv_trend,
        'bebe_bad_vol_trend': bv_trend,
        'bebe_good_vol_trend_long': gv_trend_l,
        'bebe_bad_vol_trend_long': bv_trend_l,
        'bebe_vol_asymmetry': round(asym, 3),
        'bebe_sigma_lstm': sigma_lstm,
    }
    return result


def _empty_bebe_result():
    return {
        'bebe_good_vol': 0, 'bebe_bad_vol': 0, 'bebe_total_vol': 0,
        'bebe_good_bad_ratio': 1.0, 'bebe_regime': 'unknown', 'bebe_skew': 0,
        'bebe_vrp_proxy': 0, 'bebe_vrp_signal': 'neutral',
        'bebe_good_vol_trend': 'stable', 'bebe_bad_vol_trend': 'stable',
        'bebe_good_vol_trend_long': 'stable', 'bebe_bad_vol_trend_long': 'stable',
        'bebe_vol_asymmetry': 0.5,
    }


def compute_sigma_lstm_volatility_features(df, lookback=50):
    """
    σ-LSTM波动率特征（Rodikov & Antulov-Fantulin, 2022）

    将GARCH(1,1)条件方差与LSTM遗忘门机制融合:
    - h_t = ω + α·r²_{t-1} + β·h_{t-1}  (GARCH条件方差)
    - f_t = sigmoid(r_t / √h_t)            (LSTM遗忘门代理)
    - c_t = (1-f_t)·h_t + f_t·c_{t-1}      (LSTM细胞状态代理)

    遗忘门f_t的含义:
    - f_t → 1: 当前收益与波动预期一致 → 保持记忆 → 稳定环境
    - f_t → 0: 当前收益超出波动预期 → 丢弃旧记忆 → 冲击/变盘

    Returns 5 key features + regime label
    """
    if len(df) < lookback:
        return _empty_sigma_lstm_result()

    close = df['close'].astype(float).values
    returns = np.diff(np.log(close))[-lookback:]
    ret = returns * 100  # 百分比收益率

    # GARCH(1,1) 参数估计（方法矩近似 + scipy MLE）
    omega, alpha, beta = _fit_garch_mle(ret)

    # 计算条件方差序列
    h = np.zeros(len(ret))
    h[0] = np.var(ret[:20]) if len(ret) >= 20 else np.var(ret)
    for t in range(1, len(ret)):
        h[t] = omega + alpha * ret[t-1]**2 + beta * h[t-1]
    h = np.maximum(h, 1e-10)

    # LSTM遗忘门代理
    f = 1.0 / (1.0 + np.exp(-ret / np.sqrt(h)))  # sigmoid(r_t / √h_t)
    f = np.clip(f, 1e-6, 1.0 - 1e-6)

    # LSTM细胞状态代理
    c = np.zeros(len(ret))
    c[0] = h[0]
    for t in range(1, len(ret)):
        c[t] = (1.0 - f[t]) * h[t] + f[t] * c[t-1]

    cur_h = float(h[-1])
    cur_c = float(c[-1])
    cur_f = float(f[-1])
    mean_f = float(np.mean(f[-20:]))

    # 遗忘门趋势
    f_short = np.mean(f[-5:])
    f_long = np.mean(f[-20:])
    if f_short < 0.3:
        f_state = "rapid_forgetting"
    elif f_short < 0.5:
        f_state = "moderate_forgetting"
    elif f_short < 0.7:
        f_state = "normal"
    else:
        f_state = "strong_memory"

    # 环境判定（融合GARCH+LSTM）
    garch_vol = np.sqrt(omega / max(1.0 - alpha - beta, 0.01))
    h_ratio = cur_h / max(np.mean(h[-20:]), 1e-10)
    c_h_gap = (cur_c - cur_h) / max(cur_h, 1e-10)

    if h_ratio > 1.5 and f_state in ("rapid_forgetting", "moderate_forgetting"):
        lstm_regime = "volatile_shock"
    elif f_state == "rapid_forgetting" and c_h_gap > 0.3:
        lstm_regime = "regime_shift"
    elif f_state in ("normal", "strong_memory") and h_ratio < 0.8:
        lstm_regime = "stable_low_vol"
    elif f_state == "strong_memory":
        lstm_regime = "stable_persistent"
    else:
        lstm_regime = "transitional"

    return {
        'sigma_lstm_cell': round(cur_c, 6),
        'sigma_lstm_hidden': round(cur_h, 6),
        'sigma_lstm_forget': round(cur_f, 4),
        'sigma_lstm_forget_mean': round(mean_f, 4),
        'sigma_lstm_regime': lstm_regime,
        'sigma_lstm_garch_omega': round(omega, 6),
        'sigma_lstm_garch_alpha': round(alpha, 4),
        'sigma_lstm_garch_beta': round(beta, 4),
        'sigma_lstm_h_ratio': round(h_ratio, 3),
        'sigma_lstm_c_h_gap': round(c_h_gap, 3),
    }


def _fit_garch_mle(ret):
    """GARCH(1,1) MLE拟合（带方法矩回退）"""
    try:
        from scipy.optimize import minimize

        def garch_loglik(params):
            omega, alpha, beta = params
            if omega <= 1e-8 or alpha <= 0 or beta <= 0 or alpha + beta >= 0.999:
                return 1e15
            h = np.zeros(len(ret))
            h[0] = np.var(ret)
            for t in range(1, len(ret)):
                h[t] = omega + alpha * ret[t-1]**2 + beta * h[t-1]
            return np.sum(np.log(h) + ret**2 / h)

        var0 = np.var(ret)
        init = [var0 * 0.1, 0.1, 0.8]
        bounds = [(1e-8, None), (1e-6, 0.5), (1e-6, 0.98)]
        result = minimize(garch_loglik, init, bounds=bounds, method='L-BFGS-B')
        if result.success:
            return result.x[0], result.x[1], result.x[2]
    except Exception:
        pass

    # 方法矩回退
    var0 = np.var(ret)
    acf1 = np.corrcoef(ret[1:]**2, ret[:-1]**2)[0, 1] if len(ret) > 2 else 0.1
    beta = max(0.5, min(0.9, acf1))
    alpha = 0.05
    omega = var0 * (1.0 - alpha - beta)
    return omega, alpha, beta


def _empty_sigma_lstm_result():
    return {
        'sigma_lstm_cell': 0, 'sigma_lstm_hidden': 0,
        'sigma_lstm_forget': 0.5, 'sigma_lstm_forget_mean': 0.5,
        'sigma_lstm_regime': 'no_data',
        'sigma_lstm_garch_omega': 0, 'sigma_lstm_garch_alpha': 0,
        'sigma_lstm_garch_beta': 0, 'sigma_lstm_h_ratio': 1.0,
        'sigma_lstm_c_h_gap': 0,
    }


def _empty_vol_signal_result():
    return {
        'vol_signal': 'normal', 'vol_advice': '数据不足', 'vol_position_mult': 0.5,
        'vol_position_advice': '数据不足建议轻仓', 'vol_adaptive_stop_pct': 2.0,
        'vol_adaptive_stop_price': 0, 'vol_entry_quality': '未知',
        'vol_atr14': 0, 'vol_atr_pct': 0,
        'vol_bebe_regime': 'unknown', 'vol_bebe_vrp': 'neutral', 'vol_bebe_asymmetry': 0.5,
    }


# ===================== 15. 贝叶斯多信号置信度融合 =====================
# 基于贝叶斯定理，将多个独立信号维度的证据融合为单一后验概率
# 实现:似然比乘法 → log-odds加法 → 后验概率

# 全局校准缓存:从历史数据学习到的经验似然比
# 格式: {dim_name: {'bullish': LLR, 'bearish': LLR, 'neutral': LLR, 'n_samples': N}}
_EMPIRICAL_LLR_CACHE = None


def calibrate_bayesian_likelihoods(df_daily, forward_days=5, symbol=None):
    """
    从历史数据中校准贝叶斯似然比
    对每个维度，计算:LLR(state) = log(P(state | forward_up) / P(state | forward_down))

    方法:
    1. 对历史每一天计算指标快照（滑动窗口）
    2. 标记forward N天的实际涨跌
    3. 对每个维度，统计信号状态×涨跌的列联表
    4. 计算经验似然比

    缓存策略（三级）:
    1. 会话级内存缓存 → 2. SQLite持久化缓存 → 3. 重新计算

    返回校准字典，可直接传入 multi_signal_bayesian_fusion 的 calibration 参数
    """
    global _EMPIRICAL_LLR_CACHE

    # 一级:内存缓存
    if _EMPIRICAL_LLR_CACHE is not None:
        return _EMPIRICAL_LLR_CACHE

    # 二级:SQLite持久化缓存
    if symbol is not None:
        try:
            from trade_logger import load_bayesian_calibration
            cached = load_bayesian_calibration(symbol)
            if cached and cached.get('n_samples', 0) >= 50:
                _EMPIRICAL_LLR_CACHE = cached
                return cached
        except Exception:
            pass

    if df_daily is None or len(df_daily) < 80:
        return None

    close = df_daily['close'].astype(float).values
    n = len(close)

    # 为每个历史日期收集信号状态和forward label
    records = {dim: {'bullish': {'up': 0, 'down': 0}, 'bearish': {'up': 0, 'down': 0},
                     'neutral': {'up': 0, 'down': 0}}
               for dim in ['chanlun', 'elliott', 'tang', 'livermore', 'busch', 'classical', 'risk']}

    # 在历史数据上滑动计算
    for t in range(60, n - forward_days - 1, 5):  # 每5天一个样本，减少计算量
        window = df_daily.iloc[t-60:t+1].copy()
        if len(window) < 30:
            continue

        # forward label: 未来N天的涨跌
        forward_return = (close[min(t + forward_days, n - 1)] - close[t]) / close[t]
        label = 'up' if forward_return > 0 else 'down'
        if abs(forward_return) < 0.005:  # 太小忽略
            continue

        # 提取各维度信号状态
        try:
            sig_states = _extract_signal_states_from_window(window)
            for dim, state in sig_states.items():
                if state in records[dim]:
                    records[dim][state][label] += 1
        except Exception:
            continue

    # 计算经验LLR（使用classical维度作为基准total_up/down）
    calibration = {}
    total_up = sum(records['classical'][s]['up'] for s in ['bullish', 'bearish', 'neutral'])
    total_down = sum(records['classical'][s]['down'] for s in ['bullish', 'bearish', 'neutral'])

    if total_up < 10 or total_down < 10:
        return None  # 数据不足，用启发式

    for dim, states in records.items():
        dim_llrs = {}
        for state in ['bullish', 'bearish', 'neutral']:
            n_up = states[state]['up']
            n_down = states[state]['down']
            # 拉普拉斯平滑
            p_state_given_up = (n_up + 1) / (total_up + 3)
            p_state_given_down = (n_down + 1) / (total_down + 3)
            if p_state_given_down > 0 and p_state_given_up > 0:
                llr = np.log(p_state_given_up / p_state_given_down)
            else:
                llr = 0.0
            dim_llrs[state] = round(llr, 4)
        dim_llrs['n_samples'] = total_up + total_down
        calibration[dim] = dim_llrs

    _EMPIRICAL_LLR_CACHE = calibration

    # 持久化到SQLite（三级缓存写入）
    if symbol is not None:
        try:
            from trade_logger import save_bayesian_calibration
            save_bayesian_calibration(symbol, calibration, total_up + total_down)
        except Exception:
            pass

    return calibration


def _extract_signal_states_from_window(window_df):
    """从历史窗口中提取15大学派的离散状态（与_current_dimension_states对齐）"""
    close_w = window_df['close'].astype(float).values
    high_w = window_df['high'].astype(float).values
    low_w = window_df['low'].astype(float).values
    vol_w = window_df['volume'].astype(float).values
    n = len(close_w)

    states = {}

    # 1. 缠论: 尝试运行chanlun_analysis（可能较慢，回退到简化版）
    try:
        from advanced_indicators import chanlun_analysis
        chan = chanlun_analysis(window_df)
        chan_bull = len(chan.get('chanlun_buy', []) or [])
        chan_bear = len(chan.get('chanlun_sell', []) or [])
        stroke_st = str(chan.get('chanlun_stroke_state', ''))
        div_t = str(chan.get('chanlun_divergence_type', ''))
        if '(1,1)' in stroke_st or '(1,0)' in stroke_st: chan_bull += 1
        elif '(-1,1)' in stroke_st or '(-1,0)' in stroke_st: chan_bear += 1
        if '底背驰' in div_t: chan_bull += 2
        if '顶背驰' in div_t: chan_bear += 2
        states['chanlun'] = 'bullish' if chan_bull > chan_bear else ('bearish' if chan_bear > chan_bull else 'neutral')
    except Exception:
        states['chanlun'] = 'neutral'

    # 2. 艾略特波浪
    if n >= 60:
        try:
            ew_res = elliott_wave_analysis(window_df)
            if ew_res['trade_setup']['signal'] == 'bullish':
                states['elliott'] = 'bullish'
            elif ew_res['trade_setup']['signal'] == 'bearish':
                states['elliott'] = 'bearish'
            else:
                states['elliott'] = 'neutral'
        except Exception:
            states['elliott'] = 'neutral'
    else:
        states['elliott'] = 'neutral'

    # 3. 唐能通: 价托价压
    if n >= 60:
        try:
            tang_jt = tang_jiato_jiaya(window_df)
            tang_triple = tang_sanjincha_sansicha(window_df)
            tang_lyt = tang_laoyatou(window_df)
            tang_bull = 1 if tang_jt.get('tang_jiato', False) else 0
            tang_bear = 1 if tang_jt.get('tang_jiaya', False) else 0
            triple = str(tang_triple.get('tang_triple_status', ''))
            if 'golden' in triple: tang_bull += 2
            if 'death' in triple: tang_bear += 2
            if tang_lyt.get('tang_laoyatou', False) and tang_lyt.get('tang_laoyatou_score', 0) > 50:
                tang_bull += 1
            states['tang'] = 'bullish' if tang_bull > tang_bear else ('bearish' if tang_bear > tang_bull else 'neutral')
        except Exception:
            states['tang'] = 'neutral'
    else:
        states['tang'] = 'neutral'

    # 4. 利弗莫尔
    if n >= 60:
        try:
            liv = livermore_pivotal_points(window_df)
            danger = livermore_danger_signal(window_df)
            if danger.get('livermore_danger', False) and danger.get('livermore_danger_level', 0) >= 50:
                states['livermore'] = 'bearish'
            elif 'strong_buy' in str(liv.get('livermore_signal', '')):
                states['livermore'] = 'bullish'
            elif 'strong_sell' in str(liv.get('livermore_signal', '')):
                states['livermore'] = 'bearish'
            else:
                states['livermore'] = 'neutral'
        except Exception:
            states['livermore'] = 'neutral'
    else:
        states['livermore'] = 'neutral'

    # 5. Busch量化
    if n >= 60:
        try:
            b2560 = busch_2560_strategy(window_df)
            vpa = volume_price_algebra(window_df)
            cva = comprehensive_volume_analysis(window_df)
            b = str(b2560.get('b2560_signal', ''))
            busch_bull = 1 if b in ('strong_buy', 'hold_long') else 0
            busch_bear = 1 if b in ('sell', 'hold_short') else 0
            if 'bullish' in str(vpa.get('vpa_signal', '')): busch_bull += 1
            if 'bearish' in str(vpa.get('vpa_signal', '')): busch_bear += 1
            if '看多' in str(cva.get('cva_composite_signal', '')): busch_bull += 1
            if '看空' in str(cva.get('cva_composite_signal', '')): busch_bear += 1
            states['busch'] = 'bullish' if busch_bull > busch_bear else ('bearish' if busch_bear > busch_bull else 'neutral')
        except Exception:
            states['busch'] = 'neutral'
    else:
        states['busch'] = 'neutral'

    # 6. 经典技术分析
    cla_bull = 0
    cla_bear = 0
    if n >= 60:
        ma5 = talib.SMA(close_w, 5)[-1]
        ma20 = talib.SMA(close_w, 20)[-1]
        ma60 = talib.SMA(close_w, 60)[-1]
        if ma5 > ma20 > ma60: cla_bull += 2
        elif ma5 < ma20 < ma60: cla_bear += 2
        elif ma5 > ma20: cla_bull += 1
        elif ma5 < ma20: cla_bear += 1
    if n >= 26:
        macd_dif, macd_dea, _ = talib.MACD(close_w)
        if macd_dif[-1] > macd_dea[-1]: cla_bull += 1
        else: cla_bear += 1
    if n >= 14:
        rsi = talib.RSI(close_w, 14)[-1]
        if rsi > 60: cla_bull += 1
        elif rsi < 40: cla_bear += 1
    if n >= 20:
        pat = candlestick_patterns(window_df)
        bull_pat = pat['counts'].get('morning_star', 0) + pat['counts'].get('bottom_fractal', 0)
        bear_pat = pat['counts'].get('evening_star', 0) + pat['counts'].get('top_fractal', 0)
        cla_bull += bull_pat
        cla_bear += bear_pat

    if cla_bull > cla_bear:
        states['classical'] = 'bullish'
    elif cla_bear > cla_bull:
        states['classical'] = 'bearish'
    else:
        states['classical'] = 'neutral'

    # 7. 风险环境
    risk_bull = 0
    risk_bear = 0
    if n >= 20:
        rv = compute_realized_volatility(window_df)
        if rv['rv_level'] in ['极低波动', '低波动']: risk_bull += 1
        elif rv['rv_level'] == '极高波动': risk_bear += 1
    if n >= 60:
        try:
            mp = market_phase_detection(window_df)
            mph = str(mp.get('mp_phase', ''))
            if mph == '拉升': risk_bull += 1
            elif mph in ['盘头', '下跌']: risk_bear += 1
        except Exception:
            pass
        try:
            bebe = bebe_volatility_decomposition(window_df)
            if 'bad_environment' in str(bebe.get('bebe_regime', '')): risk_bear += 1
            elif 'good_environment' in str(bebe.get('bebe_regime', '')): risk_bull += 1
        except Exception:
            pass
        try:
            top = top_escape_signals(window_df)
            if str(top.get('top_escape_grade', '')) in ['critical', 'high_risk']:
                risk_bear += 2
        except Exception:
            pass

    if risk_bull > risk_bear:
        states['risk'] = 'bullish'
    elif risk_bear > risk_bull:
        states['risk'] = 'bearish'
    else:
        states['risk'] = 'neutral'

    # 8. 江恩理论（历史窗口）
    gann_bull = 0
    gann_bear = 0
    if n >= 20:
        adx_w = talib.ADX(high_w, low_w, close_w, 14)[-1]
        if adx_w < 18:
            gann_bull = max(0, gann_bull - 1)
            gann_bear = max(0, gann_bear - 1)
    if n >= 60:
        ma5_w = talib.SMA(close_w, 5)[-1]
        ma20_w = talib.SMA(close_w, 20)[-1]
        ma60_w = talib.SMA(close_w, 60)[-1]
        if ma5_w > ma20_w > ma60_w:
            gann_bull += 2
        elif ma5_w < ma20_w < ma60_w:
            gann_bear += 2
        elif ma5_w > ma20_w:
            gann_bull += 1
        elif ma5_w < ma20_w:
            gann_bear += 1
    if n >= 14:
        rsi_w = talib.RSI(close_w, 14)[-1]
        if rsi_w < 30:
            gann_bull += 1
        elif rsi_w > 70:
            gann_bear += 1
    if n >= 20:
        pat_w = candlestick_patterns(window_df)
        bot_n = pat_w['counts'].get('bottom_fractal', 0)
        top_n = pat_w['counts'].get('top_fractal', 0)
        if bot_n >= 2:
            gann_bull += 2
        elif bot_n == 1:
            gann_bull += 1
        if top_n >= 2:
            gann_bear += 2
        elif top_n == 1:
            gann_bear += 1
    if n >= 60:
        try:
            mp_w = market_phase_detection(window_df)
            mph_w = str(mp_w.get('mp_phase', ''))
            if mph_w == '拉升':
                gann_bull += 1
            elif mph_w in ('盘头', '下跌'):
                gann_bear += 1
        except Exception:
            pass

    if gann_bull > gann_bear:
        states['gann'] = 'bullish'
    elif gann_bear > gann_bull:
        states['gann'] = 'bearish'
    else:
        states['gann'] = 'neutral'

    # 9. 威科夫量价（历史窗口）
    wyck_bull = 0
    wyck_bear = 0
    if n >= 20:
        obv_w = talib.OBV(close_w, vol_w)
        if n >= 5 and obv_w[-1] > obv_w[-5]:
            wyck_bull += 1
        elif n >= 5:
            wyck_bear += 1
        rsi_w2 = talib.RSI(close_w, 14)[-1]
        if rsi_w2 < 35:
            wyck_bull += 1
        elif rsi_w2 > 65:
            wyck_bear += 1
    if n >= 6:
        ret_5d_w = (close_w[-1] - close_w[-6]) / close_w[-6]
        avg_vol_5d_w = np.mean(vol_w[-5:])
        avg_vol_20d_w = np.mean(vol_w[-20:])
        vr_w = avg_vol_5d_w / avg_vol_20d_w if avg_vol_20d_w > 0 else 1.0
        if vr_w > 1.5 and abs(ret_5d_w) < 0.02:
            if close_w[-1] > np.mean(close_w[-20:]):
                wyck_bear += 2
            else:
                wyck_bull += 2
        elif vr_w < 0.6 and abs(ret_5d_w) < 0.01:
            wyck_bull += 1
    if n >= 20:
        pat2_w = candlestick_patterns(window_df)
        bot2_n = pat2_w['counts'].get('bottom_fractal', 0)
        top2_n = pat2_w['counts'].get('top_fractal', 0)
        if bot2_n >= 1:
            wyck_bull += 1
        if top2_n >= 1:
            wyck_bear += 1

    if wyck_bull > wyck_bear:
        states['wyckoff'] = 'bullish'
    elif wyck_bear > wyck_bull:
        states['wyckoff'] = 'bearish'
    else:
        states['wyckoff'] = 'neutral'

    # 10. 道氏理论（历史窗口）
    dow_bull = 0; dow_bear = 0
    if n >= 60:
        ma60_w = talib.SMA(close_w, 60)[-1]
        ma120_w = talib.SMA(close_w, 120)[-1] if n >= 120 else ma60_w
        ma20_w_d = talib.SMA(close_w, 20)[-1]
        if ma20_w_d > ma60_w > ma120_w: dow_bull += 2
        elif ma20_w_d < ma60_w < ma120_w: dow_bear += 2
    if n >= 20:
        ret_60d = (close_w[-1] - close_w[-min(60,n)]) / close_w[-min(60,n)] if len(close_w) > 1 else 0
        if ret_60d > 0.05: dow_bull += 1
        elif ret_60d < -0.05: dow_bear += 1
        adx_w_d = talib.ADX(high_w, low_w, close_w, 14)[-1]
        if adx_w_d < 18: dow_bull = max(0, dow_bull-1); dow_bear = max(0, dow_bear-1)
    states['dow'] = 'bullish' if dow_bull > dow_bear else ('bearish' if dow_bear > dow_bull else 'neutral')

    # 11. CANSLIM（历史窗口）
    can_bull = 0; can_bear = 0
    if n >= 14:
        rsi_c = talib.RSI(close_w, 14)[-1]
        roc_l_c = (close_w[-1] - close_w[-min(60,n)]) / close_w[-min(60,n)] if len(close_w) > 1 else 0
        if roc_l_c > 0.05: can_bull += 2
        elif roc_l_c < -0.05: can_bear += 1
        if rsi_c > 55: can_bull += 1
        elif rsi_c < 45: can_bear += 1
    if n >= 20:
        avg_vol_5 = np.mean(vol_w[-5:]); avg_vol_20 = np.mean(vol_w[-20:])
        if avg_vol_5 > avg_vol_20 * 1.3: can_bull += 1
    states['canslim'] = 'bullish' if can_bull > can_bear else ('bearish' if can_bear > can_bull else 'neutral')

    # 12. 海龟交易（历史窗口）
    turt_bull = 0; turt_bear = 0
    if n >= 20:
        hh20 = np.max(high_w[-20:]); ll20 = np.min(low_w[-20:])
        if close_w[-1] >= hh20 * 0.98: turt_bull += 2
        elif close_w[-1] <= ll20 * 1.02: turt_bear += 1
    if n >= 60:
        ma20_t = talib.SMA(close_w, 20)[-1]; ma60_t = talib.SMA(close_w, 60)[-1]
        if ma20_t > ma60_t: turt_bull += 1
        else: turt_bear += 1
    if n >= 14:
        adx_t = talib.ADX(high_w, low_w, close_w, 14)[-1]
        if adx_t < 18: turt_bull = max(0, turt_bull-1); turt_bear = max(0, turt_bear-1)
    states['turtle'] = 'bullish' if turt_bull > turt_bear else ('bearish' if turt_bear > turt_bull else 'neutral')

    # 13. 谐波形态（历史窗口）
    harm_bull = 0; harm_bear = 0
    if n >= 14:
        rsi_h = talib.RSI(close_w, 14)[-1]
        if rsi_h < 30: harm_bull += 2
        elif rsi_h > 70: harm_bear += 2
    if n >= 20:
        pat_h = candlestick_patterns(window_df)
        if pat_h['counts'].get('bottom_fractal', 0) > 0: harm_bull += 1
        if pat_h['counts'].get('top_fractal', 0) > 0: harm_bear += 1
        if pat_h['counts'].get('morning_star', 0) > 0: harm_bull += 1
        if pat_h['counts'].get('evening_star', 0) > 0: harm_bear += 1
    states['harmonic'] = 'bullish' if harm_bull > harm_bear else ('bearish' if harm_bear > harm_bull else 'neutral')

    # 14. 市场轮廓（历史窗口）
    mpf_bull = 0; mpf_bear = 0
    if n >= 20:
        bb_m_mp = talib.SMA(close_w, 20)[-1]
        std_mp = np.std(close_w[-20:])
        bb_u_mp = bb_m_mp + 2*std_mp; bb_l_mp = bb_m_mp - 2*std_mp
        if bb_u_mp > bb_l_mp:
            pos = (close_w[-1] - bb_l_mp) / (bb_u_mp - bb_l_mp)
            if pos < 0.2: mpf_bull += 2
            elif pos > 0.8: mpf_bear += 2
    if n >= 5:
        ret5_mp = (close_w[-1] - close_w[-5]) / close_w[-5] if n >= 5 else 0
        if ret5_mp > 0.02: mpf_bull += 1
        elif ret5_mp < -0.02: mpf_bear += 1
    states['marketprofile'] = 'bullish' if mpf_bull > mpf_bear else ('bearish' if mpf_bear > mpf_bull else 'neutral')

    # 15. 混沌交易法（历史窗口）
    chao_bull = 0; chao_bear = 0
    if n >= 60:
        ma5_c = talib.SMA(close_w, 5)[-1]; ma20_c = talib.SMA(close_w, 20)[-1]; ma60_c = talib.SMA(close_w, 60)[-1]
        if ma5_c > ma20_c > ma60_c: chao_bull += 2
        elif ma5_c < ma20_c < ma60_c: chao_bear += 2
    if n >= 20:
        pat_c = candlestick_patterns(window_df)
        if pat_c['counts'].get('bottom_fractal', 0) >= 2: chao_bull += 2
        if pat_c['counts'].get('top_fractal', 0) >= 2: chao_bear += 2
    if n >= 14:
        macd_dif_c, macd_dea_c, _ = talib.MACD(close_w)
        if macd_dif_c[-1] > macd_dea_c[-1]: chao_bull += 1
        else: chao_bear += 1
    if n >= 14:
        adx_c = talib.ADX(high_w, low_w, close_w, 14)[-1]
        if adx_c < 18: chao_bull = max(0, chao_bull-1); chao_bear = max(0, chao_bear-1)
    states['chaos'] = 'bullish' if chao_bull > chao_bear else ('bearish' if chao_bear > chao_bull else 'neutral')

    return states


def _heuristic_llr(score, scale=1.8):
    """简单启发式LLR映射（无校准数据时的回退方案）"""
    return scale * np.tanh(score * 2.5)


def _current_dimension_states(indicators):
    """
    从已计算的指标字典中提取15大学派的离散状态（bullish/bearish/neutral）
    状态离散化标准与 _extract_signal_states_from_window 保持一致
    """
    states = {}

    # 1. 缠论: 综合买卖点+笔状态+背驰+拐点评分
    chan_buy = indicators.get('chanlun_buy', [])
    chan_sell = indicators.get('chanlun_sell', [])
    stroke_st = str(indicators.get('chanlun_stroke_state', ''))
    div_type = str(indicators.get('chanlun_divergence_type', ''))
    cl_turn_sig = str(indicators.get('cl_turn_signal', ''))

    chan_bull = len(chan_buy) if chan_buy else 0
    chan_bear = len(chan_sell) if chan_sell else 0
    if '(1,1)' in stroke_st or '(1,0)' in stroke_st:
        chan_bull += 1
    elif '(-1,1)' in stroke_st or '(-1,0)' in stroke_st:
        chan_bear += 1
    if '底背驰' in div_type: chan_bull += 2
    if '顶背驰' in div_type: chan_bear += 2
    if 'strong_buy' in cl_turn_sig: chan_bull += 1
    if 'strong_sell' in cl_turn_sig: chan_bear += 1

    if chan_bull > chan_bear:
        states['chanlun'] = 'bullish'
    elif chan_bear > chan_bull:
        states['chanlun'] = 'bearish'
    else:
        states['chanlun'] = 'neutral'

    # 2. 艾略特波浪
    ew_signal = indicators.get('ew_trade_signal', 'neutral')
    ew_conf = indicators.get('ew_confidence', 0)
    if ew_signal == 'bullish' and ew_conf >= 0.4:
        states['elliott'] = 'bullish'
    elif ew_signal == 'bearish' and ew_conf >= 0.4:
        states['elliott'] = 'bearish'
    else:
        states['elliott'] = 'neutral'

    # 3. 唐能通: 三金叉+价托价压+老鸭头
    tang_triple = str(indicators.get('tang_triple_status', ''))
    tang_jt = indicators.get('tang_jiato', False)
    tang_jy = indicators.get('tang_jiaya', False)
    tang_lyt = indicators.get('tang_laoyatou', False)
    tang_lyt_score = _safe(indicators.get('tang_laoyatou_score'), 0)

    tang_bull = 1 if 'golden' in tang_triple else 0
    tang_bear = 1 if 'death' in tang_triple else 0
    if tang_jt: tang_bull += 1
    if tang_jy: tang_bear += 1
    if tang_lyt and tang_lyt_score > 50: tang_bull += 1

    if tang_bull > tang_bear:
        states['tang'] = 'bullish'
    elif tang_bear > tang_bull:
        states['tang'] = 'bearish'
    else:
        states['tang'] = 'neutral'

    # 4. 利弗莫尔
    liv_sig = str(indicators.get('livermore_signal', ''))
    liv_danger = indicators.get('livermore_danger', False)
    liv_dl = _safe(indicators.get('livermore_danger_level'), 0)
    if liv_danger and liv_dl >= 50:
        states['livermore'] = 'bearish'
    elif 'strong_buy' in liv_sig or liv_sig == 'buy':
        states['livermore'] = 'bullish'
    elif 'strong_sell' in liv_sig or liv_sig == 'sell':
        states['livermore'] = 'bearish'
    else:
        states['livermore'] = 'neutral'

    # 5. Busch量化: 2560 VPA CVA
    b2560 = str(indicators.get('b2560_signal', ''))
    vpa_sig = str(indicators.get('vpa_signal', ''))
    cva = str(indicators.get('cva_composite_signal', ''))

    busch_bull = 1 if b2560 in ('strong_buy', 'hold_long') else 0
    busch_bear = 1 if b2560 in ('sell', 'hold_short') else 0
    if 'bullish' in vpa_sig: busch_bull += 1
    if 'bearish' in vpa_sig: busch_bear += 1
    if '看多' in cva: busch_bull += 1
    if '看空' in cva: busch_bear += 1

    if busch_bull > busch_bear:
        states['busch'] = 'bullish'
    elif busch_bear > busch_bull:
        states['busch'] = 'bearish'
    else:
        states['busch'] = 'neutral'

    # 6. 经典技术分析: MA+MACD+RSI+形态+OBV+DMA+DMI综合
    ma5 = indicators.get('ma5', 0)
    ma20 = indicators.get('ma20', 0)
    ma60 = indicators.get('ma60', 0)
    macd_dif = _safe(indicators.get('macd_dif'))
    macd_dea = _safe(indicators.get('macd_dea'))
    rsi = _safe(indicators.get('rsi'), 50)
    pat_rec = str(indicators.get('pattern_recent', ''))
    obv_div = str(indicators.get('obv_divergence', ''))

    cla_bull = 0
    cla_bear = 0
    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60: cla_bull += 2
        elif ma5 < ma20 < ma60: cla_bear += 2
        elif ma5 > ma20: cla_bull += 1
        elif ma5 < ma20: cla_bear += 1
    if macd_dif > macd_dea: cla_bull += 1
    else: cla_bear += 1
    if rsi > 60: cla_bull += 1
    elif rsi < 40: cla_bear += 1
    bull_pat = sum(1 for k in ['底分型', '晨星', '吞没看涨', '刺透'] if k in pat_rec)
    bear_pat = sum(1 for k in ['顶分型', '暮星', '吞没看跌', '乌云盖顶'] if k in pat_rec)
    cla_bull += bull_pat
    cla_bear += bear_pat
    if '底背离' in obv_div: cla_bull += 1
    if '顶背离' in obv_div: cla_bear += 1

    if cla_bull > cla_bear:
        states['classical'] = 'bullish'
    elif cla_bear > cla_bull:
        states['classical'] = 'bearish'
    else:
        states['classical'] = 'neutral'

    # 7. 风险环境
    rv_level = indicators.get('rv_level', '')
    har_dir = indicators.get('har_direction', '')
    top_grade = str(indicators.get('top_escape_grade', ''))
    mphase = str(indicators.get('mp_phase', ''))
    bebe_reg = str(indicators.get('bebe_regime', ''))

    risk_bull = 0
    risk_bear = 0
    if rv_level in ['极低波动', '低波动']: risk_bull += 1
    elif rv_level == '极高波动': risk_bear += 1
    if mphase == '拉升': risk_bull += 1
    elif mphase in ['盘头', '下跌']: risk_bear += 1
    if top_grade in ['critical', 'high_risk']: risk_bear += 2
    if 'bad_environment' in bebe_reg: risk_bear += 1
    elif 'good_environment' in bebe_reg: risk_bull += 1

    if risk_bull > risk_bear:
        states['risk'] = 'bullish'
    elif risk_bear > risk_bull:
        states['risk'] = 'bearish'
    else:
        states['risk'] = 'neutral'

    # 8. 江恩理论: 趋势+百分位回调+时间窗口+共振
    ma5_v = _safe(indicators.get('ma5'), 0)
    ma20_v = _safe(indicators.get('ma20'), 0)
    ma60_v = _safe(indicators.get('ma60'), 0)
    adx = _safe(indicators.get('dmi_adx'), 20)
    rsi_g = _safe(indicators.get('rsi'), 50)
    obv_div_g = str(indicators.get('obv_divergence', ''))
    ver_sup_g = indicators.get('verified_support', [])
    ver_res_g = indicators.get('verified_resistance', [])
    pat_bot_n_g = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    pat_top_n_g = _safe(indicators.get('pattern_top_fractal_n'), 0)
    mp_phase_g = str(indicators.get('mp_phase', ''))

    gann_bull = 0
    gann_bear = 0
    if ma5_v > ma20_v > ma60_v and ma20_v > 0:
        gann_bull += 2
    elif ma5_v < ma20_v < ma60_v and ma20_v > 0:
        gann_bear += 2
    elif ma5_v > ma20_v and ma20_v > 0:
        gann_bull += 1
    elif ma5_v < ma20_v and ma20_v > 0:
        gann_bear += 1
    if rsi_g < 30:
        gann_bull += 1
    elif rsi_g > 70:
        gann_bear += 1
    if '底背离' in obv_div_g:
        gann_bull += 2
    if '顶背离' in obv_div_g:
        gann_bear += 2
    if ver_sup_g:
        gann_bull += 1
    if ver_res_g:
        gann_bear += 1
    if pat_bot_n_g >= 2:
        gann_bull += 2
    elif pat_bot_n_g == 1:
        gann_bull += 1
    if pat_top_n_g >= 2:
        gann_bear += 2
    elif pat_top_n_g == 1:
        gann_bear += 1
    if mp_phase_g == '拉升':
        gann_bull += 1
    elif mp_phase_g in ('盘头', '下跌'):
        gann_bear += 1
    if adx < 18:
        gann_bull = max(0, gann_bull - 1)
        gann_bear = max(0, gann_bear - 1)

    if gann_bull > gann_bear:
        states['gann'] = 'bullish'
    elif gann_bear > gann_bull:
        states['gann'] = 'bearish'
    else:
        states['gann'] = 'neutral'

    # 9. 威科夫量价: 供需+努力vs结果+Spring/UTAD+吸筹/派发区
    wyck_bull = 0
    wyck_bear = 0
    # OBV供需
    obv_div_w = str(indicators.get('obv_divergence', ''))
    if '底背离' in obv_div_w:
        wyck_bull += 2
    if '顶背离' in obv_div_w:
        wyck_bear += 2
    # 主力资金
    sm_sig_w = str(indicators.get('sm_smart_signal', ''))
    if 'strong_accumulation' in sm_sig_w:
        wyck_bull += 2
    elif 'accumulation' in sm_sig_w:
        wyck_bull += 1
    elif 'strong_distribution' in sm_sig_w:
        wyck_bear += 2
    elif 'distribution' in sm_sig_w:
        wyck_bear += 1
    # 努力vs结果
    if indicators.get('high_vol_stagnation', False):
        wyck_bear += 2  # 放量滞涨→派发
    if indicators.get('low_vol_pullback', False):
        wyck_bull += 2  # 缩量回踩→洗盘
    if indicators.get('vol_stacking', False):
        wyck_bull += 1  # 堆量→吸筹
    # RSI位置（spring/UTAD参考）
    rsi_w = _safe(indicators.get('rsi'), 50)
    if rsi_w < 35:
        wyck_bull += 1
    elif rsi_w > 65:
        wyck_bear += 1
    # 市场阶段
    mp_phase_w = str(indicators.get('mp_phase', ''))
    if mp_phase_w == '拉升':
        wyck_bull += 1
    elif mp_phase_w == '筑底':
        wyck_bull += 1
    elif mp_phase_w in ('盘头', '下跌'):
        wyck_bear += 1
    # 分形检测（Spring=底分形, UTAD=顶分形）
    pat_bot_w = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
    pat_top_w = _safe(indicators.get('pattern_top_fractal_n'), 0)
    if pat_bot_w >= 1:
        wyck_bull += 1
    if pat_top_w >= 1:
        wyck_bear += 1
    # ADX震荡市惩罚
    adx_w = _safe(indicators.get('dmi_adx'), 20)
    if adx_w < 18:
        wyck_bull = max(0, wyck_bull - 1)
        wyck_bear = max(0, wyck_bear - 1)

    if wyck_bull > wyck_bear:
        states['wyckoff'] = 'bullish'
    elif wyck_bear > wyck_bull:
        states['wyckoff'] = 'bearish'
    else:
        states['wyckoff'] = 'neutral'

    # 10. 道氏理论
    dow_bull = 0; dow_bear = 0
    ma20_d, ma60_d, ma120_d = _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0), _safe(indicators.get('ma120'),0)
    if ma20_d > ma60_d > ma120_d and ma60_d > 0: dow_bull += 2
    elif ma20_d < ma60_d < ma120_d and ma60_d > 0: dow_bear += 2
    mp_ret_60d_d = _safe(indicators.get('mp_ret_60d'), 0)
    if mp_ret_60d_d > 0.05: dow_bull += 1
    elif mp_ret_60d_d < -0.05: dow_bear += 1
    dow_mp = str(indicators.get('mp_phase', '')); dow_mp_c = _safe(indicators.get('mp_confidence'), 0)
    if dow_mp == '拉升' and dow_mp_c > 0.5: dow_bull += 1
    elif dow_mp == '下跌' and dow_mp_c > 0.5: dow_bear += 1
    adx_d = _safe(indicators.get('dmi_adx'), 20)
    if adx_d < 18: dow_bull = max(0, dow_bull-1); dow_bear = max(0, dow_bear-1)
    states['dow'] = 'bullish' if dow_bull > dow_bear else ('bearish' if dow_bear > dow_bull else 'neutral')

    # 11. CANSLIM
    can_bull = 0; can_bear = 0
    roc_l_c = _safe(indicators.get('roc_long'), 0); roc_s_c = _safe(indicators.get('roc_short'), 0)
    if roc_l_c > 0.05: can_bull += 2
    elif roc_l_c < -0.05: can_bear += 1
    if roc_s_c > 0.02: can_bull += 1
    can_mp = str(indicators.get('mp_phase', '')); can_mp_c = _safe(indicators.get('mp_confidence'), 0)
    if can_mp == '拉升' and can_mp_c > 0.5: can_bull += 1
    elif can_mp == '下跌' and can_mp_c > 0.5: can_bear += 2
    if indicators.get('vol_stacking', False): can_bull += 1
    states['canslim'] = 'bullish' if can_bull > can_bear else ('bearish' if can_bear > can_bull else 'neutral')

    # 12. 海龟交易
    turt_bull = 0; turt_bear = 0
    ma5_t, ma20_t, ma60_t = _safe(indicators.get('ma5'),0), _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0)
    if ma5_t > ma20_t > ma60_t and ma20_t > 0: turt_bull += 2
    elif ma5_t < ma20_t < ma60_t and ma20_t > 0: turt_bear += 2
    adx_t = _safe(indicators.get('dmi_adx'), 20)
    if adx_t > 25: turt_bull += 1
    elif adx_t < 18: turt_bull = max(0, turt_bull-1); turt_bear = max(0, turt_bear-1)
    atr_pct_t = _safe(indicators.get('vol_atr_pct'), 0.02)
    if atr_pct_t > 0.05: turt_bull = max(0, turt_bull-1)
    states['turtle'] = 'bullish' if turt_bull > turt_bear else ('bearish' if turt_bear > turt_bull else 'neutral')

    # 13. 谐波形态
    harm_bull = 0; harm_bear = 0
    rsi_h = _safe(indicators.get('rsi'), 50); cci_h = _safe(indicators.get('cci_short'), 0)
    if rsi_h < 30: harm_bull += 2
    elif rsi_h > 70: harm_bear += 2
    if cci_h < -150: harm_bull += 1
    elif cci_h > 150: harm_bear += 1
    if _safe(indicators.get('pattern_bottom_fractal_n'), 0) > 0: harm_bull += 1
    if _safe(indicators.get('pattern_top_fractal_n'), 0) > 0: harm_bear += 1
    if _safe(indicators.get('pattern_morning_star_n'), 0) > 0: harm_bull += 1
    if _safe(indicators.get('pattern_evening_star_n'), 0) > 0: harm_bear += 1
    states['harmonic'] = 'bullish' if harm_bull > harm_bear else ('bearish' if harm_bear > harm_bull else 'neutral')

    # 14. 市场轮廓
    mpf_bull = 0; mpf_bear = 0
    cp2 = _safe(indicators.get('current_price'), 0)
    bb_u2, bb_l2 = _safe(indicators.get('bb_upper'),0), _safe(indicators.get('bb_lower'),0)
    if cp2 > 0 and bb_u2 > 0:
        bb_range2 = bb_u2 - bb_l2
        if bb_range2 > 0:
            pos2 = (cp2 - bb_l2) / bb_range2
            if pos2 < 0.2: mpf_bull += 2
            elif pos2 > 0.8: mpf_bear += 2
    mp_ret_20d_mpf = _safe(indicators.get('mp_ret_20d'), 0)
    if mp_ret_20d_mpf > 0.03: mpf_bull += 1
    elif mp_ret_20d_mpf < -0.03: mpf_bear += 1
    adx_mpf = _safe(indicators.get('dmi_adx'), 20)
    if adx_mpf < 18: mpf_bull = max(0, mpf_bull-1); mpf_bear = max(0, mpf_bear-1)
    states['marketprofile'] = 'bullish' if mpf_bull > mpf_bear else ('bearish' if mpf_bear > mpf_bull else 'neutral')

    # 15. 混沌交易法
    chao_bull = 0; chao_bear = 0
    ma5_c, ma20_c, ma60_c = _safe(indicators.get('ma5'),0), _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0)
    if ma5_c > ma20_c > ma60_c and ma20_c > 0: chao_bull += 2
    elif ma5_c < ma20_c < ma60_c and ma20_c > 0: chao_bear += 2
    if _safe(indicators.get('pattern_bottom_fractal_n'), 0) >= 2: chao_bull += 2
    if _safe(indicators.get('pattern_top_fractal_n'), 0) >= 2: chao_bear += 2
    dif_c, dea_c = _safe(indicators.get('macd_dif'),0), _safe(indicators.get('macd_dea'),0)
    if dif_c > dea_c: chao_bull += 1
    else: chao_bear += 1
    adx_c2 = _safe(indicators.get('dmi_adx'), 20)
    if adx_c2 < 18: chao_bull = max(0, chao_bull-1); chao_bear = max(0, chao_bear-1)
    states['chaos'] = 'bullish' if chao_bull > chao_bear else ('bearish' if chao_bear > chao_bull else 'neutral')

    return states


def _dimension_llr_dict(indicators, calibration=None):
    """
    从指标字典中提取15大学派维度的对数似然比

    优先使用经验校准数据（calibration），无校准时回退到启发式映射。
    calibration 格式: {school_dim: {'bullish': LLR, 'bearish': LLR, 'neutral': LLR}}
    """
    llrs = {}
    dims = ['chanlun', 'elliott', 'tang', 'livermore', 'busch', 'classical', 'risk', 'gann', 'wyckoff',
            'dow', 'canslim', 'turtle', 'harmonic', 'marketprofile', 'chaos']

    if calibration is not None:
        states = _current_dimension_states(indicators)
        for dim in dims:
            if dim in calibration and dim in states:
                state = states[dim]
                llrs[dim] = calibration[dim].get(state, 0.0)
            else:
                llrs[dim] = 0.0
    else:
        # 启发式回退:从各学派指标提取连续评分→tanh映射

        # 1. 缠论
        chan_score = 0.0
        stroke_st = str(indicators.get('chanlun_stroke_state', ''))
        if '(1,1)' in stroke_st: chan_score += 0.5
        elif '(1,0)' in stroke_st: chan_score += 0.2
        elif '(-1,1)' in stroke_st: chan_score -= 0.5
        elif '(-1,0)' in stroke_st: chan_score -= 0.2
        div_type = str(indicators.get('chanlun_divergence_type', ''))
        div_str = _safe(indicators.get('chanlun_divergence_strength'), 0)
        if '底背驰' in div_type: chan_score += 0.5 * (0.5 + div_str)
        if '顶背驰' in div_type: chan_score -= 0.5 * (0.5 + div_str)
        cl_ts_sig = str(indicators.get('cl_turn_signal', ''))
        cl_ts = _safe(indicators.get('cl_turn_score'), 0)
        if 'strong_buy' in cl_ts_sig: chan_score += 0.4 * (cl_ts / 100)
        if 'strong_sell' in cl_ts_sig: chan_score -= 0.4 * (cl_ts / 100)
        chan_buy = indicators.get('chanlun_buy', [])
        chan_sell = indicators.get('chanlun_sell', [])
        if chan_buy: chan_score += 0.3 * len(chan_buy)
        if chan_sell: chan_score -= 0.3 * len(chan_sell)
        llrs['chanlun'] = _heuristic_llr(np.clip(chan_score, -1, 1))

        # 2. 艾略特波浪
        ew_signal = indicators.get('ew_trade_signal', 'neutral')
        ew_conf = _safe(indicators.get('ew_confidence'), 0)
        if ew_signal == 'bullish' and ew_conf >= 0.4:
            llrs['elliott'] = _heuristic_llr(ew_conf * 0.8)
        elif ew_signal == 'bearish' and ew_conf >= 0.4:
            llrs['elliott'] = _heuristic_llr(-ew_conf * 0.8)
        else:
            llrs['elliott'] = _heuristic_llr(0.0)

        # 3. 唐能通
        tang_score = 0.0
        tang_triple = str(indicators.get('tang_triple_status', ''))
        if tang_triple == 'triple_golden': tang_score += 0.6
        elif tang_triple == 'triple_death': tang_score -= 0.6
        elif tang_triple == 'partial_golden': tang_score += 0.2
        elif tang_triple == 'partial_death': tang_score -= 0.2
        if indicators.get('tang_jiato', False):
            tang_score += 0.3 * (_safe(indicators.get('tang_jiato_strength'), 40) / 100)
        if indicators.get('tang_jiaya', False):
            tang_score -= 0.3 * (_safe(indicators.get('tang_jiaya_strength'), 40) / 100)
        lyt_score = _safe(indicators.get('tang_laoyatou_score'), 0)
        if indicators.get('tang_laoyatou', False) and lyt_score > 50:
            tang_score += 0.3 * (lyt_score / 100)
        llrs['tang'] = _heuristic_llr(np.clip(tang_score, -1, 1))

        # 4. 利弗莫尔
        liv_score = 0.0
        liv_sig = str(indicators.get('livermore_signal', ''))
        if liv_sig == 'strong_buy': liv_score += 0.7
        elif liv_sig == 'buy': liv_score += 0.3
        elif liv_sig == 'strong_sell': liv_score -= 0.7
        elif liv_sig == 'sell': liv_score -= 0.3
        if indicators.get('livermore_danger', False):
            dl = _safe(indicators.get('livermore_danger_level'), 30)
            liv_score -= 0.5 * (dl / 100)
        llrs['livermore'] = _heuristic_llr(np.clip(liv_score, -1, 1))

        # 5. Busch量化
        busch_score = 0.0
        b2560 = str(indicators.get('b2560_signal', ''))
        if b2560 == 'strong_buy': busch_score += 0.4
        elif b2560 == 'sell': busch_score -= 0.35
        elif b2560 == 'hold_long': busch_score += 0.15
        elif b2560 == 'hold_short': busch_score -= 0.15
        vpa_sig = str(indicators.get('vpa_signal', ''))
        if vpa_sig == 'bullish': busch_score += 0.3
        elif vpa_sig == 'bearish': busch_score -= 0.3
        cva = str(indicators.get('cva_composite_signal', ''))
        if '看多' in cva: busch_score += 0.2
        if '看空' in cva: busch_score -= 0.2
        llrs['busch'] = _heuristic_llr(np.clip(busch_score, -1, 1))

        # 6. 经典技术分析
        cla_score = 0.0
        ma5 = indicators.get('ma5', 0)
        ma20 = indicators.get('ma20', 0)
        ma60 = indicators.get('ma60', 0)
        if ma5 and ma20 and ma60:
            if ma5 > ma20 > ma60: cla_score += 0.3
            elif ma5 < ma20 < ma60: cla_score -= 0.3
            elif ma5 > ma20: cla_score += 0.1
            elif ma5 < ma20: cla_score -= 0.1
        macd_dif = _safe(indicators.get('macd_dif'))
        macd_dea = _safe(indicators.get('macd_dea'))
        macd_hist = _safe(indicators.get('macd_hist'))
        if macd_dif > macd_dea and macd_hist > 0: cla_score += 0.2
        elif macd_dif < macd_dea and macd_hist < 0: cla_score -= 0.2
        rsi = _safe(indicators.get('rsi'), 50)
        if rsi > 70: cla_score -= 0.1
        elif rsi < 30: cla_score += 0.1
        elif rsi > 60: cla_score += 0.1
        elif rsi < 40: cla_score -= 0.1
        pat_rec = str(indicators.get('pattern_recent', ''))
        pat_score = 0.0
        for kw, sign in [('晨星', 0.3), ('暮星', -0.3), ('吞没看涨', 0.2), ('吞没看跌', -0.2),
                         ('刺透', 0.2), ('乌云盖顶', -0.2), ('底分型', 0.1), ('顶分型', -0.1)]:
            if kw in pat_rec: pat_score += sign
        cla_score += np.clip(pat_score, -0.4, 0.4)
        obv_div = str(indicators.get('obv_divergence', ''))
        if '底背离' in obv_div: cla_score += 0.25
        if '顶背离' in obv_div: cla_score -= 0.25
        if indicators.get('vol_stacking', False): cla_score += 0.15
        if indicators.get('high_vol_stagnation', False): cla_score -= 0.2
        sm_sig = str(indicators.get('sm_smart_signal', ''))
        if 'accumulation' in sm_sig: cla_score += 0.15
        if 'distribution' in sm_sig: cla_score -= 0.15
        llrs['classical'] = _heuristic_llr(np.clip(cla_score, -1, 1))

        # 7. 风险环境
        risk_score = 0.0
        rv_level = str(indicators.get('rv_level', ''))
        if rv_level in ['极低波动', '低波动']: risk_score += 0.15
        elif rv_level == '极高波动': risk_score -= 0.2
        har_dir = str(indicators.get('har_direction', ''))
        if har_dir == '收缩': risk_score += 0.1
        elif har_dir == '扩张': risk_score -= 0.1
        bebe_reg = str(indicators.get('bebe_regime', ''))
        if 'good_environment' in bebe_reg: risk_score += 0.2
        elif 'bad_environment' in bebe_reg: risk_score -= 0.25
        top_grade = str(indicators.get('top_escape_grade', ''))
        if top_grade == 'critical': risk_score -= 0.4
        elif top_grade == 'high_risk': risk_score -= 0.25
        mphase = str(indicators.get('mp_phase', ''))
        if mphase == '拉升': risk_score += 0.1
        elif mphase in ['盘头', '下跌']: risk_score -= 0.15
        llrs['risk'] = _heuristic_llr(np.clip(risk_score, -1, 1))

        # 8. 江恩理论
        gann_score = 0.0
        ma5_g = indicators.get('ma5', 0)
        ma20_g = indicators.get('ma20', 0)
        ma60_g = indicators.get('ma60', 0)
        if ma5_g and ma20_g and ma60_g:
            if ma5_g > ma20_g > ma60_g:
                gann_score += 0.25
            elif ma5_g < ma20_g < ma60_g:
                gann_score -= 0.25
            elif ma5_g > ma20_g:
                gann_score += 0.10
            elif ma5_g < ma20_g:
                gann_score -= 0.10
        rsi_g = _safe(indicators.get('rsi'), 50)
        if rsi_g < 30:
            gann_score += 0.20
        elif rsi_g > 70:
            gann_score -= 0.15
        obv_div_g = str(indicators.get('obv_divergence', ''))
        if '底背离' in obv_div_g:
            gann_score += 0.25
        if '顶背离' in obv_div_g:
            gann_score -= 0.25
        pat_bot_g = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
        pat_top_g = _safe(indicators.get('pattern_top_fractal_n'), 0)
        if pat_bot_g >= 2:
            gann_score += 0.25
        elif pat_bot_g == 1:
            gann_score += 0.10
        if pat_top_g >= 2:
            gann_score -= 0.25
        elif pat_top_g == 1:
            gann_score -= 0.10
        ver_sup_g = indicators.get('verified_support', [])
        ver_res_g = indicators.get('verified_resistance', [])
        if ver_sup_g:
            gann_score += 0.10
        if ver_res_g:
            gann_score -= 0.10
        mp_phase_g = str(indicators.get('mp_phase', ''))
        if mp_phase_g == '拉升':
            gann_score += 0.10
        elif mp_phase_g in ('盘头', '下跌'):
            gann_score -= 0.15
        adx_g = _safe(indicators.get('dmi_adx'), 20)
        if adx_g < 18:
            gann_score *= 0.60  # 震荡市江恩建议观望
        # 共振检测（多维度指向同一方向）
        rv_lvl_g = str(indicators.get('rv_level', ''))
        if '极高' in rv_lvl_g:
            gann_score *= 0.50  # 极端波动减仓
        llrs['gann'] = _heuristic_llr(np.clip(gann_score, -1, 1))

        # 9. 威科夫量价: 供需+努力vs结果+Spring/UTAD+市场阶段
        wyck_score = 0.0
        # OBV供需
        obv_div_w = str(indicators.get('obv_divergence', ''))
        if '底背离' in obv_div_w:
            wyck_score += 0.25
        if '顶背离' in obv_div_w:
            wyck_score -= 0.25
        # 主力资金
        sm_sig_w = str(indicators.get('sm_smart_signal', ''))
        if 'strong_accumulation' in sm_sig_w:
            wyck_score += 0.30
        elif 'accumulation' in sm_sig_w:
            wyck_score += 0.15
        elif 'strong_distribution' in sm_sig_w:
            wyck_score -= 0.30
        elif 'distribution' in sm_sig_w:
            wyck_score -= 0.15
        # 努力vs结果（威科夫核心）
        if indicators.get('high_vol_stagnation', False):
            wyck_score -= 0.30  # 放量滞涨→严重派发
        if indicators.get('low_vol_pullback', False):
            wyck_score += 0.25  # 缩量回踩→供应枯竭
        if indicators.get('vol_stacking', False):
            wyck_score += 0.20  # 堆量→主力介入
        vr_w = _safe(indicators.get('vol_ratio'), 1.0)
        if indicators.get('vol_price_resonance', False) and vr_w > 1.5:
            wyck_score += 0.15  # 量价共振→健康
        elif indicators.get('vol_price_resonance', False) and vr_w < 0.7:
            wyck_score += 0.08
        # RSI超卖/超买（Spring/UTAD参考）
        rsi_w = _safe(indicators.get('rsi'), 50)
        if rsi_w < 30:
            wyck_score += 0.20
        elif rsi_w > 70:
            wyck_score -= 0.15
        # 市场阶段
        mp_phase_w = str(indicators.get('mp_phase', ''))
        if mp_phase_w == '拉升':
            wyck_score += 0.15
        elif mp_phase_w == '筑底':
            wyck_score += 0.10
        elif mp_phase_w == '下跌':
            wyck_score -= 0.20
        elif mp_phase_w == '盘头':
            wyck_score -= 0.15
        # 分形（Spring/UTAD）
        pat_bot_w = _safe(indicators.get('pattern_bottom_fractal_n'), 0)
        pat_top_w = _safe(indicators.get('pattern_top_fractal_n'), 0)
        if pat_bot_w >= 2:
            wyck_score += 0.20
        elif pat_bot_w == 1:
            wyck_score += 0.10
        if pat_top_w >= 2:
            wyck_score -= 0.20
        elif pat_top_w == 1:
            wyck_score -= 0.10
        # VR
        vr_val_w = _safe(indicators.get('vr_value'), 100)
        vr_sig_w = str(indicators.get('vr_signal', ''))
        if vr_sig_w == 'bearish' and vr_val_w > 300:
            wyck_score -= 0.15
        elif vr_sig_w == 'contrarian_bullish':
            wyck_score += 0.15
        # ADX
        adx_w = _safe(indicators.get('dmi_adx'), 20)
        if adx_w < 18:
            wyck_score *= 0.55  # 震荡市无方向→等待
        # 极端波动惩罚
        rv_lvl_w = str(indicators.get('rv_level', ''))
        if '极高' in rv_lvl_w:
            wyck_score *= 0.45
        elif '高' in rv_lvl_w:
            wyck_score *= 0.70
        llrs['wyckoff'] = _heuristic_llr(np.clip(wyck_score, -1, 1))

        # 10. 道氏理论
        dow_s = 0.0
        ma20_d, ma60_d, ma120_d = _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0), _safe(indicators.get('ma120'),0)
        if ma20_d > ma60_d > ma120_d and ma60_d > 0: dow_s += 0.30
        elif ma20_d < ma60_d < ma120_d and ma60_d > 0: dow_s -= 0.30
        ret60_d = _safe(indicators.get('mp_ret_60d'), 0)
        if ret60_d > 0.05: dow_s += 0.10
        elif ret60_d < -0.05: dow_s -= 0.10
        mp_d = str(indicators.get('mp_phase', '')); mp_c_d = _safe(indicators.get('mp_confidence'), 0)
        if mp_d == '拉升' and mp_c_d > 0.5: dow_s += 0.10
        elif mp_d == '下跌' and mp_c_d > 0.5: dow_s -= 0.15
        adx_d = _safe(indicators.get('dmi_adx'), 20)
        if adx_d < 18: dow_s *= 0.55
        llrs['dow'] = _heuristic_llr(np.clip(dow_s, -1, 1))

        # 11. CANSLIM
        can_s = 0.0
        roc_l_c = _safe(indicators.get('roc_long'), 0); roc_s_c = _safe(indicators.get('roc_short'), 0)
        if roc_l_c > 0.05: can_s += 0.25
        elif roc_l_c < -0.05: can_s -= 0.15
        if roc_s_c > 0.03 and roc_l_c > 0.03: can_s += 0.15
        if indicators.get('vol_stacking', False): can_s += 0.15
        if indicators.get('high_vol_stagnation', False): can_s -= 0.20
        mp_c = str(indicators.get('mp_phase', '')); mp_cc = _safe(indicators.get('mp_confidence'), 0)
        if mp_c == '拉升' and mp_cc > 0.5: can_s += 0.10
        elif mp_c == '下跌' and mp_cc > 0.5: can_s -= 0.20
        elif mp_c == '盘头' and mp_cc > 0.5: can_s *= 0.60
        sm_s_c = str(indicators.get('sm_smart_signal', ''))
        if 'accumulation' in sm_s_c: can_s += 0.10
        llrs['canslim'] = _heuristic_llr(np.clip(can_s, -1, 1))

        # 12. 海龟交易
        turt_s = 0.0
        ma5_t, ma20_t, ma60_t = _safe(indicators.get('ma5'),0), _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0)
        if ma5_t > ma20_t > ma60_t and ma20_t > 0: turt_s += 0.25
        elif ma5_t < ma20_t < ma60_t and ma20_t > 0: turt_s -= 0.25
        atr_pct_t = _safe(indicators.get('vol_atr_pct'), 0.02)
        if atr_pct_t > 0.05: turt_s *= 0.50
        elif atr_pct_t > 0.035: turt_s *= 0.75
        adx_t = _safe(indicators.get('dmi_adx'), 20)
        if adx_t > 25: turt_s *= 1.15
        elif adx_t < 18: turt_s *= 0.45
        llrs['turtle'] = _heuristic_llr(np.clip(turt_s, -1, 1))

        # 13. 谐波形态
        harm_s = 0.0
        rsi_h = _safe(indicators.get('rsi'), 50); cci_h = _safe(indicators.get('cci_short'), 0)
        if rsi_h < 30: harm_s += 0.25
        elif rsi_h > 70: harm_s -= 0.20
        if cci_h < -150: harm_s += 0.15
        elif cci_h > 150: harm_s -= 0.15
        if _safe(indicators.get('pattern_bottom_fractal_n'), 0) >= 1: harm_s += 0.10
        if _safe(indicators.get('pattern_top_fractal_n'), 0) >= 1: harm_s -= 0.10
        pat_rec_h = str(indicators.get('pattern_recent', ''))
        for kw, s in [('晨星',0.15),('吞没看涨',0.15),('暮星',-0.15),('吞没看跌',-0.15)]:
            if kw in pat_rec_h: harm_s += s; break
        llrs['harmonic'] = _heuristic_llr(np.clip(harm_s, -1, 1))

        # 14. 市场轮廓
        mpf_s = 0.0
        cp_mp = _safe(indicators.get('current_price'), 0)
        bb_u_mpf, bb_l_mpf = _safe(indicators.get('bb_upper'),0), _safe(indicators.get('bb_lower'),0)
        if cp_mp > 0 and bb_u_mpf > 0:
            rng = bb_u_mpf - bb_l_mpf
            if rng > 0:
                pos = (cp_mp - bb_l_mpf) / rng
                if pos < 0.15: mpf_s += 0.25
                elif pos > 0.85: mpf_s -= 0.20
        ret20_mpf = _safe(indicators.get('mp_ret_20d'), 0)
        if ret20_mpf > 0.03: mpf_s += 0.10
        elif ret20_mpf < -0.03: mpf_s -= 0.10
        if indicators.get('vol_stacking', False): mpf_s += 0.12
        adx_mpf = _safe(indicators.get('dmi_adx'), 20)
        if adx_mpf < 18: mpf_s *= 0.55
        llrs['marketprofile'] = _heuristic_llr(np.clip(mpf_s, -1, 1))

        # 15. 混沌交易法
        chao_s = 0.0
        ma5_c, ma20_c, ma60_c = _safe(indicators.get('ma5'),0), _safe(indicators.get('ma20'),0), _safe(indicators.get('ma60'),0)
        if ma5_c > ma20_c > ma60_c and ma20_c > 0: chao_s += 0.20
        elif ma5_c < ma20_c < ma60_c and ma20_c > 0: chao_s -= 0.20
        if _safe(indicators.get('pattern_bottom_fractal_n'), 0) >= 2: chao_s += 0.20
        if _safe(indicators.get('pattern_top_fractal_n'), 0) >= 2: chao_s -= 0.20
        dif_c, dea_c = _safe(indicators.get('macd_dif'),0), _safe(indicators.get('macd_dea'),0)
        if dif_c > dea_c: chao_s += 0.15
        else: chao_s -= 0.15
        roc_s_c2 = _safe(indicators.get('roc_short'), 0)
        if roc_s_c2 > 0.02: chao_s += 0.10
        elif roc_s_c2 < -0.02: chao_s -= 0.10
        adx_c = _safe(indicators.get('dmi_adx'), 20)
        if adx_c < 18: chao_s *= 0.50
        llrs['chaos'] = _heuristic_llr(np.clip(chao_s, -1, 1))

    return llrs


def _estimate_dimension_correlations_from_states(state_vectors):
    """
    Estimate empirical dimension correlation matrix from historical state vectors.

    Args:
        state_vectors: (T, D) array where T=time steps, D=dimensions.
                       Values should be normalized to [-1, 1] range (z-scored or LLR-based).

    Returns: (corr_matrix, n_samples) or (None, 0) if insufficient data.
    """
    import numpy as np
    state_vectors = np.asarray(state_vectors)
    if state_vectors.ndim != 2 or state_vectors.shape[0] < 30 or state_vectors.shape[1] < 3:
        return None, 0
    # Clip extreme values and compute robust correlation
    state_vectors = np.clip(state_vectors, -5, 5)
    try:
        empirical_corr = np.corrcoef(state_vectors.T)
        # Ensure positive semi-definite
        eigenvalues = np.linalg.eigvalsh(empirical_corr)
        if np.min(eigenvalues) < 0:
            # Higham projection to nearest PSD matrix
            eigvecs = np.linalg.eigh(empirical_corr)[1]
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            empirical_corr = eigvecs @ np.diag(eigenvalues) @ eigvecs.T
        return empirical_corr, int(state_vectors.shape[0])
    except Exception:
        return None, 0


def _dimension_correlation_effective_rank(empirical_corr=None, empirical_n=0):
    """
    Prior-guided correlation estimation with optional empirical shrinkage.

    Prior = domain-knowledge correlations (hard-coded, serves as Bayesian prior).
    When empirical data is available (empirical_corr, empirical_n), blend via
    shrinkage: corr = (1 - α) * prior + α * empirical, where α ∝ empirical_n.

    Args:
        empirical_corr: (D,D) empirical correlation matrix, or None
        empirical_n: number of time steps used for empirical estimate

    Returns effective rank and decorrelation factor.
    """
    import numpy as np

    dims = ['chanlun', 'elliott', 'tang', 'livermore', 'busch', 'classical', 'risk', 'gann', 'wyckoff',
            'dow', 'canslim', 'turtle', 'harmonic', 'marketprofile', 'chaos']
    n = len(dims)
    idx = {d: i for i, d in enumerate(dims)}

    # Prior correlation matrix (domain knowledge)
    corr = np.eye(n)
    pairs = [
        ('chanlun', 'elliott', 0.35), ('chanlun', 'classical', 0.40),
        ('elliott', 'classical', 0.30), ('tang', 'classical', 0.40),
        ('tang', 'busch', 0.30), ('busch', 'classical', 0.45),
        ('livermore', 'classical', 0.30), ('livermore', 'chanlun', 0.25),
        ('risk', 'classical', 0.25), ('risk', 'elliott', 0.20),
        ('risk', 'chanlun', 0.20), ('chanlun', 'tang', 0.25),
        ('elliott', 'tang', 0.25), ('busch', 'livermore', 0.20),
        ('tang', 'livermore', 0.20), ('elliott', 'busch', 0.20),
        ('gann', 'classical', 0.35), ('gann', 'risk', 0.25),
        ('gann', 'elliott', 0.25), ('gann', 'livermore', 0.25),
        ('gann', 'chanlun', 0.20), ('gann', 'busch', 0.20),
        ('gann', 'tang', 0.20),
        ('wyckoff', 'busch', 0.40), ('wyckoff', 'classical', 0.35),
        ('wyckoff', 'tang', 0.25), ('wyckoff', 'livermore', 0.25),
        ('wyckoff', 'risk', 0.20), ('wyckoff', 'chanlun', 0.20),
        ('wyckoff', 'gann', 0.25), ('wyckoff', 'elliott', 0.20),
        ('dow', 'classical', 0.40), ('dow', 'gann', 0.30),
        ('dow', 'risk', 0.25), ('dow', 'elliott', 0.25),
        ('dow', 'wyckoff', 0.25), ('dow', 'chanlun', 0.20),
        ('canslim', 'classical', 0.40), ('canslim', 'risk', 0.25),
        ('canslim', 'dow', 0.30), ('canslim', 'tang', 0.25),
        ('canslim', 'busch', 0.20), ('canslim', 'livermore', 0.20),
        ('turtle', 'dow', 0.35), ('turtle', 'classical', 0.30),
        ('turtle', 'gann', 0.25), ('turtle', 'risk', 0.25),
        ('turtle', 'wyckoff', 0.15),
        ('harmonic', 'classical', 0.30), ('harmonic', 'chanlun', 0.30),
        ('harmonic', 'wyckoff', 0.25), ('harmonic', 'elliott', 0.25),
        ('harmonic', 'gann', 0.20), ('harmonic', 'tang', 0.15),
        ('marketprofile', 'wyckoff', 0.40), ('marketprofile', 'classical', 0.30),
        ('marketprofile', 'risk', 0.25), ('marketprofile', 'dow', 0.25),
        ('marketprofile', 'busch', 0.20),
        ('chaos', 'classical', 0.35), ('chaos', 'chanlun', 0.30),
        ('chaos', 'elliott', 0.25), ('chaos', 'wyckoff', 0.20),
        ('chaos', 'gann', 0.20), ('chaos', 'harmonic', 0.20),
    ]
    for d1, d2, r in pairs:
        if d1 in idx and d2 in idx:
            corr[idx[d1], idx[d2]] = r
            corr[idx[d2], idx[d1]] = r

    # Shrinkage blend with empirical data if available
    shrinkage_weight = 0.0
    if empirical_corr is not None and empirical_n >= 30:
        # Blend weight ∝ empirical_n, capped at 0.6 (prior always has weight ≥ 0.4)
        shrinkage_weight = min(0.60, empirical_n / (empirical_n + 120))
        empirical_corr = np.asarray(empirical_corr)
        if empirical_corr.shape == (n, n):
            corr = (1 - shrinkage_weight) * corr + shrinkage_weight * empirical_corr

    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)
    eff_rank = (sum_eig ** 2) / sum_eig_sq if sum_eig_sq > 0 else 1.0

    decorr_factor = np.sqrt(float(n) / eff_rank)

    return {
        'effective_rank': round(float(eff_rank), 2),
        'decorr_factor': round(float(decorr_factor), 3),
        'n_dimensions': n,
        'eigenvalues': [round(float(e), 3) for e in eigenvalues],
        'shrinkage_weight': round(shrinkage_weight, 3),
        'empirical_samples': empirical_n,
    }


def multi_signal_bayesian_fusion(indicators, prior=0.5, calibration=None):
    """
    多信号贝叶斯融合:将15大学派维度的证据融合为单一后验概率

    数学原理:
    log(P(bull|E)/P(bear|E)) = log(P(bull)/P(bear)) + Σ log(P(E_i|bull)/P(E_i|bear))

    15大学派:缠论/波浪/唐能通/利弗莫尔/Busch/经典TA/风险/江恩/威科夫/道氏/CANSLIM/海龟/谐波/市场轮廓/混沌

    prior: 先验概率（默认0.5）
    calibration: 经验校准数据，优先使用
    """
    llrs = _dimension_llr_dict(indicators, calibration=calibration)

    # 层次贝叶斯收缩:极端维度值被拉向群体均值（Gelman BDA3 Ch.5）
    shrunk_llrs, shrinkage_factors, hier_mu, most_shrunk = hierarchical_bayesian_fusion(llrs)

    # 先验log-odds
    log_odds_prior = np.log(prior / (1 - prior)) if prior > 0 and prior < 1 else 0

    # 后验log-odds = 先验 + 收缩后LLR之和 / 去相关因子
    # 去相关因子基于9大学派领域知识相关矩阵的有效秩
    total_llr = sum(shrunk_llrs.values())
    n_active = sum(1 for _, v in shrunk_llrs.items() if abs(v) > 0.05)

    corr_info = _dimension_correlation_effective_rank()
    eff_rank = corr_info['effective_rank']
    decorr_factor = corr_info['decorr_factor']

    if n_active > 0:
        # 结构感知去相关因子:学派级维度间相关性低于旧版指标级
        if n_active == 1:
            pass  # 单学派无共线性问题
        elif n_active == 2:
            total_llr = total_llr / min(decorr_factor, 1.15)
        else:
            total_llr = total_llr / decorr_factor

    log_odds_posterior = log_odds_prior + total_llr

    # 转为概率
    posterior = 1.0 / (1.0 + np.exp(-log_odds_posterior))

    # 香农熵
    p = posterior
    if p > 0 and p < 1:
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    else:
        entropy = 0

    # 信号方向
    if posterior > 0.55:
        bayes_signal = 'bullish'
    elif posterior < 0.45:
        bayes_signal = 'bearish'
    else:
        bayes_signal = 'neutral'

    # 维度贡献排序（使用收缩后的LLR）
    dim_contributions = sorted(shrunk_llrs.items(), key=lambda x: abs(x[1]), reverse=True)
    dim_summary = [f"{dim}({llr:+.2f})" for dim, llr in dim_contributions if abs(llr) > 0.1]

    # 有效维度数（有实质信息的维度）
    active_dims = sum(1 for _, llr in dim_contributions if abs(llr) > 0.1)

    return {
        'bayes_posterior': round(posterior, 4),
        'bayes_signal': bayes_signal,
        'bayes_entropy': round(entropy, 4),
        'bayes_log_odds': round(log_odds_posterior, 4),
        'bayes_prior': prior,
        'bayes_dimensions_active': active_dims,
        'bayes_dimensions_total': len(llrs),
        'bayes_dimension_contributions': dim_summary,
        'bayes_dimension_llrs': {k: round(v, 4) for k, v in shrunk_llrs.items()},
        'bayes_hierarchical_mu': round(hier_mu, 4),
        'bayes_hierarchical_shrinkage': {k: round(v, 4) for k, v in shrinkage_factors.items()},
        'bayes_most_shrunk_dims': most_shrunk,
        # 去相关信息
        'bayes_effective_rank': corr_info['effective_rank'],
        'bayes_decorr_factor': decorr_factor,
        'bayes_dimension_eigenvalues': corr_info['eigenvalues'],
    }


def beta_bernoulli_sequential_update(df, lookback=20, alpha_prior=2, beta_prior=2):
    """
    Beta-Bernoulli顺序更新趋势后验
    每根K线视作一次伯努利试验（上涨=1，下跌=0）

    后验 ∼ Beta(α + Σup, β + N - Σup)
    后验均值 = (α + up_count) / (α + β + N)

    返回当前趋势后验概率和更新历史
    """
    if len(df) < 5:
        return {'beta_posterior': 0.5, 'beta_signal': 'neutral', 'beta_entropy': 1.0}

    close = df['close'].astype(float).values
    up_count = 0
    n = min(lookback, len(close) - 1)
    for i in range(len(close) - n, len(close)):
        if close[i] > close[i-1]:
            up_count += 1

    alpha_post = alpha_prior + up_count
    beta_post = beta_prior + n - up_count
    posterior_mean = alpha_post / (alpha_post + beta_post)

    # 香农熵
    p = posterior_mean
    if p > 0 and p < 1:
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    else:
        entropy = 0

    if posterior_mean > 0.55:
        beta_signal = 'bullish'
    elif posterior_mean < 0.45:
        beta_signal = 'bearish'
    else:
        beta_signal = 'neutral'

    return {
        'beta_posterior': round(posterior_mean, 4),
        'beta_signal': beta_signal,
        'beta_entropy': round(entropy, 4),
        'beta_up_count': up_count,
        'beta_n': n,
        'beta_alpha_post': alpha_post,
        'beta_beta_post': beta_post
    }


def bayesian_confidence_fusion(indicators, df_daily=None):
    """
    综合贝叶斯置信度引擎
    融合多信号后验 + Beta-Bernoulli趋势后验 → 综合贝叶斯置信度

    当df_daily提供且长度≥80时，自动调用calibrate_bayesian_likelihoods
    获取经验LLR校准数据，替代启发式映射。
    symbol参数用于三级缓存持久化（内存→SQLite→重新计算）。
    """
    # 尝试获取经验校准数据（含三级缓存）
    calibration = None
    symbol = indicators.get('_symbol', None) if isinstance(indicators, dict) else None
    if df_daily is not None and len(df_daily) >= 80:
        calibration = calibrate_bayesian_likelihoods(df_daily, symbol=symbol)

    # 多信号贝叶斯融合（传入校准数据）
    mbf = multi_signal_bayesian_fusion(indicators, prior=0.5, calibration=calibration)

    # Beta-Bernoulli趋势后验
    if df_daily is not None and len(df_daily) >= 5:
        bbu = beta_bernoulli_sequential_update(df_daily)
    else:
        bbu = {'beta_posterior': 0.5, 'beta_signal': 'neutral', 'beta_entropy': 1.0}

    # 融合两个贝叶斯源（加权平均，权重由各自熵决定）
    w1 = (1.0 - mbf['bayes_entropy']) if mbf['bayes_entropy'] > 0 else 0.5
    w2 = (1.0 - bbu['beta_entropy']) if bbu['beta_entropy'] > 0 else 0.5
    total_w = w1 + w2

    if total_w > 0:
        fused_posterior = (w1 * mbf['bayes_posterior'] + w2 * bbu['beta_posterior']) / total_w
    else:
        fused_posterior = 0.5

    # 熵融合
    p = fused_posterior
    fused_entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p) if p > 0 and p < 1 else 0

    if fused_posterior > 0.55:
        fused_signal = 'bullish'
    elif fused_posterior < 0.45:
        fused_signal = 'bearish'
    else:
        fused_signal = 'neutral'

    return {
        'bayes_fused_posterior': round(fused_posterior, 4),
        'bayes_fused_signal': fused_signal,
        'bayes_fused_entropy': round(fused_entropy, 4),
        'bayes_multi_signal_posterior': mbf['bayes_posterior'],
        'bayes_beta_trend_posterior': bbu['beta_posterior'],
        'bayes_dimensions_active': mbf['bayes_dimensions_active'],
        'bayes_dimension_contributions': mbf['bayes_dimension_contributions'],
        'bayes_entropy_high': fused_entropy > 0.85,  # 高熵=不确定性大=不宜交易
    }


# ===================== 16. 凯利公式仓位管理 =====================
# 基于 Kelly (1956) 和 Merton 连续形式
# f* = (bp - q) / b  (二元)  或  f* = μ / σ² (连续)

def kelly_binary(win_prob, avg_win_pct, avg_loss_pct):
    """
    二元凯利公式:f* = (b·p - q) / b
    win_prob: 胜率 (0-1)
    avg_win_pct: 平均盈利百分比 (如0.05表示5%)
    avg_loss_pct: 平均亏损百分比（正数，如0.03表示3%）

    返回最优仓位比例
    """
    if avg_loss_pct <= 0 or win_prob <= 0:
        return 0.0

    b = avg_win_pct / avg_loss_pct  # 盈亏比
    q = 1.0 - win_prob

    f_star = (b * win_prob - q) / b if b > 0 else 0
    return max(0.0, f_star)


def kelly_continuous(expected_return, volatility, risk_free=0.0):
    """
    连续凯利公式（Merton形式）:f* = (μ - r) / σ²
    expected_return: 年化期望收益
    volatility: 年化波动率
    risk_free: 无风险利率
    """
    excess = expected_return - risk_free
    var = volatility ** 2
    if var <= 0 or excess <= 0:
        return 0.0
    return excess / var


def kelly_from_deepseek_signal(signal_dict, current_price, rv_composite=0.02, atr_pct=0.02):
    """
    从DeepSeek AI信号中提取参数，计算凯利最优仓位

    输入:
      - AI返回的signal JSON（含confidence, batch_targets, stop_loss）
      - current_price: 当前价格
      - rv_composite: 已实现波动率（用于智能默认值）
      - atr_pct: ATR百分比（用于止损默认值，小数形式如0.02=2%）

    返回:凯利仓位计算详情
    """
    if signal_dict is None:
        return _empty_kelly_result()

    confidence = signal_dict.get('confidence', 0.5) or 0.5
    signal_type = signal_dict.get('signal', 'neutral') or 'neutral'
    stop_loss = signal_dict.get('stop_loss', '') or ''
    batch_targets = signal_dict.get('batch_targets') or []
    entry_zone = signal_dict.get('entry_zone', '') or ''

    if signal_type == 'neutral' or len(batch_targets) == 0:
        return {**_empty_kelly_result(), 'kelly_rationale': '中性信号或无目标价，凯利不适用'}

    # ---- 估算胜率 ----
    # 使用AI confidence，向50%做适度收缩（5次虚拟观测）
    # 这不是"shrink to 0.5"而是Laplace平滑:避免AI过度自信
    prior_weight = 5
    n_effective = 10
    p_shrunk = (confidence * n_effective + 0.5 * prior_weight) / (n_effective + prior_weight)

    # ---- 估算平均盈利% ----
    try:
        entry_price = float(entry_zone.split('-')[0]) if '-' in str(entry_zone) else float(
            entry_zone.replace('附近', '').replace('突破', '').strip()) if entry_zone else current_price
    except (ValueError, AttributeError):
        entry_price = current_price

    avg_win_pct = 0
    for bt in batch_targets[:2]:
        try:
            target_price = float(bt['price'])
            gain_pct = (target_price - entry_price) / entry_price
            avg_win_pct += gain_pct * float(bt.get('ratio', '50%').replace('%', '')) / 100
        except (ValueError, KeyError, TypeError):
            pass

    # 默认值基于波动率:高波动股票应有更大预期盈利
    if avg_win_pct <= 0:
        avg_win_pct = max(0.02, atr_pct * 1.5)

    # ---- 估算平均亏损% ----
    try:
        stop_price = float(stop_loss)
        avg_loss_pct = abs((entry_price - stop_price) / entry_price)
    except (ValueError, TypeError):
        # 默认止损基于ATR:2倍ATR
        avg_loss_pct = max(0.015, atr_pct * 2.0)

    if avg_loss_pct <= 0:
        avg_loss_pct = max(0.015, atr_pct * 2.0)

    # ---- 凯利计算 ----
    f_full = kelly_binary(p_shrunk, avg_win_pct, avg_loss_pct)

    # 盈亏比
    payoff_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0

    # 期望值
    ev = p_shrunk * avg_win_pct - (1 - p_shrunk) * avg_loss_pct

    return {
        'kelly_f_full': round(f_full, 4),
        'kelly_f_half': round(f_full * 0.5, 4),
        'kelly_f_quarter': round(f_full * 0.25, 4),
        'kelly_win_prob': round(p_shrunk, 4),
        'kelly_payoff_ratio': round(payoff_ratio, 2),
        'kelly_expected_value': round(ev, 4),
        'kelly_avg_win_pct': round(avg_win_pct, 4),
        'kelly_avg_loss_pct': round(avg_loss_pct, 4),
        'kelly_rationale': f'半凯利建议仓位={f_full*50:.1f}%（f*={f_full*100:.1f}%）',
    }


def kelly_portfolio_multi(positions_dict, total_capital=1.0):
    """
    多资产凯利:f* = Σ⁻¹ μ
    positions_dict: {symbol: {'p': win_prob, 'b': payoff_ratio, 'sigma': vol}, ...}
    简化版:独立假设 + 折扣因子
    """
    if len(positions_dict) == 0:
        return {}

    n = len(positions_dict)
    results = {}

    for sym, params in positions_dict.items():
        p = params.get('p', 0.5)
        b = params.get('b', 1.0)
        individual_kelly = (b * p - (1 - p)) / b if b > 0 else 0

        # 多资产折扣:随着同时持仓数增加，降低单仓位
        diversification_discount = 1.0 / np.sqrt(n)
        adjusted_kelly = individual_kelly * diversification_discount

        results[sym] = {
            'individual_kelly': round(individual_kelly, 4),
            'adjusted_kelly': round(adjusted_kelly, 4),
            'half_kelly': round(adjusted_kelly * 0.5, 4),
        }

    return results


# ===================== 17. 非遍历性保护 =====================

def non_ergodicity_protections(kelly_f, rv_composite, current_drawdown_pct=0.0):
    """
    非遍历性保护机制
    1. 硬上限:单仓位≤25%
    2. 波动率拖累修正:几何收益 = μ - σ²/2
    3. 回撤熔断:回撤>15%时仓位减半
    4. 负增长检测:σ²/2 > μ 时不入场
    """
    # 硬上限
    capped_f = min(kelly_f, 0.25)

    # 波动率拖累修正 (rv_composite is annualized, e.g. 0.30 = 30%)
    # 几何收益 = μ - σ²/2, where σ²/2 is the annual vol drag
    vol_drag = (rv_composite ** 2) / 2.0
    # 如果波动率拖累大于凯利期望收益，仓位归零
    safe_threshold = 0.05  # 5%期望年收益门槛
    if vol_drag > safe_threshold:
        capped_f *= max(0, 1 - vol_drag / safe_threshold)

    # 回撤熔断
    if current_drawdown_pct > 0.20:
        capped_f *= 0.25  # 回撤>20%→仓位×0.25
    elif current_drawdown_pct > 0.15:
        capped_f *= 0.5   # 回撤>15%→仓位减半
    elif current_drawdown_pct > 0.10:
        capped_f *= 0.75  # 回撤>10%→仓位×0.75

    # 最终限制
    final_f = max(0.0, min(0.25, capped_f))

    protections = []
    if kelly_f > 0.25: protections.append(f'硬上限: {kelly_f:.1%}→25%')
    if vol_drag > safe_threshold: protections.append(f'波动拖累修正({vol_drag:.1%})')
    if current_drawdown_pct > 0.10: protections.append(f'回撤熔断({current_drawdown_pct:.0%})')

    return {
        'ergo_kelly_raw': round(kelly_f, 4),
        'ergo_final_position': round(final_f, 4),
        'ergo_vol_drag': round(vol_drag, 4),
        'ergo_protections': protections,
        'ergo_safe_to_enter': vol_drag <= safe_threshold,
        'ergo_hard_cap': 0.25,
    }


def _empty_kelly_result():
    return {
        'kelly_f_full': 0, 'kelly_f_half': 0, 'kelly_f_quarter': 0,
        'kelly_win_prob': 0.5, 'kelly_payoff_ratio': 0, 'kelly_expected_value': 0,
        'kelly_avg_win_pct': 0, 'kelly_avg_loss_pct': 0, 'kelly_rationale': '无AI信号'
    }


# ===================== 18. VR 容量比率指标 =====================
# 来源:成交量核心技术
# VR = (AVS + CVS/2) / (BVS + CVS/2) × 100
# AVS: N日内上涨日成交量之和
# BVS: N日内下跌日成交量之和
# CVS: N日内平盘日成交量之和

def vr_indicator(df, n=26):
    """
    VR 容量比率指标
    VR>450: 头部警戒区；VR<70: 底部区域
    VR 160-450: 强势区；VR 70-160: 安全区
    """
    if len(df) < n + 5:
        return _empty_vr_result()

    close = df['close'].astype(float).values
    vol = df['volume'].astype(float).values
    avs, bvs, cvs = 0.0, 0.0, 0.0

    for i in range(-n, 0):
        if i + 1 < 0:
            chg = close[i + 1] - close[i]
        else:
            chg = 0
        if chg > 0.001 * close[i]:
            avs += vol[i + 1] if i + 1 < 0 else vol[i]
        elif chg < -0.001 * close[i]:
            bvs += vol[i + 1] if i + 1 < 0 else vol[i]
        else:
            cvs += vol[i + 1] if i + 1 < 0 else vol[i]

    denom = bvs + cvs / 2.0
    if denom < 1:
        vr_val = 200.0
    else:
        vr_val = (avs + cvs / 2.0) / denom * 100.0

    # VR MA (24日)
    vr_history = []
    for t in range(n + 5, len(df) + 1):
        a2, b2, c2 = 0.0, 0.0, 0.0
        for j in range(t - n, t - 1):
            chg2 = close[j + 1] - close[j]
            if chg2 > 0.001 * close[j]:
                a2 += vol[j + 1]
            elif chg2 < -0.001 * close[j]:
                b2 += vol[j + 1]
            else:
                c2 += vol[j + 1]
        d2 = b2 + c2 / 2.0
        vr_history.append((a2 + c2 / 2.0) / d2 * 100.0 if d2 > 0 else 200.0)

    vr_ma = np.mean(vr_history[-24:]) if len(vr_history) >= 24 else vr_val

    # 区域判定
    if vr_val > 450:
        zone = '头部警戒区'
        signal = 'bearish'
    elif vr_val > 300:
        zone = '超买区'
        signal = 'bearish_warning'
    elif vr_val > 160:
        zone = '强势区'
        signal = 'bullish'
    elif vr_val > 100:
        zone = '偏强区'
        signal = 'mild_bullish'
    elif vr_val > 70:
        zone = '安全区'
        signal = 'neutral'
    elif vr_val > 40:
        zone = '偏弱区'
        signal = 'mild_bearish'
    else:
        zone = '底部区域'
        signal = 'contrarian_bullish'

    # 背离检测
    vr_arr = np.array(vr_history[-20:])
    price_arr = close[-20:]
    vr_divergence = 'none'
    if len(vr_arr) >= 10:
        if price_arr[-1] > price_arr[-10] and vr_arr[-1] < vr_arr[-10]:
            vr_divergence = '顶背离(价升VR降)'
        elif price_arr[-1] < price_arr[-10] and vr_arr[-1] > vr_arr[-10]:
            vr_divergence = '底背离(价跌VR升)'

    return {
        'vr_value': round(vr_val, 2),
        'vr_ma': round(vr_ma, 2),
        'vr_zone': zone,
        'vr_signal': signal,
        'vr_divergence': vr_divergence,
        'vr_avs': round(avs, 0),
        'vr_bvs': round(bvs, 0),
        'vr_cvs': round(cvs, 0),
    }


def _empty_vr_result():
    return {'vr_value': 100, 'vr_ma': 100, 'vr_zone': '数据不足', 'vr_signal': 'neutral', 'vr_divergence': 'none', 'vr_avs': 0, 'vr_bvs': 0, 'vr_cvs': 0}


# ===================== 19. 量价关系代数 (Andre Busch) =====================
# (+)+(+) = (+) 强: 价升量增 → 持有/加仓
# (+)+(-) = (-) 弱: 价升量缩 → 准备离场
# (-)+(+) = (-) 强空: 价跌量增 → 持有/加空
# (-)+(-) = (+) 弱空衰减: 价跌量缩 → 准备离场空头

def volume_price_algebra(df, lookback=5):
    """Busch量价代数系统"""
    if len(df) < lookback + 20:
        return _empty_vpa_result()

    close = df['close'].astype(float).values
    vol = df['volume'].astype(float).values
    n = len(close)

    price_up = close[-1] > close[-lookback]
    vol_ma20 = np.mean(vol[-21:-1])
    vol_up = vol[-1] > vol_ma20 * 1.1

    if price_up and vol_up:
        formula = '(+)+(+) = (+)'
        interpretation = '强: 价升量增，趋势健康'
        action = '持有/加仓多头'
        signal = 'bullish'
    elif price_up and not vol_up:
        formula = '(+)+(-) = (-)'
        interpretation = '弱: 价升量缩，多头乏力'
        action = '准备减仓/离场'
        signal = 'bearish_warning'
    elif not price_up and vol_up:
        formula = '(-)+(+) = (-)'
        interpretation = '强空: 价跌量增，空方主导'
        action = '持有/加仓空头'
        signal = 'bearish'
    else:
        formula = '(-)+(-) = (+)'
        interpretation = '弱空衰减: 价跌量缩，空方力竭'
        action = '准备回补/观望'
        signal = 'bullish_warning'

    # 天量检测
    vol_max_60 = np.max(vol[-61:-1]) if len(vol) >= 61 else np.max(vol[:-1])
    is_extreme_vol = vol[-1] > vol_max_60 * 0.95
    extreme_warning = ''
    if is_extreme_vol:
        extreme_warning = '⚠天量: 趋势衰竭信号，最后一批追涨/杀跌者入场'

    return {
        'vpa_formula': formula,
        'vpa_interpretation': interpretation,
        'vpa_action': action,
        'vpa_signal': signal,
        'vpa_price_up': price_up,
        'vpa_vol_up': vol_up,
        'vpa_is_extreme_vol': is_extreme_vol,
        'vpa_extreme_warning': extreme_warning,
    }


def _empty_vpa_result():
    return {'vpa_formula': '', 'vpa_interpretation': '数据不足', 'vpa_action': '观望', 'vpa_signal': 'neutral', 'vpa_price_up': False, 'vpa_vol_up': False, 'vpa_is_extreme_vol': False, 'vpa_extreme_warning': ''}


# ===================== 20. 2560战法 (Andre Busch) =====================
# MA5/MA25交叉 + 量MA5/MA60交叉 + MA25方向过滤

def busch_2560_strategy(df):
    """
    Busch 2560战法
    做多条件: MA5上穿MA25 AND 量MA5上穿量MA60 AND MA25上行
    做空条件: 反向
    周线版: MA20/MA55 (即MA4周/MA11周)
    """
    if len(df) < 65:
        return _empty_2560_result()

    close = df['close'].astype(float).values
    vol = df['volume'].astype(float).values

    ma5 = talib.SMA(close, 5)
    ma25 = talib.SMA(close, 25)
    vol_ma5 = talib.SMA(vol, 5)
    vol_ma60 = talib.SMA(vol, 60)

    # 当前值
    ma5_now = float(ma5[-1])
    ma25_now = float(ma25[-1])
    vol_ma5_now = float(vol_ma5[-1])
    vol_ma60_now = float(vol_ma60[-1])

    # 前值(判断交叉)
    ma5_prev = float(ma5[-2]) if len(ma5) >= 2 else ma5_now
    ma25_prev = float(ma25[-2]) if len(ma25) >= 2 else ma25_now
    vol_ma5_prev = float(vol_ma5[-2]) if len(vol_ma5) >= 2 else vol_ma5_now
    vol_ma60_prev = float(vol_ma60[-2]) if len(vol_ma60) >= 2 else vol_ma60_now

    # MA25方向
    ma25_direction = '上行' if ma25_now > float(ma25[-5]) else '下行' if ma25_now < float(ma25[-5]) else '走平'

    # 交叉检测
    ma_golden_cross = ma5_prev <= ma25_prev and ma5_now > ma25_now
    ma_dead_cross = ma5_prev >= ma25_prev and ma5_now < ma25_now
    vol_golden_cross = vol_ma5_prev <= vol_ma60_prev and vol_ma5_now > vol_ma60_now
    vol_dead_cross = vol_ma5_prev >= vol_ma60_prev and vol_ma5_now < vol_ma60_now

    # 综合信号
    if ma_golden_cross and vol_golden_cross and ma25_direction == '上行':
        signal_2560 = 'strong_buy'
        desc = '2560战法买入: MA5金叉MA25 + 量金叉 + MA25上行'
    elif ma_dead_cross and not vol_golden_cross:
        signal_2560 = 'sell'
        desc = '2560战法卖出: MA5死叉MA25'
    elif ma5_now > ma25_now and ma25_direction == '上行':
        signal_2560 = 'hold_long'
        desc = '2560持仓: 价格在MA25上方且MA25上行'
    elif ma5_now < ma25_now and ma25_direction == '下行':
        signal_2560 = 'hold_short'
        desc = '2560空仓/观望: 价格在MA25下方且MA25下行'
    else:
        signal_2560 = 'neutral'
        desc = '2560中性: 条件不满足'

    # 周线版 (用日线近似: MA20=4周, MA55≈11周)
    ma20 = talib.SMA(close, 20)
    ma55 = talib.SMA(close, 55)
    weekly_signal = 'neutral'
    if len(ma20) >= 3 and len(ma55) >= 3:
        if ma20[-1] > ma55[-1] and ma20[-2] <= ma55[-2]:
            weekly_signal = 'weekly_golden_cross'
        elif ma20[-1] < ma55[-1] and ma20[-2] >= ma55[-2]:
            weekly_signal = 'weekly_dead_cross'

    return {
        'b2560_signal': signal_2560,
        'b2560_description': desc,
        'b2560_ma5': round(ma5_now, 2),
        'b2560_ma25': round(ma25_now, 2),
        'b2560_ma25_direction': ma25_direction,
        'b2560_vol_ma5': round(vol_ma5_now, 0),
        'b2560_vol_ma60': round(vol_ma60_now, 0),
        'b2560_ma_golden_cross': ma_golden_cross,
        'b2560_ma_dead_cross': ma_dead_cross,
        'b2560_vol_golden_cross': vol_golden_cross,
        'b2560_weekly_signal': weekly_signal,
    }


def _empty_2560_result():
    return {'b2560_signal': 'neutral', 'b2560_description': '数据不足', 'b2560_ma5': 0, 'b2560_ma25': 0, 'b2560_ma25_direction': '未知', 'b2560_vol_ma5': 0, 'b2560_vol_ma60': 0, 'b2560_ma_golden_cross': False, 'b2560_ma_dead_cross': False, 'b2560_vol_golden_cross': False, 'b2560_weekly_signal': 'neutral'}


# ===================== 21. 市场四阶段检测 (证券交易之道) =====================
# 筑底→拉升→盘头→下跌 四阶段循环

def market_phase_detection(df):
    """
    市场四阶段模型
    基于MA排列、成交量特征、价格结构综合判定
    返回当前所处阶段及置信度
    """
    if len(df) < 120:
        return _empty_phase_result()

    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    vol = df['volume'].astype(float).values
    n = len(close)

    ma20 = talib.SMA(close, 20)
    ma60 = talib.SMA(close, 60)
    ma120 = talib.SMA(close, 120) if n >= 120 else ma60

    # 价格结构
    high_60 = np.max(high[-60:])
    low_60 = np.min(low[-60:])
    price_range_60 = high_60 - low_60
    price_position = (close[-1] - low_60) / price_range_60 if price_range_60 > 0 else 0.5

    # 近期涨跌幅
    ret_20 = (close[-1] - close[-20]) / close[-20] if close[-20] > 0 else 0
    ret_60 = (close[-1] - close[-60]) / close[-60] if close[-60] > 0 else 0

    # 成交量特征
    vol_ma20 = np.mean(vol[-20:])
    vol_ma60 = np.mean(vol[-60:])
    vol_ratio = vol_ma20 / vol_ma60 if vol_ma60 > 0 else 1.0

    # 波动率特征
    atr14 = talib.ATR(high, low, close, 14)
    atr_now = float(atr14[-1])
    atr_ma60 = np.mean(atr14[-60:]) if len(atr14) >= 60 else atr_now
    atr_ratio = atr_now / atr_ma60 if atr_ma60 > 0 else 1.0

    phase = 'unknown'
    phase_confidence = 0.0
    phase_signals = []

    # 阶段判定逻辑
    if ret_60 < -0.15 and price_position < 0.35:
        # 长期下跌后低位盘整
        if atr_ratio < 0.85 and vol_ratio < 0.8:
            phase = '筑底'
            phase_confidence = 0.65
            phase_signals.append('长期下跌后低位缩量窄幅震荡')
        elif vol_ratio > 1.2 and abs(ret_20) < 0.05:
            phase = '筑底'
            phase_confidence = 0.55
            phase_signals.append('低位放量止跌，底部吸筹迹象')

    if ma20[-1] > ma60[-1] and ret_20 > 0.03 and vol_ratio > 1.1:
        if phase == '筑底':
            phase = '拉升初期'
            phase_confidence = 0.7
            phase_signals.append('突破底部区域，放量启动')
        elif ret_20 > 0.08:
            phase = '拉升'
            phase_confidence = 0.75
            phase_signals.append('趋势加速，量价配合')

    if ret_60 > 0.30 and price_position > 0.7:
        if vol_ratio < 0.9 and atr_ratio > 1.15:
            phase = '盘头'
            phase_confidence = 0.6
            phase_signals.append('高位放量滞涨或缩量，波动加大')
        elif ret_20 < 0 and vol_ratio > 1.1:
            phase = '盘头'
            phase_confidence = 0.55
            phase_signals.append('高位量增价跌，出货迹象')

    if phase == 'unknown':
        if ma20[-1] < ma60[-1] and ret_20 < -0.02:
            phase = '下跌'
            phase_confidence = 0.6
            phase_signals.append('均线空头排列，持续下行')
        elif ret_60 < -0.1 and vol_ratio < 0.7:
            phase = '下跌'
            phase_confidence = 0.55
            phase_signals.append('缩量阴跌')

    # 三势判定 (大势/中势/小势)
    if len(ma120) > 0 and ma120[-1] > 0:
        big_trend = '牛市' if close[-1] > ma120[-1] else '熊市'
    else:
        big_trend = '未知'

    if ma20[-1] > ma60[-1] > 0:
        mid_trend = '上升'
    elif ma20[-1] < ma60[-1]:
        mid_trend = '下降'
    else:
        mid_trend = '震荡'

    if ret_20 > 0.03:
        small_trend = '上升'
    elif ret_20 < -0.03:
        small_trend = '下降'
    else:
        small_trend = '震荡'

    # 趋势可信度权重
    tf_scores = {'牛市+上升+上升': 0.9, '牛市+上升+震荡': 0.7, '牛市+震荡+上升': 0.65,
                 '熊市+下降+下降': 0.9, '熊市+下降+震荡': 0.7, '熊市+震荡+下降': 0.65}
    tf_key = f'{big_trend}+{mid_trend}+{small_trend}'
    trend_credibility = tf_scores.get(tf_key, 0.5)

    return {
        'mp_phase': phase,
        'mp_confidence': round(phase_confidence, 2),
        'mp_signals': phase_signals,
        'mp_price_position': round(price_position, 3),
        'mp_ret_20d': round(ret_20, 4),
        'mp_ret_60d': round(ret_60, 4),
        'mp_vol_ratio': round(vol_ratio, 3),
        'mp_atr_ratio': round(atr_ratio, 3),
        'mp_big_trend': big_trend,
        'mp_mid_trend': mid_trend,
        'mp_small_trend': small_trend,
        'mp_trend_credibility': round(trend_credibility, 2),
    }


def _empty_phase_result():
    return {'mp_phase': 'unknown', 'mp_confidence': 0.0, 'mp_signals': [], 'mp_price_position': 0.5, 'mp_ret_20d': 0, 'mp_ret_60d': 0, 'mp_vol_ratio': 1.0, 'mp_atr_ratio': 1.0, 'mp_big_trend': '未知', 'mp_mid_trend': '未知', 'mp_small_trend': '未知', 'mp_trend_credibility': 0.5}


# ===================== 22. DMI增强 (Andre Busch) =====================
# Busch改良DMI用法: 在空方力量最大/开始减弱时入场(而非单纯DM+/DM-交叉)

def busch_dmi_analysis(df):
    """
    Busch DMI分析法
    最佳入场: DM-达到最大值后开始回落，DM+开始增强 → 转折点
    弱势市场: DM+和DM-都在ADX之下
    大动量: ADX从DM+和DM-之间向上穿越
    """
    if len(df) < 30:
        return _empty_busch_dmi_result()

    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values

    pdi = talib.PLUS_DI(high, low, close, 14)
    mdi = talib.MINUS_DI(high, low, close, 14)
    adx = talib.ADX(high, low, close, 14)

    pdi_now = float(pdi[-1])
    mdi_now = float(mdi[-1])
    adx_now = float(adx[-1])

    # Busch关键: DM-是否处于极大值区域并开始回落
    mdi_max_20 = np.max(mdi[-21:-1]) if len(mdi) >= 21 else mdi_now
    mdi_peaking = mdi_now < mdi_max_20 * 0.92 and mdi_now < float(mdi[-3])

    # DM+是否开始增强
    pdi_rising = pdi_now > float(pdi[-3]) and pdi_now > float(pdi[-5])

    # ADX位置
    adx_above_di = adx_now > pdi_now and adx_now > mdi_now

    # Busch转折信号
    if mdi_peaking and pdi_rising:
        busch_signal = 'bullish_reversal'
        busch_desc = 'Busch转折买入: DM-见顶回落，DM+开始增强'
    elif pdi_now > mdi_now and adx_now > 25:
        busch_signal = 'bullish_trend'
        busch_desc = 'Busch趋势做多: DM+>DM-且ADX>25'
    elif mdi_now > pdi_now and adx_now > 25:
        busch_signal = 'bearish_trend'
        busch_desc = 'Busch趋势做空: DM->DM+且ADX>25'
    elif adx_above_di:
        busch_signal = 'weak_market'
        busch_desc = 'Busch弱势: DM+和DM-都在ADX之下，无方向'
    else:
        busch_signal = 'neutral'
        busch_desc = 'Busch中性'

    # 大动量检测
    adx_prev = float(adx[-5])
    max_di_prev = max(float(pdi[-5]), float(mdi[-5]))
    min_di_prev = min(float(pdi[-5]), float(mdi[-5]))
    big_momentum = adx_prev < min_di_prev and adx_now > max_di_prev

    return {
        'bdmi_signal': busch_signal,
        'bdmi_description': busch_desc,
        'bdmi_pdi': round(pdi_now, 2),
        'bdmi_mdi': round(mdi_now, 2),
        'bdmi_adx': round(adx_now, 2),
        'bdmi_mdi_peaking': mdi_peaking,
        'bdmi_pdi_rising': pdi_rising,
        'bdmi_big_momentum': big_momentum,
        'bdmi_weak_market': adx_above_di,
    }


def _empty_busch_dmi_result():
    return {'bdmi_signal': 'neutral', 'bdmi_description': '数据不足', 'bdmi_pdi': 0, 'bdmi_mdi': 0, 'bdmi_adx': 0, 'bdmi_mdi_peaking': False, 'bdmi_pdi_rising': False, 'bdmi_big_momentum': False, 'bdmi_weak_market': False}


# ===================== 23. 艾略特波浪增强函数 =====================
# 12项增强: 修正浪模式 / 浪3vs浪5对比 / 完整斐波那契 / 时间比率
# 交替规则 / 通道技术 / 浪5目标 / 延长浪检测 / 浪个性评分
# 多时间框架验证 / 失效级别 / 倾斜三角形

# 完整斐波那契比率集
FIB_RETRACEMENTS_FULL = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_EXTENSIONS_FULL = [0.618, 1.0, 1.272, 1.618, 2.0, 2.618, 4.236]
FIB_TIME_NUMBERS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]


def ew_detect_corrective_pattern(pivots, trend):
    """识别修正浪类型: zigzag(5-3-5), flat(3-3-5), triangle(A-B-C-D-E)"""
    if len(pivots) < 4:
        return {'corrective_type': 'unknown', 'corrective_confidence': 0.0}

    points = [{'idx': p[0], 'price': p[1], 'type': p[3]} for p in pivots[-6:]]
    result = {'corrective_type': 'unknown', 'corrective_confidence': 0.0,
              'c_targets': {}, 'triangle_thrust': None}

    if len(points) >= 4:
        # 提取A-B-C三浪
        a_len = abs(points[-3]['price'] - points[-4]['price']) if len(points) >= 4 else 0
        b_move = abs(points[-2]['price'] - points[-3]['price']) if len(points) >= 3 else 0
        c_len = abs(points[-1]['price'] - points[-2]['price']) if len(points) >= 2 else 0

        if a_len > 0:
            b_retrace = b_move / a_len
            # Zigzag: B回撤38-62%, A是5波(尖锐)
            if 0.3 < b_retrace < 0.65:
                result['corrective_type'] = 'zigzag'
                result['corrective_confidence'] = 0.6
                c_target_eq = points[-2]['price'] + (points[-4]['price'] - points[-3]['price']) * 1.0
                c_target_1618 = points[-2]['price'] + (points[-4]['price'] - points[-3]['price']) * 1.618
                result['c_targets'] = {'c_eq_a': round(c_target_eq, 2), 'c_1618a': round(c_target_1618, 2)}
            # Flat: B回到A起点附近
            elif b_retrace > 0.8:
                result['corrective_type'] = 'expanded_flat' if b_retrace > 1.0 else 'regular_flat'
                result['corrective_confidence'] = 0.55
                if b_retrace > 1.0:
                    c_extra = (points[-3]['price'] - points[-4]['price']) * 1.236
                    result['c_targets'] = {'c_1236a': round(points[-2]['price'] + c_extra, 2)}
                else:
                    result['c_targets'] = {'c_100a': round(points[-2]['price'] + a_len, 2)}

    return result


def ew_detect_extension(wp):
    """检测哪一浪延长 (仅一浪可延长)"""
    if not wp:
        return {'extended_wave': None, 'extension_magnitude': 1.0}

    w1 = abs(wp['wave_1_top']['price'] - wp['wave_0_start']['price']) if wp.get('wave_1_top') and wp.get('wave_0_start') else 0
    w3 = abs(wp['wave_3_top']['price'] - wp['wave_2_bottom']['price']) if wp.get('wave_3_top') and wp.get('wave_2_bottom') else 0
    w5 = abs(wp['wave_5_top']['price'] - wp['wave_4_bottom']['price']) if wp.get('wave_5_top') and wp.get('wave_4_bottom') else 0

    if w1 + w3 + w5 == 0:
        return {'extended_wave': None, 'extension_magnitude': 1.0}

    waves = sorted([(w1, 1), (w3, 3), (w5, 5)], reverse=True)
    if waves[0][0] > 0 and waves[1][0] > 0 and waves[0][0] / waves[1][0] >= 1.618:
        return {'extended_wave': waves[0][1], 'extension_magnitude': round(waves[0][0] / waves[1][0], 2)}
    return {'extended_wave': None, 'extension_magnitude': 1.0}


def ew_compute_wave5_targets(wp, trend):
    """计算浪5所有可能的斐波那契目标"""
    targets = {}
    if not wp.get('wave_0_start') or not wp.get('wave_1_top') or not wp.get('wave_4_bottom'):
        return targets

    w1_len = abs(wp['wave_1_top']['price'] - wp['wave_0_start']['price'])
    w4_price = wp['wave_4_bottom']['price']
    direction = 1 if trend == 'up' else -1

    # 基于浪1的倍数
    for mult, label in [(1.0, 'w5=w1'), (1.618, 'w5=1.618w1'), (2.618, 'w5=2.618w1'), (0.618, 'w5=0.618w1')]:
        targets[label] = round(w4_price + direction * w1_len * mult, 2)

    # 基于浪0→浪3的0.618
    if wp.get('wave_3_top'):
        w03 = abs(wp['wave_3_top']['price'] - wp['wave_0_start']['price'])
        targets['w5=0.618(w0→w3)'] = round(w4_price + direction * w03 * 0.618, 2)

    return targets


def ew_fibonacci_time_ratios(pivots, wave_points):
    """计算波浪时间比率 (斐波那契时间)"""
    if not wave_points or len(pivots) < 4:
        return {'time_ratios': {}, 'time_projection': None}

    # 提取各浪的时间跨度
    def bar_count(p_from, p_to):
        if p_from and p_to:
            return abs(p_to['idx'] - p_from['idx'])
        return None

    wp = wave_points
    times = {}
    if wp.get('wave_0_start') and wp.get('wave_1_top'):
        times['t1'] = bar_count(wp['wave_0_start'], wp['wave_1_top'])
    if wp.get('wave_1_top') and wp.get('wave_2_bottom'):
        times['t2'] = bar_count(wp['wave_1_top'], wp['wave_2_bottom'])
    if wp.get('wave_2_bottom') and wp.get('wave_3_top'):
        times['t3'] = bar_count(wp['wave_2_bottom'], wp['wave_3_top'])
    if wp.get('wave_3_top') and wp.get('wave_4_bottom'):
        times['t4'] = bar_count(wp['wave_3_top'], wp['wave_4_bottom'])

    ratios = {}
    if times.get('t1') and times.get('t2') and times['t1'] > 0:
        ratios['t2/t1'] = round(times['t2'] / times['t1'], 2)
    if times.get('t2') and times.get('t3') and times['t2'] > 0:
        ratios['t3/t2'] = round(times['t3'] / times['t2'], 2)

    # 时间投影
    projection = None
    if times.get('t1') and times.get('t4'):
        # 寻找最近斐波那契数作为下一时间目标
        base = max(times['t1'], times.get('t4', 0))
        for fn in FIB_TIME_NUMBERS:
            if fn > base:
                projection = fn
                break

    return {'time_ratios': ratios, 'time_durations': times, 'time_projection': projection}


def ew_alternation_rule(wp):
    """
    交替规则: 浪2和浪4应在以下方面不同:
    1. 形态: 尖锐(zigzag) vs 横向(flat/triangle)
    2. 回调深度: 深 vs 浅
    3. 复杂度: 简单 vs 复杂
    """
    if not wp.get('wave_1_top') or not wp.get('wave_2_bottom') or not wp.get('wave_3_top') or not wp.get('wave_4_bottom'):
        return {'alternation_score': 0.0, 'alternation_valid': False}

    w0 = wp['wave_0_start']['price']
    w1 = wp['wave_1_top']['price']
    w2 = wp['wave_2_bottom']['price']
    w3 = wp['wave_3_top']['price']
    w4 = wp['wave_4_bottom']['price']

    wave1_len = w1 - w0
    if wave1_len == 0:
        return {'alternation_score': 0.0, 'alternation_valid': False}

    w2_retrace = (w1 - w2) / wave1_len
    wave3_len = w3 - w2
    w4_retrace = (w3 - w4) / wave3_len if wave3_len > 0 else 0

    score = 0.0
    reasons = []

    # 深度交替检验
    if abs(w2_retrace - w4_retrace) > 0.2:
        score += 0.25
        reasons.append(f'深度交替: 浪2回撤{w2_retrace:.0%}, 浪4回撤{w4_retrace:.0%}')

    # 形态提示:浪2通常是尖锐(zigzag, 深回撤), 浪4通常是横向(flat, 浅回撤)
    if w2_retrace > 0.5 and w4_retrace < 0.5:
        score += 0.3
        reasons.append('经典交替: 浪2深回调(zigzag) → 浪4浅回调(flat)')
    elif w2_retrace < 0.4 and w4_retrace > 0.5:
        score += 0.2
        reasons.append('反向交替: 浪2浅回调 → 浪4深回调')

    # 时间交替
    valid = score >= 0.3
    return {'alternation_score': round(score, 2), 'alternation_valid': valid,
            'alternation_reasons': reasons, 'w2_retrace_pct': round(w2_retrace, 3),
            'w4_retrace_pct': round(w4_retrace, 3)}


def ew_channeling(wp, stage):
    """
    艾略特通道技术
    Stage 1: (浪0-浪2底) 连线下轨，过浪1顶画平行线上轨
    Stage 2: (浪2底-浪4底) 连线下轨，过浪3顶画平行线上轨
    """
    if not wp.get('wave_0_start'):
        return {'channel_upper': None, 'channel_lower': None, 'channel_stage': stage}

    w0 = wp['wave_0_start']
    result = {'channel_stage': stage}

    if stage == 1 and wp.get('wave_1_top') and wp.get('wave_2_bottom'):
        slope = (wp['wave_2_bottom']['price'] - w0['price']) / max(1, wp['wave_2_bottom']['idx'] - w0['idx'])
        proj_idx = wp['wave_2_bottom']['idx'] + 10
        upper_at_proj = wp['wave_1_top']['price'] + slope * (proj_idx - wp['wave_1_top']['idx'])
        lower_at_proj = wp['wave_2_bottom']['price'] + slope * (proj_idx - wp['wave_2_bottom']['idx'])
        result['channel_upper'] = round(upper_at_proj, 2)
        result['channel_lower'] = round(lower_at_proj, 2)
        result['channel_slope'] = round(slope, 4)

    elif stage == 2 and wp.get('wave_2_bottom') and wp.get('wave_3_top') and wp.get('wave_4_bottom'):
        w2 = wp['wave_2_bottom']
        w4 = wp['wave_4_bottom']
        slope = (w4['price'] - w2['price']) / max(1, w4['idx'] - w2['idx'])
        proj_idx = w4['idx'] + 10
        upper_at_proj = wp['wave_3_top']['price'] + slope * (proj_idx - wp['wave_3_top']['idx'])
        result['channel_upper'] = round(upper_at_proj, 2)
        result['channel_lower'] = round(w4['price'] + slope * (proj_idx - w4['idx']), 2)
        result['channel_slope'] = round(slope, 4)

    return result


def ew_score_wave_personality(df, wp, wave_label, trend):
    """浪个性评分:根据量价行为与预期模板的吻合度"""
    if not wp or len(df) < 60:
        return {'personality_score': 0.5, 'personality_checks': []}

    vol = df['volume'].astype(float).values
    checks = []
    score = 0.5

    try:
        wave_num = int(str(wave_label)[-1]) if wave_label and str(wave_label)[-1].isdigit() else None
    except (ValueError, IndexError):
        wave_num = None

    if wave_num == 3 and wp.get('wave_1_top') and wp.get('wave_3_top'):
        w1_range = range(wp['wave_0_start']['idx'], wp['wave_1_top']['idx'])
        w3_range = range(wp['wave_2_bottom']['idx'], wp['wave_3_top']['idx'])
        vol_w1 = np.mean([vol[i] for i in w1_range if 0 <= i < len(vol)]) if w1_range else 0
        vol_w3 = np.mean([vol[i] for i in w3_range if 0 <= i < len(vol)]) if w3_range else 0
        if vol_w1 > 0 and vol_w3 > vol_w1 * 1.2:
            score += 0.2
            checks.append('浪3放量(>1.2x浪1)')

    if wave_num == 5 and wp.get('wave_3_top') and wp.get('wave_5_top'):
        w3_top_price = wp['wave_3_top']['price']
        w5_top_price = wp['wave_5_top']['price']
        if trend == 'up' and w5_top_price > w3_top_price:
            # 检查动量背离: 价格更高但RSI更低
            rsi_arr = talib.RSI(df['close'].astype(float).values, 14)
            rsi_w3 = rsi_arr[wp['wave_3_top']['idx']] if wp['wave_3_top']['idx'] < len(rsi_arr) else 50
            rsi_now = rsi_arr[-1]
            if rsi_now < rsi_w3 - 5:
                score += 0.2
                checks.append('浪5动量背离(RSI)')
            else:
                score += 0.05
                checks.append('浪5无背离(可能延长)')

    return {'personality_score': round(min(1.0, score), 2), 'personality_checks': checks}


def ew_multi_timeframe_verify(df_daily, wave_result_daily, df_weekly=None, df_hourly=None):
    """多时间框架波浪验证"""
    boost = 0.0
    checks = []

    if df_weekly is not None and len(df_weekly) >= 60:
        weekly_close = df_weekly['close'].astype(float).values
        ma20_w = talib.SMA(weekly_close, 20)
        if len(ma20_w) >= 2:
            weekly_trend = 'up' if ma20_w[-1] > ma20_w[-5] else 'down'
            daily_trend = wave_result_daily.get('trend_direction', 'unknown')
            if weekly_trend == daily_trend:
                boost += 0.1
                checks.append('周线趋势与日线一致')

    if df_hourly is not None and len(df_hourly) >= 60:
        daily_wave = wave_result_daily.get('current_wave', '')
        if '浪3' in str(daily_wave) or '浪5' in str(daily_wave):
            hr_ew = elliott_wave_analysis(df_hourly)
            if hr_ew.get('trend_direction') == wave_result_daily.get('trend_direction'):
                boost += 0.08
                checks.append('小时线微观结构确认')

    return {'mtf_confidence_boost': round(boost, 2), 'mtf_checks': checks}


def ew_compute_invalidation_levels(wp, trend):
    """计算精确的波浪失效级别（硬止损位）"""
    levels = {}
    if not wp:
        return levels

    if trend == 'up':
        if wp.get('wave_0_start'):
            levels['rule1_invalidation'] = wp['wave_0_start']['price']
        if wp.get('wave_1_top'):
            levels['rule3_invalidation'] = wp['wave_1_top']['price']
        if wp.get('wave_2_bottom') and wp.get('wave_1_top') and wp.get('wave_0_start'):
            w1_len = wp['wave_1_top']['price'] - wp['wave_0_start']['price']
            levels['wave3_minimum'] = round(wp['wave_2_bottom']['price'] + w1_len * 1.001, 2)
    else:
        if wp.get('wave_0_start'):
            levels['rule1_invalidation'] = wp['wave_0_start']['price']
        if wp.get('wave_1_bottom'):
            levels['rule3_invalidation'] = wp['wave_1_bottom']['price']

    return levels


def ew_detect_diagonal_triangle(swing_points, trend):
    """
    检测倾斜三角形（楔形）
    Ending Diagonal (3-3-3-3-3): 出现在浪5或浪C位置
    Leading Diagonal (5-3-5-3-5): 出现在浪1或浪A位置
    """
    if len(swing_points) < 5:
        return {'is_diagonal': False}

    points = [{'idx': p[0], 'price': p[1], 'type': p[3]} for p in swing_points[-5:]]

    # 上轨: 连接点0和点2, 下轨: 连接点1和点3
    if points[0]['idx'] == points[2]['idx'] or points[1]['idx'] == points[3]['idx']:
        return {'is_diagonal': False}

    upper_slope = (points[2]['price'] - points[0]['price']) / (points[2]['idx'] - points[0]['idx'])
    lower_slope = (points[3]['price'] - points[1]['price']) / (points[3]['idx'] - points[1]['idx'])

    if trend == 'up':
        is_converging = lower_slope > 0 and upper_slope > 0 and lower_slope > upper_slope
    else:
        is_converging = lower_slope < 0 and upper_slope < 0 and abs(lower_slope) < abs(upper_slope)

    if not is_converging:
        return {'is_diagonal': False}

    # 第5子浪明显慢于第3子浪
    wave3_sub = abs(points[2]['price'] - points[1]['price'])
    wave5_sub = abs(points[4]['price'] - points[3]['price'])

    if wave3_sub > 0 and wave5_sub / wave3_sub < 0.75:
        diag_type = 'ending' if len(swing_points) >= 7 else 'possible_ending'
        diag_height = abs(points[0]['price'] - points[4]['price'])
        thrust = points[4]['price'] + diag_height * (1.0 if diag_type == 'ending' else 1.618)
        return {'is_diagonal': True, 'type': diag_type, 'convergence_ratio': round(wave5_sub / wave3_sub, 2),
                'post_thrust_target': round(thrust, 2)}

    return {'is_diagonal': False}


def ew_enhanced_analysis(df, trend_direction='auto'):
    """
    增强版艾略特波浪分析:整合12项增强
    在原有elliott_wave_analysis基础上叠加所有新功能
    """
    base_result = elliott_wave_analysis(df, trend_direction)
    if base_result['wave_pattern'] in ['数据不足', '枢轴不足']:
        return {**base_result, 'ew_enhanced': {}}

    swing_highs, swing_lows, pivots = find_swing_points(df, min_bars=2, min_pct=0.01)
    wp = base_result.get('wave_points', {})
    trend = base_result.get('trend_direction', 'up')

    enhanced = {}

    # 1. 修正浪模式识别
    enhanced['corrective_pattern'] = ew_detect_corrective_pattern(pivots, trend)

    # 2. 延长浪检测
    if wp:
        enhanced['extension'] = ew_detect_extension(wp)
    else:
        enhanced['extension'] = {'extended_wave': None, 'extension_magnitude': 1.0}

    # 3. 浪5斐波那契目标
    enhanced['wave5_targets'] = ew_compute_wave5_targets(wp, trend)

    # 4. 斐波那契时间比率
    enhanced['time_ratios'] = ew_fibonacci_time_ratios(pivots, wp)

    # 5. 交替规则
    enhanced['alternation'] = ew_alternation_rule(wp)

    # 6. 通道技术 (当前阶段)
    if wp.get('wave_4_bottom'):
        enhanced['channel'] = ew_channeling(wp, stage=2)
    else:
        enhanced['channel'] = ew_channeling(wp, stage=1)

    # 7. 浪个性评分
    enhanced['personality'] = ew_score_wave_personality(df, wp, base_result.get('current_wave', ''), trend)

    # 8. 失效级别
    enhanced['invalidation_levels'] = ew_compute_invalidation_levels(wp, trend)

    # 9. 倾斜三角形检测
    enhanced['diagonal'] = ew_detect_diagonal_triangle(pivots, trend)

    # 10. 完整斐波那契回调位
    if wp.get('wave_0_start') and wp.get('wave_3_top'):
        w0_price = wp['wave_0_start']['price']
        w3_price = wp['wave_3_top']['price']
        enhanced['fib_retracements_full'] = {f'{lvl:.1%}': round(w3_price - (w3_price - w0_price) * lvl, 2)
                                              for lvl in FIB_RETRACEMENTS_FULL}

    # 合并置信度
    conf = base_result.get('confidence', 0.0)
    if enhanced['alternation'].get('alternation_valid'):
        conf += 0.05
    if enhanced['diagonal'].get('is_diagonal'):
        conf += 0.03
    if enhanced['extension'].get('extended_wave'):
        conf += 0.05
    conf = min(0.95, conf)

    return {**base_result, 'ew_enhanced': enhanced, 'confidence': round(conf, 2)}


# ===================== 24. 行为风控护栏 (赢家的习惯) =====================
# 交易前检查清单 + 冷却期 + 报复交易检测 + 部分止盈

def behavioral_guardrails(trade_history, current_drawdown_pct=0.0, consecutive_losses=0):
    """
    行为风控护栏
    返回: 是否允许交易 + 风险等级 + 冷却期剩余
    """
    guard = {
        'allow_trading': True,
        'risk_level': 'normal',
        'blocks': [],
        'warnings': [],
    }

    # 1. 连续亏损熔断
    if consecutive_losses >= 4:
        guard['allow_trading'] = False
        guard['risk_level'] = 'critical'
        guard['blocks'].append(f'连续亏损{consecutive_losses}次:强制停止交易，至少冷却1天')
    elif consecutive_losses >= 3:
        guard['risk_level'] = 'high'
        guard['warnings'].append(f'连续亏损{consecutive_losses}次:建议减仓50%并冷却半天')
    elif consecutive_losses >= 2:
        guard['risk_level'] = 'elevated'
        guard['warnings'].append(f'连续亏损{consecutive_losses}次:下一笔仓位减半')

    # 2. 回撤熔断 (Busch规则: 25%上限)
    if current_drawdown_pct > 0.25:
        guard['allow_trading'] = False
        guard['risk_level'] = 'critical'
        guard['blocks'].append(f'账户回撤{current_drawdown_pct:.0%}>25%:BUSCH规则强制停止交易该账户')
    elif current_drawdown_pct > 0.15:
        guard['risk_level'] = 'high'
        guard['warnings'].append(f'回撤{current_drawdown_pct:.0%}>15%:转为保守模式，仓位降至1/3')
    elif current_drawdown_pct > 0.10:
        guard['warnings'].append(f'回撤{current_drawdown_pct:.0%}>10%:注意风险')

    # 3. 报复交易检测
    if len(trade_history) >= 3:
        recent_trades = trade_history[-3:]
        if all(t['pnl'] < 0 for t in recent_trades):
            times_between = []
            for i in range(1, len(recent_trades)):
                delta = (recent_trades[i]['time'] - recent_trades[i-1]['time']).total_seconds()
                times_between.append(delta)
            if times_between and min(times_between) < 300:  # 5分钟内连续交易
                guard['allow_trading'] = False
                guard['risk_level'] = 'critical'
                guard['blocks'].append('疑似报复交易:连续亏损后5分钟内频繁操作，强制冷却30分钟')

    return guard


# ===================== 25. Busch三模式仓位管理 =====================

def busch_position_sizing(account_capital, accumulated_profit, base_risk_pct=0.02):
    """
    Busch三模式仓位管理:defensive(保守)/normal(正常)/aggressive(激进)

    三种模式区别在于风险资金池大小，但单笔交易风险始终硬封顶在2%（Busch铁律）。
    aggressive模式指可动用更多累计利润作为风险资金，而非放宽单笔止损比例。
    """
    total = account_capital + accumulated_profit

    if accumulated_profit <= 0:
        mode = 'defensive'
        risk_capital = total * 0.01
    elif accumulated_profit < total * 0.05:
        mode = 'normal'
        risk_capital = total * base_risk_pct
    else:
        # aggressive: increased risk budget from accumulated profits
        # NOTE: the "aggressive" label refers to risk CAPITAL allocation (up to ~16% of total),
        # not per-trade risk. Per-trade risk is ALWAYS capped at 2% by Busch's iron rule below.
        mode = 'aggressive'
        profit_tiers = int(accumulated_profit / (total * 0.05))
        bonus_pct = min(10, profit_tiers)  # 最多额外10%
        aggressive_pct = 0.06 + bonus_pct * 0.01
        risk_capital = total * aggressive_pct + accumulated_profit * 0.20

    # Busch铁律:单笔最大亏损≤2%（Regardless of mode, per-trade risk is hard-capped）
    max_risk_per_trade = min(risk_capital, total * 0.02)

    return {
        'busch_mode': mode,
        'busch_risk_capital': round(risk_capital, 2),
        'busch_max_risk_per_trade': round(max_risk_per_trade, 2),
        'busch_max_position_pct': round(min(0.25, risk_capital / total), 3) if total > 0 else 0,
    }


# ===================== 26. Velez短线工具 (短线交易大师) =====================

def velez_nrb_detection(df):
    """NRB (Narrow Range Bar) 检测: 窄幅K线预示反转"""
    if len(df) < 10:
        return {'nrb_signal': 'none', 'nrb_type': None, 'nrb_description': ''}

    close = df['close'].astype(float).values
    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    n = len(close)

    ranges = high - low
    avg_range_10 = np.mean(ranges[-11:-1])

    if ranges[-1] < avg_range_10 * 0.5:
        prev_direction = 'down' if close[-5] < close[-10] else 'up'
        if prev_direction == 'down':
            return {'nrb_signal': 'bullish_reversal', 'nrb_type': '底部NRB',
                    'nrb_description': '连续下跌后出现窄幅K线，底部反转信号，突破NRB高点买入'}
        else:
            return {'nrb_signal': 'bearish_reversal', 'nrb_type': '顶部NRB',
                    'nrb_description': '连续上涨后出现窄幅K线，顶部反转信号，跌破NRB低点卖出'}

    return {'nrb_signal': 'none', 'nrb_type': None, 'nrb_description': ''}


def velez_time_stop(entry_date, current_date, max_days=5):
    """Velez 5-day time stop: exit unconditionally if position held 5 days without hitting target or stop"""
    days_held = (current_date - entry_date).days
    if days_held >= max_days:
        return {'time_stop_triggered': True, 'time_stop_days': days_held,
                'time_stop_action': f'Velez规则:持仓{days_held}天(≥{max_days}天)，无条件离场'}
    return {'time_stop_triggered': False, 'time_stop_days': days_held,
            'time_stop_action': f'剩余{max_days - days_held}天'}


# ===================== 27. 量能形态分类 (成交量核心技术) =====================

def volume_pattern_classify(df, n=20):
    """
    五种量能形态分类:
    常量、聚量、变量、量能消散、地量
    """
    if len(df) < n + 10:
        return _empty_vol_classify_result()

    vol = df['volume'].astype(float).values
    close = df['close'].astype(float).values

    vol_ma5 = np.mean(vol[-6:-1])
    vol_ma20 = np.mean(vol[-21:-1])
    vol_std20 = np.std(vol[-21:-1])
    vol_cv = vol_std20 / vol_ma20 if vol_ma20 > 0 else 0

    price_trend = 'up' if close[-1] > close[-n] else 'down'

    if vol[-1] < vol_ma20 * 0.5:
        pattern = '地量'
        description = '成交量极度萎缩，底部区域或变盘前兆'
        signal = 'contrarian_bullish' if price_trend == 'down' else 'caution'
    elif vol_cv < 0.3:
        pattern = '常量'
        description = '成交量稳定，趋势延续概率高'
        signal = 'trend_continuation'
    elif vol[-1] > vol_ma20 * 1.5 and vol[-1] > vol_ma5 * 1.3:
        pattern = '聚量'
        description = '成交量急剧放大，主力资金异动'
        signal = 'breakout' if price_trend == 'up' else 'distribution'
    elif vol[-1] < vol_ma5 * 0.7 and vol_ma5 < vol_ma20:
        pattern = '量能消散'
        description = '成交量持续萎缩，趋势动能衰减'
        signal = 'trend_weakening'
    else:
        pattern = '变量'
        description = '成交量不规则波动，方向不明'
        signal = 'neutral'

    return {
        'vc_pattern': pattern,
        'vc_description': description,
        'vc_signal': signal,
        'vc_cv': round(vol_cv, 3),
        'vc_vol_ratio': round(vol[-1] / vol_ma20, 2) if vol_ma20 > 0 else 1.0,
    }


def _empty_vol_classify_result():
    return {'vc_pattern': '未知', 'vc_description': '数据不足', 'vc_signal': 'neutral', 'vc_cv': 0, 'vc_vol_ratio': 1.0}


# ===================== 28. 综合量价分析快捷入口 =====================

def comprehensive_volume_analysis(df):
    """一站式量价综合分析:VR + 量价代数 + 2560 + 量能形态 + 格兰维尔"""
    if len(df) < 65:
        return {'error': '数据不足(需要≥65根K线)'}

    vr = vr_indicator(df)
    vpa = volume_price_algebra(df)
    b2560 = busch_2560_strategy(df)
    vc = volume_pattern_classify(df)

    # 综合量价信号
    bull_signals = 0
    bear_signals = 0
    for r in [vr, vpa, b2560, vc]:
        sig = r.get('vr_signal', '') or r.get('vpa_signal', '') or r.get('b2560_signal', '') or r.get('vc_signal', '')
        if sig in ['bullish', 'strong_buy', 'hold_long', 'breakout', 'contrarian_bullish']:
            bull_signals += 1
        elif sig in ['bearish', 'sell', 'bearish_warning', 'distribution', 'trend_weakening']:
            bear_signals += 1

    if bull_signals >= 3:
        composite = '强烈看多'
    elif bull_signals >= 2:
        composite = '偏多'
    elif bear_signals >= 3:
        composite = '强烈看空'
    elif bear_signals >= 2:
        composite = '偏空'
    else:
        composite = '中性/分歧'

    return {
        'cva_vr': vr,
        'cva_vpa': vpa,
        'cva_2560': b2560,
        'cva_volume_class': vc,
        'cva_composite_signal': composite,
        'cva_bull_count': bull_signals,
        'cva_bear_count': bear_signals,
    }


# ===================== 29. 九大量价关系完整版 =====================

def volume_9_relations(df):
    """
    经典九大量价关系分析（完整版）

    1. 价涨量增 → 多头健康，趋势延续
    2. 价涨量缩 → 顶背离，上涨乏力
    3. 价涨量平 → 上涨趋缓，警惕变盘
    4. 价跌量增 → 空头强势，趋势延续
    5. 价跌量缩 → 底背离，下跌衰竭
    6. 价跌量平 → 下跌趋缓，关注止跌
    7. 价平量增 → 多空分歧加大，即将变盘
    8. 价平量缩 → 交投清淡，方向不明
    9. 价稳量增 → 底部吸筹信号
    """
    if len(df) < 22:
        return _empty_9relations()

    close = df['close'].astype(float).values
    vol = df['volume'].astype(float).values

    price_chg = (close[-1] - close[-2]) / close[-2] if close[-2] != 0 else 0
    vol_chg = (vol[-1] - vol[-2]) / vol[-2] if vol[-2] != 0 else 0
    vol_ma5 = np.mean(vol[-6:-1])
    vol_ma20 = np.mean(vol[-21:-1])

    # Price classification
    price_state = _classify_price_change(price_chg)
    # Volume classification
    vol_state = _classify_volume_change(vol_chg, vol[-1], vol_ma20)

    relation_id, relation_name, interpretation, action, strength = \
        _match_9relation(price_state, vol_state, price_chg, vol_chg)

    # Multi-period consistency check
    periods = [5, 10, 20]
    consistency = _check_vol_price_consistency(close, vol, periods)

    return {
        'relation_id': relation_id,
        'relation_name': relation_name,
        'price_state': price_state,
        'volume_state': vol_state,
        'interpretation': interpretation,
        'action': action,
        'strength': strength,
        'price_change_pct': round(price_chg * 100, 2),
        'volume_change_pct': round(vol_chg * 100, 2),
        'vol_vs_ma20': round(vol[-1] / vol_ma20, 2) if vol_ma20 > 0 else 1.0,
        'consistency': consistency,
    }


def _classify_price_change(price_chg):
    if price_chg > 0.01: return '涨'
    elif price_chg < -0.01: return '跌'
    elif abs(price_chg) <= 0.005: return '平'
    else: return '稳'


def _classify_volume_change(vol_chg, vol_now, vol_ma20):
    ratio = vol_now / vol_ma20 if vol_ma20 > 0 else 1.0
    if vol_chg > 0.15 or ratio > 1.5: return '增'
    elif vol_chg < -0.15 or ratio < 0.5: return '缩'
    elif abs(vol_chg) <= 0.08 and 0.8 <= ratio <= 1.2: return '平'
    else: return '增' if vol_chg > 0 else '缩'


def _match_9relation(price_state, vol_state, price_chg, vol_chg):
    mapping = {
        ('涨', '增'): (1, '价涨量增', '多头强势，量价配合良好，趋势延续', '顺势做多', 'strong_bullish'),
        ('涨', '缩'): (2, '价涨量缩', '顶背离，上涨动能衰减，追高需谨慎', '减仓/观望', 'bearish_warning'),
        ('涨', '平'): (3, '价涨量平', '上涨趋缓，量能未跟进，警惕短期变盘', '持仓观望', 'caution'),
        ('跌', '增'): (4, '价跌量增', '空头强势，恐慌抛售或主力出货', '止损/做空', 'strong_bearish'),
        ('跌', '缩'): (5, '价跌量缩', '底背离，抛压减轻，下跌动能衰竭', '关注反弹', 'bullish_hint'),
        ('跌', '平'): (6, '价跌量平', '下跌趋缓，卖方力量减弱，关注止跌', '观望', 'neutral'),
        ('平', '增'): (7, '价平量增', '多空分歧加剧，大资金博弈，即将变盘', '等待突破方向', 'breakout_warning'),
        ('平', '缩'): (8, '价平量缩', '交投清淡，市场缺乏方向，观望为宜', '观望', 'neutral'),
        ('稳', '增'): (9, '价稳量增', '底部吸筹信号，主力资金暗中进场', '关注做多机会', 'bullish'),
    }
    return mapping.get((price_state, vol_state),
                        (0, '量价不明', '关系不明确', '观望', 'neutral'))


def _check_vol_price_consistency(close, vol, periods):
    results = {}
    for p in periods:
        if len(close) > p:
            pc = (close[-1] - close[-p]) / close[-p] if close[-p] != 0 else 0
            vc = np.mean(vol[-p:]) / np.mean(vol[-2*p:-p]) - 1 if len(vol) >= 2*p else 0
            results[f'{p}d'] = '一致' if pc * vc >= 0 else '背离'
    return results


def _empty_9relations():
    return {
        'relation_id': 0, 'relation_name': '数据不足', 'price_state': '?',
        'volume_state': '?', 'interpretation': '', 'action': '观望',
        'strength': 'neutral', 'price_change_pct': 0, 'volume_change_pct': 0,
        'vol_vs_ma20': 1.0, 'consistency': {},
    }


# ===================== 30. 主力资金检测 =====================

def smart_money_detection(df):
    """
    主力资金流向检测

    三大工具:
    1. MFI (Money Flow Index) — 量价资金流指标
    2. VWAP偏离度 — 机构交易基准偏离
    3. A/D线 (Accumulation/Distribution) — 筹码收集/派发
    """
    if len(df) < 30:
        return _empty_smart_money()

    high = df['high'].astype(float).values
    low = df['low'].astype(float).values
    close = df['close'].astype(float).values
    vol = df['volume'].astype(float).values

    # 1. MFI (14-period)
    typical = (high + low + close) / 3
    raw_money = typical * vol
    mfi_pos = np.zeros_like(raw_money)
    mfi_neg = np.zeros_like(raw_money)
    for i in range(1, len(typical)):
        if typical[i] >= typical[i-1]:
            mfi_pos[i] = raw_money[i]
        else:
            mfi_neg[i] = raw_money[i]
    n_mfi = 14
    pos_sum = pd.Series(mfi_pos).rolling(n_mfi).sum().values
    neg_sum = pd.Series(mfi_neg).rolling(n_mfi).sum().values
    mr = np.divide(pos_sum, neg_sum, out=np.ones_like(pos_sum), where=neg_sum != 0)
    mfi = 100 - (100 / (1 + mr))
    mfi_now = mfi[-1] if len(mfi) > 0 else 50

    if mfi_now > 80:
        mfi_signal = 'overbought'
        mfi_desc = 'MFI超买(>80)，资金过度流入，警惕回调'
    elif mfi_now < 20:
        mfi_signal = 'oversold'
        mfi_desc = 'MFI超卖(<20)，资金过度流出，关注反弹'
    elif mfi_now > 50 and mfi[-2] <= 50:
        mfi_signal = 'cross_up'
        mfi_desc = 'MFI上穿50，资金开始流入'
    elif mfi_now < 50 and mfi[-2] >= 50:
        mfi_signal = 'cross_down'
        mfi_desc = 'MFI下穿50，资金开始流出'
    elif mfi_now > mfi[-2]:
        mfi_signal = 'rising'
        mfi_desc = 'MFI上升中，资金持续流入'
    else:
        mfi_signal = 'falling'
        mfi_desc = 'MFI下降中，资金持续流出'

    # 2. VWAP偏离度
    vwap = np.cumsum(typical * vol) / np.cumsum(vol)
    vwap_now = vwap[-1]
    vwap_deviation = (close[-1] - vwap_now) / vwap_now if vwap_now != 0 else 0

    if vwap_deviation > 0.05:
        vwap_signal = 'premium'
        vwap_desc = f'价格高于VWAP {vwap_deviation:.1%}，短期溢价'
    elif vwap_deviation < -0.05:
        vwap_signal = 'discount'
        vwap_desc = f'价格低于VWAP {abs(vwap_deviation):.1%}，短期折价'
    elif vwap_deviation > 0.02:
        vwap_signal = 'slight_premium'
        vwap_desc = '价格略高于VWAP，多头控盘'
    elif vwap_deviation < -0.02:
        vwap_signal = 'slight_discount'
        vwap_desc = '价格略低于VWAP，空头控盘'
    else:
        vwap_signal = 'fair'
        vwap_desc = '价格接近VWAP均衡位'

    # 3. A/D线 (Chaikin Accumulation/Distribution)
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    ad_line = np.cumsum(clv * vol)
    ad_ma10 = np.mean(ad_line[-10:])
    ad_ma20 = np.mean(ad_line[-20:])
    ad_trend = 'accumulating' if ad_ma10 > ad_ma20 else 'distributing'

    if ad_trend == 'accumulating':
        ad_desc = 'A/D线上升，主力收集筹码'
    else:
        ad_desc = 'A/D线下降，主力派发筹码'

    # A/D与价格背离
    ad_divergence = ''
    if len(close) >= 20:
        price_20d_up = close[-1] > close[-20]
        ad_20d_up = ad_line[-1] > ad_line[-20]
        if price_20d_up and not ad_20d_up:
            ad_divergence = '顶背离: 价格新高但A/D线未新高'
        elif not price_20d_up and ad_20d_up:
            ad_divergence = '底背离: 价格新低但A/D线未新低'

    # 综合主力信号
    bull_score = 0
    if mfi_signal in ['oversold', 'cross_up']: bull_score += 1
    if vwap_signal in ['discount', 'slight_discount']: bull_score += 1
    if ad_trend == 'accumulating': bull_score += 1
    if '底背离' in ad_divergence: bull_score += 1

    bear_score = 0
    if mfi_signal in ['overbought', 'cross_down']: bear_score += 1
    if vwap_signal in ['premium', 'slight_premium']: bear_score += 1
    if ad_trend == 'distributing': bear_score += 1
    if '顶背离' in ad_divergence: bear_score += 1

    if bull_score >= 3:
        smart_signal = 'strong_accumulation'
    elif bull_score >= 2:
        smart_signal = 'accumulation'
    elif bear_score >= 3:
        smart_signal = 'strong_distribution'
    elif bear_score >= 2:
        smart_signal = 'distribution'
    else:
        smart_signal = 'neutral'

    return {
        'sm_mfi': round(mfi_now, 2),
        'sm_mfi_signal': mfi_signal,
        'sm_mfi_desc': mfi_desc,
        'sm_vwap': round(vwap_now, 2),
        'sm_vwap_deviation': round(vwap_deviation, 4),
        'sm_vwap_signal': vwap_signal,
        'sm_vwap_desc': vwap_desc,
        'sm_ad_trend': ad_trend,
        'sm_ad_desc': ad_desc,
        'sm_ad_divergence': ad_divergence,
        'sm_smart_signal': smart_signal,
        'sm_bull_score': bull_score,
        'sm_bear_score': bear_score,
    }


def _empty_smart_money():
    return {
        'sm_mfi': 50, 'sm_mfi_signal': 'neutral', 'sm_mfi_desc': '数据不足',
        'sm_vwap': 0, 'sm_vwap_deviation': 0, 'sm_vwap_signal': 'neutral', 'sm_vwap_desc': '',
        'sm_ad_trend': 'neutral', 'sm_ad_desc': '', 'sm_ad_divergence': '',
        'sm_smart_signal': 'neutral', 'sm_bull_score': 0, 'sm_bear_score': 0,
    }


# ===================== 31. 格兰维尔八准则完整版 =====================

def granville_8_complete(df):
    """
    格兰维尔移动均线八准则（基于MA200/MA60）

    四大买入准则:
    1. MA走平转升+价格上穿MA → 突破买入
    2. MA上升+价格跌破MA后回升 → 回踩买入
    3. MA上升+价格跌破MA但快速收回 → 假跌破买入
    4. MA下降+价格远离MA后反弹 → 超跌反弹(逆势)

    四大卖出准则:
    5. MA走平转降+价格下穿MA → 破位卖出
    6. MA下降+价格升破MA后回落 → 反抽卖出
    7. MA下降+价格升破MA但快速回落 → 假突破卖出
    8. MA上升+价格远离MA后回落 → 超涨回调(逆势)
    """
    if len(df) < 200:
        return _empty_granville()

    close = df['close'].astype(float).values
    vol = df['volume'].astype(float).values

    ma60 = pd.Series(close).rolling(60).mean().values
    ma200 = pd.Series(close).rolling(200).mean().values

    cp = close[-1]
    ma60_now = ma60[-1]
    ma60_prev = ma60[-2]
    ma200_now = ma200[-1]

    # MA方向
    ma60_rising = ma60_now > ma60_prev
    ma60_slope = (ma60_now - ma60[-11]) / ma60[-11] if ma60[-11] != 0 else 0

    # 价格与MA关系
    above_ma60 = cp > ma60_now
    above_ma200 = cp > ma200_now
    price_ma60_pct = (cp - ma60_now) / ma60_now if ma60_now != 0 else 0
    price_ma200_pct = (cp - ma200_now) / ma200_now if ma200_now != 0 else 0

    # 偏离度 (标准差衡量)
    dev60 = price_ma60_pct / (np.std(close[-60:]) / ma60_now) if ma60_now != 0 else 0
    dev200 = price_ma200_pct / (np.std(close[-200:]) / ma200_now) if ma200_now != 0 else 0

    active_rule = None
    rule_category = None
    description = ''

    # 买入准则检测
    if ma60_rising:
        # 准则1: MA上升+价格突破MA
        if cp > ma60_now and close[-2] <= ma60[-2]:
            active_rule = 1
            rule_category = 'buy'
            description = '准则1(突破买入): MA60走平转升+收盘站上MA60'

        # 准则2: MA上升+价格回踩MA后回升
        elif cp > ma60_now:
            recent_low = np.min(close[-20:])
            if recent_low < ma60_now * 0.98 and cp > ma60_now * 1.01:
                active_rule = 2
                rule_category = 'buy'
                description = '准则2(回踩买入): MA60上升+价格回踩MA60后回升'

        # 准则3: MA上升+价格假跌破后快速收回
        elif cp > ma60_now and np.any(close[-10:-1] < ma60[-10:-1]):
            active_rule = 3
            rule_category = 'buy'
            description = '准则3(假跌破买入): MA60上升+价格跌破后3日内收回'

    if not ma60_rising:
        # 准则5: MA下降+价格下穿MA
        if cp < ma60_now and close[-2] >= ma60[-2]:
            active_rule = 5
            rule_category = 'sell'
            description = '准则5(破位卖出): MA60走平转降+收盘跌破MA60'

        # 准则6: MA下降+价格反抽MA后回落
        elif cp < ma60_now:
            recent_high = np.max(close[-20:])
            if recent_high > ma60_now * 1.02 and cp < ma60_now * 0.99:
                active_rule = 6
                rule_category = 'sell'
                description = '准则6(反抽卖出): MA60下降+价格反弹MA60后回落'

    # 准则4 (超跌反弹) 和 准则8 (超涨回调) 基于偏离度
    if abs(dev60) > 2.0:
        if cp < ma60_now and dev60 < -2.0:
            active_rule = 4
            rule_category = 'buy_contrarian'
            description = f'准则4(超跌反弹): 价格远离MA60({price_ma60_pct:.1%}),逆势反弹机会'
        elif cp > ma60_now and dev60 > 2.0:
            active_rule = 8
            rule_category = 'sell_contrarian'
            description = f'准则8(超涨回调): 价格远离MA60({price_ma60_pct:.1%}),逆势回调风险'

    # 准则7: MA下降+假突破后快速回落
    if not ma60_rising and not active_rule:
        if cp < ma60_now and np.any(close[-10:-1] > ma60[-10:-1]):
            active_rule = 7
            rule_category = 'sell'
            description = '准则7(假突破卖出): MA60下降+价格升破MA60后3日内回落'

    # 量价配合检查（准则1放量、准则5放量增强信号）
    vol_confirmed = False
    if active_rule in [1, 5]:
        vol_ma20 = np.mean(vol[-21:-1])
        vol_confirmed = vol[-1] > vol_ma20 * 1.2

    # 200日均线大背景
    ma200_trend = '上升' if ma200[-1] > ma200[-21] else '下降'
    long_term_context = f'MA200{ma200_trend}中'

    # 地量区域分析
    vol_ma20_val = np.mean(vol[-21:-1])
    land_volume_days = int(np.sum(vol[-60:] < vol_ma20_val * 0.5))
    land_zone = ''
    if land_volume_days >= 10:
        land_zone = '地量密集区(10+天地量)→底部区域'
    elif land_volume_days >= 5:
        land_zone = '地量出现(5-9天地量)→关注变盘'

    return {
        'g8_active_rule': active_rule,
        'g8_rule_category': rule_category or 'none',
        'g8_description': description or '无明确准则触发',
        'g8_ma60_rising': ma60_rising,
        'g8_ma60_slope': round(ma60_slope, 4),
        'g8_price_vs_ma60': round(price_ma60_pct, 4),
        'g8_price_vs_ma200': round(price_ma200_pct, 4),
        'g8_deviation_sigma': round(dev60, 2),
        'g8_vol_confirmed': vol_confirmed,
        'g8_long_term': long_term_context,
        'g8_ma200_above': above_ma200,
        'g8_land_volume_zone': land_zone,
        'g8_land_volume_days': land_volume_days,
    }


def _empty_granville():
    return {
        'g8_active_rule': None, 'g8_rule_category': 'none', 'g8_description': '数据不足(需200根K线)',
        'g8_ma60_rising': False, 'g8_ma60_slope': 0, 'g8_price_vs_ma60': 0, 'g8_price_vs_ma200': 0,
        'g8_deviation_sigma': 0, 'g8_vol_confirmed': False, 'g8_long_term': '数据不足',
        'g8_ma200_above': False, 'g8_land_volume_zone': '', 'g8_land_volume_days': 0,
    }


# ===================== 32. 贝叶斯核心:收缩估计与决策论 =====================

def james_stein_shrinkage(estimates, std_errors, prior_mean=None):
    """
    James-Stein收缩估计器 — 对多个资产的alpha/收益率进行收缩
    delta_JS = prior_mean + (1 - (p-2)*sigma^2 / ||x - prior_mean||^2)^+ * (x - prior_mean)

    参数:
        estimates: 各资产的原始估计值 (e.g., 样本alpha)
        std_errors: 各资产的标准误
        prior_mean: 先验均值（None则用cross-sectional均值）
    返回:
        shrunk: 收缩后的估计值
        shrinkage_factor: 收缩因子 (0=完全收缩, 1=不收缩)
    """
    x = np.asarray(estimates, dtype=float)
    s = np.asarray(std_errors, dtype=float)
    p = len(x)
    if p < 3:
        return x, np.zeros(p)
    if prior_mean is None:
        prior_mean = np.mean(x)
    sigma2 = np.mean(s ** 2)
    ssq = np.sum((x - prior_mean) ** 2)
    if ssq < 1e-15:
        return np.full(p, prior_mean), np.ones(p) * 0.01
    # Positive-part James-Stein
    shrinkage = max(0.0, 1.0 - (p - 2) * sigma2 / ssq)
    shrunk = prior_mean + shrinkage * (x - prior_mean)
    return shrunk, np.full(p, shrinkage)


def empirical_bayes_shrinkage(estimates, std_errors):
    """
    Empirical Bayes层次收缩（ML-II）
    theta_i = mu + (1 - sigma_i^2 / (sigma_i^2 + tau^2)) * (x_i - mu)
    tau^2 = max(0, s^2 - sigma_bar^2)  # 方法矩估计
    """
    x = np.asarray(estimates, dtype=float)
    s = np.asarray(std_errors, dtype=float)
    p = len(x)
    if p < 3:
        return x, np.zeros(p)
    mu_hat = np.average(x, weights=1.0 / (s ** 2 + 1e-10))
    s2 = np.var(x)
    sigma_bar2 = np.mean(s ** 2)
    tau2_hat = max(0.0, s2 - sigma_bar2)
    shrinkage_factors = s ** 2 / (s ** 2 + tau2_hat + 1e-10)
    shrunk = mu_hat + (1.0 - shrinkage_factors) * (x - mu_hat)
    return shrunk, 1.0 - shrinkage_factors


def hierarchical_bayesian_fusion(dimension_llrs, se_dict=None, n_dims=9):
    """
    层次贝叶斯LLR收缩（Gelman BDA3 Ch.5 — 可交换层次模型）

    每个维度的LLR视为从群体分布中抽取:
      LLR_i ~ N(mu_population, tau²_population)

    τ²由方法矩估计:τ² = max(0, Var(LLRs) - mean(se²))
    收缩因子 B_i = se²_i / (se²_i + τ²)
    收缩后的LLR'_i = mu + (1 - B_i) * (LLR_i - mu)

    效果:极端维度值被拉向群体均值，降低噪声维度的过度影响。
    """
    dims = list(dimension_llrs.keys())
    n = len(dims)
    if n < 3:
        return dimension_llrs, {}, 0.0

    llr_vals = np.array([dimension_llrs[d] for d in dims], dtype=float)

    if se_dict is None:
        se_dict = {d: 0.5 for d in dims}
    se_vals = np.array([se_dict.get(d, 0.5) for d in dims], dtype=float)

    mu_pop = np.mean(llr_vals)
    s2 = np.var(llr_vals)
    sigma_bar2 = np.mean(se_vals ** 2)
    tau2 = max(0.0, s2 - sigma_bar2)

    if tau2 < 1e-10:
        B = np.ones(n)
    else:
        B = se_vals ** 2 / (se_vals ** 2 + tau2)

    shrunk_llr_vals = mu_pop + (1.0 - B) * (llr_vals - mu_pop)
    shrunk_llrs = {d: float(shrunk_llr_vals[i]) for i, d in enumerate(dims)}

    most_shrunk = [dims[i] for i in np.argsort(-B)[:3] if B[i] > 0.3]

    return shrunk_llrs, {d: float(1.0 - B[i]) for i, d in enumerate(dims)}, float(mu_pop), most_shrunk


def stock_loss(true_return, predicted, alpha=100.0):
    """
    交易预测损失函数（Davidson-Pilon Bayesian Methods for Hackers Ch.5）
    惩罚错误方向远大于数量误差，产生稀疏预测
    """
    true_return = np.asarray(true_return)
    predicted = np.asarray(predicted)
    loss = np.where(
        true_return * predicted < 0,
        alpha * predicted ** 2 - np.sign(true_return) * predicted + np.abs(true_return),
        np.abs(true_return - predicted)
    )
    return np.mean(loss)


def asymmetric_trading_loss(theta_true, theta_pred, k_under=2.0, k_over=1.0):
    """
    非对称交易损失 — 低估/高估代价不同
    k_under > k_over: 错过机会比错误进场更贵
    k_over > k_under: 错误进场比错过机会更贵（保守）
    """
    diff = theta_true - theta_pred
    loss = np.where(diff > 0, k_under * diff, -k_over * diff)
    return np.mean(loss)


def compute_hdi(samples, cred_mass=0.95):
    """
    最高密度区间 (Highest Density Interval)
    返回 (lower, upper) — 在给定cred_mass下最短的区间
    """
    samples = np.sort(samples)
    n = len(samples)
    if n < 10:
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    n_in_interval = int(np.floor(cred_mass * n))
    if n_in_interval < 1: n_in_interval = 1
    widths = samples[n_in_interval:] - samples[:n - n_in_interval]
    idx = np.argmin(widths)
    return float(samples[idx]), float(samples[idx + n_in_interval])


def rope_decision(posterior_samples, rope=(-0.01, 0.01), cred_mass=0.95):
    """
    ROPE决策规则 (Kruschke DBDA)
    返回: 'reject_null' | 'accept_null' | 'undecided'
    - reject_null: HDI完全在ROPE之外 → 效应显著
    - accept_null: HDI完全在ROPE之内 → 效应可忽略
    - undecided: HDI与ROPE重叠 → 需要更多数据
    """
    hdi_low, hdi_high = compute_hdi(posterior_samples, cred_mass)
    rope_low, rope_high = rope
    if hdi_high < rope_low or hdi_low > rope_high:
        return 'reject_null', hdi_low, hdi_high
    elif hdi_low >= rope_low and hdi_high <= rope_high:
        return 'accept_null', hdi_low, hdi_high
    else:
        return 'undecided', hdi_low, hdi_high


def compute_waic(log_lik_matrix):
    """
    WAIC (Widely Applicable Information Criterion) — Gelman BDA Ch.7

    WAIC = -2 * (lppd - p_waic)
    where:
      lppd = sum_i log(1/S * sum_s p(y_i|theta_s))  [log pointwise predictive density]
      p_waic = sum_i Var_s(log p(y_i|theta_s))       [effective number of parameters]

    log_lik_matrix: shape (n_data_points, n_posterior_samples)
    返回: waic, lppd, p_waic
    """
    n, s = log_lik_matrix.shape
    # Log pointwise predictive density (log-sum-exp trick)
    lppd = 0.0
    for i in range(n):
        max_ll = np.max(log_lik_matrix[i, :])
        lppd += max_ll + np.log(np.mean(np.exp(log_lik_matrix[i, :] - max_ll)))
    # Effective number of parameters
    p_waic = 0.0
    for i in range(n):
        p_waic += np.var(log_lik_matrix[i, :])
    waic = -2.0 * (lppd - p_waic)
    return {'waic': waic, 'lppd': lppd, 'p_waic': p_waic}


def posterior_predictive_check(observed, posterior_predictive_samples, test_func, n_tests=1000):
    """
    后验预测检验 (Gelman BDA Ch.6)
    返回 Bayesian p-value: P(T(y_rep) >= T(y_obs) | model)
    """
    T_obs = test_func(observed)
    T_rep = np.array([test_func(posterior_predictive_samples[i, :]) for i in range(len(posterior_predictive_samples))])
    p_value = np.mean(T_rep >= T_obs)
    return {'p_value': p_value, 'T_obs': T_obs, 'T_rep_mean': np.mean(T_rep), 'T_rep_std': np.std(T_rep)}


# ===================== 33. MCMC收敛诊断与误差估计 =====================

def integrated_autocorr_time(x, max_lag=None):
    """
    积分自相关时间 (Newman & Barkema, Section 3.3)
    tau_int = 0.5 + sum_{t=1}^{M} chi(t)
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 500)
    var = np.var(x)
    if var < 1e-15:
        return 0.5
    chi = np.zeros(max_lag)
    for t in range(max_lag):
        chi[t] = np.mean(x[:n - t] * x[t:]) / var
    tau_int = 0.5
    for t in range(max_lag):
        tau_int += chi[t]
        if chi[t] < 0:
            break
    return max(tau_int, 0.5)


def effective_sample_size(samples, max_lag=None):
    """MCMC有效样本量 (ESS) — N_eff = N / (1 + 2*tau_int)"""
    samples = np.asarray(samples)
    n = len(samples)
    tau = integrated_autocorr_time(samples, max_lag)
    return n / (1.0 + 2.0 * tau)


def gr_diagnostic(chains):
    """
    Gelman-Rubin R-hat诊断 (多链收敛)
    chains: list of arrays, each array is one chain's samples
    返回 R-hat: < 1.1 表示收敛
    """
    K = len(chains)
    if K < 2:
        return 1.0
    n = min(len(c) for c in chains)
    chain_means = np.array([np.mean(c[-n:]) for c in chains])
    chain_vars = np.array([np.var(c[-n:], ddof=1) for c in chains])
    B = n * np.var(chain_means, ddof=1)  # between-chain variance
    W = np.mean(chain_vars)               # within-chain variance
    var_plus = (n - 1) / n * W + B / n
    if W < 1e-15:
        return 1.0
    R_hat = np.sqrt(var_plus / W)
    return round(float(R_hat), 4)


def bootstrap_error(samples, statistic_func, n_bootstrap=1000, thin_factor=None):
    """
    Bootstrap标准误估计 (Newman & Barkema Section 3.4.3)
    自动考虑相关性 — thin_factor用于间疏
    """
    samples = np.asarray(samples)
    if thin_factor is not None and thin_factor > 1:
        samples = samples[::thin_factor]
    n = len(samples)
    estimates = np.zeros(n_bootstrap)
    for k in range(n_bootstrap):
        idx = np.random.randint(0, n, n)
        estimates[k] = statistic_func(samples[idx])
    return {
        'mean': np.mean(estimates),
        'std': np.std(estimates, ddof=1),
        'ci_95': (float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))),
    }


def log_sum_exp(x):
    """Log-sum-exp数值稳定计算 — 避免大指数溢出"""
    x = np.asarray(x)
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))


def importance_reweight(samples, target_log_prob, proposal_log_prob):
    """
    重要性采样重加权 — 从提议分布采样，重加权到目标分布
    返回: 加权均值, 有效样本量比
    """
    log_weights = target_log_prob - proposal_log_prob
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)
    weights /= np.sum(weights)
    weighted_mean = np.sum(samples * weights)
    ess_ratio = 1.0 / np.sum(weights ** 2) / len(samples) if len(samples) > 0 else 0
    return weighted_mean, float(ess_ratio)


def parallel_tempering_swap(beta_i, beta_j, energy_i, energy_j):
    """
    并行回火交换接受概率 (Newman & Barkema Section 6.4)
    用于多温度链之间的状态交换
    """
    delta = (beta_i - beta_j) * (energy_i - energy_j)
    return min(1.0, np.exp(delta))


def bayesian_bandit_update(wins, trials, rate=1.0):
    """
    贝叶斯多臂老虎机更新 (Beta-Binomial conjugate)
    posterior: Beta(rate*alpha_prior + wins, rate*beta_prior + trials - wins)
    rate < 1: 适应市场变化; rate = 1: 标准更新
    """
    wins = np.asarray(wins, dtype=float)
    trials = np.asarray(trials, dtype=float)
    alpha_post = 1.0 + rate * wins
    beta_post = 1.0 + rate * (trials - wins)
    samples = np.random.beta(alpha_post, beta_post)
    return samples, alpha_post, beta_post


def sigma_event_check(value, mean, std, use_robust=False, mad=None):
    """
    Sigma事件检测 — 5+ sigma ≈ 模型错误而非罕见事件 (Kurt Ch.12)

    注意:金融收益率是厚尾分布（峰度>3），基于正态假设的sigma阈值会低估极端事件概率。
    当 use_robust=True 时，使用MAD(中位数绝对偏差)替代标准差，对厚尾更稳健。
    MAD = median(|x_i - median(x)|), 缩放因子 1.4826 使其与正态标准差一致。

    返回: z_score, is_anomaly (|z| > 5), anomaly_level
    """
    if use_robust and mad is not None and mad > 1e-10:
        z = abs((value - mean) / (mad * 1.4826))
    elif std < 1e-10:
        return 0.0, False, 'normal'
    else:
        z = abs((value - mean) / std)
    if z >= 5:
        level = 'critical_model_error'
    elif z >= 4:
        level = 'severe_anomaly'
    elif z >= 3:
        level = 'anomaly'
    elif z >= 2:
        level = 'unusual'
    else:
        level = 'normal'
    return float(z), z >= 5, level


# ============================================================================
# Section 34: 广通唐能通系统 (Tang Nengtong / Guangtong Indicators)
# Source: 广通新生300天, 唐能通
# Core: 价托/价压, 三金叉/三死叉, 老鸭头, 跑道厚度, 2T-1L法则, 三三过滤制
# ============================================================================

def tang_jiato_jiaya(high, low, close, ma5=None, ma10=None, ma20=None):
    """
    价托/价压检测 — 唐能通核心战法

    价托 (Golden Tripod / Bullish):
      5MA上穿10MA + 5MA上穿20MA + 10MA上穿20MA 三者汇合形成三角形支撑区
      自然形成的价托: 5穿10, 5穿20, 10穿20 依次发生（最强）
      交叉形成的价托: 三条均线在交叉点同时形成三角

    价压 (Death Tripod / Bearish):
      5MA下穿10MA + 5MA下穿20MA + 10MA下穿20MA 形成三角形压力区

    返回: jiato_present, jiaya_present, jiato_strength (0-100), jiaya_strength (0-100),
          jiato_position (0=无, 1=刚形成/苗头期, 2=发展中, 3=成熟),
          jiaya_position, triangle_area
    """
    n = len(close)
    if n < 30:
        return False, False, 0.0, 0.0, 0, 0, 0.0

    if ma5 is None:
        ma5 = np.convolve(close, np.ones(5)/5, mode='same')
    if ma10 is None:
        ma10 = np.convolve(close, np.ones(10)/10, mode='same')
    if ma20 is None:
        ma20 = np.convolve(close, np.ones(20)/20, mode='same')

    # 价托检测 — 寻找三角形形成区域
    jiato_present = False
    jiato_strength = 0.0
    jiato_position = 0

    # 检查最近60根K线内的价托形成
    for i in range(max(30, n-60), n-5):
        # 条件: 5>10>20 (多头排列) 且在最近形成交叉
        cross_5_10 = (ma5[i-1] <= ma10[i-1] and ma5[i] > ma10[i])
        cross_5_20 = (ma5[i-1] <= ma20[i-1] and ma5[i] > ma20[i])
        cross_10_20 = (ma10[i-1] <= ma20[i-1] and ma10[i] > ma20[i])

        # 三个交叉在20根K线窗口内
        window_crosses = sum([cross_5_10, cross_5_20, cross_10_20])
        if i > 5:
            cross_5_10_any = np.any((ma5[i-19:i] > ma10[i-19:i]) &
                                     (np.roll(ma5[i-19:i], 1) <= np.roll(ma10[i-19:i], 1)))
            cross_5_20_any = np.any((ma5[i-19:i] > ma20[i-19:i]) &
                                     (np.roll(ma5[i-19:i], 1) <= np.roll(ma20[i-19:i], 1)))
            cross_10_20_any = np.any((ma10[i-19:i] > ma20[i-19:i]) &
                                      (np.roll(ma10[i-19:i], 1) <= np.roll(ma20[i-19:i], 1)))
            total_crosses = sum([cross_5_10_any, cross_5_20_any, cross_10_20_any])
        else:
            total_crosses = 0

        if total_crosses >= 3:
            jiato_present = True
            # 三角形面积 = 均线间距离的离散度
            spread = np.std([ma5[i], ma10[i], ma20[i]])
            avg_price = np.mean([ma5[i], ma10[i], ma20[i]])
            triangle_area = spread / avg_price * 100 if avg_price > 0 else 0

            # 强弱评级
            if ma5[i] > ma10[i] > ma20[i] and ma5[i] > ma5[i-5]:
                jiato_strength = min(100, 40 + triangle_area * 20 + total_crosses * 10)
                jiato_position = 3  # 成熟多头
            elif ma5[i] > ma10[i]:
                jiato_strength = min(100, 25 + triangle_area * 15 + total_crosses * 8)
                jiato_position = 2  # 发展中
            else:
                jiato_strength = min(100, 15 + triangle_area * 10 + total_crosses * 5)
                jiato_position = 1  # 苗头期
            break

    # 价压检测
    jiaya_present = False
    jiaya_strength = 0.0
    jiaya_position = 0

    for i in range(max(30, n-60), n-5):
        cross_5_10_down = (ma5[i-1] >= ma10[i-1] and ma5[i] < ma10[i])
        cross_5_20_down = (ma5[i-1] >= ma20[i-1] and ma5[i] < ma20[i])
        cross_10_20_down = (ma10[i-1] >= ma20[i-1] and ma10[i] < ma20[i])

        if i > 5:
            cross_5_10_any = np.any((ma5[i-19:i] < ma10[i-19:i]) &
                                     (np.roll(ma5[i-19:i], 1) >= np.roll(ma10[i-19:i], 1)))
            cross_5_20_any = np.any((ma5[i-19:i] < ma20[i-19:i]) &
                                     (np.roll(ma5[i-19:i], 1) >= np.roll(ma20[i-19:i], 1)))
            cross_10_20_any = np.any((ma10[i-19:i] < ma20[i-19:i]) &
                                      (np.roll(ma10[i-19:i], 1) >= np.roll(ma20[i-19:i], 1)))
            total_crosses = sum([cross_5_10_any, cross_5_20_any, cross_10_20_any])
        else:
            total_crosses = 0

        if total_crosses >= 3:
            jiaya_present = True
            spread = np.std([ma5[i], ma10[i], ma20[i]])
            avg_price = np.mean([ma5[i], ma10[i], ma20[i]])
            triangle_area = spread / avg_price * 100 if avg_price > 0 else 0

            if ma5[i] < ma10[i] < ma20[i] and ma5[i] < ma5[i-5]:
                jiaya_strength = min(100, 40 + triangle_area * 20)
                jiaya_position = 3
            elif ma5[i] < ma10[i]:
                jiaya_strength = min(100, 25 + triangle_area * 15)
                jiaya_position = 2
            else:
                jiaya_strength = min(100, 15 + triangle_area * 10)
                jiaya_position = 1
            break

    return jiato_present, jiaya_present, jiato_strength, jiaya_strength, \
           jiato_position, jiaya_position, triangle_area if 'triangle_area' in dir() else 0.0


def tang_sanjincha_sansicha(close, volume, macd_diff=None, macd_dea=None,
                              ma5=None, ma10=None, ma20=None):
    """
    三金叉/三死叉检测 — 唐能通量价时空核心

    三金叉 (Triple Golden Cross):
      1. 5MA上穿10MA (均线金叉)
      2. 成交量5日均量线上穿10日均量线 (量能金叉)
      3. MACD金叉 (MACD上穿信号线 或 MACD柱转正)
      三者同时满足 = 最强买入信号

    三死叉 (Triple Death Cross):
      1. 5MA下穿10MA (均线死叉)
      2. 成交量5日均量线下穿10日均量线 (量能死叉)
      3. MACD死叉
      三者同时满足 = 最强卖出信号

    返回: golden_cross, death_cross, golden_score (0-100), death_score (0-100),
          cross_days_ago (正=金叉天数, 负=死叉天数), triple_status
    """
    n = len(close)
    if n < 30:
        return False, False, 0.0, 0.0, 0, 'insufficient_data'

    if ma5 is None:
        ma5 = np.convolve(close, np.ones(5)/5, mode='same')
    if ma10 is None:
        ma10 = np.convolve(close, np.ones(10)/10, mode='same')
    if ma20 is None:
        ma20 = np.convolve(close, np.ones(20)/20, mode='same')

    vol_ma5 = np.convolve(volume, np.ones(5)/5, mode='same')
    vol_ma10 = np.convolve(volume, np.ones(10)/10, mode='same')

    # MACD
    if macd_diff is None:
        ema12 = np.zeros(n)
        ema26 = np.zeros(n)
        ema12[0] = close[0]
        ema26[0] = close[0]
        for i in range(1, n):
            ema12[i] = ema12[i-1] * 11/13 + close[i] * 2/13
            ema26[i] = ema26[i-1] * 25/27 + close[i] * 2/27
        macd_diff = ema12 - ema26
        macd_dea = np.convolve(macd_diff, np.ones(9)/9, mode='same')

    # 检测最近的三金叉/三死叉
    golden_cross = False
    death_cross = False
    golden_score = 0.0
    death_score = 0.0
    cross_days_ago = 0
    triple_status = 'none'

    lookback = min(60, n-10)
    for offset in range(0, lookback):
        idx = n - 1 - offset
        if idx < 5:
            continue

        # 均线交叉检测
        ma_golden = (ma5[idx-1] <= ma10[idx-1] and ma5[idx] > ma10[idx])
        ma_death = (ma5[idx-1] >= ma10[idx-1] and ma5[idx] < ma10[idx])

        # 量能交叉检测
        vol_golden = (vol_ma5[idx-1] <= vol_ma10[idx-1] and vol_ma5[idx] > vol_ma10[idx])
        vol_death = (vol_ma5[idx-1] >= vol_ma10[idx-1] and vol_ma5[idx] < vol_ma10[idx])

        # MACD交叉检测
        macd_golden = (macd_diff[idx-1] <= macd_dea[idx-1] and macd_diff[idx] > macd_dea[idx])
        macd_death = (macd_diff[idx-1] >= macd_dea[idx-1] and macd_diff[idx] < macd_dea[idx])

        # 三金叉
        if ma_golden and vol_golden and macd_golden:
            golden_cross = True
            cross_days_ago = offset
            # 评分: 基于均线发散度和量比
            divergence = abs(ma5[idx] - ma10[idx]) / close[idx] * 100
            vol_ratio = vol_ma5[idx] / (vol_ma10[idx] + 1e-10)
            macd_strength = abs(macd_diff[idx] - macd_dea[idx]) / (abs(macd_dea[idx]) + 1e-10)
            golden_score = min(100, 30 + divergence * 10 + (vol_ratio - 1) * 50 + macd_strength * 20)
            triple_status = 'triple_golden'
            break

        # 三死叉
        if ma_death and vol_death and macd_death:
            death_cross = True
            cross_days_ago = -offset
            divergence = abs(ma5[idx] - ma10[idx]) / close[idx] * 100
            vol_ratio = vol_ma5[idx] / (vol_ma10[idx] + 1e-10)
            macd_strength = abs(macd_diff[idx] - macd_dea[idx]) / (abs(macd_dea[idx]) + 1e-10)
            death_score = min(100, 30 + divergence * 10 + abs(vol_ratio - 1) * 50 + macd_strength * 20)
            triple_status = 'triple_death'
            break

        # 部分金叉（两项满足）
        partial_score = sum([ma_golden, vol_golden, macd_golden])
        if partial_score >= 2 and not golden_cross:
            golden_score = max(golden_score, partial_score * 20)
            triple_status = 'partial_golden' if triple_status == 'none' else triple_status

    return golden_cross, death_cross, golden_score, death_score, cross_days_ago, triple_status


def tang_paodao_houdu(close, ma60=None, ma120=None):
    """
    跑道厚度 (Runway Thickness) — 唐能通关键指标

    跑道: 60MA和120MA之间的区域，形似飞机跑道
    跑道厚度 RT = (跑道表面 - 60MA) / 跑道表面

    跑道厚度等级:
      RT < 0.05: 跑道太薄，跑道可能被击穿 → 不参与
      RT 0.05-0.10: 跑道偏薄，谨慎参与
      RT 0.10-0.20: 跑道厚度适中，正常参与
      RT > 0.20: 跑道厚实，可重仓参与

    返回: runway_present, runway_thickness, runway_grade,
          runway_slope (60MA斜率), runway_surface_price
    """
    n = len(close)
    if n < 130:
        return False, 0.0, 'insufficient_data', 0.0, 0.0

    if ma60 is None:
        ma60 = np.convolve(close, np.ones(60)/60, mode='same')
    if ma120 is None:
        ma120 = np.convolve(close, np.ones(120)/120, mode='same')

    # 跑道条件: 60MA > 120MA (多头跑道)
    runway_present = ma60[-1] > ma120[-1]

    # 跑道厚度
    runway_surface = ma60[-1]  # 跑道表面 = 60MA
    runway_base = ma120[-1]    # 跑道底部 = 120MA

    if runway_surface > 0:
        runway_thickness = (runway_surface - runway_base) / runway_surface
    else:
        runway_thickness = 0.0

    # 跑道斜率 (5日变化率)
    if n >= 65:
        runway_slope = (ma60[-1] - ma60[-6]) / (abs(ma60[-6]) + 1e-10)
    else:
        runway_slope = 0.0

    # 跑道等级
    if runway_thickness >= 0.20:
        runway_grade = 'thick'       # 厚实，可重仓
    elif runway_thickness >= 0.10:
        runway_grade = 'moderate'    # 适中
    elif runway_thickness >= 0.05:
        runway_grade = 'thin'        # 偏薄
    elif runway_thickness > 0:
        runway_grade = 'too_thin'    # 太薄
    else:
        runway_grade = 'no_runway'   # 无跑道（空头）

    return runway_present, float(runway_thickness), runway_grade, \
           float(runway_slope), float(runway_surface)


def tang_2T_1L_target(entry_price, recent_high, recent_low, direction='long'):
    """
    【2T-1L】和【2MP】目标位公式 — 唐能通目标测算法则

    【2T-1L】法则 (Two Tops minus One Low):
      目标位 = 2 × 前高 - 前低
      适用于突破前高后的上升目标测算

    【2MP】法则 (Two Bottoms - One Peak for shorts):
      目标位 = 2 × 前低 - 前高
      适用于跌破前低后的下跌目标测算

    参数:
      entry_price: 入场价
      recent_high: 最近显著高点
      recent_low: 最近显著低点
      direction: 'long' or 'short'

    返回: target_price, target_pct (从入场到目标), stop_level, risk_reward_ratio
    """
    if direction == 'long':
        target_2t1l = 2 * recent_high - recent_low
        # 保守目标
        conservative_target = recent_high + (recent_high - recent_low) * 0.618
        stop_level = recent_low - (recent_high - recent_low) * 0.05
    else:
        target_2t1l = 2 * recent_low - recent_high
        conservative_target = recent_low - (recent_high - recent_low) * 0.618
        stop_level = recent_high + (recent_high - recent_low) * 0.05

    target_pct = (target_2t1l - entry_price) / entry_price if entry_price > 0 else 0
    risk = abs(entry_price - stop_level)
    reward = abs(target_2t1l - entry_price)
    risk_reward = reward / risk if risk > 0 else 0

    return float(target_2t1l), float(conservative_target), float(target_pct), \
           float(stop_level), float(risk_reward)


def tang_33_filter(close, high, low, breakout_price, direction='long', lookback=20):
    """
    三三过滤制 (3-3 Filter System) — 唐能通假突破过滤

    规则: 突破后需满足
      1. 3%过滤: 价格必须超过突破点3%以上（过滤小幅度假突破）
      2. 3天过滤: 价格必须连续3天站稳突破点上方（过滤瞬时假突破）
      3. 成交量确认: 突破当天成交量应大于前3日平均成交量的1.5倍

    T-line突破 (唐能通T线):
      T线 = 最近显著高点的水平线
      突破T线 + 三三过滤确认 = 有效突破

    返回: valid_breakout, filter_score (0-100), days_above, pct_above,
          volume_confirm, details
    """
    n = len(close)
    if n < lookback + 5:
        return False, 0.0, 0, 0.0, False, {}

    current_price = close[-1]

    if direction == 'long':
        pct_above = (current_price - breakout_price) / breakout_price * 100
    else:
        pct_above = (breakout_price - current_price) / breakout_price * 100

    # 3%过滤
    filter_3pct = pct_above >= 3.0

    # 3天过滤: 最近3天价格都在突破点上方
    if direction == 'long':
        days_above = sum(1 for i in range(1, 4) if close[-i] > breakout_price)
    else:
        days_above = sum(1 for i in range(1, 4) if close[-i] < breakout_price)
    filter_3days = days_above >= 3

    # 成交量确认 (最近1天 vs 前3天均值)
    if n >= 4:
        avg_vol_prev = np.mean([close[n-5], close[n-4], close[n-3]]) if n >= 5 else close[n-4]
        vol_confirm = close[-2] if n >= 2 else close[-1]  # Using close as vol proxy
        # Note: caller should pass actual volume
        filter_vol = True  # placeholder
    else:
        filter_vol = True

    # 综合评分
    filter_score = 0
    if filter_3pct:
        filter_score += 40
    if filter_3days:
        filter_score += 35
    if filter_vol:
        filter_score += 25

    # 部分满足
    if not filter_3pct and pct_above >= 1.5:
        filter_score += 20
    if not filter_3days and days_above >= 2:
        filter_score += 15

    valid_breakout = filter_3pct and filter_3days

    details = {
        'pct_above': float(pct_above),
        'days_above': days_above,
        'filter_3pct_pass': filter_3pct,
        'filter_3days_pass': filter_3days,
        'volume_confirm': filter_vol
    }

    return valid_breakout, float(filter_score), days_above, float(pct_above), \
           filter_vol, details


def tang_laoyatou(high, low, close, ma60=None, ma120=None):
    """
    老鸭头 (Old Duck Head) 形态检测 — 唐能通经典形态

    老鸭头是唐能通最著名的形态之一，形似鸭头:
      鸭颈: 股价沿5/10日线缓慢上升（鸭脖子）
      鸭头: 股价快速上升形成头部（鸭头）
      鸭嘴: 股价回调到60日线附近再回升（鸭嘴）

    五要素:
      1. 前期60MA上升（鸭颈支撑）
      2. 股价远离60MA形成高点（鸭头）
      3. 回调不破60MA（鸭嘴形成）
      4. 回调时缩量
      5. 再次放量突破前高（鸭嘴张开）

    返回: pattern_present, pattern_phase, pattern_score (0-100),
          neck_length, head_height, mouth_angle
    """
    n = len(close)
    if n < 150:
        return False, 'none', 0.0, 0, 0.0, 0.0

    if ma60 is None:
        ma60 = np.convolve(close, np.ones(60)/60, mode='same')
    if ma120 is None:
        ma120 = np.convolve(close, np.ones(120)/120, mode='same')

    pattern_present = False
    pattern_phase = 'none'
    pattern_score = 0.0

    # 寻找近期高点 (鸭头)
    recent_window = close[-80:-10]
    if len(recent_window) < 20:
        return False, 'none', 0.0, 0, 0.0, 0.0

    head_idx = np.argmax(recent_window) + (n - 80)
    head_price = close[head_idx]
    head_ma60 = ma60[head_idx]

    # 鸭头高度
    head_height = (head_price - head_ma60) / head_ma60 if head_ma60 > 0 else 0

    # 鸭颈: head之前20-60天，60MA上升
    neck_start = max(0, head_idx - 60)
    neck_end = max(0, head_idx - 10)
    if neck_end > neck_start:
        neck_ma60_slope = (ma60[neck_end] - ma60[neck_start]) / (abs(ma60[neck_start]) + 1e-10)
    else:
        neck_ma60_slope = 0

    # 鸭嘴: head之后回调到60MA附近
    mouth_start = head_idx
    mouth_window = close[mouth_start:]
    if len(mouth_window) > 10:
        # 寻找回调低点
        mouth_low_idx = np.argmin(mouth_window[:min(30, len(mouth_window))]) + mouth_start
        mouth_low = close[mouth_low_idx]
        mouth_distance = (mouth_low - ma60[mouth_low_idx]) / ma60[mouth_low_idx] \
            if ma60[mouth_low_idx] > 0 else 1.0

        # 鸭嘴条件: 回调到60MA的±5%范围内
        mouth_near_60ma = abs(mouth_distance) < 0.05
    else:
        mouth_low_idx = mouth_start
        mouth_near_60ma = False
        mouth_distance = 1.0

    # 当前价格回升 (鸭嘴张开)
    current_above_60ma = close[-1] > ma60[-1]
    price_recovering = close[-1] > close[-5] if n >= 5 else False

    # 评分
    score = 0.0
    if head_height > 0.10:  # 鸭头够高
        score += 20
    if neck_ma60_slope > 0:  # 鸭颈上升
        score += 20
    if mouth_near_60ma:  # 鸭嘴回调到位
        score += 30
    if current_above_60ma and price_recovering:  # 鸭嘴张开
        score += 30
        pattern_present = True
        pattern_phase = 'mouth_opening'
    elif mouth_near_60ma:
        pattern_phase = 'mouth_forming'
        pattern_present = True
    elif head_height > 0.10:
        pattern_phase = 'head_formed'

    pattern_score = score

    # 鸭颈长度 (head之前的上升天数)
    if neck_end > neck_start:
        neck_length = neck_end - neck_start
    else:
        neck_length = 0

    # 鸭嘴角度 (当前5MA斜率)
    if n >= 5:
        ma5 = np.convolve(close, np.ones(5)/5, mode='same')
        mouth_angle = (ma5[-1] - ma5[-5]) / (abs(ma5[-5]) + 1e-10) * 100
    else:
        mouth_angle = 0.0

    return pattern_present, pattern_phase, float(pattern_score), \
           neck_length, float(head_height), float(mouth_angle)


# ============================================================================
# Section 35: 利弗莫尔战法 (Livermore Battle Tactics)
# Source: 股票大作手操盘术, 以趋势交易为生
# Core: 关键点, 6点/3点自适应过滤器, 金字塔加仓, 危险信号, 50%利润提取
# ============================================================================

def livermore_pivotal_points(high, low, close, atr=None):
    """
    利弗莫尔关键点 (Pivotal Points) 检测

    利弗莫尔核心战法: 只在关键点交易
    关键点类型:
      1. 历史高点/低点 (自然阻力/支撑)
      2. 整数关口 (50/100/200的倍数)
      3. 前期密集成交区突破点
      4. 6点/3点自适应过滤器

    6点规则 (经典):
      价格从关键点延伸超过6个点时确认趋势有效
      适用于高价股 (>30)

    3点规则 (适应):
      适用于低价股或波动率较低的时期

    自适应: 使用ATR替代固定点数
      关键点确认 = 突破关键点 + 1 ATR

    返回: pivotal_points (list of dicts), nearest_pivotal,
          pivotal_breakout (bool), breakout_strength (0-100),
          livermore_signal ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')
    """
    n = len(close)
    if n < 60:
        return [], {}, False, 0.0, 'hold'

    if atr is None:
        atr = np.zeros(n)
        tr = np.maximum(high[-1] - low[-1],
                        np.maximum(abs(high[-1] - np.roll(close, 1)[-1]),
                                   abs(low[-1] - np.roll(close, 1)[-1])))
        atr[-1] = tr

    current_price = close[-1]
    current_atr = atr[-1] if hasattr(atr, '__getitem__') else atr

    # 寻找关键点
    pivotal_points = []

    # 1. 60日高点/低点
    high_60 = max(high[-60:])
    low_60 = min(low[-60:])
    pivotal_points.append({
        'type': '60d_high', 'price': float(high_60),
        'distance_pct': (high_60 - current_price) / current_price * 100
    })
    pivotal_points.append({
        'type': '60d_low', 'price': float(low_60),
        'distance_pct': (current_price - low_60) / current_price * 100
    })

    # 2. 整数关口
    for mult in [50, 100, 200]:
        round_up = np.ceil(current_price / mult) * mult
        round_down = np.floor(current_price / mult) * mult
        if abs(round_up - current_price) / current_price < 0.05:
            pivotal_points.append({
                'type': f'round_{mult}_up', 'price': float(round_up),
                'distance_pct': (round_up - current_price) / current_price * 100
            })
        if abs(round_down - current_price) / current_price < 0.05:
            pivotal_points.append({
                'type': f'round_{mult}_down', 'price': float(round_down),
                'distance_pct': (current_price - round_down) / current_price * 100
            })

    # 3. 前期密集成交区 (简化: 20日均线)
    ma20 = np.mean(close[-20:])
    pivotal_points.append({
        'type': 'ma20', 'price': float(ma20),
        'distance_pct': (current_price - ma20) / current_price * 100
    })

    # 找到最近的关键点
    nearest_pivotal = min(pivotal_points,
                         key=lambda x: abs(x.get('distance_pct', 0))) \
        if pivotal_points else {}

    # 关键点突破检测 (使用ATR自适应)
    pivotal_breakout = False
    breakout_strength = 0.0
    livermore_signal = 'hold'

    if current_atr > 0:
        # 检查是否突破关键点至少1 ATR
        for pp in pivotal_points:
            dist_pct = pp.get('distance_pct', 0)
            atr_pct = current_atr / current_price * 100

            if abs(dist_pct) < atr_pct * 2:  # 在2 ATR范围内
                pivotal_breakout = True
                penetration = abs(dist_pct) / atr_pct
                breakout_strength = max(breakout_strength, min(100, penetration * 50))

    # 生成利弗莫尔信号
    above_ma20 = current_price > ma20
    above_60high = current_price > high_60

    if above_60high and breakout_strength > 60:
        livermore_signal = 'strong_buy'
    elif above_ma20 and breakout_strength > 30:
        livermore_signal = 'buy'
    elif current_price < low_60 and breakout_strength > 60:
        livermore_signal = 'strong_sell'
    elif not above_ma20 and breakout_strength > 30:
        livermore_signal = 'sell'

    return pivotal_points, nearest_pivotal, pivotal_breakout, \
           float(breakout_strength), livermore_signal


def livermore_danger_signal(open_, high, low, close, atr=None):
    """
    利弗莫尔危险信号检测

    危险信号类型:
      1. 日内反转 >1 ATR (冲高回落/探底回升)
      2. 连续3日高低点下移 (下降趋势确认)
      3. 关键点突破后立即回撤 (假突破)
      4. 连续缩量上涨 (量价背离)
      5. 高位长上影线 (上影线 > 实体2倍)

    利弗莫尔原则:
      当危险信号出现 → 立即离场观望
      "当我看见危险信号时，我不会和它争论，我先躲开"

    返回: danger_present, danger_signals (list), danger_level (0-100),
          immediate_action ('exit', 'reduce', 'alert', 'none')
    """
    n = len(close)
    if n < 5:
        return False, [], 0.0, 'none'

    if atr is None:
        tr_current = max(high[-1] - low[-1],
                        abs(high[-1] - close[-2]) if n >= 2 else 0,
                        abs(low[-1] - close[-2]) if n >= 2 else 0)
        atr = np.array([tr_current])

    current_atr = atr[-1] if hasattr(atr, '__getitem__') else atr
    danger_signals = []
    danger_level = 0.0

    # 1. 日内反转检测
    if n >= 2:
        intraday_range = high[-1] - low[-1]
        body = abs(close[-1] - open_[-1])
        upper_shadow = high[-1] - max(close[-1], open_[-1])
        lower_shadow = min(close[-1], open_[-1]) - low[-1]

        # 冲高回落: 上影线 > 实体2倍
        if upper_shadow > body * 2 and body > 0:
            danger_signals.append({
                'type': 'upper_shadow_reversal',
                'severity': min(100, upper_shadow / (body + 1e-10) * 20),
                'desc': '高位长上影线，冲高回落'
            })
            danger_level += 25

        # 探底回升但在下降趋势中
        if lower_shadow > body * 2 and close[-1] < close[-5]:
            danger_signals.append({
                'type': 'lower_shadow_in_downtrend',
                'severity': min(100, lower_shadow / (body + 1e-10) * 15),
                'desc': '下跌趋势中探底回升，空头陷阱'
            })
            danger_level += 15

        # 日内反转 > 1 ATR
        if current_atr > 0:
            reversal_range = abs(close[-1] - open_[-1])
            if reversal_range > current_atr * 1.5:
                danger_signals.append({
                    'type': 'large_intraday_reversal',
                    'severity': min(100, reversal_range / current_atr * 30),
                    'desc': '日内大幅反转 > 1.5 ATR'
                })
                danger_level += 30

    # 2. 连续3日高低点下移
    if n >= 4:
        lower_highs = all(high[-i] < high[-i-1] for i in range(1, 4))
        lower_lows = all(low[-i] < low[-i-1] for i in range(1, 4))
        if lower_highs and lower_lows:
            danger_signals.append({
                'type': 'descending_price_pattern',
                'severity': 60,
                'desc': '连续3日高低点下移'
            })
            danger_level += 25

    # 3. 突破后回撤 (假突破)
    if n >= 10:
        high_10 = max(high[-10:-1])
        if close[-2] > high_10 > close[-1] and close[-1] < close[-2] * 0.97:
            danger_signals.append({
                'type': 'fake_breakout',
                'severity': 50,
                'desc': '突破前高后立即回撤3%以上'
            })
            danger_level += 20

    # 4. 高位长上影线但缩量
    if n >= 5:
        recent_high = max(close[-20:]) if n >= 20 else max(close)
        near_high = close[-1] > recent_high * 0.95
        if near_high and upper_shadow > body * 2:
            danger_signals.append({
                'type': 'high_level_shadow',
                'severity': 40,
                'desc': '高位出现长上影线'
            })
            danger_level += 20

    danger_present = danger_level >= 30

    # 立即行动建议
    if danger_level >= 70:
        immediate_action = 'exit'
    elif danger_level >= 40:
        immediate_action = 'reduce'
    elif danger_level >= 20:
        immediate_action = 'alert'
    else:
        immediate_action = 'none'

    return danger_present, danger_signals, float(min(100, danger_level)), immediate_action


# ============================================================================
# Section 36: 解缠论增强 (Enhanced Chan Theory)
# Source: 解缠论1.0/2.0/3.0 - 余井强 (Yu Jingqiang)
# Core: 力度比, 中枢重心偏移, 5条件拐点评分, 笔内背离
# ============================================================================

def chanlun_lidubi(high, low, close, stroke_strength_current=None,
                    stroke_strength_prior=None):
    """
    力度比 (Stroke Power Ratio) — 余井强解缠论核心指标

    力度比 = (当前笔的力度) / (前一笔的力度)
      力度 = 笔的涨跌幅 / 笔的K线数量

    背离阈值:
      力度比 > 1.382: 趋势加速（没有背离）
      力度比 0.618-1.382: 正常范围
      力度比 < 0.618: 顶背驰预警（上涨力度衰竭）
      力度比 < 0.382: 顶背驰确认（即将反转）

    返回: power_ratio, divergence_type, divergence_warning,
          current_power, prior_power, ratio_level
    """
    n = len(close)
    if n < 20:
        return 1.0, 'none', False, 0.0, 0.0, 'insufficient_data'

    if stroke_strength_current is None:
        # 最近10根K线作为一个笔段
        current_move = close[-1] - close[-10]
        current_range = max(high[-10:]) - min(low[-10:])
        current_power = abs(current_move) / (current_range + 1e-10) * 10
    else:
        current_power = stroke_strength_current

    if stroke_strength_prior is None:
        # 前10-20根K线作为前一笔段
        if n >= 20:
            prior_move = close[-10] - close[-20]
            prior_range = max(high[-20:-10]) - min(low[-20:-10])
            prior_power = abs(prior_move) / (prior_range + 1e-10) * 10
        else:
            prior_power = current_power
    else:
        prior_power = stroke_strength_prior

    if prior_power > 0:
        power_ratio = current_power / prior_power
    else:
        power_ratio = 1.0

    # 判断背离
    divergence_type = 'none'
    divergence_warning = False

    # 检查方向
    if close[-1] > close[-10]:  # 上涨笔
        if power_ratio < 0.618:
            divergence_type = 'top_divergence_warning'
            divergence_warning = True
        if power_ratio < 0.382:
            divergence_type = 'top_divergence_confirmed'
    elif close[-1] < close[-10]:  # 下跌笔
        if power_ratio > 1.618:  # 下跌力度衰减（反向）
            divergence_type = 'bottom_divergence_warning'
            divergence_warning = True
        if power_ratio > 2.618:
            divergence_type = 'bottom_divergence_confirmed'

    # 力度比等级
    if power_ratio < 0.382:
        ratio_level = 'severe_decay'
    elif power_ratio < 0.618:
        ratio_level = 'decay'
    elif power_ratio <= 1.382:
        ratio_level = 'normal'
    elif power_ratio <= 1.618:
        ratio_level = 'accelerating'
    else:
        ratio_level = 'strong_acceleration'

    return float(power_ratio), divergence_type, divergence_warning, \
           float(current_power), float(prior_power), ratio_level


def chanlun_zhongshu_cg_drift(close, zhongshu_highs=None, zhongshu_lows=None, lookback=50):
    """
    中枢重心偏移 (Zhongshu Center-of-Gravity Drift) — 余井强解缠论

    中枢重心 CG = (中枢高点 + 中枢低点) / 2
    重心连续上移 → 多头趋势加强
    重心连续下移 → 空头趋势加强
    重心走平 → 盘整

    三级别中枢:
      周线中枢 CG → 长期趋势方向
      日线中枢 CG → 中期趋势方向
      60分钟中枢 CG → 短期交易方向

    返回: cg_current, cg_prev, cg_drift, drift_direction,
          cg_trend_strength (0-100), consecutive_drifts
    """
    n = len(close)
    if n < lookback:
        return 0.0, 0.0, 0.0, 'flat', 0.0, 0

    if zhongshu_highs is None:
        # 简化: 用最近20根K线的高低点作为中枢
        zhongshu_high = max(high[-20:]) if 'high' in dir() else max(close[-20:])
        zhongshu_low = min(low[-20:]) if 'low' in dir() else min(close[-20:])
    else:
        zhongshu_high = zhongshu_highs[-1] if hasattr(zhongshu_highs, '__getitem__') else zhongshu_highs
        zhongshu_low = zhongshu_lows[-1] if hasattr(zhongshu_lows, '__getitem__') else zhongshu_lows

    # 计算多个中枢窗口的重心
    cg_series = []
    window_size = min(20, n // 3)
    for i in range(min(5, n // window_size - 1)):
        start_idx = n - (i + 1) * window_size
        end_idx = n - i * window_size
        if start_idx < 0:
            continue
        h = max(close[start_idx:end_idx])
        l = min(close[start_idx:end_idx])
        cg_series.append((h + l) / 2)

    if len(cg_series) < 2:
        return 0.0, 0.0, 0.0, 'flat', 0.0, 0

    cg_current = cg_series[0]
    cg_prev = cg_series[1]
    cg_drift = (cg_current - cg_prev) / (abs(cg_prev) + 1e-10) * 100

    # 漂移方向
    if cg_drift > 0.5:
        drift_direction = 'up'
    elif cg_drift < -0.5:
        drift_direction = 'down'
    else:
        drift_direction = 'flat'

    # 连续漂移计数
    consecutive_drifts = 0
    for i in range(len(cg_series) - 1):
        if cg_series[i] > cg_series[i + 1]:
            if consecutive_drifts >= 0:
                consecutive_drifts += 1
            else:
                break
        elif cg_series[i] < cg_series[i + 1]:
            if consecutive_drifts <= 0:
                consecutive_drifts -= 1
            else:
                break

    # 趋势强度
    cg_trend_strength = min(100, abs(consecutive_drifts) * 20 + abs(cg_drift) * 5)

    return float(cg_current), float(cg_prev), float(cg_drift), \
           drift_direction, float(cg_trend_strength), consecutive_drifts


def chanlun_turn_score(high, low, close, volume, ma5=None, ma10=None, ma60=None, macd_diff=None):
    """
    5条件拐点评分系统 (100分制) — 余井强解缠论3.0

    五个条件各20分:
      1. 力度背驰确认 (笔力度比<0.618或>1.618) — 20分
      2. 中枢支撑/压力位到位 (触及中枢边界) — 20分
      3. K线反转形态确认 (锤子线/吞没/星线等) — 20分
      4. 量能确认 (放量或缩量到位) — 20分
      5. 均线/MACD辅助确认 — 20分

    总分 >= 80: 高概率拐点
    总分 >= 60: 中等概率拐点
    总分 < 60: 低概率拐点，不操作

    返回: turn_score (0-100), turn_direction ('top', 'bottom', 'none'),
          condition_scores (dict), turn_confidence, trade_signal
    """
    n = len(close)
    if n < 30:
        return 0.0, 'none', {}, 'low', 'no_action'

    if ma5 is None:
        ma5 = np.convolve(close, np.ones(5)/5, mode='same')
    if ma10 is None:
        ma10 = np.convolve(close, np.ones(10)/10, mode='same')
    if ma60 is None:
        ma60 = np.convolve(close, np.ones(60)/60, mode='same')

    scores = {}
    total_score = 0.0

    # 条件1: 力度背驰 (20分)
    power_ratio, div_type, div_warn, _, _, _ = chanlun_lidubi(high, low, close)
    if div_warn:
        if 'confirmed' in div_type:
            scores['power_divergence'] = 20
        else:
            scores['power_divergence'] = 12
    else:
        scores['power_divergence'] = 0
    total_score += scores['power_divergence']

    # 条件2: 中枢位置 (20分)
    recent_high = max(high[-30:])
    recent_low = min(low[-30:])
    zhongshu_range = recent_high - recent_low
    if zhongshu_range > 0:
        position = (close[-1] - recent_low) / zhongshu_range
        # 在中枢边界附近 (0-20% 或 80-100%)
        if position < 0.2 or position > 0.8:
            scores['zhongshu_position'] = 20
        elif position < 0.35 or position > 0.65:
            scores['zhongshu_position'] = 10
        else:
            scores['zhongshu_position'] = 5
    else:
        scores['zhongshu_position'] = 5
    total_score += scores['zhongshu_position']

    # 条件3: K线反转形态 (20分)
    body = abs(close[-1] - open_[-1]) if 'open_' in dir() else abs(close[-1] - close[-2])
    upper_shadow = high[-1] - max(close[-1], open_[-1] if 'open_' in dir() else close[-2])
    lower_shadow = min(close[-1], open_[-1] if 'open_' in dir() else close[-2]) - low[-1]

    kline_score = 0
    # 锤子线
    if lower_shadow > body * 2 and upper_shadow < body * 0.3:
        kline_score = 20
    # 倒锤子
    elif upper_shadow > body * 2 and lower_shadow < body * 0.3:
        kline_score = 15
    # 十字星
    elif body < (high[-1] - low[-1]) * 0.1:
        kline_score = 10
    scores['kline_pattern'] = kline_score
    total_score += kline_score

    # 条件4: 量能确认 (20分)
    if 'volume' in dir() and len(volume) >= 10:
        vol_ratio = volume[-1] / (np.mean(volume[-10:]) + 1e-10)
        if 1.5 <= vol_ratio <= 3.0:  # 温和放量
            scores['volume_confirm'] = 20
        elif vol_ratio > 3.0:  # 巨量，可能是高潮
            scores['volume_confirm'] = 10
        elif vol_ratio < 0.5:  # 缩量到极致
            scores['volume_confirm'] = 15
        else:
            scores['volume_confirm'] = 5
    else:
        scores['volume_confirm'] = 5
    total_score += scores['volume_confirm']

    # 条件5: 均线/MACD (20分)
    ma_score = 0
    if ma5[-1] > ma10[-1] and ma5[-5] < ma10[-5]:  # 均线金叉
        ma_score += 10
    if abs(close[-1] - ma60[-1]) / ma60[-1] < 0.03:  # 接近60MA
        ma_score += 10
    scores['ma_macd'] = ma_score
    total_score += ma_score

    # 确定拐点方向
    if 'top' in div_type:
        turn_direction = 'top'
    elif 'bottom' in div_type:
        turn_direction = 'bottom'
    elif close[-1] > ma5[-1] and ma5[-1] > ma10[-1]:
        turn_direction = 'top_possible'
    elif close[-1] < ma5[-1] and ma5[-1] < ma10[-1]:
        turn_direction = 'bottom_possible'
    else:
        turn_direction = 'none'

    # 置信度
    if total_score >= 80:
        turn_confidence = 'high'
    elif total_score >= 60:
        turn_confidence = 'medium'
    else:
        turn_confidence = 'low'

    # 交易信号
    if turn_confidence == 'high' and 'bottom' in turn_direction:
        trade_signal = 'strong_buy'
    elif turn_confidence == 'medium' and 'bottom' in turn_direction:
        trade_signal = 'buy_watch'
    elif turn_confidence == 'high' and 'top' in turn_direction:
        trade_signal = 'strong_sell'
    elif turn_confidence == 'medium' and 'top' in turn_direction:
        trade_signal = 'sell_watch'
    else:
        trade_signal = 'no_action'

    return float(total_score), turn_direction, scores, turn_confidence, trade_signal


def chanlun_intra_stroke_divergence(high, low, close, lookback=30):
    """
    笔内背离 (Intra-Stroke Divergence) 检测 — 余井强解缠论3.0

    笔内背离: 在同一笔内部，价格创新高但MACD柱缩短
    这是比标准背驰更早期的信号

    检测逻辑:
      在最近一笔上涨中:
        - 价格创新高
        - 但MACD柱(或RSI/量能)在缩短
        → 笔内顶背离

      在最近一笔下跌中:
        - 价格创新低
        - 但MACD柱在缩短(变长)或RSI在上升
        → 笔内底背离

    返回: divergence_present, divergence_type, divergence_severity (0-100),
          price_trend, indicator_trend, bars_since_divergence
    """
    n = len(close)
    if n < lookback:
        return False, 'none', 0.0, 'flat', 'flat', 0

    # 识别最近一笔
    segment = close[-lookback:]
    segment_highs = high[-lookback:]
    segment_lows = low[-lookback:]

    # 简化MACD计算
    ema12 = np.zeros(lookback)
    ema26 = np.zeros(lookback)
    ema12[0] = segment[0]
    ema26[0] = segment[0]
    for i in range(1, lookback):
        ema12[i] = ema12[i-1] * 11/13 + segment[i] * 2/13
        ema26[i] = ema26[i-1] * 25/27 + segment[i] * 2/27
    macd_hist = (ema12 - ema26) - np.convolve(ema12 - ema26, np.ones(9)/9, mode='same')

    # 找笔的方向
    if close[-1] > close[-lookback]:
        stroke_direction = 'up'
        # 检查顶背离: 价格创新高但MACD柱下降
        peak_idx = np.argmax(segment_highs[-15:]) + (lookback - 15)
        if peak_idx < lookback - 3:
            price_trend = 'up' if segment[-1] > segment[peak_idx] else 'flat'
            macd_trend = 'down' if macd_hist[-1] < macd_hist[peak_idx] else 'up'

            if price_trend == 'up' and macd_trend == 'down':
                divergence_type = 'intra_stroke_top'
                severity = min(100, abs(macd_hist[-1] - macd_hist[peak_idx]) /
                              (abs(macd_hist[peak_idx]) + 1e-10) * 50)
                return True, divergence_type, float(severity), price_trend, macd_trend, 0
    else:
        stroke_direction = 'down'
        # 检查底背离: 价格创新低但MACD柱上升
        trough_idx = np.argmin(segment_lows[-15:]) + (lookback - 15)
        if trough_idx < lookback - 3:
            price_trend = 'down' if segment[-1] < segment[trough_idx] else 'flat'
            macd_trend = 'up' if macd_hist[-1] > macd_hist[trough_idx] else 'down'

            if price_trend == 'down' and macd_trend == 'up':
                divergence_type = 'intra_stroke_bottom'
                severity = min(100, abs(macd_hist[-1] - macd_hist[trough_idx]) /
                              (abs(macd_hist[trough_idx]) + 1e-10) * 50)
                return True, divergence_type, float(severity), price_trend, macd_trend, 0

    return False, 'none', 0.0, 'flat', 'flat', 0


# ============================================================================
# Section 37: 逃顶检测 (Top Escape Detection)
# Source: 逃顶十二招, CAN SLIM (William O'Neil), 华尔街操盘手日记
# Core: 12种顶部逃逸信号, 分配日计数, 顶部概率评分
# ============================================================================

def top_escape_signals(high, low, close, open_=None, volume=None, atr=None,
                        ma50=None, ma200=None):
    """
    逃顶十二招综合检测 — 多种顶部逃逸技术融合

    检测的12种顶部信号:
      1. 高位长上影线 (上影>实体2倍, 位置>80%振幅区间)
      2. 高位放量滞涨 (量增价平, 顶部出货)
      3. 高位十字星/墓碑线
      4. 三只乌鸦 (连续3根阴线, 每根开盘在前一根实体内)
      5. 黄昏之星 (阳线+小实体星+阴线吞没)
      6. MACD顶背离 (价格新高, MACD低)
      7. RSI顶背离 (价格新高, RSI低)
      8. 巨量长阴 (成交量>5日均量2倍, 阴线>2%)
      9. 跌破关键均线 (跌破50MA且50MA走平/下弯)
      10. 双顶/头肩顶形态 (简化检测)
      11. 布林带收窄后突破下轨
      12. 连续缩量反弹 (反弹无量, 顶部确认)

    返回: total_signals_count, active_signals (list), top_probability (0-100),
          top_grade, immediate_action
    """
    n = len(close)
    if n < 60:
        return 0, [], 0.0, 'insufficient_data', 'none'

    if open_ is None:
        open_ = np.roll(close, 1)

    if atr is None:
        tr = [max(high[i] - low[i],
                  abs(high[i] - close[i-1]) if i > 0 else 0,
                  abs(low[i] - close[i-1]) if i > 0 else 0) for i in range(n)]
        atr = np.array(tr)

    if ma50 is None:
        ma50 = np.convolve(close, np.ones(50)/50, mode='same')
    if ma200 is None and n >= 200:
        ma200 = np.convolve(close, np.ones(200)/200, mode='same')

    active_signals = []

    # 辅助变量
    body = close[-1] - open_[-1]
    upper_shadow = high[-1] - max(close[-1], open_[-1])
    lower_shadow = min(close[-1], open_[-1]) - low[-1]
    body_abs = abs(body)
    total_range = high[-1] - low[-1]

    # 信号1: 高位长上影线
    if body_abs > 0 and upper_shadow > body_abs * 2:
        # 检查是否在相对高位 (>80% 振幅区间)
        high_60 = max(high[-60:])
        low_60 = min(low[-60:])
        position = (close[-1] - low_60) / (high_60 - low_60 + 1e-10)
        if position > 0.75:
            active_signals.append({
                'id': 1, 'name': '高位长上影线',
                'severity': 20, 'action': 'reduce'
            })

    # 信号2: 高位放量滞涨
    if volume is not None and len(volume) >= 10:
        vol_ma5 = np.mean(volume[-5:])
        vol_ma10 = np.mean(volume[-10:])
        price_change = (close[-1] - close[-5]) / close[-5]
        if vol_ma5 > vol_ma10 * 1.3 and abs(price_change) < 0.02:
            active_signals.append({
                'id': 2, 'name': '高位放量滞涨',
                'severity': 25, 'action': 'reduce'
            })

    # 信号3: 高位十字星/墓碑线
    if total_range > 0 and body_abs < total_range * 0.1:
        position = (close[-1] - low_60) / (high_60 - low_60 + 1e-10) if 'high_60' in dir() else 0.5
        if position > 0.7:
            active_signals.append({
                'id': 3, 'name': '高位十字星',
                'severity': 15, 'action': 'alert'
            })

    # 信号4: 三只乌鸦
    if n >= 4:
        three_black = all(
            close[-i] < open_[-i] and  # 阴线
            open_[-i] < close[-(i+1)] and  # 开在前一根实体内
            close[-i] < close[-(i+1)]  # 收盘低于前一根
            for i in range(1, 4)
        )
        if three_black:
            active_signals.append({
                'id': 4, 'name': '三只乌鸦',
                'severity': 30, 'action': 'exit'
            })

    # 信号5: 黄昏之星
    if n >= 4:
        # 阳线 + 小实体星 + 阴线吞没
        candle1_bull = close[-3] > open_[-3]
        candle1_body = abs(close[-3] - open_[-3])
        candle2_small = abs(close[-2] - open_[-2]) < candle1_body * 0.3
        candle3_bear = close[-1] < open_[-1]
        candle3_engulf = close[-1] < (close[-3] + open_[-3]) / 2
        if candle1_bull and candle2_small and candle3_bear and candle3_engulf:
            active_signals.append({
                'id': 5, 'name': '黄昏之星',
                'severity': 25, 'action': 'exit'
            })

    # 信号6: MACD顶背离
    if n >= 60:
        ema12 = np.zeros(n)
        ema26 = np.zeros(n)
        ema12[0] = close[0]; ema26[0] = close[0]
        for i in range(1, n):
            ema12[i] = ema12[i-1] * 11/13 + close[i] * 2/13
            ema26[i] = ema26[i-1] * 25/27 + close[i] * 2/27
        macd_diff = ema12 - ema26
        macd_dea = np.convolve(macd_diff, np.ones(9)/9, mode='same')
        macd_hist = macd_diff - macd_dea

        recent_high_price = max(close[-20:])
        recent_high_idx = np.argmax(close[-20:]) + (n - 20)
        if recent_high_idx < n - 3 and close[-1] < recent_high_price * 0.98:
            if macd_hist[-1] < macd_hist[recent_high_idx]:
                active_signals.append({
                    'id': 6, 'name': 'MACD顶背离',
                    'severity': 20, 'action': 'exit'
                })

    # 信号7: RSI顶背离 (简化)
    if n >= 20:
        delta = np.diff(close)
        gain = np.maximum(delta, 0)
        loss = np.abs(np.minimum(delta, 0))
        avg_gain = np.mean(gain[-14:])
        avg_loss = np.mean(loss[-14:])
        rsi = 100 - 100 / (1 + avg_gain / (avg_loss + 1e-10))
        rsi_prev = 100 - 100 / (1 + np.mean(gain[-28:-14]) / (np.mean(loss[-28:-14]) + 1e-10))

        if close[-1] > close[-10] and rsi < rsi_prev and rsi > 60:
            active_signals.append({
                'id': 7, 'name': 'RSI顶背离',
                'severity': 15, 'action': 'reduce'
            })

    # 信号8: 巨量长阴
    if volume is not None and len(volume) >= 6:
        vol_ratio = volume[-1] / (np.mean(volume[-6:-1]) + 1e-10)
        if vol_ratio > 2.0 and body < 0 and abs(body) / close[-1] > 0.02:
            active_signals.append({
                'id': 8, 'name': '巨量长阴',
                'severity': 30, 'action': 'exit'
            })

    # 信号9: 跌破关键均线
    if ma50 is not None:
        ma50_slope = (ma50[-1] - ma50[-20]) / (abs(ma50[-20]) + 1e-10)
        if close[-1] < ma50[-1] and ma50_slope < 0.001:
            active_signals.append({
                'id': 9, 'name': '跌破50MA且均线走平',
                'severity': 20, 'action': 'reduce'
            })

    # 信号10: 双顶形态 (简化)
    if n >= 40:
        high_20 = max(high[-40:-20])
        high_current = max(high[-20:])
        if abs(high_current - high_20) / high_20 < 0.03 and close[-1] < high_current * 0.95:
            active_signals.append({
                'id': 10, 'name': '双顶形态',
                'severity': 25, 'action': 'exit'
            })

    # 信号11: 布林带收窄后向下突破
    if n >= 30:
        bb_std = np.std(close[-20:])
        bb_mid = np.mean(close[-20:])
        bb_width = bb_std * 4 / bb_mid
        bb_width_prev = np.std(close[-40:-20]) * 4 / np.mean(close[-40:-20])
        if bb_width < bb_width_prev * 0.7 and close[-1] < bb_mid - bb_std:
            active_signals.append({
                'id': 11, 'name': '布林带收窄后下破',
                'severity': 20, 'action': 'reduce'
            })

    # 信号12: 连续缩量反弹
    if volume is not None and len(volume) >= 20:
        recent_vol_trend = all(volume[-i] < volume[-i-1] for i in range(1, 4))
        recent_price_up = close[-1] > close[-5]
        if recent_vol_trend and recent_price_up:
            active_signals.append({
                'id': 12, 'name': '缩量反弹',
                'severity': 15, 'action': 'alert'
            })

    # 综合评分
    total_severity = sum(s['severity'] for s in active_signals)
    total_count = len(active_signals)
    top_probability = min(100, total_severity)

    # 等级
    if top_probability >= 70:
        top_grade = 'critical'
        immediate_action = 'exit_all'
    elif top_probability >= 45:
        top_grade = 'high_risk'
        immediate_action = 'exit_partial'
    elif top_probability >= 25:
        top_grade = 'elevated_risk'
        immediate_action = 'reduce'
    elif top_probability >= 10:
        top_grade = 'early_warning'
        immediate_action = 'alert'
    else:
        top_grade = 'safe'
        immediate_action = 'none'

    return total_count, active_signals, float(top_probability), top_grade, immediate_action


def distribution_days_count(close, volume, lookback=25):
    """
    分配日计数 (Distribution Days) — CAN SLIM 顶部检测

    分配日定义:
      当日收盘价低于前一日收盘价且成交量大于前一日成交量
      或者当日收盘价比前一日收盘价低0.2%以上且成交量放大

    规则 (O'Neil):
      在25个交易日内出现5个或以上分配日 → 市场顶部信号
      在25个交易日内出现3-4个分配日 → 警告信号
      出现分配日后, 如果指数创新高, 该分配日失效

    返回: dist_days_count, dist_day_indices, market_top_warning,
          dist_day_pct (分配日占比), staleness_count (失效分配日)
    """
    n = len(close)
    if n < lookback + 1:
        return 0, [], False, 0.0, 0

    start_idx = n - lookback - 1
    dist_days = []
    staleness_count = 0

    for i in range(start_idx + 1, n):
        price_down = close[i] < close[i-1]
        vol_up = volume[i] > volume[i-1]
        pct_change = (close[i] - close[i-1]) / close[i-1]

        is_dist_day = price_down and (vol_up or pct_change < -0.002)

        if is_dist_day:
            # 检查是否失效 (后续创新高)
            future_high = max(close[i:]) if i < n - 1 else close[i]
            if future_high > close[i] * 1.02:
                staleness_count += 1
                continue
            dist_days.append(i)

    dist_count = len(dist_days)
    dist_pct = dist_count / lookback * 100

    # 市场顶部警告
    if dist_count >= 5:
        market_top_warning = True
    elif dist_count >= 3:
        market_top_warning = True
    else:
        market_top_warning = False

    return dist_count, dist_days, market_top_warning, float(dist_pct), staleness_count


# ===================== 缠论K线重叠中枢（零师独家破解） =====================

def chanlun_kline_overlap_hub(high, low, close, timeframe='daily'):
    """
    缠论K线重叠中枢识别 — 零师独家破解法

    核心原理:用K线重叠数量来过滤级别，替代复杂的笔/线段划分。
    这种方法比传统笔-线段-中枢构建更简单、更贴近走势原貌。

    级别对应规则:
      日线图:
        2-3K重叠 = 1F中枢 (次次次级别)
        5-20K重叠 = 5F中枢 (次次级别)
        20+K重叠或标准笔中枢 = 30F中枢 (次级别)
      30F图:
        2-3K重叠 = 1F线段中枢
        5-20K重叠 = 1F中枢
        20+K重叠或笔 = 5F中枢
      5F图:
        2-3K重叠 = 次1F中枢
        5-20K重叠 = 1F线段中枢
        20+K重叠或笔 = 1F中枢

    返回:
      hubs: list of dicts {start_idx, end_idx, overlap_count, level, level_name,
                          high, low, midpoint, strength}
      current_level: 当前最相关的中枢级别
      trend_structure: 走势结构描述
    """
    n = len(high)
    if n < 3:
        return [], None, 'insufficient_data'

    # 时间框架参数配置
    tf_config = {
        'daily': {'name': '日线', 'level_1': (2, 3), 'level_2': (4, 20), 'level_3': (20, 999)},
        '30min': {'name': '30F', 'level_1': (2, 3), 'level_2': (4, 20), 'level_3': (20, 999)},
        '5min': {'name': '5F', 'level_1': (2, 3), 'level_2': (4, 20), 'level_3': (20, 999)},
    }
    level_names = {
        'daily': {1: '1F中枢', 2: '5F中枢', 3: '30F中枢'},
        '30min': {1: '线段中枢', 2: '1F中枢', 3: '5F中枢'},
        '5min': {1: '次1F中枢', 2: '线段中枢', 3: '1F中枢'},
    }
    cfg = tf_config.get(timeframe, tf_config['daily'])
    names = level_names.get(timeframe, level_names['daily'])

    # 识别K线重叠区间
    hubs = []
    i = 0
    while i < n - 2:
        # 找到至少3根K线重叠的起点
        overlap_high = min(high[i], high[i+1])
        overlap_low = max(low[i], low[i+1])
        if overlap_high <= overlap_low:
            i += 1
            continue

        j = i + 2
        while j < n:
            new_overlap_high = min(overlap_high, high[j])
            new_overlap_low = max(overlap_low, low[j])
            if new_overlap_high <= new_overlap_low:
                break
            overlap_high = new_overlap_high
            overlap_low = new_overlap_low
            j += 1

        overlap_count = j - i
        if overlap_count >= 2:
            if overlap_count <= cfg['level_1'][1]:
                level = 1
            elif overlap_count <= cfg['level_2'][1]:
                level = 2
            else:
                level = 3

            hub_high = max(high[i:j])
            hub_low = min(low[i:j])
            midpoint = (hub_high + hub_low) / 2
            strength = min(1.0, overlap_count / 30.0)

            # 判断是否为"上下上"或"下上下"结构
            mid_section = close[i:j]
            is_standard = False
            if len(mid_section) >= 3:
                direction_changes = sum(
                    1 for k in range(1, len(mid_section) - 1)
                    if (mid_section[k] - mid_section[k-1]) * (mid_section[k+1] - mid_section[k]) < 0
                )
                is_standard = direction_changes >= 1

            hubs.append({
                'start_idx': int(i), 'end_idx': int(j - 1),
                'overlap_count': overlap_count, 'level': level,
                'level_name': names[level],
                'high': float(hub_high), 'low': float(hub_low),
                'midpoint': float(midpoint), 'strength': float(strength),
                'is_standard_structure': is_standard
            })
        i = j
        continue

    # 确定当前级别和走势结构
    if hubs:
        current_level = max(h['level'] for h in hubs[-3:]) if len(hubs) >= 3 else hubs[-1]['level']
    else:
        current_level = 0

    # 走势结构描述
    if len(hubs) >= 2:
        last_two = hubs[-2:]
        if last_two[0]['high'] < last_two[1]['low']:
            trend_structure = 'trend_up'
        elif last_two[0]['low'] > last_two[1]['high']:
            trend_structure = 'trend_down'
        elif abs(last_two[0]['midpoint'] - last_two[1]['midpoint']) < 0.02 * close[-1]:
            trend_structure = 'extension'
        else:
            trend_structure = 'consolidation'
    elif len(hubs) == 1:
        trend_structure = 'single_hub'
    else:
        trend_structure = 'no_structure'

    return hubs, current_level, trend_structure


def chanlun_circle_method(high, low, close, timeframe='daily'):
    """
    缠论圈圈论 — 简化的中枢+走势类型连接法

    "立足中枢、立足次级别趋势背驰" — 不纠缠笔和线段
    大圈圈 = 大级别中枢，小圈圈 = 连接段的小级别中枢

    返回:
      big_circles: 大级别中枢列表
      small_circles: 连接段小中枢列表
      structure: 走势结构分析
      turning_points: 可能的转折点
    """
    n = len(close)
    if n < 20:
        return [], [], {'type': 'insufficient'}, []

    # 先用K线重叠法找出所有中枢
    hubs, current_level, _ = chanlun_kline_overlap_hub(high, low, close, timeframe)

    if len(hubs) < 2:
        return hubs, [], {'type': 'single_hub_insufficient'}, []

    # 分离大圈圈和小圈圈
    max_level = max(h['level'] for h in hubs)
    big_circles = [h for h in hubs if h['level'] >= max_level]
    small_circles = [h for h in hubs if h['level'] < max_level]

    # 走势结构分析
    structure = {'type': 'unknown', 'direction': 'neutral'}

    if len(big_circles) >= 2:
        bc = big_circles
        # 判断中枢新生、扩展、延伸
        if bc[-1]['low'] > bc[-2]['high']:
            structure = {'type': 'zhongshu_birth', 'direction': 'up'}
        elif bc[-1]['high'] < bc[-2]['low']:
            structure = {'type': 'zhongshu_birth', 'direction': 'down'}
        elif abs(bc[-1]['midpoint'] - bc[-2]['midpoint']) < 0.01 * close[-1]:
            structure = {'type': 'zhongshu_extension', 'direction': 'neutral'}
        else:
            structure = {'type': 'zhongshu_expansion', 'direction': 'neutral'}

    # 寻找可能的转折点
    turning_points = []
    for i in range(1, len(hubs)):
        prev_hub = hubs[i-1]
        curr_hub = hubs[i]
        # 大圈圈+小圈圈背驰 = 转折点
        if (prev_hub['level'] >= 2 and curr_hub['level'] <= 1 and
            curr_hub['overlap_count'] < prev_hub['overlap_count'] * 0.5):
            turning_points.append({
                'index': int(curr_hub['end_idx']),
                'type': 'potential_reversal',
                'confidence': 0.6 + 0.2 * (prev_hub['level'] - curr_hub['level'])
            })

    return big_circles, small_circles, structure, turning_points


# ===================== 江恩理论技术指标 =====================

def gann_three_day_chart(high, low):
    """
    江恩三天图 — 以时间决定趋势方向

    规则:
      - 连续三天出现较高的底位及高位 → 上升趋势，线移至第三天高点
      - 连续三天创新低 → 下降趋势，线下移至第三天低点
      - 下跌两天后第三天再创新高 → 垂直上移至当天高点

    返回:
      trend: 1=上升, -1=下降, 0=无明确趋势
      signal_days: 最近的信号日列表
      chart_points: [(day_idx, price, direction), ...]
    """
    n = len(high)
    if n < 3:
        return 0, [], []

    trend = 0
    chart_points = []
    signal_days = []

    i = 0
    while i < n - 2:
        # 检查连续三天高点
        if high[i] < high[i+1] < high[i+2] and low[i] < low[i+1] < low[i+2]:
            trend = 1
            chart_points.append((int(i+2), float(high[i+2]), 1))
            signal_days.append(int(i+2))
        # 检查连续三天低点
        elif low[i] > low[i+1] > low[i+2]:
            trend = -1
            chart_points.append((int(i+2), float(low[i+2]), -1))
            signal_days.append(int(i+2))
        # 下跌两天后第三天再创新高
        elif high[i] > high[i+1] and high[i+2] > high[i+1]:
            trend = 1
            chart_points.append((int(i+2), float(high[i+2]), 1))
            signal_days.append(int(i+2))
        i += 1

    return trend, signal_days[-5:] if signal_days else [], chart_points[-10:]


def gann_nine_point_swing(close, swing_points=9.0):
    """
    江恩九点平均波动图 — 以价位上落幅度决定趋势走向

    规则 (基于江恩对道琼斯工业平均指数1912-1949的统计):
      - 反弹低于9点 → 反弹乏力
      - 反弹超过9点 → 可能转势 (~88%成功率)
      - 反弹至10-20点 → 可能反弹至20点
      - 反弹超过20点 → 可能反弹至30-31点
      - 反弹超过30点的情况极罕见

    对于A股，9点对应约0.3-0.5%的波动（视股价而定）。
    参数swing_points可调整: 对于高价股用百分比代替。

    返回:
      swing_signal: 'weak_bounce', 'potential_reversal', 'strong_reversal', 'extreme_move'
      swing_value: 实际波动幅度
      swing_thresholds: 各阈值
      recent_swings: 最近几次摆动
    """
    n = len(close)
    if n < 5:
        return 'insufficient', 0.0, {}, []

    # 动态计算摆动阈值（基于近期价格）
    avg_price = sum(close[-20:]) / min(20, n)
    pct_threshold = swing_points / 100.0  # 默认0.09%

    thresholds = {
        'weak': avg_price * pct_threshold * 0.5,
        'potential_reversal': avg_price * pct_threshold,
        'strong_reversal': avg_price * pct_threshold * 2.2,
        'extreme': avg_price * pct_threshold * 3.5
    }

    # 检测最近的摆动
    recent_swings = []
    local_extreme = close[0]
    extreme_idx = 0

    for i in range(1, n):
        # 检测局部极值点
        if i >= 2:
            if close[i-1] > close[i-2] and close[i-1] > close[i]:
                local_extreme = close[i-1]
                extreme_idx = i - 1
            elif close[i-1] < close[i-2] and close[i-1] < close[i]:
                local_extreme = close[i-1]
                extreme_idx = i - 1

        swing = abs(close[i] - local_extreme)
        if swing > thresholds['weak']:
            direction = 'up' if close[i] > local_extreme else 'down'
            recent_swings.append({
                'from_idx': int(extreme_idx), 'to_idx': int(i),
                'swing': float(swing), 'direction': direction,
                'swing_pct': float(swing / local_extreme * 100)
            })

    # 判断信号
    if not recent_swings:
        return 'weak_bounce', 0.0, thresholds, []

    last_swing = recent_swings[-1]
    swing_val = last_swing['swing']
    swing_pct = last_swing['swing_pct']

    if swing_pct < pct_threshold * 50:
        signal = 'weak_bounce'
    elif swing_pct < pct_threshold * 100:
        signal = 'potential_reversal'
    elif swing_pct < pct_threshold * 220:
        signal = 'strong_reversal'
    else:
        signal = 'extreme_move'

    return signal, swing_val, thresholds, recent_swings[-5:]


def gann_retracement_levels(high_price, low_price, current_price=None):
    """
    江恩回调法则 — 关键百分比回撤位

    最重要的价位:50%、63%、75%、100%
    分别对应几何角度:45°、63°、90°

    规则:
      1. 价格通常在50%回调位反转
      2. 突破50% → 下一目标63%
      3. 突破63% → 下一目标75%
      4. 突破75% → 下一目标100%

    支持/阻力位也出现在: 3-5%, 10-12%, 20-25%, 33-37%, 45-50%,
                          62-67%, 72-78%, 85-87%

    返回:
      levels: {pct: price, ...}
      nearest_level: 当前价格最近的支撑/阻力位
      nearest_pct: 最近的百分比
      position_desc: 当前价格在各回撤位中的位置描述
    """
    price_range = high_price - low_price
    if price_range <= 0:
        return {}, None, 0.0, 'invalid'

    key_pcts = [0.0, 0.125, 0.25, 0.375, 0.50, 0.625, 0.75, 0.875, 1.0]
    gann_pcts = [0.03, 0.10, 0.20, 0.33, 0.50, 0.63, 0.75, 0.85, 0.87, 1.0]

    levels = {}
    for pct in set(key_pcts + gann_pcts):
        price = low_price + price_range * pct
        angle = 90.0 * pct
        levels[round(pct * 100, 1)] = {
            'price': round(float(price), 3),
            'angle': round(float(angle), 1),
            'is_key': pct in [0.50, 0.63, 0.75, 1.0]
        }

    result = {'range': float(price_range), 'levels': levels}

    if current_price:
        nearest_pct = None
        nearest_dist = float('inf')
        for pct_key, info in levels.items():
            dist = abs(current_price - info['price'])
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pct = pct_key
                nearest_price = info['price']

        result['nearest_pct'] = nearest_pct
        result['nearest_price'] = nearest_price

        # 位置描述
        if current_price <= levels[50.0]['price']:
            if current_price <= levels[25.0]['price']:
                pos = 'deep_retracement'
            else:
                pos = 'normal_retracement'
        elif current_price <= levels[75.0]['price']:
            pos = 'shallow_retracement'
        else:
            pos = 'near_full_recovery'
        result['position_desc'] = pos

    return result


def gann_time_cycle_detection(dates_or_indices, prices=None):
    """
    江恩时间周期检测

    江恩重要时间周期（从重要顶/底起计）:
      7-12天, 18-21天, 28-31天, 42-49天, 57-65天,
      85-92天, 112-120天, 150-157天, 175-185天

    短期趋势: 42-49天, 中期趋势: 85-92天, 中/长期趋势: 175-185天

    重要循环周期:
      短期: 1h,2h,4h,...18h,24h,3周,7周,13周,15周,3月,7月
      中期: 1年,2年,3年,5年,7年,10年,13年,15年
      长期: 20年,30年,45年,49年,60年,82/84年,90年,100年

    30年=360个月=360°圆周循环

    返回:
      upcoming_windows: 即将到来的时间窗口
      cycle_phase: 当前在周期中的位置
      fibonacci_days: 斐波那契时间目标
    """
    n = len(dates_or_indices)
    if n < 10:
        return [], 'insufficient', []

    # 关键周期日数
    gann_cycles = [7, 12, 18, 21, 28, 31, 42, 49, 57, 65,
                   85, 92, 112, 120, 150, 157, 175, 185]

    # 寻找重要顶底作为起点
    if prices is not None and len(prices) == n:
        # 找最重要的转折点
        turning_indices = []
        for i in range(5, n - 5):
            if (prices[i] > max(prices[i-5:i]) and prices[i] > max(prices[i+1:i+6])):
                turning_indices.append((i, 'top', prices[i]))
            if (prices[i] < min(prices[i-5:i]) and prices[i] < min(prices[i+1:i+6])):
                turning_indices.append((i, 'bottom', prices[i]))
    else:
        turning_indices = [(0, 'start', 0)]

    # 从最重要的转折点计算时间窗口
    upcoming_windows = []
    if turning_indices:
        major_turn = turning_indices[-1]
        base_idx = major_turn[0]

        for cycle in gann_cycles:
            target_idx = base_idx + cycle
            if target_idx < n + 30:  # 包含未来30个周期
                days_away = target_idx - n + 1
                if 0 <= days_away <= 60:  # 未来60天内的窗口
                    upcoming_windows.append({
                        'cycle_days': cycle,
                        'days_away': days_away,
                        'from_turn': major_turn[1],
                        'importance': 'high' if cycle in [42, 49, 85, 92, 175, 185] else 'medium'
                    })
    upcoming_windows.sort(key=lambda x: x['days_away'])

    # 斐波那契时间目标
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    fibonacci_days = []
    if turning_indices:
        base_idx = turning_indices[-1][0]
        for f in fib:
            target = base_idx + f
            if 0 <= target - n + 1 <= 90:
                fibonacci_days.append({
                    'fib_number': f,
                    'days_away': target - n + 1
                })

    # 判断当前周期位置
    all_cycles_sorted = sorted(gann_cycles)
    current_day = n
    if turning_indices:
        days_from_turn = n - turning_indices[-1][0]
        # 找到最近的周期
        for c in all_cycles_sorted:
            if c >= days_from_turn:
                cycle_phase = f'approaching_{c}d_cycle'
                break
        else:
            cycle_phase = f'beyond_{all_cycles_sorted[-1]}d'
    else:
        cycle_phase = 'unknown'

    return upcoming_windows[:8], cycle_phase, fibonacci_days[:5]


def gann_resonance_score(indicators_dict):
    """
    江恩共振理论 — 多指标共振检测

    江恩共振条件:
      1. 长/中/短期投资者在同一时间同向操作
      2. 长/中/短期时间周期交汇于同一时间点且方向相同
      3. 长/中/短期均线交汇于同一价位且方向相同
      4. K线/均线/成交量/KDJ/MACD/布林线等多种指标同时发出信号
      5. 金融/财政/经济政策方向一致
      6. 基本面和技术面方向一致
      7. 上市公司各层面方向一致

    共振条件满足越多 → 威力越大

    返回:
      resonance_score: 0-10 (越高越强)
      resonance_grade: 'none'/'weak'/'moderate'/'strong'/'extreme'
      aligned_signals: 对齐的信号列表
      direction: 共振方向 'bullish'/'bearish'/'neutral'
    """
    score = 0.0
    aligned_bullish = 0
    aligned_bearish = 0
    signal_details = []

    # 1. 均线共振 (长期/中期/短期)
    ma_signals = []
    for key in ['ma5', 'ma10', 'ma20', 'ma60', 'ma120', 'ma250']:
        if key in indicators_dict:
            ma_signals.append(indicators_dict[key])
    if len(ma_signals) >= 3:
        # 短期均线组 vs 长期均线组
        short_mas = ma_signals[:3]
        long_mas = ma_signals[3:]
        short_avg = sum(short_mas) / len(short_mas)
        long_avg = sum(long_mas) / len(long_mas)
        if short_avg > long_avg:
            aligned_bullish += 1
            signal_details.append('ma_alignment_bullish')
        else:
            aligned_bearish += 1
            signal_details.append('ma_alignment_bearish')

    # 2. MACD共振
    if 'macd_diff' in indicators_dict and 'macd_dea' in indicators_dict:
        if indicators_dict['macd_diff'] > 0 and indicators_dict['macd_dea'] > 0:
            aligned_bullish += 1
            signal_details.append('macd_bullish')
        elif indicators_dict['macd_diff'] < 0 and indicators_dict['macd_dea'] < 0:
            aligned_bearish += 1
            signal_details.append('macd_bearish')

    # 3. KDJ共振
    if 'kdj_k' in indicators_dict and 'kdj_d' in indicators_dict:
        k = indicators_dict['kdj_k']
        d = indicators_dict['kdj_d']
        if k > d and k > 50:
            aligned_bullish += 1
            signal_details.append('kdj_bullish')
        elif k < d and k < 50:
            aligned_bearish += 1
            signal_details.append('kdj_bearish')

    # 4. 成交量共振
    if 'volume_ratio' in indicators_dict:
        vol_ratio = indicators_dict['volume_ratio']
        if vol_ratio > 1.5 and aligned_bullish > aligned_bearish:
            aligned_bullish += 1
            signal_details.append('volume_confirmation')
        elif vol_ratio > 1.5 and aligned_bearish > aligned_bullish:
            aligned_bearish += 1
            signal_details.append('volume_confirmation')

    # 5. 布林带共振
    if 'boll_mid' in indicators_dict and 'close' in indicators_dict:
        close_p = indicators_dict.get('close', 0)
        if close_p > indicators_dict['boll_mid']:
            aligned_bullish += 0.5
        else:
            aligned_bearish += 0.5

    # 6. ADX趋势强度共振
    if 'adx' in indicators_dict:
        adx = indicators_dict['adx']
        if adx > 25:
            signal_details.append(f'adx_strong_{adx:.0f}')
            if aligned_bullish > aligned_bearish:
                aligned_bullish += 0.5
            else:
                aligned_bearish += 0.5
        else:
            signal_details.append('adx_weak')

    total_signals = aligned_bullish + aligned_bearish
    if total_signals > 0:
        if aligned_bullish > aligned_bearish:
            resonance_pct = aligned_bullish / max(total_signals, 1)
            direction = 'bullish'
        elif aligned_bearish > aligned_bullish:
            resonance_pct = aligned_bearish / max(total_signals, 1)
            direction = 'bearish'
        else:
            resonance_pct = 0.5
            direction = 'neutral'

        # 评分公式: 信号数量 × 一致性比例 × 2
        score = min(10.0, total_signals * resonance_pct * 2.0)
        score = round(score, 1)
    else:
        resonance_pct = 0.0
        direction = 'neutral'

    # 共振等级
    if score >= 8:
        grade = 'extreme'
    elif score >= 6:
        grade = 'strong'
    elif score >= 4:
        grade = 'moderate'
    elif score >= 2:
        grade = 'weak'
    else:
        grade = 'none'

    return score, grade, signal_details, direction


def gann_angle_line(start_price, start_date_idx, current_idx, angle_ratios=None):
    """
    江恩角度线 (Gann Fan)

    基本比率: 1:1 (45°线) — 一个单位时间对应一个单位价格
    其他重要角度: 1x2(63.75°), 2x1(26.25°), 1x4, 4x1等

    对于A股日线:
      1x1: 每天1点 (约0.1元)
      1x2: 每天2点
      2x1: 每2天1点

    返回:
      angles: {ratio: current_price_at_angle, ...}
      current_position: 当前价格相对各角度线的位置
    """
    if angle_ratios is None:
        angle_ratios = {
            '8x1': 0.125, '4x1': 0.25, '3x1': 0.333, '2x1': 0.5,
            '1x1': 1.0,
            '1x2': 2.0, '1x3': 3.0, '1x4': 4.0, '1x8': 8.0
        }

    time_diff = current_idx - start_date_idx
    if time_diff <= 0:
        return {}, 'invalid'

    # 默认价格单位: 每天0.1元 (A股常见)
    price_unit = 0.1

    angles = {}
    for name, ratio in angle_ratios.items():
        price_change = price_unit * ratio * time_diff
        angles[name] = round(float(start_price + price_change), 3)
        angles[f'{name}_dn'] = round(float(start_price - price_change), 3)

    return angles, time_diff


# ===================== 威科夫量价分析 (Wyckoff VPA) =====================

def wyckoff_effort_vs_result(open_p, high, low, close, volume, lookback=20):
    """
    威科夫努力vs结果法则 (Effort vs Result)

    核心原则:价格运动（结果）必须有成交量（努力）匹配。
    每一根K线只评估两件事之一:确认或异常。

    异常类型:
      - 大阳线+低量 = 熊市异常（诱多陷阱）
      - 小阳线+高量 = 熊市异常（努力大结果小，供给压制需求）
      - 小阴线+高量 = 牛市异常（努力大无法推动下跌，买方吸收）
      - 上升趋势+缩量 = 熊市警告（上涨乏力）
      - 下跌趋势+缩量 = 牛市信号（卖压枯竭）

    返回:
      regime: 'confirmation'/'anomaly_bearish'/'anomaly_bullish'/'neutral'
      anomaly_type: 具体异常类型
      effort_ratio: 努力/结果比率
      signal_strength: 信号强度 0-100
    """
    n = len(close)
    if n < lookback + 1:
        return 'neutral', 'insufficient', 0.0, 0.0

    # 计算平均成交量和近期价格变化
    avg_vol = sum(volume[-lookback:]) / lookback
    avg_range = sum(high[-lookback:] - low[-lookback:]) / lookback

    if avg_vol <= 0 or avg_range <= 0:
        return 'neutral', 'zero_data', 0.0, 0.0

    # 分析最近3根K线
    anomalies = []
    confirmations = []

    for i in range(n-3, n):
        body = close[i] - open_p[i]
        candle_range = high[i] - low[i]
        upper_wick = high[i] - max(open_p[i], close[i])
        lower_wick = min(open_p[i], close[i]) - low[i]

        if candle_range <= 0:
            continue

        vol_ratio = volume[i] / avg_vol
        body_ratio = abs(body) / candle_range if candle_range > 0 else 0
        range_ratio = candle_range / avg_range if avg_range > 0 else 1.0

        is_bullish = body > 0
        is_bearish = body < 0

        # 大实体K线 (body > 60% of range)
        is_high_body = body_ratio > 0.6
        # 小实体K线 (body < 30% of range)
        is_low_body = body_ratio < 0.3
        # 高量 (volume > 1.5x avg)
        is_high_vol = vol_ratio > 1.5
        # 低量 (volume < 0.6x avg)
        is_low_vol = vol_ratio < 0.6

        # 异常检测
        if is_high_body and is_low_vol:
            anomalies.append({
                'type': 'high_body_low_vol',
                'bias': 'bearish' if is_bullish else 'bullish',
                'detail': '诱多陷阱' if is_bullish else '诱空陷阱',
                'strength': abs(body_ratio - 0.6) * 50 + abs(0.6 - vol_ratio) * 30
            })
        elif is_low_body and is_high_vol:
            anomalies.append({
                'type': 'low_body_high_vol',
                'bias': 'bearish' if is_bullish else 'bullish',
                'detail': '努力大结果小，供给压制需求' if is_bullish else '买方吸收供给',
                'strength': abs(0.3 - body_ratio) * 40 + abs(vol_ratio - 1.5) * 30
            })
        else:
            confirmations.append({
                'type': 'confirmation',
                'bias': 'bullish' if is_bullish else 'bearish',
                'strength': min(100, vol_ratio * 30 + body_ratio * 40)
            })

    # 趋势级别确认/异常
    if n >= lookback:
        recent_close = close[-5:]
        recent_vol = volume[-5:]
        price_trend = recent_close[-1] - recent_close[0]
        vol_trend = sum(recent_vol[-2:]) - sum(recent_vol[:2])

        trend_signal = None
        if price_trend > 0 and vol_trend < 0:
            trend_signal = ('anomaly_bearish', '上升趋势缩量:上涨乏力')
        elif price_trend < 0 and vol_trend < 0:
            trend_signal = ('anomaly_bullish', '下跌趋势缩量:卖压枯竭')
        elif price_trend > 0 and vol_trend > 0:
            trend_signal = ('confirmation', '上升放量:真实涨势')
        elif price_trend < 0 and vol_trend > 0:
            trend_signal = ('confirmation', '下跌放量:真实跌势')

    # 综合判定
    if anomalies:
        last_anomaly = anomalies[-1]
        regime = f"anomaly_{last_anomaly['bias']}"
        anomaly_type = last_anomaly['detail']
        signal_strength = last_anomaly['strength']
    elif confirmations:
        regime = 'confirmation'
        anomaly_type = 'normal'
        signal_strength = confirmations[-1]['strength']
    else:
        regime = 'neutral'
        anomaly_type = 'unclear'
        signal_strength = 0.0

    effort_ratio = avg_vol / (avg_range * close[-1]) if close[-1] > 0 else 0

    return regime, anomaly_type, float(effort_ratio), float(signal_strength)


def wyckoff_climax_detection(open_p, high, low, close, volume, lookback=30):
    """
    威科夫买入/卖出高潮检测

    卖出高潮 (Selling Climax):
      - 2-3根连续的长下影线 (锤子线)
      - 收盘价接近开盘价（小实体）
      - 成交量极高 (极端量大)
      - 出现在下跌趋势末端

    买入高潮 (Buying Climax):
      - 2-3根连续的长上影线 (射击之星)
      - 收盘价接近开盘价（小实体）
      - 成交量极高
      - 出现在上涨趋势末端

    返回:
      climax_type: 'buying_climax'/'selling_climax'/'none'
      climax_strength: 0-100
      climax_candles: 高潮K线索引
      phase: 'accumulation'/'distribution'/'markup'/'markdown'/'none'
    """
    n = len(close)
    if n < lookback + 5:
        return 'none', 0.0, [], 'none'

    avg_vol = sum(volume[-lookback:]) / lookback
    if avg_vol <= 0:
        return 'none', 0.0, [], 'none'

    # 识别锤子线和射击之星
    hammers = []
    stars = []

    for i in range(max(0, n - lookback), n):
        body = close[i] - open_p[i]
        candle_range = high[i] - low[i]
        if candle_range <= 0:
            continue

        upper_wick = high[i] - max(open_p[i], close[i])
        lower_wick = min(open_p[i], close[i]) - low[i]
        vol_ratio = volume[i] / avg_vol

        # 锤子线:长下影线，小实体在顶部
        if lower_wick > candle_range * 0.6 and upper_wick < candle_range * 0.2:
            hammers.append({
                'idx': int(i),
                'lower_wick_pct': float(lower_wick / candle_range),
                'vol_ratio': float(vol_ratio),
                'close': float(close[i])
            })
        # 射击之星:长上影线，小实体在底部
        if upper_wick > candle_range * 0.6 and lower_wick < candle_range * 0.2:
            stars.append({
                'idx': int(i),
                'upper_wick_pct': float(upper_wick / candle_range),
                'vol_ratio': float(vol_ratio),
                'close': float(close[i])
            })

    # 检测连续高潮
    climax_type = 'none'
    climax_strength = 0.0
    climax_candles = []
    phase = 'none'

    # 卖出高潮检测:连续锤子线 + 极高成交量 + 下跌趋势末端
    if len(hammers) >= 2:
        recent_hammers = hammers[-3:]
        for i in range(len(recent_hammers) - 1):
            h1, h2 = recent_hammers[i], recent_hammers[i+1]
            # 连续或接近连续
            if 0 <= h2['idx'] - h1['idx'] <= 3:
                avg_wick = (h1['lower_wick_pct'] + h2['lower_wick_pct']) / 2
                avg_vr = (h1['vol_ratio'] + h2['vol_ratio']) / 2
                # 高量 + 长下影线
                if avg_vr > 1.3 and avg_wick > 0.55:
                    # 确认下跌趋势
                    recent_ret = (close[-1] - close[-lookback]) / close[-lookback]
                    if recent_ret < -0.03:
                        climax_type = 'selling_climax'
                        climax_strength = min(100, avg_wick * 60 + avg_vr * 30)
                        climax_candles = [h1['idx'], h2['idx']]
                        phase = 'accumulation'

    # 买入高潮检测:连续射击之星 + 极高成交量 + 上涨趋势末端
    if climax_type == 'none' and len(stars) >= 2:
        recent_stars = stars[-3:]
        for i in range(len(recent_stars) - 1):
            s1, s2 = recent_stars[i], recent_stars[i+1]
            if 0 <= s2['idx'] - s1['idx'] <= 3:
                avg_wick = (s1['upper_wick_pct'] + s2['upper_wick_pct']) / 2
                avg_vr = (s1['vol_ratio'] + s2['vol_ratio']) / 2
                if avg_vr > 1.3 and avg_wick > 0.55:
                    recent_ret = (close[-1] - close[-lookback]) / close[-lookback]
                    if recent_ret > 0.03:
                        climax_type = 'buying_climax'
                        climax_strength = min(100, avg_wick * 60 + avg_vr * 30)
                        climax_candles = [s1['idx'], s2['idx']]
                        phase = 'distribution'

    # 检查stopping volume (减速信号)
    if climax_type == 'none' and n >= 8:
        last_4_bodies = [abs(close[i] - open_p[i]) for i in range(n-4, n)]
        last_4_vols = [volume[i] for i in range(n-4, n)]
        bodies_shrinking = all(last_4_bodies[i] >= last_4_bodies[i+1] * 0.95 for i in range(3))
        vols_rising = all(last_4_vols[i] <= last_4_vols[i+1] * 1.05 for i in range(3))
        if bodies_shrinking and vols_rising:
            recent_ret = (close[-1] - close[-lookback]) / close[-lookback]
            if recent_ret < -0.02:
                climax_type = 'stopping_volume_down'
                climax_strength = 55.0
                phase = 'pre_accumulation'
            elif recent_ret > 0.02:
                climax_type = 'stopping_volume_up'
                climax_strength = 55.0
                phase = 'pre_distribution'

    return climax_type, climax_strength, climax_candles, phase


def wyckoff_supply_demand_test(high, low, close, volume, lookback=30):
    """
    威科夫供给/需求测试

    供给测试 (Supply Test / 二次测试):
      - 价格上涨后回落至前期吸筹区
      - 回落时成交量极低（成功:无剩余卖盘 → 上涨在即）
      - 回落时成交量高（失败:仍有卖盘 → 继续吸筹）

    需求测试 (Demand Test):
      - 价格下跌后反弹至前期派发区
      - 反弹时成交量极低（成功:无剩余买盘 → 下跌在即）
      - 反弹时成交量高（失败:仍有买盘 → 继续派发）

    返回:
      test_type: 'supply_test'/'demand_test'/'none'
      test_result: 'passed'/'failed'/'none'
      test_confidence: 0-100
      zone_high, zone_low: 测试区域
    """
    n = len(close)
    if n < lookback + 10:
        return 'none', 'none', 0.0, None, None

    avg_vol = sum(volume[-lookback:]) / lookback
    if avg_vol <= 0:
        return 'none', 'none', 0.0, None, None

    # 寻找最近的交易区间（中枢/盘整区）
    recent_high = max(high[-lookback:-5])
    recent_low = min(low[-lookback:-5])
    zone_range = recent_high - recent_low

    if zone_range <= 0 or zone_range < 0.01 * close[-1]:
        return 'none', 'none', 0.0, None, None

    zone_mid = (recent_high + recent_low) / 2

    # 判断当前价格相对于区间的位子
    current_price = close[-1]
    position_in_zone = (current_price - recent_low) / zone_range

    test_type = 'none'
    test_result = 'none'
    test_confidence = 0.0

    # 供给测试:价格从上方回落进入区间，成交量萎缩
    if 0.3 < position_in_zone < 0.7:
        if close[-1] < close[-3] < close[-5]:
            # 下降进入区间
            recent_vol_ratio = volume[-1] / avg_vol
            if recent_vol_ratio < 0.7:
                test_type = 'supply_test'
                test_result = 'passed'
                test_confidence = min(100, (1.0 - recent_vol_ratio) * 80 + 20)
            elif recent_vol_ratio > 1.3:
                test_type = 'supply_test'
                test_result = 'failed'
                test_confidence = min(100, recent_vol_ratio * 30)

    # 需求测试:价格从下方反弹进入区间，成交量萎缩
    if test_type == 'none' and 0.3 < position_in_zone < 0.7:
        if close[-1] > close[-3] > close[-5]:
            recent_vol_ratio = volume[-1] / avg_vol
            if recent_vol_ratio < 0.7:
                test_type = 'demand_test'
                test_result = 'passed'
                test_confidence = min(100, (1.0 - recent_vol_ratio) * 80 + 20)
            elif recent_vol_ratio > 1.3:
                test_type = 'demand_test'
                test_result = 'failed'
                test_confidence = min(100, recent_vol_ratio * 30)

    return test_type, test_result, test_confidence, float(recent_high), float(recent_low)


def wyckoff_phase_detection(close, high, low, volume, lookback=60):
    """
    威科夫市场周期阶段检测

    四大阶段:
      Accumulation (吸筹): 区间震荡，下跌放量（主力买入），反弹缩量
      Markup (拉升): 上升趋势，正常成交量，缩量回调
      Distribution (派发): 区间震荡，上涨放量（主力卖出），下跌缩量
      Markdown (下跌): 下降趋势，放量下跌

    返回:
      phase: 'accumulation'/'markup'/'distribution'/'markdown'/'transition'
      phase_confidence: 0-100
      phase_details: 阶段细节
      cycle_position: 0-100 (在周期中的位置%)
    """
    n = len(close)
    if n < lookback + 10:
        return 'transition', 0.0, {}, 0.0

    half = lookback // 2
    first_half = close[n-lookback:n-half]
    second_half = close[n-half:n]
    recent_vol = volume[-half:]

    price_change = (close[-1] - close[-lookback]) / close[-lookback]
    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)
    vol_avg = sum(recent_vol) / len(recent_vol)
    vol_first = sum(volume[-lookback:-half]) / max(half, 1)
    vol_second = sum(volume[-half:]) / len(recent_vol)
    vol_trend = (vol_second - vol_first) / vol_first if vol_first > 0 else 0

    # 计算价格波动范围
    range_high = max(high[-lookback:])
    range_low = min(low[-lookback:])
    total_range = range_high - range_low
    if total_range <= 0:
        return 'transition', 0.0, {}, 0.0

    current_position = (close[-1] - range_low) / total_range

    # 阶段判定
    phase = 'transition'
    confidence = 0.0

    if abs(price_change) < 0.05:
        # 窄幅震荡 → 吸筹或派发
        if second_avg < first_avg and vol_trend > 0.1:
            phase = 'accumulation'
            confidence = 40 + abs(price_change) * 200
        elif second_avg > first_avg and vol_trend < -0.1:
            phase = 'distribution'
            confidence = 40 + abs(price_change) * 200
        else:
            # 检查成交量模式
            down_vol = sum(volume[i] for i in range(n-lookback, n) if close[i] < close[i-1])
            up_vol = sum(volume[i] for i in range(n-lookback, n) if close[i] > close[i-1])
            total_directional_vol = down_vol + up_vol
            if total_directional_vol > 0:
                if down_vol > up_vol * 1.3:
                    phase = 'accumulation'
                    confidence = 35
                elif up_vol > down_vol * 1.3:
                    phase = 'distribution'
                    confidence = 35
    elif price_change > 0.05:
        phase = 'markup'
        confidence = 40 + price_change * 200
    elif price_change < -0.05:
        phase = 'markdown'
        confidence = 40 + abs(price_change) * 200

    confidence = min(100, confidence)

    # 周期位置
    cycle_position = current_position * 100 if phase in ('accumulation', 'distribution') else (50 + price_change * 200)

    details = {
        'price_change_pct': round(float(price_change * 100), 2),
        'vol_trend': round(float(vol_trend * 100), 1),
        'range_width_pct': round(float(total_range / close[-1] * 100), 2),
        'position_in_range': round(float(current_position * 100), 1)
    }

    return phase, confidence, details, float(cycle_position)


def chanlun_wolf_defense_filter(macd_diff, dea=None, timeframe='60min'):
    """
    缠论防狼术 — MACD 0轴过滤器

    简单规则:在选定最小时间框架上，MACD黄白线在0轴下方时，
    远离所有市场和股票。只有当MACD黄白线稳定站上0轴后才重新进入。

    防狼术的量化版本:
      - wolf_state: 'safe' (MACD>0) / 'danger' (MACD<0) / 'warning' (接近0轴)
      - 用于过滤买入信号:wolf_state='danger'时禁止买入

    返回:
      wolf_state: 'safe'/'danger'/'warning'/'recovering'/'weakening'
      macd_position: MACD相对于0轴的位子
      days_since_cross: 距离上次穿越0轴的天数
      filter_active: 是否应该激活过滤器
    """
    n = len(macd_diff)
    if n < 5:
        return 'warning', 0.0, 0, False

    current_macd = macd_diff[-1]
    prev_macd = macd_diff[-5]

    # 基本状态
    if current_macd > 0:
        if prev_macd <= 0:
            wolf_state = 'recovering'
        else:
            wolf_state = 'safe'
    elif current_macd < 0:
        if prev_macd >= 0:
            wolf_state = 'weakening'
        else:
            wolf_state = 'danger'
    else:
        wolf_state = 'warning'

    # DEA确认
    if dea is not None and len(dea) >= 5:
        current_dea = dea[-1]
        if wolf_state == 'safe' and current_dea <= 0:
            wolf_state = 'warning'
        if wolf_state == 'danger' and current_dea >= 0:
            wolf_state = 'warning'

    # 计算穿越天数
    days_since_cross = 0
    for i in range(n - 2, max(0, n - 120), -1):
        if (macd_diff[i] > 0 and macd_diff[i+1] <= 0) or (macd_diff[i] <= 0 and macd_diff[i+1] > 0):
            days_since_cross = n - 1 - i
            break

    macd_position = float(current_macd)
    filter_active = wolf_state in ('danger', 'weakening')

    return wolf_state, macd_position, days_since_cross, filter_active


def chanlun_consolidation_divergence(df_or_high_low_close, bi_points=None, macd_hist=None):
    """
    缠论盘整背驰 (Consolidation Divergence)

    与趋势背驰完全不同的概念:
      趋势背驰:比较MACD黄白线+柱面积，必须有0轴回拉，保证回到最后一个中枢
      盘整背驰:只比较MACD柱面积（不看黄白线），不保证回到中枢区间

    盘整背驰三种情况:
      1. 不破中枢（必然回拉）
      2. 破中枢（必须先走，再评估）
      3. 形成三买/三卖

    返回:
      div_type: 'consolidation_bull'/'consolidation_bear'/'trend_bull'/'trend_bear'/'none'
      confidence: 0-100
      area_ratio: 当前段面积/前段面积
      guaranteed_pullback: 是否保证回拉
    """
    # 兼容DataFrame和原始数组输入
    if hasattr(df_or_high_low_close, 'columns'):
        close = df_or_high_low_close['close'].values
    elif isinstance(df_or_high_low_close, (list, tuple)) and len(df_or_high_low_close) >= 3:
        close = df_or_high_low_close[2]
    else:
        close = df_or_high_low_close

    n = len(close)
    if n < 20:
        return 'none', 0.0, 1.0, False

    # 使用简单MACD柱面积比较（如果未提供macd_hist，计算一个简化版本）
    if macd_hist is None or len(macd_hist) < n:
        ema12 = close * 1.0
        ema26 = close * 1.0
        alpha12 = 2.0 / 13.0
        alpha26 = 2.0 / 27.0
        for i in range(1, n):
            ema12[i] = ema12[i-1] * (1 - alpha12) + close[i] * alpha12
            ema26[i] = ema26[i-1] * (1 - alpha26) + close[i] * alpha26
        diff = ema12 - ema26
        dea_alpha = 2.0 / 10.0
        dea = diff * 1.0
        for i in range(1, n):
            dea[i] = dea[i-1] * (1 - dea_alpha) + diff[i] * dea_alpha
        macd_hist = 2 * (diff - dea)

    # 寻找盘整背驰段
    # 将最近的价格运动分为段
    segments = []
    segment_start = max(0, n - 60)
    direction = 0
    seg_high = close[segment_start]
    seg_low = close[segment_start]

    for i in range(segment_start + 1, n):
        if direction == 0:
            direction = 1 if close[i] > close[i-1] else -1
            seg_high = max(close[segment_start], close[i])
            seg_low = min(close[segment_start], close[i])
            continue

        if direction == 1:
            if close[i] < close[i-1] * 0.995:
                segments.append({
                    'start': int(segment_start), 'end': int(i-1),
                    'direction': 'up',
                    'high': float(seg_high), 'low': float(seg_low),
                    'area': float(sum(abs(macd_hist[segment_start:i])))
                })
                segment_start = i
                direction = -1
                seg_high = close[i]
                seg_low = close[i]
            else:
                seg_high = max(seg_high, close[i])
        else:
            if close[i] > close[i-1] * 1.005:
                segments.append({
                    'start': int(segment_start), 'end': int(i-1),
                    'direction': 'down',
                    'high': float(seg_high), 'low': float(seg_low),
                    'area': float(sum(abs(macd_hist[segment_start:i])))
                })
                segment_start = i
                direction = 1
                seg_high = close[i]
                seg_low = close[i]
            else:
                seg_low = min(seg_low, close[i])

    if len(segments) < 3:
        return 'none', 0.0, 1.0, False

    # 比较相邻同向段
    last_seg = segments[-1]
    for i in range(len(segments) - 2, 0, -1):
        if segments[i]['direction'] == last_seg['direction']:
            prev_seg = segments[i]
            if prev_seg['area'] > 0:
                area_ratio = last_seg['area'] / prev_seg['area']
            else:
                area_ratio = 2.0

            # 盘整背驰:面积缩小
            if area_ratio < 0.7:
                is_trend_div = False
                # 检查是否有0轴回拉（判断是趋势背驰还是盘整背驰）
                min_idx = min(last_seg['start'], prev_seg['start'])
                max_idx = max(last_seg['end'], prev_seg['end'])
                crossed_zero = (macd_hist[min_idx:max_idx].min() < 0 and
                               macd_hist[min_idx:max_idx].max() > 0)

                if crossed_zero:
                    div_type = 'trend_bull' if last_seg['direction'] == 'down' else 'trend_bear'
                    guaranteed = True
                else:
                    div_type = 'consolidation_bull' if last_seg['direction'] == 'down' else 'consolidation_bear'
                    guaranteed = False

                confidence = min(95, (1.0 - area_ratio) * 100 + 20)

                return div_type, confidence, float(area_ratio), guaranteed

            break

    return 'none', 0.0, 1.0, False


# ============================================================
# Monte Carlo Physics-Inspired Methods
# (from Monte Carlo Methods in Statistical Physics, Newman & Barkema)
# ============================================================

def monte_carlo_simulated_tempering(returns, n_temps=5, n_iterations=5000, temp_range=(0.5, 5.0), random_seed=None):
    """
    Simulated Tempering for portfolio optimization.

    Runs multiple MCMC chains at different temperatures, periodically
    swapping states to escape local minima in the portfolio weight landscape.

    T_low explores local optimum (exploitation), T_high escapes basins (exploration).
    Acceptance probability: P_swap = min(1, exp((β_low - β_high)(E_high - E_low)))

    Args:
        returns: (n_periods, n_assets) array of asset returns
        n_temps: number of temperature levels
        n_iterations: MCMC iterations per chain
        temp_range: (T_min, T_max) for geometric spacing

    Returns:
        optimal_weights, tempering_diagnostics, swap_acceptance_rate
    """
    n_assets = returns.shape[1] if returns.ndim > 1 else 1
    if n_assets < 2:
        return np.ones(1) / max(np.sum(np.ones(1)), 1), {}, 0.0

    mean_ret = np.mean(returns, axis=0)
    # Robust covariance via Ledoit-Wolf shrinkage (fixes N > T singularity)
    try:
        from covariance_shrinkage import robust_covariance
        cov = robust_covariance(returns.T, method='ledoit_wolf')
    except ImportError:
        cov = np.cov(returns.T)
    cov_inv = np.linalg.pinv(cov)

    # Geometric temperature ladder
    temps = np.geomspace(temp_range[0], temp_range[1], n_temps)
    betas = 1.0 / temps

    # Energy function: negative Sharpe-like utility
    def energy(w):
        port_ret = np.dot(w, mean_ret)
        port_var = np.dot(w, np.dot(cov, w))
        return -(port_ret / max(np.sqrt(port_var), 1e-8))

    # Initialize chains at random weights
    rng = np.random.RandomState(random_seed)
    chains = []
    for _ in range(n_temps):
        w = rng.dirichlet(np.ones(n_assets))
        chains.append({'weights': w, 'energy': energy(w), 'history': [w.copy()]})

    swap_accepted = 0
    swap_attempted = 0

    for iteration in range(n_iterations):
        # Local Metropolis step for each chain
        for t in range(n_temps):
            w_current = chains[t]['weights']
            # Propose new weights via Dirichlet perturbation
            proposal_scale = 0.05 * temps[t]
            delta = rng.normal(0, proposal_scale, n_assets)
            w_proposal = w_current + delta
            w_proposal = np.maximum(w_proposal, 0.001)
            w_proposal /= np.sum(w_proposal)

            e_current = energy(w_current)
            e_proposal = energy(w_proposal)
            delta_e = e_proposal - e_current

            # Metropolis acceptance
            if delta_e < 0 or rng.random() < np.exp(-betas[t] * delta_e):
                chains[t]['weights'] = w_proposal
                chains[t]['energy'] = e_proposal
                chains[t]['history'].append(w_proposal.copy())

        # Temperature swap (parallel tempering) every 10 iterations
        if iteration % 10 == 0 and iteration > 0:
            for t in range(n_temps - 1):
                swap_attempted += 1
                e_high = chains[t]['energy']
                e_low = chains[t + 1]['energy']
                beta_high = betas[t]
                beta_low = betas[t + 1]

                swap_prob = min(1.0, np.exp((beta_low - beta_high) * (e_high - e_low)))
                if rng.random() < swap_prob:
                    chains[t]['weights'], chains[t + 1]['weights'] = chains[t + 1]['weights'], chains[t]['weights']
                    chains[t]['energy'], chains[t + 1]['energy'] = chains[t + 1]['energy'], chains[t]['energy']
                    swap_accepted += 1

    swap_rate = swap_accepted / max(swap_attempted, 1)

    # Optimal weights from lowest temperature chain (best exploitation)
    optimal_weights = chains[0]['weights']

    # Convergence diagnostics
    history_stack = np.array(chains[0]['history'][-500:])
    weight_std = float(np.mean(np.std(history_stack, axis=0)))

    diagnostics = {
        'n_temperatures': n_temps,
        'temperature_range': (float(temps[0]), float(temps[-1])),
        'swap_acceptance_rate': round(swap_rate, 3),
        'convergence_flag': weight_std < 0.02,
        'weight_uncertainty': round(weight_std, 4),
        'final_energy': round(float(chains[0]['energy']), 4),
        'energies_by_temp': [round(float(c['energy']), 4) for c in chains],
    }

    return optimal_weights, diagnostics, swap_rate


def monte_carlo_wolff_cluster(returns, threshold=1.5):
    """
    Wolff Cluster algorithm for market regime detection.

    Identifies clusters of highly-correlated assets/periods using the
    Wolff single-cluster Monte Carlo algorithm adapted for financial data.

    P_add(bond) = 1 - exp(-2 * beta * J_ij) where:
      - beta = inverse temperature (controls cluster size)
      - J_ij = correlation strength between asset i and j

    Small clusters → mean-reverting / fragmented market
    Large clusters → trending / herding behavior
    Cluster size ∝ susceptibility χ = β * <n>

    Args:
        returns: (n_periods, n_assets) array
        threshold: correlation threshold for bond activation

    Returns:
        cluster_labels, cluster_sizes, susceptibility, regime_assessment
    """
    n_assets = returns.shape[1] if returns.ndim > 1 else 1
    if n_assets < 2:
        return np.zeros(1), [1.0], 0.0, 'single_asset'

    # Correlation matrix as interaction strength J_ij
    corr = np.corrcoef(returns.T)
    J = np.abs(corr)
    np.fill_diagonal(J, 0)

    # Adaptive beta: higher dispersion → lower beta (smaller clusters)
    mean_corr = np.mean(np.abs(corr[np.triu_indices(n_assets, 1)]))
    beta = 1.0 / max(mean_corr + 0.1, 0.15)

    # Wolff algorithm: flip clusters
    n_runs = 20
    all_cluster_sizes = []

    for run in range(n_runs):
        # Random seed asset
        seed = np.random.randint(0, n_assets)
        cluster = {seed}
        frontier = {seed}

        while frontier:
            new_frontier = set()
            for i in frontier:
                for j in range(n_assets):
                    if j not in cluster and J[i, j] > threshold / n_assets * 2:
                        p_add = 1.0 - np.exp(-2.0 * beta * J[i, j])
                        if np.random.random() < p_add:
                            new_frontier.add(j)
                            cluster.add(j)
            frontier = new_frontier

        all_cluster_sizes.append(len(cluster))

    avg_cluster_size = np.mean(all_cluster_sizes)
    susceptibility = beta * avg_cluster_size / n_assets

    # Final clustering using average proximity
    bond_prob = 1.0 - np.exp(-2.0 * beta * J)
    np.fill_diagonal(bond_prob, 0)

    # Simple agglomerative: assets that frequently co-cluster are same regime
    from collections import defaultdict
    co_occurrence = np.zeros((n_assets, n_assets))
    for _ in range(50):
        seed = np.random.randint(0, n_assets)
        cluster = {seed}
        frontier = {seed}
        visited = set()
        while frontier:
            new_frontier = set()
            for i in frontier:
                visited.add(i)
                for j in range(n_assets):
                    if j not in visited and bond_prob[i, j] > 0.3:
                        if np.random.random() < bond_prob[i, j]:
                            new_frontier.add(j)
                            cluster.add(j)
            frontier = new_frontier
        cluster_list = list(cluster)
        for i in cluster_list:
            for j in cluster_list:
                co_occurrence[i, j] += 1

    co_occurrence /= 50
    cluster_labels = np.zeros(n_assets, dtype=int)
    label = 1
    assigned = set()
    for i in range(n_assets):
        if i not in assigned:
            group = [j for j in range(n_assets) if co_occurrence[i, j] > 0.5]
            for j in group:
                cluster_labels[j] = label
                assigned.add(j)
            label += 1

    n_clusters = label - 1
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_size_ratio = max(counts) / n_assets if n_assets > 0 else 0

    # Regime assessment
    if susceptibility > 0.7:
        regime = 'trending_herding'
        description = '大簇团→羊群效应/趋势市'
    elif susceptibility > 0.4:
        regime = 'sector_rotation'
        description = '中等簇团→板块轮动'
    elif susceptibility > 0.2:
        regime = 'fragmented'
        description = '小簇团→个股分化/均值回归'
    else:
        regime = 'independent'
        description = '无簇团→完全随机/无方向'

    return (cluster_labels,
            {'max_cluster_ratio': round(float(cluster_size_ratio), 3),
             'n_clusters': n_clusters,
             'susceptibility': round(float(susceptibility), 3),
             'regime': regime,
             'description': description})


def monte_carlo_finite_size_scaling(signals_dict, timeframes=None):
    """
    Finite-Size Scaling for multi-timeframe pattern validation.

    From statistical physics: near critical points, system properties obey
    scaling relations independent of system size L.

    χ_L(t) = L^{γ/ν} · X(L^{1/ν} · t)

    Applied to trading: if a pattern is "real" (near a critical transition),
    it should exhibit data collapse across timeframes.

    Args:
        signals_dict: {timeframe_name: signal_strength_array}
        timeframes: list of timeframe names, e.g. ['5min','30min','daily','weekly']

    Returns:
        scaling_exponent, collapse_quality, multi_tf_consensus, regime_assessment
    """
    if timeframes is None:
        timeframes = list(signals_dict.keys())

    if len(timeframes) < 2:
        return {'scaling_exponent': None, 'collapse_quality': 0.0,
                'consensus': 'insufficient_data'}

    # Compute signal properties at each timeframe
    tf_stats = {}
    for tf_name in timeframes:
        sig = signals_dict.get(tf_name)
        if sig is None or len(sig) < 5:
            continue
        tf_stats[tf_name] = {
            'mean': float(np.mean(sig)),
            'std': float(np.std(sig)),
            'L': len(sig),  # "system size" = number of bars
            'skew': float(np.mean((sig - np.mean(sig))**3) / max(np.std(sig)**3, 1e-8)),
            'kurtosis': float(np.mean((sig - np.mean(sig))**4) / max(np.std(sig)**4, 1e-8)),
        }

    if len(tf_stats) < 2:
        return {'scaling_exponent': 0.0, 'collapse_quality': 0.0,
                'consensus': 'insufficient_data'}

    sizes = np.array([s['L'] for s in tf_stats.values()])
    means = np.array([s['mean'] for s in tf_stats.values()])
    stds = np.array([s['std'] for s in tf_stats.values()])

    # Estimate scaling exponent γ/ν from log-log regression:
    # std(L) ∝ L^{γ/ν} → log(std) = (γ/ν)·log(L) + const
    log_L = np.log(sizes)
    log_std = np.log(np.maximum(stds, 1e-8))
    slope = np.polyfit(log_L, log_std, 1)[0]
    gamma_over_nu = float(slope)

    # Data collapse attempt: scale signals to test if they overlap
    scaled_signals = []
    for i, (tf_name, stats) in enumerate(tf_stats.items()):
        sig = signals_dict[tf_name]
        scaled = (sig - stats['mean']) / max(stats['std'], 1e-8)
        scaled = scaled * (sizes[i] ** (-gamma_over_nu))
        scaled_signals.append(scaled)

    # Collapse quality: how well do scaled distributions overlap?
    # Uses K-S statistic between pairs of scaled distributions
    from scipy import stats as scipy_stats
    ks_values = []
    for i in range(len(scaled_signals)):
        for j in range(i + 1, len(scaled_signals)):
            ks_stat, _ = scipy_stats.ks_2samp(scaled_signals[i], scaled_signals[j])
            ks_values.append(ks_stat)

    avg_ks = float(np.mean(ks_values)) if ks_values else 1.0
    collapse_quality = max(0.0, 1.0 - avg_ks)

    # Consensus assessment
    mean_direction = np.mean(np.sign(means))
    if collapse_quality > 0.7:
        if abs(mean_direction) > 0.5:
            consensus = 'strong_consensus'
        else:
            consensus = 'structural_consensus'
    elif collapse_quality > 0.4:
        consensus = 'moderate_consensus'
    else:
        consensus = 'no_consensus'

    # Physical interpretation of γ/ν
    if gamma_over_nu > 0.5:
        scaling_regime = 'critical_point_nearby'
    elif gamma_over_nu > 0.1:
        scaling_regime = 'correlated_regime'
    elif gamma_over_nu > -0.1:
        scaling_regime = 'mean_field'
    else:
        scaling_regime = 'sub_correlated'

    return {
        'scaling_exponent': round(gamma_over_nu, 4),
        'gamma_over_nu': round(gamma_over_nu, 4),
        'collapse_quality': round(collapse_quality, 4),
        'consensus': consensus,
        'scaling_regime': scaling_regime,
        'timeframe_stats': tf_stats,
        'collapse_assessment': (
            'strong pattern validity (data collapse across TFs)'
            if collapse_quality > 0.7
            else 'moderate pattern validity'
            if collapse_quality > 0.4
            else 'weak pattern — may be noise'
        ),
    }


# ============================================================
# AA土匪版缠论 Advanced Methods
# (from AA土匪版108课简体 annotations)
# ============================================================

def chanlun_bardo_stage(close, high, low, bb_upper, bb_mid, bb_lower, zn_high, zn_low):
    """
    中阴阶段检测 (Bardo Stage Detection)

    缠中说禅第88-90课核心概念:两个已完成的同级别中枢之间，价格处于
    "中阴阶段" — 方向未定，所有买卖点仅在该阶段内部有效。

    特征:
    - 价格脱离前一中枢但未进入（或未确认）新中枢
    - BOLL通道辅助:收口=中阴蓄势；开口=方向即将明朗
    - 三种结局:三买确认（上破+回踩）、三卖确认（下破+反抽）、中枢扩展

    Args:
        close, high, low: price arrays
        bb_upper, bb_mid, bb_lower: Bollinger Band arrays
        zn_high, zn_low: 前一中枢的上下轨

    Returns:
        bardo_state, bardo_direction, bardo_confidence, bardo_phase
    """
    n = len(close)
    if n < 20:
        return 'no_bardo', 'neutral', 0.0, {}

    current_price = close[-1]
    bb_width = (bb_upper[-1] - bb_lower[-1]) / max(bb_mid[-1], 0.01)
    bb_width_prev = (bb_upper[-10] - bb_lower[-10]) / max(bb_mid[-10], 0.01)

    # 中阴阶段判定:价格在两个中枢之间
    above_zn = current_price > zn_high
    below_zn = current_price < zn_low

    # BOLL收口检测
    boll_squeezing = bb_width < bb_width_prev * 0.85

    # 价格相对BOLL位置
    price_vs_bb = (current_price - bb_lower[-1]) / max(bb_upper[-1] - bb_lower[-1], 0.01)

    bardo_state = 'no_bardo'
    bardo_direction = 'neutral'
    confidence = 0.0
    phase_info = {}

    if not above_zn and not below_zn:
        bardo_state = 'no_bardo'
        bardo_direction = 'neutral'
        confidence = 0.0
        phase_info = {'reason': '价格在中枢内部，非中阴阶段'}
    elif above_zn:
        # 价格在中枢上方 — 潜在三买或回落
        bardo_state = 'bardo_above'

        # 检查BOLL形态
        if boll_squeezing and price_vs_bb > 0.7:
            bardo_direction = 'bullish_bias'
            confidence = 60
            phase_info = {'reason': '中阴上方+BOLL收口高位→蓄势上破', 'risk': '假突破风险'}
        elif boll_squeezing and price_vs_bb < 0.5:
            bardo_direction = 'bearish_bias'
            confidence = 50
            phase_info = {'reason': '中阴上方+BOLL收口回落→可能三卖', 'risk': '三买失败风险'}
        elif bb_width > bb_width_prev * 1.15:
            # BOLL开口 — 方向已出
            if close[-1] > close[-5]:
                bardo_direction = 'bullish'
                confidence = 75
                phase_info = {'reason': 'BOLL扩张向上→三买确认中', 'risk': '追高风险'}
            else:
                bardo_direction = 'bearish'
                confidence = 65
                phase_info = {'reason': 'BOLL扩张向下→三买失败/三卖形成', 'risk': '假跌破风险'}
        else:
            bardo_direction = 'neutral'
            confidence = 30
            phase_info = {'reason': '中阴上方震荡→等待方向确认'}
    else:
        # 价格在中枢下方
        bardo_state = 'bardo_below'

        if boll_squeezing and price_vs_bb < 0.3:
            bardo_direction = 'bearish_bias'
            confidence = 60
            phase_info = {'reason': '中阴下方+BOLL收口低位→蓄势下破', 'risk': '空头陷阱风险'}
        elif boll_squeezing and price_vs_bb > 0.5:
            bardo_direction = 'bullish_bias'
            confidence = 50
            phase_info = {'reason': '中阴下方+BOLL收口反弹→可能三买', 'risk': '反弹失败风险'}
        elif bb_width > bb_width_prev * 1.15:
            if close[-1] < close[-5]:
                bardo_direction = 'bearish'
                confidence = 75
                phase_info = {'reason': 'BOLL扩张向下→三卖确认中', 'risk': '杀跌风险'}
            else:
                bardo_direction = 'bullish'
                confidence = 65
                phase_info = {'reason': 'BOLL扩张向上→三卖失败/三买形成', 'risk': '假突破风险'}
        else:
            bardo_direction = 'neutral'
            confidence = 30
            phase_info = {'reason': '中阴下方震荡→等待方向确认'}

    return bardo_state, bardo_direction, round(confidence, 1), {
        'bb_width_current': round(float(bb_width), 4),
        'bb_squeezing': boll_squeezing,
        'price_vs_bb_position': round(float(price_vs_bb), 3),
        **phase_info,
    }


def chanlun_same_level_decomposition(high, low, close, timeframe='daily'):
    """
    同级别分解机械系统 (Same-Level Mechanical Decomposition)

    AA土匪版缠论核心系统之一:严格在同级别上分解走势为"向上段+向下段"的
    交替序列。每一段都由至少3笔构成，且与相邻段不重叠。

    机械交易规则:
    - 向上段结束=卖点（该级别一卖）
    - 向下段结束=买点（该级别一买）
    - 段内部不交易（等段确认后再操作）

    Returns mechanical buy/sell signals at the specified level.
    """
    n = len(high)
    if n < 20:
        return [], 'insufficient_data'

    # Use simple swing detection for segment identification
    swings_high = []
    swings_low = []
    for i in range(2, n - 2):
        if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
            swings_high.append((i, high[i]))
        if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
            swings_low.append((i, low[i]))

    if len(swings_high) < 3 or len(swings_low) < 3:
        return [], 'insufficient_swings'

    # Merge swings into alternating segments
    all_swings = [(idx, 'H', val) for idx, val in swings_high] + [(idx, 'L', val) for idx, val in swings_low]
    all_swings.sort(key=lambda x: x[0])

    # Build segments: alternating high-low-high-low...
    segments = []
    seg_start = all_swings[0]
    for i in range(1, len(all_swings)):
        curr = all_swings[i]
        if curr[1] != seg_start[1]:
            segments.append({
                'start_idx': seg_start[0],
                'end_idx': curr[0],
                'type': 'up' if seg_start[1] == 'L' else 'down',
                'start_price': seg_start[2],
                'end_price': curr[2],
                'magnitude': abs(curr[2] - seg_start[2]) / max(abs(seg_start[2]), 0.01),
            })
            seg_start = curr

    if len(segments) < 3:
        return [], 'insufficient_segments'

    # Mechanical signals: trade only at segment boundaries
    signals = []
    for i in range(1, len(segments)):
        seg = segments[i]
        prev_seg = segments[i-1]

        if seg['type'] == 'up' and prev_seg['type'] == 'down':
            # Down segment ended → buy signal (same-level buy point)
            if seg['magnitude'] > 0.01:
                signals.append({
                    'type': 'buy',
                    'idx': seg['start_idx'],
                    'price': round(seg['start_price'], 2),
                    'reason': f'向下段结束→同级别一买',
                    'strength': 'strong' if seg['magnitude'] > 0.03 else 'normal',
                })

        elif seg['type'] == 'down' and prev_seg['type'] == 'up':
            # Up segment ended → sell signal
            if seg['magnitude'] > 0.01:
                signals.append({
                    'type': 'sell',
                    'idx': seg['start_idx'],
                    'price': round(seg['start_price'], 2),
                    'reason': f'向上段结束→同级别一卖',
                    'strength': 'strong' if seg['magnitude'] > 0.03 else 'normal',
                })

    # Latest assessment
    latest_seg = segments[-1]
    if latest_seg['type'] == 'up':
        current_assessment = 'in_up_segment'
        recommendation = '持有/等待向上段完成'
    else:
        current_assessment = 'in_down_segment'
        recommendation = '观望/等待向下段完成'

    return signals, {
        'n_segments': len(segments),
        'latest_segment': latest_seg,
        'current_assessment': current_assessment,
        'recommendation': recommendation,
        'recent_signals': signals[-3:] if signals else [],
    }


def chanlun_sector_strength_ranking(sector_data):
    """
    板块强弱指标 (Sector Strength Quantitative Ranking)

    AA土匪版缠论:通过比较各板块的走势结构来选出最强/最弱板块。
    强势板块特征:日线中枢上移、周线向上笔延伸、板块内多数个股站上MA60
    弱势板块特征:中枢下移、周线向下笔、多数个股跌破MA60

    Args:
        sector_data: dict of {sector_name: {
            'zn_centers': list of中枢中心价格 (recent N),
            'price': current index price,
            'ma60': MA60 value,
            'weekly_stroke_dir': 'up'/'down',
            'pct_above_ma60': percentage of constituent stocks above MA60,
            'relative_strength': 相对大盘强度 (20-day),
        }}

    Returns:
        rankings list, strongest, weakest, sector_rotation_signal
    """
    if not sector_data:
        return [], None, None, 'no_data'

    scores = []
    for name, data in sector_data.items():
        score = 50.0  # baseline

        # 1. 中枢移动方向 (+/- 20)
        zn_centers = data.get('zn_centers', [])
        if len(zn_centers) >= 3:
            zn_change = (zn_centers[-1] - zn_centers[0]) / max(abs(zn_centers[0]), 0.01)
            score += np.clip(zn_change * 100, -20, 20)

        # 2. 价格 vs MA60 (+/- 15)
        price = data.get('price', 0)
        ma60 = data.get('ma60', 1)
        if price > 0 and ma60 > 0:
            pct_above = (price - ma60) / ma60 * 100
            score += np.clip(pct_above * 0.5, -15, 15)

        # 3. 周线笔方向 (+/- 15)
        weekly_dir = data.get('weekly_stroke_dir', 'neutral')
        if weekly_dir == 'up':
            score += 12
        elif weekly_dir == 'down':
            score -= 12

        # 4. 板块内个股强度 (+/- 20)
        pct_above_ma = data.get('pct_above_ma60', 50)
        score += np.clip((pct_above_ma - 50) * 0.4, -20, 20)

        # 5. 相对强度 (+/- 10)
        rel_str = data.get('relative_strength', 0)
        score += np.clip(rel_str * 50, -10, 10)

        scores.append({
            'sector': name,
            'score': round(score, 1),
            'strength_level': '强' if score > 70 else '偏强' if score > 55 else '中性' if score > 45 else '偏弱' if score > 30 else '弱',
        })

    scores.sort(key=lambda x: x['score'], reverse=True)
    strongest = scores[0] if scores else None
    weakest = scores[-1] if scores else None

    # Rotation signal: if strongest-weakest spread is widening
    if len(scores) >= 3:
        spread = scores[0]['score'] - scores[-1]['score']
        if spread > 40:
            rotation = 'extreme_divergence'
        elif spread > 25:
            rotation = 'diverging'
        elif spread > 15:
            rotation = 'normal'
        else:
            rotation = 'converging'
    else:
        rotation = 'insufficient_data'

    return scores, strongest, weakest, rotation


def chanlun_ma_kiss_system(ma_short, ma_long, close, volume=None):
    """
    均线吻系统 (MA Kiss System)

    缠中说禅独创的均线分析框架。两根均线(如MA5/MA10)的关系分为三种吻:

    1. 飞吻 (Flying Kiss): 两线短暂靠近但未交叉即分离 → 原趋势极强，顺势操作
    2. 唇吻 (Lip Kiss): 两线接触/轻微交叉后立即恢复原方向 → 趋势延续确认
    3. 湿吻 (Wet Kiss): 两线反复交叉缠绕 → 趋势可能转折，震荡蓄势

    操作含义:
    - 飞吻后=最佳顺势追进点
    - 唇吻后=趋势确认加仓点
    - 湿吻后=变盘预警，可能形成新中枢

    Args:
        ma_short, ma_long: arrays of short/long MA values
        close: close price array (same length)
        volume: optional volume array

    Returns:
        kiss_type, kiss_phase, trading_signal, kiss_details
    """
    n = len(ma_short)
    if n < 20:
        return 'unknown', 'insufficient_data', 'hold', {}

    # Compute MA distance
    ma_diff = ma_short - ma_long
    ma_diff_pct = ma_diff / np.maximum(np.abs(ma_long), 0.01)

    # Detect crossings and near-crossings
    cross_threshold = 0.005  # 0.5% of MA value = "kiss range"

    # Recent period analysis
    recent_diff = ma_diff_pct[-20:]
    recent_abs_diff = np.abs(recent_diff)

    # Minimum distance in recent window
    min_dist_idx = np.argmin(recent_abs_diff)
    min_dist = float(recent_abs_diff[min_dist_idx])

    # Current MA orientation
    ma_short_slope = (ma_short[-1] - ma_short[-5]) / max(abs(ma_short[-5]), 0.01)
    ma_long_slope = (ma_long[-1] - ma_long[-5]) / max(abs(ma_long[-5]), 0.01)

    # Volume confirmation
    if volume is not None and len(volume) == n:
        vol_ratio = volume[-5:].mean() / max(volume[-20:-5].mean(), 1)
    else:
        vol_ratio = 1.0

    # Classify kiss type
    if min_dist > 0.02:
        # Well separated → normal trending, no kiss
        kiss_type = 'no_kiss'
        if ma_diff[-1] > 0:
            kiss_phase = 'bullish_above'
            signal = 'hold_long'
            detail = '均线多头排列无吻→持仓'
        else:
            kiss_phase = 'bearish_below'
            signal = 'hold_short'
            detail = '均线空头排列无吻→观望'

    elif min_dist > cross_threshold:
        # Near but no cross → 飞吻
        kiss_type = 'flying_kiss'
        if ma_diff[-1] > 0:
            kiss_phase = 'bullish_flying'
            signal = 'strong_buy'
            detail = '飞吻(线上)→趋势极强，顺势追进'
        else:
            kiss_phase = 'bearish_flying'
            signal = 'strong_sell'
            detail = '飞吻(线下)→空头极强，顺势做空'

    elif min_dist > -cross_threshold:
        # Brief touch or minimal cross → 唇吻
        kiss_type = 'lip_kiss'
        n_crosses = np.sum(np.diff(np.sign(recent_diff)) != 0)
        if n_crosses <= 2:
            if ma_diff[-1] > 0:
                kiss_phase = 'bullish_lip'
                signal = 'buy'
                detail = '唇吻后恢复多头→趋势确认，可加仓'
            else:
                kiss_phase = 'bearish_lip'
                signal = 'sell'
                detail = '唇吻后恢复空头→趋势确认，可加空'
        else:
            kiss_phase = 'lip_transition'
            signal = 'wait'
            detail = '唇吻多次→方向待确认'

    else:
        # Deep cross / repeated crossing → 湿吻
        kiss_type = 'wet_kiss'
        n_crosses = np.sum(np.diff(np.sign(recent_diff)) != 0)
        if n_crosses >= 3:
            kiss_phase = 'wet_consolidation'
            signal = 'neutral'
            # 湿吻后方向:看斜率
            if ma_short_slope > 0.01 and ma_long_slope > 0:
                detail = '湿吻蓄势→中枢形成，偏多突破预期'
                signal = 'buy_watch'
            elif ma_short_slope < -0.01 and ma_long_slope < 0:
                detail = '湿吻蓄势→中枢形成，偏空突破预期'
                signal = 'sell_watch'
            else:
                detail = '湿吻缠绕→中枢震荡，等待方向'
        else:
            kiss_phase = 'wet_early'
            signal = 'wait'
            detail = '湿吻初期→趋势转折预警'

    # Volume adjustment
    if vol_ratio > 1.5:
        if 'buy' in signal:
            signal = 'strong_buy' if 'strong' not in signal else signal
            detail += '; 放量确认'
        elif 'sell' in signal:
            signal = 'strong_sell' if 'strong' not in signal else signal
            detail += '; 放量确认'

    details = {
        'kiss_type': kiss_type,
        'kiss_phase': kiss_phase,
        'min_distance_pct': round(min_dist * 100, 2),
        'ma_short_slope': round(float(ma_short_slope), 4),
        'ma_long_slope': round(float(ma_long_slope), 4),
        'volume_confirmation': round(float(vol_ratio), 2),
        'detail': detail,
    }

    return kiss_type, kiss_phase, signal, details


def chanlun_three_independent_systems(sys1_probs, sys2_probs, sys3_probs, min_threshold=0.5):
    """
    三个独立系统乘法原则 (Three Independent Systems Multiplication)

    缠中说禅核心技术哲学:选择三个完全独立的交易系统，将它们给出的
    概率用乘法结合。如果三个系统相互独立，联合判断的可靠性呈指数级提升。

    系统1: 技术分析系统（走势类型/背驰/买卖点）→ P(win|sys1)
    系统2: 比价关系系统（板块强弱/相对估值）→ P(win|sys2)
    系统3: 基本面/资金面系统（资金流向/宏观经济）→ P(win|sys3)

    乘法原则: P(win|all) ≈ P1 × P2 × P3（假设条件独立）

    Args:
        sys1_probs: {direction: probability} from technical system
        sys2_probs: {direction: probability} from relative value system
        sys3_probs: {direction: probability} from fundamental/flow system

    Returns:
        joint_probs, meets_threshold, best_direction, system_contributions
    """
    directions = ['bullish', 'bearish', 'neutral']
    joint_probs = {}

    for d in directions:
        p1 = sys1_probs.get(d, 0.33)
        p2 = sys2_probs.get(d, 0.33)
        p3 = sys3_probs.get(d, 0.33)
        joint_probs[d] = p1 * p2 * p3

    # Normalize
    total = sum(joint_probs.values())
    if total > 0:
        for d in joint_probs:
            joint_probs[d] /= total

    best_direction = max(joint_probs, key=joint_probs.get)
    best_prob = joint_probs[best_direction]
    meets_threshold = best_prob >= min_threshold

    # Individual system contributions
    contributions = {}
    for sys_name, probs in [('系统1(技术)', sys1_probs), ('系统2(比价)', sys2_probs), ('系统3(基本面)', sys3_probs)]:
        prob_for_best = probs.get(best_direction, 0.33)
        contributions[sys_name] = {
            'probability': round(prob_for_best, 3),
            'contribution_strength': '强' if prob_for_best > 0.6 else '中' if prob_for_best > 0.4 else '弱',
        }

    # Independence check warning
    # If all three systems give nearly identical probabilities, they may not be independent
    sys_probs = [sys1_probs.get(best_direction, 0.33),
                 sys2_probs.get(best_direction, 0.33),
                 sys3_probs.get(best_direction, 0.33)]
    prob_spread = max(sys_probs) - min(sys_probs)
    independence_warning = prob_spread < 0.1

    return {
        'joint_probabilities': {k: round(v, 4) for k, v in joint_probs.items()},
        'best_direction': best_direction,
        'best_probability': round(float(best_prob), 3),
        'meets_threshold': meets_threshold,
        'system_contributions': contributions,
        'independence_warning': independence_warning,
        'reliability': '高' if best_prob > 0.6 and not independence_warning
                       else '中' if best_prob > 0.4
                       else '低',
        'action': f'{best_direction}方向' if meets_threshold else '不满足阈值→观望',
    }


def chanlun_profit_rate_maximization(entry_price, zn_high, zn_low, trend_direction):
    """
    利润率最大定理 (Profit Rate Maximization Theorem)

    AA土匪版缠论核心定理:在中枢震荡中，利润最大化策略并非追涨杀跌，
    而是在中枢下沿附近买入、上沿附近卖出（做多时），且仓位与距离成正比。

    定理推导:
    - 买入点距中枢下沿越近，潜在利润(空间)越大，但成交概率越低
    - 最优买入点 = zn_low + (zn_high - zn_low) × α*
    - α*取决于趋势方向的非对称性

    做多时:最优买入 = zn_low + (zn_high - zn_low) × f(趋势强度)
    做空时:最优卖出 = zn_high - (zn_high - zn_low) × f(趋势强度)

    Args:
        entry_price: current price (or planned entry)
        zn_high, zn_low: 中枢上下轨
        trend_direction: 'up'/'down'/'consolidation'

    Returns:
        optimal_entry, optimal_exit, max_profit_rate, position_advice
    """
    zn_range = zn_high - zn_low
    if zn_range <= 0:
        return {'optimal_entry': entry_price, 'optimal_exit': entry_price,
                'max_profit_rate': 0, 'error': '无效中枢区间'}

    current_position_pct = (entry_price - zn_low) / zn_range

    # Asymmetric factor based on trend
    if trend_direction == 'up':
        # 上升趋势:中枢下沿附近买入
        optimal_entry_alpha = 0.15  # 下沿上方15%处
        optimal_exit_alpha = 0.95  # 上沿附近（留余地）
        position_factor = 1.0
    elif trend_direction == 'down':
        # 下降趋势:中枢上沿附近卖出
        optimal_entry_alpha = 0.85
        optimal_exit_alpha = 0.05
        position_factor = -1.0
    else:
        # 震荡:对称操作
        optimal_entry_alpha = 0.2
        optimal_exit_alpha = 0.8
        position_factor = 0.5

    optimal_entry = zn_low + zn_range * optimal_entry_alpha
    optimal_exit = zn_low + zn_range * optimal_exit_alpha

    if trend_direction == 'up':
        max_profit_rate = (optimal_exit - optimal_entry) / optimal_entry
    elif trend_direction == 'down':
        max_profit_rate = (optimal_entry - optimal_exit) / optimal_exit
    else:
        max_profit_rate = (optimal_exit - optimal_entry) / optimal_entry * 0.5

    # Position sizing: closer to optimal entry → larger position
    distance_from_optimal = abs(entry_price - optimal_entry) / zn_range
    position_multiplier = max(0.2, 1.0 - distance_from_optimal)

    return {
        'optimal_entry': round(optimal_entry, 2),
        'optimal_exit': round(optimal_exit, 2),
        'max_profit_rate': round(float(max_profit_rate * 100), 2),
        'current_position_in_zn': round(float(current_position_pct * 100), 1),
        'distance_from_optimal_pct': round(float(distance_from_optimal * 100), 1),
        'position_multiplier': round(float(position_multiplier), 2),
        'advice': (
            '当前价接近最优买入区→可执行' if distance_from_optimal < 0.15
            else '当前价偏离最优区→等待回调' if distance_from_optimal < 0.3
            else '当前价远离最优区→暂不操作'
        ),
    }


# ===================== 24. 统计套利与量化技术 (OCR analysis) =====================

def hurst_exponent(ts, max_lag=50):
    """
    Hurst exponent via R/S analysis (rescaled range).

    H > 0.5: trending/persistent
    H ≈ 0.5: random walk (Brownian)
    H < 0.5: mean-reverting/anti-persistent

    From: 量化投资策略与技术 (丁鹏), 量化交易 (Ernest Chan)
    """
    ts = np.asarray(ts, dtype=float)
    n = len(ts)
    if n < max_lag:
        max_lag = max(n // 4, 10)
    lags = np.unique(np.logspace(1, np.log10(max_lag), num=20).astype(int))
    lags = lags[lags < n // 2]

    rs_values = []
    for lag in lags:
        segments = n // lag
        if segments < 2:
            continue
        rs = []
        for s in range(segments):
            chunk = ts[s * lag:(s + 1) * lag]
            mean = chunk.mean()
            deviations = chunk - mean
            cum_dev = np.cumsum(deviations)
            r = cum_dev.max() - cum_dev.min()
            std = chunk.std(ddof=1)
            if std > 1e-12:
                rs.append(r / std)
        if rs:
            rs_values.append((lag, np.mean(rs)))

    if len(rs_values) < 3:
        return {'hurst': 0.5, 'regime': 'random_walk', 'confidence': 0.0, 'rs_data': []}

    log_lags = np.log([r[0] for r in rs_values])
    log_rs = np.log([r[1] for r in rs_values])
    slope, intercept = np.polyfit(log_lags, log_rs, 1)
    r2 = np.corrcoef(log_lags, log_rs)[0, 1] ** 2

    if slope > 0.60:
        regime = 'trending'
    elif slope < 0.40:
        regime = 'mean_reverting'
    else:
        regime = 'random_walk'

    return {
        'hurst': round(float(slope), 4),
        'regime': regime,
        'confidence': round(float(r2), 3),
        'intercept': round(float(intercept), 4),
        'half_life_approx': round(float(np.log(0.5) / np.log(slope)) if slope > 0 and slope != 1 else 0, 1),
        'rs_data': [(int(l), round(float(rs), 4)) for l, rs in rs_values],
    }


def variance_ratio_test(prices, k_list=(2, 4, 8, 16)):
    """
    Lo-MacKinlay variance ratio test for random walk hypothesis.

    VR(k) = Var(r_k) / (k * Var(r_1))
    VR < 1: mean-reverting (negative autocorrelation)
    VR > 1: trending/momentum (positive autocorrelation)

    From: 量化交易 (Ernest Chan), 统计套利 (Andrew Pole)
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < 32:
        return {'error': 'insufficient_data', 'n': n}

    log_returns = np.diff(np.log(prices))
    results = {}

    for k in k_list:
        if k > n // 4:
            continue
        # k-period overlapping log returns: r_t(k) = log(p_t) - log(p_{t-k})
        log_p = np.log(prices)
        r_k_period = log_p[k:] - log_p[:-k]  # k-period non-overlapping
        var_k = np.var(r_k_period, ddof=1) / k
        var_1 = np.var(log_returns, ddof=1)

        if var_1 < 1e-15:
            vr = 1.0
            z_score = 0.0
        else:
            vr = var_k / var_1
            z_score = np.sqrt(n) * (vr - 1) / np.sqrt(2 * (2 * k - 1) * (k - 1) / (3 * k))

        # 2-tail p-value approximation
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))

        results[f'VR({k})'] = {
            'vr': round(float(vr), 4),
            'z_score': round(float(z_score), 3),
            'p_value': round(float(p_value), 4),
            'interpretation': 'mean_reverting' if vr < 0.9 else 'trending' if vr > 1.1 else 'random_walk',
        }

    vr_values = [v['vr'] for v in results.values()]
    avg_vr = np.mean(vr_values) if vr_values else 1.0

    if avg_vr < 0.85:
        consensus = 'mean_reverting'
    elif avg_vr > 1.15:
        consensus = 'trending_momentum'
    else:
        consensus = 'random_walk'

    return {
        'results': results,
        'consensus': consensus,
        'avg_vr': round(float(avg_vr), 4),
    }


def ou_process_half_life(prices, max_lag=30):
    """
    Estimate mean-reversion half-life via OU process calibration.

    Model: dy(t) = θ(μ - y(t))dt + σ dW
    Half-life = ln(2) / θ

    Steps:
    1. Run regression: Δy_t = α + β·y_{t-1} + ε_t
    2. θ = -β (with dt), half-life ≈ ln(2) / |β| for daily data

    From: 量化交易 (Ernest Chan), 统计套利 (Andrew Pole)
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < max_lag + 5:
        max_lag = max(n // 3, 5)

    y = np.log(prices)
    dy = np.diff(y)
    y_lag = y[:-1]

    slope, intercept = np.polyfit(y_lag, dy, 1)
    beta = slope
    alpha = intercept

    if beta >= 0:
        return {
            'half_life': float('inf'),
            'is_mean_reverting': False,
            'theta': 0.0,
            'mu': float(np.mean(prices)),
            'message': '非均值回复(β≥0)，价格呈发散趋势',
        }

    theta = -beta
    half_life = np.log(2) / theta

    residuals = dy - (alpha + beta * y_lag)
    residual_var = np.var(residuals, ddof=1)
    sigma_ou = np.sqrt(residual_var * 2 * theta) if theta > 0 else 0.0
    mu_ou = -alpha / beta if beta != 0 else np.mean(y)

    half_life_quality = 'reliable' if half_life < n / 3 else 'unstable'

    return {
        'half_life': round(float(half_life), 1),
        'theta': round(float(theta), 6),
        'mu_log': round(float(mu_ou), 6),
        'mu_price': round(float(np.exp(mu_ou)), 4),
        'sigma_ou': round(float(sigma_ou), 6),
        'is_mean_reverting': True,
        'half_life_quality': half_life_quality,
        'optimal_holding_estimate': round(float(half_life * 0.7), 1),
        'message': f'半衰期={half_life:.1f}天，最优持仓≈{half_life * 0.7:.1f}天',
    }


def dow_confirmation_model(index1_close, index2_close, volume=None, lookback=50):
    """
    Dow Theory cross-index confirmation model.

    Core principle: Both averages (e.g., industrials + transports) must confirm
    each other for a valid trend signal. Non-confirmation = warning.

    From: 道氏理论 (Dow Theory), 丁鹏量化策略
    """
    idx1 = np.asarray(index1_close, dtype=float)
    idx2 = np.asarray(index2_close, dtype=float)
    n = min(len(idx1), len(idx2))
    if n < lookback:
        lookback = max(n // 2, 10)

    idx1 = idx1[-lookback:]
    idx2 = idx2[-lookback:]

    # Trend detection via peak/trough analysis
    ma1_20 = pd.Series(idx1).rolling(20).mean().values
    ma1_50 = pd.Series(idx1).rolling(min(50, lookback // 2)).mean().values
    ma2_20 = pd.Series(idx2).rolling(20).mean().values
    ma2_50 = pd.Series(idx2).rolling(min(50, lookback // 2)).mean().values

    # Check for higher highs / higher lows
    def trend_assess(close_series):
        half = len(close_series) // 2
        first_half_high = np.max(close_series[:half])
        second_half_high = np.max(close_series[half:])
        first_half_low = np.min(close_series[:half])
        second_half_low = np.min(close_series[half:])

        if second_half_high > first_half_high and second_half_low > first_half_low:
            return 'uptrend', 1
        elif second_half_high < first_half_high and second_half_low < first_half_low:
            return 'downtrend', -1
        else:
            return 'ranging', 0

    trend1, dir1 = trend_assess(idx1)
    trend2, dir2 = trend_assess(idx2)

    # Volume confirmation (rising volume on trend direction = valid)
    vol_confirm = True
    if volume is not None and len(volume) >= lookback:
        vol_arr = np.asarray(volume[-lookback:], dtype=float)
        first_vol = np.mean(vol_arr[:len(vol_arr) // 2])
        second_vol = np.mean(vol_arr[len(vol_arr) // 2:])
        vol_confirm = second_vol > first_vol

    # MA alignment check
    if ma1_20[-1] is not None and ma2_20[-1] is not None:
        ma1_bull = ma1_20[-1] > ma1_50[-1] if not np.isnan(ma1_50[-1]) else True
        ma2_bull = ma2_20[-1] > ma2_50[-1] if not np.isnan(ma2_50[-1]) else True
    else:
        ma1_bull = ma2_bull = True

    # Determine confirmation status
    if dir1 == dir2 and dir1 != 0:
        confirmed = True
        phase = '牛市确认' if dir1 > 0 else '熊市确认'
        strength = '强' if vol_confirm and ma1_bull and ma2_bull else '中'
    elif dir1 == dir2 == 0:
        confirmed = False
        phase = '盘整/无趋势'
        strength = '弱'
    else:
        confirmed = False
        phase = '背离警告:指数不同步'
        strength = '警告'

    return {
        'index1_trend': trend1,
        'index2_trend': trend2,
        'confirmed': confirmed,
        'phase': phase,
        'strength': strength,
        'volume_confirmation': vol_confirm,
        'ma_alignment': {'index1': ma1_bull, 'index2': ma2_bull},
        'signal': 1 if (confirmed and dir1 > 0 and vol_confirm) else
                 -1 if (confirmed and dir1 < 0 and vol_confirm) else 0,
        'advice': '趋势确认→顺势操作' if confirmed and vol_confirm else
                  '趋势背离→降低仓位或观望' if not confirmed else
                  '方向不明→等待确认信号',
    }


def market_impact_square_root(volume_shares, daily_vol, price=100.0,
                               sigma_daily=0.02, participation_rate=0.01):
    """
    Square-root market impact model for transaction cost estimation.

    ΔP/P ≈ σ · sign(Q) · sqrt(Q / ADV)  (Almgren-Chriss / square-root law)

    From: 高频交易 (Aldridge), 打开量化投资的黑箱 (Narang)

    Args:
        volume_shares: number of shares to trade
        daily_vol: average daily volume (shares)
        price: current stock price
        sigma_daily: daily volatility
        participation_rate: target participation rate

    Returns impact percentage, total cost, and execution advice.
    """
    if daily_vol <= 0:
        return {'error': 'invalid_daily_volume'}

    adv = daily_vol
    q = volume_shares
    sigma = sigma_daily

    # Permanent impact: ~σ * sqrt(Q/ADV) (square root model)
    impact_perm = sigma * np.sign(q) * np.sqrt(abs(q) / adv)

    # Temporary impact: depends on participation rate
    if participation_rate > 0:
        impact_temp = sigma * np.sign(q) * (abs(q) / (adv * participation_rate)) ** 0.6 * 0.5
    else:
        impact_temp = 0.0

    total_impact_pct = impact_perm + impact_temp
    cost_per_share = price * abs(total_impact_pct)
    total_cost = cost_per_share * abs(q)

    # VWAP execution advice based on order size
    if q / adv < 0.01:
        exec_advice = '小额订单:可一次性执行，冲击成本可忽略'
        risk_level = 'low'
    elif q / adv < 0.05:
        exec_advice = f'中额订单:建议TWAP分时执行，预期冲击{abs(total_impact_pct)*100:.2f}%'
        risk_level = 'medium'
    elif q / adv < 0.10:
        exec_advice = f'大额订单:建议VWAP算法或冰山订单，预期冲击{abs(total_impact_pct)*100:.2f}%'
        risk_level = 'high'
    else:
        exec_advice = f'超大订单:强烈建议多日分散执行，单日冲击达{abs(total_impact_pct)*100:.2f}%'
        risk_level = 'extreme'

    return {
        'permanent_impact_pct': round(float(impact_perm * 100), 4),
        'temporary_impact_pct': round(float(impact_temp * 100), 4),
        'total_impact_pct': round(float(total_impact_pct * 100), 4),
        'cost_per_share': round(float(cost_per_share), 4),
        'total_cost': round(float(total_cost), 2),
        'participation_ratio': round(float(q / adv * 100), 2),
        'risk_level': risk_level,
        'execution_advice': exec_advice,
    }


def volatility_targeted_position_size(account_equity, realized_vol=None,
                                       predicted_vol=None, target_vol=0.15,
                                       max_leverage=2.0, min_leverage=0.1):
    """
    Volatility-targeted position sizing with dynamic leverage adjustment.

    Position = (target_vol / max(realized_vol, predicted_vol)) × equity × leverage_factor

    This stabilizes portfolio risk by reducing exposure in high-vol regimes
    and increasing in low-vol regimes.

    From: 量化交易 (Ernest Chan), Statistical Arbitrage (Andrew Pole)
    """
    if realized_vol is None and predicted_vol is None:
        return {'error': 'need_at_least_one_vol_estimate'}

    # Use the higher of realized vs predicted for conservative sizing
    effective_vol = max(
        realized_vol if realized_vol is not None else 0.0,
        predicted_vol if predicted_vol is not None else 0.0,
        0.01  # floor
    )

    unlevered_size = account_equity * (target_vol / effective_vol)
    leverage = unlevered_size / account_equity if account_equity > 0 else 0.0

    # Apply leverage bounds
    leverage_clamped = np.clip(leverage, min_leverage, max_leverage)
    position_size = account_equity * leverage_clamped

    # Regime classification
    vol_ratio = target_vol / effective_vol
    if vol_ratio > 1.5:
        regime = 'low_vol_deploy'
    elif vol_ratio > 0.8:
        regime = 'normal'
    elif vol_ratio > 0.4:
        regime = 'high_vol_reduce'
    else:
        regime = 'extreme_vol_minimize'

    return {
        'position_size': round(float(position_size), 2),
        'leverage': round(float(leverage_clamped), 2),
        'effective_volatility': round(float(effective_vol * 100), 2),
        'target_volatility': round(float(target_vol * 100), 2),
        'vol_ratio': round(float(vol_ratio), 3),
        'regime': regime,
        'advice': {
            'low_vol_deploy': '低波动环境→可适度增加杠杆',
            'normal': '正常波动→维持目标敞口',
            'high_vol_reduce': '高波动环境→降低仓位保护本金',
            'extreme_vol_minimize': '极端波动→最小仓位等待回归',
        }.get(regime, '保持当前仓位'),
    }


def stochastic_resonance_signal(signal_raw, noise_levels=None, method='optimal', random_seed=None):
    """
    Stochastic resonance: noise-enhanced signal detection.

    Theory: Adding controlled noise can amplify weak periodic signals in nonlinear
    systems. Applied here to detect weak trading signals below noise floor.

    Process:
    1. Add controlled noise at multiple levels (η ∈ [0.1σ, 3.0σ])
    2. Measure signal-to-noise ratio (SNR) or mutual information at each level
    3. Find optimal noise level that maximizes detectability
    4. Return noise-enhanced signal

    From: 量化投资策略与技术 (丁鹏), Statistical Arbitrage (Andrew Pole)

    Args:
        signal_raw: 1D array of raw signal values (e.g., alpha scores, LLRs)
        noise_levels: list of noise std multipliers to test, or None for auto
        method: 'optimal' (find best noise) or 'all' (return all levels)

    Returns:
        enhanced signal, optimal noise level, SNR improvement
    """
    signal_raw = np.asarray(signal_raw, dtype=float)
    n = len(signal_raw)

    if n < 10:
        return {'error': 'insufficient_data', 'n': n}

    signal_std = np.std(signal_raw)
    if signal_std < 1e-12:
        return {'enhanced_signal': signal_raw, 'snr_improvement': 0.0,
                'optimal_noise_sigma': 0.0, 'message': '信号无波动→无需增强'}

    if noise_levels is None:
        noise_levels = np.linspace(0.1, 2.5, 20)

    # Estimate baseline SNR
    signal_power = np.var(signal_raw)
    # Noise floor estimated from high-frequency component (residual after smoothing)
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(signal_raw, size=max(3, n // 20))
    residuals = signal_raw - smoothed
    noise_power = np.var(residuals)
    baseline_snr = signal_power / max(noise_power, 1e-12)

    best_snr = baseline_snr
    best_noise_sigma = 0.0
    best_enhanced = signal_raw.copy()
    all_results = []

    np.random.seed(random_seed)
    for eta in noise_levels:
        noise_sigma = eta * signal_std
        noise = np.random.normal(0, noise_sigma, n)
        noisy_signal = signal_raw + noise

        # Apply threshold-crossing detection (nonlinearity = stochastic resonance key)
        # Weak signals amplified by noise → cross detection threshold more reliably
        threshold = np.percentile(signal_raw, 70)
        detections_raw = np.sum(np.abs(signal_raw) > threshold)
        detections_noisy = np.sum(np.abs(noisy_signal) > threshold)

        # Mutual information proxy: correlation between raw and noise-enhanced rank
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(signal_raw, noisy_signal)

        # Combined score: detection improvement × rank preservation
        detection_improvement = detections_noisy / max(detections_raw, 1)
        snr_metric = detection_improvement * rank_corr

        all_results.append({
            'eta': round(float(eta), 3),
            'noise_sigma': round(float(noise_sigma), 6),
            'snr_metric': round(float(snr_metric), 4),
            'detection_improvement': round(float(detection_improvement), 3),
            'rank_correlation': round(float(rank_corr), 4),
        })

        if snr_metric > best_snr:
            best_snr = snr_metric
            best_noise_sigma = noise_sigma
            best_enhanced = noisy_signal

    snr_improvement = best_snr / max(baseline_snr, 1e-12) - 1.0

    return {
        'enhanced_signal': best_enhanced.tolist() if method == 'optimal' else None,
        'optimal_noise_sigma': round(float(best_noise_sigma), 6),
        'optimal_eta': round(float(best_noise_sigma / signal_std) if signal_std > 0 else 0, 3),
        'baseline_snr': round(float(baseline_snr), 2),
        'enhanced_snr': round(float(best_snr), 4),
        'snr_improvement_pct': round(float(max(0, snr_improvement) * 100), 1),
        'is_useful': snr_improvement > 0.05,
        'message': (f'噪声增强有效: SNR提升{max(0,snr_improvement)*100:.1f}%'
                    if snr_improvement > 0.05 else
                    '噪声增强无显著效果→原始信号已足够强'),
        'all_levels': all_results if method == 'all' else None,
    }


# ===================== 25. 新增贝叶斯与统计技术 (from unprocessed books) =====================

def bayesian_changepoint_detection(series, n_samples=2000, rate_prior=(1.0, 1.0), random_seed=None):
    """
    Bayesian switchpoint (changepoint) detection via MCMC.

    Model: y_i ~ Poi(lambda_1) for i < tau; y_i ~ Poi(lambda_2) for i >= tau
    tau ~ DiscreteUniform(1, N-1)
    lambda_1, lambda_2 ~ Gamma(alpha, beta)

    NOTE: Poisson likelihood is used as an approximation for non-negative continuous
    data (shifted to positive range). For truly continuous financial returns, a Normal
    likelihood with unknown variance would be more appropriate, but the Poisson-Gamma
    conjugacy enables efficient Gibbs sampling. The changepoint LOCATION (tau) is the
    primary quantity of interest and is robust to this approximation.

    Returns posterior p(tau | data) giving probability of regime change at each index.

    From: Bayesian Methods for Hackers (Davidson-Pilon, Ch.1)
    """
    import numpy as np
    data = np.asarray(series, dtype=float)
    n = len(data)

    if n < 10:
        return {'error': 'insufficient_data', 'n': n}

    # Ensure non-negative for Poisson (shift if needed)
    data_min = data.min()
    if data_min < 0:
        data = data - data_min + 1e-6

    alpha_prior, beta_prior = rate_prior

    # Gibbs sampling
    tau_samples = np.zeros(n_samples)
    lambda_1_samples = np.zeros(n_samples)
    lambda_2_samples = np.zeros(n_samples)

    # Initialize
    rng = np.random.RandomState(random_seed)
    tau = n // 2
    lambda_1 = data[:tau].mean()
    lambda_2 = data[tau:].mean()

    for i in range(n_samples):
        # Update lambda_1
        alpha_post = alpha_prior + data[:tau].sum()
        beta_post = beta_prior + tau
        lambda_1 = rng.gamma(alpha_post, 1.0 / beta_post)

        # Update lambda_2
        alpha_post = alpha_prior + data[tau:].sum()
        beta_post = beta_prior + (n - tau)
        lambda_2 = rng.gamma(alpha_post, 1.0 / beta_post)

        # Update tau (Metropolis step)
        tau_new = tau + rng.randint(-5, 6)
        tau_new = max(1, min(n - 2, tau_new))

        if tau_new != tau:
            # Log likelihood ratio
            ll_old = (np.sum(data[:tau]) * np.log(lambda_1) - tau * lambda_1 +
                      np.sum(data[tau:]) * np.log(lambda_2) - (n - tau) * lambda_2)
            ll_new = (np.sum(data[:tau_new]) * np.log(lambda_1) - tau_new * lambda_1 +
                      np.sum(data[tau_new:]) * np.log(lambda_2) - (n - tau_new) * lambda_2)
            if np.log(rng.random()) < ll_new - ll_old:
                tau = tau_new

        tau_samples[i] = tau
        lambda_1_samples[i] = lambda_1
        lambda_2_samples[i] = lambda_2

    # Posterior analysis
    burn = n_samples // 4
    tau_posterior = tau_samples[burn:]
    tau_counts = np.bincount(tau_posterior.astype(int), minlength=n)
    tau_probs = tau_counts / len(tau_posterior)

    # Find most likely changepoint
    best_tau = int(np.argmax(tau_probs[1:-1]) + 1)
    best_prob = tau_probs[best_tau]

    # Identify all significant changepoints (peaks in tau distribution)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(tau_probs[1:-1], height=0.02, distance=max(5, n // 10))
    significant_taus = [(p + 1, round(float(tau_probs[p + 1]), 3)) for p in peaks
                        if tau_probs[p + 1] > 0.03]

    lambda_1_mean = lambda_1_samples[burn:].mean()
    lambda_2_mean = lambda_2_samples[burn:].mean()
    effect_size = abs(lambda_2_mean - lambda_1_mean) / max(lambda_1_mean, 1e-10)

    return {
        'best_changepoint_index': best_tau,
        'best_changepoint_prob': round(float(best_prob), 3),
        'is_significant': best_prob > 0.10,
        'significant_changepoints': significant_taus,
        'lambda_before': round(float(lambda_1_mean), 4),
        'lambda_after': round(float(lambda_2_mean), 4),
        'effect_size': round(float(effect_size), 3),
        'regime_change_type': 'bearish' if lambda_2_mean < lambda_1_mean else 'bullish',
        'tau_posterior': tau_probs.tolist(),
        'advice': (f'高概率变盘点(指数{best_tau}, p={best_prob:.2f})'
                   if best_prob > 0.15 else '无明显单一变盘点→渐变式调整'),
    }


def benford_anomaly_detection(data, n_first_digit=1):
    """
    Benford's Law anomaly detection for data integrity / manipulation.

    P(first digit = d) = log10(1 + 1/d), for d in {1,...,9}

    Large deviation from Benford suggests manipulated/artificial data.
    Useful for: order flow analysis, volume data quality, price manipulation detection.

    From: 贝叶斯的博弈 (黄黎原, Ch.11)

    Returns chi-squared statistic, p-value, anomalous digits.
    """
    import numpy as np
    from scipy.stats import chi2

    data = np.asarray(data, dtype=float)
    data = data[data > 0]
    n = len(data)

    if n < 30:
        return {'error': 'insufficient_data', 'n': n}

    # Extract first digits
    first_digits = np.array([int(str(abs(x)).replace('.', '').lstrip('0')[:n_first_digit] or '1')
                              for x in data])
    first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]

    # Benford expected proportions
    benford_probs = np.array([np.log10(1 + 1.0 / d) for d in range(1, 10)])
    benford_expected = benford_probs * len(first_digits)

    observed = np.array([np.sum(first_digits == d) for d in range(1, 10)])

    # Chi-squared test
    with np.errstate(divide='ignore', invalid='ignore'):
        chi_sq = np.sum((observed - benford_expected) ** 2 / np.maximum(benford_expected, 1))
    p_value = 1 - chi2.cdf(chi_sq, 8)

    # K-L divergence
    obs_probs = observed / max(len(first_digits), 1)
    # KL divergence — mask zeros to avoid 0*log(0) = NaN
    mask = obs_probs > 0
    kl_div = np.sum(obs_probs[mask] * np.log(obs_probs[mask] / benford_probs[mask])) if mask.any() else 0.0

    # Find most anomalous digits
    deviations = (obs_probs - benford_probs) / benford_probs
    anomalous = [(d + 1, round(float(deviations[d]), 3)) for d in range(9)
                 if abs(deviations[d]) > 0.3]

    return {
        'is_benford_compliant': p_value > 0.05,
        'chi_squared': round(float(chi_sq), 2),
        'p_value': round(float(p_value), 4),
        'kl_divergence': round(float(kl_div), 4),
        'anomalous_digits': anomalous,
        'n_samples': len(first_digits),
        'interpretation': ('数据符合Benford分布→自然数据' if p_value > 0.05 else
                           '数据偏离Benford→可能存在人为操作/筛选'),
    }


def laplace_succession_rule(successes, trials, group_successes=None, group_trials=None):
    """
    Laplace's Rule of Succession for rare-event probability estimation.

    P(success|k in n) = (k + 1) / (n + 2)

    With hierarchical shrinkage toward group rate when group data available:
    P = (k + alpha) / (n + alpha + beta)
    alpha = group_rate * kappa, beta = (1 - group_rate) * kappa

    From: Berger (Ch.3.6), 贝叶斯的博弈 (黄黎原, Ch.6)

    Use: Estimating tail-event/crash probabilities, win rate of infrequent signals.
    """
    if trials <= 0:
        return {'error': 'zero_trials'}

    # Base Laplace estimate
    laplace_prob = (successes + 1) / (trials + 2)

    # Hierarchical shrinkage
    if group_successes is not None and group_trials is not None and group_trials > 0:
        group_rate = group_successes / group_trials
        kappa = np.sqrt(trials)  # weight proportional to sqrt of own data
        alpha = group_rate * kappa
        beta_param = (1 - group_rate) * kappa
        hierarchical_prob = (successes + alpha) / (trials + alpha + beta_param)
    else:
        hierarchical_prob = laplace_prob
        group_rate = None

    # Wilson score interval (95% CI)
    z = 1.96
    p = laplace_prob
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p * (1 - p) / trials + z**2 / (4 * trials**2))) / denominator
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)

    # Expected number of additional trials to see next success
    if laplace_prob > 1e-10:
        expected_wait = 1.0 / laplace_prob
    else:
        expected_wait = float('inf')

    return {
        'laplace_probability': round(float(laplace_prob), 6),
        'hierarchical_probability': round(float(hierarchical_prob), 6),
        'ci_95': [round(float(ci_lower), 6), round(float(ci_upper), 6)],
        'expected_wait_trials': round(float(expected_wait), 1),
        'shrinkage_target': round(float(group_rate), 6) if group_rate else None,
        'is_reliable': trials > 20,
        'advice': (f'估计稳定(p≈{laplace_prob:.1%})' if trials > 20 else
                   f'样本不足(仅{trials}次)，建议使用层级估计' if group_successes else
                   f'样本不足(仅{trials}次)→增加试验'),
    }


def odds_ratio_signal_test(signal, returns, threshold=0.0):
    """
    Odds Ratio test for binary trading signal evaluation.

    Contingency table:
                  Return+  Return-
    Signal ON        n11      n12
    Signal OFF       n21      n22

    OR = (n11 * n22) / (n12 * n21)
    OR > 1: signal is positively associated with returns

    From: Ott & Longnecker (A First Course in Statistical Methods, Ch.10)
    """
    import numpy as np
    from scipy.stats import fisher_exact, chi2

    signal = np.asarray(signal, dtype=float)
    returns = np.asarray(returns, dtype=float)
    n = min(len(signal), len(returns))

    if n < 20:
        return {'error': 'insufficient_data', 'n': n}

    sig_on = signal > threshold
    ret_pos = returns > 0

    n11 = np.sum(sig_on & ret_pos)
    n12 = np.sum(sig_on & ~ret_pos)
    n21 = np.sum(~sig_on & ret_pos)
    n22 = np.sum(~sig_on & ~ret_pos)

    if n12 == 0 or n21 == 0:
        odds_ratio = float('inf') if n11 * n22 > 0 else 0.0
    else:
        odds_ratio = (n11 * n22) / (n12 * n21)

    if n12 + n21 + n11 + n22 == 0:
        return {'error': 'no_valid_observations'}

    # Fisher's exact test
    table = np.array([[n11, n12], [n21, n22]])
    _, p_value = fisher_exact(table) if min(table.shape) > 0 else (0, 1.0)

    # Signal quality metrics
    precision = n11 / max(n11 + n12, 1)
    recall = n11 / max(n11 + n21, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    accuracy = (n11 + n22) / (n11 + n12 + n21 + n22)

    # Bernstein interval for OR
    log_or = np.log(max(odds_ratio, 1e-10))
    se_log_or = np.sqrt(1.0 / max(n11, 1) + 1.0 / max(n12, 1) +
                         1.0 / max(n21, 1) + 1.0 / max(n22, 1))
    or_ci_low = np.exp(log_or - 1.96 * se_log_or)
    or_ci_high = np.exp(log_or + 1.96 * se_log_or)

    return {
        'odds_ratio': round(float(odds_ratio), 4) if odds_ratio != float('inf') else 'inf',
        'p_value': round(float(p_value), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'f1_score': round(float(f1), 4),
        'accuracy': round(float(accuracy), 4),
        'or_ci_95': [round(float(or_ci_low), 4), round(float(or_ci_high), 4)],
        'contingency': {'TP': int(n11), 'FP': int(n12), 'FN': int(n21), 'TN': int(n22)},
        'is_predictive': p_value < 0.05 and (odds_ratio > 1.5 if odds_ratio != 'inf' else True),
        'signal_quality': ('强预测信号' if odds_ratio != 'inf' and odds_ratio > 2.0 and p_value < 0.01 else
                           '中等预测信号' if odds_ratio != 'inf' and odds_ratio > 1.3 and p_value < 0.05 else
                           '弱/无预测力'),
    }


def bege_volatility_decomposition(returns, lookback=60):  # DEPRECATED: unused, use bebe_volatility_decomposition(df, window=20) instead
    """
    BEGE (Bad Environment - Good Environment) volatility decomposition.

    Decomposes variance into "good" (upside) and "bad" (downside) gamma-distributed
    consumption growth components, each with its own volatility loading.

    VRP = r^2_good * [1/(1-m_good)^2 - 1] * p + r^2_bad * [1/(1-m_bad)^2 - 1] * n

    Where p and n are gamma-distributed good/bad volatility components.

    From: BEGE Model (Bekaert-Engstrom), NBER w27108
    """
    import numpy as np
    ret = np.asarray(returns, dtype=float)
    n = len(ret)

    if n < lookback:
        lookback = max(n // 2, 10)

    # Separate positive and negative returns
    pos_ret = np.maximum(ret[-lookback:], 0)
    neg_ret = np.maximum(-ret[-lookback:], 0)

    # Fit gamma distributions via method of moments
    def fit_gamma_mom(data):
        data_pos = data[data > 0]
        if len(data_pos) < 5:
            return 1.0, 1.0
        mu = data_pos.mean()
        var = data_pos.var(ddof=1)
        if var < 1e-12 or mu < 1e-12:
            return 1.0, 1.0
        shape = mu**2 / var
        scale = var / mu
        return shape, scale

    shape_good, scale_good = fit_gamma_mom(pos_ret)
    shape_bad, scale_bad = fit_gamma_mom(neg_ret)

    # Gamma means = shape * scale
    p_t = shape_good * scale_good  # good component mean
    n_t = shape_bad * scale_bad   # bad component mean

    # Correlation between returns and squared returns (leverage-like effect)
    if n >= 2:
        corr_good = np.corrcoef(pos_ret, pos_ret**2)[0, 1] if len(pos_ret) > 1 else 0
        corr_bad = np.corrcoef(neg_ret, neg_ret**2)[0, 1] if len(neg_ret) > 1 else 0
    else:
        corr_good = corr_bad = 0.0

    # BEGE loading factors (m parameters capture persistence)
    m_good = min(0.95, max(0.1, np.abs(corr_good)))
    m_bad = min(0.95, max(0.1, np.abs(corr_bad)))

    # r parameters (volatility loadings)
    r_good = np.std(pos_ret, ddof=1) if len(pos_ret) > 1 else 0.01
    r_bad = np.std(neg_ret, ddof=1) if len(neg_ret) > 1 else 0.01

    # BEGE variance components
    var_good = r_good**2 * (1.0 / (1.0 - m_good)**2 - 1.0) * p_t
    var_bad = r_bad**2 * (1.0 / (1.0 - m_bad)**2 - 1.0) * n_t

    total_var = var_good + var_bad
    good_fraction = var_good / max(total_var, 1e-12)

    # VRP-like decomposition
    vrp_bad_component = var_bad - var_good  # positive = downside risk dominates

    return {
        'good_variance': round(float(var_good * 1e4), 2),
        'bad_variance': round(float(var_bad * 1e4), 2),
        'total_basis_points': round(float(total_var * 1e4), 2),
        'good_fraction': round(float(good_fraction), 3),
        'bad_fraction': round(float(1 - good_fraction), 3),
        'gamma_shape_good': round(float(shape_good), 3),
        'gamma_scale_good': round(float(scale_good * 100), 4),
        'gamma_shape_bad': round(float(shape_bad), 3),
        'gamma_scale_bad': round(float(scale_bad * 100), 4),
        'm_good_persistence': round(float(m_good), 4),
        'm_bad_persistence': round(float(m_bad), 4),
        'vrp_imbalance': round(float(vrp_bad_component * 1e4), 2),
        'risk_asymmetry': 'bad_dominates' if vrp_bad_component > 0 else
                          'balanced' if abs(vrp_bad_component) < 0.01 * max(total_var, 1e-12) else
                          'good_dominates',
        'interpretation': (f'下行风险主导(坏波动占比{100*(1-good_fraction):.0f}%)'
                           if 1 - good_fraction > 0.55 else
                           f'波动均衡分布' if 0.45 <= good_fraction <= 0.55 else
                           f'上行波动主导(好波动占比{100*good_fraction:.0f}%)'),
    }


def realized_har_garch(realized_var, daily_ret, lookback=120):
    """
    Realized HAR-GARCH: unified framework combining HAR structure with GARCH.

    Measurement: h_tilde_t = omega + beta*h_tilde_{t-1} + gamma_d*x_{t-1}
                  + gamma_w*(1/4)*sum_{i=2}^5 x_{t-i}
                  + gamma_m*(1/17)*sum_{i=6}^{22} x_{t-i}

    Produces ARMA(22,22) reduced form capturing long-memory volatility structure.

    From: Huang et al. (2015) Realized HAR GARCH, cited in DoubleAdapt

    Args:
        realized_var: daily realized variance (RV)
        daily_ret: daily returns (for GARCH component)

    Returns HAR-GARCH forecast and diagnostics.
    """
    import numpy as np
    from scipy.optimize import minimize

    rv = np.asarray(realized_var, dtype=float)
    ret = np.asarray(daily_ret, dtype=float)
    n = min(len(rv), len(ret))

    if n < lookback:
        lookback = max(n // 2, 22)

    rv = rv[-lookback:]
    ret = ret[-lookback:]
    x = np.sqrt(np.maximum(rv, 0))  # realized vol

    # Build HAR components
    daily_lag = np.roll(x, 1)
    daily_lag[0] = x[:2].mean()

    weekly_avg = np.array([np.mean(x[max(0, i-5):i]) for i in range(len(x))])
    weekly_lag = np.roll(weekly_avg, 1)
    weekly_lag[0] = weekly_avg[0]

    monthly_avg = np.array([np.mean(x[max(0, i-22):i]) for i in range(len(x))])
    monthly_lag = np.roll(monthly_avg, 1)
    monthly_lag[0] = monthly_avg[0]

    # OLS estimation (fast approximation to full HAR-GARCH QMLE)
    X = np.column_stack([np.ones(len(x)), daily_lag, weekly_lag, monthly_lag])
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(x)
    X_v, y_v = X[valid], x[valid]

    if len(y_v) < 30:
        return {'error': 'insufficient_valid_data'}

    coeffs, residuals, rank, _ = np.linalg.lstsq(X_v, y_v, rcond=None)

    omega, gamma_d, gamma_w, gamma_m = coeffs
    resid_std = np.std(residuals, ddof=len(coeffs))

    # GARCH(1,1) on residuals
    resid_sq = residuals**2
    def garch_ll(params):
        alpha0, alpha1, beta1 = np.exp(params) / (1 + np.exp(params))  # sigmoid bounds
        h = np.ones(len(resid_sq)) * alpha0 / (1 - alpha1 - beta1) if alpha1 + beta1 < 1 else np.ones(len(resid_sq)) * np.mean(resid_sq)
        for t in range(1, len(resid_sq)):
            h[t] = alpha0 + alpha1 * resid_sq[t-1] + beta1 * h[t-1]
        return -np.sum(-0.5 * np.log(2 * np.pi * h) - 0.5 * resid_sq / np.maximum(h, 1e-10))

    try:
        result = minimize(garch_ll, [np.log(0.05), np.log(0.1), np.log(0.85)], method='Nelder-Mead')
        alpha0, alpha1, beta1 = np.exp(result.x) / (1 + np.exp(result.x))
        garch_success = True
    except Exception:
        alpha0, alpha1, beta1 = np.mean(resid_sq) * 0.05, 0.1, 0.85
        garch_success = False

    # Latest HAR forecast
    latest_forecast = omega + gamma_d * x[-1] + gamma_w * weekly_avg[-1] + gamma_m * monthly_avg[-1]
    vol_forecast = max(0, latest_forecast)

    # R-squared
    fitted = X_v @ coeffs
    ss_res = np.sum((y_v - fitted)**2)
    ss_tot = np.sum((y_v - np.mean(y_v))**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)

    return {
        'har_omega': round(float(omega), 6),
        'har_gamma_daily': round(float(gamma_d), 4),
        'har_gamma_weekly': round(float(gamma_w), 4),
        'har_gamma_monthly': round(float(gamma_m), 4),
        'garch_omega': round(float(alpha0), 8),
        'garch_alpha': round(float(alpha1), 4),
        'garch_beta': round(float(beta1), 4),
        'garch_persistence': round(float(alpha1 + beta1), 4),
        'r_squared': round(float(r2), 4),
        'volatility_forecast': round(float(vol_forecast * 100), 2),
        'annualized_vol': round(float(vol_forecast * np.sqrt(252) * 100), 2),
        'long_memory_ratio': round(float(gamma_m / max(gamma_d, 1e-10)), 3),
        'garch_converged': garch_success,
        'dominant_component': 'monthly' if gamma_m > max(gamma_d, gamma_w) else
                              'weekly' if gamma_w > max(gamma_d, gamma_m) else 'daily',
        'advice': (f'长期记忆主导→趋势延续概率高' if gamma_m > max(gamma_d, gamma_w) else
                   f'短期波动主导→灵活应对'),
    }


def gamma_minimax_allocation(returns, epsilon=0.10, n_samples=2000):
    """
    Gamma-minimax (epsilon-contamination) robust portfolio allocation.

    Prior class: Gamma = {(1-epsilon) * pi_0 + epsilon * q : any q}
    Decision rule: minimize sup_{prior in Gamma} Bayes risk

    This is a "stress test" for the Bayesian allocation: it finds the worst-case
    perturbation within epsilon of the base prior and sizes accordingly.

    From: Statistical Decision Theory and Bayesian Analysis (Berger, Ch.4.7)

    Args:
        returns: (n_assets, n_periods) array of historical returns
        epsilon: contamination fraction (0 = standard Bayes, 1 = fully adversarial)
        n_samples: Monte Carlo samples

    Returns robust weights and diagnostics.
    """
    import numpy as np
    returns = np.asarray(returns, dtype=float)

    if returns.ndim == 1:
        returns = returns.reshape(1, -1)

    n_assets, n_periods = returns.shape
    if n_assets < 2 or n_periods < 10:
        return {'error': 'insufficient_data', 'n_assets': n_assets, 'n_periods': n_periods}

    # Base prior: empirical means and covariance
    mu_hat = returns.mean(axis=1)
    sigma_hat = np.cov(returns)

    # Bayesian (non-informative) posterior mean = MLE
    bayes_mu = mu_hat.copy()

    # Contamination: consider worst-case mean within epsilon-perturbed set
    # For each asset: mu in [mu_i - delta_i, mu_i + delta_i]
    # Where delta_i = z_{1-epsilon/2} * sigma_i / sqrt(n)
    from scipy.stats import norm as norm_dist
    z = norm_dist.ppf(1 - epsilon / 2)
    sigmas = np.sqrt(np.diag(sigma_hat))
    delta = z * sigmas / np.sqrt(n_periods)

    # Worst-case for a long-only investor: lower means
    worst_case_mu = bayes_mu - delta * (n_periods / max(n_periods - n_assets, 1))

    # Standard Bayesian optimal (max Sharpe)
    def solve_weights(mu_vec, sigma_mat):
        try:
            inv_sigma = np.linalg.inv(sigma_mat)
            w = inv_sigma @ mu_vec
            w = w / np.sum(np.abs(w))
            w = np.maximum(w, 0)
            return w / w.sum()
        except np.linalg.LinAlgError:
            return np.ones(n_assets) / n_assets

    bayes_w = solve_weights(bayes_mu, sigma_hat)
    robust_w = solve_weights(worst_case_mu, sigma_hat)

    # Compute expected returns under both
    bayes_ret = bayes_w @ bayes_mu
    robust_ret = robust_w @ bayes_mu
    bayes_risk = np.sqrt(bayes_w @ sigma_hat @ bayes_w)
    robust_risk = np.sqrt(robust_w @ sigma_hat @ robust_w)

    # Robustness penalty
    ret_penalty = (bayes_ret - robust_ret) / max(abs(bayes_ret), 1e-10)

    # Monte Carlo robustness check
    rng = np.random.RandomState(random_seed)
    mc_port_returns = []
    for _ in range(n_samples):
        # Sample from perturbed distribution
        perturbation = rng.uniform(-delta, delta)
        mc_mu = bayes_mu + epsilon * perturbation + (1 - epsilon) * 0
        mc_w = solve_weights(mc_mu, sigma_hat)
        mc_port_returns.append(mc_w @ bayes_mu)

    mc_port_returns = np.array(mc_port_returns)
    worst_case_ret = np.percentile(mc_port_returns, 5)

    return {
        'bayesian_weights': {f'asset_{i}': round(float(w), 4) for i, w in enumerate(bayes_w)},
        'robust_weights': {f'asset_{i}': round(float(w), 4) for i, w in enumerate(robust_w)},
        'bayesian_return': round(float(bayes_ret * 100), 3),
        'robust_return': round(float(robust_ret * 100), 3),
        'bayesian_risk': round(float(bayes_risk * 100), 3),
        'robust_risk': round(float(robust_risk * 100), 3),
        'return_penalty_pct': round(float(ret_penalty * 100), 2),
        'epsilon': epsilon,
        'worst_case_5pct': round(float(worst_case_ret * 100), 3),
        'hedge_ratio': round(float(np.sum(np.abs(robust_w - bayes_w)) / 2), 4),
        'advice': (f'需显著对冲(权重偏离{np.sum(np.abs(robust_w-bayes_w))/2:.0%})，epsilon={epsilon}'
                   if np.sum(np.abs(robust_w - bayes_w)) / 2 > 0.15 else
                   '先验稳健→标准贝叶斯分配可行'),
    }


def gp_volatility_surface(returns, n_prediction=10, kernel='rbf', random_seed=None):
    """
    Gaussian Process volatility surface estimation.

    Models volatility as a smooth function of time: sigma(t) ~ GP(0, K(t,t'))
    with RBF kernel: K(t,t') = tau^2 * exp(-|t-t'|^2 / 2l^2)

    Returns posterior mean and credible intervals for future volatility.

    From: Bayesian Data Analysis (Gelman, Ch.21)

    Args:
        returns: daily return series
        n_prediction: number of days to forecast
        kernel: 'rbf' or 'matern'

    Returns smoothed vol surface and forecasts with uncertainty.
    """
    import numpy as np
    ret = np.asarray(returns, dtype=float)
    n = len(ret)

    if n < 20:
        return {'error': 'insufficient_data', 'n': n}

    # Use realized vol (rolling 5-day) as observations
    vol_obs = np.array([np.std(ret[max(0, i-4):i+1], ddof=1) for i in range(n)])
    t = np.arange(n).reshape(-1, 1).astype(float)
    y = vol_obs.reshape(-1, 1)

    # Normalize
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / max(y_std, 1e-10)

    # GP kernel
    tau2 = 1.0  # signal variance
    l = n / 4.0  # length scale
    noise = 0.1  # observation noise

    if kernel == 'matern':
        def k(x1, x2):
            d = np.abs(x1 - x2.T)
            sqrt3 = np.sqrt(3)
            return tau2 * (1 + sqrt3 * d / l) * np.exp(-sqrt3 * d / l)
    else:
        def k(x1, x2):
            return tau2 * np.exp(-(x1 - x2.T)**2 / (2 * l**2))

    # Compute posterior
    K = k(t, t) + noise * np.eye(n)
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        K_inv = np.linalg.pinv(K)

    # Prediction points
    t_pred = np.arange(n, n + n_prediction).reshape(-1, 1).astype(float)
    K_star = k(t_pred, t)
    K_star_star = k(t_pred, t_pred)

    # Posterior mean and variance
    f_mean = K_star @ K_inv @ y_norm
    f_var = np.diag(K_star_star - K_star @ K_inv @ K_star.T)
    f_std = np.sqrt(np.maximum(f_var, 0))

    # Un-normalize
    f_mean_raw = f_mean.flatten() * y_std + y_mean
    f_std_raw = f_std * y_std

    # Smoothed historical fit
    f_hist = (K[:, :] @ K_inv @ y_norm).flatten() * y_std + y_mean

    # 95% credible intervals
    ci_lower = f_mean_raw - 1.96 * f_std_raw
    ci_upper = f_mean_raw + 1.96 * f_std_raw

    return {
        'current_vol_estimate': round(float(vol_obs[-1] * 100), 2),
        'forecast_vol_mean': [round(float(v * 100), 2) for v in f_mean_raw],
        'forecast_vol_std': [round(float(v * 100), 2) for v in f_std_raw],
        'forecast_ci_95_lower': [round(float(v * 100), 2) for v in ci_lower],
        'forecast_ci_95_upper': [round(float(v * 100), 2) for v in ci_upper],
        'length_scale_days': round(float(l), 1),
        'signal_variance': round(float(tau2), 3),
        'trend_direction': 'increasing' if f_mean_raw[-1] > vol_obs[-1] else
                           'decreasing' if f_mean_raw[-1] < vol_obs[-1] else 'stable',
        'smoothed_vol': [round(float(v * 100), 2) for v in f_hist[-n_prediction:]],
        'advice': (f'波动率预测{f_mean_raw[-1]*100:.1f}% (CI: [{ci_lower[-1]*100:.1f}, {ci_upper[-1]*100:.1f}]%)'),
    }


# ===================== 26. 高级贝叶斯技术 (from deep textbook analysis) =====================

def bayesian_logistic_regression(X, y, n_samples=2000, prior_scale=1.0, random_seed=None):
    """
    Bayesian logistic regression: p(y=1 | beta) = 1/(1+exp(-beta·X))

    With Normal priors on beta: beta_j ~ N(0, prior_scale^2).
    Returns full posterior over coefficients + predictive probabilities.

    Use: Binary outcome prediction (direction, crash, regime) with uncertainty quantification.

    From: Bayesian Methods for Hackers (Ch.2), Kruschke (Ch.21)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape if X.ndim > 1 else (len(X), 1)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if n < 10 or p > n:
        return {'error': 'insufficient_data', 'n': n, 'p': p}

    y = (y > np.median(y)).astype(float) if len(np.unique(y)) > 2 else y

    # Standardize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0, ddof=1)
    X_std[X_std < 1e-10] = 1.0
    X_scaled = (X - X_mean) / X_std

    # Add intercept
    X_design = np.column_stack([np.ones(n), X_scaled])

    # Metropolis-Hastings
    rng = np.random.RandomState(random_seed)
    beta_samples = np.zeros((n_samples, p + 1))
    beta = np.zeros(p + 1)
    beta[0] = np.log(np.mean(y) / max(1 - np.mean(y), 1e-10))
    current_ll = _logistic_log_likelihood(X_design, y, beta, prior_scale)

    accepted = 0
    for i in range(n_samples):
        beta_prop = beta + rng.normal(0, 0.1, p + 1)
        prop_ll = _logistic_log_likelihood(X_design, y, beta_prop, prior_scale)

        if np.log(rng.random()) < prop_ll - current_ll:
            beta = beta_prop
            current_ll = prop_ll
            accepted += 1

        beta_samples[i] = beta

    burn = n_samples // 4
    beta_post = beta_samples[burn:]

    # Posterior summaries
    beta_mean = beta_post.mean(axis=0)
    beta_std = beta_post.std(axis=0, ddof=1)
    beta_hdi = np.percentile(beta_post, [2.5, 97.5], axis=0)

    # Predictive probability
    logit = X_design @ beta_mean
    pred_prob = 1.0 / (1.0 + np.exp(-logit))

    # Significance: P(beta_j > 0 | data) for each coefficient
    prob_positive = (beta_post > 0).mean(axis=0)

    return {
        'coefficients': {
            'intercept': round(float(beta_mean[0]), 6),
            'features': [round(float(b), 6) for b in beta_mean[1:]],
        },
        'std_errors': [round(float(b), 6) for b in beta_std],
        'hdi_95': [{'low': round(float(beta_hdi[0, j]), 6), 'high': round(float(beta_hdi[1, j]), 6)}
                    for j in range(p + 1)],
        'prob_positive': [round(float(pp), 3) for pp in prob_positive],
        'predictive_prob': [round(float(pp), 4) for pp in pred_prob[-10:]],
        'acceptance_rate': round(float(accepted / n_samples), 3),
        'significant_features': [j for j in range(1, p + 1) if prob_positive[j] > 0.95 or prob_positive[j] < 0.05],
    }


def _logistic_log_likelihood(X, y, beta, prior_scale):
    """Log posterior for logistic regression (likelihood + normal prior)"""
    logit = X @ beta
    # Clip for numerical stability
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    ll = np.sum(y * np.log(np.maximum(p, 1e-15)) + (1 - y) * np.log(np.maximum(1 - p, 1e-15)))
    lp = -0.5 * np.sum(beta[1:]**2) / prior_scale**2
    return ll + lp


def limited_translation_shrinkage(estimates, prior_mean, std_errors, M=2.0):
    """
    Limited Translation Estimator (Efron-Morris).

    Shrinks estimates toward prior mean only when within M standard deviations.
    For extreme observations (e.g., crash returns), automatically reverts to raw data.

    delta(x) = x - (x - mu) * min{1, sqrt(M * sigma^2 / ((sigma^2 + tau^2)(x - mu)^2))}

    This provides crash-safe shrinkage — unlike James-Stein, extreme outliers are
    NOT pulled toward the mean, preserving fat-tail signals.

    From: Statistical Decision Theory (Berger, Ch.4.7.7), Efron-Morris
    """
    estimates = np.asarray(estimates, dtype=float)
    std_errors = np.asarray(std_errors, dtype=float)
    n = len(estimates)

    if n < 3:
        return {'error': 'need_at_least_3_estimates', 'n': n}

    sigma_sq = std_errors**2
    tau_sq = max(0, np.var(estimates, ddof=1) - np.mean(sigma_sq))
    total_var = sigma_sq + tau_sq

    # Distance in std units
    z_scores = np.abs(estimates - prior_mean) / np.sqrt(total_var)

    # Limited translation: only shrink when |z| < M
    shrink_factor = np.minimum(1.0, np.sqrt(M * sigma_sq / np.maximum(total_var * (estimates - prior_mean)**2, 1e-15)))
    shrink_factor[z_scores < M] = 1.0  # full shrinkage within M sigma

    limited_estimates = estimates - (estimates - prior_mean) * shrink_factor

    return {
        'original': [round(float(x), 4) for x in estimates],
        'limited_shrinkage': [round(float(x), 4) for x in limited_estimates],
        'shrinkage_factor': [round(float(s), 4) for s in shrink_factor],
        'z_scores': [round(float(z), 2) for z in z_scores],
        'M_parameter': M,
        'crash_protected': [bool(z > M) for z in z_scores],
        'tau_sq': round(float(tau_sq), 8),
    }


def epsilon_contamination_robustness(posterior_mean, posterior_se, prior_mean, prior_se,
                                      epsilon=0.10, direction='bull'):
    """
    Epsilon-contamination robustness bounds for Bayesian decisions.

    Posterior bounds under prior class Gamma = {(1-eps)*pi_0 + eps*q : any q}:

    Lower posterior prob = P_0 * [1 + eps * f(x|theta_hat) / ((1-eps) * m(x|pi_0))]^(-1)

    Directly quantifies how sensitive a trading signal is to prior misspecification.
    If nominal posterior is 0.95 but infimum is 0.55, the signal is NOT reliable.

    From: Statistical Decision Theory (Berger, Ch.4.7.4)

    Returns: nominal posterior, worst-case (infimum) posterior, fragility score.
    """
    # Normal approximation to posterior
    # P(theta > 0 | x) under base prior
    from scipy.stats import norm as norm_dist
    posterior_z = posterior_mean / max(posterior_se, 1e-10)
    nominal_prob = norm_dist.cdf(posterior_z if direction == 'bull' else -posterior_z)

    # Marginal likelihood ratio for worst-case
    # theta_hat = MLE maximizing f(x|theta) in opposite direction
    # For normal: f(x|theta_hat) / m(x|pi_0) ≈ exp(-0.5 * z^2) / sqrt(1 + prior_se^2/posterior_se^2)
    z_sq = posterior_z**2
    marginal_ratio = np.exp(-0.5 * z_sq) / np.sqrt(1 + min(prior_se**2 / max(posterior_se**2, 1e-10), 100))

    # Infimum posterior probability
    infimum_factor = 1.0 / (1.0 + epsilon * marginal_ratio / max((1.0 - epsilon), 0.01))
    infimum_prob = nominal_prob * infimum_factor

    # Maximum posterior probability
    supremum_factor = 1.0 / (1.0 - epsilon * marginal_ratio / max((1.0 - epsilon), 0.01))
    supremum_prob = min(1.0, nominal_prob * supremum_factor)

    fragility = nominal_prob - infimum_prob
    is_robust = fragility < 0.15

    return {
        'nominal_posterior': round(float(nominal_prob), 4),
        'infimum_posterior': round(float(infimum_prob), 4),
        'supremum_posterior': round(float(supremum_prob), 4),
        'fragility': round(float(fragility), 4),
        'is_robust': is_robust,
        'epsilon': epsilon,
        'robustness_verdict': ('稳健—后验判断对先验不敏感' if is_robust else
                               f'脆弱—后验在不利先验下可能降至{infimum_prob:.2f}（名义{nominal_prob:.2f}）'),
        'action_advice': ('信号可靠→可执行' if is_robust else
                          '信号脆弱→降低仓位或等待更多数据确认'),
    }


def lower_credible_bound_rank(successes_list, trials_list, prior_a=1, prior_b=1):
    """
    Rank items by the LOWER bound of posterior 95% credible interval.

    This conservatively ranks strategies/signals — items with fewer observations
    (higher uncertainty) get penalized via wider intervals.

    Fast approximation: mu = a/(a+b), se = 1.65 * sqrt(a*b/((a+b)^2*(a+b+1)))
    lower_bound = mu - se

    From: Bayesian Methods for Hackers (Ch.4, Reddit ranking)

    Args:
        successes_list: list of success counts per strategy
        trials_list: list of trial counts per strategy
        prior_a, prior_b: Beta prior hyperparameters (default Beta(1,1) = uniform)

    Returns ranked indices with posterior statistics.
    """
    n_strategies = len(successes_list)
    if n_strategies < 2:
        return {'error': 'need_at_least_2_strategies'}

    results = []
    for i in range(n_strategies):
        s = successes_list[i]
        t = trials_list[i]

        # Beta posterior: Beta(prior_a + s, prior_b + t - s)
        a_post = prior_a + s
        b_post = prior_b + max(0, t - s)

        # Posterior mean
        mu = a_post / (a_post + b_post)

        # Posterior standard deviation
        se = np.sqrt(a_post * b_post / ((a_post + b_post)**2 * (a_post + b_post + 1)))

        # 95% lower credible bound (1.65 ≈ z_0.95 for one-sided)
        lower_bound = mu - 1.65 * se

        # Also compute 95% central HDI bounds
        from scipy.stats import beta as beta_dist
        hdi_low = beta_dist.ppf(0.025, a_post, b_post)
        hdi_high = beta_dist.ppf(0.975, a_post, b_post)

        results.append({
            'index': i,
            'mean': round(float(mu), 4),
            'lower_bound': round(float(lower_bound), 4),
            'std_error': round(float(se), 4),
            'hdi_95': [round(float(hdi_low), 4), round(float(hdi_high), 4)],
            'successes': int(s),
            'trials': int(t),
            'n_effective': int(a_post + b_post),
        })

    # Rank by lower bound (descending)
    ranked = sorted(results, key=lambda x: x['lower_bound'], reverse=True)
    for rank, r in enumerate(ranked, 1):
        r['rank'] = rank

    best = ranked[0]
    return {
        'rankings': ranked,
        'best_strategy': best['index'],
        'best_lower_bound': best['lower_bound'],
        'ranking_method': '95% lower credible bound (conservative)',
    }


def trimmed_mean_performance(returns, trim_pct=0.05, window=60):
    """
    Outlier-resistant performance metrics using trimmed estimators.

    Computes trimmed-mean Sharpe ratio, trimmed Sortino ratio, and
    trimmed win-rate — more robust than standard estimators for
    non-normal financial returns.

    IQR-based outlier detection: Q1 - 1.5*IQR and Q3 + 1.5*IQR

    From: Ott & Longnecker (Ch.3), Statistical Methods
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < window:
        window = max(10, n)

    ret = returns[-window:]

    # Trimmed mean (remove extreme observations)
    n_trim = max(1, int(len(ret) * trim_pct))
    ret_sorted = np.sort(ret)
    ret_trimmed = ret_sorted[n_trim:-n_trim] if len(ret_sorted) > 2 * n_trim else ret

    trimmed_mean = np.mean(ret_trimmed)
    trimmed_std = np.std(ret_trimmed, ddof=1)

    # IQR-based outlier detection
    q1, q3 = np.percentile(ret, [25, 75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers = ret[(ret < lower_fence) | (ret > upper_fence)]

    # Trimmed Sharpe (annualized)
    if trimmed_std > 1e-10:
        trimmed_sharpe = trimmed_mean / trimmed_std * np.sqrt(252)
    else:
        trimmed_sharpe = 0.0

    # Trimmed Sortino
    downside = ret_trimmed[ret_trimmed < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else trimmed_std
    trimmed_sortino = trimmed_mean / max(downside_std, 1e-10) * np.sqrt(252)

    # Standard metrics for comparison
    standard_mean = np.mean(ret)
    standard_std = np.std(ret, ddof=1)
    standard_sharpe = standard_mean / max(standard_std, 1e-10) * np.sqrt(252)

    return {
        'trimmed_sharpe': round(float(trimmed_sharpe), 3),
        'standard_sharpe': round(float(standard_sharpe), 3),
        'trimmed_sortino': round(float(trimmed_sortino), 3),
        'trimmed_mean_daily': round(float(trimmed_mean * 100), 4),
        'standard_mean_daily': round(float(standard_mean * 100), 4),
        'trimmed_vol_daily': round(float(trimmed_std * 100), 4),
        'outlier_count': len(outliers),
        'outlier_pct': round(float(len(outliers) / n * 100), 1),
        'iqr_bounds': [round(float(lower_fence * 100), 4), round(float(upper_fence * 100), 4)],
        'sharpe_distortion': round(float(abs(trimmed_sharpe - standard_sharpe)), 3),
        'outlier_sensitivity': ('高敏感—原始Sharpe受异常值显著影响' if abs(trimmed_sharpe - standard_sharpe) > 0.5
                                else '中度敏感' if abs(trimmed_sharpe - standard_sharpe) > 0.2
                                else '稳健—异常值影响不大'),
    }


def morris_peb_full_variance(estimates, std_errors):
    """
    Morris Parametric Empirical Bayes with full variance adjustment.

    mu_i^EB = x_i - B * (x_i - x_bar)
    V_i^EB = sigma^2 * [1 - (p-3)/(p-1)*B] + (2/(p-3)) * B^2 * (x_i - x_bar)^2

    The second variance term accounts for uncertainty in estimating the
    shrinkage factor — critical for small cross-sections (5-15 assets).

    From: Statistical Decision Theory (Berger, Ch.4.5.2), Morris (1983)
    """
    estimates = np.asarray(estimates, dtype=float)
    std_errors = np.asarray(std_errors, dtype=float)
    p = len(estimates)

    if p < 4:
        return {'error': 'need_at_least_4_estimates', 'p': p}

    sigma_sq = np.mean(std_errors**2)
    x_bar = np.mean(estimates)
    s_sq = np.var(estimates, ddof=1)

    # James-Stein style shrinkage factor
    B = (p - 3) / (p - 1) * min(1.0, sigma_sq / max(s_sq, 1e-15))

    # Shrunk estimates
    eb_estimates = estimates - B * (estimates - x_bar)

    # Full variance: systematic + estimation uncertainty
    # V1: standard variance reduction
    v1 = sigma_sq * (1 - (p - 3) / (p - 1) * B)

    # V2: additional uncertainty from estimating shrinkage
    v2 = (2.0 / max(p - 3, 1)) * B**2 * (estimates - x_bar)**2

    full_variance = v1 + v2
    full_se = np.sqrt(np.maximum(full_variance, 0))

    # Effective degrees of freedom
    df_eff = p - 2 * B

    return {
        'original': [round(float(x), 4) for x in estimates],
        'eb_shrunk': [round(float(x), 4) for x in eb_estimates],
        'se_original': [round(float(s), 6) for s in std_errors],
        'se_full_variance': [round(float(s), 6) for s in full_se],
        'shrinkage_factor_B': round(float(B), 4),
        'df_effective': round(float(df_eff), 2),
        'variance_inflation': [round(float(v2_i / max(v1, 1e-15)), 2) for v2_i in v2],
        'most_shrunk_idx': int(np.argmax(np.abs(estimates - eb_estimates))),
        'shrinkage_pct': [round(float(abs(estimates[i] - eb_estimates[i]) / max(abs(estimates[i]), 1e-10) * 100), 1)
                          for i in range(p)],
    }


# ===================== 27. BDA3高级技术 (from deep chapter analysis) =====================

def half_cauchy_prior_variance(group_estimates, scale_A=1.0, n_grid=500):
    """
    Half-Cauchy prior for hierarchical variance parameters.

    p(tau) ∝ 1 / (1 + (tau/A)^2), tau > 0

    Unlike Inverse-Gamma(eps, eps), the Half-Cauchy keeps mass at zero but
    has heavy enough tails to let the data dominate when signal is strong.
    Avoids the bias problem where IG priors pull small variances toward zero.

    From: BDA3 (Gelman, Ch.5.7)

    Returns: posterior mean/mode for tau, shrinkage calibration diagnostics.
    """
    group_estimates = np.asarray(group_estimates, dtype=float)
    J = len(group_estimates)

    if J < 3:
        return {'error': 'need_at_least_3_groups', 'J': J}

    overall_mean = np.mean(group_estimates)
    s_sq = np.var(group_estimates, ddof=1)
    sigma_sq = np.mean([np.var(group_estimates) / J])  # rough estimate

    # Grid approximation for tau posterior
    tau_grid = np.linspace(0.001, max(5 * np.sqrt(max(s_sq, 1e-6)), 5.0), n_grid)

    log_prior = -np.log(1 + (tau_grid / scale_A)**2)
    # Approximate marginal likelihood: y_j ~ N(mu, sigma^2 + tau^2)
    log_lik = np.zeros(n_grid)
    for j, tau in enumerate(tau_grid):
        var_total = sigma_sq + tau**2
        log_lik[j] = -0.5 * J * np.log(2 * np.pi * var_total) - 0.5 * np.sum((group_estimates - overall_mean)**2) / var_total

    log_post = log_prior + log_lik
    log_post -= np.max(log_post)  # stabilize
    post = np.exp(log_post)
    post /= post.sum()

    tau_map = tau_grid[np.argmax(log_post)]
    tau_mean = np.sum(tau_grid * post)
    tau_sd = np.sqrt(np.sum(tau_grid**2 * post) - tau_mean**2)

    # Shrinkage factor: B = sigma^2 / (sigma^2 + tau^2)
    B_hc = sigma_sq / max(sigma_sq + tau_map**2, 1e-15)

    return {
        'tau_map': round(float(tau_map), 6),
        'tau_posterior_mean': round(float(tau_mean), 6),
        'tau_posterior_sd': round(float(tau_sd), 6),
        'shrinkage_factor': round(float(B_hc), 4),
        'scale_A': scale_A,
        'J_groups': J,
        'squared_error': round(float(s_sq), 8),
        'prior_type': 'Half-Cauchy (robust to small variances)',
        'comparison': ('强收缩→组间差异小' if B_hc < 0.3 else
                       '中等收缩→适度池化' if B_hc < 0.7 else
                       '弱收缩→组间差异大，不应过度池化'),
    }


def compute_dic(log_likelihoods, posterior_samples=None, method='mean'):
    """
    DIC (Deviance Information Criterion) for trading model comparison.

    p_DIC = 2(log p(y|theta_bayes) - E_post[log p(y|theta)])
    DIC = -2 * D_bar + 2 * p_DIC

    Simpler than WAIC and works for dependent data (e.g., time series returns)
    where per-point WAIC partitioning is problematic.

    From: BDA3 (Gelman, Ch.7.2)

    Args:
        log_likelihoods: array of shape (n_samples, n_observations) or (n_samples,)
        posterior_samples: optional parameter samples
        method: 'mean' = average over posterior, 'max' = at posterior mode

    Returns DIC, effective parameters, comparison with WAIC if possible.
    """
    log_lik = np.asarray(log_likelihoods, dtype=float)

    if log_lik.ndim == 1:
        log_lik = log_lik.reshape(-1, 1)

    n_samples, n_obs = log_lik.shape

    # D_bar = -2 * E_post[log p(y|theta)]
    D_bar = -2 * np.mean(np.sum(log_lik, axis=1))

    if method == 'mean':
        # D_theta_bayes = -2 * log p(y|E[theta])
        mean_ll = np.mean(log_lik, axis=0)
        D_theta_bayes = -2 * np.sum(mean_ll)
    else:
        # D_theta_bayes = -2 * max_theta log p(y|theta)
        max_idx = np.argmax(np.sum(log_lik, axis=1))
        D_theta_bayes = -2 * np.sum(log_lik[max_idx])

    p_DIC = D_bar - D_theta_bayes
    dic = D_bar + p_DIC  # = D_bar + (D_bar - D_theta_bayes)

    return {
        'DIC': round(float(dic), 2),
        'D_bar': round(float(D_bar), 2),
        'p_DIC': round(float(p_DIC), 2),
        'method': method,
        'n_samples': n_samples,
        'n_observations': n_obs,
        'interpretation': ('模型复杂度=' + str(round(float(p_DIC), 1)) + '个有效参数'),
    }


def box_cox_transform(returns, phi_range=(-1.0, 2.0), n_grid=50):
    """
    Box-Cox power transformation for non-normal returns.

    y^(phi) = (y^phi - 1) / phi  for phi != 0
    y^(phi) = log(y)              for phi = 0

    Finds optimal phi by maximizing log-likelihood of transformed normality.
    Essential when Gaussian models (Kalman, linear regression) are applied
    to heavy-tailed or skewed financial returns.

    From: BDA3 (Gelman, Ch.7.5-7.6)

    Returns: optimal phi, transformed data, normality test.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)

    if n < 20:
        return {'error': 'insufficient_data', 'n': n}

    # Shift to positive values
    min_ret = returns.min()
    shift = max(0, -min_ret + 1e-6)
    y = returns + shift

    phi_values = np.linspace(phi_range[0], phi_range[1], n_grid)
    best_phi = 1.0
    best_ll = -float('inf')

    for phi in phi_values:
        if abs(phi) < 1e-10:
            y_trans = np.log(y)
        else:
            y_trans = (y**phi - 1) / phi

        mu_ml = np.mean(y_trans)
        sigma_ml = np.std(y_trans, ddof=1)
        if sigma_ml < 1e-10:
            continue

        # Log-likelihood under normality
        ll = -n * np.log(sigma_ml) - 0.5 * np.sum((y_trans - mu_ml)**2) / sigma_ml**2
        # Jacobian adjustment: (phi - 1) * sum(log(y))
        ll += (phi - 1) * np.sum(np.log(y))

        if ll > best_ll:
            best_ll = ll
            best_phi = phi

    # Transform with best phi
    if abs(best_phi) < 1e-10:
        transformed = np.log(y)
    else:
        transformed = (y**best_phi - 1) / best_phi

    # Normality test on transformed data
    from scipy.stats import shapiro, jarque_bera
    shapiro_stat, shapiro_p = shapiro(transformed[:min(5000, n)])
    jb_stat, jb_p = jarque_bera(transformed)

    skewness = np.mean((transformed - np.mean(transformed))**3) / np.std(transformed, ddof=1)**3
    kurtosis = np.mean((transformed - np.mean(transformed))**4) / np.std(transformed, ddof=1)**4

    return {
        'optimal_phi': round(float(best_phi), 4),
        'shift_applied': round(float(shift), 6),
        'shapiro_p': round(float(shapiro_p), 4),
        'jarque_bera_p': round(float(jb_p), 4),
        'skewness_original': round(float(np.mean((returns - np.mean(returns))**3) / np.std(returns, ddof=1)**3), 3),
        'skewness_transformed': round(float(skewness), 3),
        'kurtosis_transformed': round(float(kurtosis), 3),
        'is_normal': shapiro_p > 0.05 and jb_p > 0.05,
        'transformed_data': transformed.tolist() if n <= 100 else None,
        'advice': ('phi≈1→已接近正态' if abs(best_phi - 1) < 0.15 else
                   'phi≈0→对数变换最优' if abs(best_phi) < 0.15 else
                   f'phi={best_phi:.2f}→建议使用Box-Cox变换'),
    }


def cpo_outlier_detection(log_lik_matrix):
    """
    Conditional Predictive Ordinate (CPO) for outlier/novelty detection.

    CPO_i = p(y_i | y_{-i}) — leave-one-out predictive density.

    Low CPO identifies observations poorly explained by the model:
    - Flash crashes, data errors, regime outliers
    - Surprising returns that "shouldn't happen" under the model

    From: BDA3 (Gelman, Ch.6.3)

    Args:
        log_lik_matrix: (n_samples, n_observations) matrix of log-likelihoods

    Returns CPO values, outlier flags, and model criticism diagnostics.
    """
    log_lik = np.asarray(log_lik_matrix, dtype=float)
    n_samples, n_obs = log_lik.shape

    if n_obs < 5:
        return {'error': 'insufficient_observations', 'n_obs': n_obs}

    # CPO via harmonic mean: CPO_i^(-1) = mean(1 / p(y_i|theta_s))
    # log(CPO_i) = -log(mean(exp(-log p_i)))
    # More stable: log(CPO_i) = -log_sum_exp(-log p_i - log S) / S → not quite
    # Use log-sum-exp trick for stability
    neg_log_lik = -log_lik
    max_neg_ll = neg_log_lik.max(axis=0)
    shifted = neg_log_lik - max_neg_ll
    log_mean_inv = max_neg_ll + np.log(np.mean(np.exp(shifted), axis=0))
    log_cpo = -log_mean_inv

    # Pseudo log-marginal-likelihood
    lpml = np.sum(log_cpo)

    # Identify outliers: CPO below threshold
    cpo_values = np.exp(log_cpo)
    cpo_median = np.median(cpo_values)
    cpo_threshold = cpo_median * 0.01
    outlier_idx = np.where(cpo_values < cpo_threshold)[0]
    outlier_severity = np.where(cpo_values < cpo_threshold,
                                 1 - cpo_values / max(cpo_median, 1e-10), 0)

    return {
        'lpml': round(float(lpml), 2),
        'median_cpo': round(float(cpo_median), 8),
        'n_outliers': len(outlier_idx),
        'outlier_indices': outlier_idx.tolist(),
        'outlier_severity_max': round(float(np.max(outlier_severity)), 4) if len(outlier_idx) > 0 else 0.0,
        'outlier_pct': round(float(len(outlier_idx) / n_obs * 100), 1),
        'model_fit_assessment': ('良好—无显著离群点' if len(outlier_idx) < n_obs * 0.03 else
                                 '注意—存在离群点需调查' if len(outlier_idx) < n_obs * 0.10 else
                                 '差—模型未充分描述数据尾部'),
    }


# ===================== 34. 日亏损限额 =====================

def daily_loss_limit_check(trade_results_today, max_daily_loss_pct=0.03,
                           rolling_avg_daily_profit=0.0):
    """
    单日最大亏损限额检测。

    规则（SpeedTrade体系）:
    - 当日累计亏损 > 账户×max_daily_loss_pct → 熔断，停止交易
    - 当日亏损 > 滚动平均日盈利 → 警告，减半仓位
    - 连续亏损笔数 > 3 → 冷却期

    Args:
        trade_results_today: list of dicts [{pnl_pct, symbol, time}, ...]
        max_daily_loss_pct: 单日最大亏损比例
        rolling_avg_daily_profit: 滚动20日平均日盈利（%账户）

    Returns:
        dict with halt_trading, reduce_position, cooldown_minutes, daily_pnl_pct
    """
    if not trade_results_today:
        return {
            'halt_trading': False, 'reduce_position': False,
            'cooldown_minutes': 0, 'daily_pnl_pct': 0.0,
            'consecutive_losses': 0, 'status': '正常交易',
        }

    daily_pnl_pct = sum(r.get('pnl_pct', 0) for r in trade_results_today)
    abs_daily_pnl = abs(daily_pnl_pct)

    # 连续亏损计数
    consecutive = 0
    for r in reversed(trade_results_today):
        if r.get('pnl_pct', 0) < 0:
            consecutive += 1
        else:
            break

    halt_trading = False
    reduce_position = False
    cooldown_minutes = 0
    status = '正常交易'

    # 硬熔断:单日亏损超限
    if abs_daily_pnl > max_daily_loss_pct * 100:  # pnl_pct是百分比
        halt_trading = True
        status = f'熔断:日亏损{abs_daily_pnl:.1f}%超过上限{max_daily_loss_pct*100:.0f}%'

    # 速度限制:日亏损超过滚动平均日盈利
    if rolling_avg_daily_profit > 0 and daily_pnl_pct < -rolling_avg_daily_profit:
        reduce_position = True
        if not halt_trading:
            status = f'减速:日亏损超过滚动日均盈利'

    # 连续亏损冷却
    if consecutive >= 3:
        cooldown_minutes = 30 if consecutive >= 4 else 15
        reduce_position = True
        if not halt_trading:
            status = f'冷却:连续{consecutive}笔亏损'

    return {
        'halt_trading': halt_trading,
        'reduce_position': reduce_position,
        'cooldown_minutes': cooldown_minutes,
        'daily_pnl_pct': round(daily_pnl_pct, 2),
        'consecutive_losses': consecutive,
        'status': status,
    }


# ===================== 35. 组合层面风险管理 =====================

def portfolio_risk_assessment(positions, returns_matrix=None):
    """
    组合层面风险评估（A股T+1做多环境）。

    A股环境下bearish=减仓/观望而非做空，因此净敞口仅计bullish仓位。
    相关性通过行业分组+beta代理估计（无returns_matrix时）。

    Args:
        positions: list of dicts [{symbol, direction, weight, sector, beta, atr_pct}, ...]
        returns_matrix: DataFrame of historical returns per symbol (optional, for VaR/corr)

    Returns:
        dict with portfolio risk metrics and warnings
    """
    n_positions = len(positions)
    if n_positions == 0:
        return {
            'n_positions': 0, 'total_exposure_pct': 0.0,
            'net_exposure_pct': 0.0, 'max_sector_pct': 0.0,
            'correlation_risk': '无持仓', 'concentration_warning': False,
            'var_95_pct': 0.0, 'cvar_95_pct': 0.0,
            'diversification_ratio': 0.0, 'stress_loss_5pct': 0.0,
            'risk_contribution': [], 'warnings': [],
        }

    weights_arr = np.array([abs(p.get('weight', 0)) for p in positions])
    total_weight = float(np.sum(weights_arr))
    warnings = []

    # ---- 1. 总敞口（A股:仅bullish计入多头敞口） ----
    bullish_weight = sum(p.get('weight', 0) for p in positions if p.get('direction') == 'bullish')
    bearish_count = sum(1 for p in positions if p.get('direction') == 'bearish')

    if total_weight > 0.90:
        warnings.append(f'总敞口{total_weight*100:.0f}%过高（建议≤80%），无现金缓冲')
    elif total_weight > 0.75:
        warnings.append(f'总敞口{total_weight*100:.0f}%偏高（建议预留≥20%现金）')
    elif total_weight < 0.20 and n_positions >= 3:
        warnings.append(f'总敞口仅{total_weight*100:.0f}%，仓位过轻可能踏空')

    # ---- 2. 行业集中度（HHI + 最大行业） ----
    sector_weights = {}
    for p in positions:
        sector = p.get('sector', '未知')
        sector_weights[sector] = sector_weights.get(sector, 0) + abs(p.get('weight', 0))
    max_sector_name = max(sector_weights, key=sector_weights.get) if sector_weights else '未知'
    max_sector = max(sector_weights.values()) if sector_weights else 0

    # 行业HHI
    sector_total = sum(sector_weights.values()) or 1.0
    sector_hhi = sum((w / sector_total) ** 2 for w in sector_weights.values())
    n_effective_sectors = 1.0 / max(sector_hhi, 0.01)

    if max_sector > 0.50:
        warnings.append(f'行业集中度危险（{max_sector*100:.0f}% in {max_sector_name}），单一行业>50%')
    elif max_sector > 0.35:
        warnings.append(f'行业集中度偏高（{max_sector*100:.0f}% in {max_sector_name}），建议≤35%')
    if n_effective_sectors < 2.0 and n_positions >= 3:
        warnings.append(f'有效行业数仅{n_effective_sectors:.1f}个，行业分散严重不足')

    # ---- 3. 牛熊比（A股:bearish=回避信号） ----
    bullish_n = sum(1 for p in positions if p.get('direction') == 'bullish')
    bearish_n = sum(1 for p in positions if p.get('direction') == 'bearish')
    bull_bear_ratio = bullish_n / max(bearish_n, 1)

    if bearish_n > bullish_n and total_weight > 0.40:
        warnings.append(f'看空信号占多数（牛{bullish_n}/熊{bearish_n}），但总敞口仍{total_weight*100:.0f}%—建议减仓')
    if bull_bear_ratio > 4 and n_positions >= 5:
        pass  # 一致性看多，不报警

    # ---- 4. 相关性估计（行业代理 + beta） ----
    corr_risk = 'unknown'
    mean_corr_est = 0.0
    if returns_matrix is not None and returns_matrix.shape[1] >= 2:
        corr_matrix = returns_matrix.corr()
        mean_corr_est = float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean())
    else:
        # 代理估计:同行业→高相关(0.7)，不同行业→中等(0.35)，beta差异调整
        sectors_list = [p.get('sector', '未知') for p in positions]
        betas = np.array([p.get('beta', 1.0) for p in positions])
        n = len(sectors_list)
        if n >= 2:
            proxy_corrs = []
            for i in range(n):
                for j in range(i + 1, n):
                    base_corr = 0.70 if sectors_list[i] == sectors_list[j] else 0.35
                    beta_adj = 1.0 - 0.15 * abs(betas[i] - betas[j])
                    proxy_corrs.append(base_corr * beta_adj)
            mean_corr_est = float(np.mean(proxy_corrs)) if proxy_corrs else 0.0

    if mean_corr_est > 0.70:
        corr_risk = '高相关风险—分散化失效'
        warnings.append(f'估计平均相关性{mean_corr_est:.2f}过高，分散化效果差')
    elif mean_corr_est > 0.45:
        corr_risk = '中等相关'
    else:
        corr_risk = '低相关—分散化有效'

    # ---- 5. 分散化比率（有效持仓数） ----
    herfindahl = float(np.sum(weights_arr ** 2)) if total_weight > 0 else 1.0
    diversification_ratio = 1.0 / max(herfindahl, 0.01)
    n_effective = diversification_ratio
    if n_effective < 2.0 and n_positions >= 4:
        warnings.append(f'有效持仓数仅{n_effective:.1f}只（共{n_positions}只），权重过度集中')

    # ---- 6. VaR / CVaR ----
    var_95 = 0.0
    cvar_95 = 0.0
    if returns_matrix is not None and returns_matrix.shape[1] >= n_positions:
        portf_returns = returns_matrix.mean(axis=1) * total_weight
        portf_returns_clean = portf_returns.dropna()
        var_95 = abs(float(np.percentile(portf_returns_clean, 5)))
        tail = portf_returns_clean[portf_returns_clean <= -var_95]
        cvar_95 = abs(float(tail.mean())) if len(tail) > 0 else var_95 * 1.4
    else:
        # 参数法近似:假设正态，95% VaR ≈ 1.645 * σ_portfolio
        atr_pcts = np.array([p.get('atr_pct', 2.0) / 100 for p in positions])
        # 组合波动率（考虑相关性）
        if n_positions >= 2:
            cov_proxy = mean_corr_est * np.outer(atr_pcts, atr_pcts) + \
                        (1 - mean_corr_est) * np.diag(atr_pcts ** 2)
            weights_col = weights_arr.reshape(-1, 1)
            portf_var = float((weights_col.T @ cov_proxy @ weights_col).item())
            portf_vol = np.sqrt(max(portf_var, 1e-10))
        else:
            portf_vol = float(np.sum(weights_arr * atr_pcts))
        var_95 = 1.645 * portf_vol
        cvar_95 = var_95 * 1.45  # 肥尾修正

    if var_95 > 0.05:
        warnings.append(f'日VaR(95%)={var_95*100:.1f}%偏高，单日潜在亏损超5%')

    # ---- 7. 压力测试 ----
    stress_loss_5pct = 0.0
    stress_loss_10pct = 0.0
    for p in positions:
        beta = p.get('beta', 1.0)
        w = abs(p.get('weight', 0))
        stress_loss_5pct += w * beta * 0.05
        stress_loss_10pct += w * beta * 0.10
    if stress_loss_5pct > 0.04:
        warnings.append(f'市场下跌5%时估计亏损{stress_loss_5pct*100:.1f}%（超过4%警戒线）')
    if stress_loss_10pct > 0.08:
        warnings.append(f'市场下跌10%时估计亏损{stress_loss_10pct*100:.1f}%（超过8%警戒线）')

    # ---- 8. 风险贡献（各持仓对组合VaR的边际贡献） ----
    risk_contributions = []
    for i, p in enumerate(positions):
        w_i = weights_arr[i]
        beta_i = p.get('beta', 1.0)
        atr_i = p.get('atr_pct', 2.0) / 100
        # 边际风险 ≈ w_i * beta_i * σ_market + w_i * idio_vol
        rc_pct = w_i * (beta_i * 0.18 + atr_i * 0.5) / max(total_weight * 0.18, 0.001)
        risk_contributions.append({
            'symbol': p.get('symbol', ''),
            'risk_pct': round(rc_pct * 100, 1),
            'concentration_flag': rc_pct > 0.40,
        })
        if rc_pct > 0.50:
            warnings.append(f"{p.get('symbol','?')}风险贡献{rc_pct*100:.0f}%（过度集中）")

    return {
        'n_positions': n_positions,
        'total_exposure_pct': round(total_weight * 100, 1),
        'net_exposure_pct': round(bullish_weight * 100, 1),  # A股:仅计多头
        'max_sector_pct': round(max_sector * 100, 1),
        'max_sector_name': max_sector_name,
        'n_effective_sectors': round(n_effective_sectors, 1),
        'correlation_risk': corr_risk,
        'mean_correlation': round(mean_corr_est, 3),
        'concentration_warning': max_sector > 0.35,
        'var_95_pct': round(var_95 * 100, 2),
        'cvar_95_pct': round(cvar_95 * 100, 2),
        'diversification_ratio': round(float(diversification_ratio), 1),
        'stress_loss_5pct': round(stress_loss_5pct * 100, 1),
        'stress_loss_10pct': round(stress_loss_10pct * 100, 1),
        'bull_bear_ratio': round(bull_bear_ratio, 1),
        'risk_contribution': risk_contributions,
        'warnings': warnings,
    }


# ===================== 36. 缺口风险评估 =====================

def gap_risk_assessment(daily_df, current_position=None):
    """
    隔夜/周末缺口风险评估。

    Args:
        daily_df: DataFrame with OHLCV daily data (≥60 bars)
        current_position: dict {direction, entry_price, shares} or None

    Returns:
        dict with gap risk metrics
    """
    if daily_df is None or len(daily_df) < 30:
        return {
            'gap_risk_level': 'unknown', 'avg_gap_pct': 0.0,
            'max_gap_pct': 0.0, 'weekend_gap_multiplier': 0.0,
            'limit_down_risk': False, 'position_reduce_factor': 1.0,
        }

    close = daily_df['close'].values
    opens = daily_df['open'].values
    n = len(close)

    # 隔夜缺口 = |今开 - 昨收| / 昨收
    gaps = np.abs(opens[1:] - close[:-1]) / close[:-1]
    avg_gap = float(np.mean(gaps))
    max_gap = float(np.max(gaps))

    # 周末/周一缺口倍数
    monday_gaps = []
    weekday_gaps = []
    if hasattr(daily_df.index, 'weekday'):
        for i in range(1, n):
            prev_day = daily_df.index[i - 1]
            curr_day = daily_df.index[i]
            gap = gaps[i - 1]
            if hasattr(prev_day, 'weekday') and prev_day.weekday() == 4:  # 周五→周一
                monday_gaps.append(gap)
            elif hasattr(curr_day, 'weekday') and curr_day.weekday() != 0:
                weekday_gaps.append(gap)

    avg_monday_gap = float(np.mean(monday_gaps)) if monday_gaps else avg_gap * 1.5
    avg_weekday_gap = float(np.mean(weekday_gaps)) if weekday_gaps else avg_gap
    weekend_multiplier = avg_monday_gap / max(avg_weekday_gap, 0.001)

    # 跌停风险评估（A股±10%）
    limit_down_count = int(np.sum(close[1:] <= close[:-1] * 0.91))
    limit_down_risk = limit_down_count > 2  # 60日内>2次跌停

    # 仓位削减因子
    reduce_factor = 1.0
    gap_risk_level = 'low'
    if avg_gap > 0.03 or max_gap > 0.08:
        gap_risk_level = 'high'
        reduce_factor = 0.5
    elif avg_gap > 0.02 or max_gap > 0.05:
        gap_risk_level = 'medium'
        reduce_factor = 0.75

    if weekend_multiplier > 2.5:
        reduce_factor *= 0.7
        gap_risk_level = 'high' if gap_risk_level == 'medium' else gap_risk_level

    if limit_down_risk:
        reduce_factor *= 0.5
        gap_risk_level = 'critical'

    # 如果是持仓过周末的建议
    position_reduce_factor = reduce_factor
    if current_position:
        direction = current_position.get('direction', 'bullish')
        if direction == 'bearish':
            # 空头过周末:涨停缺口风险
            limit_up_count = int(np.sum(close[1:] >= close[:-1] * 1.09))
            if limit_up_count > 2:
                position_reduce_factor *= 0.5

    return {
        'gap_risk_level': gap_risk_level,
        'avg_gap_pct': round(avg_gap * 100, 2),
        'max_gap_pct': round(max_gap * 100, 2),
        'weekend_gap_multiplier': round(float(weekend_multiplier), 2),
        'limit_down_risk': limit_down_risk,
        'position_reduce_factor': round(position_reduce_factor, 2),
        'limit_down_count_60d': limit_down_count,
        'pre_weekend_advisory': '减仓50%过周末' if gap_risk_level in ('high', 'critical') else
                               '减仓25%过周末' if gap_risk_level == 'medium' else
                               '正常持仓过周末',
    }