# reversal_detector.py
import json
import requests
import talib
import numpy as np
from advanced_indicators import td_sequential, chanlun_analysis, dynamic_support_resistance, candlestick_patterns
from config import DEEPSEEK_API_KEY

def _compute_price_info(direction, current_price, dyn_sup, dyn_res):
    """根据方向、当前价和动态支撑阻力，计算入场/止损/目标价"""
    entry = f"{current_price:.2f}"
    if direction == 'bullish':
        nearest_sup = dyn_sup[0][0] if dyn_sup else current_price * 0.97
        nearest_res = dyn_res[0][0] if dyn_res else current_price * 1.05
        next_res = dyn_res[1][0] if len(dyn_res) > 1 else round(nearest_res * 1.03, 2)
        stop_price = round(nearest_sup * 0.98, 2)
        targets = [f"第一目标 {nearest_res:.2f}（最近阻力）", f"第二目标 {next_res:.2f}（次阻力）"]
    else:
        nearest_res = dyn_res[0][0] if dyn_res else current_price * 1.03
        nearest_sup = dyn_sup[0][0] if dyn_sup else current_price * 0.97
        next_sup = dyn_sup[1][0] if len(dyn_sup) > 1 else round(nearest_sup * 0.98, 2)
        stop_price = round(nearest_res * 1.02, 2)
        targets = [f"第一目标 {nearest_sup:.2f}（最近支撑）", f"第二目标 {next_sup:.2f}（次支撑）"]
    return {
        'entry_zone': entry,
        'stop_loss': str(stop_price),
        'targets': targets,
        'position_advice': '反转信号，轻仓试探10%-15%',
        'weekly_align': '未计算',
        'source': '本地反转检测',
        'confidence': 0.55
    }


def _score_pattern(pattern):
    """为检测到的反转形态打分，用于选择最强信号"""
    type_scores = {
        'TD13完成': 0.85,
        '缠论买点': 0.80,
        '缠论卖点': 0.80,
        '跌破支撑': 0.65,
        '突破阻力': 0.65,
        '顶背离(CCI)': 0.60,
        '底背离(CCI)': 0.60,
        '量价顶背离': 0.55,
        '量价底背离': 0.55,
        '形态反转-看跌': 0.50,
        '形态反转-看涨': 0.50,
    }
    base = type_scores.get(pattern.get('type', ''), 0.4)
    return base


def check_reversal(df_5min, df_daily, symbol):
    """
    本地反转检测（增加动量背离），每种检测均附带入场/止损/目标价。
    收集所有检测到的反转形态，返回得分最高的信号。
    CCI背离需持续2根以上K线才触发；成交量基准使用20周期中位数。
    """
    if df_5min is None or df_5min.empty or df_daily is None or df_daily.empty:
        return None

    current_price = float(df_5min['close'].iloc[-1])
    dyn_sup, dyn_res = dynamic_support_resistance(df_daily)

    patterns = []  # 收集所有检测到的形态

    # 1. TD13完成
    td = td_sequential(df_daily)
    if td.get('completed'):
        direction = 'bearish' if td['direction'] == 'sell' else 'bullish'
        result = {
            'type': 'TD13完成',
            'direction': direction,
            'detail': f"TD13结构完成，方向：{td['direction']}",
            'symbol': symbol
        }
        result.update(_compute_price_info(direction, current_price, dyn_sup, dyn_res))
        patterns.append(result)

    # 2. 缠论买卖点
    chan = chanlun_analysis(df_daily)
    if chan.get('buy_points'):
        result = {
            'type': '缠论买点', 'direction': 'bullish',
            'detail': f"缠论买点：{chan['buy_points']}", 'symbol': symbol
        }
        result.update(_compute_price_info('bullish', current_price, dyn_sup, dyn_res))
        patterns.append(result)
    if chan.get('sell_points'):
        result = {
            'type': '缠论卖点', 'direction': 'bearish',
            'detail': f"缠论卖点：{chan['sell_points']}", 'symbol': symbol
        }
        result.update(_compute_price_info('bearish', current_price, dyn_sup, dyn_res))
        patterns.append(result)

    # 3. 关键位突破
    for price, strength in dyn_sup:
        if current_price < price and strength >= 50:
            result = {
                'type': '跌破支撑', 'direction': 'bearish',
                'detail': f"价格{current_price:.2f}跌破支撑{price:.2f}(强度{strength})",
                'symbol': symbol
            }
            result.update(_compute_price_info('bearish', current_price, dyn_sup, dyn_res))
            patterns.append(result)
            break  # 只取最显著的一个
    for price, strength in dyn_res:
        if current_price > price and strength >= 50:
            result = {
                'type': '突破阻力', 'direction': 'bullish',
                'detail': f"价格{current_price:.2f}突破阻力{price:.2f}(强度{strength})",
                'symbol': symbol
            }
            result.update(_compute_price_info('bullish', current_price, dyn_sup, dyn_res))
            patterns.append(result)
            break  # 只取最显著的一个

    # 4. 动量背离（CCI）- 需持续2根以上K线
    if len(df_daily) >= 30:
        high_d = df_daily['high'].astype(float).values
        low_d = df_daily['low'].astype(float).values
        close_d = df_daily['close'].astype(float).values

        cci10 = talib.CCI(high_d, low_d, close_d, timeperiod=10)

        # --- 顶背离(CCI): 价格创新高但CCI未跟随, 需持续2+ bar ---
        recent_10_high = high_d[-10:]
        recent_10_cci = cci10[-10:]
        max_price_idx = np.argmax(recent_10_high)
        max_cci_idx = np.argmax(recent_10_cci)
        if max_price_idx != max_cci_idx and recent_10_cci[max_price_idx] < recent_10_cci[max_cci_idx] - 30:
            # 检查持久性：价格在近期高位附近且CCI低于峰值的bar数
            persistent_count = 0
            price_peak = recent_10_high[max_price_idx]
            cci_peak = recent_10_cci[max_cci_idx]
            for i in range(max(0, len(recent_10_high) - 5), len(recent_10_high)):
                if recent_10_high[i] >= price_peak * 0.99 and recent_10_cci[i] < cci_peak - 20:
                    persistent_count += 1
            if persistent_count >= 2:
                result = {
                    'type': '顶背离(CCI)', 'direction': 'bearish',
                    'detail': f'价格创新高但CCI10未跟随(持续{persistent_count}bar)，可能见顶', 'symbol': symbol
                }
                result.update(_compute_price_info('bearish', current_price, dyn_sup, dyn_res))
                patterns.append(result)

        # --- 底背离(CCI): 价格创新低但CCI未跟随, 需持续2+ bar ---
        recent_10_low = low_d[-10:]
        min_price_idx = np.argmin(recent_10_low)
        min_cci_idx = np.argmin(recent_10_cci)
        if min_price_idx != min_cci_idx and recent_10_cci[min_price_idx] > recent_10_cci[min_cci_idx] + 30:
            persistent_count = 0
            price_trough = recent_10_low[min_price_idx]
            cci_trough = recent_10_cci[min_cci_idx]
            for i in range(max(0, len(recent_10_low) - 5), len(recent_10_low)):
                if recent_10_low[i] <= price_trough * 1.01 and recent_10_cci[i] > cci_trough + 20:
                    persistent_count += 1
            if persistent_count >= 2:
                result = {
                    'type': '底背离(CCI)', 'direction': 'bullish',
                    'detail': f'价格创新低但CCI10未跟随(持续{persistent_count}bar)，可能见底', 'symbol': symbol
                }
                result.update(_compute_price_info('bullish', current_price, dyn_sup, dyn_res))
                patterns.append(result)

    # 5. 量价背离（使用20周期中位数作为基准）
    if len(df_daily) >= 20:
        close_d = df_daily['close'].astype(float).values
        volume_d = df_daily['volume'].astype(float).values
        high_d = df_daily['high'].astype(float).values
        low_d = df_daily['low'].astype(float).values

        median_vol = np.median(volume_d[-20:])  # 20周期成交量中位数

        recent_10_high = high_d[-10:]
        recent_10_vol = volume_d[-10:]
        max_price_idx = np.argmax(recent_10_high)
        # 量价顶背离：价格创新高但成交量低于中位数
        if close_d[-1] >= high_d[-10:].max() * 0.98 and recent_10_vol[max_price_idx] < median_vol:
            result = {
                'type': '量价顶背离', 'direction': 'bearish',
                'detail': '价格创新高但成交量萎缩(低于20日中位数)，警惕回调', 'symbol': symbol
            }
            result.update(_compute_price_info('bearish', current_price, dyn_sup, dyn_res))
            patterns.append(result)

        recent_10_low = low_d[-10:]
        min_price_idx = np.argmin(recent_10_low)
        # 量价底背离：价格创新低但成交量急剧放大（>1.5x中位数）
        if (close_d[-1] <= recent_10_low.min() * 1.02 and
            recent_10_vol[min_price_idx] > median_vol * 1.5):
            result = {
                'type': '量价底背离', 'direction': 'bullish',
                'detail': '价格创新低但成交量急剧放大(超20日中位数1.5x)，恐慌抛售疑似见底', 'symbol': symbol
            }
            result.update(_compute_price_info('bullish', current_price, dyn_sup, dyn_res))
            patterns.append(result)

    # 6. K线形态反转
    pat = candlestick_patterns(df_daily)
    # 黄昏之星 → 看跌
    if pat['counts'].get('evening_star', 0) > 0 or pat['counts'].get('dark_cloud_cover', 0) > 0:
        result = {
            'type': '形态反转-看跌', 'direction': 'bearish',
            'detail': f"K线形态: {pat['dominant_pattern']}，{pat['recent_signals'][:2]}",
            'symbol': symbol
        }
        result.update(_compute_price_info('bearish', current_price, dyn_sup, dyn_res))
        patterns.append(result)
    # 启明星 → 看涨
    if pat['counts'].get('morning_star', 0) > 0 or pat['counts'].get('piercing', 0) > 0:
        result = {
            'type': '形态反转-看涨', 'direction': 'bullish',
            'detail': f"K线形态: {pat['dominant_pattern']}，{pat['recent_signals'][:2]}",
            'symbol': symbol
        }
        result.update(_compute_price_info('bullish', current_price, dyn_sup, dyn_res))
        patterns.append(result)

    # 收集完毕，返回得分最高的信号
    if not patterns:
        return None

    # 按得分排序，返回最高分
    patterns.sort(key=_score_pattern, reverse=True)
    return patterns[0]


def verify_with_ai(reversal_signal, df_daily, df_hourly, symbol):
    """
    AI二次验证：使用至少20根日线和20根小时线，提交给DeepSeek判断。
    """
    if not reversal_signal:
        return None

    daily_table = df_daily.tail(20).to_string()
    hourly_table = df_hourly.tail(20).to_string() if df_hourly is not None else "无小时线"

    prompt = f"""
你是一位专业交易验证员。请基于下面提供的K线数据（最近20根日线、20根小时线），验证一个本地系统检测到的反转信号是否可靠，并给出最终决策。

【本地检测信号】
类型：{reversal_signal['type']}
方向：{reversal_signal['direction']}
详情：{reversal_signal['detail']}
股票：{symbol}

【日线K线（最近20根）】
{daily_table}

【小时K线（最近20根）】
{hourly_table}

请输出严格JSON（不含其他文字）：
{{
  "confirmed": true 或 false,
  "confidence": 0.0到1.0之间,
  "final_direction": "bullish" 或 "bearish" 或 "neutral",
  "entry_zone": "入场价格区间",
  "stop_loss": "止损价",
  "take_profit": "止盈价",
  "reason": "简明验证理由"
}}
"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500
    }

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        content = data['choices'][0]['message']['content']
        # 提取JSON
        start = content.find('{')
        end = content.rfind('}')
        if start == -1 or end == -1:
            print(f"⚠️ AI验证返回非JSON: {content}")
            return None
        result = json.loads(content[start:end+1])

        if result.get('confirmed'):
            return {
                'symbol': symbol,
                'signal': result.get('final_direction', reversal_signal['direction']),
                'confidence': result.get('confidence', 0.6),
                'resonance_detail': f"[AI验证] {reversal_signal['detail']} - {result.get('reason', '')}",
                'entry_zone': result.get('entry_zone', ''),
                'stop_loss': result.get('stop_loss', ''),
                'targets': [result.get('take_profit', '')],
                'position_advice': '反转确认，轻仓试探',
                'invalidation_condition': '',
                'risk_reward_analysis': '',
                'weekly_align': '未计算',
                'source': 'AI验证反转'
            }
        else:
            print(f"🤖 AI验证未通过: {result.get('reason')}")
            return None

    except Exception as e:
        print(f"❌ AI验证请求失败: {e}")
        return None