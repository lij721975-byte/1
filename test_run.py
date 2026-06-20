# test_run.py — 单股票快速测试脚本
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from data_loader import get_daily_kline, get_hourly_kline, get_weekly_kline, get_stock_name
from indicators_v2 import compute_all_indicators_v2
from deepseek_analysis import get_ai_signal
from local_trade_plan import extract_local_trade_plan
from expert_ensemble import compute_timeframe_coordination, compute_expert_ensemble
from report_generator import generate_html_report_v2
import json

SYMBOL = "002241"  # 歌尔股份 — 一支活跃股票测试

def test_run():
    name = get_stock_name(SYMBOL)
    print(f"======== 测试: {name}({SYMBOL}) ========")

    # 1. 加载数据
    print("\n[1/6] 加载K线数据...")
    daily = get_daily_kline(SYMBOL)
    hourly = get_hourly_kline(SYMBOL)
    weekly = get_weekly_kline(SYMBOL)

    if daily is None:
        print("ERROR: 日线数据缺失，请确保富易已打开并下载了数据")
        return

    print(f"  日线: {len(daily)}根, 小时线: {len(hourly) if hourly is not None else 0}根, 周线: {len(weekly) if weekly is not None else 0}根")

    # 2. 计算指标
    print("\n[2/6] 计算量化指标...")
    indicators = compute_all_indicators_v2(daily, hourly, weekly)
    if indicators is None:
        print("ERROR: 指标计算失败")
        return

    # 关键指标检查
    vol_keys = [k for k in indicators.keys() if k.startswith('bebe_')]
    print(f"  BEGE字段数: {len(vol_keys)}")
    print(f"  bebe_regime: {indicators.get('bebe_regime', 'MISSING')}")
    print(f"  bebe_good_vol: {indicators.get('bebe_good_vol', 0):.4f}, bebe_bad_vol: {indicators.get('bebe_bad_vol', 0):.4f}")
    print(f"  bebe_vrp_signal: {indicators.get('bebe_vrp_signal', 'MISSING')}")
    print(f"  bebe_vol_asymmetry: {indicators.get('bebe_vol_asymmetry', 0):.3f}")
    print(f"  vol_bebe_regime: {indicators.get('vol_bebe_regime', 'MISSING')}")
    print(f"  vol_bebe_vrp: {indicators.get('vol_bebe_vrp', 'MISSING')}")
    print(f"  HAR方向: {indicators.get('har_direction', 'MISSING')}, HAR衰减: {indicators.get('har_decay', 'MISSING')}")
    print(f"  RV水平: {indicators.get('rv_level', 'MISSING')}, 综合RV: {indicators.get('rv_composite', 0):.4f}")

    # 3. 15学派集成
    print("\n[3/6] 15学派集成计算...")
    ensemble = compute_expert_ensemble(indicators)
    print(f"  集成信号: {ensemble['ensemble_signal']}, 置信度: {ensemble['ensemble_confidence']:.3f}")
    print(f"  多样性: {ensemble['diversity']:.3f}, 协同对: {len(ensemble.get('synergy_pairs',[]))}组, 冲突对: {len(ensemble.get('conflict_pairs',[]))}组")
    print(f"  学派投票: {ensemble['votes']}")
    for sname, sdata in ensemble['school_signals'].items():
        reasons_str = '; '.join(sdata.get('reasons', [])[:2])
        print(f"    {sdata['label']}: {sdata['direction']}({sdata['confidence']:.0%}) — {reasons_str}")

    # 4. 本地交易计划
    print("\n[4/6] 本地交易计划...")
    local_plan = extract_local_trade_plan(indicators, symbol=SYMBOL)
    print(f"  信号: {local_plan['signal']}, 置信度: {local_plan['confidence']:.3f}")
    print(f"  入场区间: {local_plan['entry_zone']}, 止损: {local_plan['stop_loss']}")
    print(f"  融合类型: {local_plan.get('fusion_type', '?')}")
    bege_keys = [k for k in local_plan['key_values'].keys() if 'bebe' in k]
    print(f"  key_values中BEGE字段: {bege_keys}")

    # 打印BEGE相关原则
    bege_principles = [p for p in local_plan.get('principles', []) if 'BEGE' in p or 'VRP' in p]
    for bp in bege_principles:
        print(f"  {bp}")

    # 5. AI分析
    print("\n[5/6] DeepSeek AI分析...")
    signal = get_ai_signal(daily, hourly, indicators, SYMBOL)
    if signal:
        print(f"  AI信号: {signal.get('signal')}, 置信度: {signal.get('confidence')}")
        print(f"  周线对齐: {signal.get('weekly_align', '?')}")
        print(f"  贝叶斯对齐: {signal.get('bayesian_alignment', '?')}")
        print(f"  凯利仓位%: {signal.get('kelly_position_pct', '?')}")
        print(f"  入场区: {signal.get('entry_zone', '?')}")
        print(f"  止损: {signal.get('stop_loss', '?')}")
        print(f"  BEGE环境: {signal.get('bebe_environment', 'MISSING')}")
        print(f"  分布偏移预警: {signal.get('distribution_shift_warning', 'MISSING')}")
        print(f"  波动率模型: {signal.get('volatility_model_confidence', 'MISSING')}")
        # 检查专家投票
        votes = signal.get('expert_votes', {})
        for ename, edata in votes.items():
            print(f"    {ename}: {edata.get('direction','?')}({edata.get('confidence',0):.0%}) — {edata.get('reason','')[:60]}")
    else:
        print("  ERROR: AI分析返回None")

    # 6. 生成报告
    print("\n[6/6] 生成HTML报告...")
    if signal:
        signal['stock_name'] = name
        signal.setdefault('source', 'DeepSeek')
        tf_coord = compute_timeframe_coordination(
            indicators,
            hourly_indicators=None,
            weekly_trend=indicators.get('weekly_trend', '震荡'),
        )
        signal['_local_plan'] = local_plan
        signal['_tf_coordination'] = tf_coord
        report_path = generate_html_report_v2([signal], report_type="单股测试")
        print(f"  报告: {report_path}")

    print(f"\n======== 测试完成 ========")


if __name__ == '__main__':
    test_run()
