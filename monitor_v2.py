# monitor_v2.py — 时间表驱动的盘中监控
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_loader import get_daily_kline, get_hourly_kline, get_weekly_kline, get_stock_name
from indicators_v2 import compute_all_indicators_v2
from deepseek_analysis import get_ai_signal
from trade_logger import init_db, log_signal, log_indicators, log_local_plan, log_deepseek_response, log_run
from reversal_detector import check_reversal, verify_with_ai
from report_generator import generate_html_report, generate_html_report_v2
from backtest_feedback import evaluate_past_signals, get_historical_stats
from local_trade_plan import extract_local_trade_plan
from expert_ensemble import compute_timeframe_coordination
from advanced_indicators import portfolio_risk_assessment
from config import STOCK_POOL, A_SHARE_HOLIDAYS

MAX_WORKERS = 15  # DeepSeek API并发数

# 时间表（24小时制）
PRE_MARKET_HOUR, PRE_MARKET_MIN = 9, 26     # 盘前完整分析
MONITOR_START_H, MONITOR_START_M = 9, 30     # 监控开始
SESSION_END_H, SESSION_END_M = 15, 2          # 收市停止
REVERSAL_INTERVAL = 30 * 60                    # 反转扫描间隔（秒）
FULL_ANALYSIS_INTERVAL = 60 * 60              # DeepSeek完整分析间隔（秒）
NEXT_DAY_START_H, NEXT_DAY_START_M = 9, 25    # 次日自启动


def _now():
    return datetime.datetime.now()


def _time_str():
    return _now().strftime("%H:%M:%S")


def _sleep_until(hour, minute):
    """Sleep until the next occurrence of hour:minute."""
    now = _now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += datetime.timedelta(days=1)
    seconds = (target - now).total_seconds()
    if seconds > 0:
        print(f"[{_time_str()}] 休眠至 {target.strftime('%Y-%m-%d %H:%M')} ({seconds/60:.0f}分钟)")
        time.sleep(seconds)


def _is_weekend():
    return _now().weekday() >= 5  # 5=周六, 6=周日


def _is_lunch_break():
    """A股午间休市 11:30-13:00"""
    now = _now()
    t = now.hour * 100 + now.minute
    return 1130 <= t < 1300


def _is_trading_day():
    """检查今天是否为A股交易日（排除周末和节假日）"""
    now = _now()
    # 周末直接排除
    if now.weekday() >= 5:
        return False
    # 检查是否为假期
    today_str = now.strftime("%Y-%m-%d")
    if today_str in A_SHARE_HOLIDAYS:
        return False
    # 检查是否在交易时段内（9:30-15:00 才有实际交易）
    t = now.hour * 100 + now.minute
    if not (930 <= t <= 1500):
        return False
    return True


def _in_session():
    """是否在交易时段内（9:25 - 15:02）"""
    now = _now()
    t = now.hour * 100 + now.minute
    return 925 <= t <= 1502


def _prepare_one(symbol):
    """Phase 1: 加载数据+计算指标（本地计算）"""
    name = get_stock_name(symbol)
    daily = get_daily_kline(symbol)
    if daily is None:
        return None, symbol, name
    hourly = get_hourly_kline(symbol)
    weekly = get_weekly_kline(symbol)
    indicators = compute_all_indicators_v2(daily, hourly, weekly, symbol=symbol)
    if indicators is None:
        return None, symbol, name

    # Inject full daily DataFrame for schools that need time-series context
    indicators['_df'] = daily

    local_plan = extract_local_trade_plan(indicators, daily_df=daily)
    tf_coord = compute_timeframe_coordination(
        indicators,
        hourly_indicators=None,
        weekly_trend=indicators.get('weekly_trend', '震荡'),
    )
    return {'daily': daily, 'hourly': hourly, 'indicators': indicators,
            'local_plan': local_plan, 'tf_coord': tf_coord, 'name': name}, symbol, name


def _call_ai_one(symbol, prep):
    """Phase 2: 调用DeepSeek API"""
    if prep is None:
        return None
    signal = get_ai_signal(prep['daily'], prep['hourly'], prep['indicators'], symbol)
    if signal:
        signal.setdefault('source', 'DeepSeek')
        signal['stock_name'] = prep['name']
        signal['_local_plan'] = prep['local_plan']
        signal['_tf_coordination'] = prep['tf_coord']
        signal['_indicators'] = prep['indicators']
        print(f"  [{prep['name']}({symbol})] 信号:{signal.get('signal')}, 周线:{signal.get('weekly_align')}")
    else:
        print(f"  [{prep['name']}({symbol})] 未获得有效信号")
    return signal


def run_full_analysis(all_signals_out=None):
    """对全部股票做 DeepSeek 完整分析（3阶段并发），返回信号列表"""
    t0 = time.time()
    print(f"\n{'='*40}")
    print(f"[{_time_str()}] === DeepSeek 完整分析 ({len(STOCK_POOL)}只) ===")
    print(f"{'='*40}")

    # Phase 1: 本地计算（串行，快速）
    prepared = []
    skipped = []
    for symbol in STOCK_POOL:
        prep, _, name = _prepare_one(symbol)
        if prep is None:
            print(f"  [{name}({symbol})] 跳过（数据缺失）")
            skipped.append(symbol)
        else:
            prepared.append((symbol, prep))
    print(f"  Phase1完成: {len(prepared)}只就绪, {len(skipped)}只跳过 ({time.time()-t0:.1f}s)")

    # Phase 2: AI分析（并发）
    signals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_call_ai_one, s, p): s for s, p in prepared}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                signal = future.result()
                if signal:
                    signals.append(signal)
            except Exception as e:
                print(f"  [{symbol}] AI调用异常: {e}")

    elapsed = time.time() - t0
    print(f"  Phase2完成: {len(signals)}/{len(prepared)}成功, 总耗时{elapsed:.1f}s")

    # Phase 3: 日志写入（串行）
    for signal in signals:
        symbol = signal.get('symbol', '')
        local_plan = signal.pop('_local_plan', None)
        indicators = signal.pop('_indicators', {})
        raw = signal.pop('_raw_response', None)
        signal_id = log_signal(signal)
        if local_plan:
            log_local_plan(signal_id, symbol, local_plan)
        log_indicators(signal_id, symbol, indicators)
        if raw:
            log_deepseek_response(signal_id, raw)

    # 组合层面风险评估
    if len(signals) >= 2:
        positions = _build_portfolio_positions(signals)
        if positions:
            portf_risk = portfolio_risk_assessment(positions)
            for w in portf_risk.get('warnings', []):
                print(f"  [组合风险] ⚠️ {w}")

    if all_signals_out is not None:
        all_signals_out.extend(signals)

    return signals


def _build_portfolio_positions(signals):
    """从信号列表构建组合仓位快照供风险评估"""
    positions = []
    for s in signals:
        sig = s.get('signal', 'neutral')
        if sig == 'neutral':
            continue
        lp = s.get('_local_plan', {})
        ps = lp.get('position_sizing', {}) if lp else {}
        positions.append({
            'symbol': s.get('symbol', ''),
            'direction': sig,
            'weight': ps.get('position_pct', 0) / 100 if ps.get('position_pct', 0) else 0.05,
            'sector': '未知',
            'beta': 1.0,
            'atr_pct': lp.get('key_values', {}).get('atr_pct', 2.0) if lp else 2.0,
        })
    return positions


def run_reversal_scan(all_signals_out=None):
    """对所有股票做本地反转扫描 + AI验证"""
    print(f"\n--- [{_time_str()}] 反转扫描 ---")

    signals = []
    for symbol in STOCK_POOL:
        try:
            name = get_stock_name(symbol)
            daily = get_daily_kline(symbol)
            hourly = get_hourly_kline(symbol)

            if daily is None:
                continue

            reversal = check_reversal(hourly, daily, symbol)
            if not reversal:
                continue

            print(f"  {name}({symbol}) 本地:{reversal['type']} {reversal['direction']}，提交AI验证...")
            verified = verify_with_ai(reversal, daily, hourly, symbol)

            if verified:
                verified.setdefault('source', 'AI验证反转')
                verified['stock_name'] = name
                print(f"    AI确认: {verified['resonance_detail'][:60]}")
                signal_id = log_signal(verified)
                log_indicators(signal_id, symbol, reversal.get('indicators', {}))
                signals.append(verified)
            else:
                print(f"    AI拒绝")
                reversal.setdefault('signal', reversal.get('direction', 'neutral'))
                reversal.setdefault('confidence', 0.55)
                reversal.setdefault('source', '本地检测(未验证)')
                reversal['stock_name'] = name
                signals.append(reversal)
        except Exception as e:
            print(f"  [{symbol}] 反转扫描异常: {e}")
            continue

    print(f"[扫描完成] {len(signals)} 条信号")

    if all_signals_out is not None:
        all_signals_out.extend(signals)

    return signals


def generate_and_log_report(cycle_signals, report_type):
    """生成HTML报告并记录运行摘要"""
    eval_stats = None
    hist_stats = None

    try:
        eval_stats = evaluate_past_signals()
        if eval_stats.get('total', 0) > 0:
            print(f"[评估] {eval_stats['total']} 条历史已评估，胜率 {eval_stats['win_rate']}%")
    except Exception as e:
        print(f"[评估] 失败: {e}")

    try:
        hist_stats = get_historical_stats()
    except Exception:
        pass

    if cycle_signals:
        report_path = generate_html_report_v2(cycle_signals, report_type=report_type,
                                              eval_stats=eval_stats, hist_stats=hist_stats)
        bullish_n = sum(1 for s in cycle_signals if s.get('signal') == 'bullish')
        bearish_n = sum(1 for s in cycle_signals if s.get('signal') == 'bearish')
        neutral_n = sum(1 for s in cycle_signals if s.get('signal') == 'neutral')
        kelly_vals = [s.get('kelly_position_pct', 0) or 0 for s in cycle_signals
                      if s.get('signal') in ('bullish', 'bearish')]
        avg_kelly = sum(kelly_vals) / len(kelly_vals) if kelly_vals else 0
        aligned = sum(1 for s in cycle_signals if s.get('bayesian_alignment') == '一致')
        conflict = sum(1 for s in cycle_signals if s.get('bayesian_alignment') == '冲突')
        log_run(len(STOCK_POOL), len(cycle_signals), bullish_n, bearish_n, neutral_n,
                report_path or '', avg_kelly, aligned, conflict)

        # 组合层面风险评估
        if len(cycle_signals) >= 2:
            positions = _build_portfolio_positions(cycle_signals)
            if positions:
                portf_risk = portfolio_risk_assessment(positions)
                for w in portf_risk.get('warnings', []):
                    print(f"  [组合风险] ⚠️ {w}")

        return report_path
    else:
        print("  无新信号，跳过报告")
        return None


def run_session():
    """执行一个完整交易日的监控会话"""
    today = _now().strftime("%Y-%m-%d")
    print(f"\n{'#'*50}")
    print(f"# {today} 交易日监控启动")
    print(f"{'#'*50}")

    # ===== 9:26 盘前 DeepSeek 完整分析 =====
    if _now().hour < PRE_MARKET_HOUR or (_now().hour == PRE_MARKET_HOUR and _now().minute < PRE_MARKET_MIN):
        _sleep_until(PRE_MARKET_HOUR, PRE_MARKET_MIN)

    pre_signals = []
    if _now().hour == PRE_MARKET_HOUR and _now().minute >= PRE_MARKET_MIN:
        run_full_analysis(pre_signals)
        if pre_signals:
            generate_and_log_report(pre_signals, f"盘前分析-{today}")

    # ===== 等待 9:30 =====
    if _now().hour < MONITOR_START_H or (_now().hour == MONITOR_START_H and _now().minute < MONITOR_START_M):
        _sleep_until(MONITOR_START_H, MONITOR_START_M)

    # ===== 盘中监控循环 (9:30 - 15:02) =====
    all_signals = []
    last_reversal_time = None
    last_full_time = _now()  # 盘前刚跑过，从当前时间开始计时

    print(f"\n[{_time_str()}] 盘中监控开始（反转每30分钟 / 完整分析每60分钟）")

    while _in_session():
        now = _now()
        cycle_signals = []

        # 午间休市（11:30-13:00），跳过数据获取，等待开市
        if _is_lunch_break():
            print(f"[{_time_str()}] 午间休市，等待13:00开市...")
            # 计算距离13:00的秒数
            resume = now.replace(hour=13, minute=0, second=0, microsecond=0)
            if now >= resume:
                resume += datetime.timedelta(days=1)
            wait = min((resume - now).total_seconds(), 300)  # 最多等5分钟再检查
            time.sleep(wait)
            continue

        # 每30分钟：反转扫描
        if last_reversal_time is None or (now - last_reversal_time).total_seconds() >= REVERSAL_INTERVAL:
            run_reversal_scan(cycle_signals)
            last_reversal_time = now

        # 每60分钟：DeepSeek 完整分析
        if (now - last_full_time).total_seconds() >= FULL_ANALYSIS_INTERVAL:
            run_full_analysis(cycle_signals)
            last_full_time = now

        # 生成报告
        if cycle_signals:
            all_signals.extend(cycle_signals)
            generate_and_log_report(cycle_signals, f"盘中监控-{today}")

        # 每分钟检查一次
        time.sleep(60)

    # ===== 15:02 收市 =====
    print(f"\n[{_time_str()}] === 今日交易时段结束 (15:02) ===")

    if all_signals:
        print(f"今日共产生 {len(all_signals)} 条信号")
        bullish_n = sum(1 for s in all_signals if s.get('signal') == 'bullish')
        bearish_n = sum(1 for s in all_signals if s.get('signal') == 'bearish')
        neutral_n = sum(1 for s in all_signals if s.get('signal') == 'neutral')
        print(f"  看多:{bullish_n}  看空:{bearish_n}  观望:{neutral_n}")

        generate_and_log_report(all_signals, f"收盘汇总-{today}")

    # 评估历史信号
    try:
        eval_stats = evaluate_past_signals()
        if eval_stats.get('total', 0) > 0:
            print(f"[评估] {eval_stats['total']} 条历史信号已评估，胜率 {eval_stats['win_rate']}%")
    except Exception as e:
        print(f"[评估] 失败: {e}")


def main():
    init_db()
    print(f"[{_time_str()}] 量化监控系统启动")
    print(f"  时间表: {PRE_MARKET_HOUR:02d}:{PRE_MARKET_MIN:02d} 盘前分析 → "
          f"{MONITOR_START_H:02d}:{MONITOR_START_M:02d} 盘中监控 → "
          f"{SESSION_END_H:02d}:{SESSION_END_M:02d} 收市")

    while True:
        now = _now()

        # 周末跳过
        if _is_weekend():
            # 计算到周一9:25的时间
            days_until_monday = 7 - now.weekday()
            if days_until_monday == 7:
                days_until_monday = 0  # 周日，weekday=6，下周一就是1天后
            # 实际上周一=0，周日=6。如果周日，下周一=1天；如果周六，下周一=2天
            if now.weekday() == 5:  # 周六
                days_until_monday = 2
            elif now.weekday() == 6:  # 周日
                days_until_monday = 1

            target = now.replace(hour=NEXT_DAY_START_H, minute=NEXT_DAY_START_M,
                                 second=0, microsecond=0) + datetime.timedelta(days=days_until_monday)
            seconds = (target - now).total_seconds()
            print(f"[{_time_str()}] 周末，休眠至周一 {target.strftime('%Y-%m-%d %H:%M')}")
            time.sleep(seconds)
            continue

        # 如果当前时间在 15:02 之后，休眠至次日 9:25
        stop_today = now.replace(hour=SESSION_END_H, minute=SESSION_END_M, second=0, microsecond=0)
        if now >= stop_today:
            _sleep_until(NEXT_DAY_START_H, NEXT_DAY_START_M)
            continue

        # 如果在 9:25 之前，休眠至 9:25
        start_today = now.replace(hour=NEXT_DAY_START_H, minute=NEXT_DAY_START_M, second=0, microsecond=0)
        if now < start_today:
            _sleep_until(NEXT_DAY_START_H, NEXT_DAY_START_M)
            continue

        # 检查是否为交易日（排除节假日）
        if not _is_trading_day():
            print(f"[{_time_str()}] 非交易日，跳过监控")
            # 休眠到下一个工作日9:25
            days_until_next = 1
            if now.weekday() == 4:  # 周五，跳到下周一
                days_until_next = 3
            elif now.weekday() == 5:  # 周六
                days_until_next = 2
            target = now.replace(hour=NEXT_DAY_START_H, minute=NEXT_DAY_START_M,
                                 second=0, microsecond=0) + datetime.timedelta(days=days_until_next)
            seconds = (target - now).total_seconds()
            if seconds > 0:
                time.sleep(seconds)
            continue

        # 在交易时段内，执行今日监控
        try:
            run_session()
        except Exception as e:
            print(f"[{_time_str()}] 会话异常: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)  # 出错后等待一分钟再试


if __name__ == '__main__':
    main()
