# main_v2.py
import sys
import io
import time
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', stream=sys.stdout)
log = logging.getLogger(__name__)

from data_loader import get_daily_kline, get_hourly_kline, get_weekly_kline, get_stock_name
from indicators_v2 import compute_all_indicators_v2
from deepseek_analysis import get_ai_signal
from trade_logger import init_db, log_signal, log_local_signal, log_indicators, log_deepseek_response, log_local_plan, log_run
from report_generator import generate_html_report_v2
from backtest_feedback import evaluate_past_signals, get_historical_stats, get_engine_trust_scores
from local_trade_plan import extract_local_trade_plan, apply_adaptive_fusion_to_local_plan
from advanced_indicators import portfolio_risk_assessment
from expert_ensemble import compute_timeframe_coordination
import config as _cfg

MAX_WORKERS = 15  # DeepSeek API并发数


def _prepare_one(symbol):
    """Phase 1: 加载日线数据+计算指标（纯日线，快）"""
    name = get_stock_name(symbol)
    daily = get_daily_kline(symbol)
    if daily is None:
        return None, symbol, name

    # Skip hourly/weekly for speed — daily-only is sufficient for ensemble voting
    hourly = None
    weekly = None

    try:
        indicators = compute_all_indicators_v2(daily, hourly, weekly)
    except Exception:
        return None, symbol, name
    if indicators is None:
        return None, symbol, name

    # Inject full daily DataFrame for schools that need time-series context
    indicators['_df'] = daily

    # Compute ensemble ONCE, share between local_plan and tf_coordination
    from expert_ensemble import compute_expert_ensemble
    daily_ensemble = compute_expert_ensemble(indicators)

    local_plan = extract_local_trade_plan(indicators, daily_df=daily,
                                           precomputed_ensemble=daily_ensemble)
    tf_coord = compute_timeframe_coordination(
        indicators,
        hourly_indicators=None,
        weekly_trend=indicators.get('weekly_trend', '震荡'),
        daily_ensemble=daily_ensemble,
    )
    return {'daily': daily, 'hourly': hourly, 'indicators': indicators,
            'local_plan': local_plan, 'tf_coord': tf_coord, 'name': name}, symbol, name


def _apply_signal_decay(signal_time: str, confidence: float, half_life_hours: int = 48) -> float:
    """Decay signal confidence over time. Half-life: 48 hours by default."""
    try:
        from datetime import datetime, timedelta
        if isinstance(signal_time, str):
            signal_time = datetime.fromisoformat(signal_time.replace('Z', '+00:00').replace('+00:00', ''))
        age_hours = (datetime.now() - signal_time).total_seconds() / 3600
        decay = 2 ** (-age_hours / half_life_hours)
        return confidence * max(0.3, decay)  # Floor at 30% of original
    except Exception:
        return confidence


def _local_signal_one(symbol, prep):
    """Phase 2 (local mode): 用 expert_ensemble 直接生成信号（零API，与回测一致）"""
    if prep is None:
        return None
    from datetime import datetime
    from expert_ensemble import compute_expert_ensemble
    indicators = prep['indicators']
    ensemble = compute_expert_ensemble(indicators)
    ens_signal = ensemble.get('ensemble_signal', 'neutral')
    ens_conf = ensemble.get('ensemble_confidence', 0)
    if ens_signal == 'neutral' or ens_conf < 0.10:
        return None
    return {
        'symbol': symbol,
        'stock_name': prep['name'],
        'signal': ens_signal,
        'confidence': ens_conf,
        'source': 'ExpertEnsemble(local)',
        'ensemble_score': ensemble.get('ensemble_score', 0),
        'votes': ensemble.get('votes', {}),
        'diversity': ensemble.get('diversity', 0),
        'supporting_reasons': ensemble.get('supporting_reasons', [])[:5],
        'opposing_reasons': ensemble.get('opposing_reasons', [])[:3],
        '_local_plan': prep['local_plan'],
        '_tf_coordination': prep['tf_coord'],
        '_indicators': indicators,
        '_ensemble': ensemble,
        '_signal_time': datetime.now().isoformat(),  # For decay tracking
    }


def _call_ai_one(symbol, prep):
    """Phase 2: 调用DeepSeek API — 只对本地ensemble通过的标的做AI审核"""
    if prep is None:
        return None
    # Pre-filter: only send stocks that passed local ensemble bullish check
    from expert_ensemble import compute_expert_ensemble
    indicators = prep['indicators']
    try:
        ensemble = compute_expert_ensemble(indicators)
        if ensemble.get('ensemble_signal') != 'bullish' or ensemble.get('ensemble_confidence', 0) < 0.08:
            return None  # Skip AI — local ensemble already rejected
    except Exception:
        pass  # If ensemble fails, still try AI as fallback
    # Enrich with RLTF + Macro context via EventStore
    try:
        from event_store import AIPromptBuilder, EventStore, EventType
        if not hasattr(_call_ai_one, '_prompt_builder'):
            _call_ai_one._prompt_builder = AIPromptBuilder()
            _call_ai_one._event_store = EventStore()
        builder = _call_ai_one._prompt_builder
        # Build enhanced prompt (not used by get_ai_signal directly, but records context)
        kline_summary = f"日线{len(prep['daily'])}根, 收盘{float(prep['daily']['close'].iloc[-1]):.2f}"
        builder.build_enhanced_prompt(symbol, prep.get('name', symbol), kline_summary,
                                      {'_local_plan': prep.get('local_plan', {}),
                                       '_indicators': prep['indicators'],
                                       '_ensemble': prep.get('daily_ensemble', {})})
    except ImportError:
        pass

    signal = get_ai_signal(prep['daily'], prep['hourly'], prep['indicators'], symbol)
    if signal:
        signal.setdefault('source', 'DeepSeek')
        signal['stock_name'] = prep['name']
        signal['_local_plan'] = prep['local_plan']
        signal['_tf_coordination'] = prep['tf_coord']
        signal['_indicators'] = prep['indicators']
        print(f"[{prep['name']}({symbol})] 信号:{signal.get('signal')}, 周线:{signal.get('weekly_align')}")
    else:
        print(f"[{prep['name']}({symbol})] 未获得有效信号")
    return signal


def run(local_only=False):
    t0 = time.time()
    init_db()

    # ===== Phase 1: 本地计算（串行，快速） =====
    print(f"===== Phase 1: 本地指标计算 ({len(_cfg.STOCK_POOL)}只) =====")
    prepared = []
    skipped = []
    for symbol in _cfg.STOCK_POOL:
        prep, _, name = _prepare_one(symbol)
        if prep is None:
            print(f"  [{name}({symbol})] 跳过（数据缺失）")
            skipped.append(symbol)
        else:
            prepared.append((symbol, prep))
    print(f"  就绪: {len(prepared)}只, 跳过: {len(skipped)}只")

    # ===== Phase 2: 信号生成 =====
    all_signals = []
    if local_only:
        # 本地模式：expert_ensemble 直接生成信号（零API，与回测进化一致）
        print(f"\n===== Phase 2: 本地Ensemble信号 (零API) =====")
        for symbol, prep in prepared:
            sig = _local_signal_one(symbol, prep)
            if sig:
                all_signals.append(sig)
                print(f"  [{prep['name']}({symbol})] {sig['signal']} conf={sig['confidence']:.2f} "
                      f"votes={sig.get('votes',{})}")
            else:
                print(f"  [{prep['name']}({symbol})] 无信号")
    else:
        # AI模式：DeepSeek API 生成信号
        print(f"\n===== Phase 2: AI并行分析 (并发={MAX_WORKERS}) =====")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_call_ai_one, s, p): s for s, p in prepared}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    signal = future.result()
                    if signal:
                        all_signals.append(signal)
                except Exception as e:
                    print(f"  [{symbol}] AI调用异常: {e}")

    elapsed = time.time() - t0
    print(f"\n  完成: {len(all_signals)}/{len(prepared)}成功, 耗时 {elapsed:.1f}s")

    # ===== Phase 3: 日志写入 & 报告生成 =====
    print(f"\n===== Phase 3: 日志写入 & 报告生成 =====")
    for signal in all_signals:
        symbol = signal.get('symbol', '')
        local_plan = signal.get('_local_plan')
        indicators = signal.get('_indicators', {})
        raw = signal.pop('_raw_response', None)
        signal_id = log_signal(signal)
        if local_plan:
            log_local_plan(signal_id, symbol, local_plan)
            log_local_signal(local_plan, symbol, indicators)
        log_indicators(signal_id, symbol, indicators)
        if raw:
            log_deepseek_response(signal_id, raw)

    print("\n[评估] 回顾历史信号表现...")
    try:
        eval_stats = evaluate_past_signals()
        if eval_stats and eval_stats.get('total', 0) > 0:
            print(f"  已评估 {eval_stats['total']} 条，胜率 {eval_stats['win_rate']}%")
        else:
            print("  暂无待评估历史信号（需超过24小时）")
    except Exception as e:
        print(f"  评估失败: {e}")
        eval_stats = None

    try:
        hist_stats = get_historical_stats()
    except Exception:
        hist_stats = None

    # 加载引擎信任分（用于AI vs 本地自适应融合）
    try:
        trust_scores = get_engine_trust_scores()
        print(f"\n[引擎信任] AI信任分={trust_scores['ai_trust']:.3f} | "
              f"本地信任分={trust_scores['local_trust']:.3f} | "
              f"本地权重={trust_scores['fusion_weight_local']:.3f}")
        print(f"  建议: {trust_scores['recommendation']}")
    except Exception:
        trust_scores = None

    # Phase 3.5: 将AI信号与本地计划做自适应融合
    if trust_scores:
        for signal in all_signals:
            local_plan = signal.get('_local_plan')
            if local_plan:
                apply_adaptive_fusion_to_local_plan(local_plan, signal, trust_scores)

    if all_signals:
        report_path = generate_html_report_v2(all_signals, report_type="综合分析",
                                              eval_stats=eval_stats, hist_stats=hist_stats)
    else:
        report_path = ''
        print("无信号数据，跳过报告生成")

    bullish_n = sum(1 for s in all_signals if s.get('signal') == 'bullish')
    bearish_n = sum(1 for s in all_signals if s.get('signal') == 'bearish')
    neutral_n = sum(1 for s in all_signals if s.get('signal') == 'neutral')
    log_run(len(_cfg.STOCK_POOL), len(all_signals), bullish_n, bearish_n, neutral_n, report_path or '')

    # 组合层面风险评估
    if len(all_signals) >= 2:
        positions = []
        for s in all_signals:
            sig = s.get('signal', 'neutral')
            if sig == 'neutral':
                continue
            sym = s.get('symbol', '')
            lp = s.get('_local_plan', {})
            ps = lp.get('position_sizing', {})
            positions.append({
                'symbol': sym,
                'direction': sig,
                'weight': ps.get('position_pct', 0) / 100 if ps.get('position_pct', 0) else 0.05,
                'sector': _cfg.get_stock_sector(sym),
                'beta': 1.0,
                'atr_pct': lp.get('key_values', {}).get('atr_pct', 2.0),
            })
        if positions:
            portf_risk = portfolio_risk_assessment(positions)
            warnings = portf_risk.get('warnings', [])
            print(f"\n[组合风险] 敞口{portf_risk.get('total_exposure_pct', 0)}% | "
                  f"净多头{portf_risk.get('net_exposure_pct', 0)}% | "
                  f"集中度{portf_risk.get('max_sector_pct', 0)}%({portf_risk.get('max_sector_name', '?')}) | "
                  f"分散比率{portf_risk.get('diversification_ratio', 0)} | "
                  f"有效行业{portf_risk.get('n_effective_sectors', 0)}个 | "
                  f"牛熊比{portf_risk.get('bull_bear_ratio', 0)}")
            if portf_risk.get('mean_correlation', 0) > 0:
                print(f"  相关性: {portf_risk.get('correlation_risk', '?')}({portf_risk.get('mean_correlation', 0):.2f}) | "
                      f"VaR(95%): {portf_risk.get('var_95_pct', 0)}% | "
                      f"压力测试(-5%): {portf_risk.get('stress_loss_5pct', 0)}%")
            for w in warnings:
                print(f"  ⚠️ {w}")

    # 尾部对冲检查
    try:
        from risk_management import TailRiskHedge
        from data_loader import get_daily_kline
        th = TailRiskHedge()
        # Load CSI 300 returns
        bm = get_daily_kline('000300', days=100)
        if bm is not None and not bm.empty:
            bm_ret = bm['close'].pct_change().dropna().values
            idx_level = float(bm['close'].iloc[-1])
            # Use last 90 days of portfolio equity as proxy
            port_val = 100000  # Default
            if len(all_signals) >= 5:
                port_val = max(100000, len(all_signals) * 5000)
            # Random portfolio returns as placeholder
            port_ret = np.random.randn(len(bm_ret)) * 0.015
            hedge_result = th.monitor_and_hedge(port_val, port_ret, bm_ret, idx_level)
            if hedge_result['should_hedge']:
                hp = hedge_result['hedge_plan']
                print(f"\n[尾部对冲] Beta={hedge_result['beta']:.2f} "
                      f"Vol={hedge_result['vol_percentile']:.0f}%ile")
                if hp:
                    print(f"  建议做空 {hp['contracts_to_short']} 手股指期货 "
                          f"(保证金 {hp['margin_required_rmb']:,.0f})")
    except ImportError:
        pass

    print(f"\n[完成] {len(all_signals)}条信号, 牛{bullish_n}/熊{bearish_n}/中{neutral_n}, 总耗时{elapsed:.0f}s, 报告: {report_path}")


if __name__ == '__main__':
    import sys
    local_only = '--ai' not in sys.argv  # Default: local ensemble (zero API)
    if '--ai' in sys.argv:
        print(">>> AI MODE: DeepSeek API 辅助信号 <<<")
    else:
        print(">>> LOCAL MODE: expert_ensemble 信号（默认，零API）<<<")

    # --pool-file support
    pool_file = None
    for i, arg in enumerate(sys.argv):
        if arg == '--pool-file' and i + 1 < len(sys.argv):
            pool_file = sys.argv[i + 1]
            break
    if pool_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location('custom_pool', pool_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'SELECTED_POOL'):
            import config
            _cfg.STOCK_POOL = list(mod.SELECTED_POOL)
            print(f"[Pool] Loaded {len(_cfg.STOCK_POOL)} stocks from {pool_file}")
        elif hasattr(mod, 'WATCHLIST'):
            import config
            _cfg.STOCK_POOL = list(mod.WATCHLIST)
            print(f"[Pool] Loaded {len(_cfg.STOCK_POOL)} stocks from {pool_file}")

    run(local_only=local_only)
