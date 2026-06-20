#!/usr/bin/env python
# auto_evolve.py
"""
自动随机选股 → 回测 → 自我迭代进化 → 报告

一条命令完成全流程：
  python auto_evolve.py                    # 100只随机, 330天, 5轮迭代
  python auto_evolve.py --stocks 200       # 200只
  python auto_evolve.py --iterations 10    # 10轮迭代
  python auto_evolve.py --walk-forward     # Walk-Forward验证
  python auto_evolve.py --quick            # 快速测试(20只, 90天)
"""

import argparse
import os
import random
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    STOCK_POOL, DB_PATH, DEFAULT_STOP_PCT, DEFAULT_TARGET_PCT,
    DEFAULT_POSITION_PCT, MIN_DAILY_AMOUNT, NUWA_VERBOSE,
)
from trade_logger import init_db

BANNER = """
+==========================================================================+
|        AUTO-EVOLVE: 随机选股 · 自动回测 · 自我迭代 · 持续进化             |
+==========================================================================+
"""


def select_liquid_pool(n: int, seed: int = None, verbose: bool = True,
                        cutoff_date: date = None) -> List[str]:
    """
    Randomly select N liquid stocks from the full pool.

    Filters by avg daily amount > MIN_DAILY_AMOUNT using data UP TO cutoff_date.
    If cutoff_date is provided, uses the 20 bars before that date (eliminates
    survivorship bias and lookahead from using "today's" liquidity snapshot).
    """
    if seed is not None:
        random.seed(seed)
    else:
        random.seed(int(time.time()))

    from data_loader import get_daily_kline

    full_pool = list(STOCK_POOL)
    random.shuffle(full_pool)

    liquid = []
    scanned = 0
    for sym in full_pool:
        if len(liquid) >= n:
            break
        scanned += 1
        try:
            df = get_daily_kline(sym, days=250)
            if df is None or df.empty or len(df) < 60:
                continue
            # ---- Anti-lookahead: slice to cutoff_date ----
            if cutoff_date is not None:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.DatetimeIndex(df.index)
                df = df[df.index <= pd.Timestamp(cutoff_date)]
            if len(df) < 20:
                continue
            avg_vol_lots = float(df['volume'].tail(20).mean())
            avg_close = float(df['close'].tail(20).mean())
            avg_amount = avg_vol_lots * 100 * avg_close
            if avg_amount >= MIN_DAILY_AMOUNT:
                liquid.append(sym)
        except Exception:
            pass
        if scanned >= min(500, len(full_pool)):
            break

    if verbose:
        print(f"[Pool] Scanned {scanned} stocks, found {len(liquid)} liquid "
              f"(target={n}, cutoff={'today' if cutoff_date is None else str(cutoff_date)})")
    return sorted(liquid)


def run_auto_evolve(
    stock_pool: List[str],
    start_date: date,
    end_date: date,
    max_iterations: int = 5,
    confidence_threshold: float = 0.10,
    workers: int = 6,
) -> Dict[str, Any]:
    """Run the full backtest + iterative evolution pipeline."""
    from run_iterative_backtest import IterativeBacktestRunner

    print("\n" + "=" * 70)
    print("  AUTO-EVOLVE: 迭代回测 + 学派权重进化")
    print("=" * 70)
    print(f"  Stocks: {len(stock_pool)}  |  Period: {start_date} -> {end_date}")
    print(f"  Iterations: {max_iterations}  |  Confidence: {confidence_threshold}")
    print(f"  Stop: {DEFAULT_STOP_PCT*100:.0f}%  |  Target: {DEFAULT_TARGET_PCT*100:.0f}%")

    runner = IterativeBacktestRunner(
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100_000,
        position_pct=DEFAULT_POSITION_PCT,
        stop_pct=DEFAULT_STOP_PCT,
        target_pct=DEFAULT_TARGET_PCT,
        max_positions=10,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        convergence_tol=0.05,
        clear_learned_before_start=True,
        workers=7,  # ProcessPool: 7 workers for CPU-bound indicator compute
    )
    results = runner.run()

    # Show final top weights
    if runner.learner:
        print("\n  Top Learned Weights (final):")
        top = runner.learner.compute_learned_weights(regime='transitional')
        for name, w in sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]:
            label = name.replace('school_', '')
            print(f"    {label:<22s}: {w:.3f}")

    return {
        'iteration_results': results,
        'learner': runner.learner,
    }


def main():
    parser = argparse.ArgumentParser(description='Auto-Evolve: fully automated quant pipeline')
    parser.add_argument('--stocks', '-s', type=int, default=100,
                        help='Number of liquid stocks to select (default: 100)')
    parser.add_argument('--days', '-d', type=int, default=330,
                        help='Backtest lookback days (default: 330)')
    parser.add_argument('--iterations', '-n', type=int, default=5,
                        help='Evolution iterations (default: 5)')
    parser.add_argument('--confidence', '-c', type=float, default=0.06,
                        help='Confidence threshold (default: 0.06)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for stock selection (default: time-based)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 20 stocks, 90 days, 3 iterations')
    parser.add_argument('--walk-forward', action='store_true',
                        help='Run walk-forward validation instead of evolution')
    parser.add_argument('--workers', '-w', type=int, default=6,
                        help='Parallel workers (default: 6)')
    parser.add_argument('--pool-file', type=str, default=None,
                        help='Use a custom stock pool file instead of random selection')
    args = parser.parse_args()

    print(BANNER)

    # ---- Quick mode overrides ----
    if args.quick:
        args.stocks = 20
        args.days = 90
        args.iterations = 3
        print("  [QUICK MODE] 20 stocks, 90 days, 3 iterations")

    # ---- Stock pool ----
    if args.pool_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location('custom_pool', args.pool_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'SELECTED_POOL'):
            pool = list(mod.SELECTED_POOL)
        elif hasattr(mod, 'WATCHLIST'):
            pool = list(mod.WATCHLIST)
        else:
            print('[ERROR] Pool file must define SELECTED_POOL or WATCHLIST')
            sys.exit(1)
        print(f"[Pool] Loaded {len(pool)} stocks from {args.pool_file}")
    # ---- Date range (computed BEFORE pool selection to avoid lookahead) ----
    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    if not args.pool_file:
        pool = select_liquid_pool(args.stocks, seed=args.seed, cutoff_date=start_date)
        if len(pool) < 10:
            print(f"[ERROR] Only {len(pool)} liquid stocks found. Lower MIN_DAILY_AMOUNT or check data.")
            sys.exit(1)

    # ---- Initialize DB ----
    init_db()

    total_t0 = time.time()

    if args.walk_forward:
        from run_local_evolution import run_walk_forward
        wf_result = run_walk_forward(
            stock_pool=pool,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=args.confidence,
            position_pct=DEFAULT_POSITION_PCT,
            stop_pct=DEFAULT_STOP_PCT,
            target_pct=DEFAULT_TARGET_PCT,
            max_positions=10,
            capital=100_000,
            max_iterations=args.iterations,
            workers=args.workers,
            n_windows=1,
        )
    else:
        result = run_auto_evolve(
            stock_pool=pool,
            start_date=start_date,
            end_date=end_date,
            max_iterations=args.iterations,
            confidence_threshold=args.confidence,
            workers=args.workers,
        )

    total_elapsed = time.time() - total_t0
    print(f"\n[OK] Auto-Evolve complete in {total_elapsed/60:.1f} min")
    print(f"  Pool: {len(pool)} liquid stocks")
    print(f"  Period: {start_date} -> {end_date}")
    print(f"  Run: python auto_evolve.py --stocks {args.stocks} --days {args.days} --iterations {args.iterations}")

    # ---- Auto daily report ----
    _generate_daily_report(pool, start_date, end_date, result if not args.walk_forward else wf_result,
                           total_elapsed, is_walk_forward=args.walk_forward)


def _generate_daily_report(pool, start_date, end_date, result, elapsed_min, is_walk_forward=False):
    """Generate automated daily report file."""
    import datetime
    report_dir = os.path.join(os.path.dirname(__file__), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir,
        f"report_{datetime.date.today().isoformat()}.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"量化回测日报\n")
        f.write(f"{'='*60}\n")
        f.write(f"日期: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"股票池: {len(pool)} 只\n")
        f.write(f"周期: {start_date} → {end_date}\n")
        f.write(f"耗时: {elapsed_min/60:.1f} 分钟\n")
        f.write(f"模式: {'Walk-Forward' if is_walk_forward else '迭代进化'}\n")

        if is_walk_forward and isinstance(result, dict):
            wf = result
            f.write(f"\n--- Walk-Forward 结果 ---\n")
            train_s = wf.get('avg_train_sharpe', 0)
            test_s = wf.get('avg_test_sharpe', 0)
            gap = wf.get('avg_overfit_gap', 0)
            f.write(f"训练集 Sharpe: {train_s:.3f}\n")
            f.write(f"测试集 Sharpe: {test_s:.3f}\n")
            f.write(f"过拟合差距: {gap:+.3f} {'⚠️ 过拟合' if gap > 0.5 else 'OK'}\n")

        elif isinstance(result, dict) and 'iteration_results' in result:
            iters = result['iteration_results']
            if iters:
                last = iters[-1]
                f.write(f"\n--- 回测结果 ---\n")
                f.write(f"Sharpe: {last.get('sharpe_ratio', 0):.3f}\n")
                f.write(f"胜率: {last.get('win_rate_pct', 0):.1f}%\n")
                f.write(f"交易: {last.get('total_trades', 0)} 笔\n")
                f.write(f"收益: {last.get('total_return_pct', 0):+.2f}%\n")

                learner = result.get('learner')
                if learner:
                    f.write(f"\n--- Top 5 学派权重 ---\n")
                    top = learner.compute_learned_weights(regime='transitional')
                    for name, w in sorted(top.items(), key=lambda x: x[1], reverse=True)[:5]:
                        f.write(f"  {name}: {w:.3f}\n")

        f.write(f"\n{'='*60}\n")

    print(f"  Report: {report_path}")


if __name__ == '__main__':
    main()
