#!/usr/bin/env python
# run_local_evolution.py
"""
Local-only quantitative evolution runner — ZERO external API calls.

This script orchestrates the complete self-learning pipeline:
  Phase 1 ─ Quick verification test (small subset, short period)
  Phase 2 ─ Parameter sweep (grid search for optimal stop/target/position)
  Phase 3 ─ Iterative school-weight evolution (backtest → attribute → learn → repeat)
  Phase 4 ─ Final report & best-config export

All computation is LOCAL: indicators_v2 + expert_ensemble + backtest_engine.
No DeepSeek API, no network calls beyond local TDX data files.

Usage:
  python run_local_evolution.py                        # Full pipeline
  python run_local_evolution.py --quick                # Quick test only
  python run_local_evolution.py --skip-sweep           # Skip parameter sweep
  python run_local_evolution.py --iterations 8         # More evolution iterations
  python run_local_evolution.py --stocks 50            # Limit stock count
  python run_local_evolution.py --days 365             # Backtest period (days)
  python run_local_evolution.py --confidence 0.15      # Higher ensemble threshold
  python run_local_evolution.py --resume               # Skip phases already done
"""

import argparse
import os
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Any, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DB_PATH, get_stock_sector, A_SHARE_LOT_SIZE
from trade_logger import init_db
from mypool_3000 import SELECTED_POOL as _RAW_MPOOL


def _normalize_pool(raw_pool):
    """清洗股票池格式：去空格、去前缀(sh./sz.)、统一6位、去重、排序"""
    import re
    cleaned = []
    seen = set()
    for s in raw_pool:
        s = str(s).strip()
        # 去掉 sh. / sz. / SH / SZ 等前缀
        s = re.sub(r'^(sh\.|sz\.|SH\.|SZ\.|sh|sz|SH|SZ)', '', s)
        # 只保留数字
        s = re.sub(r'[^0-9]', '', s)
        # 补齐6位（上证3开头补到6位、深证0开头补到6位）
        if len(s) == 5 and s.startswith('0'):
            s = '0' + s  # 深证 00001 → 000001 一般不缺
        if len(s) == 5 and s.startswith('6'):
            s = s  # 极少数，保持
        if len(s) != 6:
            continue  # 过滤非法代码
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return sorted(cleaned)


STOCK_POOL = _normalize_pool(_RAW_MPOOL)


# ==========================================================================
# Banner
# ==========================================================================

BANNER = """
+==========================================================================+
|            LOCAL QUANTITATIVE EVOLUTION SYSTEM v2.0                      |
|                                                                          |
|  学派自动进化 · 权重自我迭代 · 全本地零API                                |
|                                                                          |
|  Pipeline: 验证 → 参数寻优 → 学派权重进化 → 报告                         |
+==========================================================================+
"""


# ==========================================================================
# CLI
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Local Quant Evolution — Full Pipeline (Zero API)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_local_evolution.py                         # Full pipeline
  python run_local_evolution.py --quick                 # Quick test only
  python run_local_evolution.py --skip-sweep --iterations 10
  python run_local_evolution.py --stocks 30 --days 250 --confidence 0.20
  python run_local_evolution.py --full --iterations 8   # All 5074 stocks (mypool_3000)
        """,
    )
    # Mode flags
    p.add_argument('--quick', action='store_true',
                   help='Run ONLY the quick verification test, then exit.')
    p.add_argument('--skip-sweep', action='store_true',
                   help='Skip the parameter-sweep phase.')
    p.add_argument('--skip-evolution', action='store_true',
                   help='Skip the iterative weight-evolution phase.')
    p.add_argument('--full', action='store_true',
                   help='Use full 5074-stock pool from mypool_3000 (default: first 20 for quick, 100 for full run).')
    p.add_argument('--random', type=int, default=None,
                   help='Randomly select N stocks from the full pool (e.g. --random 500).')
    p.add_argument('--pool-file', type=str, default=None,
                   help='Path to a Python file with a SELECTED_POOL or WATCHLIST list.')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for --random selection (default: 42).')
    p.add_argument('--resume', action='store_true',
                   help='Skip phases whose DB data already exists.')

    # Parameter overrides
    p.add_argument('--stocks', type=int, default=None,
                   help='Number of stocks from pool (default: 20 quick, 100 full run).')
    p.add_argument('--days', type=int, default=None,
                   help='Backtest lookback days (default: 90 quick, 330 full run).')
    p.add_argument('--iterations', '-n', type=int, default=5,
                   help='Max evolution iterations (default: 5).')
    p.add_argument('--confidence', '-c', type=float, default=0.10,
                   help='Ensemble confidence threshold (default: 0.10).')
    p.add_argument('--position-pct', type=float, default=None,
                   help='Position size as fraction of capital (default: 0.05).')
    p.add_argument('--stop-pct', type=float, default=None,
                   help='Stop-loss percentage (default: 0.05).')
    p.add_argument('--target-pct', type=float, default=None,
                   help='Take-profit percentage (default: 0.10).')
    p.add_argument('--max-positions', type=int, default=10,
                   help='Max concurrent positions (default: 10).')
    p.add_argument('--capital', type=float, default=100_000,
                   help='Initial capital in CNY (default: 100000).')
    p.add_argument('--tol', type=float, default=0.05,
                   help='Sharpe convergence tolerance (default: 0.05).')
    p.add_argument('--clear', action='store_true',
                   help='Clear previously learned weights before evolution.')
    p.add_argument('--data-start', type=str, default=None,
                   help='Override data start date (YYYY-MM-DD). Auto-detected if omitted.')
    p.add_argument('--data-end', type=str, default=None,
                   help='Override data end date (YYYY-MM-DD). Defaults to today.')
    p.add_argument('--workers', '-w', type=int, default=0,
                   help='Parallel workers: 0=auto, 1=serial, N=explicit count (default: auto).')
    p.add_argument('--walk-forward', action='store_true',
                   help='Run walk-forward validation (80%% train / 20%% test split).')
    p.add_argument('--wf-windows', type=int, default=1,
                   help='Number of walk-forward windows (default: 1, 80/20 split).')
    return p.parse_args()


# ==========================================================================
# Helper: auto-detect earliest data date
# ==========================================================================

def detect_earliest_data_date(stock_pool: List[str], min_bars: int = 60) -> date:
    """
    Scan a subset of stocks to find the earliest date with sufficient data.
    Returns a date at least `min_bars` trading days after the true earliest bar.
    """
    from data_loader import get_daily_kline

    earliest_reliable: Optional[date] = None
    sample = stock_pool[:min(30, len(stock_pool))]

    print(f"[Detect] Scanning {len(sample)} stocks for data range ...")
    for sym in sample:
        df = get_daily_kline(sym, days=600)
        if df is None or df.empty:
            continue
        if not hasattr(df.index, 'date'):
            df.index = pd.to_datetime(df.index)
        first_date = df.index[0].date() if hasattr(df.index[0], 'date') else pd.Timestamp(df.index[0]).date()
        if earliest_reliable is None or first_date < earliest_reliable:
            earliest_reliable = first_date

    if earliest_reliable is None:
        print("[Detect] WARNING: No data found — falling back to 365 days ago.")
        return date.today() - timedelta(days=365)

    # Add buffer of min_bars trading days (~3 calendar months) to ensure
    # indicator warm-up period is covered
    safe_start = earliest_reliable + timedelta(days=90)
    print(f"[Detect] Earliest bar: {earliest_reliable} → safe start: {safe_start}")
    return safe_start


# ==========================================================================
# Phase 1: Quick verification
# ==========================================================================

def run_quick_verification(
    stock_pool: List[str],
    start_date: date,
    end_date: date,
    confidence_threshold: float = 0.10,
    position_pct: float = 0.05,
    stop_pct: float = 0.05,
    target_pct: float = 0.10,
    max_positions: int = 10,
    capital: float = 100_000,
) -> Optional[Dict[str, Any]]:
    """
    Run a single backtest on a small subset to verify the pipeline works.
    Uses the first 10 stocks and 90 days.
    """
    print("\n" + "=" * 70)
    print("  PHASE 1: QUICK VERIFICATION TEST")
    print("=" * 70)

    pool = list(stock_pool)
    verif_start = end_date - timedelta(days=90)

    print(f"  Stocks     : {len(pool)} (first {pool[0]}..{pool[-1]})")
    print(f"  Period     : {verif_start} → {end_date} (~90 days)")
    print(f"  Confidence : {confidence_threshold}")
    print(f"  Position   : {position_pct:.0%}  Stop: {stop_pct:.0%}  Target: {target_pct:.0%}")
    print(f"  Max pos    : {max_positions}  Capital: {capital:,.0f} CNY")

    from backtest_engine import BacktestEngine

    t0 = time.time()
    engine = BacktestEngine(
        stock_pool=pool,
        start_date=verif_start,
        end_date=end_date,
        initial_capital=capital,
        position_pct=position_pct,
        stop_pct=stop_pct,
        target_pct=target_pct,
        max_positions=max_positions,
        confidence_threshold=confidence_threshold,
    )
    engine.run_backtest(verbose=True)
    stats = engine.compute_statistics()
    elapsed = time.time() - t0

    print(f"\n  [OK] Verification complete in {elapsed/60:.1f} min")
    print(f"  Sharpe={stats['sharpe_ratio']:.3f}  "
          f"Return={stats['total_return_pct']:+.2f}%  "
          f"MaxDD={stats['max_drawdown_pct']:.2f}%  "
          f"WinRate={stats['win_rate_pct']:.1f}%  "
          f"Trades={stats['total_trades']}")

    if stats['total_trades'] == 0:
        print("\n  [WARN] WARNING: 0 trades! Check:")
        print("    1. Data files exist in TDX directory")
        print("    2. Confidence threshold is not too high")
        print("    3. Date range contains trading days")
        return None

    run_id = engine.save_to_db(run_label="quick_verify")
    print(f"  Saved as run_id={run_id}")

    return {
        'engine': engine,
        'stats': stats,
        'run_id': run_id,
        'elapsed_min': elapsed / 60.0,
    }


# ==========================================================================
# Phase 2: Parameter sweep
# ==========================================================================

def run_parameter_sweep_phase(
    stock_pool: List[str],
    start_date: date,
    end_date: date,
    confidence_threshold: float = 0.10,
    capital: float = 100_000,
    quick: bool = False,
) -> List[Dict[str, Any]]:
    """
    Grid search over position sizing, stop/target, max positions.
    Uses a reduced stock subset for speed.
    """
    print("\n" + "=" * 70)
    print("  PHASE 2: PARAMETER SWEEP")
    print("=" * 70)

    from backtest_sweep import run_parameter_sweep

    # Use a reduced grid and stock subset for speed
    if quick:
        sweep_pool = stock_pool[:min(15, len(stock_pool))]
        sweep_days = 120
        param_grid = {
            'position_pct': [0.05, 0.08],
            'stop_pct': [0.05, 0.07],
            'target_pct': [0.10, 0.15],
            'max_positions': [5, 10],
        }
    else:
        sweep_pool = stock_pool[:min(50, len(stock_pool))]
        sweep_days = (end_date - start_date).days
        param_grid = {
            'position_pct': [0.03, 0.05, 0.08, 0.10],
            'stop_pct': [0.03, 0.05, 0.07],
            'target_pct': [0.08, 0.10, 0.15],
            'max_positions': [5, 8, 10, 15],
        }

    sweep_start = end_date - timedelta(days=sweep_days)
    if sweep_start < start_date:
        sweep_start = start_date

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    print(f"  Stocks     : {len(sweep_pool)} (subset for speed)")
    print(f"  Period     : {sweep_start} → {end_date} ({sweep_days}d)")
    print(f"  Grid       : {n_combos} combinations")
    print(f"  Confidence : {confidence_threshold}")

    results = run_parameter_sweep(
        stock_pool=sweep_pool,
        start_date=sweep_start,
        end_date=end_date,
        initial_capital=capital,
        param_grid=param_grid,
        verbose=False,
        confidence_threshold=confidence_threshold,
    )

    if results:
        best = results[0]
        print(f"\n  * Best: {best.get('params', {})}")
        print(f"    Sharpe={best.get('sharpe_ratio', 0):.3f}  "
              f"Return={best.get('total_return_pct', 0):+.2f}%  "
              f"MaxDD={best.get('max_drawdown_pct', 0):.2f}%")
        return results
    else:
        print("  [WARN] No valid sweep results. Using default parameters.")
        return []


# ==========================================================================
# Phase 3: Iterative school-weight evolution
# ==========================================================================

def run_iterative_evolution(
    stock_pool: List[str],
    start_date: date,
    end_date: date,
    max_iterations: int = 5,
    convergence_tol: float = 0.05,
    clear_learned: bool = False,
    confidence_threshold: float = 0.10,
    position_pct: float = 0.05,
    stop_pct: float = 0.05,
    target_pct: float = 0.10,
    max_positions: int = 10,
    capital: float = 100_000,
    best_params: Optional[Dict[str, Any]] = None,
    workers: int = 6,
) -> Dict[str, Any]:
    """
    Run the iterative backtest → attribute → learn → repeat loop.

    Uses SchoolWeightLearner to track per-school P&L attribution and
    evolve the ensemble weights. Each iteration's backtest picks up the
    improved weights via get_nuwa_school_weights() → DB lookup.
    """
    print("\n" + "=" * 70)
    print("  PHASE 3: ITERATIVE SCHOOL-WEIGHT EVOLUTION")
    print("=" * 70)

    # Apply best params from sweep if available
    if best_params:
        position_pct = best_params.get('position_pct', position_pct)
        stop_pct = best_params.get('stop_pct', stop_pct)
        target_pct = best_params.get('target_pct', target_pct)
        max_positions = int(best_params.get('max_positions', max_positions))

    print(f"  Stocks     : {len(stock_pool)}")
    print(f"  Period     : {start_date} → {end_date}")
    print(f"  Iterations : {max_iterations} (tol={convergence_tol})")
    print(f"  Confidence : {confidence_threshold}")
    print(f"  Position   : {position_pct:.0%}  Stop: {stop_pct:.0%}  Target: {target_pct:.0%}")
    print(f"  Max pos    : {max_positions}  Capital: {capital:,.0f} CNY")
    print(f"  Clear prev : {clear_learned}")

    from run_iterative_backtest import IterativeBacktestRunner

    runner = IterativeBacktestRunner(
        stock_pool=stock_pool,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        position_pct=position_pct,
        stop_pct=stop_pct,
        target_pct=target_pct,
        max_positions=max_positions,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        clear_learned_before_start=clear_learned,
        workers=workers,
    )
    results = runner.run()

    return {
        'iteration_results': results,
        'learner': runner.learner,
        'best_params': {
            'position_pct': position_pct,
            'stop_pct': stop_pct,
            'target_pct': target_pct,
            'max_positions': max_positions,
        },
    }


# ==========================================================================
# Phase 4: Final report
# ==========================================================================

def print_final_report(
    verify_result: Optional[Dict],
    sweep_results: List[Dict],
    evolution_result: Optional[Dict],
    stock_pool: List[str],
    start_date: date,
    end_date: date,
    total_elapsed: float,
) -> None:
    """Print comprehensive final report."""
    print("\n" + "+" + "=" * 78 + "+")
    print("|" + "  LOCAL QUANT EVOLUTION — FINAL REPORT".center(78) + "|")
    print("+" + "=" * 78 + "+")
    print(f"|  {'Pool':<20s}: {len(stock_pool):>5d} stocks".ljust(79) + "|")
    print(f"|  {'Period':<20s}: {start_date} → {end_date}".ljust(79) + "|")
    print(f"|  {'Total time':<20s}: {total_elapsed/60:.1f} min".ljust(79) + "|")
    print("+" + "=" * 78 + "+")

    # Phase 1
    if verify_result and verify_result.get('stats'):
        s = verify_result['stats']
        print("|  [Phase 1] Quick Verify".ljust(79) + "|")
        print(f"|    Sharpe={s.get('sharpe_ratio', 0):.3f}  "
              f"Return={s.get('total_return_pct', 0):+.2f}%  "
              f"Trades={s.get('total_trades', 0)}".ljust(79) + "|")

    # Phase 2
    if sweep_results:
        best = sweep_results[0]
        bp = best.get('params', {})
        print("|  [Phase 2] Parameter Sweep".ljust(79) + "|")
        print(f"|    Best: pos={bp.get('position_pct', '?')}  "
              f"stop={bp.get('stop_pct', '?')}  "
              f"tgt={bp.get('target_pct', '?')}  "
              f"max={bp.get('max_positions', '?')}".ljust(79) + "|")
        print(f"|    Sharpe={best.get('sharpe_ratio', 0):.3f}  "
              f"Return={best.get('total_return_pct', 0):+.2f}%  "
              f"MaxDD={best.get('max_drawdown_pct', 0):.2f}%".ljust(79) + "|")

    # Phase 3
    if evolution_result and evolution_result.get('iteration_results'):
        iters = evolution_result['iteration_results']
        print("|  [Phase 3] Weight Evolution".ljust(79) + "|")
        print(f"|    {'Iter':<6} {'Sharpe':>8} {'Return':>9} {'Trades':>7} {'Win%':>7} {'ΔSharpe':>9}".ljust(79) + "|")
        first_s = iters[0].get('sharpe_ratio', 0)
        for r in iters:
            d = r.get('sharpe_delta', 0) or 0
            ds = f"{d:+.4f}" if d else "     --"
            print(f"|    {r['iteration']:<6} {r['sharpe_ratio']:>8.3f} "
                  f"{r['total_return_pct']:>+8.2f}% "
                  f"{r['total_trades']:>7} {r['win_rate_pct']:>6.1f}% "
                  f"{ds:>9}".ljust(79) + "|")
        final_s = iters[-1].get('sharpe_ratio', 0)
        improvement = final_s - first_s
        print(f"|    Improvement: {improvement:+.4f}".ljust(79) + "|")

        # Top learned weights
        learner = evolution_result.get('learner')
        if learner:
            print("|  [Top Learned Weights]".ljust(79) + "|")
            from school_evolution import SCHOOL_NAMES
            top_weights = learner.compute_learned_weights(regime='trending')
            ranked = sorted(top_weights.items(), key=lambda x: x[1], reverse=True)
            for name, w in ranked[:5]:
                print(f"|    {name:<25s}: {w:.3f}".ljust(79) + "|")

    print("+" + "=" * 78 + "+")

    # Save report to file
    report_path = os.path.join(os.path.dirname(__file__), 'reports', 'evolution_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import datetime
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Local Quant Evolution Report\n")
        f.write(f"Generated: {datetime.datetime.now()}\n")
        f.write(f"Pool: {len(stock_pool)} stocks\n")
        f.write(f"Period: {start_date} → {end_date}\n")
        f.write(f"Total time: {total_elapsed/60:.1f} min\n")
        if evolution_result and evolution_result.get('iteration_results'):
            f.write("\nIteration Results:\n")
            for r in evolution_result['iteration_results']:
                f.write(f"  Iter {r['iteration']}: Sharpe={r['sharpe_ratio']:.3f} "
                        f"Return={r['total_return_pct']:+.2f}% Trades={r['total_trades']}\n")
    print(f"\n  Report saved to: {report_path}")


# ==========================================================================
# Walk-Forward Validation
# ==========================================================================

def run_walk_forward(
    stock_pool: List[str],
    start_date: date,
    end_date: date,
    confidence_threshold: float = 0.10,
    position_pct: float = 0.05,
    stop_pct: float = 0.06,
    target_pct: float = 0.30,
    max_positions: int = 10,
    capital: float = 100_000,
    max_iterations: int = 5,
    workers: int = 6,
    n_windows: int = 1,
) -> Dict[str, Any]:
    """
    Walk-forward validation: train on early data, test on later data.

    For n_windows=1: single 80/20 split.
    For n_windows>1: rolling windows of equal size.

    Returns dict with train_sharpe, test_sharpe, overfit_ratio, details.
    """
    total_days = (end_date - start_date).days
    if n_windows <= 1:
        # Single 80/20 split
        split_date = start_date + timedelta(days=int(total_days * 0.80))
        windows = [(start_date, split_date, split_date, end_date)]
    else:
        # Rolling windows
        window_size = total_days // n_windows
        windows = []
        for w in range(n_windows - 1):
            w_start = start_date + timedelta(days=w * window_size // 2)
            w_train_end = w_start + timedelta(days=int(window_size * 0.80))
            w_test_end = w_start + timedelta(days=window_size)
            if w_test_end > end_date:
                w_test_end = end_date
            windows.append((w_start, w_train_end, w_train_end, w_test_end))

    print("\n" + "=" * 70)
    print("  WALK-FORWARD VALIDATION")
    print("=" * 70)
    print(f"  Windows: {len(windows)}  |  Stocks: {len(stock_pool)}")
    print(f"  Iterations per window: {max_iterations}")

    from run_iterative_backtest import IterativeBacktestRunner
    from trade_logger import init_db
    init_db()

    all_results = []
    for wi, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\n{'#'*60}")
        print(f"# WINDOW {wi+1}/{len(windows)}")
        print(f"# Train: {train_start} -> {train_end}  |  Test: {test_start} -> {test_end}")
        print(f"{'#'*60}")

        # Train on training period
        clear_before = (wi == 0)
        runner = IterativeBacktestRunner(
            stock_pool=stock_pool,
            start_date=train_start,
            end_date=train_end,
            initial_capital=capital,
            position_pct=position_pct,
            stop_pct=stop_pct,
            target_pct=target_pct,
            max_positions=max_positions,
            confidence_threshold=confidence_threshold,
            max_iterations=max_iterations,
            clear_learned_before_start=clear_before,
            workers=workers,
        )
        train_results = runner.run()
        train_final = train_results[-1] if train_results else {}
        train_sharpe = train_final.get('sharpe_ratio', 0)

        # Test on test period (using learned weights from training)
        from backtest_engine import BacktestEngine
        test_engine = BacktestEngine(
            stock_pool=stock_pool,
            start_date=test_start,
            end_date=test_end,
            initial_capital=capital,
            position_pct=position_pct,
            stop_pct=stop_pct,
            target_pct=target_pct,
            max_positions=max_positions,
            confidence_threshold=confidence_threshold,
        )
        test_engine.run_backtest(verbose=False)
        test_stats = test_engine.compute_statistics()
        test_sharpe = test_stats.get('sharpe_ratio', 0)

        overfit = train_sharpe - test_sharpe
        print(f"\n  Window {wi+1}: Train Sharpe={train_sharpe:.3f}  "
              f"Test Sharpe={test_sharpe:.3f}  Overfit gap={overfit:+.3f}")

        all_results.append({
            'window': wi + 1,
            'train_start': str(train_start), 'train_end': str(train_end),
            'test_start': str(test_start), 'test_end': str(test_end),
            'train_sharpe': train_sharpe,
            'test_sharpe': test_sharpe,
            'overfit_gap': overfit,
            'test_return': test_stats.get('total_return_pct', 0),
            'test_win_rate': test_stats.get('win_rate_pct', 0),
            'test_trades': test_stats.get('total_trades', 0),
        })

    # Summary
    train_sharpes = [r['train_sharpe'] for r in all_results]
    test_sharpes = [r['test_sharpe'] for r in all_results]
    avg_overfit = sum(r['overfit_gap'] for r in all_results) / len(all_results)

    print("\n" + "=" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"  Win {r['window']}: Train={r['train_sharpe']:.3f}  Test={r['test_sharpe']:.3f}  "
              f"Δ={r['overfit_gap']:+.3f}  Ret={r['test_return']:+.1f}%  "
              f"WR={r['test_win_rate']:.1f}%  Trades={r['test_trades']}")
    print(f"  Avg Train Sharpe: {sum(train_sharpes)/len(train_sharpes):.3f}")
    print(f"  Avg Test Sharpe:  {sum(test_sharpes)/len(test_sharpes):.3f}")
    print(f"  Avg Overfit Gap:  {avg_overfit:+.3f} "
          f"({'OVERFITTING' if avg_overfit > 0.5 else 'OK'})")
    if avg_overfit > 0.5:
        print(f"  >>> WARNING: High overfitting detected. Consider:")
        print(f"      1. Reduce iterations (--iterations 3)")
        print(f"      2. Reduce school count (remove low-accuracy schools)")
        print(f"      3. Increase confidence threshold (-c 0.10)")
        print(f"      4. Use longer backtest period (--days 330)")
    print("=" * 70)

    return {
        'windows': all_results,
        'avg_train_sharpe': sum(train_sharpes) / len(train_sharpes),
        'avg_test_sharpe': sum(test_sharpes) / len(test_sharpes),
        'avg_overfit_gap': avg_overfit,
    }


# ==========================================================================
# Main
# ==========================================================================

def main():
    args = parse_args()

    # ---- Mode determination ----
    is_quick_only = args.quick
    is_full_pool = args.full

    # ---- Stock pool ----
    if args.pool_file:
        # Load from external Python file
        import importlib.util
        spec = importlib.util.spec_from_file_location('custom_pool', args.pool_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, 'SELECTED_POOL'):
            pool = list(mod.SELECTED_POOL)
        elif hasattr(mod, 'WATCHLIST'):
            pool = list(mod.WATCHLIST)
        else:
            print(f'[ERROR] Pool file {args.pool_file} must define SELECTED_POOL or WATCHLIST')
            sys.exit(1)
        print(f'[Pool] Loaded {len(pool)} stocks from {args.pool_file}')
    elif args.random:
        # Random selection from full pool with liquidity filter
        import random
        random.seed(args.seed)
        full = list(STOCK_POOL)
        n = min(args.random, len(full))
        # Liquidity filter: prefer actively traded stocks, fallback to full pool
        from data_loader import get_daily_kline
        liquid_pool = []
        print(f'[Pool] Scanning {min(500, len(full))} stocks for liquidity ...')
        for sym in full[:min(500, len(full))]:
            try:
                df = get_daily_kline(sym, days=60)
                if df is not None and not df.empty and len(df) >= 30:
                    avg_vol = float(df['volume'].tail(20).mean())
                    avg_price = float(df['close'].tail(20).mean())
                    if avg_vol > 1000 and avg_price > 0:
                        liquid_pool.append(sym)
            except:
                pass
        if len(liquid_pool) >= n:
            print(f'[Pool] {len(liquid_pool)} liquid stocks, selecting {n}')
            pool = sorted(random.sample(liquid_pool, n))
        else:
            print(f'[Pool] Only {len(liquid_pool)} liquid, falling back to full pool')
            pool = sorted(random.sample(full, n))
        print(f'[Pool] Selected {len(pool)} liquid stocks (seed={args.seed})')
    elif args.stocks is not None:
        pool = list(STOCK_POOL)[:args.stocks]
    elif is_quick_only:
        pool = list(STOCK_POOL)[:20]
    elif is_full_pool:
        pool = list(STOCK_POOL)
    else:
        pool = list(STOCK_POOL)[:100]  # Default: moderate subset

    # ---- Date range ----
    end_date = date.today()
    if args.data_end:
        from datetime import datetime as dt
        end_date = dt.strptime(args.data_end, '%Y-%m-%d').date()

    if args.data_start:
        from datetime import datetime as dt
        start_date = dt.strptime(args.data_start, '%Y-%m-%d').date()
    elif args.days is not None:
        start_date = end_date - timedelta(days=args.days)
    elif is_quick_only:
        start_date = end_date - timedelta(days=90)
    else:
        # Auto-detect from actual data (for 16-month coverage)
        import pandas as pd
        start_date = detect_earliest_data_date(pool)
        if (end_date - start_date).days > 500:
            start_date = end_date - timedelta(days=330)  # ~11 months for speed
        print(f"[Main] Auto-detected start: {start_date}")

    # ---- Parameters ----
    from config import DEFAULT_STOP_PCT, DEFAULT_TARGET_PCT, DEFAULT_POSITION_PCT
    confidence = args.confidence
    position_pct = args.position_pct or DEFAULT_POSITION_PCT
    stop_pct = args.stop_pct or DEFAULT_STOP_PCT
    target_pct = args.target_pct or DEFAULT_TARGET_PCT
    max_positions = args.max_positions
    capital = args.capital
    workers = args.workers
    if workers <= 0:
        workers = 6  # default to 6 subprocess workers for Phase 2 & 3

    # Set sweep workers via env var (read by backtest_sweep)
    os.environ['SWEEP_WORKERS'] = str(workers)

    print(BANNER)
    print(f"\n  Configuration:")
    print(f"    Stock pool : {len(pool)} stocks")
    print(f"    Date range : {start_date} → {end_date} ({(end_date - start_date).days} days)")
    print(f"    Confidence : {confidence}")
    print(f"    Workers    : {'auto' if workers == 0 else workers}")
    print(f"    Mode       : {'QUICK-ONLY' if is_quick_only else 'FULL' if is_full_pool else 'STANDARD'}")
    print()

    # ---- Ensure DB is initialized ----
    init_db()

    total_t0 = time.time()
    verify_result = None
    sweep_results = []
    evolution_result = None

    # ---- Phase 1: Quick verification (always run) ----
    verify_result = run_quick_verification(
        stock_pool=pool,
        start_date=start_date,
        end_date=end_date,
        confidence_threshold=confidence,
        position_pct=position_pct,
        stop_pct=stop_pct,
        target_pct=target_pct,
        max_positions=max_positions,
        capital=capital,
    )

    if verify_result is None:
        print("\n  [ERROR] Quick verification FAILED — check data and configuration.")
        print("  Suggestions:")
        print("    1. Verify TDX data path in config.py (FUYI_TDX_DIR)")
        print("    2. Run: python test_data.py  (if available)")
        print("    3. Try lower confidence threshold: --confidence 0.05")
        sys.exit(1)

    if is_quick_only:
        total_elapsed = time.time() - total_t0
        print_final_report(verify_result, [], None, pool, start_date, end_date, total_elapsed)
        return

    # ---- Walk-Forward mode (overrides normal pipeline) ----
    if args.walk_forward:
        wf_result = run_walk_forward(
            stock_pool=pool,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=confidence,
            position_pct=position_pct,
            stop_pct=stop_pct,
            target_pct=target_pct,
            max_positions=max_positions,
            capital=capital,
            max_iterations=args.iterations,
            workers=workers,
            n_windows=args.wf_windows,
        )
        total_elapsed = time.time() - total_t0
        print(f"\n  [OK] Walk-forward validation complete in {total_elapsed/60:.1f} min")
        return

    # ---- Phase 2: Parameter sweep ----
    if not args.skip_sweep:
        sweep_quick = not is_full_pool  # Quick sweep for standard mode
        sweep_results = run_parameter_sweep_phase(
            stock_pool=pool,
            start_date=start_date,
            end_date=end_date,
            confidence_threshold=confidence,
            capital=capital,
            quick=sweep_quick,
        )
    else:
        print("\n  [Phase 2] SKIPPED (--skip-sweep)")

    # ---- Phase 3: Iterative evolution ----
    if not args.skip_evolution:
        best_params = sweep_results[0].get('params', {}) if sweep_results else None
        evolution_result = run_iterative_evolution(
            stock_pool=pool,
            start_date=start_date,
            end_date=end_date,
            max_iterations=args.iterations,
            convergence_tol=args.tol,
            clear_learned=args.clear,
            confidence_threshold=confidence,
            position_pct=position_pct,
            stop_pct=stop_pct,
            target_pct=target_pct,
            max_positions=max_positions,
            capital=capital,
            best_params=best_params,
            workers=workers,
        )
    else:
        print("\n  [Phase 3] SKIPPED (--skip-evolution)")

    # ---- Phase 4: Report ----
    total_elapsed = time.time() - total_t0
    print_final_report(
        verify_result, sweep_results, evolution_result,
        pool, start_date, end_date, total_elapsed,
    )

    print("\n  [OK] Local evolution pipeline complete.")
    print(f"  Total elapsed: {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
