#!/usr/bin/env python
# run_iterative_backtest.py
"""
Self-iterative backtesting with SKILL weight evolution.

Each iteration:
  1. Run backtest using CURRENT school weights (loaded from DB via get_nuwa_school_weights)
  2. Feed resulting trades to SchoolWeightLearner for attribution
  3. Persist updated per-school metrics and regime weights to DB
  4. Next iteration, get_nuwa_school_weights() picks up the improved weights
  5. Repeat until Sharpe converges or max iterations reached.

Usage:
  python run_iterative_backtest.py [--iterations N] [--stocks N] [--days N]
  python run_iterative_backtest.py --full --days 180 --iterations 5 --clear
"""

import argparse
import time
from datetime import date, timedelta
from typing import Dict, List, Any, Optional

from config import STOCK_POOL, DB_PATH
from trade_logger import init_db
from school_evolution import SchoolWeightLearner


class IterativeBacktestRunner:
    """Orchestrates the backtest → learn → improve → repeat loop."""

    def __init__(
        self,
        stock_pool: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        initial_capital: float = 100_000,
        position_pct: float = 0.05,
        stop_pct: float = 0.05,
        target_pct: float = 0.10,
        max_positions: int = 10,
        confidence_threshold: float = 0.10,
        max_iterations: int = 5,
        convergence_tol: float = 0.05,
        clear_learned_before_start: bool = False,
        workers: int = 6,
    ):
        self.stock_pool = stock_pool or list(STOCK_POOL)
        self.start_date = start_date or (date.today() - timedelta(days=180))
        self.end_date = end_date or date.today()
        self.initial_capital = float(initial_capital)
        self.position_pct = float(position_pct)
        self.stop_pct = float(stop_pct)
        self.target_pct = float(target_pct)
        self.max_positions = int(max_positions)
        self.confidence_threshold = float(confidence_threshold)
        self.max_iterations = int(max_iterations)
        self.convergence_tol = float(convergence_tol)
        self.clear_learned_before_start = clear_learned_before_start
        self.workers = int(workers)

        self.learner = SchoolWeightLearner()
        self.iteration_results: List[Dict[str, Any]] = []

    def _clear_learned_weights(self) -> None:
        """Reset learned weights so iteration starts from default regime weights."""
        import duckdb
        conn = duckdb.connect(DB_PATH)
        conn.execute("DELETE FROM regime_weights_learned")
        conn.commit()
        conn.close()
        print("[Iterative] Cleared previously learned weights from DB.")

    def _run_iteration_parallel(self, iteration: int, verbose: bool) -> Dict[str, Any]:
        """Split stock pool into chunks, run 6 subprocess backtests in parallel, merge results."""
        import json, subprocess, sys, os, time as time_mod

        t0 = time.time()

        # Split stocks into chunks
        stocks = list(self.stock_pool)
        n_chunks = min(max(self.workers, 12), len(stocks) // 20 + 1)
        chunk_size = max(1, len(stocks) // n_chunks)
        chunks = [stocks[i:i + chunk_size] for i in range(0, len(stocks), chunk_size)]
        # Merge trailing small chunk
        while len(chunks) > n_chunks:
            leftovers = chunks.pop()
            chunks[-1].extend(leftovers)

        n_chunks = len(chunks)
        capital_per_chunk = self.initial_capital / n_chunks

        # Build work items
        script_dir = os.path.dirname(os.path.abspath(__file__))
        worker_script = os.path.join(script_dir, '_sweep_worker_subprocess.py')
        work_items = []
        for i, chunk in enumerate(chunks):
            label = f"iter{iteration}_chunk{i}"
            work_items.append((
                chunk,
                str(self.start_date), str(self.end_date),
                capital_per_chunk,
                self.position_pct, self.stop_pct, self.target_pct,
                self.max_positions, self.confidence_threshold, label
            ))

        print(f"  [Iter {iteration}] Launching {n_chunks} subprocess workers "
              f"({len(stocks)} stocks, {chunk_size}/chunk) ...")

        # Launch subprocess workers
        procs = []
        results = []
        for w in work_items:
            json_args = json.dumps(w, ensure_ascii=False)
            p = subprocess.Popen(
                [sys.executable, '-u', worker_script, json_args],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=script_dir,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            procs.append(p)
            if len(procs) >= n_chunks:
                # Wait for all in this batch
                for p2 in list(procs):
                    stdout, stderr = p2.communicate(timeout=86400)
                    try:
                        r = json.loads(stdout.decode('utf-8', errors='replace'))
                    except json.JSONDecodeError:
                        r = {'error': stdout.decode('utf-8', errors='replace')[:200]}
                    results.append(r)
                procs.clear()

        # Collect remaining
        for p2 in list(procs):
            stdout, stderr = p2.communicate(timeout=86400)
            try:
                r = json.loads(stdout.decode('utf-8', errors='replace'))
            except json.JSONDecodeError:
                r = {'error': stdout.decode('utf-8', errors='replace')[:200]}
            results.append(r)

        # Merge: collect trades from all chunks, compute aggregate stats
        all_trades = []
        total_trades = 0
        merged = {
            'total_return_pct': 0.0,
            'total_trades': 0,
            'win_rate_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'profit_factor': 0.0,
            'annualized_return_pct': 0.0,
        }

        for r in results:
            if 'error' in r:
                print(f"  [Iter {iteration}] Chunk FAILED: {r['error'][:100]}")
                continue
            # Extract trades from DB
            run_id = r.get('run_id')
            if run_id:
                from trade_logger import load_backtest_run
                _, trades = load_backtest_run(run_id)
                if trades:
                    all_trades.extend(trades)
                    total_trades += len(trades)

        if not all_trades:
            print(f"  [Iter {iteration}] ABORT: 0 trades across all chunks")
            merged['total_trades'] = 0
            return {
                'iteration': iteration,
                'run_id': None,
                **merged,
                'elapsed_min': (time.time() - t0) / 60.0,
            }

        # Sort trades by entry date
        all_trades.sort(key=lambda t: t.get('entry_date', ''))

        # Compute aggregate stats from merged trades
        n_trades = len(all_trades)
        if n_trades > 0:
            pnl_pcts = [t.get('net_pnl_pct', 0) for t in all_trades]
            winners = [p for p in pnl_pcts if p > 0]
            losers = [p for p in pnl_pcts if p < 0]
            avg_win = sum(winners) / len(winners) if winners else 0
            avg_loss = sum(losers) / len(losers) if losers else 0

            merged['total_trades'] = n_trades
            merged['total_return_pct'] = round(sum(pnl_pcts), 2)
            merged['win_rate_pct'] = round(len(winners) / n_trades * 100, 1)
            merged['max_drawdown_pct'] = round(min(pnl_pcts) if losers else 0, 2)
            merged['profit_factor'] = round(abs(sum(winners) / sum(losers)) if losers else 0, 2)

            # Sharpe approximation
            import numpy as np
            if len(pnl_pcts) > 1:
                mean_pnl = np.mean(pnl_pcts)
                std_pnl = np.std(pnl_pcts) + 1e-10
                merged['sharpe_ratio'] = round(float(mean_pnl / std_pnl), 3)
                merged['annualized_return_pct'] = round(mean_pnl * 252, 1)

        # Feed merged trades to learner
        if all_trades:
            self.learner.update_from_backtest(all_trades)
            from trade_logger import save_school_performance, save_regime_weights_learned
            run_id = iteration  # Use iteration # as run_id for tracking
            tuned = self.learner.auto_tune_regime_weights()
            for regime, weights in tuned.items():
                total = sum(int(self.learner.metrics[regime][s]['n_signals'])
                           for s in self.learner.metrics[regime])
                save_regime_weights_learned(regime, weights, int(total))

        elapsed = time.time() - t0
        result = {
            'iteration': iteration,
            'run_id': iteration,
            **merged,
            'elapsed_min': elapsed / 60.0,
        }
        return result

    def _run_iteration(self, iteration: int, verbose: bool) -> Dict[str, Any]:
        """Execute one iteration using BacktestEngine with thread-level parallelism."""
        # Always use serial mode (subprocess parallel mode is broken on Windows)
        from backtest_engine import BacktestEngine
        t0 = time.time()
        engine = BacktestEngine(
            stock_pool=self.stock_pool,
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital,
            position_pct=self.position_pct,
            stop_pct=self.stop_pct,
            target_pct=self.target_pct,
        )
        engine.run_backtest(verbose=verbose, workers=self.workers)
        stats = engine.compute_statistics()
        trades = engine.trades
        n_trades = len(trades)
        if n_trades > 0:
            engine.save_to_db(run_label=f"iterative_{iteration}")
            self.learner.update_from_backtest_engine(engine)
        elapsed = time.time() - t0
        return {
            'iteration': iteration,
            'run_id': iteration,
            'sharpe_ratio': stats.get('sharpe_ratio', 0),
            'total_return_pct': stats.get('total_return_pct', 0),
            'max_drawdown_pct': stats.get('max_drawdown_pct', 0),
            'total_trades': n_trades,
            'win_rate_pct': stats.get('win_rate_pct', 0),
            'profit_factor': stats.get('profit_factor', 0),
            'annualized_return_pct': stats.get('annualized_return_pct', 0),
            'elapsed_min': elapsed / 60.0,
            '_trades': trades,
        }

    def run(self) -> List[Dict[str, Any]]:
        """Run walk-forward evolution (replaces old in-sample iterative loop).

        Architecture:
          For each window [T-window, T]:
            1. Backtest on training window → collect trades
            2. Feed trades to SchoolWeightLearner → update weights
            3. Lock learned weights
            4. Backtest on test window [T, T+step] WITH LOCKED WEIGHTS → OOS trades
            5. Slide window forward by step
            6. Concatenate all OOS trades → final pure out-of-sample statistics
        """
        window_days = 180
        step_days = 30

        print()
        print("=" * 70)
        print("  WALK-FORWARD BACKTEST WITH SKILL WEIGHT EVOLUTION")
        print("=" * 70)
        print(f"  Stocks: {len(self.stock_pool)}  |  "
              f"Period: {self.start_date} -> {self.end_date}")
        print(f"  Train window: {window_days}d  |  Test step: {step_days}d")
        print(f"  Confidence: {self.confidence_threshold}")
        print("=" * 70)

        init_db()

        if self.clear_learned_before_start:
            self._clear_learned_weights()

        current = self.start_date
        all_oos_trades = []
        window_results = []
        window_idx = 0

        while current + timedelta(days=window_days + step_days) <= self.end_date:
            train_end = current + timedelta(days=window_days)
            test_start = train_end + timedelta(days=1)
            test_end = min(test_start + timedelta(days=step_days - 1), self.end_date)
            window_idx += 1

            print(f"\n{'#' * 70}")
            print(f"# WINDOW {window_idx}: Train [{current} -> {train_end}]  "
                  f"Test [{test_start} -> {test_end}]")
            print(f"{'#' * 70}")

            # ── Phase A: Backtest on training window ──
            train_runner = IterativeBacktestRunner(
                stock_pool=self.stock_pool,
                start_date=current,
                end_date=train_end,
                initial_capital=self.initial_capital,
                position_pct=self.position_pct,
                stop_pct=self.stop_pct,
                target_pct=self.target_pct,
                max_positions=self.max_positions,
                confidence_threshold=self.confidence_threshold,
                max_iterations=1,
                clear_learned_before_start=False,
                workers=self.workers,
            )
            # Use the raw single-iteration backtest (NOT the recursive run())
            train_result = train_runner._run_iteration(1, verbose=False)
            train_trades = train_result.get('_trades', [])

            # ── Phase B: Learn weights from training window ──
            if train_trades:
                self.learner.update_from_backtest(train_trades)
                tuned = self.learner.auto_tune_regime_weights()
                from trade_logger import save_regime_weights_learned
                save_regime_weights_learned(tuned)
                print(f"  Train: {len(train_trades)} trades "
                      f"Sharpe={train_result.get('sharpe_ratio', 0):.3f} "
                      f"→ weights updated")
            else:
                print(f"  Train: 0 trades → weights unchanged")

            # ── Phase C: Backtest on test window WITH LOCKED WEIGHTS ──
            # Snapshot current weights — they are locked for this OOS window
            test_runner = IterativeBacktestRunner(
                stock_pool=self.stock_pool,
                start_date=test_start,
                end_date=test_end,
                initial_capital=self.initial_capital,
                position_pct=self.position_pct,
                stop_pct=self.stop_pct,
                target_pct=self.target_pct,
                max_positions=self.max_positions,
                confidence_threshold=self.confidence_threshold,
                max_iterations=1,
                clear_learned_before_start=False,
                workers=self.workers,
            )
            test_result = test_runner._run_iteration(1, verbose=False)
            test_trades = test_result.get('_trades', [])
            if test_trades:
                all_oos_trades.extend(test_trades)

            test_sharpe = test_result.get('sharpe_ratio', 0)
            print(f"  Test:  {len(test_trades)} trades "
                  f"Sharpe={test_sharpe:.3f} "
                  f"Return={test_result.get('total_return_pct', 0):+.2f}% ")

            # ── Log window result ──
            window_results.append({
                'window': window_idx,
                'train_start': str(current),
                'train_end': str(train_end),
                'test_start': str(test_start),
                'test_end': str(test_end),
                'train_trades': len(train_trades),
                'train_sharpe': train_result.get('sharpe_ratio', 0),
                'test_trades': len(test_trades),
                'test_sharpe': test_sharpe,
                'test_return': test_result.get('total_return_pct', 0),
            })

            # ── Slide forward ──
            current += timedelta(days=step_days)

        # ── Final: aggregate pure OOS statistics ──
        print("\n" + "=" * 90)
        print("  WALK-FORWARD OUT-OF-SAMPLE RESULTS")
        print("=" * 90)

        if not all_oos_trades:
            print("  No OOS trades generated.")
            return window_results

        all_oos_trades.sort(key=lambda t: t.get('entry_date', ''))
        n_total = len(all_oos_trades)
        pnl_pcts = [t.get('net_pnl_pct', 0) for t in all_oos_trades]
        winners = [p for p in pnl_pcts if p > 0]
        losers = [p for p in pnl_pcts if p < 0]

        import numpy as np
        final_stats = {
            'total_windows': window_idx,
            'total_oos_trades': n_total,
            'total_return_pct': round(sum(pnl_pcts), 2),
            'win_rate_pct': round(len(winners) / max(n_total, 1) * 100, 1),
            'profit_factor': round(abs(sum(winners) / (sum(losers) + 1e-10)), 2),
            'avg_return_per_trade': round(np.mean(pnl_pcts), 4) if pnl_pcts else 0.0,
        }
        if len(pnl_pcts) > 1:
            final_stats['sharpe_ratio'] = round(
                float(np.mean(pnl_pcts) / (np.std(pnl_pcts) + 1e-10)), 3)
        else:
            final_stats['sharpe_ratio'] = 0.0

        print(f"  Windows completed: {window_idx}")
        print(f"  Total OOS trades:  {n_total}")
        print(f"  OOS Sharpe:        {final_stats['sharpe_ratio']:.3f}")
        print(f"  OOS Return:        {final_stats['total_return_pct']:+.2f}%")
        print(f"  OOS Win Rate:      {final_stats['win_rate_pct']:.1f}%")
        print(f"  OOS Profit Factor: {final_stats['profit_factor']:.2f}")
        print(f"  OOS Avg PnL/trade: {final_stats['avg_return_per_trade']:.4f}%")
        print("=" * 90)

        # Per-window table
        print(f"\n  {'Win':<5} {'Train':<24} {'Test':<24} {'TrTrades':>9} {'TeTrades':>9} {'TeSharpe':>9}")
        print("-" * 95)
        for w in window_results:
            print(f"  {w['window']:<5} "
                  f"{w['train_start']}→{w['train_end']:<10}  "
                  f"{w['test_start']}→{w['test_end']:<10}  "
                  f"{w['train_trades']:>9} {w['test_trades']:>9} "
                  f"{w['test_sharpe']:>9.3f}")
        print("-" * 95)

        self.iteration_results = window_results
        return window_results

    def _print_final_summary(self) -> None:
        """Print iteration-by-iteration comparison table."""
        results = self.iteration_results
        if not results:
            print("\n  No results to display.")
            return

        print("\n" + "=" * 90)
        print("  ITERATIVE BACKTEST RESULTS")
        print("=" * 90)
        print(f"  {'Iter':<6} {'Sharpe':>8} {'Return%':>9} {'MaxDD%':>8} "
              f"{'Trades':>7} {'Win%':>7} {'PF':>7} {'AnnRet%':>9} {'ΔSharpe':>9}  {'Time':>6}")
        print("-" * 90)

        first_sharpe = results[0].get('sharpe_ratio', 0) if results else 0
        for r in results:
            delta = r.get('sharpe_delta', 0) or 0
            delta_str = f"{delta:+.4f}" if delta else "     --"
            print(f"  {r['iteration']:<6} {r['sharpe_ratio']:>8.3f} "
                  f"{r['total_return_pct']:>+8.2f}% "
                  f"{r['max_drawdown_pct']:>7.2f}% "
                  f"{r['total_trades']:>7} {r['win_rate_pct']:>6.1f}% "
                  f"{r['profit_factor']:>7.2f} "
                  f"{r['annualized_return_pct']:>+8.2f}% "
                  f"{delta_str:>9}  "
                  f"{r['elapsed_min']:>5.1f}m")

        print("-" * 90)

        last = results[-1]
        improvement = last.get('sharpe_ratio', 0) - first_sharpe
        total_min = sum(r['elapsed_min'] for r in results)
        total_trades_attr = self.learner._total_trades_attributed

        print(f"  Final Sharpe: {last.get('sharpe_ratio', 0):.3f}  |  "
              f"Improvement: {improvement:+.4f}  |  "
              f"Total time: {total_min:.1f} min")
        print(f"  Total trades attributed: {total_trades_attr}  |  "
              f"Iterations: {len(results)}")

        # Top learned weights
        print("\n  Top Learned Weights (transitional):")
        top = self.learner.compute_learned_weights(regime='transitional')
        from expert_ensemble import _REGIME_WEIGHTS
        default_weights = _REGIME_WEIGHTS.get('transitional', {})
        ranked = sorted(top.items(), key=lambda x: x[1], reverse=True)
        for name, w in ranked[:5]:
            dw = default_weights.get(name, 0)
            marker = " ↑" if w > dw + 0.05 else (" ↓" if w < dw - 0.05 else "")
            print(f"    {name}: {w:.3f} (default={dw:.2f}){marker}")

        print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Self-iterative backtesting with SKILL weight evolution'
    )
    parser.add_argument('--iterations', '-n', type=int, default=5,
                        help='Max iterations (default: 5)')
    parser.add_argument('--stocks', '-s', type=int, default=10,
                        help='Number of stocks from pool (default: 10)')
    parser.add_argument('--days', '-d', type=int, default=180,
                        help='Backtest lookback days (default: 180)')
    parser.add_argument('--confidence', '-c', type=float, default=0.10,
                        help='Ensemble confidence threshold (default: 0.10)')
    parser.add_argument('--clear', action='store_true',
                        help='Clear learned weights before starting')
    parser.add_argument('--full', action='store_true',
                        help='Use full stock pool (overrides --stocks)')
    parser.add_argument('--position-pct', type=float, default=0.05,
                        help='Position size as fraction of capital (default: 0.05)')
    parser.add_argument('--stop-pct', type=float, default=0.05,
                        help='Stop loss percentage (default: 0.05)')
    parser.add_argument('--target-pct', type=float, default=0.10,
                        help='Take-profit percentage (default: 0.10)')
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Max concurrent positions (default: 10)')
    parser.add_argument('--tol', type=float, default=0.05,
                        help='Convergence tolerance on Sharpe (default: 0.05)')
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=args.days)

    pool = list(STOCK_POOL) if args.full else list(STOCK_POOL)[:args.stocks]

    runner = IterativeBacktestRunner(
        stock_pool=pool,
        start_date=start,
        end_date=end,
        position_pct=args.position_pct,
        stop_pct=args.stop_pct,
        target_pct=args.target_pct,
        max_positions=args.max_positions,
        confidence_threshold=args.confidence,
        max_iterations=args.iterations,
        convergence_tol=args.tol,
        clear_learned_before_start=args.clear,
    )
    runner.run()


if __name__ == '__main__':
    main()
