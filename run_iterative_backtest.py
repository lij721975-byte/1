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
        }

    def run(self) -> List[Dict[str, Any]]:
        """Run the full iterative loop."""
        print()
        print("=" * 70)
        print("  SELF-ITERATIVE BACKTEST WITH SKILL WEIGHT EVOLUTION")
        print("=" * 70)
        print(f"  Stocks: {len(self.stock_pool)}  |  "
              f"Period: {self.start_date} -> {self.end_date}")
        print(f"  Confidence threshold: {self.confidence_threshold}  |  "
              f"Max iterations: {self.max_iterations}")
        print(f"  Convergence tolerance: {self.convergence_tol} (Sharpe delta)")
        print(f"  Clear before start: {self.clear_learned_before_start}")
        print("=" * 70)

        init_db()

        if self.clear_learned_before_start:
            self._clear_learned_weights()

        prev_sharpe: Optional[float] = None
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n{'#' * 70}")
            print(f"# ITERATION {iteration}/{self.max_iterations}")
            print(f"{'#' * 70}")

            result = self._run_iteration(iteration, verbose=(iteration == 1))
            self.iteration_results.append(result)

            sharpe = result['sharpe_ratio']
            n_trades = result['total_trades']

            print(f"\n  Iter {iteration} | Sharpe={sharpe:.3f}  "
                  f"Return={result['total_return_pct']:+.2f}%  "
                  f"MaxDD={result['max_drawdown_pct']:.2f}%  "
                  f"Trades={n_trades}  Win={result['win_rate_pct']:.1f}%  "
                  f"Time={result['elapsed_min']:.1f}m")

            # Show current weight state
            print(f"  Learner: {self.learner._total_trades_attributed} total "
                  f"trades attributed (α={min(0.60, self.learner._total_trades_attributed / 200.0):.3f})")

            if iteration > 1 and prev_sharpe is not None:
                prev_result = self.iteration_results[-2]
                prev_trades = prev_result.get('total_trades', 0)
                trade_change = abs(n_trades - prev_trades) / max(prev_trades, 1)

                delta = abs(sharpe - prev_sharpe)
                result['sharpe_delta'] = delta
                print(f"  ΔSharpe={delta:.4f}  ΔTrades={trade_change:.0%}")

                # Converge only when ALL conditions met:
                # 1. Sharpe delta < tolerance
                # 2. Trade count stable (< 20% change)
                # 3. At least 2 iterations completed
                if (delta < self.convergence_tol and
                    trade_change < 0.20 and
                    iteration >= 3):
                    print(f"  >>> CONVERGED (Sharpe stable, trades stable)")
                    converged = True
                    break
                elif delta < self.convergence_tol and trade_change >= 0.20:
                    print(f"  >>> CONTINUING: Sharpe stable but trades still changing ({trade_change:.0%})")

            prev_sharpe = sharpe

            if n_trades == 0:
                print("  >>> ABORT: 0 trades — check confidence threshold or data")
                break

        self._print_final_summary()
        return self.iteration_results

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
