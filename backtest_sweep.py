# backtest_sweep.py
"""
Grid-search parameter optimization for the backtest engine.

Two modes:
  1. run_parameter_sweep()   — grid search over position sizing, stop/target, max positions
  2. run_weight_optimization() — compare different school-weighting schemes

Results are persisted to the database and ranked by Sharpe ratio.
"""

import itertools
import json
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from config import (
    BACKTEST_SWEEP_PARAMS,
    BACKTEST_WEIGHT_SCHEMES,
    STOCK_POOL,
)
# BacktestEngine imported lazily inside functions (avoids mootdx dependency at import time)

# =========================================================================
# Parallel sweep worker (module-level for pickling)
# =========================================================================

def _sweep_worker_safe(args):
    """Wrapper that catches all exceptions for safe multiprocessing."""
    try:
        return _sweep_worker(args)
    except Exception as e:
        import traceback
        return {
            'run_id': None,
            'params': {},
            'label': 'error',
            'sharpe_ratio': -999,
            'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }


def _sweep_worker(args):
    """
    Multiprocessing worker: run a single backtest with given params.

    Args tuple: (stock_pool, start_date_str, end_date_str, capital,
                 position_pct, stop_pct, target_pct, max_positions,
                 confidence_threshold, label)
    Returns: result dict with stats or error
    """
    (stock_pool, start_str, end_str, capital, pos_pct, stop_pct,
     tgt_pct, max_pos, conf_thresh, label) = args

    start_d = date.fromisoformat(start_str)
    end_d = date.fromisoformat(end_str)

    from backtest_engine import BacktestEngine
    engine = BacktestEngine(
        stock_pool=stock_pool,
        start_date=start_d,
        end_date=end_d,
        initial_capital=capital,
        position_pct=pos_pct,
        stop_pct=stop_pct,
        target_pct=tgt_pct,
        max_positions=int(max_pos),
        confidence_threshold=conf_thresh,
    )
    engine.run_backtest(verbose=False)
    stats = engine.compute_statistics()
    run_id = engine.save_to_db(run_label=label)

    return {
        'run_id': run_id,
        'params': {'position_pct': pos_pct, 'stop_pct': stop_pct,
                   'target_pct': tgt_pct, 'max_positions': int(max_pos)},
        'label': label,
        **stats,
    }


def run_parameter_sweep(
    stock_pool: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    initial_capital: float = 100_000,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    max_runs: Optional[int] = None,
    verbose: bool = True,
    confidence_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Grid search over position sizing, stop/target, and max positions.

    Parameters
    ----------
    stock_pool : list or None
        Stock symbols. Defaults to config.STOCK_POOL.
    start_date, end_date : date or None
        Backtest period. Defaults to last 180 days.
    initial_capital : float
        Starting capital in CNY.
    param_grid : dict or None
        Grid dimensions. Defaults to config.BACKTEST_SWEEP_PARAMS.
    max_runs : int or None
        Limit total runs (useful for quick smoke tests).
    verbose : bool
        Print progress.

    Returns
    -------
    list of dicts, each with params + stats, sorted by Sharpe ratio descending.
    """
    if param_grid is None:
        param_grid = BACKTEST_SWEEP_PARAMS

    if start_date is None:
        start_date = date.today() - timedelta(days=180)
    if end_date is None:
        end_date = date.today()

    stock_pool = stock_pool or list(STOCK_POOL)

    # Build cartesian product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    if max_runs is not None:
        combinations = combinations[:max_runs]

    total = len(combinations)
    print(f"[Sweep] Parameter grid: {total} combinations across {keys}")
    print(f"[Sweep] Period: {start_date} -> {end_date}, capital={initial_capital:,.0f}")
    print(f"[Sweep] Stock pool: {len(stock_pool)} stocks")

    results: List[Dict[str, Any]] = []
    best_sharpe = -999.0
    best_params = None

    t_start = time.time()

    # ---- Parallel mode: run multiple combos concurrently ----
    sweep_workers = int(os.environ.get('SWEEP_WORKERS', '0'))
    if sweep_workers <= 0:
        sweep_workers = 6
    sweep_workers = min(sweep_workers, total)

    if sweep_workers > 1 and total > 1:
        print(f"[Sweep] Parallel: {sweep_workers} threads")

        # Build work items
        work_items = []
        for combo in combinations:
            params = dict(zip(keys, combo))
            label = (
                f"sweep_pos{params['position_pct']}_"
                f"stop{params['stop_pct']}_"
                f"tgt{params['target_pct']}_"
                f"max{params['max_positions']}"
            )
            work_items.append((
                stock_pool, str(start_date), str(end_date), initial_capital,
                params['position_pct'], params['stop_pct'], params['target_pct'],
                int(params['max_positions']), confidence_threshold, label
            ))

        # Launch subprocess workers (avoids Windows multiprocessing spawn bugs)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        worker_script = os.path.join(script_dir, '_sweep_worker_subprocess.py')
        procs: List[subprocess.Popen] = []
        proc_map: Dict[int, str] = {}  # pid -> label
        completed = 0

        for w in work_items:
            json_args = json.dumps(w, ensure_ascii=False)
            p = subprocess.Popen(
                [sys.executable, '-u', worker_script, json_args],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                cwd=script_dir,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
            procs.append(p)
            proc_map[p.pid] = w[-1]  # label is last element
            if len(procs) >= sweep_workers:
                # Wait for one to finish before launching more
                finished = False
                while not finished:
                    for p in list(procs):
                        ret = p.poll()
                        if ret is not None:
                            completed += 1
                            label = proc_map.get(p.pid, '?')
                            stdout, stderr = p.communicate()
                            try:
                                r = json.loads(stdout.decode('utf-8', errors='replace'))
                            except json.JSONDecodeError:
                                r = {'error': stdout.decode('utf-8', errors='replace')[:200]}
                            results.append(r)
                            procs.remove(p)
                            del proc_map[p.pid]
                            finished = True

                            if 'sharpe_ratio' in r:
                                sharpe = r['sharpe_ratio']
                                ret = r.get('total_return_pct', 0)
                                trades = r.get('total_trades', 0)
                                if sharpe > best_sharpe:
                                    best_sharpe = sharpe
                                    best_params = r['params']
                                    print(f"  [{completed}/{total}] * New best: Sharpe={sharpe:.3f} {r['params']}")
                                else:
                                    print(f"  [{completed}/{total}] Sharpe={sharpe:.3f} Return={ret:+.1f}% Trades={trades}")
                            else:
                                print(f"  [{completed}/{total}] FAILED: {r.get('error', str(r)[:100])}")

                            elapsed = time.time() - t_start
                            if completed > 0:
                                per_run = elapsed / completed
                                remaining = per_run * (total - completed)
                                print(f"  [Progress] {completed}/{total} done, elapsed={elapsed/60:.1f}m, ETA={remaining/60:.1f}m")
                            break
                    if not finished:
                        time.sleep(1)

        # Wait for remaining processes
        for p in list(procs):
            completed += 1
            label = proc_map.get(p.pid, '?')
            stdout, stderr = p.communicate(timeout=86400)
            try:
                r = json.loads(stdout.decode('utf-8', errors='replace'))
            except json.JSONDecodeError:
                r = {'error': stdout.decode('utf-8', errors='replace')[:200]}
            results.append(r)
            procs.remove(p)
            del proc_map[p.pid]

            if 'sharpe_ratio' in r:
                sharpe = r['sharpe_ratio']
                ret = r.get('total_return_pct', 0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = r['params']
                    print(f"  [{completed}/{total}] * New best: Sharpe={sharpe:.3f} {r['params']}")
                else:
                    print(f"  [{completed}/{total}] Sharpe={sharpe:.3f} Return={ret:+.1f}% Trades={trades}")
            else:
                print(f"  [{completed}/{total}] FAILED: {r.get('error', str(r)[:100])}")

    else:
        # ---- Serial mode ----
        print(f"[Sweep] Serial mode: 1 worker")
        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            label = (
                f"sweep_pos{params['position_pct']}_"
                f"stop{params['stop_pct']}_"
                f"tgt{params['target_pct']}_"
                f"max{params['max_positions']}"
            )

            print(f"\n{'='*60}")
            print(f"  Run {idx + 1}/{total}: {label}")
            print(f"{'='*60}")

            try:
                from backtest_engine import BacktestEngine
                engine = BacktestEngine(
                    stock_pool=stock_pool,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    position_pct=params['position_pct'],
                    stop_pct=params['stop_pct'],
                    target_pct=params['target_pct'],
                    max_positions=int(params['max_positions']),
                    confidence_threshold=confidence_threshold,
                )
                engine.run_backtest(verbose=verbose)
                stats = engine.compute_statistics()

                run_id = engine.save_to_db(run_label=label)

                result = {
                    'run_id': run_id,
                    'params': params,
                    'label': label,
                    **stats,
                }
                results.append(result)

                sharpe = stats.get('sharpe_ratio', 0)
                ret = stats.get('total_return_pct', 0)
                dd = stats.get('max_drawdown_pct', 0)
                trades = stats.get('total_trades', 0)

                print(f"  -> Sharpe={sharpe:.3f}  Return={ret:+.2f}%  "
                      f"MaxDD={dd:.2f}%  Trades={trades}")

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    print(f"  * New best Sharpe: {sharpe:.3f}")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'run_id': None,
                    'params': params,
                    'label': label,
                    'error': str(e),
                })

            # Progress estimate
            elapsed = time.time() - t_start
            if idx > 0:
                per_run = elapsed / (idx + 1)
                remaining = per_run * (total - idx - 1)
                print(f"  [Progress] {idx + 1}/{total} done, "
                      f"elapsed={elapsed/60:.1f}m, "
                      f"ETA remaining={remaining/60:.1f}m")

    # ---- Sort by Sharpe ----
    valid_results = [r for r in results if 'sharpe_ratio' in r]
    valid_results.sort(key=lambda r: r.get('sharpe_ratio', -999), reverse=True)

    # ---- Print final ranking ----
    _print_sweep_summary(valid_results, best_params, best_sharpe, time.time() - t_start)

    return valid_results


def run_weight_optimization(
    stock_pool: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    initial_capital: float = 100_000,
    position_pct: float = 0.05,
    stop_pct: float = 0.05,
    target_pct: float = 0.10,
    max_positions: int = 10,
    weight_schemes: Optional[List[str]] = None,
    verbose: bool = True,
    confidence_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Compare different school-weighting schemes on the same backtest period.

    Schemes:
      - 'equal'          : all schools weighted equally (1.0)
      - 'regime_only'    : hardcoded _REGIME_WEIGHTS only
      - 'nuwa_adaptive'  : regime × trust_score (current default)
      - 'learned'        : data-driven weights from SchoolWeightLearner

    Returns dict mapping scheme_name -> {stats, run_id, weights}
    """
    if weight_schemes is None:
        weight_schemes = BACKTEST_WEIGHT_SCHEMES

    if start_date is None:
        start_date = date.today() - timedelta(days=180)
    if end_date is None:
        end_date = date.today()

    stock_pool = stock_pool or list(STOCK_POOL)

    print(f"[WeightOpt] Comparing {len(weight_schemes)} weight schemes:")
    print(f"  Schemes: {weight_schemes}")
    print(f"  Period: {start_date} -> {end_date}")

    results: Dict[str, Any] = {}

    for scheme in weight_schemes:
        print(f"\n{'='*60}")
        print(f"  Weight scheme: {scheme}")
        print(f"{'='*60}")

        try:
            # Build override weights for the scheme
            nuwa_override = _build_weight_override(scheme, stock_pool)

            from backtest_engine import BacktestEngine
            engine = BacktestEngine(
                stock_pool=stock_pool,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                position_pct=position_pct,
                stop_pct=stop_pct,
                target_pct=target_pct,
                max_positions=max_positions,
                confidence_threshold=confidence_threshold,
            )
            engine.run_backtest(
                nuwa_weights_override=nuwa_override,
                verbose=verbose,
            )
            stats = engine.compute_statistics()

            run_id = engine.save_to_db(run_label=f"weights_{scheme}")

            results[scheme] = {
                'run_id': run_id,
                'stats': stats,
                'weights': nuwa_override,
            }

            print(f"  -> Sharpe={stats['sharpe_ratio']:.3f}  "
                  f"Return={stats['total_return_pct']:+.2f}%  "
                  f"MaxDD={stats['max_drawdown_pct']:.2f}%  "
                  f"WinRate={stats['win_rate_pct']:.1f}%  "
                  f"Trades={stats['total_trades']}")

        except Exception as e:
            print(f"  [FAIL] Scheme '{scheme}' FAILED: {e}")
            import traceback
            traceback.print_exc()

    # ---- Print comparison ----
    _print_weight_comparison(results)

    return results


def _build_weight_override(
    scheme: str,
    stock_pool: List[str],
) -> Optional[Dict[str, float]]:
    """
    Build school weight dict for a given scheme.

    Returns None for 'nuwa_adaptive' (let the engine use its internal logic).
    """
    if scheme == 'nuwa_adaptive':
        return None  # Engine uses get_nuwa_school_weights() internally

    school_names = [
        'school_chanlun', 'school_tang',
        'school_livermore', 'school_busch', 'school_classical',
        'school_risk', 'school_gann', 'school_wyckoff',
        'school_harmonic',
        'school_roc_breakout',
        'school_volume_profile', 'school_fusion',
        'school_mean_reversion',
        'school_capital_flow',
        'school_pattern_features', 'school_brooks_pa',
    ]

    if scheme == 'equal':
        return {name: 1.0 for name in school_names}

    if scheme == 'regime_only':
        from expert_ensemble import _REGIME_WEIGHTS
        return dict(_REGIME_WEIGHTS['transitional'])

    if scheme == 'learned':
        try:
            from trade_logger import load_regime_weights_learned
            all_learned = load_regime_weights_learned()
            if all_learned:
                # Use trending or first available regime
                for regime in ['trending', 'transitional', 'ranging', 'volatile']:
                    if regime in all_learned:
                        return all_learned[regime]
        except Exception:
            pass
        print("  [WARN] No learned weights in DB — falling back to regime_only")
        from expert_ensemble import _REGIME_WEIGHTS
        return dict(_REGIME_WEIGHTS['transitional'])

    print(f"  [WARN] Unknown scheme '{scheme}' — using equal weights")
    return {name: 1.0 for name in school_names}


def run_quick_sweep(
    stock_pool: Optional[List[str]] = None,
    days: int = 90,
    verbose: bool = False,
    confidence_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Fast sweep over a reduced grid for quick sanity checking.

    Uses a smaller stock subset (first 10), shorter period, and fewer combos.
    """
    end = date.today()
    start = end - timedelta(days=days)

    pool = (stock_pool or list(STOCK_POOL))[:10]

    quick_grid = {
        'position_pct': [0.05, 0.08],
        'stop_pct': [0.05],
        'target_pct': [0.10, 0.15],
        'max_positions': [5, 10],
    }

    print(f"[QuickSweep] {len(pool)} stocks, {days} days, "
          f"{np.prod([len(v) for v in quick_grid.values()])} combos")

    return run_parameter_sweep(
        stock_pool=pool,
        start_date=start,
        end_date=end,
        param_grid=quick_grid,
        verbose=verbose,
        confidence_threshold=confidence_threshold,
    )


# =========================================================================
# Reporting helpers
# =========================================================================

def _print_sweep_summary(
    results: List[Dict[str, Any]],
    best_params: Optional[Dict[str, Any]],
    best_sharpe: float,
    elapsed: float,
) -> None:
    """Print ranked parameter sweep results."""
    print("\n" + "=" * 80)
    print("  PARAMETER SWEEP RESULTS (ranked by Sharpe)")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
          f"{'Win%':>6} {'Trades':>7} {'Pos%':>6} {'Stop%':>6} "
          f"{'Tgt%':>6} {'MaxPos':>7}  Label")
    print("-" * 80)

    for rank, r in enumerate(results[:30], 1):
        p = r.get('params', {})
        label = r.get('label', '')[:50]
        print(
            f"  {rank:<5} {r.get('sharpe_ratio', 0):>7.3f} "
            f"{r.get('total_return_pct', 0):>+7.1f}% "
            f"{r.get('max_drawdown_pct', 0):>6.1f}% "
            f"{r.get('win_rate_pct', 0):>5.1f}% "
            f"{r.get('total_trades', 0):>7} "
            f"{p.get('position_pct', '?'):>6} "
            f"{p.get('stop_pct', '?'):>6} "
            f"{p.get('target_pct', '?'):>6} "
            f"{p.get('max_positions', '?'):>7}  "
            f"{label}"
        )

    print("-" * 80)
    if best_params:
        print(f"  Best params: {best_params}  (Sharpe={best_sharpe:.3f})")
    print(f"  Total runs: {len(results)}  |  Elapsed: {elapsed/60:.1f} min")
    print("=" * 80)


def _print_weight_comparison(results: Dict[str, Any]) -> None:
    """Print side-by-side weight scheme comparison."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("  WEIGHT SCHEME COMPARISON")
    print("=" * 80)
    print(f"  {'Scheme':<18} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} "
          f"{'Win%':>6} {'Trades':>7} {'PF':>7} {'AnnRet':>8}")
    print("-" * 80)

    # Sort by Sharpe
    sorted_schemes = sorted(
        results.items(),
        key=lambda x: x[1]['stats'].get('sharpe_ratio', -999),
        reverse=True,
    )

    best_sharpe = None
    for scheme, data in sorted_schemes:
        s = data['stats']
        sharpe = s.get('sharpe_ratio', 0)
        if best_sharpe is None:
            best_sharpe = sharpe
        marker = " *" if sharpe == best_sharpe else ""
        print(
            f"  {scheme:<18} {sharpe:>7.3f} "
            f"{s.get('total_return_pct', 0):>+7.1f}% "
            f"{s.get('max_drawdown_pct', 0):>6.1f}% "
            f"{s.get('win_rate_pct', 0):>5.1f}% "
            f"{s.get('total_trades', 0):>7} "
            f"{s.get('profit_factor', 0):>7.2f} "
            f"{s.get('annualized_return_pct', 0):>+7.1f}%"
            f"{marker}"
        )

    print("=" * 80)

    # Show which scheme won and by how much
    if len(sorted_schemes) >= 2:
        best_name = sorted_schemes[0][0]
        best_s = sorted_schemes[0][1]['stats']['sharpe_ratio']
        worst_name = sorted_schemes[-1][0]
        worst_s = sorted_schemes[-1][1]['stats']['sharpe_ratio']
        delta = best_s - worst_s
        print(f"  Winner: {best_name} (Sharpe={best_s:.3f})")
        print(f"  Spread: ΔSharpe = {delta:.3f} vs worst ({worst_name})")


# =========================================================================
# __main__ — quick demo
# =========================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        print("=== FULL PARAMETER SWEEP ===")
        run_parameter_sweep()
    elif len(sys.argv) > 1 and sys.argv[1] == 'weights':
        print("=== WEIGHT SCHEME COMPARISON ===")
        run_weight_optimization()
    else:
        print("=== QUICK SWEEP (reduced grid, 10 stocks, 90 days) ===")
        run_quick_sweep()
        print("\nUsage: python backtest_sweep.py [quick|full|weights]")
