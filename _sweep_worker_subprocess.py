#!/usr/bin/env python
# _sweep_worker_subprocess.py
"""
Standalone subprocess worker for parameter sweep.
Takes a JSON work item as command-line argument, runs backtest, prints result as JSON.
Completely independent — avoids all Windows multiprocessing spawn issues.
"""
import json
import sys
import os
import traceback

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No work item provided"}))
        sys.exit(1)

    try:
        work_item = json.loads(sys.argv[1])
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"}))
        sys.exit(1)

    try:
        from datetime import date
        from backtest_engine import BacktestEngine

        (stock_pool, start_str, end_str, capital, pos_pct, stop_pct,
         tgt_pct, max_pos, conf_thresh, label) = work_item

        start_d = date.fromisoformat(start_str)
        end_d = date.fromisoformat(end_str)

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
        # Suppress debug output during backtest
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
        try:
            engine.run_backtest(verbose=False, workers=1)  # workers=1 skips parallel precompute
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        stats = engine.compute_statistics()
        run_id = engine.save_to_db(run_label=label)

        result = {
            'run_id': run_id,
            'params': {'position_pct': pos_pct, 'stop_pct': stop_pct,
                       'target_pct': tgt_pct, 'max_positions': int(max_pos)},
            'label': label,
            **stats,
        }
        # Only this JSON line goes to stdout
        print(json.dumps(result, ensure_ascii=False, default=str))
        sys.exit(0)

    except Exception as e:
        result = {
            'run_id': None,
            'params': {},
            'label': 'error',
            'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }
        print(json.dumps(result, ensure_ascii=False, default=str))
        sys.exit(1)


if __name__ == '__main__':
    main()
