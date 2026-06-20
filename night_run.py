#!/usr/bin/env python
# night_run.py — Fully automated data-accumulation -> train -> shadow-backtest pipeline.
import subprocess, sys, os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def banner(title):
    print(f'\n{"="*70}\n  {title}\n  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n{"="*70}', flush=True)

def run_step(name, cmd):
    banner(f'START: {name}')
    print(f'  CMD: {" ".join(cmd)}', flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f'\n  FATAL: {name} failed (exit {r.returncode}) — pipeline halted.', flush=True)
        sys.exit(1)
    banner(f'DONE: {name}')

if __name__ == '__main__':
    import glob

    # ── Purge stale SQLite/DuckDB files (eliminate cross-module format pollution) ──
    for pattern in ['data/trades.duckdb', 'data/trades.duckdb.wal', 'data/trades.duckdb.tmp']:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f'Purged: {f}', flush=True)
            except Exception as e:
                print(f'Warning: could not remove {f} — {e}', flush=True)

    # Prevent event_store from re-creating SQLite-format file at DuckDB path
    os.environ['EVENT_STORE_DISABLED'] = '1'

    banner('NIGHT PIPELINE: Data -> Train -> Shadow Backtest')

    run_step('Step 1/3: Full-universe backtest',  [PYTHON, 'run_backtest.py'])
    run_step('Step 2/3: Train XGBoost',           [PYTHON, 'train_xgb.py'])
    run_step('Step 3/3: ML shadow backtest',      [PYTHON, 'run_backtest.py'])

    banner('PIPELINE COMPLETE')
