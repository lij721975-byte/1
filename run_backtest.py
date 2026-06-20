#!/usr/bin/env python
# run_backtest.py — Full-universe backtest entry point
# Extracted from night_run.py inline code to avoid subprocess quoting hell.
import sys; sys.path.insert(0, '.')
import random, numpy as np
from datetime import date, timedelta
from watchlist.stocks import WATCHLIST

random.seed(42)
pool = list(WATCHLIST)[:200]
random.shuffle(pool)
end_date = date.today()
start_date = end_date - timedelta(days=60)
print(f'Universe: {len(pool)} stocks | {start_date} -> {end_date}')

from backtest_engine import BacktestEngine

if __name__ == '__main__':
    BATCH_SIZE = 500
    total_batches = (len(pool) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f'Total Universe: {len(pool)} stocks | Batches: {total_batches} | BATCH_SIZE: {BATCH_SIZE}')

    for i in range(total_batches):
        batch_start = i * BATCH_SIZE
        batch_end = min((i + 1) * BATCH_SIZE, len(pool))
        batch_pool = pool[batch_start:batch_end]

        print(f'\n--- Batch {i+1}/{total_batches}: {len(batch_pool)} stocks ---')
        engine = BacktestEngine(
            stock_pool=batch_pool, start_date=start_date, end_date=end_date,
            initial_capital=1_000_000, position_pct=0.15, stop_pct=0.08,
            target_pct=0.25, max_positions=10, confidence_threshold=0.08,
        )
        engine.run_backtest(verbose=False)

        stats = engine.compute_statistics()
        ts = stats.get('trend_stats', {})
        t0s = stats.get('t0_stats', {})
        print(f'  RETURN={stats.get("total_return_pct",0):+.2f}% | Trend: {ts.get("total_trades",0)}t Win={ts.get("win_rate_pct",0):.1f}% PF={ts.get("profit_factor",0):.2f} | T+0: PnL={t0s.get("total_pnl_rmb",0):,.0f}')

        try:
            engine.save_to_db(run_label=f'night_run_batch_{i+1}')
            print(f'  Batch {i+1}/{total_batches} SAVED_OK')
        except Exception as e:
            print(f'  Batch {i+1}/{total_batches} SAVE_SKIPPED: {e}')

    print(f'\n--- All {total_batches} batches complete ---')
