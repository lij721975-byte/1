#!/usr/bin/env python
"""Compare baseline (Rule) vs shadow (ML) backtest runs from DuckDB."""
import duckdb
import pandas as pd

DB = 'data/trades.duckdb'

with duckdb.connect(DB) as conn:
    df = conn.execute(
        "SELECT id, run_label, total_trades, win_rate_pct, profit_factor, "
        "total_return_pct, sharpe_ratio FROM backtest_runs ORDER BY id"
    ).fetchdf()

# Split
baseline = df.iloc[:5]
shadow   = df.iloc[5:10]

# Aggregate
def mean(x):
    return float(x.mean())

rows = [
    ('Total Trades',    int(baseline['total_trades'].sum()),    int(shadow['total_trades'].sum()),      'int'),
    ('Avg Win Rate %',  mean(baseline['win_rate_pct']),        mean(shadow['win_rate_pct']),           'pct'),
    ('Avg Profit Factor', mean(baseline['profit_factor']),     mean(shadow['profit_factor']),           'num'),
    ('Avg Total Return %', mean(baseline['total_return_pct']), mean(shadow['total_return_pct']),        'pct'),
    ('Avg Sharpe',      mean(baseline['sharpe_ratio']),        mean(shadow['sharpe_ratio']),            'num'),
]

# Print
print()
print(f"{'Metric':<22} {'Rule (Baseline)':>16} {'ML (Shadow)':>16} {'Delta':>12}")
print('-' * 66)
for label, base_val, shad_val, kind in rows:
    if kind == 'int':
        delta = f'{shad_val - base_val:+d}'
        print(f'{label:<22} {base_val:>16,d} {shad_val:>16,d} {delta:>12}')
    elif kind == 'pct':
        delta = shad_val - base_val
        print(f'{label:<22} {base_val:>15.2f}% {shad_val:>14.2f}% {delta:>+11.2f}%')
    else:
        delta = shad_val - base_val
        print(f'{label:<22} {base_val:>16.3f} {shad_val:>16.3f} {delta:>+12.3f}')
print()
