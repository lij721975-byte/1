# rebuild_trade_logger.py — one-time reconstruction script
import os

target = r"C:\Users\longn\Desktop\my_quant\trade_logger.py"

# Read current file to extract the function bodies after init_db
with open(target, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where _migrate starts and extract everything from there
migrate_start = None
for i, line in enumerate(lines):
    if line.strip().startswith('def _migrate(c):'):
        migrate_start = i
        break

# The file corruption is in init_db. Restore from migrate onwards.
# Reconstruct init_db cleanly, then append the rest

new_init_db = '''# trade_logger.py
import duckdb
import json
import os
import datetime
from config import DB_PATH


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = duckdb.connect(DB_PATH)
    c = conn.cursor()

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS signals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  symbol TEXT,
                  signal TEXT,
                  confidence REAL,
                  entry_zone TEXT,
                  stop_loss TEXT,
                  targets TEXT,
                  position_advice TEXT,
                  resonance_detail TEXT,
                  weekly_align TEXT,
                  source TEXT DEFAULT \\'DeepSeek\\',
                  evaluated INTEGER DEFAULT 0)\'\'\')

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS outcomes
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  signal_id INTEGER,
                  actual_pnl REAL,
                  hit_target INTEGER,
                  stopped_out INTEGER,
                  evaluated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(signal_id) REFERENCES signals(id))\'\'\')

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS indicator_snapshots
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  signal_id INTEGER UNIQUE,
                  symbol TEXT,
                  snapshot_json TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(signal_id) REFERENCES signals(id))\'\'\')

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS deepseek_responses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  signal_id INTEGER UNIQUE,
                  raw_response TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(signal_id) REFERENCES signals(id))\'\'\')

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS run_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  run_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                  stocks_total INTEGER,
                  stocks_scanned INTEGER,
                  bullish_count INTEGER,
                  bearish_count INTEGER,
                  neutral_count INTEGER,
                  report_path TEXT)\'\'\')

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS local_plan_snapshots
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  signal_id INTEGER UNIQUE,
                  symbol TEXT,
                  plan_json TEXT,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY(signal_id) REFERENCES signals(id))\'\'\')

    c.execute(\'\'\'CREATE TABLE IF NOT EXISTS bayesian_calibration_cache
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT UNIQUE,
                  calibration_json TEXT,
                  n_samples INTEGER,
                  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)\'\'\')

    _migrate(c)

    # WAL + indexes
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_evaluated ON signals(evaluated)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_signals_signal ON signals(signal)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_signal_id ON outcomes(signal_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_indicator_snapshots_signal_id ON indicator_snapshots(signal_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_deepseek_responses_signal_id ON deepseek_responses(signal_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_local_plan_snapshots_signal_id ON local_plan_snapshots(signal_id)")

    conn.commit()
    conn.close()

'''

with open(target, 'w', encoding='utf-8') as f:
    f.write(new_init_db)
    if migrate_start:
        f.writelines(lines[migrate_start:])

print(f"Reconstructed. Migrate starts at line {migrate_start}")
