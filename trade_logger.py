# trade_logger.py — DuckDB backend (zero-lock, columnar, vectorized)
import duckdb
import json
import os
import datetime
from config import DB_PATH

CURRENT_SCHEMA_VERSION = 3
_conn: duckdb.DuckDBPyConnection = None


def _get_conn() -> duckdb.DuckDBPyConnection:
    """Lazy-init singleton DuckDB connection (thread-safe, no WAL needed)."""
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(DB_PATH) if os.path.dirname(DB_PATH) else '.', exist_ok=True)
        _conn = duckdb.connect(DB_PATH)
        _conn.execute("SET threads = 4")
        _conn.execute("SET memory_limit = '2GB'")
    return _conn


def _close_conn():
    """Explicitly close the singleton DuckDB connection to release file locks."""
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


def init_db():
    conn = _get_conn()
    c = conn

    c.execute('''CREATE TABLE IF NOT EXISTS signals
                 (id INTEGER PRIMARY KEY,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  symbol VARCHAR,
                  signal VARCHAR,
                  confidence DOUBLE,
                  entry_zone VARCHAR,
                  stop_loss VARCHAR,
                  targets VARCHAR,
                  position_advice VARCHAR,
                  resonance_detail VARCHAR,
                  weekly_align VARCHAR,
                  source VARCHAR DEFAULT 'DeepSeek',
                  evaluated INTEGER DEFAULT 0)''')



    c.execute('''CREATE TABLE IF NOT EXISTS outcomes
                 (id INTEGER PRIMARY KEY,
                  signal_id INTEGER,
                  actual_pnl DOUBLE,
                  hit_target INTEGER,
                  stopped_out INTEGER,
                  evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS indicator_snapshots
                 (id INTEGER PRIMARY KEY,
                  signal_id INTEGER UNIQUE,
                  symbol VARCHAR,
                  snapshot_json VARCHAR,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS deepseek_responses
                 (id INTEGER PRIMARY KEY,
                  signal_id INTEGER UNIQUE,
                  raw_response VARCHAR,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS run_history
                 (id INTEGER PRIMARY KEY,
                  run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  stocks_total INTEGER,
                  stocks_scanned INTEGER,
                  bullish_count INTEGER,
                  bearish_count INTEGER,
                  neutral_count INTEGER,
                  report_path VARCHAR)''')

    c.execute('''CREATE TABLE IF NOT EXISTS local_plan_snapshots
                 (id INTEGER PRIMARY KEY,
                  signal_id INTEGER UNIQUE,
                  symbol VARCHAR,
                  plan_json VARCHAR,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS bayesian_calibration_cache
                 (id INTEGER PRIMARY KEY,
                  symbol VARCHAR UNIQUE,
                  calibration_json VARCHAR,
                  n_samples INTEGER,
                  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS backtest_runs (
                 id INTEGER PRIMARY KEY,
                 run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 run_label VARCHAR,
                 start_date VARCHAR,
                 end_date VARCHAR,
                 initial_capital DOUBLE,
                 position_pct DOUBLE,
                 stop_pct DOUBLE,
                 target_pct DOUBLE,
                 max_positions INTEGER,
                 total_return_pct DOUBLE,
                 annualized_return_pct DOUBLE,
                 sharpe_ratio DOUBLE,
                 max_drawdown_pct DOUBLE,
                 total_trades INTEGER,
                 win_rate_pct DOUBLE,
                 avg_win_pct DOUBLE,
                 avg_loss_pct DOUBLE,
                 profit_factor DOUBLE,
                 config_json VARCHAR,
                 equity_curve_json VARCHAR,
                 weight_config VARCHAR,
                 school_weights_json VARCHAR)''')

    c.execute('''CREATE TABLE IF NOT EXISTS backtest_trades (
                 id INTEGER PRIMARY KEY,
                 run_id INTEGER,
                 symbol VARCHAR,
                 direction VARCHAR,
                 entry_date VARCHAR,
                 exit_date VARCHAR,
                 entry_price DOUBLE,
                 exit_price DOUBLE,
                 shares INTEGER,
                 gross_pnl_pct DOUBLE,
                 net_pnl_pct DOUBLE,
                 net_pnl_rmb DOUBLE,
                 exit_reason VARCHAR,
                 holding_days INTEGER,
                 ensemble_confidence DOUBLE,
                 school_votes_json VARCHAR)''')
    # ── Shadow Trading: add columns for Champion/Challenger logging ──
    for col, col_type in [('shadow_rule_dir', 'VARCHAR'),
                          ('shadow_rule_conf', 'DOUBLE'),
                          ('shadow_ml_dir', 'VARCHAR'),
                          ('shadow_ml_conf', 'DOUBLE'),
                          ('master_source', 'VARCHAR')]:
        try:
            c.execute(f'ALTER TABLE backtest_trades ADD COLUMN IF NOT EXISTS {col} {col_type}')
        except Exception:
            pass  # column already exists or unsupported — safe to skip

    c.execute('''CREATE TABLE IF NOT EXISTS school_performance (
                 id INTEGER PRIMARY KEY,
                 run_id INTEGER,
                 school_name VARCHAR,
                 regime VARCHAR,
                 window_label VARCHAR,
                 n_signals INTEGER,
                 n_correct INTEGER,
                 win_rate DOUBLE,
                 avg_pnl_pct DOUBLE,
                 avg_confidence DOUBLE,
                 sharpe_like_score DOUBLE,
                 direction_bias VARCHAR,
                 contribution_weight DOUBLE,
                 updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    c.execute('''CREATE TABLE IF NOT EXISTS regime_weights_learned (
                 id INTEGER PRIMARY KEY,
                 regime VARCHAR UNIQUE,
                 weights_json VARCHAR,
                 n_samples INTEGER,
                 last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Apply column migrations (idempotent)
    _migrate(c)

    # DuckDB indexes (lightweight — columnar engine doesn't need heavy indexing)
    for idx_sql in [
        "CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_signals_eval ON signals(evaluated)",
        "CREATE INDEX IF NOT EXISTS idx_signals_sym ON signals(symbol)",
        "CREATE INDEX IF NOT EXISTS idx_outcomes_sid ON outcomes(signal_id)",
        "CREATE INDEX IF NOT EXISTS idx_bt_trades_run ON backtest_trades(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_bt_runs_label ON backtest_runs(run_label)",
        "CREATE INDEX IF NOT EXISTS idx_school_perf_school ON school_performance(school_name)",
        "CREATE INDEX IF NOT EXISTS idx_school_perf_run ON school_performance(run_id)",
    ]:
        try:
            c.execute(idx_sql)
        except Exception:
            pass

    print("[trade_logger] DuckDB initialized — columnar engine ready")


def _migrate(c):
    """Add missing columns idempotently (DuckDB: IF NOT EXISTS syntax)."""
    migrations = [
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS kelly_f_star DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS kelly_f_half DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS kelly_win_prob DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS kelly_payoff_ratio DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS kelly_ev DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS kelly_position_pct DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ergo_protections VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ergo_safe_to_enter INTEGER",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ergo_vol_drag DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS bayesian_posterior DOUBLE",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS bayesian_alignment VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS stop_loss_rule VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS trailing_stop VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS invalidation_condition VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS risk_reward_analysis VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_reason VARCHAR",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS batch_targets_json VARCHAR",
        "ALTER TABLE run_history ADD COLUMN IF NOT EXISTS avg_kelly_pct DOUBLE",
        "ALTER TABLE run_history ADD COLUMN IF NOT EXISTS aligned_count INTEGER",
        "ALTER TABLE run_history ADD COLUMN IF NOT EXISTS conflict_count INTEGER",
    ]
    for sql in migrations:
        try:
            c.execute(sql)
        except Exception:
            pass  # Column already exists or table not yet created


def _row_to_dict(cols, row):
    """Convert tuple row to dict using column names."""
    if row is None:
        return {}
    result = {}
    for i in range(len(cols)):
        c = cols[i]
        # DuckDB returns tuples like ('col_name', 'TYPE', ...) — extract [0]
        # SQLite or some DuckDB modes return plain strings
        key = c[0] if isinstance(c, (tuple, list)) else str(c)
        result[key] = row[i]
    return result


def log_signal(signal):
    conn = _get_conn()
    direction = signal.get('signal', signal.get('direction', 'unknown'))
    confidence = signal.get('confidence', 0.5)
    symbol = signal.get('symbol', '')
    entry_zone = str(signal.get('entry_zone', signal.get('entry_price', '')))
    stop_loss = str(signal.get('stop_loss', ''))
    position = str(signal.get('position_advice', ''))
    resonance = str(signal.get('resonance_detail', signal.get('reasons', '')))
    weekly = str(signal.get('weekly_align', ''))
    source = str(signal.get('source', 'DeepSeek'))
    batch_targets = signal.get('batch_targets', [])
    if batch_targets:
        targets_str = json.dumps(batch_targets, ensure_ascii=False)
    else:
        targets_old = signal.get('targets', [])
        targets_str = ','.join(str(t) for t in targets_old) if isinstance(targets_old, list) else str(targets_old or '')
    ergo = signal.get('ergo_protections', None)
    if isinstance(ergo, list):
        ergo = json.dumps(ergo, ensure_ascii=False)

    conn.execute('''INSERT INTO signals (
        symbol, signal, confidence, entry_zone, stop_loss, targets,
        position_advice, resonance_detail, weekly_align, source,
        kelly_f_star, kelly_f_half, kelly_win_prob, kelly_payoff_ratio,
        kelly_ev, kelly_position_pct, ergo_protections, ergo_safe_to_enter,
        ergo_vol_drag, bayesian_posterior, bayesian_alignment,
        stop_loss_rule, trailing_stop, invalidation_condition,
        risk_reward_analysis, entry_reason, batch_targets_json
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
    (symbol, direction, confidence, entry_zone, stop_loss, targets_str,
     position, resonance, weekly, source,
     signal.get('kelly_f_star'), signal.get('kelly_f_half'),
     signal.get('kelly_win_prob'), signal.get('kelly_payoff_ratio'),
     signal.get('kelly_ev'), signal.get('kelly_position_pct'),
     ergo, 1 if signal.get('ergo_safe_to_enter') else 0,
     signal.get('ergo_vol_drag'), signal.get('bayesian_posterior'),
     str(signal.get('bayesian_alignment', '')),
     str(signal.get('stop_loss_rule', '')), str(signal.get('trailing_stop', '')),
     str(signal.get('invalidation_condition', '')),
     str(signal.get('risk_reward_analysis', '')),
     str(signal.get('entry_reason', '')),
     json.dumps(batch_targets, ensure_ascii=False) if batch_targets else None))
    return conn.execute("SELECT MAX(id) FROM signals").fetchone()[0]


def log_indicators(signal_id, symbol, indicators_dict):
    if indicators_dict is None:
        return
    conn = _get_conn()
    cleaned = {}
    for k, v in indicators_dict.items():
        if hasattr(v, 'item'):
            cleaned[k] = v.item()
        elif isinstance(v, list):
            cleaned[k] = [x.item() if hasattr(x, 'item') else x for x in v]
        elif isinstance(v, (int, float, str, bool, type(None))):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    conn.execute('''INSERT INTO indicator_snapshots (signal_id, symbol, snapshot_json)
                    VALUES (?,?,?)
                    ON CONFLICT (signal_id) DO UPDATE SET symbol=EXCLUDED.symbol, snapshot_json=EXCLUDED.snapshot_json''',
                 (signal_id, symbol, json.dumps(cleaned, ensure_ascii=False)))


def log_local_plan(signal_id, symbol, local_plan):
    if local_plan is None:
        return
    _get_conn().execute('''INSERT INTO local_plan_snapshots (signal_id, symbol, plan_json)
                           VALUES (?,?,?)
                           ON CONFLICT (signal_id) DO UPDATE SET symbol=EXCLUDED.symbol, plan_json=EXCLUDED.plan_json''',
                        (signal_id, symbol, json.dumps(local_plan, ensure_ascii=False, default=str)))


def log_local_signal(local_plan, symbol, indicators=None, daily_df=None):
    if local_plan is None:
        return None
    conn = _get_conn()
    direction = local_plan.get('signal', 'neutral')
    confidence = local_plan.get('confidence', 0.5)
    entry_zone = str(local_plan.get('entry_zone', ''))
    stop_loss = str(local_plan.get('stop_loss', ''))
    position_advice = str(local_plan.get('position_sizing', {}).get('note', ''))
    resonance_detail = '; '.join(local_plan.get('principles', [])[:5])
    weekly_align = str(indicators.get('weekly_trend', '')) if indicators else ''
    targets = local_plan.get('targets', [])
    batch_targets = [{'price': str(t.get('price', '')), 'ratio': str(t.get('ratio', '')),
                       'reason': str(t.get('reason', ''))} for t in targets]
    targets_str = json.dumps(batch_targets, ensure_ascii=False) if batch_targets else ''
    ps = local_plan.get('position_sizing', {})
    fusion_type = str(local_plan.get('fusion_type', ''))
    ergo_protections = json.dumps([
        f"gap_reduce={ps.get('gap_reduce_factor', 1.0):.2f}", f"fusion={fusion_type}"
    ], ensure_ascii=False)
    conn.execute('''INSERT INTO signals (
        symbol, signal, confidence, entry_zone, stop_loss, targets,
        position_advice, resonance_detail, weekly_align, source,
        kelly_position_pct, stop_loss_rule, entry_reason,
        risk_reward_analysis, batch_targets_json, ergo_protections, ergo_safe_to_enter
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
    (symbol, direction, confidence, entry_zone, stop_loss, targets_str,
     position_advice, resonance_detail, weekly_align, 'local',
     ps.get('position_pct', 0), str(local_plan.get('stop_loss_rule', '')),
     str(local_plan.get('entry_rationale', '')),
     f"盈亏比={ps.get('reward_risk_ratio', 0):.1f} 预期值={ps.get('expected_value_pct', 0):.2f}%",
     json.dumps(batch_targets, ensure_ascii=False) if batch_targets else None,
     ergo_protections, 1))
    signal_id = conn.execute("SELECT MAX(id) FROM signals").fetchone()[0]
    if signal_id:
        conn.execute('''INSERT INTO local_plan_snapshots (signal_id, symbol, plan_json)
                        VALUES (?,?,?)
                        ON CONFLICT (signal_id) DO UPDATE SET symbol=EXCLUDED.symbol, plan_json=EXCLUDED.plan_json''',
                     (signal_id, symbol, json.dumps(local_plan, ensure_ascii=False, default=str)))
    return signal_id


def log_deepseek_response(signal_id, raw_text):
    if not raw_text:
        return
    _get_conn().execute('''INSERT INTO deepseek_responses (signal_id, raw_response)
                           VALUES (?,?)
                           ON CONFLICT (signal_id) DO UPDATE SET raw_response=EXCLUDED.raw_response''', (signal_id, raw_text))


def log_run(stocks_total, stocks_scanned, bullish_count, bearish_count, neutral_count, report_path='',
            avg_kelly_pct=None, aligned_count=None, conflict_count=None):
    _get_conn().execute('''INSERT INTO run_history (stocks_total, stocks_scanned, bullish_count,
                           bearish_count, neutral_count, report_path, avg_kelly_pct,
                           aligned_count, conflict_count)
                           VALUES (?,?,?,?,?,?,?,?,?)''',
                        (stocks_total, stocks_scanned, bullish_count, bearish_count, neutral_count,
                         report_path, avg_kelly_pct, aligned_count, conflict_count))


def save_bayesian_calibration(symbol, calibration_dict, n_samples):
    if calibration_dict is None:
        return
    cleaned = {}
    for dim, states in calibration_dict.items():
        if isinstance(states, dict):
            cleaned[dim] = {k: float(v) if k != 'n_samples' else v
                            for k, v in states.items()}
    cleaned['n_samples'] = n_samples
    _get_conn().execute('''INSERT INTO bayesian_calibration_cache
                           (symbol, calibration_json, n_samples, updated_at)
                           VALUES (?,?,?,CURRENT_TIMESTAMP)
                           ON CONFLICT (symbol) DO UPDATE SET calibration_json=EXCLUDED.calibration_json, n_samples=EXCLUDED.n_samples''',
                        (symbol, json.dumps(cleaned, ensure_ascii=False), n_samples))


def load_bayesian_calibration(symbol):
    row = _get_conn().execute(
        'SELECT calibration_json, n_samples, updated_at FROM bayesian_calibration_cache WHERE symbol=?',
        (symbol,)).fetchone()
    if row:
        cal = json.loads(row[0])
        cal['n_samples'] = row[1]
        return cal
    return None


def get_unevaluated_signals():
    rows = _get_conn().execute(
        "SELECT * FROM signals WHERE evaluated=0 AND signal IN ('bullish','bearish')").fetchall()
    cols = [d[0] for d in _get_conn().description]
    return [_row_to_dict(cols, r) for r in rows]


def mark_signal_evaluated(signal_id, pnl, hit_target, stopped_out):
    conn = _get_conn()
    conn.execute("UPDATE signals SET evaluated=1 WHERE id=?", (signal_id,))
    conn.execute('''INSERT INTO outcomes (signal_id, actual_pnl, hit_target, stopped_out)
                    VALUES (?,?,?,?)''', (signal_id, pnl, hit_target, stopped_out))


def get_performance_stats():
    conn = _get_conn()
    by_type = [dict(zip(['signal','source','cnt','avg_pnl','wins','hit_targets'], r))
               for r in conn.execute('''SELECT s.signal, s.source, COUNT(*) as cnt,
                   AVG(o.actual_pnl) as avg_pnl,
                   SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN o.hit_target = 1 THEN 1 ELSE 0 END) as hit_targets
                   FROM signals s JOIN outcomes o ON s.id = o.signal_id
                   GROUP BY s.signal, s.source''').fetchall()]
    overall_row = conn.execute('''SELECT COUNT(*) as total, AVG(o.actual_pnl) as avg_pnl,
        SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) as wins FROM outcomes o''').fetchone()
    overall = dict(zip(['total','avg_pnl','wins'], overall_row)) if overall_row else {}
    calibration = [dict(zip(['conf_bucket','cnt','avg_pnl','win_rate'], r))
                   for r in conn.execute('''SELECT
        CASE WHEN s.confidence >= 0.8 THEN '高置信(≥80%)'
             WHEN s.confidence >= 0.7 THEN '中高置信(70-80%)'
             WHEN s.confidence >= 0.6 THEN '中置信(60-70%)'
             ELSE '低置信(<60%)' END as conf_bucket,
        COUNT(*) as cnt, AVG(o.actual_pnl) as avg_pnl,
        SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate
        FROM signals s JOIN outcomes o ON s.id = o.signal_id
        GROUP BY conf_bucket ORDER BY MIN(s.confidence) DESC''').fetchall()]
    return {'by_type': by_type, 'overall': overall, 'confidence_calibration': calibration}


def get_run_history(limit=20):
    rows = _get_conn().execute(
        "SELECT * FROM run_history ORDER BY run_time DESC LIMIT ?", (limit,)).fetchall()
    cols = [d[0] for d in _get_conn().description]
    return [_row_to_dict(cols, r) for r in rows]


def get_latest_signals(limit=100):
    rows = _get_conn().execute('''SELECT s.*, i.snapshot_json, o.actual_pnl, o.hit_target, o.stopped_out,
        l.plan_json as local_plan_json FROM signals s
        LEFT JOIN indicator_snapshots i ON s.id = i.signal_id
        LEFT JOIN outcomes o ON s.id = o.signal_id
        LEFT JOIN local_plan_snapshots l ON s.id = l.signal_id
        ORDER BY s.timestamp DESC LIMIT ?''', (limit,)).fetchall()
    cols = [d[0] for d in _get_conn().description]
    return [_row_to_dict(cols, r) for r in rows]


def get_confidence_calibration_data():
    rows = _get_conn().execute('''SELECT s.confidence, s.signal, s.bayesian_posterior,
        s.kelly_position_pct, s.bayesian_alignment, o.actual_pnl, o.hit_target, o.stopped_out
        FROM signals s JOIN outcomes o ON s.id = o.signal_id
        WHERE s.signal IN ('bullish','bearish') ORDER BY s.timestamp DESC''').fetchall()
    cols = [d[0] for d in _get_conn().description]
    return [_row_to_dict(cols, r) for r in rows]


def bulk_log_signals(signals_list):
    if not signals_list:
        return
    conn = _get_conn()
    rows = [(s.get('symbol','?'), s.get('signal','neutral'), s.get('confidence',0),
             s.get('entry_zone',''), s.get('stop_loss',''), s.get('targets',''),
             s.get('position_advice',''), s.get('resonance_detail',''),
             s.get('weekly_align',''), s.get('source','ExpertEnsemble')) for s in signals_list]
    conn.executemany('''INSERT INTO signals (symbol, signal, confidence, entry_zone,
        stop_loss, targets, position_advice, resonance_detail, weekly_align, source)
        VALUES (?,?,?,?,?,?,?,?,?,?)''', rows)


def save_backtest_run(run_meta, equity_curve, trades):
    conn = _get_conn()
    run_meta_c = dict(run_meta)
    run_meta_c['config_json'] = json.dumps(run_meta.get('config_json', {}), ensure_ascii=False)
    run_meta_c['equity_curve_json'] = json.dumps(equity_curve, ensure_ascii=False)
    run_meta_c['school_weights_json'] = json.dumps(run_meta.get('school_weights_json', {}), ensure_ascii=False)
    cols = ['run_label','start_date','end_date','initial_capital','position_pct',
            'stop_pct','target_pct','max_positions','total_return_pct','annualized_return_pct',
            'sharpe_ratio','max_drawdown_pct','total_trades','win_rate_pct','avg_win_pct',
            'avg_loss_pct','profit_factor','config_json','equity_curve_json','weight_config','school_weights_json']
    vals = [run_meta_c.get(k) for k in cols]
    conn.execute(f"INSERT INTO backtest_runs ({','.join(cols)}) VALUES ({','.join(['?']*len(cols))})", vals)
    run_id = conn.execute("SELECT MAX(id) FROM backtest_runs").fetchone()[0]
    t_cols = ['run_id','symbol','direction','entry_date','exit_date','entry_price',
              'exit_price','shares','gross_pnl_pct','net_pnl_pct','net_pnl_rmb',
              'exit_reason','holding_days','ensemble_confidence','school_votes_json',
              'shadow_rule_dir','shadow_rule_conf','shadow_ml_dir','shadow_ml_conf',
              'master_source']
    sv_idx = t_cols.index('school_votes_json')  # computed once, O(1) outside loop
    for t in trades:
        t_vals = [run_id] + [t.get(k) for k in t_cols[1:]]
        t_vals[sv_idx] = json.dumps(t_vals[sv_idx]) if not isinstance(t_vals[sv_idx], str) else t_vals[sv_idx]
        conn.execute(f"INSERT INTO backtest_trades ({','.join(t_cols)}) VALUES ({','.join(['?']*len(t_cols))})", t_vals)
    _close_conn()  # release DuckDB write-lock so downstream processes can access the file
    return run_id


def load_backtest_run(run_id):
    """Load a backtest run + its trades using a fresh connection."""
    _close_conn()  # release singleton so new connection can open
    import duckdb
    with duckdb.connect(DB_PATH) as conn:
        run = conn.execute("SELECT * FROM backtest_runs WHERE id=?", (run_id,)).fetchone()
        run_dict = _row_to_dict([d for d in conn.description], run) if run else {}
        trades_rows = conn.execute(
            "SELECT * FROM backtest_trades WHERE run_id=? ORDER BY entry_date", (run_id,)).fetchall()
        trades = [_row_to_dict([d for d in conn.description], r) for r in trades_rows]
        return run_dict, trades


def list_backtest_runs(limit=20):
    """List recent backtest runs using a fresh connection."""
    _close_conn()  # release singleton so new connection can open
    import duckdb
    with duckdb.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT id, run_time, run_label, start_date, end_date, total_return_pct, "
            "sharpe_ratio, max_drawdown_pct, total_trades, win_rate_pct "
            "FROM backtest_runs ORDER BY run_time DESC LIMIT ?",
            (limit,)).fetchall()
        cols = [d[0] for d in conn.description]
        return [_row_to_dict(cols, r) for r in rows]


def save_school_performance(run_id, school_metrics, regime=None):
    conn = _get_conn()
    for school, metrics in school_metrics.items():
        conn.execute('''INSERT INTO school_performance
            (run_id, school_name, regime, window_label, n_signals, n_correct,
             win_rate, avg_pnl_pct, avg_confidence, sharpe_like_score,
             direction_bias, contribution_weight)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)''', (
            run_id, school, regime, metrics.get('window_label',''),
            metrics.get('n_signals',0), metrics.get('n_correct',0),
            metrics.get('win_rate',0), metrics.get('avg_pnl_pct',0),
            metrics.get('avg_confidence',0), metrics.get('sharpe_like_score',0),
            metrics.get('direction_bias',''), metrics.get('contribution_weight',0)))


def save_regime_weights_learned(regime, weights_dict, n_samples):
    conn = _get_conn()
    conn.execute("DELETE FROM regime_weights_learned WHERE regime=?", (regime,))
    conn.execute('''INSERT INTO regime_weights_learned
        (regime, weights_json, n_samples, last_updated)
        VALUES (?,?,?,CURRENT_TIMESTAMP)''',
        (regime, json.dumps(weights_dict, ensure_ascii=False), n_samples))


def load_regime_weights_learned(regime=None):
    conn = _get_conn()
    if regime:
        rows = conn.execute(
            "SELECT weights_json, n_samples FROM regime_weights_learned WHERE regime=? AND n_samples>=5",
            (regime,)).fetchall()
    else:
        rows = conn.execute(
            "SELECT regime, weights_json, n_samples FROM regime_weights_learned WHERE n_samples>=5").fetchall()
    if not rows:
        return None
    if regime:
        return json.loads(rows[0][0])
    return {r[0]: json.loads(r[1]) for r in rows}
