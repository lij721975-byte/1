# backtest_feedback.py
"""
Historical signal backtesting and performance evaluation.

SURVIVORSHIP BIAS WARNING:
Backtest results are upward-biased due to survivorship bias — the stock pool
consists of currently-listed stocks only. Stocks that were delisted, suspended,
or went bankrupt during the backtest period are excluded, which inflates
win rates and average returns. Interpret historical performance with caution.
"""
import json
import logging
import re
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import get_daily_kline
from config import DB_PATH, MIN_LOOKBACK_DAYS, DEFAULT_STOP_PCT, DEFAULT_TARGET_PCT

logger = logging.getLogger(__name__)

# A-share trading costs
STAMP_DUTY = 0.001       # 印花税 0.1% (sell only)
COMMISSION = 0.00025     # 佣金 0.025% (per side)
ROUND_TRIP_COST_PCT = (STAMP_DUTY + 2 * COMMISSION) * 100  # ~0.15% round-trip


def _parse_timestamp(ts_str):
    """健壮的时间戳解析，支持多种格式"""
    if not ts_str:
        return None
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%dT%H:%M:%S.%f',
        '%Y-%m-%d',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    # 最后尝试 ISO 8601 格式（如 2025-06-15T14:30:00+08:00）
    try:
        from datetime import timezone
        ts_str_clean = re.sub(r'([+-]\d{2}):(\d{2})$', r'\1\2', ts_str)
        for fmt in ['%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%dT%H:%M:%S.%f%z']:
            try:
                return datetime.strptime(ts_str_clean, fmt)
            except ValueError:
                continue
    except Exception:
        pass
    return None


def _parse_target_prices(sig):
    """从信号中解析实际目标价列表（兼容多种输入格式）"""
    targets = []

    # 1. batch_targets: Python list 或 JSON字符串
    bt = sig.get('batch_targets') or sig.get('batch_targets_json')
    if bt is not None:
        if isinstance(bt, str):
            try:
                bt = json.loads(bt)
            except json.JSONDecodeError:
                bt = None
        if isinstance(bt, list):
            for t in bt:
                if isinstance(t, dict):
                    price = t.get('price')
                    if price is not None:
                        try:
                            targets.append(float(price))
                        except (ValueError, TypeError):
                            pass

    # 2. 旧格式 targets: 逗号分隔字符串
    if not targets:
        old_targets = sig.get('targets', '')
        if old_targets:
            if isinstance(old_targets, str):
                # 先尝试JSON解析
                try:
                    parsed = json.loads(old_targets)
                    if isinstance(parsed, list):
                        for t in parsed:
                            if isinstance(t, dict):
                                try:
                                    targets.append(float(t['price']))
                                except (ValueError, KeyError, TypeError):
                                    pass
                            elif isinstance(t, (int, float)):
                                targets.append(float(t))
                    else:
                        raise json.JSONDecodeError('not list', '', 0)
                except (json.JSONDecodeError, ValueError):
                    # 逗号分隔
                    for part in old_targets.split(','):
                        try:
                            targets.append(float(part.strip()))
                        except ValueError:
                            nums = re.findall(r'[\d]+\.?[\d]*', part)
                            if nums:
                                targets.append(float(nums[0]))
            elif isinstance(old_targets, list):
                for t in old_targets:
                    try:
                        targets.append(float(t) if isinstance(t, (int, float)) else float(str(t)))
                    except (ValueError, TypeError):
                        pass

    return targets


def _is_limit_locked(df_row, direction):
    """
    Check if a bar is limit-locked (unexecutable entry).

    A-shares have ±10% limits (±20% for 科创/创业板). A bar is "locked"
    when the close price is at (or very near) the daily limit:
    - For bullish entry: close ≈ high means limit-up locked (cannot buy)
    - For bearish entry: close ≈ low means limit-down locked (cannot sell)

    Uses 0.1% tolerance to account for rounding/ticks.
    """
    bar_high = float(df_row['high'])
    bar_low = float(df_row['low'])
    bar_close = float(df_row['close'])

    if bar_high <= 0:
        return False, None

    if direction == 'bullish':
        if bar_close >= bar_high * 0.999:  # within 0.1% of high
            return True, 'limit_up_locked'
    else:
        if bar_close <= bar_low * 1.001 and bar_low > 0:  # within 0.1% of low
            return True, 'limit_down_locked'
    return False, None


def replay_bars_for_exit(df, entry_idx, entry_open, direction,
                         stop_price, actual_targets, fallback_target,
                         trailing_stop_enabled=True,
                         trailing_activation=0.05,
                         trailing_atr_mult=2.0):
    """
    Replay future bars from entry_idx to find exit point.

    Shared function used by both evaluate_past_signals() and backtest_engine.py.

    Args:
        df: DataFrame with DatetimeIndex, columns [open, high, low, close]
        entry_idx: pandas index label of the entry bar
        entry_open: actual entry price (T+1 next-day open)
        direction: 'bullish' or 'bearish'
        stop_price: stop-loss trigger price
        actual_targets: sorted list of target prices (ascending for bullish, descending for bearish)
        fallback_target: default target if actual_targets is empty
        trailing_stop_enabled: enable ATR-based trailing stop
        trailing_activation: profit % to activate trailing stop (e.g. 0.05 = 5%)
        trailing_atr_mult: ATR multiplier for trailing distance

    Returns dict with: exit_price, hit_target, stopped_out, gap_stopped, limit_hit,
                       trailing_stopped, eval_code, target_hit_price, stop_hit_price
    """
    # T+1 sell rule: cannot sell on the entry bar. Start from the next bar.
    future_data = df[df.index > entry_idx]
    exit_price = None
    hit_target = 0
    stopped_out = 0
    gap_stopped = 0
    limit_hit = 0
    trailing_stopped = 0
    target_hit_price = None
    stop_hit_price = None
    eval_code = 1

    # Trailing stop state
    trailing_active = False
    trailing_high = entry_open
    trailing_stop_price = stop_price

    for bar_idx, bar in future_data.iterrows():
        bar_open = float(bar['open'])
        bar_high = float(bar['high'])
        bar_low = float(bar['low'])
        bar_close = float(bar['close'])

        if direction == 'bullish':
            # 涨停检测: 涨停当天不卖, 跳过今日退出判断
            limit_up_pct = (bar_close - bar_open) / (bar_open + 1e-9)
            is_limit_up = (limit_up_pct > 0.095) or (bar_high == bar_close and limit_up_pct > 0.07)

            # 6% 回撤止损 (intraday drawdown from entry)
            drawdown_pct = (entry_open - bar_low) / (entry_open + 1e-9)

            if drawdown_pct >= 0.06:
                stopped_out = 1
                stop_hit_price = min(entry_open * 0.94, bar_open)
                exit_price = stop_hit_price
                eval_code = 2; break

            if not is_limit_up:
                if bar_open <= stop_price:
                    stopped_out = 1; gap_stopped = 1
                    stop_hit_price = bar_open; exit_price = bar_open
                    eval_code = 2; break

                if bar_low <= stop_price:
                    stopped_out = 1
                    stop_hit_price = stop_price; exit_price = stop_price
                    break

            if bar_idx == entry_idx:
                if not is_limit_up:
                    locked, _ = _is_limit_locked(bar, direction)
                    if locked:
                        limit_hit = 1; eval_code = 3
                        exit_price = entry_open; break

            # --- Trailing stop ---
            if trailing_stop_enabled and not is_limit_up:
                # Update trailing high
                if bar_high > trailing_high:
                    trailing_high = bar_high
                profit_pct = (trailing_high - entry_open) / entry_open

                # Activate trailing stop after profit exceeds activation threshold
                if profit_pct >= trailing_activation:
                    trailing_active = True

                if trailing_active:
                    # Compute ATR (simple estimate from bar range)
                    bar_atr = bar_high - bar_low
                    trailing_stop_price = trailing_high - trailing_atr_mult * max(bar_atr, entry_open * 0.01)
                    # Trail only upward
                    trailing_stop_price = max(trailing_stop_price, stop_price)

                # Check trailing stop hit
                if trailing_active and bar_low <= trailing_stop_price:
                    trailing_stopped = 1
                    exit_price = trailing_stop_price
                    eval_code = 2; break

            if not is_limit_up:
                target_hit_this_bar = False
                if actual_targets:
                    for tp in sorted(actual_targets):
                        if tp > entry_open and bar_close >= tp:
                            hit_target = 1
                            target_hit_price = tp; exit_price = tp
                            target_hit_this_bar = True; break
                if not target_hit_this_bar and bar_close >= fallback_target:
                    hit_target = 1; exit_price = fallback_target
                if hit_target:
                    break

        else:  # bearish
            if bar_open >= stop_price:
                stopped_out = 1; gap_stopped = 1
                stop_hit_price = bar_open; exit_price = bar_open
                eval_code = 2; break

            if bar_high >= stop_price:
                stopped_out = 1
                stop_hit_price = stop_price; exit_price = stop_price
                break

            if bar_idx == entry_idx:
                locked, _ = _is_limit_locked(bar, direction)
                if locked:
                    limit_hit = 1; eval_code = 3
                    exit_price = entry_open; break

            target_hit_this_bar = False
            if actual_targets:
                for tp in sorted(actual_targets, reverse=True):
                    if tp < entry_open and bar_close <= tp:
                        hit_target = 1
                        target_hit_price = tp; exit_price = tp
                        target_hit_this_bar = True; break
            if not target_hit_this_bar and bar_close <= fallback_target:
                hit_target = 1; exit_price = fallback_target
            if hit_target:
                break

    return {
        'exit_price': exit_price,
        'hit_target': hit_target,
        'stopped_out': stopped_out,
        'gap_stopped': gap_stopped,
        'limit_hit': limit_hit,
        'trailing_stopped': trailing_stopped,
        'eval_code': eval_code,
        'target_hit_price': target_hit_price,
        'stop_hit_price': stop_hit_price,
    }


def evaluate_past_signals():
    """
    评估所有未评估的历史信号（基于最新行情）。

    改进：
    - 解析信号中实际的目标价/止损价而非硬编码5%
    - 健壮的时间戳解析
    - 记录更多评估维度
    - 涨跌停封板检测（涨停买不到/跌停卖不出）
    - 生存偏差过滤（数据不足的股票跳过）
    """
    conn = duckdb.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM signals WHERE evaluated=0 AND signal IN ('bullish','bearish')")
    signals = c.fetchall()

    evaluated_count = 0
    skipped_count = 0
    results = []

    for row in signals:
        sig = dict(row)
        sig_id = sig['id']
        symbol = sig['symbol']
        direction = sig['signal']

        sig_time = _parse_timestamp(sig.get('timestamp', ''))
        if sig_time is None:
            c.execute("UPDATE signals SET evaluated=9 WHERE id=?", (sig_id,))
            skipped_count += 1
            continue

        # 跳过最近24小时内的信号（仍然未评估，等下次）
        if (datetime.now() - sig_time).total_seconds() < 86400:
            continue

        # 获取最新日线数据
        try:
            df = get_daily_kline(symbol, 200)
            if df is None or df.empty:
                c.execute("UPDATE signals SET evaluated=9 WHERE id=?", (sig_id,))
                skipped_count += 1
                continue

            df.index = pd.to_datetime(df.index)

            # 生存偏差过滤：数据不足的股票跳过
            if len(df) < MIN_LOOKBACK_DAYS:
                logger.warning(
                    "Skipping %s (id=%s): only %d bars available, need %d (survivorship filter)",
                    symbol, sig_id, len(df), MIN_LOOKBACK_DAYS
                )
                c.execute("UPDATE signals SET evaluated=9 WHERE id=?", (sig_id,))
                skipped_count += 1
                continue

            # T+1: find first trading day strictly AFTER signal time, enter at OPEN
            entry_idx = None
            entry_price = None
            entry_open = None
            for idx in sorted(df.index):
                if idx > sig_time:
                    entry_idx = idx
                    entry_open = float(df.loc[idx, 'open'])
                    entry_price = float(df.loc[idx, 'close'])
                    break

            if entry_price is None or entry_price <= 0:
                continue

            current_price = float(df['close'].iloc[-1])

            # === 解析目标价和止损价 ===
            actual_targets = _parse_target_prices(sig)

            stop_loss_str = str(sig.get('stop_loss', '')).strip()
            try:
                stop_price = float(stop_loss_str)
            except (ValueError, TypeError):
                stop_price = None

            if not (stop_price and stop_price > 0):
                stop_price = entry_open * (1.0 - DEFAULT_STOP_PCT) if direction == 'bullish' else entry_open * (1.0 + DEFAULT_STOP_PCT)
            fallback_target = entry_open * (1.0 + DEFAULT_TARGET_PCT) if direction == 'bullish' else entry_open * (1.0 - DEFAULT_TARGET_PCT)

            # 验证：止损幅度不应大于止盈幅度（否则预期收益为负）
            if DEFAULT_STOP_PCT > DEFAULT_TARGET_PCT:
                logger.warning(
                    "Default stop (%.1f%%) > target (%.1f%%): negative expectancy for %s (id=%s)",
                    DEFAULT_STOP_PCT * 100, DEFAULT_TARGET_PCT * 100, symbol, sig_id
                )

            # === 逐K线路径回放（实盘约束）—— 调用共享函数 ===
            replay = replay_bars_for_exit(
                df, entry_idx, entry_open, direction,
                stop_price, actual_targets, fallback_target
            )
            exit_price = replay['exit_price']
            hit_target = replay['hit_target']
            stopped_out = replay['stopped_out']
            gap_stopped = replay['gap_stopped']
            limit_hit = replay['limit_hit']
            eval_code = replay['eval_code']
            target_hit_price = replay['target_hit_price']
            stop_hit_price = replay['stop_hit_price']

            # === P&L计算 ===
            if exit_price is not None and exit_price > 0:
                if direction == 'bullish':
                    gross_pnl = (exit_price - entry_open) / entry_open * 100
                else:
                    gross_pnl = (entry_open - exit_price) / entry_open * 100
            else:
                # 仍未平仓：用当前价估算
                if direction == 'bullish':
                    gross_pnl = (current_price - entry_open) / entry_open * 100
                else:
                    gross_pnl = (entry_open - current_price) / entry_open * 100

            pnl_pct = gross_pnl - ROUND_TRIP_COST_PCT

            # 写入结果（含扩展字段）
            c.execute("UPDATE signals SET evaluated=? WHERE id=?", (eval_code, sig_id,))
            c.execute('''INSERT INTO outcomes (signal_id, actual_pnl, hit_target, stopped_out)
                         VALUES (?,?,?,?)''',
                      (sig_id, round(pnl_pct, 2), hit_target, stopped_out))

            evaluated_count += 1
            days = (datetime.now() - sig_time).days
            results.append({
                'symbol': symbol,
                'direction': direction,
                'pnl': round(pnl_pct, 2),
                'hit_target': hit_target,
                'stopped_out': stopped_out,
                'gap_stopped': gap_stopped,
                'limit_hit': limit_hit,
                'days': days,
                'entry_price': entry_open,
                'exit_price': exit_price,
                'current_price': current_price,
                'confidence': sig.get('confidence', 0),
                'targets_used': actual_targets if actual_targets else ['5%_estimate'],
            })

            if evaluated_count % 5 == 0:
                conn.commit()

        except Exception as e:
            logger.error(
                "Signal evaluation failed for id=%s symbol=%s direction=%s: %s",
                sig_id, symbol, direction, e, exc_info=True
            )
            try:
                c.execute("UPDATE signals SET evaluated=9 WHERE id=?", (sig_id,))
            except Exception:
                pass
            skipped_count += 1
            continue

    try:
        conn.commit()
    finally:
        conn.close()

    stats = _compute_eval_stats(results)
    stats['evaluated_count'] = evaluated_count
    return stats


def _compute_eval_stats(results):
    """计算评估统计（含置信度分档校准分析）"""
    if not results:
        return {'total': 0, 'win_rate': 0, 'avg_pnl': 0, 'results': [],
                'confidence_calibration': []}

    total = len(results)
    wins = sum(1 for r in results if r['pnl'] > 0)
    avg_pnl = sum(r['pnl'] for r in results) / total if total > 0 else 0
    hit_targets = sum(1 for r in results if r['hit_target'])
    stopped = sum(1 for r in results if r['stopped_out'])

    # 置信度校准分析：按置信度分档统计实际胜率
    conf_buckets = {'高置信(≥80%)': (0.80, 1.01),
                    '中高置信(70-80%)': (0.70, 0.80),
                    '中置信(60-70%)': (0.60, 0.70),
                    '低置信(<60%)': (0.00, 0.60)}

    calibration = []
    for label, (lo, hi) in conf_buckets.items():
        bucket = [r for r in results if lo <= r.get('confidence', 0) < hi]
        if bucket:
            b_wins = sum(1 for r in bucket if r['pnl'] > 0)
            b_avg = sum(r['pnl'] for r in bucket) / len(bucket)
            calibration.append({
                'bucket': label,
                'count': len(bucket),
                'win_rate': round(b_wins / len(bucket) * 100, 1),
                'avg_pnl': round(b_avg, 2),
                'expected_confidence': round((lo + hi) / 2 * 100),
            })

    return {
        'total': total,
        'wins': wins,
        'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
        'avg_pnl': round(avg_pnl, 2),
        'hit_targets': hit_targets,
        'stopped_out': stopped,
        'results': results,
        'confidence_calibration': calibration,
    }


def get_historical_stats():
    """获取历史全部评估统计（含置信度校准 + AI vs 本地对比）"""
    conn = duckdb.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 按信号类型分区
    c.execute('''SELECT s.signal, COUNT(*) as cnt,
                 ROUND(AVG(o.actual_pnl), 2) as avg_pnl,
                 SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) as wins,
                 SUM(CASE WHEN o.hit_target = 1 THEN 1 ELSE 0 END) as hit_targets,
                 SUM(CASE WHEN o.stopped_out = 1 THEN 1 ELSE 0 END) as stopped
                 FROM signals s
                 JOIN outcomes o ON s.id = o.signal_id
                 GROUP BY s.signal''')
    by_type = [dict(r) for r in c.fetchall()]

    # 总体
    c.execute('''SELECT COUNT(*) as total,
                 ROUND(AVG(o.actual_pnl), 2) as avg_pnl,
                 SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) as wins
                 FROM outcomes o''')
    row = c.fetchone()
    overall = dict(row) if row else {'total': 0, 'avg_pnl': 0, 'wins': 0}

    # 置信度校准分档
    c.execute('''SELECT
                 CASE
                   WHEN s.confidence >= 0.80 THEN '高置信(≥80%)'
                   WHEN s.confidence >= 0.70 THEN '中高置信(70-80%)'
                   WHEN s.confidence >= 0.60 THEN '中置信(60-70%)'
                   ELSE '低置信(<60%)'
                 END as conf_bucket,
                 COUNT(*) as cnt,
                 ROUND(AVG(o.actual_pnl), 2) as avg_pnl,
                 ROUND(SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate
                 FROM signals s
                 JOIN outcomes o ON s.id = o.signal_id
                 GROUP BY conf_bucket
                 ORDER BY MIN(s.confidence) DESC''')
    calibration = [dict(r) for r in c.fetchall()]

    # AI vs 本地模型对比
    ai_vs_local = _compute_ai_vs_local(c)

    # 贝叶斯对齐准确度
    bayes_perf = _compute_bayes_alignment_perf(c)

    try:
        return {
            'by_type': by_type,
            'overall': overall,
            'confidence_calibration': calibration,
            'ai_vs_local': ai_vs_local,
            'bayes_alignment_perf': bayes_perf,
        }
    finally:
        conn.close()


def _compute_ai_vs_local(c):
    """对比AI信号与本地模型的准确度（优先用source列直接对比，回退到JOIN方式）"""
    try:
        # 方法1: 用source列直接对比（本地模型有独立信号行时）
        c.execute('''SELECT s.source, s.signal, o.actual_pnl, o.hit_target, o.stopped_out
                     FROM signals s
                     JOIN outcomes o ON s.id = o.signal_id
                     WHERE s.signal IN ('bullish','bearish')
                       AND s.source IN ('DeepSeek', 'local')''')
        source_rows = [dict(r) for r in c.fetchall()]

        ai_rows = [r for r in source_rows if r['source'] == 'DeepSeek']
        local_rows = [r for r in source_rows if r['source'] == 'local']

        if len(ai_rows) >= 3 and len(local_rows) >= 3:
            def _calc_accuracy(rows):
                if not rows:
                    return 0, 0
                correct = sum(1 for r in rows if (r['actual_pnl'] or 0) > 0)
                return correct, len(rows)

            ai_correct, ai_total = _calc_accuracy(ai_rows)
            local_correct, local_total = _calc_accuracy(local_rows)

            # 注: 配对样本分析已移除（原为死代码，local_rows无symbol字段无法配对）
            return {
                'total': max(ai_total, local_total),
                'ai_accuracy': round(ai_correct / ai_total * 100, 1) if ai_total > 0 else 0,
                'local_accuracy': round(local_correct / local_total * 100, 1) if local_total > 0 else 0,
                'ai_evaluated': ai_total,
                'local_evaluated': local_total,
                'comparison_method': 'source_column',
                'both_correct_pct': None,
                'both_wrong_pct': None,
            }

        # 方法2: 回退到JOIN方式（AI信号 + local_plan_snapshots配对）
        c.execute('''SELECT s.signal as ai_signal, s.confidence,
                     l.plan_json, o.actual_pnl, o.hit_target, o.stopped_out
                     FROM signals s
                     JOIN outcomes o ON s.id = o.signal_id
                     JOIN local_plan_snapshots l ON s.id = l.signal_id
                     WHERE s.signal IN ('bullish','bearish')''')
        rows = [dict(r) for r in c.fetchall()]
        if not rows:
            return None

        ai_correct = 0
        local_correct = 0
        both_correct = 0
        both_wrong = 0
        total = len(rows)

        for r in rows:
            ai_right = (r['ai_signal'] == 'bullish' and (r['actual_pnl'] or 0) > 0) or \
                       (r['ai_signal'] == 'bearish' and (r['actual_pnl'] or 0) > 0)
            try:
                plan = json.loads(r['plan_json']) if isinstance(r['plan_json'], str) else (r['plan_json'] or {})
            except (json.JSONDecodeError, TypeError):
                plan = {}
            local_sig = plan.get('signal', 'neutral')
            local_right = (local_sig == 'bullish' and (r['actual_pnl'] or 0) > 0) or \
                          (local_sig == 'bearish' and (r['actual_pnl'] or 0) > 0)

            if ai_right: ai_correct += 1
            if local_right: local_correct += 1
            if ai_right and local_right: both_correct += 1
            if not ai_right and not local_right: both_wrong += 1

        return {
            'total': total,
            'ai_accuracy': round(ai_correct / total * 100, 1) if total > 0 else 0,
            'local_accuracy': round(local_correct / total * 100, 1) if total > 0 else 0,
            'both_correct_pct': round(both_correct / total * 100, 1) if total > 0 else 0,
            'both_wrong_pct': round(both_wrong / total * 100, 1) if total > 0 else 0,
            'comparison_method': 'join_snapshot',
        }
    except Exception:
        return None


def _compute_bayes_alignment_perf(c):
    """分析贝叶斯对齐状态与最终收益的关系"""
    try:
        c.execute('''SELECT s.bayesian_alignment, COUNT(*) as cnt,
                     ROUND(AVG(o.actual_pnl), 2) as avg_pnl,
                     ROUND(SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as win_rate
                     FROM signals s
                     JOIN outcomes o ON s.id = o.signal_id
                     WHERE s.bayesian_alignment IS NOT NULL
                     GROUP BY s.bayesian_alignment''')
        rows = [dict(r) for r in c.fetchall()]
        return rows if rows else None
    except Exception:
        return None


def get_calibration_factors():
    """
    从历史回测数据计算置信度校准因子。

    原理：如果历史"高置信(≥80%)"信号的胜率只有65%，则校准因子=65%/80%=0.81。
    所有专家置信度应乘以该校准因子以纠正系统性高估。

    Returns:
        dict: {
            'calibration_factor': float,  # 全局置信度乘数
            'calibrated': bool,           # 是否有足够数据进行校准
            'total_samples': int,         # 用于校准的样本数
            'buckets': list,              # 各分桶校准详情
        }
    """
    conn = duckdb.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute('''SELECT
                 CASE
                   WHEN s.confidence >= 0.80 THEN 0.85
                   WHEN s.confidence >= 0.70 THEN 0.75
                   WHEN s.confidence >= 0.60 THEN 0.65
                   ELSE 0.55
                 END as expected_conf,
                 COUNT(*) as cnt,
                 ROUND(SUM(CASE WHEN o.actual_pnl > 0 THEN 1 ELSE 0 END) * 100.0 / MAX(COUNT(*), 1), 1) as win_rate
                 FROM signals s
                 JOIN outcomes o ON s.id = o.signal_id
                 WHERE s.confidence IS NOT NULL
                 GROUP BY expected_conf
                 ORDER BY expected_conf DESC''')
    rows = c.fetchall()
    conn.close()

    if not rows or sum(r['cnt'] for r in rows) < 10:
        return {'calibration_factor': 1.0, 'calibrated': False,
                'total_samples': sum(r['cnt'] for r in rows) if rows else 0,
                'buckets': [], 'note': '数据不足（需≥10条评估记录），使用默认置信度'}

    # 计算加权校准因子
    total_weighted_ratio = 0.0
    total_samples = 0
    buckets = []
    for r in rows:
        cnt = r['cnt']
        expected = r['expected_conf']
        actual = r['win_rate'] / 100.0
        if cnt > 0 and expected > 0:
            ratio = min(actual / expected, 1.0)  # 不放大，只收缩
            total_weighted_ratio += ratio * cnt
            total_samples += cnt
            buckets.append({
                'expected_conf': round(expected, 2),
                'count': cnt,
                'actual_win_rate': r['win_rate'],
                'ratio': round(ratio, 3),
            })

    calibration_factor = round(total_weighted_ratio / max(total_samples, 1), 3)
    calibration_factor = max(0.3, min(1.0, calibration_factor))  # 限制在[0.3, 1.0]

    return {
        'calibration_factor': calibration_factor,
        'calibrated': True,
        'total_samples': total_samples,
        'buckets': buckets,
        'note': f'基于{total_samples}条历史数据的校准因子={calibration_factor:.3f}',
    }


def get_engine_trust_scores(lookback=30, recency_halflife=15):
    """
    基于历史回测数据，计算AI引擎与本地引擎的动态信任分。

    使用指数衰减权重，近期表现权重更大。
    返回每引擎的信任分(0-1)和诊断信息。

    Returns:
        {
            'ai_trust': float,       # AI引擎信任分(0-1)
            'local_trust': float,    # 本地引擎信任分(0-1)
            'ai_samples': int,       # AI评估样本数
            'local_samples': int,    # 本地评估样本数
            'fusion_weight_local': float,  # 本地模型融合权重
            'trust_ratio': float,    # local_trust / max(ai_trust, 0.01)
            'recommendation': str,   # 自适应建议
        }
    """
    conn = duckdb.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # 用source列直接查询各引擎的近期表现
    c.execute('''SELECT s.source, s.signal, s.timestamp,
                 o.actual_pnl, o.hit_target, o.stopped_out
                 FROM signals s
                 JOIN outcomes o ON s.id = o.signal_id
                 WHERE s.signal IN ('bullish','bearish')
                   AND s.source IN ('DeepSeek', 'local')
                 ORDER BY o.evaluated_at DESC
                 LIMIT ?''', (lookback * 3,))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()

    if not rows:
        return {
            'ai_trust': 0.5, 'local_trust': 0.5,
            'ai_samples': 0, 'local_samples': 0,
            'fusion_weight_local': 0.5, 'trust_ratio': 1.0,
            'recommendation': '无历史数据，默认等权',
        }

    # 指数衰减权重
    now = datetime.now()
    ai_weighted_wins = 0.0
    ai_weight_sum = 0.0
    local_weighted_wins = 0.0
    local_weight_sum = 0.0
    ai_count = 0
    local_count = 0

    for i, r in enumerate(rows):
        # 时间衰减
        ts = _parse_timestamp(r.get('timestamp', ''))
        if ts:
            age_days = max(0, (now - ts).total_seconds() / 86400)
        else:
            age_days = i  # fallback: use position in list

        decay = np.exp(-np.log(2) * age_days / recency_halflife)
        win = 1.0 if (r['actual_pnl'] or 0) > 0 else 0.0
        source = r.get('source', 'DeepSeek')

        if source == 'local':
            local_weighted_wins += win * decay
            local_weight_sum += decay
            local_count += 1
        else:
            ai_weighted_wins += win * decay
            ai_weight_sum += decay
            ai_count += 1

    ai_trust = ai_weighted_wins / ai_weight_sum if ai_weight_sum > 0 else 0.5
    local_trust = local_weighted_wins / local_weight_sum if local_weight_sum > 0 else 0.5

    # 贝叶斯收缩：样本少时向0.5回归
    ai_trust = (ai_trust * ai_weight_sum + 0.5 * 5) / (ai_weight_sum + 5)
    local_trust = (local_trust * local_weight_sum + 0.5 * 5) / (local_weight_sum + 5)

    # 本地融合权重
    trust_ratio = local_trust / max(ai_trust, 0.01)
    # sigmoid: 将trust_ratio映射到[0.3, 0.7]区间的融合权重
    fusion_weight_local = 0.5 + 0.2 * np.tanh((trust_ratio - 1.0) * 2.0)

    # 诊断建议
    if trust_ratio > 1.3:
        recommendation = '本地模型近期显著优于AI，冲突时倾向本地模型'
    elif trust_ratio > 1.1:
        recommendation = '本地模型略优于AI，轻微倾向本地'
    elif trust_ratio < 0.7:
        recommendation = 'AI近期显著优于本地，冲突时倾向AI'
    elif trust_ratio < 0.9:
        recommendation = 'AI略优于本地，轻微倾向AI'
    else:
        recommendation = '两引擎表现接近，冲突时大幅降置信度观望'

    return {
        'ai_trust': round(ai_trust, 3),
        'local_trust': round(local_trust, 3),
        'ai_samples': ai_count,
        'local_samples': local_count,
        'fusion_weight_local': round(fusion_weight_local, 3),
        'trust_ratio': round(trust_ratio, 2),
        'recommendation': recommendation,
    }


def get_school_trust_scores(lookback=60, recency_halflife=20):
    """
    基于历史回测数据，计算各学派的近期准确率。
    Nüwa路由器使用此函数自适应调整学派投票权重。

    DB尚未记录学派级别来源，因此基于全局胜率基线 + 体制适配
    启发式调整。当学派级别追踪上线后，可直接从DB读取准确率。

    Returns:
        {school_name: trust_score (0-1), ...}
    """
    conn = duckdb.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Query recent outcomes
    c.execute('''SELECT s.signal, s.timestamp, o.actual_pnl, o.hit_target, o.stopped_out
                 FROM signals s
                 JOIN outcomes o ON s.id = o.signal_id
                 WHERE s.signal IN ('bullish','bearish')
                 ORDER BY o.evaluated_at DESC
                 LIMIT ?''', (lookback * 2,))
    rows = [dict(r) for r in c.fetchall()]

    # Also query recent indicator snapshots to estimate current market regime
    ind_rows = []
    try:
        c.execute('''SELECT s.symbol, s.timestamp, ind.indicator_json
                     FROM signals s
                     JOIN indicator_snapshots ind ON s.id = ind.signal_id
                     WHERE s.evaluated != 9
                     ORDER BY s.timestamp DESC LIMIT 10''')
        ind_rows = [dict(r) for r in c.fetchall()]
    except Exception:
        ind_rows = []  # Table may not exist or have different schema
    conn.close()

    schools = [
        'school_chanlun', 'school_tang',
        'school_livermore', 'school_busch', 'school_classical', 'school_risk',
        'school_gann', 'school_wyckoff',
        'school_harmonic',
        'school_roc_breakout',
        'school_volume_profile', 'school_fusion',
        'school_mean_reversion', 'school_capital_flow',
        'school_pattern_features', 'school_brooks_pa',
    ]
    scores = {s: 0.5 for s in schools}

    if not rows or len(rows) < 5:
        return scores

    # Global baseline accuracy with exponential recency weighting
    now = datetime.now()
    weighted_wins = 0.0
    weight_sum = 0.0

    for i, r in enumerate(rows):
        ts = _parse_timestamp(r.get('timestamp', ''))
        if ts:
            age_days = max(0, (now - ts).total_seconds() / 86400)
        else:
            age_days = i
        decay = np.exp(-np.log(2) * age_days / recency_halflife)
        win = 1.0 if (r['actual_pnl'] or 0) > 0 else 0.0
        weighted_wins += win * decay
        weight_sum += decay

    baseline_accuracy = weighted_wins / weight_sum if weight_sum > 0 else 0.5
    # Bayesian shrinkage toward 0.5
    baseline_accuracy = (baseline_accuracy * weight_sum + 0.5 * 10) / (weight_sum + 10)

    # Estimate current market regime from recent indicator snapshots
    regime_adj = _estimate_regime_from_snapshots(ind_rows)

    # Regime-conditional per-school adjustments (±3-8%)
    # Trending: trend-following schools get bonus, reversal/range schools get penalty
    trending_schools = {'school_classical', 'school_gann',
                        'school_chanlun', 'school_roc_breakout',
                        'school_brooks_pa', 'school_livermore'}
    ranging_schools = {'school_tang', 'school_wyckoff', 'school_harmonic',
                       'school_volume_profile', 'school_fusion',
                       'school_mean_reversion', 'school_capital_flow',
                       'school_pattern_features'}
    risk_schools = {'school_risk', 'school_busch'}

    if regime_adj == 'trending':
        for s in schools:
            if s in trending_schools:
                scores[s] = round(min(0.95, baseline_accuracy + 0.05), 3)
            elif s in ranging_schools:
                scores[s] = round(max(0.05, baseline_accuracy - 0.03), 3)
            else:
                scores[s] = round(baseline_accuracy, 3)
    elif regime_adj == 'ranging':
        for s in schools:
            if s in ranging_schools:
                scores[s] = round(min(0.95, baseline_accuracy + 0.05), 3)
            elif s in trending_schools:
                scores[s] = round(max(0.05, baseline_accuracy - 0.03), 3)
            else:
                scores[s] = round(baseline_accuracy, 3)
    elif regime_adj == 'volatile':
        for s in schools:
            if s in risk_schools:
                scores[s] = round(min(0.95, baseline_accuracy + 0.08), 3)
            elif s in trending_schools:
                scores[s] = round(max(0.05, baseline_accuracy - 0.04), 3)
            else:
                scores[s] = round(max(0.05, baseline_accuracy - 0.02), 3)
    else:
        for s in schools:
            scores[s] = round(baseline_accuracy, 3)

    return scores


def _estimate_regime_from_snapshots(ind_rows):
    """Estimate dominant market regime from recent indicator snapshots."""
    if not ind_rows:
        return 'unknown'
    adx_vals = []
    for r in ind_rows:
        try:
            j = json.loads(r['indicator_json']) if isinstance(r['indicator_json'], str) else (r['indicator_json'] or {})
            adx = j.get('dmi_adx', 20)
            if adx is not None:
                adx_vals.append(float(adx))
        except Exception:
            continue
    if not adx_vals:
        return 'unknown'
    avg_adx = np.mean(adx_vals)
    if avg_adx > 25:
        return 'trending'
    elif avg_adx < 18:
        return 'ranging'
    # Check for high volatility in recent snapshots
    rv_high = 0
    for r in ind_rows:
        try:
            j = json.loads(r['indicator_json']) if isinstance(r['indicator_json'], str) else (r['indicator_json'] or {})
            rv_pct = j.get('rv_percentile', 0.5)
            if rv_pct is not None and float(rv_pct) > 0.85:
                rv_high += 1
        except Exception:
            continue
    if rv_high >= len(ind_rows) * 0.3:
        return 'volatile'
    return 'transitional'


def get_self_learning_report():
    """
    自学习报告：综合所有历史数据给出系统改进建议
    包含：置信度校准曲线、AI vs 本地准确度、贝叶斯对齐效力、最佳仓位区间
    """
    stats = get_historical_stats()
    if not stats or stats['overall'].get('total', 0) < 5:
        return {'status': 'insufficient_data', 'message': '历史评估数据不足（需≥5条），继续积累中...'}

    report = {
        'status': 'ok',
        'total_evaluated': stats['overall']['total'],
        'overall_win_rate': round(stats['overall']['wins'] / stats['overall']['total'] * 100, 1) if stats['overall']['total'] > 0 else 0,
    }

    # 1. 置信度是否校准良好？
    cal = stats.get('confidence_calibration', [])
    if cal:
        # 理想情况：高置信 → 高胜率，单调递减
        win_rates = [c.get('win_rate', 0) for c in cal]
        is_monotonic = all(win_rates[i] >= win_rates[i+1] for i in range(len(win_rates)-1)) if len(win_rates) > 1 else True
        report['confidence_well_calibrated'] = is_monotonic
        report['confidence_buckets'] = cal
        if not is_monotonic:
            report['calibration_issue'] = '置信度与实际胜率不单调，建议在prompt中强调置信度应反映真实把握'

    # 2. AI vs 本地模型
    avl = stats.get('ai_vs_local')
    if avl:
        report['ai_vs_local'] = avl
        if avl['ai_accuracy'] < avl['local_accuracy']:
            report['ai_underperforming'] = True
            report['suggestion'] = 'AI准确度低于本地模型，考虑提高本地模型权重或减少AI依赖'
        elif avl['ai_accuracy'] >= avl['local_accuracy']:
            report['ai_outperforming'] = True

    # 3. 贝叶斯对齐效力
    bap = stats.get('bayes_alignment_perf')
    if bap:
        report['bayes_alignment_perf'] = bap
        for b in bap:
            if b.get('bayesian_alignment') == '一致' and b.get('win_rate', 0) > 60:
                report['alignment_valuable'] = 'AI/贝叶斯一致时胜率显著更高，对齐是有效信号'
            elif b.get('bayesian_alignment') == '冲突' and b.get('win_rate', 0) < 40:
                report['conflict_predictive'] = '冲突时胜率显著更低，冲突检测有效'

    return report


def compute_waic_model_comparison(config_loglik_dict):
    """
    WAIC模型比较（Gelman BDA3 Ch.7 — Watanabe-Akaike Information Criterion）

    比较不同专家配置/策略的预测性能。
    WAIC = -2 * (lppd - p_waic)

    config_loglik_dict: {config_name: log_lik_matrix(SxN)}
      其中S=后验抽样数, N=观测数

    返回WAIC表、最佳配置、证据权重
    """
    from advanced_indicators import compute_waic as _compute_waic_single

    results = []
    for config_name, loglik_matrix in config_loglik_dict.items():
        try:
            waic_info = _compute_waic_single(loglik_matrix)
            results.append({
                'config': config_name,
                'waic': waic_info['waic'],
                'lppd': waic_info['lppd'],
                'p_waic': waic_info['p_waic'],
            })
        except Exception:
            continue

    if not results:
        return {'status': 'error', 'message': 'No valid WAIC computations'}

    waics = np.array([r['waic'] for r in results])
    min_waic = np.min(waics)
    delta_waics = waics - min_waic
    akaike_weights = np.exp(-0.5 * delta_waics)
    akaike_weights /= np.sum(akaike_weights)

    for i, r in enumerate(results):
        r['delta_waic'] = round(float(delta_waics[i]), 2)
        r['akaike_weight'] = round(float(akaike_weights[i]), 4)

    best_idx = int(np.argmin(waics))
    best_config = results[best_idx]['config']
    runner_up = results[1]['config'] if len(results) > 1 else 'N/A'
    evidence_ratio = float(akaike_weights[best_idx] / max(akaike_weights[1], 1e-10)) if len(results) > 1 else float('inf')

    return {
        'status': 'ok',
        'waic_table': results,
        'best_config': best_config,
        'best_waic': results[best_idx]['waic'],
        'evidence_ratio_vs_runner_up': round(evidence_ratio, 1),
        'interpretation': f"{best_config} is preferred; evidence ratio vs {runner_up} = {evidence_ratio:.1f}x"
    }


def track_regime_shifts(outcome_history, window=30):
    """
    DoubleAdapt启发的市场体制转换检测（Zhao et al., KDD 2024）

    在滚动窗口中检测胜率和方向比的结构性变化。
    当检测到体制转换时，应降低旧校准数据的权重。

    outcome_history: list of dicts with {pnl_pct, signal_type, timestamp}
    """
    n = len(outcome_history)
    if n < window * 2:
        return {'shift_detected': False, 'regime_label': 'insufficient_data',
                'current_calibration_weight': 1.0, 'shift_points': []}

    wins = []
    directions = []
    for o in outcome_history:
        pnl = o.get('pnl_pct', 0) or 0
        stype = str(o.get('signal_type', 'neutral'))
        wins.append(1 if pnl > 0 else 0)
        directions.append(1 if stype == 'bullish' else (-1 if stype == 'bearish' else 0))

    shift_points = []
    win_rates = []
    dir_ratios = []
    for i in range(window, n - window + 1, max(1, window // 3)):
        wr = np.mean(wins[i:i + window])
        dr = np.mean([1 if d > 0 else 0 for d in directions[i:i + window]])
        win_rates.append(wr)
        dir_ratios.append(dr)

    # Detect shifts
    for i in range(1, len(win_rates)):
        wr_delta = abs(win_rates[i] - win_rates[i - 1])
        dr_delta = abs(dir_ratios[i] - dir_ratios[i - 1])
        if wr_delta > 0.20 or dr_delta > 0.25:
            shift_points.append(i * (window // 3))

    shift_detected = len(shift_points) > 0

    # Calibration weight: exponential decay from last shift
    if shift_points:
        trades_since_last_shift = n - shift_points[-1]
        cal_weight = min(1.0, 0.5 * np.exp(trades_since_last_shift / 50.0))
    else:
        cal_weight = 1.0

    # Regime label
    recent_wr = np.mean(wins[-min(window, n):])
    recent_dr = np.mean([1 if d > 0 else 0 for d in directions[-min(window, n):]])
    if recent_wr > 0.55:
        regime_label = "bull_market"
    elif recent_wr < 0.40:
        regime_label = "bear_market"
    elif abs(recent_dr - 0.5) > 0.3:
        regime_label = "directional_regime"
    else:
        regime_label = "choppy"

    return {
        'shift_detected': shift_detected,
        'shift_points': shift_points,
        'shift_count': len(shift_points),
        'current_calibration_weight': round(cal_weight, 3),
        'regime_label': regime_label,
        'recent_win_rate': round(recent_wr, 3),
        'recent_direction_ratio': round(recent_dr, 3),
    }
