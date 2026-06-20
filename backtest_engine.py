# backtest_engine.py
"""
Historical backtest engine for A-share stocks.

Replays historical K-line data day-by-day using the exact same code paths
as live trading, simulating realistic A-share execution with T+1 settlement,
limit-lock detection, and trading costs.

SURVIVORSHIP BIAS WARNING:
    Backtest results are upward-biased due to survivorship bias — the stock
    pool consists of currently-listed stocks only. Stocks that were delisted,
    suspended, or went bankrupt during the backtest period are excluded, which
    inflates win rates and average returns. Interpret historical performance
    with caution.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any

from config import (
    STOCK_POOL,
    A_SHARE_LOT_SIZE,
    A_SHARE_HOLIDAYS,
    DEFAULT_STOP_PCT,
    DEFAULT_TARGET_PCT,
    DEFAULT_POSITION_PCT,
    DB_PATH,
    MIN_DAILY_VOLUME,
    MIN_DAILY_AMOUNT,
    TREND_FILTER_MA,
    TREND_FILTER_ENABLED,
    TRAILING_STOP_ENABLED,
    TRAILING_STOP_ACTIVATION,
    TRAILING_STOP_ATR_MULT,
    CHANDELIER_ENABLED,
    CHANDELIER_ATR_MULT,
    CHANDELIER_INITIAL_MULT,
    CHANDELIER_ACTIVATION_PCT,
    DISABLE_MA15_EXIT,
    MIN_HOLDING_DAYS,
    VOL_ADAPTIVE_SIZING,
    TARGET_VOL_PCT,
    MAX_POSITION_PCT,
    MIN_POSITION_PCT,
    MAX_HOLDING_DAYS,
    TIME_STOP_ENABLED,
    NUWA_VERBOSE,
    get_stock_sector,
)
from data_loader import get_daily_kline, get_hourly_kline, resample_5min_to_60min
from indicators_v2 import compute_all_indicators_v2
from expert_ensemble import compute_expert_ensemble, get_nuwa_school_weights
from backtest_feedback import replay_bars_for_exit, _is_limit_locked
from trade_logger import save_backtest_run

# Worker count for parallel operations (0 = auto-detect, 1 = serial)
_MAX_WORKERS = 0  # Will be set from config or CLI
_SWEEP_PARALLEL = True  # Enable parallel parameter sweep


# =============================================================================
# Parallel precompute worker (module-level, used by ThreadPoolExecutor)
# =============================================================================

def _precompute_stock_worker(args):
    """
    ThreadPoolExecutor worker: load data and compute all signals for a chunk.

    Args tuple: (symbols, tdx_dir, trading_day_strs)
    Each worker loads its own data to avoid pickling massive DataFrames.
    Returns dict: {cache_key: signal_dict | None}
    """
    import pandas as pd
    from mootdx.reader import Reader

    symbols, tdx_dir, trading_day_strs = args

    reader = Reader.factory(market='std', tdxdir=tdx_dir)
    trading_days = [date.fromisoformat(d) for d in trading_day_strs]

    results = {}
    for symbol in symbols:
        # Load data within worker
        try:
            df = reader.daily(symbol=symbol)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # ---- Dead-fish pre-filter: skip illiquid / deep-bear stocks ----
        if len(df) >= 60:
            recent = df.tail(20)
            avg_amt = float(recent['volume'].mean()) * 100 * float(recent['close'].mean())
            if avg_amt < 50_000_000:
                continue
            close_px = float(df['close'].iloc[-1])
            ma60_px = float(df['close'].tail(60).mean())
            if ma60_px > 0 and close_px < ma60_px * 0.75:
                continue

        # Hourly data (skipped for speed — unused by ensemble)
        hour_df = None

        for date_obj in trading_days:
            cache_key = f"{symbol}_{date_obj.isoformat()}"
            ts = pd.Timestamp(date_obj)

            df_sliced = df[df.index <= ts]
            if len(df_sliced) < 60:
                results[cache_key] = None
                continue

            last_row = df_sliced.iloc[-1]
            ohlcv = {
                'open': float(last_row['open']),
                'high': float(last_row['high']),
                'low': float(last_row['low']),
                'close': float(last_row['close']),
                'volume': float(last_row['volume']),
            }

            # 60-min resampled for multi-timeframe
            df_60min = None
            if hour_df is not None:
                h_sliced = hour_df[hour_df.index <= ts]
                if len(h_sliced) >= 60:
                    df_60min = resample_5min_to_60min(h_sliced)

            try:
                indicators = compute_all_indicators_v2(df_sliced, df_60min, None, symbol=symbol)
            except Exception:
                results[cache_key] = None
                continue

            if indicators is None:
                results[cache_key] = None
                continue

            try:
                ensemble = compute_expert_ensemble(indicators)
            except Exception:
                results[cache_key] = None
                continue

            results[cache_key] = {
                'symbol': symbol,
                'date': date_obj,
                **ohlcv,
                'indicators': indicators,
                'ensemble': ensemble,
            }

    return results


def _precompute_chunk_worker(args):
    """
    ProcessPoolExecutor worker: loads TDX data internally, computes signals.

    Accepts (symbols, tdx_dir, trading_day_strs).
    Each worker creates its own Reader instance — ZERO DataFrame pickling.
    Returns only lightweight signal dicts (no raw OHLCV DataFrames).
    """
    import pandas as pd
    from datetime import date
    from mootdx.reader import Reader
    from indicators_v2 import compute_all_indicators_v2
    from expert_ensemble import compute_expert_ensemble

    symbols, tdx_dir, trading_day_strs = args
    trading_days = [date.fromisoformat(d) for d in trading_day_strs]

    reader = Reader.factory(market='std', tdxdir=tdx_dir)
    results = {}

    for symbol in symbols:
        # ---- Load data inside worker (zero IPC overhead) ----
        try:
            df = reader.daily(symbol=symbol)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # ---- Dead-fish pre-filter ----
        if len(df) >= 60:
            recent = df.tail(20)
            avg_amt = float(recent['volume'].mean()) * 100 * float(recent['close'].mean())
            if avg_amt < 50_000_000:
                continue
            close_px = float(df['close'].iloc[-1])
            ma60_px = float(df['close'].tail(60).mean())
            if ma60_px > 0 and close_px < ma60_px * 0.75:
                continue

        # ---- Compute signals for each trading day ----
        for date_obj in trading_days:
            cache_key = f"{symbol}_{date_obj.isoformat()}"
            ts = pd.Timestamp(date_obj)

            df_sliced = df[df.index <= ts]
            if len(df_sliced) < 60:
                results[cache_key] = None
                continue

            last_row = df_sliced.iloc[-1]
            ohlcv = {
                'open': float(last_row['open']),
                'high': float(last_row['high']),
                'low': float(last_row['low']),
                'close': float(last_row['close']),
                'volume': float(last_row['volume']),
            }

            try:
                indicators = compute_all_indicators_v2(df_sliced, None, None, symbol=symbol)
            except Exception:
                results[cache_key] = None; continue
            if indicators is None:
                results[cache_key] = None; continue

            try:
                ensemble = compute_expert_ensemble(indicators)
            except Exception:
                results[cache_key] = None; continue

            results[cache_key] = {
                'symbol': symbol, 'date': date_obj,
                **ohlcv, 'indicators': indicators, 'ensemble': ensemble,
            }

    return results


# =============================================================================
# BacktestEngine
# =============================================================================

# ---------------------------------------------------------------------------
# Trading costs (A-share standard — defined here since config.py lacks them)
# ---------------------------------------------------------------------------
STAMP_DUTY: float = 0.001       # 0.1% stamp duty (sell only)
COMMISSION: float = 0.00025     # 0.025% commission (per side)
SLIPPAGE: float = 0.001         # 0.1% slippage (applied to entry/exit price)

# ---------------------------------------------------------------------------
# Default stop / target as ratios (used when signal does not specify)
# ---------------------------------------------------------------------------
DEFAULT_STOP_RATIO: float = 1.0 - DEFAULT_STOP_PCT   # e.g. 0.95
DEFAULT_TARGET_RATIO: float = 1.0 + DEFAULT_TARGET_PCT  # e.g. 1.10


class BacktestEngine:
    """
    Replay historical K-line data day-by-day and simulate A-share execution.

    The engine uses the identical indicator and ensemble code paths as live
    trading so that backtest results are directly comparable to live signals.
    """

    def __init__(
        self,
        stock_pool: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        initial_capital: float = 100_000,
        position_pct: float = DEFAULT_POSITION_PCT,
        stop_pct: float = DEFAULT_STOP_PCT,
        target_pct: float = DEFAULT_TARGET_PCT,
        max_positions: int = 10,
        confidence_threshold: float = 0.08,
    ) -> None:
        self.stock_pool: List[str] = stock_pool if stock_pool is not None else list(STOCK_POOL)
        self.start_date: Optional[date] = start_date
        self.end_date: date = end_date if end_date is not None else date.today()
        self.initial_capital: float = float(initial_capital)
        self.position_pct: float = float(position_pct)
        self.stop_pct: float = float(stop_pct)
        self.target_pct: float = float(target_pct)
        self.max_positions: int = int(max_positions)
        self.confidence_threshold: float = float(confidence_threshold)

        # ---- Runtime state ----
        self.all_daily_data: Dict[str, pd.DataFrame] = {}
        self.all_hourly_data: Dict[str, pd.DataFrame] = {}
        self.trading_days: List[date] = []
        self.cash: float = self.initial_capital
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Tuple[date, float]] = []
        self.signal_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        # Per-stock quality cache: trend OK, avg volume, avg amount, atr_pct
        self._stock_quality: Dict[str, Dict[str, float]] = {}

    # ========================================================================
    # Data loading
    # ========================================================================

    def load_all_data(self, workers: int = 0) -> None:
        """Load daily and hourly K-line for every stock in the pool.

        Args:
            workers: 0=serial, >0=thread pool size, <0=auto (cpu_count*2)
        """
        n_total = len(self.stock_pool)
        if workers == 0 or workers == 1 or n_total <= 5:
            self._load_all_data_serial()
            return

        # Auto-detect worker count
        if workers < 0:
            import os
            workers = min(16, os.cpu_count() or 4)

        print(f"[Backtest] Loading daily + hourly K-line for {n_total} stocks "
              f"(parallel, {workers} workers) ...")

        loaded = [0]
        skipped = [0]

        def _load_one(symbol):
            df = get_daily_kline(symbol, days=400)
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
            return symbol, df

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_load_one, s): s for s in self.stock_pool}
            for i, fut in enumerate(as_completed(futures)):
                sym, df = fut.result()
                if df is None or df.empty:
                    skipped[0] += 1
                else:
                    self.all_daily_data[sym] = df
                    loaded[0] += 1
                if (i + 1) % 50 == 0:
                    print(f"  ... {i + 1}/{n_total} (loaded={loaded[0]}, "
                          f"skipped={skipped[0]})")

        print(f"[Backtest] Data load complete: {loaded[0]} daily, {skipped[0]} skipped.")

    def _load_all_data_serial(self) -> None:
        """Original serial data loader (fallback)."""
        n_total = len(self.stock_pool)
        loaded = 0
        skipped = 0
        hourly_loaded = 0

        print(f"[Backtest] Loading daily K-line for {n_total} stocks (serial)...")
        for i, symbol in enumerate(self.stock_pool):
            df = get_daily_kline(symbol, days=400)
            if df is None or df.empty:
                skipped += 1
                if (i + 1) % 20 == 0:
                    print(f"  ... {i + 1}/{n_total} (loaded={loaded}, skipped={skipped})")
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            self.all_daily_data[symbol] = df
            loaded += 1
            # Hourly data skipped — unused by ensemble, saves 80% load time

            if (i + 1) % 20 == 0:
                print(f"  ... {i + 1}/{n_total} (loaded={loaded}, skipped={skipped})")

        print(f"[Backtest] Data load complete: {loaded} daily, {skipped} skipped.")

    # ========================================================================
    # Market index direction filter (防狼术)
    # ========================================================================

    def _load_market_index(self) -> None:
        """Load market index (000001 上证指数) for trend direction filter."""
        try:
            from data_loader import get_daily_kline
            df = get_daily_kline('000001', days=400)
            if df is not None and not df.empty:
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                df['ma60'] = df['close'].rolling(60).mean()
                df['atr14'] = self._calc_atr_series(df, 14)
                self._market_index = df
                return
        except Exception:
            pass
        self._market_index = None

    @staticmethod
    def _calc_atr_series(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _get_market_features(self, today: date) -> Dict[str, float]:
        """
        Return continuous market regime features for XGBoost ranking.
        NO hard gating — all stocks compete on relative strength regardless of regime.

        Returns:
            market_ma_deviation: (price - ma60) / ma60  — 连续乖离率
            market_atr_pct: ATR as % of price
            market_regime_bullish: 1 if price > ma60*1.01 else 0
            market_regime_bearish: 1 if price < ma60*0.99 else 0
            dynamic_threshold: base confidence threshold (no regime adjustment)
        """
        if self._market_index is None:
            return {
                'market_ma_deviation': 0.0,
                'market_atr_pct': 0.02,
                'market_regime_bullish': 0,
                'market_regime_bearish': 0,
                'dynamic_threshold': self.confidence_threshold,
            }

        idx = self._market_index
        mask = idx.index <= pd.Timestamp(today)
        if not mask.any():
            return {
                'market_ma_deviation': 0.0,
                'market_atr_pct': 0.02,
                'market_regime_bullish': 0,
                'market_regime_bearish': 0,
                'dynamic_threshold': self.confidence_threshold,
            }

        row = idx.loc[mask].iloc[-1]
        price = float(row['close'])
        ma60 = float(row.get('ma60', price))
        atr14 = float(row.get('atr14', price * 0.02))
        atr_pct = atr14 / price if price > 0 else 0.02
        deviation = (price - ma60) / ma60 if ma60 > 0 else 0.0

        return {
            'market_ma_deviation': round(deviation, 4),
            'market_atr_pct': round(atr_pct, 4),
            'market_regime_bullish': 1 if price > ma60 * 1.01 else 0,
            'market_regime_bearish': 1 if price < ma60 * 0.99 else 0,
            'dynamic_threshold': self.confidence_threshold,
        }

    # ========================================================================
    # Per-stock quality filter (trend + liquidity)
    # ========================================================================

    def _precompute_stock_quality(self, today: date) -> None:
        """Compute trend/liquidity metrics for all stocks (recompute trend daily)."""
        for sym in self.stock_pool:
            df = self.all_daily_data.get(sym)
            if df is None or df.empty:
                self._stock_quality[sym] = {'trend_ok': False, 'avg_vol': 0, 'avg_amount': 0, 'atr_pct': 0.03}
                continue

            df_to_date = df[df.index <= pd.Timestamp(today)]
            if len(df_to_date) < TREND_FILTER_MA:
                self._stock_quality[sym] = {'trend_ok': False, 'avg_vol': 0, 'avg_amount': 0, 'atr_pct': 0.03}
                continue

            recent = df_to_date.tail(TREND_FILTER_MA)
            close = float(recent['close'].iloc[-1])
            ma = float(recent['close'].mean())
            avg_vol_lots = float(recent['volume'].mean())  # TDX volume in 手
            avg_vol_shares = avg_vol_lots * 100  # Convert to shares
            avg_amount = avg_vol_shares * close

            # ATR for vol-adaptive sizing
            high, low, c = recent['high'], recent['low'], recent['close']
            tr1 = high - low
            tr2 = abs(high - c.shift(1))
            tr3 = abs(low - c.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())
            atr_pct = atr / close if close > 0 else 0.03

            self._stock_quality[sym] = {
                'trend_ok': close >= ma if TREND_FILTER_ENABLED else True,
                'avg_vol': avg_vol_shares,  # Now in shares
                'avg_amount': avg_amount,
                'atr_pct': atr_pct,
                'close': close,
            }

    def _passes_quality_filter(self, sym: str, today: date) -> Tuple[bool, str]:
        """
        Check if a stock passes entry quality filters.

        Returns (passed, reason).
        """
        self._precompute_stock_quality(today)
        q = self._stock_quality.get(sym, {})
        if not q:
            return False, 'no_data'

        if TREND_FILTER_ENABLED and not q.get('trend_ok', True):
            return False, 'below_ma'

        avg_amount = q.get('avg_amount', 0)
        if avg_amount > 0 and avg_amount < MIN_DAILY_AMOUNT:
            return False, f'low_amount({avg_amount/1e4:.0f}w)'

        avg_vol = q.get('avg_vol', 0)
        if avg_vol > 0 and avg_vol < MIN_DAILY_VOLUME:
            return False, f'low_vol({avg_vol/1e4:.0f}w)'

        return True, 'ok'

    # ========================================================================
    # Trading calendar
    # ========================================================================

    def get_trading_days(self) -> List[date]:
        """
        Build the union of all trading dates across stocks in [self.start_date, self.end_date].

        Excludes weekends (Mon=0 ... Fri=4, i.e. weekday < 5) and any date that
        appears in ``config.A_SHARE_HOLIDAYS`` (stored as "YYYY-MM-DD" strings).
        """
        if not self.all_daily_data:
            self.load_all_data()

        all_dates: set[date] = set()
        for df in self.all_daily_data.values():
            for dt in df.index:
                d = dt.date() if hasattr(dt, 'date') else pd.Timestamp(dt).date()
                all_dates.add(d)

        # Apply date range filter
        if self.start_date is not None:
            all_dates = {d for d in all_dates if d >= self.start_date}
        all_dates = {d for d in all_dates if d <= self.end_date}

        # Exclude weekends and holidays
        holiday_set = set(A_SHARE_HOLIDAYS)
        trading = []
        for d in all_dates:
            if d.weekday() >= 5:                # Saturday(5) or Sunday(6)
                continue
            if d.strftime('%Y-%m-%d') in holiday_set:
                continue
            trading.append(d)

        trading.sort()
        self.trading_days = trading

        day_count = len(trading)
        if self.start_date and self.end_date:
            print(f"[Backtest] Trading days: {day_count} "
                  f"({self.start_date} -> {self.end_date})")
        else:
            print(f"[Backtest] Trading days: {day_count}")

        if day_count == 0:
            print("[Backtest] WARNING: No trading days in range. Check data or holidays.")

        return trading

    # ========================================================================
    # Per-day signal computation (NO look-ahead)
    # ========================================================================

    def precompute_signals_parallel(
        self,
        trading_days: List[date],
        workers: int = 0,
    ) -> int:
        """
        Single-phase parallel precompute — zero main-process data loading.

        Architecture:
          Main process:  filter symbols → chunk → dispatch to workers
          Each worker:   creates its own Reader, loads TDX data internally,
                         applies dead-fish filter, computes indicators+ensemble,
                         returns ONLY lightweight signal dicts.

        Eliminates the old Phase 1 (pre-fetch 5000 DataFrames into memory)
        and Phase 2 (pickle massive DataFrames to subprocesses) — both
        were causing IPC serialization bottlenecks and OOM risk.
        """
        import os, time
        if workers <= 0:
            workers = max(1, (os.cpu_count() or 4) - 1)
        workers = min(workers, len(self.stock_pool))

        if workers <= 1 or len(self.stock_pool) < 20:
            return 0

        from config import FUYI_TDX_DIR
        symbols = list(self.stock_pool)
        trading_day_strs = [d.isoformat() for d in trading_days]

        # ---- Chunk symbols (no data pre-loading) ----
        chunk_size = max(1, len(symbols) // workers)
        chunks = []
        for i in range(0, len(symbols), chunk_size):
            chunk_syms = symbols[i:i + chunk_size]
            chunks.append((chunk_syms, FUYI_TDX_DIR, trading_day_strs))

        print(f"[Precompute] {workers} workers x {len(chunks)} chunks "
              f"({len(symbols)} stocks, zero pre-fetch)")

        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_precompute_chunk_worker, c): i
                       for i, c in enumerate(chunks)}
            for fut in as_completed(futures):
                chunk_idx = futures[fut]
                try:
                    r = fut.result()
                    self.signal_cache.update(r)
                    done_count = sum(1 for v in r.values() if v is not None)
                    print(f"  [Precompute] Chunk {chunk_idx+1}/{len(chunks)} "
                          f"→ {done_count} signals")
                except Exception as e:
                    print(f"  [Precompute] Chunk {chunk_idx+1} FAILED: {e}")

        elapsed = time.time() - t0
        n_cached = sum(1 for v in self.signal_cache.values() if v is not None)
        print(f"[Precompute] done: {n_cached} signals in {elapsed/60:.1f}min")
        return n_cached

    def compute_daily_signals(
        self,
        symbol: str,
        date_obj: date,
    ) -> Optional[Dict[str, Any]]:
        """
        Compute indicators and expert-ensemble signal for *symbol* using
        only data up to (and including) *date_obj*.

        Returns a dict with keys:
            symbol, date, open, high, low, close, volume,
            indicators, ensemble
        or ``None`` if data is insufficient.
        """
        cache_key = f"{symbol}_{date_obj.isoformat()}"
        if cache_key in self.signal_cache:
            return self.signal_cache[cache_key]

        df = self.all_daily_data.get(symbol)
        if df is None or df.empty:
            self.signal_cache[cache_key] = None
            return None

        ts = pd.Timestamp(date_obj)
        df_sliced = df[df.index <= ts]
        if len(df_sliced) < 60:
            self.signal_cache[cache_key] = None
            return None

        # Last row OHLCV
        last_row = df_sliced.iloc[-1]
        ohlcv = {
            'open': float(last_row['open']),
            'high': float(last_row['high']),
            'low': float(last_row['low']),
            'close': float(last_row['close']),
            'volume': float(last_row['volume']),
        }

        # Indicators — slice hourly data and resample to 60-min for multi-timeframe
        df_60min = None
        if symbol in self.all_hourly_data:
            hdf = self.all_hourly_data[symbol]
            h_sliced = hdf[hdf.index <= ts]
            if len(h_sliced) >= 60:
                df_60min = resample_5min_to_60min(h_sliced)

        try:
            indicators = compute_all_indicators_v2(df_sliced, df_60min, None, symbol=symbol)
        except Exception as e:
            print(f"  [WARN] Indicator computation failed for {symbol} @ {date_obj}: {e}")
            self.signal_cache[cache_key] = None
            return None

        if indicators is None:
            self.signal_cache[cache_key] = None
            return None

        # Ensemble
        try:
            ensemble = compute_expert_ensemble(indicators)
        except Exception as e:
            print(f"  [WARN] Ensemble computation failed for {symbol} @ {date_obj}: {e}")
            self.signal_cache[cache_key] = None
            return None

        result: Dict[str, Any] = {
            'symbol': symbol,
            'date': date_obj,
            **ohlcv,
            'indicators': indicators,
            'ensemble': ensemble,
        }
        self.signal_cache[cache_key] = result
        return result

    # ========================================================================
    # Main backtest loop
    # ========================================================================

    def run_backtest(
        self,
        nuwa_weights_override: Optional[Dict[str, float]] = None,
        verbose: bool = True,
        workers: int = 0,
    ) -> None:
        """
        Execute the full day-by-day backtest.

        Parameters
        ----------
        nuwa_weights_override : dict or None
            Reserved for future weight override plumbing.
        verbose : bool
            Print progress and summary information.
        workers : int
            0=auto (parallel data load + signal precompute),
            1=serial, >1=explicit worker count.
        """
        # ---- 1. Load data (threaded) ----
        if not self.all_daily_data:
            self.load_all_data(workers=workers if workers != 1 else 0)

        # ---- 1.5 Load market index for direction filter ----
        self._load_market_index()

        # ---- 2. Build trading days ----
        trading_days = self.get_trading_days()
        if not trading_days:
            print("[Backtest] ABORT: no trading days in range.")
            return

        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.signal_cache.clear()

        # ---- 2.5 Pre-compute all signals (parallel) ----
        if workers != 1:
            n_cached = self.precompute_signals_parallel(trading_days, workers=workers)
            if n_cached > 0 and verbose:
                print(f"[Backtest] Using {n_cached} pre-computed signals "
                      f"({len(self.signal_cache)} cache slots)")

        # Signal pass-rate counters
        self._bullish_total = 0
        self._bullish_passed = 0
        # Post-exit cooldown: don't re-enter a stock for N days after exit
        self._cooldown: Dict[str, date] = {}
        COOLDOWN_DAYS = 10

        # Pending entries: those decided on day T, executed at T+1 open.
        # Each element: {'symbol', 'signal', 'signal_date'}
        pending_entries: List[Dict[str, Any]] = []

        # ---- 3. Day-by-day loop ----
        n_days = len(trading_days)
        signal_counter = 0

        for i, today in enumerate(trading_days):
            next_day = trading_days[i + 1] if i + 1 < n_days else None

            # ----- 3a. Execute pending entries at TODAY's open -----
            for pe in pending_entries:
                self._open_long_position(
                    symbol=pe['symbol'],
                    signal=pe['signal'],
                    today=pe['signal_date'],
                    next_trading_day=today,
                )
            pending_entries.clear()

            # ----- 3b. Check existing positions for exits -----
            closed_symbols: List[str] = []
            for sym, pos in self.positions.items():
                df = self.all_daily_data.get(sym)
                if df is None:
                    continue

                today_bar_slice = df[df.index == pd.Timestamp(today)]
                if today_bar_slice.empty:
                    # Stock may be suspended — skip
                    continue

                # Build data slice from entry bar to today
                entry_idx = pos['entry_idx']
                df_slice = df[(df.index >= entry_idx) & (df.index <= pd.Timestamp(today))]
                if df_slice.empty:
                    continue

                # Build targets list
                actual_targets = pos.get('targets', [])
                fallback_target = pos.get('fallback_target', pos['entry_price'] * DEFAULT_TARGET_RATIO)

                # Vectorized exit matching (replaces iterrows-based replay_bars_for_exit)
                try:
                    from matching_engine import match_exits_vectorized, get_limit_ratio, compute_dynamic_slippage
                    # Prepare future bars (T+1 compliant: skip entry bar)
                    future_bars = df[(df.index > entry_idx) & (df.index <= pd.Timestamp(today))]
                    if len(future_bars) > 0:
                        all_targets = list(actual_targets) if actual_targets else [fallback_target]
                        limit_ratio = get_limit_ratio(sym)
                        dyn_slip = compute_dynamic_slippage(
                            self._stock_quality.get(sym, {}).get('atr_pct', 0.03))
                        entry_date_str = str(df.index[df.index == entry_idx][0].date())
                        vec_result = match_exits_vectorized(
                            future_bars, pos['entry_price'], pos['stop_loss'],
                            np.array(sorted(all_targets)),
                            trailing_activation=TRAILING_STOP_ACTIVATION,
                            trailing_atr_mult=TRAILING_STOP_ATR_MULT,
                            limit_ratio=limit_ratio,
                            slippage_entry=dyn_slip,
                            symbol=sym,
                            entry_date=entry_date_str,
                        )
                        # Map to legacy replay format
                        replay = {
                            'exit_price': vec_result['exit_price'],
                            'hit_target': vec_result['hit_target'],
                            'stopped_out': vec_result['stopped_out'],
                            'gap_stopped': vec_result['gap_stopped'],
                            'limit_hit': vec_result['limit_hit'],
                            'trailing_stopped': vec_result['trailing_stopped'],
                            'eval_code': vec_result['eval_code'],
                            'target_hit_price': vec_result['target_hit_price'],
                            'stop_hit_price': vec_result['stop_hit_price'],
                        }
                    else:
                        replay = {'exit_price': None, 'hit_target': 0, 'stopped_out': 0,
                                  'gap_stopped': 0, 'limit_hit': 0, 'trailing_stopped': 0,
                                  'eval_code': 99, 'target_hit_price': None, 'stop_hit_price': None}
                except ImportError:
                    # Fallback to original iterrows-based engine
                    replay = replay_bars_for_exit(
                        df=df_slice,
                        entry_idx=entry_idx,
                        entry_open=pos['entry_price'],
                        direction='bullish',
                        stop_price=pos['stop_loss'],
                        actual_targets=actual_targets,
                        fallback_target=fallback_target,
                        trailing_stop_enabled=TRAILING_STOP_ENABLED,
                        trailing_activation=TRAILING_STOP_ACTIVATION,
                        trailing_atr_mult=TRAILING_STOP_ATR_MULT,
                    )

                # ---- MACD/RSI divergence exit (DISABLED — too sensitive, 66% of exits) ----
                divergence_exit = False
                df_full = self.all_daily_data.get(sym)
                # Divergence check disabled — was catching noise, not real reversals
                # Keeping the code for future refinement with better thresholds

                # MA15 exit: DISABLED — 8.8% win rate, toxic to trend following
                ma_broken = False
                if not DISABLE_MA15_EXIT and df_full is not None and pos['shares'] > 0:
                    df_to_date = df_full[df_full.index <= pd.Timestamp(today)]
                    if len(df_to_date) >= 15:
                        today_close = float(df_to_date['close'].iloc[-1])
                        today_open = float(df_to_date['open'].iloc[-1])
                        daily_drop_pct = (today_close - today_open) / today_open
                        ma15_val = float(df_to_date['close'].tail(15).mean())
                        if today_close < ma15_val and daily_drop_pct < -0.03:
                            ma_broken = True
                            replay['exit_price'] = today_close
                            replay['ma_broken'] = True

                # ---- Chandelier Exit (吊灯止损) — daily state update ----
                # Only runs if the vectorized exit check above did NOT already trigger
                # AND chandelier is active on this position.
                chandelier_triggered = False
                if (CHANDELIER_ENABLED and pos.get('chandelier_active', False)
                        and replay['exit_price'] is None):
                    today_bar_ch = df_slice.iloc[-1]
                    today_high = float(today_bar_ch['high'])
                    today_low  = float(today_bar_ch['low'])
                    today_close_ch = float(today_bar_ch['close'])
                    today_open_ch  = float(today_bar_ch['open'])
                    # ATR lookup: O(1) from precomputed column in df
                    today_atr = float(today_bar_ch.get('atr14', 0))
                    if today_atr <= 0:
                        today_atr = today_close_ch * 0.02

                    # ── Compute today's holding days ──
                    entry_dt_ch = pos.get('actual_entry_date', pos.get('signal_date'))
                    if isinstance(entry_dt_ch, (date, datetime)):
                        holding_days_today = (today - (entry_dt_ch if isinstance(entry_dt_ch, date)
                                               else entry_dt_ch.date())).days
                    else:
                        holding_days_today = 999

                    # ── 1. Determine today's effective stop ──
                    # Use YESTERDAY's stop (already stored in pos['chandelier_stop']).
                    # The stop was computed at the end of the prior trading day.
                    initial_stop = pos['entry_price'] - CHANDELIER_INITIAL_MULT * float(
                        df_slice[df_slice.index == entry_idx].iloc[0].get('atr14',
                        pos['entry_price'] * 0.02)) if len(df_slice[df_slice.index == entry_idx]) > 0 else pos['entry_price'] * 0.94
                    current_stop = pos.get('chandelier_stop', initial_stop)

                    # ── 2. Check exit trigger ──
                    # Hard stop (initial stop) always active regardless of immunity.
                    # Trailing stop only active after MIN_HOLDING_DAYS.
                    hard_stop_breach = today_low < initial_stop
                    trailing_stop_breach = (holding_days_today >= MIN_HOLDING_DAYS
                                            and (today_low < current_stop or today_close_ch < current_stop))

                    if hard_stop_breach and holding_days_today < MIN_HOLDING_DAYS:
                        # Within immunity window: only hard stop can trigger
                        chandelier_triggered = True
                        replay['exit_price'] = min(initial_stop, today_open_ch)
                        replay['chandelier_triggered'] = True
                        replay['chandelier_stop'] = round(initial_stop, 3)
                    elif trailing_stop_breach:
                        # Immunity window passed: trailing stop can trigger
                        chandelier_triggered = True
                        if today_low < current_stop:
                            # Gap-down: true exit price = min(stop, open)
                            replay['exit_price'] = min(current_stop, today_open_ch)
                        else:
                            replay['exit_price'] = today_close_ch
                        replay['chandelier_triggered'] = True
                        replay['chandelier_stop'] = round(current_stop, 3)
                        replay['chandelier_highest'] = round(
                            pos.get('chandelier_highest', pos['entry_price']), 3)
                    else:
                        # ── 3. Survived — update state for TOMORROW's check ──
                        # Update highest high since entry (only ratchet UP)
                        prev_highest = pos.get('chandelier_highest', pos['entry_price'])
                        if today_high > prev_highest:
                            pos['chandelier_highest'] = today_high
                        current_highest = pos['chandelier_highest']

                        # Activation gate: only ratchet trailing stop UP
                        # once profit exceeds CHANDELIER_ACTIVATION_PCT
                        profit_pct = (current_highest - pos['entry_price']) / pos['entry_price']
                        if profit_pct >= CHANDELIER_ACTIVATION_PCT:
                            atr_mult = pos.get('chandelier_atr_mult', CHANDELIER_ATR_MULT)
                            new_stop = current_highest - atr_mult * today_atr
                            # Only ratchet UP, never down. Floor = initial_stop.
                            if new_stop > current_stop:
                                pos['chandelier_stop'] = new_stop
                        # else: keep current_stop unchanged — not activated yet

                # ---- T+0 Grid: cost reduction for locked positions on high-vol days ----
                try:
                    from execution_algos import T0GridAlgo
                    t0 = T0GridAlgo()
                    today_bar = df_slice.iloc[-1]
                    t0_open = float(today_bar['open'])
                    t0_high = float(today_bar['high'])
                    t0_low = float(today_bar['low'])
                    t0_close = float(today_bar['close'])
                    # Sellable = original shares minus today's buys (simplified)
                    sellable = pos.get('shares', 0)
                    cash_avail = self.cash * 0.1  # Use max 10% cash for T+0
                    if t0.should_activate(t0_open, t0_high, t0_low, sellable, cash_avail):
                        t0_pnl, t0_logs = t0.simulate_backtest(
                            open_price=t0_open, high=t0_high, low=t0_low,
                            close=t0_close, sellable_shares=sellable,
                            available_cash=cash_avail,
                            current_cost=pos.get('entry_price', t0_open))
                        # Record T+0 P&L regardless of sign (如实记录盈亏)
                        self.cash += t0_pnl
                        if not hasattr(self, '_t0_pseudo_trades'):
                            self._t0_pseudo_trades: List[Dict[str, Any]] = []
                        self._t0_pseudo_trades.append({
                            'symbol': sym,
                            'direction': 'long',
                            'trade_type': 't0_grid',
                            'entry_date': str(today),
                            'exit_date': str(today),
                            'entry_price': round(float(t0_close), 3),
                            'exit_price': round(float(t0_close), 3),
                            'shares': 0,
                            'gross_pnl_pct': 0.0,
                            'net_pnl_pct': 0.0,
                            'net_pnl_rmb': round(t0_pnl, 2),
                            'exit_reason': 't0_grid',
                            'holding_days': 0,
                            'ensemble_confidence': 0.0,
                            'school_votes_json': {},
                        })
                except ImportError:
                    pass

                # Time stop: auto-close positions held too long
                time_stopped = False
                if TIME_STOP_ENABLED:
                    entry_dt = pos.get('actual_entry_date', pos.get('signal_date'))
                    if isinstance(entry_dt, (date, datetime)):
                        entry_d = entry_dt if isinstance(entry_dt, date) else entry_dt.date()
                        holding = (today - entry_d).days
                        if holding >= MAX_HOLDING_DAYS:
                            time_stopped = True
                            today_bar = df_slice.iloc[-1]
                            replay['exit_price'] = float(today_bar['close'])
                            replay['time_stopped'] = True

                if replay['exit_price'] is not None:
                    # ── Gap-down slippage for hard stop exits ──
                    # If today opened BELOW the stop price (gap-down), the true
                    # exit is at the open (first available price in T+1), not the
                    # stop price.  min(stop, open) captures this correctly.
                    if (replay.get('stopped_out') or replay.get('gap_stopped')
                            or replay.get('chandelier_triggered')):
                        today_bar_exit = df_slice.iloc[-1]
                        today_open_exit = float(today_bar_exit['open'])
                        replay['exit_price'] = min(replay['exit_price'], today_open_exit)

                    # Apply slippage to exit price
                    replay['exit_price'] = replay['exit_price'] * (1.0 - SLIPPAGE)
                    exit_reason = 'unknown'
                    if replay.get('chandelier_triggered'):
                        exit_reason = 'chandelier_stop'
                    elif replay.get('stopped_out'):
                        exit_reason = 'stop_loss_gap' if replay.get('gap_stopped') else 'stop_loss'
                    elif replay.get('divergence_exit'):
                        exit_reason = 'divergence'
                    elif replay.get('trailing_stopped'):
                        exit_reason = 'trailing_stop'
                    elif replay.get('ma_broken'):
                        exit_reason = 'ma15_broken'
                    elif replay.get('time_stopped'):
                        exit_reason = 'time_stop'
                    elif replay.get('hit_target'):
                        exit_reason = 'target'
                    elif replay.get('limit_hit'):
                        exit_reason = 'limit_locked'

                    trade = self._close_position(
                        pos=pos,
                        exit_price=replay['exit_price'],
                        exit_date=today,
                        reason=exit_reason,
                    )
                    self.trades.append(trade)
                    self.cash += pos['shares'] * replay['exit_price']
                    closed_symbols.append(sym)
                    # Set cooldown after exit
                    self._cooldown[sym] = today + timedelta(days=COOLDOWN_DAYS)

            for sym in closed_symbols:
                del self.positions[sym]

            # ----- 3c. Compute signals (only if room for new positions) -----
            available_slots = self.max_positions - len(self.positions)
            bullish_signals: List[Dict[str, Any]] = []

            if available_slots > 0:
                # Pre-compute stock quality metrics for ALL stocks (called once per day)
                self._precompute_stock_quality(today)

                n_stocks = len(self.stock_pool)
                for j, sym in enumerate(self.stock_pool):
                    if sym in self.positions:
                        continue
                    cooldown_until = self._cooldown.get(sym)
                    if cooldown_until is not None and today <= cooldown_until:
                        continue

                    # ---- Lightweight pre-filter (liquidity only) ----
                    q = self._stock_quality.get(sym, {})
                    if q.get('avg_amount', 0) < MIN_DAILY_AMOUNT:
                        continue

                    sig = self.compute_daily_signals(sym, today)
                    signal_counter += 1

                    if signal_counter % 50 == 0 and verbose:
                        print(f"  [Signals] Computed {signal_counter} total signals ...")

                    if sig is None:
                        continue

                    ens = sig.get('ensemble', {})
                    ens_signal = ens.get('ensemble_signal', 'neutral')
                    ens_conf = ens.get('ensemble_confidence', 0)
                    self._bullish_total += 1

                    # ── Probe: count per-school vote directions ──
                    if not hasattr(self, '_school_vote_counts'):
                        from collections import defaultdict
                        self._school_vote_counts = defaultdict(lambda: {'bull':0,'bear':0,'neut':0})
                    for sn, sv in ens.get('school_signals', {}).items():
                        d = sv.get('direction','neutral') if isinstance(sv, dict) else 'neutral'
                        self._school_vote_counts[sn][d[:4]] += 1

                    # ── Hard gate: Long-Only — non-bullish signals MUST NOT enter buy queue ──
                    if ens_signal != 'bullish':
                        continue

                    # ── Entry Filter A: Low-volatility dead-stock gate ──
                    atr14_val = q.get('atr_pct', 0.03)
                    if atr14_val < 0.02:
                        continue  # ATR < 2% → stagnant water, skip

                    # ── Entry Filter B: Bullish candle confirmation (ANTI-FUTURE-LEAK) ──
                    # T日收盘 > T日开盘 AND T日收盘 > T-1日收盘 → 阳线确认，不接飞刀。
                    df_entry = self.all_daily_data.get(sym)
                    if df_entry is not None and len(df_entry) >= 2:
                        df_signal = df_entry[df_entry.index <= pd.Timestamp(today)]
                        if len(df_signal) >= 2:
                            signal_bar   = df_signal.iloc[-1]         # T日 (signal day)
                            signal_close = float(signal_bar['close'])
                            signal_open  = float(signal_bar['open'])
                            prev_close   = float(df_signal.iloc[-2]['close'])  # T-1日
                            if not (signal_close > signal_open and signal_close > prev_close):
                                continue  # Signal day not bullish → skip

                    # Inject continuous market features into ensemble score
                    # (replaces hard gating — all stocks compete via ranking)
                    mkt = self._get_market_features(today)
                    dyn_threshold = mkt['dynamic_threshold']

                    # === TASK 1: Bearish regime → confidence penalty (0.70×) ===
                    if mkt['market_regime_bearish']:
                        ens_conf *= 0.70

                    # === TASK 3: Stock below MA60 → flat confidence deduction (-0.10) ===
                    # Replaces the old hard `if close < MA60: continue`.
                    stock_dev = 0.0
                    if q.get('close', 0) > 0 and q.get('avg_vol', 0) > 0:
                        ma60_est = q.get('avg_amount', 0) / q['avg_vol']
                        if ma60_est > 0:
                            stock_dev = (q['close'] - ma60_est) / ma60_est
                    if stock_dev < -0.02:
                        ens_conf -= 0.10

                    # Dynamic confidence threshold (after all penalties applied)
                    if ens_conf < dyn_threshold:
                        continue

                    # === TASK 2: Dynamic risk budget flag for downstream position sizing ===
                    sig['_market_features'] = {
                        'market_dev': mkt['market_ma_deviation'],
                        'stock_dev': round(stock_dev, 4),
                        'adjusted_confidence': round(ens_conf, 3),
                        'regime_bearish': bool(mkt['market_regime_bearish']),
                        'regime_neutral': bool(not mkt['market_regime_bullish']
                                               and not mkt['market_regime_bearish']),
                    }

                    self._bullish_passed += 1
                    bullish_signals.append(sig)

            # ----- 3d/e. Sort by confidence, apply sector limit, fill slots -----
            bullish_signals.sort(
                key=lambda s: s['ensemble']['ensemble_confidence'],
                reverse=True,
            )
            # Sector concentration limit: max 3 per sector
            MAX_PER_SECTOR = 3
            sector_counts: Dict[str, int] = {}
            # Count existing positions' sectors
            for sym in self.positions:
                sec = get_stock_sector(sym)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

            # ----- 3f/g. Create pending entries for T+1 -----
            if next_day is not None:
                for sig in bullish_signals:
                    if len(pending_entries) >= available_slots:
                        break
                    sym = sig['symbol']
                    sec = get_stock_sector(sym)

                    # Sector capacity
                    if sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
                        continue

                    pending_entries.append({
                        'symbol': sym,
                        'signal': sig,
                        'signal_date': today,
                    })
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1

            # ----- 3h. Max drawdown circuit breaker -----
            self._update_equity(today)
            if self.equity_curve:
                current_eq = self.equity_curve[-1][1]
                dd_pct = (self.initial_capital - current_eq) / self.initial_capital
                if dd_pct > 0.25:  # 25% max drawdown → stop trading
                    if verbose:
                        print(f"  [BREAKER] Max drawdown {dd_pct*100:.1f}% > 25% — stopping!")
                    # Close all positions at today's close
                    for sym, pos in list(self.positions.items()):
                        df_s = self.all_daily_data.get(sym)
                        if df_s is not None:
                            today_b = df_s[df_s.index == pd.Timestamp(today)]
                            if not today_b.empty:
                                exit_px = float(today_b.iloc[0]['close']) * (1.0 - SLIPPAGE)
                                trade = self._close_position(pos, exit_px, today, 'drawdown_breaker')
                                self.trades.append(trade)
                                self.cash += pos['shares'] * exit_px
                    self.positions.clear()
                    break

            if verbose and (i + 1) % 50 == 0:
                eq = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
                print(
                    f"  [Day {i + 1}/{n_days}] {today}  "
                    f"equity={eq:,.0f}  positions={len(self.positions)}  "
                    f"trades={len(self.trades)}"
                )

        # ---- 4. Close remaining positions at last day close ----
        if self.positions and trading_days:
            last_day = trading_days[-1]
            for sym, pos in list(self.positions.items()):
                df = self.all_daily_data.get(sym)
                if df is None or df.empty:
                    continue
                last_bar = df[df.index == pd.Timestamp(last_day)]
                if last_bar.empty:
                    # Try last available bar
                    last_bar = df.iloc[-1:]
                if last_bar.empty:
                    continue
                exit_price = float(last_bar.iloc[-1]['close'])
                trade = self._close_position(pos, exit_price, last_day, reason='end_of_backtest')
                self.trades.append(trade)
                self.cash += pos['shares'] * exit_price
            self.positions.clear()
            self._update_equity(last_day)

        # ---- 4.5 Inject T+0 pseudo-trades into self.trades ----
        # T+0 grid profits are added to cash during the backtest but bypass
        # self.trades. Inject them now so compute_statistics() can produce
        # separate Trend vs T+0 dual-track metrics.
        if hasattr(self, '_t0_pseudo_trades') and self._t0_pseudo_trades:
            self.trades.extend(self._t0_pseudo_trades)

        # ── Vote probe report ──
        if hasattr(self, '_school_vote_counts'):
            from expert_ensemble import SCHOOLS
            vc = self._school_vote_counts
            with open('school_votes_report.txt', 'w', encoding='utf-8') as f:
                f.write('='*85 + '\n')
                f.write('  SCHOOL VOTE DISTRIBUTION\n')
                f.write('='*85 + '\n')
                f.write('  %-24s %7s %8s %8s %8s %8s\n' % ('School','Total','Bullish','Bearish','Neutral','Bull%'))
                f.write('  ' + '-'*70 + '\n')
                for name in sorted(vc.keys(), key=lambda n: vc[n]['bull']+vc[n]['bear']+vc[n]['neut'], reverse=True):
                    c = vc[name]; total = c['bull']+c['bear']+c['neut']
                    if total == 0: continue
                    label = SCHOOLS.get(name,{}).get('short_label', name)[:22]
                    bp = c['bull']/total*100
                    f.write('  %-24s %7d %8d %8d %8d %7.2f%%\n' % (label, total, c['bull'], c['bear'], c['neut'], bp))
                f.write('='*85 + '\n')

        # ---- 5. Compute statistics ----
        self.stats = self.compute_statistics()

        # ---- 6. Print summary ----
        if verbose:
            self._print_summary()

    # ========================================================================
    # Position management
    # ========================================================================

    def _open_long_position(
        self,
        symbol: str,
        signal: Dict[str, Any],
        today: date,
        next_trading_day: date,
    ) -> None:
        """
        Open a long position at *next_trading_day*'s open price.

        Called when a pending entry from *today* reaches its execution date.
        Skips the position if the entry bar is limit-locked or shares round
        to zero.
        """
        df = self.all_daily_data.get(symbol)
        if df is None or df.empty:
            return

        next_bar = df[df.index == pd.Timestamp(next_trading_day)]
        if next_bar.empty:
            return

        bar = next_bar.iloc[0]
        entry_price = float(bar['open'])

        # Guard: skip if price data is invalid (zero, NaN, or negative)
        if entry_price <= 0 or np.isnan(entry_price):
            return

        # Limit-lock check
        locked, lock_type = _is_limit_locked(bar, 'bullish')
        if locked:
            return

        # ---- VWAP execution for large orders (>100万 RMB) ----
        raw_price = entry_price
        vwap_info = None
        try:
            from execution_algos import execute_with_vwap
            # Use total equity (not just cash) for VWAP eligibility check
            today_ts_vwap = pd.Timestamp(today)
            positions_value_pre = 0.0
            for sym_p, pos_p in self.positions.items():
                df_p = self.all_daily_data.get(sym_p)
                if df_p is not None and not df_p.empty:
                    df_sliced_v = df_p[df_p.index <= today_ts_vwap]
                    if not df_sliced_v.empty:
                        positions_value_pre += pos_p['shares'] * float(df_sliced_v['close'].iloc[-1])
                    else:
                        positions_value_pre += pos_p['shares'] * pos_p['entry_price']
                else:
                    positions_value_pre += pos_p['shares'] * pos_p['entry_price']
            pre_equity = self.cash + positions_value_pre
            est_value = pre_equity * self.position_pct
            if est_value >= 1_000_000:
                q_vwap = self._stock_quality.get(symbol, {})
                adv_shares_vwap = int(q_vwap.get('avg_vol', 1000000))
                vwap_info = execute_with_vwap(est_value, 0, raw_price,
                                              adv_shares_vwap, q_vwap.get('atr_pct', 0.03))
                vwap_px = vwap_info.get('execution_price', 0)
                if vwap_px > 0 and not np.isnan(vwap_px):
                    entry_price = vwap_px
                # else: keep original entry_price (VWAP returned invalid — fallback)
        except ImportError:
            pass

        # ---- Position sizing ----
        if VOL_ADAPTIVE_SIZING:
            q_vs = self._stock_quality.get(symbol, {})
            atr_pct_vs = q_vs.get('atr_pct', 0.03)
            vol_adj = TARGET_VOL_PCT / max(atr_pct_vs, 0.005)
            adjusted_pct = self.position_pct * vol_adj
            adjusted_pct = min(MAX_POSITION_PCT, max(MIN_POSITION_PCT, adjusted_pct))
        else:
            adjusted_pct = self.position_pct

        # === TASK 2: Dynamic Risk Budgeting ===
        # In bearish/neutral regimes, halve position size.
        # "敢于试错，但绝不重仓"
        mkt_features = signal.get('_market_features', {})
        if mkt_features.get('regime_bearish') or mkt_features.get('regime_neutral'):
            adjusted_pct *= 0.50
            adjusted_pct = max(MIN_POSITION_PCT, adjusted_pct)

        # Step 0: compute total equity (cash + mark-to-market positions at TODAY)
        today_ts = pd.Timestamp(today)
        positions_value = 0.0
        for sym_p, pos_p in self.positions.items():
            df_p = self.all_daily_data.get(sym_p)
            if df_p is not None and not df_p.empty:
                df_sliced = df_p[df_p.index <= today_ts]
                if not df_sliced.empty:
                    last_close_p = float(df_sliced['close'].iloc[-1])
                else:
                    last_close_p = pos_p['entry_price']
                positions_value += pos_p['shares'] * last_close_p
            else:
                positions_value += pos_p['shares'] * pos_p['entry_price']
        total_equity = self.cash + positions_value

        # Step 1: compute target purchase amount from total equity
        target_amount = total_equity * adjusted_pct

        # Cash constraint: cannot spend more than available cash
        if target_amount > self.cash:
            target_amount = self.cash

        raw_price = entry_price
        if raw_price <= 0 or np.isnan(raw_price) or target_amount < raw_price * A_SHARE_LOT_SIZE:
            return
        estimated_shares = int(target_amount / raw_price / A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
        if estimated_shares < A_SHARE_LOT_SIZE:
            return

        # Step 2: compute Almgren-Chriss slippage from estimated shares
        try:
            from market_microstructure import AlmgrenChrissImpact
            impact_model = AlmgrenChrissImpact()
            q_slip = self._stock_quality.get(symbol, {})
            adv_shares = int(q_slip.get('avg_vol', 1000000))
            atr_pct_ac = q_slip.get('atr_pct', 0.03)
            ac_slippage = impact_model.effective_slippage(
                estimated_shares, adv_shares, atr_pct_ac, is_buy=True)
            entry_price = raw_price * (1.0 + max(SLIPPAGE, ac_slippage))
        except ImportError:
            entry_price = raw_price * (1.0 + SLIPPAGE)

        # Step 3: compute actual shares at real entry price (with cash cap)
        shares = int(target_amount / entry_price / A_SHARE_LOT_SIZE) * A_SHARE_LOT_SIZE
        if shares < A_SHARE_LOT_SIZE:
            return

        # Stop loss and targets
        ensemble = signal.get('ensemble', {})
        local_plan = signal.get('local_plan', {})

        # Stop loss priority: local_plan > ensemble > default
        stop_loss = None
        if local_plan:
            sl = local_plan.get('stop_loss')
            if sl is not None:
                try:
                    stop_loss = float(sl)
                except (ValueError, TypeError):
                    pass
        if stop_loss is None:
            stop_loss = entry_price * (1.0 - self.stop_pct)

        # Targets priority: local_plan targets > ensemble > default
        targets: List[float] = []
        if local_plan:
            lp_targets = local_plan.get('targets', [])
            for t in lp_targets:
                if isinstance(t, dict):
                    try:
                        targets.append(float(t.get('price', 0)))
                    except (ValueError, TypeError):
                        pass
                elif isinstance(t, (int, float)):
                    targets.append(float(t))

        if not targets:
            targets = [entry_price * (1.0 + self.target_pct)]

        fallback_target = entry_price * (1.0 + self.target_pct)
        primary_target = min(t for t in targets if t > entry_price) if any(
            t > entry_price for t in targets
        ) else fallback_target

        # Snapshot of school signals for post-hoc analysis
        school_signals = ensemble.get('school_signals', {})

        # ── Chandelier Exit: initial state ──
        # Look up entry-day ATR (already precomputed as df['atr14'])
        entry_atr = stop_loss * 0  # will be overridden
        if df is not None and 'atr14' in df.columns:
            entry_row = df[df.index == pd.Timestamp(next_trading_day)]
            if not entry_row.empty:
                entry_atr = float(entry_row.iloc[0]['atr14'])
        if entry_atr <= 0:
            entry_atr = entry_price * 0.02  # fallback: 2% of price

        chandelier_active = CHANDELIER_ENABLED
        chandelier_initial_stop = entry_price - CHANDELIER_INITIAL_MULT * entry_atr
        chandelier_highest = entry_price  # tracks highest high since entry

        pos: Dict[str, Any] = {
            'symbol': symbol,
            'direction': 'long',
            'signal_date': today,
            'actual_entry_date': next_trading_day,
            'entry_idx': pd.Timestamp(next_trading_day),
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'target_price': primary_target,
            'targets': targets,
            'fallback_target': fallback_target,
            'ensemble_confidence': ensemble.get('ensemble_confidence', 0),
            'school_signals': school_signals,
            'init_capital_alloc': shares * entry_price,
            # ── Shadow Trading fields ──
            'shadow_rule_dir': ensemble.get('shadow_rule_dir', 'neutral'),
            'shadow_rule_conf': ensemble.get('shadow_rule_conf', 0.0),
            'shadow_ml_dir': ensemble.get('shadow_ml_dir', 'neutral'),
            'shadow_ml_conf': ensemble.get('shadow_ml_conf', 0.0),
            'master_source': ensemble.get('master_source', 'rule'),
            # ── Chandelier Exit state ──
            'chandelier_active': chandelier_active,
            'chandelier_atr_mult': CHANDELIER_ATR_MULT,
            'chandelier_initial_mult': CHANDELIER_INITIAL_MULT,
            'chandelier_highest': chandelier_highest,
            'chandelier_stop': chandelier_initial_stop,
        }
        self.positions[symbol] = pos
        self.cash -= shares * entry_price

    def _close_position(
        self,
        pos: Dict[str, Any],
        exit_price: float,
        exit_date: date,
        reason: str,
    ) -> Dict[str, Any]:
        """
        Compute P&L for a closed position and return a trade record dict.

        Costs: stamp duty (0.1 % sell only) + commission (0.025 % per side).
        All closes logged to EventStore for crash recovery.
        """
        entry_price = pos['entry_price']
        shares = pos['shares']

        gross_pnl_pct = (exit_price - entry_price) / entry_price

        # Trading costs
        buy_cost = entry_price * shares * COMMISSION
        sell_cost = exit_price * shares * (COMMISSION + STAMP_DUTY)
        gross_rmb = (exit_price - entry_price) * shares
        net_rmb = gross_rmb - buy_cost - sell_cost
        net_pnl_pct = net_rmb / (entry_price * shares)

        entry_dt = pos.get('actual_entry_date', pos.get('signal_date'))
        if isinstance(entry_dt, (date, datetime)):
            holding_days = (exit_date - (entry_dt if isinstance(entry_dt, date) else entry_dt.date())).days
        else:
            holding_days = 0

        trade: Dict[str, Any] = {
            'symbol': pos['symbol'],
            'direction': 'long',
            'trade_type': 'trend',
            'entry_date': str(entry_dt),
            'exit_date': str(exit_date),
            'entry_price': round(entry_price, 3),
            'exit_price': round(exit_price, 3),
            'shares': int(shares),
            'gross_pnl_pct': round(gross_pnl_pct * 100, 4),
            'net_pnl_pct': round(net_pnl_pct * 100, 4),
            'net_pnl_rmb': round(net_rmb, 2),
            'exit_reason': reason,
            'holding_days': int(holding_days),
            'ensemble_confidence': pos.get('ensemble_confidence', 0),
            'school_votes_json': pos.get('school_signals', {}),
            # ── Shadow Trading fields ──
            'shadow_rule_dir': pos.get('shadow_rule_dir', 'neutral'),
            'shadow_rule_conf': pos.get('shadow_rule_conf', 0.0),
            'shadow_ml_dir': pos.get('shadow_ml_dir', 'neutral'),
            'shadow_ml_conf': pos.get('shadow_ml_conf', 0.0),
            'master_source': pos.get('master_source', 'rule'),
        }
        # Event Sourcing: log every close to append-only event store
        try:
            from event_store import EventStore, EventType
            if not hasattr(self, '_event_store'):
                try:
                    self._event_store = EventStore()
                except Exception:
                    self._event_store = None  # DB not available — skip event logging
            is_win = net_pnl_pct > 0
            if self._event_store is not None:
                self._event_store.append(
                    EventType.POSITION_CLOSED, pos['symbol'],
                    {'shares': int(shares), 'entry_price': round(entry_price, 3),
                     'exit_price': round(exit_price, 3), 'pnl': round(net_rmb, 2),
                     'pnl_pct': round(net_pnl_pct * 100, 4), 'win': is_win,
                     'exit_reason': reason, 'holding_days': int(holding_days)}
                )
            # RLTF: learn from losing trades
            if not is_win and hasattr(self, '_rltf_learner'):
                from event_store import RLTFLearner
                self._rltf_learner.record_trade(
                    pos['symbol'],
                    {'confidence': pos.get('ensemble_confidence', 0),
                     '_indicators': {}, '_local_plan': {}},
                    {'pnl_pct': net_pnl_pct * 100, 'win': False,
                     'exit_reason': reason}
                )
        except ImportError:
            pass
        return trade

    # ========================================================================
    # Equity curve
    # ========================================================================

    def _update_equity(self, today: date) -> None:
        """
        Record equity = cash + market value of all open positions (at today's close).
        """
        total = self.cash
        for sym, pos in self.positions.items():
            df = self.all_daily_data.get(sym)
            if df is None or df.empty:
                total += pos['shares'] * pos['entry_price']  # fallback
                continue

            today_bar = df[df.index == pd.Timestamp(today)]
            if today_bar.empty:
                # Use last known price
                last_close = float(df['close'].iloc[-1])
                total += pos['shares'] * last_close
            else:
                close_price = float(today_bar.iloc[0]['close'])
                total += pos['shares'] * close_price

        self.equity_curve.append((today, round(total, 2)))

    # ========================================================================
    # Statistics
    # ========================================================================

    def compute_statistics(self) -> Dict[str, Any]:
        """
        Dual-track statistics: Trend signals vs T+0 grid, plus combined.

        Trades are split by ``trade_type``:
          - ``'trend'``   — 15-school ensemble entry → directional P&L
          - ``'t0_grid'`` — intraday grid scalping → cost-reduction P&L

        Returns a dict with three stat groups:
          - ``trend_stats``     — trend-only metrics
          - ``t0_stats``        — T+0-only metrics
          - ``combined_stats``  — Sharpe, total_return, max_dd (equity-curve based)
          PLUS flat keys for backward compatibility (total_trades, profit_factor, etc.)
        """
        # ── Helper: compute per-pool stats ──────────────────────────
        def _pool_stats(trades: list, pnl_key: str = 'net_pnl_pct') -> dict:
            """Compute win_rate, avg_win/loss, profit_factor for a trade sub-list."""
            n = len(trades)
            if n == 0:
                return {
                    'total_trades': 0, 'win_rate_pct': 0.0,
                    'avg_win': 0.0, 'avg_loss': 0.0,
                    'profit_factor': 0.0, 'total_pnl_rmb': 0.0,
                }
            pnl_vals = [t.get(pnl_key, 0) for t in trades]
            wins = [p for p in pnl_vals if p > 0]
            losses = [p for p in pnl_vals if p <= 0]
            total_pnl_rmb = sum(t.get('net_pnl_rmb', 0) for t in trades)
            return {
                'total_trades': n,
                'win_rate_pct': round(len(wins) / n * 100, 2),
                'avg_win': round(float(np.mean(wins)), 4) if wins else 0.0,
                'avg_loss': round(float(np.mean(losses)), 4) if losses else 0.0,
                'profit_factor': round(sum(wins) / abs(sum(losses)), 4) if sum(losses) != 0 else (999.99 if sum(wins) > 0 else 0.0),
                'total_pnl_rmb': round(total_pnl_rmb, 2),
            }

        # ── Equity-curve stats (combined — always from full curve) ───
        if not self.equity_curve:
            empty_pool = _pool_stats([])
            return {
                'total_return_pct': 0.0, 'annualized_return_pct': 0.0,
                'annualized_vol_pct': 0.0, 'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0,
                'total_trades': 0, 'win_rate_pct': 0.0,
                'avg_win_pct': 0.0, 'avg_loss_pct': 0.0,
                'profit_factor': 0.0, 'avg_holding_days': 0.0,
                'trend_stats': empty_pool, 't0_stats': empty_pool,
                'combined_stats': {},
            }

        dates, equities = zip(*self.equity_curve)
        eq_arr = np.array(equities, dtype=np.float64)
        initial_eq = self.initial_capital
        final_eq = eq_arr[-1]
        total_return = (final_eq - initial_eq) / initial_eq
        n_days = len(eq_arr)

        if n_days >= 2:
            daily_returns = np.diff(eq_arr) / eq_arr[:-1]
            ann_return = (final_eq / initial_eq) ** (252.0 / n_days) - 1.0
            ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252))
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns, ddof=1) * np.sqrt(252)) if np.std(daily_returns, ddof=1) > 0 else 0.0
            peak = np.maximum.accumulate(eq_arr)
            drawdowns = (peak - eq_arr) / peak
            max_dd = float(np.max(drawdowns))
        else:
            ann_return = total_return; ann_vol = 0.0; sharpe = 0.0; max_dd = 0.0

        # ── Split trades by type (backward-compat: no tag → 'trend') ─
        all_trades = self.trades
        trend_trades = [t for t in all_trades if t.get('trade_type', 'trend') == 'trend']
        t0_trades    = [t for t in all_trades if t.get('trade_type') == 't0_grid']

        trend_stats = _pool_stats(trend_trades, pnl_key='net_pnl_pct')
        t0_stats    = _pool_stats(t0_trades,    pnl_key='net_pnl_rmb')

        # ── Combined trade stats (all trades, %-based for trend, RMB for T0) ──
        combined_trades = len(all_trades)
        if combined_trades > 0:
            # Use net_pnl_rmb for win/loss (unified across trend + T0)
            all_pnl_rmb = [t.get('net_pnl_rmb', 0) for t in all_trades]
            all_wins = [p for p in all_pnl_rmb if p > 0]
            all_losses = [p for p in all_pnl_rmb if p <= 0]
            combined_win_rate = len(all_wins) / combined_trades
            combined_avg_win = float(np.mean(all_wins)) if all_wins else 0.0
            combined_avg_loss = float(np.mean(all_losses)) if all_losses else 0.0
            combined_pf = sum(all_wins) / abs(sum(all_losses)) if sum(all_losses) != 0 else (999.99 if sum(all_wins) > 0 else 0.0)
        else:
            combined_win_rate = 0.0; combined_avg_win = 0.0; combined_avg_loss = 0.0; combined_pf = 0.0

        # ── Avg holding days (trend trades only — T+0 has holding=0) ──
        if trend_trades:
            hd = [t.get('holding_days', 0) for t in trend_trades]
            avg_holding = float(np.mean(hd))
        else:
            avg_holding = 0.0

        rrr = (combined_avg_win / abs(combined_avg_loss)) if (combined_avg_loss != 0 and combined_avg_win != 0) else 0.0

        return {
            # Flat keys — backward compatible with save_to_db() & print_summary()
            'total_return_pct':      round(total_return * 100, 2),
            'annualized_return_pct': round(ann_return * 100, 2),
            'annualized_vol_pct':    round(ann_vol * 100, 2),
            'sharpe_ratio':          round(sharpe, 3),
            'max_drawdown_pct':      round(max_dd * 100, 2),
            'total_trades':          combined_trades,
            'win_rate_pct':          round(combined_win_rate * 100, 2),
            'avg_win_pct':           round(combined_avg_win, 2),
            'avg_loss_pct':          round(combined_avg_loss, 2),
            'profit_factor':         round(combined_pf, 2) if combined_pf != 999.99 else 999.99,
            'avg_holding_days':      round(avg_holding, 1),
            'reward_to_risk':        round(rrr, 2),
            'total_return_rmb':      round(final_eq - initial_eq, 2),
            'final_equity':          round(final_eq, 2),
            # ── Nested dual-track stats ──
            'trend_stats':           trend_stats,
            't0_stats':              t0_stats,
            'combined_stats': {
                'sharpe_ratio':      round(sharpe, 3),
                'total_return_pct':  round(total_return * 100, 2),
                'max_drawdown_pct':  round(max_dd * 100, 2),
                'annualized_return_pct': round(ann_return * 100, 2),
            },
        }

    # ========================================================================
    # Persistence
    # ========================================================================

    def save_to_db(self, run_label: str = "default") -> int:
        """
        Persist the completed backtest run to DuckDB using a context-managed
        connection (guaranteed release on success or failure — no lock leakage).
        """
        import os
        import json
        import duckdb

        stats = getattr(self, 'stats', None) or self.compute_statistics()
        db_path = 'data/trades.duckdb'
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)

        config_json = {
            'stock_pool': self.stock_pool,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'initial_capital': self.initial_capital,
            'position_pct': self.position_pct,
            'stop_pct': self.stop_pct,
            'target_pct': self.target_pct,
            'max_positions': self.max_positions,
            'stamp_duty': STAMP_DUTY,
            'commission': COMMISSION,
        }

        run_meta: Dict[str, Any] = {
            'run_label': run_label,
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'initial_capital': self.initial_capital,
            'position_pct': self.position_pct,
            'stop_pct': self.stop_pct,
            'target_pct': self.target_pct,
            'max_positions': self.max_positions,
            'total_return_pct': stats.get('total_return_pct', 0),
            'annualized_return_pct': stats.get('annualized_return_pct', 0),
            'sharpe_ratio': stats.get('sharpe_ratio', 0),
            'max_drawdown_pct': stats.get('max_drawdown_pct', 0),
            'total_trades': stats.get('total_trades', 0),
            'win_rate_pct': stats.get('win_rate_pct', 0),
            'avg_win_pct': stats.get('avg_win_pct', 0),
            'avg_loss_pct': stats.get('avg_loss_pct', 0),
            'profit_factor': stats.get('profit_factor', 0),
            'config_json': config_json,
            'weight_config': 'nuwa_adaptive',
            'school_weights_json': {},
        }

        # Serialize equity curve
        eq_serializable = [(str(d), v) for d, v in self.equity_curve]

        # ── Context-managed DuckDB: lock released on exit (success OR exception) ──
        with duckdb.connect(db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER,
                run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                run_label VARCHAR, start_date VARCHAR, end_date VARCHAR,
                initial_capital DOUBLE, position_pct DOUBLE, stop_pct DOUBLE,
                target_pct DOUBLE, max_positions INTEGER,
                total_return_pct DOUBLE, annualized_return_pct DOUBLE,
                sharpe_ratio DOUBLE, max_drawdown_pct DOUBLE,
                total_trades INTEGER, win_rate_pct DOUBLE,
                avg_win_pct DOUBLE, avg_loss_pct DOUBLE, profit_factor DOUBLE,
                config_json VARCHAR, equity_curve_json VARCHAR,
                weight_config VARCHAR, school_weights_json VARCHAR)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER,
                run_id INTEGER, symbol VARCHAR, direction VARCHAR,
                entry_date VARCHAR, exit_date VARCHAR,
                entry_price DOUBLE, exit_price DOUBLE, shares INTEGER,
                gross_pnl_pct DOUBLE, net_pnl_pct DOUBLE, net_pnl_rmb DOUBLE,
                exit_reason VARCHAR, holding_days INTEGER,
                ensemble_confidence DOUBLE, school_votes_json VARCHAR,
                shadow_rule_dir VARCHAR, shadow_rule_conf DOUBLE,
                shadow_ml_dir VARCHAR, shadow_ml_conf DOUBLE,
                master_source VARCHAR)''')

            # Manual auto-increment (DuckDB 1.5.3 does not support DEFAULT nextval)
            run_id = (conn.execute("SELECT COALESCE(MAX(id),0) FROM backtest_runs").fetchone()[0] or 0) + 1
            max_trade_id = (conn.execute("SELECT COALESCE(MAX(id),0) FROM backtest_trades").fetchone()[0] or 0)

            # Insert run metadata
            run_cols = ['id','run_label','start_date','end_date','initial_capital','position_pct',
                        'stop_pct','target_pct','max_positions','total_return_pct',
                        'annualized_return_pct','sharpe_ratio','max_drawdown_pct',
                        'total_trades','win_rate_pct','avg_win_pct','avg_loss_pct',
                        'profit_factor','config_json','equity_curve_json','weight_config',
                        'school_weights_json']
            run_meta['config_json'] = json.dumps(run_meta['config_json'], ensure_ascii=False)
            run_meta['equity_curve_json'] = json.dumps(eq_serializable, ensure_ascii=False)
            run_meta['school_weights_json'] = json.dumps(run_meta['school_weights_json'], ensure_ascii=False)
            run_vals = [run_id] + [run_meta.get(k) for k in run_cols[1:]]
            conn.execute(f"INSERT INTO backtest_runs ({','.join(run_cols)}) VALUES ({','.join(['?']*len(run_cols))})", run_vals)

            # Insert trades (manual auto-increment id)
            t_cols = ['id','run_id','symbol','direction','entry_date','exit_date','entry_price',
                      'exit_price','shares','gross_pnl_pct','net_pnl_pct','net_pnl_rmb',
                      'exit_reason','holding_days','ensemble_confidence','school_votes_json',
                      'shadow_rule_dir','shadow_rule_conf','shadow_ml_dir','shadow_ml_conf',
                      'master_source']
            sv_idx = t_cols.index('school_votes_json')
            for t in self.trades:
                max_trade_id += 1
                t_vals = [max_trade_id, run_id] + [t.get(k) for k in t_cols[2:]]
                t_vals[sv_idx] = json.dumps(t_vals[sv_idx]) if not isinstance(t_vals[sv_idx], str) else t_vals[sv_idx]
                conn.execute(f"INSERT INTO backtest_trades ({','.join(t_cols)}) VALUES ({','.join(['?']*len(t_cols))})", t_vals)

        return run_id

    # ========================================================================
    # Summary
    # ========================================================================

    def _get_benchmark_return(self) -> Optional[float]:
        """Compute CSI 300 (000300) buy-and-hold return over the backtest period."""
        try:
            from data_loader import get_daily_kline
            bm = get_daily_kline('000300', days=400)
            if bm is None or bm.empty:
                return None
            if not isinstance(bm.index, pd.DatetimeIndex):
                bm.index = pd.to_datetime(bm.index)
            bm.sort_index(inplace=True)
            start_mask = bm.index <= pd.Timestamp(self.start_date or self.trading_days[0])
            end_mask = bm.index <= pd.Timestamp(self.end_date)
            if start_mask.any() and end_mask.any():
                bm_start = float(bm.loc[start_mask, 'close'].iloc[-1])
                bm_end = float(bm.loc[end_mask, 'close'].iloc[-1])
                return (bm_end - bm_start) / bm_start * 100
        except Exception:
            pass
        return None

    def _print_summary(self) -> None:
        """Pretty-print backtest results."""
        stats = getattr(self, 'stats', None) or self.compute_statistics()
        benchmark_ret = self._get_benchmark_return()
        alpha = stats.get('total_return_pct', 0) - (benchmark_ret or 0)
        print("\n" + "=" * 60)
        print("  BACKTEST SUMMARY")
        print("=" * 60)
        print(f"  Period          : {self.start_date} -> {self.end_date}")
        print(f"  Trading days    : {len(self.trading_days)}")
        print(f"  Initial capital : {self.initial_capital:,.0f} CNY")
        print(f"  Final equity    : {stats.get('final_equity', 0):,.0f} CNY")
        print(f"  Total return    : {stats.get('total_return_pct', 0):+.2f}%")
        print(f"  Annualized ret  : {stats.get('annualized_return_pct', 0):+.2f}%")
        print(f"  Annualized vol  : {stats.get('annualized_vol_pct', 0):.2f}%")
        print(f"  Sharpe ratio    : {stats.get('sharpe_ratio', 0):.3f}")
        print(f"  Max drawdown    : {stats.get('max_drawdown_pct', 0):.2f}%")
        if benchmark_ret is not None:
            print(f"  CSI 300 B&H     : {benchmark_ret:+.2f}%")
            print(f"  Alpha (excess)  : {alpha:+.2f}%")
        print(f"  Total trades    : {stats.get('total_trades', 0)}")
        print(f"  Win rate        : {stats.get('win_rate_pct', 0):.1f}%")
        print(f"  Avg win         : {stats.get('avg_win_pct', 0):+.2f}%")
        print(f"  Avg loss        : {stats.get('avg_loss_pct', 0):+.2f}%")
        print(f"  Profit factor   : {stats.get('profit_factor', 0):.2f}")
        print(f"  Avg hold (days) : {stats.get('avg_holding_days', 0):.1f}")
        print(f"  Reward/Risk     : {stats.get('reward_to_risk', 0):.2f}")
        # Exit reason breakdown
        if self.trades:
            from collections import Counter
            reasons = Counter(t.get('exit_reason', 'unknown') for t in self.trades)
            reason_str = ' | '.join(f'{k}={v}' for k, v in reasons.most_common(5))
            print(f"  Exit reasons    : {reason_str}")
        if hasattr(self, '_bullish_total') and self._bullish_total > 0:
            pct = self._bullish_passed / self._bullish_total * 100
            print(f"  Signal filter   : {self._bullish_passed}/{self._bullish_total} "
                  f"bullish passed ({pct:.1f}%)  threshold={self.confidence_threshold:.2f}")
        # ── Dual-track breakdown ──
        trend_s = stats.get('trend_stats', {})
        t0_s    = stats.get('t0_stats', {})
        if trend_s or t0_s:
            print("  ---- Trend vs T+0 Breakdown ----")
            def _p(stat, label, fmt=''):
                t_v = trend_s.get(stat, 0); t0_v = t0_s.get(stat, 0)
                print(f"  {label:<20} Trend: {t_v:{fmt}}  |  T+0: {t0_v:{fmt}}")
            _p('total_trades',    'Trades',          '>5d')
            _p('win_rate_pct',    'Win Rate',        '>6.1f')
            _p('profit_factor',   'Profit Factor',   '>6.2f')
            _p('total_pnl_rmb',   'Total PnL (RMB)', '>,.0f')
        # ── School attribution ──
        try:
            attr_df = self.compute_school_attribution()
            self._print_school_attribution(attr_df)
        except Exception:
            pass  # attribution is informative; never crash the backtest
        print("=" * 60)


# =========================================================================
# __main__ — quick demo
# =========================================================================

    # ========================================================================
    # School Attribution (Task 2)
    # ========================================================================

    def compute_school_attribution(self) -> 'pd.DataFrame':
        """
        Fractional P&L attribution across the 15-school ensemble.

        **Fractional rule**: a trade's net P&L is split equally among all
        schools that voted *bullish* at entry.  If 5 schools voted bullish and
        the trade made +10,000 RMB, each school is credited +2,000 RMB.
        Losses are split identically — schools share both glory and blame.

        **Lone-wolf bonus**: trades where only a single school voted bullish
        are tracked separately as "independent" P&L — the purest alpha signal.

        Returns
        -------
        pd.DataFrame indexed by school key, with columns:
          - label, short_label
          - participations   : number of trades the school joined
          - lone_wolf_trades : trades where it was the ONLY voter
          - lone_wolf_pnl    : total P&L from lone-wolf trades (RMB)
          - attributed_pnl   : fractional P&L summed across all joined trades
          - avg_pnl_per_trade: attributed_pnl / participations
          - win_rate         : fraction of joined trades that were winners
        """
        import pandas as pd
        from expert_ensemble import SCHOOLS

        # Only trend trades carry school votes
        trend_trades = [t for t in self.trades if t.get('trade_type', 'trend') == 'trend']
        if not trend_trades:
            return pd.DataFrame()

        # ── Accumulators per school ──
        schools = list(SCHOOLS.keys())
        acc = {s: {'parts': 0, 'wins': 0, 'attr_pnl': 0.0,
                   'lone_trades': 0, 'lone_pnl': 0.0}
               for s in schools}

        for trade in trend_trades:
            sv = trade.get('school_votes_json', {})
            if isinstance(sv, str):
                try:
                    import json
                    sv = json.loads(sv)
                except Exception:
                    sv = {}

            if not sv:
                continue

            # Find bullish-voting schools
            bullish_schools = []
            for name in schools:
                vote = sv.get(name, {})
                if not isinstance(vote, dict):
                    continue
                if str(vote.get('direction', '')).lower() == 'bullish':
                    bullish_schools.append(name)

            n_voters = len(bullish_schools)
            if n_voters == 0:
                continue

            pnl = trade.get('net_pnl_rmb', 0)
            is_win = pnl > 0
            share = pnl / n_voters

            for name in bullish_schools:
                acc[name]['parts'] += 1
                acc[name]['attr_pnl'] += share
                if is_win:
                    acc[name]['wins'] += 1

            # Lone-wolf detection
            if n_voters == 1:
                lone_name = bullish_schools[0]
                acc[lone_name]['lone_trades'] += 1
                acc[lone_name]['lone_pnl'] += pnl

        # ── Build DataFrame ──
        rows = []
        for name in schools:
            a = acc[name]
            p = a['parts']
            rows.append({
                'school':        name,
                'label':         SCHOOLS.get(name, {}).get('label', name),
                'short_label':   SCHOOLS.get(name, {}).get('short_label', ''),
                'participations': p,
                'win_rate':      round(a['wins'] / p * 100, 1) if p > 0 else 0.0,
                'attributed_pnl': round(a['attr_pnl'], 2),
                'avg_pnl_per_trade': round(a['attr_pnl'] / p, 2) if p > 0 else 0.0,
                'lone_wolf_trades': a['lone_trades'],
                'lone_wolf_pnl':    round(a['lone_pnl'], 2),
            })

        df = pd.DataFrame(rows).set_index('school')
        # Sort by attributed PnL descending
        df = df.sort_values('attributed_pnl', ascending=False)
        return df

    def _print_school_attribution(self, df: 'pd.DataFrame') -> None:
        """Pretty-print the school attribution table."""
        if df.empty:
            return
        print("\n" + "=" * 85)
        print("  SCHOOL ATTRIBUTION — Fractional P&L (Trend Trades Only)")
        print("=" * 85)
        header = (f"  {'School':<18} {'Part':>5} {'Win%':>7} "
                  f"{'Attr P&L':>12} {'Avg/Trade':>10} {'Lone':>5} {'Lone P&L':>10}")
        print(header)
        print("  " + "-" * 80)
        for idx, row in df.iterrows():
            label = row.get('short_label', idx)[:16]
            print(f"  {label:<18} {int(row['participations']):>5} "
                  f"{row['win_rate']:>6.1f}% {row['attributed_pnl']:>11,.0f} "
                  f"{row['avg_pnl_per_trade']:>9,.0f} "
                  f"{int(row['lone_wolf_trades']):>5} {row['lone_wolf_pnl']:>9,.0f}")
        print("=" * 85)


if __name__ == '__main__':
    end = date.today()
    start = end - timedelta(days=180)
    engine = BacktestEngine(start_date=start, end_date=end)
    engine.run_backtest()
    stats = engine.compute_statistics()
    run_id = engine.save_to_db(run_label=f"default_{start}_{end}")
    print(f"\nSaved as run_id={run_id}")
