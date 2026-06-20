# data_loader.py
import pandas as pd
import os
import csv
from datetime import date
from mootdx.reader import Reader
from config import FUYI_TDX_DIR, STOCK_NAMES

_reader = None  # 延迟初始化，避免导入时TDX路径不存在导致崩溃
_delisted: dict = None  # symbol -> delist_date mapping


def _load_delisted():
    """Load delisted stocks registry from CSV (one-time)."""
    global _delisted
    if _delisted is not None:
        return _delisted
    _delisted = {}
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'delisted_stocks.csv')
    if not os.path.exists(csv_path):
        # Create empty template if missing
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            f.write('symbol,delist_date,name\n')
        return _delisted
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = str(row.get('symbol', '')).strip().zfill(6)
            d = str(row.get('delist_date', '')).strip()
            if sym and d:
                _delisted[sym] = d
    return _delisted


def is_delisted(symbol: str, check_date: date = None) -> bool:
    """Check if a stock was already delisted by check_date."""
    d = _load_delisted()
    if symbol not in d:
        return False
    if check_date is None:
        return True
    return check_date.isoformat() >= d[symbol]


def _get_reader():
    global _reader
    if _reader is None:
        _reader = Reader.factory(market='std', tdxdir=FUYI_TDX_DIR)
    return _reader


def get_stock_name(symbol):
    """获取股票名称，优先从映射表读取"""
    return STOCK_NAMES.get(symbol, symbol)


def get_daily_kline(symbol, days=250):
    """获取日线，默认250根（≥100根日线等量要求）。
    自动应用前复权(qfq)，并检查退市状态。
    """
    try:
        df = _get_reader().daily(symbol=symbol)
        if df is None or df.empty:
            print(f"[WARN] {symbol} 本地无日线数据")
            return None

        # 前复权(qfq): 如果存在qfq列则使用，否则尝试从reader获取复权数据
        if 'qfq' not in [c.lower() for c in df.columns]:
            try:
                df_qfq = _get_reader().daily(symbol=symbol, factor='qfq')
                if df_qfq is not None and not df_qfq.empty:
                    # Use forward-adjusted close as primary OHLC
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df_qfq.columns:
                            df[col] = df_qfq[col]
            except Exception:
                pass  # Fallback: use raw prices

        return df.tail(days)
    except Exception as e:
        print(f"获取日线出错 {symbol}: {e}")
        return None


def get_hourly_kline(symbol, hours=120):
    """小时线（由1分钟线合成5分钟线，再截取小时数），默认120小时（5个交易日）"""
    try:
        df = _get_reader().minute(symbol=symbol)
        if df is None or df.empty:
            print(f"⚠️ {symbol} 本地无分钟线数据")
            return None

        # 统一列名，兼容不同mootdx版本
        if 'close' not in df.columns and 'price' in df.columns:
            df.rename(columns={'price': 'close'}, inplace=True)
        if 'close' not in df.columns:
            rename_map = {}
            for col in df.columns:
                if col.lower() == 'close':
                    rename_map[col] = 'close'
                elif col.lower() == 'volume':
                    rename_map[col] = 'volume'
            if rename_map:
                df.rename(columns=rename_map, inplace=True)
            else:
                df = df.iloc[:, :5]
                df.columns = ['open', 'high', 'low', 'close', 'volume']

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        ohlc = df['close'].resample('5min').ohlc()
        vol  = df['volume'].resample('5min').sum() if 'volume' in df.columns else pd.Series(index=ohlc.index, data=0)
        df_5min = pd.concat([ohlc, vol.rename('volume')], axis=1).dropna()

        return df_5min.tail(hours * 12)

    except Exception as e:
        print(f"小时线出错 {symbol}: {e}")
        return None


def resample_5min_to_60min(df_5min):
    """将5分钟OHLCV重采样为60分钟OHLCV。"""
    if df_5min is None or df_5min.empty:
        return None
    if not isinstance(df_5min.index, pd.DatetimeIndex):
        df_5min.index = pd.to_datetime(df_5min.index)
    ohlc = df_5min['close'].resample('60min').ohlc()
    vol = df_5min['volume'].resample('60min').sum() if 'volume' in df_5min.columns else pd.Series(index=ohlc.index, data=0.0)
    df_60 = pd.concat([ohlc, vol.rename('volume')], axis=1).dropna()
    return df_60


def get_weekly_kline(symbol, weeks=150):
    """获取周线（通过日线合成），默认150周"""
    daily = get_daily_kline(symbol, days=weeks * 5 + 10)
    if daily is None or daily.empty:
        return None
    weekly = daily.resample('W').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    return weekly