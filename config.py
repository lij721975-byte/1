# config.py
import os

# DeepSeek API Key（优先从环境变量读取，否则使用默认值）
# 设置环境变量: set DEEPSEEK_API_KEY=your_key (Windows) 或 export DEEPSEEK_API_KEY=your_key (Linux/Mac)
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', 'KEY密匙')


# 账户资金（默认10万，实盘请修改）
ACCOUNT_EQUITY = 100_000

# 富易安装目录（一般是 C:\zd_ghtth，若不是请修改）
FUYI_TDX_DIR = r"E:\new_haitong"

# 股票池 — 从自选股目录加载（优先读取 自选股/stocks.py，否则回退到硬编码列表）
try:
    from watchlist.stocks import WATCHLIST as STOCK_POOL
except ImportError:
    STOCK_POOL = [
        "000012", "000059", "000060", "000400", "000629",
        "000725", "000762", "000930", "000932", "000938",
        "002010", "002092", "002185", "002352", "002532",
        "159985", "600110", "600127",
        "600330", "600460", "600585", "600707", "600740",
        "601168", "601628", "601678", "603072", "603078",
        "603659", "603690", "603799", "603936", "605020",
        "002007", "002173", "002241", "002317", "002405",
        "002565", "002624", "300015", "300115", "300146",
        "300189", "300285", "300302", "300316", "300319",
        "300429", "300498", "300589", "300623", "301188",
        "600226", "600265", "600820", "603002", "603983",
        "605319",
    ]

# 行情服务器（备用，用于分钟线或实时补充）
TDX_SERVER_IP = "119.147.212.81"
TDX_SERVER_PORT = 7709

# 数据库路径
DB_PATH = "data/trades.duckdb"

# 股票名称映射（代码 → 名称）
STOCK_NAMES = {
    "002007": "华兰生物", "002173": "创新医疗", "002241": "歌尔股份",
    "002317": "众生药业", "002405": "四维图新", "002565": "顺灏股份",
    "002624": "完美世界", "300015": "爱尔眼科", "300115": "长盈精密",
    "300146": "汤臣倍健", "300189": "神农种业", "300285": "国瓷材料",
    "300302": "同有科技", "300316": "晶盛机电", "300319": "麦捷科技",
    "300429": "强力新材", "300498": "温氏股份", "300589": "江龙船艇",
    "300623": "捷捷微电", "301188": "力诺特玻", "600110": "诺德股份",
    "600226": "瀚叶股份", "600265": "景谷林业", "600707": "彩虹股份",
    "600820": "隧道股份", "603002": "宏昌电子", "603983": "丸美股份",
    "000012": "南玻A", "000059": "华锦股份", "000060": "中金岭南",
    "000400": "许继电气", "000629": "钒钛股份", "000725": "京东方A",
    "000762": "西藏矿业", "000930": "中粮科技", "000932": "华菱钢铁",
    "000938": "紫光股份", "002010": "传化智联", "002092": "中泰化学",
    "002185": "华天科技", "002352": "韵达股份", "002532": "天山铝业",
    "159985": "豆粕ETF",
    "600127": "金健米业", "600330": "天通股份", "600460": "士兰微",
    "600585": "海螺水泥", "600740": "山西焦化", "601168": "西部矿业",
    "601628": "中国人寿", "601678": "滨化股份", "603072": "天和磁材",
    "603078": "江化微", "603659": "璞泰来", "603690": "至纯科技",
    "603799": "华友钴业", "603936": "博敏电子", "605020": "永和股份",
    "605319": "无锡振华",
}

# 行业板块分类（基于申万一级行业简化）
STOCK_SECTORS = {
    # 电子
    "000725": "电子", "002185": "电子", "002241": "电子", "300115": "电子",
    "300319": "电子", "300623": "电子", "600460": "电子", "600707": "电子",
    "603005": "电子", "603078": "电子", "603690": "电子", "603936": "电子",
    # 新能源/电力设备
    "000400": "电力设备", "300316": "电力设备", "300429": "电力设备",
    "603659": "电力设备", "600110": "电力设备", "600330": "电力设备",
    # 有色/材料
    "000059": "化工", "000060": "有色金属", "000629": "有色金属", "000762": "有色金属",
    "000930": "农林牧渔", "000932": "钢铁", "002092": "化工", "002532": "有色金属",
    "300285": "基础化工", "600740": "煤炭", "601168": "有色金属",
    "603072": "有色金属", "603799": "有色金属", "605020": "基础化工",
    # 医药生物
    "002007": "医药生物", "002173": "医药生物", "002317": "医药生物",
    "300015": "医药生物", "300146": "医药生物", "603259": "医药生物",
    # 计算机/传媒/通信
    "000938": "计算机", "002405": "计算机", "002624": "传媒",
    "300302": "计算机", "300339": "计算机",
    # 金融
    "601628": "非银金融",
    # 建材/地产
    "000012": "建筑材料", "600585": "建筑材料",
    # 消费/零售
    "002010": "交通运输", "002352": "交通运输",
    "002565": "轻工制造", "603983": "美容护理",
    # 农牧
    "159985": "商品ETF", "300189": "农林牧渔",
    "300498": "农林牧渔", "600127": "农林牧渔",
    # 机械/汽车
    "300589": "国防军工", "600820": "建筑装饰",
    "605319": "汽车", "301188": "医药生物",
    # 公用/环保
    "601678": "基础化工", "600226": "基础化工",
    "600265": "农林牧渔", "603002": "电子",
}


def get_stock_sector(symbol):
    """获取股票所属行业板块"""
    return STOCK_SECTORS.get(symbol, _guess_sector_by_prefix(symbol))


def _guess_sector_by_prefix(symbol):
    """根据代码前缀猜测板块"""
    s = str(symbol)
    if s.startswith('60') or s.startswith('603') or s.startswith('605'):
        return '沪市主板'
    elif s.startswith('00'):
        return '深市主板'
    elif s.startswith('30'):
        return '创业板'
    elif s.startswith('688'):
        return '科创板'
    return '其他'


# ============================================================
# Central indicator parameters (previously hardcoded literals)
# ============================================================

# Moving average periods
MA_SHORT = 5
MA_MEDIUM = 20
MA_LONG = 60
MA_EXTRA_LONG = 120

# Bollinger Bands
BB_PERIOD = 20
BB_STDDEV_UP = 2.0
BB_STDDEV_DOWN = 2.0

# RSI
RSI_PERIOD = 14

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# ATR
ATR_PERIOD = 14

# KMeans clustering
KMEANS_N_CLUSTERS = 6

# Verification
SR_VERIFICATION_WINDOW = 20      # bars to check price near S/R level
SR_PROXIMITY_PCT = 0.015         # 1.5% threshold for "near" a level
SR_STRENGTH_BOOST = 15           # bonus for volume-verified levels

# Lookback windows
TANG_LOOKBACK_DAYS = 60          # Tang Nengtong 2T-1L & 33-filter lookback
VOL_MA_DAYS = 21                 # Volume moving average days for verification

# Data loading
DAILY_BARS = 250                 # Default daily K-line count
HOURLY_BARS = 120                # Default hourly resolution bars
WEEKLY_BARS = 150                # Default weekly bars

# Threading
MAX_DEEPSEEK_WORKERS = 15        # Parallel API call count

# Risk management
MAX_PORTFOLIO_EXPOSURE = 0.80    # Max total portfolio weight
MAX_SECTOR_EXPOSURE = 0.35       # Max single sector weight
DEFAULT_POSITION_PCT = 0.05      # Default position size when not specified
A_SHARE_LOT_SIZE = 100           # A股最小交易单位（1手=100股）

# ============================================================
# A-Share holiday calendar (2025-2026)
# ============================================================
A_SHARE_HOLIDAYS = {
    # 2025
    "2025-01-01",  # New Year
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",  # Spring Festival
    "2025-02-03", "2025-02-04",  # Spring Festival (cont.)
    "2025-04-04", "2025-04-05", "2025-04-06",  # Qingming
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",  # Labor Day
    "2025-05-31", "2025-06-01", "2025-06-02",  # Dragon Boat
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05",  # National Day + Mid-Autumn
    "2025-10-06", "2025-10-07", "2025-10-08",
    # 2026
    "2026-01-01", "2026-01-02", "2026-01-03",  # New Year
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",  # Spring Festival
    "2026-02-21", "2026-02-22",
    "2026-04-05", "2026-04-06", "2026-04-07",  # Qingming
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",  # Labor Day
    "2026-06-19", "2026-06-20", "2026-06-21",  # Dragon Boat
    "2026-09-25", "2026-09-26", "2026-09-27",  # Mid-Autumn
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",  # National Day
    "2026-10-06", "2026-10-07",
}

# Backtest / feedback
FEEDBACK_LOOKBACK = 30           # Signals to evaluate
RECENCY_HALFLIFE = 15            # Days for half-weight in recency-weighted scoring
MIN_LOOKBACK_DAYS = 250          # Minimum trading days required for a stock to be evaluated
DEFAULT_STOP_PCT = 0.06          # Default stop-loss percentage (6%)
DEFAULT_TARGET_PCT = 0.20        # Default take-profit percentage (20%)
DEFAULT_POSITION_PCT = 0.05      # Default position size (5% of capital)

# ---- Time-based exit ----
MAX_HOLDING_DAYS = 30            # Auto-close positions held longer than this
TIME_STOP_ENABLED = True         # Enable time-based exit

# ---- Entry quality filters ----
MIN_DAILY_VOLUME = 3_000_000     # Minimum avg daily volume (shares) for liquidity
MIN_DAILY_AMOUNT = 50_000_000    # Minimum avg daily turnover (CNY) ≈5000万
TREND_FILTER_MA = 60             # MA period for individual stock trend filter
TREND_FILTER_ENABLED = True      # Block entries when price < MA(trend_period)

# ---- Trailing stop ----
TRAILING_STOP_ENABLED = True     # Enable ATR-based trailing stop (primary exit)
TRAILING_STOP_ACTIVATION = 0.03  # Activate after 3% profit from entry
TRAILING_STOP_ATR_MULT = 2.0     # Trail at 2×ATR below highest high

# ---- Chandelier Exit (吊灯止损) ----
CHANDELIER_ENABLED = False          # DISABLED — trailing_stop replaces it (53.8% vs 28.6% win rate)
CHANDELIER_ATR_MULT = 4.0           # Trailing-stop distance: N × ATR below highest high
CHANDELIER_INITIAL_MULT = 3.0       # Initial-stop distance: N × entry_ATR below entry
CHANDELIER_ACTIVATION_PCT = 0.03    # Only ratchet stop UP after profit exceeds 3%

# ---- Consensus gate & school blacklist ----
BANNED_SCHOOLS = []                  # All 15 schools participate
MIN_CONSENSUS_VOTES = 3              # Minimum valid bullish votes
MAX_CONSENSUS_VOTES = 8              # Max consensus before herding

# ---- Trade immunity & noise filters ----
MIN_HOLDING_DAYS = 5                # No trailing exit before 5 days (hard stop excepted)
DISABLE_MA15_EXIT = True            # Disable toxic MA15 exit (8.8% win rate)

# ---- ML Shadow Trading (Champion/Challenger) ----
USE_ML_AS_MASTER = False            # False=规则引擎实盘,ML跑影子; True=ML实盘,规则跑影子

# ---- Volatility-adaptive position sizing ----
VOL_ADAPTIVE_SIZING = True       # Scale position by vol target
TARGET_VOL_PCT = 0.02            # Target daily volatility (2% of capital)
MAX_POSITION_PCT = 0.10          # Cap position size at 10%
MIN_POSITION_PCT = 0.02          # Floor position size at 2%

# ---- Nüwa / Ensemble verbosity ----
NUWA_VERBOSE = False             # Set True for debugging school weights

# Backtest parameter sweep — grid search dimensions
BACKTEST_SWEEP_PARAMS = {
    'position_pct': [0.03, 0.05, 0.08, 0.10],
    'stop_pct': [0.05, 0.06, 0.07],
    'target_pct': [0.15, 0.20, 0.30],
    'max_positions': [5, 8, 10, 15],
}
# Weight schemes to compare in sweep
BACKTEST_WEIGHT_SCHEMES = [
    'nuwa_adaptive',
    'regime_only',
    'equal',
    'learned',
]