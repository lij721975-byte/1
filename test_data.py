# test_data.py
from mootdx.reader import Reader

# 你的实际数据目录（vipdoc 所在的目录）
reader = Reader.factory(market='std', tdxdir=r'C:\GTJA\RichEZ\newVer')

# ================== 日线测试 ==================
df_day = reader.daily(symbol='600036')
if df_day is None or df_day.empty:
    print("❌ 日线数据为空，请检查日线是否已下载，或路径是否正确")
else:
    print("✅ 日线最后几行：")
    print(df_day.tail())

print("-" * 40)

# ================== 1分钟线测试 ==================
df_min = reader.minute(symbol='600036')
if df_min is None:
    print("❌ 1分钟线数据为 None！通常是未下载1分钟线数据，请用富易下载")
elif df_min.empty:
    print("❌ 1分钟线数据为空 DataFrame")
else:
    print("✅ 1分钟线前几行：")
    print(df_min.head())

# 如果需要测试5分钟线，可以加：
# df_5min = reader.fzline(symbol='600036')
# print(df_5min.head())