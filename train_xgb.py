#!/usr/bin/env python
# train_xgb.py — Offline XGBoost training from DuckDB backtest trades
import os
import json
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path

# ⚠️ 绝对红线：这里的顺序必须与 expert_ensemble.py 中的 _ML_SCHOOL_KEYS 100% 保持一致！
ML_SCHOOL_KEYS = [
    'school_chanlun', 'school_tang', 'school_livermore', 'school_busch',
    'school_classical', 'school_risk', 'school_gann', 'school_wyckoff',
    'school_harmonic', 'school_roc_breakout', 'school_volume_profile',
    'school_fusion', 'school_mean_reversion', 'school_capital_flow',
    'school_pattern_features', 'school_brooks_pa',
]

DB_PATH = 'data/trades.duckdb'
MODEL_DIR = Path(__file__).parent / 'models'

def load_data_from_db():
    print(">>> 1. 从 DuckDB 读取回测历史数据...")
    conn = duckdb.connect(DB_PATH)
    # 提取需要做特征和标签的数据，加入 net_pnl_pct 和 holding_days 做标签惩罚
    query = """
        SELECT 
            school_votes_json, 
            net_pnl_pct, 
            holding_days
        FROM backtest_trades
        WHERE school_votes_json IS NOT NULL
    """
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def feature_engineering(df):
    print(">>> 2. 解析 JSON 并进行特征工程降维...")
    X_list = []
    Y_list = []

    for idx, row in df.iterrows():
        try:
            # 1. 解析投票数据
            votes = json.loads(row['school_votes_json']) if isinstance(row['school_votes_json'], str) else row['school_votes_json']
            
            # 2. 构建特征向量 (Feature Vector)
            feat = np.zeros(len(ML_SCHOOL_KEYS), dtype=np.float64)
            for j, key in enumerate(ML_SCHOOL_KEYS):
                sig = votes.get(key, {})
                direction = str(sig.get('direction', 'neutral')).lower()
                confidence = float(sig.get('confidence', 0.0))
                
                # 特征值 = 方向映射 * 置信度
                if direction == 'bullish':
                    d_num = 1.0
                elif direction == 'bearish':
                    d_num = -1.0
                else:
                    d_num = 0.0
                feat[j] = d_num * confidence
            
            # 3. 标签打标 (Labeling) - 引入盈利阈值与持仓期惩罚
            pnl = row['net_pnl_pct']
            days = row['holding_days']
            
            if pnl > 0.03:  # 利润大于3%才是真正优质的 Alpha
                y = 1  # Bullish
            elif pnl < -0.02 or (pnl < 0 and days > 5): 
                # 亏损超过2%，或者虽然亏损不多但死扛超过5天（慢性放血），视为毒药交易
                y = -1 # Bearish / Avoid
            else:
                y = 0  # Neutral (鸡肋交易，模型应学会观望)

            X_list.append(feat)
            Y_list.append(y)
            
        except Exception as e:
            continue

    # 将 Y 从 [-1, 0, 1] 映射为 XGBoost 要求的类别索引 [0, 1, 2]
    # 映射关系: -1 -> 0, 0 -> 1, 1 -> 2
    Y_mapped = [y + 1 for y in Y_list]
    
    return np.array(X_list), np.array(Y_mapped)

def train_and_save_model(X, Y):
    print(f">>> 3. 启动 XGBoost 训练 (样本量: {len(X)})...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 针对金融数据的超参数调优 (防过拟合)
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        n_estimators=150,
        max_depth=4,         # 限制树深度，防止在噪声中过拟合
        learning_rate=0.05,
        subsample=0.8,       # 随机抽取80%样本训练
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10
    )

    print("\n>>> 4. 样本外测试集准确率评估:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Bearish(-1)', 'Neutral(0)', 'Bullish(1)']))

    # 保存模型
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'xgb_master.json'
    model.save_model(str(model_path))
    print(f">>> 5. 模型已成功保存至: {model_path}")

if __name__ == '__main__':
    df_raw = load_data_from_db()
    if df_raw.empty:
        print("[ERROR] DuckDB has no backtest data. Run baseline backtest first with USE_ML_AS_MASTER=False.")
    else:
        X, Y = feature_engineering(df_raw)
        train_and_save_model(X, Y)