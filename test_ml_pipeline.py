# test_ml_pipeline.py
import sys
import io
from datetime import date

# 强制终端输出 UTF-8，防止中文乱码报错
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from ml_feature_pipeline import build_and_train
from mypool_3000 import SELECTED_POOL

if __name__ == '__main__':
    print("="*50)
    print("🚀 开始机器学习特征采集与训练测试 (极小样本)...")
    print("="*50)
    
    # 设定测试时间段：只跑最近 1 年的数据（例如 2023-01-01 到 2026-06-05）
    # 请确保您的富易本地数据已经下载了这段时间的日线
    start_d = date(2025, 1, 1)
    end_d = date(2026, 6, 1)
    
    try:
        # 使用 100 只股票的测试池
        build_and_train(SELECTED_POOL, start_d, end_d)
        print("\n✅ 测试完美通过！您现在拥有了一个机构级的 XGBoost 模型。")
    except Exception as e:
        print("\n❌ 测试运行失败，报错信息如下：")
        import traceback
        traceback.print_exc()