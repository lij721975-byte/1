# refresh_reminder.py
import time
from plyer import notification

INTERVAL_MINUTES = 15   # 提醒间隔（分钟）

if __name__ == '__main__':
    while True:
        notification.notify(
            title="⏰ 数据刷新",
            message="请在富易中按 Ctrl+N 下载5分钟线数据",
            timeout=15
        )
        time.sleep(INTERVAL_MINUTES * 60)
