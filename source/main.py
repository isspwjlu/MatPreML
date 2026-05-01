"""MatPreML - 材料性能机器学习预测平台 启动入口"""

import sys
import os
# 添加上级目录到系统路径，确保可以直接运行main.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from MatPreML_refactored.data_analyzer import DataAnalyzer


def main():
    """主函数 - 创建并启动应用程序"""
    # 启用高分屏适配
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName('MatPreML')
    app.setApplicationVersion('2.4')

    window = DataAnalyzer()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
