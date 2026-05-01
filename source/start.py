"""启动MatPreML"""
import sys, os
# 添加上级目录到系统路径，使得 MatPreML_refactored 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MatPreML_refactored.main import main
main()
