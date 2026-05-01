"""配置和常量模块 - MatPreML"""

import os
import sys
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

# 元素周期表数据（原子序数：元素符号）
PERIODIC_TABLE = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# 原子序数到元素符号的反向映射
ATOMIC_NUMBERS = {v: k for k, v in PERIODIC_TABLE.items()}

# 可选依赖可用性标志
try:
    from skopt import BayesSearchCV  # noqa: F401
    from skopt.space import Real, Integer, Categorical  # noqa: F401
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("Warning: scikit-optimize not installed. Bayesian optimization will be limited.")

try:
    import xgboost as xgb  # noqa: F401
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. Some functionality will be limited.")

try:
    from bayesian_bootstrap import BayesianBootstrapBagging  # noqa: F401
    BAYESIAN_BOOTSTRAP_AVAILABLE = True
except ImportError:
    BAYESIAN_BOOTSTRAP_AVAILABLE = False
    print("Warning: bayesian_bootstrap not installed. Some functionality will be limited.")

# 所有可用模型列表
ALL_MODELS = ['BayesianRidge', 'SVR', 'RandomForest', 'GradientBoosting',
              'Lasso', 'DecisionTree', 'NeuralNetwork']

COPYRIGHT_TEXT = ("本程序由中科院固体物理所功能材料研究部开发。\n"
                  "本软件可在学术研究中自由使用，未经许可不可用于商业用途。\n"
                  " \n作者：鲁文建     Email: wjlu@issp.ac.cn \n版本号：V2.4")

DEFAULT_DEEPSEEK_CONFIG = {
    "api_url": "https://api.deepseek.com/v1/chat/completions",
    "api_key": ""
}

# 配置目录基于文件位置计算（相对路径：从包目录上一级到 configure/）
_CONFIG_BASE = os.path.dirname(os.path.abspath(__file__))  # MatPreML_refactored/
CONFIG_DIR = os.path.abspath(os.path.join(_CONFIG_BASE, '..', 'configure'))
DEEPSEEK_CONFIG_FILE = os.path.join(CONFIG_DIR, 'deepseek_config.json')
REGISTRATION_FILE = os.path.join(CONFIG_DIR, 'registration.json')
