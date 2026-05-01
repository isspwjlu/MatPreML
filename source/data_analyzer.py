"""主程序入口模块 - MatPreML

DataAnalyzer类继承QMainWindow和所有功能Mixin，
组装完整的应用程序。
"""

import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QMainWindow, QApplication
from PyQt6.QtCore import Qt

from .config import BAYESIAN_OPT_AVAILABLE, BAYESIAN_BOOTSTRAP_AVAILABLE
from .threads import TrainingThread
from .mixins.ui import UIMixin
from .mixins.data import DataMixin
from .mixins.project import ProjectMixin
from .mixins.analysis import AnalysisMixin
from .mixins.training import TrainingMixin
from .mixins.deepseek import DeepSeekMixin
from .mixins.feature_engineering import FeatureEngineeringMixin
from .mixins.registration import RegistrationMixin


class DataAnalyzer(
    QMainWindow, UIMixin, DataMixin, ProjectMixin, AnalysisMixin,
    TrainingMixin, DeepSeekMixin, FeatureEngineeringMixin,
    RegistrationMixin
):
    """主应用程序类 - 通过多重继承组合所有功能模块"""

    def __init__(self):
        super().__init__()

        # ==== 初始化共享属性 ====
        # 项目相关
        self.project_loaded = False
        self.project_path = None

        # 数据相关
        self.data = None
        self.prediction_data = None
        self.data_file_path = None
        self.training_elements = []

        # 模型相关
        self.model = None
        self.model_name = None
        self.model_params = None
        self.model_results = {}
        self.ranking_text = ""
        self.ranking_filename = ""

        # 训练相关
        self.training_all_models = False
        self.stop_training_flag = False
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.bayesian_opt_enabled = False
        self.bayesian_params = {
            'n_iter': 100,
            'cv': 5,
            'random_state': 42
        }

        # 特征工程线程引用
        self.feature_engineering_thread = None

        # 初始化注册状态
        self.check_registration()

        # 初始化UI
        self.initUI()

        # 禁用所有功能
        self.disable_all_features()

        # 初始化训练线程
        self.training_thread = TrainingThread(self)
        self.training_thread.training_started.connect(
            self.on_training_started
        )
        self.training_thread.training_finished.connect(
            self.on_training_finished
        )
        self.training_thread.training_error.connect(
            self.on_training_error
        )

        # 居中显示窗口
        self.center_window()
        self.show()
