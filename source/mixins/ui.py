"""UI界面设置Mixin模块 - MatPreML"""

import os
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QComboBox, QFileDialog, QLabel,
                             QMessageBox, QDialog, QFormLayout, QSpinBox,
                             QCheckBox, QTextEdit, QSizePolicy)
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QIcon, QAction, QFont
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from ..config import COPYRIGHT_TEXT, BAYESIAN_OPT_AVAILABLE, XGBOOST_AVAILABLE, CONFIG_DIR


class UIMixin:
    """UI界面设置Mixin类 - 包含initUI、菜单、面板等UI相关方法"""

    def initUI(self):
        """初始化用户界面"""
        if not self.is_registered:
            self.setWindowTitle(
                'MatPreML  (Material Prediction via Machine Learning)  '
                '利用机器学习预测材料及性质  (标准版)'
            )
        else:
            self.setWindowTitle(
                'MatPreML  (Material Prediction via Machine Learning)  '
                '利用机器学习预测材料及性质  (专业版)'
            )

        self.setMinimumSize(1100, 700)

        self.icon = QIcon(os.path.join(CONFIG_DIR, 'icon.png'))
        self.setWindowIcon(self.icon)

        self.create_menus()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.create_left_panel()
        self.create_right_panel()

        content_layout.addWidget(self.left_widget)
        content_layout.addWidget(self.right_widget)

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 1)

        main_layout.addLayout(content_layout)

        self.init_plot()

    def center_window(self):
        """将窗口移动到屏幕中央"""
        if not self.isMaximized():
            screen = self.screen().availableGeometry()
            window = self.geometry()
            x = (screen.width() - window.width()) // 2
            y = (screen.height() - window.height()) // 2
            self.move(x, y)

    def resizeEvent(self, event):
        """处理窗口大小调整事件"""
        from PyQt6.QtWidgets import QMainWindow
        QMainWindow.resizeEvent(self, event)
        if hasattr(self, 'canvas'):
            self.canvas.draw()

    def create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        font = menubar.font()
        font.setPointSize(int(font.pointSize() * 1.20))
        menubar.setFont(font)

        menu_font = menubar.font()
        menu_font.setPointSize(int(menu_font.pointSize() * 1.05))

        # 项目菜单
        project_menu = menubar.addMenu('项目')
        project_menu.setFont(menu_font)
        self.new_project_action = project_menu.addAction('新建项目')
        self.new_project_action.setFont(menu_font)
        self.open_project_action = project_menu.addAction('打开项目')
        self.open_project_action.setFont(menu_font)
        self.save_project_action = project_menu.addAction('保存模型')
        self.save_project_action.setFont(menu_font)
        self.load_project_action = project_menu.addAction('导入模型')
        self.load_project_action.setFont(menu_font)

        self.new_project_action.triggered.connect(self.new_project)
        self.open_project_action.triggered.connect(self.open_project)
        self.save_project_action.triggered.connect(self.save_project)
        self.load_project_action.triggered.connect(self.load_project)

        # 数据菜单
        data_menu = menubar.addMenu('数据')
        data_menu.setFont(menu_font)
        self.import_data_action = data_menu.addAction('导入数据')
        self.import_data_action.setFont(menu_font)
        self.import_data_action.triggered.connect(self.load_data)

        self.feature_engineering_action = data_menu.addAction('特征工程')
        self.feature_engineering_action.setFont(menu_font)
        self.feature_engineering_action.triggered.connect(
            self.show_feature_engineering
        )

        # 训练菜单
        train_menu = menubar.addMenu('训练')
        train_menu.setFont(menu_font)
        self.select_model_action = train_menu.addAction('选择模型')
        self.select_model_action.setFont(menu_font)
        self.start_train_action = train_menu.addAction('开始训练')
        self.start_train_action.setFont(menu_font)
        self.stop_train_action = train_menu.addAction('停止训练')
        self.stop_train_action.setFont(menu_font)

        self.select_model_action.triggered.connect(self.show_model_selection)
        self.start_train_action.triggered.connect(self.train_model)
        self.stop_train_action.triggered.connect(self.stop_training)

        # 评估菜单
        eval_menu = menubar.addMenu('评估')
        eval_menu.setFont(menu_font)
        self.train_eval_action = eval_menu.addAction('训练集模型评估')
        self.train_eval_action.setFont(menu_font)
        self.test_eval_action = eval_menu.addAction('测试集模型评估')
        self.test_eval_action.setFont(menu_font)
        self.train_eval_action.triggered.connect(self.evaluate_train_set)
        self.test_eval_action.triggered.connect(self.evaluate_test_set)

        # 预测菜单
        predict_menu = menubar.addMenu('预测')
        predict_menu.setFont(menu_font)
        self.import_predict_data_action = predict_menu.addAction('导入预测数据')
        self.import_predict_data_action.setFont(menu_font)
        self.import_predict_data_action.triggered.connect(self.predict_data)

        self.generate_predict_data_action = predict_menu.addAction(
            '生成预测数据'
        )
        self.generate_predict_data_action.setFont(menu_font)
        self.generate_predict_data_action.triggered.connect(
            self.generate_predict_data
        )

        self.import_model_action = predict_menu.addAction('导入训练模型')
        self.import_model_action.setFont(menu_font)
        self.import_model_action.triggered.connect(self.load_trained_model)

        # 分析菜单
        analysis_menu = menubar.addMenu('分析')
        analysis_menu.setFont(menu_font)
        self.correlation_analysis_action = analysis_menu.addAction(
            '特征相关性分析'
        )
        self.correlation_analysis_action.setFont(menu_font)
        self.correlation_analysis_action.triggered.connect(
            self.correlation_analysis
        )

        self.importance_analysis_action = analysis_menu.addAction(
            '特征重要性分析'
        )
        self.importance_analysis_action.setFont(menu_font)
        self.importance_analysis_action.triggered.connect(
            self.importance_analysis
        )

        # 设置菜单
        settings_menu = menubar.addMenu('设置')
        settings_menu.setFont(menu_font)
        bayesian_opt_action = settings_menu.addAction('超参调优设置')
        bayesian_opt_action.setFont(menu_font)
        bayesian_opt_action.triggered.connect(self.show_bayesian_opt_dialog)

        cpu_parallel_settings_action = settings_menu.addAction('CPU并行设置')
        cpu_parallel_settings_action.setFont(menu_font)
        cpu_parallel_settings_action.triggered.connect(
            self.show_parallel_settings
        )

        deepseek_settings_action = settings_menu.addAction('DeepSeek设置')
        deepseek_settings_action.setFont(menu_font)
        deepseek_settings_action.triggered.connect(self.show_deepseek_settings)

        # 教程菜单
        help_menu = menubar.addMenu('教程')
        help_menu.setFont(menu_font)
        help_action = help_menu.addAction('使用说明')
        help_action.setFont(menu_font)
        tutorial_action = help_menu.addAction('使用教程')
        tutorial_action.setFont(menu_font)
        models_action = help_menu.addAction('模型说明')
        models_action.setFont(menu_font)
        help_action.triggered.connect(self.show_help)
        tutorial_action.triggered.connect(self.show_tutorial)
        models_action.triggered.connect(self.show_model_readme)

        # AI问答菜单
        ai_menu = menubar.addMenu('AI助手')
        ai_menu.setFont(menu_font)
        deepseek_action = ai_menu.addAction('DeepSeek 智能助手')
        deepseek_action.setFont(menu_font)
        deepseek_action.triggered.connect(self.show_deepseek_dialog)

        # 注册菜单
        register_menu = menubar.addMenu('注册')
        register_menu.setFont(menu_font)
        register_action = register_menu.addAction('专业版')
        register_action.setFont(menu_font)
        register_action.triggered.connect(self.show_register_dialog)

        # 关于菜单
        about_menu = menubar.addMenu('关于')
        about_menu.setFont(menu_font)
        copyright_action = about_menu.addAction('版权说明')
        copyright_action.setFont(menu_font)
        copyright_action.triggered.connect(self.show_copyright)

    def create_left_panel(self):
        """创建左侧按钮面板"""
        self.left_widget = QWidget()
        left_layout = QVBoxLayout(self.left_widget)
        self.left_widget.setFixedWidth(165)

        self.load_button = QPushButton('导入学习数据')
        self.load_button.clicked.connect(self.load_data)
        button_font = self.load_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.load_button.setFont(button_font)
        self.load_button.setFixedHeight(
            int(self.load_button.sizeHint().height() * 1.2)
        )

        button_height_half = (
            self.load_button.sizeHint().height() // 2
            if self.load_button.sizeHint().height() > 0
            else 12
        )

        left_layout.addSpacing(self.load_button.sizeHint().height())

        left_layout.addWidget(self.load_button)

        self.train_result_button = QPushButton('训练结果')
        self.train_result_button.clicked.connect(self.show_train_result)
        self.train_result_button.setEnabled(False)
        button_font = self.train_result_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.train_result_button.setFont(button_font)
        self.train_result_button.setFixedHeight(
            int(self.train_result_button.sizeHint().height() * 1.2)
        )

        # 画图选择变量下拉菜单
        self.variable_label = QLabel('选择特征值(X变量)画图:')
        label_font = self.variable_label.font()
        label_font.setPointSize(int(label_font.pointSize() * 1.2))
        self.variable_label.setFont(label_font)
        self.variable_label.setFixedHeight(22)
        self.variable_combo = QComboBox()
        self.variable_combo.addItems(['None'])
        combo_font = self.variable_combo.font()
        combo_font.setPointSize(int(combo_font.pointSize() * 1.2))
        self.variable_combo.setFont(combo_font)
        self.variable_combo.setFixedHeight(
            int(self.variable_combo.sizeHint().height() * 1.2)
        )
        self.variable_combo.currentIndexChanged.connect(self.on_variable_change)
        left_layout.addSpacing(button_height_half)
        left_layout.addWidget(self.variable_label)
        left_layout.addWidget(self.variable_combo)
        left_layout.addSpacing(button_height_half)

        # 模型选择下拉菜单
        self.model_label = QLabel('选择模型:')
        self.model_label.setFixedHeight(20)
        label_font = self.model_label.font()
        label_font.setPointSize(int(label_font.pointSize() * 1.2))
        self.model_label.setFont(label_font)
        self.model_combo = QComboBox()
        models = ['GradientBoosting', 'BayesianRidge', 'BayesianBootstrap',
                   'SVR', 'RandomForest', 'Lasso', 'DecisionTree', 'NeuralNetwork']
        if XGBOOST_AVAILABLE:
            models.insert(0, 'XGBoost')
        self.model_combo.addItems(models)
        combo_font = self.model_combo.font()
        combo_font.setPointSize(int(combo_font.pointSize() * 1.2))
        self.model_combo.setFont(combo_font)
        self.model_combo.setFixedHeight(
            int(self.model_combo.sizeHint().height() * 1.2)
        )
        left_layout.addWidget(self.model_label)
        left_layout.addWidget(self.model_combo)
        left_layout.addSpacing(button_height_half)

        # 模型选择改变时启用超参调优复选框
        def on_model_change(index):
            model_name = self.model_combo.currentText()
            if model_name != '选择模型':
                self.bayesian_opt_checkbox.setEnabled(True)
            else:
                self.bayesian_opt_checkbox.setEnabled(False)

        self.model_combo.currentIndexChanged.connect(on_model_change)

        # 超参调优复选框
        self.bayesian_opt_checkbox = QCheckBox('超参调优 (高精度)')
        checkbox_font = self.bayesian_opt_checkbox.font()
        checkbox_font.setPointSize(int(checkbox_font.pointSize() * 1.2))
        self.bayesian_opt_checkbox.setFont(checkbox_font)
        self.bayesian_opt_checkbox.setFixedHeight(
            int(self.bayesian_opt_checkbox.sizeHint().height() * 1.2)
        )
        self.bayesian_opt_checkbox.setEnabled(True)
        self.bayesian_opt_checkbox.stateChanged.connect(
            self.on_bayesian_opt_toggle
        )
        left_layout.addWidget(self.bayesian_opt_checkbox)
        left_layout.addSpacing(button_height_half)

        # 训练 (单个模型) 按钮
        self.train_button = QPushButton('训练 (单个模型)')
        self.train_button.clicked.connect(self.train_model)
        button_font = self.train_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.train_button.setFont(button_font)
        self.train_button.setFixedHeight(
            int(self.train_button.sizeHint().height() * 1.2)
        )
        left_layout.addWidget(self.train_button)
        left_layout.addSpacing(button_height_half)

        # 训练 (所有模型) 按钮
        self.train_all_button = QPushButton('训练 (所有模型)')
        self.train_all_button.clicked.connect(self.train_all_models)
        button_font = self.train_all_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.train_all_button.setFont(button_font)
        self.train_all_button.setFixedHeight(
            int(self.train_all_button.sizeHint().height() * 1.2)
        )
        left_layout.addWidget(self.train_all_button)
        left_layout.addSpacing(button_height_half)

        # 预测按钮
        self.predict_button = QPushButton('预测')
        self.predict_button.clicked.connect(self.predict_data)
        button_font = self.predict_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.predict_button.setFont(button_font)
        self.predict_button.setFixedHeight(
            int(self.predict_button.sizeHint().height() * 1.2)
        )
        left_layout.addWidget(self.predict_button)
        left_layout.addSpacing(button_height_half)

        # 特征相关性按钮
        self.correlation_button = QPushButton('特征相关性')
        self.correlation_button.clicked.connect(self.correlation_analysis)
        button_font = self.correlation_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.correlation_button.setFont(button_font)
        self.correlation_button.setFixedHeight(
            int(self.correlation_button.sizeHint().height() * 1.2)
        )
        left_layout.addWidget(self.correlation_button)
        left_layout.addSpacing(button_height_half)

        # 特征重要性按钮
        self.importance_button = QPushButton('特征重要性')
        self.importance_button.clicked.connect(self.importance_analysis)
        button_font = self.importance_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.importance_button.setFont(button_font)
        self.importance_button.setFixedHeight(
            int(self.importance_button.sizeHint().height() * 1.2)
        )
        left_layout.addWidget(self.importance_button)
        left_layout.addSpacing(button_height_half)

        # 当前(最佳)模型显示区域
        self.current_model_label = QLabel('当前(最佳)模型:')
        label_font = self.current_model_label.font()
        label_font.setPointSize(int(label_font.pointSize() * 1.2))
        self.current_model_label.setFont(label_font)
        self.current_model_label.setFixedHeight(24)
        left_layout.addWidget(self.current_model_label)

        self.current_model_text = QTextEdit()
        self.current_model_text.setReadOnly(True)
        self.current_model_text.setStyleSheet(
            "background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;"
        )
        self.current_model_text.setFixedHeight(40)
        text_font = self.current_model_text.font()
        text_font.setPointSize(int(text_font.pointSize() * 1.2))
        self.current_model_text.setFont(text_font)
        self.current_model_text.setLineWrapMode(
            QTextEdit.LineWrapMode.WidgetWidth
        )
        self.current_model_text.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        left_layout.addWidget(self.current_model_text)
        left_layout.addSpacing(button_height_half)

        # 模型评估指标显示区域
        self.metrics_label = QLabel('模型评估指标:')
        label_font = self.metrics_label.font()
        label_font.setPointSize(int(label_font.pointSize() * 1.2))
        self.metrics_label.setFont(label_font)
        self.metrics_label.setFixedHeight(24)
        left_layout.addWidget(self.metrics_label)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet(
            "background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;"
        )
        self.metrics_text.setFixedHeight(55)
        text_font = self.metrics_text.font()
        text_font.setPointSize(int(text_font.pointSize()))
        self.metrics_text.setFont(text_font)
        self.metrics_text.setAcceptRichText(True)
        self.metrics_text.setLineWrapMode(
            QTextEdit.LineWrapMode.WidgetWidth
        )
        self.metrics_text.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        left_layout.addWidget(self.metrics_text)
        left_layout.addSpacing(button_height_half)

        # 状态显示标签
        self.status_label_name = QLabel('当前状态：')
        label_font = self.status_label_name.font()
        label_font.setPointSize(int(label_font.pointSize() * 1.2))
        self.status_label_name.setFont(label_font)
        self.status_label_name.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        left_layout.addWidget(self.status_label_name)

        self.status_label = QLabel('')
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
        )
        self.status_label.setStyleSheet(
            "background-color: #e0e0e0; padding: 2px; font-size: 12px; "
            "font-weight: bold; color: red;"
        )
        self.status_label.setFixedHeight(36)
        left_layout.addWidget(self.status_label)

        # 停止训练按钮
        self.stop_train_button = QPushButton('停止训练')
        button_font = self.stop_train_button.font()
        button_font.setPointSize(int(button_font.pointSize() * 1.2))
        self.stop_train_button.setFont(button_font)
        self.stop_train_button.setFixedHeight(
            int(self.stop_train_button.sizeHint().height() * 1.2)
        )
        self.stop_train_button.clicked.connect(self.stop_training)
        self.stop_train_button.setEnabled(False)
        left_layout.addSpacing(2 * button_height_half)
        left_layout.addWidget(self.stop_train_button)

        left_layout.addStretch()

    def create_right_panel(self):
        """创建右侧绘图面板"""
        self.right_widget = QWidget()
        right_layout = QVBoxLayout(self.right_widget)
        right_layout.setContentsMargins(5, 0, 15, 10)
        right_layout.setSpacing(0)

        self.figure = plt.figure(tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setContentsMargins(0, 0, 0, 0)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self.right_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )

    def init_plot(self):
        """初始化绘图区域 - 显示demo图片或默认演示图表"""
        demo_image_path = os.path.join(CONFIG_DIR, 'demo.png')
        if os.path.exists(demo_image_path):
            try:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                img = plt.imread(demo_image_path)
                ax.set_xlim(0, 1)
                ax.set_ylim(0.2, 0.8)
                ax.axis('off')

                img_height, img_width = img.shape[:2]
                display_size = 0.5

                if img_width > img_height:
                    width = display_size
                    height = display_size * img_height / img_width
                else:
                    height = display_size
                    width = display_size * img_width / img_height

                x = (1 - width) / 2
                y = (1 - height) / 2

                ax.imshow(
                    img, extent=[x, x + width, y, y + height], aspect='auto'
                )
                self.figure.subplots_adjust(
                    left=0.1, right=0.9, top=0.9, bottom=0.1
                )
                self.canvas.draw()
                return
            except Exception as e:
                print(f"加载demo.png时出错: {str(e)}")

        # 默认演示图表
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        np.random.seed(42)
        x_data = np.linspace(0, 3 * np.pi, 50)
        y_true = (
            np.sin(1.5 * x_data)
            + 0.4 * np.sin(2 * x_data)
            + 0.5 * np.random.normal(0, 0.3, len(x_data))
        )
        y_pred = np.sin(1.5 * x_data) + 0.4 * np.sin(2 * x_data)

        ax.scatter(
            x_data, y_true, c='blue', marker='o', label='真实值',
            alpha=0.8, s=40, edgecolors='navy', linewidth=0.5
        )
        ax.scatter(
            x_data, y_pred, c='red', marker='s', label='预测值',
            alpha=0.8, s=40, edgecolors='darkred', linewidth=0.5
        )
        ax.plot(x_data, y_true, c='blue', linewidth=1, alpha=0.4,
                linestyle='--')
        ax.plot(x_data, y_pred, c='red', linewidth=2, alpha=0.6)

        button_font_size = 9
        ax.set_title(
            '机器学习预测结果（演示）',
            fontproperties=fm.FontProperties(family='SimHei'),
            fontsize=button_font_size, fontweight='bold'
        )
        ax.set_xlabel(
            '特征值',
            fontproperties=fm.FontProperties(family='SimHei'),
            fontsize=button_font_size
        )
        ax.set_ylabel(
            '目标值',
            fontproperties=fm.FontProperties(family='SimHei'),
            fontsize=button_font_size
        )
        ax.legend(loc='upper right', fontsize=button_font_size - 1)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        self.canvas.draw()

    def show_copyright(self):
        """显示版权说明"""
        QMessageBox.about(self, '版权说明', COPYRIGHT_TEXT)

    def show_parallel_settings(self):
        """显示并行设置对话框"""
        import multiprocessing
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, \
            QPushButton, QLabel, QSpinBox

        max_cores = multiprocessing.cpu_count()

        settings = QSettings("MPP", "ML")
        current_cores = settings.value("cpu_cores", max_cores, type=int)

        dialog = QDialog(self)
        dialog.setWindowTitle('CPU并行设置')
        dialog.setModal(True)
        dialog.resize(300, 150)

        layout = QVBoxLayout(dialog)

        label = QLabel(f'设置CPU核心使用数目 (最大: {max_cores})')
        layout.addWidget(label)

        self.cpu_cores_spinbox = QSpinBox()
        self.cpu_cores_spinbox.setRange(1, max_cores)
        self.cpu_cores_spinbox.setValue(current_cores)
        layout.addWidget(self.cpu_cores_spinbox)

        button_layout = QHBoxLayout()
        ok_button = QPushButton('确定')
        cancel_button = QPushButton('取消')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_cores = self.cpu_cores_spinbox.value()
            settings.setValue("cpu_cores", new_cores)
            QMessageBox.information(
                self, '成功', f'CPU核心数已设置为: {new_cores}'
            )

    def get_cpu_cores(self):
        """获取配置的CPU核心数"""
        settings = QSettings("MPP", "ML")
        return settings.value("cpu_cores", 4, type=int)

    def on_bayesian_opt_toggle(self, state):
        """超参调优复选框状态改变处理"""
        if not self.is_registered and state == Qt.CheckState.Checked.value:
            QMessageBox.warning(
                self, '警告', '只有注册专业版的用户才能够使用此功能。'
            )
            self.bayesian_opt_checkbox.setChecked(False)
            return
        self.bayesian_opt_enabled = (state == Qt.CheckState.Checked.value)

    def show_bayesian_opt_dialog(self):
        """显示贝叶斯超参优化设置对话框"""
        if not self.is_registered:
            QMessageBox.warning(
                self, '警告', '只有注册专业版的用户才能够使用此功能。'
            )
            return

        if not BAYESIAN_OPT_AVAILABLE:
            QMessageBox.warning(
                self, '警告',
                '未安装scikit-optimize库，无法使用贝叶斯优化功能'
            )
            self.bayesian_opt_checkbox.setChecked(False)
            self.bayesian_opt_enabled = False
            return

        dialog = QDialog(self)
        dialog.setWindowTitle('贝叶斯超参优化设置')
        dialog.setModal(True)
        dialog.resize(300, 200)

        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        self.n_iter_spinbox = QSpinBox()
        self.n_iter_spinbox.setRange(30, 300)
        self.n_iter_spinbox.setValue(100)
        form_layout.addRow('优化迭代次数:', self.n_iter_spinbox)

        self.cv_folds_spinbox = QSpinBox()
        self.cv_folds_spinbox.setRange(2, 10)
        self.cv_folds_spinbox.setValue(5)
        form_layout.addRow('交叉验证折数:', self.cv_folds_spinbox)

        self.random_state_spinbox = QSpinBox()
        self.random_state_spinbox.setRange(0, 1000)
        self.random_state_spinbox.setValue(42)
        form_layout.addRow('随机状态:', self.random_state_spinbox)

        layout.addLayout(form_layout)

        button_layout = QHBoxLayout()
        ok_button = QPushButton('确定')
        cancel_button = QPushButton('取消')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.bayesian_params = {
                'n_iter': self.n_iter_spinbox.value(),
                'cv': self.cv_folds_spinbox.value(),
                'random_state': self.random_state_spinbox.value()
            }
        else:
            self.bayesian_opt_checkbox.setChecked(False)
            self.bayesian_opt_enabled = False
