"""项目管理Mixin模块 - MatPreML"""

import os
import pickle
import pandas as pd
from PyQt6.QtWidgets import (
    QFileDialog, QMessageBox, QDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QGridLayout, QTextEdit
)
from PyQt6.QtCore import Qt

from ..config import CONFIG_DIR


class ProjectMixin:
    """项目管理Mixin类 - 项目创建、打开、保存、帮助等"""

    def new_project(self):
        """新建项目"""
        class NewProjectDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle('新建项目')
                self.setModal(True)
                self.resize(500, 200)
                self.setFixedSize(500, 200)

                layout = QVBoxLayout()
                layout.setSpacing(15)

                title_label = QLabel('创建新项目')
                title_font = title_label.font()
                title_font.setPointSize(int(title_font.pointSize() * 1.2))
                title_label.setFont(title_font)
                title_label.setStyleSheet(
                    "font-weight: bold; padding: 10px;"
                )
                layout.addWidget(title_label)

                grid_layout = QGridLayout()
                grid_layout.setSpacing(10)
                grid_layout.setColumnStretch(1, 1)

                name_label = QLabel('项目名称:')
                label_font = name_label.font()
                label_font.setPointSize(int(label_font.pointSize() * 1.2))
                name_label.setFont(label_font)
                name_label.setAlignment(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )
                self.name_edit = QLineEdit()
                edit_font = self.name_edit.font()
                edit_font.setPointSize(int(edit_font.pointSize() * 1.2))
                self.name_edit.setFont(edit_font)
                self.name_edit.setFixedHeight(
                    int(self.name_edit.sizeHint().height() * 1.2)
                )
                self.name_edit.setPlaceholderText("输入项目名称")
                grid_layout.addWidget(name_label, 0, 0)
                grid_layout.addWidget(self.name_edit, 0, 1)

                folder_label = QLabel('项目位置:')
                label_font = folder_label.font()
                label_font.setPointSize(int(label_font.pointSize() * 1.2))
                folder_label.setFont(label_font)
                folder_label.setAlignment(
                    Qt.AlignmentFlag.AlignRight
                    | Qt.AlignmentFlag.AlignVCenter
                )
                folder_layout = QHBoxLayout()
                self.folder_edit = QLineEdit()
                edit_font = self.folder_edit.font()
                edit_font.setPointSize(int(edit_font.pointSize() * 1.2))
                self.folder_edit.setFont(edit_font)
                self.folder_edit.setFixedHeight(
                    int(self.folder_edit.sizeHint().height() * 1.2)
                )
                self.folder_edit.setPlaceholderText("选择项目保存位置")
                folder_button = QPushButton('浏览...')
                button_font = folder_button.font()
                button_font.setPointSize(
                    int(button_font.pointSize() * 1.2)
                )
                folder_button.setFont(button_font)
                folder_button.setFixedHeight(
                    int(folder_button.sizeHint().height() * 1.2)
                )
                folder_button.setFixedWidth(80)
                folder_layout.addWidget(self.folder_edit)
                folder_layout.addWidget(folder_button)
                grid_layout.addWidget(folder_label, 1, 0)
                grid_layout.addLayout(folder_layout, 1, 1)

                layout.addLayout(grid_layout)

                line = QLabel()
                line.setFrameShape(QLabel.Shape.HLine)
                line.setFrameShadow(QLabel.Shadow.Sunken)
                layout.addWidget(line)

                button_layout = QHBoxLayout()
                button_layout.addStretch()
                self.ok_button = QPushButton('创建')
                button_font = self.ok_button.font()
                button_font.setPointSize(
                    int(button_font.pointSize() * 1.2)
                )
                self.ok_button.setFont(button_font)
                self.ok_button.setFixedHeight(
                    int(self.ok_button.sizeHint().height() * 1.2)
                )
                self.ok_button.setDefault(True)
                self.ok_button.setFixedWidth(80)
                self.cancel_button = QPushButton('取消')
                button_font = self.cancel_button.font()
                button_font.setPointSize(
                    int(button_font.pointSize() * 1.2)
                )
                self.cancel_button.setFont(button_font)
                self.cancel_button.setFixedHeight(
                    int(self.cancel_button.sizeHint().height() * 1.2)
                )
                self.cancel_button.setFixedWidth(80)
                button_layout.addWidget(self.ok_button)
                button_layout.addWidget(self.cancel_button)
                layout.addLayout(button_layout)

                self.setLayout(layout)

                folder_button.clicked.connect(self.browse_folder)
                self.ok_button.clicked.connect(self.accept)
                self.cancel_button.clicked.connect(self.reject)
                self.name_edit.textChanged.connect(self.on_input_changed)
                self.folder_edit.textChanged.connect(self.on_input_changed)

                self.ok_button.setEnabled(False)
                self.name_edit.setFocus()

            def browse_folder(self):
                folder = QFileDialog.getExistingDirectory(
                    self, "选择项目位置"
                )
                if folder:
                    self.folder_edit.setText(folder)

            def on_input_changed(self, text):
                self.update_ok_button()

            def update_ok_button(self):
                enabled = bool(
                    self.name_edit.text().strip()
                    and self.folder_edit.text().strip()
                )
                self.ok_button.setEnabled(enabled)

            def get_project_info(self):
                return (
                    self.name_edit.text().strip(),
                    self.folder_edit.text().strip()
                )

        dialog = NewProjectDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        project_name, project_dir = dialog.get_project_info()
        if not project_name or not project_dir:
            return

        import re
        project_name = re.sub(r'[<>:"/\\|?*]', '_', project_name)
        project_file = os.path.join(
            project_dir, f"{project_name}.proj"
        )

        try:
            with open(project_file, 'w', encoding='utf-8') as f:
                f.write(f"Project Name: {project_name}\n")
                f.write(f"Project Path: {project_dir}\n")
                f.write("Created: " + str(pd.Timestamp.now()) + "\n")

            self.project_path = project_dir
            self.clear_data()
            self.enable_all_features()

            QMessageBox.information(
                self, '项目创建成功',
                f'项目已创建: {project_file}'
            )
        except Exception as e:
            QMessageBox.critical(
                self, '创建失败',
                f'创建项目失败: {str(e)}'
            )

    def open_project(self):
        """打开项目"""
        project_file, _ = QFileDialog.getOpenFileName(
            self, '打开项目', '', 'Project files (*.proj)'
        )

        if not project_file:
            return

        try:
            project_dir = os.path.dirname(project_file)
            self.project_path = project_dir
            self.project_loaded = True
            self.enable_all_features()

            project_name = os.path.basename(project_file).replace(
                '.proj', ''
            )
            QMessageBox.information(
                self, '项目打开成功',
                f'项目 "{project_name}" 已打开\n路径: {project_file}'
            )
        except Exception as e:
            QMessageBox.critical(
                self, '打开失败',
                f'打开项目失败: {str(e)}'
            )

    def enable_all_features(self):
        """启用所有功能"""
        self.project_loaded = True

        self.save_project_action.setEnabled(True)
        self.load_project_action.setEnabled(True)
        self.import_data_action.setEnabled(True)
        self.select_model_action.setEnabled(True)
        self.start_train_action.setEnabled(True)
        self.stop_train_action.setEnabled(True)
        self.train_eval_action.setEnabled(True)
        self.test_eval_action.setEnabled(True)
        self.import_predict_data_action.setEnabled(True)
        self.generate_predict_data_action.setEnabled(True)
        self.import_model_action.setEnabled(True)
        self.correlation_analysis_action.setEnabled(True)
        self.importance_analysis_action.setEnabled(True)

        self.load_button.setEnabled(True)
        self.train_result_button.setEnabled(False)
        self.model_combo.setEnabled(True)
        self.bayesian_opt_checkbox.setEnabled(True)
        self.train_button.setEnabled(True)
        self.train_all_button.setEnabled(True)
        self.predict_button.setEnabled(True)
        self.correlation_button.setEnabled(True)
        self.importance_button.setEnabled(True)
        self.variable_combo.setEnabled(True)

    def check_project_status(self):
        """检查项目状态"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return False
        return True

    def disable_all_features(self):
        """设置项目未加载状态"""
        self.project_loaded = False

    def save_project(self):
        """保存项目"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None:
            QMessageBox.warning(self, '警告', '没有训练好的模型可以保存')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, '保存项目', '', 'Pickle files (*.pkl)'
        )

        if file_path:
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'model_name': self.model_name,
                        'X_columns': (
                            self.data.columns[:-1]
                            if self.data is not None else None
                        )
                    }, f)
                QMessageBox.information(self, '成功', '项目保存成功')
            except Exception as e:
                QMessageBox.critical(
                    self, '错误',
                    f'保存项目时出错: {str(e)}'
                )

    def load_project(self):
        """导入项目"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, '导入项目', '', 'Pickle files (*.pkl)'
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    project_data = pickle.load(f)

                self.model = project_data['model']
                self.model_name = project_data['model_name']

                QMessageBox.information(self, '成功', '项目导入成功')
            except Exception as e:
                QMessageBox.critical(
                    self, '错误',
                    f'导入项目时出错: {str(e)}'
                )

    def load_trained_model(self):
        """导入训练好的模型"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, '导入训练模型', '', 'Pickle files (*.pkl)'
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.model_name = model_data['model_name']

                if self.model_name:
                    index = self.model_combo.findText(self.model_name)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)

                QMessageBox.information(
                    self, '成功',
                    f'训练模型导入成功: {self.model_name}'
                )
            except Exception as e:
                QMessageBox.critical(
                    self, '错误',
                    f'导入训练模型时出错: {str(e)}'
                )

    def show_help(self):
        """显示使用说明"""
        try:
            with open(os.path.join(CONFIG_DIR, 'readme.txt'), 'r', encoding='utf-8') as f:
                help_text = f.read()
        except FileNotFoundError:
            help_text = (
                "使用说明文件未找到。\n\n基本操作：\n"
                "1. 通过菜单栏或左侧按钮导入数据\n"
                "2. 选择合适的机器学习模型\n"
                "3. 点击训练按钮训练模型\n"
                "4. 导入预测数据进行预测"
            )
        except Exception as e:
            help_text = f"读取使用说明时出错: {str(e)}"

        dialog = QDialog(self)
        dialog.setWindowTitle('使用说明')
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(help_text)
        layout.addWidget(text_edit)

        button_layout = QHBoxLayout()
        close_button = QPushButton('关闭')
        close_button.clicked.connect(dialog.close)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def show_tutorial(self):
        """显示使用教程"""
        try:
            with open(
                os.path.join(CONFIG_DIR, 'tutorial.txt'), 'r', encoding='utf-8'
            ) as f:
                help_text = f.read()
        except FileNotFoundError:
            help_text = (
                "使用教程文件未找到。\n\n基本操作：\n"
                "1. 通过菜单栏或左侧按钮导入数据\n"
                "2. 选择合适的机器学习模型\n"
                "3. 点击训练按钮训练模型\n"
                "4. 导入预测数据进行预测"
            )
        except Exception as e:
            help_text = f"读取使用教程时出错: {str(e)}"

        dialog = QDialog(self)
        dialog.setWindowTitle('使用教程')
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(help_text)
        layout.addWidget(text_edit)

        button_layout = QHBoxLayout()
        close_button = QPushButton('关闭')
        close_button.clicked.connect(dialog.close)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        dialog.exec()

    def show_model_readme(self):
        """显示模型说明"""
        try:
            with open(
                os.path.join(CONFIG_DIR, 'models.txt'), 'r', encoding='utf-8'
            ) as f:
                help_text = f.read()
        except FileNotFoundError:
            help_text = (
                "模型说明文件未找到。\n\n算法选择建议：\n"
                "对线性关系：优先使用线性回归或岭回归\n"
                "对特征选择：考虑Lasso回归或弹性网络\n"
                "对非线性关系：尝试SVR、决策树或神经网络\n"
                "对大数据集：随机森林和神经网络表现更佳"
            )
        except Exception as e:
            help_text = f"读取模型说明时出错: {str(e)}"

        dialog = QDialog(self)
        dialog.setWindowTitle('模型说明')
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(help_text)
        layout.addWidget(text_edit)

        button_layout = QHBoxLayout()
        close_button = QPushButton('关闭')
        close_button.clicked.connect(dialog.close)
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

        dialog.exec()
