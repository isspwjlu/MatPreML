"""特征工程Mixin模块 - MatPreML"""

import os
import pandas as pd
from PyQt6.QtWidgets import (
    QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QScrollArea,
    QWidget, QRadioButton, QProgressDialog, QButtonGroup
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..chemical import ChemicalFormulaProcessor
from ..threads import FeatureEngineeringThread


class FeatureEngineeringMixin:
    """特征工程Mixin类 - Matminer特征提取、进度管理"""

    def show_feature_engineering(self):
        """显示特征工程对话框"""
        if not self.is_registered:
            QMessageBox.warning(
                self, '警告',
                '只有注册专业版的用户才能够使用此功能。'
            )
            return

        if self.data is None or self.data.empty:
            QMessageBox.warning(self, '警告', '请先导入数据')
            return

        try:
            from matminer.featurizers.base import MultipleFeaturizer
            from matminer.featurizers import composition as cf
            from matminer.featurizers.conversions import (
                StrToComposition
            )
            from matminer.utils.data import PymatgenData

            available_featurizers = [
                (
                    "ElementProperty(magpie)",
                    "ElementPropertyMagpie",
                    "基于magpie预设的元素属性特征"
                ),
                (
                    "ElementProperty(deml)",
                    "ElementPropertyDeml",
                    "基于deml预设的元素属性特征"
                ),
                (
                    "OxidationStates",
                    "OxidationStates",
                    "氧化态相关特征"
                ),
                (
                    "AtomicOrbitals",
                    "AtomicOrbitals",
                    "原子轨道特征"
                ),
                (
                    "ElectronegativityDiff",
                    "ElectronegativityDiff",
                    "电负性差异特征"
                ),
                (
                    "Stoichiometry",
                    "Stoichiometry",
                    "化学计量特征"
                ),
                (
                    "ValenceOrbital",
                    "ValenceOrbital",
                    "价轨道特征"
                )
            ]

            dialog = QDialog(self)
            dialog.setWindowTitle('Matminer-特征工程')
            dialog.setModal(True)
            dialog.resize(700, 500)

            layout = QVBoxLayout(dialog)

            info_label = QLabel(
                '选择要应用的特征工程选项:  '
                '(注意：特征工程要求导入数据的第一列必须为化学式。'
                '最下方选择处理的是学习数据还是预测数据。)'
            )
            info_label.setStyleSheet("font-weight: bold; padding: 7px;")
            layout.addWidget(info_label)

            scroll_area = QScrollArea()
            scroll_widget = QWidget()
            scroll_layout = QVBoxLayout(scroll_widget)

            checkboxes = []
            for name, key, description in available_featurizers:
                checkbox = QCheckBox(
                    f"{name} - {description}"
                )
                checkbox.setProperty("featurizer_key", key)
                checkbox.setFont(QFont("Microsoft YaHei", 10))
                scroll_layout.addWidget(checkbox)
                checkboxes.append(checkbox)

            scroll_widget.setLayout(scroll_layout)
            scroll_area.setWidget(scroll_widget)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area)

            button_layout = QHBoxLayout()
            select_all_button = QPushButton('全选')
            deselect_all_button = QPushButton('全不选')
            ok_button = QPushButton('生成特征')
            cancel_button = QPushButton('取消')
            select_all_button.setFixedHeight(40)
            deselect_all_button.setFixedHeight(40)
            ok_button.setFixedHeight(40)
            ok_button.setFixedWidth(100)
            cancel_button.setFixedHeight(40)

            def select_all():
                for cb in checkboxes:
                    cb.setChecked(True)

            def deselect_all():
                for cb in checkboxes:
                    cb.setChecked(False)

            select_all_button.clicked.connect(select_all)
            deselect_all_button.clicked.connect(deselect_all)

            learning_radio = QRadioButton('学习数据   ')
            prediction_radio = QRadioButton('预测数据   ')
            learning_radio.setFixedHeight(40)
            prediction_radio.setFixedHeight(40)
            learning_radio.setChecked(True)

            radio_group = QButtonGroup(dialog)
            radio_group.addButton(learning_radio)
            radio_group.addButton(prediction_radio)

            def apply_feature_engineering():
                selected_featurizers = []
                for cb in checkboxes:
                    if cb.isChecked():
                        featurizer_key = cb.property("featurizer_key")
                        selected_featurizers.append(featurizer_key)

                if not selected_featurizers:
                    QMessageBox.warning(
                        self, '警告',
                        '请至少选择一个特征工程选项'
                    )
                    return

                is_learning_data = learning_radio.isChecked()
                self.perform_feature_engineering(
                    selected_featurizers, is_learning_data
                )
                dialog.accept()

            ok_button.clicked.connect(apply_feature_engineering)
            cancel_button.clicked.connect(dialog.reject)

            button_layout.addWidget(select_all_button)
            button_layout.addWidget(deselect_all_button)
            button_layout.addSpacing(20)
            button_layout.addWidget(learning_radio)
            button_layout.addWidget(prediction_radio)
            button_layout.addStretch()
            button_layout.addWidget(ok_button)
            button_layout.addWidget(cancel_button)
            layout.addLayout(button_layout)

            dialog.exec()

        except ImportError as e:
            QMessageBox.critical(
                self, '错误',
                f'未安装Matminer库或相关依赖: {str(e)}\n'
                f'请使用pip install matminer安装'
            )
        except Exception as e:
            QMessageBox.critical(
                self, '错误',
                f'显示特征工程选项时出错: {str(e)}'
            )

    def perform_feature_engineering(
        self, selected_featurizers, is_learning_data=True
    ):
        """执行特征工程"""
        self.feature_engineering_thread = FeatureEngineeringThread(
            self, selected_featurizers, is_learning_data
        )
        self.feature_engineering_thread.finished.connect(
            self.on_feature_engineering_finished
        )
        self.feature_engineering_thread.error.connect(
            self.on_feature_engineering_error
        )

        self.progress_dialog = QProgressDialog(
            "特征工程正在执行中...", "取消", 0, 100, self
        )
        self.progress_dialog.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        self.progress_dialog.setWindowTitle("特征工程进度")
        self.progress_dialog.setFixedHeight(150)
        self.progress_dialog.setFixedWidth(400)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.show()

        self.feature_engineering_thread.progress_updated.connect(
            self.on_feature_engineering_progress
        )

        self.feature_engineering_thread.start()

    def on_feature_engineering_progress(self, progress, message):
        """特征工程进度更新"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(progress)
            self.progress_dialog.setLabelText(message)

    def on_feature_engineering_finished(
        self, result_data, engineered_file, selected_featurizers
    ):
        """特征工程完成回调"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        if self.feature_engineering_thread is not None:
            try:
                self.feature_engineering_thread.quit()
                self.feature_engineering_thread.wait()
            except Exception:
                pass
            self.feature_engineering_thread = None

        self.data = result_data
        self.update_variable_combo()
        self.status_label.setText("特征工程完成！")

        QMessageBox.information(
            self, '成功',
            f'特征工程完成！\n'
            f'生成了{len(selected_featurizers)}组特征\n'
            f'数据已保存到: {engineered_file}'
        )

        if self.variable_combo.count() > 0:
            self.variable_combo.setCurrentIndex(0)
            self.plot_selected_variable()

    def on_feature_engineering_error(self, error_msg):
        """特征工程错误回调"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

        if self.feature_engineering_thread is not None:
            try:
                self.feature_engineering_thread.quit()
                self.feature_engineering_thread.wait()
            except Exception:
                pass
            self.feature_engineering_thread = None

        self.status_label.setText("特征工程过程中出现错误")
        QMessageBox.critical(
            self, '错误',
            f'执行特征工程时出错: {error_msg}'
        )

    def _perform_feature_engineering_internal(
        self, selected_featurizers, is_learning_data=True
    ):
        """在后台线程中执行的实际特征工程逻辑"""
        try:
            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.progress_updated.emit(
                    5, "正在读取数据文件..."
                )

            if (not hasattr(self, 'data_file_path')
                    or not self.data_file_path):
                raise Exception(
                    '未找到数据文件路径，请先导入数据'
                )

            import pandas as pd
            if self.data_file_path.endswith('.csv'):
                original_data = pd.read_csv(
                    self.data_file_path, encoding='gbk'
                )
            else:
                original_data = pd.read_excel(self.data_file_path)
            feature_data = original_data.copy()

            original_columns = list(feature_data.columns)

            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.progress_updated.emit(
                    10, "正在验证化学分子式..."
                )

            if not feature_data.empty:
                first_column = feature_data.columns[0]
                first_value = (
                    feature_data.iloc[0, 0]
                    if len(feature_data) > 0 else ""
                )

                chemical_processor = ChemicalFormulaProcessor()
                if not chemical_processor.is_chemical_formula(
                    str(first_value)
                ):
                    raise Exception(
                        f'数据第一列 "{first_column}" '
                        f'不是化学分子式格式\n无法执行特征工程'
                    )

            from matminer.featurizers.base import (
                MultipleFeaturizer, BaseFeaturizer
            )
            from matminer.featurizers import composition as cf
            from matminer.featurizers.conversions import (
                StrToComposition,
                CompositionToOxidComposition
            )

            BaseFeaturizer.n_jobs = 1

            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.progress_updated.emit(
                    15, "正在初始化特征化器..."
                )

            featurizer_map = {
                "ElementPropertyMagpie": (
                    cf.ElementProperty.from_preset("magpie")
                ),
                "ElementPropertyDeml": (
                    cf.ElementProperty.from_preset("deml")
                ),
                "OxidationStates": None,
                "AtomicOrbitals": cf.AtomicOrbitals(),
                "ElectronegativityDiff": cf.ElectronegativityDiff(),
                "Stoichiometry": cf.Stoichiometry(),
                "ValenceOrbital": cf.ValenceOrbital()
            }

            featurizers = []
            needs_oxidation = False

            for key in selected_featurizers:
                if key == "OxidationStates":
                    needs_oxidation = True
                elif key in featurizer_map and featurizer_map[key] is not None:
                    featurizers.append(featurizer_map[key])

            first_column = feature_data.columns[0]

            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.progress_updated.emit(
                    20, "正在转换化学分子式..."
                )

            str_to_comp = StrToComposition(
                target_col_id="composition"
            )
            feature_data = str_to_comp.featurize_dataframe(
                feature_data, col_id=first_column,
                ignore_errors=True, pbar=False
            )

            if needs_oxidation:
                if (hasattr(self, 'feature_engineering_thread')
                        and self.feature_engineering_thread is not None):
                    self.feature_engineering_thread.progress_updated.emit(
                        30, "正在处理氧化态特征..."
                    )

                comp_to_oxid = CompositionToOxidComposition()
                feature_data = comp_to_oxid.featurize_dataframe(
                    feature_data, col_id="composition",
                    ignore_errors=True, pbar=False
                )

            if featurizers:
                if (hasattr(self, 'feature_engineering_thread')
                        and self.feature_engineering_thread is not None):
                    self.feature_engineering_thread.progress_updated.emit(
                        40, "正在计算特征..."
                    )

                multi_featurizer = MultipleFeaturizer(featurizers)
                feature_data = multi_featurizer.featurize_dataframe(
                    feature_data, col_id="composition",
                    ignore_errors=True, pbar=False
                )

            if needs_oxidation:
                if (hasattr(self, 'feature_engineering_thread')
                        and self.feature_engineering_thread is not None):
                    self.feature_engineering_thread.progress_updated.emit(
                        70, "正在计算氧化态特征..."
                    )

                oxidation_featurizer = cf.OxidationStates()
                feature_data = oxidation_featurizer.featurize_dataframe(
                    feature_data, col_id="composition_oxid",
                    ignore_errors=True, pbar=False
                )

            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.progress_updated.emit(
                    90, "正在保存结果..."
                )

            if self.project_path:
                features_dir = os.path.join(
                    self.project_path, 'features'
                )
            else:
                features_dir = './features'
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)

            engineered_file = os.path.join(
                features_dir, 'engineered_features.csv'
            )

            engineered_columns = [
                col for col in feature_data.columns
                if col not in original_columns
            ]

            engineered_data = feature_data[engineered_columns]
            engineered_data = engineered_data.dropna(
                axis=1, how='all'
            )

            if 'composition_oxid' in engineered_data.columns:
                engineered_data = engineered_data.drop(
                    columns=['composition_oxid']
                )

            if (is_learning_data
                    and len(original_data.columns) > 1):
                last_column = original_data.columns[-1]
                engineered_data[last_column] = (
                    original_data[last_column].values
                )

            if 'HOMO_character' in engineered_data.columns:
                engineered_data['HOMO_character'] = (
                    engineered_data['HOMO_character'].replace({
                        's': 0, 'p': 1, 'd': 2, 'f': 3
                    })
                )

            if 'HOMO_element' in engineered_data.columns:
                element_to_atomic_number = {
                    'H': 1, 'He': 2, 'Li': 3, 'Be': 4,
                    'B': 5, 'C': 6, 'N': 7, 'O': 8,
                    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
                    'Al': 13, 'Si': 14, 'P': 15, 'S': 16,
                    'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
                    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24,
                    'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
                    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32,
                    'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
                    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
                    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44,
                    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
                    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52,
                    'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
                    'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
                    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
                    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68,
                    'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
                    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76,
                    'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
                    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84,
                    'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88,
                    'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
                    'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
                    'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
                    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104,
                    'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108,
                    'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
                    'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116,
                    'Ts': 117, 'Og': 118
                }
                engineered_data['HOMO_element'] = (
                    engineered_data['HOMO_element'].map(
                        element_to_atomic_number
                    )
                )

            if 'LUMO_character' in engineered_data.columns:
                engineered_data['LUMO_character'] = (
                    engineered_data['LUMO_character'].replace({
                        's': 0, 'p': 1, 'd': 2, 'f': 3
                    })
                )

            if 'LUMO_element' in engineered_data.columns:
                engineered_data['LUMO_element'] = (
                    engineered_data['LUMO_element'].map(
                        element_to_atomic_number
                    )
                )

            engineered_data.to_csv(
                engineered_file, index=False,
                encoding='utf-8-sig'
            )

            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.finished.emit(
                    feature_data, engineered_file,
                    selected_featurizers
                )

        except Exception as e:
            if (hasattr(self, 'feature_engineering_thread')
                    and self.feature_engineering_thread is not None):
                self.feature_engineering_thread.error.emit(str(e))
