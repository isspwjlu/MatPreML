"""数据处理Mixin模块 - MatPreML"""

import os
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from ..chemical import ChemicalFormulaProcessor


class DataMixin:
    """数据处理Mixin类 - 数据加载、预处理、格式检查等"""

    def load_data(self):
        """加载数据文件"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择数据文件', '',
            'CSV files (*.csv);;TXT files (*.txt);;'
            'DAT files (*.dat);;All files (*.*)'
        )

        if not file_path:
            return

        try:
            self.data_file_path = file_path
            self.data = self.read_data_with_delimiter(file_path)

            if self.data is not None:
                self.data = self.check_and_process_data_format(
                    self.data, file_path
                )

                if self.data is not None and not self.data.empty:
                    self.data = self.preprocess_data(self.data)

                    if self.data is not None and not self.data.empty:
                        self.prediction_data = None
                        self.update_variable_combo()
                        QMessageBox.information(
                            self, '成功',
                            f'成功导入数据，共{len(self.data)}行'
                        )
                        self.variable_combo.setCurrentIndex(0)
                        self.plot_selected_variable()
                    else:
                        QMessageBox.warning(
                            self, '错误',
                            '数据预处理后为空，请检查数据格式'
                        )
                else:
                    QMessageBox.warning(
                        self, '错误',
                        '数据格式检查后为空，请检查数据格式'
                    )
            else:
                QMessageBox.warning(
                    self, '错误',
                    '无法读取数据文件，请检查文件格式'
                )
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导入数据时出错: {str(e)}')

    def read_data_with_delimiter(self, file_path):
        """尝试不同的分隔符读取数据"""
        delimiters = [',', ';', '\t', ' ']

        for delimiter in delimiters:
            try:
                if delimiter == ' ':
                    data = pd.read_csv(
                        file_path, sep='\s+', header=0, encoding='utf-8'
                    )
                else:
                    data = pd.read_csv(
                        file_path, delimiter=delimiter,
                        header=0, encoding='utf-8'
                    )
                if not data.empty and len(data.columns) > 1:
                    return data
            except Exception:
                continue

        # 尝试pandas自动检测
        try:
            data = pd.read_csv(file_path, header=0, encoding='utf-8')
            if not data.empty and len(data.columns) > 1:
                return data
        except Exception:
            pass

        # 尝试gbk编码
        try:
            data = pd.read_csv(file_path, header=0, encoding='gbk')
            if not data.empty and len(data.columns) > 1:
                return data
        except Exception:
            pass

        return None

    def check_and_process_data_format(self, data, file_path):
        """检查数据格式并进行相应处理"""
        if data is None or data.empty:
            return data

        chemical_processor = ChemicalFormulaProcessor()

        first_row = data.iloc[0]
        is_first_row_numeric = True

        for value in first_row:
            try:
                float(value)
            except (ValueError, TypeError):
                is_first_row_numeric = False
                break

        if (is_first_row_numeric
                and not any(str(col).startswith('Feature_')
                           for col in data.columns)):
            num_features = len(data.columns) - 1
            feature_names = (
                [f'特征值-{i+1}' for i in range(num_features)] + ['目标值']
            )
            feature_row = pd.DataFrame(
                [feature_names], columns=data.columns
            )
            data = pd.concat([feature_row, data], ignore_index=True)

        processed_data, elements = chemical_processor.process_chemical_data(
            data
        )

        if processed_data is not data:
            self.training_elements = elements

            base_name = os.path.splitext(file_path)[0]
            new_file_path = f"{base_name}_processed.csv"

            processed_data.to_csv(
                new_file_path, index=False, encoding='utf-8'
            )

            QMessageBox.information(
                self, '信息',
                f'检测到化学分子式数据，已处理并保存到: {new_file_path}'
            )

        return processed_data

    def check_and_process_prediction_data_format(self, data, file_path):
        """检查预测数据格式并进行相应处理"""
        if data is None or data.empty:
            return data

        chemical_processor = ChemicalFormulaProcessor()

        first_row = data.iloc[0]
        is_first_row_numeric = True

        for value in first_row:
            try:
                float(value)
            except (ValueError, TypeError):
                is_first_row_numeric = False
                break

        if (is_first_row_numeric
                and not any(str(col).startswith('Feature_')
                           for col in data.columns)):
            num_features = len(data.columns)
            feature_names = [f'特征值-{i+1}' for i in range(num_features)]
            feature_row = pd.DataFrame(
                [feature_names], columns=data.columns
            )
            data = pd.concat([feature_row, data], ignore_index=True)

        processed_data, _ = chemical_processor.process_chemical_data(
            data, self.training_elements
        )

        if processed_data is not data:
            base_name = os.path.splitext(file_path)[0]
            new_file_path = f"{base_name}_processed.csv"

            processed_data.to_csv(
                new_file_path, index=False, encoding='utf-8'
            )

            QMessageBox.information(
                self, '信息',
                f'检测到化学分子式数据，已处理并保存到: {new_file_path}'
            )

        return processed_data

    def preprocess_data(self, data):
        """数据预处理：数值转换、空值处理等"""
        if data is None or data.empty:
            return None

        original_rows = len(data)

        chemical_processor = ChemicalFormulaProcessor()
        first_column = data.iloc[:, 0]
        chemical_count = 0
        total_count = len(first_column)

        for value in first_column:
            if pd.isna(value):
                continue
            if chemical_processor.is_chemical_formula(str(value)):
                chemical_count += 1

        has_chemical_data = (
            chemical_count / total_count > 0.5 and total_count > 0
        ) if total_count > 0 else False

        if has_chemical_data:
            header_row = data.iloc[:1]
            data_rows = data.iloc[1:]
            data_rows = data_rows.dropna(how='all')

            if data_rows.empty:
                return None

            for col in data_rows.columns:
                data_rows[col] = pd.to_numeric(
                    data_rows[col], errors='coerce'
                )

            rows_before_clean = len(data_rows)
            data_rows = data_rows.dropna()
            rows_after_clean = len(data_rows)

            data_rows = data_rows.loc[
                ~(data_rows == 0).all(axis=1)
            ]

            data = pd.concat(
                [header_row, data_rows], ignore_index=True
            )
        else:
            data = data.dropna(how='all')

            if data.empty:
                return None

            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            rows_before_clean = len(data)
            data = data.dropna()
            rows_after_clean = len(data)

            data = data.loc[~(data == 0).all(axis=1)]

        cleaned_rows = rows_before_clean - rows_after_clean
        if cleaned_rows > 0:
            print(
                f"数据清洗: 删除了 {cleaned_rows} 行包含空值的数据"
            )

        return data

    def update_variable_combo(self):
        """更新变量下拉菜单"""
        self.variable_combo.clear()
        if self.data is not None and not self.data.empty:
            chemical_processor = ChemicalFormulaProcessor()

            first_column = self.data.iloc[:, 0]
            chemical_count = 0
            total_count = len(first_column)

            for value in first_column:
                if pd.isna(value):
                    continue
                if chemical_processor.is_chemical_formula(str(value)):
                    chemical_count += 1

            has_chemical_data = (
                chemical_count / total_count > 0.5 and total_count > 0
            )

            if has_chemical_data:
                chemical_columns = []
                for col in self.data.columns[:-1]:
                    if col in chemical_processor.periodic_table:
                        chemical_columns.append(col)

                non_chemical_columns = self.data.columns[
                    len(chemical_columns):-1
                ]
                for i in range(len(non_chemical_columns) - 1, -1, -1):
                    self.variable_combo.addItem(
                        non_chemical_columns[i]
                    )
            else:
                for i in range(
                    len(self.data.columns) - 2, -1, -1
                ):
                    self.variable_combo.addItem(
                        self.data.columns[i]
                    )

    def update_prediction_variable_combo(self):
        """更新预测数据的变量下拉菜单"""
        self.variable_combo.clear()
        if (hasattr(self, 'prediction_data')
                and self.prediction_data is not None
                and not self.prediction_data.empty):
            columns = list(self.prediction_data.columns)
            if 'Prediction' in columns:
                self.variable_combo.addItem('Prediction')
                columns.remove('Prediction')

            for i in range(len(columns) - 1, -1, -1):
                self.variable_combo.addItem(columns[i])

    def on_variable_change(self):
        """变量选择改变时的处理"""
        if ((hasattr(self, 'prediction_data')
                and self.prediction_data is not None
                and not self.prediction_data.empty)
                or (self.data is not None
                    and not self.data.empty)):
            self.plot_selected_variable()

    def clear_data(self):
        """清除数据"""
        self.data = None
        self.prediction_data = None
        self.variable_combo.clear()
        self.train_result_button.setEnabled(False)
        self.init_plot()
