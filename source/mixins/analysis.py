"""数据分析Mixin模块 - MatPreML"""

import os
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QMessageBox, QApplication
import matplotlib.font_manager as fm

from ..chemical import ChemicalFormulaProcessor


class AnalysisMixin:
    """数据分析Mixin类 - 相关性分析、特征重要性分析、变量绘图等"""

    def correlation_analysis(self):
        """特征相关性分析"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.data is None or self.data.empty:
            QMessageBox.warning(self, '错误', '请先导入数据')
            return

        try:
            self.status_label.setText("正在进行特征相关性分析...")
            QApplication.processEvents()

            has_chemical_data = False
            chemical_processor = ChemicalFormulaProcessor()

            first_column = self.data.iloc[:, 0]
            chemical_count = 0
            total_count = len(first_column)

            for value in first_column:
                if pd.isna(value):
                    continue
                if chemical_processor.is_chemical_formula(str(value)):
                    chemical_count += 1

            if chemical_count / total_count > 0.5 and total_count > 0:
                has_chemical_data = True

            if has_chemical_data:
                chemical_columns = []
                for col in self.data.columns[:-1]:
                    if col in chemical_processor.periodic_table:
                        chemical_columns.append(col)

                analysis_data = self.data.iloc[1:, len(chemical_columns):]
                analysis_data = analysis_data.iloc[:, :-1]
                feature_names = self.data.columns[len(chemical_columns):-1]
            else:
                analysis_data = self.data.iloc[1:, :-1]
                feature_names = self.data.columns[:-1]

            correlation_matrix = analysis_data.corr()

            if self.project_path:
                features_dir = os.path.join(self.project_path, 'features')
            else:
                features_dir = './features'
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)

            correlation_file = os.path.join(
                features_dir, 'feature_correlation.csv'
            )
            correlation_matrix.to_csv(
                correlation_file, encoding='utf-8-sig'
            )

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            import seaborn as sns
            sns.heatmap(
                correlation_matrix, annot=False, cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": .8}, ax=ax
            )

            ax.set_title(
                '特征相关性分析热力图',
                fontproperties=fm.FontProperties()
            )
            ax.set_xlabel(
                '特征', fontproperties=fm.FontProperties()
            )
            ax.set_ylabel(
                '特征', fontproperties=fm.FontProperties()
            )

            self.figure.subplots_adjust(
                left=0.1, right=0.95, top=0.9, bottom=0.1
            )
            self.canvas.draw()

            self.status_label.setText("特征相关性分析完成！")

            detail = (
                "包含化学分子式数据，已跳过化学元素列"
                if has_chemical_data
                else "分析了所有特征列"
            )
            QMessageBox.information(
                self, '成功',
                f'特征相关性分析完成！\n{detail}\n'
                f'数据已保存到: {correlation_file}'
            )

        except Exception as e:
            self.status_label.setText("相关性分析过程中出现错误")
            QMessageBox.critical(
                self, '错误',
                f'特征相关性分析时出错: {str(e)}'
            )

    def importance_analysis(self):
        """特征重要性分析"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.data is None or self.data.empty:
            QMessageBox.warning(self, '错误', '请先导入数据')
            return

        if self.model is None:
            QMessageBox.warning(self, '错误', '请先训练模型')
            return

        try:
            self.status_label.setText("正在进行特征重要性分析...")
            QApplication.processEvents()

            feature_names = None
            importance_scores = None

            if hasattr(self.model, 'feature_importances_'):
                importance_scores = self.model.feature_importances_

                if hasattr(self, 'data') and self.data is not None:
                    chemical_processor = ChemicalFormulaProcessor()

                    first_column = self.data.iloc[:, 0]
                    chemical_count = 0
                    total_count = len(first_column)

                    for value in first_column:
                        if pd.isna(value):
                            continue
                        if chemical_processor.is_chemical_formula(
                            str(value)
                        ):
                            chemical_count += 1

                    if (chemical_count / total_count > 0.5
                            and total_count > 0):
                        has_chemical_data = True
                    else:
                        has_chemical_data = False

                    if has_chemical_data:
                        chemical_columns = []
                        for col in self.data.columns[:-1]:
                            if col in chemical_processor.periodic_table:
                                chemical_columns.append(col)

                        feature_names = self.data.columns[
                            len(chemical_columns):-1
                        ]
                    else:
                        feature_names = self.data.columns[:-1]
            else:
                from sklearn.inspection import permutation_importance
                from sklearn.metrics import mean_squared_error

                X_test = self.X_train
                y_test = self.y_train

                if (hasattr(self, 'model_name')
                        and self.model_name == 'BayesianBootstrap'):
                    perm_importance = permutation_importance(
                        self.model, X_test, y_test,
                        scoring=lambda estimator, X, y: (
                            -mean_squared_error(
                                y, estimator.predict(X)
                            )
                        ),
                        random_state=42
                    )
                else:
                    perm_importance = permutation_importance(
                        self.model, X_test, y_test, random_state=42
                    )
                importance_scores = perm_importance.importances_mean

                if hasattr(self, 'data') and self.data is not None:
                    chemical_processor = ChemicalFormulaProcessor()

                    first_column = self.data.iloc[:, 0]
                    chemical_count = 0
                    total_count = len(first_column)

                    for value in first_column:
                        if pd.isna(value):
                            continue
                        if chemical_processor.is_chemical_formula(
                            str(value)
                        ):
                            chemical_count += 1

                    if (chemical_count / total_count > 0.5
                            and total_count > 0):
                        has_chemical_data = True
                    else:
                        has_chemical_data = False

                    if has_chemical_data:
                        chemical_columns = []
                        for col in self.data.columns[:-1]:
                            if col in chemical_processor.periodic_table:
                                chemical_columns.append(col)

                        feature_names = self.data.columns[
                            len(chemical_columns):-1
                        ]
                    else:
                        feature_names = self.data.columns[:-1]

            if importance_scores is None or feature_names is None:
                raise ValueError("无法获取特征重要性信息")

            if len(importance_scores) != len(feature_names):
                raise ValueError(
                    f"特征数量不匹配: 模型有{len(importance_scores)}个特征, "
                    f"数据中有{len(feature_names)}个特征列\n"
                    "请确保使用的是训练时的数据"
                )

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)

            if self.project_path:
                features_dir = os.path.join(
                    self.project_path, 'features'
                )
            else:
                features_dir = './features'
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)

            importance_file = os.path.join(
                features_dir, 'feature_importance.csv'
            )
            importance_df.to_csv(
                importance_file, index=False, encoding='utf-8-sig'
            )

            top_n = min(8, len(importance_df))
            top_importance_df = importance_df.head(top_n)

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            import seaborn as sns
            sns.barplot(
                data=top_importance_df, x='importance',
                y='feature', palette='viridis', ax=ax
            )

            ax.set_title(
                f'特征重要性分析 (Top {top_n})',
                fontproperties=fm.FontProperties()
            )
            ax.set_xlabel(
                '重要性得分', fontproperties=fm.FontProperties()
            )
            ax.set_ylabel(
                '特征', fontproperties=fm.FontProperties()
            )

            for i, v in enumerate(top_importance_df['importance']):
                ax.text(
                    v, i, f'{v:.4f}', va='center',
                    fontproperties=fm.FontProperties()
                )

            self.figure.subplots_adjust(
                left=0.2, right=0.95, top=0.9, bottom=0.1
            )
            self.canvas.draw()

            self.status_label.setText("特征重要性分析完成！")

            QMessageBox.information(
                self, '成功',
                f'特征重要性分析完成！\n'
                f'显示了重要性排名前{top_n}的特征\n'
                f'数据已保存到: {importance_file}'
            )

        except Exception as e:
            self.status_label.setText("特征重要性分析过程中出现错误")
            QMessageBox.critical(
                self, '错误',
                f'特征重要性分析时出错: {str(e)}'
            )

    def plot_selected_variable(self):
        """绘制选中的变量与Y的关系图"""
        if self.data is not None and not self.data.empty:
            show_prediction = (
                hasattr(self, 'prediction_data')
                and self.prediction_data is not None
                and not self.prediction_data.empty
                and self.variable_combo.count() > 0
            )

            if show_prediction and self.variable_combo.count() > 0:
                combo_items = [
                    self.variable_combo.itemText(i)
                    for i in range(self.variable_combo.count())
                ]
                show_prediction = all(
                    item in self.prediction_data.columns
                    for item in combo_items
                )

            if show_prediction:
                self.plot_prediction_data()
                return

            chemical_processor = ChemicalFormulaProcessor()

            first_column = self.data.iloc[:, 0]
            chemical_count = 0
            total_count = len(first_column)

            for value in first_column:
                if pd.isna(value):
                    continue
                if chemical_processor.is_chemical_formula(str(value)):
                    chemical_count += 1

            if chemical_count / total_count > 0.5 and total_count > 0:
                has_chemical_data = True
            else:
                has_chemical_data = False

            if has_chemical_data:
                chemical_columns = []
                for col in self.data.columns[:-1]:
                    if col in chemical_processor.periodic_table:
                        chemical_columns.append(col)

                non_chemical_columns = self.data.columns[
                    len(chemical_columns):-1
                ]
                reversed_index = (
                    len(non_chemical_columns)
                    - 1 - self.variable_combo.currentIndex()
                )
                actual_selected_index = (
                    len(chemical_columns) + reversed_index
                )

                if (actual_selected_index >= len(chemical_columns)
                        and actual_selected_index
                        < len(self.data.columns) - 1):
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)

                    x_data = self.data.iloc[1:, actual_selected_index]
                    y_data = self.data.iloc[1:, -1]

                    x_label = self.data.columns[actual_selected_index]
                    y_label = self.data.columns[-1]

                    ax.plot(
                        x_data, y_data, 'o',
                        markersize=5, linewidth=1
                    )
                    ax.set_xlabel(
                        x_label,
                        fontproperties=fm.FontProperties()
                    )
                    ax.set_ylabel(
                        y_label,
                        fontproperties=fm.FontProperties()
                    )
                    ax.set_title(
                        f'{x_label} vs {y_label}',
                        fontproperties=fm.FontProperties()
                    )
                    ax.grid(True, alpha=0.3)
                    self.figure.subplots_adjust(
                        left=0.1, right=0.95,
                        top=0.9, bottom=0.15
                    )
                    self.canvas.draw()
            else:
                selected_index = (
                    len(self.data.columns)
                    - 2 - self.variable_combo.currentIndex()
                )
                if (selected_index >= 0
                        and selected_index
                        < len(self.data.columns) - 1):
                    self.figure.clear()
                    ax = self.figure.add_subplot(111)

                    x_data = self.data.iloc[1:, selected_index]
                    y_data = self.data.iloc[1:, -1]

                    x_label = self.data.columns[selected_index]
                    y_label = self.data.columns[-1]

                    ax.plot(
                        x_data, y_data, 'o',
                        markersize=5, linewidth=1
                    )
                    ax.set_xlabel(
                        x_label,
                        fontproperties=fm.FontProperties()
                    )
                    ax.set_ylabel(
                        y_label,
                        fontproperties=fm.FontProperties()
                    )
                    ax.set_title(
                        f'{x_label} vs {y_label}',
                        fontproperties=fm.FontProperties()
                    )
                    ax.grid(True, alpha=0.3)
                    self.figure.subplots_adjust(
                        left=0.1, right=0.95,
                        top=0.9, bottom=0.15
                    )
                    self.canvas.draw()

        elif (hasattr(self, 'prediction_data')
              and self.prediction_data is not None
              and not self.prediction_data.empty):
            self.plot_prediction_data()

    def plot_prediction_data(self):
        """绘制预测数据"""
        if (not hasattr(self, 'prediction_data')
                or self.prediction_data is None
                or self.prediction_data.empty):
            return

        selected_index = self.variable_combo.currentIndex()
        if selected_index >= 0:
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            columns = list(self.prediction_data.columns)
            if selected_index < len(columns):
                has_prediction = 'Prediction' in columns

                if has_prediction and selected_index == 0:
                    x_data = self.prediction_data['Prediction']
                    ax.hist(
                        x_data, bins=30,
                        alpha=0.7, color='blue'
                    )
                    ax.set_xlabel(
                        '预测值',
                        fontproperties=fm.FontProperties()
                    )
                    ax.set_ylabel(
                        '频次',
                        fontproperties=fm.FontProperties()
                    )
                    ax.set_title(
                        '预测结果分布',
                        fontproperties=fm.FontProperties()
                    )
                else:
                    feature_columns = [
                        col for col in columns if col != 'Prediction'
                    ]
                    actual_selected_index = selected_index
                    if has_prediction:
                        actual_selected_index = selected_index - 1

                    if actual_selected_index < len(feature_columns):
                        reversed_index = (
                            len(feature_columns)
                            - 1 - actual_selected_index
                        )
                        if reversed_index >= 0:
                            x_label = feature_columns[reversed_index]
                            x_data = self.prediction_data[x_label]

                            if 'Prediction' in columns:
                                y_data = self.prediction_data[
                                    'Prediction'
                                ]
                                ax.plot(
                                    x_data, y_data, 'o',
                                    markersize=5, linewidth=1
                                )
                                ax.set_xlabel(
                                    x_label,
                                    fontproperties=fm.FontProperties()
                                )
                                ax.set_ylabel(
                                    '预测值',
                                    fontproperties=fm.FontProperties()
                                )
                                ax.set_title(
                                    f'{x_label} vs 预测值',
                                    fontproperties=fm.FontProperties()
                                )
                            else:
                                ax.plot(
                                    x_data, 'o-',
                                    markersize=5, linewidth=1
                                )
                                ax.set_xlabel(
                                    '样本',
                                    fontproperties=fm.FontProperties()
                                )
                                ax.set_ylabel(
                                    x_label,
                                    fontproperties=fm.FontProperties()
                                )
                                ax.set_title(
                                    f'{x_label} 数据分布',
                                    fontproperties=fm.FontProperties()
                                )

                ax.grid(True, alpha=0.3)
                self.figure.subplots_adjust(
                    left=0.1, right=0.95,
                    top=0.9, bottom=0.15
                )
                self.canvas.draw()
