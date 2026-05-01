"""训练/预测/评估Mixin模块 - MatPreML"""

import os
import pickle
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QMessageBox, QFileDialog, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QApplication
)
from PyQt6.QtCore import QThread
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from ..config import (
    BAYESIAN_OPT_AVAILABLE, BAYESIAN_BOOTSTRAP_AVAILABLE, XGBOOST_AVAILABLE,
    ALL_MODELS
)
from ..models import (
    create_model, NeuralNetworkWrapper, get_bayesian_search_space,
    get_available_models
)
from ..chemical import ChemicalFormulaProcessor
from ..threads import TrainingThread

if BAYESIAN_OPT_AVAILABLE:
    from skopt import BayesSearchCV

if BAYESIAN_BOOTSTRAP_AVAILABLE:
    from bayesian_bootstrap import BayesianBootstrapBagging


def _check_data_infinity(data):
    """检查数据中是否包含无穷大值"""
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        inf_mask = np.isinf(numeric_data.values)
        if inf_mask.any():
            inf_cols = numeric_data.columns[inf_mask.any(axis=0)].tolist()
            raise ValueError(
                f"数据中包含无穷大值(Inf)，请检查以下列: {inf_cols}"
            )


class TrainingMixin:
    """训练/预测/评估Mixin类 - 模型训练、预测、评估等"""

    def train_model(self):
        """训练模型"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.data is None or self.data.empty:
            QMessageBox.warning(self, '错误', '请先导入数据')
            return

        if self.bayesian_opt_enabled and not self.is_registered:
            QMessageBox.warning(self, '警告', '只有注册专业版的用户才能够使用超参调优功能。')
            return

        model_name = self.model_combo.currentText()
        if model_name == '选择模型':
            QMessageBox.warning(self, '错误', '请选择模型')
            return

        self.training_all_models = False

        self.training_thread = TrainingThread(self)
        self.training_thread.training_started.connect(self.on_training_started)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.training_error.connect(self.on_training_error)
        self.training_thread.start()

    def train_all_models(self):
        """训练所有模型"""
        if self.bayesian_opt_enabled and not self.is_registered:
            QMessageBox.warning(self, '警告', '只有注册专业版的用户才能够使用超参调优功能。')
            return

        if self.data is None or self.data.empty:
            QMessageBox.warning(self, '错误', '请先导入数据')
            return

        self.training_thread = TrainingThread(self)
        self.training_thread.training_started.connect(self.on_training_started)
        self.training_thread.training_finished.connect(
            self.on_all_models_training_finished
        )
        self.training_thread.training_error.connect(self.on_training_error)
        self.training_all_models = True
        self.training_thread.start()

    def stop_training(self):
        """停止训练"""
        if (self.training_thread is not None
                and self.training_thread.isRunning()):
            self.training_thread.terminate()
            self.training_thread.wait()
            self.training_thread = None
            self.status_label.setText("训练已被用户终止")
            self.train_button.setEnabled(True)
            self.stop_train_button.setEnabled(False)
            QMessageBox.information(self, '信息', '训练已被终止')
        else:
            QMessageBox.information(self, '信息', '当前没有正在运行的训练')

    def load_previous_model(self):
        """导入先前训练好的模型"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, '导入先前模型', '', 'Pickle files (*.pkl)'
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.model_name = model_data['model_name']

                if self.model_name:
                    index = self.model_combo.findText(self.model_name)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)

                QMessageBox.information(
                    self, '成功', f'先前模型导入成功: {self.model_name}'
                )
            except Exception as e:
                QMessageBox.critical(
                    self, '错误', f'导入先前模型时出错: {str(e)}'
                )

    # ---- 内部训练逻辑 ----

    def _perform_training(self):
        """执行实际的训练逻辑（在训练线程中调用）"""
        if self.data is None or self.data.empty:
            raise ValueError('请先导入数据')

        model_name = self.model_combo.currentText()
        if model_name == '选择模型':
            raise ValueError('请选择模型')

        _check_data_infinity(self.data)

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
        ) if total_count > 0 else False

        if has_chemical_data:
            chemical_columns = []
            for col in self.data.columns[:-1]:
                if col in chemical_processor.periodic_table:
                    chemical_columns.append(col)
            self.X = self.data.iloc[1:, len(chemical_columns):-1]
        else:
            self.X = self.data.iloc[1:, :-1]
        self.y = self.data.iloc[1:, -1]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.X_train_scaled = self.X_train
        self.X_test_scaled = self.X_test

        self.model_name = model_name
        best_params_str = ""

        if self.bayesian_opt_enabled and BAYESIAN_OPT_AVAILABLE:
            search_space = get_bayesian_search_space(model_name)

            if search_space is not None:
                if (model_name == 'BayesianBootstrap'
                        and BAYESIAN_BOOTSTRAP_AVAILABLE):
                    class BayesianBootstrapWrapper(BaseEstimator, RegressorMixin):
                        def __init__(self, n_replications=50, resample_size=None):
                            self.n_replications = n_replications
                            self.resample_size = resample_size

                        def fit(self, X, y):
                            self.model_ = BayesianBootstrapBagging(
                                base_learner=LinearRegression(),
                                n_replications=self.n_replications,
                                resample_size=self.resample_size,
                                low_mem=True,
                                seed=42
                            )
                            self.model_.fit(X, y)
                            return self

                        def predict(self, X):
                            return self.model_.predict(X)

                    current_cpu_cores = min(self.get_cpu_cores() // 2, 4)
                    opt = BayesSearchCV(
                        estimator=BayesianBootstrapWrapper(),
                        search_spaces=search_space,
                        n_iter=self.bayesian_params.get('n_iter', 100),
                        cv=self.bayesian_params.get('cv', 5),
                        random_state=self.bayesian_params.get('random_state', 42),
                        n_jobs=1,
                        scoring='neg_mean_squared_error'
                    )
                    self.status_label.setText("正在超参调优中，请等待...")
                    QApplication.processEvents()
                    opt.fit(self.X_train, self.y_train)
                    self.model = opt.best_estimator_.model_
                    best_params_str = "\n".join(
                        [f"{k}: {v}" for k, v in opt.best_params_.items()]
                    )
                else:
                    if model_name == 'NeuralNetwork':
                        base_model = NeuralNetworkWrapper()
                    else:
                        base_model = create_model(model_name)

                    current_cpu_cores = min(self.get_cpu_cores() // 2, 4)
                    opt = BayesSearchCV(
                        estimator=base_model,
                        search_spaces=search_space,
                        n_iter=self.bayesian_params.get('n_iter', 100),
                        cv=self.bayesian_params.get('cv', 5),
                        random_state=self.bayesian_params.get('random_state', 42),
                        n_jobs=current_cpu_cores,
                        scoring='neg_mean_squared_error'
                    )
                    self.status_label.setText("正在超参调优中，请等待...")
                    QApplication.processEvents()
                    opt.fit(self.X_train_scaled, self.y_train)
                    self.model = opt.best_estimator_
                    best_params_str = "\n".join(
                        [f"{k}: {v}" for k, v in opt.best_params_.items()]
                    )
            else:
                self.model = create_model(model_name)
                if (model_name == 'BayesianBootstrap'
                        and BAYESIAN_BOOTSTRAP_AVAILABLE):
                    self.model.fit(self.X_train, self.y_train)
                else:
                    self.model.fit(self.X_train_scaled, self.y_train)
        else:
            self.model = create_model(model_name)
            if (model_name == 'BayesianBootstrap'
                    and BAYESIAN_BOOTSTRAP_AVAILABLE):
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model.fit(self.X_train_scaled, self.y_train)

        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        metrics_text = f"MSE: {mse:.4f}\nR²: {r2:.4f}"
        if not best_params_str:
            best_params_str = "未使用超参调优"

        return {
            'model': self.model,
            'y_test': self.y_test,
            'y_pred': y_pred,
            'metrics_text': metrics_text,
            'best_params_str': best_params_str
        }

    def _perform_all_models_training(self):
        """执行所有模型的训练逻辑（在训练线程中调用）"""
        all_models = get_available_models()
        model_results = []
        original_model_index = self.model_combo.currentIndex()

        for model_name in all_models:
            try:
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
                ) if total_count > 0 else False

                if has_chemical_data:
                    chemical_columns = []
                    for col in self.data.columns[:-1]:
                        if col in chemical_processor.periodic_table:
                            chemical_columns.append(col)
                    self.X = self.data.iloc[1:, len(chemical_columns):-1]
                else:
                    self.X = self.data.iloc[1:, :-1]
                self.y = self.data.iloc[1:, -1]

                self.X_train, self.X_test, self.y_train, self.y_test = \
                    train_test_split(
                        self.X, self.y, test_size=0.2, random_state=42
                    )

                self.X_train_scaled = self.X_train
                self.X_test_scaled = self.X_test

                self.model_name = model_name
                self.model = create_model(model_name)

                if self.bayesian_opt_enabled and BAYESIAN_OPT_AVAILABLE:
                    search_space = get_bayesian_search_space(model_name)

                    if search_space is not None:
                        if (model_name == 'BayesianBootstrap'
                                and BAYESIAN_BOOTSTRAP_AVAILABLE):
                            class BayesianBootstrapWrapper(
                                BaseEstimator, RegressorMixin
                            ):
                                def __init__(self, n_replications=50,
                                             resample_size=None):
                                    self.n_replications = n_replications
                                    self.resample_size = resample_size

                                def fit(self, X, y):
                                    self.model_ = BayesianBootstrapBagging(
                                        base_learner=LinearRegression(),
                                        n_replications=self.n_replications,
                                        resample_size=self.resample_size,
                                        low_mem=True,
                                        seed=42
                                    )
                                    self.model_.fit(X, y)
                                    return self

                                def predict(self, X):
                                    return self.model_.predict(X)

                            current_cpu_cores = min(
                                self.get_cpu_cores() // 2, 4
                            )
                            opt = BayesSearchCV(
                                estimator=BayesianBootstrapWrapper(),
                                search_spaces=search_space,
                                n_iter=self.bayesian_params.get('n_iter', 100),
                                cv=self.bayesian_params.get('cv', 5),
                                random_state=self.bayesian_params.get(
                                    'random_state', 42
                                ),
                                n_jobs=1,
                                scoring='neg_mean_squared_error'
                            )
                            self.status_label.setText("正在超参调优中，请等待...")
                            QApplication.processEvents()
                            opt.fit(self.X_train, self.y_train)
                            self.model = opt.best_estimator_.model_
                        else:
                            if model_name == 'NeuralNetwork':
                                base_model = NeuralNetworkWrapper()
                            else:
                                base_model = create_model(model_name)

                            current_cpu_cores = min(
                                self.get_cpu_cores() // 2, 4
                            )
                            opt = BayesSearchCV(
                                estimator=base_model,
                                search_spaces=search_space,
                                n_iter=self.bayesian_params.get('n_iter', 100),
                                cv=self.bayesian_params.get('cv', 5),
                                random_state=self.bayesian_params.get(
                                    'random_state', 42
                                ),
                                n_jobs=1,
                                scoring='neg_mean_squared_error'
                            )
                            self.status_label.setText("正在超参调优中，请等待...")
                            QApplication.processEvents()
                            opt.fit(self.X_train_scaled, self.y_train)
                            self.model = opt.best_estimator_
                    else:
                        if (model_name == 'BayesianBootstrap'
                                and BAYESIAN_BOOTSTRAP_AVAILABLE):
                            self.model.fit(self.X_train, self.y_train)
                        else:
                            self.model.fit(
                                self.X_train_scaled, self.y_train
                            )
                else:
                    if (model_name == 'BayesianBootstrap'
                            and BAYESIAN_BOOTSTRAP_AVAILABLE):
                        self.model.fit(self.X_train, self.y_train)
                    else:
                        self.model.fit(self.X_train_scaled, self.y_train)

                y_pred = self.model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)

                if self.project_path:
                    model_dir = os.path.join(self.project_path, 'models')
                else:
                    model_dir = 'models'
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                model_filename = os.path.join(
                    model_dir, f"{model_name}_model.pkl"
                )
                with open(model_filename, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'model_name': model_name,
                        'mse': mse,
                        'r2': r2
                    }, f)

                model_results.append({
                    'model_name': model_name,
                    'mse': mse,
                    'r2': r2,
                    'y_test': self.y_test,
                    'y_pred': y_pred,
                    'model_obj': self.model
                })

            except Exception as e:
                print(f"训练模型 {model_name} 时出错: {str(e)}")
                continue

        model_results.sort(key=lambda x: x['r2'], reverse=True)

        if self.project_path:
            ranking_dir = os.path.join(self.project_path, 'results')
        else:
            ranking_dir = 'results'
        if not os.path.exists(ranking_dir):
            os.makedirs(ranking_dir)

        ranking_filename = os.path.join(ranking_dir, 'model_ranking.txt')
        with open(ranking_filename, 'w', encoding='utf-8') as f:
            f.write("模型排名 (按R²分数从高到低):\n")
            f.write("=" * 40 + "\n")
            for i, result in enumerate(model_results, 1):
                f.write(
                    f"{i}. {result['model_name']}: R² = {result['r2']:.4f}, "
                    f"MSE = {result['mse']:.4f}\n"
                )

        y_pred_last = y_pred if 'y_pred' in locals() else None
        return {
            'model_results': model_results,
            'y_test': self.y_test if hasattr(self, 'y_test') else None,
            'y_pred': y_pred_last,
            'ranking_text': (
                "模型排名 (按R²分数从高到低):\n\n"
                + "\n".join([
                    f"{i}. {result['model_name']}: R² = {result['r2']:.4f}"
                    for i, result in enumerate(model_results, 1)
                ])
            ),
            'ranking_filename': ranking_filename
        }

    # ---- 训练回调 ----

    def on_training_started(self):
        """训练开始时的处理"""
        if (hasattr(self, 'training_all_models')
                and self.training_all_models):
            self.status_label.setText("训练所有模型中，请等待...")
        else:
            model_name = self.model_combo.currentText()
            if model_name != '选择模型':
                self.current_model_text.setText(model_name)
            self.status_label.setText("训练进行中，请等待...")
        self.stop_train_button.setEnabled(True)

    def on_training_finished(self, model, y_test, y_pred, metrics_text,
                             best_params_str):
        """训练完成时的处理"""
        self.model = model
        self.y_test = y_test
        self.metrics_text.setHtml(metrics_text.replace('\n', '<br>').replace('R²', 'R<sup>2</sup>'))

        self.figure.clear()
        has_train_data = (
            hasattr(self, 'X_train') and hasattr(self, 'y_train')
            and self.X_train is not None and self.y_train is not None
        )
        has_test_data = (
            y_test is not None and y_pred is not None
            and len(y_test) > 0 and len(y_pred) > 0
        )

        if has_train_data and has_test_data:
            ax1 = self.figure.add_subplot(121)
            y_pred_train = self.model.predict(self.X_train)
            mse_train = mean_squared_error(self.y_train, y_pred_train)
            r2_train = r2_score(self.y_train, y_pred_train)

            ax1.scatter(self.y_train, y_pred_train, alpha=0.7)
            ax1.plot(
                [self.y_train.min(), self.y_train.max()],
                [self.y_train.min(), self.y_train.max()],
                'r--', lw=2
            )
            ax1.set_xlabel('真实值')
            ax1.set_ylabel('预测值')
            ax1.set_title(
                f"{self.model_name} 训练集评估\n"
                f"MSE: {mse_train:.4f}   R$^2$: {r2_train:.4f}"
            )
            ax1.grid(True, alpha=0.3)

            ax2 = self.figure.add_subplot(122)
            mse_test = mean_squared_error(y_test, y_pred)
            r2_test = r2_score(y_test, y_pred)
            ax2.scatter(y_test, y_pred, alpha=0.7)
            ax2.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2
            )
            ax2.set_xlabel('真实值')
            ax2.set_ylabel('预测值')
            ax2.set_title(
                f"{self.model_name} 测试集评估\n"
                f"MSE: {mse_test:.4f}   R$^2$: {r2_test:.4f}"
            )
            ax2.grid(True, alpha=0.3)
            self.metrics_text.setHtml(
                f"MSE: {mse_test:.4f}<br>R<sup>2</sup>: {r2_test:.4f}"
            )

        elif has_train_data:
            ax1 = self.figure.add_subplot(111)
            y_pred_train = self.model.predict(self.X_train)
            mse_train = mean_squared_error(self.y_train, y_pred_train)
            r2_train = r2_score(self.y_train, y_pred_train)
            html_text = f"MSE: {mse_train:.4f}<br>R<sup>2</sup>: {r2_train:.4f}"
            self.metrics_text.setHtml(html_text)

            ax1.scatter(self.y_train, y_pred_train, alpha=0.7)
            ax1.plot(
                [self.y_train.min(), self.y_train.max()],
                [self.y_train.min(), self.y_train.max()],
                'r--', lw=2
            )
            ax1.set_xlabel('真实值')
            ax1.set_ylabel('预测值')
            ax1.set_title(
                f"{self.model_name} 训练集评估\n"
                f"MSE: {mse_train:.4f}   R$^2$: {r2_train:.4f}"
            )
            ax1.grid(True, alpha=0.3)

        elif has_test_data:
            ax1 = self.figure.add_subplot(111)
            mse_test = mean_squared_error(y_test, y_pred)
            r2_test = r2_score(y_test, y_pred)
            html_text = f"MSE: {mse_test:.4f}<br>R<sup>2</sup>: {r2_test:.4f}"
            self.metrics_text.setHtml(html_text)

            ax1.scatter(y_test, y_pred, alpha=0.7)
            ax1.plot(
                [y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                'r--', lw=2
            )
            ax1.set_xlabel('真实值')
            ax1.set_ylabel('预测值')
            ax1.set_title(
                f"{self.model_name} 测试集评估\n"
                f"MSE: {mse_test:.4f}   R$^2$: {r2_test:.4f}"
            )
            ax1.grid(True, alpha=0.3)

        else:
            ax1 = self.figure.add_subplot(111)
            ax1.text(
                0.5, 0.5, '没有数据可显示',
                ha='center', va='center', transform=ax1.transAxes
            )
            self.metrics_text.setText("无数据")

        self.figure.subplots_adjust(
            left=0.1, right=0.95, top=0.85, bottom=0.15
        )
        self.canvas.draw()
        self.save_model()
        self.save_training_and_test_results()
        self.status_label.setText("模型训练已结束！")
        self.train_result_button.setEnabled(True)
        self.stop_train_button.setEnabled(False)

    def on_all_models_training_finished(self, model_results, y_test, y_pred,
                                        metrics_text, best_params_str):
        """所有模型训练完成时的处理"""
        self.status_label.setText("所有模型训练已完成！")
        ranking_file = self.save_model_ranking(model_results)

        test_ranking_text = "模型排名 (按测试集R²分数从高到低):\n\n"
        test_ranking_text += "\n".join([
            f"{i}. {result['model_name']}: R² = {result['r2']:.4f}"
            for i, result in enumerate(model_results, 1)
        ])
        QMessageBox.information(
            self, '模型训练完成',
            test_ranking_text + f"\n详细排名已保存到: {ranking_file}"
        )

        if model_results:
            best_model = model_results[0]
            self.current_model_text.setText(best_model['model_name'])
            self.model = best_model['model_obj']
            self.model_name = best_model['model_name']
            self.y_test = best_model['y_test']

            if (best_model['y_test'] is not None
                    and best_model['y_pred'] is not None):
                mse_test = mean_squared_error(
                    best_model['y_test'], best_model['y_pred']
                )
                r2_test = r2_score(
                    best_model['y_test'], best_model['y_pred']
                )
                self.metrics_text.setHtml(
                    f"MSE: {mse_test:.4f}<br>R<sup>2</sup>: {r2_test:.4f}"
                )

            self.figure.clear()
            has_train_data = (
                hasattr(self, 'X_train') and hasattr(self, 'y_train')
                and self.X_train is not None and self.y_train is not None
            )
            has_test_data = (
                best_model['y_test'] is not None
                and best_model['y_pred'] is not None
                and len(best_model['y_test']) > 0
                and len(best_model['y_pred']) > 0
            )

            if has_train_data and has_test_data:
                ax1 = self.figure.add_subplot(121)
                y_pred_train = self.model.predict(self.X_train)
                mse_train = mean_squared_error(
                    self.y_train, y_pred_train
                )
                r2_train = r2_score(self.y_train, y_pred_train)

                ax1.scatter(self.y_train, y_pred_train, alpha=0.7)
                ax1.plot(
                    [self.y_train.min(), self.y_train.max()],
                    [self.y_train.min(), self.y_train.max()],
                    'r--', lw=2
                )
                ax1.set_xlabel('真实值')
                ax1.set_ylabel('预测值')
                ax1.set_title(
                    f"{self.model_name} 训练集评估\n"
                    f"MSE: {mse_train:.4f}   R$^2$: {r2_train:.4f}"
                )
                ax1.grid(True, alpha=0.3)

                ax2 = self.figure.add_subplot(122)
                mse_test = mean_squared_error(
                    best_model['y_test'], best_model['y_pred']
                )
                r2_test = r2_score(
                    best_model['y_test'], best_model['y_pred']
                )
                ax2.scatter(
                    best_model['y_test'], best_model['y_pred'], alpha=0.7
                )
                ax2.plot(
                    [best_model['y_test'].min(),
                     best_model['y_test'].max()],
                    [best_model['y_test'].min(),
                     best_model['y_test'].max()],
                    'r--', lw=2
                )
                ax2.set_xlabel('真实值')
                ax2.set_ylabel('预测值')
                ax2.set_title(
                    f"{self.model_name} 测试集评估\n"
                    f"MSE: {mse_test:.4f}   R$^2$: {r2_test:.4f}"
                )
                ax2.grid(True, alpha=0.3)

            elif has_train_data:
                ax1 = self.figure.add_subplot(111)
                y_pred_train = self.model.predict(self.X_train)
                mse_train = mean_squared_error(
                    self.y_train, y_pred_train
                )
                r2_train = r2_score(self.y_train, y_pred_train)

                ax1.scatter(self.y_train, y_pred_train, alpha=0.7)
                ax1.plot(
                    [self.y_train.min(), self.y_train.max()],
                    [self.y_train.min(), self.y_train.max()],
                    'r--', lw=2
                )
                ax1.set_xlabel('真实值')
                ax1.set_ylabel('预测值')
                ax1.set_title(
                    f"{self.model_name} 训练集评估\n"
                    f"MSE: {mse_train:.4f}   R$^2$: {r2_train:.4f}"
                )
                ax1.grid(True, alpha=0.3)

            elif has_test_data:
                ax1 = self.figure.add_subplot(111)
                mse_test = mean_squared_error(
                    best_model['y_test'], best_model['y_pred']
                )
                r2_test = r2_score(
                    best_model['y_test'], best_model['y_pred']
                )

                ax1.scatter(
                    best_model['y_test'], best_model['y_pred'], alpha=0.7
                )
                ax1.plot(
                    [best_model['y_test'].min(),
                     best_model['y_test'].max()],
                    [best_model['y_test'].min(),
                     best_model['y_test'].max()],
                    'r--', lw=2
                )
                ax1.set_xlabel('真实值')
                ax1.set_ylabel('预测值')
                ax1.set_title(
                    f"{self.model_name} 测试集评估\n"
                    f"MSE: {mse_test:.4f}   R$^2$: {r2_test:.4f}"
                )
                ax1.grid(True, alpha=0.3)

            else:
                ax1 = self.figure.add_subplot(111)
                ax1.text(
                    0.5, 0.5, '没有数据可显示',
                    ha='center', va='center', transform=ax1.transAxes
                )

            self.figure.subplots_adjust(
                left=0.1, right=0.95, top=0.85, bottom=0.15
            )
            self.canvas.draw()
            self.save_all_models_results(model_results)

        self.train_button.setEnabled(True)
        self.stop_train_button.setEnabled(False)

    def on_training_error(self, error_message):
        """训练出错时的处理"""
        self.status_label.setText("训练过程中出现错误")
        self.stop_train_button.setEnabled(False)
        QMessageBox.critical(self, '错误', f'训练模型时出错: {error_message}')

    # ---- 模型评估 ----

    def show_train_result(self):
        """显示训练结果"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None or self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, '错误', '请先训练模型')
            return

        try:
            if self.model_name == 'BayesianBootstrap':
                y_pred = self.model.predict(self.X_test)
            else:
                y_pred = self.model.predict(self.X_test_scaled)

            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)

            self.figure.clear()
            ax = self.figure.add_subplot(111)

            ax.scatter(
                self.y_test, y_pred, c='blue', marker='o',
                label='数据点', alpha=0.7, s=30
            )
            min_val = min(min(self.y_test), min(y_pred))
            max_val = max(max(self.y_test), max(y_pred))
            ax.plot(
                [min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='理想预测线'
            )
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(
                f'{self.model_name} 训练结果对比\n'
                f'MSE: {mse:.4f}, R$^2$: {r2:.4f}'
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.figure.subplots_adjust(
                left=0.1, right=0.95, top=0.9, bottom=0.15
            )
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(
                self, '错误', f'显示训练结果时出错: {str(e)}'
            )

    def evaluate_train_set(self):
        """评估训练集"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None or self.X_train is None or self.y_train is None:
            QMessageBox.warning(self, '错误', '请先训练模型')
            return

        try:
            y_pred_train = self.model.predict(self.X_train)
            mse_train = mean_squared_error(self.y_train, y_pred_train)
            r2_train = r2_score(self.y_train, y_pred_train)

            if self.project_path:
                results_dir = os.path.join(self.project_path, 'results')
            else:
                results_dir = './results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            eval_file_train = os.path.join(
                results_dir, 'train_evaluation.txt'
            )
            with open(eval_file_train, 'w', encoding='utf-8') as f:
                f.write(f"训练集评估结果\n")
                f.write(f"模型: {self.model_name}\n")
                f.write(f"MSE: {mse_train:.4f}\n")
                f.write(f"R²: {r2_train:.4f}\n")

            if (self.X_test is not None and self.y_test is not None):
                y_pred_test = self.model.predict(self.X_test)
                mse_test = mean_squared_error(self.y_test, y_pred_test)
                r2_test = r2_score(self.y_test, y_pred_test)

                eval_file_test = os.path.join(
                    results_dir, 'test_evaluation.txt'
                )
                with open(eval_file_test, 'w', encoding='utf-8') as f:
                    f.write(f"测试集评估结果\n")
                    f.write(f"模型: {self.model_name}\n")
                    f.write(f"MSE: {mse_test:.4f}\n")
                    f.write(f"R²: {r2_test:.4f}\n")

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.scatter(
                self.y_train, y_pred_train, c='blue', marker='o',
                label='数据点', alpha=0.7, s=30
            )
            min_val = min(min(self.y_train), min(y_pred_train))
            max_val = max(max(self.y_train), max(y_pred_train))
            ax.plot(
                [min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='理想预测线'
            )
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(
                f'{self.model_name} 训练集评估\n'
                f'MSE: {mse_train:.4f}, R$^2$: {r2_train:.4f}'
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.figure.subplots_adjust(
                left=0.1, right=0.95, top=0.9, bottom=0.15
            )
            self.canvas.draw()

            QMessageBox.information(
                self, '成功',
                f'训练集评估完成，结果已保存到 {eval_file_train}\n'
                f'测试集评估结果也已保存'
            )
        except Exception as e:
            QMessageBox.critical(
                self, '错误', f'训练集评估时出错: {str(e)}'
            )

    def evaluate_test_set(self):
        """评估测试集"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None or self.X_test is None or self.y_test is None:
            QMessageBox.warning(self, '错误', '请先训练模型')
            return

        try:
            y_pred_test = self.model.predict(self.X_test)
            mse_test = mean_squared_error(self.y_test, y_pred_test)
            r2_test = r2_score(self.y_test, y_pred_test)

            if self.project_path:
                results_dir = os.path.join(self.project_path, 'results')
            else:
                results_dir = './results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            eval_file_test = os.path.join(
                results_dir, 'test_evaluation.txt'
            )
            with open(eval_file_test, 'w', encoding='utf-8') as f:
                f.write(f"测试集评估结果\n")
                f.write(f"模型: {self.model_name}\n")
                f.write(f"MSE: {mse_test:.4f}\n")
                f.write(f"R²: {r2_test:.4f}\n")

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.scatter(
                self.y_test, y_pred_test, c='blue', marker='o',
                label='数据点', alpha=0.7, s=30
            )
            min_val = min(min(self.y_test), min(y_pred_test))
            max_val = max(max(self.y_test), max(y_pred_test))
            ax.plot(
                [min_val, max_val], [min_val, max_val],
                'r--', lw=2, label='理想预测线'
            )
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(
                f'{self.model_name} 测试集评估\n'
                f'MSE: {mse_test:.4f}, R$^2$: {r2_test:.4f}'
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            self.figure.subplots_adjust(
                left=0.1, right=0.95, top=0.9, bottom=0.15
            )
            self.canvas.draw()

            QMessageBox.information(
                self, '成功',
                f'测试集评估完成，结果已保存到 {eval_file_test}'
            )
        except Exception as e:
            QMessageBox.critical(
                self, '错误', f'测试集评估时出错: {str(e)}'
            )

    def show_model_selection(self):
        """显示模型选择"""
        dialog = QDialog(self)
        dialog.setWindowTitle('选择模型')
        dialog.setModal(True)
        dialog.resize(200, 80)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        model_label = QLabel('选择模型:')
        model_combo = QComboBox()
        model_combo.addItems(['选择模型'])
        if BAYESIAN_BOOTSTRAP_AVAILABLE:
            model_combo.addItem('BayesianBootstrap')
        model_combo.addItems([
            'BayesianRidge', 'SVR', 'RandomForest', 'GradientBoosting',
            'Lasso', 'DecisionTree', 'NeuralNetwork'
        ])
        if XGBOOST_AVAILABLE:
            model_combo.addItem('XGBoost')

        model_combo.setCurrentIndex(self.model_combo.currentIndex())

        def sync_model_selection(index):
            self.model_combo.setCurrentIndex(index)

        model_combo.currentIndexChanged.connect(sync_model_selection)

        layout.addWidget(model_label)
        layout.addWidget(model_combo)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        ok_button = QPushButton('确定')
        cancel_button = QPushButton('取消')
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        dialog.exec()

    # ---- 模型保存 ----

    def save_model(self):
        """保存模型"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None:
            return

        try:
            if self.project_path:
                model_dir = os.path.join(self.project_path, 'models')
            else:
                model_dir = 'models'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_filename = os.path.join(
                model_dir, f"{self.model_name}_model.pkl"
            )
            with open(model_filename, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'model_name': self.model_name
                }, f)
        except Exception as e:
            QMessageBox.warning(self, '警告', f'保存模型时出错: {str(e)}')

    def save_model_ranking(self, model_results):
        """保存模型排名（训练集和测试集）"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return "./results/model_ranking.txt"

        try:
            if self.project_path:
                results_dir = os.path.join(self.project_path, 'results')
            else:
                results_dir = './results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            if (hasattr(self, 'X_train') and hasattr(self, 'y_train')
                    and self.X_train is not None and self.y_train is not None):
                train_ranking = []
                for result in model_results:
                    y_pred_train = result['model_obj'].predict(self.X_train)
                    r2_train = r2_score(self.y_train, y_pred_train)
                    train_ranking.append({
                        'model_name': result['model_name'],
                        'r2_train': r2_train
                    })
                train_ranking.sort(
                    key=lambda x: x['r2_train'], reverse=True
                )
            else:
                train_ranking = []

            test_ranking = []
            for result in model_results:
                test_ranking.append({
                    'model_name': result['model_name'],
                    'r2_test': result['r2']
                })
            test_ranking.sort(key=lambda x: x['r2_test'], reverse=True)

            ranking_file = os.path.join(results_dir, 'model_ranking.txt')
            with open(ranking_file, 'w', encoding='utf-8') as f:
                f.write("模型排名 (按训练集R²分数从高到低):\n")
                f.write("=" * 40 + "\n")
                for i, rank in enumerate(train_ranking, 1):
                    f.write(
                        f"{i}. {rank['model_name']}: "
                        f"R² = {rank['r2_train']:.4f}\n"
                    )
                f.write("\n")
                f.write("模型排名 (按测试集R²分数从高到低):\n")
                f.write("=" * 40 + "\n")
                for i, rank in enumerate(test_ranking, 1):
                    f.write(
                        f"{i}. {rank['model_name']}: "
                        f"R² = {rank['r2_test']:.4f}\n"
                    )
                f.write("\n")

            return ranking_file
        except Exception:
            return "./results/model_ranking.txt"

    def save_all_models_results(self, model_results):
        """保存所有模型的训练集和测试集评估结果以及预测值"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        try:
            if self.project_path:
                results_dir = os.path.join(self.project_path, 'results')
            else:
                results_dir = './results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            for result in model_results:
                model_name = result['model_name'].replace(
                    "/", "_"
                ).replace("\\", "_")

                if (hasattr(self, 'X_train') and hasattr(self, 'y_train')
                        and self.X_train is not None
                        and self.y_train is not None):
                    y_pred_train = result['model_obj'].predict(self.X_train)
                    mse_train = mean_squared_error(
                        self.y_train, y_pred_train
                    )
                    r2_train = r2_score(self.y_train, y_pred_train)

                    train_eval_file = os.path.join(
                        results_dir, f"{model_name}_train_evaluation.txt"
                    )
                    with open(train_eval_file, 'w', encoding='utf-8') as f:
                        f.write(f"训练集评估结果\n")
                        f.write(f"模型: {result['model_name']}\n")
                        f.write(f"MSE: {mse_train:.4f}\n")
                        f.write(f"R²: {r2_train:.4f}\n")

                    train_pred_file = os.path.join(
                        results_dir,
                        f"{model_name}_train_predictions.csv"
                    )
                    train_data = self.X_train.copy()
                    train_data['真实值'] = self.y_train
                    train_data['预测值'] = y_pred_train
                    train_data.to_csv(
                        train_pred_file, index=False, encoding='utf-8-sig'
                    )

                if (hasattr(self, 'X_test') and hasattr(self, 'y_test')
                        and self.X_test is not None
                        and self.y_test is not None):
                    y_pred_test = result['model_obj'].predict(self.X_test)
                    mse_test = mean_squared_error(
                        self.y_test, y_pred_test
                    )
                    r2_test = r2_score(self.y_test, y_pred_test)

                    test_eval_file = os.path.join(
                        results_dir, f"{model_name}_test_evaluation.txt"
                    )
                    with open(test_eval_file, 'w', encoding='utf-8') as f:
                        f.write(f"测试集评估结果\n")
                        f.write(f"模型: {result['model_name']}\n")
                        f.write(f"MSE: {mse_test:.4f}\n")
                        f.write(f"R²: {r2_test:.4f}\n")

                    test_pred_file = os.path.join(
                        results_dir,
                        f"{model_name}_test_predictions.csv"
                    )
                    test_data = self.X_test.copy()
                    test_data['真实值'] = self.y_test
                    test_data['预测值'] = y_pred_test
                    test_data.to_csv(
                        test_pred_file, index=False, encoding='utf-8-sig'
                    )
        except Exception:
            pass

    def save_training_and_test_results(self):
        """保存训练集和测试集评估结果以及预测值"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        try:
            if self.project_path:
                results_dir = os.path.join(self.project_path, 'results')
            else:
                results_dir = './results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            if not (hasattr(self, 'model') and self.model is not None):
                return

            if (hasattr(self, 'X_train') and hasattr(self, 'y_train')
                    and self.X_train is not None
                    and self.y_train is not None):
                y_pred_train = self.model.predict(self.X_train)
                mse_train = mean_squared_error(
                    self.y_train, y_pred_train
                )
                r2_train = r2_score(self.y_train, y_pred_train)

                train_eval_file = os.path.join(
                    results_dir, 'train_evaluation.txt'
                )
                with open(train_eval_file, 'w', encoding='utf-8') as f:
                    f.write(f"训练集评估结果\n")
                    f.write(f"模型: {self.model_name}\n")
                    f.write(f"MSE: {mse_train:.4f}\n")
                    f.write(f"R²: {r2_train:.4f}\n")

                train_pred_file = os.path.join(
                    results_dir, 'train_predictions.csv'
                )
                train_data = self.X_train.copy()
                train_data['真实值'] = self.y_train
                train_data['预测值'] = y_pred_train
                train_data.to_csv(
                    train_pred_file, index=False, encoding='utf-8-sig'
                )

            if (hasattr(self, 'X_test') and hasattr(self, 'y_test')
                    and self.X_test is not None
                    and self.y_test is not None):
                y_pred_test = self.model.predict(self.X_test)
                mse_test = mean_squared_error(self.y_test, y_pred_test)
                r2_test = r2_score(self.y_test, y_pred_test)

                test_eval_file = os.path.join(
                    results_dir, 'test_evaluation.txt'
                )
                with open(test_eval_file, 'w', encoding='utf-8') as f:
                    f.write(f"测试集评估结果\n")
                    f.write(f"模型: {self.model_name}\n")
                    f.write(f"MSE: {mse_test:.4f}\n")
                    f.write(f"R²: {r2_test:.4f}\n")

                test_pred_file = os.path.join(
                    results_dir, 'test_predictions.csv'
                )
                test_data = self.X_test.copy()
                test_data['真实值'] = self.y_test
                test_data['预测值'] = y_pred_test
                test_data.to_csv(
                    test_pred_file, index=False, encoding='utf-8-sig'
                )
        except Exception:
            pass

    # ---- 预测功能 ----

    def generate_predict_data(self):
        """生成预测数据"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.data is None or self.data.empty:
            QMessageBox.warning(self, '错误', '请先导入学习数据')
            return

        try:
            self.status_label.setText("正在生成预测数据...")
            QApplication.processEvents()

            if self.project_path:
                predict_dir = os.path.join(self.project_path, 'predict')
            else:
                predict_dir = 'predict'
            if not os.path.exists(predict_dir):
                os.makedirs(predict_dir)

            feature_columns = self.data.columns[:-1]
            num_features = len(feature_columns)

            column_values = []
            for col in feature_columns:
                col_min = self.data[col].min()
                col_max = self.data[col].max()

                extended_min = (
                    col_min * 1.5 if col_min >= 0 else col_min * 1.5
                )
                extended_max = (
                    col_max * 1.5 if col_max >= 0 else col_max * 1.5
                )

                if extended_min == extended_max:
                    values = [extended_min] * 1
                else:
                    min_val = min(extended_min, extended_max)
                    max_val = max(extended_min, extended_max)
                    if num_features <= 5:
                        values = np.linspace(min_val, max_val, 10)
                    elif 5 < num_features <= 10:
                        values = np.linspace(min_val, max_val, 4)
                    else:
                        values = np.linspace(min_val, max_val, 1)
                column_values.append(values)

            from itertools import product
            generated_data = list(product(*column_values))
            generated_df = pd.DataFrame(
                generated_data, columns=feature_columns
            )

            predict_filename = os.path.join(
                predict_dir, 'generated_predict_data.csv'
            )
            if num_features <= 10:
                generated_df.to_csv(
                    predict_filename, index=False, encoding='utf-8'
                )
                QMessageBox.information(
                    self, '成功',
                    f'预测数据已生成并保存到: {predict_filename}\n'
                    f'共生成{len(generated_data)}行数据'
                )
            else:
                QMessageBox.warning(
                    self, '警告',
                    '由于特征值个数过多，导致生成的数据量过大，'
                    '请手动生成预测数据'
                )

        except Exception as e:
            QMessageBox.critical(
                self, '错误', f'生成预测数据时出错: {str(e)}'
            )

    def predict_data(self):
        """预测数据"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None:
            reply = QMessageBox.question(
                self, '提示',
                '尚未导入训练模型，是否现在导入模型进行预测？',
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                if not self.load_trained_model_for_prediction():
                    QMessageBox.information(
                        self, '提示', '请先导入训练模型再进行预测'
                    )
                    return
            else:
                QMessageBox.information(
                    self, '提示', '请先导入训练模型再进行预测'
                )
                return

        if not hasattr(self, 'X') or self.X is None:
            QMessageBox.warning(
                self, '错误',
                '无法获取训练数据的特征信息，请重新导入模型'
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择预测数据文件', '',
            'CSV files (*.csv);;TXT files (*.txt);;'
            'DAT files (*.dat);;All files (*.*)'
        )

        if file_path:
            try:
                self.prediction_data_file_path = file_path
                pred_data = self.read_data_with_delimiter(file_path)
                pred_data = self.check_and_process_prediction_data_format(
                    pred_data, file_path
                )
                pred_data = self.preprocess_data(pred_data)

                if pred_data is not None and not pred_data.empty:
                    if hasattr(self, 'X') and self.X is not None:
                        train_features_count = self.X.shape[1]
                    else:
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

                        has_chemical_data = (
                            chemical_count / total_count > 0.5
                            and total_count > 0
                        ) if total_count > 0 else False

                        if has_chemical_data:
                            chemical_columns = []
                            for col in self.data.columns[:-1]:
                                if col in chemical_processor.periodic_table:
                                    chemical_columns.append(col)
                            train_features_count = (
                                len(self.data.columns)
                                - len(chemical_columns) - 1
                            )
                        else:
                            train_features_count = (
                                len(self.data.columns) - 1
                            )

                    if pred_data.shape[1] == train_features_count:
                        pred_result = self.model.predict(pred_data)

                        if self.project_path:
                            result_dir = os.path.join(
                                self.project_path, 'predict'
                            )
                        else:
                            result_dir = 'predict'
                        if not os.path.exists(result_dir):
                            os.makedirs(result_dir)

                        combined_df = pred_data.copy()
                        combined_df['Prediction'] = pred_result

                        result_filename = os.path.join(
                            result_dir, 'prediction_result.csv'
                        )
                        combined_df.to_csv(
                            result_filename, index=False,
                            encoding='utf-8-sig'
                        )

                        sorted_combined_df = combined_df.sort_values(
                            by='Prediction', ascending=False
                        )
                        sorted_filename = os.path.join(
                            result_dir, 'prediction_result_sort.csv'
                        )
                        sorted_combined_df.to_csv(
                            sorted_filename, index=False,
                            encoding='utf-8-sig'
                        )

                        self.auto_import_prediction_result(
                            result_filename, combined_df
                        )
                        self.status_label.setText(
                            f"预测完成，结果已保存到 {result_filename}"
                        )
                        QMessageBox.information(
                            self, '成功',
                            f'预测完成，结果已保存到 {result_filename}'
                        )
                    else:
                        QMessageBox.warning(
                            self, '错误',
                            f'预测数据列数({pred_data.shape[1]})'
                            f'与训练数据特征数({train_features_count})不匹配'
                        )
                else:
                    QMessageBox.warning(self, '错误', '无法读取预测数据')

            except Exception as e:
                QMessageBox.critical(
                    self, '错误', f'预测时出错: {str(e)}'
                )

    def predict_generated_data(self, file_path):
        """对生成的预测数据进行预测"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return

        if self.model is None:
            reply = QMessageBox.question(
                self, '提示',
                '尚未导入训练模型，是否现在导入模型进行预测？',
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                if not self.load_trained_model_for_prediction():
                    QMessageBox.information(
                        self, '提示', '请先导入训练模型再进行预测'
                    )
                    return
            else:
                QMessageBox.information(
                    self, '提示', '请先导入训练模型再进行预测'
                )
                return

        try:
            pred_data = self.read_data_with_delimiter(file_path)
            pred_data = self.check_and_process_prediction_data_format(
                pred_data, file_path
            )

            if pred_data is not None and not pred_data.empty:
                train_features_count = 0
                if hasattr(self, 'X') and self.X is not None:
                    train_features_count = self.X.shape[1]
                elif (hasattr(self, 'data')
                      and self.data is not None):
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

                    has_chemical_data = (
                        chemical_count / total_count > 0.5
                        and total_count > 0
                    ) if total_count > 0 else False

                    if has_chemical_data:
                        chemical_columns = []
                        for col in self.data.columns[:-1]:
                            if col in chemical_processor.periodic_table:
                                chemical_columns.append(col)
                        train_features_count = (
                            len(self.data.columns)
                            - len(chemical_columns) - 1
                        )
                    else:
                        train_features_count = (
                            len(self.data.columns) - 1
                        )

                if train_features_count == 0:
                    QMessageBox.warning(
                        self, '错误',
                        '无法获取训练数据的特征信息，请重新导入模型'
                    )
                    return

                if pred_data.shape[1] == train_features_count:
                    pred_result = self.model.predict(pred_data)

                    if self.project_path:
                        result_dir = os.path.join(
                            self.project_path, 'predict'
                        )
                    else:
                        result_dir = 'predict'
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    combined_df = pred_data.copy()
                    combined_df['Prediction'] = pred_result

                    result_filename = os.path.join(
                        result_dir,
                        'generated_prediction_result.csv'
                    )
                    combined_df.to_csv(
                        result_filename, index=False,
                        encoding='utf-8-sig'
                    )

                    sorted_combined_df = combined_df.sort_values(
                        by='Prediction', ascending=False
                    )
                    sorted_filename = os.path.join(
                        result_dir,
                        'generated_prediction_result_sort.csv'
                    )
                    sorted_combined_df.to_csv(
                        sorted_filename, index=False,
                        encoding='utf-8-sig'
                    )

                    self.auto_import_prediction_result(
                        result_filename, combined_df
                    )
                    self.status_label.setText(
                        f"预测完成，结果已保存到 {result_filename}"
                    )
                    QMessageBox.information(
                        self, '成功',
                        f'预测完成，结果已保存到 {result_filename}'
                    )
                else:
                    QMessageBox.warning(
                        self, '错误',
                        f'预测数据列数({pred_data.shape[1]})'
                        f'与训练数据特征数({train_features_count})不匹配'
                    )
            else:
                QMessageBox.warning(self, '错误', '无法读取预测数据')

        except Exception as e:
            QMessageBox.critical(
                self, '错误', f'预测时出错: {str(e)}'
            )

    def auto_import_prediction_result(self, file_path, pred_data):
        """自动导入预测结果数据并在绘图窗口显示"""
        try:
            self.prediction_data = pred_data
            self.update_prediction_variable_combo()
            if self.variable_combo.count() > 0:
                self.variable_combo.setCurrentIndex(0)
                self.plot_selected_variable()
        except Exception as e:
            QMessageBox.critical(
                self, '错误',
                f'自动导入预测结果时出错: {str(e)}'
            )

    def load_trained_model_for_prediction(self):
        """为预测导入训练好的模型"""
        if not self.project_loaded:
            QMessageBox.warning(self, '警告', '请先创建或打开项目')
            return False

        file_path, _ = QFileDialog.getOpenFileName(
            self, '导入训练模型', '', 'Pickle files (*.pkl)'
        )

        if file_path:
            try:
                with open(file_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['model']
                self.model_name = model_data.get(
                    'model_name', '未知模型'
                )

                QMessageBox.information(
                    self, '成功',
                    f'训练模型导入成功: {self.model_name}'
                )
                return True
            except Exception as e:
                QMessageBox.critical(
                    self, '错误',
                    f'导入训练模型时出错: {str(e)}'
                )
                return False
        return False

    # ---- 高分辨率图片保存 ----

    def add_high_res_save_action(self):
        """添加自定义的高分辨率保存按钮"""
        from PyQt6.QtWidgets import QAction

        save_action = QAction('保存高分辨率图片', self.toolbar)
        save_action.setToolTip('点我,保存高分辨率图片 (300 DPI)')
        save_action.triggered.connect(self.save_high_res_figure)
        self.toolbar.addAction(save_action)

    def save_high_res_figure(self):
        """保存高分辨率图片"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, '保存高分辨率图片', '',
            'PNG files (*.png);;JPEG files (*.jpg);;'
            'PDF files (*.pdf);;SVG files (*.svg)'
        )

        if file_path:
            try:
                self.figure.savefig(
                    file_path, dpi=300, bbox_inches='tight'
                )
                QMessageBox.information(
                    self, '成功',
                    f'图片已保存为高分辨率 (300 DPI): {file_path}'
                )
            except Exception as e:
                QMessageBox.critical(
                    self, '错误', f'保存图片时出错: {str(e)}'
                )
