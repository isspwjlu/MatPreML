"""机器学习模型定义模块 - MatPreML"""

import ast
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from .config import (
    BAYESIAN_OPT_AVAILABLE, XGBOOST_AVAILABLE, BAYESIAN_BOOTSTRAP_AVAILABLE
)

if XGBOOST_AVAILABLE:
    import xgboost as xgb

if BAYESIAN_BOOTSTRAP_AVAILABLE:
    from bayesian_bootstrap import BayesianBootstrapBagging

if BAYESIAN_OPT_AVAILABLE:
    from skopt.space import Real, Integer, Categorical


class NeuralNetworkWrapper(BaseEstimator, RegressorMixin):
    """神经网络包装器，用于贝叶斯优化"""

    def __init__(self, hidden_layer_sizes_str='(50, 25)', alpha=0.001,
                 learning_rate_init=0.001, max_iter=200, **kwargs):
        self.hidden_layer_sizes_str = hidden_layer_sizes_str
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X, y):
        hidden_layer_sizes = ast.literal_eval(self.hidden_layer_sizes_str)
        self.model_ = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            random_state=42,
            **self.kwargs
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True):
        params = {
            'hidden_layer_sizes_str': self.hidden_layer_sizes_str,
            'alpha': self.alpha,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter
        }
        if deep:
            params.update(self.kwargs)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_model(model_name):
    """创建指定的模型"""
    if model_name == 'BayesianRidge':
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            compute_score=True, max_iter=300, tol=1e-4
        )
    elif model_name == 'SVR':
        return SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale', tol=1e-4)
    elif model_name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_split=5,
            min_samples_leaf=3, max_features=0.4, bootstrap=True,
            oob_score=True, random_state=42, n_jobs=4
        )
    elif model_name == 'GradientBoosting':
        return GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            min_samples_split=8, min_samples_leaf=4, max_features=0.7,
            random_state=42
        )
    elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.5, max_depth=5,
            min_child_weight=4, gamma=0.2, subsample=0.75,
            colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=1.5,
            random_state=42, n_jobs=4
        )
    elif model_name == 'BayesianBootstrap' and BAYESIAN_BOOTSTRAP_AVAILABLE:
        return BayesianBootstrapBagging(
            base_learner=LinearRegression(), n_replications=100,
            resample_size=None, low_mem=True, seed=42
        )
    elif model_name == 'Lasso':
        return Lasso(alpha=0.1, max_iter=2000, tol=1e-4, random_state=42)
    elif model_name == 'DecisionTree':
        return DecisionTreeRegressor(
            max_depth=5, min_samples_split=8, min_samples_leaf=5,
            max_features=0.7, random_state=42
        )
    elif model_name == 'NeuralNetwork':
        return MLPRegressor(
            hidden_layer_sizes=(100, 50), activation='relu',
            solver='adam', alpha=0.001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001,
            max_iter=200, shuffle=True, random_state=42, tol=1e-4,
            verbose=False, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10
        )
    else:
        raise ValueError(f'不支持的模型: {model_name}')


def create_model_with_params(model_name, **params):
    """创建带有指定参数的模型"""
    if model_name == 'NeuralNetwork':
        if 'hidden_layer_sizes_str' in params:
            hls_str = params.pop('hidden_layer_sizes_str')
            params['hidden_layer_sizes'] = ast.literal_eval(hls_str)
        elif 'hidden_layer_sizes' not in params:
            params['hidden_layer_sizes'] = (50, 25)
        return MLPRegressor(**params)
    else:
        return create_model(model_name)


def get_bayesian_search_space(model_name):
    """获取指定模型的贝叶斯优化搜索空间"""
    if not BAYESIAN_OPT_AVAILABLE:
        return None

    if model_name == 'BayesianRidge':
        return {
            'alpha_1': Real(1e-8, 1e0, 'log-uniform'),
            'alpha_2': Real(1e-8, 1e0, 'log-uniform'),
            'lambda_1': Real(1e-8, 1e0, 'log-uniform'),
            'lambda_2': Real(1e-8, 1e0, 'log-uniform'),
            'compute_score': Categorical([True, False])
        }
    elif model_name == 'SVR':
        return {
            'C': Real(1e-3, 1e1, 'log-uniform'),
            'gamma': Real(1e-5, 1e1, 'log-uniform'),
            'epsilon': Real(1e-3, 1e0, 'log-uniform')
        }
    elif model_name == 'RandomForest':
        return {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(1, 10),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 10)
        }
    elif model_name == 'GradientBoosting':
        return {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(1, 10),
            'learning_rate': Real(1e-4, 1e0, 'log-uniform'),
            'subsample': Real(0.5, 1.0, 'uniform'),
            'min_samples_split': Integer(2, 10)
        }
    elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        return {
            'n_estimators': Integer(10, 200),
            'max_depth': Integer(1, 10),
            'learning_rate': Real(1e-4, 1e0, 'log-uniform'),
            'subsample': Real(0.5, 1.0, 'uniform'),
            'colsample_bytree': Real(0.5, 1.0, 'uniform')
        }
    elif model_name == 'BayesianBootstrap' and BAYESIAN_BOOTSTRAP_AVAILABLE:
        return {'n_replications': Integer(10, 200)}
    elif model_name == 'Lasso':
        return {
            'alpha': Real(1e-6, 1e0, 'log-uniform'),
            'max_iter': Integer(10, 1000),
            'tol': Real(1e-6, 1e-2, 'log-uniform')
        }
    elif model_name == 'DecisionTree':
        return {
            'max_depth': Integer(1, 10),
            'min_samples_split': Integer(2, 20),
            'min_samples_leaf': Integer(1, 10),
            'max_features': Real(0.1, 1.0, 'uniform')
        }
    elif model_name == 'NeuralNetwork':
        return {
            'hidden_layer_sizes_str': Categorical(
                ['(50,)', '(100,)', '(100, 50)', '(100, 100)', '(200, 100, 50)']
            ),
            'alpha': Real(1e-5, 1e-1, 'log-uniform'),
            'learning_rate_init': Real(1e-4, 1e-1, 'log-uniform'),
            'max_iter': Integer(100, 1000)
        }
    else:
        return None


def get_available_models():
    """获取当前可用的模型列表"""
    models = []
    if BAYESIAN_BOOTSTRAP_AVAILABLE:
        models.append('BayesianBootstrap')
    models.extend(['BayesianRidge', 'SVR', 'RandomForest', 'GradientBoosting',
                   'Lasso', 'DecisionTree', 'NeuralNetwork'])
    if XGBOOST_AVAILABLE:
        models.append('XGBoost')
    return models
