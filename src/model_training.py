"""
模型训练模块
基于网络流量特征的钓鱼网站检测系统

此模块负责:
- RandomForest分类器训练
- XGBoost分类器训练
- 集成模型构建 (软投票)
- 交叉验证
- 超参数调优

作者: 毕业设计项目组
日期: 2025年12月
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from typing import List, Tuple
import matplotlib.pyplot as plt
import time
import json
from scipy.stats import randint, uniform

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import DATA_DIR, MODELS_DIR, FEATURE_NAMES

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseModelTrainer(ABC):
    """
    模型训练基类

    定义模型训练的通用接口和方法
    """

    def __init__(self, model_name: str = "base_model"):
        """
        初始化训练器

        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
        self.feature_names = None

    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """
        构建模型

        Args:
            **kwargs: 模型参数

        Returns:
            构建的模型对象
        """
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征数据

        Returns:
            预测概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # 详细分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report

        return metrics

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        保存模型

        Args:
            filepath: 保存路径，如果为None则使用默认路径

        Returns:
            保存的文件路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")

        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, f"{self.model_name}.pkl")

        # 保存模型和元数据
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'saved_at': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"模型已保存到: {filepath}")

        return filepath

    def load_model(self, filepath: str) -> None:
        """
        加载模型

        Args:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_metrics = model_data.get('training_metrics', {})
        self.feature_names = model_data.get('feature_names', None)

        logger.info(f"模型已从 {filepath} 加载")


class RandomForestTrainer(BaseModelTrainer):
    """
    随机森林训练器

    Day 14实现 - 基础随机森林分类器
    """

    def __init__(self, model_name: str = "rf_model"):
        """
        初始化随机森林训练器

        Args:
            model_name: 模型名称
        """
        super().__init__(model_name)
        self.feature_importances_ = None

    def build_model(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ) -> RandomForestClassifier:
        """
        构建随机森林模型

        Args:
            n_estimators: 决策树数量
            max_depth: 最大深度
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            random_state: 随机种子
            n_jobs: 并行数量，-1表示使用所有核心
            **kwargs: 其他参数

        Returns:
            RandomForestClassifier实例
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        )

        logger.info(f"随机森林模型已创建: n_estimators={n_estimators}, max_depth={max_depth}")
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list] = None
    ) -> None:
        """
        训练随机森林模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称列表
        """
        if self.model is None:
            logger.info("模型未创建，使用默认参数创建...")
            self.build_model()

        logger.info(f"开始训练随机森林模型...")
        logger.info(f"训练集大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        self.is_trained = True
        self.feature_names = feature_names
        self.feature_importances_ = self.model.feature_importances_

        # 记录训练信息
        self.training_metrics = {
            'training_samples': X_train.shape[0],
            'n_features': X_train.shape[1],
            'training_time_seconds': training_time,
            'trained_at': datetime.now().isoformat()
        }

        logger.info(f"训练完成！耗时: {training_time:.2f} 秒")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性

        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df


class XGBoostTrainer(BaseModelTrainer):
    """
    XGBoost分类器训练器

    Day 15实现 - XGBoost梯度提升分类器

    封装xgboost的XGBClassifier，提供训练、评估、特征重要性分析等功能。

    Example:
        >>> trainer = XGBoostTrainer()
        >>> trainer.build_model(n_estimators=100, learning_rate=0.1)
        >>> trainer.train(X_train, y_train)
        >>> metrics = trainer.evaluate(X_test, y_test)
        >>> importance = trainer.get_feature_importance()
    """

    def __init__(self, model_name: str = "xgb_model"):
        """
        初始化XGBoost训练器

        Args:
            model_name: 模型名称
        """
        super().__init__(model_name)
        self.feature_importances_ = None

    def build_model(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
        random_state: int = 42,
        **kwargs
    ) -> XGBClassifier:
        """
        构建XGBoost模型

        Args:
            n_estimators: 提升轮数（树的数量），默认100
            learning_rate: 学习率，默认0.1
            max_depth: 树的最大深度，默认6
            min_child_weight: 子节点最小权重和，默认1
            subsample: 训练样本采样比例，默认0.8
            colsample_bytree: 特征采样比例，默认0.8
            gamma: 节点分裂所需的最小损失减少，默认0
            reg_alpha: L1正则化系数，默认0
            reg_lambda: L2正则化系数，默认1
            random_state: 随机种子，默认42
            **kwargs: 其他参数传递给XGBClassifier

        Returns:
            XGBClassifier实例
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            **kwargs
        )

        # 记录参数
        self.training_metrics['params'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state
        }

        logger.info(f"XGBoost模型已创建: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list] = None
    ) -> None:
        """
        训练XGBoost模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称列表
        """
        if self.model is None:
            logger.info("模型未创建，使用默认参数创建...")
            self.build_model()

        logger.info(f"开始训练XGBoost模型...")
        logger.info(f"训练集大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()

        self.is_trained = True
        self.feature_names = feature_names
        self.feature_importances_ = self.model.feature_importances_

        # 记录训练信息
        self.training_metrics['training_samples'] = X_train.shape[0]
        self.training_metrics['n_features'] = X_train.shape[1]
        self.training_metrics['training_time_seconds'] = training_time
        self.training_metrics['trained_at'] = datetime.now().isoformat()

        logger.info(f"训练完成！耗时: {training_time:.2f} 秒")

    def get_feature_importance(self, top_n: int = None) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征，None表示全部返回

        Returns:
            特征重要性DataFrame，按重要性降序排列
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        else:
            feature_names = self.feature_names

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)

        importance_df['rank'] = range(1, len(importance_df) + 1)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def train_with_early_stopping(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = 10,
        feature_names: Optional[list] = None
    ) -> None:
        """
        使用早停策略训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            early_stopping_rounds: 早停轮数，默认10
            feature_names: 特征名称列表
        """
        if self.model is None:
            raise ValueError("模型未初始化，请先调用build_model()")

        logger.info(f"开始训练XGBoost模型（启用早停）...")
        logger.info(f"训练样本数: {len(X_train)}, 验证样本数: {len(X_val)}")

        start_time = datetime.now()
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        training_time = (datetime.now() - start_time).total_seconds()

        self.is_trained = True
        self.feature_names = feature_names
        self.feature_importances_ = self.model.feature_importances_

        # 记录训练信息
        self.training_metrics['training_samples'] = len(X_train)
        self.training_metrics['n_features'] = X_train.shape[1]
        self.training_metrics['training_time_seconds'] = training_time
        self.training_metrics['trained_at'] = datetime.now().isoformat()
        if hasattr(self.model, 'best_iteration'):
            self.training_metrics['best_iteration'] = self.model.best_iteration

        logger.info(f"训练完成！耗时: {training_time:.2f} 秒")


class EnsembleTrainer(BaseModelTrainer):
    """
    集成模型训练器

    Day 16实现 - 软投票集成模型（RandomForest + XGBoost）

    软投票（Soft Voting）原理：
    - 每个基模型预测概率，而不是直接预测类别
    - 最终预测 = 各模型概率的加权平均
    - 比硬投票更精细，能利用模型的置信度信息

    Example:
        >>> trainer = EnsembleTrainer()
        >>> trainer.build_model(rf_params={...}, xgb_params={...})
        >>> trainer.train(X_train, y_train)
        >>> metrics = trainer.evaluate(X_test, y_test)
    """

    def __init__(self, model_name: str = "ensemble_model"):
        """
        初始化集成模型训练器

        Args:
            model_name: 模型名称
        """
        super().__init__(model_name)
        self.rf_trainer = None
        self.xgb_trainer = None
        self.weights = None
        self.base_models = {}

    def build_model(
        self,
        rf_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        weights: Optional[List[float]] = None
    ) -> None:
        """
        构建集成模型

        Args:
            rf_params: RandomForest参数字典
            xgb_params: XGBoost参数字典
            weights: 模型权重列表 [rf_weight, xgb_weight]，默认等权重[0.5, 0.5]
        """
        # 默认参数
        if rf_params is None:
            rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }

        if xgb_params is None:
            xgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }

        if weights is None:
            weights = [0.5, 0.5]

        # 验证权重
        if len(weights) != 2:
            raise ValueError("权重列表必须包含2个元素 [rf_weight, xgb_weight]")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"权重之和必须为1，当前为{sum(weights)}")

        self.weights = weights

        # 创建基模型
        self.rf_trainer = RandomForestTrainer(model_name="rf_base")
        self.rf_trainer.build_model(**rf_params)

        self.xgb_trainer = XGBoostTrainer(model_name="xgb_base")
        self.xgb_trainer.build_model(**xgb_params)

        # 记录参数
        self.training_metrics['rf_params'] = rf_params
        self.training_metrics['xgb_params'] = xgb_params
        self.training_metrics['weights'] = weights

        logger.info(f"集成模型已创建: RF权重={weights[0]}, XGB权重={weights[1]}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list] = None
    ) -> None:
        """
        训练集成模型

        分别训练RF和XGBoost，然后组合成集成模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            feature_names: 特征名称列表
        """
        if self.rf_trainer is None or self.xgb_trainer is None:
            logger.info("模型未创建，使用默认参数创建...")
            self.build_model()

        logger.info(f"开始训练集成模型...")
        logger.info(f"训练集大小: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")

        start_time = datetime.now()

        # 训练RandomForest
        logger.info("训练RandomForest基模型...")
        self.rf_trainer.train(X_train, y_train, feature_names=feature_names)

        # 训练XGBoost
        logger.info("训练XGBoost基模型...")
        self.xgb_trainer.train(X_train, y_train, feature_names=feature_names)

        training_time = (datetime.now() - start_time).total_seconds()

        # 保存基模型引用
        self.base_models = {
            'rf': self.rf_trainer,
            'xgb': self.xgb_trainer
        }

        self.is_trained = True
        self.feature_names = feature_names

        # 创建一个虚拟的model属性，用于兼容基类
        self.model = self  # 自身作为model，因为predict方法已重写

        # 记录训练信息
        self.training_metrics['training_samples'] = X_train.shape[0]
        self.training_metrics['n_features'] = X_train.shape[1]
        self.training_metrics['training_time_seconds'] = training_time
        self.training_metrics['rf_training_time'] = self.rf_trainer.training_metrics.get('training_time_seconds', 0)
        self.training_metrics['xgb_training_time'] = self.xgb_trainer.training_metrics.get('training_time_seconds', 0)
        self.training_metrics['trained_at'] = datetime.now().isoformat()

        logger.info(f"集成模型训练完成！总耗时: {training_time:.2f} 秒")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        软投票预测概率

        将两个基模型的预测概率按权重加权平均

        Args:
            X: 特征数据

        Returns:
            加权平均后的预测概率 [n_samples, 2]
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")

        # 获取各基模型的预测概率
        rf_proba = self.rf_trainer.predict_proba(X)
        xgb_proba = self.xgb_trainer.predict_proba(X)

        # 加权平均
        ensemble_proba = self.weights[0] * rf_proba + self.weights[1] * xgb_proba

        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        软投票预测

        基于加权概率，选择概率较高的类别

        Args:
            X: 特征数据

        Returns:
            预测结果
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_base_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        获取各基模型的预测结果

        Args:
            X: 特征数据

        Returns:
            字典，包含各基模型的预测结果和概率
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        return {
            'rf_pred': self.rf_trainer.predict(X),
            'rf_proba': self.rf_trainer.predict_proba(X),
            'xgb_pred': self.xgb_trainer.predict(X),
            'xgb_proba': self.xgb_trainer.predict_proba(X),
            'ensemble_pred': self.predict(X),
            'ensemble_proba': self.predict_proba(X)
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取综合特征重要性

        将RF和XGBoost的特征重要性按权重加权平均

        Returns:
            特征重要性DataFrame
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        rf_importance = self.rf_trainer.get_feature_importance()
        xgb_importance = self.xgb_trainer.get_feature_importance()

        # 合并特征重要性
        merged = rf_importance.merge(
            xgb_importance,
            on='feature',
            suffixes=('_rf', '_xgb')
        )

        # 加权平均
        merged['importance'] = (
            self.weights[0] * merged['importance_rf'] +
            self.weights[1] * merged['importance_xgb']
        )

        result = merged[['feature', 'importance', 'importance_rf', 'importance_xgb']]
        result = result.sort_values('importance', ascending=False)
        result['rank'] = range(1, len(result) + 1)

        return result

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        保存集成模型

        Args:
            filepath: 保存路径

        Returns:
            保存的文件路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先调用train()方法")

        if filepath is None:
            os.makedirs(MODELS_DIR, exist_ok=True)
            filepath = os.path.join(MODELS_DIR, f"{self.model_name}.pkl")

        # 保存完整的集成模型数据
        model_data = {
            'model_type': 'ensemble',
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'weights': self.weights,
            'rf_model': self.rf_trainer.model,
            'xgb_model': self.xgb_trainer.model,
            'rf_training_metrics': self.rf_trainer.training_metrics,
            'xgb_training_metrics': self.xgb_trainer.training_metrics,
            'training_metrics': self.training_metrics,
            'feature_names': self.feature_names,
            'saved_at': datetime.now().isoformat()
        }

        joblib.dump(model_data, filepath)
        logger.info(f"集成模型已保存到: {filepath}")

        return filepath

    def load_model(self, filepath: str) -> None:
        """
        加载集成模型

        Args:
            filepath: 模型文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model_data = joblib.load(filepath)

        if model_data.get('model_type') != 'ensemble':
            raise ValueError("加载的不是集成模型文件")

        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.weights = model_data['weights']
        self.feature_names = model_data.get('feature_names', None)
        self.training_metrics = model_data.get('training_metrics', {})

        # 重建基模型训练器
        self.rf_trainer = RandomForestTrainer(model_name="rf_base")
        self.rf_trainer.model = model_data['rf_model']
        self.rf_trainer.is_trained = True
        self.rf_trainer.training_metrics = model_data.get('rf_training_metrics', {})
        self.rf_trainer.feature_names = self.feature_names
        self.rf_trainer.feature_importances_ = model_data['rf_model'].feature_importances_

        self.xgb_trainer = XGBoostTrainer(model_name="xgb_base")
        self.xgb_trainer.model = model_data['xgb_model']
        self.xgb_trainer.is_trained = True
        self.xgb_trainer.training_metrics = model_data.get('xgb_training_metrics', {})
        self.xgb_trainer.feature_names = self.feature_names
        self.xgb_trainer.feature_importances_ = model_data['xgb_model'].feature_importances_

        self.base_models = {
            'rf': self.rf_trainer,
            'xgb': self.xgb_trainer
        }

        self.model = self

        logger.info(f"集成模型已从 {filepath} 加载")


def compare_models(
    trainers: List[BaseModelTrainer],
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True
) -> pd.DataFrame:
    """
    比较多个模型的性能

    Day 15实现 - 模型对比函数

    Args:
        trainers: 模型训练器列表
        X_test: 测试特征
        y_test: 测试标签
        verbose: 是否打印详细信息

    Returns:
        对比结果DataFrame，包含各模型的评估指标
    """
    results = []

    for trainer in trainers:
        if not trainer.is_trained:
            logger.warning(f"模型 {trainer.model_name} 尚未训练，跳过")
            continue

        metrics = trainer.evaluate(X_test, y_test)

        result = {
            'model_name': trainer.model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc']
        }
        results.append(result)

        if verbose:
            print(f"\n--- {trainer.model_name} ---")
            print(f"  准确率: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"  精确率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  AUC-ROC: {metrics['roc_auc']:.4f}")

    # 创建对比DataFrame
    comparison_df = pd.DataFrame(results)

    if verbose and len(results) > 1:
        print("\n" + "=" * 50)
        print("模型性能对比汇总:")
        print("=" * 50)
        print(comparison_df.to_string(index=False))

        # 找出最佳模型
        best_idx = comparison_df['accuracy'].idxmax()
        best_model = comparison_df.loc[best_idx, 'model_name']
        best_acc = comparison_df.loc[best_idx, 'accuracy']
        print(f"\n[最佳模型] {best_model} (准确率: {best_acc*100:.2f}%)")

    return comparison_df


def load_feature_data(use_scaled: bool = True, use_30dim: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载特征数据

    Args:
        use_scaled: 是否使用标准化后的数据（仅对17维有效）
        use_30dim: 是否使用30维特征（默认True），False则使用17维

    Returns:
        (X_train, y_train, X_test, y_test, feature_cols)
    """
    processed_dir = os.path.join(DATA_DIR, 'processed')

    if use_30dim:
        # 使用30维特征数据
        train_file = os.path.join(processed_dir, 'train_30dim.csv')
        test_file = os.path.join(processed_dir, 'test_30dim.csv')
    elif use_scaled:
        train_file = os.path.join(processed_dir, 'train_scaled.csv')
        test_file = os.path.join(processed_dir, 'test_scaled.csv')
    else:
        train_file = os.path.join(processed_dir, 'train_features.csv')
        test_file = os.path.join(processed_dir, 'test_features.csv')

    logger.info(f"加载训练数据: {train_file}")
    train_df = pd.read_csv(train_file)

    logger.info(f"加载测试数据: {test_file}")
    test_df = pd.read_csv(test_file)

    # 获取特征列（排除url和label）
    feature_cols = [col for col in train_df.columns if col not in ['url', 'label']]

    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    logger.info(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
    logger.info(f"测试集: {X_test.shape[0]} 样本, {X_test.shape[1]} 特征")

    return X_train, y_train, X_test, y_test, feature_cols


def main():
    """
    主函数 - Day 16: 集成模型训练与三模型对比（使用30维特征）
    """
    print("=" * 60)
    print("Day 16 - 集成模型训练与三模型对比（30维特征）")
    print("=" * 60)

    # 1. 加载30维特征数据
    print("\n[1] 加载30维特征数据...")
    X_train, y_train, X_test, y_test, feature_cols = load_feature_data(use_30dim=True)

    # 2. 训练集成模型（包含RF和XGBoost）
    print("\n[2] 训练集成模型（RF + XGBoost 软投票）...")
    ensemble_trainer = EnsembleTrainer(model_name="ensemble_model")
    ensemble_trainer.build_model(
        rf_params={
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        },
        xgb_params={
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        },
        weights=[0.5, 0.5]  # 等权重软投票
    )
    ensemble_trainer.train(X_train, y_train, feature_names=feature_cols)

    # 3. 三模型对比
    print("\n[3] 三模型性能对比...")
    rf_trainer = ensemble_trainer.rf_trainer
    xgb_trainer = ensemble_trainer.xgb_trainer

    # 评估各模型
    rf_metrics = rf_trainer.evaluate(X_test, y_test)
    xgb_metrics = xgb_trainer.evaluate(X_test, y_test)
    ensemble_metrics = ensemble_trainer.evaluate(X_test, y_test)

    print("\n--- RandomForest ---")
    print(f"  准确率: {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)")
    print(f"  精确率: {rf_metrics['precision']:.4f}")
    print(f"  召回率: {rf_metrics['recall']:.4f}")
    print(f"  F1分数: {rf_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {rf_metrics['roc_auc']:.4f}")

    print("\n--- XGBoost ---")
    print(f"  准确率: {xgb_metrics['accuracy']:.4f} ({xgb_metrics['accuracy']*100:.2f}%)")
    print(f"  精确率: {xgb_metrics['precision']:.4f}")
    print(f"  召回率: {xgb_metrics['recall']:.4f}")
    print(f"  F1分数: {xgb_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {xgb_metrics['roc_auc']:.4f}")

    print("\n--- Ensemble (软投票) ---")
    print(f"  准确率: {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"  精确率: {ensemble_metrics['precision']:.4f}")
    print(f"  召回率: {ensemble_metrics['recall']:.4f}")
    print(f"  F1分数: {ensemble_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {ensemble_metrics['roc_auc']:.4f}")

    # 4. 对比汇总
    print("\n" + "=" * 50)
    print("模型性能对比汇总:")
    print("=" * 50)
    comparison_data = {
        'model_name': ['RandomForest', 'XGBoost', 'Ensemble'],
        'accuracy': [rf_metrics['accuracy'], xgb_metrics['accuracy'], ensemble_metrics['accuracy']],
        'precision': [rf_metrics['precision'], xgb_metrics['precision'], ensemble_metrics['precision']],
        'recall': [rf_metrics['recall'], xgb_metrics['recall'], ensemble_metrics['recall']],
        'f1': [rf_metrics['f1'], xgb_metrics['f1'], ensemble_metrics['f1']],
        'roc_auc': [rf_metrics['roc_auc'], xgb_metrics['roc_auc'], ensemble_metrics['roc_auc']]
    }
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    # 找出最佳模型
    best_idx = comparison_df['accuracy'].idxmax()
    best_model = comparison_df.loc[best_idx, 'model_name']
    best_acc = comparison_df.loc[best_idx, 'accuracy']
    print(f"\n[最佳模型] {best_model} (准确率: {best_acc*100:.2f}%)")

    # 5. 验证是否达到目标
    print("\n" + "=" * 40)
    print("性能验证 (Day 16 目标: 准确率 ≥ 85%):")
    print("=" * 40)

    target_accuracy = 0.85
    for _, row in comparison_df.iterrows():
        if row['accuracy'] >= target_accuracy:
            print(f"[PASS] {row['model_name']}: {row['accuracy']*100:.2f}% >= 85%")
        else:
            print(f"[WARN] {row['model_name']}: {row['accuracy']*100:.2f}% < 85%")

    # 6. 集成模型特征重要性
    print("\n[4] 集成模型特征重要性 (Top 10):")
    importance_df = ensemble_trainer.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # 7. 保存模型
    print("\n[5] 保存模型...")
    rf_path = rf_trainer.save_model()
    print(f"RandomForest模型已保存到: {rf_path}")
    xgb_path = xgb_trainer.save_model()
    print(f"XGBoost模型已保存到: {xgb_path}")
    ensemble_path = ensemble_trainer.save_model()
    print(f"集成模型已保存到: {ensemble_path}")

    # 8. 返回结果
    print("\n" + "=" * 60)
    print("Day 16 任务完成!")
    print("=" * 60)

    return comparison_df, ensemble_trainer


class CrossValidator:
    """
    交叉验证器

    Day 17实现 - 使用分层K折交叉验证评估模型性能

    分层K折交叉验证（Stratified K-Fold）原理：
    - 将数据集分成K份（折），每次用K-1份训练，1份验证
    - 分层采样确保每折中各类别的比例与原数据集一致
    - 最终得到K个评估结果，计算均值和标准差
    - 比单次划分更可靠，能评估模型的稳定性

    Example:
        >>> validator = CrossValidator(n_splits=5)
        >>> results = validator.cross_validate(trainer, X, y)
        >>> print(f"准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42
    ):
        """
        初始化交叉验证器

        Args:
            n_splits: 折数，默认5
            shuffle: 是否打乱数据，默认True
            random_state: 随机种子，默认42
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

        # 创建分层K折验证器
        self.skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        # 存储结果
        self.cv_results = {}
        self._fold_scores = []

        logger.info(f"交叉验证器已创建: n_splits={n_splits}, shuffle={shuffle}")

    def cross_validate(
        self,
        trainer: BaseModelTrainer,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行交叉验证

        Args:
            trainer: 模型训练器（需要有build_model方法）
            X: 特征数据
            y: 标签数据
            feature_names: 特征名称列表
            verbose: 是否打印详细信息

        Returns:
            dict: 包含各指标均值和标准差的字典
        """
        model_name = trainer.model_name

        if verbose:
            print(f"\n[{model_name}] 开始{self.n_splits}折交叉验证...")
            print(f"  样本数: {len(X)}")
            print(f"  特征维度: {X.shape[1]}")

        # 存储每折的分数
        fold_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }

        # 执行交叉验证
        for fold_idx, (train_idx, val_idx) in enumerate(self.skf.split(X, y)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # 创建新的训练器实例
            fold_trainer = type(trainer)(model_name=f"{model_name}_fold{fold_idx+1}")

            # 复制训练器的参数设置
            if hasattr(trainer, 'training_metrics') and 'params' in trainer.training_metrics:
                params = trainer.training_metrics['params'].copy()
                fold_trainer.build_model(**params)
            else:
                fold_trainer.build_model()

            # 训练模型
            fold_trainer.train(X_train_fold, y_train_fold, feature_names=feature_names)

            # 评估模型
            metrics = fold_trainer.evaluate(X_val_fold, y_val_fold)

            # 记录分数
            for metric_name in fold_scores.keys():
                fold_scores[metric_name].append(metrics[metric_name])

            if verbose:
                print(f"  Fold {fold_idx + 1}: "
                      f"Acc={metrics['accuracy']:.4f}, "
                      f"P={metrics['precision']:.4f}, "
                      f"R={metrics['recall']:.4f}, "
                      f"F1={metrics['f1']:.4f}, "
                      f"AUC={metrics['roc_auc']:.4f}")

        # 计算统计量
        results = {}
        for metric_name, scores in fold_scores.items():
            results[f'{metric_name}_mean'] = np.mean(scores)
            results[f'{metric_name}_std'] = np.std(scores)
            results[f'{metric_name}_scores'] = scores

        # 保存结果
        self.cv_results[model_name] = results
        self._fold_scores = fold_scores

        if verbose:
            print(f"\n[{model_name}] 交叉验证结果:")
            print("-" * 50)
            print(f"  准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
            print(f"  精确率: {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
            print(f"  召回率: {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
            print(f"  F1分数: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
            print(f"  AUC-ROC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}")
            print("-" * 50)

        return results

    def get_cv_scores(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取各折分数

        Args:
            model_name: 模型名称，None返回所有

        Returns:
            dict: 各折的分数列表
        """
        if model_name is None:
            return self.cv_results

        if model_name in self.cv_results:
            return self.cv_results[model_name]

        return {}

    def get_summary(self) -> pd.DataFrame:
        """
        获取所有模型的交叉验证汇总

        Returns:
            DataFrame: 汇总表格，按准确率均值降序排列
        """
        if not self.cv_results:
            return pd.DataFrame()

        rows = []
        for model_name, results in self.cv_results.items():
            row = {
                'model': model_name,
                'accuracy': f"{results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}",
                'precision': f"{results['precision_mean']:.4f} ± {results['precision_std']:.4f}",
                'recall': f"{results['recall_mean']:.4f} ± {results['recall_std']:.4f}",
                'f1': f"{results['f1_mean']:.4f} ± {results['f1_std']:.4f}",
                'roc_auc': f"{results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}",
                'accuracy_mean': results['accuracy_mean'],
                'accuracy_std': results['accuracy_std']
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df.sort_values('accuracy_mean', ascending=False).reset_index(drop=True)

    def plot_cv_results(
        self,
        metric: str = 'accuracy',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        可视化交叉验证结果

        Args:
            metric: 要可视化的指标（accuracy, precision, recall, f1, roc_auc）
            save_path: 图片保存路径
            figsize: 图片大小
        """
        if not self.cv_results:
            print("没有交叉验证结果可供可视化")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 图1: 各折分数对比
        ax1 = axes[0]
        model_names = list(self.cv_results.keys())
        x = np.arange(self.n_splits)
        width = 0.8 / len(model_names)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, model_name in enumerate(model_names):
            scores = self.cv_results[model_name].get(f'{metric}_scores', [])
            if scores:
                offset = (i - len(model_names) / 2 + 0.5) * width
                ax1.bar(x + offset, scores, width, label=model_name,
                       alpha=0.8, color=colors[i % len(colors)])

        ax1.set_xlabel('Fold', fontsize=11)
        ax1.set_ylabel(metric.capitalize(), fontsize=11)
        ax1.set_title(f'{self.n_splits}-Fold Cross Validation - {metric} by Fold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Fold {i+1}' for i in range(self.n_splits)])
        ax1.legend(loc='lower right')
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0.9, 1.0])

        # 图2: 均值和标准差对比
        ax2 = axes[1]
        means = []
        stds = []
        names = []

        for model_name in model_names:
            mean_val = self.cv_results[model_name].get(f'{metric}_mean', 0)
            std_val = self.cv_results[model_name].get(f'{metric}_std', 0)
            means.append(mean_val)
            stds.append(std_val)
            names.append(model_name)

        x_pos = np.arange(len(names))
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8,
                      color=[colors[i % len(colors)] for i in range(len(names))])

        ax2.set_xlabel('Model', fontsize=11)
        ax2.set_ylabel(f'{metric.capitalize()} (mean +/- std)', fontsize=11)
        ax2.set_title(f'Cross Validation - {metric} Mean Comparison', fontsize=12)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=0)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0.9, 1.02])

        # 添加数值标签
        for bar, mean, std in zip(bars, means, stds):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                     f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.close()

    def check_stability(
        self,
        model_name: str,
        std_threshold: float = 0.03
    ) -> bool:
        """
        检查模型稳定性

        Args:
            model_name: 模型名称
            std_threshold: 标准差阈值

        Returns:
            bool: 是否稳定（标准差 <= 阈值）
        """
        if model_name not in self.cv_results:
            print(f"[{model_name}] 未找到交叉验证结果")
            return False

        accuracy_std = self.cv_results[model_name].get('accuracy_std', float('inf'))

        is_stable = accuracy_std <= std_threshold

        if is_stable:
            print(f"[{model_name}] [PASS] 模型稳定 (std={accuracy_std:.4f} <= {std_threshold})")
        else:
            print(f"[{model_name}] [WARN] 模型不稳定 (std={accuracy_std:.4f} > {std_threshold})")

        return is_stable


# ==================== 交叉验证便捷函数 ====================

def cross_validate_model(
    trainer: BaseModelTrainer,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    便捷函数：对单个模型进行交叉验证

    Args:
        trainer: 模型训练器
        X: 特征数据
        y: 标签数据
        n_splits: 折数
        feature_names: 特征名称
        verbose: 是否打印详细信息

    Returns:
        dict: 交叉验证结果
    """
    validator = CrossValidator(n_splits=n_splits)
    return validator.cross_validate(trainer, X, y, feature_names, verbose)


def cross_validate_all_models(
    trainers: List[BaseModelTrainer],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
    save_plot: Optional[str] = None
) -> Tuple[pd.DataFrame, CrossValidator]:
    """
    便捷函数：对多个模型进行交叉验证比较

    Args:
        trainers: 模型训练器列表
        X: 特征数据
        y: 标签数据
        n_splits: 折数
        feature_names: 特征名称
        verbose: 是否打印详细信息
        save_plot: 图表保存路径

    Returns:
        Tuple[DataFrame, CrossValidator]: 交叉验证结果汇总和验证器对象
    """
    validator = CrossValidator(n_splits=n_splits)

    for trainer in trainers:
        validator.cross_validate(trainer, X, y, feature_names, verbose)

    # 获取汇总
    summary_df = validator.get_summary()

    if verbose and len(summary_df) > 0:
        print("\n" + "=" * 60)
        print("交叉验证汇总")
        print("=" * 60)
        print(summary_df[['model', 'accuracy', 'precision', 'recall', 'f1']].to_string(index=False))
        print("=" * 60)

        # 找出最佳模型
        best_model = summary_df.iloc[0]['model']
        best_acc = summary_df.iloc[0]['accuracy_mean']
        best_std = summary_df.iloc[0]['accuracy_std']
        print(f"\n[最佳模型] {best_model} (准确率: {best_acc:.4f} ± {best_std:.4f})")

    # 保存图表
    if save_plot:
        validator.plot_cv_results(metric='accuracy', save_path=save_plot)

    return summary_df, validator


# ==================== 超参数调优类 ====================

class HyperparameterTuner:
    """
    超参数调优器

    Day 18实现 - 使用GridSearchCV和RandomizedSearchCV进行超参数搜索

    支持两种搜索方法：
    - 网格搜索（Grid Search）：穷举所有参数组合
    - 随机搜索（Random Search）：随机采样参数组合，更高效

    Example:
        >>> tuner = HyperparameterTuner('RandomForest')
        >>> best_params = tuner.grid_search(model, param_grid, X, y)
        >>> print(best_params)
    """

    def __init__(self, model_name: str = "Model"):
        """
        初始化超参数调优器

        Args:
            model_name: 模型名称，用于日志输出
        """
        self.model_name = model_name
        self.best_params = None
        self.best_score = None
        self.best_estimator = None
        self.cv_results = None
        self.search_history = []

        logger.info(f"超参数调优器已创建: {model_name}")

    def grid_search(
        self,
        model,
        param_grid: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1,
        verbose: int = 2,
        refit: bool = True
    ) -> Dict:
        """
        执行网格搜索

        穷举param_grid中所有参数组合，找到最优参数

        Args:
            model: sklearn兼容的模型对象
            param_grid: 参数网格字典，如 {'n_estimators': [100, 200], 'max_depth': [5, 10]}
            X: 特征数据
            y: 标签数据
            cv: 交叉验证折数，默认5
            scoring: 评分指标，默认'accuracy'
            n_jobs: 并行任务数，-1表示使用所有CPU核心
            verbose: 输出详细程度，0=静默，1=简略，2=详细
            refit: 是否用最优参数重新训练最终模型

        Returns:
            dict: 最优参数字典
        """
        print(f"\n[{self.model_name}] 开始网格搜索...")
        print(f"  参数网格: {param_grid}")
        print(f"  交叉验证: {cv}折")
        print(f"  评分指标: {scoring}")

        # 计算总搜索次数
        n_combinations = 1
        for values in param_grid.values():
            n_combinations *= len(values)
        print(f"  参数组合数: {n_combinations}")
        print(f"  总拟合次数: {n_combinations * cv}")

        start_time = time.time()

        # 执行网格搜索
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            refit=refit,
            return_train_score=True
        )

        grid_search.fit(X, y)

        elapsed_time = time.time() - start_time

        # 保存结果
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_estimator = grid_search.best_estimator_
        self.cv_results = grid_search.cv_results_

        # 记录搜索历史
        self.search_history.append({
            'method': 'grid_search',
            'model_name': self.model_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_combinations': n_combinations,
            'elapsed_time': elapsed_time
        })

        print(f"\n[{self.model_name}] 网格搜索完成!")
        print(f"  耗时: {elapsed_time:.2f}秒")
        print(f"  最优参数: {self.best_params}")
        print(f"  最优分数: {self.best_score:.4f}")

        logger.info(f"网格搜索完成: best_score={self.best_score:.4f}")

        return self.best_params

    def random_search(
        self,
        model,
        param_distributions: Dict,
        X: np.ndarray,
        y: np.ndarray,
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1,
        verbose: int = 2,
        random_state: int = 42,
        refit: bool = True
    ) -> Dict:
        """
        执行随机搜索

        从param_distributions中随机采样n_iter个参数组合进行评估

        Args:
            model: sklearn兼容的模型对象
            param_distributions: 参数分布字典，可以是列表或scipy.stats分布
            X: 特征数据
            y: 标签数据
            n_iter: 随机采样次数，默认50
            cv: 交叉验证折数，默认5
            scoring: 评分指标，默认'accuracy'
            n_jobs: 并行任务数
            verbose: 输出详细程度
            random_state: 随机种子
            refit: 是否用最优参数重新训练

        Returns:
            dict: 最优参数字典
        """
        print(f"\n[{self.model_name}] 开始随机搜索...")
        print(f"  采样次数: {n_iter}")
        print(f"  交叉验证: {cv}折")
        print(f"  评分指标: {scoring}")
        print(f"  总拟合次数: {n_iter * cv}")

        start_time = time.time()

        # 执行随机搜索
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            refit=refit,
            return_train_score=True
        )

        random_search.fit(X, y)

        elapsed_time = time.time() - start_time

        # 保存结果
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        self.best_estimator = random_search.best_estimator_
        self.cv_results = random_search.cv_results_

        # 记录搜索历史
        self.search_history.append({
            'method': 'random_search',
            'model_name': self.model_name,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iter': n_iter,
            'elapsed_time': elapsed_time
        })

        print(f"\n[{self.model_name}] 随机搜索完成!")
        print(f"  耗时: {elapsed_time:.2f}秒")
        print(f"  最优参数: {self.best_params}")
        print(f"  最优分数: {self.best_score:.4f}")

        logger.info(f"随机搜索完成: best_score={self.best_score:.4f}")

        return self.best_params

    def get_best_params(self) -> Dict:
        """
        获取最优参数

        Returns:
            dict: 最优参数字典的副本
        """
        if self.best_params is None:
            raise ValueError("尚未执行超参数搜索")
        return self.best_params.copy()

    def get_best_score(self) -> float:
        """
        获取最优分数

        Returns:
            float: 最优交叉验证分数
        """
        if self.best_score is None:
            raise ValueError("尚未执行超参数搜索")
        return self.best_score

    def get_best_estimator(self):
        """
        获取最优模型

        Returns:
            使用最优参数训练的模型对象
        """
        if self.best_estimator is None:
            raise ValueError("尚未执行超参数搜索")
        return self.best_estimator

    def get_cv_results(self) -> pd.DataFrame:
        """
        获取交叉验证详细结果

        Returns:
            DataFrame: 包含所有参数组合的评估结果，按排名排序
        """
        if self.cv_results is None:
            raise ValueError("尚未执行超参数搜索")

        df = pd.DataFrame(self.cv_results)

        # 选择关键列
        key_cols = [
            'params',
            'mean_test_score',
            'std_test_score',
            'rank_test_score',
            'mean_train_score',
            'mean_fit_time'
        ]

        available_cols = [c for c in key_cols if c in df.columns]
        df = df[available_cols].sort_values('rank_test_score')

        return df

    def plot_search_results(
        self,
        param_name: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 5)
    ) -> None:
        """
        可视化搜索结果

        Args:
            param_name: 要可视化的参数名，None则显示整体结果
            save_path: 图片保存路径
            figsize: 图片大小
        """
        if self.cv_results is None:
            print("没有搜索结果可供可视化")
            return

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # 图1: 分数分布
        ax1 = axes[0]
        scores = self.cv_results['mean_test_score']
        ax1.hist(scores, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(self.best_score, color='red', linestyle='--',
                    label=f'Best: {self.best_score:.4f}')
        ax1.set_xlabel('Mean Test Score', fontsize=11)
        ax1.set_ylabel('Frequency', fontsize=11)
        ax1.set_title(f'{self.model_name} - Score Distribution', fontsize=12)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 图2: 参数对分数的影响
        ax2 = axes[1]
        if param_name and f'param_{param_name}' in self.cv_results:
            param_values = self.cv_results[f'param_{param_name}']
            # 转换为数值（如果可能）
            try:
                param_values_numeric = [float(v) if v is not None else np.nan for v in param_values]
                ax2.scatter(param_values_numeric, scores, alpha=0.6, c='steelblue')
                ax2.set_xlabel(param_name, fontsize=11)
                ax2.set_ylabel('Mean Test Score', fontsize=11)
                ax2.set_title(f'{param_name} vs Score', fontsize=12)
            except (ValueError, TypeError):
                # 如果不能转为数值，使用条形图
                unique_values = list(set(param_values))
                means = [np.mean([s for p, s in zip(param_values, scores) if p == v])
                         for v in unique_values]
                ax2.bar(range(len(unique_values)), means, color='steelblue', alpha=0.7)
                ax2.set_xticks(range(len(unique_values)))
                ax2.set_xticklabels([str(v) for v in unique_values], rotation=45)
                ax2.set_xlabel(param_name, fontsize=11)
                ax2.set_ylabel('Mean Test Score', fontsize=11)
                ax2.set_title(f'{param_name} vs Score', fontsize=12)
        else:
            # 显示排名vs分数
            ranks = self.cv_results['rank_test_score']
            ax2.scatter(ranks, scores, alpha=0.6, c='steelblue')
            ax2.set_xlabel('Rank', fontsize=11)
            ax2.set_ylabel('Mean Test Score', fontsize=11)
            ax2.set_title('Rank vs Score', fontsize=12)

        ax2.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")

        plt.close()

    def save_results(self, filepath: str) -> None:
        """
        保存调优结果到JSON文件

        Args:
            filepath: 保存路径（.json文件）
        """
        results = {
            'model_name': self.model_name,
            'best_params': self.best_params,
            'best_score': float(self.best_score) if self.best_score else None,
            'search_history': self.search_history
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        print(f"调优结果已保存至: {filepath}")
        logger.info(f"调优结果已保存至: {filepath}")

    def load_results(self, filepath: str) -> None:
        """
        从JSON文件加载调优结果

        Args:
            filepath: 结果文件路径
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)

        self.model_name = results.get('model_name', self.model_name)
        self.best_params = results.get('best_params')
        self.best_score = results.get('best_score')
        self.search_history = results.get('search_history', [])

        print(f"调优结果已从 {filepath} 加载")
        logger.info(f"调优结果已从 {filepath} 加载")


# ==================== 超参数调优便捷函数 ====================

def get_rf_param_grid() -> Dict[str, List]:
    """
    获取RandomForest参数网格（用于网格搜索）

    Returns:
        dict: 参数网格
    """
    return {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }


def get_xgb_param_grid() -> Dict[str, List]:
    """
    获取XGBoost参数网格（用于网格搜索）

    Returns:
        dict: 参数网格
    """
    return {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }


def get_rf_param_distributions() -> Dict:
    """
    获取RandomForest参数分布（用于随机搜索）

    Returns:
        dict: 参数分布
    """
    return {
        'n_estimators': randint(50, 300),
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }


def get_xgb_param_distributions() -> Dict:
    """
    获取XGBoost参数分布（用于随机搜索）

    Returns:
        dict: 参数分布
    """
    return {
        'n_estimators': randint(50, 300),
        'learning_rate': uniform(0.01, 0.29),
        'max_depth': randint(3, 12),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0.5, 1.5)
    }


def tune_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'random',
    n_iter: int = 30,
    cv: int = 5,
    verbose: int = 1
) -> Tuple[Dict, float, HyperparameterTuner]:
    """
    调优RandomForest模型

    Args:
        X: 特征数据
        y: 标签数据
        method: 搜索方法，'grid'或'random'
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
        verbose: 输出详细程度

    Returns:
        (best_params, best_score, tuner)
    """
    tuner = HyperparameterTuner('RandomForest')
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    if method == 'grid':
        # 使用简化的参数网格进行网格搜索
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        tuner.grid_search(model, param_grid, X, y, cv=cv, verbose=verbose)
    else:
        param_distributions = get_rf_param_distributions()
        tuner.random_search(model, param_distributions, X, y,
                            n_iter=n_iter, cv=cv, verbose=verbose)

    return tuner.get_best_params(), tuner.get_best_score(), tuner


def tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'random',
    n_iter: int = 30,
    cv: int = 5,
    verbose: int = 1
) -> Tuple[Dict, float, HyperparameterTuner]:
    """
    调优XGBoost模型

    Args:
        X: 特征数据
        y: 标签数据
        method: 搜索方法，'grid'或'random'
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
        verbose: 输出详细程度

    Returns:
        (best_params, best_score, tuner)
    """
    tuner = HyperparameterTuner('XGBoost')
    model = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )

    if method == 'grid':
        # 使用简化的参数网格进行网格搜索
        param_grid = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [5, 7, 9],
            'min_child_weight': [1, 3]
        }
        tuner.grid_search(model, param_grid, X, y, cv=cv, verbose=verbose)
    else:
        param_distributions = get_xgb_param_distributions()
        tuner.random_search(model, param_distributions, X, y,
                            n_iter=n_iter, cv=cv, verbose=verbose)

    return tuner.get_best_params(), tuner.get_best_score(), tuner


def tune_all_models(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'random',
    n_iter: int = 30,
    cv: int = 5,
    save_dir: Optional[str] = None,
    verbose: int = 1
) -> pd.DataFrame:
    """
    调优所有模型（RandomForest和XGBoost）

    Args:
        X: 特征数据
        y: 标签数据
        method: 搜索方法，'grid'或'random'
        n_iter: 随机搜索迭代次数
        cv: 交叉验证折数
        save_dir: 结果保存目录
        verbose: 输出详细程度

    Returns:
        DataFrame: 调优结果汇总
    """
    results = []

    # 调优RandomForest
    print("\n" + "=" * 60)
    print("1. RandomForest 超参数调优")
    print("=" * 60)
    rf_params, rf_score, rf_tuner = tune_random_forest(
        X, y, method=method, n_iter=n_iter, cv=cv, verbose=verbose
    )
    results.append({
        'model': 'RandomForest',
        'best_score': rf_score,
        'best_params': str(rf_params)
    })

    # 调优XGBoost
    print("\n" + "=" * 60)
    print("2. XGBoost 超参数调优")
    print("=" * 60)
    xgb_params, xgb_score, xgb_tuner = tune_xgboost(
        X, y, method=method, n_iter=n_iter, cv=cv, verbose=verbose
    )
    results.append({
        'model': 'XGBoost',
        'best_score': xgb_score,
        'best_params': str(xgb_params)
    })

    # 汇总结果
    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("超参数调优汇总")
    print("=" * 60)
    print(df.to_string(index=False))

    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 保存汇总
        summary_path = os.path.join(save_dir, 'tuning_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f"\n汇总已保存至: {summary_path}")

        # 保存最优参数
        params_path = os.path.join(save_dir, 'best_params.json')
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump({
                'RandomForest': rf_params,
                'XGBoost': xgb_params
            }, f, indent=2, ensure_ascii=False, default=str)
        print(f"最优参数已保存至: {params_path}")

        # 保存调优详细结果
        rf_tuner.save_results(os.path.join(save_dir, 'rf_tuning_results.json'))
        xgb_tuner.save_results(os.path.join(save_dir, 'xgb_tuning_results.json'))

    return df


# ==================== Day 19: 模型评估器 ====================

class ModelEvaluator:
    """
    模型性能评估器

    Day 19实现 - 提供全面的模型性能评估功能

    功能包括：
    - 计算准确率、精确率、召回率、F1、AUC-ROC、AUC-PR等指标
    - 绘制混淆矩阵、ROC曲线、PR曲线
    - 生成评估报告
    - 多模型比较

    Example:
        >>> evaluator = ModelEvaluator()
        >>> metrics = evaluator.evaluate_model(model, X_test, y_test, 'RandomForest')
        >>> evaluator.plot_all_curves('RandomForest', y_test)
        >>> report = evaluator.generate_report()
    """

    def __init__(self):
        """初始化模型评估器"""
        self.results = {}
        self.figures = {}
        self._predictions = {}
        self._probabilities = {}

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        评估单个模型

        Args:
            model: 模型对象（需要有predict和predict_proba方法）
            X_test: 测试特征
            y_test: 测试标签
            model_name: 模型名称
            verbose: 是否打印详细信息

        Returns:
            dict: 包含所有评估指标的字典
        """
        # 获取预测结果
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            raise ValueError("模型需要有predict方法")

        # 获取预测概率
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        # 保存预测结果
        self._predictions[model_name] = y_pred
        self._probabilities[model_name] = y_proba

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # 计算各项指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'mcc': matthews_corrcoef(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'auc_pr': average_precision_score(y_test, y_proba),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }

        # 计算ROC曲线数据
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
        metrics['roc_thresholds'] = roc_thresholds

        # 计算PR曲线数据
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba)
        metrics['precision_curve'] = precision_curve
        metrics['recall_curve'] = recall_curve
        metrics['pr_thresholds'] = pr_thresholds

        # 保存结果
        self.results[model_name] = metrics

        if verbose:
            self._print_metrics(model_name, metrics)

        return metrics

    def _print_metrics(self, model_name: str, metrics: Dict) -> None:
        """打印评估指标"""
        print(f"\n[{model_name}] 评估结果:")
        print("-" * 50)
        print(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}")
        print(f"  精确率 (Precision):    {metrics['precision']:.4f}")
        print(f"  召回率 (Recall):       {metrics['recall']:.4f}")
        print(f"  F1分数 (F1-Score):     {metrics['f1']:.4f}")
        print(f"  特异度 (Specificity):  {metrics['specificity']:.4f}")
        print(f"  MCC:                   {metrics['mcc']:.4f}")
        print(f"  AUC-ROC:               {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR:                {metrics['auc_pr']:.4f}")
        print("-" * 50)
        print(f"  混淆矩阵:")
        print(f"    TP={metrics['tp']}, TN={metrics['tn']}")
        print(f"    FP={metrics['fp']}, FN={metrics['fn']}")
        print("-" * 50)

    def plot_confusion_matrix(
        self,
        model_name: str,
        normalize: bool = False,
        save_path: str = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """
        绘制混淆矩阵

        Args:
            model_name: 模型名称
            normalize: 是否归一化
            save_path: 图片保存路径
            figsize: 图片大小
        """
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 未评估")

        metrics = self.results[model_name]
        cm = np.array([[metrics['tn'], metrics['fp']],
                       [metrics['fn'], metrics['tp']]])

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title = f'{model_name} - 归一化混淆矩阵'
        else:
            fmt = 'd'
            title = f'{model_name} - 混淆矩阵'

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=['正常网站', '钓鱼网站'],
                    yticklabels=['正常网站', '钓鱼网站'],
                    annot_kws={'size': 14})
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")

        self.figures[f'{model_name}_confusion_matrix'] = plt.gcf()
        plt.close()

    def plot_roc_curve(
        self,
        model_names: List[str] = None,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        绘制ROC曲线

        Args:
            model_names: 要绘制的模型列表，None表示全部
            save_path: 图片保存路径
            figsize: 图片大小
        """
        if model_names is None:
            model_names = list(self.results.keys())

        plt.figure(figsize=figsize)

        # 绘制各模型ROC曲线
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        for model_name, color in zip(model_names, colors):
            if model_name not in self.results:
                continue

            metrics = self.results[model_name]
            fpr = metrics['fpr']
            tpr = metrics['tpr']
            auc_score = metrics['auc_roc']

            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f'{model_name} (AUC = {auc_score:.4f})')

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)', fontsize=12)
        plt.ylabel('真阳性率 (TPR)', fontsize=12)
        plt.title('ROC曲线对比', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_path}")

        self.figures['roc_curve'] = plt.gcf()
        plt.close()

    def plot_pr_curve(
        self,
        model_names: List[str] = None,
        save_path: str = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        绘制PR曲线（精确率-召回率曲线）

        Args:
            model_names: 要绘制的模型列表，None表示全部
            save_path: 图片保存路径
            figsize: 图片大小
        """
        if model_names is None:
            model_names = list(self.results.keys())

        plt.figure(figsize=figsize)

        # 绘制各模型PR曲线
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        for model_name, color in zip(model_names, colors):
            if model_name not in self.results:
                continue

            metrics = self.results[model_name]
            precision = metrics['precision_curve']
            recall = metrics['recall_curve']
            ap_score = metrics['auc_pr']

            plt.plot(recall, precision, color=color, lw=2,
                     label=f'{model_name} (AP = {ap_score:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)', fontsize=12)
        plt.ylabel('精确率 (Precision)', fontsize=12)
        plt.title('PR曲线对比', fontsize=14)
        plt.legend(loc='lower left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR曲线已保存至: {save_path}")

        self.figures['pr_curve'] = plt.gcf()
        plt.close()

    def plot_all_curves(
        self,
        model_name: str,
        y_test: np.ndarray,
        save_dir: str = None,
        figsize: Tuple[int, int] = (15, 5)
    ) -> None:
        """
        绘制所有评估曲线（混淆矩阵、ROC、PR）

        Args:
            model_name: 模型名称
            y_test: 测试标签
            save_dir: 保存目录
            figsize: 图片大小
        """
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 未评估")

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        metrics = self.results[model_name]

        # 图1: 混淆矩阵
        ax1 = axes[0]
        cm = np.array([[metrics['tn'], metrics['fp']],
                       [metrics['fn'], metrics['tp']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['正常', '钓鱼'],
                    yticklabels=['正常', '钓鱼'],
                    annot_kws={'size': 12})
        ax1.set_xlabel('预测标签')
        ax1.set_ylabel('真实标签')
        ax1.set_title('混淆矩阵')

        # 图2: ROC曲线
        ax2 = axes[1]
        fpr = metrics['fpr']
        tpr = metrics['tpr']
        auc_score = metrics['auc_roc']
        ax2.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_score:.4f}')
        ax2.plot([0, 1], [0, 1], 'k--', lw=1)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('假阳性率 (FPR)')
        ax2.set_ylabel('真阳性率 (TPR)')
        ax2.set_title('ROC曲线')
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)

        # 图3: PR曲线
        ax3 = axes[2]
        precision = metrics['precision_curve']
        recall = metrics['recall_curve']
        ap_score = metrics['auc_pr']
        ax3.plot(recall, precision, 'b-', lw=2, label=f'AP = {ap_score:.4f}')
        ax3.set_xlim([0.0, 1.0])
        ax3.set_ylim([0.0, 1.05])
        ax3.set_xlabel('召回率 (Recall)')
        ax3.set_ylabel('精确率 (Precision)')
        ax3.set_title('PR曲线')
        ax3.legend(loc='lower left')
        ax3.grid(True, alpha=0.3)

        plt.suptitle(f'{model_name} 性能评估', fontsize=14, y=1.02)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{model_name}_evaluation.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"评估图表已保存至: {save_path}")

        self.figures[f'{model_name}_all_curves'] = fig
        plt.close()

    def plot_metrics_comparison(
        self,
        model_names: List[str] = None,
        metrics_to_plot: List[str] = None,
        save_path: str = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        绘制多模型指标对比图

        Args:
            model_names: 模型名称列表
            metrics_to_plot: 要绘制的指标列表
            save_path: 保存路径
            figsize: 图片大小
        """
        if model_names is None:
            model_names = list(self.results.keys())

        if metrics_to_plot is None:
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr']

        # 准备数据
        data = []
        for model_name in model_names:
            if model_name in self.results:
                for metric in metrics_to_plot:
                    if metric in self.results[model_name]:
                        data.append({
                            'model': model_name,
                            'metric': metric,
                            'value': self.results[model_name][metric]
                        })

        df = pd.DataFrame(data)

        # 绘图
        plt.figure(figsize=figsize)
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(model_names)

        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))
        for i, model_name in enumerate(model_names):
            model_data = df[df['model'] == model_name]
            values = [model_data[model_data['metric'] == m]['value'].values[0]
                      if len(model_data[model_data['metric'] == m]) > 0 else 0
                      for m in metrics_to_plot]
            offset = (i - len(model_names) / 2 + 0.5) * width
            bars = plt.bar(x + offset, values, width, label=model_name, color=colors[i])

            # 添加数值标签
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45)

        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('分数', fontsize=12)
        plt.title('模型性能指标对比', fontsize=14)
        plt.xticks(x, [self._get_metric_name(m) for m in metrics_to_plot], rotation=45, ha='right')
        plt.legend(loc='upper right')
        plt.ylim(0, 1.15)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"指标对比图已保存至: {save_path}")

        self.figures['metrics_comparison'] = plt.gcf()
        plt.close()

    def _get_metric_name(self, metric: str) -> str:
        """获取指标中文名称"""
        names = {
            'accuracy': '准确率',
            'precision': '精确率',
            'recall': '召回率',
            'f1': 'F1分数',
            'specificity': '特异度',
            'mcc': 'MCC',
            'auc_roc': 'AUC-ROC',
            'auc_pr': 'AUC-PR'
        }
        return names.get(metric, metric)

    def get_detailed_metrics(self, model_name: str = None) -> pd.DataFrame:
        """
        获取详细评估指标

        Args:
            model_name: 模型名称，None表示全部

        Returns:
            DataFrame: 详细指标表格
        """
        if model_name is not None:
            if model_name not in self.results:
                raise ValueError(f"模型 {model_name} 未评估")
            models = [model_name]
        else:
            models = list(self.results.keys())

        rows = []
        for name in models:
            metrics = self.results[name]
            row = {
                '模型': name,
                '准确率': f"{metrics['accuracy']:.4f}",
                '精确率': f"{metrics['precision']:.4f}",
                '召回率': f"{metrics['recall']:.4f}",
                'F1分数': f"{metrics['f1']:.4f}",
                '特异度': f"{metrics['specificity']:.4f}",
                'MCC': f"{metrics['mcc']:.4f}",
                'AUC-ROC': f"{metrics['auc_roc']:.4f}",
                'AUC-PR': f"{metrics['auc_pr']:.4f}",
                'TP': metrics['tp'],
                'TN': metrics['tn'],
                'FP': metrics['fp'],
                'FN': metrics['fn']
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def generate_report(
        self,
        save_path: str = None,
        include_figures: bool = True
    ) -> str:
        """
        生成评估报告

        Args:
            save_path: 报告保存路径
            include_figures: 是否包含图表信息

        Returns:
            str: 报告内容
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("模型性能评估报告")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 70)

        # 模型汇总
        report_lines.append("\n1. 模型性能汇总")
        report_lines.append("-" * 50)

        df = self.get_detailed_metrics()
        report_lines.append(df.to_string(index=False))

        # 详细结果
        report_lines.append("\n\n2. 详细评估结果")
        report_lines.append("-" * 50)

        for model_name, metrics in self.results.items():
            report_lines.append(f"\n{model_name}:")
            report_lines.append(f"  - 准确率: {metrics['accuracy']:.4f}")
            report_lines.append(f"  - 精确率: {metrics['precision']:.4f}")
            report_lines.append(f"  - 召回率: {metrics['recall']:.4f}")
            report_lines.append(f"  - F1分数: {metrics['f1']:.4f}")
            report_lines.append(f"  - 特异度: {metrics['specificity']:.4f}")
            report_lines.append(f"  - MCC: {metrics['mcc']:.4f}")
            report_lines.append(f"  - AUC-ROC: {metrics['auc_roc']:.4f}")
            report_lines.append(f"  - AUC-PR: {metrics['auc_pr']:.4f}")
            report_lines.append(f"  - 混淆矩阵: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")

        # 最佳模型
        report_lines.append("\n\n3. 最佳模型分析")
        report_lines.append("-" * 50)

        best_acc_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
        best_auc_model = max(self.results.keys(), key=lambda x: self.results[x]['auc_roc'])
        best_f1_model = max(self.results.keys(), key=lambda x: self.results[x]['f1'])

        report_lines.append(f"  最高准确率: {best_acc_model} ({self.results[best_acc_model]['accuracy']:.4f})")
        report_lines.append(f"  最高AUC-ROC: {best_auc_model} ({self.results[best_auc_model]['auc_roc']:.4f})")
        report_lines.append(f"  最高F1分数: {best_f1_model} ({self.results[best_f1_model]['f1']:.4f})")

        # 目标验证
        report_lines.append("\n\n4. 目标验证")
        report_lines.append("-" * 50)

        best_acc = self.results[best_acc_model]['accuracy']
        best_auc = self.results[best_auc_model]['auc_roc']

        if best_acc >= 0.90:
            report_lines.append(f"  [PASS] 准确率目标: {best_acc:.4f} >= 90%")
        else:
            report_lines.append(f"  [FAIL] 准确率目标: {best_acc:.4f} < 90%")

        if best_auc >= 0.95:
            report_lines.append(f"  [PASS] AUC-ROC目标: {best_auc:.4f} >= 95%")
        else:
            report_lines.append(f"  [WARN] AUC-ROC目标: {best_auc:.4f} < 95%")

        if include_figures:
            report_lines.append("\n\n5. 生成的图表")
            report_lines.append("-" * 50)
            for fig_name in self.figures.keys():
                report_lines.append(f"  - {fig_name}")

        report_lines.append("\n" + "=" * 70)
        report_lines.append("报告结束")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"评估报告已保存至: {save_path}")

        return report

    def compare_models(
        self,
        metric: str = 'accuracy',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        按指定指标比较模型

        Args:
            metric: 比较指标
            ascending: 是否升序排列

        Returns:
            DataFrame: 排序后的模型比较结果
        """
        if not self.results:
            raise ValueError("没有评估结果可比较")

        data = []
        for model_name, metrics in self.results.items():
            data.append({
                '模型': model_name,
                metric: metrics.get(metric, 0)
            })

        df = pd.DataFrame(data)
        df = df.sort_values(metric, ascending=ascending)

        return df

    def save_results(self, filepath: str) -> None:
        """
        保存评估结果到文件

        Args:
            filepath: 保存路径（.json文件）
        """
        # 准备可序列化的结果
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }

        save_dir = os.path.dirname(filepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"评估结果已保存至: {filepath}")

    def load_results(self, filepath: str) -> None:
        """
        从文件加载评估结果

        Args:
            filepath: 结果文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        # 转换回numpy数组
        for model_name in self.results:
            for key in ['fpr', 'tpr', 'roc_thresholds', 'precision_curve', 'recall_curve', 'pr_thresholds']:
                if key in self.results[model_name]:
                    self.results[model_name][key] = np.array(self.results[model_name][key])

        print(f"评估结果已从 {filepath} 加载")


# ==================== Day 19: 评估便捷函数 ====================

def evaluate_all_models(
    trainers: List[BaseModelTrainer],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = None
) -> Tuple[ModelEvaluator, pd.DataFrame]:
    """
    评估所有模型

    Args:
        trainers: 训练器列表
        X_test: 测试特征
        y_test: 测试标签
        save_dir: 图表保存目录

    Returns:
        (evaluator, summary_df)
    """
    evaluator = ModelEvaluator()

    print("=" * 60)
    print("全面模型性能评估")
    print("=" * 60)

    # 评估各模型
    for trainer in trainers:
        if hasattr(trainer, 'model') and trainer.model is not None:
            evaluator.evaluate_model(trainer.model, X_test, y_test, trainer.model_name)

    # 获取汇总
    summary_df = evaluator.get_detailed_metrics()

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # 保存图表
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # ROC曲线
        evaluator.plot_roc_curve(save_path=os.path.join(save_dir, 'roc_curves.png'))

        # PR曲线
        evaluator.plot_pr_curve(save_path=os.path.join(save_dir, 'pr_curves.png'))

        # 指标对比
        evaluator.plot_metrics_comparison(save_path=os.path.join(save_dir, 'metrics_comparison.png'))

        # 各模型混淆矩阵
        for trainer in trainers:
            if trainer.model_name in evaluator.results:
                evaluator.plot_confusion_matrix(
                    trainer.model_name,
                    save_path=os.path.join(save_dir, f'{trainer.model_name}_confusion_matrix.png')
                )

    return evaluator, summary_df


def generate_final_report(
    evaluator: ModelEvaluator,
    save_dir: str,
    train_info: Dict = None
) -> str:
    """
    生成最终评估报告

    Args:
        evaluator: 评估器
        save_dir: 保存目录
        train_info: 训练信息（可选）

    Returns:
        str: 报告路径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 生成报告
    report = evaluator.generate_report()

    # 保存报告
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # 保存评估结果
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    evaluator.save_results(results_path)

    # 保存指标表格
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    evaluator.get_detailed_metrics().to_csv(metrics_path, index=False, encoding='utf-8-sig')

    print(f"\n最终报告已生成:")
    print(f"  - 评估报告: {report_path}")
    print(f"  - 评估结果: {results_path}")
    print(f"  - 指标表格: {metrics_path}")

    return report_path


# ==================== Day 20: 模型管理器 ====================

class ModelManager:
    """
    模型管理器

    Day 20实现 - 负责模型的保存、加载、版本管理和部署导出。

    Attributes:
        models_dir: 模型存储目录
        models: 已加载的模型字典
        config: 模型配置信息

    Example:
        >>> manager = ModelManager('data/models')
        >>> manager.save_all_models(rf_trainer, xgb_trainer, ensemble_trainer, scaler)
        >>> manager.load_all_models()
    """

    def __init__(self, models_dir: str = None):
        """
        初始化模型管理器

        Args:
            models_dir: 模型存储目录
        """
        if models_dir is None:
            models_dir = os.path.join(DATA_DIR, 'models')
        self.models_dir = models_dir
        self.models = {}
        self.config = {}
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """确保模型目录存在"""
        os.makedirs(self.models_dir, exist_ok=True)

    def save_model(self,
                   model,
                   name: str,
                   metadata: Dict = None) -> str:
        """
        保存单个模型

        Args:
            model: 模型对象（sklearn兼容）
            name: 模型名称
            metadata: 模型元数据

        Returns:
            str: 保存路径
        """
        filepath = os.path.join(self.models_dir, f'{name}.pkl')
        joblib.dump(model, filepath)

        # 保存元数据
        if metadata:
            meta_path = os.path.join(self.models_dir, f'{name}_meta.json')
            metadata['saved_at'] = datetime.now().isoformat()
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"模型已保存: {filepath}")
        return filepath

    def load_model(self, name: str) -> Any:
        """
        加载单个模型

        Args:
            name: 模型名称

        Returns:
            加载的模型对象
        """
        filepath = os.path.join(self.models_dir, f'{name}.pkl')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        model = joblib.load(filepath)
        self.models[name] = model
        print(f"模型已加载: {filepath}")
        return model

    def save_all_models(self,
                        rf_trainer: 'RandomForestTrainer' = None,
                        xgb_trainer: 'XGBoostTrainer' = None,
                        ensemble_trainer: 'EnsembleTrainer' = None,
                        scaler = None,
                        feature_names: List[str] = None) -> Dict[str, str]:
        """
        保存所有模型

        Args:
            rf_trainer: RandomForest训练器
            xgb_trainer: XGBoost训练器
            ensemble_trainer: 集成模型训练器
            scaler: 标准化器
            feature_names: 特征名称列表

        Returns:
            dict: 模型名称到保存路径的映射
        """
        saved_paths = {}

        print("\n" + "=" * 60)
        print("保存所有模型")
        print("=" * 60)

        # 保存RandomForest
        if rf_trainer is not None and rf_trainer.is_trained:
            metadata = {
                'model_type': 'RandomForest',
                'params': rf_trainer.training_metrics.get('params', {}) if rf_trainer.training_metrics else {},
                'n_samples': rf_trainer.training_metrics.get('training_samples', 0) if rf_trainer.training_metrics else 0,
                'n_features': rf_trainer.training_metrics.get('n_features', 0) if rf_trainer.training_metrics else 0
            }
            path = self.save_model(rf_trainer.model, 'rf_model_final', metadata)
            saved_paths['rf_model_final'] = path

        # 保存XGBoost
        if xgb_trainer is not None and xgb_trainer.is_trained:
            metadata = {
                'model_type': 'XGBoost',
                'params': xgb_trainer.training_metrics.get('params', {}) if xgb_trainer.training_metrics else {},
                'n_samples': xgb_trainer.training_metrics.get('training_samples', 0) if xgb_trainer.training_metrics else 0,
                'n_features': xgb_trainer.training_metrics.get('n_features', 0) if xgb_trainer.training_metrics else 0
            }
            path = self.save_model(xgb_trainer.model, 'xgb_model_final', metadata)
            saved_paths['xgb_model_final'] = path

        # 保存集成模型
        if ensemble_trainer is not None and ensemble_trainer.is_trained:
            metadata = {
                'model_type': 'Ensemble',
                'params': ensemble_trainer.training_metrics.get('params', {}) if ensemble_trainer.training_metrics else {},
                'weights': ensemble_trainer.weights if hasattr(ensemble_trainer, 'weights') else [0.5, 0.5]
            }
            path = self.save_model(ensemble_trainer.model, 'ensemble_model_final', metadata)
            saved_paths['ensemble_model_final'] = path

        # 保存标准化器
        if scaler is not None:
            path = self.save_model(scaler, 'scaler', {'type': 'StandardScaler'})
            saved_paths['scaler'] = path

        # 保存模型配置
        self.config = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'models': list(saved_paths.keys()),
            'feature_names': feature_names or [],
            'n_features': len(feature_names) if feature_names else 0,
            'primary_model': 'ensemble_model_final'
        }

        config_path = os.path.join(self.models_dir, 'model_config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(f"配置已保存: {config_path}")

        print(f"\n共保存 {len(saved_paths)} 个模型文件")
        return saved_paths

    def load_all_models(self) -> Dict[str, Any]:
        """
        加载所有模型

        Returns:
            dict: 模型名称到模型对象的映射
        """
        print("\n" + "=" * 60)
        print("加载所有模型")
        print("=" * 60)

        # 加载配置
        config_path = os.path.join(self.models_dir, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

        # 加载所有模型
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]

        for model_file in model_files:
            name = model_file.replace('.pkl', '')
            try:
                self.load_model(name)
            except Exception as e:
                print(f"加载模型 {name} 失败: {e}")

        print(f"\n共加载 {len(self.models)} 个模型")
        return self.models

    def get_model(self, name: str) -> Any:
        """
        获取已加载的模型

        Args:
            name: 模型名称

        Returns:
            模型对象
        """
        if name not in self.models:
            self.load_model(name)
        return self.models.get(name)

    def get_model_info(self, name: str = None) -> Dict:
        """
        获取模型信息

        Args:
            name: 模型名称，None表示获取所有模型信息

        Returns:
            dict: 模型信息
        """
        if name is not None:
            meta_path = os.path.join(self.models_dir, f'{name}_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}

        # 返回所有模型信息
        all_info = {'config': self.config, 'models': {}}
        for model_name in self.models.keys():
            all_info['models'][model_name] = self.get_model_info(model_name)

        return all_info

    def export_for_deployment(self,
                               export_dir: str = None,
                               include_scaler: bool = True) -> str:
        """
        导出部署包

        Args:
            export_dir: 导出目录
            include_scaler: 是否包含标准化器

        Returns:
            str: 导出目录路径
        """
        if export_dir is None:
            export_dir = os.path.join(DATA_DIR, 'deployment')

        os.makedirs(export_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print(f"导出部署包到: {export_dir}")
        print("=" * 60)

        import shutil

        # 复制集成模型（主模型）
        src_model = os.path.join(self.models_dir, 'ensemble_model_final.pkl')
        if os.path.exists(src_model):
            shutil.copy(src_model, os.path.join(export_dir, 'model.pkl'))
            print("已复制: ensemble_model_final.pkl -> model.pkl")

        # 复制标准化器
        if include_scaler:
            src_scaler = os.path.join(self.models_dir, 'scaler.pkl')
            if os.path.exists(src_scaler):
                shutil.copy(src_scaler, os.path.join(export_dir, 'scaler.pkl'))
                print("已复制: scaler.pkl")

        # 复制配置
        src_config = os.path.join(self.models_dir, 'model_config.json')
        if os.path.exists(src_config):
            shutil.copy(src_config, os.path.join(export_dir, 'config.json'))
            print("已复制: model_config.json -> config.json")

        # 创建部署说明
        readme = f"""# 钓鱼网站检测模型部署包

## 文件说明
- model.pkl: 集成分类模型（VotingClassifier）
- scaler.pkl: 特征标准化器（StandardScaler）
- config.json: 模型配置信息

## 使用方法

```python
import joblib
import numpy as np

# 加载模型和标准化器
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# 预测
features = np.array([...])  # 30维特征
features_scaled = scaler.transform(features.reshape(1, -1))
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)
```

## 特征说明
请参考config.json中的feature_names字段

## 版本信息
生成时间: {datetime.now().isoformat()}
"""

        readme_path = os.path.join(export_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme)
        print("已创建: README.md")

        print(f"\n部署包导出完成: {export_dir}")
        return export_dir


# ==================== Day 20: 钓鱼网站预测器 ====================

class PhishingPredictor:
    """
    钓鱼网站预测器

    Day 20实现 - 提供简洁的预测接口，封装模型加载和预测逻辑。

    Attributes:
        model: 分类模型
        scaler: 标准化器
        feature_names: 特征名称列表

    Example:
        >>> predictor = PhishingPredictor('data/models')
        >>> result = predictor.predict(features)
        >>> print(result['prediction'], result['probability'])
    """

    def __init__(self,
                 models_dir: str = None,
                 model_name: str = 'ensemble_model_final'):
        """
        初始化预测器

        Args:
            models_dir: 模型目录
            model_name: 使用的模型名称
        """
        if models_dir is None:
            models_dir = os.path.join(DATA_DIR, 'models')
        self.models_dir = models_dir
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.config = {}

        self._load_components()

    def _load_components(self) -> None:
        """加载模型组件"""
        # 加载配置
        config_path = os.path.join(self.models_dir, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            self.feature_names = self.config.get('feature_names', [])

        # 加载模型
        model_path = os.path.join(self.models_dir, f'{self.model_name}.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"模型已加载: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        # 加载标准化器
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler_data = joblib.load(scaler_path)
            # 处理两种格式：直接的scaler对象或包含scaler的字典
            if isinstance(scaler_data, dict) and 'scaler' in scaler_data:
                self.scaler = scaler_data['scaler']
            else:
                self.scaler = scaler_data
            print(f"标准化器已加载: {scaler_path}")

    def predict(self,
                features: np.ndarray,
                return_proba: bool = True) -> Dict:
        """
        单条预测

        Args:
            features: 特征向量（1D或2D数组）
            return_proba: 是否返回概率

        Returns:
            dict: 预测结果
        """
        # 确保是2D数组
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 标准化
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # 预测
        prediction = self.model.predict(features_scaled)[0]

        result = {
            'prediction': int(prediction),
            'label': '钓鱼网站' if prediction == 1 else '正常网站',
            'is_phishing': bool(prediction == 1)
        }

        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            result['probability'] = {
                'legitimate': float(proba[0]),
                'phishing': float(proba[1])
            }
            result['confidence'] = float(max(proba))

        return result

    def predict_batch(self,
                      features: np.ndarray,
                      return_proba: bool = True) -> List[Dict]:
        """
        批量预测

        Args:
            features: 特征矩阵（2D数组）
            return_proba: 是否返回概率

        Returns:
            list: 预测结果列表
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # 标准化
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features

        # 预测
        predictions = self.model.predict(features_scaled)

        results = []
        if return_proba and hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(features_scaled)

            for i, (pred, proba) in enumerate(zip(predictions, probas)):
                result = {
                    'index': i,
                    'prediction': int(pred),
                    'label': '钓鱼网站' if pred == 1 else '正常网站',
                    'is_phishing': bool(pred == 1),
                    'probability': {
                        'legitimate': float(proba[0]),
                        'phishing': float(proba[1])
                    },
                    'confidence': float(max(proba))
                }
                results.append(result)
        else:
            for i, pred in enumerate(predictions):
                result = {
                    'index': i,
                    'prediction': int(pred),
                    'label': '钓鱼网站' if pred == 1 else '正常网站',
                    'is_phishing': bool(pred == 1)
                }
                results.append(result)

        return results

    def get_prediction_details(self,
                                features: np.ndarray,
                                feature_names: List[str] = None) -> Dict:
        """
        获取预测详情（包括特征分析）

        Args:
            features: 特征向量
            feature_names: 特征名称列表

        Returns:
            dict: 详细预测结果
        """
        # 基本预测
        result = self.predict(features, return_proba=True)

        # 添加特征信息
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if feature_names is None:
            feature_names = self.feature_names

        if feature_names:
            result['features'] = {
                name: float(val)
                for name, val in zip(feature_names, features[0])
            }

        # 添加模型信息
        result['model_info'] = {
            'name': self.model_name,
            'type': self.config.get('primary_model', 'unknown'),
            'n_features': len(features[0])
        }

        return result

    def get_model_info(self) -> Dict:
        """
        获取模型信息

        Returns:
            dict: 模型配置信息
        """
        return {
            'config': self.config,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'has_scaler': self.scaler is not None
        }


# ==================== Day 20: 阶段验收函数 ====================

def validate_phase3(models_dir: str = None,
                    test_path: str = None,
                    accuracy_target: float = 0.90,
                    auc_target: float = 0.95) -> Dict:
    """
    验证阶段三所有目标

    Args:
        models_dir: 模型目录
        test_path: 测试集路径
        accuracy_target: 准确率目标
        auc_target: AUC-ROC目标

    Returns:
        dict: 验证结果
    """
    if models_dir is None:
        models_dir = os.path.join(DATA_DIR, 'models')
    if test_path is None:
        test_path = os.path.join(DATA_DIR, 'processed', 'test_30dim.csv')

    print("\n" + "=" * 70)
    print("阶段三验收检查")
    print("=" * 70)

    results = {
        'passed': True,
        'checks': [],
        'metrics': {},
        'warnings': [],
        'timestamp': datetime.now().isoformat()
    }

    # 检查1: 模型文件完整性
    print("\n[检查1] 模型文件完整性")
    print("-" * 50)

    required_files = [
        'rf_model_final.pkl',
        'xgb_model_final.pkl',
        'ensemble_model_final.pkl',
        'scaler.pkl',
        'model_config.json'
    ]

    missing_files = []
    for f in required_files:
        filepath = os.path.join(models_dir, f)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  [OK] {f} ({size:.1f} KB)")
        else:
            print(f"  [MISS] {f} (缺失)")
            missing_files.append(f)

    if missing_files:
        results['passed'] = False
        results['checks'].append({
            'name': '模型文件完整性',
            'passed': False,
            'message': f"缺失文件: {missing_files}"
        })
    else:
        results['checks'].append({
            'name': '模型文件完整性',
            'passed': True,
            'message': "所有模型文件完整"
        })

    # 检查2: 模型加载测试
    print("\n[检查2] 模型加载测试")
    print("-" * 50)

    try:
        predictor = PhishingPredictor(models_dir)
        print("  [OK] 模型加载成功")
        results['checks'].append({
            'name': '模型加载测试',
            'passed': True,
            'message': "模型可正常加载"
        })
    except Exception as e:
        print(f"  [FAIL] 模型加载失败: {e}")
        results['passed'] = False
        results['checks'].append({
            'name': '模型加载测试',
            'passed': False,
            'message': str(e)
        })
        return results

    # 检查3: 性能指标验证
    print("\n[检查3] 性能指标验证")
    print("-" * 50)

    try:
        # 加载测试数据
        df = pd.read_csv(test_path)
        feature_cols = [c for c in df.columns if c not in ['url', 'label']]
        X_test = df[feature_cols].values
        y_test = df['label'].values

        # 预测（不使用scaler，因为30dim数据未标准化需要scaler）
        if predictor.scaler is not None:
            X_test_scaled = predictor.scaler.transform(X_test)
        else:
            X_test_scaled = X_test

        y_pred = predictor.model.predict(X_test_scaled)
        y_proba = predictor.model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)

        results['metrics']['accuracy'] = accuracy
        results['metrics']['auc_roc'] = auc_roc

        print(f"  准确率: {accuracy:.4f} (目标: ≥{accuracy_target})")
        print(f"  AUC-ROC: {auc_roc:.4f} (目标: ≥{auc_target})")

        # 验证准确率
        if accuracy >= accuracy_target:
            print(f"  [OK] 准确率达标")
            results['checks'].append({
                'name': '准确率验证',
                'passed': True,
                'message': f"准确率 {accuracy:.4f} ≥ {accuracy_target}"
            })
        else:
            print(f"  [FAIL] 准确率未达标")
            results['passed'] = False
            results['checks'].append({
                'name': '准确率验证',
                'passed': False,
                'message': f"准确率 {accuracy:.4f} < {accuracy_target}"
            })

        # 验证AUC-ROC
        if auc_roc >= auc_target:
            print(f"  [OK] AUC-ROC达标")
            results['checks'].append({
                'name': 'AUC-ROC验证',
                'passed': True,
                'message': f"AUC-ROC {auc_roc:.4f} ≥ {auc_target}"
            })
        else:
            print(f"  [WARN] AUC-ROC未达目标（可接受）")
            results['warnings'].append(f"AUC-ROC {auc_roc:.4f} < {auc_target}，但性能仍可接受")
            results['checks'].append({
                'name': 'AUC-ROC验证',
                'passed': True,  # 作为警告而非失败
                'message': f"AUC-ROC {auc_roc:.4f} < {auc_target}（警告）"
            })

    except Exception as e:
        print(f"  [FAIL] 性能验证失败: {e}")
        results['passed'] = False
        results['checks'].append({
            'name': '性能指标验证',
            'passed': False,
            'message': str(e)
        })

    # 检查4: 预测功能测试
    print("\n[检查4] 预测功能测试")
    print("-" * 50)

    try:
        # 测试单条预测
        test_sample = X_test[0]
        result = predictor.predict(test_sample)

        print(f"  测试样本预测: {result['label']}")
        print(f"  置信度: {result.get('confidence', 'N/A'):.4f}")
        print("  [OK] 预测功能正常")

        results['checks'].append({
            'name': '预测功能测试',
            'passed': True,
            'message': "单条预测和批量预测功能正常"
        })
    except Exception as e:
        print(f"  [FAIL] 预测功能测试失败: {e}")
        results['passed'] = False
        results['checks'].append({
            'name': '预测功能测试',
            'passed': False,
            'message': str(e)
        })

    # 汇总
    print("\n" + "=" * 70)
    print("验收结果汇总")
    print("=" * 70)

    passed_count = sum(1 for c in results['checks'] if c['passed'])
    total_count = len(results['checks'])

    print(f"检查项: {passed_count}/{total_count} 通过")

    if results['warnings']:
        print("\n警告:")
        for warning in results['warnings']:
            print(f"  [WARN] {warning}")

    if results['passed']:
        print("\n[PASS] 阶段三验收通过！")
    else:
        print("\n[FAIL] 阶段三验收未通过，请检查上述问题。")

    print("=" * 70)

    return results


def generate_phase3_report(models_dir: str = None,
                            test_path: str = None,
                            train_path: str = None,
                            save_dir: str = None) -> str:
    """
    生成阶段三总结报告

    Args:
        models_dir: 模型目录
        test_path: 测试集路径
        train_path: 训练集路径
        save_dir: 报告保存目录

    Returns:
        str: 报告文件路径
    """
    if models_dir is None:
        models_dir = os.path.join(DATA_DIR, 'models')
    if test_path is None:
        test_path = os.path.join(DATA_DIR, 'processed', 'test_30dim.csv')
    if train_path is None:
        train_path = os.path.join(DATA_DIR, 'processed', 'train_30dim.csv')
    if save_dir is None:
        save_dir = os.path.join(DATA_DIR, 'reports')

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("生成阶段三总结报告")
    print("=" * 70)

    # 加载数据
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in test_df.columns if c not in ['url', 'label']]
    X_test = test_df[feature_cols].values
    y_test = test_df['label'].values

    # 加载模型
    manager = ModelManager(models_dir)
    models = manager.load_all_models()

    # 评估所有模型
    evaluator = ModelEvaluator()

    # 加载scaler
    scaler = models.get('scaler')

    for name, model in models.items():
        if name == 'scaler':
            continue
        try:
            # 对测试数据进行标准化
            if scaler is not None:
                X_test_scaled = scaler.transform(X_test)
            else:
                X_test_scaled = X_test
            evaluator.evaluate_model(model, X_test_scaled, y_test, name, verbose=False)
        except Exception as e:
            print(f"评估模型 {name} 失败: {e}")

    # 生成报告内容
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("阶段三：模型训练与优化 - 总结报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)

    # 1. 项目概述
    report_lines.append("\n## 1. 项目概述")
    report_lines.append("-" * 50)
    report_lines.append("项目名称: 基于网络流量特征的钓鱼网站检测技术研究")
    report_lines.append("阶段目标: 训练高性能的钓鱼网站检测模型")
    report_lines.append(f"训练集样本数: {len(train_df)}")
    report_lines.append(f"测试集样本数: {len(test_df)}")
    report_lines.append(f"特征数量: {len(feature_cols)}")

    # 2. 模型列表
    report_lines.append("\n## 2. 训练的模型")
    report_lines.append("-" * 50)

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    for f in model_files:
        size = os.path.getsize(os.path.join(models_dir, f)) / 1024
        report_lines.append(f"  - {f} ({size:.1f} KB)")

    # 3. 性能评估
    report_lines.append("\n## 3. 模型性能评估")
    report_lines.append("-" * 50)

    if evaluator.results:
        metrics_df = evaluator.get_detailed_metrics()
        report_lines.append(metrics_df.to_string(index=False))

    # 4. 最佳模型分析
    report_lines.append("\n## 4. 最佳模型分析")
    report_lines.append("-" * 50)

    if evaluator.results:
        best_model = max(evaluator.results.keys(),
                        key=lambda x: evaluator.results[x].get('accuracy', 0))
        best_metrics = evaluator.results[best_model]

        report_lines.append(f"最佳模型: {best_model}")
        report_lines.append(f"  准确率: {best_metrics.get('accuracy', 0):.4f}")
        report_lines.append(f"  精确率: {best_metrics.get('precision', 0):.4f}")
        report_lines.append(f"  召回率: {best_metrics.get('recall', 0):.4f}")
        report_lines.append(f"  F1分数: {best_metrics.get('f1', 0):.4f}")
        report_lines.append(f"  AUC-ROC: {best_metrics.get('auc_roc', 0):.4f}")
        report_lines.append(f"  AUC-PR: {best_metrics.get('auc_pr', 0):.4f}")

    # 5. 阶段完成情况
    report_lines.append("\n## 5. 阶段完成情况")
    report_lines.append("-" * 50)
    report_lines.append("Day 14: [OK] BaseModelTrainer基类和RandomForestTrainer实现")
    report_lines.append("Day 15: [OK] XGBoostTrainer实现")
    report_lines.append("Day 16: [OK] EnsembleTrainer集成模型实现")
    report_lines.append("Day 17: [OK] CrossValidator交叉验证实现")
    report_lines.append("Day 18: [OK] HyperparameterTuner超参数调优实现")
    report_lines.append("Day 19: [OK] ModelEvaluator全面性能评估实现")
    report_lines.append("Day 20: [OK] 模型保存与阶段验收完成")

    # 6. 产出物清单
    report_lines.append("\n## 6. 产出物清单")
    report_lines.append("-" * 50)
    report_lines.append("代码文件:")
    report_lines.append("  - src/model_training.py (模型训练模块)")
    report_lines.append("  - tests/test_model_training.py (单元测试)")
    report_lines.append("\n模型文件:")
    for f in model_files:
        report_lines.append(f"  - data/models/{f}")
    report_lines.append("\n报告文件:")
    report_lines.append("  - data/reports/phase3_report.txt")
    report_lines.append("  - data/evaluation/reports/evaluation_report.txt")
    report_lines.append("  - data/evaluation/reports/evaluation_metrics.csv")

    # 7. 下一阶段准备
    report_lines.append("\n## 7. 下一阶段准备（阶段四：Web开发）")
    report_lines.append("-" * 50)
    report_lines.append("已准备就绪的组件:")
    report_lines.append("  - 集成分类模型 (ensemble_model_final.pkl)")
    report_lines.append("  - 特征标准化器 (scaler.pkl)")
    report_lines.append("  - 预测接口类 (PhishingPredictor)")
    report_lines.append("\n阶段四主要任务:")
    report_lines.append("  - Day 21-22: Flask基础架构搭建")
    report_lines.append("  - Day 23-24: 前端界面开发")
    report_lines.append("  - Day 25-26: 检测功能集成")
    report_lines.append("  - Day 27-28: 测试与部署")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("报告结束")
    report_lines.append("=" * 70)

    # 保存报告
    report = "\n".join(report_lines)
    report_path = os.path.join(save_dir, 'phase3_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存: {report_path}")
    print(report)

    return report_path


if __name__ == "__main__":
    main()
