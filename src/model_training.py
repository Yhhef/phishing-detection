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
    confusion_matrix, classification_report, roc_auc_score
)

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
    XGBoost训练器

    Day 15实现 - 预留接口
    """

    def __init__(self, model_name: str = "xgb_model"):
        super().__init__(model_name)

    def build_model(self, **kwargs):
        """构建XGBoost模型 - Day 15实现"""
        raise NotImplementedError("XGBoost训练器将在Day 15实现")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练XGBoost模型 - Day 15实现"""
        raise NotImplementedError("XGBoost训练器将在Day 15实现")


class EnsembleTrainer(BaseModelTrainer):
    """
    集成模型训练器

    Day 16实现 - 预留接口
    """

    def __init__(self, model_name: str = "ensemble_model"):
        super().__init__(model_name)

    def build_model(self, **kwargs):
        """构建集成模型 - Day 16实现"""
        raise NotImplementedError("集成模型训练器将在Day 16实现")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """训练集成模型 - Day 16实现"""
        raise NotImplementedError("集成模型训练器将在Day 16实现")


def load_feature_data(use_scaled: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载特征数据

    Args:
        use_scaled: 是否使用标准化后的数据

    Returns:
        (X_train, y_train, X_test, y_test)
    """
    processed_dir = os.path.join(DATA_DIR, 'processed')

    if use_scaled:
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
    主函数 - Day 14: RandomForest训练与评估
    """
    print("=" * 60)
    print("Day 14 - RandomForest分类器训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载特征数据...")
    X_train, y_train, X_test, y_test, feature_cols = load_feature_data(use_scaled=True)

    # 2. 创建并构建模型
    print("\n[2] 构建RandomForest模型...")
    trainer = RandomForestTrainer(model_name="rf_model")
    trainer.build_model(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    # 3. 训练模型
    print("\n[3] 训练模型...")
    trainer.train(X_train, y_train, feature_names=feature_cols)

    # 4. 评估模型
    print("\n[4] 评估模型性能...")
    metrics = trainer.evaluate(X_test, y_test)

    print("\n" + "=" * 40)
    print("模型评估结果:")
    print("=" * 40)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1分数 (F1-Score):  {metrics['f1']:.4f}")
    print(f"AUC-ROC:            {metrics['roc_auc']:.4f}")

    print("\n混淆矩阵:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"  预测合法  预测钓鱼")
    print(f"实际合法  {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"实际钓鱼  {cm[1][0]:5d}   {cm[1][1]:5d}")

    # 5. 验证是否达到目标
    print("\n" + "=" * 40)
    print("性能验证 (Day 14 目标: 准确率 ≥ 85%):")
    print("=" * 40)

    target_accuracy = 0.85
    if metrics['accuracy'] >= target_accuracy:
        print(f"[PASS] 准确率 {metrics['accuracy']*100:.2f}% >= 85% 目标达成!")
    else:
        print(f"[WARN] 准确率 {metrics['accuracy']*100:.2f}% < 85% 未达目标")

    # 6. 特征重要性
    print("\n[5] 特征重要性 (Top 10):")
    importance_df = trainer.get_feature_importance()
    print(importance_df.head(10).to_string(index=False))

    # 7. 保存模型
    print("\n[6] 保存模型...")
    model_path = trainer.save_model()
    print(f"模型已保存到: {model_path}")

    # 8. 返回结果
    print("\n" + "=" * 60)
    print("Day 14 任务完成!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    main()
