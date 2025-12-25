"""
特征预处理模块

src/preprocessor.py

提供特征预处理功能：
- 缺失值处理
- 异常值处理
- 特征标准化
- 预处理器持久化
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
import joblib
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """
    特征预处理器

    提供完整的特征预处理流程：
    1. 缺失值检测和填充
    2. 异常值检测和截断
    3. 特征标准化

    Attributes:
        scaler (StandardScaler): 标准化器
        feature_names (list): 特征名列表
        fill_values (dict): 缺失值填充值
        bounds (dict): 异常值边界

    Example:
        >>> preprocessor = FeaturePreprocessor()
        >>> X_train_scaled = preprocessor.fit_transform(X_train)
        >>> X_test_scaled = preprocessor.transform(X_test)
        >>> preprocessor.save('scaler.pkl')
    """

    # 默认缺失值标记
    MISSING_VALUES = [-1, np.nan, None]

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        初始化特征预处理器

        Args:
            feature_names: 特征名列表（可选）
        """
        self.scaler = StandardScaler()
        self.feature_names = feature_names

        # 缺失值填充值（fit时计算）
        self.fill_values = {}

        # 异常值边界（fit时计算）
        self.bounds = {}

        # 拟合状态
        self._fitted = False

        # 统计信息
        self.stats = {
            'missing_count': {},
            'outlier_count': {},
            'feature_means': {},
            'feature_stds': {}
        }

    def _detect_missing(self, X: np.ndarray) -> np.ndarray:
        """
        检测缺失值

        Args:
            X: 特征矩阵

        Returns:
            布尔矩阵，True表示缺失
        """
        # 检测NaN
        missing_mask = np.isnan(X)

        # 检测-1值（我们的默认缺失值标记）
        missing_mask = missing_mask | (X == -1)

        return missing_mask

    def _fill_missing(
        self,
        X: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """
        填充缺失值

        Args:
            X: 特征矩阵
            fit: 是否计算填充值

        Returns:
            填充后的特征矩阵
        """
        X_filled = X.copy()
        n_features = X.shape[1]

        for i in range(n_features):
            col = X_filled[:, i]
            missing_mask = self._detect_missing(col.reshape(-1, 1)).flatten()

            # 统计缺失数量
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            self.stats['missing_count'][feature_name] = int(np.sum(missing_mask))

            if fit:
                # 计算非缺失值的均值
                valid_values = col[~missing_mask]
                if len(valid_values) > 0:
                    self.fill_values[i] = np.mean(valid_values)
                else:
                    self.fill_values[i] = 0

            # 填充缺失值
            if i in self.fill_values:
                X_filled[missing_mask, i] = self.fill_values[i]

        return X_filled

    def _detect_outliers_iqr(
        self,
        X: np.ndarray,
        fit: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用IQR方法检测异常值

        IQR = Q3 - Q1
        下界 = Q1 - 1.5 * IQR
        上界 = Q3 + 1.5 * IQR

        Args:
            X: 特征矩阵
            fit: 是否计算边界

        Returns:
            (下界数组, 上界数组)
        """
        n_features = X.shape[1]
        lower_bounds = np.zeros(n_features)
        upper_bounds = np.zeros(n_features)

        for i in range(n_features):
            col = X[:, i]

            if fit:
                Q1 = np.percentile(col, 25)
                Q3 = np.percentile(col, 75)
                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                self.bounds[i] = (lower, upper)

            if i in self.bounds:
                lower_bounds[i], upper_bounds[i] = self.bounds[i]

        return lower_bounds, upper_bounds

    def _clip_outliers(
        self,
        X: np.ndarray,
        fit: bool = False
    ) -> np.ndarray:
        """
        截断异常值到边界

        Args:
            X: 特征矩阵
            fit: 是否计算边界

        Returns:
            截断后的特征矩阵
        """
        X_clipped = X.copy()
        lower_bounds, upper_bounds = self._detect_outliers_iqr(X, fit=fit)

        for i in range(X.shape[1]):
            col = X_clipped[:, i]
            lower, upper = lower_bounds[i], upper_bounds[i]

            # 统计异常值数量
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            outlier_count = np.sum((col < lower) | (col > upper))
            self.stats['outlier_count'][feature_name] = int(outlier_count)

            # 截断到边界
            X_clipped[:, i] = np.clip(col, lower, upper)

        return X_clipped

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'FeaturePreprocessor':
        """
        拟合预处理器

        计算缺失值填充值、异常值边界和标准化参数。

        Args:
            X: 训练集特征矩阵

        Returns:
            self
        """
        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values

        X = X.astype(np.float64)

        logger.info(f"开始拟合预处理器: {X.shape[0]} 样本, {X.shape[1]} 特征")

        # 1. 填充缺失值
        X_filled = self._fill_missing(X, fit=True)

        # 2. 截断异常值
        X_clipped = self._clip_outliers(X_filled, fit=True)

        # 3. 拟合标准化器
        self.scaler.fit(X_clipped)

        # 保存统计信息
        for i in range(X.shape[1]):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            self.stats['feature_means'][feature_name] = float(self.scaler.mean_[i])
            self.stats['feature_stds'][feature_name] = float(self.scaler.scale_[i])

        self._fitted = True
        logger.info("预处理器拟合完成")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        转换特征矩阵

        Args:
            X: 特征矩阵

        Returns:
            预处理后的特征矩阵
        """
        if not self._fitted:
            raise RuntimeError("预处理器尚未拟合，请先调用fit()")

        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = X.astype(np.float64)

        # 1. 填充缺失值
        X_filled = self._fill_missing(X, fit=False)

        # 2. 截断异常值
        X_clipped = self._clip_outliers(X_filled, fit=False)

        # 3. 标准化
        X_scaled = self.scaler.transform(X_clipped)

        return X_scaled

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        拟合并转换特征矩阵

        Args:
            X: 训练集特征矩阵

        Returns:
            预处理后的特征矩阵
        """
        self.fit(X)
        return self.transform(X)

    def save(self, path: str):
        """
        保存预处理器

        Args:
            path: 保存路径
        """
        if not self._fitted:
            raise RuntimeError("预处理器尚未拟合，无法保存")

        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # 保存整个预处理器对象
        save_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'fill_values': self.fill_values,
            'bounds': self.bounds,
            'stats': self.stats
        }

        joblib.dump(save_data, path)
        logger.info(f"预处理器已保存: {path}")

    @classmethod
    def load(cls, path: str) -> 'FeaturePreprocessor':
        """
        加载预处理器

        Args:
            path: 加载路径

        Returns:
            FeaturePreprocessor实例
        """
        save_data = joblib.load(path)

        preprocessor = cls(feature_names=save_data['feature_names'])
        preprocessor.scaler = save_data['scaler']
        preprocessor.fill_values = save_data['fill_values']
        preprocessor.bounds = save_data['bounds']
        preprocessor.stats = save_data['stats']
        preprocessor._fitted = True

        logger.info(f"预处理器已加载: {path}")
        return preprocessor

    def get_statistics(self) -> Dict:
        """
        获取预处理统计信息

        Returns:
            统计信息字典
        """
        return self.stats

    def get_feature_importance_by_variance(self) -> Dict[str, float]:
        """
        根据标准化后的方差获取特征重要性

        方差越大，特征区分度越高。

        Returns:
            特征名到方差的映射
        """
        if not self._fitted:
            raise RuntimeError("预处理器尚未拟合")

        importance = {}
        for i, name in enumerate(self.feature_names or range(len(self.scaler.scale_))):
            # 标准化后方差为1，这里返回原始标准差作为参考
            importance[name] = float(self.scaler.scale_[i])

        return importance


# ==================== 便捷函数 ====================

def preprocess_dataset(
    train_path: str,
    test_path: str,
    output_dir: str,
    scaler_path: str,
    feature_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    便捷函数：预处理数据集

    Args:
        train_path: 训练集CSV路径
        test_path: 测试集CSV路径
        output_dir: 输出目录
        scaler_path: 标准化器保存路径
        feature_columns: 特征列名（可选，默认排除url和label）

    Returns:
        (训练集DataFrame, 测试集DataFrame)

    Example:
        >>> train_df, test_df = preprocess_dataset(
        ...     'train_features.csv',
        ...     'test_features.csv',
        ...     'processed/',
        ...     'models/scaler.pkl'
        ... )
    """
    # 读取数据
    logger.info(f"读取训练集: {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"读取测试集: {test_path}")
    test_df = pd.read_csv(test_path)

    # 确定特征列
    if feature_columns is None:
        feature_columns = [col for col in train_df.columns
                         if col not in ['url', 'label']]

    logger.info(f"特征列: {len(feature_columns)} 个")

    # 提取特征和标签
    X_train = train_df[feature_columns]
    y_train = train_df['label']
    X_test = test_df[feature_columns]
    y_test = test_df['label']

    # 创建预处理器
    preprocessor = FeaturePreprocessor(feature_names=feature_columns)

    # 预处理
    logger.info("预处理训练集...")
    X_train_scaled = preprocessor.fit_transform(X_train)

    logger.info("预处理测试集...")
    X_test_scaled = preprocessor.transform(X_test)

    # 保存预处理器
    preprocessor.save(scaler_path)

    # 构建输出DataFrame
    train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    train_scaled_df['url'] = train_df['url'].values
    train_scaled_df['label'] = y_train.values

    test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
    test_scaled_df['url'] = test_df['url'].values
    test_scaled_df['label'] = y_test.values

    # 调整列顺序
    cols = ['url', 'label'] + feature_columns
    train_scaled_df = train_scaled_df[cols]
    test_scaled_df = test_scaled_df[cols]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存预处理后的数据
    train_output = os.path.join(output_dir, 'train_scaled.csv')
    test_output = os.path.join(output_dir, 'test_scaled.csv')

    train_scaled_df.to_csv(train_output, index=False)
    test_scaled_df.to_csv(test_output, index=False)

    logger.info(f"训练集已保存: {train_output}")
    logger.info(f"测试集已保存: {test_output}")

    # 输出统计信息
    stats = preprocessor.get_statistics()
    logger.info("\n预处理统计:")
    logger.info(f"  缺失值总数: {sum(stats['missing_count'].values())}")
    logger.info(f"  异常值总数: {sum(stats['outlier_count'].values())}")

    return train_scaled_df, test_scaled_df


def validate_preprocessing(
    original_path: str,
    scaled_path: str
) -> Dict:
    """
    验证预处理结果

    Args:
        original_path: 原始数据路径
        scaled_path: 预处理后数据路径

    Returns:
        验证结果字典
    """
    original_df = pd.read_csv(original_path)
    scaled_df = pd.read_csv(scaled_path)

    feature_columns = [col for col in scaled_df.columns
                      if col not in ['url', 'label']]

    results = {
        'record_count_match': len(original_df) == len(scaled_df),
        'feature_count': len(feature_columns),
        'has_nan': scaled_df[feature_columns].isna().any().any(),
        'has_inf': np.isinf(scaled_df[feature_columns].values).any(),
        'mean_close_to_zero': {},
        'std_close_to_one': {}
    }

    # 检查标准化效果
    for col in feature_columns:
        mean = scaled_df[col].mean()
        std = scaled_df[col].std()
        results['mean_close_to_zero'][col] = abs(mean) < 0.1
        results['std_close_to_one'][col] = abs(std - 1) < 0.2

    return results


if __name__ == '__main__':
    # 测试代码
    import tempfile

    print("=" * 60)
    print("FeaturePreprocessor 测试")
    print("=" * 60)

    # 创建测试数据
    X = np.array([
        [1, 2, -1, 100],  # 包含缺失值和异常值
        [4, 5, 6, 7],
        [7, 8, 9, 10],
        [-1, 11, 12, 13],  # 包含缺失值
        [10, 14, 15, 16]
    ], dtype=float)

    print("\n[1] 原始数据:")
    print(X)

    # 创建预处理器
    preprocessor = FeaturePreprocessor(feature_names=['a', 'b', 'c', 'd'])
    X_scaled = preprocessor.fit_transform(X)

    print("\n[2] 预处理后:")
    print(X_scaled)

    print("\n[3] 统计信息:")
    stats = preprocessor.get_statistics()
    print(f"  缺失值: {stats['missing_count']}")
    print(f"  异常值: {stats['outlier_count']}")

    print("\n[4] 标准化效果:")
    print(f"  均值: {X_scaled.mean(axis=0)}")
    print(f"  标准差: {X_scaled.std(axis=0)}")

    # 测试保存和加载
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        preprocessor.save(f.name)
        loaded = FeaturePreprocessor.load(f.name)
        X_scaled2 = loaded.transform(X)
        assert np.allclose(X_scaled, X_scaled2)
        print("\n[5] 保存/加载验证通过")

    print("\n" + "=" * 60)
    print("[SUCCESS] 测试完成!")
    print("=" * 60)
