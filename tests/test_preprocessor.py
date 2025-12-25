"""
特征预处理单元测试
"""

import pytest
import sys
import os
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import (
    FeaturePreprocessor,
    preprocess_dataset,
    validate_preprocessing
)


class TestFeaturePreprocessor:
    """FeaturePreprocessor测试类"""

    def test_init(self):
        """测试初始化"""
        preprocessor = FeaturePreprocessor()
        assert preprocessor._fitted == False
        assert preprocessor.scaler is not None

    def test_fit_transform_array(self):
        """测试numpy数组的fit_transform"""
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        assert X_scaled.shape == X.shape
        assert preprocessor._fitted == True

    def test_fit_transform_dataframe(self):
        """测试DataFrame的fit_transform"""
        df = pd.DataFrame({
            'a': [1, 4, 7, 10],
            'b': [2, 5, 8, 11],
            'c': [3, 6, 9, 12]
        })

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(df)

        assert X_scaled.shape == (4, 3)
        assert preprocessor.feature_names == ['a', 'b', 'c']

    def test_missing_value_handling(self):
        """测试缺失值处理"""
        X = np.array([
            [1, 2, 3],
            [-1, 5, 6],  # 第一个特征缺失
            [7, -1, 9],  # 第二个特征缺失
            [10, 11, 12]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        # 缺失值应该被填充，不应该是-1
        assert not np.any(X_scaled == -1)

    def test_outlier_handling(self):
        """测试异常值处理"""
        X = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [100, 4, 5],  # 第一个特征是异常值
            [4, 5, 6]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        # 异常值应该被截断
        assert X_scaled[3, 0] < 10  # 不应该是标准化后的100

    def test_transform_without_fit(self):
        """测试未fit时调用transform"""
        X = np.array([[1, 2, 3]])
        preprocessor = FeaturePreprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(X)

    def test_standardization_effect(self):
        """测试标准化效果"""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50  # 均值50，标准差10

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X)

        # 标准化后均值应接近0，标准差应接近1
        assert np.abs(X_scaled.mean()) < 0.5
        assert np.abs(X_scaled.std() - 1) < 0.3

    def test_save_and_load(self):
        """测试保存和加载"""
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=float)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            # 拟合并保存
            preprocessor1 = FeaturePreprocessor()
            X_scaled1 = preprocessor1.fit_transform(X)
            preprocessor1.save(f.name)

            # 加载并转换
            preprocessor2 = FeaturePreprocessor.load(f.name)
            X_scaled2 = preprocessor2.transform(X)

            # 结果应该相同
            np.testing.assert_array_almost_equal(X_scaled1, X_scaled2)

    def test_get_statistics(self):
        """测试获取统计信息"""
        X = np.array([
            [1, -1, 3],
            [4, 5, 6],
            [-1, 8, 9]
        ], dtype=float)

        preprocessor = FeaturePreprocessor(feature_names=['a', 'b', 'c'])
        preprocessor.fit_transform(X)

        stats = preprocessor.get_statistics()

        assert 'missing_count' in stats
        assert 'outlier_count' in stats
        assert 'feature_means' in stats
        assert 'feature_stds' in stats


class TestPreprocessorEdgeCases:
    """边界情况测试"""

    def test_single_sample(self):
        """测试单样本"""
        X = np.array([[1, 2, 3]], dtype=float)

        preprocessor = FeaturePreprocessor()
        # 单样本会导致标准差为0，应该能处理
        X_scaled = preprocessor.fit_transform(X)
        assert X_scaled.shape == (1, 3)

    def test_all_missing(self):
        """测试全部缺失"""
        X = np.array([
            [-1, -1, -1],
            [-1, -1, -1]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        # 应该不崩溃，填充为0
        assert X_scaled.shape == (2, 3)

    def test_constant_feature(self):
        """测试常量特征"""
        X = np.array([
            [1, 5, 3],
            [1, 5, 4],
            [1, 5, 5]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        # 常量特征标准化后应该是0
        assert np.allclose(X_scaled[:, 1], 0)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_preprocess_dataset(self):
        """测试数据集预处理函数"""
        # 创建临时训练集
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f_train:
            f_train.write("url,label,feat1,feat2,feat3\n")
            f_train.write("http://a.com,0,1,2,3\n")
            f_train.write("http://b.com,1,4,5,6\n")
            f_train.write("http://c.com,0,7,8,9\n")
            train_path = f_train.name

        # 创建临时测试集
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f_test:
            f_test.write("url,label,feat1,feat2,feat3\n")
            f_test.write("http://d.com,0,2,3,4\n")
            f_test.write("http://e.com,1,5,6,7\n")
            test_path = f_test.name

        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, 'scaler.pkl')

            train_df, test_df = preprocess_dataset(
                train_path=train_path,
                test_path=test_path,
                output_dir=tmpdir,
                scaler_path=scaler_path
            )

            assert len(train_df) == 3
            assert len(test_df) == 2
            assert os.path.exists(scaler_path)

    def test_validate_preprocessing(self):
        """测试预处理验证函数"""
        # 创建原始数据
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f_orig:
            f_orig.write("url,label,feat1,feat2\n")
            f_orig.write("http://a.com,0,1,2\n")
            f_orig.write("http://b.com,1,3,4\n")
            orig_path = f_orig.name

        # 创建标准化后数据（模拟标准化效果）
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f_scaled:
            f_scaled.write("url,label,feat1,feat2\n")
            f_scaled.write("http://a.com,0,-1,0\n")
            f_scaled.write("http://b.com,1,1,0\n")
            scaled_path = f_scaled.name

        result = validate_preprocessing(orig_path, scaled_path)

        assert result['record_count_match'] == True
        assert result['feature_count'] == 2
        assert result['has_nan'] == False


class TestMissingValueDetection:
    """缺失值检测测试"""

    def test_detect_minus_one(self):
        """测试-1值检测"""
        X = np.array([
            [1, 2, 3],
            [-1, 5, 6],
            [7, 8, -1]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        missing_mask = preprocessor._detect_missing(X)

        assert missing_mask[1, 0] == True  # -1 at row 1, col 0
        assert missing_mask[2, 2] == True  # -1 at row 2, col 2
        assert missing_mask[0, 0] == False  # 1 is not missing

    def test_detect_nan(self):
        """测试NaN值检测"""
        X = np.array([
            [1, 2, np.nan],
            [np.nan, 5, 6],
            [7, 8, 9]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        missing_mask = preprocessor._detect_missing(X)

        assert missing_mask[0, 2] == True  # NaN at row 0, col 2
        assert missing_mask[1, 0] == True  # NaN at row 1, col 0


class TestOutlierDetection:
    """异常值检测测试"""

    def test_iqr_bounds(self):
        """测试IQR边界计算"""
        # 创建已知分布的数据
        X = np.array([
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
            [9],
            [10]
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        preprocessor._detect_outliers_iqr(X, fit=True)

        # Q1=3, Q3=8, IQR=5
        # lower=3-7.5=-4.5, upper=8+7.5=15.5
        assert 0 in preprocessor.bounds
        lower, upper = preprocessor.bounds[0]
        assert lower < 1  # 下界应小于最小值
        assert upper > 10  # 上界应大于最大值

    def test_outlier_clipping(self):
        """测试异常值截断"""
        X = np.array([
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 5],
            [100, 5]  # 100是异常值
        ], dtype=float)

        preprocessor = FeaturePreprocessor()
        X_clipped = preprocessor._clip_outliers(X, fit=True)

        # 100应该被截断
        assert X_clipped[4, 0] < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
