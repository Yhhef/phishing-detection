"""
模型训练模块单元测试

测试 BaseModelTrainer 和 RandomForestTrainer 的核心功能
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import (
    BaseModelTrainer,
    RandomForestTrainer,
    load_feature_data
)


class TestRandomForestTrainer:
    """RandomForestTrainer测试类"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    def test_build_model(self):
        """测试模型构建"""
        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)

        assert trainer.model is not None
        assert trainer.model.n_estimators == 10

    def test_build_model_default_params(self):
        """测试默认参数构建模型"""
        trainer = RandomForestTrainer()
        trainer.build_model()

        assert trainer.model is not None
        assert trainer.model.n_estimators == 100
        assert trainer.model.max_depth == 10

    def test_train(self, sample_data):
        """测试模型训练"""
        X_train, y_train, _, _ = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        assert trainer.is_trained == True
        assert trainer.training_metrics is not None
        assert 'training_samples' in trainer.training_metrics

    def test_train_without_build(self, sample_data):
        """测试不先构建就训练（应自动构建）"""
        X_train, y_train, _, _ = sample_data

        trainer = RandomForestTrainer()
        # 不调用 build_model，直接训练
        trainer.train(X_train, y_train)

        assert trainer.is_trained == True
        assert trainer.model is not None

    def test_predict(self, sample_data):
        """测试预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        y_pred = trainer.predict(X_test)

        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        """测试概率预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        y_proba = trainer.predict_proba(X_test)

        assert y_proba.shape == (len(X_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_evaluate(self, sample_data):
        """测试评估"""
        X_train, y_train, X_test, y_test = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        metrics = trainer.evaluate(X_test, y_test)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_feature_importance(self, sample_data):
        """测试特征重要性"""
        X_train, y_train, _, _ = sample_data
        feature_names = [f'feat_{i}' for i in range(10)]

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train, feature_names=feature_names)

        importance_df = trainer.get_feature_importance()

        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        # 验证按重要性降序排列
        assert importance_df['importance'].is_monotonic_decreasing

    def test_feature_importance_without_names(self, sample_data):
        """测试无特征名称时的特征重要性"""
        X_train, y_train, _, _ = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)  # 不传特征名称

        importance_df = trainer.get_feature_importance()

        assert len(importance_df) == 10
        # 应该自动生成特征名称
        assert importance_df['feature'].iloc[0].startswith('feature_')

    def test_save_and_load(self, sample_data):
        """测试模型保存和加载"""
        X_train, y_train, X_test, _ = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        # 保存
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            trainer.save_model(filepath)
            assert os.path.exists(filepath)

            # 加载
            trainer2 = RandomForestTrainer()
            trainer2.load_model(filepath)

            # 验证预测一致
            y_pred1 = trainer.predict(X_test)
            y_pred2 = trainer2.predict(X_test)

            assert np.array_equal(y_pred1, y_pred2)
            assert trainer2.is_trained == True

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_untrained_model_error(self):
        """测试未训练模型的错误处理"""
        trainer = RandomForestTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.predict(np.random.randn(10, 5))

    def test_untrained_predict_proba_error(self):
        """测试未训练模型预测概率的错误处理"""
        trainer = RandomForestTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.predict_proba(np.random.randn(10, 5))

    def test_untrained_evaluate_error(self, sample_data):
        """测试未训练模型评估的错误处理"""
        _, _, X_test, y_test = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.evaluate(X_test, y_test)

    def test_untrained_save_error(self):
        """测试未训练模型保存的错误处理"""
        trainer = RandomForestTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.save_model("test.pkl")

    def test_untrained_feature_importance_error(self):
        """测试未训练模型获取特征重要性的错误处理"""
        trainer = RandomForestTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.get_feature_importance()

    def test_load_nonexistent_model_error(self):
        """测试加载不存在模型的错误处理"""
        trainer = RandomForestTrainer()

        with pytest.raises(FileNotFoundError):
            trainer.load_model("nonexistent_model.pkl")


class TestLoadFeatureData:
    """数据加载函数测试"""

    def test_load_feature_data(self, tmp_path):
        """测试数据加载"""
        # 创建测试数据目录结构
        processed_dir = tmp_path / 'data' / 'processed'
        processed_dir.mkdir(parents=True)

        # 创建测试数据
        train_df = pd.DataFrame({
            'url': ['http://a.com', 'http://b.com'],
            'feature1': [1.0, 2.0],
            'feature2': [3.0, 4.0],
            'label': [0, 1]
        })
        test_df = pd.DataFrame({
            'url': ['http://c.com'],
            'feature1': [5.0],
            'feature2': [6.0],
            'label': [0]
        })

        train_path = processed_dir / 'train_scaled.csv'
        test_path = processed_dir / 'test_scaled.csv'

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        # 使用 monkeypatch 来修改 DATA_DIR
        import src.model_training as mt
        original_data_dir = mt.DATA_DIR

        try:
            mt.DATA_DIR = str(tmp_path / 'data')

            # 加载
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_scaled=True)

            assert X_train.shape == (2, 2)
            assert X_test.shape == (1, 2)
            assert len(y_train) == 2
            assert len(y_test) == 1
            assert 'feature1' in feature_names
            assert 'feature2' in feature_names

        finally:
            mt.DATA_DIR = original_data_dir


class TestBaseModelTrainer:
    """基类测试"""

    def test_abstract_class(self):
        """测试抽象类不能直接实例化（通过子类测试方法）"""
        # BaseModelTrainer 是抽象类，不能直接实例化
        # 通过 RandomForestTrainer 来验证基类方法
        trainer = RandomForestTrainer()
        assert trainer.model_name == "rf_model"
        assert trainer.model is None
        assert trainer.is_trained == False


class TestIntegration:
    """集成测试 - 使用真实数据"""

    @pytest.fixture
    def real_data(self):
        """尝试加载真实数据，如果不存在则跳过"""
        try:
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_scaled=True)
            return X_train, y_train, X_test, y_test, feature_names
        except FileNotFoundError:
            pytest.skip("真实数据文件不存在，跳过集成测试")

    def test_real_data_training(self, real_data):
        """测试真实数据训练"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=100, max_depth=10)
        trainer.train(X_train, y_train, feature_names=feature_names)

        metrics = trainer.evaluate(X_test, y_test)

        # Day 14 目标：准确率 >= 85%
        assert metrics['accuracy'] >= 0.85, f"准确率 {metrics['accuracy']:.4f} 低于 85% 目标"

        print(f"\n真实数据测试结果:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
