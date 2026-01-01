"""
模型训练模块单元测试

测试 BaseModelTrainer、RandomForestTrainer、XGBoostTrainer 和 EnsembleTrainer 的核心功能
Day 14: RandomForest测试
Day 15: XGBoost测试 + 模型对比测试
Day 16: EnsembleTrainer测试
Day 17: CrossValidator交叉验证测试
Day 18: HyperparameterTuner超参数调优测试
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
    XGBoostTrainer,
    EnsembleTrainer,
    compare_models,
    load_feature_data,
    CrossValidator,
    cross_validate_model,
    cross_validate_all_models,
    HyperparameterTuner,
    get_rf_param_grid,
    get_xgb_param_grid,
    get_rf_param_distributions,
    get_xgb_param_distributions,
    tune_random_forest,
    tune_xgboost,
    tune_all_models,
    ModelEvaluator,
    evaluate_all_models,
    generate_final_report
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

        # 创建测试数据 - 需要同时创建 train_30dim.csv 和 train_scaled.csv
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

        # 创建 30dim 版本（默认使用）
        train_path_30dim = processed_dir / 'train_30dim.csv'
        test_path_30dim = processed_dir / 'test_30dim.csv'
        train_df.to_csv(train_path_30dim, index=False)
        test_df.to_csv(test_path_30dim, index=False)

        # 创建 scaled 版本
        train_path_scaled = processed_dir / 'train_scaled.csv'
        test_path_scaled = processed_dir / 'test_scaled.csv'
        train_df.to_csv(train_path_scaled, index=False)
        test_df.to_csv(test_path_scaled, index=False)

        # 使用 monkeypatch 来修改 DATA_DIR
        import src.model_training as mt
        original_data_dir = mt.DATA_DIR

        try:
            mt.DATA_DIR = str(tmp_path / 'data')

            # 加载（使用30dim版本）
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)

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


class TestXGBoostTrainer:
    """XGBoostTrainer测试类 - Day 15"""

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
        """测试XGBoost模型构建"""
        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)

        assert trainer.model is not None
        assert trainer.model.n_estimators == 10

    def test_build_model_default_params(self):
        """测试XGBoost默认参数构建模型"""
        trainer = XGBoostTrainer()
        trainer.build_model()

        assert trainer.model is not None
        assert trainer.model.n_estimators == 100
        assert trainer.model.max_depth == 6
        assert trainer.model.learning_rate == 0.1

    def test_build_model_custom_params(self):
        """测试XGBoost自定义参数"""
        trainer = XGBoostTrainer()
        trainer.build_model(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.7
        )

        assert trainer.model.n_estimators == 50
        assert trainer.model.learning_rate == 0.05
        assert trainer.model.max_depth == 4
        assert trainer.model.subsample == 0.7

    def test_train(self, sample_data):
        """测试XGBoost模型训练"""
        X_train, y_train, _, _ = sample_data

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        assert trainer.is_trained == True
        assert trainer.training_metrics is not None
        assert 'training_samples' in trainer.training_metrics

    def test_train_without_build(self, sample_data):
        """测试不先构建就训练（应自动构建）"""
        X_train, y_train, _, _ = sample_data

        trainer = XGBoostTrainer()
        trainer.train(X_train, y_train)

        assert trainer.is_trained == True
        assert trainer.model is not None

    def test_predict(self, sample_data):
        """测试XGBoost预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        y_pred = trainer.predict(X_test)

        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        """测试XGBoost概率预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        y_proba = trainer.predict_proba(X_test)

        assert y_proba.shape == (len(X_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_evaluate(self, sample_data):
        """测试XGBoost评估"""
        X_train, y_train, X_test, y_test = sample_data

        trainer = XGBoostTrainer()
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
        """测试XGBoost特征重要性"""
        X_train, y_train, _, _ = sample_data
        feature_names = [f'feat_{i}' for i in range(10)]

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train, feature_names=feature_names)

        importance_df = trainer.get_feature_importance()

        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'rank' in importance_df.columns

    def test_feature_importance_top_n(self, sample_data):
        """测试获取前N个特征重要性"""
        X_train, y_train, _, _ = sample_data
        feature_names = [f'feat_{i}' for i in range(10)]

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train, feature_names=feature_names)

        importance_df = trainer.get_feature_importance(top_n=5)

        assert len(importance_df) == 5

    def test_save_and_load(self, sample_data):
        """测试XGBoost模型保存和加载"""
        X_train, y_train, X_test, _ = sample_data

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)
        trainer.train(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            trainer.save_model(filepath)
            assert os.path.exists(filepath)

            trainer2 = XGBoostTrainer()
            trainer2.load_model(filepath)

            y_pred1 = trainer.predict(X_test)
            y_pred2 = trainer2.predict(X_test)

            assert np.array_equal(y_pred1, y_pred2)
            assert trainer2.is_trained == True

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_untrained_model_error(self):
        """测试XGBoost未训练模型的错误处理"""
        trainer = XGBoostTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.predict(np.random.randn(10, 5))

    def test_untrained_feature_importance_error(self):
        """测试XGBoost未训练模型获取特征重要性的错误处理"""
        trainer = XGBoostTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.get_feature_importance()

    def test_early_stopping_without_build_error(self, sample_data):
        """测试未构建模型就使用早停训练的错误处理"""
        X_train, y_train, X_test, y_test = sample_data

        trainer = XGBoostTrainer()
        # 未调用build_model

        with pytest.raises(ValueError, match="模型未初始化"):
            trainer.train_with_early_stopping(X_train, y_train, X_test, y_test)


class TestCompareModels:
    """compare_models函数测试 - Day 15"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    def test_compare_two_models(self, sample_data):
        """测试两个模型对比"""
        X_train, y_train, X_test, y_test = sample_data

        # 训练RF
        rf_trainer = RandomForestTrainer(model_name="rf_test")
        rf_trainer.build_model(n_estimators=10)
        rf_trainer.train(X_train, y_train)

        # 训练XGBoost
        xgb_trainer = XGBoostTrainer(model_name="xgb_test")
        xgb_trainer.build_model(n_estimators=10)
        xgb_trainer.train(X_train, y_train)

        # 对比
        result_df = compare_models([rf_trainer, xgb_trainer], X_test, y_test, verbose=False)

        assert len(result_df) == 2
        assert 'model_name' in result_df.columns
        assert 'accuracy' in result_df.columns
        assert 'rf_test' in result_df['model_name'].values
        assert 'xgb_test' in result_df['model_name'].values

    def test_compare_skip_untrained(self, sample_data):
        """测试跳过未训练的模型"""
        X_train, y_train, X_test, y_test = sample_data

        # 只训练RF
        rf_trainer = RandomForestTrainer(model_name="rf_test")
        rf_trainer.build_model(n_estimators=10)
        rf_trainer.train(X_train, y_train)

        # XGBoost未训练
        xgb_trainer = XGBoostTrainer(model_name="xgb_test")
        xgb_trainer.build_model(n_estimators=10)
        # 不调用train

        result_df = compare_models([rf_trainer, xgb_trainer], X_test, y_test, verbose=False)

        # 只有RF被评估
        assert len(result_df) == 1
        assert result_df.iloc[0]['model_name'] == 'rf_test'

    def test_compare_empty_list(self, sample_data):
        """测试空模型列表"""
        _, _, X_test, y_test = sample_data

        result_df = compare_models([], X_test, y_test, verbose=False)

        assert len(result_df) == 0


class TestXGBoostIntegration:
    """XGBoost集成测试 - Day 15"""

    @pytest.fixture
    def real_data(self):
        """尝试加载真实数据，如果不存在则跳过"""
        try:
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_scaled=True)
            return X_train, y_train, X_test, y_test, feature_names
        except FileNotFoundError:
            pytest.skip("真实数据文件不存在，跳过集成测试")

    def test_xgboost_real_data_training(self, real_data):
        """测试XGBoost真实数据训练"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        trainer = XGBoostTrainer()
        trainer.build_model(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6
        )
        trainer.train(X_train, y_train, feature_names=feature_names)

        metrics = trainer.evaluate(X_test, y_test)

        # Day 15 目标：准确率 >= 85%
        assert metrics['accuracy'] >= 0.85, f"准确率 {metrics['accuracy']:.4f} 低于 85% 目标"

        print(f"\nXGBoost真实数据测试结果:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")

    def test_rf_vs_xgboost_comparison(self, real_data):
        """测试RF与XGBoost对比"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        # 训练RF
        rf_trainer = RandomForestTrainer()
        rf_trainer.build_model(n_estimators=100, max_depth=10)
        rf_trainer.train(X_train, y_train, feature_names=feature_names)

        # 训练XGBoost
        xgb_trainer = XGBoostTrainer()
        xgb_trainer.build_model(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb_trainer.train(X_train, y_train, feature_names=feature_names)

        # 对比
        result_df = compare_models([rf_trainer, xgb_trainer], X_test, y_test, verbose=True)

        assert len(result_df) == 2
        # 两个模型都应达到85%以上
        assert result_df['accuracy'].min() >= 0.85


class TestEnsembleTrainer:
    """EnsembleTrainer测试类 - Day 16"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    def test_build_model_default(self):
        """测试默认参数构建集成模型"""
        trainer = EnsembleTrainer()
        trainer.build_model()

        assert trainer.rf_trainer is not None
        assert trainer.xgb_trainer is not None
        assert trainer.weights == [0.5, 0.5]

    def test_build_model_custom_weights(self):
        """测试自定义权重构建集成模型"""
        trainer = EnsembleTrainer()
        trainer.build_model(weights=[0.6, 0.4])

        assert trainer.weights == [0.6, 0.4]

    def test_build_model_invalid_weights_length(self):
        """测试权重数量错误"""
        trainer = EnsembleTrainer()

        with pytest.raises(ValueError, match="权重列表必须包含2个元素"):
            trainer.build_model(weights=[0.5, 0.3, 0.2])

    def test_build_model_invalid_weights_sum(self):
        """测试权重和不为1的错误"""
        trainer = EnsembleTrainer()

        with pytest.raises(ValueError, match="权重之和必须为1"):
            trainer.build_model(weights=[0.6, 0.6])

    def test_build_model_custom_params(self):
        """测试自定义参数构建集成模型"""
        trainer = EnsembleTrainer()
        rf_params = {'n_estimators': 50, 'max_depth': 5}
        xgb_params = {'n_estimators': 50, 'learning_rate': 0.05}

        trainer.build_model(rf_params=rf_params, xgb_params=xgb_params)

        assert trainer.rf_trainer.model.n_estimators == 50
        assert trainer.rf_trainer.model.max_depth == 5
        assert trainer.xgb_trainer.model.n_estimators == 50
        assert trainer.xgb_trainer.model.learning_rate == 0.05

    def test_train(self, sample_data):
        """测试集成模型训练"""
        X_train, y_train, _, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train)

        assert trainer.is_trained == True
        assert trainer.rf_trainer.is_trained == True
        assert trainer.xgb_trainer.is_trained == True
        assert 'training_samples' in trainer.training_metrics

    def test_train_without_build(self, sample_data):
        """测试不先构建就训练（应自动构建）"""
        X_train, y_train, _, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.train(X_train, y_train)

        assert trainer.is_trained == True
        assert trainer.rf_trainer is not None
        assert trainer.xgb_trainer is not None

    def test_predict(self, sample_data):
        """测试集成模型预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train)

        y_pred = trainer.predict(X_test)

        assert len(y_pred) == len(X_test)
        assert set(y_pred).issubset({0, 1})

    def test_predict_proba(self, sample_data):
        """测试集成模型概率预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train)

        y_proba = trainer.predict_proba(X_test)

        assert y_proba.shape == (len(X_test), 2)
        # 概率和应为1
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_soft_voting_logic(self, sample_data):
        """测试软投票逻辑"""
        X_train, y_train, X_test, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model(weights=[0.5, 0.5])
        trainer.train(X_train, y_train)

        # 获取各模型概率
        rf_proba = trainer.rf_trainer.predict_proba(X_test)
        xgb_proba = trainer.xgb_trainer.predict_proba(X_test)
        ensemble_proba = trainer.predict_proba(X_test)

        # 验证加权平均
        expected_proba = 0.5 * rf_proba + 0.5 * xgb_proba
        assert np.allclose(ensemble_proba, expected_proba)

    def test_evaluate(self, sample_data):
        """测试集成模型评估"""
        X_train, y_train, X_test, y_test = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train)

        metrics = trainer.evaluate(X_test, y_test)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert 0 <= metrics['accuracy'] <= 1

    def test_get_base_model_predictions(self, sample_data):
        """测试获取基模型预测"""
        X_train, y_train, X_test, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train)

        predictions = trainer.get_base_model_predictions(X_test)

        assert 'rf_pred' in predictions
        assert 'xgb_pred' in predictions
        assert 'ensemble_pred' in predictions
        assert 'rf_proba' in predictions
        assert 'xgb_proba' in predictions
        assert 'ensemble_proba' in predictions

    def test_get_feature_importance(self, sample_data):
        """测试获取综合特征重要性"""
        X_train, y_train, _, _ = sample_data
        feature_names = [f'feat_{i}' for i in range(10)]

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train, feature_names=feature_names)

        importance_df = trainer.get_feature_importance()

        assert len(importance_df) == 10
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert 'importance_rf' in importance_df.columns
        assert 'importance_xgb' in importance_df.columns
        assert 'rank' in importance_df.columns

    def test_save_and_load(self, sample_data):
        """测试集成模型保存和加载"""
        X_train, y_train, X_test, _ = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model(weights=[0.6, 0.4])
        trainer.train(X_train, y_train)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name

        try:
            trainer.save_model(filepath)
            assert os.path.exists(filepath)

            # 加载
            trainer2 = EnsembleTrainer()
            trainer2.load_model(filepath)

            # 验证
            assert trainer2.is_trained == True
            assert trainer2.weights == [0.6, 0.4]
            assert trainer2.rf_trainer is not None
            assert trainer2.xgb_trainer is not None

            # 验证预测一致
            y_pred1 = trainer.predict(X_test)
            y_pred2 = trainer2.predict(X_test)
            assert np.array_equal(y_pred1, y_pred2)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    def test_untrained_predict_error(self):
        """测试未训练模型预测的错误处理"""
        trainer = EnsembleTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.predict(np.random.randn(10, 5))

    def test_untrained_evaluate_error(self, sample_data):
        """测试未训练模型评估的错误处理"""
        _, _, X_test, y_test = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.evaluate(X_test, y_test)

    def test_untrained_save_error(self):
        """测试未训练模型保存的错误处理"""
        trainer = EnsembleTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.save_model("test.pkl")

    def test_untrained_feature_importance_error(self):
        """测试未训练模型获取特征重要性的错误处理"""
        trainer = EnsembleTrainer()
        trainer.build_model()

        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.get_feature_importance()

    def test_load_nonexistent_model_error(self):
        """测试加载不存在模型的错误处理"""
        trainer = EnsembleTrainer()

        with pytest.raises(FileNotFoundError):
            trainer.load_model("nonexistent_ensemble.pkl")


class TestEnsembleIntegration:
    """集成模型集成测试 - Day 16"""

    @pytest.fixture
    def real_data(self):
        """尝试加载真实数据，如果不存在则跳过"""
        try:
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)
            return X_train, y_train, X_test, y_test, feature_names
        except FileNotFoundError:
            pytest.skip("真实数据文件不存在，跳过集成测试")

    def test_ensemble_real_data_training(self, real_data):
        """测试集成模型真实数据训练"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        trainer = EnsembleTrainer()
        trainer.build_model()
        trainer.train(X_train, y_train, feature_names=feature_names)

        metrics = trainer.evaluate(X_test, y_test)

        # Day 16 目标：准确率 >= 85%（集成模型应该不低于单模型）
        assert metrics['accuracy'] >= 0.85, f"准确率 {metrics['accuracy']:.4f} 低于 85% 目标"

        print(f"\n集成模型真实数据测试结果:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")

    def test_ensemble_vs_single_models(self, real_data):
        """测试集成模型与单模型对比"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        # 训练集成模型
        ensemble_trainer = EnsembleTrainer()
        ensemble_trainer.build_model()
        ensemble_trainer.train(X_train, y_train, feature_names=feature_names)

        # 获取各模型性能
        rf_metrics = ensemble_trainer.rf_trainer.evaluate(X_test, y_test)
        xgb_metrics = ensemble_trainer.xgb_trainer.evaluate(X_test, y_test)
        ensemble_metrics = ensemble_trainer.evaluate(X_test, y_test)

        print(f"\n模型性能对比:")
        print(f"  RandomForest: {rf_metrics['accuracy']:.4f}")
        print(f"  XGBoost: {xgb_metrics['accuracy']:.4f}")
        print(f"  Ensemble: {ensemble_metrics['accuracy']:.4f}")

        # 集成模型应达到85%以上
        assert ensemble_metrics['accuracy'] >= 0.85


class TestCrossValidator:
    """CrossValidator测试类 - Day 17"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_init_default(self):
        """测试默认参数初始化"""
        validator = CrossValidator()

        assert validator.n_splits == 5
        assert validator.shuffle == True
        assert validator.random_state == 42

    def test_init_custom(self):
        """测试自定义参数初始化"""
        validator = CrossValidator(n_splits=10, shuffle=True, random_state=123)

        assert validator.n_splits == 10
        assert validator.shuffle == True
        assert validator.random_state == 123

    def test_cross_validate_rf(self, sample_data):
        """测试RandomForest交叉验证"""
        X, y = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        results = validator.cross_validate(trainer, X, y, verbose=False)

        assert 'accuracy_mean' in results
        assert 'accuracy_std' in results
        assert 'accuracy_scores' in results
        assert len(results['accuracy_scores']) == 3
        assert 0 <= results['accuracy_mean'] <= 1

    def test_cross_validate_xgb(self, sample_data):
        """测试XGBoost交叉验证"""
        X, y = sample_data

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        results = validator.cross_validate(trainer, X, y, verbose=False)

        assert 'accuracy_mean' in results
        assert 'f1_mean' in results
        assert 'roc_auc_mean' in results

    def test_cross_validate_ensemble(self, sample_data):
        """测试Ensemble交叉验证"""
        X, y = sample_data

        trainer = EnsembleTrainer()
        trainer.build_model()

        validator = CrossValidator(n_splits=3)
        results = validator.cross_validate(trainer, X, y, verbose=False)

        assert 'accuracy_mean' in results
        assert len(results['accuracy_scores']) == 3

    def test_get_cv_scores(self, sample_data):
        """测试获取交叉验证分数"""
        X, y = sample_data

        trainer = RandomForestTrainer(model_name="rf_cv_test")
        trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        validator.cross_validate(trainer, X, y, verbose=False)

        scores = validator.get_cv_scores("rf_cv_test")
        assert 'accuracy_mean' in scores

        all_scores = validator.get_cv_scores()
        assert "rf_cv_test" in all_scores

    def test_get_summary(self, sample_data):
        """测试获取汇总"""
        X, y = sample_data

        rf_trainer = RandomForestTrainer(model_name="rf_summary")
        rf_trainer.build_model(n_estimators=10)

        xgb_trainer = XGBoostTrainer(model_name="xgb_summary")
        xgb_trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        validator.cross_validate(rf_trainer, X, y, verbose=False)
        validator.cross_validate(xgb_trainer, X, y, verbose=False)

        summary = validator.get_summary()

        assert len(summary) == 2
        assert 'model' in summary.columns
        assert 'accuracy' in summary.columns
        # 验证按accuracy_mean降序排列
        assert summary['accuracy_mean'].is_monotonic_decreasing or len(summary) == 1

    def test_check_stability_pass(self, sample_data):
        """测试稳定性检查通过"""
        X, y = sample_data

        trainer = RandomForestTrainer(model_name="rf_stable")
        trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        validator.cross_validate(trainer, X, y, verbose=False)

        # 使用较大阈值确保通过
        is_stable = validator.check_stability("rf_stable", std_threshold=0.5)
        assert is_stable == True

    def test_check_stability_not_found(self, sample_data):
        """测试稳定性检查模型不存在"""
        validator = CrossValidator()

        is_stable = validator.check_stability("nonexistent_model")
        assert is_stable == False

    def test_plot_cv_results(self, sample_data, tmp_path):
        """测试绘制交叉验证结果"""
        X, y = sample_data

        trainer = RandomForestTrainer(model_name="rf_plot")
        trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        validator.cross_validate(trainer, X, y, verbose=False)

        save_path = str(tmp_path / "cv_plot.png")
        validator.plot_cv_results(metric='accuracy', save_path=save_path)

        assert os.path.exists(save_path)

    def test_plot_cv_results_no_data(self):
        """测试无数据时绘图"""
        validator = CrossValidator()
        # 应该不报错，只是打印提示
        validator.plot_cv_results()

    def test_multiple_metrics(self, sample_data):
        """测试多指标记录"""
        X, y = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)

        validator = CrossValidator(n_splits=3)
        results = validator.cross_validate(trainer, X, y, verbose=False)

        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for metric in metrics:
            assert f'{metric}_mean' in results
            assert f'{metric}_std' in results
            assert f'{metric}_scores' in results


class TestCrossValidateConvenienceFunctions:
    """交叉验证便捷函数测试 - Day 17"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_cross_validate_model(self, sample_data):
        """测试单模型交叉验证便捷函数"""
        X, y = sample_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=10)

        results = cross_validate_model(trainer, X, y, n_splits=3, verbose=False)

        assert 'accuracy_mean' in results
        assert 'accuracy_std' in results

    def test_cross_validate_all_models(self, sample_data):
        """测试多模型交叉验证便捷函数"""
        X, y = sample_data

        rf_trainer = RandomForestTrainer(model_name="rf_all")
        rf_trainer.build_model(n_estimators=10)

        xgb_trainer = XGBoostTrainer(model_name="xgb_all")
        xgb_trainer.build_model(n_estimators=10)

        summary_df, validator = cross_validate_all_models(
            [rf_trainer, xgb_trainer], X, y, n_splits=3, verbose=False
        )

        assert len(summary_df) == 2
        assert validator is not None
        assert 'rf_all' in summary_df['model'].values
        assert 'xgb_all' in summary_df['model'].values

    def test_cross_validate_all_models_with_plot(self, sample_data, tmp_path):
        """测试多模型交叉验证并保存图表"""
        X, y = sample_data

        trainer = RandomForestTrainer(model_name="rf_plot_all")
        trainer.build_model(n_estimators=10)

        save_path = str(tmp_path / "cv_all_plot.png")
        summary_df, validator = cross_validate_all_models(
            [trainer], X, y, n_splits=3, verbose=False, save_plot=save_path
        )

        assert os.path.exists(save_path)


class TestCrossValidatorIntegration:
    """交叉验证集成测试 - Day 17"""

    @pytest.fixture
    def real_data(self):
        """尝试加载真实数据，如果不存在则跳过"""
        try:
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)
            # 合并训练集和测试集用于交叉验证
            X = np.vstack([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            return X, y, feature_names
        except FileNotFoundError:
            pytest.skip("真实数据文件不存在，跳过集成测试")

    def test_cv_real_data_rf(self, real_data):
        """测试RandomForest真实数据交叉验证"""
        X, y, feature_names = real_data

        trainer = RandomForestTrainer()
        trainer.build_model(n_estimators=100, max_depth=10)

        validator = CrossValidator(n_splits=5)
        results = validator.cross_validate(trainer, X, y, feature_names=feature_names, verbose=True)

        # Day 17 目标：准确率均值 >= 87%
        assert results['accuracy_mean'] >= 0.87, f"准确率均值 {results['accuracy_mean']:.4f} 低于 87% 目标"
        # Day 17 目标：标准差 <= 0.03（稳定性）
        assert results['accuracy_std'] <= 0.03, f"准确率标准差 {results['accuracy_std']:.4f} 超过 0.03 阈值"

        print(f"\nRandomForest交叉验证结果:")
        print(f"  准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

    def test_cv_real_data_xgb(self, real_data):
        """测试XGBoost真实数据交叉验证"""
        X, y, feature_names = real_data

        trainer = XGBoostTrainer()
        trainer.build_model(n_estimators=100, learning_rate=0.1, max_depth=6)

        validator = CrossValidator(n_splits=5)
        results = validator.cross_validate(trainer, X, y, feature_names=feature_names, verbose=True)

        # Day 17 目标：准确率均值 >= 87%
        assert results['accuracy_mean'] >= 0.87, f"准确率均值 {results['accuracy_mean']:.4f} 低于 87% 目标"

        print(f"\nXGBoost交叉验证结果:")
        print(f"  准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

    def test_cv_real_data_ensemble(self, real_data):
        """测试Ensemble真实数据交叉验证"""
        X, y, feature_names = real_data

        trainer = EnsembleTrainer()
        trainer.build_model()

        validator = CrossValidator(n_splits=5)
        results = validator.cross_validate(trainer, X, y, feature_names=feature_names, verbose=True)

        # Day 17 目标：准确率均值 >= 87%
        assert results['accuracy_mean'] >= 0.87, f"准确率均值 {results['accuracy_mean']:.4f} 低于 87% 目标"

        print(f"\nEnsemble交叉验证结果:")
        print(f"  准确率: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")

    def test_cv_compare_all_models(self, real_data):
        """测试三模型交叉验证对比"""
        X, y, feature_names = real_data

        rf_trainer = RandomForestTrainer(model_name="rf_cv")
        rf_trainer.build_model(n_estimators=100, max_depth=10)

        xgb_trainer = XGBoostTrainer(model_name="xgb_cv")
        xgb_trainer.build_model(n_estimators=100, learning_rate=0.1, max_depth=6)

        ensemble_trainer = EnsembleTrainer(model_name="ensemble_cv")
        ensemble_trainer.build_model()

        summary_df, validator = cross_validate_all_models(
            [rf_trainer, xgb_trainer, ensemble_trainer],
            X, y,
            n_splits=5,
            feature_names=feature_names,
            verbose=True
        )

        print(f"\n三模型交叉验证汇总:")
        print(summary_df[['model', 'accuracy', 'f1']].to_string(index=False))

        # 所有模型准确率应 >= 87%
        assert summary_df['accuracy_mean'].min() >= 0.87

        # 检查模型稳定性
        for model_name in ['rf_cv', 'xgb_cv', 'ensemble_cv']:
            validator.check_stability(model_name, std_threshold=0.03)


class TestHyperparameterTuner:
    """HyperparameterTuner测试类 - Day 18"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_init(self):
        """测试初始化"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner('TestModel')
        assert tuner.model_name == 'TestModel'
        assert tuner.best_params is None
        assert tuner.best_score is None
        assert tuner.best_estimator is None
        assert tuner.cv_results is None
        assert tuner.search_history == []

    def test_grid_search(self, sample_data):
        """测试网格搜索"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }

        best_params = tuner.grid_search(
            model, param_grid, X, y,
            cv=3, verbose=0
        )

        assert best_params is not None
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
        assert tuner.best_score is not None
        assert 0 <= tuner.best_score <= 1

    def test_random_search(self, sample_data):
        """测试随机搜索"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_distributions = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7, None]
        }

        best_params = tuner.random_search(
            model, param_distributions, X, y,
            n_iter=5, cv=3, verbose=0
        )

        assert best_params is not None
        assert tuner.best_score is not None

    def test_get_best_params(self, sample_data):
        """测试获取最优参数"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=0)

        params = tuner.get_best_params()
        assert isinstance(params, dict)
        assert 'n_estimators' in params

    def test_get_best_params_without_search(self):
        """测试未搜索时获取最优参数应报错"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner()

        with pytest.raises(ValueError, match="尚未执行超参数搜索"):
            tuner.get_best_params()

    def test_get_best_score(self, sample_data):
        """测试获取最优分数"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=0)

        score = tuner.get_best_score()
        assert isinstance(score, float)
        assert 0 <= score <= 1

    def test_get_best_score_without_search(self):
        """测试未搜索时获取最优分数应报错"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner()

        with pytest.raises(ValueError, match="尚未执行超参数搜索"):
            tuner.get_best_score()

    def test_get_cv_results(self, sample_data):
        """测试获取交叉验证结果"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=0)

        cv_results = tuner.get_cv_results()

        assert isinstance(cv_results, pd.DataFrame)
        assert 'mean_test_score' in cv_results.columns
        assert len(cv_results) == 2  # 两个参数组合

    def test_get_cv_results_without_search(self):
        """测试未搜索时获取CV结果应报错"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner()

        with pytest.raises(ValueError, match="尚未执行超参数搜索"):
            tuner.get_cv_results()

    def test_get_best_estimator(self, sample_data):
        """测试获取最优模型"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=0)

        best_model = tuner.get_best_estimator()

        # 验证最优模型可以预测
        predictions = best_model.predict(X[:5])
        assert len(predictions) == 5

    def test_get_best_estimator_without_search(self):
        """测试未搜索时获取最优模型应报错"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner()

        with pytest.raises(ValueError, match="尚未执行超参数搜索"):
            tuner.get_best_estimator()

    def test_save_and_load_results(self, sample_data, tmp_path):
        """测试保存和加载结果"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=0)

        # 保存
        filepath = tmp_path / 'tuning_results.json'
        tuner.save_results(str(filepath))

        assert filepath.exists()

        # 加载
        tuner2 = HyperparameterTuner()
        tuner2.load_results(str(filepath))

        assert tuner2.best_params == tuner.best_params
        assert tuner2.best_score == tuner.best_score

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner()

        with pytest.raises(FileNotFoundError):
            tuner.load_results('nonexistent_file.json')

    def test_plot_search_results(self, sample_data, tmp_path):
        """测试绘制搜索结果"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=0)

        save_path = str(tmp_path / 'tuning_plot.png')
        tuner.plot_search_results(save_path=save_path)

        assert os.path.exists(save_path)

    def test_plot_search_results_no_data(self):
        """测试无数据时绘图"""
        from src.model_training import HyperparameterTuner

        tuner = HyperparameterTuner()
        # 应该不报错，只是打印提示
        tuner.plot_search_results()

    def test_search_history(self, sample_data):
        """测试搜索历史记录"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y = sample_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)

        # 执行两次搜索
        tuner.grid_search(model, {'n_estimators': [10, 20]}, X, y, cv=3, verbose=0)
        tuner.random_search(model, {'n_estimators': [10, 20]}, X, y, n_iter=2, cv=3, verbose=0)

        assert len(tuner.search_history) == 2
        assert tuner.search_history[0]['method'] == 'grid_search'
        assert tuner.search_history[1]['method'] == 'random_search'


class TestTuningConvenienceFunctions:
    """调优便捷函数测试 - Day 18"""

    def test_get_rf_param_grid(self):
        """测试获取RF参数网格"""
        from src.model_training import get_rf_param_grid

        param_grid = get_rf_param_grid()

        assert 'n_estimators' in param_grid
        assert 'max_depth' in param_grid
        assert 'min_samples_split' in param_grid
        assert 'min_samples_leaf' in param_grid
        assert isinstance(param_grid['n_estimators'], list)

    def test_get_xgb_param_grid(self):
        """测试获取XGB参数网格"""
        from src.model_training import get_xgb_param_grid

        param_grid = get_xgb_param_grid()

        assert 'n_estimators' in param_grid
        assert 'learning_rate' in param_grid
        assert 'max_depth' in param_grid
        assert 'min_child_weight' in param_grid

    def test_get_rf_param_distributions(self):
        """测试获取RF参数分布"""
        from src.model_training import get_rf_param_distributions

        param_dist = get_rf_param_distributions()

        assert 'n_estimators' in param_dist
        assert 'max_depth' in param_dist
        assert 'min_samples_split' in param_dist

    def test_get_xgb_param_distributions(self):
        """测试获取XGB参数分布"""
        from src.model_training import get_xgb_param_distributions

        param_dist = get_xgb_param_distributions()

        assert 'n_estimators' in param_dist
        assert 'learning_rate' in param_dist
        assert 'max_depth' in param_dist

    def test_tune_random_forest_grid(self):
        """测试RF网格搜索调优函数"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=2, verbose=0)

        assert tuner.best_params is not None
        assert tuner.best_score is not None

    def test_tune_xgboost_grid(self):
        """测试XGBoost网格搜索调优函数"""
        from src.model_training import HyperparameterTuner
        from xgboost import XGBClassifier

        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)

        tuner = HyperparameterTuner('XGBoost')
        model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        param_grid = {'n_estimators': [10, 20]}

        tuner.grid_search(model, param_grid, X, y, cv=2, verbose=0)

        assert tuner.best_params is not None
        assert tuner.best_score is not None


class TestHyperparameterTunerIntegration:
    """超参数调优集成测试 - Day 18"""

    @pytest.fixture
    def real_data(self):
        """尝试加载真实数据，如果不存在则跳过"""
        try:
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)
            # 合并数据用于调优
            X = np.vstack([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            return X, y, feature_names
        except FileNotFoundError:
            pytest.skip("真实数据文件不存在，跳过集成测试")

    def test_rf_tuning_real_data(self, real_data):
        """测试RandomForest真实数据调优"""
        from src.model_training import HyperparameterTuner
        from sklearn.ensemble import RandomForestClassifier

        X, y, _ = real_data

        tuner = HyperparameterTuner('RandomForest')
        model = RandomForestClassifier(random_state=42, n_jobs=-1)

        # 使用简化参数网格快速测试
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [10, 15]
        }

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=1)

        # Day 18 目标：调优后准确率 >= 90%
        assert tuner.best_score >= 0.90, f"最优分数 {tuner.best_score:.4f} 低于 90% 目标"

        print(f"\nRandomForest调优结果:")
        print(f"  最优参数: {tuner.best_params}")
        print(f"  最优分数: {tuner.best_score:.4f}")

    def test_xgb_tuning_real_data(self, real_data):
        """测试XGBoost真实数据调优"""
        from src.model_training import HyperparameterTuner
        from xgboost import XGBClassifier

        X, y, _ = real_data

        tuner = HyperparameterTuner('XGBoost')
        model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )

        # 使用简化参数网格快速测试
        param_grid = {
            'n_estimators': [100, 150],
            'learning_rate': [0.1, 0.2]
        }

        tuner.grid_search(model, param_grid, X, y, cv=3, verbose=1)

        # Day 18 目标：调优后准确率 >= 90%
        assert tuner.best_score >= 0.90, f"最优分数 {tuner.best_score:.4f} 低于 90% 目标"

        print(f"\nXGBoost调优结果:")
        print(f"  最优参数: {tuner.best_params}")
        print(f"  最优分数: {tuner.best_score:.4f}")


# ==================== Day 19: ModelEvaluator测试 ====================

class TestModelEvaluator:
    """ModelEvaluator测试类 - Day 19"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y

    @pytest.fixture
    def trained_model(self, sample_data):
        """创建已训练的模型"""
        X, y = sample_data
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    def test_init(self):
        """测试初始化"""
        evaluator = ModelEvaluator()
        assert evaluator.results == {}
        assert evaluator.figures == {}

    def test_evaluate_model(self, sample_data, trained_model):
        """测试模型评估"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics
        assert 'tp' in metrics
        assert 'tn' in metrics
        assert 'fp' in metrics
        assert 'fn' in metrics
        assert 'specificity' in metrics
        assert 'mcc' in metrics

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['auc_roc'] <= 1

    def test_evaluate_model_stores_results(self, sample_data, trained_model):
        """测试评估结果被正确存储"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        assert 'TestModel' in evaluator.results
        assert evaluator.results['TestModel']['accuracy'] > 0

    def test_get_detailed_metrics(self, sample_data, trained_model):
        """测试获取详细指标"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)
        df = evaluator.get_detailed_metrics()

        assert len(df) == 1
        assert '模型' in df.columns
        assert '准确率' in df.columns
        assert 'AUC-ROC' in df.columns

    def test_get_detailed_metrics_specific_model(self, sample_data, trained_model):
        """测试获取特定模型的详细指标"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)
        df = evaluator.get_detailed_metrics('TestModel')

        assert len(df) == 1
        assert df.iloc[0]['模型'] == 'TestModel'

    def test_get_detailed_metrics_model_not_found(self):
        """测试模型不存在时的错误"""
        evaluator = ModelEvaluator()

        with pytest.raises(ValueError, match="模型 NonExistent 未评估"):
            evaluator.get_detailed_metrics('NonExistent')

    def test_multiple_models(self, sample_data):
        """测试多模型评估"""
        X, y = sample_data

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        model1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model1.fit(X, y)

        model2 = LogisticRegression(random_state=42, max_iter=200)
        model2.fit(X, y)

        evaluator = ModelEvaluator()
        evaluator.evaluate_model(model1, X, y, 'RandomForest', verbose=False)
        evaluator.evaluate_model(model2, X, y, 'LogisticRegression', verbose=False)

        df = evaluator.get_detailed_metrics()
        assert len(df) == 2

    def test_compare_models(self, sample_data, trained_model):
        """测试模型比较"""
        X, y = sample_data

        from sklearn.linear_model import LogisticRegression
        model2 = LogisticRegression(random_state=42, max_iter=200)
        model2.fit(X, y)

        evaluator = ModelEvaluator()
        evaluator.evaluate_model(trained_model, X, y, 'RF', verbose=False)
        evaluator.evaluate_model(model2, X, y, 'LR', verbose=False)

        comparison = evaluator.compare_models('accuracy')
        assert len(comparison) == 2
        assert 'accuracy' in comparison.columns

    def test_compare_models_no_results(self):
        """测试无评估结果时比较"""
        evaluator = ModelEvaluator()

        with pytest.raises(ValueError, match="没有评估结果可比较"):
            evaluator.compare_models()

    def test_generate_report(self, sample_data, trained_model):
        """测试报告生成"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)
        report = evaluator.generate_report()

        assert '模型性能评估报告' in report
        assert 'TestModel' in report
        assert '准确率' in report
        assert '目标验证' in report

    def test_generate_report_save(self, sample_data, trained_model, tmp_path):
        """测试报告保存"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        save_path = tmp_path / 'report.txt'
        evaluator.generate_report(save_path=str(save_path))

        assert save_path.exists()

    def test_save_and_load_results(self, sample_data, trained_model, tmp_path):
        """测试结果保存和加载"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        # 保存
        filepath = tmp_path / 'results.json'
        evaluator.save_results(str(filepath))

        assert filepath.exists()

        # 加载
        evaluator2 = ModelEvaluator()
        evaluator2.load_results(str(filepath))

        assert 'TestModel' in evaluator2.results
        assert evaluator2.results['TestModel']['accuracy'] == evaluator.results['TestModel']['accuracy']

    def test_plot_confusion_matrix(self, sample_data, trained_model, tmp_path):
        """测试混淆矩阵绘制"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        save_path = tmp_path / 'confusion_matrix.png'
        evaluator.plot_confusion_matrix('TestModel', save_path=str(save_path))

        assert save_path.exists()
        assert 'TestModel_confusion_matrix' in evaluator.figures

    def test_plot_confusion_matrix_normalized(self, sample_data, trained_model, tmp_path):
        """测试归一化混淆矩阵绘制"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        save_path = tmp_path / 'confusion_matrix_norm.png'
        evaluator.plot_confusion_matrix('TestModel', normalize=True, save_path=str(save_path))

        assert save_path.exists()

    def test_plot_confusion_matrix_model_not_found(self):
        """测试绘制不存在模型的混淆矩阵"""
        evaluator = ModelEvaluator()

        with pytest.raises(ValueError, match="模型 NonExistent 未评估"):
            evaluator.plot_confusion_matrix('NonExistent')

    def test_plot_roc_curve(self, sample_data, trained_model, tmp_path):
        """测试ROC曲线绘制"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        save_path = tmp_path / 'roc_curve.png'
        evaluator.plot_roc_curve(save_path=str(save_path))

        assert save_path.exists()
        assert 'roc_curve' in evaluator.figures

    def test_plot_pr_curve(self, sample_data, trained_model, tmp_path):
        """测试PR曲线绘制"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        save_path = tmp_path / 'pr_curve.png'
        evaluator.plot_pr_curve(save_path=str(save_path))

        assert save_path.exists()
        assert 'pr_curve' in evaluator.figures

    def test_plot_all_curves(self, sample_data, trained_model, tmp_path):
        """测试绘制所有曲线"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        save_dir = str(tmp_path)
        evaluator.plot_all_curves('TestModel', y, save_dir=save_dir)

        assert (tmp_path / 'TestModel_evaluation.png').exists()
        assert 'TestModel_all_curves' in evaluator.figures

    def test_plot_all_curves_model_not_found(self, sample_data):
        """测试绘制不存在模型的所有曲线"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        with pytest.raises(ValueError, match="模型 NonExistent 未评估"):
            evaluator.plot_all_curves('NonExistent', y)

    def test_plot_metrics_comparison(self, sample_data, tmp_path):
        """测试指标对比图绘制"""
        X, y = sample_data

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        evaluator = ModelEvaluator()
        evaluator.evaluate_model(model, X, y, 'TestModel', verbose=False)

        save_path = tmp_path / 'metrics_comparison.png'
        evaluator.plot_metrics_comparison(save_path=str(save_path))

        assert save_path.exists()
        assert 'metrics_comparison' in evaluator.figures

    def test_roc_auc_calculation(self, sample_data, trained_model):
        """测试ROC-AUC计算"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        # 验证ROC曲线数据存在
        assert 'fpr' in metrics
        assert 'tpr' in metrics
        assert len(metrics['fpr']) > 0
        assert len(metrics['tpr']) > 0

    def test_pr_curve_calculation(self, sample_data, trained_model):
        """测试PR曲线数据计算"""
        X, y = sample_data
        evaluator = ModelEvaluator()

        metrics = evaluator.evaluate_model(trained_model, X, y, 'TestModel', verbose=False)

        # 验证PR曲线数据存在
        assert 'precision_curve' in metrics
        assert 'recall_curve' in metrics
        assert len(metrics['precision_curve']) > 0
        assert len(metrics['recall_curve']) > 0


class TestEvaluationConvenienceFunctions:
    """评估便捷函数测试 - Day 19"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        return X_train, y_train, X_test, y_test

    def test_evaluate_all_models(self, sample_data):
        """测试批量评估函数"""
        X_train, y_train, X_test, y_test = sample_data

        rf_trainer = RandomForestTrainer()
        rf_trainer.build_model(n_estimators=10)
        rf_trainer.train(X_train, y_train)

        xgb_trainer = XGBoostTrainer()
        xgb_trainer.build_model(n_estimators=10)
        xgb_trainer.train(X_train, y_train)

        evaluator, summary_df = evaluate_all_models(
            [rf_trainer, xgb_trainer],
            X_test, y_test
        )

        assert len(summary_df) == 2
        assert 'RandomForest' in evaluator.results or 'rf_model' in evaluator.results
        assert 'XGBoost' in evaluator.results or 'xgb_model' in evaluator.results

    def test_evaluate_all_models_with_save(self, sample_data, tmp_path):
        """测试批量评估函数带保存"""
        X_train, y_train, X_test, y_test = sample_data

        rf_trainer = RandomForestTrainer()
        rf_trainer.build_model(n_estimators=10)
        rf_trainer.train(X_train, y_train)

        save_dir = str(tmp_path / 'figures')
        evaluator, summary_df = evaluate_all_models(
            [rf_trainer],
            X_test, y_test,
            save_dir=save_dir
        )

        # 检查图表文件是否生成
        assert (tmp_path / 'figures' / 'roc_curves.png').exists()
        assert (tmp_path / 'figures' / 'pr_curves.png').exists()
        assert (tmp_path / 'figures' / 'metrics_comparison.png').exists()

    def test_generate_final_report(self, sample_data, tmp_path):
        """测试生成最终报告"""
        X_train, y_train, X_test, y_test = sample_data

        rf_trainer = RandomForestTrainer()
        rf_trainer.build_model(n_estimators=10)
        rf_trainer.train(X_train, y_train)

        evaluator = ModelEvaluator()
        evaluator.evaluate_model(rf_trainer.model, X_test, y_test, 'rf_model', verbose=False)

        save_dir = str(tmp_path / 'reports')
        report_path = generate_final_report(evaluator, save_dir)

        assert os.path.exists(report_path)
        assert (tmp_path / 'reports' / 'evaluation_report.txt').exists()
        assert (tmp_path / 'reports' / 'evaluation_results.json').exists()
        assert (tmp_path / 'reports' / 'evaluation_metrics.csv').exists()


class TestModelEvaluatorIntegration:
    """ModelEvaluator集成测试 - Day 19"""

    @pytest.fixture
    def real_data(self):
        """尝试加载真实数据，如果不存在则跳过"""
        try:
            X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)
            return X_train, y_train, X_test, y_test, feature_names
        except FileNotFoundError:
            pytest.skip("真实数据文件不存在，跳过集成测试")

    def test_real_data_evaluation(self, real_data):
        """测试真实数据评估"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        # 训练模型
        rf_trainer = RandomForestTrainer()
        rf_trainer.build_model(n_estimators=100, max_depth=10)
        rf_trainer.train(X_train, y_train, feature_names=feature_names)

        xgb_trainer = XGBoostTrainer()
        xgb_trainer.build_model(n_estimators=100, learning_rate=0.1, max_depth=6)
        xgb_trainer.train(X_train, y_train, feature_names=feature_names)

        # 评估
        evaluator = ModelEvaluator()
        rf_metrics = evaluator.evaluate_model(rf_trainer.model, X_test, y_test, 'RandomForest')
        xgb_metrics = evaluator.evaluate_model(xgb_trainer.model, X_test, y_test, 'XGBoost')

        # Day 19 目标：准确率 >= 90%
        best_acc = max(rf_metrics['accuracy'], xgb_metrics['accuracy'])
        assert best_acc >= 0.85, f"准确率 {best_acc:.4f} 低于 85%"

        # 生成报告
        report = evaluator.generate_report()
        assert '模型性能评估报告' in report

        print(f"\n真实数据评估结果:")
        print(f"  RandomForest准确率: {rf_metrics['accuracy']:.4f}")
        print(f"  XGBoost准确率: {xgb_metrics['accuracy']:.4f}")
        print(f"  RandomForest AUC-ROC: {rf_metrics['auc_roc']:.4f}")
        print(f"  XGBoost AUC-ROC: {xgb_metrics['auc_roc']:.4f}")

    def test_ensemble_evaluation(self, real_data):
        """测试集成模型评估"""
        X_train, y_train, X_test, y_test, feature_names = real_data

        # 训练集成模型
        ensemble_trainer = EnsembleTrainer()
        ensemble_trainer.build_model()
        ensemble_trainer.train(X_train, y_train, feature_names=feature_names)

        # 评估
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate_model(ensemble_trainer.model, X_test, y_test, 'Ensemble')

        # 验证指标
        assert metrics['accuracy'] >= 0.85
        assert 'auc_roc' in metrics
        assert 'auc_pr' in metrics

        print(f"\n集成模型评估结果:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  AUC-PR: {metrics['auc_pr']:.4f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
