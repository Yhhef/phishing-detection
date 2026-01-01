"""
Day 20 验证脚本 - 模型保存与阶段验收

验证内容：
1. ModelManager类功能
2. PhishingPredictor类功能
3. 阶段三验收
4. 阶段三报告生成
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_training import (
    ModelManager,
    PhishingPredictor,
    RandomForestTrainer,
    XGBoostTrainer,
    EnsembleTrainer,
    load_feature_data,
    validate_phase3,
    generate_phase3_report
)
import numpy as np

def main():
    print("=" * 70)
    print("Day 20 模型保存与阶段验收验证")
    print("=" * 70)

    # 加载数据
    print("\n[步骤1] 加载数据...")
    try:
        X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)
        print(f"  训练集: {X_train.shape}")
        print(f"  测试集: {X_test.shape}")
        print(f"  特征数: {len(feature_names)}")
    except FileNotFoundError as e:
        print(f"  错误: {e}")
        print("  请确保已完成Day 6-13的特征工程工作")
        return

    # 训练模型
    print("\n[步骤2] 训练模型...")

    print("  训练RandomForest...")
    rf_trainer = RandomForestTrainer()
    rf_trainer.build_model(n_estimators=100, max_depth=10)
    rf_trainer.train(X_train, y_train, feature_names=feature_names)

    print("  训练XGBoost...")
    xgb_trainer = XGBoostTrainer()
    xgb_trainer.build_model(n_estimators=100, learning_rate=0.1, max_depth=6)
    xgb_trainer.train(X_train, y_train, feature_names=feature_names)

    print("  训练Ensemble...")
    ensemble_trainer = EnsembleTrainer()
    # 归一化权重
    weights = [1, 1.2]
    total = sum(weights)
    normalized_weights = [w/total for w in weights]
    ensemble_trainer.build_model(weights=normalized_weights)
    ensemble_trainer.train(X_train, y_train, feature_names=feature_names)

    # 保存模型
    print("\n[步骤3] 保存模型...")
    manager = ModelManager('data/models')

    # 创建新的scaler用于30维特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("  已创建30维特征的标准化器")

    saved_paths = manager.save_all_models(
        rf_trainer=rf_trainer,
        xgb_trainer=xgb_trainer,
        ensemble_trainer=ensemble_trainer,
        scaler=scaler,
        feature_names=feature_names
    )

    print(f"\n  已保存 {len(saved_paths)} 个模型文件")

    # 导出部署包
    print("\n[步骤4] 导出部署包...")
    export_dir = manager.export_for_deployment('data/deployment')
    print(f"  部署包已导出到: {export_dir}")

    # 测试预测器
    print("\n[步骤5] 测试预测器...")
    predictor = PhishingPredictor('data/models')

    # 单条预测
    result = predictor.predict(X_test[0])
    print(f"  测试样本预测: {result['label']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  概率: 正常={result['probability']['legitimate']:.4f}, 钓鱼={result['probability']['phishing']:.4f}")

    # 批量预测
    batch_results = predictor.predict_batch(X_test[:5])
    print(f"\n  批量预测5个样本:")
    for i, res in enumerate(batch_results):
        print(f"    样本{i+1}: {res['label']} (置信度: {res['confidence']:.4f})")

    # 阶段验收
    print("\n[步骤6] 执行阶段验收...")
    validation = validate_phase3()

    # 生成报告
    print("\n[步骤7] 生成阶段报告...")
    report_path = generate_phase3_report()

    # 保存验收结果
    import json
    validation_path = 'data/reports/phase3_validation.json'
    os.makedirs('data/reports', exist_ok=True)
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    print(f"\n验收结果已保存: {validation_path}")

    # 最终汇总
    print("\n" + "=" * 70)
    print("Day 20 验证完成!")
    print("=" * 70)

    print("\n产出物清单:")
    print("  模型文件:")
    for name, path in saved_paths.items():
        print(f"    - {path}")
    print("  部署包:")
    print(f"    - {export_dir}")
    print("  报告文件:")
    print(f"    - {validation_path}")
    print(f"    - {report_path}")

    if validation['passed']:
        print("\n[PASS] 阶段三全部完成，可以进入阶段四（Web开发）！")
    else:
        print("\n[WARN] 阶段三存在待解决问题，请检查后再进入下一阶段。")

    print("=" * 70)


if __name__ == "__main__":
    main()
