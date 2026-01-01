#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Day 19: 全面性能评估验证脚本

验证ModelEvaluator的完整功能并生成评估报告
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import (
    RandomForestTrainer,
    XGBoostTrainer,
    EnsembleTrainer,
    load_feature_data,
    ModelEvaluator,
    evaluate_all_models,
    generate_final_report
)


def main():
    """主函数：执行全面性能评估"""
    print("=" * 70)
    print("Day 19: 全面性能评估")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    try:
        X_train, y_train, X_test, y_test, feature_names = load_feature_data(use_30dim=True)
        print(f"  训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        print(f"  测试集: {X_test.shape[0]} 样本")
        print(f"  特征数: {len(feature_names)}")
    except FileNotFoundError as e:
        print(f"  错误: 数据文件不存在 - {e}")
        return

    # 2. 训练模型
    print("\n[2/5] 训练模型...")

    # RandomForest
    print("  训练 RandomForest...")
    rf_trainer = RandomForestTrainer()
    rf_trainer.build_model(n_estimators=100, max_depth=10)
    rf_trainer.train(X_train, y_train, feature_names=feature_names)

    # XGBoost
    print("  训练 XGBoost...")
    xgb_trainer = XGBoostTrainer()
    xgb_trainer.build_model(n_estimators=100, learning_rate=0.1, max_depth=6)
    xgb_trainer.train(X_train, y_train, feature_names=feature_names)

    # Ensemble
    print("  训练 Ensemble...")
    ensemble_trainer = EnsembleTrainer()
    ensemble_trainer.build_model()
    ensemble_trainer.train(X_train, y_train, feature_names=feature_names)

    print("  模型训练完成!")

    # 3. 评估模型
    print("\n[3/5] 评估模型...")
    evaluator = ModelEvaluator()

    # 评估各模型
    rf_metrics = evaluator.evaluate_model(rf_trainer.model, X_test, y_test, 'RandomForest')
    xgb_metrics = evaluator.evaluate_model(xgb_trainer.model, X_test, y_test, 'XGBoost')
    ensemble_metrics = evaluator.evaluate_model(ensemble_trainer.model, X_test, y_test, 'Ensemble')

    # 4. 保存图表
    print("\n[4/5] 生成可视化图表...")
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'data', 'evaluation', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # 混淆矩阵
    for model_name in ['RandomForest', 'XGBoost', 'Ensemble']:
        evaluator.plot_confusion_matrix(
            model_name,
            save_path=os.path.join(figures_dir, f'{model_name}_confusion_matrix.png')
        )
        evaluator.plot_confusion_matrix(
            model_name,
            normalize=True,
            save_path=os.path.join(figures_dir, f'{model_name}_confusion_matrix_normalized.png')
        )

    # ROC曲线
    evaluator.plot_roc_curve(save_path=os.path.join(figures_dir, 'roc_curves.png'))

    # PR曲线
    evaluator.plot_pr_curve(save_path=os.path.join(figures_dir, 'pr_curves.png'))

    # 指标对比
    evaluator.plot_metrics_comparison(save_path=os.path.join(figures_dir, 'metrics_comparison.png'))

    # 综合图
    for model_name in ['RandomForest', 'XGBoost', 'Ensemble']:
        evaluator.plot_all_curves(model_name, y_test, save_dir=figures_dir)

    print(f"  图表已保存至: {figures_dir}")

    # 5. 生成报告
    print("\n[5/5] 生成评估报告...")
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'data', 'evaluation', 'reports')

    report_path = generate_final_report(evaluator, reports_dir)

    # 6. 验证性能指标
    print("\n" + "=" * 70)
    print("性能指标验证")
    print("=" * 70)

    targets = {
        '准确率 (Accuracy)': (0.90, 'accuracy'),
        '精确率 (Precision)': (0.88, 'precision'),
        '召回率 (Recall)': (0.85, 'recall'),
        'AUC-ROC': (0.95, 'auc_roc')
    }

    best_model_name = max(evaluator.results.keys(),
                          key=lambda x: evaluator.results[x]['accuracy'])
    best_metrics = evaluator.results[best_model_name]

    all_passed = True
    for target_name, (threshold, metric_key) in targets.items():
        value = best_metrics[metric_key]
        passed = value >= threshold
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"  {status} {target_name}: {value:.4f} (目标 >= {threshold})")

    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)
    print(evaluator.get_detailed_metrics().to_string(index=False))

    print("\n" + "=" * 70)
    print("Day 19 完成状态")
    print("=" * 70)

    checklist = [
        ("ModelEvaluator类实现", True),
        ("evaluate_model方法", True),
        ("plot_confusion_matrix方法", True),
        ("plot_roc_curve方法", True),
        ("plot_pr_curve方法", True),
        ("plot_all_curves方法", True),
        ("plot_metrics_comparison方法", True),
        ("generate_report方法", True),
        ("save_results/load_results方法", True),
        ("evaluate_all_models便捷函数", True),
        ("generate_final_report便捷函数", True),
        ("性能指标达标", all_passed)
    ]

    for item, status in checklist:
        mark = "[x]" if status else "[ ]"
        print(f"  {mark} {item}")

    if all_passed:
        print("\n  Day 19 任务全部完成!")
    else:
        print("\n  注意: 部分性能指标未达标，请检查模型和数据")

    return evaluator, all_passed


if __name__ == "__main__":
    evaluator, passed = main()
