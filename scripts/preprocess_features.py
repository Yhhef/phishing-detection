#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征预处理执行脚本

用法：
    python scripts/preprocess_features.py

功能：
    1. 读取train_features.csv和test_features.csv
    2. 执行缺失值填充、异常值处理、标准化
    3. 输出train_scaled.csv和test_scaled.csv
    4. 保存scaler.pkl
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessor import preprocess_dataset, validate_preprocessing


def main():
    # 配置路径
    train_path = project_root / 'data' / 'processed' / 'train_features.csv'
    test_path = project_root / 'data' / 'processed' / 'test_features.csv'
    output_dir = project_root / 'data' / 'processed'
    scaler_path = project_root / 'data' / 'models' / 'scaler.pkl'

    # 确保目录存在
    (project_root / 'data' / 'models').mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("特征预处理")
    print("=" * 60)

    # 检查输入文件
    if not train_path.exists():
        print(f"错误: 训练集文件不存在 - {train_path}")
        print("请先运行 scripts/extract_features.py 提取特征")
        sys.exit(1)

    if not test_path.exists():
        print(f"错误: 测试集文件不存在 - {test_path}")
        print("请先运行 scripts/extract_features.py 提取特征")
        sys.exit(1)

    # 执行预处理
    train_df, test_df = preprocess_dataset(
        train_path=str(train_path),
        test_path=str(test_path),
        output_dir=str(output_dir),
        scaler_path=str(scaler_path)
    )

    print("\n" + "-" * 40)
    print("预处理结果")
    print("-" * 40)
    print(f"训练集: {len(train_df)} 条, {len(train_df.columns)} 列")
    print(f"测试集: {len(test_df)} 条, {len(test_df.columns)} 列")

    # 验证预处理结果
    print("\n" + "-" * 40)
    print("验证预处理结果")
    print("-" * 40)

    train_scaled_path = output_dir / 'train_scaled.csv'
    validation = validate_preprocessing(str(train_path), str(train_scaled_path))

    print(f"记录数匹配: {validation['record_count_match']}")
    print(f"特征数: {validation['feature_count']}")
    print(f"包含NaN: {validation['has_nan']}")
    print(f"包含Inf: {validation['has_inf']}")

    # 检查标准化效果
    mean_ok = all(validation['mean_close_to_zero'].values())
    std_ok = all(validation['std_close_to_one'].values())
    print(f"均值接近0: {mean_ok}")
    print(f"标准差接近1: {std_ok}")

    print("\n" + "=" * 60)
    print("[SUCCESS] 特征预处理完成!")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  - {output_dir / 'train_scaled.csv'}")
    print(f"  - {output_dir / 'test_scaled.csv'}")
    print(f"  - {scaler_path}")


if __name__ == '__main__':
    main()
