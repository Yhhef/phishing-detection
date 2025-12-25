#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量特征提取执行脚本

用法：
    python scripts/extract_features.py [--mode fast|full] [--resume]

示例：
    # 快速模式（仅URL特征，17维）
    python scripts/extract_features.py --mode fast

    # 完整模式（全部特征，30维）
    python scripts/extract_features.py --mode full

    # 断点续传
    python scripts/extract_features.py --mode fast --resume
"""

import argparse
import sys
import os
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.feature_extraction import extract_from_csv, batch_extract_features
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='批量特征提取')
    parser.add_argument(
        '--mode',
        choices=['fast', 'full'],
        default='fast',
        help='提取模式: fast=仅URL特征(17维), full=全部特征(30维)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从断点续传'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='仅处理训练集'
    )
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='仅处理测试集'
    )

    args = parser.parse_args()

    # 配置路径 - 数据在 data/processed/ 目录下
    train_input = project_root / 'data' / 'processed' / 'train.csv'
    test_input = project_root / 'data' / 'processed' / 'test.csv'
    train_output = project_root / 'data' / 'processed' / 'train_features.csv'
    test_output = project_root / 'data' / 'processed' / 'test_features.csv'

    # 确保输出目录存在
    (project_root / 'data' / 'processed').mkdir(parents=True, exist_ok=True)

    include_network = (args.mode == 'full')

    print("=" * 60)
    print("批量特征提取")
    print("=" * 60)
    print(f"模式: {'完整30维' if include_network else '快速17维'}")
    print(f"断点续传: {'是' if args.resume else '否'}")
    print()

    # 处理训练集
    if not args.test_only:
        print("-" * 40)
        print("处理训练集...")
        print("-" * 40)

        if not train_input.exists():
            print(f"[ERROR] 训练集文件不存在: {train_input}")
            sys.exit(1)

        start_time = time.time()
        train_df = extract_from_csv(
            input_path=str(train_input),
            output_path=str(train_output),
            include_network=include_network
        )
        train_time = time.time() - start_time

        print(f"\n训练集处理完成!")
        print(f"  记录数: {len(train_df)}")
        print(f"  特征数: {len(train_df.columns) - 2}")  # 减去url和label
        print(f"  耗时: {train_time:.2f}秒")
        print(f"  输出: {train_output}")

    # 处理测试集
    if not args.train_only:
        print("\n" + "-" * 40)
        print("处理测试集...")
        print("-" * 40)

        if not test_input.exists():
            print(f"[ERROR] 测试集文件不存在: {test_input}")
            sys.exit(1)

        start_time = time.time()
        test_df = extract_from_csv(
            input_path=str(test_input),
            output_path=str(test_output),
            include_network=include_network
        )
        test_time = time.time() - start_time

        print(f"\n测试集处理完成!")
        print(f"  记录数: {len(test_df)}")
        print(f"  特征数: {len(test_df.columns) - 2}")
        print(f"  耗时: {test_time:.2f}秒")
        print(f"  输出: {test_output}")

    print("\n" + "=" * 60)
    print("[SUCCESS] 批量特征提取完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
