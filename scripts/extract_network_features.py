"""
网络特征分批提取脚本
支持断点续传，每批100条保存一次

使用方法:
    python scripts/extract_network_features.py --dataset train --batch-size 100
    python scripts/extract_network_features.py --dataset test --batch-size 100

断点续传:
    如果中断了，再次运行相同命令会自动从上次断点继续
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import socket

# 禁用警告
warnings.filterwarnings('ignore')

# 添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.feature_extraction import (
    HTTPFeatureExtractor,
    SSLFeatureExtractor,
    DNSFeatureExtractor
)


def extract_network_features_single(url: str, timeout: int = 5) -> dict:
    """
    提取单个URL的网络特征（13维）

    Args:
        url: URL字符串
        timeout: 超时时间（秒）

    Returns:
        13维网络特征字典
    """
    # 设置全局socket超时
    original_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)

    features = {}

    try:
        # HTTP特征 (5维)
        http_ext = HTTPFeatureExtractor(url)
        http_ext.TIMEOUT = timeout
        http_features = http_ext.extract_all()
        features.update(http_features)
    except Exception as e:
        features.update({
            'http_status_code': -1,
            'http_response_time': -1,
            'http_redirect_count': -1,
            'content_length': -1,
            'server_type': -1
        })

    try:
        # SSL特征 (5维)
        ssl_ext = SSLFeatureExtractor(url)
        ssl_ext.TIMEOUT = timeout
        ssl_features = ssl_ext.extract_all()
        features.update(ssl_features)
    except Exception as e:
        features.update({
            'ssl_cert_valid': -1,
            'ssl_cert_days': -1,
            'ssl_issuer_type': -1,
            'ssl_self_signed': -1,
            'ssl_cert_age': -1
        })

    try:
        # DNS特征 (3维)
        dns_ext = DNSFeatureExtractor(url)
        dns_ext.TIMEOUT = timeout
        dns_features = dns_ext.extract_all()
        features.update(dns_features)
    except Exception as e:
        features.update({
            'domain_entropy': -1,
            'dns_resolve_time': -1,
            'dns_record_count': -1
        })

    # 恢复超时设置
    socket.setdefaulttimeout(original_timeout)

    return features


def load_progress(progress_file: str) -> int:
    """加载进度"""
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return int(f.read().strip())
    return 0


def save_progress(progress_file: str, index: int):
    """保存进度"""
    with open(progress_file, 'w') as f:
        f.write(str(index))


def main():
    parser = argparse.ArgumentParser(description='分批提取网络特征')
    parser.add_argument('--dataset', type=str, choices=['train', 'test'],
                       required=True, help='数据集类型')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='每批处理数量')
    parser.add_argument('--timeout', type=int, default=5,
                       help='单个URL超时时间（秒）')
    parser.add_argument('--restart', action='store_true',
                       help='从头开始，忽略之前的进度')

    args = parser.parse_args()

    # 文件路径
    data_dir = os.path.join(project_root, 'data', 'processed')
    input_file = os.path.join(data_dir, f'{args.dataset}_features.csv')
    output_file = os.path.join(data_dir, f'{args.dataset}_network_features.csv')
    progress_file = os.path.join(data_dir, f'{args.dataset}_network_progress.txt')

    print("=" * 60)
    print(f"网络特征分批提取 - {args.dataset}数据集")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"批次大小: {args.batch_size}")
    print(f"超时时间: {args.timeout}秒")

    # 加载数据
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在 {input_file}")
        return 1

    df = pd.read_csv(input_file)
    total = len(df)
    print(f"总记录数: {total}")

    # 加载进度
    if args.restart:
        start_idx = 0
        if os.path.exists(output_file):
            os.remove(output_file)
        if os.path.exists(progress_file):
            os.remove(progress_file)
        print("从头开始提取...")
    else:
        start_idx = load_progress(progress_file)
        if start_idx > 0:
            print(f"从断点续传: 已完成 {start_idx}/{total}")

    if start_idx >= total:
        print("所有记录已处理完成！")
        return 0

    # 网络特征列名
    network_cols = [
        'http_status_code', 'http_response_time', 'http_redirect_count',
        'content_length', 'server_type',
        'ssl_cert_valid', 'ssl_cert_days', 'ssl_issuer_type',
        'ssl_self_signed', 'ssl_cert_age',
        'domain_entropy', 'dns_resolve_time', 'dns_record_count'
    ]

    # 加载已有结果
    if start_idx > 0 and os.path.exists(output_file):
        result_df = pd.read_csv(output_file)
        results = result_df.to_dict('records')
    else:
        results = []

    # 开始提取
    print(f"\n开始处理 [{start_idx+1} - {total}]...")
    print("-" * 60)

    batch_start_time = time.time()
    success_count = 0
    error_count = 0

    for i in range(start_idx, total):
        url = df.iloc[i]['url']

        # 提取网络特征
        try:
            features = extract_network_features_single(url, timeout=args.timeout)

            # 检查是否全部失败
            if all(v == -1 for v in features.values()):
                error_count += 1
            else:
                success_count += 1

        except Exception as e:
            print(f"  [{i+1}] 错误: {url[:50]}... - {str(e)[:30]}")
            features = {col: -1 for col in network_cols}
            error_count += 1

        # 添加结果
        record = {'url': url, 'label': df.iloc[i]['label']}
        record.update(features)
        results.append(record)

        # 进度显示
        if (i + 1) % 10 == 0:
            elapsed = time.time() - batch_start_time
            speed = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / speed if speed > 0 else 0
            print(f"  进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) | "
                  f"速度: {speed:.2f} url/s | "
                  f"预计剩余: {eta/60:.1f}分钟")

        # 每批保存一次
        if (i + 1) % args.batch_size == 0 or (i + 1) == total:
            # 保存结果
            result_df = pd.DataFrame(results)
            result_df.to_csv(output_file, index=False)
            save_progress(progress_file, i + 1)

            batch_time = time.time() - batch_start_time
            print(f"\n  [检查点] 已保存 {i+1}/{total} 条, 本批耗时: {batch_time:.1f}秒")
            print(f"           成功: {success_count}, 失败: {error_count}")
            print("-" * 60)

            batch_start_time = time.time()

    # 完成
    print("\n" + "=" * 60)
    print("提取完成!")
    print("=" * 60)
    print(f"总记录: {total}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"输出文件: {output_file}")

    # 清理进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return 0


if __name__ == '__main__':
    exit(main())
