#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNS特征补充提取脚本
对已有的30dim_results.csv补充提取DNS特征
"""

import os
import sys
import time
import json
import pandas as pd
import dns.resolver
from urllib.parse import urlparse
from datetime import datetime

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')


def extract_dns_features(url: str, timeout: int = 3) -> dict:
    """提取DNS特征 (3维)"""
    features = {
        'dns_resolve_time': -1,
        'dns_record_count': -1,
        'dns_has_mx': -1
    }

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.split(':')[0]

        # 移除www前缀
        if domain.startswith('www.'):
            domain = domain[4:]

        if not domain:
            return features

        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout

        # A记录解析时间
        start_time = time.time()
        try:
            a_records = resolver.resolve(domain, 'A')
            features['dns_resolve_time'] = round(time.time() - start_time, 3)
            features['dns_record_count'] = len(a_records)
        except Exception:
            features['dns_resolve_time'] = -1
            features['dns_record_count'] = 0

        # MX记录检测
        try:
            mx_records = resolver.resolve(domain, 'MX')
            features['dns_has_mx'] = 1 if len(mx_records) > 0 else 0
        except Exception:
            features['dns_has_mx'] = 0

    except Exception:
        pass

    return features


def load_progress():
    """加载进度"""
    progress_file = os.path.join(DATA_DIR, 'dns_fix_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {'current_index': 0, 'success_count': 0, 'fail_count': 0}


def save_progress(progress):
    """保存进度"""
    progress_file = os.path.join(DATA_DIR, 'dns_fix_progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress, f)


def main():
    print("=" * 60)
    print("DNS特征补充提取")
    print("=" * 60)

    # 加载已有数据
    result_file = os.path.join(DATA_DIR, '30dim_results.csv')
    if not os.path.exists(result_file):
        print(f"错误: 文件不存在 {result_file}")
        return 1

    df = pd.read_csv(result_file)
    total = len(df)
    print(f"数据总量: {total} 条")

    # 检查是否需要修复
    dns_cols = ['dns_resolve_time', 'dns_record_count', 'dns_has_mx']
    need_fix = (df[dns_cols] == -1).all(axis=1).sum()
    print(f"需要修复DNS特征: {need_fix} 条")

    if need_fix == 0:
        print("DNS特征已完整，无需修复")
        return 0

    # 加载进度
    progress = load_progress()
    start_idx = progress['current_index']

    if start_idx > 0:
        print(f"\n[断点续传] 从第 {start_idx} 条继续...")
        print(f"  已成功: {progress['success_count']}")
        print(f"  已失败: {progress['fail_count']}")

    print(f"\n开始处理 [{start_idx + 1} - {total}]...")
    print("-" * 60)

    batch_start = time.time()

    for i in range(start_idx, total):
        url = df.iloc[i]['url']

        # 提取DNS特征
        dns_features = extract_dns_features(url, timeout=3)

        # 更新数据
        for col, val in dns_features.items():
            df.at[i, col] = val

        # 统计
        if dns_features['dns_resolve_time'] != -1:
            progress['success_count'] += 1
        else:
            progress['fail_count'] += 1

        progress['current_index'] = i + 1

        # 进度显示
        if (i + 1) % 10 == 0:
            elapsed = time.time() - batch_start
            speed = 10 / elapsed if elapsed > 0 else 0
            remaining = total - i - 1
            eta_minutes = (remaining / speed / 60) if speed > 0 else 0

            success_rate = progress['success_count'] / (i + 1) * 100

            print(f"[{i+1:5d}/{total}] "
                  f"成功:{progress['success_count']:4d} "
                  f"失败:{progress['fail_count']:4d} "
                  f"({success_rate:.1f}%) "
                  f"| {speed:.1f}/s "
                  f"| 剩余:{eta_minutes:.0f}分钟")

            batch_start = time.time()

        # 每100条保存一次
        if (i + 1) % 100 == 0:
            print(f"\n>>> 保存检查点 [{i+1}/{total}] <<<")
            df.to_csv(result_file, index=False)
            save_progress(progress)
            print("-" * 60)

    # 最终保存
    print("\n" + "=" * 60)
    print("DNS特征补充完成!")
    print("=" * 60)

    df.to_csv(result_file, index=False)

    # 统计结果
    dns_success = (df['dns_resolve_time'] != -1).sum()
    print(f"\nDNS特征统计:")
    print(f"  成功解析: {dns_success} ({dns_success/total*100:.1f}%)")
    print(f"  解析失败: {total - dns_success} ({(total-dns_success)/total*100:.1f}%)")

    # 清理进度文件
    progress_file = os.path.join(DATA_DIR, 'dns_fix_progress.json')
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return 0


if __name__ == '__main__':
    exit(main())
