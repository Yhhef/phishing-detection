#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30维流量特征提取脚本
====================

设计思路：
1. 对10000条数据进行编号 (0-9999)
2. 逐一获取30维特征，记录成功/失败状态
3. 超时或失败的标记为获取失败并跳过
4. 成功的按编号记录到结果文件
5. 每100条保存一次，支持断点续传
6. 最终生成的数据集自动剔除失败记录

使用方法：
    python scripts/extract_30dim_features.py

断点续传：
    再次运行会自动从上次断点继续

重新开始：
    python scripts/extract_30dim_features.py --restart
"""

import os
import sys
import time
import json
import socket
import argparse
import warnings
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
import numpy as np

# 禁用警告
warnings.filterwarnings('ignore')

# 设置项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')

# 30维特征列名
FEATURE_COLUMNS = [
    # URL词法特征 (17维)
    'url_length', 'domain_length', 'path_length', 'num_dots', 'num_hyphens',
    'num_underscores', 'num_slashes', 'num_digits', 'has_ip', 'has_at',
    'num_subdomains', 'has_https', 'path_depth', 'has_port', 'entropy',
    'is_shortening', 'has_suspicious',
    # HTTP响应特征 (5维)
    'http_status_code', 'http_response_time', 'http_redirect_count',
    'content_length', 'server_type',
    # SSL证书特征 (5维)
    'ssl_valid', 'ssl_days_remaining', 'ssl_issuer_trusted',
    'ssl_self_signed', 'ssl_cert_age_days',
    # DNS特征 (3维)
    'dns_resolve_time', 'dns_record_count', 'dns_has_mx'
]


class FeatureExtractor30Dim:
    """30维特征提取器"""

    # 短链服务列表
    SHORTENING_SERVICES = [
        'bit.ly', 'goo.gl', 't.co', 'tinyurl.com', 'is.gd', 'cli.gs',
        'ow.ly', 'j.mp', 'buff.ly', 'adf.ly', 'tiny.cc', 'lnkd.in',
        'db.tt', 'qr.ae', 'bitly.com', 'cur.lv', 'ity.im', 'q.gs',
        'po.st', 'bc.vc', 'u.to', 'v.gd', 'shorturl.at'
    ]

    # 可疑关键词
    SUSPICIOUS_WORDS = [
        'login', 'signin', 'verify', 'secure', 'account', 'update',
        'confirm', 'password', 'credential', 'bank', 'paypal', 'ebay',
        'amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix',
        'support', 'security', 'alert', 'suspended', 'locked', 'unusual'
    ]

    def __init__(self, timeout=5):
        self.timeout = timeout

    def extract_url_features(self, url: str) -> dict:
        """提取URL词法特征 (17维) - 纯本地计算，不需要网络"""
        features = {}

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path

            # 基本长度特征
            features['url_length'] = len(url)
            features['domain_length'] = len(domain)
            features['path_length'] = len(path)

            # 字符统计
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_digits'] = sum(c.isdigit() for c in url)

            # IP地址检测
            import re
            ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
            features['has_ip'] = 1 if re.search(ip_pattern, domain) else 0

            # @ 符号检测
            features['has_at'] = 1 if '@' in url else 0

            # 子域名数量
            domain_parts = domain.split('.')
            features['num_subdomains'] = max(0, len(domain_parts) - 2)

            # HTTPS检测
            features['has_https'] = 1 if parsed.scheme == 'https' else 0

            # 路径深度
            features['path_depth'] = path.count('/') - 1 if path else 0

            # 端口检测
            features['has_port'] = 1 if ':' in domain and not domain.startswith('[') else 0

            # URL熵
            import math
            url_lower = url.lower()
            freq = {}
            for c in url_lower:
                freq[c] = freq.get(c, 0) + 1
            entropy = 0
            for count in freq.values():
                p = count / len(url_lower)
                entropy -= p * math.log2(p)
            features['entropy'] = round(entropy, 4)

            # 短链检测
            features['is_shortening'] = 1 if any(s in domain for s in self.SHORTENING_SERVICES) else 0

            # 可疑词检测
            features['has_suspicious'] = 1 if any(w in url.lower() for w in self.SUSPICIOUS_WORDS) else 0

        except Exception as e:
            # URL解析失败，返回默认值
            features = {k: 0 for k in FEATURE_COLUMNS[:17]}

        return features

    def extract_http_features(self, url: str) -> dict:
        """提取HTTP响应特征 (5维)"""
        features = {
            'http_status_code': -1,
            'http_response_time': -1,
            'http_redirect_count': -1,
            'content_length': -1,
            'server_type': -1
        }

        try:
            import requests

            start_time = time.time()
            response = requests.get(
                url,
                timeout=self.timeout,
                allow_redirects=True,
                verify=False,  # 忽略SSL验证
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'}
            )
            response_time = time.time() - start_time

            features['http_status_code'] = response.status_code
            features['http_response_time'] = round(response_time, 3)
            features['http_redirect_count'] = len(response.history)
            features['content_length'] = len(response.content)

            # 服务器类型编码
            server = response.headers.get('Server', '').lower()
            if 'apache' in server:
                features['server_type'] = 1
            elif 'nginx' in server:
                features['server_type'] = 2
            elif 'iis' in server:
                features['server_type'] = 3
            elif 'cloudflare' in server:
                features['server_type'] = 4
            elif server:
                features['server_type'] = 5
            else:
                features['server_type'] = 0

        except Exception:
            pass

        return features

    def extract_ssl_features(self, url: str) -> dict:
        """提取SSL证书特征 (5维)"""
        features = {
            'ssl_valid': -1,
            'ssl_days_remaining': -1,
            'ssl_issuer_trusted': -1,
            'ssl_self_signed': -1,
            'ssl_cert_age_days': -1
        }

        parsed = urlparse(url)
        if parsed.scheme != 'https':
            # 非HTTPS，标记为无SSL
            features = {k: 0 for k in features.keys()}
            return features

        try:
            import ssl
            import socket
            from datetime import datetime

            hostname = parsed.netloc.split(':')[0]
            port = 443

            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with socket.create_connection((hostname, port), timeout=self.timeout) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert(binary_form=True)

            # 解析证书
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend

            cert_obj = x509.load_der_x509_certificate(cert, default_backend())

            # 证书有效性
            now = datetime.utcnow()
            not_after = cert_obj.not_valid_after_utc.replace(tzinfo=None) if hasattr(cert_obj.not_valid_after_utc, 'replace') else cert_obj.not_valid_after
            not_before = cert_obj.not_valid_before_utc.replace(tzinfo=None) if hasattr(cert_obj.not_valid_before_utc, 'replace') else cert_obj.not_valid_before

            # 处理时区
            if hasattr(not_after, 'tzinfo') and not_after.tzinfo is not None:
                not_after = not_after.replace(tzinfo=None)
            if hasattr(not_before, 'tzinfo') and not_before.tzinfo is not None:
                not_before = not_before.replace(tzinfo=None)

            features['ssl_valid'] = 1 if not_before <= now <= not_after else 0
            features['ssl_days_remaining'] = max(0, (not_after - now).days)
            features['ssl_cert_age_days'] = max(0, (now - not_before).days)

            # 发行者检测
            issuer = cert_obj.issuer.rfc4514_string().lower()
            trusted_issuers = ['digicert', 'comodo', 'globalsign', 'godaddy',
                             'letsencrypt', 'sectigo', 'geotrust', 'thawte']
            features['ssl_issuer_trusted'] = 1 if any(t in issuer for t in trusted_issuers) else 0

            # 自签名检测
            features['ssl_self_signed'] = 1 if cert_obj.issuer == cert_obj.subject else 0

        except Exception:
            pass

        return features

    def extract_dns_features(self, url: str) -> dict:
        """提取DNS特征 (3维)"""
        features = {
            'dns_resolve_time': -1,
            'dns_record_count': -1,
            'dns_has_mx': -1
        }

        try:
            import dns.resolver

            parsed = urlparse(url)
            domain = parsed.netloc.split(':')[0]

            # 移除www前缀
            if domain.startswith('www.'):
                domain = domain[4:]

            resolver = dns.resolver.Resolver()
            resolver.timeout = self.timeout
            resolver.lifetime = self.timeout

            # A记录解析时间
            start_time = time.time()
            try:
                a_records = resolver.resolve(domain, 'A')
                features['dns_resolve_time'] = round(time.time() - start_time, 3)
                features['dns_record_count'] = len(a_records)
            except:
                features['dns_resolve_time'] = -1
                features['dns_record_count'] = 0

            # MX记录检测
            try:
                mx_records = resolver.resolve(domain, 'MX')
                features['dns_has_mx'] = 1 if len(mx_records) > 0 else 0
            except:
                features['dns_has_mx'] = 0

        except Exception:
            pass

        return features

    def extract_all(self, url: str) -> tuple:
        """
        提取所有30维特征

        Returns:
            (success: bool, features: dict)
        """
        all_features = {}

        # 1. URL特征（本地计算，始终成功）
        url_features = self.extract_url_features(url)
        all_features.update(url_features)

        # 2. HTTP特征
        http_features = self.extract_http_features(url)
        all_features.update(http_features)

        # 3. SSL特征
        ssl_features = self.extract_ssl_features(url)
        all_features.update(ssl_features)

        # 4. DNS特征
        dns_features = self.extract_dns_features(url)
        all_features.update(dns_features)

        # 判断是否成功（至少HTTP或DNS有一个成功）
        http_success = http_features['http_status_code'] != -1
        dns_success = dns_features['dns_resolve_time'] != -1
        success = http_success or dns_success

        return success, all_features


def load_raw_data():
    """加载原始数据并编号"""
    dataset_file = os.path.join(DATA_DIR, 'dataset.csv')

    if not os.path.exists(dataset_file):
        print(f"错误: 数据文件不存在 {dataset_file}")
        sys.exit(1)

    df = pd.read_csv(dataset_file)
    df['id'] = range(len(df))  # 添加编号

    return df


def load_progress():
    """加载进度"""
    progress_file = os.path.join(DATA_DIR, '30dim_progress.json')

    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    return {
        'current_index': 0,
        'success_count': 0,
        'fail_count': 0,
        'start_time': datetime.now().isoformat(),
        'last_update': None
    }


def save_progress(progress):
    """保存进度"""
    progress_file = os.path.join(DATA_DIR, '30dim_progress.json')
    progress['last_update'] = datetime.now().isoformat()

    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_results():
    """加载已有结果"""
    result_file = os.path.join(DATA_DIR, '30dim_results.csv')

    if os.path.exists(result_file):
        return pd.read_csv(result_file)

    return pd.DataFrame()


def save_results(results_df):
    """保存结果"""
    result_file = os.path.join(DATA_DIR, '30dim_results.csv')
    results_df.to_csv(result_file, index=False)


def load_status():
    """加载状态记录（成功/失败）"""
    status_file = os.path.join(DATA_DIR, '30dim_status.csv')

    if os.path.exists(status_file):
        return pd.read_csv(status_file)

    return pd.DataFrame(columns=['id', 'url', 'status', 'error_msg'])


def save_status(status_df):
    """保存状态记录"""
    status_file = os.path.join(DATA_DIR, '30dim_status.csv')
    status_df.to_csv(status_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='30维流量特征提取')
    parser.add_argument('--restart', action='store_true', help='重新开始')
    parser.add_argument('--timeout', type=int, default=5, help='超时时间（秒）')
    parser.add_argument('--batch-size', type=int, default=100, help='保存间隔')
    args = parser.parse_args()

    print("=" * 70)
    print("30维流量特征提取")
    print("=" * 70)

    # 加载数据
    df = load_raw_data()
    total = len(df)
    print(f"数据总量: {total} 条")

    # 加载或重置进度
    if args.restart:
        print("\n[重新开始] 清除之前的进度...")
        for f in ['30dim_progress.json', '30dim_results.csv', '30dim_status.csv']:
            fp = os.path.join(DATA_DIR, f)
            if os.path.exists(fp):
                os.remove(fp)
        progress = load_progress()
        results_list = []
        status_list = []
    else:
        progress = load_progress()
        results_df = load_results()
        results_list = results_df.to_dict('records') if len(results_df) > 0 else []
        status_df = load_status()
        status_list = status_df.to_dict('records') if len(status_df) > 0 else []

        if progress['current_index'] > 0:
            print(f"\n[断点续传] 从第 {progress['current_index']} 条继续...")
            print(f"  已成功: {progress['success_count']}")
            print(f"  已失败: {progress['fail_count']}")

    start_idx = progress['current_index']

    if start_idx >= total:
        print("\n所有数据已处理完成！")
        return

    # 初始化提取器
    extractor = FeatureExtractor30Dim(timeout=args.timeout)

    print(f"\n超时设置: {args.timeout}秒")
    print(f"保存间隔: 每 {args.batch_size} 条")
    print("-" * 70)
    print(f"开始处理: [{start_idx + 1} - {total}]")
    print("-" * 70)

    batch_start = time.time()

    for i in range(start_idx, total):
        row = df.iloc[i]
        url = row['url']
        label = row['label']
        idx = row['id']

        # 提取特征
        try:
            success, features = extractor.extract_all(url)

            if success:
                # 成功 - 记录结果
                record = {
                    'id': idx,
                    'url': url,
                    'label': label
                }
                record.update(features)
                results_list.append(record)

                status_list.append({
                    'id': idx,
                    'url': url[:100],
                    'status': 'success',
                    'error_msg': ''
                })
                progress['success_count'] += 1
            else:
                # 失败 - 仅记录状态
                status_list.append({
                    'id': idx,
                    'url': url[:100],
                    'status': 'fail',
                    'error_msg': '网络特征获取失败'
                })
                progress['fail_count'] += 1

        except Exception as e:
            # 异常 - 记录错误
            status_list.append({
                'id': idx,
                'url': url[:100],
                'status': 'error',
                'error_msg': str(e)[:50]
            })
            progress['fail_count'] += 1

        progress['current_index'] = i + 1

        # 进度显示
        if (i + 1) % 10 == 0:
            elapsed = time.time() - batch_start
            speed = 10 / elapsed if elapsed > 0 else 0
            remaining = total - i - 1
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_minutes = eta_seconds / 60

            success_rate = progress['success_count'] / (i + 1) * 100

            print(f"[{i+1:5d}/{total}] "
                  f"成功:{progress['success_count']:4d} "
                  f"失败:{progress['fail_count']:4d} "
                  f"(成功率:{success_rate:.1f}%) "
                  f"| 速度:{speed:.1f}/s "
                  f"| 剩余:{eta_minutes:.0f}分钟")

            batch_start = time.time()

        # 每batch_size条保存一次
        if (i + 1) % args.batch_size == 0:
            print(f"\n>>> 保存检查点 [{i+1}/{total}] <<<")

            results_df = pd.DataFrame(results_list)
            status_df = pd.DataFrame(status_list)

            save_results(results_df)
            save_status(status_df)
            save_progress(progress)

            print(f"    结果文件: {len(results_df)} 条成功记录")
            print(f"    状态文件: {len(status_df)} 条状态记录")
            print("-" * 70)

    # 最终保存
    print("\n" + "=" * 70)
    print("提取完成！")
    print("=" * 70)

    results_df = pd.DataFrame(results_list)
    status_df = pd.DataFrame(status_list)

    save_results(results_df)
    save_status(status_df)
    save_progress(progress)

    print(f"\n统计结果:")
    print(f"  总数据: {total}")
    print(f"  成功: {progress['success_count']} ({progress['success_count']/total*100:.1f}%)")
    print(f"  失败: {progress['fail_count']} ({progress['fail_count']/total*100:.1f}%)")

    print(f"\n输出文件:")
    print(f"  结果: data/processed/30dim_results.csv ({len(results_df)} 条)")
    print(f"  状态: data/processed/30dim_status.csv ({len(status_df)} 条)")
    print(f"  进度: data/processed/30dim_progress.json")

    # 清理进度文件
    progress_file = os.path.join(DATA_DIR, '30dim_progress.json')
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("\n进度文件已清理")


if __name__ == '__main__':
    main()