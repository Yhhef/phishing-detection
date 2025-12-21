"""
数据采集模块
负责从PhishTank和Tranco下载原始数据

作者: 毕业设计项目组
日期: 2025年12月
"""

import os
import json
import requests
import gzip
import io
from datetime import datetime
from tqdm import tqdm
import logging

# 导入项目配置
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR
from src.utils import ensure_dir, save_json, get_timestamp, logger


class PhishTankCollector:
    """
    PhishTank数据采集器

    从PhishTank下载已验证的钓鱼URL数据
    """

    # PhishTank数据下载URL（JSON格式，gzip压缩）
    DOWNLOAD_URL = "http://data.phishtank.com/data/online-valid.json.gz"

    def __init__(self, output_dir=None):
        """
        初始化采集器

        Args:
            output_dir: 数据输出目录，默认为 data/raw/
        """
        self.output_dir = output_dir or RAW_DATA_DIR
        ensure_dir(self.output_dir)

    def download(self, max_records=None):
        """
        下载PhishTank数据

        Args:
            max_records: 最大记录数，None表示全部下载

        Returns:
            dict: 下载的数据
        """
        logger.info("开始下载PhishTank数据...")
        logger.info(f"数据源: {self.DOWNLOAD_URL}")

        try:
            # 设置请求头，模拟浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # 下载gzip压缩的JSON数据
            response = requests.get(
                self.DOWNLOAD_URL,
                headers=headers,
                timeout=300,  # 5分钟超时
                stream=True
            )
            response.raise_for_status()

            # 获取文件大小
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                logger.info(f"文件大小: {total_size / 1024 / 1024:.2f} MB")

            # 下载并解压
            content = b''
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    pbar.update(len(chunk))

            # 解压gzip
            logger.info("解压数据...")
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
                raw_data = json.loads(f.read().decode('utf-8'))

            logger.info(f"原始数据量: {len(raw_data)} 条")

            # 清洗和筛选数据
            cleaned_data = self._clean_data(raw_data, max_records)

            return cleaned_data

        except requests.RequestException as e:
            logger.error(f"下载失败: {e}")
            raise

    def _clean_data(self, raw_data, max_records=None):
        """
        清洗和筛选数据

        Args:
            raw_data: 原始数据列表
            max_records: 最大记录数

        Returns:
            dict: 清洗后的数据
        """
        logger.info("开始数据清洗...")

        cleaned = []
        seen_urls = set()

        for item in tqdm(raw_data, desc="清洗中"):
            # 只保留已验证的在线钓鱼URL
            if not item.get('verified') == 'yes':
                continue
            if not item.get('online') == 'yes':
                continue

            url = item.get('url', '').strip()

            # URL有效性检查
            if not url:
                continue
            if not url.startswith(('http://', 'https://')):
                continue
            if len(url) > 2048:  # URL长度限制
                continue

            # 去重
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # 提取需要的字段
            cleaned.append({
                'id': item.get('phish_id'),
                'url': url,
                'verified': True,
                'online': True,
                'submission_time': item.get('submission_time'),
                'verification_time': item.get('verification_time'),
                'target': item.get('target', 'Unknown')
            })

            # 达到最大记录数则停止
            if max_records and len(cleaned) >= max_records:
                break

        logger.info(f"清洗后数据量: {len(cleaned)} 条")

        # 构建输出数据结构
        result = {
            'download_time': datetime.now().isoformat(),
            'source': 'PhishTank',
            'total_count': len(cleaned),
            'data': cleaned
        }

        return result

    def save(self, data, filename=None):
        """
        保存数据到文件

        Args:
            data: 要保存的数据
            filename: 文件名，默认按日期命名

        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"phishtank_{datetime.now().strftime('%Y%m%d')}.json"

        filepath = os.path.join(self.output_dir, filename)
        save_json(data, filepath)

        logger.info(f"数据已保存到: {filepath}")
        return filepath

    def collect(self, max_records=6000):
        """
        执行完整的数据采集流程

        Args:
            max_records: 最大记录数

        Returns:
            tuple: (数据, 文件路径)
        """
        # 下载数据
        data = self.download(max_records)

        # 保存数据
        filepath = self.save(data)

        # 输出统计信息
        self._print_statistics(data)

        return data, filepath

    def _print_statistics(self, data):
        """打印数据统计信息"""
        print("\n" + "="*50)
        print("PhishTank数据采集完成")
        print("="*50)
        print(f"下载时间: {data['download_time']}")
        print(f"数据来源: {data['source']}")
        print(f"总记录数: {data['total_count']}")

        # 统计目标网站分布
        if data['data']:
            targets = {}
            for item in data['data']:
                target = item.get('target', 'Unknown')
                targets[target] = targets.get(target, 0) + 1

            print("\n目标网站分布 (Top 10):")
            sorted_targets = sorted(targets.items(), key=lambda x: x[1], reverse=True)[:10]
            for target, count in sorted_targets:
                print(f"  {target}: {count}")

        print("="*50)


# 备用数据源：OpenPhish
class OpenPhishCollector:
    """
    OpenPhish数据采集器（备用）

    当PhishTank不可用时使用
    """

    DOWNLOAD_URL = "https://openphish.com/feed.txt"

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or RAW_DATA_DIR
        ensure_dir(self.output_dir)

    def download(self, max_records=None):
        """下载OpenPhish数据"""
        logger.info("开始下载OpenPhish数据...")

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(self.DOWNLOAD_URL, headers=headers, timeout=60)
            response.raise_for_status()

            urls = response.text.strip().split('\n')
            urls = [url.strip() for url in urls if url.strip()]

            logger.info(f"获取到 {len(urls)} 条URL")

            # 清洗数据
            cleaned = []
            seen = set()

            for url in urls:
                if not url.startswith(('http://', 'https://')):
                    continue
                if url in seen:
                    continue
                seen.add(url)

                cleaned.append({
                    'id': len(cleaned) + 1,
                    'url': url,
                    'verified': True,
                    'online': True,
                    'submission_time': datetime.now().isoformat(),
                    'target': 'Unknown'
                })

                if max_records and len(cleaned) >= max_records:
                    break

            result = {
                'download_time': datetime.now().isoformat(),
                'source': 'OpenPhish',
                'total_count': len(cleaned),
                'data': cleaned
            }

            return result

        except Exception as e:
            logger.error(f"OpenPhish下载失败: {e}")
            raise

    def collect(self, max_records=6000):
        """执行采集"""
        data = self.download(max_records)

        filename = f"openphish_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = os.path.join(self.output_dir, filename)
        save_json(data, filepath)

        logger.info(f"数据已保存到: {filepath}")

        # 打印统计
        print("\n" + "="*50)
        print("OpenPhish数据采集完成")
        print("="*50)
        print(f"下载时间: {data['download_time']}")
        print(f"数据来源: {data['source']}")
        print(f"总记录数: {data['total_count']}")
        print("="*50)

        return data, filepath


class TrancoCollector:
    """
    Tranco数据采集器

    从Tranco List下载正常网站排名数据
    Tranco是综合Alexa、Umbrella、Majestic等多个排名榜单的可靠数据源
    """

    # Tranco数据下载URL（CSV格式，zip压缩）
    DOWNLOAD_URL = "https://tranco-list.eu/top-1m.csv.zip"

    def __init__(self, output_dir=None):
        """
        初始化采集器

        Args:
            output_dir: 数据输出目录，默认为 data/raw/
        """
        self.output_dir = output_dir or RAW_DATA_DIR
        ensure_dir(self.output_dir)

    def download(self, max_records=6000):
        """
        下载Tranco数据

        Args:
            max_records: 最大记录数

        Returns:
            dict: 下载的数据
        """
        import zipfile

        logger.info("开始下载Tranco数据...")
        logger.info(f"数据源: {self.DOWNLOAD_URL}")

        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # 下载zip文件
            response = requests.get(
                self.DOWNLOAD_URL,
                headers=headers,
                timeout=300,
                stream=True
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                logger.info(f"文件大小: {total_size / 1024 / 1024:.2f} MB")

            # 下载内容
            content = b''
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="下载中") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    content += chunk
                    pbar.update(len(chunk))

            # 解压zip
            logger.info("解压数据...")
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                # 获取CSV文件名
                csv_name = zf.namelist()[0]
                csv_content = zf.read(csv_name).decode('utf-8')

            # 解析CSV数据
            lines = csv_content.strip().split('\n')
            logger.info(f"原始数据量: {len(lines)} 条")

            # 清洗数据
            cleaned_data = self._clean_data(lines, max_records)

            return cleaned_data

        except requests.RequestException as e:
            logger.error(f"下载失败: {e}")
            raise

    def _clean_data(self, lines, max_records=6000):
        """
        清洗和筛选数据

        Args:
            lines: CSV行列表
            max_records: 最大记录数

        Returns:
            dict: 清洗后的数据
        """
        logger.info("开始数据清洗...")

        cleaned = []
        seen_domains = set()

        for line in tqdm(lines[:max_records * 2], desc="清洗中"):
            # CSV格式: rank,domain
            parts = line.strip().split(',')
            if len(parts) != 2:
                continue

            rank, domain = parts
            domain = domain.strip().lower()

            # 域名有效性检查
            if not domain:
                continue
            if len(domain) < 3:
                continue
            if '.' not in domain:
                continue

            # 去重
            if domain in seen_domains:
                continue
            seen_domains.add(domain)

            # 构建完整URL
            url = f"https://{domain}"

            cleaned.append({
                'id': int(rank),
                'rank': int(rank),
                'domain': domain,
                'url': url,
                'source': 'Tranco'
            })

            if len(cleaned) >= max_records:
                break

        logger.info(f"清洗后数据量: {len(cleaned)} 条")

        # 构建输出数据结构
        result = {
            'download_time': datetime.now().isoformat(),
            'source': 'Tranco',
            'total_count': len(cleaned),
            'data': cleaned
        }

        return result

    def save(self, data, filename=None):
        """
        保存数据到文件

        Args:
            data: 要保存的数据
            filename: 文件名，默认按日期命名

        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"tranco_{datetime.now().strftime('%Y%m%d')}.json"

        filepath = os.path.join(self.output_dir, filename)
        save_json(data, filepath)

        logger.info(f"数据已保存到: {filepath}")
        return filepath

    def collect(self, max_records=6000):
        """
        执行完整的数据采集流程

        Args:
            max_records: 最大记录数

        Returns:
            tuple: (数据, 文件路径)
        """
        # 下载数据
        data = self.download(max_records)

        # 保存数据
        filepath = self.save(data)

        # 输出统计信息
        self._print_statistics(data)

        return data, filepath

    def _print_statistics(self, data):
        """打印数据统计信息"""
        print("\n" + "="*50)
        print("Tranco数据采集完成")
        print("="*50)
        print(f"下载时间: {data['download_time']}")
        print(f"数据来源: {data['source']}")
        print(f"总记录数: {data['total_count']}")

        # 显示Top 10网站
        if data['data']:
            print("\nTop 10 网站:")
            for item in data['data'][:10]:
                print(f"  {item['rank']}. {item['domain']}")

        print("="*50)


def main():
    """主函数 - 执行数据采集"""
    print("="*60)
    print("钓鱼网站检测系统 - 数据采集模块")
    print("="*60)

    # 尝试从PhishTank采集
    try:
        collector = PhishTankCollector()
        data, filepath = collector.collect(max_records=6000)
        print(f"\n[SUCCESS] PhishTank数据采集成功!")
        print(f"文件位置: {filepath}")

    except Exception as e:
        print(f"\n[FAILED] PhishTank采集失败: {e}")
        print("尝试使用备用数据源 OpenPhish...")

        try:
            collector = OpenPhishCollector()
            data, filepath = collector.collect(max_records=6000)
            print(f"\n[SUCCESS] OpenPhish数据采集成功!")
            print(f"文件位置: {filepath}")
        except Exception as e2:
            print(f"[FAILED] OpenPhish采集也失败: {e2}")
            return None

    return filepath


if __name__ == '__main__':
    main()
