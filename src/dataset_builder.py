"""
数据集构建模块
负责合并、标签化、平衡和划分数据集

作者: 毕业设计项目组
日期: 2025年12月
"""

import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from tqdm import tqdm

# 导入项目配置
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils import ensure_dir, logger


class DatasetBuilder:
    """
    数据集构建器

    负责将原始数据整合为可用于模型训练的数据集
    """

    def __init__(self, raw_dir=None, output_dir=None):
        """
        初始化构建器

        Args:
            raw_dir: 原始数据目录
            output_dir: 输出目录
        """
        self.raw_dir = raw_dir or RAW_DATA_DIR
        self.output_dir = output_dir or PROCESSED_DATA_DIR
        ensure_dir(self.output_dir)

    def load_phishing_data(self, filename=None):
        """
        加载钓鱼数据

        Args:
            filename: 指定文件名，None则自动查找最新

        Returns:
            DataFrame: 钓鱼数据
        """
        if filename is None:
            # 查找最新的PhishTank或OpenPhish文件
            files = [f for f in os.listdir(self.raw_dir)
                     if f.startswith(('phishtank', 'openphish')) and f.endswith('.json')]
            if not files:
                raise FileNotFoundError("未找到钓鱼数据文件")
            filename = sorted(files)[-1]

        filepath = os.path.join(self.raw_dir, filename)
        logger.info(f"加载钓鱼数据: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为DataFrame
        df = pd.DataFrame(data['data'])
        df['label'] = 1  # 钓鱼标签
        df['source'] = data['source'].lower()

        # 提取域名
        df['domain'] = df['url'].apply(self._extract_domain)

        logger.info(f"加载钓鱼数据: {len(df)} 条")
        return df

    def load_normal_data(self, filename=None):
        """
        加载正常数据

        Args:
            filename: 指定文件名，None则自动查找最新

        Returns:
            DataFrame: 正常数据
        """
        if filename is None:
            files = [f for f in os.listdir(self.raw_dir)
                     if f.startswith('tranco') and f.endswith('.json')]
            if not files:
                raise FileNotFoundError("未找到正常数据文件")
            filename = sorted(files)[-1]

        filepath = os.path.join(self.raw_dir, filename)
        logger.info(f"加载正常数据: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 转换为DataFrame
        df = pd.DataFrame(data['data'])
        df['label'] = 0  # 正常标签
        df['source'] = 'tranco'

        # 确保有domain字段
        if 'domain' not in df.columns:
            df['domain'] = df['url'].apply(self._extract_domain)

        logger.info(f"加载正常数据: {len(df)} 条")
        return df

    def _extract_domain(self, url):
        """从URL提取域名"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # 移除端口号
            if ':' in domain:
                domain = domain.split(':')[0]
            return domain.lower()
        except:
            return ''

    def merge_data(self, phishing_df, normal_df, balance=True, target_count=5000):
        """
        合并数据集

        Args:
            phishing_df: 钓鱼数据
            normal_df: 正常数据
            balance: 是否平衡数据
            target_count: 每类目标数量

        Returns:
            DataFrame: 合并后的数据
        """
        logger.info("开始合并数据...")

        # 平衡数据
        if balance:
            # 采样钓鱼数据
            if len(phishing_df) > target_count:
                phishing_df = phishing_df.sample(n=target_count, random_state=42)
            # 采样正常数据
            if len(normal_df) > target_count:
                normal_df = normal_df.sample(n=target_count, random_state=42)

            logger.info(f"平衡后 - 钓鱼: {len(phishing_df)}, 正常: {len(normal_df)}")

        # 合并
        merged = pd.concat([phishing_df, normal_df], ignore_index=True)

        # 选择需要的列
        columns = ['url', 'label', 'source', 'domain']
        merged = merged[columns]

        # 去重
        original_count = len(merged)
        merged = merged.drop_duplicates(subset=['url'], keep='first')
        removed_count = original_count - len(merged)

        if removed_count > 0:
            logger.info(f"移除重复URL: {removed_count} 条")

        # 重新生成ID
        merged = merged.reset_index(drop=True)
        merged['id'] = merged.index + 1

        # 调整列顺序
        merged = merged[['id', 'url', 'label', 'source', 'domain']]

        # 打乱顺序
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
        merged['id'] = merged.index + 1

        logger.info(f"合并后总数: {len(merged)} 条")
        return merged

    def split_data(self, df, test_size=0.2, random_state=42):
        """
        划分训练集和测试集

        Args:
            df: 完整数据集
            test_size: 测试集比例
            random_state: 随机种子

        Returns:
            tuple: (train_df, test_df)
        """
        logger.info(f"划分数据集 (测试集比例: {test_size})...")

        # 使用分层采样
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['label']  # 保持类别比例
        )

        # 重置索引
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # 更新ID
        train_df['id'] = train_df.index + 1
        test_df['id'] = test_df.index + 1

        logger.info(f"训练集: {len(train_df)} 条")
        logger.info(f"测试集: {len(test_df)} 条")

        return train_df, test_df

    def save_dataset(self, df, filename):
        """
        保存数据集

        Args:
            df: 数据集
            filename: 文件名

        Returns:
            str: 文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"保存数据集: {filepath}")
        return filepath

    def build(self, target_count=5000, test_size=0.2):
        """
        执行完整的数据集构建流程

        Args:
            target_count: 每类目标数量
            test_size: 测试集比例

        Returns:
            dict: 构建结果
        """
        print("\n" + "="*60)
        print("开始构建数据集")
        print("="*60)

        # 1. 加载数据
        phishing_df = self.load_phishing_data()
        normal_df = self.load_normal_data()

        # 2. 合并数据
        merged_df = self.merge_data(phishing_df, normal_df,
                                     balance=True, target_count=target_count)

        # 3. 划分数据
        train_df, test_df = self.split_data(merged_df, test_size=test_size)

        # 4. 保存数据
        dataset_path = self.save_dataset(merged_df, 'dataset.csv')
        train_path = self.save_dataset(train_df, 'train.csv')
        test_path = self.save_dataset(test_df, 'test.csv')

        # 5. 生成统计报告
        stats = self._generate_statistics(merged_df, train_df, test_df)

        # 6. 保存统计报告
        stats_path = os.path.join(self.output_dir, 'dataset_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        # 打印报告
        self._print_statistics(stats)

        return {
            'dataset': dataset_path,
            'train': train_path,
            'test': test_path,
            'statistics': stats
        }

    def _generate_statistics(self, full_df, train_df, test_df):
        """生成数据集统计信息"""
        stats = {
            'build_time': datetime.now().isoformat(),
            'full_dataset': {
                'total': len(full_df),
                'phishing': int((full_df['label'] == 1).sum()),
                'normal': int((full_df['label'] == 0).sum()),
                'ratio': f"{(full_df['label'] == 1).sum()}:{(full_df['label'] == 0).sum()}"
            },
            'train_set': {
                'total': len(train_df),
                'phishing': int((train_df['label'] == 1).sum()),
                'normal': int((train_df['label'] == 0).sum())
            },
            'test_set': {
                'total': len(test_df),
                'phishing': int((test_df['label'] == 1).sum()),
                'normal': int((test_df['label'] == 0).sum())
            },
            'sources': full_df['source'].value_counts().to_dict()
        }
        return stats

    def _print_statistics(self, stats):
        """打印统计信息"""
        print("\n" + "="*60)
        print("数据集构建完成")
        print("="*60)
        print(f"\n构建时间: {stats['build_time']}")

        print("\n[完整数据集]")
        print(f"  总数: {stats['full_dataset']['total']}")
        print(f"  钓鱼样本: {stats['full_dataset']['phishing']}")
        print(f"  正常样本: {stats['full_dataset']['normal']}")
        print(f"  比例: {stats['full_dataset']['ratio']}")

        print("\n[训练集]")
        print(f"  总数: {stats['train_set']['total']}")
        print(f"  钓鱼: {stats['train_set']['phishing']}")
        print(f"  正常: {stats['train_set']['normal']}")

        print("\n[测试集]")
        print(f"  总数: {stats['test_set']['total']}")
        print(f"  钓鱼: {stats['test_set']['phishing']}")
        print(f"  正常: {stats['test_set']['normal']}")

        print("\n[数据来源]")
        for source, count in stats['sources'].items():
            print(f"  {source}: {count}")

        print("="*60)


def main():
    """主函数"""
    builder = DatasetBuilder()
    result = builder.build(target_count=5000, test_size=0.2)

    print("\n[SUCCESS] 数据集构建完成!")
    print(f"完整数据集: {result['dataset']}")
    print(f"训练集: {result['train']}")
    print(f"测试集: {result['test']}")


if __name__ == '__main__':
    main()
