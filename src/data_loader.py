"""
数据加载模块
提供统一的数据加载接口

作者: 毕业设计项目组
日期: 2025年12月
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_DIR
from src.utils import logger


class DataLoader:
    """
    数据加载器

    提供加载训练集、测试集和完整数据集的统一接口
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录路径，默认使用配置中的PROCESSED_DATA_DIR
        """
        self.data_dir = data_dir or PROCESSED_DATA_DIR

    def load_dataset(self) -> pd.DataFrame:
        """
        加载完整数据集

        Returns:
            DataFrame: 完整数据集
        """
        filepath = os.path.join(self.data_dir, 'dataset.csv')
        df = pd.read_csv(filepath)
        logger.info(f"加载完整数据集: {len(df)} 条")
        return df

    def load_train(self) -> pd.DataFrame:
        """
        加载训练集

        Returns:
            DataFrame: 训练集
        """
        filepath = os.path.join(self.data_dir, 'train.csv')
        df = pd.read_csv(filepath)
        logger.info(f"加载训练集: {len(df)} 条")
        return df

    def load_test(self) -> pd.DataFrame:
        """
        加载测试集

        Returns:
            DataFrame: 测试集
        """
        filepath = os.path.join(self.data_dir, 'test.csv')
        df = pd.read_csv(filepath)
        logger.info(f"加载测试集: {len(df)} 条")
        return df

    def load_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        同时加载训练集和测试集

        Returns:
            tuple: (训练集, 测试集)
        """
        return self.load_train(), self.load_test()

    def load_urls(self, dataset: str = 'train') -> pd.Series:
        """
        只加载URL列

        Args:
            dataset: 'train', 'test', 或 'all'

        Returns:
            Series: URL列表
        """
        if dataset == 'train':
            df = self.load_train()
        elif dataset == 'test':
            df = self.load_test()
        else:
            df = self.load_dataset()

        return df['url']

    def load_with_labels(self, dataset: str = 'train') -> Tuple[pd.Series, pd.Series]:
        """
        加载URL和标签

        Args:
            dataset: 'train', 'test', 或 'all'

        Returns:
            tuple: (URLs, labels)
        """
        if dataset == 'train':
            df = self.load_train()
        elif dataset == 'test':
            df = self.load_test()
        else:
            df = self.load_dataset()

        return df['url'], df['label']

    def get_statistics(self) -> dict:
        """
        获取数据集统计信息

        Returns:
            dict: 统计信息
        """
        dataset = self.load_dataset()
        train = self.load_train()
        test = self.load_test()

        stats = {
            'total_samples': len(dataset),
            'train_samples': len(train),
            'test_samples': len(test),
            'phishing_total': int((dataset['label'] == 1).sum()),
            'normal_total': int((dataset['label'] == 0).sum()),
            'train_phishing': int((train['label'] == 1).sum()),
            'train_normal': int((train['label'] == 0).sum()),
            'test_phishing': int((test['label'] == 1).sum()),
            'test_normal': int((test['label'] == 0).sum()),
            'sources': dataset['source'].value_counts().to_dict() if 'source' in dataset.columns else {}
        }

        return stats

    def print_info(self):
        """打印数据集信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 50)
        print("数据集信息")
        print("=" * 50)
        print(f"总样本数: {stats['total_samples']}")
        print(f"  - 钓鱼: {stats['phishing_total']}")
        print(f"  - 正常: {stats['normal_total']}")
        print(f"\n训练集: {stats['train_samples']}")
        print(f"  - 钓鱼: {stats['train_phishing']}")
        print(f"  - 正常: {stats['train_normal']}")
        print(f"\n测试集: {stats['test_samples']}")
        print(f"  - 钓鱼: {stats['test_phishing']}")
        print(f"  - 正常: {stats['test_normal']}")

        if stats['sources']:
            print(f"\n数据来源分布:")
            for source, count in stats['sources'].items():
                print(f"  - {source}: {count}")

        print("=" * 50)

    def get_label_distribution(self) -> dict:
        """
        获取标签分布

        Returns:
            dict: 各数据集的标签分布
        """
        stats = self.get_statistics()
        return {
            'total': {
                'phishing': stats['phishing_total'],
                'normal': stats['normal_total'],
                'ratio': stats['phishing_total'] / stats['normal_total'] if stats['normal_total'] > 0 else 0
            },
            'train': {
                'phishing': stats['train_phishing'],
                'normal': stats['train_normal'],
                'ratio': stats['train_phishing'] / stats['train_normal'] if stats['train_normal'] > 0 else 0
            },
            'test': {
                'phishing': stats['test_phishing'],
                'normal': stats['test_normal'],
                'ratio': stats['test_phishing'] / stats['test_normal'] if stats['test_normal'] > 0 else 0
            }
        }


# 便捷函数
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """快速加载训练和测试数据"""
    loader = DataLoader()
    return loader.load_train_test()


def load_all() -> pd.DataFrame:
    """加载完整数据集"""
    loader = DataLoader()
    return loader.load_dataset()


if __name__ == '__main__':
    # 测试数据加载器
    loader = DataLoader()
    loader.print_info()

    # 测试加载功能
    print("\n测试加载功能:")
    train, test = loader.load_train_test()
    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")

    # 显示标签分布
    print("\n标签分布:")
    dist = loader.get_label_distribution()
    print(f"总体比例(钓鱼/正常): {dist['total']['ratio']:.2f}")
