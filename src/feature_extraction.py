"""
特征提取模块
基于网络流量特征的钓鱼网站检测系统

此模块负责:
- URL词法特征提取 (17维)
- TLS证书特征提取 (5维)
- HTTP响应特征提取 (5维)
- DNS特征提取 (3维)

作者: 毕业设计项目组
日期: 2025年12月
"""

import re
import numpy as np
from urllib.parse import urlparse
from typing import Dict, List, Union, Optional
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SHORTENING_SERVICES, SUSPICIOUS_WORDS

logger = logging.getLogger(__name__)


class URLFeatureExtractor:
    """
    URL词法特征提取器

    从URL字符串中提取17维词法特征，用于钓鱼网站检测。
    这些特征无需访问目标网站，仅通过分析URL字符串即可获得。

    Attributes:
        url (str): 原始URL字符串
        parsed: urlparse解析结果
        domain (str): 域名部分

    Example:
        >>> extractor = URLFeatureExtractor("https://example.com/path")
        >>> features = extractor.extract_basic_features()
        >>> print(features['url_length'])
        25
    """

    # 常见短链接服务
    SHORTENING_SERVICES = SHORTENING_SERVICES

    # 可疑关键词
    SUSPICIOUS_WORDS = SUSPICIOUS_WORDS

    def __init__(self, url: str):
        """
        初始化特征提取器

        Args:
            url: 待提取特征的URL字符串
        """
        # 统一转换为小写，确保处理一致性
        self.url = url.strip().lower() if url else ''

        # 使用标准库urlparse解析URL
        self.parsed = urlparse(self.url)

        # 获取域名部分
        self.domain = self.parsed.netloc

        # 移除端口号
        if ':' in self.domain:
            self.domain = self.domain.split(':')[0]

    # ==================== 长度类特征 (3维) ====================

    def url_length(self) -> int:
        """
        计算URL总长度

        钓鱼URL通常较长，因为攻击者需要在URL中嵌入伪装信息
        或使用随机字符来绕过检测。

        Returns:
            int: URL字符串的总长度
        """
        return len(self.url)

    def domain_length(self) -> int:
        """
        计算域名长度

        钓鱼域名可能很长，以包含品牌名称或伪装信息。

        Returns:
            int: 域名部分的字符长度
        """
        return len(self.domain)

    def path_length(self) -> int:
        """
        计算路径长度

        复杂或过长的路径可能是可疑特征。

        Returns:
            int: URL路径部分的字符长度
        """
        return len(self.parsed.path)

    # ==================== 字符统计特征 (5维) ====================

    def num_dots(self) -> int:
        """
        统计点号(.)数量

        钓鱼URL常有多级子域名导致点号较多，
        例如: www.paypal.com.evil.site.com

        Returns:
            int: URL中点号的数量
        """
        return self.url.count('.')

    def num_hyphens(self) -> int:
        """
        统计连字符(-)数量

        连字符常用于伪造相似域名，
        例如: paypal-secure.com, amazon-login.com

        Returns:
            int: URL中连字符的数量
        """
        return self.url.count('-')

    def num_underscores(self) -> int:
        """
        统计下划线(_)数量

        在域名中使用下划线是非常规做法，可能是可疑特征。
        正常网站域名极少使用下划线。

        Returns:
            int: URL中下划线的数量
        """
        return self.url.count('_')

    def num_slashes(self) -> int:
        """
        统计斜杠(/)数量

        斜杠数量反映URL路径的复杂程度，
        过多的斜杠可能表示复杂的重定向路径。

        Returns:
            int: URL中斜杠的数量
        """
        return self.url.count('/')

    def num_digits(self) -> int:
        """
        统计数字字符数量

        钓鱼URL常包含随机数字串用于绕过检测，
        例如: bank123456.com, login.site/abc123

        Returns:
            int: URL中数字字符的总数
        """
        return sum(c.isdigit() for c in self.url)

    # ==================== 基础特征提取方法 ====================

    def extract_basic_features(self) -> Dict[str, int]:
        """
        提取基础词法特征（8维）

        包括长度类特征(3维)和字符统计特征(5维)。

        Returns:
            dict: 包含8个特征的字典
        """
        return {
            # 长度类特征
            'url_length': self.url_length(),
            'domain_length': self.domain_length(),
            'path_length': self.path_length(),

            # 字符统计特征
            'num_dots': self.num_dots(),
            'num_hyphens': self.num_hyphens(),
            'num_underscores': self.num_underscores(),
            'num_slashes': self.num_slashes(),
            'num_digits': self.num_digits()
        }

    def extract_basic_features_array(self) -> np.ndarray:
        """
        以numpy数组格式返回基础特征

        Returns:
            ndarray: 包含8个特征值的一维数组
        """
        features = self.extract_basic_features()
        return np.array(list(features.values()))


# ==================== 便捷函数 ====================

def extract_url_basic_features(url: str) -> Dict[str, int]:
    """
    便捷函数：提取单个URL的基础特征

    Args:
        url: URL字符串

    Returns:
        dict: 特征字典
    """
    extractor = URLFeatureExtractor(url)
    return extractor.extract_basic_features()


def batch_extract_basic_features(urls: List[str]) -> List[Dict[str, int]]:
    """
    批量提取URL基础特征

    Args:
        urls: URL列表

    Returns:
        list: 特征字典列表
    """
    from tqdm import tqdm
    results = []
    for url in tqdm(urls, desc="提取基础特征"):
        try:
            features = extract_url_basic_features(url)
            results.append(features)
        except Exception as e:
            logger.warning(f"特征提取失败 {url}: {e}")
            # 返回默认值
            results.append({
                'url_length': len(url) if url else 0,
                'domain_length': 0,
                'path_length': 0,
                'num_dots': 0,
                'num_hyphens': 0,
                'num_underscores': 0,
                'num_slashes': 0,
                'num_digits': 0
            })
    return results


# ==================== 预留的其他特征提取器 ====================

class HTTPFeatureExtractor:
    """HTTP响应特征提取器"""
    # TODO: Day 8-9 实现
    pass


class SSLFeatureExtractor:
    """SSL证书特征提取器"""
    # TODO: Day 10-11 实现
    pass


class DNSFeatureExtractor:
    """DNS特征提取器"""
    # TODO: Day 12-13 实现
    pass


class FeatureExtractor:
    """综合特征提取器"""
    # TODO: 整合所有特征提取器
    pass


if __name__ == '__main__':
    # 测试代码
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/admin/login.php",
        "https://paypal-secure-login.fake-site.com/verify",
        "http://bit.ly/abc123"
    ]

    print("URL特征提取测试")
    print("=" * 60)

    for url in test_urls:
        print(f"\nURL: {url}")
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_basic_features()
        for name, value in features.items():
            print(f"  {name}: {value}")
