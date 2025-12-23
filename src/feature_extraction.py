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
import tldextract
import requests
import time
import warnings

# 禁用SSL警告（因为我们设置verify=False）
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

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

    # ==================== 高级特征 (9维) ====================

    def has_ip_address(self) -> int:
        """
        检测URL中是否包含IP地址

        使用IP地址而非域名是钓鱼网站的常见特征，
        因为这样可以绕过域名黑名单检测。

        Returns:
            int: 1表示包含IP地址，0表示不包含
        """
        pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        return 1 if re.search(pattern, self.url) else 0

    def has_at_symbol(self) -> int:
        """
        检测URL中是否包含@符号

        @符号在URL中可用于混淆用户，
        例如: http://trusted.com@evil.com 实际访问的是evil.com

        Returns:
            int: 1表示包含@符号，0表示不包含
        """
        return 1 if '@' in self.url else 0

    def num_subdomains(self) -> int:
        """
        统计子域名数量

        钓鱼网站常使用多级子域名来伪装，
        例如: www.paypal.com.evil.site

        Returns:
            int: 子域名的数量
        """
        try:
            extracted = tldextract.extract(self.url)
            subdomain = extracted.subdomain
            if not subdomain:
                return 0
            return subdomain.count('.') + 1
        except Exception:
            return 0

    def has_https(self) -> int:
        """
        检测是否使用HTTPS协议

        虽然HTTPS不能保证网站安全，但缺少HTTPS是可疑特征。
        现代合法网站通常都使用HTTPS。

        Returns:
            int: 1表示使用HTTPS，0表示不使用
        """
        return 1 if self.parsed.scheme == 'https' else 0

    def path_depth(self) -> int:
        """
        计算URL路径深度

        过深的路径层级可能表示复杂的重定向或伪装路径。

        Returns:
            int: 路径的深度（目录层级数）
        """
        path = self.parsed.path
        if not path or path == '/':
            return 0
        # 去除首尾斜杠后计算层级
        return path.strip('/').count('/') + 1

    def has_port(self) -> int:
        """
        检测是否显式指定端口号

        非标准端口（非80/443）可能是可疑特征。

        Returns:
            int: 1表示指定了端口，0表示未指定
        """
        return 1 if self.parsed.port else 0

    def entropy(self) -> float:
        """
        计算URL的信息熵

        信息熵衡量URL中字符的随机性。
        钓鱼URL常包含随机字符串，导致熵值较高。

        计算公式: H = -Σ(p × log2(p))

        Returns:
            float: URL的信息熵值
        """
        if not self.url:
            return 0.0
        # 计算每个字符的出现概率
        prob = [self.url.count(c) / len(self.url) for c in set(self.url)]
        # 应用信息熵公式
        return -sum(p * np.log2(p) for p in prob if p > 0)

    def is_shortening_service(self) -> int:
        """
        检测是否使用短链接服务

        短链接服务（如bit.ly、tinyurl等）常被用于隐藏真实URL，
        是钓鱼攻击的常用手段。

        Returns:
            int: 1表示是短链接，0表示不是
        """
        return 1 if any(s in self.domain for s in self.SHORTENING_SERVICES) else 0

    def has_suspicious_words(self) -> int:
        """
        检测URL中是否包含可疑关键词

        钓鱼URL常包含品牌名称或诱导性词汇，
        如: login, verify, secure, paypal等

        Returns:
            int: 1表示包含可疑词，0表示不包含
        """
        url_lower = self.url.lower()
        return 1 if any(w in url_lower for w in self.SUSPICIOUS_WORDS) else 0

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

    def extract_all_url_features(self) -> Dict[str, Union[int, float]]:
        """
        提取全部17个URL词法特征

        包括：
        - 基础特征 (8维): 长度和字符统计
        - 高级特征 (9维): 语义和结构分析

        Returns:
            dict: 包含17个特征的字典
        """
        return {
            # 基础特征 (8维)
            'url_length': self.url_length(),
            'domain_length': self.domain_length(),
            'path_length': self.path_length(),
            'num_dots': self.num_dots(),
            'num_hyphens': self.num_hyphens(),
            'num_underscores': self.num_underscores(),
            'num_slashes': self.num_slashes(),
            'num_digits': self.num_digits(),
            # 高级特征 (9维)
            'has_ip': self.has_ip_address(),
            'has_at': self.has_at_symbol(),
            'num_subdomains': self.num_subdomains(),
            'has_https': self.has_https(),
            'path_depth': self.path_depth(),
            'has_port': self.has_port(),
            'entropy': self.entropy(),
            'is_shortening': self.is_shortening_service(),
            'has_suspicious': self.has_suspicious_words()
        }

    def extract_all_url_features_array(self) -> np.ndarray:
        """
        以numpy数组格式返回全部17个URL特征

        Returns:
            ndarray: 包含17个特征值的一维数组
        """
        features = self.extract_all_url_features()
        return np.array(list(features.values()))

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
    便捷函数：提取单个URL的基础特征（8维）

    Args:
        url: URL字符串

    Returns:
        dict: 特征字典
    """
    extractor = URLFeatureExtractor(url)
    return extractor.extract_basic_features()


def extract_url_all_features(url: str) -> Dict[str, Union[int, float]]:
    """
    便捷函数：提取单个URL的全部特征（17维）

    Args:
        url: URL字符串

    Returns:
        dict: 特征字典
    """
    extractor = URLFeatureExtractor(url)
    return extractor.extract_all_url_features()


def batch_extract_basic_features(urls: List[str]) -> List[Dict[str, int]]:
    """
    批量提取URL基础特征（8维）

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


def batch_extract_all_features(urls: List[str]) -> List[Dict[str, Union[int, float]]]:
    """
    批量提取URL全部特征（17维）

    Args:
        urls: URL列表

    Returns:
        list: 特征字典列表
    """
    from tqdm import tqdm
    results = []
    for url in tqdm(urls, desc="提取全部URL特征"):
        try:
            features = extract_url_all_features(url)
            results.append(features)
        except Exception as e:
            logger.warning(f"特征提取失败 {url}: {e}")
            # 返回默认值（17维）
            results.append({
                'url_length': len(url) if url else 0,
                'domain_length': 0,
                'path_length': 0,
                'num_dots': 0,
                'num_hyphens': 0,
                'num_underscores': 0,
                'num_slashes': 0,
                'num_digits': 0,
                'has_ip': 0,
                'has_at': 0,
                'num_subdomains': 0,
                'has_https': 0,
                'path_depth': 0,
                'has_port': 0,
                'entropy': 0.0,
                'is_shortening': 0,
                'has_suspicious': 0
            })
    return results


# ==================== HTTP响应特征提取器 ====================

class HTTPFeatureExtractor:
    """
    HTTP响应特征提取器

    通过发送HTTP请求获取目标网站的响应信息，提取5维特征。
    这些特征需要实际访问目标网站，提取速度较慢（通常1-10秒）。

    对于无法访问的URL，所有特征返回默认值-1。

    Attributes:
        url (str): 目标URL
        TIMEOUT (int): 请求超时时间（秒）
        USER_AGENT (str): 浏览器User-Agent

    Example:
        >>> extractor = HTTPFeatureExtractor("https://google.com")
        >>> features = extractor.extract_all()
        >>> print(features['http_status_code'])
        200
    """

    # 请求超时时间（秒）
    TIMEOUT = 10

    # 模拟Chrome浏览器的User-Agent
    USER_AGENT = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )

    # 常见的Web服务器类型
    KNOWN_SERVERS = ['nginx', 'apache', 'iis', 'cloudflare', 'gws', 'microsoft']

    def __init__(self, url: str):
        """
        初始化HTTP特征提取器

        Args:
            url: 目标URL字符串
        """
        self.url = url.strip() if url else ''

        # 补全协议前缀
        if self.url and not self.url.startswith(('http://', 'https://')):
            self.url = 'http://' + self.url

        # 响应对象（惰性加载）
        self._response = None
        self._response_time = -1
        self._fetched = False
        self._error = None

    def _fetch(self):
        """
        发送HTTP请求获取响应（惰性加载）

        只在第一次调用特征方法时发送请求，
        后续调用复用已获取的响应对象。
        """
        if self._fetched:
            return

        self._fetched = True

        if not self.url:
            self._error = "URL为空"
            return

        try:
            # 记录请求开始时间
            start_time = time.time()

            # 发送GET请求
            self._response = requests.get(
                self.url,
                timeout=self.TIMEOUT,
                allow_redirects=True,  # 允许重定向
                headers={'User-Agent': self.USER_AGENT},
                verify=False  # 忽略SSL证书验证
            )

            # 计算响应时间
            self._response_time = time.time() - start_time

        except requests.exceptions.Timeout:
            self._error = "请求超时"
            self._response = None
            self._response_time = -1

        except requests.exceptions.ConnectionError:
            self._error = "连接失败"
            self._response = None
            self._response_time = -1

        except requests.exceptions.SSLError:
            self._error = "SSL错误"
            self._response = None
            self._response_time = -1

        except requests.exceptions.TooManyRedirects:
            self._error = "重定向过多"
            self._response = None
            self._response_time = -1

        except requests.exceptions.RequestException as e:
            self._error = f"请求异常: {str(e)}"
            self._response = None
            self._response_time = -1

        except Exception as e:
            self._error = f"未知错误: {str(e)}"
            self._response = None
            self._response_time = -1

    def http_status_code(self) -> int:
        """
        获取HTTP响应状态码

        常见状态码含义：
        - 200: 请求成功
        - 301/302: 重定向
        - 403: 禁止访问
        - 404: 页面不存在
        - 500: 服务器错误
        - -1: 请求失败

        Returns:
            int: HTTP状态码，失败返回-1
        """
        self._fetch()
        if self._response is None:
            return -1
        return self._response.status_code

    def http_response_time(self) -> float:
        """
        获取HTTP响应时间

        计算从发送请求到收到完整响应的时间。
        钓鱼网站通常响应较慢。

        Returns:
            float: 响应时间（秒），保留3位小数，失败返回-1
        """
        self._fetch()
        if self._response_time < 0:
            return -1
        return round(self._response_time, 3)

    def http_redirect_count(self) -> int:
        """
        获取重定向次数

        统计请求过程中经历的重定向跳转次数。
        钓鱼网站常有多次重定向以隐藏真实地址。

        Returns:
            int: 重定向次数，失败返回-1
        """
        self._fetch()
        if self._response is None:
            return -1
        # response.history 是重定向响应的列表
        return len(self._response.history)

    def content_length(self) -> int:
        """
        获取响应内容长度

        钓鱼页面通常较简单，内容长度较小。

        Returns:
            int: 内容字节数，失败返回-1
        """
        self._fetch()
        if self._response is None:
            return -1
        return len(self._response.content)

    def server_type(self) -> int:
        """
        获取服务器类型编码

        从Server响应头提取服务器信息并编码：
        - 1: 常见服务器（nginx, apache, iis, cloudflare等）
        - 0: 其他服务器
        - -1: 无Server头信息

        Returns:
            int: 服务器类型编码
        """
        self._fetch()
        if self._response is None:
            return -1

        # 获取Server响应头
        server = self._response.headers.get('Server', '').lower()

        if not server:
            return -1

        # 检查是否为常见服务器
        for known in self.KNOWN_SERVERS:
            if known in server:
                return 1

        return 0

    def extract_all(self) -> Dict[str, Union[int, float]]:
        """
        提取全部5维HTTP响应特征

        Returns:
            dict: 包含5个HTTP特征的字典
        """
        return {
            'http_status_code': self.http_status_code(),
            'http_response_time': self.http_response_time(),
            'http_redirect_count': self.http_redirect_count(),
            'content_length': self.content_length(),
            'server_type': self.server_type()
        }

    def get_error(self) -> str:
        """
        获取请求错误信息

        Returns:
            str: 错误信息，无错误返回None
        """
        self._fetch()
        return self._error


def extract_http_features(url: str) -> Dict[str, Union[int, float]]:
    """
    便捷函数：提取单个URL的HTTP响应特征

    Args:
        url: URL字符串

    Returns:
        dict: HTTP特征字典
    """
    extractor = HTTPFeatureExtractor(url)
    return extractor.extract_all()


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

    print("URL特征提取测试 - 全部17维特征")
    print("=" * 60)

    for url in test_urls:
        print(f"\nURL: {url}")
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_all_url_features()
        print(f"特征数量: {len(features)}")
        for name, value in features.items():
            print(f"  {name}: {value}")

    # 验收检查
    print("\n" + "=" * 60)
    print("验收检查")
    print("=" * 60)

    for url in test_urls:
        ext = URLFeatureExtractor(url)
        features = ext.extract_all_url_features()
        assert len(features) == 17, f"应该有17个特征，实际有{len(features)}个"

    print("\n[OK] Day 7验证通过! 全部17个URL词法特征已实现。")
