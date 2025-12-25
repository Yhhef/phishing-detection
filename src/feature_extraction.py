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
import ssl
import socket
import numpy as np
from urllib.parse import urlparse
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime
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


# ==================== TLS/SSL证书特征提取器 ====================

class SSLFeatureExtractor:
    """
    TLS/SSL证书特征提取器

    通过SSL连接获取目标网站的证书信息，提取5维特征。
    这些特征需要建立SSL连接，提取速度较慢（通常1-10秒）。

    对于非HTTPS网站或无法建立SSL连接的URL，所有特征返回默认值。

    Attributes:
        url (str): 目标URL
        domain (str): 域名（不含端口）
        TIMEOUT (int): 连接超时时间（秒）

    Example:
        >>> extractor = SSLFeatureExtractor("https://google.com")
        >>> features = extractor.extract_all()
        >>> print(features['ssl_cert_valid'])
        1
    """

    # 连接超时时间（秒）
    TIMEOUT = 10

    # 知名的商业证书颁发机构
    KNOWN_CAS = [
        'DigiCert', 'Comodo', 'GlobalSign', 'Symantec',
        'GeoTrust', 'Thawte', 'VeriSign', 'Entrust',
        'GoDaddy', 'Sectigo', 'RapidSSL'
    ]

    # 免费的证书颁发机构
    FREE_CAS = [
        "Let's Encrypt", 'ZeroSSL', 'Buypass',
        'Cloudflare', 'Google Trust Services'
    ]

    def __init__(self, url: str):
        """
        初始化SSL特征提取器

        Args:
            url: 目标URL字符串
        """
        self.url = url.strip() if url else ''

        # 解析URL获取域名
        try:
            parsed = urlparse(self.url)
            self.domain = parsed.netloc
            # 移除端口号
            if ':' in self.domain:
                self.domain = self.domain.split(':')[0]
        except Exception:
            self.domain = ''

        # 证书信息（惰性加载）
        self._cert = None
        self._fetched = False
        self._error = None

    def _fetch_cert(self):
        """
        获取SSL证书信息（惰性加载）

        只在第一次调用特征方法时获取证书，
        后续调用复用已获取的证书信息。
        """
        if self._fetched:
            return

        self._fetched = True

        if not self.domain:
            self._error = "域名为空"
            return

        try:
            # 创建SSL上下文
            context = ssl.create_default_context()

            # 建立TCP连接
            with socket.create_connection(
                (self.domain, 443),
                timeout=self.TIMEOUT
            ) as sock:
                # 包装为SSL连接
                with context.wrap_socket(
                    sock,
                    server_hostname=self.domain
                ) as ssock:
                    # 获取证书信息（字典格式）
                    self._cert = ssock.getpeercert()

        except ssl.SSLCertVerificationError as e:
            self._error = f"证书验证失败: {str(e)}"
            # 尝试不验证证书获取信息
            self._fetch_cert_unverified()

        except ssl.SSLError as e:
            self._error = f"SSL错误: {str(e)}"
            self._cert = None

        except socket.timeout:
            self._error = "连接超时"
            self._cert = None

        except socket.error as e:
            self._error = f"连接失败: {str(e)}"
            self._cert = None

        except Exception as e:
            self._error = f"未知错误: {str(e)}"
            self._cert = None

    def _fetch_cert_unverified(self):
        """
        不验证证书获取信息（用于处理自签名或过期证书）
        """
        try:
            # 创建不验证证书的上下文
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            with socket.create_connection(
                (self.domain, 443),
                timeout=self.TIMEOUT
            ) as sock:
                with context.wrap_socket(
                    sock,
                    server_hostname=self.domain
                ) as ssock:
                    # 获取证书（DER格式）
                    cert_der = ssock.getpeercert(binary_form=True)
                    if cert_der:
                        # 尝试获取解析后的证书
                        self._cert = ssock.getpeercert()

        except Exception as e:
            self._error = f"获取证书失败: {str(e)}"
            self._cert = None

    def _parse_cert_date(self, date_str: str) -> Optional[datetime]:
        """
        解析证书日期字符串

        Args:
            date_str: 证书日期字符串，如 'Dec 31 23:59:59 2024 GMT'

        Returns:
            datetime对象，解析失败返回None
        """
        if not date_str:
            return None

        try:
            # 标准格式
            return datetime.strptime(date_str, '%b %d %H:%M:%S %Y %Z')
        except ValueError:
            pass

        try:
            # 备用格式
            return datetime.strptime(date_str, '%b %d %H:%M:%S %Y')
        except ValueError:
            pass

        return None

    def _get_issuer_org(self) -> str:
        """获取证书颁发者组织名"""
        if not self._cert:
            return ''

        try:
            issuer = self._cert.get('issuer', [])
            # issuer格式: ((('组织名类型', '组织名'),), ...)
            for item in issuer:
                for key, value in item:
                    if key == 'organizationName':
                        return value
        except Exception:
            pass

        return ''

    def _get_subject_org(self) -> str:
        """获取证书主体组织名"""
        if not self._cert:
            return ''

        try:
            subject = self._cert.get('subject', [])
            for item in subject:
                for key, value in item:
                    if key == 'organizationName':
                        return value
        except Exception:
            pass

        return ''

    def ssl_cert_valid(self) -> int:
        """
        检测SSL证书是否有效

        能够成功获取证书信息即认为有效。

        Returns:
            int: 1=有效, 0=无效或无法获取
        """
        self._fetch_cert()
        return 1 if self._cert else 0

    def ssl_cert_days(self) -> int:
        """
        获取证书剩余有效天数

        计算当前时间到证书过期时间的天数。
        负数表示证书已过期。

        Returns:
            int: 剩余天数，-1表示无法获取
        """
        self._fetch_cert()

        if not self._cert:
            return -1

        try:
            not_after = self._cert.get('notAfter', '')
            expire_date = self._parse_cert_date(not_after)

            if expire_date:
                delta = expire_date - datetime.now()
                return delta.days

        except Exception:
            pass

        return -1

    def ssl_issuer_type(self) -> int:
        """
        获取证书颁发机构类型

        编码规则：
        - 1: 知名商业CA（DigiCert、Comodo等）
        - 0: 免费CA（Let's Encrypt等）
        - -1: 其他或自签名

        Returns:
            int: 颁发机构类型编码
        """
        self._fetch_cert()

        if not self._cert:
            return -1

        issuer_org = self._get_issuer_org()

        if not issuer_org:
            return -1

        # 检查是否为知名CA
        for ca in self.KNOWN_CAS:
            if ca.lower() in issuer_org.lower():
                return 1

        # 检查是否为免费CA
        for ca in self.FREE_CAS:
            if ca.lower() in issuer_org.lower():
                return 0

        return -1

    def ssl_self_signed(self) -> int:
        """
        检测是否为自签名证书

        自签名证书的颁发者和主体相同。

        Returns:
            int: 1=自签名, 0=非自签名, -1=无法判断
        """
        self._fetch_cert()

        if not self._cert:
            return -1

        issuer_org = self._get_issuer_org()
        subject_org = self._get_subject_org()

        if not issuer_org or not subject_org:
            return -1

        # 颁发者和主体相同则为自签名
        return 1 if issuer_org.lower() == subject_org.lower() else 0

    def ssl_cert_age(self) -> int:
        """
        获取证书已颁发天数

        计算证书颁发时间到当前时间的天数。

        Returns:
            int: 已颁发天数，-1表示无法获取
        """
        self._fetch_cert()

        if not self._cert:
            return -1

        try:
            not_before = self._cert.get('notBefore', '')
            issue_date = self._parse_cert_date(not_before)

            if issue_date:
                delta = datetime.now() - issue_date
                return delta.days

        except Exception:
            pass

        return -1

    def extract_all(self) -> Dict[str, int]:
        """
        提取全部5维SSL证书特征

        Returns:
            dict: 包含5个SSL特征的字典
        """
        return {
            'ssl_cert_valid': self.ssl_cert_valid(),
            'ssl_cert_days': self.ssl_cert_days(),
            'ssl_issuer_type': self.ssl_issuer_type(),
            'ssl_self_signed': self.ssl_self_signed(),
            'ssl_cert_age': self.ssl_cert_age()
        }

    def get_error(self) -> Optional[str]:
        """
        获取错误信息

        Returns:
            str: 错误信息，无错误返回None
        """
        self._fetch_cert()
        return self._error

    def get_cert_info(self) -> Optional[Dict]:
        """
        获取完整证书信息（用于调试）

        Returns:
            dict: 证书信息字典，无证书返回None
        """
        self._fetch_cert()
        return self._cert


def extract_ssl_features(url: str) -> Dict[str, int]:
    """
    便捷函数：提取单个URL的SSL证书特征

    Args:
        url: URL字符串

    Returns:
        dict: SSL特征字典
    """
    extractor = SSLFeatureExtractor(url)
    return extractor.extract_all()


# ==================== DNS特征提取器 ====================

class DNSFeatureExtractor:
    """
    DNS特征提取器

    通过DNS查询获取域名的解析信息，提取3维特征。

    Attributes:
        url (str): 目标URL
        domain (str): 域名
        TIMEOUT (int): DNS查询超时时间（秒）

    Example:
        >>> extractor = DNSFeatureExtractor("https://google.com")
        >>> features = extractor.extract_all()
        >>> print(features['dns_record_count'])
        5
    """

    # DNS查询超时时间（秒）
    TIMEOUT = 5

    def __init__(self, url: str):
        """
        初始化DNS特征提取器

        Args:
            url: 目标URL字符串
        """
        self.url = url.strip() if url else ''

        # 解析URL获取域名
        try:
            parsed = urlparse(self.url if '://' in self.url else f'http://{self.url}')
            self.domain = parsed.netloc
            # 移除端口号
            if ':' in self.domain:
                self.domain = self.domain.split(':')[0]
        except Exception:
            self.domain = ''

        # DNS查询结果（惰性加载）
        self._dns_result = None
        self._resolve_time = -1
        self._fetched = False
        self._error = None

    def _fetch_dns(self):
        """
        执行DNS查询（惰性加载）
        """
        if self._fetched:
            return

        self._fetched = True

        if not self.domain:
            self._error = "域名为空"
            return

        # 设置socket超时
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(self.TIMEOUT)

        try:
            # 记录开始时间
            start_time = time.time()

            # 执行DNS查询
            # gethostbyname_ex返回: (hostname, aliaslist, ipaddrlist)
            self._dns_result = socket.gethostbyname_ex(self.domain)

            # 计算解析时间
            self._resolve_time = time.time() - start_time

        except socket.gaierror as e:
            self._error = f"DNS解析失败: {str(e)}"
            self._dns_result = None
            self._resolve_time = -1

        except socket.timeout:
            self._error = "DNS查询超时"
            self._dns_result = None
            self._resolve_time = -1

        except Exception as e:
            self._error = f"未知错误: {str(e)}"
            self._dns_result = None
            self._resolve_time = -1

        finally:
            # 恢复原始超时设置
            socket.setdefaulttimeout(original_timeout)

    def domain_entropy(self) -> float:
        """
        计算域名字符串的信息熵

        信息熵反映字符串的随机程度。
        随机生成的域名（如DGA生成）熵值通常较高。

        计算公式：H = -Σ(p × log2(p))
        其中p为各字符出现的概率。

        Returns:
            float: 域名的信息熵值，保留4位小数
        """
        if not self.domain:
            return 0.0

        # 统计每个字符出现的次数
        char_count = {}
        for c in self.domain:
            char_count[c] = char_count.get(c, 0) + 1

        # 计算概率和熵
        length = len(self.domain)
        entropy_value = 0.0
        for count in char_count.values():
            prob = count / length
            if prob > 0:
                entropy_value -= prob * np.log2(prob)

        return round(entropy_value, 4)

    def dns_resolve_time(self) -> float:
        """
        获取DNS解析时间

        测量从发起DNS查询到获得结果的时间。
        新注册的域名或CDN域名可能解析较慢。

        Returns:
            float: 解析时间（秒），保留4位小数，-1表示解析失败
        """
        self._fetch_dns()

        if self._resolve_time < 0:
            return -1

        return round(self._resolve_time, 4)

    def dns_record_count(self) -> int:
        """
        获取域名对应的IP地址数量

        大型网站通常有多个IP地址用于负载均衡。
        只有一个IP的域名可能是可疑特征。

        Returns:
            int: IP记录数量，-1表示解析失败
        """
        self._fetch_dns()

        if self._dns_result is None:
            return -1

        # dns_result格式: (hostname, aliaslist, ipaddrlist)
        ip_list = self._dns_result[2]
        return len(ip_list)

    def extract_all(self) -> Dict[str, Union[int, float]]:
        """
        提取全部3维DNS特征

        Returns:
            dict: 包含3个DNS特征的字典
        """
        return {
            'domain_entropy': self.domain_entropy(),
            'dns_resolve_time': self.dns_resolve_time(),
            'dns_record_count': self.dns_record_count()
        }

    def get_error(self) -> Optional[str]:
        """
        获取错误信息

        Returns:
            str: 错误信息，无错误返回None
        """
        self._fetch_dns()
        return self._error

    def get_ip_addresses(self) -> list:
        """
        获取解析到的IP地址列表

        Returns:
            list: IP地址列表，解析失败返回空列表
        """
        self._fetch_dns()

        if self._dns_result is None:
            return []

        return self._dns_result[2]


def extract_dns_features(url: str) -> Dict[str, Union[int, float]]:
    """
    便捷函数：提取单个URL的DNS特征

    Args:
        url: URL字符串

    Returns:
        dict: DNS特征字典
    """
    extractor = DNSFeatureExtractor(url)
    return extractor.extract_all()


# ==================== 特征提取主类 ====================

class FeatureExtractor:
    """
    特征提取主类

    整合URL词法特征（17维）、HTTP响应特征（5维）、
    SSL证书特征（5维）和DNS特征（3维），共30维特征。

    提供统一的特征提取接口，支持仅提取URL特征（快速）
    或提取全部特征（需要网络请求，较慢）。

    Attributes:
        url (str): 目标URL
        url_extractor: URL词法特征提取器
        http_extractor: HTTP响应特征提取器
        ssl_extractor: SSL证书特征提取器
        dns_extractor: DNS特征提取器

    Example:
        >>> extractor = FeatureExtractor("https://google.com")
        >>> # 仅URL特征（快速，17维）
        >>> url_features = extractor.extract_url_only()
        >>> # 全部特征（较慢，30维）
        >>> all_features = extractor.extract_all()
    """

    # 特征名称列表（按顺序）
    URL_FEATURE_NAMES = [
        'url_length', 'domain_length', 'path_length',
        'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits',
        'has_ip', 'has_at', 'num_subdomains', 'has_https', 'path_depth',
        'has_port', 'entropy', 'is_shortening', 'has_suspicious'
    ]

    HTTP_FEATURE_NAMES = [
        'http_status_code', 'http_response_time', 'http_redirect_count',
        'content_length', 'server_type'
    ]

    SSL_FEATURE_NAMES = [
        'ssl_cert_valid', 'ssl_cert_days', 'ssl_issuer_type',
        'ssl_self_signed', 'ssl_cert_age'
    ]

    DNS_FEATURE_NAMES = [
        'domain_entropy', 'dns_resolve_time', 'dns_record_count'
    ]

    ALL_FEATURE_NAMES = URL_FEATURE_NAMES + HTTP_FEATURE_NAMES + SSL_FEATURE_NAMES + DNS_FEATURE_NAMES

    def __init__(self, url: str):
        """
        初始化特征提取器

        Args:
            url: 目标URL字符串
        """
        self.url = url.strip() if url else ''

        # 初始化各个子提取器
        self.url_extractor = URLFeatureExtractor(self.url)
        self.http_extractor = HTTPFeatureExtractor(self.url)
        self.ssl_extractor = SSLFeatureExtractor(self.url)
        self.dns_extractor = DNSFeatureExtractor(self.url)

    def extract_url_only(self) -> Dict[str, Union[int, float]]:
        """
        仅提取URL词法特征（17维）

        无需网络请求，速度快（毫秒级）。
        适用于需要快速分析大量URL的场景。

        Returns:
            dict: 17维URL词法特征
        """
        return self.url_extractor.extract_all_url_features()

    def extract_http_features(self) -> Dict[str, Union[int, float]]:
        """
        提取HTTP响应特征（5维）

        需要发送HTTP请求，速度较慢。

        Returns:
            dict: 5维HTTP特征
        """
        return self.http_extractor.extract_all()

    def extract_ssl_features(self) -> Dict[str, int]:
        """
        提取SSL证书特征（5维）

        需要建立SSL连接，速度较慢。

        Returns:
            dict: 5维SSL特征
        """
        return self.ssl_extractor.extract_all()

    def extract_dns_features(self) -> Dict[str, Union[int, float]]:
        """
        提取DNS特征（3维）

        需要DNS查询，速度中等。

        Returns:
            dict: 3维DNS特征
        """
        return self.dns_extractor.extract_all()

    def extract_network_features(self) -> Dict[str, Union[int, float]]:
        """
        提取所有网络特征（13维）

        包括HTTP（5维）+ SSL（5维）+ DNS（3维）。
        需要网络请求，速度较慢。

        Returns:
            dict: 13维网络特征
        """
        features = {}
        features.update(self.extract_http_features())
        features.update(self.extract_ssl_features())
        features.update(self.extract_dns_features())
        return features

    def extract_all(self) -> Dict[str, Union[int, float]]:
        """
        提取全部30维特征

        包括：
        - URL词法特征（17维）
        - HTTP响应特征（5维）
        - SSL证书特征（5维）
        - DNS特征（3维）

        需要网络请求，速度较慢（通常5-30秒）。

        Returns:
            dict: 30维完整特征
        """
        features = {}

        # URL特征（17维）
        features.update(self.extract_url_only())

        # HTTP特征（5维）
        features.update(self.extract_http_features())

        # SSL特征（5维）
        features.update(self.extract_ssl_features())

        # DNS特征（3维）
        features.update(self.extract_dns_features())

        return features

    def extract_all_array(self) -> np.ndarray:
        """
        以numpy数组格式返回全部30维特征

        按照 ALL_FEATURE_NAMES 定义的顺序返回。

        Returns:
            ndarray: 30维特征向量
        """
        features = self.extract_all()
        return np.array([features[name] for name in self.ALL_FEATURE_NAMES])

    def extract_url_only_array(self) -> np.ndarray:
        """
        以numpy数组格式返回URL词法特征

        Returns:
            ndarray: 17维特征向量
        """
        features = self.extract_url_only()
        return np.array([features[name] for name in self.URL_FEATURE_NAMES])

    @classmethod
    def get_feature_names(cls) -> list:
        """
        获取全部特征名称列表

        Returns:
            list: 30个特征名称
        """
        return cls.ALL_FEATURE_NAMES.copy()

    @classmethod
    def get_url_feature_names(cls) -> list:
        """
        获取URL特征名称列表

        Returns:
            list: 17个URL特征名称
        """
        return cls.URL_FEATURE_NAMES.copy()


# ==================== 批量特征提取器 ====================

class BatchFeatureExtractor:
    """
    批量特征提取器

    支持大规模URL数据集的特征提取，具有以下特性：
    - 进度条显示
    - 断点续传
    - 错误处理和日志记录

    Attributes:
        output_path (str): 输出文件路径
        checkpoint_interval (int): 检查点保存间隔
        include_network (bool): 是否包含网络特征

    Example:
        >>> extractor = BatchFeatureExtractor('output.csv')
        >>> extractor.extract(urls, labels)
    """

    def __init__(
        self,
        output_path: str,
        checkpoint_interval: int = 100,
        include_network: bool = False
    ):
        """
        初始化批量特征提取器

        Args:
            output_path: 输出CSV文件路径
            checkpoint_interval: 每处理多少条保存一次检查点
            include_network: 是否提取网络特征（HTTP/SSL/DNS）
        """
        self.output_path = output_path
        self.checkpoint_interval = checkpoint_interval
        self.include_network = include_network

        # 确保输出目录存在
        import os
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 特征名列表
        self.feature_names = FeatureExtractor.get_feature_names() if include_network \
            else FeatureExtractor.get_url_feature_names()

        # 统计信息
        self.success_count = 0
        self.error_count = 0
        self.errors = []

    def _get_checkpoint(self) -> int:
        """
        获取已处理的记录数（用于断点续传）

        Returns:
            int: 已处理的记录数
        """
        import os
        import pandas as pd

        if not os.path.exists(self.output_path):
            return 0

        try:
            df = pd.read_csv(self.output_path)
            return len(df)
        except Exception:
            return 0

    def _extract_single(self, url: str) -> Dict[str, Union[int, float]]:
        """
        提取单个URL的特征

        Args:
            url: URL字符串

        Returns:
            dict: 特征字典
        """
        try:
            extractor = FeatureExtractor(url)
            if self.include_network:
                return extractor.extract_all()
            else:
                return extractor.extract_url_only()
        except Exception as e:
            logger.warning(f"特征提取失败 [{url}]: {str(e)}")
            # 返回默认值
            return {name: -1 for name in self.feature_names}

    def extract(
        self,
        urls: List[str],
        labels: List[int],
        resume: bool = True
    ):
        """
        批量提取特征

        Args:
            urls: URL列表
            labels: 标签列表（0=正常, 1=钓鱼）
            resume: 是否从断点续传

        Returns:
            DataFrame: 包含URL、标签和特征的数据框
        """
        import pandas as pd
        from tqdm import tqdm

        # 检查输入
        if len(urls) != len(labels):
            raise ValueError("URLs和labels数量不匹配")

        total = len(urls)
        logger.info(f"开始批量特征提取: {total} 条URL")
        logger.info(f"特征模式: {'完整30维' if self.include_network else '仅URL词法17维'}")

        # 断点续传
        start_idx = 0
        results = []

        if resume:
            start_idx = self._get_checkpoint()
            if start_idx > 0:
                logger.info(f"从断点续传: 已处理 {start_idx} 条，继续处理剩余 {total - start_idx} 条")
                # 读取已有结果
                existing_df = pd.read_csv(self.output_path)
                results = existing_df.to_dict('records')

        # 处理剩余URL
        remaining_urls = urls[start_idx:]
        remaining_labels = labels[start_idx:]

        # 使用tqdm显示进度
        with tqdm(total=len(remaining_urls), desc="特征提取", unit="url") as pbar:
            for i, (url, label) in enumerate(zip(remaining_urls, remaining_labels)):
                # 提取特征
                features = self._extract_single(url)

                # 构建记录
                record = {'url': url, 'label': label}
                record.update(features)
                results.append(record)

                # 更新统计
                if all(v != -1 for v in features.values()):
                    self.success_count += 1
                else:
                    self.error_count += 1
                    self.errors.append(url)

                # 更新进度条
                pbar.update(1)

                # 定期保存检查点
                if (i + 1) % self.checkpoint_interval == 0:
                    self._save_checkpoint(results)
                    logger.info(f"检查点保存: {start_idx + i + 1}/{total}")

        # 最终保存
        df = pd.DataFrame(results)
        df.to_csv(self.output_path, index=False)

        # 输出统计
        logger.info(f"特征提取完成!")
        logger.info(f"  成功: {self.success_count}")
        logger.info(f"  失败: {self.error_count}")
        logger.info(f"  输出: {self.output_path}")

        return df

    def _save_checkpoint(self, results: List[Dict]):
        """
        保存检查点

        Args:
            results: 当前结果列表
        """
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(self.output_path, index=False)

    def get_statistics(self) -> Dict:
        """
        获取提取统计信息

        Returns:
            dict: 统计信息
        """
        return {
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / (self.success_count + self.error_count)
                if (self.success_count + self.error_count) > 0 else 0,
            'error_urls': self.errors[:10]  # 只返回前10个错误URL
        }


# ==================== 批量提取便捷函数 ====================

def batch_extract_features(
    urls: List[str],
    labels: List[int],
    output_path: str,
    include_network: bool = False,
    checkpoint_interval: int = 100,
    resume: bool = True
):
    """
    便捷函数：批量提取URL特征

    Args:
        urls: URL列表
        labels: 标签列表
        output_path: 输出CSV路径
        include_network: 是否包含网络特征
        checkpoint_interval: 检查点间隔
        resume: 是否断点续传

    Returns:
        DataFrame: 特征数据框

    Example:
        >>> urls = ['https://google.com', 'http://phishing.com']
        >>> labels = [0, 1]
        >>> df = batch_extract_features(urls, labels, 'features.csv')
    """
    extractor = BatchFeatureExtractor(
        output_path=output_path,
        checkpoint_interval=checkpoint_interval,
        include_network=include_network
    )
    return extractor.extract(urls, labels, resume=resume)


def extract_from_csv(
    input_path: str,
    output_path: str,
    url_column: str = 'url',
    label_column: str = 'label',
    include_network: bool = False
):
    """
    从CSV文件读取URL并提取特征

    Args:
        input_path: 输入CSV路径
        output_path: 输出CSV路径
        url_column: URL列名
        label_column: 标签列名
        include_network: 是否包含网络特征

    Returns:
        DataFrame: 特征数据框
    """
    import pandas as pd

    # 读取输入文件
    df = pd.read_csv(input_path)

    if url_column not in df.columns:
        raise ValueError(f"找不到URL列: {url_column}")
    if label_column not in df.columns:
        raise ValueError(f"找不到标签列: {label_column}")

    urls = df[url_column].tolist()
    labels = df[label_column].tolist()

    return batch_extract_features(
        urls=urls,
        labels=labels,
        output_path=output_path,
        include_network=include_network
    )


# ==================== 综合便捷函数 ====================

def extract_features(url: str, include_network: bool = True) -> Dict[str, Union[int, float]]:
    """
    便捷函数：提取单个URL的特征

    Args:
        url: URL字符串
        include_network: 是否包含网络特征（HTTP/SSL/DNS）

    Returns:
        dict: 特征字典（17维或30维）
    """
    extractor = FeatureExtractor(url)
    if include_network:
        return extractor.extract_all()
    else:
        return extractor.extract_url_only()


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
