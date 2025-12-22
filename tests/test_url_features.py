"""
URL特征提取单元测试

测试 URLFeatureExtractor 类的8个基础特征提取方法

作者: 毕业设计项目组
日期: 2025年12月
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import URLFeatureExtractor, extract_url_basic_features


class TestURLFeatureExtractor:
    """URLFeatureExtractor测试类"""

    def test_url_length(self):
        """测试URL长度计算"""
        url = "https://example.com/path"
        extractor = URLFeatureExtractor(url)
        assert extractor.url_length() == len(url.lower())

    def test_domain_length(self):
        """测试域名长度计算"""
        url = "https://example.com/path"
        extractor = URLFeatureExtractor(url)
        assert extractor.domain_length() == len("example.com")

    def test_path_length(self):
        """测试路径长度计算"""
        url = "https://example.com/path/to/page"
        extractor = URLFeatureExtractor(url)
        assert extractor.path_length() == len("/path/to/page")

    def test_num_dots(self):
        """测试点号数量"""
        url = "https://www.sub.example.com/path"
        extractor = URLFeatureExtractor(url)
        # www.sub.example.com 有3个点
        assert extractor.num_dots() == 3

    def test_num_hyphens(self):
        """测试连字符数量"""
        url = "https://my-secure-site.com/login-page"
        extractor = URLFeatureExtractor(url)
        assert extractor.num_hyphens() == 3

    def test_num_underscores(self):
        """测试下划线数量"""
        url = "https://example.com/path_to_page"
        extractor = URLFeatureExtractor(url)
        assert extractor.num_underscores() == 2

    def test_num_slashes(self):
        """测试斜杠数量"""
        url = "https://example.com/a/b/c/"
        extractor = URLFeatureExtractor(url)
        # https:// 有2个斜杠, /a/b/c/ 有4个斜杠，共6个
        assert extractor.num_slashes() == 6

    def test_num_digits(self):
        """测试数字数量"""
        url = "https://example123.com/page456"
        extractor = URLFeatureExtractor(url)
        assert extractor.num_digits() == 6

    def test_extract_basic_features(self):
        """测试基础特征提取"""
        url = "https://example.com/path"
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_basic_features()

        assert len(features) == 8
        assert 'url_length' in features
        assert 'domain_length' in features
        assert 'path_length' in features
        assert 'num_dots' in features
        assert 'num_hyphens' in features
        assert 'num_underscores' in features
        assert 'num_slashes' in features
        assert 'num_digits' in features

    def test_empty_url(self):
        """测试空URL处理"""
        extractor = URLFeatureExtractor("")
        features = extractor.extract_basic_features()
        assert features['url_length'] == 0
        assert features['domain_length'] == 0
        assert features['path_length'] == 0

    def test_none_url(self):
        """测试None URL处理"""
        extractor = URLFeatureExtractor(None)
        features = extractor.extract_basic_features()
        assert features['url_length'] == 0

    def test_url_with_port(self):
        """测试带端口的URL"""
        url = "https://example.com:8080/path"
        extractor = URLFeatureExtractor(url)
        # 域名应该不包含端口号
        assert extractor.domain_length() == len("example.com")

    def test_case_insensitive(self):
        """测试大小写不敏感"""
        url1 = "HTTPS://EXAMPLE.COM/PATH"
        url2 = "https://example.com/path"
        ext1 = URLFeatureExtractor(url1)
        ext2 = URLFeatureExtractor(url2)
        assert ext1.extract_basic_features() == ext2.extract_basic_features()

    def test_convenience_function(self):
        """测试便捷函数"""
        url = "https://example.com/path"
        features = extract_url_basic_features(url)
        assert len(features) == 8

    def test_extract_basic_features_array(self):
        """测试numpy数组格式返回"""
        url = "https://example.com/path"
        extractor = URLFeatureExtractor(url)
        arr = extractor.extract_basic_features_array()
        assert len(arr) == 8
        assert arr.dtype in [int, float, 'int64', 'float64']

    def test_phishing_url_characteristics(self):
        """测试钓鱼URL特征提取"""
        phishing_url = "http://paypal-secure-login.fake-site.com/verify/account123"
        extractor = URLFeatureExtractor(phishing_url)
        features = extractor.extract_basic_features()

        # 钓鱼URL通常有更多连字符
        assert features['num_hyphens'] >= 3
        # 钓鱼URL通常有更多点号（子域名）
        assert features['num_dots'] >= 2
        # 路径中包含数字
        assert features['num_digits'] >= 3


class TestEdgeCases:
    """边界情况测试"""

    def test_url_with_query_string(self):
        """测试带查询字符串的URL"""
        url = "https://example.com/search?q=test&page=1"
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_basic_features()
        assert features['num_digits'] == 1  # 只有1个数字

    def test_url_with_fragment(self):
        """测试带锚点的URL"""
        url = "https://example.com/page#section1"
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_basic_features()
        assert features['num_digits'] == 1

    def test_ip_address_url(self):
        """测试IP地址URL"""
        url = "http://192.168.1.100/admin/login.php"
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_basic_features()
        # IP地址包含较多数字
        assert features['num_digits'] >= 10

    def test_shortening_url(self):
        """测试短链接URL"""
        url = "http://bit.ly/abc123"
        extractor = URLFeatureExtractor(url)
        features = extractor.extract_basic_features()
        # 短链接通常较短
        assert features['url_length'] < 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
