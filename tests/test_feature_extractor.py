"""
特征提取主类测试（整合测试）
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import (
    FeatureExtractor,
    DNSFeatureExtractor,
    extract_features,
    extract_dns_features
)


class TestDNSFeatureExtractor:
    """DNSFeatureExtractor测试类"""

    def test_domain_entropy(self):
        """测试域名熵计算"""
        extractor = DNSFeatureExtractor("https://www.google.com")
        entropy = extractor.domain_entropy()
        assert entropy > 0
        assert entropy < 5  # 合理范围

    def test_domain_entropy_empty(self):
        """测试空域名熵计算"""
        extractor = DNSFeatureExtractor("")
        entropy = extractor.domain_entropy()
        assert entropy == 0.0

    def test_dns_resolve_time(self):
        """测试DNS解析时间"""
        extractor = DNSFeatureExtractor("https://www.baidu.com")
        resolve_time = extractor.dns_resolve_time()
        assert resolve_time > 0 or resolve_time == -1

    def test_dns_record_count(self):
        """测试IP记录数量"""
        extractor = DNSFeatureExtractor("https://www.baidu.com")
        count = extractor.dns_record_count()
        assert count >= 1 or count == -1

    def test_extract_all(self):
        """测试提取全部DNS特征"""
        extractor = DNSFeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()
        assert len(features) == 3
        assert 'domain_entropy' in features
        assert 'dns_resolve_time' in features
        assert 'dns_record_count' in features

    def test_nonexistent_domain(self):
        """测试不存在的域名"""
        extractor = DNSFeatureExtractor("https://nonexistent-domain-12345.com")
        features = extractor.extract_all()
        assert features['dns_resolve_time'] == -1
        assert features['dns_record_count'] == -1

    def test_lazy_loading(self):
        """测试惰性加载"""
        extractor = DNSFeatureExtractor("https://www.baidu.com")
        # 创建后不应该立即发送DNS查询
        assert extractor._fetched == False

        # 调用特征方法后应该已发送DNS查询
        _ = extractor.dns_record_count()
        assert extractor._fetched == True

    def test_get_ip_addresses(self):
        """测试获取IP地址列表"""
        extractor = DNSFeatureExtractor("https://www.baidu.com")
        ips = extractor.get_ip_addresses()
        # 应该是列表类型
        assert isinstance(ips, list)

    def test_get_error_on_failure(self):
        """测试失败时获取错误信息"""
        extractor = DNSFeatureExtractor("https://nonexistent-domain-12345.com")
        _ = extractor.extract_all()
        error = extractor.get_error()
        assert error is not None


class TestDNSFeatureExtractorConvenience:
    """DNS便捷函数测试"""

    def test_convenience_function(self):
        """测试便捷函数"""
        features = extract_dns_features("https://www.baidu.com")
        assert len(features) == 3
        assert 'domain_entropy' in features


class TestFeatureExtractor:
    """FeatureExtractor主类测试"""

    def test_extract_url_only(self):
        """测试仅提取URL特征"""
        extractor = FeatureExtractor("https://www.example.com/path")
        features = extractor.extract_url_only()
        assert len(features) == 17

    def test_extract_url_only_keys(self):
        """测试URL特征键名"""
        extractor = FeatureExtractor("https://www.example.com/path")
        features = extractor.extract_url_only()

        expected_keys = [
            'url_length', 'domain_length', 'path_length',
            'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits',
            'has_ip', 'has_at', 'num_subdomains', 'has_https', 'path_depth',
            'has_port', 'entropy', 'is_shortening', 'has_suspicious'
        ]

        for key in expected_keys:
            assert key in features, f"缺少特征: {key}"

    def test_extract_all(self):
        """测试提取全部30维特征"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()
        assert len(features) == 30

    def test_extract_all_keys(self):
        """测试全部特征键名"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()

        # 检查各类特征键名
        url_keys = ['url_length', 'domain_length', 'path_length', 'num_dots',
                    'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits',
                    'has_ip', 'has_at', 'num_subdomains', 'has_https', 'path_depth',
                    'has_port', 'entropy', 'is_shortening', 'has_suspicious']
        http_keys = ['http_status_code', 'http_response_time', 'http_redirect_count',
                     'content_length', 'server_type']
        ssl_keys = ['ssl_cert_valid', 'ssl_cert_days', 'ssl_issuer_type',
                    'ssl_self_signed', 'ssl_cert_age']
        dns_keys = ['domain_entropy', 'dns_resolve_time', 'dns_record_count']

        for key in url_keys + http_keys + ssl_keys + dns_keys:
            assert key in features, f"缺少特征: {key}"

    def test_extract_all_array(self):
        """测试数组格式输出"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all_array()
        assert len(features) == 30

    def test_extract_url_only_array(self):
        """测试URL特征数组格式输出"""
        extractor = FeatureExtractor("https://www.example.com")
        features = extractor.extract_url_only_array()
        assert len(features) == 17

    def test_feature_names(self):
        """测试特征名称列表"""
        names = FeatureExtractor.get_feature_names()
        assert len(names) == 30

    def test_url_feature_names(self):
        """测试URL特征名称列表"""
        names = FeatureExtractor.get_url_feature_names()
        assert len(names) == 17

    def test_feature_names_order(self):
        """测试特征名称顺序"""
        names = FeatureExtractor.get_feature_names()
        # 前17个应该是URL特征
        assert names[:17] == FeatureExtractor.URL_FEATURE_NAMES
        # 接下来5个是HTTP特征
        assert names[17:22] == FeatureExtractor.HTTP_FEATURE_NAMES
        # 接下来5个是SSL特征
        assert names[22:27] == FeatureExtractor.SSL_FEATURE_NAMES
        # 最后3个是DNS特征
        assert names[27:30] == FeatureExtractor.DNS_FEATURE_NAMES


class TestFeatureExtractorEdgeCases:
    """边界情况测试"""

    def test_empty_url(self):
        """测试空URL"""
        extractor = FeatureExtractor("")
        features = extractor.extract_url_only()
        assert len(features) == 17
        assert features['url_length'] == 0

    def test_extract_http_features_method(self):
        """测试单独提取HTTP特征"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_http_features()
        assert len(features) == 5

    def test_extract_ssl_features_method(self):
        """测试单独提取SSL特征"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_ssl_features()
        assert len(features) == 5

    def test_extract_dns_features_method(self):
        """测试单独提取DNS特征"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_dns_features()
        assert len(features) == 3

    def test_extract_network_features(self):
        """测试提取所有网络特征"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_network_features()
        assert len(features) == 13  # HTTP(5) + SSL(5) + DNS(3)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_extract_features_with_network(self):
        """测试包含网络特征"""
        features = extract_features("https://www.baidu.com", include_network=True)
        assert len(features) == 30

    def test_extract_features_without_network(self):
        """测试不包含网络特征"""
        features = extract_features("https://www.example.com", include_network=False)
        assert len(features) == 17


class TestKnownSites:
    """已知网站测试"""

    def test_baidu_features(self):
        """测试百度特征提取"""
        extractor = FeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()

        # 百度应该有有效的HTTP响应
        if features['http_status_code'] != -1:
            assert features['http_status_code'] == 200

    def test_qq_features(self):
        """测试腾讯特征提取"""
        extractor = FeatureExtractor("https://www.qq.com")
        features = extractor.extract_all()

        # 腾讯应该有有效的SSL证书
        if features['ssl_cert_valid'] == 1:
            assert features['ssl_cert_days'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
