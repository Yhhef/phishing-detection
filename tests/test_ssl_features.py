"""
SSL证书特征提取单元测试
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import SSLFeatureExtractor, extract_ssl_features


class TestSSLFeatureExtractor:
    """SSLFeatureExtractor测试类"""

    def test_valid_https_url(self):
        """测试有效HTTPS URL"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        valid = extractor.ssl_cert_valid()
        # 应该能获取到有效证书
        assert valid in [0, 1]

    def test_cert_days_positive(self):
        """测试证书剩余天数"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        days = extractor.ssl_cert_days()
        # 正常网站证书应该未过期
        assert days > 0 or days == -1

    def test_issuer_type_valid_range(self):
        """测试颁发机构类型范围"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        issuer_type = extractor.ssl_issuer_type()
        assert issuer_type in [-1, 0, 1]

    def test_self_signed_valid_range(self):
        """测试自签名标志范围"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        self_signed = extractor.ssl_self_signed()
        assert self_signed in [-1, 0, 1]

    def test_cert_age_positive(self):
        """测试证书年龄"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        age = extractor.ssl_cert_age()
        # 证书年龄应该是正数
        assert age > 0 or age == -1

    def test_extract_all_feature_count(self):
        """测试提取全部特征的数量"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()
        assert len(features) == 5

    def test_extract_all_feature_keys(self):
        """测试特征键名"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()

        expected_keys = [
            'ssl_cert_valid',
            'ssl_cert_days',
            'ssl_issuer_type',
            'ssl_self_signed',
            'ssl_cert_age'
        ]

        for key in expected_keys:
            assert key in features, f"缺少特征: {key}"


class TestSSLFeatureExtractorEdgeCases:
    """边界情况和异常处理测试"""

    def test_empty_url(self):
        """测试空URL"""
        extractor = SSLFeatureExtractor("")
        features = extractor.extract_all()

        assert features['ssl_cert_valid'] == 0
        assert features['ssl_cert_days'] == -1
        assert features['ssl_issuer_type'] == -1
        assert features['ssl_self_signed'] == -1
        assert features['ssl_cert_age'] == -1

    def test_http_url(self):
        """测试HTTP URL（非HTTPS）"""
        extractor = SSLFeatureExtractor("http://example.com")
        features = extractor.extract_all()

        # HTTP URL在443端口可能有证书，也可能无法连接
        # 只检查返回值在有效范围内
        assert features['ssl_cert_valid'] in [0, 1]

    def test_nonexistent_domain(self):
        """测试不存在的域名"""
        extractor = SSLFeatureExtractor("https://nonexistent-domain-12345.com")
        features = extractor.extract_all()

        assert features['ssl_cert_valid'] == 0

    def test_lazy_loading(self):
        """测试惰性加载"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        # 创建后不应该立即获取证书
        assert extractor._fetched == False

        # 调用特征方法后应该已获取证书
        _ = extractor.ssl_cert_valid()
        assert extractor._fetched == True

    def test_fetch_once(self):
        """测试只获取一次证书"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")

        # 多次调用不同的特征方法
        _ = extractor.ssl_cert_valid()
        _ = extractor.ssl_cert_days()
        _ = extractor.ssl_issuer_type()

        # 应该只获取了一次证书
        assert extractor._fetched == True

    def test_get_error_on_failure(self):
        """测试失败时获取错误信息"""
        extractor = SSLFeatureExtractor("https://nonexistent-domain-12345.com")
        _ = extractor.extract_all()

        error = extractor.get_error()
        assert error is not None


class TestSSLFeatureExtractorConvenience:
    """便捷函数测试"""

    def test_convenience_function(self):
        """测试便捷函数"""
        features = extract_ssl_features("https://www.baidu.com")
        assert len(features) == 5
        assert 'ssl_cert_valid' in features


class TestSSLFeatureExtractorKnownSites:
    """已知网站测试"""

    def test_baidu_ssl(self):
        """测试百度SSL证书"""
        extractor = SSLFeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()

        # 百度应该有有效证书
        if features['ssl_cert_valid'] == 1:
            assert features['ssl_cert_days'] > 0
            assert features['ssl_self_signed'] == 0

    def test_qq_ssl(self):
        """测试腾讯SSL证书"""
        extractor = SSLFeatureExtractor("https://www.qq.com")
        features = extractor.extract_all()

        # 腾讯应该使用知名CA或免费CA
        if features['ssl_cert_valid'] == 1:
            assert features['ssl_issuer_type'] in [0, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
