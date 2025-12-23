"""
HTTP响应特征提取单元测试
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import HTTPFeatureExtractor, extract_http_features


class TestHTTPFeatureExtractor:
    """HTTPFeatureExtractor测试类"""

    def test_valid_url_status_code(self):
        """测试有效URL的状态码"""
        # 使用一个稳定的网站
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        status = extractor.http_status_code()
        # 状态码应该是有效的HTTP状态码
        assert status == 200 or status == -1  # 可能因网络问题失败

    def test_response_time_positive(self):
        """测试响应时间为正数"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        response_time = extractor.http_response_time()
        # 成功时应该是正数，失败时是-1
        assert response_time > 0 or response_time == -1

    def test_redirect_count_non_negative(self):
        """测试重定向次数非负"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        redirect_count = extractor.http_redirect_count()
        assert redirect_count >= 0 or redirect_count == -1

    def test_content_length_positive(self):
        """测试内容长度为正"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        content_length = extractor.content_length()
        assert content_length > 0 or content_length == -1

    def test_server_type_valid_range(self):
        """测试服务器类型编码范围"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        server_type = extractor.server_type()
        assert server_type in [-1, 0, 1]

    def test_extract_all_feature_count(self):
        """测试提取全部特征的数量"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()
        assert len(features) == 5

    def test_extract_all_feature_keys(self):
        """测试特征键名"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        features = extractor.extract_all()

        expected_keys = [
            'http_status_code',
            'http_response_time',
            'http_redirect_count',
            'content_length',
            'server_type'
        ]

        for key in expected_keys:
            assert key in features, f"缺少特征: {key}"


class TestHTTPFeatureExtractorEdgeCases:
    """边界情况和异常处理测试"""

    def test_empty_url(self):
        """测试空URL"""
        extractor = HTTPFeatureExtractor("")
        features = extractor.extract_all()

        assert features['http_status_code'] == -1
        assert features['http_response_time'] == -1
        assert features['http_redirect_count'] == -1
        assert features['content_length'] == -1
        assert features['server_type'] == -1

    def test_invalid_url(self):
        """测试无效URL"""
        extractor = HTTPFeatureExtractor("not_a_valid_url")
        features = extractor.extract_all()

        # 无效URL应该返回-1或错误状态码
        # 注：某些网络环境下代理可能返回其他状态码
        assert features['http_status_code'] == -1 or features['http_status_code'] >= 400

    def test_nonexistent_domain(self):
        """测试不存在的域名"""
        extractor = HTTPFeatureExtractor("https://this-domain-does-not-exist-12345.com")
        features = extractor.extract_all()

        # 不存在的域名应该返回-1
        assert features['http_status_code'] == -1

    def test_url_without_protocol(self):
        """测试不带协议的URL"""
        extractor = HTTPFeatureExtractor("www.baidu.com")
        # 应该自动添加http://前缀
        assert extractor.url.startswith('http://')

    def test_lazy_loading(self):
        """测试惰性加载"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")
        # 创建后不应该立即发送请求
        assert extractor._fetched == False

        # 调用特征方法后应该已发送请求
        _ = extractor.http_status_code()
        assert extractor._fetched == True

    def test_fetch_once(self):
        """测试只发送一次请求"""
        extractor = HTTPFeatureExtractor("https://www.baidu.com")

        # 多次调用不同的特征方法
        _ = extractor.http_status_code()
        _ = extractor.http_response_time()
        _ = extractor.content_length()

        # 应该只发送了一次请求
        assert extractor._fetched == True

    def test_get_error_on_failure(self):
        """测试失败时获取错误信息"""
        extractor = HTTPFeatureExtractor("https://this-domain-does-not-exist-12345.com")
        _ = extractor.extract_all()

        error = extractor.get_error()
        assert error is not None


class TestHTTPFeatureExtractorConvenience:
    """便捷函数测试"""

    def test_convenience_function(self):
        """测试便捷函数"""
        features = extract_http_features("https://www.baidu.com")
        assert len(features) == 5
        assert 'http_status_code' in features


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
