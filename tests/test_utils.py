"""
工具函数单元测试
"""

import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    validate_url,
    normalize_url,
    extract_domain,
    get_timestamp,
    is_valid_ip,
    safe_int,
    safe_float
)


class TestURLFunctions:
    """URL相关函数测试"""

    def test_validate_url_valid(self):
        """测试有效URL"""
        assert validate_url('https://www.google.com') == True
        assert validate_url('http://example.com') == True
        assert validate_url('https://192.168.1.1') == True

    def test_validate_url_invalid(self):
        """测试无效URL"""
        assert validate_url('') == False
        assert validate_url('invalid') == False
        assert validate_url('ftp://example.com') == False
        assert validate_url(None) == False

    def test_normalize_url(self):
        """测试URL标准化"""
        assert normalize_url('example.com') == 'http://example.com'
        assert normalize_url('https://example.com/') == 'https://example.com'
        assert normalize_url('') == ''

    def test_extract_domain(self):
        """测试域名提取"""
        assert extract_domain('https://www.google.com/search') == 'www.google.com'
        assert extract_domain('http://example.com:8080/path') == 'example.com:8080'


class TestIPValidation:
    """IP验证测试"""

    def test_valid_ip(self):
        """测试有效IP"""
        assert is_valid_ip('192.168.1.1') == True
        assert is_valid_ip('0.0.0.0') == True
        assert is_valid_ip('255.255.255.255') == True

    def test_invalid_ip(self):
        """测试无效IP"""
        assert is_valid_ip('256.1.1.1') == False
        assert is_valid_ip('192.168.1') == False
        assert is_valid_ip('invalid') == False
        assert is_valid_ip('') == False


class TestSafeConversion:
    """安全转换函数测试"""

    def test_safe_int(self):
        """测试安全整数转换"""
        assert safe_int('123') == 123
        assert safe_int('invalid') == 0
        assert safe_int(None, -1) == -1

    def test_safe_float(self):
        """测试安全浮点数转换"""
        assert safe_float('3.14') == 3.14
        assert safe_float('invalid') == 0.0
        assert safe_float(None, -1.0) == -1.0


class TestTimestamp:
    """时间戳函数测试"""

    def test_get_timestamp(self):
        """测试获取时间戳"""
        ts = get_timestamp()
        assert len(ts) == 15  # YYYYMMDD_HHMMSS
        assert '_' in ts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
