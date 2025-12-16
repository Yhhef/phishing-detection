"""
工具函数模块
提供项目通用的工具函数

作者: 毕业设计项目组
日期: 2025年12月
"""

import os
import sys
import json
import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# ============================================
# 日志配置
# ============================================

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回logger实例

    Parameters:
    -----------
    name : str
        logger名称
    log_file : str, optional
        日志文件路径
    level : int
        日志级别

    Returns:
    --------
    logging.Logger
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件handler（如果指定）
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 创建默认logger
logger = setup_logger('phishing_detection')


# ============================================
# 目录和文件操作
# ============================================

def ensure_dir(directory: str) -> None:
    """
    确保目录存在，不存在则创建

    Parameters:
    -----------
    directory : str
        目录路径
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")


def save_json(data: Any, filepath: str) -> None:
    """
    保存数据为JSON文件

    Parameters:
    -----------
    data : Any
        要保存的数据
    filepath : str
        文件路径
    """
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON to: {filepath}")


def load_json(filepath: str) -> Any:
    """
    加载JSON文件

    Parameters:
    -----------
    filepath : str
        文件路径

    Returns:
    --------
    Any
        加载的数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from: {filepath}")
    return data


# ============================================
# 时间和标识
# ============================================

def get_timestamp(format_str: str = '%Y%m%d_%H%M%S') -> str:
    """
    获取当前时间戳字符串

    Parameters:
    -----------
    format_str : str
        时间格式字符串

    Returns:
    --------
    str
        格式化的时间字符串
    """
    return datetime.now().strftime(format_str)


def get_date_str() -> str:
    """获取当前日期字符串 (YYYY-MM-DD)"""
    return datetime.now().strftime('%Y-%m-%d')


def generate_id(text: str) -> str:
    """
    根据文本生成唯一ID

    Parameters:
    -----------
    text : str
        输入文本

    Returns:
    --------
    str
        MD5哈希值的前8位
    """
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ============================================
# URL处理
# ============================================

def validate_url(url: str) -> bool:
    """
    验证URL格式是否有效

    Parameters:
    -----------
    url : str
        待验证的URL

    Returns:
    --------
    bool
        URL是否有效
    """
    if not url or not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    # 检查协议
    if not url.startswith(('http://', 'https://')):
        return False

    # 尝试解析
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    标准化URL

    Parameters:
    -----------
    url : str
        原始URL

    Returns:
    --------
    str
        标准化后的URL
    """
    if not url:
        return ''

    url = url.strip()

    # 添加协议头
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    # 移除末尾斜杠
    url = url.rstrip('/')

    return url


def extract_domain(url: str) -> str:
    """
    从URL中提取域名

    Parameters:
    -----------
    url : str
        URL字符串

    Returns:
    --------
    str
        域名
    """
    try:
        url = normalize_url(url)
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ''


# ============================================
# 数据验证
# ============================================

def is_valid_ip(ip: str) -> bool:
    """
    验证是否为有效的IPv4地址

    Parameters:
    -----------
    ip : str
        待验证的IP地址

    Returns:
    --------
    bool
        是否为有效IP
    """
    try:
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            num = int(part)
            if num < 0 or num > 255:
                return False
        return True
    except (ValueError, AttributeError):
        return False


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为整数"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为浮点数"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ============================================
# 打印和展示
# ============================================

def print_progress(current: int, total: int, prefix: str = 'Progress',
                   bar_length: int = 50) -> None:
    """
    打印进度条

    Parameters:
    -----------
    current : int
        当前进度
    total : int
        总数
    prefix : str
        前缀文字
    bar_length : int
        进度条长度
    """
    percent = current / total
    filled = int(bar_length * percent)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f'\r{prefix}: [{bar}] {percent:.1%} ({current}/{total})', end='')
    if current == total:
        print()


def print_dict(d: Dict, indent: int = 2) -> None:
    """格式化打印字典"""
    print(json.dumps(d, ensure_ascii=False, indent=indent))


# ============================================
# 主函数测试
# ============================================

if __name__ == '__main__':
    # 测试工具函数
    print("=" * 50)
    print("工具函数测试")
    print("=" * 50)

    # 测试URL验证
    test_urls = [
        'https://www.google.com',
        'http://192.168.1.1',
        'invalid-url',
        '',
        'ftp://example.com'
    ]

    print("\n1. URL验证测试:")
    for url in test_urls:
        result = validate_url(url)
        print(f"  {url}: {result}")

    # 测试URL标准化
    print("\n2. URL标准化测试:")
    print(f"  'example.com' -> '{normalize_url('example.com')}'")

    # 测试时间戳
    print(f"\n3. 当前时间戳: {get_timestamp()}")

    # 测试域名提取
    print(f"\n4. 域名提取: {extract_domain('https://www.google.com/search')}")

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)
