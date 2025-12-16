"""
src模块初始化文件
基于网络流量特征的钓鱼网站检测系统

此模块包含:
- feature_extraction: 特征提取模块
- model_training: 模型训练模块
- prediction: 预测模块
- utils: 工具函数模块
"""

from .utils import (
    ensure_dir,
    save_json,
    load_json,
    get_timestamp,
    validate_url,
    normalize_url
)

__version__ = '1.0.0'
__author__ = '毕业设计项目组'

__all__ = [
    'ensure_dir',
    'save_json',
    'load_json',
    'get_timestamp',
    'validate_url',
    'normalize_url'
]
