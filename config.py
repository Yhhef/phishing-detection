"""
项目配置文件
基于网络流量特征的钓鱼网站检测系统

作者: 毕业设计项目组
日期: 2025年12月
"""

import os

# ============================================
# 路径配置
# ============================================

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# 模型文件路径
MODEL_PATH = os.path.join(MODELS_DIR, 'ensemble_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# 数据库路径
DATABASE_PATH = os.path.join(BASE_DIR, 'web', 'database.db')

# 日志目录
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# ============================================
# 特征提取配置
# ============================================

FEATURE_CONFIG = {
    'request_timeout': 10,  # HTTP请求超时时间（秒）
    'ssl_timeout': 10,      # SSL连接超时时间（秒）
    'dns_timeout': 5,       # DNS解析超时时间（秒）
}

# ============================================
# URL分析配置
# ============================================

# 短链接服务列表
SHORTENING_SERVICES = [
    'bit.ly', 'goo.gl', 't.co', 'tinyurl.com',
    'ow.ly', 'is.gd', 'buff.ly', 'adf.ly', 'bit.do',
    'cutt.ly', 'rb.gy', 'shorturl.at'
]

# 可疑关键词列表
SUSPICIOUS_WORDS = [
    'login', 'signin', 'bank', 'account', 'update',
    'verify', 'secure', 'confirm', 'password', 'suspend',
    'paypal', 'ebay', 'amazon', 'apple', 'microsoft',
    'netflix', 'facebook', 'instagram', 'whatsapp'
]

# ============================================
# SSL证书配置
# ============================================

# 知名证书颁发机构
KNOWN_CAS = [
    'DigiCert', 'Comodo', 'GlobalSign', 'Symantec',
    'GeoTrust', 'Thawte', 'VeriSign', 'Entrust',
    'GoDaddy', 'Sectigo'
]

# 免费证书颁发机构
FREE_CAS = ["Let's Encrypt", 'ZeroSSL', 'Buypass', 'Cloudflare']

# ============================================
# 模型参数配置
# ============================================

MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
}

# ============================================
# Flask配置
# ============================================

FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'secret_key': 'phishing-detection-secret-key-2025'
}

# ============================================
# 性能目标
# ============================================

PERFORMANCE_TARGETS = {
    'accuracy': 0.92,      # 准确率 ≥ 92%
    'precision': 0.90,     # 精确率 ≥ 90%
    'recall': 0.88,        # 召回率 ≥ 88%
    'f1_score': 0.89,      # F1分数 ≥ 89%
    'response_time': 3     # 响应时间 ≤ 3秒
}

# ============================================
# 数据集配置
# ============================================

DATASET_CONFIG = {
    'total_samples': 10000,        # 总样本数
    'phishing_samples': 5000,      # 钓鱼样本数
    'normal_samples': 5000,        # 正常样本数
    'train_ratio': 0.8,            # 训练集比例
    'test_ratio': 0.2,             # 测试集比例
    'random_state': 42             # 随机种子
}

# ============================================
# 30维特征列表
# ============================================

FEATURE_NAMES = [
    # URL词法特征 (17维)
    'url_length',           # 1. URL总长度
    'domain_length',        # 2. 域名长度
    'path_length',          # 3. 路径长度
    'num_dots',             # 4. 点号数量
    'num_hyphens',          # 5. 连字符数量
    'num_underscores',      # 6. 下划线数量
    'num_slashes',          # 7. 斜杠数量
    'num_digits',           # 8. 数字数量
    'has_ip',               # 9. 是否包含IP
    'has_at',               # 10. 是否包含@
    'num_subdomains',       # 11. 子域名数量
    'has_https',            # 12. 是否HTTPS
    'path_depth',           # 13. 路径深度
    'has_port',             # 14. 是否含端口
    'entropy',              # 15. 信息熵
    'is_shortening',        # 16. 是否短链接
    'has_suspicious',       # 17. 是否含可疑词

    # TLS证书特征 (5维)
    'ssl_cert_valid',       # 18. 证书有效性
    'ssl_cert_days',        # 19. 剩余天数
    'ssl_issuer_type',      # 20. 颁发机构类型
    'ssl_self_signed',      # 21. 是否自签名
    'ssl_cert_age',         # 22. 证书年龄

    # HTTP响应特征 (5维)
    'http_status_code',     # 23. 状态码
    'http_response_time',   # 24. 响应时间
    'http_redirect_count',  # 25. 重定向次数
    'content_length',       # 26. 内容长度
    'server_type',          # 27. 服务器类型

    # DNS特征 (3维)
    'domain_entropy',       # 28. 域名熵值
    'dns_resolve_time',     # 29. 解析时间
    'dns_record_count'      # 30. 记录数量
]
