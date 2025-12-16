# 基于网络流量特征的钓鱼网站检测系统

## 项目简介

本项目是本科毕业设计作品，旨在开发一个基于机器学习的钓鱼网站检测系统。系统通过提取URL的多维特征（包括URL词法特征、TLS证书特征、HTTP响应特征和DNS特征），利用Random Forest和XGBoost的集成模型进行钓鱼网站识别。

## 技术栈

- **编程语言**: Python 3.9+
- **机器学习**: Scikit-learn, XGBoost
- **Web框架**: Flask
- **前端**: Bootstrap 5
- **数据库**: SQLite
- **数据来源**: PhishTank, Tranco List

## 项目结构

```
phishing-detection/
├── data/
│   ├── raw/              # 原始数据（PhishTank、Tranco）
│   ├── processed/        # 处理后的特征数据
│   └── models/           # 训练好的模型文件
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py   # 30维特征提取
│   ├── model_training.py       # 模型训练
│   ├── prediction.py           # 预测模块
│   └── utils.py                # 工具函数
├── web/
│   ├── app.py            # Flask主程序
│   ├── templates/        # HTML模板
│   └── static/           # 静态资源
├── notebooks/            # Jupyter notebooks
├── tests/                # 单元测试
├── logs/                 # 日志文件
├── config.py             # 配置文件
├── requirements.txt      # 依赖库
└── README.md
```

## 特征说明

系统提取30维特征：

### URL词法特征（17维）
1. URL长度
2. 域名长度
3. 路径长度
4. 点号数量
5. 连字符数量
6. 下划线数量
7. 斜杠数量
8. 数字数量
9. 是否包含IP
10. 是否包含@
11. 子域名数量
12. 是否HTTPS
13. 路径深度
14. 是否含端口
15. 信息熵
16. 是否短链接
17. 是否含可疑词

### TLS证书特征（5维）
18. 证书有效性
19. 剩余天数
20. 颁发机构类型
21. 是否自签名
22. 证书年龄

### HTTP响应特征（5维）
23. 状态码
24. 响应时间
25. 重定向次数
26. 内容长度
27. 服务器类型

### DNS特征（3维）
28. 域名熵值
29. 解析时间
30. 记录数量

## 性能目标

| 指标 | 目标值 |
|-----|--------|
| 准确率 | ≥92% |
| 精确率 | ≥90% |
| 召回率 | ≥88% |
| F1分数 | ≥89% |
| 响应时间 | ≤3秒 |

## 安装与运行

### 1. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行Web应用

```bash
cd web
python app.py
```

访问 http://localhost:5000 即可使用检测系统。

## 开发进度

- [x] Day 1: 环境搭建与项目初始化
- [ ] Day 2-5: 数据采集
- [ ] Day 6-13: 特征工程
- [ ] Day 14-20: 模型训练
- [ ] Day 21-28: Web系统开发
- [ ] Day 29-40: 论文撰写

## 作者

毕业设计项目组 - 2025年

## 许可证

本项目仅供学习和研究使用。
