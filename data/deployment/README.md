# 钓鱼网站检测模型部署包

## 文件说明
- model.pkl: 集成分类模型（VotingClassifier）
- scaler.pkl: 特征标准化器（StandardScaler）
- config.json: 模型配置信息

## 使用方法

```python
import joblib
import numpy as np

# 加载模型和标准化器
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# 预测
features = np.array([...])  # 30维特征
features_scaled = scaler.transform(features.reshape(1, -1))
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)
```

## 特征说明
请参考config.json中的feature_names字段

## 版本信息
生成时间: 2026-01-01T17:07:17.164136
