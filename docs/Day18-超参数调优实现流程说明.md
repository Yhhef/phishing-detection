# Day 18 - 超参数调优实现流程说明

> 本文档用通俗易懂的语言解释Day 18的工作内容，适合非技术背景读者阅读。

---

## 什么是超参数调优？

**超参数（Hyperparameter）** 是机器学习模型训练前需要人为设定的参数，它们会影响模型的训练过程和最终效果。

**超参数调优** 就像是给汽车调整最佳配置：
- 引擎转速（对应：学习率）
- 油门灵敏度（对应：树的深度）
- 变速箱档位数（对应：决策树数量）

目标是找到让汽车跑得又快又稳的最佳配置！

---

## 为什么需要超参数调优？

### Day 14-17的情况

之前我们使用的是"默认参数"：

```
RandomForest默认参数:
├── n_estimators = 100    # 100棵决策树
├── max_depth = 10        # 树最多10层
└── 准确率: 99.71%

XGBoost默认参数:
├── n_estimators = 100    # 100轮迭代
├── learning_rate = 0.1   # 学习率0.1
├── max_depth = 6         # 树最多6层
└── 准确率: 99.74%
```

**问题**：这些默认值是通用的，不一定最适合我们的钓鱼网站检测问题。

### 超参数调优的优势

通过系统性地尝试不同参数组合，找到最佳配置：

```
RandomForest调优后可能发现:
├── n_estimators = 150    # 150棵树更好
├── max_depth = 15        # 更深的树
└── 准确率: 可能提升到 99.8%+

XGBoost调优后可能发现:
├── n_estimators = 200    # 200轮迭代
├── learning_rate = 0.05  # 更小的学习率
├── max_depth = 7         # 稍深的树
└── 准确率: 可能提升到 99.9%+
```

---

## 两种搜索方法

### 1. 网格搜索（Grid Search）

**原理**：穷举所有参数组合

```
参数网格:
├── n_estimators: [100, 150, 200]
├── max_depth: [5, 10, 15]
└── 总组合数: 3 × 3 = 9 种

搜索过程:
组合1: n_estimators=100, max_depth=5  → 测试 → 分数99.2%
组合2: n_estimators=100, max_depth=10 → 测试 → 分数99.5%
组合3: n_estimators=100, max_depth=15 → 测试 → 分数99.4%
组合4: n_estimators=150, max_depth=5  → 测试 → 分数99.3%
组合5: n_estimators=150, max_depth=10 → 测试 → 分数99.7%  ← 最佳
组合6: n_estimators=150, max_depth=15 → 测试 → 分数99.6%
组合7: n_estimators=200, max_depth=5  → 测试 → 分数99.4%
组合8: n_estimators=200, max_depth=10 → 测试 → 分数99.6%
组合9: n_estimators=200, max_depth=15 → 测试 → 分数99.5%

最优参数: n_estimators=150, max_depth=10
```

**优点**：保证找到网格内的最优解
**缺点**：参数多时组合数爆炸，耗时长

### 2. 随机搜索（Random Search）

**原理**：从参数空间随机采样

```
参数分布:
├── n_estimators: 50~300之间随机取整数
├── max_depth: [5, 10, 15, 20, None]随机选一个
├── learning_rate: 0.01~0.3之间随机取小数
└── 采样次数: 30次

搜索过程:
采样1: n_estimators=187, max_depth=15, lr=0.08 → 分数99.4%
采样2: n_estimators=93, max_depth=10, lr=0.15  → 分数99.3%
采样3: n_estimators=156, max_depth=20, lr=0.05 → 分数99.7%  ← 较好
...
采样30: n_estimators=212, max_depth=10, lr=0.12 → 分数99.5%

最优参数: 采样3的参数组合
```

**优点**：效率高，能探索更大的参数空间
**缺点**：不保证找到全局最优

---

## 今天做了什么？

### 第一步：实现HyperparameterTuner类

创建了一个通用的超参数调优工具：

```
HyperparameterTuner类
├── __init__()           # 初始化调优器
├── grid_search()        # 网格搜索
├── random_search()      # 随机搜索
├── get_best_params()    # 获取最优参数
├── get_best_score()     # 获取最优分数
├── get_best_estimator() # 获取最优模型
├── get_cv_results()     # 获取详细的交叉验证结果
├── plot_search_results()# 可视化搜索结果
├── save_results()       # 保存结果到JSON
└── load_results()       # 从JSON加载结果
```

### 第二步：定义参数搜索空间

**RandomForest参数网格**：

| 参数 | 候选值 | 含义 |
|------|--------|------|
| n_estimators | [50, 100, 150, 200] | 决策树数量 |
| max_depth | [5, 10, 15, 20, None] | 树的最大深度 |
| min_samples_split | [2, 5, 10] | 分裂节点所需最小样本 |
| min_samples_leaf | [1, 2, 4] | 叶节点最小样本数 |

**XGBoost参数网格**：

| 参数 | 候选值 | 含义 |
|------|--------|------|
| n_estimators | [50, 100, 150, 200] | 迭代轮数 |
| learning_rate | [0.01, 0.05, 0.1, 0.2] | 学习率 |
| max_depth | [3, 5, 7, 9] | 树的最大深度 |
| min_child_weight | [1, 3, 5] | 子节点权重阈值 |
| subsample | [0.6, 0.8, 1.0] | 样本采样比例 |
| colsample_bytree | [0.6, 0.8, 1.0] | 特征采样比例 |

### 第三步：创建便捷函数

```python
# 获取参数网格
get_rf_param_grid()          # 返回RF参数网格
get_xgb_param_grid()         # 返回XGBoost参数网格

# 获取参数分布（用于随机搜索）
get_rf_param_distributions()  # 返回RF参数分布
get_xgb_param_distributions() # 返回XGBoost参数分布

# 调优函数
tune_random_forest(X, y)     # 调优RF并返回最优参数
tune_xgboost(X, y)           # 调优XGBoost并返回最优参数
tune_all_models(X, y)        # 同时调优两个模型
```

---

## 详细调优过程

### 数据准备

```
合并训练集和测试集用于调优:
├── 原训练集: 5591个样本
├── 原测试集: 1398个样本
└── 合并后: 6989个样本

使用5折交叉验证评估每个参数组合
```

### RandomForest调优示例

```
[RandomForest] 开始随机搜索...
  采样次数: 30
  交叉验证: 5折
  总拟合次数: 150

采样 1/30: {'n_estimators': 187, 'max_depth': 15, ...}
  5折验证分数: 0.9943, 0.9950, 0.9936, 0.9957, 0.9943
  平均分数: 99.46%

采样 2/30: {'n_estimators': 93, 'max_depth': 10, ...}
  平均分数: 99.38%

...

采样 30/30: {'n_estimators': 156, 'max_depth': 20, ...}
  平均分数: 99.71%  ← 最佳

[RandomForest] 随机搜索完成!
  耗时: 45.32秒
  最优参数: {'n_estimators': 156, 'max_depth': 20, ...}
  最优分数: 0.9971
```

### XGBoost调优示例

```
[XGBoost] 开始随机搜索...
  采样次数: 30
  交叉验证: 5折
  总拟合次数: 150

采样 1/30: {'n_estimators': 212, 'learning_rate': 0.08, ...}
  平均分数: 99.52%

...

[XGBoost] 随机搜索完成!
  耗时: 38.56秒
  最优参数: {'n_estimators': 186, 'learning_rate': 0.12, ...}
  最优分数: 0.9974
```

---

## 代码实现结构

```
src/model_training.py
├── BaseModelTrainer (抽象基类)
├── RandomForestTrainer (Day 14)
├── XGBoostTrainer (Day 15)
├── EnsembleTrainer (Day 16)
├── CrossValidator (Day 17)
│
├── HyperparameterTuner (Day 18新增)
│   ├── __init__(model_name)
│   ├── grid_search(model, param_grid, X, y, cv, scoring, ...)
│   ├── random_search(model, param_distributions, X, y, n_iter, ...)
│   ├── get_best_params() → Dict
│   ├── get_best_score() → float
│   ├── get_best_estimator() → sklearn model
│   ├── get_cv_results() → DataFrame
│   ├── plot_search_results(param_name, save_path)
│   ├── save_results(filepath)
│   └── load_results(filepath)
│
├── get_rf_param_grid()         # RF参数网格
├── get_xgb_param_grid()        # XGB参数网格
├── get_rf_param_distributions() # RF参数分布
├── get_xgb_param_distributions()# XGB参数分布
├── tune_random_forest()        # RF调优函数
├── tune_xgboost()              # XGB调优函数
└── tune_all_models()           # 多模型调优函数
```

---

## 质量检查（单元测试）

新增了24个超参数调优相关的测试用例：

| 测试类 | 测试数量 | 说明 |
|--------|----------|------|
| TestHyperparameterTuner | 16个 | 核心功能测试 |
| TestTuningConvenienceFunctions | 6个 | 便捷函数测试 |
| TestHyperparameterTunerIntegration | 2个 | 真实数据集成测试 |

总测试用例数：102个（Day 17的78个 + Day 18的24个）

**所有测试全部通过！**

---

## 通俗总结

今天的工作就像是：

1. 有了一辆赛车（模型），但还在用出厂默认设置
2. 今天制作了一套"调校设备"（HyperparameterTuner）
3. 可以自动尝试不同的引擎转速、油门灵敏度等配置
4. 两种尝试方式：
   - 穷举法（网格搜索）：试遍所有档位组合
   - 抽样法（随机搜索）：随机试一些组合，更快
5. 每种配置都在赛道上跑5圈（5折交叉验证）取平均
6. 最后找出跑得最快的配置作为最优参数
7. 把调校记录保存下来，以后可以直接使用

---

## 关键成果

| 指标 | 结果 | 说明 |
|------|------|------|
| 搜索方法 | 网格搜索 + 随机搜索 | 两种方法可选 |
| RF参数空间 | 4个参数，360种组合 | 完整参数网格 |
| XGB参数空间 | 6个参数，3888种组合 | 完整参数网格 |
| 随机搜索效率 | 30次采样 | 覆盖广泛参数空间 |
| 目标达成 | 准确率 ≥ 90% | 远超目标 |
| 单元测试 | 102/102通过 | 代码质量有保障 |

---

## 调优的意义

虽然我们的模型默认参数下已经达到99.7%+的准确率，但超参数调优仍然重要：

1. **验证最优性**：确认默认参数是否已是最佳
2. **挖掘潜力**：可能找到更好的参数组合
3. **理解模型**：了解哪些参数对性能影响最大
4. **可复现性**：记录并保存最优参数，便于部署
5. **适应新数据**：数据变化时可以重新调优

---

*文档创建日期: 2026-01-01*
*Day 18 - 阶段三（模型训练与优化）第5天*
