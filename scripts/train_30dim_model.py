#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30维特征模型训练脚本
训练 RandomForest + XGBoost 集成模型
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, classification_report, confusion_matrix)
import xgboost as xgb
import joblib

# 项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')


def main():
    print('=' * 60)
    print('30维特征 集成模型训练')
    print('=' * 60)

    # 加载数据
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_30dim.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_30dim.csv'))

    feature_cols = [c for c in train_df.columns if c != 'label']
    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    print(f'\n训练集: {len(X_train)} 条, 特征维度: {len(feature_cols)}')
    print(f'测试集: {len(X_test)} 条')

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 定义模型
    print('\n训练 RandomForest...')
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print('训练 XGBoost...')
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )

    # 集成模型 - 软投票
    print('构建集成模型...')
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)

    # 单独训练用于对比
    rf.fit(X_train_scaled, y_train)
    xgb_clf.fit(X_train_scaled, y_train)

    # 评估
    print('\n' + '=' * 60)
    print('模型性能对比')
    print('=' * 60)

    models = {
        'RandomForest': (rf, rf.predict(X_test_scaled)),
        'XGBoost': (xgb_clf, xgb_clf.predict(X_test_scaled)),
        'Ensemble': (ensemble, ensemble.predict(X_test_scaled))
    }

    for name, (model, y_pred) in models.items():
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f'{name:15s}: Acc={acc*100:.2f}% | Prec={prec*100:.2f}% | '
              f'Rec={rec*100:.2f}% | F1={f1*100:.2f}%')

    # 详细评估集成模型
    y_pred_ens = models['Ensemble'][1]
    print('\n' + '=' * 60)
    print('集成模型详细评估')
    print('=' * 60)
    print('\n分类报告:')
    print(classification_report(y_test, y_pred_ens, target_names=['正常', '钓鱼']))

    print('混淆矩阵:')
    cm = confusion_matrix(y_test, y_pred_ens)
    print(f'  真负例(TN): {cm[0,0]:4d}  假正例(FP): {cm[0,1]:4d}')
    print(f'  假负例(FN): {cm[1,0]:4d}  真正例(TP): {cm[1,1]:4d}')

    # 特征重要性
    print('\n特征重要性 Top 10 (RandomForest):')
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance.head(10).iterrows():
        print(f'  {row["feature"]:25s}: {row["importance"]*100:.2f}%')

    # 保存模型
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(ensemble, os.path.join(MODEL_DIR, 'ensemble_30dim_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler_30dim.pkl'))
    joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_30dim_model.pkl'))

    # 保存特征列名
    with open(os.path.join(MODEL_DIR, 'feature_columns_30dim.txt'), 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f'\n模型已保存到 {MODEL_DIR}:')
    print('  - ensemble_30dim_model.pkl')
    print('  - rf_30dim_model.pkl')
    print('  - scaler_30dim.pkl')
    print('  - feature_columns_30dim.txt')


if __name__ == '__main__':
    main()
