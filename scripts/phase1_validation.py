"""
阶段一验收脚本
检查Day 1-5的所有交付物

运行: python scripts/phase1_validation.py
"""

import os
import sys

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_mark(condition):
    """返回检查标记"""
    return "[OK]" if condition else "[FAIL]"


def main():
    print("=" * 60)
    print("阶段一验收检查 (Day 1-5)")
    print("=" * 60)

    results = {}

    # ==========================================
    # 1. 导入检查
    # ==========================================
    print("\n【1. 核心库导入检查】")
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import flask
        import requests
        import tldextract
        import matplotlib
        import seaborn
        results['imports'] = True
        print(f"  {check_mark(True)} 所有核心库导入成功")
        print(f"      - pandas: {pd.__version__}")
        print(f"      - numpy: {np.__version__}")
        print(f"      - sklearn: {sklearn.__version__}")
        print(f"      - xgboost: {xgboost.__version__}")
        print(f"      - flask: {flask.__version__}")
    except ImportError as e:
        results['imports'] = False
        print(f"  {check_mark(False)} 导入失败: {e}")

    # ==========================================
    # 2. 配置检查
    # ==========================================
    print("\n【2. 配置文件检查】")
    try:
        from config import (BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                           MODELS_DIR, FEATURE_NAMES, MODEL_CONFIG, PERFORMANCE_TARGETS)
        results['config'] = True
        print(f"  {check_mark(True)} 配置加载成功")
        print(f"      - BASE_DIR: {BASE_DIR}")
        print(f"      - 特征数量: {len(FEATURE_NAMES)}")
    except Exception as e:
        results['config'] = False
        print(f"  {check_mark(False)} 配置加载失败: {e}")

    # ==========================================
    # 3. 模块检查
    # ==========================================
    print("\n【3. 项目模块检查】")
    modules_ok = True
    try:
        from src.utils import logger, ensure_dir, validate_url
        print(f"  {check_mark(True)} src/utils.py")
    except ImportError as e:
        modules_ok = False
        print(f"  {check_mark(False)} src/utils.py: {e}")

    try:
        from src.data_loader import DataLoader, load_data, load_all
        print(f"  {check_mark(True)} src/data_loader.py")
    except ImportError as e:
        modules_ok = False
        print(f"  {check_mark(False)} src/data_loader.py: {e}")

    try:
        from src.data_collector import PhishTankCollector, TrancoCollector
        print(f"  {check_mark(True)} src/data_collector.py")
    except ImportError as e:
        modules_ok = False
        print(f"  {check_mark(False)} src/data_collector.py: {e}")

    try:
        from src.dataset_builder import DatasetBuilder
        print(f"  {check_mark(True)} src/dataset_builder.py")
    except ImportError as e:
        modules_ok = False
        print(f"  {check_mark(False)} src/dataset_builder.py: {e}")

    results['modules'] = modules_ok

    # ==========================================
    # 4. 目录结构检查
    # ==========================================
    print("\n【4. 目录结构检查】")
    from config import BASE_DIR
    required_dirs = [
        'src',
        'data',
        'data/raw',
        'data/processed',
        'data/processed/figures',
        'data/models',
        'notebooks',
        'web',
        'tests',
        'scripts',
        'logs'
    ]

    dirs_ok = True
    for dir_name in required_dirs:
        dir_path = os.path.join(BASE_DIR, dir_name)
        exists = os.path.isdir(dir_path)
        if not exists:
            dirs_ok = False
        print(f"  {check_mark(exists)} {dir_name}/")

    results['directories'] = dirs_ok

    # ==========================================
    # 5. 数据文件检查
    # ==========================================
    print("\n【5. 数据文件检查】")
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    # 原始数据
    raw_files = os.listdir(RAW_DATA_DIR) if os.path.exists(RAW_DATA_DIR) else []
    phishtank_exists = any('phishtank' in f for f in raw_files)
    tranco_exists = any('tranco' in f for f in raw_files)
    print(f"  {check_mark(phishtank_exists)} data/raw/phishtank_*.json")
    print(f"  {check_mark(tranco_exists)} data/raw/tranco_*.json")

    # 处理后数据
    dataset_path = os.path.join(PROCESSED_DATA_DIR, 'dataset.csv')
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train.csv')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test.csv')

    dataset_exists = os.path.exists(dataset_path)
    train_exists = os.path.exists(train_path)
    test_exists = os.path.exists(test_path)

    print(f"  {check_mark(dataset_exists)} data/processed/dataset.csv")
    print(f"  {check_mark(train_exists)} data/processed/train.csv")
    print(f"  {check_mark(test_exists)} data/processed/test.csv")

    results['data_files'] = all([phishtank_exists, tranco_exists,
                                  dataset_exists, train_exists, test_exists])

    # ==========================================
    # 6. 数据质量检查
    # ==========================================
    print("\n【6. 数据质量检查】")
    try:
        loader = DataLoader()
        stats = loader.get_statistics()

        total_ok = stats['total_samples'] >= 9000
        train_ok = stats['train_samples'] >= 7000
        test_ok = stats['test_samples'] >= 1800
        balance_ok = 0.9 <= (stats['phishing_total'] / stats['normal_total']) <= 1.1

        print(f"  {check_mark(total_ok)} 总样本数: {stats['total_samples']} (要求≥9000)")
        print(f"  {check_mark(train_ok)} 训练集: {stats['train_samples']}")
        print(f"  {check_mark(test_ok)} 测试集: {stats['test_samples']}")
        print(f"  {check_mark(balance_ok)} 类别平衡: 钓鱼{stats['phishing_total']}/正常{stats['normal_total']}")

        # 检查重复和空值
        dataset = loader.load_dataset()
        duplicates = dataset['url'].duplicated().sum()
        null_count = dataset.isnull().sum().sum()

        dup_ok = duplicates == 0
        null_ok = null_count == 0
        print(f"  {check_mark(dup_ok)} 重复URL: {duplicates}")
        print(f"  {check_mark(null_ok)} 空值: {null_count}")

        results['data_quality'] = all([total_ok, train_ok, test_ok, balance_ok, dup_ok, null_ok])
    except Exception as e:
        results['data_quality'] = False
        print(f"  {check_mark(False)} 数据加载失败: {e}")

    # ==========================================
    # 7. 分析报告检查
    # ==========================================
    print("\n【7. 分析报告检查】")
    figures_dir = os.path.join(PROCESSED_DATA_DIR, 'figures')

    notebook_path = os.path.join(BASE_DIR, 'notebooks', '01_data_exploration.ipynb')
    notebook_exists = os.path.exists(notebook_path)
    print(f"  {check_mark(notebook_exists)} notebooks/01_data_exploration.ipynb")

    expected_figures = [
        '01_label_distribution.png',
        '02_url_length.png',
        '03_domain_length.png',
        '04_special_chars.png',
        '05_source_distribution.png',
        '06_top_domains.png',
        '07_correlation_heatmap.png',
        'analysis_report.json'
    ]

    figures_ok = True
    for fig in expected_figures:
        fig_path = os.path.join(figures_dir, fig)
        exists = os.path.exists(fig_path)
        if not exists:
            figures_ok = False
        print(f"  {check_mark(exists)} figures/{fig}")

    results['analysis'] = notebook_exists and figures_ok

    # ==========================================
    # 8. 验收总结
    # ==========================================
    print("\n" + "=" * 60)
    print("阶段一验收总结")
    print("=" * 60)

    all_passed = all(results.values())
    passed_count = sum(results.values())
    total_count = len(results)

    check_items = [
        ('核心库导入', results.get('imports', False)),
        ('配置文件', results.get('config', False)),
        ('项目模块', results.get('modules', False)),
        ('目录结构', results.get('directories', False)),
        ('数据文件', results.get('data_files', False)),
        ('数据质量', results.get('data_quality', False)),
        ('分析报告', results.get('analysis', False))
    ]

    print("\n检查项目汇总:")
    for name, passed in check_items:
        print(f"  {check_mark(passed)} {name}")

    print(f"\n通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.0f}%)")

    if all_passed:
        print("\n" + "=" * 60)
        print("Congratulations! Phase 1 all checks passed!")
        print("   Ready for Phase 2: Feature Engineering (Day 6-13)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING: Some checks failed, please fix them")
        print("=" * 60)

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
