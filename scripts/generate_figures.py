"""
生成数据分析可视化图表
用于Day 5数据分析任务

运行: python scripts/generate_figures.py
"""

import os
import sys
import json

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import PROCESSED_DATA_DIR
from src.data_loader import DataLoader


def count_special_chars(url):
    """统计URL中的特殊字符数量"""
    return {
        'dots': url.count('.'),
        'hyphens': url.count('-'),
        'underscores': url.count('_'),
        'slashes': url.count('/'),
        'at_symbol': url.count('@'),
        'digits': sum(c.isdigit() for c in url)
    }


def main():
    print("=" * 60)
    print("开始生成数据分析图表")
    print("=" * 60)

    # 创建图表目录
    figures_dir = os.path.join(PROCESSED_DATA_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    print(f"\n图表保存目录: {figures_dir}")

    # 加载数据
    loader = DataLoader()
    df = loader.load_dataset()
    print(f"数据加载完成: {len(df)} 条")

    # 添加计算列
    df['url_length'] = df['url'].str.len()
    df['domain_length'] = df['domain'].str.len()

    # 计算特殊字符
    char_stats = df['url'].apply(lambda x: pd.Series(count_special_chars(x)))
    df = pd.concat([df, char_stats], axis=1)

    # ===== 图表1: 标签分布 =====
    print("\n生成图表 1/7: 标签分布...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = ['正常网站', '钓鱼网站']
    counts = df['label'].value_counts().sort_index()
    colors = ['#2ecc71', '#e74c3c']

    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('数量', fontsize=12)
    axes[0].set_title('数据集标签分布', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%',
                colors=colors, explode=(0.02, 0.02), shadow=True,
                textprops={'fontsize': 12})
    axes[1].set_title('标签比例', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '01_label_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 图表2: URL长度分布 =====
    print("生成图表 2/7: URL长度分布...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, color, name in [(0, '#2ecc71', '正常'), (1, '#e74c3c', '钓鱼')]:
        subset = df[df['label'] == label]['url_length']
        axes[0].hist(subset, bins=50, alpha=0.6, label=name, color=color, edgecolor='white')

    axes[0].set_xlabel('URL长度', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('URL长度分布对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)

    box_data = [df[df['label']==0]['url_length'], df[df['label']==1]['url_length']]
    bp = axes[1].boxplot(box_data, labels=['正常', '钓鱼'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_xlabel('类别', fontsize=12)
    axes[1].set_ylabel('URL长度', fontsize=12)
    axes[1].set_title('URL长度箱线图', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02_url_length.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 图表3: 域名长度分布 =====
    print("生成图表 3/7: 域名长度分布...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, color, name in [(0, '#2ecc71', '正常'), (1, '#e74c3c', '钓鱼')]:
        subset = df[df['label'] == label]['domain_length']
        axes[0].hist(subset, bins=30, alpha=0.6, label=name, color=color, edgecolor='white')

    axes[0].set_xlabel('域名长度', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('域名长度分布对比', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)

    box_data = [df[df['label']==0]['domain_length'], df[df['label']==1]['domain_length']]
    bp = axes[1].boxplot(box_data, labels=['正常', '钓鱼'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    axes[1].set_xlabel('类别', fontsize=12)
    axes[1].set_ylabel('域名长度', fontsize=12)
    axes[1].set_title('域名长度箱线图', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03_domain_length.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 图表4: 特殊字符分布 =====
    print("生成图表 4/7: 特殊字符分布...")
    features = ['dots', 'hyphens', 'slashes', 'digits']
    feature_names = ['点号(.)', '连字符(-)', '斜杠(/)', '数字']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (feature, fname) in enumerate(zip(features, feature_names)):
        ax = axes[idx // 2, idx % 2]
        for label, color, name in [(0, '#2ecc71', '正常'), (1, '#e74c3c', '钓鱼')]:
            subset = df[df['label'] == label][feature]
            ax.hist(subset, bins=20, alpha=0.6, label=name, color=color, edgecolor='white')
        ax.set_xlabel(f'{fname}数量', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{fname}分布对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04_special_chars.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 图表5: 数据来源分布 =====
    print("生成图表 5/7: 数据来源分布...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    source_counts = df['source'].value_counts()
    colors_source = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(source_counts)]

    axes[0].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%',
                colors=colors_source, explode=[0.02]*len(source_counts), shadow=True,
                textprops={'fontsize': 11})
    axes[0].set_title('数据来源分布', fontsize=14, fontweight='bold')

    source_label = df.groupby(['source', 'label']).size().unstack(fill_value=0)
    source_label.columns = ['正常', '钓鱼']
    source_label.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1].set_xlabel('数据来源', fontsize=12)
    axes[1].set_ylabel('数量', fontsize=12)
    axes[1].set_title('各来源的标签分布', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05_source_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 图表6: Top域名 =====
    print("生成图表 6/7: Top域名分析...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    normal_domains = df[df['label']==0]['domain'].value_counts().head(10)
    axes[0].barh(normal_domains.index[::-1], normal_domains.values[::-1], color='#2ecc71', edgecolor='black')
    axes[0].set_xlabel('数量', fontsize=12)
    axes[0].set_title('正常网站 Top 10 域名', fontsize=14, fontweight='bold')

    phishing_domains = df[df['label']==1]['domain'].value_counts().head(10)
    axes[1].barh(phishing_domains.index[::-1], phishing_domains.values[::-1], color='#e74c3c', edgecolor='black')
    axes[1].set_xlabel('数量', fontsize=12)
    axes[1].set_title('钓鱼网站 Top 10 域名', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '06_top_domains.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 图表7: 相关性热力图 =====
    print("生成图表 7/7: 特征相关性热力图...")
    import seaborn as sns

    numeric_features = ['url_length', 'domain_length', 'dots', 'hyphens',
                        'underscores', 'slashes', 'digits', 'label']
    corr_matrix = df[numeric_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn_r', center=0,
                square=True, linewidths=0.5, fmt='.2f',
                cbar_kws={'shrink': 0.8})
    ax.set_title('特征相关性热力图', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ===== 保存分析报告 =====
    print("\n保存分析报告...")
    analysis_report = {
        'dataset_info': {
            'total_samples': int(len(df)),
            'train_samples': 8000,
            'test_samples': 2000,
            'phishing_count': int((df['label']==1).sum()),
            'normal_count': int((df['label']==0).sum()),
            'balance_ratio': 1.0
        },
        'url_statistics': {
            'normal_avg_length': float(df[df['label']==0]['url_length'].mean()),
            'phishing_avg_length': float(df[df['label']==1]['url_length'].mean()),
            'normal_max_length': int(df[df['label']==0]['url_length'].max()),
            'phishing_max_length': int(df[df['label']==1]['url_length'].max())
        },
        'domain_statistics': {
            'normal_avg_length': float(df[df['label']==0]['domain_length'].mean()),
            'phishing_avg_length': float(df[df['label']==1]['domain_length'].mean())
        },
        'sources': {k: int(v) for k, v in df['source'].value_counts().items()},
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    report_path = os.path.join(figures_dir, 'analysis_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)

    # 打印总结
    print("\n" + "=" * 60)
    print("图表生成完成!")
    print("=" * 60)
    print(f"\n生成的图表文件:")
    for f in sorted(os.listdir(figures_dir)):
        filepath = os.path.join(figures_dir, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"  - {f} ({size_kb:.1f} KB)")

    print(f"\n分析报告: {report_path}")

    # 打印关键统计
    print("\n" + "=" * 60)
    print("数据分析总结")
    print("=" * 60)
    print(f"总样本数: {len(df)}")
    print(f"正常网站: {(df['label']==0).sum()} | 钓鱼网站: {(df['label']==1).sum()}")
    print(f"正常URL平均长度: {df[df['label']==0]['url_length'].mean():.1f}")
    print(f"钓鱼URL平均长度: {df[df['label']==1]['url_length'].mean():.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
