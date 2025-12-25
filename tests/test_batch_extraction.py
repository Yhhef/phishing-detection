"""
批量特征提取单元测试
"""

import pytest
import sys
import os
import tempfile
import pandas as pd
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import (
    BatchFeatureExtractor,
    batch_extract_features,
    extract_from_csv
)


class TestBatchFeatureExtractor:
    """BatchFeatureExtractor测试类"""

    def test_init(self):
        """测试初始化"""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(f.name)
            assert extractor.checkpoint_interval == 100
            assert extractor.include_network == False

    def test_extract_url_only(self):
        """测试仅URL特征提取"""
        urls = [
            "https://www.google.com",
            "http://example.com/path",
            "https://test.example.org/page"
        ]
        labels = [0, 0, 0]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            df = extractor.extract(urls, labels, resume=False)

            assert len(df) == 3
            assert 'url' in df.columns
            assert 'label' in df.columns
            assert 'url_length' in df.columns
            assert len(df.columns) == 19  # url + label + 17 features

    def test_extract_with_invalid_url(self):
        """测试包含无效URL的提取"""
        urls = [
            "https://www.google.com",
            "",  # 空URL
            "not_a_valid_url"
        ]
        labels = [0, 1, 1]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            df = extractor.extract(urls, labels, resume=False)

            assert len(df) == 3
            # 无效URL应该返回默认值但不崩溃

    def test_checkpoint_resume(self):
        """测试断点续传"""
        urls = ["https://example.com"] * 10
        labels = [0] * 10

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            # 第一次提取
            extractor1 = BatchFeatureExtractor(
                output_path=f.name,
                checkpoint_interval=5,
                include_network=False
            )
            df1 = extractor1.extract(urls[:5], labels[:5], resume=False)
            assert len(df1) == 5

            # 续传提取
            extractor2 = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            checkpoint = extractor2._get_checkpoint()
            assert checkpoint == 5

    def test_feature_count_url_only(self):
        """测试仅URL特征时的特征数"""
        urls = ["https://example.com"]
        labels = [0]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            df = extractor.extract(urls, labels, resume=False)

            # url + label + 17 URL features = 19
            assert len(df.columns) == 19

    def test_mismatched_lengths(self):
        """测试URL和标签数量不匹配"""
        urls = ["https://example.com", "https://test.com"]
        labels = [0]  # 数量不匹配

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            with pytest.raises(ValueError):
                extractor.extract(urls, labels, resume=False)


class TestBatchConvenienceFunctions:
    """便捷函数测试"""

    def test_batch_extract_features(self):
        """测试批量提取便捷函数"""
        urls = ["https://google.com", "https://example.com"]
        labels = [0, 0]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df = batch_extract_features(
                urls=urls,
                labels=labels,
                output_path=f.name,
                include_network=False
            )

            assert len(df) == 2
            assert 'url_length' in df.columns

    def test_extract_from_csv(self):
        """测试从CSV文件提取"""
        # 创建临时输入文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f_in:
            f_in.write("url,label\n")
            f_in.write("https://google.com,0\n")
            f_in.write("https://example.com,0\n")
            input_path = f_in.name

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f_out:
            df = extract_from_csv(
                input_path=input_path,
                output_path=f_out.name,
                include_network=False
            )

            assert len(df) == 2

    def test_extract_from_csv_missing_column(self):
        """测试CSV缺少必要列"""
        # 创建临时输入文件（缺少label列）
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ) as f_in:
            f_in.write("url\n")
            f_in.write("https://google.com\n")
            input_path = f_in.name

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f_out:
            with pytest.raises(ValueError):
                extract_from_csv(
                    input_path=input_path,
                    output_path=f_out.name,
                    include_network=False
                )


class TestBatchFeatureExtractorStatistics:
    """统计信息测试"""

    def test_get_statistics(self):
        """测试获取统计信息"""
        urls = ["https://google.com", ""]
        labels = [0, 1]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            extractor.extract(urls, labels, resume=False)

            stats = extractor.get_statistics()
            assert 'success_count' in stats
            assert 'error_count' in stats
            assert 'success_rate' in stats

    def test_statistics_values(self):
        """测试统计信息值"""
        urls = ["https://google.com", "https://example.com", "https://test.org"]
        labels = [0, 0, 0]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            extractor = BatchFeatureExtractor(
                output_path=f.name,
                include_network=False
            )
            extractor.extract(urls, labels, resume=False)

            stats = extractor.get_statistics()
            total = stats['success_count'] + stats['error_count']
            assert total == 3


class TestOutputFileFormat:
    """输出文件格式测试"""

    def test_csv_header(self):
        """测试CSV文件表头"""
        urls = ["https://example.com"]
        labels = [0]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            batch_extract_features(
                urls=urls,
                labels=labels,
                output_path=f.name,
                include_network=False
            )

            # 读取生成的CSV
            df = pd.read_csv(f.name)

            # 检查必要的列
            assert 'url' in df.columns
            assert 'label' in df.columns
            assert 'url_length' in df.columns
            assert 'domain_length' in df.columns
            assert 'entropy' in df.columns

    def test_csv_values(self):
        """测试CSV文件值"""
        urls = ["https://www.example.com/path"]
        labels = [0]

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            batch_extract_features(
                urls=urls,
                labels=labels,
                output_path=f.name,
                include_network=False
            )

            df = pd.read_csv(f.name)

            # 检查值是否合理
            assert df.iloc[0]['url_length'] > 0
            assert df.iloc[0]['label'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
