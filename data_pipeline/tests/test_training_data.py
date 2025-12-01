#!/usr/bin/env python3
"""
Tests for Training Data Processor

Tests the unification and joining of data from all sources
"""

import unittest
import tempfile
import shutil
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from data_pipeline.core.training_data import (
    TrainingDataProcessor,
    StockDataTower,
    ContextDataTower,
)


class TestStockDataTower(unittest.TestCase):
    """Test stock data tower functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, 'raw')
        os.makedirs(os.path.join(self.raw_dir, 'financial_data'), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, 'stock_movements'), exist_ok=True)

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_load_financial_data(self):
        """Test loading financial data"""
        # Create test data
        fin_file = os.path.join(self.raw_dir, 'financial_data', 'test.json')
        test_data = [
            {
                'data_type': 'financial_data',
                'ticker': 'AAPL',
                'timestamp': datetime.utcnow().isoformat(),
                'price': 150.0,
                'volume': 1000000,
            }
        ]
        
        with open(fin_file, 'w') as f:
            json.dump(test_data, f)

        # Load data
        config = {'data_root': self.test_dir}
        tower = StockDataTower(config)
        df = tower._load_financial_data(None, None, ['AAPL'])

        # Verify
        self.assertGreater(len(df), 0)
        self.assertIn('ticker', df.columns)
        self.assertEqual(df['ticker'].iloc[0], 'AAPL')

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_stock_tower_normalization(self):
        """Test stock data tower normalization"""
        config = {'data_root': self.test_dir}
        tower = StockDataTower(config)

        # Create sample dataframe
        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'price': [150.0, 380.0],
        })

        # Normalize
        normalized = tower.normalize_schema(df)

        # Verify required columns
        self.assertIn('timestamp', normalized.columns)
        self.assertIn('volume', normalized.columns)
        self.assertIn('market_cap', normalized.columns)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_merge_data_frames(self):
        """Test merging financial and movement data"""
        config = {'data_root': self.test_dir}
        tower = StockDataTower(config)

        # Create test dataframes
        fin_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'timestamp': [datetime(2024, 1, 1, 10, 0, 0)],
            'price': [150.0],
        })

        mov_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'timestamp': [datetime(2024, 1, 1, 10, 5, 0)],
            'rsi': [72.5],
        })

        # Merge
        merged = tower._merge_data_frames(fin_df, mov_df)

        # Verify
        self.assertGreater(len(merged), 0)
        self.assertIn('price', merged.columns)
        self.assertIn('rsi', merged.columns)


class TestContextDataTower(unittest.TestCase):
    """Test context data tower functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, 'raw')
        os.makedirs(os.path.join(self.raw_dir, 'news'), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, 'macroeconomic_data'), exist_ok=True)
        os.makedirs(os.path.join(self.raw_dir, 'policy_data'), exist_ok=True)

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_load_news_data(self):
        """Test loading news data"""
        # Create test data
        news_file = os.path.join(self.raw_dir, 'news', 'test.json')
        test_data = [
            {
                'data_type': 'news',
                'ticker': 'AAPL',
                'timestamp': datetime.utcnow().isoformat(),
                'headline': 'Test headline',
                'sentiment': {'polarity': 0.8, 'subjectivity': 0.5},
            }
        ]
        
        with open(news_file, 'w') as f:
            json.dump(test_data, f)

        # Load data
        config = {'data_root': self.test_dir}
        tower = ContextDataTower(config)
        df = tower._load_news_data(None, None, ['AAPL'])

        # Verify
        self.assertGreater(len(df), 0)
        self.assertIn('headline', df.columns)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_combine_context_data(self):
        """Test combining multiple context sources"""
        config = {'data_root': self.test_dir}
        tower = ContextDataTower(config)

        # Create sample dataframes
        news_df = pd.DataFrame({
            'ticker': ['AAPL'],
            'timestamp': [datetime(2024, 1, 1, 10, 0, 0)],
            'headline': ['Test news'],
        })

        macro_df = pd.DataFrame({
            'ticker': ['MARKET'],
            'timestamp': [datetime(2024, 1, 1, 10, 0, 0)],
            'interest_rate': [5.33],
        })

        policy_df = pd.DataFrame()

        # Combine
        combined = tower._combine_context_data(news_df, macro_df, policy_df)

        # Verify
        self.assertGreater(len(combined), 0)
        self.assertIn('ticker', combined.columns)


class TestTrainingDataProcessor(unittest.TestCase):
    """Test main training data processor"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, 'raw')
        
        # Create data directories
        for data_type in ['financial_data', 'stock_movements', 'news', 'macroeconomic_data', 'policy_data']:
            os.makedirs(os.path.join(self.raw_dir, data_type), exist_ok=True)

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_processor_initialization(self):
        """Test processor initialization"""
        config = {
            'data_root': self.test_dir,
            'output_format': 'parquet',
        }
        
        processor = TrainingDataProcessor(config)

        # Verify
        self.assertEqual(processor.data_root, self.test_dir)
        self.assertEqual(processor.output_format, 'parquet')
        self.assertIsNotNone(processor.stock_tower)
        self.assertIsNotNone(processor.context_tower)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_generate_empty_training_data(self):
        """Test generating training data with empty sources"""
        config = {
            'data_root': self.test_dir,
            'output_format': 'csv',
        }
        
        processor = TrainingDataProcessor(config)
        
        # Generate without actual data
        training_data = processor.generate_training_data(save=False)

        # Verify - should return empty DataFrame
        self.assertIsInstance(training_data, pd.DataFrame)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_join_towers_empty(self):
        """Test joining empty dataframes"""
        config = {'data_root': self.test_dir}
        processor = TrainingDataProcessor(config)

        stock_df = pd.DataFrame()
        context_df = pd.DataFrame()

        result = processor._join_towers(stock_df, context_df)

        # Verify
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_join_towers_stock_only(self):
        """Test joining when only stock data exists"""
        config = {'data_root': self.test_dir}
        processor = TrainingDataProcessor(config)

        stock_df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 1)],
            'price': [150.0, 380.0],
        })
        context_df = pd.DataFrame()

        result = processor._join_towers(stock_df, context_df)

        # Verify
        self.assertEqual(len(result), 2)
        self.assertIn('price', result.columns)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_get_feature_summary(self):
        """Test feature summary generation"""
        config = {'data_root': self.test_dir}
        processor = TrainingDataProcessor(config)

        df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT'],
            'timestamp': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            'price': [150.0, 380.0],
        })

        summary = processor.get_feature_summary(df)

        # Verify
        self.assertEqual(summary['total_records'], 2)
        self.assertEqual(summary['total_features'], 3)
        self.assertEqual(len(summary['tickers']), 2)
        self.assertIn('missing_values', summary)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_save_parquet_format(self):
        """Test saving training data in Parquet format"""
        output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        config = {
            'data_root': self.test_dir,
            'output_format': 'parquet',
            'output_path': output_dir,
        }

        processor = TrainingDataProcessor(config)

        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'timestamp': [datetime(2024, 1, 1)],
            'price': [150.0],
        })

        filepath = processor._save_training_data(df)

        # Verify file exists and is readable
        self.assertTrue(os.path.exists(filepath))
        loaded_df = pd.read_parquet(filepath)
        self.assertEqual(len(loaded_df), 1)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_save_csv_format(self):
        """Test saving training data in CSV format"""
        output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        config = {
            'data_root': self.test_dir,
            'output_format': 'csv',
            'output_path': output_dir,
        }

        processor = TrainingDataProcessor(config)

        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'timestamp': [datetime(2024, 1, 1)],
            'price': [150.0],
        })

        filepath = processor._save_training_data(df)

        # Verify
        self.assertTrue(os.path.exists(filepath))
        loaded_df = pd.read_csv(filepath)
        self.assertEqual(len(loaded_df), 1)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_save_json_format(self):
        """Test saving training data in JSON format"""
        output_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(output_dir, exist_ok=True)

        config = {
            'data_root': self.test_dir,
            'output_format': 'json',
            'output_path': output_dir,
        }

        processor = TrainingDataProcessor(config)

        df = pd.DataFrame({
            'ticker': ['AAPL'],
            'timestamp': [datetime(2024, 1, 1)],
            'price': [150.0],
        })

        filepath = processor._save_training_data(df)

        # Verify
        self.assertTrue(os.path.exists(filepath))
        loaded_df = pd.read_json(filepath)
        self.assertEqual(len(loaded_df), 1)


class TestDataTowerIntegration(unittest.TestCase):
    """Integration tests for data tower interaction"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, 'raw')
        self._setup_test_data()

    def tearDown(self):
        """Clean up test files"""
        shutil.rmtree(self.test_dir)

    def _setup_test_data(self):
        """Create realistic test data files"""
        # Create directories
        for data_type in ['financial_data', 'stock_movements', 'news', 'macroeconomic_data', 'policy_data']:
            os.makedirs(os.path.join(self.raw_dir, data_type), exist_ok=True)

        now = datetime.utcnow()

        # Financial data
        fin_data = [
            {
                'data_type': 'financial_data',
                'ticker': 'AAPL',
                'timestamp': now.isoformat(),
                'price': 150.0,
                'open': 149.0,
                'high': 152.0,
                'low': 148.0,
                'volume': 50000000,
                'market_cap': 2400000000000,
                'pe_ratio': 28.5,
            }
        ]
        
        with open(os.path.join(self.raw_dir, 'financial_data', 'fin.json'), 'w') as f:
            json.dump(fin_data, f)

        # Stock movements
        mov_data = [
            {
                'data_type': 'stock_movement',
                'ticker': 'AAPL',
                'timestamp': now.isoformat(),
                'sma_20': 148.5,
                'sma_50': 145.0,
                'rsi': 72.5,
                'macd': 2.5,
            }
        ]
        
        with open(os.path.join(self.raw_dir, 'stock_movements', 'mov.json'), 'w') as f:
            json.dump(mov_data, f)

        # News data
        news_data = [
            {
                'data_type': 'news',
                'ticker': 'AAPL',
                'timestamp': now.isoformat(),
                'headline': 'Apple reports strong Q2 earnings',
                'sentiment': {'polarity': 0.85, 'subjectivity': 0.5},
            }
        ]
        
        with open(os.path.join(self.raw_dir, 'news', 'news.json'), 'w') as f:
            json.dump(news_data, f)

    @unittest.skipIf(not PANDAS_AVAILABLE, "pandas not available")
    def test_full_pipeline(self):
        """Test complete training data generation pipeline"""
        config = {
            'data_root': self.test_dir,
            'output_format': 'parquet',
        }

        processor = TrainingDataProcessor(config)
        training_data = processor.generate_training_data(save=False)

        # Verify structure
        self.assertIsInstance(training_data, pd.DataFrame)
        if not training_data.empty:
            self.assertIn('ticker', training_data.columns)
            self.assertIn('timestamp', training_data.columns)


if __name__ == '__main__':
    unittest.main()
