#!/usr/bin/env python3
"""
Unit tests for data sources - Financial, Movement, News, Macro, Policy
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import os

# Note: These tests use mocking to avoid actual API calls


class TestFinancialDataSource(unittest.TestCase):
    """Test financial data source"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'batch_size': 100,
            'timeout': 30,
            'retry_attempts': 3,
            'ticker_list': ['AAPL', 'MSFT', 'GOOGL']
        }
    
    def test_initialization(self):
        """Test source initialization"""
        try:
            from data_pipeline.models.financial_source import FinancialDataSource
            source = FinancialDataSource(self.config)
            self.assertEqual(source.batch_size, 100)
            self.assertEqual(len(source.ticker_list), 3)
        except ImportError:
            self.skipTest("yfinance not installed")
    
    @patch('yfinance.Ticker')
    def test_fetch_data_success(self, mock_ticker):
        """Test successful data fetch"""
        try:
            from data_pipeline.models.financial_source import FinancialDataSource
            
            # Mock ticker response
            mock_instance = MagicMock()
            mock_ticker.return_value = mock_instance
            mock_instance.info = {
                'marketCap': 2000000000000,
                'trailingPE': 25.5,
                'dividendYield': 0.005
            }
            
            # Mock history
            mock_hist = pd.DataFrame({
                'Close': [150.0],
                'Open': [149.0],
                'High': [151.0],
                'Low': [149.0],
                'Volume': [50000000]
            })
            mock_instance.history.return_value = mock_hist
            
            source = FinancialDataSource(self.config)
            data = source.fetch_data()
            
            self.assertIsInstance(data, list)
            if data:
                self.assertIn('ticker', data[0])
                self.assertIn('price', data[0])
                self.assertIn('data_type', data[0])
        
        except ImportError:
            self.skipTest("yfinance not installed")
    
    def test_fetch_data_empty_list(self):
        """Test fetch with empty tickers"""
        try:
            from data_pipeline.models.financial_source import FinancialDataSource
            
            config = self.config.copy()
            config['ticker_list'] = []
            source = FinancialDataSource(config)
            data = source.fetch_data()
            
            self.assertEqual(len(data), 0)
        except ImportError:
            self.skipTest("yfinance not installed")


class TestStockMovementSource(unittest.TestCase):
    """Test stock movement source"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'batch_size': 100,
            'interval_minutes': 60,
            'indices': ['SP500'],
            'include_indicators': ['SMA_20', 'RSI']
        }
    
    def test_initialization(self):
        """Test source initialization"""
        try:
            from data_pipeline.models.movement_source import StockMovementSource
            source = StockMovementSource(self.config)
            self.assertIsNotNone(source)
            self.assertEqual(source.interval_minutes, 60)
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_calculate_indicators(self):
        """Test indicator calculation"""
        try:
            from data_pipeline.models.movement_source import StockMovementSource
            
            source = StockMovementSource(self.config)
            
            # Create sample data
            dates = pd.date_range('2024-01-01', periods=100)
            prices = pd.Series(range(100, 200), index=dates)
            df = pd.DataFrame({'Close': prices})
            
            indicators = source.calculate_indicators(df)
            
            self.assertIsInstance(indicators, dict)
            if 'SMA_20' in self.config['include_indicators']:
                self.assertIn('SMA_20', indicators)
        
        except ImportError:
            self.skipTest("Required packages not installed")


class TestNewsDataSource(unittest.TestCase):
    """Test news data source"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'batch_size': 50,
            'news_sources': ['finnhub'],
            'sentiment_analysis': True,
            'api_keys': {'finnhub': 'test_key'}
        }
    
    def test_initialization(self):
        """Test source initialization"""
        try:
            from data_pipeline.models.news_source import NewsDataSource
            source = NewsDataSource(self.config)
            self.assertTrue(source.sentiment_analysis)
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        try:
            from data_pipeline.models.news_source import NewsDataSource
            source = NewsDataSource(self.config)
            
            sentiment = source.analyze_sentiment("This is great news!")
            self.assertIn('polarity', sentiment)
            self.assertIn('subjectivity', sentiment)
        
        except ImportError:
            self.skipTest("TextBlob not installed")


class TestMacroeconomicDataSource(unittest.TestCase):
    """Test macroeconomic data source"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'batch_size': 10,
            'update_frequency': 'daily',
            'indicators': ['interest_rate', 'unemployment_rate'],
            'api_keys': {'fred': 'test_key'}
        }
    
    def test_initialization(self):
        """Test source initialization"""
        try:
            from data_pipeline.models.macro_source import MacroeconomicDataSource
            source = MacroeconomicDataSource(self.config)
            self.assertIsNotNone(source)
        except ImportError:
            self.skipTest("pandas_datareader not installed")
    
    def test_get_unit(self):
        """Test unit retrieval"""
        try:
            from data_pipeline.models.macro_source import MacroeconomicDataSource
            source = MacroeconomicDataSource(self.config)
            
            unit = source._get_unit('interest_rate')
            self.assertEqual(unit, '%')
        except ImportError:
            self.skipTest("pandas_datareader not installed")


class TestPolicyDataSource(unittest.TestCase):
    """Test policy data source"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'batch_size': 20,
            'update_frequency': 'weekly',
            'data_types': ['policy_announcements'],
            'api_keys': {}
        }
    
    def test_initialization(self):
        """Test source initialization"""
        try:
            from data_pipeline.models.policy_source import PolicyDataSource
            source = PolicyDataSource(self.config)
            self.assertIsNotNone(source)
        except ImportError:
            self.skipTest("Required packages not installed")


class TestBaseDataSource(unittest.TestCase):
    """Test base data source class"""
    
    def test_base_source_properties(self):
        """Test base source properties"""
        try:
            from data_pipeline.models.financial_source import BaseDataSource
            
            config = {
                'batch_size': 100,
                'timeout': 30,
                'retry_attempts': 3
            }
            
            # BaseDataSource is abstract, so we use a concrete implementation
            from data_pipeline.models.financial_source import FinancialDataSource
            source = FinancialDataSource(config)
            
            self.assertEqual(source.batch_size, 100)
            self.assertEqual(source.timeout, 30)
            self.assertEqual(source.retry_attempts, 3)
        
        except ImportError:
            self.skipTest("yfinance not installed")


if __name__ == '__main__':
    unittest.main()
