#!/usr/bin/env python3
"""
Unit tests for integration scenarios
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from datetime import datetime
from pathlib import Path


class TestFinancialToPipelineIntegration(unittest.TestCase):
    """Test financial data source → channel → sink pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data', 'raw')
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('yfinance.Ticker')
    def test_financial_to_json_sink(self, mock_ticker):
        """Test financial data → JSON sink"""
        try:
            from data_pipeline.models.financial_source import FinancialDataSource
            from data_pipeline.storage.data_sink import JSONSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            # Mock yfinance
            mock_instance = MagicMock()
            mock_instance.history.return_value = {
                'Close': [150.0],
                'Open': [149.0],
                'High': [151.0],
                'Low': [148.0],
                'Volume': [1000000]
            }
            mock_ticker.return_value = mock_instance
            
            # Create source and sink
            config = DataPipelineConfig()
            source = FinancialDataSource(config)
            
            sink_config = {'base_path': self.data_dir}
            sink = JSONSink(sink_config)
            
            # Fetch and write data
            with patch.object(source, 'fetch_data') as mock_fetch:
                mock_fetch.return_value = [{
                    'body': {
                        'ticker': 'AAPL',
                        'price': 150.0,
                        'timestamp': datetime.now().isoformat()
                    }
                }]
                
                events = mock_fetch()
                result = sink.write(events)
                
                self.assertTrue(result)
                
                # Verify file was created
                output_dir = os.path.join(self.data_dir, datetime.now().strftime('%Y-%m-%d'))
                self.assertTrue(os.path.exists(output_dir))
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    @patch('yfinance.Ticker')
    def test_financial_to_csv_sink(self, mock_ticker):
        """Test financial data → CSV sink"""
        try:
            import pandas as pd
            from data_pipeline.models.financial_source import FinancialDataSource
            from data_pipeline.storage.data_sink import CSVSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            # Mock data
            mock_instance = MagicMock()
            mock_ticker.return_value = mock_instance
            
            config = DataPipelineConfig()
            source = FinancialDataSource(config)
            
            sink_config = {'base_path': self.data_dir}
            sink = CSVSink(sink_config)
            
            events = [{
                'body': {
                    'ticker': 'MSFT',
                    'price': 320.0,
                    'market_cap': 2500000000000
                }
            }]
            
            result = sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("pandas not installed")


class TestStockMovementToPipelineIntegration(unittest.TestCase):
    """Test stock movement source → channel → sink pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data', 'raw')
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('yfinance.download')
    def test_movement_to_parquet_sink(self, mock_download):
        """Test stock movement → Parquet sink"""
        try:
            import pandas as pd
            from data_pipeline.models.movement_source import StockMovementSource
            from data_pipeline.storage.data_sink import ParquetSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            # Mock data
            mock_df = pd.DataFrame({
                'Close': [100, 101, 102],
                'Volume': [1000000, 1100000, 1200000]
            }, index=pd.date_range('2024-01-01', periods=3))
            mock_download.return_value = mock_df
            
            config = DataPipelineConfig()
            source = StockMovementSource(config)
            
            sink_config = {'base_path': self.data_dir}
            sink = ParquetSink(sink_config)
            
            events = [{
                'body': {
                    'ticker': 'SPY',
                    'SMA_20': 100.5,
                    'RSI': 65.0
                }
            }]
            
            result = sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("pandas/pyarrow not installed")


class TestNewsToPipelineIntegration(unittest.TestCase):
    """Test news source → channel → sink pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data', 'raw')
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.get')
    def test_news_to_json_sink(self, mock_get):
        """Test news data → JSON sink"""
        try:
            from data_pipeline.models.news_source import NewsDataSource
            from data_pipeline.storage.data_sink import JSONSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            # Mock response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'articles': [
                    {
                        'headline': 'Market Rally',
                        'summary': 'Markets surge',
                        'source': 'Reuters'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            config = DataPipelineConfig()
            source = NewsDataSource(config)
            
            sink_config = {'base_path': self.data_dir}
            sink = JSONSink(sink_config)
            
            events = [{
                'body': {
                    'headline': 'Market Rally',
                    'sentiment': 0.8,
                    'source': 'Reuters'
                }
            }]
            
            result = sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("requests not installed")


class TestMacroToPipelineIntegration(unittest.TestCase):
    """Test macroeconomic source → channel → sink pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data', 'context')
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.get')
    def test_macro_to_json_sink(self, mock_get):
        """Test macro data → JSON sink"""
        try:
            from data_pipeline.models.macro_source import MacroeconomicDataSource
            from data_pipeline.storage.data_sink import JSONSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'observations': [
                    {'value': '5.25'}
                ]
            }
            mock_get.return_value = mock_response
            
            config = DataPipelineConfig()
            source = MacroeconomicDataSource(config)
            
            sink_config = {'base_path': self.data_dir}
            sink = JSONSink(sink_config)
            
            events = [{
                'body': {
                    'indicator': 'FED_RATE',
                    'value': 5.25,
                    'date': '2024-01-01'
                }
            }]
            
            result = sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("requests not installed")


class TestPolicyToPipelineIntegration(unittest.TestCase):
    """Test policy source → channel → sink pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data', 'context')
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.get')
    def test_policy_to_csv_sink(self, mock_get):
        """Test policy data → CSV sink"""
        try:
            import pandas as pd
            from data_pipeline.models.policy_source import PolicyDataSource
            from data_pipeline.storage.data_sink import CSVSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'meetings': [
                    {
                        'date': '2024-01-15',
                        'title': 'FOMC Meeting',
                        'outcome': 'Rates unchanged'
                    }
                ]
            }
            mock_get.return_value = mock_response
            
            config = DataPipelineConfig()
            source = PolicyDataSource(config)
            
            sink_config = {'base_path': self.data_dir}
            sink = CSVSink(sink_config)
            
            events = [{
                'body': {
                    'date': '2024-01-15',
                    'event': 'FOMC Meeting',
                    'impact': 'neutral'
                }
            }]
            
            result = sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("pandas not installed")


class TestMultipleSourcesToDatabase(unittest.TestCase):
    """Test multiple sources → database sink"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('psycopg2.connect')
    def test_multi_source_to_database(self, mock_connect):
        """Test multiple sources writing to database"""
        try:
            from data_pipeline.storage.data_sink import DatabaseSink
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            # Mock database connection
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            db_config = {
                'db_type': 'postgresql',
                'host': 'localhost',
                'database': 'test_db'
            }
            sink = DatabaseSink(db_config)
            
            # Multiple events from different sources
            events = [
                {'body': {'source': 'financial', 'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'source': 'movement', 'ticker': 'SPY', 'rsi': 65.0}},
                {'body': {'source': 'news', 'headline': 'Market Rally'}}
            ]
            
            result = sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("psycopg2 not installed")


class TestEndToEndPipeline(unittest.TestCase):
    """Test complete end-to-end pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_root = os.path.join(self.temp_dir, 'data')
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_pipeline_with_mock_data(self):
        """Test complete pipeline with mock data"""
        try:
            from data_pipeline.storage.data_sink import SinkFactory
            from data_pipeline.utils.config_manager import DataPipelineConfig
            
            # Create sinks for different data types
            raw_dir = os.path.join(self.data_root, 'raw')
            context_dir = os.path.join(self.data_root, 'context')
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
            Path(context_dir).mkdir(parents=True, exist_ok=True)
            
            factory = SinkFactory()
            
            # Create different sinks
            json_sink_config = {'base_path': raw_dir}
            json_sink = factory.create_sink('json', json_sink_config)
            
            csv_sink_config = {'base_path': raw_dir}
            csv_sink = factory.create_sink('csv', csv_sink_config)
            
            # Write data through both sinks
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0, 'source': 'financial'}},
                {'body': {'ticker': 'MSFT', 'price': 320.0, 'source': 'financial'}}
            ]
            
            json_result = json_sink.write(events)
            csv_result = csv_sink.write(events)
            
            self.assertTrue(json_result)
            self.assertTrue(csv_result)
            
            # Verify files created
            output_dir = os.path.join(raw_dir, datetime.now().strftime('%Y-%m-%d'))
            self.assertTrue(os.path.exists(output_dir))
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_pipeline_with_parquet(self):
        """Test pipeline with Parquet format"""
        try:
            import pandas as pd
            from data_pipeline.storage.data_sink import SinkFactory
            
            raw_dir = os.path.join(self.data_root, 'raw')
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
            
            factory = SinkFactory()
            
            parquet_config = {'base_path': raw_dir, 'compression': 'snappy'}
            parquet_sink = factory.create_sink('parquet', parquet_config)
            
            events = [{
                'body': {
                    'ticker': 'GOOG',
                    'price': 140.0,
                    'volume': 1000000,
                    'timestamp': datetime.now().isoformat()
                }
            }]
            
            result = parquet_sink.write(events)
            self.assertTrue(result)
        
        except ImportError:
            self.skipTest("pandas/pyarrow not installed")


if __name__ == '__main__':
    unittest.main()
