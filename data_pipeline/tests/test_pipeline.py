#!/usr/bin/env python3
"""
Unit tests for data pipeline components
"""

import unittest
import tempfile
import json
import os
from datetime import datetime
from pathlib import Path

# Mock imports for testing without actual Flume
class MockEvent:
    def __init__(self, headers, body):
        self.headers = headers
        self.body = body


class TestFinancialDataSource(unittest.TestCase):
    """Test financial data source"""
    
    def test_initialization(self):
        """Test source initialization"""
        from data_pipeline.models.financial_source import FinancialDataSource
        
        config = {
            'batch_size': 100,
            'timeout': 30,
            'retry_attempts': 3,
            'ticker_list': ['AAPL', 'MSFT']
        }
        
        try:
            source = FinancialDataSource(config)
            self.assertIsNotNone(source)
            self.assertEqual(source.batch_size, 100)
        except ImportError:
            self.skipTest("yfinance not installed")


class TestDataSinks(unittest.TestCase):
    """Test data sink implementations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_sink(self):
        """Test JSON sink"""
        from data_pipeline.storage.data_sink import JSONSink
        
        config = {
            'path': self.temp_dir,
            'file_prefix': 'test_',
            'file_suffix': '.json',
            'batch_size': 10
        }
        
        sink = JSONSink(config)
        events = [
            {'body': {'ticker': 'AAPL', 'price': 150.0}},
            {'body': {'ticker': 'MSFT', 'price': 320.0}}
        ]
        
        result = sink.write(events)
        self.assertTrue(result)
        
        # Check file was created
        files = os.listdir(self.temp_dir)
        self.assertTrue(any(f.endswith('.json') for f in files))
    
    def test_csv_sink(self):
        """Test CSV sink"""
        try:
            import pandas as pd
            from data_pipeline.storage.data_sink import CSVSink
            
            config = {
                'path': self.temp_dir,
                'file_prefix': 'test_',
                'file_suffix': '.csv',
                'batch_size': 10
            }
            
            sink = CSVSink(config)
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'ticker': 'MSFT', 'price': 320.0}}
            ]
            
            result = sink.write(events)
            self.assertTrue(result)
            
            # Check file was created
            files = os.listdir(self.temp_dir)
            self.assertTrue(any(f.endswith('.csv') for f in files))
        
        except ImportError:
            self.skipTest("pandas not installed")


class TestConfigManager(unittest.TestCase):
    """Test configuration management"""
    
    def test_initialization(self):
        """Test config manager initialization"""
        from data_pipeline.utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        self.assertIsNotNone(config_manager.config)
        self.assertIsNotNone(config_manager.credentials)
    
    def test_api_keys(self):
        """Test API key management"""
        from data_pipeline.utils.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        keys = config_manager.get_api_keys()
        
        self.assertIsInstance(keys, dict)
        self.assertIn('finnhub', keys)
        self.assertIn('newsapi', keys)


class TestPipelineScheduler(unittest.TestCase):
    """Test pipeline scheduler"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        from data_pipeline.core.pipeline_scheduler import CollectionScheduler
        
        scheduler = CollectionScheduler()
        self.assertIsNotNone(scheduler)
        self.assertFalse(scheduler.scheduler.running)
    
    def test_add_interval_job(self):
        """Test adding interval job"""
        from data_pipeline.core.pipeline_scheduler import CollectionScheduler
        
        scheduler = CollectionScheduler()
        
        def dummy_task():
            pass
        
        job_id = scheduler.add_interval_job(
            dummy_task,
            'test_job',
            seconds=3600
        )
        
        self.assertEqual(job_id, 'test_job')
        self.assertIn('test_job', scheduler.jobs)


class TestDataPipelineClient(unittest.TestCase):
    """Test offline data pipeline client"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_client_initialization(self):
        """Test client initialization"""
        from data_pipeline.client.pipeline_client import DataPipelineClient
        
        client = DataPipelineClient(data_root=self.temp_dir)
        self.assertIsNotNone(client)
        self.assertEqual(client.data_root, self.temp_dir)
    
    def test_data_summary(self):
        """Test data summary"""
        from data_pipeline.client.pipeline_client import DataPipelineClient
        
        client = DataPipelineClient(data_root=self.temp_dir)
        summary = client.get_data_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('data_sources', summary)
        self.assertIn('total_size_bytes', summary)


class TestFlumeSever(unittest.TestCase):
    """Test Flume server"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'config.yaml')
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_server_initialization(self):
        """Test server initialization"""
        # Create minimal config file
        config = {
            'agents': {}
        }
        
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
        
        from data_pipeline.core.flume_server import StockDataCollector
        
        try:
            server = StockDataCollector(config_path=self.config_file)
            self.assertIsNotNone(server)
            self.assertEqual(server.config_path, self.config_file)
        except ImportError:
            self.skipTest("flume not installed")


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
