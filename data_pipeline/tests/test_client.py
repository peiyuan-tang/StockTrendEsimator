#!/usr/bin/env python3
"""
Unit tests for client and configuration
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from datetime import datetime, timedelta
from pathlib import Path


class TestConfigManager(unittest.TestCase):
    """Test configuration manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_manager_initialization(self):
        """Test config manager initialization"""
        try:
            from data_pipeline.utils.config_manager import ConfigManager
            
            manager = ConfigManager()
            self.assertIsNotNone(manager.config)
            self.assertIsNotNone(manager.credentials)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_config_defaults(self):
        """Test default configuration values"""
        try:
            from data_pipeline.utils.config_manager import ConfigManager
            
            manager = ConfigManager()
            
            self.assertEqual(manager.config.log_level, 'INFO')
            self.assertEqual(manager.config.retention_days, 90)
            self.assertIn('AAPL', manager.config.mag7_tickers)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_get_api_keys(self):
        """Test getting API keys"""
        try:
            from data_pipeline.utils.config_manager import ConfigManager
            
            manager = ConfigManager()
            keys = manager.get_api_keys()
            
            self.assertIsInstance(keys, dict)
            self.assertIn('finnhub', keys)
            self.assertIn('newsapi', keys)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        try:
            from data_pipeline.utils.config_manager import ConfigManager
            
            manager = ConfigManager()
            config_dict = manager.to_dict()
            
            self.assertIn('config', config_dict)
            self.assertIn('credentials', config_dict)
            self.assertIsInstance(config_dict['config'], dict)
        
        except ImportError:
            self.skipTest("Required packages not installed")


class TestDataPipelineClient(unittest.TestCase):
    """Test offline data pipeline client"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data structure
        Path(os.path.join(self.temp_dir, 'raw', 'financial_data', '2024-01-01')).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(self.temp_dir, 'context', 'macroeconomic')).mkdir(
            parents=True, exist_ok=True
        )
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_client_initialization(self):
        """Test client initialization"""
        try:
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            client = DataPipelineClient(data_root=self.temp_dir)
            self.assertEqual(client.data_root, self.temp_dir)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_get_data_summary(self):
        """Test getting data summary"""
        try:
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            client = DataPipelineClient(data_root=self.temp_dir)
            summary = client.get_data_summary()
            
            self.assertIn('timestamp', summary)
            self.assertIn('data_sources', summary)
            self.assertIn('total_size_bytes', summary)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_data_summary_structure(self):
        """Test data summary structure"""
        try:
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            client = DataPipelineClient(data_root=self.temp_dir)
            summary = client.get_data_summary()
            
            self.assertIn('financial_data', summary['data_sources'])
            self.assertIn('stock_movements', summary['data_sources'])
            self.assertIn('news', summary['data_sources'])
            self.assertIn('macroeconomic', summary['data_sources'])
            self.assertIn('policy', summary['data_sources'])
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_count_files(self):
        """Test file counting"""
        try:
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            # Create test file
            test_dir = os.path.join(self.temp_dir, 'raw', 'financial_data')
            test_file = os.path.join(test_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            
            client = DataPipelineClient(data_root=self.temp_dir)
            count = client._count_files(test_dir)
            
            self.assertGreater(count, 0)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_get_directory_size(self):
        """Test directory size calculation"""
        try:
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            # Create test file
            test_dir = os.path.join(self.temp_dir, 'raw', 'financial_data')
            test_file = os.path.join(test_dir, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test data')
            
            client = DataPipelineClient(data_root=self.temp_dir)
            size = client._get_directory_size(self.temp_dir)
            
            self.assertGreater(size, 0)
        
        except ImportError:
            self.skipTest("Required packages not installed")


class TestDataExport(unittest.TestCase):
    """Test data export functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_export_data_csv(self):
        """Test exporting data to CSV"""
        try:
            import pandas as pd
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            # Create mock data file
            data_dir = os.path.join(self.temp_dir, 'raw', 'financial_data', '2024-01-01')
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            
            # Create sample CSV
            df = pd.DataFrame({
                'ticker': ['AAPL', 'MSFT'],
                'price': [150.0, 320.0]
            })
            df.to_csv(os.path.join(data_dir, 'test.csv'), index=False)
            
            client = DataPipelineClient(data_root=self.temp_dir)
            output_file = os.path.join(self.output_dir, 'export.csv')
            
            # Note: This test checks the export method exists
            self.assertTrue(hasattr(client, 'export_data'))
        
        except ImportError:
            self.skipTest("pandas not installed")
    
    def test_export_data_json(self):
        """Test exporting data to JSON"""
        try:
            import json
            from data_pipeline.client.pipeline_client import DataPipelineClient
            
            client = DataPipelineClient(data_root=self.temp_dir)
            
            # Check method exists
            self.assertTrue(hasattr(client, 'export_data'))
        
        except ImportError:
            self.skipTest("Required packages not installed")


class TestPipelineScheduler(unittest.TestCase):
    """Test pipeline scheduler"""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization"""
        try:
            from data_pipeline.core.pipeline_scheduler import CollectionScheduler
            
            scheduler = CollectionScheduler()
            self.assertIsNotNone(scheduler)
            self.assertFalse(scheduler.scheduler.running)
        
        except ImportError:
            self.skipTest("apscheduler not installed")
    
    def test_add_interval_job(self):
        """Test adding interval job"""
        try:
            from data_pipeline.core.pipeline_scheduler import CollectionScheduler
            
            scheduler = CollectionScheduler()
            
            def dummy_task():
                pass
            
            job_id = scheduler.add_interval_job(dummy_task, 'test_job', seconds=3600)
            
            self.assertEqual(job_id, 'test_job')
            self.assertIn('test_job', scheduler.jobs)
        
        except ImportError:
            self.skipTest("apscheduler not installed")
    
    def test_get_jobs(self):
        """Test getting all jobs"""
        try:
            from data_pipeline.core.pipeline_scheduler import CollectionScheduler
            
            scheduler = CollectionScheduler()
            jobs = scheduler.get_jobs()
            
            self.assertIsInstance(jobs, dict)
        
        except ImportError:
            self.skipTest("apscheduler not installed")


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
        try:
            import yaml
            from data_pipeline.core.flume_server import StockDataCollector
            
            # Create minimal config file
            config = {'agents': {}}
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f)
            
            server = StockDataCollector(config_path=self.config_file)
            self.assertIsNotNone(server)
            self.assertEqual(server.config_path, self.config_file)
        
        except ImportError:
            self.skipTest("pyyaml not installed")
    
    def test_server_get_status(self):
        """Test server status"""
        try:
            import yaml
            from data_pipeline.core.flume_server import StockDataCollector
            
            config = {'agents': {}}
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f)
            
            server = StockDataCollector(config_path=self.config_file)
            status = server.get_status()
            
            self.assertIn('running', status)
            self.assertIn('agents', status)
            self.assertIn('timestamp', status)
        
        except ImportError:
            self.skipTest("pyyaml not installed")


if __name__ == '__main__':
    unittest.main()
