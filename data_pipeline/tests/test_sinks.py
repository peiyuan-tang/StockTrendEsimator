#!/usr/bin/env python3
"""
Unit tests for data sinks - JSON, CSV, Parquet, Database
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime


class TestJSONSink(unittest.TestCase):
    """Test JSON sink"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'path': self.temp_dir,
            'file_prefix': 'test_',
            'file_suffix': '.json',
            'batch_size': 10
        }
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_sink_initialization(self):
        """Test JSON sink initialization"""
        try:
            from data_pipeline.storage.data_sink import JSONSink
            sink = JSONSink(self.config)
            self.assertIsNotNone(sink)
            self.assertEqual(sink.file_prefix, 'test_')
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_json_sink_write(self):
        """Test writing to JSON sink"""
        try:
            from data_pipeline.storage.data_sink import JSONSink
            
            sink = JSONSink(self.config)
            
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'ticker': 'MSFT', 'price': 320.0}}
            ]
            
            result = sink.write(events)
            
            self.assertTrue(result)
            files = os.listdir(self.temp_dir)
            self.assertTrue(any(f.endswith('.json') for f in files))
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_json_sink_file_content(self):
        """Test JSON file content"""
        try:
            from data_pipeline.storage.data_sink import JSONSink
            
            sink = JSONSink(self.config)
            
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'ticker': 'MSFT', 'price': 320.0}}
            ]
            
            sink.write(events)
            
            # Read and verify content
            files = [f for f in os.listdir(self.temp_dir) if f.endswith('.json')]
            self.assertEqual(len(files), 1)
            
            with open(os.path.join(self.temp_dir, files[0]), 'r') as f:
                data = json.load(f)
                self.assertEqual(len(data), 2)
                self.assertEqual(data[0]['ticker'], 'AAPL')
        
        except ImportError:
            self.skipTest("Required packages not installed")


class TestCSVSink(unittest.TestCase):
    """Test CSV sink"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'path': self.temp_dir,
            'file_prefix': 'test_',
            'file_suffix': '.csv',
            'batch_size': 10
        }
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_csv_sink_initialization(self):
        """Test CSV sink initialization"""
        try:
            import pandas as pd
            from data_pipeline.storage.data_sink import CSVSink
            
            sink = CSVSink(self.config)
            self.assertIsNotNone(sink)
            self.assertEqual(sink.file_suffix, '.csv')
        
        except ImportError:
            self.skipTest("pandas not installed")
    
    def test_csv_sink_write(self):
        """Test writing to CSV sink"""
        try:
            import pandas as pd
            from data_pipeline.storage.data_sink import CSVSink
            
            sink = CSVSink(self.config)
            
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'ticker': 'MSFT', 'price': 320.0}}
            ]
            
            result = sink.write(events)
            
            self.assertTrue(result)
            files = os.listdir(self.temp_dir)
            self.assertTrue(any(f.endswith('.csv') for f in files))
        
        except ImportError:
            self.skipTest("pandas not installed")
    
    def test_csv_sink_content(self):
        """Test CSV file content"""
        try:
            import pandas as pd
            from data_pipeline.storage.data_sink import CSVSink
            
            sink = CSVSink(self.config)
            
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'ticker': 'MSFT', 'price': 320.0}}
            ]
            
            sink.write(events)
            
            # Read and verify
            files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
            self.assertEqual(len(files), 1)
            
            df = pd.read_csv(os.path.join(self.temp_dir, files[0]))
            self.assertEqual(len(df), 2)
            self.assertIn('ticker', df.columns)
            self.assertIn('price', df.columns)
        
        except ImportError:
            self.skipTest("pandas not installed")


class TestParquetSink(unittest.TestCase):
    """Test Parquet sink"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'path': self.temp_dir,
            'file_prefix': 'test_',
            'file_suffix': '.parquet',
            'batch_size': 10,
            'compression': 'snappy'
        }
    
    def tearDown(self):
        """Clean up"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_parquet_sink_initialization(self):
        """Test Parquet sink initialization"""
        try:
            import pyarrow
            from data_pipeline.storage.data_sink import ParquetSink
            
            sink = ParquetSink(self.config)
            self.assertIsNotNone(sink)
            self.assertEqual(sink.file_suffix, '.parquet')
        
        except ImportError:
            self.skipTest("pyarrow not installed")
    
    def test_parquet_sink_write(self):
        """Test writing to Parquet sink"""
        try:
            import pyarrow
            from data_pipeline.storage.data_sink import ParquetSink
            
            sink = ParquetSink(self.config)
            
            events = [
                {'body': {'ticker': 'AAPL', 'price': 150.0}},
                {'body': {'ticker': 'MSFT', 'price': 320.0}}
            ]
            
            result = sink.write(events)
            
            self.assertTrue(result)
            files = os.listdir(self.temp_dir)
            self.assertTrue(any(f.endswith('.parquet') for f in files))
        
        except ImportError:
            self.skipTest("pyarrow not installed")


class TestBaseSink(unittest.TestCase):
    """Test base sink class"""
    
    def test_base_sink_path_expansion(self):
        """Test path expansion with datetime patterns"""
        try:
            from data_pipeline.storage.data_sink import BaseSink
            
            config = {
                'path': '/data/test/%Y-%m-%d/',
                'file_prefix': 'test_',
                'file_suffix': '.txt',
                'batch_size': 10
            }
            
            sink = BaseSink(config)
            expanded = sink._expand_path()
            
            # Should contain year, month, day
            self.assertNotIn('%Y', expanded)
            self.assertNotIn('%m', expanded)
            self.assertNotIn('%d', expanded)
            self.assertIn('/data/test/', expanded)
        
        except ImportError:
            self.skipTest("Required packages not installed")


class TestSinkFactory(unittest.TestCase):
    """Test sink factory"""
    
    def test_create_json_sink(self):
        """Test creating JSON sink"""
        try:
            from data_pipeline.storage.data_sink import SinkFactory, JSONSink
            
            config = {'path': '/tmp', 'file_prefix': 'test_', 'file_suffix': '.json'}
            sink = SinkFactory.create_sink('json', config)
            
            self.assertIsInstance(sink, JSONSink)
        
        except ImportError:
            self.skipTest("Required packages not installed")
    
    def test_create_csv_sink(self):
        """Test creating CSV sink"""
        try:
            import pandas
            from data_pipeline.storage.data_sink import SinkFactory, CSVSink
            
            config = {'path': '/tmp', 'file_prefix': 'test_', 'file_suffix': '.csv'}
            sink = SinkFactory.create_sink('csv', config)
            
            self.assertIsInstance(sink, CSVSink)
        
        except ImportError:
            self.skipTest("pandas not installed")
    
    def test_create_parquet_sink(self):
        """Test creating Parquet sink"""
        try:
            import pyarrow
            from data_pipeline.storage.data_sink import SinkFactory, ParquetSink
            
            config = {
                'path': '/tmp',
                'file_prefix': 'test_',
                'file_suffix': '.parquet',
                'compression': 'snappy'
            }
            sink = SinkFactory.create_sink('parquet', config)
            
            self.assertIsInstance(sink, ParquetSink)
        
        except ImportError:
            self.skipTest("pyarrow not installed")
    
    def test_unknown_sink_type(self):
        """Test error on unknown sink type"""
        try:
            from data_pipeline.storage.data_sink import SinkFactory
            
            config = {'path': '/tmp'}
            with self.assertRaises(ValueError):
                SinkFactory.create_sink('unknown', config)
        
        except ImportError:
            self.skipTest("Required packages not installed")


if __name__ == '__main__':
    unittest.main()
