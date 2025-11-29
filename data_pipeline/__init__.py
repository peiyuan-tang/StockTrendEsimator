#!/usr/bin/env python3
"""
Package initialization for data pipeline
"""

from data_pipeline.core.flume_server import StockDataCollector
from data_pipeline.client.pipeline_client import DataPipelineClient, get_data_client
from data_pipeline.utils.config_manager import ConfigManager, get_config_manager

__all__ = [
    'StockDataCollector',
    'DataPipelineClient',
    'get_data_client',
    'ConfigManager',
    'get_config_manager',
]

__version__ = '1.0.0'
__author__ = 'Stock Trend Estimator Team'
