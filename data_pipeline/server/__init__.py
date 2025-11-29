#!/usr/bin/env python3
"""
Package initialization for server
"""

from data_pipeline.server.flume_server import StockDataCollector

__all__ = [
    'StockDataCollector',
]
