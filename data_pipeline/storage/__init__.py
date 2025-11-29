#!/usr/bin/env python3
"""
Storage package - Data sink implementations
Handles writing data to various formats (JSON, CSV, Parquet, Database)
"""

from data_pipeline.storage.data_sink import (
    BaseSink,
    JSONSink,
    ParquetSink,
    CSVSink,
    DatabaseSink,
    SinkFactory,
)

__all__ = [
    'BaseSink',
    'JSONSink',
    'ParquetSink',
    'CSVSink',
    'DatabaseSink',
    'SinkFactory',
]
