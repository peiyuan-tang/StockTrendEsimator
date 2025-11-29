#!/usr/bin/env python3
"""
Storage package - Data sink implementations
Handles writing data to various formats (JSON, CSV, Parquet, Database, Protobuf)
"""

from data_pipeline.storage.data_sink import (
    BaseSink,
    JSONSink,
    ParquetSink,
    CSVSink,
    DatabaseSink,
    ProtobufSink,
    SinkFactory,
)

__all__ = [
    'BaseSink',
    'JSONSink',
    'ParquetSink',
    'CSVSink',
    'DatabaseSink',
    'ProtobufSink',
    'SinkFactory',
]
