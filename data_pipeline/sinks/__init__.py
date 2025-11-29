#!/usr/bin/env python3
"""
Package initialization for sinks
"""

from data_pipeline.sinks.data_sink import (
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
