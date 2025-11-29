#!/usr/bin/env python3
"""
Package initialization for client
"""

from data_pipeline.client.pipeline_client import DataPipelineClient, get_data_client

__all__ = [
    'DataPipelineClient',
    'get_data_client',
]
