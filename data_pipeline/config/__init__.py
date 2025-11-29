#!/usr/bin/env python3
"""
Package initialization for config
"""

from data_pipeline.config.config_manager import ConfigManager, get_config_manager

__all__ = [
    'ConfigManager',
    'get_config_manager',
]
