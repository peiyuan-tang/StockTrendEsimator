#!/usr/bin/env python3
"""
Package initialization for sources
"""

from data_pipeline.sources.financial_source import FinancialDataSource, BaseDataSource
from data_pipeline.sources.movement_source import StockMovementSource
from data_pipeline.sources.news_source import NewsDataSource
from data_pipeline.sources.macro_source import MacroeconomicDataSource
from data_pipeline.sources.policy_source import PolicyDataSource

__all__ = [
    'BaseDataSource',
    'FinancialDataSource',
    'StockMovementSource',
    'NewsDataSource',
    'MacroeconomicDataSource',
    'PolicyDataSource',
]
