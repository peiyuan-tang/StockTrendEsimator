#!/usr/bin/env python3
"""
Protocol Buffer Utilities
Helpers for serialization, deserialization, and format conversion
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ProtobufUtils:
    """Utilities for working with Protocol Buffers"""
    
    @staticmethod
    def dict_to_protobuf_dict(record: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Convert a dictionary record to a protobuf-compatible dictionary
        This prepares data for protobuf serialization
        """
        data_type = record.get('data_type', data_type)
        pb_dict = {
            'data_type': record.get('data_type', ''),
            'timestamp': record.get('timestamp', datetime.utcnow().isoformat()),
        }
        
        if data_type == 'financial_data':
            pb_dict.update({
                'ticker': record.get('ticker', ''),
                'price': float(record.get('price', 0)),
                'open': float(record.get('open', 0)),
                'high': float(record.get('high', 0)),
                'low': float(record.get('low', 0)),
                'volume': int(record.get('volume', 0)),
                'market_cap': int(record.get('market_cap', 0)) if record.get('market_cap') else 0,
                'pe_ratio': float(record.get('pe_ratio', 0)) if record.get('pe_ratio') is not None else 0,
                'dividend_yield': float(record.get('dividend_yield', 0)) if record.get('dividend_yield') is not None else 0,
                'week_52_high': float(record.get('52_week_high', 0)) if record.get('52_week_high') is not None else 0,
                'week_52_low': float(record.get('52_week_low', 0)) if record.get('52_week_low') is not None else 0,
            })
            
        elif data_type == 'stock_movement':
            pb_dict.update({
                'ticker': record.get('ticker', ''),
                'price': float(record.get('price', 0)),
                'price_change': float(record.get('price_change', 0)) if record.get('price_change') is not None else 0,
                'price_change_percent': float(record.get('price_change_percent', 0)) if record.get('price_change_percent') is not None else 0,
                'high_52week': float(record.get('high_52week', 0)) if record.get('high_52week') is not None else 0,
                'low_52week': float(record.get('low_52week', 0)) if record.get('low_52week') is not None else 0,
                'sma_20': float(record.get('SMA_20', 0)) if record.get('SMA_20') is not None else 0,
                'sma_50': float(record.get('SMA_50', 0)) if record.get('SMA_50') is not None else 0,
                'rsi': float(record.get('RSI', 0)) if record.get('RSI') is not None else 0,
                'macd': float(record.get('MACD', 0)) if record.get('MACD') is not None else 0,
                'macd_signal': float(record.get('MACD_Signal', 0)) if record.get('MACD_Signal') is not None else 0,
                'volume': int(record.get('volume', 0)),
            })
            
        elif data_type == 'news':
            pb_dict.update({
                'ticker': record.get('ticker', ''),
                'headline': record.get('headline', ''),
                'summary': record.get('summary', ''),
                'source': record.get('source', ''),
                'url': record.get('url', ''),
                'sentiment_polarity': float(record.get('sentiment_polarity', 0)) if record.get('sentiment_polarity') is not None else 0,
                'sentiment_subjectivity': float(record.get('sentiment_subjectivity', 0)) if record.get('sentiment_subjectivity') is not None else 0,
                'published_date': record.get('published_date', ''),
            })
            
        elif data_type == 'macroeconomic_data':
            pb_dict.update({
                'indicator': record.get('indicator', ''),
                'symbol': record.get('symbol', ''),
                'value': float(record.get('value', 0)),
                'unit': record.get('unit', ''),
                'date': record.get('date', ''),
                'source': record.get('source', ''),
            })
            
        elif data_type == 'policy_data':
            pb_dict.update({
                'event_type': record.get('event_type', ''),
                'title': record.get('title', ''),
                'description': record.get('description', ''),
                'date': record.get('date', ''),
                'impact_level': record.get('impact_level', ''),
                'source': record.get('source', ''),
                'metadata': record.get('metadata', {}),
            })
        
        return pb_dict
    
    @staticmethod
    def protobuf_to_dict(message: Any) -> Dict[str, Any]:
        """Convert a protobuf message to a dictionary"""
        try:
            from google.protobuf.json_format import MessageToDict
            return MessageToDict(message, preserving_proto_field_name=True)
        except ImportError:
            logger.error("protobuf not installed")
            return {}
    
    @staticmethod
    def protobuf_to_json(message: Any, indent: Optional[int] = 2) -> str:
        """Convert a protobuf message to JSON string"""
        try:
            from google.protobuf.json_format import MessageToJson
            return MessageToJson(message, preserving_proto_field_name=True, indent=indent)
        except ImportError:
            logger.error("protobuf not installed")
            return "{}"
    
    @staticmethod
    def json_to_protobuf(json_str: str, message_class: Any) -> Any:
        """Convert JSON string to protobuf message"""
        try:
            from google.protobuf.json_format import Parse
            message = message_class()
            Parse(json_str, message, ignore_unknown_fields=True)
            return message
        except ImportError:
            logger.error("protobuf not installed")
            return None


class ProtobufSchema:
    """Protobuf schema definitions and documentation"""
    
    SCHEMAS = {
        'financial_data': {
            'name': 'FinancialData',
            'description': 'Financial data for stocks',
            'fields': {
                'data_type': {'type': 'string', 'description': 'Type of data'},
                'ticker': {'type': 'string', 'description': 'Stock ticker symbol'},
                'timestamp': {'type': 'string', 'description': 'ISO 8601 timestamp'},
                'price': {'type': 'double', 'description': 'Current stock price'},
                'open': {'type': 'double', 'description': 'Opening price'},
                'high': {'type': 'double', 'description': 'Daily high'},
                'low': {'type': 'double', 'description': 'Daily low'},
                'volume': {'type': 'int64', 'description': 'Trading volume'},
                'market_cap': {'type': 'int64', 'description': 'Market capitalization'},
                'pe_ratio': {'type': 'double', 'description': 'Price-to-earnings ratio'},
                'dividend_yield': {'type': 'double', 'description': 'Dividend yield'},
                'week_52_high': {'type': 'double', 'description': '52-week high'},
                'week_52_low': {'type': 'double', 'description': '52-week low'},
            },
        },
        'stock_movement': {
            'name': 'StockMovement',
            'description': 'Stock price movement and technical indicators',
            'fields': {
                'data_type': {'type': 'string', 'description': 'Type of data'},
                'ticker': {'type': 'string', 'description': 'Stock ticker'},
                'timestamp': {'type': 'string', 'description': 'ISO 8601 timestamp'},
                'price': {'type': 'double', 'description': 'Current price'},
                'price_change': {'type': 'double', 'description': 'Price change amount'},
                'price_change_percent': {'type': 'double', 'description': 'Price change %'},
                'high_52week': {'type': 'double', 'description': '52-week high'},
                'low_52week': {'type': 'double', 'description': '52-week low'},
                'sma_20': {'type': 'double', 'description': 'Simple moving average 20-day'},
                'sma_50': {'type': 'double', 'description': 'Simple moving average 50-day'},
                'rsi': {'type': 'double', 'description': 'Relative strength index'},
                'macd': {'type': 'double', 'description': 'MACD indicator'},
                'macd_signal': {'type': 'double', 'description': 'MACD signal line'},
                'volume': {'type': 'int64', 'description': 'Trading volume'},
            },
        },
        'news': {
            'name': 'NewsData',
            'description': 'News headlines and sentiment data',
            'fields': {
                'data_type': {'type': 'string', 'description': 'Type of data'},
                'ticker': {'type': 'string', 'description': 'Related stock ticker'},
                'headline': {'type': 'string', 'description': 'News headline'},
                'summary': {'type': 'string', 'description': 'Article summary'},
                'source': {'type': 'string', 'description': 'News source'},
                'url': {'type': 'string', 'description': 'Article URL'},
                'timestamp': {'type': 'string', 'description': 'Fetch timestamp'},
                'sentiment_polarity': {'type': 'double', 'description': 'Sentiment polarity (-1 to 1)'},
                'sentiment_subjectivity': {'type': 'double', 'description': 'Sentiment subjectivity (0 to 1)'},
                'published_date': {'type': 'string', 'description': 'Publication date'},
            },
        },
        'macroeconomic_data': {
            'name': 'MacroeconomicData',
            'description': 'Macroeconomic indicators',
            'fields': {
                'data_type': {'type': 'string', 'description': 'Type of data'},
                'indicator': {'type': 'string', 'description': 'Economic indicator name'},
                'symbol': {'type': 'string', 'description': 'Indicator symbol'},
                'value': {'type': 'double', 'description': 'Indicator value'},
                'unit': {'type': 'string', 'description': 'Measurement unit'},
                'date': {'type': 'string', 'description': 'Data date'},
                'timestamp': {'type': 'string', 'description': 'Fetch timestamp'},
                'source': {'type': 'string', 'description': 'Data source'},
            },
        },
        'policy_data': {
            'name': 'PolicyData',
            'description': 'Policy announcements and decisions',
            'fields': {
                'data_type': {'type': 'string', 'description': 'Type of data'},
                'event_type': {'type': 'string', 'description': 'Type of policy event'},
                'title': {'type': 'string', 'description': 'Event title'},
                'description': {'type': 'string', 'description': 'Event description'},
                'date': {'type': 'string', 'description': 'Event date'},
                'timestamp': {'type': 'string', 'description': 'Fetch timestamp'},
                'impact_level': {'type': 'string', 'description': 'Impact level (high/medium/low)'},
                'source': {'type': 'string', 'description': 'Data source'},
                'metadata': {'type': 'map<string,string>', 'description': 'Additional metadata'},
            },
        },
    }
    
    @classmethod
    def get_schema(cls, data_type: str) -> Dict[str, Any]:
        """Get schema for a data type"""
        return cls.SCHEMAS.get(data_type, {})
    
    @classmethod
    def print_schema(cls, data_type: str = None):
        """Print schema information"""
        if data_type:
            schema = cls.SCHEMAS.get(data_type)
            if schema:
                print(f"\n{schema['name']}: {schema['description']}")
                print("Fields:")
                for field_name, field_info in schema['fields'].items():
                    print(f"  - {field_name} ({field_info['type']}): {field_info['description']}")
        else:
            for dt, schema in cls.SCHEMAS.items():
                print(f"\n{schema['name']}: {schema['description']}")

