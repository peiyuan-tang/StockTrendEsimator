#!/usr/bin/env python3
"""
Data Sink Implementations - Store processed data to various formats
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseSink(ABC):
    """Base class for data sinks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.path = config.get('path', '/data/default')
        self.file_prefix = config.get('file_prefix', 'data_')
        self.file_suffix = config.get('file_suffix', '.json')
        self.batch_size = config.get('batch_size', 100)

    @abstractmethod
    def write(self, events: List[Dict[str, Any]]) -> bool:
        """Write events to sink"""
        pass

    def _expand_path(self) -> str:
        """Expand path with datetime patterns"""
        path_template = self.path
        now = datetime.utcnow()
        
        # Replace datetime patterns
        path_expanded = path_template.replace('%Y', str(now.year))
        path_expanded = path_expanded.replace('%m', f"{now.month:02d}")
        path_expanded = path_expanded.replace('%d', f"{now.day:02d}")
        path_expanded = path_expanded.replace('%H', f"{now.hour:02d}")
        
        return path_expanded

    def _ensure_directory(self, directory: str):
        """Ensure directory exists"""
        Path(directory).mkdir(parents=True, exist_ok=True)


class JSONSink(BaseSink):
    """Write events to JSON files"""

    def write(self, events: List[Dict[str, Any]]) -> bool:
        """Write events to JSON file"""
        try:
            directory = self._expand_path()
            self._ensure_directory(directory)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.file_prefix}{timestamp}{self.file_suffix}"
            filepath = os.path.join(directory, filename)
            
            # Extract event bodies
            data = [event.get('body', event) for event in events]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Wrote {len(events)} events to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to JSON sink: {str(e)}")
            return False


class ParquetSink(BaseSink):
    """Write events to Parquet files"""

    def write(self, events: List[Dict[str, Any]]) -> bool:
        """Write events to Parquet file"""
        try:
            import pandas as pd
            
            directory = self._expand_path()
            self._ensure_directory(directory)
            
            # Extract event bodies
            data = [event.get('body', event) for event in events]
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.file_prefix}{timestamp}{self.file_suffix}"
            filepath = os.path.join(directory, filename)
            
            # Get compression from config
            compression = self.config.get('compression', 'snappy')
            
            # Write Parquet
            df.to_parquet(
                filepath,
                compression=compression,
                index=False,
                engine='pyarrow'
            )
            
            logger.info(f"Wrote {len(events)} events to {filepath}")
            return True
            
        except ImportError:
            logger.error("pandas or pyarrow not installed. Install with: pip install pandas pyarrow")
            return False
        except Exception as e:
            logger.error(f"Error writing to Parquet sink: {str(e)}")
            return False


class CSVSink(BaseSink):
    """Write events to CSV files"""

    def write(self, events: List[Dict[str, Any]]) -> bool:
        """Write events to CSV file"""
        try:
            import pandas as pd
            
            directory = self._expand_path()
            self._ensure_directory(directory)
            
            # Extract event bodies
            data = [event.get('body', event) for event in events]
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.file_prefix}{timestamp}{self.file_suffix}"
            filepath = os.path.join(directory, filename)
            
            # Write CSV
            df.to_csv(filepath, index=False)
            
            logger.info(f"Wrote {len(events)} events to {filepath}")
            return True
            
        except ImportError:
            logger.error("pandas not installed. Install with: pip install pandas")
            return False
        except Exception as e:
            logger.error(f"Error writing to CSV sink: {str(e)}")
            return False


class DatabaseSink(BaseSink):
    """Write events to database (PostgreSQL, MongoDB, etc.)"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db_type = config.get('db_type', 'postgresql')
        self.connection_string = config.get('connection_string')
        self.table_name = config.get('table_name', 'events')

    def write(self, events: List[Dict[str, Any]]) -> bool:
        """Write events to database"""
        try:
            if self.db_type == 'postgresql':
                return self._write_postgresql(events)
            elif self.db_type == 'mongodb':
                return self._write_mongodb(events)
            else:
                logger.error(f"Unsupported database type: {self.db_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing to database sink: {str(e)}")
            return False

    def _write_postgresql(self, events: List[Dict[str, Any]]) -> bool:
        """Write to PostgreSQL"""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.connection_string)
            cursor = conn.cursor()
            
            # Extract event bodies
            data = [event.get('body', event) for event in events]
            
            for record in data:
                columns = record.keys()
                values = [record[col] for col in columns]
                
                insert_sql = f"""
                    INSERT INTO {self.table_name} ({','.join(columns)})
                    VALUES ({','.join(['%s'] * len(columns))})
                """
                
                cursor.execute(insert_sql, values)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Wrote {len(events)} events to PostgreSQL")
            return True
            
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            return False

    def _write_mongodb(self, events: List[Dict[str, Any]]) -> bool:
        """Write to MongoDB"""
        try:
            import pymongo
            
            client = pymongo.MongoClient(self.connection_string)
            db = client.get_database()
            collection = db[self.table_name]
            
            # Extract event bodies
            data = [event.get('body', event) for event in events]
            
            result = collection.insert_many(data)
            
            logger.info(f"Wrote {len(result.inserted_ids)} events to MongoDB")
            return True
            
        except ImportError:
            logger.error("pymongo not installed. Install with: pip install pymongo")
            return False


class ProtobufSink(BaseSink):
    """Write events to Protocol Buffer (protobuf) format"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.file_suffix = config.get('file_suffix', '.pb')
        self.batch_records = config.get('batch_records', False)
        
        try:
            from google.protobuf.json_format import MessageToJson, Parse
            from data_pipeline.storage import events_pb2
            self.events_pb2 = events_pb2
            self.MessageToJson = MessageToJson
            self.Parse = Parse
        except ImportError as e:
            self.logger.error(f"protobuf not installed. Install with: pip install protobuf")
            raise

    def _dict_to_protobuf(self, record: Dict[str, Any]):
        """Convert dictionary record to appropriate protobuf message"""
        data_type = record.get('data_type', 'unknown')
        
        try:
            if data_type == 'financial_data':
                msg = self.events_pb2.FinancialData()
                msg.data_type = record.get('data_type', '')
                msg.ticker = record.get('ticker', '')
                msg.timestamp = record.get('timestamp', '')
                msg.price = float(record.get('price', 0))
                msg.open = float(record.get('open', 0))
                msg.high = float(record.get('high', 0))
                msg.low = float(record.get('low', 0))
                msg.volume = int(record.get('volume', 0))
                msg.market_cap = int(record.get('market_cap', 0)) if record.get('market_cap') else 0
                msg.pe_ratio = float(record.get('pe_ratio', 0)) if record.get('pe_ratio') else 0
                msg.dividend_yield = float(record.get('dividend_yield', 0)) if record.get('dividend_yield') else 0
                msg.week_52_high = float(record.get('52_week_high', 0)) if record.get('52_week_high') else 0
                msg.week_52_low = float(record.get('52_week_low', 0)) if record.get('52_week_low') else 0
                return msg
                
            elif data_type == 'stock_movement':
                msg = self.events_pb2.StockMovement()
                msg.data_type = record.get('data_type', '')
                msg.ticker = record.get('ticker', '')
                msg.timestamp = record.get('timestamp', '')
                msg.price = float(record.get('price', 0))
                msg.price_change = float(record.get('price_change', 0)) if record.get('price_change') else 0
                msg.price_change_percent = float(record.get('price_change_percent', 0)) if record.get('price_change_percent') else 0
                msg.high_52week = float(record.get('high_52week', 0)) if record.get('high_52week') else 0
                msg.low_52week = float(record.get('low_52week', 0)) if record.get('low_52week') else 0
                msg.sma_20 = float(record.get('SMA_20', 0)) if record.get('SMA_20') else 0
                msg.sma_50 = float(record.get('SMA_50', 0)) if record.get('SMA_50') else 0
                msg.rsi = float(record.get('RSI', 0)) if record.get('RSI') else 0
                msg.macd = float(record.get('MACD', 0)) if record.get('MACD') else 0
                msg.macd_signal = float(record.get('MACD_Signal', 0)) if record.get('MACD_Signal') else 0
                msg.volume = int(record.get('volume', 0))
                return msg
                
            elif data_type == 'news':
                msg = self.events_pb2.NewsData()
                msg.data_type = record.get('data_type', '')
                msg.ticker = record.get('ticker', '')
                msg.headline = record.get('headline', '')
                msg.summary = record.get('summary', '')
                msg.source = record.get('source', '')
                msg.url = record.get('url', '')
                msg.timestamp = record.get('timestamp', '')
                msg.sentiment_polarity = float(record.get('sentiment_polarity', 0)) if record.get('sentiment_polarity') else 0
                msg.sentiment_subjectivity = float(record.get('sentiment_subjectivity', 0)) if record.get('sentiment_subjectivity') else 0
                msg.published_date = record.get('published_date', '')
                return msg
                
            elif data_type == 'macroeconomic_data':
                msg = self.events_pb2.MacroeconomicData()
                msg.data_type = record.get('data_type', '')
                msg.indicator = record.get('indicator', '')
                msg.symbol = record.get('symbol', '')
                msg.value = float(record.get('value', 0))
                msg.unit = record.get('unit', '')
                msg.date = record.get('date', '')
                msg.timestamp = record.get('timestamp', '')
                msg.source = record.get('source', '')
                return msg
                
            elif data_type == 'policy_data':
                msg = self.events_pb2.PolicyData()
                msg.data_type = record.get('data_type', '')
                msg.event_type = record.get('event_type', '')
                msg.title = record.get('title', '')
                msg.description = record.get('description', '')
                msg.date = record.get('date', '')
                msg.timestamp = record.get('timestamp', '')
                msg.impact_level = record.get('impact_level', '')
                msg.source = record.get('source', '')
                
                # Add metadata as key-value pairs
                if 'metadata' in record and isinstance(record['metadata'], dict):
                    for key, value in record['metadata'].items():
                        msg.metadata[key] = str(value)
                
                return msg
            else:
                # Generic event for unknown types
                msg = self.events_pb2.Event()
                msg.event_type = data_type
                msg.timestamp = record.get('timestamp', '')
                if 'headers' in record and isinstance(record['headers'], dict):
                    for key, value in record['headers'].items():
                        msg.headers[key] = str(value)
                msg.body = json.dumps(record).encode('utf-8')
                return msg
                
        except Exception as e:
            logger.warning(f"Error converting record to protobuf: {str(e)}")
            return None

    def write(self, events: List[Dict[str, Any]]) -> bool:
        """Write events to protobuf format"""
        try:
            directory = self._expand_path()
            self._ensure_directory(directory)
            
            # Extract event bodies
            data = [event.get('body', event) for event in events]
            
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.file_prefix}{timestamp}{self.file_suffix}"
            filepath = os.path.join(directory, filename)
            
            if self.batch_records:
                # Write as batch
                batch = self.events_pb2.DataBatch()
                batch.batch_timestamp = datetime.utcnow().isoformat()
                
                for record in data:
                    msg = self._dict_to_protobuf(record)
                    if msg:
                        data_type = record.get('data_type', '')
                        if data_type == 'financial_data':
                            batch.financial_data.append(msg)
                        elif data_type == 'stock_movement':
                            batch.stock_movements.append(msg)
                        elif data_type == 'news':
                            batch.news_data.append(msg)
                        elif data_type == 'macroeconomic_data':
                            batch.macro_data.append(msg)
                        elif data_type == 'policy_data':
                            batch.policy_data.append(msg)
                
                with open(filepath, 'wb') as f:
                    f.write(batch.SerializeToString())
            else:
                # Write individual messages (delimited)
                with open(filepath, 'wb') as f:
                    for record in data:
                        msg = self._dict_to_protobuf(record)
                        if msg:
                            # Write message length-delimited (protobuf convention)
                            msg_bytes = msg.SerializeToString()
                            f.write(len(msg_bytes).to_bytes(4, byteorder='big'))
                            f.write(msg_bytes)
            
            logger.info(f"Wrote {len(events)} events to protobuf file {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing to protobuf sink: {str(e)}")
            return False


class SinkFactory:
    """Factory for creating sink instances"""

    SINK_TYPES = {
        'json': JSONSink,
        'parquet': ParquetSink,
        'csv': CSVSink,
        'database': DatabaseSink,
        'protobuf': ProtobufSink,
    }

    @classmethod
    def create_sink(cls, sink_type: str, config: Dict[str, Any]) -> BaseSink:
        """Create a sink instance"""
        if sink_type not in cls.SINK_TYPES:
            raise ValueError(f"Unknown sink type: {sink_type}")
        
        sink_class = cls.SINK_TYPES[sink_type]
        return sink_class(config)
