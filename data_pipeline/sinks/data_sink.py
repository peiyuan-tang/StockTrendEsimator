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


class SinkFactory:
    """Factory for creating sink instances"""

    SINK_TYPES = {
        'json': JSONSink,
        'parquet': ParquetSink,
        'csv': CSVSink,
        'database': DatabaseSink,
    }

    @classmethod
    def create_sink(cls, sink_type: str, config: Dict[str, Any]) -> BaseSink:
        """Create a sink instance"""
        if sink_type not in cls.SINK_TYPES:
            raise ValueError(f"Unknown sink type: {sink_type}")
        
        sink_class = cls.SINK_TYPES[sink_type]
        return sink_class(config)
