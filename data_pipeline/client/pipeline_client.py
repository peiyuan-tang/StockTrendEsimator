#!/usr/bin/env python3
"""
Data Pipeline Client - Interface for querying and managing the data collection pipeline
Pure offline client - no real-time serving
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import json

logger = logging.getLogger(__name__)


class DataPipelineClient:
    """
    Client interface for the data collection pipeline.
    Queries and manages collected data without real-time serving.
    """

    def __init__(self, data_root: str = '/data'):
        """
        Initialize the client
        
        Args:
            data_root: Root directory for collected data
        """
        self.data_root = data_root
        self.raw_data_path = os.path.join(data_root, 'raw')
        self.context_data_path = os.path.join(data_root, 'context')

    def get_financial_data(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve financial data for Mag 7 stocks
        
        Args:
            tickers: List of tickers (default: Mag 7)
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with financial data
        """
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
        
        data_dir = os.path.join(self.raw_data_path, 'financial_data')
        return self._load_data_files(data_dir, start_date, end_date)

    def get_stock_movements(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve stock movement trends for S&P 500
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            indicators: List of indicators to include
            
        Returns:
            DataFrame with stock movement data
        """
        data_dir = os.path.join(self.raw_data_path, 'stock_movements')
        df = self._load_data_files(data_dir, start_date, end_date)
        
        if indicators and 'indicators' in df.columns:
            # Filter to specific indicators if requested
            pass
        
        return df

    def get_news_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sentiment_filter: Optional[tuple] = None,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Retrieve news data for S&P 500 stocks
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            sentiment_filter: Tuple of (min_polarity, max_polarity)
            tickers: Filter by specific tickers
            
        Returns:
            DataFrame with news data and sentiment scores
        """
        data_dir = os.path.join(self.raw_data_path, 'news')
        df = self._load_data_files(data_dir, start_date, end_date)
        
        if sentiment_filter and 'sentiment' in df.columns:
            min_pol, max_pol = sentiment_filter
            df = df[(df['sentiment'].apply(lambda x: x.get('polarity', 0)) >= min_pol) &
                    (df['sentiment'].apply(lambda x: x.get('polarity', 0)) <= max_pol)]
        
        if tickers:
            df = df[df['ticker'].isin(tickers)]
        
        return df

    def get_macroeconomic_data(
        self,
        indicators: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve macroeconomic data for Mag 7 stocks
        
        Args:
            indicators: List of specific indicators
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with macroeconomic indicators
        """
        data_dir = os.path.join(self.context_data_path, 'macroeconomic')
        df = self._load_data_files(data_dir, start_date, end_date)
        
        if indicators and 'indicators' in df.columns:
            # Filter to specific indicators if requested
            pass
        
        return df

    def get_policy_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Retrieve fiscal and monetary policy data for Mag 7 stocks
        
        Args:
            data_types: List of data types (e.g., 'fed_announcement', 'fomc_meeting')
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame with policy data
        """
        data_dir = os.path.join(self.context_data_path, 'policy')
        df = self._load_data_files(data_dir, start_date, end_date)
        
        if data_types and 'data_type' in df.columns:
            df = df[df['data_type'].isin(data_types)]
        
        return df

    def _load_data_files(
        self,
        directory: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load data files from directory with date filtering
        
        Args:
            directory: Directory to load from
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Combined DataFrame from all matching files
        """
        dfs = []
        
        if not os.path.exists(directory):
            logger.warning(f"Data directory not found: {directory}")
            return pd.DataFrame()
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    
                    # Check date filters
                    if start_date or end_date:
                        file_date_str = os.path.basename(root)
                        try:
                            file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                            if start_date and file_date < start_date:
                                continue
                            if end_date and file_date > end_date:
                                continue
                        except ValueError:
                            pass
                    
                    # Load file based on format
                    try:
                        if file.endswith('.parquet'):
                            df = pd.read_parquet(filepath)
                        elif file.endswith('.csv'):
                            df = pd.read_csv(filepath)
                        elif file.endswith('.json'):
                            df = pd.read_json(filepath)
                        else:
                            continue
                        
                        dfs.append(df)
                        logger.debug(f"Loaded {len(df)} records from {file}")
                    except Exception as e:
                        logger.warning(f"Error loading {filepath}: {str(e)}")
            
            if dfs:
                return pd.concat(dfs, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data files: {str(e)}")
            return pd.DataFrame()

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data"""
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_sources': {
                'financial_data': self._count_files(os.path.join(self.raw_data_path, 'financial_data')),
                'stock_movements': self._count_files(os.path.join(self.raw_data_path, 'stock_movements')),
                'news': self._count_files(os.path.join(self.raw_data_path, 'news')),
                'macroeconomic': self._count_files(os.path.join(self.context_data_path, 'macroeconomic')),
                'policy': self._count_files(os.path.join(self.context_data_path, 'policy')),
            },
            'total_size_bytes': self._get_directory_size(self.data_root),
        }
        return summary

    def _count_files(self, directory: str) -> int:
        """Count files in directory"""
        if not os.path.exists(directory):
            return 0
        
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if not f.startswith('.')])
        return count

    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        if os.path.exists(directory):
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        return total_size

    def export_data(
        self,
        data_type: str,
        output_path: str,
        format: str = 'csv',
        **kwargs
    ) -> bool:
        """
        Export collected data to specified format
        
        Args:
            data_type: Type of data to export (financial_data, movements, news, macro, policy)
            output_path: Path for output file
            format: Output format (csv, parquet, json)
            **kwargs: Additional arguments passed to data retrieval methods
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if data_type == 'financial_data':
                df = self.get_financial_data(**kwargs)
            elif data_type == 'movements':
                df = self.get_stock_movements(**kwargs)
            elif data_type == 'news':
                df = self.get_news_data(**kwargs)
            elif data_type == 'macro':
                df = self.get_macroeconomic_data(**kwargs)
            elif data_type == 'policy':
                df = self.get_policy_data(**kwargs)
            else:
                logger.error(f"Unknown data type: {data_type}")
                return False
            
            if df.empty:
                logger.warning(f"No data found for {data_type}")
                return False
            
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'csv':
                df.to_csv(output_path, index=False)
            elif format == 'parquet':
                df.to_parquet(output_path, index=False)
            elif format == 'json':
                df.to_json(output_path, orient='records', indent=2, default_handler=str)
            else:
                logger.error(f"Unknown format: {format}")
                return False
            
            logger.info(f"Exported {len(df)} records to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False


# Global client instance
_client = None


def get_data_client(data_root: str = '/data') -> DataPipelineClient:
    """Get global data pipeline client"""
    global _client
    if _client is None:
        _client = DataPipelineClient(data_root=data_root)
    return _client
