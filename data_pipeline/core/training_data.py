#!/usr/bin/env python3
"""
Training Data Processor - Unifies and joins data from all sources

Combines:
1. Stock Data Tower: Financial data + Stock movements
2. Context Data Tower: News + Macroeconomic + Policy data

Generates unified training dataset for ML models
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataTower(ABC):
    """Base class for data towers"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data tower"""
        self.config = config
        self.data_root = config.get('data_root', '/data')
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load and process data for this tower"""
        pass

    @abstractmethod
    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize schema for joining"""
        pass


class StockDataTower(DataTower):
    """
    Stock Data Tower: Combines financial and technical indicator data
    
    Structure:
    - ticker: Stock symbol (AAPL, MSFT, etc.)
    - timestamp: Event timestamp
    - price: Current stock price
    - ohlc: Open, High, Low, Close
    - volume: Trading volume
    - market_cap: Market capitalization
    - pe_ratio: P/E ratio
    - dividend_yield: Dividend yield
    - 52_week_high/low: 52-week range
    - sma_20/50: Simple moving averages
    - rsi: Relative strength index
    - macd: MACD indicator
    """

    def load_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load financial and stock movement data
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            tickers: List of tickers to load (default: Mag 7)
            
        Returns:
            DataFrame with stock data
        """
        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

        # Load financial data
        financial_df = self._load_financial_data(start_date, end_date, tickers)
        self.logger.info(f"Loaded {len(financial_df)} financial records")

        # Load stock movements (technical indicators)
        movement_df = self._load_stock_movements(start_date, end_date)
        self.logger.info(f"Loaded {len(movement_df)} movement records")

        # Merge on ticker and timestamp (allowing for minor time mismatches)
        merged_df = self._merge_data_frames(financial_df, movement_df)
        self.logger.info(f"Merged stock data: {len(merged_df)} records")

        return merged_df

    def _load_financial_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        tickers: List[str]
    ) -> pd.DataFrame:
        """Load financial data files"""
        data_dir = os.path.join(self.data_root, 'raw', 'financial_data')
        
        return self._load_json_files(
            data_dir,
            start_date,
            end_date,
            filter_tickers=tickers
        )

    def _load_stock_movements(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load stock movement data with technical indicators"""
        data_dir = os.path.join(self.data_root, 'raw', 'stock_movements')
        
        return self._load_json_files(data_dir, start_date, end_date)

    def _load_json_files(
        self,
        data_dir: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        filter_tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load and parse JSON files from directory"""
        records = []
        
        if not os.path.exists(data_dir):
            self.logger.warning(f"Data directory not found: {data_dir}")
            return pd.DataFrame()

        try:
            for file in os.listdir(data_dir):
                if not file.endswith('.json'):
                    continue

                file_path = os.path.join(data_dir, file)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        # Handle list of records
                        if isinstance(data, list):
                            records.extend(data)
                        elif isinstance(data, dict):
                            records.append(data)
                            
                except Exception as e:
                    self.logger.warning(f"Error reading {file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading files from {data_dir}: {str(e)}")

        df = pd.DataFrame(records)
        
        # Filter by date range
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]

        # Filter by tickers if provided
        if filter_tickers and 'ticker' in df.columns:
            df = df[df['ticker'].isin(filter_tickers)]

        return df

    def _merge_data_frames(self, financial_df: pd.DataFrame, movement_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge financial and movement data
        Joins on ticker and nearest timestamp
        """
        if financial_df.empty or movement_df.empty:
            return financial_df if not financial_df.empty else movement_df

        # Rename timestamp columns to distinguish during merge
        financial_df = financial_df.copy()
        movement_df = movement_df.copy()

        financial_df.rename(columns={'timestamp': 'fin_timestamp'}, inplace=True)
        movement_df.rename(columns={'timestamp': 'mov_timestamp'}, inplace=True)

        # Merge on ticker with nearest timestamp (asof merge)
        if 'ticker' in financial_df.columns and 'ticker' in movement_df.columns:
            financial_df = financial_df.sort_values('fin_timestamp')
            movement_df = movement_df.sort_values('mov_timestamp')
            
            merged = pd.merge_asof(
                financial_df,
                movement_df,
                on=['ticker', 'fin_timestamp'],
                by='ticker',
                direction='nearest',
                tolerance=pd.Timedelta(minutes=60)
            )
            
            # Use financial timestamp as primary
            merged['timestamp'] = merged['fin_timestamp']
            merged = merged.drop(columns=['fin_timestamp', 'mov_timestamp'], errors='ignore')
        else:
            merged = financial_df.copy()

        return merged

    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize stock data schema"""
        normalized = df.copy()
        
        # Ensure required columns exist
        required_columns = {
            'ticker': str,
            'timestamp': 'datetime64[ns]',
            'price': float,
            'volume': float,
            'market_cap': float,
        }

        for col, dtype in required_columns.items():
            if col not in normalized.columns:
                normalized[col] = None
            if dtype == float:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
            elif dtype == 'datetime64[ns]':
                normalized[col] = pd.to_datetime(normalized[col], errors='coerce')

        return normalized


class ContextDataTower(DataTower):
    """
    Context Data Tower: Combines contextual information
    
    Structure:
    - ticker: Stock symbol
    - timestamp: Event timestamp
    - news_sentiment: Sentiment polarity and subjectivity
    - macro_indicators: Economic indicators
    - policy_events: Policy announcements
    """

    def load_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load news, macroeconomic, and policy data
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            tickers: List of tickers to load
            
        Returns:
            DataFrame with context data
        """
        # Load all context data sources
        news_df = self._load_news_data(start_date, end_date, tickers)
        self.logger.info(f"Loaded {len(news_df)} news records")

        macro_df = self._load_macro_data(start_date, end_date)
        self.logger.info(f"Loaded {len(macro_df)} macro records")

        policy_df = self._load_policy_data(start_date, end_date)
        self.logger.info(f"Loaded {len(policy_df)} policy records")

        # Combine context sources
        combined_df = self._combine_context_data(news_df, macro_df, policy_df)
        self.logger.info(f"Combined context data: {len(combined_df)} records")

        return combined_df

    def _load_news_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        tickers: Optional[List[str]]
    ) -> pd.DataFrame:
        """Load news data with sentiment"""
        data_dir = os.path.join(self.data_root, 'raw', 'news')
        
        df = self._load_json_files(data_dir, start_date, end_date)
        
        if not df.empty and tickers:
            if 'ticker' in df.columns:
                df = df[df['ticker'].isin(tickers)]

        return df

    def _load_macro_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load macroeconomic data"""
        data_dir = os.path.join(self.data_root, 'raw', 'macroeconomic_data')
        
        return self._load_json_files(data_dir, start_date, end_date)

    def _load_policy_data(
        self,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Load policy announcements"""
        data_dir = os.path.join(self.data_root, 'raw', 'policy_data')
        
        return self._load_json_files(data_dir, start_date, end_date)

    def _load_json_files(
        self,
        data_dir: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Load and parse JSON files"""
        records = []
        
        if not os.path.exists(data_dir):
            self.logger.warning(f"Data directory not found: {data_dir}")
            return pd.DataFrame()

        try:
            for file in os.listdir(data_dir):
                if not file.endswith('.json'):
                    continue

                file_path = os.path.join(data_dir, file)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                        if isinstance(data, list):
                            records.extend(data)
                        elif isinstance(data, dict):
                            records.append(data)
                            
                except Exception as e:
                    self.logger.warning(f"Error reading {file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading files from {data_dir}: {str(e)}")

        df = pd.DataFrame(records)
        
        # Filter by date range
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]

        return df

    def _combine_context_data(
        self,
        news_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        policy_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine multiple context data sources"""
        combined = pd.DataFrame()

        # Add news data with prefix
        if not news_df.empty:
            news_df = news_df.copy()
            news_cols = {col: f'news_{col}' for col in news_df.columns if col not in ['ticker', 'timestamp']}
            news_df.rename(columns=news_cols, inplace=True)
            combined = news_df

        # Add macro data (typically doesn't have ticker)
        if not macro_df.empty:
            macro_df = macro_df.copy()
            if 'ticker' not in macro_df.columns:
                macro_df['ticker'] = 'MARKET'
            
            macro_cols = {col: f'macro_{col}' for col in macro_df.columns if col not in ['ticker', 'timestamp']}
            macro_df.rename(columns=macro_cols, inplace=True)
            
            if combined.empty:
                combined = macro_df
            else:
                combined = pd.merge(combined, macro_df, on=['ticker', 'timestamp'], how='outer')

        # Add policy data (typically doesn't have ticker)
        if not policy_df.empty:
            policy_df = policy_df.copy()
            if 'ticker' not in policy_df.columns:
                policy_df['ticker'] = 'MARKET'
            
            policy_cols = {col: f'policy_{col}' for col in policy_df.columns if col not in ['ticker', 'timestamp']}
            policy_df.rename(columns=policy_cols, inplace=True)
            
            if combined.empty:
                combined = policy_df
            else:
                combined = pd.merge(combined, policy_df, on=['ticker', 'timestamp'], how='outer')

        return combined

    def normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize context data schema"""
        normalized = df.copy()
        
        # Ensure required columns exist
        required_columns = {
            'ticker': str,
            'timestamp': 'datetime64[ns]',
        }

        for col, dtype in required_columns.items():
            if col not in normalized.columns:
                normalized[col] = None
            if dtype == 'datetime64[ns]':
                normalized[col] = pd.to_datetime(normalized[col], errors='coerce')

        return normalized


class TrainingDataProcessor:
    """
    Main processor that orchestrates data tower loading and joining
    
    Outputs unified training dataset with:
    - Stock features (price, volume, technical indicators)
    - Context features (news sentiment, macro indicators, policy)
    - Aligned timestamps for time-series analysis
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize training data processor
        
        Args:
            config: Configuration dictionary with keys:
                - data_root: Root directory for data (default: /data)
                - output_format: 'csv', 'parquet', 'json' (default: 'parquet')
                - output_path: Path to save training data (default: /data/training)
        """
        self.config = config
        self.data_root = config.get('data_root', '/data')
        self.output_format = config.get('output_format', 'parquet')
        self.output_path = config.get('output_path', os.path.join(self.data_root, 'training'))
        
        self.stock_tower = StockDataTower(config)
        self.context_tower = ContextDataTower(config)
        
        self.logger = logging.getLogger(__name__)

    def generate_training_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tickers: Optional[List[str]] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Generate unified training dataset
        
        Args:
            start_date: Start date for data (default: 30 days ago)
            end_date: End date for data (default: today)
            tickers: List of tickers (default: Mag 7)
            save: Whether to save the output
            
        Returns:
            DataFrame with training data
        """
        # Set default date range
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

        self.logger.info(
            f"Generating training data for {len(tickers)} tickers "
            f"from {start_date.date()} to {end_date.date()}"
        )

        # Load data from both towers
        stock_data = self.stock_tower.load_data(start_date, end_date, tickers)
        stock_data = self.stock_tower.normalize_schema(stock_data)
        
        context_data = self.context_tower.load_data(start_date, end_date, tickers)
        context_data = self.context_tower.normalize_schema(context_data)

        # Join stock and context data
        training_data = self._join_towers(stock_data, context_data)

        self.logger.info(f"Generated training data with {len(training_data)} rows and {len(training_data.columns)} columns")

        # Save if requested
        if save:
            self._save_training_data(training_data)

        return training_data

    def _join_towers(self, stock_df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join stock and context data towers
        
        Performs outer join to preserve all records from both sources
        """
        if stock_df.empty and context_df.empty:
            self.logger.warning("Both data towers are empty")
            return pd.DataFrame()

        if stock_df.empty:
            return context_df
        if context_df.empty:
            return stock_df

        # Join on ticker and timestamp
        merged = pd.merge(
            stock_df,
            context_df,
            on=['ticker', 'timestamp'],
            how='outer'
        )

        # Sort by ticker and timestamp
        merged = merged.sort_values(['ticker', 'timestamp']).reset_index(drop=True)

        self.logger.info(f"Joined towers: {len(merged)} records")

        return merged

    def _save_training_data(self, df: pd.DataFrame) -> str:
        """
        Save training data to disk
        
        Args:
            df: Training dataset
            
        Returns:
            Path to saved file
        """
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        if self.output_format == 'csv':
            filepath = os.path.join(self.output_path, f'training_data_{timestamp}.csv')
            df.to_csv(filepath, index=False)
            
        elif self.output_format == 'parquet':
            filepath = os.path.join(self.output_path, f'training_data_{timestamp}.parquet')
            df.to_parquet(filepath, index=False)
            
        elif self.output_format == 'json':
            filepath = os.path.join(self.output_path, f'training_data_{timestamp}.json')
            df.to_json(filepath, orient='records', date_format='iso')
            
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        self.logger.info(f"Saved training data to {filepath} ({file_size:.2f} MB)")

        return filepath

    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of training data
        
        Args:
            df: Training dataset
            
        Returns:
            Dictionary with feature summary
        """
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'date_range': {
                'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None,
            },
            'tickers': df['ticker'].unique().tolist() if 'ticker' in df.columns else [],
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numeric_summary': df.describe().to_dict(),
        }

        return summary

    def print_feature_summary(self, df: pd.DataFrame):
        """Print formatted feature summary"""
        summary = self.get_feature_summary(df)
        
        print("\n" + "="*80)
        print("TRAINING DATA SUMMARY")
        print("="*80)
        print(f"Total Records: {summary['total_records']}")
        print(f"Total Features: {summary['total_features']}")
        print(f"Tickers: {', '.join(summary['tickers'])}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print(f"\nFeature Columns:")
        for col, dtype in summary['data_types'].items():
            print(f"  - {col}: {dtype}")
        
        print(f"\nMissing Values:")
        for col, count in summary['missing_values'].items():
            pct = (count / summary['total_records'] * 100) if summary['total_records'] > 0 else 0
            if count > 0:
                print(f"  - {col}: {count} ({pct:.2f}%)")
        
        print("="*80 + "\n")
