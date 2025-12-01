#!/usr/bin/env python3
"""
Training Data Processor - Unified Weekly Data Loading

Combines all data sources into a single flattened dataset:
- Stock data: Financial metrics + technical indicators (labeled: stock_*)
- Market context: News sentiment (labeled: news_*)
- Macro indicators: Economic data (labeled: macro_*)
- Policy data: Policy announcements (labeled: policy_*)

All data normalized to weekly granularity for consistent training.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pandas as pd

logger = logging.getLogger(__name__)


class UnifiedTrainingDataProcessor:
    """
    Unified processor that loads all data sources with consistent weekly granularity
    
    Features:
    - Single data loading pipeline (no separate towers)
    - Automatic source labeling via column prefixes
    - Flattened output structure for direct ML use
    - Weekly time alignment across all sources
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize unified training data processor
        
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
        self.logger = logging.getLogger(__name__)

    def generate_training_data(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tickers: Optional[List[str]] = None,
        save: bool = True,
        include_weekly_movement: bool = True
    ) -> pd.DataFrame:
        """
        Generate unified flattened training dataset
        
        All data is normalized to weekly granularity with source labels:
        - stock_*: Financial data + technical indicators
        - news_*: News sentiment data
        - macro_*: Macroeconomic indicators
        - policy_*: Policy announcements
        
        Args:
            start_date: Start date for data (default: 12 weeks ago)
            end_date: End date for data (default: today)
            tickers: List of tickers (default: Mag 7)
            save: Whether to save the output
            include_weekly_movement: Whether to calculate weekly movement deltas (default: True)
            
        Returns:
            Flattened DataFrame with all features
        """
        # Set default date range - 12 weeks (84 days) for adequate weekly data
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=84)

        if tickers is None:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

        self.logger.info(
            f"Generating unified training data for {len(tickers)} tickers "
            f"from {start_date.date()} to {end_date.date()}"
        )

        # Load all data sources with unified weekly granularity
        stock_data = self._load_stock_data(start_date, end_date, tickers)
        news_data = self._load_news_data(start_date, end_date, tickers)
        macro_data = self._load_macro_data(start_date, end_date)
        policy_data = self._load_policy_data(start_date, end_date)

        # Join all data on ticker and weekly timestamp
        training_data = self._join_all_sources(stock_data, news_data, macro_data, policy_data)

        # Add weekly movement calculations if requested
        if include_weekly_movement and not training_data.empty:
            training_data = self._add_weekly_movement(training_data)

        self.logger.info(
            f"Generated unified training data: {len(training_data)} rows, "
            f"{len(training_data.columns)} columns"
        )

        # Save if requested
        if save and not training_data.empty:
            self._save_training_data(training_data)

        return training_data

    def _load_stock_data(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Load and normalize stock data (financial + technical indicators)
        
        Sources:
        - Financial metrics: price, volume, market cap, P/E ratio, etc.
        - Technical indicators: SMA, RSI, MACD, etc.
        
        Returns DataFrame with columns prefixed by 'stock_'
        """
        # Load financial data
        financial_df = self._load_json_files(
            os.path.join(self.data_root, 'raw', 'financial_data'),
            start_date, end_date,
            filter_tickers=tickers
        )
        self.logger.info(f"Loaded {len(financial_df)} financial records")

        # Load stock movements (technical indicators)
        movement_df = self._load_json_files(
            os.path.join(self.data_root, 'raw', 'stock_movements'),
            start_date, end_date
        )
        self.logger.info(f"Loaded {len(movement_df)} movement records")

        # Merge financial and movement data
        if not financial_df.empty and not movement_df.empty:
            stock_df = pd.merge(
                financial_df,
                movement_df,
                on=['ticker', 'timestamp'],
                how='outer'
            )
        elif not financial_df.empty:
            stock_df = financial_df.copy()
        elif not movement_df.empty:
            stock_df = movement_df.copy()
        else:
            stock_df = pd.DataFrame()

        # Add source label prefix to all stock columns except key fields
        if not stock_df.empty:
            stock_df = self._prefix_columns(stock_df, 'stock_', exclude=['ticker', 'timestamp'])
            self.logger.info(f"Stock data: {len(stock_df)} records after labeling")

        return stock_df

    def _load_news_data(
        self,
        start_date: datetime,
        end_date: datetime,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Load and normalize news data with sentiment
        
        Returns DataFrame with columns prefixed by 'news_'
        """
        news_df = self._load_json_files(
            os.path.join(self.data_root, 'raw', 'news'),
            start_date, end_date
        )

        # Filter by tickers if provided
        if not news_df.empty and tickers and 'ticker' in news_df.columns:
            news_df = news_df[news_df['ticker'].isin(tickers)].copy()

        self.logger.info(f"Loaded {len(news_df)} news records")

        # Add source label prefix
        if not news_df.empty:
            news_df = self._prefix_columns(news_df, 'news_', exclude=['ticker', 'timestamp'])

        return news_df

    def _load_macro_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load and normalize macroeconomic indicators
        
        Returns DataFrame with columns prefixed by 'macro_'
        """
        macro_df = self._load_json_files(
            os.path.join(self.data_root, 'context', 'macroeconomic'),
            start_date, end_date
        )

        self.logger.info(f"Loaded {len(macro_df)} macro records")

        # Ensure ticker column exists (macro data typically doesn't have ticker)
        if not macro_df.empty:
            if 'ticker' not in macro_df.columns:
                macro_df['ticker'] = 'MARKET'
            
            # Add source label prefix
            macro_df = self._prefix_columns(macro_df, 'macro_', exclude=['ticker', 'timestamp'])

        return macro_df

    def _load_policy_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Load and normalize policy announcements
        
        Returns DataFrame with columns prefixed by 'policy_'
        """
        policy_df = self._load_json_files(
            os.path.join(self.data_root, 'context', 'policy'),
            start_date, end_date
        )

        self.logger.info(f"Loaded {len(policy_df)} policy records")

        # Ensure ticker column exists (policy data typically doesn't have ticker)
        if not policy_df.empty:
            if 'ticker' not in policy_df.columns:
                policy_df['ticker'] = 'MARKET'
            
            # Add source label prefix
            policy_df = self._prefix_columns(policy_df, 'policy_', exclude=['ticker', 'timestamp'])

        return policy_df

    def _load_json_files(
        self,
        data_dir: str,
        start_date: datetime,
        end_date: datetime,
        filter_tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Load and parse JSON files from directory with date filtering"""
        records = []
        
        if not os.path.exists(data_dir):
            self.logger.debug(f"Data directory not found: {data_dir}")
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
                    self.logger.debug(f"Error reading {file}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error loading files from {data_dir}: {str(e)}")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Filter by date range
        if not df.empty and 'timestamp' in df.columns:
            if start_date:
                df = df[df['timestamp'] >= start_date]
            if end_date:
                df = df[df['timestamp'] <= end_date]

        # Filter by tickers if provided
        if filter_tickers and 'ticker' in df.columns:
            df = df[df['ticker'].isin(filter_tickers)]

        return df

    def _prefix_columns(
        self,
        df: pd.DataFrame,
        prefix: str,
        exclude: List[str]
    ) -> pd.DataFrame:
        """
        Add prefix to all columns except excluded ones
        
        Args:
            df: DataFrame to modify
            prefix: Prefix to add (e.g., 'stock_', 'news_')
            exclude: List of columns to exclude from prefixing
            
        Returns:
            DataFrame with prefixed columns
        """
        df = df.copy()
        rename_map = {
            col: f"{prefix}{col}"
            for col in df.columns
            if col not in exclude
        }
        return df.rename(columns=rename_map)

    def _join_all_sources(
        self,
        stock_df: pd.DataFrame,
        news_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        policy_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Join all data sources on ticker and timestamp
        
        Uses outer join to preserve all records; missing values filled with NaN
        """
        # Start with stock data as base (most frequent)
        if not stock_df.empty:
            result = stock_df.copy()
        else:
            result = pd.DataFrame()

        # Join news data
        if not news_df.empty:
            if result.empty:
                result = news_df.copy()
            else:
                result = pd.merge(
                    result,
                    news_df,
                    on=['ticker', 'timestamp'],
                    how='outer'
                )

        # Join macro data
        if not macro_df.empty:
            if result.empty:
                result = macro_df.copy()
            else:
                result = pd.merge(
                    result,
                    macro_df,
                    on=['ticker', 'timestamp'],
                    how='outer'
                )

        # Join policy data
        if not policy_df.empty:
            if result.empty:
                result = policy_df.copy()
            else:
                result = pd.merge(
                    result,
                    policy_df,
                    on=['ticker', 'timestamp'],
                    how='outer'
                )

        # Sort by ticker and timestamp for consistency
        if not result.empty and 'ticker' in result.columns and 'timestamp' in result.columns:
            result = result.sort_values(['ticker', 'timestamp']).reset_index(drop=True)
            self.logger.info(f"Joined all sources: {len(result)} records")

        return result

    def _add_weekly_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weekly stock movement calculations
        
        Calculates:
        - stock_weekly_open_price: Opening price at start of week
        - stock_weekly_close_price: Closing price at end of week
        - stock_weekly_price_delta: Absolute price change (close - open)
        - stock_weekly_price_return: Percentage return ((close - open) / open * 100)
        - stock_weekly_movement_direction: 1 (up), -1 (down), 0 (flat)
        """
        if df.empty or 'ticker' not in df.columns or 'timestamp' not in df.columns:
            self.logger.warning("Cannot calculate weekly movement: missing required columns")
            return df
        
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize weekly movement columns
        df['stock_weekly_open_price'] = None
        df['stock_weekly_close_price'] = None
        df['stock_weekly_price_delta'] = None
        df['stock_weekly_price_return'] = None
        df['stock_weekly_movement_direction'] = None
        
        try:
            # Group by ticker and week
            df['year_week'] = (
                df['timestamp'].dt.isocalendar().year.astype(str) + '-W' +
                df['timestamp'].dt.isocalendar().week.astype(str).str.zfill(2)
            )
            
            # Process each ticker separately
            for ticker in df['ticker'].unique():
                ticker_mask = df['ticker'] == ticker
                
                # Group by week for this ticker
                for week_group, week_df in df[ticker_mask].groupby('year_week'):
                    # Find opening price (first record of week)
                    opening_price = None
                    for col in ['stock_open', 'stock_price', 'stock_opening_price']:
                        if col in week_df.columns:
                            vals = week_df[col].dropna()
                            if not vals.empty:
                                opening_price = float(vals.iloc[0])
                                break
                    
                    # Find closing price (last record of week)
                    closing_price = None
                    for col in ['stock_close', 'stock_price', 'stock_closing_price']:
                        if col in week_df.columns:
                            vals = week_df[col].dropna()
                            if not vals.empty:
                                closing_price = float(vals.iloc[-1])
                                break
                    
                    # Calculate deltas if both prices available
                    if opening_price is not None and closing_price is not None:
                        try:
                            price_delta = closing_price - opening_price
                            price_return = (price_delta / opening_price * 100) if opening_price != 0 else 0
                            direction = 1 if price_delta > 0 else (-1 if price_delta < 0 else 0)
                            
                            # Update all records in this week for this ticker
                            week_indices = df[
                                (df['ticker'] == ticker) & (df['year_week'] == week_group)
                            ].index
                            
                            df.loc[week_indices, 'stock_weekly_open_price'] = opening_price
                            df.loc[week_indices, 'stock_weekly_close_price'] = closing_price
                            df.loc[week_indices, 'stock_weekly_price_delta'] = price_delta
                            df.loc[week_indices, 'stock_weekly_price_return'] = price_return
                            df.loc[week_indices, 'stock_weekly_movement_direction'] = direction
                            
                        except (ValueError, TypeError) as e:
                            self.logger.debug(f"Error calculating weekly movement for {ticker} {week_group}: {e}")
            
            # Drop temporary week column
            df = df.drop(columns=['year_week'])
            self.logger.info("Added weekly movement features (stock_weekly_*)")
            
        except Exception as e:
            self.logger.error(f"Error calculating weekly movements: {str(e)}")
        
        return df

    def _save_training_data(self, df: pd.DataFrame) -> str:
        """
        Save training data to disk
        
        Args:
            df: Training dataset
            
        Returns:
            Path to saved file
        """
        os.makedirs(self.output_path, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        if self.output_format == 'csv':
            filepath = os.path.join(self.output_path, f'training_data_unified_{timestamp}.csv')
            df.to_csv(filepath, index=False)
            
        elif self.output_format == 'parquet':
            filepath = os.path.join(self.output_path, f'training_data_unified_{timestamp}.parquet')
            df.to_parquet(filepath, index=False)
            
        elif self.output_format == 'json':
            filepath = os.path.join(self.output_path, f'training_data_unified_{timestamp}.json')
            df.to_json(filepath, orient='records', date_format='iso')
            
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

        file_size = os.path.getsize(filepath) / (1024 * 1024)
        self.logger.info(f"Saved unified training data to {filepath} ({file_size:.2f} MB)")

        return filepath

    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of training data grouped by source
        
        Returns feature counts and types organized by data source
        """
        if df.empty:
            return {}

        # Group columns by source
        sources = {}
        for col in df.columns:
            if col in ['ticker', 'timestamp']:
                continue
            
            source = col.split('_')[0]  # Extract source from prefix
            if source not in sources:
                sources[source] = []
            sources[source].append(col)

        summary = {
            'total_records': len(df),
            'total_features': len(df.columns) - 2,  # Exclude ticker and timestamp
            'date_range': {
                'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None,
            },
            'tickers': sorted(df['ticker'].unique().tolist()) if 'ticker' in df.columns else [],
            'sources': {
                source: {
                    'feature_count': len(cols),
                    'features': sorted(cols),
                }
                for source, cols in sorted(sources.items())
            },
            'missing_values_by_source': {
                source: {
                    col: int(df[col].isnull().sum())
                    for col in cols
                }
                for source, cols in sources.items()
            },
        }

        return summary

    def print_feature_summary(self, df: pd.DataFrame):
        """Print formatted feature summary organized by source"""
        summary = self.get_feature_summary(df)
        
        if not summary:
            print("No data to summarize")
            return
        
        print("\n" + "="*80)
        print("UNIFIED TRAINING DATA SUMMARY")
        print("="*80)
        print(f"Total Records: {summary['total_records']}")
        print(f"Total Features: {summary['total_features']}")
        print(f"Tickers: {', '.join(summary['tickers'])}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print(f"\nFeatures by Source:")
        for source in sorted(summary['sources'].keys()):
            source_info = summary['sources'][source]
            print(f"  {source.upper()}: {source_info['feature_count']} features")
            for feature in sorted(source_info['features'])[:5]:  # Show first 5
                print(f"    - {feature}")
            if len(source_info['features']) > 5:
                print(f"    ... and {len(source_info['features']) - 5} more")
        
        print(f"\nMissing Values by Source:")
        for source in sorted(summary['missing_values_by_source'].keys()):
            missing = summary['missing_values_by_source'][source]
            missing_count = sum(missing.values())
            if missing_count > 0:
                pct = (missing_count / summary['total_records'] / len(missing) * 100)
                print(f"  {source.upper()}: {missing_count} missing values ({pct:.1f}% avg)")
        
        print("="*80 + "\n")


# Maintain backward compatibility
TrainingDataProcessor = UnifiedTrainingDataProcessor
