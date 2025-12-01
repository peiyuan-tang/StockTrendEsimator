#!/usr/bin/env python3
"""
Training Data Processing Examples

Demonstrates different ways to generate and use training data
for ML model development.
"""

import logging
from datetime import datetime, timedelta
from data_pipeline.core import TrainingDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic_generation():
    """Example 1: Basic training data generation"""
    logger.info("Example 1: Basic Training Data Generation")
    logger.info("=" * 60)
    
    config = {
        'data_root': '/data',
        'output_format': 'parquet',
        'output_path': '/data/training',
    }
    
    processor = TrainingDataProcessor(config)
    
    # Generate training data for last 30 days
    training_data = processor.generate_training_data()
    
    # Display summary
    processor.print_feature_summary(training_data)
    
    return training_data


def example_custom_date_range():
    """Example 2: Custom date range and tickers"""
    logger.info("\nExample 2: Custom Date Range and Tickers")
    logger.info("=" * 60)
    
    config = {
        'data_root': '/data',
        'output_format': 'parquet',
    }
    
    processor = TrainingDataProcessor(config)
    
    # Define custom date range
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    # Specific tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    training_data = processor.generate_training_data(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        save=True
    )
    
    logger.info(f"Generated {len(training_data)} records for {tickers}")
    logger.info(f"Columns: {list(training_data.columns)[:5]}...")
    
    return training_data


def example_individual_towers():
    """Example 3: Load from individual data towers"""
    logger.info("\nExample 3: Individual Data Tower Loading")
    logger.info("=" * 60)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    # Load stock data separately
    start = datetime(2024, 1, 1)
    end = datetime(2024, 12, 31)
    
    stock_data = processor.stock_tower.load_data(start, end)
    logger.info(f"Stock data: {len(stock_data)} records")
    if not stock_data.empty:
        logger.info(f"Stock columns: {list(stock_data.columns)}")
    
    # Load context data separately
    context_data = processor.context_tower.load_data(start, end)
    logger.info(f"Context data: {len(context_data)} records")
    if not context_data.empty:
        logger.info(f"Context columns: {list(context_data.columns)}")
    
    return stock_data, context_data


def example_multiple_formats():
    """Example 4: Save in different formats"""
    logger.info("\nExample 4: Multiple Output Formats")
    logger.info("=" * 60)
    
    formats = ['csv', 'parquet', 'json']
    
    for fmt in formats:
        config = {
            'data_root': '/data',
            'output_format': fmt,
            'output_path': f'/data/training/{fmt}',
        }
        
        processor = TrainingDataProcessor(config)
        
        # Generate for short date range (faster)
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"Generating in {fmt.upper()} format...")
        training_data = processor.generate_training_data(
            start_date=start_date,
            end_date=end_date,
            tickers=['AAPL', 'MSFT'],
            save=True
        )
        
        logger.info(f"✓ Saved {fmt.upper()} with {len(training_data)} records")


def example_feature_analysis():
    """Example 5: Analyze generated features"""
    logger.info("\nExample 5: Feature Analysis")
    logger.info("=" * 60)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    training_data = processor.generate_training_data(
        save=False
    )
    
    if training_data.empty:
        logger.warning("No training data generated")
        return
    
    # Get summary
    summary = processor.get_feature_summary(training_data)
    
    # Feature statistics
    logger.info(f"\nTotal Features: {summary['total_features']}")
    logger.info(f"Total Records: {summary['total_records']}")
    
    # Missing data analysis
    logger.info("\nMissing Data Analysis:")
    for col, count in summary['missing_values'].items():
        if count > 0:
            pct = (count / summary['total_records'] * 100)
            logger.info(f"  {col}: {count} ({pct:.1f}%)")
    
    # Data type summary
    logger.info("\nData Types:")
    for col, dtype in summary['data_types'].items():
        logger.info(f"  {col}: {dtype}")


def example_incremental_processing():
    """Example 6: Process data in time chunks"""
    logger.info("\nExample 6: Incremental Processing (Memory Efficient)")
    logger.info("=" * 60)
    
    import pandas as pd
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    all_data = []
    chunk_start = datetime(2024, 1, 1)
    chunk_size = timedelta(days=30)
    end_date = datetime(2024, 12, 31)
    
    chunk_num = 0
    while chunk_start < end_date:
        chunk_end = min(chunk_start + chunk_size, end_date)
        
        logger.info(f"Processing chunk {chunk_num + 1}: {chunk_start.date()} to {chunk_end.date()}")
        
        chunk_data = processor.generate_training_data(
            start_date=chunk_start,
            end_date=chunk_end,
            save=False
        )
        
        if not chunk_data.empty:
            all_data.append(chunk_data)
        
        chunk_start = chunk_end + timedelta(seconds=1)
        chunk_num += 1
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"✓ Combined {len(all_data)} chunks into {len(combined_data)} total records")
    else:
        logger.warning("No data collected from any chunks")


def example_data_quality_checks():
    """Example 7: Quality checks on generated data"""
    logger.info("\nExample 7: Data Quality Checks")
    logger.info("=" * 60)
    
    import pandas as pd
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    training_data = processor.generate_training_data(save=False)
    
    if training_data.empty:
        logger.warning("No training data to check")
        return
    
    # Quality metrics
    logger.info("\nData Quality Metrics:")
    
    # Completeness
    completeness = (1 - training_data.isnull().sum().sum() / (len(training_data) * len(training_data.columns))) * 100
    logger.info(f"  Completeness: {completeness:.2f}%")
    
    # Temporal coverage
    if 'timestamp' in training_data.columns:
        date_range = training_data['timestamp'].max() - training_data['timestamp'].min()
        logger.info(f"  Date range: {date_range.days} days")
    
    # Ticker coverage
    if 'ticker' in training_data.columns:
        ticker_count = training_data['ticker'].nunique()
        record_count = len(training_data)
        records_per_ticker = record_count / ticker_count if ticker_count > 0 else 0
        logger.info(f"  Tickers: {ticker_count}")
        logger.info(f"  Records per ticker: {records_per_ticker:.0f}")
    
    # Numeric columns
    numeric_cols = training_data.select_dtypes(include=['float64', 'int64']).columns
    logger.info(f"  Numeric features: {len(numeric_cols)}")
    logger.info(f"  Categorical features: {len(training_data.columns) - len(numeric_cols) - 2}")  # Subtract ticker, timestamp


def example_export_for_ml():
    """Example 8: Export data ready for ML models"""
    logger.info("\nExample 8: Export for ML Model Training")
    logger.info("=" * 60)
    
    import pandas as pd
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    training_data = processor.generate_training_data(save=False)
    
    if training_data.empty:
        logger.warning("No training data available")
        return
    
    # Basic preprocessing for ML
    ml_data = training_data.copy()
    
    # Ensure proper types
    numeric_cols = ml_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        ml_data[col] = pd.to_numeric(ml_data[col], errors='coerce')
    
    # Handle missing values
    ml_data[numeric_cols] = ml_data[numeric_cols].fillna(ml_data[numeric_cols].mean())
    
    logger.info(f"ML-ready data shape: {ml_data.shape}")
    logger.info(f"Features: {len(ml_data.columns)}")
    logger.info(f"Ready for training: {not ml_data.isnull().any().any()}")


if __name__ == '__main__':
    logger.info("\nTraining Data Processing Examples")
    logger.info("=" * 60)
    
    try:
        # Run examples
        # example_basic_generation()
        # example_custom_date_range()
        # example_individual_towers()
        # example_multiple_formats()
        # example_feature_analysis()
        # example_incremental_processing()
        example_data_quality_checks()
        # example_export_for_ml()
        
        logger.info("\n✓ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}", exc_info=True)
