#!/usr/bin/env python3
"""
Training Data Generation Examples - Unified Architecture

Examples demonstrating the new flattened training data structure
with intelligent source labeling (stock_, news_, macro_, policy_)
"""

import logging
from datetime import datetime, timedelta
from data_pipeline.core.training_data import TrainingDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Training Data Generation
# ============================================================================
def example_basic_generation():
    """Generate basic unified training dataset"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training Data Generation")
    print("="*80)
    
    config = {
        'data_root': '/data',
        'output_format': 'parquet',
        'output_path': '/data/training'
    }
    
    processor = TrainingDataProcessor(config)
    
    # Generate with default parameters (12 weeks, Mag 7 tickers)
    df = processor.generate_training_data()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Records: {len(df)}")
    
    # Show column prefixes
    prefixes = set(col.split('_')[0] for col in df.columns if '_' in col)
    print(f"\nData sources: {sorted(prefixes)}")
    
    return df


# ============================================================================
# Example 2: Custom Date Range and Tickers
# ============================================================================
def example_custom_parameters():
    """Generate training data with custom parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Custom Date Range and Tickers")
    print("="*80)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    # Custom 8-week period for specific tickers
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=56)  # 8 weeks
    tickers = ['AAPL', 'MSFT', 'NVDA']
    
    df = processor.generate_training_data(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        save=True
    )
    
    print(f"\nDate range: {start_date.date()} to {end_date.date()}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Records: {len(df)}")
    
    return df


# ============================================================================
# Example 3: Accessing Source-Specific Features
# ============================================================================
def example_source_specific_access():
    """Access features by source using column prefixes"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Accessing Source-Specific Features")
    print("="*80)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    df = processor.generate_training_data()
    
    # Extract stock features
    stock_cols = [col for col in df.columns if col.startswith('stock_')]
    print(f"\nStock features ({len(stock_cols)}):")
    for col in sorted(stock_cols)[:5]:
        print(f"  - {col}")
    if len(stock_cols) > 5:
        print(f"  ... and {len(stock_cols) - 5} more")
    
    # Extract news features
    news_cols = [col for col in df.columns if col.startswith('news_')]
    print(f"\nNews features ({len(news_cols)}):")
    for col in sorted(news_cols):
        print(f"  - {col}")
    
    # Extract macro features
    macro_cols = [col for col in df.columns if col.startswith('macro_')]
    print(f"\nMacro features ({len(macro_cols)}):")
    for col in sorted(macro_cols):
        print(f"  - {col}")
    
    # Extract policy features
    policy_cols = [col for col in df.columns if col.startswith('policy_')]
    print(f"\nPolicy features ({len(policy_cols)}):")
    for col in sorted(policy_cols):
        print(f"  - {col}")
    
    # Get stock data for specific ticker
    aapl_stock = df[df['ticker'] == 'AAPL'][['timestamp'] + stock_cols]
    print(f"\nAAPL stock data points: {len(aapl_stock)}")
    
    return df


# ============================================================================
# Example 4: Weekly Movement Analysis
# ============================================================================
def example_weekly_movement():
    """Analyze weekly stock movements using calculated features"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Weekly Movement Analysis")
    print("="*80)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    df = processor.generate_training_data(
        include_weekly_movement=True
    )
    
    # Get weekly movement columns
    movement_cols = [col for col in df.columns if 'weekly' in col]
    print(f"\nWeekly movement features ({len(movement_cols)}):")
    for col in sorted(movement_cols):
        print(f"  - {col}")
    
    # Analyze movements
    movement_df = df[['ticker', 'timestamp'] + movement_cols].dropna(subset=movement_cols)
    
    print(f"\nRecords with movement data: {len(movement_df)}")
    
    if not movement_df.empty:
        # Summary by ticker
        print("\nMovement summary by ticker:")
        for ticker in sorted(df['ticker'].unique()):
            ticker_data = movement_df[movement_df['ticker'] == ticker]
            if not ticker_data.empty:
                avg_return = ticker_data['stock_weekly_price_return'].mean()
                max_return = ticker_data['stock_weekly_price_return'].max()
                min_return = ticker_data['stock_weekly_price_return'].min()
                print(f"  {ticker}: avg={avg_return:.2f}%, max={max_return:.2f}%, min={min_return:.2f}%")
    
    return df


# ============================================================================
# Example 5: Feature Summary and Data Quality
# ============================================================================
def example_feature_summary():
    """Display feature summary grouped by source"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Feature Summary and Data Quality")
    print("="*80)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    df = processor.generate_training_data()
    
    # Print formatted summary
    processor.print_feature_summary(df)
    
    # Get detailed summary as dict
    summary = processor.get_feature_summary(df)
    
    print("\nDetailed Source Analysis:")
    for source, info in summary['sources'].items():
        print(f"\n{source.upper()}:")
        print(f"  Feature Count: {info['feature_count']}")
        
        # Check missing values
        missing_info = summary['missing_values_by_source'][source]
        missing_count = sum(missing_info.values())
        if missing_count > 0:
            missing_pct = (missing_count / (summary['total_records'] * len(missing_info))) * 100
            print(f"  Missing Values: {missing_count} ({missing_pct:.1f}%)")
    
    return df


# ============================================================================
# Example 6: Data Export in Different Formats
# ============================================================================
def example_export_formats():
    """Generate and export training data in multiple formats"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Export in Different Formats")
    print("="*80)
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=28)  # 4 weeks
    
    # CSV format
    print("\nGenerating CSV export...")
    config_csv = {
        'data_root': '/data',
        'output_format': 'csv',
        'output_path': '/data/training/csv'
    }
    processor_csv = TrainingDataProcessor(config_csv)
    df_csv = processor_csv.generate_training_data(
        start_date=start_date,
        end_date=end_date,
        tickers=['AAPL', 'MSFT'],
        save=True
    )
    print(f"✓ CSV export complete: {len(df_csv)} records")
    
    # Parquet format (optimized for ML)
    print("\nGenerating Parquet export...")
    config_parquet = {
        'data_root': '/data',
        'output_format': 'parquet',
        'output_path': '/data/training/parquet'
    }
    processor_parquet = TrainingDataProcessor(config_parquet)
    df_parquet = processor_parquet.generate_training_data(
        start_date=start_date,
        end_date=end_date,
        tickers=['AAPL', 'MSFT'],
        save=True
    )
    print(f"✓ Parquet export complete: {len(df_parquet)} records")
    
    # JSON format
    print("\nGenerating JSON export...")
    config_json = {
        'data_root': '/data',
        'output_format': 'json',
        'output_path': '/data/training/json'
    }
    processor_json = TrainingDataProcessor(config_json)
    df_json = processor_json.generate_training_data(
        start_date=start_date,
        end_date=end_date,
        tickers=['AAPL', 'MSFT'],
        save=True
    )
    print(f"✓ JSON export complete: {len(df_json)} records")
    
    return {
        'csv': df_csv,
        'parquet': df_parquet,
        'json': df_json
    }


# ============================================================================
# Example 7: Prepare Data for ML Pipeline
# ============================================================================
def example_ml_preparation():
    """Prepare training data for machine learning models"""
    print("\n" + "="*80)
    print("EXAMPLE 7: ML Pipeline Preparation")
    print("="*80)
    
    import numpy as np
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    df = processor.generate_training_data(
        tickers=['AAPL', 'MSFT', 'GOOGL'],
        include_weekly_movement=True
    )
    
    print(f"\nOriginal dataset: {df.shape}")
    
    # 1. Remove key identification columns for modeling
    feature_df = df.drop(columns=['ticker', 'timestamp'])
    print(f"Features for model: {feature_df.shape}")
    
    # 2. Handle missing values
    feature_df_filled = feature_df.fillna(feature_df.mean())
    print(f"After filling NaN: {feature_df_filled.shape}")
    
    # 3. Group features by source for feature engineering
    stock_features = [col for col in feature_df.columns if col.startswith('stock_')]
    news_features = [col for col in feature_df.columns if col.startswith('news_')]
    macro_features = [col for col in feature_df.columns if col.startswith('macro_')]
    policy_features = [col for col in feature_df.columns if col.startswith('policy_')]
    
    print(f"\nFeature Groups:")
    print(f"  Stock features: {len(stock_features)}")
    print(f"  News features: {len(news_features)}")
    print(f"  Macro features: {len(macro_features)}")
    print(f"  Policy features: {len(policy_features)}")
    
    # 4. Optional: Create feature subsets
    print(f"\nFeature Subset Examples:")
    
    # Stock-only features
    stock_only = feature_df_filled[stock_features]
    print(f"  Stock-only model inputs: {stock_only.shape}")
    
    # Stock + context
    context_features = [col for col in feature_df.columns 
                       if col.startswith(('news_', 'macro_', 'policy_'))]
    stock_with_context = feature_df_filled[stock_features + context_features]
    print(f"  Stock + context model inputs: {stock_with_context.shape}")
    
    # All features
    all_features = feature_df_filled
    print(f"  All features model inputs: {all_features.shape}")
    
    return {
        'full_df': df,
        'features': feature_df_filled,
        'stock_only': stock_only,
        'stock_with_context': stock_with_context,
        'all_features': all_features
    }


# ============================================================================
# Example 8: Time Series Analysis
# ============================================================================
def example_time_series_analysis():
    """Analyze time series characteristics of unified data"""
    print("\n" + "="*80)
    print("EXAMPLE 8: Time Series Analysis")
    print("="*80)
    
    config = {'data_root': '/data'}
    processor = TrainingDataProcessor(config)
    
    df = processor.generate_training_data(
        tickers=['AAPL', 'MSFT'],
        include_weekly_movement=True
    )
    
    print(f"\nTimestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Analyze temporal distribution
    df['week'] = df['timestamp'].dt.isocalendar().week
    print(f"\nWeeks in dataset: {df['week'].nunique()}")
    
    # Records per ticker
    print(f"\nRecords per ticker:")
    for ticker in sorted(df['ticker'].unique()):
        count = len(df[df['ticker'] == ticker])
        print(f"  {ticker}: {count} records")
    
    # Check data continuity
    print(f"\nData continuity check:")
    for ticker in sorted(df['ticker'].unique()):
        ticker_data = df[df['ticker'] == ticker].sort_values('timestamp')
        if len(ticker_data) > 1:
            dates = ticker_data['timestamp'].unique()
            gaps = sum(1 for i in range(1, len(dates)) 
                      if (dates[i] - dates[i-1]).days > 7)
            print(f"  {ticker}: {gaps} gaps > 7 days")
    
    return df


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*80)
    print("UNIFIED TRAINING DATA GENERATION EXAMPLES")
    print("="*80)
    
    # Run examples
    try:
        df1 = example_basic_generation()
        df2 = example_custom_parameters()
        df3 = example_source_specific_access()
        df4 = example_weekly_movement()
        df5 = example_feature_summary()
        df6 = example_export_formats()
        df7 = example_ml_preparation()
        df8 = example_time_series_analysis()
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}", exc_info=True)
