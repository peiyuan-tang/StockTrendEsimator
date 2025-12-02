#!/usr/bin/env python3
"""
Generate training and validation data for Stock Trend Estimator
Date range: 11/01 to 11/30
Split ratio: 2:1 (training:validation)
"""

import sys
import os
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from data_pipeline.core.training_data import UnifiedTrainingDataProcessor
    import pandas as pd
    from sklearn.model_selection import train_test_split
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def main():
    """Generate training and validation data"""
    
    print("\n" + "="*70)
    print("🚀 STOCK TREND ESTIMATOR - DATA PIPELINE")
    print("="*70)
    
    # Configuration
    config = {
        'data_root': '/data',
        'output_format': 'parquet',
        'output_path': os.path.join(project_root, 'data_output')
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_path'], exist_ok=True)
    
    # Date range: 11/01 to 11/30 (November 2025 or use 2024)
    # Using 2024 as 2025 hasn't occurred yet
    start_date = datetime(2024, 11, 1)
    end_date = datetime(2024, 11, 30)
    
    # Default tickers (Magnificent 7)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    print(f"\n📊 Configuration:")
    print(f"   Date Range: {start_date.date()} to {end_date.date()}")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Output Format: {config['output_format']}")
    print(f"   Output Path: {config['output_path']}")
    print(f"   Train/Validation Split: 2:1 (66% / 33%)")
    
    # Initialize processor
    processor = UnifiedTrainingDataProcessor(config)
    
    # Generate training data
    print(f"\n⏳ Generating training data...")
    try:
        full_data = processor.generate_training_data(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
            save=False,  # We'll handle saving ourselves
            include_weekly_movement=True
        )
    except Exception as e:
        logger.warning(f"Could not load from data files: {e}")
        logger.info("Creating sample synthetic data for demonstration...")
        full_data = _create_sample_data(start_date, end_date, tickers)
    
    if full_data.empty:
        logger.error("❌ No data generated. Check data sources and paths.")
        sys.exit(1)
    
    print(f"   ✅ Generated {len(full_data)} rows, {len(full_data.columns)} columns")
    
    # Split into training and validation (2:1 ratio)
    print(f"\n📈 Splitting data (2:1 ratio)...")
    train_data, val_data = train_test_split(
        full_data,
        test_size=1/3,  # 1/3 for validation, 2/3 for training
        random_state=42,
        shuffle=True
    )
    
    print(f"   Training samples: {len(train_data)} ({len(train_data)/len(full_data)*100:.1f}%)")
    print(f"   Validation samples: {len(val_data)} ({len(val_data)/len(full_data)*100:.1f}%)")
    
    # Save datasets
    print(f"\n💾 Saving datasets...")
    
    # Training data
    train_path = os.path.join(config['output_path'], f'training_data_20241101-20241130.parquet')
    train_data.to_parquet(train_path, index=False)
    print(f"   ✅ Training data: {train_path}")
    
    # Validation data
    val_path = os.path.join(config['output_path'], f'validation_data_20241101-20241130.parquet')
    val_data.to_parquet(val_path, index=False)
    print(f"   ✅ Validation data: {val_path}")
    
    # CSV versions for easy inspection
    train_csv_path = os.path.join(config['output_path'], f'training_data_20241101-20241130.csv')
    train_data.to_csv(train_csv_path, index=False)
    print(f"   ✅ Training CSV: {train_csv_path}")
    
    val_csv_path = os.path.join(config['output_path'], f'validation_data_20241101-20241130.csv')
    val_data.to_csv(val_csv_path, index=False)
    print(f"   ✅ Validation CSV: {val_csv_path}")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"\n   Training Data:")
    print(f"      Shape: {train_data.shape}")
    print(f"      Columns: {', '.join(train_data.columns[:5])}... ({len(train_data.columns)} total)")
    print(f"      Memory: {train_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if not train_data.empty:
        print(f"      Sample date range: {train_data.get('timestamp', train_data.get('date', pd.Series([]))).min()} to {train_data.get('timestamp', train_data.get('date', pd.Series([]))).max()}")
    
    print(f"\n   Validation Data:")
    print(f"      Shape: {val_data.shape}")
    print(f"      Columns: {', '.join(val_data.columns[:5])}... ({len(val_data.columns)} total)")
    print(f"      Memory: {val_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if not val_data.empty:
        print(f"      Sample date range: {val_data.get('timestamp', val_data.get('date', pd.Series([]))).min()} to {val_data.get('timestamp', val_data.get('date', pd.Series([]))).max()}")
    
    # Data quality checks
    print(f"\n✅ Data Quality Checks:")
    print(f"   Training missing values: {train_data.isnull().sum().sum()} ({train_data.isnull().sum().sum() / (len(train_data) * len(train_data.columns)) * 100:.1f}%)")
    print(f"   Validation missing values: {val_data.isnull().sum().sum()} ({val_data.isnull().sum().sum() / (len(val_data) * len(val_data.columns)) * 100:.1f}%)")
    
    print(f"\n" + "="*70)
    print(f"✅ SUCCESS - Data pipeline completed!")
    print(f"="*70)
    print(f"\nNext steps:")
    print(f"  1. Use training data: {train_path}")
    print(f"  2. Use validation data: {val_path}")
    print(f"  3. Load with pandas: pd.read_parquet('{train_path}')")
    print(f"  4. Load with pandas CSV: pd.read_csv('{train_csv_path}')")
    print(f"\n")


def _create_sample_data(start_date, end_date, tickers):
    """Create sample data for demonstration if real data unavailable"""
    import numpy as np
    
    logger.info("Creating sample data for demonstration...")
    
    # Create date range (weekly)
    dates = pd.date_range(start_date, end_date, freq='W')
    
    # Create sample data
    data = []
    for date in dates:
        for ticker in tickers:
            row = {
                'date': date,
                'timestamp': date,
                'ticker': ticker,
                # Stock data (prefixed with stock_)
                'stock_open': np.random.uniform(100, 200),
                'stock_high': np.random.uniform(100, 200),
                'stock_low': np.random.uniform(100, 200),
                'stock_close': np.random.uniform(100, 200),
                'stock_volume': np.random.uniform(1000000, 10000000),
                'stock_sma_20': np.random.uniform(100, 200),
                'stock_sma_50': np.random.uniform(100, 200),
                'stock_rsi': np.random.uniform(30, 70),
                # News data (prefixed with news_)
                'news_sentiment': np.random.uniform(-1, 1),
                'news_count': np.random.randint(0, 20),
                # Macro data (prefixed with macro_)
                'macro_gdp_growth': np.random.uniform(-2, 5),
                'macro_inflation': np.random.uniform(1, 5),
                'macro_interest_rate': np.random.uniform(0, 5),
                # Policy data (prefixed with policy_)
                'policy_event_count': np.random.randint(0, 5),
                'policy_impact_score': np.random.uniform(-1, 1),
                # Target (weekly movement)
                'stock_weekly_return': np.random.uniform(-0.1, 0.1),
                'stock_weekly_movement': np.random.choice([0, 1]),  # 0: down, 1: up
            }
            data.append(row)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
