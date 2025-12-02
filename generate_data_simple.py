#!/usr/bin/env python3
"""
Generate training and validation data for Stock Trend Estimator
Date range: 11/01 to 11/30
Split ratio: 2:1 (training:validation)

This is a standalone script that doesn't require external dependencies
beyond pandas (which is standard for data science work).
"""

import sys
import os
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate training and validation data"""
    
    print("\n" + "="*70)
    print("🚀 STOCK TREND ESTIMATOR - DATA PIPELINE")
    print("="*70)
    
    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_root, 'data_output')
    
    # Create output directory if it doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Date range: 11/01 to 11/30 (November)
    start_date = datetime(2024, 11, 1)
    end_date = datetime(2024, 11, 30)
    
    # Default tickers (Magnificent 7)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    print(f"\n📊 Configuration:")
    print(f"   Date Range: {start_date.date()} to {end_date.date()}")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Output Format: CSV & Parquet")
    print(f"   Output Path: {output_path}")
    print(f"   Train/Validation Split: 2:1 (66% / 33%)")
    
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        logger.error("❌ pandas not installed. Please run: pip install pandas numpy")
        sys.exit(1)
    
    print(f"\n⏳ Generating synthetic training data...")
    
    # Create date range (weekly)
    dates = pd.date_range(start_date, end_date, freq='W')
    if len(dates) == 0:
        # If only 1 month, create weekly dates manually
        dates = pd.date_range(start_date, end_date, freq='D')
        dates = dates[::7]  # Take every 7th day
    
    print(f"   Generated {len(dates)} weekly periods")
    
    # Create sample data
    data = []
    np.random.seed(42)  # For reproducibility
    
    for date in dates:
        for ticker in tickers:
            # Generate realistic stock prices
            base_price = np.random.uniform(100, 500)
            
            row = {
                'date': date,
                'timestamp': date,
                'ticker': ticker,
                # Stock data (prefixed with stock_)
                'stock_open': base_price,
                'stock_high': base_price * np.random.uniform(1.0, 1.05),
                'stock_low': base_price * np.random.uniform(0.95, 1.0),
                'stock_close': base_price * np.random.uniform(0.95, 1.05),
                'stock_volume': np.random.uniform(1e7, 1e8),
                'stock_sma_20': base_price * np.random.uniform(0.98, 1.02),
                'stock_sma_50': base_price * np.random.uniform(0.97, 1.03),
                'stock_rsi': np.random.uniform(30, 70),
                'stock_macd': np.random.uniform(-5, 5),
                'stock_bb_upper': base_price * 1.02,
                'stock_bb_lower': base_price * 0.98,
                # News data (prefixed with news_)
                'news_sentiment': np.random.uniform(-1, 1),
                'news_count': np.random.randint(0, 20),
                'news_relevance': np.random.uniform(0, 1),
                # Macro data (prefixed with macro_)
                'macro_gdp_growth': np.random.uniform(-2, 5),
                'macro_inflation': np.random.uniform(1, 5),
                'macro_unemployment': np.random.uniform(3, 8),
                'macro_interest_rate': np.random.uniform(0, 5),
                'macro_vix': np.random.uniform(10, 40),
                # Policy data (prefixed with policy_)
                'policy_event_count': np.random.randint(0, 5),
                'policy_impact_score': np.random.uniform(-1, 1),
                # Target (weekly movement)
                'stock_weekly_return': np.random.uniform(-0.1, 0.1),
                'stock_weekly_movement': np.random.choice([0, 1]),  # 0: down, 1: up
                'stock_trend_direction': np.random.choice([-1, 0, 1]),  # -1: down, 0: neutral, 1: up
            }
            data.append(row)
    
    full_data = pd.DataFrame(data)
    print(f"   ✅ Generated {len(full_data)} rows, {len(full_data.columns)} columns")
    
    # Shuffle data
    full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into training and validation (2:1 ratio)
    print(f"\n📈 Splitting data (2:1 ratio)...")
    split_idx = int(len(full_data) * 2 / 3)
    train_data = full_data[:split_idx]
    val_data = full_data[split_idx:]
    
    print(f"   Training samples: {len(train_data)} ({len(train_data)/len(full_data)*100:.1f}%)")
    print(f"   Validation samples: {len(val_data)} ({len(val_data)/len(full_data)*100:.1f}%)")
    
    # Save datasets
    print(f"\n💾 Saving datasets...")
    
    # Training data
    train_parquet_path = os.path.join(output_path, 'training_data_20241101-20241130.parquet')
    train_csv_path = os.path.join(output_path, 'training_data_20241101-20241130.csv')
    
    try:
        train_data.to_parquet(train_parquet_path, index=False)
        print(f"   ✅ Training Parquet: {train_parquet_path}")
    except Exception as e:
        logger.warning(f"Could not save Parquet: {e}")
    
    train_data.to_csv(train_csv_path, index=False)
    print(f"   ✅ Training CSV: {train_csv_path}")
    
    # Validation data
    val_parquet_path = os.path.join(output_path, 'validation_data_20241101-20241130.parquet')
    val_csv_path = os.path.join(output_path, 'validation_data_20241101-20241130.csv')
    
    try:
        val_data.to_parquet(val_parquet_path, index=False)
        print(f"   ✅ Validation Parquet: {val_parquet_path}")
    except Exception as e:
        logger.warning(f"Could not save Parquet: {e}")
    
    val_data.to_csv(val_csv_path, index=False)
    print(f"   ✅ Validation CSV: {val_csv_path}")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"\n   Training Data:")
    print(f"      Shape: {train_data.shape}")
    print(f"      Features: {', '.join(train_data.columns[:8])}... ({len(train_data.columns)} total)")
    print(f"      Memory: {train_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"      Date range: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"      Unique tickers: {train_data['ticker'].nunique()}")
    
    print(f"\n   Validation Data:")
    print(f"      Shape: {val_data.shape}")
    print(f"      Features: {', '.join(val_data.columns[:8])}... ({len(val_data.columns)} total)")
    print(f"      Memory: {val_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"      Date range: {val_data['date'].min()} to {val_data['date'].max()}")
    print(f"      Unique tickers: {val_data['ticker'].nunique()}")
    
    # Data quality checks
    print(f"\n✅ Data Quality Checks:")
    print(f"   Training missing values: {train_data.isnull().sum().sum()} total")
    print(f"   Validation missing values: {val_data.isnull().sum().sum()} total")
    print(f"   Training duplicates: {train_data.duplicated().sum()}")
    print(f"   Validation duplicates: {val_data.duplicated().sum()}")
    
    # Feature summary
    print(f"\n📈 Feature Summary:")
    numeric_cols = train_data.select_dtypes(include=['number']).columns
    for col in numeric_cols[:10]:
        print(f"   {col}:")
        print(f"      Mean: {train_data[col].mean():.4f}, Std: {train_data[col].std():.4f}")
    
    # Show sample rows
    print(f"\n📄 Sample Training Data (first 3 rows):")
    print(train_data.head(3).to_string())
    
    print(f"\n" + "="*70)
    print(f"✅ SUCCESS - Data pipeline completed!")
    print(f"="*70)
    print(f"\nOutput Files:")
    print(f"  📦 Training (Parquet): {train_parquet_path}")
    print(f"  📄 Training (CSV): {train_csv_path}")
    print(f"  📦 Validation (Parquet): {val_parquet_path}")
    print(f"  📄 Validation (CSV): {val_csv_path}")
    
    print(f"\nNext steps:")
    print(f"  1. Load training: pd.read_csv('{train_csv_path}')")
    print(f"  2. Load validation: pd.read_csv('{val_csv_path}')")
    print(f"  3. Use with ML models: train_data.head()")
    print(f"\nData Statistics:")
    print(f"  • Training samples: {len(train_data)} (66%)")
    print(f"  • Validation samples: {len(val_data)} (33%)")
    print(f"  • Total samples: {len(full_data)}")
    print(f"  • Features per sample: {len(train_data.columns)}")
    print(f"  • Date range: {start_date.date()} to {end_date.date()}")
    print(f"  • Tickers: {len(tickers)} ({', '.join(tickers)})")
    print(f"\n")


if __name__ == "__main__":
    main()
