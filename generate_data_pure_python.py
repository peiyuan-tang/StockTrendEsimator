#!/usr/bin/env python3
"""
Generate training and validation data for Stock Trend Estimator
Pure Python implementation - No external dependencies required
Date range: 11/01 to 11/30
Split ratio: 2:1 (training:validation)
"""

import sys
import os
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_weekly_dates(start_date, end_date):
    """Generate weekly dates from start to end"""
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=7)
    return dates


def generate_sample_row(date, ticker, index):
    """Generate a single sample row of data"""
    import random
    random.seed(42 + index)  # Reproducible results
    
    base_price = 100 + random.uniform(0, 400)
    
    return {
        'date': date.strftime('%Y-%m-%d'),
        'timestamp': date.isoformat(),
        'ticker': ticker,
        # Stock data
        'stock_open': round(base_price, 2),
        'stock_high': round(base_price * random.uniform(1.0, 1.05), 2),
        'stock_low': round(base_price * random.uniform(0.95, 1.0), 2),
        'stock_close': round(base_price * random.uniform(0.95, 1.05), 2),
        'stock_volume': round(random.uniform(1e7, 1e8)),
        'stock_sma_20': round(base_price * random.uniform(0.98, 1.02), 2),
        'stock_sma_50': round(base_price * random.uniform(0.97, 1.03), 2),
        'stock_rsi': round(random.uniform(30, 70), 2),
        'stock_macd': round(random.uniform(-5, 5), 2),
        'stock_bb_upper': round(base_price * 1.02, 2),
        'stock_bb_lower': round(base_price * 0.98, 2),
        # News data
        'news_sentiment': round(random.uniform(-1, 1), 3),
        'news_count': random.randint(0, 20),
        'news_relevance': round(random.uniform(0, 1), 3),
        # Macro data
        'macro_gdp_growth': round(random.uniform(-2, 5), 2),
        'macro_inflation': round(random.uniform(1, 5), 2),
        'macro_unemployment': round(random.uniform(3, 8), 2),
        'macro_interest_rate': round(random.uniform(0, 5), 2),
        'macro_vix': round(random.uniform(10, 40), 2),
        # Policy data
        'policy_event_count': random.randint(0, 5),
        'policy_impact_score': round(random.uniform(-1, 1), 3),
        # Target
        'stock_weekly_return': round(random.uniform(-0.1, 0.1), 4),
        'stock_weekly_movement': random.choice([0, 1]),
        'stock_trend_direction': random.choice([-1, 0, 1]),
    }


def main():
    """Generate training and validation data"""
    
    print("\n" + "="*70)
    print("🚀 STOCK TREND ESTIMATOR - DATA PIPELINE")
    print("="*70)
    
    # Configuration
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(project_root, 'data_output')
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Date range: 11/01 to 11/30 (November)
    start_date = datetime(2024, 11, 1)
    end_date = datetime(2024, 11, 30)
    
    # Tickers (Magnificent 7)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    
    print(f"\n📊 Configuration:")
    print(f"   Date Range: {start_date.date()} to {end_date.date()}")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Output Format: CSV")
    print(f"   Output Path: {output_path}")
    print(f"   Train/Validation Split: 2:1 (66% / 33%)")
    
    print(f"\n⏳ Generating synthetic training data...")
    
    # Generate dates
    dates = generate_weekly_dates(start_date, end_date)
    print(f"   Generated {len(dates)} weekly periods")
    
    # Generate all data
    all_data = []
    index = 0
    for date in dates:
        for ticker in tickers:
            row = generate_sample_row(date, ticker, index)
            all_data.append(row)
            index += 1
    
    print(f"   ✅ Generated {len(all_data)} rows, {len(all_data[0])} columns")
    
    # Shuffle data (simple shuffle using built-in random)
    import random
    random.seed(42)
    random.shuffle(all_data)
    
    # Split into training and validation (2:1 ratio)
    print(f"\n📈 Splitting data (2:1 ratio)...")
    split_idx = int(len(all_data) * 2 / 3)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"   Training samples: {len(train_data)} ({len(train_data)/len(all_data)*100:.1f}%)")
    print(f"   Validation samples: {len(val_data)} ({len(val_data)/len(all_data)*100:.1f}%)")
    
    # Save as CSV
    print(f"\n💾 Saving datasets...")
    
    train_csv_path = os.path.join(output_path, 'training_data_20241101-20241130.csv')
    val_csv_path = os.path.join(output_path, 'validation_data_20241101-20241130.csv')
    
    # Save training data
    with open(train_csv_path, 'w', newline='') as f:
        if train_data:
            writer = csv.DictWriter(f, fieldnames=train_data[0].keys())
            writer.writeheader()
            writer.writerows(train_data)
    print(f"   ✅ Training CSV: {train_csv_path}")
    
    # Save validation data
    with open(val_csv_path, 'w', newline='') as f:
        if val_data:
            writer = csv.DictWriter(f, fieldnames=val_data[0].keys())
            writer.writeheader()
            writer.writerows(val_data)
    print(f"   ✅ Validation CSV: {val_csv_path}")
    
    # Save as JSON for reference
    train_json_path = os.path.join(output_path, 'training_data_20241101-20241130.json')
    val_json_path = os.path.join(output_path, 'validation_data_20241101-20241130.json')
    
    with open(train_json_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"   ✅ Training JSON: {train_json_path}")
    
    with open(val_json_path, 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"   ✅ Validation JSON: {val_json_path}")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"\n   Training Data:")
    print(f"      Rows: {len(train_data)}")
    print(f"      Columns: {len(train_data[0]) if train_data else 0}")
    print(f"      File size: {os.path.getsize(train_csv_path) / 1024:.2f} KB")
    
    print(f"\n   Validation Data:")
    print(f"      Rows: {len(val_data)}")
    print(f"      Columns: {len(val_data[0]) if val_data else 0}")
    print(f"      File size: {os.path.getsize(val_csv_path) / 1024:.2f} KB")
    
    # Show column names
    if train_data:
        columns = list(train_data[0].keys())
        print(f"\n📋 Features ({len(columns)} total):")
        for i, col in enumerate(columns, 1):
            print(f"      {i:2d}. {col}")
    
    # Show sample rows
    print(f"\n📄 Sample Training Data (first 3 rows):")
    if train_data:
        for i, row in enumerate(train_data[:3], 1):
            print(f"\n   Row {i}:")
            for key, value in list(row.items())[:8]:
                print(f"      {key}: {value}")
            print(f"      ... ({len(row) - 8} more fields)")
    
    print(f"\n" + "="*70)
    print(f"✅ SUCCESS - Data pipeline completed!")
    print(f"="*70)
    print(f"\nOutput Files:")
    print(f"  📄 Training (CSV): {train_csv_path}")
    print(f"  📄 Validation (CSV): {val_csv_path}")
    print(f"  📊 Training (JSON): {train_json_path}")
    print(f"  📊 Validation (JSON): {val_json_path}")
    
    print(f"\nData Statistics:")
    print(f"  • Training samples: {len(train_data)} (66%)")
    print(f"  • Validation samples: {len(val_data)} (33%)")
    print(f"  • Total samples: {len(all_data)}")
    print(f"  • Features per sample: {len(all_data[0]) if all_data else 0}")
    print(f"  • Date range: {start_date.date()} to {end_date.date()}")
    print(f"  • Tickers: {len(tickers)} ({', '.join(tickers)})")
    print(f"\nNext Steps:")
    print(f"  1. Load with Python: import json; data = json.load(open('{val_csv_path}'))")
    print(f"  2. Load with Pandas: pd.read_csv('{train_csv_path}')")
    print(f"  3. Use for model training: from data import train_data")
    print(f"\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
