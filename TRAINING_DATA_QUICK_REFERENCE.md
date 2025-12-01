# Training Data Processor - Quick Reference

## Quick Start (30 seconds)

```python
from data_pipeline.core import TrainingDataProcessor

# Create processor
processor = TrainingDataProcessor({'data_root': '/data'})

# Generate training data
training_data = processor.generate_training_data()

# View summary
processor.print_feature_summary(training_data)
```

## Common Tasks

### 1. Generate Training Data (Default: Last 30 Days)

```python
training_data = processor.generate_training_data()
```

### 2. Specific Date Range

```python
from datetime import datetime

start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)

training_data = processor.generate_training_data(
    start_date=start,
    end_date=end
)
```

### 3. Specific Stocks

```python
training_data = processor.generate_training_data(
    tickers=['AAPL', 'MSFT', 'GOOGL']
)
```

### 4. Save to Different Format

```python
# Parquet (recommended for ML)
processor.config['output_format'] = 'parquet'
processor.generate_training_data()

# CSV
processor.config['output_format'] = 'csv'
processor.generate_training_data()

# JSON
processor.config['output_format'] = 'json'
processor.generate_training_data()
```

### 5. Don't Save to Disk

```python
training_data = processor.generate_training_data(save=False)
```

### 6. Get Feature Summary

```python
summary = processor.get_feature_summary(training_data)

print(f"Records: {summary['total_records']}")
print(f"Features: {summary['total_features']}")
print(f"Tickers: {summary['tickers']}")
print(f"Date range: {summary['date_range']}")
```

### 7. Access Individual Data Towers

```python
# Stock data tower (financial + technical indicators)
stock_data = processor.stock_tower.load_data()

# Context data tower (news + macro + policy)
context_data = processor.context_tower.load_data()
```

## Output Data Format

### Columns Available

**Stock Features:**
- `ticker` - Stock symbol
- `timestamp` - Event timestamp
- `price` - Current price
- `open`, `high`, `low`, `close` - OHLC
- `volume` - Trading volume
- `market_cap` - Market capitalization
- `pe_ratio` - P/E ratio
- `dividend_yield` - Dividend yield
- `52_week_high`, `52_week_low` - 52-week range
- `sma_20`, `sma_50` - Moving averages
- `rsi` - Relative strength index
- `macd`, `macd_signal` - MACD indicator

**Context Features:**
- `news_headline` - News headline
- `news_summary` - News summary
- `news_sentiment` - Sentiment polarity/subjectivity
- `macro_interest_rate` - Interest rates
- `macro_unemployment_rate` - Unemployment
- `policy_event_type` - Policy event type
- (+ more macro and policy fields)

### Sample Row

```
ticker: AAPL
timestamp: 2024-06-15 14:30:00
price: 192.53
volume: 45200000
sma_20: 189.43
rsi: 72.45
news_headline: "Apple reports strong Q2 earnings"
news_sentiment: {'polarity': 0.85, 'subjectivity': 0.6}
macro_interest_rate: 5.33
policy_event_type: "FED_ANNOUNCEMENT"
```

## Configuration

```python
config = {
    'data_root': '/data',              # Root data directory
    'output_format': 'parquet',        # csv, parquet, or json
    'output_path': '/data/training',   # Where to save files
}

processor = TrainingDataProcessor(config)
```

## File Locations

| Data Type | Location |
|-----------|----------|
| Raw financial | `/data/raw/financial_data/` |
| Raw movements | `/data/raw/stock_movements/` |
| Raw news | `/data/raw/news/` |
| Raw macro | `/data/raw/macroeconomic_data/` |
| Raw policy | `/data/raw/policy_data/` |
| Training output | `/data/training/` |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Data directory not found" | Run data collection pipeline first |
| Empty dataset | Check date range matches available data |
| Missing columns | Some data sources may be empty; use `.fillna()` |
| Memory errors on large datasets | Process in time chunks (see examples) |
| Slow performance | Use Parquet format instead of CSV |

## Examples

### Example 1: Load and Inspect

```python
processor = TrainingDataProcessor({'data_root': '/data'})
data = processor.generate_training_data(save=False)

print(data.shape)           # (rows, columns)
print(data.info())          # Column types and nulls
print(data.describe())      # Statistics
```

### Example 2: For Scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = processor.generate_training_data()

X = data.drop(['ticker', 'timestamp', 'target'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### Example 3: Time Series (60-day window)

```python
import pandas as pd

data = processor.generate_training_data()

# For each ticker, get 60-day windows
for ticker in data['ticker'].unique():
    ticker_data = data[data['ticker'] == ticker].sort_values('timestamp')
    
    # Create 60-day rolling windows
    for i in range(len(ticker_data) - 60):
        window = ticker_data.iloc[i:i+60]
        target = ticker_data.iloc[i+60]
        
        # Use window as features, target for label
        features = window.drop(['ticker', 'timestamp', 'target'], axis=1)
        label = target['target']
```

### Example 4: Data Quality Check

```python
data = processor.generate_training_data()

# Check completeness
print(f"Completeness: {(1 - data.isnull().sum().sum() / data.size) * 100:.2f}%")

# Check numeric ranges
print(f"Price range: {data['price'].min():.2f} - {data['price'].max():.2f}")

# Check coverage
print(f"Tickers: {data['ticker'].nunique()}")
print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
```

## Performance Notes

| Operation | Time | Memory |
|-----------|------|--------|
| Load 100K records | ~2-3s | ~200MB |
| Join towers | ~1s | ~100MB |
| Save Parquet | ~500ms | inline |
| Save CSV | ~2-3s | inline |

## Next Steps

1. **Generate training data**: Run `processor.generate_training_data()`
2. **Inspect output**: Check files in `/data/training/`
3. **Explore features**: Use `.describe()` and `.info()`
4. **Build models**: Use with scikit-learn, XGBoost, TensorFlow, etc.
5. **Iterate**: Adjust features, date ranges, tickers as needed

## Files

| File | Purpose |
|------|---------|
| `data_pipeline/core/training_data.py` | Main processor implementation |
| `TRAINING_DATA_GUIDE.md` | Comprehensive guide |
| `examples/training_data_examples.py` | Code examples |
| `/data/training/` | Output directory |

## Support

For detailed information, see `TRAINING_DATA_GUIDE.md`

For code examples, see `examples/training_data_examples.py`

For implementation details, see `data_pipeline/core/training_data.py`
