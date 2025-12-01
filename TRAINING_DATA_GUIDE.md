# Training Data Processor Guide

## Overview

The `TrainingDataProcessor` unifies and joins data from all pipeline sources into a single training dataset for ML models. It orchestrates two data towers that work together to create comprehensive feature sets.

## Architecture

### Two-Tower Design

```
┌─────────────────────────────────────────────────────┐
│         Training Data Processor                      │
└─────────────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
        ▼                               ▼
┌──────────────────────┐      ┌──────────────────────┐
│  Stock Data Tower    │      │  Context Data Tower  │
│  ══════════════════  │      │  ═════════════════   │
│ • Financial Data     │      │ • News Sentiment     │
│ • Stock Movements    │      │ • Macro Indicators   │
│ • Tech Indicators    │      │ • Policy Events      │
└──────────────────────┘      └──────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
                    [JOIN]
                        │
                        ▼
            ┌───────────────────────┐
            │  Training Dataset     │
            │ (Multi-feature, time- │
            │  aligned records)     │
            └───────────────────────┘
```

## Stock Data Tower

Combines **financial data** and **stock movement** data into a unified stock feature set.

### Data Sources

1. **Financial Data**
   - Ticker: Stock symbol (AAPL, MSFT, etc.)
   - Price: Current stock price
   - OHLC: Open, High, Low, Close prices
   - Volume: Trading volume
   - Market Cap: Company market capitalization
   - P/E Ratio: Price-to-earnings ratio
   - Dividend Yield: Annual dividend yield
   - 52-Week Range: High/low prices for year

2. **Stock Movements (Technical Indicators)**
   - SMA 20/50: Simple moving averages
   - RSI: Relative strength index
   - MACD: Moving average convergence divergence

### Schema (After Normalization)

```python
{
    'ticker': str,              # Stock symbol
    'timestamp': datetime,      # Event timestamp
    'price': float,             # Current price
    'open': float,              # Open price
    'high': float,              # High price
    'low': float,               # Low price
    'close': float,             # Close price
    'volume': float,            # Trading volume
    'market_cap': float,        # Market capitalization
    'pe_ratio': float,          # P/E ratio
    'dividend_yield': float,    # Dividend yield
    '52_week_high': float,      # 52-week high
    '52_week_low': float,       # 52-week low
    'sma_20': float,            # 20-day SMA
    'sma_50': float,            # 50-day SMA
    'rsi': float,               # RSI indicator
    'macd': float,              # MACD value
    'macd_signal': float,       # MACD signal line
}
```

## Context Data Tower

Combines **news**, **macroeconomic**, and **policy** data into contextual features.

### Data Sources

1. **News Data**
   - Headlines and summaries
   - Sentiment analysis (polarity, subjectivity)
   - Source and publication timestamp
   - Related ticker(s)

2. **Macroeconomic Data**
   - Interest rates
   - Unemployment rate
   - GDP growth
   - Inflation rate
   - Fed funds rate

3. **Policy Data**
   - Federal announcements
   - Policy event types
   - Impact levels
   - Metadata

### Schema (After Normalization)

```python
{
    'ticker': str,              # Stock symbol or 'MARKET'
    'timestamp': datetime,      # Event timestamp
    
    # News columns (prefixed with 'news_')
    'news_headline': str,       # News headline
    'news_summary': str,        # News summary
    'news_sentiment': dict,     # {'polarity': float, 'subjectivity': float}
    'news_source': str,         # News source
    'news_url': str,            # Article URL
    
    # Macro columns (prefixed with 'macro_')
    'macro_interest_rate': float,
    'macro_unemployment_rate': float,
    'macro_gdp_growth': float,
    'macro_inflation_rate': float,
    'macro_fed_funds_rate': float,
    
    # Policy columns (prefixed with 'policy_')
    'policy_event_type': str,
    'policy_title': str,
    'policy_impact_level': str,
}
```

## Usage Guide

### Basic Usage

```python
from data_pipeline.core import TrainingDataProcessor
from datetime import datetime, timedelta

# Initialize processor
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
```

### Advanced Usage

```python
from datetime import datetime

# Custom date range and tickers
start = datetime(2024, 1, 1)
end = datetime(2024, 12, 31)
tickers = ['AAPL', 'MSFT', 'GOOGL']

training_data = processor.generate_training_data(
    start_date=start,
    end_date=end,
    tickers=tickers,
    save=True
)

# Access individual towers
stock_data = processor.stock_tower.load_data(start, end, tickers)
context_data = processor.context_tower.load_data(start, end, tickers)
```

### Output Formats

```python
# Parquet (recommended for ML - compressed, preserves types)
config = {'output_format': 'parquet'}

# CSV (human-readable, slower I/O)
config = {'output_format': 'csv'}

# JSON (portable, flexible)
config = {'output_format': 'json'}
```

### Accessing Results

```python
# Get summary statistics
summary = processor.get_feature_summary(training_data)

print(f"Records: {summary['total_records']}")
print(f"Features: {summary['total_features']}")
print(f"Date range: {summary['date_range']}")
print(f"Missing values: {summary['missing_values']}")
```

## Data Joining Strategy

### Stock Data Tower Join

- **Primary Key**: ticker + timestamp
- **Strategy**: Merge-asof on timestamp (allows 60-minute tolerance)
- **Result**: Combines financial and technical data per ticker per time

### Context Data Tower Join

- **Strategy**: Outer join to preserve all news, macro, and policy events
- **Market-Level Data**: Macro and policy data assigned ticker='MARKET'
- **Result**: Enriched context for each ticker and time

### Final Join

- **Primary Key**: ticker + timestamp
- **Strategy**: Outer join between stock and context towers
- **Result**: All available features aligned by ticker and time

## Feature Engineering

The processor supports post-processing for ML models:

```python
import pandas as pd

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for ML models"""
    df = df.copy()
    
    # Price changes and returns
    df['price_change'] = df.groupby('ticker')['price'].diff()
    df['daily_return'] = df.groupby('ticker')['price'].pct_change()
    
    # Volatility
    df['volatility'] = df.groupby('ticker')['daily_return'].rolling(20).std()
    
    # Volume signal
    df['volume_sma_ratio'] = df['volume'] / df.groupby('ticker')['volume'].rolling(20).mean()
    
    # Sentiment features
    df['news_sentiment_avg'] = df.get('news_sentiment', {}).apply(lambda x: x.get('polarity', 0))
    
    return df

training_data = processor.generate_training_data()
training_data = add_engineered_features(training_data)
```

## Performance Considerations

### Data Loading

- **Parallel Loading**: Multiple data towers load independently
- **Directory Scanning**: Efficient file discovery and filtering
- **JSON Parsing**: Streaming for large files

### Memory Management

- **Chunk Processing**: Handle large datasets via pandas operations
- **Type Conversion**: Automatic dtype optimization
- **Missing Data**: Handled gracefully with NULL values

### Output Optimization

- **Parquet**: 60-80% size reduction vs CSV
- **Compression**: Built-in compression support
- **Indexing**: Timestamp index recommended for time-series analysis

### Example Performance Metrics

```
100K records across 7 tickers (30 days):
- Load time: ~2-3 seconds
- Join time: ~1 second
- Save (Parquet): ~500ms
- Total: ~4 seconds
```

## Troubleshooting

### Issue: "Data directory not found"

**Solution**: Ensure data exists in expected structure:
```
/data/
├── raw/
│   ├── financial_data/
│   ├── stock_movements/
│   ├── news/
│   ├── macroeconomic_data/
│   └── policy_data/
└── training/
```

### Issue: Empty training data

**Cause**: Data sources not populated or date range mismatches

**Solution**: Verify data was collected and check date filters:
```python
# Check what data exists
stock_data = processor.stock_tower.load_data()
print(f"Stock records: {len(stock_data)}")

context_data = processor.context_tower.load_data()
print(f"Context records: {len(context_data)}")
```

### Issue: Missing columns in output

**Cause**: Source data missing expected fields

**Solution**: Check normalization - processor adds missing columns as NULL:
```python
# Find missing values
print(training_data.isnull().sum())

# Drop or fill as needed
training_data = training_data.fillna(method='ffill')
```

### Issue: Memory errors on large datasets

**Solution**: Process in date chunks:
```python
import pandas as pd
from datetime import datetime, timedelta

chunk_start = datetime(2024, 1, 1)
all_data = []

for i in range(12):  # 12 months
    chunk_end = chunk_start + timedelta(days=30)
    chunk_data = processor.generate_training_data(
        start_date=chunk_start,
        end_date=chunk_end,
        save=False  # Don't save each chunk
    )
    all_data.append(chunk_data)
    chunk_start = chunk_end

# Combine all chunks
training_data = pd.concat(all_data, ignore_index=True)
```

## Integration with ML Pipelines

### Scikit-learn

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

training_data = processor.generate_training_data()

# Prepare features and target
X = training_data.drop(['ticker', 'timestamp', 'target'], axis=1)
y = training_data['target']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

### XGBoost

```python
import xgboost as xgb

# Create DMatrix from training data
training_data = processor.generate_training_data()
dtrain = xgb.DMatrix(
    training_data.drop(['ticker', 'timestamp'], axis=1),
    label=training_data['target']
)

# Train model
model = xgb.train({}, dtrain)
```

### TensorFlow/Keras

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

training_data = processor.generate_training_data()

# Create sequences for LSTM
def create_sequences(data, seq_length=60):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length, -1])
    return np.array(sequences), np.array(targets)

X, y = create_sequences(training_data.values)
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1)
])
```

## Next Steps

1. **Run Training Data Generation**:
   ```bash
   python -c "from data_pipeline.core import TrainingDataProcessor; \
   p = TrainingDataProcessor({}); \
   d = p.generate_training_data(); \
   p.print_feature_summary(d)"
   ```

2. **Inspect Output Files**:
   ```bash
   ls -lh /data/training/
   ```

3. **Load and Explore**:
   ```python
   import pandas as pd
   df = pd.read_parquet('/data/training/training_data_*.parquet')
   print(df.describe())
   ```

4. **Build ML Models**: Use the training data with your preferred ML framework

## See Also

- `data_pipeline/core/pipeline_scheduler.py` - Automate data collection
- `data_pipeline/storage/data_sink.py` - Configure output formats
- `data_pipeline/sources/` - Data source implementations
- `PROTOBUF_GUIDE.md` - Protobuf serialization options
