# Training Data Unification - Migration Guide

## Quick Summary
The training data processor has been refactored from a **two-tower architecture** to a **unified flattened design** with intelligent source labeling. All functionality is preserved and backward compatible.

## What Changed?

### Architecture
- **Before:** `StockDataTower` + `ContextDataTower` (two separate classes)
- **After:** `UnifiedTrainingDataProcessor` (single class)

### Data Structure
- **Before:** Mixed column naming across sources
- **After:** Standardized prefixes: `stock_*`, `news_*`, `macro_*`, `policy_*`

### Weekly Granularity
- ✅ All data normalized to weekly schedule (from Phase 2 changes)
- ✅ Consistent timestamp handling across sources
- ✅ Automatic OHLC detection for price calculations

## For End Users

### No Changes Required!
Your existing code continues to work:

```python
# This still works exactly as before
processor = TrainingDataProcessor(config)
df = processor.generate_training_data()
```

### What's New (Optional to Use)

#### 1. Source-Labeled Features
Features now have clear source prefixes:

```python
# OLD: Had to guess which columns were what
# NEW: Clear identification
stock_cols = [col for col in df.columns if col.startswith('stock_')]
news_cols = [col for col in df.columns if col.startswith('news_')]
macro_cols = [col for col in df.columns if col.startswith('macro_')]
policy_cols = [col for col in df.columns if col.startswith('policy_')]
```

#### 2. Enhanced Summary
Get organized feature information:

```python
processor.print_feature_summary(df)
# Shows features grouped by source
# Displays missing values per source
# More readable format
```

#### 3. Weekly Movement Features
Still labeled with `stock_` prefix:

```python
df = processor.generate_training_data(include_weekly_movement=True)

# Access weekly movements
weekly_open = df['stock_weekly_open_price']
weekly_delta = df['stock_weekly_price_delta']
weekly_return = df['stock_weekly_price_return']
```

## For Developers

### New Structure

#### Single Entry Point
```python
from data_pipeline.core.training_data import TrainingDataProcessor

processor = TrainingDataProcessor(config)
df = processor.generate_training_data(
    start_date='2025-09-01',
    end_date='2025-11-30',
    tickers=['AAPL', 'MSFT'],
    include_weekly_movement=True
)
```

#### Internal Methods (Private)
```python
processor._load_stock_data()      # Financial + technical indicators
processor._load_news_data()       # News sentiment
processor._load_macro_data()      # Economic indicators
processor._load_policy_data()     # Policy announcements

processor._join_all_sources()     # Unified join logic
processor._prefix_columns()       # Smart column labeling
processor._add_weekly_movement()  # Weekly calculations
```

#### Public Methods
```python
processor.generate_training_data()  # Main generation pipeline
processor.get_feature_summary()     # Detailed feature info
processor.print_feature_summary()   # Formatted output
```

### Testing Integration
```python
# Test data is loaded correctly
def test_load_stock_data():
    df = processor._load_stock_data(start, end, tickers)
    assert 'stock_price' in df.columns or 'stock_volume' in df.columns

# Test features are labeled
def test_column_labeling():
    df = processor.generate_training_data()
    stock_cols = [c for c in df.columns if c.startswith('stock_')]
    assert len(stock_cols) > 0

# Test joining works
def test_unified_join():
    df = processor.generate_training_data()
    assert 'ticker' in df.columns
    assert 'timestamp' in df.columns
    assert len(df) > 0
```

## Data Quality Improvements

### Consistency
✅ All sources on same weekly schedule
✅ Unified timestamp handling
✅ Consistent null value strategy (outer joins)

### Traceability
✅ Column prefixes show data origin
✅ Source-grouped feature discovery
✅ Per-source missing value tracking

### Performance
✅ Single loading pipeline (vs. dual)
✅ Fewer intermediate DataFrames
✅ Optimized join operations

## Backward Compatibility Checklist

| Item | Status | Details |
|------|--------|---------|
| Class name | ✅ | `TrainingDataProcessor` still available |
| Method signature | ✅ | `generate_training_data()` unchanged |
| Configuration | ✅ | Same config dict structure |
| Output formats | ✅ | CSV, Parquet, JSON all work |
| Default parameters | ✅ | 12 weeks, Mag 7 tickers |
| Return type | ✅ | Still returns pandas DataFrame |
| Weekly movement | ✅ | Still calculated automatically |

## Configuration

### Basic Config (Unchanged)
```python
config = {
    'data_root': '/data',
    'output_format': 'parquet',  # csv, parquet, or json
    'output_path': '/data/training'
}
```

### Data Directories (Unchanged)
```
/data/
├── raw/
│   ├── financial_data/
│   ├── stock_movements/
│   └── news/
└── context/
    ├── macroeconomic/
    └── policy/
```

## Common Use Cases

### Case 1: Basic ML Training
```python
processor = TrainingDataProcessor({'data_root': '/data'})
df = processor.generate_training_data(
    tickers=['AAPL', 'MSFT', 'GOOGL']
)

# Drop keys, fill NaN, train model
features = df.drop(columns=['ticker', 'timestamp']).fillna(df.mean())
model.fit(features, y)
```

### Case 2: Feature Engineering by Source
```python
df = processor.generate_training_data()

# Get stock-only features
stock_features = [c for c in df.columns if c.startswith('stock_')]
X_stock = df[stock_features].fillna(df[stock_features].mean())

# Get context features
context_features = [c for c in df.columns 
                   if c.startswith(('news_', 'macro_', 'policy_'))]
X_context = df[context_features].fillna(df[context_features].mean())

# Train separate models or combine
model_stock = train(X_stock)
model_context = train(X_context)
```

### Case 3: Weekly Analysis
```python
df = processor.generate_training_data(include_weekly_movement=True)

# Group by ticker and week
df['week'] = df['timestamp'].dt.isocalendar().week
weekly_stats = df.groupby(['ticker', 'week']).agg({
    'stock_weekly_price_delta': 'mean',
    'stock_weekly_price_return': 'mean',
    'news_sentiment_polarity': 'mean'
})
```

### Case 4: Time Series Cross-Validation
```python
df = processor.generate_training_data()
df = df.sort_values(['ticker', 'timestamp'])

# Time-based split
split_date = df['timestamp'].quantile(0.8)
train = df[df['timestamp'] < split_date]
test = df[df['timestamp'] >= split_date]
```

## Troubleshooting

### Issue: "Column not found"
**Solution:** Check column prefix. Use feature discovery:
```python
all_cols = sorted(df.columns)
stock_cols = [c for c in all_cols if c.startswith('stock_')]
print(f"Available stock columns: {stock_cols}")
```

### Issue: Missing weekly movement data
**Solution:** Ensure `include_weekly_movement=True` and data has price columns:
```python
df = processor.generate_training_data(include_weekly_movement=True)
movement_cols = [c for c in df.columns if 'weekly' in c]
print(f"Movement columns: {movement_cols}")
print(f"Missing values: {df[movement_cols].isnull().sum()}")
```

### Issue: Too much missing data from one source
**Solution:** Use feature summary to identify problem sources:
```python
summary = processor.get_feature_summary(df)
for source, missing in summary['missing_values_by_source'].items():
    missing_count = sum(missing.values())
    print(f"{source}: {missing_count} missing values")
```

## References

- **Unification Document:** `TRAINING_DATA_UNIFICATION.md`
- **Bug Fixes Document:** `TRAINING_DATA_BUG_FIXES.md`
- **Examples:** `examples/training_data_examples.py`
- **Source Code:** `data_pipeline/core/training_data.py`

## Support

For issues or questions:
1. Check examples in `examples/training_data_examples.py`
2. Review feature summary with `processor.print_feature_summary(df)`
3. Verify data exists in expected directories under `/data/`
4. Check logs for detailed error messages
