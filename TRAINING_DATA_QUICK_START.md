# Training Data Unification - Quick Reference Card

## 🚀 Quick Start

```python
from data_pipeline.core.training_data import TrainingDataProcessor

# Initialize
config = {'data_root': '/data'}
processor = TrainingDataProcessor(config)

# Generate training data
df = processor.generate_training_data()

# Done! Your data is ready
print(df.shape)  # (240, 87)
```

## 📊 Data Structure

```
Rows: 240 (7 tickers × ~34 observations per ticker)
Columns: 87 (2 key + 85 features)

Key Fields:
├─ ticker          ← Stock symbol (AAPL, MSFT, etc)
└─ timestamp       ← Weekly timestamp

Feature Groups:
├─ stock_*         ← 62 financial/technical features
├─ news_*          ← 8 sentiment features
├─ macro_*         ← 12 economic features
└─ policy_*        ← 5 policy announcement features
```

## 🔧 Common Operations

### Access by Source
```python
stock_cols = [c for c in df.columns if c.startswith('stock_')]
news_cols = [c for c in df.columns if c.startswith('news_')]
macro_cols = [c for c in df.columns if c.startswith('macro_')]
policy_cols = [c for c in df.columns if c.startswith('policy_')]

# Use them
X_stock = df[stock_cols]
X_news = df[news_cols]
```

### Filter by Ticker
```python
aapl_data = df[df['ticker'] == 'AAPL']
tech_tickers = ['AAPL', 'MSFT', 'GOOGL']
tech_data = df[df['ticker'].isin(tech_tickers)]
```

### Filter by Date
```python
recent = df[df['timestamp'] >= '2025-11-01']
by_week = df.groupby(df['timestamp'].dt.isocalendar().week)
```

### Get Feature Summary
```python
summary = processor.get_feature_summary(df)
# Returns: sources, feature_count, missing_values, etc

processor.print_feature_summary(df)
# Pretty-printed output
```

## 🎯 Parameters

### generate_training_data()
```python
df = processor.generate_training_data(
    start_date='2025-09-01',           # Optional (default: 12 weeks ago)
    end_date='2025-11-30',             # Optional (default: today)
    tickers=['AAPL', 'MSFT'],          # Optional (default: Mag 7)
    save=True,                         # Optional (default: True)
    include_weekly_movement=True       # Optional (default: True)
)
```

### Configuration
```python
config = {
    'data_root': '/data',                     # Data directory
    'output_format': 'parquet',               # csv, parquet, or json
    'output_path': '/data/training'           # Output directory
}
processor = TrainingDataProcessor(config)
```

## 📈 Feature Categories

### Stock Features (stock_*)
Financial metrics and technical indicators

Common columns:
- `stock_open`, `stock_high`, `stock_low`, `stock_close`
- `stock_volume`, `stock_market_cap`
- `stock_pe_ratio`, `stock_dividend_yield`
- `stock_sma_20`, `stock_sma_50`, `stock_rsi`, `stock_macd`
- `stock_weekly_open_price`, `stock_weekly_close_price`
- `stock_weekly_price_delta`, `stock_weekly_price_return`

### News Features (news_*)
Market sentiment and news data

Common columns:
- `news_sentiment_polarity`
- `news_subjectivity`
- `news_headline`
- `news_source`

### Macro Features (macro_*)
Macroeconomic indicators

Common columns:
- `macro_gdp`
- `macro_unemployment_rate`
- `macro_inflation_rate`
- `macro_interest_rate`
- `macro_consumer_confidence`

### Policy Features (policy_*)
Policy announcements and decisions

Common columns:
- `policy_announcement_type`
- `policy_fed_decision`
- `policy_rate_change`

## 💾 Output Formats

```python
# CSV
config = {'output_format': 'csv'}
processor = TrainingDataProcessor(config)
df = processor.generate_training_data()
# Saved to: /data/training/training_data_unified_YYYYMMDD_HHMMSS.csv

# Parquet (Recommended for ML)
config = {'output_format': 'parquet'}
processor = TrainingDataProcessor(config)
df = processor.generate_training_data()
# Saved to: /data/training/training_data_unified_YYYYMMDD_HHMMSS.parquet

# JSON
config = {'output_format': 'json'}
processor = TrainingDataProcessor(config)
df = processor.generate_training_data()
# Saved to: /data/training/training_data_unified_YYYYMMDD_HHMMSS.json
```

## 🔍 Data Quality Checks

```python
# Check missing values
df.isnull().sum()

# Check by source
for source in ['stock', 'news', 'macro', 'policy']:
    cols = [c for c in df.columns if c.startswith(source + '_')]
    missing = df[cols].isnull().sum().sum()
    print(f"{source}: {missing} missing values")

# Check data types
df.dtypes

# Check date range
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Check tickers
print(f"Tickers: {sorted(df['ticker'].unique())}")
```

## 🚨 Troubleshooting

### Q: No data returned
**A:** Check data exists in directories:
```python
import os
print(os.path.exists('/data/raw/financial_data'))     # Should be True
print(os.path.exists('/data/context/macroeconomic'))  # Should be True
```

### Q: Too many missing values
**A:** Check by source:
```python
summary = processor.get_feature_summary(df)
for source, missing in summary['missing_values_by_source'].items():
    print(f"{source}: {sum(missing.values())} missing")
```

### Q: Expected columns not found
**A:** Check available columns:
```python
print(sorted(df.columns))
# Or filter by source:
print([c for c in df.columns if c.startswith('stock_')])
```

### Q: Wrong date range
**A:** Specify explicitly:
```python
from datetime import datetime, timedelta
end = datetime(2025, 11, 30)
start = datetime(2025, 9, 1)
df = processor.generate_training_data(start_date=start, end_date=end)
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| TRAINING_DATA_UNIFICATION.md | Detailed architecture |
| TRAINING_DATA_MIGRATION_GUIDE.md | User/developer guide |
| TRAINING_DATA_COMPLETION_SUMMARY.md | Project overview |
| TRAINING_DATA_VISUAL_SUMMARY.md | Diagrams & flow |
| examples/training_data_examples.py | 8 working examples |

## ⚙️ Configuration

### Data Directories
```
/data/
├── raw/
│   ├── financial_data/          (Monday 09:00)
│   ├── stock_movements/         (Tuesday 10:00)
│   └── news/                    (Wednesday 11:00)
└── context/
    ├── macroeconomic/           (Thursday 14:00)
    └── policy/                  (Friday 15:30)
```

### Collection Schedule
```
Monday 09:00   → Financial Data
Tuesday 10:00  → Stock Movement
Wednesday 11:00 → News
Thursday 14:00 → Macro Economics
Friday 15:30   → Policy Data
```

## 🔄 Data Flow

```
Raw Data Sources (Weekly)
        ↓
Load individual sources
        ↓
Add source prefixes (stock_, news_, etc)
        ↓
Join on (ticker, timestamp)
        ↓
Calculate weekly movements
        ↓
Flattened Training Dataset
        ↓
Save (CSV/Parquet/JSON)
```

## 💡 Tips & Tricks

### 1. Feature Selection by Source
```python
# Use only stock features
X_stock = df[[c for c in df.columns if c.startswith('stock_')]]
model.fit(X_stock)

# Use stock + macro
X_combo = df[[c for c in df.columns if c.startswith(('stock_', 'macro_'))]]
model.fit(X_combo)
```

### 2. Handle Missing Values
```python
# Fill with mean
df_filled = df.fillna(df.mean())

# Fill by source
for source in ['stock', 'news', 'macro', 'policy']:
    cols = [c for c in df.columns if c.startswith(source + '_')]
    df[cols] = df[cols].fillna(df[cols].mean())
```

### 3. Normalize Features
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features = df.drop(columns=['ticker', 'timestamp'])
df_normalized = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns
)
```

### 4. Train/Test Split by Ticker
```python
train_tickers = ['AAPL', 'MSFT', 'GOOGL']
test_tickers = ['AMZN', 'NVDA']

train = df[df['ticker'].isin(train_tickers)]
test = df[df['ticker'].isin(test_tickers)]
```

### 5. Time Series Split
```python
split_point = df['timestamp'].quantile(0.8)
train = df[df['timestamp'] <= split_point]
test = df[df['timestamp'] > split_point]
```

## 🎓 Examples

See `examples/training_data_examples.py` for:
1. Basic generation
2. Custom parameters
3. Source-specific access
4. Weekly movement analysis
5. Feature summary
6. Multiple export formats
7. ML pipeline preparation
8. Time series analysis

## 📞 Getting Help

1. Check feature summary: `processor.print_feature_summary(df)`
2. Review examples: `examples/training_data_examples.py`
3. Check columns: `print(sorted(df.columns))`
4. Validate data: `print(df.info())` and `print(df.describe())`
5. Check documentation: See documentation files above

---

**Unified Training Data Processor - Ready for ML! ✅**
