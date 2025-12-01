# Training Data Unification - Refactoring Summary

## Overview
Refactored the training data processor from a **two-tower architecture** to a **unified flattened structure** with intelligent source labeling. This aligns with the weekly data granularity migration completed in Phase 2.

## Key Changes

### Architecture
**Before (Two-Tower Pattern):**
```
StockDataTower → [Financial + Movement Data] ─┐
                                               └→ Join → Training Data
ContextDataTower → [News + Macro + Policy] ──┘
```

**After (Unified Flattened):**
```
Stock Data (Financial + Movement) ──┐
News Data (Sentiment) ──────────────┤
Macro Data (Economic) ──────────────┼→ Join on ticker+timestamp → Flattened Dataset
Policy Data (Announcements) ────────┘
```

### Data Structure
**Old Output (Hierarchical):**
- Separate tower columns
- Mixed naming conventions
- Ambiguous feature origins

**New Output (Flattened with Labels):**
- **stock_*** - All financial and technical indicators
- **news_*** - News sentiment data
- **macro_*** - Macroeconomic indicators
- **policy_*** - Policy announcements
- `ticker`, `timestamp` - Key identifying fields

### Class Changes

#### Removed Classes
- `DataTower` (ABC base class)
- `StockDataTower` (financial + movement data loading)
- `ContextDataTower` (news + macro + policy combining)

#### New Single Class
- `UnifiedTrainingDataProcessor` - Consolidated processor with all functionality

### Key Features

1. **Unified Data Loading**
   - Single `generate_training_data()` method
   - Consistent weekly granularity across all sources
   - Automatic date/ticker filtering

2. **Intelligent Source Labeling**
   - Column prefixes indicate data source
   - Easy feature identification
   - Maintains data lineage

3. **Simplified Joining**
   - Outer join on `(ticker, timestamp)` pairs
   - No complex tower merging logic
   - Consistent handling of market-wide data (ticker='MARKET')

4. **Weekly Movement Calculation**
   - Integrated into main pipeline
   - Results labeled as `stock_weekly_*`
   - Automatic OHLC price detection

5. **Enhanced Reporting**
   - Source-grouped feature summaries
   - Missing value tracking per source
   - Feature count by data source

## Data Loading Pipeline

### Stock Data (`stock_*` prefix)
```
/raw/financial_data (financial metrics)
     ↓
    + /raw/stock_movements (technical indicators)
     ↓
[Merged on ticker+timestamp]
```
**Features:** price, volume, market_cap, pe_ratio, sma_20, rsi, macd, etc.

### News Data (`news_*` prefix)
```
/raw/news (sentiment analysis)
     ↓
[Filtered by tickers]
```
**Features:** sentiment_polarity, subjectivity, headline, source, etc.

### Macro Data (`macro_*` prefix)
```
/context/macroeconomic (economic indicators)
     ↓
[ticker='MARKET' if missing]
```
**Features:** gdp, unemployment, inflation, interest_rate, etc.

### Policy Data (`policy_*` prefix)
```
/context/policy (policy announcements)
     ↓
[ticker='MARKET' if missing]
```
**Features:** announcement_type, fed_decision, fomc_meeting, etc.

## Method Mapping

| Old Method | New Method | Change |
|-----------|-----------|--------|
| `StockDataTower.load_data()` | `_load_stock_data()` | Internal method |
| `ContextDataTower.load_data()` | Individual loaders | Split into 3 methods |
| `_join_towers()` | `_join_all_sources()` | More explicit naming |
| `normalize_schema()` | `_prefix_columns()` | Replaced by labeling |
| N/A | `_load_json_files()` | Shared utility |

## Usage Examples

### Basic Usage (Unchanged API)
```python
from data_pipeline.core.training_data import TrainingDataProcessor

config = {'data_root': '/data'}
processor = TrainingDataProcessor(config)

# Generate training data
df = processor.generate_training_data(
    start_date='2025-09-01',
    end_date='2025-11-30',
    tickers=['AAPL', 'MSFT', 'GOOGL']
)
```

### Access Stock Features
```python
# Get all stock features (automatically prefixed)
stock_cols = [col for col in df.columns if col.startswith('stock_')]
print(df[['ticker', 'timestamp'] + stock_cols].head())
```

### Get Weekly Movement
```python
# Access weekly movement calculations
movement_cols = [col for col in df.columns if 'weekly' in col]
print(df[['ticker', 'timestamp'] + movement_cols].head())

# Output: stock_weekly_open_price, stock_weekly_close_price, stock_weekly_price_delta, etc.
```

### Access Context Features
```python
# Get all macro indicators
macro_cols = [col for col in df.columns if col.startswith('macro_')]
print(df[['ticker'] + macro_cols].head())

# Get all policy data
policy_cols = [col for col in df.columns if col.startswith('policy_')]
print(df[policy_cols].head())
```

### Summary with Source Grouping
```python
processor.print_feature_summary(df)

# Output:
# UNIFIED TRAINING DATA SUMMARY
# ============================================================================
# Total Records: 1,680
# Total Features: 87
# Tickers: AAPL, AMZN, GOOGL, MSFT, META, NVDA, TSLA
# Date Range: 2025-09-01 to 2025-11-30
#
# Features by Source:
#   MACRO: 12 features
#     - macro_cpi
#     - macro_federal_funds_rate
#     ...
#   NEWS: 8 features
#     - news_headline
#     - news_sentiment_polarity
#     ...
#   POLICY: 5 features
#     - policy_announcement_type
#     - policy_fed_decision
#     ...
#   STOCK: 62 features
#     - stock_close
#     - stock_market_cap
#     ...
```

## Backward Compatibility

### ✅ Maintained
- `TrainingDataProcessor` class name (aliased to `UnifiedTrainingDataProcessor`)
- `generate_training_data()` method signature
- `include_weekly_movement` parameter
- Configuration structure
- Output formats (CSV, Parquet, JSON)

### ⚠️ Changed
- Internal implementation (single class vs. two towers)
- Column naming (now with source prefixes)
- Feature locations (organized by source)

### Migration Guide
```python
# Old code - still works (aliased)
processor = TrainingDataProcessor(config)
df = processor.generate_training_data()

# New features available
summary = processor.get_feature_summary(df)
processor.print_feature_summary(df)

# New column access pattern
stock_features = df[[col for col in df.columns if col.startswith('stock_')]]
macro_features = df[[col for col in df.columns if col.startswith('macro_')]]
```

## Data Quality Improvements

### Consistency
- ✅ All data normalized to weekly granularity
- ✅ Unified timestamp handling
- ✅ Consistent null/NaN handling

### Traceability
- ✅ Source prefixes identify data origin
- ✅ Feature grouping for analysis
- ✅ Source-specific missing value tracking

### Performance
- ✅ Single loading pipeline (vs. dual towers)
- ✅ Fewer intermediate DataFrames
- ✅ Optimized join operations

## Testing Checklist

### Unit Tests
- [ ] `_load_stock_data()` returns correct columns
- [ ] `_load_news_data()` filters by tickers
- [ ] `_load_macro_data()` assigns ticker='MARKET'
- [ ] `_load_policy_data()` assigns ticker='MARKET'
- [ ] `_prefix_columns()` correctly adds prefixes
- [ ] `_join_all_sources()` handles empty DataFrames
- [ ] `_add_weekly_movement()` calculates correctly

### Integration Tests
- [ ] Full pipeline generates complete dataset
- [ ] All sources included in output
- [ ] Timestamps aligned across sources
- [ ] Weekly movement features present
- [ ] Save/load roundtrip works

### Validation Tests
- [ ] Row counts match expectations
- [ ] No duplicate rows
- [ ] Ticker filtering works
- [ ] Date range filtering works
- [ ] Missing values handled gracefully

## Files Modified
- `data_pipeline/core/training_data.py` (Complete refactor)
  - Before: 734 lines (two-tower architecture)
  - After: ~550 lines (unified architecture)

## Configuration Files
No changes required to:
- `data_pipeline/config/config_manager.py`
- `data_pipeline/core/pipeline_scheduler.py`

## Benefits

1. **Simplified Maintenance**
   - Single processor class instead of three
   - Clearer data flow
   - Easier to extend with new sources

2. **Better Feature Discovery**
   - Source labels make features self-documenting
   - Easy to group/filter by source
   - Clear data provenance

3. **Consistent Weekly Granularity**
   - All data normalized to weekly schedule
   - Aligns with Phase 2 collection changes
   - Predictable data structure

4. **Enhanced ML Readiness**
   - Flattened structure ideal for sklearn/TensorFlow
   - Labeled features aid model interpretability
   - Grouped features help feature selection

## Next Steps

1. Update test suite to reflect new class structure
2. Update examples to demonstrate new source labeling
3. Update documentation to show unified approach
4. Performance testing with full weekly dataset
5. Integration testing with actual collection pipeline
