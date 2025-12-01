# Training Data Processor - Architecture & Data Flow Diagrams

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING DATA PROCESSOR                             │
│                         (data_pipeline/core/training_data.py)              │
└────────────────────────────────────────────────────────────────────────────┘
                            ▲                             ▲
                            │                             │
                            │                             │
                ┌───────────────────┐       ┌─────────────────────┐
                │ StockDataTower    │       │ ContextDataTower    │
                ├───────────────────┤       ├─────────────────────┤
                │                   │       │                     │
                │ Financial Data:   │       │ News Data:          │
                │ • ticker          │       │ • headline          │
                │ • price           │       │ • sentiment         │
                │ • OHLC            │       │                     │
                │ • volume          │       │ Macro Data:         │
                │ • market_cap      │       │ • interest_rate     │
                │ • PE, dividend    │       │ • unemployment      │
                │ • 52-week         │       │ • GDP, inflation    │
                │                   │       │                     │
                │ +                 │       │ Policy Data:        │
                │                   │       │ • event_type        │
                │ Stock Movements:  │       │ • title             │
                │ • SMA_20          │       │ • impact_level      │
                │ • SMA_50          │       │                     │
                │ • RSI             │       │                     │
                │ • MACD            │       │                     │
                │                   │       │                     │
                └─────────┬─────────┘       └──────────┬──────────┘
                          │                           │
                          │ Merge on                  │ Combine &
                          │ ticker +                 │ prefix
                          │ timestamp                │ (news_,
                          │ (±60min)                 │  macro_,
                          │                          │  policy_)
                          │                          │
                          ▼                          ▼
                   ┌──────────────┐         ┌──────────────┐
                   │  Stock Data  │         │ Context Data │
                   │   (merged)   │         │  (combined)  │
                   └──────────┬───┘         └───────┬──────┘
                              │                    │
                              │                    │
                              └────────┬───────────┘
                                       │
                                       │ Outer join on
                                       │ ticker +
                                       │ timestamp
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │  UNIFIED TRAINING    │
                            │      DATASET         │
                            ├──────────────────────┤
                            │ • 30+ Features       │
                            │ • Time-aligned       │
                            │ • Multi-dimensional  │
                            │ • ML-ready           │
                            └──────────────────────┘
                                       │
                    ┌──────────┬────────┼────────┬─────────┐
                    ▼          ▼        ▼        ▼         ▼
                  CSV        JSON    Parquet  Summary  Analytics
```

## Data Tower Joining Strategy

### Stock Data Tower - Merge Process

```
Financial Data                  Stock Movements
┌─────────────────┐             ┌─────────────────┐
│ AAPL 10:00 AM   │             │ AAPL 10:02 AM   │
│ Price: 150.0    │    Merge    │ RSI: 72.5       │
│ Volume: 50M     │   (±60min)  │ SMA_20: 148.5   │
│ Market Cap: 2.4T├────────────┤ MACD: 2.5       │
│ PE: 28.5        │             │ ...             │
│ ...             │             │                 │
└─────────────────┘             └─────────────────┘
                                        │
                                        ▼
                          ┌──────────────────────┐
                          │  STOCK DATA TOWER    │
                          ├──────────────────────┤
                          │ AAPL 10:00 AM        │
                          │ Price: 150.0         │
                          │ Volume: 50M          │
                          │ Market Cap: 2.4T     │
                          │ PE: 28.5             │
                          │ RSI: 72.5            │
                          │ SMA_20: 148.5        │
                          │ MACD: 2.5            │
                          │ ...                  │
                          └──────────────────────┘
```

### Context Data Tower - Combine Process

```
News Data              Macro Data             Policy Data
┌──────────────┐      ┌──────────────┐       ┌──────────────┐
│ AAPL News    │      │ Market Data  │       │ Fed Action   │
│ Headline: .. │      │ Int Rate: .. │       │ Event: ...   │
│ Sentiment: ..│      │ Unemployment │       │ Impact: ...  │
└──────────────┘      └──────────────┘       └──────────────┘
       │                      │                      │
       └──────────┬───────────┴──────────┬───────────┘
                  │                      │
            Prefix with:          Prefix with:
          (news_headline,        (macro_interest,
           news_sentiment)        policy_event)
                  │                      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ CONTEXT DATA TOWER   │
                  ├──────────────────────┤
                  │ AAPL / MARKET Time   │
                  │ news_headline: ...   │
                  │ news_sentiment: ...  │
                  │ macro_interest: ...  │
                  │ policy_event: ...    │
                  │ ...                  │
                  └──────────────────────┘
```

### Final Join - Stock + Context

```
Stock Data Tower          Context Data Tower
┌──────────────────┐     ┌──────────────────┐
│ ticker: AAPL     │     │ ticker: AAPL     │
│ timestamp: 10:00 │     │ timestamp: 10:00 │
│ price: 150.0     │     │ news_headline: ..│
│ volume: 50M      │     │ sentiment: 0.85  │
│ sma_20: 148.5    │     │ macro_interest: ..│
│ rsi: 72.5        │     │ policy_event: ...│
└──────────────────┘     └──────────────────┘
         │                      │
         └──────────┬───────────┘
                    │
                    │ Outer Join on
                    │ (ticker, timestamp)
                    │
                    ▼
         ┌──────────────────────┐
         │  TRAINING DATASET    │
         ├──────────────────────┤
         │ ticker: AAPL         │
         │ timestamp: 10:00     │
         │ price: 150.0         │ Stock
         │ volume: 50M          │ Features
         │ sma_20: 148.5        │
         │ rsi: 72.5            │
         │ news_headline: ...   │ Context
         │ sentiment: 0.85      │ Features
         │ macro_interest: ...  │
         │ policy_event: ...    │
         │ ...                  │
         └──────────────────────┘
```

## Data Flow: End-to-End

```
STAGE 1: DATA COLLECTION
├─ Financial Data Source
│  └─ Yahoo Finance → /data/raw/financial_data/
├─ Stock Movement Source
│  └─ Technical Indicators → /data/raw/stock_movements/
├─ News Source
│  └─ News + Sentiment → /data/raw/news/
├─ Macro Source
│  └─ Economic Indicators → /data/raw/macroeconomic_data/
└─ Policy Source
   └─ Fed Announcements → /data/raw/policy_data/

STAGE 2: LOADING
├─ StockDataTower.load_data()
│  ├─ Load /data/raw/financial_data/*.json
│  ├─ Load /data/raw/stock_movements/*.json
│  └─ Parse JSON files into DataFrames
└─ ContextDataTower.load_data()
   ├─ Load /data/raw/news/*.json
   ├─ Load /data/raw/macroeconomic_data/*.json
   ├─ Load /data/raw/policy_data/*.json
   └─ Parse JSON files into DataFrames

STAGE 3: NORMALIZATION
├─ StockDataTower.normalize_schema()
│  ├─ Ensure required columns exist
│  ├─ Convert types (float, datetime)
│  └─ Handle missing values (fill with NULL)
└─ ContextDataTower.normalize_schema()
   ├─ Ensure required columns exist
   ├─ Convert types
   └─ Handle missing values

STAGE 4: TOWER PROCESSING
├─ Stock Tower: Merge financial + movements
│  └─ merge_asof() on (ticker, timestamp)
└─ Context Tower: Combine news + macro + policy
   └─ Multiple merges with prefix naming

STAGE 5: JOINING
├─ Final join: Stock + Context
│  ├─ pd.merge() on (ticker, timestamp)
│  ├─ how='outer' to preserve all records
│  └─ Sort by ticker and timestamp
└─ Result: Unified training dataset

STAGE 6: OUTPUT
├─ Save to disk
│  ├─ CSV: /data/training/training_data_YYYYMMDD_HHMMSS.csv
│  ├─ Parquet: /data/training/training_data_YYYYMMDD_HHMMSS.parquet
│  └─ JSON: /data/training/training_data_YYYYMMDD_HHMMSS.json
└─ Generate summary statistics
   ├─ Total records and features
   ├─ Missing value analysis
   ├─ Data type summary
   └─ Date range coverage

STAGE 7: ML USAGE
├─ Load training data
├─ Prepare features and targets
├─ Split train/test sets
├─ Train ML model
│  ├─ Scikit-learn
│  ├─ XGBoost
│  ├─ TensorFlow/Keras
│  └─ Other frameworks
└─ Evaluate and iterate
```

## Class Hierarchy

```
BaseDataTower (Abstract)
├── StockDataTower
│   ├─ _load_financial_data(start, end, tickers)
│   ├─ _load_stock_movements(start, end)
│   ├─ _load_json_files(dir, start, end, filter_tickers)
│   ├─ _merge_data_frames(fin_df, mov_df)
│   └─ normalize_schema(df)
│
└── ContextDataTower
    ├─ _load_news_data(start, end, tickers)
    ├─ _load_macro_data(start, end)
    ├─ _load_policy_data(start, end)
    ├─ _load_json_files(dir, start, end)
    ├─ _combine_context_data(news_df, macro_df, policy_df)
    └─ normalize_schema(df)

TrainingDataProcessor
├─ stock_tower: StockDataTower
├─ context_tower: ContextDataTower
├─ generate_training_data(start, end, tickers, save)
├─ _join_towers(stock_df, context_df)
├─ _save_training_data(df)
├─ get_feature_summary(df)
└─ print_feature_summary(df)
```

## Feature Set Expansion

```
Input Features (Raw)          Generated Features        Output Features
                            (via normalization)        (for ML models)

Financial Data                                         
├─ ticker                    [normalized]            ├─ ticker ✓
├─ price                     [float] ────────────────├─ price ✓
├─ open                      [float] ────────────────├─ open ✓
├─ high                      [float] ────────────────├─ high ✓
├─ low                       [float] ────────────────├─ low ✓
├─ close                     [float] ────────────────├─ close ✓
├─ volume                    [float] ────────────────├─ volume ✓
├─ market_cap                [float] ────────────────├─ market_cap ✓
├─ pe_ratio                  [float] ────────────────├─ pe_ratio ✓
├─ dividend_yield            [float] ────────────────├─ dividend_yield ✓
├─ 52_week_high              [float] ────────────────├─ 52_week_high ✓
└─ 52_week_low               [float] ────────────────└─ 52_week_low ✓

Stock Movements
├─ sma_20                    [float] ────────────────├─ sma_20 ✓
├─ sma_50                    [float] ────────────────├─ sma_50 ✓
├─ rsi                       [float] ────────────────├─ rsi ✓
├─ macd                      [float] ────────────────├─ macd ✓
└─ macd_signal               [float] ────────────────└─ macd_signal ✓

News
├─ headline                  [prefixed]            ├─ news_headline ✓
├─ sentiment                 [normalized]          ├─ news_sentiment_polarity ✓
└─ summary                   [prefixed]            └─ news_sentiment_subjectivity ✓

Macro
├─ interest_rate             [prefixed]            ├─ macro_interest_rate ✓
├─ unemployment_rate         [prefixed]            ├─ macro_unemployment_rate ✓
├─ gdp_growth                [prefixed]            ├─ macro_gdp_growth ✓
├─ inflation_rate            [prefixed]            ├─ macro_inflation_rate ✓
└─ fed_funds_rate            [prefixed]            └─ macro_fed_funds_rate ✓

Policy
├─ event_type                [prefixed]            ├─ policy_event_type ✓
├─ title                     [prefixed]            ├─ policy_title ✓
└─ impact_level              [prefixed]            └─ policy_impact_level ✓

Metadata
├─ timestamp                 [datetime] ────────────├─ timestamp ✓
└─ (added during join)       [aligned] ─────────────└─ (aligned)
```

## Configuration Options

```
TrainingDataProcessor Configuration
{
    'data_root': str,                # Root directory (default: '/data')
    'output_format': str,            # 'csv' | 'parquet' | 'json'
    'output_path': str,              # Output directory path
}

generate_training_data() Parameters
├─ start_date: datetime             # Default: 30 days ago
├─ end_date: datetime               # Default: today
├─ tickers: List[str]               # Default: Mag 7
└─ save: bool                        # Default: True
```

## Performance Characteristics

```
MEMORY USAGE
           Stock Tower        Context Tower       Combined
100 records:    ~5 MB              ~3 MB              ~8 MB
10K records:    ~50 MB             ~30 MB             ~80 MB
100K records:   ~500 MB            ~300 MB            ~800 MB
1M records:     ~5 GB              ~3 GB              ~8 GB

PROCESSING TIME (100K records)
┌──────────────────┬──────────┐
│ Operation        │ Time     │
├──────────────────┼──────────┤
│ Load financial   │ ~1-2s    │
│ Load movements   │ ~1s      │
│ Load context     │ ~1s      │
│ Merge stock      │ ~500ms   │
│ Combine context  │ ~500ms   │
│ Join towers      │ ~500ms   │
├──────────────────┼──────────┤
│ Total            │ ~5-7s    │
├──────────────────┼──────────┤
│ Save (Parquet)   │ ~500ms   │
│ Save (CSV)       │ ~2-3s    │
│ Save (JSON)      │ ~1-2s    │
└──────────────────┴──────────┘

OUTPUT FILE SIZES (100K records)
┌──────────────────┬──────────┐
│ Format           │ Size     │
├──────────────────┼──────────┤
│ Parquet          │ 6-8 MB   │
│ CSV              │ 30-40 MB │
│ JSON             │ 25-35 MB │
└──────────────────┴──────────┘
```

## Error Handling Flow

```
Data Loading
    │
    ├─ Directory not found?
    │  └─ Log WARNING, return empty DataFrame
    │
    ├─ JSON parse error?
    │  └─ Log WARNING for file, skip to next
    │
    ├─ Missing required columns?
    │  └─ Add with NULL values
    │
    └─ Type conversion error?
       └─ Use pd.to_numeric(..., errors='coerce')

Merging
    │
    ├─ Empty dataframes?
    │  └─ Return the non-empty one
    │
    ├─ No common keys?
    │  └─ Log WARNING, return stock data
    │
    └─ Timestamp mismatch?
       └─ Use merge_asof with tolerance

Output
    │
    ├─ Invalid format?
    │  └─ Raise ValueError
    │
    ├─ Directory doesn't exist?
    │  └─ Create with os.makedirs()
    │
    └─ Write permission error?
       └─ Raise with helpful message
```

## Integration Points

```
Data Pipeline Components
├─ Data Sources (sources/*.py)
│  └─ Feed raw data to /data/raw/
│
├─ Training Data Processor (core/training_data.py)
│  ├─ Reads from /data/raw/
│  └─ Outputs to /data/training/
│
├─ ML Models (client or external)
│  ├─ Read from /data/training/
│  ├─ Train and evaluate
│  └─ Generate predictions
│
└─ Storage (storage/*.py)
   └─ Optional: Store training data via Sinks
      (JSONSink, CSVSink, ProtobufSink)
```

## Usage Patterns

```
Pattern 1: Batch Processing
for ticker in ticker_list:
    data = processor.generate_training_data(tickers=[ticker])
    train_model(data, ticker)

Pattern 2: Time Series Splitting
for start_date in date_ranges:
    end_date = start_date + timedelta(days=30)
    data = processor.generate_training_data(start_date, end_date)
    train_model(data)

Pattern 3: Rolling Window
for i in range(num_windows):
    window_start = base_date + (i * window_size)
    window_end = window_start + timedelta(days=window_size)
    data = processor.generate_training_data(window_start, window_end)
    evaluate_model(data)

Pattern 4: Incremental Learning
for new_data_batch in streaming_data:
    training_data.append(new_data_batch)
    if len(training_data) >= batch_threshold:
        retrain_model(training_data)
        training_data.clear()
```

---

This document provides a complete visual reference for understanding the Training Data Processor architecture, data flow, and usage patterns.
