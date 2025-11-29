# Stock Trend Estimator - Architecture Documentation

## System Overview

The Stock Trend Estimator data pipeline is a pure offline data collection system using Flume Python. It aggregates financial data, market trends, news, and macroeconomic indicators for stock trend estimation without real-time serving requirements.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          EXTERNAL DATA SOURCES                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Financial APIs          Economic Data          News & Policy      │
│  ├─ Yahoo Finance         ├─ FRED               ├─ Finnhub         │
│  ├─ Alpha Vantage         ├─ World Bank         ├─ NewsAPI         │
│  └─ Finnhub              └─ Treasury            └─ Federal Reserve  │
│                                                                      │
└────────────────┬───────────────────────────────────────────────────┘
                 │
┌────────────────▼───────────────────────────────────────────────────┐
│              FLUME DATA COLLECTION AGENTS                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Agent 1: Raw Data Collection                               │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                             │   │
│  │  Sources:                   Channels:         Sinks:       │   │
│  │  ┌─────────────────────┐   ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Financial API       │   │ Memory       │  │ Parquet  │ │   │
│  │  │ (Mag 7)            │→→→ Channel 1    │→→→ Sink 1   │ │   │
│  │  └─────────────────────┘   └──────────────┘  └──────────┘ │   │
│  │                                                             │   │
│  │  ┌─────────────────────┐   ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Stock Movements     │   │ File         │  │ Parquet  │ │   │
│  │  │ (S&P 500)          │→→→ Channel 2    │→→→ Sink 2   │ │   │
│  │  └─────────────────────┘   └──────────────┘  └──────────┘ │   │
│  │                                                             │   │
│  │  ┌─────────────────────┐   ┌──────────────┐  ┌──────────┐ │   │
│  │  │ News Data           │   │ Memory       │  │ Parquet  │ │   │
│  │  │ (S&P 500)          │→→→ Channel 3    │→→→ Sink 3   │ │   │
│  │  └─────────────────────┘   └──────────────┘  └──────────┘ │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Agent 2: Context Data Collection                            │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                             │   │
│  │  Sources:                   Channels:         Sinks:       │   │
│  │  ┌─────────────────────┐   ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Macroeconomic       │   │ File         │  │ Parquet  │ │   │
│  │  │ Indicators (Mag 7)  │→→→ Channel 4    │→→→ Sink 4   │ │   │
│  │  └─────────────────────┘   └──────────────┘  └──────────┘ │   │
│  │                                                             │   │
│  │  ┌─────────────────────┐   ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Policy Data         │   │ File         │  │ Parquet  │ │   │
│  │  │ (Mag 7)            │→→→ Channel 5    │→→→ Sink 5   │ │   │
│  │  └─────────────────────┘   └──────────────┘  └──────────┘ │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────┐
│                        PERSISTENT DATA STORAGE                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  /data/raw/                     /data/context/                       │
│  ├─ financial_data/             ├─ macroeconomic/                   │
│  │  └─ YYYY-MM-DD/              │  └─ YYYY-MM-DD/                  │
│  │     ├─ mag7_financial_*.parquet  │     └─ macro_indicators_*.parquet
│  │     └─ ...                    │        └─ ...                    │
│  ├─ stock_movements/            ├─ policy/                          │
│  │  └─ YYYY-MM-DD/              │  └─ YYYY-MM-DD/                  │
│  │     ├─ sp500_trends_*.parquet    │     ├─ policy_data_*.parquet │
│  │     └─ ...                    │        └─ ...                    │
│  └─ news/                                                            │
│     └─ YYYY-MM-DD/                                                  │
│        ├─ sp500_news_*.parquet                                      │
│        └─ ...                                                        │
│                                                                      │
└──────────────────────────────────────┬───────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────┐
│                      OFFLINE CLIENT INTERFACE                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DataPipelineClient                                                 │
│  ├─ get_financial_data()        → Query & Filter                    │
│  ├─ get_stock_movements()       → Analytics                         │
│  ├─ get_news_data()             → Sentiment Analysis                │
│  ├─ get_macroeconomic_data()    → Context Data                      │
│  ├─ get_policy_data()           → Policy Information                │
│  ├─ export_data()               → Multi-format Export               │
│  └─ get_data_summary()          → Statistics                        │
│                                                                      │
│  Supported Formats: CSV, Parquet, JSON, PostgreSQL, MongoDB         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Sources Layer

#### Financial API Source (Mag 7)
- **Collects**: Prices, OHLC, Market Cap, P/E Ratio, Dividend Yield
- **Stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **APIs**: Yahoo Finance, Alpha Vantage, Finnhub
- **Frequency**: Hourly
- **Batch Size**: 100 events
- **Error Handling**: 3 retry attempts, 30-second timeout

#### Stock Movement Source (S&P 500)
- **Collects**: Price changes, 52-week highs/lows, Technical indicators
- **Stocks**: All S&P 500 stocks (limited to subset in demo)
- **Indicators**: SMA (20, 50), RSI (14), MACD
- **APIs**: Yahoo Finance, Alpha Vantage
- **Frequency**: Hourly
- **Batch Size**: 500 events
- **History**: 3 months for indicator calculation

#### News Data Source (S&P 500)
- **Collects**: Headlines, summaries, sources, sentiment scores
- **Stocks**: All S&P 500 stocks
- **APIs**: Finnhub, NewsAPI
- **Sentiment**: TextBlob (polarity, subjectivity)
- **Frequency**: Hourly
- **Batch Size**: 50 events

#### Macroeconomic Data Source (Mag 7)
- **Collects**: Interest rates, unemployment, GDP growth, inflation, Fed rate
- **Sources**: FRED, World Bank, Alpha Vantage
- **Frequency**: Daily
- **Batch Size**: 10 events
- **Update Time**: 09:00 UTC

#### Policy Data Source (Mag 7)
- **Collects**: Fed announcements, FOMC meetings, Treasury decisions, Economic calendar
- **Sources**: Federal Reserve, Treasury, Economic indicators
- **Frequency**: Weekly (plus on-demand for announcements)
- **Batch Size**: 20 events
- **Update Time**: 09:00 UTC Monday

### 2. Channel Layer

#### Memory Channels
- **Use**: Fast, low-latency buffering
- **Capacity**: 1000-5000 events
- **Persistence**: None (data lost on restart)
- **Used for**: Financial data, News (quick processing)
- **Advantages**: Fast, minimal disk I/O
- **Disadvantages**: No recovery capability

#### File Channels
- **Use**: Persistent buffering with recovery
- **Capacity**: 5000-10000 events
- **Persistence**: Checkpoint + data files
- **Used for**: Stock movements, Macro, Policy (slow processing)
- **Advantages**: Data recovery, reliable
- **Disadvantages**: Slower than memory

### 3. Sink Layer

#### Parquet Sink (Primary)
- **Format**: Apache Parquet (columnar)
- **Compression**: Snappy (financial/news), Gzip (macro/policy)
- **Advantages**: 
  - Efficient compression
  - Fast columnar queries
  - Schema evolution support
  - Suitable for big data tools
- **Used for**: All data types

#### CSV Sink (Secondary)
- **Format**: Comma-separated values
- **Advantages**: 
  - Human-readable
  - Excel compatible
  - No dependencies
- **Use case**: Quick analysis, sharing

#### JSON Sink (Flexible)
- **Format**: JSON Lines
- **Advantages**:
  - Flexible schema
  - Easy API integration
  - Supports nested data
- **Use case**: API responses, complex structures

#### Database Sinks (Advanced)
- **PostgreSQL**: Structured relational storage
- **MongoDB**: Document-based flexible storage
- **Use case**: Long-term analysis, dashboards

### 4. Storage Strategy

#### Directory Structure
```
/data/
├── raw/
│   ├── financial_data/YYYY-MM-DD/mag7_financial_HHMMSS.parquet
│   ├── stock_movements/YYYY-MM-DD/sp500_trends_HHMMSS.parquet
│   └── news/YYYY-MM-DD/sp500_news_HHMMSS.parquet
└── context/
    ├── macroeconomic/YYYY-MM-DD/macro_indicators_HHMMSS.parquet
    └── policy/YYYY-MM-DD/policy_data_HHMMSS.parquet
```

#### Retention Policy
- **Financial Data**: 90 days
- **Stock Movements**: 90 days
- **News Data**: 60 days
- **Macroeconomic**: 1 year (historical)
- **Policy Data**: 2 years (historical)

### 5. Client Interface (Offline)

```python
class DataPipelineClient:
    """Pure offline client - queries collected data"""
    
    # Query methods
    get_financial_data(tickers, start_date, end_date)
    get_stock_movements(start_date, end_date, indicators)
    get_news_data(start_date, end_date, sentiment_filter, tickers)
    get_macroeconomic_data(indicators, start_date, end_date)
    get_policy_data(data_types, start_date, end_date)
    
    # Export methods
    export_data(data_type, output_path, format)
    
    # Utility methods
    get_data_summary()
    get_data_quality_metrics()
```

## Data Flow

### Example: Financial Data Collection Flow

```
1. TRIGGER (Hourly)
   └─ Scheduler triggers financial_api_source

2. SOURCE (FinancialDataSource)
   ├─ Query Yahoo Finance API for AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
   ├─ Extract: price, OHLC, volume, market_cap, pe_ratio, dividend_yield
   ├─ Convert to Event objects with metadata
   └─ Return list of events

3. CHANNEL (Memory Channel)
   ├─ Receive events
   ├─ Buffer up to 1000 events
   ├─ Transaction grouping
   └─ Forward to sink

4. SINK (Parquet Sink)
   ├─ Batch events (max 100)
   ├─ Convert to DataFrame
   ├─ Compress with Snappy
   ├─ Write to /data/raw/financial_data/YYYY-MM-DD/mag7_financial_HHMMSS.parquet
   └─ Log success/failure

5. STORAGE
   └─ File persisted to disk

6. CLIENT RETRIEVAL
   ├─ Client calls get_financial_data()
   ├─ Scan /data/raw/financial_data/ by date filter
   ├─ Load Parquet files → DataFrame
   ├─ Apply optional ticker/indicator filters
   └─ Return results
```

## Scalability Considerations

### Current Scale
- **Mag 7 stocks**: 7 tickers
- **S&P 500 stocks**: ~500 tickers (limited to 100 in demo)
- **Collection frequency**: Hourly → ~7,000 records/day

### Data Growth
```
Financial Data:     7 tickers × 1 record/hour × 24 = 168 rec/day × 1KB ≈ 170 KB/day
Stock Movements:   100 tickers × 1 record/hour × 24 = 2,400 rec/day × 2KB ≈ 5 MB/day
News Data:         100 tickers × 5 articles/hour × 24 = 12,000 rec/day × 1KB ≈ 12 MB/day
Macroeconomic:     1 record/day × 10 indicators ≈ 1 MB/month
Policy Data:       ~20 records/week ≈ 1 MB/month

TOTAL: ~20 MB/day ≈ 600 MB/month ≈ 7.2 GB/year (uncompressed)
WITH COMPRESSION: ~2 GB/year
```

### Future Scale-up
1. **Add more data sources** (earnings calls, insider trading, etc.)
2. **Increase S&P 500 coverage** (all 500 stocks)
3. **Increase collection frequency** (real-time -> minutes)
4. **Add international stocks** (FTSE 100, DAX, Nikkei, etc.)

Solutions:
- Distributed Flume agents (multiple machines)
- Kafka for inter-agent communication
- Hadoop/Spark for batch processing
- Data warehouse (Snowflake, BigQuery)

## Reliability & Fault Tolerance

### Error Handling

#### Source Level
- Retry failed API calls (configurable attempts)
- Exponential backoff for rate limiting
- Timeout protection (configurable seconds)
- Graceful degradation (skip unavailable tickers)

#### Channel Level
- File channels for recovery on restart
- Checkpoint mechanism for data integrity
- Automatic recovery of in-flight events

#### Sink Level
- Write failures trigger retry
- Failed batches stored for manual inspection
- Duplicate detection and handling

### Recovery Mechanisms

```
Failure Point        →  Recovery Strategy
─────────────────────────────────────────
API timeout          →  Retry with exponential backoff
Network error        →  Queue to file channel
Disk full            →  Alert + graceful shutdown
Memory overflow      →  Reduce batch size
Corrupted data       →  Skip record, log error
Lost checkpoint      →  Restart from latest position
```

## Security Considerations

### API Credentials
- Store in `credentials.json` with restricted permissions (0600)
- Use environment variables in production
- Never commit credentials to version control
- Rotate API keys periodically

### Data Protection
- Encrypt sensitive data at rest (optional)
- Use HTTPS for all API calls
- Validate SSL certificates
- Sanitize logs to exclude credentials

### Access Control
- Restrict `/data/` directory permissions
- Use OS-level file permissions
- Database user separation
- API rate limiting

## Performance Optimization

### Collection Optimization
```
Memory vs. File Channels:
- Memory: 50ms latency, no recovery
- File: 100-500ms latency, full recovery

Batch Size Impact:
- Larger batches = fewer writes, better throughput
- Smaller batches = lower latency, quicker failure detection
```

### Storage Optimization
```
Compression Impact:
- Snappy: 30-50% compression, 100ms overhead per file
- Gzip: 70-80% compression, 500ms overhead per file
- Uncompressed: 0% compression, 0ms overhead

Format Impact:
- Parquet: 50% smaller, 10x faster queries vs CSV
- CSV: 100% size, good for export
- JSON: 200% size, flexible schema
```

### Query Optimization
```
Good Practices:
1. Always filter by date range
2. Use Parquet for large datasets
3. Cache frequently accessed data
4. Use columnar queries (Parquet advantage)

Query Example:
# Slow: Load all data
df = client.get_financial_data()

# Fast: Filter by date
df = client.get_financial_data(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)
```

## Monitoring & Observability

### Key Metrics
1. **Collection Success Rate**: > 95%
2. **Data Freshness**: < 2 hours old
3. **API Response Time**: < 5 seconds
4. **Storage Growth**: Track daily increase
5. **Error Rate**: < 1% of records

### Logging Strategy
- **INFO**: Collection start/completion, data counts
- **WARNING**: API timeouts, partial failures, rate limits
- **ERROR**: Collection failures, data corruption, disk errors
- **DEBUG**: Detailed API responses, transformation steps

### Example Metrics Collection
```python
from data_pipeline.client.pipeline_client import get_data_client
import datetime

client = get_data_client()
summary = client.get_data_summary()

print(f"Financial data files: {summary['data_sources']['financial_data']}")
print(f"Total storage: {summary['total_size_bytes'] / (1024**3):.2f} GB")
print(f"Latest collection: {summary['timestamp']}")
```

## Conclusion

The Stock Trend Estimator data pipeline provides a robust, scalable offline data collection system using Flume Python. It handles multiple data sources, formats, and storage backends while maintaining high reliability and performance.

Key features:
- ✅ Pure offline operation
- ✅ Multiple data sources (financial, news, economic)
- ✅ Flexible storage formats (Parquet, CSV, JSON, Database)
- ✅ Fault tolerance and recovery
- ✅ Easy client interface for data queries
- ✅ Highly configurable and extensible
