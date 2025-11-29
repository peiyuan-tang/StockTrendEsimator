# Stock Trend Estimator - Offline Data Collection Pipeline

## Overview

A comprehensive offline data collection pipeline using Flume Python for aggregating financial data, market trends, news, and macroeconomic indicators for stock trend estimation.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Flume Data Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RAW DATA COLLECTION (Agent 1)                             │
│  ├─ Financial Data Source (Mag 7 Stocks)                  │
│  ├─ Stock Movement Trends (S&P 500)                       │
│  └─ News Data (S&P 500)                                   │
│         ↓                                                   │
│  CONTEXT DATA COLLECTION (Agent 2)                         │
│  ├─ Macroeconomic Indicators (Mag 7)                      │
│  └─ Policy & Monetary Data (Mag 7)                        │
│                                                             │
│  CHANNELS (Memory & File-based)                            │
│  ├─ Financial Channel (Memory)                             │
│  ├─ Movement Channel (File)                                │
│  ├─ News Channel (Memory)                                  │
│  ├─ Macro Channel (File)                                   │
│  └─ Policy Channel (File)                                  │
│         ↓                                                   │
│  SINKS (Multi-format Storage)                              │
│  ├─ Parquet (Columnar - Efficient)                         │
│  ├─ CSV (Tabular)                                          │
│  ├─ JSON (Flexible)                                        │
│  └─ Database (PostgreSQL, MongoDB)                         │
│         ↓                                                   │
│  DATA STORAGE                                              │
│  ├─ /data/raw/financial_data/                              │
│  ├─ /data/raw/stock_movements/                             │
│  ├─ /data/raw/news/                                        │
│  ├─ /data/context/macroeconomic/                           │
│  └─ /data/context/policy/                                  │
│                                                             │
│  CLIENT INTERFACE (Pure Offline)                           │
│  └─ DataPipelineClient                                     │
│     (Query, Filter, Export - No Real-time Serving)         │
└─────────────────────────────────────────────────────────────┘
```

## Data Collection Pipelines

### 1. Raw Data Collection (Agent 1)

#### Financial Data Pipeline
- **Source**: Yahoo Finance, Alpha Vantage, Finnhub
- **Stocks**: Mag 7 (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
- **Frequency**: Hourly
- **Data**:
  - Current price, OHLC
  - Market cap, P/E ratio
  - Dividend yield
  - 52-week high/low
- **Format**: Parquet (Snappy compression)
- **Sink**: `/data/raw/financial_data/`

#### Stock Movement Trends Pipeline
- **Source**: Yahoo Finance, Alpha Vantage
- **Stocks**: All S&P 500 stocks
- **Frequency**: Hourly
- **Data**:
  - Price changes and percentages
  - 52-week highs/lows
  - 90-day average volume
  - Technical indicators:
    - SMA (20, 50)
    - RSI (14-period)
    - MACD
- **Format**: Parquet (Snappy compression)
- **Sink**: `/data/raw/stock_movements/`

#### News Data Pipeline
- **Source**: Finnhub API, NewsAPI
- **Stocks**: All S&P 500 stocks
- **Frequency**: Hourly
- **Data**:
  - Headline, summary
  - Source, URL
  - Publication date
  - Sentiment analysis (polarity, subjectivity)
- **Format**: Parquet (Snappy compression)
- **Sink**: `/data/raw/news/`

### 2. Context Data Collection (Agent 2)

#### Macroeconomic Data Pipeline
- **Source**: FRED (Federal Reserve), World Bank, Alpha Vantage
- **Applicable to**: Mag 7 Stocks
- **Frequency**: Daily
- **Indicators**:
  - Interest rates (10-year Treasury yield)
  - Unemployment rate
  - GDP growth
  - Inflation rate (CPI)
  - Fed funds rate
- **Format**: Parquet (Gzip compression)
- **Sink**: `/data/context/macroeconomic/`

#### Fiscal & Monetary Policy Pipeline
- **Source**: Federal Reserve, Treasury Department, Economic Calendar
- **Applicable to**: Mag 7 Stocks
- **Frequency**: Weekly (on-demand for announcements)
- **Data Types**:
  - Policy announcements
  - FOMC meeting schedule and minutes
  - Treasury decisions
  - Economic indicators schedule
- **Format**: Parquet (Gzip compression)
- **Sink**: `/data/context/policy/`

## Project Structure

```
StockTrendEsimator/
├── data_pipeline/
│   ├── config/
│   │   ├── flume_config.yaml           # Flume agent configuration
│   │   ├── config_manager.py           # Configuration management
│   │   ├── pipeline_config.json        # Pipeline settings (generated)
│   │   └── credentials.json            # API credentials (generated)
│   │
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── financial_source.py         # Financial data collection
│   │   ├── movement_source.py          # Stock trends collection
│   │   ├── news_source.py              # News data collection
│   │   ├── macro_source.py             # Macro indicators collection
│   │   └── policy_source.py            # Policy data collection
│   │
│   ├── sinks/
│   │   ├── __init__.py
│   │   └── data_sink.py                # Multi-format data sinks
│   │
│   ├── server/
│   │   ├── __init__.py
│   │   ├── flume_server.py             # Main Flume server
│   │   └── pipeline_scheduler.py       # Task scheduling
│   │
│   ├── client/
│   │   ├── __init__.py
│   │   └── pipeline_client.py          # Client interface (offline)
│   │
│   └── tests/
│       ├── test_sources.py
│       ├── test_sinks.py
│       └── test_client.py
│
├── requirements.txt                     # Python dependencies
├── setup.py                            # Package setup
└── README.md                           # This file
```

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/peiyuan-tang/StockTrendEsimator.git
cd StockTrendEsimator
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create `data_pipeline/config/credentials.json`:

```json
{
  "finnhub_api_key": "your_finnhub_key",
  "newsapi_key": "your_newsapi_key",
  "alpha_vantage_key": "your_alpha_vantage_key",
  "fred_api_key": "your_fred_api_key"
}
```

> **Note**: Get API keys from:
> - [Finnhub](https://finnhub.io/) - Free tier available
> - [NewsAPI](https://newsapi.org/) - Free tier available
> - [Alpha Vantage](https://www.alphavantage.co/) - Free tier available
> - [FRED](https://fredaccount.stlouisfed.org/apikey) - Free, requires registration

### 5. Create Data Directories

```bash
mkdir -p /data/raw/{financial_data,stock_movements,news}
mkdir -p /data/context/{macroeconomic,policy}
mkdir -p /var/data/{checkpoint,queue}/{movement,macro,policy}
mkdir -p /var/log/stock_pipeline
```

## Usage

### Start the Data Collection Pipeline

```bash
python data_pipeline/server/flume_server.py \
  --config data_pipeline/config/flume_config.yaml \
  --log-level INFO
```

### Query Collected Data (Client Interface)

```python
from data_pipeline.client.pipeline_client import get_data_client

# Get client
client = get_data_client()

# Retrieve financial data
financial_df = client.get_financial_data()

# Retrieve stock movements with date filter
from datetime import datetime, timedelta
end_date = datetime.utcnow()
start_date = end_date - timedelta(days=7)
movements_df = client.get_stock_movements(start_date=start_date, end_date=end_date)

# Retrieve news with sentiment filter
news_df = client.get_news_data(sentiment_filter=(0.0, 1.0))  # Positive sentiment

# Retrieve macroeconomic data
macro_df = client.get_macroeconomic_data(
    indicators=['interest_rate', 'unemployment_rate']
)

# Retrieve policy data
policy_df = client.get_policy_data(
    data_types=['policy_announcements', 'fomc_meeting']
)

# Get data summary
summary = client.get_data_summary()
print(summary)

# Export data
client.export_data(
    'financial_data',
    'exports/mag7_financial.csv',
    format='csv'
)
```

## Configuration

### Flume Configuration (YAML)

Edit `data_pipeline/config/flume_config.yaml`:

```yaml
agents:
  financial_data_agent:
    sources:
      - financial_api_source    # Mag 7 financial data
      - stock_movement_source   # S&P 500 trends
      - news_source             # S&P 500 news
```

### Pipeline Configuration

Edit `data_pipeline/config/pipeline_config.json`:

```json
{
  "mag7_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
  "sp500_limit": 500,
  "financial_data_interval": 3600,
  "macro_interval": 86400,
  "data_root_path": "/data",
  "retention_days": 90
}
```

## Features

### Data Sources
- ✅ Financial APIs (Yahoo Finance, Alpha Vantage, Finnhub)
- ✅ Economic Data (FRED, World Bank)
- ✅ News Aggregation (Finnhub, NewsAPI)
- ✅ Technical Indicators (SMA, RSI, MACD)
- ✅ Sentiment Analysis (TextBlob)
- ✅ Policy Data (Federal Reserve, Treasury)

### Storage Formats
- ✅ Parquet (Columnar, compressed)
- ✅ CSV (Tabular)
- ✅ JSON (Flexible)
- ✅ PostgreSQL (Relational)
- ✅ MongoDB (Document store)

### Channels
- ✅ Memory channels (fast, in-memory buffering)
- ✅ File channels (persistent, recovery)

### Offline Client Features
- ✅ Query collected data by type and date range
- ✅ Filter by ticker, sentiment, indicators
- ✅ Export to multiple formats
- ✅ Data summary and statistics
- ✅ No real-time serving requirement

## Data Retention Policy

- **Financial Data**: 90 days (configurable)
- **Stock Movements**: 90 days (configurable)
- **News Data**: 60 days (configurable)
- **Macroeconomic Data**: 1 year (historical)
- **Policy Data**: 2 years (historical)

Older data is automatically archived or removed based on retention settings.

## Monitoring & Logging

### Log Files

- Main pipeline: `/var/log/stock_pipeline/pipeline.log`
- Agent logs: `/var/log/stock_pipeline/agents/`
- Error logs: `/var/log/stock_pipeline/errors.log`

### Log Level

Set in command line or config:

```bash
--log-level DEBUG    # Maximum verbosity
--log-level INFO     # Standard
--log-level WARNING  # Errors and warnings only
--log-level ERROR    # Errors only
```

## Performance Considerations

### Channel Configuration
- **Memory Channels**: Fast but data lost on restart
  - Capacity: 1000-5000 events
  - Use for: Financial data, news (fast processing)

- **File Channels**: Slower but persistent
  - Capacity: 5000-10000 events
  - Use for: Stock movements, macro data (long processing)

### Batch Sizes
- Financial: 100 events/batch
- Stock Movement: 500 events/batch
- News: 50 events/batch
- Macro: 10 events/batch
- Policy: 20 events/batch

### API Rate Limits
- Finnhub: 60 requests/minute (free tier)
- NewsAPI: 100 requests/day (free tier)
- Alpha Vantage: 5 requests/minute (free tier)
- FRED: Unlimited

Batch processing and caching mitigate rate limits.

## Troubleshooting

### Common Issues

**Issue**: API key errors
```bash
# Check credentials file
cat data_pipeline/config/credentials.json
# Verify API keys are valid
```

**Issue**: Data not being collected
```bash
# Check Flume logs
tail -f /var/log/stock_pipeline/pipeline.log
# Verify data directories exist
ls -la /data/raw/ /data/context/
```

**Issue**: Out of memory
```bash
# Reduce batch sizes in config
# Reduce memory channel capacity
# Use file-based channels instead
```

**Issue**: Slow data loading
```python
# Use Parquet format (more efficient)
client.export_data('financial_data', 'output.parquet', format='parquet')
# Apply date filters
client.get_financial_data(start_date=start, end_date=end)
```

## Advanced Features

### Custom Sources

Create a custom data source:

```python
from data_pipeline.sources.financial_source import BaseDataSource

class CustomSource(BaseDataSource):
    def fetch_data(self):
        # Implement your data collection logic
        return [{'data_type': 'custom', 'value': 100}]
```

### Custom Sinks

Create a custom sink:

```python
from data_pipeline.sinks.data_sink import BaseSink

class CustomSink(BaseSink):
    def write(self, events):
        # Implement your storage logic
        return True
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/peiyuan-tang/StockTrendEsimator/issues)
- Email: support@stocktrendestimator.com

## Roadmap

- [ ] Streaming data support (optional real-time)
- [ ] Data quality monitoring
- [ ] Automated data validation
- [ ] Machine learning pipeline integration
- [ ] Dashboard and visualization (future)
- [ ] Distributed processing (Spark integration)
- [ ] Kubernetes deployment support
- [ ] Data warehouse integration (Snowflake, BigQuery)

## Related Projects

- [Flume Python](https://github.com/apache/flume)
- [Pandas](https://pandas.pydata.org/)
- [YFinance](https://github.com/ranaroussi/yfinance)
