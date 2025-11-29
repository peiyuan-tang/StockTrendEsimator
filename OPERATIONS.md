# Stock Trend Estimator - Operational Guide

## Quick Start

### 1. Initial Setup

```bash
# Clone and setup
git clone https://github.com/peiyuan-tang/StockTrendEsimator.git
cd StockTrendEsimator

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p /data/raw/{financial_data,stock_movements,news}
mkdir -p /data/context/{macroeconomic,policy}
mkdir -p /var/log/stock_pipeline
```

### 2. Configure API Keys

Create `data_pipeline/config/credentials.json`:

```json
{
  "finnhub_api_key": "your_finnhub_key",
  "newsapi_key": "your_newsapi_key",
  "alpha_vantage_key": "your_alpha_vantage_key",
  "fred_api_key": "your_fred_api_key"
}
```

### 3. Start the Pipeline

```bash
# Run with default configuration
python data_pipeline/server/flume_server.py

# Run with custom config
python data_pipeline/server/flume_server.py \
  --config custom_config.yaml \
  --log-level DEBUG

# Run as background service
nohup python data_pipeline/server/flume_server.py > /var/log/stock_pipeline/pipeline.log 2>&1 &
```

## Daily Operations

### Monitor Data Collection

```bash
# View latest logs
tail -f /var/log/stock_pipeline/pipeline.log

# Check pipeline status
python -c "
from data_pipeline.server.flume_server import StockDataCollector
collector = StockDataCollector('data_pipeline/config/flume_config.yaml')
print(collector.get_status())
"
```

### Query Collected Data

```python
from data_pipeline.client.pipeline_client import get_data_client
from datetime import datetime, timedelta

client = get_data_client()

# Get today's financial data
df = client.get_financial_data()

# Get last week's movements
end = datetime.utcnow()
start = end - timedelta(days=7)
df = client.get_stock_movements(start_date=start, end_date=end)

# Get positive news
df = client.get_news_data(sentiment_filter=(0.5, 1.0))

# Export to CSV
client.export_data('financial_data', 'export.csv', format='csv')
```

### Data Backup

```bash
# Backup all collected data
tar -czf backup_data_$(date +%Y%m%d).tar.gz /data/

# Backup configuration
tar -czf backup_config_$(date +%Y%m%d).tar.gz data_pipeline/config/

# Archive old data (>90 days)
find /data/raw -name "*.parquet" -mtime +90 -exec gzip {} \;
```

## Maintenance Tasks

### Weekly

```bash
# Clean old temporary files
find /var/data/queue -type f -mtime +7 -delete

# Verify data integrity
python -c "
from data_pipeline.client.pipeline_client import get_data_client
client = get_data_client()
summary = client.get_data_summary()
print(f'Data size: {summary[\"total_size_bytes\"] / (1024**3):.2f} GB')
"

# Check for collection errors
grep ERROR /var/log/stock_pipeline/pipeline.log | tail -20
```

### Monthly

```bash
# Archive old data
find /data -name "*.parquet" -mtime +30 -exec mv {} {}.gz \;

# Generate data quality report
python scripts/data_quality_report.py

# Update configuration and API keys if needed
nano data_pipeline/config/credentials.json

# Test all data sources
python -m pytest data_pipeline/tests/
```

### Quarterly

```bash
# Full system health check
- Review collection statistics
- Check error rates and logs
- Verify data quality metrics
- Test disaster recovery procedures
- Update dependencies

# Generate quarterly report
python scripts/generate_quarterly_report.py
```

## Troubleshooting

### Problem: No data being collected

**Check 1: Verify pipeline is running**
```bash
ps aux | grep flume_server.py
```

**Check 2: Verify API keys**
```bash
python -c "
from data_pipeline.config.config_manager import get_config_manager
config = get_config_manager()
print(config.get_api_keys())
"
```

**Check 3: Check data directories**
```bash
ls -la /data/raw/
```

**Check 4: Review logs**
```bash
tail -100 /var/log/stock_pipeline/pipeline.log | grep ERROR
```

### Problem: High memory usage

**Solution 1: Reduce batch sizes**
```yaml
# In flume_config.yaml
batch_size: 50  # Reduce from 100
```

**Solution 2: Use file channels instead of memory**
```yaml
channels:
  financial_channel:
    type: file  # Change from memory
```

**Solution 3: Limit S&P 500 stocks processed**
```python
# In movement_source.py
tickers = tickers[:50]  # Reduce from 500
```

### Problem: Slow data loading from client

**Solution 1: Use Parquet format**
```python
client.export_data('financial_data', 'output.parquet', format='parquet')
```

**Solution 2: Filter by date range**
```python
start = datetime(2024, 1, 1)
end = datetime(2024, 1, 7)
df = client.get_financial_data(start_date=start, end_date=end)
```

**Solution 3: Process in chunks**
```python
for date in date_range:
    df = client.get_financial_data(
        start_date=date,
        end_date=date + timedelta(days=1)
    )
    # Process chunk
```

## Performance Tuning

### Optimize Collection

```python
# In flume_config.yaml
sources:
  financial_api_source:
    batch_size: 100      # Increase for more throughput
    timeout: 60          # Increase if timeouts occur
    retry_attempts: 5    # More retries for reliability

channels:
  financial_channel:
    capacity: 5000       # Increase buffer size
    transactionCapacity: 200
```

### Optimize Storage

```yaml
sinks:
  financial_sink:
    compression: snappy  # Faster than gzip
    batch_size: 500      # Larger batches
```

### Optimize Client Queries

```python
# Use date filters
df = client.get_financial_data(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 1, 31)
)

# Use ticker filters
df = client.get_news_data(tickers=['AAPL', 'MSFT'])

# Cache frequently accessed data
import pickle
cache = pickle.dump(df, open('cache.pkl', 'wb'))
```

## Integration Examples

### With Machine Learning Pipeline

```python
from data_pipeline.client.pipeline_client import get_data_client
from sklearn.preprocessing import StandardScaler
import numpy as np

client = get_data_client()

# Get training data
financial_df = client.get_financial_data()
movements_df = client.get_stock_movements()
macro_df = client.get_macroeconomic_data()

# Merge datasets
combined = financial_df.merge(movements_df, on='ticker')
combined = combined.merge(macro_df, on=['timestamp'])

# Prepare features
scaler = StandardScaler()
X = scaler.fit_transform(combined[['price', 'volume', 'RSI', ...]])
y = combined['price_change']

# Train model
# model.fit(X, y)
```

### With Data Warehouse

```python
import psycopg2

client = get_data_client()
conn = psycopg2.connect("dbname=stock_db user=postgres")

# Export to PostgreSQL
financial_df = client.get_financial_data()
for _, row in financial_df.iterrows():
    # Insert into database
    pass

conn.close()
```

## Automation

### Cron Jobs

```bash
# Add to crontab (crontab -e)

# Daily: Full data backup at 2 AM
0 2 * * * tar -czf /backups/data_$(date +\%Y\%m\%d).tar.gz /data/

# Weekly: Data quality check every Monday at 3 AM
0 3 * * 1 python /home/user/scripts/quality_check.py

# Monthly: Archive old data first day at 4 AM
0 4 1 * * python /home/user/scripts/archive_old_data.py

# Every 6 hours: Check pipeline status
0 */6 * * * curl http://localhost:5000/status
```

### Systemd Service

Create `/etc/systemd/system/stock-pipeline.service`:

```ini
[Unit]
Description=Stock Trend Estimator Data Pipeline
After=network.target

[Service]
Type=simple
User=pipeline
WorkingDirectory=/home/pipeline/StockTrendEsimator
ExecStart=/home/pipeline/StockTrendEsimator/venv/bin/python \
  data_pipeline/server/flume_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable stock-pipeline
sudo systemctl start stock-pipeline
sudo systemctl status stock-pipeline
```

## Monitoring & Alerting

### Key Metrics to Monitor

1. **Collection Success Rate**
   - Target: >95%
   - Alert: <90%

2. **Data Freshness**
   - Latest data age
   - Alert: >2 hours old

3. **API Response Time**
   - Target: <5 seconds
   - Alert: >10 seconds

4. **Storage Usage**
   - Monitor: /data usage
   - Alert: >80% disk full

5. **Error Rate**
   - Target: <1% errors
   - Alert: >5% errors

### Monitoring Script

```python
# scripts/monitor_pipeline.py
from data_pipeline.client.pipeline_client import get_data_client
from datetime import datetime, timedelta
import json

def check_pipeline_health():
    client = get_data_client()
    summary = client.get_data_summary()
    
    now = datetime.utcnow()
    alert_time = now - timedelta(hours=2)
    
    metrics = {
        'timestamp': now.isoformat(),
        'data_sources': summary['data_sources'],
        'storage_gb': summary['total_size_bytes'] / (1024**3),
        'status': 'healthy'
    }
    
    if metrics['storage_gb'] > 100:
        metrics['status'] = 'warning'
        # Send alert
    
    return metrics

if __name__ == '__main__':
    health = check_pipeline_health()
    print(json.dumps(health, indent=2))
```

## Disaster Recovery

### Backup Strategy

```bash
# Daily incremental backup
tar -czf /backups/daily_$(date +%Y%m%d).tar.gz /data/

# Weekly full backup
tar -czf /backups/weekly_$(date +%Y_W%V).tar.gz /data/

# Store off-site
aws s3 cp /backups/weekly_*.tar.gz s3://stock-backup/
```

### Recovery Procedure

```bash
# 1. Stop pipeline
systemctl stop stock-pipeline

# 2. Restore from backup
tar -xzf /backups/weekly_2024_W10.tar.gz -C /

# 3. Verify data integrity
python scripts/verify_backup.py

# 4. Restart pipeline
systemctl start stock-pipeline
```

## Support & Resources

- **Documentation**: [DATA_PIPELINE.md](DATA_PIPELINE.md)
- **Examples**: `examples/pipeline_examples.py`
- **Tests**: `data_pipeline/tests/`
- **Issues**: GitHub Issues
- **Community**: GitHub Discussions
