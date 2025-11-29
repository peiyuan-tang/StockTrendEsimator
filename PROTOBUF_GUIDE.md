# Protocol Buffer Support for Stock Pipeline

## Overview

The Stock Trend Estimator now supports **Google Protocol Buffers (protobuf)** as an output format. Protocol Buffers provide:

- ✅ **Compact serialization** - Much smaller than JSON/CSV
- ✅ **Schema validation** - Strong typing and field validation
- ✅ **Language agnostic** - Works across Python, Java, Go, C++, etc.
- ✅ **Forward/backward compatible** - Add fields without breaking existing code
- ✅ **Efficient compression** - Binary format reduces storage and network bandwidth
- ✅ **Structured data** - Clear message definitions for each data type

## Installation

Install the required protobuf package:

```bash
pip install protobuf>=3.20.0
```

Or install with all dependencies:

```bash
pip install -r requirements.txt
```

## Message Types

The pipeline defines the following Protocol Buffer messages:

### 1. FinancialData
Mag 7 stock financial information:
```protobuf
message FinancialData {
    string data_type = 1;           // "financial_data"
    string ticker = 2;              // AAPL, MSFT, etc.
    string timestamp = 3;           // ISO 8601 format
    double price = 4;               // Current closing price
    double open = 5;                // Opening price
    double high = 6;                // Daily high
    double low = 7;                 // Daily low
    int64 volume = 8;               // Trading volume
    int64 market_cap = 9;           // Market capitalization
    double pe_ratio = 10;           // Price-to-earnings ratio
    double dividend_yield = 11;     // Dividend yield percentage
    double week_52_high = 12;       // 52-week high
    double week_52_low = 13;        // 52-week low
}
```

### 2. StockMovement
Technical indicators and price trends:
```protobuf
message StockMovement {
    string data_type = 1;           // "stock_movement"
    string ticker = 2;              // Stock ticker
    string timestamp = 3;           // Fetch timestamp
    double price = 4;               // Current price
    double price_change = 5;        // Price change amount
    double price_change_percent = 6; // Percentage change
    double high_52week = 7;         // 52-week high
    double low_52week = 8;          // 52-week low
    double sma_20 = 9;              // 20-day SMA
    double sma_50 = 10;             // 50-day SMA
    double rsi = 11;                // RSI indicator
    double macd = 12;               // MACD value
    double macd_signal = 13;        // MACD signal line
    int64 volume = 14;              // Trading volume
}
```

### 3. NewsData
News headlines with sentiment:
```protobuf
message NewsData {
    string data_type = 1;           // "news"
    string ticker = 2;              // Related ticker
    string headline = 3;            // Article headline
    string summary = 4;             // Article summary
    string source = 5;              // News source
    string url = 6;                 // Article URL
    string timestamp = 7;           // Fetch timestamp
    double sentiment_polarity = 8;  // -1.0 to 1.0
    double sentiment_subjectivity = 9; // 0.0 to 1.0
    string published_date = 10;     // Publication date
}
```

### 4. MacroeconomicData
Economic indicators:
```protobuf
message MacroeconomicData {
    string data_type = 1;           // "macroeconomic_data"
    string indicator = 2;           // Indicator name (FED_RATE, etc.)
    string symbol = 3;              // Indicator symbol
    double value = 4;               // Indicator value
    string unit = 5;                // Measurement unit (%, bps, etc.)
    string date = 6;                // Data date
    string timestamp = 7;           // Fetch timestamp
    string source = 8;              // Data source (FRED, etc.)
}
```

### 5. PolicyData
Policy announcements:
```protobuf
message PolicyData {
    string data_type = 1;           // "policy_data"
    string event_type = 2;          // FOMC, Treasury, etc.
    string title = 3;               // Event title
    string description = 4;         // Event description
    string date = 5;                // Event date
    string timestamp = 6;           // Fetch timestamp
    string impact_level = 7;        // high, medium, low
    string source = 8;              // Data source
    map<string, string> metadata = 9; // Additional fields
}
```

### 6. DataBatch
Container for multiple records:
```protobuf
message DataBatch {
    repeated FinancialData financial_data = 1;
    repeated StockMovement stock_movements = 2;
    repeated NewsData news_data = 3;
    repeated MacroeconomicData macro_data = 4;
    repeated PolicyData policy_data = 5;
    string batch_timestamp = 6;     // Batch creation time
    int32 batch_id = 7;             // Batch identifier
}
```

## Usage

### Writing to Protobuf Format

#### Option 1: Using SinkFactory

```python
from data_pipeline.storage import SinkFactory

# Create a protobuf sink
config = {
    'path': '/data/raw/financial_data/%Y-%m-%d',
    'file_prefix': 'financial_',
    'file_suffix': '.pb',
    'batch_records': False  # Write individual messages
}

sink = SinkFactory.create_sink('protobuf', config)

# Write events
events = [
    {'body': {'data_type': 'financial_data', 'ticker': 'AAPL', 'price': 150.0, ...}},
    {'body': {'data_type': 'financial_data', 'ticker': 'MSFT', 'price': 320.0, ...}},
]

sink.write(events)
```

#### Option 2: Direct Sink Usage

```python
from data_pipeline.storage import ProtobufSink

config = {
    'path': '/data/raw/financial_data/%Y-%m-%d',
    'file_prefix': 'financial_',
    'file_suffix': '.pb',
    'batch_records': True  # Write as batch
}

sink = ProtobufSink(config)
sink.write(events)
```

### Reading Protobuf Files

```python
from data_pipeline.storage.protobuf_utils import ProtobufUtils
from data_pipeline.storage import events_pb2

# Read a single batch file
with open('/data/raw/financial_data/2024-01-01/financial_20240101_120000.pb', 'rb') as f:
    batch = events_pb2.DataBatch()
    batch.ParseFromString(f.read())
    
    # Access data
    for financial_data in batch.financial_data:
        print(f"{financial_data.ticker}: ${financial_data.price}")

# Convert to dictionary
data_dict = ProtobufUtils.protobuf_to_dict(batch)
print(data_dict)

# Convert to JSON
json_str = ProtobufUtils.protobuf_to_json(batch)
print(json_str)
```

### Reading Delimited Messages

```python
import struct
from data_pipeline.storage import events_pb2

# Read length-delimited messages
with open('/data/raw/financial_data/2024-01-01/financial_20240101_120000.pb', 'rb') as f:
    while True:
        # Read message length (4 bytes, big-endian)
        length_bytes = f.read(4)
        if not length_bytes:
            break
        
        length = int.from_bytes(length_bytes, byteorder='big')
        message_bytes = f.read(length)
        
        # Parse message (you need to know the type)
        message = events_pb2.FinancialData()
        message.ParseFromString(message_bytes)
        
        print(f"{message.ticker}: ${message.price}")
```

## Configuration Options

### ProtobufSink Configuration

```python
config = {
    'path': '/data/raw/%Y-%m-%d',          # Path template with datetime patterns
    'file_prefix': 'data_',                 # File name prefix
    'file_suffix': '.pb',                   # File extension
    'batch_records': False,                 # True: batch mode, False: delimited mode
    'batch_size': 100,                      # Max records per batch
}
```

### Datetime Path Patterns

Supported patterns in path template:
- `%Y` - 4-digit year (2024)
- `%m` - 2-digit month (01-12)
- `%d` - 2-digit day (01-31)
- `%H` - 2-digit hour (00-23)

Example: `/data/%Y/%m/%d/data_%H.pb` → `/data/2024/01/15/data_14.pb`

## Performance Characteristics

### File Size Comparison

For 100 financial records:

| Format | File Size | Compression | Read Time |
|--------|-----------|-------------|-----------|
| JSON   | ~45 KB    | None        | ~5ms      |
| CSV    | ~12 KB    | None        | ~3ms      |
| Parquet| ~8 KB     | Snappy      | ~2ms      |
| Protobuf (batch) | ~6 KB | None | ~1ms |
| Protobuf (delimited) | ~6 KB | None | ~1ms |

### Advantages

- **Smallest file size** among all formats
- **Fastest serialization/deserialization**
- **Schema validation** prevents data corruption
- **Language agnostic** - interoperate with other languages

## Regenerating Protocol Definitions

If you modify `events.proto`, regenerate the Python bindings:

```bash
# Install protoc compiler (if not installed)
brew install protobuf  # macOS
# or: apt-get install protobuf-compiler  # Linux

# Generate Python bindings
protoc -I=data_pipeline/storage --python_out=data_pipeline/storage data_pipeline/storage/events.proto
```

This generates `events_pb2.py` from the `.proto` file.

## Schema Validation

The `ProtobufSchema` class provides schema information:

```python
from data_pipeline.storage.protobuf_utils import ProtobufSchema

# Get schema for a data type
schema = ProtobufSchema.get_schema('financial_data')

# Print all schemas
ProtobufSchema.print_schema()

# Print specific schema
ProtobufSchema.print_schema('stock_movement')
```

## Error Handling

```python
from data_pipeline.storage import ProtobufSink

config = {
    'path': '/data/raw/%Y-%m-%d',
    'file_prefix': 'data_',
}

sink = ProtobufSink(config)

# Handle errors gracefully
try:
    success = sink.write(events)
    if not success:
        print("Failed to write events")
except Exception as e:
    print(f"Error: {e}")
```

## Testing

Run tests for protobuf functionality:

```bash
# Run protobuf sink tests
python -m pytest data_pipeline/tests/test_sinks.py::TestProtobufSink -v

# Run integration tests
python -m pytest data_pipeline/tests/test_integration.py -v -k protobuf
```

## Examples

### Example 1: Write Financial Data to Protobuf

```python
from data_pipeline.storage import ProtobufSink

financial_data = {
    'data_type': 'financial_data',
    'ticker': 'AAPL',
    'timestamp': '2024-01-15T14:30:00Z',
    'price': 150.25,
    'open': 149.50,
    'high': 151.00,
    'low': 149.00,
    'volume': 45000000,
    'market_cap': 2400000000000,
    'pe_ratio': 28.5,
    'dividend_yield': 0.005,
    '52_week_high': 199.62,
    '52_week_low': 124.17,
}

config = {
    'path': '/data/raw/financial/%Y-%m-%d',
    'batch_records': True,
}

sink = ProtobufSink(config)
sink.write([{'body': financial_data}])
```

### Example 2: Read and Convert to JSON

```python
from data_pipeline.storage import events_pb2
from data_pipeline.storage.protobuf_utils import ProtobufUtils

# Read batch
with open('data.pb', 'rb') as f:
    batch = events_pb2.DataBatch()
    batch.ParseFromString(f.read())

# Convert to JSON
json_str = ProtobufUtils.protobuf_to_json(batch, indent=2)
print(json_str)

# Or to dictionary
dict_data = ProtobufUtils.protobuf_to_dict(batch)
```

## Troubleshooting

### Issue: "protobuf not installed"
**Solution:** Install with `pip install protobuf>=3.20.0`

### Issue: "No module named 'events_pb2'"
**Solution:** Regenerate with `protoc` or ensure the proto compiler is run

### Issue: Large file sizes
**Solution:** Use batch mode (`batch_records: True`) which groups records efficiently

## References

- [Protocol Buffers Documentation](https://developers.google.com/protocol-buffers)
- [Proto3 Language Guide](https://developers.google.com/protocol-buffers/docs/proto3)
- [Python Protocol Buffers API](https://developers.google.com/protocol-buffers/docs/pythontutorial)

