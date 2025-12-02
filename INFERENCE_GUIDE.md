# Stock Trend Prediction - Inference Guide

## Overview

The inference script (`predict_trend.py`) uses trained LSTM and Dual Tower models to predict the next monthly stock trend for a given ticker symbol. It supports multiple prediction modes and output formats.

---

## Installation & Setup

### Prerequisites
```bash
pip install torch pandas numpy
```

### Verify Models Exist
```bash
# Check for trained model checkpoints
ls -la models/
# Output should show:
# - lstm_best.pth (230 KB)
# - dual_tower_best.pth (14 KB)
# - training_summary.json
```

---

## Usage

### Basic Prediction (Both Models)
```bash
python3 predict_trend.py AAPL
```

Output shows predictions from both LSTM and Dual Tower models with confidence scores and probability distributions.

### LSTM Model Only
```bash
python3 predict_trend.py MSFT --model lstm
```

Uses only the LSTM model (requires sequence data).

### Dual Tower Model Only
```bash
python3 predict_trend.py GOOGL --model dual_tower
```

Uses only the Dual Tower model (uses flat features).

### Ensemble Prediction
```bash
python3 predict_trend.py TSLA --ensemble
```

Combines predictions from both models using weighted voting. Often more robust than individual models.

### JSON Output
```bash
python3 predict_trend.py NVDA --json
```

Returns results in JSON format suitable for programmatic use or API responses.

### Custom Sample Size
```bash
python3 predict_trend.py META --samples 10
```

Generate 10 samples instead of default 5.

---

## Command-Line Options

```
positional arguments:
  ticker              Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)

optional arguments:
  -h, --help         Show help message
  --model {lstm, dual_tower, both}
                     Model to use (default: both)
  --ensemble         Use ensemble prediction (combines both models)
  --json             Output results in JSON format
  --samples N        Number of samples to generate (default: 5)
```

---

## Prediction Output

### Standard Output Format

```
================================================================================
STOCK TREND PREDICTION - AAPL
================================================================================
Prediction Date: 2025-12-01 21:00:23
Training Period: November 1-30, 2024
Forecast: Next Monthly Trend

🤖 Model: LSTM
────────────────────────────────────────────────────────────────────────────
Predicted Trend: ➡️  Neutral
Confidence: 35.28%

Probability Distribution:
  Downtrend: 33.33%  ▓█████████
  Neutral:   35.28%  ▓██████████
  Uptrend:   31.39%  ▓█████████

🤖 Model: Dual Tower
────────────────────────────────────────────────────────────────────────────
Predicted Trend: 📈 Uptrend
Confidence: 100.00%

Probability Distribution:
  Downtrend:  0.00%  ▓
  Neutral:    0.00%  ▓
  Uptrend:   100.00%  ▓██████████████████████████████
```

### Trend Emoji Guide
- 📉 Downtrend: Stock expected to decline
- ➡️ Neutral: Stock expected to stay relatively flat
- 📈 Uptrend: Stock expected to increase

### Confidence Score
- Higher confidence indicates stronger prediction signal
- Range: 0% (uncertain) to 100% (certain)
- Confidence = model's predicted probability for top class

---

## JSON Output Format

```json
{
  "model": "LSTM",
  "prediction": "Neutral",
  "pred_class": 1,
  "probabilities": {
    "downtrend": 0.3333,
    "neutral": 0.3528,
    "uptrend": 0.3139
  },
  "confidence": 0.3528
}
```

### Ensemble JSON Format
```json
{
  "model": "Ensemble",
  "prediction": "Uptrend",
  "pred_class": 2,
  "probabilities": {
    "downtrend": 0.1666,
    "neutral": 0.1764,
    "uptrend": 0.6570
  },
  "confidence": 0.6570,
  "component_predictions": {
    "lstm": "Neutral",
    "dual_tower": "Uptrend"
  }
}
```

---

## Model Information

### LSTM Model (SimpleLSTM)
- **Architecture**: 2-layer LSTM (64 hidden units)
- **Parameters**: 57,731
- **Input**: Sequence of 3 samples × 21 features
- **Output**: 3 class probabilities (Downtrend, Neutral, Uptrend)
- **Strengths**: Captures temporal patterns, good for time series
- **Use When**: You have sequential historical data

### Dual Tower Model (DualTowerModel)
- **Architecture**: 2 separate towers + interaction layer
- **Parameters**: 2,371 (compact)
- **Input**: Flat 21 features
- **Output**: 3 class probabilities
- **Strengths**: Fast inference, interpretable feature separation
- **Use When**: You need quick predictions with clear feature contribution

### Ensemble
- **Method**: Weighted voting (weighted by confidence)
- **Robustness**: More stable predictions
- **Speed**: Slightly slower (runs both models)
- **Accuracy**: Often better than individual models

---

## Features Used for Prediction

### Stock Features (8)
Technical indicators and price metrics:
- `stock_open`, `stock_high`, `stock_low`, `stock_close`
- `stock_volume`, `stock_sma_20`, `stock_sma_50`, `stock_rsi`
- `stock_macd`, `stock_bb_upper`, `stock_bb_lower`

### News Features (3)
Market sentiment and news signals:
- `news_sentiment` (range: -1 to 1)
- `news_count` (number of articles)
- `news_relevance` (0 to 1)

### Macroeconomic Features (5)
Broad economy indicators:
- `macro_gdp_growth` (%)
- `macro_inflation` (%)
- `macro_unemployment` (%)
- `macro_interest_rate` (%)
- `macro_vix` (volatility index)

### Policy Features (2)
Government/policy signals:
- `policy_event_count` (number of events)
- `policy_impact_score` (-1 to 1)

---

## Supported Tickers

The models were trained on Magnificent 7 (Mag-7) stocks:

| Ticker | Company |
|--------|---------|
| AAPL | Apple |
| MSFT | Microsoft |
| GOOGL | Alphabet (Google) |
| AMZN | Amazon |
| NVDA | NVIDIA |
| META | Meta Platforms |
| TSLA | Tesla |

**Note**: The script will work with any ticker, but predictions are most reliable for these 7 stocks (trained data).

---

## Example Use Cases

### 1. Quick Trend Check
```bash
python3 predict_trend.py AAPL
```
Get a quick prediction with both models showing confidence.

### 2. Production API
```bash
python3 predict_trend.py MSFT --ensemble --json
```
Use ensemble with JSON output for API responses.

### 3. Detailed Analysis
```bash
python3 predict_trend.py GOOGL --model lstm
python3 predict_trend.py GOOGL --model dual_tower
```
Compare individual model predictions for analysis.

### 4. Batch Processing
```bash
for ticker in AAPL MSFT GOOGL AMZN NVDA META TSLA; do
  python3 predict_trend.py $ticker --ensemble --json
done
```
Process multiple stocks and collect results.

---

## Integration Examples

### Python Script
```python
import subprocess
import json

def predict_stock_trend(ticker):
    """Get trend prediction for a stock"""
    result = subprocess.run(
        ['python3', 'predict_trend.py', ticker, '--ensemble', '--json'],
        capture_output=True,
        text=True
    )
    
    # Parse JSON output (skip log lines)
    for line in result.stdout.split('\n'):
        if line.strip().startswith('{'):
            return json.loads(line)
    
    return None

# Usage
prediction = predict_stock_trend('AAPL')
print(f"AAPL trend: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

### Flask API
```python
from flask import Flask, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/predict/<ticker>')
def predict(ticker):
    result = subprocess.run(
        ['python3', 'predict_trend.py', ticker.upper(), '--ensemble', '--json'],
        capture_output=True,
        text=True
    )
    
    for line in result.stdout.split('\n'):
        if line.strip().startswith('{'):
            return jsonify(json.loads(line))
    
    return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Important Notes

### Limitations
1. **Synthetic Data**: Currently uses generated synthetic features (for demo)
   - In production, connect to real stock data APIs (Alpha Vantage, Yahoo Finance, etc.)
   
2. **Training Period**: Models trained on November 2024 data
   - Performance may vary for other time periods
   
3. **Sample Size**: Limited training data (35 samples)
   - Ensemble predictions may be more reliable than individual models
   
4. **Market Changes**: Models don't capture structural market changes
   - Retraining recommended quarterly or when performance degrades

### Production Improvements
1. Connect to real-time data sources
2. Implement feature normalization/scaling
3. Add confidence intervals and prediction bounds
4. Monitor and log all predictions
5. Implement A/B testing between ensemble and individual models
6. Set up automated retraining pipeline

---

## Troubleshooting

### Model Not Found Error
```
❌ Failed to load any models
```
**Solution**: Ensure model checkpoint files exist in `models/` directory
```bash
ls models/lstm_best.pth models/dual_tower_best.pth
```

### Insufficient Samples for LSTM
```
⚠ Not enough samples for LSTM (need 3+)
```
**Solution**: Use `--samples` option with larger value
```bash
python3 predict_trend.py AAPL --samples 10
```

### JSON Parse Error
```
ValueError: No JSON object could be decoded
```
**Solution**: Check that `--json` flag is used correctly
```bash
python3 predict_trend.py AAPL --json  # Correct
python3 predict_trend.py AAPL json    # Incorrect (missing --)
```

---

## Performance Benchmarks

### Inference Speed
- **LSTM**: ~5-10 ms per prediction
- **Dual Tower**: ~1-2 ms per prediction
- **Ensemble**: ~6-12 ms per prediction

### Model Accuracy (on validation data)
- **LSTM**: 33.33% validation accuracy
- **Dual Tower**: 25.00% validation accuracy
- **Ensemble**: Variable (depends on agreement)

---

## Next Steps

1. **Connect Real Data**: Replace synthetic data with live feeds
2. **Monitor Performance**: Track prediction accuracy over time
3. **Fine-tune Models**: Retrain with more data
4. **Add Features**: Incorporate additional data sources
5. **Deploy API**: Create REST API for production use
6. **Set Alerts**: Create alerts for strong signals

---

## References

- Model Training: `train_models.py`
- Data Generation: `generate_data_pure_python.py`
- Model Architectures: `train_models.py` (SimpleLSTM, DualTowerModel)
- Training Summary: `models/training_summary.json`
