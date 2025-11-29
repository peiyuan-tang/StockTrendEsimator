#!/usr/bin/env bash
# Quick Start Script for Stock Trend Estimator Pipeline
# Run this to set up the environment quickly

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   Stock Trend Estimator - Data Pipeline Quick Start       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "✓ Checking Python version..."
python3 --version

# Create virtual environment
echo "✓ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "✓ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "✓ Creating data directories..."
mkdir -p /data/raw/{financial_data,stock_movements,news}
mkdir -p /data/context/{macroeconomic,policy}
mkdir -p /var/data/{checkpoint,queue}/{movement,macro,policy}
mkdir -p /var/log/stock_pipeline
mkdir -p examples

# Create credentials template
echo "✓ Creating credentials template..."
if [ ! -f "data_pipeline/config/credentials.json" ]; then
    cat > data_pipeline/config/credentials.json << 'EOF'
{
  "finnhub_api_key": "your_finnhub_key_here",
  "newsapi_key": "your_newsapi_key_here",
  "alpha_vantage_key": "your_alpha_vantage_key_here",
  "fred_api_key": "your_fred_api_key_here"
}
EOF
    echo "  Credentials template created at data_pipeline/config/credentials.json"
    echo "  ⚠️  Please update with your API keys!"
fi

# Create pipeline config template
echo "✓ Creating pipeline config template..."
if [ ! -f "data_pipeline/config/pipeline_config.json" ]; then
    cat > data_pipeline/config/pipeline_config.json << 'EOF'
{
  "mag7_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
  "sp500_limit": 100,
  "financial_data_interval": 3600,
  "stock_movement_interval": 3600,
  "news_interval": 3600,
  "macro_interval": 86400,
  "policy_interval": 604800,
  "data_root_path": "/data",
  "backup_enabled": true,
  "retention_days": 90,
  "batch_size_financial": 100,
  "batch_size_movement": 500,
  "batch_size_news": 50,
  "batch_size_macro": 10,
  "log_level": "INFO",
  "log_path": "/var/log/stock_pipeline"
}
EOF
    echo "  Pipeline config created at data_pipeline/config/pipeline_config.json"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Setup Complete! Next Steps:                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "1. Update API credentials:"
echo "   nano data_pipeline/config/credentials.json"
echo ""
echo "2. Start the data pipeline:"
echo "   python data_pipeline/server/flume_server.py"
echo ""
echo "3. Query collected data (in another terminal):"
echo "   python examples/pipeline_examples.py"
echo ""
echo "4. View documentation:"
echo "   - Full guide: cat DATA_PIPELINE.md"
echo "   - Architecture: cat ARCHITECTURE.md"
echo "   - Operations: cat OPERATIONS.md"
echo ""
echo "Get API Keys:"
echo "  • Finnhub: https://finnhub.io/"
echo "  • NewsAPI: https://newsapi.org/"
echo "  • Alpha Vantage: https://www.alphavantage.co/"
echo "  • FRED: https://fredaccount.stlouisfed.org/apikey"
echo ""
