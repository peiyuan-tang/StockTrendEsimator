# 📈 StockTrendEsimator

**Estimate Stock Moving Trends using Deep Learning**

A comprehensive machine learning framework for predicting stock price trends using dual-tower and LSTM architectures with multi-modal data integration (financial, macroeconomic, news sentiment, and policy indicators).

---

## ✨ Features

- **Dual-Tower Architecture**: Separate towers for context and stock data with cross-attention
- **LSTM Models**: Multi-horizon time-series forecasting with attention mechanisms
- **Multi-Modal Data Integration**: Financial, macro, news, policy, and movement data
- **Advanced Training**: Multi-task learning, regularization, and volatility-aware loss functions
- **Unified Package Structure**: Clean semantic organization (architectures, losses, data, trainers)
- **Production-Ready**: Full training loops, evaluation metrics, and inference support
- **Extensible Design**: Easy to add new data sources and model architectures

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/peiyuan-tang/StockTrendEsimator.git
cd StockTrendEsimator

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage - Dual Tower Model

```python
from data_pipeline.models import (
    create_dual_tower_model,
    create_dual_tower_data_loaders,
    create_dual_tower_optimizer,
    DualTowerTrainer,
    DualTowerLoss
)

# Create model
model = create_dual_tower_model()

# Create data loaders
train_loader, val_loader, test_loader = create_dual_tower_data_loaders(
    batch_size=32,
    train_split=0.7,
    val_split=0.15
)

# Create optimizer and loss
optimizer = create_dual_tower_optimizer(model, learning_rate=0.001)
loss_fn = DualTowerLoss()

# Train
trainer = DualTowerTrainer(model, optimizer, loss_fn)
trainer.train(train_loader, val_loader, num_epochs=50)

# Evaluate
metrics = trainer.evaluate(test_loader)
```

### Basic Usage - LSTM Model

```python
from data_pipeline.models import (
    create_lstm_model,
    create_lstm_data_loaders,
    create_lstm_optimizer,
    LSTMTrainer,
    LSTMLoss
)

# Similar pattern for LSTM
model = create_lstm_model()
train_loader, val_loader, test_loader = create_lstm_data_loaders(batch_size=32)
optimizer = create_lstm_optimizer(model, learning_rate=0.001)
loss_fn = LSTMLoss()

trainer = LSTMTrainer(model, optimizer, loss_fn)
trainer.train(train_loader, val_loader, num_epochs=50)
```

### Run Examples

```bash
# Dual tower examples
python examples/dual_tower_examples.py

# LSTM examples
python examples/lstm_examples.py

# Data pipeline examples
python examples/pipeline_examples.py

# Training data examples
python examples/training_data_examples.py
```

---

## 📁 Project Structure

```
StockTrendEsimator/
├── data_pipeline/                      # Main data pipeline package
│   ├── client/                         # API clients
│   ├── config/                         # Configuration management
│   ├── core/                           # Core utilities
│   ├── integrations/                   # External integrations
│   ├── models/                         # Data sources & ML models
│   │   ├── ml_models/                  # ← Unified ML models (SEMANTIC ORGANIZATION)
│   │   │   ├── architectures/          # Model definitions (DualTower, LSTM)
│   │   │   ├── losses/                 # Loss functions
│   │   │   ├── data_loaders/           # Dataset & data loaders
│   │   │   ├── trainers/               # Training loops & optimizers
│   │   │   ├── model_configs.py        # Centralized configuration
│   │   │   └── __init__.py             # Unified exports
│   │   ├── financial_source.py         # Financial data source
│   │   ├── macro_source.py             # Macro indicators source
│   │   ├── movement_source.py          # Stock movement source
│   │   ├── news_source.py              # News sentiment source
│   │   ├── policy_source.py            # Policy data source
│   │   └── __init__.py
│   ├── server/                         # Server/API components
│   ├── sinks/                          # Data sinks
│   ├── sources/                        # Data sources
│   ├── storage/                        # Storage layer
│   ├── tests/                          # Unit tests
│   ├── utils/                          # Utilities
│   └── __init__.py
├── examples/                           # Example scripts
│   ├── dual_tower_examples.py          # Dual tower model examples
│   ├── lstm_examples.py                # LSTM model examples
│   ├── pipeline_examples.py            # Data pipeline examples
│   └── training_data_examples.py       # Training data examples
├── modelling/                          # ← Backward compatibility shim
│   ├── __init__.py                     # Re-exports from data_pipeline.models
│   ├── README.md                       # Legacy documentation
│   └── [archived directories]
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup
├── LICENSE                             # MIT License
└── README.md                           # This file
```

---

## 🏗️ Architecture Overview

### Dual-Tower Model

The dual-tower architecture processes context and stock data through separate pathways with cross-attention:

```
Context Data               Stock Data
    │                         │
    ▼                         ▼
┌─────────────────┐     ┌─────────────────┐
│ Context Tower   │     │ Stock Tower     │
│ - Embeddings    │     │ - Embeddings    │
│ - LSTM Layers   │     │ - LSTM Layers   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │ Attention       │
            │ Mechanism       │
            └────────┬────────┘
                     ▼
            ┌─────────────────┐
            │ Fusion Layer    │
            │ & Prediction    │
            └────────┬────────┘
                     ▼
              Stock Trend
              Prediction
```

### LSTM Model

Multi-horizon forecasting with attention mechanisms:

```
Time Series Data
    │
    ▼
┌──────────────┐
│ LSTM         │
│ Encoder      │
│ (Bidirectional)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Attention    │
│ Module       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Prediction   │
│ Heads        │
│ (Multi-      │
│  Horizon)    │
└──────┬───────┘
       │
       ▼
  Forecasts
  (1, 5, 10, 20 days)
```

---

## 📊 Data Sources

### Integrated Data Streams

1. **Financial Data** (`FinancialDataSource`)
   - OHLCV data
   - Technical indicators
   - Volume patterns

2. **Macroeconomic Data** (`MacroDataSource`)
   - Interest rates
   - GDP growth
   - Inflation rates
   - Market indices

3. **Stock Movement** (`StockMovementSource`)
   - Historical prices
   - Returns
   - Volatility metrics

4. **News Sentiment** (`NewsDataSource`)
   - Sentiment scores
   - News frequency
   - Relevance weights

5. **Policy Data** (`PolicyDataSource`)
   - Policy announcements
   - Regulatory changes
   - Impact indicators

---

## 🔧 API Reference

### Model Creation

```python
from data_pipeline.models import (
    create_dual_tower_model,
    create_lstm_model,
)

# Dual Tower
model = create_dual_tower_model(config=None)

# LSTM
model = create_lstm_model(config=None)
```

### Data Loading

```python
from data_pipeline.models import (
    create_dual_tower_data_loaders,
    create_lstm_data_loaders,
)

# Dual Tower
train_loader, val_loader, test_loader = create_dual_tower_data_loaders(
    batch_size=32,
    train_split=0.7,
    val_split=0.15,
    data_path=None
)

# LSTM
loaders = create_lstm_data_loaders(batch_size=32, sequence_length=20)
```

### Training

```python
from data_pipeline.models import DualTowerTrainer, LSTMTrainer

# Dual Tower
trainer = DualTowerTrainer(model, optimizer, loss_fn, device='cuda')
trainer.train(train_loader, val_loader, num_epochs=50, early_stopping=True)

# LSTM
trainer = LSTMTrainer(model, optimizer, loss_fn, device='cuda')
trainer.train(train_loader, val_loader, num_epochs=50)
```

### Configuration

```python
from data_pipeline.models import (
    ConfigManager,
    DualTowerModelConfig,
    LSTMModelConfig,
    TrainingConfig,
)

# Load/save configurations
config_mgr = ConfigManager()
config = config_mgr.load_config('path/to/config.yaml')

# Create custom config
model_config = DualTowerModelConfig(
    context_dim=256,
    stock_dim=128,
    attention_heads=4,
    num_layers=3,
)

training_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    num_epochs=50,
    early_stopping_patience=10,
)
```

---

## 📚 Documentation

### Main Guides
- **[Unified Package Structure](UNIFICATION_COMPLETE.md)** - Complete guide to the unified import structure
- **[Dual Tower Quick Start](DUAL_TOWER_QUICK_START.md)** - Get started with dual tower model
- **[LSTM Guide](LSTM_MODEL_GUIDE.md)** - Comprehensive LSTM documentation
- **[Quick Reference](QUICK_REFERENCE.md)** - Handy reference for common operations

### Architecture & Design
- **[Architecture Overview](ARCHITECTURE.md)** - System architecture
- **[Architecture Diagram](ARCHITECTURE_DIAGRAM.md)** - Visual architecture
- **[Data Pipeline](DATA_PIPELINE.md)** - Data flow documentation

### Implementation Details
- **[Dual Tower Implementation](DUAL_TOWER_IMPLEMENTATION_SUMMARY.md)** - Implementation details
- **[LSTM Implementation](LSTM_IMPLEMENTATION_SUMMARY.md)** - LSTM details
- **[Training Data Guide](TRAINING_DATA_QUICK_START.md)** - Training data management

---

## 🎯 Import Patterns

### Recommended: Unified Imports

```python
# From unified package
from data_pipeline.models import (
    DualTowerRelevanceModel,
    LSTMRelevanceModel,
    DualTowerLoss,
    LSTMLoss,
    create_dual_tower_model,
    create_lstm_model,
    ConfigManager,
)
```

### Granular Imports

```python
# Direct from semantic subdirectories
from data_pipeline.models.ml_models.architectures import DualTowerRelevanceModel
from data_pipeline.models.ml_models.losses import DualTowerLoss
from data_pipeline.models.ml_models.trainers import DualTowerTrainer
from data_pipeline.models.ml_models.data_loaders import create_dual_tower_data_loaders
```

### Backward Compatibility

```python
# Old imports still work (re-exported for compatibility)
from modelling import DualTowerRelevanceModel, DualTowerLoss
```

---

## �� Example Usage

### Complete Training Pipeline

```python
import torch
from data_pipeline.models import (
    create_dual_tower_model,
    create_dual_tower_data_loaders,
    create_dual_tower_optimizer,
    create_dual_tower_scheduler,
    DualTowerLoss,
    DualTowerTrainer,
    ConfigManager,
)

# 1. Load configuration
config_mgr = ConfigManager()
model_config = config_mgr.get_default_dual_tower_config()
training_config = config_mgr.get_default_training_config()

# 2. Create model
model = create_dual_tower_model(model_config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 3. Create data
train_loader, val_loader, test_loader = create_dual_tower_data_loaders(
    batch_size=training_config.batch_size,
    train_split=0.7,
)

# 4. Create optimizer and scheduler
optimizer = create_dual_tower_optimizer(model, training_config.learning_rate)
scheduler = create_dual_tower_scheduler(optimizer, training_config.num_epochs)

# 5. Create loss and trainer
loss_fn = DualTowerLoss()
trainer = DualTowerTrainer(model, optimizer, loss_fn, device=device)

# 6. Train
history = trainer.train(
    train_loader,
    val_loader,
    num_epochs=training_config.num_epochs,
    early_stopping=True,
    early_stopping_patience=10,
)

# 7. Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Test Metrics: {metrics}")

# 8. Save model
torch.save(model.state_dict(), 'model.pth')
```

---

## 🧪 Testing

```bash
# Run tests
python -m pytest data_pipeline/tests/

# Run specific test
python -m pytest data_pipeline/tests/test_models.py

# With coverage
python -m pytest --cov=data_pipeline data_pipeline/tests/
```

---

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- scikit-learn
- See `requirements.txt` for complete list

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Support

For issues, questions, or suggestions, please open an issue on the [GitHub repository](https://github.com/peiyuan-tang/StockTrendEsimator/issues).

---

## 🗂️ Project Status

- ✅ Dual-Tower Architecture
- ✅ LSTM Architecture
- ✅ Multi-Modal Data Integration
- ✅ Training Infrastructure
- ✅ Unified Package Structure
- ✅ Complete Documentation
- 🚀 Production Ready

---

**Last Updated**: December 2025  
**Version**: 1.0.0
