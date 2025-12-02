# LSTM Model - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies (Already Included)
```bash
# PyTorch, numpy, pandas, scikit-learn
```

### 2. Import Model Components
```python
from modelling.ml_models import (
    create_lstm_model,
    create_lstm_data_loaders,
    LSTMLoss,
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
)
```

### 3. Prepare Your Data
```python
import pandas as pd

# Data format: datetime index, 62 features + 2 labels
data = pd.DataFrame({
    'feature_0': [...],
    ...
    'feature_61': [...],
    'trend_7day': [...],   # Target: [-1, 1]
    'trend_30day': [...],  # Target: [-1, 1]
}, index=pd.date_range('2015-01-01', periods=500, freq='W'))
```

### 4. Create Data Loaders
```python
train_loader, val_loader, test_loader = create_lstm_data_loaders(
    data,
    sequence_length=12,
    batch_size=32,
    normalize=True,
    time_aware_split=True,
)
```

### 5. Create Model
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = create_lstm_model(
    input_dim=62,
    hidden_dim=128,
    num_lstm_layers=2,
    num_attention_heads=4,
    device=device
)
```

### 6. Setup Training
```python
optimizer = create_lstm_optimizer(model, learning_rate=0.001)
scheduler = create_lstm_scheduler(optimizer, total_epochs=100)
loss_fn = LSTMLoss()

trainer = LSTMTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)
```

### 7. Train Model
```python
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    early_stopping_patience=15,
)
```

### 8. Evaluate & Predict
```python
test_metrics = trainer.validate(test_loader)
print(f"Test MAE (7-day): {test_metrics['mae_7day']:.6f}")
print(f"Direction Accuracy: {test_metrics['direction_acc_7day']:.4f}")
```

---

## Common Tasks

### Adjust Model Complexity

```python
# Smaller model (faster, less memory)
model = create_lstm_model(
    hidden_dim=64,          # 128 → 64
    num_lstm_layers=1,      # 2 → 1
    num_attention_heads=2,  # 4 → 2
    device=device
)

# Larger model (slower, more capacity)
model = create_lstm_model(
    hidden_dim=256,         # 128 → 256
    num_lstm_layers=3,      # 2 → 3
    num_attention_heads=8,  # 4 → 8
    device=device
)
```

### Change Learning Rate
```python
optimizer = create_lstm_optimizer(model, learning_rate=0.0001)  # Lower = slower but stable
```

### Modify Sequence Length
```python
train_loader, val_loader, test_loader = create_lstm_data_loaders(
    data,
    sequence_length=24,  # 12 → 24 weeks of history
    batch_size=32,
)
```

### Load Trained Model
```python
trainer.load_checkpoint('checkpoints/lstm/best_model.pt')
```

### Visualize Attention
```python
# See LSTM_MODEL_GUIDE.md for detailed attention analysis
# Or run: python examples/lstm_examples.py
```

---

## Configuration Files

All configs are in `/modelling/configs/`:

```python
from modelling.configs import (
    LSTMModelConfig,
    LSTMTrainingConfig,
    LSTMSequenceConfig,
)

# Create and customize configs
model_config = LSTMModelConfig(hidden_dim=256)
training_config = LSTMTrainingConfig(batch_size=64)
```

---

## File Structure

```
/modelling/
├── ml_models/
│   ├── lstm_model.py        # Model architecture
│   ├── lstm_loss.py         # Loss functions
│   ├── lstm_data.py         # Data loading
│   ├── lstm_trainer.py      # Training loop
│   └── __init__.py          # Exports
├── configs/
│   ├── model_configs.py     # Config classes (includes LSTM)
│   └── __init__.py          # Config exports
└── __init__.py              # Module-level exports

/examples/
└── lstm_examples.py         # Complete examples (4 scenarios)
```

---

## Key Metrics

| Metric | What It Measures | Good Value |
|--------|------------------|-----------|
| **MAE** | Avg absolute error in trend score | < 0.4 |
| **RMSE** | Weighted error in trend score | < 0.5 |
| **Direction Accuracy** | % correct up/down predictions | > 55% |
| **Correlation** | Correlation with actual trends | > 0.4 |

---

## Attention Weights

Attention shows which historical weeks matter most:

```python
# Get attention for batch
attention = outputs['attention_weights']  # (batch_size, 12)

# Visualize
import matplotlib.pyplot as plt
plt.bar(range(12), attention[0].cpu().numpy())
plt.xlabel('Week')
plt.ylabel('Importance')
plt.show()
```

---

## Multi-Horizon Predictions

Model predicts two horizons:

```python
# 7-day (shorter-term, more predictable)
pred_7d = outputs['7day_trend']
dir_7d = outputs['7day_direction']

# 30-day (longer-term, harder to predict)
pred_30d = outputs['30day_trend']
dir_30d = outputs['30day_direction']

# Direction: 0=down, 1=up (get from argmax of logits)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Model not learning** | Lower learning rate, check data normalization |
| **Memory error** | Reduce batch_size or sequence_length |
| **Unstable training** | Already has gradient clipping, check learning rate |
| **Poor predictions** | Train longer, ensure diverse training data |

---

## Next Steps

1. **Run examples**: `python examples/lstm_examples.py`
2. **Read guide**: `LSTM_MODEL_GUIDE.md`
3. **Compare with Dual-Tower**: Both models in same modelling package
4. **Fine-tune**: Adjust hyperparameters based on your data

---

## Quick Commands

```bash
# Run examples
python examples/lstm_examples.py

# Check model architecture
python -c "from modelling.ml_models import create_lstm_model; m = create_lstm_model(); print(m)"

# Test data loading
python -c "from modelling.ml_models import create_lstm_data_loaders; help(create_lstm_data_loaders)"
```

---

**For detailed information, see `LSTM_MODEL_GUIDE.md`**
