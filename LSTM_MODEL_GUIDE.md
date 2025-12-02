# LSTM Model Implementation Guide

## Overview

This guide covers the complete LSTM model implementation for stock trend prediction. The model combines:
- **Bidirectional LSTM** for sequence processing
- **Multi-head Attention** for temporal importance weighting
- **Dual Prediction Heads** for 7-day and 30-day forecasting
- **Multi-task Learning** combining regression and classification

## Architecture

### Model Components

```
Input Sequence (batch_size, seq_len=12, input_dim=62)
        ↓
[LSTM Encoder]
- Bidirectional LSTM (2 layers)
- Hidden dimension: 128
- Dropout: 0.2
        ↓
(batch_size, seq_len, 128)
        ↓
[Attention Layer]
- Multi-head attention (4 heads)
- Query from final hidden state
- Keys/values from LSTM output
        ↓
(batch_size, 128)
        ↓
[Dual Prediction Heads]
├─→ 7-day Head: (trend_score, direction_logits)
└─→ 30-day Head: (trend_score, direction_logits)
```

### Key Features

#### 1. **LSTM Encoder** (`LSTMEncoder`)
- Stacked bidirectional LSTM layers
- Configurable hidden dimensions and dropout
- Optional projection layer for bidirectional output

#### 2. **Attention Mechanism** (`AttentionLayer`)
- Multi-head attention for temporal weighting
- Identifies which historical periods matter most
- Provides interpretability through attention weights

#### 3. **Prediction Heads** (`PredictionHead`)
- Separate heads for each prediction horizon
- Outputs:
  - **Trend Score**: Continuous value in [-1, 1]
    - 1.0 = strong uptrend
    - 0.0 = neutral
    - -1.0 = strong downtrend
  - **Direction Logits**: Classification for up/down prediction

## Configuration

### Model Config (`LSTMModelConfig`)

```python
from modelling.configs import LSTMModelConfig

config = LSTMModelConfig(
    input_dim=62,              # Stock market features
    hidden_dim=128,            # LSTM hidden dimension
    num_lstm_layers=2,         # Number of stacked LSTM layers
    num_attention_heads=4,     # Multi-head attention heads
    dropout_rate=0.2,          # Regularization
    bidirectional=True,        # Bidirectional LSTM
    head_hidden_dims=[64, 32], # Hidden dims in prediction heads
)
```

### Training Config (`LSTMTrainingConfig`)

```python
from modelling.configs import LSTMTrainingConfig

config = LSTMTrainingConfig(
    batch_size=32,
    epochs=100,
    early_stopping_patience=15,
    max_grad_norm=1.0,
    loss=LSTMLossConfig(
        regression_weight_7d=1.0,
        classification_weight_7d=0.5,
        regression_weight_30d=1.0,
        classification_weight_30d=0.5,
    ),
    optimizer=OptimizerConfig(
        optimizer_type='adam',
        base_learning_rate=0.001,
        weight_decay=1e-5,
    ),
    scheduler=SchedulerConfig(
        scheduler_type='cosine',
        total_epochs=100,
        warmup_epochs=5,
    ),
    sequence=LSTMSequenceConfig(
        sequence_length=12,  # weeks
        label_horizons=[7, 30],  # days
        normalize_features=True,
    ),
)
```

## Data Preparation

### Input Format

The LSTM expects time series data as sequences:

```
DataFrame with:
- Index: datetime (weekly granularity)
- Columns: 
  - feature_0 to feature_61 (stock market features)
  - trend_7day (target for 7-day prediction)
  - trend_30day (target for 30-day prediction)
```

### Creating Data Loaders

```python
from modelling.ml_models import create_lstm_data_loaders
import pandas as pd

# Your data
data = pd.DataFrame({
    'feature_0': [...],
    ...,
    'feature_61': [...],
    'trend_7day': [...],
    'trend_30day': [...],
}, index=pd.date_range('2015-01-01', periods=500, freq='W'))

# Create loaders
train_loader, val_loader, test_loader = create_lstm_data_loaders(
    data,
    sequence_length=12,      # 12 weeks of history
    batch_size=32,
    normalize=True,
    time_aware_split=True,   # Preserve temporal order
    train_fraction=0.7,
    val_fraction=0.15,
    test_fraction=0.15,
)
```

### Data Module Features

- **Time-aware splitting**: Preserves temporal order for realistic evaluation
- **Normalization**: StandardScaler fit on training data, applied to all sets
- **Sequence batching**: Handles padding and masking automatically
- **Feature separation**: Easily configure which features to use

## Training

### Basic Training Loop

```python
from modelling.ml_models import (
    create_lstm_model,
    LSTMLoss,
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
model = create_lstm_model(
    input_dim=62,
    hidden_dim=128,
    num_lstm_layers=2,
    num_attention_heads=4,
    device=device
)

# Create components
optimizer = create_lstm_optimizer(model, learning_rate=0.001)
scheduler = create_lstm_scheduler(optimizer, total_epochs=100)
loss_fn = LSTMLoss(
    regression_weight_7d=1.0,
    classification_weight_7d=0.5,
    regression_weight_30d=1.0,
    classification_weight_30d=0.5,
)

# Create trainer
trainer = LSTMTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    checkpoint_dir='./checkpoints/lstm',
)

# Train
history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    early_stopping_patience=15,
    min_delta=1e-4,
)

# Visualize training
trainer.plot_history(output_path='training_history.png')
```

### Loss Function

The loss combines multiple objectives:

```python
loss_fn = LSTMLoss(
    regression_weight_7d=1.0,    # 7-day trend prediction
    classification_weight_7d=0.5, # 7-day direction prediction
    regression_weight_30d=1.0,    # 30-day trend prediction
    classification_weight_30d=0.5, # 30-day direction prediction
)

# Returns per-component losses
loss_dict = loss_fn(predictions, targets)
# {
#     'total_loss': scalar,
#     'loss_7day_regression': scalar,
#     'loss_7day_classification': scalar,
#     'loss_30day_regression': scalar,
#     'loss_30day_classification': scalar,
# }
```

### Trainer Features

- **Early Stopping**: Stops when validation loss doesn't improve
- **Checkpointing**: Saves best model automatically
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with warmup or ReduceLROnPlateau
- **Comprehensive Metrics**:
  - MSE, RMSE, MAE for regression tasks
  - Direction accuracy for classification tasks
  - Correlation coefficient

## Inference

### Making Predictions

```python
from modelling.ml_models import create_lstm_model

# Load model
model = create_lstm_model(device=device)
model.load_state_dict(torch.load('best_model.pt')['model_state_dict'])

# Prepare data
test_sequences = torch.randn(32, 12, 62).to(device)

# Get predictions
model.eval()
with torch.no_grad():
    outputs = model(test_sequences)

# Outputs contain:
# - '7day_trend': (batch_size, 1) - continuous trend score [-1, 1]
# - '7day_direction': (batch_size, 2) - logits for up/down
# - '30day_trend': (batch_size, 1) - continuous trend score [-1, 1]
# - '30day_direction': (batch_size, 2) - logits for up/down
# - 'attention_weights': (batch_size, 12) - temporal attention
```

### Interpreting Outputs

```python
# Trend scores (continuous)
trend_7day = outputs['7day_trend']  # Range: [-1, 1]
# 1.0  = very strong uptrend
# 0.5  = moderate uptrend
# 0.0  = neutral
# -0.5 = moderate downtrend
# -1.0 = very strong downtrend

# Direction (classification)
direction_logits = outputs['7day_direction']  # (batch_size, 2)
direction_probs = torch.softmax(direction_logits, dim=1)
predicted_direction = torch.argmax(direction_logits, dim=1)
# 1 = up, 0 = down

# Attention weights (interpretability)
attention_weights = outputs['attention_weights']  # (batch_size, seq_len)
# Shows which time steps were most important for prediction
```

## Attention Mechanism

### Understanding Attention Weights

The attention layer computes weights across all 12 time steps, showing which historical periods are most important:

```python
# Get attention weights
attention_weights = outputs['attention_weights']  # (batch_size, 12)

# Visualize for a single sample
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.bar(range(12), attention_weights[0].cpu().numpy())
plt.xlabel('Time Step (Weeks)')
plt.ylabel('Attention Weight')
plt.title('Which historical weeks matter most?')
plt.tight_layout()
plt.show()
```

### Key Insights

- **Recent weeks**: Typically have higher attention (more recent data is more relevant)
- **Pattern reversals**: Model may pay attention to earlier weeks if patterns repeat
- **Volatility**: Attention may vary based on market conditions
- **Feature importance**: Interact with feature saliency analysis for deeper insights

## Evaluation Metrics

### Regression Metrics

- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Correlation**: Pearson correlation between predictions and targets

### Classification Metrics

- **Direction Accuracy**: Percentage of correct up/down predictions
- **Precision/Recall**: For uptrend predictions (if needed)
- **F1-Score**: Harmonic mean of precision and recall

### Multi-Horizon Analysis

```python
# Compare 7-day vs 30-day performance
print(f"7-day MAE: {metrics['mae_7day']:.6f}")
print(f"30-day MAE: {metrics['mae_30day']:.6f}")

# Observations:
# - 30-day predictions typically have higher MAE (harder to predict far future)
# - 7-day predictions may have better direction accuracy (more predictable)
# - Attention patterns may differ between horizons
```

## Advanced Usage

### Custom Loss Weighting

```python
from modelling.ml_models import WeightedLSTMLoss

# Create loss function
weighted_loss_fn = WeightedLSTMLoss(
    regression_weight_7d=1.0,
    classification_weight_7d=0.5,
    regression_weight_30d=1.0,
    classification_weight_30d=0.5,
)

# Apply per-sample weights
sample_weights = torch.ones(batch_size)
sample_weights[volatile_indices] *= 2.0  # Weight high-volatility samples more

loss_dict = weighted_loss_fn(predictions, targets, sample_weights)
```

### Volatility-Aware Loss

```python
from modelling.ml_models import VolatilityAwareLoss

volatility_loss = VolatilityAwareLoss()

# Weight errors by predicted volatility
loss = volatility_loss(predictions, targets, volatility)
# Higher weight for low-volatility samples (model should be more confident)
```

### Custom Attention Heads

```python
# Modify attention mechanism
custom_attention = AttentionLayer(
    hidden_dim=128,
    num_heads=8,  # More heads for fine-grained attention
    dropout_rate=0.3,
)
```

## Troubleshooting

### High Training Loss

- **Issue**: Loss not decreasing
- **Solution**: 
  - Check data normalization
  - Reduce learning rate
  - Increase warmup epochs
  - Check for NaN/Inf in data

### Unstable Training

- **Issue**: Loss oscillates or explodes
- **Solution**:
  - Enable gradient clipping (already enabled, increase max_grad_norm if needed)
  - Reduce learning rate
  - Batch normalize input features

### Poor Prediction Accuracy

- **Issue**: Predictions are random or all same value
- **Solution**:
  - Train longer (increase epochs)
  - Ensure labels are well-distributed (check target statistics)
  - Try different learning rates
  - Check attention weights (are they learning meaningful patterns?)

### Memory Issues

- **Issue**: CUDA out of memory
- **Solution**:
  - Reduce batch size
  - Reduce sequence length
  - Reduce hidden_dim
  - Use CPU training (slower but works)

## Comparison with Dual-Tower Model

| Aspect | LSTM | Dual-Tower |
|--------|------|-----------|
| **Input** | Time series sequences | Context + Stock features |
| **Temporal** | Explicit (LSTM) | Implicit (embeddings) |
| **Attention** | Temporal attention | Tower independence |
| **Best for** | Time series patterns | Multi-modal fusion |
| **Training** | Moderate | Fast |
| **Interpretability** | Attention weights | Tower embeddings |

## Examples

See `examples/lstm_examples.py` for:
1. **Example 1**: Basic training from scratch
2. **Example 2**: Attention weight analysis
3. **Example 3**: Multi-horizon comparison
4. **Example 4**: Inference on new data

Run examples:
```bash
python examples/lstm_examples.py
```

## Performance Guidelines

### Expected Accuracy

On real stock data:
- **7-day prediction**: 55-65% direction accuracy
- **30-day prediction**: 50-58% direction accuracy
- **Trend score MAE**: 0.3-0.5 (on scale -1 to 1)

### Training Time

- **Single epoch**: ~10-30 seconds (depends on data size and hardware)
- **Full training (100 epochs)**: 15-60 minutes (CPU: 2-4 hours)

### Hardware Requirements

- **GPU**: RTX 2060 or better recommended
- **CPU**: Intel i7 or equivalent (significantly slower)
- **Memory**: 8GB RAM minimum, 16GB+ recommended

## References

### Key Papers

- **LSTM**: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
- **Attention**: Vaswani et al. (2017) - "Attention is All You Need"
- **Multi-task Learning**: Caruana (1997) - "Multitask Learning"

### Related Architectures

- Transformer-based models (improved attention)
- Temporal Convolutional Networks (TCN)
- Attention-based seq2seq models
- Hybrid LSTM-CNN architectures

## Support & Questions

For issues or questions:
1. Check the troubleshooting section above
2. Review the examples in `lstm_examples.py`
3. Check gradient clipping and learning rate schedules
4. Compare results with Dual-Tower model

---

**Last Updated**: December 2024
**Status**: Production Ready
**Version**: 2.0
