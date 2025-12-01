# Dual-Tower Model: Quick Start Guide

## 🎯 What is the Dual-Tower Model?

The **Dual-Tower Relevance Model** predicts how much market context (policy, news, macro data) influences stock movements across two time horizons:

- **7-day relevance**: Short-term trading impact
- **30-day relevance**: Long-term trend impact

It learns both **positive relevance** (context drives price up) and **negative relevance** (hedging effects where context drives price down).

---

## 📊 Architecture Overview

```
Context Data (25 features)          Stock Data (62 features)
  ├─ News (8)                        ├─ OHLCV (5)
  ├─ Policy (5)                      ├─ Technical (20+)
  └─ Macro (12)                      ├─ Returns (5)
       ↓                              └─ Volatility (10+)
  ┌─────────────────┐              ┌──────────────────┐
  │  Context Tower  │              │   Stock Tower    │
  │    (128→64→32)  │              │   (256→128→64)   │
  └─────────────────┘              └──────────────────┘
         ↓                                ↓
    [32-dim embed]                  [64-dim embed]
         │                                │
         └────────────┬────────────────────┘
                      ↓
            ┌─────────────────────┐
            │  Relevance Heads    │
            │  (7-day & 30-day)   │
            └─────────────────────┘
                      ↓
         Score: [-1, 1] & Direction
```

---

## 🚀 Quick Start: 5 Minutes

### Step 1: Install Dependencies

```bash
pip install torch pandas numpy scikit-learn tensorboard
```

### Step 2: Generate Training Data

```python
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor

# Generate unified training dataset
config = {'data_root': '/data'}
processor = UnifiedTrainingDataProcessor(config)

df = processor.generate_training_data(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
    include_weekly_movement=True
)

print(f"Dataset shape: {df.shape}")  # Should be ~240 samples × 87 features
```

### Step 3: Create & Train Model

```python
from data_pipeline.models.dual_tower_model import create_model
from data_pipeline.models.dual_tower_data import create_data_loaders
from data_pipeline.models.dual_tower_loss import DualTowerLoss
from data_pipeline.models.dual_tower_trainer import (
    DualTowerTrainer, create_optimizer, create_scheduler
)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    df, batch_size=32, normalize=True
)

# Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(device=device)

# Create optimizer, scheduler, and loss
optimizer = create_optimizer(model)
scheduler = create_scheduler(optimizer, total_epochs=50)
loss_fn = DualTowerLoss()

# Train
trainer = DualTowerTrainer(model, loss_fn, optimizer, scheduler, device)
history = trainer.train(train_loader, val_loader, epochs=50)
```

### Step 4: Make Predictions

```python
# Load best model
trainer.load_best_checkpoint()

# Get predictions
model.eval()
with torch.no_grad():
    for context, stock, labels in test_loader:
        outputs = model(context.to(device), stock.to(device))
        
        score_7d = outputs['score_7d']  # Relevance score [-1, 1]
        score_30d = outputs['score_30d']
        
        print(f"7-day relevance: {score_7d[0].item():.4f}")
        print(f"30-day relevance: {score_30d[0].item():.4f}")
        break
```

---

## 📖 Understanding Predictions

### Relevance Score [-1, 1]

```
 -1.0                0                +1.0
  │                 │                 │
  ├─────────────────┼─────────────────┤
  │
  Strong Negative    Neutral          Strong Positive
  (Anti-corr)        (Uncorr)         (Corr)
  
  Context opposes    Context has      Context supports
  stock movement     no effect        stock movement
  (Hedging)         (Independent)     (Aligned)
```

### Example Interpretations

| Score | Meaning | Action |
|-------|---------|--------|
| **+0.85** | Strong positive correlation | Context strongly drives stock price |
| **+0.35** | Weak positive correlation | Context somewhat supports movement |
| **0.02** | Nearly neutral | Stock moves independently |
| **-0.40** | Weak negative correlation | Context opposes movement (hedging) |
| **-0.90** | Strong negative correlation | Strong anti-correlation/hedging effect |

---

## 🔧 Model Configuration

### Default Architecture

```python
context_input_dim = 25          # news(8) + policy(5) + macro(12)
stock_input_dim = 62            # financial + technical indicators

# Context Tower
context_hidden_dims = [128, 64, 32]
context_embedding_dim = 32
context_dropout = 0.2

# Stock Tower
stock_hidden_dims = [256, 128, 64]
stock_embedding_dim = 64
stock_dropout = 0.3

# Relevance Heads (both 7-day and 30-day)
relevance_hidden_dims = [16, 8]
head_dropout = 0.2
```

### Custom Configuration

```python
from data_pipeline.models.dual_tower_model import DualTowerRelevanceModel

model = DualTowerRelevanceModel(
    context_input_dim=25,
    stock_input_dim=62,
    context_hidden_dims=[256, 128, 64],  # Larger towers
    stock_hidden_dims=[512, 256, 128],
    context_embedding_dim=64,
    stock_embedding_dim=128,
    context_dropout=0.3,
    stock_dropout=0.4,
)
```

---

## 📊 Training Configuration

### Loss Function Weights

```python
loss_fn = DualTowerLoss(
    regression_weight_7d=1.0,        # Weight for 7-day MSE
    regression_weight_30d=1.0,       # Weight for 30-day MSE
    classification_weight_7d=0.5,    # Weight for 7-day direction
    classification_weight_30d=0.5,   # Weight for 30-day direction
    regularization_weight=0.01,      # Weight for tower regularization
)
```

### Optimizer Settings

```python
from data_pipeline.models.dual_tower_trainer import create_optimizer

optimizer = create_optimizer(
    model,
    learning_rate=0.001,             # Base LR
    weight_decay=1e-5                # L2 regularization
)
# Note: Stock tower gets 0.5x learning rate due to higher capacity
```

### Scheduler Settings

```python
from data_pipeline.models.dual_tower_trainer import create_scheduler

scheduler = create_scheduler(
    optimizer,
    total_epochs=100,
    warmup_epochs=5,
    scheduler_type='cosine'          # 'cosine' or 'plateau'
)
```

---

## 🎓 Training Tips

### For Better Results

```python
# 1. Use early stopping
trainer = DualTowerTrainer(...)
history = trainer.train(
    train_loader, val_loader,
    epochs=200,
    early_stopping_patience=20  # Stop if no improvement for 20 epochs
)

# 2. Monitor loss components
for epoch in history['epochs']:
    print(f"Epoch {epoch}:")
    print(f"  Regression 7d: {history['metrics'][epoch]['regression_7d']:.4f}")
    print(f"  Regression 30d: {history['metrics'][epoch]['regression_30d']:.4f}")
    print(f"  Classification: {history['metrics'][epoch]['classification_7d']:.4f}")
    print(f"  Regularization: {history['metrics'][epoch]['regularization']:.4f}")

# 3. Normalize features
from data_pipeline.models.dual_tower_data import create_data_loaders
train_loader, val_loader, test_loader = create_data_loaders(
    df,
    batch_size=32,
    normalize=True  # IMPORTANT: Always normalize
)

# 4. Use time-aware split
train_loader, val_loader, test_loader = create_data_loaders(
    df,
    batch_size=32,
    time_aware_split=True  # Prevents data leakage
)
```

---

## 📈 Evaluation Metrics

### Understanding Metrics

```python
from examples.dual_tower_examples import example_5_evaluation_metrics

metrics = example_5_evaluation_metrics(model, test_loader)

# 7-day metrics
print(f"7-day MSE:           {metrics['7d']['mse']:.6f}")
print(f"7-day Correlation:   {metrics['7d']['corr']:.4f}")
print(f"7-day Direction Acc: {metrics['7d']['acc']:.4f}")

# 30-day metrics
print(f"30-day MSE:          {metrics['30d']['mse']:.6f}")
print(f"30-day Correlation:  {metrics['30d']['corr']:.4f}")
print(f"30-day Direction Acc:{metrics['30d']['acc']:.4f}")
```

### Target Performance

```
Reasonable Performance:
├─ MSE < 0.15 (explains ~85% of variance)
├─ Correlation > 0.60
└─ Direction Accuracy > 65%

Good Performance:
├─ MSE < 0.10 (explains ~90% of variance)
├─ Correlation > 0.75
└─ Direction Accuracy > 70%

Excellent Performance:
├─ MSE < 0.08 (explains ~92% of variance)
├─ Correlation > 0.85
└─ Direction Accuracy > 75%
```

---

## 🔍 Feature Analysis

### Analyze Context Importance

```python
from examples.dual_tower_examples import example_4_feature_importance

example_4_feature_importance(model, test_loader)

# Output shows how much each context group (news, policy, macro)
# impacts the model's predictions
```

### Extract Embeddings

```python
# Get tower embeddings (useful for visualization/clustering)
context_embed, stock_embed = model.extract_embeddings(
    context_features.to(device),
    stock_features.to(device)
)

print(f"Context embedding shape: {context_embed.shape}")  # (batch, 32)
print(f"Stock embedding shape: {stock_embed.shape}")      # (batch, 64)

# Can be used for t-SNE, UMAP visualization, or clustering
```

---

## 🛠️ Common Tasks

### Task 1: Train on Different Tickers

```python
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor

processor = UnifiedTrainingDataProcessor({'data_root': '/data'})

# Tech stocks only
df = processor.generate_training_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    include_weekly_movement=True
)

# Then train model as usual
train_loader, val_loader, test_loader = create_data_loaders(df)
```

### Task 2: Predict on New Data

```python
# New data point
context_features = torch.randn(1, 25).to(device)
stock_features = torch.randn(1, 62).to(device)

model.eval()
with torch.no_grad():
    outputs = model(context_features, stock_features)

# Interpret
relevance_7d = outputs['score_7d'].item()
confidence_7d = outputs['pos_prob_7d'].item()

print(f"7-day relevance: {relevance_7d:.4f}")
print(f"Direction: {'Positive' if relevance_7d > 0 else 'Negative'}")
print(f"Confidence: {confidence_7d:.4f}")
```

### Task 3: Analyze 7-Day vs 30-Day

```python
predictions = example_2_predictions(model, test_loader)

# Find samples with diverging horizons
divergence = np.abs(predictions['score_7d'] - predictions['score_30d'])
high_divergence_idx = np.argsort(-divergence)[:10]

print("Top 10 samples with diverging time horizons:")
for idx in high_divergence_idx:
    print(f"  7d: {predictions['score_7d'][idx]:.4f}, "
          f"30d: {predictions['score_30d'][idx]:.4f}")
```

---

## 🐛 Troubleshooting

### Problem: Model Not Converging

**Solution:**
```python
# Check learning rate
optimizer = create_optimizer(model, learning_rate=0.0005)  # Reduce LR

# Check batch size
train_loader, val_loader, test_loader = create_data_loaders(df, batch_size=16)  # Reduce

# Check gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm():.4f}")
```

### Problem: Loss is NaN

**Solution:**
```python
# Check for invalid features
print(f"Context features - Min: {context.min()}, Max: {context.max()}")
print(f"Stock features - Min: {stock.min()}, Max: {stock.max()}")

# Enable gradient clipping
trainer = DualTowerTrainer(..., max_grad_norm=1.0)
```

### Problem: Poor Validation Performance

**Solution:**
```python
# Use more data
# Or increase regularization
loss_fn = DualTowerLoss(regularization_weight=0.1)  # Increase from 0.01

# Or use different train/val split
train_loader, val_loader, test_loader = create_data_loaders(
    df, 
    val_fraction=0.1,  # Use less data for val
    test_fraction=0.1
)
```

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `data_pipeline/models/dual_tower_model.py` | Model architecture |
| `data_pipeline/models/dual_tower_loss.py` | Loss functions |
| `data_pipeline/models/dual_tower_data.py` | Data loading & preprocessing |
| `data_pipeline/models/dual_tower_trainer.py` | Training loop & evaluation |
| `examples/dual_tower_examples.py` | Complete examples |
| `DUAL_TOWER_MODEL_DESIGN.md` | Detailed design document |

---

## 📞 Next Steps

1. **Read Design Document**: `DUAL_TOWER_MODEL_DESIGN.md` for full technical details
2. **Run Examples**: Execute `examples/dual_tower_examples.py` for working demonstrations
3. **Train Your Model**: Use your own data with the quick start above
4. **Analyze Results**: Use interpretation functions to understand predictions
5. **Deploy**: Save trained model and deploy for real-time predictions

---

## 💡 Key Concepts

### Multi-Task Learning
- Predicts both continuous score (regression) and direction (classification)
- Improves generalization by learning complementary tasks

### Multi-Horizon Learning
- Separate heads for 7-day and 30-day predictions
- Captures short vs long-term relationships

### Tower Architecture
- **Context Tower**: Encodes market context efficiently
- **Stock Tower**: Encodes stock state separately
- **Interaction**: Learns how context and stock interact

### Bidirectional Relevance
- **Positive**: Context and stock move together
- **Negative**: Context and stock move opposite (hedging)
- Model learns both equally well

---

## 🎯 Use Cases

✅ **Portfolio Hedging**: Identify when news contradicts stock movement

✅ **Risk Management**: Quantify context sensitivity for different stocks

✅ **Strategy Optimization**: Understand which external factors matter

✅ **Model Explainability**: Understand why stocks move

✅ **Regime Detection**: Identify when relationships change

---

**Happy training!** 🚀

For questions, see `DUAL_TOWER_MODEL_DESIGN.md` or the comprehensive examples in `examples/dual_tower_examples.py`.
