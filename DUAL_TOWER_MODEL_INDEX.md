# Dual-Tower Model - Complete Implementation Index

## 🎯 Welcome to the Dual-Tower Model for Context-Stock Relevance Prediction

This is your complete **production-ready implementation** of a dual-tower deep neural network that predicts how market context (policy, news, macro) influences stock movements.

---

## 📚 Where to Start

### 🚀 **Quick Start (5 Minutes)**
👉 **Read**: [`DUAL_TOWER_QUICK_START.md`](DUAL_TOWER_QUICK_START.md)

Your fast track to understanding and using the model:
- What is it?
- How to train?
- How to predict?
- What do results mean?

### 📖 **Deep Dive (30 Minutes)**
👉 **Read**: [`DUAL_TOWER_MODEL_DESIGN.md`](DUAL_TOWER_MODEL_DESIGN.md)

Complete technical specification with 12 sections:
- Problem statement
- Architecture details
- Loss functions
- Training procedure
- Inference guide
- Implementation roadmap

### 💻 **Working Examples (15 Minutes)**
👉 **Run**: [`examples/dual_tower_examples.py`](examples/dual_tower_examples.py)

Five complete, runnable examples:
1. Basic training
2. Making predictions
3. Interpreting results
4. Feature importance
5. Evaluation metrics

### 📋 **Project Summary (10 Minutes)**
👉 **Read**: [`DUAL_TOWER_IMPLEMENTATION_SUMMARY.md`](DUAL_TOWER_IMPLEMENTATION_SUMMARY.md)

Complete project overview:
- All deliverables
- Architecture summary
- Training specifications
- Getting started checklist

---

## 🏗️ Implementation Files

### Core Model Architecture
```
data_pipeline/models/dual_tower_model.py
├─ ContextTower              (128→64→32 dims)
├─ StockTower                (256→128→64 dims)
├─ RelevanceHead             (7-day & 30-day)
└─ DualTowerRelevanceModel   (main model)
```
**What it does**: Implements the neural network architecture

### Loss Functions
```
data_pipeline/models/dual_tower_loss.py
├─ RelevanceRegressionLoss       (MSE for score)
├─ RelevanceDirectionLoss        (classification)
├─ TowerRegularizationLoss       (orthogonality)
├─ EmbeddingMagnitudeLoss        (regularization)
└─ DualTowerLoss                 (combined)
```
**What it does**: Defines all loss functions with proper weighting

### Data Loading
```
data_pipeline/models/dual_tower_data.py
├─ DualTowerDataset             (PyTorch dataset)
├─ DualTowerDataModule          (train/val/test split)
└─ create_data_loaders()        (factory function)
```
**What it does**: Loads data, separates features, generates labels

### Training Loop
```
data_pipeline/models/dual_tower_trainer.py
├─ DualTowerTrainer             (trainer class)
├─ create_optimizer()           (task-specific LR)
├─ create_scheduler()           (learning rate schedule)
└─ (training, validation, evaluation)
```
**What it does**: Complete training pipeline with checkpointing

### Examples
```
examples/dual_tower_examples.py
├─ example_1_basic_training()       (train from scratch)
├─ example_2_predictions()          (make predictions)
├─ example_3_interpretation()       (understand results)
├─ example_4_feature_importance()   (analyze features)
└─ example_5_evaluation_metrics()   (compute metrics)
```
**What it does**: Five working examples showing how to use everything

---

## 🎓 Model Overview

### Problem We're Solving
**How much do policy, news, and macroeconomic data influence stock price movements?**

- Does context **support** the stock movement? (positive relevance)
- Does context **oppose** the stock movement? (negative relevance/hedging)
- How much is each effect? (magnitude)
- Different for 7-day vs 30-day horizons?

### Solution: Dual-Tower Architecture
```
Context Data              Stock Data
(News, Policy, Macro)    (Financial, Technical)
    ↓                          ↓
  [Tower]                    [Tower]
    ↓                          ↓
   [Embed]                   [Embed]
    └─────────┬──────────────┘
              ↓
        [Interaction]
              ↓
    [7-day & 30-day Heads]
              ↓
    Predictions + Confidence
```

### What You Get

**For 7-day horizon**:
- Relevance score: -1 to +1 (negative to positive correlation)
- Direction: positive or negative
- Confidence: how sure are we?

**For 30-day horizon**: Same structure

---

## 🚀 Quick Start: 3 Steps

### Step 1: Generate Training Data
```python
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor

processor = UnifiedTrainingDataProcessor({'data_root': '/data'})
df = processor.generate_training_data()
```

### Step 2: Create & Train Model
```python
from data_pipeline.models.dual_tower_model import create_model
from data_pipeline.models.dual_tower_data import create_data_loaders
from data_pipeline.models.dual_tower_trainer import DualTowerTrainer, create_optimizer, create_scheduler
from data_pipeline.models.dual_tower_loss import DualTowerLoss

# Setup
train_loader, val_loader, test_loader = create_data_loaders(df, batch_size=32)
model = create_model(device='cuda')
optimizer = create_optimizer(model)
scheduler = create_scheduler(optimizer, total_epochs=50)
loss_fn = DualTowerLoss()

# Train
trainer = DualTowerTrainer(model, loss_fn, optimizer, scheduler, device='cuda')
history = trainer.train(train_loader, val_loader, epochs=50)
```

### Step 3: Make Predictions
```python
trainer.load_best_checkpoint()
model.eval()

with torch.no_grad():
    for context, stock, labels in test_loader:
        outputs = model(context.to(device), stock.to(device))
        print(f"7-day relevance: {outputs['score_7d']}")
        print(f"30-day relevance: {outputs['score_30d']}")
        break
```

---

## 📊 Architecture Specs

### Input Dimensions
- **Context**: 25 features
  - News (8): sentiment, volume, diversity
  - Policy (5): announcement type, urgency, sector impact
  - Macro (12): inflation, rates, GDP, employment
- **Stock**: 62 features
  - OHLCV (5), Technical indicators (20+), Returns (5), Volatility (10+), Volume (15+)

### Tower Specifications
- **Context Tower**: 25 → 128 → 64 → 32 dimensions
- **Stock Tower**: 62 → 256 → 128 → 64 dimensions

### Output Predictions
- **7-day relevance score**: -1 to 1
- **7-day direction**: positive or negative
- **7-day confidence**: 0 to 1
- **30-day relevance score**: -1 to 1
- **30-day direction**: positive or negative
- **30-day confidence**: 0 to 1

---

## 📈 Training Setup

| Component | Setting |
|-----------|---------|
| Optimizer | Adam with task-specific learning rates |
| Learning Rate | 0.001 (context), 0.0005 (stock), 0.001 (heads) |
| Scheduler | Cosine annealing with 5-epoch warmup |
| Batch Size | 32 |
| Epochs | 50-100 (with early stopping) |
| Loss Weights | Regression: 1.0, Classification: 0.5, Regularization: 0.01 |
| Early Stopping Patience | 15 epochs |
| Gradient Clipping | max_norm=1.0 |

---

## 💡 Key Concepts

### Multi-Task Learning
- **Primary task**: Predict continuous relevance score (regression)
- **Secondary task**: Predict direction (classification)
- Result: More robust, better generalization

### Multi-Horizon Learning
- **7-day head**: Captures short-term trading impacts
- **30-day head**: Captures long-term trend impacts
- Learns different relationship dynamics for each horizon

### Bidirectional Relevance
- **Positive relevance** (+0.8): Context drives price upward
- **Negative relevance** (-0.8): Context drives price downward (hedging)
- Model treats both equally

### Tower Independence
- Separate architectures: Context vs Stock
- Prevents "tower collapse" (both learning same thing)
- Specialized for different data characteristics

---

## 📚 Documentation Roadmap

```
BEGINNER PATH (Start Here)
  ↓
[DUAL_TOWER_QUICK_START.md] ← 5-minute overview
  ↓
[examples/dual_tower_examples.py] ← Run examples
  ↓
SUCCESS: You can train and predict!

INTERMEDIATE PATH
  ↓
[DUAL_TOWER_MODEL_DESIGN.md] ← Technical deep dive
  ↓
Understand architecture and loss functions
  ↓
SUCCESS: You understand the how & why

ADVANCED PATH
  ↓
[Source code] ← Review implementation
  ↓
Customize architectures and training
  ↓
SUCCESS: You can extend and optimize
```

---

## 🎯 What Can You Do?

✅ **Train Models**
- On your own data
- With custom architectures
- Different hyperparameters

✅ **Make Predictions**
- On new data
- Get confidence scores
- Understand direction

✅ **Analyze Results**
- Feature importance (which context matters?)
- Time horizon comparison (7-day vs 30-day)
- Embedding visualization

✅ **Deploy**
- Save trained models
- Integrate with trading systems
- Real-time predictions

---

## 🔧 Customization Examples

### Use Different Architecture
```python
model = DualTowerRelevanceModel(
    context_hidden_dims=[256, 128, 64, 32],  # Deeper
    stock_hidden_dims=[512, 256, 128, 64],   # Larger
    context_embedding_dim=64,
    stock_embedding_dim=128,
)
```

### Adjust Loss Weights
```python
loss_fn = DualTowerLoss(
    regression_weight_7d=2.0,      # Focus on 7-day
    classification_weight_7d=1.0,  # More weight to direction
    regularization_weight=0.05,    # Stronger regularization
)
```

### Custom Training
```python
trainer = DualTowerTrainer(
    model=model,
    max_grad_norm=0.5,             # Tighter clipping
)

trainer.train(
    train_loader, val_loader,
    epochs=200,
    early_stopping_patience=25     # More patience
)
```

---

## 🐛 Troubleshooting

### Problem: Model not converging
**Solution**: Reduce learning rate, check feature normalization

### Problem: Loss is NaN
**Solution**: Check for invalid features, reduce batch size

### Problem: Poor validation performance
**Solution**: Use more data, increase regularization, longer training

See `DUAL_TOWER_QUICK_START.md` for detailed troubleshooting!

---

## 📞 Support & Resources

| Need | File |
|------|------|
| Quick start | `DUAL_TOWER_QUICK_START.md` |
| Technical details | `DUAL_TOWER_MODEL_DESIGN.md` |
| Implementation overview | `DUAL_TOWER_IMPLEMENTATION_SUMMARY.md` |
| Working code | `examples/dual_tower_examples.py` |
| Model source | `data_pipeline/models/dual_tower_model.py` |
| Loss functions | `data_pipeline/models/dual_tower_loss.py` |
| Data loading | `data_pipeline/models/dual_tower_data.py` |
| Training | `data_pipeline/models/dual_tower_trainer.py` |

---

## ✅ Implementation Status

- [x] Architecture designed
- [x] Model implemented
- [x] Loss functions implemented
- [x] Data loading pipeline
- [x] Training loop
- [x] Evaluation metrics
- [x] Examples provided
- [x] Documentation complete
- [x] Production ready

**Status: READY FOR USE** 🚀

---

## 🎓 Learning Paths

### Path 1: Beginner (30 minutes)
1. Read: `DUAL_TOWER_QUICK_START.md` (5 min)
2. Run: `example_1_basic_training()` (10 min)
3. Run: `example_2_predictions()` (5 min)
4. Run: `example_3_interpretation()` (5 min)
5. Run: `example_5_evaluation_metrics()` (5 min)

**Result**: Can train and evaluate model

### Path 2: Intermediate (2 hours)
1. Complete Beginner path
2. Read: `DUAL_TOWER_MODEL_DESIGN.md` sections 1-5 (45 min)
3. Study: Loss function details (Section 4) (20 min)
4. Run: All 5 examples (20 min)
5. Modify: Architecture or loss weights (15 min)

**Result**: Understand design and can customize

### Path 3: Advanced (4+ hours)
1. Complete Intermediate path
2. Read: Complete `DUAL_TOWER_MODEL_DESIGN.md` (60 min)
3. Review: Source code (60 min)
4. Implement: Custom losses or metrics (60+ min)
5. Optimize: Hyperparameters for your data (variable)

**Result**: Full mastery, can extend and deploy

---

## 🚀 Next Steps

1. **Install Dependencies**
   ```bash
   pip install torch pandas numpy scikit-learn
   ```

2. **Read Quick Start**
   - Open: `DUAL_TOWER_QUICK_START.md`
   - Time: 5 minutes

3. **Run First Example**
   - File: `examples/dual_tower_examples.py`
   - Function: `example_1_basic_training()`
   - Time: 10-30 minutes

4. **Make Your First Predictions**
   - Function: `example_2_predictions()`
   - Time: 5 minutes

5. **Explore Design Document**
   - File: `DUAL_TOWER_MODEL_DESIGN.md`
   - Time: 30 minutes

6. **Train on Your Data**
   - Use quick start section
   - Time: Variable

---

## 📋 Files at a Glance

```
/StockTrendEsimator/
│
├── Documentation
│   ├── DUAL_TOWER_QUICK_START.md              ← START HERE
│   ├── DUAL_TOWER_MODEL_DESIGN.md             ← Full spec
│   ├── DUAL_TOWER_IMPLEMENTATION_SUMMARY.md   ← Project overview
│   └── DUAL_TOWER_MODEL_INDEX.md              ← This file
│
├── Implementation
│   └── data_pipeline/models/
│       ├── dual_tower_model.py                ← Architecture
│       ├── dual_tower_loss.py                 ← Loss functions
│       ├── dual_tower_data.py                 ← Data loading
│       └── dual_tower_trainer.py              ← Training
│
├── Examples
│   └── examples/dual_tower_examples.py        ← 5 examples
│
└── Data Pipeline (existing)
    └── data_pipeline/core/training_data.py    ← Unified data
```

---

## 🎉 You're All Set!

Everything you need to:
- ✅ Understand the model
- ✅ Train on your data
- ✅ Make predictions
- ✅ Analyze results
- ✅ Deploy to production

**Start with**: [`DUAL_TOWER_QUICK_START.md`](DUAL_TOWER_QUICK_START.md)

**Questions?** See the comprehensive documentation above!

---

**Last Updated**: 2025-12-01
**Status**: ✅ COMPLETE & PRODUCTION READY
