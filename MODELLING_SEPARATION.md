# Modelling vs Data Pipeline: Separation of Concerns

## Quick Reference

| Aspect | Data Pipeline | Modelling |
|--------|---|---|
| **Location** | `/data_pipeline/` | `/modelling/` |
| **Purpose** | Collect & process raw data | Train & evaluate ML models |
| **Inputs** | Financial APIs, News APIs, Economic APIs | Preprocessed DataFrames from data_pipeline |
| **Outputs** | Clean pandas DataFrames | Trained models, predictions, embeddings |
| **Key Classes** | UnifiedTrainingDataProcessor | DualTowerRelevanceModel, DualTowerTrainer |
| **Key Files** | financial_source.py, news_source.py | dual_tower_model.py, dual_tower_trainer.py |
| **Responsibility** | Data collection/integration | Machine learning training/inference |

---

## Directory Structure After Reorganization

### Before
```
data_pipeline/models/
├── financial_source.py          ← Data source
├── macro_source.py              ← Data source
├── movement_source.py           ← Data source
├── news_source.py               ← Data source
├── policy_source.py             ← Data source
├── dual_tower_model.py          ← ML MODEL (shouldn't be here!)
├── dual_tower_loss.py           ← ML MODEL (shouldn't be here!)
├── dual_tower_data.py           ← ML MODEL (shouldn't be here!)
└── dual_tower_trainer.py        ← ML MODEL (shouldn't be here!)
```

### After
```
data_pipeline/models/
├── financial_source.py          ← Data source ✓
├── macro_source.py              ← Data source ✓
├── movement_source.py           ← Data source ✓
├── news_source.py               ← Data source ✓
└── policy_source.py             ← Data source ✓

modelling/ml_models/
├── dual_tower_model.py          ← ML MODEL ✓
├── dual_tower_loss.py           ← ML MODEL ✓
├── dual_tower_data.py           ← ML MODEL ✓
└── dual_tower_trainer.py        ← ML MODEL ✓

modelling/configs/
└── model_configs.py             ← Configuration ✓
```

---

## Responsibilities

### Data Pipeline (`data_pipeline/`)

**What it does:**
- ✅ Collects raw data from external sources
- ✅ Cleans and validates data
- ✅ Performs feature engineering
- ✅ Generates labels
- ✅ Exports preprocessed DataFrames

**Example**:
```python
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor

processor = UnifiedTrainingDataProcessor(config)
df = processor.generate_training_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    include_weekly_movement=True,
)
# Returns DataFrame with 87 features + labels, ready for ML
```

### Modelling (`modelling/`)

**What it does:**
- ✅ Defines ML architectures
- ✅ Implements loss functions
- ✅ Loads and preprocesses data for training
- ✅ Trains models
- ✅ Makes predictions
- ✅ Manages hyperparameters

**Example**:
```python
from modelling import create_model, DualTowerTrainer
from modelling.ml_models import create_data_loaders

# Load data from pipeline
train_loader, val_loader, test_loader = create_data_loaders(df, ...)

# Create model
model = create_model()

# Train
trainer = DualTowerTrainer(model, loss_fn, optimizer, scheduler)
trainer.train(train_loader, val_loader, epochs=100)
```

---

## Integration Pattern

```
┌─────────────────────────────────┐
│    Data Pipeline                │
│                                 │
│  Collects & Processes Data      │
│  ├─ Yahoo Finance               │
│  ├─ NewsAPI                     │
│  ├─ FRED                        │
│  └─ UnifiedTrainingDataProcessor│
│      └─ Output: DataFrame       │
└────────────────┬────────────────┘
                 │
                 │ Pass DataFrame
                 │
                 ↓
┌─────────────────────────────────┐
│    Modelling                    │
│                                 │
│  Trains ML Models               │
│  ├─ create_data_loaders()       │
│  ├─ create_model()              │
│  ├─ DualTowerTrainer.train()    │
│  └─ Output: Model checkpoint    │
└─────────────────────────────────┘
```

---

## Import Patterns

### ✅ CORRECT: Import data infrastructure from data_pipeline

```python
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor
from data_pipeline.models.financial_source import FinancialDataSource
```

### ✅ CORRECT: Import models from modelling

```python
from modelling import (
    DualTowerRelevanceModel,
    create_model,
    DualTowerLoss,
    DualTowerTrainer,
)
from modelling.ml_models import create_data_loaders
from modelling.configs import ConfigManager
```

### ❌ WRONG: Don't import models from data_pipeline

```python
# This won't work anymore (old location)
from data_pipeline.models.dual_tower_model import create_model
```

---

## Migration Checklist

If you have code using the old import paths:

- [ ] Find all `from data_pipeline.models.dual_tower_*` imports
- [ ] Replace with `from modelling import ...` or `from modelling.ml_models import ...`
- [ ] Test that imports work
- [ ] Run training/inference code to verify functionality
- [ ] Update documentation/comments if needed

### Quick Find & Replace

**Find**: `from data_pipeline.models.dual_tower_model import`
**Replace**: `from modelling.ml_models import`

**Find**: `from data_pipeline.models.dual_tower_loss import`
**Replace**: `from modelling.ml_models import`

**Find**: `from data_pipeline.models.dual_tower_data import`
**Replace**: `from modelling.ml_models import`

**Find**: `from data_pipeline.models.dual_tower_trainer import`
**Replace**: `from modelling.ml_models import`

---

## Why This Separation?

### Problem It Solves

**Before:**
- ML models mixed with data source definitions
- Unclear what should be modified when
- Hard to reuse models with different data
- Hard to improve data without retraining models

**After:**
- Clear responsibility boundaries
- Models are reusable with any compatible data
- Data improvements don't force model retraining
- Easy to test models and data independently

### Benefits

| Benefit | Example |
|---------|---------|
| **Reusability** | Use same model with different data sources |
| **Independence** | Update data processing without changing models |
| **Clarity** | New developers know where to look for each concern |
| **Testing** | Test data and models separately |
| **Scaling** | Add new models/data sources in parallel |

---

## Common Workflows

### Workflow 1: Add New Data Source

```
1. Create new source in data_pipeline/models/
2. Update UnifiedTrainingDataProcessor
3. Regenerate data: df = processor.generate_training_data(...)
4. Models automatically work with new data
   └─ No changes needed to modelling/!
```

### Workflow 2: Improve Model Architecture

```
1. Create new model in modelling/ml_models/
2. Load data from data_pipeline: df = processor.generate_training_data(...)
3. Train new model: trainer.train(...)
4. Compare old vs new model performance
   └─ Data pipeline unchanged!
```

### Workflow 3: Retrain with Latest Data

```
1. Run data_pipeline to collect latest data
2. Load preprocessing: df = processor.generate_training_data(...)
3. Retrain existing model: trainer.train(...)
4. Deploy new checkpoint
```

---

## Key Files Locations

### Data Pipeline
```
data_pipeline/
├── models/
│   ├── financial_source.py      ← Financial data definitions
│   ├── macro_source.py          ← Macro data definitions
│   ├── movement_source.py       ← Stock movement definitions
│   ├── news_source.py           ← News data definitions
│   └── policy_source.py         ← Policy data definitions
│
├── core/
│   ├── training_data.py         ← UnifiedTrainingDataProcessor (main integration point)
│   └── ...
│
└── ...
```

### Modelling
```
modelling/
├── ml_models/
│   ├── dual_tower_model.py      ← Neural network architecture
│   ├── dual_tower_loss.py       ← Loss functions
│   ├── dual_tower_data.py       ← Data loading for training
│   └── dual_tower_trainer.py    ← Training loop
│
├── configs/
│   └── model_configs.py         ← Hyperparameter management
│
└── README.md                    ← Detailed modelling documentation
```

---

## Common Questions

### Q: Where should I put a new ML model?
**A:** In `modelling/ml_models/`. It should follow the pattern of dual_tower_model.py.

### Q: Where should I add a new data source?
**A:** In `data_pipeline/models/`. It should follow the pattern of financial_source.py.

### Q: Can I import from data_pipeline in modelling?
**A:** Yes! Modelling can import data infrastructure from data_pipeline. The reverse (data_pipeline importing from modelling) should be avoided.

### Q: Do I need to regenerate data if I update a model?
**A:** No! Models are independent of data. You only need to regenerate data if you change data sources or feature engineering.

### Q: Can I use the same data for different models?
**A:** Yes! Any model in modelling/ can use data from data_pipeline.generate_training_data(). This is the whole point!

### Q: How do I add a new hyperparameter?
**A:** Add it to the appropriate config class in modelling/configs/model_configs.py, then use it in your model.

---

## Testing Strategy

### Data Pipeline Tests
- Data collection works
- Features are correctly computed
- Labels are properly generated
- Output DataFrame has expected shape and values

### Modelling Tests
- Model initializes without errors
- Forward pass works
- Loss computation is correct
- Training loop runs
- Predictions are reasonable

### Integration Tests
- Data from pipeline loads into model
- Training completes successfully
- Predictions are consistent

---

## Deployment Considerations

### Training
```python
# Collect and prepare data
processor = UnifiedTrainingDataProcessor(config)
df = processor.generate_training_data(...)

# Train model
train_loader, val_loader, test_loader = create_data_loaders(df, ...)
trainer = DualTowerTrainer(model, loss_fn, optimizer, scheduler)
trainer.train(train_loader, val_loader, epochs=100)

# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model_config,
}, 'best_model.pth')
```

### Inference
```python
# Load trained model
checkpoint = torch.load('best_model.pth')
model = create_model()
model.load_state_dict(checkpoint['model_state_dict'])

# Preprocess live data using pipeline
new_data = processor.generate_training_data(...)

# Make predictions
with torch.no_grad():
    predictions = model(context_data, stock_data)
```

---

## Performance Characteristics

### Data Pipeline
- Data collection: 1-5 sec per stock
- Feature engineering: 100-500ms per stock
- Memory: ~500MB for 1 year data

### Modelling
- Model training: 30-60 min (100 epochs, 1 GPU)
- Inference: <1ms per sample (GPU)
- Model size: 1.5MB

---

## Documentation Index

- **modelling/README.md**: Complete modelling module guide
- **ARCHITECTURE.md**: System architecture overview (original)
- **DUAL_TOWER_MODEL_DESIGN.md**: Technical specification
- **DUAL_TOWER_QUICK_START.md**: Quick start guide
- **examples/dual_tower_examples.py**: Working code examples
