# Refactored Architecture: Visual Guide

## Complete Directory Structure

```
StockTrendEsimator/
│
├── modelling/                          (Main ML package - REFACTORED)
│   │
│   ├── architectures/                 ✨ MODEL DEFINITIONS
│   │   ├── __init__.py
│   │   ├── dual_tower.py             (ContextTower, StockTower, DualTowerRelevanceModel)
│   │   └── lstm.py                   (LSTMEncoder, PredictionHead, LSTMRelevanceModel)
│   │
│   ├── losses/                        ✨ LOSS FUNCTIONS
│   │   ├── __init__.py
│   │   ├── dual_tower.py             (DualTowerLoss, WeightedDualTowerLoss, etc.)
│   │   └── lstm.py                   (LSTMLoss, WeightedLSTMLoss, etc.)
│   │
│   ├── data/                          ✨ DATA LOADING & PREPROCESSING
│   │   ├── __init__.py
│   │   ├── dual_tower.py             (DualTowerDataset, create_dual_tower_data_loaders)
│   │   └── lstm.py                   (LSTMDataset, create_lstm_data_loaders)
│   │
│   ├── trainers/                      ✨ TRAINING LOOPS
│   │   ├── __init__.py
│   │   ├── dual_tower.py             (DualTowerTrainer, create_dual_tower_optimizer)
│   │   └── lstm.py                   (LSTMTrainer, create_lstm_optimizer)
│   │
│   ├── configs/                       ✨ CONFIGURATION
│   │   ├── __init__.py
│   │   └── model_configs.py          (ConfigManager, all Config dataclasses)
│   │
│   ├── ml_models/                     📦 BACKWARD COMPATIBILITY
│   │   ├── __init__.py
│   │   ├── dual_tower_model.py
│   │   ├── dual_tower_loss.py
│   │   ├── dual_tower_data.py
│   │   ├── dual_tower_trainer.py
│   │   ├── lstm_model.py
│   │   ├── lstm_loss.py
│   │   ├── lstm_data.py
│   │   └── lstm_trainer.py
│   │
│   ├── __init__.py                   (Main exports - imports from all categories)
│   └── README.md                      (Module documentation)
│
├── data_pipeline/                     (Data infrastructure)
│   ├── models/
│   │   ├── financial_source.py
│   │   ├── macro_source.py
│   │   ├── movement_source.py
│   │   ├── news_source.py
│   │   └── policy_source.py
│   ├── core/
│   │   └── training_data.py          (UnifiedTrainingDataProcessor)
│   └── ...
│
├── examples/
│   └── dual_tower_examples.py        (Uses modelling imports)
│
└── docs/
    ├── REFACTORING_REPORT.md         ✨ NEW: Complete refactoring guide
    ├── ARCHITECTURE_DIAGRAM.md       ✨ NEW: This file
    ├── DUAL_TOWER_MODEL_DESIGN.md
    ├── DUAL_TOWER_QUICK_START.md
    └── ...
```

---

## Semantic Organization Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Main Modelling Package                        │
│                   modelling/__init__.py                          │
│                  (Central export point)                          │
└────┬────────────────────────────────────────────────────────────┘
     │
     ├─────────────────────────────────────────────────────────────┐
     │                                                              │
     ↓                      LAYER 1: SEMANTICS                     │
┌─────────────────────────────────────────────────────────────────┐
│  Components organized by FUNCTION, not MODEL                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │ architectures│  │    losses    │  │     data     │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│         │                 │                  │                   │
│         ↓                 ↓                  ↓                   │
│    [dual_tower.py]  [dual_tower.py]  [dual_tower.py]          │
│    [lstm.py]        [lstm.py]        [lstm.py]                │
│         │                 │                  │                   │
└─────────────────────────────────────────────────────────────────┘
     │
     ├─────────────────────────────────────────────────────────────┐
     │                                                              │
     ↓                      LAYER 2: MODELS                        │
┌─────────────────────────────────────────────────────────────────┐
│  Components grouped by MODEL FAMILY                             │
│                                                                  │
│  ┌────────────────────────┐      ┌─────────────────────────┐   │
│  │   DUAL-TOWER MODELS    │      │    LSTM MODELS          │   │
│  ├────────────────────────┤      ├─────────────────────────┤   │
│  │ architectures/         │      │ architectures/          │   │
│  │  - ContextTower        │      │  - LSTMEncoder          │   │
│  │  - StockTower          │      │  - PredictionHead       │   │
│  │ losses/                │      │ losses/                 │   │
│  │  - DualTowerLoss       │      │  - LSTMMultiTaskLoss    │   │
│  │ data/                  │      │ data/                   │   │
│  │  - DualTowerDataset    │      │  - LSTMDataset          │   │
│  │ trainers/              │      │ trainers/               │   │
│  │  - DualTowerTrainer    │      │  - LSTMTrainer          │   │
│  └────────────────────────┘      └─────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
     │
     ├─────────────────────────────────────────────────────────────┐
     │                                                              │
     ↓                    LAYER 3: SUPPORT                         │
┌─────────────────────────────────────────────────────────────────┐
│  Configs, utilities, backward compatibility                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │    configs/  │  │   ml_models/ │                             │
│  │  - ConfigMgr │  │  (old files) │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Dependency Flow

```
Data Pipeline
    ↓
    ├─────────→ modelling/data/dual_tower.py
    │               ├─→ DualTowerDataset
    │               └─→ create_dual_tower_data_loaders()
    │
    ├─────────→ modelling/architectures/dual_tower.py
                    ├─→ ContextTower (25 → 32 dims)
                    ├─→ StockTower (62 → 64 dims)
                    └─→ DualTowerRelevanceModel

                        ↓ (receives data)

        modelling/losses/dual_tower.py
            ├─→ RelevanceRegressionLoss
            ├─→ RelevanceDirectionLoss
            └─→ DualTowerLoss (combines all)

                        ↓ (computes loss)

        modelling/trainers/dual_tower.py
            ├─→ DualTowerTrainer
            ├─→ create_dual_tower_optimizer()
            └─→ create_dual_tower_scheduler()

                        ↓ (trains model)

    Output: Trained Model Checkpoint
```

---

## Import Hierarchy

### SEMANTIC IMPORTS (Recommended New Way)

```python
# Level 1: By semantic function
from modelling.architectures import DualTowerRelevanceModel
from modelling.losses import DualTowerLoss
from modelling.data import create_dual_tower_data_loaders
from modelling.trainers import DualTowerTrainer

# Level 2: Within semantic package
from modelling.architectures.dual_tower import ContextTower, StockTower
from modelling.losses.dual_tower import RelevanceRegressionLoss
from modelling.data.dual_tower import DualTowerDataset
from modelling.trainers.dual_tower import create_dual_tower_optimizer
```

### MAIN PACKAGE IMPORTS (Backward Compatible)

```python
# Level 3: All from main package (still works!)
from modelling import (
    DualTowerRelevanceModel,
    DualTowerLoss,
    create_dual_tower_data_loaders,
    DualTowerTrainer,
)
```

### OLD ML_MODELS IMPORTS (Deprecated but works)

```python
# Level 4: Old location (for backward compatibility)
from modelling.ml_models import DualTowerRelevanceModel  # ⚠️ Works but not recommended
```

---

## Adding New Components: Where Things Go

### Scenario: Adding a Transformer Model

```
1. New Model Architecture:
   modelling/architectures/transformer.py
   ├── class TransformerEncoder
   ├── class TransformerDecoder  
   ├── class TransformerRelevanceModel
   └── create_transformer_model()

2. New Loss Functions:
   modelling/losses/transformer.py
   ├── class TransformerRegressionLoss
   ├── class TransformerClassificationLoss
   └── class TransformerMultiTaskLoss

3. New Data Module:
   modelling/data/transformer.py
   ├── class TransformerDataset
   ├── class TransformerDataModule
   └── create_transformer_data_loaders()

4. New Trainer:
   modelling/trainers/transformer.py
   ├── class TransformerTrainer
   ├── create_transformer_optimizer()
   └── create_transformer_scheduler()

5. Update configs (if needed):
   modelling/configs/model_configs.py
   ├── @dataclass TransformerModelConfig
   ├── @dataclass TransformerTrainingConfig
   └── Add to ConfigManager

6. Update __init__.py files:
   modelling/architectures/__init__.py   → Add imports
   modelling/losses/__init__.py          → Add imports
   modelling/data/__init__.py            → Add imports
   modelling/trainers/__init__.py        → Add imports
   modelling/__init__.py                 → Add exports to __all__
```

**Result**: Consistent, predictable pattern! ✓

---

## Directory Statistics

### File Counts
```
architectures/  2 Python files  +  1 __init__.py
losses/         2 Python files  +  1 __init__.py
data/           2 Python files  +  1 __init__.py
trainers/       2 Python files  +  1 __init__.py
configs/        1 Python file   +  1 __init__.py
ml_models/      8 Python files  +  1 __init__.py  (backward compat)
─────────────────────────────────────────────────
Total:          17 functional modules + 6 __init__.py
```

### Component Counts
```
architectures/  6 components (model classes, factories)
losses/         10 components (loss classes)
data/           6 components (dataset, datamodule, factories)
trainers/       6 components (trainer, optimizers, schedulers)
configs/        15 components (config classes, manager)
─────────────────────────────────────────────────
Total:          43 organized components
```

---

## Comparison: Before vs After Organization

### BEFORE: Finding stuff was hard

```
Q: "Where are all the loss functions?"
A: Look in ml_models/ for *loss.py files 😕

Q: "Show me all data loading code"
A: Look in ml_models/ for *data.py files 😕

Q: "What does my training pipeline look like?"
A: Read ml_models/*trainer.py files 😕

Q: "How should I organize my new model?"
A: ???  No pattern! 😕
```

### AFTER: Everything is clear

```
Q: "Where are all the loss functions?"
A: modelling/losses/ - crystal clear! ✓

Q: "Show me all data loading code"
A: modelling/data/ - immediately obvious! ✓

Q: "What does my training pipeline look like?"
A: modelling/trainers/ - easy to follow! ✓

Q: "How should I organize my new model?"
A: Follow the same pattern in each directory! ✓
```

---

## Package Initialization Order

When you `import modelling`:

```
1. modelling/__init__.py
   ├── from modelling.architectures import ...  ← Gets architectures/__init__.py
   │   └── dual_tower.py and lstm.py files
   │
   ├── from modelling.losses import ...         ← Gets losses/__init__.py
   │   └── dual_tower.py and lstm.py files
   │
   ├── from modelling.data import ...           ← Gets data/__init__.py
   │   └── dual_tower.py and lstm.py files
   │
   ├── from modelling.trainers import ...       ← Gets trainers/__init__.py
   │   └── dual_tower.py and lstm.py files
   │
   └── from modelling.configs import ...        ← Gets configs/__init__.py
       └── model_configs.py
```

All components loaded and exported in `modelling.__all__`

---

## Backward Compatibility Matrix

```
┌─────────────────────────────────────────────────────────────┐
│                    IMPORT STYLES                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ NEW (Recommended) ✓ Semantic
│ from modelling.architectures import DualTowerRelevanceModel
│ from modelling.losses import DualTowerLoss
│ from modelling.data import create_dual_tower_data_loaders
│ from modelling.trainers import DualTowerTrainer
│                                                              │
│ COMPATIBLE ✓ Main package
│ from modelling import DualTowerRelevanceModel
│ from modelling import DualTowerLoss
│ from modelling import create_dual_tower_data_loaders
│ from modelling import DualTowerTrainer
│                                                              │
│ DEPRECATED ⚠️ Old ml_models
│ from modelling.ml_models import DualTowerRelevanceModel
│ from modelling.ml_models import DualTowerLoss
│ (Still works via backward compat, but not recommended)
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

All styles work! Users can migrate at their own pace.

---

## Migration Timeline (Suggested)

```
Phase 1: NEW - Users adopt semantic imports
  └─ modelling.architectures, modelling.losses, etc.

Phase 2: COMPATIBLE - Main package imports still work
  └─ from modelling import DualTowerRelevanceModel

Phase 3: DEPRECATED - Old ml_models imports discouraged
  └─ from modelling.ml_models import ... (works, but not recommended)

Phase 4 (Future): OPTIONAL - Remove ml_models/ folder if desired
  └─ All users have migrated to semantic imports
```

---

## Key Principles Applied

### 1️⃣ Single Responsibility
Each directory handles one concern:
- architectures → model definitions
- losses → training objectives
- data → input/output handling
- trainers → optimization

### 2️⃣ Semantic Grouping
Files grouped by **what they do**, not **which model**

### 3️⃣ Consistency
Same pattern for all models:
- architectures/model_name.py
- losses/model_name.py
- data/model_name.py
- trainers/model_name.py

### 4️⃣ Clarity
Directory names match functionality:
- architectures (not models)
- losses (not objectives)
- data (not datasets)
- trainers (not training)

### 5️⃣ Extensibility
Easy to add new models or components

---

## 🎯 Summary

The refactored codebase now:

✅ **Organizes by semantic similarity** - not model type
✅ **Groups related components** - easy to understand relationships
✅ **Provides clear patterns** - for extending with new models
✅ **Maintains backward compatibility** - old imports still work
✅ **Improves navigation** - know exactly where to look
✅ **Enables better testing** - test each concern separately

**Result**: A cleaner, more maintainable, more scalable ML codebase! 🚀

---

**Status**: ✅ Refactoring Complete and Verified
**Backward Compatibility**: ✅ 100% Maintained
**Documentation**: ✅ Comprehensive
**Ready for Production**: ✅ Yes
