# Modelling Module Documentation

## Overview

The `modelling` module contains all machine learning models and training infrastructure, cleanly separated from the data pipeline. This architectural separation ensures:

- **Reusability**: Models can be trained on any compatible data source
- **Clarity**: Clear distinction between data collection and ML analysis
- **Scalability**: Easy to add new models or data sources independently
- **Maintainability**: Focused concerns make debugging and improvements easier

## Directory Structure

```
modelling/
├── __init__.py                 # Main module exports
├── ml_models/                  # ML model implementations
│   ├── __init__.py            # Model package exports
│   ├── dual_tower_model.py    # Dual-tower architecture
│   ├── dual_tower_loss.py     # Multi-task loss functions
│   ├── dual_tower_data.py     # Data loading and preprocessing
│   └── dual_tower_trainer.py  # Training loop and evaluation
└── configs/                    # Configuration management
    ├── __init__.py            # Config package exports
    └── model_configs.py       # Centralized config classes
```

## Quick Start

### Basic Imports

```python
# Import from main modelling package
from modelling import DualTowerRelevanceModel, create_model, DualTowerLoss

# Or import from subpackages for more granular control
from modelling.ml_models import DualTowerTrainer, create_data_loaders
from modelling.configs import ConfigManager, DualTowerModelConfig
```

### Training Example

```python
from modelling import (
    create_model,
    DualTowerLoss,
    DualTowerTrainer,
    create_optimizer,
    create_scheduler,
)
from modelling.ml_models import create_data_loaders
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor

# 1. Load training data
processor = UnifiedTrainingDataProcessor(config={'data_root': '/data'})
df = processor.generate_training_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    include_weekly_movement=True
)

# 2. Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    df,
    batch_size=32,
    normalize=True,
)

# 3. Create model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = create_model(device=device)

# 4. Set up training infrastructure
loss_fn = DualTowerLoss(
    regression_weight_7d=1.0,
    regression_weight_30d=1.0,
    classification_weight_7d=0.5,
    classification_weight_30d=0.5,
    regularization_weight=0.01,
)
optimizer = create_optimizer(model, learning_rate=0.001)
scheduler = create_scheduler(optimizer, total_epochs=100)

# 5. Train
trainer = DualTowerTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
)
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping_patience=15,
)

# 6. Evaluate
metrics = trainer.evaluate(test_loader)
print(f"Test metrics: {metrics}")
```

## Module Components

### ML Models (`ml_models/`)

#### Architecture: `dual_tower_model.py`

**Purpose**: PyTorch neural network implementation

**Key Classes**:
- `ContextTower`: Encodes context features (25 → 32 dims)
  - News features (8): sentiment, volume, diversity
  - Policy features (5): announcement type, urgency, sector impact
  - Macro features (12): inflation, rates, GDP, employment, etc.

- `StockTower`: Encodes stock features (62 → 64 dims)
  - Financial indicators
  - Technical indicators

- `RelevanceHead`: Predicts relevance for specific horizon
  - 7-day head: Predicts week-ahead relevance
  - 30-day head: Predicts month-ahead relevance

- `DualTowerRelevanceModel`: Combines all components
  - Multi-task output: score + direction
  - Bidirectional: positive and negative relevance

**Key Functions**:
- `create_model(device='cuda')`: Factory function to create model instance
- `count_parameters(model)`: Get total parameter count

**Usage**:
```python
from modelling import create_model

model = create_model(device='cuda')
params = model.count_parameters()
print(f"Total parameters: {params}")
```

#### Loss Functions: `dual_tower_loss.py`

**Purpose**: Multi-task loss functions with regularization

**Key Classes**:
- `RelevanceRegressionLoss`: MSE for continuous relevance score
- `RelevanceDirectionLoss`: CrossEntropy for direction classification
- `TowerRegularizationLoss`: Orthogonality regularization
- `EmbeddingMagnitudeLoss`: Magnitude regularization
- `DualTowerLoss`: Combined multi-task loss
- `WeightedDualTowerLoss`: Support for sample weighting

**Loss Components**:
1. **Regression Loss**: Predicts score in [-1, 1]
   - Positive: context drives stock movement
   - Negative: context opposes movement (hedging)

2. **Classification Loss**: Predicts direction (positive/negative/none)
   - Multi-class cross-entropy

3. **Regularization**:
   - Tower orthogonality: Ensures towers learn complementary representations
   - Embedding magnitude: Prevents unbounded embeddings

**Usage**:
```python
from modelling import DualTowerLoss

loss_fn = DualTowerLoss(
    regression_weight_7d=1.0,
    regression_weight_30d=1.0,
    classification_weight_7d=0.5,
    classification_weight_30d=0.5,
    regularization_weight=0.01,
)

# In training loop
predictions = model(context_data, stock_data)
loss = loss_fn(predictions, labels_7d, labels_30d)
loss.backward()
```

#### Data Loading: `dual_tower_data.py`

**Purpose**: Data preparation and batching

**Key Classes**:
- `DualTowerDataset`: PyTorch Dataset with feature separation
  - Automatically separates context vs stock features
  - Handles label generation for both horizons

- `DualTowerDataModule`: Encapsulates data preparation
  - Time-aware splitting (preserves temporal order)
  - Train/val/test separation

**Key Functions**:
- `create_data_loaders(df, batch_size, normalize, time_aware_split)`: Creates DataLoaders

**Features**:
- Automatic feature separation (context vs stock)
- Optional normalization (StandardScaler)
- Time-aware splitting preserves temporal dependencies
- Multi-horizon label support (7-day and 30-day)

**Usage**:
```python
from modelling.ml_models import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(
    df,
    batch_size=32,
    normalize=True,
    time_aware_split=True,
)

for batch in train_loader:
    context_data = batch['context_features']
    stock_data = batch['stock_features']
    labels_7d = batch['labels_7d']
    labels_30d = batch['labels_30d']
```

#### Training: `dual_tower_trainer.py`

**Purpose**: Complete training loop with optimization

**Key Classes**:
- `DualTowerTrainer`: Full training infrastructure
  - Early stopping and checkpointing
  - Learning rate scheduling
  - Gradient clipping
  - Comprehensive metrics tracking

**Key Functions**:
- `create_optimizer(model, learning_rate)`: Creates Adam optimizer
- `create_scheduler(optimizer, total_epochs)`: Creates CosineAnnealing scheduler

**Features**:
- Early stopping with patience and min_delta
- Learning rate scheduling (cosine annealing or plateau)
- Gradient clipping to prevent exploding gradients
- Per-component learning rates (context, stock, heads)
- Checkpoint saving

**Usage**:
```python
from modelling import DualTowerTrainer, create_optimizer, create_scheduler

optimizer = create_optimizer(model, learning_rate=0.001)
scheduler = create_scheduler(optimizer, total_epochs=100)

trainer = DualTowerTrainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device='cuda',
    checkpoint_dir='./checkpoints',
    max_grad_norm=1.0,
)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    early_stopping_patience=15,
)

metrics = trainer.evaluate(test_loader)
```

### Configuration (`configs/`)

#### Model Config: `model_configs.py`

**Purpose**: Centralized configuration management

**Key Classes**:
- `ContextTowerConfig`: Context tower hyperparameters
- `StockTowerConfig`: Stock tower hyperparameters
- `RelevanceHeadConfig`: Head hyperparameters
- `DualTowerModelConfig`: Complete model configuration
- `LossConfig`: Loss function weights
- `OptimizerConfig`: Optimizer hyperparameters
- `SchedulerConfig`: Learning rate scheduler configuration
- `TrainingConfig`: Complete training configuration
- `DataConfig`: Data loading configuration
- `ConfigManager`: Centralized config management

**Features**:
- Dataclass-based configuration (type-safe)
- YAML serialization/deserialization
- Dictionary conversion for easy access
- Sensible defaults for all components

**Usage**:
```python
from modelling.configs import ConfigManager

# Create with defaults
config = ConfigManager()

# Access configurations
print(config.model_config.context_tower.embedding_dim)
print(config.training_config.batch_size)
print(config.data_config.tickers)

# Convert to dictionary
config_dict = config.to_dict()

# Save and load from YAML
config.save_config('my_config.yaml')
loaded_config = ConfigManager.load_config('my_config.yaml')

# Create custom config
from modelling.configs import TrainingConfig
custom_training = TrainingConfig(
    batch_size=64,
    epochs=200,
    early_stopping_patience=20,
)
```

## Integration with Data Pipeline

The modelling module integrates with `data_pipeline` but maintains independence:

```python
# Data pipeline provides data
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor

processor = UnifiedTrainingDataProcessor(config)
df = processor.generate_training_data(...)

# Modelling module processes and trains on data
from data_pipeline.models import create_dual_tower_model, create_dual_tower_optimizer
from data_pipeline.models import create_dual_tower_data_loaders

loaders = create_dual_tower_data_loaders(df, ...)
model = create_dual_tower_model()
...
```

**Key Points**:
- Models don't directly access data sources
- Models work with preprocessed DataFrames from UnifiedTrainingDataProcessor
- Easy to swap data sources or preprocessing approaches
- Models can be evaluated independently of data pipeline

## Architectural Benefits

### Separation of Concerns
- **Data Pipeline** (`data_pipeline/`): Collects and processes raw data
- **Modelling** (`modelling/`): Trains and evaluates ML models
- Clear responsibility boundaries

### Reusability
- Train models on different data sources without modification
- Export models and use in production inference pipelines
- Easy to test models independently

### Scalability
- Add new models to `ml_models/` without touching data pipeline
- Add new data sources to `data_pipeline/` without affecting models
- Parallel development of new models and data sources

### Maintainability
- Focused module responsibilities
- Self-contained configurations
- Clear import paths

## Common Tasks

### Train a New Model

See Quick Start section above.

### Load a Trained Model

```python
import torch
from modelling import DualTowerRelevanceModel

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pth')
model = DualTowerRelevanceModel()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    output = model(context_features, stock_features)
```

### Customize Configuration

```python
from modelling.configs import (
    ConfigManager,
    TrainingConfig,
    OptimizerConfig,
)

config = ConfigManager()

# Customize training
config.training_config = TrainingConfig(
    batch_size=64,
    epochs=200,
    early_stopping_patience=20,
    optimizer=OptimizerConfig(
        base_learning_rate=0.0005,
        weight_decay=1e-4,
    ),
)

# Use in training
trainer = DualTowerTrainer(...)
trainer.train(..., **config.training_config.__dict__)
```

### Analyze Model Outputs

```python
from modelling import DualTowerTrainer

trainer = DualTowerTrainer(model, loss_fn, optimizer, scheduler, device)

# Get predictions with embeddings
predictions, embeddings = trainer.predict_with_embeddings(test_loader)

# Analyze embeddings
context_embeddings = embeddings['context']  # (N, 32)
stock_embeddings = embeddings['stock']      # (N, 64)

# Get feature importance
importance = model.get_feature_importance(context_features, stock_features)
```

## Performance Characteristics

### Model Complexity
- Total Parameters: ~350,000
- Context Tower: ~130,000 parameters
- Stock Tower: ~155,000 parameters
- Relevance Heads (2x): ~65,000 parameters

### Training Time
- Batch size 32: ~50-100ms per batch on GPU
- Full training (100 epochs): 30-60 minutes on single GPU
- Memory usage: ~2-3GB per GPU

### Inference Time
- Single sample: <1ms on GPU
- Batch of 32: ~5-10ms on GPU
- Model size: ~1.5MB (checkpoint)

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'modelling'`:
1. Ensure `/modelling/` directory exists at workspace root
2. Check that `__init__.py` files are present
3. Verify Python path includes workspace root

### Cross-Directory Imports

Models use `sys.path` insertion to import from data_pipeline:
```python
# In dual_tower_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_pipeline.core.training_data import UnifiedTrainingDataProcessor
```

If imports fail:
1. Check file paths are correct
2. Verify data_pipeline package structure
3. Ensure all `__init__.py` files exist

### Training Issues

See DUAL_TOWER_MODEL_DESIGN.md and DUAL_TOWER_QUICK_START.md for detailed troubleshooting.

## Further Documentation

- `DUAL_TOWER_MODEL_DESIGN.md`: Complete technical specification
- `DUAL_TOWER_QUICK_START.md`: 5-minute beginner guide
- `DUAL_TOWER_IMPLEMENTATION_SUMMARY.md`: Project overview
- `examples/dual_tower_examples.py`: 5 complete working examples
