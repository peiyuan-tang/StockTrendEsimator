"""
Modelling Module - ML Model Training and Inference

This module contains all machine learning models and training infrastructure,
separated from the data pipeline for clear architectural separation.

Submodules:
- ml_models: Model implementations (dual-tower, etc.)
- configs: Model configurations and hyperparameters
"""

__version__ = "1.0"
__author__ = "Stock Trend Estimator Team"

from .ml_models import (
    DualTowerRelevanceModel,
    ContextTower,
    StockTower,
    RelevanceHead,
    create_model,
    count_parameters,
    DualTowerLoss,
    WeightedDualTowerLoss,
    DualTowerTrainer,
    create_optimizer,
    create_scheduler,
    DualTowerDataset,
    DualTowerDataModule,
    create_data_loaders,
)

from .configs import (
    DualTowerModelConfig,
    TrainingConfig,
    DataConfig,
    ConfigManager,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_DATA_CONFIG,
)

__all__ = [
    # Model architecture
    'DualTowerRelevanceModel',
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'create_model',
    'count_parameters',
    # Loss functions
    'DualTowerLoss',
    'WeightedDualTowerLoss',
    # Training
    'DualTowerTrainer',
    'create_optimizer',
    'create_scheduler',
    # Data loading
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_data_loaders',
    # Configuration
    'DualTowerModelConfig',
    'TrainingConfig',
    'DataConfig',
    'ConfigManager',
    'DEFAULT_MODEL_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_DATA_CONFIG',
]
]
