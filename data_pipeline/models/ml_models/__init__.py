"""
ML Models Package - Unified Architecture

Complete ML modeling suite organized by semantic function:
- architectures/: Neural network models
- losses/: Training loss functions  
- data_loaders/: Data preprocessing and loading
- trainers/: Training loops and optimization
- model_configs.py: Configuration management

This package unifies the modelling and data_pipeline/models structure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Architecture imports
from .architectures import (
    ContextTower,
    StockTower,
    RelevanceHead,
    DualTowerRelevanceModel,
    create_dual_tower_model,
    count_dual_tower_parameters,
    LSTMEncoder,
    PredictionHead,
    LSTMRelevanceModel,
    create_lstm_model,
    count_lstm_parameters,
)

# Loss imports
from .losses import (
    RelevanceRegressionLoss,
    RelevanceDirectionLoss,
    TowerRegularizationLoss,
    EmbeddingMagnitudeLoss,
    DualTowerLoss,
    WeightedDualTowerLoss,
    LSTMRegressionLoss,
    LSTMDirectionLoss,
    LSTMSequenceLoss,
    LSTMMultiTaskLoss,
    WeightedLSTMLoss,
)

# Data loader imports
from .data_loaders import (
    DualTowerDataset,
    DualTowerDataModule,
    create_dual_tower_data_loaders,
    LSTMDataset,
    LSTMDataModule,
    create_lstm_data_loaders,
)

# Trainer imports
from .trainers import (
    DualTowerTrainer,
    create_dual_tower_optimizer,
    create_dual_tower_scheduler,
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
)

# Config imports
from .model_configs import (
    DualTowerModelConfig,
    TrainingConfig,
    DataConfig,
    LSTMModelConfig,
    LSTMTrainingConfig,
    LSTMSequenceConfig,
    ConfigManager,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_LSTM_MODEL_CONFIG,
    DEFAULT_LSTM_TRAINING_CONFIG,
)

__all__ = [
    # Architectures
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'DualTowerRelevanceModel',
    'create_dual_tower_model',
    'count_dual_tower_parameters',
    'LSTMEncoder',
    'PredictionHead',
    'LSTMRelevanceModel',
    'create_lstm_model',
    'count_lstm_parameters',
    # Losses
    'RelevanceRegressionLoss',
    'RelevanceDirectionLoss',
    'TowerRegularizationLoss',
    'EmbeddingMagnitudeLoss',
    'DualTowerLoss',
    'WeightedDualTowerLoss',
    'LSTMRegressionLoss',
    'LSTMDirectionLoss',
    'LSTMSequenceLoss',
    'LSTMMultiTaskLoss',
    'WeightedLSTMLoss',
    # Data loaders
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_dual_tower_data_loaders',
    'LSTMDataset',
    'LSTMDataModule',
    'create_lstm_data_loaders',
    # Trainers
    'DualTowerTrainer',
    'create_dual_tower_optimizer',
    'create_dual_tower_scheduler',
    'LSTMTrainer',
    'create_lstm_optimizer',
    'create_lstm_scheduler',
    # Configs
    'DualTowerModelConfig',
    'TrainingConfig',
    'DataConfig',
    'LSTMModelConfig',
    'LSTMTrainingConfig',
    'LSTMSequenceConfig',
    'ConfigManager',
    'DEFAULT_MODEL_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_DATA_CONFIG',
    'DEFAULT_LSTM_MODEL_CONFIG',
    'DEFAULT_LSTM_TRAINING_CONFIG',
]
