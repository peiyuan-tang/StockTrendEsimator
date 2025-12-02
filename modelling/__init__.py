"""
Modelling Module - ML Model Training and Inference

This module contains all machine learning models and training infrastructure,
separated from the data pipeline for clear architectural separation.

Submodules:
- ml_models: Model implementations (dual-tower, LSTM, etc.)
- configs: Model configurations and hyperparameters

Models:
- DualTowerRelevanceModel: Context-stock relevance prediction
- LSTMTrendPredictor: Time series trend forecasting with attention
"""

__version__ = "2.0"
__author__ = "Stock Trend Estimator Team"

# Dual-Tower imports
from .ml_models import (
    # Dual-Tower Model
    DualTowerRelevanceModel,
    ContextTower,
    StockTower,
    RelevanceHead,
    create_model,
    count_parameters,
    # Dual-Tower Loss
    DualTowerLoss,
    WeightedDualTowerLoss,
    # Dual-Tower Training
    DualTowerTrainer,
    create_optimizer,
    create_scheduler,
    # Dual-Tower Data
    DualTowerDataset,
    DualTowerDataModule,
    create_data_loaders,
    # LSTM Model
    LSTMTrendPredictor,
    LSTMEncoder,
    AttentionLayer,
    PredictionHead,
    create_lstm_model,
    count_lstm_parameters,
    # LSTM Loss
    LSTMLoss,
    WeightedLSTMLoss,
    # LSTM Training
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
    # LSTM Data
    LSTMSequenceDataset,
    LSTMDataModule,
    create_lstm_data_loaders,
)

from .configs import (
    # Dual-Tower Config
    DualTowerModelConfig,
    TrainingConfig,
    DataConfig,
    # LSTM Config
    LSTMModelConfig,
    LSTMTrainingConfig,
    LSTMSequenceConfig,
    # General
    ConfigManager,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    DEFAULT_DATA_CONFIG,
    DEFAULT_LSTM_MODEL_CONFIG,
    DEFAULT_LSTM_TRAINING_CONFIG,
)

__all__ = [
    # Dual-Tower: Model architecture
    'DualTowerRelevanceModel',
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'create_model',
    'count_parameters',
    # Dual-Tower: Loss functions
    'DualTowerLoss',
    'WeightedDualTowerLoss',
    # Dual-Tower: Training
    'DualTowerTrainer',
    'create_optimizer',
    'create_scheduler',
    # Dual-Tower: Data loading
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_data_loaders',
    # LSTM: Model architecture
    'LSTMTrendPredictor',
    'LSTMEncoder',
    'AttentionLayer',
    'PredictionHead',
    'create_lstm_model',
    'count_lstm_parameters',
    # LSTM: Loss functions
    'LSTMLoss',
    'WeightedLSTMLoss',
    # LSTM: Training
    'LSTMTrainer',
    'create_lstm_optimizer',
    'create_lstm_scheduler',
    # LSTM: Data loading
    'LSTMSequenceDataset',
    'LSTMDataModule',
    'create_lstm_data_loaders',
    # Configuration
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
]
