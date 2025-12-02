"""
Modelling Module - ML Model Training and Inference

Refactored architecture organized by semantic/functional similarity:

Subpackages:
- architectures/: Model architecture definitions (dual_tower, lstm)
- losses/: Loss function implementations (dual_tower, lstm)
- data/: Data loading and preprocessing (dual_tower, lstm)
- trainers/: Training loops and optimization (dual_tower, lstm)
- configs/: Configuration management and hyperparameters

Models:
- DualTowerRelevanceModel: Context-stock relevance prediction
- LSTMRelevanceModel: Time series trend forecasting with LSTM
"""

__version__ = "2.0"
__author__ = "Stock Trend Estimator Team"

# Architecture imports (Model definitions)
from .architectures import (
    # Dual-Tower
    ContextTower,
    StockTower,
    RelevanceHead,
    DualTowerRelevanceModel,
    create_dual_tower_model,
    count_dual_tower_parameters,
    # LSTM
    LSTMEncoder,
    PredictionHead,
    LSTMRelevanceModel,
    create_lstm_model,
    count_lstm_parameters,
)

# Loss imports (Loss functions)
from .losses import (
    # Dual-Tower losses
    RelevanceRegressionLoss,
    RelevanceDirectionLoss,
    TowerRegularizationLoss,
    EmbeddingMagnitudeLoss,
    DualTowerLoss,
    WeightedDualTowerLoss,
    # LSTM losses
    LSTMRegressionLoss,
    LSTMDirectionLoss,
    LSTMSequenceLoss,
    LSTMMultiTaskLoss,
    WeightedLSTMLoss,
)

# Data imports (Data loading and preprocessing)
from .data import (
    # Dual-Tower data
    DualTowerDataset,
    DualTowerDataModule,
    create_dual_tower_data_loaders,
    # LSTM data
    LSTMDataset,
    LSTMDataModule,
    create_lstm_data_loaders,
)

# Trainer imports (Training loops and optimization)
from .trainers import (
    # Dual-Tower training
    DualTowerTrainer,
    create_dual_tower_optimizer,
    create_dual_tower_scheduler,
    # LSTM training
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
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
    # ==================== ARCHITECTURES ====================
    # Dual-Tower architecture
    'DualTowerRelevanceModel',
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'create_dual_tower_model',
    'count_dual_tower_parameters',
    # LSTM architecture
    'LSTMRelevanceModel',
    'LSTMEncoder',
    'PredictionHead',
    'create_lstm_model',
    'count_lstm_parameters',
    
    # ==================== LOSS FUNCTIONS ====================
    # Dual-Tower losses
    'RelevanceRegressionLoss',
    'RelevanceDirectionLoss',
    'TowerRegularizationLoss',
    'EmbeddingMagnitudeLoss',
    'DualTowerLoss',
    'WeightedDualTowerLoss',
    # LSTM losses
    'LSTMRegressionLoss',
    'LSTMDirectionLoss',
    'LSTMSequenceLoss',
    'LSTMMultiTaskLoss',
    'WeightedLSTMLoss',
    
    # ==================== DATA LOADING ====================
    # Dual-Tower data
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_dual_tower_data_loaders',
    # LSTM data
    'LSTMDataset',
    'LSTMDataModule',
    'create_lstm_data_loaders',
    
    # ==================== TRAINING ====================
    # Dual-Tower training
    'DualTowerTrainer',
    'create_dual_tower_optimizer',
    'create_dual_tower_scheduler',
    # LSTM training
    'LSTMTrainer',
    'create_lstm_optimizer',
    'create_lstm_scheduler',
    
    # ==================== CONFIGURATION ====================
    # Dual-Tower config
    'DualTowerModelConfig',
    'TrainingConfig',
    'DataConfig',
    # LSTM config
    'LSTMModelConfig',
    'LSTMTrainingConfig',
    'LSTMSequenceConfig',
    # General
    'ConfigManager',
    'DEFAULT_MODEL_CONFIG',
    'DEFAULT_TRAINING_CONFIG',
    'DEFAULT_DATA_CONFIG',
    'DEFAULT_LSTM_MODEL_CONFIG',
    'DEFAULT_LSTM_TRAINING_CONFIG',
]
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
