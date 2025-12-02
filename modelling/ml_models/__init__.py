"""
ML Models Package - Dual-Tower and LSTM models for stock trend prediction

Models:
- DualTowerRelevanceModel: Predicts context-stock trend relevance
- LSTMTrendPredictor: LSTM-based time series trend prediction with attention
"""

from .dual_tower_model import (
    DualTowerRelevanceModel,
    ContextTower,
    StockTower,
    RelevanceHead,
    create_model,
    count_parameters,
)

from .dual_tower_loss import (
    DualTowerLoss,
    RelevanceRegressionLoss,
    RelevanceDirectionLoss,
    TowerRegularizationLoss,
    EmbeddingMagnitudeLoss,
    WeightedDualTowerLoss,
)

from .dual_tower_data import (
    DualTowerDataset,
    DualTowerDataModule,
    create_data_loaders,
)

from .dual_tower_trainer import (
    DualTowerTrainer,
    create_optimizer,
    create_scheduler,
)

# LSTM imports
from .lstm_model import (
    LSTMTrendPredictor,
    LSTMEncoder,
    AttentionLayer,
    PredictionHead,
    create_lstm_model,
    count_lstm_parameters,
)

from .lstm_loss import (
    TrendRegressionLoss,
    DirectionClassificationLoss,
    VolatilityAwareLoss,
    LSTMLoss,
    WeightedLSTMLoss,
)

from .lstm_data import (
    LSTMSequenceDataset,
    LSTMDataModule,
    create_lstm_data_loaders,
)

from .lstm_trainer import (
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
)

__all__ = [
    # Dual-Tower: Model
    'DualTowerRelevanceModel',
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'create_model',
    'count_parameters',
    # Dual-Tower: Loss
    'DualTowerLoss',
    'RelevanceRegressionLoss',
    'RelevanceDirectionLoss',
    'TowerRegularizationLoss',
    'EmbeddingMagnitudeLoss',
    'WeightedDualTowerLoss',
    # Dual-Tower: Data
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_data_loaders',
    # Dual-Tower: Training
    'DualTowerTrainer',
    'create_optimizer',
    'create_scheduler',
    # LSTM: Model
    'LSTMTrendPredictor',
    'LSTMEncoder',
    'AttentionLayer',
    'PredictionHead',
    'create_lstm_model',
    'count_lstm_parameters',
    # LSTM: Loss
    'TrendRegressionLoss',
    'DirectionClassificationLoss',
    'VolatilityAwareLoss',
    'LSTMLoss',
    'WeightedLSTMLoss',
    # LSTM: Data
    'LSTMSequenceDataset',
    'LSTMDataModule',
    'create_lstm_data_loaders',
    # LSTM: Training
    'LSTMTrainer',
    'create_lstm_optimizer',
    'create_lstm_scheduler',
]
