"""
ML Models Package - Dual-Tower and other models for relevance prediction

Models:
- DualTowerRelevanceModel: Predicts context-stock trend relevance
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

__all__ = [
    # Model
    'DualTowerRelevanceModel',
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'create_model',
    'count_parameters',
    # Loss
    'DualTowerLoss',
    'RelevanceRegressionLoss',
    'RelevanceDirectionLoss',
    'TowerRegularizationLoss',
    'EmbeddingMagnitudeLoss',
    'WeightedDualTowerLoss',
    # Data
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_data_loaders',
    # Training
    'DualTowerTrainer',
    'create_optimizer',
    'create_scheduler',
]
