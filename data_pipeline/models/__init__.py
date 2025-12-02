#!/usr/bin/env python3
"""
Models package - Unified structure combining:
1. Data source implementations (financial, news, policy, macro)
2. ML model implementations (architectures, losses, trainers, data loaders)
"""

# Data source imports
from data_pipeline.models.financial_source import FinancialDataSource, BaseDataSource
from data_pipeline.models.movement_source import StockMovementSource
from data_pipeline.models.news_source import NewsDataSource
from data_pipeline.models.macro_source import MacroeconomicDataSource
from data_pipeline.models.policy_source import PolicyDataSource

# ML models imports
from data_pipeline.models.ml_models import (
    # Architectures
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
    # Losses
    DualTowerLoss,
    WeightedDualTowerLoss,
    LSTMMultiTaskLoss,
    WeightedLSTMLoss,
    # Data loaders
    DualTowerDataset,
    DualTowerDataModule,
    create_dual_tower_data_loaders,
    LSTMDataset,
    LSTMDataModule,
    create_lstm_data_loaders,
    # Trainers
    DualTowerTrainer,
    create_dual_tower_optimizer,
    create_dual_tower_scheduler,
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
    # Configs
    ConfigManager,
)

__all__ = [
    # Data sources
    'BaseDataSource',
    'FinancialDataSource',
    'StockMovementSource',
    'NewsDataSource',
    'MacroeconomicDataSource',
    'PolicyDataSource',
    # ML Models
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
    'DualTowerLoss',
    'WeightedDualTowerLoss',
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
    'ConfigManager',
]
