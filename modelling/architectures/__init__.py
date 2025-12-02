"""
Model Architectures Package

Contains neural network architecture implementations for different models:
- Dual-Tower: Multi-tower architecture for context-stock relevance
- LSTM: Sequential model for time-series prediction
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Dual-Tower imports
from .dual_tower import (
    ContextTower,
    StockTower,
    RelevanceHead,
    DualTowerRelevanceModel,
    create_model as create_dual_tower_model,
    count_parameters as count_dual_tower_parameters,
)

# LSTM imports
from .lstm import (
    LSTMEncoder,
    PredictionHead,
    LSTMRelevanceModel,
    create_lstm_model,
    count_lstm_parameters,
)

__all__ = [
    # Dual-Tower
    'ContextTower',
    'StockTower',
    'RelevanceHead',
    'DualTowerRelevanceModel',
    'create_dual_tower_model',
    'count_dual_tower_parameters',
    # LSTM
    'LSTMEncoder',
    'PredictionHead',
    'LSTMRelevanceModel',
    'create_lstm_model',
    'count_lstm_parameters',
]
