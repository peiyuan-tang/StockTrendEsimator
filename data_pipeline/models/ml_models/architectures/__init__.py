"""
Model Architectures - Neural network architecture definitions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

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
