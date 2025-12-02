"""
Loss Functions Package - Training objective functions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Dual-Tower loss imports
from .dual_tower import (
    RelevanceRegressionLoss,
    RelevanceDirectionLoss,
    TowerRegularizationLoss,
    EmbeddingMagnitudeLoss,
    DualTowerLoss,
    WeightedDualTowerLoss,
)

# LSTM loss imports
from .lstm import (
    LSTMRegressionLoss,
    LSTMDirectionLoss,
    LSTMSequenceLoss,
    LSTMMultiTaskLoss,
    WeightedLSTMLoss,
)

__all__ = [
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
]
