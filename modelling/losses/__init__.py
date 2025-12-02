"""
Loss Functions Package

Contains loss function implementations for different models:
- Dual-Tower: Multi-task loss (regression + classification + regularization)
- LSTM: Sequential prediction loss functions
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
