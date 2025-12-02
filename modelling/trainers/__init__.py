"""
Training Loops and Optimization Package

Contains trainer implementations for different models:
- Dual-Tower: Multi-task training with early stopping
- LSTM: Sequence training with time-aware validation
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Dual-Tower trainer imports
from .dual_tower import (
    DualTowerTrainer,
    create_optimizer as create_dual_tower_optimizer,
    create_scheduler as create_dual_tower_scheduler,
)

# LSTM trainer imports
from .lstm import (
    LSTMTrainer,
    create_lstm_optimizer,
    create_lstm_scheduler,
)

__all__ = [
    # Dual-Tower training
    'DualTowerTrainer',
    'create_dual_tower_optimizer',
    'create_dual_tower_scheduler',
    # LSTM training
    'LSTMTrainer',
    'create_lstm_optimizer',
    'create_lstm_scheduler',
]
