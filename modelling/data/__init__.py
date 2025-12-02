"""
Data Loading and Preprocessing Package

Contains data loading implementations for different models:
- Dual-Tower: Feature separation and multi-horizon label generation
- LSTM: Sequence generation for recurrent models
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Dual-Tower data imports
from .dual_tower import (
    DualTowerDataset,
    DualTowerDataModule,
    create_data_loaders as create_dual_tower_data_loaders,
)

# LSTM data imports
from .lstm import (
    LSTMDataset,
    LSTMDataModule,
    create_lstm_data_loaders,
)

__all__ = [
    # Dual-Tower data
    'DualTowerDataset',
    'DualTowerDataModule',
    'create_dual_tower_data_loaders',
    # LSTM data
    'LSTMDataset',
    'LSTMDataModule',
    'create_lstm_data_loaders',
]
