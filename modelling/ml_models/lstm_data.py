#!/usr/bin/env python3
"""
Data Loading for LSTM Model

Implements:
1. Sequence dataset for time series data
2. Sequence batching with padding
3. Train/val/test splitting with temporal awareness
4. Data normalization and feature separation
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class LSTMSequenceDataset(Dataset):
    """
    LSTM Sequence Dataset
    
    Converts time series data into sequences for LSTM input.
    Each sample consists of a sequence of historical observations
    and corresponding labels for multiple prediction horizons.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 sequence_length: int = 12,
                 label_horizons: List[int] = None,
                 feature_columns: Optional[List[str]] = None,
                 label_columns: Optional[Dict[str, str]] = None,
                 normalize: bool = True,
                 scaler: Optional[StandardScaler] = None):
        """
        Initialize LSTM sequence dataset
        
        Args:
            data: DataFrame with features and labels (index must be datetime)
            sequence_length: Number of time steps in each sequence (default: 12 weeks)
            label_horizons: Prediction horizons in days (default: [7, 30])
            feature_columns: List of feature column names to use
            label_columns: Dict mapping horizon to label column names
            normalize: Whether to normalize features
            scaler: Pre-fitted scaler (if None, will fit on data)
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.label_horizons = label_horizons or [7, 30]
        self.normalize = normalize
        
        # Identify feature columns
        if feature_columns is None:
            # Assume all numeric columns except labels are features
            label_cols = set()
            if label_columns:
                label_cols.update(label_columns.values())
            self.feature_columns = [col for col in self.data.columns 
                                   if col not in label_cols and self.data[col].dtype in [np.float32, np.float64, int]]
        else:
            self.feature_columns = feature_columns
        
        # Identify label columns
        if label_columns is None:
            self.label_columns = {
                7: 'trend_7day',
                30: 'trend_30day'
            }
        else:
            self.label_columns = label_columns
        
        # Normalize features
        if self.normalize:
            if scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(self.data[self.feature_columns])
            else:
                self.scaler = scaler
            
            self.data[self.feature_columns] = self.scaler.transform(self.data[self.feature_columns])
        else:
            self.scaler = None
        
        # Create sequences
        self.sequences = []
        self.labels = []
        
        self._create_sequences()
        
        logger.info(f"Created {len(self.sequences)} sequences of length {sequence_length}")
    
    def _create_sequences(self):
        """Create sequences from data"""
        data_values = self.data[self.feature_columns].values  # (T, F)
        
        for i in range(len(self.data) - self.sequence_length - max(self.label_horizons) // 5):
            # Extract sequence
            seq = data_values[i:i + self.sequence_length]  # (seq_len, num_features)
            
            # Extract labels
            label_dict = {}
            for horizon in self.label_horizons:
                horizon_steps = horizon // 7  # Convert days to weeks
                label_idx = i + self.sequence_length + horizon_steps - 1
                
                if label_idx < len(self.data):
                    label_col = self.label_columns.get(horizon, f'trend_{horizon}day')
                    if label_col in self.data.columns:
                        label_dict[f'trend_{horizon}day'] = self.data.iloc[label_idx][label_col]
                        
                        # Direction label: 1 if uptrend, 0 if downtrend
                        label_dict[f'direction_{horizon}day'] = 1 if self.data.iloc[label_idx][label_col] > 0 else 0
            
            if len(label_dict) == 2 * len(self.label_horizons):  # Have labels for all horizons
                self.sequences.append(seq)
                self.labels.append(label_dict)
    
    def __len__(self) -> int:
        """Return number of sequences"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get sequence and labels
        
        Args:
            idx: Index of sequence
            
        Returns:
            (sequence, labels) where:
            - sequence: (seq_len, num_features)
            - labels: Dict with trend and direction labels
        """
        seq = torch.FloatTensor(self.sequences[idx])
        labels = {k: torch.FloatTensor([v]) if 'trend' in k else torch.LongTensor([v])
                 for k, v in self.labels[idx].items()}
        
        return seq, labels


class LSTMDataModule:
    """
    Data module for LSTM training
    
    Handles train/val/test splitting with temporal awareness
    and creates data loaders with proper batching.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 sequence_length: int = 12,
                 label_horizons: List[int] = None,
                 train_fraction: float = 0.7,
                 val_fraction: float = 0.15,
                 test_fraction: float = 0.15,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 normalize: bool = True,
                 time_aware_split: bool = True):
        """
        Initialize LSTM data module
        
        Args:
            data: DataFrame with features and labels
            sequence_length: Number of time steps in each sequence
            label_horizons: Prediction horizons
            train_fraction: Fraction for training
            val_fraction: Fraction for validation
            test_fraction: Fraction for testing
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            normalize: Whether to normalize features
            time_aware_split: Whether to use time-aware split
        """
        self.data = data
        self.sequence_length = sequence_length
        self.label_horizons = label_horizons or [7, 30]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.time_aware_split = time_aware_split
        
        # Split data
        self.train_data, self.val_data, self.test_data = self._split_data(
            train_fraction, val_fraction, test_fraction
        )
        
        # Fit scaler on training data
        if self.normalize:
            self.scaler = StandardScaler()
            feature_cols = [col for col in self.train_data.columns if col not in ['trend_7day', 'trend_30day']]
            self.scaler.fit(self.train_data[feature_cols])
        else:
            self.scaler = None
        
        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self._create_datasets()
    
    def _split_data(self, train_frac: float, val_frac: float, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets
        
        Uses time-aware split to prevent data leakage.
        """
        if self.time_aware_split:
            # Time-aware split: chronological order
            n = len(self.data)
            train_end = int(n * train_frac)
            val_end = train_end + int(n * val_frac)
            
            train_data = self.data.iloc[:train_end]
            val_data = self.data.iloc[train_end:val_end]
            test_data = self.data.iloc[val_end:]
        else:
            # Random split
            indices = np.random.permutation(len(self.data))
            n = len(self.data)
            train_idx = indices[:int(n * train_frac)]
            val_idx = indices[int(n * train_frac):int(n * (train_frac + val_frac))]
            test_idx = indices[int(n * (train_frac + val_frac)):]
            
            train_data = self.data.iloc[train_idx].sort_index()
            val_data = self.data.iloc[val_idx].sort_index()
            test_data = self.data.iloc[test_idx].sort_index()
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        return train_data, val_data, test_data
    
    def _create_datasets(self):
        """Create LSTM datasets"""
        self.train_dataset = LSTMSequenceDataset(
            self.train_data,
            sequence_length=self.sequence_length,
            label_horizons=self.label_horizons,
            normalize=self.normalize,
            scaler=self.scaler
        )
        
        self.val_dataset = LSTMSequenceDataset(
            self.val_data,
            sequence_length=self.sequence_length,
            label_horizons=self.label_horizons,
            normalize=self.normalize,
            scaler=self.scaler
        )
        
        self.test_dataset = LSTMSequenceDataset(
            self.test_data,
            sequence_length=self.sequence_length,
            label_horizons=self.label_horizons,
            normalize=self.normalize,
            scaler=self.scaler
        )
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def create_lstm_data_loaders(
    data: pd.DataFrame,
    sequence_length: int = 12,
    batch_size: int = 32,
    normalize: bool = True,
    time_aware_split: bool = True,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create LSTM data loaders
    
    Args:
        data: DataFrame with features and labels
        sequence_length: Number of time steps in sequence
        batch_size: Batch size for loading
        normalize: Whether to normalize features
        time_aware_split: Whether to use time-aware split
        train_fraction: Training set fraction
        val_fraction: Validation set fraction
        test_fraction: Test set fraction
        num_workers: Number of workers for data loading
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    data_module = LSTMDataModule(
        data=data,
        sequence_length=sequence_length,
        batch_size=batch_size,
        normalize=normalize,
        time_aware_split=time_aware_split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        num_workers=num_workers,
    )
    
    return (
        data_module.get_train_loader(),
        data_module.get_val_loader(),
        data_module.get_test_loader(),
    )


if __name__ == '__main__':
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    dates = pd.date_range('2020-01-01', periods=500, freq='W')
    data = pd.DataFrame({
        f'feature_{i}': np.random.randn(500) for i in range(62)
    }, index=dates)
    
    data['trend_7day'] = np.random.randn(500) * 0.5
    data['trend_30day'] = np.random.randn(500) * 0.3
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_lstm_data_loaders(
        data,
        sequence_length=12,
        batch_size=32,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check batch structure
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Sequences shape: {sequences.shape}")
        for key, value in labels.items():
            print(f"  Labels {key} shape: {value.shape}")
        if batch_idx == 0:
            break
