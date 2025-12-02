#!/usr/bin/env python3
"""
Data Loading for Dual-Tower Model

Handles:
1. Loading training data from UnifiedTrainingDataProcessor
2. Separating features: context (news, policy, macro) vs stock (financial, technical)
3. Generating labels: 7-day and 30-day normalized returns
4. Creating PyTorch DataLoaders with proper splitting
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import logging
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class DualTowerDataset(Dataset):
    """
    PyTorch Dataset for Dual-Tower Model
    
    Handles feature separation and label generation
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 label_horizon_days: List[int] = [7, 30],
                 normalize: bool = True):
        """
        Initialize dataset
        
        Args:
            df: Unified training data from UnifiedTrainingDataProcessor
               Must contain columns:
               - ticker, timestamp
               - stock_* features (62 total)
               - news_* features (8 total)
               - macro_* features (12 total)
               - policy_* features (5 total)
            label_horizon_days: Horizons for label generation [7, 30]
            normalize: Whether to normalize features
        """
        self.df = df.reset_index(drop=True)
        self.label_horizon_days = label_horizon_days
        self.normalize = normalize
        
        # Feature separation
        self.context_cols = self._get_context_columns()
        self.stock_cols = self._get_stock_columns()
        
        logger.info(f"Dataset created with {len(df)} samples")
        logger.info(f"Context features: {len(self.context_cols)}")
        logger.info(f"Stock features: {len(self.stock_cols)}")
        
        # Generate labels
        self.labels = self._generate_labels()
        
        # Normalize features
        if normalize:
            self._normalize_features()
    
    def _get_context_columns(self) -> List[str]:
        """Extract context columns (news, policy, macro)"""
        cols = []
        for col in self.df.columns:
            if col.startswith('news_') or col.startswith('policy_') or col.startswith('macro_'):
                cols.append(col)
        return sorted(cols)
    
    def _get_stock_columns(self) -> List[str]:
        """Extract stock columns (financial, technical)"""
        cols = []
        for col in self.df.columns:
            if col.startswith('stock_') and not col.startswith('stock_weekly_'):
                cols.append(col)
        return sorted(cols)
    
    def _generate_labels(self) -> Dict[str, torch.Tensor]:
        """
        Generate labels for each horizon
        
        Label = normalized return over horizon
        label_norm = tanh(label * 10)  # Scale and map to [-1, 1]
        """
        labels = {}
        
        # Assumes 'timestamp' column exists
        if 'timestamp' not in self.df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        # Get close price column (should be stock_close or similar)
        close_col = None
        for col in self.stock_cols:
            if 'close' in col.lower():
                close_col = col
                break
        
        if close_col is None:
            raise ValueError("No close price column found in stock features")
        
        # Generate labels for each horizon
        for horizon in self.label_horizon_days:
            # Calculate return over horizon
            # This is simplified - assumes weekly data with exactly horizon weeks
            # In production, would use actual date calculations
            
            returns = []
            direction_labels = []
            
            for idx in range(len(self.df)):
                future_idx = idx + (horizon // 7)  # Assuming weekly data
                
                if future_idx < len(self.df):
                    current_price = self.df.iloc[idx][close_col]
                    future_price = self.df.iloc[future_idx][close_col]
                    
                    # Calculate return
                    if current_price > 0:
                        ret = (future_price - current_price) / current_price
                    else:
                        ret = 0.0
                    
                    # Normalize to [-1, 1]
                    ret_normalized = np.tanh(ret * 10)
                    returns.append(ret_normalized)
                    
                    # Direction label: positive, negative, or neutral
                    if ret_normalized >= 0.1:
                        direction_labels.append(0)  # Positive
                    elif ret_normalized <= -0.1:
                        direction_labels.append(1)  # Negative
                    else:
                        direction_labels.append(0)  # Default to positive if neutral
                else:
                    # Future data not available - skip
                    returns.append(np.nan)
                    direction_labels.append(-1)  # Invalid
            
            labels[f'return_{horizon}d'] = np.array(returns)
            labels[f'direction_{horizon}d'] = np.array(direction_labels)
        
        return labels
    
    def _normalize_features(self):
        """Normalize features using z-score normalization"""
        # Store mean and std for later use
        self.context_mean = self.df[self.context_cols].mean().values
        self.context_std = self.df[self.context_cols].std().values
        self.stock_mean = self.df[self.stock_cols].mean().values
        self.stock_std = self.df[self.stock_cols].std().values
        
        # Avoid division by zero
        self.context_std[self.context_std < 1e-8] = 1.0
        self.stock_std[self.stock_std < 1e-8] = 1.0
        
        # Normalize in-place
        self.df[self.context_cols] = (
            (self.df[self.context_cols] - self.context_mean) / self.context_std
        )
        self.df[self.stock_cols] = (
            (self.df[self.stock_cols] - self.stock_mean) / self.stock_std
        )
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get single sample
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (context_features, stock_features, labels)
        """
        row = self.df.iloc[idx]
        
        # Extract features
        context = torch.tensor(
            row[self.context_cols].values,
            dtype=torch.float32
        )
        stock = torch.tensor(
            row[self.stock_cols].values,
            dtype=torch.float32
        )
        
        # Extract labels
        labels = {}
        for horizon in self.label_horizon_days:
            ret = self.labels[f'return_{horizon}d'][idx]
            direction = self.labels[f'direction_{horizon}d'][idx]
            
            if not np.isnan(ret) and direction >= 0:
                labels[f'label_{horizon}d'] = torch.tensor(ret, dtype=torch.float32)
                labels[f'direction_{horizon}d'] = torch.tensor(direction, dtype=torch.long)
            else:
                # Invalid sample
                labels[f'label_{horizon}d'] = torch.tensor(0.0, dtype=torch.float32)
                labels[f'direction_{horizon}d'] = torch.tensor(0, dtype=torch.long)
        
        return context, stock, labels


class DualTowerDataModule:
    """
    Data module for train/val/test splitting and DataLoader creation
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 batch_size: int = 32,
                 val_fraction: float = 0.2,
                 test_fraction: float = 0.2,
                 normalize: bool = True,
                 time_aware_split: bool = True,
                 label_horizons: List[int] = [7, 30]):
        """
        Initialize data module
        
        Args:
            df: Unified training data
            batch_size: Batch size for DataLoaders
            val_fraction: Fraction of data for validation
            test_fraction: Fraction of data for testing
            normalize: Whether to normalize features
            time_aware_split: Use temporal split (preserve causality)
            label_horizons: Label generation horizons
        """
        self.df = df
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.normalize = normalize
        self.time_aware_split = time_aware_split
        self.label_horizons = label_horizons
        
        # Split data
        self._split_data()
    
    def _split_data(self):
        """Split data into train/val/test"""
        n = len(self.df)
        
        if self.time_aware_split:
            # Time-aware split: train -> val -> test
            # Preserves temporal order for causality
            train_idx = int(n * (1 - self.val_fraction - self.test_fraction))
            val_idx = int(n * (1 - self.test_fraction))
            
            self.df_train = self.df.iloc[:train_idx]
            self.df_val = self.df.iloc[train_idx:val_idx]
            self.df_test = self.df.iloc[val_idx:]
        else:
            # Random split
            indices = np.arange(n)
            np.random.shuffle(indices)
            
            train_idx = int(n * (1 - self.val_fraction - self.test_fraction))
            val_idx = int(n * (1 - self.test_fraction))
            
            self.df_train = self.df.iloc[indices[:train_idx]]
            self.df_val = self.df.iloc[indices[train_idx:val_idx]]
            self.df_test = self.df.iloc[indices[val_idx:]]
        
        logger.info(f"Data split: Train {len(self.df_train)}, "
                   f"Val {len(self.df_val)}, Test {len(self.df_test)}")
    
    def create_train_loader(self) -> DataLoader:
        """Create training DataLoader"""
        dataset = DualTowerDataset(
            self.df_train,
            label_horizon_days=self.label_horizons,
            normalize=self.normalize
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def create_val_loader(self) -> DataLoader:
        """Create validation DataLoader"""
        dataset = DualTowerDataset(
            self.df_val,
            label_horizon_days=self.label_horizons,
            normalize=self.normalize
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def create_test_loader(self) -> DataLoader:
        """Create test DataLoader"""
        dataset = DualTowerDataset(
            self.df_test,
            label_horizon_days=self.label_horizons,
            normalize=self.normalize
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def create_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create all loaders"""
        return (
            self.create_train_loader(),
            self.create_val_loader(),
            self.create_test_loader()
        )


def create_data_loaders(df: pd.DataFrame,
                       batch_size: int = 32,
                       **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create data loaders
    
    Args:
        df: Unified training DataFrame
        batch_size: Batch size
        **kwargs: Additional arguments for DualTowerDataModule
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = DualTowerDataModule(df, batch_size=batch_size, **kwargs)
    return data_module.create_loaders()


if __name__ == '__main__':
    # Test data loading
    # Create dummy data similar to UnifiedTrainingDataProcessor output
    
    np.random.seed(42)
    n_samples = 240
    
    data = {
        'ticker': np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'], 
                                  n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='W'),
    }
    
    # Add stock features (62)
    for i in range(62):
        data[f'stock_feature_{i}'] = np.random.randn(n_samples)
    
    # Add news features (8)
    for i in range(8):
        data[f'news_feature_{i}'] = np.random.randn(n_samples)
    
    # Add macro features (12)
    for i in range(12):
        data[f'macro_feature_{i}'] = np.random.randn(n_samples)
    
    # Add policy features (5)
    for i in range(5):
        data[f'policy_feature_{i}'] = np.random.randn(n_samples)
    
    df = pd.DataFrame(data)
    
    # Test data loading
    train_loader, val_loader, test_loader = create_data_loaders(df, batch_size=32)
    
    print("Data loading test successful!")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # Test batch
    for context, stock, labels in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Context: {context.shape}")
        print(f"  Stock: {stock.shape}")
        print(f"  Labels: {labels.keys()}")
        for key, val in labels.items():
            print(f"    {key}: {val.shape}")
        break
