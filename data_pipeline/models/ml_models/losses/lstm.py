#!/usr/bin/env python3
"""
Loss Functions for LSTM Trend Prediction Model

Implements:
1. Trend regression loss (continuous score prediction)
2. Direction classification loss (up/down prediction)
3. Multi-task loss combining both objectives
4. Sample weighting for imbalanced data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TrendRegressionLoss(nn.Module):
    """
    Smooth L1 loss for trend score prediction
    
    Predicts continuous trend scores in [-1, 1]:
    - 1: Strong uptrend
    - 0: Neutral
    - -1: Strong downtrend
    """
    
    def __init__(self, beta: float = 0.1, reduction: str = 'mean'):
        """
        Initialize regression loss
        
        Args:
            beta: Smoothing parameter for SmoothL1Loss
            reduction: 'mean', 'sum', or 'none'
        """
        super(TrendRegressionLoss, self).__init__()
        self.criterion = nn.SmoothL1Loss(beta=beta, reduction=reduction)
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute regression loss
        
        Args:
            predictions: Predicted trend scores (batch_size, 1) in [-1, 1]
            targets: Ground truth trend scores (batch_size, 1) in [-1, 1]
            
        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


class DirectionClassificationLoss(nn.Module):
    """
    Cross-entropy loss for trend direction prediction
    
    Predicts binary direction:
    - 0: Down (or flat)
    - 1: Up
    """
    
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        """
        Initialize classification loss
        
        Args:
            weight: Optional class weights for imbalanced data
            reduction: 'mean', 'sum', or 'none'
        """
        super(DirectionClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss
        
        Args:
            logits: Predicted logits (batch_size, 2)
            targets: Ground truth labels (batch_size,) with values 0 or 1
            
        Returns:
            Scalar loss value
        """
        return self.criterion(logits, targets.long())


class VolatilityAwareLoss(nn.Module):
    """
    Regression loss that weights errors by predicted volatility
    
    Higher weight on low-volatility predictions where the model
    should be more confident.
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize volatility-aware loss
        
        Args:
            reduction: 'mean' or 'sum'
        """
        super(VolatilityAwareLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                volatility: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted regression loss
        
        Args:
            predictions: Predicted trend scores (batch_size, 1)
            targets: Ground truth scores (batch_size, 1)
            volatility: Historical volatility (batch_size, 1)
            
        Returns:
            Scalar loss value
        """
        # Weight inversely by volatility (lower vol = higher weight)
        weights = 1.0 / (volatility + 1e-6)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Smooth L1 loss
        loss = F.smooth_l1_loss(predictions, targets, reduction='none')
        
        # Apply weights
        weighted_loss = loss * weights
        
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss


class LSTMLoss(nn.Module):
    """
    Combined multi-task loss for LSTM model
    
    Combines:
    1. Regression loss for 7-day trend (weight: 1.0)
    2. Classification loss for 7-day direction (weight: 0.5)
    3. Regression loss for 30-day trend (weight: 1.0)
    4. Classification loss for 30-day direction (weight: 0.5)
    """
    
    def __init__(self,
                 regression_weight_7d: float = 1.0,
                 classification_weight_7d: float = 0.5,
                 regression_weight_30d: float = 1.0,
                 classification_weight_30d: float = 0.5):
        """
        Initialize combined loss
        
        Args:
            regression_weight_7d: Weight for 7-day regression
            classification_weight_7d: Weight for 7-day classification
            regression_weight_30d: Weight for 30-day regression
            classification_weight_30d: Weight for 30-day classification
        """
        super(LSTMLoss, self).__init__()
        
        self.regression_weight_7d = regression_weight_7d
        self.classification_weight_7d = classification_weight_7d
        self.regression_weight_30d = regression_weight_30d
        self.classification_weight_30d = classification_weight_30d
        
        self.regression_loss = TrendRegressionLoss()
        self.classification_loss = DirectionClassificationLoss()
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dictionary with model predictions
                - '7day_trend': (batch_size, 1)
                - '7day_direction': (batch_size, 2)
                - '30day_trend': (batch_size, 1)
                - '30day_direction': (batch_size, 2)
            targets: Dictionary with ground truth
                - 'trend_7day': (batch_size, 1)
                - 'direction_7day': (batch_size,)
                - 'trend_30day': (batch_size, 1)
                - 'direction_30day': (batch_size,)
                
        Returns:
            Dictionary with individual and total losses
        """
        # 7-day regression
        loss_7day_reg = self.regression_loss(
            predictions['7day_trend'],
            targets['trend_7day']
        )
        
        # 7-day classification
        loss_7day_cls = self.classification_loss(
            predictions['7day_direction'],
            targets['direction_7day']
        )
        
        # 30-day regression
        loss_30day_reg = self.regression_loss(
            predictions['30day_trend'],
            targets['trend_30day']
        )
        
        # 30-day classification
        loss_30day_cls = self.classification_loss(
            predictions['30day_direction'],
            targets['direction_30day']
        )
        
        # Combined loss
        total_loss = (
            self.regression_weight_7d * loss_7day_reg +
            self.classification_weight_7d * loss_7day_cls +
            self.regression_weight_30d * loss_30day_reg +
            self.classification_weight_30d * loss_30day_cls
        )
        
        return {
            'total_loss': total_loss,
            'loss_7day_regression': loss_7day_reg,
            'loss_7day_classification': loss_7day_cls,
            'loss_30day_regression': loss_30day_reg,
            'loss_30day_classification': loss_30day_cls,
        }


class WeightedLSTMLoss(nn.Module):
    """
    LSTM loss with per-sample weighting
    
    Supports weighted learning for imbalanced datasets or
    when certain samples are more important for prediction.
    """
    
    def __init__(self,
                 regression_weight_7d: float = 1.0,
                 classification_weight_7d: float = 0.5,
                 regression_weight_30d: float = 1.0,
                 classification_weight_30d: float = 0.5):
        """
        Initialize weighted LSTM loss
        
        Args:
            regression_weight_7d: Weight for 7-day regression
            classification_weight_7d: Weight for 7-day classification
            regression_weight_30d: Weight for 30-day regression
            classification_weight_30d: Weight for 30-day classification
        """
        super(WeightedLSTMLoss, self).__init__()
        
        self.regression_weight_7d = regression_weight_7d
        self.classification_weight_7d = classification_weight_7d
        self.regression_weight_30d = regression_weight_30d
        self.classification_weight_30d = classification_weight_30d
        
        # Loss functions with reduction='none' to get per-sample losses
        self.regression_loss = TrendRegressionLoss(reduction='none')
        self.classification_loss = DirectionClassificationLoss(reduction='none')
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                sample_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss
        
        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth
            sample_weights: Per-sample weights (batch_size,)
                
        Returns:
            Dictionary with weighted losses
        """
        # Compute per-sample losses
        loss_7day_reg = self.regression_loss(
            predictions['7day_trend'],
            targets['trend_7day']
        ).squeeze()  # (batch_size,)
        
        loss_7day_cls = self.classification_loss(
            predictions['7day_direction'],
            targets['direction_7day']
        )  # (batch_size,)
        
        loss_30day_reg = self.regression_loss(
            predictions['30day_trend'],
            targets['trend_30day']
        ).squeeze()  # (batch_size,)
        
        loss_30day_cls = self.classification_loss(
            predictions['30day_direction'],
            targets['direction_30day']
        )  # (batch_size,)
        
        # Apply sample weights
        weighted_loss_7day_reg = (loss_7day_reg * sample_weights).mean()
        weighted_loss_7day_cls = (loss_7day_cls * sample_weights).mean()
        weighted_loss_30day_reg = (loss_30day_reg * sample_weights).mean()
        weighted_loss_30day_cls = (loss_30day_cls * sample_weights).mean()
        
        # Combined weighted loss
        total_loss = (
            self.regression_weight_7d * weighted_loss_7day_reg +
            self.classification_weight_7d * weighted_loss_7day_cls +
            self.regression_weight_30d * weighted_loss_30day_reg +
            self.classification_weight_30d * weighted_loss_30day_cls
        )
        
        return {
            'total_loss': total_loss,
            'loss_7day_regression': weighted_loss_7day_reg,
            'loss_7day_classification': weighted_loss_7day_cls,
            'loss_30day_regression': weighted_loss_30day_reg,
            'loss_30day_classification': weighted_loss_30day_cls,
        }


if __name__ == '__main__':
    # Test loss functions
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy predictions and targets
    batch_size = 32
    predictions = {
        '7day_trend': torch.randn(batch_size, 1),
        '7day_direction': torch.randn(batch_size, 2),
        '30day_trend': torch.randn(batch_size, 1),
        '30day_direction': torch.randn(batch_size, 2),
    }
    
    targets = {
        'trend_7day': torch.randn(batch_size, 1),
        'direction_7day': torch.randint(0, 2, (batch_size,)),
        'trend_30day': torch.randn(batch_size, 1),
        'direction_30day': torch.randint(0, 2, (batch_size,)),
    }
    
    # Test combined loss
    loss_fn = LSTMLoss()
    loss_dict = loss_fn(predictions, targets)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    
    # Test weighted loss
    sample_weights = torch.ones(batch_size)
    weighted_loss_fn = WeightedLSTMLoss()
    weighted_loss_dict = weighted_loss_fn(predictions, targets, sample_weights)
    
    print("\nWeighted loss components:")
    for key, value in weighted_loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
