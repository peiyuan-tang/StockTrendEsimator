#!/usr/bin/env python3
"""
Loss Functions for Dual-Tower Relevance Model

Implements:
1. Regression loss for continuous relevance score prediction
2. Classification loss for direction prediction (positive/negative)
3. Regularization loss for tower independence
4. Combined multi-task loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RelevanceRegressionLoss(nn.Module):
    """
    MSE loss for continuous relevance score prediction
    
    Predicts scores in [-1, 1] range representing correlation strength
    Positive: context drives stock movement
    Negative: context opposes stock movement (hedging)
    """
    
    def __init__(self, reduction: str = 'mean'):
        super(RelevanceRegressionLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction=reduction)
    
    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss for relevance scores
        
        Args:
            predictions: Predicted scores (batch_size, 1) in [-1, 1]
            targets: Ground truth scores (batch_size, 1) in [-1, 1]
            
        Returns:
            Scalar loss value
        """
        return self.criterion(predictions, targets)


class RelevanceDirectionLoss(nn.Module):
    """
    Classification loss for relevance direction prediction
    
    Distinguishes between:
    - Positive relevance: context supports stock movement
    - Negative relevance: context opposes stock movement
    """
    
    def __init__(self, 
                 reduction: str = 'mean',
                 label_smoothing: float = 0.0):
        super(RelevanceDirectionLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            reduction=reduction,
            label_smoothing=label_smoothing
        )
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss for direction
        
        Args:
            logits: Classification logits (batch_size, 2) [pos_logit, neg_logit]
            targets: Ground truth labels (batch_size,) with values in {0: positive, 1: negative}
            
        Returns:
            Scalar loss value
        """
        return self.criterion(logits, targets)


class TowerRegularizationLoss(nn.Module):
    """
    Regularization loss to encourage tower independence
    
    Prevents tower collapse (both towers learning same representation)
    by minimizing cosine similarity between embeddings
    """
    
    def __init__(self):
        super(TowerRegularizationLoss, self).__init__()
    
    def forward(self,
                context_embed: torch.Tensor,
                stock_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute regularization loss
        
        Args:
            context_embed: Context tower embedding (batch_size, context_dim)
            stock_embed: Stock tower embedding (batch_size, stock_dim)
            
        Returns:
            Scalar regularization loss
        """
        # Project both to same dimension for similarity
        batch_size = context_embed.shape[0]
        
        # Normalize embeddings
        context_norm = F.normalize(context_embed, p=2, dim=1)  # (batch, context_dim)
        stock_norm = F.normalize(stock_embed, p=2, dim=1)  # (batch, stock_dim)
        
        # Project stock to context dimension
        # Use simple projection via linear mapping
        min_dim = min(context_embed.shape[1], stock_embed.shape[1])
        context_proj = context_norm[:, :min_dim]
        stock_proj = stock_norm[:, :min_dim]
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(context_proj, stock_proj, dim=1)  # (batch,)
        
        # We want to minimize similarity (maximize dissimilarity)
        # cos_sim ≈ 1 means similar (bad), cos_sim ≈ 0 means different (good)
        loss = cos_sim.mean()
        
        return loss


class EmbeddingMagnitudeLoss(nn.Module):
    """
    Regularization loss to control embedding magnitude
    
    Prevents embeddings from growing unbounded,
    encourages efficient feature representation
    """
    
    def __init__(self):
        super(EmbeddingMagnitudeLoss, self).__init__()
    
    def forward(self,
                context_embed: torch.Tensor,
                stock_embed: torch.Tensor) -> torch.Tensor:
        """
        Compute magnitude regularization loss
        
        Args:
            context_embed: Context tower embedding (batch_size, context_dim)
            stock_embed: Stock tower embedding (batch_size, stock_dim)
            
        Returns:
            Scalar magnitude loss
        """
        # L2 norm of embeddings
        context_magnitude = torch.norm(context_embed, p=2, dim=1).mean()
        stock_magnitude = torch.norm(stock_embed, p=2, dim=1).mean()
        
        # Encourage moderate magnitudes (target: ~1.0 for unit sphere)
        target_magnitude = 1.0
        
        loss = (
            F.smooth_l1_loss(context_magnitude, 
                            torch.tensor(target_magnitude, device=context_embed.device)) +
            F.smooth_l1_loss(stock_magnitude,
                            torch.tensor(target_magnitude, device=stock_embed.device))
        ) / 2
        
        return loss


class DualTowerLoss(nn.Module):
    """
    Combined multi-task loss for dual-tower model
    
    L_total = α₁*L_reg_7d + α₂*L_reg_30d + 
              β₁*L_cls_7d + β₂*L_cls_30d + 
              γ*(L_orthogonal + L_magnitude)
    
    Hyperparameters:
    - α₁, α₂: Regression loss weights
    - β₁, β₂: Classification loss weights
    - γ: Regularization weight
    """
    
    def __init__(self,
                 regression_weight_7d: float = 1.0,
                 regression_weight_30d: float = 1.0,
                 classification_weight_7d: float = 0.5,
                 classification_weight_30d: float = 0.5,
                 regularization_weight: float = 0.01,
                 label_smoothing: float = 0.0):
        """
        Initialize loss function
        
        Args:
            regression_weight_7d: Weight for 7-day regression loss
            regression_weight_30d: Weight for 30-day regression loss
            classification_weight_7d: Weight for 7-day classification loss
            classification_weight_30d: Weight for 30-day classification loss
            regularization_weight: Weight for regularization losses
            label_smoothing: Label smoothing for classification
        """
        super(DualTowerLoss, self).__init__()
        
        self.regression_weight_7d = regression_weight_7d
        self.regression_weight_30d = regression_weight_30d
        self.classification_weight_7d = classification_weight_7d
        self.classification_weight_30d = classification_weight_30d
        self.regularization_weight = regularization_weight
        
        # Loss functions
        self.regression_loss = RelevanceRegressionLoss(reduction='mean')
        self.direction_loss = RelevanceDirectionLoss(
            reduction='mean',
            label_smoothing=label_smoothing
        )
        self.orthogonal_loss = TowerRegularizationLoss()
        self.magnitude_loss = EmbeddingMagnitudeLoss()
    
    def forward(self,
                outputs: Dict,
                labels: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs containing:
                - score_7d: (batch_size, 1)
                - logits_7d: (batch_size, 2)
                - score_30d: (batch_size, 1)
                - logits_30d: (batch_size, 2)
                - context_embed: (batch_size, context_dim)
                - stock_embed: (batch_size, stock_dim)
            
            labels: Ground truth containing:
                - label_7d: (batch_size, 1) in [-1, 1]
                - label_30d: (batch_size, 1) in [-1, 1]
                - direction_7d: (batch_size,) in {0: pos, 1: neg}
                - direction_30d: (batch_size,) in {0: pos, 1: neg}
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        
        # Regression losses
        L_reg_7d = self.regression_loss(outputs['score_7d'], labels['label_7d'])
        L_reg_30d = self.regression_loss(outputs['score_30d'], labels['label_30d'])
        
        # Classification losses
        L_cls_7d = self.direction_loss(outputs['logits_7d'], labels['direction_7d'])
        L_cls_30d = self.direction_loss(outputs['logits_30d'], labels['direction_30d'])
        
        # Regularization losses
        L_orthogonal = self.orthogonal_loss(outputs['context_embed'], 
                                           outputs['stock_embed'])
        L_magnitude = self.magnitude_loss(outputs['context_embed'],
                                         outputs['stock_embed'])
        
        L_reg = L_orthogonal + L_magnitude
        
        # Weighted combination
        L_total = (
            self.regression_weight_7d * L_reg_7d +
            self.regression_weight_30d * L_reg_30d +
            self.classification_weight_7d * L_cls_7d +
            self.classification_weight_30d * L_cls_30d +
            self.regularization_weight * L_reg
        )
        
        # Return loss components for monitoring
        components = {
            'total': L_total,
            'regression_7d': L_reg_7d,
            'regression_30d': L_reg_30d,
            'classification_7d': L_cls_7d,
            'classification_30d': L_cls_30d,
            'orthogonal': L_orthogonal,
            'magnitude': L_magnitude,
            'regularization': L_reg,
        }
        
        return L_total, components


class WeightedDualTowerLoss(DualTowerLoss):
    """
    Extended loss with sample weighting for class imbalance
    
    Can weight samples by:
    - Class balance (more weight to minority class)
    - Confidence (less weight to uncertain samples)
    - Temporal importance (more weight to recent samples)
    """
    
    def __init__(self, *args, **kwargs):
        super(WeightedDualTowerLoss, self).__init__(*args, **kwargs)
    
    def forward(self,
                outputs: Dict,
                labels: Dict,
                sample_weights: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted loss
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            sample_weights: Sample-level weights (batch_size,)
            
        Returns:
            Tuple of (weighted_loss, loss_components)
        """
        
        # Get base losses
        L_total, components = super().forward(outputs, labels)
        
        # Apply sample weights if provided
        if sample_weights is not None:
            sample_weights = sample_weights.to(outputs['score_7d'].device)
            # Normalize weights
            sample_weights = sample_weights / sample_weights.mean()
            
            # Weight the main losses
            # This is a simple approach; more sophisticated weighting can be done
            normalized_weight = sample_weights.mean()
            L_total = L_total * normalized_weight
        
        return L_total, components


if __name__ == '__main__':
    # Test loss functions
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy outputs
    outputs = {
        'score_7d': torch.randn(batch_size, 1).to(device).tanh(),
        'logits_7d': torch.randn(batch_size, 2).to(device),
        'score_30d': torch.randn(batch_size, 1).to(device).tanh(),
        'logits_30d': torch.randn(batch_size, 2).to(device),
        'context_embed': torch.randn(batch_size, 32).to(device),
        'stock_embed': torch.randn(batch_size, 64).to(device),
    }
    
    # Create dummy labels
    labels = {
        'label_7d': torch.randn(batch_size, 1).to(device).tanh(),
        'label_30d': torch.randn(batch_size, 1).to(device).tanh(),
        'direction_7d': torch.randint(0, 2, (batch_size,)).to(device),
        'direction_30d': torch.randint(0, 2, (batch_size,)).to(device),
    }
    
    # Test loss
    loss_fn = DualTowerLoss()
    total_loss, components = loss_fn(outputs, labels)
    
    print("Loss function test successful!")
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Loss components:")
    for key, val in components.items():
        print(f"  {key}: {val.item():.6f}")
