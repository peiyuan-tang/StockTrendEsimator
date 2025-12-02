#!/usr/bin/env python3
"""
Dual-Tower Model for Context-Stock Trend Relevance Prediction

Predicts the relevance between market context (policy, news, macro) and
stock movements across two time horizons (7-day and 30-day).

Supports both positive relevance (context drives movement) and negative
relevance (hedging/anti-correlation effects).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ContextTower(nn.Module):
    """
    Context Tower: Encodes policy, news, and macro data
    
    Inputs: 25-dimensional context features
    - News (8): sentiment scores, volume, diversity
    - Policy (5): announcement type, urgency, sector impact
    - Macro (12): inflation, rates, GDP, employment, etc.
    
    Outputs: 32-dimensional context embedding
    """
    
    def __init__(self, 
                 input_dim: int = 25,
                 hidden_dims: list = None,
                 embedding_dim: int = 32,
                 dropout_rate: float = 0.2):
        """
        Initialize Context Tower
        
        Args:
            input_dim: Input feature dimension (default: 25)
            hidden_dims: Hidden layer dimensions
            embedding_dim: Output embedding dimension
            dropout_rate: Dropout rate for regularization
        """
        super(ContextTower, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Context features (batch_size, 25)
            
        Returns:
            Context embedding (batch_size, 32)
        """
        return self.network(x)


class StockTower(nn.Module):
    """
    Stock Tower: Encodes financial and technical indicator data
    
    Inputs: 62-dimensional stock features
    - OHLCV (5): Open, High, Low, Close, Volume
    - Technical indicators (20+): RSI, MACD, Bollinger Bands, ATR
    - Returns (5): 1d, 5d, 20d, 60d, 252d returns
    - Volatility (10+): historical vol, realized vol, VIX proxies
    - Volume analysis (15+): VWAP, volume trends
    
    Outputs: 64-dimensional stock embedding
    """
    
    def __init__(self,
                 input_dim: int = 62,
                 hidden_dims: list = None,
                 embedding_dim: int = 64,
                 dropout_rate: float = 0.3):
        """
        Initialize Stock Tower
        
        Args:
            input_dim: Input feature dimension (default: 62)
            hidden_dims: Hidden layer dimensions
            embedding_dim: Output embedding dimension
            dropout_rate: Dropout rate for regularization
        """
        super(StockTower, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final embedding layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Stock features (batch_size, 62)
            
        Returns:
            Stock embedding (batch_size, 64)
        """
        return self.network(x)


class RelevanceHead(nn.Module):
    """
    Relevance Prediction Head for a specific time horizon
    
    Predicts:
    1. Continuous relevance score in [-1, 1]
    2. Probability of positive relevance
    3. Probability of negative relevance
    """
    
    def __init__(self,
                 interaction_dim: int,
                 hidden_dims: list = None,
                 output_dim: int = 3,
                 dropout_rate: float = 0.2):
        """
        Initialize Relevance Head
        
        Args:
            interaction_dim: Dimension of interacted embeddings (context_dim)
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (3: score, pos_prob, neg_prob)
            dropout_rate: Dropout rate
        """
        super(RelevanceHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [16, 8]
        
        self.interaction_dim = interaction_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = interaction_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer: 3 logits [score, pos, neg]
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Interacted embeddings (batch_size, interaction_dim)
            
        Returns:
            Dictionary with:
            - 'score': Relevance score in [-1, 1] (batch_size, 1)
            - 'logits': Classification logits (batch_size, 2)
            - 'pos_prob': Positive probability (batch_size,)
            - 'neg_prob': Negative probability (batch_size,)
        """
        logits = self.network(x)  # (batch_size, 3)
        
        # Split outputs
        score_logit = logits[:, 0:1]  # (batch_size, 1)
        pos_neg_logits = logits[:, 1:3]  # (batch_size, 2)
        
        # Relevance score: map to [-1, 1] via tanh
        score = torch.tanh(score_logit)  # (batch_size, 1)
        
        # Direction probabilities via softmax
        direction_probs = F.softmax(pos_neg_logits, dim=1)  # (batch_size, 2)
        pos_prob = direction_probs[:, 0]  # (batch_size,)
        neg_prob = direction_probs[:, 1]  # (batch_size,)
        
        return {
            'score': score,
            'logits': pos_neg_logits,
            'pos_prob': pos_prob,
            'neg_prob': neg_prob,
        }


class DualTowerRelevanceModel(nn.Module):
    """
    Dual-Tower Model for Context-Stock Trend Relevance Prediction
    
    Architecture:
    - ContextTower: Encodes policy, news, macro data
    - StockTower: Encodes financial/technical data
    - Two RelevanceHeads: 7-day and 30-day predictions
    
    Training objective:
    - Predict continuous relevance score (regression)
    - Predict direction of relevance (classification)
    - Learn diverse tower representations (regularization)
    """
    
    def __init__(self,
                 context_input_dim: int = 25,
                 stock_input_dim: int = 62,
                 context_hidden_dims: list = None,
                 stock_hidden_dims: list = None,
                 context_embedding_dim: int = 32,
                 stock_embedding_dim: int = 64,
                 relevance_hidden_dims: list = None,
                 context_dropout: float = 0.2,
                 stock_dropout: float = 0.3,
                 head_dropout: float = 0.2):
        """
        Initialize Dual-Tower Model
        
        Args:
            context_input_dim: Context features (news + policy + macro)
            stock_input_dim: Stock features (financial + technical)
            context_hidden_dims: Context tower hidden dimensions
            stock_hidden_dims: Stock tower hidden dimensions
            context_embedding_dim: Context embedding dimension
            stock_embedding_dim: Stock embedding dimension
            relevance_hidden_dims: Relevance head hidden dimensions
            context_dropout: Dropout for context tower
            stock_dropout: Dropout for stock tower
            head_dropout: Dropout for relevance heads
        """
        super(DualTowerRelevanceModel, self).__init__()
        
        if context_hidden_dims is None:
            context_hidden_dims = [128, 64, 32]
        if stock_hidden_dims is None:
            stock_hidden_dims = [256, 128, 64]
        if relevance_hidden_dims is None:
            relevance_hidden_dims = [16, 8]
        
        # Towers
        self.context_tower = ContextTower(
            input_dim=context_input_dim,
            hidden_dims=context_hidden_dims,
            embedding_dim=context_embedding_dim,
            dropout_rate=context_dropout
        )
        
        self.stock_tower = StockTower(
            input_dim=stock_input_dim,
            hidden_dims=stock_hidden_dims,
            embedding_dim=stock_embedding_dim,
            dropout_rate=stock_dropout
        )
        
        # Relevance heads for different time horizons
        # Context embedding is used for interaction
        self.relevance_head_7d = RelevanceHead(
            interaction_dim=context_embedding_dim,
            hidden_dims=relevance_hidden_dims,
            output_dim=3,
            dropout_rate=head_dropout
        )
        
        self.relevance_head_30d = RelevanceHead(
            interaction_dim=context_embedding_dim,
            hidden_dims=relevance_hidden_dims,
            output_dim=3,
            dropout_rate=head_dropout
        )
        
        self.context_embedding_dim = context_embedding_dim
        self.stock_embedding_dim = stock_embedding_dim
    
    def forward(self, 
                context_features: torch.Tensor,
                stock_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual towers and relevance heads
        
        Args:
            context_features: Context data (batch_size, 25)
            stock_features: Stock data (batch_size, 62)
            
        Returns:
            Dictionary containing:
            - 7-day predictions: score_7d, logits_7d, pos_prob_7d, neg_prob_7d
            - 30-day predictions: score_30d, logits_30d, pos_prob_30d, neg_prob_30d
            - Embeddings: context_embed, stock_embed
        """
        # Tower forward passes
        context_embed = self.context_tower(context_features)  # (batch, 32)
        stock_embed = self.stock_tower(stock_features)  # (batch, 64)
        
        # Interaction: element-wise multiplication
        # Use context embedding for relevance prediction
        interaction = context_embed  # (batch, 32)
        
        # 7-day relevance prediction
        output_7d = self.relevance_head_7d(interaction)
        
        # 30-day relevance prediction
        output_30d = self.relevance_head_30d(interaction)
        
        return {
            # 7-day outputs
            'score_7d': output_7d['score'],  # (batch, 1)
            'logits_7d': output_7d['logits'],  # (batch, 2)
            'pos_prob_7d': output_7d['pos_prob'],  # (batch,)
            'neg_prob_7d': output_7d['neg_prob'],  # (batch,)
            
            # 30-day outputs
            'score_30d': output_30d['score'],  # (batch, 1)
            'logits_30d': output_30d['logits'],  # (batch, 2)
            'pos_prob_30d': output_30d['pos_prob'],  # (batch,)
            'neg_prob_30d': output_30d['neg_prob'],  # (batch,)
            
            # Embeddings for analysis
            'context_embed': context_embed,  # (batch, 32)
            'stock_embed': stock_embed,  # (batch, 64)
        }
    
    def get_context_embedding(self) -> torch.Tensor:
        """Get context embedding for current batch (used in loss)"""
        # This is typically called after forward pass
        return self._context_embed
    
    def get_stock_embedding(self) -> torch.Tensor:
        """Get stock embedding for current batch (used in loss)"""
        # This is typically called after forward pass
        return self._stock_embed
    
    def extract_embeddings(self,
                          context_features: torch.Tensor,
                          stock_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract tower embeddings without predictions
        
        Useful for analysis and visualization
        
        Args:
            context_features: Context data (batch_size, 25)
            stock_features: Stock data (batch_size, 62)
            
        Returns:
            Tuple of (context_embed, stock_embed)
        """
        context_embed = self.context_tower(context_features)
        stock_embed = self.stock_tower(stock_features)
        return context_embed, stock_embed


def create_model(device: str = 'cuda',
                 **kwargs) -> DualTowerRelevanceModel:
    """
    Factory function to create and configure model
    
    Args:
        device: 'cuda' or 'cpu'
        **kwargs: Additional configuration parameters
        
    Returns:
        DualTowerRelevanceModel instance
    """
    model = DualTowerRelevanceModel(**kwargs)
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    
    logger.info(f"Created DualTowerRelevanceModel on {device}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in model components
    
    Args:
        model: DualTowerRelevanceModel
        
    Returns:
        Dictionary with parameter counts
    """
    return {
        'context_tower': sum(p.numel() for p in model.context_tower.parameters()),
        'stock_tower': sum(p.numel() for p in model.stock_tower.parameters()),
        'relevance_head_7d': sum(p.numel() for p in model.relevance_head_7d.parameters()),
        'relevance_head_30d': sum(p.numel() for p in model.relevance_head_30d.parameters()),
        'total': sum(p.numel() for p in model.parameters()),
    }


if __name__ == '__main__':
    # Test model creation and forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(device=device)
    
    # Create dummy data
    batch_size = 32
    context_features = torch.randn(batch_size, 25).to(device)
    stock_features = torch.randn(batch_size, 62).to(device)
    
    # Forward pass
    outputs = model(context_features, stock_features)
    
    print("Model created successfully!")
    print(f"Device: {device}")
    print(f"Parameters: {count_parameters(model)}")
    print(f"\nOutput shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
    
    # Test inference
    print(f"\n7-day predictions (batch 0):")
    print(f"  Score: {outputs['score_7d'][0].item():.4f}")
    print(f"  Pos Prob: {outputs['pos_prob_7d'][0].item():.4f}")
    print(f"  Neg Prob: {outputs['neg_prob_7d'][0].item():.4f}")
