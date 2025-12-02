#!/usr/bin/env python3
"""
LSTM Model for Time Series Stock Trend Prediction

Implements:
1. Bidirectional LSTM with multi-layer support
2. Attention mechanism for temporal importance weighting
3. Dual-head architecture for trend and volatility prediction
4. Multi-horizon forecasting (7-day and 30-day)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """
    Multi-head attention mechanism for LSTM output
    
    Computes attention weights across time steps to identify
    which historical periods are most important for prediction.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout_rate: float = 0.1):
        """
        Initialize attention layer
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout_rate: Dropout rate for attention weights
        """
        super(AttentionLayer, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, values: torch.Tensor, keys: torch.Tensor, 
                query: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention
        
        Args:
            values: (batch_size, seq_len, hidden_dim)
            keys: (batch_size, seq_len, hidden_dim)
            query: (batch_size, hidden_dim)
            mask: Optional mask for padding
            
        Returns:
            attention_output: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.query(query).unsqueeze(1)  # (batch_size, 1, hidden_dim)
        K = self.key(keys)  # (batch_size, seq_len, hidden_dim)
        V = self.value(values)  # (batch_size, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, 1, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch_size, num_heads, 1, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, 1, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous()  # (batch_size, 1, num_heads, head_dim)
        context = context.view(batch_size, 1, self.hidden_dim)  # (batch_size, 1, hidden_dim)
        context = context.squeeze(1)  # (batch_size, hidden_dim)
        
        # Final linear projection
        output = self.fc_out(context)  # (batch_size, hidden_dim)
        
        # Average attention weights across heads for interpretability
        avg_attention_weights = attention_weights.mean(dim=1).squeeze(1)  # (batch_size, seq_len)
        
        return output, avg_attention_weights


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder for sequence processing
    
    Processes time series data through stacked LSTM layers
    with optional bidirectional processing.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 bidirectional: bool = True):
        """
        Initialize LSTM encoder
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate between layers
            bidirectional: Whether to use bidirectional LSTM
        """
        super(LSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Projection layer if bidirectional
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor, 
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode sequence through LSTM
        
        Args:
            x: (batch_size, seq_len, input_dim)
            hidden: Optional initial hidden state
            
        Returns:
            lstm_output: (batch_size, seq_len, hidden_dim)
            (h_n, c_n): Final hidden and cell states
        """
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        if self.bidirectional:
            # Project bidirectional output back to hidden_dim
            lstm_out = self.projection(lstm_out)  # (batch_size, seq_len, hidden_dim)
            # Concatenate final states from both directions
            h_n = h_n.view(self.num_layers, 2, x.shape[0], self.hidden_dim)
            h_n = h_n[-1]  # Take last layer: (2, batch_size, hidden_dim)
            h_n = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch_size, hidden_dim*2)
            h_n = self.projection(h_n)  # Project to hidden_dim
        
        return lstm_out, (h_n, c_n)


class PredictionHead(nn.Module):
    """
    Prediction head for trend forecasting
    
    Predicts both trend direction and confidence score
    for a specific time horizon.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.2):
        """
        Initialize prediction head
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super(PredictionHead, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output: (trend_score, direction_logits)
        # trend_score: continuous [-1, 1] representing strength
        # direction_logits: [up_logit, down_logit]
        layers.append(nn.Linear(prev_dim, 3))  # 1 trend score + 2 direction logits
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict trend
        
        Args:
            x: (batch_size, input_dim)
            
        Returns:
            trend_score: (batch_size, 1)
            direction_logits: (batch_size, 2)
        """
        output = self.network(x)  # (batch_size, 3)
        trend_score = torch.tanh(output[:, :1])  # (batch_size, 1) in [-1, 1]
        direction_logits = output[:, 1:]  # (batch_size, 2)
        return trend_score, direction_logits


class LSTMTrendPredictor(nn.Module):
    """
    Complete LSTM-based trend prediction model
    
    Architecture:
    1. LSTM encoder for sequence processing
    2. Attention layer for temporal importance weighting
    3. Dual prediction heads for 7-day and 30-day horizons
    """
    
    def __init__(self,
                 input_dim: int = 62,
                 hidden_dim: int = 128,
                 num_lstm_layers: int = 2,
                 num_attention_heads: int = 4,
                 dropout_rate: float = 0.2,
                 bidirectional: bool = True,
                 head_hidden_dims: List[int] = None):
        """
        Initialize LSTM trend predictor
        
        Args:
            input_dim: Input feature dimension (default: 62 - stock features)
            hidden_dim: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            num_attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            head_hidden_dims: Hidden dimensions for prediction heads
        """
        super(LSTMTrendPredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # LSTM encoder
        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout_rate=dropout_rate
        )
        
        # Prediction heads for different horizons
        if head_hidden_dims is None:
            head_hidden_dims = [64, 32]
        
        self.head_7day = PredictionHead(
            input_dim=hidden_dim,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        self.head_30day = PredictionHead(
            input_dim=hidden_dim,
            hidden_dims=head_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for trend prediction
        
        Args:
            x: (batch_size, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions:
            - '7day_trend': (batch_size, 1)
            - '7day_direction': (batch_size, 2)
            - '30day_trend': (batch_size, 1)
            - '30day_direction': (batch_size, 2)
            - 'attention_weights': (batch_size, seq_len)
        """
        # Encode sequence
        lstm_out, (h_n, c_n) = self.encoder(x)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Apply attention
        context, attention_weights = self.attention(
            values=lstm_out,
            keys=lstm_out,
            query=h_n
        )  # context: (batch_size, hidden_dim)
        
        # Apply dropout
        context = self.dropout(context)
        
        # 7-day prediction
        trend_7day, direction_7day = self.head_7day(context)
        
        # 30-day prediction
        trend_30day, direction_30day = self.head_30day(context)
        
        return {
            '7day_trend': trend_7day,
            '7day_direction': direction_7day,
            '30day_trend': trend_30day,
            '30day_direction': direction_30day,
            'attention_weights': attention_weights,
        }


def create_lstm_model(
    input_dim: int = 62,
    hidden_dim: int = 128,
    num_lstm_layers: int = 2,
    num_attention_heads: int = 4,
    dropout_rate: float = 0.2,
    bidirectional: bool = True,
    device: str = 'cuda'
) -> LSTMTrendPredictor:
    """
    Factory function to create LSTM model
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: LSTM hidden dimension
        num_lstm_layers: Number of LSTM layers
        num_attention_heads: Number of attention heads
        dropout_rate: Dropout rate
        bidirectional: Whether to use bidirectional LSTM
        device: Device to place model on ('cuda' or 'cpu')
        
    Returns:
        LSTMTrendPredictor model on specified device
    """
    model = LSTMTrendPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        num_attention_heads=num_attention_heads,
        dropout_rate=dropout_rate,
        bidirectional=bidirectional,
    )
    model = model.to(device)
    logger.info(f"Created LSTM model with {count_lstm_parameters(model):,} parameters")
    return model


def count_lstm_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test model creation and forward pass
    logging.basicConfig(level=logging.INFO)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_lstm_model(device=device)
    
    # Create dummy input: (batch_size, seq_len, input_dim)
    x = torch.randn(32, 12, 62).to(device)  # 12 weeks of data, 62 features
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
