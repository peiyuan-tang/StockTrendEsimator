"""
Model Configurations for Stock Trend Prediction Models

Centralized configuration for:
- Dual-Tower Model (context + stock tower architecture)
- LSTM Model (sequence-based time series prediction)

Supports hyperparameter management, architecture configuration, and training settings.
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field
import yaml
from pathlib import Path


@dataclass
class ContextTowerConfig:
    """Configuration for Context Tower"""
    input_dim: int = 25
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    embedding_dim: int = 32
    dropout_rate: float = 0.2
    activation: str = 'relu'
    normalization: str = 'batch'


@dataclass
class StockTowerConfig:
    """Configuration for Stock Tower"""
    input_dim: int = 62
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    embedding_dim: int = 64
    dropout_rate: float = 0.3
    activation: str = 'relu'
    normalization: str = 'batch'


@dataclass
class RelevanceHeadConfig:
    """Configuration for Relevance Prediction Heads"""
    hidden_dims: list = field(default_factory=lambda: [16, 8])
    output_dim: int = 3  # score, pos_prob, neg_prob
    dropout_rate: float = 0.2
    activation: str = 'relu'


@dataclass
class DualTowerModelConfig:
    """Complete configuration for Dual-Tower Model"""
    context_tower: ContextTowerConfig = field(default_factory=ContextTowerConfig)
    stock_tower: StockTowerConfig = field(default_factory=StockTowerConfig)
    relevance_head: RelevanceHeadConfig = field(default_factory=RelevanceHeadConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'context_tower': {
                'input_dim': self.context_tower.input_dim,
                'hidden_dims': self.context_tower.hidden_dims,
                'embedding_dim': self.context_tower.embedding_dim,
                'dropout_rate': self.context_tower.dropout_rate,
            },
            'stock_tower': {
                'input_dim': self.stock_tower.input_dim,
                'hidden_dims': self.stock_tower.hidden_dims,
                'embedding_dim': self.stock_tower.embedding_dim,
                'dropout_rate': self.stock_tower.dropout_rate,
            },
            'relevance_head': {
                'hidden_dims': self.relevance_head.hidden_dims,
                'output_dim': self.relevance_head.output_dim,
                'dropout_rate': self.relevance_head.dropout_rate,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DualTowerModelConfig':
        """Create from dictionary"""
        return cls(
            context_tower=ContextTowerConfig(**config_dict.get('context_tower', {})),
            stock_tower=StockTowerConfig(**config_dict.get('stock_tower', {})),
            relevance_head=RelevanceHeadConfig(**config_dict.get('relevance_head', {})),
        )


@dataclass
class LossConfig:
    """Configuration for Loss Function"""
    regression_weight_7d: float = 1.0
    regression_weight_30d: float = 1.0
    classification_weight_7d: float = 0.5
    classification_weight_30d: float = 0.5
    regularization_weight: float = 0.01
    label_smoothing: float = 0.0


@dataclass
class OptimizerConfig:
    """Configuration for Optimizer"""
    optimizer_type: str = 'adam'
    base_learning_rate: float = 0.001
    context_tower_lr: float = 0.001
    stock_tower_lr: float = 0.0005
    head_lr: float = 0.001
    weight_decay: float = 1e-5
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for Learning Rate Scheduler"""
    scheduler_type: str = 'cosine'  # 'cosine' or 'plateau'
    total_epochs: int = 100
    warmup_epochs: int = 5
    eta_min: float = 1e-6
    plateau_factor: float = 0.5
    plateau_patience: int = 5


@dataclass
class TrainingConfig:
    """Complete training configuration"""
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    normalize_features: bool = True
    time_aware_split: bool = True
    
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class DataConfig:
    """Configuration for Data Loading"""
    data_root: str = '/data'
    raw_data_path: str = '/data/raw'
    context_data_path: str = '/data/context'
    training_data_path: str = '/data/training'
    
    tickers: list = field(default_factory=lambda: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'])
    label_horizons: list = field(default_factory=lambda: [7, 30])
    default_lookback_days: int = 84  # 12 weeks
    weekly_granularity: int = 604800  # seconds


class ConfigManager:
    """Manages all configurations"""
    
    def __init__(self):
        self.model_config = DualTowerModelConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert all configs to dictionary"""
        return {
            'model': self.model_config.to_dict(),
            'training': {
                'batch_size': self.training_config.batch_size,
                'epochs': self.training_config.epochs,
                'early_stopping_patience': self.training_config.early_stopping_patience,
                'loss': {
                    'regression_7d': self.training_config.loss.regression_weight_7d,
                    'regression_30d': self.training_config.loss.regression_weight_30d,
                    'classification_7d': self.training_config.loss.classification_weight_7d,
                    'classification_30d': self.training_config.loss.classification_weight_30d,
                    'regularization': self.training_config.loss.regularization_weight,
                },
                'optimizer': {
                    'type': self.training_config.optimizer.optimizer_type,
                    'base_lr': self.training_config.optimizer.base_learning_rate,
                    'weight_decay': self.training_config.optimizer.weight_decay,
                },
                'scheduler': {
                    'type': self.training_config.scheduler.scheduler_type,
                    'total_epochs': self.training_config.scheduler.total_epochs,
                    'warmup_epochs': self.training_config.scheduler.warmup_epochs,
                }
            },
            'data': {
                'data_root': self.data_config.data_root,
                'tickers': self.data_config.tickers,
                'label_horizons': self.data_config.label_horizons,
            }
        }
    
    def save_config(self, path: str):
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, path: str) -> 'ConfigManager':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config_manager = cls()
        # Update from loaded config
        if 'model' in config_dict:
            config_manager.model_config = DualTowerModelConfig.from_dict(config_dict['model'])
        # Could add more sophisticated loading for training/data configs
        return config_manager


# ============================================================================
# LSTM Model Configuration
# ============================================================================

@dataclass
class LSTMEncoderConfig:
    """Configuration for LSTM Encoder"""
    input_dim: int = 62
    hidden_dim: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    bidirectional: bool = True


@dataclass
class AttentionConfig:
    """Configuration for Attention Layer"""
    hidden_dim: int = 128
    num_heads: int = 4
    dropout_rate: float = 0.1


@dataclass
class LSTMPredictionHeadConfig:
    """Configuration for LSTM Prediction Head"""
    input_dim: int = 128
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    output_dim: int = 3  # trend_score + 2 direction logits
    dropout_rate: float = 0.2


@dataclass
class LSTMModelConfig:
    """Complete configuration for LSTM Trend Predictor"""
    input_dim: int = 62
    hidden_dim: int = 128
    num_lstm_layers: int = 2
    num_attention_heads: int = 4
    dropout_rate: float = 0.2
    bidirectional: bool = True
    head_hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    
    encoder: LSTMEncoderConfig = field(default_factory=LSTMEncoderConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    prediction_head: LSTMPredictionHeadConfig = field(default_factory=LSTMPredictionHeadConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_lstm_layers': self.num_lstm_layers,
            'num_attention_heads': self.num_attention_heads,
            'dropout_rate': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'head_hidden_dims': self.head_hidden_dims,
            'encoder': {
                'input_dim': self.encoder.input_dim,
                'hidden_dim': self.encoder.hidden_dim,
                'num_layers': self.encoder.num_layers,
                'dropout_rate': self.encoder.dropout_rate,
                'bidirectional': self.encoder.bidirectional,
            },
            'attention': {
                'hidden_dim': self.attention.hidden_dim,
                'num_heads': self.attention.num_heads,
                'dropout_rate': self.attention.dropout_rate,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LSTMModelConfig':
        """Create from dictionary"""
        return cls(
            input_dim=config_dict.get('input_dim', 62),
            hidden_dim=config_dict.get('hidden_dim', 128),
            num_lstm_layers=config_dict.get('num_lstm_layers', 2),
            num_attention_heads=config_dict.get('num_attention_heads', 4),
            dropout_rate=config_dict.get('dropout_rate', 0.2),
            bidirectional=config_dict.get('bidirectional', True),
            head_hidden_dims=config_dict.get('head_hidden_dims', [64, 32]),
        )


@dataclass
class LSTMLossConfig:
    """Configuration for LSTM Loss Function"""
    regression_weight_7d: float = 1.0
    classification_weight_7d: float = 0.5
    regression_weight_30d: float = 1.0
    classification_weight_30d: float = 0.5


@dataclass
class LSTMSequenceConfig:
    """Configuration for LSTM Data Sequences"""
    sequence_length: int = 12  # weeks
    label_horizons: List[int] = field(default_factory=lambda: [7, 30])
    normalize_features: bool = True
    feature_scaler: str = 'standard'  # 'standard' or 'minmax'


@dataclass
class LSTMTrainingConfig:
    """Complete training configuration for LSTM"""
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    max_grad_norm: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    time_aware_split: bool = True
    
    loss: LSTMLossConfig = field(default_factory=LSTMLossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    sequence: LSTMSequenceConfig = field(default_factory=LSTMSequenceConfig)


# Default configurations
DEFAULT_MODEL_CONFIG = DualTowerModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_LSTM_MODEL_CONFIG = LSTMModelConfig()
DEFAULT_LSTM_TRAINING_CONFIG = LSTMTrainingConfig()

if __name__ == '__main__':
    # Example usage
    config_mgr = ConfigManager()
    print(config_mgr.to_dict())
