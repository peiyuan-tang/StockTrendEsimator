"""
Model Configurations for Dual-Tower Relevance Model

Centralized configuration for hyperparameters, architecture, and training settings.
"""

from typing import Dict, Any
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


# Default configurations
DEFAULT_MODEL_CONFIG = DualTowerModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_DATA_CONFIG = DataConfig()

if __name__ == '__main__':
    # Example usage
    config_mgr = ConfigManager()
    print(config_mgr.to_dict())
