#!/usr/bin/env python3
"""
Training Loop for LSTM Model

Implements:
1. Complete training loop with validation
2. Learning rate scheduling (cosine annealing with warmup)
3. Early stopping and checkpointing
4. Gradient clipping and optimization
5. Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class LSTMTrainer:
    """
    Trainer class for LSTM Model
    
    Handles training, validation, and evaluation with
    early stopping, checkpointing, and comprehensive logging.
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_fn,
                 optimizer: optim.Optimizer,
                 scheduler,
                 device: str = 'cuda',
                 checkpoint_dir: str = './checkpoints',
                 max_grad_norm: float = 1.0):
        """
        Initialize trainer
        
        Args:
            model: LSTMTrendPredictor
            loss_fn: Loss function (LSTMLoss)
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_dir: Directory for saving checkpoints
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        all_predictions = {'7day': [], '30day': []}
        all_targets = {'7day': [], '30day': []}
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            # Move to device
            sequences = sequences.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Compute loss
            loss_dict = self.loss_fn(predictions, labels)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            batch_size = sequences.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Store predictions for metrics
            all_predictions['7day'].append(predictions['7day_trend'].detach().cpu().numpy())
            all_predictions['30day'].append(predictions['30day_trend'].detach().cpu().numpy())
            all_targets['7day'].append(labels['trend_7day'].detach().cpu().numpy())
            all_targets['30day'].append(labels['trend_30day'].detach().cpu().numpy())
            
            self.global_step += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / total_samples
                logger.debug(f"Epoch {self.epoch}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {avg_loss:.6f}")
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        all_predictions = {'7day': [], '30day': []}
        all_targets = {'7day': [], '30day': []}
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                # Move to device
                sequences = sequences.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Compute loss
                loss_dict = self.loss_fn(predictions, labels)
                loss = loss_dict['total_loss']
                
                # Track loss
                batch_size = sequences.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Store predictions
                all_predictions['7day'].append(predictions['7day_trend'].cpu().numpy())
                all_predictions['30day'].append(predictions['30day_trend'].cpu().numpy())
                all_targets['7day'].append(labels['trend_7day'].cpu().numpy())
                all_targets['30day'].append(labels['trend_30day'].cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / total_samples
        
        metrics = self._compute_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _compute_metrics(self, predictions: Dict[str, List], targets: Dict[str, List]) -> Dict[str, float]:
        """
        Compute evaluation metrics
        
        Args:
            predictions: Dictionary with predictions for each horizon
            targets: Dictionary with targets for each horizon
            
        Returns:
            Dictionary with computed metrics
        """
        metrics = {}
        
        for horizon in ['7day', '30day']:
            pred = np.concatenate(predictions[horizon])  # (N, 1)
            targ = np.concatenate(targets[horizon])  # (N, 1)
            
            # MSE
            mse = np.mean((pred - targ) ** 2)
            metrics[f'mse_{horizon}'] = mse
            
            # RMSE
            rmse = np.sqrt(mse)
            metrics[f'rmse_{horizon}'] = rmse
            
            # MAE
            mae = np.mean(np.abs(pred - targ))
            metrics[f'mae_{horizon}'] = mae
            
            # Direction accuracy
            pred_direction = (pred > 0).astype(int)
            targ_direction = (targ > 0).astype(int)
            direction_acc = np.mean(pred_direction == targ_direction)
            metrics[f'direction_acc_{horizon}'] = direction_acc
            
            # Correlation
            if np.std(pred) > 0 and np.std(targ) > 0:
                correlation = np.corrcoef(pred.flatten(), targ.flatten())[0, 1]
                metrics[f'correlation_{horizon}'] = correlation
            else:
                metrics[f'correlation_{horizon}'] = 0.0
        
        return metrics
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 100,
            early_stopping_patience: int = 15,
            min_delta: float = 1e-4) -> Dict[str, List]:
        """
        Train model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        self.patience_counter = 0
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            logger.info(f"  Train Loss: {train_metrics['loss']:.6f}")
            logger.info(f"  Val Loss: {val_metrics['loss']:.6f}")
            logger.info(f"  7-day MAE: {val_metrics['mae_7day']:.6f}, Direction Acc: {val_metrics['direction_acc_7day']:.4f}")
            logger.info(f"  30-day MAE: {val_metrics['mae_30day']:.6f}, Direction Acc: {val_metrics['direction_acc_30day']:.4f}")
            
            # Early stopping and checkpointing
            if val_metrics['loss'] < self.best_val_loss - min_delta:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save checkpoint
                self.save_checkpoint(is_best=True)
                logger.info(f"  ✓ New best validation loss: {self.best_val_loss:.6f}")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"\nEarly stopping after {epoch+1} epochs")
                    break
        
        logger.info("\nTraining complete!")
        return self.history
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    
    def plot_history(self, output_path: Optional[str] = None):
        """
        Plot training history
        
        Args:
            output_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.history['learning_rates'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True)
        
        # 7-day metrics
        val_mae_7d = [m['mae_7day'] for m in self.history['val_metrics']]
        val_dir_7d = [m['direction_acc_7day'] for m in self.history['val_metrics']]
        axes[1, 0].plot(val_mae_7d, label='MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('7-Day Validation MAE')
        axes[1, 0].grid(True)
        
        # 30-day metrics
        val_mae_30d = [m['mae_30day'] for m in self.history['val_metrics']]
        val_dir_30d = [m['direction_acc_30day'] for m in self.history['val_metrics']]
        axes[1, 1].plot(val_mae_30d, label='MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('30-Day Validation MAE')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training plot: {output_path}")
        else:
            plt.show()


def create_lstm_optimizer(model: nn.Module,
                         learning_rate: float = 0.001,
                         weight_decay: float = 1e-5) -> optim.Adam:
    """
    Create optimizer for LSTM model
    
    Args:
        model: LSTM model
        learning_rate: Learning rate
        weight_decay: Weight decay for L2 regularization
        
    Returns:
        Adam optimizer
    """
    return optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def create_lstm_scheduler(optimizer: optim.Optimizer,
                         total_epochs: int = 100,
                         warmup_epochs: int = 5,
                         scheduler_type: str = 'cosine') -> optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler for LSTM
    
    Args:
        optimizer: Optimizer
        total_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        scheduler_type: 'cosine' or 'plateau'
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_epochs - warmup_epochs,
            T_mult=1,
            eta_min=1e-6
        )
    elif scheduler_type == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == '__main__':
    # Test trainer (requires data)
    logging.basicConfig(level=logging.INFO)
    logger.info("LSTM Trainer module loaded successfully")
