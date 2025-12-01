#!/usr/bin/env python3
"""
Training Loop for Dual-Tower Model

Implements:
1. Complete training loop with validation
2. Learning rate scheduling (cosine annealing with warmup)
3. Early stopping and checkpointing
4. Gradient clipping and optimization
5. Evaluation metrics
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

logger = logging.getLogger(__name__)


class DualTowerTrainer:
    """
    Trainer class for Dual-Tower Model
    
    Handles training, validation, and evaluation
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
            model: DualTowerRelevanceModel
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: 'cuda' or 'cpu'
            checkpoint_dir: Directory to save checkpoints
            max_grad_norm: Max gradient norm for clipping
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_grad_norm = max_grad_norm
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, 
                   train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of loss components
        """
        self.model.train()
        
        total_loss = 0
        loss_components = {}
        n_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            context, stock, labels = batch
            context = context.to(self.device)
            stock = stock.to(self.device)
            
            # Move labels to device
            labels_device = {}
            for key, val in labels.items():
                labels_device[key] = val.to(self.device)
            
            # Forward pass
            outputs = self.model(context, stock)
            
            # Compute loss
            loss, components = self.loss_fn(outputs, labels_device)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm
            )
            
            # Update
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            n_batches += 1
            
            # Accumulate components
            for key, val in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += val.item()
        
        # Average
        avg_loss = total_loss / n_batches
        for key in loss_components:
            loss_components[key] /= n_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }
    
    def validate(self, 
                val_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate on validation set
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (avg_loss, metrics)
        """
        self.model.eval()
        
        total_loss = 0
        loss_components = {}
        
        predictions_7d = []
        predictions_30d = []
        labels_7d = []
        labels_30d = []
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                context, stock, labels = batch
                context = context.to(self.device)
                stock = stock.to(self.device)
                
                # Move labels to device
                labels_device = {}
                for key, val in labels.items():
                    labels_device[key] = val.to(self.device)
                
                # Forward pass
                outputs = self.model(context, stock)
                
                # Compute loss
                loss, components = self.loss_fn(outputs, labels_device)
                
                total_loss += loss.item()
                n_batches += 1
                
                # Accumulate components
                for key, val in components.items():
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += val.item()
                
                # Collect predictions for metrics
                predictions_7d.append(outputs['score_7d'].cpu())
                predictions_30d.append(outputs['score_30d'].cpu())
                labels_7d.append(labels_device['label_7d'].cpu())
                labels_30d.append(labels_device['label_30d'].cpu())
        
        # Average
        avg_loss = total_loss / n_batches
        for key in loss_components:
            loss_components[key] /= n_batches
        
        # Compute metrics
        pred_7d = torch.cat(predictions_7d, dim=0).squeeze()
        pred_30d = torch.cat(predictions_30d, dim=0).squeeze()
        label_7d = torch.cat(labels_7d, dim=0).squeeze()
        label_30d = torch.cat(labels_30d, dim=0).squeeze()
        
        metrics = {
            'mse_7d': ((pred_7d - label_7d) ** 2).mean().item(),
            'mse_30d': ((pred_30d - label_30d) ** 2).mean().item(),
            'mae_7d': torch.abs(pred_7d - label_7d).mean().item(),
            'mae_30d': torch.abs(pred_30d - label_30d).mean().item(),
            'correlation_7d': self._compute_correlation(pred_7d, label_7d),
            'correlation_30d': self._compute_correlation(pred_30d, label_30d),
        }
        
        return avg_loss, {**loss_components, **metrics}
    
    @staticmethod
    def _compute_correlation(pred: torch.Tensor, 
                           target: torch.Tensor) -> float:
        """Compute Pearson correlation"""
        if len(pred) < 2:
            return 0.0
        
        pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
        target_norm = (target - target.mean()) / (target.std() + 1e-8)
        corr = (pred_norm * target_norm).mean().item()
        
        return corr
    
    def save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        checkpoint_path = self.checkpoint_dir / f'model_epoch_{epoch:03d}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only best 3 checkpoints
        all_checkpoints = sorted(self.checkpoint_dir.glob('model_epoch_*.pt'))
        if len(all_checkpoints) > 3:
            for old_ckpt in all_checkpoints[:-3]:
                old_ckpt.unlink()
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             epochs: int = 100,
             early_stopping_patience: int = 15) -> Dict:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.validate(val_loader)
            
            # Learning rate update
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(f"Epoch {epoch+1:3d}/{epochs}")
            logger.info(f"  Train Loss: {train_metrics['total_loss']:.6f}")
            logger.info(f"  Val Loss:   {val_loss:.6f}")
            logger.info(f"  LR:         {current_lr:.2e}")
            logger.info(f"  Metrics (7d MSE: {val_metrics['mse_7d']:.6f}, "
                       f"Corr: {val_metrics['correlation_7d']:.4f})")
            
            # Early stopping
            if val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_loss)
                logger.info(f"  ✓ New best model saved")
            else:
                self.patience_counter += 1
                logger.info(f"  → Patience: {self.patience_counter}/{early_stopping_patience}")
            
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1} (best: {self.best_epoch+1})")
                break
            
            # History
            history['epochs'].append(epoch)
            history['train_loss'].append(train_metrics['total_loss'])
            history['val_loss'].append(val_loss)
            history['metrics'].append(val_metrics)
            
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_loss)
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best epoch: {self.best_epoch+1}, Best val loss: {self.best_val_loss:.6f}")
        
        return history
    
    def load_best_checkpoint(self):
        """Load best checkpoint"""
        best_ckpt = self.checkpoint_dir / f'model_epoch_{self.best_epoch:03d}.pt'
        if best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint from epoch {self.best_epoch+1}")
        else:
            logger.warning("Best checkpoint not found")


def create_optimizer(model: nn.Module,
                    learning_rate: float = 0.001,
                    weight_decay: float = 1e-5) -> optim.Optimizer:
    """
    Create Adam optimizer with task-specific learning rates
    
    Args:
        model: DualTowerRelevanceModel
        learning_rate: Base learning rate
        weight_decay: L2 regularization
        
    Returns:
        Optimizer instance
    """
    
    # Separate parameter groups for towers
    context_params = list(model.context_tower.parameters())
    stock_params = list(model.stock_tower.parameters())
    head_params = (
        list(model.relevance_head_7d.parameters()) +
        list(model.relevance_head_30d.parameters())
    )
    
    param_groups = [
        {'params': context_params, 'lr': learning_rate},
        {'params': stock_params, 'lr': learning_rate * 0.5},  # Lower for stock tower
        {'params': head_params, 'lr': learning_rate},
    ]
    
    optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    return optimizer


def create_scheduler(optimizer: optim.Optimizer,
                    total_epochs: int = 100,
                    warmup_epochs: int = 5,
                    scheduler_type: str = 'cosine') -> optim.lr_scheduler.LRScheduler:
    """
    Create learning rate scheduler
    
    Args:
        optimizer: Optimizer
        total_epochs: Total epochs for training
        warmup_epochs: Warmup epochs
        scheduler_type: 'cosine' or 'plateau'
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=total_epochs - warmup_epochs,
            T_mult=1,
            eta_min=1e-6
        )
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


if __name__ == '__main__':
    # Test trainer setup
    import sys
    sys.path.insert(0, '/Users/davetang/Documents/GitHub/StockTrendEsimator')
    
    from data_pipeline.models.dual_tower_model import DualTowerRelevanceModel
    from data_pipeline.models.dual_tower_loss import DualTowerLoss
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = DualTowerRelevanceModel().to(device)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_epochs=100)
    
    # Create loss function
    loss_fn = DualTowerLoss()
    
    # Create trainer
    trainer = DualTowerTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir='./checkpoints'
    )
    
    print("Trainer initialized successfully!")
    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
