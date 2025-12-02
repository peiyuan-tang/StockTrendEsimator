#!/usr/bin/env python3
"""
LSTM Model: Complete Examples & Usage Guide

Demonstrates:
1. Creating and training LSTM model for stock trend prediction
2. Sequence data preparation and batching
3. Multi-horizon forecasting (7-day and 30-day)
4. Attention weight visualization and interpretation
5. Evaluation and performance analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# EXAMPLE 1: Basic LSTM Model Training
# ==============================================================================

def example_1_basic_lstm_training():
    """
    Example 1: Create, train, and evaluate LSTM model with synthetic data
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic LSTM Model Training")
    logger.info("=" * 80)
    
    from data_pipeline.models import (
        create_lstm_model,
        create_lstm_data_loaders,
        LSTMLoss,
        LSTMTrainer,
        create_lstm_optimizer,
        create_lstm_scheduler,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Step 1: Generate synthetic training data
    logger.info("Step 1: Generating synthetic training data...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create synthetic time series (500 weeks, 62 features)
    dates = pd.date_range('2015-01-01', periods=500, freq='W')
    n_features = 62
    
    data = pd.DataFrame({
        f'feature_{i}': np.random.randn(500).cumsum() * 0.01 + np.random.randn(500) * 0.5
        for i in range(n_features)
    }, index=dates)
    
    # Add synthetic trend labels (7-day and 30-day)
    data['trend_7day'] = np.random.randn(500) * 0.5
    data['trend_30day'] = np.random.randn(500) * 0.3
    
    logger.info(f"  Generated {len(data)} time steps with {n_features} features")
    logger.info(f"  Date range: {dates[0].date()} to {dates[-1].date()}\n")
    
    # Step 2: Create data loaders
    logger.info("Step 2: Creating LSTM data loaders...")
    train_loader, val_loader, test_loader = create_lstm_data_loaders(
        data,
        sequence_length=12,  # 12 weeks of history
        batch_size=32,
        normalize=True,
        time_aware_split=True,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
    )
    
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}\n")
    
    # Step 3: Create LSTM model
    logger.info("Step 3: Creating LSTM model...")
    model = create_lstm_model(
        input_dim=n_features,
        hidden_dim=128,
        num_lstm_layers=2,
        num_attention_heads=4,
        dropout_rate=0.2,
        bidirectional=True,
        device=device
    )
    logger.info(f"  Model created successfully\n")
    
    # Step 4: Create optimizer and scheduler
    logger.info("Step 4: Setting up optimizer and scheduler...")
    optimizer = create_lstm_optimizer(model, learning_rate=0.001, weight_decay=1e-5)
    scheduler = create_lstm_scheduler(optimizer, total_epochs=50, warmup_epochs=5)
    logger.info(f"  Optimizer: Adam (lr=0.001)")
    logger.info(f"  Scheduler: Cosine Annealing with warmup\n")
    
    # Step 5: Create loss function
    logger.info("Step 5: Creating loss function...")
    loss_fn = LSTMLoss(
        regression_weight_7d=1.0,
        classification_weight_7d=0.5,
        regression_weight_30d=1.0,
        classification_weight_30d=0.5,
    )
    logger.info(f"  Multi-task loss configured\n")
    
    # Step 6: Create trainer
    logger.info("Step 6: Creating trainer...")
    trainer = LSTMTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir='./checkpoints/lstm',
        max_grad_norm=1.0,
    )
    logger.info(f"  Trainer initialized\n")
    
    # Step 7: Train model
    logger.info("Step 7: Training model...")
    logger.info("-" * 80)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        early_stopping_patience=10,
        min_delta=1e-4,
    )
    
    logger.info("-" * 80)
    logger.info("Training complete!\n")
    
    # Step 8: Evaluate on test set
    logger.info("Step 8: Evaluating on test set...")
    test_metrics = trainer.validate(test_loader)
    
    logger.info(f"  Test Loss: {test_metrics['loss']:.6f}")
    logger.info(f"  7-day MAE: {test_metrics['mae_7day']:.6f}")
    logger.info(f"  7-day Direction Accuracy: {test_metrics['direction_acc_7day']:.4f}")
    logger.info(f"  30-day MAE: {test_metrics['mae_30day']:.6f}")
    logger.info(f"  30-day Direction Accuracy: {test_metrics['direction_acc_30day']:.4f}\n")
    
    logger.info("Example 1 complete!\n\n")


# ==============================================================================
# EXAMPLE 2: Attention Weight Analysis
# ==============================================================================

def example_2_attention_analysis():
    """
    Example 2: Analyze attention weights to understand which time steps matter
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 2: Attention Weight Analysis")
    logger.info("=" * 80)
    
    from data_pipeline.models import create_lstm_model, create_lstm_data_loaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Create synthetic data
    np.random.seed(42)
    torch.manual_seed(42)
    
    dates = pd.date_range('2015-01-01', periods=300, freq='W')
    data = pd.DataFrame({
        f'feature_{i}': np.random.randn(300).cumsum() * 0.01 + np.random.randn(300) * 0.5
        for i in range(62)
    }, index=dates)
    data['trend_7day'] = np.random.randn(300) * 0.5
    data['trend_30day'] = np.random.randn(300) * 0.3
    
    # Create data loaders
    _, val_loader, _ = create_lstm_data_loaders(
        data,
        sequence_length=12,
        batch_size=32,
    )
    
    # Create model
    model = create_lstm_model(device=device)
    model.eval()
    
    # Collect attention weights
    all_attention_weights = []
    
    logger.info("Step 1: Extracting attention weights from validation set...")
    with torch.no_grad():
        for sequences, _ in val_loader:
            sequences = sequences.to(device)
            
            # Get model outputs
            outputs = model(sequences)
            attention_weights = outputs['attention_weights']  # (batch_size, seq_len)
            
            all_attention_weights.append(attention_weights.cpu().numpy())
    
    all_attention_weights = np.concatenate(all_attention_weights)  # (N, 12)
    logger.info(f"  Collected attention weights from {len(all_attention_weights)} samples\n")
    
    # Analyze attention patterns
    logger.info("Step 2: Analyzing attention patterns...")
    mean_attention = all_attention_weights.mean(axis=0)
    std_attention = all_attention_weights.std(axis=0)
    
    for t in range(12):
        logger.info(f"  Week {t}: Mean attention = {mean_attention[t]:.4f} ± {std_attention[t]:.4f}")
    
    logger.info("\nStep 3: Key findings:")
    most_important = np.argmax(mean_attention)
    logger.info(f"  Most important time step: Week {most_important} (attention: {mean_attention[most_important]:.4f})")
    
    least_important = np.argmin(mean_attention)
    logger.info(f"  Least important time step: Week {least_important} (attention: {mean_attention[least_important]:.4f})")
    
    # Plot attention weights
    logger.info("\nStep 4: Visualizing attention weights...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Average attention
    axes[0].bar(range(12), mean_attention, color='steelblue', alpha=0.7, label='Mean')
    axes[0].fill_between(range(12), mean_attention - std_attention, mean_attention + std_attention, 
                         alpha=0.3, color='steelblue', label='±1 Std')
    axes[0].set_xlabel('Time Step (Weeks)')
    axes[0].set_ylabel('Attention Weight')
    axes[0].set_title('Average Attention Weights Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Heatmap of attention for first 50 samples
    im = axes[1].imshow(all_attention_weights[:50, :], aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Time Step (Weeks)')
    axes[1].set_ylabel('Sample')
    axes[1].set_title('Attention Weight Heatmap (First 50 Samples)')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('attention_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("  Saved visualization to 'attention_analysis.png'\n")
    
    logger.info("Example 2 complete!\n\n")


# ==============================================================================
# EXAMPLE 3: Multi-Horizon Prediction Comparison
# ==============================================================================

def example_3_multi_horizon_comparison():
    """
    Example 3: Compare 7-day vs 30-day predictions and their accuracy
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 3: Multi-Horizon Prediction Comparison")
    logger.info("=" * 80)
    
    from data_pipeline.models import create_lstm_model, create_lstm_data_loaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Create data
    np.random.seed(42)
    torch.manual_seed(42)
    
    dates = pd.date_range('2015-01-01', periods=300, freq='W')
    data = pd.DataFrame({
        f'feature_{i}': np.random.randn(300).cumsum() * 0.01 + np.random.randn(300) * 0.5
        for i in range(62)
    }, index=dates)
    data['trend_7day'] = np.random.randn(300) * 0.5
    data['trend_30day'] = np.random.randn(300) * 0.3
    
    # Create loaders
    _, val_loader, _ = create_lstm_data_loaders(
        data,
        sequence_length=12,
        batch_size=32,
    )
    
    # Create model
    model = create_lstm_model(device=device)
    model.eval()
    
    # Collect predictions and targets
    pred_7day_list = []
    targ_7day_list = []
    pred_30day_list = []
    targ_30day_list = []
    
    logger.info("Step 1: Collecting predictions from validation set...")
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            
            outputs = model(sequences)
            
            pred_7day_list.append(outputs['7day_trend'].cpu().numpy())
            targ_7day_list.append(labels['trend_7day'].numpy())
            pred_30day_list.append(outputs['30day_trend'].cpu().numpy())
            targ_30day_list.append(labels['trend_30day'].numpy())
    
    pred_7day = np.concatenate(pred_7day_list)
    targ_7day = np.concatenate(targ_7day_list)
    pred_30day = np.concatenate(pred_30day_list)
    targ_30day = np.concatenate(targ_30day_list)
    
    logger.info(f"  Collected {len(pred_7day)} samples\n")
    
    # Compute metrics
    logger.info("Step 2: Computing metrics by horizon...")
    
    mae_7day = np.mean(np.abs(pred_7day - targ_7day))
    rmse_7day = np.sqrt(np.mean((pred_7day - targ_7day) ** 2))
    dir_acc_7day = np.mean((pred_7day > 0) == (targ_7day > 0))
    
    mae_30day = np.mean(np.abs(pred_30day - targ_30day))
    rmse_30day = np.sqrt(np.mean((pred_30day - targ_30day) ** 2))
    dir_acc_30day = np.mean((pred_30day > 0) == (targ_30day > 0))
    
    logger.info(f"\n  7-Day Predictions:")
    logger.info(f"    MAE: {mae_7day:.6f}")
    logger.info(f"    RMSE: {rmse_7day:.6f}")
    logger.info(f"    Direction Accuracy: {dir_acc_7day:.4f}")
    
    logger.info(f"\n  30-Day Predictions:")
    logger.info(f"    MAE: {mae_30day:.6f}")
    logger.info(f"    RMSE: {rmse_30day:.6f}")
    logger.info(f"    Direction Accuracy: {dir_acc_30day:.4f}\n")
    
    # Plot comparison
    logger.info("Step 3: Visualizing predictions...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 7-day scatter
    axes[0, 0].scatter(targ_7day, pred_7day, alpha=0.5, s=20)
    axes[0, 0].plot([-1, 1], [-1, 1], 'r--', lw=2)
    axes[0, 0].set_xlabel('Target')
    axes[0, 0].set_ylabel('Prediction')
    axes[0, 0].set_title('7-Day Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 30-day scatter
    axes[0, 1].scatter(targ_30day, pred_30day, alpha=0.5, s=20, color='orange')
    axes[0, 1].plot([-1, 1], [-1, 1], 'r--', lw=2)
    axes[0, 1].set_xlabel('Target')
    axes[0, 1].set_ylabel('Prediction')
    axes[0, 1].set_title('30-Day Predictions')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Residuals 7-day
    res_7day = pred_7day - targ_7day
    axes[1, 0].hist(res_7day, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'7-Day Errors (Mean: {res_7day.mean():.4f}, Std: {res_7day.std():.4f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals 30-day
    res_30day = pred_30day - targ_30day
    axes[1, 1].hist(res_30day, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'30-Day Errors (Mean: {res_30day.mean():.4f}, Std: {res_30day.std():.4f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_horizon_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("  Saved visualization to 'multi_horizon_comparison.png'\n")
    
    logger.info("Example 3 complete!\n\n")


# ==============================================================================
# EXAMPLE 4: Inference and Prediction
# ==============================================================================

def example_4_inference():
    """
    Example 4: Use trained model for inference on new data
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 4: Model Inference on New Data")
    logger.info("=" * 80)
    
    from data_pipeline.models import create_lstm_model, create_lstm_data_loaders
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}\n")
    
    # Create new test data
    logger.info("Step 1: Creating new test sequences...")
    np.random.seed(123)  # Different seed for "new" data
    
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    new_data = pd.DataFrame({
        f'feature_{i}': np.random.randn(100).cumsum() * 0.01 + np.random.randn(100) * 0.5
        for i in range(62)
    }, index=dates)
    new_data['trend_7day'] = np.random.randn(100) * 0.5
    new_data['trend_30day'] = np.random.randn(100) * 0.3
    
    logger.info(f"  Created {len(new_data)} samples\n")
    
    # Load data
    _, _, test_loader = create_lstm_data_loaders(
        new_data,
        sequence_length=12,
        batch_size=16,
    )
    
    # Create and use model
    model = create_lstm_model(device=device)
    model.eval()
    
    logger.info("Step 2: Making predictions...")
    
    predictions_list = []
    attention_weights_list = []
    
    with torch.no_grad():
        for batch_idx, (sequences, _) in enumerate(test_loader):
            sequences = sequences.to(device)
            
            outputs = model(sequences)
            
            predictions = {
                '7day_trend': outputs['7day_trend'].cpu().numpy(),
                '7day_direction': torch.argmax(outputs['7day_direction'], dim=1).cpu().numpy(),
                '30day_trend': outputs['30day_trend'].cpu().numpy(),
                '30day_direction': torch.argmax(outputs['30day_direction'], dim=1).cpu().numpy(),
            }
            
            predictions_list.append(predictions)
            attention_weights_list.append(outputs['attention_weights'].cpu().numpy())
    
    logger.info(f"  Generated predictions for {len(test_loader)} batches\n")
    
    # Display sample predictions
    logger.info("Step 3: Sample predictions:")
    logger.info(f"  Batch 0, Sample 0:")
    logger.info(f"    7-day trend: {predictions_list[0]['7day_trend'][0, 0]:.4f}")
    logger.info(f"    7-day direction: {'UP' if predictions_list[0]['7day_direction'][0] == 1 else 'DOWN'}")
    logger.info(f"    30-day trend: {predictions_list[0]['30day_trend'][0, 0]:.4f}")
    logger.info(f"    30-day direction: {'UP' if predictions_list[0]['30day_direction'][0] == 1 else 'DOWN'}\n")
    
    logger.info("Example 4 complete!\n\n")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    logger.info("\n" + "=" * 80)
    logger.info("LSTM MODEL EXAMPLES")
    logger.info("=" * 80 + "\n")
    
    # Run examples
    example_1_basic_lstm_training()
    example_2_attention_analysis()
    example_3_multi_horizon_comparison()
    example_4_inference()
    
    logger.info("=" * 80)
    logger.info("ALL EXAMPLES COMPLETE")
    logger.info("=" * 80)
    logger.info("\nFor more information, see LSTM_MODEL_GUIDE.md")
