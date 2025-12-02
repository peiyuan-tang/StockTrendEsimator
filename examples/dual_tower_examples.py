#!/usr/bin/env python3
"""
Dual-Tower Model: Complete Example & Usage Guide

Demonstrates:
1. Loading training data from UnifiedTrainingDataProcessor
2. Creating and training the dual-tower model
3. Making predictions and interpreting results
4. Analyzing feature importance and model behavior
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# EXAMPLE 1: Basic Model Training
# ==============================================================================

def example_1_basic_training():
    """
    Example 1: Load data and train dual-tower model
    """
    logger.info("=" * 80)
    logger.info("EXAMPLE 1: Basic Model Training")
    logger.info("=" * 80)
    
    # Import components
    from data_pipeline.core.training_data import UnifiedTrainingDataProcessor
    from data_pipeline.models import (
        create_dual_tower_model,
        create_dual_tower_data_loaders,
        DualTowerLoss,
        DualTowerTrainer,
        create_dual_tower_optimizer,
        create_dual_tower_scheduler,
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Step 1: Generate training data from unified processor
    logger.info("\nStep 1: Generating training data...")
    config = {'data_root': '/data'}
    processor = UnifiedTrainingDataProcessor(config)
    
    df = processor.generate_training_data(
        tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        include_weekly_movement=True
    )
    logger.info(f"  Generated {len(df)} samples with {len(df.columns)} features")
    
    # Step 2: Create data loaders
    logger.info("\nStep 2: Creating data loaders...")
    train_loader, val_loader, test_loader = create_dual_tower_data_loaders(
        df,
        batch_size=32,
        normalize=True,
        time_aware_split=True
    )
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    # Step 3: Create model
    logger.info("\nStep 3: Creating dual-tower model...")
    model = create_dual_tower_model(device=device)
    logger.info(f"  Model: DualTowerRelevanceModel")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Step 4: Create optimizer and scheduler
    logger.info("\nStep 4: Setting up optimizer and scheduler...")
    optimizer = create_dual_tower_optimizer(model, learning_rate=0.001)
    scheduler = create_dual_tower_scheduler(optimizer, total_epochs=50)
    
    # Step 5: Create loss function
    logger.info("\nStep 5: Creating loss function...")
    loss_fn = DualTowerLoss(
        regression_weight_7d=1.0,
        regression_weight_30d=1.0,
        classification_weight_7d=0.5,
        classification_weight_30d=0.5,
        regularization_weight=0.01
    )
    
    # Step 6: Create trainer
    logger.info("\nStep 6: Creating trainer...")
    trainer = DualTowerTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir='./checkpoints'
    )
    
    # Step 7: Train model
    logger.info("\nStep 7: Training model...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        early_stopping_patience=10
    )
    
    # Step 8: Load best checkpoint
    logger.info("\nStep 8: Loading best checkpoint...")
    trainer.load_best_checkpoint()
    
    logger.info("\n✓ Example 1 completed!")
    return model, trainer, test_loader


# ==============================================================================
# EXAMPLE 2: Making Predictions
# ==============================================================================

def example_2_predictions(model, test_loader, device='cuda'):
    """
    Example 2: Make predictions on test set and interpret results
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 2: Making Predictions")
    logger.info("=" * 80)
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predictions = {
        'score_7d': [],
        'score_30d': [],
        'direction_7d': [],
        'direction_30d': [],
        'confidence_7d': [],
        'confidence_30d': [],
    }
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            context, stock, labels = batch
            context = context.to(device)
            stock = stock.to(device)
            
            # Get predictions
            outputs = model(context, stock)
            
            # Store predictions
            predictions['score_7d'].append(outputs['score_7d'].cpu().numpy())
            predictions['score_30d'].append(outputs['score_30d'].cpu().numpy())
            
            # Direction
            dir_7d = (outputs['score_7d'] > 0).cpu().numpy()
            dir_30d = (outputs['score_30d'] > 0).cpu().numpy()
            predictions['direction_7d'].append(dir_7d)
            predictions['direction_30d'].append(dir_30d)
            
            # Confidence
            conf_7d = torch.max(
                outputs['pos_prob_7d'].unsqueeze(1),
                outputs['neg_prob_7d'].unsqueeze(1),
                dim=1
            )[0].cpu().numpy()
            conf_30d = torch.max(
                outputs['pos_prob_30d'].unsqueeze(1),
                outputs['neg_prob_30d'].unsqueeze(1),
                dim=1
            )[0].cpu().numpy()
            predictions['confidence_7d'].append(conf_7d)
            predictions['confidence_30d'].append(conf_30d)
    
    # Concatenate all predictions
    for key in predictions:
        predictions[key] = np.concatenate(predictions[key])
    
    # Display statistics
    logger.info("\n7-Day Predictions Statistics:")
    logger.info(f"  Score mean: {predictions['score_7d'].mean():.4f}")
    logger.info(f"  Score std: {predictions['score_7d'].std():.4f}")
    logger.info(f"  Positive relevance: {(predictions['direction_7d']).mean() * 100:.1f}%")
    logger.info(f"  Avg confidence: {predictions['confidence_7d'].mean():.4f}")
    
    logger.info("\n30-Day Predictions Statistics:")
    logger.info(f"  Score mean: {predictions['score_30d'].mean():.4f}")
    logger.info(f"  Score std: {predictions['score_30d'].std():.4f}")
    logger.info(f"  Positive relevance: {(predictions['direction_30d']).mean() * 100:.1f}%")
    logger.info(f"  Avg confidence: {predictions['confidence_30d'].mean():.4f}")
    
    # Display sample predictions
    logger.info("\nSample Predictions (first 10 samples):")
    logger.info("-" * 80)
    logger.info(f"{'7-Day Score':>12} | {'7-Day Dir':>11} | {'7-Day Conf':>11} | "
               f"{'30-Day Score':>13} | {'30-Day Dir':>12} | {'30-Day Conf':>12}")
    logger.info("-" * 80)
    
    for i in range(min(10, len(predictions['score_7d']))):
        logger.info(
            f"{predictions['score_7d'][i]:12.4f} | "
            f"{'Positive' if predictions['direction_7d'][i] else 'Negative':>11} | "
            f"{predictions['confidence_7d'][i]:11.4f} | "
            f"{predictions['score_30d'][i]:13.4f} | "
            f"{'Positive' if predictions['direction_30d'][i] else 'Negative':>12} | "
            f"{predictions['confidence_30d'][i]:12.4f}"
        )
    
    logger.info("\n✓ Example 2 completed!")
    return predictions


# ==============================================================================
# EXAMPLE 3: Interpreting Predictions
# ==============================================================================

def example_3_interpretation(predictions):
    """
    Example 3: Interpret what predictions mean
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 3: Interpreting Predictions")
    logger.info("=" * 80)
    
    logger.info("\nRelevance Score Interpretation Guide:")
    logger.info("-" * 80)
    
    score_7d = predictions['score_7d']
    
    # Categorize predictions
    strong_positive = score_7d >= 0.7
    weak_positive = (score_7d >= 0.1) & (score_7d < 0.7)
    neutral = (score_7d > -0.1) & (score_7d < 0.1)
    weak_negative = (score_7d > -0.7) & (score_7d <= -0.1)
    strong_negative = score_7d <= -0.7
    
    logger.info(f"\nStrong Positive (≥0.7):")
    logger.info(f"  Count: {strong_positive.sum()}")
    logger.info(f"  Percent: {strong_positive.mean() * 100:.1f}%")
    logger.info(f"  Meaning: Context strongly supports stock movement")
    
    logger.info(f"\nWeak Positive (0.1-0.7):")
    logger.info(f"  Count: {weak_positive.sum()}")
    logger.info(f"  Percent: {weak_positive.mean() * 100:.1f}%")
    logger.info(f"  Meaning: Context somewhat supports stock movement")
    
    logger.info(f"\nNeutral (-0.1-0.1):")
    logger.info(f"  Count: {neutral.sum()}")
    logger.info(f"  Percent: {neutral.mean() * 100:.1f}%")
    logger.info(f"  Meaning: Context and stock movements are uncorrelated")
    
    logger.info(f"\nWeak Negative (-0.7 to -0.1):")
    logger.info(f"  Count: {weak_negative.sum()}")
    logger.info(f"  Percent: {weak_negative.mean() * 100:.1f}%")
    logger.info(f"  Meaning: Context somewhat opposes stock movement (hedging)")
    
    logger.info(f"\nStrong Negative (≤-0.7):")
    logger.info(f"  Count: {strong_negative.sum()}")
    logger.info(f"  Percent: {strong_negative.mean() * 100:.1f}%")
    logger.info(f"  Meaning: Context strongly opposes stock movement")
    
    # Time horizon comparison
    logger.info("\n\nTime Horizon Comparison:")
    logger.info("-" * 80)
    
    score_7d = predictions['score_7d']
    score_30d = predictions['score_30d']
    
    # Find samples where 7-day and 30-day differ
    horizon_divergence = np.abs(score_7d - score_30d) > 0.5
    
    logger.info(f"\nSamples with significant horizon divergence (|7d-30d| > 0.5):")
    logger.info(f"  Count: {horizon_divergence.sum()}")
    logger.info(f"  Percent: {horizon_divergence.mean() * 100:.1f}%")
    logger.info(f"  Interpretation: Context has different short vs long-term effects")
    
    logger.info("\n✓ Example 3 completed!")


# ==============================================================================
# EXAMPLE 4: Analyzing Feature Importance
# ==============================================================================

def example_4_feature_importance(model, test_loader, device='cuda'):
    """
    Example 4: Analyze which features are most important
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 4: Feature Importance Analysis")
    logger.info("=" * 80)
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("\nAnalyzing feature importance via ablation...")
    logger.info("-" * 80)
    
    # Get one batch to analyze
    context, stock, labels = next(iter(test_loader))
    context = context.to(device)
    stock = stock.to(device)
    
    # Baseline prediction
    with torch.no_grad():
        baseline_output = model(context, stock)
        baseline_score = baseline_output['score_7d'].mean().item()
    
    logger.info(f"Baseline 7-day score: {baseline_score:.4f}")
    
    # Ablate context features
    logger.info("\nContext Feature Groups Importance (7-day):")
    context_ablations = {
        'news': 0,          # First 8 features
        'policy': 8,        # Next 5 features
        'macro': 13,        # Last 12 features
    }
    
    for name, start_idx in context_ablations.items():
        context_ablated = context.clone()
        
        # Determine end index
        if name == 'news':
            end_idx = 8
        elif name == 'policy':
            end_idx = 13
        else:  # macro
            end_idx = 25
        
        # Zero out features
        context_ablated[:, start_idx:end_idx] = 0
        
        with torch.no_grad():
            ablated_output = model(context_ablated, stock)
            ablated_score = ablated_output['score_7d'].mean().item()
        
        importance = abs(baseline_score - ablated_score)
        logger.info(f"  {name:10s}: {importance:.4f} "
                   f"(impact on average score)")
    
    logger.info("\n✓ Example 4 completed!")


# ==============================================================================
# EXAMPLE 5: Model Evaluation Metrics
# ==============================================================================

def example_5_evaluation_metrics(model, test_loader, device='cuda'):
    """
    Example 5: Compute comprehensive evaluation metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE 5: Model Evaluation Metrics")
    logger.info("=" * 80)
    
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predictions_7d = []
    predictions_30d = []
    labels_7d = []
    labels_30d = []
    
    with torch.no_grad():
        for batch in test_loader:
            context, stock, labels = batch
            context = context.to(device)
            stock = stock.to(device)
            
            outputs = model(context, stock)
            
            predictions_7d.append(outputs['score_7d'].cpu())
            predictions_30d.append(outputs['score_30d'].cpu())
            
            labels_7d.append(labels['label_7d'].cpu())
            labels_30d.append(labels['label_30d'].cpu())
    
    pred_7d = torch.cat(predictions_7d).numpy().flatten()
    pred_30d = torch.cat(predictions_30d).numpy().flatten()
    label_7d = torch.cat(labels_7d).numpy().flatten()
    label_30d = torch.cat(labels_30d).numpy().flatten()
    
    # Compute metrics
    mse_7d = np.mean((pred_7d - label_7d) ** 2)
    mse_30d = np.mean((pred_30d - label_30d) ** 2)
    mae_7d = np.mean(np.abs(pred_7d - label_7d))
    mae_30d = np.mean(np.abs(pred_30d - label_30d))
    
    # Correlation
    corr_7d = np.corrcoef(pred_7d, label_7d)[0, 1]
    corr_30d = np.corrcoef(pred_30d, label_30d)[0, 1]
    
    # Direction accuracy
    pred_dir_7d = (pred_7d > 0).astype(int)
    true_dir_7d = (label_7d > 0).astype(int)
    acc_7d = np.mean(pred_dir_7d == true_dir_7d)
    
    pred_dir_30d = (pred_30d > 0).astype(int)
    true_dir_30d = (label_30d > 0).astype(int)
    acc_30d = np.mean(pred_dir_30d == true_dir_30d)
    
    logger.info("\n7-Day Predictions Metrics:")
    logger.info(f"  MSE:                 {mse_7d:.6f}")
    logger.info(f"  MAE:                 {mae_7d:.6f}")
    logger.info(f"  Correlation:         {corr_7d:.4f}")
    logger.info(f"  Direction Accuracy:  {acc_7d:.4f}")
    
    logger.info("\n30-Day Predictions Metrics:")
    logger.info(f"  MSE:                 {mse_30d:.6f}")
    logger.info(f"  MAE:                 {mae_30d:.6f}")
    logger.info(f"  Correlation:         {corr_30d:.4f}")
    logger.info(f"  Direction Accuracy:  {acc_30d:.4f}")
    
    logger.info("\n✓ Example 5 completed!")
    
    return {
        '7d': {'mse': mse_7d, 'mae': mae_7d, 'corr': corr_7d, 'acc': acc_7d},
        '30d': {'mse': mse_30d, 'mae': mae_30d, 'corr': corr_30d, 'acc': acc_30d},
    }


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """
    Run all examples
    """
    logger.info("\n" + "=" * 80)
    logger.info("DUAL-TOWER MODEL: COMPLETE EXAMPLES")
    logger.info("=" * 80)
    
    # Note: In production, data would come from actual collection pipeline
    # For this example, you would run:
    # model, trainer, test_loader = example_1_basic_training()
    # predictions = example_2_predictions(model, test_loader)
    # example_3_interpretation(predictions)
    # example_4_feature_importance(model, test_loader)
    # metrics = example_5_evaluation_metrics(model, test_loader)
    
    logger.info("""
    To run the complete training pipeline:
    
    1. Ensure your training data is available:
       config = {'data_root': '/data'}
       processor = UnifiedTrainingDataProcessor(config)
       df = processor.generate_training_data()
    
    2. Run training:
       model, trainer, test_loader = example_1_basic_training()
    
    3. Make predictions:
       predictions = example_2_predictions(model, test_loader)
    
    4. Analyze results:
       example_3_interpretation(predictions)
       example_4_feature_importance(model, test_loader)
       metrics = example_5_evaluation_metrics(model, test_loader)
    
    See DUAL_TOWER_MODEL_DESIGN.md for complete documentation.
    """)


if __name__ == '__main__':
    main()
