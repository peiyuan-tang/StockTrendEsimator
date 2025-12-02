#!/usr/bin/env python3
"""
Quick Start Guide - Using Trained Models

This script demonstrates how to load and use the trained LSTM and Dual Tower models.
"""

import torch
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from train_models import SimpleLSTM, DualTowerModel

print("\n" + "=" * 80)
print("QUICK START - USING TRAINED MODELS")
print("=" * 80)

# ==============================================================================
# LOAD TRAINED MODELS
# ==============================================================================

print("\n1️⃣  Loading Trained Models")
print("-" * 80)

device = 'cpu'

# Load LSTM model
print("Loading LSTM model...")
lstm_model = SimpleLSTM(
    input_size=21,
    hidden_size=64,
    num_layers=2,
    num_classes=3,
    dropout=0.3
).to(device)
lstm_model.load_state_dict(torch.load('models/lstm_best.pth'))
lstm_model.eval()
print("✓ LSTM model loaded")

# Load Dual Tower model
print("Loading Dual Tower model...")
dt_model = DualTowerModel(
    input_size=21,
    tower_hidden_size=32,
    hidden_size=16,
    num_classes=3,
    dropout=0.3
).to(device)
dt_model.load_state_dict(torch.load('models/dual_tower_best.pth'))
dt_model.eval()
print("✓ Dual Tower model loaded")

# ==============================================================================
# LOAD TRAINING DATA AS EXAMPLES
# ==============================================================================

print("\n2️⃣  Loading Example Data")
print("-" * 80)

train_df = pd.read_csv('data_output/training_data_20241101-20241130.csv')
val_df = pd.read_csv('data_output/validation_data_20241101-20241130.csv')

# Get feature columns
exclude_cols = {'date', 'timestamp', 'ticker', 'stock_trend_direction', 
                'stock_weekly_return', 'stock_weekly_movement'}
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

print(f"Loaded {len(train_df)} training samples")
print(f"Loaded {len(val_df)} validation samples")
print(f"Features: {len(feature_cols)} columns")
print(f"Feature columns: {feature_cols[:3]}... (showing first 3)")

# ==============================================================================
# EXAMPLE 1: LSTM PREDICTIONS
# ==============================================================================

print("\n3️⃣  LSTM Model - Making Predictions")
print("-" * 80)

# Prepare sequence data (3 consecutive samples)
lstm_sequence = torch.FloatTensor(
    train_df[feature_cols].iloc[0:3].values
).unsqueeze(0).to(device)  # Add batch dimension

print(f"Input sequence shape: {lstm_sequence.shape}")
print(f"  - Batch size: {lstm_sequence.shape[0]}")
print(f"  - Sequence length: {lstm_sequence.shape[1]}")
print(f"  - Features: {lstm_sequence.shape[2]}")

with torch.no_grad():
    lstm_output = lstm_model(lstm_sequence)
    lstm_probs = torch.softmax(lstm_output, dim=1)
    lstm_pred = torch.argmax(lstm_output, dim=1)

print(f"\nLSTM Output shape: {lstm_output.shape}")
print(f"Predicted class (0-based): {lstm_pred.item()}")
print(f"  Downtrend (0): {lstm_probs[0, 0].item():.4f}")
print(f"  Neutral   (1): {lstm_probs[0, 1].item():.4f}")
print(f"  Uptrend   (2): {lstm_probs[0, 2].item():.4f}")
print(f"Predicted trend: {['Downtrend', 'Neutral', 'Uptrend'][lstm_pred.item()]}")

# ==============================================================================
# EXAMPLE 2: DUAL TOWER PREDICTIONS
# ==============================================================================

print("\n4️⃣  Dual Tower Model - Making Predictions")
print("-" * 80)

# Prepare flat feature data
dt_features = torch.FloatTensor(
    train_df[feature_cols].iloc[0:5].values
).to(device)

print(f"Input features shape: {dt_features.shape}")
print(f"  - Batch size: {dt_features.shape[0]}")
print(f"  - Features: {dt_features.shape[1]}")

with torch.no_grad():
    dt_output = dt_model(dt_features)
    dt_probs = torch.softmax(dt_output, dim=1)
    dt_pred = torch.argmax(dt_output, dim=1)

print(f"\nDual Tower Output shape: {dt_output.shape}")
print("\nPredictions for first 5 samples:")
for i in range(5):
    trend = ['Downtrend', 'Neutral', 'Uptrend'][dt_pred[i].item()]
    confidence = dt_probs[i, dt_pred[i]].item()
    print(f"  Sample {i+1}: {trend:10s} (confidence: {confidence:.4f})")

# ==============================================================================
# EXAMPLE 3: BATCH PREDICTIONS
# ==============================================================================

print("\n5️⃣  Batch Predictions")
print("-" * 80)

# Make predictions on entire validation set
batch_size = 4
all_lstm_preds = []
all_dt_preds = []

print(f"Processing {len(val_df)} validation samples in batches...")

lstm_model.eval()
dt_model.eval()

with torch.no_grad():
    for i in range(0, len(val_df) - 2, batch_size):
        # LSTM predictions (need sequences)
        if i + 3 <= len(val_df):
            seq = torch.FloatTensor(
                val_df[feature_cols].iloc[i:i+3].values
            ).unsqueeze(0).to(device)
            lstm_out = lstm_model(seq)
            lstm_preds = torch.argmax(lstm_out, dim=1)
            all_lstm_preds.extend(lstm_preds.cpu().numpy())
        
        # Dual Tower predictions
        if i + batch_size <= len(val_df):
            features = torch.FloatTensor(
                val_df[feature_cols].iloc[i:i+batch_size].values
            ).to(device)
            dt_out = dt_model(features)
            dt_preds = torch.argmax(dt_out, dim=1)
            all_dt_preds.extend(dt_preds.cpu().numpy())

print(f"LSTM predictions: {len(all_lstm_preds)} samples")
print(f"  Downtrend: {sum(p == 0 for p in all_lstm_preds)}")
print(f"  Neutral:   {sum(p == 1 for p in all_lstm_preds)}")
print(f"  Uptrend:   {sum(p == 2 for p in all_lstm_preds)}")

print(f"\nDual Tower predictions: {len(all_dt_preds)} samples")
print(f"  Downtrend: {sum(p == 0 for p in all_dt_preds)}")
print(f"  Neutral:   {sum(p == 1 for p in all_dt_preds)}")
print(f"  Uptrend:   {sum(p == 2 for p in all_dt_preds)}")

# ==============================================================================
# EXAMPLE 4: FEATURE ANALYSIS
# ==============================================================================

print("\n6️⃣  Feature Analysis")
print("-" * 80)

print("\nStock features (first 8):")
for i, col in enumerate(feature_cols[:8]):
    print(f"  {i}: {col}")

print("\nNews features (next 3):")
for i, col in enumerate(feature_cols[8:11], 8):
    print(f"  {i}: {col}")

print("\nMacro features (next 5):")
for i, col in enumerate(feature_cols[11:16], 11):
    print(f"  {i}: {col}")

print("\nPolicy features (last 2):")
for i, col in enumerate(feature_cols[16:18], 16):
    print(f"  {i}: {col}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✅ Models loaded successfully
✅ LSTM predictions generated (sequence-based)
✅ Dual Tower predictions generated (feature-based)
✅ Batch processing demonstrated
✅ Feature architecture explained

📌 Key Points:
  • LSTM requires sequence input (batch_size, seq_len, features)
  • Dual Tower requires flat features (batch_size, features)
  • Output classes: 0=Downtrend, 1=Neutral, 2=Uptrend
  • Use torch.softmax() for confidence scores
  • Both models trained on 21 features from Nov 2024 data

🚀 Next Steps:
  1. Integrate models into prediction pipeline
  2. Create real-time inference API
  3. Monitor model performance on new data
  4. Retrain periodically with new data
  5. Fine-tune hyperparameters for better accuracy
""")
print("=" * 80 + "\n")
