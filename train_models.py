#!/usr/bin/env python3
"""
Train LSTM and Dual Tower Models with Generated Training Data

Trains both models using the training and validation data generated from the pipeline.
Saves trained models and evaluation metrics.
"""

import os
import sys
import json
import logging
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


# ==============================================================================
# DATASET CLASSES
# ==============================================================================

class StockDataset(Dataset):
    """Dataset for stock trend prediction"""
    
    def __init__(self, df, feature_cols, target_col='stock_trend_direction'):
        """
        Args:
            df: DataFrame with all features
            feature_cols: List of feature column names
            target_col: Target column name
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Extract features and target
        self.features = torch.FloatTensor(df[feature_cols].values)
        # Convert targets from [-1, 0, 1] to [0, 1, 2]
        targets = df[target_col].values + 1
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMDataset(Dataset):
    """Dataset for LSTM model with sequence data"""
    
    def __init__(self, df, feature_cols, target_col='stock_trend_direction', sequence_length=5):
        """
        Args:
            df: DataFrame with all features
            feature_cols: List of feature column names
            target_col: Target column name
            sequence_length: Length of sequence for LSTM
        """
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Extract features and target
        features = df[feature_cols].values
        targets = df[target_col].values + 1  # Convert from [-1, 0, 1] to [0, 1, 2]
        
        self.sequences = []
        self.targets_list = []
        
        # Create sequences
        for i in range(len(features) - sequence_length):
            seq = torch.FloatTensor(features[i:i+sequence_length])
            target = torch.LongTensor([targets[i+sequence_length]])
            self.sequences.append(seq)
            self.targets_list.append(target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets_list[idx].squeeze()


# ==============================================================================
# SIMPLE LSTM MODEL
# ==============================================================================

class SimpleLSTM(nn.Module):
    """Simple LSTM model for trend prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output
        last_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        x = self.dropout(last_out)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ==============================================================================
# ATTENTION MODEL
# ==============================================================================

class AttentionLayer(nn.Module):
    """Multi-head attention layer for sequence features"""
    
    def __init__(self, hidden_size, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        # Linear transformations
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention, V)
        
        # Reshape back
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        # Final linear transformation
        output = self.fc_out(context)
        
        return output, attention


class AttentionModel(nn.Module):
    """Attention-based model for trend prediction"""
    
    def __init__(self, input_size, hidden_size=64, num_heads=4, num_classes=3, dropout=0.3):
        super(AttentionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Attention layers
        self.attention1 = AttentionLayer(hidden_size, num_heads=num_heads)
        self.attention2 = AttentionLayer(hidden_size, num_heads=num_heads)
        
        # Normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Classification head
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size) or (batch_size, input_size)
        if x.dim() == 2:
            # Add sequence dimension if not present
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = self.dropout(x)
        
        # First attention block
        attn_out1, attn_weights1 = self.attention1(x, x, x)
        x = self.norm1(x + attn_out1)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Second attention block
        attn_out2, attn_weights2 = self.attention2(x, x, x)
        x = x + attn_out2
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, hidden_size)
        
        # Classification head
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x, attn_weights1


# ==============================================================================
# DUAL TOWER MODEL
# ==============================================================================

class DualTowerModel(nn.Module):
    """Dual Tower model for relevance ranking"""
    
    def __init__(self, input_size, tower_hidden_size=32, hidden_size=16, num_classes=3, dropout=0.3):
        super(DualTowerModel, self).__init__()
        
        # Tower 1: Stock Features (first 8 features)
        self.tower1 = nn.Sequential(
            nn.Linear(8, tower_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_hidden_size, tower_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Tower 2: Context Features (remaining features)
        context_size = input_size - 8
        self.tower2 = nn.Sequential(
            nn.Linear(context_size, tower_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_hidden_size, tower_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Interaction layer
        combined_size = (tower_hidden_size // 2) * 2
        self.interaction = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # Split features
        stock_features = x[:, :8]
        context_features = x[:, 8:]
        
        # Process through towers
        tower1_out = self.tower1(stock_features)
        tower2_out = self.tower2(context_features)
        
        # Concatenate and process through interaction layer
        combined = torch.cat([tower1_out, tower2_out], dim=1)
        output = self.interaction(combined)
        
        return output


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def load_data(train_path, val_path):
    """Load training and validation data"""
    logger.info("Loading data...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Validation data shape: {val_df.shape}")
    
    return train_df, val_df


def get_feature_columns(df):
    """Extract feature columns from dataframe"""
    exclude_cols = {'date', 'timestamp', 'ticker', 'stock_trend_direction', 'stock_weekly_return', 'stock_weekly_movement'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def train_lstm_model(train_df, val_df, device, epochs=20, batch_size=8):
    """Train LSTM model"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING LSTM MODEL")
    logger.info("=" * 80)
    
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    # Create datasets
    sequence_length = 3  # Use 3 samples as sequence
    train_dataset = LSTMDataset(train_df, feature_cols, sequence_length=sequence_length)
    val_dataset = LSTMDataset(val_df, feature_cols, sequence_length=sequence_length)
    
    logger.info(f"Train sequences: {len(train_dataset)}")
    logger.info(f"Val sequences: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = len(feature_cols)
    model = SimpleLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3
    ).to(device)
    
    logger.info(f"\nModel: SimpleLSTM")
    logger.info(f"  Input size: {input_size}")
    logger.info(f"  Hidden size: 64")
    logger.info(f"  Num layers: 2")
    logger.info(f"  Output classes: 3")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_val_acc = 0
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        train_history.append({'loss': train_loss, 'acc': train_acc})
        val_history.append({'loss': val_loss, 'acc': val_acc})
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/lstm_best.pth')
    
    logger.info(f"\n✓ LSTM training complete!")
    logger.info(f"  Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"  Model saved to: models/lstm_best.pth")
    
    return model, best_val_acc, train_history, val_history


def train_dual_tower_model(train_df, val_df, device, epochs=20, batch_size=8):
    """Train Dual Tower model"""
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING DUAL TOWER MODEL")
    logger.info("=" * 80)
    
    feature_cols = get_feature_columns(train_df)
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    # Create datasets
    train_dataset = StockDataset(train_df, feature_cols)
    val_dataset = StockDataset(val_df, feature_cols)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_size = len(feature_cols)
    model = DualTowerModel(
        input_size=input_size,
        tower_hidden_size=32,
        hidden_size=16,
        num_classes=3,
        dropout=0.3
    ).to(device)
    
    logger.info(f"\nModel: DualTowerModel")
    logger.info(f"  Input size: {input_size}")
    logger.info(f"  Tower hidden size: 32")
    logger.info(f"  Interaction hidden size: 16")
    logger.info(f"  Output classes: 3")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # Training loop
    best_val_acc = 0
    train_history = []
    val_history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step()
        
        train_history.append({'loss': train_loss, 'acc': train_acc})
        val_history.append({'loss': val_loss, 'acc': val_acc})
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                       f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/dual_tower_best.pth')
    
    logger.info(f"\n✓ Dual Tower training complete!")
    logger.info(f"  Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"  Model saved to: models/dual_tower_best.pth")
    
    return model, best_val_acc, train_history, val_history


def save_training_summary(lstm_acc, dual_tower_acc, train_df, val_df):
    """Save training summary"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'training_data_path': 'data_output/training_data_20241101-20241130.csv',
        'validation_data_path': 'data_output/validation_data_20241101-20241130.csv',
        'training_samples': len(train_df),
        'validation_samples': len(val_df),
        'models': {
            'lstm': {
                'type': 'SimpleLSTM',
                'best_validation_accuracy': float(lstm_acc),
                'checkpoint': 'models/lstm_best.pth'
            },
            'dual_tower': {
                'type': 'DualTowerModel',
                'best_validation_accuracy': float(dual_tower_acc),
                'checkpoint': 'models/dual_tower_best.pth'
            }
        }
    }
    
    with open('models/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Training summary saved to: models/training_summary.json")


def main():
    """Main training script"""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " STOCK TREND ESTIMATOR - MODEL TRAINING ".center(78) + "║")
    logger.info("╚" + "=" * 78 + "╝")
    logger.info("\n")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Load data
    train_path = 'data_output/training_data_20241101-20241130.csv'
    val_path = 'data_output/validation_data_20241101-20241130.csv'
    
    if not Path(train_path).exists():
        logger.error(f"Training data not found at {train_path}")
        return
    
    if not Path(val_path).exists():
        logger.error(f"Validation data not found at {val_path}")
        return
    
    train_df, val_df = load_data(train_path, val_path)
    
    # Train models
    lstm_model, lstm_acc, lstm_train_hist, lstm_val_hist = train_lstm_model(
        train_df, val_df, device, epochs=20, batch_size=8
    )
    
    dual_tower_model, dt_acc, dt_train_hist, dt_val_hist = train_dual_tower_model(
        train_df, val_df, device, epochs=20, batch_size=8
    )
    
    # Save training summary
    save_training_summary(lstm_acc, dt_acc, train_df, val_df)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\n✅ LSTM Model - Best Validation Accuracy: {lstm_acc:.2f}%")
    logger.info(f"✅ Dual Tower Model - Best Validation Accuracy: {dt_acc:.2f}%")
    logger.info(f"\n📁 Models saved in: models/")
    logger.info(f"📊 Training summary: models/training_summary.json")
    logger.info("\n")


if __name__ == '__main__':
    main()
