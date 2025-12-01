# Dual-Tower Model Design: Context-Stock Trend Relevance Prediction

## Executive Summary

This document describes a **Dual-Tower Deep Neural Network (DNN)** model that predicts the relevance between market context (policy + news) and stock movement trends across two time horizons: **7-day trends** and **30-day trends**.

The model learns to measure both **positive and negative relevance** (correlation/anti-correlation) between context signals and stock movements, enabling better understanding of how external factors drive or inhibit stock price changes.

---

## 1. Problem Statement

**Core Question:** *How much do policy and news events influence stock movements, and in which direction?*

### Key Challenges
- **Information asymmetry**: Different sources (stock, news, policy) operate at different scales and speeds
- **Non-linear relationships**: Context influence is not proportional; events may have delayed or threshold effects
- **Bidirectional influence**: Both positive and negative correlations matter equally
- **Multi-horizon effects**: Policy impacts 7-day trends differently than 30-day trends
- **Temporal alignment**: Weekly granularity requires synchronized feature representation

### Target Outcomes
- Predict whether context is **positively relevant** (high correlation) to stock movement
- Predict whether context is **negatively relevant** (anti-correlation/hedging)
- Predict **strength of relevance** (magnitude of correlation)
- Separately model **7-day and 30-day horizons** for different trading strategies
- Support **explainability**: Which context features drive predictions?

---

## 2. Model Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Dual-Tower Model Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

INPUT DATA (Weekly)
├─ Stock Features (62 features)           
│  └─ Financial metrics, technical indicators, volatility, etc.
├─ News Features (8 features)
│  └─ Sentiment scores, headline volume, source diversity
├─ Macro Features (12 features)
│  └─ Economic indicators, interest rates, GDP, inflation
└─ Policy Features (5 features)
   └─ Fed announcements, policy changes, regulatory events

         ↓                                    ↓
    
  ┌──────────────────┐            ┌──────────────────┐
  │ CONTEXT TOWER    │            │  STOCK TOWER     │
  │  (Policy+News)   │            │  (Movement)      │
  └──────────────────┘            └──────────────────┘
  
  Input: 25 dims                  Input: 62 dims
  (news: 8, policy: 5,            (financial + technical)
   macro: 12)
  
  ├─ Dense(25 → 128)              ├─ Dense(62 → 256)
  ├─ BatchNorm + ReLU             ├─ BatchNorm + ReLU
  ├─ Dropout(0.2)                 ├─ Dropout(0.3)
  │                               │
  ├─ Dense(128 → 64)              ├─ Dense(256 → 128)
  ├─ BatchNorm + ReLU             ├─ BatchNorm + ReLU
  ├─ Dropout(0.2)                 ├─ Dropout(0.3)
  │                               │
  ├─ Dense(64 → 32)               ├─ Dense(128 → 64)
  ├─ BatchNorm + ReLU             ├─ BatchNorm + ReLU
  └─ Dropout(0.2)                 └─ Dropout(0.3)
  
  Output: 32 dims                 Output: 64 dims
  (context embedding)             (stock embedding)
  
         ↓                              ↓
         └──────────────┬───────────────┘
                        │
          ┌─────────────┴─────────────┐
          │                           │
    ┌─────▼────────┐            ┌─────▼────────┐
    │ 7-DAY HEAD   │            │ 30-DAY HEAD  │
    │ (Relevance)  │            │ (Relevance)  │
    └──────────────┘            └──────────────┘
    
    ├─ Interaction layer:        ├─ Interaction layer:
    │  context ⊙ stock            │  context ⊙ stock
    │  (element-wise mul)          │  (element-wise mul)
    │                              │
    ├─ Dense(32 → 16)            ├─ Dense(32 → 16)
    ├─ BatchNorm + ReLU          ├─ BatchNorm + ReLU
    │                              │
    ├─ Dense(16 → 8)             ├─ Dense(16 → 8)
    ├─ ReLU                       ├─ ReLU
    │                              │
    └─ Dense(8 → 3)              └─ Dense(8 → 3)
       Output: 3 values             Output: 3 values
       (relevance_score,            (relevance_score,
        positive_prob,              positive_prob,
        negative_prob)              negative_prob)
        
               ↓                           ↓
          
    ┌──────────────────────────────────────────┐
    │    Multi-Task Loss Calculation           │
    │  Loss = L_7day + L_30day + L_aux         │
    └──────────────────────────────────────────┘
```

### 2.2 Tower Specifications

#### **Context Tower** (Policy + News + Macro)
```
Input Dimension: 25
├─ News features (8): sentiment, volume, diversity, etc.
├─ Policy features (5): announcement type, urgency, sector impact, etc.
└─ Macro features (12): inflation, rates, GDP, employment, etc.

Architecture:
- Dense(25 → 128) + BatchNorm + ReLU + Dropout(0.2)
- Dense(128 → 64) + BatchNorm + ReLU + Dropout(0.2)
- Dense(64 → 32) + BatchNorm + ReLU + Dropout(0.2)

Output: 32-dimensional context embedding
- Captures aggregated market context
- Learned representations of policy/news interactions
- Compression reduces noise and focuses on signal
```

#### **Stock Tower** (Financial Movement Data)
```
Input Dimension: 62
├─ OHLCV data (5): Open, High, Low, Close, Volume
├─ Technical indicators (20+): RSI, MACD, Bollinger Bands, ATR, etc.
├─ Returns (5): 1-day, 5-day, 20-day, 60-day, 252-day
├─ Volatility measures (10+): historical vol, realized vol, VIX proxies
└─ Volume analysis (15+): VWAP, volume trends, ratio indicators

Architecture:
- Dense(62 → 256) + BatchNorm + ReLU + Dropout(0.3)
- Dense(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
- Dense(128 → 64) + BatchNorm + ReLU + Dropout(0.3)

Output: 64-dimensional stock embedding
- Captures current market state
- Learned representations of price patterns
- Higher dimension reflects richer information
```

### 2.3 Relevance Heads (Multi-Horizon)

#### **7-Day Relevance Head**
```
Purpose: Predict short-term context impact (trading week horizon)

Input: Interacted embeddings [context_32 × stock_64]

Layers:
1. Interaction Layer:
   - interaction = context_embed ⊙ stock_embed (element-wise mult)
   - Result: 32-dim vector capturing co-activations

2. Relevance Projection:
   - Dense(32 → 16) + BatchNorm + ReLU
   - Dense(16 → 8) + ReLU
   - Dense(8 → 3) → Output logits

Output (3 values):
├─ relevance_score: [-1, 1] range via tanh
│  (How much does context drive stock movement?)
│  Positive: context supports price move
│  Negative: context opposes price move (hedging effect)
│
├─ positive_probability: [0, 1] via softmax
│  (Confidence that relevance is positive)
│
└─ negative_probability: [0, 1] via softmax
   (Confidence that relevance is negative)
```

#### **30-Day Relevance Head**
```
Purpose: Predict medium-term context impact (30-day trading horizon)

Same architecture as 7-day head but separate parameters
- Captures longer-term relationships
- Lower frequency oscillations
- Structural economic impacts vs tactical moves
```

### 2.4 Model Specifications

```python
class DualTowerRelevanceModel(nn.Module):
    """
    Context-Stock Relevance Prediction Model
    
    Predicts both positive and negative relevance between
    market context and stock movements across two time horizons.
    """
    
    # Hyperparameters
    context_input_dim = 25        # news(8) + policy(5) + macro(12)
    stock_input_dim = 62          # financial + technical indicators
    
    context_hidden_dims = [128, 64, 32]    # Progressive compression
    stock_hidden_dims = [256, 128, 64]     # Higher capacity
    
    context_embedding_dim = 32
    stock_embedding_dim = 64
    
    relevance_head_dims = [16, 8, 3]       # Project to [score, pos, neg]
    
    dropout_rates = {
        'context_tower': 0.2,
        'stock_tower': 0.3,
        'relevance_heads': 0.2
    }
    
    activation = ReLU
    normalization = BatchNorm1d
```

---

## 3. Training Data & Labels

### 3.1 Data Preparation

```
Raw Data (Weekly granularity from UnifiedTrainingDataProcessor):
├─ 7 tickers (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA)
├─ 12 weeks history (~240 total samples)
└─ 87 features (stock_*, news_*, macro_*, policy_*)

Train/Val/Test Split:
├─ Training: 60% (~144 samples)
├─ Validation: 20% (~48 samples)
└─ Test: 20% (~48 samples)

Time-aware splitting to preserve temporal causality
```

### 3.2 Label Generation

#### **7-Day Label**
```
label_7day = (price_t+7 - price_t) / price_t
├─ Positive: stock gains ground (context supported move)
└─ Negative: stock loses ground (context opposed move or counteracted)

Normalized to [-1, 1] range:
label_7day_norm = tanh(label_7day * 10)  # Scale for better learning
```

#### **30-Day Label**
```
label_30day = (price_t+30 - price_t) / price_t

Normalized to [-1, 1] range:
label_30day_norm = tanh(label_30day * 10)
```

#### **Label Categorization**
```
For classification outputs (positive/negative probability):

If label >= 0.1:  → Positive relevance class (1.0, 0.0)
If label <= -0.1: → Negative relevance class (0.0, 1.0)
Otherwise:        → Neutral class (0.5, 0.5)
```

### 3.3 Data Batching

```python
batch_size = 32
epochs = 100

Each batch contains:
├─ Context features (batch_size × 25)
├─ Stock features (batch_size × 62)
├─ 7-day labels (batch_size × 1)
├─ 30-day labels (batch_size × 1)
└─ Sample weights (for class balancing)
```

---

## 4. Loss Functions

### 4.1 Primary Loss: Relevance Score Regression

```
Purpose: Predict continuous [-1, 1] relevance score

L_regression_7d = MSE(pred_score_7d, label_7d)
L_regression_30d = MSE(pred_score_30d, label_30d)

Where:
- pred_score: model output through tanh activation [-1, 1]
- label: normalized stock percentage change [-1, 1]

This captures both magnitude and direction of relevance
```

### 4.2 Secondary Loss: Relevance Direction Classification

```
Purpose: Classify whether relevance is positive or negative

L_classification_7d = CrossEntropy(
    logits=[pos_logit, neg_logit],
    targets=[positive_label, negative_label]
)

L_classification_30d = CrossEntropy(
    logits=[pos_logit, neg_logit],
    targets=[positive_label, negative_label]
)

This ensures model learns clear decision boundaries
```

### 4.3 Auxiliary Loss: Embedding Regularization

```
Purpose: Prevent tower collapse and encourage diverse representations

L_aux = L_orthogonal + L_magnitude

L_orthogonal: Encourages context and stock embeddings to be diverse
    = -cos_similarity(context_embed, stock_embed)
    (Prevents perfect correlation where towers learn same thing)

L_magnitude: Prevents embeddings from growing unbounded
    = ||context_embed||_2 + ||stock_embed||_2
    (Encourages efficient feature use)
```

### 4.4 Total Loss Function

```
L_total = α₁ * L_regression_7d 
        + α₂ * L_regression_30d 
        + β₁ * L_classification_7d 
        + β₂ * L_classification_30d 
        + γ * L_aux

Where:
α₁ = 1.0     (7-day regression weight)
α₂ = 1.0     (30-day regression weight)
β₁ = 0.5     (7-day classification weight)
β₂ = 0.5     (30-day classification weight)
γ = 0.01     (Regularization weight)

Intuition:
- Equal weight for 7-day and 30-day (multi-horizon symmetry)
- Classification loss < regression loss (regression is primary)
- Small regularization loss (prevents degenerate solutions)
```

### 4.5 Loss Computation Details

```python
def compute_loss(model_output, labels, lambda_aux=0.01):
    """
    Compute multi-task loss for dual-tower model
    
    model_output:
    ├─ relevance_score_7d: (batch_size, 1) in [-1, 1]
    ├─ pos_prob_7d: (batch_size, 2) logits
    ├─ neg_prob_7d: (batch_size, 2) logits
    ├─ relevance_score_30d: (batch_size, 1) in [-1, 1]
    ├─ pos_prob_30d: (batch_size, 2) logits
    └─ neg_prob_30d: (batch_size, 2) logits
    
    labels:
    ├─ label_7d: (batch_size, 1) in [-1, 1]
    ├─ label_30d: (batch_size, 1) in [-1, 1]
    ├─ label_7d_class: (batch_size,) in {0: neg, 1: pos}
    └─ label_30d_class: (batch_size,) in {0: neg, 1: pos}
    """
    
    # Regression losses
    L_reg_7d = MSELoss()(output['score_7d'], labels['label_7d'])
    L_reg_30d = MSELoss()(output['score_30d'], labels['label_30d'])
    
    # Classification losses
    L_cls_7d = CrossEntropyLoss()(output['logits_7d'], labels['class_7d'])
    L_cls_30d = CrossEntropyLoss()(output['logits_30d'], labels['class_30d'])
    
    # Auxiliary losses
    context_embed = model.get_context_embedding()
    stock_embed = model.get_stock_embedding()
    
    cos_sim = torch.nn.functional.cosine_similarity(
        context_embed, stock_embed, dim=1
    ).mean()
    L_orthogonal = cos_sim  # Minimize similarity
    
    L_magnitude = (
        torch.norm(context_embed) + 
        torch.norm(stock_embed)
    ) / (2 * batch_size)
    
    L_aux = L_orthogonal + L_magnitude
    
    # Total loss
    L_total = (
        1.0 * L_reg_7d + 
        1.0 * L_reg_30d + 
        0.5 * L_cls_7d + 
        0.5 * L_cls_30d + 
        lambda_aux * L_aux
    )
    
    return L_total, {
        'reg_7d': L_reg_7d,
        'reg_30d': L_reg_30d,
        'cls_7d': L_cls_7d,
        'cls_30d': L_cls_30d,
        'aux': L_aux,
        'total': L_total
    }
```

---

## 5. Optimization Strategy

### 5.1 Optimizer Configuration

```
Optimizer: Adam (adaptive learning rates for tower-specific needs)

Context Tower:
├─ Learning rate: 0.001
├─ Betas: (0.9, 0.999)
└─ Weight decay: 1e-5

Stock Tower:
├─ Learning rate: 0.0005  (Lower due to higher capacity)
├─ Betas: (0.9, 0.999)
└─ Weight decay: 1e-5

Relevance Heads:
├─ Learning rate: 0.001
├─ Betas: (0.9, 0.999)
└─ Weight decay: 1e-4 (Higher regularization for heads)
```

### 5.2 Learning Rate Schedule

```
Strategy: Cosine annealing with warm-up

Warm-up phase (epochs 1-5):
└─ LR_t = initial_lr * (t / warmup_epochs)

Main phase (epochs 5-100):
└─ LR_t = 0.5 * initial_lr * (1 + cos(π * (t - warmup) / (total - warmup)))

Rationale:
- Warm-up prevents unstable early training
- Cosine annealing smoothly reduces learning rate
- Enables longer convergence period
```

### 5.3 Early Stopping & Checkpointing

```
Monitor: Validation loss on combined metric

Val_loss = 0.4 * val_loss_7d + 0.4 * val_loss_30d + 0.2 * val_loss_aux

Early stopping:
├─ Patience: 15 epochs (no improvement)
├─ Min delta: 1e-4 (minimum improvement threshold)
└─ Restore: Best model checkpoint

Checkpointing:
├─ Save: Every epoch where val_loss improves
├─ Keep: Top 3 checkpoints
└─ Path: models/dual_tower_best.pt
```

---

## 6. Training Procedure

### 6.1 Complete Training Loop

```python
def train_epoch(model, train_loader, optimizer, device):
    """
    Single training epoch
    """
    model.train()
    total_loss = 0
    loss_components = defaultdict(float)
    
    for batch_idx, (context, stock, labels) in enumerate(train_loader):
        # Forward pass
        context = context.to(device)
        stock = stock.to(device)
        
        outputs = model(context, stock)
        
        # Compute loss
        loss, loss_dict = compute_loss(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        for key, val in loss_dict.items():
            loss_components[key] += val.item()
    
    # Average
    avg_loss = total_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def train(model, train_loader, val_loader, epochs=100, device='cuda'):
    """
    Complete training loop with validation
    """
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validation
        val_loss, val_components = validate(
            model, val_loader, device
        )
        
        # Learning rate update
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Components: {val_components}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/dual_tower_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= 15:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model
```

### 6.2 Validation & Evaluation Metrics

```python
def validate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    
    predictions_7d = []
    predictions_30d = []
    labels_7d = []
    labels_30d = []
    
    with torch.no_grad():
        for context, stock, labels in val_loader:
            context = context.to(device)
            stock = stock.to(device)
            
            outputs = model(context, stock)
            loss, _ = compute_loss(outputs, labels)
            
            total_loss += loss.item()
            
            # Collect predictions for metrics
            predictions_7d.append(outputs['score_7d'].cpu())
            predictions_30d.append(outputs['score_30d'].cpu())
            labels_7d.append(labels['label_7d'].cpu())
            labels_30d.append(labels['label_30d'].cpu())
    
    # Metrics
    pred_7d = torch.cat(predictions_7d)
    pred_30d = torch.cat(predictions_30d)
    label_7d = torch.cat(labels_7d)
    label_30d = torch.cat(labels_30d)
    
    metrics = {
        'mse_7d': ((pred_7d - label_7d) ** 2).mean().item(),
        'mse_30d': ((pred_30d - label_30d) ** 2).mean().item(),
        'mae_7d': (torch.abs(pred_7d - label_7d)).mean().item(),
        'mae_30d': (torch.abs(pred_30d - label_30d)).mean().item(),
        'correlation_7d': compute_correlation(pred_7d, label_7d),
        'correlation_30d': compute_correlation(pred_30d, label_30d),
    }
    
    return total_loss / len(val_loader), metrics
```

---

## 7. Inference & Interpretation

### 7.1 Making Predictions

```python
def predict_relevance(model, context_features, stock_features):
    """
    Predict context-stock relevance
    
    Returns:
    ├─ relevance_7d: float in [-1, 1]
    │  Positive: context supports stock movement
    │  Negative: context opposes stock movement
    │
    ├─ confidence_7d: float in [0, 1]
    │  Confidence in positive vs negative direction
    │
    ├─ relevance_30d: float in [-1, 1]
    │
    └─ confidence_30d: float in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        outputs = model(context_features, stock_features)
    
    return {
        'relevance_7d': outputs['score_7d'].item(),
        'confidence_7d': outputs['pos_prob_7d'].item(),  # Max of [pos, neg]
        'relevance_30d': outputs['score_30d'].item(),
        'confidence_30d': outputs['pos_prob_30d'].item(),
        'direction_7d': 'positive' if outputs['score_7d'] > 0 else 'negative',
        'direction_30d': 'positive' if outputs['score_30d'] > 0 else 'negative',
    }
```

### 7.2 Interpretation Guide

```
Relevance Score [-1, 1]:

Score = 0.8 (Strong Positive):
├─ Context strongly supports stock movement
├─ Example: Good news pushes stock higher
└─ Action: Trend is driven by external factors

Score = -0.7 (Strong Negative):
├─ Context opposes stock movement
├─ Example: Good news but stock drops (hedging)
└─ Action: Movement is internally driven/contrarian

Score ≈ 0 (Neutral):
├─ Context and stock movements are uncorrelated
├─ Example: News irrelevant to this stock
└─ Action: Stock movement driven by other factors

Confidence [0, 1]:
├─ How certain is the model about direction?
├─ 0.5 = Uncertain (could flip direction)
└─ 0.95 = Very confident in direction
```

### 7.3 Feature Importance (Tower Analysis)

```python
def analyze_tower_importance(model, context_features):
    """
    Compute importance of different context components
    """
    # Ablation: remove each feature group
    
    # Full prediction
    full_output = model(context_features, stock_features)
    
    # Without news (set to zero)
    context_no_news = context_features.clone()
    context_no_news[:, :8] = 0
    output_no_news = model(context_no_news, stock_features)
    
    # Without policy
    context_no_policy = context_features.clone()
    context_no_policy[:, 8:13] = 0
    output_no_policy = model(context_no_policy, stock_features)
    
    # Without macro
    context_no_macro = context_features.clone()
    context_no_macro[:, 13:25] = 0
    output_no_macro = model(context_no_macro, stock_features)
    
    # Compute importance as change in output
    importance = {
        'news': (full_output - output_no_news).abs().mean(),
        'policy': (full_output - output_no_policy).abs().mean(),
        'macro': (full_output - output_no_macro).abs().mean(),
    }
    
    return importance
```

---

## 8. Implementation Roadmap

### Phase 1: Core Model (Week 1)
- [ ] Implement ContextTower class
- [ ] Implement StockTower class
- [ ] Implement DualTowerRelevanceModel
- [ ] Implement loss functions
- [ ] Unit tests for forward pass

### Phase 2: Data Pipeline (Week 2)
- [ ] Create DataLoader from UnifiedTrainingDataProcessor
- [ ] Implement label generation (7-day, 30-day)
- [ ] Implement train/val/test split
- [ ] Data normalization and augmentation

### Phase 3: Training Loop (Week 3)
- [ ] Implement training loop
- [ ] Implement validation
- [ ] Learning rate scheduling
- [ ] Early stopping and checkpointing

### Phase 4: Inference & Evaluation (Week 4)
- [ ] Implement prediction function
- [ ] Implement evaluation metrics
- [ ] Create interpretation utilities
- [ ] Demo notebook with examples

### Phase 5: Optimization & Deployment (Week 5)
- [ ] Hyperparameter tuning
- [ ] Model quantization for inference
- [ ] REST API for predictions
- [ ] Documentation and examples

---

## 9. Expected Outcomes

### 9.1 Model Performance Targets

```
Regression Metrics (MSE):
├─ 7-day: MSE < 0.1 (explains ~90% of variance)
└─ 30-day: MSE < 0.15 (explains ~85% of variance)

Classification Metrics:
├─ Accuracy: > 70% (distinguish positive vs negative)
└─ F1-score: > 0.65 (balanced precision/recall)

Correlation Metrics:
├─ Prediction-to-Label Correlation 7d: > 0.75
└─ Prediction-to-Label Correlation 30d: > 0.70
```

### 9.2 Business Insights

```
Enables understanding of:
✓ Which news events move Mag 7 stocks
✓ How policy announcements affect different stocks
✓ Which macro indicators drive market movements
✓ Positive vs negative relevance (hedging effects)
✓ Time-horizon specific impacts (7-day vs 30-day)
✓ Confidence in trend direction
```

### 9.3 Risk Management

```
Use cases:
✓ Identify when stock movement contradicts context (alert)
✓ Measure context relevance for strategy decisions
✓ Detect regime changes (relevance flips)
✓ Quantify hedging requirements
✓ Explain model predictions to stakeholders
```

---

## 10. Technical Considerations

### 10.1 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Class Imbalance** | Weighted loss, stratified sampling |
| **Tower Collapse** | Orthogonality loss, gradient clipping |
| **Overfitting** | Dropout, L2 regularization, early stopping |
| **Data Scarcity** | Data augmentation, transfer learning |
| **Temporal Causality** | Time-aware train/val split |
| **Feature Drift** | Online adaptation, periodic retraining |

### 10.2 Hyperparameter Sensitivity

```
High Sensitivity:
├─ Dropout rates (context: 0.2, stock: 0.3)
├─ Loss weights (regression vs classification)
└─ Learning rates (task-specific)

Medium Sensitivity:
├─ Hidden dimensions (context: [128,64,32], stock: [256,128,64])
├─ Batch size (32 recommended)
└─ L2 regularization strength

Low Sensitivity:
├─ Activation functions (ReLU is robust)
└─ Normalization methods (BatchNorm works well)
```

### 10.3 Monitoring & Debugging

```
Key metrics to track:
├─ Train vs validation loss divergence (overfitting)
├─ Loss component breakdown (which task struggles?)
├─ Gradient norms per tower (training stability)
├─ Tower embedding orthogonality (tower independence)
└─ Prediction calibration (confidence vs accuracy)

Debug checklist:
□ Check for NaN/Inf in loss
□ Verify gradient flow (use hooks)
□ Compare to baseline (linear model)
□ Analyze failure cases
□ Test on held-out data
```

---

## 11. Integration with Existing Pipeline

### 11.1 Data Flow

```
UnifiedTrainingDataProcessor (existing)
           ↓
   Generate unified dataset
   (stock_*, news_*, macro_*, policy_*)
           ↓
   DualTowerDataLoader (new)
           ↓
   Split into [context_features, stock_features, labels]
           ↓
   DualTowerRelevanceModel (new)
           ↓
   Predictions + Explanations
```

### 11.2 Configuration

```yaml
model:
  type: dual_tower_relevance
  context_input_dim: 25  # news(8) + policy(5) + macro(12)
  stock_input_dim: 62    # financial + technical
  
  context_tower:
    hidden_dims: [128, 64, 32]
    dropout: 0.2
    
  stock_tower:
    hidden_dims: [256, 128, 64]
    dropout: 0.3
    
  relevance_heads:
    horizons: [7, 30]  # days
    hidden_dims: [16, 8, 3]
    
training:
  optimizer: adam
  batch_size: 32
  epochs: 100
  early_stopping_patience: 15
  learning_rate: 0.001
  
  loss_weights:
    regression_7d: 1.0
    regression_30d: 1.0
    classification_7d: 0.5
    classification_30d: 0.5
    auxiliary: 0.01
```

---

## 12. Conclusion

The **Dual-Tower Relevance Model** provides a principled approach to understanding context-stock relationships through:

1. **Specialized Architectures**: Separate towers for context vs stock leverage domain knowledge
2. **Multi-Horizon Learning**: 7-day and 30-day heads capture different trading dynamics
3. **Bidirectional Relevance**: Supports both positive and negative correlations
4. **Interpretable Outputs**: Clear scores, probabilities, and direction indicators
5. **Scalable Design**: Can extend to more towers/horizons as needed

This model bridges the gap between traditional market analysis and modern deep learning, enabling data-driven understanding of stock-context interactions.

