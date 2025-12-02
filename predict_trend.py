#!/usr/bin/env python3
"""
Stock Trend Prediction - Inference Script

Accepts a stock ticker and predicts the next monthly trend using
both LSTM and Dual Tower models trained on November 2024 data.

Usage:
    python3 predict_trend.py AAPL
    python3 predict_trend.py MSFT --model lstm
    python3 predict_trend.py GOOGL --ensemble
"""

import argparse
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train_models import SimpleLSTM, DualTowerModel


# ==============================================================================
# DATA GENERATION FOR INFERENCE
# ==============================================================================

def generate_synthetic_features(ticker: str, num_samples: int = 5) -> pd.DataFrame:
    """
    Generate synthetic features for a given ticker
    
    In a production system, this would fetch real data from APIs
    (Alpha Vantage, Yahoo Finance, etc.)
    
    Args:
        ticker: Stock ticker symbol
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with 21 features
    """
    np.random.seed(hash(ticker) % 2**32)  # Seed based on ticker
    
    data = {
        'date': pd.date_range(end=datetime.now(), periods=num_samples, freq='D'),
        'ticker': [ticker] * num_samples,
        'stock_open': np.random.uniform(100, 200, num_samples),
        'stock_high': np.random.uniform(105, 210, num_samples),
        'stock_low': np.random.uniform(95, 190, num_samples),
        'stock_close': np.random.uniform(100, 200, num_samples),
        'stock_volume': np.random.uniform(1e6, 1e8, num_samples),
        'stock_sma_20': np.random.uniform(100, 200, num_samples),
        'stock_sma_50': np.random.uniform(100, 200, num_samples),
        'stock_rsi': np.random.uniform(30, 70, num_samples),
        'stock_macd': np.random.uniform(-5, 5, num_samples),
        'stock_bb_upper': np.random.uniform(105, 210, num_samples),
        'stock_bb_lower': np.random.uniform(95, 190, num_samples),
        'news_sentiment': np.random.uniform(-1, 1, num_samples),
        'news_count': np.random.randint(0, 20, num_samples),
        'news_relevance': np.random.uniform(0, 1, num_samples),
        'macro_gdp_growth': np.random.uniform(-2, 4, num_samples),
        'macro_inflation': np.random.uniform(1, 5, num_samples),
        'macro_unemployment': np.random.uniform(3, 6, num_samples),
        'macro_interest_rate': np.random.uniform(0.5, 4, num_samples),
        'macro_vix': np.random.uniform(10, 40, num_samples),
        'policy_event_count': np.random.randint(0, 5, num_samples),
        'policy_impact_score': np.random.uniform(-1, 1, num_samples),
    }
    
    return pd.DataFrame(data)


# ==============================================================================
# MODEL INFERENCE
# ==============================================================================

class TrendPredictor:
    """Predict stock trend direction using trained models"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize predictor with trained models
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.feature_cols = [
            'stock_open', 'stock_high', 'stock_low', 'stock_close',
            'stock_volume', 'stock_sma_20', 'stock_sma_50', 'stock_rsi',
            'stock_macd', 'stock_bb_upper', 'stock_bb_lower',
            'news_sentiment', 'news_count', 'news_relevance',
            'macro_gdp_growth', 'macro_inflation', 'macro_unemployment',
            'macro_interest_rate', 'macro_vix',
            'policy_event_count', 'policy_impact_score'
        ]
        
        self.class_names = ['Downtrend', 'Neutral', 'Uptrend']
        self.class_map = {0: 'Downtrend', 1: 'Neutral', 2: 'Uptrend'}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained LSTM and Dual Tower models"""
        logger.info("Loading trained models...")
        
        try:
            # Load LSTM
            self.lstm_model = SimpleLSTM(
                input_size=21,
                hidden_size=64,
                num_layers=2,
                num_classes=3,
                dropout=0.3
            ).to(self.device)
            
            if Path('models/lstm_best.pth').exists():
                self.lstm_model.load_state_dict(
                    torch.load('models/lstm_best.pth', map_location=self.device)
                )
                self.lstm_model.eval()
                logger.info("✓ LSTM model loaded")
            else:
                logger.warning("⚠ LSTM model checkpoint not found")
                self.lstm_model = None
        
        except Exception as e:
            logger.warning(f"⚠ Failed to load LSTM model: {e}")
            self.lstm_model = None
        
        try:
            # Load Dual Tower
            self.dt_model = DualTowerModel(
                input_size=21,
                tower_hidden_size=32,
                hidden_size=16,
                num_classes=3,
                dropout=0.3
            ).to(self.device)
            
            if Path('models/dual_tower_best.pth').exists():
                self.dt_model.load_state_dict(
                    torch.load('models/dual_tower_best.pth', map_location=self.device)
                )
                self.dt_model.eval()
                logger.info("✓ Dual Tower model loaded")
            else:
                logger.warning("⚠ Dual Tower model checkpoint not found")
                self.dt_model = None
        
        except Exception as e:
            logger.warning(f"⚠ Failed to load Dual Tower model: {e}")
            self.dt_model = None
        
        if self.lstm_model is None and self.dt_model is None:
            logger.error("❌ Failed to load any models")
            sys.exit(1)
    
    def predict_lstm(self, df: pd.DataFrame) -> Dict:
        """
        Predict using LSTM model (requires sequence)
        
        Args:
            df: DataFrame with features
            
        Returns:
            Prediction results
        """
        if self.lstm_model is None:
            return None
        
        try:
            # Use last 3 samples as sequence
            if len(df) < 3:
                logger.warning("⚠ Not enough samples for LSTM (need 3+)")
                return None
            
            seq = torch.FloatTensor(
                df[self.feature_cols].iloc[-3:].values
            ).unsqueeze(0).to(self.device)  # (1, 3, 21)
            
            with torch.no_grad():
                output = self.lstm_model(seq)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1)
            
            return {
                'model': 'LSTM',
                'prediction': self.class_map[pred.item()],
                'pred_class': pred.item(),
                'probabilities': {
                    'downtrend': probs[0, 0].item(),
                    'neutral': probs[0, 1].item(),
                    'uptrend': probs[0, 2].item()
                },
                'confidence': probs[0, pred.item()].item()
            }
        
        except Exception as e:
            logger.error(f"❌ LSTM prediction error: {e}")
            return None
    
    def predict_dual_tower(self, df: pd.DataFrame) -> Dict:
        """
        Predict using Dual Tower model (uses flat features)
        
        Args:
            df: DataFrame with features
            
        Returns:
            Prediction results
        """
        if self.dt_model is None:
            return None
        
        try:
            # Use most recent sample (or average of last few)
            features = torch.FloatTensor(
                df[self.feature_cols].iloc[-1:].values
            ).to(self.device)  # (1, 21)
            
            with torch.no_grad():
                output = self.dt_model(features)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1)
            
            return {
                'model': 'Dual Tower',
                'prediction': self.class_map[pred.item()],
                'pred_class': pred.item(),
                'probabilities': {
                    'downtrend': probs[0, 0].item(),
                    'neutral': probs[0, 1].item(),
                    'uptrend': probs[0, 2].item()
                },
                'confidence': probs[0, pred.item()].item()
            }
        
        except Exception as e:
            logger.error(f"❌ Dual Tower prediction error: {e}")
            return None
    
    def predict_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        Ensemble prediction combining both models
        
        Args:
            df: DataFrame with features
            
        Returns:
            Ensemble prediction results
        """
        lstm_result = self.predict_lstm(df)
        dt_result = self.predict_dual_tower(df)
        
        if lstm_result is None and dt_result is None:
            return None
        
        # Weighted voting based on model confidence
        predictions = []
        confidences = []
        
        if lstm_result is not None:
            predictions.append(lstm_result['pred_class'])
            confidences.append(lstm_result['confidence'])
        
        if dt_result is not None:
            predictions.append(dt_result['pred_class'])
            confidences.append(dt_result['confidence'])
        
        # Weighted prediction
        weights = np.array(confidences)
        weights = weights / weights.sum()
        weighted_pred = np.round(np.average(predictions, weights=weights)).astype(int)
        
        # Average probabilities
        avg_probs = {
            'downtrend': 0.0,
            'neutral': 0.0,
            'uptrend': 0.0
        }
        
        count = 0
        if lstm_result is not None:
            for key in avg_probs:
                avg_probs[key] += lstm_result['probabilities'][key]
            count += 1
        
        if dt_result is not None:
            for key in avg_probs:
                avg_probs[key] += dt_result['probabilities'][key]
            count += 1
        
        for key in avg_probs:
            avg_probs[key] /= count
        
        ensemble_confidence = max(avg_probs.values())
        
        return {
            'model': 'Ensemble',
            'prediction': self.class_map[weighted_pred],
            'pred_class': weighted_pred,
            'probabilities': avg_probs,
            'confidence': ensemble_confidence,
            'component_predictions': {
                'lstm': lstm_result['prediction'] if lstm_result else 'N/A',
                'dual_tower': dt_result['prediction'] if dt_result else 'N/A'
            }
        }


# ==============================================================================
# FORMATTING AND DISPLAY
# ==============================================================================

def print_prediction_header(ticker: str):
    """Print header for prediction results"""
    print("\n" + "=" * 80)
    print(f"STOCK TREND PREDICTION - {ticker}")
    print("=" * 80)
    print(f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training Period: November 1-30, 2024")
    print(f"Forecast: Next Monthly Trend")
    print()


def print_prediction_result(result: Dict):
    """Format and print prediction result"""
    if result is None:
        print("❌ Prediction failed")
        return
    
    model_name = result['model']
    prediction = result['prediction']
    confidence = result['confidence']
    probs = result['probabilities']
    
    # Color coding for trends
    trend_emoji = {'Downtrend': '📉', 'Neutral': '➡️', 'Uptrend': '📈'}
    emoji = trend_emoji.get(prediction, '❓')
    
    print(f"\n🤖 Model: {model_name}")
    print("-" * 80)
    print(f"Predicted Trend: {emoji} {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print()
    print("Probability Distribution:")
    print(f"  Downtrend: {probs['downtrend']:>6.2%}  ▓" + "█" * int(probs['downtrend'] * 30))
    print(f"  Neutral:   {probs['neutral']:>6.2%}  ▓" + "█" * int(probs['neutral'] * 30))
    print(f"  Uptrend:   {probs['uptrend']:>6.2%}  ▓" + "█" * int(probs['uptrend'] * 30))


def print_ensemble_result(result: Dict):
    """Format and print ensemble prediction result"""
    if result is None:
        print("❌ Ensemble prediction failed")
        return
    
    prediction = result['prediction']
    confidence = result['confidence']
    probs = result['probabilities']
    components = result['component_predictions']
    
    trend_emoji = {'Downtrend': '📉', 'Neutral': '➡️', 'Uptrend': '📈'}
    emoji = trend_emoji.get(prediction, '❓')
    
    print(f"\n🔮 Ensemble Prediction (Combined Model)")
    print("=" * 80)
    print(f"Final Prediction: {emoji} {prediction}")
    print(f"Ensemble Confidence: {confidence:.2%}")
    print()
    print("Component Predictions:")
    print(f"  LSTM Model:        {components['lstm']}")
    print(f"  Dual Tower Model:  {components['dual_tower']}")
    print()
    print("Ensemble Probability Distribution:")
    print(f"  Downtrend: {probs['downtrend']:>6.2%}  ▓" + "█" * int(probs['downtrend'] * 30))
    print(f"  Neutral:   {probs['neutral']:>6.2%}  ▓" + "█" * int(probs['neutral'] * 30))
    print(f"  Uptrend:   {probs['uptrend']:>6.2%}  ▓" + "█" * int(probs['uptrend'] * 30))


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main prediction script"""
    parser = argparse.ArgumentParser(
        description='Predict next monthly stock trend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 predict_trend.py AAPL              # Predict using both models
  python3 predict_trend.py MSFT --model lstm # LSTM only
  python3 predict_trend.py GOOGL --model dual_tower  # Dual Tower only
  python3 predict_trend.py TSLA --ensemble   # Ensemble prediction
  python3 predict_trend.py NVDA --json       # JSON output format
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str.upper,
        help='Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)'
    )
    parser.add_argument(
        '--model',
        choices=['lstm', 'dual_tower', 'both'],
        default='both',
        help='Model to use for prediction (default: both)'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Use ensemble prediction (combines both models)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples to generate for inference (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate ticker
    valid_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    if args.ticker not in valid_tickers:
        print(f"⚠️  Warning: {args.ticker} not in training data (Mag-7 stocks)")
        print(f"   Valid tickers: {', '.join(valid_tickers)}")
        print("   Proceeding with synthetic data for demonstration...\n")
    
    # Generate features
    logger.info(f"Generating synthetic features for {args.ticker}...")
    df = generate_synthetic_features(args.ticker, num_samples=args.samples)
    logger.info(f"Generated {len(df)} samples with 21 features")
    
    # Initialize predictor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = TrendPredictor(device=device)
    
    # Print header
    print_prediction_header(args.ticker)
    
    # Make predictions
    results = {}
    
    if args.ensemble:
        result = predictor.predict_ensemble(df)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_ensemble_result(result)
        results['ensemble'] = result
    
    else:
        if args.model in ['lstm', 'both']:
            result = predictor.predict_lstm(df)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_prediction_result(result)
            results['lstm'] = result
        
        if args.model in ['dual_tower', 'both']:
            result = predictor.predict_dual_tower(df)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_prediction_result(result)
            results['dual_tower'] = result
    
    # Print footer
    print("\n" + "=" * 80)
    print("Model Information:")
    print(f"  LSTM: 2-layer LSTM (57,731 parameters)")
    print(f"  Dual Tower: Tower-based (2,371 parameters)")
    print(f"  Training Data: November 1-30, 2024")
    print(f"  Features: 21 (stock, news, macro, policy)")
    print("=" * 80 + "\n")
    
    return results


if __name__ == '__main__':
    main()
