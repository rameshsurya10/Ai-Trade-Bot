#!/usr/bin/env python3
"""
Train LSTM Model
================

Trains the LSTM model for price direction prediction.

Usage:
    python scripts/train_model.py

Options:
    --epochs 100        Training epochs (default: 100)
    --batch-size 32     Batch size (default: 32)
    --validation 0.2    Validation split (default: 0.2)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
import yaml

from src.data_service import DataService
from src.analysis_engine import AnalysisEngine, LSTMModel, FeatureCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml"):
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_sequences(df: pd.DataFrame, feature_columns: list,
                      sequence_length: int = 60) -> tuple:
    """
    Prepare sequences for LSTM training.

    Returns:
        X: shape (samples, sequence_length, features)
        y: shape (samples,) - 1 if price went up, 0 if down
    """
    # Calculate features
    df_features = FeatureCalculator.calculate_all(df)

    # Create target: 1 if next candle closes higher, 0 otherwise
    df_features['target'] = (df_features['close'].shift(-1) > df_features['close']).astype(int)

    # Get features
    features = df_features[feature_columns].values
    targets = df_features['target'].values

    # Normalize features (z-score)
    feature_means = np.nanmean(features, axis=0)
    feature_stds = np.nanstd(features, axis=0)
    features = (features - feature_means) / (feature_stds + 1e-8)

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features) - 1):
        X.append(features[i-sequence_length:i])
        y.append(targets[i])

    X = np.array(X)
    y = np.array(y)

    # Remove any remaining NaN targets
    valid = ~np.isnan(y)
    X = X[valid]
    y = y[valid]

    return X, y, feature_means, feature_stds


def train_model(config_path: str = "config.yaml",
                epochs: int = 100,
                batch_size: int = 32,
                validation_split: float = 0.2):
    """Train the LSTM model."""

    config = load_config(config_path)

    logger.info("="*60)
    logger.info("TRAINING LSTM MODEL")
    logger.info("="*60)

    # Load data
    logger.info("Loading data...")
    data_service = DataService(config_path)
    df = data_service.get_candles(limit=100000)

    if len(df) < 1000:
        logger.error(f"Not enough data! Have {len(df)} candles, need at least 1000")
        logger.info("Run: python scripts/download_data.py --days 365")
        sys.exit(1)

    logger.info(f"Loaded {len(df)} candles")

    # Model config
    sequence_length = config['model']['sequence_length']
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']

    feature_columns = FeatureCalculator.get_feature_columns()
    input_size = len(feature_columns)

    logger.info(f"Features: {input_size}")
    logger.info(f"Sequence length: {sequence_length}")

    # Prepare data
    logger.info("Preparing sequences...")
    X, y, feature_means, feature_stds = prepare_sequences(
        df, feature_columns, sequence_length
    )

    logger.info(f"Total sequences: {len(X)}")
    logger.info(f"Class balance: {y.mean():.2%} positive")

    # Time series split (walk-forward)
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 20
    patience_counter = 0

    logger.info("\nStarting training...")
    logger.info("-"*60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0

            # Save model
            model_path = Path(config['model']['path'])
            model_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'model_state_dict': model.state_dict(),
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'config': {
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'sequence_length': sequence_length
                },
                'training_info': {
                    'epoch': epoch,
                    'val_accuracy': val_acc,
                    'val_loss': val_loss,
                    'timestamp': datetime.now().isoformat()
                }
            }, model_path)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("-"*60)
    logger.info("TRAINING COMPLETE")
    logger.info("-"*60)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2%}")
    logger.info(f"Best Validation Loss: {best_val_loss:.4f}")
    logger.info(f"Model saved: {config['model']['path']}")

    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("MODEL EVALUATION")
    logger.info("="*60)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch).squeeze()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    predictions = (all_preds > 0.5).astype(int)
    accuracy = (predictions == all_targets).mean()

    # Confidence distribution
    high_conf = all_preds[(all_preds > 0.65) | (all_preds < 0.35)]
    logger.info(f"\nOverall Accuracy: {accuracy:.2%}")
    logger.info(f"High Confidence Predictions: {len(high_conf)} ({len(high_conf)/len(all_preds):.1%})")

    if len(high_conf) > 0:
        high_conf_preds = (all_preds > 0.65) | (all_preds < 0.35)
        high_conf_acc = (predictions[high_conf_preds] == all_targets[high_conf_preds]).mean()
        logger.info(f"High Confidence Accuracy: {high_conf_acc:.2%}")

    logger.info("\n⚠️  IMPORTANT:")
    logger.info("   - 52-55% accuracy is GOOD for trading")
    logger.info("   - Combined with 1:2 risk:reward = profitable")
    logger.info("   - Never expect >60% accuracy consistently")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--validation', type=float, default=0.2,
                       help='Validation split')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file path')

    args = parser.parse_args()

    train_model(
        config_path=args.config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation
    )


if __name__ == "__main__":
    main()
