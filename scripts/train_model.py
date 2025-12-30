#!/usr/bin/env python3
"""
Unified Training Orchestrator
==============================

Trains the complete Unbreakable Trading Prediction System.

Components trained:
1. SVMD Signal Decomposition
2. GMM-HMM Regime Detection
3. TCN-LSTM-Attention (Deep Learning)
4. XGBoost, LightGBM, CatBoost (Gradient Boosting)
5. Stacking Meta-Learner
6. Risk Management Calibration
7. Continuous Learning Initialization

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 100 --days 180
    python scripts/train_model.py --quick  # Quick training for testing
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

from src.data_service import DataService
from src.ml import UnbreakablePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'logs' / f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train the Unbreakable Trading Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard training with default settings
    python scripts/train_model.py

    # Extended training for better performance
    python scripts/train_model.py --epochs 100 --days 365

    # Quick training for testing/validation
    python scripts/train_model.py --quick

    # Custom configuration
    python scripts/train_model.py --config custom_config.yaml --model-dir models/v2
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=None,
        help='Number of days of historical data to use (default: from config)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs for neural networks (default: 50)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Sequence length for LSTM models (default: 60)'
    )

    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='Hidden layer size for neural networks (default: 128)'
    )

    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Validation split ratio (default: 0.2)'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models/unbreakable',
        help='Directory to save trained models (default: models/unbreakable)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU usage even if available'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode for testing (10 epochs, 30 days)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print detailed training progress'
    )

    parser.add_argument(
        '--fetch-fresh',
        action='store_true',
        help='Fetch fresh data from exchange instead of using cached'
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_logs_directory():
    """Ensure logs directory exists."""
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)


def validate_data(df: pd.DataFrame, min_samples: int = 500) -> bool:
    """Validate the data before training."""
    if df.empty:
        logger.error("Data is empty!")
        return False

    if len(df) < min_samples:
        logger.error(f"Insufficient data: {len(df)} samples, need at least {min_samples}")
        return False

    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False

    # Check for NaN values
    nan_counts = df[required_columns[1:]].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")
        logger.info("Filling NaN values...")
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    logger.info(f"Data validation passed: {len(df)} samples")
    return True


def print_training_summary(predictor: UnbreakablePredictor, training_time: float):
    """Print training summary."""
    status = predictor.get_status()

    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Status:          {'FITTED' if status['is_fitted'] else 'NOT FITTED'}")
    print(f"Device:          {status['device']}")
    print(f"Base Models:     {status['n_base_models']}")
    for model_name in status['base_models']:
        print(f"                 - {model_name}")
    print(f"Sequence Length: {status['sequence_length']}")
    print(f"Hidden Size:     {status['hidden_size']}")
    print(f"Training Time:   {training_time:.1f} seconds")

    if status['continual_learning']:
        cl = status['continual_learning']
        print(f"\nContinual Learning:")
        print(f"  Drift Score:   {cl.get('drift_score', 0):.4f}")
        print(f"  Buffer Size:   {cl.get('buffer_size', 0)}")

    print("="*60 + "\n")


def main():
    """Main training function."""
    ensure_logs_directory()
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        args.epochs = 10
        args.days = 30
        logger.info("Quick mode enabled: 10 epochs, 30 days")

    print("\n" + "="*60)
    print("UNBREAKABLE TRADING SYSTEM - TRAINING")
    print("="*60)
    print(f"Config:          {args.config}")
    print(f"Days of data:    {args.days or 'from config'}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Hidden size:     {args.hidden_size}")
    print(f"Validation:      {args.validation_split*100:.0f}%")
    print(f"Model dir:       {args.model_dir}")
    print(f"GPU:             {'Disabled' if args.no_gpu else 'Auto-detect'}")
    print("="*60 + "\n")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    # Initialize data service
    logger.info("Initializing data service...")
    data_service = DataService(config_path=args.config)

    # Get training data
    logger.info("Loading training data...")
    if args.fetch_fresh or args.days:
        days = args.days or config['data'].get('history_days', 90)
        logger.info(f"Fetching {days} days of fresh data...")
        df = data_service.fetch_historical_data(days=days)
        data_service.save_candles(df)
    else:
        df = data_service.get_candles(limit=100000)

    # Validate data
    min_samples = max(500, args.sequence_length * 5)
    if not validate_data(df, min_samples=min_samples):
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)

    logger.info(f"Training data: {len(df)} samples")
    logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Initialize predictor
    logger.info("Initializing Unbreakable Predictor...")
    predictor = UnbreakablePredictor(
        config_path=args.config,
        model_dir=args.model_dir,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        use_gpu=not args.no_gpu
    )

    # Train
    start_time = datetime.now()
    logger.info("Starting training...")

    try:
        predictor.fit(
            df=df,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=args.validation_split,
            verbose=args.verbose
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    training_time = (datetime.now() - start_time).total_seconds()

    # Print summary
    print_training_summary(predictor, training_time)

    # Test prediction
    logger.info("Testing prediction on latest data...")
    try:
        signal = predictor.predict(df.tail(100))
        print("\nTest Prediction:")
        print(f"  Direction:     {signal.direction}")
        print(f"  Confidence:    {signal.confidence:.2%}")
        print(f"  Probability:   {signal.probability:.4f}")
        print(f"  Regime:        {signal.regime} ({signal.regime_confidence:.2%})")
        print(f"  Entry Price:   ${signal.entry_price:,.2f}")
        print(f"  Stop Loss:     ${signal.stop_loss:,.2f}")
        print(f"  Take Profit:   ${signal.take_profit:,.2f}")
        print(f"  Position Size: {signal.position_size_pct:.2%}")
        print(f"  Risk/Reward:   {signal.risk_reward_ratio:.2f}")
        if signal.warnings:
            print(f"  Warnings:      {', '.join(signal.warnings)}")
    except Exception as e:
        logger.warning(f"Test prediction failed: {e}")

    logger.info(f"Models saved to: {args.model_dir}")
    logger.info("Training complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
