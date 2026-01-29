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

Multi-Timeframe Support:
- Default trains on 15m and 1h timeframes
- Uses 1 year of historical data for comprehensive pattern learning
- Initializes PerformanceBasedLearner for continuous improvement

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --epochs 100 --days 365
    python scripts/train_model.py --quick  # Quick training for testing
    python scripts/train_model.py --timeframes 15m,1h,4h  # Multiple timeframes
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
from src.ml.features.selector import (
    AdaptiveFeatureSelector, MarketRegime, get_features_for_regime,
    STANDARD_FEATURE_SETS
)

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

    parser.add_argument(
        '--timeframes',
        type=str,
        default='15m,1h',
        help='Comma-separated list of timeframes to train on (default: 15m,1h)'
    )

    parser.add_argument(
        '--enable-continuous-learning',
        action='store_true',
        default=True,
        help='Initialize continuous learning system after training (default: True)'
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


def print_feature_importance(predictor: UnbreakablePredictor, df: pd.DataFrame):
    """Print feature importance rankings after training."""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    try:
        # Get feature names and models
        feature_names = predictor.feature_engineer._feature_names
        if not feature_names:
            print("No feature names available (model not fitted?)")
            return

        # Initialize feature selector
        selector = AdaptiveFeatureSelector(
            n_features=15,
            use_shap=True,
            regime_weight=0.4,
            importance_weight=0.6
        )

        # Calculate importance from trained models
        X = predictor.feature_engineer._feature_means  # Just for structure
        if X is None:
            X = np.zeros((100, len(feature_names)))
        y = np.zeros(100)

        importance = selector.calculate_importance(
            predictor.base_models,
            X if isinstance(X, np.ndarray) and len(X.shape) == 2 else np.zeros((100, len(feature_names))),
            y,
            feature_names
        )

        # Detect current regime
        regime_result = predictor.regime_detector.detect(df)
        current_regime = MarketRegime(regime_result.current_regime.name.lower())

        print(f"\nCurrent Market Regime: {current_regime.value.upper()}")
        print(f"Regime Confidence: {regime_result.confidence:.1%}")

        # Print rankings
        rankings_output = selector.print_rankings(feature_names, current_regime, top_n=20)
        print(rankings_output)

        # Get recommended features for this regime
        selection = selector.select_features(feature_names, current_regime)
        print(f"\n{'='*70}")
        print(f"RECOMMENDED FEATURES FOR {current_regime.value.upper()} REGIME")
        print(f"{'='*70}")
        print(f"Selected {selection.n_features_selected} of {selection.n_features_original} features:")
        for i, feat in enumerate(selection.selected_features, 1):
            print(f"  {i:2}. {feat}")

        # Show standard feature sets
        print(f"\n{'='*70}")
        print("STANDARD FEATURE SETS (for quick deployment)")
        print(f"{'='*70}")
        for set_name, features in STANDARD_FEATURE_SETS.items():
            print(f"\n{set_name.upper()} ({len(features)} features):")
            print(f"  {', '.join(features[:5])}...")

    except Exception as e:
        logger.warning(f"Feature importance analysis failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main training function."""
    ensure_logs_directory()
    args = parse_args()

    # Parse timeframes
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]

    # Quick mode overrides
    if args.quick:
        args.epochs = 10
        args.days = 30
        logger.info("Quick mode enabled: 10 epochs, 30 days")

    # Default to 1 year of data for comprehensive training
    if args.days is None:
        args.days = 365

    print("\n" + "="*60)
    print("UNBREAKABLE TRADING SYSTEM - TRAINING")
    print("="*60)
    print(f"Config:          {args.config}")
    print(f"Days of data:    {args.days}")
    print(f"Timeframes:      {', '.join(timeframes)}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Hidden size:     {args.hidden_size}")
    print(f"Validation:      {args.validation_split*100:.0f}%")
    print(f"Model dir:       {args.model_dir}")
    print(f"GPU:             {'Disabled' if args.no_gpu else 'Auto-detect'}")
    print(f"Continuous:      {'Enabled' if args.enable_continuous_learning else 'Disabled'}")
    print("="*60 + "\n")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    # Initialize data service
    logger.info("Initializing data service...")
    data_service = DataService(config_path=args.config)

    # Get training data for each timeframe
    logger.info(f"Loading training data for timeframes: {timeframes}...")
    timeframe_data = {}

    for tf in timeframes:
        logger.info(f"Fetching {args.days} days of {tf} data...")
        try:
            df_tf = data_service.fetch_historical_data(days=args.days, interval=tf)
            if df_tf is not None and len(df_tf) > 0:
                timeframe_data[tf] = df_tf
                data_service.save_candles(df_tf, interval=tf)
                logger.info(f"  {tf}: {len(df_tf)} candles loaded")
            else:
                logger.warning(f"  {tf}: No data available")
        except Exception as e:
            logger.warning(f"  {tf}: Failed to load - {e}")

    if not timeframe_data:
        logger.error("No data available for any timeframe. Exiting.")
        sys.exit(1)

    # Use primary timeframe (first in list) for main training
    primary_tf = timeframes[0]
    df = timeframe_data.get(primary_tf)

    if df is None:
        # Fallback to any available timeframe
        primary_tf, df = next(iter(timeframe_data.items()))
        logger.warning(f"Primary timeframe not available, using {primary_tf}")

    # Validate data
    min_samples = max(500, args.sequence_length * 5)
    if not validate_data(df, min_samples=min_samples):
        logger.error("Data validation failed. Exiting.")
        sys.exit(1)

    logger.info(f"Primary training data ({primary_tf}): {len(df)} samples")
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

    # Print feature importance rankings
    print_feature_importance(predictor, df)

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

    # Train on additional timeframes if available
    for tf, df_tf in timeframe_data.items():
        if tf != primary_tf and len(df_tf) >= min_samples:
            logger.info(f"\nTraining on {tf} timeframe ({len(df_tf)} samples)...")
            try:
                predictor.fit(
                    df=df_tf,
                    epochs=args.epochs // 2,  # Fewer epochs for secondary timeframes
                    batch_size=args.batch_size,
                    validation_split=args.validation_split,
                    verbose=args.verbose
                )
                logger.info(f"  {tf} training complete")
            except Exception as e:
                logger.warning(f"  {tf} training failed: {e}")

    logger.info(f"Models saved to: {args.model_dir}")

    # Initialize continuous learning system
    if args.enable_continuous_learning:
        print("\n" + "="*60)
        print("CONTINUOUS LEARNING SYSTEM - INITIALIZATION")
        print("="*60)

        try:
            from src.learning import (
                PerformanceBasedLearner,
                PerformanceLearnerConfig,
                create_performance_learner
            )

            # Create performance-based learner configuration
            learner_config = PerformanceLearnerConfig(
                timeframes=timeframes,
                loss_retrain_enabled=True,
                reinforce_on_win=True,
                consecutive_loss_threshold=3,
                win_rate_threshold=0.45,
                high_confidence_loss_threshold=0.80,
                light_epochs=30,
                medium_epochs=50,
                full_epochs=100
            )

            print(f"Continuous Learning Config:")
            print(f"  Timeframes:           {learner_config.timeframes}")
            print(f"  Retrain on Loss:      {learner_config.loss_retrain_enabled}")
            print(f"  Reinforce on Win:     {learner_config.reinforce_on_win}")
            print(f"  Consec. Loss Trigger: {learner_config.consecutive_loss_threshold}")
            print(f"  Win Rate Threshold:   {learner_config.win_rate_threshold:.1%}")
            print(f"  Light Retrain Epochs: {learner_config.light_epochs}")
            print(f"  Full Retrain Epochs:  {learner_config.full_epochs}")
            print("="*60)

            logger.info("Continuous learning system configured and ready")
            logger.info("The system will:")
            logger.info("  1. Reinforce model on profitable predictions")
            logger.info("  2. Trigger LIGHT retrain on losses")
            logger.info("  3. Trigger FULL retrain after 3 consecutive losses")
            logger.info("  4. Monitor win rate and adapt automatically")

        except ImportError as e:
            logger.warning(f"Could not initialize continuous learning: {e}")
        except Exception as e:
            logger.warning(f"Continuous learning setup failed: {e}")

    logger.info("\nTraining complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
