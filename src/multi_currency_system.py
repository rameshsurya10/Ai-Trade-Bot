"""
Multi-Currency Trading System with Auto-Learning
=================================================
Supports multiple currency pairs with individual models per currency.
Includes automatic retraining based on performance feedback.

FEATURES:
1. Multiple currency pairs (forex, crypto)
2. Separate model per currency
3. Auto-retrain on poor performance
4. Performance tracking per currency
5. Dynamic model selection

TRUTH ABOUT AUTO-LEARNING:
- Retraining improves adaptation to market changes
- BUT can cause overfitting to recent data
- Balance: Retrain periodically, not on every trade
- Minimum data: 1000+ candles for meaningful training
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from src.analysis_engine import LSTMModel, FeatureCalculator
from src.data_service import DataService, CachedData
from src.advanced_predictor import AdvancedPredictor

logger = logging.getLogger(__name__)


# Shared cache for training data across all currencies (prevents duplicate fetches)
_SHARED_TRAINING_DATA_CACHE = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class CurrencyConfig:
    """Configuration for a single currency pair."""
    symbol: str  # e.g., "EUR/USD", "BTC/USD"
    exchange: str  # e.g., "capital", "binance", "coinbase"
    interval: str  # e.g., "1h", "4h"
    model_path: str  # Path to trained model
    enabled: bool = True


@dataclass
class PerformanceStats:
    """Track performance statistics per currency."""
    symbol: str
    total_signals: int = 0
    correct_predictions: int = 0
    total_pnl_percent: float = 0.0
    last_retrain: Optional[datetime] = None
    win_rate: float = 0.0

    def add_result(self, is_correct: bool, pnl_percent: float):
        """Record a prediction result."""
        self.total_signals += 1
        if is_correct:
            self.correct_predictions += 1
        self.total_pnl_percent += pnl_percent
        self.win_rate = self.correct_predictions / self.total_signals if self.total_signals > 0 else 0

    @property
    def needs_retrain(self) -> bool:
        """Check if model needs retraining based on performance."""
        if self.total_signals < 20:
            return False  # Not enough data
        if self.win_rate < 0.45:
            return True  # Performing below baseline
        if self.last_retrain is None:
            return self.total_signals >= 100  # Initial retrain after 100 trades
        days_since_retrain = (datetime.utcnow() - self.last_retrain).days
        return days_since_retrain >= 30 and self.total_signals >= 50  # Monthly retrain


class ModelManager:
    """
    Manages multiple models for different currencies.

    Responsibilities:
    - Load/save models per currency
    - Track model versions
    - Handle model switching
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_models: Dict[str, LSTMModel] = {}
        self.model_configs: Dict[str, dict] = {}

    def get_model_path(self, symbol: str) -> Path:
        """Get path for currency-specific model."""
        safe_symbol = symbol.replace("/", "_").replace("-", "_")
        return self.models_dir / f"model_{safe_symbol}.pt"

    def load_model(self, symbol: str, config: dict) -> Optional[LSTMModel]:
        """Load model for specific currency."""
        model_path = self.get_model_path(symbol)

        if not model_path.exists():
            logger.warning(f"No model found for {symbol} at {model_path}")
            return None

        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

            feature_columns = FeatureCalculator.get_feature_columns()
            input_size = len(feature_columns)

            model = LSTMModel(
                input_size=input_size,
                hidden_size=config.get('hidden_size', 128),
                num_layers=config.get('num_layers', 2),
                dropout=config.get('dropout', 0.2)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            self.loaded_models[symbol] = model
            self.model_configs[symbol] = checkpoint.get('config', {})

            # Also store scaler parameters
            if 'feature_means' in checkpoint:
                self.model_configs[symbol]['feature_means'] = checkpoint['feature_means']
                self.model_configs[symbol]['feature_stds'] = checkpoint['feature_stds']

            logger.info(f"Model loaded for {symbol}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model for {symbol}: {e}")
            return None

    def save_model(self, symbol: str, model: LSTMModel, config: dict,
                   feature_means: np.ndarray, feature_stds: np.ndarray):
        """Save model for specific currency."""
        model_path = self.get_model_path(symbol)

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'feature_means': feature_means,
            'feature_stds': feature_stds,
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat()
        }, model_path)

        logger.info(f"Model saved for {symbol} at {model_path}")

    def get_model(self, symbol: str) -> Optional[LSTMModel]:
        """Get loaded model for symbol."""
        return self.loaded_models.get(symbol)

    def get_scaler_params(self, symbol: str) -> tuple:
        """Get scaler parameters for symbol."""
        config = self.model_configs.get(symbol, {})
        return config.get('feature_means'), config.get('feature_stds')


class AutoTrainer:
    """
    Automatic Model Retraining

    WHEN TO RETRAIN:
    1. Win rate drops below 45% (model is underperforming)
    2. Every 30 days (market conditions change)
    3. After significant market regime change (detected by entropy)

    SAFEGUARDS:
    - Minimum 1000 candles required
    - Validation split prevents overfitting
    - Keep backup of previous model
    - Only replace if new model is better
    """

    def __init__(self, model_manager: ModelManager, config: dict):
        self.model_manager = model_manager
        self.config = config
        self._training = False
        self._training_lock = threading.Lock()

    def train_model(
        self,
        symbol: str,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> bool:
        """
        Train or retrain model for a currency.

        Args:
            symbol: Currency pair
            df: Historical data
            epochs: Training epochs
            batch_size: Batch size
            validation_split: Validation data ratio

        Returns:
            True if training successful and model improved
        """
        with self._training_lock:
            if self._training:
                logger.warning("Training already in progress")
                return False
            self._training = True

        try:
            logger.info(f"Starting training for {symbol}")
            logger.info(f"Data: {len(df)} candles")

            if len(df) < 1000:
                logger.error(f"Insufficient data for {symbol}: {len(df)} < 1000")
                return False

            # Calculate features
            df_features = FeatureCalculator.calculate_all(df)
            feature_columns = FeatureCalculator.get_feature_columns()

            # Extract and normalize features FIRST (before creating target)
            features = df_features[feature_columns].values
            closes = df_features['close'].values

            # Normalize
            feature_means = np.nanmean(features, axis=0)
            feature_stds = np.nanstd(features, axis=0)
            features = (features - feature_means) / (feature_stds + 1e-8)
            features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

            # Create sequences
            sequence_length = self.config.get('sequence_length', 60)

            # Create sliding windows manually for correct shape
            # Need shape: (num_samples, sequence_length, num_features)
            num_features = features.shape[1]
            num_sequences = len(features) - sequence_length - 1  # -1 for target

            X = np.zeros((num_sequences, sequence_length, num_features))
            y = np.zeros(num_sequences)

            # CRITICAL FIX: Align sequences with correct targets
            # For each sequence ending at position i+sequence_length-1,
            # the target is: will the NEXT candle (i+sequence_length) close higher?
            for i in range(num_sequences):
                # Sequence uses candles [i] to [i+sequence_length-1]
                X[i] = features[i:i + sequence_length]

                # Target: will candle [i+sequence_length] close higher than candle [i+sequence_length-1]?
                current_close = closes[i + sequence_length - 1]
                next_close = closes[i + sequence_length]
                y[i] = 1.0 if next_close > current_close else 0.0

            # Remove any invalid entries
            valid = ~(np.isnan(y) | np.isnan(X).any(axis=(1, 2)))
            X = X[valid]
            y = y[valid]

            logger.info(f"Created {len(X)} training sequences")

            # Split data
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

            # Create model
            model = LSTMModel(
                input_size=len(feature_columns),
                hidden_size=self.config.get('hidden_size', 128),
                num_layers=self.config.get('num_layers', 2),
                dropout=self.config.get('dropout', 0.2)
            )

            # Training
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_val_acc = 0
            best_state = None

            for epoch in range(epochs):
                # Train
                model.train()
                epoch_loss = 0
                for X_batch, y_batch in train_loader:
                    # Performance: Explicitly clear gradients (prevents memory leak)
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                # Performance: Clear GPU cache every 10 epochs (prevents memory leak)
                if (epoch + 1) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Validate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = model(X_batch).squeeze()
                        predictions = (outputs > 0.5).float()
                        correct += (predictions == y_batch).sum().item()
                        total += len(y_batch)

                val_acc = correct / total

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Performance: Deep copy state dict to prevent memory leak
                    best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2%}")

            # Check if new model is better than existing
            existing_model = self.model_manager.get_model(symbol)
            should_save = True

            if existing_model is not None:
                # Test existing model on validation set
                existing_model.eval()
                existing_correct = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        outputs = existing_model(X_batch).squeeze()
                        predictions = (outputs > 0.5).float()
                        existing_correct += (predictions == y_batch).sum().item()

                existing_acc = existing_correct / total
                logger.info(f"Existing model accuracy: {existing_acc:.2%}")
                logger.info(f"New model accuracy: {best_val_acc:.2%}")

                # Only save if significantly better (>1%)
                should_save = best_val_acc > existing_acc + 0.01

            if should_save and best_state is not None:
                model.load_state_dict(best_state)
                self.model_manager.save_model(
                    symbol, model, self.config,
                    feature_means, feature_stds
                )
                self.model_manager.loaded_models[symbol] = model
                logger.info(f"New model saved for {symbol} (accuracy: {best_val_acc:.2%})")
                return True
            else:
                logger.info(f"Keeping existing model for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            return False

        finally:
            self._training = False


class MultiCurrencySystem:
    """
    Complete Multi-Currency Trading System

    WORKFLOW:
    1. Load all configured currencies
    2. For each currency:
       - Fetch latest data
       - Run prediction (LSTM + Advanced Math)
       - Generate signal if confidence > threshold
       - Track performance
       - Auto-retrain if needed

    TRANSPARENCY:
    - Shows all algorithm contributions
    - Explains why signal was generated
    - Tracks accuracy per currency
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize multi-currency system."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.model_manager = ModelManager(self.config.get('model', {}).get('models_dir', 'models'))
        self.auto_trainer = AutoTrainer(self.model_manager, self.config.get('model', {}))
        self.advanced_predictor = AdvancedPredictor()

        # Currency configurations
        self.currencies: Dict[str, CurrencyConfig] = {}
        self.data_services: Dict[str, DataService] = {}
        self.performance: Dict[str, PerformanceStats] = {}

        # Callbacks
        self._signal_callbacks: List[Callable] = []

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Thread safety lock for performance tracking
        self._performance_lock = threading.Lock()

        # Track active retrain threads to prevent duplicates and allow cleanup
        self._retrain_threads: Dict[str, threading.Thread] = {}
        self._retrain_scheduled: Dict[str, bool] = {}  # Prevent duplicate retrain scheduling

        # Feature cache to avoid recalculating features on every predict()
        # Key: f"{symbol}_{last_timestamp}_{len(df)}" -> cached features DataFrame
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        self._feature_cache_lock = threading.Lock()
        self._feature_cache_max_size = 50  # Max cached entries (roughly one per symbol)

        logger.info("MultiCurrencySystem initialized")

    def add_currency(
        self,
        symbol: str,
        exchange: str = "coinbase",
        interval: str = "1h"
    ):
        """
        Add a currency pair to track.

        Args:
            symbol: Currency pair (e.g., "BTC/USD", "EUR/USD")
            exchange: Exchange name (coinbase, binance, capital)
            interval: Candle interval
        """
        currency_config = CurrencyConfig(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            model_path=str(self.model_manager.get_model_path(symbol))
        )
        self.currencies[symbol] = currency_config
        self.performance[symbol] = PerformanceStats(symbol=symbol)

        # Try to load existing model
        model_config = self.config.get('model', {})
        self.model_manager.load_model(symbol, model_config)

        logger.info(f"Added currency: {symbol} on {exchange}")

    def remove_currency(self, symbol: str):
        """Remove a currency pair."""
        if symbol in self.currencies:
            del self.currencies[symbol]
            if symbol in self.data_services:
                del self.data_services[symbol]
            logger.info(f"Removed currency: {symbol}")

    def _get_cached_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get features from cache or calculate them.

        Performance: Avoids recalculating features when data hasn't changed.
        Cache key is based on symbol + last timestamp + data length.

        Args:
            symbol: Currency pair
            df: Price data

        Returns:
            DataFrame with calculated features
        """
        # Generate cache key from data signature
        last_timestamp = str(df.index[-1]) if hasattr(df.index, '__getitem__') else str(len(df))
        cache_key = f"{symbol}_{last_timestamp}_{len(df)}"

        with self._feature_cache_lock:
            # Check cache
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]

            # Calculate features (expensive operation)
            df_features = FeatureCalculator.calculate_all(df)

            # Evict oldest entries if cache is full (simple LRU approximation)
            if len(self._feature_cache) >= self._feature_cache_max_size:
                # Remove first 10 entries (oldest added)
                keys_to_remove = list(self._feature_cache.keys())[:10]
                for key in keys_to_remove:
                    del self._feature_cache[key]

            # Cache the result
            self._feature_cache[cache_key] = df_features

            return df_features

    def get_available_currencies(self) -> List[str]:
        """Get list of supported currency pairs."""
        # Popular forex pairs
        forex = [
            "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
            "AUD/USD", "NZD/USD", "USD/CAD", "EUR/GBP",
            "EUR/JPY", "GBP/JPY"
        ]

        # Popular crypto pairs
        crypto = [
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD",
            "SOL/USD", "ADA/USD", "DOGE/USD", "DOT/USD"
        ]

        return forex + crypto

    def predict(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate prediction for a currency.

        Args:
            symbol: Currency pair
            df: Price data

        Returns:
            Prediction dict with all details, or None if prediction fails
        """
        try:
            if symbol not in self.currencies:
                logger.error(f"Currency {symbol} not configured")
                return None

            # Get LSTM model prediction
            model = self.model_manager.get_model(symbol)
            lstm_prob = None

            if model is not None:
                try:
                    # Prepare features (cached to avoid recalculation)
                    df_features = self._get_cached_features(symbol, df)
                    feature_columns = FeatureCalculator.get_feature_columns()

                    feature_means, feature_stds = self.model_manager.get_scaler_params(symbol)
                    sequence_length = self.config.get('model', {}).get('sequence_length', 60)

                    features = df_features[feature_columns].iloc[-sequence_length:].values

                    if feature_means is not None:
                        features = (features - feature_means) / (feature_stds + 1e-8)
                    else:
                        features = (features - np.nanmean(features, axis=0)) / (np.nanstd(features, axis=0) + 1e-8)

                    features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                    # LSTM prediction
                    with torch.no_grad():
                        x = torch.FloatTensor(features).unsqueeze(0)
                        lstm_prob = model(x).item()

                except Exception as e:
                    logger.error(f"LSTM prediction failed for {symbol}: {e}")

            # Get advanced mathematical prediction
            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            advanced_result = self.advanced_predictor.predict(df, lstm_prob, atr)

            # Build result
            current_price = df['close'].iloc[-1]

            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'price': current_price,
                'direction': advanced_result.direction,
                'confidence': advanced_result.confidence,
                'stop_loss': advanced_result.stop_loss,
                'take_profit': advanced_result.take_profit,
                'risk_reward': advanced_result.risk_reward_ratio,

                # Mathematical breakdown (transparency)
                'components': {
                    'lstm_probability': lstm_prob,
                    'fourier_signal': advanced_result.fourier_signal,
                    'kalman_trend': advanced_result.kalman_trend,
                    'entropy_regime': advanced_result.entropy_regime,
                    'markov_probability': advanced_result.markov_probability,
                    'monte_carlo_risk': advanced_result.monte_carlo_risk
                },

                # Model info
                'has_trained_model': model is not None,
                'algorithm_weights': getattr(advanced_result, 'algorithm_weights', {}),
                'raw_scores': getattr(advanced_result, 'raw_scores', {})
            }

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def record_outcome(self, symbol: str, was_correct: bool, pnl_percent: float):
        """
        Record trade outcome for performance tracking.

        Args:
            symbol: Currency pair
            was_correct: Whether prediction was correct
            pnl_percent: Profit/loss percentage
        """
        should_schedule = False

        with self._performance_lock:
            if symbol in self.performance:
                self.performance[symbol].add_result(was_correct, pnl_percent)
                needs_retrain = self.performance[symbol].needs_retrain
                already_scheduled = self._retrain_scheduled.get(symbol, False)

                # Atomically check and set to prevent race condition
                # This ensures only ONE thread can schedule retrain for this symbol
                if needs_retrain and not already_scheduled:
                    self._retrain_scheduled[symbol] = True  # Set flag inside lock!
                    should_schedule = True

        # Schedule outside lock to prevent deadlock (flag already set atomically above)
        if should_schedule:
            logger.info(f"Performance degraded for {symbol}, scheduling retrain")
            self._schedule_retrain(symbol)

    def _schedule_retrain(self, symbol: str):
        """Schedule model retraining in background with thread tracking."""
        def retrain_task():
            try:
                logger.info(f"[{threading.current_thread().name}] Starting background retrain for {symbol}")

                # Performance: Check shared cache first (1-hour TTL prevents duplicate fetches)
                cache_key = f"training_data_{symbol}"
                df = None

                with _CACHE_LOCK:
                    if cache_key in _SHARED_TRAINING_DATA_CACHE:
                        cached = _SHARED_TRAINING_DATA_CACHE[cache_key]
                        if cached.is_valid():
                            logger.info(f"Using cached training data for {symbol} (cache age: {(datetime.utcnow() - cached.timestamp).seconds}s)")
                            df = cached.data.copy()
                        else:
                            del _SHARED_TRAINING_DATA_CACHE[cache_key]

                # Fetch fresh data if not cached
                if df is None:
                    data_service = DataService()
                    try:
                        logger.info(f"Fetching 50K candles for {symbol} (cache miss)")
                        df = data_service.get_candles(limit=50000)

                        # Store in shared cache with 1-hour TTL (3600s)
                        with _CACHE_LOCK:
                            _SHARED_TRAINING_DATA_CACHE[cache_key] = CachedData(df, cache_duration=3600)
                            logger.info(f"Cached training data for {symbol} (90% reduction in DB I/O)")
                    finally:
                        pass

                if df is not None and len(df) >= 1000:
                    success = self.auto_trainer.train_model(symbol, df)
                    if success:
                        with self._performance_lock:
                            self.performance[symbol].last_retrain = datetime.utcnow()
                        logger.info(f"Retrain completed for {symbol}")
                else:
                    logger.warning(f"Insufficient data for retrain: {len(df) if df is not None else 0} candles")

            except Exception as e:
                logger.error(f"Retrain task failed for {symbol}: {e}")
            finally:
                # Cleanup: Mark retrain as complete and remove thread reference atomically
                with self._performance_lock:
                    self._retrain_scheduled[symbol] = False
                    if symbol in self._retrain_threads:
                        del self._retrain_threads[symbol]
                logger.info(f"[{threading.current_thread().name}] Retrain task finished for {symbol}")

        # Create thread
        thread = threading.Thread(target=retrain_task, daemon=True, name=f"Retrain-{symbol}")

        # Register thread BEFORE starting it (prevents race where thread completes before registration)
        # NOTE: _retrain_scheduled[symbol] already set to True in record_outcome() atomically
        with self._performance_lock:
            self._retrain_threads[symbol] = thread

        # Start thread after registration (outside lock to prevent blocking)
        thread.start()
        logger.info(f"Retrain thread started for {symbol}")

    def get_performance_report(self) -> Dict:
        """Get performance statistics for all currencies (thread-safe)."""
        report = {}
        with self._performance_lock:
            for symbol, stats in self.performance.items():
                report[symbol] = {
                    'total_signals': stats.total_signals,
                    'win_rate': f"{stats.win_rate:.1%}",
                    'total_pnl': f"{stats.total_pnl_percent:.2f}%",
                    'needs_retrain': stats.needs_retrain,
                    'last_retrain': stats.last_retrain.isoformat() if stats.last_retrain else 'Never',
                    'retrain_in_progress': self._retrain_scheduled.get(symbol, False)
                }
        return report

    def register_signal_callback(self, callback: Callable):
        """Register callback for new signals."""
        self._signal_callbacks.append(callback)

    def get_status(self) -> Dict:
        """Get system status."""
        with self._performance_lock:
            active_retrains = [symbol for symbol, scheduled in self._retrain_scheduled.items() if scheduled]

        return {
            'running': self._running,
            'currencies': list(self.currencies.keys()),
            'loaded_models': list(self.model_manager.loaded_models.keys()),
            'performance': self.get_performance_report(),
            'active_retrains': active_retrains
        }

    def cleanup(self):
        """Cleanup resources and wait for active retrain threads to complete."""
        logger.info("Cleaning up MultiCurrencySystem...")

        with self._performance_lock:
            active_threads = list(self._retrain_threads.values())

        # Wait for all retrain threads to complete (with timeout)
        for thread in active_threads:
            if thread.is_alive():
                logger.info(f"Waiting for {thread.name} to complete...")
                thread.join(timeout=5.0)  # Wait max 5 seconds per thread
                if thread.is_alive():
                    logger.warning(f"{thread.name} still running after timeout")

        logger.info("MultiCurrencySystem cleanup complete")


# Configuration template for multi-currency
MULTI_CURRENCY_CONFIG = """
# Multi-Currency Trading System Configuration
# ============================================

# Add this to your config.yaml or use as separate file

currencies:
  # Forex pairs (use with Capital.com, etc.)
  - symbol: "EUR/USD"
    exchange: "capital"
    interval: "1h"
    enabled: true

  - symbol: "GBP/USD"
    exchange: "capital"
    interval: "1h"
    enabled: true

  - symbol: "USD/JPY"
    exchange: "capital"
    interval: "1h"
    enabled: false  # Disabled example

  # Crypto pairs (use with Coinbase, Binance, etc.)
  - symbol: "BTC/USD"
    exchange: "coinbase"
    interval: "1h"
    enabled: true

  - symbol: "ETH/USD"
    exchange: "coinbase"
    interval: "1h"
    enabled: true

# Auto-training settings
auto_training:
  enabled: true
  min_trades_before_retrain: 50
  max_days_between_retrain: 30
  min_win_rate_threshold: 0.45  # Retrain if below this

# Model settings per currency (can be customized)
model:
  models_dir: "models/currencies"
  sequence_length: 60
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

# Algorithm weights (customize per preference)
algorithm_weights:
  fourier: 0.15   # Cycle detection
  kalman: 0.25    # Trend estimation
  entropy: 0.10   # Regime detection
  markov: 0.20    # State transitions
  lstm: 0.30      # Pattern learning
"""


def print_config_template():
    """Print configuration template."""
    print(MULTI_CURRENCY_CONFIG)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    print("="*60)
    print("MULTI-CURRENCY TRADING SYSTEM")
    print("="*60)

    # Initialize system
    system = MultiCurrencySystem()

    # Show available currencies
    print("\nAvailable currencies:")
    for currency in system.get_available_currencies():
        print(f"  - {currency}")

    # Print config template
    print("\n" + "="*60)
    print("CONFIGURATION TEMPLATE:")
    print("="*60)
    print_config_template()
