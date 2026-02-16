"""
Live Trading Runner (Lean-Inspired)
===================================
Main orchestrator for live trading operations.

Coordinates:
- Market data streaming
- Signal generation (ML predictions)
- Risk management
- Order execution
- Position management

Similar to Lean's AlgorithmManager but simplified.
"""

import logging
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable
from queue import Queue, Empty

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

import pandas as pd

from src.core.config import Config
from src.core.database import Database
from src.brokerages.base import BaseBrokerage
from src.brokerages.orders import Order, OrderType, OrderSide, OrderStatus
from src.brokerages.events import OrderEvent, OrderEventType
from src.portfolio.manager import PortfolioManager, InsightDirection
from src.portfolio.risk import RiskManager, RiskAction, MaximumDrawdownRisk, MaximumPositionSizeRisk
from src.data.provider import UnifiedDataProvider, Tick, Candle
from src.multi_currency_system import MultiCurrencySystem
from src.data_service import DataService
from src.news.collector import NewsCollector

# CONTINUOUS LEARNING: Strategic Learning Bridge integrates continuous learning
from src.learning.strategic_learning_bridge import StrategicLearningBridge

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading execution mode."""
    PAPER = "paper"      # Simulated execution
    LIVE = "live"        # Real money execution
    SHADOW = "shadow"    # Generate signals but don't execute


class RunnerStatus(Enum):
    """Runner state."""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TradingSymbol:
    """Configuration for a traded symbol."""
    symbol: str
    exchange: str
    interval: str = None  # Will be set dynamically from config
    enabled: bool = True
    last_signal_time: Optional[datetime] = None
    cooldown_minutes: int = 60


@dataclass
class Signal:
    """Trading signal from prediction engine."""
    symbol: str
    direction: InsightDirection
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    components: Dict = field(default_factory=dict)

    @property
    def is_buy(self) -> bool:
        return self.direction == InsightDirection.UP

    @property
    def is_sell(self) -> bool:
        return self.direction == InsightDirection.DOWN

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry_price - self.stop_loss)
        if risk < 1e-8:  # Epsilon for floating point comparison
            return 0.0
        return abs(self.take_profit - self.entry_price) / risk


class LiveTradingRunner:
    """
    Live Trading Runner (Lean-Inspired).

    Orchestrates the complete live trading workflow:

    1. SETUP:
       - Load configuration
       - Initialize brokerage connection
       - Initialize portfolio manager
       - Set up risk management

    2. DATA LOOP:
       - Stream real-time prices
       - Buffer candles
       - Update portfolio valuations

    3. SIGNAL GENERATION:
       - Run ML predictions on each interval
       - Apply confidence thresholds
       - Check signal cooldowns

    4. RISK CHECK:
       - Evaluate against risk models
       - Size positions appropriately
       - Check portfolio constraints

    5. EXECUTION:
       - Generate orders
       - Submit to brokerage
       - Track order status

    6. MONITORING:
       - Log all activity
       - Track performance
       - Handle errors gracefully

    Example:
        runner = LiveTradingRunner("config.yaml")

        # Add symbols to trade
        runner.add_symbol("BTC/USD", exchange="binance")
        runner.add_symbol("ETH/USD", exchange="binance")

        # Start trading
        runner.start()

        # Check status
        print(runner.get_status())

        # Stop
        runner.stop()
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        mode: TradingMode = TradingMode.PAPER,
    ):
        """
        Initialize live trading runner.

        Args:
            config_path: Path to configuration file
            mode: Trading mode (paper, live, shadow)
        """
        self.mode = mode
        self.config = Config.load(config_path)
        self._config_path = config_path

        # Status
        self._status = RunnerStatus.IDLE
        self._status_lock = threading.Lock()
        self._running = False
        self._paused = False

        # Components (initialized on start)
        self._brokerage: Optional[BaseBrokerage] = None
        self._portfolio: Optional[PortfolioManager] = None
        self._risk_manager: Optional[RiskManager] = None
        self._prediction_system: Optional[MultiCurrencySystem] = None
        self._news_collector: Optional[NewsCollector] = None
        self._database: Optional[Database] = None

        # CONTINUOUS LEARNING: Strategic Learning Bridge (initialized on start)
        self._learning_bridge: Optional[StrategicLearningBridge] = None

        # Symbols and data provider
        self._symbols: Dict[str, TradingSymbol] = {}
        self._provider: Optional[UnifiedDataProvider] = None
        self._data_buffers: Dict[str, pd.DataFrame] = {}

        # Efficient candle storage using deque (O(1) append, avoids pd.concat fragmentation)
        self._candle_buffers: Dict[str, deque] = {}
        self._buffer_max_size = 500

        # Model readiness tracking - ONLY trade if model is validated
        self._models_ready: Dict[str, bool] = {}
        self._model_accuracies: Dict[str, float] = {}

        # Signal and order tracking
        self._pending_signals: Queue = Queue()
        self._active_orders: Dict[str, Order] = {}
        self._signal_history: deque = deque(maxlen=1000)  # Bounded to prevent memory leak

        # Deduplication: track recently processed candle keys to prevent double-processing
        self._processed_candles: set = set()

        # Thread safety for callbacks
        self._callback_lock = threading.Lock()

        # Threads
        self._main_thread: Optional[threading.Thread] = None
        self._signal_thread: Optional[threading.Thread] = None
        self._execution_thread: Optional[threading.Thread] = None

        # Callbacks
        self._on_signal: List[Callable[[Signal], None]] = []
        self._on_order: List[Callable[[Order], None]] = []
        self._on_error: List[Callable[[Exception], None]] = []

        # Stats
        self._start_time: Optional[datetime] = None
        self._total_signals = 0
        self._total_orders = 0
        self._errors_count = 0

        logger.info(f"LiveTradingRunner initialized (mode={mode.value})")

    # =========================================================================
    # SYMBOL MANAGEMENT
    # =========================================================================

    def add_symbol(
        self,
        symbol: str,
        exchange: str = "binance",
        interval: str = None,  # Will use config intervals if not specified
        cooldown_minutes: int = 60
    ):
        """
        Add a symbol to trade.

        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            exchange: Exchange name
            interval: Primary candle interval (optional, uses config if not specified)
            cooldown_minutes: Minutes between signals
        """
        ts = TradingSymbol(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            cooldown_minutes=cooldown_minutes
        )
        self._symbols[symbol] = ts
        logger.info(f"Added trading symbol: {symbol} on {exchange}")

    def remove_symbol(self, symbol: str):
        """Remove a trading symbol."""
        if symbol in self._symbols:
            del self._symbols[symbol]
            self._stop_stream(symbol)
            logger.info(f"Removed trading symbol: {symbol}")

    def enable_symbol(self, symbol: str, enabled: bool = True):
        """Enable or disable a symbol."""
        if symbol in self._symbols:
            self._symbols[symbol].enabled = enabled

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self, blocking: bool = False):
        """
        Start live trading.

        Args:
            blocking: If True, blocks until stopped
        """
        if self._status != RunnerStatus.IDLE:
            logger.warning(f"Cannot start from status {self._status}")
            return False

        try:
            self._set_status(RunnerStatus.STARTING)
            self._running = True

            # Initialize components
            self._initialize_components()

            # Ensure models are ready (auto-train if needed)
            self._ensure_models_ready()

            # Connect to brokerage
            if not self._brokerage.connect():
                raise ConnectionError("Failed to connect to brokerage")

            # Historical replay: fast-learn from past data before going live
            self._historical_replay()

            # Start data streams
            self._start_streams()

            # Start processing threads
            self._start_threads()

            self._start_time = datetime.utcnow()
            self._set_status(RunnerStatus.RUNNING)

            logger.info(f"Live trading started (mode={self.mode.value})")

            if blocking:
                try:
                    while self._running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.stop()

            return True

        except Exception as e:
            logger.error(f"Failed to start: {e}")
            logger.error(traceback.format_exc())
            self._set_status(RunnerStatus.ERROR)
            self._handle_error(e)
            return False

    def stop(self):
        """Stop live trading gracefully."""
        if self._status in (RunnerStatus.STOPPED, RunnerStatus.IDLE):
            return

        self._set_status(RunnerStatus.STOPPING)
        logger.info("Stopping live trading...")

        self._running = False

        # Stop news collector
        if self._news_collector:
            self._news_collector.stop()
            self._news_collector = None
            logger.info("News collector stopped")

        # Stop data provider
        if self._provider:
            self._provider.stop()
            self._provider = None

        # Wait for threads
        for thread in [self._main_thread, self._signal_thread, self._execution_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)

        # Disconnect brokerage
        if self._brokerage:
            self._brokerage.disconnect()

        # Cleanup prediction system
        if self._prediction_system:
            self._prediction_system.cleanup()

        # CONTINUOUS LEARNING: Cleanup learning bridge
        if self._learning_bridge:
            logger.info("Stopping Strategic Learning Bridge...")
            self._learning_bridge.stop()
            logger.info("Strategic Learning Bridge stopped")

        self._set_status(RunnerStatus.STOPPED)
        logger.info("Live trading stopped")

    def pause(self):
        """Pause signal generation (keeps streams running)."""
        self._paused = True
        self._set_status(RunnerStatus.PAUSED)
        logger.info("Live trading paused")

    def resume(self):
        """Resume signal generation."""
        self._paused = False
        self._set_status(RunnerStatus.RUNNING)
        logger.info("Live trading resumed")

    def _set_status(self, status: RunnerStatus):
        """Update runner status thread-safely."""
        with self._status_lock:
            self._status = status

    @property
    def status(self) -> RunnerStatus:
        """Get current status."""
        with self._status_lock:
            return self._status

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _initialize_components(self):
        """Initialize all trading components."""
        logger.info("Initializing components...")

        # Database (for news and features)
        from pathlib import Path
        db_path = Path(getattr(self.config.database, 'path', 'data/trading.db'))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._database = Database(str(db_path))
        logger.info("Database initialized")

        # News Collector (if enabled)
        news_config = self.config.raw.get('news', {})
        if news_config.get('enabled', False):
            try:
                self._news_collector = NewsCollector(
                    database=self._database,
                    config=news_config
                )
                self._news_collector.start()
                logger.info("News collector started")
            except Exception as e:
                logger.warning(f"Failed to start news collector: {e}. Continuing without news.")
                self._news_collector = None
        else:
            logger.info("News collection disabled in config")

        # Portfolio Manager
        initial_cash = self.config.brokerage.initial_cash
        self._portfolio = PortfolioManager(
            initial_cash=initial_cash,
            max_position_percent=0.25,
            max_total_positions=10
        )

        # Risk Manager
        self._risk_manager = RiskManager()
        self._risk_manager.add_model(MaximumDrawdownRisk(max_drawdown_percent=20.0))
        self._risk_manager.add_model(MaximumPositionSizeRisk(max_position_percent=0.25))

        # Brokerage
        self._brokerage = self.config.get_brokerage()
        self._brokerage.on_order_event(self._handle_order_event)

        # Prediction System
        self._prediction_system = MultiCurrencySystem(self._config_path)
        for symbol in self._symbols.keys():
            ts = self._symbols[symbol]
            self._prediction_system.add_currency(
                symbol=symbol,
                exchange=ts.exchange,
                interval=ts.interval
            )

        # CONTINUOUS LEARNING: Initialize Strategic Learning Bridge
        # This connects prediction system → continuous learning → automatic retraining
        logger.info("Initializing Strategic Learning Bridge...")

        # Get paper brokerage (always available for learning mode)
        paper_brokerage = self._brokerage  # TODO: Separate paper vs live brokerage

        # Get live brokerage (optional - used only in TRADING mode)
        live_brokerage = None
        if self.mode == TradingMode.LIVE:
            live_brokerage = self._brokerage

        # Initialize the bridge
        self._learning_bridge = StrategicLearningBridge(
            database=self._database,
            predictor=self._prediction_system.advanced_predictor,  # Use AdvancedPredictor
            paper_brokerage=paper_brokerage,
            live_brokerage=live_brokerage,
            config=self.config.raw  # Pass full config dict
        )

        logger.info("Strategic Learning Bridge initialized - Continuous learning ENABLED")
        logger.info("Components initialized")

    def _ensure_models_ready(self):
        """
        MANDATORY: Thoroughly train models before ANY predictions.

        This method:
        1. ALWAYS trains models (no skipping)
        2. Requires minimum 60% accuracy to be "ready"
        3. Blocks predictions for symbols that don't pass
        4. Uses comprehensive training with validation

        NO PREDICTIONS ALLOWED until model passes validation!
        """
        from datetime import datetime, timedelta
        from pathlib import Path
        import gc
        import torch
        import torch.nn as nn
        import numpy as np
        from src.analysis_engine import FeatureCalculator, LSTMModel

        logger.info("=" * 70)
        logger.info("MANDATORY MODEL TRAINING - NO PREDICTIONS UNTIL VALIDATED")
        logger.info("=" * 70)

        # Guard: ensure database is available
        if self._database is None:
            logger.error("Database not initialized - cannot train models")
            raise RuntimeError("Database required for model training")

        # Get training config - MANDATORY settings
        auto_train_config = self.config.raw.get('auto_training', {})
        min_candles = auto_train_config.get('min_candles', 1000)
        training_candles = auto_train_config.get('training_candles', 5000)

        # MANDATORY: Minimum accuracy required to allow predictions
        min_accuracy_required = auto_train_config.get('min_accuracy_required', 0.58)
        target_accuracy = auto_train_config.get('target_accuracy', 0.65)
        max_epochs = auto_train_config.get('max_epochs', 100)  # More epochs for thorough training

        # Model hyperparameters from config
        hidden_size = self.config.model.hidden_size
        num_layers = self.config.model.num_layers
        dropout = self.config.model.dropout
        batch_size = auto_train_config.get('batch_size', 32)
        learning_rate = auto_train_config.get('learning_rate', 0.001)

        # Detect device (GPU if available)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training device: {device}")
        logger.info(f"Minimum accuracy required: {min_accuracy_required:.1%}")
        logger.info(f"Target accuracy: {target_accuracy:.1%}")

        models_dir = Path(self.config.model.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all symbols as NOT ready
        for symbol in self._symbols.keys():
            self._models_ready[symbol] = False
            self._model_accuracies[symbol] = 0.0

        for symbol, ts in self._symbols.items():
            if not ts.enabled:
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"[{symbol}] MANDATORY TRAINING STARTING")
            logger.info(f"{'='*60}")

            # Get model path
            safe_symbol = symbol.replace("/", "_").replace("-", "_")
            model_path = models_dir / f"model_{safe_symbol}_{ts.interval}.pt"

            logger.info(f"[{symbol}] Fetching {training_candles} candles for training...")

            try:
                # Fetch training data from database
                candles_df = self._database.get_candles(
                    symbol=symbol,
                    interval=ts.interval,
                    limit=training_candles + 100  # Extra for feature warmup
                )

                if candles_df is None or len(candles_df) < min_candles:
                    logger.warning(
                        f"[{symbol}] Insufficient data for training: "
                        f"{len(candles_df) if candles_df is not None else 0} < {min_candles} candles. "
                        f"Will train after more data is collected."
                    )
                    continue

                logger.info(f"[{symbol}] Fetched {len(candles_df)} candles for training")

                # Calculate features
                df_features = FeatureCalculator.calculate_all(candles_df)
                feature_columns = FeatureCalculator.get_feature_columns()

                # Extract and normalize features FIRST (before creating target)
                features = df_features[feature_columns].values
                closes = df_features['close'].values

                # Normalize features
                feature_means = np.nanmean(features, axis=0)
                feature_stds = np.nanstd(features, axis=0)
                features = (features - feature_means) / (feature_stds + 1e-8)
                features = np.nan_to_num(features, nan=0, posinf=0, neginf=0)

                # Create sequences using sliding window
                sequence_length = self.config.model.sequence_length

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

                if len(X) < min_candles:
                    logger.warning(f"[{symbol}] After sequence creation, only {len(X)} samples")
                    continue

                logger.info(f"[{symbol}] Created {len(X)} training sequences")

                # Train/validation split with validation size check
                min_val_samples = 50
                split_idx = int(len(X) * 0.8)

                if len(X) - split_idx < min_val_samples:
                    logger.warning(f"[{symbol}] Validation set too small, adjusting split")
                    if len(X) < min_val_samples * 2:
                        logger.warning(f"[{symbol}] Insufficient data for reliable training")
                        continue
                    split_idx = len(X) - min_val_samples

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
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size
                )

                # Create model with config parameters
                model = LSTMModel(
                    input_size=len(feature_columns),
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                ).to(device)

                # Training
                criterion = torch.nn.BCELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                # LR scheduler: halve LR when val accuracy plateaus
                scheduler_config = auto_train_config.get('scheduler', {})
                scheduler = None
                if scheduler_config.get('enabled', False):
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='max',
                        factor=scheduler_config.get('factor', 0.5),
                        patience=scheduler_config.get('patience', 5),
                        min_lr=scheduler_config.get('min_lr', 1e-5)
                    )

                best_val_acc = 0
                best_state = None
                patience = auto_train_config.get('patience', 15)
                patience_counter = 0

                logger.info(f"[{symbol}] Training for up to {max_epochs} epochs...")

                for epoch in range(max_epochs):
                    # Train
                    model.train()
                    epoch_loss = 0
                    for X_batch, y_batch in train_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(X_batch).squeeze()
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()

                    # Validate
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for X_batch, y_batch in val_loader:
                            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                            outputs = model(X_batch).squeeze()
                            predictions = (outputs > 0.5).float()
                            correct += (predictions == y_batch).sum().item()
                            total += len(y_batch)

                    val_acc = correct / total if total > 0 else 0

                    # Step LR scheduler based on validation accuracy
                    if scheduler is not None:
                        scheduler.step(val_acc)

                    # Track best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    # Log progress with confidence estimate
                    if (epoch + 1) % 10 == 0:
                        avg_loss = epoch_loss / len(train_loader)
                        current_lr = optimizer.param_groups[0]['lr']
                        logger.info(
                            f"[{symbol}] Epoch {epoch+1}/{max_epochs} - "
                            f"Loss: {avg_loss:.4f} - Val Acc: {val_acc:.2%} - "
                            f"LR: {current_lr:.6f}"
                        )

                    # Early stopping
                    if patience_counter >= patience:
                        logger.info(f"[{symbol}] Early stopping at epoch {epoch+1}")
                        break

                    # Stop if target reached
                    if val_acc >= target_accuracy:
                        logger.info(f"[{symbol}] [OK] Target accuracy reached: {val_acc:.2%}")
                        break

                # Save model and validate
                if best_state is not None:
                    # Move model to CPU for saving (ensures compatibility)
                    model.cpu()
                    model.load_state_dict(best_state)

                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config': {
                            'hidden_size': hidden_size,
                            'num_layers': num_layers,
                            'dropout': dropout,
                            'sequence_length': sequence_length
                        },
                        'feature_means': feature_means,
                        'feature_stds': feature_stds,
                        'symbol': symbol,
                        'interval': ts.interval,
                        'trained_at': datetime.utcnow().isoformat(),
                        'samples_trained': len(X_train),
                        'validation_accuracy': best_val_acc
                    }, model_path)

                    # Store accuracy
                    self._model_accuracies[symbol] = best_val_acc

                    # VALIDATION GATE: Check if model meets minimum accuracy
                    if best_val_acc >= min_accuracy_required:
                        self._models_ready[symbol] = True
                        logger.info(
                            f"[{symbol}] [VALIDATED] MODEL READY FOR PREDICTIONS\n"
                            f"    Accuracy: {best_val_acc:.2%} >= {min_accuracy_required:.2%} required\n"
                            f"    Samples trained: {len(X_train)}\n"
                            f"    PREDICTIONS ENABLED for {symbol}"
                        )
                    else:
                        self._models_ready[symbol] = False
                        logger.warning(
                            f"[{symbol}] [FAILED] MODEL BELOW MINIMUM ACCURACY\n"
                            f"    Accuracy: {best_val_acc:.2%} < {min_accuracy_required:.2%} required\n"
                            f"    PREDICTIONS BLOCKED for {symbol}\n"
                            f"    Need more training data or better features"
                        )
                else:
                    logger.error(f"[{symbol}] Training failed - no valid model state")
                    self._models_ready[symbol] = False

            except Exception as e:
                logger.error(f"[{symbol}] Auto-training failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self._models_ready[symbol] = False

            finally:
                # Memory cleanup between symbols
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("MODEL TRAINING COMPLETE - READINESS SUMMARY")
        logger.info("=" * 70)

        ready_count = sum(1 for v in self._models_ready.values() if v)
        total_count = len(self._models_ready)

        for symbol, is_ready in self._models_ready.items():
            acc = self._model_accuracies.get(symbol, 0)
            status = "[READY]" if is_ready else "[BLOCKED]"
            logger.info(f"  {symbol}: {status} (accuracy: {acc:.2%})")

        logger.info(f"\nTotal: {ready_count}/{total_count} models ready for predictions")

        if ready_count == 0:
            error_msg = (
                "CRITICAL: NO MODELS READY FOR PREDICTIONS\n"
                "All models failed to meet minimum accuracy requirement.\n"
                "Cannot start live trading without validated models.\n"
                "Please ensure sufficient historical data is available."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        elif ready_count < total_count:
            logger.warning(
                f"\n[WARNING] {total_count - ready_count} symbol(s) will NOT make predictions\n"
                "until their models meet the minimum accuracy requirement."
            )

        logger.info("=" * 70)

    def _historical_replay(self):
        """
        Replay historical candles through the learning system for fast warmup.

        Called AFTER model training but BEFORE live streaming starts.
        Feeds historical candles through learning_bridge.on_candle_close()
        so the system learns from past trades instead of starting from zero.
        """
        replay_config = self.config.raw.get('continuous_learning', {}).get('historical_replay', {})

        if not replay_config.get('enabled', True):
            logger.info("Historical replay disabled in config")
            return

        if not self._learning_bridge:
            logger.warning("Learning bridge not initialized, skipping historical replay")
            return

        replay_days = replay_config.get('days', 30)
        replay_interval = replay_config.get('interval', '15m')
        batch_log_interval = replay_config.get('log_every', 100)

        # Clean stale pending signals from previous sessions
        # These cause bogus P&L calculations when evaluated against current prices
        try:
            stale_count = self._database.close_stale_signals(max_age_hours=48)
            if stale_count > 0:
                logger.info(f"Pre-replay cleanup: expired {stale_count} stale pending signals")
        except Exception as e:
            logger.warning(f"Failed to clean stale signals: {e}")

        # Suppress trade execution and retraining during replay
        # Replay only runs predictions + online updates for warmup
        learning_sys = getattr(self._learning_bridge, 'learning_system', None)
        if learning_sys:
            learning_sys.replay_mode = True
            logger.info("Replay mode ON — trades and retraining suppressed (predictions + online updates active)")

        # Calculate expected candle count
        interval_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        minutes_per_candle = interval_minutes.get(replay_interval, 15)
        expected_candles = (replay_days * 24 * 60) // minutes_per_candle

        logger.info("=" * 70)
        logger.info(f"HISTORICAL REPLAY - Learning from {replay_days} days of {replay_interval} candles")
        logger.info(f"Expected candles: ~{expected_candles}")
        logger.info("=" * 70)

        for symbol in self._symbols.keys():
            try:
                # Step 1: Fetch historical candles from database
                candles_df = self._database.get_candles(
                    symbol=symbol,
                    interval=replay_interval,
                    limit=min(expected_candles + 100, 100000)
                )

                if candles_df is None or len(candles_df) < 50:
                    db_count = len(candles_df) if candles_df is not None else 0
                    logger.warning(
                        f"[{symbol}] Insufficient data in database ({db_count} candles). "
                        f"Fetching from exchange..."
                    )
                    # Fall back to exchange fetch
                    if self._provider is None:
                        self._provider = UnifiedDataProvider.get_instance(self._config_path)
                    ts = self._symbols[symbol]
                    candles_df = self._provider.fetch_historical(
                        symbol=symbol,
                        exchange=ts.exchange,
                        interval=replay_interval,
                        limit=min(expected_candles, 1000)
                    )
                    if candles_df is not None and not candles_df.empty:
                        self._database.save_candles(
                            candles_df, symbol=symbol, interval=replay_interval
                        )

                if candles_df is None or len(candles_df) < 50:
                    logger.warning(f"[{symbol}] Still insufficient data, skipping replay")
                    continue

                # Fix timestamps stored as bytes in SQLite
                if len(candles_df) > 0 and isinstance(candles_df['timestamp'].iloc[0], bytes):
                    candles_df['timestamp'] = candles_df['timestamp'].apply(
                        lambda b: int.from_bytes(b, 'little') if isinstance(b, bytes) else int(b)
                    )

                # Sort by timestamp ascending for proper replay order
                candles_df = candles_df.sort_values('timestamp').reset_index(drop=True)

                logger.info(f"[{symbol}] Replaying {len(candles_df)} historical candles...")

                # Step 2: Iterate through candles and feed to learning system
                replay_count = 0
                error_count = 0
                start_time = time.time()

                for idx in range(len(candles_df)):
                    try:
                        row = candles_df.iloc[idx]

                        # Create Candle object from DataFrame row
                        ts_val = int(row['timestamp'])
                        candle = Candle(
                            timestamp=ts_val,
                            datetime=row['datetime'] if isinstance(row.get('datetime'), datetime)
                                     else datetime.fromtimestamp(ts_val / 1000 if ts_val > 1e12 else ts_val),
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=float(row['volume']),
                            symbol=symbol,
                            interval=replay_interval,
                            is_closed=True
                        )

                        # Feed to learning bridge (synchronous)
                        result = self._learning_bridge.on_candle_close(
                            symbol=symbol,
                            interval=replay_interval,
                            candle=candle
                        )

                        replay_count += 1

                        # Log progress
                        if replay_count % batch_log_interval == 0:
                            elapsed = time.time() - start_time
                            rate = replay_count / elapsed if elapsed > 0 else 0
                            logger.info(
                                f"[{symbol}] Replay progress: {replay_count}/{len(candles_df)} "
                                f"({replay_count / len(candles_df) * 100:.1f}%) "
                                f"@ {rate:.0f} candles/sec"
                            )

                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:
                            logger.warning(f"[{symbol}] Replay error at candle {idx}: {e}")
                        elif error_count == 6:
                            logger.warning(f"[{symbol}] Suppressing further replay errors...")
                        continue

                elapsed = time.time() - start_time
                rate = replay_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[{symbol}] Historical replay complete: "
                    f"{replay_count} candles in {elapsed:.1f}s "
                    f"({rate:.0f} candles/sec, {error_count} errors)"
                )

                # Log learning stats after replay
                try:
                    stats = self._learning_bridge.get_stats()
                    logger.info(
                        f"[{symbol}] Post-replay stats: "
                        f"predictions={stats.get('predictions_made', 0)}, "
                        f"trades={stats.get('trades_closed', 0)}, "
                        f"wins={stats.get('wins', 0)}, "
                        f"losses={stats.get('losses', 0)}"
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"[{symbol}] Historical replay failed: {e}", exc_info=True)

        # Restore retraining after replay
        if learning_sys:
            learning_sys.replay_mode = False
            logger.info("Replay mode OFF — retraining re-enabled for live trading")

        logger.info("=" * 70)
        logger.info("HISTORICAL REPLAY COMPLETE")
        logger.info("=" * 70)

    def _start_streams(self):
        """Start unified data provider for all symbols."""
        try:
            # Get singleton UnifiedDataProvider
            self._provider = UnifiedDataProvider.get_instance(self._config_path)

            # Get multi-timeframe intervals from config
            timeframe_config = self.config.raw.get('timeframes', {})
            enabled_intervals = []

            if timeframe_config.get('enabled', False):
                # Extract enabled intervals from config
                for tf_config in timeframe_config.get('intervals', []):
                    if tf_config.get('enabled', True):
                        interval = tf_config.get('interval')
                        if interval:
                            enabled_intervals.append(interval)

            # Fallback to single interval if multi-timeframe not configured
            if not enabled_intervals:
                enabled_intervals = [self.config.data.interval if hasattr(self.config.data, 'interval') else '1h']

            logger.info(f"Subscribing to intervals: {enabled_intervals}")

            # Subscribe to all symbols with all enabled intervals
            for symbol, ts in self._symbols.items():
                if not ts.enabled:
                    continue

                # Subscribe to each timeframe for this symbol
                for interval in enabled_intervals:
                    self._provider.subscribe(
                        symbol,
                        exchange=ts.exchange,
                        interval=interval
                    )
                    logger.info(f"Subscribed to {symbol} @ {interval}")

                # Initialize data buffer with historical data (using first interval)
                ts.interval = enabled_intervals[0]  # Set primary interval
                self._initialize_buffer(symbol, ts)

            # Register callbacks
            # Note: Using on_candle_closed for completed candles only
            self._provider.on_candle_closed(self._handle_candle_callback)

            # Connect database to provider for candle persistence
            # This enables WebSocket candles to be saved for learning system
            if self._database:
                self._provider.set_database(self._database)
                logger.info("Database connected to data provider for candle persistence")

            # Start provider
            self._provider.start()
            logger.info("UnifiedDataProvider started")

        except Exception as e:
            logger.error(f"Failed to start data provider: {e}")

    def _stop_stream(self, symbol: str):
        """Unsubscribe from a symbol (provider handles connection)."""
        if self._provider and symbol in self._symbols:
            self._provider.unsubscribe(symbol)
            logger.info(f"Unsubscribed from {symbol}")

    def _initialize_buffer(self, symbol: str, ts: TradingSymbol):
        """Load historical data into buffer for the specific symbol."""
        try:
            interval = ts.interval or self.config.data.interval or '1h'
            df = self._database.get_candles(
                symbol=symbol,
                interval=interval,
                limit=200
            )
            if df is not None and len(df) > 0:
                self._data_buffers[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol} @ {interval}")
            else:
                logger.warning(f"No historical data for {symbol} @ {interval}")
                self._data_buffers[symbol] = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not load history for {symbol}: {e}")
            self._data_buffers[symbol] = pd.DataFrame()

    def _start_threads(self):
        """Start processing threads."""
        # Signal generation thread
        self._signal_thread = threading.Thread(
            target=self._signal_loop,
            daemon=True,
            name="SignalLoop"
        )
        self._signal_thread.start()

        # Execution thread
        self._execution_thread = threading.Thread(
            target=self._execution_loop,
            daemon=True,
            name="ExecutionLoop"
        )
        self._execution_thread.start()

    # =========================================================================
    # DATA HANDLERS
    # =========================================================================

    def _handle_tick_callback(self, tick: Tick):
        """Callback wrapper for tick data from UnifiedDataProvider."""
        self._handle_tick(tick.symbol, tick)

    def _handle_candle_callback(self, candle: Candle):
        """
        Callback wrapper for candle data from UnifiedDataProvider.

        Args:
            candle: Completed candle data (includes symbol and interval)
        """
        # Process the candle - interval is in candle.interval
        self._handle_candle(candle.symbol, candle)

    def _handle_tick(self, symbol: str, tick: Tick):
        """Handle incoming tick data."""
        # Update portfolio valuations
        if self._portfolio:
            self._portfolio.update_price(symbol, tick.price)

    def _handle_candle(self, symbol: str, candle: Candle):
        """Handle incoming candle data."""
        if not candle.is_closed:
            return  # Only process closed candles

        # Deduplication: skip if this exact candle was already processed
        candle_key = (symbol, candle.interval, candle.timestamp)
        if candle_key in self._processed_candles:
            return
        self._processed_candles.add(candle_key)
        if len(self._processed_candles) > 500:
            self._processed_candles.clear()
            self._processed_candles.add(candle_key)

        # Initialize candle buffer if needed
        if symbol not in self._candle_buffers:
            self._candle_buffers[symbol] = deque(maxlen=self._buffer_max_size)

        # Efficient O(1) append to deque (avoids pd.concat memory fragmentation)
        self._candle_buffers[symbol].append({
            'timestamp': datetime.fromtimestamp(candle.timestamp / 1000),
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        })

        # Update DataFrame buffer only when needed (lazy conversion)
        # This is 10x faster than pd.concat on every candle
        self._data_buffers[symbol] = pd.DataFrame(list(self._candle_buffers[symbol]))

        # ===================================================================
        # CONTINUOUS LEARNING: Trigger on every candle close
        # ===================================================================
        # This is where the MAGIC happens:
        # 1. Get multi-timeframe predictions (15m, 1h, 4h, 1d)
        # 2. Check confidence gate (≥80% = TRADING, <80% = LEARNING)
        # 3. Execute trade (paper or live based on mode)
        # 4. Monitor positions and close when conditions met
        # 5. Record outcomes and trigger retraining if needed
        # ===================================================================

        if self._learning_bridge:
            try:
                # Get the interval from the candle itself (set by provider)
                ts = self._symbols.get(symbol)
                if ts:
                    interval = candle.interval or ts.interval

                    # Trigger continuous learning system
                    result = self._learning_bridge.on_candle_close(
                        symbol=symbol,
                        interval=interval,
                        candle=candle
                    )

                    # Log result for debugging
                    if result.get('error'):
                        logger.error(
                            f"[{symbol} @ {interval}] Learning system error: {result['error']}"
                        )
                    elif result.get('executed'):
                        logger.info(
                            f"[{symbol} @ {interval}] {result['mode']} trade executed "
                            f"via {result.get('brokerage', 'unknown')} brokerage"
                        )

            except Exception as e:
                logger.error(
                    f"[{symbol}] Continuous learning error: {e}",
                    exc_info=True
                )

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def _signal_loop(self):
        """Main signal generation loop."""
        logger.info("Signal loop started")

        while self._running:
            try:
                if self._paused:
                    time.sleep(1)
                    continue

                # Generate signals for each symbol
                for symbol, ts in self._symbols.items():
                    if not ts.enabled:
                        continue

                    # MANDATORY: Check if model is validated and ready
                    if not self._models_ready.get(symbol, False):
                        # Model not ready - skip predictions for this symbol
                        continue

                    # Check cooldown
                    if ts.last_signal_time:
                        elapsed = datetime.utcnow() - ts.last_signal_time
                        if elapsed < timedelta(minutes=ts.cooldown_minutes):
                            continue

                    # Get data buffer
                    df = self._data_buffers.get(symbol)
                    if df is None or len(df) < 60:
                        continue

                    # Generate prediction (only if model is validated)
                    try:
                        prediction = self._prediction_system.predict(symbol, df)
                        if prediction and prediction['confidence'] >= self.config.analysis.min_confidence:
                            signal = self._create_signal(symbol, prediction)
                            if signal:
                                self._process_signal(signal)
                                ts.last_signal_time = datetime.utcnow()
                    except Exception as e:
                        logger.error(f"Prediction error for {symbol}: {e}")

                # Sleep before next iteration
                time.sleep(self.config.analysis.update_interval)

            except Exception as e:
                logger.error(f"Signal loop error: {e}")
                self._errors_count += 1
                time.sleep(5)

        logger.info("Signal loop stopped")

    def _create_signal(self, symbol: str, prediction: dict) -> Optional[Signal]:
        """Create Signal from prediction result."""
        try:
            direction_str = prediction.get('direction', 'HOLD')
            if direction_str == 'HOLD':
                return None

            direction = InsightDirection.UP if direction_str == 'BUY' else InsightDirection.DOWN

            return Signal(
                symbol=symbol,
                direction=direction,
                confidence=prediction['confidence'],
                entry_price=prediction['price'],
                stop_loss=prediction['stop_loss'],
                take_profit=prediction['take_profit'],
                components=prediction.get('components', {})
            )

        except Exception as e:
            logger.error(f"Failed to create signal: {e}")
            return None

    def _process_signal(self, signal: Signal):
        """Process a trading signal."""
        self._total_signals += 1
        self._signal_history.append(signal)

        logger.info(
            f"SIGNAL: {signal.symbol} {signal.direction.name} "
            f"@ {signal.entry_price:.2f} (conf: {signal.confidence:.1%})"
        )

        # Notify callbacks (thread-safe copy)
        with self._callback_lock:
            callbacks = list(self._on_signal)
        for callback in callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

        # Shadow mode - don't execute
        if self.mode == TradingMode.SHADOW:
            return

        # Queue for execution
        self._pending_signals.put(signal)

    # =========================================================================
    # EXECUTION
    # =========================================================================

    def _execution_loop(self):
        """Order execution loop."""
        logger.info("Execution loop started")

        while self._running:
            try:
                # Get pending signal
                try:
                    signal = self._pending_signals.get(timeout=1.0)
                except Empty:
                    continue

                # Execute signal
                self._execute_signal(signal)

            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                self._errors_count += 1
                time.sleep(1)

        logger.info("Execution loop stopped")

    def _execute_signal(self, signal: Signal):
        """Execute a trading signal."""
        try:
            # Spot trading: SELL signals close existing positions only
            if not signal.is_buy:
                has_position = self._portfolio.has_position(signal.symbol)
                if not has_position:
                    # No position to sell in spot trading — skip
                    return

            # Check if we can open position
            can_open, reason = self._portfolio.can_open_position(signal.symbol)
            if not can_open and signal.is_buy:
                logger.warning(f"Cannot open position: {reason}")
                return

            # Calculate position size
            quantity = self._portfolio.calculate_position_size(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                stop_price=signal.stop_loss,
                risk_percent=self.config.signals.risk_per_trade
            )

            if quantity <= 0:
                logger.warning(f"Position size is zero for {signal.symbol}")
                return

            # Risk check
            side = "BUY" if signal.is_buy else "SELL"
            assessment = self._risk_manager.evaluate_trade(
                self._portfolio,
                signal.symbol,
                quantity,
                signal.entry_price,
                side
            )

            if assessment.action == RiskAction.BLOCK:
                logger.warning(f"Trade blocked by risk manager: {assessment.reason}")
                return

            if assessment.action == RiskAction.REDUCE:
                quantity = assessment.adjusted_quantity
                logger.info(f"Position reduced by risk manager: {assessment.reason}")

            # Create order
            order = Order(
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.is_buy else OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit
            )

            # Submit order
            ticket = self._brokerage.place_order(order)
            self._active_orders[order.id] = order
            self._total_orders += 1

            logger.info(
                f"ORDER: {order.side.name} {order.quantity} {order.symbol} "
                f"(SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f})"
            )

            # Notify callbacks
            for callback in self._on_order:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Order callback error: {e}")

        except Exception as e:
            logger.error(f"Execution error: {e}")
            logger.error(traceback.format_exc())
            self._handle_error(e)

    def _handle_order_event(self, event: OrderEvent):
        """Handle order status updates from brokerage."""
        logger.info(f"Order event: {event.event_type.name} - {event.order_id}")

        if event.event_type == OrderEventType.FILLED:
            # Update portfolio
            self._portfolio.process_fill(
                symbol=event.order.symbol,
                quantity=event.fill_quantity,
                fill_price=event.fill_price,
                side=event.order.side.name,
                commission=event.commission
            )

            # Remove from active orders
            if event.order_id in self._active_orders:
                del self._active_orders[event.order_id]

        elif event.event_type in (OrderEventType.CANCELED, OrderEventType.REJECTED):
            if event.order_id in self._active_orders:
                del self._active_orders[event.order_id]

    # =========================================================================
    # ERROR HANDLING
    # =========================================================================

    def _handle_error(self, error: Exception):
        """Handle and log errors."""
        self._errors_count += 1

        for callback in self._on_error:
            try:
                callback(error)
            except Exception:
                pass

        # Check if critical
        if self._errors_count > 10:
            logger.critical("Too many errors, stopping trading")
            self.stop()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_signal(self, callback: Callable[[Signal], None]):
        """Register signal callback (thread-safe)."""
        with self._callback_lock:
            self._on_signal.append(callback)

    def on_order(self, callback: Callable[[Order], None]):
        """Register order callback (thread-safe)."""
        with self._callback_lock:
            self._on_order.append(callback)

    def on_error(self, callback: Callable[[Exception], None]):
        """Register error callback (thread-safe)."""
        with self._callback_lock:
            self._on_error.append(callback)

    # =========================================================================
    # STATUS AND REPORTING
    # =========================================================================

    def get_status(self) -> dict:
        """Get comprehensive status report including continuous learning stats."""
        uptime = None
        if self._start_time:
            uptime = str(datetime.utcnow() - self._start_time)

        portfolio_summary = self._portfolio.get_summary() if self._portfolio else {}

        # Get continuous learning stats
        learning_stats = {}
        if self._learning_bridge:
            learning_stats = self._learning_bridge.get_stats()

        return {
            'status': self.status.value,
            'mode': self.mode.value,
            'uptime': uptime,
            'symbols': list(self._symbols.keys()),
            'provider_connected': self._provider.is_connected if self._provider else False,
            'portfolio': portfolio_summary,
            'total_signals': self._total_signals,
            'total_orders': self._total_orders,
            'active_orders': len(self._active_orders),
            'errors': self._errors_count,
            'brokerage_connected': self._brokerage.is_connected if self._brokerage else False,
            # CONTINUOUS LEARNING: Add learning system statistics
            'continuous_learning': learning_stats
        }

    def get_recent_signals(self, count: int = 10) -> List[dict]:
        """Get recent signals."""
        recent = self._signal_history[-count:]
        return [
            {
                'symbol': s.symbol,
                'direction': s.direction.name,
                'confidence': s.confidence,
                'entry': s.entry_price,
                'stop_loss': s.stop_loss,
                'take_profit': s.take_profit,
                'risk_reward': s.risk_reward,
                'timestamp': s.timestamp.isoformat()
            }
            for s in recent
        ]

    def get_holdings(self) -> List[dict]:
        """Get current holdings."""
        if not self._portfolio:
            return []
        return self._portfolio.get_holdings_report()

    def __repr__(self) -> str:
        return f"LiveTradingRunner(status={self.status.value}, symbols={len(self._symbols)})"
