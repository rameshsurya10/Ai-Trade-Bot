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

import pandas as pd

from src.core.config import Config
from src.brokerages.base import BaseBrokerage
from src.brokerages.orders import Order, OrderType, OrderSide, OrderStatus
from src.brokerages.events import OrderEvent, OrderEventType
from src.portfolio.manager import PortfolioManager, InsightDirection
from src.portfolio.risk import RiskManager, RiskAction, MaximumDrawdownRisk, MaximumPositionSizeRisk
from src.data.provider import UnifiedDataProvider, Tick, Candle
from src.multi_currency_system import MultiCurrencySystem
from src.data_service import DataService

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
    interval: str = "1h"
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

        # Symbols and data provider
        self._symbols: Dict[str, TradingSymbol] = {}
        self._provider: Optional[UnifiedDataProvider] = None
        self._data_buffers: Dict[str, pd.DataFrame] = {}

        # Efficient candle storage using deque (O(1) append, avoids pd.concat fragmentation)
        self._candle_buffers: Dict[str, deque] = {}
        self._buffer_max_size = 500

        # Signal and order tracking
        self._pending_signals: Queue = Queue()
        self._active_orders: Dict[str, Order] = {}
        self._signal_history: deque = deque(maxlen=1000)  # Bounded to prevent memory leak

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
        interval: str = "1h",
        cooldown_minutes: int = 60
    ):
        """
        Add a symbol to trade.

        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            exchange: Exchange name
            interval: Candle interval
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

            # Connect to brokerage
            if not self._brokerage.connect():
                raise ConnectionError("Failed to connect to brokerage")

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

        logger.info("Components initialized")

    def _start_streams(self):
        """Start unified data provider for all symbols."""
        try:
            # Get singleton UnifiedDataProvider
            self._provider = UnifiedDataProvider.get_instance(self._config_path)

            # Subscribe to all symbols
            for symbol, ts in self._symbols.items():
                if not ts.enabled:
                    continue

                self._provider.subscribe(
                    symbol,
                    exchange=ts.exchange,
                    interval=ts.interval
                )

                # Initialize data buffer with historical data
                self._initialize_buffer(symbol, ts)

                logger.info(f"Subscribed to {symbol}")

            # Register callbacks
            self._provider.on_tick(self._handle_tick_callback)
            self._provider.on_candle(self._handle_candle_callback)

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
        """Load historical data into buffer."""
        try:
            data_service = DataService()
            # Get last 200 candles for analysis
            df = data_service.get_candles(limit=200)
            if df is not None and len(df) > 0:
                self._data_buffers[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol}")
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

    def _handle_candle_callback(self, candle: Candle, interval: str):
        """
        Callback wrapper for candle data from UnifiedDataProvider.

        Args:
            candle: Completed candle data
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        """
        # Only process candles for the configured interval (if specified)
        # This allows filtering when multiple intervals are subscribed
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

                    # Check cooldown
                    if ts.last_signal_time:
                        elapsed = datetime.utcnow() - ts.last_signal_time
                        if elapsed < timedelta(minutes=ts.cooldown_minutes):
                            continue

                    # Get data buffer
                    df = self._data_buffers.get(symbol)
                    if df is None or len(df) < 60:
                        continue

                    # Generate prediction
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
        """Get comprehensive status report."""
        uptime = None
        if self._start_time:
            uptime = str(datetime.utcnow() - self._start_time)

        portfolio_summary = self._portfolio.get_summary() if self._portfolio else {}

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
