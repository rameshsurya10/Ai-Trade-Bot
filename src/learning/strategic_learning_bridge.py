"""
Strategic Learning Bridge
==========================

BRIDGES live trading runner with continuous learning system.

This is the MISSING PIECE that connects:
1. LiveTradingRunner (live trading)
2. ContinuousLearningSystem (learning orchestrator)
3. MultiTimeframeModelManager (model management)
4. OutcomeTracker (performance tracking)
5. RetrainingEngine (automatic retraining)

WORKFLOW (Complete End-to-End):
================================

1. CANDLE ARRIVES (from WebSocket)
   ↓
2. STRATEGIC LEARNING TRIGGER
   - Check if this timeframe needs prediction
   - Fetch historical data (1 year for long-term patterns)
   - Calculate features (32 technical + 7 sentiment = 39)
   ↓
3. MULTI-TIMEFRAME PREDICTION
   - Get predictions from ALL timeframes (15m, 1h, 4h, 1d)
   - Each model analyzes patterns at its scale
   - Aggregate using weighted voting
   ↓
4. CONFIDENCE GATING
   - Check if confidence ≥ 80% (TRADING mode)
   - Or < 80% (LEARNING mode - paper trading only)
   ↓
5. TRADE EXECUTION
   - TRADING mode: Real trades (if live_brokerage configured)
   - LEARNING mode: Paper trades (always)
   - Record signal to database
   ↓
6. POSITION MONITORING
   - Track open positions
   - Check stop-loss, take-profit, time limits
   - Close when conditions met
   ↓
7. OUTCOME TRACKING
   - Record win/loss
   - Add to experience replay buffer
   - Check retraining triggers
   ↓
8. AUTOMATIC RETRAINING (if triggered)
   - Fetch recent 5,000 candles
   - Mix with experience replay (30% ratio)
   - Train with EWC (prevent forgetting)
   - Validate until confidence ≥ 80%
   - Save improved model
   ↓
9. MODE TRANSITION
   - LEARNING → TRADING (when confidence ≥ 80%)
   - TRADING → LEARNING (when accuracy drops)
   ↓
10. REPEAT FOR EVERY CANDLE

This file makes your system TRULY INTELLIGENT by:
- Learning from EVERY trade outcome
- Automatically adapting to market changes
- Using 1-year historical patterns for strategic decisions
- Preventing catastrophic forgetting with EWC
- Multi-timeframe analysis (15m, 1h, 4h, 1d)
"""

import logging
import threading
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque

from src.core.database import Database
from src.learning.continuous_learner import ContinuousLearningSystem
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.brokerages.base import BaseBrokerage
from src.data.provider import Candle

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Tracks an active trade for outcome recording."""
    signal_id: int
    symbol: str
    interval: str
    entry_price: float
    entry_time: datetime
    direction: str  # 'BUY' or 'SELL'
    confidence: float
    stop_loss: float
    take_profit: float
    features: any  # numpy array
    regime: str
    is_paper: bool


class StrategicLearningBridge:
    """
    Bridges LiveTradingRunner with Continuous Learning System.

    This class is the GLUE that makes everything work together.

    Responsibilities:
    1. Trigger predictions after candle close
    2. Execute trades via brokerage
    3. Monitor open positions
    4. Record outcomes when trades complete
    5. Trigger automatic retraining when needed
    6. Transition between LEARNING and TRADING modes

    Thread-safe: All operations are thread-safe
    """

    def __init__(
        self,
        database: Database,
        predictor: any,  # UnbreakablePredictor or AdvancedPredictor
        paper_brokerage: BaseBrokerage,
        live_brokerage: Optional[BaseBrokerage] = None,
        config: dict = None
    ):
        """
        Initialize Strategic Learning Bridge.

        Args:
            database: Database instance
            predictor: Prediction system (UnbreakablePredictor, AdvancedPredictor, etc.)
            paper_brokerage: Paper trading brokerage
            live_brokerage: Live trading brokerage (optional)
            config: Configuration dict from config.yaml
        """
        self.database = database
        self.predictor = predictor
        self.paper_brokerage = paper_brokerage
        self.live_brokerage = live_brokerage
        self.config = config or {}

        # Initialize Continuous Learning System
        logger.info("Initializing Continuous Learning System...")
        self.learning_system = ContinuousLearningSystem(
            predictor=predictor,
            database=database,
            paper_brokerage=paper_brokerage,
            live_brokerage=live_brokerage,
            config=config
        )

        # Track open trades (for outcome recording)
        self._open_trades: Dict[int, TradeRecord] = {}  # signal_id -> TradeRecord
        self._trades_lock = threading.Lock()
        self._mode_lock = threading.Lock()  # Thread safety for mode tracking

        # Statistics
        self._stats = {
            'candles_processed': 0,
            'predictions_made': 0,
            'trades_opened': 0,
            'trades_closed': 0,
            'wins': 0,
            'losses': 0,
            'retrainings_triggered': 0,
            'learning_mode_time': 0.0,
            'trading_mode_time': 0.0
        }
        self._stats_lock = threading.Lock()

        # Mode tracking per symbol
        self._current_modes: Dict[str, str] = {}  # symbol -> 'LEARNING' or 'TRADING'
        self._mode_transitions: deque = deque(maxlen=1000)  # Bounded to prevent memory leak

        logger.info(
            "Strategic Learning Bridge initialized\n"
            f"  Paper brokerage: {paper_brokerage.__class__.__name__}\n"
            f"  Live brokerage: {live_brokerage.__class__.__name__ if live_brokerage else 'None'}\n"
            f"  Enabled timeframes: {self.learning_system.enabled_intervals}"
        )

    def on_candle_close(
        self,
        symbol: str,
        interval: str,
        candle: Candle
    ) -> dict:
        """
        MAIN ENTRY POINT - Called when a candle completes.

        This is where ALL the magic happens:
        1. Get multi-timeframe predictions
        2. Aggregate signals
        3. Check confidence gate
        4. Execute trade (paper or live)
        5. Monitor open positions
        6. Record outcomes
        7. Trigger retraining if needed

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            interval: Timeframe (e.g., '1h', '4h', '1d')
            candle: Closed candle data

        Returns:
            dict: Result with prediction, execution status, mode, etc.
        """
        with self._stats_lock:
            self._stats['candles_processed'] += 1

        logger.info(f"[{symbol} @ {interval}] 📊 Candle closed at {datetime.fromtimestamp(candle.timestamp / 1000)}")

        try:
            # 1. TRIGGER CONTINUOUS LEARNING SYSTEM
            result = self.learning_system.on_candle_closed(
                symbol=symbol,
                interval=interval,
                candle=candle,
                data=None  # Will fetch from database
            )

            # 2. UPDATE STATISTICS
            with self._stats_lock:
                self._stats['predictions_made'] += 1

                # Track mode time
                mode = result.get('mode', 'LEARNING')
                if mode == 'TRADING':
                    self._stats['trading_mode_time'] += 1
                else:
                    self._stats['learning_mode_time'] += 1

            # 3. UPDATE MODE TRACKING (thread-safe)
            with self._mode_lock:
                current_mode = self._current_modes.get(symbol)
                new_mode = result.get('mode')

                if current_mode != new_mode:
                    self._record_mode_transition(symbol, interval, current_mode, new_mode, result)
                    self._current_modes[symbol] = new_mode

            # 4. TRACK TRADE IF EXECUTED
            if result.get('executed') and result.get('signal_id'):
                self._track_new_trade(
                    signal_id=result['signal_id'],
                    symbol=symbol,
                    interval=interval,
                    prediction=result['aggregated_signal'],
                    is_paper=(result['mode'] == 'LEARNING')
                )

            # 5. CHECK FOR COMPLETED TRADES
            self._check_and_close_trades(symbol, candle)

            # Log result
            if result.get('executed'):
                aggregated = result.get('aggregated_signal', {})
                logger.info(
                    f"[{symbol}] {result['mode']} MODE: "
                    f"{aggregated.get('direction', 'UNKNOWN')} @ "
                    f"{aggregated.get('confidence', 0):.2%} "
                    f"({result.get('brokerage', 'unknown')} brokerage)"
                )

            return result

        except Exception as e:
            logger.error(
                f"[{symbol} @ {interval}] Error in on_candle_close: {e}",
                exc_info=True
            )
            return {
                'error': str(e),
                'mode': 'LEARNING',
                'executed': False
            }

    def _track_new_trade(
        self,
        signal_id: int,
        symbol: str,
        interval: str,
        prediction: dict,
        is_paper: bool
    ):
        """
        Track a newly opened trade for outcome recording.

        Args:
            signal_id: Database signal ID
            symbol: Trading pair
            interval: Timeframe
            prediction: Prediction dict from aggregated signal
            is_paper: True if paper trade
        """
        try:
            # Extract trade details from prediction
            direction = prediction.get('direction', 'BUY')
            confidence = prediction.get('confidence', 0.5)
            entry_price = prediction.get('entry_price', 0.0)
            stop_loss = prediction.get('stop_loss', 0.0)
            take_profit = prediction.get('take_profit', 0.0)
            regime = prediction.get('regime', 'NORMAL')

            # Create trade record
            trade = TradeRecord(
                signal_id=signal_id,
                symbol=symbol,
                interval=interval,
                entry_price=entry_price,
                entry_time=datetime.utcnow(),
                direction=direction,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features=None,  # TODO: Extract from prediction if available
                regime=regime,
                is_paper=is_paper
            )

            # Store in open trades
            with self._trades_lock:
                self._open_trades[signal_id] = trade

            with self._stats_lock:
                self._stats['trades_opened'] += 1

            logger.info(
                f"[{symbol}] Tracking new trade: "
                f"{direction} @ {entry_price:.2f} "
                f"(SL: {stop_loss:.2f}, TP: {take_profit:.2f}) "
                f"{'[PAPER]' if is_paper else '[LIVE]'}"
            )

        except Exception as e:
            logger.error(f"Failed to track new trade: {e}", exc_info=True)

    def _check_and_close_trades(self, symbol: str, candle: Candle):
        """
        Check open trades for exit conditions and record outcomes.

        Exit conditions (in priority order):
        1. Stop-loss hit
        2. Take-profit hit
        3. Max holding period
        4. Opposite signal

        Args:
            symbol: Trading pair
            candle: Latest candle
        """
        try:
            current_price = candle.close
            current_time = datetime.fromtimestamp(candle.timestamp / 1000)

            # Get config parameters
            exit_config = self.config.get('continuous_learning', {}).get('exit_logic', {})
            stop_loss_pct = exit_config.get('stop_loss_pct', 2.0)
            take_profit_pct = exit_config.get('take_profit_pct', 4.0)
            max_holding_hours = exit_config.get('max_holding_hours', 24)

            # Check all open trades for this symbol
            trades_to_close = []

            with self._trades_lock:
                for signal_id, trade in list(self._open_trades.items()):
                    if trade.symbol != symbol:
                        continue

                    # VALIDATION: Sanity check prices
                    if trade.entry_price <= 0 or current_price <= 0:
                        logger.error(
                            f"[{symbol}] Invalid prices detected: "
                            f"entry={trade.entry_price}, current={current_price}. Skipping trade."
                        )
                        continue

                    # Calculate P&L
                    if trade.direction == 'BUY':
                        pnl_pct = ((current_price - trade.entry_price) / trade.entry_price) * 100
                    else:  # SELL
                        pnl_pct = ((trade.entry_price - current_price) / trade.entry_price) * 100

                    should_close = False
                    close_reason = None

                    # 1. STOP-LOSS CHECK
                    if pnl_pct <= -stop_loss_pct:
                        should_close = True
                        close_reason = "stop_loss"
                        logger.info(
                            f"[{symbol}] 🛑 Stop-loss hit: {pnl_pct:.2f}% ≤ -{stop_loss_pct:.2f}%"
                        )

                    # 2. TAKE-PROFIT CHECK
                    elif pnl_pct >= take_profit_pct:
                        should_close = True
                        close_reason = "take_profit"
                        logger.info(
                            f"[{symbol}] ✅ Take-profit hit: {pnl_pct:.2f}% ≥ {take_profit_pct:.2f}%"
                        )

                    # 3. MAX HOLDING PERIOD CHECK
                    elif (current_time - trade.entry_time) > timedelta(hours=max_holding_hours):
                        should_close = True
                        close_reason = "max_holding_period"
                        logger.info(
                            f"[{symbol}] ⏰ Max holding period reached: "
                            f"{(current_time - trade.entry_time).total_seconds() / 3600:.1f}h > {max_holding_hours}h"
                        )

                    if should_close:
                        trades_to_close.append((signal_id, trade, current_price, close_reason))

            # Close trades (outside lock to prevent deadlock)
            for signal_id, trade, exit_price, close_reason in trades_to_close:
                self._close_trade_and_record_outcome(
                    trade=trade,
                    exit_price=exit_price,
                    close_reason=close_reason
                )

                # Remove from open trades
                with self._trades_lock:
                    if signal_id in self._open_trades:
                        del self._open_trades[signal_id]

                with self._stats_lock:
                    self._stats['trades_closed'] += 1

        except Exception as e:
            logger.error(f"Error checking trades: {e}", exc_info=True)

    def _close_trade_and_record_outcome(
        self,
        trade: TradeRecord,
        exit_price: float,
        close_reason: str
    ):
        """
        Close trade and record outcome to continuous learning system.

        This triggers:
        1. Outcome recording in database
        2. Experience replay buffer update
        3. Retraining trigger check
        4. Performance statistics update

        Args:
            trade: TradeRecord instance
            exit_price: Exit price
            close_reason: Reason for closing ('stop_loss', 'take_profit', etc.)
        """
        try:
            # Record outcome via OutcomeTracker
            outcome = self.learning_system.outcome_tracker.record_outcome(
                signal_id=trade.signal_id,
                symbol=trade.symbol,
                interval=trade.interval,
                entry_price=trade.entry_price,
                exit_price=exit_price,
                predicted_direction=trade.direction,
                confidence=trade.confidence,
                features=trade.features,
                regime=trade.regime,
                is_paper_trade=trade.is_paper
            )

            # Update statistics
            with self._stats_lock:
                if outcome['was_correct']:
                    self._stats['wins'] += 1
                else:
                    self._stats['losses'] += 1

                # Check if retraining was triggered
                if outcome.get('should_retrain'):
                    self._stats['retrainings_triggered'] += 1

            # Log outcome
            logger.info(
                f"[{trade.symbol}] Trade closed: "
                f"{'✓ WIN' if outcome['was_correct'] else '✗ LOSS'} "
                f"({outcome['pnl_percent']:+.2f}%) "
                f"- Reason: {close_reason} "
                f"{'[PAPER]' if trade.is_paper else '[LIVE]'}"
            )

            # If retraining triggered, log it
            if outcome.get('should_retrain'):
                logger.info(
                    f"[{trade.symbol}] 🔄 Retraining triggered: {outcome['trigger_reason']}"
                )

        except Exception as e:
            logger.error(f"Failed to record outcome: {e}", exc_info=True)

    def _record_mode_transition(
        self,
        symbol: str,
        interval: str,
        old_mode: Optional[str],
        new_mode: str,
        result: dict
    ):
        """
        Record mode transition for analytics.

        Args:
            symbol: Trading pair
            interval: Timeframe
            old_mode: Previous mode ('LEARNING' or 'TRADING')
            new_mode: New mode
            result: Result dict from continuous learning system
        """
        transition = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'interval': interval,
            'from_mode': old_mode,
            'to_mode': new_mode,
            'confidence': result.get('aggregated_signal', {}).get('confidence', 0.0),
            'reason': result.get('reason', 'unknown')
        }

        self._mode_transitions.append(transition)

        # Log transition
        logger.info(
            f"[{symbol} @ {interval}] Mode transition: "
            f"{old_mode or 'INITIAL'} → {new_mode} "
            f"(reason: {transition['reason']})"
        )

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        with self._stats_lock:
            stats = self._stats.copy()

        # Add derived metrics
        total_trades = stats['trades_closed']
        if total_trades > 0:
            stats['win_rate'] = stats['wins'] / total_trades
        else:
            stats['win_rate'] = 0.0

        # Add learning system stats
        stats['learning_system'] = self.learning_system.get_stats()

        # Add open trades count
        with self._trades_lock:
            stats['open_trades'] = len(self._open_trades)

        # Add mode distribution
        total_candles = stats['learning_mode_time'] + stats['trading_mode_time']
        if total_candles > 0:
            stats['learning_mode_pct'] = (stats['learning_mode_time'] / total_candles) * 100
            stats['trading_mode_pct'] = (stats['trading_mode_time'] / total_candles) * 100
        else:
            stats['learning_mode_pct'] = 0.0
            stats['trading_mode_pct'] = 0.0

        return stats

    def get_open_trades(self) -> List[dict]:
        """Get list of currently open trades."""
        with self._trades_lock:
            return [
                {
                    'signal_id': trade.signal_id,
                    'symbol': trade.symbol,
                    'interval': trade.interval,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'entry_time': trade.entry_time.isoformat(),
                    'confidence': trade.confidence,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'is_paper': trade.is_paper,
                    'age_hours': (datetime.utcnow() - trade.entry_time).total_seconds() / 3600
                }
                for trade in self._open_trades.values()
            ]

    def get_mode_transitions(self, symbol: str = None, limit: int = 10) -> List[dict]:
        """
        Get recent mode transitions.

        Args:
            symbol: Filter by symbol (None = all)
            limit: Max number to return

        Returns:
            List of transition dicts
        """
        transitions = self._mode_transitions

        if symbol:
            transitions = [t for t in transitions if t['symbol'] == symbol]

        return transitions[-limit:]

    def stop(self):
        """Stop the bridge and underlying learning system."""
        logger.info("Stopping Strategic Learning Bridge...")

        # Stop continuous learning system
        self.learning_system.stop()

        # Log final stats
        stats = self.get_stats()
        logger.info(
            f"Strategic Learning Bridge stopped\n"
            f"  Total candles processed: {stats['candles_processed']}\n"
            f"  Total predictions: {stats['predictions_made']}\n"
            f"  Total trades: {stats['trades_closed']}\n"
            f"  Win rate: {stats['win_rate']:.2%}\n"
            f"  Retrainings triggered: {stats['retrainings_triggered']}\n"
            f"  Learning mode: {stats['learning_mode_pct']:.1f}%\n"
            f"  Trading mode: {stats['trading_mode_pct']:.1f}%"
        )
