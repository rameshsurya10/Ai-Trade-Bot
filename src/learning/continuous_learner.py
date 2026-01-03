"""
Continuous Learning System
===========================

Main orchestrator for continuous learning trading system.

Workflow:
1. Get predictions from all enabled timeframes
2. Aggregate signals using configured method
3. Check confidence gate
4. Execute trade (live or paper based on mode)
5. Track outcomes when trades close
6. Trigger retraining if needed
7. Perform online learning updates

Features:
- 80% confidence threshold for live trading
- Immediate retraining on losses
- Multi-timeframe signal aggregation
- Paper trading during learning mode
- Event-driven architecture (webhook-based)
"""

import logging
import threading
import time
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from src.core.database import Database
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.multi_timeframe.aggregator import SignalAggregator, TimeframeSignal, AggregatedSignal
from src.learning.confidence_gate import ConfidenceGate
from src.learning.state_manager import LearningStateManager
from src.learning.outcome_tracker import OutcomeTracker
from src.learning.retraining_engine import RetrainingEngine
from src.ml.learning.continual import ContinualLearner

logger = logging.getLogger(__name__)


class ContinuousLearningSystem:
    """
    Main orchestrator for continuous learning.

    Thread-safe: All operations are thread-safe
    Event-driven: Triggered by candle close events
    Configurable: All parameters from config.yaml
    """

    def __init__(
        self,
        predictor: any,  # UnbreakablePredictor instance
        database: Database,
        paper_brokerage: any,
        live_brokerage: any = None,
        config: dict = None
    ):
        """
        Initialize continuous learning system.

        Args:
            predictor: UnbreakablePredictor instance
            database: Database instance
            paper_brokerage: Paper trading brokerage
            live_brokerage: Live trading brokerage (optional)
            config: Configuration dict from config.yaml
        """
        self.predictor = predictor
        self.database = database
        self.paper_brokerage = paper_brokerage
        self.live_brokerage = live_brokerage
        self.config = config or {}

        # Get configurations
        self.cl_config = self.config.get('continuous_learning', {})
        self.timeframes_config = self.config.get('timeframes', {})

        # Initialize components
        self.signal_aggregator = SignalAggregator(
            config=self.timeframes_config
        )

        self.confidence_gate = ConfidenceGate(
            config=self.cl_config.get('confidence', {})
        )

        self.state_manager = LearningStateManager(
            database=database
        )

        # Get continual learner from predictor
        continual_learner = getattr(predictor, 'continual_learner', None)
        if not continual_learner:
            logger.warning("Predictor has no continual_learner, creating new one")
            continual_learner = ContinualLearner(
                config=self.cl_config.get('ewc', {})
            )

        self.outcome_tracker = OutcomeTracker(
            database=database,
            continual_learner=continual_learner,
            config=self.cl_config.get('retraining', {})
        )

        # Get model manager from predictor
        model_manager = getattr(predictor, 'model_manager', None)
        if not model_manager:
            logger.warning("Predictor has no model_manager")
            model_manager = MultiTimeframeModelManager(
                config=self.config,
                database=database
            )

        self.retraining_engine = RetrainingEngine(
            model_manager=model_manager,
            continual_learner=continual_learner,
            database=database,
            config=self.cl_config.get('retraining', {})
        )

        # Track active retraining threads
        self.retraining_threads: Dict[str, threading.Thread] = {}
        self._threads_lock = threading.Lock()

        # Statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            'candles_processed': 0,
            'predictions_made': 0,
            'trades_executed': 0,
            'paper_trades': 0,
            'live_trades': 0,
            'retrainings_triggered': 0,
            'online_updates': 0,
            'mode_transitions': 0
        }

        # Get enabled intervals
        self.enabled_intervals = self._get_enabled_intervals()

        logger.info(
            f"ContinuousLearningSystem initialized: "
            f"intervals={self.enabled_intervals}, "
            f"aggregation={self.timeframes_config.get('aggregation_method', 'weighted_vote')}"
        )

    def _get_enabled_intervals(self) -> List[str]:
        """Get list of enabled intervals from config."""
        intervals = []

        for interval_config in self.timeframes_config.get('intervals', []):
            if interval_config.get('enabled', True):
                intervals.append(interval_config['interval'])

        return intervals

    def on_candle_closed(
        self,
        symbol: str,
        interval: str,
        candle: any,
        data: dict = None
    ) -> dict:
        """
        Main event handler - called when a candle completes.

        This is the entry point for all continuous learning logic.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            interval: Timeframe that closed (e.g., '1h')
            candle: Candle object
            data: Pre-fetched data (optional)

        Returns:
            Dict with prediction results, execution status, mode, etc.
        """
        try:
            with self._stats_lock:
                self._stats['candles_processed'] += 1

            logger.info(
                f"[{symbol} @ {interval}] Candle closed at {candle.timestamp}"
            )

            # 1. ONLINE LEARNING UPDATE (fast, incremental)
            if self.cl_config.get('online_learning', {}).get('enabled', True):
                self._online_update(symbol, interval, candle, data)

            # 2. GET PREDICTIONS FROM ALL TIMEFRAMES
            timeframe_signals = self._get_all_timeframe_predictions(
                symbol=symbol,
                data=data
            )

            # 3. AGGREGATE SIGNALS
            aggregated = self.signal_aggregator.aggregate(timeframe_signals)

            # 4. GET CURRENT MODE
            # Use highest weighted timeframe for mode determination
            primary_interval = self._get_primary_interval()
            current_mode = self.state_manager.get_current_mode(
                symbol=symbol,
                interval=primary_interval
            )

            # 5. CHECK CONFIDENCE GATE
            can_trade_live, reason = self.confidence_gate.should_trade(
                confidence=aggregated.confidence,
                current_mode=current_mode,
                regime=aggregated.regime
            )

            # 6. DETERMINE MODE AND BROKERAGE
            if can_trade_live and aggregated.confidence >= self.cl_config.get('confidence', {}).get('trading_threshold', 0.80):
                mode = 'TRADING'
                brokerage = self.live_brokerage if self.live_brokerage else self.paper_brokerage
                brokerage_type = 'live' if self.live_brokerage else 'paper_fallback'
            else:
                mode = 'LEARNING'
                brokerage = self.paper_brokerage
                brokerage_type = 'paper'

            # Update state if changed
            if mode != current_mode:
                self._handle_mode_transition(
                    symbol=symbol,
                    interval=primary_interval,
                    new_mode=mode,
                    confidence=aggregated.confidence,
                    reason=reason
                )

            # 7. EXECUTE TRADE (if signal exists)
            executed = False
            signal_id = None

            if aggregated.direction != 'NEUTRAL':
                signal_id = self._execute_trade(
                    brokerage=brokerage,
                    symbol=symbol,
                    prediction=aggregated,
                    is_paper=(mode == 'LEARNING')
                )
                executed = True

                logger.info(
                    f"[{symbol}] {mode} MODE: {aggregated.direction} @ "
                    f"{aggregated.confidence:.2%} ({brokerage_type})"
                )

                with self._stats_lock:
                    self._stats['trades_executed'] += 1
                    if mode == 'LEARNING':
                        self._stats['paper_trades'] += 1
                    else:
                        self._stats['live_trades'] += 1

            # 8. CHECK FOR COMPLETED TRADES
            self._check_completed_trades(symbol, interval, candle)

            # Record prediction
            with self._stats_lock:
                self._stats['predictions_made'] += 1

            return {
                'aggregated_signal': aggregated.to_dict(),
                'mode': mode,
                'executed': executed,
                'brokerage': brokerage_type,
                'timeframe_signals': {
                    k: v.to_dict() for k, v in timeframe_signals.items()
                },
                'reason': reason,
                'signal_id': signal_id
            }

        except Exception as e:
            logger.error(
                f"[{symbol} @ {interval}] Error in on_candle_closed: {e}",
                exc_info=True
            )
            return {
                'error': str(e),
                'mode': 'LEARNING',
                'executed': False
            }

    def _get_all_timeframe_predictions(
        self,
        symbol: str,
        data: dict = None
    ) -> Dict[str, TimeframeSignal]:
        """
        Get predictions from all enabled timeframes.

        Args:
            symbol: Trading pair
            data: Pre-fetched data (optional)

        Returns:
            Dict mapping interval to TimeframeSignal
        """
        predictions = {}

        for interval in self.enabled_intervals:
            try:
                # Get data for this timeframe
                if data and data.get('interval') == interval:
                    df = data.get('candles')
                else:
                    # Fetch from database
                    sequence_length = self._get_sequence_length(interval)
                    df = self.database.get_candles(
                        symbol=symbol,
                        interval=interval,
                        limit=sequence_length + 100  # Extra for feature calculation
                    )

                if df is None or len(df) < 10:
                    logger.warning(
                        f"Insufficient data for {symbol} @ {interval}: {len(df) if df is not None else 0} candles"
                    )
                    continue

                # Get prediction from model
                result = self.predictor.predict(
                    df=df,
                    symbol=symbol,
                    interval=interval
                )

                # Create TimeframeSignal
                signal = TimeframeSignal(
                    interval=interval,
                    direction=result.direction,
                    confidence=result.confidence,
                    lstm_prob=result.lstm_probability,
                    advanced_result=result,
                    timestamp=datetime.utcnow()
                )

                predictions[interval] = signal

                logger.debug(
                    f"[{symbol} @ {interval}] Prediction: {result.direction} @ "
                    f"{result.confidence:.2%}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to get prediction for {symbol} @ {interval}: {e}",
                    exc_info=True
                )
                continue

        return predictions

    def _get_sequence_length(self, interval: str) -> int:
        """Get sequence length for interval from config."""
        for interval_config in self.timeframes_config.get('intervals', []):
            if interval_config['interval'] == interval:
                return interval_config.get('sequence_length', 60)

        return 60  # Default

    def _get_primary_interval(self) -> str:
        """Get highest weighted interval."""
        intervals = self.timeframes_config.get('intervals', [])

        if not intervals:
            return '1h'  # Default

        # Find max weight
        primary = max(
            [i for i in intervals if i.get('enabled', True)],
            key=lambda x: x.get('weight', 0.0)
        )

        return primary['interval']

    def _handle_mode_transition(
        self,
        symbol: str,
        interval: str,
        new_mode: str,
        confidence: float,
        reason: str
    ):
        """Handle mode transition (LEARNING <-> TRADING)."""
        if new_mode == 'TRADING':
            self.state_manager.transition_to_trading(
                symbol=symbol,
                interval=interval,
                confidence=confidence
            )
            logger.info(
                f"✓ [{symbol} @ {interval}] Transitioned to TRADING mode "
                f"(confidence: {confidence:.2%})"
            )
        else:
            self.state_manager.transition_to_learning(
                symbol=symbol,
                interval=interval,
                reason=reason
            )
            logger.info(
                f"← [{symbol} @ {interval}] Transitioned to LEARNING mode "
                f"(reason: {reason})"
            )

        with self._stats_lock:
            self._stats['mode_transitions'] += 1

    def _execute_trade(
        self,
        brokerage: any,
        symbol: str,
        prediction: AggregatedSignal,
        is_paper: bool
    ) -> Optional[int]:
        """
        Execute trade via brokerage.

        Args:
            brokerage: Brokerage instance
            symbol: Trading pair
            prediction: Aggregated prediction
            is_paper: Whether this is a paper trade

        Returns:
            Signal ID or None
        """
        try:
            # Save signal to database
            signal_data = {
                'timestamp': int(prediction.timestamp.timestamp()),
                'datetime': prediction.timestamp.isoformat(),
                'symbol': symbol,
                'direction': prediction.direction,
                'confidence': prediction.confidence,
                'regime': prediction.regime,
                'is_paper': is_paper,
                'metadata': prediction.to_dict()
            }

            signal_id = self.database.save_signal(signal_data)

            # Execute via brokerage
            if hasattr(brokerage, 'execute_signal'):
                brokerage.execute_signal(
                    symbol=symbol,
                    signal=prediction.direction,
                    confidence=prediction.confidence,
                    signal_id=signal_id
                )

            return signal_id

        except Exception as e:
            logger.error(f"Failed to execute trade: {e}", exc_info=True)
            return None

    def _check_completed_trades(
        self,
        symbol: str,
        interval: str,
        candle: any
    ):
        """
        Check for completed trades and record outcomes.

        Args:
            symbol: Trading pair
            interval: Timeframe
            candle: Latest candle
        """
        try:
            # Get pending signals for this symbol
            pending_signals = self.database.get_pending_signals(symbol)

            for pending in pending_signals:
                # Check if trade should be closed
                should_close, close_reason = self._should_close_trade(pending, candle)

                if should_close:
                    # Record outcome
                    outcome = self.outcome_tracker.record_outcome(
                        signal_id=pending['id'],
                        symbol=symbol,
                        interval=pending.get('interval', interval),
                        entry_price=pending['entry_price'],
                        exit_price=candle.close,
                        predicted_direction=pending['direction'],
                        confidence=pending['confidence'],
                        features=pending.get('features', np.array([])),
                        regime=pending.get('regime', 'NORMAL')
                    )

                    logger.info(
                        f"[{symbol}] Trade closed: "
                        f"{'✓ WIN' if outcome['was_correct'] else '✗ LOSS'} "
                        f"({outcome['pnl_percent']:.2f}%)"
                    )

                    # Trigger retraining if needed
                    if outcome['should_retrain']:
                        self._schedule_retrain(
                            symbol=symbol,
                            interval=interval,
                            reason=outcome['trigger_reason']
                        )

        except Exception as e:
            logger.error(f"Error checking completed trades: {e}", exc_info=True)

    def _should_close_trade(self, signal: dict, candle: any) -> tuple[bool, str]:
        """
        Determine if a trade should be closed with proper risk management.

        Exit conditions (in priority order):
        1. Stop-loss hit (default: -2%)
        2. Take-profit hit (default: +4% = 2:1 R:R)
        3. Max holding period (default: 24 hours)
        4. Opposite signal (TODO: check current prediction)

        Args:
            signal: Pending signal dict with entry_price, direction, datetime
            candle: Current candle with close price

        Returns:
            (should_close: bool, reason: str)
        """
        from datetime import datetime, timedelta

        # Get exit parameters from config (with sensible defaults)
        exit_config = self.config.get('continuous_learning', {}).get('exit_logic', {})
        stop_loss_pct = exit_config.get('stop_loss_pct', 2.0)
        take_profit_pct = exit_config.get('take_profit_pct', 4.0)
        max_holding_hours = exit_config.get('max_holding_hours', 24)

        # Extract trade info
        entry_price = signal.get('entry_price')
        if entry_price is None or entry_price <= 0:
            logger.warning(f"Invalid entry_price for signal {signal.get('id')}: {entry_price}")
            return (True, "invalid_entry_price")

        current_price = candle.close
        direction = signal.get('direction', 'BUY')

        # Calculate P&L percentage
        if direction == 'BUY':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100
        elif direction == 'SELL':
            pnl_pct = ((entry_price - current_price) / entry_price) * 100
        else:
            logger.warning(f"Unknown direction '{direction}' for signal {signal.get('id')}")
            return (True, "unknown_direction")

        # 1. STOP-LOSS CHECK (highest priority - protect capital)
        if pnl_pct <= -stop_loss_pct:
            logger.info(
                f"🛑 Stop-loss triggered for {signal.get('symbol')} {direction}: "
                f"{pnl_pct:.2f}% ≤ -{stop_loss_pct:.2f}% "
                f"(entry: {entry_price:.2f}, current: {current_price:.2f})"
            )
            return (True, "stop_loss")

        # 2. TAKE-PROFIT CHECK (lock in gains)
        if pnl_pct >= take_profit_pct:
            logger.info(
                f"✅ Take-profit triggered for {signal.get('symbol')} {direction}: "
                f"{pnl_pct:.2f}% ≥ {take_profit_pct:.2f}% "
                f"(entry: {entry_price:.2f}, current: {current_price:.2f})"
            )
            return (True, "take_profit")

        # 3. MAX HOLDING PERIOD CHECK (prevent stale positions)
        try:
            signal_time = datetime.fromisoformat(signal['datetime'])
            current_time = datetime.utcfromtimestamp(candle.timestamp)
            duration = current_time - signal_time

            if duration > timedelta(hours=max_holding_hours):
                logger.info(
                    f"⏰ Max holding period reached for {signal.get('symbol')} {direction}: "
                    f"{duration.total_seconds() / 3600:.1f}h > {max_holding_hours}h "
                    f"(P&L: {pnl_pct:.2f}%)"
                )
                return (True, "max_holding_period")

        except (KeyError, ValueError) as e:
            logger.warning(f"Failed to parse signal datetime: {e}")
            return (True, "invalid_datetime")

        # 4. OPPOSITE SIGNAL CHECK (future enhancement)
        # TODO: Implement by checking if current prediction has opposite direction
        # with high confidence. Requires access to latest prediction.

        # Keep position open
        return (False, "holding")

    def _should_close_trade_wrapper(self, signal: dict, candle: any) -> bool:
        """
        Wrapper for backward compatibility.

        Returns:
            bool: True if should close
        """
        should_close, reason = self._should_close_trade(signal, candle)
        return should_close

    def _schedule_retrain(
        self,
        symbol: str,
        interval: str,
        reason: str
    ):
        """
        Schedule background retraining.

        Args:
            symbol: Trading pair
            interval: Timeframe
            reason: Trigger reason
        """
        key = f"{symbol}_{interval}"

        with self._threads_lock:
            # Check if already retraining
            if key in self.retraining_threads and self.retraining_threads[key].is_alive():
                logger.debug(f"Retraining already in progress for {key}")
                return

            def retrain_task():
                try:
                    logger.info(f"⚙ [{key}] Starting retraining (reason: {reason})")

                    result = self.retraining_engine.retrain(
                        symbol=symbol,
                        interval=interval,
                        trigger_reason=reason
                    )

                    if result['success']:
                        # Transition to TRADING mode
                        self.state_manager.transition_to_trading(
                            symbol=symbol,
                            interval=interval,
                            confidence=result['validation_confidence']
                        )

                        logger.info(
                            f"✓ [{key}] Retraining successful! "
                            f"Confidence: {result['validation_confidence']:.2%}, "
                            f"Duration: {result['duration_seconds']:.1f}s"
                        )
                    else:
                        logger.warning(
                            f"[{key}] Retraining completed but confidence still low: "
                            f"{result.get('validation_confidence', 0):.2%}"
                        )

                except Exception as e:
                    logger.error(f"[{key}] Retraining failed: {e}", exc_info=True)

            # Start background thread
            thread = threading.Thread(
                target=retrain_task,
                daemon=True,
                name=f"Retrain-{key}"
            )
            self.retraining_threads[key] = thread
            thread.start()

            with self._stats_lock:
                self._stats['retrainings_triggered'] += 1

        logger.info(f"[{key}] Retraining scheduled (reason: {reason})")

    def _online_update(
        self,
        symbol: str,
        interval: str,
        candle: any,
        data: dict = None
    ):
        """
        Perform online learning update.

        Small, incremental update based on latest candle.

        Args:
            symbol: Trading pair
            interval: Timeframe
            candle: Latest candle
            data: Pre-fetched data (optional)
        """
        try:
            # Get online learning config
            online_config = self.cl_config.get('online_learning', {})

            if not online_config.get('enabled', True):
                return

            # Get data
            if data and data.get('candles') is not None:
                df = data['candles']
            else:
                sequence_length = self._get_sequence_length(interval)
                df = self.database.get_candles(
                    symbol=symbol,
                    interval=interval,
                    limit=sequence_length + 10
                )

            if df is None or len(df) < 10:
                return

            # Perform small online update
            # (Implementation depends on predictor's online learning capabilities)
            if hasattr(self.predictor, 'online_update'):
                self.predictor.online_update(
                    df=df,
                    symbol=symbol,
                    interval=interval,
                    learning_rate=online_config.get('learning_rate', 0.0001)
                )

                with self._stats_lock:
                    self._stats['online_updates'] += 1

                logger.debug(f"[{symbol} @ {interval}] Online learning update performed")

        except Exception as e:
            logger.error(f"Online update failed: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """Get continuous learning system statistics."""
        with self._stats_lock:
            stats = {
                **self._stats,
                'aggregator_stats': self.signal_aggregator.get_stats(),
                'confidence_gate_stats': self.confidence_gate.get_stats(),
                'outcome_tracker_stats': self.outcome_tracker.get_stats(),
                'retraining_stats': self.retraining_engine.get_stats(),
                'active_retrainings': sum(
                    1 for t in self.retraining_threads.values() if t.is_alive()
                )
            }

        return stats

    def stop(self):
        """Stop continuous learning system and wait for retraining threads."""
        logger.info("Stopping ContinuousLearningSystem...")

        # Wait for active retraining threads
        with self._threads_lock:
            active_threads = [
                t for t in self.retraining_threads.values() if t.is_alive()
            ]

        if active_threads:
            logger.info(f"Waiting for {len(active_threads)} retraining threads...")
            for thread in active_threads:
                thread.join(timeout=30)

        logger.info("ContinuousLearningSystem stopped")
