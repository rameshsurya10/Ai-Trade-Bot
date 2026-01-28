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
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.core.database import Database
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.multi_timeframe.aggregator import SignalAggregator, TimeframeSignal, AggregatedSignal
from src.learning.confidence_gate import ConfidenceGate
from src.learning.state_manager import LearningStateManager
from src.learning.outcome_tracker import OutcomeTracker
from src.learning.retraining_engine import RetrainingEngine
from src.learning.strategy_analyzer import StrategyAnalyzer
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

        # Create ConfidenceGateConfig from dict
        conf_dict = self.cl_config.get('confidence', {})
        from src.learning.confidence_gate import ConfidenceGateConfig
        conf_config = ConfidenceGateConfig(
            trading_threshold=conf_dict.get('trading_threshold', 0.8),
            hysteresis=conf_dict.get('hysteresis', 0.05),
            smoothing_alpha=conf_dict.get('smoothing_alpha', 0.3),
            regime_adjustment=conf_dict.get('regime_adjustment', True)
        )

        self.confidence_gate = ConfidenceGate(config=conf_config)

        self.state_manager = LearningStateManager(
            database=database
        )

        # Get continual learner from predictor
        continual_learner = getattr(predictor, 'continual_learner', None)
        if not continual_learner and hasattr(predictor, 'model'):
            logger.warning("Predictor has no continual_learner, creating new one")
            ewc_config = self.cl_config.get("ewc", {})
            continual_learner = ContinualLearner(
                model=predictor.model,
                ewc_lambda=ewc_config.get("lambda", 1000.0),
                replay_buffer_size=self.cl_config.get("experience_replay", {}).get("buffer_size", 10000),
                replay_batch_size=32,
                drift_window=100
            )
        elif not continual_learner:
            logger.info("Predictor is not a neural network model, skipping continual learner")
            continual_learner = None

        self.outcome_tracker = OutcomeTracker(
            database=database,
            continual_learner=continual_learner,
            config=self.cl_config.get('retraining', {})
        )

        # Get model manager from predictor
        model_manager = getattr(predictor, 'model_manager', None)
        if not model_manager:
            logger.warning("Predictor has no model_manager, creating one")
            model_manager = MultiTimeframeModelManager(
                models_dir=self.config.get('model', {}).get('models_dir', 'models'),
                config=self.config.get('model', {})
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

        # Store latest predictions per symbol for opposite signal detection
        self._latest_predictions: Dict[str, AggregatedSignal] = {}
        self._predictions_lock = threading.Lock()

        # Track last processed outcome per symbol/interval to avoid duplicate updates
        self._last_online_update_outcome_id: Dict[str, int] = {}
        self._outcome_tracking_lock = threading.Lock()

        # Strategy analyzer for strategy-based trade decisions
        db_path = self.config.get('database', {}).get('path', 'data/trading.db')
        try:
            self.strategy_analyzer = StrategyAnalyzer(
                database_path=db_path,
                config_path='config.yaml'
            )
        except Exception as e:
            logger.warning(f"Strategy analyzer initialization failed: {e}. Strategy-based filtering disabled.")
            self.strategy_analyzer = None

        self._strategy_cache: Dict[str, dict] = {}  # symbol -> best strategy
        self._strategy_cache_lock = threading.Lock()
        self._last_strategy_analysis: Dict[str, datetime] = {}

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

            # 7. STRATEGY EVALUATION - Check all strategies before trading
            best_strategy = self._evaluate_strategies_for_trade(
                symbol=symbol,
                interval=primary_interval,
                direction=aggregated.direction,
                confidence=aggregated.confidence,
                regime=aggregated.regime
            )

            # 8. EXECUTE TRADE (only if signal exists AND strategy approves)
            executed = False
            signal_id = None
            strategy_approved = best_strategy is not None and best_strategy.get('is_recommended', False)

            if aggregated.direction != 'NEUTRAL':
                # In LEARNING mode, always execute paper trades for training data
                # In TRADING mode, only trade if strategy is recommended
                should_execute = (mode == 'LEARNING') or (mode == 'TRADING' and strategy_approved)

                if should_execute:
                    signal_id = self._execute_trade(
                        brokerage=brokerage,
                        symbol=symbol,
                        prediction=aggregated,
                        is_paper=(mode == 'LEARNING'),
                        strategy_name=best_strategy.get('strategy_name') if best_strategy else None
                    )
                    executed = True

                    strategy_info = f" [Strategy: {best_strategy['strategy_name']}]" if best_strategy else ""
                    logger.info(
                        f"[{symbol}] {mode} MODE: {aggregated.direction} @ "
                        f"{aggregated.confidence:.2%} ({brokerage_type}){strategy_info}"
                    )

                    with self._stats_lock:
                        self._stats['trades_executed'] += 1
                        if mode == 'LEARNING':
                            self._stats['paper_trades'] += 1
                        else:
                            self._stats['live_trades'] += 1
                else:
                    logger.info(
                        f"[{symbol}] Trade SKIPPED: {aggregated.direction} @ "
                        f"{aggregated.confidence:.2%} - No recommended strategy "
                        f"(best: {best_strategy.get('strategy_name', 'None') if best_strategy else 'None'})"
                    )

            # 8. STORE LATEST PREDICTION (for opposite signal detection)
            with self._predictions_lock:
                self._latest_predictions[symbol] = aggregated

            # 9. CHECK FOR COMPLETED TRADES
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
                'signal_id': signal_id,
                'strategy': best_strategy,
                'strategy_approved': strategy_approved
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

    def _evaluate_strategies_for_trade(
        self,
        symbol: str,
        interval: str,
        direction: str,
        confidence: float,
        regime: str
    ) -> Optional[dict]:
        """
        Evaluate all strategies and determine if we should trade.

        This method:
        1. Analyzes historical trade outcomes to discover strategies
        2. Ranks strategies by Sharpe ratio
        3. Saves strategy performance to database
        4. Returns the best strategy if it's recommended for trading

        Args:
            symbol: Trading pair
            interval: Timeframe
            direction: Predicted direction ('BUY', 'SELL', 'NEUTRAL')
            confidence: Model confidence
            regime: Current market regime

        Returns:
            Best strategy dict with is_recommended flag, or None
        """
        # If strategy analyzer not available, skip strategy-based filtering
        if self.strategy_analyzer is None:
            logger.debug("Strategy analyzer not available, skipping strategy evaluation")
            return None

        try:
            # Check if we need to refresh strategy analysis (every hour)
            key = f"{symbol}_{interval}"
            now = datetime.utcnow()

            # Thread-safe check for cache freshness
            with self._strategy_cache_lock:
                last_analysis = self._last_strategy_analysis.get(key)
                should_refresh = (
                    last_analysis is None or
                    (now - last_analysis).total_seconds() > 3600  # 1 hour
                )

                # If another thread is already refreshing, use existing cache
                if should_refresh and key in self._strategy_cache:
                    # Return cached while another thread refreshes
                    cached = self._strategy_cache.get(key)
                    if cached:
                        # Another thread might be refreshing, just use cache
                        pass

            if should_refresh:
                logger.info(f"[{symbol} @ {interval}] Refreshing strategy analysis...")

                # Discover strategies from historical data
                strategies = self.strategy_analyzer.discover_strategies(lookback_days=365)

                if strategies:
                    # Rank strategies
                    ranked = self.strategy_analyzer.rank_strategies(by='sharpe')

                    # Save all strategies to database
                    for strategy_name, strategy in strategies.items():
                        is_recommended = (
                            strategy.sharpe_ratio > 1.0 and
                            strategy.win_rate > 0.50 and
                            strategy.total_trades >= 10
                        )

                        recommendation = self.strategy_analyzer._get_recommendation(strategy)

                        self.database.save_strategy_performance(
                            strategy_name=strategy_name,
                            symbol=symbol,
                            interval=interval,
                            metrics={
                                'total_trades': strategy.total_trades,
                                'win_rate': strategy.win_rate,
                                'avg_profit_pct': strategy.avg_profit_pct,
                                'avg_loss_pct': strategy.avg_loss_pct,
                                'profit_factor': strategy.profit_factor,
                                'sharpe_ratio': strategy.sharpe_ratio,
                                'max_drawdown_pct': strategy.max_drawdown_pct,
                                'avg_holding_hours': strategy.avg_holding_hours,
                                'best_regime': strategy.best_regime,
                                'confidence_threshold': strategy.confidence_threshold,
                                'is_recommended': is_recommended,
                                'recommendation': recommendation
                            }
                        )

                    logger.info(
                        f"[{symbol} @ {interval}] Saved {len(strategies)} strategies to database"
                    )

                    # Cache the best strategy
                    if ranked:
                        best_name, best_strategy = ranked[0]
                        is_rec = (
                            best_strategy.sharpe_ratio > 1.0 and
                            best_strategy.win_rate > 0.50 and
                            best_strategy.total_trades >= 10
                        )

                        with self._strategy_cache_lock:
                            self._strategy_cache[key] = {
                                'strategy_name': best_name,
                                'sharpe_ratio': best_strategy.sharpe_ratio,
                                'win_rate': best_strategy.win_rate,
                                'profit_factor': best_strategy.profit_factor,
                                'total_trades': best_strategy.total_trades,
                                'best_regime': best_strategy.best_regime,
                                'is_recommended': is_rec,
                                'recommendation': self.strategy_analyzer._get_recommendation(best_strategy)
                            }

                # Update timestamp inside the lock
                self._last_strategy_analysis[key] = now

            # Get cached best strategy (already within lock scope above for refresh case)
            with self._strategy_cache_lock:
                best_strategy = self._strategy_cache.get(key)

            # If no strategy found, try from database
            if not best_strategy:
                db_strategy = self.database.get_best_strategy(symbol, interval)
                if db_strategy:
                    best_strategy = dict(db_strategy)

            return best_strategy

        except Exception as e:
            logger.error(f"Strategy evaluation failed: {e}", exc_info=True)
            return None

    def _execute_trade(
        self,
        brokerage: any,
        symbol: str,
        prediction: AggregatedSignal,
        is_paper: bool,
        strategy_name: str = None
    ) -> Optional[int]:
        """
        Execute trade via brokerage.

        Args:
            brokerage: Brokerage instance
            symbol: Trading pair
            prediction: Aggregated prediction
            is_paper: Whether this is a paper trade
            strategy_name: Name of the strategy being used

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
            from datetime import timezone

            # Parse signal time (should always be UTC-aware)
            signal_time = datetime.fromisoformat(signal['datetime'])
            if signal_time.tzinfo is None:
                # WARN: This shouldn't happen if data is clean
                logger.warning(
                    f"Signal {signal.get('id', 'unknown')} has naive datetime, "
                    f"assuming UTC. This may indicate a data storage bug."
                )
                signal_time = signal_time.replace(tzinfo=timezone.utc)

            # Current time from candle timestamp (always UTC)
            current_time = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
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

        # 4. OPPOSITE SIGNAL CHECK
        # Close trade if current prediction has opposite direction with high confidence
        opposite_signal_threshold = exit_config.get('opposite_signal_threshold', 0.70)
        symbol = signal.get('symbol', '')

        # Thread-safe: Copy attributes inside lock to prevent race conditions
        with self._predictions_lock:
            latest = self._latest_predictions.get(symbol)
            if latest is not None:
                latest_direction = getattr(latest, 'direction', None)
                latest_confidence = getattr(latest, 'confidence', 0.0)
            else:
                latest_direction = None
                latest_confidence = 0.0

        # Check for opposite signal (outside lock, using copied values)
        if latest_direction is not None:
            # Determine if opposite signal
            is_opposite = (
                (direction == 'BUY' and latest_direction == 'SELL') or
                (direction == 'SELL' and latest_direction == 'BUY')
            )

            if is_opposite and latest_confidence >= opposite_signal_threshold:
                logger.info(
                    f"Opposite signal detected for {symbol} {direction}: "
                    f"new signal is {latest_direction} @ {latest_confidence:.2%} "
                    f"(threshold: {opposite_signal_threshold:.2%})"
                )
                return (True, "opposite_signal")

        # Keep position open
        return (False, "holding")

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
        Check for completed trades and perform online learning update with confirmed outcomes.

        Online learning requires actual outcomes to avoid information leakage.
        This method checks if we have pending trades that can now be evaluated,
        and only then performs the online update.

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

            if not hasattr(self.predictor, 'online_update'):
                return

            # Get recent confirmed outcomes for this symbol/interval
            # These are outcomes that have been recorded with actual price movements
            recent_outcomes = self.outcome_tracker.get_performance_summary(
                symbol=symbol,
                interval=interval,
                limit=5
            ).get('recent_outcomes', [])

            if not recent_outcomes:
                return

            # Only update with confirmed outcomes that haven't been used yet
            latest_outcome = recent_outcomes[0] if recent_outcomes else None
            if latest_outcome is None:
                return

            # Get outcome ID for tracking (avoid duplicate updates)
            outcome_id = latest_outcome.get('id') or latest_outcome.get('signal_id') or latest_outcome.get('outcome_id')
            key = f"{symbol}_{interval}"

            with self._outcome_tracking_lock:
                if self._last_online_update_outcome_id.get(key) == outcome_id:
                    # Already processed this outcome
                    return
                self._last_online_update_outcome_id[key] = outcome_id

            # Get the actual outcome (1 if price went up, 0 if down)
            # Must consider the original trade direction for correct labeling
            predicted_direction = latest_outcome.get('predicted_direction', 'BUY')
            was_correct = latest_outcome.get('was_correct', False)

            # Derive actual price movement from prediction and correctness
            # BUY correct -> price UP (1), BUY wrong -> price DOWN (0)
            # SELL correct -> price DOWN (0), SELL wrong -> price UP (1)
            if predicted_direction == 'BUY':
                actual_outcome = 1 if was_correct else 0
            else:  # SELL
                actual_outcome = 0 if was_correct else 1

            # Get data for features
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

            # Perform online update with the actual outcome
            self.predictor.online_update(
                df=df,
                symbol=symbol,
                interval=interval,
                learning_rate=online_config.get('learning_rate', 0.0001),
                actual_outcome=actual_outcome
            )

            with self._stats_lock:
                self._stats['online_updates'] += 1

            logger.debug(
                f"[{symbol} @ {interval}] Online learning update performed "
                f"(direction={predicted_direction}, was_correct={was_correct}, outcome={actual_outcome})"
            )

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
