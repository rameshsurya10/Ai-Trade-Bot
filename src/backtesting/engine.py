"""
Backtest Engine
===============
Core backtesting logic for validating trading strategies.

Simple Flow:
1. Load historical data
2. For each candle: generate prediction -> check entry
3. If signal: track until stop loss or take profit hit
4. Calculate performance metrics
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
import sys
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.types import SignalType, SignalStrength
from src.core.config import Config
from src.analysis_engine import AnalysisEngine, FeatureCalculator
from .metrics import BacktestMetrics, calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """Track an open position during backtesting."""
    signal_id: int
    entry_price: float
    entry_time: datetime
    direction: SignalType
    stop_loss: float
    take_profit: float
    strength: SignalStrength
    confidence: float


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    # Data range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Position management
    max_open_positions: int = 1
    allow_concurrent: bool = False

    # Simulation settings
    slippage_percent: float = 0.05  # 0.05% slippage per trade
    commission_percent: float = 0.1  # 0.1% commission per trade

    # Exit rules
    max_hold_candles: int = 24  # Force exit after N candles
    use_trailing_stop: bool = False
    trailing_stop_percent: float = 1.0


class BacktestEngine:
    """
    Backtest trading strategies against historical data.

    Usage:
        engine = BacktestEngine(config_path="config.yaml")
        results = engine.run()
        print(results.summary())
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        backtest_config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtest engine.

        Args:
            config_path: Path to trading config
            backtest_config: Backtesting-specific settings
        """
        self.config = Config.load(config_path)
        self.bt_config = backtest_config or BacktestConfig()

        # Analysis engine for predictions
        self.analysis = AnalysisEngine(config_path)

        # State
        self.trades: List[Dict[str, Any]] = []
        self.open_positions: List[OpenPosition] = []
        self.signals_generated: int = 0
        self.current_index: int = 0

        # Data
        self.df: Optional[pd.DataFrame] = None

    def load_data(self, df: Optional[pd.DataFrame] = None) -> bool:
        """
        Load historical data for backtesting.

        Args:
            df: Optional DataFrame to use. If None, loads from database.

        Returns:
            True if data loaded successfully
        """
        if df is not None:
            self.df = df.copy()
        else:
            # Load from database
            from src.data_service import DataService
            data_service = DataService()
            self.df = data_service.get_candles(limit=100000)

        if self.df.empty or len(self.df) < 200:
            logger.error(f"Insufficient data: {len(self.df) if self.df is not None else 0} candles")
            return False

        # Calculate features
        self.df = FeatureCalculator.calculate_all(self.df)

        # Apply date filters
        if self.bt_config.start_date:
            self.df = self.df[self.df['datetime'] >= self.bt_config.start_date]
        if self.bt_config.end_date:
            self.df = self.df[self.df['datetime'] <= self.bt_config.end_date]

        self.df = self.df.reset_index(drop=True)

        logger.info(f"Loaded {len(self.df)} candles for backtesting")
        logger.info(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")

        return True

    def run(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BacktestMetrics:
        """
        Run the backtest.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            BacktestMetrics with results
        """
        if self.df is None or self.df.empty:
            if not self.load_data():
                return BacktestMetrics()

        # Load model
        self.analysis.load_model()

        # Reset state
        self.trades = []
        self.open_positions = []
        self.signals_generated = 0

        # Minimum lookback for features
        min_lookback = max(200, self.config.model.sequence_length + 50)
        total_candles = len(self.df) - min_lookback

        logger.info(f"Starting backtest with {total_candles} tradeable candles")
        logger.info("-" * 60)

        # Main loop: iterate through candles
        for i in range(min_lookback, len(self.df)):
            self.current_index = i
            current_candle = self.df.iloc[i]
            current_time = current_candle['datetime']
            current_price = current_candle['close']
            current_high = current_candle['high']
            current_low = current_candle['low']

            # 1. Check open positions for exit
            self._check_exits(current_candle)

            # 2. Generate prediction
            lookback_df = self.df.iloc[i - min_lookback:i + 1].copy()
            prediction = self.analysis.predict(lookback_df)

            # 3. Check for signal
            if self._should_enter(prediction):
                self._open_position(prediction, current_candle)

            # Progress update
            if progress_callback and (i - min_lookback) % 100 == 0:
                progress_callback(i - min_lookback, total_candles)

        # Close any remaining positions at end
        self._close_all_positions(self.df.iloc[-1])

        logger.info("-" * 60)
        logger.info(f"Backtest complete: {len(self.trades)} trades from {self.signals_generated} signals")

        # Calculate metrics
        return calculate_metrics(self.trades)

    def _should_enter(self, prediction: dict) -> bool:
        """Check if we should enter a position based on prediction."""
        signal = prediction.get('signal', 'NEUTRAL')
        confidence = prediction.get('confidence', 0)

        # Skip neutral/wait signals
        if signal in ['NEUTRAL', 'WAIT']:
            return False

        # Check confidence threshold
        if confidence < self.config.analysis.min_confidence:
            return False

        # Check max positions
        if len(self.open_positions) >= self.bt_config.max_open_positions:
            return False

        # Count as valid signal
        self.signals_generated += 1
        return True

    def _open_position(self, prediction: dict, candle: pd.Series):
        """Open a new position."""
        entry_price = candle['close']
        entry_time = candle['datetime']

        # Apply slippage
        slippage = entry_price * (self.bt_config.slippage_percent / 100)
        if 'BUY' in prediction['signal']:
            entry_price += slippage  # Worse fill for buy
        else:
            entry_price -= slippage  # Worse fill for sell

        # Determine strength
        confidence = prediction['confidence']
        if confidence >= self.config.signals.strong_signal:
            strength = SignalStrength.STRONG
        elif confidence >= self.config.signals.medium_signal:
            strength = SignalStrength.MEDIUM
        else:
            strength = SignalStrength.WEAK

        # Determine direction
        if 'BUY' in prediction['signal']:
            direction = SignalType.BUY
        else:
            direction = SignalType.SELL

        position = OpenPosition(
            signal_id=self.signals_generated,
            entry_price=entry_price,
            entry_time=entry_time,
            direction=direction,
            stop_loss=prediction.get('stop_loss', entry_price * 0.98),
            take_profit=prediction.get('take_profit', entry_price * 1.04),
            strength=strength,
            confidence=confidence,
        )

        self.open_positions.append(position)

        logger.debug(
            f"OPEN {direction.value} @ ${entry_price:.2f} | "
            f"SL: ${position.stop_loss:.2f} | TP: ${position.take_profit:.2f}"
        )

    def _check_exits(self, candle: pd.Series):
        """Check if any open positions should be closed."""
        current_high = candle['high']
        current_low = candle['low']
        current_close = candle['close']
        current_time = candle['datetime']

        positions_to_close = []

        for pos in self.open_positions:
            exit_price = None
            hit_target = False
            hit_stop = False

            if pos.direction == SignalType.BUY:
                # Long position
                if current_low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    hit_stop = True
                elif current_high >= pos.take_profit:
                    exit_price = pos.take_profit
                    hit_target = True

            elif pos.direction == SignalType.SELL:
                # Short position
                if current_high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    hit_stop = True
                elif current_low <= pos.take_profit:
                    exit_price = pos.take_profit
                    hit_target = True

            # Check max hold time
            candles_held = self.current_index - self._get_entry_index(pos)
            if candles_held >= self.bt_config.max_hold_candles and exit_price is None:
                exit_price = current_close
                logger.debug(f"Force exit after {candles_held} candles")

            if exit_price is not None:
                positions_to_close.append((pos, exit_price, hit_target, hit_stop, current_time))

        # Close positions
        for pos, exit_price, hit_target, hit_stop, exit_time in positions_to_close:
            self._close_position(pos, exit_price, exit_time, hit_target, hit_stop)

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        hit_target: bool,
        hit_stop: bool
    ):
        """Close a position and record the trade."""
        # Apply slippage on exit
        slippage = exit_price * (self.bt_config.slippage_percent / 100)
        if pos.direction == SignalType.BUY:
            exit_price -= slippage  # Worse fill for sell
        else:
            exit_price += slippage  # Worse fill for buy cover

        # Calculate PnL
        if pos.direction == SignalType.BUY:
            pnl_percent = ((exit_price - pos.entry_price) / pos.entry_price) * 100
        else:
            pnl_percent = ((pos.entry_price - exit_price) / pos.entry_price) * 100

        # Subtract commission
        pnl_percent -= self.bt_config.commission_percent * 2  # Entry + exit

        is_winner = pnl_percent > 0
        duration_minutes = (exit_time - pos.entry_time).total_seconds() / 60

        trade = {
            'signal_id': pos.signal_id,
            'entry_price': pos.entry_price,
            'entry_time': pos.entry_time,
            'exit_price': exit_price,
            'exit_time': exit_time,
            'direction': pos.direction.value,
            'stop_loss': pos.stop_loss,
            'take_profit': pos.take_profit,
            'strength': pos.strength.value,
            'confidence': pos.confidence,
            'hit_target': hit_target,
            'hit_stop': hit_stop,
            'pnl_percent': pnl_percent,
            'is_winner': is_winner,
            'duration_minutes': duration_minutes,
        }

        self.trades.append(trade)
        self.open_positions.remove(pos)

        result = "WIN" if is_winner else "LOSS"
        reason = "TARGET" if hit_target else "STOP" if hit_stop else "TIMEOUT"

        logger.debug(
            f"CLOSE {pos.direction.value} | {result} ({reason}) | "
            f"PnL: {pnl_percent:+.2f}% | Duration: {duration_minutes:.0f}m"
        )

    def _close_all_positions(self, candle: pd.Series):
        """Close all remaining positions at market price."""
        current_close = candle['close']
        current_time = candle['datetime']

        for pos in self.open_positions[:]:  # Copy list for iteration
            self._close_position(pos, current_close, current_time, False, False)

    def _get_entry_index(self, pos: OpenPosition) -> int:
        """Get the candle index where position was opened."""
        mask = self.df['datetime'] == pos.entry_time
        indices = self.df.index[mask]
        if len(indices) > 0:
            return indices[0]
        return self.current_index

    def get_equity_curve(self) -> List[float]:
        """Get cumulative PnL over time."""
        equity = [0]
        for trade in self.trades:
            equity.append(equity[-1] + trade['pnl_percent'])
        return equity

    def get_trade_details(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)


def run_backtest(
    config_path: str = "config.yaml",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = True
) -> BacktestMetrics:
    """
    Convenience function to run a backtest.

    Args:
        config_path: Path to config file
        start_date: Optional start date (YYYY-MM-DD)
        end_date: Optional end date (YYYY-MM-DD)
        verbose: Whether to print progress

    Returns:
        BacktestMetrics
    """
    bt_config = BacktestConfig()

    if start_date:
        bt_config.start_date = datetime.fromisoformat(start_date)
    if end_date:
        bt_config.end_date = datetime.fromisoformat(end_date)

    engine = BacktestEngine(config_path, bt_config)

    def progress(current, total):
        if verbose:
            pct = current / total * 100 if total > 0 else 0
            print(f"\rBacktesting: {pct:.1f}% ({current}/{total})", end="", flush=True)

    results = engine.run(progress_callback=progress if verbose else None)

    if verbose:
        print("\n")
        print(results.summary())

    return results


# CLI entry point
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    args = parser.parse_args()

    results = run_backtest(
        config_path=args.config,
        start_date=args.start,
        end_date=args.end,
        verbose=not args.quiet
    )
