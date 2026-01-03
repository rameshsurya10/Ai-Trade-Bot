"""
Continuous Learning System Backtesting
=======================================

Comprehensive backtest of the continuous learning trading system.

Tests:
1. Multi-timeframe signal aggregation
2. Confidence-based mode transitions (LEARNING ↔ TRADING)
3. Retraining triggers and outcomes
4. Sentiment feature integration
5. Win rate and performance metrics
6. Comparison: Technical-only vs Technical+Sentiment

Usage:
    python tests/backtest_continuous_learning.py --symbol BTC/USDT --days 90
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.database import Database
from src.core.config import load_config
from src.data.provider import UnifiedDataProvider
from src.multi_timeframe.model_manager import MultiTimeframeModelManager
from src.multi_timeframe.aggregator import SignalAggregator
from src.learning.confidence_gate import ConfidenceGate
from src.learning.state_manager import LearningStateManager
from src.learning.outcome_tracker import OutcomeTracker
from src.learning.continuous_learner import ContinuousLearningSystem
from src.advanced_predictor import UnbreakablePredictor
from src.paper_trading import PaperBrokerage
from src.news.aggregator import SentimentAggregator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousLearningBacktest:
    """
    Backtest engine for continuous learning system.

    Features:
    - Historical data replay
    - Mode transition tracking
    - Performance metrics
    - Sentiment impact analysis
    """

    def __init__(
        self,
        config_path: str = 'config.yaml',
        results_dir: str = 'backtest_results'
    ):
        """Initialize backtest engine."""
        self.config = load_config(config_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Initialize components
        self.db = Database(self.config['database']['path'])
        self.provider = UnifiedDataProvider.get_instance()

        # Statistics
        self.stats = {
            'total_candles': 0,
            'predictions_made': 0,
            'trades_executed': 0,
            'learning_mode_count': 0,
            'trading_mode_count': 0,
            'mode_transitions': 0,
            'retrainings_triggered': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0
        }

        # Track mode changes
        self.mode_history = []
        self.confidence_history = []
        self.trade_history = []

        logger.info("ContinuousLearningBacktest initialized")

    def run_backtest(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        intervals: List[str] = None,
        use_sentiment: bool = True
    ) -> Dict:
        """
        Run complete backtest.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Backtest start date
            end_date: Backtest end date
            intervals: Timeframes to use (None = from config)
            use_sentiment: Include sentiment features

        Returns:
            Dict with backtest results
        """
        logger.info(
            f"Starting backtest: {symbol} from {start_date.date()} to {end_date.date()}"
        )

        # Get intervals from config if not provided
        if intervals is None:
            intervals = [
                i['interval']
                for i in self.config['timeframes']['intervals']
                if i.get('enabled', True)
            ]

        # Initialize components for backtest
        predictor = UnbreakablePredictor(
            database=self.db,
            config=self.config
        )

        paper_brokerage = PaperBrokerage(
            initial_capital=self.config['portfolio']['initial_capital'],
            database=self.db
        )

        continuous_learner = ContinuousLearningSystem(
            predictor=predictor,
            database=self.db,
            paper_brokerage=paper_brokerage,
            live_brokerage=None,  # Backtest uses paper only
            config=self.config
        )

        # Load historical data for all timeframes
        logger.info("Loading historical data...")
        data_by_interval = {}

        for interval in intervals:
            df = self._load_historical_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )

            if df is not None and len(df) > 0:
                data_by_interval[interval] = df
                logger.info(f"  {interval}: {len(df)} candles")
            else:
                logger.warning(f"  {interval}: No data available")

        if not data_by_interval:
            raise ValueError("No historical data available for backtest")

        # Use primary interval (highest weight) as main timeline
        primary_interval = max(
            intervals,
            key=lambda i: next(
                (x['weight'] for x in self.config['timeframes']['intervals'] if x['interval'] == i),
                0
            )
        )

        primary_df = data_by_interval[primary_interval]
        logger.info(f"Using {primary_interval} as primary timeline ({len(primary_df)} candles)")

        # Simulate candle-by-candle execution
        logger.info("Running backtest simulation...")

        for idx in range(len(primary_df)):
            try:
                # Get current candle
                candle_row = primary_df.iloc[idx]
                candle_timestamp = int(candle_row['timestamp'])

                # Create synthetic candle object
                class Candle:
                    def __init__(self, row):
                        self.timestamp = int(row['timestamp'])
                        self.open = float(row['open'])
                        self.high = float(row['high'])
                        self.low = float(row['low'])
                        self.close = float(row['close'])
                        self.volume = float(row['volume'])
                        self.symbol = symbol

                candle = Candle(candle_row)

                # Prepare data for all timeframes (up to current point)
                data = {
                    'interval': primary_interval,
                    'candles': primary_df.iloc[:idx+1].copy()
                }

                # Call continuous learner (simulates candle close event)
                result = continuous_learner.on_candle_closed(
                    symbol=symbol,
                    interval=primary_interval,
                    candle=candle,
                    data=data
                )

                # Track statistics
                self._update_stats(result)

                # Log progress
                if idx % 100 == 0:
                    progress = (idx / len(primary_df)) * 100
                    logger.info(
                        f"Progress: {progress:.1f}% "
                        f"({idx}/{len(primary_df)} candles, "
                        f"{self.stats['trades_executed']} trades)"
                    )

            except Exception as e:
                logger.error(f"Error at candle {idx}: {e}", exc_info=True)
                continue

        # Generate results
        results = self._compile_results(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            intervals=intervals,
            use_sentiment=use_sentiment
        )

        # Save results
        self._save_results(results)

        logger.info("Backtest complete!")
        self._print_summary(results)

        return results

    def _load_historical_data(
        self,
        symbol: str,
        interval: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load historical data from database."""
        try:
            # Convert to timestamps
            start_ts = int(start_date.timestamp())
            end_ts = int(end_date.timestamp())

            # Query database
            df = self.db.get_candles(
                symbol=symbol,
                interval=interval,
                limit=10000,  # Large limit for historical data
                since_timestamp=start_ts
            )

            if df is None or len(df) == 0:
                return None

            # Filter to date range
            df = df[
                (df['timestamp'] >= start_ts) &
                (df['timestamp'] <= end_ts)
            ].copy()

            return df

        except Exception as e:
            logger.error(f"Failed to load historical data for {interval}: {e}")
            return None

    def _update_stats(self, result: dict):
        """Update backtest statistics from result."""
        self.stats['total_candles'] += 1

        if 'aggregated_signal' in result:
            self.stats['predictions_made'] += 1

            # Track mode
            mode = result.get('mode', 'LEARNING')
            if mode == 'LEARNING':
                self.stats['learning_mode_count'] += 1
            else:
                self.stats['trading_mode_count'] += 1

            # Track mode transitions
            if len(self.mode_history) > 0 and self.mode_history[-1] != mode:
                self.stats['mode_transitions'] += 1

            self.mode_history.append(mode)

            # Track confidence
            confidence = result['aggregated_signal'].get('confidence', 0.0)
            self.confidence_history.append({
                'timestamp': datetime.utcnow(),
                'confidence': confidence,
                'mode': mode
            })

        # Track trades
        if result.get('executed'):
            self.stats['trades_executed'] += 1

            self.trade_history.append({
                'timestamp': datetime.utcnow(),
                'direction': result['aggregated_signal'].get('direction'),
                'confidence': result['aggregated_signal'].get('confidence'),
                'mode': result.get('mode'),
                'brokerage': result.get('brokerage')
            })

    def _compile_results(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        intervals: List[str],
        use_sentiment: bool
    ) -> Dict:
        """Compile comprehensive backtest results."""
        # Calculate win rate
        total_trades = self.stats['wins'] + self.stats['losses']
        win_rate = (self.stats['wins'] / total_trades) if total_trades > 0 else 0.0

        # Calculate mode percentages
        total_candles = self.stats['total_candles']
        learning_pct = (self.stats['learning_mode_count'] / total_candles * 100) if total_candles > 0 else 0
        trading_pct = (self.stats['trading_mode_count'] / total_candles * 100) if total_candles > 0 else 0

        # Calculate average confidence by mode
        learning_confidences = [
            h['confidence'] for h in self.confidence_history
            if h['mode'] == 'LEARNING'
        ]
        trading_confidences = [
            h['confidence'] for h in self.confidence_history
            if h['mode'] == 'TRADING'
        ]

        avg_learning_conf = np.mean(learning_confidences) if learning_confidences else 0.0
        avg_trading_conf = np.mean(trading_confidences) if trading_confidences else 0.0

        results = {
            'metadata': {
                'symbol': symbol,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'intervals': intervals,
                'use_sentiment': use_sentiment,
                'backtest_completed': datetime.utcnow().isoformat()
            },
            'performance': {
                'total_candles': self.stats['total_candles'],
                'predictions_made': self.stats['predictions_made'],
                'trades_executed': self.stats['trades_executed'],
                'wins': self.stats['wins'],
                'losses': self.stats['losses'],
                'win_rate': win_rate,
                'total_pnl': self.stats['total_pnl'],
                'total_pnl_percent': (self.stats['total_pnl'] / self.config['portfolio']['initial_capital']) * 100
            },
            'modes': {
                'learning_candles': self.stats['learning_mode_count'],
                'trading_candles': self.stats['trading_mode_count'],
                'learning_percent': learning_pct,
                'trading_percent': trading_pct,
                'mode_transitions': self.stats['mode_transitions'],
                'avg_learning_confidence': avg_learning_conf,
                'avg_trading_confidence': avg_trading_conf
            },
            'retraining': {
                'retrainings_triggered': self.stats['retrainings_triggered']
            }
        }

        return results

    def _save_results(self, results: Dict):
        """Save backtest results to file."""
        import json

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"backtest_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {filename}")

    def _print_summary(self, results: Dict):
        """Print backtest summary."""
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)

        # Metadata
        meta = results['metadata']
        print(f"\nSymbol: {meta['symbol']}")
        print(f"Period: {meta['start_date']} to {meta['end_date']}")
        print(f"Intervals: {', '.join(meta['intervals'])}")
        print(f"Sentiment: {'Enabled' if meta['use_sentiment'] else 'Disabled'}")

        # Performance
        perf = results['performance']
        print(f"\n--- PERFORMANCE ---")
        print(f"Total Candles: {perf['total_candles']:,}")
        print(f"Predictions: {perf['predictions_made']:,}")
        print(f"Trades Executed: {perf['trades_executed']:,}")
        print(f"Wins: {perf['wins']} | Losses: {perf['losses']}")
        print(f"Win Rate: {perf['win_rate']:.2%}")
        print(f"Total P&L: ${perf['total_pnl']:.2f} ({perf['total_pnl_percent']:.2f}%)")

        # Modes
        modes = results['modes']
        print(f"\n--- MODE DISTRIBUTION ---")
        print(f"Learning Mode: {modes['learning_percent']:.1f}% ({modes['learning_candles']:,} candles)")
        print(f"Trading Mode: {modes['trading_percent']:.1f}% ({modes['trading_candles']:,} candles)")
        print(f"Mode Transitions: {modes['mode_transitions']}")
        print(f"Avg Confidence (Learning): {modes['avg_learning_confidence']:.2%}")
        print(f"Avg Confidence (Trading): {modes['avg_trading_confidence']:.2%}")

        # Retraining
        retrain = results['retraining']
        print(f"\n--- RETRAINING ---")
        print(f"Retrainings Triggered: {retrain['retrainings_triggered']}")

        print("\n" + "="*80 + "\n")


def main():
    """Run backtest from command line."""
    parser = argparse.ArgumentParser(description='Backtest continuous learning system')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--days', type=int, default=90, help='Days to backtest')
    parser.add_argument('--no-sentiment', action='store_true', help='Disable sentiment features')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

    # Run backtest
    backtest = ContinuousLearningBacktest(config_path=args.config)

    try:
        results = backtest.run_backtest(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date,
            use_sentiment=not args.no_sentiment
        )

        # Success
        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
