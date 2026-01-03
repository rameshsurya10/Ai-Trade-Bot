#!/usr/bin/env python3
"""
Paper Trading Validation Runner
================================

Live paper trading validation of the continuous learning system.

Features:
- Real-time market data via WebSocket
- Continuous learning system execution
- Performance monitoring and alerts
- Safety checks and validations
- Comprehensive logging

Usage:
    python run_paper_trading_validation.py --duration 48  # Run for 48 hours
    python run_paper_trading_validation.py --symbols BTC/USDT ETH/USDT
"""

import sys
import signal
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import json

from src.core.database import Database
from src.core.config import load_config
from src.data.provider import UnifiedDataProvider
from src.learning.continuous_learner import ContinuousLearningSystem
from src.advanced_predictor import UnbreakablePredictor
from src.paper_trading import PaperBrokerage
from src.news.collector import NewsCollector
from src.news.sentiment import SentimentAnalyzer
from src.news.aggregator import SentimentAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/paper_trading_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PaperTradingValidator:
    """
    Paper trading validation system.

    Runs continuous learning system with live market data
    and tracks performance for production readiness assessment.
    """

    def __init__(
        self,
        config_path: str = 'config.yaml',
        validation_duration_hours: int = 48
    ):
        """
        Initialize paper trading validator.

        Args:
            config_path: Path to configuration file
            validation_duration_hours: How long to run validation
        """
        self.config = load_config(config_path)
        self.validation_duration = timedelta(hours=validation_duration_hours)
        self.start_time = datetime.utcnow()

        # Results directory
        self.results_dir = Path('validation_results')
        self.results_dir.mkdir(exist_ok=True)

        # Initialize database
        self.db = Database(self.config['database']['path'])

        # Initialize data provider
        self.provider = UnifiedDataProvider.get_instance()

        # Initialize paper brokerage
        self.paper_brokerage = PaperBrokerage(
            initial_capital=self.config['portfolio']['initial_capital'],
            database=self.db
        )

        # Initialize predictor
        self.predictor = UnbreakablePredictor(
            database=self.db,
            config=self.config
        )

        # Initialize continuous learning system
        self.continuous_learner = ContinuousLearningSystem(
            predictor=self.predictor,
            database=self.db,
            paper_brokerage=self.paper_brokerage,
            live_brokerage=None,  # Paper only
            config=self.config
        )

        # Initialize news collection (if enabled)
        self.news_collector = None
        if self.config.get('news', {}).get('enabled', False):
            self.news_collector = NewsCollector(
                database=self.db,
                config=self.config.get('news', {})
            )

        # Validation tracking
        self.validation_stats = {
            'start_time': self.start_time.isoformat(),
            'duration_hours': validation_duration_hours,
            'candles_processed': 0,
            'predictions_made': 0,
            'trades_executed': 0,
            'errors_encountered': 0,
            'mode_transitions': 0,
            'retrainings_triggered': 0,
            'current_mode': 'LEARNING',
            'latest_confidence': 0.0,
            'safety_checks_passed': True
        }

        # Safety thresholds
        self.safety_config = {
            'max_drawdown_percent': 15.0,
            'min_win_rate': 0.45,
            'max_consecutive_losses': 5,
            'max_trades_per_hour': 10
        }

        # Running flag
        self.running = False

        logger.info(
            f"PaperTradingValidator initialized: "
            f"duration={validation_duration_hours}h, "
            f"initial_capital=${self.config['portfolio']['initial_capital']}"
        )

    def run(self, symbols: List[str] = None):
        """
        Run paper trading validation.

        Args:
            symbols: List of symbols to trade (None = from config)
        """
        # Get symbols from config if not provided
        if symbols is None:
            symbols = self.config.get('live_trading', {}).get('default_symbols', ['BTC/USDT'])

        logger.info("="*80)
        logger.info("PAPER TRADING VALIDATION STARTED")
        logger.info("="*80)
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Duration: {self.validation_duration}")
        logger.info(f"End Time: {self.start_time + self.validation_duration}")
        logger.info("="*80)

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Start news collection
            if self.news_collector:
                self.news_collector.start()
                logger.info("News collection started")

            # Subscribe to market data
            self._subscribe_to_symbols(symbols)

            # Register candle close callback
            self.provider.on_candle_closed(self._on_candle_closed)

            # Connect to WebSocket
            logger.info("Connecting to market data...")
            self.provider.connect()

            # Set running flag
            self.running = True

            # Main validation loop
            self._validation_loop()

        except KeyboardInterrupt:
            logger.info("Validation interrupted by user")
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            self.validation_stats['errors_encountered'] += 1
        finally:
            self._cleanup()

    def _subscribe_to_symbols(self, symbols: List[str]):
        """Subscribe to all required timeframes for each symbol."""
        intervals = [
            i['interval']
            for i in self.config['timeframes']['intervals']
            if i.get('enabled', True)
        ]

        for symbol in symbols:
            for interval in intervals:
                self.provider.subscribe(
                    symbol=symbol,
                    interval=interval,
                    exchange=self.config['live_trading']['default_exchange']
                )
                logger.info(f"Subscribed: {symbol} @ {interval}")

    def _on_candle_closed(self, candle: any, interval: str):
        """
        Callback when a candle completes.

        This is the main entry point for the continuous learning system.
        """
        try:
            symbol = candle.symbol

            logger.info(
                f"[{symbol} @ {interval}] Candle closed: "
                f"O={candle.open:.2f} H={candle.high:.2f} "
                f"L={candle.low:.2f} C={candle.close:.2f}"
            )

            # Run continuous learning system
            result = self.continuous_learner.on_candle_closed(
                symbol=symbol,
                interval=interval,
                candle=candle
            )

            # Update validation statistics
            self._update_validation_stats(result)

            # Run safety checks
            self._run_safety_checks()

            # Log result
            self._log_result(result, symbol, interval)

        except Exception as e:
            logger.error(f"Error processing candle: {e}", exc_info=True)
            self.validation_stats['errors_encountered'] += 1

    def _validation_loop(self):
        """Main validation monitoring loop."""
        logger.info("Validation loop started. Press Ctrl+C to stop.")

        last_report_time = datetime.utcnow()
        report_interval = timedelta(hours=1)

        while self.running:
            # Check if validation duration exceeded
            elapsed = datetime.utcnow() - self.start_time
            if elapsed >= self.validation_duration:
                logger.info("Validation duration completed!")
                break

            # Generate periodic reports
            if datetime.utcnow() - last_report_time >= report_interval:
                self._generate_progress_report()
                last_report_time = datetime.utcnow()

            # Sleep to avoid busy waiting
            time.sleep(10)

        # Generate final report
        self._generate_final_report()

    def _update_validation_stats(self, result: dict):
        """Update validation statistics from result."""
        self.validation_stats['candles_processed'] += 1

        if 'aggregated_signal' in result:
            self.validation_stats['predictions_made'] += 1

            # Track mode
            mode = result.get('mode', 'LEARNING')
            if mode != self.validation_stats['current_mode']:
                self.validation_stats['mode_transitions'] += 1
                self.validation_stats['current_mode'] = mode

            # Track confidence
            confidence = result['aggregated_signal'].get('confidence', 0.0)
            self.validation_stats['latest_confidence'] = confidence

        # Track trades
        if result.get('executed'):
            self.validation_stats['trades_executed'] += 1

    def _run_safety_checks(self):
        """Run safety checks to ensure system is operating correctly."""
        try:
            # Get paper brokerage statistics
            portfolio = self.paper_brokerage.get_portfolio_summary()

            # Check 1: Maximum drawdown
            drawdown = portfolio.get('max_drawdown_percent', 0.0)
            if abs(drawdown) > self.safety_config['max_drawdown_percent']:
                logger.warning(
                    f"⚠ SAFETY WARNING: Drawdown {drawdown:.2f}% exceeds limit "
                    f"{self.safety_config['max_drawdown_percent']}%"
                )
                self.validation_stats['safety_checks_passed'] = False

            # Check 2: Minimum win rate
            win_rate = portfolio.get('win_rate', 0.0)
            if portfolio.get('total_trades', 0) > 10 and win_rate < self.safety_config['min_win_rate']:
                logger.warning(
                    f"⚠ SAFETY WARNING: Win rate {win_rate:.2%} below minimum "
                    f"{self.safety_config['min_win_rate']:.2%}"
                )

            # Check 3: Error rate
            error_rate = self.validation_stats['errors_encountered'] / max(self.validation_stats['candles_processed'], 1)
            if error_rate > 0.05:  # >5% error rate
                logger.warning(
                    f"⚠ SAFETY WARNING: Error rate {error_rate:.2%} is high"
                )
                self.validation_stats['safety_checks_passed'] = False

        except Exception as e:
            logger.error(f"Safety check failed: {e}", exc_info=True)

    def _log_result(self, result: dict, symbol: str, interval: str):
        """Log prediction result."""
        if 'aggregated_signal' in result:
            signal = result['aggregated_signal']
            logger.info(
                f"[{symbol} @ {interval}] "
                f"Mode: {result.get('mode', 'UNKNOWN')} | "
                f"Signal: {signal.get('direction', 'NONE')} | "
                f"Confidence: {signal.get('confidence', 0):.2%} | "
                f"Executed: {result.get('executed', False)}"
            )

    def _generate_progress_report(self):
        """Generate periodic progress report."""
        elapsed = datetime.utcnow() - self.start_time
        remaining = self.validation_duration - elapsed

        logger.info("\n" + "="*80)
        logger.info("VALIDATION PROGRESS REPORT")
        logger.info("="*80)
        logger.info(f"Elapsed: {elapsed}")
        logger.info(f"Remaining: {remaining}")
        logger.info(f"Candles Processed: {self.validation_stats['candles_processed']}")
        logger.info(f"Predictions Made: {self.validation_stats['predictions_made']}")
        logger.info(f"Trades Executed: {self.validation_stats['trades_executed']}")
        logger.info(f"Current Mode: {self.validation_stats['current_mode']}")
        logger.info(f"Latest Confidence: {self.validation_stats['latest_confidence']:.2%}")
        logger.info(f"Mode Transitions: {self.validation_stats['mode_transitions']}")
        logger.info(f"Errors: {self.validation_stats['errors_encountered']}")

        # Get portfolio summary
        portfolio = self.paper_brokerage.get_portfolio_summary()
        logger.info(f"\nPortfolio Value: ${portfolio.get('total_value', 0):.2f}")
        logger.info(f"Total P&L: ${portfolio.get('total_pnl', 0):.2f}")
        logger.info(f"Win Rate: {portfolio.get('win_rate', 0):.2%}")
        logger.info("="*80 + "\n")

    def _generate_final_report(self):
        """Generate final validation report."""
        logger.info("\n" + "="*80)
        logger.info("FINAL VALIDATION REPORT")
        logger.info("="*80)

        # Calculate duration
        end_time = datetime.utcnow()
        actual_duration = end_time - self.start_time

        # Get final portfolio summary
        portfolio = self.paper_brokerage.get_portfolio_summary()

        # Get continuous learner statistics
        cl_stats = self.continuous_learner.get_stats()

        # Compile report
        report = {
            'validation_info': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'planned_duration_hours': self.validation_duration.total_seconds() / 3600,
                'actual_duration_hours': actual_duration.total_seconds() / 3600
            },
            'statistics': self.validation_stats,
            'portfolio': portfolio,
            'continuous_learner_stats': cl_stats,
            'safety_status': {
                'all_checks_passed': self.validation_stats['safety_checks_passed'],
                'max_drawdown': portfolio.get('max_drawdown_percent', 0.0),
                'win_rate': portfolio.get('win_rate', 0.0),
                'error_rate': self.validation_stats['errors_encountered'] / max(self.validation_stats['candles_processed'], 1)
            },
            'production_readiness': self._assess_production_readiness(portfolio)
        }

        # Save report
        self._save_report(report)

        # Print summary
        self._print_final_summary(report)

        return report

    def _assess_production_readiness(self, portfolio: dict) -> dict:
        """Assess if system is ready for production."""
        criteria_met = 0
        total_criteria = 5

        criteria = {}

        # Criterion 1: Win rate ≥ 50%
        win_rate = portfolio.get('win_rate', 0.0)
        criteria['win_rate'] = {
            'passed': win_rate >= 0.50,
            'value': win_rate,
            'threshold': 0.50
        }
        if criteria['win_rate']['passed']:
            criteria_met += 1

        # Criterion 2: Max drawdown ≤ 15%
        max_dd = abs(portfolio.get('max_drawdown_percent', 0.0))
        criteria['max_drawdown'] = {
            'passed': max_dd <= 15.0,
            'value': max_dd,
            'threshold': 15.0
        }
        if criteria['max_drawdown']['passed']:
            criteria_met += 1

        # Criterion 3: Error rate < 5%
        error_rate = self.validation_stats['errors_encountered'] / max(self.validation_stats['candles_processed'], 1)
        criteria['error_rate'] = {
            'passed': error_rate < 0.05,
            'value': error_rate,
            'threshold': 0.05
        }
        if criteria['error_rate']['passed']:
            criteria_met += 1

        # Criterion 4: Positive P&L
        total_pnl = portfolio.get('total_pnl', 0.0)
        criteria['profitability'] = {
            'passed': total_pnl > 0,
            'value': total_pnl,
            'threshold': 0.0
        }
        if criteria['profitability']['passed']:
            criteria_met += 1

        # Criterion 5: System stability (mode transitions reasonable)
        transitions_per_hour = self.validation_stats['mode_transitions'] / max((datetime.utcnow() - self.start_time).total_seconds() / 3600, 1)
        criteria['stability'] = {
            'passed': transitions_per_hour < 2.0,  # Less than 2 transitions per hour
            'value': transitions_per_hour,
            'threshold': 2.0
        }
        if criteria['stability']['passed']:
            criteria_met += 1

        # Overall assessment
        if criteria_met >= 4:
            recommendation = "APPROVED"
            message = "✓ System is READY for production deployment"
        elif criteria_met >= 3:
            recommendation = "CONDITIONAL"
            message = "⚠ System shows promise but needs monitoring"
        else:
            recommendation = "NOT_READY"
            message = "✗ System is NOT ready for production"

        return {
            'criteria_met': criteria_met,
            'total_criteria': total_criteria,
            'criteria': criteria,
            'recommendation': recommendation,
            'message': message
        }

    def _save_report(self, report: dict):
        """Save validation report to file."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"validation_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {filename}")

    def _print_final_summary(self, report: dict):
        """Print final validation summary."""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)

        # Duration
        info = report['validation_info']
        print(f"\nDuration: {info['actual_duration_hours']:.1f} hours")

        # Statistics
        stats = report['statistics']
        print(f"\nCandles Processed: {stats['candles_processed']:,}")
        print(f"Predictions Made: {stats['predictions_made']:,}")
        print(f"Trades Executed: {stats['trades_executed']:,}")
        print(f"Mode Transitions: {stats['mode_transitions']}")
        print(f"Errors: {stats['errors_encountered']}")

        # Portfolio
        portfolio = report['portfolio']
        print(f"\nPortfolio Value: ${portfolio.get('total_value', 0):.2f}")
        print(f"Total P&L: ${portfolio.get('total_pnl', 0):.2f}")
        print(f"Win Rate: {portfolio.get('win_rate', 0):.2%}")
        print(f"Max Drawdown: {portfolio.get('max_drawdown_percent', 0):.2f}%")

        # Production Readiness
        readiness = report['production_readiness']
        print(f"\n--- PRODUCTION READINESS ---")
        print(f"Criteria Met: {readiness['criteria_met']}/{readiness['total_criteria']}")

        for name, criterion in readiness['criteria'].items():
            status = "✓" if criterion['passed'] else "✗"
            print(f"  {status} {name}: {criterion['value']:.4f} (threshold: {criterion['threshold']:.4f})")

        print(f"\n{readiness['message']}")
        print("="*80 + "\n")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up...")

        # Stop news collector
        if self.news_collector:
            self.news_collector.stop()

        # Stop continuous learner
        if hasattr(self, 'continuous_learner'):
            self.continuous_learner.stop()

        # Disconnect data provider
        if hasattr(self, 'provider'):
            self.provider.disconnect()

        logger.info("Cleanup complete")


def main():
    """Run paper trading validation from command line."""
    parser = argparse.ArgumentParser(description='Paper trading validation')
    parser.add_argument('--duration', type=int, default=48, help='Validation duration in hours')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Create validator
    validator = PaperTradingValidator(
        config_path=args.config,
        validation_duration_hours=args.duration
    )

    # Run validation
    try:
        validator.run(symbols=args.symbols)
        return 0
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
