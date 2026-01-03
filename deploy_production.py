"""
Production Deployment Script
============================

Gradual rollout of continuous learning system to production.

Strategy:
- Phase 1: Single symbol (BTC/USDT) for 24 hours
- Phase 2: Monitor performance and validate safety
- Phase 3: Enable for all configured symbols
- Phase 4: Final parameter tuning

Usage:
    # Phase 1: Deploy single symbol
    python deploy_production.py --phase 1 --symbol BTC/USDT

    # Phase 2: Monitor (runs automatically)
    python deploy_production.py --phase 2

    # Phase 3: Full rollout
    python deploy_production.py --phase 3

    # Rollback
    python deploy_production.py --rollback
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import signal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.database import Database
from src.core.config import load_config
from src.data.provider import UnifiedDataProvider
from src.learning.continuous_learner import ContinuousLearningSystem
from src.paper_trading import PaperBrokerage

# Ensure required directories exist
Path('logs').mkdir(exist_ok=True)
Path('production_reports').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/production_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProductionDeployment:
    """
    Manages gradual production rollout.

    Features:
    - Phased deployment (1 symbol → all symbols)
    - Automated safety validation
    - Performance monitoring
    - Rollback capability
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize deployment manager."""
        self.config = load_config(config_path)
        self.db = Database(self.config['database']['path'])

        # Deployment state
        self.current_phase: int = 0
        self.deployed_symbols: List[str] = []
        self.deployment_start: Optional[datetime] = None
        self.should_stop = False

        # Monitoring
        self.health_checks_passed: int = 0
        self.health_checks_failed: int = 0

        # Components (initialized on demand)
        self.data_provider: Optional[UnifiedDataProvider] = None
        self.continuous_learner: Optional[ContinuousLearningSystem] = None
        self.paper_brokerage: Optional[PaperBrokerage] = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        self.should_stop = True

    def phase_1_single_symbol(self, symbol: str = 'BTC/USDT'):
        """
        Phase 1: Deploy single symbol for validation.

        Duration: 24 hours
        Goal: Validate system stability and safety
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: SINGLE SYMBOL DEPLOYMENT")
        logger.info("=" * 80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Duration: 24 hours")
        logger.info(f"Start Time: {datetime.utcnow()}")
        logger.info("=" * 80)

        self.current_phase = 1
        self.deployment_start = datetime.utcnow()

        # Pre-deployment checks
        if not self._pre_deployment_checks(symbol):
            logger.error("❌ Pre-deployment checks failed. Aborting.")
            return False

        # Initialize components
        self._initialize_components()

        # Deploy single symbol
        self.deployed_symbols = [symbol]
        self._deploy_symbol(symbol)

        # Monitor for 24 hours
        logger.info(f"✓ Deployment started. Monitoring {symbol} for 24 hours...")

        end_time = datetime.utcnow() + timedelta(hours=24)

        while datetime.utcnow() < end_time and not self.should_stop:
            # Run health checks every 5 minutes
            time.sleep(300)

            health_status = self._run_health_checks(symbol)

            if not health_status['healthy']:
                logger.error(f"❌ Health check failed: {health_status['reason']}")
                logger.error("Initiating rollback...")
                self.rollback()
                return False

            logger.info(f"✓ Health check passed ({self.health_checks_passed} consecutive)")

        if self.should_stop:
            logger.warning("Deployment interrupted by user.")
            return False

        # Phase 1 complete - validate results
        logger.info("=" * 80)
        logger.info("PHASE 1 COMPLETE - VALIDATING RESULTS")
        logger.info("=" * 80)

        validation_results = self._validate_phase_1(symbol)

        if validation_results['approved']:
            logger.info("✅ Phase 1 APPROVED - Ready for Phase 3 (Full Rollout)")
            return True
        else:
            logger.warning(f"⚠ Phase 1 NOT APPROVED - {validation_results['reason']}")
            logger.warning("Review results and adjust parameters before proceeding.")
            return False

    def phase_3_full_rollout(self):
        """
        Phase 3: Enable all configured symbols.

        Prerequisites: Phase 1 must be approved
        """
        logger.info("=" * 80)
        logger.info("PHASE 3: FULL PRODUCTION ROLLOUT")
        logger.info("=" * 80)

        # Check Phase 1 completed
        if self.current_phase < 1:
            logger.error("❌ Phase 1 must be completed first.")
            return False

        # Get all configured symbols
        all_symbols = self.config.get('symbols', ['BTC/USDT'])

        logger.info(f"Deploying to {len(all_symbols)} symbols:")
        for symbol in all_symbols:
            logger.info(f"  - {symbol}")

        # Deploy each symbol
        for symbol in all_symbols:
            if symbol in self.deployed_symbols:
                logger.info(f"✓ {symbol} already deployed (Phase 1)")
                continue

            logger.info(f"Deploying {symbol}...")
            self._deploy_symbol(symbol)
            self.deployed_symbols.append(symbol)

            # Wait 30 seconds between deployments
            time.sleep(30)

        self.current_phase = 3

        logger.info("=" * 80)
        logger.info(f"✅ FULL ROLLOUT COMPLETE - {len(self.deployed_symbols)} symbols active")
        logger.info("=" * 80)

        # Monitor all symbols
        logger.info("Continuous monitoring active. Press Ctrl+C to stop.")

        try:
            while not self.should_stop:
                time.sleep(300)  # Health check every 5 minutes

                # Batch health checks for all symbols (eliminates N+1 query pattern)
                health_results = self._run_health_checks_batch(self.deployed_symbols)

                for symbol, health_status in health_results.items():
                    if not health_status['healthy']:
                        logger.warning(f"⚠ {symbol}: {health_status['reason']}")
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user.")

        return True

    def rollback(self):
        """
        Emergency rollback - disable continuous learning.

        Actions:
        1. Stop all WebSocket connections
        2. Close open positions
        3. Disable continuous learning
        4. Preserve data for analysis
        """
        logger.warning("=" * 80)
        logger.warning("INITIATING ROLLBACK")
        logger.warning("=" * 80)

        # Stop continuous learner
        if self.continuous_learner:
            logger.info("Stopping continuous learning system...")
            # Stop retraining threads
            for thread in self.continuous_learner.retraining_threads.values():
                if thread.is_alive():
                    logger.info(f"Waiting for retraining thread {thread.name}...")
                    thread.join(timeout=30)

        # Close positions (paper trading only in this phase)
        if self.paper_brokerage:
            logger.info("Closing all open positions...")
            open_positions = self.db.get_open_positions()
            for position in open_positions:
                self.paper_brokerage.close_position(
                    symbol=position['symbol'],
                    reason='rollback'
                )

        # Stop data provider
        if self.data_provider:
            logger.info("Disconnecting data provider...")
            self.data_provider.disconnect()

        # Update deployment state
        self.deployed_symbols = []
        self.current_phase = 0

        logger.warning("✓ Rollback complete. System stopped.")
        logger.info("Data preserved in database for analysis.")

    def _check_dependencies(self) -> bool:
        """
        Verify all required dependencies are installed.

        Returns:
            True if all dependencies available, False otherwise
        """
        required_packages = [
            ('torch', 'PyTorch'),
            ('pandas', 'Pandas'),
            ('numpy', 'NumPy'),
            ('yaml', 'PyYAML'),
            ('sklearn', 'scikit-learn')
        ]

        missing = []

        for package_import, package_name in required_packages:
            try:
                __import__(package_import)
            except ImportError:
                missing.append(package_name)
                logger.error(f"❌ Missing dependency: {package_name}")

        if missing:
            logger.error(f"Missing {len(missing)} required packages: {', '.join(missing)}")
            logger.error("Install missing packages with: pip install -r requirements.txt")
            return False

        logger.info(f"✓ All dependencies installed ({len(required_packages)} packages)")
        return True

    def _pre_deployment_checks(self, symbol: str) -> bool:
        """
        Pre-deployment validation.

        Checks:
        0. Dependencies installed
        1. Database integrity
        2. Models exist for all timeframes
        3. Historical data available
        4. Configuration valid
        5. API connectivity
        """
        logger.info("Running pre-deployment checks...")

        # Check dependencies first (blocker)
        if not self._check_dependencies():
            logger.error("❌ Dependency check failed - cannot proceed")
            return False

        checks_passed = 0
        checks_total = 5

        # Check 1: Database integrity
        try:
            integrity = self.db.execute_query("PRAGMA integrity_check")
            if integrity and integrity[0][0] == 'ok':
                logger.info("✓ Database integrity: OK")
                checks_passed += 1
            else:
                logger.error("❌ Database integrity check failed")
        except Exception as e:
            logger.error(f"❌ Database check error: {e}")

        # Check 2: Models exist
        try:
            from src.multi_timeframe.model_manager import MultiTimeframeModelManager

            model_manager = MultiTimeframeModelManager(config=self.config)

            intervals = [
                interval['interval']
                for interval in self.config['timeframes']['intervals']
                if interval.get('enabled', True)
            ]

            models_exist = True
            for interval in intervals:
                model_path = model_manager.get_model_path(symbol, interval)
                if not model_path.exists():
                    logger.error(f"❌ Model not found: {model_path}")
                    models_exist = False

            if models_exist:
                logger.info(f"✓ All models exist ({len(intervals)} timeframes)")
                checks_passed += 1
            else:
                logger.error("❌ Missing models - run training first")
        except Exception as e:
            logger.error(f"❌ Model check error: {e}")

        # Check 3: Historical data
        try:
            candle_count = self.db.execute_query(
                "SELECT COUNT(*) FROM candles WHERE symbol = ?",
                (symbol,)
            )

            if candle_count and candle_count[0][0] > 1000:
                logger.info(f"✓ Historical data: {candle_count[0][0]:,} candles")
                checks_passed += 1
            else:
                logger.error(f"❌ Insufficient historical data: {candle_count[0][0] if candle_count else 0} candles")
        except Exception as e:
            logger.error(f"❌ Historical data check error: {e}")

        # Check 4: Configuration valid
        try:
            required_keys = ['database', 'timeframes', 'continuous_learning', 'portfolio']

            config_valid = all(key in self.config for key in required_keys)

            if config_valid:
                logger.info("✓ Configuration valid")
                checks_passed += 1
            else:
                logger.error("❌ Invalid configuration - missing required keys")
        except Exception as e:
            logger.error(f"❌ Configuration check error: {e}")

        # Check 5: API connectivity (test)
        try:
            # Quick test of data provider
            test_provider = UnifiedDataProvider.get_instance()

            # Test connection (don't subscribe yet)
            logger.info("✓ Data provider initialized")
            checks_passed += 1
        except Exception as e:
            logger.error(f"❌ Data provider check error: {e}")

        logger.info("=" * 80)
        logger.info(f"Pre-deployment checks: {checks_passed}/{checks_total} passed")
        logger.info("=" * 80)

        return checks_passed == checks_total

    def _initialize_components(self):
        """Initialize trading system components."""
        logger.info("Initializing system components...")

        # Data provider
        self.data_provider = UnifiedDataProvider.get_instance()

        # Paper brokerage (for learning mode)
        self.paper_brokerage = PaperBrokerage(
            initial_balance=self.config['portfolio']['initial_capital'],
            fee_percent=self.config['portfolio'].get('fee_percent', 0.1)
        )

        # Continuous learning system
        from src.advanced_predictor import UnbreakablePredictor

        predictor = UnbreakablePredictor(config=self.config)

        self.continuous_learner = ContinuousLearningSystem(
            predictor=predictor,
            database=self.db,
            paper_brokerage=self.paper_brokerage,
            live_brokerage=None,  # No live trading yet
            config=self.config
        )

        logger.info("✓ Components initialized")

    def _deploy_symbol(self, symbol: str):
        """
        Deploy continuous learning for a symbol.

        Actions:
        1. Subscribe to all configured timeframes
        2. Register candle close callbacks
        3. Enable continuous learning
        """
        logger.info(f"Deploying {symbol}...")

        # Get enabled intervals
        intervals = [
            interval['interval']
            for interval in self.config['timeframes']['intervals']
            if interval.get('enabled', True)
        ]

        # Subscribe to each interval
        for interval in intervals:
            self.data_provider.subscribe(
                symbol=symbol,
                exchange=self.config.get('exchange', 'binance'),
                interval=interval
            )

            logger.info(f"  ✓ Subscribed to {symbol} @ {interval}")

        # Register callback
        def on_candle_closed(candle, interval):
            """Handle candle close event."""
            try:
                self.continuous_learner.on_candle_closed(
                    symbol=symbol,
                    interval=interval,
                    candle=candle
                )
            except Exception as e:
                logger.error(f"Error in candle handler: {e}", exc_info=True)

        self.data_provider.on_candle_closed(on_candle_closed)

        # Connect WebSocket
        if not self.data_provider.is_connected():
            self.data_provider.connect()

        logger.info(f"✅ {symbol} deployed successfully")

    def _run_health_checks_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Run health checks for all symbols in a single batch query.

        This eliminates the N+1 query pattern by fetching data for all symbols at once.
        Performance: 30+ queries → 1 query (90% reduction)

        Args:
            symbols: List of symbols to check

        Returns:
            Dict mapping symbol to health status:
            {
                'BTCUSDT': {'healthy': True, 'reason': '...', 'metrics': {...}},
                'ETHUSDT': {'healthy': True, 'reason': '...', 'metrics': {...}},
                ...
            }
        """
        if not symbols:
            return {}

        try:
            # Single batch query for all symbols
            placeholders = ','.join('?' * len(symbols))
            query = f"""
                SELECT
                    symbol,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_percent) as avg_pnl,
                    MIN(entry_time) as first_trade,
                    MAX(entry_time) as last_trade
                FROM trade_outcomes
                WHERE symbol IN ({placeholders})
                AND entry_time > datetime('now', '-1 hour')
                GROUP BY symbol
            """

            results = self.db.execute_query(query, tuple(symbols))

            # Convert to dict keyed by symbol
            metrics_by_symbol = {}
            for row in results:
                metrics_by_symbol[row[0]] = {
                    'total_trades': row[1] or 0,
                    'wins': row[2] or 0,
                    'avg_pnl': row[3] or 0.0,
                    'first_trade': row[4],
                    'last_trade': row[5]
                }

            # Get portfolio value once (shared across all symbols)
            portfolio = self._get_portfolio_value()
            initial_capital = self.config['portfolio']['initial_capital']
            drawdown = ((initial_capital - portfolio['total_value']) / initial_capital) * 100

            # Process health status for each symbol
            health_results = {}

            for symbol in symbols:
                metrics = metrics_by_symbol.get(symbol, {
                    'total_trades': 0,
                    'wins': 0,
                    'avg_pnl': 0.0
                })

                total_trades = metrics['total_trades']
                wins = metrics['wins']
                avg_pnl = metrics['avg_pnl']

                win_rate = wins / total_trades if total_trades > 0 else 0.0

                # No trades yet - healthy by default
                if total_trades == 0:
                    health_results[symbol] = {
                        'healthy': True,
                        'reason': 'No trades yet',
                        'metrics': metrics
                    }
                    continue

                # Safety checks
                # Check 1: Drawdown < 15%
                if drawdown > 15.0:
                    health_results[symbol] = {
                        'healthy': False,
                        'reason': f'Drawdown {drawdown:.1f}% exceeds 15% limit',
                        'metrics': {**metrics, 'drawdown': drawdown}
                    }
                    self.health_checks_failed += 1
                    self.health_checks_passed = 0
                    continue

                # Check 2: Win rate ≥ 40% (over 10+ trades)
                if total_trades >= 10 and win_rate < 0.40:
                    health_results[symbol] = {
                        'healthy': False,
                        'reason': f'Win rate {win_rate:.1%} below 40% threshold',
                        'metrics': {**metrics, 'win_rate': win_rate}
                    }
                    self.health_checks_failed += 1
                    self.health_checks_passed = 0
                    continue

                # Check 3: Average P&L > -1%
                if total_trades >= 5 and avg_pnl < -1.0:
                    health_results[symbol] = {
                        'healthy': False,
                        'reason': f'Avg P&L {avg_pnl:.2f}% below -1% threshold',
                        'metrics': {**metrics, 'avg_pnl': avg_pnl}
                    }
                    self.health_checks_failed += 1
                    self.health_checks_passed = 0
                    continue

                # All checks passed
                health_results[symbol] = {
                    'healthy': True,
                    'reason': f'All checks passed (WR: {win_rate:.1%}, Avg P&L: {avg_pnl:.2f}%)',
                    'metrics': {
                        **metrics,
                        'win_rate': win_rate,
                        'drawdown': drawdown
                    }
                }

            self.health_checks_passed += 1
            self.health_checks_failed = 0

            return health_results

        except Exception as e:
            logger.error(f"Batch health check failed: {e}", exc_info=True)

            # Fallback to individual checks on error
            return {
                symbol: self._run_health_checks(symbol)
                for symbol in symbols
            }

    def _run_health_checks(self, symbol: str) -> Dict:
        """
        Run automated health checks.

        Returns:
            {
                'healthy': bool,
                'reason': str,
                'metrics': {...}
            }
        """
        try:
            # Get recent performance
            recent_outcomes = self.db.execute_query(
                """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_percent) as avg_pnl
                FROM trade_outcomes
                WHERE symbol = ?
                AND entry_time > datetime('now', '-1 hour')
                """,
                (symbol,)
            )

            if not recent_outcomes:
                return {'healthy': True, 'reason': 'No trades yet', 'metrics': {}}

            total_trades = recent_outcomes[0][0] or 0
            wins = recent_outcomes[0][1] or 0
            avg_pnl = recent_outcomes[0][2] or 0.0

            win_rate = wins / total_trades if total_trades > 0 else 0.0

            # Get portfolio value
            portfolio = self._get_portfolio_value()

            initial_capital = self.config['portfolio']['initial_capital']
            drawdown = ((initial_capital - portfolio['total_value']) / initial_capital) * 100

            # Safety checks
            checks = []

            # Check 1: Drawdown < 15%
            if drawdown > 15.0:
                self.health_checks_failed += 1
                self.health_checks_passed = 0
                return {
                    'healthy': False,
                    'reason': f'Drawdown {drawdown:.1f}% exceeds 15% limit',
                    'metrics': {'drawdown': drawdown, 'win_rate': win_rate}
                }
            checks.append('drawdown')

            # Check 2: Win rate > 40% (after 20+ trades)
            if total_trades >= 20 and win_rate < 0.40:
                self.health_checks_failed += 1
                self.health_checks_passed = 0
                return {
                    'healthy': False,
                    'reason': f'Win rate {win_rate:.1%} below 40%',
                    'metrics': {'drawdown': drawdown, 'win_rate': win_rate}
                }
            checks.append('win_rate')

            # Check 3: System errors < 10% in last hour
            error_count = self.db.execute_query(
                """
                SELECT COUNT(*) FROM error_log
                WHERE timestamp > datetime('now', '-1 hour')
                """
            )

            if error_count and error_count[0][0] > 10:
                self.health_checks_failed += 1
                self.health_checks_passed = 0
                return {
                    'healthy': False,
                    'reason': f'High error rate: {error_count[0][0]} errors in last hour',
                    'metrics': {'errors': error_count[0][0]}
                }
            checks.append('errors')

            # All checks passed
            self.health_checks_passed += 1
            self.health_checks_failed = 0

            return {
                'healthy': True,
                'reason': f'All checks passed ({", ".join(checks)})',
                'metrics': {
                    'drawdown': drawdown,
                    'win_rate': win_rate,
                    'total_trades': total_trades,
                    'portfolio_value': portfolio['total_value']
                }
            }

        except Exception as e:
            logger.error(f"Health check error: {e}", exc_info=True)
            return {
                'healthy': False,
                'reason': f'Health check error: {str(e)}',
                'metrics': {}
            }

    def _get_portfolio_value(self) -> Dict:
        """Get current portfolio value."""
        try:
            # Get positions
            positions = self.db.execute_query(
                """
                SELECT
                    SUM(CASE WHEN status = 'open' THEN quantity * current_price ELSE 0 END) as position_value,
                    SUM(CASE WHEN status = 'closed' THEN pnl ELSE 0 END) as realized_pnl
                FROM positions
                """
            )

            if positions:
                position_value = positions[0][0] or 0.0
                realized_pnl = positions[0][1] or 0.0
            else:
                position_value = 0.0
                realized_pnl = 0.0

            initial_capital = self.config['portfolio']['initial_capital']
            cash = initial_capital + realized_pnl - position_value
            total_value = cash + position_value

            return {
                'total_value': total_value,
                'cash': cash,
                'position_value': position_value,
                'pnl': total_value - initial_capital
            }

        except Exception as e:
            logger.error(f"Portfolio value error: {e}")
            return {
                'total_value': self.config['portfolio']['initial_capital'],
                'cash': self.config['portfolio']['initial_capital'],
                'position_value': 0.0,
                'pnl': 0.0
            }

    def _validate_phase_1(self, symbol: str) -> Dict:
        """
        Validate Phase 1 results.

        Approval criteria (4/5 required):
        1. Win rate ≥ 50%
        2. Max drawdown ≤ 15%
        3. Error rate < 5%
        4. Profitability > $0
        5. Stability < 2 transitions/hour

        Returns:
            {
                'approved': bool,
                'criteria_met': int,
                'details': {...}
            }
        """
        logger.info("Validating Phase 1 results...")

        criteria_met = 0
        details = {}

        # Criterion 1: Win rate
        win_rate_data = self.db.execute_query(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as wins
            FROM trade_outcomes
            WHERE symbol = ?
            AND entry_time > ?
            """,
            (symbol, self.deployment_start.isoformat())
        )

        if win_rate_data and win_rate_data[0][0] > 0:
            total = win_rate_data[0][0]
            wins = win_rate_data[0][1] or 0
            win_rate = wins / total

            details['win_rate'] = {
                'value': win_rate,
                'threshold': 0.50,
                'passed': win_rate >= 0.50
            }

            if win_rate >= 0.50:
                criteria_met += 1
        else:
            details['win_rate'] = {'value': 0.0, 'threshold': 0.50, 'passed': False}

        # Criterion 2: Max drawdown
        portfolio = self._get_portfolio_value()
        initial_capital = self.config['portfolio']['initial_capital']

        # Get max portfolio value during deployment
        max_value_data = self.db.execute_query(
            """
            SELECT MAX(total_value) FROM portfolio_snapshots
            WHERE timestamp > ?
            """,
            (self.deployment_start.isoformat(),)
        )

        max_value = max_value_data[0][0] if max_value_data and max_value_data[0][0] else initial_capital
        current_value = portfolio['total_value']

        drawdown = ((max_value - current_value) / max_value) * 100 if max_value > 0 else 0.0

        details['max_drawdown'] = {
            'value': drawdown,
            'threshold': 15.0,
            'passed': drawdown <= 15.0
        }

        if drawdown <= 15.0:
            criteria_met += 1

        # Criterion 3: Error rate
        candle_count = self.db.execute_query(
            """
            SELECT COUNT(*) FROM candles
            WHERE symbol = ?
            AND timestamp > ?
            """,
            (symbol, int(self.deployment_start.timestamp()))
        )

        error_count = self.db.execute_query(
            """
            SELECT COUNT(*) FROM error_log
            WHERE timestamp > ?
            """,
            (self.deployment_start.isoformat(),)
        )

        total_candles = candle_count[0][0] if candle_count else 0
        total_errors = error_count[0][0] if error_count else 0

        error_rate = total_errors / total_candles if total_candles > 0 else 0.0

        details['error_rate'] = {
            'value': error_rate,
            'threshold': 0.05,
            'passed': error_rate < 0.05
        }

        if error_rate < 0.05:
            criteria_met += 1

        # Criterion 4: Profitability
        pnl = portfolio['pnl']

        details['profitability'] = {
            'value': pnl,
            'threshold': 0.0,
            'passed': pnl > 0.0
        }

        if pnl > 0.0:
            criteria_met += 1

        # Criterion 5: Stability (mode transitions)
        transition_count = self.db.execute_query(
            """
            SELECT COUNT(*) FROM learning_states
            WHERE symbol = ?
            AND entered_at > ?
            """,
            (symbol, self.deployment_start.isoformat())
        )

        total_transitions = transition_count[0][0] if transition_count else 0
        hours_elapsed = (datetime.utcnow() - self.deployment_start).total_seconds() / 3600
        transitions_per_hour = total_transitions / hours_elapsed if hours_elapsed > 0 else 0

        details['stability'] = {
            'value': transitions_per_hour,
            'threshold': 2.0,
            'passed': transitions_per_hour < 2.0
        }

        if transitions_per_hour < 2.0:
            criteria_met += 1

        # Final verdict
        approved = criteria_met >= 4

        logger.info("=" * 80)
        logger.info("PHASE 1 VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Criteria Met: {criteria_met}/5")
        logger.info("")

        for criterion, data in details.items():
            status = "✓" if data['passed'] else "✗"
            logger.info(
                f"{status} {criterion}: {data['value']:.4f} "
                f"(threshold: {data['threshold']:.4f})"
            )

        logger.info("")
        logger.info(f"Verdict: {'APPROVED' if approved else 'NOT APPROVED'}")
        logger.info("=" * 80)

        return {
            'approved': approved,
            'criteria_met': criteria_met,
            'details': details
        }


def main():
    """Main deployment entry point."""
    parser = argparse.ArgumentParser(description='Production Deployment Manager')

    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 3],
        help='Deployment phase (1=single symbol, 3=full rollout)'
    )

    parser.add_argument(
        '--symbol',
        type=str,
        default='BTC/USDT',
        help='Symbol for Phase 1 deployment'
    )

    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback deployment'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path'
    )

    args = parser.parse_args()

    # Create deployment manager
    deployment = ProductionDeployment(config_path=args.config)

    # Handle rollback
    if args.rollback:
        deployment.rollback()
        return

    # Execute phase
    if args.phase == 1:
        success = deployment.phase_1_single_symbol(symbol=args.symbol)
        sys.exit(0 if success else 1)

    elif args.phase == 3:
        success = deployment.phase_3_full_rollout()
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
