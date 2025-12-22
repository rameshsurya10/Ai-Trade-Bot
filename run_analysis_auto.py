#!/usr/bin/env python3
"""
AI Trade Bot - Auto-Learning Version
====================================

ENHANCED VERSION WITH AUTO-RETRAINING:
    python run_analysis_auto.py

This starts:
1. Data collection (runs forever)
2. Analysis engine with auto-retrain (runs forever)
3. Signal service (runs forever)
4. Notifier (desktop + sound alerts)
5. Performance tracking (tracks win rate)
6. Auto-retrain trigger (when performance drops)

AUTO-RETRAIN CONDITIONS:
- Win rate drops below 45% → Retrain immediately
- Every 30 days → Retrain to adapt to market changes
- After first 100 signals → Initial retrain with real data

RUNS IN BACKGROUND - Dashboard is optional!
Analysis continues even if:
- Dashboard is closed
- Browser tab is changed
- Screen is off

ONLY STOPS WHEN:
- You run: python stop_analysis.py
- You press Ctrl+C
- Data source disconnects

NO AUTO-TRADING - You trade manually based on signals!
"""

import os
import sys
import signal
import time
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_service import DataService
from src.multi_currency_system import MultiCurrencySystem
from src.signal_service import SignalService
from src.notifier import Notifier

# PID file for stop command
PID_FILE = Path(__file__).parent / "data" / ".analysis.pid"


def setup_logging():
    """Configure logging."""
    log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))

    # File handler
    log_dir = Path(__file__).parent / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "trading.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    return logging.getLogger(__name__)


def save_pid():
    """Save current process ID for stop command."""
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def clear_pid():
    """Remove PID file."""
    if PID_FILE.exists():
        PID_FILE.unlink()


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ██╗    ████████╗██████╗  █████╗ ██████╗ ███████╗  ║
║    ██╔══██╗██║    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  ║
║    ███████║██║       ██║   ██████╔╝███████║██║  ██║█████╗    ║
║    ██╔══██║██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    ║
║    ██║  ██║██║       ██║   ██║  ██║██║  ██║██████╔╝███████╗  ║
║    ╚═╝  ╚═╝╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  ║
║                                                              ║
║         MANUAL TRADING SIGNAL SYSTEM + AUTO-LEARNING        ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  ✓ NO auto-trading - You control your trades                 ║
║  ✓ Continuous analysis - Never stops until YOU stop it       ║
║  ✓ Desktop alerts - Works with browser closed                ║
║  ✓ Sound notifications - Never miss a signal                 ║
║  ✓ AUTO-RETRAIN - Adapts to market changes automatically     ║
║  ✓ Performance tracking - Win rate monitoring                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  AUTO-RETRAIN TRIGGERS:                                      ║
║  • Win rate drops below 45%                                  ║
║  • Every 30 days since last retrain                          ║
║  • After first 100 signals                                   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  To STOP: python stop_analysis.py  OR  Ctrl+C                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


class AutoLearningTradingBot:
    """
    Auto-Learning Trading Bot Coordinator.

    Connects all services together with auto-retrain capability:
    DataService -> MultiCurrencySystem -> SignalService -> Notifier
                        ↓
                   Performance Tracking
                        ↓
                   Auto-Retrain Trigger
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.data_service = DataService(config_path)
        self.multi_currency = MultiCurrencySystem(config_path)
        self.signal_service = SignalService(config_path)
        self.notifier = Notifier(config_path)

        self._running = False

        # Add BTC/USDT as default currency
        symbol = self.data_service.symbol
        exchange = self.data_service.exchange
        interval = self.data_service.interval
        self.multi_currency.add_currency(symbol, exchange, interval)

        # Wire up the pipeline
        self._connect_services()

    def _connect_services(self):
        """Connect services in pipeline."""
        # DataService -> MultiCurrencySystem
        self.data_service.register_callback(self._on_new_candles)

        # MultiCurrencySystem -> SignalService
        self.multi_currency.register_callback(self._on_prediction)

        # SignalService -> Notifier
        self.signal_service.register_callback(self.notifier.on_signal)

        self.logger.info("Services connected with auto-retrain enabled")

    def _on_new_candles(self, df):
        """Handle new candles from data service."""
        # Get full history for analysis
        full_df = self.data_service.get_candles(limit=500)
        if len(full_df) >= 100:
            # Generate prediction using multi-currency system
            symbol = self.data_service.symbol
            prediction = self.multi_currency.predict(symbol, full_df)

            if prediction:
                # Pass to signal service
                self._on_prediction(prediction)

    def _on_prediction(self, prediction: dict):
        """Handle prediction from multi-currency system."""
        # Convert to format expected by signal service
        signal_data = {
            'timestamp': prediction.get('timestamp', datetime.utcnow()),
            'price': prediction.get('price', 0),
            'signal': prediction.get('direction', 'NEUTRAL'),
            'confidence': prediction.get('confidence', 0),
            'stop_loss': prediction.get('stop_loss'),
            'take_profit': prediction.get('take_profit'),
            'algorithms': prediction.get('algorithms', {}),
            'performance': prediction.get('performance', {})
        }

        # Send to signal service
        self.signal_service.on_prediction(signal_data)

    def start(self):
        """Start all services."""
        self.logger.info("Starting auto-learning trading bot...")

        # 1. Try to load existing model (optional for auto-retrain system)
        symbol = self.data_service.symbol
        model = self.multi_currency.model_manager.get_model(symbol)

        if model is None:
            self.logger.warning("═" * 60)
            self.logger.warning("NO TRAINED MODEL FOUND!")
            self.logger.warning("System will work but with reduced accuracy.")
            self.logger.warning("Recommendation: Train initial model:")
            self.logger.warning("  python scripts/train_model.py")
            self.logger.warning("")
            self.logger.warning("Auto-retrain will create a model after 100 signals.")
            self.logger.warning("═" * 60)

        # 2. Start data collection
        self.logger.info("Starting data collection...")
        self.data_service.start()

        self._running = True
        self.logger.info("Auto-learning trading bot running - waiting for signals...")

    def stop(self):
        """Stop all services."""
        self.logger.info("Stopping auto-learning trading bot...")
        self._running = False
        self.data_service.stop()
        self.logger.info("Auto-learning trading bot stopped")

    def get_status(self) -> dict:
        """Get full system status including retrain info."""
        symbol = self.data_service.symbol
        performance_report = self.multi_currency.get_performance_report()

        return {
            'running': self._running,
            'data_service': self.data_service.get_status(),
            'signal_service': self.signal_service.get_status(),
            'notifier': self.notifier.get_status(),
            'performance': performance_report.get(symbol, {}),
            'auto_retrain_enabled': self.multi_currency.config.get('auto_training', {}).get('enabled', False),
            'started_at': datetime.utcnow().isoformat()
        }


def main():
    """Main entry point."""
    print_banner()
    logger = setup_logging()

    # Check if already running
    if PID_FILE.exists():
        existing_pid = PID_FILE.read_text().strip()
        try:
            os.kill(int(existing_pid), 0)
            print(f"\n⚠️  Analysis already running (PID: {existing_pid})")
            print("   To stop it: python stop_analysis.py")
            print("   To force restart: python stop_analysis.py && python run_analysis_auto.py")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            clear_pid()

    # Create bot
    bot = AutoLearningTradingBot()

    # Handle shutdown
    def shutdown(signum, frame):
        print("\n\n🛑 Shutdown signal received...")
        bot.stop()
        clear_pid()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Save PID
    save_pid()
    logger.info(f"PID saved: {os.getpid()}")

    # Start bot
    bot.start()

    symbol = bot.data_service.symbol
    model = bot.multi_currency.model_manager.get_model(symbol)

    print("\n" + "="*60)
    print("✅ AUTO-LEARNING ANALYSIS RUNNING")
    print("="*60)
    print(f"📊 Symbol: {bot.data_service.symbol}")
    print(f"⏱️  Interval: {bot.data_service.interval}")
    print(f"🤖 ML Model: {'Loaded' if model else 'Will train after 100 signals'}")
    print(f"🔄 Auto-Retrain: Enabled")
    print("="*60)
    print("\n🧠 AUTO-RETRAIN TRIGGERS:")
    print("   • Win rate < 45% → Immediate retrain")
    print("   • Every 30 days → Periodic adaptation")
    print("   • After 100 signals → Initial auto-train")
    print("\n💡 Waiting for trading signals...")
    print("   You'll get desktop notifications when signals appear")
    print("   Performance is tracked automatically")
    print("   Model will retrain when needed")
    print("   Press Ctrl+C to stop\n")

    # Keep running
    try:
        while bot._running:
            # Print heartbeat every 5 minutes with performance stats
            time.sleep(300)
            status = bot.get_status()

            candles = status['data_service']['total_candles']
            price = status['data_service']['latest_price']
            signals = status['signal_service']['total_signals']

            perf = status.get('performance', {})
            win_rate = perf.get('win_rate', 'N/A')
            needs_retrain = perf.get('needs_retrain', False)
            last_retrain = perf.get('last_retrain', 'Never')

            logger.info(
                f"💓 Heartbeat | Price: ${price:,.2f} | "
                f"Candles: {candles} | Signals: {signals} | "
                f"Win Rate: {win_rate} | "
                f"Needs Retrain: {needs_retrain} | "
                f"Last Retrain: {last_retrain}"
            )

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"Error in main loop: {e}")
    finally:
        bot.stop()
        clear_pid()


if __name__ == "__main__":
    main()
