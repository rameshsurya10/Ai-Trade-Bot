#!/usr/bin/env python3
"""
AI Trade Bot - Start Analysis
=============================

ONE COMMAND TO START EVERYTHING:
    python run_analysis.py

This starts:
1. Data collection (runs forever)
2. Analysis engine (runs forever)
3. Signal service (runs forever)
4. Notifier (desktop + sound alerts)

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
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_service import DataService
from src.analysis_engine import AnalysisEngine
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
║              MANUAL TRADING SIGNAL SYSTEM                    ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  ✓ NO auto-trading - You control your trades                 ║
║  ✓ Continuous analysis - Never stops until YOU stop it       ║
║  ✓ Desktop alerts - Works with browser closed                ║
║  ✓ Sound notifications - Never miss a signal                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  To STOP: python stop_analysis.py  OR  Ctrl+C                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


class TradingBot:
    """
    Main trading bot coordinator.

    Connects all services together:
    DataService -> AnalysisEngine -> SignalService -> Notifier
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)

        # Initialize services
        self.data_service = DataService(config_path)
        self.analysis_engine = AnalysisEngine(config_path)
        self.signal_service = SignalService(config_path)
        self.notifier = Notifier(config_path)

        self._running = False

        # Wire up the pipeline
        self._connect_services()

    def _connect_services(self):
        """Connect services in pipeline."""
        # DataService -> AnalysisEngine
        self.data_service.register_callback(self._on_new_candles)

        # AnalysisEngine -> SignalService
        self.analysis_engine.register_callback(self.signal_service.on_prediction)

        # SignalService -> Notifier
        self.signal_service.register_callback(self.notifier.on_signal)

        self.logger.info("Services connected")

    def _on_new_candles(self, df):
        """Handle new candles from data service."""
        # Get full history for analysis
        full_df = self.data_service.get_candles(limit=500)
        if len(full_df) >= 100:
            self.analysis_engine.on_new_data(full_df)

    def start(self):
        """Start all services."""
        self.logger.info("Starting trading bot...")

        # Load ML model (optional - works without it too)
        self.analysis_engine.load_model()

        # Start data collection
        self.data_service.start()

        self._running = True
        self.logger.info("Trading bot running - waiting for signals...")

    def stop(self):
        """Stop all services."""
        self.logger.info("Stopping trading bot...")
        self._running = False
        self.data_service.stop()
        self.logger.info("Trading bot stopped")

    def get_status(self) -> dict:
        """Get full system status."""
        return {
            'running': self._running,
            'data_service': self.data_service.get_status(),
            'analysis_engine': self.analysis_engine.get_status(),
            'signal_service': self.signal_service.get_status(),
            'notifier': self.notifier.get_status(),
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
            print("   To force restart: python stop_analysis.py && python run_analysis.py")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            clear_pid()

    # Create bot
    bot = TradingBot()

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

    print("\n" + "="*60)
    print("✅ ANALYSIS RUNNING")
    print("="*60)
    print(f"📊 Symbol: {bot.data_service.symbol}")
    print(f"⏱️  Interval: {bot.data_service.interval}")
    print(f"🤖 ML Model: {'Loaded' if bot.analysis_engine._model else 'Not found (using indicators)'}")
    print("="*60)
    print("\n💡 Waiting for trading signals...")
    print("   You'll get desktop notifications when signals appear")
    print("   Press Ctrl+C to stop\n")

    # Keep running
    try:
        while bot._running:
            # Print heartbeat every 5 minutes
            time.sleep(300)
            status = bot.get_status()
            candles = status['data_service']['total_candles']
            price = status['data_service']['latest_price']
            signals = status['signal_service']['total_signals']

            logger.info(
                f"💓 Heartbeat | Price: ${price:,.2f} | "
                f"Candles: {candles} | Signals: {signals}"
            )

    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
