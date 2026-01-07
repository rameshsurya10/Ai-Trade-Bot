#!/usr/bin/env python3
"""
AI Trade Bot - Analysis & Trading System
=========================================

ONE COMMAND TO START EVERYTHING:
    python run_analysis.py

This starts:
1. Real-time data streaming
2. AI/ML prediction engine
3. Signal generation with notifications
4. Paper trading by default (safe)

Modes:
    python run_analysis.py                 # Paper trading (default)
    python run_analysis.py --mode paper    # Explicit paper trading
    python run_analysis.py --mode live     # Live trading (REAL MONEY!)

RUNS IN BACKGROUND - Dashboard is optional!

ONLY STOPS WHEN:
- You run: python stop_analysis.py
- You press Ctrl+C
"""

import os
import sys
import signal
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables FIRST before any other imports
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# PID file for stop command
ROOT = Path(__file__).parent
PID_FILE = ROOT / "run_analysis.pid"


def setup_logging():
    """Configure logging."""
    log_format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))

    # File handler
    log_dir = ROOT / "data"
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


def print_banner(mode: str):
    """Print startup banner."""
    mode_display = mode.upper()

    mode_warning = ""
    if mode == "live":
        mode_warning = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  ⚠️  WARNING: LIVE TRADING MODE - REAL MONEY AT RISK! ⚠️     ║
    ╚══════════════════════════════════════════════════════════════╝
"""
    elif mode == "paper":
        mode_warning = """
    ╔══════════════════════════════════════════════════════════════╗
    ║  📝 PAPER TRADING MODE - Simulated trades, no real money     ║
    ╚══════════════════════════════════════════════════════════════╝
"""

    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     █████╗ ██╗    ████████╗██████╗  █████╗ ██████╗ ███████╗  ║
║    ██╔══██╗██║    ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝  ║
║    ███████║██║       ██║   ██████╔╝███████║██║  ██║█████╗    ║
║    ██╔══██║██║       ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝    ║
║    ██║  ██║██║       ██║   ██║  ██║██║  ██║██████╔╝███████╗  ║
║    ╚═╝  ╚═╝╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝  ║
║                                                              ║
║                 AI TRADING SIGNAL SYSTEM                     ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║  Mode: {mode_display:^51} ║
╠══════════════════════════════════════════════════════════════╣
║  Features:                                                   ║
║    ✓ LSTM + Mathematical ensemble predictions                ║
║    ✓ Continuous analysis (never stops until YOU stop it)     ║
║    ✓ Desktop + sound notifications                           ║
║    ✓ Portfolio tracking with risk management                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  To STOP: python stop_analysis.py  OR  Ctrl+C                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{mode_warning}"""
    print(banner)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Trade Bot - Analysis & Trading System"
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    parser.add_argument(
        '--symbols', '-s',
        nargs='+',
        default=['BTC/USDT'],
        help='Symbols to analyze (default: BTC/USDT)'
    )
    parser.add_argument(
        '--exchange', '-e',
        default='binance',
        help='Exchange to use (default: binance)'
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Config file path (default: config.yaml)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print_banner(args.mode)
    logger = setup_logging()

    # Safety check for live mode
    if args.mode == "live":
        confirm = input("\n⚠️  You are about to start LIVE trading with REAL MONEY.\n"
                       "Type 'YES I UNDERSTAND' to continue: ")
        if confirm != "YES I UNDERSTAND":
            print("Aborted. Use --mode paper for safe testing.")
            sys.exit(0)

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

    # Import after logging is set up
    try:
        from src.live_trading import LiveTradingRunner, TradingMode, RunnerStatus
        from src.notifier import Notifier

        # Map mode string to enum
        mode_map = {
            'paper': TradingMode.PAPER,
            'live': TradingMode.LIVE
        }
        mode = mode_map[args.mode]

    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    # Create runner
    logger.info(f"Initializing trading system (mode={mode.value})...")

    try:
        runner = LiveTradingRunner(
            config_path=args.config,
            mode=mode
        )
        notifier = Notifier(args.config)
    except Exception as e:
        logger.critical(f"Failed to initialize: {e}")
        sys.exit(1)

    # Handle shutdown
    def shutdown(signum, frame):
        print("\n\n🛑 Shutdown signal received...")
        runner.stop()
        clear_pid()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Save PID
    save_pid()
    logger.info(f"PID saved: {os.getpid()}")

    # Signal callback with notifications
    def on_signal(signal_data: dict):
        symbol = signal_data.get('symbol', 'N/A')
        direction = signal_data.get('direction', 'N/A')
        confidence = signal_data.get('confidence', 0) * 100
        price = signal_data.get('price', 0)
        stop_loss = signal_data.get('stop_loss', 0)
        take_profit = signal_data.get('take_profit', 0)

        logger.info(
            f"🔔 SIGNAL: {direction} {symbol} @ ${price:,.2f} "
            f"(Confidence: {confidence:.1f}%)"
        )

        # Send notification
        try:
            notifier.notify(
                title=f"📊 {direction} Signal: {symbol}",
                message=f"Price: ${price:,.2f}\n"
                       f"Confidence: {confidence:.1f}%\n"
                       f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}",
                priority="high" if confidence >= 70 else "normal"
            )
        except Exception as e:
            logger.debug(f"Notification error: {e}")

    runner.on_signal(on_signal)

    # Add symbols
    for symbol in args.symbols:
        try:
            runner.add_symbol(symbol, exchange=args.exchange)
            logger.info(f"Added symbol: {symbol}")
        except Exception as e:
            logger.error(f"Failed to add {symbol}: {e}")

    # Start runner
    try:
        runner.start()
    except Exception as e:
        logger.critical(f"Failed to start: {e}")
        clear_pid()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ANALYSIS RUNNING")
    print("=" * 60)
    print(f"📊 Symbols: {', '.join(args.symbols)}")
    print(f"💱 Exchange: {args.exchange}")
    print(f"🎮 Mode: {args.mode.upper()}")
    print("=" * 60)
    print("\n💡 Waiting for trading signals...")
    print("   You'll get desktop notifications when signals appear")
    print("   Press Ctrl+C to stop\n")

    # Keep running with status updates
    try:
        heartbeat_interval = 300  # 5 minutes
        last_heartbeat = time.time()

        while runner.status == RunnerStatus.RUNNING:
            time.sleep(10)

            # Heartbeat every 5 minutes
            if time.time() - last_heartbeat >= heartbeat_interval:
                status = runner.get_status()
                portfolio = status.get('portfolio', {})
                signals = status.get('signals_generated', 0)
                equity = portfolio.get('total_value', 0)

                logger.info(
                    f"💓 Heartbeat | Equity: ${equity:,.2f} | "
                    f"Signals: {signals} | "
                    f"Positions: {portfolio.get('position_count', 0)}"
                )
                last_heartbeat = time.time()

    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
