#!/usr/bin/env python3
"""
Start AI Trade Bot with Continuous Learning
============================================

This script starts the LiveTradingRunner with full continuous learning enabled.

Features:
- Automatic model training on startup (using 1-year data from database)
- Continuous learning from every candle
- Automatic retraining when performance drops
- Multi-timeframe analysis (15m, 1h, 4h, 1d)
- Strategy discovery (run analyze_strategies.py to see results)

Usage:
    python run_trading.py
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.live_trading.runner import LiveTradingRunner, TradingMode
from src.core.config import Config

def main():
    print("="*70)
    print("AI TRADE BOT - CONTINUOUS LEARNING MODE")
    print("="*70)
    print()
    print("Features:")
    print("  ✅ Automatic training on 1-year historical data")
    print("  ✅ Continuous learning from every trade")
    print("  ✅ Automatic retraining when accuracy drops")
    print("  ✅ Multi-timeframe analysis (15m, 1h, 4h, 1d)")
    print("  ✅ Strategy discovery and comparison")
    print("  ✅ Crypto (Binance) + Forex/Metals (Twelve Data)")
    print()

    # Initialize runner
    print("Initializing LiveTradingRunner...")
    runner = LiveTradingRunner(
        config_path="config.yaml",
        mode=TradingMode.PAPER  # Change to LIVE when ready for real trading
    )

    config = Config.load("config.yaml")

    # Add crypto symbols (Binance)
    print("Adding crypto symbols (Binance)...")
    runner.add_symbol("BTC/USDT", exchange="binance", interval="1h")
    runner.add_symbol("ETH/USDT", exchange="binance", interval="1h")

    # Add forex/metals symbols from Twelve Data config
    twelvedata_config = config.raw.get('twelvedata', {})
    if twelvedata_config.get('enabled', False):
        print("Adding forex/metals symbols (Twelve Data)...")
        for pair in twelvedata_config.get('pairs', ['EUR/USD', 'XAU/USD']):
            runner.add_symbol(pair, exchange="twelvedata", interval="1h")
            print(f"  + {pair}")
    else:
        print("Twelve Data disabled in config")

    print()
    print("="*70)
    print("✅ Configuration complete!")
    print("="*70)
    print()
    print("What happens now:")
    print("  1. Loads/trains models for all symbols and timeframes")
    print("  2. Connects to Binance WebSocket (crypto) + Twelve Data REST (forex)")
    print("  3. Makes predictions on every candle close")
    print("  4. Executes paper trades when confidence is high")
    print("  5. Records outcomes and retrains when needed")
    print()
    print("To analyze discovered strategies later, run:")
    print("  python scripts/analyze_strategies.py")
    print()
    print("="*70)
    print("Starting trading... (Press Ctrl+C to stop)")
    print("="*70)
    print()

    # Start trading (blocking - runs until Ctrl+C)
    try:
        runner.start(blocking=True)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("Stopping...")
        print("="*70)
        runner.stop()
        print("\n✅ Stopped gracefully")
        print()
        print("To see what strategies were discovered, run:")
        print("  python scripts/analyze_strategies.py")
        print()

if __name__ == "__main__":
    main()
