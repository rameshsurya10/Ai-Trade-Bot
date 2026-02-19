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
    print("  ✅ Dual-market: Crypto (Binance) + Forex (Capital.com / MT5)")
    print()

    # Initialize runner
    print("Initializing LiveTradingRunner...")
    runner = LiveTradingRunner(
        config_path="config.yaml",
        mode=TradingMode.PAPER  # Change to LIVE when ready for real trading
    )

    # Add crypto symbols (Binance)
    print("Adding crypto symbols...")
    runner.add_symbol("BTC/USDT", exchange="binance", interval="1h")
    runner.add_symbol("ETH/USDT", exchange="binance", interval="1h")

    # Add forex symbols (MT5) - requires MT5 terminal + demo account
    config = Config.load("config.yaml")
    if config.mt5.enabled:
        print("Adding forex symbols (MT5)...")
        runner.add_symbol("EUR/USD", exchange="mt5", interval="1h")
        runner.add_symbol("GBP/USD", exchange="mt5", interval="1h")
        runner.add_symbol("USD/JPY", exchange="mt5", interval="1h")
        runner.add_symbol("USD/CHF", exchange="mt5", interval="1h")
        runner.add_symbol("AUD/USD", exchange="mt5", interval="1h")
        runner.add_symbol("NZD/USD", exchange="mt5", interval="1h")
        runner.add_symbol("USD/CAD", exchange="mt5", interval="1h")
    else:
        print("MT5 disabled in config, skipping forex symbols")

    # Add forex symbols (Capital.com) - REST API, works on Linux
    capital_config = config.raw.get('capital', {})
    if capital_config.get('enabled', False):
        print("Adding forex symbols (Capital.com)...")
        for pair in capital_config.get('pairs', ['EUR/USD', 'GBP/USD', 'USD/JPY']):
            runner.add_symbol(pair, exchange="capital", interval="1h")
    else:
        print("Capital.com disabled in config, skipping Capital.com forex")

    print()
    print("="*70)
    print("✅ Configuration complete!")
    print("="*70)
    print()
    print("What happens now:")
    print("  1. Loads/trains models for all symbols and timeframes")
    print("  2. Connects to Binance WebSocket (crypto) + MT5 terminal (forex)")
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
