#!/usr/bin/env python3
"""
Populate Database with Historical Data

This script fetches historical candle data from Binance and saves it to the database.
Required before training models.

Usage:
    python scripts/populate_database.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.provider import UnifiedDataProvider
from src.core.database import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Fetch and save historical data to database."""

    # Initialize database
    db_path = ROOT / "data" / "trading.db"
    db_path.parent.mkdir(exist_ok=True)
    db = Database(str(db_path))

    logger.info(f"Database initialized: {db_path}")

    # Get data provider singleton
    config_path = ROOT / "config.yaml"
    provider = UnifiedDataProvider.get_instance(str(config_path))

    # Connect database to provider (enables auto-save)
    provider.set_database(db)
    logger.info("Database connected to provider")

    # Subscribe to symbols (this triggers historical fetch)
    symbols = [
        ("BTC/USDT", "1h"),
        ("ETH/USDT", "1h"),
        ("BTC/USDT", "15m"),
        ("ETH/USDT", "15m"),
        ("BTC/USDT", "4h"),
        ("ETH/USDT", "4h"),
        ("BTC/USDT", "1d"),
        ("ETH/USDT", "1d"),
    ]

    for symbol, interval in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fetching historical data for {symbol} @ {interval}")
        logger.info(f"{'='*60}")

        provider.subscribe(
            symbol,
            exchange="binance",
            interval=interval
        )

        # The subscribe() call automatically:
        # 1. Fetches historical data (up to 1 year based on config)
        # 2. Saves to database (because we called set_database)

    # Wait for all async historical fetches to complete
    import time
    logger.info(f"\n{'='*60}")
    logger.info("Waiting for all historical data fetches to complete...")
    logger.info(f"{'='*60}")

    # Wait up to 60 seconds for all fetches to complete
    max_wait = 60
    start_time = time.time()
    all_complete = False

    while time.time() - start_time < max_wait and not all_complete:
        time.sleep(2)

        # Check if all timeframes have data
        missing = []
        for symbol, interval in symbols:
            df = db.get_candles(symbol, interval, limit=10)
            if df is None or len(df) == 0:
                missing.append(f"{symbol} @ {interval}")

        if not missing:
            all_complete = True
            logger.info("✅ All historical data fetched and saved!")
        else:
            logger.info(f"⏳ Still waiting for {len(missing)} timeframes... ({int(time.time() - start_time)}s elapsed)")

    # Final verification
    logger.info(f"\n{'='*60}")
    logger.info("DATABASE POPULATION COMPLETE")
    logger.info(f"{'='*60}")

    total_candles = 0
    for symbol, interval in symbols:
        df = db.get_candles(symbol, interval, limit=100000)
        count = len(df) if df is not None else 0
        status = '✅' if count >= 1000 else '⚠️'
        logger.info(f"{status} {symbol:12} @ {interval:4} = {count:6,} candles")
        total_candles += count

    logger.info(f"{'='*60}")
    logger.info(f"TOTAL: {total_candles:,} candles saved")
    logger.info(f"{'='*60}")
    logger.info("You can now run: python run_trading.py")

if __name__ == "__main__":
    main()
