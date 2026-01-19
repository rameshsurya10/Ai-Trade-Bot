#!/usr/bin/env python3
"""
Synchronous Database Population Script
Fetches historical data synchronously to avoid async timing issues.
"""
import sys
import time
import ccxt
from pathlib import Path

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.database import Database

def fetch_and_save_data(db, exchange, symbol, interval, days=365):
    """Fetch historical data from exchange and save to database."""
    print(f"\n{'='*60}")
    print(f"Fetching {symbol} @ {interval} ({days} days)...")
    print(f"{'='*60}")

    # Map interval to ccxt timeframe
    since = exchange.parse8601((exchange.iso8601(exchange.milliseconds() - days * 24 * 60 * 60 * 1000)))

    all_candles = []
    try:
        # Fetch in chunks
        while True:
            candles = exchange.fetch_ohlcv(symbol, interval, since=since, limit=1000)
            if not candles:
                break

            print(f"  Fetched {len(candles)} candles...")
            all_candles.extend(candles)

            # Update since for next batch
            since = candles[-1][0] + 1

            # Stop if we've caught up to present
            if since >= exchange.milliseconds():
                break

            time.sleep(exchange.rateLimit / 1000)  # Respect rate limits

        # Save to database
        if all_candles:
            import pandas as pd

            # Convert to DataFrame with correct format
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Keep timestamp as int (milliseconds), add datetime column
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            df['interval'] = interval

            # Save to database
            db.save_candles(df, symbol=symbol, interval=interval)

            print(f"✅ Saved {len(all_candles)} candles to database")
            return len(all_candles)
        else:
            print(f"⚠️  No candles fetched")
            return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    print("="*60)
    print("SYNCHRONOUS DATABASE POPULATION")
    print("="*60)

    # Initialize database
    db_path = ROOT / "data" / "trading.db"
    db_path.parent.mkdir(exist_ok=True)
    db = Database(str(db_path))
    print(f"✅ Database initialized: {db_path}")

    # Initialize Binance
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    print(f"✅ Connected to Binance")

    # Fetch data for all symbol/interval combinations
    symbols = ["BTC/USDT", "ETH/USDT"]
    intervals = ["15m", "1h", "4h", "1d"]

    total_candles = 0
    results = []

    for symbol in symbols:
        for interval in intervals:
            count = fetch_and_save_data(db, exchange, symbol, interval, days=365)
            total_candles += count
            results.append((symbol, interval, count))

    # Summary
    print("\n" + "="*60)
    print("POPULATION COMPLETE")
    print("="*60)

    for symbol, interval, count in results:
        status = '✅' if count >= 1000 else '⚠️'
        print(f"{status} {symbol:12} @ {interval:4} = {count:6,} candles")

    print("="*60)
    print(f"TOTAL: {total_candles:,} candles")
    print("="*60)
    print("\n✅ Ready to run: python run_trading.py")

if __name__ == "__main__":
    main()
