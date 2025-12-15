"""
Data Service - 24/7 Price Collection
=====================================
Runs continuously in background, collecting price data.
Does NOT stop when dashboard is closed.
Only stops when: you run stop command, or data source disconnects.
"""

import asyncio
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable
import threading
import time

import ccxt
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class DataService:
    """
    Continuous data collection service.

    - Runs in background thread
    - Collects data every interval
    - Stores in SQLite database
    - Never stops unless explicitly told to
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data service with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Data settings
        self.symbol = self.config['data']['symbol']
        self.exchange_name = self.config['data']['exchange']
        self.interval = self.config['data']['interval']
        self.history_days = self.config['data']['history_days']

        # Database
        self.db_path = Path(self.config['database']['path'])
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._exchange = None
        self._last_update = None
        self._callbacks: list[Callable] = []

        # Initialize database
        self._init_database()

        logger.info(f"DataService initialized: {self.symbol} @ {self.interval}")

    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Price data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER UNIQUE,
                datetime TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                symbol TEXT,
                interval TEXT
            )
        ''')

        # Create index for fast queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON candles(timestamp DESC)
        ''')

        # Signals table (for history)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                datetime TEXT,
                signal TEXT,
                confidence REAL,
                price REAL,
                stop_loss REAL,
                take_profit REAL,
                notified INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")

    def _get_exchange(self):
        """Get or create exchange connection."""
        if self._exchange is None:
            exchange_class = getattr(ccxt, self.exchange_name)
            self._exchange = exchange_class({
                'enableRateLimit': True,
            })
        return self._exchange

    def _interval_to_ms(self) -> int:
        """Convert interval string to milliseconds."""
        unit = self.interval[-1]
        value = int(self.interval[:-1])

        multipliers = {
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
        }

        return value * multipliers.get(unit, 60 * 60 * 1000)

    def fetch_historical_data(self, days: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            days: Number of days to fetch (default: from config)

        Returns:
            DataFrame with OHLCV data
        """
        days = days or self.history_days
        exchange = self._get_exchange()

        # Calculate start time
        since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)

        all_candles = []

        logger.info(f"Fetching {days} days of {self.symbol} data...")

        while True:
            try:
                candles = exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=self.interval,
                    since=since,
                    limit=1000
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next batch
                since = candles[-1][0] + 1

                # Check if we've reached current time
                if since > int(datetime.utcnow().timestamp() * 1000):
                    break

                # Rate limiting
                time.sleep(exchange.rateLimit / 1000)

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        if not all_candles:
            logger.warning("No candles fetched")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp')

        logger.info(f"Fetched {len(df)} candles")

        return df

    def save_candles(self, df: pd.DataFrame):
        """Save candles to database."""
        if df.empty:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for _, row in df.iterrows():
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO candles
                    (timestamp, datetime, open, high, low, close, volume, symbol, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    int(row['timestamp']),
                    row['datetime'].isoformat() if hasattr(row['datetime'], 'isoformat') else str(row['datetime']),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    self.symbol,
                    self.interval
                ))
            except Exception as e:
                logger.debug(f"Error saving candle: {e}")

        conn.commit()
        conn.close()

    def get_candles(self, limit: int = 500) -> pd.DataFrame:
        """
        Get most recent candles from database.

        Args:
            limit: Maximum number of candles to return

        Returns:
            DataFrame with OHLCV data
        """
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query(f'''
            SELECT timestamp, datetime, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', conn, params=(self.symbol, self.interval, limit))

        conn.close()

        if not df.empty:
            df = df.sort_values('timestamp')
            df['datetime'] = pd.to_datetime(df['datetime'])

        return df

    def _fetch_latest(self):
        """Fetch latest candle(s)."""
        try:
            exchange = self._get_exchange()

            candles = exchange.fetch_ohlcv(
                self.symbol,
                timeframe=self.interval,
                limit=5
            )

            if candles:
                df = pd.DataFrame(
                    candles,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

                self.save_candles(df)
                self._last_update = datetime.utcnow()

                # Notify callbacks (analysis engine)
                for callback in self._callbacks:
                    try:
                        callback(df)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                return True

        except Exception as e:
            logger.error(f"Error fetching latest data: {e}")
            return False

    def _collection_loop(self):
        """
        Main collection loop - runs forever until stopped.

        THIS IS THE KEY: It runs in background thread.
        Dashboard can close, this keeps running.
        """
        logger.info("=== DATA COLLECTION STARTED ===")
        logger.info("Will run until you stop it or data source disconnects")

        update_interval = self.config['analysis']['update_interval']
        consecutive_errors = 0
        max_errors = 10

        while self._running:
            success = self._fetch_latest()

            if success:
                consecutive_errors = 0
                logger.debug(f"Data updated at {datetime.utcnow()}")
            else:
                consecutive_errors += 1
                logger.warning(f"Fetch error ({consecutive_errors}/{max_errors})")

                if consecutive_errors >= max_errors:
                    logger.error("Too many consecutive errors. Stopping.")
                    self._running = False
                    break

            # Wait for next update
            for _ in range(update_interval):
                if not self._running:
                    break
                time.sleep(1)

        logger.info("=== DATA COLLECTION STOPPED ===")

    def register_callback(self, callback: Callable):
        """Register callback to be called when new data arrives."""
        self._callbacks.append(callback)

    def start(self):
        """
        Start continuous data collection in background.

        Returns immediately - collection runs in separate thread.
        """
        if self._running:
            logger.warning("Data service already running")
            return

        # First, fetch historical data if database is empty
        existing = self.get_candles(limit=1)
        if existing.empty:
            logger.info("No existing data, fetching historical...")
            historical = self.fetch_historical_data()
            self.save_candles(historical)

        self._running = True
        self._thread = threading.Thread(
            target=self._collection_loop,
            daemon=False,  # NOT daemon - survives main thread
            name="DataCollectionThread"
        )
        self._thread.start()

        logger.info("Data service started in background")

    def stop(self):
        """Stop data collection."""
        logger.info("Stopping data service...")
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)

        logger.info("Data service stopped")

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running

    @property
    def last_update(self) -> Optional[datetime]:
        """Get last update time."""
        return self._last_update

    def get_status(self) -> dict:
        """Get service status."""
        candles = self.get_candles(limit=1)
        latest_price = candles['close'].iloc[-1] if not candles.empty else None

        return {
            'running': self._running,
            'symbol': self.symbol,
            'interval': self.interval,
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'latest_price': latest_price,
            'total_candles': len(self.get_candles(limit=10000)),
        }


# For direct testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    service = DataService()

    try:
        service.start()

        # Keep running
        while service.is_running:
            status = service.get_status()
            print(f"\rPrice: ${status['latest_price']:.2f} | "
                  f"Candles: {status['total_candles']} | "
                  f"Last: {status['last_update']}", end='')
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopping...")
        service.stop()
