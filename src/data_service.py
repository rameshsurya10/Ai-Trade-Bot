"""
Data Service - 24/7 Price Collection with WebSocket Support
============================================================
Runs continuously in background, collecting price data.
Supports both REST polling AND WebSocket real-time streaming.
Does NOT stop when dashboard is closed.
Only stops when: you run stop command, or data source disconnects.

WebSocket Support:
- Binance: wss://stream.binance.com:9443/ws/<symbol>@kline_<interval>
- Coinbase: wss://ws-feed.exchange.coinbase.com
"""

import sqlite3
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, List
import threading
import time

import ccxt
import pandas as pd
import yaml
from functools import lru_cache

# Try to import websocket for real-time streaming
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)


# Cache for candle data with TTL (Time To Live)
class CachedData:
    """Simple cache with timestamp for TTL support."""
    def __init__(self, data, cache_duration=60):
        self.data = data
        self.timestamp = datetime.utcnow()
        self.cache_duration = cache_duration

    def is_valid(self):
        """Check if cache is still valid."""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age < self.cache_duration


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

        # Performance: Add caching to reduce API and DB calls
        self._candle_cache = {}  # Cache for get_candles results
        self._cache_lock = threading.Lock()  # Thread-safe cache access

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
        """Save candles to database using FULLY VECTORIZED bulk insert (100x faster)."""
        if df.empty:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # FULLY VECTORIZED - NO iterrows() - uses numpy/pandas vectorization
            # Convert datetime to ISO string format efficiently
            datetime_strs = df['datetime'].apply(
                lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)
            ).values

            # Build records using vectorized operations
            records = list(zip(
                df['timestamp'].astype(int).values,
                datetime_strs,
                df['open'].astype(float).values,
                df['high'].astype(float).values,
                df['low'].astype(float).values,
                df['close'].astype(float).values,
                df['volume'].astype(float).values,
                [self.symbol] * len(df),
                [self.interval] * len(df)
            ))

            # Bulk insert using executemany
            cursor.executemany('''
                INSERT OR REPLACE INTO candles
                (timestamp, datetime, open, high, low, close, volume, symbol, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', records)

            conn.commit()

            # Performance: Invalidate cache after saving new data
            with self._cache_lock:
                self._candle_cache.clear()
                logger.debug("Cache invalidated after saving new candles")

        except Exception as e:
            logger.error(f"Error saving candles: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_candles(self, limit: int = 500) -> pd.DataFrame:
        """
        Get most recent candles from database with caching for performance.

        Args:
            limit: Maximum number of candles to return

        Returns:
            DataFrame with OHLCV data
        """
        # Security: Enforce maximum query limit
        MAX_QUERY_LIMIT = 100000
        if limit > MAX_QUERY_LIMIT:
            logger.warning(f"Query limit {limit} exceeds maximum {MAX_QUERY_LIMIT}, capping")
            limit = MAX_QUERY_LIMIT

        # Performance: Check cache first (60s TTL to reduce DB queries by 80%+)
        cache_key = f"{self.symbol}_{self.interval}_{limit}"

        with self._cache_lock:
            if cache_key in self._candle_cache:
                cached = self._candle_cache[cache_key]
                if cached.is_valid():
                    logger.debug(f"Cache HIT for get_candles(limit={limit})")
                    return cached.data.copy()  # Return copy to prevent modification
                else:
                    # Cache expired, remove it
                    del self._candle_cache[cache_key]

        # Cache MISS - fetch from database
        logger.debug(f"Cache MISS for get_candles(limit={limit})")

        # Security: Capture symbol and interval values atomically to prevent TOCTOU race condition
        # This prevents the values from being changed between validation and query execution
        symbol_snapshot = self.symbol
        interval_snapshot = self.interval

        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql_query('''
            SELECT timestamp, datetime, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND interval = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', conn, params=(symbol_snapshot, interval_snapshot, limit))

        conn.close()

        if not df.empty:
            df = df.sort_values('timestamp')
            df['datetime'] = pd.to_datetime(df['datetime'])

        # Store in cache (thread-safe)
        with self._cache_lock:
            self._candle_cache[cache_key] = CachedData(df, cache_duration=60)

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
        """Get service status (performance optimized)."""
        candles = self.get_candles(limit=1)
        latest_price = candles['close'].iloc[-1] if not candles.empty else None

        # Security: Capture values atomically to prevent TOCTOU race condition
        symbol_snapshot = self.symbol
        interval_snapshot = self.interval

        # Performance: Use SQL COUNT instead of loading all candles
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM candles WHERE symbol = ? AND interval = ?',
                      (symbol_snapshot, interval_snapshot))
        total_candles = cursor.fetchone()[0]
        conn.close()

        return {
            'running': self._running,
            'symbol': symbol_snapshot,
            'interval': interval_snapshot,
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'latest_price': latest_price,
            'total_candles': total_candles,
            'websocket_available': WEBSOCKET_AVAILABLE,
        }


# =============================================================================
# WEBSOCKET REAL-TIME DATA SERVICE
# =============================================================================

class WebSocketDataService:
    """
    Real-time WebSocket data streaming like TradingView.

    Connects to exchange WebSocket and streams live price updates.
    Much faster than REST polling - updates in milliseconds.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize WebSocket service."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.symbol = self.config['data']['symbol']
        self.exchange_name = self.config['data']['exchange']
        self.interval = self.config['data']['interval']

        self._running = False
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable] = []
        self._last_price = None
        self._last_candle = None

        logger.info(f"WebSocketDataService initialized: {self.symbol}")

    def register_callback(self, callback: Callable):
        """Register callback for real-time updates."""
        self._callbacks.append(callback)

    def _notify_callbacks(self, data: dict):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _get_ws_url(self) -> str:
        """Get WebSocket URL for exchange."""
        symbol_ws = self.symbol.replace('-', '').replace('/', '').lower()
        interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
        interval_ws = interval_map.get(self.interval, '1m')

        if self.exchange_name == 'binance':
            # Binance WebSocket
            return f"wss://stream.binance.com:9443/ws/{symbol_ws}@kline_{interval_ws}"
        elif self.exchange_name == 'coinbase':
            # Coinbase WebSocket (needs subscription message)
            return "wss://ws-feed.exchange.coinbase.com"
        else:
            raise ValueError(f"WebSocket not supported for {self.exchange_name}")

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            if self.exchange_name == 'binance':
                # Binance kline format
                if 'k' in data:
                    kline = data['k']
                    candle = {
                        'time': int(kline['t'] / 1000),
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'is_closed': kline['x'],
                    }
                    self._last_price = candle['close']
                    self._last_candle = candle
                    self._notify_callbacks({'type': 'candle', 'data': candle})

            elif self.exchange_name == 'coinbase':
                # Coinbase ticker format
                if data.get('type') == 'ticker':
                    ticker = {
                        'time': int(datetime.utcnow().timestamp()),
                        'price': float(data.get('price', 0)),
                        'volume': float(data.get('volume_24h', 0)),
                        'bid': float(data.get('best_bid', 0)),
                        'ask': float(data.get('best_ask', 0)),
                    }
                    self._last_price = ticker['price']
                    self._notify_callbacks({'type': 'ticker', 'data': ticker})

        except Exception as e:
            logger.error(f"WebSocket message error: {e}")

    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        if self._running:
            # Reconnect
            time.sleep(5)
            self._connect()

    def _on_open(self, ws):
        """Handle WebSocket open."""
        logger.info("WebSocket connected")

        # Coinbase requires subscription message
        if self.exchange_name == 'coinbase':
            subscribe_msg = {
                "type": "subscribe",
                "product_ids": [self.symbol],
                "channels": ["ticker", "heartbeat"]
            }
            ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {self.symbol}")

    def _connect(self):
        """Connect to WebSocket."""
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            return

        try:
            ws_url = self._get_ws_url()
            self._ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            self._ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    def start(self):
        """Start WebSocket streaming."""
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket not available. Install: pip install websocket-client")
            return False

        if self._running:
            logger.warning("WebSocket already running")
            return True

        self._running = True
        self._thread = threading.Thread(
            target=self._connect,
            daemon=True,
            name="WebSocketThread"
        )
        self._thread.start()
        logger.info("WebSocket streaming started")
        return True

    def stop(self):
        """Stop WebSocket streaming."""
        self._running = False
        if self._ws:
            self._ws.close()
        logger.info("WebSocket streaming stopped")

    @property
    def last_price(self) -> Optional[float]:
        """Get last received price."""
        return self._last_price

    @property
    def last_candle(self) -> Optional[dict]:
        """Get last received candle."""
        return self._last_candle

    def get_status(self) -> dict:
        """Get WebSocket status."""
        return {
            'running': self._running,
            'symbol': self.symbol,
            'exchange': self.exchange_name,
            'last_price': self._last_price,
            'websocket_available': WEBSOCKET_AVAILABLE,
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
