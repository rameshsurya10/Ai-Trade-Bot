"""
Live WebSocket Streaming Service
================================
Real-time market data streaming via WebSocket.
Provides millisecond-latency price updates like TradingView/Binance.

Supported Exchanges:
- Binance (wss://stream.binance.com)
- Coinbase (wss://ws-feed.exchange.coinbase.com)
- Bybit (wss://stream.bybit.com)

Usage:
    from src.live_stream import LiveStream

    stream = LiveStream(exchange='binance', symbol='BTC/USDT')
    stream.on_tick(lambda tick: print(f"Price: {tick['price']}"))
    stream.start()
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, List, Optional, Any
from dataclasses import dataclass, field
from queue import Queue

# WebSocket client
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Tick:
    """Single price tick from exchange."""
    timestamp: int
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    symbol: str = ""
    exchange: str = ""


@dataclass
class LiveCandle:
    """Live updating candle (not yet closed)."""
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = False
    symbol: str = ""


@dataclass
class OrderBookLevel:
    """Single level in order book."""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Order book snapshot."""
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp: int = 0


class LiveStream:
    """
    Real-time WebSocket market data stream.

    Features:
    - Tick-by-tick price updates
    - Live candle updates (forming candle)
    - Order book streaming
    - Automatic reconnection
    - Multiple callback support

    Example:
        stream = LiveStream('binance', 'BTC/USDT', '1m')

        @stream.on_tick
        def handle_tick(tick):
            print(f"BTC: ${tick.price:,.2f}")

        @stream.on_candle
        def handle_candle(candle):
            if candle.is_closed:
                print(f"New candle closed: {candle.close}")

        stream.start()
    """

    # WebSocket URLs per exchange
    WS_URLS = {
        'binance': 'wss://stream.binance.com:9443/ws',
        'binance_futures': 'wss://fstream.binance.com/ws',
        'coinbase': 'wss://ws-feed.exchange.coinbase.com',
        'bybit': 'wss://stream.bybit.com/v5/public/spot',
        'kraken': 'wss://ws.kraken.com',
    }

    def __init__(
        self,
        exchange: str,
        symbol: str,
        interval: str = '1m',
        include_orderbook: bool = False
    ):
        """
        Initialize live stream.

        Args:
            exchange: Exchange name (binance, coinbase, bybit, kraken)
            symbol: Trading pair (BTC/USDT, ETH-USD, etc.)
            interval: Candle interval (1m, 5m, 15m, 1h, etc.)
            include_orderbook: Whether to stream order book data
        """
        self.exchange = exchange.lower()
        self.symbol = symbol
        self.interval = interval
        self.include_orderbook = include_orderbook

        # Normalize symbol for WebSocket
        self._ws_symbol = self._normalize_symbol(symbol)

        # State
        self._running = False
        self._connected = False
        self._ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._reconnect_count = 0
        self._max_reconnects = 10

        # Latest data
        self._last_tick: Optional[Tick] = None
        self._last_candle: Optional[LiveCandle] = None
        self._orderbook: Optional[OrderBook] = None

        # Callbacks
        self._tick_callbacks: List[Callable[[Tick], None]] = []
        self._candle_callbacks: List[Callable[[LiveCandle], None]] = []
        self._orderbook_callbacks: List[Callable[[OrderBook], None]] = []
        self._error_callbacks: List[Callable[[Exception], None]] = []

        # Message queue for async processing
        self._message_queue: Queue = Queue()

        logger.info(f"LiveStream initialized: {exchange} {symbol} @ {interval}")

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to exchange-specific WebSocket format."""
        # Remove common separators
        clean = symbol.upper().replace('-', '').replace('/', '').replace('_', '')

        if self.exchange == 'binance':
            return clean.lower()  # btcusdt
        elif self.exchange == 'coinbase':
            return symbol.replace('/', '-')  # BTC-USD
        elif self.exchange == 'bybit':
            return clean  # BTCUSDT
        elif self.exchange == 'kraken':
            return symbol.replace('/', '')  # BTCUSD
        return clean.lower()

    def _get_ws_url(self) -> str:
        """Get WebSocket URL with stream subscriptions."""
        base_url = self.WS_URLS.get(self.exchange)
        if not base_url:
            raise ValueError(f"Unsupported exchange: {self.exchange}")

        if self.exchange == 'binance':
            # Binance combined streams
            streams = [f"{self._ws_symbol}@kline_{self.interval}"]
            if self.include_orderbook:
                streams.append(f"{self._ws_symbol}@depth10@100ms")
            streams.append(f"{self._ws_symbol}@trade")
            return f"{base_url}/{'/'.join(streams)}"

        return base_url

    def _build_subscription_message(self) -> Optional[dict]:
        """Build subscription message for exchanges that require it."""
        if self.exchange == 'coinbase':
            channels = ["ticker", "heartbeat"]
            if self.include_orderbook:
                channels.append("level2")
            return {
                "type": "subscribe",
                "product_ids": [self.symbol.replace('/', '-')],
                "channels": channels
            }
        elif self.exchange == 'bybit':
            topics = [f"kline.{self.interval}.{self._ws_symbol}"]
            if self.include_orderbook:
                topics.append(f"orderbook.50.{self._ws_symbol}")
            return {
                "op": "subscribe",
                "args": topics
            }
        elif self.exchange == 'kraken':
            return {
                "event": "subscribe",
                "pair": [self.symbol],
                "subscription": {"name": "ticker"}
            }
        return None

    # =========================================================================
    # CALLBACK DECORATORS
    # =========================================================================

    def on_tick(self, callback: Callable[[Tick], None]) -> Callable:
        """Register tick callback (decorator or direct call)."""
        self._tick_callbacks.append(callback)
        return callback

    def on_candle(self, callback: Callable[[LiveCandle], None]) -> Callable:
        """Register candle callback."""
        self._candle_callbacks.append(callback)
        return callback

    def on_orderbook(self, callback: Callable[[OrderBook], None]) -> Callable:
        """Register order book callback."""
        self._orderbook_callbacks.append(callback)
        return callback

    def on_error(self, callback: Callable[[Exception], None]) -> Callable:
        """Register error callback."""
        self._error_callbacks.append(callback)
        return callback

    # =========================================================================
    # MESSAGE HANDLERS
    # =========================================================================

    def _handle_message(self, message: str):
        """Route message to appropriate handler."""
        try:
            data = json.loads(message)

            if self.exchange == 'binance':
                self._handle_binance_message(data)
            elif self.exchange == 'coinbase':
                self._handle_coinbase_message(data)
            elif self.exchange == 'bybit':
                self._handle_bybit_message(data)
            elif self.exchange == 'kraken':
                self._handle_kraken_message(data)

        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            self._notify_error(e)

    def _handle_binance_message(self, data: dict):
        """Handle Binance WebSocket messages."""
        # Combined stream format
        if 'stream' in data:
            stream = data['stream']
            payload = data['data']

            if '@kline_' in stream:
                self._process_binance_kline(payload)
            elif '@trade' in stream:
                self._process_binance_trade(payload)
            elif '@depth' in stream:
                self._process_binance_depth(payload)
        # Single stream format
        elif 'e' in data:
            if data['e'] == 'kline':
                self._process_binance_kline(data)
            elif data['e'] == 'trade':
                self._process_binance_trade(data)
            elif data['e'] == 'depthUpdate':
                self._process_binance_depth(data)

    def _process_binance_kline(self, data: dict):
        """Process Binance kline (candle) message."""
        k = data.get('k', data)
        candle = LiveCandle(
            timestamp=int(k['t']),
            open=float(k['o']),
            high=float(k['h']),
            low=float(k['l']),
            close=float(k['c']),
            volume=float(k['v']),
            is_closed=k.get('x', False),
            symbol=self.symbol
        )
        self._last_candle = candle

        # Also create tick from candle close
        tick = Tick(
            timestamp=int(k['t']),
            price=float(k['c']),
            volume=float(k['v']),
            symbol=self.symbol,
            exchange=self.exchange
        )
        self._last_tick = tick

        self._notify_candle(candle)
        self._notify_tick(tick)

    def _process_binance_trade(self, data: dict):
        """Process Binance trade message."""
        tick = Tick(
            timestamp=int(data['T']),
            price=float(data['p']),
            volume=float(data['q']),
            symbol=self.symbol,
            exchange=self.exchange
        )
        self._last_tick = tick
        self._notify_tick(tick)

    def _process_binance_depth(self, data: dict):
        """Process Binance order book depth."""
        orderbook = OrderBook(
            bids=[OrderBookLevel(float(b[0]), float(b[1])) for b in data.get('bids', [])[:10]],
            asks=[OrderBookLevel(float(a[0]), float(a[1])) for a in data.get('asks', [])[:10]],
            timestamp=int(time.time() * 1000)
        )
        self._orderbook = orderbook
        self._notify_orderbook(orderbook)

    def _handle_coinbase_message(self, data: dict):
        """Handle Coinbase WebSocket messages."""
        msg_type = data.get('type')

        if msg_type == 'ticker':
            tick = Tick(
                timestamp=int(datetime.utcnow().timestamp() * 1000),
                price=float(data.get('price', 0)),
                volume=float(data.get('volume_24h', 0)),
                bid=float(data.get('best_bid', 0)),
                ask=float(data.get('best_ask', 0)),
                symbol=self.symbol,
                exchange=self.exchange
            )
            self._last_tick = tick
            self._notify_tick(tick)

        elif msg_type == 'l2update':
            # Order book update
            pass  # TODO: Implement Coinbase L2 updates

    def _handle_bybit_message(self, data: dict):
        """Handle Bybit WebSocket messages."""
        topic = data.get('topic', '')

        if 'kline' in topic:
            for item in data.get('data', []):
                candle = LiveCandle(
                    timestamp=int(item['start']),
                    open=float(item['open']),
                    high=float(item['high']),
                    low=float(item['low']),
                    close=float(item['close']),
                    volume=float(item['volume']),
                    is_closed=item.get('confirm', False),
                    symbol=self.symbol
                )
                self._last_candle = candle

                tick = Tick(
                    timestamp=int(item['start']),
                    price=float(item['close']),
                    volume=float(item['volume']),
                    symbol=self.symbol,
                    exchange=self.exchange
                )
                self._last_tick = tick

                self._notify_candle(candle)
                self._notify_tick(tick)

    def _handle_kraken_message(self, data: Any):
        """Handle Kraken WebSocket messages."""
        if isinstance(data, list) and len(data) >= 2:
            # Kraken ticker format: [channelID, tickerData, channelName, pair]
            ticker_data = data[1]
            if isinstance(ticker_data, dict):
                # c = close, v = volume, b = bid, a = ask
                tick = Tick(
                    timestamp=int(time.time() * 1000),
                    price=float(ticker_data.get('c', [0])[0]),
                    volume=float(ticker_data.get('v', [0])[0]),
                    bid=float(ticker_data.get('b', [0])[0]),
                    ask=float(ticker_data.get('a', [0])[0]),
                    symbol=self.symbol,
                    exchange=self.exchange
                )
                self._last_tick = tick
                self._notify_tick(tick)

    # =========================================================================
    # NOTIFICATION METHODS
    # =========================================================================

    def _notify_tick(self, tick: Tick):
        """Notify all tick callbacks."""
        for callback in self._tick_callbacks:
            try:
                callback(tick)
            except Exception as e:
                logger.error(f"Tick callback error: {e}")

    def _notify_candle(self, candle: LiveCandle):
        """Notify all candle callbacks."""
        for callback in self._candle_callbacks:
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Candle callback error: {e}")

    def _notify_orderbook(self, orderbook: OrderBook):
        """Notify all order book callbacks."""
        for callback in self._orderbook_callbacks:
            try:
                callback(orderbook)
            except Exception as e:
                logger.error(f"Orderbook callback error: {e}")

    def _notify_error(self, error: Exception):
        """Notify all error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    # =========================================================================
    # WEBSOCKET HANDLERS
    # =========================================================================

    def _on_open(self, ws):
        """WebSocket connected."""
        logger.info(f"WebSocket connected to {self.exchange}")
        self._connected = True
        self._reconnect_count = 0

        # Send subscription message if required
        sub_msg = self._build_subscription_message()
        if sub_msg:
            ws.send(json.dumps(sub_msg))
            logger.info(f"Sent subscription: {sub_msg}")

    def _on_message(self, ws, message: str):
        """WebSocket message received."""
        self._handle_message(message)

    def _on_error(self, ws, error):
        """WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        self._notify_error(Exception(str(error)))

    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket closed."""
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
        self._connected = False

        # Attempt reconnection
        if self._running and self._reconnect_count < self._max_reconnects:
            self._reconnect_count += 1
            wait_time = min(2 ** self._reconnect_count, 60)  # Exponential backoff, max 60s
            logger.info(f"Reconnecting in {wait_time}s (attempt {self._reconnect_count})")
            time.sleep(wait_time)
            self._connect()
        elif self._reconnect_count >= self._max_reconnects:
            logger.error("Max reconnection attempts reached")
            self._running = False

    def _connect(self):
        """Establish WebSocket connection."""
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websocket-client not installed. Run: pip install websocket-client")

        try:
            ws_url = self._get_ws_url()
            logger.info(f"Connecting to: {ws_url}")

            self._ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            self._ws.run_forever(
                ping_interval=30,
                ping_timeout=10
            )
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._notify_error(e)

    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================

    def start(self, blocking: bool = False):
        """
        Start WebSocket streaming.

        Args:
            blocking: If True, blocks current thread. If False, runs in background.
        """
        if not WEBSOCKET_AVAILABLE:
            logger.error("websocket-client not installed")
            return False

        if self._running:
            logger.warning("Stream already running")
            return True

        self._running = True

        if blocking:
            self._connect()
        else:
            self._thread = threading.Thread(
                target=self._connect,
                daemon=True,
                name=f"LiveStream-{self.exchange}-{self.symbol}"
            )
            self._thread.start()

        logger.info(f"Live stream started: {self.exchange} {self.symbol}")
        return True

    def stop(self):
        """Stop WebSocket streaming."""
        logger.info("Stopping live stream...")
        self._running = False

        if self._ws:
            self._ws.close()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("Live stream stopped")

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected

    @property
    def is_running(self) -> bool:
        """Check if stream is running."""
        return self._running

    @property
    def last_tick(self) -> Optional[Tick]:
        """Get last received tick."""
        return self._last_tick

    @property
    def last_candle(self) -> Optional[LiveCandle]:
        """Get last received candle."""
        return self._last_candle

    @property
    def orderbook(self) -> Optional[OrderBook]:
        """Get current order book."""
        return self._orderbook

    def get_status(self) -> dict:
        """Get stream status."""
        return {
            'running': self._running,
            'connected': self._connected,
            'exchange': self.exchange,
            'symbol': self.symbol,
            'interval': self.interval,
            'last_price': self._last_tick.price if self._last_tick else None,
            'reconnect_count': self._reconnect_count,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_binance_stream(symbol: str, interval: str = '1m') -> LiveStream:
    """Create a Binance live stream."""
    return LiveStream('binance', symbol, interval)


def create_coinbase_stream(symbol: str) -> LiveStream:
    """Create a Coinbase live stream."""
    return LiveStream('coinbase', symbol)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    # Test with Binance
    stream = LiveStream('binance', 'BTC/USDT', '1m')

    @stream.on_tick
    def print_tick(tick: Tick):
        print(f"BTC: ${tick.price:,.2f} | Vol: {tick.volume:.4f}")

    @stream.on_candle
    def print_candle(candle: LiveCandle):
        status = "CLOSED" if candle.is_closed else "FORMING"
        print(f"Candle [{status}]: O:{candle.open:.2f} H:{candle.high:.2f} L:{candle.low:.2f} C:{candle.close:.2f}")

    try:
        stream.start(blocking=True)
    except KeyboardInterrupt:
        stream.stop()
        print("\nStream stopped")
