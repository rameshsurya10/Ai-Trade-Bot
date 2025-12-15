"""
Prometheus Metrics
==================
Export metrics for monitoring and alerting.

Metrics:
- Trading signals generated
- Prediction confidence
- Data update latency
- Error counts
- Account balance/P&L
"""

import logging
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed, metrics disabled")


class MetricsCollector:
    """
    Prometheus metrics collector for trading bot.

    Collects and exposes metrics for:
    - Signal generation
    - Trade execution
    - Data collection
    - System health

    Usage:
        from src.monitoring import metrics

        # Record a signal
        metrics.record_signal("BTC/USDT", "BUY", 0.75)

        # Track prediction latency
        with metrics.prediction_timer():
            result = model.predict(features)

        # Update account balance
        metrics.update_balance(10500.0, 500.0)
    """

    def __init__(self):
        """Initialize metrics."""
        if not PROMETHEUS_AVAILABLE:
            self._enabled = False
            return

        self._enabled = True

        # =================================================================
        # SIGNAL METRICS
        # =================================================================

        self.signals_total = Counter(
            'trading_signals_total',
            'Total trading signals generated',
            ['symbol', 'signal_type', 'strength']
        )

        self.signal_confidence = Gauge(
            'trading_signal_confidence',
            'Latest signal confidence',
            ['symbol']
        )

        self.active_signals = Gauge(
            'trading_active_signals',
            'Currently active signals',
            ['symbol', 'signal_type']
        )

        # =================================================================
        # PREDICTION METRICS
        # =================================================================

        self.predictions_total = Counter(
            'trading_predictions_total',
            'Total predictions made',
            ['symbol', 'using_ml']
        )

        self.prediction_latency = Histogram(
            'trading_prediction_latency_seconds',
            'Time to make prediction',
            ['symbol'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )

        self.model_probability = Gauge(
            'trading_model_probability',
            'Model output probability',
            ['symbol']
        )

        # =================================================================
        # DATA COLLECTION METRICS
        # =================================================================

        self.data_updates_total = Counter(
            'trading_data_updates_total',
            'Total data updates received',
            ['symbol', 'source']  # source: websocket, polling
        )

        self.data_latency = Histogram(
            'trading_data_latency_seconds',
            'Data update latency',
            ['symbol', 'source'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
        )

        self.websocket_status = Gauge(
            'trading_websocket_connected',
            'WebSocket connection status (1=connected, 0=disconnected)',
            ['exchange']
        )

        self.last_price = Gauge(
            'trading_last_price',
            'Latest price for symbol',
            ['symbol']
        )

        # =================================================================
        # TRADING METRICS
        # =================================================================

        self.trades_total = Counter(
            'trading_trades_total',
            'Total trades executed',
            ['symbol', 'side', 'outcome']  # outcome: win, loss
        )

        self.trade_pnl = Histogram(
            'trading_trade_pnl',
            'Trade profit/loss distribution',
            ['symbol'],
            buckets=(-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000)
        )

        self.position_size = Gauge(
            'trading_position_size',
            'Current position size',
            ['symbol']
        )

        self.position_pnl = Gauge(
            'trading_position_unrealized_pnl',
            'Unrealized P&L for position',
            ['symbol']
        )

        # =================================================================
        # ACCOUNT METRICS
        # =================================================================

        self.account_balance = Gauge(
            'trading_account_balance',
            'Current account balance',
            ['currency']
        )

        self.account_equity = Gauge(
            'trading_account_equity',
            'Total account equity',
            ['currency']
        )

        self.account_pnl = Gauge(
            'trading_account_pnl',
            'Total account P&L',
            ['type']  # realized, unrealized
        )

        self.win_rate = Gauge(
            'trading_win_rate',
            'Trade win rate percentage'
        )

        # =================================================================
        # SYSTEM METRICS
        # =================================================================

        self.errors_total = Counter(
            'trading_errors_total',
            'Total errors',
            ['component', 'error_type']
        )

        self.api_requests_total = Counter(
            'trading_api_requests_total',
            'Total API requests',
            ['exchange', 'endpoint', 'status']
        )

        self.api_latency = Histogram(
            'trading_api_latency_seconds',
            'API request latency',
            ['exchange', 'endpoint'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )

        self.component_status = Gauge(
            'trading_component_status',
            'Component health status (1=healthy, 0=unhealthy)',
            ['component']
        )

        logger.info("Prometheus metrics initialized")

    @property
    def enabled(self) -> bool:
        return self._enabled

    # =================================================================
    # SIGNAL METHODS
    # =================================================================

    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        strength: str = "medium"
    ):
        """Record a trading signal."""
        if not self._enabled:
            return

        self.signals_total.labels(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength
        ).inc()

        self.signal_confidence.labels(symbol=symbol).set(confidence)

    def set_active_signal(self, symbol: str, signal_type: str, active: bool = True):
        """Set active signal status."""
        if not self._enabled:
            return

        self.active_signals.labels(
            symbol=symbol,
            signal_type=signal_type
        ).set(1 if active else 0)

    # =================================================================
    # PREDICTION METHODS
    # =================================================================

    def record_prediction(
        self,
        symbol: str,
        probability: float,
        using_ml: bool,
        latency_seconds: float
    ):
        """Record a model prediction."""
        if not self._enabled:
            return

        self.predictions_total.labels(
            symbol=symbol,
            using_ml=str(using_ml)
        ).inc()

        self.prediction_latency.labels(symbol=symbol).observe(latency_seconds)
        self.model_probability.labels(symbol=symbol).set(probability)

    def prediction_timer(self, symbol: str = "default"):
        """Context manager for timing predictions."""
        if not self._enabled:
            return _DummyTimer()
        return _Timer(self.prediction_latency.labels(symbol=symbol))

    # =================================================================
    # DATA METHODS
    # =================================================================

    def record_data_update(
        self,
        symbol: str,
        source: str,
        price: float,
        latency_seconds: float = 0
    ):
        """Record data update."""
        if not self._enabled:
            return

        self.data_updates_total.labels(symbol=symbol, source=source).inc()
        self.last_price.labels(symbol=symbol).set(price)

        if latency_seconds > 0:
            self.data_latency.labels(symbol=symbol, source=source).observe(latency_seconds)

    def set_websocket_status(self, exchange: str, connected: bool):
        """Set WebSocket connection status."""
        if not self._enabled:
            return

        self.websocket_status.labels(exchange=exchange).set(1 if connected else 0)

    # =================================================================
    # TRADING METHODS
    # =================================================================

    def record_trade(
        self,
        symbol: str,
        side: str,
        pnl: float
    ):
        """Record a completed trade."""
        if not self._enabled:
            return

        outcome = "win" if pnl > 0 else "loss"
        self.trades_total.labels(symbol=symbol, side=side, outcome=outcome).inc()
        self.trade_pnl.labels(symbol=symbol).observe(pnl)

    def update_position(
        self,
        symbol: str,
        size: float,
        unrealized_pnl: float
    ):
        """Update position metrics."""
        if not self._enabled:
            return

        self.position_size.labels(symbol=symbol).set(size)
        self.position_pnl.labels(symbol=symbol).set(unrealized_pnl)

    # =================================================================
    # ACCOUNT METHODS
    # =================================================================

    def update_account(
        self,
        balance: float,
        equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        win_rate_pct: float,
        currency: str = "USDT"
    ):
        """Update account metrics."""
        if not self._enabled:
            return

        self.account_balance.labels(currency=currency).set(balance)
        self.account_equity.labels(currency=currency).set(equity)
        self.account_pnl.labels(type="realized").set(realized_pnl)
        self.account_pnl.labels(type="unrealized").set(unrealized_pnl)
        self.win_rate.set(win_rate_pct)

    # =================================================================
    # SYSTEM METHODS
    # =================================================================

    def record_error(self, component: str, error_type: str):
        """Record an error."""
        if not self._enabled:
            return

        self.errors_total.labels(component=component, error_type=error_type).inc()

    def record_api_request(
        self,
        exchange: str,
        endpoint: str,
        status: str,
        latency_seconds: float
    ):
        """Record API request."""
        if not self._enabled:
            return

        self.api_requests_total.labels(
            exchange=exchange,
            endpoint=endpoint,
            status=status
        ).inc()

        self.api_latency.labels(
            exchange=exchange,
            endpoint=endpoint
        ).observe(latency_seconds)

    def set_component_status(self, component: str, healthy: bool):
        """Set component health status."""
        if not self._enabled:
            return

        self.component_status.labels(component=component).set(1 if healthy else 0)


class _Timer:
    """Timer context manager for Histogram observation."""

    def __init__(self, histogram):
        self.histogram = histogram
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.histogram.observe(time.time() - self.start)


class _DummyTimer:
    """Dummy timer when metrics disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics endpoint."""

    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass


def start_metrics_server(port: int = 9090, host: str = "0.0.0.0") -> Optional[HTTPServer]:
    """
    Start Prometheus metrics server.

    Args:
        port: Port to listen on
        host: Host to bind to

    Returns:
        HTTPServer instance or None if not available
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus not available, metrics server not started")
        return None

    server = HTTPServer((host, port), MetricsHandler)

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
        name="MetricsServer"
    )
    thread.start()

    logger.info(f"Metrics server started on http://{host}:{port}/metrics")
    return server


# Global metrics instance
metrics = MetricsCollector()


# Decorator for timing functions
def timed(metric_name: str = "default"):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with metrics.prediction_timer(metric_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# CLI usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if PROMETHEUS_AVAILABLE:
        # Start server
        start_metrics_server(port=9090)

        # Record some test metrics
        metrics.record_signal("BTC/USDT", "BUY", 0.75, "strong")
        metrics.record_prediction("BTC/USDT", 0.72, True, 0.05)
        metrics.record_data_update("BTC/USDT", "websocket", 50000.0, 0.001)
        metrics.update_account(10000, 10500, 500, 100, 65.0)

        print("Metrics server running on http://localhost:9090/metrics")
        print("Press Ctrl+C to stop")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
    else:
        print("prometheus_client not installed")
        print("Install with: pip install prometheus-client")
