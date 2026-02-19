"""
MT5 Bridge Client (Linux-side)
==============================

TCP client that proxies MetaTrader5 API calls to the Windows bridge server.
Implements the same interface as the MetaTrader5 Python package so it can
be used as a drop-in replacement.

Usage:
    client = MT5BridgeClient(host='192.168.1.100', port=5555)

    # Same API as MetaTrader5 package
    client.initialize()
    client.login(12345, password='demo', server='MetaQuotes-Demo')
    rates = client.copy_rates_from_pos('EURUSD', client.TIMEFRAME_H1, 0, 100)
    tick = client.symbol_info_tick('EURUSD')
    client.shutdown()
"""

import logging
import socket
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from .protocol import MT5Request, read_message

logger = logging.getLogger(__name__)


# MT5 timeframe constants (replicated for cross-platform use)
TIMEFRAME_M1 = 1
TIMEFRAME_M2 = 2
TIMEFRAME_M3 = 3
TIMEFRAME_M4 = 4
TIMEFRAME_M5 = 5
TIMEFRAME_M6 = 6
TIMEFRAME_M10 = 10
TIMEFRAME_M12 = 12
TIMEFRAME_M15 = 15
TIMEFRAME_M20 = 20
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H2 = 16386
TIMEFRAME_H3 = 16387
TIMEFRAME_H4 = 16388
TIMEFRAME_H6 = 16390
TIMEFRAME_H8 = 16392
TIMEFRAME_H12 = 16396
TIMEFRAME_D1 = 16408
TIMEFRAME_W1 = 32769
TIMEFRAME_MN1 = 49153

# Trade action constants
TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_MODIFY = 7
TRADE_ACTION_REMOVE = 8

# Order type constants
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5

# Filling modes
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

# Time types
ORDER_TIME_GTC = 0
ORDER_TIME_DAY = 1

# Trade return codes
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_REQUOTE = 10004


@dataclass
class _NamedTupleProxy:
    """
    Proxy object that allows attribute access on dict data.

    Mimics MT5's named tuple responses (account_info, symbol_info, etc.)
    so that code written for direct MT5 access works unchanged.
    """
    _data: dict

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")

    def _asdict(self):
        return dict(self._data)


class _RatesArray:
    """
    Proxy for numpy structured array returned by copy_rates_*.

    Supports both dict-key access (rate['time']) and iteration.
    """

    def __init__(self, data: list):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return _RatesArray(self._data[idx])
        row = self._data[idx]
        return _RateRow(row)

    def __iter__(self):
        for row in self._data:
            yield _RateRow(row)


class _RateRow:
    """Single row from rates array, supports dict-key access."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattribute__(name)
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"No attribute '{name}'")


class MT5BridgeClient:
    """
    TCP client proxying MetaTrader5 API calls to bridge server.

    Drop-in replacement for the MetaTrader5 module. All method calls
    are forwarded to the Windows bridge server via TCP.
    """

    # Expose timeframe constants as class attributes
    TIMEFRAME_M1 = TIMEFRAME_M1
    TIMEFRAME_M5 = TIMEFRAME_M5
    TIMEFRAME_M15 = TIMEFRAME_M15
    TIMEFRAME_M30 = TIMEFRAME_M30
    TIMEFRAME_H1 = TIMEFRAME_H1
    TIMEFRAME_H4 = TIMEFRAME_H4
    TIMEFRAME_D1 = TIMEFRAME_D1
    TIMEFRAME_W1 = TIMEFRAME_W1

    # Trade actions
    TRADE_ACTION_DEAL = TRADE_ACTION_DEAL
    TRADE_ACTION_PENDING = TRADE_ACTION_PENDING
    TRADE_ACTION_SLTP = TRADE_ACTION_SLTP
    TRADE_ACTION_MODIFY = TRADE_ACTION_MODIFY
    TRADE_ACTION_REMOVE = TRADE_ACTION_REMOVE

    # Order types
    ORDER_TYPE_BUY = ORDER_TYPE_BUY
    ORDER_TYPE_SELL = ORDER_TYPE_SELL
    ORDER_TYPE_BUY_LIMIT = ORDER_TYPE_BUY_LIMIT
    ORDER_TYPE_SELL_LIMIT = ORDER_TYPE_SELL_LIMIT
    ORDER_TYPE_BUY_STOP = ORDER_TYPE_BUY_STOP
    ORDER_TYPE_SELL_STOP = ORDER_TYPE_SELL_STOP

    # Filling modes
    ORDER_FILLING_FOK = ORDER_FILLING_FOK
    ORDER_FILLING_IOC = ORDER_FILLING_IOC
    ORDER_FILLING_RETURN = ORDER_FILLING_RETURN

    # Trade return codes
    TRADE_RETCODE_DONE = TRADE_RETCODE_DONE
    TRADE_RETCODE_PLACED = TRADE_RETCODE_PLACED

    def __init__(self, host: str = 'localhost', port: int = 5555, timeout: float = 30.0):
        """
        Initialize bridge client.

        Args:
            host: Bridge server hostname/IP
            port: Bridge server port
            timeout: Socket timeout in seconds
        """
        self._host = host
        self._port = port
        self._timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._last_error_val = None

    def _connect(self) -> None:
        """Establish TCP connection to bridge server."""
        if self._socket:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.settimeout(self._timeout)
        self._socket.connect((self._host, self._port))
        logger.info(f"Connected to MT5 bridge at {self._host}:{self._port}")

    def _disconnect(self) -> None:
        """Close TCP connection."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

    def _call(self, method: str, *args, **kwargs) -> Any:
        """
        Send a method call to the bridge server and return the result.

        Args:
            method: MT5 function name
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Deserialized result from the server
        """
        with self._lock:
            try:
                self._connect()

                request = MT5Request(
                    id=str(uuid.uuid4())[:8],
                    method=method,
                    args=list(args),
                    kwargs=kwargs,
                )
                self._socket.sendall(request.to_bytes())

                msg = read_message(self._socket)
                if msg is None:
                    self._disconnect()
                    raise ConnectionError("Bridge server disconnected")

                if not msg.get('success', False):
                    error = msg.get('error', 'Unknown error')
                    self._last_error_val = (-1, error)
                    logger.error(f"Bridge error for {method}: {error}")
                    return None

                return self._deserialize(method, msg.get('data'))

            except (ConnectionError, socket.error, OSError) as e:
                logger.error(f"Bridge connection error: {e}")
                self._disconnect()
                self._last_error_val = (-1, str(e))
                return None

    def _deserialize(self, method: str, data: Any) -> Any:
        """
        Deserialize response data based on method type.

        Wraps dicts in proxy objects for attribute access compatibility.
        """
        if data is None:
            return None

        # Methods that return structured arrays
        if method.startswith('copy_rates') or method.startswith('copy_ticks'):
            if isinstance(data, list):
                return _RatesArray(data)
            return data

        # Methods that return named tuples
        if method in ('account_info', 'terminal_info', 'symbol_info', 'symbol_info_tick'):
            if isinstance(data, dict):
                return _NamedTupleProxy(_data=data)
            return data

        # Methods that return tuples of named tuples
        if method in ('positions_get', 'orders_get', 'symbols_get',
                       'history_orders_get', 'history_deals_get'):
            if isinstance(data, list):
                return tuple(
                    _NamedTupleProxy(_data=item) if isinstance(item, dict) else item
                    for item in data
                )
            return data

        # order_send returns a named tuple
        if method == 'order_send':
            if isinstance(data, dict):
                return _NamedTupleProxy(_data=data)
            return data

        return data

    # ========== MT5 API Methods ==========

    def initialize(self, path: str = None, **kwargs) -> bool:
        """Initialize MT5 terminal."""
        if path:
            kwargs['path'] = path
        result = self._call('initialize', **kwargs)
        return bool(result)

    def shutdown(self) -> None:
        """Shutdown MT5."""
        try:
            self._call('shutdown')
        except Exception:
            pass
        self._disconnect()

    def login(self, login: int, password: str = '', server: str = '', **kwargs) -> bool:
        """Login to MT5 account."""
        result = self._call('login', login, password=password, server=server, **kwargs)
        return bool(result)

    def account_info(self):
        """Get account info."""
        return self._call('account_info')

    def terminal_info(self):
        """Get terminal info."""
        return self._call('terminal_info')

    def symbol_info(self, symbol: str):
        """Get symbol info."""
        return self._call('symbol_info', symbol)

    def symbol_info_tick(self, symbol: str):
        """Get latest tick for symbol."""
        return self._call('symbol_info_tick', symbol)

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        """Enable/disable symbol in market watch."""
        result = self._call('symbol_select', symbol, enable)
        return bool(result)

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int):
        """Get rates from position."""
        return self._call('copy_rates_from_pos', symbol, timeframe, start_pos, count)

    def copy_rates_range(self, symbol: str, timeframe: int, date_from, date_to):
        """Get rates in date range."""
        return self._call('copy_rates_range', symbol, timeframe, date_from, date_to)

    def order_send(self, request: dict):
        """Send trade order."""
        return self._call('order_send', request)

    def order_check(self, request: dict):
        """Check trade order (dry run)."""
        return self._call('order_check', request)

    def positions_get(self, **kwargs):
        """Get open positions."""
        return self._call('positions_get', **kwargs)

    def positions_total(self) -> int:
        """Get total number of open positions."""
        result = self._call('positions_total')
        return int(result) if result is not None else 0

    def orders_get(self, **kwargs):
        """Get pending orders."""
        return self._call('orders_get', **kwargs)

    def last_error(self):
        """Get last error."""
        result = self._call('last_error')
        if result is None and self._last_error_val:
            return self._last_error_val
        return result
