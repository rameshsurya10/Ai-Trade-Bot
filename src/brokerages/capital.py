"""
Capital.com Brokerage Implementation
====================================

Full brokerage integration for Capital.com Forex trading.

Features:
- REST API connection with session-based authentication
- Real-time price streaming via WebSocket
- Order management (Market, Limit, Stop)
- Position tracking
- Multi-currency balance support
- Symbol normalization (EUR/USD ↔ EURUSD)

Requirements:
- Capital.com account (demo or live)
- API key (generated from Settings > API integrations)
- 2FA must be enabled on account

Environment Variables:
    CAPITAL_API_KEY: Your Capital.com API key
    CAPITAL_EMAIL: Your Capital.com email
    CAPITAL_PASSWORD: Your Capital.com password
    CAPITAL_DEMO: Set to "true" for demo account

API Documentation: https://open-api.capital.com/

Usage:
    from src.brokerages.capital import CapitalBrokerage

    brokerage = CapitalBrokerage(demo=True)
    brokerage.connect()

    # Place order
    order = Order(
        symbol="EUR/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1000
    )
    ticket = brokerage.place_order(order)

    # Get positions
    positions = brokerage.get_positions()
"""

import os
import logging
import threading
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass

import requests

from .base import BaseBrokerage, CashBalance, Position
from .orders import Order, OrderTicket, OrderType, OrderSide, OrderStatus, TimeInForce
from .events import OrderEvent, OrderEventType
from .utils.symbol_normalizer import to_capital_format, from_capital_format

logger = logging.getLogger(__name__)

# Capital.com API URLs
CAPITAL_DEMO_URL = "https://demo-api-capital.backend-capital.com"
CAPITAL_LIVE_URL = "https://api-capital.backend-capital.com"
CAPITAL_DEMO_WS_URL = "wss://api-streaming-capital.backend-capital.com/connect"
CAPITAL_LIVE_WS_URL = "wss://api-streaming-capital.backend-capital.com/connect"


@dataclass
class CapitalConfig:
    """Capital.com configuration."""
    api_key: str
    email: str
    password: str
    demo: bool = True
    default_leverage: float = 30.0
    max_retries: int = 3
    timeout: int = 30

    @property
    def api_url(self) -> str:
        """Get API URL based on environment."""
        return CAPITAL_DEMO_URL if self.demo else CAPITAL_LIVE_URL

    @property
    def ws_url(self) -> str:
        """Get WebSocket URL based on environment."""
        return CAPITAL_DEMO_WS_URL if self.demo else CAPITAL_LIVE_WS_URL


class CapitalBrokerage(BaseBrokerage):
    """
    Capital.com Forex Brokerage Implementation.

    Implements BaseBrokerage interface for Capital.com REST API.

    Features:
    - Connect to demo or live accounts
    - Place, modify, and cancel orders
    - Track positions and balances
    - Stream real-time prices (via WebSocket)

    Rate Limits:
    - 10 requests per second (general)
    - 1 request per 0.1 seconds (order placement)
    - Session expires after 10 minutes of inactivity

    Example:
        brokerage = CapitalBrokerage(demo=True)
        brokerage.connect()

        # Check account
        balance = brokerage.get_cash_balance()
        print(f"Account balance: ${balance[0].amount:,.2f}")

        # Place market order
        order = Order(
            symbol="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        ticket = brokerage.place_order(order)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        demo: bool = True,
        default_leverage: float = 30.0
    ):
        """
        Initialize Capital.com brokerage.

        Args:
            api_key: Capital.com API key (or CAPITAL_API_KEY env var)
            email: Capital.com email (or CAPITAL_EMAIL env var)
            password: Capital.com password (or CAPITAL_PASSWORD env var)
            demo: Use demo account
            default_leverage: Default leverage (30:1 typical)
        """
        super().__init__("Capital.com")

        # Load config from env vars if not provided
        self.config = CapitalConfig(
            api_key=api_key or os.environ.get("CAPITAL_API_KEY", ""),
            email=email or os.environ.get("CAPITAL_EMAIL", ""),
            password=password or os.environ.get("CAPITAL_PASSWORD", ""),
            demo=demo if demo is not None else os.environ.get("CAPITAL_DEMO", "true").lower() == "true",
            default_leverage=default_leverage
        )

        # Session tokens
        self._cst: Optional[str] = None  # Client session token
        self._security_token: Optional[str] = None  # X-SECURITY-TOKEN
        self._account_id: Optional[str] = None

        # Session management
        self._session = requests.Session()
        self._last_request_time = 0.0
        self._session_lock = threading.Lock()

        # Streaming
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_stream = threading.Event()
        self._price_callbacks: List[Callable] = []

        # Position and price cache
        self._positions_cache: Dict[str, Position] = {}
        self._last_prices: Dict[str, Dict[str, float]] = {}

        # Keep session alive thread
        self._keepalive_thread: Optional[threading.Thread] = None
        self._stop_keepalive = threading.Event()

        logger.info(
            f"CapitalBrokerage initialized: "
            f"demo={self.config.demo}, "
            f"leverage={self.config.default_leverage}:1"
        )

    # ========== Connection ==========

    def connect(self) -> bool:
        """
        Connect to Capital.com API.

        Authenticates and establishes a session.

        Returns:
            True if connected successfully
        """
        if not self.config.api_key:
            logger.error("CAPITAL_API_KEY not configured")
            self._emit_message("Connection failed: No API key")
            return False

        if not self.config.email or not self.config.password:
            logger.error("CAPITAL_EMAIL or CAPITAL_PASSWORD not configured")
            self._emit_message("Connection failed: Missing credentials")
            return False

        try:
            # Create session
            url = f"{self.config.api_url}/api/v1/session"
            headers = {
                "X-CAP-API-KEY": self.config.api_key,
                "Content-Type": "application/json"
            }
            payload = {
                "identifier": self.config.email,
                "password": self.config.password
            }

            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )

            if response.status_code == 200:
                # Extract session tokens from headers
                self._cst = response.headers.get("CST")
                self._security_token = response.headers.get("X-SECURITY-TOKEN")

                # Get account info
                data = response.json()
                self._account_id = data.get("currentAccountId")

                self._is_connected = True
                env_name = "Demo" if self.config.demo else "Live"
                self._emit_message(f"Connected to Capital.com {env_name}")
                logger.info(f"Connected to Capital.com {env_name} account: {self._account_id}")

                # Start keepalive thread
                self._start_keepalive()

                return True
            else:
                error_msg = response.json().get("errorCode", response.text)
                logger.error(f"Failed to connect to Capital.com: {error_msg}")
                self._emit_message(f"Connection failed: {error_msg}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Capital.com: {e}")
            self._emit_message(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Capital.com API."""
        # Stop keepalive
        self._stop_keepalive.set()
        if self._keepalive_thread and self._keepalive_thread.is_alive():
            self._keepalive_thread.join(timeout=5)

        # Stop price stream
        self._stop_stream.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)

        # End session
        if self._is_connected:
            try:
                self._make_request("DELETE", "/api/v1/session")
            except Exception as e:
                logger.warning(f"Error ending session: {e}")

        self._is_connected = False
        self._cst = None
        self._security_token = None
        self._emit_message("Disconnected from Capital.com")
        logger.info("Disconnected from Capital.com")

    def _start_keepalive(self) -> None:
        """Start session keepalive thread."""
        self._stop_keepalive.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_worker,
            daemon=True
        )
        self._keepalive_thread.start()

    def _keepalive_worker(self) -> None:
        """Background worker to keep session alive."""
        while not self._stop_keepalive.wait(timeout=300):  # Ping every 5 minutes
            if self._is_connected:
                try:
                    # Simple ping - get accounts
                    self._make_request("GET", "/api/v1/accounts")
                    logger.debug("Session keepalive ping successful")
                except Exception as e:
                    logger.warning(f"Session keepalive failed: {e}")

    # ========== API Helpers ==========

    def _get_headers(self) -> Dict[str, str]:
        """Get authenticated headers."""
        headers = {
            "X-CAP-API-KEY": self.config.api_key,
            "Content-Type": "application/json"
        }
        if self._cst:
            headers["CST"] = self._cst
        if self._security_token:
            headers["X-SECURITY-TOKEN"] = self._security_token
        return headers

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        retry_count: int = 0
    ) -> Dict:
        """
        Make authenticated API request with rate limiting.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body
            retry_count: Current retry attempt

        Returns:
            Response JSON
        """
        with self._session_lock:
            # Rate limiting - 10 requests per second
            elapsed = time.time() - self._last_request_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

            url = f"{self.config.api_url}{endpoint}"
            headers = self._get_headers()

            try:
                if method == "GET":
                    response = self._session.get(
                        url, headers=headers, params=params,
                        timeout=self.config.timeout
                    )
                elif method == "POST":
                    response = self._session.post(
                        url, headers=headers, json=data,
                        timeout=self.config.timeout
                    )
                elif method == "PUT":
                    response = self._session.put(
                        url, headers=headers, json=data,
                        timeout=self.config.timeout
                    )
                elif method == "DELETE":
                    response = self._session.delete(
                        url, headers=headers,
                        timeout=self.config.timeout
                    )
                else:
                    raise ValueError(f"Unsupported method: {method}")

                self._last_request_time = time.time()

                # Handle session refresh
                if "CST" in response.headers:
                    self._cst = response.headers["CST"]
                if "X-SECURITY-TOKEN" in response.headers:
                    self._security_token = response.headers["X-SECURITY-TOKEN"]

                if response.status_code in [200, 201]:
                    return response.json() if response.text else {}
                elif response.status_code == 401 and retry_count < self.config.max_retries:
                    # Session expired - reconnect
                    logger.warning("Session expired, reconnecting...")
                    if self.connect():
                        return self._make_request(method, endpoint, params, data, retry_count + 1)
                else:
                    error_data = response.json() if response.text else {}
                    raise Exception(f"API error {response.status_code}: {error_data.get('errorCode', response.text)}")

            except requests.exceptions.RequestException as e:
                if retry_count < self.config.max_retries:
                    time.sleep(1)
                    return self._make_request(method, endpoint, params, data, retry_count + 1)
                raise

    # ========== Orders ==========

    def _validate_order(self, order: Order) -> Optional[str]:
        """
        Validate order parameters before submission.

        Args:
            order: Order to validate

        Returns:
            Error message if invalid, None if valid
        """
        if order.quantity <= 0:
            return f"Invalid quantity: {order.quantity}"

        if order.order_type == OrderType.LIMIT:
            if not order.limit_price or order.limit_price <= 0:
                return "Limit orders require a valid limit_price > 0"

        if order.order_type == OrderType.STOP_MARKET:
            if not order.stop_price or order.stop_price <= 0:
                return "Stop orders require a valid stop_price > 0"

        if order.stop_loss is not None and order.stop_loss <= 0:
            return f"Invalid stop_loss: {order.stop_loss}"

        if order.take_profit is not None and order.take_profit <= 0:
            return f"Invalid take_profit: {order.take_profit}"

        return None

    def place_order(self, order: Order) -> OrderTicket:
        """
        Submit order to Capital.com.

        Args:
            order: Order to submit

        Returns:
            OrderTicket for tracking/modifying
        """
        if not self._is_connected:
            order.status = OrderStatus.REJECTED
            return OrderTicket(order=order, _brokerage=self)

        # Validate order parameters
        validation_error = self._validate_order(order)
        if validation_error:
            logger.error(f"Order validation failed: {validation_error}")
            order.status = OrderStatus.REJECTED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.REJECTED,
                order=order,
                message=validation_error
            ))
            return OrderTicket(order=order, _brokerage=self)

        try:
            # Convert symbol to Capital.com format
            capital_symbol = to_capital_format(order.symbol)

            # Determine direction
            direction = "BUY" if order.side == OrderSide.BUY else "SELL"

            # Build order data
            order_data = {
                "epic": capital_symbol,
                "direction": direction,
                "size": order.quantity,
            }

            # Set order type specific fields
            if order.order_type == OrderType.MARKET:
                order_data["orderType"] = "MARKET"
            elif order.order_type == OrderType.LIMIT:
                order_data["orderType"] = "LIMIT"
                order_data["level"] = order.limit_price
            elif order.order_type == OrderType.STOP_MARKET:
                order_data["orderType"] = "STOP"
                order_data["level"] = order.stop_price

            # Add stop loss if specified
            if order.stop_loss:
                order_data["stopLevel"] = order.stop_loss
                order_data["guaranteedStop"] = False

            # Add take profit if specified
            if order.take_profit:
                order_data["profitLevel"] = order.take_profit

            # Submit order
            if order.order_type == OrderType.MARKET:
                # Market orders use positions endpoint
                response = self._make_request("POST", "/api/v1/positions", data=order_data)
            else:
                # Limit/Stop orders use workingorders endpoint
                response = self._make_request("POST", "/api/v1/workingorders", data=order_data)

            # Parse response
            deal_reference = response.get("dealReference")
            if deal_reference:
                # Confirm the deal
                confirm = self._make_request("GET", f"/api/v1/confirms/{deal_reference}")

                if confirm.get("dealStatus") == "ACCEPTED":
                    order.broker_id = confirm.get("dealId")
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.average_fill_price = confirm.get("level", 0)
                    order.filled_time = datetime.utcnow()

                    self._emit_order_event(OrderEvent(
                        order_id=order.id,
                        event_type=OrderEventType.FILLED,
                        order=order,
                        fill_price=order.average_fill_price,
                        fill_quantity=order.filled_quantity
                    ))
                else:
                    reason = confirm.get("reason", "Unknown")
                    order.status = OrderStatus.REJECTED
                    self._emit_order_event(OrderEvent(
                        order_id=order.id,
                        event_type=OrderEventType.REJECTED,
                        order=order,
                        message=reason
                    ))

            # Track order
            self._orders[order.id] = order

            logger.info(f"Order placed: {order}")
            return OrderTicket(order=order, _brokerage=self)

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.REJECTED,
                order=order,
                message=str(e)
            ))
            return OrderTicket(order=order, _brokerage=self)

    def update_order(
        self,
        order: Order,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            order: Order to modify
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            stop_price: New stop price (optional)

        Returns:
            True if update accepted
        """
        if not self._is_connected or not order.broker_id:
            return False

        # Cannot update both limit and stop price simultaneously
        if limit_price is not None and stop_price is not None:
            logger.error("Cannot update both limit_price and stop_price simultaneously")
            return False

        try:
            update_data = {}

            if limit_price is not None:
                update_data["level"] = limit_price
            elif stop_price is not None:
                update_data["level"] = stop_price

            if not update_data:
                return True  # Nothing to update

            self._make_request(
                "PUT",
                f"/api/v1/workingorders/{order.broker_id}",
                data=update_data
            )

            # Update local order
            if limit_price is not None:
                order.limit_price = limit_price
            if stop_price is not None:
                order.stop_price = stop_price

            logger.info(f"Order updated: {order}")
            return True

        except Exception as e:
            logger.error(f"Failed to update order: {e}")
            return False

    def cancel_order(self, order: Order) -> bool:
        """
        Cancel an open order.

        Args:
            order: Order to cancel

        Returns:
            True if cancellation accepted
        """
        if not self._is_connected or not order.broker_id:
            return False

        try:
            self._make_request(
                "DELETE",
                f"/api/v1/workingorders/{order.broker_id}"
            )

            order.status = OrderStatus.CANCELED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.CANCELED,
                order=order
            ))

            logger.info(f"Order canceled: {order}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    # ========== Account ==========

    def get_cash_balance(self) -> List[CashBalance]:
        """
        Get cash balances.

        Returns:
            List of cash balances
        """
        if not self._is_connected:
            return []

        try:
            response = self._make_request("GET", "/api/v1/accounts")

            balances = []
            for account in response.get("accounts", []):
                currency = account.get("currency", "USD")
                balance = float(account.get("balance", {}).get("balance", 0))
                available = float(account.get("balance", {}).get("available", 0))

                balances.append(CashBalance(
                    currency=currency,
                    amount=balance,
                    available=available
                ))

            return balances

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return []

    def get_positions(self) -> List[Position]:
        """
        Get all current positions.

        Returns:
            List of positions
        """
        if not self._is_connected:
            return []

        try:
            response = self._make_request("GET", "/api/v1/positions")

            positions = []
            for pos in response.get("positions", []):
                market = pos.get("market", {})
                position_data = pos.get("position", {})

                symbol = from_capital_format(market.get("epic", ""))
                direction = position_data.get("direction", "BUY")
                size = float(position_data.get("size", 0))
                open_level = float(position_data.get("level", 0))
                current_level = float(market.get("bid", 0))

                # Calculate P&L
                if direction == "BUY":
                    unrealized = (current_level - open_level) * size
                    side = "long"
                else:
                    unrealized = (open_level - current_level) * size
                    side = "short"

                market_value = size * current_level
                cost_basis = size * open_level
                unrealized_pct = (unrealized / cost_basis * 100) if cost_basis > 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    quantity=size,
                    average_price=open_level,
                    market_value=market_value,
                    unrealized_pnl=unrealized,
                    unrealized_pnl_percent=unrealized_pct,
                    side=side
                ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_account_value(self) -> float:
        """
        Get total account value (equity).

        Returns:
            Total account value
        """
        if not self._is_connected:
            return 0.0

        try:
            response = self._make_request("GET", "/api/v1/accounts")

            for account in response.get("accounts", []):
                if account.get("accountId") == self._account_id:
                    return float(account.get("balance", {}).get("balance", 0))

            return 0.0

        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return 0.0

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive account summary.

        Returns:
            Dict with account details
        """
        if not self._is_connected:
            return {}

        try:
            response = self._make_request("GET", "/api/v1/accounts")

            for account in response.get("accounts", []):
                if account.get("accountId") == self._account_id:
                    balance_data = account.get("balance", {})
                    return {
                        "id": account.get("accountId"),
                        "name": account.get("accountName"),
                        "currency": account.get("currency"),
                        "balance": float(balance_data.get("balance", 0)),
                        "available": float(balance_data.get("available", 0)),
                        "deposit": float(balance_data.get("deposit", 0)),
                        "profit_loss": float(balance_data.get("profitLoss", 0)),
                    }

            return {}

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}

    # ========== Market Data ==========

    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid/ask price for a symbol.

        Args:
            symbol: Currency pair

        Returns:
            Dict with 'bid', 'ask', 'mid' or None
        """
        if not self._is_connected:
            return None

        try:
            capital_symbol = to_capital_format(symbol)

            response = self._make_request(
                "GET",
                f"/api/v1/markets/{capital_symbol}"
            )

            snapshot = response.get("snapshot", {})
            bid = float(snapshot.get("bid", 0))
            ask = float(snapshot.get("offer", 0))
            mid = (bid + ask) / 2

            return {"bid": bid, "ask": ask, "mid": mid}

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_candles(
        self,
        symbol: str,
        resolution: str = "HOUR",
        max_candles: int = 100
    ) -> List[Dict]:
        """
        Get historical candles.

        Args:
            symbol: Currency pair
            resolution: Candle timeframe (MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK)
            max_candles: Number of candles

        Returns:
            List of candle dicts with open, high, low, close, volume
        """
        if not self._is_connected:
            return []

        try:
            capital_symbol = to_capital_format(symbol)

            response = self._make_request(
                "GET",
                f"/api/v1/prices/{capital_symbol}",
                params={
                    "resolution": resolution,
                    "max": max_candles
                }
            )

            candles = []
            for price in response.get("prices", []):
                candles.append({
                    "time": price.get("snapshotTime"),
                    "open": float(price.get("openPrice", {}).get("mid", 0)),
                    "high": float(price.get("highPrice", {}).get("mid", 0)),
                    "low": float(price.get("lowPrice", {}).get("mid", 0)),
                    "close": float(price.get("closePrice", {}).get("mid", 0)),
                    "volume": int(price.get("lastTradedVolume", 0))
                })

            return candles

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return []

    # ========== Position Management ==========

    def close_position(self, symbol: str, percentage: float = 100.0) -> bool:
        """
        Close a position.

        Args:
            symbol: Currency pair
            percentage: Percentage to close (default 100%)

        Returns:
            True if successful
        """
        if not self._is_connected:
            return False

        try:
            # Get current position
            position = self.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False

            # Find the deal ID for this position
            response = self._make_request("GET", "/api/v1/positions")

            capital_symbol = to_capital_format(symbol)
            deal_id = None

            for pos in response.get("positions", []):
                if pos.get("market", {}).get("epic") == capital_symbol:
                    deal_id = pos.get("position", {}).get("dealId")
                    break

            if not deal_id:
                logger.error(f"Could not find deal ID for {symbol}")
                return False

            # Close position
            close_data = {}
            if percentage < 100:
                close_data["size"] = position.quantity * percentage / 100

            self._make_request(
                "DELETE",
                f"/api/v1/positions/{deal_id}",
                data=close_data if close_data else None
            )

            logger.info(f"Closed {percentage}% of {symbol} position")
            return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def get_trade_history(self, count: int = 50) -> List[Dict]:
        """
        Get recent trade history.

        Args:
            count: Number of trades to fetch

        Returns:
            List of trade dicts
        """
        if not self._is_connected:
            return []

        try:
            response = self._make_request(
                "GET",
                "/api/v1/history/activity",
                params={"maxResults": count}
            )

            trades = []
            for activity in response.get("activities", []):
                if activity.get("type") == "POSITION":
                    trades.append({
                        "id": activity.get("dealId"),
                        "symbol": from_capital_format(activity.get("epic", "")),
                        "direction": activity.get("direction"),
                        "size": float(activity.get("size", 0)),
                        "price": float(activity.get("level", 0)),
                        "profit_loss": float(activity.get("profit", 0)),
                        "time": activity.get("date"),
                    })

            return trades

        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []

    # ========== Streaming ==========

    def start_price_stream(
        self,
        symbols: List[str],
        callback: Callable[[str, float, float], None]
    ) -> None:
        """
        Start streaming prices for symbols.

        Note: Capital.com WebSocket requires a separate implementation.
        This is a polling fallback.

        Args:
            symbols: List of currency pairs
            callback: Function called with (symbol, bid, ask) on each tick
        """
        if not self._is_connected:
            logger.error("Cannot start stream: not connected")
            return

        self._price_callbacks.append(callback)

        if self._stream_thread and self._stream_thread.is_alive():
            logger.info("Price stream already running")
            return

        self._stop_stream.clear()
        self._stream_thread = threading.Thread(
            target=self._price_poll_worker,
            args=(symbols,),
            daemon=True
        )
        self._stream_thread.start()

    def stop_price_stream(self) -> None:
        """Stop price streaming."""
        self._stop_stream.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=5)
        self._price_callbacks.clear()

    def _price_poll_worker(self, symbols: List[str]) -> None:
        """
        Background worker for price polling.

        Note: Uses polling as WebSocket requires additional setup.
        """
        while not self._stop_stream.is_set():
            for symbol in symbols:
                if self._stop_stream.is_set():
                    break

                try:
                    price = self.get_current_price(symbol)
                    if price:
                        bid = price["bid"]
                        ask = price["ask"]

                        # Cache latest price
                        self._last_prices[symbol] = {
                            "bid": bid,
                            "ask": ask,
                            "mid": price["mid"],
                            "time": datetime.utcnow()
                        }

                        # Notify callbacks
                        for callback in self._price_callbacks:
                            try:
                                callback(symbol, bid, ask)
                            except Exception as e:
                                logger.error(f"Price callback error: {e}")

                except Exception as e:
                    logger.error(f"Price poll error for {symbol}: {e}")

            # Respect rate limits - poll every second
            self._stop_stream.wait(timeout=1.0)

    # ========== Utility Methods ==========

    def get_available_markets(self, search_term: str = "") -> List[Dict]:
        """
        Search for available markets.

        Args:
            search_term: Search query

        Returns:
            List of market info dicts
        """
        if not self._is_connected:
            return []

        try:
            params = {}
            if search_term:
                params["searchTerm"] = search_term

            response = self._make_request(
                "GET",
                "/api/v1/markets",
                params=params
            )

            markets = []
            for market in response.get("markets", []):
                markets.append({
                    "epic": market.get("epic"),
                    "name": market.get("instrumentName"),
                    "type": market.get("instrumentType"),
                    "expiry": market.get("expiry"),
                })

            return markets

        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return []
