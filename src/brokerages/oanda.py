"""
OANDA Brokerage Implementation
==============================

Full brokerage integration for OANDA Forex trading.

Features:
- REST API connection with Bearer token authentication
- Real-time price streaming
- Order management (Market, Limit, Stop, Trailing Stop)
- Position tracking (handles OANDA's long/short separation)
- Multi-currency balance support
- Symbol normalization (EUR/USD ↔ EUR_USD)

Requirements:
- OANDA API key (from OANDA portal)
- OANDA Account ID
- oandapyV20 package

Environment Variables:
    OANDA_API_KEY: Your OANDA API key
    OANDA_ACCOUNT_ID: Your OANDA account ID
    OANDA_PRACTICE: Set to "true" for practice/demo account

Usage:
    from src.brokerages.oanda import OandaBrokerage

    brokerage = OandaBrokerage(practice=True)
    brokerage.connect()

    # Place order
    order = Order(
        symbol="EUR/USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10000
    )
    ticket = brokerage.place_order(order)

    # Get positions
    positions = brokerage.get_positions()
"""

import os
import logging
import threading
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass

from .base import BaseBrokerage, CashBalance, Position
from .orders import Order, OrderTicket, OrderType, OrderSide, OrderStatus, TimeInForce
from .events import OrderEvent, OrderEventType
from .utils.symbol_normalizer import to_oanda_format, from_oanda_format

logger = logging.getLogger(__name__)

# OANDA API URLs
OANDA_PRACTICE_URL = "https://api-fxpractice.oanda.com"
OANDA_LIVE_URL = "https://api-fxtrade.oanda.com"
OANDA_STREAM_PRACTICE_URL = "https://stream-fxpractice.oanda.com"
OANDA_STREAM_LIVE_URL = "https://stream-fxtrade.oanda.com"


@dataclass
class OandaConfig:
    """OANDA configuration."""
    api_key: str
    account_id: str
    practice: bool = True
    default_leverage: float = 50.0
    max_retries: int = 3
    timeout: int = 30

    @property
    def api_url(self) -> str:
        """Get API URL based on environment."""
        return OANDA_PRACTICE_URL if self.practice else OANDA_LIVE_URL

    @property
    def stream_url(self) -> str:
        """Get streaming URL based on environment."""
        return OANDA_STREAM_PRACTICE_URL if self.practice else OANDA_STREAM_LIVE_URL


class OandaBrokerage(BaseBrokerage):
    """
    OANDA Forex Brokerage Implementation.

    Implements BaseBrokerage interface for OANDA REST API.

    Features:
    - Connect to practice or live accounts
    - Place, modify, and cancel orders
    - Track positions and balances
    - Stream real-time prices

    Example:
        brokerage = OandaBrokerage(practice=True)
        brokerage.connect()

        # Check account
        balance = brokerage.get_cash_balance()
        print(f"Account balance: ${balance[0].amount:,.2f}")

        # Place market order
        order = Order(
            symbol="EUR/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10000  # 0.1 lot
        )
        ticket = brokerage.place_order(order)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        practice: bool = True,
        default_leverage: float = 50.0
    ):
        """
        Initialize OANDA brokerage.

        Args:
            api_key: OANDA API key (or OANDA_API_KEY env var)
            account_id: OANDA account ID (or OANDA_ACCOUNT_ID env var)
            practice: Use practice/demo account
            default_leverage: Default leverage (50:1 for US)
        """
        super().__init__("OANDA")

        # Load config from env vars if not provided
        self.config = OandaConfig(
            api_key=api_key or os.environ.get("OANDA_API_KEY", ""),
            account_id=account_id or os.environ.get("OANDA_ACCOUNT_ID", ""),
            practice=practice or os.environ.get("OANDA_PRACTICE", "true").lower() == "true",
            default_leverage=default_leverage
        )

        # API client
        self._api = None
        self._ctx = None

        # Streaming
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_stream = threading.Event()
        self._price_callbacks: List[Callable] = []

        # Position and order cache
        self._positions_cache: Dict[str, Position] = {}
        self._last_prices: Dict[str, Dict[str, float]] = {}

        logger.info(
            f"OandaBrokerage initialized: "
            f"practice={self.config.practice}, "
            f"leverage={self.config.default_leverage}:1"
        )

    # ========== Connection ==========

    def connect(self) -> bool:
        """
        Connect to OANDA API.

        Returns:
            True if connected successfully
        """
        if not self.config.api_key:
            logger.error("OANDA_API_KEY not configured")
            self._emit_message("Connection failed: No API key")
            return False

        if not self.config.account_id:
            logger.error("OANDA_ACCOUNT_ID not configured")
            self._emit_message("Connection failed: No account ID")
            return False

        try:
            import oandapyV20
            from oandapyV20 import API

            # Create API client
            self._api = API(
                access_token=self.config.api_key,
                environment="practice" if self.config.practice else "live"
            )

            # Test connection by fetching account
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(self.config.account_id)
            self._api.request(request)

            self._is_connected = True
            env_name = "Practice" if self.config.practice else "Live"
            self._emit_message(f"Connected to OANDA {env_name}")
            logger.info(f"Connected to OANDA {env_name} account")

            return True

        except ImportError:
            logger.error("oandapyV20 package not installed. Run: pip install oandapyV20")
            self._emit_message("Connection failed: oandapyV20 not installed")
            return False

        except Exception as e:
            logger.error(f"Failed to connect to OANDA: {e}")
            self._emit_message(f"Connection failed: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from OANDA API."""
        self._stop_stream.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=5)

        self._is_connected = False
        self._api = None
        self._emit_message("Disconnected from OANDA")
        logger.info("Disconnected from OANDA")

    # ========== Orders ==========

    def place_order(self, order: Order) -> OrderTicket:
        """
        Submit order to OANDA.

        Args:
            order: Order to submit

        Returns:
            OrderTicket for tracking/modifying
        """
        if not self._is_connected:
            order.status = OrderStatus.REJECTED
            return OrderTicket(order=order, _brokerage=self)

        try:
            # Convert symbol to OANDA format
            oanda_symbol = to_oanda_format(order.symbol)

            # Build order data
            order_data = self._build_order_data(order, oanda_symbol)

            # Submit to OANDA
            from oandapyV20.endpoints.orders import OrderCreate
            request = OrderCreate(self.config.account_id, data=order_data)
            response = self._api.request(request)

            # Parse response
            if "orderFillTransaction" in response:
                # Market order filled immediately
                fill = response["orderFillTransaction"]
                order.broker_id = fill.get("id")
                order.status = OrderStatus.FILLED
                order.filled_quantity = float(fill.get("units", 0))
                order.average_fill_price = float(fill.get("price", 0))
                order.filled_time = datetime.utcnow()

                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.FILLED,
                    symbol=order.symbol,
                    quantity=order.filled_quantity,
                    fill_price=order.average_fill_price
                ))

            elif "orderCreateTransaction" in response:
                # Pending order created
                create = response["orderCreateTransaction"]
                order.broker_id = create.get("id")
                order.status = OrderStatus.SUBMITTED
                order.submitted_time = datetime.utcnow()

                self._emit_order_event(OrderEvent(
                    order_id=order.id,
                    event_type=OrderEventType.SUBMITTED,
                    symbol=order.symbol,
                    quantity=order.quantity
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
                symbol=order.symbol,
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

        try:
            # Build replacement order data
            replace_data = {"order": {}}

            if order.order_type == OrderType.LIMIT:
                replace_data["order"]["type"] = "LIMIT"
                replace_data["order"]["price"] = str(limit_price or order.limit_price)
            elif order.order_type == OrderType.STOP_MARKET:
                replace_data["order"]["type"] = "STOP"
                replace_data["order"]["price"] = str(stop_price or order.stop_price)

            replace_data["order"]["units"] = str(int(quantity or order.quantity))
            replace_data["order"]["instrument"] = to_oanda_format(order.symbol)
            replace_data["order"]["timeInForce"] = self._map_tif(order.time_in_force)

            from oandapyV20.endpoints.orders import OrderReplace
            request = OrderReplace(
                self.config.account_id,
                orderID=order.broker_id,
                data=replace_data
            )
            self._api.request(request)

            # Update local order
            if quantity:
                order.quantity = quantity
            if limit_price:
                order.limit_price = limit_price
            if stop_price:
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
            from oandapyV20.endpoints.orders import OrderCancel
            request = OrderCancel(self.config.account_id, orderID=order.broker_id)
            self._api.request(request)

            order.status = OrderStatus.CANCELED
            self._emit_order_event(OrderEvent(
                order_id=order.id,
                event_type=OrderEventType.CANCELED,
                symbol=order.symbol
            ))

            logger.info(f"Order canceled: {order}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False

    def _build_order_data(self, order: Order, oanda_symbol: str) -> dict:
        """Build OANDA order request data."""
        # Determine units (positive for buy, negative for sell)
        units = int(order.quantity)
        if order.side == OrderSide.SELL:
            units = -units

        order_data = {
            "order": {
                "instrument": oanda_symbol,
                "units": str(units),
                "timeInForce": self._map_tif(order.time_in_force),
            }
        }

        # Set order type specific fields
        if order.order_type == OrderType.MARKET:
            order_data["order"]["type"] = "MARKET"

        elif order.order_type == OrderType.LIMIT:
            order_data["order"]["type"] = "LIMIT"
            order_data["order"]["price"] = str(order.limit_price)

        elif order.order_type == OrderType.STOP_MARKET:
            order_data["order"]["type"] = "STOP"
            order_data["order"]["price"] = str(order.stop_price)

        elif order.order_type == OrderType.STOP_LIMIT:
            order_data["order"]["type"] = "STOP"
            order_data["order"]["price"] = str(order.stop_price)
            order_data["order"]["priceBound"] = str(order.limit_price)

        elif order.order_type == OrderType.TRAILING_STOP:
            order_data["order"]["type"] = "TRAILING_STOP_LOSS"
            order_data["order"]["distance"] = str(order.trailing_amount)

        # Add stop loss if specified
        if order.stop_loss:
            order_data["order"]["stopLossOnFill"] = {
                "price": str(order.stop_loss)
            }

        # Add take profit if specified
        if order.take_profit:
            order_data["order"]["takeProfitOnFill"] = {
                "price": str(order.take_profit)
            }

        return order_data

    def _map_tif(self, tif: TimeInForce) -> str:
        """Map TimeInForce to OANDA format."""
        mapping = {
            TimeInForce.GTC: "GTC",
            TimeInForce.DAY: "GFD",
            TimeInForce.IOC: "IOC",
            TimeInForce.FOK: "FOK",
        }
        return mapping.get(tif, "GTC")

    # ========== Account ==========

    def get_cash_balance(self) -> List[CashBalance]:
        """
        Get cash balances.

        Returns:
            List of cash balances (OANDA accounts are single currency)
        """
        if not self._is_connected:
            return []

        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(self.config.account_id)
            response = self._api.request(request)

            account = response.get("account", {})
            currency = account.get("currency", "USD")
            balance = float(account.get("balance", 0))
            margin_available = float(account.get("marginAvailable", 0))

            return [CashBalance(
                currency=currency,
                amount=balance,
                available=margin_available
            )]

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
            from oandapyV20.endpoints.positions import OpenPositions
            request = OpenPositions(self.config.account_id)
            response = self._api.request(request)

            positions = []
            for pos in response.get("positions", []):
                # OANDA separates long and short positions
                long_units = float(pos.get("long", {}).get("units", 0))
                short_units = float(pos.get("short", {}).get("units", 0))

                # Combine into net position
                net_units = long_units + short_units  # short is negative

                if net_units == 0:
                    continue

                # Get average price and P&L
                if net_units > 0:
                    avg_price = float(pos.get("long", {}).get("averagePrice", 0))
                    unrealized = float(pos.get("long", {}).get("unrealizedPL", 0))
                    side = "long"
                else:
                    avg_price = float(pos.get("short", {}).get("averagePrice", 0))
                    unrealized = float(pos.get("short", {}).get("unrealizedPL", 0))
                    side = "short"

                # Convert symbol from OANDA format
                symbol = from_oanda_format(pos.get("instrument", ""))

                # Calculate market value
                market_value = abs(net_units) * avg_price

                # Calculate unrealized P&L percent
                cost_basis = abs(net_units) * avg_price
                unrealized_pct = (unrealized / cost_basis * 100) if cost_basis > 0 else 0

                positions.append(Position(
                    symbol=symbol,
                    quantity=abs(net_units),
                    average_price=avg_price,
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
        Get total account value (NAV).

        Returns:
            Total account value
        """
        if not self._is_connected:
            return 0.0

        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(self.config.account_id)
            response = self._api.request(request)

            return float(response.get("account", {}).get("NAV", 0))

        except Exception as e:
            logger.error(f"Failed to get account value: {e}")
            return 0.0

    def get_margin_used(self) -> float:
        """Get total margin in use."""
        if not self._is_connected:
            return 0.0

        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(self.config.account_id)
            response = self._api.request(request)

            return float(response.get("account", {}).get("marginUsed", 0))

        except Exception as e:
            logger.error(f"Failed to get margin used: {e}")
            return 0.0

    def get_margin_available(self) -> float:
        """Get available margin."""
        if not self._is_connected:
            return 0.0

        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(self.config.account_id)
            response = self._api.request(request)

            return float(response.get("account", {}).get("marginAvailable", 0))

        except Exception as e:
            logger.error(f"Failed to get margin available: {e}")
            return 0.0

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
            from oandapyV20.endpoints.pricing import PricingInfo
            oanda_symbol = to_oanda_format(symbol)

            request = PricingInfo(
                self.config.account_id,
                params={"instruments": oanda_symbol}
            )
            response = self._api.request(request)

            prices = response.get("prices", [])
            if prices:
                price = prices[0]
                bid = float(price.get("bids", [{}])[0].get("price", 0))
                ask = float(price.get("asks", [{}])[0].get("price", 0))
                mid = (bid + ask) / 2

                return {"bid": bid, "ask": ask, "mid": mid}

            return None

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_candles(
        self,
        symbol: str,
        granularity: str = "H1",
        count: int = 100
    ) -> List[Dict]:
        """
        Get historical candles.

        Args:
            symbol: Currency pair
            granularity: Candle timeframe (S5, M1, M5, M15, M30, H1, H4, D, W, M)
            count: Number of candles

        Returns:
            List of candle dicts with open, high, low, close, volume
        """
        if not self._is_connected:
            return []

        try:
            from oandapyV20.endpoints.instruments import InstrumentsCandles
            oanda_symbol = to_oanda_format(symbol)

            request = InstrumentsCandles(
                instrument=oanda_symbol,
                params={
                    "granularity": granularity,
                    "count": count,
                    "price": "M"  # Mid prices
                }
            )
            response = self._api.request(request)

            candles = []
            for candle in response.get("candles", []):
                if candle.get("complete", False):
                    mid = candle.get("mid", {})
                    candles.append({
                        "time": candle.get("time"),
                        "open": float(mid.get("o", 0)),
                        "high": float(mid.get("h", 0)),
                        "low": float(mid.get("l", 0)),
                        "close": float(mid.get("c", 0)),
                        "volume": int(candle.get("volume", 0))
                    })

            return candles

        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return []

    # ========== Streaming ==========

    def start_price_stream(
        self,
        symbols: List[str],
        callback: Callable[[str, float, float], None]
    ) -> None:
        """
        Start streaming prices for symbols.

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
            target=self._price_stream_worker,
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

    def _price_stream_worker(self, symbols: List[str]) -> None:
        """Background worker for price streaming."""
        try:
            from oandapyV20.endpoints.pricing import PricingStream

            oanda_symbols = ",".join(to_oanda_format(s) for s in symbols)

            request = PricingStream(
                self.config.account_id,
                params={"instruments": oanda_symbols}
            )

            while not self._stop_stream.is_set():
                try:
                    for response in self._api.request(request):
                        if self._stop_stream.is_set():
                            break

                        if response.get("type") == "PRICE":
                            symbol = from_oanda_format(response.get("instrument", ""))
                            bid = float(response.get("bids", [{}])[0].get("price", 0))
                            ask = float(response.get("asks", [{}])[0].get("price", 0))

                            # Cache latest price
                            self._last_prices[symbol] = {
                                "bid": bid,
                                "ask": ask,
                                "mid": (bid + ask) / 2,
                                "time": datetime.utcnow()
                            }

                            # Notify callbacks
                            for callback in self._price_callbacks:
                                try:
                                    callback(symbol, bid, ask)
                                except Exception as e:
                                    logger.error(f"Price callback error: {e}")

                except Exception as e:
                    if not self._stop_stream.is_set():
                        logger.error(f"Stream error: {e}")
                        time.sleep(5)  # Reconnect delay

        except Exception as e:
            logger.error(f"Price stream failed: {e}")

    # ========== Utility Methods ==========

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
            from oandapyV20.endpoints.positions import PositionClose
            oanda_symbol = to_oanda_format(symbol)

            # Get current position
            position = self.get_position(symbol)
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False

            close_data = {}
            if percentage >= 100:
                close_data["longUnits" if position.side == "long" else "shortUnits"] = "ALL"
            else:
                units = int(position.quantity * percentage / 100)
                close_data["longUnits" if position.side == "long" else "shortUnits"] = str(units)

            request = PositionClose(
                self.config.account_id,
                instrument=oanda_symbol,
                data=close_data
            )
            self._api.request(request)

            logger.info(f"Closed {percentage}% of {symbol} position")
            return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive account summary.

        Returns:
            Dict with account details
        """
        if not self._is_connected:
            return {}

        try:
            from oandapyV20.endpoints.accounts import AccountSummary
            request = AccountSummary(self.config.account_id)
            response = self._api.request(request)

            account = response.get("account", {})
            return {
                "id": account.get("id"),
                "currency": account.get("currency"),
                "balance": float(account.get("balance", 0)),
                "nav": float(account.get("NAV", 0)),
                "unrealized_pnl": float(account.get("unrealizedPL", 0)),
                "margin_used": float(account.get("marginUsed", 0)),
                "margin_available": float(account.get("marginAvailable", 0)),
                "position_count": int(account.get("openPositionCount", 0)),
                "pending_order_count": int(account.get("pendingOrderCount", 0)),
                "last_transaction_id": account.get("lastTransactionID"),
            }

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}

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
            from oandapyV20.endpoints.trades import TradesList
            request = TradesList(
                self.config.account_id,
                params={"state": "ALL", "count": count}
            )
            response = self._api.request(request)

            trades = []
            for trade in response.get("trades", []):
                trades.append({
                    "id": trade.get("id"),
                    "symbol": from_oanda_format(trade.get("instrument", "")),
                    "units": float(trade.get("currentUnits", 0)),
                    "price": float(trade.get("price", 0)),
                    "unrealized_pnl": float(trade.get("unrealizedPL", 0)),
                    "realized_pnl": float(trade.get("realizedPL", 0)),
                    "state": trade.get("state"),
                    "open_time": trade.get("openTime"),
                })

            return trades

        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
