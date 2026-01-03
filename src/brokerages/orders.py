"""
Order Types and Management (Lean-Inspired)
==========================================
Comprehensive order system supporting all major order types.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from .base import BaseBrokerage


class OrderType(Enum):
    """Order types supported (Lean-compatible)."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MARKET_ON_OPEN = "market_on_open"
    MARKET_ON_CLOSE = "market_on_close"


class OrderSide(Enum):
    """Order direction."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order lifecycle states (Lean-compatible)."""
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partial"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Order duration (Lean-compatible)."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


@dataclass
class Order:
    """
    Trading order (Lean-inspired).

    Supports all major order types with proper lifecycle management.

    Examples:
        # Market order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )

        # Limit order
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10,
            limit_price=150.00
        )

        # Stop loss order
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_MARKET,
            quantity=0.1,
            stop_price=45000.00
        )
    """
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    broker_id: Optional[str] = None

    # Core properties
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0

    # Price levels
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_amount: Optional[float] = None
    trailing_as_percent: bool = False

    # Execution
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.NEW

    # Fill info
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0

    # Timestamps
    created_time: datetime = field(default_factory=datetime.utcnow)
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None

    # Risk management (attached SL/TP)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Tag for tracking
    tag: str = ""

    @property
    def is_open(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.NEW,
            OrderStatus.SUBMITTED,
            OrderStatus.PARTIALLY_FILLED
        ]

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percent(self) -> float:
        """Get fill percentage."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def value(self) -> float:
        """Get order value (quantity * price)."""
        price = self.average_fill_price or self.limit_price or 0
        return self.quantity * price

    def __repr__(self) -> str:
        return (
            f"Order({self.id}: {self.side.value} {self.quantity} {self.symbol} "
            f"@ {self.order_type.value} [{self.status.value}])"
        )


@dataclass
class OrderTicket:
    """
    Order management handle (Lean-inspired).

    Allows updating/canceling orders after submission.

    Example:
        ticket = brokerage.place_order(order)

        # Check status
        if ticket.status == OrderStatus.FILLED:
            print(f"Filled at {ticket.order.average_fill_price}")

        # Update order
        ticket.update(limit_price=155.00)

        # Cancel order
        ticket.cancel()
    """
    order: Order
    _brokerage: Optional['BaseBrokerage'] = field(default=None, repr=False)

    def update(
        self,
        quantity: Optional[float] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> bool:
        """
        Update order parameters.

        Args:
            quantity: New quantity
            limit_price: New limit price
            stop_price: New stop price

        Returns:
            True if update accepted
        """
        if not self.order.is_open:
            return False
        if self._brokerage is None:
            return False
        return self._brokerage.update_order(
            self.order, quantity, limit_price, stop_price
        )

    def cancel(self) -> bool:
        """
        Cancel the order.

        Returns:
            True if cancellation accepted
        """
        if not self.order.is_open:
            return False
        if self._brokerage is None:
            return False
        return self._brokerage.cancel_order(self.order)

    @property
    def status(self) -> OrderStatus:
        """Get current order status."""
        return self.order.status

    @property
    def is_filled(self) -> bool:
        """Check if order is filled."""
        return self.order.is_filled

    @property
    def fill_price(self) -> float:
        """Get average fill price."""
        return self.order.average_fill_price
