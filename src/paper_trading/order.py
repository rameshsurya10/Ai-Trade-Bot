"""
Order Types and Management
==========================
Order definitions for paper trading.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
import uuid


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Trading order.

    Represents a single order in the paper trading system.
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET

    # Pricing
    price: Optional[float] = None  # Limit price
    stop_price: Optional[float] = None  # Stop trigger price

    # Order management
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Execution details
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Metadata
    signal_id: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        """Validate order."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")

        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")

        if self.order_type in (OrderType.STOP_LOSS, OrderType.STOP_LIMIT):
            if self.stop_price is None:
                raise ValueError("Stop orders require a stop_price")

    @property
    def is_active(self) -> bool:
        """Check if order is active."""
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity

    @property
    def total_cost(self) -> float:
        """Get total cost including commission."""
        return (self.filled_quantity * self.filled_price) + self.commission

    def fill(self, price: float, quantity: Optional[float] = None, commission: float = 0.0):
        """
        Fill order (fully or partially).

        Args:
            price: Execution price
            quantity: Quantity to fill (None = remaining)
            commission: Commission charged
        """
        quantity = quantity or self.remaining_quantity

        if quantity > self.remaining_quantity:
            quantity = self.remaining_quantity

        # Update filled amounts
        self.filled_quantity += quantity
        self.filled_price = price
        self.commission += commission
        self.slippage = abs(price - (self.price or price))

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self):
        """Cancel order."""
        if not self.is_active:
            return False

        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()
        return True

    def reject(self, reason: str = ""):
        """Reject order."""
        self.status = OrderStatus.REJECTED
        self.notes = reason

    def should_trigger(self, current_price: float) -> bool:
        """
        Check if order should trigger at current price.

        Args:
            current_price: Current market price

        Returns:
            True if order should trigger
        """
        if self.order_type == OrderType.MARKET:
            return True

        if self.order_type == OrderType.LIMIT:
            if self.side == OrderSide.BUY:
                return current_price <= self.price
            else:
                return current_price >= self.price

        if self.order_type == OrderType.STOP_LOSS:
            if self.side == OrderSide.BUY:
                return current_price >= self.stop_price
            else:
                return current_price <= self.stop_price

        if self.order_type == OrderType.TAKE_PROFIT:
            if self.side == OrderSide.BUY:
                return current_price <= self.stop_price
            else:
                return current_price >= self.stop_price

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'slippage': self.slippage,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
        }

    @classmethod
    def from_signal(
        cls,
        symbol: str,
        signal: str,
        price: float,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        signal_id: Optional[str] = None,
    ) -> 'Order':
        """
        Create order from trading signal.

        Args:
            symbol: Trading symbol
            signal: Signal type (BUY, SELL)
            price: Current price
            quantity: Order quantity
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_id: Optional signal ID

        Returns:
            Order instance
        """
        side = OrderSide.BUY if 'BUY' in signal.upper() else OrderSide.SELL

        return cls(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_id=signal_id,
        )
