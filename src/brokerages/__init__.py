"""
Brokerages Module (Lean-Inspired)
=================================
Unified brokerage abstraction for live and paper trading.

Supports:
- Multiple order types (Market, Limit, Stop, Trailing)
- Multiple brokers (Alpaca, Binance, Paper)
- Event-driven order management
- Position and portfolio tracking

Usage:
    from src.brokerages import AlpacaBrokerage, Order, OrderSide, OrderType

    brokerage = AlpacaBrokerage(paper=True)
    brokerage.connect()

    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10
    )
    ticket = brokerage.place_order(order)
"""

from .orders import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    OrderTicket
)
from .events import OrderEvent, OrderEventType
from .base import BaseBrokerage, CashBalance, Position

# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "AlpacaBrokerage":
        from .alpaca import AlpacaBrokerage
        return AlpacaBrokerage
    elif name == "BinanceBrokerage":
        from .binance import BinanceBrokerage
        return BinanceBrokerage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Orders
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'TimeInForce',
    'OrderTicket',
    # Events
    'OrderEvent',
    'OrderEventType',
    # Base
    'BaseBrokerage',
    'CashBalance',
    'Position',
    # Implementations (lazy loaded)
    'AlpacaBrokerage',
    'BinanceBrokerage',
]
