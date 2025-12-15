"""
Paper Trading Module
====================
Simulated trading for strategy testing without real money.

Features:
- Virtual portfolio management
- Realistic order execution with slippage
- Position tracking and P&L calculation
- Trade history and performance metrics
"""

from .engine import PaperTradingEngine, PaperAccount, Position
from .order import Order, OrderType, OrderSide, OrderStatus

__all__ = [
    'PaperTradingEngine',
    'PaperAccount',
    'Position',
    'Order',
    'OrderType',
    'OrderSide',
    'OrderStatus',
]
