"""
Paper Trading Simulator
=======================
Full trading simulator with virtual portfolio and order execution.
Practice trading without risking real money.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_time: Optional[datetime] = None
    created_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """Open position."""
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    unrealized_pnl: float = 0
    unrealized_pnl_pct: float = 0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PaperTradingSimulator:
    """
    Full paper trading simulator.

    Features:
    - Virtual portfolio management
    - Order execution simulation
    - Position tracking
    - P&L calculation
    - Risk management
    - Trade history
    """

    def __init__(self, initial_capital: float = 10000, commission: float = 0.001):
        """
        Initialize paper trading simulator.

        Args:
            initial_capital: Starting virtual capital
            commission: Trading commission (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission

        # Trading state
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trade_history: List[Dict] = []
        self._order_counter = 0
        self._lock = threading.Lock()

        logger.info(f"Paper trading initialized with ${initial_capital:,.2f}")

    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value (cash + positions).

        Returns:
            Total portfolio value
        """
        with self._lock:
            positions_value = sum(
                pos.quantity * pos.current_price
                for pos in self.positions.values()
            )
            return self.cash + positions_value

    def get_portfolio_stats(self) -> Dict:
        """
        Get comprehensive portfolio statistics.

        Returns:
            Dict with portfolio metrics
        """
        with self._lock:
            total_value = self.get_portfolio_value()
            total_pnl = total_value - self.initial_capital
            total_return = (total_pnl / self.initial_capital) * 100

            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            realized_pnl = sum(trade['pnl'] for trade in self.trade_history)

            winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
            total_trades = len(self.trade_history)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            return {
                'total_value': total_value,
                'cash': self.cash,
                'positions_value': total_value - self.cash,
                'total_pnl': total_pnl,
                'total_return_pct': total_return,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'num_positions': len(self.positions),
                'total_trades': total_trades,
                'win_rate': win_rate,
                'initial_capital': self.initial_capital
            }

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Order:
        """
        Place a trading order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            quantity: Order quantity
            order_type: Order type
            price: Limit price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order object
        """
        with self._lock:
            self._order_counter += 1
            order_id = f"PAPER_{self._order_counter:06d}"

            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=None
            )

            self.orders.append(order)

            logger.info(f"Order placed: {order_id} {side.value} {quantity} {symbol} @ {order_type.value}")

            return order

    def execute_market_order(
        self,
        order: Order,
        current_price: float
    ) -> bool:
        """
        Execute a market order immediately.

        Args:
            order: Order to execute
            current_price: Current market price

        Returns:
            True if executed successfully
        """
        with self._lock:
            # Calculate required capital
            trade_value = order.quantity * current_price
            commission_cost = trade_value * self.commission

            if order.side == OrderSide.BUY:
                total_cost = trade_value + commission_cost

                # Check if enough cash
                if total_cost > self.cash:
                    logger.warning(f"Insufficient cash for order {order.order_id}: need ${total_cost:,.2f}, have ${self.cash:,.2f}")
                    order.status = OrderStatus.REJECTED
                    return False

                # Execute buy
                self.cash -= total_cost

                # Create or update position
                if order.symbol in self.positions:
                    # Average up position
                    pos = self.positions[order.symbol]
                    total_quantity = pos.quantity + order.quantity
                    avg_price = ((pos.entry_price * pos.quantity) + (current_price * order.quantity)) / total_quantity
                    pos.quantity = total_quantity
                    pos.entry_price = avg_price
                else:
                    # New position
                    self.positions[order.symbol] = Position(
                        symbol=order.symbol,
                        side=OrderSide.BUY,
                        quantity=order.quantity,
                        entry_price=current_price,
                        entry_time=datetime.utcnow(),
                        current_price=current_price
                    )

            else:  # SELL
                # Check if position exists
                if order.symbol not in self.positions:
                    logger.warning(f"No position to sell for {order.symbol}")
                    order.status = OrderStatus.REJECTED
                    return False

                pos = self.positions[order.symbol]

                # Check if enough quantity
                if order.quantity > pos.quantity:
                    logger.warning(f"Insufficient quantity: have {pos.quantity}, trying to sell {order.quantity}")
                    order.status = OrderStatus.REJECTED
                    return False

                # Execute sell
                sale_proceeds = trade_value - commission_cost
                self.cash += sale_proceeds

                # Calculate P&L for this trade
                entry_value = order.quantity * pos.entry_price
                pnl = sale_proceeds - entry_value
                pnl_pct = (pnl / entry_value) * 100

                # Record trade
                self.trade_history.append({
                    'symbol': order.symbol,
                    'side': order.side.value,
                    'quantity': order.quantity,
                    'entry_price': pos.entry_price,
                    'exit_price': current_price,
                    'entry_time': pos.entry_time,
                    'exit_time': datetime.utcnow(),
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'commission': commission_cost * 2  # Entry + exit
                })

                # Update or close position
                if order.quantity == pos.quantity:
                    # Close entire position
                    del self.positions[order.symbol]
                else:
                    # Partial close
                    pos.quantity -= order.quantity

            # Mark order as filled
            order.status = OrderStatus.FILLED
            order.filled_price = current_price
            order.filled_time = datetime.utcnow()

            logger.info(f"Order executed: {order.order_id} @ ${current_price:,.2f}")

            return True

    def update_positions(self, prices: Dict[str, float]) -> None:
        """
        Update all positions with current prices.

        Args:
            prices: Dict of {symbol: current_price}
        """
        with self._lock:
            for symbol, pos in self.positions.items():
                if symbol in prices:
                    pos.current_price = prices[symbol]

                    # Calculate unrealized P&L
                    if pos.side == OrderSide.BUY:
                        pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
                    else:
                        pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity

                    pos.unrealized_pnl_pct = (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100

                    # Check stop loss / take profit
                    if pos.stop_loss and pos.current_price <= pos.stop_loss:
                        logger.info(f"Stop loss hit for {symbol} @ ${pos.current_price:,.2f}")
                        self._close_position(symbol, pos.current_price, "STOP_LOSS")

                    if pos.take_profit and pos.current_price >= pos.take_profit:
                        logger.info(f"Take profit hit for {symbol} @ ${pos.current_price:,.2f}")
                        self._close_position(symbol, pos.current_price, "TAKE_PROFIT")

    def _close_position(self, symbol: str, price: float, reason: str) -> None:
        """Close a position (internal use)."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Create sell order
        sell_order = Order(
            order_id=f"AUTO_{self._order_counter:06d}",
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.quantity
        )

        self.orders.append(sell_order)
        self.execute_market_order(sell_order, price)

    def reset(self) -> None:
        """Reset simulator to initial state."""
        with self._lock:
            self.cash = self.initial_capital
            self.positions.clear()
            self.orders.clear()
            self.trade_history.clear()
            self._order_counter = 0

            logger.info("Paper trading simulator reset")

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get recent trade history."""
        with self._lock:
            return self.trade_history[-limit:]

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        with self._lock:
            return list(self.positions.values())

    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders."""
        with self._lock:
            return [o for o in self.orders if o.status == OrderStatus.PENDING]
