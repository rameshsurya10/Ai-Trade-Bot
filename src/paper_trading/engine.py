"""
Paper Trading Engine
====================
Virtual trading environment for testing strategies.

Features:
- Virtual account management
- Position tracking
- Realistic order execution
- P&L and performance metrics
"""

import logging
import threading
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from enum import Enum

import yaml

from .order import Order, OrderType, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """
    Open trading position.

    Tracks an open position with entry details and P&L.
    """
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    entry_time: datetime = field(default_factory=datetime.utcnow)

    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # P&L tracking
    current_price: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0

    # Metadata
    order_id: str = ""
    signal_id: str = ""

    def __post_init__(self):
        self.current_price = self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.side == OrderSide.BUY:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        cost = self.entry_price * self.quantity
        if cost == 0:
            return 0.0
        return (self.unrealized_pnl / cost) * 100

    @property
    def market_value(self) -> float:
        """Get current market value."""
        return self.current_price * self.quantity

    def update_price(self, price: float):
        """Update current price."""
        self.current_price = price

    def should_stop_loss(self) -> bool:
        """Check if stop loss is triggered."""
        if self.stop_loss is None:
            return False

        if self.side == OrderSide.BUY:
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        """Check if take profit is triggered."""
        if self.take_profit is None:
            return False

        if self.side == OrderSide.BUY:
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'market_value': self.market_value,
        }


@dataclass
class PaperAccount:
    """
    Virtual trading account.

    Tracks balance, positions, and trading history.
    """
    initial_balance: float = 10000.0
    currency: str = "USDT"

    # Balance tracking
    cash_balance: float = field(default=0.0)
    reserved_balance: float = 0.0  # For open orders

    # Positions
    positions: Dict[str, Position] = field(default_factory=dict)

    # Orders
    open_orders: Dict[str, Order] = field(default_factory=dict)
    order_history: List[Order] = field(default_factory=list)

    # Trade history
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    # Statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_commission: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.cash_balance == 0.0:
            self.cash_balance = self.initial_balance

    @property
    def total_equity(self) -> float:
        """Calculate total equity (cash + positions value)."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash_balance + positions_value

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L across all positions."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L."""
        return sum(t.get('pnl', 0) for t in self.trade_history)

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        """Total return percentage."""
        return ((self.total_equity - self.initial_balance) / self.initial_balance) * 100

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_balance': self.initial_balance,
            'cash_balance': self.cash_balance,
            'total_equity': self.total_equity,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_return_pct': self.total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'positions': {s: p.to_dict() for s, p in self.positions.items()},
            'open_orders': len(self.open_orders),
        }


class PaperTradingEngine:
    """
    Paper trading engine for simulated trading.

    Features:
    - Account management
    - Order execution with slippage
    - Position management
    - Risk management (stop loss / take profit)
    - Performance tracking

    Usage:
        engine = PaperTradingEngine(initial_balance=10000)
        engine.on_trade(lambda t: print(f"Trade: {t}"))

        # Place order from signal
        engine.execute_signal(
            symbol="BTC/USDT",
            signal="BUY",
            price=50000,
            quantity=0.1,
            stop_loss=49000,
            take_profit=52000
        )

        # Update prices
        engine.update_price("BTC/USDT", 51000)
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,  # 0.1%
        slippage_rate: float = 0.0005,  # 0.05%
        max_position_size: float = 0.25,  # 25% of equity
        db_path: Optional[str] = None,
    ):
        """
        Initialize paper trading engine.

        Args:
            initial_balance: Starting balance
            commission_rate: Commission per trade (decimal)
            slippage_rate: Expected slippage (decimal)
            max_position_size: Max position as fraction of equity
            db_path: Optional database path for persistence
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.max_position_size = max_position_size

        # Account
        self.account = PaperAccount(initial_balance=initial_balance)

        # Database
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_database()

        # Callbacks
        self._trade_callbacks: List[Callable[[Dict], None]] = []
        self._order_callbacks: List[Callable[[Order], None]] = []

        # Thread safety
        self._lock = threading.Lock()

        # Current prices
        self._prices: Dict[str, float] = {}

        logger.info(f"PaperTradingEngine initialized: ${initial_balance:.2f}")

    @classmethod
    def from_config(cls, config_path: str) -> 'PaperTradingEngine':
        """Create engine from config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        paper_config = config.get('paper_trading', {})

        return cls(
            initial_balance=paper_config.get('initial_balance', 10000.0),
            commission_rate=paper_config.get('commission_rate', 0.001),
            slippage_rate=paper_config.get('slippage_rate', 0.0005),
            max_position_size=paper_config.get('max_position_size', 0.25),
            db_path=config.get('database', {}).get('path'),
        )

    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                commission REAL,
                pnl REAL,
                order_id TEXT,
                notes TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_account_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                equity REAL,
                cash REAL,
                positions_value REAL,
                unrealized_pnl REAL,
                realized_pnl REAL
            )
        ''')

        conn.commit()
        conn.close()

    def on_trade(self, callback: Callable[[Dict], None]):
        """Register trade callback."""
        self._trade_callbacks.append(callback)

    def on_order(self, callback: Callable[[Order], None]):
        """Register order callback."""
        self._order_callbacks.append(callback)

    def _notify_trade(self, trade: Dict):
        """Notify trade callbacks."""
        for callback in self._trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    def _notify_order(self, order: Order):
        """Notify order callbacks."""
        for callback in self._order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    def _calculate_slippage(self, price: float, side: OrderSide) -> float:
        """Calculate slippage-adjusted price."""
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for trade."""
        return quantity * price * self.commission_rate

    def update_price(self, symbol: str, price: float):
        """
        Update current price for symbol.

        Also checks for stop loss / take profit triggers.
        """
        with self._lock:
            self._prices[symbol] = price

            # Update position prices
            if symbol in self.account.positions:
                position = self.account.positions[symbol]
                position.update_price(price)

                # Check stop loss
                if position.should_stop_loss():
                    logger.info(f"Stop loss triggered for {symbol}")
                    self._close_position(symbol, price, "Stop loss triggered")

                # Check take profit
                elif position.should_take_profit():
                    logger.info(f"Take profit triggered for {symbol}")
                    self._close_position(symbol, price, "Take profit triggered")

            # Check pending orders
            for order_id, order in list(self.account.open_orders.items()):
                if order.symbol == symbol and order.should_trigger(price):
                    self._execute_order(order, price)

            self.account.last_updated = datetime.utcnow()

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol."""
        return self._prices.get(symbol)

    def place_order(self, order: Order) -> bool:
        """
        Place a new order.

        Args:
            order: Order to place

        Returns:
            True if order placed successfully
        """
        with self._lock:
            # Validate order
            if order.side == OrderSide.BUY:
                required_margin = order.quantity * (order.price or self._prices.get(order.symbol, 0))
                required_margin *= (1 + self.commission_rate + self.slippage_rate)

                if required_margin > self.account.cash_balance:
                    order.reject("Insufficient balance")
                    logger.warning(f"Order rejected: insufficient balance")
                    self._notify_order(order)
                    return False

                # Check position size limit
                if required_margin > self.account.total_equity * self.max_position_size:
                    order.reject("Exceeds max position size")
                    logger.warning(f"Order rejected: exceeds position limit")
                    self._notify_order(order)
                    return False

            else:  # SELL
                # Must have position to sell (no shorting in simple mode)
                if order.symbol not in self.account.positions:
                    order.reject("No position to sell")
                    logger.warning(f"Order rejected: no position")
                    self._notify_order(order)
                    return False

            # Add to open orders
            order.status = OrderStatus.OPEN
            self.account.open_orders[order.order_id] = order

            logger.info(f"Order placed: {order.side.value} {order.quantity} {order.symbol}")
            self._notify_order(order)

            # Execute market orders immediately
            if order.order_type == OrderType.MARKET:
                price = self._prices.get(order.symbol, order.price)
                if price:
                    self._execute_order(order, price)

            return True

    def _execute_order(self, order: Order, price: float):
        """Execute an order at given price."""
        execution_price = self._calculate_slippage(price, order.side)
        commission = self._calculate_commission(order.quantity, execution_price)

        order.fill(execution_price, commission=commission)
        self.account.total_commission += commission

        # Remove from open orders
        if order.order_id in self.account.open_orders:
            del self.account.open_orders[order.order_id]

        # Update positions
        if order.side == OrderSide.BUY:
            # Open or add to position
            if order.symbol in self.account.positions:
                # Average into existing position
                pos = self.account.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                avg_price = ((pos.quantity * pos.entry_price) +
                            (order.quantity * execution_price)) / total_quantity
                pos.quantity = total_quantity
                pos.entry_price = avg_price
                pos.stop_loss = order.stop_loss or pos.stop_loss
                pos.take_profit = order.take_profit or pos.take_profit
            else:
                # New position
                self.account.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    side=OrderSide.BUY,
                    quantity=order.quantity,
                    entry_price=execution_price,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    order_id=order.order_id,
                )

            # Deduct from cash
            self.account.cash_balance -= (order.quantity * execution_price) + commission

        else:  # SELL
            pnl = self._close_position(order.symbol, execution_price, "Order filled")
            order.notes = f"PnL: ${pnl:.2f}"

        # Add to history
        self.account.order_history.append(order)

        logger.info(f"Order executed: {order.side.value} {order.quantity} {order.symbol} @ ${execution_price:.2f}")
        self._notify_order(order)

    def _close_position(self, symbol: str, price: float, reason: str = "") -> float:
        """Close a position at given price."""
        if symbol not in self.account.positions:
            return 0.0

        position = self.account.positions[symbol]
        commission = self._calculate_commission(position.quantity, price)

        # Calculate P&L
        if position.side == OrderSide.BUY:
            pnl = (price - position.entry_price) * position.quantity - commission
        else:
            pnl = (position.entry_price - price) * position.quantity - commission

        # Update account
        self.account.cash_balance += (position.quantity * price) - commission
        self.account.total_commission += commission
        self.account.total_trades += 1

        if pnl > 0:
            self.account.winning_trades += 1
        else:
            self.account.losing_trades += 1

        # Record trade
        trade = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': 'sell',
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'exit_price': price,
            'pnl': pnl,
            'commission': commission,
            'reason': reason,
        }
        self.account.trade_history.append(trade)
        self._notify_trade(trade)

        # Save to database
        if self.db_path:
            self._save_trade(trade)

        # Remove position
        del self.account.positions[symbol]

        logger.info(f"Position closed: {symbol} PnL: ${pnl:.2f} ({reason})")
        return pnl

    def _save_trade(self, trade: Dict):
        """Save trade to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO paper_trades
            (timestamp, symbol, side, quantity, price, commission, pnl, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade['timestamp'],
            trade['symbol'],
            trade['side'],
            trade['quantity'],
            trade['exit_price'],
            trade['commission'],
            trade['pnl'],
            trade.get('reason', ''),
        ))

        conn.commit()
        conn.close()

    def execute_signal(
        self,
        symbol: str,
        signal: str,
        price: float,
        quantity: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        signal_id: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Execute a trading signal.

        Args:
            symbol: Trading symbol
            signal: Signal type (BUY, SELL, etc.)
            price: Current price
            quantity: Order quantity (auto-calculated if None)
            stop_loss: Stop loss price
            take_profit: Take profit price
            signal_id: Optional signal ID

        Returns:
            Created order or None
        """
        # Ignore neutral signals
        if signal in ('NEUTRAL', 'WAIT', 'WEAK_BUY', 'WEAK_SELL'):
            return None

        # Update price
        self.update_price(symbol, price)

        # Auto-calculate quantity if not provided
        if quantity is None:
            max_allocation = self.account.total_equity * self.max_position_size
            quantity = max_allocation / price

        # Create order
        order = Order.from_signal(
            symbol=symbol,
            signal=signal,
            price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_id=signal_id,
        )

        # Place order
        if self.place_order(order):
            return order

        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        with self._lock:
            if order_id in self.account.open_orders:
                order = self.account.open_orders[order_id]
                if order.cancel():
                    del self.account.open_orders[order_id]
                    self.account.order_history.append(order)
                    logger.info(f"Order cancelled: {order_id}")
                    self._notify_order(order)
                    return True
        return False

    def cancel_all_orders(self):
        """Cancel all open orders."""
        with self._lock:
            for order_id in list(self.account.open_orders.keys()):
                self.cancel_order(order_id)

    def close_all_positions(self):
        """Close all open positions at current prices."""
        with self._lock:
            for symbol in list(self.account.positions.keys()):
                price = self._prices.get(symbol)
                if price:
                    self._close_position(symbol, price, "Manual close")

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary."""
        with self._lock:
            return self.account.to_dict()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.account.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return dict(self.account.positions)

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history."""
        return self.account.trade_history[-limit:]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        trades = self.account.trade_history

        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
            }

        profits = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] < 0]

        avg_profit = sum(profits) / len(profits) if profits else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        total_profit = sum(profits)
        total_loss = abs(sum(losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Calculate drawdown
        equity_curve = [self.account.initial_balance]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])

        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'total_trades': self.account.total_trades,
            'winning_trades': self.account.winning_trades,
            'losing_trades': self.account.losing_trades,
            'win_rate': self.account.win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_pnl': self.account.realized_pnl,
            'total_return': self.account.total_return_pct,
            'total_commission': self.account.total_commission,
        }

    def reset(self):
        """Reset account to initial state."""
        with self._lock:
            self.account = PaperAccount(initial_balance=self.account.initial_balance)
            self._prices.clear()
            logger.info("Paper trading account reset")


# CLI usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    engine = PaperTradingEngine(initial_balance=10000)

    def on_trade(trade):
        print(f"Trade: {trade['symbol']} PnL: ${trade['pnl']:.2f}")

    engine.on_trade(on_trade)

    # Simulate some trades
    engine.execute_signal(
        symbol="BTC/USDT",
        signal="BUY",
        price=50000,
        stop_loss=48000,
        take_profit=55000,
    )

    print(f"After BUY: {engine.get_account_summary()}")

    # Price goes up
    engine.update_price("BTC/USDT", 54000)
    print(f"After price increase: {engine.get_account_summary()}")

    # Take profit triggers
    engine.update_price("BTC/USDT", 55000)
    print(f"After take profit: {engine.get_account_summary()}")

    # Performance
    print(f"\nPerformance: {engine.get_performance_metrics()}")
