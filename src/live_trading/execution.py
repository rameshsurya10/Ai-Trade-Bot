"""
Execution Engine (Lean-Inspired)
================================
Order execution with smart routing and fills tracking.

Features:
- Order queue management
- Fill tracking and reconciliation
- Execution reporting
- Slippage tracking
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING
from queue import Queue, Empty

if TYPE_CHECKING:
    from src.brokerages.base import BaseBrokerage
    from src.brokerages.orders import Order

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """
    Report of an order execution (Lean-style).

    Tracks all details of how an order was filled.
    """
    order_id: str
    symbol: str
    side: str               # BUY or SELL
    quantity: float
    filled_quantity: float
    average_price: float
    commission: float

    # Timing
    submitted_time: datetime
    filled_time: Optional[datetime] = None

    # Analysis
    expected_price: float = 0.0
    slippage: float = 0.0
    slippage_percent: float = 0.0

    # Status
    is_complete: bool = False
    is_partial: bool = False
    error_message: str = ""

    @property
    def fill_value(self) -> float:
        """Total value of filled quantity."""
        return self.filled_quantity * self.average_price

    @property
    def total_cost(self) -> float:
        """Total cost including commission."""
        return self.fill_value + self.commission

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled."""
        if self.quantity == 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100

    @property
    def latency_ms(self) -> Optional[float]:
        """Execution latency in milliseconds."""
        if not self.filled_time:
            return None
        return (self.filled_time - self.submitted_time).total_seconds() * 1000

    def calculate_slippage(self, expected_price: float):
        """Calculate slippage from expected price."""
        self.expected_price = expected_price
        if expected_price > 0 and self.average_price > 0:
            self.slippage = self.average_price - expected_price
            self.slippage_percent = (self.slippage / expected_price) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'fill_rate': f"{self.fill_rate:.1f}%",
            'average_price': self.average_price,
            'expected_price': self.expected_price,
            'slippage': self.slippage,
            'slippage_percent': f"{self.slippage_percent:.3f}%",
            'commission': self.commission,
            'total_cost': self.total_cost,
            'latency_ms': self.latency_ms,
            'is_complete': self.is_complete,
        }


@dataclass
class ExecutionStats:
    """Aggregate execution statistics."""
    total_orders: int = 0
    filled_orders: int = 0
    partial_fills: int = 0
    rejected_orders: int = 0

    total_commission: float = 0.0
    total_slippage: float = 0.0
    avg_slippage_percent: float = 0.0
    avg_latency_ms: float = 0.0

    # Lists for detailed analysis
    _slippages: List[float] = field(default_factory=list)
    _latencies: List[float] = field(default_factory=list)

    def add_execution(self, report: ExecutionReport):
        """Add execution to statistics."""
        self.total_orders += 1
        self.total_commission += report.commission
        self.total_slippage += abs(report.slippage)

        if report.is_complete:
            self.filled_orders += 1
        elif report.is_partial:
            self.partial_fills += 1
        elif report.error_message:
            self.rejected_orders += 1

        if report.slippage_percent != 0:
            self._slippages.append(report.slippage_percent)
            self.avg_slippage_percent = sum(self._slippages) / len(self._slippages)

        if report.latency_ms:
            self._latencies.append(report.latency_ms)
            self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    @property
    def fill_rate(self) -> float:
        """Overall fill rate."""
        if self.total_orders == 0:
            return 0.0
        return (self.filled_orders / self.total_orders) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'total_orders': self.total_orders,
            'filled_orders': self.filled_orders,
            'fill_rate': f"{self.fill_rate:.1f}%",
            'partial_fills': self.partial_fills,
            'rejected_orders': self.rejected_orders,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'avg_slippage_percent': f"{self.avg_slippage_percent:.3f}%",
            'avg_latency_ms': f"{self.avg_latency_ms:.1f}ms",
        }


class ExecutionEngine:
    """
    Order Execution Engine (Lean-Inspired).

    Handles:
    - Order queue processing
    - Smart order routing (future)
    - Fill tracking
    - Execution analysis

    Example:
        engine = ExecutionEngine(brokerage)

        # Submit order
        report = engine.execute(order, expected_price=150.0)

        # Get statistics
        stats = engine.get_statistics()
        print(f"Avg slippage: {stats.avg_slippage_percent}%")
    """

    def __init__(self, brokerage: 'BaseBrokerage'):
        """
        Initialize execution engine.

        Args:
            brokerage: Connected brokerage instance
        """
        self._brokerage = brokerage
        self._lock = threading.Lock()

        # Tracking
        self._pending_orders: Dict[str, 'Order'] = {}
        self._reports: Dict[str, ExecutionReport] = {}
        self._stats = ExecutionStats()

        # Order queue for async processing
        self._order_queue: Queue = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info("ExecutionEngine initialized")

    def start(self):
        """Start execution engine (async mode)."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._process_queue,
            daemon=True,
            name="ExecutionEngine"
        )
        self._thread.start()
        logger.info("ExecutionEngine started")

    def stop(self):
        """Stop execution engine."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("ExecutionEngine stopped")

    def _process_queue(self):
        """Process order queue."""
        while self._running:
            try:
                order, expected_price = self._order_queue.get(timeout=1.0)
                self._execute_order(order, expected_price)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def execute(
        self,
        order: 'Order',
        expected_price: float = 0.0,
        sync: bool = True
    ) -> Optional[ExecutionReport]:
        """
        Execute an order.

        Args:
            order: Order to execute
            expected_price: Expected fill price (for slippage calc)
            sync: If True, wait for completion. If False, queue async.

        Returns:
            ExecutionReport if sync, None if async
        """
        if sync:
            return self._execute_order(order, expected_price)
        else:
            self._order_queue.put((order, expected_price))
            return None

    def _execute_order(
        self,
        order: 'Order',
        expected_price: float
    ) -> ExecutionReport:
        """Internal order execution."""
        # Create report
        report = ExecutionReport(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.name,
            quantity=order.quantity,
            filled_quantity=0,
            average_price=0,
            commission=0,
            submitted_time=datetime.utcnow(),
            expected_price=expected_price
        )

        try:
            # Submit to brokerage
            ticket = self._brokerage.place_order(order)

            with self._lock:
                self._pending_orders[order.id] = order

            # Wait for fill (with timeout)
            filled = ticket.wait_for_fill(timeout=30.0)

            # Update report
            report.filled_quantity = order.filled_quantity
            report.average_price = order.average_fill_price or expected_price
            report.filled_time = datetime.utcnow()
            report.is_complete = filled and order.filled_quantity >= order.quantity
            report.is_partial = 0 < order.filled_quantity < order.quantity

            # Calculate slippage
            if expected_price > 0:
                report.calculate_slippage(expected_price)

            # Get commission from brokerage if available
            # (simplified - actual commission comes from broker)
            report.commission = report.fill_value * 0.001  # 0.1% estimate

            with self._lock:
                if order.id in self._pending_orders:
                    del self._pending_orders[order.id]

        except Exception as e:
            report.error_message = str(e)
            report.is_complete = False
            logger.error(f"Execution error: {e}")

        # Record statistics
        with self._lock:
            self._reports[order.id] = report
            self._stats.add_execution(report)

        return report

    def get_report(self, order_id: str) -> Optional[ExecutionReport]:
        """Get execution report for an order."""
        with self._lock:
            return self._reports.get(order_id)

    def get_statistics(self) -> ExecutionStats:
        """Get aggregate execution statistics."""
        with self._lock:
            return self._stats

    def get_recent_reports(self, count: int = 10) -> List[dict]:
        """Get recent execution reports."""
        with self._lock:
            reports = list(self._reports.values())[-count:]
            return [r.to_dict() for r in reports]

    def get_pending_count(self) -> int:
        """Get count of pending orders."""
        with self._lock:
            return len(self._pending_orders)
