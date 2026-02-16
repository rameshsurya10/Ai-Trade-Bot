"""
Portfolio Management Module (Lean-Inspired)
============================================
Centralized portfolio management with position sizing, risk management,
and multi-asset coordination.

Inspired by QuantConnect's Lean engine portfolio management system.

Features:
- Multi-asset position tracking
- Risk-based position sizing
- Portfolio-level constraints
- Margin and buying power management
- P&L tracking and reporting

Usage:
    from src.portfolio import PortfolioManager, InsightDirection

    portfolio = PortfolioManager(initial_cash=100000)

    # Add holdings
    portfolio.process_fill("AAPL", 100, 150.0, OrderSide.BUY)

    # Calculate position size
    size = portfolio.calculate_position_size("AAPL", 150.0, 145.0)

    # Get portfolio state
    print(f"Total Value: ${portfolio.total_value:,.2f}")
"""

from .manager import (
    PortfolioManager,
    Holding,
    PortfolioTarget,
    InsightDirection,
)
from .risk import (
    RiskManager,
    RiskModel,
    MaximumDrawdownRisk,
    MaximumPositionSizeRisk,
)
from .sizing import (
    PositionSizer,
    EqualWeightSizer,
    KellyCriterionSizer,
)

__all__ = [
    # Manager
    'PortfolioManager',
    'Holding',
    'PortfolioTarget',
    'InsightDirection',
    # Risk
    'RiskManager',
    'RiskModel',
    'MaximumDrawdownRisk',
    'MaximumPositionSizeRisk',
    # Sizing
    'PositionSizer',
    'EqualWeightSizer',
    'KellyCriterionSizer',
]
