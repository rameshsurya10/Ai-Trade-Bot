"""
Backtesting Module
==================
Validate trading strategies against historical data.
"""

from .engine import BacktestEngine
from .metrics import BacktestMetrics, calculate_metrics

__all__ = [
    'BacktestEngine',
    'BacktestMetrics',
    'calculate_metrics',
]
