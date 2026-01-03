"""
Live Trading Module (Lean-Inspired)
===================================
Production-ready live trading with real brokerages.

Connects ML predictions to actual order execution with:
- Real-time data streaming
- Risk management
- Portfolio management
- Error recovery
- Comprehensive logging

Usage:
    from src.live_trading import LiveTradingRunner

    runner = LiveTradingRunner(config_path="config.yaml")
    runner.add_symbol("BTC/USD", exchange="binance")
    runner.add_symbol("ETH/USD", exchange="binance")

    runner.start()  # Begins live trading loop
"""

from .runner import (
    LiveTradingRunner,
    TradingMode,
    RunnerStatus,
)
from .execution import (
    ExecutionEngine,
    ExecutionReport,
)

__all__ = [
    'LiveTradingRunner',
    'TradingMode',
    'RunnerStatus',
    'ExecutionEngine',
    'ExecutionReport',
]
