"""
Core Module - Shared Components
===============================
Base classes, configuration, and utilities used across the system.
"""

from .config import Config
from .database import Database
from .logger import setup_logger, get_logger
from .types import Signal, SignalType, SignalStrength, Candle, Prediction, TradeResult

__all__ = [
    'Config',
    'Database',
    'setup_logger',
    'get_logger',
    'Signal',
    'SignalType',
    'SignalStrength',
    'Candle',
    'Prediction',
    'TradeResult',
]
