"""
Core Module - Shared Components
===============================
Base classes, configuration, and utilities used across the system.

Includes:
- Config: Configuration management
- Database: SQLite database operations
- Metrics: Performance metrics calculation
- Validation: Order validation
- Resilience: Circuit breaker, reconnection
- Types: Shared data types
"""

from .config import Config
from .database import Database
from .logger import setup_logger, get_logger
from .types import Signal, SignalType, SignalStrength, Candle, Prediction, TradeResult
from .metrics import MetricsCalculator, PerformanceMetrics, SignalQualityScorer
from .validation import OrderValidator, OrderValidationResult, validate_order
from .resilience import CircuitBreaker, ReconnectionManager, HealthMonitor

__all__ = [
    # Config
    'Config',

    # Database
    'Database',

    # Logging
    'setup_logger',
    'get_logger',

    # Types
    'Signal',
    'SignalType',
    'SignalStrength',
    'Candle',
    'Prediction',
    'TradeResult',

    # Metrics
    'MetricsCalculator',
    'PerformanceMetrics',
    'SignalQualityScorer',

    # Validation
    'OrderValidator',
    'OrderValidationResult',
    'validate_order',

    # Resilience
    'CircuitBreaker',
    'ReconnectionManager',
    'HealthMonitor',
]
