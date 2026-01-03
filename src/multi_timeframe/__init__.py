"""
Multi-Timeframe Trading Components
===================================

Components for multi-timeframe analysis and continuous learning.

- ModelManager: Manages models per (symbol, interval)
- SignalAggregator: Combines predictions across timeframes
"""

from .model_manager import MultiTimeframeModelManager, ModelMetadata, get_model_manager
from .aggregator import SignalAggregator, TimeframeSignal, AggregatedSignal, AggregationMethod

__all__ = [
    # Model management
    'MultiTimeframeModelManager',
    'ModelMetadata',
    'get_model_manager',

    # Signal aggregation
    'SignalAggregator',
    'TimeframeSignal',
    'AggregatedSignal',
    'AggregationMethod'
]
