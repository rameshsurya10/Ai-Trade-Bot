"""
Monitoring Module
=================
Prometheus metrics and health checks for production monitoring.

Features:
- Trading metrics (signals, trades, P&L)
- System metrics (latency, errors)
- Health endpoints
"""

from .metrics import (
    MetricsCollector,
    metrics,
    start_metrics_server,
)

__all__ = [
    'MetricsCollector',
    'metrics',
    'start_metrics_server',
]
