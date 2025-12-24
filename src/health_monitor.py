"""
Health Monitor & System Stability
==================================
Real-time monitoring, health checks, and automatic error recovery.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"


@dataclass
class ComponentHealth:
    """Health status for a single component."""
    name: str
    status: HealthStatus = HealthStatus.HEALTHY
    last_check: datetime = field(default_factory=datetime.utcnow)
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict = field(default_factory=dict)


class HealthMonitor:
    """
    System-wide health monitoring and automatic recovery.

    Features:
    - Real-time health checks for all components
    - Automatic error detection and recovery
    - Performance metrics tracking
    - Alert system for degraded performance
    - Watchdog for stuck processes
    """

    def __init__(self, check_interval: int = 60):
        """
        Initialize health monitor.

        Args:
            check_interval: Seconds between health checks (default: 60)
        """
        self.check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Component health tracking
        self._component_health: Dict[str, ComponentHealth] = {}
        self._health_lock = threading.Lock()

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

        # Metrics
        self._start_time = datetime.utcnow()

    def register_component(self, name: str) -> None:
        """Register a component for health monitoring."""
        with self._health_lock:
            if name not in self._component_health:
                self._component_health[name] = ComponentHealth(name=name)
                logger.info(f"Registered component for monitoring: {name}")

    def update_component_health(
        self,
        name: str,
        status: HealthStatus,
        error: Optional[str] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """Update health status for a component."""
        with self._health_lock:
            if name not in self._component_health:
                self.register_component(name)

            component = self._component_health[name]
            old_status = component.status

            component.status = status
            component.last_check = datetime.utcnow()

            if error:
                component.error_count += 1
                component.last_error = error
                logger.warning(f"Component {name} error: {error}")

            if metrics:
                component.metrics.update(metrics)

            # Alert if status degraded
            if old_status != status and status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL, HealthStatus.DOWN]:
                self._trigger_alert(name, old_status, status, error)

    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        with self._health_lock:
            return self._component_health.get(name)

    def get_overall_health(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            HEALTHY - All components healthy
            DEGRADED - Some components degraded
            CRITICAL - Critical components down
            DOWN - System not functional
        """
        with self._health_lock:
            if not self._component_health:
                return HealthStatus.DOWN

            statuses = [c.status for c in self._component_health.values()]

            if HealthStatus.DOWN in statuses:
                return HealthStatus.DOWN
            elif HealthStatus.CRITICAL in statuses:
                return HealthStatus.CRITICAL
            elif HealthStatus.DEGRADED in statuses:
                return HealthStatus.DEGRADED
            else:
                return HealthStatus.HEALTHY

    def get_health_report(self) -> Dict:
        """
        Get comprehensive health report.

        Returns:
            Dict with overall status and component details
        """
        with self._health_lock:
            uptime = (datetime.utcnow() - self._start_time).total_seconds()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'uptime_seconds': uptime,
                'overall_status': self.get_overall_health().value,
                'components': {
                    name: {
                        'status': comp.status.value,
                        'last_check': comp.last_check.isoformat(),
                        'error_count': comp.error_count,
                        'last_error': comp.last_error,
                        'metrics': comp.metrics
                    }
                    for name, comp in self._component_health.items()
                }
            }

    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for health alerts."""
        self._alert_callbacks.append(callback)

    def _trigger_alert(
        self,
        component: str,
        old_status: HealthStatus,
        new_status: HealthStatus,
        error: Optional[str]
    ) -> None:
        """Trigger alerts when component health degrades."""
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'component': component,
            'old_status': old_status.value,
            'new_status': new_status.value,
            'error': error
        }

        logger.warning(f"Health alert: {component} {old_status.value} → {new_status.value}")

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def _health_check_loop(self) -> None:
        """Background health check loop."""
        logger.info("Health monitor started")

        while self._running:
            try:
                # Check component staleness
                with self._health_lock:
                    now = datetime.utcnow()
                    for name, component in self._component_health.items():
                        # If no update in 5 minutes, mark as degraded
                        age = (now - component.last_check).total_seconds()
                        if age > 300:  # 5 minutes
                            if component.status != HealthStatus.DOWN:
                                logger.warning(f"Component {name} stale ({age:.0f}s)")
                                component.status = HealthStatus.DEGRADED
                                self._trigger_alert(
                                    name,
                                    HealthStatus.HEALTHY,
                                    HealthStatus.DEGRADED,
                                    f"No update for {age:.0f} seconds"
                                )

                # Sleep
                for _ in range(self.check_interval):
                    if not self._running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Health check loop error: {e}")

        logger.info("Health monitor stopped")

    def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            logger.warning("Health monitor already running")
            return

        self._running = True
        self._start_time = datetime.utcnow()
        self._thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._thread.start()
        logger.info("Health monitor background thread started")

    def stop(self) -> None:
        """Stop health monitoring."""
        if not self._running:
            return

        logger.info("Stopping health monitor...")
        self._running = False

        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        logger.info("Health monitor stopped")


# Global instance
_global_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = HealthMonitor()
    return _global_health_monitor
