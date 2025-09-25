"""
Health Checker

This module provides health checking capabilities for the MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import threading
from loguru import logger


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float
    metadata: Dict[str, Any]


@dataclass
class HealthReport:
    """Overall health report."""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: float
    summary: Dict[str, int]


class HealthChecker:
    """
    Performs health checks on system components.
    
    This class provides:
    - Component health checking
    - Health status aggregation
    - Health monitoring and alerting
    """
    
    def __init__(self):
        """Initialize the health checker."""
        self.checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.lock = threading.Lock()
        self.logger = logger.bind(name="health_checker")
        
        # Health callbacks
        self.callbacks: List[Callable[[HealthReport], None]] = []
    
    def register_check(self, name: str, check_function: Callable[[], HealthCheck]) -> None:
        """
        Register a health check.
        
        Args:
            name: Name of the health check
            check_function: Function that performs the health check
        """
        with self.lock:
            self.checks[name] = check_function
            self.logger.debug(f"Registered health check: {name}")
    
    def unregister_check(self, name: str) -> None:
        """
        Unregister a health check.
        
        Args:
            name: Name of the health check to remove
        """
        with self.lock:
            if name in self.checks:
                del self.checks[name]
                self.logger.debug(f"Unregistered health check: {name}")
    
    def run_check(self, name: str) -> Optional[HealthCheck]:
        """
        Run a specific health check.
        
        Args:
            name: Name of the health check to run
            
        Returns:
            Health check result or None if check not found
        """
        with self.lock:
            check_function = self.checks.get(name)
        
        if not check_function:
            self.logger.warning(f"Health check '{name}' not found")
            return None
        
        try:
            return check_function()
        except Exception as e:
            self.logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                timestamp=time.time(),
                duration=0.0,
                metadata={"error": str(e)}
            )
    
    def run_all_checks(self) -> HealthReport:
        """
        Run all registered health checks.
        
        Returns:
            Health report with all check results
        """
        checks = []
        start_time = time.time()
        
        with self.lock:
            check_names = list(self.checks.keys())
        
        for name in check_names:
            check_result = self.run_check(name)
            if check_result:
                checks.append(check_result)
        
        # Determine overall status
        overall_status = self._determine_overall_status(checks)
        
        # Create summary
        summary = {
            "total_checks": len(checks),
            "healthy": len([c for c in checks if c.status == HealthStatus.HEALTHY]),
            "degraded": len([c for c in checks if c.status == HealthStatus.DEGRADED]),
            "unhealthy": len([c for c in checks if c.status == HealthStatus.UNHEALTHY]),
            "unknown": len([c for c in checks if c.status == HealthStatus.UNKNOWN])
        }
        
        report = HealthReport(
            overall_status=overall_status,
            checks=checks,
            timestamp=time.time(),
            summary=summary
        )
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(report)
            except Exception as e:
                self.logger.error(f"Error in health check callback: {e}")
        
        return report
    
    def _determine_overall_status(self, checks: List[HealthCheck]) -> HealthStatus:
        """
        Determine overall health status from individual checks.
        
        Args:
            checks: List of health check results
            
        Returns:
            Overall health status
        """
        if not checks:
            return HealthStatus.UNKNOWN
        
        # Priority: UNHEALTHY > DEGRADED > UNKNOWN > HEALTHY
        statuses = [check.status for check in checks]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def add_callback(self, callback: Callable[[HealthReport], None]) -> None:
        """
        Add a callback for health reports.
        
        Args:
            callback: Function to call when health reports are generated
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[HealthReport], None]) -> None:
        """
        Remove a health check callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_check_names(self) -> List[str]:
        """
        Get names of all registered health checks.
        
        Returns:
            List of health check names
        """
        with self.lock:
            return list(self.checks.keys())
    
    def is_healthy(self) -> bool:
        """
        Quick check if system is healthy.
        
        Returns:
            True if system is healthy
        """
        report = self.run_all_checks()
        return report.overall_status == HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system health.
        
        Returns:
            Health summary
        """
        report = self.run_all_checks()
        
        return {
            "overall_status": report.overall_status.value,
            "total_checks": report.summary["total_checks"],
            "healthy_checks": report.summary["healthy"],
            "degraded_checks": report.summary["degraded"],
            "unhealthy_checks": report.summary["unhealthy"],
            "unknown_checks": report.summary["unknown"],
            "timestamp": report.timestamp,
            "is_healthy": report.overall_status == HealthStatus.HEALTHY
        }


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_global_health_checker() -> HealthChecker:
    """
    Get the global health checker instance.
    
    Returns:
        Global health checker
    """
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def register_health_check(name: str, check_function: Callable[[], HealthCheck]) -> None:
    """
    Register a health check in the global checker.
    
    Args:
        name: Name of the health check
        check_function: Function that performs the health check
    """
    get_global_health_checker().register_check(name, check_function)


def run_health_check(name: str) -> Optional[HealthCheck]:
    """
    Run a health check from the global checker.
    
    Args:
        name: Name of the health check to run
        
    Returns:
        Health check result or None if check not found
    """
    return get_global_health_checker().run_check(name)


def run_all_health_checks() -> HealthReport:
    """
    Run all health checks from the global checker.
    
    Returns:
        Health report with all check results
    """
    return get_global_health_checker().run_all_checks()


def is_system_healthy() -> bool:
    """
    Quick check if the system is healthy.
    
    Returns:
        True if system is healthy
    """
    return get_global_health_checker().is_healthy()

