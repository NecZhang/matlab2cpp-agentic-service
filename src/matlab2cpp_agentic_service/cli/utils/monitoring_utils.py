"""
Monitoring Utilities for CLI

This module provides monitoring helper functions for the CLI commands.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from ...infrastructure.monitoring import (
    get_global_metrics_collector,
    get_global_health_checker,
    AgentPerformanceMonitor,
    HealthChecker,
    HealthStatus,
    HealthCheck,
    record_metric,
    record_timing,
    record_count,
    register_health_check
)


class CLIMonitoringManager:
    """Manages monitoring for CLI operations."""
    
    def __init__(self):
        self.logger = logger.bind(name="cli_monitoring")
        self.metrics_collector = get_global_metrics_collector()
        self.health_checker = get_global_health_checker()
        self.performance_monitor = None
        self.operation_id = None
        
        # Register CLI-specific health checks
        self._register_cli_health_checks()
    
    def _register_cli_health_checks(self):
        """Register health checks specific to CLI operations."""
        
        def check_llm_connectivity():
            """Check if LLM service is accessible."""
            try:
                start_time = time.time()
                # Import here to avoid circular imports
                from ...infrastructure.tools.llm_client import test_llm_connection
                
                success = test_llm_connection()
                duration = time.time() - start_time
                
                return HealthCheck(
                    name="llm_connectivity",
                    status=HealthStatus.HEALTHY if success else HealthStatus.UNHEALTHY,
                    message=f"LLM connection {'OK' if success else 'Failed'}",
                    timestamp=time.time(),
                    duration=duration,
                    metadata={"endpoint": "vllm", "latency": duration}
                )
            except Exception as e:
                return HealthCheck(
                    name="llm_connectivity",
                    status=HealthStatus.UNHEALTHY,
                    message=f"LLM connection error: {e}",
                    timestamp=time.time(),
                    duration=0.0,
                    metadata={"error": str(e)}
                )
        
        def check_disk_space():
            """Check available disk space."""
            try:
                import shutil
                free_space = shutil.disk_usage('.').free
                free_gb = free_space / (1024**3)
                
                status = HealthStatus.HEALTHY
                if free_gb < 1.0:  # Less than 1GB
                    status = HealthStatus.UNHEALTHY
                elif free_gb < 5.0:  # Less than 5GB
                    status = HealthStatus.DEGRADED
                
                return HealthCheck(
                    name="disk_space",
                    status=status,
                    message=f"Free disk space: {free_gb:.1f}GB",
                    timestamp=time.time(),
                    duration=0.0,
                    metadata={"free_gb": free_gb, "threshold_gb": 1.0}
                )
            except Exception as e:
                return HealthCheck(
                    name="disk_space",
                    status=HealthStatus.UNKNOWN,
                    message=f"Could not check disk space: {e}",
                    timestamp=time.time(),
                    duration=0.0,
                    metadata={"error": str(e)}
                )
        
        def check_output_directory():
            """Check if output directory is writable."""
            try:
                start_time = time.time()
                test_dir = Path("output")
                test_dir.mkdir(exist_ok=True)
                
                # Try to write a test file
                test_file = test_dir / "test_write.tmp"
                test_file.write_text("test")
                test_file.unlink()
                
                duration = time.time() - start_time
                
                return HealthCheck(
                    name="output_directory",
                    status=HealthStatus.HEALTHY,
                    message="Output directory is writable",
                    timestamp=time.time(),
                    duration=duration,
                    metadata={"test_dir": str(test_dir)}
                )
            except Exception as e:
                return HealthCheck(
                    name="output_directory",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Output directory error: {e}",
                    timestamp=time.time(),
                    duration=time.time() - start_time,
                    metadata={"error": str(e)}
                )
        
        # Register health checks
        register_health_check("llm_connectivity", check_llm_connectivity)
        register_health_check("disk_space", check_disk_space)
        register_health_check("output_directory", check_output_directory)
    
    def run_health_check(self) -> bool:
        """Run quick health check."""
        try:
            return self.health_checker.is_healthy()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def get_detailed_health_report(self) -> Dict[str, Any]:
        """Get detailed health report."""
        try:
            report = self.health_checker.run_all_checks()
            return {
                "overall_status": report.overall_status.value,
                "total_checks": report.summary["total_checks"],
                "healthy": report.summary["healthy"],
                "degraded": report.summary["degraded"],
                "unhealthy": report.summary["unhealthy"],
                "unknown": report.summary["unknown"],
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status.value,
                        "message": check.message,
                        "duration": check.duration,
                        "metadata": check.metadata
                    }
                    for check in report.checks
                ],
                "timestamp": report.timestamp
            }
        except Exception as e:
            self.logger.error(f"Failed to get health report: {e}")
            return {"error": str(e)}
    
    def start_conversion_monitoring(self, matlab_path: Path, project_name: str, 
                                  max_turns: int, conversion_mode: str) -> str:
        """Start monitoring a conversion operation."""
        try:
            # Initialize performance monitor
            self.performance_monitor = AgentPerformanceMonitor(enable_real_time_monitoring=True)
            
            # Start operation tracking
            self.operation_id = self.performance_monitor.start_operation(
                agent_name="cli_converter",
                operation_name="convert_project",
                metadata={
                    "matlab_path": str(matlab_path),
                    "project_name": project_name,
                    "max_turns": max_turns,
                    "conversion_mode": conversion_mode,
                    "is_multi_file": matlab_path.is_dir()
                }
            )
            
            # Record conversion start
            record_count("conversions_started", 1, tags={
                "project_type": "multi_file" if matlab_path.is_dir() else "single_file",
                "max_turns": str(max_turns),
                "conversion_mode": conversion_mode
            })
            
            self.logger.info(f"Started monitoring conversion operation: {self.operation_id}")
            return self.operation_id
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return None
    
    def end_conversion_monitoring(self, success: bool, result: Optional[Dict[str, Any]] = None, 
                                error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """End monitoring a conversion operation."""
        try:
            if not self.performance_monitor or not self.operation_id:
                self.logger.warning("No active monitoring session to end")
                return None
            
            # End performance monitoring
            metrics = self.performance_monitor.end_operation(
                operation_id=self.operation_id,
                success=success,
                error_message=error_message
            )
            
            if success and result:
                # Record success metrics
                record_timing("conversion_duration", result.get("total_time", 0), tags={
                    "project_type": "multi_file" if result.get("is_multi_file", False) else "single_file",
                    "success": "true"
                })
                
                record_metric("final_quality_score", result.get("final_quality_score", 0), tags={
                    "project_name": result.get("project_name", "unknown"),
                    "optimization_turns": str(result.get("optimization_turns", 0))
                })
                
                record_count("files_generated", result.get("generated_files", 0), tags={
                    "project_type": "multi_file" if result.get("is_multi_file", False) else "single_file"
                })
                
                record_count("conversions_completed", 1, tags={
                    "success": "true",
                    "project_type": "multi_file" if result.get("is_multi_file", False) else "single_file"
                })
            else:
                # Record failure metrics
                record_count("conversions_failed", 1, tags={
                    "error_type": type(error_message).__name__ if error_message else "unknown"
                })
            
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
            
            self.logger.info(f"Ended monitoring conversion operation: {self.operation_id}")
            
            # Reset state
            self.performance_monitor = None
            self.operation_id = None
            
            return {
                "metrics": metrics,
                "success": success,
                "error_message": error_message
            }
            
        except Exception as e:
            self.logger.error(f"Failed to end monitoring: {e}")
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        try:
            if not self.performance_monitor:
                self.performance_monitor = AgentPerformanceMonitor()
            
            summary = self.performance_monitor.get_all_performance_summary()
            
            return {
                "total_operations": summary["total_operations"],
                "total_agents": summary["total_agents"],
                "system_metrics": summary["system_metrics"],
                "agents": summary["agents"],
                "insights": summary["performance_insights"],
                "timestamp": time.time()
            }
        except Exception as e:
            self.logger.error(f"Failed to get performance report: {e}")
            return {"error": str(e)}
    
    def get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        try:
            if not self.performance_monitor:
                self.performance_monitor = AgentPerformanceMonitor()
            
            return self.performance_monitor.get_performance_recommendations()
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return [f"Error getting recommendations: {e}"]
    
    def export_metrics(self, output_path: Path) -> bool:
        """Export metrics data to file."""
        try:
            # Export metrics data
            metrics_data = self.metrics_collector.export_metrics()
            
            # Export performance data if available
            performance_data = None
            if self.performance_monitor:
                performance_data = self.performance_monitor.export_performance_data(
                    output_path / "performance_data.json"
                )
            
            # Combine data
            export_data = {
                "export_timestamp": time.time(),
                "metrics": metrics_data,
                "performance": performance_data,
                "health_report": self.get_detailed_health_report()
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported monitoring data to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def clear_metrics(self) -> bool:
        """Clear all metrics data."""
        try:
            self.metrics_collector.clear_metrics()
            self.logger.info("Cleared all metrics data")
            return True
        except Exception as e:
            self.logger.error(f"Failed to clear metrics: {e}")
            return False
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        try:
            return self.metrics_collector.get_collector_summary()
        except Exception as e:
            self.logger.error(f"Failed to get metrics summary: {e}")
            return {"error": str(e)}


# Global monitoring manager instance
_global_monitoring_manager: Optional[CLIMonitoringManager] = None


def get_monitoring_manager() -> CLIMonitoringManager:
    """Get the global monitoring manager instance."""
    global _global_monitoring_manager
    if _global_monitoring_manager is None:
        _global_monitoring_manager = CLIMonitoringManager()
    return _global_monitoring_manager


def run_health_check() -> bool:
    """Run quick health check."""
    return get_monitoring_manager().run_health_check()


def get_health_report() -> Dict[str, Any]:
    """Get detailed health report."""
    return get_monitoring_manager().get_detailed_health_report()


def setup_conversion_monitoring(matlab_path: Path, project_name: str, 
                               max_turns: int, conversion_mode: str) -> str:
    """Setup monitoring for conversion operation."""
    return get_monitoring_manager().start_conversion_monitoring(
        matlab_path, project_name, max_turns, conversion_mode
    )


def export_conversion_metrics(result: Dict[str, Any], output_path: Path) -> bool:
    """Export metrics after conversion."""
    return get_monitoring_manager().export_metrics(output_path)


def get_performance_report() -> Dict[str, Any]:
    """Generate performance report."""
    return get_monitoring_manager().get_performance_report()


def get_performance_recommendations() -> List[str]:
    """Get performance recommendations."""
    return get_monitoring_manager().get_performance_recommendations()



