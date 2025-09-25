"""
Agent Performance Monitor

This module provides performance monitoring and optimization capabilities
for LangGraph agents in the MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import statistics
import json
from pathlib import Path
from loguru import logger

import psutil
import threading


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent operation."""
    agent_name: str
    operation_name: str
    start_time: float
    end_time: float
    execution_time: float
    success: bool
    memory_usage_mb: float
    cpu_percent: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceProfile:
    """Performance profile for a specific agent."""
    agent_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    success_rate: float = 0.0
    recent_operations: List[PerformanceMetrics] = field(default_factory=list)
    performance_trend: List[float] = field(default_factory=list)


class AgentPerformanceMonitor:
    """
    Performance monitor for LangGraph agents.
    
    This class provides comprehensive performance tracking, analysis,
    and optimization recommendations for agent operations.
    """
    
    def __init__(self, max_history_size: int = 1000, enable_real_time_monitoring: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            max_history_size: Maximum number of metrics to keep in history
            enable_real_time_monitoring: Whether to enable real-time monitoring
        """
        self.max_history_size = max_history_size
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # Performance data storage
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.agent_profiles: Dict[str, AgentPerformanceProfile] = {}
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.system_metrics: deque = deque(maxlen=100)
        
        self.logger = logger.bind(name="performance_monitor")
        
        if enable_real_time_monitoring:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start real-time system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Started real-time performance monitoring")
    
    def stop_monitoring(self):
        """Stop real-time system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped real-time performance monitoring")
    
    def _monitor_system(self):
        """Monitor system performance in background thread."""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                system_metric = {
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
                
                self.system_metrics.append(system_metric)
                
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
            
            time.sleep(5)  # Monitor every 5 seconds
    
    def start_operation(self, agent_name: str, operation_name: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start monitoring an operation.
        
        Args:
            agent_name: Name of the agent
            operation_name: Name of the operation
            metadata: Optional metadata for the operation
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{agent_name}_{operation_name}_{time.time()}"
        
        # Store operation start data
        operation_data = {
            "operation_id": operation_id,
            "agent_name": agent_name,
            "operation_name": operation_name,
            "start_time": time.time(),
            "metadata": metadata or {}
        }
        
        # Store in temporary tracking (will be completed when operation ends)
        self.operation_stats[operation_id] = operation_data
        
        self.logger.debug(f"Started monitoring operation: {operation_id}")
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool, 
                     error_message: Optional[str] = None) -> PerformanceMetrics:
        """
        End monitoring an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether the operation was successful
            error_message: Error message if operation failed
            
        Returns:
            Performance metrics for the operation
        """
        if operation_id not in self.operation_stats:
            self.logger.warning(f"Unknown operation ID: {operation_id}")
            return None
        
        operation_data = self.operation_stats[operation_id]
        end_time = time.time()
        
        # Calculate metrics
        execution_time = end_time - operation_data["start_time"]
        
        # Get current system metrics
        current_memory = psutil.virtual_memory().percent
        current_cpu = psutil.cpu_percent()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            agent_name=operation_data["agent_name"],
            operation_name=operation_data["operation_name"],
            start_time=operation_data["start_time"],
            end_time=end_time,
            execution_time=execution_time,
            success=success,
            memory_usage_mb=current_memory,
            cpu_percent=current_cpu,
            error_message=error_message,
            metadata=operation_data["metadata"]
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        # Update agent profile
        self._update_agent_profile(metrics)
        
        # Clean up operation data
        del self.operation_stats[operation_id]
        
        self.logger.debug(f"Completed monitoring operation: {operation_id} - "
                         f"{execution_time:.2f}s - {'✓' if success else '✗'}")
        
        return metrics
    
    def _update_agent_profile(self, metrics: PerformanceMetrics):
        """Update agent performance profile with new metrics."""
        agent_name = metrics.agent_name
        
        if agent_name not in self.agent_profiles:
            self.agent_profiles[agent_name] = AgentPerformanceProfile(agent_name=agent_name)
        
        profile = self.agent_profiles[agent_name]
        
        # Update basic stats
        profile.total_operations += 1
        if metrics.success:
            profile.successful_operations += 1
        else:
            profile.failed_operations += 1
        
        # Update execution time stats
        profile.avg_execution_time = (
            (profile.avg_execution_time * (profile.total_operations - 1) + metrics.execution_time) 
            / profile.total_operations
        )
        profile.min_execution_time = min(profile.min_execution_time, metrics.execution_time)
        profile.max_execution_time = max(profile.max_execution_time, metrics.execution_time)
        
        # Update memory and CPU stats
        profile.avg_memory_usage = (
            (profile.avg_memory_usage * (profile.total_operations - 1) + metrics.memory_usage_mb) 
            / profile.total_operations
        )
        profile.avg_cpu_usage = (
            (profile.avg_cpu_usage * (profile.total_operations - 1) + metrics.cpu_percent) 
            / profile.total_operations
        )
        
        # Update success rate
        profile.success_rate = profile.successful_operations / profile.total_operations
        
        # Add to recent operations (keep last 10)
        profile.recent_operations.append(metrics)
        if len(profile.recent_operations) > 10:
            profile.recent_operations.pop(0)
        
        # Update performance trend
        profile.performance_trend.append(metrics.execution_time)
        if len(profile.performance_trend) > 20:  # Keep last 20 operations
            profile.performance_trend.pop(0)
    
    def get_agent_performance(self, agent_name: str) -> Optional[AgentPerformanceProfile]:
        """Get performance profile for a specific agent."""
        return self.agent_profiles.get(agent_name)
    
    def get_all_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all agent performance."""
        summary = {
            "total_agents": len(self.agent_profiles),
            "total_operations": len(self.metrics_history),
            "agents": {},
            "system_metrics": self._get_system_summary(),
            "performance_insights": self._generate_performance_insights()
        }
        
        for agent_name, profile in self.agent_profiles.items():
            summary["agents"][agent_name] = {
                "total_operations": profile.total_operations,
                "success_rate": profile.success_rate,
                "avg_execution_time": profile.avg_execution_time,
                "avg_memory_usage": profile.avg_memory_usage,
                "avg_cpu_usage": profile.avg_cpu_usage,
                "performance_trend": profile.performance_trend[-5:]  # Last 5 operations
            }
        
        return summary
    
    def _get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        if not self.system_metrics:
            return {"status": "No system metrics available"}
        
        recent_metrics = list(self.system_metrics)[-10:]  # Last 10 measurements
        
        return {
            "avg_cpu_percent": statistics.mean([m["cpu_percent"] for m in recent_metrics]),
            "avg_memory_percent": statistics.mean([m["memory_percent"] for m in recent_metrics]),
            "avg_memory_available_gb": statistics.mean([m["memory_available_gb"] for m in recent_metrics]),
            "avg_disk_free_gb": statistics.mean([m["disk_free_gb"] for m in recent_metrics]),
            "monitoring_duration_minutes": (recent_metrics[-1]["timestamp"] - recent_metrics[0]["timestamp"]) / 60
        }
    
    def _generate_performance_insights(self) -> List[str]:
        """Generate performance insights and recommendations."""
        insights = []
        
        # Analyze agent performance
        for agent_name, profile in self.agent_profiles.items():
            if profile.total_operations < 3:  # Need at least 3 operations for insights
                continue
            
            # Success rate insights
            if profile.success_rate < 0.8:
                insights.append(f"Agent '{agent_name}' has low success rate ({profile.success_rate:.1%}). Consider reviewing error handling.")
            
            # Performance trend insights
            if len(profile.performance_trend) >= 5:
                recent_avg = statistics.mean(profile.performance_trend[-3:])
                older_avg = statistics.mean(profile.performance_trend[-6:-3])
                
                if recent_avg > older_avg * 1.5:  # 50% slower
                    insights.append(f"Agent '{agent_name}' shows performance degradation. Recent operations are {recent_avg/older_avg:.1f}x slower.")
                elif recent_avg < older_avg * 0.7:  # 30% faster
                    insights.append(f"Agent '{agent_name}' shows performance improvement. Recent operations are {older_avg/recent_avg:.1f}x faster.")
        
        # System insights
        if self.system_metrics:
            recent_memory = list(self.system_metrics)[-1]["memory_percent"]
            if recent_memory > 80:
                insights.append(f"High system memory usage ({recent_memory:.1f}%). Consider optimizing memory usage.")
        
        return insights
    
    def export_performance_data(self, file_path: Union[str, Path]) -> bool:
        """Export performance data to JSON file."""
        try:
            export_data = {
                "export_timestamp": time.time(),
                "summary": self.get_all_performance_summary(),
                "detailed_metrics": [
                    {
                        "agent_name": m.agent_name,
                        "operation_name": m.operation_name,
                        "execution_time": m.execution_time,
                        "success": m.success,
                        "memory_usage_mb": m.memory_usage_mb,
                        "cpu_percent": m.cpu_percent,
                        "timestamp": m.start_time,
                        "error_message": m.error_message
                    }
                    for m in self.metrics_history
                ],
                "system_metrics": list(self.system_metrics)
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported performance data to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export performance data: {e}")
            return False
    
    def get_performance_recommendations(self, agent_name: Optional[str] = None) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []
        
        if agent_name:
            profile = self.agent_profiles.get(agent_name)
            if not profile:
                return [f"No performance data available for agent: {agent_name}"]
            
            # Agent-specific recommendations
            if profile.success_rate < 0.9:
                recommendations.append(f"Increase retry count or improve error handling for {agent_name}")
            
            if profile.avg_execution_time > 30:  # More than 30 seconds
                recommendations.append(f"Consider optimizing {agent_name} - average execution time is {profile.avg_execution_time:.1f}s")
            
            if profile.avg_memory_usage > 100:  # More than 100MB
                recommendations.append(f"Consider memory optimization for {agent_name} - average usage is {profile.avg_memory_usage:.1f}MB")
        
        else:
            # General recommendations
            total_operations = sum(p.total_operations for p in self.agent_profiles.values())
            if total_operations > 100:
                recommendations.append("Consider implementing caching for frequently used operations")
            
            # Check for system resource issues
            if self.system_metrics:
                recent_memory = list(self.system_metrics)[-1]["memory_percent"]
                if recent_memory > 85:
                    recommendations.append("System memory usage is high - consider reducing concurrent operations")
        
        return recommendations
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.stop_monitoring()
