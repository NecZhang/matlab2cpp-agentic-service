"""
Metrics Collector

This module provides metrics collection capabilities for the MATLAB2C++ conversion service.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import threading
from loguru import logger


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary of metrics for a specific name."""
    name: str
    count: int
    sum_value: float
    min_value: float
    max_value: float
    avg_value: float
    latest_value: float
    latest_timestamp: float


class MetricsCollector:
    """
    Collects and aggregates metrics for the conversion service.
    
    This class provides:
    - Metric collection and storage
    - Real-time aggregation
    - Metric querying and filtering
    - Export capabilities
    """
    
    def __init__(self, max_metrics_per_name: int = 1000):
        """
        Initialize the metrics collector.
        
        Args:
            max_metrics_per_name: Maximum number of metrics to keep per name
        """
        self.max_metrics_per_name = max_metrics_per_name
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_name))
        self.lock = threading.Lock()
        self.logger = logger.bind(name="metrics_collector")
        
        # Metric callbacks
        self.callbacks: List[Callable[[Metric], None]] = []
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for categorization
            metadata: Optional metadata
        """
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.error(f"Error in metric callback: {e}")
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timing metric.
        
        Args:
            name: Metric name
            duration: Duration in seconds
            tags: Optional tags
        """
        self.record_metric(f"{name}_duration", duration, tags)
    
    def record_count(self, name: str, count: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a count metric.
        
        Args:
            name: Metric name
            count: Count value
            tags: Optional tags
        """
        self.record_metric(f"{name}_count", count, tags)
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a gauge metric.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags
        """
        self.record_metric(f"{name}_gauge", value, tags)
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            
        Returns:
            Metric summary or None if no data
        """
        with self.lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            values = [m.value for m in self.metrics[name]]
            latest = self.metrics[name][-1]
            
            return MetricSummary(
                name=name,
                count=len(values),
                sum_value=sum(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=sum(values) / len(values),
                latest_value=latest.value,
                latest_timestamp=latest.timestamp
            )
    
    def get_metrics_by_name(self, name: str, limit: Optional[int] = None) -> List[Metric]:
        """
        Get metrics by name.
        
        Args:
            name: Metric name
            limit: Maximum number of metrics to return
            
        Returns:
            List of metrics
        """
        with self.lock:
            if name not in self.metrics:
                return []
            
            metrics = list(self.metrics[name])
            if limit:
                metrics = metrics[-limit:]
            
            return metrics
    
    def get_metrics_by_tag(self, tag_key: str, tag_value: str, 
                          limit: Optional[int] = None) -> List[Metric]:
        """
        Get metrics by tag.
        
        Args:
            tag_key: Tag key to filter by
            tag_value: Tag value to filter by
            limit: Maximum number of metrics to return
            
        Returns:
            List of matching metrics
        """
        matching_metrics = []
        
        with self.lock:
            for name, metrics in self.metrics.items():
                for metric in metrics:
                    if metric.tags.get(tag_key) == tag_value:
                        matching_metrics.append(metric)
        
        # Sort by timestamp
        matching_metrics.sort(key=lambda m: m.timestamp)
        
        if limit:
            matching_metrics = matching_metrics[-limit:]
        
        return matching_metrics
    
    def get_all_metric_names(self) -> List[str]:
        """
        Get all metric names.
        
        Returns:
            List of metric names
        """
        with self.lock:
            return list(self.metrics.keys())
    
    def clear_metrics(self, name: Optional[str] = None) -> None:
        """
        Clear metrics.
        
        Args:
            name: Specific metric name to clear, or None to clear all
        """
        with self.lock:
            if name:
                if name in self.metrics:
                    self.metrics[name].clear()
                    self.logger.info(f"Cleared metrics for: {name}")
            else:
                self.metrics.clear()
                self.logger.info("Cleared all metrics")
    
    def add_callback(self, callback: Callable[[Metric], None]) -> None:
        """
        Add a callback for new metrics.
        
        Args:
            callback: Function to call when new metrics are recorded
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Metric], None]) -> None:
        """
        Remove a metric callback.
        
        Args:
            callback: Callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_collector_summary(self) -> Dict[str, Any]:
        """
        Get summary of the metrics collector.
        
        Returns:
            Collector summary
        """
        with self.lock:
            total_metrics = sum(len(metrics) for metrics in self.metrics.values())
            metric_names = list(self.metrics.keys())
            
            return {
                "total_metrics": total_metrics,
                "metric_names": metric_names,
                "metrics_per_name": {
                    name: len(metrics) for name, metrics in self.metrics.items()
                },
                "max_metrics_per_name": self.max_metrics_per_name,
                "callback_count": len(self.callbacks)
            }
    
    def export_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Export metrics data.
        
        Args:
            name: Specific metric name to export, or None for all
            
        Returns:
            Exported metrics data
        """
        with self.lock:
            export_data = {
                "timestamp": time.time(),
                "metrics": {}
            }
            
            if name:
                if name in self.metrics:
                    export_data["metrics"][name] = [
                        {
                            "value": m.value,
                            "timestamp": m.timestamp,
                            "tags": m.tags,
                            "metadata": m.metadata
                        }
                        for m in self.metrics[name]
                    ]
            else:
                for metric_name, metrics in self.metrics.items():
                    export_data["metrics"][metric_name] = [
                        {
                            "value": m.value,
                            "timestamp": m.timestamp,
                            "tags": m.tags,
                            "metadata": m.metadata
                        }
                        for m in metrics
                    ]
            
            return export_data


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_global_metrics_collector() -> MetricsCollector:
    """
    Get the global metrics collector instance.
    
    Returns:
        Global metrics collector
    """
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Record a metric in the global collector.
    
    Args:
        name: Metric name
        value: Metric value
        tags: Optional tags
        metadata: Optional metadata
    """
    get_global_metrics_collector().record_metric(name, value, tags, metadata)


def record_timing(name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a timing metric in the global collector.
    
    Args:
        name: Metric name
        duration: Duration in seconds
        tags: Optional tags
    """
    get_global_metrics_collector().record_timing(name, duration, tags)


def record_count(name: str, count: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a count metric in the global collector.
    
    Args:
        name: Metric name
        count: Count value
        tags: Optional tags
    """
    get_global_metrics_collector().record_count(name, count, tags)


def record_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a gauge metric in the global collector.
    
    Args:
        name: Metric name
        value: Gauge value
        tags: Optional tags
    """
    get_global_metrics_collector().record_gauge(name, value, tags)

