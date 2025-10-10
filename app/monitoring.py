"""
Production-ready monitoring and error handling for the RAG pipeline.
"""
from __future__ import annotations
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from .utils import get_logger

logger = get_logger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for tracking query performance and quality."""
    query_id: str
    session_id: str
    query_text: str
    query_category: str
    processing_time_ms: float
    chunks_retrieved: int
    chunks_filtered: int
    sources_shown: bool
    user_satisfaction: Optional[float] = None
    error_occurred: bool = False
    error_message: Optional[str] = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class SystemMetrics:
    """System-level metrics for monitoring."""
    timestamp: float
    active_sessions: int
    total_queries_today: int
    average_response_time_ms: float
    error_rate_percent: float
    memory_usage_mb: float
    cpu_usage_percent: float

class MetricsCollector:
    """Collects and stores metrics for monitoring and analytics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.query_metrics: List[QueryMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self._lock = None  # Would use threading.Lock in production
    
    def record_query(self, metrics: QueryMetrics) -> None:
        """Record query metrics."""
        try:
            self.query_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.query_metrics) > self.max_metrics:
                self.query_metrics = self.query_metrics[-self.max_metrics:]
            
            # Log important metrics
            if metrics.error_occurred:
                logger.error(f"Query error: {metrics.error_message} (Query: {metrics.query_text[:100]})")
            elif metrics.processing_time_ms > 5000:  # Slow queries
                logger.warning(f"Slow query: {metrics.processing_time_ms:.0f}ms (Query: {metrics.query_text[:100]})")
            
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")
    
    def record_system_metrics(self, metrics: SystemMetrics) -> None:
        """Record system metrics."""
        try:
            self.system_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.system_metrics) > 1000:
                self.system_metrics = self.system_metrics[-1000:]
            
            # Log system issues
            if metrics.error_rate_percent > 5.0:
                logger.warning(f"High error rate: {metrics.error_rate_percent:.1f}%")
            if metrics.memory_usage_mb > 1000:
                logger.warning(f"High memory usage: {metrics.memory_usage_mb:.0f}MB")
                
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def get_query_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Get query statistics for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.query_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"total_queries": 0}
        
        total_queries = len(recent_metrics)
        error_count = sum(1 for m in recent_metrics if m.error_occurred)
        avg_response_time = sum(m.processing_time_ms for m in recent_metrics) / total_queries
        
        # Category breakdown
        categories = {}
        for m in recent_metrics:
            categories[m.query_category] = categories.get(m.query_category, 0) + 1
        
        return {
            "total_queries": total_queries,
            "error_rate_percent": (error_count / total_queries) * 100,
            "average_response_time_ms": avg_response_time,
            "categories": categories,
            "sources_shown_rate": sum(1 for m in recent_metrics if m.sources_shown) / total_queries * 100
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.system_metrics:
            return {"status": "unknown"}
        
        latest = self.system_metrics[-1]
        
        # Determine health status
        if latest.error_rate_percent > 10:
            status = "critical"
        elif latest.error_rate_percent > 5 or latest.memory_usage_mb > 1000:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "active_sessions": latest.active_sessions,
            "error_rate_percent": latest.error_rate_percent,
            "memory_usage_mb": latest.memory_usage_mb,
            "cpu_usage_percent": latest.cpu_usage_percent,
            "timestamp": latest.timestamp
        }

class ErrorHandler:
    """Centralized error handling for the RAG pipeline."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def handle_query_error(self, error: Exception, query_text: str, session_id: str) -> str:
        """Handle query processing errors gracefully."""
        error_message = str(error)
        
        # Record error metrics
        metrics = QueryMetrics(
            query_id=f"error_{int(time.time())}",
            session_id=session_id,
            query_text=query_text,
            query_category="error",
            processing_time_ms=0,
            chunks_retrieved=0,
            chunks_filtered=0,
            sources_shown=False,
            error_occurred=True,
            error_message=error_message
        )
        self.metrics_collector.record_query(metrics)
        
        # Return user-friendly error message
        if "timeout" in error_message.lower():
            return "I'm experiencing some delays right now. Please try again in a moment."
        elif "connection" in error_message.lower():
            return "I'm having trouble connecting to our property database. Please try again shortly."
        elif "rate limit" in error_message.lower():
            return "I'm receiving many requests right now. Please wait a moment and try again."
        else:
            return "I encountered an issue processing your request. Please try rephrasing your question or try again later."
    
    def handle_embedding_error(self, error: Exception, text: str) -> None:
        """Handle embedding generation errors."""
        logger.error(f"Embedding error for text '{text[:100]}...': {error}")
        # Could implement fallback strategies here
    
    def handle_vector_search_error(self, error: Exception, query: str) -> None:
        """Handle vector search errors."""
        logger.error(f"Vector search error for query '{query}': {error}")
        # Could implement fallback search strategies here

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation_id: str) -> None:
        """Start timing an operation."""
        self.start_times[operation_id] = time.time()
    
    def end_timer(self, operation_id: str) -> float:
        """End timing an operation and return duration in milliseconds."""
        if operation_id not in self.start_times:
            return 0.0
        
        duration_ms = (time.time() - self.start_times[operation_id]) * 1000
        del self.start_times[operation_id]
        return duration_ms
    
    def get_system_resources(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            import psutil
            return {
                "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
                "cpu_usage_percent": psutil.cpu_percent(),
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            # Fallback if psutil not available - use basic system info
            try:
                # Basic memory info from /proc/meminfo (Linux)
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                mem_total = int([line for line in meminfo.split('\n') if 'MemTotal' in line][0].split()[1]) * 1024
                mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable' in line][0].split()[1]) * 1024
                mem_used = mem_total - mem_available
                
                return {
                    "memory_usage_mb": mem_used / 1024 / 1024,
                    "cpu_usage_percent": 0.0,  # Cannot get CPU without psutil
                    "disk_usage_percent": 0.0  # Cannot get disk without psutil
                }
            except (FileNotFoundError, IndexError, ValueError):
                return {
                    "memory_usage_mb": 0.0,
                    "cpu_usage_percent": 0.0,
                    "disk_usage_percent": 0.0
                }

# Global instances for the application
metrics_collector = MetricsCollector()
error_handler = ErrorHandler(metrics_collector)
performance_monitor = PerformanceMonitor()

def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return error_handler

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor
