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

class MetricsCollector:
    """Collects and stores metrics for monitoring and analytics."""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self.query_metrics: List[QueryMetrics] = []
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
                logger.debug(f"Slow query: {metrics.processing_time_ms:.0f}ms (Query: {metrics.query_text[:100]})")
            
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")

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
