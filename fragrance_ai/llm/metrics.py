# fragrance_ai/llm/metrics.py
"""
Prometheus Metrics for LLM Ensemble
레이트/지연/에러 메트릭 수집
"""

import time
import logging
from typing import Optional, Dict, Callable
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import prometheus_client, but make it optional
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Info,
        generate_latest,
        REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("prometheus_client not installed. Metrics will be disabled.")
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Metrics Definitions
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Request counters
    llm_requests_total = Counter(
        'llm_requests_total',
        'Total number of LLM requests',
        ['model', 'mode', 'status']
    )

    # Latency histograms
    llm_inference_duration_seconds = Histogram(
        'llm_inference_duration_seconds',
        'LLM inference duration in seconds',
        ['model', 'mode'],
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float('inf'))
    )

    # Error counters
    llm_errors_total = Counter(
        'llm_errors_total',
        'Total number of LLM errors',
        ['model', 'error_type']
    )

    # Cache metrics
    llm_cache_hits_total = Counter(
        'llm_cache_hits_total',
        'Total number of cache hits'
    )

    llm_cache_misses_total = Counter(
        'llm_cache_misses_total',
        'Total number of cache misses'
    )

    # Model status gauges
    llm_model_loaded = Gauge(
        'llm_model_loaded',
        'Whether model is loaded (1=loaded, 0=not loaded)',
        ['model']
    )

    llm_model_memory_bytes = Gauge(
        'llm_model_memory_bytes',
        'Model memory usage in bytes',
        ['model', 'memory_type']
    )

    # System info
    llm_info = Info(
        'llm_ensemble',
        'LLM Ensemble information'
    )

    # Request rate (requests per second)
    llm_requests_per_second = Gauge(
        'llm_requests_per_second',
        'Current request rate per second',
        ['model', 'mode']
    )

    # Active requests
    llm_active_requests = Gauge(
        'llm_active_requests',
        'Number of requests currently being processed',
        ['model']
    )

else:
    # Dummy metrics when Prometheus is not available
    class DummyMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    llm_requests_total = DummyMetric()
    llm_inference_duration_seconds = DummyMetric()
    llm_errors_total = DummyMetric()
    llm_cache_hits_total = DummyMetric()
    llm_cache_misses_total = DummyMetric()
    llm_model_loaded = DummyMetric()
    llm_model_memory_bytes = DummyMetric()
    llm_info = DummyMetric()
    llm_requests_per_second = DummyMetric()
    llm_active_requests = DummyMetric()


# ============================================================================
# Metrics Collection Helpers
# ============================================================================

@contextmanager
def track_inference_time(model: str, mode: str):
    """
    Context manager to track inference time

    Usage:
        with track_inference_time('qwen', 'fast'):
            result = model.infer(input_text)
    """
    start_time = time.time()
    llm_active_requests.labels(model=model).inc()

    try:
        yield
    finally:
        duration = time.time() - start_time
        llm_inference_duration_seconds.labels(model=model, mode=mode).observe(duration)
        llm_active_requests.labels(model=model).dec()


def record_request(model: str, mode: str, status: str = "success"):
    """
    Record a request

    Args:
        model: Model name (qwen, mistral, llama)
        mode: Request mode (fast, balanced, creative)
        status: Request status (success, error, timeout)
    """
    llm_requests_total.labels(model=model, mode=mode, status=status).inc()


def record_error(model: str, error_type: str):
    """
    Record an error

    Args:
        model: Model name
        error_type: Error type (timeout, oom, inference_error, etc.)
    """
    llm_errors_total.labels(model=model, error_type=error_type).inc()


def record_cache_hit():
    """Record a cache hit"""
    llm_cache_hits_total.inc()


def record_cache_miss():
    """Record a cache miss"""
    llm_cache_misses_total.inc()


def update_model_status(model: str, loaded: bool, memory_bytes: Optional[float] = None):
    """
    Update model status

    Args:
        model: Model name
        loaded: Whether model is loaded
        memory_bytes: Memory usage in bytes
    """
    llm_model_loaded.labels(model=model).set(1 if loaded else 0)

    if memory_bytes is not None:
        llm_model_memory_bytes.labels(model=model, memory_type='ram').set(memory_bytes)


def set_ensemble_info(qwen_version: str, mistral_version: str, llama_version: str):
    """
    Set ensemble information

    Args:
        qwen_version: Qwen model version
        mistral_version: Mistral model version
        llama_version: Llama model version
    """
    llm_info.info({
        'qwen_version': qwen_version,
        'mistral_version': mistral_version,
        'llama_version': llama_version,
        'ensemble_mode': 'qwen_mistral_llama'
    })


# ============================================================================
# Decorators
# ============================================================================

def track_llm_request(model: str, mode: str):
    """
    Decorator to track LLM requests

    Usage:
        @track_llm_request('qwen', 'fast')
        def infer_brief(text):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with track_inference_time(model, mode):
                try:
                    result = func(*args, **kwargs)
                    record_request(model, mode, "success")
                    return result
                except TimeoutError:
                    record_request(model, mode, "timeout")
                    record_error(model, "timeout")
                    raise
                except MemoryError:
                    record_request(model, mode, "error")
                    record_error(model, "oom")
                    raise
                except Exception as e:
                    record_request(model, mode, "error")
                    record_error(model, "inference_error")
                    raise

        return wrapper
    return decorator


# ============================================================================
# Metrics Export
# ============================================================================

def get_metrics() -> bytes:
    """
    Get Prometheus metrics in text format

    Returns:
        Metrics in Prometheus text format
    """
    if PROMETHEUS_AVAILABLE:
        return generate_latest(REGISTRY)
    else:
        return b"# Prometheus client not installed\n"


def get_metrics_dict() -> Dict[str, any]:
    """
    Get metrics as dictionary (for debugging)

    Returns:
        Dictionary of metrics
    """
    if not PROMETHEUS_AVAILABLE:
        return {"error": "Prometheus client not installed"}

    from prometheus_client import REGISTRY

    metrics = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            key = f"{sample.name}_{','.join(f'{k}={v}' for k, v in sample.labels.items())}"
            metrics[key] = sample.value

    return metrics


# ============================================================================
# Request Rate Calculator
# ============================================================================

class RequestRateCalculator:
    """Calculate request rate over time windows"""

    def __init__(self, window_seconds: int = 60):
        """
        Initialize calculator

        Args:
            window_seconds: Time window for rate calculation
        """
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}  # {model: [(timestamp, count), ...]}

    def record_request(self, model: str):
        """Record a request"""
        now = time.time()

        if model not in self.requests:
            self.requests[model] = []

        self.requests[model].append((now, 1))

        # Cleanup old entries
        cutoff = now - self.window_seconds
        self.requests[model] = [
            (ts, count) for ts, count in self.requests[model]
            if ts >= cutoff
        ]

    def get_rate(self, model: str) -> float:
        """
        Get request rate for model

        Args:
            model: Model name

        Returns:
            Requests per second
        """
        if model not in self.requests or not self.requests[model]:
            return 0.0

        now = time.time()
        cutoff = now - self.window_seconds

        # Count requests in window
        count = sum(c for ts, c in self.requests[model] if ts >= cutoff)

        # Calculate rate
        return count / self.window_seconds


# Global rate calculator
_rate_calculator = RequestRateCalculator(window_seconds=60)


def update_request_rates():
    """Update request rate gauges (call periodically)"""
    for model in ['qwen', 'mistral', 'llama']:
        for mode in ['fast', 'balanced', 'creative']:
            rate = _rate_calculator.get_rate(f"{model}_{mode}")
            llm_requests_per_second.labels(model=model, mode=mode).set(rate)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'PROMETHEUS_AVAILABLE',
    'track_inference_time',
    'record_request',
    'record_error',
    'record_cache_hit',
    'record_cache_miss',
    'update_model_status',
    'set_ensemble_info',
    'track_llm_request',
    'get_metrics',
    'get_metrics_dict',
    'RequestRateCalculator',
    'update_request_rates'
]
