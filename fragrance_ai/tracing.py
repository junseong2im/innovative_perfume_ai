"""
OpenTelemetry Tracing Integration
전체 요청 추적: /evolve/options → /evolve/feedback
"""

import time
import functools
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    # Dummy implementations for when OpenTelemetry is not installed


# ============================================================================
# Dummy Implementations (when OpenTelemetry not available)
# ============================================================================

class DummySpan:
    """Dummy span for when tracing is disabled"""

    def set_attribute(self, key: str, value: Any):
        pass

    def set_status(self, status, description: str = ""):
        pass

    def record_exception(self, exception: Exception):
        pass

    def end(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DummyTracer:
    """Dummy tracer for when tracing is disabled"""

    def start_as_current_span(self, name: str, **kwargs):
        return DummySpan()

    def start_span(self, name: str, **kwargs):
        return DummySpan()


# ============================================================================
# Tracer Configuration
# ============================================================================

class TracingConfig:
    """Tracing configuration"""

    def __init__(
        self,
        service_name: str = "fragrance-ai",
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        sample_rate: float = 1.0
    ):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.otlp_endpoint = otlp_endpoint
        self.console_export = console_export
        self.sample_rate = sample_rate


def setup_tracing(config: TracingConfig) -> Optional[trace.Tracer]:
    """
    Setup OpenTelemetry tracing

    Args:
        config: Tracing configuration

    Returns:
        Tracer instance or None if not available
    """
    if not OTEL_AVAILABLE:
        print("OpenTelemetry not available. Tracing disabled.")
        return None

    # Create resource
    resource = Resource.create({
        "service.name": config.service_name,
        "service.version": "1.0.0",
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add exporters
    if config.jaeger_endpoint:
        jaeger_exporter = JaegerExporter(
            agent_host_name=config.jaeger_endpoint.split(":")[0],
            agent_port=int(config.jaeger_endpoint.split(":")[1]) if ":" in config.jaeger_endpoint else 6831,
        )
        provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        print(f"Jaeger tracing enabled: {config.jaeger_endpoint}")

    if config.otlp_endpoint:
        otlp_exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        print(f"OTLP tracing enabled: {config.otlp_endpoint}")

    if config.console_export:
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(console_exporter))
        print("Console tracing enabled")

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer
    tracer = trace.get_tracer(__name__)
    print(f"Tracing initialized for service: {config.service_name}")

    return tracer


# ============================================================================
# Global Tracer
# ============================================================================

_global_tracer: Optional[trace.Tracer] = None


def get_tracer() -> trace.Tracer:
    """Get global tracer"""
    global _global_tracer

    if _global_tracer is None:
        if OTEL_AVAILABLE:
            # Initialize with default config
            config = TracingConfig(console_export=False)  # Disabled by default
            _global_tracer = setup_tracing(config)

        if _global_tracer is None:
            # Return dummy tracer
            _global_tracer = DummyTracer()

    return _global_tracer


def set_tracer(tracer: trace.Tracer):
    """Set global tracer"""
    global _global_tracer
    _global_tracer = tracer


# ============================================================================
# Tracing Decorators
# ============================================================================

def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Decorator to trace a function

    Usage:
        @trace_function(name="process_request", attributes={"user_id": "123"})
        def my_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)

                # Add function info
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Mark as successful
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.OK))

                    return result

                except Exception as e:
                    # Record exception
                    span.record_exception(e)
                    if OTEL_AVAILABLE:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None
):
    """
    Context manager to create a traced span

    Usage:
        with trace_span("database_query", {"table": "users"}):
            ...
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span

            # Mark as successful
            if OTEL_AVAILABLE:
                span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Record exception
            span.record_exception(e)
            if OTEL_AVAILABLE:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


# ============================================================================
# Domain-Specific Tracing
# ============================================================================

class ArtisanTracer:
    """Tracing for Artisan-specific operations"""

    def __init__(self):
        self.tracer = get_tracer()

    @contextmanager
    def trace_evolution(
        self,
        experiment_id: str,
        algorithm: str,
        num_options: int
    ):
        """
        Trace evolution request

        /evolve/options → complete evolution process
        """
        with self.tracer.start_as_current_span("evolution.request") as span:
            span.set_attribute("experiment.id", experiment_id)
            span.set_attribute("evolution.algorithm", algorithm)
            span.set_attribute("evolution.num_options", num_options)
            span.set_attribute("operation", "evolution")

            try:
                yield span
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_feedback(
        self,
        experiment_id: str,
        chosen_id: str,
        rating: int
    ):
        """
        Trace feedback submission

        /evolve/feedback → complete feedback processing
        """
        with self.tracer.start_as_current_span("feedback.submission") as span:
            span.set_attribute("experiment.id", experiment_id)
            span.set_attribute("feedback.chosen_id", chosen_id)
            span.set_attribute("feedback.rating", rating)
            span.set_attribute("operation", "feedback")

            try:
                yield span
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_llm_call(
        self,
        model: str,
        mode: str,
        cache_hit: bool = False
    ):
        """Trace LLM call"""
        with self.tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.mode", mode)
            span.set_attribute("llm.cache_hit", cache_hit)
            span.set_attribute("component", "LLM")

            start_time = time.time()

            try:
                yield span

                # Add latency
                latency_ms = (time.time() - start_time) * 1000
                span.set_attribute("llm.latency_ms", latency_ms)

                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))

            except Exception as e:
                span.record_exception(e)
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_rl_update(
        self,
        algorithm: str,
        iteration: int
    ):
        """Trace RL update"""
        with self.tracer.start_as_current_span("rl.update") as span:
            span.set_attribute("rl.algorithm", algorithm)
            span.set_attribute("rl.iteration", iteration)
            span.set_attribute("component", "RL")

            try:
                yield span
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_ga_generation(
        self,
        generation: int,
        population_size: int
    ):
        """Trace GA generation"""
        with self.tracer.start_as_current_span("ga.generation") as span:
            span.set_attribute("ga.generation", generation)
            span.set_attribute("ga.population_size", population_size)
            span.set_attribute("component", "GA")

            try:
                yield span
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.record_exception(e)
                if OTEL_AVAILABLE:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                raise


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_fastapi_middleware():
    """
    Create FastAPI middleware for tracing

    Usage:
        from fastapi import FastAPI
        from fragrance_ai.tracing import create_fastapi_middleware

        app = FastAPI()
        app.add_middleware(create_fastapi_middleware())
    """
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request

        class TracingMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                tracer = get_tracer()

                # Extract trace context from headers
                if OTEL_AVAILABLE:
                    ctx = TraceContextTextMapPropagator().extract(carrier=request.headers)
                    token = trace.attach(ctx)

                # Create span for request
                with tracer.start_as_current_span(
                    f"{request.method} {request.url.path}"
                ) as span:
                    # Add request attributes
                    span.set_attribute("http.method", request.method)
                    span.set_attribute("http.url", str(request.url))
                    span.set_attribute("http.route", request.url.path)

                    # Process request
                    start_time = time.time()

                    try:
                        response = await call_next(request)

                        # Add response attributes
                        span.set_attribute("http.status_code", response.status_code)

                        # Add latency
                        latency_ms = (time.time() - start_time) * 1000
                        span.set_attribute("http.response_time_ms", latency_ms)

                        # Set status
                        if OTEL_AVAILABLE:
                            if response.status_code >= 500:
                                span.set_status(Status(StatusCode.ERROR))
                            else:
                                span.set_status(Status(StatusCode.OK))

                        return response

                    except Exception as e:
                        span.record_exception(e)
                        if OTEL_AVAILABLE:
                            span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

                    finally:
                        if OTEL_AVAILABLE:
                            trace.detach(token)

        return TracingMiddleware

    except ImportError:
        # Return dummy middleware if starlette not available
        class DummyMiddleware:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                return await self.app(scope, receive, send)

        return DummyMiddleware


# ============================================================================
# Global Instances
# ============================================================================

artisan_tracer = ArtisanTracer()


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Configuration
    'TracingConfig',
    'setup_tracing',

    # Tracer
    'get_tracer',
    'set_tracer',

    # Decorators
    'trace_function',
    'trace_span',

    # Domain-specific
    'ArtisanTracer',
    'artisan_tracer',

    # FastAPI
    'create_fastapi_middleware',

    # Constants
    'OTEL_AVAILABLE',
]
