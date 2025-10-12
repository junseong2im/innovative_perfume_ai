# fragrance_ai/observability.py
"""
Observability module for JSON logging and metrics collection
Provides structured logging and Prometheus metrics
"""

import json
import logging
import time
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from functools import wraps
import hashlib
from enum import Enum

# Prometheus imports (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ============================================================================
# Security - Log Masking and PII Protection
# ============================================================================

class LogMasker:
    """Mask sensitive information in logs"""

    # Patterns for sensitive data
    API_KEY_PATTERNS = [
        r'(?i)(api[_-]?key|apikey|api[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
        r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{20,})',
        r'(?i)(token\s*[:=]\s*)["\']?([a-zA-Z0-9_\-\.]{20,})["\']?',
    ]

    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'

    # Credit card patterns
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'

    # AWS/Cloud keys
    AWS_ACCESS_KEY = r'(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*["\']?([A-Z0-9]{20})["\']?'
    AWS_SECRET_KEY = r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'

    # Database URLs
    DB_URL_PATTERN = r'(?i)(postgres|mysql|mongodb):\/\/([^:]+):([^@]+)@([^\/]+)'

    @classmethod
    def mask_api_keys(cls, text: str) -> str:
        """Mask API keys and tokens"""
        for pattern in cls.API_KEY_PATTERNS:
            text = re.sub(pattern, r'\1=***MASKED***', text)

        # AWS keys
        text = re.sub(cls.AWS_ACCESS_KEY, r'\1=***MASKED***', text)
        text = re.sub(cls.AWS_SECRET_KEY, r'\1=***MASKED***', text)

        return text

    @classmethod
    def mask_pii(cls, text: str) -> str:
        """Mask personally identifiable information"""
        # Email addresses
        text = re.sub(cls.EMAIL_PATTERN, '***EMAIL_MASKED***', text)

        # Phone numbers
        text = re.sub(cls.PHONE_PATTERN, '***PHONE_MASKED***', text)

        # Credit cards
        text = re.sub(cls.CREDIT_CARD_PATTERN, '***CARD_MASKED***', text)

        return text

    @classmethod
    def mask_db_credentials(cls, text: str) -> str:
        """Mask database credentials in URLs"""
        return re.sub(cls.DB_URL_PATTERN, r'\1://***USER***:***PASS***@\4', text)

    @classmethod
    def hash_user_id(cls, user_id: str) -> str:
        """Hash user ID for privacy"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    @classmethod
    def anonymize_user_id(cls, user_id: str) -> str:
        """Anonymize user ID (same as hash for consistency)"""
        return cls.hash_user_id(user_id)

    @classmethod
    def mask_all(cls, text: str) -> str:
        """Apply all masking rules"""
        text = cls.mask_api_keys(text)
        text = cls.mask_pii(text)
        text = cls.mask_db_credentials(text)
        return text

    @classmethod
    def sanitize_log_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize entire log data dictionary"""
        sanitized = {}

        for key, value in data.items():
            # Hash user identifiers
            if key in ('user_id', 'userId', 'user'):
                sanitized[f"{key}_hash"] = cls.hash_user_id(str(value))
                continue

            # Mask sensitive keys
            if key in ('api_key', 'apiKey', 'token', 'password', 'secret', 'auth'):
                sanitized[key] = '***MASKED***'
                continue

            # Handle nested dictionaries
            if isinstance(value, dict):
                sanitized[key] = cls.sanitize_log_data(value)
            # Handle strings
            elif isinstance(value, str):
                sanitized[key] = cls.mask_all(value)
            # Other types pass through
            else:
                sanitized[key] = value

        return sanitized


# ============================================================================
# JSON Logger Configuration
# ============================================================================

class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging with security masking"""

    def __init__(self, enable_masking: bool = True):
        super().__init__()
        self.enable_masking = enable_masking

    def format(self, record):
        # Get base message
        message = record.getMessage()

        # Apply masking to message
        if self.enable_masking:
            message = LogMasker.mask_all(message)

        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            extra_fields = record.extra_fields
            # Sanitize extra fields
            if self.enable_masking:
                extra_fields = LogMasker.sanitize_log_data(extra_fields)
            log_obj.update(extra_fields)

        # Add exception info if present
        if record.exc_info:
            exception_msg = self.formatException(record.exc_info)
            if self.enable_masking:
                exception_msg = LogMasker.mask_all(exception_msg)
            log_obj["exception"] = exception_msg

        return json.dumps(log_obj, ensure_ascii=False)


class ObservabilityLogger:
    """Enhanced logger with JSON output and metrics"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # Add JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)

    def log(self, level: str, message: str, **kwargs):
        """Log with extra fields"""
        extra_fields = kwargs
        self.logger.log(
            getattr(logging, level.upper()),
            message,
            extra={'extra_fields': extra_fields}
        )

    def debug(self, message: str, **kwargs):
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        self.log(LogLevel.CRITICAL, message, **kwargs)


# ============================================================================
# GA (MOGA) Logging
# ============================================================================

class GALogger:
    """Logger for Genetic Algorithm operations"""

    def __init__(self):
        self.logger = ObservabilityLogger("fragrance_ai.ga")

    def log_generation(
        self,
        generation: int,
        population_size: int,
        violation_rate: float,
        novelty: float,
        cost_norm: float,
        f_total: float,
        pareto_size: int,
        **kwargs
    ):
        """Log GA generation statistics"""
        self.logger.info(
            "GA generation completed",
            component="GA",
            generation=generation,
            population_size=population_size,
            violation_rate=round(violation_rate, 4),
            novelty=round(novelty, 4),
            cost_norm=round(cost_norm, 2),
            f_total=round(f_total, 4),
            pareto_size=pareto_size,
            **kwargs
        )

    def log_mutation(
        self,
        mutation_type: str,
        before_sum: float,
        after_sum: float,
        negatives_found: int,
        ifra_violations: int
    ):
        """Log mutation operation"""
        self.logger.debug(
            "Mutation applied",
            component="GA",
            mutation_type=mutation_type,
            before_sum=round(before_sum, 4),
            after_sum=round(after_sum, 4),
            negatives_found=negatives_found,
            ifra_violations=ifra_violations
        )

    def log_crossover(
        self,
        crossover_type: str,
        parent1_fitness: float,
        parent2_fitness: float,
        child_fitness: float
    ):
        """Log crossover operation"""
        self.logger.debug(
            "Crossover performed",
            component="GA",
            crossover_type=crossover_type,
            parent1_fitness=round(parent1_fitness, 4),
            parent2_fitness=round(parent2_fitness, 4),
            child_fitness=round(child_fitness, 4)
        )


# ============================================================================
# RL Logging
# ============================================================================

class RLLogger:
    """Logger for Reinforcement Learning operations"""

    def __init__(self):
        self.logger = ObservabilityLogger("fragrance_ai.rl")

    def log_update(
        self,
        algorithm: str,
        loss: float,
        reward: float,
        entropy: Optional[float] = None,
        accept_prob: Optional[float] = None,
        clip_frac: Optional[float] = None,
        value_loss: Optional[float] = None,
        policy_loss: Optional[float] = None,
        **kwargs
    ):
        """Log RL update metrics"""
        metrics = {
            "component": "RL",
            "algorithm": algorithm,
            "loss": round(loss, 6),
            "reward": round(reward, 4),
        }

        if entropy is not None:
            metrics["entropy"] = round(entropy, 4)
        if accept_prob is not None:
            metrics["accept_prob"] = round(accept_prob, 4)
        if clip_frac is not None:
            metrics["clip_frac"] = round(clip_frac, 4)
        if value_loss is not None:
            metrics["value_loss"] = round(value_loss, 6)
        if policy_loss is not None:
            metrics["policy_loss"] = round(policy_loss, 6)

        metrics.update(kwargs)

        self.logger.info("RL update completed", **metrics)

    def log_action_selection(
        self,
        state_norm: float,
        action: int,
        log_prob: float,
        value: Optional[float] = None
    ):
        """Log action selection"""
        self.logger.debug(
            "Action selected",
            component="RL",
            state_norm=round(state_norm, 4),
            action=action,
            log_prob=round(log_prob, 4),
            value=round(value, 4) if value else None
        )

    def log_rollout(
        self,
        episode_length: int,
        total_reward: float,
        avg_reward: float,
        advantage_mean: Optional[float] = None,
        advantage_std: Optional[float] = None
    ):
        """Log rollout statistics"""
        self.logger.info(
            "Rollout completed",
            component="RL",
            episode_length=episode_length,
            total_reward=round(total_reward, 4),
            avg_reward=round(avg_reward, 4),
            advantage_mean=round(advantage_mean, 4) if advantage_mean else None,
            advantage_std=round(advantage_std, 4) if advantage_std else None
        )


# ============================================================================
# Orchestrator Logging
# ============================================================================

class OrchestratorLogger:
    """Logger for orchestration layer"""

    def __init__(self):
        self.logger = ObservabilityLogger("fragrance_ai.orchestrator")

    def log_experiment(
        self,
        experiment_id: str,
        user_id: str,
        action: str,
        timing_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log experiment action with timing"""
        # Hash user_id for privacy
        user_id_hash = hashlib.sha256(user_id.encode()).hexdigest()[:16]

        self.logger.info(
            f"Experiment {action}",
            component="Orchestrator",
            experiment_id=experiment_id,
            user_id_hash=user_id_hash,
            action=action,
            timing_ms=round(timing_ms, 2),
            success=success,
            error=error,
            **kwargs
        )

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        user_agent: Optional[str] = None
    ):
        """Log API request"""
        self.logger.info(
            "API request",
            component="Orchestrator",
            method=method,
            path=path,
            status_code=status_code,
            response_time_ms=round(response_time_ms, 2),
            user_agent=user_agent
        )

    def log_dna_creation(
        self,
        dna_id: str,
        ingredient_count: int,
        ifra_compliant: bool,
        total_cost: Optional[float] = None,
        timing_ms: float = 0
    ):
        """Log DNA creation"""
        self.logger.info(
            "DNA created",
            component="Orchestrator",
            dna_id=dna_id,
            ingredient_count=ingredient_count,
            ifra_compliant=ifra_compliant,
            total_cost=round(total_cost, 2) if total_cost else None,
            timing_ms=round(timing_ms, 2)
        )


# ============================================================================
# Timing Decorator
# ============================================================================

def log_timing(logger: ObservabilityLogger, action: str):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                timing_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"{action} completed",
                    action=action,
                    function=func.__name__,
                    timing_ms=round(timing_ms, 2),
                    success=success,
                    error=error
                )
            return result
        return wrapper
    return decorator


# ============================================================================
# Metrics Collection (Prometheus)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    # Create registry
    registry = CollectorRegistry()

    # GA Metrics
    ga_generation_counter = Counter(
        'fragrance_ga_generations_total',
        'Total number of GA generations',
        registry=registry
    )

    ga_violation_rate = Gauge(
        'fragrance_ga_violation_rate',
        'Current IFRA violation rate',
        registry=registry
    )

    ga_fitness_histogram = Histogram(
        'fragrance_ga_fitness',
        'GA fitness distribution',
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        registry=registry
    )

    # RL Metrics
    rl_updates_counter = Counter(
        'fragrance_rl_updates_total',
        'Total number of RL updates',
        ['algorithm'],
        registry=registry
    )

    rl_reward_gauge = Gauge(
        'fragrance_rl_reward',
        'Current average reward',
        ['algorithm'],
        registry=registry
    )

    rl_loss_histogram = Histogram(
        'fragrance_rl_loss',
        'RL loss distribution',
        ['algorithm'],
        buckets=(0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        registry=registry
    )

    # API Metrics
    api_requests_counter = Counter(
        'fragrance_api_requests_total',
        'Total API requests',
        ['method', 'endpoint', 'status'],
        registry=registry
    )

    api_response_time = Histogram(
        'fragrance_api_response_seconds',
        'API response time',
        ['method', 'endpoint'],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
        registry=registry
    )

    # Experiment Metrics
    experiment_counter = Counter(
        'fragrance_experiments_total',
        'Total experiments created',
        registry=registry
    )

    experiment_duration = Summary(
        'fragrance_experiment_duration_seconds',
        'Experiment duration',
        registry=registry
    )


class MetricsCollector:
    """Collect and expose metrics"""

    def __init__(self):
        self.enabled = PROMETHEUS_AVAILABLE

    def record_ga_generation(
        self,
        violation_rate: float,
        fitness: float
    ):
        """Record GA generation metrics"""
        if self.enabled:
            ga_generation_counter.inc()
            ga_violation_rate.set(violation_rate)
            ga_fitness_histogram.observe(fitness)

    def record_rl_update(
        self,
        algorithm: str,
        loss: float,
        reward: float
    ):
        """Record RL update metrics"""
        if self.enabled:
            rl_updates_counter.labels(algorithm=algorithm).inc()
            rl_reward_gauge.labels(algorithm=algorithm).set(reward)
            rl_loss_histogram.labels(algorithm=algorithm).observe(loss)

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        response_time: float
    ):
        """Record API request metrics"""
        if self.enabled:
            api_requests_counter.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            api_response_time.labels(
                method=method,
                endpoint=endpoint
            ).observe(response_time)

    def record_experiment(self, duration_seconds: float):
        """Record experiment metrics"""
        if self.enabled:
            experiment_counter.inc()
            experiment_duration.observe(duration_seconds)

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        if self.enabled:
            return generate_latest(registry)
        return b"# Prometheus not available"


# ============================================================================
# Global Instances
# ============================================================================

# Create global logger instances
ga_logger = GALogger()
rl_logger = RLLogger()
orchestrator_logger = OrchestratorLogger()
metrics_collector = MetricsCollector()


# ============================================================================
# Convenience Functions
# ============================================================================

def get_logger(component: str) -> ObservabilityLogger:
    """Get logger for component"""
    return ObservabilityLogger(f"fragrance_ai.{component}")


def log_json(component: str, message: str, **kwargs):
    """Quick JSON log"""
    logger = get_logger(component)
    logger.info(message, component=component, **kwargs)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Security
    'LogMasker',

    # Loggers
    'ObservabilityLogger',
    'GALogger',
    'RLLogger',
    'OrchestratorLogger',

    # Global instances
    'ga_logger',
    'rl_logger',
    'orchestrator_logger',
    'metrics_collector',

    # Utilities
    'get_logger',
    'log_json',
    'log_timing',

    # Metrics
    'MetricsCollector',
]