# fragrance_ai/llm/health_check.py
"""
LLM Health Check System
모델 로딩/추론 가능 여부/메모리 사용량 체크
"""

import time
import psutil
import logging
from typing import Dict, Optional, Literal, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Health Status
# ============================================================================

class HealthStatus(str, Enum):
    """Health status codes"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ModelHealth:
    """Health information for a model"""
    model_name: str
    status: HealthStatus
    loaded: bool
    inference_ready: bool
    last_inference_ms: Optional[float]
    memory_mb: Optional[float]
    gpu_memory_mb: Optional[float]
    error_message: Optional[str]
    checked_at: str
    uptime_seconds: Optional[float]


# ============================================================================
# Health Checker
# ============================================================================

class LLMHealthChecker:
    """Health checker for LLM models"""

    def __init__(self):
        self.model_cache: Dict[str, Any] = {}
        self.model_load_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register_model(self, model_name: str, model_instance: Any):
        """
        Register a model instance for health checking

        Args:
            model_name: Model identifier
            model_instance: Loaded model instance
        """
        with self._lock:
            self.model_cache[model_name] = model_instance
            self.model_load_times[model_name] = time.time()
            logger.info(f"Registered model for health check: {model_name}")

    def unregister_model(self, model_name: str):
        """Unregister a model"""
        with self._lock:
            if model_name in self.model_cache:
                del self.model_cache[model_name]
            if model_name in self.model_load_times:
                del self.model_load_times[model_name]
            logger.info(f"Unregistered model: {model_name}")

    def check_model_health(
        self,
        model_name: Literal["qwen", "mistral", "llama"],
        run_inference: bool = False
    ) -> ModelHealth:
        """
        Check health of a specific model

        Args:
            model_name: Model to check
            run_inference: Whether to run test inference

        Returns:
            ModelHealth
        """
        from datetime import datetime

        checked_at = datetime.utcnow().isoformat()

        # Check if model is registered
        with self._lock:
            model_instance = self.model_cache.get(model_name)
            load_time = self.model_load_times.get(model_name)

        if model_instance is None:
            return ModelHealth(
                model_name=model_name,
                status=HealthStatus.UNHEALTHY,
                loaded=False,
                inference_ready=False,
                last_inference_ms=None,
                memory_mb=None,
                gpu_memory_mb=None,
                error_message=f"Model {model_name} not loaded",
                checked_at=checked_at,
                uptime_seconds=None
            )

        # Calculate uptime
        uptime = time.time() - load_time if load_time else None

        # Get memory usage
        memory_mb = self._get_process_memory_mb()
        gpu_memory_mb = self._get_gpu_memory_mb() if self._is_gpu_available() else None

        # Test inference if requested
        inference_ready = True
        last_inference_ms = None
        error_message = None

        if run_inference:
            try:
                start = time.time()
                success = self._test_inference(model_name, model_instance)
                last_inference_ms = (time.time() - start) * 1000

                if not success:
                    inference_ready = False
                    error_message = "Inference test failed"
            except Exception as e:
                inference_ready = False
                error_message = f"Inference error: {str(e)}"
                logger.error(f"Inference test failed for {model_name}: {e}")

        # Determine overall status
        if not inference_ready:
            status = HealthStatus.UNHEALTHY
        elif memory_mb and memory_mb > 30000:  # >30GB memory usage
            status = HealthStatus.DEGRADED
        elif last_inference_ms and last_inference_ms > 5000:  # >5s inference
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return ModelHealth(
            model_name=model_name,
            status=status,
            loaded=True,
            inference_ready=inference_ready,
            last_inference_ms=last_inference_ms,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            error_message=error_message,
            checked_at=checked_at,
            uptime_seconds=uptime
        )

    def check_all_models(self, run_inference: bool = False) -> Dict[str, ModelHealth]:
        """
        Check health of all models

        Args:
            run_inference: Whether to run test inference

        Returns:
            Dictionary of {model_name: ModelHealth}
        """
        results = {}

        for model_name in ["qwen", "mistral", "llama"]:
            results[model_name] = self.check_model_health(
                model_name=model_name,
                run_inference=run_inference
            )

        return results

    def get_summary_status(self) -> Dict[str, Any]:
        """
        Get overall health summary

        Returns:
            Summary status dictionary
        """
        all_health = self.check_all_models(run_inference=False)

        # Count statuses
        healthy_count = sum(1 for h in all_health.values() if h.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for h in all_health.values() if h.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for h in all_health.values() if h.status == HealthStatus.UNHEALTHY)

        # Overall status
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "healthy_count": healthy_count,
            "degraded_count": degraded_count,
            "unhealthy_count": unhealthy_count,
            "total_models": len(all_health),
            "models": {name: health.status.value for name, health in all_health.items()},
            "memory_mb": self._get_process_memory_mb(),
            "gpu_available": self._is_gpu_available()
        }

    def _test_inference(self, model_name: str, model_instance: Any) -> bool:
        """
        Test inference with a simple input

        Args:
            model_name: Model identifier
            model_instance: Model instance

        Returns:
            True if inference successful
        """
        test_input = "test perfume"

        try:
            if model_name == "qwen":
                from .qwen_client import infer_brief_qwen
                result = infer_brief_qwen(test_input)
                return result is not None

            elif model_name == "mistral":
                # Mistral is validation-only, just check it's loaded
                return model_instance is not None

            elif model_name == "llama":
                from .llama_hints import generate_creative_hints
                result = generate_creative_hints(test_input)
                return isinstance(result, list)

            return False

        except Exception as e:
            logger.error(f"Inference test failed for {model_name}: {e}")
            return False

    @staticmethod
    def _get_process_memory_mb() -> float:
        """Get current process memory usage in MB"""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return 0.0

    @staticmethod
    def _get_gpu_memory_mb() -> Optional[float]:
        """Get GPU memory usage in MB"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
            return None
        except Exception:
            return None

    @staticmethod
    def _is_gpu_available() -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


# ============================================================================
# Global Health Checker Instance
# ============================================================================

_health_checker: Optional[LLMHealthChecker] = None


def get_health_checker() -> LLMHealthChecker:
    """Get global health checker instance"""
    global _health_checker
    if _health_checker is None:
        _health_checker = LLMHealthChecker()
    return _health_checker


# ============================================================================
# Readiness and Liveness Probes
# ============================================================================

def check_readiness() -> bool:
    """
    Kubernetes readiness probe
    Returns True if system is ready to accept traffic
    """
    checker = get_health_checker()
    summary = checker.get_summary_status()

    # Ready if at least one model is healthy
    return summary["healthy_count"] > 0


def check_liveness() -> bool:
    """
    Kubernetes liveness probe
    Returns True if system is alive (not deadlocked)
    """
    try:
        checker = get_health_checker()
        # Just check if we can get status (not deadlocked)
        _ = checker.get_summary_status()
        return True
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return False


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'HealthStatus',
    'ModelHealth',
    'LLMHealthChecker',
    'get_health_checker',
    'check_readiness',
    'check_liveness'
]
