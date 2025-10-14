"""
Health Check System - LLM 모델 헬스체크

각 모델(Qwen/Mistral/Llama)의 상태를 체크:
- 로딩 상태
- 가용성
- 메모리 사용량
"""

import psutil
import time
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """모델 상태"""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


class HealthStatus(Enum):
    """헬스 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ModelHealthInfo:
    """모델 헬스 정보"""
    model_name: str
    status: ModelStatus
    available: bool
    memory_mb: float
    memory_percent: float
    load_time_ms: Optional[float] = None
    last_inference_ms: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['status'] = self.status.value
        return data


@dataclass
class SystemHealthInfo:
    """시스템 전체 헬스 정보"""
    health_status: HealthStatus
    models: Dict[str, ModelHealthInfo]
    system_memory_percent: float
    system_cpu_percent: float
    available_models: int
    total_models: int
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "health_status": self.health_status.value,
            "models": {name: info.to_dict() for name, info in self.models.items()},
            "system_memory_percent": self.system_memory_percent,
            "system_cpu_percent": self.system_cpu_percent,
            "available_models": self.available_models,
            "total_models": self.total_models,
            "timestamp": self.timestamp
        }


class HealthChecker:
    """
    헬스체커

    각 LLM 모델의 상태를 모니터링하고 헬스체크 결과를 반환합니다.
    """

    def __init__(self):
        self.models: Dict[str, Any] = {}  # model_name -> model object
        self.model_health: Dict[str, ModelHealthInfo] = {}
        self.memory_threshold_mb = 8000  # 8GB
        self.memory_threshold_percent = 80.0  # 80%

    def register_model(self, model_name: str, model: Any):
        """
        모델 등록

        Args:
            model_name: qwen, mistral, llama
            model: 모델 객체
        """
        self.models[model_name] = model
        logger.info(f"Registered model: {model_name}")

    def check_model_health(self, model_name: str) -> ModelHealthInfo:
        """
        특정 모델의 헬스 체크

        Args:
            model_name: qwen, mistral, llama

        Returns:
            ModelHealthInfo
        """
        if model_name not in self.models:
            return ModelHealthInfo(
                model_name=model_name,
                status=ModelStatus.UNAVAILABLE,
                available=False,
                memory_mb=0,
                memory_percent=0,
                error_message=f"Model {model_name} not registered"
            )

        model = self.models[model_name]

        try:
            # Check model status
            status = self._get_model_status(model)

            # Check memory usage
            memory_info = self._get_model_memory_usage(model)

            # Check availability
            available = (
                status == ModelStatus.READY and
                memory_info['memory_mb'] < self.memory_threshold_mb and
                memory_info['memory_percent'] < self.memory_threshold_percent
            )

            health_info = ModelHealthInfo(
                model_name=model_name,
                status=status,
                available=available,
                memory_mb=memory_info['memory_mb'],
                memory_percent=memory_info['memory_percent']
            )

            # Cache health info
            self.model_health[model_name] = health_info

            return health_info

        except Exception as e:
            logger.error(f"Health check failed for {model_name}: {e}")
            return ModelHealthInfo(
                model_name=model_name,
                status=ModelStatus.ERROR,
                available=False,
                memory_mb=0,
                memory_percent=0,
                error_message=str(e)
            )

    def check_all_models(self) -> SystemHealthInfo:
        """
        모든 모델 헬스 체크

        Returns:
            SystemHealthInfo
        """
        model_healths = {}
        available_count = 0

        for model_name in self.models.keys():
            health = self.check_model_health(model_name)
            model_healths[model_name] = health
            if health.available:
                available_count += 1

        # System-wide metrics
        system_memory = psutil.virtual_memory()
        system_cpu = psutil.cpu_percent(interval=0.1)

        # Determine overall health status
        if available_count == len(self.models):
            health_status = HealthStatus.HEALTHY
        elif available_count > 0:
            health_status = HealthStatus.DEGRADED
        else:
            health_status = HealthStatus.UNHEALTHY

        return SystemHealthInfo(
            health_status=health_status,
            models=model_healths,
            system_memory_percent=system_memory.percent,
            system_cpu_percent=system_cpu,
            available_models=available_count,
            total_models=len(self.models)
        )

    def _get_model_status(self, model: Any) -> ModelStatus:
        """
        모델 상태 확인

        Args:
            model: 모델 객체

        Returns:
            ModelStatus
        """
        # Check if model has status attribute
        if hasattr(model, 'status'):
            status_str = str(model.status).lower()
            if 'loading' in status_str:
                return ModelStatus.LOADING
            elif 'ready' in status_str or 'loaded' in status_str:
                return ModelStatus.READY
            elif 'error' in status_str:
                return ModelStatus.ERROR

        # Check if model is callable (simple test)
        if callable(model):
            return ModelStatus.READY

        # Check if model has generate method
        if hasattr(model, 'generate'):
            return ModelStatus.READY

        # Default: assume ready if no issues
        return ModelStatus.READY

    def _get_model_memory_usage(self, model: Any) -> Dict[str, float]:
        """
        모델 메모리 사용량 확인

        Args:
            model: 모델 객체

        Returns:
            {'memory_mb': float, 'memory_percent': float}
        """
        # Get process memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

        # Get system memory
        system_memory = psutil.virtual_memory()
        memory_percent = (memory_mb / (system_memory.total / (1024 * 1024))) * 100

        return {
            'memory_mb': memory_mb,
            'memory_percent': memory_percent
        }

    def get_health_summary(self) -> str:
        """
        헬스 상태 요약 문자열

        Returns:
            요약 문자열
        """
        system_health = self.check_all_models()

        summary_lines = [
            f"Health Status: {system_health.health_status.value.upper()}",
            f"Available Models: {system_health.available_models}/{system_health.total_models}",
            f"System Memory: {system_health.system_memory_percent:.1f}%",
            f"System CPU: {system_health.system_cpu_percent:.1f}%",
            "",
            "Models:"
        ]

        for name, health in system_health.models.items():
            status_icon = "[OK]" if health.available else "[X]"
            summary_lines.append(
                f"  {status_icon} {name}: {health.status.value} "
                f"(Memory: {health.memory_mb:.0f}MB, {health.memory_percent:.1f}%)"
            )

        return "\n".join(summary_lines)


# =============================================================================
# Mock Model for Testing
# =============================================================================

class MockLLMModel:
    """테스트용 Mock LLM 모델"""

    def __init__(self, model_name: str, simulate_error: bool = False):
        self.model_name = model_name
        self.status = "ready"
        self.simulate_error = simulate_error

    def generate(self, prompt: str) -> str:
        """생성 메서드"""
        if self.simulate_error:
            raise RuntimeError(f"{self.model_name} generation failed")
        return f"Generated by {self.model_name}"


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create health checker
    health_checker = HealthChecker()

    # Register models
    qwen_model = MockLLMModel("qwen")
    mistral_model = MockLLMModel("mistral")
    llama_model = MockLLMModel("llama", simulate_error=True)  # Simulate error

    health_checker.register_model("qwen", qwen_model)
    health_checker.register_model("mistral", mistral_model)
    health_checker.register_model("llama", llama_model)

    # Check individual model
    print("=== Individual Model Health Check ===")
    qwen_health = health_checker.check_model_health("qwen")
    print(f"Qwen: {qwen_health.to_dict()}\n")

    # Check all models
    print("=== All Models Health Check ===")
    system_health = health_checker.check_all_models()
    print(system_health.to_dict())
    print()

    # Health summary
    print("=== Health Summary ===")
    print(health_checker.get_health_summary())
