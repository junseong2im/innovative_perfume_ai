"""
Health Check API Endpoints

/health/llm?model=qwen|mistral|llama
→ 로딩/가용/메모리 OK 체크
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any
from fragrance_ai.guards.health_check import HealthChecker, ModelStatus, HealthStatus
import logging

logger = logging.getLogger(__name__)

# Create router
health_router = APIRouter(prefix="/health", tags=["health"])

# Global health checker (should be initialized with models)
health_checker = HealthChecker()


def init_health_checker(checker: HealthChecker):
    """
    헬스체커 초기화 (앱 시작 시 호출)

    Args:
        checker: HealthChecker 인스턴스
    """
    global health_checker
    health_checker = checker


@health_router.get("")
async def health_check_basic() -> Dict[str, Any]:
    """
    기본 헬스체크

    Returns:
        {"status": "healthy"}
    """
    return {
        "status": "healthy",
        "service": "fragrance-ai",
        "version": "1.0.0"
    }


@health_router.get("/llm")
async def health_check_llm(
    model: Optional[str] = Query(None, description="Model name: qwen, mistral, llama")
) -> Dict[str, Any]:
    """
    LLM 모델 헬스체크

    Args:
        model: 모델 이름 (optional)
            - qwen: Qwen 2.5-7B
            - mistral: Mistral 7B
            - llama: Llama 3-8B
            - None: 모든 모델

    Returns:
        {
            "model": "qwen",
            "status": "ready",
            "available": true,
            "memory_mb": 5120.5,
            "memory_percent": 12.5
        }

        또는 (모든 모델)

        {
            "health_status": "healthy",
            "models": {...},
            "system_memory_percent": 45.2,
            "system_cpu_percent": 20.5,
            "available_models": 3,
            "total_models": 3
        }
    """
    if model:
        # Check specific model
        if model not in ["qwen", "mistral", "llama"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model name. Must be one of: qwen, mistral, llama"
            )

        health_info = health_checker.check_model_health(model)

        return health_info.to_dict()

    else:
        # Check all models
        system_health = health_checker.check_all_models()

        return system_health.to_dict()


@health_router.get("/llm/summary")
async def health_check_llm_summary() -> Dict[str, Any]:
    """
    LLM 헬스 요약

    Returns:
        {
            "summary": "Health Status: HEALTHY\\n...",
            "health_status": "healthy",
            "available_models": 3,
            "total_models": 3
        }
    """
    summary_text = health_checker.get_health_summary()
    system_health = health_checker.check_all_models()

    return {
        "summary": summary_text,
        "health_status": system_health.health_status.value,
        "available_models": system_health.available_models,
        "total_models": system_health.total_models
    }


@health_router.get("/readiness")
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness 체크 (Kubernetes용)

    모든 모델이 사용 가능한 경우에만 ready

    Returns:
        {"ready": true/false}
    """
    system_health = health_checker.check_all_models()

    ready = (
        system_health.health_status == HealthStatus.HEALTHY and
        system_health.available_models == system_health.total_models
    )

    if not ready:
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {system_health.available_models}/{system_health.total_models} models available"
        )

    return {
        "ready": True,
        "available_models": system_health.available_models,
        "total_models": system_health.total_models
    }


@health_router.get("/liveness")
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness 체크 (Kubernetes용)

    서비스가 살아있는지만 체크 (모델 상태 무관)

    Returns:
        {"alive": true}
    """
    return {
        "alive": True,
        "service": "fragrance-ai"
    }


# =============================================================================
# Usage in FastAPI App
# =============================================================================

"""
# app/main.py

from fastapi import FastAPI
from fragrance_ai.api.health_endpoints import health_router, init_health_checker
from fragrance_ai.guards.health_check import HealthChecker

app = FastAPI(title="Fragrance AI API")

# Initialize health checker
health_checker = HealthChecker()

# Register models (after loading)
health_checker.register_model("qwen", qwen_model)
health_checker.register_model("mistral", mistral_model)
health_checker.register_model("llama", llama_model)

# Initialize health check endpoints
init_health_checker(health_checker)

# Include health router
app.include_router(health_router)

# Endpoints:
# GET /health              - Basic health check
# GET /health/llm          - All models health
# GET /health/llm?model=qwen - Specific model health
# GET /health/llm/summary  - Health summary
# GET /health/readiness    - Kubernetes readiness
# GET /health/liveness     - Kubernetes liveness
"""
