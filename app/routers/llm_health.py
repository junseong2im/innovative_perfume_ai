# app/routers/llm_health.py
"""
LLM Health Check API Endpoints
/health/llm?qwen|mistral|llama → 로딩/추론 가능 여부/메모리 사용량
"""

from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Literal, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

from fragrance_ai.llm.health_check import (
    get_health_checker,
    check_readiness,
    check_liveness,
    HealthStatus
)
from fragrance_ai.llm.graceful_reload import get_reload_manager
from fragrance_ai.llm.metrics import get_metrics, get_metrics_dict, PROMETHEUS_AVAILABLE

router = APIRouter(prefix="/health", tags=["Health"])


# ============================================================================
# Response Models
# ============================================================================

class ModelHealthResponse(BaseModel):
    """Health status for a single model"""
    model_name: str
    status: str
    loaded: bool
    inference_ready: bool
    last_inference_ms: Optional[float]
    memory_mb: Optional[float]
    gpu_memory_mb: Optional[float]
    error_message: Optional[str]
    checked_at: str
    uptime_seconds: Optional[float]


class AllModelsHealthResponse(BaseModel):
    """Health status for all models"""
    status: str
    healthy_count: int
    degraded_count: int
    unhealthy_count: int
    total_models: int
    models: Dict[str, str]
    memory_mb: float
    gpu_available: bool
    checked_at: str


# ============================================================================
# Health Check Endpoints
# ============================================================================

@router.get("/llm", response_model=ModelHealthResponse)
async def llm_health_check(
    model: Literal["qwen", "mistral", "llama"] = Query(..., description="Model to check"),
    run_inference: bool = Query(False, description="Run test inference")
):
    """
    Check health of specific LLM model

    **Query Parameters:**
    - `model`: Model to check (qwen, mistral, llama)
    - `run_inference`: Whether to run test inference (slower but thorough)

    **Response:**
    - `status`: health | degraded | unhealthy | unknown
    - `loaded`: Whether model is loaded in memory
    - `inference_ready`: Whether model can perform inference
    - `last_inference_ms`: Last inference time in milliseconds
    - `memory_mb`: Memory usage in MB
    - `gpu_memory_mb`: GPU memory usage in MB (if available)
    - `uptime_seconds`: Time since model was loaded

    **Status Codes:**
    - `200`: Model is healthy or degraded
    - `503`: Model is unhealthy
    """
    try:
        checker = get_health_checker()
        health = checker.check_model_health(model, run_inference=run_inference)

        # Convert to response model
        response = ModelHealthResponse(
            model_name=health.model_name,
            status=health.status.value,
            loaded=health.loaded,
            inference_ready=health.inference_ready,
            last_inference_ms=health.last_inference_ms,
            memory_mb=health.memory_mb,
            gpu_memory_mb=health.gpu_memory_mb,
            error_message=health.error_message,
            checked_at=health.checked_at,
            uptime_seconds=health.uptime_seconds
        )

        # Return 503 if unhealthy
        if health.status == HealthStatus.UNHEALTHY:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump()
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/llm/all", response_model=AllModelsHealthResponse)
async def all_llm_health_check(
    run_inference: bool = Query(False, description="Run test inference on all models")
):
    """
    Check health of all LLM models

    **Query Parameters:**
    - `run_inference`: Whether to run test inference (slower but thorough)

    **Response:**
    - `status`: Overall health status
    - `healthy_count`: Number of healthy models
    - `degraded_count`: Number of degraded models
    - `unhealthy_count`: Number of unhealthy models
    - `models`: Dictionary of {model_name: status}
    - `memory_mb`: Total process memory usage
    - `gpu_available`: Whether GPU is available

    **Status Codes:**
    - `200`: At least one model is healthy
    - `503`: All models are unhealthy
    """
    try:
        checker = get_health_checker()

        # Get individual model health
        if run_inference:
            all_health = checker.check_all_models(run_inference=True)
        else:
            summary = checker.get_summary_status()
            all_health = {
                model: checker.check_model_health(model, run_inference=False)
                for model in ["qwen", "mistral", "llama"]
            }

        # Calculate summary
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

        response = AllModelsHealthResponse(
            status=overall_status.value,
            healthy_count=healthy_count,
            degraded_count=degraded_count,
            unhealthy_count=unhealthy_count,
            total_models=len(all_health),
            models={name: health.status.value for name, health in all_health.items()},
            memory_mb=all_health["qwen"].memory_mb or 0.0,
            gpu_available=all_health["qwen"].gpu_memory_mb is not None,
            checked_at=datetime.utcnow().isoformat()
        )

        # Return 503 if all unhealthy
        if healthy_count == 0:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.model_dump()
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/llm/reload/{model}")
async def get_reload_status(model: Literal["qwen", "mistral", "llama"]):
    """
    Get reload status for a model

    **Path Parameters:**
    - `model`: Model name (qwen, mistral, llama)

    **Response:**
    - `state`: idle | warming_up | ready | switching | completed | failed
    - `progress_percent`: Progress percentage (0-100)
    - `message`: Status message
    - `started_at`: Reload start timestamp
    - `completed_at`: Reload completion timestamp (if completed)
    - `error`: Error message (if failed)
    """
    try:
        manager = get_reload_manager()
        reload_status = manager.get_reload_status(model)

        if reload_status is None:
            return {
                "model": model,
                "state": "idle",
                "progress_percent": 0.0,
                "message": "No reload in progress"
            }

        return {
            "model": model,
            "state": reload_status.state.value,
            "progress_percent": reload_status.progress_percent,
            "message": reload_status.message,
            "started_at": reload_status.started_at,
            "completed_at": reload_status.completed_at,
            "error": reload_status.error
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get reload status: {str(e)}"
        )


# ============================================================================
# Kubernetes Probes
# ============================================================================

@router.get("/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe

    Returns 200 if system is ready to accept traffic,
    503 if not ready (e.g., models still loading)
    """
    try:
        if check_readiness():
            return {"status": "ready"}
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "No models ready"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "error": str(e)}
        )


@router.get("/live")
async def liveness_probe():
    """
    Kubernetes liveness probe

    Returns 200 if system is alive (not deadlocked),
    503 if system is deadlocked or unresponsive
    """
    try:
        if check_liveness():
            return {"status": "alive"}
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "dead"}
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "error", "error": str(e)}
        )


# ============================================================================
# Metrics Endpoint
# ============================================================================

@router.get("/metrics")
async def llm_metrics():
    """
    Get Prometheus metrics for LLM ensemble

    Returns metrics in Prometheus text format
    """
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Prometheus client not installed"
        )

    from fastapi.responses import Response

    metrics_data = get_metrics()
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4"
    )


@router.get("/metrics/json")
async def llm_metrics_json():
    """
    Get metrics as JSON (for debugging)

    Returns metrics as JSON dictionary
    """
    return get_metrics_dict()


# ============================================================================
# Export
# ============================================================================

__all__ = ['router']
