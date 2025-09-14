from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import time
import asyncio
from datetime import datetime, timedelta

from ...core.monitoring import metrics_collector, HealthChecker
from ...core.config import settings
from ...core.logging_config import get_logger, performance_logger
from ...core.exceptions import SystemException, ErrorCode, AuthenticationException
from ..dependencies import verify_api_key

logger = get_logger(__name__)
router = APIRouter()
security = HTTPBearer()


@router.get("/health")
async def basic_health_check():
    """기본 헬스 체크 엔드포인트 (인증 불필요)"""
    
    try:
        # 기본 상태만 체크
        current_time = time.time()
        
        # 메모리 사용량 간단 체크
        import psutil
        memory = psutil.virtual_memory()
        
        if memory.percent > 95:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Critical memory usage",
                    "timestamp": current_time
                }
            )
        
        return {
            "status": "healthy",
            "message": "Service is running",
            "timestamp": current_time,
            "version": settings.app_version
        }
        
    except Exception as e:
        logger.error(f"Basic health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Health check failed",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/health/detailed")
async def detailed_health_check(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """상세 헬스 체크 (인증 필요)"""
    
    try:
        # API 키 검증
        verify_api_key(credentials.credentials)
        
        # 애플리케이션에서 HealthChecker 인스턴스 가져오기
        from ...api.main import app
        health_checker = HealthChecker(app)
        
        # 모든 컴포넌트 헬스 체크
        health_results = await health_checker.check_all_components()
        
        # 전체 상태 요약
        overall_health = health_checker.get_overall_health()
        
        # 성능 메트릭 포함
        performance_logger.log_execution_time(
            operation="detailed_health_check",
            execution_time=sum(
                result.response_time_ms or 0 
                for result in health_results.values()
            ),
            success=overall_health["status"] != "unhealthy"
        )
        
        return {
            "overall": overall_health,
            "components": {name: result.to_dict() for name, result in health_results.items()},
            "timestamp": time.time()
        }
        
    except AuthenticationException:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/health/readiness")
async def readiness_check():
    """준비 상태 체크 (Kubernetes 등에서 사용)"""
    
    try:
        from ...api.main import app
        
        # 핵심 컴포넌트들이 준비되었는지 확인
        required_components = [
            hasattr(app.state, 'vector_store') and app.state.vector_store is not None,
            hasattr(app.state, 'embedding_model') and app.state.embedding_model is not None,
            hasattr(app.state, 'generator') and app.state.generator is not None
        ]
        
        if not all(required_components):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "message": "Required components not initialized",
                    "timestamp": time.time()
                }
            )
        
        return {
            "status": "ready",
            "message": "All required components are ready",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Readiness check failed",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/health/liveness")
async def liveness_check():
    """생존 상태 체크 (Kubernetes 등에서 사용)"""
    
    try:
        # 프로세스가 살아있고 기본 응답이 가능한지 확인
        return {
            "status": "alive",
            "message": "Service is alive",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - (
                getattr(liveness_check, '_start_time', time.time())
            )
        }
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "dead",
                "message": "Liveness check failed",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/metrics")
async def get_system_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    history_minutes: int = Query(default=10, ge=1, le=60)
):
    """시스템 메트릭 조회"""
    
    try:
        # API 키 검증
        verify_api_key(credentials.credentials)
        
        # 현재 메트릭 수집
        system_metrics = metrics_collector.collect_system_metrics()
        app_metrics = metrics_collector.collect_application_metrics()
        
        # 히스토리 데이터
        history = metrics_collector.get_metrics_history(history_minutes)
        
        return {
            "current": {
                "system": system_metrics.to_dict(),
                "application": app_metrics.to_dict()
            },
            "history": history,
            "collection_time": time.time()
        }
        
    except AuthenticationException:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except SystemException as e:
        raise HTTPException(status_code=500, detail=e.user_message)
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/metrics/summary")
async def get_metrics_summary(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """메트릭 요약 정보"""
    
    try:
        # API 키 검증
        verify_api_key(credentials.credentials)
        
        # 최신 메트릭
        latest_system = metrics_collector.get_latest_system_metrics()
        latest_app = metrics_collector.get_latest_app_metrics()
        
        if not latest_system or not latest_app:
            raise HTTPException(
                status_code=503,
                detail="Metrics not available yet"
            )
        
        # 간단한 요약 생성
        summary = {
            "system": {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "disk_usage_percent": latest_system.disk_usage_percent,
                "status": _get_system_status(latest_system)
            },
            "application": {
                "active_requests": latest_app.active_requests,
                "total_requests": latest_app.total_requests,
                "error_rate": (
                    latest_app.error_requests / max(latest_app.total_requests, 1)
                ) * 100,
                "avg_response_time_ms": latest_app.avg_response_time_ms,
                "status": _get_app_status(latest_app)
            },
            "timestamp": time.time()
        }
        
        return summary
        
    except AuthenticationException:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics summary")


@router.post("/metrics/reset")
async def reset_metrics(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """메트릭 카운터 리셋 (관리자용)"""
    
    try:
        # API 키 검증
        verify_api_key(credentials.credentials)
        
        # 카운터들 리셋
        with metrics_collector.lock:
            metrics_collector.counters.clear()
            metrics_collector.request_times.clear()
            metrics_collector.inference_times.clear()
            metrics_collector.vector_search_times.clear()
            metrics_collector.db_query_times.clear()
        
        logger.info("Metrics counters reset by admin")
        
        return {
            "status": "success",
            "message": "Metrics counters have been reset",
            "timestamp": time.time()
        }
        
    except AuthenticationException:
        raise HTTPException(status_code=401, detail="Invalid API key")
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset metrics")


@router.get("/status")
async def get_service_status():
    """전체 서비스 상태 요약 (공개)"""
    
    try:
        # 기본 정보
        status_info = {
            "service": settings.app_name,
            "version": settings.app_version,
            "timestamp": time.time(),
            "status": "operational"
        }
        
        # 간단한 헬스 체크
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # 임계치 체크
            if cpu_percent > 90 or memory.percent > 90:
                status_info["status"] = "degraded"
            
            status_info["system"] = {
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%"
            }
            
        except Exception as e:
            logger.warning(f"Failed to get system status: {e}")
            status_info["status"] = "unknown"
        
        return status_info
        
    except Exception as e:
        logger.error(f"Failed to get service status: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Failed to retrieve service status",
                "timestamp": time.time()
            }
        )


def _get_system_status(metrics) -> str:
    """시스템 메트릭 기반 상태 판단"""
    
    if metrics.cpu_percent > 90 or metrics.memory_percent > 90 or metrics.disk_usage_percent > 95:
        return "critical"
    elif metrics.cpu_percent > 70 or metrics.memory_percent > 80 or metrics.disk_usage_percent > 90:
        return "warning"
    else:
        return "healthy"


def _get_app_status(metrics) -> str:
    """애플리케이션 메트릭 기반 상태 판단"""
    
    error_rate = (metrics.error_requests / max(metrics.total_requests, 1)) * 100
    
    if error_rate > 10 or metrics.avg_response_time_ms > 5000:
        return "critical"
    elif error_rate > 5 or metrics.avg_response_time_ms > 2000:
        return "warning"
    else:
        return "healthy"


# 시작 시간 기록 (liveness check용)
liveness_check._start_time = time.time()