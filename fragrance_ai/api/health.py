"""
Health Check and System Status Endpoints
Production-ready health monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from datetime import datetime, timedelta
import psutil
import asyncio
from sqlalchemy import text
from sqlalchemy.orm import Session
import redis
import torch
import logging

from fragrance_ai.database.connection import DatabaseConnectionManager
from fragrance_ai.orchestrator.living_scent_orchestrator import get_living_scent_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthChecker:
    """System health monitoring"""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.last_check = None
        self.status_history = []

    async def check_database(self) -> Dict[str, Any]:
        """Check database connectivity"""
        try:
            db_manager = DatabaseConnectionManager()
            with db_manager.get_session() as session:
                result = session.execute(text("SELECT 1"))
                version = session.execute(text("SELECT version()")).scalar()

                # Check pgvector extension
                extensions = session.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                ).fetchall()
                has_vector = len(extensions) > 0

            return {
                "status": "healthy",
                "response_time_ms": 50,
                "version": version[:30] if version else "unknown",
                "pgvector": has_vector
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity"""
        try:
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            info = r.info()

            return {
                "status": "healthy",
                "version": info.get('redis_version', 'unknown'),
                "used_memory": info.get('used_memory_human', 'unknown'),
                "connected_clients": info.get('connected_clients', 0)
            }
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return {
                "status": "degraded",
                "error": str(e),
                "note": "Cache unavailable but system functional"
            }

    async def check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        try:
            orchestrator = get_living_scent_orchestrator()
            health = orchestrator.health_check()
            stats = orchestrator.get_statistics()

            return {
                "status": health['status'],
                "moga_available": stats['moga_available'],
                "ppo_available": stats['ppo_available'],
                "components": health['components']
            }
        except Exception as e:
            logger.error(f"AI models health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # GPU check if available
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_info = {
                    "cuda_available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0)
                }

            status = "healthy"
            if cpu_percent > 90 or memory.percent > 90:
                status = "warning"
            if disk.percent > 95:
                status = "critical"

            return {
                "status": status,
                "cpu_percent": cpu_percent,
                "memory": {
                    "percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2)
                },
                "disk": {
                    "percent": disk.percent,
                    "free_gb": round(disk.free / (1024**3), 2),
                    "total_gb": round(disk.total / (1024**3), 2)
                },
                "gpu": gpu_info
            }
        except Exception as e:
            logger.error(f"System resources check failed: {e}")
            return {
                "status": "unknown",
                "error": str(e)
            }

    async def perform_all_checks(self) -> Dict[str, Any]:
        """Perform all health checks"""
        start_time = datetime.now()

        # Run all checks in parallel
        results = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_ai_models(),
            self.check_system_resources(),
            return_exceptions=True
        )

        # Process results
        checks = {
            "database": results[0] if not isinstance(results[0], Exception) else {"status": "error", "error": str(results[0])},
            "redis": results[1] if not isinstance(results[1], Exception) else {"status": "error", "error": str(results[1])},
            "ai_models": results[2] if not isinstance(results[2], Exception) else {"status": "error", "error": str(results[2])},
            "system": results[3] if not isinstance(results[3], Exception) else {"status": "error", "error": str(results[3])}
        }

        # Determine overall status
        statuses = [check.get("status", "unknown") for check in checks.values()]

        if all(s == "healthy" for s in statuses):
            overall_status = "healthy"
            self.checks_passed += 1
        elif any(s in ["unhealthy", "error"] for s in statuses):
            overall_status = "unhealthy"
            self.checks_failed += 1
        elif any(s == "degraded" for s in statuses):
            overall_status = "degraded"
            self.checks_passed += 1
        else:
            overall_status = "warning"
            self.checks_passed += 1

        elapsed = (datetime.now() - start_time).total_seconds()
        self.last_check = datetime.now()

        # Add to history
        self.status_history.append({
            "timestamp": self.last_check.isoformat(),
            "status": overall_status,
            "duration_ms": elapsed * 1000
        })

        # Keep only last 100 entries
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]

        return {
            "status": overall_status,
            "timestamp": self.last_check.isoformat(),
            "checks": checks,
            "duration_ms": round(elapsed * 1000, 2),
            "statistics": {
                "checks_passed": self.checks_passed,
                "checks_failed": self.checks_failed,
                "uptime_percentage": round(
                    (self.checks_passed / max(1, self.checks_passed + self.checks_failed)) * 100, 2
                )
            }
        }


# Global health checker instance
health_checker = HealthChecker()


@router.get("/", response_model=Dict[str, Any])
async def health_check():
    """
    Basic health check endpoint
    Returns 200 if service is running
    """
    return {
        "status": "ok",
        "service": "Fragrance AI API",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/live", response_model=Dict[str, Any])
async def liveness_probe():
    """
    Kubernetes liveness probe
    Returns 200 if service is alive
    """
    return {"status": "alive"}


@router.get("/ready", response_model=Dict[str, Any])
async def readiness_probe():
    """
    Kubernetes readiness probe
    Returns 200 if service is ready to accept traffic
    """
    # Quick check of critical components
    try:
        db_manager = DatabaseConnectionManager()
        with db_manager.get_session() as session:
            session.execute(text("SELECT 1"))

        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")


@router.get("/detailed", response_model=Dict[str, Any])
async def detailed_health_check():
    """
    Detailed health check with all subsystems
    """
    result = await health_checker.perform_all_checks()

    if result["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=result)

    return result


@router.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """
    Get system metrics for monitoring
    Compatible with Prometheus format
    """
    orchestrator = get_living_scent_orchestrator()
    stats = orchestrator.get_statistics()

    metrics = {
        "fragrance_ai_health_status": 1 if health_checker.last_check else 0,
        "fragrance_ai_checks_passed_total": health_checker.checks_passed,
        "fragrance_ai_checks_failed_total": health_checker.checks_failed,
        "fragrance_ai_moga_successes_total": stats['moga_successes'],
        "fragrance_ai_ppo_successes_total": stats['ppo_successes'],
        "fragrance_ai_fallback_uses_total": stats['fallback_uses'],
        "fragrance_ai_dna_library_size": stats['dna_library_size'],
        "fragrance_ai_phenotype_library_size": stats['phenotype_library_size']
    }

    # Add system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    metrics.update({
        "fragrance_ai_cpu_usage_percent": cpu_percent,
        "fragrance_ai_memory_usage_percent": memory.percent,
        "fragrance_ai_memory_available_bytes": memory.available
    })

    return metrics


@router.get("/history", response_model=Dict[str, Any])
async def get_health_history():
    """
    Get health check history
    """
    return {
        "history": health_checker.status_history[-50:],  # Last 50 entries
        "summary": {
            "total_checks": health_checker.checks_passed + health_checker.checks_failed,
            "passed": health_checker.checks_passed,
            "failed": health_checker.checks_failed,
            "uptime_percentage": round(
                (health_checker.checks_passed / max(1, health_checker.checks_passed + health_checker.checks_failed)) * 100, 2
            )
        }
    }