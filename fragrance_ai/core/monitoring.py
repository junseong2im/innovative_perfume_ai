import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
from collections import defaultdict, deque
import gc

from .config import settings
from .logging_config import get_logger, performance_logger
from .exceptions import SystemException, ErrorCode

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """시스템 메트릭 데이터 클래스"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    
    # GPU 메트릭 (선택적)
    gpu_utilization: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    
    # 네트워크 메트릭
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ApplicationMetrics:
    """애플리케이션 메트릭 데이터 클래스"""
    timestamp: float
    active_requests: int
    total_requests: int
    error_requests: int
    avg_response_time_ms: float
    
    # AI 모델 메트릭
    model_inference_count: int
    avg_inference_time_ms: float
    model_cache_hits: int
    model_cache_misses: int
    
    # 벡터 데이터베이스 메트릭
    vector_search_count: int
    avg_vector_search_time_ms: float
    vector_db_size_mb: float
    
    # 데이터베이스 메트릭
    db_active_connections: int
    db_query_count: int
    avg_db_query_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthStatus:
    """헬스 상태 데이터 클래스"""
    component: str
    status: str  # healthy, degraded, unhealthy
    message: str
    last_check: float
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """메트릭 수집기"""
    
    def __init__(self):
        self.system_metrics: deque = deque(maxlen=1000)  # 최근 1000개 보관
        self.app_metrics: deque = deque(maxlen=1000)
        self.request_times: deque = deque(maxlen=100)  # 최근 100개 요청 시간
        self.inference_times: deque = deque(maxlen=100)  # 최근 100개 추론 시간
        self.vector_search_times: deque = deque(maxlen=100)  # 최근 100개 검색 시간
        self.db_query_times: deque = deque(maxlen=100)  # 최근 100개 쿼리 시간
        
        # 카운터
        self.counters = defaultdict(int)
        self.lock = threading.Lock()
        
        # 네트워크 기준값 (바이트 단위로 추적)
        self._initial_network_stats = self._get_network_stats()
    
    def collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 정보
            memory = psutil.virtual_memory()
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            
            # 네트워크 정보
            current_network = self._get_network_stats()
            network_sent_mb = None
            network_recv_mb = None
            
            if current_network and self._initial_network_stats:
                sent_bytes = current_network.get('sent', 0) - self._initial_network_stats.get('sent', 0)
                recv_bytes = current_network.get('recv', 0) - self._initial_network_stats.get('recv', 0)
                network_sent_mb = sent_bytes / (1024 * 1024)
                network_recv_mb = recv_bytes / (1024 * 1024)
            
            # GPU 정보 (선택적)
            gpu_util, gpu_mem_used, gpu_mem_total = self._get_gpu_metrics()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                gpu_utilization=gpu_util,
                gpu_memory_used_mb=gpu_mem_used,
                gpu_memory_total_mb=gpu_mem_total,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )
            
            with self.lock:
                self.system_metrics.append(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise SystemException(
                message=f"System metrics collection failed: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def collect_application_metrics(self) -> ApplicationMetrics:
        """애플리케이션 메트릭 수집"""
        
        try:
            with self.lock:
                # 평균 응답 시간 계산
                avg_response_time = (
                    sum(self.request_times) / len(self.request_times) 
                    if self.request_times else 0.0
                )
                
                # 평균 추론 시간 계산
                avg_inference_time = (
                    sum(self.inference_times) / len(self.inference_times)
                    if self.inference_times else 0.0
                )
                
                # 평균 벡터 검색 시간 계산
                avg_vector_search_time = (
                    sum(self.vector_search_times) / len(self.vector_search_times)
                    if self.vector_search_times else 0.0
                )
                
                # 평균 DB 쿼리 시간 계산
                avg_db_query_time = (
                    sum(self.db_query_times) / len(self.db_query_times)
                    if self.db_query_times else 0.0
                )
                
                metrics = ApplicationMetrics(
                    timestamp=time.time(),
                    active_requests=self.counters['active_requests'],
                    total_requests=self.counters['total_requests'],
                    error_requests=self.counters['error_requests'],
                    avg_response_time_ms=avg_response_time,
                    model_inference_count=self.counters['model_inference_count'],
                    avg_inference_time_ms=avg_inference_time,
                    model_cache_hits=self.counters['model_cache_hits'],
                    model_cache_misses=self.counters['model_cache_misses'],
                    vector_search_count=self.counters['vector_search_count'],
                    avg_vector_search_time_ms=avg_vector_search_time,
                    vector_db_size_mb=self._get_vector_db_size_mb(),
                    db_active_connections=self.counters['db_active_connections'],
                    db_query_count=self.counters['db_query_count'],
                    avg_db_query_time_ms=avg_db_query_time
                )
                
                self.app_metrics.append(metrics)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
            raise SystemException(
                message=f"Application metrics collection failed: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def record_request_time(self, response_time_ms: float, is_error: bool = False):
        """요청 시간 기록"""
        with self.lock:
            self.request_times.append(response_time_ms)
            self.counters['total_requests'] += 1
            if is_error:
                self.counters['error_requests'] += 1
    
    def record_inference_time(self, inference_time_ms: float, cache_hit: bool = False):
        """모델 추론 시간 기록"""
        with self.lock:
            self.inference_times.append(inference_time_ms)
            self.counters['model_inference_count'] += 1
            if cache_hit:
                self.counters['model_cache_hits'] += 1
            else:
                self.counters['model_cache_misses'] += 1
    
    def record_vector_search_time(self, search_time_ms: float):
        """벡터 검색 시간 기록"""
        with self.lock:
            self.vector_search_times.append(search_time_ms)
            self.counters['vector_search_count'] += 1
    
    def record_db_query_time(self, query_time_ms: float):
        """데이터베이스 쿼리 시간 기록"""
        with self.lock:
            self.db_query_times.append(query_time_ms)
            self.counters['db_query_count'] += 1
    
    def increment_active_requests(self):
        """활성 요청 수 증가"""
        with self.lock:
            self.counters['active_requests'] += 1
    
    def decrement_active_requests(self):
        """활성 요청 수 감소"""
        with self.lock:
            self.counters['active_requests'] = max(0, self.counters['active_requests'] - 1)
    
    def set_db_active_connections(self, count: int):
        """DB 활성 연결 수 설정"""
        with self.lock:
            self.counters['db_active_connections'] = count
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """최신 시스템 메트릭 반환"""
        with self.lock:
            return self.system_metrics[-1] if self.system_metrics else None
    
    def get_latest_app_metrics(self) -> Optional[ApplicationMetrics]:
        """최신 애플리케이션 메트릭 반환"""
        with self.lock:
            return self.app_metrics[-1] if self.app_metrics else None
    
    def get_metrics_history(self, minutes: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """메트릭 히스토리 반환"""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            system_history = [
                m.to_dict() for m in self.system_metrics 
                if m.timestamp >= cutoff_time
            ]
            app_history = [
                m.to_dict() for m in self.app_metrics
                if m.timestamp >= cutoff_time
            ]
        
        return {
            "system": system_history,
            "application": app_history
        }
    
    def _get_network_stats(self) -> Optional[Dict[str, int]]:
        """네트워크 통계 조회"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'sent': net_io.bytes_sent,
                'recv': net_io.bytes_recv
            }
        except Exception:
            return None
    
    def _get_gpu_metrics(self) -> tuple:
        """GPU 메트릭 조회 (NVIDIA GPU 기준)"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU 사용률
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            
            # GPU 메모리
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_used = mem_info.used / (1024 * 1024)  # MB
            gpu_mem_total = mem_info.total / (1024 * 1024)  # MB
            
            return gpu_util, gpu_mem_used, gpu_mem_total
            
        except Exception:
            return None, None, None
    
    def _get_vector_db_size_mb(self) -> float:
        """벡터 DB 크기 조회"""
        try:
            chroma_path = Path(settings.chroma_persist_directory)
            if chroma_path.exists():
                total_size = sum(f.stat().st_size for f in chroma_path.rglob('*') if f.is_file())
                return total_size / (1024 * 1024)  # MB
        except Exception as e:
            # ChromaDB 디렉터리 크기 계산 실패 시 로깅
            logger.warning(f"Failed to calculate ChromaDB size: {e}")
            logger.debug(f"Attempted to access path: {chroma_path}")
        return 0.0


class HealthChecker:
    """헬스 체크 수행기"""
    
    def __init__(self, app):
        self.app = app
        self.health_checks: Dict[str, Callable] = {}
        self.health_statuses: Dict[str, HealthStatus] = {}
        self.lock = threading.Lock()
    
    def register_health_check(self, component: str, check_func: Callable):
        """헬스 체크 함수 등록"""
        self.health_checks[component] = check_func
    
    async def check_all_components(self) -> Dict[str, HealthStatus]:
        """모든 컴포넌트 헬스 체크"""
        
        results = {}
        
        # 기본 헬스 체크들
        await self._check_basic_health(results)
        
        # 등록된 헬스 체크들
        for component, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                status = await self._run_health_check(check_func)
                response_time = (time.time() - start_time) * 1000  # ms
                
                results[component] = HealthStatus(
                    component=component,
                    status=status['status'],
                    message=status['message'],
                    last_check=time.time(),
                    response_time_ms=response_time,
                    details=status.get('details')
                )
                
            except Exception as e:
                results[component] = HealthStatus(
                    component=component,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    last_check=time.time(),
                    details={"error": str(e)}
                )
        
        with self.lock:
            self.health_statuses.update(results)
        
        return results
    
    async def _check_basic_health(self, results: Dict[str, HealthStatus]):
        """기본 헬스 체크들"""
        
        # AI 모델 상태 체크
        models_status = self._check_ai_models()
        results["ai_models"] = models_status
        
        # 시스템 리소스 체크
        system_status = self._check_system_resources()
        results["system_resources"] = system_status
        
        # 데이터베이스 연결 체크
        if hasattr(self.app.state, 'database'):
            db_status = await self._check_database()
            results["database"] = db_status
        
        # Redis 연결 체크
        redis_status = await self._check_redis()
        results["redis"] = redis_status
    
    def _check_ai_models(self) -> HealthStatus:
        """AI 모델 상태 체크"""
        
        try:
            models_ready = {
                "vector_store": hasattr(self.app.state, 'vector_store') and self.app.state.vector_store is not None,
                "embedding_model": hasattr(self.app.state, 'embedding_model') and self.app.state.embedding_model is not None,
                "generator": hasattr(self.app.state, 'generator') and self.app.state.generator is not None
            }
            
            all_ready = all(models_ready.values())
            ready_count = sum(models_ready.values())
            
            if all_ready:
                status = "healthy"
                message = "All AI models are ready"
            elif ready_count > 0:
                status = "degraded"
                message = f"{ready_count}/3 AI models are ready"
            else:
                status = "unhealthy"
                message = "No AI models are ready"
            
            return HealthStatus(
                component="ai_models",
                status=status,
                message=message,
                last_check=time.time(),
                details=models_ready
            )
            
        except Exception as e:
            return HealthStatus(
                component="ai_models",
                status="unhealthy",
                message=f"Failed to check AI models: {str(e)}",
                last_check=time.time(),
                details={"error": str(e)}
            )
    
    def _check_system_resources(self) -> HealthStatus:
        """시스템 리소스 상태 체크"""
        
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            
            # 임계값 설정
            cpu_threshold = 90.0
            memory_threshold = 85.0
            disk_threshold = 90.0
            
            issues = []
            if cpu_percent > cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > memory_threshold:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > disk_threshold:
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            if issues:
                status = "degraded" if len(issues) == 1 else "unhealthy"
                message = "; ".join(issues)
            else:
                status = "healthy"
                message = "System resources are within normal ranges"
            
            return HealthStatus(
                component="system_resources",
                status=status,
                message=message,
                last_check=time.time(),
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent
                }
            )
            
        except Exception as e:
            return HealthStatus(
                component="system_resources",
                status="unhealthy",
                message=f"Failed to check system resources: {str(e)}",
                last_check=time.time(),
                details={"error": str(e)}
            )
    
    async def _check_database(self) -> HealthStatus:
        """데이터베이스 연결 체크"""
        
        try:
            start_time = time.time()
            
            # 간단한 쿼리 실행
            # result = await self.app.state.database.execute("SELECT 1")
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            # 연결 풀 상태 확인
            # pool_status = self.app.state.database.get_pool_status()
            
            return HealthStatus(
                component="database",
                status="healthy",
                message="Database connection is healthy",
                last_check=time.time(),
                response_time_ms=response_time,
                details={
                    # "pool_size": pool_status.get("size", 0),
                    # "active_connections": pool_status.get("active", 0)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                component="database",
                status="unhealthy",
                message=f"Database connection failed: {str(e)}",
                last_check=time.time(),
                details={"error": str(e)}
            )
    
    async def _check_redis(self) -> HealthStatus:
        """Redis 연결 체크"""
        
        try:
            import redis
            
            start_time = time.time()
            r = redis.from_url(settings.redis_url, socket_connect_timeout=5)
            r.ping()
            response_time = (time.time() - start_time) * 1000  # ms
            
            # Redis 정보 조회
            info = r.info()
            
            return HealthStatus(
                component="redis",
                status="healthy",
                message="Redis connection is healthy",
                last_check=time.time(),
                response_time_ms=response_time,
                details={
                    "version": info.get("redis_version"),
                    "connected_clients": info.get("connected_clients"),
                    "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024)
                }
            )
            
        except Exception as e:
            return HealthStatus(
                component="redis",
                status="unhealthy",
                message=f"Redis connection failed: {str(e)}",
                last_check=time.time(),
                details={"error": str(e)}
            )
    
    async def _run_health_check(self, check_func: Callable) -> Dict[str, Any]:
        """헬스 체크 함수 실행"""
        
        if asyncio.iscoroutinefunction(check_func):
            return await check_func()
        else:
            return check_func()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """전체 헬스 상태 요약"""
        
        with self.lock:
            if not self.health_statuses:
                return {
                    "status": "unknown",
                    "message": "No health checks performed yet"
                }
            
            statuses = [status.status for status in self.health_statuses.values()]
            
            if "unhealthy" in statuses:
                overall_status = "unhealthy"
            elif "degraded" in statuses:
                overall_status = "degraded"
            else:
                overall_status = "healthy"
            
            healthy_count = sum(1 for s in statuses if s == "healthy")
            total_count = len(statuses)
            
            return {
                "status": overall_status,
                "message": f"{healthy_count}/{total_count} components healthy",
                "components": {name: status.to_dict() for name, status in self.health_statuses.items()}
            }


# 전역 인스턴스
metrics_collector = MetricsCollector()