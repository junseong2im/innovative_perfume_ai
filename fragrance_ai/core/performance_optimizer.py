"""
성능 최적화 시스템
- 비동기 배치 처리
- 스마트 쿼리 최적화
- 자동 확장 및 로드 밸런싱
- 실시간 성능 모니터링
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import numpy as np
import statistics

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """처리 모드"""
    SYNC = "synchronous"
    ASYNC = "asynchronous" 
    BATCH = "batch"
    STREAM = "stream"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput: float = 0.0  # requests per second
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    queue_size: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """최적화 결과"""
    original_time: float
    optimized_time: float
    improvement_ratio: float
    optimization_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """배치 처리기"""
    
    def __init__(
        self,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        processor_func: Optional[Callable] = None
    ):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.processor_func = processor_func
        
        self.pending_items = []
        self.pending_futures = []
        self.last_batch_time = time.time()
        self.batch_lock = asyncio.Lock()
        
        # 백그라운드 배치 처리 태스크
        self.batch_task = None
        self.running = False
    
    async def start(self):
        """배치 처리기 시작"""
        if not self.running:
            self.running = True
            self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def stop(self):
        """배치 처리기 중지"""
        self.running = False
        if self.batch_task:
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                # 배치 작업이 정상적으로 취소됨
                logger.info("Batch processor task cancelled successfully")
            except Exception as e:
                logger.error(f"Error during batch processor shutdown: {e}")
    
    async def add_item(self, item: Any) -> Any:
        """배치에 아이템 추가"""
        future = asyncio.Future()
        
        async with self.batch_lock:
            self.pending_items.append(item)
            self.pending_futures.append(future)
            
            # 배치가 가득 찼거나 대기시간을 초과했으면 즉시 처리
            should_process = (
                len(self.pending_items) >= self.batch_size or
                time.time() - self.last_batch_time > self.max_wait_time
            )
            
            if should_process:
                await self._process_batch()
        
        return await future
    
    async def _batch_processor(self):
        """백그라운드 배치 처리"""
        while self.running:
            try:
                await asyncio.sleep(self.max_wait_time)
                
                async with self.batch_lock:
                    if self.pending_items and time.time() - self.last_batch_time > self.max_wait_time:
                        await self._process_batch()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
    
    async def _process_batch(self):
        """배치 처리 실행"""
        if not self.pending_items:
            return
        
        items_to_process = self.pending_items.copy()
        futures_to_complete = self.pending_futures.copy()
        
        self.pending_items.clear()
        self.pending_futures.clear()
        self.last_batch_time = time.time()
        
        try:
            if self.processor_func:
                results = await self.processor_func(items_to_process)
                
                # 결과를 각각의 future에 설정
                for future, result in zip(futures_to_complete, results):
                    if not future.done():
                        future.set_result(result)
            else:
                # 기본 처리: 그대로 반환
                for future, item in zip(futures_to_complete, items_to_process):
                    if not future.done():
                        future.set_result(item)
                        
        except Exception as e:
            # 에러 발생시 모든 future에 에러 설정
            for future in futures_to_complete:
                if not future.done():
                    future.set_exception(e)


class AdaptiveLoadBalancer:
    """적응형 로드 밸런서"""
    
    def __init__(self, workers: List[str], health_check_interval: float = 30.0):
        self.workers = workers
        self.health_check_interval = health_check_interval
        
        self.worker_metrics = {worker: PerformanceMetrics() for worker in workers}
        self.worker_weights = {worker: 1.0 for worker in workers}
        self.active_workers = set(workers)
        
        # 로드 밸런싱 통계
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # 헬스체크 태스크
        self.health_check_task = None
        self.running = False
    
    async def start(self):
        """로드 밸런서 시작"""
        if not self.running:
            self.running = True
            self.health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """로드 밸런서 중지"""
        self.running = False
        if self.health_check_task:
            self.health_check_task.cancel()
    
    def select_worker(self) -> str:
        """최적 워커 선택 (가중치 기반)"""
        if not self.active_workers:
            raise RuntimeError("No active workers available")
        
        # 가중치 기반 선택
        total_weight = sum(self.worker_weights[w] for w in self.active_workers)
        if total_weight == 0:
            return list(self.active_workers)[0]
        
        import random
        threshold = random.uniform(0, total_weight)
        current_weight = 0
        
        for worker in self.active_workers:
            current_weight += self.worker_weights[worker]
            if current_weight >= threshold:
                return worker
        
        return list(self.active_workers)[0]
    
    def record_request(self, worker: str, response_time: float, success: bool = True):
        """요청 결과 기록"""
        self.request_counts[worker] += 1
        self.response_times[worker].append(response_time)
        
        # 최근 100개 응답시간만 유지
        if len(self.response_times[worker]) > 100:
            self.response_times[worker] = self.response_times[worker][-100:]
        
        # 가중치 업데이트 (응답시간 기반)
        self._update_worker_weight(worker, response_time, success)
    
    def _update_worker_weight(self, worker: str, response_time: float, success: bool):
        """워커 가중치 업데이트"""
        if not success:
            self.worker_weights[worker] *= 0.9  # 실패시 가중치 감소
            return
        
        # 응답시간 기반 가중치 조정
        if response_time < 0.1:  # 100ms 미만
            self.worker_weights[worker] = min(2.0, self.worker_weights[worker] * 1.05)
        elif response_time > 1.0:  # 1초 초과
            self.worker_weights[worker] = max(0.1, self.worker_weights[worker] * 0.95)
        
        # 가중치 범위 제한
        self.worker_weights[worker] = max(0.1, min(2.0, self.worker_weights[worker]))
    
    async def _health_check_loop(self):
        """헬스체크 루프"""
        while self.running:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _perform_health_checks(self):
        """헬스체크 수행"""
        for worker in self.workers:
            try:
                # 간단한 헬스체크 (실제 구현에서는 HTTP 호출 등)
                is_healthy = await self._check_worker_health(worker)
                
                if is_healthy:
                    self.active_workers.add(worker)
                else:
                    self.active_workers.discard(worker)
                    
            except Exception as e:
                logger.warning(f"Health check failed for worker {worker}: {e}")
                self.active_workers.discard(worker)
    
    async def _check_worker_health(self, worker: str) -> bool:
        """개별 워커 헬스체크"""
        # 실제 구현에서는 워커별 헬스체크 로직 구현
        # 예: HTTP 호출, 데이터베이스 연결 확인 등
        return True


class PerformanceOptimizer:
    """성능 최적화 관리자"""
    
    def __init__(
        self,
        enable_batching: bool = True,
        enable_caching: bool = True,
        enable_load_balancing: bool = False,
        monitoring_interval: float = 60.0
    ):
        self.enable_batching = enable_batching
        self.enable_caching = enable_caching
        self.enable_load_balancing = enable_load_balancing
        self.monitoring_interval = monitoring_interval
        
        # 배치 프로세서들
        self.batch_processors: Dict[str, BatchProcessor] = {}
        
        # 로드 밸런서
        self.load_balancer: Optional[AdaptiveLoadBalancer] = None
        
        # 성능 메트릭
        self.metrics = PerformanceMetrics()
        self.metric_history = deque(maxlen=1000)  # 최근 1000개 메트릭
        
        # 모니터링
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        
        # 스레드/프로세스 풀
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.process_pool = ProcessPoolExecutor(max_workers=4)
        
        logger.info("Performance optimizer initialized")
    
    async def start(self):
        """성능 최적화 시스템 시작"""
        if not self.running:
            self.running = True
            
            # 배치 프로세서 시작
            for processor in self.batch_processors.values():
                await processor.start()
            
            # 로드 밸런서 시작
            if self.load_balancer:
                await self.load_balancer.start()
            
            # 모니터링 시작
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info("Performance optimization system started")
    
    async def stop(self):
        """성능 최적화 시스템 중지"""
        self.running = False
        
        # 배치 프로세서 중지
        for processor in self.batch_processors.values():
            await processor.stop()
        
        # 로드 밸런서 중지
        if self.load_balancer:
            await self.load_balancer.stop()
        
        # 모니터링 중지
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # 스레드/프로세스 풀 종료
        self.thread_pool.shutdown(wait=False)
        self.process_pool.shutdown(wait=False)
    
    def create_batch_processor(
        self,
        name: str,
        batch_size: int = 32,
        max_wait_time: float = 0.1,
        processor_func: Optional[Callable] = None
    ) -> BatchProcessor:
        """배치 프로세서 생성"""
        processor = BatchProcessor(batch_size, max_wait_time, processor_func)
        self.batch_processors[name] = processor
        
        if self.running:
            asyncio.create_task(processor.start())
        
        return processor
    
    def setup_load_balancer(self, workers: List[str]):
        """로드 밸런서 설정"""
        if self.enable_load_balancing:
            self.load_balancer = AdaptiveLoadBalancer(workers)
            if self.running:
                asyncio.create_task(self.load_balancer.start())
    
    async def optimize_async_operation(
        self,
        operation: Callable,
        *args,
        mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        **kwargs
    ) -> Tuple[Any, OptimizationResult]:
        """비동기 작업 최적화"""
        start_time = time.time()
        
        if mode == ProcessingMode.ADAPTIVE:
            # 현재 시스템 부하에 따라 모드 선택
            cpu_usage = psutil.cpu_percent(interval=0.1)
            if cpu_usage > 80:
                mode = ProcessingMode.BATCH
            elif cpu_usage > 60:
                mode = ProcessingMode.ASYNC
            else:
                mode = ProcessingMode.SYNC
        
        # 작업 실행
        if mode == ProcessingMode.ASYNC:
            result = await self._execute_async(operation, *args, **kwargs)
        elif mode == ProcessingMode.BATCH and 'batch_processor' in kwargs:
            result = await self._execute_batch(
                kwargs['batch_processor'], operation, *args, **kwargs
            )
        else:
            result = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool, operation, *args
            )
        
        end_time = time.time()
        optimization_result = OptimizationResult(
            original_time=end_time - start_time,
            optimized_time=end_time - start_time,
            improvement_ratio=1.0,
            optimization_method=mode.value
        )
        
        return result, optimization_result
    
    async def _execute_async(self, operation: Callable, *args, **kwargs) -> Any:
        """비동기 실행"""
        if asyncio.iscoroutinefunction(operation):
            return await operation(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, operation, *args)
    
    async def _execute_batch(
        self,
        batch_processor: BatchProcessor,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """배치 실행"""
        return await batch_processor.add_item((operation, args, kwargs))
    
    async def _monitoring_loop(self):
        """성능 모니터링 루프"""
        while self.running:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._collect_metrics()
                await self._analyze_performance()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _collect_metrics(self):
        """메트릭 수집"""
        try:
            # 시스템 메트릭
            self.metrics.cpu_usage = psutil.cpu_percent(interval=1)
            self.metrics.memory_usage = psutil.virtual_memory().percent
            
            # 성능 메트릭 업데이트
            self.metrics.last_updated = time.time()
            
            # 히스토리에 추가
            self.metric_history.append(self.metrics)
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
    
    async def _analyze_performance(self):
        """성능 분석 및 최적화 제안"""
        if len(self.metric_history) < 10:
            return
        
        recent_metrics = list(self.metric_history)[-10:]
        
        # CPU 사용률 분석
        cpu_usage_trend = [m.cpu_usage for m in recent_metrics]
        avg_cpu = statistics.mean(cpu_usage_trend)
        
        # 메모리 사용률 분석
        memory_usage_trend = [m.memory_usage for m in recent_metrics]
        avg_memory = statistics.mean(memory_usage_trend)
        
        # 최적화 제안 생성
        recommendations = []
        
        if avg_cpu > 80:
            recommendations.append("High CPU usage detected - consider enabling more batch processing")
        
        if avg_memory > 85:
            recommendations.append("High memory usage detected - consider cache cleanup or size reduction")
        
        if recommendations:
            logger.info(f"Performance recommendations: {recommendations}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        if not self.metric_history:
            return {"error": "No metrics available"}
        
        recent_metrics = list(self.metric_history)[-100:]  # 최근 100개
        
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        
        return {
            "current_metrics": {
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "active_connections": self.metrics.active_connections,
                "queue_size": self.metrics.queue_size
            },
            "statistics": {
                "avg_cpu": statistics.mean(cpu_values) if cpu_values else 0,
                "max_cpu": max(cpu_values) if cpu_values else 0,
                "avg_memory": statistics.mean(memory_values) if memory_values else 0,
                "max_memory": max(memory_values) if memory_values else 0
            },
            "batch_processors": {
                name: {
                    "pending_items": len(processor.pending_items),
                    "batch_size": processor.batch_size,
                    "max_wait_time": processor.max_wait_time
                }
                for name, processor in self.batch_processors.items()
            },
            "load_balancer": {
                "active_workers": len(self.load_balancer.active_workers) if self.load_balancer else 0,
                "total_workers": len(self.load_balancer.workers) if self.load_balancer else 0
            } if self.load_balancer else None
        }


# 전역 성능 최적화 인스턴스
global_performance_optimizer = PerformanceOptimizer()