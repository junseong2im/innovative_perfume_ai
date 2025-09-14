"""
고급 캐싱 시스템 구현
- Redis 기반 분산 캐싱
- LRU, LFU 캐시 정책
- 캐시 워밍업 및 무효화 전략
- 메트릭 기반 캐시 최적화
"""

from typing import Dict, Any, Optional, Union, List, Callable, Tuple
import asyncio
import json
import pickle
import time
import hashlib
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor
import redis.asyncio as redis
import numpy as np

logger = logging.getLogger(__name__)


class CachePolicy(Enum):
    """캐시 정책"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # 적응적 정책


@dataclass
class CacheStats:
    """캐시 통계"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    cache_size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def miss_rate(self) -> float:
        return self.misses / self.total_requests if self.total_requests > 0 else 0.0


@dataclass
class CacheEntry:
    """캐시 항목"""
    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[float] = None
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class AdvancedCacheManager:
    """고급 캐시 관리자"""
    
    def __init__(
        self,
        max_size: int = 10000,
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        redis_url: Optional[str] = None,
        enable_metrics: bool = True,
        default_ttl: Optional[float] = 3600,  # 1 hour
        warmup_enabled: bool = True
    ):
        self.max_size = max_size
        self.policy = policy
        self.default_ttl = default_ttl
        self.enable_metrics = enable_metrics
        self.warmup_enabled = warmup_enabled
        
        # Local cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.cache_lock = threading.RLock()
        
        # Redis connection (optional distributed cache)
        self.redis_client = None
        if redis_url:
            self._init_redis(redis_url)
        
        # Statistics
        self.stats = CacheStats(max_size=max_size)
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_task: Optional[asyncio.Task] = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Advanced cache initialized - Policy: {policy}, Max size: {max_size}")
    
    def _init_redis(self, redis_url: str):
        """Redis 초기화"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            logger.info("Redis cache backend initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}")
            self.redis_client = None
    
    def _start_background_tasks(self):
        """백그라운드 작업 시작"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return  # No event loop available
        
        # Cleanup task
        self.cleanup_task = loop.create_task(self._background_cleanup())
        
        # Metrics collection task
        if self.enable_metrics:
            self.metrics_task = loop.create_task(self._background_metrics())
    
    async def get(
        self, 
        key: str, 
        default: Any = None,
        use_redis: bool = True
    ) -> Any:
        """캐시에서 값 조회"""
        start_time = time.time()
        
        try:
            # 1. Local cache 확인
            with self.cache_lock:
                if key in self.cache:
                    entry = self.cache[key]
                    
                    # 만료 확인
                    if entry.is_expired():
                        del self.cache[key]
                        if self.enable_metrics:
                            self.stats.evictions += 1
                    else:
                        # 액세스 정보 업데이트
                        entry.accessed_at = time.time()
                        entry.access_count += 1
                        self._update_access_history(key)
                        
                        # LRU 정책의 경우 맨 끝으로 이동
                        if self.policy == CachePolicy.LRU:
                            self.cache.move_to_end(key)
                        
                        self._record_hit(time.time() - start_time)
                        return entry.value
            
            # 2. Redis cache 확인 (활성화된 경우)
            if self.redis_client and use_redis:
                redis_value = await self._get_from_redis(key)
                if redis_value is not None:
                    # Local cache에 저장
                    await self.set(key, redis_value, use_redis=False)
                    self._record_hit(time.time() - start_time)
                    return redis_value
            
            # 3. Cache miss
            self._record_miss(time.time() - start_time)
            return default
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self._record_miss(time.time() - start_time)
            return default
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        use_redis: bool = True
    ) -> bool:
        """캐시에 값 저장"""
        try:
            ttl = ttl or self.default_ttl
            current_time = time.time()
            
            # 값 크기 계산
            size_bytes = self._calculate_size(value)
            
            # Local cache에 저장
            with self.cache_lock:
                # 공간 확보가 필요한지 확인
                while len(self.cache) >= self.max_size:
                    self._evict_entry()
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=current_time,
                    accessed_at=current_time,
                    access_count=1,
                    ttl=ttl,
                    size_bytes=size_bytes
                )
                
                self.cache[key] = entry
                self._update_access_history(key)
            
            # Redis에도 저장 (활성화된 경우)
            if self.redis_client and use_redis:
                await self._set_to_redis(key, value, ttl)
            
            # 통계 업데이트
            if self.enable_metrics:
                self.stats.cache_size = len(self.cache)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str, use_redis: bool = True) -> bool:
        """캐시에서 값 삭제"""
        try:
            # Local cache에서 삭제
            with self.cache_lock:
                if key in self.cache:
                    del self.cache[key]
                
                if key in self.access_history:
                    del self.access_history[key]
            
            # Redis에서도 삭제
            if self.redis_client and use_redis:
                await self.redis_client.delete(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self, use_redis: bool = True):
        """캐시 전체 삭제"""
        try:
            with self.cache_lock:
                self.cache.clear()
                self.access_history.clear()
            
            if self.redis_client and use_redis:
                await self.redis_client.flushdb()
            
            # 통계 리셋
            self.stats = CacheStats(max_size=self.max_size)
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def _evict_entry(self):
        """캐시 정책에 따른 항목 제거"""
        if not self.cache:
            return
        
        if self.policy == CachePolicy.LRU:
            # 가장 오래전에 사용된 항목 제거
            key, _ = self.cache.popitem(last=False)
        elif self.policy == CachePolicy.LFU:
            # 가장 적게 사용된 항목 제거
            lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[lfu_key]
            key = lfu_key
        elif self.policy == CachePolicy.FIFO:
            # 가장 먼저 들어온 항목 제거
            key, _ = self.cache.popitem(last=False)
        elif self.policy == CachePolicy.ADAPTIVE:
            # 적응적 제거 (액세스 패턴 기반)
            key = self._adaptive_eviction()
        else:
            # 기본값: LRU
            key, _ = self.cache.popitem(last=False)
        
        # 액세스 히스토리도 제거
        if key in self.access_history:
            del self.access_history[key]
        
        if self.enable_metrics:
            self.stats.evictions += 1
    
    def _adaptive_eviction(self) -> str:
        """적응적 제거 알고리즘"""
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            # 점수 계산: 최근성 + 빈도 + 크기 고려
            recency_score = 1.0 / (current_time - entry.accessed_at + 1)
            frequency_score = entry.access_count
            size_penalty = entry.size_bytes / 1024  # KB 단위
            
            # 가중 점수 (낮을수록 제거 대상)
            scores[key] = (recency_score * 0.4 + frequency_score * 0.4) - (size_penalty * 0.2)
        
        # 가장 낮은 점수의 키 반환
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[evict_key]
        return evict_key
    
    def _update_access_history(self, key: str):
        """액세스 히스토리 업데이트"""
        current_time = time.time()
        self.access_history[key].append(current_time)
        
        # 너무 오래된 히스토리는 제거 (최대 100개)
        if len(self.access_history[key]) > 100:
            self.access_history[key] = self.access_history[key][-100:]
    
    async def _get_from_redis(self, key: str) -> Any:
        """Redis에서 값 조회"""
        try:
            redis_data = await self.redis_client.get(key)
            if redis_data:
                return pickle.loads(redis_data)
            return None
        except Exception as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return None
    
    async def _set_to_redis(self, key: str, value: Any, ttl: Optional[float]):
        """Redis에 값 저장"""
        try:
            serialized_value = pickle.dumps(value)
            if ttl:
                await self.redis_client.setex(key, int(ttl), serialized_value)
            else:
                await self.redis_client.set(key, serialized_value)
        except Exception as e:
            logger.warning(f"Redis set error for key {key}: {e}")
    
    def _calculate_size(self, value: Any) -> int:
        """값의 크기 계산 (바이트)"""
        try:
            return len(pickle.dumps(value))
        except:
            # 직렬화 불가능한 경우 추정값 반환
            return len(str(value)) * 2  # 대략적인 추정
    
    def _record_hit(self, response_time: float):
        """Cache hit 기록"""
        if self.enable_metrics:
            self.stats.hits += 1
            self.stats.total_requests += 1
            self._update_avg_response_time(response_time)
    
    def _record_miss(self, response_time: float):
        """Cache miss 기록"""
        if self.enable_metrics:
            self.stats.misses += 1
            self.stats.total_requests += 1
            self._update_avg_response_time(response_time)
    
    def _update_avg_response_time(self, response_time: float):
        """평균 응답시간 업데이트"""
        if self.stats.total_requests == 1:
            self.stats.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.stats.avg_response_time
            )
    
    async def _background_cleanup(self):
        """백그라운드 정리 작업"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다 실행
                
                current_time = time.time()
                expired_keys = []
                
                with self.cache_lock:
                    for key, entry in self.cache.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
                        if key in self.access_history:
                            del self.access_history[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    if self.enable_metrics:
                        self.stats.evictions += len(expired_keys)
                        self.stats.cache_size = len(self.cache)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    async def _background_metrics(self):
        """백그라운드 메트릭 수집"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 실행
                
                # 메트릭 로깅
                logger.info(f"Cache Stats - Hit Rate: {self.stats.hit_rate:.2%}, "
                           f"Size: {len(self.cache)}/{self.max_size}, "
                           f"Avg Response: {self.stats.avg_response_time:.3f}ms")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background metrics error: {e}")
    
    async def warmup(self, warmup_function: Callable[[], Dict[str, Any]]):
        """캐시 워밍업"""
        if not self.warmup_enabled:
            return
        
        try:
            logger.info("Starting cache warmup...")
            start_time = time.time()
            
            # 워밍업 데이터 생성
            loop = asyncio.get_event_loop()
            warmup_data = await loop.run_in_executor(self.executor, warmup_function)
            
            # 배치로 캐시에 저장
            tasks = []
            for key, value in warmup_data.items():
                tasks.append(self.set(key, value))
            
            await asyncio.gather(*tasks)
            
            warmup_time = time.time() - start_time
            logger.info(f"Cache warmup completed - {len(warmup_data)} entries in {warmup_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Cache warmup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        stats_dict = asdict(self.stats)
        stats_dict['cache_size'] = len(self.cache)
        return stats_dict
    
    def get_hot_keys(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """가장 자주 액세스되는 키들"""
        key_counts = [(key, entry.access_count) for key, entry in self.cache.items()]
        return sorted(key_counts, key=lambda x: x[1], reverse=True)[:top_k]
    
    def __del__(self):
        """리소스 정리"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


class FragranceCacheManager(AdvancedCacheManager):
    """향수 도메인 특화 캐시 매니저"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.embedding_cache_namespace = "embeddings:"
        self.recipe_cache_namespace = "recipes:"
        self.search_cache_namespace = "search:"
    
    async def cache_embedding(self, text: str, embedding: np.ndarray, model_type: str = "default"):
        """임베딩 캐시"""
        cache_key = f"{self.embedding_cache_namespace}{model_type}:{self._hash_text(text)}"
        await self.set(cache_key, embedding.tobytes(), ttl=86400)  # 24시간
    
    async def get_cached_embedding(self, text: str, model_type: str = "default") -> Optional[np.ndarray]:
        """캐시된 임베딩 조회"""
        cache_key = f"{self.embedding_cache_namespace}{model_type}:{self._hash_text(text)}"
        cached_bytes = await self.get(cache_key)
        
        if cached_bytes:
            return np.frombuffer(cached_bytes, dtype=np.float32)
        return None
    
    async def cache_recipe(self, recipe_id: str, recipe_data: Dict[str, Any]):
        """레시피 캐시"""
        cache_key = f"{self.recipe_cache_namespace}{recipe_id}"
        await self.set(cache_key, recipe_data, ttl=3600)  # 1시간
    
    async def cache_search_result(self, query: str, results: List[Dict[str, Any]]):
        """검색 결과 캐시"""
        cache_key = f"{self.search_cache_namespace}{self._hash_text(query)}"
        await self.set(cache_key, results, ttl=1800)  # 30분
    
    def _hash_text(self, text: str) -> str:
        """텍스트 해시 생성"""
        return hashlib.md5(text.encode()).hexdigest()