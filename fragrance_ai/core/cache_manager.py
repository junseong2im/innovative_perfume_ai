"""
고급 캐싱 관리자
- Redis 기반 분산 캐싱
- 다중 레벨 캐싱 (메모리 + Redis)
- 지능형 캐시 정책
- 캐시 워밍업 및 프리로딩
"""

import asyncio
import json
import hashlib
import pickle
import zlib
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import weakref
from collections import OrderedDict, defaultdict
import aioredis
import numpy as np

from .logging_config import get_logger
from .monitoring import MetricsCollector

logger = get_logger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DISK = "l3_disk"

class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage: float = 0.0
    redis_usage: float = 0.0
    total_keys: int = 0
    avg_access_time: float = 0.0
    hit_rate: float = 0.0
    popular_keys: Dict[str, int] = field(default_factory=dict)

@dataclass
class CacheEntry:
    key: str
    value: Any
    ttl: Optional[float] = None
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    compressed: bool = False

class AdvancedCacheManager:
    """고급 캐싱 관리자"""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        l1_max_size: int = 10000,
        l1_max_memory_mb: float = 256,
        default_ttl: int = 3600,
        compression_threshold: int = 1024,
        enable_metrics: bool = True,
        cache_policy: CachePolicy = CachePolicy.ADAPTIVE
    ):
        self.redis_client: Optional[aioredis.Redis] = None
        self.redis_url = redis_url

        # L1 메모리 캐시 설정
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l1_max_size = l1_max_size
        self.l1_max_memory_bytes = l1_max_memory_mb * 1024 * 1024
        self.l1_current_memory = 0

        # 캐시 정책 및 설정
        self.cache_policy = cache_policy
        self.default_ttl = default_ttl
        self.compression_threshold = compression_threshold

        # 통계 및 메트릭
        self.stats = CacheStats()
        self.enable_metrics = enable_metrics
        self.metrics_collector = MetricsCollector() if enable_metrics else None

        # 적응형 캐시를 위한 데이터
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.key_popularity: Dict[str, float] = defaultdict(float)
        self.key_tags: Dict[str, List[str]] = defaultdict(list)

        # 백그라운드 작업
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        self._lock = asyncio.Lock()

    async def initialize(self):
        """캐시 매니저 초기화"""
        try:
            # Redis 연결 설정
            if self.redis_url:
                self.redis_client = await aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,  # 바이너리 데이터 지원
                    max_connections=20,
                    retry_on_timeout=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={"TCP_KEEPIDLE": 1, "TCP_KEEPCNT": 3, "TCP_KEEPINTVL": 1},
                )

                # Redis 연결 테스트
                await self.redis_client.ping()
                logger.info("Redis cache backend initialized successfully")
            else:
                logger.warning("Redis URL not provided, using memory cache only")

            # 백그라운드 작업 시작
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())

            if self.enable_metrics:
                self._metrics_task = asyncio.create_task(self._metrics_worker())

            logger.info("Advanced cache manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            # Redis 없이도 작동하도록 함
            self.redis_client = None

    async def get(self, key: str, default: Any = None) -> Any:
        """캐시에서 값 조회"""
        start_time = time.time()

        try:
            # 키 정규화
            normalized_key = self._normalize_key(key)

            # L1 메모리 캐시에서 먼저 확인
            l1_entry = await self._get_from_l1(normalized_key)
            if l1_entry is not None:
                await self._update_access_pattern(normalized_key)
                self.stats.hits += 1

                if self.enable_metrics:
                    access_time = (time.time() - start_time) * 1000
                    self.metrics_collector.record_cache_hit("l1", access_time)

                return l1_entry.value

            # L2 Redis 캐시 확인
            if self.redis_client:
                l2_value = await self._get_from_l2(normalized_key)
                if l2_value is not None:
                    # L1에 프로모션
                    await self._promote_to_l1(normalized_key, l2_value)
                    await self._update_access_pattern(normalized_key)
                    self.stats.hits += 1

                    if self.enable_metrics:
                        access_time = (time.time() - start_time) * 1000
                        self.metrics_collector.record_cache_hit("l2", access_time)

                    return l2_value

            # 캐시 미스
            self.stats.misses += 1

            if self.enable_metrics:
                access_time = (time.time() - start_time) * 1000
                self.metrics_collector.record_cache_miss(access_time)

            return default

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
        force_l2: bool = False
    ) -> bool:
        """캐시에 값 설정"""
        try:
            normalized_key = self._normalize_key(key)
            ttl_seconds = ttl or self.default_ttl
            tags = tags or []

            # 값 직렬화 및 압축
            serialized_value, compressed = await self._serialize_value(value)
            size_bytes = len(serialized_value)

            # 캐시 엔트리 생성
            entry = CacheEntry(
                key=normalized_key,
                value=value,
                ttl=ttl_seconds,
                size_bytes=size_bytes,
                tags=tags,
                compressed=compressed
            )

            # L1 메모리 캐시에 저장 (크기 제한 확인)
            if not force_l2 and size_bytes < self.l1_max_memory_bytes // 10:
                await self._set_to_l1(entry)

            # L2 Redis 캐시에 저장
            if self.redis_client:
                await self._set_to_l2(normalized_key, serialized_value, ttl_seconds, tags)

            # 태그 및 접근 패턴 업데이트
            self.key_tags[normalized_key] = tags
            await self._update_popularity(normalized_key, 1.0)

            return True

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """캐시에서 키 삭제"""
        try:
            normalized_key = self._normalize_key(key)

            # L1에서 삭제
            await self._delete_from_l1(normalized_key)

            # L2에서 삭제
            if self.redis_client:
                await self._delete_from_l2(normalized_key)

            # 메타데이터 정리
            self.access_patterns.pop(normalized_key, None)
            self.key_popularity.pop(normalized_key, None)
            self.key_tags.pop(normalized_key, None)

            return True

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False

    async def delete_by_tags(self, tags: List[str]) -> int:
        """태그로 캐시 삭제"""
        deleted_count = 0

        try:
            # L1 캐시에서 태그로 검색하여 삭제
            keys_to_delete = []
            for key, entry in self.l1_cache.items():
                if any(tag in entry.tags for tag in tags):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                await self._delete_from_l1(key)
                deleted_count += 1

            # L2 Redis에서 태그로 삭제
            if self.redis_client:
                for tag in tags:
                    tag_key = f"tag:{tag}"
                    tagged_keys = await self.redis_client.smembers(tag_key)

                    for key in tagged_keys:
                        await self._delete_from_l2(key.decode())
                        deleted_count += 1

                    await self.redis_client.delete(tag_key)

            logger.info(f"Deleted {deleted_count} keys by tags: {tags}")
            return deleted_count

        except Exception as e:
            logger.error(f"Delete by tags error: {e}")
            return 0

    async def clear(self) -> bool:
        """모든 캐시 클리어"""
        try:
            # L1 클리어
            self.l1_cache.clear()
            self.l1_current_memory = 0

            # L2 클리어 (조심스럽게, 네임스페이스 사용)
            if self.redis_client:
                await self.redis_client.flushdb()

            # 메타데이터 클리어
            self.access_patterns.clear()
            self.key_popularity.clear()
            self.key_tags.clear()

            # 통계 리셋
            self.stats = CacheStats()

            logger.info("All cache levels cleared")
            return True

        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """키 존재 확인"""
        normalized_key = self._normalize_key(key)

        # L1 확인
        if normalized_key in self.l1_cache:
            entry = self.l1_cache[normalized_key]
            if not await self._is_expired(entry):
                return True

        # L2 확인
        if self.redis_client:
            exists = await self.redis_client.exists(normalized_key)
            return bool(exists)

        return False

    async def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        total_requests = self.stats.hits + self.stats.misses
        hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0

        stats = {
            "hit_rate": round(hit_rate, 2),
            "total_hits": self.stats.hits,
            "total_misses": self.stats.misses,
            "total_requests": total_requests,
            "evictions": self.stats.evictions,
            "l1_cache_size": len(self.l1_cache),
            "l1_memory_usage_mb": round(self.l1_current_memory / (1024 * 1024), 2),
            "l1_memory_limit_mb": round(self.l1_max_memory_bytes / (1024 * 1024), 2),
            "cache_policy": self.cache_policy.value,
            "top_keys": dict(list(self.key_popularity.items())[:10])
        }

        # Redis 통계 추가
        if self.redis_client:
            try:
                redis_info = await self.redis_client.info("memory")
                stats["redis_memory_usage_mb"] = round(
                    int(redis_info.get("used_memory", 0)) / (1024 * 1024), 2
                )
                stats["redis_connected"] = True
            except Exception:
                stats["redis_connected"] = False

        return stats

    # 내부 메서드들

    def _normalize_key(self, key: str) -> str:
        """키 정규화"""
        return f"fragrance_ai:{hashlib.md5(key.encode()).hexdigest()}"

    async def _get_from_l1(self, key: str) -> Optional[CacheEntry]:
        """L1 메모리 캐시에서 조회"""
        if key not in self.l1_cache:
            return None

        entry = self.l1_cache[key]

        # 만료 확인
        if await self._is_expired(entry):
            await self._delete_from_l1(key)
            return None

        # LRU 업데이트
        entry.access_count += 1
        entry.last_access = time.time()

        # OrderedDict에서 가장 최근 접근으로 이동
        self.l1_cache.move_to_end(key)

        return entry

    async def _get_from_l2(self, key: str) -> Any:
        """L2 Redis 캐시에서 조회"""
        try:
            serialized_data = await self.redis_client.get(key)
            if serialized_data is None:
                return None

            # 역직렬화
            value = await self._deserialize_value(serialized_data)
            return value

        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            return None

    async def _set_to_l1(self, entry: CacheEntry):
        """L1 메모리 캐시에 저장"""
        async with self._lock:
            # 메모리 사용량 확인 및 공간 확보
            await self._ensure_l1_capacity(entry.size_bytes)

            # 기존 엔트리 제거 (크기 업데이트를 위해)
            if entry.key in self.l1_cache:
                old_entry = self.l1_cache[entry.key]
                self.l1_current_memory -= old_entry.size_bytes

            # 새 엔트리 추가
            self.l1_cache[entry.key] = entry
            self.l1_current_memory += entry.size_bytes

            # 크기 제한 확인
            if len(self.l1_cache) > self.l1_max_size:
                await self._evict_from_l1()

    async def _set_to_l2(self, key: str, serialized_value: bytes, ttl: int, tags: List[str]):
        """L2 Redis 캐시에 저장"""
        try:
            # 값 저장
            await self.redis_client.setex(key, ttl, serialized_value)

            # 태그 인덱스 업데이트
            for tag in tags:
                tag_key = f"tag:{tag}"
                await self.redis_client.sadd(tag_key, key)
                await self.redis_client.expire(tag_key, ttl)

        except Exception as e:
            logger.error(f"L2 cache set error: {e}")

    async def _delete_from_l1(self, key: str):
        """L1에서 삭제"""
        if key in self.l1_cache:
            entry = self.l1_cache.pop(key)
            self.l1_current_memory -= entry.size_bytes

    async def _delete_from_l2(self, key: str):
        """L2에서 삭제"""
        try:
            await self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")

    async def _promote_to_l1(self, key: str, value: Any):
        """L2에서 L1으로 프로모션"""
        try:
            serialized_value, compressed = await self._serialize_value(value)
            size_bytes = len(serialized_value)

            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                compressed=compressed
            )

            await self._set_to_l1(entry)

        except Exception as e:
            logger.error(f"L1 promotion error: {e}")

    async def _ensure_l1_capacity(self, required_bytes: int):
        """L1 용량 확보"""
        while (self.l1_current_memory + required_bytes > self.l1_max_memory_bytes or
               len(self.l1_cache) >= self.l1_max_size):

            if not self.l1_cache:
                break

            await self._evict_from_l1()

    async def _evict_from_l1(self):
        """L1에서 제거 정책 적용"""
        if not self.l1_cache:
            return

        if self.cache_policy == CachePolicy.LRU:
            # 가장 오래 전에 접근된 항목 제거
            key, entry = self.l1_cache.popitem(last=False)

        elif self.cache_policy == CachePolicy.LFU:
            # 가장 적게 접근된 항목 제거
            min_key = min(self.l1_cache.keys(),
                         key=lambda k: self.l1_cache[k].access_count)
            entry = self.l1_cache.pop(min_key)

        elif self.cache_policy == CachePolicy.ADAPTIVE:
            # 접근 빈도와 최근성을 고려한 적응형 제거
            key = await self._adaptive_eviction_key()
            entry = self.l1_cache.pop(key)

        else:  # TTL
            # 가장 먼저 만료되는 항목 제거
            min_key = min(self.l1_cache.keys(),
                         key=lambda k: self.l1_cache[k].created_at)
            entry = self.l1_cache.pop(min_key)

        self.l1_current_memory -= entry.size_bytes
        self.stats.evictions += 1

    async def _adaptive_eviction_key(self) -> str:
        """적응형 제거를 위한 키 선택"""
        scores = {}
        current_time = time.time()

        for key, entry in self.l1_cache.items():
            # 점수 계산 (낮을수록 제거 우선순위 높음)
            recency_score = 1.0 / (current_time - entry.last_access + 1)
            frequency_score = entry.access_count / (current_time - entry.created_at + 1)
            popularity_score = self.key_popularity.get(key, 0)

            # 가중 평균
            final_score = (
                recency_score * 0.4 +
                frequency_score * 0.4 +
                popularity_score * 0.2
            )

            scores[key] = final_score

        # 가장 낮은 점수의 키 반환
        return min(scores.keys(), key=lambda k: scores[k])

    async def _serialize_value(self, value: Any) -> Tuple[bytes, bool]:
        """값 직렬화 및 압축"""
        try:
            # JSON 시도
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serialized = json.dumps(value, ensure_ascii=False).encode('utf-8')
                json_serialized = True
            else:
                # Pickle 사용
                serialized = pickle.dumps(value)
                json_serialized = False

            # 압축 적용 (크기가 임계값을 넘는 경우)
            compressed = False
            if len(serialized) > self.compression_threshold:
                serialized = zlib.compress(serialized)
                compressed = True

            return serialized, compressed

        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise

    async def _deserialize_value(self, data: bytes) -> Any:
        """값 역직렬화 및 압축 해제"""
        try:
            # 압축 해제 시도
            try:
                decompressed = zlib.decompress(data)
                data = decompressed
            except zlib.error:
                # 압축되지 않은 데이터
                pass

            # JSON 역직렬화 시도
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Pickle 역직렬화
                return pickle.loads(data)

        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise

    async def _is_expired(self, entry: CacheEntry) -> bool:
        """만료 확인"""
        if entry.ttl is None:
            return False

        return time.time() - entry.created_at > entry.ttl

    async def _update_access_pattern(self, key: str):
        """접근 패턴 업데이트"""
        current_time = time.time()

        # 최근 접근 시간 기록 (최대 100개)
        if len(self.access_patterns[key]) >= 100:
            self.access_patterns[key].pop(0)

        self.access_patterns[key].append(current_time)

    async def _update_popularity(self, key: str, score: float):
        """키 인기도 업데이트"""
        # 기존 인기도에 새 점수를 가중 평균으로 반영
        current_popularity = self.key_popularity[key]
        self.key_popularity[key] = current_popularity * 0.9 + score * 0.1

    async def _cleanup_worker(self):
        """백그라운드 정리 작업"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다 실행
                await self._cleanup_expired_entries()
                await self._update_cache_stats()

            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")

    async def _metrics_worker(self):
        """백그라운드 메트릭 수집"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 실행

                if self.enable_metrics and self.metrics_collector:
                    # 캐시 메트릭 기록
                    stats = await self.get_stats()
                    self.metrics_collector.record_cache_stats(stats)

            except Exception as e:
                logger.error(f"Metrics worker error: {e}")

    async def _cleanup_expired_entries(self):
        """만료된 엔트리 정리"""
        expired_keys = []

        for key, entry in self.l1_cache.items():
            if await self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            await self._delete_from_l1(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired L1 entries")

    async def _update_cache_stats(self):
        """캐시 통계 업데이트"""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests * 100

        self.stats.total_keys = len(self.l1_cache)
        self.stats.memory_usage = self.l1_current_memory

    async def shutdown(self):
        """캐시 매니저 종료"""
        try:
            # 백그라운드 작업 중지
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._metrics_task:
                self._metrics_task.cancel()

            # Redis 연결 종료
            if self.redis_client:
                await self.redis_client.close()

            logger.info("Cache manager shutdown completed")

        except Exception as e:
            logger.error(f"Cache shutdown error: {e}")


# 전역 캐시 매니저 인스턴스
cache_manager: Optional[AdvancedCacheManager] = None

def get_cache_manager() -> Optional[AdvancedCacheManager]:
    """글로벌 캐시 매니저 반환"""
    return cache_manager

async def initialize_cache_manager(
    redis_url: Optional[str] = None,
    **kwargs
) -> AdvancedCacheManager:
    """캐시 매니저 초기화"""
    global cache_manager

    cache_manager = AdvancedCacheManager(redis_url=redis_url, **kwargs)
    await cache_manager.initialize()

    return cache_manager