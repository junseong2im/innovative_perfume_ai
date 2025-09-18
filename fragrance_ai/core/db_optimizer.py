"""
데이터베이스 최적화 및 연결 관리자
- 연결 풀 최적화
- 쿼리 캐싱 및 최적화
- 읽기/쓰기 분산
- 배치 처리 최적화
"""

import asyncio
import time
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import defaultdict, OrderedDict
import sqlalchemy
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, NullPool
import psycopg2
from psycopg2 import OperationalError
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from .logging_config import get_logger
from .monitoring import MetricsCollector
from .cache_manager import AdvancedCacheManager

logger = get_logger(__name__)

@dataclass
class QueryMetrics:
    query_hash: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_executed: float = field(default_factory=time.time)
    error_count: int = 0

@dataclass
class ConnectionStats:
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connection_errors: int = 0
    avg_connection_time: float = 0.0
    query_count: int = 0
    transaction_count: int = 0

class DatabaseOptimizer:
    """데이터베이스 최적화 관리자"""

    def __init__(
        self,
        database_url: str,
        read_replica_urls: Optional[List[str]] = None,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        enable_query_cache: bool = True,
        cache_manager: Optional[AdvancedCacheManager] = None,
        enable_metrics: bool = True
    ):
        self.database_url = database_url
        self.read_replica_urls = read_replica_urls or []

        # 연결 풀 설정
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout

        # 엔진 및 세션
        self.write_engine: Optional[Engine] = None
        self.read_engines: List[Engine] = []
        self.async_write_engine = None
        self.async_read_engines: List = []

        self.SessionLocal: Optional[sessionmaker] = None
        self.AsyncSessionLocal: Optional[async_sessionmaker] = None

        # 쿼리 최적화
        self.enable_query_cache = enable_query_cache
        self.cache_manager = cache_manager
        self.query_cache: OrderedDict[str, Any] = OrderedDict()
        self.query_cache_max_size = 10000

        # 메트릭 및 통계
        self.enable_metrics = enable_metrics
        self.metrics_collector = MetricsCollector() if enable_metrics else None
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.connection_stats = ConnectionStats()

        # 쿼리 분석 및 최적화
        self.slow_query_threshold = 1.0  # 1초
        self.query_patterns: Dict[str, int] = defaultdict(int)

        # 라우팅
        self.read_replica_index = 0

        # 백그라운드 작업
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        # 설정 플래그
        self._initialized = False

    async def initialize(self):
        """데이터베이스 최적화 시스템 초기화"""
        try:
            logger.info("Initializing database optimizer...")

            # 동기 엔진 설정
            await self._setup_sync_engines()

            # 비동기 엔진 설정
            await self._setup_async_engines()

            # 연결 테스트
            await self._test_connections()

            # 이벤트 리스너 설정
            self._setup_event_listeners()

            # 백그라운드 작업 시작
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())

            if self.enable_metrics:
                self._metrics_task = asyncio.create_task(self._metrics_worker())

            self._initialized = True
            logger.info("Database optimizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database optimizer: {e}")
            raise

    async def _setup_sync_engines(self):
        """동기 엔진 설정"""
        # 쓰기 엔진
        self.write_engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=3600,  # 1시간마다 연결 재생성
            pool_pre_ping=True,  # 연결 유효성 검사
            echo=False,  # 프로덕션에서는 False
            future=True
        )

        # 읽기 엔진들 (read replicas)
        for replica_url in self.read_replica_urls:
            read_engine = create_engine(
                replica_url,
                poolclass=QueuePool,
                pool_size=max(self.pool_size // 2, 5),  # 읽기 전용이므로 작은 풀
                max_overflow=max(self.max_overflow // 2, 10),
                pool_timeout=self.pool_timeout,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False,
                future=True
            )
            self.read_engines.append(read_engine)

        # 세션 팩토리
        self.SessionLocal = sessionmaker(
            bind=self.write_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

    async def _setup_async_engines(self):
        """비동기 엔진 설정"""
        # 비동기 쓰기 엔진
        async_write_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        self.async_write_engine = create_async_engine(
            async_write_url,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_timeout=self.pool_timeout,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False
        )

        # 비동기 읽기 엔진들
        for replica_url in self.read_replica_urls:
            async_replica_url = replica_url.replace("postgresql://", "postgresql+asyncpg://")
            async_read_engine = create_async_engine(
                async_replica_url,
                pool_size=max(self.pool_size // 2, 5),
                max_overflow=max(self.max_overflow // 2, 10),
                pool_timeout=self.pool_timeout,
                pool_recycle=3600,
                pool_pre_ping=True,
                echo=False
            )
            self.async_read_engines.append(async_read_engine)

        # 비동기 세션 팩토리
        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.async_write_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def _test_connections(self):
        """연결 테스트"""
        # 동기 연결 테스트
        with self.write_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.info("Write database connection test passed")

        for i, read_engine in enumerate(self.read_engines):
            with read_engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
                logger.info(f"Read replica {i+1} connection test passed")

        # 비동기 연결 테스트
        async with self.async_write_engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            logger.info("Async write database connection test passed")

        for i, async_read_engine in enumerate(self.async_read_engines):
            async with async_read_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                assert result.scalar() == 1
                logger.info(f"Async read replica {i+1} connection test passed")

    def _setup_event_listeners(self):
        """이벤트 리스너 설정"""
        # 쿼리 실행 시간 추적
        @event.listens_for(self.write_engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            context._query_start_time = time.time()

        @event.listens_for(self.write_engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - context._query_start_time
            self._record_query_metrics(statement, total_time)

        # 연결 통계 추적
        @event.listens_for(self.write_engine, "connect")
        def receive_connect(dbapi_connection, connection_record):
            self.connection_stats.total_connections += 1

        @event.listens_for(self.write_engine, "checkout")
        def receive_checkout(dbapi_connection, connection_record, connection_proxy):
            self.connection_stats.active_connections += 1

        @event.listens_for(self.write_engine, "checkin")
        def receive_checkin(dbapi_connection, connection_record):
            self.connection_stats.active_connections -= 1

    def get_read_engine(self) -> Engine:
        """로드 밸런싱으로 읽기 엔진 선택"""
        if not self.read_engines:
            return self.write_engine

        # Round-robin 방식
        engine = self.read_engines[self.read_replica_index]
        self.read_replica_index = (self.read_replica_index + 1) % len(self.read_engines)
        return engine

    def get_async_read_engine(self):
        """비동기 읽기 엔진 선택"""
        if not self.async_read_engines:
            return self.async_write_engine

        # Round-robin 방식
        engine = self.async_read_engines[self.read_replica_index]
        self.read_replica_index = (self.read_replica_index + 1) % len(self.async_read_engines)
        return engine

    @asynccontextmanager
    async def get_db_session(self, read_only: bool = False) -> AsyncGenerator[AsyncSession, None]:
        """최적화된 데이터베이스 세션"""
        engine = self.get_async_read_engine() if read_only else self.async_write_engine

        async with engine.begin() as conn:
            session = AsyncSession(bind=conn, expire_on_commit=False)
            try:
                yield session
                if not read_only:
                    await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_db_session(self, read_only: bool = False) -> Session:
        """동기 데이터베이스 세션"""
        engine = self.get_read_engine() if read_only else self.write_engine
        return Session(bind=engine, expire_on_commit=False)

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        read_only: bool = False,
        cache_key: Optional[str] = None,
        cache_ttl: int = 300
    ) -> Any:
        """최적화된 쿼리 실행"""
        start_time = time.time()

        try:
            # 캐시 확인
            if cache_key and self.enable_query_cache:
                cached_result = await self._get_cached_query_result(cache_key)
                if cached_result is not None:
                    return cached_result

            # 쿼리 실행
            async with self.get_db_session(read_only=read_only) as session:
                result = await session.execute(text(query), params or {})

                if query.strip().upper().startswith('SELECT'):
                    # SELECT 쿼리인 경우 결과 페치
                    rows = result.fetchall()
                    result_data = [dict(row._mapping) for row in rows]
                else:
                    # INSERT/UPDATE/DELETE 쿼리인 경우 rowcount
                    result_data = result.rowcount

                # 캐시에 저장
                if cache_key and self.enable_query_cache and read_only:
                    await self._cache_query_result(cache_key, result_data, cache_ttl)

                execution_time = time.time() - start_time
                self._record_query_metrics(query, execution_time)

                return result_data

        except Exception as e:
            execution_time = time.time() - start_time
            self._record_query_metrics(query, execution_time, error=True)
            logger.error(f"Query execution failed: {e}")
            raise

    async def execute_batch_queries(
        self,
        queries: List[Tuple[str, Optional[Dict[str, Any]]]],
        read_only: bool = False
    ) -> List[Any]:
        """배치 쿼리 실행 (최적화)"""
        start_time = time.time()
        results = []

        try:
            async with self.get_db_session(read_only=read_only) as session:
                for query, params in queries:
                    result = await session.execute(text(query), params or {})

                    if query.strip().upper().startswith('SELECT'):
                        rows = result.fetchall()
                        result_data = [dict(row._mapping) for row in rows]
                    else:
                        result_data = result.rowcount

                    results.append(result_data)

                execution_time = time.time() - start_time
                logger.info(f"Executed {len(queries)} queries in {execution_time:.3f}s")

                return results

        except Exception as e:
            logger.error(f"Batch query execution failed: {e}")
            raise

    async def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """테이블 통계 조회"""
        cache_key = f"table_stats:{table_name}"

        query = """
        SELECT
            schemaname,
            tablename,
            attname as column_name,
            n_distinct,
            correlation,
            null_frac
        FROM pg_stats
        WHERE tablename = :table_name
        """

        try:
            result = await self.execute_query(
                query,
                {"table_name": table_name},
                read_only=True,
                cache_key=cache_key,
                cache_ttl=3600  # 1시간 캐시
            )

            # 추가 통계 정보 수집
            size_query = """
        SELECT
            pg_size_pretty(pg_total_relation_size(:table_name)) as total_size,
            pg_size_pretty(pg_relation_size(:table_name)) as table_size,
            (SELECT reltuples::BIGINT FROM pg_class WHERE relname = :table_name) as row_count
        """

            size_result = await self.execute_query(
                size_query,
                {"table_name": table_name},
                read_only=True,
                cache_key=f"{cache_key}:size",
                cache_ttl=3600
            )

            return {
                "column_stats": result,
                "size_info": size_result[0] if size_result else {}
            }

        except Exception as e:
            logger.error(f"Failed to get table stats for {table_name}: {e}")
            return {}

    async def analyze_slow_queries(self) -> List[Dict[str, Any]]:
        """느린 쿼리 분석"""
        slow_queries = []

        for query_hash, metrics in self.query_metrics.items():
            if metrics.avg_time > self.slow_query_threshold:
                slow_queries.append({
                    "query_hash": query_hash,
                    "execution_count": metrics.execution_count,
                    "avg_time": metrics.avg_time,
                    "max_time": metrics.max_time,
                    "total_time": metrics.total_time,
                    "error_count": metrics.error_count
                })

        # 평균 실행 시간으로 정렬
        slow_queries.sort(key=lambda x: x["avg_time"], reverse=True)
        return slow_queries[:20]  # 상위 20개

    async def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """테이블 최적화 실행"""
        optimization_results = {}

        try:
            # ANALYZE 실행
            analyze_start = time.time()
            await self.execute_query(f"ANALYZE {table_name}", read_only=False)
            optimization_results["analyze_time"] = time.time() - analyze_start

            # VACUUM 실행 (필요시)
            vacuum_start = time.time()
            await self.execute_query(f"VACUUM ANALYZE {table_name}", read_only=False)
            optimization_results["vacuum_time"] = time.time() - vacuum_start

            # 통계 업데이트 후 상태 확인
            stats = await self.get_table_stats(table_name)
            optimization_results["updated_stats"] = stats

            logger.info(f"Table optimization completed for {table_name}")
            return optimization_results

        except Exception as e:
            logger.error(f"Table optimization failed for {table_name}: {e}")
            raise

    def _record_query_metrics(self, query: str, execution_time: float, error: bool = False):
        """쿼리 메트릭 기록"""
        # 쿼리 해시 생성 (파라미터 제외)
        normalized_query = self._normalize_query(query)
        query_hash = hashlib.md5(normalized_query.encode()).hexdigest()

        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(query_hash=query_hash)

        metrics = self.query_metrics[query_hash]
        metrics.execution_count += 1
        metrics.total_time += execution_time
        metrics.avg_time = metrics.total_time / metrics.execution_count
        metrics.min_time = min(metrics.min_time, execution_time)
        metrics.max_time = max(metrics.max_time, execution_time)
        metrics.last_executed = time.time()

        if error:
            metrics.error_count += 1

        # 쿼리 패턴 분석
        pattern = self._extract_query_pattern(normalized_query)
        self.query_patterns[pattern] += 1

        # 메트릭 수집기에 기록
        if self.enable_metrics and self.metrics_collector:
            self.metrics_collector.record_query_execution(
                query_type=pattern,
                execution_time=execution_time,
                success=not error
            )

    def _normalize_query(self, query: str) -> str:
        """쿼리 정규화 (파라미터 제거)"""
        # 간단한 정규화 (실제로는 더 정교한 파서 필요)
        import re

        # 파라미터 플레이스홀더 정규화
        normalized = re.sub(r':\w+', ':param', query)
        normalized = re.sub(r'\$\d+', '$param', normalized)
        normalized = re.sub(r"'[^']*'", "'value'", normalized)
        normalized = re.sub(r'\d+', 'N', normalized)

        # 공백 정리
        normalized = re.sub(r'\s+', ' ', normalized.strip())

        return normalized.upper()

    def _extract_query_pattern(self, query: str) -> str:
        """쿼리 패턴 추출"""
        if query.startswith('SELECT'):
            return 'SELECT'
        elif query.startswith('INSERT'):
            return 'INSERT'
        elif query.startswith('UPDATE'):
            return 'UPDATE'
        elif query.startswith('DELETE'):
            return 'DELETE'
        elif query.startswith('ANALYZE'):
            return 'ANALYZE'
        elif query.startswith('VACUUM'):
            return 'VACUUM'
        else:
            return 'OTHER'

    async def _get_cached_query_result(self, cache_key: str) -> Any:
        """캐시된 쿼리 결과 조회"""
        # 로컬 캐시 먼저 확인
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # 외부 캐시 매니저 사용
        if self.cache_manager:
            return await self.cache_manager.get(f"query:{cache_key}")

        return None

    async def _cache_query_result(self, cache_key: str, result: Any, ttl: int):
        """쿼리 결과 캐시"""
        # 로컬 캐시 저장 (크기 제한)
        if len(self.query_cache) >= self.query_cache_max_size:
            self.query_cache.popitem(last=False)  # LRU 제거

        self.query_cache[cache_key] = result

        # 외부 캐시 매니저 사용
        if self.cache_manager:
            await self.cache_manager.set(f"query:{cache_key}", result, ttl=ttl)

    async def get_performance_report(self) -> Dict[str, Any]:
        """성능 보고서 생성"""
        return {
            "connection_stats": {
                "total_connections": self.connection_stats.total_connections,
                "active_connections": self.connection_stats.active_connections,
                "idle_connections": self.connection_stats.idle_connections,
                "connection_errors": self.connection_stats.connection_errors,
                "query_count": self.connection_stats.query_count,
                "transaction_count": self.connection_stats.transaction_count
            },
            "query_metrics": {
                "total_queries": len(self.query_metrics),
                "slow_queries": len([m for m in self.query_metrics.values()
                                   if m.avg_time > self.slow_query_threshold]),
                "query_patterns": dict(self.query_patterns)
            },
            "cache_stats": {
                "query_cache_size": len(self.query_cache),
                "query_cache_max_size": self.query_cache_max_size
            },
            "slow_queries": await self.analyze_slow_queries()
        }

    async def _cleanup_worker(self):
        """백그라운드 정리 작업"""
        while True:
            try:
                await asyncio.sleep(1800)  # 30분마다 실행

                # 오래된 쿼리 메트릭 정리
                current_time = time.time()
                expired_keys = [
                    key for key, metrics in self.query_metrics.items()
                    if current_time - metrics.last_executed > 86400  # 24시간
                ]

                for key in expired_keys:
                    del self.query_metrics[key]

                # 로컬 쿼리 캐시 정리
                if len(self.query_cache) > self.query_cache_max_size * 0.8:
                    # 80% 이상 찰 때 25% 제거
                    remove_count = len(self.query_cache) // 4
                    for _ in range(remove_count):
                        if self.query_cache:
                            self.query_cache.popitem(last=False)

                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired query metrics")

            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")

    async def _metrics_worker(self):
        """백그라운드 메트릭 수집"""
        while True:
            try:
                await asyncio.sleep(60)  # 1분마다 실행

                if self.enable_metrics and self.metrics_collector:
                    # DB 성능 메트릭 수집
                    perf_report = await self.get_performance_report()
                    self.metrics_collector.record_db_performance(perf_report)

            except Exception as e:
                logger.error(f"Metrics worker error: {e}")

    async def shutdown(self):
        """데이터베이스 최적화 시스템 종료"""
        try:
            # 백그라운드 작업 중지
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._metrics_task:
                self._metrics_task.cancel()

            # 엔진 정리
            if self.write_engine:
                self.write_engine.dispose()

            for read_engine in self.read_engines:
                read_engine.dispose()

            if self.async_write_engine:
                await self.async_write_engine.dispose()

            for async_read_engine in self.async_read_engines:
                await async_read_engine.dispose()

            logger.info("Database optimizer shutdown completed")

        except Exception as e:
            logger.error(f"Database optimizer shutdown error: {e}")


# 전역 데이터베이스 최적화 인스턴스
db_optimizer: Optional[DatabaseOptimizer] = None

def get_db_optimizer() -> Optional[DatabaseOptimizer]:
    """글로벌 DB 최적화 인스턴스 반환"""
    return db_optimizer

async def initialize_db_optimizer(
    database_url: str,
    **kwargs
) -> DatabaseOptimizer:
    """DB 최적화 시스템 초기화"""
    global db_optimizer

    db_optimizer = DatabaseOptimizer(database_url=database_url, **kwargs)
    await db_optimizer.initialize()

    return db_optimizer