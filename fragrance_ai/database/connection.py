import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import asynccontextmanager, contextmanager
from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError
import threading
import logging
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..core.logging_config import get_logger, performance_logger
from ..core.exceptions import (
    DatabaseException, SystemException, ErrorCode,
    FragranceAIException
)
from ..core.monitoring import metrics_collector

logger = get_logger(__name__)


class ConnectionState(str, Enum):
    """연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"


@dataclass
class ConnectionMetrics:
    """연결 메트릭"""
    total_connections: int
    active_connections: int
    idle_connections: int
    pool_size: int
    overflow_connections: int
    checked_out_connections: int
    connection_errors: int
    avg_connection_time_ms: float
    last_connection_time: Optional[float]


# Alias for compatibility
DatabaseManager = 'DatabaseConnectionManager'

class DatabaseConnectionManager:
    """데이터베이스 연결 관리자"""
    
    def __init__(self):
        self.state = ConnectionState.DISCONNECTED
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        self._connection_metrics = ConnectionMetrics(
            total_connections=0,
            active_connections=0,
            idle_connections=0,
            pool_size=0,
            overflow_connections=0,
            checked_out_connections=0,
            connection_errors=0,
            avg_connection_time_ms=0.0,
            last_connection_time=None
        )
        
        self._connection_times: List[float] = []
        self._max_connection_times = 100
        self._lock = threading.Lock()
        
        # 연결 이벤트 핸들러
        self._connection_listeners: List[Callable] = []
        self._disconnection_listeners: List[Callable] = []
    
    def initialize(self):
        """데이터베이스 연결 초기화"""
        
        try:
            logger.info("Initializing database connection...")
            self.state = ConnectionState.CONNECTING
            
            start_time = time.time()
            
            # 동기 엔진 생성
            self.engine = self._create_sync_engine()
            
            # 비동기 엔진 생성
            self.async_engine = self._create_async_engine()
            
            # 세션 팩토리 생성
            self.session_factory = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # 연결 테스트
            self._test_connection()
            
            # 메트릭 업데이트
            connection_time = (time.time() - start_time) * 1000
            self._record_connection_time(connection_time)
            
            self.state = ConnectionState.CONNECTED
            self._notify_connection_listeners("connected")
            
            logger.info(f"Database connection initialized successfully in {connection_time:.2f}ms")
            
        except Exception as e:
            self.state = ConnectionState.ERROR
            logger.error(f"Failed to initialize database connection: {e}")
            self._notify_connection_listeners("error", str(e))
            
            raise DatabaseException(
                message=f"Database connection initialization failed: {str(e)}",
                cause=e
            )
    
    def _create_sync_engine(self):
        """동기 엔진 생성"""
        
        engine_config = {
            "echo": settings.debug,
            "future": True,
            "pool_pre_ping": True,  # 연결 유효성 확인
            "pool_recycle": 3600,   # 1시간마다 연결 재생성
            "connect_args": {}
        }
        
        if settings.database_url.startswith("sqlite"):
            engine_config.update({
                "poolclass": StaticPool,
                "connect_args": {
                    "check_same_thread": False,
                    "timeout": 20
                }
            })
        else:
            # PostgreSQL 설정
            engine_config.update({
                "poolclass": QueuePool,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "connect_args": {
                    "connect_timeout": 10,
                    "application_name": f"{settings.app_name}_{settings.app_version}"
                }
            })
        
        return create_engine(settings.database_url, **engine_config)
    
    def _create_async_engine(self):
        """비동기 엔진 생성"""
        
        # PostgreSQL의 경우 asyncpg 드라이버 사용
        async_url = settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
        
        if settings.database_url.startswith("sqlite"):
            async_url = settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
            
            return create_async_engine(
                async_url,
                echo=settings.debug,
                future=True,
                pool_pre_ping=True,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False}
            )
        else:
            return create_async_engine(
                async_url,
                echo=settings.debug,
                future=True,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600
            )
    
    def _test_connection(self):
        """연결 테스트"""
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
                
            logger.info("Database connection test passed")
            
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise DatabaseException(
                message=f"Database connection test failed: {str(e)}",
                cause=e
            )
    
    @contextmanager
    def get_session(self):
        """동기 세션 컨텍스트 매니저"""
        
        if self.state != ConnectionState.CONNECTED:
            raise DatabaseException(
                message="Database not connected",
                error_code=ErrorCode.DATABASE_ERROR
            )
        
        session = self.session_factory()
        start_time = time.time()
        
        try:
            with self._lock:
                self._connection_metrics.active_connections += 1
                metrics_collector.set_db_active_connections(
                    self._connection_metrics.active_connections
                )
            
            yield session
            session.commit()
            
        except Exception as e:
            logger.error(f"Database session error: {e}")
            session.rollback()
            
            with self._lock:
                self._connection_metrics.connection_errors += 1
            
            if isinstance(e, (DisconnectionError, OperationalError)):
                self._handle_connection_error(e)
            
            raise DatabaseException(
                message=f"Database session error: {str(e)}",
                cause=e
            )
            
        finally:
            session.close()
            
            query_time = (time.time() - start_time) * 1000
            metrics_collector.record_db_query_time(query_time)
            
            with self._lock:
                self._connection_metrics.active_connections = max(
                    0, self._connection_metrics.active_connections - 1
                )
                metrics_collector.set_db_active_connections(
                    self._connection_metrics.active_connections
                )
    
    @asynccontextmanager
    async def get_async_session(self):
        """비동기 세션 컨텍스트 매니저"""
        
        if self.state != ConnectionState.CONNECTED:
            raise DatabaseException(
                message="Database not connected",
                error_code=ErrorCode.DATABASE_ERROR
            )
        
        session = self.async_session_factory()
        start_time = time.time()
        
        try:
            with self._lock:
                self._connection_metrics.active_connections += 1
                metrics_collector.set_db_active_connections(
                    self._connection_metrics.active_connections
                )
            
            yield session
            await session.commit()
            
        except Exception as e:
            logger.error(f"Async database session error: {e}")
            await session.rollback()
            
            with self._lock:
                self._connection_metrics.connection_errors += 1
            
            if isinstance(e, (DisconnectionError, OperationalError)):
                self._handle_connection_error(e)
            
            raise DatabaseException(
                message=f"Async database session error: {str(e)}",
                cause=e
            )
            
        finally:
            await session.close()
            
            query_time = (time.time() - start_time) * 1000
            metrics_collector.record_db_query_time(query_time)
            
            with self._lock:
                self._connection_metrics.active_connections = max(
                    0, self._connection_metrics.active_connections - 1
                )
                metrics_collector.set_db_active_connections(
                    self._connection_metrics.active_connections
                )
    
    def health_check(self) -> Dict[str, Any]:
        """데이터베이스 헬스 체크"""
        
        health_data = {
            "status": "unhealthy",
            "message": "",
            "connection_state": self.state.value,
            "details": {}
        }
        
        try:
            if self.state != ConnectionState.CONNECTED:
                health_data["message"] = f"Database not connected (state: {self.state.value})"
                return health_data
            
            # 연결 테스트
            start_time = time.time()
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            response_time = (time.time() - start_time) * 1000
            
            # 풀 상태 확인
            pool = self.engine.pool
            pool_status = {
                "size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalidated": pool.invalidated()
            }
            
            health_data.update({
                "status": "healthy",
                "message": "Database connection is healthy",
                "response_time_ms": response_time,
                "details": {
                    "pool_status": pool_status,
                    "connection_metrics": self._connection_metrics.__dict__
                }
            })
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            health_data["message"] = f"Health check failed: {str(e)}"
            health_data["details"]["error"] = str(e)
            
            # 연결 오류인 경우 재연결 시도
            if isinstance(e, (DisconnectionError, OperationalError)):
                self._handle_connection_error(e)
        
        return health_data
    
    async def async_health_check(self) -> Dict[str, Any]:
        """비동기 데이터베이스 헬스 체크"""
        
        health_data = {
            "status": "unhealthy",
            "message": "",
            "connection_state": self.state.value,
            "details": {}
        }
        
        try:
            if self.state != ConnectionState.CONNECTED or not self.async_engine:
                health_data["message"] = "Async database not connected"
                return health_data
            
            # 비동기 연결 테스트
            start_time = time.time()
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            
            response_time = (time.time() - start_time) * 1000
            
            health_data.update({
                "status": "healthy",
                "message": "Async database connection is healthy",
                "response_time_ms": response_time,
                "details": {
                    "connection_metrics": self._connection_metrics.__dict__
                }
            })
            
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            health_data["message"] = f"Async health check failed: {str(e)}"
            health_data["details"]["error"] = str(e)
        
        return health_data
    
    def get_connection_metrics(self) -> ConnectionMetrics:
        """연결 메트릭 반환"""
        
        try:
            # 풀 상태 업데이트
            if self.engine and self.engine.pool:
                pool = self.engine.pool
                with self._lock:
                    self._connection_metrics.pool_size = pool.size()
                    self._connection_metrics.checked_out_connections = pool.checkedout()
                    self._connection_metrics.idle_connections = pool.checkedin()
                    self._connection_metrics.overflow_connections = pool.overflow()
                    
                    # 평균 연결 시간 계산
                    if self._connection_times:
                        self._connection_metrics.avg_connection_time_ms = (
                            sum(self._connection_times) / len(self._connection_times)
                        )
            
            return self._connection_metrics
            
        except Exception as e:
            logger.error(f"Failed to get connection metrics: {e}")
            return self._connection_metrics
    
    def add_connection_listener(self, callback: Callable):
        """연결 이벤트 리스너 추가"""
        self._connection_listeners.append(callback)
    
    def add_disconnection_listener(self, callback: Callable):
        """연결 해제 이벤트 리스너 추가"""
        self._disconnection_listeners.append(callback)
    
    def _handle_connection_error(self, error: Exception):
        """연결 오류 처리"""
        
        logger.warning(f"Database connection error detected: {error}")
        
        if self.state == ConnectionState.CONNECTED:
            self.state = ConnectionState.RECONNECTING
            self._notify_disconnection_listeners(str(error))
            
            # 백그라운드에서 재연결 시도
            threading.Thread(target=self._attempt_reconnection).start()
    
    def _attempt_reconnection(self):
        """재연결 시도"""
        
        max_attempts = 5
        base_delay = 1.0
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Attempting database reconnection ({attempt}/{max_attempts})")
                
                # 기존 연결 정리
                self._cleanup_connections()
                
                # 재초기화
                self.initialize()
                
                logger.info("Database reconnection successful")
                return
                
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt} failed: {e}")
                
                if attempt < max_attempts:
                    delay = base_delay * (2 ** (attempt - 1))  # 지수 백오프
                    time.sleep(delay)
        
        logger.error("All reconnection attempts failed")
        self.state = ConnectionState.ERROR
        self._notify_connection_listeners("reconnection_failed")
    
    def _cleanup_connections(self):
        """연결 정리"""
        
        try:
            if self.engine:
                self.engine.dispose()
            
            if self.async_engine:
                asyncio.create_task(self.async_engine.dispose())
                
        except Exception as e:
            logger.error(f"Error during connection cleanup: {e}")
    
    def _record_connection_time(self, connection_time: float):
        """연결 시간 기록"""
        
        with self._lock:
            self._connection_times.append(connection_time)
            
            # 최근 N개만 유지
            if len(self._connection_times) > self._max_connection_times:
                self._connection_times = self._connection_times[-self._max_connection_times:]
            
            self._connection_metrics.last_connection_time = time.time()
            self._connection_metrics.total_connections += 1
    
    def _notify_connection_listeners(self, event: str, details: str = None):
        """연결 리스너에게 알림"""
        
        for listener in self._connection_listeners:
            try:
                listener(event, details)
            except Exception as e:
                logger.error(f"Connection listener error: {e}")
    
    def _notify_disconnection_listeners(self, details: str = None):
        """연결 해제 리스너에게 알림"""
        
        for listener in self._disconnection_listeners:
            try:
                listener(details)
            except Exception as e:
                logger.error(f"Disconnection listener error: {e}")
    
    def close(self):
        """연결 종료"""
        
        logger.info("Closing database connections...")
        
        try:
            self.state = ConnectionState.DISCONNECTED
            self._cleanup_connections()
            logger.info("Database connections closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


# 전역 연결 관리자 인스턴스
db_connection_manager = DatabaseConnectionManager()


def get_db_session():
    """데이터베이스 세션 의존성"""
    return db_connection_manager.get_session()


async def get_async_db_session():
    """비동기 데이터베이스 세션 의존성"""
    return db_connection_manager.get_async_session()


def initialize_database():
    """데이터베이스 초기화"""
    
    try:
        db_connection_manager.initialize()
        
        # 모니터링 시스템에 헬스 체크 등록
        from ..core.monitoring import HealthChecker
        # health_checker.register_health_check("database", db_connection_manager.health_check)
        
        performance_logger.log_execution_time(
            operation="database_initialization",
            execution_time=0.0,  # 실제로는 초기화 시간 측정
            success=True
        )
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def shutdown_database():
    """데이터베이스 종료"""
    
    try:
        db_connection_manager.close()
        
    except Exception as e:
        logger.error(f"Database shutdown failed: {e}")
        raise