"""
Circuit Breaker Pattern Implementation
서비스 실패를 방지하기 위한 Circuit Breaker 패턴
"""

import time
import logging
from typing import Dict, Optional
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit Breaker 상태"""
    CLOSED = "closed"  # 정상 작동
    OPEN = "open"      # 차단 상태
    HALF_OPEN = "half_open"  # 테스트 상태


class CircuitBreaker:
    """
    Circuit Breaker 구현
    
    - 연속된 실패 시 서홂 호출 차단
    - 일정 시간 후 자동 복구 시도
    - 서비스별 독립적 관리
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """
        초기화
        
        Args:
            failure_threshold: 차단 트리거 실패 횟수
            recovery_timeout: 복구 시도까지 대기 시간(초)
            expected_exception: 감지할 예외 타입
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        # 서비스별 상태 관리
        self._states: Dict[str, CircuitState] = defaultdict(lambda: CircuitState.CLOSED)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._last_failure_times: Dict[str, Optional[datetime]] = {}
        self._success_counts: Dict[str, int] = defaultdict(int)
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Circuit Breaker initialized with threshold={failure_threshold}, timeout={recovery_timeout}s")
    
    def is_open(self, service_name: str) -> bool:
        """
        서홂 대한 Circuit이 열려있는지 확인
        
        Args:
            service_name: 서비스 이름
            
        Returns:
            True if circuit is open (blocking calls)
        """
        with self._lock:
            state = self._states[service_name]
            
            if state == CircuitState.OPEN:
                # 복구 시간 체크
                if self._should_attempt_reset(service_name):
                    self._states[service_name] = CircuitState.HALF_OPEN
                    logger.info(f"Circuit for {service_name} moved to HALF_OPEN")
                    return False
                return True
                
            return False
    
    def record_success(self, service_name: str):
        """
        성공 기록
        
        Args:
            service_name: 서비스 이름
        """
        with self._lock:
            state = self._states[service_name]
            
            if state == CircuitState.HALF_OPEN:
                self._success_counts[service_name] += 1
                
                # HALF_OPEN에서 성공 시 CLOSED로 전환
                if self._success_counts[service_name] >= 3:
                    self._states[service_name] = CircuitState.CLOSED
                    self._failure_counts[service_name] = 0
                    self._success_counts[service_name] = 0
                    logger.info(f"Circuit for {service_name} recovered to CLOSED")
                    
            elif state == CircuitState.CLOSED:
                # 정상 상태에서 성공 시 실패 카운트 리셋
                self._failure_counts[service_name] = 0
    
    def record_failure(self, service_name: str, exception: Optional[Exception] = None):
        """
        실패 기록
        
        Args:
            service_name: 서비스 이름
            exception: 발생한 예외
        """
        with self._lock:
            state = self._states[service_name]
            self._last_failure_times[service_name] = datetime.now()
            
            if state == CircuitState.HALF_OPEN:
                # HALF_OPEN에서 실패 시 즉시 OPEN으로
                self._states[service_name] = CircuitState.OPEN
                self._success_counts[service_name] = 0
                logger.warning(f"Circuit for {service_name} reopened due to failure in HALF_OPEN state")
                
            elif state == CircuitState.CLOSED:
                self._failure_counts[service_name] += 1
                
                # 임계치 초과 시 OPEN으로 전환
                if self._failure_counts[service_name] >= self.failure_threshold:
                    self._states[service_name] = CircuitState.OPEN
                    logger.error(f"Circuit for {service_name} opened after {self._failure_counts[service_name]} failures")
    
    def _should_attempt_reset(self, service_name: str) -> bool:
        """
        리셋 시도 여부 판단
        
        Args:
            service_name: 서비스 이름
            
        Returns:
            True if should attempt reset
        """
        last_failure = self._last_failure_times.get(service_name)
        if last_failure is None:
            return True
            
        time_since_failure = (datetime.now() - last_failure).total_seconds()
        return time_since_failure >= self.recovery_timeout
    
    def get_state(self, service_name: str) -> CircuitState:
        """
        서비스 상태 조회
        
        Args:
            service_name: 서비스 이름
            
        Returns:
            현재 Circuit 상태
        """
        with self._lock:
            return self._states[service_name]
    
    def reset(self, service_name: Optional[str] = None):
        """
        Circuit Breaker 리셋
        
        Args:
            service_name: 특정 서비스만 리셋 (None이면 전체)
        """
        with self._lock:
            if service_name:
                self._states[service_name] = CircuitState.CLOSED
                self._failure_counts[service_name] = 0
                self._success_counts[service_name] = 0
                self._last_failure_times[service_name] = None
                logger.info(f"Circuit for {service_name} manually reset")
            else:
                # 전체 리셋
                self._states.clear()
                self._failure_counts.clear()
                self._success_counts.clear()
                self._last_failure_times.clear()
                logger.info("All circuits manually reset")
    
    def get_status(self) -> Dict[str, Dict]:
        """
        전체 Circuit Breaker 상태 조회
        
        Returns:
            서비스별 상태 정보
        """
        with self._lock:
            status = {}
            for service in set(list(self._states.keys()) + list(self._failure_counts.keys())):
                status[service] = {
                    "state": self._states[service].value,
                    "failure_count": self._failure_counts[service],
                    "success_count": self._success_counts[service],
                    "last_failure": self._last_failure_times.get(service).isoformat() if self._last_failure_times.get(service) else None
                }
            return status


# 전역 Circuit Breaker 인스턴스
_global_circuit_breaker = None

def get_circuit_breaker() -> CircuitBreaker:
    """
    전역 Circuit Breaker 인스턴스 가져오기
    
    Returns:
        CircuitBreaker 인스턴스
    """
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
    return _global_circuit_breaker
