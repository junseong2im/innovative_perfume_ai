"""
개선된 에러 처리 시스템
중앙화된 에러 처리, 로깅, 복구 메커니즘을 제공합니다.
"""

import logging
import traceback
import asyncio
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from contextlib import asynccontextmanager, contextmanager

from .exceptions import FragranceAIException, ErrorCode
from .config import settings

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """에러 카테고리"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    MODEL_INFERENCE = "model_inference"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM = "system"
    NETWORK = "network"
    PERFORMANCE = "performance"


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """에러 기록"""
    error_id: str
    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    context: ErrorContext
    stack_trace: str
    is_handled: bool = False
    recovery_attempted: bool = False
    recovery_successful: bool = False
    user_message: Optional[str] = None
    technical_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "error_id": self.error_id,
            "exception_type": type(self.exception).__name__,
            "exception_message": str(self.exception),
            "category": self.category.value,
            "severity": self.severity.value,
            "timestamp": self.context.timestamp.isoformat(),
            "user_id": self.context.user_id,
            "request_id": self.context.request_id,
            "endpoint": self.context.endpoint,
            "is_handled": self.is_handled,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "user_message": self.user_message,
            "technical_message": self.technical_message
        }


class ErrorHandler:
    """중앙화된 에러 핸들러"""

    def __init__(self):
        self.error_records: List[ErrorRecord] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.notification_handlers: List[Callable] = []
        self.max_records = 10000
        self.logger = logging.getLogger(__name__)

    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[Exception, ErrorContext], bool]
    ):
        """복구 전략 등록"""
        if category not in self.recovery_strategies:
            self.recovery_strategies[category] = []
        self.recovery_strategies[category].append(strategy)

    def register_notification_handler(self, handler: Callable[[ErrorRecord], None]):
        """알림 핸들러 등록"""
        self.notification_handlers.append(handler)

    async def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Optional[ErrorContext] = None,
        attempt_recovery: bool = True
    ) -> ErrorRecord:
        """에러 처리"""
        if context is None:
            context = ErrorContext()

        # 에러 기록 생성
        error_record = ErrorRecord(
            error_id=context.error_id,
            exception=exception,
            category=category,
            severity=severity,
            context=context,
            stack_trace=traceback.format_exc(),
            user_message=self._generate_user_message(exception, category),
            technical_message=str(exception)
        )

        # 로깅
        await self._log_error(error_record)

        # 복구 시도
        if attempt_recovery:
            await self._attempt_recovery(error_record)

        # 알림 발송 (심각한 에러의 경우)
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_notifications(error_record)

        # 기록 저장
        self._store_error_record(error_record)

        return error_record

    async def _log_error(self, error_record: ErrorRecord):
        """에러 로깅"""
        log_data = {
            "error_id": error_record.error_id,
            "category": error_record.category.value,
            "severity": error_record.severity.value,
            "exception": str(error_record.exception),
            "user_id": error_record.context.user_id,
            "endpoint": error_record.context.endpoint
        }

        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", extra=log_data)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error occurred", extra=log_data)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error occurred", extra=log_data)
        else:
            self.logger.info("Low severity error occurred", extra=log_data)

    async def _attempt_recovery(self, error_record: ErrorRecord):
        """복구 시도"""
        strategies = self.recovery_strategies.get(error_record.category, [])

        error_record.recovery_attempted = len(strategies) > 0

        for strategy in strategies:
            try:
                success = await asyncio.get_event_loop().run_in_executor(
                    None,
                    strategy,
                    error_record.exception,
                    error_record.context
                )

                if success:
                    error_record.recovery_successful = True
                    self.logger.info(f"Recovery successful for error {error_record.error_id}")
                    break

            except Exception as recovery_error:
                self.logger.error(f"Recovery strategy failed: {recovery_error}")

    async def _send_notifications(self, error_record: ErrorRecord):
        """알림 발송"""
        for handler in self.notification_handlers:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    handler,
                    error_record
                )
            except Exception as notification_error:
                self.logger.error(f"Notification handler failed: {notification_error}")

    def _store_error_record(self, error_record: ErrorRecord):
        """에러 기록 저장"""
        self.error_records.append(error_record)

        # 메모리 사용량 제한
        if len(self.error_records) > self.max_records:
            self.error_records = self.error_records[-self.max_records//2:]

    def _generate_user_message(self, exception: Exception, category: ErrorCategory) -> str:
        """사용자 친화적 에러 메시지 생성"""
        user_messages = {
            ErrorCategory.VALIDATION: "입력하신 정보를 다시 확인해 주세요.",
            ErrorCategory.AUTHENTICATION: "로그인이 필요합니다.",
            ErrorCategory.AUTHORIZATION: "이 작업을 수행할 권한이 없습니다.",
            ErrorCategory.DATABASE: "일시적인 데이터 처리 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            ErrorCategory.EXTERNAL_API: "외부 서비스 연결에 문제가 있습니다. 잠시 후 다시 시도해 주세요.",
            ErrorCategory.MODEL_INFERENCE: "AI 모델 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
            ErrorCategory.BUSINESS_LOGIC: "요청을 처리하는 중 오류가 발생했습니다.",
            ErrorCategory.SYSTEM: "시스템 오류가 발생했습니다. 관리자에게 문의해 주세요.",
            ErrorCategory.NETWORK: "네트워크 연결에 문제가 있습니다. 연결을 확인해 주세요.",
            ErrorCategory.PERFORMANCE: "요청 처리 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요."
        }

        return user_messages.get(category, "오류가 발생했습니다. 잠시 후 다시 시도해 주세요.")

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """에러 통계 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [
            record for record in self.error_records
            if record.context.timestamp >= cutoff_time
        ]

        stats = {
            "total_errors": len(recent_errors),
            "by_category": {},
            "by_severity": {},
            "recovery_success_rate": 0,
            "most_common_errors": []
        }

        # 카테고리별 통계
        for category in ErrorCategory:
            count = len([r for r in recent_errors if r.category == category])
            stats["by_category"][category.value] = count

        # 심각도별 통계
        for severity in ErrorSeverity:
            count = len([r for r in recent_errors if r.severity == severity])
            stats["by_severity"][severity.value] = count

        # 복구 성공률
        recovery_attempted = len([r for r in recent_errors if r.recovery_attempted])
        recovery_successful = len([r for r in recent_errors if r.recovery_successful])

        if recovery_attempted > 0:
            stats["recovery_success_rate"] = recovery_successful / recovery_attempted

        # 가장 빈번한 에러
        error_types = {}
        for record in recent_errors:
            error_type = type(record.exception).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1

        stats["most_common_errors"] = sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return stats


# 글로벌 에러 핸들러 인스턴스
global_error_handler = ErrorHandler()


@asynccontextmanager
async def error_context(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[ErrorContext] = None,
    attempt_recovery: bool = True,
    raise_on_error: bool = True
):
    """에러 컨텍스트 매니저"""
    try:
        yield
    except Exception as e:
        error_record = await global_error_handler.handle_error(
            exception=e,
            category=category,
            severity=severity,
            context=context,
            attempt_recovery=attempt_recovery
        )

        if raise_on_error and not error_record.recovery_successful:
            raise e


@contextmanager
def sync_error_context(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[ErrorContext] = None,
    raise_on_error: bool = True
):
    """동기 에러 컨텍스트 매니저"""
    try:
        yield
    except Exception as e:
        # 동기 컨텍스트에서는 기본 로깅만 수행
        logger.error(
            f"Error in {category.value}: {str(e)}",
            extra={
                "category": category.value,
                "severity": severity.value,
                "exception_type": type(e).__name__
            }
        )

        if raise_on_error:
            raise e


def handle_unhandled_exception(
    exception_type,
    exception_value,
    exception_traceback
):
    """처리되지 않은 예외 핸들러"""
    logger.critical(
        "Unhandled exception occurred",
        exc_info=(exception_type, exception_value, exception_traceback)
    )

    # 중요한 시스템 에러이므로 알림 발송
    # (실제 구현에서는 이메일, Slack 등으로 알림)


# 기본 복구 전략들
async def database_recovery_strategy(exception: Exception, context: ErrorContext) -> bool:
    """데이터베이스 복구 전략"""
    # 연결 재시도, 트랜잭션 롤백 등
    logger.info(f"Attempting database recovery for error in context {context.error_id}")
    await asyncio.sleep(1)  # 잠시 대기 후 재시도
    return False  # 실제 구현에서는 복구 로직 수행


async def external_api_recovery_strategy(exception: Exception, context: ErrorContext) -> bool:
    """외부 API 복구 전략"""
    # 재시도, 대체 API 사용 등
    logger.info(f"Attempting external API recovery for error in context {context.error_id}")
    await asyncio.sleep(2)  # 잠시 대기 후 재시도
    return False  # 실제 구현에서는 복구 로직 수행


# 기본 전략 등록
global_error_handler.register_recovery_strategy(
    ErrorCategory.DATABASE,
    database_recovery_strategy
)

global_error_handler.register_recovery_strategy(
    ErrorCategory.EXTERNAL_API,
    external_api_recovery_strategy
)