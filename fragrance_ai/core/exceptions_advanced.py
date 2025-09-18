"""
고급 예외 처리 및 에러 관리 시스템
상용화 레벨의 에러 추적, 알림 및 복구 메커니즘
"""

import uuid
import traceback
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Type
from enum import Enum
import logging
import json
from contextlib import contextmanager

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog

# 구조화된 로깅 설정
logger = structlog.get_logger(__name__)

class ErrorSeverity(str, Enum):
    """에러 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(str, Enum):
    """에러 카테고리"""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    SYSTEM = "system"
    AI_MODEL = "ai_model"
    RATE_LIMIT = "rate_limit"
    CONFIGURATION = "configuration"

class ErrorCode(str, Enum):
    """표준 에러 코드"""
    # Authentication & Authorization
    INVALID_TOKEN = "AUTH_001"
    EXPIRED_TOKEN = "AUTH_002"
    INSUFFICIENT_PERMISSIONS = "AUTH_003"
    ACCOUNT_LOCKED = "AUTH_004"

    # Validation
    INVALID_INPUT = "VAL_001"
    MISSING_REQUIRED_FIELD = "VAL_002"
    INVALID_FORMAT = "VAL_003"
    VALUE_OUT_OF_RANGE = "VAL_004"

    # Business Logic
    RESOURCE_NOT_FOUND = "BIZ_001"
    RESOURCE_ALREADY_EXISTS = "BIZ_002"
    OPERATION_NOT_ALLOWED = "BIZ_003"
    QUOTA_EXCEEDED = "BIZ_004"
    FRAGRANCE_GENERATION_FAILED = "BIZ_005"

    # External Services
    EXTERNAL_API_UNAVAILABLE = "EXT_001"
    EXTERNAL_API_ERROR = "EXT_002"
    PAYMENT_SERVICE_ERROR = "EXT_003"
    EMAIL_SERVICE_ERROR = "EXT_004"

    # Database
    DATABASE_CONNECTION_ERROR = "DB_001"
    DATABASE_QUERY_ERROR = "DB_002"
    DATABASE_CONSTRAINT_VIOLATION = "DB_003"

    # System
    INTERNAL_SERVER_ERROR = "SYS_001"
    SERVICE_UNAVAILABLE = "SYS_002"
    TIMEOUT_ERROR = "SYS_003"
    MEMORY_ERROR = "SYS_004"

    # AI Model
    MODEL_LOADING_ERROR = "AI_001"
    MODEL_INFERENCE_ERROR = "AI_002"
    MODEL_TIMEOUT = "AI_003"
    EMBEDDING_ERROR = "AI_004"

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = "RATE_001"
    API_QUOTA_EXCEEDED = "RATE_002"

class FragranceAIBaseException(Exception):
    """기본 Fragrance AI 예외 클래스"""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
        cause: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)

        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.user_message = user_message or self._get_default_user_message()
        self.cause = cause
        self.context = context or {}

        # 고유 식별자 생성
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.utcnow()

        # 스택 트레이스 캡처
        self.stack_trace = traceback.format_exc()

        # 구조화된 로깅
        self._log_error()

    def _get_default_user_message(self) -> str:
        """기본 사용자 메시지 생성"""
        user_messages = {
            ErrorCode.INVALID_TOKEN: "인증 토큰이 유효하지 않습니다.",
            ErrorCode.EXPIRED_TOKEN: "인증 토큰이 만료되었습니다.",
            ErrorCode.INSUFFICIENT_PERMISSIONS: "해당 작업을 수행할 권한이 없습니다.",
            ErrorCode.INVALID_INPUT: "입력 데이터가 올바르지 않습니다.",
            ErrorCode.RESOURCE_NOT_FOUND: "요청한 리소스를 찾을 수 없습니다.",
            ErrorCode.QUOTA_EXCEEDED: "사용 한도를 초과했습니다.",
            ErrorCode.FRAGRANCE_GENERATION_FAILED: "향수 생성 중 오류가 발생했습니다.",
            ErrorCode.EXTERNAL_API_UNAVAILABLE: "외부 서비스에 일시적으로 연결할 수 없습니다.",
            ErrorCode.DATABASE_CONNECTION_ERROR: "데이터베이스 연결에 문제가 발생했습니다.",
            ErrorCode.INTERNAL_SERVER_ERROR: "서버 내부 오류가 발생했습니다.",
            ErrorCode.MODEL_LOADING_ERROR: "AI 모델을 로드하는 중 오류가 발생했습니다.",
            ErrorCode.RATE_LIMIT_EXCEEDED: "요청 빈도 제한을 초과했습니다.",
        }
        return user_messages.get(self.error_code, "오류가 발생했습니다.")

    def _log_error(self):
        """구조화된 에러 로깅"""
        log_data = {
            "error_id": self.error_id,
            "error_code": self.error_code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "stack_trace": self.stack_trace if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] else None
        }

        if self.cause:
            log_data["cause"] = str(self.cause)

        # 심각도에 따른 로그 레벨 결정
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical("Critical error occurred", **log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error("High severity error occurred", **log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning("Medium severity error occurred", **log_data)
        else:
            logger.info("Low severity error occurred", **log_data)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

# 특화된 예외 클래스들
class AuthenticationError(FragranceAIBaseException):
    """인증 관련 예외"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INVALID_TOKEN, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class AuthorizationError(FragranceAIBaseException):
    """인가 관련 예외"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INSUFFICIENT_PERMISSIONS, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class ValidationError(FragranceAIBaseException):
    """입력 검증 관련 예외"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INVALID_INPUT, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs
        )

class BusinessLogicError(FragranceAIBaseException):
    """비즈니스 로직 관련 예외"""

    def __init__(self, message: str, error_code: ErrorCode, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class ExternalServiceError(FragranceAIBaseException):
    """외부 서비스 관련 예외"""

    def __init__(self, message: str, service_name: str, error_code: ErrorCode = ErrorCode.EXTERNAL_API_ERROR, **kwargs):
        details = kwargs.get('details', {})
        details['service_name'] = service_name
        kwargs['details'] = details

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class DatabaseError(FragranceAIBaseException):
    """데이터베이스 관련 예외"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class AIModelError(FragranceAIBaseException):
    """AI 모델 관련 예외"""

    def __init__(self, message: str, model_name: str, error_code: ErrorCode = ErrorCode.MODEL_INFERENCE_ERROR, **kwargs):
        details = kwargs.get('details', {})
        details['model_name'] = model_name
        kwargs['details'] = details

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.AI_MODEL,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )

class RateLimitError(FragranceAIBaseException):
    """요청 제한 관련 예외"""

    def __init__(self, message: str, limit: int, window: str, error_code: ErrorCode = ErrorCode.RATE_LIMIT_EXCEEDED, **kwargs):
        details = kwargs.get('details', {})
        details.update({'limit': limit, 'window': window})
        kwargs['details'] = details

        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )

class SystemError(FragranceAIBaseException):
    """시스템 관련 예외"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR, **kwargs):
        super().__init__(
            message=message,
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )

# 에러 처리 유틸리티 클래스
class ErrorHandler:
    """중앙집중식 에러 처리기"""

    def __init__(self):
        self.error_alerts = []
        self.recovery_strategies = {}
        self.error_statistics = {}

    def handle_exception(self, exc: Exception, request: Optional[Request] = None) -> JSONResponse:
        """예외를 HTTP 응답으로 변환"""

        # Fragrance AI 커스텀 예외 처리
        if isinstance(exc, FragranceAIBaseException):
            return self._handle_custom_exception(exc, request)

        # FastAPI 기본 예외 처리
        if isinstance(exc, HTTPException):
            return self._handle_http_exception(exc, request)

        # 요청 검증 예외 처리
        if isinstance(exc, RequestValidationError):
            return self._handle_validation_exception(exc, request)

        # 기타 예외를 시스템 오류로 처리
        return self._handle_unknown_exception(exc, request)

    def _handle_custom_exception(self, exc: FragranceAIBaseException, request: Optional[Request]) -> JSONResponse:
        """커스텀 예외 처리"""

        # 에러 통계 업데이트
        self._update_error_statistics(exc)

        # 심각한 에러의 경우 알림 발송
        if exc.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_error_alert(exc, request)

        # HTTP 상태 코드 결정
        status_code = self._get_http_status_code(exc)

        # 응답 본문 구성
        response_data = {
            "success": False,
            "error": {
                "code": exc.error_code.value,
                "message": exc.user_message,
                "category": exc.category.value,
                "error_id": exc.error_id,
                "timestamp": exc.timestamp.isoformat()
            }
        }

        # 개발 환경에서는 상세 정보 포함
        if self._is_development_mode():
            response_data["error"]["details"] = exc.details
            response_data["error"]["internal_message"] = exc.message

        return JSONResponse(
            status_code=status_code,
            content=response_data,
            headers={"X-Error-ID": exc.error_id}
        )

    def _handle_http_exception(self, exc: HTTPException, request: Optional[Request]) -> JSONResponse:
        """HTTP 예외 처리"""
        error_id = str(uuid.uuid4())

        response_data = {
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        return JSONResponse(
            status_code=exc.status_code,
            content=response_data,
            headers={"X-Error-ID": error_id}
        )

    def _handle_validation_exception(self, exc: RequestValidationError, request: Optional[Request]) -> JSONResponse:
        """요청 검증 예외 처리"""
        error_id = str(uuid.uuid4())

        # 검증 에러 세부사항 추출
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })

        response_data = {
            "success": False,
            "error": {
                "code": ErrorCode.INVALID_INPUT.value,
                "message": "입력 데이터 검증에 실패했습니다.",
                "validation_errors": errors,
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_data,
            headers={"X-Error-ID": error_id}
        )

    def _handle_unknown_exception(self, exc: Exception, request: Optional[Request]) -> JSONResponse:
        """알 수 없는 예외 처리"""
        error_id = str(uuid.uuid4())

        # 시스템 에러로 래핑
        system_error = SystemError(
            message=f"Unexpected error: {str(exc)}",
            error_code=ErrorCode.INTERNAL_SERVER_ERROR,
            cause=exc,
            context={"request_path": request.url.path if request else None}
        )

        response_data = {
            "success": False,
            "error": {
                "code": ErrorCode.INTERNAL_SERVER_ERROR.value,
                "message": "서버 내부 오류가 발생했습니다.",
                "error_id": error_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_data,
            headers={"X-Error-ID": error_id}
        )

    def _get_http_status_code(self, exc: FragranceAIBaseException) -> int:
        """예외에 따른 HTTP 상태 코드 반환"""
        status_mapping = {
            ErrorCategory.AUTHENTICATION: status.HTTP_401_UNAUTHORIZED,
            ErrorCategory.AUTHORIZATION: status.HTTP_403_FORBIDDEN,
            ErrorCategory.VALIDATION: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCategory.BUSINESS_LOGIC: status.HTTP_400_BAD_REQUEST,
            ErrorCategory.EXTERNAL_SERVICE: status.HTTP_502_BAD_GATEWAY,
            ErrorCategory.DATABASE: status.HTTP_503_SERVICE_UNAVAILABLE,
            ErrorCategory.SYSTEM: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCategory.AI_MODEL: status.HTTP_503_SERVICE_UNAVAILABLE,
            ErrorCategory.RATE_LIMIT: status.HTTP_429_TOO_MANY_REQUESTS,
        }

        # 특정 에러 코드에 따른 매핑
        specific_mapping = {
            ErrorCode.RESOURCE_NOT_FOUND: status.HTTP_404_NOT_FOUND,
            ErrorCode.RESOURCE_ALREADY_EXISTS: status.HTTP_409_CONFLICT,
            ErrorCode.EXPIRED_TOKEN: status.HTTP_401_UNAUTHORIZED,
        }

        return specific_mapping.get(exc.error_code,
                                  status_mapping.get(exc.category, status.HTTP_500_INTERNAL_SERVER_ERROR))

    def _update_error_statistics(self, exc: FragranceAIBaseException):
        """에러 통계 업데이트"""
        key = f"{exc.category.value}_{exc.error_code.value}"
        if key not in self.error_statistics:
            self.error_statistics[key] = {"count": 0, "last_occurrence": None}

        self.error_statistics[key]["count"] += 1
        self.error_statistics[key]["last_occurrence"] = exc.timestamp

    def _send_error_alert(self, exc: FragranceAIBaseException, request: Optional[Request]):
        """에러 알림 발송"""
        alert_data = {
            "error_id": exc.error_id,
            "severity": exc.severity.value,
            "message": exc.message,
            "request_path": request.url.path if request else None,
            "timestamp": exc.timestamp.isoformat()
        }

        # 실제 구현에서는 Slack, 이메일, PagerDuty 등으로 알림 발송
        logger.critical("Critical error alert", **alert_data)

    def _is_development_mode(self) -> bool:
        """개발 모드 여부 확인"""
        # 실제 구현에서는 환경 변수 확인
        return True

# 에러 복구 데코레이터
def with_error_recovery(max_retries: int = 3, backoff_seconds: float = 1.0):
    """에러 복구 기능이 있는 데코레이터"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        # 복구 가능한 에러인지 확인
                        if isinstance(e, (ExternalServiceError, DatabaseError, AIModelError)):
                            import asyncio
                            await asyncio.sleep(backoff_seconds * (2 ** attempt))
                            continue

                    # 최대 재시도 횟수 도달 또는 복구 불가능한 에러
                    raise last_exception

            # 여기에 도달하면 안 됨
            raise last_exception

        return wrapper
    return decorator

# 컨텍스트 매니저를 통한 에러 처리
@contextmanager
def error_context(operation: str, **context_data):
    """에러 컨텍스트 매니저"""
    try:
        yield
    except Exception as e:
        if not isinstance(e, FragranceAIBaseException):
            # 일반 예외를 시스템 에러로 래핑
            raise SystemError(
                message=f"Error in {operation}: {str(e)}",
                cause=e,
                context=context_data
            )
        else:
            # 기존 예외에 컨텍스트 추가
            e.context.update(context_data)
            raise

# 글로벌 에러 핸들러 인스턴스
global_error_handler = ErrorHandler()