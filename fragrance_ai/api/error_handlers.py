from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import traceback
import time
from typing import Dict, Any, Optional
import uuid

from ..core.exceptions import (
    FragranceAIException,
    SystemException,
    ModelException,
    ValidationException,
    AuthenticationException,
    BusinessLogicException,
    ErrorHandler,
    ErrorCode
)
from ..core.logging_config import get_logger, performance_logger

logger = get_logger(__name__)


class GlobalExceptionHandler:
    """전역 예외 처리기"""
    
    @staticmethod
    def generate_request_id() -> str:
        """요청 ID 생성"""
        return str(uuid.uuid4())
    
    @staticmethod
    async def fragrance_ai_exception_handler(
        request: Request,
        exc: FragranceAIException
    ) -> JSONResponse:
        """FragranceAI 커스텀 예외 처리"""
        
        request_id = getattr(request.state, 'request_id', GlobalExceptionHandler.generate_request_id())
        
        # 에러 로깅
        logger.error(
            f"FragranceAI Exception: {exc.error_code.value}",
            extra={
                "request_id": request_id,
                "error_code": exc.error_code.value,
                "message": exc.message,
                "details": exc.details,
                "url": str(request.url),
                "method": request.method,
                "user_agent": request.headers.get("user-agent"),
                "traceback": exc.traceback
            }
        )
        
        # 상태 코드 결정
        status_code = GlobalExceptionHandler._get_status_code(exc.error_code)
        
        # 응답 생성
        response_data = ErrorHandler.create_http_error_response(exc, request_id)
        
        return JSONResponse(
            status_code=status_code,
            content=response_data
        )
    
    @staticmethod
    async def http_exception_handler(
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """HTTP 예외 처리"""
        
        request_id = getattr(request.state, 'request_id', GlobalExceptionHandler.generate_request_id())
        
        logger.warning(
            f"HTTP Exception: {exc.status_code}",
            extra={
                "request_id": request_id,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "url": str(request.url),
                "method": request.method
            }
        )
        
        response_data = {
            "error": True,
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            "request_id": request_id,
            "timestamp": time.time()
        }
        
        return JSONResponse(
            status_code=exc.status_code,
            content=response_data
        )
    
    @staticmethod
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """요청 검증 예외 처리"""
        
        request_id = getattr(request.state, 'request_id', GlobalExceptionHandler.generate_request_id())
        
        # 검증 에러를 FragranceAI 형식으로 변환
        validation_exc = ErrorHandler.handle_validation_errors(exc.errors())
        
        logger.warning(
            f"Validation Error: {len(exc.errors())} errors",
            extra={
                "request_id": request_id,
                "validation_errors": exc.errors(),
                "url": str(request.url),
                "method": request.method
            }
        )
        
        response_data = ErrorHandler.create_http_error_response(validation_exc, request_id)
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_data
        )
    
    @staticmethod
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """일반 예외 처리"""
        
        request_id = getattr(request.state, 'request_id', GlobalExceptionHandler.generate_request_id())
        
        # 예상하지 못한 에러를 시스템 에러로 래핑
        system_exc = SystemException(
            message=f"Unexpected error: {str(exc)}",
            details={
                "original_error_type": type(exc).__name__,
                "original_error_message": str(exc)
            },
            cause=exc
        )
        
        logger.critical(
            f"Unhandled Exception: {type(exc).__name__}",
            extra={
                "request_id": request_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "url": str(request.url),
                "method": request.method,
                "traceback": traceback.format_exc()
            }
        )
        
        response_data = ErrorHandler.create_http_error_response(system_exc, request_id)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_data
        )
    
    @staticmethod
    def _get_status_code(error_code: ErrorCode) -> int:
        """에러 코드에 따른 HTTP 상태 코드 반환"""
        
        status_mapping = {
            # 시스템 에러 -> 500
            ErrorCode.SYSTEM_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.DATABASE_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.REDIS_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.MODEL_LOADING_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.VECTOR_STORE_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.CONFIGURATION_ERROR: status.HTTP_500_INTERNAL_SERVER_ERROR,
            ErrorCode.RESOURCE_EXHAUSTED: status.HTTP_503_SERVICE_UNAVAILABLE,
            
            # 인증/보안 에러
            ErrorCode.AUTHENTICATION_FAILED: status.HTTP_401_UNAUTHORIZED,
            ErrorCode.AUTHORIZATION_FAILED: status.HTTP_403_FORBIDDEN,
            ErrorCode.TOKEN_EXPIRED: status.HTTP_401_UNAUTHORIZED,
            ErrorCode.INVALID_API_KEY: status.HTTP_401_UNAUTHORIZED,
            ErrorCode.RATE_LIMIT_EXCEEDED: status.HTTP_429_TOO_MANY_REQUESTS,
            
            # 요청 검증 에러 -> 400
            ErrorCode.VALIDATION_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.INVALID_INPUT: status.HTTP_400_BAD_REQUEST,
            ErrorCode.MISSING_PARAMETER: status.HTTP_400_BAD_REQUEST,
            ErrorCode.INVALID_FORMAT: status.HTTP_400_BAD_REQUEST,
            ErrorCode.BATCH_SIZE_EXCEEDED: status.HTTP_400_BAD_REQUEST,
            
            # AI 모델 에러 -> 422/503
            ErrorCode.MODEL_INFERENCE_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.EMBEDDING_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.GENERATION_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.EVALUATION_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.TRAINING_ERROR: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.MODEL_NOT_READY: status.HTTP_503_SERVICE_UNAVAILABLE,
            
            # 비즈니스 로직 에러 -> 422
            ErrorCode.SEARCH_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.RECIPE_GENERATION_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.BLEND_CALCULATION_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.MOOD_ANALYSIS_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
            ErrorCode.EVALUATION_FAILED: status.HTTP_422_UNPROCESSABLE_ENTITY,
        }
        
        return status_mapping.get(error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)


class RequestContextMiddleware:
    """요청 컨텍스트 미들웨어"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())
            scope["state"] = {"request_id": request_id}
            
            # 요청 시작 로깅
            logger.info(
                f"Request started: {scope['method']} {scope['path']}",
                extra={
                    "request_id": request_id,
                    "method": scope["method"],
                    "path": scope["path"],
                    "query_string": scope.get("query_string", b"").decode()
                }
            )
        
        await self.app(scope, receive, send)


def setup_error_handlers(app):
    """에러 핸들러 설정"""
    
    # 커스텀 예외 핸들러
    app.add_exception_handler(
        FragranceAIException,
        GlobalExceptionHandler.fragrance_ai_exception_handler
    )
    
    # HTTP 예외 핸들러
    app.add_exception_handler(
        HTTPException,
        GlobalExceptionHandler.http_exception_handler
    )
    
    app.add_exception_handler(
        StarletteHTTPException,
        GlobalExceptionHandler.http_exception_handler
    )
    
    # 요청 검증 예외 핸들러
    app.add_exception_handler(
        RequestValidationError,
        GlobalExceptionHandler.validation_exception_handler
    )
    
    # 일반 예외 핸들러
    app.add_exception_handler(
        Exception,
        GlobalExceptionHandler.general_exception_handler
    )
    
    # 요청 컨텍스트 미들웨어 추가
    app.add_middleware(RequestContextMiddleware)


class CircuitBreaker:
    """서킷 브레이커 패턴 구현"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        """서킷 브레이커를 통한 함수 호출"""
        
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise SystemException(
                    message="Service temporarily unavailable (Circuit Breaker)",
                    error_code=ErrorCode.RESOURCE_EXHAUSTED
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """리셋 시도 여부 확인"""
        return (
            self.last_failure_time is not None and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        """성공 시 처리"""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """실패 시 처리"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures",
                extra={
                    "failure_count": self.failure_count,
                    "threshold": self.failure_threshold
                }
            )


# 전역 서킷 브레이커 인스턴스들
model_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=300,  # 5분
    expected_exception=ModelException
)

database_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,  # 1분
    expected_exception=DatabaseException
)