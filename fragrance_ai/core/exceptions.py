from typing import Dict, Any, Optional, List
from fastapi import HTTPException
import traceback
import sys
from enum import Enum


class ErrorCode(str, Enum):
    """에러 코드 정의"""
    
    # 시스템 에러 (1000-1999)
    SYSTEM_ERROR = "SYS_1000"
    DATABASE_ERROR = "SYS_1001"
    REDIS_ERROR = "SYS_1002"
    MODEL_LOADING_ERROR = "SYS_1003"
    VECTOR_STORE_ERROR = "SYS_1004"
    CONFIGURATION_ERROR = "SYS_1005"
    RESOURCE_EXHAUSTED = "SYS_1006"
    
    # 인증/보안 에러 (2000-2999)
    AUTHENTICATION_FAILED = "AUTH_2000"
    AUTHORIZATION_FAILED = "AUTH_2001"
    TOKEN_EXPIRED = "AUTH_2002"
    INVALID_API_KEY = "AUTH_2003"
    RATE_LIMIT_EXCEEDED = "AUTH_2004"
    
    # 요청 검증 에러 (3000-3999)
    VALIDATION_ERROR = "VAL_3000"
    INVALID_INPUT = "VAL_3001"
    MISSING_PARAMETER = "VAL_3002"
    INVALID_FORMAT = "VAL_3003"
    BATCH_SIZE_EXCEEDED = "VAL_3004"
    
    # AI 모델 에러 (4000-4999)
    MODEL_INFERENCE_ERROR = "AI_4000"
    EMBEDDING_ERROR = "AI_4001"
    GENERATION_ERROR = "AI_4002"
    EVALUATION_ERROR = "AI_4003"
    TRAINING_ERROR = "AI_4004"
    MODEL_NOT_READY = "AI_4005"
    
    # 비즈니스 로직 에러 (5000-5999)
    SEARCH_FAILED = "BIZ_5000"
    RECIPE_GENERATION_FAILED = "BIZ_5001"
    BLEND_CALCULATION_FAILED = "BIZ_5002"
    MOOD_ANALYSIS_FAILED = "BIZ_5003"
    EVALUATION_FAILED = "BIZ_5004"


class FragranceAIException(Exception):
    """기본 예외 클래스"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        user_message: Optional[str] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.user_message = user_message or message
        self.traceback = traceback.format_exc()
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """예외 정보를 딕셔너리로 변환"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
            "traceback": self.traceback if sys.exc_info()[0] else None
        }
    
    def to_http_exception(self, status_code: int = 500) -> HTTPException:
        """HTTPException으로 변환"""
        return HTTPException(
            status_code=status_code,
            detail={
                "error_code": self.error_code.value,
                "message": self.user_message,
                "details": self.details
            }
        )


class SystemException(FragranceAIException):
    """시스템 관련 예외"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            cause=cause,
            user_message="시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        )


class DatabaseException(SystemException):
    """데이터베이스 관련 예외"""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.DATABASE_ERROR,
            details=details,
            cause=cause
        )


class ModelException(FragranceAIException):
    """AI 모델 관련 예외"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.MODEL_INFERENCE_ERROR,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            cause=cause,
            user_message="AI 모델 처리 중 오류가 발생했습니다."
        )


class ValidationException(FragranceAIException):
    """입력 검증 관련 예외"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        validation_details = details or {}
        if field:
            validation_details["field"] = field
        if value is not None:
            validation_details["invalid_value"] = value
            
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=validation_details,
            user_message=f"입력값이 올바르지 않습니다: {message}"
        )


class AuthenticationException(FragranceAIException):
    """인증 관련 예외"""
    
    def __init__(
        self,
        message: str = "인증에 실패했습니다",
        error_code: ErrorCode = ErrorCode.AUTHENTICATION_FAILED,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            user_message=message
        )


class BusinessLogicException(FragranceAIException):
    """비즈니스 로직 관련 예외"""
    
    def __init__(
        self,
        message: str,
        error_code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            user_message=user_message or message
        )


class ErrorHandler:
    """에러 처리 유틸리티 클래스"""
    
    @staticmethod
    def wrap_exception(
        func_name: str,
        original_exception: Exception,
        error_code: ErrorCode,
        user_message: Optional[str] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ) -> FragranceAIException:
        """기존 예외를 FragranceAIException으로 감싸기"""
        
        details = {
            "function": func_name,
            "original_error_type": type(original_exception).__name__,
            "original_error_message": str(original_exception)
        }
        
        if additional_details:
            details.update(additional_details)
        
        return FragranceAIException(
            message=f"Error in {func_name}: {str(original_exception)}",
            error_code=error_code,
            details=details,
            cause=original_exception,
            user_message=user_message
        )
    
    @staticmethod
    def handle_model_error(
        operation: str,
        original_exception: Exception,
        model_name: Optional[str] = None
    ) -> ModelException:
        """AI 모델 에러 처리"""
        
        details = {
            "operation": operation,
            "original_error": str(original_exception),
            "error_type": type(original_exception).__name__
        }
        
        if model_name:
            details["model_name"] = model_name
        
        # 특정 에러 타입에 따른 분류
        error_code = ErrorCode.MODEL_INFERENCE_ERROR
        if "embedding" in operation.lower():
            error_code = ErrorCode.EMBEDDING_ERROR
        elif "generation" in operation.lower():
            error_code = ErrorCode.GENERATION_ERROR
        elif "evaluation" in operation.lower():
            error_code = ErrorCode.EVALUATION_ERROR
        elif "training" in operation.lower():
            error_code = ErrorCode.TRAINING_ERROR
        
        return ModelException(
            message=f"Model operation failed: {operation}",
            error_code=error_code,
            details=details,
            cause=original_exception
        )
    
    @staticmethod
    def handle_validation_errors(errors: List[Dict[str, Any]]) -> ValidationException:
        """Pydantic 검증 에러 처리"""
        
        error_messages = []
        details = {"validation_errors": errors}
        
        for error in errors:
            field = ".".join(str(loc) for loc in error.get("loc", []))
            message = error.get("msg", "")
            error_messages.append(f"{field}: {message}")
        
        combined_message = "; ".join(error_messages)
        
        return ValidationException(
            message=combined_message,
            details=details
        )
    
    @staticmethod
    def create_http_error_response(
        exception: FragranceAIException,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """HTTP 에러 응답 생성"""
        
        response = {
            "error": True,
            "error_code": exception.error_code.value,
            "message": exception.user_message,
            "timestamp": traceback.format_exc(),
        }
        
        if request_id:
            response["request_id"] = request_id
            
        if exception.details:
            response["details"] = exception.details
        
        return response


def safe_execute(
    operation: str,
    func,
    *args,
    error_code: ErrorCode = ErrorCode.SYSTEM_ERROR,
    user_message: Optional[str] = None,
    **kwargs
):
    """안전한 함수 실행 래퍼"""
    
    try:
        return func(*args, **kwargs)
    except FragranceAIException:
        # 이미 처리된 예외는 그대로 전파
        raise
    except Exception as e:
        # 예상하지 못한 예외를 래핑
        raise ErrorHandler.wrap_exception(
            func_name=operation,
            original_exception=e,
            error_code=error_code,
            user_message=user_message
        )


def validate_required_params(**params):
    """필수 파라미터 검증"""
    missing_params = []
    
    for param_name, param_value in params.items():
        if param_value is None or (isinstance(param_value, str) and not param_value.strip()):
            missing_params.append(param_name)
    
    if missing_params:
        raise ValidationException(
            message=f"Required parameters missing: {', '.join(missing_params)}",
            details={"missing_parameters": missing_params}
        )