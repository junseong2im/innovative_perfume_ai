"""
통합 예외 처리 모듈
"""

from typing import Optional, Dict, Any
import traceback
import logging

logger = logging.getLogger(__name__)

class FragranceAIException(Exception):
    """기본 예외 클래스"""
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict] = None):
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)

class APIException(FragranceAIException):
    """API 관련 예외"""
    pass

class ModelException(FragranceAIException):
    """모델 관련 예외"""
    pass

class SearchException(FragranceAIException):
    """검색 관련 예외"""
    pass

class ValidationException(FragranceAIException):
    """검증 관련 예외"""
    pass

class VectorStoreException(FragranceAIException):
    """벡터 스토어 관련 예외"""
    pass

def global_error_handler(exception: Exception) -> Dict[str, Any]:
    """전역 에러 핸들러"""
    if isinstance(exception, FragranceAIException):
        return {
            "error": True,
            "code": exception.code,
            "message": exception.message,
            "details": exception.details
        }
    else:
        logger.error(f"Unexpected error: {exception}")
        logger.error(traceback.format_exc())
        return {
            "error": True,
            "code": "INTERNAL_ERROR",
            "message": "Internal server error occurred",
            "details": {"error": str(exception)}
        }

async def handle_exceptions_async(func):
    """비동기 예외 처리 데코레이터"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            return global_error_handler(e)
    return wrapper