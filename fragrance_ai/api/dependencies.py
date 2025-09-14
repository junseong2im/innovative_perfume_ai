from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import jwt
import logging
from datetime import datetime, timedelta
import hashlib
import hmac

from ..core.config import settings
from ..services.search_service import SearchService
from ..services.generation_service import GenerationService

logger = logging.getLogger(__name__)
security = HTTPBearer()


class AuthenticationError(HTTPException):
    """인증 오류"""
    def __init__(self, detail: str = "인증에 실패했습니다"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(HTTPException):
    """인가 오류"""
    def __init__(self, detail: str = "권한이 없습니다"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


def verify_api_key(api_key: str) -> bool:
    """API 키 검증"""
    try:
        # Simple API key validation - in production use proper key management
        if not api_key:
            raise AuthenticationError("API 키가 필요합니다")
        
        # Check if it's the master admin key (for development/demo)
        if api_key == "demo_admin_key_12345":  # Change in production
            return True
        
        # In production, validate against database or key management service
        valid_keys = [
            "fragrance_ai_key_001",
            "fragrance_ai_key_002",
            "fragrance_ai_key_003"
        ]
        
        if api_key not in valid_keys:
            raise AuthenticationError("유효하지 않은 API 키입니다")
        
        return True
        
    except Exception as e:
        logger.error(f"API key verification failed: {e}")
        raise AuthenticationError("API 키 검증에 실패했습니다")


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """액세스 토큰 생성"""
    try:
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            settings.secret_key, 
            algorithm="HS256"
        )
        
        return encoded_jwt
        
    except Exception as e:
        logger.error(f"Token creation failed: {e}")
        raise AuthenticationError("토큰 생성에 실패했습니다")


def verify_token(token: str) -> Dict[str, Any]:
    """토큰 검증"""
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=["HS256"]
        )
        
        username = payload.get("sub")
        if username is None:
            raise AuthenticationError("유효하지 않은 토큰입니다")
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("토큰이 만료되었습니다")
    except jwt.JWTError as e:
        logger.error(f"JWT verification failed: {e}")
        raise AuthenticationError("토큰 검증에 실패했습니다")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """현재 사용자 정보 조회"""
    try:
        token = credentials.credentials
        payload = verify_token(token)
        
        # In production, fetch user info from database
        user_info = {
            "username": payload.get("sub"),
            "user_id": payload.get("user_id"),
            "roles": payload.get("roles", []),
            "permissions": payload.get("permissions", [])
        }
        
        return user_info
        
    except Exception as e:
        logger.error(f"User authentication failed: {e}")
        raise AuthenticationError()


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """관리자 권한 필요"""
    if "admin" not in current_user.get("roles", []):
        raise AuthorizationError("관리자 권한이 필요합니다")
    
    return current_user


def require_permission(permission: str):
    """특정 권한 필요"""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_permissions = current_user.get("permissions", [])
        
        if permission not in user_permissions and "admin" not in current_user.get("roles", []):
            raise AuthorizationError(f"'{permission}' 권한이 필요합니다")
        
        return current_user
    
    return permission_checker


def rate_limit_key(request_info: Dict[str, Any]) -> str:
    """Rate limiting을 위한 키 생성"""
    try:
        # Combine IP, user, and endpoint for rate limiting
        key_parts = [
            request_info.get("client_ip", "unknown"),
            request_info.get("user_id", "anonymous"),
            request_info.get("endpoint", "unknown")
        ]
        
        key_string = ":".join(str(part) for part in key_parts)
        
        # Create hash for shorter key
        return hashlib.md5(key_string.encode()).hexdigest()
        
    except Exception as e:
        logger.error(f"Rate limit key generation failed: {e}")
        return "default"


def create_webhook_signature(payload: bytes, secret: str) -> str:
    """웹훅 서명 생성"""
    try:
        signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
        
    except Exception as e:
        logger.error(f"Webhook signature creation failed: {e}")
        raise


def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """웹훅 서명 검증"""
    try:
        expected_signature = create_webhook_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)
        
    except Exception as e:
        logger.error(f"Webhook signature verification failed: {e}")
        return False


class QuotaManager:
    """API 할당량 관리"""
    
    def __init__(self):
        self.user_quotas = {}  # In production, use Redis or database
        self.default_quotas = {
            "requests_per_day": 1000,
            "generations_per_day": 100,
            "searches_per_day": 500
        }
    
    def check_quota(self, user_id: str, operation_type: str) -> bool:
        """할당량 확인"""
        try:
            today = datetime.now().date()
            user_key = f"{user_id}:{today}"
            
            if user_key not in self.user_quotas:
                self.user_quotas[user_key] = self.default_quotas.copy()
            
            quota_key = f"{operation_type}_per_day"
            if quota_key not in self.user_quotas[user_key]:
                return True
            
            return self.user_quotas[user_key][quota_key] > 0
            
        except Exception as e:
            logger.error(f"Quota check failed: {e}")
            return True  # Allow on error
    
    def consume_quota(self, user_id: str, operation_type: str, amount: int = 1):
        """할당량 소모"""
        try:
            today = datetime.now().date()
            user_key = f"{user_id}:{today}"
            quota_key = f"{operation_type}_per_day"
            
            if user_key in self.user_quotas and quota_key in self.user_quotas[user_key]:
                self.user_quotas[user_key][quota_key] = max(0, 
                    self.user_quotas[user_key][quota_key] - amount)
                
        except Exception as e:
            logger.error(f"Quota consumption failed: {e}")
    
    def get_user_quota(self, user_id: str) -> Dict[str, int]:
        """사용자 할당량 조회"""
        try:
            today = datetime.now().date()
            user_key = f"{user_id}:{today}"
            
            return self.user_quotas.get(user_key, self.default_quotas.copy())
            
        except Exception as e:
            logger.error(f"Quota retrieval failed: {e}")
            return self.default_quotas.copy()


# Global quota manager instance
quota_manager = QuotaManager()

# 서비스 인스턴스들 (싱글톤 패턴)
_search_service_instance: Optional[SearchService] = None
_generation_service_instance: Optional[GenerationService] = None

async def get_search_service() -> SearchService:
    """검색 서비스 의존성"""
    global _search_service_instance
    
    if _search_service_instance is None:
        _search_service_instance = SearchService()
        await _search_service_instance.initialize()
    
    return _search_service_instance

async def get_generation_service() -> GenerationService:
    """생성 서비스 의존성"""
    global _generation_service_instance
    
    if _generation_service_instance is None:
        _generation_service_instance = GenerationService()
        await _generation_service_instance.initialize()
    
    return _generation_service_instance


def check_api_quota(operation_type: str = "requests"):
    """API 할당량 확인 의존성"""
    def quota_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_id = current_user.get("user_id", "anonymous")
        
        if not quota_manager.check_quota(user_id, operation_type):
            raise HTTPException(
                status_code=429,
                detail=f"일일 {operation_type} 할당량을 초과했습니다"
            )
        
        # Consume quota
        quota_manager.consume_quota(user_id, operation_type)
        
        return current_user
    
    return quota_checker


def validate_request_size(max_size_mb: float = 10.0):
    """요청 크기 제한"""
    def size_validator(request):
        content_length = request.headers.get("content-length")
        
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                raise HTTPException(
                    status_code=413,
                    detail=f"요청 크기가 너무 큽니다 (최대 {max_size_mb}MB)"
                )
        
        return True
    
    return size_validator