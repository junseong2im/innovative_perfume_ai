from fastapi import HTTPException, Depends, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List, Callable
import time
import json
import uuid
from datetime import datetime, timedelta
from functools import wraps
import jwt

from ..core.config import settings
from ..core.security import (
    api_key_manager, rate_limiter, security_monitor, 
    ApiKey, AccessLevel, SecurityLevel
)
from ..core.logging_config import get_logger, performance_logger
from ..core.exceptions import (
    AuthenticationException, ValidationException, 
    ErrorCode, FragranceAIException
)
from ..core.monitoring import metrics_collector

logger = get_logger(__name__)
security = HTTPBearer(auto_error=False)


class EnhancedAuthenticationError(HTTPException):
    """강화된 인증 오류"""
    
    def __init__(
        self, 
        detail: str = "인증에 실패했습니다",
        error_code: str = "AUTH_FAILED",
        extra_info: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": True,
                "error_code": error_code,
                "message": detail,
                "timestamp": datetime.utcnow().isoformat(),
                "extra_info": extra_info or {}
            },
            headers={"WWW-Authenticate": "Bearer"}
        )


class EnhancedAuthorizationError(HTTPException):
    """강화된 인가 오류"""
    
    def __init__(
        self, 
        detail: str = "권한이 없습니다",
        error_code: str = "AUTH_INSUFFICIENT",
        required_permission: Optional[str] = None
    ):
        extra_info = {}
        if required_permission:
            extra_info["required_permission"] = required_permission
            
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": True,
                "error_code": error_code,
                "message": detail,
                "timestamp": datetime.utcnow().isoformat(),
                "extra_info": extra_info
            }
        )


class RateLimitError(HTTPException):
    """레이트 리미트 오류"""
    
    def __init__(
        self, 
        detail: str = "요청 한도를 초과했습니다",
        retry_after: Optional[int] = None,
        rate_limit_info: Optional[Dict[str, Any]] = None
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
            
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": True,
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": detail,
                "timestamp": datetime.utcnow().isoformat(),
                "rate_limit_info": rate_limit_info or {}
            },
            headers=headers
        )


class AuthenticationManager:
    """인증 관리자"""
    
    def __init__(self):
        self.session_store: Dict[str, Dict[str, Any]] = {}
        self.session_timeout_minutes = 30
    
    async def authenticate_api_key(
        self, 
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None
    ) -> Dict[str, Any]:
        """API 키 인증"""
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        
        try:
            if not credentials or not credentials.credentials:
                security_monitor.log_security_event(
                    event_type="missing_api_key",
                    severity=SecurityLevel.LOW,
                    source_ip=client_ip,
                    details={"endpoint": str(request.url.path)}
                )
                raise EnhancedAuthenticationError(
                    detail="API 키가 필요합니다",
                    error_code="MISSING_API_KEY"
                )
            
            api_key = credentials.credentials
            
            # API 키 검증
            api_key_obj = api_key_manager.validate_api_key(api_key)
            if not api_key_obj:
                security_monitor.log_security_event(
                    event_type="invalid_api_key",
                    severity=SecurityLevel.MEDIUM,
                    source_ip=client_ip,
                    details={
                        "api_key_prefix": api_key[:8] if len(api_key) > 8 else "***",
                        "endpoint": str(request.url.path)
                    }
                )
                raise EnhancedAuthenticationError(
                    detail="유효하지 않은 API 키입니다",
                    error_code="INVALID_API_KEY"
                )
            
            # 레이트 리미팅 확인
            rate_limit_key = f"{api_key_obj.user_id}:{client_ip}"
            is_allowed, rate_info = rate_limiter.is_allowed(
                rate_limit_key,
                limit_per_minute=api_key_obj.rate_limit // 60,
                limit_per_hour=api_key_obj.rate_limit
            )
            
            if not is_allowed:
                security_monitor.log_security_event(
                    event_type="rate_limit_exceeded",
                    severity=SecurityLevel.MEDIUM,
                    source_ip=client_ip,
                    user_id=api_key_obj.user_id,
                    details=rate_info
                )
                
                retry_after = None
                if "reset_time" in rate_info:
                    retry_after = int(rate_info["reset_time"] - time.time())
                
                raise RateLimitError(
                    detail=rate_info.get("reason", "요청 한도를 초과했습니다"),
                    retry_after=retry_after,
                    rate_limit_info=rate_info
                )
            
            # 요청 패턴 분석
            pattern_analysis = security_monitor.analyze_request_pattern(
                client_ip, str(request.url.path), user_agent
            )
            
            # 인증된 사용자 정보 생성
            user_info = {
                "user_id": api_key_obj.user_id,
                "api_key_id": api_key_obj.key_id,
                "access_level": api_key_obj.access_level.value,
                "permissions": api_key_obj.permissions,
                "rate_limit": api_key_obj.rate_limit,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "authenticated_at": datetime.utcnow().isoformat(),
                "session_id": str(uuid.uuid4()),
                "rate_limit_info": rate_info,
                "security_analysis": pattern_analysis
            }
            
            # 세션 저장
            self.session_store[user_info["session_id"]] = user_info
            
            # 성공적인 인증 로그
            auth_time = (time.time() - start_time) * 1000
            performance_logger.log_execution_time(
                operation="api_key_authentication",
                execution_time=auth_time,
                success=True,
                extra_data={
                    "user_id": api_key_obj.user_id,
                    "access_level": api_key_obj.access_level.value,
                    "client_ip": client_ip
                }
            )
            
            return user_info
            
        except (EnhancedAuthenticationError, RateLimitError):
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            
            security_monitor.log_security_event(
                event_type="authentication_error",
                severity=SecurityLevel.HIGH,
                source_ip=client_ip,
                details={"error": str(e)}
            )
            
            raise EnhancedAuthenticationError(
                detail="인증 처리 중 오류가 발생했습니다",
                error_code="AUTH_PROCESSING_ERROR"
            )
    
    async def authenticate_jwt(
        self,
        request: Request,
        credentials: HTTPAuthorizationCredentials
    ) -> Dict[str, Any]:
        """JWT 토큰 인증"""
        
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            token = credentials.credentials
            
            # JWT 토큰 검증
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=["HS256"]
            )
            
            user_id = payload.get("sub")
            if not user_id:
                raise EnhancedAuthenticationError(
                    detail="유효하지 않은 토큰입니다",
                    error_code="INVALID_JWT_PAYLOAD"
                )
            
            # 세션 정보 확인
            session_id = payload.get("session_id")
            if session_id and session_id in self.session_store:
                session_info = self.session_store[session_id]
                
                # 세션 타임아웃 확인
                auth_time = datetime.fromisoformat(session_info["authenticated_at"])
                if datetime.utcnow() - auth_time > timedelta(minutes=self.session_timeout_minutes):
                    del self.session_store[session_id]
                    raise EnhancedAuthenticationError(
                        detail="세션이 만료되었습니다",
                        error_code="SESSION_EXPIRED"
                    )
                
                return session_info
            
            # 새 세션 생성
            user_info = {
                "user_id": user_id,
                "username": payload.get("username"),
                "roles": payload.get("roles", []),
                "permissions": payload.get("permissions", []),
                "client_ip": client_ip,
                "user_agent": request.headers.get("user-agent", ""),
                "authenticated_at": datetime.utcnow().isoformat(),
                "session_id": str(uuid.uuid4()),
                "auth_method": "jwt"
            }
            
            # 세션 저장
            self.session_store[user_info["session_id"]] = user_info
            
            auth_time = (time.time() - start_time) * 1000
            performance_logger.log_execution_time(
                operation="jwt_authentication",
                execution_time=auth_time,
                success=True,
                extra_data={"user_id": user_id, "client_ip": client_ip}
            )
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            security_monitor.log_security_event(
                event_type="expired_jwt_token",
                severity=SecurityLevel.LOW,
                source_ip=client_ip
            )
            raise EnhancedAuthenticationError(
                detail="토큰이 만료되었습니다",
                error_code="JWT_EXPIRED"
            )
        except jwt.JWTError as e:
            security_monitor.log_security_event(
                event_type="invalid_jwt_token",
                severity=SecurityLevel.MEDIUM,
                source_ip=client_ip,
                details={"error": str(e)}
            )
            raise EnhancedAuthenticationError(
                detail="유효하지 않은 토큰입니다",
                error_code="JWT_INVALID"
            )
        except Exception as e:
            logger.error(f"JWT authentication error: {e}")
            raise EnhancedAuthenticationError(
                detail="토큰 인증 처리 중 오류가 발생했습니다",
                error_code="JWT_PROCESSING_ERROR"
            )
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_info in self.session_store.items():
            auth_time = datetime.fromisoformat(session_info["authenticated_at"])
            if current_time - auth_time > timedelta(minutes=self.session_timeout_minutes):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.session_store[session_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출"""
        
        # X-Forwarded-For 헤더 확인 (프록시/로드밸런서 고려)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # 클라이언트 IP 반환
        return request.client.host if request.client else "unknown"


class AuthorizationManager:
    """인가 관리자"""
    
    def __init__(self):
        self.permission_hierarchy = {
            AccessLevel.SUPER_ADMIN: ["*"],
            AccessLevel.ADMIN: [
                "admin.*", "user.*", "system.*", "api.*",
                "generate.*", "search.*", "training.*"
            ],
            AccessLevel.AUTHORIZED: [
                "generate.*", "search.*", "user.read", "user.update"
            ],
            AccessLevel.AUTHENTICATED: [
                "search.basic", "generate.basic"
            ],
            AccessLevel.PUBLIC: [
                "health.check", "status.read"
            ]
        }
    
    def check_permission(
        self, 
        user_info: Dict[str, Any], 
        required_permission: str
    ) -> bool:
        """권한 확인"""
        
        try:
            # Super admin은 모든 권한
            user_permissions = user_info.get("permissions", [])
            if "*" in user_permissions:
                return True
            
            # 직접 권한 확인
            if required_permission in user_permissions:
                return True
            
            # 와일드카드 권한 확인
            for permission in user_permissions:
                if permission.endswith(".*"):
                    prefix = permission[:-2]
                    if required_permission.startswith(prefix + "."):
                        return True
            
            # 접근 레벨 기반 권한 확인
            access_level = user_info.get("access_level")
            if access_level:
                try:
                    level_enum = AccessLevel(access_level)
                    level_permissions = self.permission_hierarchy.get(level_enum, [])
                    
                    for permission in level_permissions:
                        if permission == "*":
                            return True
                        if permission == required_permission:
                            return True
                        if permission.endswith(".*"):
                            prefix = permission[:-2]
                            if required_permission.startswith(prefix + "."):
                                return True
                                
                except ValueError:
                    logger.warning(f"Invalid access level: {access_level}")
            
            return False
            
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return False
    
    def require_permission(self, required_permission: str):
        """권한 필요 데코레이터 생성"""
        
        def permission_decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # user_info 인자 찾기
                user_info = None
                for arg in args:
                    if isinstance(arg, dict) and "user_id" in arg:
                        user_info = arg
                        break
                
                if not user_info:
                    for key, value in kwargs.items():
                        if isinstance(value, dict) and "user_id" in value:
                            user_info = value
                            break
                
                if not user_info:
                    raise EnhancedAuthorizationError(
                        detail="사용자 인증 정보가 없습니다",
                        error_code="NO_USER_INFO"
                    )
                
                if not self.check_permission(user_info, required_permission):
                    security_monitor.log_security_event(
                        event_type="permission_denied",
                        severity=SecurityLevel.MEDIUM,
                        source_ip=user_info.get("client_ip", "unknown"),
                        user_id=user_info.get("user_id"),
                        details={
                            "required_permission": required_permission,
                            "user_permissions": user_info.get("permissions", [])
                        }
                    )
                    
                    raise EnhancedAuthorizationError(
                        detail=f"'{required_permission}' 권한이 필요합니다",
                        error_code="PERMISSION_DENIED",
                        required_permission=required_permission
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return permission_decorator


# 전역 인스턴스들
auth_manager = AuthenticationManager()
authz_manager = AuthorizationManager()


# 의존성 함수들
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """현재 사용자 정보 조회 (API 키 또는 JWT)"""
    
    if not credentials:
        raise EnhancedAuthenticationError(
            detail="인증이 필요합니다",
            error_code="AUTHENTICATION_REQUIRED"
        )
    
    # API 키 형태 확인
    token = credentials.credentials
    if token.startswith(('pk_', 'sk_', 'ak_', 'admin_', 'super_')):
        return await auth_manager.authenticate_api_key(request, credentials)
    else:
        return await auth_manager.authenticate_jwt(request, credentials)


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """선택적 사용자 정보 조회"""
    
    if not credentials:
        return None
    
    try:
        return await get_current_user(request, credentials)
    except Exception:
        return None


def require_access_level(min_level: AccessLevel):
    """최소 접근 레벨 필요"""
    
    level_order = {
        AccessLevel.PUBLIC: 0,
        AccessLevel.AUTHENTICATED: 1,
        AccessLevel.AUTHORIZED: 2,
        AccessLevel.ADMIN: 3,
        AccessLevel.SUPER_ADMIN: 4
    }
    
    def access_level_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        
        user_level = AccessLevel(current_user.get("access_level", "public"))
        
        if level_order.get(user_level, 0) < level_order.get(min_level, 0):
            raise EnhancedAuthorizationError(
                detail=f"최소 '{min_level.value}' 레벨 권한이 필요합니다",
                error_code="INSUFFICIENT_ACCESS_LEVEL"
            )
        
        return current_user
    
    return access_level_checker


def require_permission(permission: str):
    """특정 권한 필요"""
    
    def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        
        if not authz_manager.check_permission(current_user, permission):
            raise EnhancedAuthorizationError(
                detail=f"'{permission}' 권한이 필요합니다",
                error_code="PERMISSION_DENIED",
                required_permission=permission
            )
        
        return current_user
    
    return permission_checker


async def security_audit_middleware(request: Request, call_next):
    """보안 감사 미들웨어"""
    
    start_time = time.time()
    client_ip = auth_manager._get_client_ip(request)
    
    # 요청 내용 보안 검사
    if request.method in ["POST", "PUT", "PATCH"]:
        try:
            body = await request.body()
            if body:
                content = body.decode('utf-8', errors='ignore')
                suspicious_findings = security_monitor.check_suspicious_content(content, client_ip)
                
                if suspicious_findings:
                    logger.warning(f"Suspicious content detected from {client_ip}: {suspicious_findings}")
        except Exception as e:
            logger.warning(f"Failed to analyze request content: {e}")
    
    # 응답 처리
    response = await call_next(request)
    
    # 응답 시간 메트릭 기록
    response_time = (time.time() - start_time) * 1000
    metrics_collector.record_request_time(response_time, response.status_code >= 400)
    
    # 보안 헤더 추가
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response