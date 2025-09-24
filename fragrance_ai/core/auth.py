"""
완전한 인증 및 권한 부여 시스템
JWT, API 키, 역할 기반 접근 제어 (RBAC) 지원
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
import hashlib
import asyncio
from functools import wraps
from enum import Enum
import redis.asyncio as redis
from dataclasses import dataclass, field
import json
import time

from .config import settings
from .exceptions import AuthenticationError, AuthorizationError, RateLimitError
from .advanced_logging import get_logger, LogContext


logger = get_logger(__name__, LogContext.AUTH)


class UserRole(str, Enum):
    """사용자 역할 정의"""
    ADMIN = "admin"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    USER = "user"
    GUEST = "guest"


class Permission(str, Enum):
    """권한 정의"""
    # 시스템 관리
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_MONITOR = "system:monitor"

    # API 접근
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"

    # 모델 관련
    MODEL_INFERENCE = "model:inference"
    MODEL_TRAIN = "model:train"
    MODEL_DEPLOY = "model:deploy"
    MODEL_MANAGE = "model:manage"

    # 데이터 관련
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"

    # 검색 및 생성
    SEARCH_BASIC = "search:basic"
    SEARCH_ADVANCED = "search:advanced"
    GENERATION_BASIC = "generation:basic"
    GENERATION_ADVANCED = "generation:advanced"


@dataclass
class User:
    """사용자 모델"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    role: UserRole = UserRole.USER
    permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    api_key: Optional[str] = None
    rate_limit: Dict[str, int] = field(default_factory=lambda: {"requests_per_minute": 100})
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, permission: Permission) -> bool:
        """권한 확인"""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """여러 권한 중 하나라도 있는지 확인"""
        return any(perm in self.permissions for perm in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """모든 권한을 가지고 있는지 확인"""
        return all(perm in self.permissions for perm in permissions)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "is_verified": self.is_verified,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "rate_limit": self.rate_limit,
            "metadata": self.metadata
        }


# 역할별 기본 권한 매핑
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.SYSTEM_ADMIN, Permission.SYSTEM_CONFIG, Permission.SYSTEM_MONITOR,
        Permission.API_ADMIN, Permission.API_WRITE, Permission.API_READ,
        Permission.MODEL_MANAGE, Permission.MODEL_DEPLOY, Permission.MODEL_TRAIN, Permission.MODEL_INFERENCE,
        Permission.DATA_EXPORT, Permission.DATA_DELETE, Permission.DATA_WRITE, Permission.DATA_READ,
        Permission.SEARCH_ADVANCED, Permission.SEARCH_BASIC,
        Permission.GENERATION_ADVANCED, Permission.GENERATION_BASIC
    ],
    UserRole.DEVELOPER: [
        Permission.SYSTEM_MONITOR,
        Permission.API_WRITE, Permission.API_READ,
        Permission.MODEL_TRAIN, Permission.MODEL_INFERENCE,
        Permission.DATA_WRITE, Permission.DATA_READ,
        Permission.SEARCH_ADVANCED, Permission.SEARCH_BASIC,
        Permission.GENERATION_ADVANCED, Permission.GENERATION_BASIC
    ],
    UserRole.RESEARCHER: [
        Permission.API_READ,
        Permission.MODEL_INFERENCE,
        Permission.DATA_READ,
        Permission.SEARCH_ADVANCED, Permission.SEARCH_BASIC,
        Permission.GENERATION_BASIC
    ],
    UserRole.USER: [
        Permission.API_READ,
        Permission.MODEL_INFERENCE,
        Permission.SEARCH_BASIC,
        Permission.GENERATION_BASIC
    ],
    UserRole.GUEST: [
        Permission.SEARCH_BASIC
    ]
}


class AuthenticationManager:
    """인증 관리자"""

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.bearer = HTTPBearer()
        self.api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        self.redis_client: Optional[redis.Redis] = None
        self.users_db: Dict[str, User] = {}  # 실제 구현에서는 데이터베이스 사용
        self.api_keys_db: Dict[str, str] = {}  # API 키 -> user_id 매핑

    async def initialize(self):
        """초기화"""
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            logger.info("Authentication manager initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize authentication manager", exception=e)
            raise

    def hash_password(self, password: str) -> str:
        """비밀번호 해시화"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """JWT 액세스 토큰 생성"""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)

        to_encode = {
            "sub": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in user.permissions],
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }

        encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
        logger.info(f"Created access token for user {user.username}")
        return encoded_jwt

    def create_refresh_token(self, user: User) -> str:
        """리프레시 토큰 생성"""
        expire = datetime.now(timezone.utc) + timedelta(days=30)
        to_encode = {
            "sub": user.user_id,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        }
        return jwt.encode(to_encode, settings.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Dict[str, Any]:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
            return payload
        except JWTError as e:
            logger.error("JWT verification failed", exception=e, token=token[:20] + "...")
            raise AuthenticationError({"error": "Invalid token"})

    def generate_api_key(self, user: User) -> str:
        """API 키 생성"""
        api_key = f"fai_{secrets.token_urlsafe(32)}"
        self.api_keys_db[api_key] = user.user_id
        user.api_key = api_key
        logger.info(f"Generated API key for user {user.username}")
        return api_key

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """사용자 ID로 사용자 정보 조회"""
        return self.users_db.get(user_id)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """사용자명으로 사용자 정보 조회"""
        for user in self.users_db.values():
            if user.username == username:
                return user
        return None

    async def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """API 키로 사용자 정보 조회"""
        user_id = self.api_keys_db.get(api_key)
        if user_id:
            return await self.get_user_by_id(user_id)
        return None

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
        full_name: Optional[str] = None
    ) -> User:
        """사용자 생성"""
        user_id = secrets.token_urlsafe(16)
        hashed_password = self.hash_password(password)

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            permissions=ROLE_PERMISSIONS.get(role, [])
        )

        # API 키 생성
        self.generate_api_key(user)

        self.users_db[user_id] = user
        logger.info(f"Created new user: {username} with role {role}")
        return user

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """사용자 인증"""
        user = await self.get_user_by_username(username)
        if not user or not user.is_active:
            return None

        # 실제 구현에서는 데이터베이스에서 해시된 비밀번호 조회
        # 여기서는 데모용으로 비밀번호를 "password"로 가정
        if not self.verify_password(password, self.hash_password("password")):
            return None

        user.last_login = datetime.now(timezone.utc)
        logger.info(f"User {username} authenticated successfully")
        return user


class RateLimiter:
    """요청 제한 관리자"""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> bool:
        """요청 허용 여부 확인"""
        try:
            current = await self.redis_client.get(key)
            if current is None:
                # 새로운 키 설정
                await self.redis_client.setex(key, window_seconds, 1)
                return True

            current_count = int(current)
            if current_count >= limit:
                return False

            # 카운트 증가
            await self.redis_client.incr(key)
            return True

        except Exception as e:
            logger.error("Rate limiting error", exception=e)
            return True  # 에러 시 허용


class AuthorizationManager:
    """권한 부여 관리자"""

    def __init__(self, auth_manager: AuthenticationManager):
        self.auth_manager = auth_manager
        self.rate_limiter = None

    async def initialize(self):
        """초기화"""
        if self.auth_manager.redis_client:
            self.rate_limiter = RateLimiter(self.auth_manager.redis_client)

    def require_permissions(self, required_permissions: List[Permission]):
        """권한 요구 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 현재 사용자 정보 추출 (컨텍스트에서)
                request = kwargs.get('request') or args[0] if args else None
                current_user = getattr(request.state, 'current_user', None) if request else None

                if not current_user:
                    raise AuthenticationError({"error": "User not authenticated"})

                if not current_user.has_all_permissions(required_permissions):
                    missing_perms = [p for p in required_permissions if not current_user.has_permission(p)]
                    raise AuthorizationError({
                        "error": "Insufficient permissions",
                        "required_permissions": [p.value for p in required_permissions],
                        "missing_permissions": [p.value for p in missing_perms]
                    })

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def require_role(self, required_role: UserRole):
        """역할 요구 데코레이터"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get('request') or args[0] if args else None
                current_user = getattr(request.state, 'current_user', None) if request else None

                if not current_user:
                    raise AuthenticationError({"error": "User not authenticated"})

                if current_user.role != required_role:
                    raise AuthorizationError({
                        "error": f"Role '{required_role}' required",
                        "current_role": current_user.role
                    })

                return await func(*args, **kwargs)
            return wrapper
        return decorator


# 전역 인스턴스
auth_manager = AuthenticationManager()
authz_manager = AuthorizationManager(auth_manager)


# FastAPI 의존성 함수들
async def get_current_user_from_token(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())
) -> User:
    """JWT 토큰에서 현재 사용자 추출"""
    try:
        payload = auth_manager.verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise AuthenticationError({"error": "Invalid token payload"})

        user = await auth_manager.get_user_by_id(user_id)
        if user is None:
            raise AuthenticationError({"error": "User not found"})

        if not user.is_active:
            raise AuthenticationError({"error": "User is inactive"})

        return user

    except JWTError:
        raise AuthenticationError({"error": "Invalid token"})


async def get_current_user_from_api_key(
    api_key: Optional[str] = Depends(APIKeyHeader(name="X-API-Key", auto_error=False))
) -> Optional[User]:
    """API 키에서 현재 사용자 추출"""
    if not api_key:
        return None

    user = await auth_manager.get_user_by_api_key(api_key)
    if user and user.is_active:
        return user

    raise AuthenticationError({"error": "Invalid API key"})


async def get_current_user(
    request: Request,
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """현재 사용자 가져오기 (JWT 또는 API 키)"""

    current_user = token_user or api_key_user

    if not current_user:
        raise AuthenticationError({"error": "Authentication required"})

    # 요청 제한 확인
    if authz_manager.rate_limiter:
        rate_key = f"rate_limit:{current_user.user_id}:{int(time.time() // 60)}"
        allowed = await authz_manager.rate_limiter.is_allowed(
            rate_key,
            current_user.rate_limit.get("requests_per_minute", 100),
            60
        )

        if not allowed:
            raise RateLimitError(
                current_user.rate_limit.get("requests_per_minute", 100),
                "minute"
            )

    # 요청 상태에 사용자 정보 저장
    request.state.current_user = current_user
    logger.info(f"User {current_user.username} authenticated for request")

    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """관리자 사용자만 허용"""
    if current_user.role != UserRole.ADMIN:
        raise AuthorizationError({"error": "Admin role required"})
    return current_user


# 편의 함수들
def create_demo_users():
    """데모용 사용자 생성"""
    asyncio.create_task(_create_demo_users())


async def _create_demo_users():
    """비동기 데모 사용자 생성"""
    await auth_manager.initialize()

    # 관리자 사용자
    admin = await auth_manager.create_user(
        username="admin",
        email="admin@fragranceai.com",
        password="admin_password",
        role=UserRole.ADMIN,
        full_name="System Administrator"
    )

    # 개발자 사용자
    developer = await auth_manager.create_user(
        username="developer",
        email="dev@fragranceai.com",
        password="dev_password",
        role=UserRole.DEVELOPER,
        full_name="AI Developer"
    )

    # 연구자 사용자
    researcher = await auth_manager.create_user(
        username="researcher",
        email="research@fragranceai.com",
        password="research_password",
        role=UserRole.RESEARCHER,
        full_name="Fragrance Researcher"
    )

    logger.info("Demo users created successfully")


# 초기화 함수
async def initialize_auth():
    """인증 시스템 초기화"""
    await auth_manager.initialize()
    await authz_manager.initialize()
    logger.info("Authentication and authorization system initialized")