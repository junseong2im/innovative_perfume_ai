"""
통합 사용자 인증 및 권한 관리 서비스
"""

import secrets
import hashlib
import hmac
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status, Request
import bcrypt
import jwt
from email_validator import validate_email, EmailNotValidError
import re
import asyncio
import redis.asyncio as redis
from dataclasses import dataclass
import uuid
from enum import Enum

from ..core.config import settings
from ..core.logging_config import get_logger
from ..database import get_db
from ..models.user import (
    User, UserSession, ApiKey, OAuthAccount, LoginAttempt, Permission,
    Role, UserPermission, RolePermission, UserRole, EmailVerificationToken,
    PasswordResetToken, UserStatus, UserRole as UserRoleEnum, AuthProvider
)

logger = get_logger(__name__)

class AuthenticationMethod(Enum):
    PASSWORD = "password"
    OAUTH = "oauth"
    API_KEY = "api_key"
    SESSION_TOKEN = "session_token"

@dataclass
class AuthResult:
    success: bool
    user: Optional[User] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    message: Optional[str] = None
    error_code: Optional[str] = None

@dataclass
class UserRegistrationData:
    email: str
    password: Optional[str] = None
    full_name: Optional[str] = None
    username: Optional[str] = None
    oauth_provider: Optional[AuthProvider] = None
    oauth_data: Optional[Dict[str, Any]] = None

class AuthService:
    """통합 인증 서비스"""

    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.failed_attempts_limit = 5
        self.lockout_duration_minutes = 15
        self.password_requirements = {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special_chars': True
        }

    async def initialize(self):
        """서비스 초기화"""
        try:
            # Redis 연결 (세션 및 캐시용)
            if settings.redis_url:
                self.redis_client = await aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                logger.info("Auth service initialized with Redis")
            else:
                logger.warning("Auth service initialized without Redis - some features may be limited")

        except Exception as e:
            logger.error(f"Failed to initialize auth service: {e}")
            raise

    # 사용자 등록
    async def register_user(
        self,
        registration_data: UserRegistrationData,
        db: Session,
        skip_email_verification: bool = False
    ) -> AuthResult:
        """사용자 등록"""
        try:
            # 이메일 유효성 검사
            if not self._is_valid_email(registration_data.email):
                return AuthResult(
                    success=False,
                    message="유효하지 않은 이메일 주소입니다",
                    error_code="INVALID_EMAIL"
                )

            # 기존 사용자 확인
            existing_user = db.query(User).filter(
                User.email == registration_data.email.lower()
            ).first()

            if existing_user:
                return AuthResult(
                    success=False,
                    message="이미 등록된 이메일 주소입니다",
                    error_code="EMAIL_ALREADY_EXISTS"
                )

            # 사용자명 중복 확인 (제공된 경우)
            if registration_data.username:
                existing_username = db.query(User).filter(
                    User.username == registration_data.username
                ).first()
                if existing_username:
                    return AuthResult(
                        success=False,
                        message="이미 사용 중인 사용자명입니다",
                        error_code="USERNAME_ALREADY_EXISTS"
                    )

            # 비밀번호 검증 (OAuth가 아닌 경우)
            if not registration_data.oauth_provider and registration_data.password:
                password_validation = self._validate_password(registration_data.password)
                if not password_validation['valid']:
                    return AuthResult(
                        success=False,
                        message=password_validation['message'],
                        error_code="INVALID_PASSWORD"
                    )

            # 사용자 생성
            user = User(
                email=registration_data.email.lower(),
                username=registration_data.username,
                full_name=registration_data.full_name,
                status=UserStatus.PENDING_VERIFICATION if not skip_email_verification else UserStatus.ACTIVE,
                is_email_verified=skip_email_verification
            )

            # 비밀번호 설정 (제공된 경우)
            if registration_data.password:
                user.hashed_password = self._hash_password(registration_data.password)
                user.is_password_set = True

            db.add(user)
            db.commit()
            db.refresh(user)

            # OAuth 계정 연결 (해당되는 경우)
            if registration_data.oauth_provider:
                await self._link_oauth_account(
                    user, registration_data.oauth_provider,
                    registration_data.oauth_data, db
                )

            # 이메일 인증 토큰 생성 (필요한 경우)
            verification_token = None
            if not skip_email_verification:
                verification_token = await self._create_email_verification_token(user, db)

            logger.info(f"User registered successfully: {user.email}")

            return AuthResult(
                success=True,
                user=user,
                message="사용자가 성공적으로 등록되었습니다"
            )

        except Exception as e:
            logger.error(f"User registration failed: {e}")
            db.rollback()
            return AuthResult(
                success=False,
                message="사용자 등록 중 오류가 발생했습니다",
                error_code="REGISTRATION_ERROR"
            )

    # 로그인 인증
    async def authenticate_user(
        self,
        email: str,
        password: str,
        request: Request,
        db: Session
    ) -> AuthResult:
        """사용자 인증"""
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")

        try:
            # Rate limiting 확인
            if await self._is_ip_locked(client_ip):
                return AuthResult(
                    success=False,
                    message="너무 많은 로그인 시도로 인해 일시적으로 차단되었습니다",
                    error_code="IP_LOCKED"
                )

            # 사용자 조회
            user = db.query(User).filter(User.email == email.lower()).first()

            # 로그인 시도 기록
            login_attempt = LoginAttempt(
                user_id=user.id if user else None,
                email=email.lower(),
                ip_address=client_ip,
                user_agent=user_agent,
                success=False
            )

            if not user:
                login_attempt.failure_reason = "USER_NOT_FOUND"
                db.add(login_attempt)
                db.commit()
                await self._increment_failed_attempts(client_ip, email)
                return AuthResult(
                    success=False,
                    message="이메일 또는 비밀번호가 잘못되었습니다",
                    error_code="INVALID_CREDENTIALS"
                )

            # 계정 상태 확인
            if user.status == UserStatus.SUSPENDED:
                login_attempt.failure_reason = "ACCOUNT_SUSPENDED"
                db.add(login_attempt)
                db.commit()
                return AuthResult(
                    success=False,
                    message="계정이 일시 정지되었습니다",
                    error_code="ACCOUNT_SUSPENDED"
                )

            if user.status == UserStatus.BANNED:
                login_attempt.failure_reason = "ACCOUNT_BANNED"
                db.add(login_attempt)
                db.commit()
                return AuthResult(
                    success=False,
                    message="계정이 차단되었습니다",
                    error_code="ACCOUNT_BANNED"
                )

            # 비밀번호 확인
            if not user.hashed_password or not self._verify_password(password, user.hashed_password):
                login_attempt.failure_reason = "INVALID_PASSWORD"
                db.add(login_attempt)
                db.commit()
                await self._increment_failed_attempts(client_ip, email)
                return AuthResult(
                    success=False,
                    message="이메일 또는 비밀번호가 잘못되었습니다",
                    error_code="INVALID_CREDENTIALS"
                )

            # 성공적인 로그인
            login_attempt.success = True
            db.add(login_attempt)

            # 사용자 정보 업데이트
            user.last_login_at = datetime.utcnow()
            user.last_activity_at = datetime.utcnow()

            # 세션 생성
            session = await self._create_user_session(user, request, db)

            # JWT 토큰 생성
            access_token = self._create_access_token({
                "sub": str(user.id),
                "email": user.email,
                "role": user.role.value
            })

            db.commit()

            # 실패한 시도 횟수 초기화
            await self._reset_failed_attempts(client_ip, email)

            logger.info(f"User authenticated successfully: {user.email}")

            return AuthResult(
                success=True,
                user=user,
                access_token=access_token,
                refresh_token=session.refresh_token,
                message="로그인이 성공했습니다"
            )

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            db.rollback()
            return AuthResult(
                success=False,
                message="인증 처리 중 오류가 발생했습니다",
                error_code="AUTHENTICATION_ERROR"
            )

    # OAuth 인증
    async def authenticate_oauth_user(
        self,
        provider: AuthProvider,
        oauth_data: Dict[str, Any],
        request: Request,
        db: Session
    ) -> AuthResult:
        """OAuth 사용자 인증"""
        try:
            provider_user_id = oauth_data.get('id') or oauth_data.get('sub')
            provider_email = oauth_data.get('email')

            if not provider_user_id:
                return AuthResult(
                    success=False,
                    message="OAuth 사용자 ID를 찾을 수 없습니다",
                    error_code="OAUTH_ID_MISSING"
                )

            # 기존 OAuth 계정 찾기
            oauth_account = db.query(OAuthAccount).filter(
                OAuthAccount.provider == provider,
                OAuthAccount.provider_user_id == str(provider_user_id)
            ).first()

            if oauth_account and oauth_account.user:
                # 기존 사용자 로그인
                user = oauth_account.user

                # 토큰 업데이트
                oauth_account.access_token = oauth_data.get('access_token')
                oauth_account.refresh_token = oauth_data.get('refresh_token')
                oauth_account.profile_data = oauth_data
                oauth_account.updated_at = datetime.utcnow()

            else:
                # 이메일로 기존 사용자 찾기
                user = None
                if provider_email:
                    user = db.query(User).filter(User.email == provider_email.lower()).first()

                if not user:
                    # 새 사용자 생성
                    registration_data = UserRegistrationData(
                        email=provider_email or f"{provider_user_id}@{provider.value}.local",
                        full_name=oauth_data.get('name'),
                        username=oauth_data.get('preferred_username') or oauth_data.get('login'),
                        oauth_provider=provider,
                        oauth_data=oauth_data
                    )

                    auth_result = await self.register_user(
                        registration_data, db, skip_email_verification=True
                    )

                    if not auth_result.success:
                        return auth_result

                    user = auth_result.user
                else:
                    # 기존 사용자에 OAuth 계정 연결
                    await self._link_oauth_account(user, provider, oauth_data, db)

            # 사용자 정보 업데이트
            user.last_login_at = datetime.utcnow()
            user.last_activity_at = datetime.utcnow()

            # 세션 생성
            session = await self._create_user_session(user, request, db)

            # JWT 토큰 생성
            access_token = self._create_access_token({
                "sub": str(user.id),
                "email": user.email,
                "role": user.role.value
            })

            db.commit()

            logger.info(f"OAuth user authenticated successfully: {user.email} via {provider.value}")

            return AuthResult(
                success=True,
                user=user,
                access_token=access_token,
                refresh_token=session.refresh_token,
                message="OAuth 로그인이 성공했습니다"
            )

        except Exception as e:
            logger.error(f"OAuth authentication failed: {e}")
            db.rollback()
            return AuthResult(
                success=False,
                message="OAuth 인증 처리 중 오류가 발생했습니다",
                error_code="OAUTH_AUTHENTICATION_ERROR"
            )

    # 토큰 검증
    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """액세스 토큰 검증"""
        try:
            payload = jwt.decode(
                token,
                settings.secret_key,
                algorithms=["HS256"]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.JWTError as e:
            logger.error(f"Token verification failed: {e}")
            return None

    # API 키 관리
    async def create_api_key(
        self,
        user: User,
        name: str,
        scopes: List[str],
        db: Session,
        expires_in_days: Optional[int] = None
    ) -> Tuple[str, ApiKey]:
        """API 키 생성"""
        try:
            # 키 생성
            key_prefix = "pk_"  # Public key prefix
            key_body = secrets.token_urlsafe(32)
            full_key = f"{key_prefix}{key_body}"

            # 키 해시
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()

            # 키 ID 생성 (공개적으로 식별 가능)
            key_id = f"{key_prefix}{secrets.token_hex(8)}"

            # 만료 시간 설정
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            # 데이터베이스에 저장
            api_key = ApiKey(
                user_id=user.id,
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                scopes=scopes,
                expires_at=expires_at
            )

            db.add(api_key)
            db.commit()
            db.refresh(api_key)

            logger.info(f"API key created for user {user.email}: {key_id}")

            return full_key, api_key

        except Exception as e:
            logger.error(f"API key creation failed: {e}")
            db.rollback()
            raise

    async def validate_api_key(self, key: str, db: Session) -> Optional[ApiKey]:
        """API 키 검증"""
        try:
            key_hash = hashlib.sha256(key.encode()).hexdigest()

            api_key = db.query(ApiKey).filter(
                ApiKey.key_hash == key_hash,
                ApiKey.is_active == True
            ).first()

            if not api_key:
                return None

            # 만료 확인
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                return None

            # 사용 기록 업데이트
            api_key.last_used_at = datetime.utcnow()
            api_key.usage_count += 1
            db.commit()

            return api_key

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None

    # 권한 관리
    async def check_user_permission(
        self,
        user: User,
        permission_code: str,
        db: Session
    ) -> bool:
        """사용자 권한 확인"""
        try:
            # Super Admin은 모든 권한 보유
            if user.role == UserRoleEnum.SUPER_ADMIN:
                return True

            # 직접 할당된 권한 확인
            user_permission = db.query(UserPermission).join(Permission).filter(
                UserPermission.user_id == user.id,
                Permission.code == permission_code,
                UserPermission.granted == True,
                Permission.is_active == True
            ).filter(
                # 만료 확인
                (UserPermission.expires_at.is_(None)) |
                (UserPermission.expires_at > datetime.utcnow())
            ).first()

            if user_permission:
                return True

            # 역할 기반 권한 확인
            role_permission = db.query(RolePermission).join(
                UserRole, UserRole.role_id == RolePermission.role_id
            ).join(Permission).filter(
                UserRole.user_id == user.id,
                Permission.code == permission_code,
                UserRole.is_active == True,
                Permission.is_active == True
            ).filter(
                # 역할 만료 확인
                (UserRole.expires_at.is_(None)) |
                (UserRole.expires_at > datetime.utcnow())
            ).first()

            return role_permission is not None

        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False

    # 유틸리티 메서드들
    def _is_valid_email(self, email: str) -> bool:
        """이메일 유효성 검사"""
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False

    def _validate_password(self, password: str) -> Dict[str, Any]:
        """비밀번호 유효성 검사"""
        req = self.password_requirements

        if len(password) < req['min_length']:
            return {'valid': False, 'message': f"비밀번호는 최소 {req['min_length']}자 이상이어야 합니다"}

        if req['require_uppercase'] and not re.search(r'[A-Z]', password):
            return {'valid': False, 'message': "비밀번호에는 대문자가 포함되어야 합니다"}

        if req['require_lowercase'] and not re.search(r'[a-z]', password):
            return {'valid': False, 'message': "비밀번호에는 소문자가 포함되어야 합니다"}

        if req['require_numbers'] and not re.search(r'\d', password):
            return {'valid': False, 'message': "비밀번호에는 숫자가 포함되어야 합니다"}

        if req['require_special_chars'] and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return {'valid': False, 'message': "비밀번호에는 특수문자가 포함되어야 합니다"}

        return {'valid': True, 'message': "유효한 비밀번호입니다"}

    def _hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

    def _create_access_token(self, data: Dict[str, Any]) -> str:
        """액세스 토큰 생성"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, settings.secret_key, algorithm="HS256")

    async def _create_user_session(
        self,
        user: User,
        request: Request,
        db: Session
    ) -> UserSession:
        """사용자 세션 생성"""
        session_token = secrets.token_urlsafe(32)
        refresh_token = secrets.token_urlsafe(32)

        session = UserSession(
            user_id=user.id,
            session_token=session_token,
            refresh_token=refresh_token,
            user_agent=request.headers.get("user-agent", ""),
            ip_address=self._get_client_ip(request),
            expires_at=datetime.utcnow() + timedelta(days=30)
        )

        db.add(session)
        return session

    async def _link_oauth_account(
        self,
        user: User,
        provider: AuthProvider,
        oauth_data: Dict[str, Any],
        db: Session
    ):
        """OAuth 계정 연결"""
        oauth_account = OAuthAccount(
            user_id=user.id,
            provider=provider,
            provider_user_id=str(oauth_data.get('id') or oauth_data.get('sub')),
            provider_username=oauth_data.get('preferred_username') or oauth_data.get('login'),
            provider_email=oauth_data.get('email'),
            access_token=oauth_data.get('access_token'),
            refresh_token=oauth_data.get('refresh_token'),
            profile_data=oauth_data
        )
        db.add(oauth_account)

    async def _create_email_verification_token(
        self,
        user: User,
        db: Session
    ) -> EmailVerificationToken:
        """이메일 인증 토큰 생성"""
        token = EmailVerificationToken(user.id, user.email)
        db.add(token)
        return token

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 주소 가져오기"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _is_ip_locked(self, ip_address: str) -> bool:
        """IP 주소 차단 상태 확인"""
        if not self.redis_client:
            return False

        try:
            key = f"auth:failed_attempts:{ip_address}"
            attempts = await self.redis_client.get(key)
            return int(attempts or 0) >= self.failed_attempts_limit
        except Exception:
            return False

    async def _increment_failed_attempts(self, ip_address: str, email: str):
        """실패한 로그인 시도 횟수 증가"""
        if not self.redis_client:
            return

        try:
            # IP 기준 제한
            ip_key = f"auth:failed_attempts:{ip_address}"
            await self.redis_client.incr(ip_key)
            await self.redis_client.expire(ip_key, self.lockout_duration_minutes * 60)

            # 이메일 기준 제한
            email_key = f"auth:failed_attempts:email:{email}"
            await self.redis_client.incr(email_key)
            await self.redis_client.expire(email_key, self.lockout_duration_minutes * 60)
        except Exception as e:
            logger.error(f"Failed to increment failed attempts: {e}")

    async def _reset_failed_attempts(self, ip_address: str, email: str):
        """실패한 로그인 시도 횟수 초기화"""
        if not self.redis_client:
            return

        try:
            await self.redis_client.delete(
                f"auth:failed_attempts:{ip_address}",
                f"auth:failed_attempts:email:{email}"
            )
        except Exception as e:
            logger.error(f"Failed to reset failed attempts: {e}")

# 전역 인스턴스
auth_service = AuthService()