"""
사용자 모델 및 인증 관련 데이터베이스 스키마
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, ForeignKey, Enum, Index
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta
import uuid
import enum
from typing import Optional, Dict, Any, List

from ..database import Base

class UserStatus(enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    BANNED = "banned"

class UserRole(enum.Enum):
    USER = "user"
    PREMIUM_USER = "premium_user"
    MODERATOR = "moderator"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class AuthProvider(enum.Enum):
    LOCAL = "local"
    GOOGLE = "google"
    FACEBOOK = "facebook"
    GITHUB = "github"
    MICROSOFT = "microsoft"

class User(Base):
    """사용자 모델"""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=True)
    full_name = Column(String(100), nullable=True)

    # 인증 정보
    hashed_password = Column(String(255), nullable=True)
    is_password_set = Column(Boolean, default=False)

    # 계정 상태
    status = Column(Enum(UserStatus), default=UserStatus.PENDING_VERIFICATION)
    role = Column(Enum(UserRole), default=UserRole.USER)

    # 프로필 정보
    profile_image_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    location = Column(String(100), nullable=True)
    website = Column(String(200), nullable=True)

    # 기본 정보
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    last_activity_at = Column(DateTime, nullable=True)

    # 이메일 인증
    is_email_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime, nullable=True)

    # 개인정보 설정
    preferences = Column(JSON, default=dict)
    settings = Column(JSON, default=dict)

    # 관계
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    oauth_accounts = relationship("OAuthAccount", back_populates="user", cascade="all, delete-orphan")
    login_attempts = relationship("LoginAttempt", back_populates="user", cascade="all, delete-orphan")
    user_permissions = relationship("UserPermission", back_populates="user", cascade="all, delete-orphan")
    subscription = relationship("Subscription", back_populates="user", uselist=False)

    # 인덱스
    __table_args__ = (
        Index('idx_user_email_status', email, status),
        Index('idx_user_role_status', role, status),
        Index('idx_user_created_at', created_at),
    )

class UserSession(Base):
    """사용자 세션 모델"""
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)

    # 세션 정보
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True, nullable=True)

    # 디바이스 및 브라우저 정보
    device_id = Column(String(100), nullable=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 지원
    location = Column(String(100), nullable=True)

    # 세션 상태
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_used_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    user = relationship("User", back_populates="sessions")

    # 인덱스
    __table_args__ = (
        Index('idx_session_token', session_token),
        Index('idx_session_user_active', user_id, is_active),
        Index('idx_session_expires_at', expires_at),
    )

class ApiKey(Base):
    """API 키 모델"""
    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)

    # API 키 정보
    key_id = Column(String(50), unique=True, index=True, nullable=False)  # pk_xxx, sk_xxx
    key_hash = Column(String(255), nullable=False)  # 해시된 실제 키
    name = Column(String(100), nullable=False)  # 사용자 지정 이름

    # 권한 및 제한
    scopes = Column(JSON, default=list)  # 허용된 API 스코프
    rate_limit_rpm = Column(Integer, default=60)  # 분당 요청 제한
    rate_limit_rph = Column(Integer, default=1000)  # 시간당 요청 제한

    # 상태
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # 통계
    usage_count = Column(Integer, default=0)

    # 관계
    user = relationship("User", back_populates="api_keys")

class OAuthAccount(Base):
    """소셜 로그인 계정 모델"""
    __tablename__ = "oauth_accounts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)

    # OAuth 정보
    provider = Column(Enum(AuthProvider), nullable=False)
    provider_user_id = Column(String(255), nullable=False)
    provider_username = Column(String(100), nullable=True)
    provider_email = Column(String(255), nullable=True)

    # 토큰 정보
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime, nullable=True)

    # 프로필 데이터
    profile_data = Column(JSON, default=dict)

    # 상태
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계
    user = relationship("User", back_populates="oauth_accounts")

    # 고유 제약조건
    __table_args__ = (
        Index('idx_oauth_provider_user_id', provider, provider_user_id, unique=True),
        Index('idx_oauth_user_provider', user_id, provider),
    )

class LoginAttempt(Base):
    """로그인 시도 기록"""
    __tablename__ = "login_attempts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)

    # 시도 정보
    email = Column(String(255), nullable=False)
    ip_address = Column(String(45), nullable=False)
    user_agent = Column(Text, nullable=True)

    # 결과
    success = Column(Boolean, nullable=False)
    failure_reason = Column(String(100), nullable=True)

    # 시간
    attempted_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    user = relationship("User", back_populates="login_attempts")

    # 인덱스
    __table_args__ = (
        Index('idx_login_attempts_ip_time', ip_address, attempted_at),
        Index('idx_login_attempts_email_time', email, attempted_at),
        Index('idx_login_attempts_user_time', user_id, attempted_at),
    )

class Permission(Base):
    """권한 정의 모델"""
    __tablename__ = "permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 권한 정보
    code = Column(String(100), unique=True, index=True, nullable=False)  # e.g., "fragrance.create"
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), nullable=False)  # e.g., "fragrance", "user", "admin"

    # 상태
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    user_permissions = relationship("UserPermission", back_populates="permission")
    role_permissions = relationship("RolePermission", back_populates="permission")

class Role(Base):
    """역할 모델 (권한 그룹)"""
    __tablename__ = "roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # 역할 정보
    code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)

    # 계층 구조
    level = Column(Integer, default=0)  # 0이 가장 낮음

    # 상태
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    user_roles = relationship("UserRole", back_populates="role")
    role_permissions = relationship("RolePermission", back_populates="role")

class UserPermission(Base):
    """사용자별 개별 권한"""
    __tablename__ = "user_permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    permission_id = Column(UUID(as_uuid=True), ForeignKey('permissions.id'), nullable=False)

    # 권한 부여/거부
    granted = Column(Boolean, default=True)

    # 만료 시간 (선택적)
    expires_at = Column(DateTime, nullable=True)

    # 부여 정보
    granted_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    granted_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    user = relationship("User", back_populates="user_permissions", foreign_keys=[user_id])
    permission = relationship("Permission", back_populates="user_permissions")
    granted_by_user = relationship("User", foreign_keys=[granted_by])

    # 고유 제약조건
    __table_args__ = (
        Index('idx_user_permission_unique', user_id, permission_id, unique=True),
        Index('idx_user_permission_granted', user_id, granted),
    )

class RolePermission(Base):
    """역할별 권한 매핑"""
    __tablename__ = "role_permissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'), nullable=False)
    permission_id = Column(UUID(as_uuid=True), ForeignKey('permissions.id'), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    role = relationship("Role", back_populates="role_permissions")
    permission = relationship("Permission", back_populates="role_permissions")

    # 고유 제약조건
    __table_args__ = (
        Index('idx_role_permission_unique', role_id, permission_id, unique=True),
    )

class UserRole(Base):
    """사용자별 역할 할당"""
    __tablename__ = "user_roles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    role_id = Column(UUID(as_uuid=True), ForeignKey('roles.id'), nullable=False)

    # 할당 정보
    assigned_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    # 상태
    is_active = Column(Boolean, default=True)

    # 관계
    user = relationship("User", foreign_keys=[user_id])
    role = relationship("Role", back_populates="user_roles")
    assigned_by_user = relationship("User", foreign_keys=[assigned_by])

    # 고유 제약조건
    __table_args__ = (
        Index('idx_user_role_unique', user_id, role_id, unique=True),
        Index('idx_user_role_active', user_id, is_active),
    )

class EmailVerificationToken(Base):
    """이메일 인증 토큰"""
    __tablename__ = "email_verification_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)

    # 토큰 정보
    token = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), nullable=False)

    # 상태
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)

    # 관계
    user = relationship("User")

    def __init__(self, user_id: uuid.UUID, email: str, expires_hours: int = 24):
        self.user_id = user_id
        self.email = email
        self.token = str(uuid.uuid4())
        self.expires_at = datetime.utcnow() + timedelta(hours=expires_hours)

class PasswordResetToken(Base):
    """비밀번호 재설정 토큰"""
    __tablename__ = "password_reset_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)

    # 토큰 정보
    token = Column(String(255), unique=True, index=True, nullable=False)

    # 상태
    is_used = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    used_at = Column(DateTime, nullable=True)

    # 관계
    user = relationship("User")

    def __init__(self, user_id: uuid.UUID, expires_hours: int = 1):
        self.user_id = user_id
        self.token = str(uuid.uuid4())
        self.expires_at = datetime.utcnow() + timedelta(hours=expires_hours)