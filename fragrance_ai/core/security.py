import hashlib
import hmac
import secrets
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import re
import ipaddress
from collections import defaultdict
import threading
import struct

from .config import settings
from .logging_config import get_logger, performance_logger
from .exceptions import (
    AuthenticationException, ValidationException, SystemException,
    ErrorCode, FragranceAIException
)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = get_logger(__name__)


class SecurityLevel(str, Enum):
    """보안 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(str, Enum):
    """접근 레벨"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class TokenType(str, Enum):
    """토큰 타입"""
    ACCESS = "access"
    REFRESH = "refresh"
    SESSION = "session"
    CSRF = "csrf"


class ThreatType(str, Enum):
    """위협 유형"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    BRUTE_FORCE = "brute_force"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    MALWARE = "malware"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class SecurityContext:
    """보안 컨텍스트"""
    ip_address: str
    user_agent: str
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    threat_score: float = 0.0
    threat_indicators: List[str] = None
    is_blocked: bool = False

    def __post_init__(self):
        if self.threat_indicators is None:
            self.threat_indicators = []


@dataclass
class SecurityEvent:
    """보안 이벤트"""
    event_type: str
    severity: SecurityLevel
    source_ip: str
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False


@dataclass
class ApiKey:
    """API 키 정보"""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[str]
    access_level: AccessLevel
    rate_limit: int
    expires_at: Optional[datetime]
    is_active: bool
    created_at: datetime
    last_used_at: Optional[datetime]
    usage_count: int = 0


class PasswordValidator:
    """비밀번호 검증기"""
    
    def __init__(self):
        self.min_length = 12
        self.max_length = 128
        self.require_uppercase = True
        self.require_lowercase = True
        self.require_digits = True
        self.require_special = True
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # 금지된 패턴들
        self.forbidden_patterns = [
            r'(.)\1{3,}',  # 같은 문자 4번 이상 반복
            r'123456',      # 순차적인 숫자
            r'abcdef',      # 순차적인 문자
            r'qwerty',      # 키보드 패턴
        ]
        
        # 일반적인 약한 비밀번호들
        self.weak_passwords = {
            'password', 'password123', '123456789', 'qwerty123',
            'admin', 'administrator', 'root', 'user', 'guest',
            'fragrance', 'ai', 'fragranceai'
        }
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """비밀번호 유효성 검증"""
        
        errors = []
        
        # 길이 검증
        if len(password) < self.min_length:
            errors.append(f"비밀번호는 최소 {self.min_length}자 이상이어야 합니다")
        
        if len(password) > self.max_length:
            errors.append(f"비밀번호는 최대 {self.max_length}자 이하여야 합니다")
        
        # 문자 타입 검증
        if self.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("대문자가 포함되어야 합니다")
        
        if self.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("소문자가 포함되어야 합니다")
        
        if self.require_digits and not re.search(r'\d', password):
            errors.append("숫자가 포함되어야 합니다")
        
        if self.require_special and not re.search(f'[{re.escape(self.special_chars)}]', password):
            errors.append(f"특수문자({self.special_chars})가 포함되어야 합니다")
        
        # 금지된 패턴 검증
        for pattern in self.forbidden_patterns:
            if re.search(pattern, password.lower()):
                errors.append("금지된 패턴이 포함되어 있습니다")
                break
        
        # 약한 비밀번호 검증
        if password.lower() in self.weak_passwords:
            errors.append("일반적으로 사용되는 약한 비밀번호입니다")
        
        return len(errors) == 0, errors
    
    def generate_secure_password(self, length: int = 16) -> str:
        """보안 비밀번호 생성"""
        
        if length < self.min_length:
            length = self.min_length
        
        # 각 타입별 최소 1개씩 포함
        chars = ""
        password = []
        
        if self.require_uppercase:
            chars += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            password.append(secrets.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        
        if self.require_lowercase:
            chars += "abcdefghijklmnopqrstuvwxyz"
            password.append(secrets.choice("abcdefghijklmnopqrstuvwxyz"))
        
        if self.require_digits:
            chars += "0123456789"
            password.append(secrets.choice("0123456789"))
        
        if self.require_special:
            chars += self.special_chars
            password.append(secrets.choice(self.special_chars))
        
        # 나머지 길이만큼 랜덤 문자 추가
        for _ in range(length - len(password)):
            password.append(secrets.choice(chars))
        
        # 순서 섞기
        secrets.SystemRandom().shuffle(password)
        
        return ''.join(password)


class EncryptionManager:
    """암호화 관리자"""

    def __init__(self):
        # 각 암호화마다 다른 키를 생성하므로 초기화 시 마스터키 생성하지 않음
        self.logger = get_logger(__name__)
        self.logger.info("EncryptionManager initialized")

    def _derive_key_from_password_and_salt(self, password: bytes, salt: bytes) -> bytes:
        """비밀번호와 솔트로부터 키 유도"""

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )

        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key

    def encrypt(self, data: str) -> str:
        """데이터 암호화 (랜덤 솔트 사용)"""

        try:
            # 랜덤 솔트 생성 (32바이트)
            salt = os.urandom(32)

            # 키 유도
            password = settings.secret_key.encode()
            key = self._derive_key_from_password_and_salt(password, salt)
            fernet = Fernet(key)

            # 데이터 암호화
            encrypted_data = fernet.encrypt(data.encode())

            # 솔트와 암호화된 데이터를 함께 저장
            # 형식: salt_length(4bytes) + salt + encrypted_data
            combined_data = struct.pack('I', len(salt)) + salt + encrypted_data

            return base64.urlsafe_b64encode(combined_data).decode()

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SystemException(
                message="데이터 암호화에 실패했습니다",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )

    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화 (저장된 솔트 사용)"""

        try:
            # Base64 디코딩
            combined_data = base64.urlsafe_b64decode(encrypted_data.encode())

            # 솔트 길이 추출 (첫 4바이트)
            salt_length = struct.unpack('I', combined_data[:4])[0]

            # 솔트 추출
            salt = combined_data[4:4+salt_length]

            # 암호화된 데이터 추출
            encrypted_part = combined_data[4+salt_length:]

            # 키 유도
            password = settings.secret_key.encode()
            key = self._derive_key_from_password_and_salt(password, salt)
            fernet = Fernet(key)

            # 복호화
            decrypted_data = fernet.decrypt(encrypted_part)
            return decrypted_data.decode()

        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SystemException(
                message="데이터 복호화에 실패했습니다",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def hash_password(self, password: str) -> str:
        """비밀번호 해시"""
        
        try:
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise SystemException(
                message="비밀번호 해시 생성에 실패했습니다",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """비밀번호 검증"""
        
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False


class ApiKeyManager:
    """API 키 관리자"""
    
    def __init__(self):
        self.keys: Dict[str, ApiKey] = {}
        self.key_prefixes = {
            AccessLevel.PUBLIC: "pk_",
            AccessLevel.AUTHENTICATED: "sk_",
            AccessLevel.AUTHORIZED: "ak_",
            AccessLevel.ADMIN: "admin_",
            AccessLevel.SUPER_ADMIN: "super_"
        }
        self.encryption_manager = EncryptionManager()
        self._load_api_keys()
    
    def _load_api_keys(self):
        """API 키 로드"""
        
        try:
            # 기본 관리자 키 생성
            admin_key = self.create_api_key(
                name="Default Admin Key",
                user_id="admin",
                access_level=AccessLevel.ADMIN,
                permissions=["*"],
                rate_limit=10000
            )
            
            logger.info(f"Default admin API key created: {admin_key[:8]}...")
            
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
    
    def generate_api_key(self, access_level: AccessLevel) -> str:
        """API 키 생성"""
        
        prefix = self.key_prefixes[access_level]
        random_part = secrets.token_urlsafe(32)
        
        return f"{prefix}{random_part}"
    
    def create_api_key(
        self,
        name: str,
        user_id: str,
        access_level: AccessLevel,
        permissions: List[str],
        rate_limit: int = 1000,
        expires_days: Optional[int] = None
    ) -> str:
        """새 API 키 생성"""
        
        try:
            key_id = secrets.token_urlsafe(16)
            api_key = self.generate_api_key(access_level)
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            expires_at = None
            if expires_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
            
            api_key_obj = ApiKey(
                key_id=key_id,
                key_hash=key_hash,
                name=name,
                user_id=user_id,
                permissions=permissions,
                access_level=access_level,
                rate_limit=rate_limit,
                expires_at=expires_at,
                is_active=True,
                created_at=datetime.utcnow(),
                last_used_at=None
            )
            
            self.keys[key_hash] = api_key_obj
            
            logger.info(f"API key created for user {user_id}: {api_key[:8]}...")
            
            return api_key
            
        except Exception as e:
            logger.error(f"API key creation failed: {e}")
            raise SystemException(
                message="API 키 생성에 실패했습니다",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def validate_api_key(self, api_key: str) -> Optional[ApiKey]:
        """API 키 검증"""
        
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            api_key_obj = self.keys.get(key_hash)
            if not api_key_obj:
                return None
            
            # 활성 상태 확인
            if not api_key_obj.is_active:
                return None
            
            # 만료일 확인
            if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
                api_key_obj.is_active = False
                return None
            
            # 사용 정보 업데이트
            api_key_obj.last_used_at = datetime.utcnow()
            api_key_obj.usage_count += 1
            
            return api_key_obj
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """API 키 폐기"""
        
        try:
            for api_key_obj in self.keys.values():
                if api_key_obj.key_id == key_id:
                    api_key_obj.is_active = False
                    logger.info(f"API key revoked: {key_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"API key revocation failed: {e}")
            return False
    
    def get_api_key_info(self, key_id: str) -> Optional[Dict[str, Any]]:
        """API 키 정보 조회"""
        
        for api_key_obj in self.keys.values():
            if api_key_obj.key_id == key_id:
                return {
                    "key_id": api_key_obj.key_id,
                    "name": api_key_obj.name,
                    "user_id": api_key_obj.user_id,
                    "access_level": api_key_obj.access_level.value,
                    "permissions": api_key_obj.permissions,
                    "rate_limit": api_key_obj.rate_limit,
                    "is_active": api_key_obj.is_active,
                    "created_at": api_key_obj.created_at.isoformat(),
                    "last_used_at": api_key_obj.last_used_at.isoformat() if api_key_obj.last_used_at else None,
                    "usage_count": api_key_obj.usage_count,
                    "expires_at": api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None
                }
        
        return None


class EnhancedJWTManager:
    """향상된 JWT 토큰 관리자"""

    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 15  # 짧은 수명
        self.refresh_token_expire_days = 7
        self.revoked_tokens: set = set()
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis.url.split("://")[1].split(":")[0],
                    port=int(settings.redis.url.split(":")[-1]),
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

    def create_token(
        self,
        user_id: str,
        token_type: TokenType,
        additional_claims: Optional[Dict] = None,
        custom_expiry: Optional[timedelta] = None
    ) -> str:
        """토큰 생성"""

        now = datetime.utcnow()

        # 토큰 타입별 만료 시간
        if custom_expiry:
            expire = now + custom_expiry
        elif token_type == TokenType.ACCESS:
            expire = now + timedelta(minutes=self.access_token_expire_minutes)
        elif token_type == TokenType.REFRESH:
            expire = now + timedelta(days=self.refresh_token_expire_days)
        elif token_type == TokenType.SESSION:
            expire = now + timedelta(hours=2)  # 세션은 2시간
        else:
            expire = now + timedelta(minutes=5)  # CSRF는 5분

        # JWT 페이로드
        payload = {
            "sub": user_id,
            "type": token_type.value,
            "iat": now,
            "exp": expire,
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            "iss": "fragrance-ai",
            "aud": "fragrance-ai-api"
        }

        # 추가 클레임
        if additional_claims:
            payload.update(additional_claims)

        # 토큰 생성
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

        # Redis에 활성 토큰 저장 (가능한 경우)
        if self.redis_client and token_type in [TokenType.ACCESS, TokenType.SESSION]:
            try:
                key = f"active_token:{payload['jti']}"
                ttl = int((expire - now).total_seconds())
                self.redis_client.setex(key, ttl, user_id)
            except Exception as e:
                logger.error(f"Failed to store token in Redis: {e}")

        return token

    def verify_token(
        self,
        token: str,
        expected_type: Optional[TokenType] = None,
        verify_exp: bool = True
    ) -> Optional[Dict]:
        """토큰 검증"""

        try:
            # JWT 디코드
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": verify_exp},
                audience="fragrance-ai-api",
                issuer="fragrance-ai"
            )

            # 토큰 타입 검증
            if expected_type and payload.get("type") != expected_type.value:
                logger.warning(f"Invalid token type: expected {expected_type.value}, got {payload.get('type')}")
                return None

            # 토큰 폐기 여부 확인
            jti = payload.get("jti")
            if jti:
                # 메모리에서 확인
                if jti in self.revoked_tokens:
                    logger.warning(f"Token is revoked: {jti}")
                    return None

                # Redis에서 확인 (가능한 경우)
                if self.redis_client:
                    try:
                        key = f"revoked_token:{jti}"
                        if self.redis_client.exists(key):
                            logger.warning(f"Token is revoked in Redis: {jti}")
                            return None
                    except Exception as e:
                        logger.error(f"Failed to check token in Redis: {e}")

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def revoke_token(self, token: str):
        """토큰 폐기"""

        try:
            # 토큰 디코드 (만료 검증 없이)
            payload = self.verify_token(token, verify_exp=False)
            if not payload:
                return False

            jti = payload.get("jti")
            if not jti:
                return False

            # 메모리에 추가
            self.revoked_tokens.add(jti)

            # Redis에 추가 (가능한 경우)
            if self.redis_client:
                try:
                    key = f"revoked_token:{jti}"
                    # 원래 만료 시간까지 저장
                    exp = payload.get("exp")
                    if exp:
                        ttl = exp - datetime.utcnow().timestamp()
                        if ttl > 0:
                            self.redis_client.setex(key, int(ttl), "1")
                except Exception as e:
                    logger.error(f"Failed to revoke token in Redis: {e}")

            logger.info(f"Token revoked: {jti}")
            return True

        except Exception as e:
            logger.error(f"Failed to revoke token: {e}")
            return False

    def create_token_pair(self, user_id: str, additional_claims: Optional[Dict] = None) -> Dict[str, str]:
        """액세스/리프레시 토큰 쌍 생성"""

        access_token = self.create_token(
            user_id,
            TokenType.ACCESS,
            additional_claims
        )

        refresh_token = self.create_token(
            user_id,
            TokenType.REFRESH,
            additional_claims
        )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": self.access_token_expire_minutes * 60
        }

    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """리프레시 토큰으로 액세스 토큰 재발급"""

        payload = self.verify_token(refresh_token, TokenType.REFRESH)
        if not payload:
            return None

        # 새 액세스 토큰 생성
        new_access_token = self.create_token(
            payload["sub"],
            TokenType.ACCESS,
            {k: v for k, v in payload.items() if k not in ["sub", "type", "iat", "exp", "jti"]}
        )

        return new_access_token


class SessionManager:
    """세션 관리자"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis.url.split("://")[1].split(":")[0],
                    port=int(settings.redis.url.split(":")[-1]),
                    decode_responses=False  # 바이너리 데이터 처리
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed for sessions: {e}")
                self.redis_client = None

    def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        additional_data: Optional[Dict] = None
    ) -> str:
        """세션 생성"""

        session_id = secrets.token_urlsafe(32)

        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "is_active": True
        }

        if additional_data:
            session_data.update(additional_data)

        # Redis에 저장 (가능한 경우)
        if self.redis_client:
            try:
                key = f"session:{session_id}"
                self.redis_client.setex(
                    key,
                    7200,  # 2시간
                    json.dumps(session_data)
                )
            except Exception as e:
                logger.error(f"Failed to store session in Redis: {e}")
                # 메모리에 저장
                self.sessions[session_id] = session_data
        else:
            # 메모리에 저장
            self.sessions[session_id] = session_data

        logger.info(f"Session created for user {user_id}: {session_id[:8]}...")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """세션 조회"""

        # Redis에서 조회 (가능한 경우)
        if self.redis_client:
            try:
                key = f"session:{session_id}"
                data = self.redis_client.get(key)
                if data:
                    session_data = json.loads(data)
                    # 마지막 활동 시간 업데이트
                    session_data["last_activity"] = datetime.utcnow().isoformat()
                    self.redis_client.setex(key, 7200, json.dumps(session_data))
                    return session_data
            except Exception as e:
                logger.error(f"Failed to get session from Redis: {e}")

        # 메모리에서 조회
        if session_id in self.sessions:
            session_data = self.sessions[session_id]
            # 마지막 활동 시간 업데이트
            session_data["last_activity"] = datetime.utcnow().isoformat()
            return session_data

        return None

    def invalidate_session(self, session_id: str) -> bool:
        """세션 무효화"""

        success = False

        # Redis에서 삭제 (가능한 경우)
        if self.redis_client:
            try:
                key = f"session:{session_id}"
                if self.redis_client.delete(key):
                    success = True
            except Exception as e:
                logger.error(f"Failed to delete session from Redis: {e}")

        # 메모리에서 삭제
        if session_id in self.sessions:
            del self.sessions[session_id]
            success = True

        if success:
            logger.info(f"Session invalidated: {session_id[:8]}...")

        return success


class CSRFProtection:
    """CSRF 보호"""

    def __init__(self):
        self.token_length = 32
        self.tokens: Dict[str, Tuple[str, datetime]] = {}
        self.token_lifetime = timedelta(minutes=30)

    def generate_token(self, session_id: str) -> str:
        """CSRF 토큰 생성"""

        token = secrets.token_urlsafe(self.token_length)
        self.tokens[token] = (session_id, datetime.utcnow())

        # 오래된 토큰 정리
        self._cleanup_old_tokens()

        return token

    def verify_token(self, token: str, session_id: str) -> bool:
        """CSRF 토큰 검증"""

        if token not in self.tokens:
            return False

        stored_session_id, created_at = self.tokens[token]

        # 세션 ID 일치 확인
        if stored_session_id != session_id:
            return False

        # 만료 시간 확인
        if datetime.utcnow() - created_at > self.token_lifetime:
            del self.tokens[token]
            return False

        return True

    def _cleanup_old_tokens(self):
        """오래된 토큰 정리"""

        now = datetime.utcnow()
        expired_tokens = [
            token for token, (_, created_at) in self.tokens.items()
            if now - created_at > self.token_lifetime
        ]

        for token in expired_tokens:
            del self.tokens[token]


class RateLimiter:
    """향상된 레이트 리미터 (Redis 지원)"""

    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.redis_client = None

        # 기본 제한값들
        self.default_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_limit": 10,
            "block_duration_minutes": 30
        }

        # 역할별 제한값
        self.role_limits = {
            AccessLevel.SUPER_ADMIN: {
                "requests_per_minute": 300,
                "requests_per_hour": 10000
            },
            AccessLevel.ADMIN: {
                "requests_per_minute": 200,
                "requests_per_hour": 5000
            },
            AccessLevel.AUTHORIZED: {
                "requests_per_minute": 100,
                "requests_per_hour": 2000
            },
            AccessLevel.AUTHENTICATED: {
                "requests_per_minute": 60,
                "requests_per_hour": 1000
            },
            AccessLevel.PUBLIC: {
                "requests_per_minute": 30,
                "requests_per_hour": 300
            }
        }

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis.url.split("://")[1].split(":")[0],
                    port=int(settings.redis.url.split(":")[-1]),
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed for rate limiting: {e}")
                self.redis_client = None
    
    def is_allowed(
        self,
        key: str,
        limit_per_minute: int = None,
        limit_per_hour: int = None,
        access_level: Optional[AccessLevel] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """요청 허용 여부 확인"""
        
        with self.lock:
            current_time = time.time()
            
            # 차단된 IP 확인
            if key in self.blocked_ips:
                if current_time - self.blocked_ips[key] < (self.default_limits["block_duration_minutes"] * 60):
                    return False, {"reason": "IP blocked", "blocked_until": self.blocked_ips[key] + (self.default_limits["block_duration_minutes"] * 60)}
                else:
                    del self.blocked_ips[key]
            
            # 역할 기반 제한값 설정
            if access_level and access_level in self.role_limits:
                role_limits = self.role_limits[access_level]
                if limit_per_minute is None:
                    limit_per_minute = role_limits["requests_per_minute"]
                if limit_per_hour is None:
                    limit_per_hour = role_limits["requests_per_hour"]
            else:
                # 기본값 설정
                if limit_per_minute is None:
                    limit_per_minute = self.default_limits["requests_per_minute"]
                if limit_per_hour is None:
                    limit_per_hour = self.default_limits["requests_per_hour"]
            
            # 요청 기록 정리 (1시간 이전 기록 제거)
            hour_ago = current_time - 3600
            self.requests[key] = [req_time for req_time in self.requests[key] if req_time > hour_ago]
            
            # 분당 제한 확인
            minute_ago = current_time - 60
            recent_requests = [req_time for req_time in self.requests[key] if req_time > minute_ago]
            
            if len(recent_requests) >= limit_per_minute:
                # 과도한 요청 시 IP 차단
                if len(recent_requests) >= limit_per_minute * 2:
                    self.blocked_ips[key] = current_time
                    logger.warning(f"IP blocked due to excessive requests: {key}")
                
                return False, {
                    "reason": "Rate limit exceeded (per minute)",
                    "limit": limit_per_minute,
                    "requests": len(recent_requests),
                    "reset_time": minute_ago + 60
                }
            
            # 시간당 제한 확인
            if len(self.requests[key]) >= limit_per_hour:
                return False, {
                    "reason": "Rate limit exceeded (per hour)",
                    "limit": limit_per_hour,
                    "requests": len(self.requests[key]),
                    "reset_time": hour_ago + 3600
                }
            
            # 요청 기록 추가
            self.requests[key].append(current_time)
            
            return True, {
                "remaining_minute": limit_per_minute - len(recent_requests) - 1,
                "remaining_hour": limit_per_hour - len(self.requests[key]),
                "reset_time_minute": minute_ago + 60,
                "reset_time_hour": hour_ago + 3600
            }
    
    def get_stats(self, key: str) -> Dict[str, Any]:
        """레이트 리미트 통계 조회"""
        
        with self.lock:
            current_time = time.time()
            minute_ago = current_time - 60
            hour_ago = current_time - 3600
            
            requests_last_minute = len([
                req_time for req_time in self.requests[key] 
                if req_time > minute_ago
            ])
            
            requests_last_hour = len([
                req_time for req_time in self.requests[key] 
                if req_time > hour_ago
            ])
            
            is_blocked = key in self.blocked_ips
            
            return {
                "requests_last_minute": requests_last_minute,
                "requests_last_hour": requests_last_hour,
                "is_blocked": is_blocked,
                "blocked_until": self.blocked_ips.get(key, 0) + (self.default_limits["block_duration_minutes"] * 60) if is_blocked else None
            }


class SecurityMonitor:
    """고급 보안 모니터링 및 위협 탐지"""

    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.max_events = 10000
        self.lock = threading.Lock()

        # 확장된 위협 패턴 데이터베이스
        self.threat_patterns = {
            'sql_injection': [
                r'(\bUNION\b.*\bSELECT\b)',
                r'(\bSELECT\b.*\bFROM\b.*\bWHERE\b)',
                r'(\bINSERT\b.*\bINTO\b)',
                r'(\bDELETE\b.*\bFROM\b)',
                r'(\bUPDATE\b.*\bSET\b)',
                r'(\bDROP\b.*\bTABLE\b)',
                r'(\bALTER\b.*\bTABLE\b)',
                r'(\bCREATE\b.*\bTABLE\b)',
                r"('.*OR.*')",
                r'(;.*--)',
                r'(\bOR\b.*=.*)',
                r'(\bAND\b.*=.*)',
                r'(\bEXEC\b\s*\()',
                r'(\bEXECUTE\b\s*\()',
                r'(\bsp_executesql\b)',
                r'(\bxp_cmdshell\b)',
                r'(\'.*;\s*--)',
                r'(\".*;\s*--)',
                r'(\b0x[0-9a-fA-F]+)'
            ],
            'xss_attempt': [
                r'<script[^>]*>.*?</script>',
                r'<iframe[^>]*>.*?</iframe>',
                r'<object[^>]*>.*?</object>',
                r'<embed[^>]*>.*?</embed>',
                r'<form[^>]*>.*?</form>',
                r'<img[^>]*onerror[^>]*>',
                r'javascript:',
                r'vbscript:',
                r'data:text/html',
                r'on\w+\s*=',
                r'eval\s*\(',
                r'expression\s*\(',
                r'setTimeout\s*\(',
                r'setInterval\s*\(',
                r'document\.cookie',
                r'document\.write',
                r'window\.location',
                r'alert\s*\(',
                r'confirm\s*\(',
                r'prompt\s*\('
            ],
            'path_traversal': [
                r'\.\./',
                r'\.\.\\',
                r'%2e%2e%2f',
                r'%2e%2e%5c',
                r'%252e%252e%252f',
                r'%c0%ae%c0%ae%c0%af',
                r'%uff0e%uff0e%uff0f',
                r'..%2f',
                r'..%5c',
                r'%2e%2e/',
                r'%2e%2e\\',
                r'\.\.%2f',
                r'\.\.%5c'
            ],
            'command_injection': [
                r';\s*cat\s+',
                r';\s*ls\s+',
                r';\s*dir\s+',
                r';\s*type\s+',
                r';\s*more\s+',
                r';\s*less\s+',
                r';\s*head\s+',
                r';\s*tail\s+',
                r';\s*grep\s+',
                r';\s*find\s+',
                r';\s*locate\s+',
                r';\s*ps\s+',
                r';\s*kill\s+',
                r';\s*rm\s+',
                r';\s*del\s+',
                r';\s*mv\s+',
                r';\s*cp\s+',
                r';\s*chmod\s+',
                r';\s*chown\s+',
                r';\s*wget\s+',
                r';\s*curl\s+',
                r';\s*nc\s+',
                r';\s*netcat\s+',
                r';\s*telnet\s+',
                r';\s*ssh\s+',
                r';\s*ping\s+',
                r';\s*nslookup\s+',
                r'`.*`',
                r'\$\(.*\)',
                r'\|\s*sh',
                r'\|\s*bash',
                r'\|\s*cmd',
                r'\|\s*powershell'
            ],
            'nosql_injection': [
                r'\$where',
                r'\$ne',
                r'\$gt',
                r'\$lt',
                r'\$gte',
                r'\$lte',
                r'\$in',
                r'\$nin',
                r'\$regex',
                r'\$or',
                r'\$and',
                r'\$nor',
                r'\$not',
                r'\$exists',
                r'\$type',
                r'\$mod',
                r'\$all',
                r'\$size',
                r'\$elemMatch'
            ],
            'ldap_injection': [
                r'\*\)',
                r'\(\|',
                r'\(\&',
                r'\(\!',
                r'\).*\(',
                r'admin\*',
                r'\*admin',
                r'objectclass=\*'
            ],
            'header_injection': [
                r'\r\n',
                r'\n',
                r'%0d%0a',
                r'%0a',
                r'%0d',
                r'Content-Type:',
                r'Content-Length:',
                r'Set-Cookie:',
                r'Location:'
            ],
            'file_inclusion': [
                r'php://filter',
                r'php://input',
                r'data://',
                r'file://',
                r'ftp://',
                r'http://',
                r'https://',
                r'expect://',
                r'zip://',
                r'compress.zlib://',
                r'compress.bzip2://'
            ],
            'xml_injection': [
                r'<!ENTITY',
                r'<!DOCTYPE',
                r'SYSTEM\s+["\']',
                r'PUBLIC\s+["\']',
                r'&\w+;',
                r'%\w+;'
            ]
        }
    
    def log_security_event(
        self,
        event_type: str,
        severity: SecurityLevel,
        source_ip: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """보안 이벤트 로그"""
        
        with self.lock:
            event = SecurityEvent(
                event_type=event_type,
                severity=severity,
                source_ip=source_ip,
                user_id=user_id,
                timestamp=datetime.utcnow(),
                details=details or {}
            )
            
            self.events.append(event)
            
            # 이벤트 수 제한
            if len(self.events) > self.max_events:
                self.events = self.events[-self.max_events:]
            
            logger.warning(f"Security event: {event_type} from {source_ip} (severity: {severity.value})")
            
            # 중요한 이벤트는 성능 로거에도 기록
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                performance_logger.log_execution_time(
                    operation="security_event",
                    execution_time=0.0,
                    success=False,
                    extra_data={
                        "event_type": event_type,
                        "severity": severity.value,
                        "source_ip": source_ip,
                        "user_id": user_id
                    }
                )
    
    def check_suspicious_content(self, content: str, source_ip: str) -> List[str]:
        """의심스러운 콘텐츠 검사"""
        
        suspicious_findings = []
        content_lower = content.lower()
        
        for pattern_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    suspicious_findings.append(pattern_type)
                    
                    self.log_security_event(
                        event_type=f"suspicious_content_{pattern_type}",
                        severity=SecurityLevel.HIGH,
                        source_ip=source_ip,
                        details={
                            "pattern": pattern,
                            "content_snippet": content[:100]
                        }
                    )
                    break
        
        return suspicious_findings
    
    def analyze_request_pattern(self, ip: str, endpoint: str, user_agent: str) -> Dict[str, Any]:
        """요청 패턴 분석"""
        
        analysis = {
            "is_suspicious": False,
            "reasons": [],
            "risk_score": 0.0
        }
        
        # User-Agent 분석
        if not user_agent or len(user_agent) < 10:
            analysis["reasons"].append("suspicious_user_agent")
            analysis["risk_score"] += 0.3
        
        # 자동화된 요청 패턴 감지
        bot_patterns = ['bot', 'crawler', 'spider', 'scraper']
        if any(pattern in user_agent.lower() for pattern in bot_patterns):
            analysis["reasons"].append("bot_pattern")
            analysis["risk_score"] += 0.2
        
        # IP 주소 분석
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private:
                analysis["reasons"].append("private_ip")
                analysis["risk_score"] += 0.1
        except ValueError:
            analysis["reasons"].append("invalid_ip")
            analysis["risk_score"] += 0.5
        
        # 위험 점수 기준으로 의심스러운 요청 판단
        if analysis["risk_score"] > 0.5:
            analysis["is_suspicious"] = True
            
            self.log_security_event(
                event_type="suspicious_request_pattern",
                severity=SecurityLevel.MEDIUM,
                source_ip=ip,
                details={
                    "endpoint": endpoint,
                    "user_agent": user_agent,
                    "risk_score": analysis["risk_score"],
                    "reasons": analysis["reasons"]
                }
            )
        
        return analysis
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """보안 요약 정보"""
        
        with self.lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_events = [
                event for event in self.events 
                if event.timestamp > cutoff_time
            ]
            
            summary = {
                "total_events": len(recent_events),
                "events_by_severity": defaultdict(int),
                "events_by_type": defaultdict(int),
                "unique_ips": set(),
                "affected_users": set()
            }
            
            for event in recent_events:
                summary["events_by_severity"][event.severity.value] += 1
                summary["events_by_type"][event.event_type] += 1
                summary["unique_ips"].add(event.source_ip)
                if event.user_id:
                    summary["affected_users"].add(event.user_id)
            
            # Set을 리스트로 변환 (JSON 직렬화를 위해)
            summary["unique_ips"] = list(summary["unique_ips"])
            summary["affected_users"] = list(summary["affected_users"])
            summary["events_by_severity"] = dict(summary["events_by_severity"])
            summary["events_by_type"] = dict(summary["events_by_type"])
            
            return summary


# 전역 인스턴스들
password_validator = PasswordValidator()
encryption_manager = EncryptionManager()
api_key_manager = ApiKeyManager()
jwt_manager = EnhancedJWTManager()
session_manager = SessionManager()
csrf_protection = CSRFProtection()
rate_limiter = RateLimiter()
security_monitor = SecurityMonitor()


class SecurityManager:
    """통합 보안 관리자"""

    def __init__(self):
        self.password_validator = password_validator
        self.encryption_manager = encryption_manager
        self.api_key_manager = api_key_manager
        self.jwt_manager = jwt_manager
        self.session_manager = session_manager
        self.csrf_protection = csrf_protection
        self.rate_limiter = rate_limiter
        self.security_monitor = security_monitor

    def validate_password(self, password: str) -> bool:
        """비밀번호 검증"""
        return self.password_validator.validate_password(password)

    def create_api_key(self, user_id: str, name: str, **kwargs) -> str:
        """API 키 생성"""
        return self.api_key_manager.create_api_key(user_id, name, **kwargs)

    def validate_api_key(self, api_key: str) -> Optional['ApiKey']:
        """API 키 검증"""
        return self.api_key_manager.validate_api_key(api_key)

    def check_rate_limit(self, identifier: str, **kwargs) -> Tuple[bool, Dict]:
        """속도 제한 확인"""
        return self.rate_limiter.is_allowed(identifier, **kwargs)

    def create_jwt_token(self, user_id: str, token_type: TokenType, **kwargs) -> str:
        """JWT 토큰 생성"""
        return self.jwt_manager.create_token(user_id, token_type, **kwargs)

    def verify_jwt_token(self, token: str, **kwargs) -> Optional[Dict]:
        """JWT 토큰 검증"""
        return self.jwt_manager.verify_token(token, **kwargs)

    def create_session(self, user_id: str, ip_address: str, user_agent: str, **kwargs) -> str:
        """세션 생성"""
        return self.session_manager.create_session(user_id, ip_address, user_agent, **kwargs)

    def get_session(self, session_id: str) -> Optional[Dict]:
        """세션 조회"""
        return self.session_manager.get_session(session_id)

    def generate_csrf_token(self, session_id: str) -> str:
        """CSRF 토큰 생성"""
        return self.csrf_protection.generate_token(session_id)

    def verify_csrf_token(self, token: str, session_id: str) -> bool:
        """CSRF 토큰 검증"""
        return self.csrf_protection.verify_token(token, session_id)

    def verify_admin_privileges(
        self,
        token: str,
        required_level: AccessLevel = AccessLevel.ADMIN,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[Dict]]:
        """관리자 권한 검증"""

        # JWT 토큰 검증
        payload = self.jwt_manager.verify_token(token, TokenType.ACCESS)
        if not payload:
            return False, None

        # 접근 레벨 확인
        user_level = payload.get("access_level")
        if not user_level:
            return False, None

        # 레벨 비교 (enum 순서 기반)
        level_order = {
            AccessLevel.PUBLIC: 0,
            AccessLevel.AUTHENTICATED: 1,
            AccessLevel.AUTHORIZED: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SUPER_ADMIN: 4
        }

        if level_order.get(AccessLevel(user_level), 0) < level_order.get(required_level, 3):
            logger.warning(f"Insufficient privileges: {user_level} < {required_level}")
            return False, None

        # IP 주소 검증 (선택적)
        if ip_address and payload.get("ip_address") and payload["ip_address"] != ip_address:
            logger.warning(f"IP address mismatch: {payload['ip_address']} != {ip_address}")
            return False, None

        return True, payload

    def log_security_event(self, **kwargs):
        """보안 이벤트 로깅"""
        return self.security_monitor.log_security_event(**kwargs)


# 전역 보안 관리자 인스턴스
security_manager = SecurityManager()