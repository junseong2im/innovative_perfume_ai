import hashlib
import hmac
import secrets
import time
import json
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

from .config import settings
from .logging_config import get_logger, performance_logger
from .exceptions import (
    AuthenticationException, ValidationException, SystemException,
    ErrorCode, FragranceAIException
)

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
        self.master_key = self._derive_master_key()
        self.fernet = Fernet(self.master_key)
    
    def _derive_master_key(self) -> bytes:
        """마스터 키 유도"""
        
        password = settings.secret_key.encode()
        salt = b"fragrance_ai_salt_2024"  # 실제 환경에서는 더 안전한 방법 사용
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
    
    def encrypt(self, data: str) -> str:
        """데이터 암호화"""
        
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SystemException(
                message="데이터 암호화에 실패했습니다",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def decrypt(self, encrypted_data: str) -> str:
        """데이터 복호화"""
        
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
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


class RateLimiter:
    """레이트 리미터"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked_ips: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        # 기본 제한값들
        self.default_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_limit": 10,
            "block_duration_minutes": 30
        }
    
    def is_allowed(
        self,
        key: str,
        limit_per_minute: int = None,
        limit_per_hour: int = None
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
    """보안 모니터링"""
    
    def __init__(self):
        self.events: List[SecurityEvent] = []
        self.max_events = 10000
        self.suspicious_patterns = {
            'sql_injection': [r'union\s+select', r'drop\s+table', r'exec\s*\('],
            'xss_attempt': [r'<script', r'javascript:', r'onload='],
            'path_traversal': [r'\.\./', r'\.\.\\', r'%2e%2e%2f'],
            'command_injection': [r';\s*cat\s+', r';\s*ls\s+', r'`.*`']
        }
        self.lock = threading.Lock()
    
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
rate_limiter = RateLimiter()
security_monitor = SecurityMonitor()