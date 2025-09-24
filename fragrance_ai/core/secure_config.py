"""
보안 강화된 설정 관리 시스템

환경별 설정을 안전하게 관리하고, 민감한 정보를 암호화하여 저장합니다.
환경 변수, 비밀 관리 서비스, 설정 파일을 통합하여 관리합니다.
"""

import os
import json
import base64
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
import hashlib

from .production_logging import get_logger

logger = get_logger(__name__)


class Environment(str, Enum):
    """환경 유형"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class SecretLevel(str, Enum):
    """시크릿 보안 레벨"""
    LOW = "low"          # 일반 설정값
    MEDIUM = "medium"    # API 키 등
    HIGH = "high"        # 데이터베이스 비밀번호 등
    CRITICAL = "critical" # 암호화 키, 인증서 등


@dataclass
class SecretConfig:
    """시크릿 설정"""
    key: str
    value: str
    level: SecretLevel
    description: str = ""
    encrypted: bool = False
    expires_at: Optional[str] = None
    allowed_environments: List[Environment] = field(default_factory=list)


class SecretManager:
    """시크릿 관리자"""

    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or os.getenv('FRAGRANCE_AI_MASTER_KEY')
        self._cipher = None
        self._secrets: Dict[str, SecretConfig] = {}

        if self.master_key:
            self._initialize_cipher()

    def _initialize_cipher(self):
        """암호화 초기화"""
        try:
            # 마스터 키에서 암호화 키 생성
            password = self.master_key.encode()
            salt = b'fragrance_ai_salt_2024'  # 실제로는 랜덤 salt 사용

            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self._cipher = Fernet(key)

            logger.info("Secret manager initialized with encryption")

        except Exception as e:
            logger.error(f"Failed to initialize cipher: {e}")
            self._cipher = None

    def encrypt_value(self, value: str) -> str:
        """값 암호화"""
        if not self._cipher:
            logger.warning("Cipher not available, storing value in plain text")
            return value

        try:
            encrypted_value = self._cipher.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted_value).decode()

        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            return value

    def decrypt_value(self, encrypted_value: str) -> str:
        """값 복호화"""
        if not self._cipher:
            return encrypted_value

        try:
            decoded_value = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_value = self._cipher.decrypt(decoded_value)
            return decrypted_value.decode()

        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value

    def add_secret(self, config: SecretConfig, encrypt: bool = True):
        """시크릿 추가"""
        if encrypt and config.level in [SecretLevel.HIGH, SecretLevel.CRITICAL]:
            config.value = self.encrypt_value(config.value)
            config.encrypted = True

        self._secrets[config.key] = config
        logger.debug(f"Added secret: {config.key} (level: {config.level.value})")

    def get_secret(self, key: str, environment: Environment) -> Optional[str]:
        """시크릿 조회"""
        config = self._secrets.get(key)
        if not config:
            return None

        # 환경 제한 확인
        if config.allowed_environments and environment not in config.allowed_environments:
            logger.warning(f"Secret {key} not allowed in environment {environment.value}")
            return None

        # 값 복호화
        value = config.value
        if config.encrypted:
            value = self.decrypt_value(value)

        return value

    def load_secrets_from_file(self, file_path: Path):
        """파일에서 시크릿 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for secret_data in data.get('secrets', []):
                config = SecretConfig(
                    key=secret_data['key'],
                    value=secret_data['value'],
                    level=SecretLevel(secret_data.get('level', SecretLevel.LOW.value)),
                    description=secret_data.get('description', ''),
                    encrypted=secret_data.get('encrypted', False),
                    expires_at=secret_data.get('expires_at'),
                    allowed_environments=[
                        Environment(env) for env in secret_data.get('allowed_environments', [])
                    ]
                )
                self.add_secret(config, encrypt=not config.encrypted)

            logger.info(f"Loaded {len(data.get('secrets', []))} secrets from {file_path}")

        except Exception as e:
            logger.error(f"Failed to load secrets from {file_path}: {e}")

    def export_secrets_to_file(self, file_path: Path, include_encrypted: bool = False):
        """시크릿을 파일로 내보내기"""
        try:
            secrets_data = []

            for config in self._secrets.values():
                secret_data = {
                    'key': config.key,
                    'level': config.level.value,
                    'description': config.description,
                    'encrypted': config.encrypted,
                    'allowed_environments': [env.value for env in config.allowed_environments]
                }

                # 암호화된 값 포함 여부
                if include_encrypted or not config.encrypted:
                    secret_data['value'] = config.value
                else:
                    secret_data['value'] = '[ENCRYPTED]'

                if config.expires_at:
                    secret_data['expires_at'] = config.expires_at

                secrets_data.append(secret_data)

            data = {
                'version': '1.0',
                'created_at': str(pd.Timestamp.now()),
                'secrets': secrets_data
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(secrets_data)} secrets to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export secrets to {file_path}: {e}")


class SecureConfigManager:
    """보안 강화된 설정 관리자"""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.secret_manager = SecretManager()
        self._config: Dict[str, Any] = {}
        self._config_sources: List[str] = []

        # 기본 설정 로드
        self._load_default_config()
        self._load_environment_config()
        self._load_secrets()
        self._validate_config()

    def _load_default_config(self):
        """기본 설정 로드"""
        default_config = {
            # 애플리케이션 기본 설정
            'app': {
                'name': 'Fragrance AI',
                'version': '2.0.0',
                'debug': self.environment == Environment.DEVELOPMENT,
                'log_level': 'DEBUG' if self.environment == Environment.DEVELOPMENT else 'INFO'
            },

            # API 설정
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'prefix': '/api/v1',
                'cors_origins': ['http://localhost:3000'] if self.environment == Environment.DEVELOPMENT else [],
                'rate_limit': {
                    'enabled': self.environment != Environment.DEVELOPMENT,
                    'requests_per_minute': 100,
                    'requests_per_hour': 5000
                }
            },

            # 보안 설정
            'security': {
                'secret_key_required': self.environment != Environment.DEVELOPMENT,
                'jwt_expire_minutes': 30,
                'password_min_length': 8,
                'max_login_attempts': 5,
                'session_timeout_hours': 24,
                'require_https': self.environment == Environment.PRODUCTION,
                'csrf_protection': self.environment == Environment.PRODUCTION
            },

            # 데이터베이스 설정
            'database': {
                'echo_sql': self.environment == Environment.DEVELOPMENT,
                'pool_size': 5 if self.environment == Environment.DEVELOPMENT else 20,
                'max_overflow': 10 if self.environment == Environment.DEVELOPMENT else 50,
                'pool_timeout': 30,
                'pool_recycle': 3600,
                'connection_timeout': 10
            },

            # 캐시 설정
            'cache': {
                'default_ttl': 3600,
                'max_size': 1000 if self.environment == Environment.DEVELOPMENT else 10000,
                'cleanup_interval': 300
            },

            # 모니터링 설정
            'monitoring': {
                'enabled': self.environment != Environment.DEVELOPMENT,
                'metrics_enabled': True,
                'health_check_interval': 30,
                'alert_thresholds': {
                    'error_rate': 0.05,
                    'response_time_p95': 2000,
                    'memory_usage': 0.8,
                    'cpu_usage': 0.8
                }
            },

            # AI 모델 설정
            'ai': {
                'embedding_batch_size': 32,
                'generation_batch_size': 4,
                'max_sequence_length': 512,
                'model_cache_size': 2,
                'use_quantization': self.environment == Environment.PRODUCTION,
                'device': 'cpu'  # GPU 사용 시 'cuda'
            }
        }

        self._config = default_config
        self._config_sources.append('default')

    def _load_environment_config(self):
        """환경별 설정 로드"""
        config_file = f"config/config-{self.environment.value}.json"
        config_path = Path(config_file)

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    env_config = json.load(f)

                self._merge_config(env_config)
                self._config_sources.append(f'file:{config_file}')
                logger.info(f"Loaded environment config from {config_file}")

            except Exception as e:
                logger.error(f"Failed to load environment config from {config_file}: {e}")

        # 환경 변수에서 설정 오버라이드
        self._load_from_environment_variables()

    def _load_from_environment_variables(self):
        """환경 변수에서 설정 로드"""
        env_mappings = {
            # 애플리케이션
            'DEBUG': 'app.debug',
            'LOG_LEVEL': 'app.log_level',

            # API
            'API_HOST': 'api.host',
            'API_PORT': 'api.port',
            'API_PREFIX': 'api.prefix',
            'CORS_ORIGINS': 'api.cors_origins',

            # 보안
            'SECRET_KEY': 'security.secret_key',
            'JWT_EXPIRE_MINUTES': 'security.jwt_expire_minutes',
            'REQUIRE_HTTPS': 'security.require_https',

            # 데이터베이스
            'DATABASE_URL': 'database.url',
            'DATABASE_POOL_SIZE': 'database.pool_size',
            'DATABASE_MAX_OVERFLOW': 'database.max_overflow',

            # 캐시
            'REDIS_URL': 'cache.redis_url',
            'CACHE_DEFAULT_TTL': 'cache.default_ttl',

            # AI
            'EMBEDDING_MODEL_NAME': 'ai.embedding_model_name',
            'GENERATION_MODEL_NAME': 'ai.generation_model_name',
            'AI_DEVICE': 'ai.device',
        }

        for env_key, config_key in env_mappings.items():
            value = os.getenv(env_key)
            if value is not None:
                self._set_nested_value(config_key, self._parse_env_value(value))

        self._config_sources.append('environment_variables')

    def _load_secrets(self):
        """시크릿 로드"""
        # 시크릿 파일 로드
        secrets_file = f"config/secrets-{self.environment.value}.json"
        secrets_path = Path(secrets_file)

        if secrets_path.exists():
            self.secret_manager.load_secrets_from_file(secrets_path)

        # 주요 시크릿들을 설정에 추가
        secret_keys = [
            'SECRET_KEY', 'DATABASE_PASSWORD', 'REDIS_PASSWORD',
            'JWT_SECRET_KEY', 'ENCRYPTION_KEY', 'API_SECRET_KEY'
        ]

        for key in secret_keys:
            secret_value = self.secret_manager.get_secret(key, self.environment)
            if secret_value:
                # 시크릿을 안전한 경로에 설정
                safe_key = key.lower().replace('_', '.')
                self._set_nested_value(f"secrets.{safe_key}", secret_value)

    def _validate_config(self):
        """설정 검증"""
        required_keys = [
            'app.name',
            'api.host',
            'api.port'
        ]

        if self.environment == Environment.PRODUCTION:
            required_keys.extend([
                'security.secret_key',
                'database.url'
            ])

        missing_keys = []
        for key in required_keys:
            if not self._get_nested_value(key):
                missing_keys.append(key)

        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

        # 보안 검증
        self._validate_security_config()

    def _validate_security_config(self):
        """보안 설정 검증"""
        if self.environment == Environment.PRODUCTION:
            # HTTPS 강제 확인
            if not self.get('security.require_https', False):
                logger.warning("HTTPS is not required in production environment")

            # 강력한 시크릿 키 확인
            secret_key = self.get('security.secret_key')
            if secret_key and len(secret_key) < 32:
                raise ValueError("Secret key must be at least 32 characters long")

            # CORS 설정 확인
            cors_origins = self.get('api.cors_origins', [])
            if '*' in cors_origins:
                raise ValueError("Wildcard CORS origins not allowed in production")

    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        return self._get_nested_value(key) or default

    def set(self, key: str, value: Any):
        """설정값 설정"""
        self._set_nested_value(key, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """설정 섹션 조회"""
        return self._get_nested_value(section) or {}

    def _get_nested_value(self, key: str) -> Any:
        """중첩된 키로 값 조회"""
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None

        return value

    def _set_nested_value(self, key: str, value: Any):
        """중첩된 키로 값 설정"""
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def _merge_config(self, new_config: Dict[str, Any]):
        """설정 병합"""
        def merge_dicts(base: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dicts(base[key], value)
                else:
                    base[key] = value
            return base

        merge_dicts(self._config, new_config)

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, List]:
        """환경 변수 값 파싱"""
        # 불린 값
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 숫자 값
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # JSON 배열/객체
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # 쉼표로 구분된 리스트
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        return value

    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보"""
        return {
            'environment': self.environment.value,
            'config_sources': self._config_sources,
            'sections': list(self._config.keys()),
            'validation_status': 'valid',
            'secrets_loaded': len(self.secret_manager._secrets),
            'last_updated': str(pd.Timestamp.now())
        }

    def export_config(self, file_path: Path, exclude_secrets: bool = True):
        """설정 내보내기"""
        try:
            export_data = {
                'environment': self.environment.value,
                'config_sources': self._config_sources,
                'timestamp': str(pd.Timestamp.now()),
                'config': self._config.copy()
            }

            # 시크릿 제외
            if exclude_secrets and 'secrets' in export_data['config']:
                del export_data['config']['secrets']

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Config exported to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export config to {file_path}: {e}")


# 전역 설정 관리자 인스턴스
_config_manager: Optional[SecureConfigManager] = None


def get_config_manager(environment: Optional[Environment] = None) -> SecureConfigManager:
    """설정 관리자 인스턴스 반환"""
    global _config_manager

    if _config_manager is None:
        env = environment or Environment(os.getenv('ENVIRONMENT', 'development'))
        _config_manager = SecureConfigManager(env)

    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """설정값 조회 (편의 함수)"""
    return get_config_manager().get(key, default)


def get_secret(key: str) -> Optional[str]:
    """시크릿 조회 (편의 함수)"""
    config_manager = get_config_manager()
    return config_manager.secret_manager.get_secret(key, config_manager.environment)