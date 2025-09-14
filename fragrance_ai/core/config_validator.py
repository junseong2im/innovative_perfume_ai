from typing import Dict, List, Any, Optional, Tuple, Union
import os
import re
import socket
import urllib.parse
from pathlib import Path
import requests
import psycopg2
import redis
from enum import Enum
import json

from .config import settings
from .exceptions import FragranceAIException, ErrorCode, ValidationException
from .logging_config import get_logger

logger = get_logger(__name__)


class ValidationSeverity(str, Enum):
    """검증 결과 심각도"""
    ERROR = "error"      # 시스템 시작 불가
    WARNING = "warning"  # 기능 제한 가능성
    INFO = "info"        # 권장사항


class ValidationResult:
    """검증 결과 클래스"""
    
    def __init__(
        self,
        field: str,
        severity: ValidationSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.field = field
        self.severity = severity
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details
        }


class ConfigValidator:
    """설정 검증 클래스"""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
    
    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """모든 설정값 검증"""
        
        self.validation_results.clear()
        
        logger.info("Starting configuration validation...")
        
        # 각 카테고리별 검증
        self._validate_app_settings()
        self._validate_api_settings()
        self._validate_database_settings()
        self._validate_ai_model_settings()
        self._validate_huggingface_settings()
        self._validate_vector_database_settings()
        self._validate_training_settings()
        self._validate_search_settings()
        self._validate_generation_settings()
        self._validate_security_settings()
        self._validate_monitoring_settings()
        self._validate_performance_settings()
        self._validate_directories_and_permissions()
        
        # 결과 요약
        error_count = len([r for r in self.validation_results if r.severity == ValidationSeverity.ERROR])
        warning_count = len([r for r in self.validation_results if r.severity == ValidationSeverity.WARNING])
        info_count = len([r for r in self.validation_results if r.severity == ValidationSeverity.INFO])
        
        logger.info(f"Configuration validation completed", extra={
            "total_checks": len(self.validation_results),
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count
        })
        
        # 에러가 있으면 실패
        is_valid = error_count == 0
        
        if not is_valid:
            error_messages = [r.message for r in self.validation_results if r.severity == ValidationSeverity.ERROR]
            logger.error("Configuration validation failed", extra={
                "error_messages": error_messages
            })
        
        return is_valid, self.validation_results
    
    def _validate_app_settings(self):
        """앱 기본 설정 검증"""
        
        # 앱 이름 검증
        if not settings.app_name or len(settings.app_name.strip()) == 0:
            self._add_error("app_name", "Application name cannot be empty")
        
        # 버전 형식 검증
        version_pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
        if not re.match(version_pattern, settings.app_version):
            self._add_warning("app_version", f"Version '{settings.app_version}' does not follow semantic versioning")
    
    def _validate_api_settings(self):
        """API 설정 검증"""
        
        # 호스트 검증
        if settings.api_host not in ["0.0.0.0", "127.0.0.1", "localhost"]:
            try:
                socket.inet_aton(settings.api_host)
            except socket.error:
                self._add_error("api_host", f"Invalid IP address: {settings.api_host}")
        
        # 포트 검증
        if not 1 <= settings.api_port <= 65535:
            self._add_error("api_port", f"Port must be between 1-65535, got {settings.api_port}")
        
        # 포트 사용 가능성 검증
        if not self._is_port_available(settings.api_host, settings.api_port):
            self._add_warning("api_port", f"Port {settings.api_port} may already be in use")
        
        # API prefix 검증
        if not settings.api_prefix.startswith('/'):
            self._add_error("api_prefix", "API prefix must start with '/'")
    
    def _validate_database_settings(self):
        """데이터베이스 설정 검증"""
        
        try:
            # DATABASE_URL 파싱
            parsed = urllib.parse.urlparse(settings.database_url)
            
            if parsed.scheme != 'postgresql':
                self._add_error("database_url", "Only PostgreSQL is supported")
                return
            
            if not parsed.hostname:
                self._add_error("database_url", "Database hostname is missing")
            
            if not parsed.port:
                self._add_warning("database_url", "Database port not specified, using default")
            elif not 1 <= parsed.port <= 65535:
                self._add_error("database_url", f"Invalid database port: {parsed.port}")
            
            if not parsed.username:
                self._add_error("database_url", "Database username is missing")
            
            if not parsed.password:
                self._add_warning("database_url", "Database password is missing")
            
            if not parsed.path or parsed.path == '/':
                self._add_error("database_url", "Database name is missing")
            
            # 연결 테스트 (선택적)
            if self._should_test_connections():
                self._test_database_connection(settings.database_url)
                
        except Exception as e:
            self._add_error("database_url", f"Invalid database URL format: {str(e)}")
        
        # Redis URL 검증
        try:
            parsed_redis = urllib.parse.urlparse(settings.redis_url)
            if parsed_redis.scheme != 'redis':
                self._add_warning("redis_url", "Non-standard Redis URL scheme")
            
            if self._should_test_connections():
                self._test_redis_connection(settings.redis_url)
                
        except Exception as e:
            self._add_error("redis_url", f"Invalid Redis URL format: {str(e)}")
    
    def _validate_ai_model_settings(self):
        """AI 모델 설정 검증"""
        
        # 임베딩 모델 검증
        if not settings.embedding_model_name:
            self._add_error("embedding_model_name", "Embedding model name cannot be empty")
        else:
            # Hugging Face 모델 이름 형식 검증
            if '/' not in settings.embedding_model_name:
                self._add_warning("embedding_model_name", "Model name should include organization/model format")
        
        # 생성 모델 검증
        if not settings.generation_model_name:
            self._add_error("generation_model_name", "Generation model name cannot be empty")
        else:
            if '/' not in settings.generation_model_name:
                self._add_warning("generation_model_name", "Model name should include organization/model format")
    
    def _validate_huggingface_settings(self):
        """Hugging Face 설정 검증"""
        
        # HF 토큰 검증 (선택적)
        if settings.hf_token:
            if len(settings.hf_token) < 10:
                self._add_warning("hf_token", "HF token seems too short")
            
            # 토큰 형식 검증 (hf_로 시작하는지)
            if not settings.hf_token.startswith('hf_'):
                self._add_info("hf_token", "HF token should typically start with 'hf_'")
        else:
            self._add_info("hf_token", "No HuggingFace token provided - limited to public models")
        
        # 캐시 디렉토리 검증
        cache_path = Path(settings.hf_cache_dir)
        if not self._validate_directory_path(cache_path, create_if_missing=True):
            self._add_error("hf_cache_dir", f"Cannot access HuggingFace cache directory: {settings.hf_cache_dir}")
    
    def _validate_vector_database_settings(self):
        """벡터 데이터베이스 설정 검증"""
        
        # Chroma 저장 디렉토리 검증
        chroma_path = Path(settings.chroma_persist_directory)
        if not self._validate_directory_path(chroma_path, create_if_missing=True):
            self._add_error("chroma_persist_directory", f"Cannot access Chroma directory: {settings.chroma_persist_directory}")
        
        # 벡터 차원 검증
        if not 64 <= settings.vector_dimension <= 4096:
            self._add_warning("vector_dimension", f"Unusual vector dimension: {settings.vector_dimension}")
    
    def _validate_training_settings(self):
        """훈련 설정 검증"""
        
        # 시퀀스 길이 검증
        if not 64 <= settings.max_seq_length <= 4096:
            self._add_warning("max_seq_length", f"Unusual max sequence length: {settings.max_seq_length}")
        
        # 배치 크기 검증
        if not 1 <= settings.batch_size <= 128:
            self._add_warning("batch_size", f"Unusual batch size: {settings.batch_size}")
        
        # 학습률 검증
        if not 1e-6 <= settings.learning_rate <= 1e-2:
            self._add_warning("learning_rate", f"Unusual learning rate: {settings.learning_rate}")
        
        # LoRA 설정 검증
        if not 4 <= settings.lora_r <= 128:
            self._add_warning("lora_r", f"Unusual LoRA r value: {settings.lora_r}")
        
        if settings.lora_alpha < settings.lora_r:
            self._add_warning("lora_alpha", "LoRA alpha should typically be >= r value")
        
        if not 0.0 <= settings.lora_dropout <= 0.5:
            self._add_warning("lora_dropout", f"Unusual LoRA dropout: {settings.lora_dropout}")
    
    def _validate_search_settings(self):
        """검색 설정 검증"""
        
        if not 1 <= settings.search_top_k <= 100:
            self._add_warning("search_top_k", f"Unusual search top_k: {settings.search_top_k}")
        
        if not 0.0 <= settings.similarity_threshold <= 1.0:
            self._add_error("similarity_threshold", f"Similarity threshold must be between 0.0-1.0: {settings.similarity_threshold}")
    
    def _validate_generation_settings(self):
        """생성 설정 검증"""
        
        if not 16 <= settings.max_new_tokens <= 2048:
            self._add_warning("max_new_tokens", f"Unusual max new tokens: {settings.max_new_tokens}")
        
        if not 0.1 <= settings.temperature <= 2.0:
            self._add_warning("temperature", f"Unusual temperature: {settings.temperature}")
        
        if not 0.1 <= settings.top_p <= 1.0:
            self._add_error("top_p", f"Top-p must be between 0.1-1.0: {settings.top_p}")
    
    def _validate_security_settings(self):
        """보안 설정 검증"""
        
        # 시크릿 키 검증
        if settings.secret_key == "your-super-secret-key-change-in-production":
            self._add_error("secret_key", "Default secret key must be changed for production")
        
        if len(settings.secret_key) < 32:
            self._add_error("secret_key", "Secret key should be at least 32 characters")
        
        # 토큰 만료 시간 검증
        if not 5 <= settings.access_token_expire_minutes <= 1440:  # 5분 ~ 24시간
            self._add_warning("access_token_expire_minutes", f"Unusual token expiry: {settings.access_token_expire_minutes} minutes")
    
    def _validate_monitoring_settings(self):
        """모니터링 설정 검증"""
        
        # 로그 레벨 검증
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if settings.log_level not in valid_log_levels:
            self._add_error("log_level", f"Invalid log level: {settings.log_level}")
        
        # WandB 프로젝트 이름 검증
        if settings.wandb_project:
            # 프로젝트 이름 형식 검증
            if not re.match(r'^[a-zA-Z0-9_-]+$', settings.wandb_project):
                self._add_warning("wandb_project", "WandB project name contains invalid characters")
    
    def _validate_performance_settings(self):
        """성능 설정 검증"""
        
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if settings.max_workers > cpu_count:
            self._add_warning("max_workers", f"Max workers ({settings.max_workers}) exceeds CPU count ({cpu_count})")
        
        if settings.max_workers < 1:
            self._add_error("max_workers", "Max workers must be at least 1")
        
        if settings.model_cache_size < 1:
            self._add_error("model_cache_size", "Model cache size must be at least 1")
        
        if settings.model_cache_size > 10:
            self._add_warning("model_cache_size", f"Large model cache size may consume excessive memory: {settings.model_cache_size}")
    
    def _validate_directories_and_permissions(self):
        """디렉토리 및 권한 검증"""
        
        # 주요 디렉토리들 검증
        directories_to_check = [
            ("data", True),
            ("logs", True),
            ("cache", True),
            ("models", True),
        ]
        
        for dir_name, should_create in directories_to_check:
            dir_path = Path(dir_name)
            if not self._validate_directory_path(dir_path, create_if_missing=should_create):
                severity = ValidationSeverity.ERROR if not should_create else ValidationSeverity.WARNING
                self._add_result(severity, dir_name, f"Directory access issue: {dir_name}")
    
    def _validate_directory_path(self, path: Path, create_if_missing: bool = False) -> bool:
        """디렉토리 경로 검증 및 생성"""
        
        try:
            if not path.exists() and create_if_missing:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            
            if not path.exists():
                return False
            
            if not path.is_dir():
                return False
            
            # 읽기 권한 확인
            if not os.access(path, os.R_OK):
                return False
            
            # 쓰기 권한 확인
            if not os.access(path, os.W_OK):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating directory {path}: {e}")
            return False
    
    def _is_port_available(self, host: str, port: int) -> bool:
        """포트 사용 가능성 확인"""
        
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0
        except Exception:
            return False
    
    def _should_test_connections(self) -> bool:
        """연결 테스트 수행 여부 결정"""
        # 개발 환경에서만 연결 테스트
        return settings.debug
    
    def _test_database_connection(self, database_url: str):
        """데이터베이스 연결 테스트"""
        
        try:
            conn = psycopg2.connect(database_url, connect_timeout=5)
            conn.close()
            self._add_info("database_url", "Database connection test successful")
        except Exception as e:
            self._add_error("database_url", f"Database connection failed: {str(e)}")
    
    def _test_redis_connection(self, redis_url: str):
        """Redis 연결 테스트"""
        
        try:
            r = redis.from_url(redis_url, socket_connect_timeout=5)
            r.ping()
            self._add_info("redis_url", "Redis connection test successful")
        except Exception as e:
            self._add_warning("redis_url", f"Redis connection failed: {str(e)}")
    
    def _add_error(self, field: str, message: str, details: Optional[Dict[str, Any]] = None):
        """에러 결과 추가"""
        self._add_result(ValidationSeverity.ERROR, field, message, details)
    
    def _add_warning(self, field: str, message: str, details: Optional[Dict[str, Any]] = None):
        """경고 결과 추가"""
        self._add_result(ValidationSeverity.WARNING, field, message, details)
    
    def _add_info(self, field: str, message: str, details: Optional[Dict[str, Any]] = None):
        """정보 결과 추가"""
        self._add_result(ValidationSeverity.INFO, field, message, details)
    
    def _add_result(
        self,
        severity: ValidationSeverity,
        field: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """검증 결과 추가"""
        result = ValidationResult(field, severity, message, details)
        self.validation_results.append(result)


def validate_configuration() -> Tuple[bool, List[ValidationResult]]:
    """설정 검증 메인 함수"""
    
    validator = ConfigValidator()
    is_valid, results = validator.validate_all()
    
    # 검증 결과를 로그 파일에 저장
    validation_report = {
        "timestamp": "2024-01-01T00:00:00Z",  # 실제 구현시 datetime.utcnow().isoformat()
        "is_valid": is_valid,
        "summary": {
            "total": len(results),
            "errors": len([r for r in results if r.severity == ValidationSeverity.ERROR]),
            "warnings": len([r for r in results if r.severity == ValidationSeverity.WARNING]),
            "info": len([r for r in results if r.severity == ValidationSeverity.INFO])
        },
        "results": [r.to_dict() for r in results]
    }
    
    # 검증 결과 저장
    try:
        report_path = Path("logs/config_validation.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save validation report: {e}")
    
    return is_valid, results


def ensure_valid_configuration():
    """설정 검증 및 에러 시 시스템 종료"""
    
    try:
        is_valid, results = validate_configuration()
        
        if not is_valid:
            error_messages = [
                f"{r.field}: {r.message}" 
                for r in results 
                if r.severity == ValidationSeverity.ERROR
            ]
            
            raise ValidationException(
                message="Configuration validation failed",
                details={
                    "errors": error_messages,
                    "total_issues": len(results)
                }
            )
        
        # 경고가 있다면 로그로 출력
        warnings = [r for r in results if r.severity == ValidationSeverity.WARNING]
        if warnings:
            logger.warning(f"Configuration has {len(warnings)} warnings", extra={
                "warnings": [f"{w.field}: {w.message}" for w in warnings]
            })
        
        logger.info("Configuration validation passed successfully")
        
    except ValidationException:
        raise
    except Exception as e:
        raise ValidationException(
            message=f"Configuration validation error: {str(e)}",
            details={"original_error": str(e)}
        )


# 시스템 시작 시 자동 검증을 위한 함수
def auto_validate_on_import():
    """import 시점에서 자동으로 기본 검증 수행"""
    
    try:
        # 중요한 설정만 간단히 검증
        if not settings.secret_key or settings.secret_key == "your-super-secret-key-change-in-production":
            logger.warning("Using default secret key - change for production!")
        
        if not settings.database_url:
            logger.error("Database URL not configured")
            
    except Exception as e:
        logger.warning(f"Auto-validation failed: {e}")


# 모듈 import 시 자동 검증
auto_validate_on_import()