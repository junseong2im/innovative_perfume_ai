"""
Centralized Configuration Manager
중앙 집중식 설정 관리 시스템
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "postgresql://user:password@localhost/fragrance_ai"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class RedisConfig:
    """Redis configuration"""
    url: str = "redis://localhost:6379"
    ttl: int = 3600
    max_connections: int = 50


@dataclass
class ModelConfig:
    """Model configuration"""
    embedding_model: str = "jhgan/ko-sbert-nli"
    generation_model: str = "beomi/KoAlpaca-Polyglot-5.8B"
    validation_model_path: str = "./models/validation/harmony_validator.pth"
    rag_mode: str = "hybrid_retrieval"
    device: str = "auto"
    quantization: str = "4bit"


@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8001
    workers: int = 4
    reload: bool = False
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["http://localhost:3000"]


@dataclass
class OllamaConfig:
    """Ollama configuration"""
    base_url: str = "http://localhost:11434"
    models: Dict[str, str] = None
    timeout: int = 60
    
    def __post_init__(self):
        if self.models is None:
            self.models = {
                "orchestrator": "llama3:8b-instruct-q4_K_M",
                "description": "qwen:14b",
                "customer_service": "mistral:7b-instruct-q4_K_M"
            }


class ConfigManager:
    """
    중앙 설정 관리자
    
    - 환경별 설정 관리 (local, dev, production)
    - 환경 변수 오버라이드
    - 설정 검증
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        if self._initialized:
            return
            
        self.env = os.getenv("APP_ENV", "local")
        self.config_dir = Path("configs")
        self.config_file = self.config_dir / f"{self.env}.json"
        
        # 기본 설정
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.models = ModelConfig()
        self.api = APIConfig()
        self.ollama = OllamaConfig()
        
        # 설정 로드
        self._load_config()
        self._override_from_env()
        self._validate_config()
        
        self._initialized = True
        logger.info(f"ConfigManager initialized for environment: {self.env}")
    
    def _load_config(self):
        """설정 파일 로드"""
        if not self.config_file.exists():
            logger.warning(f"Config file not found: {self.config_file}. Using defaults.")
            return
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # Database 설정
            if 'database' in config_data:
                db_config = config_data['database']
                self.database.url = db_config.get('url', self.database.url)
                self.database.pool_size = db_config.get('pool_size', self.database.pool_size)
                
            # Redis 설정
            if 'redis' in config_data:
                redis_config = config_data['redis']
                self.redis.url = redis_config.get('url', self.redis.url)
                self.redis.ttl = redis_config.get('ttl', self.redis.ttl)
                
            # Model 설정
            if 'model_paths' in config_data:
                model_config = config_data['model_paths']
                self.models.embedding_model = model_config.get('embedding', self.models.embedding_model)
                self.models.validation_model_path = model_config.get('validation', self.models.validation_model_path)
                
            # Ollama 설정
            if 'llm_orchestrator' in config_data:
                llm_config = config_data['llm_orchestrator']
                self.ollama.base_url = llm_config.get('api_base', self.ollama.base_url)
                self.ollama.models['orchestrator'] = llm_config.get('model_name_or_path', self.ollama.models['orchestrator'])
                
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
    
    def _override_from_env(self):
        """환경 변수로 오버라이드"""
        # Database
        if os.getenv('DATABASE_URL'):
            self.database.url = os.getenv('DATABASE_URL')
            
        # Redis
        if os.getenv('REDIS_URL'):
            self.redis.url = os.getenv('REDIS_URL')
            
        # API
        if os.getenv('API_HOST'):
            self.api.host = os.getenv('API_HOST')
        if os.getenv('API_PORT'):
            self.api.port = int(os.getenv('API_PORT'))
            
        # Ollama
        if os.getenv('OLLAMA_BASE_URL'):
            self.ollama.base_url = os.getenv('OLLAMA_BASE_URL')
            
        # Model device
        if os.getenv('CUDA_VISIBLE_DEVICES'):
            self.models.device = "cuda"
    
    def _validate_config(self):
        """설정 검증"""
        errors = []
        
        # Database URL 형식 검증
        if not self.database.url.startswith(('postgresql://', 'sqlite://')):
            errors.append("Invalid database URL format")
            
        # Redis URL 형식 검증
        if not self.redis.url.startswith('redis://'):
            errors.append("Invalid Redis URL format")
            
        # 포트 범위 검증
        if not 1 <= self.api.port <= 65535:
            errors.append(f"Invalid port number: {self.api.port}")
            
        # 모델 파일 경로 검증 (프로덕션에서만)
        if self.env == 'production':
            model_path = Path(self.models.validation_model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            # 프로덕션에서만 중단
            if self.env == 'production':
                raise ValueError(f"Configuration validation failed: {errors}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        설정 값 가져오기
        
        Args:
            key: 설정 키 (dot notation 지원)
            default: 기본값
            
        Returns:
            설정 값
        """
        try:
            keys = key.split('.')
            value = self
            
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                elif isinstance(value, dict):
                    value = value.get(k)
                else:
                    return default
                    
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """
        설정 값 설정
        
        Args:
            key: 설정 키
            value: 설정 값
        """
        keys = key.split('.')
        
        if len(keys) == 1:
            setattr(self, key, value)
        elif len(keys) == 2:
            obj = getattr(self, keys[0])
            if hasattr(obj, keys[1]):
                setattr(obj, keys[1], value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        설정을 dictionary로 변환
        
        Returns:
            설정 dictionary
        """
        return {
            "env": self.env,
            "database": asdict(self.database),
            "redis": asdict(self.redis),
            "models": asdict(self.models),
            "api": asdict(self.api),
            "ollama": asdict(self.ollama)
        }
    
    def reload(self):
        """설정 다시 로드"""
        logger.info("Reloading configuration...")
        self._load_config()
        self._override_from_env()
        self._validate_config()
        logger.info("Configuration reloaded")


# 전역 설정 인스턴스
_config = None

def get_config() -> ConfigManager:
    """
    전역 설정 인스턴스 가져오기
    
    Returns:
        ConfigManager 인스턴스
    """
    global _config
    if _config is None:
        _config = ConfigManager()
    return _config


# 편의 함수
def config() -> ConfigManager:
    """Get config shortcut"""
    return get_config()
