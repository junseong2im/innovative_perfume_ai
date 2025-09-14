import logging
import logging.config
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
import structlog
import json
from pathlib import Path

from .config import settings


class JSONFormatter(logging.Formatter):
    """JSON 형식의 로그 포매터"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        
        # 추가 컨텍스트 정보
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'execution_time'):
            log_entry['execution_time'] = record.execution_time
            
        # 예외 정보 포함
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


class ContextFilter(logging.Filter):
    """로그에 컨텍스트 정보를 추가하는 필터"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # 기본 컨텍스트 정보 추가
        record.service = "fragrance_ai"
        record.version = settings.app_version
        record.environment = "development" if settings.debug else "production"
        return True


def setup_logging(
    log_level: str = None,
    log_dir: str = "logs",
    enable_json: bool = True,
    enable_file: bool = True,
    enable_console: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 10
) -> None:
    """로깅 시스템 설정"""
    
    if log_level is None:
        log_level = settings.log_level
        
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # 핸들러 설정
    handlers = {}
    
    # 콘솔 핸들러
    if enable_console:
        handlers['console'] = {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'json' if enable_json else 'standard',
            'stream': 'ext://sys.stdout',
            'filters': ['context_filter']
        }
    
    # 파일 핸들러 (일반 로그)
    if enable_file:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json' if enable_json else 'standard',
            'filename': str(log_path / 'app.log'),
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'filters': ['context_filter']
        }
        
        # 에러 로그 파일 핸들러
        handlers['error_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json' if enable_json else 'standard',
            'filename': str(log_path / 'error.log'),
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'filters': ['context_filter']
        }
        
        # 성능 로그 파일 핸들러
        handlers['performance'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': str(log_path / 'performance.log'),
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'filters': ['context_filter']
        }
    
    # 로깅 설정
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s (%(filename)s:%(lineno)d)'
            },
            'json': {
                '()': JSONFormatter,
            },
        },
        'filters': {
            'context_filter': {
                '()': ContextFilter,
            }
        },
        'handlers': handlers,
        'loggers': {
            '': {  # 루트 로거
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': False
            },
            'fragrance_ai': {
                'handlers': list(handlers.keys()),
                'level': log_level,
                'propagate': False
            },
            'uvicorn': {
                'handlers': ['console'] if enable_console else [],
                'level': 'INFO',
                'propagate': False
            },
            'uvicorn.access': {
                'handlers': ['file'] if enable_file else [],
                'level': 'INFO',
                'propagate': False
            },
            'sqlalchemy.engine': {
                'handlers': ['file'] if enable_file else [],
                'level': 'WARNING' if settings.debug else 'ERROR',
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(config)
    
    # Structlog 설정 (구조화된 로깅용)
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> logging.Logger:
    """로거 인스턴스 반환"""
    return logging.getLogger(name)


def get_structured_logger(name: str):
    """구조화된 로거 인스턴스 반환"""
    return structlog.get_logger(name)


class PerformanceLogger:
    """성능 측정 및 로깅 클래스"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = get_logger(logger_name)
        self.struct_logger = get_structured_logger(logger_name)
    
    def log_execution_time(
        self,
        operation: str,
        execution_time: float,
        success: bool = True,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """실행 시간 로깅"""
        data = {
            "operation": operation,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if extra_data:
            data.update(extra_data)
            
        if success:
            self.struct_logger.info("Operation completed", **data)
        else:
            self.struct_logger.error("Operation failed", **data)
    
    def log_memory_usage(
        self,
        operation: str,
        memory_usage: Dict[str, float],
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """메모리 사용량 로깅"""
        data = {
            "operation": operation,
            "memory_usage": memory_usage,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if extra_data:
            data.update(extra_data)
            
        self.struct_logger.info("Memory usage recorded", **data)
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None,
        user_id: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """API 요청 로깅"""
        data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if request_size:
            data["request_size"] = request_size
        if response_size:
            data["response_size"] = response_size
        if user_id:
            data["user_id"] = user_id
        if extra_data:
            data.update(extra_data)
            
        self.struct_logger.info("API request processed", **data)


# 전역 성능 로거 인스턴스
performance_logger = PerformanceLogger()