"""
FastAPI Application Startup Handler
애플리케이션 시작 시 모델 미리 로드
"""

import logging
import asyncio
from typing import List
from contextlib import asynccontextmanager

from ..core.model_manager import get_model_manager
from ..core.config_manager import get_config
from ..core.circuit_breaker import get_circuit_breaker

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """
    FastAPI Lifespan 이벤트 핸들러
    
    애플리케이션 시작/종료 시 실행
    """
    # Startup
    logger.info("Starting up Fragrance AI application...")
    
    try:
        # 설정 로드
        config = get_config()
        logger.info(f"Environment: {config.env}")
        
        # Circuit Breaker 초기화
        circuit_breaker = get_circuit_breaker()
        logger.info("Circuit breaker initialized")
        
        # 모델 매니저 초기화
        model_manager = get_model_manager()
        
        # 필수 모델 미리 로드
        essential_models = [
            "embedding_model",  # 임베딩 모델
            "scientific_validator",  # 검증 모델
            "ollama_client",  # Ollama 클라이언트
        ]
        
        # 환경별 추가 모델
        if config.env == "production":
            essential_models.extend([
                "rag_system",  # RAG 시스템
                "master_perfumer",  # 마스터 조향사 AI
            ])
        
        logger.info(f"Preloading {len(essential_models)} essential models...")
        
        # 비동기 모델 로드
        await asyncio.gather(*[
            asyncio.to_thread(model_manager.get_model, model_name)
            for model_name in essential_models
        ])
        
        logger.info("All essential models loaded successfully")
        
        # 메모리 사용량 로깅
        memory_info = model_manager.get_memory_usage()
        logger.info(f"Memory usage: {memory_info}")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # 애플리케이션 실행
    yield
    
    # Shutdown
    logger.info("Shutting down Fragrance AI application...")
    
    try:
        # 모델 정리
        model_manager = get_model_manager()
        model_manager.cleanup()
        
        # Circuit Breaker 상태 저장
        circuit_breaker = get_circuit_breaker()
        status = circuit_breaker.get_status()
        logger.info(f"Final circuit breaker status: {status}")
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
    
    logger.info("Application shutdown complete")


def initialize_app(app):
    """
    FastAPI 앱 초기화
    
    Args:
        app: FastAPI 인스턴스
    """
    # Lifespan 이벤트 등록
    app.router.lifespan_context = lifespan
    
    # 미들웨어 추가
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    
    config = get_config()
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # GZip 압축
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    logger.info("Application initialized")
