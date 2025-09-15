from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any, Optional
import time
from contextlib import asynccontextmanager

from ..core.config import settings
from ..core.logging_config import setup_logging, get_logger, performance_logger
from ..core.config_validator import ensure_valid_configuration, validate_configuration
from ..core.exceptions import (
    FragranceAIException, SystemException, ModelException,
    ValidationException, ErrorCode, safe_execute
)
from ..core.vector_store import VectorStore
from ..models.embedding import KoreanFragranceEmbedding
from ..models.generator import FragranceRecipeGenerator
from ..database.connection import initialize_database, shutdown_database
from .routes import search, generation, training, admin, monitoring
from .schemas import *
from .middleware import LoggingMiddleware, RateLimitMiddleware
from .dependencies import get_current_user, verify_api_key
from .error_handlers import setup_error_handlers, model_circuit_breaker

# 로깅 시스템 초기화
setup_logging(
    log_level=settings.log_level,
    enable_json=not settings.debug,
    enable_console=True,
    enable_file=True
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # Startup
    logger.info("Starting Fragrance AI API...", extra={
        "version": settings.app_version,
        "debug_mode": settings.debug,
        "log_level": settings.log_level
    })
    
    startup_time = time.time()
    
    try:
        # 1. 설정 검증
        logger.info("Validating configuration...")
        ensure_valid_configuration()
        logger.info("Configuration validation passed")
        
        # 2. 상세 설정 검증 및 보고서 생성
        is_valid, validation_results = validate_configuration()
        if validation_results:
            warnings = [r for r in validation_results if r.severity.value == "warning"]
            if warnings:
                logger.warning(f"Configuration has {len(warnings)} warnings", extra={
                    "validation_warnings": [f"{w.field}: {w.message}" for w in warnings[:5]]  # 처음 5개만
                })
        
        # 3. 데이터베이스 초기화
        logger.info("Initializing database connection...")
        initialize_database()
        logger.info("Database initialized successfully")
        
        # 4. AI 모델 초기화
        def init_vector_store():
            return VectorStore()
            
        def init_embedding_model():
            return KoreanFragranceEmbedding()
            
        def init_generator():
            return FragranceRecipeGenerator()
        
        app.state.vector_store = model_circuit_breaker.call(init_vector_store)
        app.state.embedding_model = model_circuit_breaker.call(init_embedding_model)
        app.state.generator = model_circuit_breaker.call(init_generator)
        
        initialization_time = time.time() - startup_time
        
        logger.info("AI models initialized successfully", extra={
            "initialization_time": initialization_time,
            "models_loaded": ["vector_store", "embedding_model", "generator"]
        })
        
        performance_logger.log_execution_time(
            operation="startup_initialization",
            execution_time=initialization_time,
            success=True,
            extra_data={"models_count": 3}
        )
        
    except Exception as e:
        logger.error("Failed to initialize AI models", extra={
            "error": str(e),
            "error_type": type(e).__name__,
            "initialization_time": time.time() - startup_time
        })
        
        # 시스템 예외로 래핑
        raise SystemException(
            message=f"Failed to initialize AI models: {str(e)}",
            error_code=ErrorCode.MODEL_LOADING_ERROR,
            cause=e
        )
    
    yield
    
    # Shutdown
    shutdown_time = time.time()
    logger.info("Shutting down Fragrance AI API...")
    
    try:
        # Cleanup AI models
        if hasattr(app.state, 'vector_store'):
            del app.state.vector_store
        if hasattr(app.state, 'embedding_model'):
            del app.state.embedding_model
        if hasattr(app.state, 'generator'):
            del app.state.generator
        
        # Shutdown database
        logger.info("Shutting down database connection...")
        shutdown_database()
        logger.info("Database shutdown completed")
            
        logger.info("Shutdown completed successfully", extra={
            "shutdown_time": time.time() - shutdown_time
        })
        
    except Exception as e:
        logger.error("Error during shutdown", extra={
            "error": str(e),
            "error_type": type(e).__name__
        })


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Commercial-grade AI system for fragrance recipe generation and semantic search",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

# 에러 핸들러 설정
setup_error_handlers(app)

# Include routers
app.include_router(search.router, prefix=f"{settings.api_prefix}/search", tags=["Search"])
app.include_router(generation.router, prefix=f"{settings.api_prefix}/generate", tags=["Generation"])
app.include_router(training.router, prefix=f"{settings.api_prefix}/training", tags=["Training"])
app.include_router(admin.router, prefix=f"{settings.api_prefix}/admin", tags=["Admin"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])


@app.get("/")
async def root():
    """API 루트 엔드포인트"""
    return {
        "message": "Welcome to Fragrance AI",
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "docs": "/api/docs",
            "search": f"{settings.api_prefix}/search",
            "generate": f"{settings.api_prefix}/generate",
            "training": f"{settings.api_prefix}/training",
            "admin": f"{settings.api_prefix}/admin"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        # Check AI models
        models_status = {
            "vector_store": hasattr(app.state, 'vector_store') and app.state.vector_store is not None,
            "embedding_model": hasattr(app.state, 'embedding_model') and app.state.embedding_model is not None,
            "generator": hasattr(app.state, 'generator') and app.state.generator is not None
        }
        
        all_models_ready = all(models_status.values())
        
        return {
            "status": "healthy" if all_models_ready else "unhealthy",
            "timestamp": time.time(),
            "models": models_status,
            "version": settings.app_version
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@app.get("/metrics")
async def get_metrics(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """시스템 메트릭스 (인증 필요)"""
    try:
        # Verify authorization
        verify_api_key(credentials.credentials)
        
        # Get vector store stats
        vector_stats = {}
        if hasattr(app.state, 'vector_store'):
            for collection_name in ["fragrance_notes", "recipes", "mood_descriptions"]:
                stats = app.state.vector_store.get_collection_stats(collection_name)
                vector_stats[collection_name] = stats
        
        return {
            "timestamp": time.time(),
            "vector_store": vector_stats,
            "system": {
                "version": settings.app_version,
                "debug_mode": settings.debug
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/semantic-search")
async def semantic_search(
    request: SemanticSearchRequest,
    background_tasks: BackgroundTasks
) -> SemanticSearchResponse:
    """의미 기반 검색"""
    try:
        start_time = time.time()
        
        if not hasattr(app.state, 'vector_store'):
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        # Perform search based on search type
        if request.search_type == "single_collection":
            results = app.state.vector_store.semantic_search(
                collection_name=request.collection_name or "fragrance_notes",
                query=request.query,
                top_k=request.top_k,
                filter_criteria=request.filters
            )
        elif request.search_type == "hybrid":
            collections = request.collections or ["fragrance_notes", "recipes", "mood_descriptions"]
            results = app.state.vector_store.hybrid_search(
                query=request.query,
                collections=collections,
                weights=request.collection_weights,
                top_k=request.top_k
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid search type")
        
        # Log search for analytics (background task)
        background_tasks.add_task(
            log_search_analytics,
            query=request.query,
            search_type=request.search_type,
            results_count=len(results),
            response_time=time.time() - start_time
        )
        
        return SemanticSearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            search_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/generate-recipe")
async def generate_recipe(
    request: RecipeGenerationRequest,
    background_tasks: BackgroundTasks
) -> RecipeGenerationResponse:
    """향수 레시피 생성"""
    try:
        start_time = time.time()
        
        if not hasattr(app.state, 'generator'):
            raise HTTPException(status_code=503, detail="Recipe generator not initialized")
        
        # Generate recipe
        recipe = app.state.generator.generate_recipe(
            prompt=request.prompt,
            recipe_type=request.recipe_type,
            include_story=request.include_story,
            mood=request.mood,
            season=request.season,
            notes_preference=request.notes_preference
        )
        
        # Evaluate recipe quality
        quality_scores = app.state.generator.evaluate_recipe_quality(recipe)
        
        # Log generation for analytics (background task)
        background_tasks.add_task(
            log_generation_analytics,
            prompt=request.prompt,
            recipe_type=request.recipe_type,
            quality_score=quality_scores.get("overall", 0.0),
            response_time=time.time() - start_time
        )
        
        return RecipeGenerationResponse(
            recipe=recipe,
            quality_scores=quality_scores,
            generation_time=time.time() - start_time,
            prompt=request.prompt
        )
        
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/batch-generate")
async def batch_generate_recipes(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks
) -> BatchGenerationResponse:
    """배치 레시피 생성"""
    try:
        start_time = time.time()
        
        if not hasattr(app.state, 'generator'):
            raise HTTPException(status_code=503, detail="Recipe generator not initialized")
        
        if len(request.prompts) > 20:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size too large (max 20)")
        
        # Generate recipes
        recipes = app.state.generator.batch_generate_recipes(
            prompts=request.prompts,
            batch_size=request.batch_size
        )
        
        # Calculate average quality
        total_quality = 0.0
        for recipe in recipes:
            quality = app.state.generator.evaluate_recipe_quality(recipe)
            total_quality += quality.get("overall", 0.0)
        
        avg_quality = total_quality / len(recipes) if recipes else 0.0
        
        # Log batch generation (background task)
        background_tasks.add_task(
            log_batch_generation_analytics,
            batch_size=len(request.prompts),
            avg_quality=avg_quality,
            response_time=time.time() - start_time
        )
        
        return BatchGenerationResponse(
            recipes=recipes,
            total_recipes=len(recipes),
            average_quality=avg_quality,
            generation_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task functions
async def log_search_analytics(query: str, search_type: str, results_count: int, response_time: float):
    """검색 분석 로깅"""
    try:
        analytics_data = {
            "type": "search",
            "query": query,
            "search_type": search_type,
            "results_count": results_count,
            "response_time": response_time,
            "timestamp": time.time()
        }
        # Log to analytics service or database
        logger.info(f"Search analytics: {analytics_data}")
    except Exception as e:
        logger.error(f"Failed to log search analytics: {e}")


async def log_generation_analytics(prompt: str, recipe_type: str, quality_score: float, response_time: float):
    """생성 분석 로깅"""
    try:
        analytics_data = {
            "type": "generation",
            "prompt_length": len(prompt),
            "recipe_type": recipe_type,
            "quality_score": quality_score,
            "response_time": response_time,
            "timestamp": time.time()
        }
        logger.info(f"Generation analytics: {analytics_data}")
    except Exception as e:
        logger.error(f"Failed to log generation analytics: {e}")


async def log_batch_generation_analytics(batch_size: int, avg_quality: float, response_time: float):
    """배치 생성 분석 로깅"""
    try:
        analytics_data = {
            "type": "batch_generation",
            "batch_size": batch_size,
            "avg_quality": avg_quality,
            "response_time": response_time,
            "timestamp": time.time()
        }
        logger.info(f"Batch generation analytics: {analytics_data}")
    except Exception as e:
        logger.error(f"Failed to log batch generation analytics: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "fragrance_ai.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.max_workers
    )