"""
업그레이드된 Fragrance AI API 메인 파일
- Pydantic v2
- 고급 RAG 시스템
- 성능 최적화
- 분산 캐싱
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Dict, Any, Optional, Tuple
import time
import asyncio
from contextlib import asynccontextmanager

from ..core.config import settings
# from ..core.production_logging import get_logger, LogCategory
# from ..core.logging_middleware import RequestLoggingMiddleware, SecurityLoggingMiddleware
import logging
logger = logging.getLogger(__name__)
from ..core.exceptions import FragranceAIException, SystemException, ErrorCode
# from ..core.exceptions_unified import (
#     FragranceAIException as UnifiedException,
#     APIException,
#     ModelException,
#     global_error_handler,
#     handle_exceptions_async
# )
from ..core.advanced_caching import FragranceCacheManager, CachePolicy
from ..core.performance_optimizer import global_performance_optimizer
from ..core.error_handling import (
    global_error_handler,
    error_context,
    ErrorCategory,
    ErrorSeverity,
    ErrorContext
)
from ..models.embedding import AdvancedKoreanFragranceEmbedding
from ..models.rag_system import FragranceRAGSystem, RAGMode
from ..database.connection import initialize_database, shutdown_database
from .schemas import (
    SemanticSearchRequest,
    SemanticSearchResponse,
    SearchResult,
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    SystemStatus,
    ErrorResponse
)
from .middleware import LoggingMiddleware, RateLimitMiddleware
from ..core.security_middleware import SecurityMiddleware, validate_input
from ..core.real_monitoring import get_monitoring_dashboard
from .dependencies import get_current_user, verify_api_key, require_permission
from .auth import get_current_user as auth_get_current_user, require_access_level
from .user_auth import router as user_auth_router
from .external_services import router as external_services_router
from .error_handlers import setup_error_handlers

# 프로덕션 로깅 초기화
logger = get_logger("fragrance_ai_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리 (업그레이드된 버전)"""
    startup_time = time.time()
    
    async with error_context(
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.CRITICAL,
        attempt_recovery=False
    ):
        logger.info("🚀 Starting Advanced Fragrance AI System...")

        # 1. 데이터베이스 초기화
        logger.info("📊 Initializing database...")
        initialize_database()

        # 2. 캐시 시스템 초기화
        logger.info("🗄️  Initializing advanced caching system...")
        app.state.cache_manager = FragranceCacheManager(
            max_size=50000,
            policy=CachePolicy.ADAPTIVE,
            redis_url=getattr(settings, 'redis_url', None),
            enable_metrics=True,
            warmup_enabled=True
        )

        # 3. 성능 최적화 시스템 초기화
        logger.info("⚡ Starting performance optimization system...")
        await global_performance_optimizer.start()

        # 4. AI 모델 초기화
        logger.info("🤖 Loading advanced AI models...")

        # 고급 임베딩 모델
        app.state.embedding_model = AdvancedKoreanFragranceEmbedding(
            use_adapter=True,
            enable_multi_aspect=True,
            cache_size=10000
        )

        # RAG 시스템
        app.state.rag_system = FragranceRAGSystem(
            embedding_model=app.state.embedding_model,
            rag_mode=RAGMode.ADAPTIVE_RAG,
            max_retrieved_docs=15
        )

        # 5. 배치 프로세서 설정
        logger.info("📦 Setting up batch processors...")

        # 임베딩 배치 프로세서
        embedding_processor = global_performance_optimizer.create_batch_processor(
            name="embedding_batch",
            batch_size=64,
            max_wait_time=0.05,
            processor_func=batch_embedding_processor
        )

        # 검색 배치 프로세서
        search_processor = global_performance_optimizer.create_batch_processor(
            name="search_batch",
            batch_size=32,
            max_wait_time=0.1,
            processor_func=batch_search_processor
        )

        app.state.embedding_batch_processor = embedding_processor
        app.state.search_batch_processor = search_processor

        # 6. 캐시 워밍업
        logger.info("🔥 Warming up caches...")
        await app.state.cache_manager.warmup(generate_warmup_data)

        # 7. 헬스체크 준비
        app.state.startup_time = startup_time
        app.state.ready = True

        total_startup_time = time.time() - startup_time
        logger.info(f"✅ Advanced Fragrance AI System ready in {total_startup_time:.2f}s")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Advanced Fragrance AI System...")
    
    try:
        # Performance optimizer 중지
        await global_performance_optimizer.stop()
        
        # 데이터베이스 종료
        shutdown_database()
        
        logger.info("✅ Graceful shutdown completed")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# FastAPI 앱 생성
app = FastAPI(
    title="Advanced Fragrance AI",
    version="2.0.0",
    description="""
    🌸 **고급 향수 AI 시스템**
    
    최신 AI 기술을 적용한 향수 추천 및 생성 시스템
    - 🤖 Advanced RAG (Retrieval-Augmented Generation)
    - ⚡ 고성능 비동기 처리
    - 🗄️ 지능형 분산 캐싱
    - 📊 실시간 성능 모니터링
    - 🎯 한국어 특화 임베딩
    """,
    docs_url="/api/v2/docs",
    redoc_url="/api/v2/redoc",
    openapi_url="/api/v2/openapi.json",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# CORS - 프로덕션 보안 설정
allowed_origins = [
    "http://localhost:3000",  # 개발용
    "http://127.0.0.1:3000",  # 개발용
]

# 프로덕션 환경에서는 환경변수에서 허용 도메인 가져오기
if hasattr(settings, 'cors_origins') and settings.cors_origins:
    if isinstance(settings.cors_origins, str):
        # 문자열인 경우 JSON 파싱
        import json
        try:
            allowed_origins = json.loads(settings.cors_origins)
        except json.JSONDecodeError:
            allowed_origins = [settings.cors_origins]
    else:
        allowed_origins = settings.cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=86400,  # 24시간 캐시
)

# 커스텀 미들웨어 (순서 중요 - 로깅이 가장 바깥쪽)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityLoggingMiddleware)
app.add_middleware(SecurityMiddleware,
                  rate_limit_per_minute=100,  # 분당 100 요청
                  max_request_size=2*1024*1024)  # 2MB
app.add_middleware(RateLimitMiddleware)

# 에러 핸들러
setup_error_handlers(app)

# 통합 예외 핸들러 추가
@app.exception_handler(UnifiedException)
async def fragrance_ai_exception_handler(request: Request, exc: UnifiedException):
    """FragranceAI 예외 핸들러"""
    error_info = global_error_handler.handle_error(exc)

    return JSONResponse(
        status_code=getattr(exc, 'status_code', 500),
        content={
            "error": True,
            "error_code": exc.error_code.value,
            "message": exc.message,
            "details": exc.details,
            "timestamp": exc.timestamp.isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 핸들러"""
    # 예상치 못한 예외를 FragranceAI 예외로 변환
    fragrance_exc = UnifiedException(
        message=f"Unexpected server error: {str(exc)}",
        cause=exc
    )

    error_info = global_error_handler.handle_error(fragrance_exc)

    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "error_code": fragrance_exc.error_code.value,
            "message": "Internal server error",
            "timestamp": fragrance_exc.timestamp.isoformat()
        }
    )

# 라우터 등록
app.include_router(user_auth_router, prefix="/api/v2")
app.include_router(external_services_router, prefix="/api/v2")

# 새로운 인증 및 레시피 라우터 추가
from .routes.auth import router as auth_router
from .routes.public_recipes import router as public_recipes_router
from .routes.generation import router as generation_router
from .routes.agentic import router as agentic_router
from .routes.customer_service import router as customer_service_router
from .routes.admin_auth import router as admin_auth_router  # 관리자 인증 라우터
from .routes.generation_with_ai import router as ai_generation_router  # AI 생성 라우터 추가

app.include_router(auth_router, prefix="/api/v2")
app.include_router(public_recipes_router, prefix="/api/v2")
app.include_router(generation_router, prefix="/api/v2/admin", tags=["관리자 전용"])
app.include_router(agentic_router, prefix="/api/v2", tags=["AI Orchestrator"])  # New agentic system
app.include_router(customer_service_router, tags=["Customer Service"])  # Customer service routes
app.include_router(admin_auth_router, tags=["Admin Authentication"])  # 관리자 세션 기반 인증
app.include_router(ai_generation_router, prefix="/api/v1/ai", tags=["AI Generation"])  # DEAP/RLHF AI 라우터


# 배치 처리 함수들
async def batch_embedding_processor(items: List[Tuple]) -> List[np.ndarray]:
    """임베딩 배치 처리"""
    texts = [item[0] for item in items]
    embedding_model = app.state.embedding_model
    
    result = await embedding_model.encode_async(texts, enable_caching=True)
    return [result.embeddings[i] for i in range(len(texts))]


async def batch_search_processor(items: List[Tuple]) -> List[List[SearchResult]]:
    """검색 배치 처리"""
    results = []
    for query, search_params in items:
        # 실제 검색 로직 구현
        search_results = await perform_single_search(query, search_params)
        results.append(search_results)
    return results


async def perform_single_search(query: str, params: Dict[str, Any]) -> List[SearchResult]:
    """단일 검색 수행"""
    # 캐시 확인
    cached_result = await app.state.cache_manager.get_cached_embedding(query)
    
    # RAG 시스템을 사용한 검색
    rag_result = await app.state.rag_system.generate_with_rag(
        query=query,
        temperature=0.7,
        enable_reasoning=True
    )
    
    # 결과를 SearchResult 형태로 변환
    results = []
    for i, doc in enumerate(rag_result.source_documents[:10]):
        results.append(SearchResult(
            id=f"doc_{i}",
            document=doc,
            metadata={"source": "rag_system"},
            distance=0.0,
            similarity=rag_result.confidence_score,
            collection="knowledge_base",
            rank=i + 1
        ))
    
    return results


def generate_warmup_data() -> Dict[str, Any]:
    """캐시 워밍업 데이터 생성"""
    warmup_data = {}
    
    # 인기 검색어들
    popular_queries = [
        "시트러스 향수", "플로럴 향수", "우디 향수", "여름 향수",
        "겨울 향수", "로맨틱한 향수", "프레시한 향수", "오리엔탈 향수"
    ]
    
    # 임베딩 미리 계산
    for query in popular_queries:
        cache_key = f"popular_query:{query}"
        warmup_data[cache_key] = {"query": query, "popularity": 1.0}
    
    return warmup_data


# API 엔드포인트들

@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "🌸 Advanced Fragrance AI v2.0",
        "version": "2.0.0",
        "features": [
            "🤖 Advanced RAG System",
            "⚡ High-Performance Async Processing", 
            "🗄️ Intelligent Distributed Caching",
            "📊 Real-time Performance Monitoring",
            "🎯 Korean-Specialized Embeddings"
        ],
        "endpoints": {
            "docs": "/api/v2/docs",
            "semantic_search": "/api/v2/semantic-search",
            "rag_chat": "/api/v2/rag-chat",
            "auth_login": "/api/v2/auth/login",
            "public_recipes": "/api/v2/public/recipes/generate",
            "admin_recipes": "/api/v2/admin/recipe",
            "performance": "/api/v2/performance"
        }
    }


@app.get("/health")
async def advanced_health_check(request: Request):
    """고급 헬스체크"""
    # 에러 컨텍스트 생성
    error_ctx = ErrorContext(
        endpoint="/health",
        method="GET",
        user_agent=request.headers.get("user-agent"),
        ip_address=getattr(request.client, 'host', None)
    )

    async with error_context(
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.HIGH,
        context=error_ctx,
        attempt_recovery=False,
        raise_on_error=False
    ):
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - getattr(app.state, 'startup_time', time.time()),
            "version": "2.0.0"
        }

        # 각 컴포넌트 상태 확인
        components = {
            "embedding_model": hasattr(app.state, 'embedding_model'),
            "rag_system": hasattr(app.state, 'rag_system'),
            "cache_manager": hasattr(app.state, 'cache_manager'),
            "performance_optimizer": global_performance_optimizer.running,
            "batch_processors": len(global_performance_optimizer.batch_processors) > 0
        }

        health_status["components"] = components
        health_status["all_systems_ready"] = all(components.values())

        # 성능 메트릭 추가
        if hasattr(app.state, 'cache_manager'):
            cache_stats = app.state.cache_manager.get_stats()
            health_status["cache_stats"] = {
                "hit_rate": cache_stats.get("hit_rate", 0),
                "cache_size": cache_stats.get("cache_size", 0)
            }

        # 에러 통계 추가
        error_stats = global_error_handler.get_error_statistics(hours=1)
        health_status["error_stats"] = {
            "total_errors_last_hour": error_stats["total_errors"],
            "recovery_success_rate": error_stats["recovery_success_rate"],
            "critical_errors": error_stats["by_severity"].get("critical", 0)
        }

        if not health_status["all_systems_ready"]:
            return JSONResponse(status_code=503, content=health_status)

        return health_status


@app.get("/api/v2/monitoring")
async def get_monitoring_dashboard():
    """실시간 모니터링 대시보드 데이터"""
    try:
        dashboard = get_monitoring_dashboard()
        dashboard_data = dashboard.get_dashboard_data()

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": dashboard_data,
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Failed to get monitoring data: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve monitoring data"
        )


@app.get("/api/v2/monitoring/alerts")
async def get_recent_alerts():
    """최근 알림 조회"""
    try:
        dashboard = get_monitoring_dashboard()
        alerts = dashboard.alert_manager.get_recent_alerts(limit=20)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "alerts": alerts,
                "total_count": len(alerts),
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve alerts"
        )


@app.post("/api/v2/semantic-search")
@validate_input({
    "query": {"type": "text", "max_length": 500},
    "top_k": {"type": "numeric", "min": 1, "max": 100},
    "min_similarity": {"type": "numeric", "min": 0.0, "max": 1.0}
})
async def advanced_semantic_search(
    request: SemanticSearchRequest,
    background_tasks: BackgroundTasks,
    req: Request
) -> SemanticSearchResponse:
    """고급 시맨틱 검색"""
    start_time = time.time()

    # 에러 컨텍스트 생성
    error_ctx = ErrorContext(
        endpoint="/api/v2/semantic-search",
        method="POST",
        user_agent=req.headers.get("user-agent"),
        ip_address=getattr(req.client, 'host', None),
        additional_data={"query_length": len(request.query), "top_k": request.top_k}
    )

    async with error_context(
        category=ErrorCategory.MODEL_INFERENCE,
        severity=ErrorSeverity.MEDIUM,
        context=error_ctx,
        attempt_recovery=True
    ):
        # 배치 처리 사용 여부 결정
        use_batch = len(request.query) > 100 or request.top_k > 20

        if use_batch and hasattr(app.state, 'search_batch_processor'):
            # 배치 처리
            search_params = {
                "search_type": request.search_type,
                "collections": request.collections,
                "top_k": request.top_k,
                "min_similarity": request.min_similarity
            }

            results = await app.state.search_batch_processor.add_item(
                (request.query, search_params)
            )
        else:
            # 단일 처리
            results = await perform_single_search(request.query, request.model_dump())

        # 재순위화 (활성화된 경우)
        if request.enable_reranking and len(results) > 5:
            results = await rerank_results(results, request.query)

        # 캐시에 결과 저장
        if request.use_cache:
            background_tasks.add_task(
                app.state.cache_manager.cache_search_result,
                request.query,
                [result.model_dump_optimized() for result in results]
            )

        search_time = time.time() - start_time

        return SemanticSearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            search_time=search_time,
            cached=False,
            reranked=request.enable_reranking
        )


@app.post("/api/v2/rag-chat")
async def rag_chat(
    request: Request,
    query: str,
    context: Optional[str] = None,
    temperature: float = 0.7,
    enable_reasoning: bool = True,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """RAG 기반 채팅"""
    start_time = time.time()

    # 에러 컨텍스트 생성
    error_ctx = ErrorContext(
        endpoint="/api/v2/rag-chat",
        method="POST",
        user_id=current_user.get("user_id") if current_user else None,
        user_agent=request.headers.get("user-agent"),
        ip_address=getattr(request.client, 'host', None),
        additional_data={
            "query_length": len(query),
            "temperature": temperature,
            "enable_reasoning": enable_reasoning
        }
    )

    async with error_context(
        category=ErrorCategory.MODEL_INFERENCE,
        severity=ErrorSeverity.MEDIUM,
        context=error_ctx,
        attempt_recovery=True
    ):
        # RAG 시스템으로 생성 (성능 로깅 포함)
        with logger.log_performance("rag_generation"):
            result = await app.state.rag_system.generate_with_rag(
                query=query,
                context=context,
                temperature=temperature,
                enable_reasoning=enable_reasoning
            )

        # AI 모델 성능 로그
        logger.info(
            f"RAG generation completed for query length {len(query)}",
            category=LogCategory.AI_MODEL,
            query_length=len(query),
            confidence_score=result.confidence_score,
            documents_retrieved=len(result.source_documents),
            temperature=temperature,
            enable_reasoning=enable_reasoning
        )

        response_time = time.time() - start_time

        return {
            "response": result.generated_text,
            "confidence_score": result.confidence_score,
            "source_documents": result.source_documents[:3],  # Top 3만 반환
            "reasoning_steps": result.reasoning_steps,
            "retrieval_info": {
                "retrieval_time": result.retrieval_context.retrieval_time,
                "documents_retrieved": len(result.retrieval_context.retrieved_documents),
                "avg_similarity": float(np.mean(result.retrieval_context.similarity_scores)) if result.retrieval_context.similarity_scores else 0.0
            },
            "response_time": response_time,
            "timestamp": time.time()
        }


@app.get("/api/v2/performance")
async def get_performance_metrics(
    request: Request,
    current_user: Dict[str, Any] = Depends(require_permission("system.metrics"))
):
    """성능 메트릭 조회"""
    error_ctx = ErrorContext(
        endpoint="/api/v2/performance",
        method="GET",
        user_id=current_user.get("user_id") if current_user else None,
        user_agent=request.headers.get("user-agent"),
        ip_address=getattr(request.client, 'host', None)
    )

    async with error_context(
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.LOW,
        context=error_ctx,
        attempt_recovery=True
    ):
        # 성능 최적화 보고서
        perf_report = global_performance_optimizer.get_performance_report()

        # 캐시 통계
        cache_stats = {}
        if hasattr(app.state, 'cache_manager'):
            cache_stats = app.state.cache_manager.get_stats()
            cache_stats["hot_keys"] = app.state.cache_manager.get_hot_keys(10)

        # AI 모델 상태
        model_stats = {}
        if hasattr(app.state, 'rag_system'):
            model_stats["knowledge_base"] = app.state.rag_system.get_knowledge_base_stats()

        return {
            "timestamp": time.time(),
            "performance": perf_report,
            "cache": cache_stats,
            "models": model_stats,
            "system_info": {
                "version": "2.0.0",
                "uptime": time.time() - getattr(app.state, 'startup_time', time.time())
            }
        }


@app.get("/api/v2/error-statistics")
async def get_error_statistics(
    request: Request,
    hours: int = 24,
    current_user: Dict[str, Any] = Depends(require_permission("system.admin"))
):
    """에러 통계 조회"""
    error_ctx = ErrorContext(
        endpoint="/api/v2/error-statistics",
        method="GET",
        user_id=current_user.get("user_id") if current_user else None,
        user_agent=request.headers.get("user-agent"),
        ip_address=getattr(request.client, 'host', None),
        additional_data={"hours": hours}
    )

    async with error_context(
        category=ErrorCategory.SYSTEM,
        severity=ErrorSeverity.LOW,
        context=error_ctx,
        attempt_recovery=True
    ):
        # 에러 통계 조회
        error_stats = global_error_handler.get_error_statistics(hours=hours)

        # 최근 에러 샘플 (민감한 정보 제외)
        recent_errors = []
        for record in global_error_handler.error_records[-10:]:
            recent_errors.append({
                "error_id": record.error_id,
                "category": record.category.value,
                "severity": record.severity.value,
                "timestamp": record.context.timestamp.isoformat(),
                "endpoint": record.context.endpoint,
                "is_handled": record.is_handled,
                "recovery_successful": record.recovery_successful,
                "user_message": record.user_message
            })

        return {
            "timestamp": time.time(),
            "period_hours": hours,
            "statistics": error_stats,
            "recent_errors": recent_errors,
            "system_health": {
                "error_rate": error_stats["total_errors"] / max(hours, 1),
                "recovery_rate": error_stats["recovery_success_rate"],
                "critical_issues": error_stats["by_severity"].get("critical", 0) > 0
            }
        }


async def rerank_results(results: List[SearchResult], query: str) -> List[SearchResult]:
    """검색 결과 재순위화"""
    # 간단한 재순위화 로직 (실제로는 더 정교한 모델 사용)
    query_words = set(query.lower().split())
    
    for result in results:
        doc_words = set(result.document.lower().split())
        keyword_overlap = len(query_words.intersection(doc_words))
        
        # 재순위화 점수 계산
        rerank_score = (
            result.effective_score * 0.7 +  # 기존 점수
            (keyword_overlap / len(query_words)) * 0.3  # 키워드 겹침
        )
        result.rerank_score = rerank_score
    
    # 재순위화 점수로 정렬
    results.sort(key=lambda x: x.rerank_score or x.effective_score, reverse=True)
    
    # 순위 업데이트
    for i, result in enumerate(results):
        result.rank = i + 1
    
    return results


if __name__ == "__main__":
    uvicorn.run(
        "fragrance_ai.api.main:app",
        host=getattr(settings, 'api_host', '0.0.0.0'),
        port=getattr(settings, 'api_port', 8000),
        reload=getattr(settings, 'debug', False),
        workers=1 if getattr(settings, 'debug', False) else getattr(settings, 'max_workers', 4)
    )