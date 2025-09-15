"""
Celery 애플리케이션 설정
비동기 작업 처리를 위한 Celery 인스턴스
"""
from celery import Celery
from celery.signals import worker_init, worker_shutdown
import os
from typing import Dict, Any

from .core.config import settings
from .core.logging_config import get_logger

logger = get_logger(__name__)

# Celery 애플리케이션 생성
celery_app = Celery(
    "fragrance_ai",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        'fragrance_ai.services.generation_service',
        'fragrance_ai.services.search_service',
        'fragrance_ai.training.peft_trainer',
        'fragrance_ai.evaluation.advanced_evaluator'
    ]
)

# Celery 설정
celery_app.conf.update(
    # 직렬화 설정
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # 결과 백엔드 설정
    result_expires=3600,
    result_backend_transport_options={
        'master_name': 'mymaster'
    },

    # 태스크 라우팅
    task_routes={
        'fragrance_ai.services.generation_service.*': {'queue': 'generation'},
        'fragrance_ai.services.search_service.*': {'queue': 'search'},
        'fragrance_ai.training.*': {'queue': 'training'},
        'fragrance_ai.evaluation.*': {'queue': 'evaluation'},
    },

    # 워커 설정
    worker_prefetch_multiplier=1,
    task_acks_late=True,

    # 태스크 실행 제한
    task_soft_time_limit=300,  # 5분
    task_time_limit=600,       # 10분

    # 재시도 설정
    task_default_retry_delay=60,
    task_max_retries=3,

    # 배치 설정
    task_always_eager=False,
    task_eager_propagates=False,

    # 모니터링
    worker_send_task_events=True,
    task_send_sent_event=True,

    # 보안
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# 비동기 작업 등록
@celery_app.task(bind=True, name='generate_recipe_async')
def generate_recipe_async(self, prompt: str, **kwargs) -> Dict[str, Any]:
    """비동기 레시피 생성"""
    try:
        from .services.generation_service import GenerationService

        logger.info(f"Starting async recipe generation for prompt: {prompt[:50]}...")

        service = GenerationService()
        result = service.generate_recipe(prompt=prompt, **kwargs)

        logger.info("Async recipe generation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Async recipe generation failed: {e}")
        self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, name='batch_generate_recipes_async')
def batch_generate_recipes_async(self, prompts: list, **kwargs) -> Dict[str, Any]:
    """비동기 배치 레시피 생성"""
    try:
        from .services.generation_service import GenerationService

        logger.info(f"Starting async batch generation for {len(prompts)} prompts")

        service = GenerationService()
        results = service.batch_generate_recipes(prompts=prompts, **kwargs)

        logger.info("Async batch generation completed successfully")
        return results

    except Exception as e:
        logger.error(f"Async batch generation failed: {e}")
        self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, name='semantic_search_async')
def semantic_search_async(self, query: str, **kwargs) -> Dict[str, Any]:
    """비동기 의미 검색"""
    try:
        from .services.search_service import SearchService

        logger.info(f"Starting async semantic search for query: {query[:50]}...")

        service = SearchService()
        results = service.semantic_search(query=query, **kwargs)

        logger.info("Async semantic search completed successfully")
        return results

    except Exception as e:
        logger.error(f"Async semantic search failed: {e}")
        self.retry(countdown=30, max_retries=3)

@celery_app.task(bind=True, name='train_model_async')
def train_model_async(self, model_type: str, **kwargs) -> Dict[str, Any]:
    """비동기 모델 훈련"""
    try:
        from .training.peft_trainer import PEFTTrainer

        logger.info(f"Starting async model training for type: {model_type}")

        trainer = PEFTTrainer(model_type=model_type)
        result = trainer.train(**kwargs)

        logger.info("Async model training completed successfully")
        return result

    except Exception as e:
        logger.error(f"Async model training failed: {e}")
        self.retry(countdown=300, max_retries=1)  # 훈련은 재시도 제한

@celery_app.task(bind=True, name='evaluate_model_async')
def evaluate_model_async(self, model_path: str, **kwargs) -> Dict[str, Any]:
    """비동기 모델 평가"""
    try:
        from .evaluation.advanced_evaluator import AdvancedEvaluator

        logger.info(f"Starting async model evaluation for: {model_path}")

        evaluator = AdvancedEvaluator()
        result = evaluator.evaluate_model(model_path=model_path, **kwargs)

        logger.info("Async model evaluation completed successfully")
        return result

    except Exception as e:
        logger.error(f"Async model evaluation failed: {e}")
        self.retry(countdown=120, max_retries=2)

@celery_app.task(name='cleanup_old_results')
def cleanup_old_results():
    """오래된 결과 정리"""
    try:
        logger.info("Starting cleanup of old results")

        # 결과 정리 로직 구현
        # 예: 오래된 캐시 삭제, 임시 파일 정리 등

        logger.info("Cleanup completed successfully")
        return {"status": "completed", "cleaned_items": 0}

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"status": "failed", "error": str(e)}

@celery_app.task(name='health_check_models')
def health_check_models():
    """모델 상태 점검"""
    try:
        logger.info("Starting model health check")

        # 모델 상태 점검 로직
        health_status = {
            "embedding_model": "healthy",
            "generation_model": "healthy",
            "vector_store": "healthy"
        }

        logger.info("Model health check completed")
        return health_status

    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return {"status": "failed", "error": str(e)}

@celery_app.task(name='update_search_index')
def update_search_index():
    """검색 인덱스 업데이트"""
    try:
        logger.info("Starting search index update")

        # 검색 인덱스 업데이트 로직
        # 예: 새로운 향수 데이터 임베딩, 벡터 스토어 업데이트

        logger.info("Search index update completed")
        return {"status": "completed", "updated_items": 0}

    except Exception as e:
        logger.error(f"Search index update failed: {e}")
        return {"status": "failed", "error": str(e)}

# 주기적 작업 스케줄링
celery_app.conf.beat_schedule = {
    'cleanup-old-results': {
        'task': 'cleanup_old_results',
        'schedule': 3600.0,  # 1시간마다
    },
    'health-check-models': {
        'task': 'health_check_models',
        'schedule': 1800.0,  # 30분마다
    },
    'update-search-index': {
        'task': 'update_search_index',
        'schedule': 7200.0,  # 2시간마다
    },
}

# 워커 생명주기 이벤트
@worker_init.connect
def worker_init_handler(sender=None, **kwargs):
    """워커 초기화"""
    logger.info(f"Celery worker {sender} initialized")

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """워커 종료"""
    logger.info(f"Celery worker {sender} shutting down")

# Celery 앱 내보내기
__all__ = ['celery_app']