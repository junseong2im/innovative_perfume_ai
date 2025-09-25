"""
Fragrance AI - 애플리케이션 설정 관리 모듈

이 모듈은 Fragrance AI 시스템의 모든 설정을 중앙집중식으로 관리합니다.
환경 변수를 통해 설정을 오버라이드할 수 있으며, Pydantic를 사용하여
타입 안전성과 유효성 검사를 보장합니다.

주요 기능:
- 환경별 설정 관리 (개발/스테이징/프로덕션)
- 타입 안전한 설정 값 검증
- 민감한 정보 보호 (시크릿, 토큰 등)
- AI 모델 파라미터 중앙 관리
- 성능 최적화 설정

사용 예시:
    from fragrance_ai.core.config import settings

    api_port = settings.api_port
    debug_mode = settings.debug
"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    """
    애플리케이션 설정 클래스

    모든 설정은 환경 변수를 통해 오버라이드 가능하며,
    기본값을 제공하여 로컬 개발 환경에서도 쉽게 사용할 수 있습니다.
    """

    # ==========================================
    # 애플리케이션 기본 설정
    # ==========================================
    app_name: str = "Fragrance AI"  # 애플리케이션 이름
    app_version: str = "0.1.0"      # 현재 버전
    debug: bool = Field(default=False, env="DEBUG")  # 디버그 모드 활성화
    
    # ==========================================
    # API 서버 설정
    # ==========================================
    api_host: str = Field(default="0.0.0.0", env="API_HOST")      # API 서버 호스트
    api_port: int = Field(default=8000, env="API_PORT")           # API 서버 포트
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")  # API 경로 접두사

    # ==========================================
    # 데이터베이스 설정
    # ==========================================
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/fragrance_ai",
        env="DATABASE_URL"
    )  # PostgreSQL 데이터베이스 연결 URL
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")  # Redis 캐시 서버 URL
    
    # ==========================================
    # AI 모델 설정
    # ==========================================
    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL_NAME"
    )  # 다국어 임베딩 모델 (한국어 지원)
    generation_model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        env="GENERATION_MODEL_NAME"
    )  # 텍스트 생성용 LLM 모델
    
    # ==========================================
    # Hugging Face 설정
    # ==========================================
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")  # Hugging Face 인증 토큰
    hf_cache_dir: str = Field(default="./cache/huggingface", env="HF_CACHE_DIR")  # 모델 캐시 디렉토리
    
    # ==========================================
    # 벡터 데이터베이스 설정
    # ==========================================
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY"
    )  # ChromaDB 데이터 저장 경로
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")  # 벡터 임베딩 차원
    
    # ==========================================
    # 모델 훈련 설정
    # ==========================================
    max_seq_length: int = Field(default=512, env="MAX_SEQ_LENGTH")      # 최대 시퀀스 길이
    batch_size: int = Field(default=4, env="BATCH_SIZE")                # 배치 크기
    learning_rate: float = Field(default=2e-4, env="LEARNING_RATE")     # 학습률
    num_epochs: int = Field(default=3, env="NUM_EPOCHS")                # 훈련 에포크 수

    # LoRA (Low-Rank Adaptation) 설정
    lora_r: int = Field(default=16, env="LORA_R")                        # LoRA rank 매개변수
    lora_alpha: int = Field(default=32, env="LORA_ALPHA")               # LoRA alpha 매개변수
    lora_dropout: float = Field(default=0.1, env="LORA_DROPOUT")        # LoRA 드롭아웃 비율

    # ==========================================
    # 고급 옵티마이저 설정
    # ==========================================
    optimizer_type: str = Field(default="adamw_torch", env="OPTIMIZER_TYPE")  # 옵티마이저 유형
    adam_beta1: float = Field(default=0.9, env="ADAM_BETA1")               # Adam beta1 파라미터
    adam_beta2: float = Field(default=0.999, env="ADAM_BETA2")             # Adam beta2 파라미터
    adam_epsilon: float = Field(default=1e-8, env="ADAM_EPSILON")          # Adam epsilon 파라미터
    weight_decay: float = Field(default=0.01, env="WEIGHT_DECAY")          # 가중치 감쇠 비율
    max_grad_norm: float = Field(default=1.0, env="MAX_GRAD_NORM")         # 그래디언트 클리핑 임계값

    # ==========================================
    # 학습률 스케줄러 설정
    # ==========================================
    lr_scheduler_type: str = Field(default="cosine", env="LR_SCHEDULER_TYPE")  # 스케줄러 유형
    warmup_ratio: float = Field(default=0.1, env="WARMUP_RATIO")             # 워밍업 비율
    warmup_steps: int = Field(default=500, env="WARMUP_STEPS")               # 워밍업 스텝 수
    cosine_restarts: int = Field(default=1, env="COSINE_RESTARTS")           # 코사인 재시작 횟수
    polynomial_power: float = Field(default=1.0, env="POLYNOMIAL_POWER")     # 다항식 거듭제곱

    # ==========================================
    # 그래디언트 설정
    # ==========================================
    gradient_accumulation_steps: int = Field(default=1, env="GRADIENT_ACCUMULATION_STEPS")  # 그래디언트 누적 스텝
    gradient_checkpointing: bool = Field(default=True, env="GRADIENT_CHECKPOINTING")        # 그래디언트 체크포인팅 활성화

    # ==========================================
    # 혼합 정밀도 설정
    # ==========================================
    fp16: bool = Field(default=False, env="FP16")                # 16비트 부동소수점 사용
    bf16: bool = Field(default=True, env="BF16")                 # BFloat16 사용 (NVIDIA Ampere+)
    fp16_full_eval: bool = Field(default=False, env="FP16_FULL_EVAL")  # 평가시에도 FP16 사용

    # ==========================================
    # 조기 종료 설정
    # ==========================================
    early_stopping_patience: int = Field(default=3, env="EARLY_STOPPING_PATIENCE")      # 조기 종료 대기 에포크
    early_stopping_threshold: float = Field(default=0.001, env="EARLY_STOPPING_THRESHOLD") # 조기 종료 임계값
    
    # ==========================================
    # 검색 설정
    # ==========================================
    search_top_k: int = Field(default=10, env="SEARCH_TOP_K")                  # 검색 결과 개수
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD") # 유사도 임계값
    
    # ==========================================
    # 텍스트 생성 설정
    # ==========================================
    max_new_tokens: int = Field(default=256, env="MAX_NEW_TOKENS")  # 최대 생성 토큰 수
    temperature: float = Field(default=0.7, env="TEMPERATURE")       # 생성 다양성 조절
    top_p: float = Field(default=0.9, env="TOP_P")                  # Nucleus 샘플링 파라미터
    do_sample: bool = Field(default=True, env="DO_SAMPLE")           # 샘플링 활성화
    
    # ==========================================
    # 보안 설정
    # ==========================================
    secret_key: str = Field(
        default="INSECURE-DEFAULT-CHANGE-IMMEDIATELY-FOR-PRODUCTION-USE",
        env="SECRET_KEY"
    )  # JWT 서명용 비밀 키 (프로덕션에서 반드시 변경 필요)
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )  # 액세스 토큰 만료 시간(분)
    
    # ==========================================
    # 모니터링 설정
    # ==========================================
    wandb_project: str = Field(default="fragrance-ai", env="WANDB_PROJECT")  # Weights & Biases 프로젝트명
    log_level: str = Field(default="INFO", env="LOG_LEVEL")                  # 로깅 레벨 (DEBUG/INFO/WARNING/ERROR)
    
    # ==========================================
    # 성능 설정
    # ==========================================
    max_workers: int = Field(default=4, env="MAX_WORKERS")              # 최대 워커 스레드 수
    model_cache_size: int = Field(default=2, env="MODEL_CACHE_SIZE")     # 모델 캐시 크기

    # ==========================================
    # Celery 비동기 작업 설정
    # ==========================================
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")        # Celery 메시지 브로커
    celery_result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND") # Celery 결과 저장소

    # ==========================================
    # CORS 설정
    # ==========================================
    cors_origins: Optional[str] = Field(default=None, env="CORS_ORIGINS")  # 허용할 오리진 도메인 목록

    # ==========================================
    # ChromaDB 설정
    # ==========================================
    chroma_host: str = Field(default="localhost", env="CHROMA_HOST")  # ChromaDB 서버 호스트
    chroma_port: int = Field(default=8001, env="CHROMA_PORT")         # ChromaDB 서버 포트

    # ==========================================
    # 하이브리드 배포 설정 (AI 모델 서버 분리)
    # ==========================================
    ai_server_url: Optional[str] = Field(default=None, env="AI_SERVER_URL")  # 원격 AI 서버 URL (예: http://localhost:8001)
    use_remote_ai: bool = Field(default=False, env="USE_REMOTE_AI")          # 원격 AI 서버 사용 여부

    # ==========================================
    # Pydantic 모델 설정
    # ==========================================
    model_config = {
        "env_file": ".env",          # 환경 변수 파일 경로
        "case_sensitive": False,     # 대소문자 구분 안함
        "extra": "ignore"           # 추가 필드 무시
    }


# ==========================================
# 전역 설정 인스턴스
# ==========================================
settings = Settings()  # 애플리케이션 전체에서 사용할 설정 객체

# SECRET_KEY 속성 추가 (대문자로 접근 가능하도록)
# setattr(settings, 'SECRET_KEY', settings.secret_key)

def get_settings() -> Settings:
    """설정 객체 반환 (의존성 주입용)"""
    return settings