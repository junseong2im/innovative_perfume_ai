from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # App Settings
    app_name: str = "Fragrance AI"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_prefix: str = Field(default="/api/v1", env="API_PREFIX")
    
    # Database Settings
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/fragrance_ai",
        env="DATABASE_URL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # AI Model Settings
    embedding_model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        env="EMBEDDING_MODEL_NAME"
    )
    generation_model_name: str = Field(
        default="meta-llama/Llama-2-7b-chat-hf",
        env="GENERATION_MODEL_NAME"
    )
    
    # Hugging Face Settings
    hf_token: Optional[str] = Field(default=None, env="HF_TOKEN")
    hf_cache_dir: str = Field(default="./cache/huggingface", env="HF_CACHE_DIR")
    
    # Vector Database Settings
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", env="CHROMA_PERSIST_DIRECTORY"
    )
    vector_dimension: int = Field(default=384, env="VECTOR_DIMENSION")
    
    # Training Settings
    max_seq_length: int = Field(default=512, env="MAX_SEQ_LENGTH")
    batch_size: int = Field(default=4, env="BATCH_SIZE")
    learning_rate: float = Field(default=2e-4, env="LEARNING_RATE")
    num_epochs: int = Field(default=3, env="NUM_EPOCHS")
    lora_r: int = Field(default=16, env="LORA_R")
    lora_alpha: int = Field(default=32, env="LORA_ALPHA")
    lora_dropout: float = Field(default=0.1, env="LORA_DROPOUT")
    
    # Search Settings
    search_top_k: int = Field(default=10, env="SEARCH_TOP_K")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Generation Settings
    max_new_tokens: int = Field(default=256, env="MAX_NEW_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    do_sample: bool = Field(default=True, env="DO_SAMPLE")
    
    # Security Settings
    secret_key: str = Field(
        default="your-super-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(
        default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # Monitoring Settings
    wandb_project: str = Field(default="fragrance-ai", env="WANDB_PROJECT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Performance Settings
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    model_cache_size: int = Field(default=2, env="MODEL_CACHE_SIZE")

    # Celery Settings
    celery_broker_url: str = Field(default="redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(default="redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


settings = Settings()