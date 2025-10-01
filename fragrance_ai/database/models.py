from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean, DateTime, 
    JSON, ForeignKey, Index, UniqueConstraint, CheckConstraint
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

from .base import Base

class TimestampMixin:
    """타임스탬프 믹스인"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), 
                       onupdate=func.now(), nullable=False)

class FragranceNote(Base, TimestampMixin):
    """향료 노트 모델"""
    __tablename__ = "fragrance_notes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, index=True)
    name_korean = Column(String(100), index=True)
    name_english = Column(String(100), index=True)
    
    # 노트 분류
    note_type = Column(String(20), nullable=False, index=True)  # top, middle, base
    fragrance_family = Column(String(50), index=True)  # citrus, floral, woody, oriental, etc.
    
    # 향료 특성
    intensity = Column(Float, nullable=False, default=5.0)  # 1-10 강도
    longevity = Column(Float, nullable=False, default=5.0)  # 1-10 지속성
    sillage = Column(Float, nullable=False, default=5.0)    # 1-10 확산성
    
    # 설명 및 메타데이터
    description = Column(Text)
    description_korean = Column(Text)
    origin = Column(String(100))  # 원산지
    extraction_method = Column(String(50))  # 추출 방법
    
    # 태그 및 무드
    mood_tags = Column(JSON, default=list)  # ["fresh", "romantic", "mysterious"]
    season_tags = Column(JSON, default=list)  # ["spring", "summer", "fall", "winter"]
    gender_tags = Column(JSON, default=list)  # ["masculine", "feminine", "unisex"]
    
    # 가격 정보
    price_per_ml = Column(Float)
    supplier = Column(String(100))
    grade = Column(String(20))  # premium, standard, synthetic
    
    # 검색 최적화
    search_keywords = Column(Text)  # 검색용 키워드 (한글, 영문 통합)
    
    # 관계
    recipe_ingredients = relationship("RecipeIngredient", back_populates="note")
    
    __table_args__ = (
        Index('ix_fragrance_notes_search', 'name', 'name_korean', 'name_english'),
        Index('ix_fragrance_notes_type_family', 'note_type', 'fragrance_family'),
        CheckConstraint('intensity >= 1 AND intensity <= 10', name='check_intensity_range'),
        CheckConstraint('longevity >= 1 AND longevity <= 10', name='check_longevity_range'),
        CheckConstraint('sillage >= 1 AND sillage <= 10', name='check_sillage_range'),
        CheckConstraint("note_type IN ('top', 'middle', 'base')", name='check_note_type'),
    )

class Recipe(Base, TimestampMixin):
    """향수 레시피 모델"""
    __tablename__ = "recipes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False, index=True)
    name_korean = Column(String(200), index=True)
    
    # 레시피 분류
    recipe_type = Column(String(50), nullable=False, index=True)  # basic, detailed, premium
    fragrance_family = Column(String(50), nullable=False, index=True)
    
    # 레시피 특성
    complexity = Column(Integer, nullable=False, default=5)  # 1-10 복잡도
    estimated_cost = Column(Float)  # 예상 제조 비용
    batch_size_ml = Column(Integer, default=100)  # 기본 배치 크기
    
    # 설명
    description = Column(Text)
    description_korean = Column(Text)
    concept = Column(Text)  # 컨셉트
    target_audience = Column(String(200))  # 타겟 고객
    
    # 생성 정보
    generation_model = Column(String(100))  # 생성에 사용된 모델
    generation_params = Column(JSON)  # 생성 파라미터
    quality_score = Column(Float)  # AI 품질 평가 점수
    
    # 향수 특성
    sillage = Column(Float, default=5.0)
    longevity = Column(Float, default=5.0)
    complexity_rating = Column(Float, default=5.0)
    
    # 태그
    mood_tags = Column(JSON, default=list)
    season_tags = Column(JSON, default=list) 
    gender_tags = Column(JSON, default=list)
    
    # 상태
    status = Column(String(20), default='draft', index=True)  # draft, reviewed, approved, archived
    is_public = Column(Boolean, default=False, index=True)
    
    # 생산 정보
    production_notes = Column(JSON)  # 생산 관련 메모
    maceration_time_days = Column(Integer)  # 숙성 기간
    aging_requirements = Column(Text)  # 숙성 조건
    
    # 관계
    ingredients = relationship("RecipeIngredient", back_populates="recipe", cascade="all, delete-orphan")
    evaluations = relationship("RecipeEvaluation", back_populates="recipe", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_recipes_search', 'name', 'name_korean'),
        Index('ix_recipes_type_family', 'recipe_type', 'fragrance_family'),
        Index('ix_recipes_status_public', 'status', 'is_public'),
        CheckConstraint('complexity >= 1 AND complexity <= 10', name='check_complexity_range'),
        CheckConstraint("recipe_type IN ('basic', 'detailed', 'premium', 'variation')", 
                       name='check_recipe_type'),
        CheckConstraint("status IN ('draft', 'reviewed', 'approved', 'archived')", 
                       name='check_recipe_status'),
    )

class RecipeIngredient(Base, TimestampMixin):
    """레시피 재료 모델"""
    __tablename__ = "recipe_ingredients"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recipe_id = Column(String, ForeignKey("recipes.id", ondelete="CASCADE"), nullable=False)
    note_id = Column(String, ForeignKey("fragrance_notes.id"), nullable=False)
    
    # 농도 정보
    percentage = Column(Float, nullable=False)  # 농도 (%)
    weight_grams = Column(Float)  # 100ml 기준 중량
    
    # 역할 및 특성
    role = Column(String(20), nullable=False)  # primary, accent, bridge, modifier
    note_position = Column(String(20), nullable=False)  # top, middle, base
    
    # 추가 정보
    notes = Column(Text)  # 특별한 노트나 지시사항
    is_optional = Column(Boolean, default=False)  # 선택적 재료 여부
    alternative_note_ids = Column(JSON, default=list)  # 대체 가능한 노트들
    
    # 관계
    recipe = relationship("Recipe", back_populates="ingredients")
    note = relationship("FragranceNote", back_populates="recipe_ingredients")
    
    __table_args__ = (
        Index('ix_recipe_ingredients_recipe', 'recipe_id'),
        Index('ix_recipe_ingredients_note', 'note_id'),
        UniqueConstraint('recipe_id', 'note_id', name='uq_recipe_note'),
        CheckConstraint('percentage > 0 AND percentage <= 100', name='check_percentage_range'),
        CheckConstraint("role IN ('primary', 'accent', 'bridge', 'modifier')", name='check_role'),
        CheckConstraint("note_position IN ('top', 'middle', 'base')", name='check_note_position'),
    )

class Brand(Base, TimestampMixin):
    """향수 브랜드 모델"""
    __tablename__ = "brands"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True, index=True)
    name_korean = Column(String(100), index=True)
    
    # 브랜드 정보
    country = Column(String(50))
    founded_year = Column(Integer)
    brand_type = Column(String(50))  # luxury, niche, commercial, artisan
    
    # 설명
    description = Column(Text)
    description_korean = Column(Text)
    heritage_story = Column(Text)
    
    # 특성
    signature_style = Column(String(200))  # 시그니처 스타일
    price_range = Column(String(20))  # budget, mid-range, luxury, ultra-luxury
    target_market = Column(JSON, default=list)  # 타겟 시장
    
    # 메타데이터
    website = Column(String(200))
    logo_url = Column(String(500))
    
    __table_args__ = (
        Index('ix_brands_search', 'name', 'name_korean'),
        CheckConstraint("brand_type IN ('luxury', 'niche', 'commercial', 'artisan')", 
                       name='check_brand_type'),
    )

class RecipeEvaluation(Base, TimestampMixin):
    """레시피 평가 모델"""
    __tablename__ = "recipe_evaluations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    recipe_id = Column(String, ForeignKey("recipes.id", ondelete="CASCADE"), nullable=False)
    
    # 평가자 정보
    evaluator_type = Column(String(20), nullable=False)  # ai, human, expert
    evaluator_id = Column(String(100))  # 평가자 식별자
    
    # 평가 점수 (1-10)
    overall_score = Column(Float, nullable=False)
    creativity_score = Column(Float)
    feasibility_score = Column(Float)
    balance_score = Column(Float)
    marketability_score = Column(Float)
    
    # 세부 평가
    evaluation_criteria = Column(JSON)  # 평가 기준별 점수
    strengths = Column(JSON, default=list)  # 강점들
    weaknesses = Column(JSON, default=list)  # 약점들
    improvements = Column(JSON, default=list)  # 개선 제안
    
    # 평가 메모
    comments = Column(Text)
    evaluation_notes = Column(JSON)  # 구조화된 평가 노트
    
    # 관계
    recipe = relationship("Recipe", back_populates="evaluations")
    
    __table_args__ = (
        Index('ix_recipe_evaluations_recipe', 'recipe_id'),
        Index('ix_recipe_evaluations_evaluator', 'evaluator_type', 'evaluator_id'),
        CheckConstraint('overall_score >= 1 AND overall_score <= 10', name='check_overall_score_range'),
        CheckConstraint("evaluator_type IN ('ai', 'human', 'expert')", name='check_evaluator_type'),
    )

class TrainingDataset(Base, TimestampMixin):
    """훈련 데이터셋 모델"""
    __tablename__ = "training_datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text)
    
    # 데이터셋 정보
    dataset_type = Column(String(50), nullable=False)  # recipe_generation, note_embedding, quality_evaluation
    version = Column(String(20), nullable=False, default="1.0")
    
    # 통계 정보
    total_samples = Column(Integer, nullable=False, default=0)
    training_samples = Column(Integer, nullable=False, default=0)
    validation_samples = Column(Integer, nullable=False, default=0)
    test_samples = Column(Integer, nullable=False, default=0)
    
    # 데이터 경로
    data_path = Column(String(500))
    metadata_path = Column(String(500))
    
    # 상태 및 품질
    status = Column(String(20), default='preparing')  # preparing, ready, training, completed, archived
    quality_score = Column(Float)  # 데이터 품질 점수
    
    # 생성 정보
    creation_method = Column(String(50))  # manual, automated, augmented
    source_info = Column(JSON)  # 데이터 소스 정보
    
    __table_args__ = (
        Index('ix_training_datasets_type_status', 'dataset_type', 'status'),
        CheckConstraint("dataset_type IN ('recipe_generation', 'note_embedding', 'quality_evaluation')", 
                       name='check_dataset_type'),
        CheckConstraint("status IN ('preparing', 'ready', 'training', 'completed', 'archived')", 
                       name='check_dataset_status'),
    )

class ModelCheckpoint(Base, TimestampMixin):
    """모델 체크포인트 모델"""
    __tablename__ = "model_checkpoints"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # embedding, generation, evaluation
    
    # 체크포인트 정보
    checkpoint_path = Column(String(500), nullable=False)
    version = Column(String(20), nullable=False)
    epoch = Column(Integer)
    step = Column(Integer)
    
    # 성능 메트릭
    metrics = Column(JSON, default=dict)  # 성능 지표들
    loss = Column(Float)
    validation_score = Column(Float)
    
    # 훈련 정보
    training_dataset_id = Column(String, ForeignKey("training_datasets.id"))
    training_params = Column(JSON)  # 훈련 파라미터
    training_duration_hours = Column(Float)
    
    # 상태
    status = Column(String(20), default='training')  # training, completed, deployed, archived
    is_best = Column(Boolean, default=False)  # 최고 성능 모델 여부
    
    # 메타데이터
    model_size_mb = Column(Float)
    description = Column(Text)
    tags = Column(JSON, default=list)
    
    __table_args__ = (
        Index('ix_model_checkpoints_name_type', 'model_name', 'model_type'),
        Index('ix_model_checkpoints_status', 'status'),
        CheckConstraint("model_type IN ('embedding', 'generation', 'evaluation')", 
                       name='check_model_type'),
        CheckConstraint("status IN ('training', 'completed', 'deployed', 'archived')", 
                       name='check_checkpoint_status'),
    )

class SearchLog(Base, TimestampMixin):
    """검색 로그 모델"""
    __tablename__ = "search_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # 검색 정보
    query = Column(Text, nullable=False)
    search_type = Column(String(50), nullable=False)
    collection_names = Column(JSON, default=list)
    
    # 검색 파라미터
    top_k = Column(Integer, default=10)
    similarity_threshold = Column(Float, default=0.7)
    filters = Column(JSON, default=dict)
    
    # 결과 정보
    results_count = Column(Integer, nullable=False, default=0)
    search_time_ms = Column(Float)  # 검색 시간 (밀리초)
    
    # 사용자 정보
    user_id = Column(String(100))
    session_id = Column(String(100))
    ip_address = Column(String(50))
    user_agent = Column(String(500))
    
    # 성능 메트릭
    cache_hit = Column(Boolean, default=False)
    embedding_time_ms = Column(Float)
    vector_search_time_ms = Column(Float)
    
    __table_args__ = (
        Index('ix_search_logs_query', 'query'),
        Index('ix_search_logs_user_session', 'user_id', 'session_id'),
        Index('ix_search_logs_created_at', 'created_at'),
    )

class User(Base, TimestampMixin):
    """사용자 모델"""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)

    # 프로필 정보
    full_name = Column(String(200))
    phone = Column(String(50))
    profile_image_url = Column(String(500))

    # 인증 정보
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # 권한
    role = Column(String(50), default='customer')  # customer, admin, expert

    # 선호도
    preferences = Column(JSON, default=dict)
    favorite_notes = Column(JSON, default=list)

    # 관계
    generated_recipes = relationship("GeneratedRecipe", back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint("role IN ('customer', 'admin', 'expert')", name='check_user_role'),
    )

class GeneratedRecipe(Base, TimestampMixin):
    """AI 생성 향수 레시피 모델 (IP 보호용)"""
    __tablename__ = "generated_recipes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    conversation_id = Column(String(100), nullable=False, index=True)

    # 사용자 입력
    user_prompt = Column(Text, nullable=False)
    conversation_history = Column(JSON, default=list)

    # 생성된 레시피 (마스터 출력 - IP)
    recipe_name = Column(String(200), nullable=False)
    recipe_description = Column(Text)

    # 상세 조합 (백엔드 전용)
    master_formula = Column(JSON, nullable=False)  # 완전한 레시피 데이터
    top_notes = Column(JSON, default=list)
    heart_notes = Column(JSON, default=list)
    base_notes = Column(JSON, default=list)

    # 과학적 검증 점수
    harmony_score = Column(Float)
    stability_score = Column(Float)
    longevity_score = Column(Float)
    sillage_score = Column(Float)
    overall_score = Column(Float)

    # 메타데이터
    generation_model = Column(String(100))
    validation_model = Column(String(100))
    generation_timestamp = Column(DateTime(timezone=True), default=func.now())

    # 상태 관리
    is_validated = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=False)
    admin_notes = Column(Text)

    # 사용자 피드백
    user_rating = Column(Float)
    user_feedback = Column(Text)

    # 관계
    user = relationship("User", back_populates="generated_recipes")

    __table_args__ = (
        Index('ix_generated_recipes_user', 'user_id'),
        Index('ix_generated_recipes_conversation', 'conversation_id'),
        Index('ix_generated_recipes_timestamp', 'generation_timestamp'),
        CheckConstraint('harmony_score >= 0 AND harmony_score <= 10', name='check_harmony_range'),
        CheckConstraint('stability_score >= 0 AND stability_score <= 10', name='check_stability_range'),
        CheckConstraint('longevity_score >= 0 AND longevity_score <= 10', name='check_longevity_range'),
        CheckConstraint('sillage_score >= 0 AND sillage_score <= 10', name='check_sillage_range'),
        CheckConstraint('overall_score >= 0 AND overall_score <= 10', name='check_overall_range'),
    )

class GenerationLog(Base, TimestampMixin):
    """생성 로그 모델"""
    __tablename__ = "generation_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    request_id = Column(String(100), nullable=False, index=True)

    # 생성 요청 정보
    generation_type = Column(String(50), nullable=False)
    input_params = Column(JSON, nullable=False)
    generation_config = Column(JSON, default=dict)

    # 결과 정보
    recipe_id = Column(String, ForeignKey("recipes.id"))
    generation_time_ms = Column(Float)
    quality_score = Column(Float)

    # 상태
    status = Column(String(20), nullable=False)  # requested, processing, completed, failed
    error_message = Column(Text)

    # 사용자 정보
    user_id = Column(String(100))
    session_id = Column(String(100))

    # 모델 정보
    model_version = Column(String(50))
    model_params = Column(JSON)

    __table_args__ = (
        Index('ix_generation_logs_request_id', 'request_id'),
        Index('ix_generation_logs_user_session', 'user_id', 'session_id'),
        Index('ix_generation_logs_status', 'status'),
        CheckConstraint("status IN ('requested', 'processing', 'completed', 'failed')",
                       name='check_generation_status'),
    )


class OlfactoryDNA(Base, TimestampMixin):
    """향수 DNA 모델 - 향수의 유전적 청사진"""
    __tablename__ = "olfactory_dna"

    dna_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # 계보 정보 (부모 컨셉 저장)
    lineage = Column(JSON, nullable=False, default=dict)
    # 예: {
    #   "parent_concepts": ["fresh", "citrus", "aquatic"],
    #   "generation": 1,
    #   "ancestors": ["DNA_ID_1", "DNA_ID_2"]
    # }

    # 유전자형 (노트 구성)
    genotype = Column(JSON, nullable=False)
    # 예: {
    #   "top_notes": [
    #     {"note": "bergamot", "percentage": 15, "intensity": 8},
    #     {"note": "lemon", "percentage": 10, "intensity": 7}
    #   ],
    #   "heart_notes": [
    #     {"note": "jasmine", "percentage": 20, "intensity": 6},
    #     {"note": "rose", "percentage": 15, "intensity": 7}
    #   ],
    #   "base_notes": [
    #     {"note": "sandalwood", "percentage": 25, "intensity": 5},
    #     {"note": "musk", "percentage": 15, "intensity": 4}
    #   ],
    #   "modifiers": [
    #     {"type": "enhancer", "note": "iso_e_super", "percentage": 0.5}
    #   ]
    # }

    # 스토리와 컨셉
    story = Column(Text, nullable=False)

    # DNA 특성
    dna_signature = Column(String(200))  # 고유 식별자 또는 해시
    complexity_level = Column(Integer, default=5)  # 1-10
    innovation_score = Column(Float)  # 0-10

    # 메타데이터
    creator_id = Column(String(100))  # AI 모델 또는 조향사 ID
    creation_method = Column(String(50))  # ai_generated, manual, hybrid
    tags = Column(JSON, default=list)  # ["summer", "fresh", "unisex", etc.]

    # 품질 및 검증
    is_validated = Column(Boolean, default=False)
    validation_scores = Column(JSON, default=dict)  # 각종 검증 점수

    # 관계
    phenotypes = relationship("ScentPhenotype", back_populates="dna", cascade="all, delete-orphan")

    __table_args__ = (
        Index('ix_olfactory_dna_signature', 'dna_signature'),
        Index('ix_olfactory_dna_creator', 'creator_id'),
        Index('ix_olfactory_dna_creation_method', 'creation_method'),
        CheckConstraint('complexity_level >= 1 AND complexity_level <= 10', name='check_dna_complexity'),
        CheckConstraint('innovation_score >= 0 AND innovation_score <= 10', name='check_dna_innovation'),
    )


class ScentPhenotype(Base, TimestampMixin):
    """향수 표현형 모델 - DNA의 실제 표현"""
    __tablename__ = "scent_phenotypes"

    phenotype_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))

    # DNA 참조 (Foreign Key)
    based_on_dna = Column(String, ForeignKey("olfactory_dna.dna_id", ondelete="CASCADE"), nullable=False)

    # 환경적 요인 (피드백 요약)
    epigenetic_trigger = Column(JSON, nullable=False)
    # 예: {
    #   "customer_feedback": ["too strong", "needs more freshness"],
    #   "market_trends": ["minimalist", "clean beauty"],
    #   "seasonal_factors": ["summer", "humid climate"],
    #   "cultural_preferences": ["korean_market", "subtle_fragrance"]
    # }

    # 최종 레시피 (JSON)
    recipe = Column(JSON, nullable=False)
    # 예: {
    #   "formula": {
    #     "top_notes": [...],  # DNA genotype + 조정사항
    #     "heart_notes": [...],
    #     "base_notes": [...],
    #     "final_adjustments": [
    #       {"action": "diluted", "note": "bergamot", "factor": 0.7},
    #       {"action": "added", "note": "green_tea", "percentage": 3}
    #     ]
    #   },
    #   "production_notes": {
    #     "dilution": "80% concentration",
    #     "maceration_time": "6 weeks",
    #     "filtering": "double_filtered"
    #   },
    #   "packaging": {
    #     "bottle_type": "minimalist_glass",
    #     "volume": "50ml",
    #     "atomizer": "fine_mist"
    #   }
    # }

    # 표현형 설명
    description = Column(Text, nullable=False)

    # 표현형 특성
    phenotype_name = Column(String(200))
    phenotype_code = Column(String(100), unique=True)  # 고유 식별 코드
    variant_type = Column(String(50))  # original, seasonal, limited, custom

    # 성능 메트릭
    market_performance = Column(JSON, default=dict)  # 시장 반응 데이터
    customer_rating = Column(Float)  # 평균 고객 평점
    expert_score = Column(Float)  # 전문가 평가 점수

    # 생산 정보
    production_status = Column(String(50), default='concept')  # concept, prototype, production, discontinued
    batch_info = Column(JSON, default=dict)  # 생산 배치 정보

    # 관계
    dna = relationship("OlfactoryDNA", back_populates="phenotypes")

    __table_args__ = (
        Index('ix_scent_phenotypes_dna', 'based_on_dna'),
        Index('ix_scent_phenotypes_code', 'phenotype_code'),
        Index('ix_scent_phenotypes_variant', 'variant_type'),
        Index('ix_scent_phenotypes_status', 'production_status'),
        CheckConstraint('customer_rating >= 0 AND customer_rating <= 5', name='check_phenotype_rating'),
        CheckConstraint('expert_score >= 0 AND expert_score <= 10', name='check_phenotype_expert_score'),
        CheckConstraint("variant_type IN ('original', 'seasonal', 'limited', 'custom')",
                       name='check_phenotype_variant'),
        CheckConstraint("production_status IN ('concept', 'prototype', 'production', 'discontinued')",
                       name='check_phenotype_status'),
    )