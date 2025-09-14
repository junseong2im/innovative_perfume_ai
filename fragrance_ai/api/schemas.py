from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Annotated, Literal
from enum import Enum, StrEnum
from datetime import datetime, timezone
import asyncio
from collections.abc import Sequence


class SearchType(StrEnum):
    """검색 타입 열거형"""
    SINGLE_COLLECTION = "single_collection"
    HYBRID = "hybrid"
    SIMILARITY = "similarity"
    SEMANTIC = "semantic"
    RAG = "rag"  # Retrieval-Augmented Generation


class RecipeType(StrEnum):
    """레시피 타입 열거형"""
    BASIC = "basic_recipe"
    DETAILED = "detailed_recipe"
    PREMIUM = "premium_recipe"
    ARTISAN = "artisan_recipe"
    COMMERCIAL = "commercial_recipe"


class ModelType(StrEnum):
    """AI 모델 타입"""
    EMBEDDING = "embedding"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    RAG = "rag"


class Priority(StrEnum):
    """우선순위 레벨"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Type aliases for better readability
QueryStr = Annotated[str, Field(min_length=1, max_length=500, description="검색 쿼리")]
TopK = Annotated[int, Field(ge=1, le=50, description="반환할 최대 결과 수")]
SimilarityScore = Annotated[float, Field(ge=0.0, le=1.0, description="유사도 점수")]
WeightScore = Annotated[float, Field(ge=0.0, le=2.0, description="가중치 점수")]


class BaseConfig:
    """공통 설정"""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
        extra="forbid",
        json_encoders={
            datetime: lambda v: v.isoformat() if v else None
        }
    )


class SemanticSearchRequest(BaseModel, BaseConfig):
    """시맨틱 검색 요청 모델 (Pydantic v2 최적화)"""
    
    query: QueryStr
    search_type: SearchType = SearchType.SINGLE_COLLECTION
    collection_name: str = Field(default="fragrance_notes", description="단일 컬렉션 검색시 컬렉션 이름")
    collections: Optional[Sequence[str]] = Field(default=None, description="하이브리드 검색시 컬렉션 목록")
    collection_weights: Optional[Dict[str, WeightScore]] = Field(default=None, description="컬렉션별 가중치")
    top_k: TopK = 10
    filters: Optional[Dict[str, Any]] = Field(default=None, description="필터링 조건")
    min_similarity: SimilarityScore = 0.5
    enable_reranking: bool = Field(default=False, description="재순위화 활성화")
    use_cache: bool = Field(default=True, description="캐시 사용 여부")
    
    @field_validator('collections')
    @classmethod
    def validate_collections(cls, v: Optional[Sequence[str]], info) -> Optional[Sequence[str]]:
        """컬렉션 검증"""
        if info.data.get('search_type') == SearchType.HYBRID and not v:
            raise ValueError("하이브리드 검색시 collections는 필수입니다")
        return v
    
    @model_validator(mode='after')
    def validate_search_config(self):
        """검색 설정 전체 검증"""
        if self.search_type == SearchType.RAG:
            if not self.collections:
                self.collections = ["fragrance_notes", "recipes", "knowledge_base"]
        
        if self.collection_weights and self.collections:
            # 가중치가 지정된 컬렉션이 실제 컬렉션 목록에 있는지 확인
            invalid_weights = set(self.collection_weights.keys()) - set(self.collections)
            if invalid_weights:
                raise ValueError(f"잘못된 컬렉션 가중치: {invalid_weights}")
        
        return self


@dataclass
class SearchResultMetadata:
    """검색 결과 메타데이터"""
    source: Optional[str] = None
    category: Optional[str] = None
    created_at: Optional[datetime] = None
    tags: Optional[List[str]] = None
    relevance_factors: Optional[Dict[str, float]] = None


class SearchResult(BaseModel, BaseConfig):
    """검색 결과 모델 (최적화)"""
    
    id: Annotated[str, Field(description="문서 ID")]
    document: Annotated[str, Field(min_length=1, description="문서 내용")]
    metadata: Dict[str, Any] = Field(default_factory=dict, description="메타데이터")
    distance: Annotated[float, Field(ge=0.0, description="거리 점수")]
    similarity: Optional[SimilarityScore] = Field(None, description="유사도 점수 (0-1)")
    collection: Optional[str] = Field(None, description="컬렉션 이름")
    weighted_score: Optional[WeightScore] = Field(None, description="가중 점수")
    rank: Optional[int] = Field(None, ge=1, description="순위")
    rerank_score: Optional[float] = Field(None, description="재순위화 점수")
    highlights: Optional[List[str]] = Field(default_factory=list, description="하이라이트된 텍스트")
    
    @property
    def effective_score(self) -> float:
        """효과적인 점수 계산"""
        return self.rerank_score or self.weighted_score or (1 - self.distance)
    
    def model_dump_optimized(self) -> Dict[str, Any]:
        """최적화된 딕셔너리 변환"""
        result = self.model_dump(exclude_none=True, exclude_unset=True)
        result['effective_score'] = self.effective_score
        return result


class SemanticSearchResponse(BaseModel):
    results: List[SearchResult] = Field(..., description="검색 결과")
    total_results: int = Field(..., description="총 결과 수")
    query: str = Field(..., description="검색 쿼리")
    search_time: float = Field(..., description="검색 소요 시간 (초)")
    filters_applied: Optional[Dict[str, Any]] = Field(None, description="적용된 필터")


class RecipeGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="생성 요청 프롬프트")
    recipe_type: RecipeType = Field(default=RecipeType.basic_recipe, description="레시피 타입")
    include_story: bool = Field(default=True, description="브랜드 스토리 포함 여부")
    mood: Optional[str] = Field(None, max_length=100, description="목표 무드")
    season: Optional[str] = Field(None, max_length=50, description="적합한 계절")
    notes_preference: Optional[List[str]] = Field(None, description="선호 노트")
    target_customer: Optional[str] = Field(None, max_length=200, description="타겟 고객")
    price_range: Optional[str] = Field(None, description="가격대")

    @validator('notes_preference')
    def validate_notes_preference(cls, v):
        if v and len(v) > 20:
            raise ValueError("선호 노트는 최대 20개까지 가능합니다")
        return v


class FragranceComposition(BaseModel):
    top_notes: Dict[str, Union[int, List[str]]] = Field(
        default_factory=dict, 
        description="톱노트 (percentage, ingredients)"
    )
    heart_notes: Dict[str, Union[int, List[str]]] = Field(
        default_factory=dict, 
        description="미들노트 (percentage, ingredients)"
    )
    base_notes: Dict[str, Union[int, List[str]]] = Field(
        default_factory=dict, 
        description="베이스노트 (percentage, ingredients)"
    )


class FragranceRecipe(BaseModel):
    name: Optional[str] = Field(None, description="향수 이름")
    concept: Optional[str] = Field(None, description="컨셉")
    mood: Optional[str] = Field(None, description="무드")
    season: Optional[str] = Field(None, description="계절")
    composition: FragranceComposition = Field(default_factory=FragranceComposition, description="향료 구성")
    longevity: Optional[str] = Field(None, description="지속력")
    sillage: Optional[str] = Field(None, description="실라지")
    story: Optional[str] = Field(None, description="브랜드 스토리")
    raw_text: str = Field(..., description="생성된 원본 텍스트")
    generated_at: str = Field(..., description="생성 시각")
    recipe_type: RecipeType = Field(..., description="레시피 타입")


class QualityScores(BaseModel):
    completeness: float = Field(..., ge=0.0, le=1.0, description="완성도 점수")
    coherence: float = Field(..., ge=0.0, le=1.0, description="일관성 점수")
    creativity: float = Field(..., ge=0.0, le=1.0, description="창의성 점수")
    technical_accuracy: float = Field(..., ge=0.0, le=1.0, description="기술적 정확성 점수")
    overall: float = Field(..., ge=0.0, le=1.0, description="전체 점수")


class RecipeGenerationResponse(BaseModel):
    recipe: FragranceRecipe = Field(..., description="생성된 레시피")
    quality_scores: QualityScores = Field(..., description="품질 점수")
    generation_time: float = Field(..., description="생성 소요 시간 (초)")
    prompt: str = Field(..., description="입력 프롬프트")


class BatchGenerationRequest(BaseModel):
    prompts: List[str] = Field(..., min_items=1, max_items=20, description="생성 요청 프롬프트 목록")
    recipe_type: RecipeType = Field(default=RecipeType.basic_recipe, description="레시피 타입")
    batch_size: int = Field(default=4, ge=1, le=8, description="배치 크기")
    include_story: bool = Field(default=False, description="스토리 포함 여부 (배치에서는 성능상 기본 false)")

    @validator('prompts')
    def validate_prompts(cls, v):
        if any(len(prompt.strip()) == 0 for prompt in v):
            raise ValueError("빈 프롬프트는 허용되지 않습니다")
        if any(len(prompt) > 500 for prompt in v):
            raise ValueError("프롬프트는 500자를 초과할 수 없습니다")
        return v


class BatchGenerationResponse(BaseModel):
    recipes: List[FragranceRecipe] = Field(..., description="생성된 레시피 목록")
    total_recipes: int = Field(..., description="총 생성된 레시피 수")
    average_quality: float = Field(..., description="평균 품질 점수")
    generation_time: float = Field(..., description="전체 생성 소요 시간 (초)")
    failed_generations: int = Field(default=0, description="실패한 생성 수")


class FragranceNote(BaseModel):
    id: str = Field(..., description="노트 ID")
    name: str = Field(..., description="노트 이름")
    korean_name: Optional[str] = Field(None, description="한국어 이름")
    description: str = Field(..., description="설명")
    category: str = Field(..., description="카테고리 (top/heart/base)")
    intensity: int = Field(..., ge=1, le=10, description="강도 (1-10)")
    longevity: int = Field(..., ge=1, le=10, description="지속력 (1-10)")
    tags: List[str] = Field(default_factory=list, description="태그")
    chemical_family: Optional[str] = Field(None, description="화학적 계열")
    price_per_ml: Optional[float] = Field(None, description="ml당 가격")
    supplier: Optional[str] = Field(None, description="공급업체")
    country_of_origin: Optional[str] = Field(None, description="원산지")


class AddFragranceNotesRequest(BaseModel):
    notes: List[FragranceNote] = Field(..., min_items=1, max_items=100, description="추가할 노트 목록")
    update_embeddings: bool = Field(default=True, description="임베딩 업데이트 여부")


class AddFragranceNotesResponse(BaseModel):
    added_count: int = Field(..., description="추가된 노트 수")
    failed_count: int = Field(default=0, description="실패한 노트 수")
    processing_time: float = Field(..., description="처리 소요 시간 (초)")
    errors: List[str] = Field(default_factory=list, description="오류 메시지")


class TrainingDataItem(BaseModel):
    input: str = Field(..., description="입력 텍스트")
    output: str = Field(..., description="출력 텍스트")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="추가 메타데이터")


class TrainingRequest(BaseModel):
    training_data: List[TrainingDataItem] = Field(..., min_items=10, description="훈련 데이터")
    model_name: Optional[str] = Field(None, description="기본 모델 이름")
    output_dir: str = Field(..., description="출력 디렉토리")
    num_epochs: int = Field(default=3, ge=1, le=10, description="에포크 수")
    learning_rate: float = Field(default=2e-4, gt=0.0, description="학습률")
    batch_size: int = Field(default=4, ge=1, le=16, description="배치 크기")
    lora_r: int = Field(default=16, ge=1, le=64, description="LoRA r 값")
    use_wandb: bool = Field(default=True, description="Wandb 로깅 사용 여부")


class TrainingResponse(BaseModel):
    training_id: str = Field(..., description="훈련 작업 ID")
    status: str = Field(..., description="훈련 상태")
    message: str = Field(..., description="상태 메시지")
    output_dir: str = Field(..., description="출력 디렉토리")
    started_at: datetime = Field(..., description="시작 시간")


class TrainingStatus(BaseModel):
    training_id: str = Field(..., description="훈련 작업 ID")
    status: str = Field(..., description="현재 상태")
    progress: float = Field(..., ge=0.0, le=1.0, description="진행률 (0-1)")
    current_epoch: int = Field(default=0, description="현재 에포크")
    total_epochs: int = Field(..., description="총 에포크")
    train_loss: Optional[float] = Field(None, description="훈련 손실")
    eval_loss: Optional[float] = Field(None, description="평가 손실")
    estimated_remaining_time: Optional[float] = Field(None, description="예상 남은 시간 (초)")
    logs: List[str] = Field(default_factory=list, description="최근 로그")


class ModelEvaluationRequest(BaseModel):
    model_path: str = Field(..., description="평가할 모델 경로")
    test_prompts: List[str] = Field(..., min_items=5, description="테스트 프롬프트")
    evaluation_metrics: List[str] = Field(
        default=["quality", "coherence", "creativity"], 
        description="평가 메트릭"
    )


class ModelEvaluationResponse(BaseModel):
    model_path: str = Field(..., description="평가된 모델 경로")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="전체 점수")
    metric_scores: Dict[str, float] = Field(..., description="메트릭별 점수")
    sample_outputs: List[Dict[str, str]] = Field(..., description="샘플 출력")
    evaluation_time: float = Field(..., description="평가 소요 시간 (초)")
    recommendations: List[str] = Field(default_factory=list, description="개선 권장사항")


class SystemStatus(BaseModel):
    status: str = Field(..., description="시스템 상태")
    version: str = Field(..., description="버전")
    uptime: float = Field(..., description="가동 시간 (초)")
    models_loaded: Dict[str, bool] = Field(..., description="로드된 모델 상태")
    memory_usage: Dict[str, float] = Field(..., description="메모리 사용량")
    active_connections: int = Field(..., description="활성 연결 수")
    requests_processed: int = Field(..., description="처리된 요청 수")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="오류 메시지")
    error_code: str = Field(..., description="오류 코드")
    timestamp: datetime = Field(..., description="오류 발생 시간")
    request_id: Optional[str] = Field(None, description="요청 ID")
    details: Optional[Dict[str, Any]] = Field(None, description="상세 정보")