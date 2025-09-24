"""
향수 레시피 스키마 - 고객용/관리자용 분리
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class UserRole(str, Enum):
    """사용자 역할"""
    CUSTOMER = "customer"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class FragranceNoteInfo(BaseModel):
    """향료 노트 정보 (공통)"""
    name: str = Field(..., description="향료 노트 이름")
    name_korean: str = Field(..., description="한국어 이름")
    fragrance_family: str = Field(..., description="향족")
    note_type: str = Field(..., description="노트 타입 (top/middle/base)")

class FragranceNoteCustomer(FragranceNoteInfo):
    """고객용 향료 노트 정보 (제한된 정보)"""
    description: str = Field(..., description="향료 설명")
    mood_tags: List[str] = Field(default=[], description="무드 태그")
    season_tags: List[str] = Field(default=[], description="계절 태그")

class FragranceNoteAdmin(FragranceNoteInfo):
    """관리자용 향료 노트 정보 (전체 정보)"""
    description: str = Field(..., description="향료 설명")
    description_korean: str = Field(..., description="한국어 설명")
    origin: str = Field(..., description="원산지")
    extraction_method: str = Field(..., description="추출 방법")
    intensity: float = Field(..., description="강도")
    longevity: float = Field(..., description="지속성")
    sillage: float = Field(..., description="확산성")
    price_per_ml: float = Field(..., description="ml당 가격")
    supplier: str = Field(..., description="공급업체")
    grade: str = Field(..., description="등급")
    mood_tags: List[str] = Field(default=[], description="무드 태그")
    season_tags: List[str] = Field(default=[], description="계절 태그")
    gender_tags: List[str] = Field(default=[], description="성별 태그")

class RecipeIngredientCustomer(BaseModel):
    """고객용 레시피 재료 (제한된 정보)"""
    note: FragranceNoteCustomer = Field(..., description="향료 노트")
    role: str = Field(..., description="역할 (top/middle/base/accent)")

class RecipeIngredientAdmin(BaseModel):
    """관리자용 레시피 재료 (전체 정보)"""
    note: FragranceNoteAdmin = Field(..., description="향료 노트")
    percentage: float = Field(..., description="비율 (%)")
    ml_amount: float = Field(..., description="ml 양")
    role: str = Field(..., description="역할")
    order: int = Field(..., description="추가 순서")
    notes: Optional[str] = Field(None, description="특별 참고사항")

class FragranceProfileCustomer(BaseModel):
    """고객용 향수 프로필"""
    name: str = Field(..., description="향수 이름")
    description: str = Field(..., description="향수 설명")
    story: str = Field(..., description="향수 스토리")
    fragrance_family: str = Field(..., description="주요 향족")
    mood: str = Field(..., description="분위기")
    season: str = Field(..., description="적합한 계절")
    gender: str = Field(..., description="성별")
    intensity: str = Field(..., description="강도 (light/moderate/strong)")
    longevity: str = Field(..., description="지속성 (short/medium/long)")
    sillage: str = Field(..., description="확산성 (intimate/moderate/strong)")

    # 향료 구성 (간단한 설명만)
    top_notes_description: str = Field(..., description="탑 노트 설명")
    middle_notes_description: str = Field(..., description="미들 노트 설명")
    base_notes_description: str = Field(..., description="베이스 노트 설명")

    # 사용 추천
    occasion: List[str] = Field(default=[], description="사용 occasion")
    time_of_day: List[str] = Field(default=[], description="사용 시간대")

    # 한국적 특색
    korean_inspiration: Optional[str] = Field(None, description="한국적 영감")
    regional_character: Optional[str] = Field(None, description="지역적 특성")

class FragranceRecipeAdmin(BaseModel):
    """관리자용 상세 레시피"""
    # 기본 정보
    recipe_id: str = Field(..., description="레시피 ID")
    name: str = Field(..., description="향수 이름")
    description: str = Field(..., description="향수 설명")

    # 상세 레시피 정보
    ingredients: List[RecipeIngredientAdmin] = Field(..., description="재료 목록")

    # 제조 정보
    total_volume: float = Field(..., description="총 용량 (ml)")
    alcohol_percentage: float = Field(default=70.0, description="알코올 비율")
    maturation_time: int = Field(default=30, description="숙성 기간 (일)")

    # 비용 계산
    total_cost: float = Field(..., description="총 제조 비용")
    cost_per_ml: float = Field(..., description="ml당 비용")
    suggested_price: float = Field(..., description="권장 판매가")

    # 제조 지침
    mixing_instructions: List[str] = Field(default=[], description="혼합 지침")
    storage_instructions: str = Field(..., description="보관 지침")
    safety_notes: List[str] = Field(default=[], description="안전 주의사항")

    # 품질 관리
    expected_color: str = Field(..., description="예상 색상")
    expected_clarity: str = Field(..., description="예상 투명도")

    # 메타데이터
    created_at: datetime = Field(..., description="생성 시간")
    created_by: str = Field(..., description="생성자")
    version: str = Field(default="1.0", description="버전")

class RecipeGenerationRequest(BaseModel):
    """레시피 생성 요청"""
    # 기본 요청 정보
    fragrance_family: str = Field(..., description="향족")
    mood: str = Field(..., description="원하는 분위기")
    intensity: str = Field(default="moderate", description="강도")
    season: Optional[str] = Field(None, description="계절")
    gender: Optional[str] = Field("unisex", description="성별")

    # 한국적 특색 요청
    korean_region: Optional[str] = Field(None, description="한국 지역 (제주/강원/경상/전라/충청/경기)")
    korean_season: Optional[str] = Field(None, description="한국 절기")
    traditional_element: Optional[str] = Field(None, description="전통 요소")

    # 고급 옵션
    budget_range: Optional[str] = Field("medium", description="예산 범위 (low/medium/high)")
    complexity: Optional[str] = Field("medium", description="복잡도 (simple/medium/complex)")
    unique_request: Optional[str] = Field(None, description="특별 요청사항")

class RecipeGenerationResponse(BaseModel):
    """레시피 생성 응답 (역할에 따라 다른 정보 제공)"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")

    # 공통 정보
    generation_id: str = Field(..., description="생성 ID")

    # 고객용 정보
    customer_info: Optional[FragranceProfileCustomer] = Field(None, description="고객용 향수 정보")

    # 관리자용 정보 (관리자일 때만)
    admin_recipe: Optional[FragranceRecipeAdmin] = Field(None, description="관리자용 상세 레시피")

class RecipeListResponse(BaseModel):
    """레시피 목록 응답"""
    recipes: List[Dict[str, Any]] = Field(..., description="레시피 목록")
    total_count: int = Field(..., description="총 개수")
    page: int = Field(..., description="현재 페이지")
    page_size: int = Field(..., description="페이지 크기")

class UserAuthRequest(BaseModel):
    """사용자 인증 요청"""
    username: str = Field(..., description="사용자명")
    password: str = Field(..., description="비밀번호")

class UserAuthResponse(BaseModel):
    """사용자 인증 응답"""
    success: bool = Field(..., description="인증 성공 여부")
    access_token: Optional[str] = Field(None, description="액세스 토큰")
    user_role: Optional[UserRole] = Field(None, description="사용자 역할")
    message: str = Field(..., description="응답 메시지")

class AdminStats(BaseModel):
    """관리자용 통계"""
    total_recipes: int = Field(..., description="총 레시피 수")
    total_notes: int = Field(..., description="총 향료 노트 수")
    popular_families: Dict[str, int] = Field(..., description="인기 향족")
    recent_activity: List[Dict[str, Any]] = Field(..., description="최근 활동")
    cost_analysis: Dict[str, float] = Field(..., description="비용 분석")