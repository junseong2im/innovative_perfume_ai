"""
추천 시스템 타입 정의
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


class RecommendationType(Enum):
    """추천 타입"""
    SIMILAR = "similar"                    # 유사한 향수
    COMPLEMENTARY = "complementary"        # 보완적인 향수
    SEASONAL = "seasonal"                  # 계절 맞춤
    MOOD_BASED = "mood_based"             # 무드 기반
    OCCASION_BASED = "occasion_based"     # 상황 기반
    INNOVATIVE = "innovative"             # 혁신적 조합
    SAFE_CHOICE = "safe_choice"           # 안전한 선택
    MASTERPIECE = "masterpiece"          # 마스터피스 스타일


class UserPreferenceType(Enum):
    """사용자 선호도 타입"""
    IMPLICIT = "implicit"      # 암시적 (행동 패턴)
    EXPLICIT = "explicit"      # 명시적 (직접 평가)
    CONTEXTUAL = "contextual"  # 상황적 (날씨, 시간 등)


class Season(Enum):
    """계절"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


class Mood(Enum):
    """무드"""
    ROMANTIC = "romantic"
    ENERGETIC = "energetic"
    CALM = "calm"
    CONFIDENT = "confident"
    PLAYFUL = "playful"
    SOPHISTICATED = "sophisticated"
    FRESH = "fresh"
    MYSTERIOUS = "mysterious"


class Occasion(Enum):
    """상황"""
    WORK = "work"
    DATE = "date"
    PARTY = "party"
    CASUAL = "casual"
    FORMAL = "formal"
    TRAVEL = "travel"
    SPORT = "sport"
    RELAXATION = "relaxation"


@dataclass
class WeatherCondition:
    """날씨 조건"""
    temperature: float  # 섭씨
    humidity: float     # 습도 (0-100)
    season: Season
    time_of_day: str   # "morning", "afternoon", "evening", "night"

    @property
    def is_hot(self) -> bool:
        return self.temperature > 25

    @property
    def is_humid(self) -> bool:
        return self.humidity > 70


@dataclass
class UserContext:
    """사용자 상황 정보"""
    mood: Optional[Mood] = None
    occasion: Optional[Occasion] = None
    weather: Optional[WeatherCondition] = None
    time_of_day: Optional[str] = None
    location: Optional[str] = None
    companion: Optional[str] = None  # "alone", "friends", "partner", "family"


@dataclass
class FragranceNote:
    """향수 노트"""
    name: str
    category: str  # "top", "heart", "base"
    intensity: float  # 0-1
    longevity: float  # 0-1
    projection: float  # 0-1
    tags: List[str] = field(default_factory=list)


@dataclass
class FragranceProfile:
    """향수 프로필"""
    fragrance_id: str
    name: str
    brand: str
    fragrance_family: str
    notes: List[FragranceNote]
    overall_intensity: float
    longevity: float
    projection: float
    season_scores: Dict[Season, float]
    mood_scores: Dict[Mood, float]
    occasion_scores: Dict[Occasion, float]
    target_gender: str
    price_range: str
    popularity_score: float
    quality_score: float
    uniqueness_score: float

    def get_season_score(self, season: Season) -> float:
        """계절 적합도 점수 반환"""
        return self.season_scores.get(season, 0.5)

    def get_mood_score(self, mood: Mood) -> float:
        """무드 적합도 점수 반환"""
        return self.mood_scores.get(mood, 0.5)

    def get_occasion_score(self, occasion: Occasion) -> float:
        """상황 적합도 점수 반환"""
        return self.occasion_scores.get(occasion, 0.5)


@dataclass
class UserPreference:
    """사용자 선호도"""
    user_id: str
    preference_type: UserPreferenceType
    fragrance_families: Dict[str, float]  # 향수 계열별 선호도
    notes_preference: Dict[str, float]    # 노트별 선호도
    intensity_preference: float           # 선호 강도 (0-1)
    longevity_preference: float          # 선호 지속성 (0-1)
    projection_preference: float         # 선호 확산성 (0-1)
    brand_preference: Dict[str, float]   # 브랜드별 선호도
    price_sensitivity: float             # 가격 민감도 (0-1)
    novelty_seeking: float              # 새로움 추구 성향 (0-1)
    last_updated: datetime
    confidence_score: float = 0.5      # 선호도 신뢰성 점수

    def update_note_preference(self, note: str, rating: float):
        """노트 선호도 업데이트"""
        if note not in self.notes_preference:
            self.notes_preference[note] = rating
        else:
            # 기존 선호도와 새 평가의 가중평균
            current = self.notes_preference[note]
            self.notes_preference[note] = (current * 0.7) + (rating * 0.3)

        self.last_updated = datetime.now()


@dataclass
class RecommendationContext:
    """추천 요청 컨텍스트"""
    user_id: str
    user_context: UserContext
    user_preferences: Optional[UserPreference] = None
    recommendation_type: RecommendationType = RecommendationType.SIMILAR
    max_recommendations: int = 10
    exclude_owned: bool = True
    exclude_recently_recommended: bool = True
    min_quality_score: float = 6.0
    budget_range: Optional[Tuple[float, float]] = None
    requested_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationResult:
    """추천 결과"""
    fragrance_profile: FragranceProfile
    score: float
    reasoning: str
    confidence: float
    match_details: Dict[str, float]  # 각 기준별 매칭 점수
    recommendation_type: RecommendationType

    def __post_init__(self):
        """결과 검증"""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


@dataclass
class RecommendationResponse:
    """추천 응답"""
    recommendations: List[RecommendationResult]
    total_candidates: int
    processing_time: float
    context: RecommendationContext
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_recommendation(self) -> Optional[RecommendationResult]:
        """최고 추천 결과 반환"""
        return self.recommendations[0] if self.recommendations else None

    def get_recommendations_by_type(self, rec_type: RecommendationType) -> List[RecommendationResult]:
        """타입별 추천 결과 필터링"""
        return [rec for rec in self.recommendations if rec.recommendation_type == rec_type]