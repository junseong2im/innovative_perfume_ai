"""
추천 시스템 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import asyncio

from .types import (
    RecommendationContext,
    RecommendationResponse,
    UserPreference,
    FragranceProfile
)


class RecommendationEngine(ABC):
    """추천 엔진 인터페이스"""

    @abstractmethod
    async def get_recommendations(
        self,
        context: RecommendationContext
    ) -> RecommendationResponse:
        """추천 결과 생성"""
        pass

    @abstractmethod
    async def update_user_feedback(
        self,
        user_id: str,
        fragrance_id: str,
        rating: float,
        feedback_type: str = "rating"
    ) -> bool:
        """사용자 피드백 업데이트"""
        pass

    @abstractmethod
    def get_supported_recommendation_types(self) -> List[str]:
        """지원하는 추천 타입 목록 반환"""
        pass


class UserProfileManager(ABC):
    """사용자 프로필 관리 인터페이스"""

    @abstractmethod
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreference]:
        """사용자 선호도 조회"""
        pass

    @abstractmethod
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: UserPreference
    ) -> bool:
        """사용자 선호도 업데이트"""
        pass

    @abstractmethod
    async def learn_from_interaction(
        self,
        user_id: str,
        fragrance_id: str,
        interaction_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """상호작용으로부터 학습"""
        pass

    @abstractmethod
    async def get_user_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """사용자 향수 이력 조회"""
        pass


class FragranceDataProvider(ABC):
    """향수 데이터 제공 인터페이스"""

    @abstractmethod
    async def get_fragrance_profile(self, fragrance_id: str) -> Optional[FragranceProfile]:
        """향수 프로필 조회"""
        pass

    @abstractmethod
    async def search_fragrances(
        self,
        criteria: Dict[str, Any],
        limit: int = 100
    ) -> List[FragranceProfile]:
        """향수 검색"""
        pass

    @abstractmethod
    async def get_similar_fragrances(
        self,
        fragrance_id: str,
        similarity_threshold: float = 0.7,
        limit: int = 20
    ) -> List[FragranceProfile]:
        """유사한 향수 조회"""
        pass

    @abstractmethod
    async def get_trending_fragrances(
        self,
        time_period: str = "week",
        limit: int = 50
    ) -> List[FragranceProfile]:
        """트렌딩 향수 조회"""
        pass


class RecommendationStrategy(ABC):
    """추천 전략 인터페이스"""

    @abstractmethod
    async def generate_recommendations(
        self,
        context: RecommendationContext,
        candidates: List[FragranceProfile],
        user_preferences: Optional[UserPreference] = None
    ) -> List[Dict[str, Any]]:
        """추천 생성"""
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """전략 이름 반환"""
        pass

    @abstractmethod
    def get_strategy_description(self) -> str:
        """전략 설명 반환"""
        pass

    @abstractmethod
    def calculate_match_score(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        user_preferences: Optional[UserPreference] = None
    ) -> float:
        """매칭 점수 계산"""
        pass


class CacheManager(ABC):
    """캐시 관리 인터페이스"""

    @abstractmethod
    async def get_cached_recommendations(
        self,
        cache_key: str
    ) -> Optional[RecommendationResponse]:
        """캐시된 추천 결과 조회"""
        pass

    @abstractmethod
    async def cache_recommendations(
        self,
        cache_key: str,
        recommendations: RecommendationResponse,
        ttl: int = 3600
    ) -> bool:
        """추천 결과 캐싱"""
        pass

    @abstractmethod
    async def invalidate_user_cache(self, user_id: str) -> bool:
        """사용자 관련 캐시 무효화"""
        pass

    @abstractmethod
    def generate_cache_key(self, context: RecommendationContext) -> str:
        """캐시 키 생성"""
        pass


class AnalyticsCollector(ABC):
    """분석 데이터 수집 인터페이스"""

    @abstractmethod
    async def log_recommendation_request(
        self,
        context: RecommendationContext,
        response: RecommendationResponse
    ) -> bool:
        """추천 요청 로깅"""
        pass

    @abstractmethod
    async def log_user_interaction(
        self,
        user_id: str,
        fragrance_id: str,
        interaction_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """사용자 상호작용 로깅"""
        pass

    @abstractmethod
    async def get_recommendation_metrics(
        self,
        time_period: str = "day"
    ) -> Dict[str, Any]:
        """추천 메트릭 조회"""
        pass

    @abstractmethod
    async def get_user_engagement_metrics(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """사용자 참여도 메트릭 조회"""
        pass