"""
추천 전략 구현체들
다양한 추천 알고리즘을 제공합니다.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import logging
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

from .interfaces import RecommendationStrategy
from .types import (
    RecommendationContext,
    FragranceProfile,
    UserPreference,
    RecommendationType,
    Season,
    Mood,
    Occasion
)

logger = logging.getLogger(__name__)


class SimilarityStrategy(RecommendationStrategy):
    """유사성 기반 추천 전략"""

    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold

    async def generate_recommendations(
        self,
        context: RecommendationContext,
        candidates: List[FragranceProfile],
        user_preferences: Optional[UserPreference] = None
    ) -> List[Dict[str, Any]]:
        """유사성 기반 추천 생성"""
        recommendations = []

        for fragrance in candidates:
            score = self.calculate_match_score(fragrance, context, user_preferences)

            if score >= self.similarity_threshold:
                reasoning = self._generate_similarity_reasoning(fragrance, context, score)

                recommendations.append({
                    "fragrance": fragrance,
                    "score": score,
                    "reasoning": reasoning,
                    "confidence": min(score + 0.1, 1.0),
                    "recommendation_type": RecommendationType.SIMILAR
                })

        # 점수순 정렬
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:context.max_recommendations]

    def calculate_match_score(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        user_preferences: Optional[UserPreference] = None
    ) -> float:
        """유사성 매칭 점수 계산"""
        scores = []

        # 사용자 선호도 기반 점수
        if user_preferences:
            # 향수 계열 매칭
            family_score = user_preferences.fragrance_families.get(
                fragrance.fragrance_family, 0.5
            )
            scores.append(family_score * 0.3)

            # 노트 선호도 매칭
            note_scores = []
            for note in fragrance.notes:
                note_pref = user_preferences.notes_preference.get(note.name, 0.5)
                note_scores.append(note_pref * note.intensity)

            if note_scores:
                avg_note_score = np.mean(note_scores)
                scores.append(avg_note_score * 0.4)

            # 강도, 지속성, 확산성 선호도
            intensity_match = 1 - abs(fragrance.overall_intensity - user_preferences.intensity_preference)
            longevity_match = 1 - abs(fragrance.longevity - user_preferences.longevity_preference)
            projection_match = 1 - abs(fragrance.projection - user_preferences.projection_preference)

            scores.extend([intensity_match * 0.1, longevity_match * 0.1, projection_match * 0.1])

        # 품질 점수
        quality_score = fragrance.quality_score / 10.0  # 0-1 스케일로 변환
        scores.append(quality_score * 0.2)

        return np.mean(scores) if scores else 0.5

    def get_strategy_name(self) -> str:
        return "similarity_based"

    def get_strategy_description(self) -> str:
        return "사용자의 과거 선호도와 유사한 향수를 추천합니다."

    def _generate_similarity_reasoning(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        score: float
    ) -> str:
        """유사성 추천 이유 생성"""
        reasons = []

        if score > 0.8:
            reasons.append(f"{fragrance.name}은(는) 당신의 취향과 매우 잘 맞습니다")
        elif score > 0.6:
            reasons.append(f"{fragrance.name}은(는) 당신이 좋아할 만한 향수입니다")

        reasons.append(f"품질 점수: {fragrance.quality_score}/10")

        return ". ".join(reasons)


class SeasonalStrategy(RecommendationStrategy):
    """계절 기반 추천 전략"""

    def __init__(self):
        self.seasonal_weights = {
            Season.SPRING: {"fresh": 1.2, "floral": 1.3, "citrus": 1.1},
            Season.SUMMER: {"citrus": 1.4, "aquatic": 1.3, "fresh": 1.2},
            Season.AUTUMN: {"woody": 1.3, "spicy": 1.2, "oriental": 1.1},
            Season.WINTER: {"oriental": 1.4, "woody": 1.3, "amber": 1.2}
        }

    async def generate_recommendations(
        self,
        context: RecommendationContext,
        candidates: List[FragranceProfile],
        user_preferences: Optional[UserPreference] = None
    ) -> List[Dict[str, Any]]:
        """계절 기반 추천 생성"""
        recommendations = []
        current_season = self._get_current_season(context)

        for fragrance in candidates:
            score = self.calculate_match_score(fragrance, context, user_preferences)

            # 계절 가중치 적용
            seasonal_boost = self._get_seasonal_boost(fragrance, current_season)
            final_score = min(score * seasonal_boost, 1.0)

            if final_score >= 0.6:
                reasoning = self._generate_seasonal_reasoning(fragrance, current_season, final_score)

                recommendations.append({
                    "fragrance": fragrance,
                    "score": final_score,
                    "reasoning": reasoning,
                    "confidence": min(final_score + 0.05, 1.0),
                    "recommendation_type": RecommendationType.SEASONAL
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:context.max_recommendations]

    def calculate_match_score(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        user_preferences: Optional[UserPreference] = None
    ) -> float:
        """계절 매칭 점수 계산"""
        current_season = self._get_current_season(context)
        season_score = fragrance.get_season_score(current_season)

        # 날씨 조건 고려
        weather_score = 1.0
        if context.user_context.weather:
            weather = context.user_context.weather
            if weather.is_hot and fragrance.overall_intensity > 0.7:
                weather_score *= 0.8  # 더운 날씨에 강한 향수 페널티
            elif weather.is_humid and "aquatic" not in fragrance.fragrance_family.lower():
                weather_score *= 0.9  # 습한 날씨에 아쿠아틱이 아닌 향수 약간 페널티

        return season_score * weather_score

    def get_strategy_name(self) -> str:
        return "seasonal"

    def get_strategy_description(self) -> str:
        return "현재 계절과 날씨에 어울리는 향수를 추천합니다."

    def _get_current_season(self, context: RecommendationContext) -> Season:
        """현재 계절 결정"""
        if context.user_context.weather and context.user_context.weather.season:
            return context.user_context.weather.season

        # 날씨 정보가 없으면 현재 월 기준으로 계절 추정
        month = datetime.now().month
        if 3 <= month <= 5:
            return Season.SPRING
        elif 6 <= month <= 8:
            return Season.SUMMER
        elif 9 <= month <= 11:
            return Season.AUTUMN
        else:
            return Season.WINTER

    def _get_seasonal_boost(self, fragrance: FragranceProfile, season: Season) -> float:
        """계절별 가중치 반환"""
        family_lower = fragrance.fragrance_family.lower()
        seasonal_weights = self.seasonal_weights.get(season, {})

        for family, weight in seasonal_weights.items():
            if family in family_lower:
                return weight

        return 1.0  # 기본 가중치

    def _generate_seasonal_reasoning(
        self,
        fragrance: FragranceProfile,
        season: Season,
        score: float
    ) -> str:
        """계절 추천 이유 생성"""
        season_names = {
            Season.SPRING: "봄",
            Season.SUMMER: "여름",
            Season.AUTUMN: "가을",
            Season.WINTER: "겨울"
        }

        return f"{fragrance.name}은(는) {season_names[season]}에 특히 어울리는 향수입니다 (적합도: {score:.1%})"


class MoodBasedStrategy(RecommendationStrategy):
    """무드 기반 추천 전략"""

    def __init__(self):
        self.mood_profiles = {
            Mood.ROMANTIC: {"floral": 1.3, "sweet": 1.2, "soft": 1.1},
            Mood.ENERGETIC: {"citrus": 1.4, "fresh": 1.2, "spicy": 1.1},
            Mood.CALM: {"aquatic": 1.3, "green": 1.2, "soft": 1.1},
            Mood.CONFIDENT: {"woody": 1.3, "amber": 1.2, "strong": 1.1},
            Mood.SOPHISTICATED: {"chypre": 1.4, "oriental": 1.2, "complex": 1.1}
        }

    async def generate_recommendations(
        self,
        context: RecommendationContext,
        candidates: List[FragranceProfile],
        user_preferences: Optional[UserPreference] = None
    ) -> List[Dict[str, Any]]:
        """무드 기반 추천 생성"""
        if not context.user_context.mood:
            return []

        recommendations = []
        target_mood = context.user_context.mood

        for fragrance in candidates:
            score = self.calculate_match_score(fragrance, context, user_preferences)

            if score >= 0.6:
                reasoning = self._generate_mood_reasoning(fragrance, target_mood, score)

                recommendations.append({
                    "fragrance": fragrance,
                    "score": score,
                    "reasoning": reasoning,
                    "confidence": min(score + 0.1, 1.0),
                    "recommendation_type": RecommendationType.MOOD_BASED
                })

        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return recommendations[:context.max_recommendations]

    def calculate_match_score(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        user_preferences: Optional[UserPreference] = None
    ) -> float:
        """무드 매칭 점수 계산"""
        if not context.user_context.mood:
            return 0.5

        mood_score = fragrance.get_mood_score(context.user_context.mood)

        # 향수 특성과 무드 매칭
        mood_boost = self._get_mood_boost(fragrance, context.user_context.mood)

        return min(mood_score * mood_boost, 1.0)

    def get_strategy_name(self) -> str:
        return "mood_based"

    def get_strategy_description(self) -> str:
        return "현재 기분과 분위기에 맞는 향수를 추천합니다."

    def _get_mood_boost(self, fragrance: FragranceProfile, mood: Mood) -> float:
        """무드별 가중치 반환"""
        family_lower = fragrance.fragrance_family.lower()
        mood_weights = self.mood_profiles.get(mood, {})

        for characteristic, weight in mood_weights.items():
            if characteristic in family_lower:
                return weight

        return 1.0

    def _generate_mood_reasoning(
        self,
        fragrance: FragranceProfile,
        mood: Mood,
        score: float
    ) -> str:
        """무드 추천 이유 생성"""
        mood_names = {
            Mood.ROMANTIC: "로맨틱한",
            Mood.ENERGETIC: "활기찬",
            Mood.CALM: "차분한",
            Mood.CONFIDENT: "자신감 있는",
            Mood.SOPHISTICATED: "세련된"
        }

        mood_name = mood_names.get(mood, str(mood.value))
        return f"{fragrance.name}은(는) {mood_name} 기분에 완벽하게 어울립니다 (매칭도: {score:.1%})"


class InnovativeStrategy(RecommendationStrategy):
    """혁신적 조합 추천 전략"""

    def __init__(self, novelty_threshold: float = 0.3):
        self.novelty_threshold = novelty_threshold

    async def generate_recommendations(
        self,
        context: RecommendationContext,
        candidates: List[FragranceProfile],
        user_preferences: Optional[UserPreference] = None
    ) -> List[Dict[str, Any]]:
        """혁신적 추천 생성"""
        recommendations = []

        for fragrance in candidates:
            # 새로움 점수가 높은 향수만 선별
            if fragrance.uniqueness_score >= self.novelty_threshold:
                score = self.calculate_match_score(fragrance, context, user_preferences)

                if score >= 0.5:
                    reasoning = self._generate_innovative_reasoning(fragrance, score)

                    recommendations.append({
                        "fragrance": fragrance,
                        "score": score,
                        "reasoning": reasoning,
                        "confidence": min(score * 0.9, 1.0),  # 혁신적 추천은 약간 낮은 confidence
                        "recommendation_type": RecommendationType.INNOVATIVE
                    })

        recommendations.sort(key=lambda x: x["fragrance"].uniqueness_score, reverse=True)
        return recommendations[:context.max_recommendations]

    def calculate_match_score(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        user_preferences: Optional[UserPreference] = None
    ) -> float:
        """혁신적 매칭 점수 계산"""
        # 기본 품질 점수
        quality_factor = fragrance.quality_score / 10.0

        # 독창성 점수
        uniqueness_factor = fragrance.uniqueness_score

        # 사용자의 새로움 추구 성향 고려
        novelty_factor = 1.0
        if user_preferences:
            novelty_factor = user_preferences.novelty_seeking

        return (quality_factor * 0.4 + uniqueness_factor * 0.4 + novelty_factor * 0.2)

    def get_strategy_name(self) -> str:
        return "innovative"

    def get_strategy_description(self) -> str:
        return "독특하고 혁신적인 향수를 추천합니다."

    def _generate_innovative_reasoning(
        self,
        fragrance: FragranceProfile,
        score: float
    ) -> str:
        """혁신적 추천 이유 생성"""
        return (f"{fragrance.name}은(는) 독창적이고 혁신적인 조합으로 "
                f"새로운 향수 경험을 선사할 것입니다 (독창성: {fragrance.uniqueness_score:.1%})")


class SafeChoiceStrategy(RecommendationStrategy):
    """안전한 선택 추천 전략"""

    def __init__(self, popularity_threshold: float = 0.7):
        self.popularity_threshold = popularity_threshold

    async def generate_recommendations(
        self,
        context: RecommendationContext,
        candidates: List[FragranceProfile],
        user_preferences: Optional[UserPreference] = None
    ) -> List[Dict[str, Any]]:
        """안전한 선택 추천 생성"""
        recommendations = []

        for fragrance in candidates:
            if fragrance.popularity_score >= self.popularity_threshold:
                score = self.calculate_match_score(fragrance, context, user_preferences)

                if score >= 0.6:
                    reasoning = self._generate_safe_choice_reasoning(fragrance, score)

                    recommendations.append({
                        "fragrance": fragrance,
                        "score": score,
                        "reasoning": reasoning,
                        "confidence": min(score + 0.15, 1.0),  # 안전한 선택은 높은 confidence
                        "recommendation_type": RecommendationType.SAFE_CHOICE
                    })

        recommendations.sort(key=lambda x: x["fragrance"].popularity_score, reverse=True)
        return recommendations[:context.max_recommendations]

    def calculate_match_score(
        self,
        fragrance: FragranceProfile,
        context: RecommendationContext,
        user_preferences: Optional[UserPreference] = None
    ) -> float:
        """안전한 선택 매칭 점수 계산"""
        popularity_factor = fragrance.popularity_score
        quality_factor = fragrance.quality_score / 10.0

        # 대중적 선호도와 품질의 조합
        return (popularity_factor * 0.6 + quality_factor * 0.4)

    def get_strategy_name(self) -> str:
        return "safe_choice"

    def get_strategy_description(self) -> str:
        return "대중적으로 인기 있고 검증된 향수를 추천합니다."

    def _generate_safe_choice_reasoning(
        self,
        fragrance: FragranceProfile,
        score: float
    ) -> str:
        """안전한 선택 추천 이유 생성"""
        return (f"{fragrance.name}은(는) 많은 사람들이 사랑하는 검증된 향수로 "
                f"실패할 확률이 낮습니다 (인기도: {fragrance.popularity_score:.1%})")