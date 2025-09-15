"""
실시간 조향 추천 엔진
마스터 조향사급 실시간 향수 추천 시스템
사용자의 선호도, 상황, 날씨 등을 종합적으로 고려한 지능형 추천
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import hashlib
from collections import defaultdict, deque
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

from ..knowledge.master_perfumer_principles import MasterPerfumerKnowledge
from .advanced_blending_ai import AdvancedBlendingAI
from .quality_analyzer import FragranceQualityAnalyzer
from .compatibility_matrix import FragranceCompatibilityMatrix
from ..core.config import settings

logger = logging.getLogger(__name__)

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
    IMPLICIT = "implicit"    # 암시적 (행동 패턴)
    EXPLICIT = "explicit"    # 명시적 (직접 평가)
    CONTEXTUAL = "contextual" # 상황적 (날씨, 시간 등)

@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    demographic: Dict[str, Any] = field(default_factory=dict)  # 나이, 성별 등
    preferences: Dict[str, float] = field(default_factory=dict)  # 향료별 선호도
    dislikes: List[str] = field(default_factory=list)          # 비선호 향료
    favorite_styles: List[str] = field(default_factory=list)   # 선호 스타일
    usage_history: List[Dict[str, Any]] = field(default_factory=list)  # 사용 이력
    context_preferences: Dict[str, Any] = field(default_factory=dict)  # 상황별 선호도
    personality_traits: List[str] = field(default_factory=list)        # 성격 특성
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class RecommendationRequest:
    """추천 요청"""
    user_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)  # 날씨, 시간, 장소, 상황 등
    preferences_override: Dict[str, float] = field(default_factory=dict)  # 일시적 선호도 변경
    recommendation_type: Optional[RecommendationType] = None
    num_recommendations: int = 5
    include_explanation: bool = True
    exclude_ingredients: List[str] = field(default_factory=list)
    budget_range: Optional[Tuple[float, float]] = None
    complexity_preference: Optional[str] = None  # simple, moderate, complex

@dataclass
class RecommendationResult:
    """추천 결과"""
    fragrance_formula: Dict[str, Any]
    confidence_score: float
    match_score: float
    recommendation_type: RecommendationType
    explanation: str
    key_features: List[str]
    expected_experience: Dict[str, Any]
    alternative_adjustments: List[Dict[str, Any]]
    perfumer_inspiration: str
    estimated_cost: Optional[float] = None

@dataclass
class ContextualFactors:
    """상황적 요인들"""
    weather: Optional[Dict[str, Any]] = None        # 날씨 정보
    season: Optional[str] = None                    # 계절
    time_of_day: Optional[str] = None              # 시간대
    occasion: Optional[str] = None                  # 상황/행사
    mood: Optional[str] = None                      # 현재 기분
    location_type: Optional[str] = None            # 장소 유형
    social_context: Optional[str] = None           # 사회적 상황


class UserModelingEngine:
    """사용자 모델링 엔진"""

    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.preference_history = defaultdict(deque)
        self.interaction_patterns = defaultdict(dict)
        self.scaler = StandardScaler()

    def update_user_profile(
        self,
        user_id: str,
        interaction_data: Dict[str, Any],
        preference_type: UserPreferenceType = UserPreferenceType.IMPLICIT
    ) -> UserProfile:
        """사용자 프로필 업데이트"""

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)

        profile = self.user_profiles[user_id]

        # 상호작용 데이터 처리
        if preference_type == UserPreferenceType.IMPLICIT:
            self._process_implicit_feedback(profile, interaction_data)
        elif preference_type == UserPreferenceType.EXPLICIT:
            self._process_explicit_feedback(profile, interaction_data)
        else:  # CONTEXTUAL
            self._process_contextual_feedback(profile, interaction_data)

        # 사용이력 업데이트
        profile.usage_history.append({
            "timestamp": datetime.now(),
            "interaction": interaction_data,
            "type": preference_type.value
        })

        # 이력이 너무 길어지면 오래된 것 제거
        if len(profile.usage_history) > 1000:
            profile.usage_history = profile.usage_history[-500:]

        profile.last_updated = datetime.now()
        return profile

    def _process_implicit_feedback(self, profile: UserProfile, data: Dict[str, Any]):
        """암시적 피드백 처리"""

        # 행동 패턴 분석
        if "view_duration" in data:
            # 오래 본 향수는 관심도가 높다고 가정
            if data["view_duration"] > 30:  # 30초 이상
                ingredients = data.get("ingredients", [])
                for ingredient in ingredients:
                    current_pref = profile.preferences.get(ingredient, 0.5)
                    profile.preferences[ingredient] = min(1.0, current_pref + 0.1)

        if "clicked_details" in data and data["clicked_details"]:
            # 세부정보 클릭은 관심의 지표
            ingredients = data.get("ingredients", [])
            for ingredient in ingredients:
                current_pref = profile.preferences.get(ingredient, 0.5)
                profile.preferences[ingredient] = min(1.0, current_pref + 0.05)

        if "shared" in data and data["shared"]:
            # 공유는 강한 선호의 지표
            ingredients = data.get("ingredients", [])
            for ingredient in ingredients:
                current_pref = profile.preferences.get(ingredient, 0.5)
                profile.preferences[ingredient] = min(1.0, current_pref + 0.2)

    def _process_explicit_feedback(self, profile: UserProfile, data: Dict[str, Any]):
        """명시적 피드백 처리"""

        if "rating" in data and "ingredients" in data:
            rating = data["rating"]  # 1-5 점수
            normalized_rating = (rating - 1) / 4  # 0-1로 정규화

            ingredients = data["ingredients"]
            for ingredient in ingredients:
                # 기존 선호도와 새 평점의 가중 평균
                current_pref = profile.preferences.get(ingredient, 0.5)
                profile.preferences[ingredient] = current_pref * 0.7 + normalized_rating * 0.3

        if "liked_ingredients" in data:
            for ingredient in data["liked_ingredients"]:
                profile.preferences[ingredient] = min(1.0, profile.preferences.get(ingredient, 0.5) + 0.3)

        if "disliked_ingredients" in data:
            for ingredient in data["disliked_ingredients"]:
                profile.preferences[ingredient] = max(0.0, profile.preferences.get(ingredient, 0.5) - 0.3)
                if ingredient not in profile.dislikes:
                    profile.dislikes.append(ingredient)

    def _process_contextual_feedback(self, profile: UserProfile, data: Dict[str, Any]):
        """상황적 피드백 처리"""

        context = data.get("context", {})
        ingredients = data.get("ingredients", [])
        preference_strength = data.get("preference_strength", 0.5)

        # 상황별 선호도 업데이트
        for key, value in context.items():
            if key not in profile.context_preferences:
                profile.context_preferences[key] = {}

            context_key = f"{key}:{value}"
            if context_key not in profile.context_preferences[key]:
                profile.context_preferences[key][context_key] = {}

            for ingredient in ingredients:
                current_pref = profile.context_preferences[key][context_key].get(ingredient, 0.5)
                profile.context_preferences[key][context_key][ingredient] = \
                    current_pref * 0.8 + preference_strength * 0.2

    def predict_preferences(
        self,
        user_id: str,
        ingredients: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """향료별 선호도 예측"""

        if user_id not in self.user_profiles:
            # 새 사용자는 중립적 선호도
            return {ingredient: 0.5 for ingredient in ingredients}

        profile = self.user_profiles[user_id]
        predictions = {}

        for ingredient in ingredients:
            base_preference = profile.preferences.get(ingredient, 0.5)

            # 상황적 조정
            contextual_adjustment = 0.0
            if context:
                for context_type, context_value in context.items():
                    context_key = f"{context_type}:{context_value}"
                    if context_type in profile.context_preferences:
                        if context_key in profile.context_preferences[context_type]:
                            contextual_pref = profile.context_preferences[context_type][context_key].get(ingredient, 0.5)
                            contextual_adjustment += (contextual_pref - 0.5) * 0.2

            # 비선호 향료 페널티
            dislike_penalty = -0.3 if ingredient in profile.dislikes else 0.0

            final_preference = max(0.0, min(1.0, base_preference + contextual_adjustment + dislike_penalty))
            predictions[ingredient] = final_preference

        return predictions

    def get_user_clusters(self) -> Dict[str, List[str]]:
        """사용자 클러스터링"""

        if len(self.user_profiles) < 5:
            return {"all_users": list(self.user_profiles.keys())}

        # 사용자별 선호도 벡터 생성
        all_ingredients = set()
        for profile in self.user_profiles.values():
            all_ingredients.update(profile.preferences.keys())

        all_ingredients = list(all_ingredients)
        user_vectors = []
        user_ids = list(self.user_profiles.keys())

        for user_id in user_ids:
            profile = self.user_profiles[user_id]
            vector = [profile.preferences.get(ing, 0.5) for ing in all_ingredients]
            user_vectors.append(vector)

        # K-means 클러스터링
        n_clusters = min(5, len(user_ids) // 2)
        if n_clusters < 2:
            return {"all_users": user_ids}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(user_vectors)

        # 클러스터 결과 정리
        clusters = defaultdict(list)
        for user_id, label in zip(user_ids, cluster_labels):
            clusters[f"cluster_{label}"].append(user_id)

        return dict(clusters)


class ContextualReasoningEngine:
    """상황적 추론 엔진"""

    def __init__(self):
        self.seasonal_preferences = self._initialize_seasonal_preferences()
        self.mood_fragrance_mapping = self._initialize_mood_mapping()
        self.occasion_guidelines = self._initialize_occasion_guidelines()
        self.weather_adjustments = self._initialize_weather_adjustments()

    def _initialize_seasonal_preferences(self) -> Dict[str, Dict[str, float]]:
        """계절별 선호도 매핑"""
        return {
            "spring": {
                "시트러스": 0.9, "플로럴": 0.8, "그린": 0.8, "프레시": 0.9,
                "라이트": 0.8, "오리엔탈": 0.4, "헤비": 0.3
            },
            "summer": {
                "시트러스": 1.0, "마린": 0.9, "아쿠아틱": 0.9, "프레시": 1.0,
                "라이트": 0.9, "오리엔탈": 0.2, "헤비": 0.2, "스파이시": 0.3
            },
            "autumn": {
                "우디": 0.8, "스파이시": 0.8, "오리엔탈": 0.7, "앰버": 0.8,
                "따뜻함": 0.9, "시트러스": 0.6, "프레시": 0.5
            },
            "winter": {
                "오리엔탈": 0.9, "우디": 0.8, "바닐라": 0.9, "스파이시": 0.8,
                "따뜻함": 1.0, "헤비": 0.8, "시트러스": 0.4, "라이트": 0.3
            }
        }

    def _initialize_mood_mapping(self) -> Dict[str, Dict[str, float]]:
        """기분별 향수 매핑"""
        return {
            "energetic": {
                "시트러스": 0.9, "민트": 0.8, "스파클링": 0.8, "프레시": 0.9
            },
            "romantic": {
                "로즈": 0.9, "자스민": 0.8, "바닐라": 0.7, "플로럴": 0.9
            },
            "confident": {
                "우디": 0.8, "스파이시": 0.8, "파워풀": 0.9, "인텐스": 0.8
            },
            "calm": {
                "라벤더": 0.9, "샌달우드": 0.8, "소프트": 0.8, "젠틀": 0.9
            },
            "mysterious": {
                "인센스": 0.9, "우드": 0.8, "다크": 0.8, "인트리깅": 0.9
            },
            "playful": {
                "프루티": 0.9, "스위트": 0.8, "라이트": 0.8, "펀": 0.8
            }
        }

    def _initialize_occasion_guidelines(self) -> Dict[str, Dict[str, Any]]:
        """상황별 가이드라인"""
        return {
            "business": {
                "preferred_characteristics": ["professional", "subtle", "clean"],
                "avoid_characteristics": ["overwhelming", "too_sweet", "too_bold"],
                "intensity_range": (0.3, 0.6),
                "projection_range": (0.3, 0.6)
            },
            "date": {
                "preferred_characteristics": ["romantic", "attractive", "memorable"],
                "avoid_characteristics": ["overwhelming", "too_casual"],
                "intensity_range": (0.5, 0.8),
                "projection_range": (0.4, 0.7)
            },
            "casual": {
                "preferred_characteristics": ["comfortable", "easy-going", "fresh"],
                "avoid_characteristics": ["too_formal", "overwhelming"],
                "intensity_range": (0.4, 0.7),
                "projection_range": (0.3, 0.6)
            },
            "evening": {
                "preferred_characteristics": ["sophisticated", "luxurious", "memorable"],
                "avoid_characteristics": ["too_fresh", "too_light"],
                "intensity_range": (0.6, 0.9),
                "projection_range": (0.5, 0.8)
            },
            "special_event": {
                "preferred_characteristics": ["unique", "memorable", "sophisticated"],
                "avoid_characteristics": ["too_common", "too_safe"],
                "intensity_range": (0.6, 1.0),
                "projection_range": (0.6, 0.9)
            }
        }

    def _initialize_weather_adjustments(self) -> Dict[str, Dict[str, float]]:
        """날씨별 조정"""
        return {
            "hot": {
                "시트러스": 0.3, "프레시": 0.3, "라이트": 0.2, "오리엔탈": -0.4
            },
            "cold": {
                "오리엔탈": 0.3, "우디": 0.2, "스파이시": 0.2, "시트러스": -0.2
            },
            "humid": {
                "마린": 0.2, "아쿠아틱": 0.2, "프레시": 0.2, "헤비": -0.3
            },
            "dry": {
                "크리미": 0.2, "리치": 0.1, "프레시": -0.1
            },
            "rainy": {
                "코지": 0.2, "따뜻함": 0.2, "프레시": -0.1
            }
        }

    def analyze_context(self, context: Dict[str, Any]) -> Dict[str, float]:
        """상황 분석 및 선호도 조정값 반환"""

        adjustments = defaultdict(float)

        # 계절 고려
        season = context.get("season")
        if season and season in self.seasonal_preferences:
            for char, weight in self.seasonal_preferences[season].items():
                adjustments[char] += (weight - 0.5) * 0.3

        # 기분 고려
        mood = context.get("mood")
        if mood and mood in self.mood_fragrance_mapping:
            for char, weight in self.mood_fragrance_mapping[mood].items():
                adjustments[char] += (weight - 0.5) * 0.4

        # 날씨 고려
        weather = context.get("weather", {})
        temperature = weather.get("temperature")
        humidity = weather.get("humidity")

        if temperature is not None:
            if temperature > 25:  # 더운 날씨
                for char, adj in self.weather_adjustments["hot"].items():
                    adjustments[char] += adj
            elif temperature < 10:  # 추운 날씨
                for char, adj in self.weather_adjustments["cold"].items():
                    adjustments[char] += adj

        if humidity is not None and humidity > 70:  # 습한 날씨
            for char, adj in self.weather_adjustments["humid"].items():
                adjustments[char] += adj

        # 상황/행사 고려
        occasion = context.get("occasion")
        if occasion and occasion in self.occasion_guidelines:
            guidelines = self.occasion_guidelines[occasion]
            preferred = guidelines["preferred_characteristics"]
            avoided = guidelines["avoid_characteristics"]

            for char in preferred:
                adjustments[char] += 0.2
            for char in avoided:
                adjustments[char] -= 0.3

        return dict(adjustments)

    def get_intensity_constraints(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """상황에 따른 강도 제약 조건"""

        occasion = context.get("occasion")
        if occasion and occasion in self.occasion_guidelines:
            return self.occasion_guidelines[occasion]["intensity_range"]

        # 기본값
        return (0.3, 0.8)

    def get_projection_constraints(self, context: Dict[str, Any]) -> Tuple[float, float]:
        """상황에 따른 프로젝션 제약 조건"""

        occasion = context.get("occasion")
        if occasion and occasion in self.occasion_guidelines:
            return self.occasion_guidelines[occasion]["projection_range"]

        # 기본값
        return (0.3, 0.7)


class RealtimeRecommendationEngine:
    """실시간 조향 추천 엔진 메인 클래스"""

    def __init__(self):
        # 핵심 AI 컴포넌트들 초기화
        self.perfumer_knowledge = MasterPerfumerKnowledge()
        self.blending_ai = AdvancedBlendingAI()
        self.quality_analyzer = FragranceQualityAnalyzer()
        self.compatibility_matrix = FragranceCompatibilityMatrix()

        # 추천 시스템 컴포넌트들
        self.user_modeling = UserModelingEngine()
        self.contextual_reasoning = ContextualReasoningEngine()

        # 추천 캐시
        self.recommendation_cache = {}
        self.cache_expire_time = 3600  # 1시간

        # 백그라운드 작업을 위한 스레드풀
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 추천 성능 모니터링
        self.recommendation_metrics = defaultdict(list)

        logger.info("Realtime Recommendation Engine initialized")

    async def get_recommendations(
        self,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """메인 추천 함수"""

        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(request)
            if cache_key in self.recommendation_cache:
                cached_result = self.recommendation_cache[cache_key]
                if datetime.now() - cached_result["timestamp"] < timedelta(seconds=self.cache_expire_time):
                    logger.info("Returning cached recommendations")
                    return cached_result["recommendations"]

            # 사용자 프로필 로드
            user_profile = self.user_modeling.user_profiles.get(request.user_id) if request.user_id else None

            # 상황적 분석
            contextual_adjustments = self.contextual_reasoning.analyze_context(request.context)
            intensity_constraints = self.contextual_reasoning.get_intensity_constraints(request.context)
            projection_constraints = self.contextual_reasoning.get_projection_constraints(request.context)

            # 추천 전략 결정
            recommendation_strategies = self._determine_strategies(request, user_profile)

            # 다중 전략으로 추천 생성
            recommendations = []
            for strategy in recommendation_strategies:
                strategy_recommendations = await self._generate_strategy_recommendations(
                    strategy, request, user_profile, contextual_adjustments,
                    intensity_constraints, projection_constraints
                )
                recommendations.extend(strategy_recommendations)

            # 중복 제거 및 순위 정렬
            recommendations = self._deduplicate_and_rank(recommendations, request.num_recommendations)

            # 후처리 (설명 개선, 대안 제안 등)
            recommendations = self._post_process_recommendations(recommendations, request, user_profile)

            # 캐시 저장
            self.recommendation_cache[cache_key] = {
                "recommendations": recommendations,
                "timestamp": datetime.now()
            }

            # 성능 메트릭 기록
            self._record_recommendation_metrics(request, recommendations)

            return recommendations

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            # 기본 추천 반환
            return await self._generate_fallback_recommendations(request)

    def _generate_cache_key(self, request: RecommendationRequest) -> str:
        """캐시 키 생성"""
        key_data = {
            "user_id": request.user_id,
            "context": request.context,
            "num_recommendations": request.num_recommendations,
            "recommendation_type": request.recommendation_type.value if request.recommendation_type else None,
            "exclude_ingredients": sorted(request.exclude_ingredients),
            "complexity_preference": request.complexity_preference
        }

        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _determine_strategies(
        self,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile]
    ) -> List[RecommendationType]:
        """추천 전략 결정"""

        strategies = []

        # 명시적 추천 타입이 있다면 우선 사용
        if request.recommendation_type:
            strategies.append(request.recommendation_type)
        else:
            # 사용자 프로필과 상황에 따른 전략 선택
            if user_profile and len(user_profile.usage_history) > 10:
                # 경험 있는 사용자
                strategies.extend([
                    RecommendationType.SIMILAR,
                    RecommendationType.COMPLEMENTARY,
                    RecommendationType.INNOVATIVE
                ])
            elif user_profile and len(user_profile.usage_history) > 2:
                # 일반 사용자
                strategies.extend([
                    RecommendationType.SIMILAR,
                    RecommendationType.SAFE_CHOICE,
                    RecommendationType.SEASONAL
                ])
            else:
                # 새 사용자
                strategies.extend([
                    RecommendationType.SAFE_CHOICE,
                    RecommendationType.SEASONAL,
                    RecommendationType.MASTERPIECE
                ])

        # 상황 기반 추가 전략
        if "occasion" in request.context:
            strategies.append(RecommendationType.OCCASION_BASED)

        if "mood" in request.context:
            strategies.append(RecommendationType.MOOD_BASED)

        return list(set(strategies))  # 중복 제거

    async def _generate_strategy_recommendations(
        self,
        strategy: RecommendationType,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile],
        contextual_adjustments: Dict[str, float],
        intensity_constraints: Tuple[float, float],
        projection_constraints: Tuple[float, float]
    ) -> List[RecommendationResult]:
        """전략별 추천 생성"""

        if strategy == RecommendationType.SIMILAR:
            return await self._generate_similar_recommendations(
                request, user_profile, contextual_adjustments
            )

        elif strategy == RecommendationType.COMPLEMENTARY:
            return await self._generate_complementary_recommendations(
                request, user_profile, contextual_adjustments
            )

        elif strategy == RecommendationType.SEASONAL:
            return await self._generate_seasonal_recommendations(
                request, contextual_adjustments, intensity_constraints
            )

        elif strategy == RecommendationType.MOOD_BASED:
            return await self._generate_mood_based_recommendations(
                request, contextual_adjustments, intensity_constraints
            )

        elif strategy == RecommendationType.OCCASION_BASED:
            return await self._generate_occasion_based_recommendations(
                request, contextual_adjustments, intensity_constraints, projection_constraints
            )

        elif strategy == RecommendationType.INNOVATIVE:
            return await self._generate_innovative_recommendations(
                request, user_profile, contextual_adjustments
            )

        elif strategy == RecommendationType.SAFE_CHOICE:
            return await self._generate_safe_recommendations(
                request, contextual_adjustments
            )

        elif strategy == RecommendationType.MASTERPIECE:
            return await self._generate_masterpiece_recommendations(
                request, contextual_adjustments
            )

        else:
            return []

    async def _generate_similar_recommendations(
        self,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile],
        contextual_adjustments: Dict[str, float]
    ) -> List[RecommendationResult]:
        """유사한 향수 추천"""

        if not user_profile or not user_profile.preferences:
            return []

        # 사용자가 선호하는 향료들 기반으로 조합 생성
        preferred_ingredients = [
            ingredient for ingredient, score in user_profile.preferences.items()
            if score > 0.6 and ingredient not in request.exclude_ingredients
        ]

        if len(preferred_ingredients) < 3:
            return []

        recommendations = []

        # 여러 조합 시도
        for _ in range(3):
            # 선호 향료 중 3-5개 선택
            selected_ingredients = random.sample(
                preferred_ingredients,
                min(len(preferred_ingredients), random.randint(3, 5))
            )

            # 마스터 조향사 공식 생성
            formula = self.perfumer_knowledge.generate_perfumer_formula(
                style="modern",
                target_mood=request.context.get("mood", "sophisticated"),
                preferred_notes=selected_ingredients
            )

            # 품질 평가
            quality_assessment = self.quality_analyzer.analyze_fragrance_quality(
                selected_ingredients
            )

            if quality_assessment.overall_score > 0.6:
                recommendation = RecommendationResult(
                    fragrance_formula=formula,
                    confidence_score=min(0.9, quality_assessment.overall_score + 0.1),
                    match_score=self._calculate_user_match_score(selected_ingredients, user_profile),
                    recommendation_type=RecommendationType.SIMILAR,
                    explanation=f"당신의 선호도를 바탕으로 {formula['perfumer_inspiration']} 스타일의 조향을 제안합니다.",
                    key_features=selected_ingredients[:3],
                    expected_experience=self._predict_experience(selected_ingredients),
                    alternative_adjustments=[],
                    perfumer_inspiration=formula['perfumer_inspiration']
                )
                recommendations.append(recommendation)

        return recommendations

    async def _generate_seasonal_recommendations(
        self,
        request: RecommendationRequest,
        contextual_adjustments: Dict[str, float],
        intensity_constraints: Tuple[float, float]
    ) -> List[RecommendationResult]:
        """계절 맞춤 추천"""

        season = request.context.get("season", self._detect_current_season())

        seasonal_ingredients = {
            "spring": ["베르가못", "라벤더", "프리지아", "그린 리프"],
            "summer": ["시트러스", "민트", "마린 노트", "쿠쿰버"],
            "autumn": ["시더우드", "패촐리", "베티버", "스파이스"],
            "winter": ["바닐라", "앰버", "인센스", "우드"]
        }

        base_ingredients = seasonal_ingredients.get(season, seasonal_ingredients["spring"])

        recommendations = []

        # 계절별 마스터 조향 공식들
        for i in range(2):
            ingredients = base_ingredients.copy()

            # 약간의 변화를 위해 1-2개 다른 향료 추가
            if i > 0:
                all_ingredients = [ing for ing_list in seasonal_ingredients.values() for ing in ing_list]
                additional = random.choice([ing for ing in all_ingredients if ing not in ingredients])
                ingredients.append(additional)

            formula = self.perfumer_knowledge.generate_perfumer_formula(
                style="seasonal",
                target_mood=f"{season}_appropriate",
                preferred_notes=ingredients
            )

            quality_assessment = self.quality_analyzer.analyze_fragrance_quality(ingredients)

            recommendation = RecommendationResult(
                fragrance_formula=formula,
                confidence_score=0.8,
                match_score=0.75 + (quality_assessment.overall_score * 0.25),
                recommendation_type=RecommendationType.SEASONAL,
                explanation=f"{season} 계절에 완벽한 {formula['perfumer_inspiration']} 스타일의 조향입니다.",
                key_features=ingredients[:3],
                expected_experience=self._predict_experience(ingredients),
                alternative_adjustments=self._generate_seasonal_adjustments(season),
                perfumer_inspiration=formula['perfumer_inspiration']
            )
            recommendations.append(recommendation)

        return recommendations

    async def _generate_mood_based_recommendations(
        self,
        request: RecommendationRequest,
        contextual_adjustments: Dict[str, float],
        intensity_constraints: Tuple[float, float]
    ) -> List[RecommendationResult]:
        """기분 기반 추천"""

        mood = request.context.get("mood")
        if not mood:
            return []

        mood_ingredients = {
            "energetic": ["시트러스", "페퍼민트", "유칼립투스", "로즈마리"],
            "romantic": ["로즈", "자스민", "바닐라", "일랑일랑"],
            "confident": ["우드", "스파이시", "레더", "인센스"],
            "calm": ["라벤더", "샌달우드", "캐모마일", "화이트티"],
            "mysterious": ["인센스", "미르", "블랙퍼퍼", "다크우드"],
            "playful": ["프루티노트", "스위트피", "화이트머스크", "바닐라"]
        }

        if mood not in mood_ingredients:
            return []

        base_ingredients = mood_ingredients[mood]

        recommendations = []

        # 기분별 공식 생성
        formula = self.perfumer_knowledge.generate_perfumer_formula(
            style="mood_based",
            target_mood=mood,
            preferred_notes=base_ingredients
        )

        quality_assessment = self.quality_analyzer.analyze_fragrance_quality(base_ingredients)

        recommendation = RecommendationResult(
            fragrance_formula=formula,
            confidence_score=0.85,
            match_score=0.8,
            recommendation_type=RecommendationType.MOOD_BASED,
            explanation=f"{mood} 기분에 완벽하게 어울리는 {formula['perfumer_inspiration']} 스타일 조향입니다.",
            key_features=base_ingredients[:3],
            expected_experience=self._predict_mood_experience(mood, base_ingredients),
            alternative_adjustments=[],
            perfumer_inspiration=formula['perfumer_inspiration']
        )
        recommendations.append(recommendation)

        return recommendations

    async def _generate_occasion_based_recommendations(
        self,
        request: RecommendationRequest,
        contextual_adjustments: Dict[str, float],
        intensity_constraints: Tuple[float, float],
        projection_constraints: Tuple[float, float]
    ) -> List[RecommendationResult]:
        """상황 기반 추천"""

        occasion = request.context.get("occasion")
        if not occasion:
            return []

        occasion_ingredients = {
            "business": ["시더우드", "라벤더", "베르가못", "화이트머스크"],
            "date": ["로즈", "바닐라", "앰버", "자스민"],
            "casual": ["시트러스", "그린노트", "라이트머스크", "코튼"],
            "evening": ["인센스", "로즈", "우드", "앰버"],
            "special_event": ["사프란", "로즈", "우드", "앰버그리스"]
        }

        base_ingredients = occasion_ingredients.get(occasion, occasion_ingredients["casual"])

        formula = self.perfumer_knowledge.generate_perfumer_formula(
            style="occasion_based",
            target_mood=f"{occasion}_appropriate",
            preferred_notes=base_ingredients
        )

        quality_assessment = self.quality_analyzer.analyze_fragrance_quality(base_ingredients)

        recommendation = RecommendationResult(
            fragrance_formula=formula,
            confidence_score=0.85,
            match_score=0.8,
            recommendation_type=RecommendationType.OCCASION_BASED,
            explanation=f"{occasion} 상황에 최적화된 {formula['perfumer_inspiration']} 스타일의 세련된 조향입니다.",
            key_features=base_ingredients[:3],
            expected_experience=self._predict_occasion_experience(occasion, base_ingredients),
            alternative_adjustments=self._generate_occasion_adjustments(occasion),
            perfumer_inspiration=formula['perfumer_inspiration']
        )

        return [recommendation]

    async def _generate_innovative_recommendations(
        self,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile],
        contextual_adjustments: Dict[str, float]
    ) -> List[RecommendationResult]:
        """혁신적 조합 추천"""

        # 특이한 조합들 시도
        innovative_combinations = [
            ["아가우드", "시트러스", "머스크"],  # 동서양 융합
            ["바닐라", "인센스", "로즈"],      # 구르망-오리엔탈 융합
            ["민트", "우드", "앰버"],          # 쿨-웜 대비
            ["마린노트", "스파이스", "바닐라"], # 아쿠아틱-오리엔탈 융합
        ]

        recommendations = []

        for combination in innovative_combinations[:2]:
            # 사용자가 제외하지 않은 조합만 선택
            if not any(ing in request.exclude_ingredients for ing in combination):

                formula = self.perfumer_knowledge.generate_perfumer_formula(
                    style="innovative",
                    target_mood="adventurous",
                    preferred_notes=combination
                )

                quality_assessment = self.quality_analyzer.analyze_fragrance_quality(combination)

                if quality_assessment.overall_score > 0.5:  # 혁신적이라서 기준을 낮춤
                    recommendation = RecommendationResult(
                        fragrance_formula=formula,
                        confidence_score=0.7,  # 혁신적이므로 확신도는 낮음
                        match_score=0.6,
                        recommendation_type=RecommendationType.INNOVATIVE,
                        explanation=f"새로운 시도를 위한 혁신적 조합: {formula['perfumer_inspiration']} 스타일의 독특한 조향입니다.",
                        key_features=combination,
                        expected_experience=self._predict_innovative_experience(combination),
                        alternative_adjustments=self._generate_safety_adjustments(combination),
                        perfumer_inspiration=formula['perfumer_inspiration']
                    )
                    recommendations.append(recommendation)

        return recommendations

    async def _generate_safe_recommendations(
        self,
        request: RecommendationRequest,
        contextual_adjustments: Dict[str, float]
    ) -> List[RecommendationResult]:
        """안전한 선택 추천"""

        # 검증된 클래식 조합들
        safe_combinations = [
            ["베르가못", "라벤더", "바닐라"],    # 클래식 푸제르
            ["시트러스", "로즈", "머스크"],      # 클린 플로럴
            ["우드", "앰버", "바닐라"],         # 따뜻한 오리엔탈
        ]

        recommendations = []

        for combination in safe_combinations:
            if not any(ing in request.exclude_ingredients for ing in combination):

                formula = self.perfumer_knowledge.generate_perfumer_formula(
                    style="classic",
                    target_mood="comfortable",
                    preferred_notes=combination
                )

                quality_assessment = self.quality_analyzer.analyze_fragrance_quality(combination)

                recommendation = RecommendationResult(
                    fragrance_formula=formula,
                    confidence_score=0.9,
                    match_score=0.8,
                    recommendation_type=RecommendationType.SAFE_CHOICE,
                    explanation=f"검증된 클래식 조합: {formula['perfumer_inspiration']} 스타일의 안전하고 우아한 선택입니다.",
                    key_features=combination,
                    expected_experience=self._predict_safe_experience(combination),
                    alternative_adjustments=[],
                    perfumer_inspiration=formula['perfumer_inspiration']
                )
                recommendations.append(recommendation)

        return recommendations

    async def _generate_masterpiece_recommendations(
        self,
        request: RecommendationRequest,
        contextual_adjustments: Dict[str, float]
    ) -> List[RecommendationResult]:
        """마스터피스 스타일 추천"""

        # 유명한 마스터피스들을 모티프로 한 조합
        masterpiece_inspirations = [
            {
                "inspiration": "Chanel No.5",
                "ingredients": ["알데하이드", "일랑일랑", "로즈", "자스민", "바닐라"],
                "perfumer": "Ernest Beaux"
            },
            {
                "inspiration": "Shalimar",
                "ingredients": ["베르가못", "로즈", "자스민", "바닐라", "통카빈"],
                "perfumer": "Jacques Guerlain"
            }
        ]

        recommendations = []

        for inspiration_data in masterpiece_inspirations:
            ingredients = inspiration_data["ingredients"]

            if not any(ing in request.exclude_ingredients for ing in ingredients):

                formula = self.perfumer_knowledge.generate_perfumer_formula(
                    style="masterpiece",
                    target_mood="timeless",
                    preferred_notes=ingredients
                )

                quality_assessment = self.quality_analyzer.analyze_fragrance_quality(ingredients)

                recommendation = RecommendationResult(
                    fragrance_formula=formula,
                    confidence_score=0.95,
                    match_score=0.9,
                    recommendation_type=RecommendationType.MASTERPIECE,
                    explanation=f"{inspiration_data['inspiration']} 스타일에서 영감을 받은 {inspiration_data['perfumer']} 조향사의 마스터피스 해석입니다.",
                    key_features=ingredients[:3],
                    expected_experience=self._predict_masterpiece_experience(ingredients, inspiration_data),
                    alternative_adjustments=[],
                    perfumer_inspiration=inspiration_data['perfumer']
                )
                recommendations.append(recommendation)

        return recommendations

    def _calculate_user_match_score(self, ingredients: List[str], user_profile: UserProfile) -> float:
        """사용자 매칭 점수 계산"""
        if not user_profile or not user_profile.preferences:
            return 0.5

        preferences = [user_profile.preferences.get(ing, 0.5) for ing in ingredients]
        return sum(preferences) / len(preferences) if preferences else 0.5

    def _predict_experience(self, ingredients: List[str]) -> Dict[str, Any]:
        """경험 예측"""
        return {
            "opening": f"{ingredients[0]}의 신선한 첫인상",
            "development": f"{', '.join(ingredients[1:3])}이 조화롭게 발전",
            "drydown": f"{ingredients[-1]}의 깊이 있는 마무리",
            "longevity": "6-8시간",
            "projection": "moderate"
        }

    def _predict_mood_experience(self, mood: str, ingredients: List[str]) -> Dict[str, Any]:
        """기분별 경험 예측"""
        mood_descriptions = {
            "energetic": "활기찬 에너지를 선사하며 하루를 시작하는 완벽한 동반자",
            "romantic": "로맨틱한 분위기를 연출하며 특별한 순간을 더욱 아름답게",
            "confident": "자신감을 북돋우며 당당한 존재감을 드러내는 파워풀한 향",
            "calm": "평온한 휴식을 선사하며 마음의 안정을 찾게 해주는 향",
            "mysterious": "신비로운 매력을 발산하며 독특한 개성을 표현하는 향",
            "playful": "장난기 있는 즐거움을 선사하며 유쾌한 기분을 만드는 향"
        }

        base_experience = self._predict_experience(ingredients)
        base_experience["mood_effect"] = mood_descriptions.get(mood, "특별한 경험")

        return base_experience

    def _predict_occasion_experience(self, occasion: str, ingredients: List[str]) -> Dict[str, Any]:
        """상황별 경험 예측"""
        base_experience = self._predict_experience(ingredients)

        occasion_effects = {
            "business": "전문적이고 신뢰할 만한 인상",
            "date": "매력적이고 기억에 남는 존재감",
            "casual": "편안하고 자연스러운 일상의 향",
            "evening": "우아하고 세련된 저녁의 분위기",
            "special_event": "특별하고 독특한 순간의 완성"
        }

        base_experience["occasion_effect"] = occasion_effects.get(occasion, "적절한 상황 연출")
        return base_experience

    def _predict_innovative_experience(self, ingredients: List[str]) -> Dict[str, Any]:
        """혁신적 조합 경험 예측"""
        base_experience = self._predict_experience(ingredients)
        base_experience["innovation_factor"] = "예상치 못한 조화로운 만남"
        base_experience["uniqueness"] = "매우 높음"
        return base_experience

    def _predict_safe_experience(self, ingredients: List[str]) -> Dict[str, Any]:
        """안전한 선택 경험 예측"""
        base_experience = self._predict_experience(ingredients)
        base_experience["reliability"] = "검증된 조화"
        base_experience["wearability"] = "매우 높음"
        return base_experience

    def _predict_masterpiece_experience(self, ingredients: List[str], inspiration_data: Dict[str, Any]) -> Dict[str, Any]:
        """마스터피스 경험 예측"""
        base_experience = self._predict_experience(ingredients)
        base_experience["heritage"] = f"{inspiration_data['inspiration']}의 DNA를 계승"
        base_experience["quality_level"] = "마스터피스급"
        return base_experience

    def _generate_seasonal_adjustments(self, season: str) -> List[Dict[str, Any]]:
        """계절별 조정 제안"""
        adjustments = {
            "spring": [
                {"adjustment": "시트러스 비율을 10% 증가", "reason": "봄의 상쾌함 강화"},
                {"adjustment": "플로럴 노트 추가", "reason": "봄꽃의 계절감 반영"}
            ],
            "summer": [
                {"adjustment": "마린 노트 추가", "reason": "여름의 시원함 표현"},
                {"adjustment": "전체 농도를 80%로 조정", "reason": "더위에 적합한 가벼움"}
            ],
            "autumn": [
                {"adjustment": "스파이스 비율 증가", "reason": "가을의 따뜻함 연출"},
                {"adjustment": "우디 베이스 강화", "reason": "계절의 깊이감 표현"}
            ],
            "winter": [
                {"adjustment": "오리엔탈 노트 강화", "reason": "겨울의 포근함 제공"},
                {"adjustment": "전체 농도를 120%로 조정", "reason": "추위에 대응하는 풍성함"}
            ]
        }
        return adjustments.get(season, [])

    def _generate_occasion_adjustments(self, occasion: str) -> List[Dict[str, Any]]:
        """상황별 조정 제안"""
        adjustments = {
            "business": [
                {"adjustment": "프로젝션을 약하게 조정", "reason": "업무 환경에 적합한 절제된 존재감"},
                {"adjustment": "클린 노트 강화", "reason": "전문적인 인상 강화"}
            ],
            "date": [
                {"adjustment": "로맨틱 노트 강화", "reason": "특별한 분위기 연출"},
                {"adjustment": "적당한 프로젝션 유지", "reason": "매력적인 존재감"}
            ],
            "evening": [
                {"adjustment": "인텐시티 증가", "reason": "저녁의 화려함 표현"},
                {"adjustment": "럭셔리 원료 비율 증가", "reason": "특별한 밤의 완성"}
            ]
        }
        return adjustments.get(occasion, [])

    def _generate_safety_adjustments(self, ingredients: List[str]) -> List[Dict[str, Any]]:
        """안전성 조정 제안 (혁신적 조합용)"""
        return [
            {"adjustment": "전체 농도를 90%로 시작", "reason": "새로운 조합의 안전한 테스트"},
            {"adjustment": "베이스 노트에 바닐라 추가", "reason": "친숙함으로 급진적 변화 완화"},
            {"adjustment": "시트러스 노트 소량 추가", "reason": "신선함으로 조화 향상"}
        ]

    def _deduplicate_and_rank(
        self,
        recommendations: List[RecommendationResult],
        target_count: int
    ) -> List[RecommendationResult]:
        """중복 제거 및 순위 정렬"""

        # 유사도 기반 중복 제거
        unique_recommendations = []

        for rec in recommendations:
            is_duplicate = False
            for existing_rec in unique_recommendations:
                # 주요 향료가 80% 이상 겹치면 중복으로 간주
                similarity = len(set(rec.key_features) & set(existing_rec.key_features)) / \
                           max(len(rec.key_features), len(existing_rec.key_features))

                if similarity > 0.8:
                    # 더 높은 점수의 것을 유지
                    if rec.confidence_score > existing_rec.confidence_score:
                        unique_recommendations.remove(existing_rec)
                        unique_recommendations.append(rec)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_recommendations.append(rec)

        # 점수 기반 정렬
        ranked_recommendations = sorted(
            unique_recommendations,
            key=lambda x: (x.confidence_score + x.match_score) / 2,
            reverse=True
        )

        return ranked_recommendations[:target_count]

    def _post_process_recommendations(
        self,
        recommendations: List[RecommendationResult],
        request: RecommendationRequest,
        user_profile: Optional[UserProfile]
    ) -> List[RecommendationResult]:
        """추천 결과 후처리"""

        for rec in recommendations:
            # 비용 추정 추가
            if request.budget_range:
                rec.estimated_cost = self._estimate_cost(rec.fragrance_formula, request.budget_range)

            # 설명 개선
            if request.include_explanation:
                rec.explanation = self._enhance_explanation(rec, request, user_profile)

            # 대안 조정 추가
            if not rec.alternative_adjustments:
                rec.alternative_adjustments = self._generate_generic_adjustments(rec)

        return recommendations

    def _enhance_explanation(
        self,
        recommendation: RecommendationResult,
        request: RecommendationRequest,
        user_profile: Optional[UserProfile]
    ) -> str:
        """설명 개선"""

        base_explanation = recommendation.explanation

        # 사용자별 맞춤 설명 추가
        if user_profile and user_profile.preferences:
            preferred_ingredients = [
                ing for ing in recommendation.key_features
                if user_profile.preferences.get(ing, 0.5) > 0.7
            ]

            if preferred_ingredients:
                base_explanation += f" 특히 당신이 선호하는 {', '.join(preferred_ingredients)} 향료가 포함되어 있습니다."

        # 상황별 설명 추가
        if "season" in request.context:
            base_explanation += f" {request.context['season']} 계절에 특히 적합합니다."

        if "mood" in request.context:
            base_explanation += f" {request.context['mood']} 기분을 완벽하게 표현합니다."

        return base_explanation

    def _estimate_cost(
        self,
        formula: Dict[str, Any],
        budget_range: Tuple[float, float]
    ) -> float:
        """비용 추정"""

        # 복잡성 기반 기본 비용
        complexity = formula.get("complexity_analysis", {}).get("total_ingredients", 10)
        base_cost = complexity * 5000  # 향료당 5천원 기본

        # 희귀 향료 추가 비용
        rare_ingredients = ["아가우드", "용연향", "사프란", "오리스"]
        ingredients = []

        # 구조에서 향료 추출
        structure = formula.get("structure", {})
        for layer in structure.values():
            ingredients.extend(layer.get("ingredients", []))

        rare_count = sum(1 for ing in ingredients for rare in rare_ingredients if rare in ing)
        rare_cost = rare_count * 20000  # 희귀 향료당 2만원 추가

        total_cost = base_cost + rare_cost

        # 예산 범위 내에서 조정
        min_budget, max_budget = budget_range
        return min(max_budget, max(min_budget, total_cost))

    def _generate_generic_adjustments(self, recommendation: RecommendationResult) -> List[Dict[str, Any]]:
        """일반적인 조정 제안"""
        return [
            {"adjustment": "농도 조절", "reason": "개인 선호도에 따른 강도 조정"},
            {"adjustment": "보조 향료 추가", "reason": "개성 강화를 위한 미세 조정"},
            {"adjustment": "숙성 기간 조정", "reason": "최적의 조화를 위한 시간 조절"}
        ]

    async def _generate_fallback_recommendations(
        self,
        request: RecommendationRequest
    ) -> List[RecommendationResult]:
        """기본 추천 (에러 발생시 대체)"""

        # 안전한 기본 조합
        fallback_combination = ["시트러스", "라벤더", "머스크"]

        formula = {
            "name": "안전한 기본 조향",
            "style": "classic",
            "structure": {
                "top_notes": {"ingredients": ["시트러스"], "percentage": 30},
                "heart_notes": {"ingredients": ["라벤더"], "percentage": 40},
                "base_notes": {"ingredients": ["머스크"], "percentage": 30}
            }
        }

        recommendation = RecommendationResult(
            fragrance_formula=formula,
            confidence_score=0.7,
            match_score=0.6,
            recommendation_type=RecommendationType.SAFE_CHOICE,
            explanation="시스템 오류로 인한 기본 추천입니다. 클래식하고 안전한 조합입니다.",
            key_features=fallback_combination,
            expected_experience=self._predict_experience(fallback_combination),
            alternative_adjustments=[],
            perfumer_inspiration="Classic Perfumery"
        )

        return [recommendation]

    def _detect_current_season(self) -> str:
        """현재 계절 감지"""
        month = datetime.now().month

        if 3 <= month <= 5:
            return "spring"
        elif 6 <= month <= 8:
            return "summer"
        elif 9 <= month <= 11:
            return "autumn"
        else:
            return "winter"

    def _record_recommendation_metrics(
        self,
        request: RecommendationRequest,
        recommendations: List[RecommendationResult]
    ):
        """추천 성능 메트릭 기록"""

        metrics = {
            "timestamp": datetime.now(),
            "user_id": request.user_id,
            "num_requested": request.num_recommendations,
            "num_generated": len(recommendations),
            "avg_confidence": np.mean([rec.confidence_score for rec in recommendations]) if recommendations else 0,
            "avg_match_score": np.mean([rec.match_score for rec in recommendations]) if recommendations else 0,
            "recommendation_types": [rec.recommendation_type.value for rec in recommendations]
        }

        self.recommendation_metrics["daily"].append(metrics)

        # 일일 메트릭이 너무 많이 쌓이면 정리
        if len(self.recommendation_metrics["daily"]) > 10000:
            self.recommendation_metrics["daily"] = self.recommendation_metrics["daily"][-5000:]

    async def update_user_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """사용자 피드백 업데이트"""

        try:
            # 피드백 데이터 처리
            feedback_type = UserPreferenceType.EXPLICIT if "rating" in feedback else UserPreferenceType.IMPLICIT

            # 사용자 프로필 업데이트
            self.user_modeling.update_user_profile(user_id, feedback, feedback_type)

            # 추천 시스템 개선을 위한 학습 데이터로 활용
            self._process_feedback_for_learning(user_id, recommendation_id, feedback)

            logger.info(f"User feedback updated for {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update user feedback: {e}")
            return False

    def _process_feedback_for_learning(
        self,
        user_id: str,
        recommendation_id: str,
        feedback: Dict[str, Any]
    ):
        """학습을 위한 피드백 처리"""

        # 피드백을 학습 데이터로 저장
        learning_data = {
            "user_id": user_id,
            "recommendation_id": recommendation_id,
            "feedback": feedback,
            "timestamp": datetime.now()
        }

        # 실제로는 데이터베이스나 파일 시스템에 저장
        if not hasattr(self, "feedback_data"):
            self.feedback_data = []

        self.feedback_data.append(learning_data)

    def get_recommendation_analytics(self) -> Dict[str, Any]:
        """추천 시스템 분석 데이터"""

        if not self.recommendation_metrics["daily"]:
            return {"message": "No metrics available"}

        recent_metrics = self.recommendation_metrics["daily"][-100:]  # 최근 100개

        analytics = {
            "total_recommendations": len(recent_metrics),
            "avg_confidence_score": np.mean([m["avg_confidence"] for m in recent_metrics]),
            "avg_match_score": np.mean([m["avg_match_score"] for m in recent_metrics]),
            "recommendation_type_distribution": self._analyze_type_distribution(recent_metrics),
            "user_engagement": len(set([m["user_id"] for m in recent_metrics if m["user_id"]])),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "performance_trends": self._analyze_performance_trends(recent_metrics)
        }

        return analytics

    def _analyze_type_distribution(self, metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """추천 타입 분포 분석"""

        type_counts = defaultdict(int)

        for metric in metrics:
            for rec_type in metric["recommendation_types"]:
                type_counts[rec_type] += 1

        return dict(type_counts)

    def _calculate_cache_hit_rate(self) -> float:
        """캐시 히트율 계산"""

        if not hasattr(self, "_cache_stats"):
            return 0.0

        total_requests = self._cache_stats.get("total_requests", 0)
        cache_hits = self._cache_stats.get("cache_hits", 0)

        return cache_hits / total_requests if total_requests > 0 else 0.0

    def _analyze_performance_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """성능 트렌드 분석"""

        if len(metrics) < 10:
            return {"message": "Insufficient data for trend analysis"}

        # 시간별 성능 변화
        timestamps = [m["timestamp"] for m in metrics]
        confidence_scores = [m["avg_confidence"] for m in metrics]
        match_scores = [m["avg_match_score"] for m in metrics]

        return {
            "confidence_trend": "improving" if confidence_scores[-5:] > confidence_scores[:5] else "stable",
            "match_score_trend": "improving" if match_scores[-5:] > match_scores[:5] else "stable",
            "recommendation_volume_trend": len(metrics[-10:]) - len(metrics[:10])
        }