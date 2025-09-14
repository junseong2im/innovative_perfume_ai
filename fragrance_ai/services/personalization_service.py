from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import asyncio
import json
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..core.logging_config import get_logger, performance_logger
from ..core.exceptions import SystemException, ValidationException, ErrorCode
from ..database.connection import get_db_session

logger = get_logger(__name__)


class PreferenceType(str, Enum):
    """선호도 타입"""
    SCENT_PROFILE = "scent_profile"         # 향 프로필
    MOOD_PREFERENCE = "mood_preference"     # 무드 선호도
    OCCASION_STYLE = "occasion_style"       # 상황별 스타일
    BRAND_AFFINITY = "brand_affinity"       # 브랜드 친화도
    PRICE_SENSITIVITY = "price_sensitivity" # 가격 민감도
    COMPLEXITY_LEVEL = "complexity_level"   # 복잡도 선호도
    SEASONAL_TREND = "seasonal_trend"       # 계절 트렌드


class PersonalityTrait(str, Enum):
    """성격 특성"""
    ADVENTUROUS = "adventurous"     # 모험적
    ROMANTIC = "romantic"           # 로맨틱
    SOPHISTICATED = "sophisticated" # 세련된
    PLAYFUL = "playful"            # 장난기 있는
    MYSTERIOUS = "mysterious"       # 신비로운
    CONFIDENT = "confident"         # 자신감 있는
    GENTLE = "gentle"              # 온화한
    ENERGETIC = "energetic"        # 활동적인


@dataclass
class UserProfile:
    """사용자 프로필"""
    user_id: str
    
    # 기본 정보
    demographic_info: Dict[str, Any]
    personality_traits: List[PersonalityTrait]
    lifestyle_info: Dict[str, Any]
    
    # 향수 선호도
    scent_preferences: Dict[str, float]  # 노트별 선호도 점수
    mood_associations: Dict[str, List[str]]  # 무드별 선호 향료
    occasion_preferences: Dict[str, Dict[str, Any]]  # 상황별 선호도
    
    # 영화 취향 (확장 기능)
    movie_preferences: Dict[str, Any]
    genre_affinity: Dict[str, float]
    director_preference: List[str]
    actor_preference: List[str]
    
    # 학습된 패턴
    learned_patterns: Dict[str, Any]
    preference_evolution: List[Dict[str, Any]]  # 시간별 취향 변화
    
    # 메타데이터
    confidence_score: float  # 프로필 정확도 신뢰도
    last_updated: datetime
    interaction_count: int
    created_at: datetime


@dataclass
class MovieFragranceProfile:
    """영화-향수 연결 프로필"""
    movie_id: str
    title: str
    genre: List[str]
    release_year: int
    director: str
    main_actors: List[str]
    
    # 영화 분위기 분석
    mood_analysis: Dict[str, float]  # 감정별 점수
    visual_palette: List[str]        # 주요 색상 팔레트
    setting_atmosphere: Dict[str, Any] # 배경/분위기
    narrative_elements: Dict[str, Any] # 서사적 요소
    
    # 매칭된 향료 프로필
    fragrance_profile: Dict[str, Any]
    recommended_notes: List[Dict[str, Any]]
    scent_story: str  # 향기가 담은 이야기
    
    # 인기도 및 평가
    popularity_score: float
    user_ratings: List[float]
    fragrance_match_accuracy: float
    
    created_at: datetime
    updated_at: datetime


class MovieFragranceDatabase:
    """영화-향수 데이터베이스"""
    
    def __init__(self):
        self.movie_profiles: Dict[str, MovieFragranceProfile] = {}
        self._initialize_movie_database()
    
    def _initialize_movie_database(self):
        """영화 데이터베이스 초기화"""
        
        # 다양한 장르의 대표 영화들과 매칭되는 향기 프로필
        movies_data = [
            {
                "title": "카사블랑카 (Casablanca, 1942)",
                "genre": ["romance", "drama", "classic"],
                "release_year": 1942,
                "director": "마이클 커티즈",
                "main_actors": ["험프리 보가트", "잉그리드 버그만"],
                "mood_analysis": {
                    "romantic": 0.9,
                    "nostalgic": 0.8,
                    "sophisticated": 0.9,
                    "mysterious": 0.7,
                    "melancholic": 0.6
                },
                "visual_palette": ["black", "white", "sepia", "golden"],
                "setting_atmosphere": {
                    "location": "morocco_cafe",
                    "time_period": "1940s",
                    "social_context": "wartime_romance",
                    "lighting": "dramatic_shadows"
                },
                "fragrance_profile": {
                    "primary_notes": ["bergamot", "jasmine", "sandalwood", "amber"],
                    "secondary_notes": ["rose", "cedar", "vanilla", "musk"],
                    "intensity": 0.8,
                    "longevity": 0.9,
                    "complexity": 0.8,
                    "elegance": 0.95
                },
                "scent_story": "카사블랑카의 로맨스처럼 시작은 신선한 베르가못과 자스민으로 상쾌하지만, 시간이 지나면서 깊은 샌달우드와 앰버가 영원한 사랑의 여운을 남깁니다.",
                "recommended_notes": [
                    {"name": "bergamot", "intensity": 0.7, "role": "opening", "story": "첫 만남의 설렘"},
                    {"name": "jasmine", "intensity": 0.8, "role": "heart", "story": "밤의 로맨스"},
                    {"name": "sandalwood", "intensity": 0.9, "role": "base", "story": "영원한 기억"},
                    {"name": "amber", "intensity": 0.8, "role": "base", "story": "황금빛 추억"}
                ]
            },
            {
                "title": "라라랜드 (La La Land, 2016)",
                "genre": ["musical", "romance", "drama"],
                "release_year": 2016,
                "director": "데미언 셔젤",
                "main_actors": ["라이언 고슬링", "엠마 스톤"],
                "mood_analysis": {
                    "dreamy": 0.9,
                    "optimistic": 0.8,
                    "nostalgic": 0.7,
                    "whimsical": 0.9,
                    "bittersweet": 0.6
                },
                "visual_palette": ["purple", "pink", "yellow", "blue", "golden"],
                "setting_atmosphere": {
                    "location": "los_angeles",
                    "time_period": "contemporary",
                    "social_context": "artistic_dreams",
                    "lighting": "magical_golden_hour"
                },
                "fragrance_profile": {
                    "primary_notes": ["grapefruit", "peony", "freesia", "vanilla"],
                    "secondary_notes": ["pink_pepper", "magnolia", "cedar", "white_musk"],
                    "intensity": 0.6,
                    "longevity": 0.7,
                    "complexity": 0.7,
                    "playfulness": 0.9
                },
                "scent_story": "라라랜드의 꿈처럼 상큼한 자몽으로 시작해 장난스러운 피오니와 프리지아가 어우러지며, 따뜻한 바닐라가 꿈을 현실로 만드는 마법을 선사합니다.",
                "recommended_notes": [
                    {"name": "grapefruit", "intensity": 0.8, "role": "opening", "story": "LA의 햇살"},
                    {"name": "peony", "intensity": 0.7, "role": "heart", "story": "꿈꾸는 마음"},
                    {"name": "freesia", "intensity": 0.6, "role": "heart", "story": "순수한 열정"},
                    {"name": "vanilla", "intensity": 0.8, "role": "base", "story": "달콤한 결말"}
                ]
            },
            {
                "title": "블레이드 러너 2049 (Blade Runner 2049, 2017)",
                "genre": ["sci-fi", "thriller", "neo-noir"],
                "release_year": 2017,
                "director": "드니 빌뇌브",
                "main_actors": ["라이언 고슬링", "해리슨 포드"],
                "mood_analysis": {
                    "futuristic": 0.9,
                    "melancholic": 0.8,
                    "mysterious": 0.9,
                    "contemplative": 0.8,
                    "dystopian": 0.9
                },
                "visual_palette": ["orange", "purple", "cyan", "black", "neon"],
                "setting_atmosphere": {
                    "location": "dystopian_future",
                    "time_period": "2049",
                    "social_context": "technological_isolation",
                    "lighting": "neon_cyberpunk"
                },
                "fragrance_profile": {
                    "primary_notes": ["ozone", "metal", "iris", "ambergris"],
                    "secondary_notes": ["black_pepper", "violet", "cedar", "synthetic_musk"],
                    "intensity": 0.9,
                    "longevity": 0.9,
                    "complexity": 0.9,
                    "avant_garde": 0.95
                },
                "scent_story": "2049년의 미래처럼 차가운 오존과 메탈릭 노트로 시작해, 신비로운 아이리스와 앰버그리스가 인간과 기계 사이의 경계를 탐구합니다.",
                "recommended_notes": [
                    {"name": "ozone", "intensity": 0.8, "role": "opening", "story": "미래의 공기"},
                    {"name": "metal", "intensity": 0.7, "role": "opening", "story": "기계의 차가움"},
                    {"name": "iris", "intensity": 0.9, "role": "heart", "story": "인간의 감정"},
                    {"name": "ambergris", "intensity": 0.8, "role": "base", "story": "존재의 깊이"}
                ]
            },
            {
                "title": "미드서머 (Midsommar, 2019)",
                "genre": ["horror", "thriller", "folk"],
                "release_year": 2019,
                "director": "아리 아스터",
                "main_actors": ["플로렌스 퓨", "잭 레이너"],
                "mood_analysis": {
                    "disturbing": 0.9,
                    "pastoral": 0.7,
                    "ritualistic": 0.8,
                    "hypnotic": 0.8,
                    "unnerving": 0.9
                },
                "visual_palette": ["white", "yellow", "green", "red", "floral"],
                "setting_atmosphere": {
                    "location": "swedish_countryside",
                    "time_period": "contemporary",
                    "social_context": "folk_horror",
                    "lighting": "perpetual_daylight"
                },
                "fragrance_profile": {
                    "primary_notes": ["meadow_grass", "dandelion", "elderflower", "birch"],
                    "secondary_notes": ["honey", "marigold", "pine", "earth"],
                    "intensity": 0.7,
                    "longevity": 0.8,
                    "complexity": 0.8,
                    "unsettling": 0.9
                },
                "scent_story": "미드서머의 하얀 밤처럼 아름다운 목초지와 민들레 향이 시작되지만, 숨겨진 자작나무와 흙내음이 불안한 진실을 드러냅니다.",
                "recommended_notes": [
                    {"name": "meadow_grass", "intensity": 0.8, "role": "opening", "story": "평화로운 들판"},
                    {"name": "elderflower", "intensity": 0.7, "role": "heart", "story": "순백의 축제"},
                    {"name": "honey", "intensity": 0.6, "role": "heart", "story": "달콤한 유혹"},
                    {"name": "birch", "intensity": 0.9, "role": "base", "story": "숨겨진 진실"}
                ]
            }
        ]
        
        for movie_data in movies_data:
            movie_id = str(uuid.uuid4())
            
            profile = MovieFragranceProfile(
                movie_id=movie_id,
                popularity_score=0.8,
                user_ratings=[4.2, 4.5, 4.0, 4.8, 4.3],
                fragrance_match_accuracy=0.85,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                **movie_data
            )
            
            self.movie_profiles[movie_id] = profile
            
            logger.info(f"Movie fragrance profile initialized: {profile.title}")
    
    def search_movies_by_genre(self, genres: List[str]) -> List[MovieFragranceProfile]:
        """장르별 영화 검색"""
        
        matching_movies = []
        
        for profile in self.movie_profiles.values():
            genre_overlap = set(genres) & set(profile.genre)
            if genre_overlap:
                matching_movies.append(profile)
        
        # 장르 매칭도순으로 정렬
        matching_movies.sort(
            key=lambda x: len(set(genres) & set(x.genre)),
            reverse=True
        )
        
        return matching_movies
    
    def search_movies_by_mood(self, target_moods: Dict[str, float]) -> List[Tuple[MovieFragranceProfile, float]]:
        """무드별 영화 검색"""
        
        movie_scores = []
        
        for profile in self.movie_profiles.values():
            # 무드 유사도 계산
            similarity_score = 0.0
            total_weight = 0.0
            
            for mood, target_intensity in target_moods.items():
                if mood in profile.mood_analysis:
                    movie_intensity = profile.mood_analysis[mood]
                    # 코사인 유사도 근사
                    similarity = 1 - abs(target_intensity - movie_intensity)
                    similarity_score += similarity * target_intensity
                    total_weight += target_intensity
            
            if total_weight > 0:
                final_score = similarity_score / total_weight
                movie_scores.append((profile, final_score))
        
        # 유사도순으로 정렬
        movie_scores.sort(key=lambda x: x[1], reverse=True)
        
        return movie_scores
    
    def get_movie_by_id(self, movie_id: str) -> Optional[MovieFragranceProfile]:
        """ID로 영화 조회"""
        return self.movie_profiles.get(movie_id)
    
    def get_all_movies(self) -> List[MovieFragranceProfile]:
        """모든 영화 조회"""
        return list(self.movie_profiles.values())


class PersonalizationEngine:
    """개인화 엔진"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.movie_db = MovieFragranceDatabase()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.preference_model = None
    
    async def analyze_user_preferences(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> UserProfile:
        """사용자 선호도 분석"""
        
        try:
            if user_id in self.user_profiles:
                profile = self.user_profiles[user_id]
            else:
                profile = await self._create_initial_profile(user_id, interaction_data)
            
            # 새로운 상호작용 데이터로 프로필 업데이트
            await self._update_profile_from_interaction(profile, interaction_data)
            
            # 머신러닝 기반 패턴 학습
            await self._learn_preference_patterns(profile)
            
            # 신뢰도 점수 업데이트
            profile.confidence_score = self._calculate_confidence_score(profile)
            profile.last_updated = datetime.utcnow()
            profile.interaction_count += 1
            
            self.user_profiles[user_id] = profile
            
            logger.info(f"User preferences analyzed: {user_id}")
            
            performance_logger.log_execution_time(
                operation="analyze_user_preferences",
                execution_time=0.0,
                success=True,
                extra_data={
                    "user_id": user_id,
                    "confidence_score": profile.confidence_score,
                    "interaction_count": profile.interaction_count
                }
            )
            
            return profile
            
        except Exception as e:
            logger.error(f"User preference analysis failed: {e}")
            raise SystemException(
                message=f"사용자 선호도 분석 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    async def _create_initial_profile(
        self,
        user_id: str,
        initial_data: Dict[str, Any]
    ) -> UserProfile:
        """초기 사용자 프로필 생성"""
        
        # 초기 데이터에서 기본 정보 추출
        demographic_info = initial_data.get("demographics", {})
        lifestyle_info = initial_data.get("lifestyle", {})
        
        # 기본 선호도 설정
        scent_preferences = {}
        base_notes = [
            "bergamot", "lemon", "rose", "jasmine", "lavender", 
            "sandalwood", "vanilla", "musk", "amber", "cedar"
        ]
        
        for note in base_notes:
            scent_preferences[note] = 0.5  # 중간값으로 시작
        
        # 성격 특성 추론 (설문이나 초기 상호작용 기반)
        personality_traits = self._infer_personality_traits(initial_data)
        
        # 영화 취향 초기화
        movie_preferences = initial_data.get("movie_preferences", {})
        genre_affinity = self._initialize_genre_affinity(movie_preferences)
        
        profile = UserProfile(
            user_id=user_id,
            demographic_info=demographic_info,
            personality_traits=personality_traits,
            lifestyle_info=lifestyle_info,
            scent_preferences=scent_preferences,
            mood_associations={},
            occasion_preferences={},
            movie_preferences=movie_preferences,
            genre_affinity=genre_affinity,
            director_preference=movie_preferences.get("favorite_directors", []),
            actor_preference=movie_preferences.get("favorite_actors", []),
            learned_patterns={},
            preference_evolution=[],
            confidence_score=0.3,  # 낮은 초기 신뢰도
            last_updated=datetime.utcnow(),
            interaction_count=0,
            created_at=datetime.utcnow()
        )
        
        return profile
    
    def _infer_personality_traits(self, data: Dict[str, Any]) -> List[PersonalityTrait]:
        """성격 특성 추론"""
        
        traits = []
        
        # 나이대별 특성
        age = data.get("demographics", {}).get("age")
        if age:
            if age < 25:
                traits.extend([PersonalityTrait.PLAYFUL, PersonalityTrait.ENERGETIC])
            elif age > 50:
                traits.extend([PersonalityTrait.SOPHISTICATED, PersonalityTrait.GENTLE])
        
        # 영화 취향 기반 특성
        favorite_genres = data.get("movie_preferences", {}).get("favorite_genres", [])
        for genre in favorite_genres:
            if genre in ["romance", "drama"]:
                traits.append(PersonalityTrait.ROMANTIC)
            elif genre in ["horror", "thriller"]:
                traits.append(PersonalityTrait.ADVENTUROUS)
            elif genre in ["mystery", "noir"]:
                traits.append(PersonalityTrait.MYSTERIOUS)
        
        # 중복 제거
        return list(set(traits))
    
    def _initialize_genre_affinity(self, movie_preferences: Dict[str, Any]) -> Dict[str, float]:
        """장르 친화도 초기화"""
        
        affinity = {}
        all_genres = [
            "action", "comedy", "drama", "horror", "romance", "sci-fi",
            "thriller", "animation", "documentary", "musical", "western"
        ]
        
        favorite_genres = movie_preferences.get("favorite_genres", [])
        
        for genre in all_genres:
            if genre in favorite_genres:
                affinity[genre] = 0.8
            else:
                affinity[genre] = 0.5
        
        return affinity
    
    async def _update_profile_from_interaction(
        self,
        profile: UserProfile,
        interaction_data: Dict[str, Any]
    ):
        """상호작용 데이터로 프로필 업데이트"""
        
        # 향수 평가 데이터 처리
        if "fragrance_rating" in interaction_data:
            rating_data = interaction_data["fragrance_rating"]
            await self._update_scent_preferences(profile, rating_data)
        
        # 무드 연관성 업데이트
        if "mood_selection" in interaction_data:
            mood_data = interaction_data["mood_selection"]
            await self._update_mood_associations(profile, mood_data)
        
        # 영화 평가 데이터 처리
        if "movie_rating" in interaction_data:
            movie_data = interaction_data["movie_rating"]
            await self._update_movie_preferences(profile, movie_data)
        
        # 선호도 변화 기록
        preference_snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "scent_preferences": profile.scent_preferences.copy(),
            "genre_affinity": profile.genre_affinity.copy(),
            "interaction_type": list(interaction_data.keys())
        }
        
        profile.preference_evolution.append(preference_snapshot)
        
        # 최근 10개만 유지
        if len(profile.preference_evolution) > 10:
            profile.preference_evolution = profile.preference_evolution[-10:]
    
    async def _update_scent_preferences(
        self,
        profile: UserProfile,
        rating_data: Dict[str, Any]
    ):
        """향 선호도 업데이트"""
        
        recipe_notes = rating_data.get("notes", [])
        user_rating = rating_data.get("rating", 3.0)  # 1-5 scale
        
        # 평점을 -1 ~ +1 범위로 정규화
        preference_change = (user_rating - 3.0) / 2.0
        
        for note in recipe_notes:
            if note in profile.scent_preferences:
                # 기존 선호도와 새 평가의 가중 평균
                current_pref = profile.scent_preferences[note]
                profile.scent_preferences[note] = (
                    current_pref * 0.8 + (0.5 + preference_change * 0.5) * 0.2
                )
                # 0-1 범위로 클리핑
                profile.scent_preferences[note] = max(0.0, min(1.0, profile.scent_preferences[note]))
    
    async def _update_mood_associations(
        self,
        profile: UserProfile,
        mood_data: Dict[str, Any]
    ):
        """무드 연관성 업데이트"""
        
        selected_mood = mood_data.get("mood")
        associated_notes = mood_data.get("notes", [])
        
        if selected_mood:
            if selected_mood not in profile.mood_associations:
                profile.mood_associations[selected_mood] = []
            
            # 새로운 노트들 추가 (중복 방지)
            for note in associated_notes:
                if note not in profile.mood_associations[selected_mood]:
                    profile.mood_associations[selected_mood].append(note)
            
            # 최대 10개까지 유지
            profile.mood_associations[selected_mood] = profile.mood_associations[selected_mood][-10:]
    
    async def _update_movie_preferences(
        self,
        profile: UserProfile,
        movie_data: Dict[str, Any]
    ):
        """영화 선호도 업데이트"""
        
        movie_genres = movie_data.get("genres", [])
        user_rating = movie_data.get("rating", 3.0)
        
        # 장르 친화도 업데이트
        preference_change = (user_rating - 3.0) / 2.0
        
        for genre in movie_genres:
            if genre in profile.genre_affinity:
                current_affinity = profile.genre_affinity[genre]
                profile.genre_affinity[genre] = (
                    current_affinity * 0.8 + (0.5 + preference_change * 0.5) * 0.2
                )
                profile.genre_affinity[genre] = max(0.0, min(1.0, profile.genre_affinity[genre]))
        
        # 감독/배우 선호도 업데이트
        director = movie_data.get("director")
        if director and user_rating >= 4.0:
            if director not in profile.director_preference:
                profile.director_preference.append(director)
        
        actors = movie_data.get("actors", [])
        for actor in actors:
            if user_rating >= 4.0 and actor not in profile.actor_preference:
                profile.actor_preference.append(actor)
        
        # 리스트 크기 제한
        profile.director_preference = profile.director_preference[-20:]
        profile.actor_preference = profile.actor_preference[-30:]
    
    async def _learn_preference_patterns(self, profile: UserProfile):
        """선호도 패턴 학습"""
        
        if len(profile.preference_evolution) < 3:
            return  # 데이터 부족
        
        try:
            # 시간별 선호도 변화 패턴 분석
            time_patterns = self._analyze_temporal_patterns(profile)
            
            # 향-무드 연관성 패턴 학습
            scent_mood_patterns = self._analyze_scent_mood_patterns(profile)
            
            # 영화-향수 선호도 상관관계 분석
            movie_fragrance_correlation = self._analyze_movie_fragrance_correlation(profile)
            
            profile.learned_patterns = {
                "temporal_patterns": time_patterns,
                "scent_mood_patterns": scent_mood_patterns,
                "movie_fragrance_correlation": movie_fragrance_correlation,
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pattern learning failed for user {profile.user_id}: {e}")
    
    def _analyze_temporal_patterns(self, profile: UserProfile) -> Dict[str, Any]:
        """시간별 패턴 분석"""
        
        if len(profile.preference_evolution) < 2:
            return {}
        
        # 선호도 변화 추세 분석
        changes = {}
        
        for note in profile.scent_preferences.keys():
            note_history = []
            for snapshot in profile.preference_evolution:
                if note in snapshot["scent_preferences"]:
                    note_history.append(snapshot["scent_preferences"][note])
            
            if len(note_history) >= 2:
                trend = np.polyfit(range(len(note_history)), note_history, 1)[0]
                changes[note] = {
                    "trend": "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable",
                    "slope": float(trend)
                }
        
        return {"preference_changes": changes}
    
    def _analyze_scent_mood_patterns(self, profile: UserProfile) -> Dict[str, Any]:
        """향-무드 패턴 분석"""
        
        mood_note_matrix = {}
        
        for mood, notes in profile.mood_associations.items():
            mood_note_matrix[mood] = {}
            for note in notes:
                if note in profile.scent_preferences:
                    mood_note_matrix[mood][note] = profile.scent_preferences[note]
        
        return {"mood_note_correlations": mood_note_matrix}
    
    def _analyze_movie_fragrance_correlation(self, profile: UserProfile) -> Dict[str, Any]:
        """영화-향수 상관관계 분석"""
        
        correlations = {}
        
        # 장르별 선호 향료 분석
        for genre, affinity in profile.genre_affinity.items():
            if affinity > 0.6:  # 선호하는 장르만
                # 해당 장르와 잘 맞는 향료들 찾기
                genre_movies = self.movie_db.search_movies_by_genre([genre])
                if genre_movies:
                    common_notes = []
                    for movie in genre_movies[:3]:  # 상위 3개 영화
                        movie_notes = movie.fragrance_profile.get("primary_notes", [])
                        common_notes.extend(movie_notes)
                    
                    # 빈도 계산
                    note_frequency = Counter(common_notes)
                    correlations[genre] = dict(note_frequency.most_common(5))
        
        return {"genre_note_correlations": correlations}
    
    def _calculate_confidence_score(self, profile: UserProfile) -> float:
        """프로필 신뢰도 점수 계산"""
        
        confidence = 0.0
        
        # 상호작용 횟수 기반 (0.4)
        interaction_factor = min(profile.interaction_count / 50, 1.0) * 0.4
        confidence += interaction_factor
        
        # 데이터 완성도 기반 (0.3)
        completeness = 0.0
        if profile.demographic_info:
            completeness += 0.1
        if profile.lifestyle_info:
            completeness += 0.1
        if len(profile.personality_traits) > 0:
            completeness += 0.1
        
        confidence += completeness
        
        # 선호도 분산도 (0.2) - 너무 균등하면 신뢰도 낮음
        scent_prefs = list(profile.scent_preferences.values())
        if scent_prefs:
            pref_variance = np.var(scent_prefs)
            variance_factor = min(pref_variance * 4, 1.0) * 0.2
            confidence += variance_factor
        
        # 학습된 패턴 존재 여부 (0.1)
        if profile.learned_patterns:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def recommend_movie_fragrance(
        self,
        user_id: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """영화 기반 향수 추천"""
        
        try:
            if user_id not in self.user_profiles:
                raise ValidationException("사용자 프로필을 찾을 수 없습니다")
            
            profile = self.user_profiles[user_id]
            
            # 사용자 선호 장르 추출
            preferred_genres = [
                genre for genre, affinity in profile.genre_affinity.items()
                if affinity > 0.6
            ]
            
            # 현재 무드 고려 (preferences에서)
            target_mood = preferences.get("current_mood", {}) if preferences else {}
            
            recommendations = []
            
            # 장르 기반 추천
            if preferred_genres:
                genre_movies = self.movie_db.search_movies_by_genre(preferred_genres)
                for movie in genre_movies[:3]:
                    rec = await self._create_movie_fragrance_recommendation(profile, movie, "genre_match")
                    recommendations.append(rec)
            
            # 무드 기반 추천
            if target_mood:
                mood_movies = self.movie_db.search_movies_by_mood(target_mood)
                for movie_profile, score in mood_movies[:2]:
                    if score > 0.7:  # 높은 유사도만
                        rec = await self._create_movie_fragrance_recommendation(
                            profile, movie_profile, "mood_match", mood_score=score
                        )
                        recommendations.append(rec)
            
            # 개인화 점수로 정렬
            recommendations.sort(key=lambda x: x["personalization_score"], reverse=True)
            
            return recommendations[:5]  # 상위 5개
            
        except Exception as e:
            logger.error(f"Movie fragrance recommendation failed: {e}")
            raise SystemException(
                message=f"영화 기반 향수 추천 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    async def _create_movie_fragrance_recommendation(
        self,
        profile: UserProfile,
        movie: MovieFragranceProfile,
        match_type: str,
        mood_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """영화-향수 추천 생성"""
        
        # 개인화 점수 계산
        personalization_score = await self._calculate_personalization_score(profile, movie)
        
        # 사용자 취향에 맞는 노트 조정
        customized_notes = self._customize_notes_for_user(profile, movie.recommended_notes)
        
        # 추천 이유 생성
        recommendation_reason = self._generate_recommendation_reason(
            profile, movie, match_type, personalization_score
        )
        
        return {
            "movie_id": movie.movie_id,
            "movie_title": movie.title,
            "movie_info": {
                "genre": movie.genre,
                "director": movie.director,
                "year": movie.release_year,
                "mood_profile": movie.mood_analysis
            },
            "fragrance_recommendation": {
                "scent_story": movie.scent_story,
                "primary_notes": customized_notes[:4],  # 상위 4개
                "intensity": movie.fragrance_profile["intensity"],
                "complexity": movie.fragrance_profile["complexity"],
                "longevity": movie.fragrance_profile["longevity"]
            },
            "personalization_score": personalization_score,
            "match_type": match_type,
            "mood_similarity": mood_score,
            "recommendation_reason": recommendation_reason,
            "estimated_satisfaction": self._estimate_user_satisfaction(profile, movie),
            "customization_level": self._calculate_customization_level(profile, movie)
        }
    
    async def _calculate_personalization_score(
        self,
        profile: UserProfile,
        movie: MovieFragranceProfile
    ) -> float:
        """개인화 점수 계산"""
        
        score = 0.0
        
        # 장르 친화도 (0.3)
        genre_match = 0.0
        for genre in movie.genre:
            if genre in profile.genre_affinity:
                genre_match += profile.genre_affinity[genre]
        
        if movie.genre:
            genre_score = genre_match / len(movie.genre)
            score += genre_score * 0.3
        
        # 향료 선호도 매칭 (0.4)
        movie_notes = movie.fragrance_profile.get("primary_notes", [])
        note_match_score = 0.0
        
        for note in movie_notes:
            if note in profile.scent_preferences:
                note_match_score += profile.scent_preferences[note]
        
        if movie_notes:
            note_score = note_match_score / len(movie_notes)
            score += note_score * 0.4
        
        # 성격 특성 매칭 (0.2)
        personality_match = self._calculate_personality_match(profile, movie)
        score += personality_match * 0.2
        
        # 감독/배우 선호도 (0.1)
        celebrity_match = 0.0
        if movie.director in profile.director_preference:
            celebrity_match += 0.5
        
        common_actors = set(movie.main_actors) & set(profile.actor_preference)
        celebrity_match += len(common_actors) * 0.2
        
        score += min(celebrity_match, 1.0) * 0.1
        
        return min(score, 1.0)
    
    def _calculate_personality_match(
        self,
        profile: UserProfile,
        movie: MovieFragranceProfile
    ) -> float:
        """성격 특성 매칭 점수"""
        
        # 성격과 영화 무드의 매칭
        personality_mood_map = {
            PersonalityTrait.ROMANTIC: ["romantic", "dreamy", "nostalgic"],
            PersonalityTrait.ADVENTUROUS: ["exciting", "thrilling", "dynamic"],
            PersonalityTrait.MYSTERIOUS: ["mysterious", "enigmatic", "dark"],
            PersonalityTrait.SOPHISTICATED: ["elegant", "refined", "sophisticated"],
            PersonalityTrait.PLAYFUL: ["whimsical", "fun", "lighthearted"],
            PersonalityTrait.CONFIDENT: ["bold", "strong", "assertive"]
        }
        
        match_score = 0.0
        
        for trait in profile.personality_traits:
            if trait in personality_mood_map:
                trait_moods = personality_mood_map[trait]
                for mood in trait_moods:
                    if mood in movie.mood_analysis:
                        match_score += movie.mood_analysis[mood]
        
        if profile.personality_traits:
            return match_score / len(profile.personality_traits)
        
        return 0.5  # 기본값
    
    def _customize_notes_for_user(
        self,
        profile: UserProfile,
        original_notes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """사용자 취향에 맞게 노트 조정"""
        
        customized = []
        
        for note_info in original_notes:
            note_name = note_info["name"]
            
            # 사용자 선호도에 따라 강도 조정
            if note_name in profile.scent_preferences:
                user_preference = profile.scent_preferences[note_name]
                
                # 선호도가 높으면 강도 증가, 낮으면 감소
                original_intensity = note_info.get("intensity", 0.5)
                adjusted_intensity = original_intensity * (0.5 + user_preference * 0.5)
                
                customized_note = note_info.copy()
                customized_note["intensity"] = min(adjusted_intensity, 1.0)
                customized_note["user_preference"] = user_preference
                customized_note["customized"] = abs(adjusted_intensity - original_intensity) > 0.1
                
                customized.append(customized_note)
            else:
                customized.append(note_info)
        
        # 사용자 선호도가 높은 노트 순으로 정렬
        customized.sort(
            key=lambda x: x.get("user_preference", 0.5) * x.get("intensity", 0.5),
            reverse=True
        )
        
        return customized
    
    def _generate_recommendation_reason(
        self,
        profile: UserProfile,
        movie: MovieFragranceProfile,
        match_type: str,
        personalization_score: float
    ) -> List[str]:
        """추천 이유 생성"""
        
        reasons = []
        
        # 매칭 타입별 기본 이유
        if match_type == "genre_match":
            favorite_genres = [g for g, a in profile.genre_affinity.items() if a > 0.6]
            matching_genres = list(set(favorite_genres) & set(movie.genre))
            if matching_genres:
                reasons.append(f"좋아하시는 {', '.join(matching_genres)} 장르와 완벽하게 매칭됩니다")
        
        elif match_type == "mood_match":
            reasons.append("현재 원하시는 무드와 매우 유사한 영화입니다")
        
        # 개인화 점수 기반 이유
        if personalization_score > 0.8:
            reasons.append("귀하의 취향과 매우 높은 일치도를 보입니다")
        elif personalization_score > 0.7:
            reasons.append("귀하의 선호도와 잘 맞습니다")
        
        # 특별한 요소들
        if movie.director in profile.director_preference:
            reasons.append(f"선호하시는 {movie.director} 감독의 작품입니다")
        
        common_actors = set(movie.main_actors) & set(profile.actor_preference)
        if common_actors:
            actors = list(common_actors)[:2]  # 최대 2명
            reasons.append(f"좋아하시는 배우 {', '.join(actors)}가 출연합니다")
        
        # 향료 특징
        high_pref_notes = [
            note for note, pref in profile.scent_preferences.items()
            if pref > 0.7
        ]
        movie_notes = movie.fragrance_profile.get("primary_notes", [])
        matching_notes = list(set(high_pref_notes) & set(movie_notes))
        
        if matching_notes:
            reasons.append(f"선호하시는 {matching_notes[0]} 향료가 핵심 노트로 사용됩니다")
        
        return reasons[:4]  # 최대 4개
    
    def _estimate_user_satisfaction(
        self,
        profile: UserProfile,
        movie: MovieFragranceProfile
    ) -> float:
        """사용자 만족도 예상"""
        
        # 과거 유사한 추천의 만족도 기반 (실제로는 피드백 데이터 사용)
        base_satisfaction = 0.75
        
        # 개인화 점수 기반 조정
        personalization_boost = self._calculate_personalization_score(profile, movie) * 0.2
        
        # 신뢰도 점수 기반 조정
        confidence_factor = profile.confidence_score * 0.1
        
        estimated = base_satisfaction + personalization_boost + confidence_factor
        
        return min(estimated, 1.0)
    
    def _calculate_customization_level(
        self,
        profile: UserProfile,
        movie: MovieFragranceProfile
    ) -> str:
        """개인화 수준 계산"""
        
        personalization_score = self._calculate_personalization_score(profile, movie)
        
        if personalization_score > 0.8:
            return "high"
        elif personalization_score > 0.6:
            return "medium"
        else:
            return "low"
    
    async def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """사용자 인사이트 조회"""
        
        try:
            if user_id not in self.user_profiles:
                raise ValidationException("사용자 프로필을 찾을 수 없습니다")
            
            profile = self.user_profiles[user_id]
            
            insights = {
                "profile_summary": {
                    "confidence_score": profile.confidence_score,
                    "interaction_count": profile.interaction_count,
                    "profile_age_days": (datetime.utcnow() - profile.created_at).days,
                    "personality_traits": [trait.value for trait in profile.personality_traits]
                },
                "scent_preferences": {
                    "top_preferred_notes": sorted(
                        profile.scent_preferences.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10],
                    "preference_stability": self._analyze_preference_stability(profile),
                    "unique_preferences": self._identify_unique_preferences(profile)
                },
                "movie_taste_profile": {
                    "favorite_genres": sorted(
                        profile.genre_affinity.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5],
                    "taste_evolution": self._analyze_taste_evolution(profile),
                    "genre_diversity": self._calculate_genre_diversity(profile)
                },
                "recommendations_performance": {
                    "estimated_satisfaction": self._get_avg_estimated_satisfaction(profile),
                    "recommendation_diversity": self._calculate_recommendation_diversity(profile),
                    "personalization_effectiveness": profile.confidence_score
                },
                "behavioral_patterns": profile.learned_patterns,
                "future_predictions": await self._predict_future_preferences(profile)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"User insights generation failed: {e}")
            raise SystemException(
                message=f"사용자 인사이트 생성 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _analyze_preference_stability(self, profile: UserProfile) -> Dict[str, Any]:
        """선호도 안정성 분석"""
        
        if len(profile.preference_evolution) < 3:
            return {"status": "insufficient_data"}
        
        # 최근 3개 스냅샷의 변화도 계산
        recent_snapshots = profile.preference_evolution[-3:]
        
        changes = []
        for i in range(1, len(recent_snapshots)):
            prev_prefs = recent_snapshots[i-1]["scent_preferences"]
            curr_prefs = recent_snapshots[i]["scent_preferences"]
            
            total_change = 0.0
            common_notes = set(prev_prefs.keys()) & set(curr_prefs.keys())
            
            for note in common_notes:
                change = abs(prev_prefs[note] - curr_prefs[note])
                total_change += change
            
            if common_notes:
                avg_change = total_change / len(common_notes)
                changes.append(avg_change)
        
        if changes:
            stability_score = 1.0 - (sum(changes) / len(changes))
            
            return {
                "stability_score": stability_score,
                "trend": "stable" if stability_score > 0.8 else "evolving" if stability_score > 0.5 else "volatile"
            }
        
        return {"status": "no_change_data"}
    
    def _identify_unique_preferences(self, profile: UserProfile) -> List[str]:
        """독특한 선호도 식별"""
        
        # 일반적이지 않은 높은 선호도를 가진 노트들 찾기
        unusual_preferences = []
        
        # 가상의 일반 대중 선호도 (실제로는 전체 사용자 데이터 기반)
        general_preferences = {
            "vanilla": 0.8, "rose": 0.7, "jasmine": 0.6, "bergamot": 0.6,
            "sandalwood": 0.5, "musk": 0.5, "amber": 0.4, "cedar": 0.4
        }
        
        for note, user_pref in profile.scent_preferences.items():
            general_pref = general_preferences.get(note, 0.5)
            
            # 일반 선호도보다 크게 높거나 낮으면 독특한 취향
            if abs(user_pref - general_pref) > 0.3:
                if user_pref > general_pref:
                    unusual_preferences.append(f"특별히 좋아함: {note}")
                else:
                    unusual_preferences.append(f"일반적이지 않게 선호 안함: {note}")
        
        return unusual_preferences[:5]
    
    def _analyze_taste_evolution(self, profile: UserProfile) -> Dict[str, Any]:
        """취향 변화 분석"""
        
        if len(profile.preference_evolution) < 2:
            return {"status": "insufficient_data"}
        
        first_snapshot = profile.preference_evolution[0]
        latest_snapshot = profile.preference_evolution[-1]
        
        # 장르 선호도 변화
        genre_changes = {}
        first_genres = first_snapshot.get("genre_affinity", {})
        latest_genres = latest_snapshot.get("genre_affinity", {})
        
        for genre in set(first_genres.keys()) | set(latest_genres.keys()):
            first_score = first_genres.get(genre, 0.5)
            latest_score = latest_genres.get(genre, 0.5)
            
            change = latest_score - first_score
            if abs(change) > 0.2:  # 의미있는 변화
                genre_changes[genre] = {
                    "change": change,
                    "direction": "increased" if change > 0 else "decreased"
                }
        
        return {
            "evolution_period_days": (datetime.utcnow() - profile.created_at).days,
            "genre_changes": genre_changes,
            "overall_trend": "diversifying" if len(genre_changes) > 2 else "stable"
        }
    
    def _calculate_genre_diversity(self, profile: UserProfile) -> float:
        """장르 다양성 계산"""
        
        genre_scores = list(profile.genre_affinity.values())
        
        if not genre_scores:
            return 0.0
        
        # 엔트로피 기반 다양성 계산
        total = sum(genre_scores)
        if total == 0:
            return 0.0
        
        normalized_scores = [score/total for score in genre_scores]
        entropy = -sum(p * np.log2(p) for p in normalized_scores if p > 0)
        
        # 0-1 범위로 정규화
        max_entropy = np.log2(len(genre_scores))
        diversity = entropy / max_entropy if max_entropy > 0 else 0
        
        return diversity
    
    def _get_avg_estimated_satisfaction(self, profile: UserProfile) -> float:
        """평균 예상 만족도"""
        
        # 실제로는 과거 추천의 실제 만족도를 기반으로 계산
        # 현재는 신뢰도 기반 추정치
        return 0.7 + profile.confidence_score * 0.2
    
    def _calculate_recommendation_diversity(self, profile: UserProfile) -> float:
        """추천 다양성 계산"""
        
        # 사용자가 선호하는 장르의 다양성 기반
        return self._calculate_genre_diversity(profile)
    
    async def _predict_future_preferences(self, profile: UserProfile) -> Dict[str, Any]:
        """미래 선호도 예측"""
        
        if len(profile.preference_evolution) < 3:
            return {"status": "insufficient_data"}
        
        predictions = {}
        
        # 트렌드 기반 예측
        learned_patterns = profile.learned_patterns
        if "temporal_patterns" in learned_patterns:
            temporal = learned_patterns["temporal_patterns"]
            preference_changes = temporal.get("preference_changes", {})
            
            for note, change_info in preference_changes.items():
                trend = change_info["trend"]
                current_pref = profile.scent_preferences.get(note, 0.5)
                
                if trend == "increasing":
                    predicted = min(current_pref + 0.1, 1.0)
                    predictions[note] = {"predicted_preference": predicted, "confidence": 0.7}
                elif trend == "decreasing":
                    predicted = max(current_pref - 0.1, 0.0)
                    predictions[note] = {"predicted_preference": predicted, "confidence": 0.7}
        
        return {
            "prediction_horizon_days": 30,
            "predicted_preferences": predictions,
            "overall_prediction_confidence": profile.confidence_score * 0.8
        }


# 전역 개인화 서비스 인스턴스
personalization_service = PersonalizationEngine()