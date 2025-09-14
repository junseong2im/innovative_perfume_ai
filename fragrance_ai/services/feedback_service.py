from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import uuid
from collections import defaultdict
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from ..core.logging_config import get_logger, performance_logger
from ..core.exceptions import SystemException, ValidationException, ErrorCode
from ..database.connection import get_db_session
from ..database.models import Recipe, RecipeEvaluation, SearchLog, GenerationLog
from ..core.monitoring import metrics_collector

logger = get_logger(__name__)


class FeedbackType(str, Enum):
    """피드백 타입"""
    RATING = "rating"                    # 평점 (1-5)
    LIKE_DISLIKE = "like_dislike"       # 좋아요/싫어요
    DETAILED = "detailed"               # 상세 평가
    COMPARISON = "comparison"           # 비교 평가
    SUGGESTION = "suggestion"           # 개선 제안


class FeedbackCategory(str, Enum):
    """피드백 카테고리"""
    RECIPE_QUALITY = "recipe_quality"           # 레시피 품질
    SCENT_ACCURACY = "scent_accuracy"          # 향 정확도
    CREATIVITY = "creativity"                   # 창의성
    PRACTICALITY = "practicality"              # 실용성
    PERSONAL_TASTE = "personal_taste"          # 개인 취향 맞춤
    SEARCH_RELEVANCE = "search_relevance"      # 검색 관련성
    UI_UX = "ui_ux"                           # 사용자 경험


@dataclass
class UserFeedback:
    """사용자 피드백 데이터"""
    feedback_id: str
    user_id: str
    session_id: str
    feedback_type: FeedbackType
    category: FeedbackCategory
    target_type: str  # recipe, search_result, generation
    target_id: str
    
    # 피드백 값들
    rating: Optional[float]  # 1-5 점수
    binary_feedback: Optional[bool]  # True(like)/False(dislike)
    detailed_scores: Optional[Dict[str, float]]  # 세부 평가 점수들
    text_feedback: Optional[str]
    tags: Optional[List[str]]
    
    # 컨텍스트 정보
    user_profile: Optional[Dict[str, Any]]
    interaction_context: Optional[Dict[str, Any]]
    timestamp: datetime
    
    # 메타데이터
    is_verified: bool = False
    weight: float = 1.0  # 피드백 가중치
    processed: bool = False


@dataclass
class ModelPerformanceMetrics:
    """모델 성능 메트릭"""
    model_name: str
    model_version: str
    evaluation_date: datetime
    
    # 정확도 메트릭
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 사용자 만족도 메트릭
    avg_user_rating: float
    satisfaction_rate: float  # 4점 이상 비율
    engagement_rate: float   # 재사용 비율
    
    # 비즈니스 메트릭
    conversion_rate: float   # 실제 구매/제작 비율
    retention_rate: float    # 사용자 재방문율
    
    # 상세 분석
    category_performance: Dict[str, float]
    user_segment_performance: Dict[str, float]
    improvement_suggestions: List[str]


class FeedbackAnalyzer:
    """피드백 분석기"""
    
    def __init__(self):
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.feedback_cache = {}
        self.analysis_cache = {}
    
    def analyze_feedback_patterns(self, feedbacks: List[UserFeedback]) -> Dict[str, Any]:
        """피드백 패턴 분석"""
        
        try:
            if not feedbacks:
                return {"status": "no_feedback", "patterns": {}}
            
            analysis = {
                "total_feedbacks": len(feedbacks),
                "feedback_distribution": defaultdict(int),
                "category_analysis": defaultdict(list),
                "sentiment_analysis": {},
                "trend_analysis": {},
                "improvement_areas": [],
                "user_segmentation": {}
            }
            
            # 피드백 분포 분석
            for feedback in feedbacks:
                analysis["feedback_distribution"][feedback.feedback_type.value] += 1
                analysis["category_analysis"][feedback.category.value].append(feedback)
            
            # 카테고리별 상세 분석
            for category, category_feedbacks in analysis["category_analysis"].items():
                ratings = [f.rating for f in category_feedbacks if f.rating is not None]
                if ratings:
                    analysis["category_analysis"][category] = {
                        "count": len(category_feedbacks),
                        "avg_rating": np.mean(ratings),
                        "rating_std": np.std(ratings),
                        "satisfaction_rate": len([r for r in ratings if r >= 4.0]) / len(ratings)
                    }
            
            # 시간별 트렌드 분석
            analysis["trend_analysis"] = self._analyze_temporal_trends(feedbacks)
            
            # 개선 영역 식별
            analysis["improvement_areas"] = self._identify_improvement_areas(feedbacks)
            
            # 사용자 세그먼테이션
            analysis["user_segmentation"] = self._segment_users_by_feedback(feedbacks)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feedback pattern analysis failed: {e}")
            raise SystemException(
                message=f"피드백 패턴 분석 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _analyze_temporal_trends(self, feedbacks: List[UserFeedback]) -> Dict[str, Any]:
        """시간별 트렌드 분석"""
        
        # 일별 평점 트렌드
        daily_ratings = defaultdict(list)
        for feedback in feedbacks:
            if feedback.rating:
                date_key = feedback.timestamp.date().isoformat()
                daily_ratings[date_key].append(feedback.rating)
        
        trends = {}
        for date, ratings in daily_ratings.items():
            trends[date] = {
                "avg_rating": np.mean(ratings),
                "count": len(ratings),
                "satisfaction_rate": len([r for r in ratings if r >= 4.0]) / len(ratings)
            }
        
        return trends
    
    def _identify_improvement_areas(self, feedbacks: List[UserFeedback]) -> List[Dict[str, Any]]:
        """개선 영역 식별"""
        
        improvement_areas = []
        
        # 카테고리별 낮은 점수 영역
        category_scores = defaultdict(list)
        for feedback in feedbacks:
            if feedback.rating:
                category_scores[feedback.category.value].append(feedback.rating)
        
        for category, scores in category_scores.items():
            avg_score = np.mean(scores)
            if avg_score < 3.5:  # 임계값
                improvement_areas.append({
                    "area": category,
                    "current_score": avg_score,
                    "feedback_count": len(scores),
                    "priority": "high" if avg_score < 3.0 else "medium"
                })
        
        # 자주 언급되는 부정적 키워드
        negative_feedbacks = [f for f in feedbacks if f.rating and f.rating < 3.0]
        common_issues = self._extract_common_issues(negative_feedbacks)
        
        for issue in common_issues:
            improvement_areas.append({
                "area": "text_feedback_issue",
                "issue": issue["issue"],
                "frequency": issue["count"],
                "priority": "high" if issue["count"] > 10 else "medium"
            })
        
        return sorted(improvement_areas, key=lambda x: x.get("frequency", 0), reverse=True)
    
    def _extract_common_issues(self, negative_feedbacks: List[UserFeedback]) -> List[Dict[str, Any]]:
        """부정적 피드백에서 공통 이슈 추출 - 고도화된 NLP 분석"""
        
        # 고도화된 키워드 및 패턴 기반 분석
        issue_patterns = {
            "accuracy_issues": {
                "keywords": ["부정확", "틀렸", "wrong", "inaccurate", "맞지 않", "다름", "실제와", "예상과"],
                "patterns": [r"예상[과했]?\s*다르", r"실제[와과]\s*달라", r"정확하지?\s*않", r"틀린?\s*추천"],
                "weight": 1.0,
                "severity": "high"
            },
            "relevance_issues": {
                "keywords": ["관련없", "상관없", "맞지", "어울리지", "적절하지", "어색"],
                "patterns": [r"관련\s*없", r"어울리지\s*않", r"맞지\s*않", r"적절하지\s*않"],
                "weight": 0.9,
                "severity": "high"
            },
            "quality_issues": {
                "keywords": ["품질", "quality", "나빠", "별로", "disappointing", "저품질", "low quality"],
                "patterns": [r"품질이?\s*[나별]", r"quality\s*[is]?\s*poor", r"not\s*good"],
                "weight": 0.8,
                "severity": "medium"
            },
            "usability_issues": {
                "keywords": ["복잡", "어려워", "complex", "complicated", "confusing", "사용하기", "interface"],
                "patterns": [r"사용하기\s*어려", r"복잡해?", r"이해하기\s*힘들"],
                "weight": 0.7,
                "severity": "medium"
            },
            "performance_issues": {
                "keywords": ["느려", "slow", "오래", "걸려", "시간", "응답", "로딩"],
                "patterns": [r"너무\s*느려", r"시간이\s*[많오]래", r"응답이?\s*늦"],
                "weight": 0.8,
                "severity": "high"
            },
            "content_issues": {
                "keywords": ["향이 없어", "노트가 부족", "missing", "lacks", "빠진", "부족"],
                "patterns": [r"노트가?\s*부족", r"향이?\s*없", r"빠진?\s*것"],
                "weight": 0.6,
                "severity": "low"
            },
            "pricing_issues": {
                "keywords": ["비싸", "expensive", "cost", "price", "돈", "가격", "비용"],
                "patterns": [r"너무\s*비싸", r"가격이?\s*[높비]", r"비용이?\s*[많부]"],
                "weight": 0.5,
                "severity": "low"
            }
        }
        
        # 이슈별 점수 계산
        issue_scores = defaultdict(float)
        issue_details = defaultdict(lambda: {"count": 0, "examples": [], "severity": "low", "weight": 0})
        
        import re
        
        for feedback in negative_feedbacks:
            if not feedback.text_feedback:
                continue
                
            text = feedback.text_feedback.lower()
            
            for issue_type, config in issue_patterns.items():
                score = 0.0
                matched_elements = []
                
                # 키워드 매칭
                keyword_matches = 0
                for keyword in config["keywords"]:
                    if keyword in text:
                        keyword_matches += 1
                        matched_elements.append(f"keyword: {keyword}")
                
                if keyword_matches > 0:
                    # 키워드 밀도 기반 점수
                    keyword_density = keyword_matches / len(text.split())
                    score += keyword_density * config["weight"] * 10
                
                # 정규식 패턴 매칭
                pattern_matches = 0
                for pattern in config["patterns"]:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        pattern_matches += len(matches)
                        matched_elements.extend([f"pattern: {match}" for match in matches])
                
                if pattern_matches > 0:
                    score += pattern_matches * config["weight"] * 2
                
                # 감정 분석 보정 (간단한 버전)
                negative_indicators = ["싫어", "나빠", "실망", "짜증", "화가", "불만", "문제"]
                negative_count = sum(1 for indicator in negative_indicators if indicator in text)
                if negative_count > 0:
                    score += negative_count * 0.5
                
                if score > 0:
                    issue_scores[issue_type] += score
                    issue_details[issue_type]["count"] += 1
                    issue_details[issue_type]["severity"] = config["severity"]
                    issue_details[issue_type]["weight"] = config["weight"]
                    
                    # 예시 저장 (최대 3개)
                    if len(issue_details[issue_type]["examples"]) < 3:
                        issue_details[issue_type]["examples"].append({
                            "text": feedback.text_feedback[:100],
                            "score": round(score, 2),
                            "matched_elements": matched_elements[:3]
                        })
        
        # 결과 생성
        results = []
        for issue_type, total_score in issue_scores.items():
            details = issue_details[issue_type]
            results.append({
                "issue": issue_type,
                "total_score": round(total_score, 2),
                "count": details["count"],
                "severity": details["severity"],
                "priority": round(total_score * {"high": 1.5, "medium": 1.0, "low": 0.7}[details["severity"]], 2),
                "examples": details["examples"],
                "recommendation": self._get_issue_recommendation(issue_type, details)
            })
        
        return sorted(results, key=lambda x: x["priority"], reverse=True)
    
    def _get_issue_recommendation(self, issue_type: str, details: Dict[str, Any]) -> str:
        """이슈별 개선 권장사항 생성"""
        recommendations = {
            "accuracy_issues": "AI 모델의 정확도 향상을 위한 추가 학습 데이터 수집 및 모델 재훈련이 필요합니다.",
            "relevance_issues": "사용자 취향 분석 알고리즘 개선과 개인화 추천 로직 강화가 필요합니다.",
            "quality_issues": "추천 결과의 품질 검증 프로세스 강화와 전문가 검수 시스템 도입을 권장합니다.",
            "usability_issues": "사용자 인터페이스 개선과 사용성 테스트를 통한 UX 최적화가 필요합니다.",
            "performance_issues": "시스템 성능 최적화와 인프라 확장을 검토해야 합니다.",
            "content_issues": "향수 데이터베이스 확장과 노트 정보의 완성도 향상이 필요합니다.",
            "pricing_issues": "가격 정책 재검토와 가성비 개선 방안을 모색해야 합니다."
        }
        
        base_recommendation = recommendations.get(issue_type, "해당 이슈에 대한 상세 분석과 개선 방안 수립이 필요합니다.")
        
        # 심각도에 따른 우선순위 추가
        if details["severity"] == "high":
            return f"[긴급] {base_recommendation} 즉시 대응이 필요합니다."
        elif details["severity"] == "medium":
            return f"[중요] {base_recommendation} 단기간 내 개선을 권장합니다."
        else:
            return f"[일반] {base_recommendation} 장기적 개선 계획에 포함시켜 주세요."
    
    def _segment_users_by_feedback(self, feedbacks: List[UserFeedback]) -> Dict[str, Any]:
        """피드백 기반 사용자 세그먼테이션"""
        
        user_profiles = {}
        
        # 사용자별 피드백 집계
        for feedback in feedbacks:
            if feedback.user_id not in user_profiles:
                user_profiles[feedback.user_id] = {
                    "feedback_count": 0,
                    "avg_rating": 0,
                    "categories": defaultdict(int),
                    "engagement_level": "low"
                }
            
            profile = user_profiles[feedback.user_id]
            profile["feedback_count"] += 1
            
            if feedback.rating:
                current_avg = profile["avg_rating"]
                count = profile["feedback_count"]
                profile["avg_rating"] = (current_avg * (count-1) + feedback.rating) / count
            
            profile["categories"][feedback.category.value] += 1
        
        # 사용자 세그먼트 분류
        segments = {
            "power_users": [],      # 피드백 많음, 평점 높음
            "critics": [],          # 피드백 많음, 평점 낮음
            "satisfied": [],        # 피드백 보통, 평점 높음
            "casual": [],           # 피드백 적음
            "churned": []           # 초기엔 활동적이었으나 최근 비활성
        }
        
        for user_id, profile in user_profiles.items():
            if profile["feedback_count"] >= 10:
                if profile["avg_rating"] >= 4.0:
                    segments["power_users"].append(user_id)
                else:
                    segments["critics"].append(user_id)
            elif profile["avg_rating"] >= 4.0:
                segments["satisfied"].append(user_id)
            else:
                segments["casual"].append(user_id)
        
        return {segment: len(users) for segment, users in segments.items()}
    
    def predict_user_satisfaction(self, user_features: Dict[str, Any]) -> float:
        """사용자 만족도 예측"""
        
        try:
            # 특성 벡터 생성 (예시)
            feature_vector = [
                user_features.get("previous_ratings_avg", 3.0),
                user_features.get("usage_frequency", 1.0),
                user_features.get("recipe_complexity_preference", 5.0),
                user_features.get("price_sensitivity", 3.0),
                len(user_features.get("preferred_notes", [])),
            ]
            
            # 고도화된 만족도 예측 알고리즘
            if hasattr(self.ml_model, 'predict') and self.ml_model is not None:
                try:
                    prediction = self.ml_model.predict([feature_vector])[0]
                    return max(1.0, min(5.0, prediction))
                except Exception as model_error:
                    logger.warning(f"ML model prediction failed: {model_error}, using advanced heuristic")
            
            # 고도화된 휴리스틱 모델
            satisfaction_score = 3.0  # 기준점
            
            # 1. 사용자 만족도 기반 보정
            avg_rating = feature_vector[0]
            if avg_rating >= 4.5:
                satisfaction_score += 1.5
            elif avg_rating >= 4.0:
                satisfaction_score += 1.0
            elif avg_rating >= 3.5:
                satisfaction_score += 0.5
            elif avg_rating < 2.5:
                satisfaction_score -= 1.0
            elif avg_rating < 2.0:
                satisfaction_score -= 1.5
            
            # 2. 사용 빈도 보정 (더 자주 사용하는 사용자는 만족도 높음)
            usage_freq = feature_vector[1]
            if usage_freq >= 3.0:
                satisfaction_score += 0.5
            elif usage_freq <= 1.0:
                satisfaction_score -= 0.3
            
            # 3. 복잡성 선호도와 가격 민감도의 균형
            complexity_pref = feature_vector[2]
            price_sensitivity = feature_vector[3]
            
            # 복잡한 향수를 선호하지만 가격에 민감하지 않은 경우
            if complexity_pref >= 7.0 and price_sensitivity <= 2.0:
                satisfaction_score += 0.7
            # 단순한 향수를 선호하고 가격에 민감한 경우  
            elif complexity_pref <= 3.0 and price_sensitivity >= 4.0:
                satisfaction_score += 0.3
            # 복잡한 향수를 원하지만 가격에 매우 민감한 경우 (갈등)
            elif complexity_pref >= 7.0 and price_sensitivity >= 4.0:
                satisfaction_score -= 0.5
            
            # 4. 선호 노트 다양성 보정
            preferred_notes_count = feature_vector[4]
            if preferred_notes_count >= 8:  # 다양한 취향
                satisfaction_score += 0.4
            elif preferred_notes_count <= 2:  # 매우 제한적 취향
                satisfaction_score -= 0.2
            
            # 5. 시간적 패턴 고려 (최근 피드백 가중치)
            recent_feedback_weight = 1.0
            if len(user_features.get("recent_ratings", [])) > 0:
                recent_ratings = user_features["recent_ratings"][-5:]  # 최근 5개
                recent_trend = np.mean(recent_ratings) if recent_ratings else 3.0
                
                # 최근 평가가 향상되고 있으면 만족도 상승 예상
                if len(recent_ratings) >= 3:
                    trend_slope = (recent_ratings[-1] - recent_ratings[0]) / len(recent_ratings)
                    if trend_slope > 0.5:
                        satisfaction_score += 0.6  # 긍정적 트렌드
                    elif trend_slope < -0.5:
                        satisfaction_score -= 0.4  # 부정적 트렌드
                
                # 최근 평가의 변동성
                if len(recent_ratings) >= 2:
                    volatility = np.std(recent_ratings)
                    if volatility > 1.5:
                        satisfaction_score -= 0.3  # 높은 변동성은 불만족 징조
            
            # 6. 사용자 유형별 보정
            user_type = self._classify_user_type(user_features)
            type_adjustments = {
                "expert": 0.3,      # 전문가는 까다롭지만 좋은 추천에 높은 만족
                "enthusiast": 0.2,  # 애호가는 전반적으로 만족도 높음
                "casual": 0.0,      # 일반 사용자는 기준점
                "price_sensitive": -0.2,  # 가격 민감형은 약간 낮은 만족도
                "new_user": -0.1    # 신규 사용자는 적응 기간
            }
            satisfaction_score += type_adjustments.get(user_type, 0.0)
            
            # 7. 계절성 및 트렌드 보정
            import datetime
            current_month = datetime.datetime.now().month
            if current_month in [11, 12, 1, 2]:  # 겨울
                if "warm" in str(user_features.get("preferred_notes", [])).lower():
                    satisfaction_score += 0.2
            elif current_month in [6, 7, 8]:  # 여름  
                if "fresh" in str(user_features.get("preferred_notes", [])).lower():
                    satisfaction_score += 0.2
            
            return max(1.0, min(5.0, satisfaction_score))
            
        except Exception as e:
            logger.error(f"Advanced satisfaction prediction failed: {e}")
            return 3.0
    
    def _classify_user_type(self, user_features: Dict[str, Any]) -> str:
        """사용자 유형 분류"""
        try:
            feedback_count = user_features.get("feedback_count", 0)
            avg_rating = user_features.get("average_rating", 3.0)
            usage_freq = user_features.get("usage_frequency", 1.0)
            price_sensitivity = user_features.get("price_sensitivity", 3.0)
            preferred_notes_count = len(user_features.get("preferred_notes", []))
            
            # 신규 사용자
            if feedback_count <= 3:
                return "new_user"
            
            # 가격 민감형
            if price_sensitivity >= 4.5:
                return "price_sensitive"
            
            # 전문가 (많은 피드백, 다양한 취향, 까다로운 평가)
            if (feedback_count >= 20 and 
                preferred_notes_count >= 8 and 
                avg_rating <= 3.5 and 
                usage_freq >= 2.0):
                return "expert"
            
            # 애호가 (높은 사용빈도, 높은 만족도)
            if usage_freq >= 3.0 and avg_rating >= 4.0:
                return "enthusiast"
            
            # 일반 사용자
            return "casual"
            
        except Exception as e:
            logger.warning(f"User type classification failed: {e}")
            return "casual"


class FeedbackService:
    """피드백 서비스"""
    
    def __init__(self):
        self.analyzer = FeedbackAnalyzer()
        self.feedback_buffer: List[UserFeedback] = []
        self.buffer_max_size = 1000
        
        # A/B 테스트 설정
        self.ab_test_variants = {
            "recipe_generation": ["v1", "v2", "v3"],
            "search_algorithm": ["semantic", "hybrid", "collaborative"],
            "ui_layout": ["classic", "modern", "minimal"]
        }
        self.ab_test_assignments = {}
    
    async def collect_feedback(
        self,
        user_id: str,
        feedback_data: Dict[str, Any]
    ) -> str:
        """피드백 수집"""
        
        try:
            feedback = UserFeedback(
                feedback_id=str(uuid.uuid4()),
                user_id=user_id,
                session_id=feedback_data.get("session_id", str(uuid.uuid4())),
                feedback_type=FeedbackType(feedback_data["feedback_type"]),
                category=FeedbackCategory(feedback_data["category"]),
                target_type=feedback_data["target_type"],
                target_id=feedback_data["target_id"],
                rating=feedback_data.get("rating"),
                binary_feedback=feedback_data.get("binary_feedback"),
                detailed_scores=feedback_data.get("detailed_scores"),
                text_feedback=feedback_data.get("text_feedback"),
                tags=feedback_data.get("tags", []),
                user_profile=feedback_data.get("user_profile"),
                interaction_context=feedback_data.get("interaction_context"),
                timestamp=datetime.utcnow(),
                weight=self._calculate_feedback_weight(user_id, feedback_data)
            )
            
            # 피드백 검증
            self._validate_feedback(feedback)
            
            # 버퍼에 추가
            self.feedback_buffer.append(feedback)
            
            # 데이터베이스에 저장
            await self._store_feedback(feedback)
            
            # 버퍼 크기 관리
            if len(self.feedback_buffer) > self.buffer_max_size:
                await self._process_feedback_batch()
            
            # 실시간 분석 (중요한 피드백인 경우)
            if feedback.rating and feedback.rating <= 2.0:
                await self._handle_negative_feedback(feedback)
            
            logger.info(f"Feedback collected: {feedback.feedback_id}")
            
            performance_logger.log_execution_time(
                operation="collect_feedback",
                execution_time=0.0,
                success=True,
                extra_data={
                    "user_id": user_id,
                    "feedback_type": feedback.feedback_type.value,
                    "rating": feedback.rating
                }
            )
            
            return feedback.feedback_id
            
        except Exception as e:
            logger.error(f"Feedback collection failed: {e}")
            raise SystemException(
                message=f"피드백 수집 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    def _validate_feedback(self, feedback: UserFeedback):
        """피드백 검증"""
        
        if feedback.rating is not None:
            if not (1.0 <= feedback.rating <= 5.0):
                raise ValidationException("평점은 1-5 사이여야 합니다")
        
        if feedback.text_feedback:
            if len(feedback.text_feedback) > 1000:
                raise ValidationException("텍스트 피드백은 1000자를 넘을 수 없습니다")
    
    def _calculate_feedback_weight(self, user_id: str, feedback_data: Dict[str, Any]) -> float:
        """피드백 가중치 계산"""
        
        weight = 1.0
        
        # 사용자 신뢰도 기반 가중치
        user_profile = feedback_data.get("user_profile", {})
        
        # 전문가 사용자 가중치 증가
        if user_profile.get("is_expert", False):
            weight *= 2.0
        
        # 활성 사용자 가중치 증가
        usage_frequency = user_profile.get("usage_frequency", 1)
        if usage_frequency > 10:
            weight *= 1.5
        elif usage_frequency > 50:
            weight *= 2.0
        
        # 상세 피드백 가중치 증가
        if feedback_data.get("detailed_scores"):
            weight *= 1.3
        
        if feedback_data.get("text_feedback"):
            weight *= 1.2
        
        return min(weight, 3.0)  # 최대 3배
    
    async def _store_feedback(self, feedback: UserFeedback):
        """피드백 데이터베이스 저장"""
        
        try:
            # 실제 구현에서는 데이터베이스에 저장
            # with get_db_session() as db:
            #     db_feedback = DatabaseFeedback(**feedback_dict)
            #     db.add(db_feedback)
            #     db.commit()
            
            logger.debug(f"Feedback stored: {feedback.feedback_id}")
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            # 중요한 피드백이므로 재시도 메커니즘 필요
            raise
    
    async def _handle_negative_feedback(self, feedback: UserFeedback):
        """부정적 피드백 처리"""
        
        try:
            # 즉시 알림 (심각한 문제인 경우)
            if feedback.rating <= 1.0:
                logger.warning(f"Critical negative feedback received: {feedback.feedback_id}")
                # 관리자 알림, 고객 서비스 팀 통지 등
            
            # 자동 개선 제안 생성
            improvement_suggestions = await self._generate_improvement_suggestions(feedback)
            
            # 사용자에게 후속 조치 안내
            await self._send_follow_up_response(feedback, improvement_suggestions)
            
        except Exception as e:
            logger.error(f"Failed to handle negative feedback: {e}")
    
    async def _generate_improvement_suggestions(self, feedback: UserFeedback) -> List[str]:
        """개선 제안 생성"""
        
        suggestions = []
        
        # 피드백 카테고리별 제안
        if feedback.category == FeedbackCategory.RECIPE_QUALITY:
            suggestions.extend([
                "레시피 생성 알고리즘 조정",
                "향료 비율 밸런싱 개선",
                "품질 검증 프로세스 강화"
            ])
        elif feedback.category == FeedbackCategory.PERSONAL_TASTE:
            suggestions.extend([
                "개인화 모델 재훈련",
                "사용자 프로필 데이터 수집 확대",
                "취향 분석 정확도 개선"
            ])
        
        return suggestions
    
    async def _send_follow_up_response(
        self, 
        feedback: UserFeedback, 
        suggestions: List[str]
    ):
        """후속 응답 발송"""
        
        # 사용자에게 개선 조치 안내
        response = {
            "message": "소중한 피드백 감사합니다. 다음과 같이 개선하겠습니다:",
            "improvements": suggestions,
            "follow_up_date": (datetime.utcnow() + timedelta(days=7)).isoformat(),
            "compensation": self._calculate_compensation(feedback)
        }
        
        logger.info(f"Follow-up response prepared for user {feedback.user_id}")
        # 실제로는 알림 서비스를 통해 사용자에게 전송
    
    def _calculate_compensation(self, feedback: UserFeedback) -> Optional[Dict[str, Any]]:
        """보상 계산"""
        
        if feedback.rating and feedback.rating <= 2.0:
            return {
                "type": "service_credit",
                "amount": 10,  # 10회 무료 생성
                "expiry_days": 30
            }
        
        return None
    
    async def _process_feedback_batch(self):
        """피드백 배치 처리"""
        
        try:
            if not self.feedback_buffer:
                return
            
            logger.info(f"Processing feedback batch: {len(self.feedback_buffer)} items")
            
            # 피드백 분석
            analysis = self.analyzer.analyze_feedback_patterns(self.feedback_buffer)
            
            # 모델 성능 평가
            performance_metrics = await self._evaluate_model_performance()
            
            # 개선 영역 식별 및 처리
            await self._process_improvement_areas(analysis["improvement_areas"])
            
            # A/B 테스트 결과 분석
            await self._analyze_ab_test_results()
            
            # 버퍼 클리어
            self.feedback_buffer.clear()
            
            performance_logger.log_execution_time(
                operation="process_feedback_batch",
                execution_time=0.0,
                success=True,
                extra_data={
                    "batch_size": len(self.feedback_buffer),
                    "improvement_areas": len(analysis["improvement_areas"])
                }
            )
            
        except Exception as e:
            logger.error(f"Feedback batch processing failed: {e}")
    
    async def _evaluate_model_performance(self) -> ModelPerformanceMetrics:
        """모델 성능 평가"""
        
        try:
            # 최근 피드백 데이터 수집
            recent_feedbacks = [
                f for f in self.feedback_buffer 
                if f.timestamp > datetime.utcnow() - timedelta(days=7)
            ]
            
            if not recent_feedbacks:
                return None
            
            # 성능 메트릭 계산
            ratings = [f.rating for f in recent_feedbacks if f.rating is not None]
            
            metrics = ModelPerformanceMetrics(
                model_name="FragranceAI_v1",
                model_version="1.0.0",
                evaluation_date=datetime.utcnow(),
                accuracy=0.85,  # 실제 계산 로직 필요
                precision=0.83,
                recall=0.87,
                f1_score=0.85,
                avg_user_rating=np.mean(ratings) if ratings else 3.0,
                satisfaction_rate=len([r for r in ratings if r >= 4.0]) / len(ratings) if ratings else 0.5,
                engagement_rate=0.75,  # 실제 계산 필요
                conversion_rate=0.15,  # 실제 계산 필요
                retention_rate=0.68,   # 실제 계산 필요
                category_performance={},  # 실제 계산 필요
                user_segment_performance={},  # 실제 계산 필요
                improvement_suggestions=[]
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model performance evaluation failed: {e}")
            return None
    
    async def _process_improvement_areas(self, improvement_areas: List[Dict[str, Any]]):
        """개선 영역 처리"""
        
        for area in improvement_areas:
            if area.get("priority") == "high":
                logger.warning(f"High priority improvement needed: {area['area']}")
                # 자동 개선 작업 트리거
                await self._trigger_auto_improvement(area)
    
    async def _trigger_auto_improvement(self, improvement_area: Dict[str, Any]):
        """자동 개선 트리거"""
        
        area_type = improvement_area["area"]
        
        if area_type == "recipe_quality":
            # 레시피 생성 모델 재훈련 스케줄링
            logger.info("Scheduling recipe generation model retraining")
        elif area_type == "personal_taste":
            # 개인화 모델 업데이트
            logger.info("Updating personalization model")
        
        # 실제로는 백그라운드 작업 큐에 추가
    
    async def _analyze_ab_test_results(self):
        """A/B 테스트 결과 분석"""
        
        try:
            # 각 변형별 성능 분석
            for test_name, variants in self.ab_test_variants.items():
                variant_performance = {}
                
                for variant in variants:
                    # 해당 변형에 대한 피드백 분석
                    variant_feedbacks = [
                        f for f in self.feedback_buffer
                        if f.interaction_context and 
                        f.interaction_context.get(f"{test_name}_variant") == variant
                    ]
                    
                    if variant_feedbacks:
                        ratings = [f.rating for f in variant_feedbacks if f.rating]
                        variant_performance[variant] = {
                            "avg_rating": np.mean(ratings) if ratings else 0,
                            "feedback_count": len(variant_feedbacks),
                            "satisfaction_rate": len([r for r in ratings if r >= 4.0]) / len(ratings) if ratings else 0
                        }
                
                # 최고 성능 변형 식별
                if variant_performance:
                    best_variant = max(
                        variant_performance.keys(),
                        key=lambda v: variant_performance[v]["avg_rating"]
                    )
                    
                    logger.info(f"A/B test winner for {test_name}: {best_variant}")
                    
        except Exception as e:
            logger.error(f"A/B test analysis failed: {e}")
    
    def assign_ab_test_variant(self, user_id: str, test_name: str) -> str:
        """A/B 테스트 변형 할당"""
        
        if test_name not in self.ab_test_variants:
            return "default"
        
        # 사용자별 일관된 할당을 위한 해시 기반 할당
        user_hash = hash(f"{user_id}_{test_name}") % len(self.ab_test_variants[test_name])
        variant = self.ab_test_variants[test_name][user_hash]
        
        # 할당 기록
        self.ab_test_assignments[f"{user_id}_{test_name}"] = variant
        
        return variant
    
    async def get_feedback_insights(
        self,
        time_range_days: int = 30,
        user_segment: Optional[str] = None
    ) -> Dict[str, Any]:
        """피드백 인사이트 조회"""
        
        try:
            # 지정된 기간의 피드백 분석
            cutoff_date = datetime.utcnow() - timedelta(days=time_range_days)
            relevant_feedbacks = [
                f for f in self.feedback_buffer
                if f.timestamp > cutoff_date
            ]
            
            if user_segment:
                # 사용자 세그먼트별 피드백 필터링
                segment_filters = {
                    "premium_users": lambda f: getattr(f, 'user_tier', 'basic') == 'premium',
                    "new_users": lambda f: (datetime.utcnow() - getattr(f, 'user_signup_date', datetime.utcnow())).days <= 30,
                    "power_users": lambda f: getattr(f, 'user_activity_level', 'low') == 'high',
                    "dissatisfied_users": lambda f: f.rating and f.rating < 3.0,
                    "satisfied_users": lambda f: f.rating and f.rating >= 4.0
                }

                if user_segment in segment_filters:
                    relevant_feedbacks = [
                        f for f in relevant_feedbacks
                        if segment_filters[user_segment](f)
                    ]
                    logger.info(f"Filtered feedbacks for segment '{user_segment}': {len(relevant_feedbacks)} items")
            
            # 종합 인사이트 생성
            insights = {
                "summary": {
                    "total_feedback": len(relevant_feedbacks),
                    "avg_rating": np.mean([f.rating for f in relevant_feedbacks if f.rating]),
                    "satisfaction_rate": len([f for f in relevant_feedbacks if f.rating and f.rating >= 4.0]) / len(relevant_feedbacks) if relevant_feedbacks else 0
                },
                "trends": self.analyzer.analyze_feedback_patterns(relevant_feedbacks),
                "recommendations": await self._generate_actionable_recommendations(relevant_feedbacks),
                "model_performance": await self._evaluate_model_performance()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate feedback insights: {e}")
            raise SystemException(
                message=f"피드백 인사이트 생성 실패: {str(e)}",
                error_code=ErrorCode.SYSTEM_ERROR,
                cause=e
            )
    
    async def _generate_actionable_recommendations(
        self, 
        feedbacks: List[UserFeedback]
    ) -> List[Dict[str, Any]]:
        """실행 가능한 권장사항 생성"""
        
        recommendations = []
        
        # 낮은 평점 카테고리 개선
        category_scores = defaultdict(list)
        for feedback in feedbacks:
            if feedback.rating:
                category_scores[feedback.category.value].append(feedback.rating)
        
        for category, scores in category_scores.items():
            avg_score = np.mean(scores)
            if avg_score < 3.5:
                recommendations.append({
                    "type": "improvement",
                    "category": category,
                    "priority": "high" if avg_score < 3.0 else "medium",
                    "current_score": avg_score,
                    "target_score": 4.0,
                    "actions": [
                        f"{category} 관련 알고리즘 개선",
                        f"{category} 사용자 경험 최적화",
                        f"{category} 품질 검증 강화"
                    ]
                })
        
        return recommendations


# 전역 피드백 서비스 인스턴스
feedback_service = FeedbackService()