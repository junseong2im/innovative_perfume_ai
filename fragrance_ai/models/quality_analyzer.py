"""
향수 품질 평가 및 조향 분석 AI 시스템
마스터 조향사급 품질 평가와 전문적 조향 분석을 제공
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from ..knowledge.master_perfumer_principles import MasterPerfumerKnowledge
from .advanced_blending_ai import AdvancedBlendingAI, BlendPrediction
from ..core.config import settings

logger = logging.getLogger(__name__)

class QualityMetric(Enum):
    """품질 평가 지표"""
    HARMONY = "harmony"
    COMPLEXITY = "complexity"
    LONGEVITY = "longevity"
    PROJECTION = "projection"
    UNIQUENESS = "uniqueness"
    WEARABILITY = "wearability"
    COMMERCIAL_APPEAL = "commercial_appeal"
    TECHNICAL_EXCELLENCE = "technical_excellence"

@dataclass
class QualityAssessment:
    """품질 평가 결과"""
    overall_score: float
    grade: str  # S, A+, A, B+, B, C+, C, D
    metric_scores: Dict[QualityMetric, float]
    strengths: List[str]
    weaknesses: List[str]
    improvement_suggestions: List[str]
    perfumer_level_assessment: str
    market_positioning: Dict[str, Any]
    detailed_analysis: Dict[str, Any]
    confidence_score: float

@dataclass
class FragranceProfile:
    """향수 프로필 분석 결과"""
    fragrance_id: str
    ingredients: List[str]
    proportions: Optional[List[float]]
    style_classification: Dict[str, float]
    personality_traits: List[str]
    mood_descriptors: List[str]
    seasonal_suitability: Dict[str, float]
    occasion_matching: Dict[str, float]
    demographic_appeal: Dict[str, float]
    sensory_experience: Dict[str, Any]
    evolution_timeline: Dict[str, Any]
    comparison_benchmark: Dict[str, Any]

@dataclass
class TechnicalAnalysis:
    """기술적 분석 결과"""
    molecular_analysis: Dict[str, Any]
    chemical_stability: Dict[str, float]
    volatility_curve: Dict[str, Any]
    interaction_effects: List[Dict[str, Any]]
    aging_potential: Dict[str, Any]
    formulation_insights: List[str]
    manufacturing_considerations: Dict[str, Any]


class QualityNeuralNet(nn.Module):
    """품질 평가를 위한 신경망"""

    def __init__(self, input_dim: int, num_metrics: int = 8):
        super().__init__()

        # 특성 추출층
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)

        # 품질 지표별 전문 평가 헤드
        self.quality_heads = nn.ModuleDict({
            'harmony': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'complexity': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'longevity': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'projection': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'uniqueness': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'wearability': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'commercial_appeal': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()),
            'technical_excellence': nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        })

        # 전체 품질 점수 헤드
        self.overall_head = nn.Sequential(
            nn.Linear(128 + num_metrics, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 특성 추출
        features = self.feature_layers(x)

        # 어텐션 적용 (시퀀스 차원 추가)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        attn_features, attn_weights = self.attention(features, features, features)
        attn_features = attn_features.squeeze(1)

        # 각 품질 지표 예측
        quality_scores = {}
        metric_outputs = []

        for metric, head in self.quality_heads.items():
            score = head(attn_features)
            quality_scores[metric] = score
            metric_outputs.append(score)

        # 메트릭 점수들을 결합
        combined_metrics = torch.cat(metric_outputs, dim=1)

        # 전체 점수 계산
        overall_input = torch.cat([attn_features, combined_metrics], dim=1)
        overall_score = self.overall_head(overall_input)
        quality_scores['overall'] = overall_score

        return quality_scores


class FragranceQualityAnalyzer:
    """마스터 조향사급 향수 품질 분석기"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 필요한 컴포넌트들 초기화
        self.perfumer_knowledge = MasterPerfumerKnowledge()
        self.blending_ai = AdvancedBlendingAI()

        # 품질 평가 모델
        self.quality_model = None
        self.scaler = StandardScaler()

        # 벤치마크 데이터베이스 (명작 향수들의 기준점)
        self.benchmark_fragrances = self._initialize_benchmarks()

        # 품질 등급 기준
        self.grade_thresholds = {
            'S': 0.95,    # 마스터피스
            'A+': 0.90,  # 탁월함
            'A': 0.85,   # 우수함
            'B+': 0.80,  # 양호함 상
            'B': 0.70,   # 양호함
            'C+': 0.60,  # 보통 상
            'C': 0.50,   # 보통
            'D': 0.0     # 개선필요
        }

        logger.info("Fragrance Quality Analyzer initialized")

    def _initialize_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """벤치마크 향수 데이터베이스 초기화"""
        return {
            # 클래식 마스터피스들
            "chanel_no5": {
                "name": "Chanel No.5",
                "perfumer": "Ernest Beaux",
                "year": 1921,
                "style": "aldehydic_floral",
                "benchmark_scores": {
                    QualityMetric.HARMONY: 0.98,
                    QualityMetric.COMPLEXITY: 0.95,
                    QualityMetric.LONGEVITY: 0.90,
                    QualityMetric.PROJECTION: 0.85,
                    QualityMetric.UNIQUENESS: 0.99,
                    QualityMetric.WEARABILITY: 0.80,
                    QualityMetric.COMMERCIAL_APPEAL: 0.95,
                    QualityMetric.TECHNICAL_EXCELLENCE: 0.98
                },
                "signature_ingredients": ["aldehydes", "ylang-ylang", "rose", "jasmine", "sandalwood", "vanilla"]
            },

            "terre_d_hermes": {
                "name": "Terre d'Hermès",
                "perfumer": "Jean-Claude Ellena",
                "year": 2006,
                "style": "mineral_woody",
                "benchmark_scores": {
                    QualityMetric.HARMONY: 0.96,
                    QualityMetric.COMPLEXITY: 0.85,
                    QualityMetric.LONGEVITY: 0.85,
                    QualityMetric.PROJECTION: 0.75,
                    QualityMetric.UNIQUENESS: 0.90,
                    QualityMetric.WEARABILITY: 0.95,
                    QualityMetric.COMMERCIAL_APPEAL: 0.88,
                    QualityMetric.TECHNICAL_EXCELLENCE: 0.92
                },
                "signature_ingredients": ["orange", "grapefruit", "flint", "vetiver", "cedar", "benzoin"]
            },

            "amouage_gold": {
                "name": "Amouage Gold",
                "perfumer": "Guy Robert",
                "year": 1983,
                "style": "oriental_floral",
                "benchmark_scores": {
                    QualityMetric.HARMONY: 0.93,
                    QualityMetric.COMPLEXITY: 0.98,
                    QualityMetric.LONGEVITY: 0.95,
                    QualityMetric.PROJECTION: 0.90,
                    QualityMetric.UNIQUENESS: 0.95,
                    QualityMetric.WEARABILITY: 0.70,
                    QualityMetric.COMMERCIAL_APPEAL: 0.75,
                    QualityMetric.TECHNICAL_EXCELLENCE: 0.95
                },
                "signature_ingredients": ["rose", "jasmine", "frankincense", "myrrh", "civet", "ambergris"]
            }
        }

    def analyze_fragrance_quality(
        self,
        ingredients: List[str],
        proportions: Optional[List[float]] = None,
        fragrance_name: Optional[str] = None
    ) -> QualityAssessment:
        """종합적인 향수 품질 분석"""

        try:
            # 기본 블렌드 예측
            blend_prediction = self.blending_ai.predict_blend_quality(ingredients, proportions)

            # 각 품질 지표별 상세 분석
            metric_scores = {}

            # 조화도 평가
            metric_scores[QualityMetric.HARMONY] = self._analyze_harmony(ingredients, blend_prediction)

            # 복잡성 평가
            metric_scores[QualityMetric.COMPLEXITY] = self._analyze_complexity(ingredients)

            # 지속성 평가
            metric_scores[QualityMetric.LONGEVITY] = self._analyze_longevity(ingredients)

            # 프로젝션 평가
            metric_scores[QualityMetric.PROJECTION] = self._analyze_projection(ingredients)

            # 독창성 평가
            metric_scores[QualityMetric.UNIQUENESS] = self._analyze_uniqueness(ingredients)

            # 착용감 평가
            metric_scores[QualityMetric.WEARABILITY] = self._analyze_wearability(ingredients)

            # 상업적 어필 평가
            metric_scores[QualityMetric.COMMERCIAL_APPEAL] = self._analyze_commercial_appeal(ingredients)

            # 기술적 탁월성 평가
            metric_scores[QualityMetric.TECHNICAL_EXCELLENCE] = self._analyze_technical_excellence(ingredients, blend_prediction)

            # 전체 점수 계산 (가중 평균)
            weights = {
                QualityMetric.HARMONY: 0.20,
                QualityMetric.COMPLEXITY: 0.15,
                QualityMetric.LONGEVITY: 0.15,
                QualityMetric.PROJECTION: 0.10,
                QualityMetric.UNIQUENESS: 0.15,
                QualityMetric.WEARABILITY: 0.10,
                QualityMetric.COMMERCIAL_APPEAL: 0.10,
                QualityMetric.TECHNICAL_EXCELLENCE: 0.05
            }

            overall_score = sum(metric_scores[metric] * weight for metric, weight in weights.items())

            # 등급 결정
            grade = self._determine_grade(overall_score)

            # 강점/약점 분석
            strengths, weaknesses = self._analyze_strengths_weaknesses(metric_scores)

            # 개선 제안
            improvement_suggestions = self._generate_improvement_suggestions(metric_scores, ingredients)

            # 조향사 수준 평가
            perfumer_level = self._assess_perfumer_level(overall_score, metric_scores)

            # 시장 포지셔닝
            market_positioning = self._analyze_market_positioning(metric_scores, ingredients)

            # 상세 분석
            detailed_analysis = self._generate_detailed_analysis(ingredients, metric_scores, blend_prediction)

            # 신뢰도 점수
            confidence_score = self._calculate_confidence_score(ingredients, metric_scores)

            return QualityAssessment(
                overall_score=overall_score,
                grade=grade,
                metric_scores=metric_scores,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=improvement_suggestions,
                perfumer_level_assessment=perfumer_level,
                market_positioning=market_positioning,
                detailed_analysis=detailed_analysis,
                confidence_score=confidence_score
            )

        except Exception as e:
            logger.error(f"Failed to analyze fragrance quality: {e}")
            # 기본 평가 반환
            return QualityAssessment(
                overall_score=0.5,
                grade="C",
                metric_scores={metric: 0.5 for metric in QualityMetric},
                strengths=[],
                weaknesses=["분석 오류 발생"],
                improvement_suggestions=["다시 시도하십시오"],
                perfumer_level_assessment="분석 불가",
                market_positioning={},
                detailed_analysis={"error": str(e)},
                confidence_score=0.0
            )

    def _analyze_harmony(self, ingredients: List[str], blend_prediction: BlendPrediction) -> float:
        """조화도 분석"""
        base_harmony = blend_prediction.harmony_score

        # 마스터 조향사 조화 법칙 적용
        harmony_bonus = 0.0
        for rule in self.perfumer_knowledge.harmony_rules:
            for combo in rule.ingredient_combinations:
                if any(ing in ingredients for ing in combo):
                    harmony_bonus += rule.harmony_strength * 0.1

        # 화학적 호환성 고려
        chemical_compatibility = np.mean(list(blend_prediction.chemical_compatibility.values())) if blend_prediction.chemical_compatibility else 0.5

        final_harmony = min(1.0, base_harmony + harmony_bonus * 0.2 + chemical_compatibility * 0.1)
        return final_harmony

    def _analyze_complexity(self, ingredients: List[str]) -> float:
        """복잡성 분석"""
        complexity_analysis = self.perfumer_knowledge.calculate_fragrance_complexity(ingredients)

        # 복잡성 점수 매핑
        complexity_scores = {
            "simple": 0.3,
            "moderate": 0.6,
            "complex": 0.8,
            "haute_couture": 0.95
        }

        base_score = complexity_scores.get(complexity_analysis["complexity_level"], 0.5)

        # 조화도 잠재력으로 조정
        harmony_adjustment = complexity_analysis["harmony_potential"] * 0.2

        return min(1.0, base_score + harmony_adjustment)

    def _analyze_longevity(self, ingredients: List[str]) -> float:
        """지속성 분석"""
        if not ingredients:
            return 0.0

        # 향료별 지속성 데이터 (실제로는 데이터베이스에서 가져와야 함)
        longevity_weights = {
            # 높은 지속성
            "바닐라": 0.95, "머스크": 0.90, "앰버": 0.90, "샌달우드": 0.85, "패촐리": 0.85,
            "우드": 0.80, "벤조인": 0.80, "사향": 0.95, "용연향": 0.95,

            # 중간 지속성
            "장미": 0.65, "자스민": 0.70, "라벤더": 0.60, "제라늄": 0.60,
            "일랑일랑": 0.65, "시더우드": 0.75,

            # 낮은 지속성
            "시트러스": 0.25, "레몬": 0.20, "베르가못": 0.30, "자몽": 0.25,
            "라임": 0.20, "오렌지": 0.25, "민트": 0.30
        }

        total_longevity = 0.0
        matched_ingredients = 0

        for ingredient in ingredients:
            # 정확한 매칭 또는 부분 매칭
            longevity = 0.5  # 기본값

            for known_ingredient, weight in longevity_weights.items():
                if known_ingredient.lower() in ingredient.lower() or ingredient.lower() in known_ingredient.lower():
                    longevity = weight
                    break

            total_longevity += longevity
            matched_ingredients += 1

        return total_longevity / matched_ingredients if matched_ingredients > 0 else 0.5

    def _analyze_projection(self, ingredients: List[str]) -> float:
        """프로젝션(확산력) 분석"""
        if not ingredients:
            return 0.0

        # 향료별 프로젝션 데이터
        projection_weights = {
            # 강한 프로젝션
            "앰버": 0.90, "머스크": 0.85, "패촐리": 0.90, "인센스": 0.85,
            "일랑일랑": 0.80, "튜베로즈": 0.90, "자스민": 0.85,

            # 중간 프로젝션
            "장미": 0.70, "라벤더": 0.65, "제라늄": 0.65, "바닐라": 0.75,
            "시더우드": 0.60, "샌달우드": 0.70,

            # 약한 프로젝션
            "시트러스": 0.85, "레몬": 0.90, "베르가못": 0.85, "자몽": 0.80,
            "민트": 0.75, "그린노트": 0.60
        }

        total_projection = 0.0
        matched_count = 0

        for ingredient in ingredients:
            projection = 0.5  # 기본값

            for known_ingredient, weight in projection_weights.items():
                if known_ingredient.lower() in ingredient.lower() or ingredient.lower() in known_ingredient.lower():
                    projection = weight
                    break

            total_projection += projection
            matched_count += 1

        return total_projection / matched_count if matched_count > 0 else 0.5

    def _analyze_uniqueness(self, ingredients: List[str]) -> float:
        """독창성 분석"""
        # 일반적인 향료 조합과의 차이점 분석
        common_combinations = [
            ["시트러스", "라벤더", "머스크"],
            ["장미", "자스민", "바닐라"],
            ["베르가못", "제라늄", "시더우드"],
            ["레몬", "라벤더", "앰버"]
        ]

        # 희귀 향료 보너스
        rare_ingredients = [
            "아가우드", "용연향", "사향", "캐스토리움", "시벳",
            "프랑킨센스", "미르", "오리스", "튜베로즈"
        ]

        uniqueness_score = 0.5  # 기본 점수

        # 희귀 향료 가산점
        rare_count = sum(1 for ingredient in ingredients
                        for rare in rare_ingredients
                        if rare.lower() in ingredient.lower())

        uniqueness_score += min(0.3, rare_count * 0.1)

        # 일반적인 조합과의 차이점
        for common_combo in common_combinations:
            similarity = sum(1 for ingredient in ingredients
                           for common in common_combo
                           if common.lower() in ingredient.lower())

            if similarity >= 2:  # 2개 이상 일치하면 독창성 감소
                uniqueness_score -= 0.1

        # 복잡성 보너스
        if len(ingredients) > 10:
            uniqueness_score += 0.1

        return max(0.0, min(1.0, uniqueness_score))

    def _analyze_wearability(self, ingredients: List[str]) -> float:
        """착용감 분석"""
        if not ingredients:
            return 0.0

        # 착용하기 어려운 향료들
        challenging_ingredients = [
            "아가우드", "시벳", "인돌", "캐스토리움", "튜베로즈"
        ]

        # 착용하기 쉬운 향료들
        wearable_ingredients = [
            "시트러스", "라벤더", "시더우드", "머스크", "바닐라",
            "베르가못", "제라늄", "샌달우드"
        ]

        wearability_score = 0.5

        # 착용하기 어려운 향료 패널티
        challenging_count = sum(1 for ingredient in ingredients
                              for challenging in challenging_ingredients
                              if challenging.lower() in ingredient.lower())

        wearability_score -= challenging_count * 0.15

        # 착용하기 쉬운 향료 보너스
        wearable_count = sum(1 for ingredient in ingredients
                           for wearable in wearable_ingredients
                           if wearable.lower() in ingredient.lower())

        wearability_score += min(0.4, wearable_count * 0.1)

        # 복잡성 패널티 (너무 복잡하면 착용하기 어려움)
        if len(ingredients) > 15:
            wearability_score -= 0.1

        return max(0.0, min(1.0, wearability_score))

    def _analyze_commercial_appeal(self, ingredients: List[str]) -> float:
        """상업적 어필 분석"""
        if not ingredients:
            return 0.0

        # 대중적인 향료들
        popular_ingredients = [
            "바닐라", "시트러스", "라벤더", "장미", "머스크",
            "앰버", "시더우드", "베르가못", "자스민"
        ]

        # 니치한 향료들
        niche_ingredients = [
            "아가우드", "인센스", "오리스", "미르", "프랑킨센스"
        ]

        appeal_score = 0.5

        # 대중적인 향료 보너스
        popular_count = sum(1 for ingredient in ingredients
                          for popular in popular_ingredients
                          if popular.lower() in ingredient.lower())

        appeal_score += min(0.3, popular_count * 0.08)

        # 니치 향료는 상업적 어필을 약간 감소 (하지만 고급화 가능성)
        niche_count = sum(1 for ingredient in ingredients
                        for niche in niche_ingredients
                        if niche.lower() in ingredient.lower())

        if niche_count > 0:
            appeal_score -= 0.1
            appeal_score += 0.15  # 고급화 보너스

        # 적절한 복잡성 보너스
        if 8 <= len(ingredients) <= 12:
            appeal_score += 0.1

        return max(0.0, min(1.0, appeal_score))

    def _analyze_technical_excellence(self, ingredients: List[str], blend_prediction: BlendPrediction) -> float:
        """기술적 탁월성 분석"""
        technical_score = 0.0

        # 조화도 기여도 (40%)
        technical_score += blend_prediction.harmony_score * 0.4

        # 화학적 안정성 (30%)
        stability_score = blend_prediction.stability_score if hasattr(blend_prediction, 'stability_score') else 0.7
        technical_score += stability_score * 0.3

        # 구성의 균형 (20%)
        balance_score = self._calculate_composition_balance(ingredients)
        technical_score += balance_score * 0.2

        # 혁신성 (10%)
        innovation_score = self._assess_innovation_level(ingredients)
        technical_score += innovation_score * 0.1

        return min(1.0, technical_score)

    def _calculate_composition_balance(self, ingredients: List[str]) -> float:
        """구성 균형 계산"""
        if not ingredients:
            return 0.0

        # 향료를 카테고리별로 분류
        categories = {
            "top": 0,    # 탑노트
            "heart": 0,  # 하트노트
            "base": 0    # 베이스노트
        }

        # 간단한 분류 로직
        top_keywords = ["시트러스", "레몬", "베르가못", "자몽", "오렌지", "민트"]
        heart_keywords = ["장미", "자스민", "라벤더", "제라늄", "일랑일랑"]
        base_keywords = ["바닐라", "머스크", "앰버", "샌달우드", "시더우드", "패촐리"]

        for ingredient in ingredients:
            if any(keyword.lower() in ingredient.lower() for keyword in top_keywords):
                categories["top"] += 1
            elif any(keyword.lower() in ingredient.lower() for keyword in heart_keywords):
                categories["heart"] += 1
            elif any(keyword.lower() in ingredient.lower() for keyword in base_keywords):
                categories["base"] += 1

        total = sum(categories.values())
        if total == 0:
            return 0.5

        # 이상적인 비율에 가까울수록 높은 점수
        ideal_ratios = {"top": 0.3, "heart": 0.4, "base": 0.3}
        actual_ratios = {k: v/total for k, v in categories.items()}

        balance_score = 1.0 - sum(abs(ideal_ratios[k] - actual_ratios[k]) for k in categories.keys()) / 2
        return max(0.0, balance_score)

    def _assess_innovation_level(self, ingredients: List[str]) -> float:
        """혁신성 수준 평가"""
        innovation_score = 0.5

        # 독특한 조합 탐지
        unusual_combinations = [
            (["아가우드", "시트러스"], 0.2),
            (["바닐라", "오존"], 0.3),
            (["장미", "인센스"], 0.2),
            (["민트", "앰버"], 0.25)
        ]

        for combo, bonus in unusual_combinations:
            if all(any(keyword.lower() in ing.lower() for ing in ingredients) for keyword in combo):
                innovation_score += bonus

        return min(1.0, innovation_score)

    def _determine_grade(self, overall_score: float) -> str:
        """점수를 바탕으로 등급 결정"""
        for grade, threshold in self.grade_thresholds.items():
            if overall_score >= threshold:
                return grade
        return 'D'

    def _analyze_strengths_weaknesses(self, metric_scores: Dict[QualityMetric, float]) -> Tuple[List[str], List[str]]:
        """강점과 약점 분석"""
        strengths = []
        weaknesses = []

        strength_threshold = 0.75
        weakness_threshold = 0.60

        metric_names = {
            QualityMetric.HARMONY: "뛰어난 조화도",
            QualityMetric.COMPLEXITY: "적절한 복잡성",
            QualityMetric.LONGEVITY: "우수한 지속력",
            QualityMetric.PROJECTION: "좋은 확산력",
            QualityMetric.UNIQUENESS: "독창적인 구성",
            QualityMetric.WEARABILITY: "편안한 착용감",
            QualityMetric.COMMERCIAL_APPEAL: "높은 상업성",
            QualityMetric.TECHNICAL_EXCELLENCE: "기술적 완성도"
        }

        weakness_names = {
            QualityMetric.HARMONY: "조화도 개선 필요",
            QualityMetric.COMPLEXITY: "복잡성 부족",
            QualityMetric.LONGEVITY: "지속력 개선 필요",
            QualityMetric.PROJECTION: "확산력 부족",
            QualityMetric.UNIQUENESS: "독창성 부족",
            QualityMetric.WEARABILITY: "착용감 개선 필요",
            QualityMetric.COMMERCIAL_APPEAL: "상업성 부족",
            QualityMetric.TECHNICAL_EXCELLENCE: "기술적 완성도 부족"
        }

        for metric, score in metric_scores.items():
            if score >= strength_threshold:
                strengths.append(metric_names[metric])
            elif score < weakness_threshold:
                weaknesses.append(weakness_names[metric])

        return strengths, weaknesses

    def _generate_improvement_suggestions(self, metric_scores: Dict[QualityMetric, float], ingredients: List[str]) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        if metric_scores[QualityMetric.HARMONY] < 0.6:
            suggestions.append("브릿지 노트를 추가하여 향료 간 조화를 개선하세요")
            suggestions.append("상충하는 향료 조합을 재검토하세요")

        if metric_scores[QualityMetric.LONGEVITY] < 0.6:
            suggestions.append("베이스 노트의 비중을 늘려 지속력을 향상시키세요")
            suggestions.append("픽세이티브(고착제) 사용을 고려하세요")

        if metric_scores[QualityMetric.COMPLEXITY] < 0.5:
            suggestions.append("보조 향료를 추가하여 복잡성을 높이세요")
            suggestions.append("레이어링 기법을 활용하세요")
        elif metric_scores[QualityMetric.COMPLEXITY] > 0.9:
            suggestions.append("핵심 향료에 집중하여 구성을 단순화하세요")

        if metric_scores[QualityMetric.WEARABILITY] < 0.6:
            suggestions.append("강한 향료의 비율을 줄이세요")
            suggestions.append("중화 효과가 있는 향료를 추가하세요")

        if len(ingredients) > 15:
            suggestions.append("향료 수를 줄여 명확성을 높이세요")

        return suggestions

    def _assess_perfumer_level(self, overall_score: float, metric_scores: Dict[QualityMetric, float]) -> str:
        """조향사 수준 평가"""
        if overall_score >= 0.95:
            return "마스터 조향사급 (세계 정상급)"
        elif overall_score >= 0.90:
            return "수석 조향사급 (전문가)"
        elif overall_score >= 0.80:
            return "시니어 조향사급 (숙련자)"
        elif overall_score >= 0.70:
            return "주니어 조향사급 (중급자)"
        elif overall_score >= 0.60:
            return "조향사 견습생급 (초급자)"
        else:
            return "조향 입문자급"

    def _analyze_market_positioning(self, metric_scores: Dict[QualityMetric, float], ingredients: List[str]) -> Dict[str, Any]:
        """시장 포지셔닝 분석"""
        positioning = {
            "segment": "",
            "price_point": "",
            "target_demographic": {},
            "competitive_advantages": [],
            "market_risks": []
        }

        # 세그먼트 결정
        uniqueness = metric_scores[QualityMetric.UNIQUENESS]
        commercial_appeal = metric_scores[QualityMetric.COMMERCIAL_APPEAL]
        technical_excellence = metric_scores[QualityMetric.TECHNICAL_EXCELLENCE]

        if uniqueness > 0.8 and technical_excellence > 0.8:
            positioning["segment"] = "럭셔리 니치"
            positioning["price_point"] = "프리미엄 (20-50만원)"
        elif commercial_appeal > 0.7:
            positioning["segment"] = "메인스트림 프리미엄"
            positioning["price_point"] = "중가 (8-20만원)"
        else:
            positioning["segment"] = "메인스트림"
            positioning["price_point"] = "대중가 (3-8만원)"

        # 타겟 인구통계
        wearability = metric_scores[QualityMetric.WEARABILITY]
        complexity = metric_scores[QualityMetric.COMPLEXITY]

        if wearability > 0.8:
            positioning["target_demographic"]["age"] = "전 연령대"
            positioning["target_demographic"]["experience"] = "향수 초보자 포함"
        elif complexity > 0.7:
            positioning["target_demographic"]["age"] = "25-45세"
            positioning["target_demographic"]["experience"] = "향수 애호가"
        else:
            positioning["target_demographic"]["age"] = "35세 이상"
            positioning["target_demographic"]["experience"] = "향수 전문가"

        return positioning

    def _generate_detailed_analysis(
        self,
        ingredients: List[str],
        metric_scores: Dict[QualityMetric, float],
        blend_prediction: BlendPrediction
    ) -> Dict[str, Any]:
        """상세 분석 생성"""
        return {
            "ingredient_analysis": {
                "total_count": len(ingredients),
                "complexity_level": blend_prediction.complexity_rating,
                "predicted_notes": blend_prediction.predicted_notes
            },
            "performance_metrics": {
                "harmony_score": metric_scores[QualityMetric.HARMONY],
                "longevity_hours": blend_prediction.longevity_estimate.get("estimated_hours", 6),
                "projection_rating": metric_scores[QualityMetric.PROJECTION],
                "stability_score": blend_prediction.stability_score
            },
            "market_analysis": {
                "uniqueness_factor": metric_scores[QualityMetric.UNIQUENESS],
                "commercial_potential": metric_scores[QualityMetric.COMMERCIAL_APPEAL],
                "wearability_index": metric_scores[QualityMetric.WEARABILITY]
            },
            "recommendations": blend_prediction.recommendations,
            "chemical_compatibility": blend_prediction.chemical_compatibility
        }

    def _calculate_confidence_score(self, ingredients: List[str], metric_scores: Dict[QualityMetric, float]) -> float:
        """신뢰도 점수 계산"""
        confidence_factors = []

        # 알려진 향료 비율
        if hasattr(self.blending_ai, 'ingredient_profiles'):
            known_count = sum(1 for ing in ingredients if ing in self.blending_ai.ingredient_profiles)
            known_ratio = known_count / len(ingredients) if ingredients else 0
            confidence_factors.append(known_ratio)

        # 평가 점수들의 일관성
        scores = list(metric_scores.values())
        score_std = np.std(scores) if scores else 0
        consistency = max(0, 1 - score_std)
        confidence_factors.append(consistency)

        # 향료 수의 적절성
        ingredient_count_factor = min(1.0, len(ingredients) / 15) if ingredients else 0
        confidence_factors.append(ingredient_count_factor)

        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5

    def compare_with_benchmarks(self, assessment: QualityAssessment, ingredients: List[str]) -> Dict[str, Any]:
        """벤치마크와 비교"""
        comparisons = {}

        for benchmark_name, benchmark_data in self.benchmark_fragrances.items():
            similarity_score = 0.0
            metric_comparisons = {}

            # 각 메트릭별 비교
            for metric in QualityMetric:
                benchmark_score = benchmark_data["benchmark_scores"].get(metric, 0.5)
                assessment_score = assessment.metric_scores.get(metric, 0.5)

                difference = abs(benchmark_score - assessment_score)
                metric_similarity = max(0, 1 - difference)
                metric_comparisons[metric.value] = {
                    "benchmark_score": benchmark_score,
                    "assessment_score": assessment_score,
                    "similarity": metric_similarity
                }
                similarity_score += metric_similarity

            # 향료 유사도
            benchmark_ingredients = benchmark_data.get("signature_ingredients", [])
            ingredient_overlap = len(set(ingredients) & set(benchmark_ingredients))
            ingredient_similarity = ingredient_overlap / max(len(benchmark_ingredients), 1)

            comparisons[benchmark_name] = {
                "overall_similarity": similarity_score / len(QualityMetric),
                "ingredient_similarity": ingredient_similarity,
                "metric_comparisons": metric_comparisons,
                "benchmark_info": {
                    "perfumer": benchmark_data.get("perfumer"),
                    "year": benchmark_data.get("year"),
                    "style": benchmark_data.get("style")
                }
            }

        return comparisons

    def generate_comprehensive_report(
        self,
        ingredients: List[str],
        fragrance_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """종합 품질 보고서 생성"""

        # 기본 품질 분석
        quality_assessment = self.analyze_fragrance_quality(ingredients, fragrance_name=fragrance_name)

        # 벤치마크 비교
        benchmark_comparisons = self.compare_with_benchmarks(quality_assessment, ingredients)

        # 블렌딩 인사이트
        blending_insights = self.blending_ai.get_blending_insights(ingredients)

        # 마스터 조향사 지식 적용
        master_recommendation = self.perfumer_knowledge.create_master_blend_recommendation({
            "preferred_notes": ingredients[:5],  # 처음 5개를 선호 노트로 가정
            "style": "modern"
        })

        report = {
            "analysis_metadata": {
                "fragrance_name": fragrance_name or "분석 대상 향수",
                "analyzed_at": datetime.now().isoformat(),
                "total_ingredients": len(ingredients),
                "analysis_version": "1.0"
            },

            "executive_summary": {
                "overall_grade": quality_assessment.grade,
                "overall_score": quality_assessment.overall_score,
                "perfumer_level": quality_assessment.perfumer_level_assessment,
                "key_strengths": quality_assessment.strengths[:3],
                "critical_improvements": quality_assessment.improvement_suggestions[:2]
            },

            "detailed_quality_assessment": quality_assessment,
            "benchmark_analysis": benchmark_comparisons,
            "blending_technical_analysis": blending_insights,
            "master_perfumer_insights": master_recommendation,

            "recommendations": {
                "immediate_actions": quality_assessment.improvement_suggestions[:3],
                "long_term_goals": [
                    "지속적인 조화도 개선",
                    "시장 포지셔닝 강화",
                    "기술적 완성도 향상"
                ],
                "next_steps": [
                    "제안된 개선사항 적용",
                    "테스트 배치 제작",
                    "전문가 패널 평가"
                ]
            },

            "market_insights": {
                "positioning": quality_assessment.market_positioning,
                "commercial_viability": quality_assessment.metric_scores[QualityMetric.COMMERCIAL_APPEAL],
                "differentiation_factors": [
                    f"독창성 지수: {quality_assessment.metric_scores[QualityMetric.UNIQUENESS]:.2f}",
                    f"기술적 완성도: {quality_assessment.metric_scores[QualityMetric.TECHNICAL_EXCELLENCE]:.2f}"
                ]
            }
        }

        return report