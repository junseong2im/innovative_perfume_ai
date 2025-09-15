"""
고급 향료 조합 예측 AI 모델
마스터 조향사급 블렌딩 기술과 화학적 친화성을 기반으로 한 예측 시스템
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from datetime import datetime
from ..knowledge.master_perfumer_principles import MasterPerfumerKnowledge, HarmonyRule
from ..core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class BlendPrediction:
    """블렌딩 예측 결과"""
    harmony_score: float
    stability_score: float
    complexity_rating: str
    predicted_notes: List[str]
    longevity_estimate: Dict[str, Any]
    recommendations: List[str]
    confidence: float
    chemical_compatibility: Dict[str, float]

@dataclass
class IngredientProfile:
    """향료 프로필"""
    name: str
    chemical_family: str
    volatility: float  # 휘발성 (0-1)
    intensity: float   # 강도 (0-1)
    longevity: float   # 지속성 (0-1)
    harmony_factors: Dict[str, float]  # 다른 향료와의 조화 인자
    molecular_weight: Optional[float] = None
    functional_groups: List[str] = None


class NeuralBlendingPredictor(nn.Module):
    """신경망 기반 블렌딩 예측 모델"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # 출력 레이어들
        self.feature_extractor = nn.Sequential(*layers)

        # 다중 출력 헤드
        self.harmony_head = nn.Linear(prev_dim, 1)
        self.stability_head = nn.Linear(prev_dim, 1)
        self.longevity_head = nn.Linear(prev_dim, 1)
        self.intensity_head = nn.Linear(prev_dim, 1)

        # 주의 메커니즘 (향료 간 상호작용 파악)
        self.attention = nn.MultiheadAttention(prev_dim, 8, batch_first=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)

        # 주의 메커니즘 적용 (배치 차원을 시퀀스 차원으로 확장)
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # (batch, 1, features)

        attn_features, _ = self.attention(features, features, features)
        attn_features = attn_features.squeeze(1)  # (batch, features)

        return {
            'harmony': torch.sigmoid(self.harmony_head(attn_features)),
            'stability': torch.sigmoid(self.stability_head(attn_features)),
            'longevity': torch.sigmoid(self.longevity_head(attn_features)),
            'intensity': torch.sigmoid(self.intensity_head(attn_features))
        }


class AdvancedBlendingAI:
    """고급 향료 조합 예측 AI 시스템"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 마스터 조향사 지식 시스템 로드
        self.perfumer_knowledge = MasterPerfumerKnowledge()

        # AI 모델들 초기화
        self.neural_model = None
        self.ensemble_models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

        # 향료 프로필 데이터베이스
        self.ingredient_profiles = self._initialize_ingredient_profiles()

        # 화학적 호환성 매트릭스
        self.compatibility_matrix = self._build_compatibility_matrix()

        logger.info("Advanced Blending AI initialized")

    def _initialize_ingredient_profiles(self) -> Dict[str, IngredientProfile]:
        """향료 프로필 데이터베이스 초기화"""
        profiles = {}

        # 기본 향료 프로필들 (실제로는 향료 데이터베이스에서 로드)
        base_ingredients = [
            # 시트러스
            IngredientProfile(
                name="베르가못",
                chemical_family="monoterpene",
                volatility=0.9,
                intensity=0.7,
                longevity=0.3,
                harmony_factors={"woody": 0.9, "floral": 0.8, "spicy": 0.7},
                molecular_weight=136.23,
                functional_groups=["ester", "alcohol"]
            ),
            IngredientProfile(
                name="레몬",
                chemical_family="monoterpene",
                volatility=0.95,
                intensity=0.8,
                longevity=0.2,
                harmony_factors={"woody": 0.8, "herbal": 0.9, "marine": 0.8}
            ),

            # 플로럴
            IngredientProfile(
                name="로즈",
                chemical_family="alcohol",
                volatility=0.5,
                intensity=0.9,
                longevity=0.7,
                harmony_factors={"woody": 0.9, "oriental": 0.8, "spicy": 0.7},
                molecular_weight=154.25,
                functional_groups=["alcohol", "aldehyde"]
            ),
            IngredientProfile(
                name="자스민",
                chemical_family="ester",
                volatility=0.4,
                intensity=0.95,
                longevity=0.8,
                harmony_factors={"woody": 0.8, "oriental": 0.9, "green": 0.6}
            ),

            # 우디
            IngredientProfile(
                name="샌달우드",
                chemical_family="sesquiterpene",
                volatility=0.2,
                intensity=0.6,
                longevity=0.9,
                harmony_factors={"floral": 0.9, "oriental": 0.8, "citrus": 0.7},
                molecular_weight=220.35,
                functional_groups=["alcohol"]
            ),
            IngredientProfile(
                name="시더우드",
                chemical_family="sesquiterpene",
                volatility=0.3,
                intensity=0.5,
                longevity=0.8,
                harmony_factors={"citrus": 0.8, "spicy": 0.7, "fresh": 0.6}
            ),

            # 오리엔탈
            IngredientProfile(
                name="바닐라",
                chemical_family="aldehyde",
                volatility=0.1,
                intensity=0.8,
                longevity=0.95,
                harmony_factors={"spicy": 0.9, "woody": 0.8, "gourmand": 0.95},
                molecular_weight=152.15,
                functional_groups=["aldehyde", "alcohol"]
            ),
            IngredientProfile(
                name="앰버",
                chemical_family="resin",
                volatility=0.15,
                intensity=0.7,
                longevity=0.9,
                harmony_factors={"oriental": 0.95, "woody": 0.8, "spicy": 0.8}
            )
        ]

        for profile in base_ingredients:
            profiles[profile.name] = profile

        return profiles

    def _build_compatibility_matrix(self) -> np.ndarray:
        """화학적 호환성 매트릭스 구축"""
        ingredients = list(self.ingredient_profiles.keys())
        n = len(ingredients)
        matrix = np.zeros((n, n))

        for i, ing1 in enumerate(ingredients):
            for j, ing2 in enumerate(ingredients):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    # 화학적 호환성 계산
                    compatibility = self._calculate_chemical_compatibility(
                        self.ingredient_profiles[ing1],
                        self.ingredient_profiles[ing2]
                    )
                    matrix[i][j] = compatibility

        return matrix

    def _calculate_chemical_compatibility(self, profile1: IngredientProfile, profile2: IngredientProfile) -> float:
        """두 향료의 화학적 호환성 계산"""
        compatibility = 0.5  # 기본값

        # 화학 패밀리 호환성
        family_compatibility = {
            ("monoterpene", "alcohol"): 0.8,
            ("monoterpene", "ester"): 0.9,
            ("alcohol", "ester"): 0.7,
            ("sesquiterpene", "alcohol"): 0.8,
            ("aldehyde", "alcohol"): 0.9,
            ("resin", "alcohol"): 0.7
        }

        family_pair = (profile1.chemical_family, profile2.chemical_family)
        reverse_pair = (profile2.chemical_family, profile1.chemical_family)

        if family_pair in family_compatibility:
            compatibility = family_compatibility[family_pair]
        elif reverse_pair in family_compatibility:
            compatibility = family_compatibility[reverse_pair]

        # 휘발성 차이에 따른 조정
        volatility_diff = abs(profile1.volatility - profile2.volatility)
        if volatility_diff < 0.3:  # 유사한 휘발성
            compatibility *= 1.1
        elif volatility_diff > 0.7:  # 매우 다른 휘발성 (좋을 수 있음)
            compatibility *= 1.05

        # 강도 균형 고려
        intensity_balance = 1.0 - abs(profile1.intensity - profile2.intensity)
        compatibility *= (0.8 + 0.2 * intensity_balance)

        return min(1.0, compatibility)

    def train_neural_model(self, training_data: List[Dict[str, Any]]):
        """신경망 모델 훈련"""
        try:
            # 특성 벡터 생성
            X, y = self._prepare_training_data(training_data)

            # 모델 초기화
            input_dim = X.shape[1]
            self.neural_model = NeuralBlendingPredictor(input_dim).to(self.device)

            # 훈련
            self._train_neural_network(X, y)

            logger.info("Neural blending model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train neural model: {e}")

    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """훈련 데이터 준비"""
        features = []
        targets = {'harmony': [], 'stability': [], 'longevity': [], 'intensity': []}

        for sample in training_data:
            # 향료 조합을 특성 벡터로 변환
            ingredient_vector = self._encode_ingredient_combination(sample['ingredients'])
            features.append(ingredient_vector)

            # 타겟 값들
            targets['harmony'].append(sample.get('harmony_score', 0.5))
            targets['stability'].append(sample.get('stability_score', 0.5))
            targets['longevity'].append(sample.get('longevity_score', 0.5))
            targets['intensity'].append(sample.get('intensity_score', 0.5))

        X = np.array(features)
        y = {key: np.array(values) for key, values in targets.items()}

        return X, y

    def _encode_ingredient_combination(self, ingredients: List[str]) -> np.ndarray:
        """향료 조합을 특성 벡터로 인코딩"""
        # 기본 특성들
        base_features = []

        # 각 향료의 기본 속성
        total_volatility = 0
        total_intensity = 0
        total_longevity = 0
        ingredient_count = len(ingredients)

        family_counts = {}

        for ingredient in ingredients:
            if ingredient in self.ingredient_profiles:
                profile = self.ingredient_profiles[ingredient]
                total_volatility += profile.volatility
                total_intensity += profile.intensity
                total_longevity += profile.longevity

                # 화학 패밀리 카운트
                family_counts[profile.chemical_family] = family_counts.get(profile.chemical_family, 0) + 1

        # 평균 속성들
        if ingredient_count > 0:
            base_features.extend([
                total_volatility / ingredient_count,
                total_intensity / ingredient_count,
                total_longevity / ingredient_count,
                ingredient_count
            ])
        else:
            base_features.extend([0, 0, 0, 0])

        # 화학 패밀리 분포 (원핫 인코딩)
        all_families = ["monoterpene", "alcohol", "ester", "sesquiterpene", "aldehyde", "resin"]
        for family in all_families:
            base_features.append(family_counts.get(family, 0))

        # 호환성 점수들
        compatibility_scores = []
        for i, ing1 in enumerate(ingredients):
            for j, ing2 in enumerate(ingredients[i+1:], i+1):
                if ing1 in self.ingredient_profiles and ing2 in self.ingredient_profiles:
                    score = self._calculate_chemical_compatibility(
                        self.ingredient_profiles[ing1],
                        self.ingredient_profiles[ing2]
                    )
                    compatibility_scores.append(score)

        # 평균 호환성 점수
        avg_compatibility = sum(compatibility_scores) / len(compatibility_scores) if compatibility_scores else 0.5
        base_features.append(avg_compatibility)

        # 복잡성 지수
        complexity_index = len(set(ingredients)) / max(len(ingredients), 1)
        base_features.append(complexity_index)

        return np.array(base_features, dtype=np.float32)

    def _train_neural_network(self, X: np.ndarray, y: Dict[str, np.ndarray]):
        """신경망 훈련"""
        # 데이터 정규화
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        y_tensors = {}
        for key, values in y.items():
            y_tensors[key] = torch.FloatTensor(values).unsqueeze(1).to(self.device)

        # 훈련 설정
        optimizer = torch.optim.Adam(self.neural_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # 훈련 루프
        epochs = 1000
        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.neural_model(X_tensor)

            # 다중 손실 계산
            total_loss = 0
            for key in outputs.keys():
                loss = criterion(outputs[key], y_tensors[key])
                total_loss += loss

            total_loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")

    def predict_blend_quality(self, ingredients: List[str], proportions: Optional[List[float]] = None) -> BlendPrediction:
        """블렌드 품질 예측"""
        try:
            # 특성 벡터 생성
            feature_vector = self._encode_ingredient_combination(ingredients)

            if self.neural_model is not None:
                # 신경망 예측
                X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)

                with torch.no_grad():
                    outputs = self.neural_model(X_tensor)

                harmony_score = outputs['harmony'].cpu().numpy()[0][0]
                stability_score = outputs['stability'].cpu().numpy()[0][0]
                longevity_estimate = {
                    "score": outputs['longevity'].cpu().numpy()[0][0],
                    "estimated_hours": outputs['longevity'].cpu().numpy()[0][0] * 12  # 최대 12시간
                }
            else:
                # 휴리스틱 기반 예측
                harmony_score = self._calculate_heuristic_harmony(ingredients)
                stability_score = 0.7  # 기본값
                longevity_estimate = {"score": 0.6, "estimated_hours": 6}

            # 복잡성 평가
            complexity_analysis = self.perfumer_knowledge.calculate_fragrance_complexity(ingredients)

            # 화학적 호환성 분석
            chemical_compatibility = self._analyze_chemical_compatibility(ingredients)

            # 추천사항 생성
            recommendations = self._generate_recommendations(ingredients, harmony_score, complexity_analysis)

            # 신뢰도 계산
            confidence = self._calculate_prediction_confidence(ingredients, harmony_score)

            return BlendPrediction(
                harmony_score=float(harmony_score),
                stability_score=float(stability_score),
                complexity_rating=complexity_analysis["complexity_level"],
                predicted_notes=self._predict_resulting_notes(ingredients),
                longevity_estimate=longevity_estimate,
                recommendations=recommendations,
                confidence=confidence,
                chemical_compatibility=chemical_compatibility
            )

        except Exception as e:
            logger.error(f"Failed to predict blend quality: {e}")
            return BlendPrediction(
                harmony_score=0.5,
                stability_score=0.5,
                complexity_rating="unknown",
                predicted_notes=[],
                longevity_estimate={"score": 0.5, "estimated_hours": 6},
                recommendations=["예측 오류 발생"],
                confidence=0.0,
                chemical_compatibility={}
            )

    def _calculate_heuristic_harmony(self, ingredients: List[str]) -> float:
        """휴리스틱 기반 조화도 계산"""
        if len(ingredients) < 2:
            return 1.0

        harmony_scores = []
        for i, ing1 in enumerate(ingredients):
            for ing2 in ingredients[i+1:]:
                score = self.perfumer_knowledge.get_harmony_score(ing1, ing2)
                harmony_scores.append(score)

        return sum(harmony_scores) / len(harmony_scores) if harmony_scores else 0.5

    def _analyze_chemical_compatibility(self, ingredients: List[str]) -> Dict[str, float]:
        """화학적 호환성 분석"""
        compatibility_analysis = {}

        for ingredient in ingredients:
            if ingredient in self.ingredient_profiles:
                profile = self.ingredient_profiles[ingredient]

                # 다른 향료들과의 평균 호환성
                compatibilities = []
                for other_ingredient in ingredients:
                    if other_ingredient != ingredient and other_ingredient in self.ingredient_profiles:
                        other_profile = self.ingredient_profiles[other_ingredient]
                        compatibility = self._calculate_chemical_compatibility(profile, other_profile)
                        compatibilities.append(compatibility)

                if compatibilities:
                    compatibility_analysis[ingredient] = sum(compatibilities) / len(compatibilities)

        return compatibility_analysis

    def _predict_resulting_notes(self, ingredients: List[str]) -> List[str]:
        """결과적으로 나타날 향조 예측"""
        predicted_notes = []

        # 향료 프로필 분석을 통한 예측
        family_strength = {}

        for ingredient in ingredients:
            if ingredient in self.ingredient_profiles:
                profile = self.ingredient_profiles[ingredient]

                # 화학 패밀리에 따른 예상 향조
                family_notes = {
                    "monoterpene": ["fresh", "citrusy", "bright"],
                    "alcohol": ["floral", "soft", "elegant"],
                    "ester": ["fruity", "sweet", "rich"],
                    "sesquiterpene": ["woody", "warm", "grounding"],
                    "aldehyde": ["sweet", "vanilla-like", "comforting"],
                    "resin": ["amber", "rich", "deep"]
                }

                family = profile.chemical_family
                if family in family_notes:
                    for note in family_notes[family]:
                        family_strength[note] = family_strength.get(note, 0) + profile.intensity

        # 강도 순으로 정렬하여 상위 3개 선택
        sorted_notes = sorted(family_strength.items(), key=lambda x: x[1], reverse=True)
        predicted_notes = [note for note, _ in sorted_notes[:3]]

        return predicted_notes

    def _generate_recommendations(self, ingredients: List[str], harmony_score: float, complexity_analysis: Dict[str, Any]) -> List[str]:
        """블렌드 개선 추천사항 생성"""
        recommendations = []

        if harmony_score < 0.6:
            recommendations.append("일부 향료의 조화도가 낮습니다. 브릿지 노트 추가를 고려하세요.")

        if harmony_score > 0.9:
            recommendations.append("탁월한 조화도입니다. 현재 구성을 유지하세요.")

        if complexity_analysis["total_ingredients"] > 15:
            recommendations.append("향료 수가 많습니다. 핵심 향료에 집중하여 단순화하세요.")

        if complexity_analysis["total_ingredients"] < 5:
            recommendations.append("단순한 구성입니다. 복잡성 추가를 위해 보조 향료를 고려하세요.")

        # 마스터 조향사 지식 기반 추천
        if complexity_analysis.get("recommendations"):
            recommendations.extend(complexity_analysis["recommendations"])

        return recommendations

    def _calculate_prediction_confidence(self, ingredients: List[str], harmony_score: float) -> float:
        """예측 신뢰도 계산"""
        confidence_factors = []

        # 알려진 향료 비율
        known_ingredients = sum(1 for ing in ingredients if ing in self.ingredient_profiles)
        known_ratio = known_ingredients / len(ingredients) if ingredients else 0
        confidence_factors.append(known_ratio)

        # 조화도 안정성 (극값이 아닌 경우 더 신뢰)
        harmony_stability = 1.0 - abs(harmony_score - 0.5) * 2
        confidence_factors.append(harmony_stability * 0.5 + 0.5)

        # 향료 수 적절성
        ingredient_count_factor = min(1.0, len(ingredients) / 10.0)
        confidence_factors.append(ingredient_count_factor)

        return sum(confidence_factors) / len(confidence_factors)

    def optimize_blend(self, base_ingredients: List[str], target_characteristics: Dict[str, float]) -> Dict[str, Any]:
        """블렌드 최적화"""
        try:
            best_blend = base_ingredients.copy()
            best_score = 0.0
            optimization_history = []

            # 가능한 추가 향료들
            available_ingredients = list(self.ingredient_profiles.keys())
            candidate_additions = [ing for ing in available_ingredients if ing not in base_ingredients]

            # 반복적 최적화
            for iteration in range(10):  # 최대 10회 반복
                current_prediction = self.predict_blend_quality(best_blend)
                current_score = self._calculate_target_fitness(current_prediction, target_characteristics)

                optimization_history.append({
                    "iteration": iteration,
                    "blend": best_blend.copy(),
                    "score": current_score,
                    "prediction": current_prediction
                })

                if current_score > best_score:
                    best_score = current_score

                # 다음 향료 후보 평가
                if candidate_additions:
                    best_addition = None
                    best_addition_score = current_score

                    for candidate in candidate_additions:
                        test_blend = best_blend + [candidate]
                        test_prediction = self.predict_blend_quality(test_blend)
                        test_score = self._calculate_target_fitness(test_prediction, target_characteristics)

                        if test_score > best_addition_score:
                            best_addition = candidate
                            best_addition_score = test_score

                    if best_addition:
                        best_blend.append(best_addition)
                        candidate_additions.remove(best_addition)
                        best_score = best_addition_score
                    else:
                        break  # 더 이상 개선되지 않음
                else:
                    break

            final_prediction = self.predict_blend_quality(best_blend)

            return {
                "optimized_blend": best_blend,
                "optimization_score": best_score,
                "final_prediction": final_prediction,
                "optimization_history": optimization_history,
                "improvements_made": len(best_blend) - len(base_ingredients)
            }

        except Exception as e:
            logger.error(f"Failed to optimize blend: {e}")
            return {
                "optimized_blend": base_ingredients,
                "optimization_score": 0.0,
                "error": str(e)
            }

    def _calculate_target_fitness(self, prediction: BlendPrediction, target_characteristics: Dict[str, float]) -> float:
        """목표 특성에 대한 적합도 계산"""
        fitness_scores = []

        # 조화도 적합성
        if "harmony" in target_characteristics:
            target_harmony = target_characteristics["harmony"]
            harmony_fitness = 1.0 - abs(prediction.harmony_score - target_harmony)
            fitness_scores.append(harmony_fitness)

        # 안정성 적합성
        if "stability" in target_characteristics:
            target_stability = target_characteristics["stability"]
            stability_fitness = 1.0 - abs(prediction.stability_score - target_stability)
            fitness_scores.append(stability_fitness)

        # 지속성 적합성
        if "longevity" in target_characteristics:
            target_longevity = target_characteristics["longevity"]
            longevity_fitness = 1.0 - abs(prediction.longevity_estimate["score"] - target_longevity)
            fitness_scores.append(longevity_fitness)

        # 복잡성 적합성
        if "complexity" in target_characteristics:
            target_complexity = target_characteristics["complexity"]
            complexity_levels = {"simple": 0.2, "moderate": 0.5, "complex": 0.8, "haute_couture": 1.0}
            current_complexity = complexity_levels.get(prediction.complexity_rating, 0.5)
            complexity_fitness = 1.0 - abs(current_complexity - target_complexity)
            fitness_scores.append(complexity_fitness)

        return sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0.0

    def get_blending_insights(self, ingredients: List[str]) -> Dict[str, Any]:
        """블렌딩 인사이트 제공"""
        try:
            prediction = self.predict_blend_quality(ingredients)

            insights = {
                "overall_assessment": self._generate_overall_assessment(prediction),
                "strength_analysis": self._analyze_blend_strengths(ingredients, prediction),
                "improvement_suggestions": self._suggest_improvements(ingredients, prediction),
                "perfumer_technique_match": self._match_perfumer_technique(ingredients),
                "market_potential": self._assess_market_potential(prediction),
                "technical_details": {
                    "volatility_profile": self._analyze_volatility_profile(ingredients),
                    "chemical_balance": self._analyze_chemical_balance(ingredients),
                    "complexity_breakdown": self._analyze_complexity_breakdown(ingredients)
                }
            }

            return insights

        except Exception as e:
            logger.error(f"Failed to generate blending insights: {e}")
            return {"error": str(e)}

    def _generate_overall_assessment(self, prediction: BlendPrediction) -> str:
        """전체적인 평가 생성"""
        if prediction.harmony_score >= 0.8:
            harmony_desc = "탁월한 조화"
        elif prediction.harmony_score >= 0.6:
            harmony_desc = "양호한 조화"
        else:
            harmony_desc = "조화 개선 필요"

        if prediction.stability_score >= 0.7:
            stability_desc = "안정적"
        else:
            stability_desc = "안정성 개선 필요"

        assessment = f"{harmony_desc}도를 보이며 {stability_desc}인 블렌드입니다. "
        assessment += f"복잡성은 {prediction.complexity_rating} 수준이고, "
        assessment += f"예상 지속시간은 {prediction.longevity_estimate.get('estimated_hours', 6):.1f}시간입니다."

        return assessment

    def _analyze_blend_strengths(self, ingredients: List[str], prediction: BlendPrediction) -> List[str]:
        """블렌드의 강점 분석"""
        strengths = []

        if prediction.harmony_score > 0.8:
            strengths.append("뛰어난 향료 조화도")

        if prediction.confidence > 0.8:
            strengths.append("높은 예측 신뢰도")

        if prediction.longevity_estimate.get("score", 0) > 0.7:
            strengths.append("우수한 지속력")

        if prediction.complexity_rating in ["complex", "haute_couture"]:
            strengths.append("고도의 조향 복잡성")

        return strengths

    def _suggest_improvements(self, ingredients: List[str], prediction: BlendPrediction) -> List[str]:
        """개선 제안"""
        suggestions = []

        if prediction.harmony_score < 0.6:
            suggestions.append("브릿지 노트 추가로 조화도 개선")

        if prediction.stability_score < 0.6:
            suggestions.append("안정화 성분 추가 고려")

        if len(ingredients) > 15:
            suggestions.append("핵심 향료 중심으로 단순화")

        suggestions.extend(prediction.recommendations)

        return list(set(suggestions))  # 중복 제거

    def _match_perfumer_technique(self, ingredients: List[str]) -> Dict[str, Any]:
        """조향사 기법 매칭"""
        # 마스터 조향사 지식 활용
        complexity_analysis = self.perfumer_knowledge.calculate_fragrance_complexity(ingredients)

        # 적합한 조향사 스타일 찾기
        perfumer_style = self.perfumer_knowledge.recommend_perfumer_style({
            "complexity": complexity_analysis["complexity_level"]
        })

        return {
            "recommended_perfumer": perfumer_style.perfumer_name,
            "style_characteristics": perfumer_style.style_characteristics,
            "applicable_techniques": perfumer_style.innovation_techniques
        }

    def _assess_market_potential(self, prediction: BlendPrediction) -> Dict[str, Any]:
        """시장 잠재력 평가"""
        market_score = 0.0

        # 조화도 기반 시장성
        market_score += prediction.harmony_score * 0.4

        # 복잡성 기반 차별화
        complexity_scores = {"simple": 0.3, "moderate": 0.6, "complex": 0.8, "haute_couture": 0.9}
        market_score += complexity_scores.get(prediction.complexity_rating, 0.5) * 0.3

        # 지속력 기반 실용성
        market_score += prediction.longevity_estimate.get("score", 0.5) * 0.3

        market_level = "높음" if market_score > 0.7 else "보통" if market_score > 0.5 else "낮음"

        return {
            "market_potential_score": market_score,
            "market_level": market_level,
            "target_segment": self._determine_target_segment(prediction)
        }

    def _determine_target_segment(self, prediction: BlendPrediction) -> str:
        """타겟 세그먼트 결정"""
        if prediction.complexity_rating == "haute_couture":
            return "럭셔리 니치"
        elif prediction.complexity_rating == "complex":
            return "프리미엄"
        elif prediction.harmony_score > 0.8:
            return "메인스트림 프리미엄"
        else:
            return "메인스트림"

    def _analyze_volatility_profile(self, ingredients: List[str]) -> Dict[str, Any]:
        """휘발성 프로필 분석"""
        volatilities = []

        for ingredient in ingredients:
            if ingredient in self.ingredient_profiles:
                volatilities.append(self.ingredient_profiles[ingredient].volatility)

        if not volatilities:
            return {"error": "향료 정보 부족"}

        return {
            "average_volatility": np.mean(volatilities),
            "volatility_range": np.max(volatilities) - np.min(volatilities),
            "top_note_strength": sum(1 for v in volatilities if v > 0.7) / len(volatilities),
            "base_note_strength": sum(1 for v in volatilities if v < 0.3) / len(volatilities)
        }

    def _analyze_chemical_balance(self, ingredients: List[str]) -> Dict[str, Any]:
        """화학적 균형 분석"""
        family_distribution = {}

        for ingredient in ingredients:
            if ingredient in self.ingredient_profiles:
                family = self.ingredient_profiles[ingredient].chemical_family
                family_distribution[family] = family_distribution.get(family, 0) + 1

        total = sum(family_distribution.values())
        balance_score = 1.0 - np.std(list(family_distribution.values())) / total if total > 0 else 0

        return {
            "family_distribution": family_distribution,
            "balance_score": balance_score,
            "dominant_family": max(family_distribution, key=family_distribution.get) if family_distribution else None
        }

    def _analyze_complexity_breakdown(self, ingredients: List[str]) -> Dict[str, Any]:
        """복잡성 세부 분석"""
        complexity_analysis = self.perfumer_knowledge.calculate_fragrance_complexity(ingredients)

        return {
            "total_ingredients": complexity_analysis["total_ingredients"],
            "complexity_level": complexity_analysis["complexity_level"],
            "harmony_potential": complexity_analysis["harmony_potential"],
            "balance_score": complexity_analysis["balance_score"],
            "ingredient_diversity": len(set(ingredients)) / len(ingredients) if ingredients else 0
        }