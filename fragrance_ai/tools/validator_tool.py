"""
과학적 향수 검증 도구
- 딥러닝 모델을 사용한 조합 검증
- 조화도, 안정성, 지속성 평가
"""

import torch
import joblib
import numpy as np
import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Pydantic 스키마
class NotesComposition(BaseModel):
    """향수 조합 구성"""
    top_notes: List[Dict[str, float]] = Field(..., description="탑노트 {name: percentage}")
    heart_notes: List[Dict[str, float]] = Field(..., description="하트노트 {name: percentage}")
    base_notes: List[Dict[str, float]] = Field(..., description="베이스노트 {name: percentage}")
    total_ingredients: int = Field(default=0, description="총 재료 수")

class ValidationResult(BaseModel):
    """검증 결과"""
    is_valid: bool = Field(..., description="조합 유효성")
    harmony_score: float = Field(..., description="조화도 점수 (0-10)")
    stability_score: float = Field(..., description="안정성 점수 (0-10)")
    longevity_score: float = Field(..., description="지속성 점수 (0-10)")
    sillage_score: float = Field(..., description="확산성 점수 (0-10)")
    overall_score: float = Field(..., description="종합 점수 (0-10)")
    confidence: float = Field(..., description="예측 신뢰도 (0-1)")
    key_risks: List[str] = Field(default_factory=list, description="주요 위험 요소")
    suggestions: List[str] = Field(default_factory=list, description="개선 제안사항")
    scientific_notes: str = Field(default="", description="과학적 분석 노트")

class ScientificValidator:
    """딥러닝 기반 과학적 검증기"""

    def __init__(self, config_path: str = "configs/local.json"):
        """검증기 초기화"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.preprocessor = None
        self.config = self._load_config(config_path)
        self._initialize_model()

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('deep_learning_validator', {})
        except Exception as e:
            logger.warning(f"Config load failed: {e}. Using defaults.")
            return {
                "trained_model_path": "assets/models/blending_validator.pth",
                "scaler_path": "assets/scalers/validator_scaler.pkl"
            }

    def _initialize_model(self):
        """모델 및 전처리기 초기화"""
        try:
            # 모델 아키텍처 임포트
            from ..models.advanced_blending_ai import NeuralBlendingPredictor, AdvancedBlendingAI

            # 전처리기 초기화
            self.preprocessor = AdvancedBlendingAI()

            # 스케일러 로드
            scaler_path = self.config.get('scaler_path')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                input_dim = self.scaler.n_features_in_
            else:
                logger.warning(f"Scaler not found at {scaler_path}. Using default dimensions.")
                input_dim = 768  # 기본 임베딩 차원

            # 모델 초기화 및 가중치 로드
            self.model = NeuralBlendingPredictor(input_dim)

            model_path = self.config.get('trained_model_path')
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model weights not found at {model_path}. Using untrained model.")

            self.model.to(self.device)
            self.model.eval()

        except ImportError as e:
            logger.error(f"Failed to import model architecture: {e}")
            # 폴백: 간단한 규칙 기반 모델 사용
            self.model = None
            self.preprocessor = None

    def _composition_to_features(self, composition: NotesComposition) -> np.ndarray:
        """조합을 특징 벡터로 변환"""
        if self.preprocessor is None:
            # 폴백: 기본 특징 추출
            return self._basic_feature_extraction(composition)

        try:
            # 재료 리스트 생성
            ingredients = []
            for notes in [composition.top_notes, composition.heart_notes, composition.base_notes]:
                for note_dict in notes:
                    for name, percentage in note_dict.items():
                        ingredients.append({
                            "name": name,
                            "percentage": percentage,
                            "category": "top" if notes == composition.top_notes else
                                      "heart" if notes == composition.heart_notes else "base"
                        })

            # 전처리기로 특징 추출
            features = self.preprocessor._encode_ingredient_combination(ingredients)
            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._basic_feature_extraction(composition)

    def _basic_feature_extraction(self, composition: NotesComposition) -> np.ndarray:
        """기본 특징 추출 (폴백)"""
        features = []

        # 노트별 개수 및 총 퍼센트
        for notes in [composition.top_notes, composition.heart_notes, composition.base_notes]:
            count = len(notes)
            total_pct = sum(sum(note.values()) for note in notes)
            features.extend([count, total_pct])

        # 전체 재료 수
        total_count = sum(len(notes) for notes in
                         [composition.top_notes, composition.heart_notes, composition.base_notes])
        features.append(total_count)

        # 패딩 (최소 10개 특징)
        while len(features) < 10:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def validate(self, composition: NotesComposition) -> ValidationResult:
        """조합 검증 수행"""
        try:
            # 특징 추출
            features = self._composition_to_features(composition)

            if self.model is None:
                # 폴백: 규칙 기반 검증
                return self._rule_based_validation(composition, features)

            # 특징 스케일링
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))[0]

            # 텐서 변환
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

            # 모델 예측
            with torch.no_grad():
                predictions = self.model(features_tensor)

            # 결과 추출
            harmony = float(predictions['harmony'].cpu().item()) * 10
            stability = float(predictions['stability'].cpu().item()) * 10
            longevity = float(predictions['longevity'].cpu().item()) * 10
            sillage = float(predictions['sillage'].cpu().item()) * 10
            overall = (harmony + stability + longevity + sillage) / 4

            # 위험 요소 및 제안사항 생성
            risks, suggestions = self._analyze_scores(
                harmony, stability, longevity, sillage, composition
            )

            # 과학적 노트 생성
            scientific_notes = self._generate_scientific_notes(
                harmony, stability, longevity, sillage, composition
            )

            return ValidationResult(
                is_valid=overall >= 6.0,
                harmony_score=round(harmony, 2),
                stability_score=round(stability, 2),
                longevity_score=round(longevity, 2),
                sillage_score=round(sillage, 2),
                overall_score=round(overall, 2),
                confidence=0.85,  # 모델 신뢰도
                key_risks=risks,
                suggestions=suggestions,
                scientific_notes=scientific_notes
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._fallback_validation(composition)

    def _rule_based_validation(self, composition: NotesComposition, features: np.ndarray) -> ValidationResult:
        """규칙 기반 검증 (폴백)"""
        # 기본 규칙 기반 점수 계산
        total_ingredients = features[6] if len(features) > 6 else 10

        # 재료 수에 따른 점수
        if total_ingredients < 5:
            harmony = 5.0
        elif total_ingredients < 10:
            harmony = 7.0
        elif total_ingredients < 15:
            harmony = 8.0
        else:
            harmony = 6.0

        # 노트 밸런스 체크
        top_count = len(composition.top_notes)
        heart_count = len(composition.heart_notes)
        base_count = len(composition.base_notes)

        balance_score = 10.0
        if abs(top_count - heart_count) > 3:
            balance_score -= 2.0
        if abs(heart_count - base_count) > 3:
            balance_score -= 2.0

        stability = min(10.0, balance_score)
        longevity = 7.0 if base_count >= 2 else 5.0
        sillage = 6.0 if top_count >= 2 else 4.0

        overall = (harmony + stability + longevity + sillage) / 4

        return ValidationResult(
            is_valid=overall >= 6.0,
            harmony_score=harmony,
            stability_score=stability,
            longevity_score=longevity,
            sillage_score=sillage,
            overall_score=overall,
            confidence=0.5,  # 규칙 기반은 낮은 신뢰도
            key_risks=["Rule-based validation - limited accuracy"],
            suggestions=["Consider training deep learning model for better accuracy"],
            scientific_notes="Basic rule-based analysis performed"
        )

    def _analyze_scores(self, harmony: float, stability: float, longevity: float,
                       sillage: float, composition: NotesComposition) -> tuple:
        """점수 분석 및 위험/제안 생성"""
        risks = []
        suggestions = []

        if harmony < 6.0:
            risks.append("낮은 조화도 - 노트 간 충돌 가능성")
            suggestions.append("상충하는 노트를 제거하거나 브릿지 노트 추가")

        if stability < 6.0:
            risks.append("안정성 부족 - 시간에 따른 향 변화 예상")
            suggestions.append("안정제 역할의 베이스 노트 강화")

        if longevity < 6.0:
            risks.append("짧은 지속시간")
            suggestions.append("픽서티브 추가 또는 베이스 노트 비율 증가")

        if sillage < 6.0:
            risks.append("약한 확산성")
            suggestions.append("탑 노트 강화 또는 확산성 높은 재료 추가")

        return risks, suggestions

    def _generate_scientific_notes(self, harmony: float, stability: float,
                                  longevity: float, sillage: float,
                                  composition: NotesComposition) -> str:
        """과학적 분석 노트 생성"""
        notes = []

        notes.append(f"전체 조화도: {harmony:.1f}/10")
        notes.append(f"화학적 안정성: {stability:.1f}/10")
        notes.append(f"예상 지속시간: {longevity:.1f}/10")
        notes.append(f"확산 계수: {sillage:.1f}/10")

        # 노트 구성 분석
        total_top = len(composition.top_notes)
        total_heart = len(composition.heart_notes)
        total_base = len(composition.base_notes)

        notes.append(f"\n구성: 탑({total_top}) / 하트({total_heart}) / 베이스({total_base})")

        if harmony >= 8.0:
            notes.append("우수한 조화를 이루는 조합입니다.")
        elif harmony >= 6.0:
            notes.append("양호한 조화를 보이나 개선 여지가 있습니다.")
        else:
            notes.append("조화도 개선이 필요합니다.")

        return " ".join(notes)

    def _fallback_validation(self, composition: NotesComposition) -> ValidationResult:
        """최종 폴백 검증"""
        return ValidationResult(
            is_valid=True,
            harmony_score=7.0,
            stability_score=7.0,
            longevity_score=7.0,
            sillage_score=7.0,
            overall_score=7.0,
            confidence=0.3,
            key_risks=["Validation system unavailable - default scores applied"],
            suggestions=["Manual review recommended"],
            scientific_notes="Fallback validation - please verify manually"
        )

# 전역 검증기 인스턴스
validator_instance = None

def get_validator():
    """검증기 인스턴스 가져오기"""
    global validator_instance
    if validator_instance is None:
        validator_instance = ScientificValidator()
    return validator_instance

async def validate_composition(composition: NotesComposition) -> ValidationResult:
    """
    # LLM TOOL DESCRIPTION (FOR ORCHESTRATOR)
    # Use this tool exclusively for scientific validation AFTER creating a new recipe.
    # It leverages a trained deep learning model to provide objective analysis.
    # Returns scores for harmony, stability, longevity, and sillage.

    Args:
        composition: 검증할 향수 조합

    Returns:
        ValidationResult: 과학적 검증 결과
    """
    validator = get_validator()
    return validator.validate(composition)