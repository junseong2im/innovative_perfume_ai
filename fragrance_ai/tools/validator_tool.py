"""
검증 도구 - PostgreSQL 데이터베이스 기반 실제 검증
블렌딩 규칙과 과학적 제약 조건 검증
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session
from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

from fragrance_ai.database.schema import (
    Note, BlendingRule, AccordTemplate, FragranceComposition
)
from fragrance_ai.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Pydantic 스키마 (기존 인터페이스 유지)
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

@dataclass
class DBValidationResult:
    """데이터베이스 기반 검증 결과"""
    is_valid: bool
    stability_score: float  # 0-1
    harmony_score: float    # 0-1
    feasibility_score: float  # 0-1
    issues: List[str]
    suggestions: List[str]
    details: Dict[str, Any]


class ScientificValidator:
    """데이터베이스 기반 과학적 검증기"""

    def __init__(self, config_path: str = "configs/local.json"):
        """검증기 초기화"""
        self.db_manager = DatabaseManager()
        self._notes_cache = {}
        self._blending_rules_cache = {}
        self._load_db_cache()

    def _load_db_cache(self):
        """데이터베이스에서 규칙 캐싱"""
        try:
            with self.db_manager.get_session() as session:
                # 노트 정보 캐싱
                notes = session.query(Note).all()
                self._notes_cache = {
                    note.name.lower(): {
                        "id": note.id,
                        "type": note.type,
                        "pyramid_level": note.pyramid_level,
                        "volatility": note.volatility,
                        "strength": note.strength,
                        "longevity": note.longevity,
                        "is_natural": note.is_natural
                    }
                    for note in notes
                }

                # ID로도 캐싱
                self._notes_by_id = {
                    note.id: {
                        "name": note.name,
                        "type": note.type,
                        "pyramid_level": note.pyramid_level,
                        "volatility": note.volatility,
                        "strength": note.strength,
                        "longevity": note.longevity,
                        "is_natural": note.is_natural
                    }
                    for note in notes
                }

                # 블렌딩 규칙 캐싱
                rules = session.query(BlendingRule).filter(
                    BlendingRule.is_verified == True
                ).all()

                self._blending_rules_cache = {}
                for rule in rules:
                    key = tuple(sorted([rule.note1_id, rule.note2_id]))
                    self._blending_rules_cache[key] = {
                        "compatibility": rule.compatibility,
                        "rule_type": rule.rule_type,
                        "description": rule.description
                    }

            logger.info(f"DB cache loaded: {len(self._notes_cache)} notes, {len(self._blending_rules_cache)} rules")
        except Exception as e:
            logger.warning(f"Failed to load DB cache: {e}. Using empty cache.")

    def _composition_to_note_list(self, composition: NotesComposition) -> List[Tuple[int, float]]:
        """NotesComposition을 (note_id, percentage) 리스트로 변환"""
        note_list = []

        all_notes = []
        # 모든 노트 수집
        for notes in [composition.top_notes, composition.heart_notes, composition.base_notes]:
            for note_dict in notes:
                for name, percentage in note_dict.items():
                    all_notes.append((name.lower(), percentage))

        # 이름을 ID로 변환
        for name, percentage in all_notes:
            if name in self._notes_cache:
                note_id = self._notes_cache[name]["id"]
                note_list.append((note_id, percentage))
            else:
                logger.warning(f"Note '{name}' not found in database")

        return note_list

    def validate(self, composition: NotesComposition) -> ValidationResult:
        """조합 검증 수행 - 데이터베이스 기반"""
        try:
            # NotesComposition을 note_id 리스트로 변환
            note_list = self._composition_to_note_list(composition)

            if not note_list:
                return self._fallback_validation(composition)

            # 데이터베이스 기반 검증 수행
            db_result = self.validate_composition(note_list)

            # DB 검증 결과를 ValidationResult로 변환
            harmony = db_result.harmony_score * 10
            stability = db_result.stability_score * 10
            feasibility = db_result.feasibility_score * 10

            # 지속성과 확산성은 데이터베이스 정보에서 추정
            longevity = self._estimate_longevity(note_list)
            sillage = self._estimate_sillage(note_list)

            overall = (harmony + stability + feasibility + longevity + sillage) / 5

            # 위험 요소 및 제안사항
            risks = db_result.issues[:3] if db_result.issues else []
            suggestions = db_result.suggestions[:3] if db_result.suggestions else []

            # 과학적 노트 생성
            scientific_notes = self._generate_db_scientific_notes(
                db_result, note_list
            )

            return ValidationResult(
                is_valid=db_result.is_valid,
                harmony_score=round(harmony, 2),
                stability_score=round(stability, 2),
                longevity_score=round(longevity, 2),
                sillage_score=round(sillage, 2),
                overall_score=round(overall, 2),
                confidence=0.75,  # DB 기반 신뢰도
                key_risks=risks,
                suggestions=suggestions,
                scientific_notes=scientific_notes
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return self._fallback_validation(composition)

    def validate_composition(self, composition: List[Tuple[int, float]]) -> DBValidationResult:
        """데이터베이스 기반 조합 검증"""
        issues = []
        suggestions = []
        details = {}

        # 1. 기본 제약 확인
        basic_valid, basic_issues = self._validate_basic_constraints(composition)
        issues.extend(basic_issues)

        # 2. 안정성 검증
        stability_score, stability_details = self._validate_stability(composition)
        details["stability"] = stability_details

        # 3. 조화도 검증
        harmony_score, harmony_details = self._validate_harmony(composition)
        details["harmony"] = harmony_details

        # 4. 실현 가능성 검증
        feasibility_score, feasibility_details = self._validate_feasibility(composition)
        details["feasibility"] = feasibility_details

        # 5. 개선 제안
        if stability_score < 0.7:
            suggestions.append("피라미드 균형 조정 필요")
        if harmony_score < 0.7:
            suggestions.append("충돌 노트 교체 필요")
        if feasibility_score < 0.7:
            suggestions.append("비율 조정 필요")

        is_valid = (
            basic_valid and
            stability_score >= 0.5 and
            harmony_score >= 0.5 and
            feasibility_score >= 0.5
        )

        return DBValidationResult(
            is_valid=is_valid,
            stability_score=stability_score,
            harmony_score=harmony_score,
            feasibility_score=feasibility_score,
            issues=issues,
            suggestions=suggestions,
            details=details
        )

    def _validate_basic_constraints(self, composition: List[Tuple[int, float]]) -> Tuple[bool, List[str]]:
        """기본 제약 검증"""
        issues = []

        total_percentage = sum(pct for _, pct in composition)
        if abs(total_percentage - 100.0) > 5.0:
            issues.append(f"총 비율 오류: {total_percentage:.1f}%")

        for note_id, percentage in composition:
            if percentage < 0.1:
                issues.append(f"비율 너무 낮음: {percentage}%")
            elif percentage > 40.0:
                issues.append(f"비율 너무 높음: {percentage}%")

        if len(composition) < 3:
            issues.append("최소 3개 노트 필요")
        elif len(composition) > 30:
            issues.append("노트 수 과다")

        return len(issues) == 0, issues

    def _validate_stability(self, composition: List[Tuple[int, float]]) -> Tuple[float, Dict]:
        """안정성 검증"""
        details = {"top_pct": 0, "middle_pct": 0, "base_pct": 0}

        volatilities = []
        pyramid_pct = {"top": 0, "middle": 0, "base": 0}

        for note_id, percentage in composition:
            if note_id in self._notes_by_id:
                note = self._notes_by_id[note_id]
                if note["volatility"]:
                    volatilities.append(note["volatility"])
                if note["pyramid_level"] in pyramid_pct:
                    pyramid_pct[note["pyramid_level"]] += percentage

        # 피라미드 균형 점수
        top_score = 1.0 - abs(pyramid_pct["top"] - 25) / 25
        middle_score = 1.0 - abs(pyramid_pct["middle"] - 40) / 40
        base_score = 1.0 - abs(pyramid_pct["base"] - 35) / 35

        pyramid_score = max(0, min(1, (top_score + middle_score + base_score) / 3))

        # 휘발도 분산
        variance_score = 0.7
        if volatilities:
            variance = np.var(volatilities)
            variance_score = max(0, min(1, 1.0 - abs(variance - 0.25) * 2))

        stability = pyramid_score * 0.7 + variance_score * 0.3

        details.update(pyramid_pct)
        details["pyramid_score"] = pyramid_score
        details["variance_score"] = variance_score

        return stability, details

    def _validate_harmony(self, composition: List[Tuple[int, float]]) -> Tuple[float, Dict]:
        """조화도 검증"""
        details = {"conflicts": [], "harmonies": []}

        total_score = 0
        pair_count = 0

        # 모든 쌍 검사
        for i, (note1_id, pct1) in enumerate(composition):
            for note2_id, pct2 in composition[i+1:]:
                pair_key = tuple(sorted([note1_id, note2_id]))

                if pair_key in self._blending_rules_cache:
                    rule = self._blending_rules_cache[pair_key]
                    compatibility = rule["compatibility"]

                    weight = (pct1 + pct2) / 200.0
                    total_score += compatibility * weight
                    pair_count += weight

                    if compatibility < -0.3:
                        n1 = self._notes_by_id.get(note1_id, {}).get("name", f"ID{note1_id}")
                        n2 = self._notes_by_id.get(note2_id, {}).get("name", f"ID{note2_id}")
                        details["conflicts"].append(f"{n1}-{n2}")
                    elif compatibility > 0.7:
                        n1 = self._notes_by_id.get(note1_id, {}).get("name", f"ID{note1_id}")
                        n2 = self._notes_by_id.get(note2_id, {}).get("name", f"ID{note2_id}")
                        details["harmonies"].append(f"{n1}-{n2}")

        if pair_count > 0:
            harmony = (total_score / pair_count + 1) / 2
        else:
            harmony = 0.7

        # 타입 다양성 보너스
        types = set()
        for note_id, _ in composition:
            if note_id in self._notes_by_id:
                types.add(self._notes_by_id[note_id]["type"])

        diversity_bonus = min(0.2, len(types) * 0.05)
        harmony = min(1.0, harmony + diversity_bonus)

        details["type_diversity"] = len(types)

        return harmony, details

    def _validate_feasibility(self, composition: List[Tuple[int, float]]) -> Tuple[float, Dict]:
        """실현 가능성 검증"""
        details = {"natural_pct": 0, "synthetic_pct": 0}

        natural_pct = 0
        synthetic_pct = 0
        high_strength = 0

        for note_id, percentage in composition:
            if note_id in self._notes_by_id:
                note = self._notes_by_id[note_id]
                if note["is_natural"]:
                    natural_pct += percentage
                else:
                    synthetic_pct += percentage

                if note["strength"] and note["strength"] > 0.8 and percentage > 10:
                    high_strength += 1

        details["natural_pct"] = natural_pct
        details["synthetic_pct"] = synthetic_pct

        # 균형 점수
        balance_score = 1.0 - abs(natural_pct - synthetic_pct) / 100

        # 복잡도 점수
        if len(composition) < 5:
            complexity_score = 0.5
        elif len(composition) <= 20:
            complexity_score = 1.0
        else:
            complexity_score = max(0.3, 1.0 - (len(composition) - 20) * 0.05)

        # 강도 점수
        strength_score = max(0.3, 1.0 - high_strength * 0.15)

        feasibility = (balance_score * 0.3 + complexity_score * 0.4 + strength_score * 0.3)

        return feasibility, details

    def _estimate_longevity(self, composition: List[Tuple[int, float]]) -> float:
        """지속성 추정"""
        weighted_longevity = 0
        total_weight = 0

        for note_id, percentage in composition:
            if note_id in self._notes_by_id:
                note = self._notes_by_id[note_id]
                if note["longevity"]:
                    weighted_longevity += note["longevity"] * percentage
                    total_weight += percentage

        if total_weight > 0:
            return (weighted_longevity / total_weight) * 10
        return 5.0

    def _estimate_sillage(self, composition: List[Tuple[int, float]]) -> float:
        """확산성 추정"""
        weighted_strength = 0
        total_weight = 0

        for note_id, percentage in composition:
            if note_id in self._notes_by_id:
                note = self._notes_by_id[note_id]
                if note["strength"]:
                    weighted_strength += note["strength"] * percentage
                    total_weight += percentage

        if total_weight > 0:
            return (weighted_strength / total_weight) * 10
        return 5.0

    def _generate_db_scientific_notes(self, db_result: DBValidationResult, composition: List[Tuple[int, float]]) -> str:
        """데이터베이스 기반 과학적 노트"""
        notes = []

        notes.append(f"안정성: {db_result.stability_score:.1%}")
        notes.append(f"조화도: {db_result.harmony_score:.1%}")
        notes.append(f"실현가능성: {db_result.feasibility_score:.1%}")

        if db_result.details.get("stability"):
            s = db_result.details["stability"]
            notes.append(f"피라미드: T{s.get('top', 0):.0f}% M{s.get('middle', 0):.0f}% B{s.get('base', 0):.0f}%")

        if db_result.details.get("harmony"):
            h = db_result.details["harmony"]
            if h.get("conflicts"):
                notes.append(f"충돌: {', '.join(h['conflicts'][:2])}")
            if h.get("harmonies"):
                notes.append(f"조화: {', '.join(h['harmonies'][:2])}")

        return " | ".join(notes)

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