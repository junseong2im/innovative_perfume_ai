"""
향료 호환성 매트릭스 시스템
마스터 조향사의 경험과 화학적 지식을 기반으로 한 향료간 호환성 분석
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
import logging
from enum import Enum
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

from ..knowledge.master_perfumer_principles import MasterPerfumerKnowledge
from ..core.config import settings

logger = logging.getLogger(__name__)

class CompatibilityLevel(Enum):
    """호환성 레벨"""
    PERFECT = "perfect"      # 0.9-1.0: 완벽한 조화
    EXCELLENT = "excellent"  # 0.8-0.9: 탁월한 조화
    GOOD = "good"           # 0.7-0.8: 좋은 조화
    MODERATE = "moderate"   # 0.6-0.7: 보통 조화
    FAIR = "fair"           # 0.5-0.6: 무난한 조화
    POOR = "poor"          # 0.4-0.5: 약한 조화
    INCOMPATIBLE = "incompatible"  # 0.0-0.4: 비호환

@dataclass
class CompatibilityScore:
    """호환성 점수 상세 정보"""
    score: float
    level: CompatibilityLevel
    reasoning: List[str]
    chemical_basis: str
    perfumer_notes: Optional[str] = None
    confidence: float = 0.0
    synergy_factors: List[str] = field(default_factory=list)
    conflict_factors: List[str] = field(default_factory=list)

@dataclass
class IngredientCompatibilityProfile:
    """향료 호환성 프로필"""
    ingredient_name: str
    chemical_family: str
    functional_groups: List[str]
    volatility: float
    intensity: float
    longevity: float
    molecular_weight: Optional[float] = None
    polarity: Optional[float] = None
    dominant_characteristics: List[str] = field(default_factory=list)
    best_partners: Dict[str, float] = field(default_factory=dict)
    worst_partners: Dict[str, float] = field(default_factory=dict)
    neutral_partners: Dict[str, float] = field(default_factory=dict)


class ChemicalCompatibilityEngine:
    """화학적 호환성 엔진"""

    def __init__(self):
        self.functional_group_affinities = self._initialize_functional_group_affinities()
        self.chemical_family_interactions = self._initialize_chemical_family_interactions()
        self.molecular_interaction_rules = self._initialize_molecular_rules()

    def _initialize_functional_group_affinities(self) -> Dict[str, Dict[str, float]]:
        """관능기간 친화도 매트릭스"""
        return {
            "alcohol": {
                "alcohol": 0.8, "ester": 0.9, "aldehyde": 0.85, "ketone": 0.8,
                "ether": 0.7, "phenol": 0.85, "acid": 0.9, "amine": 0.6
            },
            "ester": {
                "alcohol": 0.9, "ester": 0.8, "aldehyde": 0.8, "ketone": 0.75,
                "ether": 0.8, "phenol": 0.7, "acid": 0.6, "amine": 0.7
            },
            "aldehyde": {
                "alcohol": 0.85, "ester": 0.8, "aldehyde": 0.7, "ketone": 0.85,
                "ether": 0.75, "phenol": 0.8, "acid": 0.7, "amine": 0.65
            },
            "ketone": {
                "alcohol": 0.8, "ester": 0.75, "aldehyde": 0.85, "ketone": 0.75,
                "ether": 0.8, "phenol": 0.75, "acid": 0.7, "amine": 0.7
            },
            "ether": {
                "alcohol": 0.7, "ester": 0.8, "aldehyde": 0.75, "ketone": 0.8,
                "ether": 0.85, "phenol": 0.6, "acid": 0.5, "amine": 0.75
            },
            "phenol": {
                "alcohol": 0.85, "ester": 0.7, "aldehyde": 0.8, "ketone": 0.75,
                "ether": 0.6, "phenol": 0.9, "acid": 0.85, "amine": 0.6
            },
            "acid": {
                "alcohol": 0.9, "ester": 0.6, "aldehyde": 0.7, "ketone": 0.7,
                "ether": 0.5, "phenol": 0.85, "acid": 0.7, "amine": 0.8
            },
            "amine": {
                "alcohol": 0.6, "ester": 0.7, "aldehyde": 0.65, "ketone": 0.7,
                "ether": 0.75, "phenol": 0.6, "acid": 0.8, "amine": 0.8
            }
        }

    def _initialize_chemical_family_interactions(self) -> Dict[str, Dict[str, float]]:
        """화학 패밀리간 상호작용"""
        return {
            "monoterpene": {
                "monoterpene": 0.85, "sesquiterpene": 0.9, "aromatic": 0.8,
                "aliphatic": 0.75, "heterocyclic": 0.7
            },
            "sesquiterpene": {
                "monoterpene": 0.9, "sesquiterpene": 0.85, "aromatic": 0.85,
                "aliphatic": 0.8, "heterocyclic": 0.75
            },
            "aromatic": {
                "monoterpene": 0.8, "sesquiterpene": 0.85, "aromatic": 0.9,
                "aliphatic": 0.7, "heterocyclic": 0.85
            },
            "aliphatic": {
                "monoterpene": 0.75, "sesquiterpene": 0.8, "aromatic": 0.7,
                "aliphatic": 0.8, "heterocyclic": 0.65
            },
            "heterocyclic": {
                "monoterpene": 0.7, "sesquiterpene": 0.75, "aromatic": 0.85,
                "aliphatic": 0.65, "heterocyclic": 0.9
            }
        }

    def _initialize_molecular_rules(self) -> List[Dict[str, Any]]:
        """분자간 상호작용 규칙"""
        return [
            {
                "rule": "similar_volatility_bonus",
                "description": "유사한 휘발성을 가진 분자들은 조화롭게 증발",
                "condition": lambda v1, v2: abs(v1 - v2) < 0.3,
                "bonus": 0.15
            },
            {
                "rule": "complementary_volatility",
                "description": "보완적 휘발성 (top-base 조합)",
                "condition": lambda v1, v2: abs(v1 - v2) > 0.6,
                "bonus": 0.1
            },
            {
                "rule": "molecular_weight_harmony",
                "description": "분자량 조화 (2배 이내)",
                "condition": lambda mw1, mw2: mw1 is not None and mw2 is not None and min(mw1, mw2) * 2 >= max(mw1, mw2),
                "bonus": 0.1
            },
            {
                "rule": "intensity_balance",
                "description": "강도 균형 (한쪽이 너무 강하지 않음)",
                "condition": lambda i1, i2: abs(i1 - i2) < 0.4,
                "bonus": 0.1
            },
            {
                "rule": "polarity_compatibility",
                "description": "극성 호환성",
                "condition": lambda p1, p2: p1 is not None and p2 is not None and abs(p1 - p2) < 0.5,
                "bonus": 0.1
            }
        ]

    def calculate_chemical_compatibility(
        self,
        profile1: IngredientCompatibilityProfile,
        profile2: IngredientCompatibilityProfile
    ) -> Tuple[float, List[str]]:
        """화학적 호환성 계산"""

        compatibility_score = 0.5  # 기본값
        reasoning = []

        # 관능기 친화도
        if profile1.functional_groups and profile2.functional_groups:
            fg_scores = []
            for fg1 in profile1.functional_groups:
                for fg2 in profile2.functional_groups:
                    if fg1 in self.functional_group_affinities and fg2 in self.functional_group_affinities[fg1]:
                        fg_score = self.functional_group_affinities[fg1][fg2]
                        fg_scores.append(fg_score)

            if fg_scores:
                avg_fg_score = np.mean(fg_scores)
                compatibility_score = compatibility_score * 0.3 + avg_fg_score * 0.7
                reasoning.append(f"관능기 친화도: {avg_fg_score:.2f}")

        # 화학 패밀리 상호작용
        if profile1.chemical_family in self.chemical_family_interactions:
            if profile2.chemical_family in self.chemical_family_interactions[profile1.chemical_family]:
                family_score = self.chemical_family_interactions[profile1.chemical_family][profile2.chemical_family]
                compatibility_score = compatibility_score * 0.7 + family_score * 0.3
                reasoning.append(f"패밀리 호환성: {family_score:.2f}")

        # 분자간 상호작용 규칙 적용
        for rule in self.molecular_interaction_rules:
            if rule["rule"] == "similar_volatility_bonus":
                if rule["condition"](profile1.volatility, profile2.volatility):
                    compatibility_score += rule["bonus"]
                    reasoning.append("유사 휘발성 보너스")

            elif rule["rule"] == "complementary_volatility":
                if rule["condition"](profile1.volatility, profile2.volatility):
                    compatibility_score += rule["bonus"]
                    reasoning.append("보완적 휘발성")

            elif rule["rule"] == "molecular_weight_harmony":
                if rule["condition"](profile1.molecular_weight, profile2.molecular_weight):
                    compatibility_score += rule["bonus"]
                    reasoning.append("분자량 조화")

            elif rule["rule"] == "intensity_balance":
                if rule["condition"](profile1.intensity, profile2.intensity):
                    compatibility_score += rule["bonus"]
                    reasoning.append("강도 균형")

            elif rule["rule"] == "polarity_compatibility":
                if rule["condition"](profile1.polarity, profile2.polarity):
                    compatibility_score += rule["bonus"]
                    reasoning.append("극성 호환성")

        return min(1.0, compatibility_score), reasoning


class FragranceCompatibilityMatrix:
    """향료 호환성 매트릭스 메인 클래스"""

    def __init__(self):
        self.perfumer_knowledge = MasterPerfumerKnowledge()
        self.chemical_engine = ChemicalCompatibilityEngine()

        # 향료 프로필 데이터베이스
        self.ingredient_profiles: Dict[str, IngredientCompatibilityProfile] = {}

        # 호환성 매트릭스 (캐시)
        self.compatibility_matrix: Optional[np.ndarray] = None
        self.ingredient_index: Dict[str, int] = {}
        self.index_to_ingredient: Dict[int, str] = {}

        # 학습된 패턴 (마스터 조향사의 실제 조합들)
        self.learned_combinations = self._load_master_combinations()

        # 향료 프로필 초기화
        self._initialize_ingredient_profiles()

        logger.info("Fragrance Compatibility Matrix initialized")

    def _load_master_combinations(self) -> List[Dict[str, Any]]:
        """마스터 조향사의 검증된 조합들 로드"""
        return [
            {
                "combination": ["베르가못", "라벤더", "바닐라"],
                "harmony_score": 0.92,
                "perfumer": "Aimé Guerlain",
                "fragrance": "Jicky",
                "notes": "클래식한 푸제르 구조의 완벽한 조화"
            },
            {
                "combination": ["로즈", "자스민", "인센스"],
                "harmony_score": 0.89,
                "perfumer": "Jacques Guerlain",
                "fragrance": "L'Heure Bleue",
                "notes": "플로럴과 오리엔탈의 우아한 만남"
            },
            {
                "combination": ["시트러스", "아이리스", "시더"],
                "harmony_score": 0.91,
                "perfumer": "Jean-Claude Ellena",
                "fragrance": "Hermès garden series",
                "notes": "미니멀한 우아함의 정수"
            },
            {
                "combination": ["우드", "로즈", "사프란"],
                "harmony_score": 0.94,
                "perfumer": "Francis Kurkdjian",
                "fragrance": "Baccarat Rouge 540",
                "notes": "현대적 럭셔리의 완성"
            },
            {
                "combination": ["바닐라", "통카빈", "앰버그리스"],
                "harmony_score": 0.88,
                "perfumer": "Thierry Wasser",
                "fragrance": "Guerlain orientals",
                "notes": "구르망과 오리엔탈의 깊이"
            }
        ]

    def _initialize_ingredient_profiles(self):
        """향료 프로필 초기화"""

        # 주요 향료들의 상세 프로필
        profiles_data = [
            # 시트러스 패밀리
            {
                "ingredient_name": "베르가못",
                "chemical_family": "monoterpene",
                "functional_groups": ["ester", "alcohol"],
                "volatility": 0.9,
                "intensity": 0.7,
                "longevity": 0.3,
                "molecular_weight": 136.23,
                "polarity": 0.3,
                "dominant_characteristics": ["fresh", "citrusy", "elegant"]
            },
            {
                "ingredient_name": "레몬",
                "chemical_family": "monoterpene",
                "functional_groups": ["aldehyde", "terpene"],
                "volatility": 0.95,
                "intensity": 0.8,
                "longevity": 0.2,
                "molecular_weight": 136.15,
                "polarity": 0.2,
                "dominant_characteristics": ["sharp", "clean", "energizing"]
            },
            {
                "ingredient_name": "자몽",
                "chemical_family": "monoterpene",
                "functional_groups": ["aldehyde", "ketone"],
                "volatility": 0.9,
                "intensity": 0.75,
                "longevity": 0.25,
                "molecular_weight": 150.22,
                "polarity": 0.25,
                "dominant_characteristics": ["juicy", "bitter", "refreshing"]
            },

            # 플로럴 패밀리
            {
                "ingredient_name": "로즈",
                "chemical_family": "aromatic",
                "functional_groups": ["alcohol", "ester"],
                "volatility": 0.5,
                "intensity": 0.9,
                "longevity": 0.7,
                "molecular_weight": 154.25,
                "polarity": 0.6,
                "dominant_characteristics": ["romantic", "classic", "elegant"]
            },
            {
                "ingredient_name": "자스민",
                "chemical_family": "aromatic",
                "functional_groups": ["ester", "aldehyde"],
                "volatility": 0.4,
                "intensity": 0.95,
                "longevity": 0.8,
                "molecular_weight": 174.28,
                "polarity": 0.65,
                "dominant_characteristics": ["intoxicating", "sensual", "rich"]
            },
            {
                "ingredient_name": "라벤더",
                "chemical_family": "monoterpene",
                "functional_groups": ["alcohol", "ester"],
                "volatility": 0.7,
                "intensity": 0.6,
                "longevity": 0.5,
                "molecular_weight": 154.25,
                "polarity": 0.4,
                "dominant_characteristics": ["herbal", "calming", "classic"]
            },
            {
                "ingredient_name": "일랑일랑",
                "chemical_family": "aromatic",
                "functional_groups": ["ester", "phenol"],
                "volatility": 0.45,
                "intensity": 0.85,
                "longevity": 0.75,
                "molecular_weight": 220.35,
                "polarity": 0.55,
                "dominant_characteristics": ["exotic", "creamy", "narcotic"]
            },

            # 우디 패밀리
            {
                "ingredient_name": "샌달우드",
                "chemical_family": "sesquiterpene",
                "functional_groups": ["alcohol"],
                "volatility": 0.2,
                "intensity": 0.6,
                "longevity": 0.9,
                "molecular_weight": 220.35,
                "polarity": 0.3,
                "dominant_characteristics": ["creamy", "smooth", "meditative"]
            },
            {
                "ingredient_name": "시더우드",
                "chemical_family": "sesquiterpene",
                "functional_groups": ["ketone", "alcohol"],
                "volatility": 0.3,
                "intensity": 0.5,
                "longevity": 0.8,
                "molecular_weight": 204.35,
                "polarity": 0.25,
                "dominant_characteristics": ["dry", "pencil-like", "grounding"]
            },
            {
                "ingredient_name": "베티버",
                "chemical_family": "sesquiterpene",
                "functional_groups": ["alcohol", "ketone"],
                "volatility": 0.25,
                "intensity": 0.7,
                "longevity": 0.85,
                "molecular_weight": 218.38,
                "polarity": 0.35,
                "dominant_characteristics": ["earthy", "smoky", "sophisticated"]
            },

            # 오리엔탈 패밀리
            {
                "ingredient_name": "바닐라",
                "chemical_family": "aromatic",
                "functional_groups": ["aldehyde", "phenol"],
                "volatility": 0.1,
                "intensity": 0.8,
                "longevity": 0.95,
                "molecular_weight": 152.15,
                "polarity": 0.7,
                "dominant_characteristics": ["sweet", "comforting", "gourmand"]
            },
            {
                "ingredient_name": "앰버",
                "chemical_family": "heterocyclic",
                "functional_groups": ["ether", "alcohol"],
                "volatility": 0.15,
                "intensity": 0.7,
                "longevity": 0.9,
                "molecular_weight": 230.26,
                "polarity": 0.5,
                "dominant_characteristics": ["warm", "resinous", "animalic"]
            },
            {
                "ingredient_name": "머스크",
                "chemical_family": "aliphatic",
                "functional_groups": ["ketone", "ether"],
                "volatility": 0.05,
                "intensity": 0.6,
                "longevity": 0.98,
                "molecular_weight": 238.41,
                "polarity": 0.4,
                "dominant_characteristics": ["clean", "skin-like", "intimate"]
            },
            {
                "ingredient_name": "인센스",
                "chemical_family": "heterocyclic",
                "functional_groups": ["ester", "phenol"],
                "volatility": 0.2,
                "intensity": 0.8,
                "longevity": 0.85,
                "molecular_weight": 204.23,
                "polarity": 0.6,
                "dominant_characteristics": ["sacred", "smoky", "mystical"]
            }
        ]

        # 프로필 객체 생성
        for profile_data in profiles_data:
            profile = IngredientCompatibilityProfile(**profile_data)
            self.ingredient_profiles[profile.ingredient_name] = profile

        # 인덱스 매핑 생성
        self.ingredient_index = {name: i for i, name in enumerate(self.ingredient_profiles.keys())}
        self.index_to_ingredient = {i: name for name, i in self.ingredient_index.items()}

        logger.info(f"Initialized {len(self.ingredient_profiles)} ingredient profiles")

    def calculate_compatibility(
        self,
        ingredient1: str,
        ingredient2: str,
        include_perfumer_knowledge: bool = True
    ) -> CompatibilityScore:
        """두 향료간 호환성 계산"""

        if ingredient1 not in self.ingredient_profiles or ingredient2 not in self.ingredient_profiles:
            logger.warning(f"Unknown ingredients: {ingredient1} or {ingredient2}")
            return CompatibilityScore(
                score=0.5,
                level=CompatibilityLevel.MODERATE,
                reasoning=["알 수 없는 향료"],
                chemical_basis="정보 부족",
                confidence=0.0
            )

        profile1 = self.ingredient_profiles[ingredient1]
        profile2 = self.ingredient_profiles[ingredient2]

        # 화학적 호환성 계산
        chemical_score, chemical_reasoning = self.chemical_engine.calculate_chemical_compatibility(profile1, profile2)

        # 마스터 조향사 지식 적용
        perfumer_score = 0.5
        perfumer_reasoning = []

        if include_perfumer_knowledge:
            perfumer_score = self.perfumer_knowledge.get_harmony_score(ingredient1, ingredient2)

            # 검증된 조합 확인
            for combo in self.learned_combinations:
                if ingredient1 in combo["combination"] and ingredient2 in combo["combination"]:
                    perfumer_score = max(perfumer_score, combo["harmony_score"])
                    perfumer_reasoning.append(f"마스터 검증: {combo['perfumer']} - {combo['fragrance']}")

        # 최종 점수 계산 (가중 평균)
        final_score = chemical_score * 0.6 + perfumer_score * 0.4

        # 호환성 레벨 결정
        level = self._determine_compatibility_level(final_score)

        # 시너지/갈등 요인 분석
        synergy_factors, conflict_factors = self._analyze_interaction_factors(profile1, profile2, final_score)

        # 화학적 근거 설명
        chemical_basis = self._generate_chemical_basis(profile1, profile2)

        # 신뢰도 계산
        confidence = self._calculate_confidence(ingredient1, ingredient2, final_score)

        # 조향사 노트 생성
        perfumer_notes = self._generate_perfumer_notes(ingredient1, ingredient2, final_score, level)

        return CompatibilityScore(
            score=final_score,
            level=level,
            reasoning=chemical_reasoning + perfumer_reasoning,
            chemical_basis=chemical_basis,
            perfumer_notes=perfumer_notes,
            confidence=confidence,
            synergy_factors=synergy_factors,
            conflict_factors=conflict_factors
        )

    def _determine_compatibility_level(self, score: float) -> CompatibilityLevel:
        """점수를 바탕으로 호환성 레벨 결정"""
        if score >= 0.9:
            return CompatibilityLevel.PERFECT
        elif score >= 0.8:
            return CompatibilityLevel.EXCELLENT
        elif score >= 0.7:
            return CompatibilityLevel.GOOD
        elif score >= 0.6:
            return CompatibilityLevel.MODERATE
        elif score >= 0.5:
            return CompatibilityLevel.FAIR
        elif score >= 0.4:
            return CompatibilityLevel.POOR
        else:
            return CompatibilityLevel.INCOMPATIBLE

    def _analyze_interaction_factors(
        self,
        profile1: IngredientCompatibilityProfile,
        profile2: IngredientCompatibilityProfile,
        score: float
    ) -> Tuple[List[str], List[str]]:
        """상호작용 요인 분석"""

        synergy_factors = []
        conflict_factors = []

        # 휘발성 분석
        volatility_diff = abs(profile1.volatility - profile2.volatility)
        if volatility_diff < 0.2:
            synergy_factors.append("유사한 휘발성으로 동시 발현")
        elif volatility_diff > 0.6:
            synergy_factors.append("보완적 휘발성으로 층감 생성")

        # 강도 분석
        intensity_diff = abs(profile1.intensity - profile2.intensity)
        if intensity_diff > 0.5:
            conflict_factors.append("강도 차이로 인한 불균형 가능")
        else:
            synergy_factors.append("균형잡힌 강도")

        # 지속성 분석
        longevity_avg = (profile1.longevity + profile2.longevity) / 2
        if longevity_avg > 0.7:
            synergy_factors.append("우수한 지속성")
        elif longevity_avg < 0.4:
            conflict_factors.append("짧은 지속성")

        # 특성 분석
        common_characteristics = set(profile1.dominant_characteristics) & set(profile2.dominant_characteristics)
        if common_characteristics:
            synergy_factors.append(f"공통 특성: {', '.join(common_characteristics)}")

        # 분자량 고려
        if profile1.molecular_weight and profile2.molecular_weight:
            mw_ratio = max(profile1.molecular_weight, profile2.molecular_weight) / min(profile1.molecular_weight, profile2.molecular_weight)
            if mw_ratio > 3:
                conflict_factors.append("큰 분자량 차이")

        return synergy_factors, conflict_factors

    def _generate_chemical_basis(
        self,
        profile1: IngredientCompatibilityProfile,
        profile2: IngredientCompatibilityProfile
    ) -> str:
        """화학적 근거 설명 생성"""

        basis_elements = []

        # 관능기 호환성
        if profile1.functional_groups and profile2.functional_groups:
            common_groups = set(profile1.functional_groups) & set(profile2.functional_groups)
            if common_groups:
                basis_elements.append(f"공통 관능기 ({', '.join(common_groups)})로 인한 친화성")

        # 화학 패밀리
        if profile1.chemical_family == profile2.chemical_family:
            basis_elements.append(f"동일 화학 패밀리 ({profile1.chemical_family}) 소속")
        else:
            basis_elements.append(f"서로 다른 화학 패밀리 ({profile1.chemical_family} vs {profile2.chemical_family})의 상호작용")

        # 극성 호환성
        if profile1.polarity is not None and profile2.polarity is not None:
            polarity_diff = abs(profile1.polarity - profile2.polarity)
            if polarity_diff < 0.3:
                basis_elements.append("유사한 극성으로 인한 호환성")
            else:
                basis_elements.append("극성 차이로 인한 복합적 상호작용")

        return "; ".join(basis_elements) if basis_elements else "기본적인 분자간 상호작용"

    def _calculate_confidence(self, ingredient1: str, ingredient2: str, score: float) -> float:
        """신뢰도 계산"""
        confidence_factors = []

        # 데이터 완전성
        profile1 = self.ingredient_profiles[ingredient1]
        profile2 = self.ingredient_profiles[ingredient2]

        data_completeness = 0
        total_fields = 7  # 주요 필드 수

        for profile in [profile1, profile2]:
            filled_fields = sum([
                1 if profile.chemical_family else 0,
                1 if profile.functional_groups else 0,
                1 if profile.volatility is not None else 0,
                1 if profile.intensity is not None else 0,
                1 if profile.longevity is not None else 0,
                1 if profile.molecular_weight is not None else 0,
                1 if profile.polarity is not None else 0
            ])
            data_completeness += filled_fields / total_fields

        confidence_factors.append(data_completeness / 2)  # 두 프로필의 평균

        # 검증된 조합 여부
        is_validated = any(
            ingredient1 in combo["combination"] and ingredient2 in combo["combination"]
            for combo in self.learned_combinations
        )
        confidence_factors.append(1.0 if is_validated else 0.7)

        # 점수의 극값 여부 (극값일 경우 신뢰도 약간 감소)
        if score < 0.2 or score > 0.9:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(1.0)

        return sum(confidence_factors) / len(confidence_factors)

    def _generate_perfumer_notes(
        self,
        ingredient1: str,
        ingredient2: str,
        score: float,
        level: CompatibilityLevel
    ) -> str:
        """조향사 관점의 노트 생성"""

        profile1 = self.ingredient_profiles[ingredient1]
        profile2 = self.ingredient_profiles[ingredient2]

        if level == CompatibilityLevel.PERFECT:
            return f"{ingredient1}과 {ingredient2}의 완벽한 조화. 서로의 장점을 극대화하며 새로운 차원의 향을 창조합니다."

        elif level == CompatibilityLevel.EXCELLENT:
            return f"{ingredient1}과 {ingredient2}가 탁월한 시너지를 보입니다. 조향에서 핵심적인 역할을 할 수 있는 조합입니다."

        elif level == CompatibilityLevel.GOOD:
            return f"{ingredient1}과 {ingredient2}는 좋은 조화를 이룹니다. 적절한 비율로 사용하면 균형잡힌 향을 만들 수 있습니다."

        elif level == CompatibilityLevel.MODERATE:
            return f"{ingredient1}과 {ingredient2}는 보통 수준의 조화를 보입니다. 보조적인 역할이나 배경 향으로 활용 가능합니다."

        elif level == CompatibilityLevel.FAIR:
            return f"{ingredient1}과 {ingredient2}는 무난한 조합입니다. 특별한 효과는 기대하기 어렵지만 안정적입니다."

        elif level == CompatibilityLevel.POOR:
            return f"{ingredient1}과 {ingredient2}의 조화는 제한적입니다. 사용시 주의가 필요하며 다른 향료로 보완해야 할 수 있습니다."

        else:  # INCOMPATIBLE
            return f"{ingredient1}과 {ingredient2}는 상충하는 특성을 가집니다. 함께 사용할 경우 부조화를 일으킬 가능성이 높습니다."

    def build_full_matrix(self) -> np.ndarray:
        """전체 호환성 매트릭스 구축"""

        ingredients = list(self.ingredient_profiles.keys())
        n = len(ingredients)
        matrix = np.zeros((n, n))

        logger.info(f"Building compatibility matrix for {n} ingredients...")

        for i, ing1 in enumerate(ingredients):
            for j, ing2 in enumerate(ingredients):
                if i == j:
                    matrix[i][j] = 1.0  # 자기 자신과는 완벽한 호환성
                elif i < j:  # 대칭 매트릭스이므로 한 번만 계산
                    compatibility = self.calculate_compatibility(ing1, ing2)
                    matrix[i][j] = compatibility.score
                    matrix[j][i] = compatibility.score  # 대칭 복사
                else:
                    continue  # 이미 계산됨

        self.compatibility_matrix = matrix
        logger.info("Compatibility matrix built successfully")

        return matrix

    def get_best_partners(
        self,
        ingredient: str,
        top_k: int = 5,
        min_score: float = 0.7
    ) -> List[Tuple[str, CompatibilityScore]]:
        """특정 향료의 최적 파트너 찾기"""

        if ingredient not in self.ingredient_profiles:
            logger.warning(f"Unknown ingredient: {ingredient}")
            return []

        partners = []

        for other_ingredient in self.ingredient_profiles.keys():
            if other_ingredient == ingredient:
                continue

            compatibility = self.calculate_compatibility(ingredient, other_ingredient)

            if compatibility.score >= min_score:
                partners.append((other_ingredient, compatibility))

        # 점수순으로 정렬
        partners.sort(key=lambda x: x[1].score, reverse=True)

        return partners[:top_k]

    def get_worst_partners(
        self,
        ingredient: str,
        top_k: int = 5,
        max_score: float = 0.5
    ) -> List[Tuple[str, CompatibilityScore]]:
        """특정 향료의 최악 파트너 찾기"""

        if ingredient not in self.ingredient_profiles:
            return []

        partners = []

        for other_ingredient in self.ingredient_profiles.keys():
            if other_ingredient == ingredient:
                continue

            compatibility = self.calculate_compatibility(ingredient, other_ingredient)

            if compatibility.score <= max_score:
                partners.append((other_ingredient, compatibility))

        # 점수 오름차순으로 정렬 (낮은 점수가 먼저)
        partners.sort(key=lambda x: x[1].score)

        return partners[:top_k]

    def analyze_blend_compatibility(
        self,
        ingredients: List[str]
    ) -> Dict[str, Any]:
        """블렌드 전체의 호환성 분석"""

        if len(ingredients) < 2:
            return {"error": "최소 2개 이상의 향료가 필요합니다"}

        # 모든 쌍의 호환성 계산
        compatibility_scores = []
        detailed_analysis = {}

        for i, ing1 in enumerate(ingredients):
            for j, ing2 in enumerate(ingredients[i+1:], i+1):
                compatibility = self.calculate_compatibility(ing1, ing2)
                compatibility_scores.append(compatibility.score)
                detailed_analysis[f"{ing1}-{ing2}"] = compatibility

        # 전체 통계
        avg_compatibility = np.mean(compatibility_scores) if compatibility_scores else 0
        min_compatibility = np.min(compatibility_scores) if compatibility_scores else 0
        max_compatibility = np.max(compatibility_scores) if compatibility_scores else 0
        compatibility_std = np.std(compatibility_scores) if compatibility_scores else 0

        # 문제가 되는 조합 식별
        problematic_pairs = [
            (pair, comp) for pair, comp in detailed_analysis.items()
            if comp.score < 0.5
        ]

        # 우수한 조합 식별
        excellent_pairs = [
            (pair, comp) for pair, comp in detailed_analysis.items()
            if comp.score >= 0.8
        ]

        # 전체 등급 결정
        overall_grade = "S" if avg_compatibility >= 0.9 else \
                      "A" if avg_compatibility >= 0.8 else \
                      "B" if avg_compatibility >= 0.7 else \
                      "C" if avg_compatibility >= 0.6 else \
                      "D"

        return {
            "overall_compatibility": avg_compatibility,
            "compatibility_grade": overall_grade,
            "statistics": {
                "average": avg_compatibility,
                "minimum": min_compatibility,
                "maximum": max_compatibility,
                "standard_deviation": compatibility_std,
                "total_pairs": len(compatibility_scores)
            },
            "excellent_combinations": [(pair, comp.score) for pair, comp in excellent_pairs],
            "problematic_combinations": [(pair, comp.score) for pair, comp in problematic_pairs],
            "detailed_analysis": detailed_analysis,
            "recommendations": self._generate_blend_recommendations(
                ingredients, avg_compatibility, problematic_pairs, excellent_pairs
            )
        }

    def _generate_blend_recommendations(
        self,
        ingredients: List[str],
        avg_compatibility: float,
        problematic_pairs: List,
        excellent_pairs: List
    ) -> List[str]:
        """블렌드 개선 추천"""

        recommendations = []

        if avg_compatibility < 0.6:
            recommendations.append("전체적인 호환성이 낮습니다. 핵심 향료를 중심으로 재구성을 고려하세요.")

        if problematic_pairs:
            problematic_ingredients = set()
            for pair, _ in problematic_pairs:
                ing1, ing2 = pair.split('-')
                problematic_ingredients.update([ing1, ing2])

            recommendations.append(f"문제가 되는 향료들: {', '.join(problematic_ingredients)}. 대체 향료를 검토하세요.")

        if excellent_pairs:
            recommendations.append(f"{len(excellent_pairs)}개의 우수한 조합이 발견되었습니다. 이를 중심으로 블렌드를 발전시키세요.")

        if len(ingredients) > 15:
            recommendations.append("향료 수가 많습니다. 핵심 향료에 집중하여 단순화를 고려하세요.")

        if avg_compatibility > 0.8:
            recommendations.append("전체적으로 우수한 호환성을 보입니다. 현재 구성을 유지하면서 미세 조정하세요.")

        return recommendations

    def export_matrix(self, filepath: str, format: str = "csv") -> bool:
        """호환성 매트릭스 내보내기"""

        try:
            if self.compatibility_matrix is None:
                self.build_full_matrix()

            ingredients = list(self.ingredient_profiles.keys())

            if format.lower() == "csv":
                df = pd.DataFrame(
                    self.compatibility_matrix,
                    index=ingredients,
                    columns=ingredients
                )
                df.to_csv(filepath)

            elif format.lower() == "excel":
                df = pd.DataFrame(
                    self.compatibility_matrix,
                    index=ingredients,
                    columns=ingredients
                )
                df.to_excel(filepath)

            elif format.lower() == "json":
                data = {
                    "ingredients": ingredients,
                    "matrix": self.compatibility_matrix.tolist(),
                    "exported_at": datetime.now().isoformat()
                }
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

            elif format.lower() == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump({
                        "matrix": self.compatibility_matrix,
                        "ingredient_index": self.ingredient_index,
                        "index_to_ingredient": self.index_to_ingredient,
                        "profiles": self.ingredient_profiles
                    }, f)

            logger.info(f"Matrix exported to {filepath} in {format} format")
            return True

        except Exception as e:
            logger.error(f"Failed to export matrix: {e}")
            return False

    def visualize_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ) -> None:
        """호환성 매트릭스 시각화"""

        if self.compatibility_matrix is None:
            self.build_full_matrix()

        ingredients = list(self.ingredient_profiles.keys())

        plt.figure(figsize=figsize)
        sns.heatmap(
            self.compatibility_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            xticklabels=ingredients,
            yticklabels=ingredients,
            cbar_kws={'label': 'Compatibility Score'}
        )

        plt.title('Fragrance Ingredient Compatibility Matrix', fontsize=16, pad=20)
        plt.xlabel('Ingredients', fontsize=12)
        plt.ylabel('Ingredients', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matrix visualization saved to {save_path}")

        plt.show()

    def create_compatibility_network(
        self,
        min_compatibility: float = 0.7,
        save_path: Optional[str] = None
    ) -> nx.Graph:
        """호환성 네트워크 생성 및 시각화"""

        if self.compatibility_matrix is None:
            self.build_full_matrix()

        # 네트워크 그래프 생성
        G = nx.Graph()

        ingredients = list(self.ingredient_profiles.keys())

        # 노드 추가
        for ingredient in ingredients:
            G.add_node(ingredient)

        # 엣지 추가 (일정 호환성 이상만)
        for i, ing1 in enumerate(ingredients):
            for j, ing2 in enumerate(ingredients[i+1:], i+1):
                compatibility = self.compatibility_matrix[i][j]
                if compatibility >= min_compatibility:
                    G.add_edge(ing1, ing2, weight=compatibility)

        # 시각화
        plt.figure(figsize=(14, 10))

        pos = nx.spring_layout(G, k=2, iterations=50)

        # 엣지 그리기 (호환성에 따라 선 굵기 조절)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]

        nx.draw_networkx_edges(
            G, pos,
            width=[w * 3 for w in weights],
            alpha=0.6,
            edge_color=weights,
            edge_cmap=plt.cm.RdYlGn
        )

        # 노드 그리기
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=1500,
            alpha=0.9
        )

        # 라벨 그리기
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_weight='bold'
        )

        plt.title(f'Fragrance Compatibility Network (min compatibility: {min_compatibility})',
                  fontsize=16, pad=20)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network visualization saved to {save_path}")

        plt.show()

        return G

    def get_compatibility_insights(self, ingredient: str) -> Dict[str, Any]:
        """특정 향료의 호환성 인사이트"""

        if ingredient not in self.ingredient_profiles:
            return {"error": f"Unknown ingredient: {ingredient}"}

        profile = self.ingredient_profiles[ingredient]
        best_partners = self.get_best_partners(ingredient, top_k=5)
        worst_partners = self.get_worst_partners(ingredient, top_k=3)

        # 특성 분석
        characteristics_analysis = {
            "volatility_level": "high" if profile.volatility > 0.7 else "medium" if profile.volatility > 0.4 else "low",
            "intensity_level": "strong" if profile.intensity > 0.8 else "moderate" if profile.intensity > 0.6 else "soft",
            "longevity_level": "excellent" if profile.longevity > 0.8 else "good" if profile.longevity > 0.6 else "limited",
            "dominant_character": profile.dominant_characteristics[0] if profile.dominant_characteristics else "neutral"
        }

        # 사용 가이드라인 생성
        guidelines = self._generate_usage_guidelines(ingredient, profile, best_partners)

        return {
            "ingredient_profile": {
                "name": ingredient,
                "chemical_family": profile.chemical_family,
                "characteristics": characteristics_analysis,
                "functional_groups": profile.functional_groups,
                "molecular_properties": {
                    "molecular_weight": profile.molecular_weight,
                    "polarity": profile.polarity
                }
            },
            "compatibility_summary": {
                "best_partners": [(partner, comp.score) for partner, comp in best_partners],
                "challenging_combinations": [(partner, comp.score) for partner, comp in worst_partners]
            },
            "usage_guidelines": guidelines,
            "blending_tips": self._generate_blending_tips(ingredient, profile, best_partners)
        }

    def _generate_usage_guidelines(
        self,
        ingredient: str,
        profile: IngredientCompatibilityProfile,
        best_partners: List[Tuple[str, CompatibilityScore]]
    ) -> List[str]:
        """사용 가이드라인 생성"""

        guidelines = []

        # 휘발성 기반 가이드라인
        if profile.volatility > 0.8:
            guidelines.append("탑노트로 사용하여 첫인상을 강화하세요")
        elif profile.volatility < 0.3:
            guidelines.append("베이스노트로 사용하여 지속력을 제공하세요")
        else:
            guidelines.append("하트노트로 사용하여 향의 핵심을 구성하세요")

        # 강도 기반 가이드라인
        if profile.intensity > 0.8:
            guidelines.append("강한 향이므로 소량 사용을 권장합니다 (전체의 5-10%)")
        elif profile.intensity < 0.5:
            guidelines.append("부드러운 향이므로 충분한 양 사용 가능합니다 (전체의 15-25%)")

        # 최적 파트너 기반 가이드라인
        if best_partners:
            top_partner = best_partners[0][0]
            guidelines.append(f"{top_partner}와의 조합을 특히 추천합니다")

        return guidelines

    def _generate_blending_tips(
        self,
        ingredient: str,
        profile: IngredientCompatibilityProfile,
        best_partners: List[Tuple[str, CompatibilityScore]]
    ) -> List[str]:
        """블렌딩 팁 생성"""

        tips = []

        # 화학 패밀리 기반 팁
        if profile.chemical_family == "monoterpene":
            tips.append("다른 테르펜류와 잘 어우러지므로 시트러스나 허브 계열과 조합하세요")
        elif profile.chemical_family == "aromatic":
            tips.append("방향족 화합물이므로 플로럴 계열과의 조합에서 깊이를 더합니다")
        elif profile.chemical_family == "sesquiterpene":
            tips.append("세스퀴테르펜으로서 우디 베이스의 기반을 제공합니다")

        # 특성 기반 팁
        if "elegant" in profile.dominant_characteristics:
            tips.append("우아한 특성을 가지므로 고급스러운 조합에 적합합니다")

        if "powerful" in profile.dominant_characteristics:
            tips.append("강력한 특성이 있으므로 다른 향료들과의 균형을 신중히 맞추세요")

        # 파트너 기반 팁
        if len(best_partners) >= 3:
            tips.append("다양한 향료와 호환성이 좋으므로 블렌딩의 중심 역할을 할 수 있습니다")

        return tips