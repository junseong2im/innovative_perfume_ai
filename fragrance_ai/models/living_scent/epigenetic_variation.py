"""
Living Scent - Epigenetic Variation AI
진화 계층 - 생명체가 환경에 적응하며 분화하는 '후생유전학적 변형'

원본 DNA는 바꾸지 않고, 사용자의 피드백이라는 외부 자극에 따라
유전자의 '발현' 방식을 조절하여 새로운 버전을 만듭니다.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import copy
import json
import numpy as np
from datetime import datetime
import hashlib
from enum import Enum


class EpigeneticMarker(Enum):
    """후생유전학적 마커 유형"""
    AMPLIFICATION = "amplification"  # 증폭
    SILENCING = "silencing"  # 침묵
    MODULATION = "modulation"  # 조절
    SUBSTITUTION = "substitution"  # 대체


@dataclass
class EpigeneticModification:
    """후생유전학적 수정 기록"""
    marker_type: EpigeneticMarker
    target_gene: str  # 수정 대상 유전자
    modification_factor: float  # 수정 강도
    trigger: str  # 수정 유발 요인 (사용자 피드백)
    timestamp: str


@dataclass
class ScentPhenotype:
    """향수 표현형 - DNA가 실제로 발현된 결과"""
    phenotype_id: str  # 고유 식별자
    based_on_dna: str  # 기반 DNA ID
    epigenetic_trigger: str  # 후생유전학적 유발 요인
    recipe: Dict[str, List[str]]  # 실제 레시피
    modifications: List[EpigeneticModification]  # 적용된 수정 목록
    description: str  # 표현형 설명
    environmental_response: Dict[str, float]  # 환경 반응성
    creation_timestamp: str
    parent_phenotype: Optional[str] = None  # 부모 표현형 ID
    evolution_path: List[str] = field(default_factory=list)  # 진화 경로


class EpigeneticRegulator:
    """후생유전학적 조절자 - 유전자 발현 조절"""

    def __init__(self):
        # 피드백 해석 매핑
        self.feedback_interpretations = {
            # 강도 조절
            'stronger': {'type': EpigeneticMarker.AMPLIFICATION, 'factor': 1.5},
            '강하게': {'type': EpigeneticMarker.AMPLIFICATION, 'factor': 1.5},
            '더 강하게': {'type': EpigeneticMarker.AMPLIFICATION, 'factor': 1.8},
            'weaker': {'type': EpigeneticMarker.SILENCING, 'factor': 0.5},
            '약하게': {'type': EpigeneticMarker.SILENCING, 'factor': 0.5},
            '살짝': {'type': EpigeneticMarker.SILENCING, 'factor': 0.7},

            # 특성 추가/제거
            'smoky': {'type': EpigeneticMarker.SUBSTITUTION, 'target': 'woody', 'substitute': 'smoky'},
            '스모키': {'type': EpigeneticMarker.SUBSTITUTION, 'target': 'woody', 'substitute': 'smoky'},
            'fresh': {'type': EpigeneticMarker.MODULATION, 'target': 'citrus', 'factor': 1.3},
            '상큼': {'type': EpigeneticMarker.MODULATION, 'target': 'citrus', 'factor': 1.3},
            'sweeter': {'type': EpigeneticMarker.MODULATION, 'target': 'sweet', 'factor': 1.4},
            '달콤': {'type': EpigeneticMarker.MODULATION, 'target': 'sweet', 'factor': 1.4},

            # 지속성 조절
            'longer lasting': {'type': EpigeneticMarker.AMPLIFICATION, 'target': 'base', 'factor': 1.3},
            '오래가는': {'type': EpigeneticMarker.AMPLIFICATION, 'target': 'base', 'factor': 1.3},
            'lighter': {'type': EpigeneticMarker.SILENCING, 'target': 'base', 'factor': 0.6},
            '가볍게': {'type': EpigeneticMarker.SILENCING, 'target': 'base', 'factor': 0.6},
        }

        # 유전자 대체 맵
        self.gene_substitution_map = {
            'woody': {
                'smoky': ['Birch Tar', 'Cade', 'Guaiac Wood'],
                'earthy': ['Vetiver', 'Patchouli', 'Oakmoss'],
                'creamy': ['Sandalwood', 'Cashmeran', 'Blonde Woods']
            },
            'floral': {
                'powdery': ['Iris', 'Violet', 'Heliotrope'],
                'fresh': ['Freesia', 'Lily of the Valley', 'Peony'],
                'indolic': ['Jasmine', 'Tuberose', 'Orange Blossom']
            },
            'citrus': {
                'sparkling': ['Bergamot', 'Lemon', 'Yuzu'],
                'bitter': ['Grapefruit', 'Bitter Orange', 'Petitgrain'],
                'sweet': ['Mandarin', 'Sweet Orange', 'Tangerine']
            }
        }

    def interpret_feedback(self, feedback_text: str) -> List[Dict[str, Any]]:
        """피드백 텍스트를 후생유전학적 지시로 해석"""
        modifications = []
        feedback_lower = feedback_text.lower()

        for keyword, instruction in self.feedback_interpretations.items():
            if keyword in feedback_lower:
                modifications.append(instruction)

        # 기본 수정 (피드백이 애매한 경우)
        if not modifications:
            modifications.append({
                'type': EpigeneticMarker.MODULATION,
                'factor': 1.1
            })

        return modifications

    def apply_methylation(self, gene, modification_factor: float):
        """메틸화 - 유전자 발현 수준 조절"""
        gene.expression_level *= modification_factor
        gene.expression_level = max(0.0, min(1.0, gene.expression_level))
        return gene

    def apply_histone_modification(self, gene, target_family: str, factor: float):
        """히스톤 수정 - 특정 계열 유전자 조절"""
        if target_family in gene.odor_family:
            gene.expression_level *= factor
            gene.concentration *= factor
            gene.expression_level = max(0.0, min(1.0, gene.expression_level))
            gene.concentration = max(0.1, min(1.0, gene.concentration))
        return gene

    def apply_chromatin_remodeling(self, genotype: Dict, target_note: str, factor: float) -> Dict:
        """크로마틴 리모델링 - 특정 노트 전체 조절"""
        if target_note in genotype:
            for gene in genotype[target_note]:
                gene.expression_level *= factor
                gene.expression_level = max(0.0, min(1.0, gene.expression_level))
        return genotype


class EpigeneticVariationAI:
    """
    후생유전학적 변형 AI - 사용자 피드백에 따라
    DNA의 발현을 조절하여 새로운 표현형을 생성
    """

    def __init__(self):
        self.regulator = EpigeneticRegulator()
        # 표현형 라이브러리 (메모리 캐시)
        self.phenotype_library = {}

    def generate_phenotype_id(self, dna_id: str, trigger: str) -> str:
        """고유한 표현형 ID 생성"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{dna_id}_{trigger}_{timestamp}"
        phenotype_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:10]
        return f"PHENO_{phenotype_hash.upper()}"

    def load_dna(self, dna_id: str, dna_library: Dict) -> Any:
        """DNA 라이브러리에서 원본 DNA 로드"""
        return dna_library.get(dna_id)

    def apply_epigenetic_modifications(
        self,
        original_dna: Any,
        modifications: List[Dict[str, Any]]
    ) -> Tuple[Dict, List[EpigeneticModification]]:
        """후생유전학적 수정 적용"""
        # 원본 DNA의 깊은 복사
        modified_genotype = copy.deepcopy(original_dna.genotype)
        modification_records = []

        for mod in modifications:
            marker_type = mod['type']
            factor = mod.get('factor', 1.0)
            target = mod.get('target', 'all')

            timestamp = datetime.now().isoformat()

            if marker_type == EpigeneticMarker.AMPLIFICATION:
                # 증폭: 특정 노트 또는 전체 강화
                if target in ['top', 'middle', 'base']:
                    modified_genotype = self.regulator.apply_chromatin_remodeling(
                        modified_genotype, target, factor
                    )
                else:
                    # 전체 증폭
                    for note_type in modified_genotype:
                        for gene in modified_genotype[note_type]:
                            gene = self.regulator.apply_methylation(gene, factor)

                modification_records.append(
                    EpigeneticModification(
                        marker_type=marker_type,
                        target_gene=target,
                        modification_factor=factor,
                        trigger="User requested amplification",
                        timestamp=timestamp
                    )
                )

            elif marker_type == EpigeneticMarker.SILENCING:
                # 침묵: 특정 노트 또는 전체 약화
                if target in ['top', 'middle', 'base']:
                    modified_genotype = self.regulator.apply_chromatin_remodeling(
                        modified_genotype, target, factor
                    )
                else:
                    # 전체 약화
                    for note_type in modified_genotype:
                        for gene in modified_genotype[note_type]:
                            gene = self.regulator.apply_methylation(gene, factor)

                modification_records.append(
                    EpigeneticModification(
                        marker_type=marker_type,
                        target_gene=target,
                        modification_factor=factor,
                        trigger="User requested silencing",
                        timestamp=timestamp
                    )
                )

            elif marker_type == EpigeneticMarker.MODULATION:
                # 조절: 특정 향 계열 조절
                target_family = mod.get('target', 'floral')
                for note_type in modified_genotype:
                    for gene in modified_genotype[note_type]:
                        gene = self.regulator.apply_histone_modification(
                            gene, target_family, factor
                        )

                modification_records.append(
                    EpigeneticModification(
                        marker_type=marker_type,
                        target_gene=target_family,
                        modification_factor=factor,
                        trigger="User requested modulation",
                        timestamp=timestamp
                    )
                )

            elif marker_type == EpigeneticMarker.SUBSTITUTION:
                # 대체: 특정 향료를 다른 것으로 교체
                target_family = mod.get('target', 'woody')
                substitute_type = mod.get('substitute', 'smoky')

                substitutes = self.regulator.gene_substitution_map.get(
                    target_family, {}
                ).get(substitute_type, [])

                if substitutes:
                    for note_type in modified_genotype:
                        for i, gene in enumerate(modified_genotype[note_type]):
                            if target_family in gene.odor_family:
                                # 새로운 향료로 교체
                                from fragrance_ai.models.living_scent.olfactory_recombinator import FragranceGene
                                new_ingredient = np.random.choice(substitutes)
                                modified_genotype[note_type][i] = FragranceGene(
                                    note_type=gene.note_type,
                                    ingredient=new_ingredient,
                                    concentration=gene.concentration,
                                    volatility=gene.volatility,
                                    molecular_weight=gene.molecular_weight,
                                    odor_family=f"{substitute_type}-{target_family}",
                                    expression_level=gene.expression_level * 1.2
                                )

                modification_records.append(
                    EpigeneticModification(
                        marker_type=marker_type,
                        target_gene=f"{target_family}->{substitute_type}",
                        modification_factor=1.0,
                        trigger="User requested substitution",
                        timestamp=timestamp
                    )
                )

        return modified_genotype, modification_records

    def generate_description(
        self,
        original_dna: Any,
        modifications: List[EpigeneticModification],
        feedback_brief: Any
    ) -> str:
        """표현형 설명 생성"""
        mod_descriptions = []
        for mod in modifications:
            if mod.marker_type == EpigeneticMarker.AMPLIFICATION:
                mod_descriptions.append(f"amplified {mod.target_gene}")
            elif mod.marker_type == EpigeneticMarker.SILENCING:
                mod_descriptions.append(f"softened {mod.target_gene}")
            elif mod.marker_type == EpigeneticMarker.MODULATION:
                mod_descriptions.append(f"modulated {mod.target_gene} notes")
            elif mod.marker_type == EpigeneticMarker.SUBSTITUTION:
                mod_descriptions.append(f"transformed into {mod.target_gene}")

        modification_summary = ", ".join(mod_descriptions) if mod_descriptions else "subtly refined"

        description = (
            f"The original {original_dna.story.split('.')[0]}, "
            f"now {modification_summary}, "
            f"responding to the desire for {feedback_brief.core_emotion}. "
            f"This phenotype represents an evolution, not a replacement - "
            f"the DNA remembers its origins while embracing change."
        )

        return description

    def calculate_environmental_response(
        self,
        modified_genotype: Dict,
        feedback_brief: Any
    ) -> Dict[str, float]:
        """환경 반응성 계산"""
        response = {}

        # 온도 반응성 (휘발성 기반)
        all_volatilities = []
        for note_type in modified_genotype.values():
            all_volatilities.extend([g.volatility for g in note_type])
        response['temperature_sensitivity'] = np.mean(all_volatilities) if all_volatilities else 0.5

        # 습도 반응성 (분자량 기반)
        all_weights = []
        for note_type in modified_genotype.values():
            all_weights.extend([g.molecular_weight for g in note_type])
        response['humidity_response'] = 1 - (np.mean(all_weights) / 500) if all_weights else 0.5

        # 시간 안정성 (base note 강도 기반)
        base_expressions = [g.expression_level for g in modified_genotype.get('base', [])]
        response['temporal_stability'] = np.mean(base_expressions) if base_expressions else 0.3

        # 감정 반응성 (feedback brief 기반)
        emotion_intensity = np.mean(list(feedback_brief.emotional_palette.values()))
        response['emotional_resonance'] = emotion_intensity

        return response

    def genotype_to_recipe(self, genotype: Dict) -> Dict[str, List[str]]:
        """유전자형을 레시피로 변환"""
        recipe = {'top': [], 'middle': [], 'base': []}

        for note_type, genes in genotype.items():
            for gene in genes:
                if gene.expression_level > 0.2:  # 발현 임계값
                    # 발현 수준에 따라 마커 추가
                    if gene.expression_level > 0.8:
                        marker = "(Dominant)"
                    elif gene.expression_level > 0.5:
                        marker = "(Present)"
                    else:
                        marker = "(Trace)"

                    ingredient_str = f"{gene.ingredient} {marker}"
                    recipe[note_type].append(ingredient_str)

        return recipe

    def evolve(
        self,
        dna_id: str,
        feedback_brief: Any,
        dna_library: Dict,
        parent_phenotype_id: Optional[str] = None,
        use_rlhf: bool = True,
        user_rating: Optional[float] = None
    ) -> ScentPhenotype:
        """
        DNA를 후생유전학적으로 변형하여 새로운 표현형 생성
        RLHF 옵션 추가
        """
        # 1. 원본 DNA 로드
        original_dna = self.load_dna(dna_id, dna_library)
        if not original_dna:
            raise ValueError(f"DNA {dna_id} not found in library")

        # 2. RLHF 사용 여부에 따라 분기
        if use_rlhf and user_rating is not None:
            # 강화학습 기반 진화
            from fragrance_ai.training.reinforcement_learning import get_fragrance_rlhf
            rlhf_system = get_fragrance_rlhf()

            # 강화학습으로 수정 지시 생성
            rl_instructions = rlhf_system.evolve_fragrance(
                original_dna,
                feedback_brief.story,
                user_rating
            )

            # RL 지시를 후생유전학적 지시로 변환
            modifications_instructions = [{
                'type': EpigeneticMarker.AMPLIFICATION if rl_instructions['type'] == 'amplify'
                       else EpigeneticMarker.SILENCING if rl_instructions['type'] == 'silence'
                       else EpigeneticMarker.MODULATION if rl_instructions['type'] == 'modulate'
                       else EpigeneticMarker.SUBSTITUTION,
                'target': rl_instructions['target'],
                'factor': 1 + rl_instructions['strength'] if rl_instructions['type'] == 'amplify'
                         else rl_instructions['strength'],
            }]
        else:
            # 기존 규칙 기반 해석
            modifications_instructions = self.regulator.interpret_feedback(feedback_brief.story)

        # 3. 후생유전학적 수정 적용
        modified_genotype, modification_records = self.apply_epigenetic_modifications(
            original_dna,
            modifications_instructions
        )

        # 4. 표현형 ID 생성
        phenotype_id = self.generate_phenotype_id(dna_id, feedback_brief.core_emotion)

        # 5. 레시피 생성
        recipe = self.genotype_to_recipe(modified_genotype)

        # 6. 설명 생성
        description = self.generate_description(
            original_dna,
            modification_records,
            feedback_brief
        )

        # 7. 환경 반응성 계산
        environmental_response = self.calculate_environmental_response(
            modified_genotype,
            feedback_brief
        )

        # 8. 진화 경로 추적
        evolution_path = []
        if parent_phenotype_id:
            parent_pheno = self.phenotype_library.get(parent_phenotype_id)
            if parent_pheno:
                evolution_path = parent_pheno.evolution_path.copy()
        evolution_path.append(phenotype_id)

        # 9. ScentPhenotype 객체 생성
        phenotype = ScentPhenotype(
            phenotype_id=phenotype_id,
            based_on_dna=dna_id,
            epigenetic_trigger=feedback_brief.story,
            recipe=recipe,
            modifications=modification_records,
            description=description,
            environmental_response=environmental_response,
            creation_timestamp=datetime.now().isoformat(),
            parent_phenotype=parent_phenotype_id,
            evolution_path=evolution_path
        )

        # 10. 표현형 라이브러리에 저장
        self.phenotype_library[phenotype_id] = phenotype

        return phenotype

    def to_json(self, phenotype: ScentPhenotype) -> str:
        """ScentPhenotype을 JSON으로 변환"""
        # EpigeneticModification 객체를 딕셔너리로 변환
        modifications_dict = [
            {
                'marker_type': mod.marker_type.value,
                'target_gene': mod.target_gene,
                'modification_factor': mod.modification_factor,
                'trigger': mod.trigger,
                'timestamp': mod.timestamp
            }
            for mod in phenotype.modifications
        ]

        return json.dumps({
            'phenotype_id': phenotype.phenotype_id,
            'based_on_dna': phenotype.based_on_dna,
            'epigenetic_trigger': phenotype.epigenetic_trigger,
            'recipe': phenotype.recipe,
            'modifications': modifications_dict,
            'description': phenotype.description,
            'environmental_response': phenotype.environmental_response,
            'creation_timestamp': phenotype.creation_timestamp,
            'parent_phenotype': phenotype.parent_phenotype,
            'evolution_path': phenotype.evolution_path
        }, ensure_ascii=False, indent=2)


# 싱글톤 인스턴스
_epigenetic_variation_instance = None

def get_epigenetic_variation() -> EpigeneticVariationAI:
    """싱글톤 EpigeneticVariationAI 인스턴스 반환"""
    global _epigenetic_variation_instance
    if _epigenetic_variation_instance is None:
        _epigenetic_variation_instance = EpigeneticVariationAI()
    return _epigenetic_variation_instance