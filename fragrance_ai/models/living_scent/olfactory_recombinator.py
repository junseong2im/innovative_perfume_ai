"""
Living Scent - Olfactory Recombinator AI
창세기 계층 - 생명의 탄생 과정에서의 '유전적 재조합'

두 개의 서로 다른 개념을 부모로 삼아
세상에 없던 새로운 유전자 조합, 즉 '향의 DNA'를 창조합니다.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import uuid
import random
import numpy as np
import json
from datetime import datetime
import hashlib

# MOGA 옵티마이저 임포트
try:
    from ...training.moga_optimizer import MOGAOptimizer, create_fragrance_optimizer
    MOGA_AVAILABLE = True
except ImportError:
    MOGA_AVAILABLE = False
    print("Warning: MOGA optimizer not available, using rule-based genetic algorithm")


@dataclass
class FragranceGene:
    """향기 유전자 - DNA를 구성하는 기본 단위"""
    note_type: str  # 'top', 'middle', 'base'
    ingredient: str  # 향료 이름
    concentration: float  # 농도 (0-1)
    volatility: float  # 휘발성 (0-1)
    molecular_weight: float  # 분자량
    odor_family: str  # 향 계열
    expression_level: float = 1.0  # 유전자 발현 수준 (후생유전학적 조절 가능)


@dataclass
class OlfactoryDNA:
    """후각 DNA - 향수의 유전 정보"""
    dna_id: str  # 고유 식별자
    lineage: List[str]  # 부모 계보
    genotype: Dict[str, List[FragranceGene]]  # 유전자형 (top, middle, base)
    phenotype_potential: Dict[str, float]  # 표현형 잠재력
    story: str  # DNA의 탄생 스토리
    creation_timestamp: str
    mutation_history: List[Dict] = field(default_factory=list)
    generation: int = 1  # 세대 수
    fitness_score: float = 0.0  # 적합도 점수


class GeneticAlgorithm:
    """유전 알고리즘 - DNA 생성과 재조합 (MOGA 통합)"""

    def __init__(self, use_moga: bool = True):
        # 향료 유전자 풀 (실제 향료 데이터베이스)
        self.gene_pool = {
            'nostalgic': {
                'top': [
                    FragranceGene('top', 'Aldehydes', 0.3, 0.9, 120, 'powdery'),
                    FragranceGene('top', 'Black Pepper', 0.2, 0.8, 160, 'spicy'),
                    FragranceGene('top', 'Pink Pepper', 0.15, 0.85, 155, 'fresh-spicy')
                ],
                'middle': [
                    FragranceGene('middle', 'Orris', 0.4, 0.5, 250, 'powdery-floral'),
                    FragranceGene('middle', 'Violet', 0.3, 0.6, 200, 'powdery-sweet'),
                    FragranceGene('middle', 'Heliotrope', 0.25, 0.55, 180, 'almond-vanilla')
                ],
                'base': [
                    FragranceGene('base', 'Cedarwood', 0.5, 0.2, 300, 'woody'),
                    FragranceGene('base', 'Sandalwood', 0.4, 0.15, 320, 'creamy-woody'),
                    FragranceGene('base', 'Amber', 0.3, 0.1, 400, 'warm-resinous')
                ]
            },
            'romantic': {
                'top': [
                    FragranceGene('top', 'Rose Petals', 0.3, 0.8, 150, 'floral'),
                    FragranceGene('top', 'Peony', 0.25, 0.75, 140, 'fresh-floral'),
                    FragranceGene('top', 'Freesia', 0.2, 0.85, 130, 'light-floral')
                ],
                'middle': [
                    FragranceGene('middle', 'Jasmine', 0.4, 0.5, 200, 'indolic-floral'),
                    FragranceGene('middle', 'Ylang-Ylang', 0.35, 0.45, 220, 'creamy-floral'),
                    FragranceGene('middle', 'Tuberose', 0.3, 0.4, 240, 'heady-floral')
                ],
                'base': [
                    FragranceGene('base', 'Musk', 0.4, 0.1, 350, 'animalic'),
                    FragranceGene('base', 'Vanilla', 0.35, 0.15, 300, 'sweet-balsamic'),
                    FragranceGene('base', 'Benzoin', 0.3, 0.12, 320, 'vanilla-resinous')
                ]
            },
            'adventurous': {
                'top': [
                    FragranceGene('top', 'Bergamot', 0.35, 0.9, 120, 'citrus'),
                    FragranceGene('top', 'Grapefruit', 0.3, 0.85, 110, 'fresh-citrus'),
                    FragranceGene('top', 'Cardamom', 0.2, 0.8, 140, 'spicy-fresh')
                ],
                'middle': [
                    FragranceGene('middle', 'Geranium', 0.3, 0.6, 180, 'green-rosy'),
                    FragranceGene('middle', 'Lavender', 0.35, 0.55, 170, 'herbal-fresh'),
                    FragranceGene('middle', 'Sage', 0.25, 0.5, 160, 'herbal-camphorous')
                ],
                'base': [
                    FragranceGene('base', 'Vetiver', 0.4, 0.1, 350, 'earthy-woody'),
                    FragranceGene('base', 'Patchouli', 0.35, 0.12, 340, 'earthy-dark'),
                    FragranceGene('base', 'Oakmoss', 0.3, 0.08, 380, 'mossy-earthy')
                ]
            }
        }

        # 재조합 규칙
        self.recombination_rules = {
            'crossover_rate': 0.7,  # 교차 확률
            'mutation_rate': 0.1,    # 돌연변이 확률
            'gene_dominance': 0.6,   # 우성 유전자 발현 확률
        }

        # MOGA 옵티마이저 초기화
        self.use_moga = use_moga and MOGA_AVAILABLE
        self.moga_optimizer = None
        if self.use_moga:
            try:
                self.moga_optimizer = create_fragrance_optimizer(num_ingredients=20)
                print("MOGA optimizer initialized successfully")
            except Exception as e:
                print(f"Failed to initialize MOGA: {e}")
                self.use_moga = False

    def select_parents(self, archetype1: str, archetype2: str) -> Tuple[Dict, Dict]:
        """부모 유전자 선택"""
        parent1 = self.gene_pool.get(archetype1, self.gene_pool['nostalgic'])
        parent2 = self.gene_pool.get(archetype2, self.gene_pool['romantic'])
        return parent1, parent2

    def crossover(self, parent1: Dict, parent2: Dict) -> Dict[str, List[FragranceGene]]:
        """유전자 교차 (Crossover)"""
        offspring = {'top': [], 'middle': [], 'base': []}

        for note_type in ['top', 'middle', 'base']:
            genes1 = parent1.get(note_type, [])
            genes2 = parent2.get(note_type, [])

            # 균일 교차 (Uniform Crossover)
            for i in range(max(len(genes1), len(genes2))):
                if random.random() < self.recombination_rules['crossover_rate']:
                    if i < len(genes1) and i < len(genes2):
                        # 두 부모의 유전자를 섞음
                        if random.random() < self.recombination_rules['gene_dominance']:
                            selected_gene = genes1[i]
                        else:
                            selected_gene = genes2[i]

                        # 유전자 복사 및 미세 조정
                        new_gene = FragranceGene(
                            note_type=selected_gene.note_type,
                            ingredient=selected_gene.ingredient,
                            concentration=selected_gene.concentration * random.uniform(0.8, 1.2),
                            volatility=selected_gene.volatility,
                            molecular_weight=selected_gene.molecular_weight,
                            odor_family=selected_gene.odor_family,
                            expression_level=random.uniform(0.7, 1.0)
                        )
                        offspring[note_type].append(new_gene)

        return offspring

    def optimize_with_moga(self, objectives: Dict[str, float] = None) -> Dict[str, List[FragranceGene]]:
        """MOGA를 사용한 최적화"""
        if not self.use_moga or not self.moga_optimizer:
            # MOGA 사용 불가시 기본 방법으로 폴백
            return self.crossover(
                self.gene_pool['nostalgic'],
                self.gene_pool['romantic']
            )

        # MOGA 최적화 실행
        pareto_front = self.moga_optimizer.optimize(verbose=False)

        if not pareto_front:
            # 최적화 실패시 기본 방법으로 폴백
            return self.crossover(
                self.gene_pool['nostalgic'],
                self.gene_pool['romantic']
            )

        # 가중치 기반 최적 해 선택
        best_solution = self.moga_optimizer.get_best_solution(objectives)

        # MOGA 결과를 FragranceGene 형식으로 변환
        genotype = {'top': [], 'middle': [], 'base': []}
        ingredients_per_note = len(best_solution.genes) // 3

        # 기본 재료 목록
        top_ingredients = ['Bergamot', 'Lemon', 'Orange', 'Grapefruit', 'Lime', 'Mandarin', 'Yuzu']
        middle_ingredients = ['Rose', 'Jasmine', 'Ylang-ylang', 'Geranium', 'Lavender', 'Violet', 'Iris']
        base_ingredients = ['Sandalwood', 'Cedarwood', 'Vetiver', 'Patchouli', 'Musk', 'Amber', 'Vanilla']

        # 탑 노트
        for i in range(min(ingredients_per_note, len(top_ingredients))):
            if best_solution.genes[i] > 0.01:  # 최소 농도 이상만 포함
                genotype['top'].append(FragranceGene(
                    note_type='top',
                    ingredient=top_ingredients[i % len(top_ingredients)],
                    concentration=best_solution.genes[i],
                    volatility=random.uniform(0.7, 0.95),
                    molecular_weight=random.uniform(100, 200),
                    odor_family='fresh'
                ))

        # 미들 노트
        for i in range(ingredients_per_note, min(2 * ingredients_per_note, ingredients_per_note + len(middle_ingredients))):
            idx = i - ingredients_per_note
            if best_solution.genes[i] > 0.01:
                genotype['middle'].append(FragranceGene(
                    note_type='middle',
                    ingredient=middle_ingredients[idx % len(middle_ingredients)],
                    concentration=best_solution.genes[i],
                    volatility=random.uniform(0.4, 0.7),
                    molecular_weight=random.uniform(200, 300),
                    odor_family='floral'
                ))

        # 베이스 노트
        for i in range(2 * ingredients_per_note, min(len(best_solution.genes), 2 * ingredients_per_note + len(base_ingredients))):
            idx = i - 2 * ingredients_per_note
            if best_solution.genes[i] > 0.01:
                genotype['base'].append(FragranceGene(
                    note_type='base',
                    ingredient=base_ingredients[idx % len(base_ingredients)],
                    concentration=best_solution.genes[i],
                    volatility=random.uniform(0.05, 0.3),
                    molecular_weight=random.uniform(300, 500),
                    odor_family='woody'
                ))

        return genotype

    def mutate(self, genotype: Dict[str, List[FragranceGene]]) -> Dict[str, List[FragranceGene]]:
        """돌연변이 (Mutation)"""
        for note_type in genotype:
            for gene in genotype[note_type]:
                if random.random() < self.recombination_rules['mutation_rate']:
                    # 농도 돌연변이
                    gene.concentration *= random.uniform(0.5, 1.5)
                    gene.concentration = min(1.0, max(0.1, gene.concentration))

                    # 발현 수준 돌연변이
                    gene.expression_level *= random.uniform(0.8, 1.2)
                    gene.expression_level = min(1.0, max(0.1, gene.expression_level))

        return genotype


class OlfactoryRecombinatorAI:
    """
    후각 재조합 AI - 생명의 탄생 과정을 모방하여
    두 개의 서로 다른 개념으로부터 새로운 향수 DNA를 창조
    """

    def __init__(self, use_moga: bool = True):
        self.genetic_algorithm = GeneticAlgorithm(use_moga=use_moga)

        # DNA 라이브러리 (메모리 캐시)
        self.dna_library = {}

        # 향수 원형 매핑
        self.archetype_mapping = {
            'nostalgia': 'nostalgic',
            'romance': 'romantic',
            'adventure': 'adventurous',
            'serenity': 'nostalgic',  # 기본값
            'mysterious': 'romantic',
            'energetic': 'adventurous'
        }

    def extract_parent_concepts(self, creative_brief: Any) -> Tuple[str, str]:
        """CreativeBrief로부터 부모 개념 추출"""
        # 주요 원형
        primary_archetype = self.archetype_mapping.get(
            creative_brief.archetype,
            'nostalgic'
        )

        # 보조 원형 (감정 팔레트에서 추출)
        emotion_palette = creative_brief.emotional_palette
        sorted_emotions = sorted(emotion_palette.items(), key=lambda x: x[1], reverse=True)

        secondary_emotion = sorted_emotions[1][0] if len(sorted_emotions) > 1 else sorted_emotions[0][0]
        secondary_archetype = 'romantic'  # 기본값

        if 'warm' in secondary_emotion or 'soft' in secondary_emotion:
            secondary_archetype = 'nostalgic'
        elif 'sweet' in secondary_emotion or 'floral' in secondary_emotion:
            secondary_archetype = 'romantic'
        elif 'fresh' in secondary_emotion or 'energetic' in secondary_emotion:
            secondary_archetype = 'adventurous'

        return primary_archetype, secondary_archetype

    def generate_dna_id(self, lineage: List[str]) -> str:
        """고유한 DNA ID 생성"""
        # 타임스탬프 + 계보 해시
        timestamp = datetime.now().isoformat()
        lineage_str = '_'.join(lineage)
        hash_input = f"{timestamp}_{lineage_str}_{uuid.uuid4()}"
        dna_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"DNA_{dna_hash.upper()}"

    def calculate_phenotype_potential(self, genotype: Dict) -> Dict[str, float]:
        """표현형 잠재력 계산"""
        potential = {}

        # 향 지속성
        base_genes = genotype.get('base', [])
        potential['longevity'] = np.mean([g.molecular_weight / 400 for g in base_genes]) if base_genes else 0.5

        # 확산성
        top_genes = genotype.get('top', [])
        potential['sillage'] = np.mean([g.volatility for g in top_genes]) if top_genes else 0.5

        # 복잡성
        all_genes = []
        for note_type in genotype.values():
            all_genes.extend(note_type)
        potential['complexity'] = len(set([g.odor_family for g in all_genes])) / 10

        # 균형성
        note_counts = [len(genotype.get(nt, [])) for nt in ['top', 'middle', 'base']]
        potential['balance'] = 1 - (np.std(note_counts) / np.mean(note_counts) if np.mean(note_counts) > 0 else 0)

        return potential

    def create_dna_story(self, parent1: str, parent2: str, creative_brief: Any) -> str:
        """DNA 탄생 스토리 생성"""
        story_template = (
            f"{creative_brief.theme}의 영감으로부터 탄생한 이 DNA는 "
            f"{parent1}의 깊이와 {parent2}의 생동감이 만나 형성되었습니다. "
            f"{creative_brief.story} "
            f"이것은 단순한 향수가 아닌, 살아있는 기억의 결정체입니다."
        )
        return story_template

    def create(self, creative_brief: Any) -> OlfactoryDNA:
        """
        새로운 향수 DNA 생성
        CreativeBrief를 받아 유전적 재조합을 통해 고유한 DNA 창조
        """
        # 1. 부모 컨셉 분리
        parent1, parent2 = self.extract_parent_concepts(creative_brief)

        # 2. 부모 유전자 선택
        parent1_genes, parent2_genes = self.genetic_algorithm.select_parents(parent1, parent2)

        # 3. 유전자 재조합 (MOGA 또는 기본 Crossover)
        if self.genetic_algorithm.use_moga:
            # MOGA 최적화 사용
            objectives = {
                'balance': 1.0,  # 균형 중요도
                'longevity': 0.8,  # 지속성 중요도
                'complexity': 0.6,  # 복잡도 중요도
                'cost_efficiency': 0.4  # 비용 효율성
            }
            offspring_genotype = self.genetic_algorithm.optimize_with_moga(objectives)
        else:
            # 기본 교차 사용
            offspring_genotype = self.genetic_algorithm.crossover(parent1_genes, parent2_genes)

        # 4. 돌연변이 적용
        mutated_genotype = self.genetic_algorithm.mutate(offspring_genotype)

        # 5. DNA ID 생성
        lineage = [f"concept_{parent1}", f"concept_{parent2}"]
        dna_id = self.generate_dna_id(lineage)

        # 6. 표현형 잠재력 계산
        phenotype_potential = self.calculate_phenotype_potential(mutated_genotype)

        # 7. DNA 스토리 생성
        dna_story = self.create_dna_story(parent1, parent2, creative_brief)

        # 8. OlfactoryDNA 객체 생성
        olfactory_dna = OlfactoryDNA(
            dna_id=dna_id,
            lineage=lineage,
            genotype=mutated_genotype,
            phenotype_potential=phenotype_potential,
            story=dna_story,
            creation_timestamp=datetime.now().isoformat(),
            generation=1,
            fitness_score=np.mean(list(phenotype_potential.values()))
        )

        # 9. DNA 라이브러리에 저장
        self.dna_library[dna_id] = olfactory_dna

        return olfactory_dna

    def to_recipe(self, dna: OlfactoryDNA) -> Dict[str, List[str]]:
        """DNA를 실제 레시피로 변환"""
        recipe = {'top': [], 'middle': [], 'base': []}

        for note_type, genes in dna.genotype.items():
            for gene in genes:
                if gene.expression_level > 0.3:  # 발현 임계값
                    ingredient_str = f"{gene.ingredient} ({gene.concentration:.1%})"
                    recipe[note_type].append(ingredient_str)

        return recipe

    def to_json(self, dna: OlfactoryDNA) -> str:
        """OlfactoryDNA를 JSON으로 변환"""
        # FragranceGene 객체를 딕셔너리로 변환
        genotype_dict = {}
        for note_type, genes in dna.genotype.items():
            genotype_dict[note_type] = [
                {
                    'ingredient': g.ingredient,
                    'concentration': g.concentration,
                    'expression_level': g.expression_level,
                    'odor_family': g.odor_family
                }
                for g in genes
            ]

        return json.dumps({
            'dna_id': dna.dna_id,
            'lineage': dna.lineage,
            'genotype': genotype_dict,
            'phenotype_potential': dna.phenotype_potential,
            'story': dna.story,
            'creation_timestamp': dna.creation_timestamp,
            'generation': dna.generation,
            'fitness_score': dna.fitness_score,
            'recipe': self.to_recipe(dna)
        }, ensure_ascii=False, indent=2)


# 싱글톤 인스턴스
_olfactory_recombinator_instance = None

def get_olfactory_recombinator() -> OlfactoryRecombinatorAI:
    """싱글톤 OlfactoryRecombinatorAI 인스턴스 반환"""
    global _olfactory_recombinator_instance
    if _olfactory_recombinator_instance is None:
        _olfactory_recombinator_instance = OlfactoryRecombinatorAI()
    return _olfactory_recombinator_instance