"""
OlfactoryRecombinatorAI - DEAP를 사용한 다중 목표 유전 알고리즘
창세기 엔진: 후각적 DNA 생성 및 최적화
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random
import json
import os
import sys

# DEAP imports
from deap import base, creator, tools, algorithms
import array

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.fragrance_chemistry import FragranceChemistry, FRAGRANCE_DATABASE

# 기존 향수 데이터베이스 로드
def load_fragrance_database():
    """기존 향수 데이터베이스 로드"""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'fragrance_recipes_database.json'
    )

    if os.path.exists(db_path):
        with open(db_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

EXISTING_FRAGRANCES = load_fragrance_database()

@dataclass
class CreativeBrief:
    """창작 브리프 - 사용자 요구사항"""
    emotional_profile: Dict[str, float]  # 감정 프로필
    fragrance_family: str  # 향수 계열
    season: str  # 계절
    occasion: str  # 상황
    intensity: float  # 강도
    keywords: List[str]  # 키워드
    avoid_notes: List[str] = None  # 피해야 할 노트

    def __post_init__(self):
        if self.avoid_notes is None:
            self.avoid_notes = []


class OlfactoryRecombinatorAI:
    """창세기 엔진 - DEAP 기반 MOGA 최적화"""

    def __init__(self):
        self.chemistry = FragranceChemistry()
        self.all_ingredients = list(FRAGRANCE_DATABASE.keys())[:30]  # 상위 30개 재료
        self.ingredient_to_idx = {ing: i for i, ing in enumerate(self.all_ingredients)}

        # DEAP 설정
        self._setup_deap()

    def _setup_deap(self):
        """DEAP 환경 설정"""
        # 피트니스 클래스 생성 (다중 목표, 최대화)
        # weights=(1.0, 1.0, 1.0)으로 변경: harmony, balance, creativity 모두 최대화
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))

        # 개체 클래스 생성
        if not hasattr(creator, "Individual"):
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)

        # 툴박스 초기화
        self.toolbox = base.Toolbox()

        # 개체 생성 함수 - 0~1 사이의 실수값 (농도 비율)
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=len(self.all_ingredients))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 평가 함수 등록
        self.toolbox.register("evaluate", self._evaluate_individual)

        # 유전 연산자 등록 - 실수값에 적합한 연산자
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)  # Gaussian mutation
        self.toolbox.register("select", tools.selNSGA2)  # NSGA-II 선택

    def _individual_to_recipe(self, individual: List[float]) -> Dict[str, Dict[str, float]]:
        """개체를 향수 레시피로 변환"""
        # 농도 정규화 (0-50% 범위)
        concentrations = np.array(individual) * 50

        # 임계값 이상인 재료만 선택
        threshold = 1.0
        active_indices = np.where(concentrations > threshold)[0]

        recipe = {"top": {}, "middle": {}, "base": {}}

        for idx in active_indices:
            ingredient = self.all_ingredients[idx]
            concentration = concentrations[idx]

            # 노트 타입 결정
            note_info = FRAGRANCE_DATABASE.get(ingredient)
            if note_info:
                if note_info.category == 'top':
                    recipe["top"][ingredient] = concentration
                elif note_info.category == 'middle':
                    recipe["middle"][ingredient] = concentration
                else:
                    recipe["base"][ingredient] = concentration

        # 정규화 (합이 100%가 되도록)
        total = sum(sum(notes.values()) for notes in recipe.values())
        if total > 0:
            factor = 100.0 / total
            for note_type in recipe:
                recipe[note_type] = {k: v * factor for k, v in recipe[note_type].items()}

        return recipe

    def _calculate_harmony_score(self, recipe: Dict) -> float:
        """조화도 점수 계산 (높을수록 좋음)"""
        # 화학적 평가
        top_notes = [(k, v) for k, v in recipe["top"].items()]
        middle_notes = [(k, v) for k, v in recipe["middle"].items()]
        base_notes = [(k, v) for k, v in recipe["base"].items()]

        evaluation = self.chemistry.evaluate_fragrance_complete(
            top_notes, middle_notes, base_notes
        )

        # 조화도 = (harmony + balance) / 2
        harmony = (evaluation.get('harmony', 0) + evaluation.get('balance', 0)) / 2.0

        # 재료 수 최적화 보너스
        total_ingredients = len(recipe["top"]) + len(recipe["middle"]) + len(recipe["base"])
        if 5 <= total_ingredients <= 12:
            harmony += 0.1  # 적절한 재료 수 보너스
        elif total_ingredients < 3:
            harmony -= 0.3  # 너무 적은 재료 페널티
        elif total_ingredients > 15:
            harmony -= 0.2  # 너무 많은 재료 페널티

        return max(0.0, min(1.0, harmony))

    def _calculate_fitness_score(self, recipe: Dict, brief: CreativeBrief) -> float:
        """적합도 점수 계산 (높을수록 좋음)"""
        fitness = 1.0  # 기본 적합도

        # 1. 피해야 할 노트 체크
        all_notes = set()
        for notes in recipe.values():
            all_notes.update(notes.keys())

        for avoid_note in brief.avoid_notes:
            if avoid_note in all_notes:
                fitness -= 0.3

        # 2. 향수 계열 적합도
        family_match = 0
        if brief.fragrance_family == "floral":
            for note in ["rose", "jasmine", "violet", "ylang_ylang"]:
                if note in all_notes:
                    family_match += 0.25
        elif brief.fragrance_family == "woody":
            for note in ["sandalwood", "cedarwood", "vetiver", "patchouli"]:
                if note in all_notes:
                    family_match += 0.25
        elif brief.fragrance_family == "citrus":
            for note in ["bergamot", "lemon", "orange", "grapefruit"]:
                if note in all_notes:
                    family_match += 0.25
        elif brief.fragrance_family == "oriental":
            for note in ["vanilla", "amber", "musk", "benzoin"]:
                if note in all_notes:
                    family_match += 0.25

        fitness += min(family_match, 1.0) * 0.5  # 최대 0.5 보너스

        # 3. 강도 적합도
        total_concentration = sum(sum(notes.values()) for notes in recipe.values())
        intensity_match = 1.0 - abs(brief.intensity - (total_concentration / 100.0))
        fitness += intensity_match * 0.2

        # 4. 계절 적합도
        season_match = 0.5  # 기본값
        if brief.season == "summer":
            # 여름엔 가벼운 향이 좋음
            light_notes = ["bergamot", "lemon", "orange", "lavender"]
            heavy_notes = ["musk", "amber", "patchouli"]
            for note in light_notes:
                if note in all_notes:
                    season_match += 0.15
            for note in heavy_notes:
                if note in all_notes:
                    season_match -= 0.15
        elif brief.season == "winter":
            # 겨울엔 따뜻한 향이 좋음
            warm_notes = ["vanilla", "amber", "benzoin", "sandalwood"]
            cold_notes = ["eucalyptus", "peppermint"]
            for note in warm_notes:
                if note in all_notes:
                    season_match += 0.15
            for note in cold_notes:
                if note in all_notes:
                    season_match -= 0.15

        fitness += max(0.0, season_match) * 0.3

        return max(0.0, min(2.0, fitness))

    def _calculate_creativity_score(self, recipe: Dict) -> float:
        """창의성 점수 계산 (높을수록 창의적)"""
        if not EXISTING_FRAGRANCES:
            return 0.8  # 데이터베이스가 없으면 기본 창의성

        # 기존 향수들과의 유사도 계산
        max_similarity = 0

        recipe_set = set()
        for notes in recipe.values():
            recipe_set.update(notes.keys())

        for existing in EXISTING_FRAGRANCES.values():
            existing_set = set()
            if isinstance(existing, dict):
                for note_type in ["top", "middle", "base"]:
                    if note_type in existing:
                        existing_set.update(existing[note_type])

            # Jaccard 유사도
            if recipe_set or existing_set:
                intersection = len(recipe_set & existing_set)
                union = len(recipe_set | existing_set)
                if union > 0:
                    similarity = intersection / union
                    max_similarity = max(max_similarity, similarity)

        # 창의성 = 1 - 최대 유사도
        # 완전 다름 = 1.0, 완전 동일 = 0.0
        creativity = 1.0 - max_similarity

        # 재료 다양성 보너스
        unique_families = set()
        for ingredient in recipe_set:
            if ingredient in FRAGRANCE_DATABASE:
                unique_families.add(FRAGRANCE_DATABASE[ingredient].odor_family)

        diversity_bonus = len(unique_families) / 10.0  # 최대 0.1

        return min(1.0, creativity + diversity_bonus)

    def _evaluate_individual(self, individual: List[float], brief: Optional[CreativeBrief] = None) -> Tuple[float, float, float]:
        """개체 평가 - 다중 목표 (모두 최대화)"""
        recipe = self._individual_to_recipe(individual)

        # 빈 레시피 처리
        if not any(recipe.values()):
            return (0.0, 0.0, 0.0)

        # 기본 브리프 (평가용)
        if brief is None:
            brief = CreativeBrief(
                emotional_profile={"fresh": 0.5, "romantic": 0.5},
                fragrance_family="floral",
                season="spring",
                occasion="daily",
                intensity=0.5,
                keywords=["elegant", "subtle"],
                avoid_notes=[]
            )

        # 세 가지 목표 모두 높을수록 좋음
        harmony = self._calculate_harmony_score(recipe)  # 0~1, 높을수록 좋음
        fitness = self._calculate_fitness_score(recipe, brief)  # 0~2, 높을수록 좋음
        creativity = self._calculate_creativity_score(recipe)  # 0~1, 높을수록 좋음

        return (harmony, fitness, creativity)

    def generate_olfactory_dna(
        self,
        brief: CreativeBrief,
        population_size: int = 100,
        generations: int = 50,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """후각적 DNA 생성 - MOGA 최적화"""

        # 평가 함수 업데이트 (브리프 반영)
        self.toolbox.unregister("evaluate")
        self.toolbox.register("evaluate", lambda ind: self._evaluate_individual(ind, brief))

        # 초기 개체군 생성
        population = self.toolbox.population(n=population_size)

        # 통계 설정
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # 진화 알고리즘 실행
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "avg", "std", "min", "max"

        # 홀 오브 페임 (최고 개체 추적)
        hof = tools.ParetoFront()

        # 초기 평가
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        hof.update(population)

        if verbose:
            print("Starting MOGA optimization...")
            print(f"Population: {population_size}, Generations: {generations}")
            print(f"Objectives: Stability, Unfitness, Uncreativity (all minimizing)")

        # 진화 루프
        for gen in range(generations):
            # 선택
            offspring = self.toolbox.select(population, population_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # 교차
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:  # 교차 확률
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 돌연변이 - 개체 값을 0~1 범위로 클리핑
            for mutant in offspring:
                if random.random() < 0.2:  # 돌연변이 확률
                    self.toolbox.mutate(mutant)
                    # 값 범위 보정 (0~1)
                    for i in range(len(mutant)):
                        mutant[i] = max(0.0, min(1.0, mutant[i]))
                    del mutant.fitness.values

            # 새로운 개체 평가
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 엘리트 보존 전략
            population[:] = tools.selBest(population + offspring, population_size)

            # 홀 오브 페임 업데이트
            hof.update(population)

            # 통계 기록
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            if verbose and gen % 10 == 0:
                print(f"Generation {gen}: {record}")

        # 최적 개체 선택 (가중 합 기준)
        weights = {"harmony": 0.4, "fitness": 0.4, "creativity": 0.2}

        best_individual = None
        best_score = float('-inf')  # 최대화이므로 -inf로 초기화

        for ind in hof:
            harmony, fitness, creativity = ind.fitness.values
            score = (harmony * weights["harmony"] +
                    fitness * weights["fitness"] +
                    creativity * weights["creativity"])

            if score > best_score:  # 최대값 찾기
                best_score = score
                best_individual = ind

        # 최적 레시피 변환
        best_recipe = self._individual_to_recipe(best_individual)

        # 상세 평가
        top_notes = [(k, v) for k, v in best_recipe["top"].items()]
        middle_notes = [(k, v) for k, v in best_recipe["middle"].items()]
        base_notes = [(k, v) for k, v in best_recipe["base"].items()]

        evaluation = self.chemistry.evaluate_fragrance_complete(
            top_notes, middle_notes, base_notes
        )

        return {
            "olfactory_dna": best_individual,
            "recipe": best_recipe,
            "fitness_values": {
                "harmony": best_individual.fitness.values[0],
                "fitness": best_individual.fitness.values[1],
                "creativity": best_individual.fitness.values[2],
                "overall": best_score
            },
            "evaluation": evaluation,
            "pareto_front_size": len(hof),
            "generation_stats": logbook,
            "brief": {
                "fragrance_family": brief.fragrance_family,
                "season": brief.season,
                "intensity": brief.intensity,
                "keywords": brief.keywords
            }
        }

    def evolve_dna(
        self,
        parent_dna: List[float],
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2
    ) -> List[float]:
        """DNA 진화 - 돌연변이 적용"""
        child = parent_dna.copy()

        for i in range(len(child)):
            if random.random() < mutation_rate:
                # 가우시안 돌연변이
                child[i] += np.random.normal(0, mutation_strength)
                child[i] = np.clip(child[i], 0, 1)

        return child

    def crossover_dna(
        self,
        parent1_dna: List[float],
        parent2_dna: List[float],
        method: str = "uniform"
    ) -> Tuple[List[float], List[float]]:
        """DNA 교차 - 두 부모의 유전자 조합"""
        if method == "uniform":
            # 균일 교차
            child1, child2 = [], []
            for i in range(len(parent1_dna)):
                if random.random() < 0.5:
                    child1.append(parent1_dna[i])
                    child2.append(parent2_dna[i])
                else:
                    child1.append(parent2_dna[i])
                    child2.append(parent1_dna[i])

        elif method == "blend":
            # 블렌드 교차
            alpha = 0.5
            child1 = [alpha * p1 + (1 - alpha) * p2
                     for p1, p2 in zip(parent1_dna, parent2_dna)]
            child2 = [(1 - alpha) * p1 + alpha * p2
                     for p1, p2 in zip(parent1_dna, parent2_dna)]

        else:  # two_point
            # 2점 교차
            point1 = random.randint(0, len(parent1_dna) // 2)
            point2 = random.randint(len(parent1_dna) // 2, len(parent1_dna))

            child1 = (parent1_dna[:point1] +
                     parent2_dna[point1:point2] +
                     parent1_dna[point2:])
            child2 = (parent2_dna[:point1] +
                     parent1_dna[point1:point2] +
                     parent2_dna[point2:])

        return child1, child2


# 테스트 함수
if __name__ == "__main__":
    # 테스트 브리프 생성
    test_brief = CreativeBrief(
        emotional_profile={"romantic": 0.8, "mysterious": 0.6, "elegant": 0.7},
        fragrance_family="oriental",
        season="fall",
        occasion="evening",
        intensity=0.7,
        keywords=["seductive", "warm", "sophisticated"],
        avoid_notes=["mint", "eucalyptus"]
    )

    # 창세기 엔진 초기화
    recombinator = OlfactoryRecombinatorAI()

    # DNA 생성
    print("Generating Olfactory DNA...")
    result = recombinator.generate_olfactory_dna(
        test_brief,
        population_size=50,
        generations=30,
        verbose=True
    )

    print("\n=== Best Recipe Found ===")
    print(f"Top Notes: {result['recipe']['top']}")
    print(f"Middle Notes: {result['recipe']['middle']}")
    print(f"Base Notes: {result['recipe']['base']}")
    print(f"\nFitness: {result['fitness_values']}")
    print(f"Evaluation: {result['evaluation']}")
    print(f"Pareto Front Size: {result['pareto_front_size']}")