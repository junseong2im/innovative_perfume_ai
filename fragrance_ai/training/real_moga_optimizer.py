"""
진짜 다중 목표 유전 알고리즘 (MOGA) - 향수 최적화 특화
Real Multi-Objective Genetic Algorithm for Fragrance Optimization
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any
import random
from dataclasses import dataclass, field
import uuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.fragrance_chemistry import FragranceChemistry, FRAGRANCE_DATABASE

@dataclass
class FragranceIndividual:
    """향수 개체 (유전자 + 목적 함수 값)"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # 유전자: {향료명: 농도} 형태
    top_genes: Dict[str, float] = field(default_factory=dict)  # Top notes
    middle_genes: Dict[str, float] = field(default_factory=dict)  # Middle notes
    base_genes: Dict[str, float] = field(default_factory=dict)  # Base notes

    # 목적 함수 값들
    objectives: Dict[str, float] = field(default_factory=dict)

    # Pareto 관련
    domination_count: int = 0  # 이 개체를 지배하는 개체 수
    dominated_solutions: List[str] = field(default_factory=list)  # 이 개체가 지배하는 개체들
    rank: int = 0  # Pareto rank
    crowding_distance: float = 0.0

    # 적합도
    fitness: float = 0.0

    def to_notes_list(self) -> Tuple[List, List, List]:
        """노트 리스트 형태로 변환"""
        top_notes = [(name, conc) for name, conc in self.top_genes.items()]
        middle_notes = [(name, conc) for name, conc in self.middle_genes.items()]
        base_notes = [(name, conc) for name, conc in self.base_genes.items()]
        return top_notes, middle_notes, base_notes

    def normalize_concentrations(self):
        """농도를 100%로 정규화"""
        total = sum(self.top_genes.values()) + sum(self.middle_genes.values()) + sum(self.base_genes.values())
        if total > 0:
            factor = 100.0 / total
            self.top_genes = {k: v * factor for k, v in self.top_genes.items()}
            self.middle_genes = {k: v * factor for k, v in self.middle_genes.items()}
            self.base_genes = {k: v * factor for k, v in self.base_genes.items()}

    def copy(self) -> 'FragranceIndividual':
        """개체 복사"""
        return FragranceIndividual(
            top_genes=self.top_genes.copy(),
            middle_genes=self.middle_genes.copy(),
            base_genes=self.base_genes.copy()
        )


class RealFragranceMOGA:
    """진짜 향수 최적화 MOGA"""

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 50,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        max_ingredients_per_note: int = 5
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_ingredients_per_note = max_ingredients_per_note

        self.generation = 0
        self.population: List[FragranceIndividual] = []
        self.pareto_front: List[FragranceIndividual] = []

        # 향료 분류
        self.top_ingredients = [name for name, note in FRAGRANCE_DATABASE.items() if note.category == 'top']
        self.middle_ingredients = [name for name, note in FRAGRANCE_DATABASE.items() if note.category == 'middle']
        self.base_ingredients = [name for name, note in FRAGRANCE_DATABASE.items() if note.category == 'base']

        # 화학 계산 모듈
        self.chemistry = FragranceChemistry()

    def initialize_population(self):
        """초기 개체군 생성 - 실제 향수 레시피"""
        self.population = []

        for _ in range(self.population_size):
            individual = self.create_random_fragrance()
            self.evaluate_objectives(individual)
            self.population.append(individual)

    def create_random_fragrance(self) -> FragranceIndividual:
        """랜덤 향수 생성"""
        individual = FragranceIndividual()

        # Top notes (2-4개)
        num_top = random.randint(2, min(4, len(self.top_ingredients)))
        selected_top = random.sample(self.top_ingredients, num_top)
        for ingredient in selected_top:
            individual.top_genes[ingredient] = random.uniform(5, 30)  # 5-30%

        # Middle notes (3-5개)
        num_middle = random.randint(3, min(5, len(self.middle_ingredients)))
        selected_middle = random.sample(self.middle_ingredients, num_middle)
        for ingredient in selected_middle:
            individual.middle_genes[ingredient] = random.uniform(10, 40)  # 10-40%

        # Base notes (2-4개)
        num_base = random.randint(2, min(4, len(self.base_ingredients)))
        selected_base = random.sample(self.base_ingredients, num_base)
        for ingredient in selected_base:
            individual.base_genes[ingredient] = random.uniform(10, 40)  # 10-40%

        # 정규화
        individual.normalize_concentrations()

        return individual

    def evaluate_objectives(self, individual: FragranceIndividual):
        """실제 향수 평가 함수들"""
        top_notes, middle_notes, base_notes = individual.to_notes_list()

        # FragranceChemistry를 사용한 실제 평가
        evaluation = self.chemistry.evaluate_fragrance_complete(
            top_notes, middle_notes, base_notes
        )

        # 목적 함수 설정 (다중 목표)
        individual.objectives = {
            'harmony': evaluation['harmony'],  # 최대화
            'longevity': evaluation['longevity'],  # 최대화
            'sillage': evaluation['sillage'],  # 최대화
            'balance': evaluation['balance'],  # 최대화
            'cost': -evaluation['cost'] / 1000,  # 최소화 (음수로 변환)
            'uniqueness': evaluation['uniqueness']  # 최대화
        }

        # 전체 적합도
        individual.fitness = evaluation['overall']

    def dominates(self, ind1: FragranceIndividual, ind2: FragranceIndividual) -> bool:
        """ind1이 ind2를 지배하는지 확인 (Pareto dominance)"""
        # objectives가 비어있으면 지배할 수 없음
        if not ind1.objectives or not ind2.objectives:
            return False

        better_in_at_least_one = False

        for obj_name in ind1.objectives:
            if obj_name not in ind2.objectives:
                continue

            val1 = ind1.objectives[obj_name]
            val2 = ind2.objectives[obj_name]

            if val1 < val2:  # 더 나쁜 경우
                return False
            elif val1 > val2:  # 더 좋은 경우
                better_in_at_least_one = True

        return better_in_at_least_one

    def non_dominated_sorting(self):
        """비지배 정렬 (Fast Non-dominated Sorting)"""
        # 초기화
        for ind in self.population:
            ind.domination_count = 0
            ind.dominated_solutions = []

        fronts = [[]]

        # 모든 개체 쌍에 대해 지배 관계 확인
        for i, p in enumerate(self.population):
            for j, q in enumerate(self.population):
                if i != j:
                    if self.dominates(p, q):
                        p.dominated_solutions.append(q.id)
                    elif self.dominates(q, p):
                        p.domination_count += 1

            # 첫 번째 프론트 (지배당하지 않는 개체들)
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        # 나머지 프론트 생성
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q_id in p.dominated_solutions:
                    q = next((ind for ind in self.population if ind.id == q_id), None)
                    if q:
                        q.domination_count -= 1
                        if q.domination_count == 0:
                            q.rank = i + 1
                            next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        self.pareto_front = fronts[0] if fronts else []

    def calculate_crowding_distance(self, front: List[FragranceIndividual]):
        """Crowding distance 계산 (다양성 유지)"""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return

        # 각 목적 함수에 대해
        for obj_name in front[0].objectives:
            # 목적 함수 값으로 정렬
            front.sort(key=lambda x: x.objectives[obj_name])

            # 경계 개체는 무한대
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # 범위 계산
            obj_range = front[-1].objectives[obj_name] - front[0].objectives[obj_name]
            if obj_range == 0:
                continue

            # 중간 개체들의 거리 계산
            for i in range(1, len(front) - 1):
                distance = (front[i + 1].objectives[obj_name] - front[i - 1].objectives[obj_name]) / obj_range
                front[i].crowding_distance += distance

    def crossover(self, parent1: FragranceIndividual, parent2: FragranceIndividual) -> Tuple[FragranceIndividual, FragranceIndividual]:
        """향수 특화 교차 연산"""
        child1 = FragranceIndividual()
        child2 = FragranceIndividual()

        if random.random() < self.crossover_rate:
            # BLX-α 교차 (Blend Crossover) - 농도에 적합
            alpha = 0.5

            # Top notes 교차
            all_top = set(parent1.top_genes.keys()) | set(parent2.top_genes.keys())
            for ingredient in all_top:
                conc1 = parent1.top_genes.get(ingredient, 0)
                conc2 = parent2.top_genes.get(ingredient, 0)

                if conc1 > 0 or conc2 > 0:
                    min_val = min(conc1, conc2)
                    max_val = max(conc1, conc2)
                    range_val = max_val - min_val

                    # BLX-α로 새로운 값 생성
                    new_val1 = random.uniform(
                        max(0, min_val - alpha * range_val),
                        min(100, max_val + alpha * range_val)
                    )
                    new_val2 = random.uniform(
                        max(0, min_val - alpha * range_val),
                        min(100, max_val + alpha * range_val)
                    )

                    if new_val1 > 1:  # 최소 농도
                        child1.top_genes[ingredient] = new_val1
                    if new_val2 > 1:
                        child2.top_genes[ingredient] = new_val2

            # Middle notes 교차 (동일한 방식)
            all_middle = set(parent1.middle_genes.keys()) | set(parent2.middle_genes.keys())
            for ingredient in all_middle:
                conc1 = parent1.middle_genes.get(ingredient, 0)
                conc2 = parent2.middle_genes.get(ingredient, 0)

                if conc1 > 0 or conc2 > 0:
                    min_val = min(conc1, conc2)
                    max_val = max(conc1, conc2)
                    range_val = max_val - min_val

                    new_val1 = random.uniform(
                        max(0, min_val - alpha * range_val),
                        min(100, max_val + alpha * range_val)
                    )
                    new_val2 = random.uniform(
                        max(0, min_val - alpha * range_val),
                        min(100, max_val + alpha * range_val)
                    )

                    if new_val1 > 1:
                        child1.middle_genes[ingredient] = new_val1
                    if new_val2 > 1:
                        child2.middle_genes[ingredient] = new_val2

            # Base notes 교차
            all_base = set(parent1.base_genes.keys()) | set(parent2.base_genes.keys())
            for ingredient in all_base:
                conc1 = parent1.base_genes.get(ingredient, 0)
                conc2 = parent2.base_genes.get(ingredient, 0)

                if conc1 > 0 or conc2 > 0:
                    min_val = min(conc1, conc2)
                    max_val = max(conc1, conc2)
                    range_val = max_val - min_val

                    new_val1 = random.uniform(
                        max(0, min_val - alpha * range_val),
                        min(100, max_val + alpha * range_val)
                    )
                    new_val2 = random.uniform(
                        max(0, min_val - alpha * range_val),
                        min(100, max_val + alpha * range_val)
                    )

                    if new_val1 > 1:
                        child1.base_genes[ingredient] = new_val1
                    if new_val2 > 1:
                        child2.base_genes[ingredient] = new_val2
        else:
            # 교차 없이 부모 복사
            child1 = parent1.copy()
            child2 = parent2.copy()

        # 정규화
        child1.normalize_concentrations()
        child2.normalize_concentrations()

        return child1, child2

    def mutate(self, individual: FragranceIndividual):
        """향수 특화 돌연변이"""
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'modify', 'substitute'])

            if mutation_type == 'add':
                # 새로운 향료 추가
                note_type = random.choice(['top', 'middle', 'base'])
                if note_type == 'top' and len(individual.top_genes) < self.max_ingredients_per_note:
                    available = [i for i in self.top_ingredients if i not in individual.top_genes]
                    if available:
                        new_ingredient = random.choice(available)
                        individual.top_genes[new_ingredient] = random.uniform(5, 20)

                elif note_type == 'middle' and len(individual.middle_genes) < self.max_ingredients_per_note:
                    available = [i for i in self.middle_ingredients if i not in individual.middle_genes]
                    if available:
                        new_ingredient = random.choice(available)
                        individual.middle_genes[new_ingredient] = random.uniform(10, 30)

                elif note_type == 'base' and len(individual.base_genes) < self.max_ingredients_per_note:
                    available = [i for i in self.base_ingredients if i not in individual.base_genes]
                    if available:
                        new_ingredient = random.choice(available)
                        individual.base_genes[new_ingredient] = random.uniform(10, 30)

            elif mutation_type == 'remove':
                # 향료 제거
                note_type = random.choice(['top', 'middle', 'base'])
                if note_type == 'top' and len(individual.top_genes) > 1:
                    remove_ingredient = random.choice(list(individual.top_genes.keys()))
                    del individual.top_genes[remove_ingredient]

                elif note_type == 'middle' and len(individual.middle_genes) > 1:
                    remove_ingredient = random.choice(list(individual.middle_genes.keys()))
                    del individual.middle_genes[remove_ingredient]

                elif note_type == 'base' and len(individual.base_genes) > 1:
                    remove_ingredient = random.choice(list(individual.base_genes.keys()))
                    del individual.base_genes[remove_ingredient]

            elif mutation_type == 'modify':
                # 농도 변경
                all_genes = list(individual.top_genes.items()) + list(individual.middle_genes.items()) + list(individual.base_genes.items())
                if all_genes:
                    ingredient, old_conc = random.choice(all_genes)
                    # Gaussian 돌연변이
                    new_conc = old_conc + np.random.normal(0, old_conc * 0.3)
                    new_conc = max(1, min(50, new_conc))  # 1-50% 범위

                    if ingredient in individual.top_genes:
                        individual.top_genes[ingredient] = new_conc
                    elif ingredient in individual.middle_genes:
                        individual.middle_genes[ingredient] = new_conc
                    elif ingredient in individual.base_genes:
                        individual.base_genes[ingredient] = new_conc

            elif mutation_type == 'substitute':
                # 향료 교체
                note_type = random.choice(['top', 'middle', 'base'])
                if note_type == 'top' and individual.top_genes:
                    old_ingredient = random.choice(list(individual.top_genes.keys()))
                    available = [i for i in self.top_ingredients if i not in individual.top_genes]
                    if available:
                        new_ingredient = random.choice(available)
                        individual.top_genes[new_ingredient] = individual.top_genes[old_ingredient]
                        del individual.top_genes[old_ingredient]

                elif note_type == 'middle' and individual.middle_genes:
                    old_ingredient = random.choice(list(individual.middle_genes.keys()))
                    available = [i for i in self.middle_ingredients if i not in individual.middle_genes]
                    if available:
                        new_ingredient = random.choice(available)
                        individual.middle_genes[new_ingredient] = individual.middle_genes[old_ingredient]
                        del individual.middle_genes[old_ingredient]

                elif note_type == 'base' and individual.base_genes:
                    old_ingredient = random.choice(list(individual.base_genes.keys()))
                    available = [i for i in self.base_ingredients if i not in individual.base_genes]
                    if available:
                        new_ingredient = random.choice(available)
                        individual.base_genes[new_ingredient] = individual.base_genes[old_ingredient]
                        del individual.base_genes[old_ingredient]

            # 정규화
            individual.normalize_concentrations()

    def tournament_selection(self, tournament_size: int = 3) -> FragranceIndividual:
        """토너먼트 선택"""
        tournament = random.sample(self.population, tournament_size)

        # Pareto rank가 낮을수록 좋음
        tournament.sort(key=lambda x: (x.rank, -x.crowding_distance))
        return tournament[0]

    def evolve_generation(self):
        """한 세대 진화"""
        # 모든 개체가 평가되었는지 확인
        for ind in self.population:
            if not ind.objectives:
                self.evaluate_objectives(ind)

        # 비지배 정렬
        self.non_dominated_sorting()

        # Crowding distance 계산
        for ind in self.population:
            ind.crowding_distance = 0

        fronts = {}
        for ind in self.population:
            if ind.rank not in fronts:
                fronts[ind.rank] = []
            fronts[ind.rank].append(ind)

        for front in fronts.values():
            self.calculate_crowding_distance(front)

        # 새로운 개체군 생성
        new_population = []

        # 엘리트 보존 (상위 10%)
        elite_size = int(self.population_size * 0.1)
        elite = sorted(self.population, key=lambda x: (x.rank, -x.crowding_distance))[:elite_size]
        new_population.extend([ind.copy() for ind in elite])

        # 나머지는 교차와 돌연변이로 생성
        while len(new_population) < self.population_size:
            # 선택
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # 교차
            child1, child2 = self.crossover(parent1, parent2)

            # 돌연변이
            self.mutate(child1)
            self.mutate(child2)

            # 평가
            self.evaluate_objectives(child1)
            self.evaluate_objectives(child2)

            # 추가
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population[:self.population_size]
        self.generation += 1

    def optimize(self, verbose: bool = True) -> List[FragranceIndividual]:
        """최적화 실행"""
        self.initialize_population()

        for gen in range(self.max_generations):
            self.evolve_generation()

            if verbose and gen % 10 == 0:
                best = max(self.population, key=lambda x: x.fitness)
                print(f"Generation {gen}: Best fitness = {best.fitness:.4f}, Pareto front size = {len(self.pareto_front)}")

        # 최종 비지배 정렬
        self.non_dominated_sorting()

        return self.pareto_front

    def get_best_solution(self, preferences: Optional[Dict[str, float]] = None) -> FragranceIndividual:
        """선호도에 따른 최적 해 선택"""
        if not self.pareto_front:
            return max(self.population, key=lambda x: x.fitness)

        if preferences:
            # 가중 합으로 최적 해 선택
            best_score = -float('inf')
            best_solution = None

            for ind in self.pareto_front:
                score = sum(ind.objectives.get(obj, 0) * weight
                          for obj, weight in preferences.items())
                if score > best_score:
                    best_score = score
                    best_solution = ind

            return best_solution
        else:
            # 기본: 전체 적합도가 가장 높은 해
            return max(self.pareto_front, key=lambda x: x.fitness)