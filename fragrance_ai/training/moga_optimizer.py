"""
Multi-Objective Genetic Algorithm (MOGA) for Fragrance Optimization
실제 다중 목표 유전 알고리즘 구현
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
import random
from enum import Enum

class SelectionMethod(Enum):
    """선택 방법"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"

class CrossoverMethod(Enum):
    """교차 방법"""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    BLEND = "blend"

class MutationMethod(Enum):
    """돌연변이 방법"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"

@dataclass
class Individual:
    """개체 (향수 레시피)"""
    genes: np.ndarray  # 유전자 (재료 농도)
    objectives: Dict[str, float] = field(default_factory=dict)  # 목표 값들
    fitness: float = 0.0  # 적합도
    rank: int = 0  # Pareto 순위
    crowding_distance: float = 0.0  # 밀집도 거리
    id: str = field(default_factory=lambda: str(random.randint(0, 1000000)))  # 고유 ID

    def __eq__(self, other):
        """개체 비교 (ID 기반)"""
        if not isinstance(other, Individual):
            return False
        return self.id == other.id

    def __hash__(self):
        """해시 값 (ID 기반)"""
        return hash(self.id)

    def dominates(self, other: 'Individual') -> bool:
        """Pareto 지배 관계 확인"""
        at_least_one_better = False
        for key in self.objectives:
            if self.objectives[key] < other.objectives.get(key, float('-inf')):
                return False
            if self.objectives[key] > other.objectives.get(key, float('-inf')):
                at_least_one_better = True
        return at_least_one_better

class MOGAOptimizer:
    """다중 목표 유전 알고리즘 최적화기"""

    def __init__(
        self,
        gene_dim: int,
        gene_bounds: List[Tuple[float, float]],
        objective_functions: Dict[str, Callable],
        population_size: int = 100,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
        crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM,
        mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    ):
        """
        Args:
            gene_dim: 유전자 차원 (재료 개수)
            gene_bounds: 각 유전자의 범위 [(min, max), ...]
            objective_functions: 목표 함수들 {name: function}
            population_size: 개체군 크기
            max_generations: 최대 세대 수
            crossover_rate: 교차 확률
            mutation_rate: 돌연변이 확률
            elitism_rate: 엘리트 보존 비율
        """
        self.gene_dim = gene_dim
        self.gene_bounds = gene_bounds
        self.objective_functions = objective_functions
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method

        self.population: List[Individual] = []
        self.generation = 0
        self.best_individuals: List[Individual] = []
        self.pareto_front: List[Individual] = []

    def initialize_population(self) -> None:
        """초기 개체군 생성"""
        self.population = []
        for _ in range(self.population_size):
            genes = np.array([
                np.random.uniform(low, high)
                for low, high in self.gene_bounds
            ])
            individual = Individual(genes=genes)
            self.evaluate_objectives(individual)
            self.population.append(individual)

        self.update_fitness()

    def evaluate_objectives(self, individual: Individual) -> None:
        """목표 함수 평가"""
        for name, func in self.objective_functions.items():
            individual.objectives[name] = func(individual.genes)

    def update_fitness(self) -> None:
        """적합도 업데이트 (NSGA-II 스타일)"""
        # 1. Non-dominated sorting (Pareto 순위 계산)
        self.non_dominated_sorting()

        # 2. Crowding distance 계산
        self.calculate_crowding_distance()

        # 3. 적합도 = 1 / (rank + 1) + crowding_distance
        for individual in self.population:
            individual.fitness = 1.0 / (individual.rank + 1) + individual.crowding_distance * 0.1

    def non_dominated_sorting(self) -> None:
        """Non-dominated sorting (Pareto 순위 할당)"""
        remaining = self.population.copy()
        rank = 0

        while remaining:
            non_dominated = []
            for i, ind1 in enumerate(remaining):
                dominated = False
                for j, ind2 in enumerate(remaining):
                    if i != j and ind2.dominates(ind1):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(ind1)

            for ind in non_dominated:
                ind.rank = rank
                remaining.remove(ind)

            rank += 1

            # Pareto front 업데이트
            if rank == 1:
                self.pareto_front = non_dominated.copy()

    def calculate_crowding_distance(self) -> None:
        """밀집도 거리 계산"""
        for rank in range(max(ind.rank for ind in self.population) + 1):
            # 같은 순위의 개체들
            same_rank = [ind for ind in self.population if ind.rank == rank]

            if len(same_rank) <= 2:
                for ind in same_rank:
                    ind.crowding_distance = float('inf')
                continue

            # 각 목표별로 정렬하고 거리 계산
            for obj_name in self.objective_functions.keys():
                same_rank.sort(key=lambda x: x.objectives.get(obj_name, 0.0))

                # 경계 개체는 무한대 거리
                same_rank[0].crowding_distance = float('inf')
                same_rank[-1].crowding_distance = float('inf')

                # 중간 개체들의 거리 계산
                obj_range = (same_rank[-1].objectives.get(obj_name, 0.0) -
                           same_rank[0].objectives.get(obj_name, 0.0))

                if obj_range > 0:
                    for i in range(1, len(same_rank) - 1):
                        distance = (same_rank[i+1].objectives.get(obj_name, 0.0) -
                                  same_rank[i-1].objectives.get(obj_name, 0.0)) / obj_range
                        same_rank[i].crowding_distance += distance

    def select_parents(self) -> Tuple[Individual, Individual]:
        """부모 선택"""
        if self.selection_method == SelectionMethod.TOURNAMENT:
            return self.tournament_selection(), self.tournament_selection()
        elif self.selection_method == SelectionMethod.ROULETTE:
            return self.roulette_selection(), self.roulette_selection()
        else:  # RANK
            return self.rank_selection(), self.rank_selection()

    def tournament_selection(self, tournament_size: int = 3) -> Individual:
        """토너먼트 선택"""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def roulette_selection(self) -> Individual:
        """룰렛 휠 선택"""
        total_fitness = sum(ind.fitness for ind in self.population)
        pick = random.uniform(0, total_fitness)
        current = 0
        for ind in self.population:
            current += ind.fitness
            if current > pick:
                return ind
        return self.population[-1]

    def rank_selection(self) -> Individual:
        """순위 기반 선택"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        weights = [i for i in range(len(sorted_pop), 0, -1)]
        return random.choices(sorted_pop, weights=weights)[0]

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """교차 연산"""
        if random.random() > self.crossover_rate:
            return Individual(genes=parent1.genes.copy()), Individual(genes=parent2.genes.copy())

        if self.crossover_method == CrossoverMethod.UNIFORM:
            return self.uniform_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.SINGLE_POINT:
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_method == CrossoverMethod.TWO_POINT:
            return self.two_point_crossover(parent1, parent2)
        else:  # BLEND
            return self.blend_crossover(parent1, parent2)

    def uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """균일 교차"""
        mask = np.random.randint(0, 2, size=self.gene_dim, dtype=bool)
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """단일점 교차"""
        point = random.randint(1, self.gene_dim - 1)
        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """이점 교차"""
        point1 = random.randint(0, self.gene_dim - 2)
        point2 = random.randint(point1 + 1, self.gene_dim - 1)
        child1_genes = parent1.genes.copy()
        child2_genes = parent2.genes.copy()
        child1_genes[point1:point2] = parent2.genes[point1:point2]
        child2_genes[point1:point2] = parent1.genes[point1:point2]
        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def blend_crossover(self, parent1: Individual, parent2: Individual, alpha: float = 0.5) -> Tuple[Individual, Individual]:
        """블렌드 교차"""
        beta = np.random.uniform(-alpha, 1 + alpha, size=self.gene_dim)
        child1_genes = beta * parent1.genes + (1 - beta) * parent2.genes
        child2_genes = (1 - beta) * parent1.genes + beta * parent2.genes

        # 경계값 클리핑
        for i in range(self.gene_dim):
            child1_genes[i] = np.clip(child1_genes[i], self.gene_bounds[i][0], self.gene_bounds[i][1])
            child2_genes[i] = np.clip(child2_genes[i], self.gene_bounds[i][0], self.gene_bounds[i][1])

        return Individual(genes=child1_genes), Individual(genes=child2_genes)

    def mutate(self, individual: Individual) -> Individual:
        """돌연변이 연산"""
        if random.random() > self.mutation_rate:
            return individual

        if self.mutation_method == MutationMethod.GAUSSIAN:
            return self.gaussian_mutation(individual)
        elif self.mutation_method == MutationMethod.UNIFORM:
            return self.uniform_mutation(individual)
        else:  # ADAPTIVE
            return self.adaptive_mutation(individual)

    def gaussian_mutation(self, individual: Individual, sigma: float = 0.1) -> Individual:
        """가우시안 돌연변이"""
        mutated_genes = individual.genes.copy()
        for i in range(self.gene_dim):
            if random.random() < self.mutation_rate:
                mutated_genes[i] += np.random.normal(0, sigma * (self.gene_bounds[i][1] - self.gene_bounds[i][0]))
                mutated_genes[i] = np.clip(mutated_genes[i], self.gene_bounds[i][0], self.gene_bounds[i][1])
        return Individual(genes=mutated_genes)

    def uniform_mutation(self, individual: Individual) -> Individual:
        """균일 돌연변이"""
        mutated_genes = individual.genes.copy()
        for i in range(self.gene_dim):
            if random.random() < self.mutation_rate:
                mutated_genes[i] = np.random.uniform(self.gene_bounds[i][0], self.gene_bounds[i][1])
        return Individual(genes=mutated_genes)

    def adaptive_mutation(self, individual: Individual) -> Individual:
        """적응형 돌연변이 (세대에 따라 강도 감소)"""
        progress = self.generation / self.max_generations
        sigma = 0.3 * (1 - progress)  # 진행될수록 작아지는 돌연변이
        return self.gaussian_mutation(individual, sigma)

    def evolve_generation(self) -> None:
        """한 세대 진화"""
        new_population = []

        # 엘리트 보존
        elite_size = int(self.population_size * self.elitism_rate)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_size]
        new_population.extend([Individual(genes=ind.genes.copy()) for ind in elite])

        # 새로운 개체 생성
        while len(new_population) < self.population_size:
            # 부모 선택
            parent1, parent2 = self.select_parents()

            # 교차
            child1, child2 = self.crossover(parent1, parent2)

            # 돌연변이
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # 목표 평가
            self.evaluate_objectives(child1)
            self.evaluate_objectives(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population[:self.population_size]
        self.update_fitness()
        self.generation += 1

        # 최고 개체 추적
        best = max(self.population, key=lambda x: x.fitness)
        self.best_individuals.append(best)

    def optimize(self, verbose: bool = True) -> List[Individual]:
        """최적화 실행"""
        self.initialize_population()

        if verbose:
            print(f"Starting MOGA optimization with {self.population_size} individuals")
            print(f"Objectives: {list(self.objective_functions.keys())}")

        for gen in range(self.max_generations):
            self.evolve_generation()

            if verbose and gen % 10 == 0:
                best = self.best_individuals[-1]
                print(f"Generation {gen}: Best fitness = {best.fitness:.4f}")
                print(f"  Objectives: {best.objectives}")
                print(f"  Pareto front size: {len(self.pareto_front)}")

        if verbose:
            print(f"\nOptimization complete!")
            print(f"Final Pareto front contains {len(self.pareto_front)} solutions")

        return self.pareto_front

    def get_best_solution(self, objective_weights: Dict[str, float] = None) -> Individual:
        """가중치 기반 최적 해 선택"""
        if not objective_weights:
            # 동일 가중치
            objective_weights = {name: 1.0 for name in self.objective_functions.keys()}

        # 정규화된 가중 합으로 최적 해 선택
        best_individual = None
        best_score = float('-inf')

        for ind in self.pareto_front:
            score = sum(
                objective_weights.get(name, 1.0) * value
                for name, value in ind.objectives.items()
            )
            if score > best_score:
                best_score = score
                best_individual = ind

        return best_individual


# 향수 최적화를 위한 특화 함수들
def create_fragrance_optimizer(
    num_ingredients: int = 20,
    concentration_bounds: List[Tuple[float, float]] = None
) -> MOGAOptimizer:
    """향수 최적화기 생성"""

    if concentration_bounds is None:
        # 기본 농도 범위 (0.01% ~ 30%)
        concentration_bounds = [(0.01, 30.0)] * num_ingredients

    # 목표 함수 정의
    def balance_score(genes: np.ndarray) -> float:
        """균형 점수 (높을수록 좋음)"""
        top_notes = genes[:num_ingredients//3]
        heart_notes = genes[num_ingredients//3:2*num_ingredients//3]
        base_notes = genes[2*num_ingredients//3:]

        # 이상적 비율: 탑 30%, 하트 50%, 베이스 20%
        top_ratio = np.sum(top_notes) / np.sum(genes)
        heart_ratio = np.sum(heart_notes) / np.sum(genes)
        base_ratio = np.sum(base_notes) / np.sum(genes)

        ideal_distance = abs(top_ratio - 0.3) + abs(heart_ratio - 0.5) + abs(base_ratio - 0.2)
        return 1.0 / (1.0 + ideal_distance)

    def longevity_score(genes: np.ndarray) -> float:
        """지속성 점수 (베이스 노트 비중)"""
        base_notes = genes[2*num_ingredients//3:]
        return np.sum(base_notes) / 100.0

    def complexity_score(genes: np.ndarray) -> float:
        """복잡도 점수 (활성 재료 수)"""
        active_ingredients = np.sum(genes > 0.1)
        return active_ingredients / num_ingredients

    def cost_efficiency(genes: np.ndarray) -> float:
        """비용 효율성 (낮은 농도로 효과)"""
        total_concentration = np.sum(genes)
        return 1.0 / (1.0 + total_concentration / 100.0)

    objective_functions = {
        "balance": balance_score,
        "longevity": longevity_score,
        "complexity": complexity_score,
        "cost_efficiency": cost_efficiency
    }

    return MOGAOptimizer(
        gene_dim=num_ingredients,
        gene_bounds=concentration_bounds,
        objective_functions=objective_functions,
        population_size=100,
        max_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.15,
        elitism_rate=0.1
    )