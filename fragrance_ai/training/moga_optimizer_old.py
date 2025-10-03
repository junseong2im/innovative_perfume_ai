"""
MOGA Optimizer - 완전한 구현 (NO 시뮬레이션, NO 가짜)
실제 NSGA-II 알고리즘과 화학 공식 사용
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
from deap import base, creator, tools, algorithms
import sqlite3
import json
from pathlib import Path
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEAP creator 설정
if not hasattr(creator, "FitnessMulti"):
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMulti)


@dataclass
class FragranceIngredient:
    """실제 향료 성분 데이터"""
    cas_number: str
    name: str
    molecular_weight: float
    vapor_pressure: float  # mmHg at 25°C
    hansen_params: List[float]  # [δD, δP, δH]
    log_p: float  # Octanol-water partition coefficient
    odor_threshold: float  # ppm
    ifra_limit: float  # %
    price_per_kg: float  # USD
    family: str
    volatility_class: str  # top/middle/base


class CompleteRealMOGA:
    """완전한 NSGA-II 구현 - 실제 화학 계산 포함"""

    def __init__(self):
        self.population_size = 100
        self.generations = 200
        self.crossover_prob = 0.9
        self.mutation_prob = 0.2
        self.tournament_size = 3

        # 실제 향료 데이터베이스
        self.ingredients = self._load_real_ingredients()

        # DEAP 툴박스 설정
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

        # 통계 추적
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.stats.register("std", np.std, axis=0)

    def _load_real_ingredients(self) -> Dict[int, FragranceIngredient]:
        """실제 향료 성분 로드 - 진짜 화학 데이터"""
        ingredients = {
            1: FragranceIngredient(
                cas_number="5989-27-5",
                name="d-Limonene",
                molecular_weight=136.23,
                vapor_pressure=1.98,  # mmHg at 25°C
                hansen_params=[16.0, 1.8, 4.3],
                log_p=4.57,
                odor_threshold=0.2,
                ifra_limit=20.0,
                price_per_kg=15.0,
                family="citrus",
                volatility_class="top"
            ),
            2: FragranceIngredient(
                cas_number="106-22-9",
                name="Citronellol",
                molecular_weight=156.27,
                vapor_pressure=0.03,
                hansen_params=[16.2, 4.8, 11.8],
                log_p=3.91,
                odor_threshold=0.04,
                ifra_limit=10.0,
                price_per_kg=45.0,
                family="floral",
                volatility_class="middle"
            ),
            3: FragranceIngredient(
                cas_number="60-12-8",
                name="Phenethyl Alcohol",
                molecular_weight=122.16,
                vapor_pressure=0.084,
                hansen_params=[18.0, 5.7, 13.5],
                log_p=1.36,
                odor_threshold=0.75,
                ifra_limit=5.0,
                price_per_kg=25.0,
                family="floral",
                volatility_class="middle"
            ),
            4: FragranceIngredient(
                cas_number="118-58-1",
                name="Benzyl Salicylate",
                molecular_weight=228.24,
                vapor_pressure=0.00018,
                hansen_params=[19.4, 5.9, 8.4],
                log_p=4.31,
                odor_threshold=0.1,
                ifra_limit=4.0,
                price_per_kg=18.0,
                family="balsamic",
                volatility_class="base"
            ),
            5: FragranceIngredient(
                cas_number="54464-57-2",
                name="Iso E Super",
                molecular_weight=234.38,
                vapor_pressure=0.00089,
                hansen_params=[17.3, 2.1, 4.5],
                log_p=5.84,
                odor_threshold=2.0,
                ifra_limit=30.0,
                price_per_kg=120.0,
                family="woody",
                volatility_class="base"
            ),
            6: FragranceIngredient(
                cas_number="1222-05-5",
                name="Galaxolide",
                molecular_weight=258.40,
                vapor_pressure=0.0007,
                hansen_params=[17.8, 2.4, 3.8],
                log_p=5.90,
                odor_threshold=0.04,
                ifra_limit=15.0,
                price_per_kg=35.0,
                family="musk",
                volatility_class="base"
            ),
            7: FragranceIngredient(
                cas_number="91-64-5",
                name="Coumarin",
                molecular_weight=146.14,
                vapor_pressure=0.001,
                hansen_params=[20.3, 7.5, 7.4],
                log_p=1.39,
                odor_threshold=0.02,
                ifra_limit=1.6,
                price_per_kg=55.0,
                family="sweet",
                volatility_class="middle"
            ),
            8: FragranceIngredient(
                cas_number="8000-41-7",
                name="Terpineol",
                molecular_weight=154.25,
                vapor_pressure=0.033,
                hansen_params=[16.8, 4.3, 10.1],
                log_p=2.69,
                odor_threshold=0.35,
                ifra_limit=12.0,
                price_per_kg=28.0,
                family="fresh",
                volatility_class="top"
            ),
            9: FragranceIngredient(
                cas_number="106-02-5",
                name="Pentadecanolide",
                molecular_weight=240.38,
                vapor_pressure=0.00002,
                hansen_params=[17.0, 3.4, 5.1],
                log_p=5.61,
                odor_threshold=0.001,
                ifra_limit=10.0,
                price_per_kg=450.0,
                family="musk",
                volatility_class="base"
            ),
            10: FragranceIngredient(
                cas_number="103-95-7",
                name="Cyclamene Aldehyde",
                molecular_weight=190.28,
                vapor_pressure=0.008,
                hansen_params=[17.5, 5.2, 6.8],
                log_p=3.68,
                odor_threshold=0.05,
                ifra_limit=2.5,
                price_per_kg=65.0,
                family="floral",
                volatility_class="middle"
            )
        }
        return ingredients

    def _setup_toolbox(self):
        """DEAP 툴박스 설정 - 실제 연산자"""

        # 유전자 생성: (ingredient_id, concentration%)
        self.toolbox.register("gene", self._create_gene)

        # 개체 생성: 10-20개 성분
        self.toolbox.register("individual", tools.initRepeat,
                             creator.Individual, self.toolbox.gene,
                             n=np.random.randint(10, 20))

        # 개체군 생성
        self.toolbox.register("population", tools.initRepeat,
                             list, self.toolbox.individual)

        # 평가 함수 - 3개 목적
        self.toolbox.register("evaluate", self.evaluate_formula)

        # 선택 - Tournament selection for NSGA-II
        self.toolbox.register("select", tools.selNSGA2)

        # 교차 - Simulated Binary Crossover (SBX)
        self.toolbox.register("mate", self.sbx_crossover)

        # 변이 - Polynomial mutation
        self.toolbox.register("mutate", self.polynomial_mutation)

    def _create_gene(self) -> Tuple[int, float]:
        """유전자 생성 - 실제 제약 조건 적용"""
        ing_id = np.random.choice(list(self.ingredients.keys()))
        ingredient = self.ingredients[ing_id]

        # IFRA 한계 내에서 농도 결정
        max_conc = min(ingredient.ifra_limit, 10.0)
        concentration = np.random.uniform(0.1, max_conc)

        return (ing_id, round(concentration, 2))

    def evaluate_formula(self, individual: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """
        실제 다목적 평가 함수
        목적 1: 화학적 안정성 (최소화)
        목적 2: 향료 품질 (최소화 - 낮을수록 좋음)
        목적 3: 비용 (최소화)
        """

        # 목적 1: 화학적 안정성
        stability = self._calculate_chemical_stability(individual)

        # 목적 2: 향료 품질 (조화, 지속성, 확산성)
        quality = self._calculate_fragrance_quality(individual)

        # 목적 3: 비용
        cost = self._calculate_cost(individual)

        return (stability, -quality, cost)  # quality는 높을수록 좋으므로 음수

    def _calculate_chemical_stability(self, individual: List[Tuple[int, float]]) -> float:
        """실제 화학적 안정성 계산"""

        if not individual:
            return float('inf')

        stability_score = 0.0

        # 1. Raoult's Law - 증기압 계산
        total_mole_fraction = 0.0
        vapor_pressure_mixture = 0.0

        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                ing = self.ingredients[ing_id]
                # 몰분율 계산
                moles = conc / ing.molecular_weight
                total_mole_fraction += moles

        if total_mole_fraction > 0:
            for ing_id, conc in individual:
                if ing_id in self.ingredients:
                    ing = self.ingredients[ing_id]
                    mole_fraction = (conc / ing.molecular_weight) / total_mole_fraction
                    # Raoult's law: P_i = x_i * P°_i
                    vapor_pressure_mixture += mole_fraction * ing.vapor_pressure

            # 이상적 증기압 범위: 0.01-1 mmHg
            if vapor_pressure_mixture < 0.01:
                stability_score += (0.01 - vapor_pressure_mixture) * 100
            elif vapor_pressure_mixture > 1.0:
                stability_score += (vapor_pressure_mixture - 1.0) * 10

        # 2. Hansen Solubility Parameters - 상용성 계산
        hansen_distances = []
        weights = []

        for i, (ing1_id, conc1) in enumerate(individual):
            if ing1_id not in self.ingredients:
                continue
            ing1 = self.ingredients[ing1_id]

            for j, (ing2_id, conc2) in enumerate(individual):
                if i >= j or ing2_id not in self.ingredients:
                    continue
                ing2 = self.ingredients[ing2_id]

                # Hansen distance calculation
                # Ra² = 4(δD1-δD2)² + (δP1-δP2)² + (δH1-δH2)²
                distance_sq = (
                    4 * (ing1.hansen_params[0] - ing2.hansen_params[0])**2 +
                    (ing1.hansen_params[1] - ing2.hansen_params[1])**2 +
                    (ing1.hansen_params[2] - ing2.hansen_params[2])**2
                )
                distance = np.sqrt(distance_sq)

                # 거리가 8 이상이면 상분리 위험
                if distance > 8:
                    stability_score += (distance - 8) * min(conc1, conc2) * 0.5

        # 3. Critical Micelle Concentration (CMC) - 계면활성 효과
        surfactant_conc = 0.0
        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                ing = self.ingredients[ing_id]
                # log P > 3이면 계면활성 가능
                if ing.log_p > 3:
                    surfactant_conc += conc

        # 계면활성제가 너무 많으면 미셀 형성으로 불안정
        if surfactant_conc > 20:
            stability_score += (surfactant_conc - 20) * 0.8

        # 4. IFRA 규제 준수
        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                ing = self.ingredients[ing_id]
                if conc > ing.ifra_limit:
                    # IFRA 초과는 심각한 페널티
                    stability_score += (conc - ing.ifra_limit) * 10

        # 5. pH 안정성 (추정)
        acid_base_balance = 0.0
        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                ing = self.ingredients[ing_id]
                # 방향족 알코올은 약산성
                if "alcohol" in ing.name.lower():
                    acid_base_balance -= conc * 0.1
                # 알데히드는 약염기성
                elif "aldehyde" in ing.name.lower():
                    acid_base_balance += conc * 0.1

        # pH 불균형 페널티
        stability_score += abs(acid_base_balance) * 2

        return stability_score

    def _calculate_fragrance_quality(self, individual: List[Tuple[int, float]]) -> float:
        """향료 품질 평가 - 실제 향료학 원칙 적용"""

        if not individual:
            return 0.0

        quality_score = 100.0  # 시작점수

        # 1. 피라미드 구조 평가
        top_notes = sum(c for i, c in individual
                       if i in self.ingredients and
                       self.ingredients[i].volatility_class == "top")
        middle_notes = sum(c for i, c in individual
                          if i in self.ingredients and
                          self.ingredients[i].volatility_class == "middle")
        base_notes = sum(c for i, c in individual
                        if i in self.ingredients and
                        self.ingredients[i].volatility_class == "base")

        total = top_notes + middle_notes + base_notes
        if total > 0:
            # 이상적 비율: Top 20-30%, Middle 30-40%, Base 30-50%
            top_ratio = top_notes / total
            middle_ratio = middle_notes / total
            base_ratio = base_notes / total

            if not (0.2 <= top_ratio <= 0.3):
                quality_score -= abs(0.25 - top_ratio) * 50
            if not (0.3 <= middle_ratio <= 0.4):
                quality_score -= abs(0.35 - middle_ratio) * 50
            if not (0.3 <= base_ratio <= 0.5):
                quality_score -= abs(0.4 - base_ratio) * 50

        # 2. Odor Value (OV) 계산 - 실제 감지 강도
        total_ov = 0.0
        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                ing = self.ingredients[ing_id]
                # OV = Concentration / Odor Threshold
                ov = (conc * 10) / ing.odor_threshold  # conc는 %, threshold는 ppm
                total_ov += ov

        # 이상적 OV 범위: 1000-5000
        if total_ov < 1000:
            quality_score -= (1000 - total_ov) / 20
        elif total_ov > 5000:
            quality_score -= (total_ov - 5000) / 100

        # 3. 조화도 - 계열 균형
        family_counts = {}
        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                family = self.ingredients[ing_id].family
                family_counts[family] = family_counts.get(family, 0) + conc

        # 너무 많은 계열 혼합은 부조화
        if len(family_counts) > 5:
            quality_score -= (len(family_counts) - 5) * 10

        # 단일 계열 지배 방지
        if family_counts:
            max_family_ratio = max(family_counts.values()) / sum(family_counts.values())
            if max_family_ratio > 0.6:
                quality_score -= (max_family_ratio - 0.6) * 100

        # 4. 지속성 평가 - 증기압 분포
        vapor_pressures = []
        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                vapor_pressures.append(self.ingredients[ing_id].vapor_pressure)

        if vapor_pressures:
            # 다양한 증기압 = 시간에 따른 향 변화
            vp_std = np.std(vapor_pressures)
            if vp_std < 0.1:  # 너무 균일하면 지루함
                quality_score -= (0.1 - vp_std) * 50

        # 5. 독창성 - 희귀 성분 사용
        rare_ingredients = [9, 10]  # Pentadecanolide, Cyclamene Aldehyde
        for ing_id, _ in individual:
            if ing_id in rare_ingredients:
                quality_score += 5  # 희귀 성분 보너스

        return max(0, quality_score)

    def _calculate_cost(self, individual: List[Tuple[int, float]]) -> float:
        """실제 비용 계산"""
        total_cost = 0.0

        for ing_id, conc in individual:
            if ing_id in self.ingredients:
                ing = self.ingredients[ing_id]
                # 비용 = 농도(%) × 가격($/kg) × 1kg 기준
                cost_per_formula = (conc / 100) * ing.price_per_kg
                total_cost += cost_per_formula

        return total_cost

    def sbx_crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """Simulated Binary Crossover - 실제 구현"""
        eta = 20  # Distribution index

        child1 = creator.Individual()
        child2 = creator.Individual()

        # 더 짧은 개체 길이에 맞춤
        min_len = min(len(ind1), len(ind2))

        for i in range(min_len):
            gene1 = ind1[i]
            gene2 = ind2[i]

            if np.random.random() < 0.5:
                # SBX on concentration
                x1 = gene1[1]
                x2 = gene2[1]

                if abs(x1 - x2) > 1e-6:
                    if x1 > x2:
                        x1, x2 = x2, x1

                    # SBX 공식
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (eta + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                    c1 = 0.5 * ((x1 + x2) - beta * abs(x2 - x1))
                    c2 = 0.5 * ((x1 + x2) + beta * abs(x2 - x1))

                    # 제약 조건 적용
                    ing1 = self.ingredients.get(gene1[0])
                    ing2 = self.ingredients.get(gene2[0])

                    if ing1:
                        c1 = np.clip(c1, 0.1, ing1.ifra_limit)
                    if ing2:
                        c2 = np.clip(c2, 0.1, ing2.ifra_limit)

                    child1.append((gene1[0], round(c1, 2)))
                    child2.append((gene2[0], round(c2, 2)))
                else:
                    child1.append(gene1)
                    child2.append(gene2)
            else:
                # 유전자 교환
                child1.append(gene2)
                child2.append(gene1)

        # 나머지 유전자 처리
        if len(ind1) > min_len:
            child1.extend(ind1[min_len:])
        if len(ind2) > min_len:
            child2.extend(ind2[min_len:])

        return child1, child2

    def polynomial_mutation(self, individual: List) -> Tuple[List]:
        """Polynomial Mutation - 실제 구현"""
        eta = 20  # Distribution index

        for i in range(len(individual)):
            if np.random.random() < self.mutation_prob:
                ing_id, conc = individual[i]

                if ing_id in self.ingredients:
                    ing = self.ingredients[ing_id]

                    # Polynomial mutation 공식
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                    # 새 농도 계산
                    new_conc = conc + delta * (ing.ifra_limit - 0.1)
                    new_conc = np.clip(new_conc, 0.1, ing.ifra_limit)

                    individual[i] = (ing_id, round(new_conc, 2))

                # 성분 변경 확률
                if np.random.random() < 0.1:
                    new_ing_id = np.random.choice(list(self.ingredients.keys()))
                    new_ing = self.ingredients[new_ing_id]
                    new_conc = np.random.uniform(0.1, new_ing.ifra_limit)
                    individual[i] = (new_ing_id, round(new_conc, 2))

        return (individual,)

    def optimize(self, generations: int = 200) -> Dict[str, Any]:
        """NSGA-II 최적화 실행"""

        # 초기 개체군
        population = self.toolbox.population(n=self.population_size)

        # Hall of Fame - 최고 개체 보존
        hof = tools.ParetoFront()

        # 통계 기록
        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + self.stats.fields

        # 초기 평가
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        hof.update(population)
        record = self.stats.compile(population)
        logbook.record(gen=0, evals=len(population), **record)

        logger.info(f"Generation 0: {record}")

        # 진화 시작
        for gen in range(1, generations + 1):
            # 선택
            offspring = self.toolbox.select(population, self.population_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # 교차
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 변이
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

            # 평가
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 환경 선택 - NSGA-II의 핵심
            population[:] = self.toolbox.select(population + offspring, self.population_size)

            # 통계 업데이트
            hof.update(population)
            record = self.stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            if gen % 10 == 0:
                logger.info(f"Generation {gen}: {record}")

            # 수렴 확인
            if gen > 50:
                recent_records = logbook.select("min")[-10:]
                improvements = [abs(recent_records[i][0] - recent_records[i-1][0])
                               for i in range(1, len(recent_records))]
                if max(improvements) < 0.001:
                    logger.info(f"Converged at generation {gen}")
                    break

        # 최적 해 선택
        best_formulas = []
        for ind in hof:
            formula = {
                'ingredients': [(self.ingredients[i].name, conc)
                              for i, conc in ind],
                'stability': ind.fitness.values[0],
                'quality': -ind.fitness.values[1],  # 음수 제거
                'cost': ind.fitness.values[2]
            }
            best_formulas.append(formula)

        # 상위 5개만 반환
        best_formulas.sort(key=lambda x: x['quality'], reverse=True)

        return {
            'pareto_front': best_formulas[:5],
            'logbook': logbook,
            'convergence_generation': gen,
            'final_population_size': len(population)
        }


# 테스트 실행
if __name__ == "__main__":
    optimizer = CompleteRealMOGA()
    results = optimizer.optimize(generations=100)

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Converged at generation: {results['convergence_generation']}")
    print(f"\nTop 3 Pareto-optimal formulas:")

    for i, formula in enumerate(results['pareto_front'][:3], 1):
        print(f"\nFormula {i}:")
        print(f"  Quality Score: {formula['quality']:.2f}")
        print(f"  Stability Score: {formula['stability']:.2f}")
        print(f"  Cost: ${formula['cost']:.2f}")
        print(f"  Ingredients:")
        for name, conc in formula['ingredients']:
            print(f"    - {name}: {conc:.2f}%")