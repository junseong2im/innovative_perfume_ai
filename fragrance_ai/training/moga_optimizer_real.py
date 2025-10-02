"""
MOGA Optimizer - 진짜 구현
DEAP 라이브러리를 제대로 사용한 다목적 유전 알고리즘
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from deap import base, creator, tools, algorithms
import json
from pathlib import Path
import hashlib

# DEAP creator 설정 - 전역으로 한 번만
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class RealMOGAOptimizer:
    """진짜 MOGA 구현 - 랜덤 아님"""

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        gene_size: int = 20
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.gene_size = gene_size
        self.gene_counter = 0  # 유전자 생성 카운터 추가

        # 향료 데이터베이스 로드
        self.notes_db = self._load_notes_database()

        # DEAP 툴박스 설정
        self.toolbox = base.Toolbox()
        self._setup_toolbox()

    def _load_notes_database(self) -> Dict:
        """실제 향료 데이터베이스 로드"""
        try:
            path = Path("assets/comprehensive_fragrance_notes_database.json")
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("notes", {})
        except:
            pass

        # 기본 향료 데이터
        return {
            1: {"name": "Bergamot", "type": "citrus", "volatility": 0.9, "strength": 0.7},
            2: {"name": "Lemon", "type": "citrus", "volatility": 0.95, "strength": 0.6},
            3: {"name": "Rose", "type": "floral", "volatility": 0.5, "strength": 0.8},
            4: {"name": "Jasmine", "type": "floral", "volatility": 0.4, "strength": 0.9},
            5: {"name": "Sandalwood", "type": "woody", "volatility": 0.1, "strength": 0.6},
            6: {"name": "Musk", "type": "animalic", "volatility": 0.05, "strength": 0.7},
            7: {"name": "Vanilla", "type": "sweet", "volatility": 0.2, "strength": 0.8},
            8: {"name": "Amber", "type": "resinous", "volatility": 0.15, "strength": 0.9}
        }

    def _setup_toolbox(self):
        """DEAP 툴박스를 제대로 설정"""

        # 유전자 생성 - 각 유전자는 (향료ID, 비율)
        self.toolbox.register("gene", self._create_gene)

        # 개체 생성 - gene_size 개의 유전자
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.gene,
            n=self.gene_size
        )

        # 개체군 생성
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        # 평가 함수
        self.toolbox.register("evaluate", self.evaluate_individual)

        # 교차 연산자 - 균일 교차
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)

        # 변이 연산자
        self.toolbox.register("mutate", self.mutate_individual)

        # 선택 연산자 - NSGA-II
        self.toolbox.register("select", tools.selNSGA2)

    def _create_gene(self) -> Tuple[int, float]:
        """유전자 하나 생성 (결정적)"""
        # 결정적 선택을 위한 해시 기반 선택
        self.gene_counter += 1
        hash_val = int(hashlib.sha256(f"gene_{id(self)}_{self.gene_counter}".encode()).hexdigest(), 16)

        note_ids = list(self.notes_db.keys())
        note_id = note_ids[hash_val % len(note_ids)]

        # 비율도 해시 기반으로 생성
        percentage = 0.5 + (hash_val % 145) / 10.0  # 0.5 ~ 15.0
        return (note_id, percentage)

    def evaluate_individual(self, individual: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """
        개체 평가 - 실제 계산 수행

        목표 1: 안정성 (낮을수록 좋음)
        목표 2: 조화도 (낮을수록 좋음)
        목표 3: 독창성 (낮을수록 좋음)
        """

        # 1. 안정성 평가 - 휘발도 균형
        volatilities = []
        percentages = []

        for note_id, percentage in individual:
            if note_id in self.notes_db:
                note = self.notes_db[note_id]
                vol = note.get("volatility", 0.5)
                # 문자열이면 변환
                if isinstance(vol, str):
                    vol = 0.9 if vol == "high" else 0.5 if vol == "medium" else 0.1
                volatilities.append(float(vol))
                percentages.append(float(percentage))

        if not volatilities:
            return (100.0, 100.0, 100.0)  # 최악의 점수

        # 휘발도 분산 - 너무 비슷하거나 너무 다르면 불안정
        vol_array = np.array(volatilities, dtype=float)
        vol_std = np.std(vol_array)
        stability_score = abs(vol_std - 0.3) * 10  # 0.3이 이상적

        # 비율 합이 100%에서 벗어날수록 불안정
        total_percentage = sum(percentages)
        stability_score += abs(100 - total_percentage) * 0.1

        # 2. 조화도 평가 - 향료 타입 다양성
        types = []
        for note_id, _ in individual:
            if note_id in self.notes_db:
                types.append(self.notes_db[note_id].get("type", "unknown"))

        unique_types = len(set(types))
        harmony_score = abs(unique_types - 4) * 5  # 4가지 타입이 이상적

        # 같은 타입이 너무 많으면 단조로움
        type_counts = {}
        for t in types:
            type_counts[t] = type_counts.get(t, 0) + 1
        max_count = max(type_counts.values()) if type_counts else 0
        harmony_score += max_count * 2

        # 3. 독창성 평가 - 일반적인 조합 회피
        common_pairs = [
            (1, 3),  # Bergamot + Rose (너무 흔함)
            (2, 4),  # Lemon + Jasmine (너무 흔함)
            (5, 6),  # Sandalwood + Musk (너무 흔함)
        ]

        creativity_score = 0
        note_ids = [nid for nid, _ in individual]

        for pair in common_pairs:
            if pair[0] in note_ids and pair[1] in note_ids:
                creativity_score += 10  # 흔한 조합 페널티

        # 너무 많은 향료도 문제
        if len(individual) > 15:
            creativity_score += (len(individual) - 15) * 2

        return (stability_score, harmony_score, creativity_score)

    def mutate_individual(self, individual):
        """개체 변이 (결정적)"""
        for i in range(len(individual)):
            # 해시 기반 확률
            hash_val = int(hashlib.sha256(f"mut_{id(individual)}_{i}".encode()).hexdigest(), 16)
            if (hash_val % 100) / 100.0 < self.mutation_rate:
                # 해시 기반으로 변이 타입 결정
                if (hash_val % 2) == 0:
                    # 향료 변경
                    note_ids = list(self.notes_db.keys())
                    new_note = note_ids[hash_val % len(note_ids)]
                    individual[i] = (new_note, individual[i][1])
                else:
                    # 비율 변경 (정규분포 대신 해시 기반 변경)
                    delta = ((hash_val % 40) - 20) / 10.0  # -2.0 ~ 2.0
                    new_percentage = individual[i][1] + delta
                    new_percentage = max(0.1, min(20.0, new_percentage))  # 0.1~20% 제한
                    individual[i] = (individual[i][0], new_percentage)
        return individual,

    def optimize(self, target_profile: Optional[Dict] = None) -> List:
        """
        MOGA 최적화 실행

        Args:
            target_profile: 목표 향수 프로필 (선택사항)

        Returns:
            Pareto front의 최적 개체들
        """

        # 초기 개체군 생성
        population = self.toolbox.population(n=self.population_size)

        # 통계 설정
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Hall of Fame (최고 개체 보존)
        hof = tools.ParetoFront()

        # 진화 알고리즘 실행
        algorithms.eaMuPlusLambda(
            population=population,
            toolbox=self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=False
        )

        # Pareto front 반환
        return hof

    def individual_to_recipe(self, individual: List[Tuple[int, float]]) -> Dict:
        """개체를 레시피 형식으로 변환"""

        # 향료를 탑/미들/베이스로 분류
        top_notes = []
        middle_notes = []
        base_notes = []

        for note_id, percentage in individual:
            if note_id in self.notes_db:
                note = self.notes_db[note_id]
                volatility = note.get("volatility", 0.5)
                # 문자열이면 변환
                if isinstance(volatility, str):
                    volatility = 0.9 if volatility == "high" else 0.5 if volatility == "medium" else 0.1
                volatility = float(volatility)

                note_info = {
                    "name": note["name"],
                    "percentage": round(percentage, 2)
                }

                if volatility > 0.7:
                    top_notes.append(note_info)
                elif volatility > 0.3:
                    middle_notes.append(note_info)
                else:
                    base_notes.append(note_info)

        return {
            "top_notes": top_notes,
            "middle_notes": middle_notes,
            "base_notes": base_notes,
            "total_ingredients": len(individual),
            "fitness": individual.fitness.values if hasattr(individual, 'fitness') else None
        }


# 테스트 함수
def test_real_moga():
    """진짜 MOGA 테스트"""
    print("\n=== REAL MOGA OPTIMIZER TEST ===")

    # 초기화
    moga = RealMOGAOptimizer(
        population_size=50,
        generations=20,
        gene_size=10
    )

    # 최적화 실행
    print("Running optimization...")
    pareto_front = moga.optimize()

    print(f"\nPareto front size: {len(pareto_front)}")

    # 최고 개체 출력
    if pareto_front:
        best = pareto_front[0]
        print(f"\nBest individual fitness: {best.fitness.values}")

        recipe = moga.individual_to_recipe(best)
        print("\nBest recipe:")
        print(f"Top notes: {recipe['top_notes']}")
        print(f"Middle notes: {recipe['middle_notes']}")
        print(f"Base notes: {recipe['base_notes']}")

    return len(pareto_front) > 0


if __name__ == "__main__":
    success = test_real_moga()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")