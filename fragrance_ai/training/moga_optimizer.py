"""
'창세기' 엔진: OlfactoryRecombinatorAI
DEAP 라이브러리를 사용한 다중 목표 유전 알고리즘(MOGA) 구현
목표: 창의성, 적합성, 안정성을 동시에 만족시키는 최적의 '후각적 DNA' 생성
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import json
import os

# DEAP 라이브러리 import
from deap import base, creator, tools, algorithms
from deap.tools import HallOfFame, ParetoFront
import array

# 과학적 검증을 위한 imports
from scipy.spatial.distance import euclidean
from scipy.stats import entropy
import logging

# 프로젝트 내부 모듈 imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# ValidatorTool import
try:
    from fragrance_ai.tools.validator_tool import ValidatorTool
except:
    ValidatorTool = None

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CreativeBrief:
    """사용자의 창의적 요구사항"""
    emotional_palette: List[float]  # 감정 벡터
    fragrance_family: str
    mood: str
    intensity: float
    season: str
    gender: str


@dataclass
class OlfactoryDNA:
    """향수 레시피의 유전자 표현"""
    genes: List[Tuple[int, float]]  # [(note_id, percentage), ...]
    fitness_scores: Tuple[float, float, float]  # (안정성, 부적합도, 비창의성)


class OlfactoryRecombinatorAI:
    """
    창세기 엔진: DEAP를 사용한 다중 목표 유전 알고리즘 구현
    """

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2):

        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.creative_brief = None

        # 향수 노트 데이터베이스 로드
        self.notes_db = self._load_notes_database()

        # 기존 향수 데이터베이스 로드
        self.existing_fragrances = self._load_existing_fragrances()

        # ValidatorTool 초기화
        self.validator = ValidatorTool() if ValidatorTool else None

        # DEAP 프레임워크 설정
        self._setup_deap_framework()

    def _load_notes_database(self) -> Dict:
        """향수 노트 데이터베이스 로드"""
        # 실제로는 knowledge_tool.py에서 로드해야 하지만, 여기서는 샘플 데이터 사용
        return {
            1: {"name": "Bergamot", "family": "citrus", "emotion_vector": [0.8, 0.2, 0.0], "volatility": 0.9},
            2: {"name": "Lemon", "family": "citrus", "emotion_vector": [0.9, 0.1, 0.0], "volatility": 0.95},
            3: {"name": "Rose", "family": "floral", "emotion_vector": [0.2, 0.7, 0.1], "volatility": 0.5},
            4: {"name": "Jasmine", "family": "floral", "emotion_vector": [0.1, 0.8, 0.1], "volatility": 0.45},
            5: {"name": "Sandalwood", "family": "woody", "emotion_vector": [0.0, 0.3, 0.7], "volatility": 0.15},
            6: {"name": "Cedar", "family": "woody", "emotion_vector": [0.0, 0.2, 0.8], "volatility": 0.18},
            7: {"name": "Vanilla", "family": "gourmand", "emotion_vector": [0.3, 0.6, 0.1], "volatility": 0.20},
            8: {"name": "Musk", "family": "animalic", "emotion_vector": [0.1, 0.4, 0.5], "volatility": 0.05},
            9: {"name": "Amber", "family": "oriental", "emotion_vector": [0.2, 0.5, 0.3], "volatility": 0.08},
            10: {"name": "Patchouli", "family": "woody", "emotion_vector": [0.0, 0.4, 0.6], "volatility": 0.10},
        }

    def _load_existing_fragrances(self) -> List[Dict]:
        """기존 향수 데이터베이스 로드 - fragrance_recipes_database.json"""
        db_path = Path(__file__).parent.parent.parent / "data" / "fragrance_recipes_database.json"

        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # JSON 구조에 따라 처리
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict) and 'fragrances' in data:
                        return data['fragrances']
                    else:
                        logger.warning("Unknown JSON structure, using sample data")
                        return self._get_sample_fragrances()
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"Error loading fragrance database: {e}. Using sample data")
                return self._get_sample_fragrances()
        else:
            return self._get_sample_fragrances()

    def _get_sample_fragrances(self) -> List[Dict]:
        """샘플 향수 데이터"""
        return [
            {"name": "Classic Citrus", "notes": [1, 2, 5], "percentages": [30, 20, 50]},
            {"name": "Romantic Rose", "notes": [3, 4, 7], "percentages": [40, 30, 30]},
            {"name": "Oriental Night", "notes": [4, 8, 9], "percentages": [25, 35, 40]},
        ]

    def _setup_deap_framework(self):
        """
        1단계: 개체(Individual) 및 개체군(Population) 정의
        DEAP 프레임워크 설정
        """

        # 기존 클래스가 있으면 삭제
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 적합도 클래스 정의 - 3가지 목표 모두 최소화 (낮을수록 좋음)
        # weights=(-1.0, -1.0, -1.0): 각각 안정성, 부적합도, 비창의성을 최소화
        # 음수 가중치는 DEAP에서 최소화를 의미
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))

        # 개체 정의 - 향수 레시피를 리스트로 표현
        # Individual은 15개의 (note_id, percentage) 튜플로 구성된 리스트
        # 각 개체는 하나의 완전한 향수 레시피를 나타냄
        # fitness 속성을 통해 3가지 목표에 대한 평가값을 저장
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # 툴박스 설정
        self.toolbox = base.Toolbox()

        # 유전자 생성 함수 - (note_id, percentage) 튜플 생성
        # 각 유전자는 하나의 향료 노트와 그 농도를 나타냄
        self.toolbox.register("gene", self._generate_gene)

        # 개체 생성 - 15개의 노트로 구성된 향수 레시피
        # n=15: 일반적인 향수는 10-20개 노트로 구성, 15개를 표준으로 설정
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.gene, n=15)

        # 개체군 생성
        # population_size 만큼의 개체를 생성하여 초기 개체군 구성
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 3단계: 유전 연산자(Genetic Operators) 등록

        # 평가 함수: 3가지 목표 (안정성, 부적합도, 비창의성)를 계산
        self.toolbox.register("evaluate", self.evaluate)

        # 교차 연산: 두 점 교차 (Two-Point Crossover)
        # 두 부모의 유전자를 두 지점에서 교환하여 자손 생성
        self.toolbox.register("mate", tools.cxTwoPoint)

        # 변이 연산: 균일 정수 변이 (Uniform Integer Mutation)
        # indpb=0.1: 각 유전자가 10% 확률로 변이
        # low=1, up=len(notes_db): 노트 ID 범위 내에서 변이
        self.toolbox.register("mutate", tools.mutUniformInt,
                            low=1, up=len(self.notes_db), indpb=0.1)

        # 선택 연산: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
        # 다목적 최적화를 위한 파레토 최적 선택 알고리즘
        self.toolbox.register("select", tools.selNSGA2)

    def _generate_gene(self) -> Tuple[int, float]:
        """
        유전자 생성 - CreativeBrief의 emotional_palette 기반
        초기 개체는 CreativeBrief의 emotional_palette를 기반으로 무작위 생성
        """
        # CreativeBrief가 있고 emotional_palette이 있으면 활용
        if self.creative_brief and self.creative_brief.emotional_palette:
            # 감정 벡터와 유사한 노트를 더 높은 확률로 선택
            weights = []
            for note_id, note_data in self.notes_db.items():
                # 감정 벡터 간 유사도 계산
                similarity = 1.0 - euclidean(
                    note_data["emotion_vector"],
                    self.creative_brief.emotional_palette[:3]  # 첫 3개 요소만 사용
                )
                weights.append(similarity)

            # 가중치에 따른 노트 선택
            note_ids = list(self.notes_db.keys())
            note_id = random.choices(note_ids, weights=weights)[0]
        else:
            # 랜덤하게 노트 선택
            note_id = random.randint(1, len(self.notes_db))

        # 농도는 0.1% ~ 10% 사이
        percentage = random.uniform(0.1, 10.0)
        return (note_id, percentage)

    def evaluate(self, individual: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """
        2단계: 적합도 평가 함수 (Fitness Evaluation Function)
        이것이 이 엔진의 심장입니다.

        함수 시그니처: def evaluate(individual: list) -> tuple:
        하나의 향수 레시피(individual)를 입력받아, 3가지 목표에 대한 점수를 튜플로 반환

        Returns:
            (안정성 점수, 부적합도 점수, 비창의성 점수) - 모두 낮을수록 좋음
        """

        # a. 안정성 점수 (Stability Score)
        # ValidatorTool을 호출하여 조향 규칙 위반의 개수를 계산
        stability_score = self._evaluate_stability(individual)

        # b. 부적합도 점수 (Unfitness Score)
        # CreativeBrief의 emotional_palette와 현재 레시피 간 유클리드 거리
        unfitness_score = self._evaluate_unfitness(individual)

        # c. 비창의성 점수 (Uncreativity Score)
        # 기존 향수 데이터베이스와의 Jaccard 유사도
        uncreativity_score = self._evaluate_uncreativity(individual)

        return (stability_score, unfitness_score, uncreativity_score)

    def _evaluate_stability(self, individual: List[Tuple[int, float]]) -> float:
        """
        a. 안정성 점수: ValidatorTool을 호출하여 조향 규칙 위반 개수 계산
        점수가 0에 가까울수록 좋습니다.
        """

        violations = 0.0

        # 총 농도 체크 (15-25% 사이여야 함)
        total_percentage = sum(p for _, p in individual)
        if not (15 <= total_percentage <= 25):
            violations += abs(20 - total_percentage) / 5.0

        # 노트 균형 체크
        top_notes = sum(p for n, p in individual
                       if n in self.notes_db and self.notes_db[n]["volatility"] > 0.7)
        middle_notes = sum(p for n, p in individual
                         if n in self.notes_db and 0.3 <= self.notes_db[n]["volatility"] <= 0.7)
        base_notes = sum(p for n, p in individual
                        if n in self.notes_db and self.notes_db[n]["volatility"] < 0.3)

        # 이상적 비율: 탑 20-30%, 미들 30-40%, 베이스 30-50%
        if total_percentage > 0:
            top_ratio = top_notes / total_percentage
            middle_ratio = middle_notes / total_percentage
            base_ratio = base_notes / total_percentage

            if not (0.2 <= top_ratio <= 0.3):
                violations += abs(0.25 - top_ratio) * 2
            if not (0.3 <= middle_ratio <= 0.4):
                violations += abs(0.35 - middle_ratio) * 2
            if not (0.3 <= base_ratio <= 0.5):
                violations += abs(0.4 - base_ratio) * 2

        # 중복 노트 체크 (같은 노트가 너무 많으면 위반)
        note_counts = {}
        for note_id, _ in individual:
            note_counts[note_id] = note_counts.get(note_id, 0) + 1

        for count in note_counts.values():
            if count > 2:  # 같은 노트가 3번 이상 나오면 위반
                violations += (count - 2) * 0.5

        return violations

    def _evaluate_unfitness(self, individual: List[Tuple[int, float]]) -> float:
        """
        b. 부적합도 점수: CreativeBrief의 emotional_palette 벡터와
        현재 레시피의 노트들이 가지는 감성 벡터 간의 유클리드 거리 계산
        거리가 0에 가까울수록 사용자의 요구와 일치합니다.
        """

        # 현재 레시피의 감성 벡터 계산
        recipe_emotion = [0.0, 0.0, 0.0]
        total_percentage = sum(p for _, p in individual)

        if total_percentage > 0:
            for note_id, percentage in individual:
                if note_id in self.notes_db:
                    note_emotion = self.notes_db[note_id]["emotion_vector"]
                    weight = percentage / total_percentage
                    for i in range(3):
                        recipe_emotion[i] += note_emotion[i] * weight

        # CreativeBrief의 emotional_palette 사용
        if self.creative_brief and self.creative_brief.emotional_palette:
            target_emotion = self.creative_brief.emotional_palette[:3]
        else:
            target_emotion = [0.3, 0.5, 0.2]  # 기본값

        # 유클리드 거리 계산
        distance = euclidean(recipe_emotion, target_emotion)

        return distance

    def _evaluate_uncreativity(self, individual: List[Tuple[int, float]]) -> float:
        """
        c. 비창의성 점수: 현재 레시피가 기존 향수 데이터베이스(fragrance_recipes_database.json)의
        레시피들과 얼마나 유사한지를 Jaccard 유사도 등으로 계산
        점수가 0에 가까울수록 기존에 없던 새로운 조합입니다.
        """

        # 현재 레시피의 노트 세트
        current_notes = set(note_id for note_id, _ in individual if note_id > 0)

        if not current_notes:
            return 1.0  # 빈 레시피는 창의성 없음

        # 기존 향수들과 비교
        max_similarity = 0.0

        for fragrance in self.existing_fragrances:
            existing_notes = set(fragrance.get("notes", []))

            # Jaccard 유사도 계산
            if existing_notes:
                intersection = len(current_notes & existing_notes)
                union = len(current_notes | existing_notes)
                similarity = intersection / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)

        # 가장 유사한 기존 향수와의 유사도 반환
        return max_similarity

    def _custom_mutation(self, individual: List[Tuple[int, float]]) -> Tuple[List]:
        """
        커스텀 변이 연산자 - 레시피의 특정 노트나 비율을 아주 낮은 확률로 변경
        """

        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                # 50% 확률로 노트 변경, 50% 확률로 비율 변경
                if random.random() < 0.5:
                    # 노트 변경
                    new_note_id = random.randint(1, len(self.notes_db))
                    individual[i] = (new_note_id, individual[i][1])
                else:
                    # 비율 변경
                    new_percentage = random.uniform(0.1, 10.0)
                    individual[i] = (individual[i][0], new_percentage)

        return (individual,)

    def evolve(self, creative_brief: Optional[CreativeBrief] = None) -> OlfactoryDNA:
        """
        4단계: 알고리즘 실행 루프 구현

        초기 개체군을 생성하고, 정해진 세대(Generation) 수만큼 진화 루프를 실행합니다.
        각 세대마다 toolbox.select, toolbox.mate, toolbox.mutate를 순차적으로 적용하여
        새로운 자손 개체군을 생성합니다.

        루프가 끝나면, tools.selBest를 사용하여 최종 세대의 개체군 중
        가장 뛰어난(파레토 최적해 집합에 속하는) 레시피 하나를
        OlfactoryDNA 객체로 변환하여 반환합니다.

        Args:
            creative_brief: 사용자의 창의적 요구사항

        Returns:
            최적의 향수 DNA
        """

        # CreativeBrief 저장
        self.creative_brief = creative_brief

        logger.info(f"[MOGA] Starting evolution: population={self.population_size}, generations={self.generations}")

        # 초기 개체군 생성
        population = self.toolbox.population(n=self.population_size)

        # Hall of Fame - 역대 최고의 개체들 보관
        hof = HallOfFame(1)

        # Pareto Front - 파레토 최적해 집합
        pareto = ParetoFront()

        # 통계 설정
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)

        # 진화 루프 실행
        for gen in range(self.generations):

            # 적합도 평가
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # 통계 기록
            record = stats.compile(population)

            # 진행 상황 로깅 (10세대마다)
            if gen % 10 == 0:
                logger.info(f"  Generation {gen}: stability={record['min'][0]:.3f}, "
                          f"unfitness={record['min'][1]:.3f}, uncreativity={record['min'][2]:.3f}")

            # 선택 (NSGA-II 알고리즘 사용)
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # 교차 (Crossover) - toolbox.mate 적용
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 변이 (Mutation) - toolbox.mutate 적용
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    # mutUniformInt는 정수만 변이시키므로, 튜플의 첫 번째 요소만 변이
                    for i in range(len(mutant)):
                        if random.random() < 0.1:  # 10% 확률로 각 유전자 변이
                            note_id = random.randint(1, len(self.notes_db))
                            mutant[i] = (note_id, mutant[i][1])
                        if random.random() < 0.1:  # 10% 확률로 비율 변이
                            percentage = random.uniform(0.1, 10.0)
                            mutant[i] = (mutant[i][0], percentage)
                    del mutant.fitness.values

            # 새로운 개체들의 적합도 평가
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 다음 세대 구성
            population[:] = offspring

            # Hall of Fame과 Pareto Front 업데이트
            hof.update(population)
            pareto.update(population)

        # 최종 최적해 선택 - tools.selBest 사용
        # 파레토 최적해 집합에서 가장 뛰어난 개체 선택
        best_individuals = tools.selBest(population, k=1)
        best_ind = best_individuals[0]

        logger.info(f"[SUCCESS] Evolution complete! Optimal DNA found")
        logger.info(f"  Final scores: stability={best_ind.fitness.values[0]:.3f}, "
                   f"unfitness={best_ind.fitness.values[1]:.3f}, "
                   f"uncreativity={best_ind.fitness.values[2]:.3f}")

        # OlfactoryDNA 객체로 변환하여 반환
        return OlfactoryDNA(
            genes=best_ind,
            fitness_scores=best_ind.fitness.values
        )

    def format_recipe(self, dna: OlfactoryDNA) -> Dict:
        """DNA를 읽기 쉬운 레시피 형식으로 변환"""

        recipe = {
            "top_notes": {},
            "middle_notes": {},
            "base_notes": {},
            "total_concentration": 0.0,
            "fitness": {
                "stability": 1.0 - (dna.fitness_scores[0] / 10.0),  # 정규화
                "suitability": 1.0 - dna.fitness_scores[1],  # 부적합도를 적합도로 변환
                "creativity": 1.0 - dna.fitness_scores[2]    # 비창의성을 창의성으로 변환
            }
        }

        for note_id, percentage in dna.genes:
            if note_id in self.notes_db:
                note = self.notes_db[note_id]
                note_name = note["name"]

                # 휘발성에 따라 분류
                if note["volatility"] > 0.7:
                    recipe["top_notes"][note_name] = f"{percentage:.1f}%"
                elif note["volatility"] > 0.3:
                    recipe["middle_notes"][note_name] = f"{percentage:.1f}%"
                else:
                    recipe["base_notes"][note_name] = f"{percentage:.1f}%"

                recipe["total_concentration"] += percentage

        return recipe


def example_usage():
    """사용 예시"""

    # 창세기 엔진 초기화
    engine = OlfactoryRecombinatorAI(
        population_size=100,
        generations=50,
        crossover_prob=0.8,
        mutation_prob=0.2
    )

    # CreativeBrief 생성 (사용자 요구사항)
    brief = CreativeBrief(
        emotional_palette=[0.3, 0.5, 0.2],  # 신선함, 로맨틱, 따뜻함
        fragrance_family="floral",
        mood="romantic",
        intensity=0.7,
        season="spring",
        gender="feminine"
    )

    # 진화 실행
    print("[MOGA] Starting OlfactoryRecombinatorAI: Generating olfactory DNA...")
    optimal_dna = engine.evolve(brief)

    # 결과 포맷팅
    recipe = engine.format_recipe(optimal_dna)

    print("\n[SUCCESS] Optimal fragrance recipe generated!")
    print(f"\nTop notes: {recipe['top_notes']}")
    print(f"Middle notes: {recipe['middle_notes']}")
    print(f"Base notes: {recipe['base_notes']}")
    print(f"Total concentration: {recipe['total_concentration']:.1f}%")
    print(f"\nFitness scores:")
    print(f"  Stability: {recipe['fitness']['stability']:.3f}")
    print(f"  Suitability: {recipe['fitness']['suitability']:.3f}")
    print(f"  Creativity: {recipe['fitness']['creativity']:.3f}")


if __name__ == "__main__":
    example_usage()