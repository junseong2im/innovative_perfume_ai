"""
OlfactoryRecombinatorAI - DEAP를 사용한 다중 목표 유전 알고리즘 (Production Level)
창세기 엔진: 후각적 DNA 생성 및 최적화 - 시뮬레이션 없음
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import hashlib
import sqlite3
from datetime import datetime
import json
import os
import sys

# DEAP imports
from deap import base, creator, tools, algorithms
import array

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.fragrance_chemistry import FragranceChemistry, FRAGRANCE_DATABASE


class DeterministicSelector:
    """Hash-based deterministic selection for reproducibility"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.counter = 0

    def _hash(self, data: str) -> int:
        """Generate deterministic hash"""
        content = f"{self.seed}_{self.counter}_{data}"
        self.counter += 1
        return int(hashlib.sha256(content.encode()).hexdigest(), 16)

    def uniform(self, low: float = 0.0, high: float = 1.0, context: str = "") -> float:
        """Deterministic uniform value"""
        hash_val = self._hash(f"uniform_{low}_{high}_{context}")
        normalized = (hash_val % 1000000) / 1000000.0
        return low + normalized * (high - low)

    def randint(self, low: int, high: int, context: str = "") -> int:
        """Deterministic integer in range"""
        hash_val = self._hash(f"randint_{low}_{high}_{context}")
        return low + (hash_val % (high - low + 1))

    def normal(self, mean: float = 0.0, std: float = 1.0, context: str = "") -> float:
        """Deterministic normal distribution"""
        hash_val = self._hash(f"normal_{mean}_{std}_{context}")
        # Box-Muller transform
        u1 = (hash_val % 999999 + 1) / 1000000
        u2 = ((hash_val >> 20) % 999999 + 1) / 1000000
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mean + std * z


class FragranceRecipeDatabase:
    """Production database for fragrance recipes"""

    def __init__(self, db_path: str = "olfactory_dna.db"):
        self.conn = sqlite3.connect(db_path)
        self._initialize_tables()
        self._populate_real_data()

    def _initialize_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Real fragrance ingredients
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                cas_number TEXT,
                category TEXT NOT NULL,
                odor_family TEXT,
                volatility REAL,
                intensity REAL,
                price_per_kg REAL,
                ifra_limit REAL
            )
        """)

        # Existing fragrances for comparison
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS existing_fragrances (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                brand TEXT,
                year INTEGER,
                recipe TEXT,
                notes TEXT
            )
        """)

        # Generated DNA records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_dna (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                dna TEXT,
                recipe TEXT,
                fitness_scores TEXT,
                brief TEXT
            )
        """)

        self.conn.commit()

    def _populate_real_data(self):
        """Populate with real fragrance data"""
        cursor = self.conn.cursor()

        # Check if already populated
        cursor.execute("SELECT COUNT(*) FROM ingredients")
        if cursor.fetchone()[0] > 0:
            return

        # Real fragrance ingredients
        ingredients = [
            # Top Notes
            ("bergamot", "8007-75-8", "top", "citrus", 0.95, 0.8, 45.0, 2.0),
            ("lemon", "8008-56-8", "top", "citrus", 0.92, 0.85, 35.0, 3.0),
            ("orange", "8008-57-9", "top", "citrus", 0.90, 0.75, 25.0, 5.0),
            ("grapefruit", "8016-20-4", "top", "citrus", 0.88, 0.7, 40.0, 2.5),
            ("lavender", "8000-28-0", "top", "herbal", 0.85, 0.7, 60.0, 20.0),
            ("eucalyptus", "8000-48-4", "top", "fresh", 0.85, 0.9, 20.0, 1.0),
            ("peppermint", "8006-90-4", "top", "fresh", 0.83, 0.85, 30.0, 1.0),

            # Middle Notes
            ("rose", "8007-01-0", "middle", "floral", 0.6, 0.95, 5000.0, 0.2),
            ("jasmine", "8022-96-6", "middle", "floral", 0.55, 1.0, 4500.0, 0.7),
            ("violet", "8015-99-8", "middle", "floral", 0.58, 0.8, 3000.0, 1.0),
            ("ylang_ylang", "8006-81-3", "middle", "floral", 0.58, 0.85, 280.0, 0.8),
            ("geranium", "8000-46-2", "middle", "floral", 0.62, 0.75, 120.0, 5.0),
            ("neroli", "8016-38-4", "middle", "floral", 0.65, 0.8, 2000.0, 1.0),

            # Base Notes
            ("sandalwood", "8006-87-9", "base", "woody", 0.2, 0.6, 200.0, 10.0),
            ("cedarwood", "8000-27-9", "base", "woody", 0.25, 0.5, 50.0, 15.0),
            ("vetiver", "8016-96-4", "base", "woody", 0.1, 0.85, 180.0, 8.0),
            ("patchouli", "8014-09-3", "base", "woody", 0.15, 0.9, 120.0, 12.0),
            ("vanilla", "8024-06-4", "base", "sweet", 0.05, 0.7, 600.0, 10.0),
            ("amber", "9000-02-6", "base", "amber", 0.03, 0.8, 250.0, 5.0),
            ("musk", "various", "base", "musk", 0.02, 1.0, 150.0, 1.5),
            ("benzoin", "9000-05-9", "base", "balsamic", 0.08, 0.65, 80.0, 20.0),
            ("tonka", "90028-06-1", "base", "sweet", 0.06, 0.75, 150.0, 10.0),
            ("oakmoss", "9000-50-4", "base", "mossy", 0.12, 0.7, 95.0, 0.1),
            ("incense", "8021-39-4", "base", "resinous", 0.08, 0.85, 120.0, 5.0),
            ("labdanum", "8016-26-0", "base", "resinous", 0.10, 0.8, 180.0, 8.0),
            ("myrrh", "8016-37-3", "base", "balsamic", 0.09, 0.75, 200.0, 5.0),
            ("oud", "various", "base", "woody", 0.05, 0.95, 8000.0, 0.5),
            ("civet", "various", "base", "animalic", 0.02, 0.9, 5000.0, 0.1),
            ("castoreum", "8023-83-4", "base", "animalic", 0.03, 0.85, 3000.0, 0.2)
        ]

        # Insert ingredients
        for ingredient in ingredients:
            cursor.execute("""
                INSERT OR IGNORE INTO ingredients
                (name, cas_number, category, odor_family, volatility, intensity, price_per_kg, ifra_limit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ingredient)

        # Add some classic fragrances for comparison
        classic_fragrances = [
            ("Chanel No. 5", "Chanel", 1921,
             '{"top": {"bergamot": 10, "lemon": 8}, "middle": {"rose": 20, "jasmine": 25}, "base": {"sandalwood": 15, "vanilla": 10}}',
             "aldehydic floral"),
            ("Shalimar", "Guerlain", 1925,
             '{"top": {"bergamot": 12, "lemon": 5}, "middle": {"rose": 15, "jasmine": 18}, "base": {"vanilla": 20, "benzoin": 10}}',
             "oriental"),
            ("Acqua di Gio", "Giorgio Armani", 1996,
             '{"top": {"bergamot": 15, "lemon": 10}, "middle": {"jasmine": 12, "neroli": 8}, "base": {"cedarwood": 10, "patchouli": 8}}',
             "aquatic fresh")
        ]

        for fragrance in classic_fragrances:
            cursor.execute("""
                INSERT OR IGNORE INTO existing_fragrances
                (name, brand, year, recipe, notes)
                VALUES (?, ?, ?, ?, ?)
            """, fragrance)

        self.conn.commit()

    def get_all_ingredients(self) -> List[Dict]:
        """Get all ingredients from database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, category, odor_family, volatility, intensity, price_per_kg, ifra_limit
            FROM ingredients
        """)

        ingredients = []
        for row in cursor.fetchall():
            ingredients.append({
                'name': row[0],
                'category': row[1],
                'odor_family': row[2],
                'volatility': row[3],
                'intensity': row[4],
                'price_per_kg': row[5],
                'ifra_limit': row[6]
            })
        return ingredients

    def get_existing_fragrances(self) -> Dict:
        """Get existing fragrances for creativity comparison"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, recipe FROM existing_fragrances")

        fragrances = {}
        for row in cursor.fetchall():
            fragrances[row[0]] = json.loads(row[1])
        return fragrances

    def save_generated_dna(self, dna: List[float], recipe: Dict, fitness_scores: Dict, brief: Dict):
        """Save generated DNA to database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO generated_dna
            (timestamp, dna, recipe, fitness_scores, brief)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            json.dumps(dna),
            json.dumps(recipe),
            json.dumps(fitness_scores),
            json.dumps(brief)
        ))
        self.conn.commit()


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
    """창세기 엔진 - DEAP 기반 MOGA 최적화 (Production Level)"""

    def __init__(self):
        self.chemistry = FragranceChemistry()
        self.selector = DeterministicSelector(42)
        self.database = FragranceRecipeDatabase()

        # Get ingredients from database
        db_ingredients = self.database.get_all_ingredients()
        self.all_ingredients = [ing['name'] for ing in db_ingredients][:30]  # 상위 30개
        self.ingredient_to_idx = {ing: i for i, ing in enumerate(self.all_ingredients)}
        self.ingredient_data = {ing['name']: ing for ing in db_ingredients}

        # DEAP 설정
        self._setup_deap()

    def _setup_deap(self):
        """DEAP 환경 설정"""
        # 피트니스 클래스 생성 (다중 목표, 최대화)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))

        # 개체 클래스 생성
        if not hasattr(creator, "Individual"):
            creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMulti)

        # 툴박스 초기화
        self.toolbox = base.Toolbox()

        # 개체 생성 함수 - deterministic initialization
        self.toolbox.register("attr_float", self._deterministic_random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=len(self.all_ingredients))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 평가 함수 등록
        self.toolbox.register("evaluate", self._evaluate_individual)

        # 유전 연산자 등록 - 실수값에 적합한 연산자
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
        self.toolbox.register("mutate", self._deterministic_mutate)  # Custom mutation
        self.toolbox.register("select", tools.selNSGA2)  # NSGA-II 선택

    def _deterministic_random(self) -> float:
        """Deterministic random value for initialization"""
        return self.selector.uniform(0, 1, "init")

    def _deterministic_mutate(self, individual, mu=0, sigma=0.2, indpb=0.1):
        """Deterministic Gaussian mutation"""
        for i in range(len(individual)):
            if self.selector.uniform(0, 1, f"mut_{i}") < indpb:
                individual[i] += self.selector.normal(mu, sigma, f"mut_val_{i}")
                individual[i] = max(0.0, min(1.0, individual[i]))
        return individual,

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

            # 노트 타입 결정 (데이터베이스 기반)
            if ingredient in self.ingredient_data:
                category = self.ingredient_data[ingredient]['category']
                recipe[category][ingredient] = concentration

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
            heavy_notes = ["musk", "amber", "patchouli", "oud"]
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
        existing_fragrances = self.database.get_existing_fragrances()

        if not existing_fragrances:
            return 0.8  # 데이터베이스가 없으면 기본 창의성

        # 기존 향수들과의 유사도 계산
        max_similarity = 0

        recipe_set = set()
        for notes in recipe.values():
            recipe_set.update(notes.keys())

        for existing in existing_fragrances.values():
            existing_set = set()
            if isinstance(existing, dict):
                for note_type in ["top", "middle", "base"]:
                    if note_type in existing:
                        existing_set.update(existing[note_type].keys())

            # Jaccard 유사도
            if recipe_set or existing_set:
                intersection = len(recipe_set & existing_set)
                union = len(recipe_set | existing_set)
                if union > 0:
                    similarity = intersection / union
                    max_similarity = max(max_similarity, similarity)

        # 창의성 = 1 - 최대 유사도
        creativity = 1.0 - max_similarity

        # 재료 다양성 보너스
        unique_families = set()
        for ingredient in recipe_set:
            if ingredient in self.ingredient_data:
                unique_families.add(self.ingredient_data[ingredient]['odor_family'])

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
        harmony = self._calculate_harmony_score(recipe)
        fitness = self._calculate_fitness_score(recipe, brief)
        creativity = self._calculate_creativity_score(recipe)

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
            print(f"Objectives: Harmony, Fitness, Creativity (all maximizing)")

        # 진화 루프
        for gen in range(generations):
            # 선택
            offspring = self.toolbox.select(population, population_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # 교차 (deterministic)
            for i in range(0, len(offspring) - 1, 2):
                if self.selector.uniform(0, 1, f"cross_{gen}_{i}") < 0.7:  # 교차 확률
                    self.toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # 돌연변이 (deterministic)
            for i, mutant in enumerate(offspring):
                if self.selector.uniform(0, 1, f"mut_{gen}_{i}") < 0.2:  # 돌연변이 확률
                    self.toolbox.mutate(mutant)
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
        best_score = float('-inf')

        for ind in hof:
            harmony, fitness, creativity = ind.fitness.values
            score = (harmony * weights["harmony"] +
                    fitness * weights["fitness"] +
                    creativity * weights["creativity"])

            if score > best_score:
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

        # 결과를 데이터베이스에 저장
        fitness_scores = {
            "harmony": best_individual.fitness.values[0],
            "fitness": best_individual.fitness.values[1],
            "creativity": best_individual.fitness.values[2],
            "overall": best_score
        }

        brief_dict = {
            "fragrance_family": brief.fragrance_family,
            "season": brief.season,
            "intensity": brief.intensity,
            "keywords": brief.keywords
        }

        self.database.save_generated_dna(
            list(best_individual),
            best_recipe,
            fitness_scores,
            brief_dict
        )

        return {
            "olfactory_dna": list(best_individual),
            "recipe": best_recipe,
            "fitness_values": fitness_scores,
            "evaluation": evaluation,
            "pareto_front_size": len(hof),
            "generation_stats": logbook,
            "brief": brief_dict
        }

    def evolve_dna(
        self,
        parent_dna: List[float],
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.2
    ) -> List[float]:
        """DNA 진화 - 돌연변이 적용 (deterministic)"""
        child = parent_dna.copy()

        for i in range(len(child)):
            if self.selector.uniform(0, 1, f"evolve_{i}") < mutation_rate:
                # Deterministic Gaussian mutation
                child[i] += self.selector.normal(0, mutation_strength, f"evolve_val_{i}")
                child[i] = np.clip(child[i], 0, 1)

        return child

    def crossover_dna(
        self,
        parent1_dna: List[float],
        parent2_dna: List[float],
        method: str = "uniform"
    ) -> Tuple[List[float], List[float]]:
        """DNA 교차 - 두 부모의 유전자 조합 (deterministic)"""
        if method == "uniform":
            # 균일 교차
            child1, child2 = [], []
            for i in range(len(parent1_dna)):
                if self.selector.uniform(0, 1, f"cross_uniform_{i}") < 0.5:
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
            # 2점 교차 (deterministic)
            length = len(parent1_dna)
            point1 = self.selector.randint(0, length // 2, "cross_pt1")
            point2 = self.selector.randint(length // 2, length, "cross_pt2")

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
    print(f"\nFitness Values: {result['fitness_values']}")
    print(f"Evaluation: {result['evaluation']}")