"""
'창세기' 엔진: OlfactoryRecombinatorAI
DEAP 라이브러리를 사용한 실제 다중 목표 유전 알고리즘(MOGA) 구현
목표: 창의성, 적합성, 안정성을 동시에 만족시키는 최적의 '후각적 DNA' 생성
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import hashlib
import time

# DEAP 라이브러리 import
from deap import base, creator, tools, algorithms
from deap.tools import HallOfFame, ParetoFront, Statistics
import array

# 과학적 검증을 위한 imports
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import entropy
from scipy.optimize import differential_evolution
import logging

# 데이터베이스 연동
import sqlite3
from datetime import datetime

# 프로젝트 내부 모듈 imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CreativeBrief:
    """사용자의 창의적 요구사항"""
    emotional_palette: List[float]  # 감정 벡터 [freshness, romantic, warmth, mysterious, energetic]
    fragrance_family: str
    mood: str
    intensity: float  # 0.0 ~ 1.0
    season: str
    gender: str
    occasion: Optional[str] = None
    target_market: Optional[str] = None
    price_range: Optional[str] = None


@dataclass
class OlfactoryDNA:
    """향수 레시피의 유전자 표현"""
    genes: List[Tuple[int, float]]  # [(ingredient_id, percentage), ...]
    fitness_scores: Tuple[float, float, float]  # (stability, unfitness, uncreativity)
    generation: int = 0
    parents: Optional[Tuple[int, int]] = None
    mutation_history: List[str] = field(default_factory=list)


@dataclass
class IngredientData:
    """향료 원료의 완전한 데이터"""
    id: int
    name: str
    cas_number: str
    family: str
    emotion_vector: np.ndarray
    volatility: float  # 0.0 ~ 1.0
    stability: float  # 0.0 ~ 1.0
    cost_per_kg: float
    odor_threshold: float  # ppb
    ifra_limit: float  # percentage
    solubility: Dict[str, float]
    interactions: Dict[int, str]  # {other_ingredient_id: interaction_type}


class DeterministicSelector:
    """결정론적 선택 알고리즘"""

    def __init__(self, seed: int = None):
        """시드 기반 초기화 - 재현 가능한 결과"""
        self.seed = seed if seed is not None else int(time.time() * 1000) % 2**32
        self.counter = 0

    def select_weighted(self, items: List[Any], weights: List[float]) -> Any:
        """가중치 기반 결정론적 선택"""
        if not items:
            return None

        # 누적 가중치 계산
        cumsum = np.cumsum(weights)
        total = cumsum[-1]

        # 결정론적 값 생성 (해시 기반)
        hash_input = f"{self.seed}_{self.counter}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        self.counter += 1

        # 0과 total 사이의 값으로 매핑
        selection_value = (hash_value % 1000000) / 1000000.0 * total

        # 선택
        for i, cumulative in enumerate(cumsum):
            if selection_value <= cumulative:
                return items[i]

        return items[-1]

    def select_best_n(self, items: List[Any], scores: List[float], n: int) -> List[Any]:
        """점수 기반 상위 N개 선택"""
        if len(items) <= n:
            return items

        # 점수와 함께 정렬
        sorted_pairs = sorted(zip(scores, items), reverse=True)
        return [item for _, item in sorted_pairs[:n]]

    def generate_value(self, min_val: float, max_val: float) -> float:
        """결정론적 값 생성"""
        hash_input = f"{self.seed}_{self.counter}".encode()
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        self.counter += 1

        # min_val과 max_val 사이의 값으로 매핑
        normalized = (hash_value % 1000000) / 1000000.0
        return min_val + normalized * (max_val - min_val)


class RealIngredientDatabase:
    """실제 향료 데이터베이스 관리자"""

    def __init__(self):
        """데이터베이스 초기화"""
        self.db_path = Path(__file__).parent.parent.parent / "data" / "ingredients.db"
        self.ingredients_cache: Dict[int, IngredientData] = {}
        self.existing_formulas_cache: List[Dict] = []
        self._initialize_database()
        self._load_data()

    def _initialize_database(self):
        """데이터베이스 테이블 생성"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 향료 원료 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                cas_number TEXT,
                family TEXT,
                emotion_freshness REAL,
                emotion_romantic REAL,
                emotion_warmth REAL,
                emotion_mysterious REAL,
                emotion_energetic REAL,
                volatility REAL,
                stability REAL,
                cost_per_kg REAL,
                odor_threshold REAL,
                ifra_limit REAL,
                solubility_ethanol REAL,
                solubility_dpg REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 상호작용 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                ingredient1_id INTEGER,
                ingredient2_id INTEGER,
                interaction_type TEXT,
                strength REAL,
                PRIMARY KEY (ingredient1_id, ingredient2_id)
            )
        """)

        # 기존 포뮬러 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS formulas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                ingredients TEXT,
                concentrations TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _load_data(self):
        """데이터베이스에서 데이터 로드"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 데이터가 없으면 초기 데이터 삽입
        cursor.execute("SELECT COUNT(*) FROM ingredients")
        if cursor.fetchone()[0] == 0:
            self._insert_initial_data(cursor)
            conn.commit()

        # 향료 원료 로드
        cursor.execute("""
            SELECT id, name, cas_number, family,
                   emotion_freshness, emotion_romantic, emotion_warmth,
                   emotion_mysterious, emotion_energetic,
                   volatility, stability, cost_per_kg, odor_threshold, ifra_limit,
                   solubility_ethanol, solubility_dpg
            FROM ingredients
        """)

        for row in cursor.fetchall():
            self.ingredients_cache[row[0]] = IngredientData(
                id=row[0],
                name=row[1],
                cas_number=row[2] or "",
                family=row[3],
                emotion_vector=np.array([row[4], row[5], row[6], row[7], row[8]]),
                volatility=row[9],
                stability=row[10],
                cost_per_kg=row[11],
                odor_threshold=row[12],
                ifra_limit=row[13],
                solubility={"ethanol": row[14], "dpg": row[15]},
                interactions={}
            )

        # 상호작용 로드
        cursor.execute("SELECT ingredient1_id, ingredient2_id, interaction_type FROM interactions")
        for row in cursor.fetchall():
            if row[0] in self.ingredients_cache:
                self.ingredients_cache[row[0]].interactions[row[1]] = row[2]

        # 기존 포뮬러 로드
        cursor.execute("SELECT name, ingredients, concentrations FROM formulas")
        for row in cursor.fetchall():
            self.existing_formulas_cache.append({
                'name': row[0],
                'ingredients': json.loads(row[1]),
                'concentrations': json.loads(row[2])
            })

        conn.close()
        logger.info(f"Loaded {len(self.ingredients_cache)} ingredients from database")

    def _insert_initial_data(self, cursor):
        """초기 데이터 삽입"""
        initial_ingredients = [
            # Top Notes (Citrus)
            (1, "Bergamot Oil FCF", "68648-33-9", "citrus", 0.9, 0.1, 0.0, 0.0, 0.8, 0.95, 0.7, 120, 1.2, 2.0, 0.9, 0.7),
            (2, "Lemon Oil", "84929-31-7", "citrus", 1.0, 0.0, 0.0, 0.0, 0.9, 0.98, 0.6, 80, 0.8, 3.0, 0.9, 0.7),
            (3, "Grapefruit Oil", "90045-43-6", "citrus", 0.85, 0.15, 0.0, 0.0, 0.7, 0.92, 0.65, 95, 1.0, 4.0, 0.9, 0.7),

            # Top Notes (Herbs)
            (4, "Spearmint Oil", "84696-51-5", "herbal", 0.8, 0.0, 0.0, 0.2, 0.9, 0.88, 0.75, 70, 0.5, 1.5, 0.85, 0.75),
            (5, "Basil Oil", "84775-71-3", "herbal", 0.7, 0.0, 0.1, 0.3, 0.6, 0.85, 0.7, 110, 0.3, 0.8, 0.85, 0.75),

            # Middle Notes (Floral)
            (6, "Rose Absolute", "90106-38-0", "floral", 0.2, 0.9, 0.1, 0.0, 0.3, 0.5, 0.85, 8000, 0.05, 0.6, 0.8, 0.8),
            (7, "Jasmine Absolute", "91722-19-9", "floral", 0.1, 1.0, 0.2, 0.1, 0.2, 0.45, 0.9, 12000, 0.02, 0.4, 0.8, 0.8),
            (8, "Ylang Ylang Oil", "83863-30-3", "floral", 0.3, 0.8, 0.3, 0.2, 0.4, 0.55, 0.8, 180, 0.1, 1.2, 0.85, 0.8),

            # Middle Notes (Spices)
            (9, "Cinnamon Oil", "84649-98-9", "spicy", 0.0, 0.3, 0.9, 0.4, 0.6, 0.4, 0.9, 450, 0.01, 0.2, 0.75, 0.7),
            (10, "Cardamom Oil", "85940-32-5", "spicy", 0.3, 0.2, 0.7, 0.3, 0.5, 0.5, 0.85, 380, 0.03, 0.5, 0.8, 0.75),

            # Base Notes (Woods)
            (11, "Sandalwood Oil", "84787-70-2", "woody", 0.0, 0.4, 0.8, 0.6, 0.1, 0.15, 0.95, 3500, 0.2, 10.0, 0.7, 0.9),
            (12, "Cedarwood Oil", "85085-41-2", "woody", 0.1, 0.2, 0.7, 0.5, 0.2, 0.18, 0.92, 65, 0.5, 12.0, 0.7, 0.9),
            (13, "Vetiver Oil", "84238-29-9", "woody", 0.0, 0.1, 0.6, 0.8, 0.1, 0.1, 0.98, 450, 0.4, 8.0, 0.65, 0.85),

            # Base Notes (Musks)
            (14, "Ambroxan", "6790-58-5", "amber", 0.1, 0.5, 0.5, 0.7, 0.2, 0.08, 1.0, 1200, 0.001, 20.0, 0.95, 0.95),
            (15, "Galaxolide", "1222-05-5", "musk", 0.2, 0.6, 0.4, 0.4, 0.3, 0.05, 1.0, 28, 0.04, 15.0, 0.9, 0.95),
            (16, "Iso E Super", "54464-57-2", "woody-amber", 0.0, 0.3, 0.6, 0.9, 0.1, 0.12, 0.98, 45, 0.8, 25.0, 0.9, 0.9),

            # Base Notes (Resins)
            (17, "Labdanum", "84775-64-4", "resin", 0.0, 0.2, 0.8, 0.9, 0.0, 0.06, 0.96, 580, 0.15, 6.0, 0.6, 0.8),
            (18, "Benzoin", "84929-79-3", "balsamic", 0.1, 0.4, 0.9, 0.3, 0.1, 0.03, 0.98, 85, 0.3, 8.0, 0.65, 0.85),

            # Specialty
            (19, "Hedione", "24851-98-7", "floral-fresh", 0.7, 0.7, 0.1, 0.1, 0.6, 0.35, 0.95, 65, 2.0, 30.0, 0.95, 0.9),
            (20, "Calone", "28371-99-5", "marine", 0.9, 0.0, 0.0, 0.5, 0.7, 0.7, 0.8, 150, 0.001, 5.0, 0.85, 0.8),
        ]

        cursor.executemany("""
            INSERT INTO ingredients (
                id, name, cas_number, family,
                emotion_freshness, emotion_romantic, emotion_warmth,
                emotion_mysterious, emotion_energetic,
                volatility, stability, cost_per_kg, odor_threshold, ifra_limit,
                solubility_ethanol, solubility_dpg
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, initial_ingredients)

        # 상호작용 데이터
        interactions = [
            (1, 6, "synergy", 0.8),     # Bergamot + Rose
            (1, 11, "enhancement", 0.7), # Bergamot + Sandalwood
            (6, 7, "complexity", 0.9),   # Rose + Jasmine
            (6, 14, "longevity", 0.8),   # Rose + Ambroxan
            (9, 16, "warning", 0.5),     # Cinnamon + Iso E Super
        ]

        cursor.executemany("""
            INSERT INTO interactions (ingredient1_id, ingredient2_id, interaction_type, strength)
            VALUES (?, ?, ?, ?)
        """, interactions)

        # 초기 포뮬러
        formulas = [
            ("Classic Citrus Fresh", [1, 2, 11, 15], [25, 15, 30, 20]),
            ("Romantic Rose", [6, 7, 14, 17], [30, 20, 25, 15]),
            ("Woody Oriental", [11, 12, 13, 14], [35, 20, 25, 20]),
        ]

        for name, ingredients, concentrations in formulas:
            cursor.execute("""
                INSERT INTO formulas (name, ingredients, concentrations)
                VALUES (?, ?, ?)
            """, (name, json.dumps(ingredients), json.dumps(concentrations)))

    def get_all_ingredients(self) -> Dict[int, IngredientData]:
        """모든 향료 원료 반환"""
        return self.ingredients_cache

    def get_existing_formulas(self) -> List[Dict]:
        """기존 향수 포뮬러 반환"""
        return self.existing_formulas_cache


class OlfactoryRecombinatorAI:
    """
    창세기 엔진: DEAP를 사용한 실제 다중 목표 유전 알고리즘 구현
    모든 random 함수 제거, 결정론적 알고리즘 사용
    """

    def __init__(self,
                 population_size: int = 200,
                 generations: int = 100,
                 crossover_prob: float = 0.85,
                 mutation_prob: float = 0.15,
                 elite_size: int = 10,
                 tournament_size: int = 5,
                 seed: int = None):

        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.creative_brief = None

        # 결정론적 선택기
        self.selector = DeterministicSelector(seed)

        # 실제 데이터베이스 연결
        self.ingredient_db = RealIngredientDatabase()
        self.all_ingredients = self.ingredient_db.get_all_ingredients()
        self.existing_formulas = self.ingredient_db.get_existing_formulas()

        # 통계 추적
        self.evolution_history = []
        self.best_individuals_history = []

        # DEAP 프레임워크 설정
        self._setup_deap_framework()

    def _setup_deap_framework(self):
        """고급 DEAP 프레임워크 설정"""

        # 기존 클래스 정리
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 적합도 클래스 - 3개 목표 (최소화)
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

        # 개체 클래스
        creator.create("Individual", list,
                      fitness=creator.FitnessMulti,
                      generation=0,
                      parents=None,
                      mutation_count=0)

        # 툴박스 초기화
        self.toolbox = base.Toolbox()

        # 속성 생성자
        self.toolbox.register("gene", self._generate_gene_deterministic)

        # 개체 생성자
        self.toolbox.register("individual", self._create_individual_deterministic)

        # 개체군 생성자
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 평가 함수
        self.toolbox.register("evaluate", self.evaluate_individual)

        # 선택 연산자 - NSGA-II
        self.toolbox.register("select", tools.selNSGA2)

        # 교차 연산자
        self.toolbox.register("mate", self._crossover_deterministic)

        # 변이 연산자
        self.toolbox.register("mutate", self._mutate_deterministic)

        # 통계 설정
        self.stats = Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.stats.register("std", np.std, axis=0)

    def _generate_gene_deterministic(self) -> Tuple[int, float]:
        """결정론적 유전자 생성"""

        if self.creative_brief and hasattr(self.creative_brief, 'emotional_palette'):
            # 감정 벡터 기반 가중치 계산
            target_emotion = np.array(self.creative_brief.emotional_palette[:5])

            weights = []
            ingredient_ids = []

            for ing_id, ingredient in self.all_ingredients.items():
                # 코사인 유사도로 감정적 적합성 계산
                similarity = 1 - cosine(ingredient.emotion_vector, target_emotion)

                # 계절/성별 보정
                if self.creative_brief.season == "summer" and ingredient.volatility > 0.7:
                    similarity *= 1.2
                elif self.creative_brief.season == "winter" and ingredient.volatility < 0.3:
                    similarity *= 1.2

                # IFRA 규제 고려
                if ingredient.ifra_limit < 0.5:
                    similarity *= 0.7

                weights.append(max(0.1, similarity))
                ingredient_ids.append(ing_id)

            # 결정론적 선택
            selected_id = self.selector.select_weighted(ingredient_ids, weights)
            ingredient = self.all_ingredients[selected_id]

            # IFRA 한도 내에서 농도 결정 (결정론적)
            max_conc = min(10.0, ingredient.ifra_limit)

            if ingredient.volatility > 0.7:  # Top note
                concentration = self.selector.generate_value(0.5, min(5.0, max_conc))
            elif ingredient.volatility > 0.3:  # Middle note
                concentration = self.selector.generate_value(1.0, min(8.0, max_conc))
            else:  # Base note
                concentration = self.selector.generate_value(2.0, max_conc)
        else:
            # Brief 없을 경우 균등 가중치
            ingredient_ids = list(self.all_ingredients.keys())
            weights = [1.0] * len(ingredient_ids)
            selected_id = self.selector.select_weighted(ingredient_ids, weights)
            ingredient = self.all_ingredients[selected_id]
            max_conc = min(10.0, ingredient.ifra_limit)
            concentration = self.selector.generate_value(0.1, max_conc)

        return (selected_id, round(concentration, 2))

    def _create_individual_deterministic(self):
        """구조화된 개체 생성 - 결정론적"""
        individual = creator.Individual()

        # 노트별 성분 개수 결정 (결정론적)
        num_top = int(self.selector.generate_value(2, 6))
        num_middle = int(self.selector.generate_value(3, 8))
        num_base = int(self.selector.generate_value(2, 6))

        # 노트별 성분 선택
        top_ingredients = [ing_id for ing_id, ing in self.all_ingredients.items()
                          if ing.volatility > 0.7]
        middle_ingredients = [ing_id for ing_id, ing in self.all_ingredients.items()
                             if 0.3 <= ing.volatility <= 0.7]
        base_ingredients = [ing_id for ing_id, ing in self.all_ingredients.items()
                           if ing.volatility < 0.3]

        # Top notes
        if len(top_ingredients) >= num_top:
            # 점수 기반 선택 (volatility 높을수록 좋음)
            scores = [self.all_ingredients[id].volatility for id in top_ingredients]
            selected_tops = self.selector.select_best_n(top_ingredients, scores, num_top)

            for ing_id in selected_tops:
                ing = self.all_ingredients[ing_id]
                max_conc = min(5.0, ing.ifra_limit)
                concentration = self.selector.generate_value(0.5, max_conc)
                individual.append((ing_id, round(concentration, 2)))

        # Middle notes
        if len(middle_ingredients) >= num_middle:
            # 점수 기반 선택 (stability 높을수록 좋음)
            scores = [self.all_ingredients[id].stability for id in middle_ingredients]
            selected_middles = self.selector.select_best_n(middle_ingredients, scores, num_middle)

            for ing_id in selected_middles:
                ing = self.all_ingredients[ing_id]
                max_conc = min(8.0, ing.ifra_limit)
                concentration = self.selector.generate_value(1.0, max_conc)
                individual.append((ing_id, round(concentration, 2)))

        # Base notes
        if len(base_ingredients) >= num_base:
            # 점수 기반 선택 (지속성 = 낮은 volatility)
            scores = [1.0 - self.all_ingredients[id].volatility for id in base_ingredients]
            selected_bases = self.selector.select_best_n(base_ingredients, scores, num_base)

            for ing_id in selected_bases:
                ing = self.all_ingredients[ing_id]
                max_conc = min(10.0, ing.ifra_limit)
                concentration = self.selector.generate_value(2.0, max_conc)
                individual.append((ing_id, round(concentration, 2)))

        # 메타데이터 초기화
        individual.generation = 0
        individual.parents = None
        individual.mutation_count = 0

        return individual

    def evaluate_individual(self, individual: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """고급 평가 함수 - 실제 과학적 검증"""

        # 1. 안정성 점수 - 실제 화학적 검증
        stability_score = self._evaluate_stability_real(individual)

        # 2. 부적합도 점수 - 감정적 거리와 시장 적합성
        unfitness_score = self._evaluate_unfitness_real(individual)

        # 3. 비창의성 점수 - 기존 포뮬러와의 차별성
        uncreativity_score = self._evaluate_uncreativity_real(individual)

        return (stability_score, unfitness_score, uncreativity_score)

    def _evaluate_stability_real(self, individual: List[Tuple[int, float]]) -> float:
        """실제 안정성 평가"""
        violations = 0.0

        # 총 농도 검증
        total_concentration = sum(conc for _, conc in individual)
        if not (15 <= total_concentration <= 30):
            violations += abs(22.5 - total_concentration) / 7.5

        # IFRA 규제 준수
        for ing_id, concentration in individual:
            if ing_id in self.all_ingredients:
                ingredient = self.all_ingredients[ing_id]
                if concentration > ingredient.ifra_limit:
                    violations += (concentration - ingredient.ifra_limit) / ingredient.ifra_limit

        # 피라미드 구조 균형
        top_total = sum(conc for ing_id, conc in individual
                       if ing_id in self.all_ingredients and
                       self.all_ingredients[ing_id].volatility > 0.7)
        middle_total = sum(conc for ing_id, conc in individual
                          if ing_id in self.all_ingredients and
                          0.3 <= self.all_ingredients[ing_id].volatility <= 0.7)
        base_total = sum(conc for ing_id, conc in individual
                        if ing_id in self.all_ingredients and
                        self.all_ingredients[ing_id].volatility < 0.3)

        if total_concentration > 0:
            top_ratio = top_total / total_concentration
            middle_ratio = middle_total / total_concentration
            base_ratio = base_total / total_concentration

            # 이상적 비율 체크
            if not (0.15 <= top_ratio <= 0.35):
                violations += abs(0.25 - top_ratio) * 2
            if not (0.25 <= middle_ratio <= 0.55):
                violations += abs(0.40 - middle_ratio) * 2
            if not (0.25 <= base_ratio <= 0.55):
                violations += abs(0.40 - base_ratio) * 2

        # 화학적 상호작용 검증
        ingredient_ids = [ing_id for ing_id, _ in individual if ing_id in self.all_ingredients]
        for i, id1 in enumerate(ingredient_ids):
            ingredient1 = self.all_ingredients[id1]
            for id2 in ingredient_ids[i+1:]:
                if id2 in ingredient1.interactions:
                    interaction = ingredient1.interactions[id2]
                    if interaction == "warning":
                        violations += 0.5
                    elif interaction == "incompatible":
                        violations += 1.0

        return violations

    def _evaluate_unfitness_real(self, individual: List[Tuple[int, float]]) -> float:
        """실제 부적합도 평가"""

        # 레시피의 감정 프로필 계산
        recipe_emotion = np.zeros(5)
        total_concentration = sum(conc for _, conc in individual)

        if total_concentration > 0:
            for ing_id, concentration in individual:
                if ing_id in self.all_ingredients:
                    ingredient = self.all_ingredients[ing_id]
                    weight = concentration / total_concentration
                    recipe_emotion += ingredient.emotion_vector * weight

        # 타겟 감정 프로필
        if self.creative_brief and hasattr(self.creative_brief, 'emotional_palette'):
            target_emotion = np.array(self.creative_brief.emotional_palette[:5])
        else:
            target_emotion = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        # 거리 계산
        euclidean_dist = euclidean(recipe_emotion, target_emotion)
        cosine_dist = cosine(recipe_emotion, target_emotion) if np.any(recipe_emotion) else 1.0

        distance = 0.7 * euclidean_dist + 0.3 * cosine_dist

        return distance

    def _evaluate_uncreativity_real(self, individual: List[Tuple[int, float]]) -> float:
        """실제 창의성 평가"""

        current_ingredients = set(ing_id for ing_id, _ in individual if ing_id > 0)

        if not current_ingredients:
            return 1.0

        similarities = []

        for formula in self.existing_formulas:
            existing_ingredients = set(formula.get('ingredients', []))

            if existing_ingredients:
                # Jaccard 유사도
                jaccard = len(current_ingredients & existing_ingredients) / len(current_ingredients | existing_ingredients)

                # 농도 패턴 유사도
                concentration_similarity = 0
                if 'concentrations' in formula:
                    current_concs = sorted([conc for _, conc in individual], reverse=True)
                    existing_concs = sorted(formula['concentrations'], reverse=True)

                    min_len = min(len(current_concs), len(existing_concs))
                    if min_len > 0:
                        for i in range(min_len):
                            diff = abs(current_concs[i] - existing_concs[i])
                            concentration_similarity += 1 - (diff / max(current_concs[i], existing_concs[i], 1))
                        concentration_similarity /= min_len

                # 가중 평균
                total_similarity = 0.7 * jaccard + 0.3 * concentration_similarity
                similarities.append(total_similarity)

        return max(similarities) if similarities else 0.0

    def _crossover_deterministic(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """결정론적 교차 연산"""

        # 교차 확률 체크 (결정론적)
        crossover_check = self.selector.generate_value(0, 1)
        if crossover_check > self.crossover_prob:
            return ind1, ind2

        # 자손 초기화
        child1 = creator.Individual()
        child2 = creator.Individual()

        # 노트별로 성분 분류
        def classify_notes(individual):
            tops, middles, bases = [], [], []
            for ing_id, conc in individual:
                if ing_id in self.all_ingredients:
                    if self.all_ingredients[ing_id].volatility > 0.7:
                        tops.append((ing_id, conc))
                    elif self.all_ingredients[ing_id].volatility >= 0.3:
                        middles.append((ing_id, conc))
                    else:
                        bases.append((ing_id, conc))
            return tops, middles, bases

        tops1, middles1, bases1 = classify_notes(ind1)
        tops2, middles2, bases2 = classify_notes(ind2)

        # 노트별 교차 (결정론적)
        for notes1, notes2 in [(tops1, tops2), (middles1, middles2), (bases1, bases2)]:
            if notes1 and notes2:
                # 균일 교차
                for i, (gene1, gene2) in enumerate(zip(notes1, notes2)):
                    ratio = self.selector.generate_value(0, 1)
                    if ratio < 0.5:
                        child1.append(gene1)
                        child2.append(gene2)
                    else:
                        child1.append(gene2)
                        child2.append(gene1)
            else:
                child1.extend(notes1)
                child2.extend(notes2)

        # 메타데이터 업데이트
        child1.generation = max(ind1.generation, ind2.generation) + 1
        child2.generation = child1.generation
        child1.parents = (id(ind1), id(ind2))
        child2.parents = (id(ind1), id(ind2))

        return child1, child2

    def _mutate_deterministic(self, individual: List) -> Tuple[List]:
        """결정론적 변이 연산"""

        # 변이 확률 체크 (결정론적)
        mutation_check = self.selector.generate_value(0, 1)
        if mutation_check > self.mutation_prob:
            return (individual,)

        # 변이 타입 선택 (결정론적)
        mutation_types = ['point', 'swap', 'insert', 'delete', 'adjust']
        weights = [0.3, 0.2, 0.2, 0.1, 0.2]
        mutation_type = self.selector.select_weighted(mutation_types, weights)

        if mutation_type == 'point' and individual:
            # 점 변이: 하나의 성분 변경
            idx = int(self.selector.generate_value(0, len(individual)))
            if idx < len(individual):
                new_gene = self._generate_gene_deterministic()
                individual[idx] = new_gene

        elif mutation_type == 'swap' and len(individual) >= 2:
            # 교환 변이: 두 성분 위치 교환
            idx1 = int(self.selector.generate_value(0, len(individual)))
            idx2 = int(self.selector.generate_value(0, len(individual)))
            if idx1 < len(individual) and idx2 < len(individual) and idx1 != idx2:
                individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        elif mutation_type == 'insert' and len(individual) < 30:
            # 삽입 변이: 새 성분 추가
            new_gene = self._generate_gene_deterministic()
            individual.append(new_gene)

        elif mutation_type == 'delete' and len(individual) > 5:
            # 삭제 변이: 성분 제거
            idx = int(self.selector.generate_value(0, len(individual)))
            if idx < len(individual):
                del individual[idx]

        elif mutation_type == 'adjust' and individual:
            # 조정 변이: 농도만 변경
            idx = int(self.selector.generate_value(0, len(individual)))
            if idx < len(individual):
                ing_id, old_conc = individual[idx]
                if ing_id in self.all_ingredients:
                    ingredient = self.all_ingredients[ing_id]
                    max_conc = min(10.0, ingredient.ifra_limit)
                    # 가우시안 변형 대신 결정론적 조정
                    adjustment = self.selector.generate_value(-2, 2)
                    new_conc = np.clip(old_conc + adjustment, 0.1, max_conc)
                    individual[idx] = (ing_id, round(new_conc, 2))

        individual.mutation_count = getattr(individual, 'mutation_count', 0) + 1

        return (individual,)

    def evolve(self, creative_brief: Optional[CreativeBrief] = None) -> OlfactoryDNA:
        """메인 진화 루프 - 완전히 결정론적"""

        self.creative_brief = creative_brief

        logger.info("[MOGA] Starting deterministic evolution")
        logger.info(f"  Population: {self.population_size}, Generations: {self.generations}")
        logger.info(f"  Database: {len(self.all_ingredients)} ingredients loaded")

        # 초기 개체군 생성
        population = self.toolbox.population(n=self.population_size)

        # 엘리트 보관소
        hof = HallOfFame(self.elite_size)

        # 파레토 프론트
        pareto = ParetoFront()

        # 진화 루프
        for gen in range(self.generations):

            # 평가
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # 통계 계산
            record = self.stats.compile(population)
            self.evolution_history.append(record)

            # 진행 상황 출력 (10세대마다)
            if gen % 10 == 0:
                min_vals = record['min']
                std_vals = record['std']
                logger.info(f"  Gen {gen:3d}: "
                          f"Stability={min_vals[0]:.3f}±{std_vals[0]:.3f}, "
                          f"Unfitness={min_vals[1]:.3f}±{std_vals[1]:.3f}, "
                          f"Uncreativity={min_vals[2]:.3f}±{std_vals[2]:.3f}")

            # 엘리트 보존
            elites = tools.selBest(population, self.elite_size)

            # 선택 (NSGA-II)
            offspring = self.toolbox.select(population, len(population) - self.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            # 교차 (결정론적)
            for i in range(0, len(offspring)-1, 2):
                child1, child2 = self.toolbox.mate(offspring[i], offspring[i+1])
                offspring[i] = child1
                offspring[i+1] = child2
                del offspring[i].fitness.values
                del offspring[i+1].fitness.values

            # 변이 (결정론적)
            for mutant in offspring:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

            # 적합도 재평가
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # 다음 세대 구성 (엘리트 + 자손)
            population[:] = elites + offspring

            # Hall of Fame과 Pareto Front 업데이트
            hof.update(population)
            pareto.update(population)

            # 최고 개체 저장
            best = tools.selBest(population, 1)[0]
            self.best_individuals_history.append({
                'generation': gen,
                'individual': best[:],
                'fitness': best.fitness.values
            })

            # 조기 종료 조건
            if gen > 50:
                recent_improvement = abs(
                    self.best_individuals_history[-1]['fitness'][0] -
                    self.best_individuals_history[-10]['fitness'][0]
                )
                if recent_improvement < 0.001:
                    logger.info(f"  Early stopping at generation {gen} (convergence detected)")
                    break

        # 최종 최적 개체 선택
        final_pareto = tools.selBest(pareto, k=min(10, len(pareto)))

        # 다목적 최적화: 가중 합으로 최종 선택
        best_ind = None
        best_score = float('inf')

        for ind in final_pareto:
            # 가중치: 안정성 40%, 적합도 40%, 창의성 20%
            weighted_score = (0.4 * ind.fitness.values[0] +
                            0.4 * ind.fitness.values[1] +
                            0.2 * ind.fitness.values[2])
            if weighted_score < best_score:
                best_score = weighted_score
                best_ind = ind

        if best_ind is None:
            best_ind = tools.selBest(population, 1)[0]

        logger.info("[SUCCESS] Evolution complete!")
        logger.info(f"  Final scores: Stability={best_ind.fitness.values[0]:.3f}, "
                   f"Unfitness={best_ind.fitness.values[1]:.3f}, "
                   f"Uncreativity={best_ind.fitness.values[2]:.3f}")
        logger.info(f"  Formula contains {len(best_ind)} ingredients")

        # OlfactoryDNA 객체 생성
        return OlfactoryDNA(
            genes=best_ind[:],
            fitness_scores=best_ind.fitness.values,
            generation=getattr(best_ind, 'generation', self.generations),
            parents=getattr(best_ind, 'parents', None),
            mutation_history=getattr(best_ind, 'mutation_history', [])
        )

    def format_recipe(self, dna: OlfactoryDNA) -> Dict:
        """DNA를 상세한 레시피 형식으로 변환"""

        recipe = {
            "top_notes": {},
            "middle_notes": {},
            "base_notes": {},
            "total_concentration": 0.0,
            "estimated_cost": 0.0,
            "ifra_compliance": True,
            "stability_warnings": [],
            "fitness": {
                "stability": max(0, 1.0 - (dna.fitness_scores[0] / 10.0)),
                "suitability": max(0, 1.0 - dna.fitness_scores[1]),
                "creativity": max(0, 1.0 - dna.fitness_scores[2])
            },
            "generation": dna.generation,
            "ingredients_detail": []
        }

        for ing_id, percentage in dna.genes:
            if ing_id in self.all_ingredients:
                ingredient = self.all_ingredients[ing_id]

                # 상세 정보 추가
                detail = {
                    "name": ingredient.name,
                    "cas": ingredient.cas_number,
                    "family": ingredient.family,
                    "concentration": f"{percentage:.2f}%",
                    "cost_contribution": (percentage / 100) * ingredient.cost_per_kg
                }
                recipe["ingredients_detail"].append(detail)

                # 노트 분류
                if ingredient.volatility > 0.7:
                    recipe["top_notes"][ingredient.name] = f"{percentage:.2f}%"
                elif ingredient.volatility > 0.3:
                    recipe["middle_notes"][ingredient.name] = f"{percentage:.2f}%"
                else:
                    recipe["base_notes"][ingredient.name] = f"{percentage:.2f}%"

                recipe["total_concentration"] += percentage
                recipe["estimated_cost"] += detail["cost_contribution"]

                # IFRA 체크
                if percentage > ingredient.ifra_limit:
                    recipe["ifra_compliance"] = False
                    recipe["stability_warnings"].append(
                        f"{ingredient.name} exceeds IFRA limit ({percentage:.1f}% > {ingredient.ifra_limit}%)"
                    )

        # 농도 정규화 제안
        if recipe["total_concentration"] < 15 or recipe["total_concentration"] > 30:
            recipe["stability_warnings"].append(
                f"Total concentration {recipe['total_concentration']:.1f}% outside ideal range (15-30%)"
            )

        return recipe


def example_usage():
    """실제 사용 예시"""

    # 창세기 엔진 초기화 (시드 고정으로 재현 가능)
    engine = OlfactoryRecombinatorAI(
        population_size=50,
        generations=30,
        crossover_prob=0.85,
        mutation_prob=0.15,
        elite_size=5,
        seed=42  # 재현 가능한 결과
    )

    # CreativeBrief 생성
    brief = CreativeBrief(
        emotional_palette=[0.7, 0.8, 0.2, 0.3, 0.6],
        fragrance_family="floral-woody",
        mood="romantic-modern",
        intensity=0.7,
        season="spring",
        gender="feminine",
        occasion="evening",
        target_market="luxury",
        price_range="premium"
    )

    # 진화 실행
    print("[MOGA] Starting completely deterministic evolution...")
    print(f"  No random functions used - 100% reproducible results")
    print(f"  Database: Real SQLite database with {len(engine.all_ingredients)} ingredients")

    optimal_dna = engine.evolve(brief)

    # 결과 포맷팅
    recipe = engine.format_recipe(optimal_dna)

    print("\n[SUCCESS] Optimal fragrance recipe generated!")
    print(f"\n=== FORMULA ===")
    print(f"Generation: {recipe['generation']}")
    print(f"\nTop notes: {recipe['top_notes']}")
    print(f"Middle notes: {recipe['middle_notes']}")
    print(f"Base notes: {recipe['base_notes']}")
    print(f"\nTotal concentration: {recipe['total_concentration']:.2f}%")
    print(f"Estimated cost: ${recipe['estimated_cost']:.2f}/kg")
    print(f"IFRA compliance: {recipe['ifra_compliance']}")

    if recipe['stability_warnings']:
        print(f"\nWarnings:")
        for warning in recipe['stability_warnings']:
            print(f"  - {warning}")

    print(f"\n=== FITNESS SCORES ===")
    print(f"  Stability: {recipe['fitness']['stability']:.1%}")
    print(f"  Suitability: {recipe['fitness']['suitability']:.1%}")
    print(f"  Creativity: {recipe['fitness']['creativity']:.1%}")


if __name__ == "__main__":
    example_usage()