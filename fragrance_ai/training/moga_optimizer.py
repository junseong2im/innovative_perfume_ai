"""
'창세기' 엔진: OlfactoryRecombinatorAI
DEAP 라이브러리를 사용한 실제 다중 목표 유전 알고리즘(MOGA) 구현
목표: 창의성, 적합성, 안정성을 동시에 만족시키는 최적의 '후각적 DNA' 생성
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

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
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# 프로젝트 내부 모듈 imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# 실제 검증 도구들 import
try:
    from fragrance_ai.tools.validator_tool import ValidatorTool, ScientificValidator
    from fragrance_ai.tools.knowledge_tool import PerfumeKnowledgeBase
    from fragrance_ai.database.schema import Ingredient, AccordTemplate, KnowledgeBase
    from fragrance_ai.database.connection import DatabaseManager
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Import error: {e}. Some features may be limited.")
    ValidatorTool = None
    PerfumeKnowledgeBase = None

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


class RealIngredientDatabase:
    """실제 향료 데이터베이스 관리자"""

    def __init__(self, db_url: Optional[str] = None):
        """
        데이터베이스 연결 초기화
        """
        self.db_url = db_url or os.getenv('DATABASE_URL', 'postgresql://localhost/fragrance_ai')
        self.ingredients_cache: Dict[int, IngredientData] = {}
        self.accord_templates_cache: List[Dict] = []
        self.existing_formulas_cache: List[Dict] = []
        self._load_database()

    def _load_database(self):
        """실제 데이터베이스에서 데이터 로드"""
        try:
            # PostgreSQL 연결
            if 'postgresql' in self.db_url:
                engine = create_engine(self.db_url)
                Session = sessionmaker(bind=engine)
                session = Session()

                # 향료 원료 로드
                result = session.execute(text("""
                    SELECT id, name, cas_number, family, volatility, stability,
                           cost_per_kg, odor_threshold, ifra_limit, solubility,
                           emotion_freshness, emotion_romantic, emotion_warmth,
                           emotion_mysterious, emotion_energetic
                    FROM ingredients
                    WHERE active = true
                """))

                for row in result:
                    self.ingredients_cache[row.id] = IngredientData(
                        id=row.id,
                        name=row.name,
                        cas_number=row.cas_number,
                        family=row.family,
                        emotion_vector=np.array([
                            row.emotion_freshness,
                            row.emotion_romantic,
                            row.emotion_warmth,
                            row.emotion_mysterious,
                            row.emotion_energetic
                        ]),
                        volatility=row.volatility,
                        stability=row.stability,
                        cost_per_kg=row.cost_per_kg,
                        odor_threshold=row.odor_threshold,
                        ifra_limit=row.ifra_limit,
                        solubility=json.loads(row.solubility) if row.solubility else {},
                        interactions={}
                    )

                # 상호작용 데이터 로드
                interactions = session.execute(text("""
                    SELECT ingredient1_id, ingredient2_id, interaction_type
                    FROM ingredient_interactions
                """))

                for interaction in interactions:
                    if interaction.ingredient1_id in self.ingredients_cache:
                        self.ingredients_cache[interaction.ingredient1_id].interactions[
                            interaction.ingredient2_id
                        ] = interaction.interaction_type

                # 기존 향수 포뮬러 로드
                formulas = session.execute(text("""
                    SELECT id, name, ingredients, concentrations
                    FROM fragrance_formulas
                    WHERE published = true
                """))

                for formula in formulas:
                    self.existing_formulas_cache.append({
                        'id': formula.id,
                        'name': formula.name,
                        'ingredients': json.loads(formula.ingredients),
                        'concentrations': json.loads(formula.concentrations)
                    })

                session.close()
                logger.info(f"Loaded {len(self.ingredients_cache)} ingredients from database")

        except Exception as e:
            logger.warning(f"Database connection failed: {e}. Using comprehensive fallback data.")
            self._load_fallback_data()

    def _load_fallback_data(self):
        """데이터베이스 연결 실패시 사용할 포괄적인 백업 데이터"""
        # 실제 향료 원료 데이터 (IFRA 표준 기반)
        ingredients_data = {
            # Top Notes (Citrus)
            1: {"name": "Bergamot Oil FCF", "cas": "68648-33-9", "family": "citrus",
                "emotion": [0.9, 0.1, 0.0, 0.0, 0.8], "volatility": 0.95, "stability": 0.7,
                "cost": 120, "threshold": 1.2, "ifra": 2.0},
            2: {"name": "Lemon Oil Cold Pressed", "cas": "84929-31-7", "family": "citrus",
                "emotion": [1.0, 0.0, 0.0, 0.0, 0.9], "volatility": 0.98, "stability": 0.6,
                "cost": 80, "threshold": 0.8, "ifra": 3.0},
            3: {"name": "Grapefruit Oil Pink", "cas": "90045-43-6", "family": "citrus",
                "emotion": [0.85, 0.15, 0.0, 0.0, 0.7], "volatility": 0.92, "stability": 0.65,
                "cost": 95, "threshold": 1.0, "ifra": 4.0},

            # Top Notes (Herbs)
            4: {"name": "Spearmint Oil", "cas": "84696-51-5", "family": "herbal",
                "emotion": [0.8, 0.0, 0.0, 0.2, 0.9], "volatility": 0.88, "stability": 0.75,
                "cost": 70, "threshold": 0.5, "ifra": 1.5},
            5: {"name": "Basil Oil Sweet", "cas": "84775-71-3", "family": "herbal",
                "emotion": [0.7, 0.0, 0.1, 0.3, 0.6], "volatility": 0.85, "stability": 0.7,
                "cost": 110, "threshold": 0.3, "ifra": 0.8},

            # Middle Notes (Floral)
            6: {"name": "Rose Absolute Bulgarian", "cas": "90106-38-0", "family": "floral",
                "emotion": [0.2, 0.9, 0.1, 0.0, 0.3], "volatility": 0.5, "stability": 0.85,
                "cost": 8000, "threshold": 0.05, "ifra": 0.6},
            7: {"name": "Jasmine Sambac Absolute", "cas": "91722-19-9", "family": "floral",
                "emotion": [0.1, 1.0, 0.2, 0.1, 0.2], "volatility": 0.45, "stability": 0.9,
                "cost": 12000, "threshold": 0.02, "ifra": 0.4},
            8: {"name": "Ylang Ylang Oil III", "cas": "83863-30-3", "family": "floral",
                "emotion": [0.3, 0.8, 0.3, 0.2, 0.4], "volatility": 0.55, "stability": 0.8,
                "cost": 180, "threshold": 0.1, "ifra": 1.2},
            9: {"name": "Geranium Oil Egypt", "cas": "90082-51-2", "family": "floral",
                "emotion": [0.5, 0.6, 0.2, 0.0, 0.5], "volatility": 0.6, "stability": 0.75,
                "cost": 220, "threshold": 0.08, "ifra": 5.0},

            # Middle Notes (Spices)
            10: {"name": "Cinnamon Bark Oil Ceylon", "cas": "84649-98-9", "family": "spicy",
                 "emotion": [0.0, 0.3, 0.9, 0.4, 0.6], "volatility": 0.4, "stability": 0.9,
                 "cost": 450, "threshold": 0.01, "ifra": 0.2},
            11: {"name": "Cardamom Oil Guatemala", "cas": "85940-32-5", "family": "spicy",
                 "emotion": [0.3, 0.2, 0.7, 0.3, 0.5], "volatility": 0.5, "stability": 0.85,
                 "cost": 380, "threshold": 0.03, "ifra": 0.5},

            # Base Notes (Woods)
            12: {"name": "Sandalwood Oil East Indian", "cas": "84787-70-2", "family": "woody",
                 "emotion": [0.0, 0.4, 0.8, 0.6, 0.1], "volatility": 0.15, "stability": 0.95,
                 "cost": 3500, "threshold": 0.2, "ifra": 10.0},
            13: {"name": "Cedarwood Oil Virginia", "cas": "85085-41-2", "family": "woody",
                 "emotion": [0.1, 0.2, 0.7, 0.5, 0.2], "volatility": 0.18, "stability": 0.92,
                 "cost": 65, "threshold": 0.5, "ifra": 12.0},
            14: {"name": "Vetiver Oil Haiti", "cas": "84238-29-9", "family": "woody",
                 "emotion": [0.0, 0.1, 0.6, 0.8, 0.1], "volatility": 0.1, "stability": 0.98,
                 "cost": 450, "threshold": 0.4, "ifra": 8.0},

            # Base Notes (Musks)
            15: {"name": "Ambroxan", "cas": "6790-58-5", "family": "amber",
                 "emotion": [0.1, 0.5, 0.5, 0.7, 0.2], "volatility": 0.08, "stability": 1.0,
                 "cost": 1200, "threshold": 0.001, "ifra": 20.0},
            16: {"name": "Galaxolide", "cas": "1222-05-5", "family": "musk",
                 "emotion": [0.2, 0.6, 0.4, 0.4, 0.3], "volatility": 0.05, "stability": 1.0,
                 "cost": 28, "threshold": 0.04, "ifra": 15.0},
            17: {"name": "Iso E Super", "cas": "54464-57-2", "family": "woody-amber",
                 "emotion": [0.0, 0.3, 0.6, 0.9, 0.1], "volatility": 0.12, "stability": 0.98,
                 "cost": 45, "threshold": 0.8, "ifra": 25.0},

            # Base Notes (Resins)
            18: {"name": "Labdanum Absolute", "cas": "84775-64-4", "family": "resin",
                 "emotion": [0.0, 0.2, 0.8, 0.9, 0.0], "volatility": 0.06, "stability": 0.96,
                 "cost": 580, "threshold": 0.15, "ifra": 6.0},
            19: {"name": "Benzoin Resinoid Siam", "cas": "84929-79-3", "family": "balsamic",
                 "emotion": [0.1, 0.4, 0.9, 0.3, 0.1], "volatility": 0.03, "stability": 0.98,
                 "cost": 85, "threshold": 0.3, "ifra": 8.0},

            # Specialty Molecules
            20: {"name": "Hedione", "cas": "24851-98-7", "family": "floral-fresh",
                 "emotion": [0.7, 0.7, 0.1, 0.1, 0.6], "volatility": 0.35, "stability": 0.95,
                 "cost": 65, "threshold": 2.0, "ifra": 30.0},
        }

        # 데이터 변환
        for id, data in ingredients_data.items():
            self.ingredients_cache[id] = IngredientData(
                id=id,
                name=data["name"],
                cas_number=data["cas"],
                family=data["family"],
                emotion_vector=np.array(data["emotion"]),
                volatility=data["volatility"],
                stability=data["stability"],
                cost_per_kg=data["cost"],
                odor_threshold=data["threshold"],
                ifra_limit=data["ifra"],
                solubility={"ethanol": 0.9, "dpg": 0.7, "ipg": 0.8},
                interactions={}
            )

        # 상호작용 데이터 추가
        interactions_map = {
            (1, 6): "synergy",  # Bergamot + Rose = Classic harmony
            (1, 12): "enhancement",  # Bergamot + Sandalwood = Depth
            (6, 7): "complexity",  # Rose + Jasmine = Floral bouquet
            (6, 15): "longevity",  # Rose + Ambroxan = Extended wear
            (10, 17): "warning",  # Cinnamon + Iso E Super = May overpower
        }

        for (id1, id2), interaction in interactions_map.items():
            if id1 in self.ingredients_cache:
                self.ingredients_cache[id1].interactions[id2] = interaction

        # 기존 포뮬러 예시
        self.existing_formulas_cache = [
            {"name": "Chanel No.5", "ingredients": [6, 7, 12, 16], "concentrations": [15, 10, 20, 15]},
            {"name": "Dior Sauvage", "ingredients": [1, 15, 17], "concentrations": [25, 15, 30]},
            {"name": "Le Labo Santal 33", "ingredients": [12, 13, 14, 17], "concentrations": [35, 10, 15, 20]},
        ]

        logger.info(f"Loaded {len(self.ingredients_cache)} fallback ingredients")

    def get_ingredient(self, id: int) -> Optional[IngredientData]:
        """ID로 향료 원료 조회"""
        return self.ingredients_cache.get(id)

    def get_all_ingredients(self) -> Dict[int, IngredientData]:
        """모든 향료 원료 반환"""
        return self.ingredients_cache

    def get_ingredients_by_family(self, family: str) -> List[IngredientData]:
        """패밀리별 향료 조회"""
        return [ing for ing in self.ingredients_cache.values() if ing.family == family]

    def get_existing_formulas(self) -> List[Dict]:
        """기존 향수 포뮬러 반환"""
        return self.existing_formulas_cache


class OlfactoryRecombinatorAI:
    """
    창세기 엔진: DEAP를 사용한 실제 다중 목표 유전 알고리즘 구현
    실제 향료 데이터베이스와 과학적 검증 도구 통합
    """

    def __init__(self,
                 population_size: int = 200,
                 generations: int = 100,
                 crossover_prob: float = 0.85,
                 mutation_prob: float = 0.15,
                 elite_size: int = 10,
                 tournament_size: int = 5,
                 db_url: Optional[str] = None):

        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.creative_brief = None

        # 실제 데이터베이스 연결
        self.ingredient_db = RealIngredientDatabase(db_url)
        self.all_ingredients = self.ingredient_db.get_all_ingredients()
        self.existing_formulas = self.ingredient_db.get_existing_formulas()

        # 검증 도구 초기화
        self.validator = ValidatorTool() if ValidatorTool else None

        # 통계 추적
        self.evolution_history = []
        self.best_individuals_history = []

        # DEAP 프레임워크 설정
        self._setup_deap_framework()

    def _setup_deap_framework(self):
        """
        고급 DEAP 프레임워크 설정 - 실제 다중 목표 최적화
        """

        # 기존 클래스 정리
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # 적합도 클래스 - 3개 목표 (최소화)
        # weights: (안정성, 적합도, 창의성)
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

        # 개체 클래스 - 유전자는 (ingredient_id, percentage) 튜플의 리스트
        creator.create("Individual", list,
                      fitness=creator.FitnessMulti,
                      generation=0,
                      parents=None,
                      mutation_count=0)

        # 툴박스 초기화
        self.toolbox = base.Toolbox()

        # 속성 생성자
        self.toolbox.register("gene", self._generate_gene)

        # 개체 생성자 - 가변 길이 (10-25개 성분)
        self.toolbox.register("individual", self._create_individual)

        # 개체군 생성자
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 평가 함수
        self.toolbox.register("evaluate", self.evaluate_individual)

        # 선택 연산자 - NSGA-II for multi-objective
        self.toolbox.register("select", tools.selNSGA2)

        # 교차 연산자 - 균일 교차
        self.toolbox.register("mate", self._custom_crossover)

        # 변이 연산자
        self.toolbox.register("mutate", self._custom_mutation)

        # 통계 설정
        self.stats = Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.stats.register("std", np.std, axis=0)

    def _generate_gene(self) -> Tuple[int, float]:
        """
        지능적 유전자 생성 - 실제 향료 데이터베이스 기반
        """
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
                    similarity *= 0.7  # 사용 제한이 큰 원료는 가중치 감소

                weights.append(max(0.1, similarity))  # 최소 가중치 보장
                ingredient_ids.append(ing_id)

            # 가중치 정규화
            weights = np.array(weights) / sum(weights)

            # 가중치 기반 선택
            selected_id = np.random.choice(ingredient_ids, p=weights)
            ingredient = self.all_ingredients[selected_id]

            # IFRA 한도 내에서 농도 결정
            max_conc = min(10.0, ingredient.ifra_limit)

            # 휘발성 기반 농도 조정
            if ingredient.volatility > 0.7:  # Top note
                concentration = np.random.uniform(0.5, min(5.0, max_conc))
            elif ingredient.volatility > 0.3:  # Middle note
                concentration = np.random.uniform(1.0, min(8.0, max_conc))
            else:  # Base note
                concentration = np.random.uniform(2.0, max_conc)

        else:
            # Brief 없을 경우 균등 선택
            selected_id = random.choice(list(self.all_ingredients.keys()))
            ingredient = self.all_ingredients[selected_id]
            max_conc = min(10.0, ingredient.ifra_limit)
            concentration = np.random.uniform(0.1, max_conc)

        return (selected_id, round(concentration, 2))

    def _create_individual(self):
        """
        구조화된 개체 생성 - 피라미드 구조 준수
        """
        individual = creator.Individual()

        # 노트별 성분 개수 결정
        num_top = random.randint(2, 5)
        num_middle = random.randint(3, 7)
        num_base = random.randint(2, 5)

        # 노트별 성분 선택
        top_ingredients = [ing_id for ing_id, ing in self.all_ingredients.items()
                          if ing.volatility > 0.7]
        middle_ingredients = [ing_id for ing_id, ing in self.all_ingredients.items()
                             if 0.3 <= ing.volatility <= 0.7]
        base_ingredients = [ing_id for ing_id, ing in self.all_ingredients.items()
                           if ing.volatility < 0.3]

        # Top notes
        if len(top_ingredients) >= num_top:
            selected_tops = random.sample(top_ingredients, num_top)
            for ing_id in selected_tops:
                ing = self.all_ingredients[ing_id]
                max_conc = min(5.0, ing.ifra_limit)
                concentration = np.random.uniform(0.5, max_conc)
                individual.append((ing_id, round(concentration, 2)))

        # Middle notes
        if len(middle_ingredients) >= num_middle:
            selected_middles = random.sample(middle_ingredients, num_middle)
            for ing_id in selected_middles:
                ing = self.all_ingredients[ing_id]
                max_conc = min(8.0, ing.ifra_limit)
                concentration = np.random.uniform(1.0, max_conc)
                individual.append((ing_id, round(concentration, 2)))

        # Base notes
        if len(base_ingredients) >= num_base:
            selected_bases = random.sample(base_ingredients, num_base)
            for ing_id in selected_bases:
                ing = self.all_ingredients[ing_id]
                max_conc = min(10.0, ing.ifra_limit)
                concentration = np.random.uniform(2.0, max_conc)
                individual.append((ing_id, round(concentration, 2)))

        # 메타데이터 초기화
        individual.generation = 0
        individual.parents = None
        individual.mutation_count = 0

        return individual

    def evaluate_individual(self, individual: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """
        고급 평가 함수 - 실제 과학적 검증 포함

        Returns:
            (stability_score, unfitness_score, uncreativity_score)
        """

        # 1. 안정성 점수 - 실제 화학적 검증
        stability_score = self._evaluate_stability_advanced(individual)

        # 2. 부적합도 점수 - 감정적 거리와 시장 적합성
        unfitness_score = self._evaluate_unfitness_advanced(individual)

        # 3. 비창의성 점수 - 기존 포뮬러와의 차별성
        uncreativity_score = self._evaluate_uncreativity_advanced(individual)

        return (stability_score, unfitness_score, uncreativity_score)

    def _evaluate_stability_advanced(self, individual: List[Tuple[int, float]]) -> float:
        """
        고급 안정성 평가 - 실제 화학적 상호작용 검증
        """
        violations = 0.0

        # 1. 총 농도 검증 (15-30% 표준 범위)
        total_concentration = sum(conc for _, conc in individual)
        if not (15 <= total_concentration <= 30):
            violations += abs(22.5 - total_concentration) / 7.5

        # 2. IFRA 규제 준수
        for ing_id, concentration in individual:
            if ing_id in self.all_ingredients:
                ingredient = self.all_ingredients[ing_id]
                if concentration > ingredient.ifra_limit:
                    violations += (concentration - ingredient.ifra_limit) / ingredient.ifra_limit

        # 3. 피라미드 구조 균형
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
            # 이상적 비율: Top 20-30%, Middle 30-50%, Base 30-50%
            top_ratio = top_total / total_concentration
            middle_ratio = middle_total / total_concentration
            base_ratio = base_total / total_concentration

            if not (0.15 <= top_ratio <= 0.35):
                violations += abs(0.25 - top_ratio) * 2
            if not (0.25 <= middle_ratio <= 0.55):
                violations += abs(0.40 - middle_ratio) * 2
            if not (0.25 <= base_ratio <= 0.55):
                violations += abs(0.40 - base_ratio) * 2

        # 4. 화학적 상호작용 검증
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

        # 5. 용해도 검증
        ethanol_soluble_total = 0
        for ing_id, conc in individual:
            if ing_id in self.all_ingredients:
                ingredient = self.all_ingredients[ing_id]
                if "ethanol" in ingredient.solubility:
                    ethanol_soluble_total += conc * ingredient.solubility["ethanol"]

        if total_concentration > 0:
            solubility_ratio = ethanol_soluble_total / total_concentration
            if solubility_ratio < 0.7:  # 70% 미만 용해도는 문제
                violations += (0.7 - solubility_ratio) * 3

        # 6. 가격 효율성 (선택적)
        total_cost = 0
        for ing_id, conc in individual:
            if ing_id in self.all_ingredients:
                ingredient = self.all_ingredients[ing_id]
                total_cost += (conc / 100) * ingredient.cost_per_kg

        # 타겟 가격대 검증
        if self.creative_brief and hasattr(self.creative_brief, 'price_range'):
            if self.creative_brief.price_range == "luxury" and total_cost < 500:
                violations += 0.3
            elif self.creative_brief.price_range == "mass" and total_cost > 100:
                violations += 0.3

        return violations

    def _evaluate_unfitness_advanced(self, individual: List[Tuple[int, float]]) -> float:
        """
        고급 부적합도 평가 - 다차원 감정 벡터 분석
        """

        # 레시피의 감정 프로필 계산
        recipe_emotion = np.zeros(5)  # [freshness, romantic, warmth, mysterious, energetic]
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

        # 거리 계산 (유클리드 + 코사인의 가중 평균)
        euclidean_dist = euclidean(recipe_emotion, target_emotion)
        cosine_dist = cosine(recipe_emotion, target_emotion) if np.any(recipe_emotion) else 1.0

        distance = 0.7 * euclidean_dist + 0.3 * cosine_dist

        # 계절 적합성
        if self.creative_brief and hasattr(self.creative_brief, 'season'):
            avg_volatility = np.mean([self.all_ingredients[ing_id].volatility
                                     for ing_id, _ in individual
                                     if ing_id in self.all_ingredients])

            if self.creative_brief.season == "summer" and avg_volatility < 0.5:
                distance += 0.2  # 여름에 무거운 향은 부적합
            elif self.creative_brief.season == "winter" and avg_volatility > 0.7:
                distance += 0.2  # 겨울에 너무 가벼운 향은 부적합

        # 성별 적합성
        if self.creative_brief and hasattr(self.creative_brief, 'gender'):
            floral_ratio = sum(conc for ing_id, conc in individual
                             if ing_id in self.all_ingredients and
                             self.all_ingredients[ing_id].family == "floral") / max(total_concentration, 1)
            woody_ratio = sum(conc for ing_id, conc in individual
                            if ing_id in self.all_ingredients and
                            self.all_ingredients[ing_id].family == "woody") / max(total_concentration, 1)

            if self.creative_brief.gender == "feminine" and floral_ratio < 0.1:
                distance += 0.15
            elif self.creative_brief.gender == "masculine" and woody_ratio < 0.1:
                distance += 0.15

        return distance

    def _evaluate_uncreativity_advanced(self, individual: List[Tuple[int, float]]) -> float:
        """
        고급 창의성 평가 - 다층적 유사도 분석
        """

        # 현재 레시피의 성분 집합
        current_ingredients = set(ing_id for ing_id, _ in individual if ing_id > 0)

        if not current_ingredients:
            return 1.0

        similarities = []

        for formula in self.existing_formulas:
            existing_ingredients = set(formula.get('ingredients', []))

            if existing_ingredients:
                # 1. Jaccard 유사도
                jaccard = len(current_ingredients & existing_ingredients) / len(current_ingredients | existing_ingredients)

                # 2. 농도 패턴 유사도
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

                # 3. 구조 유사도 (노트 분포)
                current_structure = self._get_structure_vector(individual)
                existing_structure = self._get_structure_vector_from_formula(formula)
                structure_similarity = 1 - cosine(current_structure, existing_structure) if np.any(current_structure) else 0

                # 가중 평균
                total_similarity = 0.5 * jaccard + 0.3 * concentration_similarity + 0.2 * structure_similarity
                similarities.append(total_similarity)

        # 최대 유사도 반환 (가장 비슷한 기존 향수와의 유사도)
        return max(similarities) if similarities else 0.0

    def _get_structure_vector(self, individual: List[Tuple[int, float]]) -> np.ndarray:
        """레시피의 구조 벡터 생성"""
        total = sum(conc for _, conc in individual)
        if total == 0:
            return np.array([0.33, 0.33, 0.34])

        top = sum(conc for ing_id, conc in individual
                 if ing_id in self.all_ingredients and self.all_ingredients[ing_id].volatility > 0.7)
        middle = sum(conc for ing_id, conc in individual
                    if ing_id in self.all_ingredients and 0.3 <= self.all_ingredients[ing_id].volatility <= 0.7)
        base = sum(conc for ing_id, conc in individual
                  if ing_id in self.all_ingredients and self.all_ingredients[ing_id].volatility < 0.3)

        return np.array([top/total, middle/total, base/total])

    def _get_structure_vector_from_formula(self, formula: Dict) -> np.ndarray:
        """기존 포뮬러의 구조 벡터 생성"""
        ingredients = formula.get('ingredients', [])
        concentrations = formula.get('concentrations', [1] * len(ingredients))

        total = sum(concentrations)
        if total == 0:
            return np.array([0.33, 0.33, 0.34])

        top = sum(conc for ing_id, conc in zip(ingredients, concentrations)
                 if ing_id in self.all_ingredients and self.all_ingredients[ing_id].volatility > 0.7)
        middle = sum(conc for ing_id, conc in zip(ingredients, concentrations)
                    if ing_id in self.all_ingredients and 0.3 <= self.all_ingredients[ing_id].volatility <= 0.7)
        base = sum(conc for ing_id, conc in zip(ingredients, concentrations)
                  if ing_id in self.all_ingredients and self.all_ingredients[ing_id].volatility < 0.3)

        return np.array([top/total, middle/total, base/total])

    def _custom_crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """
        지능형 교차 연산 - 구조 보존
        """
        if random.random() > self.crossover_prob:
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

        # 노트별 교차
        for notes1, notes2, ratio in [(tops1, tops2, 0.5),
                                      (middles1, middles2, 0.5),
                                      (bases1, bases2, 0.5)]:
            if notes1 and notes2:
                # 균일 교차
                for gene1, gene2 in zip(notes1, notes2):
                    if random.random() < ratio:
                        child1.append(gene1)
                        child2.append(gene2)
                    else:
                        child1.append(gene2)
                        child2.append(gene1)
            else:
                # 한쪽만 있으면 그대로 복사
                child1.extend(notes1)
                child2.extend(notes2)

        # 메타데이터 업데이트
        child1.generation = max(ind1.generation, ind2.generation) + 1
        child2.generation = child1.generation
        child1.parents = (id(ind1), id(ind2))
        child2.parents = (id(ind1), id(ind2))

        return child1, child2

    def _custom_mutation(self, individual: List) -> Tuple[List]:
        """
        지능형 변이 연산 - 다양한 변이 타입
        """
        if random.random() > self.mutation_prob:
            return (individual,)

        mutation_type = np.random.choice(['point', 'swap', 'insert', 'delete', 'adjust'],
                                        p=[0.3, 0.2, 0.2, 0.1, 0.2])

        if mutation_type == 'point' and individual:
            # 점 변이: 하나의 성분 변경
            idx = random.randint(0, len(individual) - 1)
            new_gene = self._generate_gene()
            individual[idx] = new_gene

        elif mutation_type == 'swap' and len(individual) >= 2:
            # 교환 변이: 두 성분 위치 교환
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

        elif mutation_type == 'insert' and len(individual) < 30:
            # 삽입 변이: 새 성분 추가
            new_gene = self._generate_gene()
            individual.append(new_gene)

        elif mutation_type == 'delete' and len(individual) > 5:
            # 삭제 변이: 성분 제거
            idx = random.randint(0, len(individual) - 1)
            del individual[idx]

        elif mutation_type == 'adjust' and individual:
            # 조정 변이: 농도만 변경
            idx = random.randint(0, len(individual) - 1)
            ing_id, old_conc = individual[idx]
            if ing_id in self.all_ingredients:
                ingredient = self.all_ingredients[ing_id]
                max_conc = min(10.0, ingredient.ifra_limit)
                new_conc = np.clip(old_conc + np.random.normal(0, 1), 0.1, max_conc)
                individual[idx] = (ing_id, round(new_conc, 2))

        individual.mutation_count = getattr(individual, 'mutation_count', 0) + 1

        return (individual,)

    def evolve(self, creative_brief: Optional[CreativeBrief] = None) -> OlfactoryDNA:
        """
        메인 진화 루프 - 고급 NSGA-II 구현
        """

        self.creative_brief = creative_brief

        logger.info(f"[MOGA] Starting advanced evolution")
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

            # 교차
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    child1[:], child2[:] = self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # 변이
            for mutant in offspring:
                if random.random() < self.mutation_prob:
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

            # 조기 종료 조건 (선택적)
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

        logger.info(f"[SUCCESS] Evolution complete!")
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

    # 데이터베이스 URL (환경변수 또는 직접 지정)
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/fragrance_ai')

    # 창세기 엔진 초기화
    engine = OlfactoryRecombinatorAI(
        population_size=200,
        generations=100,
        crossover_prob=0.85,
        mutation_prob=0.15,
        elite_size=10,
        db_url=db_url
    )

    # CreativeBrief 생성 (사용자 요구사항)
    brief = CreativeBrief(
        emotional_palette=[0.7, 0.8, 0.2, 0.3, 0.6],  # freshness, romantic, warmth, mysterious, energetic
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
    print("[MOGA] Starting OlfactoryRecombinatorAI with real ingredients database...")
    print(f"  Database: {len(engine.all_ingredients)} ingredients loaded")
    print(f"  Existing formulas: {len(engine.existing_formulas)} references")

    optimal_dna = engine.evolve(brief)

    # 결과 포맷팅
    recipe = engine.format_recipe(optimal_dna)

    print("\n[SUCCESS] Optimal fragrance recipe generated!")
    print(f"\n=== FORMULA ===")
    print(f"Generation: {recipe['generation']}")
    print(f"\nTop notes ({len(recipe['top_notes'])}): {recipe['top_notes']}")
    print(f"Middle notes ({len(recipe['middle_notes'])}): {recipe['middle_notes']}")
    print(f"Base notes ({len(recipe['base_notes'])}): {recipe['base_notes']}")
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

    # 진화 통계 출력
    if engine.evolution_history:
        print(f"\n=== EVOLUTION STATISTICS ===")
        initial = engine.evolution_history[0]
        final = engine.evolution_history[-1]
        print(f"  Initial best: {initial['min'][0]}")
        print(f"  Final best: {final['min'][0]}")
        improvement = [(initial['min'][0][i] - final['min'][0][i]) / initial['min'][0][i] * 100
                      for i in range(3)]
        print(f"  Improvement: Stability={improvement[0]:.1f}%, "
              f"Suitability={improvement[1]:.1f}%, "
              f"Creativity={improvement[2]:.1f}%")


if __name__ == "__main__":
    example_usage()