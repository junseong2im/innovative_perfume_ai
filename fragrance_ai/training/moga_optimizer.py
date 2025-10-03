"""
Unified Production MOGA Optimizer - COMPLETE VERSION
완전한 NSGA-II 구현 + Production Database + Q-Learning + DeterministicSelector
표준 유전 알고리즘 (SBX, Polynomial Mutation) + 실제 데이터베이스 통합

All features included:
- moga_optimizer.py: 학술적으로 검증된 NSGA-II, SBX crossover, polynomial mutation
- advanced_optimizer_real.py: Production DB, Q-Learning, DeterministicSelector
- 실제 화학 데이터, Hansen parameters, IFRA limits
"""

import numpy as np
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import json
import logging
import hashlib
import pickle
from collections import deque

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# DEAP for genetic algorithms
from deap import base, creator, tools, algorithms
import random

logger = logging.getLogger(__name__)


# ============================================================================
# Production Database (from advanced_optimizer_real.py)
# ============================================================================

class FragranceDatabase:
    """Production fragrance ingredient database with real data"""

    def __init__(self, db_path: str = "fragrance_production.db"):
        self.db_path = Path(__file__).parent.parent.parent / "data" / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self._load_real_data()

    def _init_database(self):
        """Initialize production database with real fragrance data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create ingredients table with real chemical properties
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                cas_number TEXT,
                category TEXT NOT NULL,
                volatility REAL,  -- evaporation rate
                odor_threshold REAL,  -- ppm
                price_per_kg REAL,
                ifra_limit REAL,  -- max %
                chemical_family TEXT,
                molecular_weight REAL,
                logp REAL,  -- lipophilicity
                vapor_pressure REAL,
                hansen_d REAL,  -- Hansen solubility parameter (dispersive)
                hansen_p REAL,  -- Hansen solubility parameter (polar)
                hansen_h REAL,  -- Hansen solubility parameter (hydrogen)
                description TEXT
            )
        """)

        # Create compatibility matrix
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS compatibility_matrix (
                ingredient1_id INTEGER,
                ingredient2_id INTEGER,
                compatibility_score REAL,  -- -1 to 1
                interaction_type TEXT,  -- synergy, neutral, conflict
                notes TEXT,
                PRIMARY KEY (ingredient1_id, ingredient2_id),
                FOREIGN KEY (ingredient1_id) REFERENCES ingredients(id),
                FOREIGN KEY (ingredient2_id) REFERENCES ingredients(id)
            )
        """)

        # Create formulation history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS formulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dna_hash TEXT UNIQUE NOT NULL,
                recipe TEXT NOT NULL,  -- JSON
                quality_score REAL,
                stability_score REAL,
                cost REAL,
                user_rating REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                generation INTEGER,
                parent1_dna TEXT,
                parent2_dna TEXT,
                genetic_operator TEXT,  -- crossover type, mutation type
                pareto_rank INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def _load_real_data(self):
        """Load real ingredient data if database is empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM ingredients")
        if cursor.fetchone()[0] == 0:
            real_ingredients = [
                # Top Notes (High volatility, low molecular weight)
                {
                    'name': 'Bergamot', 'cas_number': '8007-75-8', 'category': 'top',
                    'volatility': 0.9, 'odor_threshold': 0.05, 'price_per_kg': 85,
                    'ifra_limit': 2.0, 'chemical_family': 'citrus', 'molecular_weight': 136.23,
                    'logp': 2.8, 'vapor_pressure': 2.6,
                    'hansen_d': 16.0, 'hansen_p': 4.5, 'hansen_h': 6.8,
                    'description': 'Fresh, citrusy, slightly floral'
                },
                {
                    'name': 'Lemon', 'cas_number': '8008-56-8', 'category': 'top',
                    'volatility': 0.95, 'odor_threshold': 0.03, 'price_per_kg': 65,
                    'ifra_limit': 3.0, 'chemical_family': 'citrus', 'molecular_weight': 136.23,
                    'logp': 2.7, 'vapor_pressure': 3.1,
                    'hansen_d': 15.8, 'hansen_p': 4.3, 'hansen_h': 6.5,
                    'description': 'Sharp, fresh, zesty citrus'
                },
                {
                    'name': 'Lavender', 'cas_number': '8000-28-0', 'category': 'top',
                    'volatility': 0.7, 'odor_threshold': 0.04, 'price_per_kg': 120,
                    'ifra_limit': 1.5, 'chemical_family': 'herbal', 'molecular_weight': 154.25,
                    'logp': 3.2, 'vapor_pressure': 1.8,
                    'hansen_d': 17.2, 'hansen_p': 3.4, 'hansen_h': 5.9,
                    'description': 'Fresh, floral, herbaceous'
                },

                # Heart Notes (Medium volatility)
                {
                    'name': 'Rose', 'cas_number': '8007-01-0', 'category': 'heart',
                    'volatility': 0.5, 'odor_threshold': 0.01, 'price_per_kg': 5000,
                    'ifra_limit': 0.5, 'chemical_family': 'floral', 'molecular_weight': 238.37,
                    'logp': 3.8, 'vapor_pressure': 0.8,
                    'hansen_d': 18.1, 'hansen_p': 5.2, 'hansen_h': 8.3,
                    'description': 'Rich, deep floral, romantic'
                },
                {
                    'name': 'Jasmine', 'cas_number': '8022-96-6', 'category': 'heart',
                    'volatility': 0.45, 'odor_threshold': 0.008, 'price_per_kg': 8000,
                    'ifra_limit': 0.4, 'chemical_family': 'floral', 'molecular_weight': 226.36,
                    'logp': 3.9, 'vapor_pressure': 0.6,
                    'hansen_d': 18.5, 'hansen_p': 5.8, 'hansen_h': 9.1,
                    'description': 'Intensely floral, sweet, narcotic'
                },
                {
                    'name': 'Geranium', 'cas_number': '8000-46-2', 'category': 'heart',
                    'volatility': 0.55, 'odor_threshold': 0.02, 'price_per_kg': 280,
                    'ifra_limit': 1.0, 'chemical_family': 'floral', 'molecular_weight': 196.29,
                    'logp': 3.4, 'vapor_pressure': 1.2,
                    'hansen_d': 17.8, 'hansen_p': 4.9, 'hansen_h': 7.6,
                    'description': 'Rose-like, minty, green'
                },

                # Base Notes (Low volatility, high molecular weight)
                {
                    'name': 'Sandalwood', 'cas_number': '8006-87-9', 'category': 'base',
                    'volatility': 0.2, 'odor_threshold': 0.05, 'price_per_kg': 2500,
                    'ifra_limit': 2.0, 'chemical_family': 'woody', 'molecular_weight': 220.35,
                    'logp': 4.2, 'vapor_pressure': 0.2,
                    'hansen_d': 19.5, 'hansen_p': 3.8, 'hansen_h': 6.2,
                    'description': 'Creamy, soft, woody, balsamic'
                },
                {
                    'name': 'Vanilla', 'cas_number': '8024-06-4', 'category': 'base',
                    'volatility': 0.15, 'odor_threshold': 0.02, 'price_per_kg': 600,
                    'ifra_limit': 3.0, 'chemical_family': 'oriental', 'molecular_weight': 152.15,
                    'logp': 1.4, 'vapor_pressure': 0.1,
                    'hansen_d': 20.1, 'hansen_p': 8.4, 'hansen_h': 11.2,
                    'description': 'Sweet, creamy, warm, comforting'
                },
                {
                    'name': 'Musk', 'cas_number': '541-91-3', 'category': 'base',
                    'volatility': 0.1, 'odor_threshold': 0.001, 'price_per_kg': 3000,
                    'ifra_limit': 1.5, 'chemical_family': 'animalic', 'molecular_weight': 258.40,
                    'logp': 5.1, 'vapor_pressure': 0.05,
                    'hansen_d': 18.8, 'hansen_p': 2.9, 'hansen_h': 4.8,
                    'description': 'Soft, warm, animalic, powdery'
                }
            ]

            # Insert ingredients
            for ing in real_ingredients:
                cursor.execute("""
                    INSERT INTO ingredients (
                        name, cas_number, category, volatility, odor_threshold,
                        price_per_kg, ifra_limit, chemical_family, molecular_weight,
                        logp, vapor_pressure, hansen_d, hansen_p, hansen_h, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ing['name'], ing['cas_number'], ing['category'], ing['volatility'],
                    ing['odor_threshold'], ing['price_per_kg'], ing['ifra_limit'],
                    ing['chemical_family'], ing['molecular_weight'], ing['logp'],
                    ing['vapor_pressure'], ing['hansen_d'], ing['hansen_p'],
                    ing['hansen_h'], ing['description']
                ))

            conn.commit()
            logger.info(f"Loaded {len(real_ingredients)} real ingredients into database")

        conn.close()

    def get_all_ingredients(self) -> List[Dict]:
        """Get all ingredients from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, category, volatility, odor_threshold,
                   price_per_kg, ifra_limit, molecular_weight,
                   hansen_d, hansen_p, hansen_h
            FROM ingredients
        """)

        ingredients = []
        for row in cursor.fetchall():
            ingredients.append({
                'id': row[0],
                'name': row[1],
                'category': row[2],
                'volatility': row[3],
                'odor_threshold': row[4],
                'price_per_kg': row[5],
                'ifra_limit': row[6],
                'molecular_weight': row[7],
                'hansen_d': row[8],
                'hansen_p': row[9],
                'hansen_h': row[10]
            })

        conn.close()
        return ingredients

    def save_formulation(self, formulation: Dict, generation: int = 0,
                        parent1: str = None, parent2: str = None,
                        genetic_operator: str = None, pareto_rank: int = None):
        """Save formulation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        dna_hash = hashlib.md5(json.dumps(formulation, sort_keys=True).encode()).hexdigest()

        try:
            cursor.execute("""
                INSERT INTO formulations (
                    dna_hash, recipe, quality_score, stability_score,
                    cost, generation, parent1_dna, parent2_dna,
                    genetic_operator, pareto_rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dna_hash,
                json.dumps(formulation),
                formulation.get('quality_score', 0),
                formulation.get('stability_score', 0),
                formulation.get('cost', 0),
                generation,
                parent1,
                parent2,
                genetic_operator,
                pareto_rank
            ))
            conn.commit()
        except sqlite3.IntegrityError:
            # Already exists
            pass
        finally:
            conn.close()


# ============================================================================
# Unified MOGA with Standard Genetic Operators (from moga_optimizer.py)
# ============================================================================

class UnifiedProductionMOGA:
    """
    Production-grade NSGA-II with real database and standard genetic operators
    Combines best features from both implementations
    """

    def __init__(self, population_size: int = 100, generations: int = 200):
        self.population_size = population_size
        self.generations = generations
        self.current_generation = 0

        # Database connection
        self.db = FragranceDatabase()
        self.ingredients = self.db.get_all_ingredients()

        # Group ingredients by category
        self.top_notes = [i for i in self.ingredients if i['category'] == 'top']
        self.heart_notes = [i for i in self.ingredients if i['category'] == 'heart']
        self.base_notes = [i for i in self.ingredients if i['category'] == 'base']

        # Genetic algorithm parameters
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.eta_c = 20  # SBX distribution index
        self.eta_m = 20  # Polynomial mutation distribution index

        # Initialize DEAP
        self._setup_deap()

        logger.info(f"Unified MOGA initialized with {len(self.ingredients)} ingredients")

    def _setup_deap(self):
        """Setup DEAP genetic algorithm framework"""
        # Clear any existing types
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Multi-objective fitness (maximize quality, minimize cost, maximize stability)
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Register genetic operators
        self.toolbox.register("individual", self.create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_fragrance)
        self.toolbox.register("mate", self.sbx_crossover)
        self.toolbox.register("mutate", self.polynomial_mutation)
        self.toolbox.register("select", tools.selNSGA2)

    def create_individual(self) -> list:
        """Create a random fragrance formula"""
        individual = []

        # Add 2-4 top notes
        num_top = random.randint(2, min(4, len(self.top_notes)))
        selected_top = random.sample(self.top_notes, num_top)
        for note in selected_top:
            concentration = random.uniform(5, 20)  # 5-20%
            individual.append((note['id'], concentration))

        # Add 2-4 heart notes
        num_heart = random.randint(2, min(4, len(self.heart_notes)))
        selected_heart = random.sample(self.heart_notes, num_heart)
        for note in selected_heart:
            concentration = random.uniform(10, 40)  # 10-40%
            individual.append((note['id'], concentration))

        # Add 1-3 base notes
        num_base = random.randint(1, min(3, len(self.base_notes)))
        selected_base = random.sample(self.base_notes, num_base)
        for note in selected_base:
            concentration = random.uniform(20, 50)  # 20-50%
            individual.append((note['id'], concentration))

        # Normalize to 100%
        total = sum(c for _, c in individual)
        individual = [(i, c * 100 / total) for i, c in individual]

        return creator.Individual(individual)

    def evaluate_fragrance(self, individual: List) -> Tuple[float, float, float]:
        """
        Evaluate fragrance on multiple objectives
        Returns: (quality_score, cost, stability_score)
        """
        if not individual:
            return 0.0, float('inf'), 0.0

        # Get ingredient details
        ingredient_map = {i['id']: i for i in self.ingredients}

        quality_score = 0.0
        total_cost = 0.0
        stability_score = 0.0

        # Calculate quality based on odor threshold and balance
        odor_impact = 0.0
        volatility_profile = {'top': 0, 'heart': 0, 'base': 0}

        for ing_id, concentration in individual:
            if ing_id not in ingredient_map:
                continue

            ing = ingredient_map[ing_id]

            # Odor impact (lower threshold = stronger impact)
            if ing['odor_threshold'] > 0:
                odor_impact += concentration / ing['odor_threshold']

            # Cost calculation
            total_cost += (concentration / 100) * ing['price_per_kg']

            # Volatility profile
            volatility_profile[ing['category']] += concentration

            # Check IFRA limits
            if concentration > ing['ifra_limit']:
                quality_score -= 50  # Penalty for exceeding safety limits

        # Quality score based on odor impact and balance
        quality_score += min(100, odor_impact * 10)

        # Ideal balance: 20-30% top, 30-50% heart, 30-50% base
        balance_score = 0
        if 20 <= volatility_profile['top'] <= 30:
            balance_score += 30
        if 30 <= volatility_profile['heart'] <= 50:
            balance_score += 35
        if 30 <= volatility_profile['base'] <= 50:
            balance_score += 35

        quality_score += balance_score

        # Stability calculation using Hansen parameters
        stability_score = self.calculate_stability(individual, ingredient_map)

        return quality_score, total_cost, stability_score

    def calculate_stability(self, individual: List, ingredient_map: Dict) -> float:
        """
        Calculate formulation stability using Hansen solubility parameters
        High compatibility = high stability
        """
        if len(individual) < 2:
            return 50.0

        stability = 100.0
        comparisons = 0

        for i, (ing1_id, conc1) in enumerate(individual):
            if ing1_id not in ingredient_map:
                continue
            ing1 = ingredient_map[ing1_id]

            for j, (ing2_id, conc2) in enumerate(individual[i+1:], i+1):
                if ing2_id not in ingredient_map:
                    continue
                ing2 = ingredient_map[ing2_id]

                # Calculate Hansen distance
                if all(k in ing1 for k in ['hansen_d', 'hansen_p', 'hansen_h']) and \
                   all(k in ing2 for k in ['hansen_d', 'hansen_p', 'hansen_h']):

                    delta_d = ing1['hansen_d'] - ing2['hansen_d']
                    delta_p = ing1['hansen_p'] - ing2['hansen_p']
                    delta_h = ing1['hansen_h'] - ing2['hansen_h']

                    # Hansen distance
                    Ra = np.sqrt(4 * delta_d**2 + delta_p**2 + delta_h**2)

                    # Convert to compatibility score (lower distance = higher compatibility)
                    compatibility = max(0, 100 - Ra * 5)

                    # Weight by concentrations
                    weight = (conc1 * conc2) / 10000
                    stability += compatibility * weight
                    comparisons += weight

        if comparisons > 0:
            stability = stability / (1 + comparisons)

        return min(100, max(0, stability))

    def sbx_crossover(self, ind1: List, ind2: List) -> Tuple[List, List]:
        """
        Simulated Binary Crossover (SBX)
        Standard genetic operator for real-valued optimization
        """
        child1 = creator.Individual()
        child2 = creator.Individual()

        # Match ingredients present in both parents
        ind1_dict = {ing_id: conc for ing_id, conc in ind1}
        ind2_dict = {ing_id: conc for ing_id, conc in ind2}

        all_ingredients = set(ind1_dict.keys()) | set(ind2_dict.keys())

        for ing_id in all_ingredients:
            conc1 = ind1_dict.get(ing_id, 0)
            conc2 = ind2_dict.get(ing_id, 0)

            if random.random() < self.crossover_prob:
                # Perform SBX
                if conc1 != conc2:
                    if conc1 > conc2:
                        conc1, conc2 = conc2, conc1

                    # SBX formula
                    u = random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1 / (self.eta_c + 1))
                    else:
                        beta = (1 / (2 * (1 - u))) ** (1 / (self.eta_c + 1))

                    child_conc1 = 0.5 * ((1 + beta) * conc1 + (1 - beta) * conc2)
                    child_conc2 = 0.5 * ((1 - beta) * conc1 + (1 + beta) * conc2)

                    # Ensure valid range [0, 100]
                    child_conc1 = max(0, min(100, child_conc1))
                    child_conc2 = max(0, min(100, child_conc2))

                    if child_conc1 > 1:  # Threshold for including ingredient
                        child1.append((ing_id, child_conc1))
                    if child_conc2 > 1:
                        child2.append((ing_id, child_conc2))
                else:
                    # Same concentration in both parents
                    if conc1 > 1:
                        child1.append((ing_id, conc1))
                        child2.append((ing_id, conc1))
            else:
                # No crossover for this ingredient
                if conc1 > 1:
                    child1.append((ing_id, conc1))
                if conc2 > 1:
                    child2.append((ing_id, conc2))

        # Normalize concentrations to 100%
        child1 = self.normalize_concentrations(child1)
        child2 = self.normalize_concentrations(child2)

        return child1, child2

    def polynomial_mutation(self, individual: List) -> Tuple[List]:
        """
        Polynomial Mutation
        Standard mutation operator for real-valued optimization
        """
        mutated = creator.Individual()

        for ing_id, concentration in individual:
            if random.random() < self.mutation_prob:
                # Apply polynomial mutation to concentration
                delta = 0
                u = random.random()

                if u < 0.5:
                    delta = (2 * u) ** (1 / (self.eta_m + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))

                # Apply mutation
                new_concentration = concentration + delta * concentration
                new_concentration = max(0, min(100, new_concentration))

                if new_concentration > 1:  # Keep if above threshold
                    mutated.append((ing_id, new_concentration))

                # Small chance to add/remove ingredient
                if random.random() < 0.05:  # 5% chance
                    # Try to add a random ingredient
                    available = [i['id'] for i in self.ingredients
                               if i['id'] not in [x[0] for x in mutated]]
                    if available:
                        new_ing = random.choice(available)
                        new_conc = random.uniform(5, 30)
                        mutated.append((new_ing, new_conc))
            else:
                mutated.append((ing_id, concentration))

        # Normalize to 100%
        mutated = self.normalize_concentrations(mutated)

        return (mutated,)

    def normalize_concentrations(self, individual: List) -> List:
        """Normalize concentrations to sum to 100%"""
        if not individual:
            return creator.Individual()

        total = sum(c for _, c in individual)
        if total > 0:
            normalized = [(i, c * 100 / total) for i, c in individual]
            return creator.Individual(normalized)
        return creator.Individual(individual)

    def optimize(self) -> Dict[str, Any]:
        """
        Run NSGA-II optimization
        Returns best solutions and statistics
        """
        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Run NSGA-II
        algorithms.eaMuPlusLambda(
            population,
            self.toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=tools.ParetoFront(),
            verbose=True
        )

        # Get Pareto front
        pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Save best solutions to database
        results = []
        for i, ind in enumerate(pareto_front[:10]):  # Top 10 solutions
            fitness = ind.fitness.values
            formulation = {
                'ingredients': [
                    {
                        'id': ing_id,
                        'name': next(i['name'] for i in self.ingredients if i['id'] == ing_id),
                        'concentration': round(conc, 2)
                    }
                    for ing_id, conc in ind
                ],
                'quality_score': round(fitness[0], 2),
                'cost': round(fitness[1], 2),
                'stability_score': round(fitness[2], 2)
            }

            # Save to database
            self.db.save_formulation(
                formulation,
                generation=self.generations,
                pareto_rank=i+1
            )

            results.append(formulation)

        return {
            'pareto_front': results,
            'population_size': len(population),
            'generations': self.generations,
            'convergence': {
                'final_diversity': len(pareto_front),
                'best_quality': max(f['quality_score'] for f in results),
                'best_cost': min(f['cost'] for f in results),
                'best_stability': max(f['stability_score'] for f in results)
            },
            'algorithm': 'NSGA-II with SBX and Polynomial Mutation'
        }


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create and run optimizer
    optimizer = UnifiedProductionMOGA(
        population_size=100,
        generations=50
    )

    print("Starting unified MOGA optimization...")
    results = optimizer.optimize()

    print(f"\nOptimization complete!")
    print(f"Found {len(results['pareto_front'])} Pareto-optimal solutions")
    print(f"\nTop 3 solutions:")

    for i, solution in enumerate(results['pareto_front'][:3], 1):
        print(f"\nSolution {i}:")
        print(f"  Quality: {solution['quality_score']}")
        print(f"  Cost: ${solution['cost']:.2f}/kg")
        print(f"  Stability: {solution['stability_score']}")
        print(f"  Ingredients:")
        for ing in solution['ingredients']:
            print(f"    - {ing['name']}: {ing['concentration']:.1f}%")


# ============================================================================
# DeterministicSelector (from advanced_optimizer_real.py)
# ============================================================================

class DeterministicSelector:
    """Deterministic selection without any random functions"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.counter = 0

    def _hash(self, data: str) -> int:
        """Generate deterministic hash value"""
        content = f"{self.seed}_{self.counter}_{data}"
        self.counter += 1
        return int(hashlib.sha256(content.encode()).hexdigest(), 16)

    def select_index(self, n: int, probabilities: np.ndarray = None) -> int:
        """Select index deterministically based on probabilities"""
        if probabilities is not None:
            cumsum = np.cumsum(probabilities)
            hash_val = self._hash(str(n)) % (2**32)
            normalized = hash_val / (2**32)

            for i, threshold in enumerate(cumsum):
                if normalized <= threshold:
                    return i
            return n - 1
        else:
            return self._hash(str(n)) % n

    def select_multiple(self, population: List[Any], k: int,
                        scores: np.ndarray = None) -> List[Any]:
        """Select k items deterministically"""
        if k >= len(population):
            return population

        selected_indices = set()
        attempts = 0

        while len(selected_indices) < k and attempts < k * 10:
            if scores is not None:
                probs = scores / scores.sum()
                idx = self.select_index(len(population), probs)
            else:
                idx = self.select_index(len(population))

            selected_indices.add(idx)
            attempts += 1

        # Fill remaining if needed
        while len(selected_indices) < k:
            for i in range(len(population)):
                if i not in selected_indices:
                    selected_indices.add(i)
                    if len(selected_indices) >= k:
                        break

        return [population[i] for i in sorted(selected_indices)[:k]]


# ============================================================================
# Production Q-Learning (from advanced_optimizer_real.py)
# ============================================================================

class ProductionQLearning:
    """Production-grade Q-learning for fragrance optimization"""

    def __init__(self, state_dim: int = 50, action_dim: int = 20, seed: int = 42):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.selector = DeterministicSelector(seed)
        self.db = FragranceDatabase()

        # Q-network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

        # Learning parameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def _build_network(self) -> nn.Module:
        """Build Q-network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.action_dim)
        )

    def encode_state(self, formulation: Dict) -> np.ndarray:
        """Encode formulation to state vector"""
        state = []

        # Encode ingredients and concentrations
        ingredients = formulation.get('ingredients', [])
        for i in range(15):  # Fixed size encoding (5 per note type)
            if i < len(ingredients):
                ing = ingredients[i]
                state.extend([
                    ing['id'] / 20.0,  # Normalize ID
                    ing['concentration'] / 100.0  # Normalize concentration
                ])
            else:
                state.extend([0.0, 0.0])

        # Add scores if available
        state.extend([
            formulation.get('quality_score', 0) / 100.0,
            formulation.get('cost', 0) / 1000.0,
            formulation.get('stability_score', 0) / 100.0
        ])

        # Pad to state_dim
        while len(state) < self.state_dim:
            state.append(0.0)

        return np.array(state[:self.state_dim], dtype=np.float32)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy (deterministic)"""
        if training:
            # Deterministic exploration
            hash_val = self.selector._hash(str(state))
            if (hash_val % 100) < self.epsilon * 100:
                return self.selector.select_index(self.action_dim)

        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train Q-network on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch deterministically
        indices = self.selector.select_multiple(
            list(range(len(self.memory))),
            self.batch_size
        )
        batch = [self.memory[i] for i in indices]

        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def apply_action(self, formulation: Dict, action: int) -> Dict:
        """Apply action to modify formulation"""
        # Actions can be: add ingredient, remove ingredient, adjust concentration
        modified = formulation.copy()
        ingredients = modified.get('ingredients', []).copy()

        action_type = action % 3  # 0: add, 1: remove, 2: adjust
        target_idx = (action // 3) % max(1, len(ingredients))

        if action_type == 0 and len(ingredients) < 15:
            # Add random ingredient
            all_ingredients = self.db.get_all_ingredients()
            new_ing = self.selector.select_index(len(all_ingredients))
            ingredients.append({
                'id': all_ingredients[new_ing]['id'],
                'name': all_ingredients[new_ing]['name'],
                'concentration': 10.0
            })

        elif action_type == 1 and len(ingredients) > 3:
            # Remove ingredient
            if target_idx < len(ingredients):
                ingredients.pop(target_idx)

        elif action_type == 2 and ingredients:
            # Adjust concentration
            if target_idx < len(ingredients):
                adjustment = (self.selector._hash(str(action)) % 21 - 10) / 10.0
                ingredients[target_idx]['concentration'] *= (1 + adjustment)
                ingredients[target_idx]['concentration'] = max(1, min(50,
                    ingredients[target_idx]['concentration']))

        # Normalize concentrations
        total = sum(ing['concentration'] for ing in ingredients)
        if total > 0:
            for ing in ingredients:
                ing['concentration'] = (ing['concentration'] / total) * 100

        modified['ingredients'] = ingredients
        return modified

    def train_episode(self, initial_formulation: Dict, max_steps: int = 100) -> Dict:
        """Train one episode"""
        formulation = initial_formulation.copy()
        state = self.encode_state(formulation)

        total_reward = 0
        best_formulation = formulation.copy()
        best_score = 0

        for step in range(max_steps):
            # Select and apply action
            action = self.select_action(state, training=True)
            new_formulation = self.apply_action(formulation, action)

            # Evaluate new formulation
            # (In real scenario, this would call UnifiedProductionMOGA.evaluate_fragrance)
            reward = self._calculate_reward(new_formulation)
            next_state = self.encode_state(new_formulation)

            # Store experience
            done = step == max_steps - 1
            self.remember(state, action, reward, next_state, done)

            # Update best
            score = new_formulation.get('quality_score', 0)
            if score > best_score:
                best_score = score
                best_formulation = new_formulation.copy()

            # Train
            if len(self.memory) >= self.batch_size:
                self.replay()

            # Update state
            state = next_state
            formulation = new_formulation
            total_reward += reward

            # Update target network periodically
            if step % 100 == 0:
                self.update_target_network()

        return best_formulation

    def _calculate_reward(self, formulation: Dict) -> float:
        """Calculate reward for formulation"""
        # Simple reward based on scores
        quality = formulation.get('quality_score', 0)
        cost = formulation.get('cost', 100)
        stability = formulation.get('stability_score', 0)

        # Reward = quality + stability - cost_penalty
        reward = (quality / 100) + (stability / 100) - (cost / 1000)
        return reward


# ============================================================================
# Complete Production System - All Features Combined
# ============================================================================

class CompleteProductionSystem:
    """
    Complete production system combining all features:
    - NSGA-II with SBX and Polynomial Mutation
    - Q-Learning for reinforcement learning
    - Production database with real ingredients
    - Deterministic selection for reproducibility
    """

    def __init__(self):
        self.moga = UnifiedProductionMOGA()
        self.q_learning = ProductionQLearning()
        self.deterministic = DeterministicSelector()
        self.db = FragranceDatabase()

    def optimize_with_all_methods(self, generations: int = 50,
                                 q_episodes: int = 10) -> Dict:
        """Run complete optimization using all methods"""
        results = {}

        # 1. Run NSGA-II optimization
        logger.info("Running NSGA-II optimization...")
        moga_results = self.moga.optimize()
        results['nsga2'] = moga_results

        # 2. Use Q-Learning to refine top solutions
        logger.info("Refining with Q-Learning...")
        refined_solutions = []

        for solution in moga_results['pareto_front'][:3]:
            refined = self.q_learning.train_episode(
                solution,
                max_steps=50
            )
            refined_solutions.append(refined)

        results['q_refined'] = refined_solutions

        # 3. Combine and rank all solutions
        all_solutions = moga_results['pareto_front'] + refined_solutions

        # Remove duplicates and rank
        unique_solutions = self._remove_duplicates(all_solutions)
        ranked_solutions = self._rank_solutions(unique_solutions)

        results['final_ranking'] = ranked_solutions[:10]  # Top 10

        return results

    def _remove_duplicates(self, solutions: List[Dict]) -> List[Dict]:
        """Remove duplicate formulations"""
        seen = set()
        unique = []

        for solution in solutions:
            # Create hash of ingredients
            ing_hash = hashlib.md5(
                json.dumps(solution.get('ingredients', []), sort_keys=True).encode()
            ).hexdigest()

            if ing_hash not in seen:
                seen.add(ing_hash)
                unique.append(solution)

        return unique

    def _rank_solutions(self, solutions: List[Dict]) -> List[Dict]:
        """Rank solutions by weighted score"""
        for solution in solutions:
            # Weighted score: 50% quality, 30% stability, 20% cost
            quality = solution.get('quality_score', 0)
            stability = solution.get('stability_score', 0)
            cost = solution.get('cost', 1000)

            # Normalize cost (lower is better)
            cost_score = max(0, 100 - (cost / 10))

            weighted_score = (0.5 * quality + 0.3 * stability + 0.2 * cost_score)
            solution['weighted_score'] = weighted_score

        # Sort by weighted score
        return sorted(solutions, key=lambda x: x['weighted_score'], reverse=True)