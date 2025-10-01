"""
Enhanced MOGA Optimizer with Full Evaluate Function Implementation
Integrates ValidatorTool, proper vector distance calculations, and similarity metrics
"""

import random
import json
import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.distance import euclidean, cosine, jaccard
from sklearn.metrics.pairwise import cosine_similarity

# DEAP imports
from deap import base, creator, tools
from deap.tools import HallOfFame, ParetoFront

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fragrance_ai.tools.unified_tools import UnifiedValidatorTool
except ImportError:
    UnifiedValidatorTool = None

try:
    from fragrance_ai.tools.validator_tool import ScientificValidator, NotesComposition
except ImportError:
    ScientificValidator = None
    NotesComposition = None

logger = logging.getLogger(__name__)


@dataclass
class CreativeBrief:
    """User's creative requirements"""
    emotional_palette: List[float]  # Emotion vector [happy, calm, fresh, romantic, etc.]
    fragrance_family: str  # floral, woody, fresh, oriental, etc.
    mood: str  # romantic, energetic, calm, mysterious, etc.
    intensity: float  # 0-1 scale
    season: str  # spring, summer, fall, winter, all
    gender: str  # masculine, feminine, unisex


class EnhancedMOGAOptimizer:
    """Enhanced Multi-Objective Genetic Algorithm Optimizer for Fragrance Creation"""

    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2):

        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # Initialize tools (if available)
        self.validator_tool = UnifiedValidatorTool() if UnifiedValidatorTool else None
        self.scientific_validator = ScientificValidator() if ScientificValidator else None

        # Load databases
        self.notes_db = self._load_notes_database()
        self.existing_fragrances = self._load_existing_fragrances()
        self.blending_rules = self._load_blending_rules()

        # Creative brief (set during optimization)
        self.creative_brief = None

        # Setup DEAP framework
        self._setup_deap_framework()

    def _load_notes_database(self) -> Dict[int, Dict]:
        """Load fragrance notes database"""
        try:
            notes_path = Path("data/comprehensive_fragrance_notes_database.json")
            if notes_path.exists():
                with open(notes_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert to indexed format
                    notes = {}
                    for idx, (name, info) in enumerate(data.get('notes', {}).items(), 1):
                        notes[idx] = {
                            'name': name,
                            'family': info.get('family', 'unknown'),
                            'volatility': info.get('volatility', 0.5),
                            'intensity': info.get('intensity', 0.5),
                            'emotion_vector': info.get('emotional_profile', [0.5, 0.5, 0.5]),
                            'accords': info.get('accords_well_with', []),
                            'conflicts': info.get('conflicts_with', []),
                            'ifra_limit': info.get('ifra_limit', 10.0)
                        }
                    return notes
        except Exception as e:
            logger.warning(f"Error loading notes database: {e}. Using sample data")

        # Fallback sample data
        return {
            1: {"name": "Bergamot", "family": "citrus", "volatility": 0.9, "intensity": 0.7,
                "emotion_vector": [0.8, 0.6, 0.9], "ifra_limit": 5.0},
            2: {"name": "Lemon", "family": "citrus", "volatility": 0.95, "intensity": 0.8,
                "emotion_vector": [0.9, 0.5, 1.0], "ifra_limit": 3.0},
            3: {"name": "Rose", "family": "floral", "volatility": 0.6, "intensity": 0.8,
                "emotion_vector": [0.7, 0.9, 0.4], "ifra_limit": 10.0},
            4: {"name": "Jasmine", "family": "floral", "volatility": 0.5, "intensity": 0.9,
                "emotion_vector": [0.6, 0.8, 0.3], "ifra_limit": 8.0},
            5: {"name": "Lavender", "family": "aromatic", "volatility": 0.7, "intensity": 0.6,
                "emotion_vector": [0.5, 0.9, 0.7], "ifra_limit": 5.0},
            6: {"name": "Sandalwood", "family": "woody", "volatility": 0.2, "intensity": 0.6,
                "emotion_vector": [0.3, 0.8, 0.2], "ifra_limit": 10.0},
            7: {"name": "Cedarwood", "family": "woody", "volatility": 0.3, "intensity": 0.7,
                "emotion_vector": [0.4, 0.7, 0.3], "ifra_limit": 10.0},
            8: {"name": "Vanilla", "family": "gourmand", "volatility": 0.1, "intensity": 0.9,
                "emotion_vector": [0.8, 0.8, 0.1], "ifra_limit": 7.0},
            9: {"name": "Amber", "family": "oriental", "volatility": 0.15, "intensity": 0.85,
                "emotion_vector": [0.6, 0.7, 0.2], "ifra_limit": 5.0},
            10: {"name": "Musk", "family": "animalic", "volatility": 0.05, "intensity": 0.9,
                 "emotion_vector": [0.7, 0.6, 0.1], "ifra_limit": 3.0}
        }

    def _load_existing_fragrances(self) -> List[Dict]:
        """Load existing fragrance recipes database"""
        try:
            recipes_path = Path("data/fragrance_recipes_database.json")
            if recipes_path.exists():
                with open(recipes_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('recipes', [])
        except Exception as e:
            logger.warning(f"Error loading recipes: {e}. Using sample data")

        # Fallback sample data
        return [
            {"name": "Classic Floral", "notes": [3, 4, 8], "percentages": [30, 40, 30]},
            {"name": "Fresh Citrus", "notes": [1, 2, 6], "percentages": [40, 30, 30]},
            {"name": "Oriental Spice", "notes": [4, 8, 9], "percentages": [25, 35, 40]}
        ]

    def _load_blending_rules(self) -> Dict[str, Any]:
        """Load perfume blending rules"""
        try:
            rules_path = Path("data/perfume_blending_rules.json")
            if rules_path.exists():
                with open(rules_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading blending rules: {e}. Using defaults")

        # Default rules
        return {
            "concentration_limits": {
                "total": {"min": 15, "max": 25},
                "top": {"min": 20, "max": 30},
                "middle": {"min": 30, "max": 50},
                "base": {"min": 20, "max": 40}
            },
            "max_ingredients": 20,
            "min_ingredients": 5,
            "forbidden_combinations": [
                ["citrus", "vanilla"],  # Example forbidden combo
            ]
        }

    def _setup_deap_framework(self):
        """Setup DEAP genetic algorithm framework"""

        # Clean up existing classes if they exist
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Define fitness class - 3 objectives to minimize
        # (stability_violations, unfitness_distance, uncreativity_similarity)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0))

        # Define individual class - fragrance recipe as list of genes
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Setup toolbox
        self.toolbox = base.Toolbox()

        # Gene generation - (note_id, percentage) tuple
        self.toolbox.register("gene", self._generate_gene)

        # Individual creation - 15 notes standard
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.gene, n=15)

        # Population creation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register genetic operators
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._custom_mutation)
        self.toolbox.register("select", tools.selNSGA2)

    def _generate_gene(self) -> Tuple[int, float]:
        """Generate a single gene (note_id, percentage)"""

        # If creative brief exists, bias towards matching notes
        if self.creative_brief and self.creative_brief.emotional_palette:
            weights = []
            for note_id, note_data in self.notes_db.items():
                # Calculate similarity to emotional palette
                try:
                    similarity = 1.0 - cosine(
                        note_data.get("emotion_vector", [0.5, 0.5, 0.5]),
                        self.creative_brief.emotional_palette[:3]
                    )
                    weights.append(max(similarity, 0.1))  # Ensure positive weight
                except:
                    weights.append(0.5)  # Default weight if calculation fails

            # Weighted random selection
            if weights and len(weights) > 0:
                note_ids = list(self.notes_db.keys())
                note_id = random.choices(note_ids, weights=weights)[0]
            else:
                note_id = random.randint(1, len(self.notes_db))
        else:
            # Random selection
            note_id = random.randint(1, len(self.notes_db))

        # Random percentage between 0.1% and 10%
        percentage = random.uniform(0.1, 10.0)

        return (note_id, percentage)

    def evaluate(self, individual: List[Tuple[int, float]]) -> Tuple[float, float, float]:
        """
        Enhanced evaluation function with full integration

        Returns:
            (stability_score, unfitness_score, uncreativity_score)
            All scores are to be minimized (lower is better)
        """

        # 1. STABILITY SCORE - Using ValidatorTool
        stability_score = self._evaluate_stability_with_validator(individual)

        # 2. UNFITNESS SCORE - Distance from creative brief
        unfitness_score = self._evaluate_unfitness_enhanced(individual)

        # 3. UNCREATIVITY SCORE - Similarity to existing fragrances
        uncreativity_score = self._evaluate_uncreativity_enhanced(individual)

        return (stability_score, unfitness_score, uncreativity_score)

    def _evaluate_stability_with_validator(self, individual: List[Tuple[int, float]]) -> float:
        """
        Evaluate stability using ValidatorTool
        Returns a score where 0 is perfect stability, higher values indicate more violations
        """

        violations = 0.0

        # Convert individual to recipe format
        recipe = self._individual_to_recipe(individual)

        # Use UnifiedValidatorTool for rule-based validation if available
        if self.validator_tool:
            try:
                import asyncio

                # Run async validation in sync context
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                validation_result = loop.run_until_complete(
                    self.validator_tool.execute(recipe, validation_level="strict")
                )
                loop.close()

                # Count violations from validation results
                for validation in validation_result.get("validations", []):
                    if not validation.get("passed", True):
                        violations += validation.get("severity", 1.0)

            except Exception as e:
                logger.debug(f"Validator tool error: {e}, using fallback")
                # Fallback to basic validation
                violations = self._basic_stability_check(individual)
        else:
            # No validator tool available, use basic check
            violations = self._basic_stability_check(individual)

        # Use ScientificValidator for deep learning validation if available
        if self.scientific_validator and NotesComposition:
            try:
                composition = self._individual_to_composition(individual)
                scientific_result = self.scientific_validator.validate(composition)

                # Add penalties based on scientific scores
                if scientific_result.harmony_score < 7.0:
                    violations += (7.0 - scientific_result.harmony_score) * 0.5
                if scientific_result.stability_score < 7.0:
                    violations += (7.0 - scientific_result.stability_score) * 0.5
                if not scientific_result.is_valid:
                    violations += 5.0

            except Exception as e:
                logger.debug(f"Scientific validator error: {e}")

        return violations

    def _evaluate_unfitness_enhanced(self, individual: List[Tuple[int, float]]) -> float:
        """
        Enhanced unfitness calculation using multiple distance metrics
        Returns distance from creative brief (0 is perfect match)
        """

        if not self.creative_brief or not self.creative_brief.emotional_palette:
            return 1.0  # Maximum unfitness if no brief

        # Calculate recipe's emotional profile
        recipe_emotion = self._calculate_emotional_profile(individual)
        target_emotion = self.creative_brief.emotional_palette[:3]

        # Check for valid vectors
        if not any(recipe_emotion) or not any(target_emotion):
            return 1.0  # Maximum unfitness if vectors are invalid

        # Use multiple distance metrics
        try:
            euclidean_dist = euclidean(recipe_emotion, target_emotion)
            cosine_dist = cosine(recipe_emotion, target_emotion) if any(recipe_emotion) and any(target_emotion) else 1.0
        except Exception:
            euclidean_dist = 1.0
            cosine_dist = 1.0

        # Calculate family match score
        family_score = self._calculate_family_match(individual)

        # Calculate seasonal appropriateness
        seasonal_score = self._calculate_seasonal_match(individual)

        # Weighted combination of distances
        unfitness = (
            euclidean_dist * 0.4 +
            cosine_dist * 0.3 +
            (1.0 - family_score) * 0.2 +
            (1.0 - seasonal_score) * 0.1
        )

        return unfitness

    def _evaluate_uncreativity_enhanced(self, individual: List[Tuple[int, float]]) -> float:
        """
        Enhanced uncreativity scoring using advanced similarity metrics
        Returns similarity to existing fragrances (0 is completely unique)
        """

        # Get note set from individual
        current_notes = set(note_id for note_id, _ in individual if note_id > 0)

        if not current_notes:
            return 1.0

        max_similarity = 0.0

        for existing in self.existing_fragrances:
            existing_notes = set(existing.get("notes", []))

            if not existing_notes:
                continue

            # 1. Jaccard similarity for note overlap
            jaccard_sim = len(current_notes & existing_notes) / len(current_notes | existing_notes)

            # 2. Weighted similarity considering percentages
            weighted_sim = self._calculate_weighted_similarity(individual, existing)

            # 3. Family similarity
            family_sim = self._calculate_family_similarity(individual, existing_notes)

            # Combined similarity
            combined_sim = (jaccard_sim * 0.5 + weighted_sim * 0.3 + family_sim * 0.2)
            max_similarity = max(max_similarity, combined_sim)

        return max_similarity

    def _individual_to_recipe(self, individual: List[Tuple[int, float]]) -> Dict:
        """Convert individual to recipe format for validator"""

        top_notes = []
        middle_notes = []
        base_notes = []

        for note_id, percentage in individual:
            if note_id in self.notes_db and percentage > 0:
                note = self.notes_db[note_id]
                note_dict = {note['name']: percentage}

                if note['volatility'] > 0.7:
                    top_notes.append(note_dict)
                elif note['volatility'] > 0.3:
                    middle_notes.append(note_dict)
                else:
                    base_notes.append(note_dict)

        total = sum(p for _, p in individual)

        return {
            "notes": {
                "top": top_notes,
                "middle": middle_notes,
                "base": base_notes
            },
            "concentrations": {
                "top": sum(sum(n.values()) for n in top_notes),
                "middle": sum(sum(n.values()) for n in middle_notes),
                "base": sum(sum(n.values()) for n in base_notes),
                "total": total
            }
        }

    def _individual_to_composition(self, individual: List[Tuple[int, float]]):
        """Convert individual to NotesComposition for scientific validator"""

        recipe = self._individual_to_recipe(individual)

        return NotesComposition(
            top_notes=recipe["notes"]["top"],
            heart_notes=recipe["notes"]["middle"],
            base_notes=recipe["notes"]["base"],
            total_ingredients=len([i for i in individual if i[1] > 0])
        )

    def _basic_stability_check(self, individual: List[Tuple[int, float]]) -> float:
        """Fallback stability check using basic rules"""

        violations = 0.0

        # Total concentration check
        total = sum(p for _, p in individual)
        target = 20.0
        if not (15 <= total <= 25):
            violations += abs(total - target) / 5.0

        # Balance check
        top_total = sum(p for n, p in individual
                       if n in self.notes_db and self.notes_db[n]['volatility'] > 0.7)
        middle_total = sum(p for n, p in individual
                          if n in self.notes_db and 0.3 <= self.notes_db[n]['volatility'] <= 0.7)
        base_total = sum(p for n, p in individual
                        if n in self.notes_db and self.notes_db[n]['volatility'] < 0.3)

        # Check pyramid structure
        if total > 0:
            top_ratio = top_total / total
            middle_ratio = middle_total / total
            base_ratio = base_total / total

            # Ideal: top 20-30%, middle 30-50%, base 20-40%
            if not (0.2 <= top_ratio <= 0.3):
                violations += abs(0.25 - top_ratio) * 2
            if not (0.3 <= middle_ratio <= 0.5):
                violations += abs(0.4 - middle_ratio) * 2
            if not (0.2 <= base_ratio <= 0.4):
                violations += abs(0.3 - base_ratio) * 2

        # IFRA compliance check
        for note_id, percentage in individual:
            if note_id in self.notes_db:
                ifra_limit = self.notes_db[note_id].get('ifra_limit', 10.0)
                if percentage > ifra_limit:
                    violations += (percentage - ifra_limit) * 0.5

        return violations

    def _calculate_emotional_profile(self, individual: List[Tuple[int, float]]) -> List[float]:
        """Calculate weighted emotional profile of the recipe"""

        profile = [0.0, 0.0, 0.0]
        total_weight = sum(p for _, p in individual)

        if total_weight > 0:
            for note_id, percentage in individual:
                if note_id in self.notes_db:
                    emotion = self.notes_db[note_id].get('emotion_vector', [0.5, 0.5, 0.5])
                    weight = percentage / total_weight
                    for i in range(min(3, len(emotion))):
                        profile[i] += emotion[i] * weight

        return profile

    def _calculate_family_match(self, individual: List[Tuple[int, float]]) -> float:
        """Calculate how well the recipe matches the requested fragrance family"""

        if not self.creative_brief:
            return 0.5

        target_family = self.creative_brief.fragrance_family
        family_percentages = {}
        total = sum(p for _, p in individual)

        if total > 0:
            for note_id, percentage in individual:
                if note_id in self.notes_db:
                    family = self.notes_db[note_id].get('family', 'unknown')
                    family_percentages[family] = family_percentages.get(family, 0) + percentage

            # Get percentage of target family
            target_percentage = family_percentages.get(target_family, 0) / total
            return min(target_percentage * 2, 1.0)  # Scale up, cap at 1.0

        return 0.0

    def _calculate_seasonal_match(self, individual: List[Tuple[int, float]]) -> float:
        """Calculate seasonal appropriateness"""

        if not self.creative_brief:
            return 0.5

        season = self.creative_brief.season

        # Seasonal note preferences
        seasonal_preferences = {
            "spring": ["floral", "green", "citrus"],
            "summer": ["citrus", "aquatic", "fruity"],
            "fall": ["woody", "spicy", "gourmand"],
            "winter": ["oriental", "woody", "gourmand"],
            "all": []  # No preference
        }

        if season == "all":
            return 1.0

        preferred_families = seasonal_preferences.get(season, [])
        if not preferred_families:
            return 0.5

        match_score = 0.0
        total = sum(p for _, p in individual)

        if total > 0:
            for note_id, percentage in individual:
                if note_id in self.notes_db:
                    family = self.notes_db[note_id].get('family', 'unknown')
                    if family in preferred_families:
                        match_score += percentage / total

        return match_score

    def _calculate_weighted_similarity(self, individual: List[Tuple[int, float]],
                                      existing: Dict) -> float:
        """Calculate weighted similarity considering percentages"""

        existing_notes = existing.get("notes", [])
        existing_percentages = existing.get("percentages", [])

        if not existing_notes or not existing_percentages:
            return 0.0

        # Create percentage dictionaries
        current_dict = {n: p for n, p in individual if n > 0}
        existing_dict = dict(zip(existing_notes, existing_percentages))

        # Find common notes
        common_notes = set(current_dict.keys()) & set(existing_dict.keys())

        if not common_notes:
            return 0.0

        # Calculate percentage difference for common notes
        similarity = 0.0
        for note in common_notes:
            diff = abs(current_dict[note] - existing_dict[note])
            similarity += 1.0 / (1.0 + diff)  # Similarity decreases with difference

        return similarity / max(len(current_dict), len(existing_dict))

    def _calculate_family_similarity(self, individual: List[Tuple[int, float]],
                                    existing_notes: set) -> float:
        """Calculate similarity based on fragrance families"""

        current_families = set()
        existing_families = set()

        for note_id, _ in individual:
            if note_id in self.notes_db:
                current_families.add(self.notes_db[note_id].get('family', 'unknown'))

        for note_id in existing_notes:
            if note_id in self.notes_db:
                existing_families.add(self.notes_db[note_id].get('family', 'unknown'))

        if not current_families or not existing_families:
            return 0.0

        return len(current_families & existing_families) / len(current_families | existing_families)

    def _custom_mutation(self, individual: List[Tuple[int, float]]) -> Tuple[List]:
        """Custom mutation operator for fragrance recipes"""

        for i in range(len(individual)):
            if random.random() < self.mutation_prob:
                # Mutate either note or percentage
                if random.random() < 0.5:
                    # Mutate note
                    individual[i] = (random.randint(1, len(self.notes_db)), individual[i][1])
                else:
                    # Mutate percentage
                    new_percentage = individual[i][1] + random.gauss(0, 1.0)
                    new_percentage = max(0.0, min(10.0, new_percentage))  # Clamp
                    individual[i] = (individual[i][0], new_percentage)

        return (individual,)

    def optimize(self, creative_brief: CreativeBrief, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the genetic algorithm optimization

        Returns:
            Dictionary with best individual, pareto front, and statistics
        """

        self.creative_brief = creative_brief

        if verbose:
            logger.info(f"Starting MOGA optimization: {self.population_size} pop, {self.generations} gen")

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Setup hall of fame and pareto front
        hof = HallOfFame(1)
        pareto = ParetoFront()

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Evolution
        for gen in range(self.generations):

            # Evaluate population
            fitnesses = list(map(self.toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Update hall of fame and pareto front
            hof.update(population)
            pareto.update(population)

            # Record statistics
            record = stats.compile(population)

            if verbose and gen % 10 == 0:
                logger.info(f"Gen {gen}: stability={record['min'][0]:.3f}, "
                          f"unfitness={record['min'][1]:.3f}, "
                          f"uncreativity={record['min'][2]:.3f}")

            # Selection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Replace population
            population[:] = offspring

        # Final evaluation
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Get best individual
        best_ind = tools.selBest(population, k=1)[0]

        if verbose:
            logger.info(f"Optimization complete! Best fitness: {best_ind.fitness.values}")

        # Format results
        results = {
            "best_individual": {
                "recipe": best_ind,
                "fitness": best_ind.fitness.values,
                "description": self._format_recipe(best_ind)
            },
            "pareto_front": [
                {
                    "recipe": ind,
                    "fitness": ind.fitness.values
                }
                for ind in pareto
            ],
            "statistics": {
                "final_generation": record,
                "hall_of_fame": [
                    {
                        "recipe": ind,
                        "fitness": ind.fitness.values
                    }
                    for ind in hof
                ]
            }
        }

        return results

    def _format_recipe(self, individual: List[Tuple[int, float]]) -> Dict:
        """Format individual as readable recipe"""

        recipe = self._individual_to_recipe(individual)

        return {
            "top_notes": recipe["notes"]["top"],
            "middle_notes": recipe["notes"]["middle"],
            "base_notes": recipe["notes"]["base"],
            "total_concentration": recipe["concentrations"]["total"],
            "balance": {
                "top": f"{recipe['concentrations']['top']:.1f}%",
                "middle": f"{recipe['concentrations']['middle']:.1f}%",
                "base": f"{recipe['concentrations']['base']:.1f}%"
            }
        }


if __name__ == "__main__":
    # Example usage
    brief = CreativeBrief(
        emotional_palette=[0.7, 0.3, 0.8],  # Happy, calm, romantic
        fragrance_family="floral",
        mood="romantic",
        intensity=0.7,
        season="spring",
        gender="feminine"
    )

    optimizer = EnhancedMOGAOptimizer(
        population_size=50,
        generations=30
    )

    results = optimizer.optimize(brief)

    print("\nBest Recipe Found:")
    print("-" * 50)
    best = results["best_individual"]
    print(f"Fitness: stability={best['fitness'][0]:.3f}, "
          f"unfitness={best['fitness'][1]:.3f}, "
          f"uncreativity={best['fitness'][2]:.3f}")
    print(f"\nComposition:")
    print(f"  Top Notes: {best['description']['top_notes']}")
    print(f"  Middle Notes: {best['description']['middle_notes']}")
    print(f"  Base Notes: {best['description']['base_notes']}")
    print(f"  Total: {best['description']['total_concentration']:.1f}%")
    print(f"\nBalance: {best['description']['balance']}")