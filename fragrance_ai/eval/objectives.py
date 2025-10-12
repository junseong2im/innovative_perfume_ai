# fragrance_ai/eval/objectives.py
"""
Evaluation Objectives for Fragrance Optimization
Implements weighted multi-objective evaluation: F_total = w_c*creativity + w_f*fitness + w_s*stability
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# Weight Profiles
# ============================================================================

class OptimizationProfile(str, Enum):
    """Pre-defined weight profiles for different optimization goals"""
    CREATIVE = "creative"        # High creativity weight
    COMMERCIAL = "commercial"    # Balanced weights
    STABLE = "stable"           # High stability weight
    FITNESS = "fitness"         # High fitness weight
    CUSTOM = "custom"           # User-defined weights


@dataclass
class WeightProfile:
    """Weight configuration for objectives"""
    creativity: float  # w_c
    fitness: float     # w_f
    stability: float   # w_s

    def __post_init__(self):
        """Validate weights sum to 1"""
        total = self.creativity + self.fitness + self.stability
        if abs(total - 1.0) > 1e-6:
            # Auto-normalize
            self.creativity /= total
            self.fitness /= total
            self.stability /= total

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            "w_creativity": self.creativity,
            "w_fitness": self.fitness,
            "w_stability": self.stability
        }


# Pre-defined weight profiles
WEIGHT_PROFILES = {
    OptimizationProfile.CREATIVE: WeightProfile(0.5, 0.3, 0.2),
    OptimizationProfile.COMMERCIAL: WeightProfile(0.33, 0.34, 0.33),
    OptimizationProfile.STABLE: WeightProfile(0.2, 0.3, 0.5),
    OptimizationProfile.FITNESS: WeightProfile(0.2, 0.5, 0.3),
}


# ============================================================================
# Creativity Objectives
# ============================================================================

class CreativityEvaluator:
    """Evaluate creativity and originality of formulations"""

    @staticmethod
    def calculate_entropy(concentrations: List[float]) -> float:
        """
        Calculate Shannon entropy as measure of complexity

        Higher entropy = more balanced/complex formula
        Lower entropy = dominated by few ingredients

        Args:
            concentrations: List of percentages

        Returns:
            Entropy value [0, log(n)]
        """
        # Filter out zeros and normalize
        nonzero = [c for c in concentrations if c > 0]
        if not nonzero:
            return 0.0

        total = sum(nonzero)
        if total == 0:
            return 0.0

        # Normalize to probabilities
        probs = [c / total for c in nonzero]

        # Calculate Shannon entropy
        entropy = -sum(p * math.log(p) for p in probs if p > 0)

        # Normalize by max possible entropy
        max_entropy = math.log(len(nonzero))
        if max_entropy > 0:
            entropy = entropy / max_entropy

        return entropy

    @staticmethod
    def calculate_uniqueness(
        formula: List[Tuple[int, float]],
        reference_formulas: List[List[Tuple[int, float]]],
        distance_threshold: float = 0.3
    ) -> float:
        """
        Calculate uniqueness compared to reference formulas

        Args:
            formula: Current formula [(ingredient_id, concentration)]
            reference_formulas: List of reference formulas
            distance_threshold: Minimum distance for uniqueness

        Returns:
            Uniqueness score [0, 1]
        """
        if not reference_formulas:
            return 1.0  # Maximally unique if no references

        # Convert to vector form for comparison
        all_ingredients = set()
        all_ingredients.update(ing_id for ing_id, _ in formula)
        for ref in reference_formulas:
            all_ingredients.update(ing_id for ing_id, _ in ref)

        ingredient_list = sorted(all_ingredients)
        n_ingredients = len(ingredient_list)

        # Create formula vector
        formula_vec = np.zeros(n_ingredients)
        for ing_id, conc in formula:
            idx = ingredient_list.index(ing_id)
            formula_vec[idx] = conc

        # Calculate distances to all references
        min_distance = float('inf')
        for ref_formula in reference_formulas:
            ref_vec = np.zeros(n_ingredients)
            for ing_id, conc in ref_formula:
                if ing_id in ingredient_list:
                    idx = ingredient_list.index(ing_id)
                    ref_vec[idx] = conc

            # Euclidean distance
            distance = np.linalg.norm(formula_vec - ref_vec) / 100.0
            min_distance = min(min_distance, distance)

        # Convert to uniqueness score
        uniqueness = min(1.0, min_distance / distance_threshold)
        return uniqueness

    @staticmethod
    def calculate_innovation(
        formula: List[Tuple[int, float]],
        rare_ingredients: set,
        common_ingredients: set
    ) -> float:
        """
        Calculate innovation based on use of rare ingredients

        Args:
            formula: [(ingredient_id, concentration)]
            rare_ingredients: Set of rare ingredient IDs
            common_ingredients: Set of common ingredient IDs

        Returns:
            Innovation score [0, 1]
        """
        if not formula:
            return 0.0

        rare_weight = 0.0
        common_weight = 0.0

        for ing_id, conc in formula:
            if ing_id in rare_ingredients:
                rare_weight += conc * 2.0  # Double weight for rare
            elif ing_id in common_ingredients:
                common_weight += conc * 0.5  # Half weight for common
            else:
                # Neutral weight for unknown
                rare_weight += conc

        total_weight = rare_weight + common_weight
        if total_weight == 0:
            return 0.5  # Neutral innovation

        innovation = rare_weight / total_weight
        return min(1.0, innovation)

    def evaluate(
        self,
        formula: List[Tuple[int, float]],
        reference_formulas: Optional[List[List[Tuple[int, float]]]] = None,
        rare_ingredients: Optional[set] = None,
        common_ingredients: Optional[set] = None
    ) -> float:
        """
        Overall creativity score

        Args:
            formula: Current formula
            reference_formulas: Reference formulas for comparison
            rare_ingredients: Set of rare ingredients
            common_ingredients: Set of common ingredients

        Returns:
            Creativity score [0, 1]
        """
        concentrations = [conc for _, conc in formula]

        # Component scores
        entropy = self.calculate_entropy(concentrations)

        uniqueness = 0.5  # Default if no references
        if reference_formulas:
            uniqueness = self.calculate_uniqueness(formula, reference_formulas)

        innovation = 0.5  # Default if no ingredient sets
        if rare_ingredients and common_ingredients:
            innovation = self.calculate_innovation(
                formula, rare_ingredients, common_ingredients
            )

        # Weighted combination
        creativity = (0.3 * entropy + 0.4 * uniqueness + 0.3 * innovation)
        return creativity


# ============================================================================
# Fitness Objectives
# ============================================================================

class FitnessEvaluator:
    """Evaluate fitness to requirements and constraints"""

    @staticmethod
    def calculate_target_match(
        actual: Dict[str, float],
        target: Dict[str, float]
    ) -> float:
        """
        Calculate match to target profile

        Args:
            actual: Actual characteristics
            target: Target characteristics

        Returns:
            Match score [0, 1]
        """
        if not target:
            return 1.0  # Perfect match if no targets

        total_deviation = 0.0
        n_metrics = 0

        for key in target:
            if key in actual:
                target_val = target[key]
                actual_val = actual[key]

                # Normalize deviation by target value
                if target_val != 0:
                    deviation = abs(actual_val - target_val) / abs(target_val)
                else:
                    deviation = abs(actual_val)

                total_deviation += min(1.0, deviation)  # Cap at 100% deviation
                n_metrics += 1

        if n_metrics == 0:
            return 1.0

        # Convert deviation to match score
        avg_deviation = total_deviation / n_metrics
        match_score = 1.0 - avg_deviation
        return max(0.0, match_score)

    @staticmethod
    def calculate_constraint_satisfaction(
        formula: List[Tuple[int, float]],
        constraints: Dict[int, Dict[str, float]]
    ) -> float:
        """
        Calculate constraint satisfaction

        Args:
            formula: [(ingredient_id, concentration)]
            constraints: {ingredient_id: {"min": x, "max": y}}

        Returns:
            Satisfaction score [0, 1]
        """
        if not constraints:
            return 1.0  # Fully satisfied if no constraints

        violations = 0
        total_constraints = 0

        for ing_id, conc in formula:
            if ing_id in constraints:
                constraint = constraints[ing_id]

                if "min" in constraint:
                    total_constraints += 1
                    if conc < constraint["min"]:
                        violations += 1

                if "max" in constraint:
                    total_constraints += 1
                    if conc > constraint["max"]:
                        violations += 1

        # Check for required ingredients not in formula
        formula_ingredients = {ing_id for ing_id, _ in formula}
        for ing_id, constraint in constraints.items():
            if ing_id not in formula_ingredients:
                if "min" in constraint and constraint["min"] > 0:
                    violations += 1
                    total_constraints += 1

        if total_constraints == 0:
            return 1.0

        satisfaction = 1.0 - (violations / total_constraints)
        return satisfaction

    @staticmethod
    def calculate_cost_fitness(
        actual_cost: float,
        target_cost: float,
        tolerance: float = 0.1
    ) -> float:
        """
        Calculate cost fitness

        Args:
            actual_cost: Actual cost per kg
            target_cost: Target cost per kg
            tolerance: Acceptable deviation ratio

        Returns:
            Cost fitness [0, 1]
        """
        if target_cost <= 0:
            return 1.0  # No cost constraint

        deviation_ratio = abs(actual_cost - target_cost) / target_cost

        if deviation_ratio <= tolerance:
            # Within tolerance - linear score
            fitness = 1.0 - (deviation_ratio / tolerance) * 0.2
        else:
            # Outside tolerance - exponential decay
            excess = deviation_ratio - tolerance
            fitness = 0.8 * math.exp(-2 * excess)

        return max(0.0, fitness)

    def evaluate(
        self,
        formula: List[Tuple[int, float]],
        actual_properties: Dict[str, float],
        target_properties: Dict[str, float],
        constraints: Optional[Dict[int, Dict[str, float]]] = None,
        actual_cost: Optional[float] = None,
        target_cost: Optional[float] = None
    ) -> float:
        """
        Overall fitness score

        Args:
            formula: Current formula
            actual_properties: Actual fragrance properties
            target_properties: Target fragrance properties
            constraints: Ingredient constraints
            actual_cost: Actual cost per kg
            target_cost: Target cost per kg

        Returns:
            Fitness score [0, 1]
        """
        # Component scores
        target_match = self.calculate_target_match(actual_properties, target_properties)

        constraint_satisfaction = 1.0  # Default if no constraints
        if constraints:
            constraint_satisfaction = self.calculate_constraint_satisfaction(
                formula, constraints
            )

        cost_fitness = 1.0  # Default if no cost targets
        if actual_cost is not None and target_cost is not None:
            cost_fitness = self.calculate_cost_fitness(actual_cost, target_cost)

        # Weighted combination
        fitness = (0.4 * target_match + 0.4 * constraint_satisfaction + 0.2 * cost_fitness)
        return fitness


# ============================================================================
# Stability Objectives
# ============================================================================

class StabilityEvaluator:
    """Evaluate formulation stability"""

    @staticmethod
    def calculate_chemical_stability(
        formula: List[Tuple[int, float]],
        incompatibilities: Dict[Tuple[int, int], float]
    ) -> float:
        """
        Calculate chemical stability based on incompatibilities

        Args:
            formula: [(ingredient_id, concentration)]
            incompatibilities: {(id1, id2): severity} where severity in [0, 1]

        Returns:
            Chemical stability [0, 1]
        """
        if not incompatibilities:
            return 1.0  # Fully stable if no known incompatibilities

        total_incompatibility = 0.0
        total_weight = 0.0

        formula_dict = dict(formula)

        for (id1, id2), severity in incompatibilities.items():
            if id1 in formula_dict and id2 in formula_dict:
                # Weight by concentrations
                weight = formula_dict[id1] * formula_dict[id2] / 10000  # Normalize
                total_incompatibility += severity * weight
                total_weight += weight

        if total_weight == 0:
            return 1.0

        # Higher incompatibility = lower stability
        stability = 1.0 - min(1.0, total_incompatibility)
        return stability

    @staticmethod
    def calculate_physical_stability(
        vapor_pressures: Dict[int, float],
        formula: List[Tuple[int, float]],
        temperature: float = 25.0
    ) -> float:
        """
        Calculate physical stability based on volatility

        Args:
            vapor_pressures: {ingredient_id: vapor_pressure} in mmHg at 25°C
            formula: [(ingredient_id, concentration)]
            temperature: Temperature in °C

        Returns:
            Physical stability [0, 1]
        """
        if not vapor_pressures:
            return 0.5  # Neutral if no data

        # Calculate weighted average vapor pressure
        total_vp = 0.0
        total_weight = 0.0

        for ing_id, conc in formula:
            if ing_id in vapor_pressures:
                vp = vapor_pressures[ing_id]
                # Temperature correction (rough approximation)
                vp_corrected = vp * math.exp(0.05 * (temperature - 25))
                total_vp += vp_corrected * conc
                total_weight += conc

        if total_weight == 0:
            return 0.5

        avg_vp = total_vp / total_weight

        # Convert vapor pressure to stability score
        # Low VP (< 1 mmHg) = stable, High VP (> 100 mmHg) = unstable
        if avg_vp <= 1:
            stability = 1.0
        elif avg_vp >= 100:
            stability = 0.0
        else:
            # Logarithmic scale
            stability = 1.0 - (math.log10(avg_vp) + 0.5) / 2.5

        return max(0.0, min(1.0, stability))

    @staticmethod
    def calculate_oxidation_stability(
        formula: List[Tuple[int, float]],
        oxidation_prone: set,
        antioxidants: set
    ) -> float:
        """
        Calculate oxidation stability

        Args:
            formula: [(ingredient_id, concentration)]
            oxidation_prone: Set of oxidation-prone ingredient IDs
            antioxidants: Set of antioxidant ingredient IDs

        Returns:
            Oxidation stability [0, 1]
        """
        prone_weight = 0.0
        antioxidant_weight = 0.0

        for ing_id, conc in formula:
            if ing_id in oxidation_prone:
                prone_weight += conc
            if ing_id in antioxidants:
                antioxidant_weight += conc * 2.0  # Antioxidants more effective

        if prone_weight == 0:
            return 1.0  # No oxidation risk

        # Protection factor
        protection = min(1.0, antioxidant_weight / prone_weight)

        # Base stability depends on prone weight
        base_stability = 1.0 - min(1.0, prone_weight / 50)  # 50% threshold

        # Combined stability
        stability = base_stability + (1 - base_stability) * protection
        return stability

    def evaluate(
        self,
        formula: List[Tuple[int, float]],
        incompatibilities: Optional[Dict[Tuple[int, int], float]] = None,
        vapor_pressures: Optional[Dict[int, float]] = None,
        oxidation_prone: Optional[set] = None,
        antioxidants: Optional[set] = None,
        temperature: float = 25.0
    ) -> float:
        """
        Overall stability score

        Args:
            formula: Current formula
            incompatibilities: Chemical incompatibilities
            vapor_pressures: Vapor pressure data
            oxidation_prone: Oxidation-prone ingredients
            antioxidants: Antioxidant ingredients
            temperature: Storage temperature

        Returns:
            Stability score [0, 1]
        """
        # Component scores with defaults
        chemical = 0.5
        if incompatibilities:
            chemical = self.calculate_chemical_stability(formula, incompatibilities)

        physical = 0.5
        if vapor_pressures:
            physical = self.calculate_physical_stability(
                vapor_pressures, formula, temperature
            )

        oxidation = 0.5
        if oxidation_prone and antioxidants:
            oxidation = self.calculate_oxidation_stability(
                formula, oxidation_prone, antioxidants
            )

        # Weighted combination
        stability = (0.4 * chemical + 0.3 * physical + 0.3 * oxidation)
        return stability


# ============================================================================
# Total Objective Function
# ============================================================================

class TotalObjective:
    """
    Combined objective function: F_total = w_c*creativity + w_f*fitness + w_s*stability
    """

    def __init__(self, profile: OptimizationProfile = OptimizationProfile.COMMERCIAL):
        """
        Initialize with weight profile

        Args:
            profile: Optimization profile determining weights
        """
        self.profile = profile
        if profile in WEIGHT_PROFILES:
            self.weights = WEIGHT_PROFILES[profile]
        else:
            self.weights = WEIGHT_PROFILES[OptimizationProfile.COMMERCIAL]

        # Initialize evaluators
        self.creativity_evaluator = CreativityEvaluator()
        self.fitness_evaluator = FitnessEvaluator()
        self.stability_evaluator = StabilityEvaluator()

    def set_custom_weights(self, w_creativity: float, w_fitness: float, w_stability: float):
        """
        Set custom weights

        Args:
            w_creativity: Weight for creativity (w_c)
            w_fitness: Weight for fitness (w_f)
            w_stability: Weight for stability (w_s)
        """
        self.profile = OptimizationProfile.CUSTOM
        self.weights = WeightProfile(w_creativity, w_fitness, w_stability)

    def evaluate(
        self,
        formula: List[Tuple[int, float]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate total objective value

        Args:
            formula: [(ingredient_id, concentration)]
            context: Optional context with additional data

        Returns:
            Dictionary with component scores and total
        """
        if context is None:
            context = {}

        # Calculate component scores
        creativity = self.creativity_evaluator.evaluate(
            formula,
            reference_formulas=context.get("reference_formulas"),
            rare_ingredients=context.get("rare_ingredients"),
            common_ingredients=context.get("common_ingredients")
        )

        fitness = self.fitness_evaluator.evaluate(
            formula,
            actual_properties=context.get("actual_properties", {}),
            target_properties=context.get("target_properties", {}),
            constraints=context.get("constraints"),
            actual_cost=context.get("actual_cost"),
            target_cost=context.get("target_cost")
        )

        stability = self.stability_evaluator.evaluate(
            formula,
            incompatibilities=context.get("incompatibilities"),
            vapor_pressures=context.get("vapor_pressures"),
            oxidation_prone=context.get("oxidation_prone"),
            antioxidants=context.get("antioxidants"),
            temperature=context.get("temperature", 25.0)
        )

        # Calculate weighted total
        total = (
            self.weights.creativity * creativity +
            self.weights.fitness * fitness +
            self.weights.stability * stability
        )

        return {
            "creativity": creativity,
            "fitness": fitness,
            "stability": stability,
            "total": total,
            "weights": self.weights.to_dict()
        }

    def evaluate_batch(
        self,
        formulas: List[List[Tuple[int, float]]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, float]]:
        """
        Evaluate multiple formulas

        Args:
            formulas: List of formulas
            context: Shared context for all formulas

        Returns:
            List of evaluation results
        """
        return [self.evaluate(formula, context) for formula in formulas]


# ============================================================================
# Pareto Optimization
# ============================================================================

class ParetoOptimizer:
    """Multi-objective optimization using Pareto dominance"""

    @staticmethod
    def is_dominated(scores1: Dict[str, float], scores2: Dict[str, float]) -> bool:
        """
        Check if scores1 is dominated by scores2

        Args:
            scores1: First score dictionary
            scores2: Second score dictionary

        Returns:
            True if scores1 is dominated by scores2
        """
        objectives = ["creativity", "fitness", "stability"]

        # Check if scores2 is at least as good in all objectives
        at_least_as_good = all(
            scores2.get(obj, 0) >= scores1.get(obj, 0)
            for obj in objectives
        )

        # Check if scores2 is strictly better in at least one objective
        strictly_better = any(
            scores2.get(obj, 0) > scores1.get(obj, 0)
            for obj in objectives
        )

        return at_least_as_good and strictly_better

    @classmethod
    def find_pareto_front(
        cls,
        formulas: List[List[Tuple[int, float]]],
        evaluator: TotalObjective,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[int, Dict[str, float]]]:
        """
        Find Pareto-optimal formulas

        Args:
            formulas: List of formulas
            evaluator: Objective evaluator
            context: Evaluation context

        Returns:
            List of (index, scores) for Pareto-optimal formulas
        """
        # Evaluate all formulas
        all_scores = evaluator.evaluate_batch(formulas, context)

        # Find non-dominated solutions
        pareto_front = []

        for i, scores_i in enumerate(all_scores):
            is_dominated = False
            for j, scores_j in enumerate(all_scores):
                if i != j and cls.is_dominated(scores_i, scores_j):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append((i, scores_i))

        return pareto_front


# ============================================================================
# Helper Functions
# ============================================================================

def create_evaluator(
    profile: OptimizationProfile = OptimizationProfile.COMMERCIAL
) -> TotalObjective:
    """Create configured evaluator"""
    return TotalObjective(profile)


def quick_evaluate(
    formula: List[Tuple[int, float]],
    w_creativity: float = 0.33,
    w_fitness: float = 0.34,
    w_stability: float = 0.33
) -> float:
    """
    Quick evaluation with custom weights

    Args:
        formula: [(ingredient_id, concentration)]
        w_creativity: Creativity weight
        w_fitness: Fitness weight
        w_stability: Stability weight

    Returns:
        Total objective value
    """
    evaluator = TotalObjective()
    evaluator.set_custom_weights(w_creativity, w_fitness, w_stability)
    result = evaluator.evaluate(formula)
    return result["total"]


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Main classes
    'TotalObjective',
    'CreativityEvaluator',
    'FitnessEvaluator',
    'StabilityEvaluator',
    'ParetoOptimizer',

    # Weight management
    'WeightProfile',
    'OptimizationProfile',
    'WEIGHT_PROFILES',

    # Helper functions
    'create_evaluator',
    'quick_evaluate'
]