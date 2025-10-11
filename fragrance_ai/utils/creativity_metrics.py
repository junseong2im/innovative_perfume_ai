# fragrance_ai/utils/creativity_metrics.py

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class CreativityMetrics:
    """
    Enhanced creativity metrics with defensive programming
    Prevents numerical instabilities and ensures valid outputs
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize creativity metrics calculator

        Args:
            config: Configuration dictionary with entropy and novelty settings
        """
        if config is None:
            config = {}

        # Entropy settings with safe defaults
        entropy_config = config.get('entropy', {})
        self.epsilon = entropy_config.get('epsilon', 1e-12)
        self.min_probability = entropy_config.get('min_probability', 0.001)

        # Category balance settings
        category_config = config.get('category_weights', {})
        self.category_weights = {
            'top_notes': category_config.get('top_notes', 0.25),
            'heart_notes': category_config.get('heart_notes', 0.40),
            'base_notes': category_config.get('base_notes', 0.35)
        }

        # Novelty settings
        novelty_config = config.get('novelty', {})
        self.novelty_enabled = novelty_config.get('enabled', True)
        self.comparison_pool_size = novelty_config.get('comparison_pool_size', 50)
        self.distance_threshold = novelty_config.get('distance_threshold', 0.3)

        # History for novelty calculation
        self.formulation_history = []

    def calculate_entropy(self, concentrations: List[float]) -> float:
        """
        Calculate Shannon entropy with defensive programming

        Args:
            concentrations: List of ingredient concentrations

        Returns:
            Entropy value (0 to log(n)) - higher means more diverse
        """
        if not concentrations or len(concentrations) == 0:
            return 0.0

        # Filter out very small concentrations to prevent numerical issues
        filtered = [c for c in concentrations if c > self.min_probability]

        if not filtered:
            return 0.0

        # Convert to probabilities (normalize to sum to 1)
        total = sum(filtered)
        if total <= 0:
            return 0.0

        probabilities = np.array([c / total for c in filtered])

        # Add epsilon for numerical stability and filter p > 0
        probabilities = probabilities[probabilities > 0]
        probabilities = np.clip(probabilities, self.epsilon, 1.0)

        # Calculate Shannon entropy
        # H = -Σ p_i * log(p_i)
        entropy = -np.sum(probabilities * np.log(probabilities + self.epsilon))

        # Normalize by maximum possible entropy
        max_entropy = np.log(len(probabilities))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0

        # Ensure valid range [0, 1]
        return float(np.clip(normalized_entropy, 0.0, 1.0))

    def calculate_category_balance(self, formulation: Dict[str, Any]) -> float:
        """
        Calculate how well balanced the fragrance categories are

        Args:
            formulation: Dictionary with ingredients and their categories

        Returns:
            Balance score (0 to 1) - 1 means perfectly balanced
        """
        if not formulation or 'ingredients' not in formulation:
            return 0.0

        # Count concentrations by category
        category_totals = {
            'top': 0.0,
            'heart': 0.0,
            'base': 0.0
        }

        total_concentration = 0.0

        for ingredient in formulation['ingredients']:
            category = ingredient.get('category', '').lower()
            concentration = float(ingredient.get('concentration', 0))

            # Defensive: ensure positive concentration
            concentration = max(0.0, concentration)

            if 'top' in category:
                category_totals['top'] += concentration
            elif 'heart' in category or 'mid' in category:
                category_totals['heart'] += concentration
            elif 'base' in category:
                category_totals['base'] += concentration

            total_concentration += concentration

        if total_concentration <= 0:
            return 0.0

        # Calculate percentages
        percentages = {
            cat: (total / total_concentration) for cat, total in category_totals.items()
        }

        # Score based on ideal ranges
        # Top: 20-30%, Heart: 30-50%, Base: 30-50%
        score = 0.0

        # Top notes scoring
        if 0.20 <= percentages['top'] <= 0.30:
            score += 0.33
        else:
            # Partial credit for being close
            if percentages['top'] < 0.20:
                score += 0.33 * (percentages['top'] / 0.20)
            else:
                score += 0.33 * max(0, 1 - (percentages['top'] - 0.30) / 0.20)

        # Heart notes scoring
        if 0.30 <= percentages['heart'] <= 0.50:
            score += 0.33
        else:
            if percentages['heart'] < 0.30:
                score += 0.33 * (percentages['heart'] / 0.30)
            else:
                score += 0.33 * max(0, 1 - (percentages['heart'] - 0.50) / 0.20)

        # Base notes scoring
        if 0.30 <= percentages['base'] <= 0.50:
            score += 0.34
        else:
            if percentages['base'] < 0.30:
                score += 0.34 * (percentages['base'] / 0.30)
            else:
                score += 0.34 * max(0, 1 - (percentages['base'] - 0.50) / 0.20)

        return float(np.clip(score, 0.0, 1.0))

    def calculate_novelty(self, formulation: Dict[str, Any]) -> float:
        """
        Calculate novelty score compared to historical formulations

        Args:
            formulation: Current formulation to evaluate

        Returns:
            Novelty score (0 to 1) - 1 means very novel
        """
        if not self.novelty_enabled or not self.formulation_history:
            return 0.5  # Neutral score if no history

        # Extract ingredient profile
        current_profile = self._extract_profile(formulation)

        if not current_profile:
            return 0.5

        # Compare with recent formulations
        distances = []
        comparison_pool = self.formulation_history[-self.comparison_pool_size:]

        for historical in comparison_pool:
            hist_profile = self._extract_profile(historical)
            if hist_profile:
                distance = self._calculate_distance(current_profile, hist_profile)
                distances.append(distance)

        if not distances:
            return 0.5

        # Calculate novelty as average distance
        avg_distance = np.mean(distances)

        # Normalize to [0, 1] range
        novelty_score = min(1.0, avg_distance / self.distance_threshold)

        return float(novelty_score)

    def _extract_profile(self, formulation: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract normalized ingredient profile from formulation"""
        if not formulation or 'ingredients' not in formulation:
            return None

        # Create a fixed-size profile vector
        profile = np.zeros(50)  # Support up to 50 unique ingredients

        for ingredient in formulation['ingredients']:
            ing_id = ingredient.get('id', 0)
            concentration = ingredient.get('concentration', 0)

            # Use modulo to map to profile index
            idx = int(ing_id) % len(profile)
            profile[idx] += concentration

        # Normalize
        total = np.sum(profile)
        if total > 0:
            profile = profile / total

        return profile

    def _calculate_distance(self, profile1: np.ndarray, profile2: np.ndarray) -> float:
        """Calculate distance between two profiles using cosine distance"""
        # Defensive: ensure same shape
        min_len = min(len(profile1), len(profile2))
        p1 = profile1[:min_len]
        p2 = profile2[:min_len]

        # Cosine distance
        dot_product = np.dot(p1, p2)
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)

        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance if one is zero

        cosine_similarity = dot_product / (norm1 * norm2)
        cosine_distance = 1 - cosine_similarity

        return float(np.clip(cosine_distance, 0.0, 1.0))

    def calculate_creativity_score(self, formulation: Dict[str, Any],
                                  concentrations: List[float] = None) -> Dict[str, float]:
        """
        Calculate comprehensive creativity score

        Args:
            formulation: Complete formulation dictionary
            concentrations: Optional list of concentrations for entropy

        Returns:
            Dictionary with creativity metrics
        """
        # Extract concentrations if not provided
        if concentrations is None and formulation and 'ingredients' in formulation:
            concentrations = [
                float(ing.get('concentration', 0))
                for ing in formulation['ingredients']
            ]

        # Calculate individual metrics
        entropy = self.calculate_entropy(concentrations) if concentrations else 0.0
        balance = self.calculate_category_balance(formulation)
        novelty = self.calculate_novelty(formulation)

        # Combined creativity score (weighted average)
        combined = (
            0.3 * entropy +  # Diversity of concentrations
            0.3 * balance +  # Category balance
            0.4 * novelty    # Novelty compared to history
        )

        # Add to history for future novelty calculations
        if formulation:
            self.formulation_history.append(formulation)
            # Keep history size manageable
            if len(self.formulation_history) > self.comparison_pool_size * 2:
                self.formulation_history = self.formulation_history[-self.comparison_pool_size:]

        return {
            'entropy': float(entropy),
            'balance': float(balance),
            'novelty': float(novelty),
            'combined': float(combined)
        }

    def validate_concentrations(self, concentrations: List[float],
                               min_concentration: float = 0.1) -> List[float]:
        """
        Validate and clean concentration values

        Args:
            concentrations: List of concentrations
            min_concentration: Minimum effective concentration

        Returns:
            Cleaned concentration list
        """
        if not concentrations:
            return []

        # Remove very small values that can cause numerical issues
        cleaned = []
        for c in concentrations:
            # Handle NaN and inf
            if not np.isfinite(c):
                logger.warning(f"Non-finite concentration detected: {c}")
                continue

            # Convert to float and check range
            c_float = float(c)
            if c_float >= min_concentration:
                cleaned.append(c_float)

        return cleaned

    def check_for_numerical_issues(self, values: List[float]) -> bool:
        """
        Check for NaN, inf, or other numerical issues

        Args:
            values: List of numerical values

        Returns:
            True if all values are valid, False otherwise
        """
        if not values:
            return True

        for v in values:
            if not np.isfinite(v):
                return False

        return True


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create metrics calculator
    config = {
        'entropy': {'epsilon': 1e-12, 'min_probability': 0.001},
        'category_weights': {'top_notes': 0.25, 'heart_notes': 0.40, 'base_notes': 0.35},
        'novelty': {'enabled': True, 'comparison_pool_size': 50, 'distance_threshold': 0.3}
    }

    metrics = CreativityMetrics(config)

    # Test entropy calculation
    concentrations = [20.5, 15.3, 30.2, 10.0, 24.0]
    entropy = metrics.calculate_entropy(concentrations)
    print(f"Entropy: {entropy:.4f}")

    # Test with edge cases
    print("\nEdge case tests:")

    # Empty concentrations
    print(f"Empty: {metrics.calculate_entropy([])}")

    # Single concentration
    print(f"Single: {metrics.calculate_entropy([100.0])}")

    # Very small concentrations
    print(f"Small: {metrics.calculate_entropy([0.0001, 0.0002, 99.9997])}")

    # Test category balance
    formulation = {
        'ingredients': [
            {'category': 'top', 'concentration': 25},
            {'category': 'heart', 'concentration': 40},
            {'category': 'base', 'concentration': 35}
        ]
    }
    balance = metrics.calculate_category_balance(formulation)
    print(f"\nCategory balance: {balance:.4f}")

    # Test complete creativity score
    creativity = metrics.calculate_creativity_score(formulation, concentrations)
    print(f"\nCreativity scores:")
    for key, value in creativity.items():
        print(f"  {key}: {value:.4f}")

    # Test numerical validation
    test_values = [1.0, 2.0, float('nan'), float('inf'), 3.0]
    print(f"\nNumerical validation: {metrics.check_for_numerical_issues(test_values)}")

    cleaned = metrics.validate_concentrations(test_values)
    print(f"Cleaned values: {cleaned}")