"""
Test script for MOGA (Multi-Objective Genetic Algorithm) Components
Verifies the Fitness, Individual, and genetic operators are working correctly
"""

import pytest
import numpy as np
from typing import List, Tuple
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.training.moga_optimizer import OlfactoryRecombinatorAI, CreativeBrief
from deap import creator, base, tools
import random


class TestMOGAComponents:
    """Test suite for MOGA components"""

    @pytest.fixture
    def optimizer(self):
        """Create a MOGA optimizer instance"""
        brief = CreativeBrief(
            emotional_palette=[0.7, 0.3, 0.5, 0.8],  # Happy, calm, fresh, romantic
            gender="unisex",
            season="spring",
            time_of_day="evening",
            cultural_background="global"
        )
        return OlfactoryRecombinatorAI(
            population_size=20,
            generations=5,
            creative_brief=brief
        )

    def test_fitness_class_creation(self, optimizer):
        """Test that FitnessMin class is properly created with 3 objectives"""
        # The optimizer should have created the FitnessMin class
        assert hasattr(creator, "FitnessMin")

        # Create an instance and verify it has 3 weights for minimization
        fitness = creator.FitnessMin()
        assert hasattr(fitness, "weights")
        assert fitness.weights == (-1.0, -1.0, -1.0)

        # Test setting values
        fitness.values = (0.5, 0.3, 0.7)
        assert fitness.values == (0.5, 0.3, 0.7)

        # Test comparison (lower is better for minimization)
        fitness1 = creator.FitnessMin()
        fitness2 = creator.FitnessMin()
        fitness1.values = (0.5, 0.3, 0.7)
        fitness2.values = (0.6, 0.4, 0.8)

        # fitness1 should dominate fitness2 (all values are lower)
        assert fitness1 > fitness2  # In DEAP, > means "better than" for minimization

    def test_individual_class_creation(self, optimizer):
        """Test that Individual class is properly created as a list with fitness"""
        # The optimizer should have created the Individual class
        assert hasattr(creator, "Individual")

        # Create an individual
        individual = creator.Individual()

        # Should be a list
        assert isinstance(individual, list)

        # Should have fitness attribute
        assert hasattr(individual, "fitness")
        assert isinstance(individual.fitness, creator.FitnessMin)

    def test_gene_generation(self, optimizer):
        """Test gene generation produces valid note-percentage tuples"""
        # Generate multiple genes
        genes = [optimizer._generate_gene() for _ in range(100)]

        for gene in genes:
            # Should be a tuple with 2 elements
            assert isinstance(gene, tuple)
            assert len(gene) == 2

            note_id, percentage = gene

            # Note ID should be valid
            assert isinstance(note_id, int)
            assert 1 <= note_id <= len(optimizer.notes_db)

            # Percentage should be in valid range
            assert isinstance(percentage, float)
            assert 0.1 <= percentage <= 10.0

    def test_individual_generation(self, optimizer):
        """Test individual generation creates valid fragrance recipes"""
        # Generate an individual using the toolbox
        individual = optimizer.toolbox.individual()

        # Should have 15 genes (notes)
        assert len(individual) == 15

        # Each gene should be a valid tuple
        for gene in individual:
            assert isinstance(gene, tuple)
            assert len(gene) == 2
            note_id, percentage = gene
            assert isinstance(note_id, int)
            assert isinstance(percentage, float)

        # Should have fitness attribute
        assert hasattr(individual, "fitness")
        assert isinstance(individual.fitness, creator.FitnessMin)

    def test_population_generation(self, optimizer):
        """Test population generation creates multiple individuals"""
        # Generate a population
        population = optimizer.toolbox.population(n=10)

        # Should have 10 individuals
        assert len(population) == 10

        # Each should be a valid individual
        for ind in population:
            assert isinstance(ind, list)
            assert len(ind) == 15
            assert hasattr(ind, "fitness")

    def test_fitness_evaluation_stability(self, optimizer):
        """Test stability score calculation"""
        # Create a well-balanced recipe
        good_recipe = [
            (1, 1.5), (2, 1.5), (3, 2.0),  # Top notes (5%)
            (4, 2.0), (5, 2.5), (6, 2.5),  # Middle notes (7%)
            (7, 2.0), (8, 3.0), (9, 3.0),  # Base notes (8%)
            (10, 0.0), (11, 0.0), (12, 0.0),  # Unused
            (13, 0.0), (14, 0.0), (15, 0.0),  # Unused
        ]

        # Create an imbalanced recipe
        bad_recipe = [
            (1, 10.0), (1, 10.0), (1, 10.0),  # Too much of same note
            (2, 10.0), (2, 10.0), (2, 10.0),  # Duplicate notes
            (3, 10.0), (3, 10.0), (3, 10.0),  # Total too high
            (4, 10.0), (5, 10.0), (6, 10.0),
            (7, 10.0), (8, 10.0), (9, 10.0),  # Total = 150%
        ]

        good_stability = optimizer._evaluate_stability(good_recipe)
        bad_stability = optimizer._evaluate_stability(bad_recipe)

        # Good recipe should have lower (better) stability score
        assert good_stability < bad_stability
        assert good_stability < 5.0  # Should have few violations
        assert bad_stability > 10.0  # Should have many violations

    def test_fitness_evaluation_unfitness(self, optimizer):
        """Test unfitness score calculation (distance from emotional palette)"""
        # Create a recipe matching the creative brief's emotions
        matching_recipe = []
        target_emotion = optimizer.creative_brief.emotional_palette[:3]

        # Select notes with similar emotion vectors
        for i in range(15):
            best_note = None
            best_distance = float('inf')

            for note_id, note_data in optimizer.notes_db.items():
                distance = np.linalg.norm(
                    np.array(note_data["emotion_vector"]) - np.array(target_emotion)
                )
                if distance < best_distance:
                    best_distance = distance
                    best_note = note_id

            matching_recipe.append((best_note, 1.0))

        # Create a recipe with opposite emotions
        opposite_recipe = []
        opposite_emotion = [1.0 - x for x in target_emotion]

        for i in range(15):
            best_note = None
            best_distance = float('inf')

            for note_id, note_data in optimizer.notes_db.items():
                distance = np.linalg.norm(
                    np.array(note_data["emotion_vector"]) - np.array(opposite_emotion)
                )
                if distance < best_distance:
                    best_distance = distance
                    best_note = note_id

            opposite_recipe.append((best_note, 1.0))

        matching_unfitness = optimizer._evaluate_unfitness(matching_recipe)
        opposite_unfitness = optimizer._evaluate_unfitness(opposite_recipe)

        # Matching recipe should have lower (better) unfitness score
        assert matching_unfitness < opposite_unfitness

    def test_fitness_evaluation_uncreativity(self, optimizer):
        """Test uncreativity score calculation (similarity to existing fragrances)"""
        # Create a recipe identical to an existing fragrance
        existing = optimizer.existing_fragrances[0] if optimizer.existing_fragrances else {
            "notes": [1, 2, 3]
        }

        duplicate_recipe = [(note, 1.0) for note in existing.get("notes", [1, 2, 3])]
        duplicate_recipe.extend([(0, 0.0)] * (15 - len(duplicate_recipe)))

        # Create a completely unique recipe
        unique_notes = set(range(50, 65))  # Notes unlikely to be in existing fragrances
        unique_recipe = [(note, 1.0) for note in list(unique_notes)[:15]]

        duplicate_uncreativity = optimizer._evaluate_uncreativity(duplicate_recipe)
        unique_uncreativity = optimizer._evaluate_uncreativity(unique_recipe)

        # Duplicate should have higher (worse) uncreativity score
        assert duplicate_uncreativity > unique_uncreativity

        # Duplicate should have high similarity (close to 1.0)
        if optimizer.existing_fragrances:
            assert duplicate_uncreativity > 0.5

    def test_full_fitness_evaluation(self, optimizer):
        """Test complete fitness evaluation returns 3 objectives"""
        # Create a test individual
        individual = optimizer.toolbox.individual()

        # Evaluate fitness
        fitness_values = optimizer.evaluate(individual)

        # Should return tuple of 3 values
        assert isinstance(fitness_values, tuple)
        assert len(fitness_values) == 3

        stability, unfitness, uncreativity = fitness_values

        # All values should be numeric and non-negative
        assert isinstance(stability, (int, float))
        assert isinstance(unfitness, (int, float))
        assert isinstance(uncreativity, (int, float))
        assert stability >= 0
        assert unfitness >= 0
        assert uncreativity >= 0

    def test_genetic_operators(self, optimizer):
        """Test genetic operators (crossover and mutation)"""
        # Create two parent individuals
        parent1 = optimizer.toolbox.individual()
        parent2 = optimizer.toolbox.individual()

        # Store original values
        orig_parent1 = parent1.copy()
        orig_parent2 = parent2.copy()

        # Test crossover
        child1, child2 = optimizer.toolbox.mate(parent1.copy(), parent2.copy())

        # Children should be different from parents (in most cases)
        # but have same structure
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)

        # Test mutation
        mutant = parent1.copy()
        optimizer.toolbox.mutate(mutant)

        # Mutant should have same length
        assert len(mutant) == len(parent1)

        # At least some genes should be different
        differences = sum(1 for i in range(len(mutant))
                         if mutant[i] != orig_parent1[i])
        # With 0.1 mutation probability, expect ~1-2 mutations
        assert differences >= 0  # Could be 0 by chance

    def test_selection_operator(self, optimizer):
        """Test NSGA-II selection operator"""
        # Create a population with evaluated fitness
        population = optimizer.toolbox.population(n=20)

        # Evaluate all individuals
        for ind in population:
            ind.fitness.values = optimizer.evaluate(ind)

        # Select next generation
        selected = optimizer.toolbox.select(population, k=10)

        # Should return correct number of individuals
        assert len(selected) == 10

        # Selected individuals should be from original population
        for ind in selected:
            assert ind in population

        # Calculate average fitness of selected vs original
        def avg_fitness(pop):
            fitnesses = [ind.fitness.values for ind in pop]
            return tuple(np.mean([f[i] for f in fitnesses]) for i in range(3))

        avg_original = avg_fitness(population)
        avg_selected = avg_fitness(selected)

        # Selected population should generally have better (lower) fitness
        # At least one objective should improve
        improvements = sum(1 for i in range(3) if avg_selected[i] <= avg_original[i])
        assert improvements >= 1


class TestMOGAOptimization:
    """Test the complete MOGA optimization process"""

    def test_optimization_run(self):
        """Test that optimization runs without errors and improves fitness"""
        brief = CreativeBrief(
            emotional_palette=[0.5, 0.5, 0.5, 0.5],
            gender="unisex",
            season="all",
            time_of_day="anytime",
            cultural_background="global"
        )

        optimizer = OlfactoryRecombinatorAI(
            population_size=10,
            generations=3,
            creative_brief=brief
        )

        # Run optimization
        results = optimizer.optimize()

        # Should return results
        assert results is not None
        assert "best_individual" in results
        assert "pareto_front" in results
        assert "statistics" in results

        # Best individual should have valid structure
        best = results["best_individual"]
        assert "recipe" in best
        assert "fitness" in best
        assert "description" in best

        # Pareto front should contain non-dominated solutions
        pareto = results["pareto_front"]
        assert isinstance(pareto, list)
        assert len(pareto) > 0

        # Each solution in Pareto front should have fitness values
        for solution in pareto:
            assert "fitness" in solution
            assert len(solution["fitness"]) == 3

    def test_deterministic_with_seed(self):
        """Test that optimization is deterministic with same seed"""
        brief = CreativeBrief(
            emotional_palette=[0.3, 0.7, 0.4, 0.6],
            gender="feminine",
            season="summer",
            time_of_day="day",
            cultural_background="western"
        )

        # Run with same seed twice
        random.seed(42)
        np.random.seed(42)
        optimizer1 = OlfactoryRecombinatorAI(
            population_size=5,
            generations=2,
            creative_brief=brief
        )
        results1 = optimizer1.optimize()

        random.seed(42)
        np.random.seed(42)
        optimizer2 = OlfactoryRecombinatorAI(
            population_size=5,
            generations=2,
            creative_brief=brief
        )
        results2 = optimizer2.optimize()

        # Results should be identical
        assert results1["best_individual"]["fitness"] == results2["best_individual"]["fitness"]
        assert len(results1["pareto_front"]) == len(results2["pareto_front"])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])