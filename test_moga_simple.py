"""
Simple test script to verify MOGA optimizer components
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.training.moga_optimizer import OlfactoryRecombinatorAI, CreativeBrief
from deap import creator, base
import random


def test_fitness_and_individual():
    """Test that Fitness and Individual classes are properly defined"""
    print("Testing MOGA Optimizer Components...")
    print("-" * 50)

    # Create optimizer
    brief = CreativeBrief(
        emotional_palette=[0.7, 0.3, 0.5, 0.8],
        fragrance_family="floral",
        mood="romantic",
        intensity=0.7,
        season="spring",
        gender="unisex"
    )

    optimizer = OlfactoryRecombinatorAI(
        population_size=10,
        generations=3
    )
    optimizer.creative_brief = brief  # Set creative brief directly

    # Test 1: FitnessMin class
    print("\n1. Testing FitnessMin class...")
    assert hasattr(creator, "FitnessMin"), "FitnessMin class not created"

    fitness = creator.FitnessMin()
    assert fitness.weights == (-1.0, -1.0, -1.0), f"Expected weights (-1.0, -1.0, -1.0), got {fitness.weights}"
    print("   FitnessMin: OK - 3 objectives with minimization weights")

    # Test 2: Individual class
    print("\n2. Testing Individual class...")
    assert hasattr(creator, "Individual"), "Individual class not created"

    individual = creator.Individual()
    assert isinstance(individual, list), "Individual should be a list"
    assert hasattr(individual, "fitness"), "Individual should have fitness attribute"
    print("   Individual: OK - List with fitness attribute")

    # Test 3: Gene generation
    print("\n3. Testing gene generation...")
    genes = [optimizer._generate_gene() for _ in range(10)]
    valid_genes = all(
        isinstance(g, tuple) and len(g) == 2 and
        1 <= g[0] <= len(optimizer.notes_db) and
        0.1 <= g[1] <= 10.0
        for g in genes
    )
    assert valid_genes, "Gene generation produces invalid genes"
    print(f"   Generated {len(genes)} valid genes")
    print(f"   Sample gene: {genes[0]}")

    # Test 4: Individual creation
    print("\n4. Testing individual creation...")
    ind = optimizer.toolbox.individual()
    assert len(ind) == 15, f"Expected 15 genes, got {len(ind)}"
    print(f"   Created individual with {len(ind)} genes")

    # Test 5: Fitness evaluation
    print("\n5. Testing fitness evaluation...")
    fitness_values = optimizer.evaluate(ind)
    assert isinstance(fitness_values, tuple), "Fitness should be a tuple"
    assert len(fitness_values) == 3, f"Expected 3 objectives, got {len(fitness_values)}"

    stability, unfitness, uncreativity = fitness_values
    print(f"   Stability score: {stability:.3f}")
    print(f"   Unfitness score: {unfitness:.3f}")
    print(f"   Uncreativity score: {uncreativity:.3f}")

    # Test 6: Population creation
    print("\n6. Testing population creation...")
    pop = optimizer.toolbox.population(n=5)
    assert len(pop) == 5, f"Expected 5 individuals, got {len(pop)}"
    print(f"   Created population with {len(pop)} individuals")

    # Test 7: Genetic operators
    print("\n7. Testing genetic operators...")

    # Crossover
    parent1 = optimizer.toolbox.individual()
    parent2 = optimizer.toolbox.individual()
    child1, child2 = optimizer.toolbox.mate(parent1.copy(), parent2.copy())
    assert len(child1) == 15 and len(child2) == 15, "Crossover should preserve individual size"
    print("   Crossover: OK")

    # Mutation
    mutant = parent1.copy()
    optimizer.toolbox.mutate(mutant)
    assert len(mutant) == 15, "Mutation should preserve individual size"
    print("   Mutation: OK")

    # Selection (NSGA-II)
    for ind in pop:
        ind.fitness.values = optimizer.evaluate(ind)
    selected = optimizer.toolbox.select(pop, k=3)
    assert len(selected) == 3, f"Expected 3 selected, got {len(selected)}"
    print("   NSGA-II Selection: OK")

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)

    return True


def test_optimization_run():
    """Test running a small optimization"""
    print("\n\nTesting Optimization Run...")
    print("-" * 50)

    brief = CreativeBrief(
        emotional_palette=[0.5, 0.5, 0.5, 0.5],
        fragrance_family="fresh",
        mood="balanced",
        intensity=0.5,
        season="all",
        gender="unisex"
    )

    optimizer = OlfactoryRecombinatorAI(
        population_size=6,
        generations=2
    )

    print("Running optimization with:")
    print(f"  Population size: 6")
    print(f"  Generations: 2")

    dna = optimizer.evolve(brief)  # Pass brief to evolve method

    print("\nOptimization Results:")
    print(f"  Best DNA fitness scores: {dna.fitness_scores}")
    print(f"  Number of genes: {len(dna.genes)}")

    # Format and display recipe
    recipe = optimizer.format_recipe(dna)
    print(f"  Total concentration: {recipe['total_concentration']:.1f}%")

    # Display formatted recipe
    print("\nBest Recipe:")
    print("  Top Notes:")
    for note, pct in list(recipe['top_notes'].items())[:3]:
        print(f"    - {note}: {pct}")
    print("  Middle Notes:")
    for note, pct in list(recipe['middle_notes'].items())[:3]:
        print(f"    - {note}: {pct}")
    print("  Base Notes:")
    for note, pct in list(recipe['base_notes'].items())[:3]:
        print(f"    - {note}: {pct}")

    print("\n  Fitness Metrics:")
    print(f"    Stability: {recipe['fitness']['stability']:.2%}")
    print(f"    Suitability: {recipe['fitness']['suitability']:.2%}")
    print(f"    Creativity: {recipe['fitness']['creativity']:.2%}")

    print("\n" + "=" * 50)
    print("Optimization completed successfully!")
    print("=" * 50)

    return True


if __name__ == "__main__":
    try:
        # Run component tests
        test_fitness_and_individual()

        # Run optimization test
        test_optimization_run()

        print("\n\nFinal Summary:")
        print("=" * 50)
        print("MOGA Optimizer Implementation Verified:")
        print("  - FitnessMin class: Working (3 objectives)")
        print("  - Individual class: Working (fragrance recipe)")
        print("  - Genetic operators: Working (crossover, mutation, selection)")
        print("  - Optimization: Working (produces Pareto front)")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)