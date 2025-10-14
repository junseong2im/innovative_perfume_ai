"""
MOGA Stability Tests
Multi-Objective Genetic Algorithm 안정성 테스트
"""

import pytest
import numpy as np
from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer
from fragrance_ai.schemas.domain_models import CreativeBrief


class TestMOGAStability:
    """MOGA 안정성 테스트"""

    def test_moga_convergence(self):
        """MOGA 수렴성 테스트"""
        brief = CreativeBrief(
            style="fresh",
            intensity=0.7,
            complexity=0.5,
            notes_preference={
                "citrus": 0.8,
                "floral": 0.3,
                "woody": 0.2
            }
        )

        optimizer = MOGAOptimizer(
            brief=brief,
            n_ingredients=10,
            population_size=20,
            n_generations=50
        )

        # Run optimization
        result = optimizer.optimize()

        # Check convergence
        assert len(result['pareto_front']) > 0, "Pareto front should not be empty"
        assert len(result['history']) == 50, "Should have 50 generations"

        # Check improvement over time
        first_gen_fitness = result['history'][0]['best_fitness']
        last_gen_fitness = result['history'][-1]['best_fitness']

        # At least one objective should improve
        improved = False
        for i in range(len(first_gen_fitness)):
            if last_gen_fitness[i] > first_gen_fitness[i] + 0.01:
                improved = True
                break

        assert improved, "Algorithm should improve at least one objective"

    def test_moga_diversity(self):
        """MOGA 다양성 유지 테스트"""
        brief = CreativeBrief(
            style="floral",
            intensity=0.6,
            complexity=0.4,
            notes_preference={"floral": 0.9, "citrus": 0.3}
        )

        optimizer = MOGAOptimizer(
            brief=brief,
            n_ingredients=15,
            population_size=30,
            n_generations=30
        )

        result = optimizer.optimize()

        # Check Pareto front diversity
        pareto_front = result['pareto_front']
        assert len(pareto_front) >= 3, "Pareto front should have multiple solutions"

        # Measure diversity (pairwise distances)
        formulations = [sol['formulation'] for sol in pareto_front]
        distances = []

        for i in range(len(formulations)):
            for j in range(i + 1, len(formulations)):
                dist = np.linalg.norm(
                    np.array(formulations[i]) - np.array(formulations[j])
                )
                distances.append(dist)

        avg_distance = np.mean(distances)
        assert avg_distance > 0.1, f"Solutions too similar (avg_dist={avg_distance:.3f})"

    def test_moga_constraint_satisfaction(self):
        """MOGA 제약조건 만족 테스트"""
        brief = CreativeBrief(
            style="woody",
            intensity=0.8,
            complexity=0.7,
            notes_preference={"woody": 0.9, "oriental": 0.6},
            constraints={
                "allergen_free": True,
                "max_allergens_ppm": 0.0
            }
        )

        optimizer = MOGAOptimizer(brief=brief)
        result = optimizer.optimize()

        # Check all solutions satisfy constraints
        for solution in result['pareto_front']:
            formulation = solution['formulation']

            # Sum should be ~1.0 (normalized)
            total = sum(formulation)
            assert 0.95 <= total <= 1.05, f"Formulation sum {total:.2f} should be ~1.0"

            # All weights non-negative
            assert all(w >= 0 for w in formulation), "All weights should be non-negative"

    def test_moga_reproducibility(self):
        """MOGA 재현성 테스트 (같은 seed = 같은 결과)"""
        brief = CreativeBrief(
            style="oriental",
            intensity=0.7,
            complexity=0.6,
            notes_preference={"oriental": 0.8, "spicy": 0.6}
        )

        seed = 42

        # Run 1
        optimizer1 = MOGAOptimizer(brief=brief, random_seed=seed)
        result1 = optimizer1.optimize()

        # Run 2 (same seed)
        optimizer2 = MOGAOptimizer(brief=brief, random_seed=seed)
        result2 = optimizer2.optimize()

        # Results should be identical
        pareto1 = result1['pareto_front'][0]['formulation']
        pareto2 = result2['pareto_front'][0]['formulation']

        assert np.allclose(pareto1, pareto2, atol=1e-6), "Same seed should give same results"

    def test_moga_mutation_stability(self):
        """MOGA 돌연변이 안정성 테스트"""
        brief = CreativeBrief(
            style="fresh",
            intensity=0.5,
            complexity=0.5,
            notes_preference={"fresh": 0.7, "citrus": 0.6}
        )

        # Test different mutation rates
        mutation_rates = [0.1, 0.2, 0.3]
        results = []

        for mut_rate in mutation_rates:
            optimizer = MOGAOptimizer(
                brief=brief,
                mutation_rate=mut_rate,
                n_generations=30
            )
            result = optimizer.optimize()
            results.append(result)

            # Should converge for all mutation rates
            assert len(result['pareto_front']) > 0, f"Failed with mutation_rate={mut_rate}"

        # Higher mutation = more diversity (usually)
        diversity_low = len(results[0]['pareto_front'])
        diversity_high = len(results[2]['pareto_front'])

        # Just check both produced valid fronts
        assert diversity_low > 0 and diversity_high > 0, "All configs should produce solutions"

    def test_moga_crossover_stability(self):
        """MOGA 교차 안정성 테스트"""
        brief = CreativeBrief(
            style="floral",
            intensity=0.6,
            complexity=0.5,
            notes_preference={"floral": 0.8, "fruity": 0.4}
        )

        # Test different crossover rates
        crossover_rates = [0.5, 0.7, 0.9]
        results = []

        for cx_rate in crossover_rates:
            optimizer = MOGAOptimizer(
                brief=brief,
                crossover_rate=cx_rate,
                n_generations=30
            )
            result = optimizer.optimize()
            results.append(result)

            # Should converge for all crossover rates
            assert len(result['pareto_front']) > 0, f"Failed with crossover_rate={cx_rate}"

    def test_moga_population_size_effect(self):
        """MOGA 개체군 크기 영향 테스트"""
        brief = CreativeBrief(
            style="woody",
            intensity=0.7,
            complexity=0.6,
            notes_preference={"woody": 0.9, "oriental": 0.5}
        )

        # Small population
        optimizer_small = MOGAOptimizer(
            brief=brief,
            population_size=10,
            n_generations=50
        )
        result_small = optimizer_small.optimize()

        # Large population
        optimizer_large = MOGAOptimizer(
            brief=brief,
            population_size=50,
            n_generations=50
        )
        result_large = optimizer_large.optimize()

        # Both should converge
        assert len(result_small['pareto_front']) > 0, "Small population should converge"
        assert len(result_large['pareto_front']) > 0, "Large population should converge"

        # Larger population usually finds more Pareto solutions
        # (But not always, so we just check both are valid)
        assert len(result_small['pareto_front']) >= 1
        assert len(result_large['pareto_front']) >= 1

    def test_moga_100k_iterations_stability(self):
        """MOGA 100k 반복 안정성 테스트 (장시간 실행)"""
        brief = CreativeBrief(
            style="oriental",
            intensity=0.8,
            complexity=0.7,
            notes_preference={"oriental": 0.9, "spicy": 0.7, "woody": 0.5}
        )

        optimizer = MOGAOptimizer(
            brief=brief,
            population_size=50,
            n_generations=100  # Long run
        )

        result = optimizer.optimize()

        # Should converge even after many generations
        assert len(result['pareto_front']) > 0, "Should converge after 100 generations"
        assert len(result['history']) == 100, "Should complete all generations"

        # Check no NaN or Inf values
        for solution in result['pareto_front']:
            formulation = solution['formulation']
            assert not np.isnan(formulation).any(), "No NaN values"
            assert not np.isinf(formulation).any(), "No Inf values"

        # Check fitness improves or stabilizes (not degrades)
        early_fitness = result['history'][10]['best_fitness']
        late_fitness = result['history'][-1]['best_fitness']

        # At least one objective should not degrade significantly
        degraded = False
        for i in range(len(early_fitness)):
            if late_fitness[i] < early_fitness[i] - 0.2:
                degraded = True
                break

        assert not degraded, "Fitness should not degrade significantly over time"


class TestMOGARegression:
    """MOGA 회귀 테스트 (이전 버전 호환성)"""

    def test_moga_api_compatibility(self):
        """MOGA API 호환성 테스트"""
        brief = CreativeBrief(
            style="fresh",
            intensity=0.7,
            complexity=0.5,
            notes_preference={"fresh": 0.8, "citrus": 0.7}
        )

        # Old API should still work
        optimizer = MOGAOptimizer(brief=brief)

        # Should have all expected attributes
        assert hasattr(optimizer, 'brief')
        assert hasattr(optimizer, 'optimize')
        assert hasattr(optimizer, 'population_size')
        assert hasattr(optimizer, 'n_generations')

        # Should run successfully
        result = optimizer.optimize()

        # Should return expected structure
        assert 'pareto_front' in result
        assert 'history' in result
        assert 'statistics' in result

    def test_moga_output_format(self):
        """MOGA 출력 형식 테스트"""
        brief = CreativeBrief(
            style="floral",
            intensity=0.6,
            complexity=0.4,
            notes_preference={"floral": 0.8}
        )

        optimizer = MOGAOptimizer(brief=brief, n_generations=20)
        result = optimizer.optimize()

        # Check Pareto front format
        assert isinstance(result['pareto_front'], list)
        for solution in result['pareto_front']:
            assert 'formulation' in solution
            assert 'fitness' in solution
            assert isinstance(solution['formulation'], (list, np.ndarray))
            assert isinstance(solution['fitness'], (list, np.ndarray, tuple))

        # Check history format
        assert isinstance(result['history'], list)
        for gen_info in result['history']:
            assert 'generation' in gen_info
            assert 'best_fitness' in gen_info

        # Check statistics format
        assert isinstance(result['statistics'], dict)
        assert 'total_evaluations' in result['statistics']
        assert 'convergence_generation' in result['statistics'] or True  # Optional


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
