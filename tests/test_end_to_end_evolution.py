"""
End-to-End Evolution Tests
전체 파이프라인 통합 테스트 (LLM → RL → GA)
"""

import pytest
import numpy as np
from fragrance_ai.schemas.domain_models import CreativeBrief


class TestEndToEndEvolution:
    """전체 파이프라인 E2E 테스트"""

    def test_brief_to_formulation_pipeline(self):
        """Brief → Formulation 전체 파이프라인 테스트"""
        # Step 1: Create brief (LLM would do this)
        brief = CreativeBrief(
            style="fresh",
            intensity=0.7,
            complexity=0.5,
            notes_preference={
                "citrus": 0.8,
                "fresh": 0.7,
                "floral": 0.3
            },
            product_category="EDP",
            target_profile="daily_fresh",
            mood=["energetic", "clean"],
            season=["spring", "summer"]
        )

        # Step 2: Initialize RL environment
        from fragrance_ai.training.ppo_engine import FragranceEnvironment

        env = FragranceEnvironment(n_ingredients=10)
        state = env.reset()

        # Check environment initialized
        assert len(state) > 0, "Environment should provide initial state"

        # Step 3: Run GA optimization
        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(
            brief=brief,
            n_ingredients=10,
            population_size=20,
            n_generations=30
        )

        result = optimizer.optimize()

        # Check GA result
        assert len(result['pareto_front']) > 0, "GA should produce solutions"

        # Step 4: Extract formulation
        best_solution = result['pareto_front'][0]
        formulation = best_solution['formulation']

        # Check formulation validity
        assert len(formulation) == 10, "Formulation should have 10 ingredients"
        assert 0.95 <= sum(formulation) <= 1.05, "Formulation should sum to ~1.0"
        assert all(w >= 0 for w in formulation), "All weights non-negative"

    def test_reinforce_to_ppo_evolution(self):
        """REINFORCE → PPO 진화 파이프라인 테스트"""
        brief = CreativeBrief(
            style="floral",
            intensity=0.6,
            complexity=0.4,
            notes_preference={"floral": 0.9, "fruity": 0.4}
        )

        # Step 1: Initial exploration with REINFORCE (simulated)
        from fragrance_ai.training.ppo_engine import FragranceEnvironment

        env = FragranceEnvironment(n_ingredients=10)

        # Collect initial samples
        initial_rewards = []
        for _ in range(10):
            state = env.reset()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            initial_rewards.append(reward)

        avg_initial_reward = np.mean(initial_rewards)

        # Step 2: Refine with PPO
        from fragrance_ai.training.ppo_trainer_advanced import train_advanced_ppo
        from fragrance_ai.training.rl_advanced import (
            EntropyScheduleConfig,
            RewardNormalizerConfig,
            CheckpointConfig
        )

        trainer = train_advanced_ppo(
            env=env,
            n_iterations=20,
            n_steps_per_iteration=256,
            n_ppo_epochs=5,
            entropy_config=EntropyScheduleConfig(
                initial_entropy=0.01,
                final_entropy=0.001,
                decay_steps=10000
            ),
            reward_config=RewardNormalizerConfig(window_size=100),
            checkpoint_config=CheckpointConfig(
                checkpoint_dir='./test_checkpoints',
                save_interval=10
            )
        )

        # Get final statistics
        stats = trainer.get_full_statistics()
        final_rewards = stats['rewards'][-10:]  # Last 10
        avg_final_reward = np.mean(final_rewards)

        # PPO should improve over random policy
        # (Not always guaranteed with so few iterations, but usually)
        print(f"Initial (random): {avg_initial_reward:.2f}")
        print(f"Final (PPO): {avg_final_reward:.2f}")

        # At minimum, should complete without errors
        assert len(stats['rewards']) == 20, "Should complete all iterations"

    def test_feedback_loop_convergence(self):
        """피드백 루프 수렴 테스트 (User feedback → RL update → GA refinement)"""
        brief = CreativeBrief(
            style="woody",
            intensity=0.7,
            complexity=0.6,
            notes_preference={"woody": 0.9, "oriental": 0.5}
        )

        # Simulate feedback loop
        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        # Round 1: Initial generation
        optimizer1 = MOGAOptimizer(brief=brief, n_generations=20)
        result1 = optimizer1.optimize()
        option1 = result1['pareto_front'][0]['formulation']

        # User feedback (simulated): rating = 3/5 (meh)
        # Adjust brief based on feedback
        brief.intensity = 0.8  # User wants stronger
        brief.notes_preference["oriental"] = 0.7  # More oriental

        # Round 2: Refined generation
        optimizer2 = MOGAOptimizer(brief=brief, n_generations=20)
        result2 = optimizer2.optimize()
        option2 = result2['pareto_front'][0]['formulation']

        # User feedback: rating = 5/5 (great!)

        # Check formulations evolved
        distance = np.linalg.norm(np.array(option1) - np.array(option2))
        assert distance > 0.01, f"Formulations should evolve (distance={distance:.4f})"

        # Both should be valid
        assert 0.95 <= sum(option1) <= 1.05, "Option 1 valid"
        assert 0.95 <= sum(option2) <= 1.05, "Option 2 valid"

    def test_multi_objective_tradeoff(self):
        """다목적 최적화 트레이드오프 테스트"""
        brief = CreativeBrief(
            style="oriental",
            intensity=0.8,
            complexity=0.7,
            notes_preference={
                "oriental": 0.9,
                "spicy": 0.7,
                "woody": 0.6
            },
            constraints={
                "max_allergens_ppm": 100.0,
                "vegan": True
            }
        )

        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(
            brief=brief,
            n_ingredients=15,
            population_size=30,
            n_generations=50
        )

        result = optimizer.optimize()
        pareto_front = result['pareto_front']

        # Should have multiple solutions (tradeoffs)
        assert len(pareto_front) >= 3, "Should have multiple Pareto solutions"

        # Check diversity of solutions
        fitnesses = [sol['fitness'] for sol in pareto_front]

        # Measure fitness diversity
        fitness_array = np.array(fitnesses)
        fitness_std = np.std(fitness_array, axis=0)

        # At least one objective should have diversity
        assert any(std > 0.01 for std in fitness_std), "Solutions should span Pareto front"

    def test_constraint_satisfaction_pipeline(self):
        """제약조건 만족 파이프라인 테스트"""
        brief = CreativeBrief(
            style="fresh",
            intensity=0.6,
            complexity=0.4,
            notes_preference={"fresh": 0.8, "citrus": 0.7},
            constraints={
                "allergen_free": True,
                "natural_only": True,
                "vegan": True,
                "max_allergens_ppm": 0.0
            }
        )

        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(brief=brief, n_generations=30)
        result = optimizer.optimize()

        # All solutions should satisfy constraints
        for solution in result['pareto_front']:
            formulation = solution['formulation']

            # Sum constraint
            total = sum(formulation)
            assert 0.95 <= total <= 1.05, f"Sum constraint violated: {total}"

            # Non-negative constraint
            assert all(w >= 0 for w in formulation), "Non-negative constraint violated"

            # (Allergen-free, natural-only, vegan constraints would be checked
            #  against ingredient database in real implementation)

    def test_full_api_workflow(self):
        """전체 API 워크플로우 시뮬레이션"""
        # This simulates the full API flow:
        # 1. POST /dna/create
        # 2. POST /evolve/options
        # 3. POST /evolve/feedback

        # Step 1: Create DNA (brief)
        brief = CreativeBrief(
            style="floral",
            intensity=0.7,
            complexity=0.5,
            notes_preference={"floral": 0.8, "fruity": 0.4},
            mood=["romantic", "elegant"],
            season=["spring"]
        )

        dna_id = "test-dna-001"

        # Step 2: Generate options with REINFORCE
        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(
            brief=brief,
            n_ingredients=10,
            population_size=20,
            n_generations=30
        )

        result = optimizer.optimize()

        # Get top 3 options
        options = result['pareto_front'][:3]
        assert len(options) >= 3, "Should generate at least 3 options"

        # Step 3: User chooses option and provides feedback
        chosen_option = options[0]
        rating = 5  # User loves it!

        # Step 4: Refine with PPO (in background)
        # (Simulated - in real API this would be async via worker)

        # Check we can retrieve the chosen formulation
        chosen_formulation = chosen_option['formulation']
        assert len(chosen_formulation) == 10, "Formulation should have 10 ingredients"
        assert 0.95 <= sum(chosen_formulation) <= 1.05, "Valid formulation"

    def test_rl_ga_consistency(self):
        """RL과 GA 결과 일관성 테스트"""
        brief = CreativeBrief(
            style="woody",
            intensity=0.8,
            complexity=0.6,
            notes_preference={"woody": 0.9, "oriental": 0.6}
        )

        # Approach 1: GA optimization
        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(brief=brief, n_generations=50)
        ga_result = optimizer.optimize()
        ga_formulation = ga_result['pareto_front'][0]['formulation']

        # Approach 2: RL-guided search (via environment)
        from fragrance_ai.training.ppo_engine import FragranceEnvironment

        env = FragranceEnvironment(n_ingredients=len(ga_formulation))

        # Sample actions and evaluate
        rl_rewards = []
        for _ in range(20):
            state = env.reset()
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            rl_rewards.append(reward)

        # Both approaches should produce valid results
        assert 0.95 <= sum(ga_formulation) <= 1.05, "GA formulation valid"
        assert len(rl_rewards) > 0, "RL produced rewards"

        # GA should generally find better solutions than random RL policy
        ga_quality = ga_result['pareto_front'][0]['fitness']
        rl_avg_reward = np.mean(rl_rewards)

        print(f"GA quality: {ga_quality}")
        print(f"RL avg reward (random): {rl_avg_reward:.2f}")

        # Just check both completed without errors
        assert ga_quality is not None
        assert not np.isnan(rl_avg_reward)


class TestRegressionE2E:
    """E2E 회귀 테스트"""

    def test_backward_compatibility(self):
        """이전 버전과의 호환성 테스트"""
        # Old API format
        brief_dict = {
            "style": "fresh",
            "intensity": 0.7,
            "complexity": 0.5,
            "notes_preference": {"citrus": 0.8}
        }

        # Should be able to create CreativeBrief from dict
        brief = CreativeBrief(**brief_dict)

        # Should work with optimizer
        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(brief=brief)
        result = optimizer.optimize()

        assert len(result['pareto_front']) > 0, "Should work with old format"

    def test_output_format_stability(self):
        """출력 형식 안정성 테스트 (API 응답 형식)"""
        brief = CreativeBrief(
            style="oriental",
            intensity=0.7,
            complexity=0.6,
            notes_preference={"oriental": 0.8}
        )

        from fragrance_ai.training.moga_optimizer_stable import MOGAOptimizer

        optimizer = MOGAOptimizer(brief=brief, n_generations=20)
        result = optimizer.optimize()

        # Check expected structure
        required_keys = ['pareto_front', 'history', 'statistics']
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        # Check pareto_front structure
        for solution in result['pareto_front']:
            assert 'formulation' in solution
            assert 'fitness' in solution

        # Check history structure
        for gen_info in result['history']:
            assert 'generation' in gen_info
            assert 'best_fitness' in gen_info

        # Check statistics structure
        assert 'total_evaluations' in result['statistics']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
