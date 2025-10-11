# tests/test_rl_ppo_complete.py

import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from fragrance_ai.training.reinforcement_learning import RLEngine
from fragrance_ai.training.reinforcement_learning_ppo import RLEnginePPO
from fragrance_ai.training.moga_optimizer import UnifiedProductionMOGA
from fragrance_ai.utils.config_loader import ConfigLoader
from fragrance_ai.utils.creativity_metrics import CreativityMetrics
from fragrance_ai.database.models import OlfactoryDNA, ScentPhenotype


class TestRLImplementation(unittest.TestCase):
    """Test suite for RL implementation (REINFORCE and PPO)"""

    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 20
        self.action_dim = 12

        # Create test DNA
        self.test_dna = OlfactoryDNA(
            dna_id="test_dna_001",
            genotype={"base_recipe": "test"},
            notes=[
                {"name": "bergamot", "intensity": 0.3},
                {"name": "rose", "intensity": 0.5},
                {"name": "sandalwood", "intensity": 0.2}
            ]
        )

        # Create test creative brief
        self.test_brief = {
            "desired_intensity": 0.7,
            "masculinity": 0.4,
            "complexity": 0.6,
            "theme": "modern elegance",
            "story": "A sophisticated evening fragrance"
        }

    def test_reinforce_initialization(self):
        """Test REINFORCE engine initialization"""
        engine = RLEngine(self.state_dim, self.action_dim)

        self.assertIsNotNone(engine.policy_network)
        self.assertIsNotNone(engine.optimizer)
        self.assertEqual(len(engine.action_space), 6)  # Default action space size

        print("✓ REINFORCE initialization successful")

    def test_ppo_initialization(self):
        """Test PPO engine initialization"""
        engine = RLEnginePPO(self.state_dim, self.action_dim, algorithm="PPO")

        self.assertIsNotNone(engine.policy_network)
        self.assertIsNotNone(engine.value_network)
        self.assertIsNotNone(engine.optimizer)
        self.assertIsNotNone(engine.memory)

        print("✓ PPO initialization successful")

    def test_state_encoding(self):
        """Test state encoding functionality"""
        engine = RLEngine(self.state_dim, self.action_dim)
        state = engine.encode_state(self.test_dna, self.test_brief)

        self.assertEqual(state.shape, (1, self.state_dim))
        self.assertTrue(torch.is_tensor(state))
        self.assertFalse(torch.isnan(state).any())

        print("✓ State encoding successful")

    def test_generate_variations(self):
        """Test variation generation"""
        engine = RLEngine(self.state_dim, self.action_dim)
        options = engine.generate_variations(self.test_dna, self.test_brief, num_options=3)

        self.assertEqual(len(options), 3)
        for option in options:
            self.assertIn("id", option)
            self.assertIn("phenotype", option)
            self.assertIn("action", option)
            self.assertIn("action_name", option)
            self.assertIn("log_prob", option)

        # Check that state and actions are saved
        self.assertIsNotNone(engine.last_state)
        self.assertIsNotNone(engine.last_saved_actions)
        self.assertEqual(len(engine.last_saved_actions), 3)

        print("✓ Variation generation successful")

    def test_reinforce_update(self):
        """Test REINFORCE policy update"""
        engine = RLEngine(self.state_dim, self.action_dim)

        # Generate variations first
        options = engine.generate_variations(self.test_dna, self.test_brief, num_options=3)

        # Select first option
        chosen_id = options[0]["id"]

        # Update policy with feedback
        result = engine.update_policy_with_feedback(
            chosen_phenotype_id=chosen_id,
            options=options,
            state=engine.last_state,
            saved_actions=engine.last_saved_actions,
            rating=4.0  # Good rating
        )

        self.assertIn("loss", result)
        self.assertIn("reward", result)
        self.assertIn("algorithm", result)
        self.assertEqual(result["algorithm"], "REINFORCE")

        # Check reward calculation
        expected_reward = (4.0 - 3) / 2.0  # (rating - 3) / 2
        self.assertAlmostEqual(result["reward"], expected_reward, places=4)

        print(f"✓ REINFORCE update successful - Loss: {result['loss']:.4f}, Reward: {result['reward']:.2f}")

    def test_ppo_update(self):
        """Test PPO policy update"""
        engine = RLEnginePPO(self.state_dim, self.action_dim, algorithm="PPO")

        # Generate multiple experiences to fill memory
        for i in range(5):
            options = engine.generate_variations(self.test_dna, self.test_brief, num_options=3)
            chosen_id = options[i % 3]["id"]

            result = engine.update_policy_with_feedback(
                chosen_phenotype_id=chosen_id,
                options=options,
                rating=3.0 + (i % 3)  # Varying ratings
            )

            # First few updates might just buffer
            if "status" in result and result["status"] == "buffering":
                print(f"  Buffering experience {i+1}/5")
            else:
                self.assertIn("loss", result)
                self.assertIn("algorithm", result)
                self.assertEqual(result["algorithm"], "PPO")
                print(f"✓ PPO update {i+1} - Loss: {result['loss']:.4f}")

    def test_algorithm_switching(self):
        """Test switching between REINFORCE and PPO"""
        # Test REINFORCE
        reinforce_engine = RLEngine(self.state_dim, self.action_dim)
        options_r = reinforce_engine.generate_variations(self.test_dna, self.test_brief)
        result_r = reinforce_engine.update_policy_with_feedback(
            options_r[0]["id"], options_r, reinforce_engine.last_state,
            reinforce_engine.last_saved_actions
        )
        self.assertEqual(result_r["algorithm"], "REINFORCE")

        # Test PPO
        ppo_engine = RLEnginePPO(self.state_dim, self.action_dim, algorithm="PPO")
        self.assertEqual(ppo_engine.algorithm, "PPO")

        print("✓ Algorithm switching successful")


class TestGAStability(unittest.TestCase):
    """Test suite for GA mutation and crossover stability"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = UnifiedProductionMOGA(population_size=10, generations=5)

    def test_mutation_positive_values(self):
        """Test that mutation always produces positive values"""
        # Create test individual
        individual = [(1, 20.0), (2, 30.0), (3, 50.0)]

        # Run mutation many times
        for _ in range(100):
            mutated = self.optimizer.polynomial_mutation(individual)[0]

            # Check all concentrations are positive
            for ing_id, concentration in mutated:
                self.assertGreater(concentration, 0, "Mutation produced negative concentration")

            # Check normalization (should sum to ~100)
            total = sum(c for _, c in mutated)
            self.assertAlmostEqual(total, 100.0, delta=1.0)

        print("✓ Mutation stability test passed (100 iterations)")

    def test_crossover_validity(self):
        """Test that crossover produces valid offspring"""
        # Create parent individuals
        parent1 = [(1, 25.0), (2, 35.0), (3, 40.0)]
        parent2 = [(1, 30.0), (3, 45.0), (4, 25.0)]

        # Run crossover many times
        for _ in range(50):
            child1, child2 = self.optimizer.sbx_crossover(parent1, parent2)

            # Check both children
            for child in [child1, child2]:
                # All concentrations positive
                for ing_id, concentration in child:
                    self.assertGreater(concentration, 0)

                # Sum to 100
                total = sum(c for _, c in child)
                self.assertAlmostEqual(total, 100.0, delta=1.0)

        print("✓ Crossover stability test passed (50 iterations)")

    def test_ifra_limits_respected(self):
        """Test that IFRA limits are always respected"""
        # Create individual with high concentrations
        individual = [(1, 50.0), (2, 30.0), (3, 20.0)]

        # Normalize with IFRA checking
        normalized = self.optimizer.normalize_concentrations(individual)

        # Check IFRA limits
        for ing_id, concentration in normalized:
            ing_data = next((i for i in self.optimizer.ingredients if i['id'] == ing_id), None)
            if ing_data and 'ifra_limit' in ing_data:
                self.assertLessEqual(concentration, ing_data['ifra_limit'],
                                   f"IFRA limit exceeded for ingredient {ing_id}")

        print("✓ IFRA limits test passed")


class TestCreativityMetrics(unittest.TestCase):
    """Test suite for creativity metrics and defensive programming"""

    def setUp(self):
        """Set up test fixtures"""
        self.metrics = CreativityMetrics()

    def test_entropy_calculation(self):
        """Test entropy calculation with various inputs"""
        # Uniform distribution (high entropy)
        uniform = [20.0, 20.0, 20.0, 20.0, 20.0]
        entropy_uniform = self.metrics.calculate_entropy(uniform)
        self.assertGreater(entropy_uniform, 0.9)

        # Skewed distribution (low entropy)
        skewed = [90.0, 2.5, 2.5, 2.5, 2.5]
        entropy_skewed = self.metrics.calculate_entropy(skewed)
        self.assertLess(entropy_skewed, 0.5)

        # Edge cases
        self.assertEqual(self.metrics.calculate_entropy([]), 0.0)
        self.assertEqual(self.metrics.calculate_entropy([100.0]), 0.0)

        print("✓ Entropy calculation test passed")

    def test_nan_handling(self):
        """Test handling of NaN and inf values"""
        # Test with NaN
        concentrations_nan = [20.0, float('nan'), 30.0, 25.0]
        cleaned = self.metrics.validate_concentrations(concentrations_nan)
        self.assertEqual(len(cleaned), 3)  # NaN should be removed

        # Test with inf
        concentrations_inf = [20.0, float('inf'), 30.0, 25.0]
        cleaned_inf = self.metrics.validate_concentrations(concentrations_inf)
        self.assertEqual(len(cleaned_inf), 3)  # inf should be removed

        # Verify no NaN in output
        for value in cleaned:
            self.assertTrue(np.isfinite(value))

        print("✓ NaN/inf handling test passed")

    def test_category_balance(self):
        """Test category balance calculation"""
        # Perfect balance
        perfect_formulation = {
            'ingredients': [
                {'category': 'top', 'concentration': 25},
                {'category': 'heart', 'concentration': 40},
                {'category': 'base', 'concentration': 35}
            ]
        }
        balance = self.metrics.calculate_category_balance(perfect_formulation)
        self.assertGreater(balance, 0.95)

        # Imbalanced
        imbalanced_formulation = {
            'ingredients': [
                {'category': 'top', 'concentration': 70},
                {'category': 'heart', 'concentration': 20},
                {'category': 'base', 'concentration': 10}
            ]
        }
        balance_imbalanced = self.metrics.calculate_category_balance(imbalanced_formulation)
        self.assertLess(balance_imbalanced, 0.5)

        print("✓ Category balance test passed")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""

    def test_end_to_end_rl_flow(self):
        """Test complete RL flow from generation to update"""
        print("\n=== End-to-End RL Flow Test ===")

        # Initialize engine with PPO
        engine = RLEnginePPO(20, 12, algorithm="PPO")

        # Create test DNA and brief
        dna = OlfactoryDNA(
            dna_id="integration_test",
            genotype={"test": "data"},
            notes=[{"name": f"note_{i}", "intensity": 0.1 * i} for i in range(1, 6)]
        )

        brief = {
            "desired_intensity": 0.8,
            "theme": "integration test",
            "story": "Testing complete flow"
        }

        # Simulate user interaction loop
        total_reward = 0
        for episode in range(10):
            # Generate options
            options = engine.generate_variations(dna, brief, num_options=3)

            # Simulate user selection (choose based on simple heuristic)
            scores = [np.random.random() for _ in options]
            best_idx = np.argmax(scores)
            chosen_id = options[best_idx]["id"]

            # Simulate user rating
            rating = 2 + scores[best_idx] * 3  # Rating between 2 and 5

            # Update policy
            result = engine.update_policy_with_feedback(
                chosen_phenotype_id=chosen_id,
                options=options,
                rating=rating
            )

            if "reward" in result:
                total_reward += result["reward"]
                print(f"Episode {episode+1}: Rating={rating:.1f}, Reward={result['reward']:.2f}")

        avg_reward = total_reward / 10
        print(f"\nAverage reward over 10 episodes: {avg_reward:.3f}")
        self.assertIsNotNone(avg_reward)

    def test_ga_evolution_stability(self):
        """Test GA evolution over multiple generations"""
        print("\n=== GA Evolution Stability Test ===")

        optimizer = UnifiedProductionMOGA(population_size=20, generations=10)

        # Track IFRA violations and negative values
        violations = []
        negative_values = []

        # Create initial population
        population = [optimizer.create_individual() for _ in range(20)]

        for generation in range(10):
            # Evaluate population
            for ind in population:
                fitness = optimizer.evaluate_fragrance(ind)
                ind.fitness.values = fitness

            # Select parents
            parents = optimizer.toolbox.select(population, len(population))

            # Apply crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i+1 < len(parents):
                    child1, child2 = optimizer.sbx_crossover(parents[i], parents[i+1])
                    child1 = optimizer.polynomial_mutation(child1)[0]
                    child2 = optimizer.polynomial_mutation(child2)[0]
                    offspring.extend([child1, child2])

            # Check for violations
            gen_violations = 0
            gen_negatives = 0

            for ind in offspring:
                for ing_id, concentration in ind:
                    if concentration < 0:
                        gen_negatives += 1

                    ing_data = next((i for i in optimizer.ingredients if i['id'] == ing_id), None)
                    if ing_data and concentration > ing_data.get('ifra_limit', 100):
                        gen_violations += 1

            violations.append(gen_violations)
            negative_values.append(gen_negatives)

            # Replace population
            population = offspring

        print(f"IFRA violations per generation: {violations}")
        print(f"Negative values per generation: {negative_values}")
        print(f"Total IFRA violations: {sum(violations)}")
        print(f"Total negative values: {sum(negative_values)}")

        # Assert no negative values
        self.assertEqual(sum(negative_values), 0, "GA produced negative concentrations")


def run_smoke_test():
    """Run a simple smoke test for quick validation"""
    print("\n" + "="*50)
    print("SMOKE TEST - Quick Validation")
    print("="*50)

    # 1. Create RL engine
    print("\n1. Creating RL engine...")
    engine = RLEnginePPO(state_dim=20, action_dim=12, algorithm="PPO")
    print("   ✓ Engine created")

    # 2. Create test DNA
    print("\n2. Creating test DNA...")
    dna = OlfactoryDNA(
        dna_id="smoke_test",
        genotype={"type": "test"},
        notes=[
            {"name": "bergamot", "intensity": 0.3},
            {"name": "jasmine", "intensity": 0.4},
            {"name": "musk", "intensity": 0.3}
        ]
    )
    print("   ✓ DNA created")

    # 3. Generate variations
    print("\n3. Generating variations...")
    brief = {"theme": "smoke test", "desired_intensity": 0.7}
    options = engine.generate_variations(dna, brief, num_options=3)
    print(f"   ✓ Generated {len(options)} variations:")
    for i, opt in enumerate(options):
        print(f"     - Option {i+1}: {opt['action_name']}")

    # 4. Simulate selection and update
    print("\n4. Simulating user selection and learning...")
    chosen = options[1]  # Select middle option
    result = engine.update_policy_with_feedback(
        chosen_phenotype_id=chosen["id"],
        options=options,
        rating=4.0
    )

    if "loss" in result:
        print(f"   ✓ Learning complete - Loss: {result['loss']:.4f}, Reward: {result['reward']:.2f}")
    else:
        print(f"   ✓ Experience buffered - {result}")

    print("\n" + "="*50)
    print("SMOKE TEST PASSED")
    print("="*50)


if __name__ == "__main__":
    # Run smoke test first
    run_smoke_test()

    # Run full test suite
    print("\n" + "="*50)
    print("RUNNING FULL TEST SUITE")
    print("="*50)

    unittest.main(verbosity=2)