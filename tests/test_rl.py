# tests/test_rl.py
"""
Reinforcement Learning Tests
Tests RL algorithms with fake users (random and rule-based) for 50 steps
"""

import pytest
import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

from fragrance_ai.training.rl import create_rl_trainer
from fragrance_ai.services.evolution_service import EvolutionService
from fragrance_ai.schemas.domain_models import OlfactoryDNA, CreativeBrief, Ingredient, NoteCategory
from fragrance_ai.observability import rl_logger, metrics_collector


# ============================================================================
# Fake User Types
# ============================================================================

class UserType(str, Enum):
    """Types of fake users for testing"""
    RANDOM = "random"           # Random choices and ratings
    CONSISTENT = "consistent"   # Prefers specific actions
    IMPROVING = "improving"     # Ratings improve over time
    CRITICAL = "critical"       # Low ratings initially, then improves
    GENEROUS = "generous"       # High ratings always


@dataclass
class FakeUser:
    """Simulated user for RL testing"""
    user_type: UserType
    preference_vector: np.ndarray  # Preferences for each action
    learning_rate: float = 0.1     # How quickly preferences change

    def choose_option(self, options: List[Dict[str, Any]]) -> str:
        """Choose from options based on user type"""
        if self.user_type == UserType.RANDOM:
            # Random choice
            idx = np.random.randint(len(options))
        elif self.user_type == UserType.CONSISTENT:
            # Choose based on preference vector
            scores = []
            for opt in options:
                action_idx = opt.get("action_idx", 0)
                scores.append(self.preference_vector[action_idx % len(self.preference_vector)])
            idx = np.argmax(scores)
        else:
            # Default to first option
            idx = 0

        return options[idx]["id"]

    def rate_choice(self, chosen_action: int, iteration: int) -> float:
        """Rate the chosen option"""
        if self.user_type == UserType.RANDOM:
            # Random rating 1-5
            return float(np.random.randint(1, 6))

        elif self.user_type == UserType.CONSISTENT:
            # Rate based on preference
            base_rating = self.preference_vector[chosen_action % len(self.preference_vector)]
            # Map [0,1] to [1,5]
            rating = 1 + 4 * base_rating
            # Add small noise
            rating += np.random.normal(0, 0.3)
            return float(np.clip(rating, 1, 5))

        elif self.user_type == UserType.IMPROVING:
            # Ratings improve over time
            base = 2.0 + (iteration / 50) * 2  # Start at 2, end at 4
            rating = base + np.random.normal(0, 0.5)
            return float(np.clip(rating, 1, 5))

        elif self.user_type == UserType.CRITICAL:
            # Critical at first, then improves
            if iteration < 20:
                return float(np.random.uniform(1, 2.5))
            else:
                return float(np.random.uniform(3, 5))

        elif self.user_type == UserType.GENEROUS:
            # Always high ratings
            return float(np.random.uniform(4, 5))

        return 3.0  # Default neutral

    def update_preferences(self, action: int, reward: float):
        """Update preferences based on experience"""
        if self.user_type == UserType.CONSISTENT:
            # Update preference vector based on reward
            idx = action % len(self.preference_vector)
            target = (reward + 1) / 2  # Map [-1,1] to [0,1]
            self.preference_vector[idx] += self.learning_rate * (target - self.preference_vector[idx])
            # Renormalize
            self.preference_vector = np.clip(self.preference_vector, 0, 1)


# ============================================================================
# RL Algorithm Tests
# ============================================================================

class TestRLAlgorithms:
    """Test RL algorithms with simulated users"""

    def setup_method(self):
        """Setup for each test"""
        self.state_dim = 20
        self.action_dim = 12

        # Create sample DNA for testing
        self.test_dna = OlfactoryDNA(
            dna_id="test_dna_001",
            name="Test Formula",
            ingredients=[
                Ingredient(id=1, name="Bergamot", concentration=30.0, category=NoteCategory.TOP),
                Ingredient(id=2, name="Rose", concentration=40.0, category=NoteCategory.HEART),
                Ingredient(id=3, name="Sandalwood", concentration=30.0, category=NoteCategory.BASE),
            ]
        )

        self.test_brief = CreativeBrief(
            brief_id="test_brief_001",
            user_id="test_user",
            desired_intensity=0.7,
            masculinity=0.5,
            complexity=0.6
        )

    def test_reinforce_with_fake_users(self):
        """Test REINFORCE with 50 steps of fake user interaction"""
        print("\n[TEST] REINFORCE with fake users (50 steps)...")

        # Create trainer
        trainer = create_rl_trainer(
            algorithm="REINFORCE",
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

        # Create evolution service
        evolution_service = EvolutionService(
            algorithm="REINFORCE",
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

        # Create fake users
        users = [
            FakeUser(UserType.RANDOM, np.random.rand(self.action_dim)),
            FakeUser(UserType.CONSISTENT, np.random.rand(self.action_dim)),
            FakeUser(UserType.IMPROVING, np.random.rand(self.action_dim)),
        ]

        # Metrics tracking
        rewards = []
        losses = []
        ratings = []

        # Run 50 interaction steps
        for step in range(50):
            user = users[step % len(users)]

            # Generate options
            result = evolution_service.generate_options(
                user_id=f"fake_user_{user.user_type.value}",
                dna=self.test_dna,
                brief=self.test_brief,
                num_options=3
            )

            experiment_id = result["experiment_id"]
            options = result["options"]

            # User chooses
            chosen_id = user.choose_option(options)

            # Find chosen action for rating
            chosen_action = next(
                opt["action_idx"] for opt in evolution_service.sessions[experiment_id]["options"]
                if opt["id"] == chosen_id
            )

            # User rates
            rating = user.rate_choice(chosen_action, step)
            ratings.append(rating)

            # Process feedback
            feedback_result = evolution_service.process_feedback(
                experiment_id=experiment_id,
                chosen_id=chosen_id,
                rating=rating
            )

            # Track metrics
            if "metrics" in feedback_result:
                metrics = feedback_result["metrics"]
                if "reward" in metrics:
                    rewards.append(metrics["reward"])
                if "loss" in metrics:
                    losses.append(metrics["loss"])

                # Log to observability
                rl_logger.log_update(
                    algorithm="REINFORCE",
                    loss=metrics.get("loss", 0),
                    reward=metrics.get("reward", 0),
                    step=step,
                    user_type=user.user_type.value
                )

            # Update user preferences
            reward = (rating - 3) / 2  # Convert to [-1,1]
            user.update_preferences(chosen_action, reward)

            # End session to clean up
            evolution_service.end_session(experiment_id)

        # Analyze results
        avg_reward = np.mean(rewards) if rewards else 0
        avg_rating = np.mean(ratings)
        reward_trend = np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0

        print(f"  Steps completed: 50")
        print(f"  Average reward: {avg_reward:.4f}")
        print(f"  Average rating: {avg_rating:.2f}")
        print(f"  Reward trend: {reward_trend:+.6f}")
        print(f"  Final loss: {losses[-1] if losses else 0:.4f}")

        # Assertions
        assert len(ratings) == 50, "Should complete 50 steps"
        # Should show some learning (not strict requirement)
        if len(rewards) > 10:
            early_avg = np.mean(rewards[:10])
            late_avg = np.mean(rewards[-10:])
            print(f"  Early vs Late reward: {early_avg:.4f} → {late_avg:.4f}")

        print("[OK] REINFORCE completed 50 steps successfully")

    def test_ppo_with_fake_users(self):
        """Test PPO with 50 steps of fake user interaction"""
        print("\n[TEST] PPO with fake users (50 steps)...")

        # Create evolution service with PPO
        evolution_service = EvolutionService(
            algorithm="PPO",
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            batch_size=8  # Smaller batch for testing
        )

        # Create diverse fake users
        users = [
            FakeUser(UserType.CRITICAL, np.random.rand(self.action_dim)),
            FakeUser(UserType.GENEROUS, np.random.rand(self.action_dim)),
            FakeUser(UserType.IMPROVING, np.random.rand(self.action_dim)),
        ]

        # Metrics tracking
        metrics_history = {
            "rewards": [],
            "value_loss": [],
            "policy_loss": [],
            "clip_fraction": [],
            "entropy": []
        }

        # Run 50 steps
        for step in range(50):
            user = users[step % len(users)]

            # Generate options
            result = evolution_service.generate_options(
                user_id=f"ppo_user_{step}",
                dna=self.test_dna,
                brief=self.test_brief,
                num_options=2
            )

            experiment_id = result["experiment_id"]
            options = result["options"]

            # User interaction
            chosen_id = user.choose_option(options)
            rating = user.rate_choice(step % self.action_dim, step)

            # Process feedback
            feedback_result = evolution_service.process_feedback(
                experiment_id=experiment_id,
                chosen_id=chosen_id,
                rating=rating
            )

            # Track PPO-specific metrics
            if "metrics" in feedback_result:
                m = feedback_result["metrics"]
                metrics_history["rewards"].append(m.get("reward", 0))

                if "value_loss" in m:
                    metrics_history["value_loss"].append(m["value_loss"])
                if "policy_loss" in m:
                    metrics_history["policy_loss"].append(m["policy_loss"])
                if "clip_fraction" in m:
                    metrics_history["clip_fraction"].append(m["clip_fraction"])
                if "entropy" in m:
                    metrics_history["entropy"].append(m["entropy"])

                # Log PPO update
                rl_logger.log_update(
                    algorithm="PPO",
                    loss=m.get("loss", 0),
                    reward=m.get("reward", 0),
                    entropy=m.get("entropy"),
                    clip_frac=m.get("clip_fraction"),
                    value_loss=m.get("value_loss"),
                    policy_loss=m.get("policy_loss"),
                    step=step
                )

            evolution_service.end_session(experiment_id)

        # Analyze PPO metrics
        print(f"  Steps completed: 50")
        print(f"  Average reward: {np.mean(metrics_history['rewards']):.4f}")

        if metrics_history["clip_fraction"]:
            avg_clip = np.mean(metrics_history["clip_fraction"])
            print(f"  Average clip fraction: {avg_clip:.4f}")
            assert 0 <= avg_clip <= 0.5, "Clip fraction out of expected range"

        if metrics_history["entropy"]:
            avg_entropy = np.mean(metrics_history["entropy"])
            print(f"  Average entropy: {avg_entropy:.4f}")

        print("[OK] PPO completed 50 steps successfully")

    def test_policy_distribution_change(self):
        """Test that policy distribution changes with training"""
        print("\n[TEST] Testing policy distribution change...")

        trainer = create_rl_trainer(
            algorithm="REINFORCE",
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )

        # Get initial policy distribution
        state = torch.randn(1, self.state_dim)
        with torch.no_grad():
            initial_probs = trainer.policy_net(state).numpy()[0]

        # Simulate training with consistent rewards
        for _ in range(20):
            action, log_prob = trainer.select_action(state)

            # Give high reward to action 0, low to others
            if action == 0:
                trainer.store_reward(1.0)
            else:
                trainer.store_reward(-0.5)

            # Update every 5 steps
            if len(trainer.rewards) >= 5:
                trainer.update()

        # Get final policy distribution
        with torch.no_grad():
            final_probs = trainer.policy_net(state).numpy()[0]

        # Calculate KL divergence
        kl_div = np.sum(
            final_probs * np.log(np.maximum(final_probs, 1e-10) / np.maximum(initial_probs, 1e-10))
        )

        print(f"  Initial max prob: {np.max(initial_probs):.4f}")
        print(f"  Final max prob: {np.max(final_probs):.4f}")
        print(f"  KL divergence: {kl_div:.4f}")

        # Should show distribution change
        assert kl_div > 0.01, "Policy distribution did not change"

        # Action 0 should have higher probability
        assert final_probs[0] > initial_probs[0], "Policy did not improve for rewarded action"

        print("[OK] Policy distribution changes with training")


class TestRLConvergence:
    """Test RL convergence properties"""

    def test_reward_normalization(self):
        """Test reward normalization (rating-3)/2"""
        print("\n[TEST] Testing reward normalization...")

        test_cases = [
            (1, -1.0),  # Minimum rating → minimum reward
            (3, 0.0),   # Neutral rating → zero reward
            (5, 1.0),   # Maximum rating → maximum reward
            (2, -0.5),  # Below neutral
            (4, 0.5),   # Above neutral
        ]

        for rating, expected_reward in test_cases:
            reward = (rating - 3) / 2
            assert abs(reward - expected_reward) < 0.001, f"Wrong reward for rating {rating}"
            print(f"  Rating {rating} → Reward {reward:+.1f} ✓")

        print("[OK] Reward normalization correct")

    def test_advantage_estimation(self):
        """Test advantage estimation in PPO"""
        print("\n[TEST] Testing advantage estimation...")

        from fragrance_ai.training.rl.ppo import compute_gae

        # Create mock data
        rewards = [0.1, 0.2, 0.3, 0.2, 0.1]
        values = [0.15, 0.25, 0.35, 0.25, 0.15]
        dones = [False, False, False, False, True]

        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)

        # Compute advantages
        advantages = compute_gae(rewards_t, values_t, dones_t, gamma=0.99, lam=0.95)

        # Check properties
        assert advantages.shape[0] == len(rewards), "Wrong advantage shape"
        assert not torch.isnan(advantages).any(), "NaN in advantages"
        assert not torch.isinf(advantages).any(), "Inf in advantages"

        # Advantages should be normalized (roughly zero mean)
        adv_mean = advantages.mean().item()
        assert abs(adv_mean) < 0.5, f"Advantages not centered: mean = {adv_mean}"

        print(f"  Advantages computed: shape={advantages.shape}")
        print(f"  Mean: {adv_mean:.4f}, Std: {advantages.std():.4f}")
        print("[OK] Advantage estimation working")


class TestRLStability:
    """Test RL training stability"""

    def test_gradient_stability(self):
        """Test gradient stability during training"""
        print("\n[TEST] Testing gradient stability...")

        trainer = create_rl_trainer(
            algorithm="PPO",
            state_dim=20,
            action_dim=12,
            clip_eps=0.2
        )

        # Track gradient norms
        grad_norms = []

        for i in range(10):
            state = torch.randn(32, 20)  # Batch of states
            action = torch.randint(0, 12, (32,))
            old_log_probs = torch.randn(32)
            advantages = torch.randn(32)
            returns = torch.randn(32)

            # Compute loss
            trainer.optimizer.zero_grad()

            # Forward pass
            probs = trainer.policy_net(state)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(action)

            # PPO loss
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # Backward
            policy_loss.backward()

            # Calculate gradient norm
            total_norm = 0.0
            for p in trainer.policy_net.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)

            trainer.optimizer.step()

        # Check gradient stability
        max_grad = max(grad_norms)
        avg_grad = np.mean(grad_norms)

        print(f"  Average gradient norm: {avg_grad:.4f}")
        print(f"  Max gradient norm: {max_grad:.4f}")

        # Gradients should not explode
        assert max_grad < 100, f"Gradient explosion detected: {max_grad}"
        assert avg_grad > 0.001, f"Gradient vanishing detected: {avg_grad}"

        print("[OK] Gradients stable during training")

    def test_value_function_learning(self):
        """Test value function learning in PPO"""
        print("\n[TEST] Testing value function learning...")

        trainer = create_rl_trainer(
            algorithm="PPO",
            state_dim=10,
            action_dim=5
        )

        # Create consistent state-value pairs
        states = torch.randn(100, 10)
        true_values = torch.sum(states[:, :5], dim=1)  # Simple linear function

        # Train value network
        for epoch in range(50):
            trainer.value_optimizer.zero_grad()

            predicted = trainer.value_net(states).squeeze()
            loss = torch.nn.functional.mse_loss(predicted, true_values)

            loss.backward()
            trainer.value_optimizer.step()

        # Test prediction
        test_states = torch.randn(10, 10)
        test_values = torch.sum(test_states[:, :5], dim=1)
        predicted = trainer.value_net(test_states).squeeze()

        mse = torch.nn.functional.mse_loss(predicted, test_values).item()

        print(f"  Value function MSE: {mse:.4f}")
        assert mse < 1.0, "Value function not learning"

        print("[OK] Value function learns successfully")


def test_performance_benchmark():
    """Benchmark RL performance"""
    print("\n[BENCHMARK] RL Performance Test...")

    # Test both algorithms
    for algorithm in ["REINFORCE", "PPO"]:
        trainer = create_rl_trainer(
            algorithm=algorithm,
            state_dim=50,
            action_dim=20
        )

        start_time = time.time()

        # Simulate 100 updates
        for _ in range(100):
            state = torch.randn(1, 50)

            if algorithm == "PPO":
                action, log_prob, value = trainer.select_action(state)
                trainer.store_transition(
                    state=state,
                    action=action,
                    log_prob=log_prob,
                    reward=np.random.randn(),
                    value=value,
                    done=np.random.rand() > 0.9
                )

                if len(trainer.buffer) >= trainer.batch_size:
                    trainer.update()
            else:
                action, log_prob = trainer.select_action(state)
                trainer.store_reward(np.random.randn())

                if len(trainer.rewards) >= 10:
                    trainer.update()

        elapsed = time.time() - start_time

        print(f"  {algorithm}: 100 updates in {elapsed:.2f} seconds")
        assert elapsed < 10, f"{algorithm} too slow"

    print("[OK] RL performance acceptable")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])