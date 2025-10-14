"""
Tests for Advanced RLHF Features
- Entropy Annealing
- Reward Normalization
- Checkpoint & Rollback
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from fragrance_ai.training.rl_advanced import (
    EntropyScheduler,
    EntropyScheduleConfig,
    RewardNormalizer,
    RewardNormalizerConfig,
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetrics
)
from fragrance_ai.training.ppo_engine import ActorCriticNetwork


# ============================================================================
# Test Entropy Annealing
# ============================================================================

class TestEntropyScheduler:
    """Entropy annealing scheduler tests"""

    def test_linear_decay(self):
        """Test linear decay from initial to final entropy"""
        config = EntropyScheduleConfig(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=100,
            schedule_type="linear"
        )
        scheduler = EntropyScheduler(config)

        # Initial value
        assert scheduler.get_entropy() == 0.01

        # Halfway through
        for _ in range(50):
            scheduler.step()

        halfway_entropy = scheduler.get_entropy()
        expected_halfway = (0.01 + 0.001) / 2
        assert abs(halfway_entropy - expected_halfway) < 1e-6

        # Final value
        for _ in range(50):
            scheduler.step()

        final_entropy = scheduler.get_entropy()
        assert abs(final_entropy - 0.001) < 1e-6

    def test_cosine_decay(self):
        """Test cosine annealing decay"""
        config = EntropyScheduleConfig(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=100,
            schedule_type="cosine"
        )
        scheduler = EntropyScheduler(config)

        # Cosine should start at initial value
        assert scheduler.get_entropy() == 0.01

        # Step through and verify it decreases
        prev_entropy = scheduler.get_entropy()
        for _ in range(100):
            current_entropy = scheduler.step()
            # Should be decreasing (monotonic)
            assert current_entropy <= prev_entropy or abs(current_entropy - prev_entropy) < 1e-10
            prev_entropy = current_entropy

        # Should reach final value (approximately)
        final_entropy = scheduler.get_entropy()
        assert abs(final_entropy - 0.001) < 1e-3  # Cosine may not reach exact final

    def test_exponential_decay(self):
        """Test exponential decay"""
        config = EntropyScheduleConfig(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=100,
            schedule_type="exponential"
        )
        scheduler = EntropyScheduler(config)

        # Initial value
        assert scheduler.get_entropy() == 0.01

        # Exponential should decrease faster at the beginning
        first_step = scheduler.step()
        assert first_step < 0.01

        # Complete decay
        for _ in range(99):
            scheduler.step()

        final_entropy = scheduler.get_entropy()
        assert abs(final_entropy - 0.001) < 1e-3

    def test_scheduler_info(self):
        """Test scheduler info tracking"""
        config = EntropyScheduleConfig(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=100
        )
        scheduler = EntropyScheduler(config)

        # Initial info
        info = scheduler.get_info()
        assert info['current_step'] == 0
        assert info['progress'] == 0.0
        assert info['remaining_steps'] == 100

        # Halfway info
        for _ in range(50):
            scheduler.step()

        info = scheduler.get_info()
        assert info['current_step'] == 50
        assert abs(info['progress'] - 0.5) < 1e-6
        assert info['remaining_steps'] == 50

    def test_scheduler_reset(self):
        """Test scheduler reset"""
        config = EntropyScheduleConfig(
            initial_entropy=0.01,
            final_entropy=0.001,
            decay_steps=100
        )
        scheduler = EntropyScheduler(config)

        # Step forward
        for _ in range(50):
            scheduler.step()

        assert scheduler.current_step == 50

        # Reset
        scheduler.reset()

        assert scheduler.current_step == 0
        assert scheduler.get_entropy() == 0.01


# ============================================================================
# Test Reward Normalization
# ============================================================================

class TestRewardNormalizer:
    """Reward normalization tests"""

    def test_basic_normalization(self):
        """Test basic reward normalization"""
        config = RewardNormalizerConfig(
            window_size=100,
            epsilon=1e-8,
            clip_range=(-10.0, 10.0)
        )
        normalizer = RewardNormalizer(config)

        # Add rewards with known distribution
        rewards = [1.0, 2.0, 3.0, 4.0, 5.0] * 20  # 100 rewards

        for r in rewards:
            normalizer.normalize(r)

        stats = normalizer.get_statistics()

        # Mean should be ~3.0
        assert abs(stats['mean'] - 3.0) < 0.1

        # Std should be ~sqrt(2)
        expected_std = np.std([1, 2, 3, 4, 5])
        assert abs(stats['std'] - expected_std) < 0.1

    def test_normalization_output(self):
        """Test normalized reward values"""
        config = RewardNormalizerConfig(
            window_size=1000,
            clip_range=None  # No clipping for this test
        )
        normalizer = RewardNormalizer(config)

        # Warm up with rewards
        for _ in range(100):
            normalizer.normalize(0.0)

        # Add a reward of +1 std above mean
        # After 100 zeros, mean=0, std~0
        # Need more variance
        for i in range(100):
            normalizer.normalize(float(i))

        stats = normalizer.get_statistics()

        # Normalize a value 1 std above mean
        test_reward = stats['mean'] + stats['std']
        normalized = normalizer.normalize(test_reward)

        # Should be approximately 1.0
        assert abs(normalized - 1.0) < 0.5  # Allow some tolerance

    def test_clipping(self):
        """Test reward clipping"""
        config = RewardNormalizerConfig(
            window_size=100,
            clip_range=(-5.0, 5.0)
        )
        normalizer = RewardNormalizer(config)

        # Warm up
        for _ in range(50):
            normalizer.normalize(0.0)

        # Add extreme reward
        normalized = normalizer.normalize(1000.0)

        # Should be clipped to 5.0
        assert normalized <= 5.0

        # Add extreme negative reward
        normalized = normalizer.normalize(-1000.0)

        # Should be clipped to -5.0
        assert normalized >= -5.0

    def test_welford_algorithm_stability(self):
        """Test Welford's algorithm for numerical stability"""
        config = RewardNormalizerConfig(window_size=10000)
        normalizer = RewardNormalizer(config)

        # Add many rewards with large values
        large_rewards = np.random.randn(1000) * 1e6 + 1e9

        for r in large_rewards:
            normalizer.normalize(r)

        stats = normalizer.get_statistics()

        # Mean should be close to 1e9
        assert abs(stats['mean'] - 1e9) < 1e8

        # Std should be reasonable (not NaN or inf)
        assert not np.isnan(stats['std'])
        assert not np.isinf(stats['std'])

    def test_batch_normalization(self):
        """Test batch reward normalization"""
        config = RewardNormalizerConfig(window_size=100)
        normalizer = RewardNormalizer(config)

        # Warm up
        for _ in range(50):
            normalizer.normalize(0.0)

        # Batch normalize
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = normalizer.normalize_batch(rewards)

        assert len(normalized) == len(rewards)
        assert isinstance(normalized, np.ndarray)

    def test_reset(self):
        """Test normalizer reset"""
        config = RewardNormalizerConfig(window_size=100)
        normalizer = RewardNormalizer(config)

        # Add rewards
        for i in range(50):
            normalizer.normalize(float(i))

        assert normalizer.count == 50

        # Reset
        normalizer.reset()

        assert normalizer.count == 0
        assert normalizer.mean == 0.0
        assert len(normalizer.reward_buffer) == 0


# ============================================================================
# Test Checkpoint & Rollback
# ============================================================================

class TestCheckpointManager:
    """Checkpoint and rollback tests"""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def dummy_network(self):
        """Create dummy network for testing"""
        return ActorCriticNetwork(state_dim=10, action_dim=5)

    @pytest.fixture
    def dummy_optimizer(self, dummy_network):
        """Create dummy optimizer"""
        return torch.optim.Adam(dummy_network.parameters(), lr=1e-3)

    def test_checkpoint_saving(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test checkpoint saving"""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval=10,
            max_checkpoints=3
        )
        manager = CheckpointManager(config)

        # Save checkpoint
        metrics = CheckpointMetrics(
            step=10,
            timestamp="2025-01-15T10:00:00",
            policy_loss=0.5,
            value_loss=0.3,
            kl_divergence=0.01,
            mean_reward=5.0,
            explained_variance=0.8
        )

        filepath = manager.save_checkpoint(
            step=10,
            network=dummy_network,
            optimizer=dummy_optimizer,
            scheduler=None,
            metrics=metrics
        )

        assert filepath.exists()
        assert len(manager.checkpoints) == 1

    def test_max_checkpoints_cleanup(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test automatic cleanup of old checkpoints"""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval=10,
            max_checkpoints=3
        )
        manager = CheckpointManager(config)

        # Save 5 checkpoints
        for step in [10, 20, 30, 40, 50]:
            metrics = CheckpointMetrics(
                step=step,
                timestamp="2025-01-15T10:00:00",
                policy_loss=0.5,
                value_loss=0.3,
                kl_divergence=0.01,
                mean_reward=float(step),
                explained_variance=0.8
            )

            manager.save_checkpoint(
                step=step,
                network=dummy_network,
                optimizer=dummy_optimizer,
                scheduler=None,
                metrics=metrics
            )

        # Should only keep 3 most recent
        assert len(manager.checkpoints) == 3

        # Verify most recent steps are kept
        kept_steps = [step for step, _, _ in manager.checkpoints]
        assert kept_steps == [30, 40, 50]

    def test_rollback_on_kl_threshold(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test rollback on high KL divergence"""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            rollback_on_kl_threshold=0.1,
            metric_window_size=5
        )
        manager = CheckpointManager(config)

        # Build history with good KL values (need at least 3 for rollback check)
        for step in [10, 20, 30]:
            good_metrics = CheckpointMetrics(
                step=step,
                timestamp="2025-01-15T10:00:00",
                policy_loss=0.5,
                value_loss=0.3,
                kl_divergence=0.01,
                mean_reward=5.0,
                explained_variance=0.8
            )

            manager.save_checkpoint(
                step=step,
                network=dummy_network,
                optimizer=dummy_optimizer,
                scheduler=None,
                metrics=good_metrics
            )

        # Create bad metrics with high KL
        bad_metrics = CheckpointMetrics(
            step=40,
            timestamp="2025-01-15T10:01:00",
            policy_loss=0.5,
            value_loss=0.3,
            kl_divergence=0.2,  # Too high!
            mean_reward=5.0,
            explained_variance=0.8
        )

        should_rollback, reason = manager.should_rollback(bad_metrics)

        assert should_rollback
        assert "KL divergence too high" in reason

    def test_rollback_on_loss_spike(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test rollback on loss spike"""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            rollback_on_loss_spike_factor=3.0,
            metric_window_size=5
        )
        manager = CheckpointManager(config)

        # Build history with normal losses
        for step in range(10, 60, 10):
            metrics = CheckpointMetrics(
                step=step,
                timestamp="2025-01-15T10:00:00",
                policy_loss=0.5,
                value_loss=0.3,
                kl_divergence=0.01,
                mean_reward=5.0,
                explained_variance=0.8
            )
            manager.metrics_history.append(metrics)

        # Create metrics with loss spike
        spike_metrics = CheckpointMetrics(
            step=60,
            timestamp="2025-01-15T10:01:00",
            policy_loss=5.0,  # Spike!
            value_loss=5.0,   # Spike!
            kl_divergence=0.01,
            mean_reward=5.0,
            explained_variance=0.8
        )

        should_rollback, reason = manager.should_rollback(spike_metrics)

        assert should_rollback
        assert "Loss spike detected" in reason

    def test_rollback_on_reward_drop(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test rollback on reward drop"""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            rollback_on_reward_drop_factor=0.5,
            metric_window_size=5
        )
        manager = CheckpointManager(config)

        # Build history with good rewards
        for step in range(10, 60, 10):
            metrics = CheckpointMetrics(
                step=step,
                timestamp="2025-01-15T10:00:00",
                policy_loss=0.5,
                value_loss=0.3,
                kl_divergence=0.01,
                mean_reward=10.0,  # Good reward
                explained_variance=0.8
            )
            manager.metrics_history.append(metrics)

        # Create metrics with reward drop
        drop_metrics = CheckpointMetrics(
            step=60,
            timestamp="2025-01-15T10:01:00",
            policy_loss=0.5,
            value_loss=0.3,
            kl_divergence=0.01,
            mean_reward=2.0,  # Dropped to 20% of mean!
            explained_variance=0.8
        )

        should_rollback, reason = manager.should_rollback(drop_metrics)

        assert should_rollback
        assert "Reward drop detected" in reason

    def test_best_checkpoint_tracking(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test best checkpoint tracking"""
        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir
        )
        manager = CheckpointManager(config)

        # Save checkpoints with varying rewards
        for step, reward in [(10, 3.0), (20, 5.0), (30, 4.0), (40, 7.0)]:
            metrics = CheckpointMetrics(
                step=step,
                timestamp="2025-01-15T10:00:00",
                policy_loss=0.5,
                value_loss=0.3,
                kl_divergence=0.01,
                mean_reward=reward,
                explained_variance=0.8
            )

            manager.save_checkpoint(
                step=step,
                network=dummy_network,
                optimizer=dummy_optimizer,
                scheduler=None,
                metrics=metrics
            )

        # Best should be step 40 with reward 7.0
        assert manager.best_checkpoint_step == 40
        assert manager.best_reward == 7.0

        # Best checkpoint file should exist
        best_path = Path(temp_checkpoint_dir) / "checkpoint_best.pth"
        assert best_path.exists()

    def test_rollback_execution(self, temp_checkpoint_dir, dummy_network, dummy_optimizer):
        """Test actual rollback execution"""
        import copy

        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir
        )
        manager = CheckpointManager(config)

        # Save initial checkpoint
        initial_weights = copy.deepcopy({k: v.cpu() for k, v in dummy_network.state_dict().items()})

        metrics = CheckpointMetrics(
            step=10,
            timestamp="2025-01-15T10:00:00",
            policy_loss=0.5,
            value_loss=0.3,
            kl_divergence=0.01,
            mean_reward=5.0,
            explained_variance=0.8
        )

        manager.save_checkpoint(
            step=10,
            network=dummy_network,
            optimizer=dummy_optimizer,
            scheduler=None,
            metrics=metrics
        )

        # Modify network weights
        for param in dummy_network.parameters():
            param.data += 1.0

        # Verify weights changed
        current_weights = {k: v.cpu() for k, v in dummy_network.state_dict().items()}
        for key in initial_weights:
            assert not torch.allclose(initial_weights[key], current_weights[key])

        # Rollback
        rolled_back_step = manager.rollback(
            network=dummy_network,
            optimizer=dummy_optimizer,
            scheduler=None,
            reason="Test rollback"
        )

        assert rolled_back_step == 10
        assert manager.rollback_count == 1

        # Verify weights restored
        restored_weights = {k: v.cpu() for k, v in dummy_network.state_dict().items()}
        for key in initial_weights:
            assert torch.allclose(initial_weights[key], restored_weights[key])


# ============================================================================
# Integration Tests
# ============================================================================

class TestAdvancedPPOIntegration:
    """Integration tests for advanced PPO features"""

    def test_entropy_annealing_integration(self):
        """Test entropy annealing in training loop"""
        from fragrance_ai.training.ppo_trainer_advanced import AdvancedPPOTrainer
        from fragrance_ai.training.ppo_engine import FragranceEnvironment

        env = FragranceEnvironment(n_ingredients=10)

        trainer = AdvancedPPOTrainer(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            entropy_config=EntropyScheduleConfig(
                initial_entropy=0.01,
                final_entropy=0.001,
                decay_steps=10
            )
        )

        initial_entropy = trainer.entropy_scheduler.get_entropy()
        assert initial_entropy == 0.01

        # Run a few training steps
        for _ in range(3):
            trainer.collect_rollout(env, n_steps=100)
            trainer.train_step(n_epochs=2, batch_size=32)

        # Entropy should have decreased
        current_entropy = trainer.entropy_scheduler.get_entropy()
        assert current_entropy < initial_entropy

    def test_reward_normalization_integration(self):
        """Test reward normalization in training loop"""
        from fragrance_ai.training.ppo_trainer_advanced import AdvancedPPOTrainer
        from fragrance_ai.training.ppo_engine import FragranceEnvironment

        env = FragranceEnvironment(n_ingredients=10)

        trainer = AdvancedPPOTrainer(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            reward_config=RewardNormalizerConfig(window_size=100)
        )

        # Collect rollout (rewards will be normalized)
        trainer.collect_rollout(env, n_steps=200)

        # Check normalizer has statistics
        stats = trainer.reward_normalizer.get_statistics()
        assert stats['count'] > 0
        assert 'mean' in stats
        assert 'std' in stats

    @pytest.mark.slow
    def test_full_training_loop(self):
        """Test complete training loop with all features"""
        from fragrance_ai.training.ppo_trainer_advanced import train_advanced_ppo
        from fragrance_ai.training.ppo_engine import FragranceEnvironment

        env = FragranceEnvironment(n_ingredients=10)

        # Run short training
        trainer = train_advanced_ppo(
            env=env,
            n_iterations=5,
            n_steps_per_iteration=100,
            n_ppo_epochs=2,
            entropy_config=EntropyScheduleConfig(
                initial_entropy=0.01,
                final_entropy=0.001,
                decay_steps=5
            ),
            reward_config=RewardNormalizerConfig(window_size=100),
            checkpoint_config=CheckpointConfig(save_interval=2)
        )

        # Verify training completed
        assert trainer.training_step > 0
        assert trainer.episode_count > 0

        # Verify features worked
        assert trainer.entropy_scheduler.current_step > 0
        assert trainer.reward_normalizer.count > 0
        assert len(trainer.checkpoint_manager.checkpoints) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
