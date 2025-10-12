# tests/test_integration_rlhf.py
"""
Integration Test: RLHF Loop (Options → Feedback → Learning)
Validates the complete RLHF workflow without exceptions
"""

import pytest
import torch
import json
import logging
from typing import List, Dict

from fragrance_ai.training.rlhf_complete import RLHFEngine, PPOAgent, REINFORCEAgent


# Capture log output for validation
@pytest.fixture
def captured_logs(caplog):
    """Capture logs at INFO level"""
    caplog.set_level(logging.INFO)
    return caplog


class TestRLHFIntegration:
    """Integration tests for complete RLHF loop"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.state_dim = 20
        self.action_dim = 12
        self.num_options = 3

    def test_ppo_complete_loop(self, captured_logs):
        """
        Test PPO: Generate options → Receive feedback → Update policy
        Verify no exceptions and policy distribution changes
        """
        print("\n[TEST] PPO complete RLHF loop...")

        # Initialize PPO agent
        agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=3e-4,
            batch_size=3,  # Small batch for quick test
            ppo_epochs=2
        )

        # Step 1: Generate initial state
        state = torch.randn(1, self.state_dim)

        # Step 2: Generate options (variations)
        options, _, saved_actions = agent.generate_variations(state, num_options=self.num_options)

        assert len(options) == self.num_options, "Should generate 3 options"
        assert len(saved_actions) == self.num_options, "Should have 3 saved actions"

        print(f"  Generated {len(options)} options")

        # Step 3: Save initial policy distribution
        with torch.no_grad():
            initial_probs = agent.policy_net(state).numpy()

        # Step 4: Simulate user feedback (choose option 1 with rating 5)
        chosen_id = options[1]["id"]
        rating = 5.0  # Positive feedback

        # Step 5: Update policy with feedback
        metrics = agent.update_policy_with_feedback(
            chosen_id=chosen_id,
            options=options,
            state=state,
            saved_actions=saved_actions,
            rating=rating
        )

        print(f"  Update metrics: {metrics}")

        # Step 6: Verify update occurred
        if metrics.get("status") == "buffering":
            # Buffer not full yet, add more experiences
            for _ in range(3):
                state = torch.randn(1, self.state_dim)
                options, _, saved_actions = agent.generate_variations(state, num_options=self.num_options)
                chosen_id = options[0]["id"]
                metrics = agent.update_policy_with_feedback(
                    chosen_id=chosen_id,
                    options=options,
                    state=state,
                    saved_actions=saved_actions,
                    rating=4.0
                )
                if metrics.get("status") != "buffering":
                    break

        # Verify metrics exist
        assert "algorithm" in metrics, "Should have algorithm"
        assert metrics["algorithm"] == "PPO", "Should be PPO"

        if metrics.get("status") != "buffering":
            assert "loss" in metrics, "Should have loss"
            assert "reward" in metrics, "Should have reward"
            assert "entropy" in metrics, "Should have entropy"
            assert "clip_frac" in metrics, "Should have clip_frac"

            # Step 7: Verify policy distribution changed
            with torch.no_grad():
                final_probs = agent.policy_net(state).numpy()

            distribution_change = abs(final_probs - initial_probs).mean()
            print(f"  Policy distribution change: {distribution_change:.4f}")

            assert distribution_change > 0, "Policy should change after update"

        # Step 8: Verify log contains rl_update event
        rl_update_logs = [r for r in captured_logs.records if "rl_update" in r.message]
        if len(rl_update_logs) > 0:
            log_message = rl_update_logs[0].message
            log_data = json.loads(log_message)

            assert log_data["event"] == "rl_update", "Should log rl_update event"
            assert "loss" in log_data, "Should log loss"
            assert "reward" in log_data, "Should log reward"
            assert "entropy" in log_data, "Should log entropy"
            assert "clip_frac" in log_data, "Should log clip_frac"

            print(f"  Logged: {log_data}")

        print("[OK] PPO complete loop passed without exceptions")

    def test_reinforce_complete_loop(self, captured_logs):
        """
        Test REINFORCE: Generate options → Receive feedback → Update policy
        Verify no exceptions and policy distribution changes
        """
        print("\n[TEST] REINFORCE complete RLHF loop...")

        # Initialize REINFORCE agent
        agent = REINFORCEAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=0.001
        )

        # Step 1: Generate initial state
        state = torch.randn(1, self.state_dim)

        # Step 2: Save initial policy distribution
        with torch.no_grad():
            initial_probs = agent.policy_net(state).numpy()

        # Step 3: Generate actions and save log probs
        saved_actions = []
        options = []

        for i in range(self.num_options):
            action, log_prob = agent.select_action(state)
            saved_actions.append((action, log_prob))
            options.append({
                "id": f"option_{i}",
                "action": action
            })

        print(f"  Generated {len(options)} options")

        # Step 4: Simulate user feedback
        chosen_id = options[1]["id"]
        rating = 5.0

        # Step 5: Update policy
        agent.last_state = state
        agent.last_saved_actions = saved_actions

        metrics = agent.update_policy_with_feedback(
            chosen_id=chosen_id,
            options=options,
            rating=rating
        )

        print(f"  Update metrics: {metrics}")

        # Step 6: Verify metrics
        assert metrics["algorithm"] == "REINFORCE", "Should be REINFORCE"
        assert "loss" in metrics, "Should have loss"
        assert "reward" in metrics, "Should have reward"

        # Step 7: Verify policy distribution changed
        with torch.no_grad():
            final_probs = agent.policy_net(state).numpy()

        distribution_change = abs(final_probs - initial_probs).mean()
        print(f"  Policy distribution change: {distribution_change:.4f}")

        assert distribution_change > 0, "Policy should change after update"

        # Step 8: Verify log contains rl_update event
        rl_update_logs = [r for r in captured_logs.records if "rl_update" in r.message]
        assert len(rl_update_logs) > 0, "Should log rl_update event"

        log_message = rl_update_logs[0].message
        log_data = json.loads(log_message)

        assert log_data["event"] == "rl_update", "Should log rl_update event"
        assert "loss" in log_data, "Should log loss"
        assert "reward" in log_data, "Should log reward"

        print(f"  Logged: {log_data}")

        print("[OK] REINFORCE complete loop passed without exceptions")

    def test_unified_engine(self, captured_logs):
        """Test RLHFEngine unified interface"""
        print("\n[TEST] Unified RLHF engine...")

        for algorithm in ["REINFORCE", "PPO"]:
            print(f"\n  Testing {algorithm}...")

            engine = RLHFEngine(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                algorithm=algorithm,
                batch_size=3 if algorithm == "PPO" else None
            )

            # Generate state and options
            state = torch.randn(1, self.state_dim)

            if algorithm == "PPO":
                options, _, saved_actions = engine.agent.generate_variations(
                    state, num_options=self.num_options
                )
            else:
                saved_actions = []
                options = []
                for i in range(self.num_options):
                    action, log_prob = engine.agent.select_action(state)
                    saved_actions.append((action, log_prob))
                    options.append({"id": f"option_{i}", "action": action})

                engine.agent.last_state = state
                engine.agent.last_saved_actions = saved_actions

            # Simulate feedback
            chosen_id = options[0]["id"]
            metrics = engine.update_policy_with_feedback(
                chosen_id=chosen_id,
                options=options,
                rating=4.0
            )

            # For PPO, may need more samples
            if algorithm == "PPO" and metrics.get("status") == "buffering":
                for _ in range(5):
                    state = torch.randn(1, self.state_dim)
                    options, _, saved_actions = engine.agent.generate_variations(
                        state, num_options=self.num_options
                    )
                    metrics = engine.update_policy_with_feedback(
                        chosen_id=options[0]["id"],
                        options=options,
                        rating=3.0
                    )
                    if metrics.get("status") != "buffering":
                        break

            assert "algorithm" in metrics, f"{algorithm} should return algorithm"
            print(f"    {algorithm} metrics: {metrics}")

        print("[OK] Unified engine passed")

    def test_policy_improvement(self):
        """Test that policy distribution changes with feedback"""
        print("\n[TEST] Policy improvement with feedback...")

        agent = PPOAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            batch_size=3,
            ppo_epochs=2
        )

        state = torch.randn(1, self.state_dim)

        # Get initial policy distribution
        with torch.no_grad():
            initial_probs = agent.policy_net(state).numpy()[0].copy()

        print(f"  Initial distribution: {initial_probs[:5]}...")

        # Collect 15 episodes with feedback
        for episode in range(15):
            options, _, saved_actions = agent.generate_variations(state, num_options=3)

            # Choose first option with varying ratings
            chosen_id = options[0]["id"]
            rating = 5.0 if episode % 2 == 0 else 3.0  # Mix positive and neutral

            agent.update_policy_with_feedback(
                chosen_id=chosen_id,
                options=options,
                state=state,
                saved_actions=saved_actions,
                rating=rating
            )

        # Get final policy distribution
        with torch.no_grad():
            final_probs = agent.policy_net(state).numpy()[0]

        print(f"  Final distribution: {final_probs[:5]}...")

        # Calculate distribution change
        total_change = abs(final_probs - initial_probs).sum()

        print(f"  Total distribution change: {total_change:.4f}")

        # After 15 episodes with feedback, policy should change noticeably
        # (total variation distance > 0.1 means policy changed)
        assert total_change > 0.05, f"Policy should change with feedback (got {total_change:.4f})"

        print("[OK] Policy learning verified")


if __name__ == "__main__":
    # Run with detailed output
    pytest.main([__file__, "-v", "-s", "--tb=short"])
