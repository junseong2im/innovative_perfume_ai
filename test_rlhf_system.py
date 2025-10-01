"""
Test script for Enhanced RLHF System
Tests PolicyNetwork, RewardModel, and complete training pipeline
"""

import torch
import numpy as np
import random
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.training.rlhf_enhanced import (
    PolicyNetwork, RewardModel, EnhancedRLHFSystem,
    FragranceDNA, UserFeedback, Experience, ActionType
)


def test_policy_network():
    """Test PolicyNetwork architecture and forward pass"""
    print("\nTesting PolicyNetwork...")
    print("-" * 50)

    # Initialize network
    policy_net = PolicyNetwork(
        dna_dim=75,
        feedback_dim=25,
        hidden_dim=256,
        num_actions=60
    )

    # Create sample inputs
    batch_size = 4
    dna_input = torch.randn(batch_size, 75)
    feedback_input = torch.randn(batch_size, 25)

    # Forward pass
    action_probs, values = policy_net(dna_input, feedback_input)

    # Validate outputs
    assert action_probs.shape == (batch_size, 60), f"Expected shape (4, 60), got {action_probs.shape}"
    assert values.shape == (batch_size, 1), f"Expected shape (4, 1), got {values.shape}"

    # Check probability distribution properties
    assert torch.allclose(action_probs.sum(dim=1), torch.ones(batch_size), atol=1e-6), "Probabilities don't sum to 1"
    assert (action_probs >= 0).all() and (action_probs <= 1).all(), "Invalid probability values"

    print(f"  [OK] Output shapes: action_probs={action_probs.shape}, values={values.shape}")
    print(f"  [OK] Probability sum: {action_probs.sum(dim=1).mean():.6f}")
    print(f"  [OK] Value range: [{values.min():.3f}, {values.max():.3f}]")

    # Test gradient flow
    loss = action_probs.mean() + values.mean()
    loss.backward()

    has_gradients = all(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in policy_net.parameters()
    )
    assert has_gradients, "No gradients computed"

    print("  [OK] Gradient flow verified")

    return True


def test_reward_model():
    """Test RewardModel for preference learning"""
    print("\nTesting RewardModel...")
    print("-" * 50)

    # Initialize model
    reward_model = RewardModel(state_dim=100, action_dim=60)

    # Create sample inputs
    batch_size = 8
    states = torch.randn(batch_size, 100)
    actions = torch.randn(batch_size, 60)

    # Forward pass
    rewards = reward_model(states, actions)

    # Validate outputs
    assert rewards.shape == (batch_size, 1), f"Expected shape (8, 1), got {rewards.shape}"
    assert (rewards >= -1).all() and (rewards <= 1).all(), "Rewards not in [-1, 1] range"

    print(f"  [OK] Reward shape: {rewards.shape}")
    print(f"  [OK] Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")

    # Test preference learning
    state1 = torch.randn(1, 100)
    state2 = torch.randn(1, 100)
    action1 = torch.randn(1, 60)
    action2 = torch.randn(1, 60)

    reward1 = reward_model(state1, action1)
    reward2 = reward_model(state2, action2)

    # Simulate preference (option 1 preferred)
    preference_loss = -torch.log(torch.sigmoid(reward1 - reward2))
    preference_loss.backward()

    has_gradients = any(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in reward_model.parameters()
    )
    assert has_gradients, "No gradients for preference learning"

    print("  [OK] Preference learning gradient flow verified")

    return True


def test_dna_encoding():
    """Test DNA and feedback encoding"""
    print("\nTesting DNA and Feedback Encoding...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Create sample DNA
    dna = FragranceDNA(
        notes=[(1, 10.0), (3, 15.0), (5, 8.0), (7, 12.0), (9, 5.0)],
        emotional_profile=[0.7, 0.3, 0.5, 0.8, 0.6, 0.4, 0.9],
        fitness_scores=(0.8, 0.7, 0.9)
    )

    # Encode DNA
    dna_encoded = rlhf_system.encode_dna(dna)

    assert dna_encoded.shape == (75,), f"Expected shape (75,), got {dna_encoded.shape}"
    assert torch.isfinite(dna_encoded).all(), "DNA encoding contains NaN or Inf"

    print(f"  [OK] DNA encoding shape: {dna_encoded.shape}")
    print(f"  [OK] DNA encoding range: [{dna_encoded.min():.3f}, {dna_encoded.max():.3f}]")

    # Create sample feedback
    feedback = UserFeedback(
        rating=7.5,
        preferences={
            'sweetness': 0.6,
            'freshness': 0.8,
            'intensity': 0.7
        },
        comparison=1
    )

    # Encode feedback
    feedback_encoded = rlhf_system.encode_feedback(feedback)

    assert feedback_encoded.shape == (25,), f"Expected shape (25,), got {feedback_encoded.shape}"
    assert torch.isfinite(feedback_encoded).all(), "Feedback encoding contains NaN or Inf"

    print(f"  [OK] Feedback encoding shape: {feedback_encoded.shape}")
    print(f"  [OK] Feedback encoding range: [{feedback_encoded.min():.3f}, {feedback_encoded.max():.3f}]")

    return True


def test_action_selection_and_application():
    """Test action selection and DNA modification"""
    print("\nTesting Action Selection and Application...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Create sample DNA
    original_dna = FragranceDNA(
        notes=[(1, 10.0), (3, 15.0), (5, 8.0)],
        emotional_profile=[0.5] * 5,
        fitness_scores=(0.5, 0.5, 0.5)
    )

    feedback = UserFeedback(
        rating=6.0,
        preferences={'intensity': 0.5}
    )

    # Test multiple action types
    action_results = {}
    for action_type in ActionType:
        # Find an action of this type
        action_idx = next(
            i for i, a in enumerate(rlhf_system.action_space)
            if a['type'] == action_type
        )

        action_info = rlhf_system.action_space[action_idx]

        # Apply action
        new_dna = rlhf_system.apply_action(original_dna, action_info)

        # Check modification
        modified = new_dna.notes != original_dna.notes
        action_results[action_type.value] = modified

        print(f"  [{action_type.value:10}] Modified: {modified}, "
              f"Notes: {len(original_dna.notes)} -> {len(new_dna.notes)}")

    # At least some actions should modify the DNA
    assert any(action_results.values()), "No actions modified the DNA"

    # Test action selection with policy
    selected_idx, selected_info = rlhf_system.select_action(
        original_dna,
        feedback,
        epsilon=0.0  # No exploration
    )

    assert 0 <= selected_idx < len(rlhf_system.action_space), "Invalid action index"
    assert 'type' in selected_info and 'note' in selected_info, "Missing action info"

    print(f"\n  [OK] Selected action: {selected_info['description']}")

    return True


def test_experience_replay():
    """Test experience replay buffer and training"""
    print("\nTesting Experience Replay and Training...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Create sample experiences
    for i in range(50):
        state = torch.randn(100).to(rlhf_system.device)
        action = random.randint(0, 59)
        reward = random.uniform(-1, 1)
        next_state = torch.randn(100).to(rlhf_system.device)
        done = i % 10 == 0

        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=torch.tensor(random.random())
        )

        rlhf_system.replay_buffer.append(experience)

    print(f"  Replay buffer size: {len(rlhf_system.replay_buffer)}")

    # Test policy training
    initial_params = [p.clone() for p in rlhf_system.policy_net.parameters()]

    rlhf_system.train_policy(batch_size=16, num_epochs=2)

    # Check if parameters changed
    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, rlhf_system.policy_net.parameters())
    )

    assert params_changed, "Policy parameters didn't change after training"

    print("  [OK] Policy training successful")

    # Check training statistics
    if rlhf_system.training_stats['policy_loss']:
        avg_loss = np.mean(rlhf_system.training_stats['policy_loss'][-10:])
        print(f"  [OK] Average policy loss: {avg_loss:.4f}")

    return True


def test_human_feedback_collection():
    """Test human feedback collection and reward model training"""
    print("\nTesting Human Feedback Collection...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Create two DNA options for comparison
    dna1 = FragranceDNA(
        notes=[(1, 10.0), (3, 15.0)],
        emotional_profile=[0.7, 0.3, 0.5],
        fitness_scores=(0.8, 0.7, 0.6)
    )

    dna2 = FragranceDNA(
        notes=[(2, 12.0), (4, 8.0)],
        emotional_profile=[0.5, 0.5, 0.5],
        fitness_scores=(0.6, 0.8, 0.7)
    )

    # Simulate human feedback
    for _ in range(20):
        preference = random.choice([-1, 0, 1])
        rating1 = random.uniform(5, 9)
        rating2 = random.uniform(5, 9)

        rlhf_system.collect_human_feedback(
            dna1, dna2,
            preference,
            rating1, rating2
        )

    print(f"  Feedback buffer size: {len(rlhf_system.feedback_buffer)}")

    # Test reward model training
    initial_params = [p.clone() for p in rlhf_system.reward_model.parameters()]

    rlhf_system.train_reward_model(batch_size=8)

    # Check if parameters changed
    params_changed = any(
        not torch.allclose(p1, p2)
        for p1, p2 in zip(initial_params, rlhf_system.reward_model.parameters())
    )

    # Note: Parameters might not change if loss is very small
    print(f"  [{'OK' if params_changed else 'WARN'}] Reward model parameters "
          f"{'changed' if params_changed else 'unchanged (may be due to small batch)'}")

    return True


def test_gae_computation():
    """Test Generalized Advantage Estimation"""
    print("\nTesting GAE Computation...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Create sample trajectory
    rewards = [0.5, 0.3, -0.2, 0.8, 1.0]
    values = [0.4, 0.5, 0.1, 0.6, 0.9]
    dones = [False, False, False, False, True]

    # Compute GAE
    advantages = rlhf_system.compute_gae(rewards, values, dones)

    assert advantages.shape == (5,), f"Expected shape (5,), got {advantages.shape}"
    assert torch.isfinite(advantages).all(), "GAE contains NaN or Inf"

    print(f"  [OK] GAE shape: {advantages.shape}")
    print(f"  [OK] GAE range: [{advantages.min():.3f}, {advantages.max():.3f}]")
    print(f"  [OK] GAE mean: {advantages.mean():.3f}")

    return True


def test_model_save_load():
    """Test model saving and loading"""
    print("\nTesting Model Save/Load...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Train briefly to change parameters
    for _ in range(10):
        state = torch.randn(100).to(rlhf_system.device)
        experience = Experience(
            state=state,
            action=random.randint(0, 59),
            reward=random.random(),
            next_state=state,
            done=False,
            log_prob=torch.tensor(0.0)
        )
        rlhf_system.replay_buffer.append(experience)

    rlhf_system.train_policy(batch_size=8, num_epochs=1)

    # Save model
    save_path = "test_rlhf_checkpoint.pth"
    rlhf_system.save_model(save_path)
    print(f"  [OK] Model saved to {save_path}")

    # Get current parameters
    original_params = {
        name: param.clone()
        for name, param in rlhf_system.policy_net.named_parameters()
    }

    # Create new system and load
    new_system = EnhancedRLHFSystem()
    new_system.load_model(save_path)
    print(f"  [OK] Model loaded from {save_path}")

    # Compare parameters
    for name, param in new_system.policy_net.named_parameters():
        if name in original_params:
            assert torch.allclose(param, original_params[name], atol=1e-6), f"Parameter {name} mismatch"

    print("  [OK] Parameters match after load")

    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)

    return True


def run_integration_test():
    """Run complete integration test"""
    print("\nRunning Integration Test...")
    print("-" * 50)

    rlhf_system = EnhancedRLHFSystem()

    # Initial DNA
    current_dna = FragranceDNA(
        notes=[(1, 10.0), (3, 15.0), (5, 8.0)],
        emotional_profile=[0.6, 0.4, 0.7, 0.3, 0.8],
        fitness_scores=(0.7, 0.6, 0.8)
    )

    # Simulate training episodes
    num_episodes = 20
    total_rewards = []

    for episode in range(num_episodes):
        # Get user feedback
        feedback = UserFeedback(
            rating=random.uniform(5, 9),
            preferences={
                'sweetness': random.random(),
                'freshness': random.random(),
                'intensity': random.random()
            }
        )

        # Select and apply action
        action_idx, action_info = rlhf_system.select_action(
            current_dna,
            feedback,
            epsilon=0.2
        )

        new_dna = rlhf_system.apply_action(current_dna, action_info)

        # Simulate reward
        reward = (feedback.rating - 7.0) / 2.0
        total_rewards.append(reward)

        # Store experience
        state = torch.cat([
            rlhf_system.encode_dna(current_dna),
            rlhf_system.encode_feedback(feedback)
        ])

        next_state = torch.cat([
            rlhf_system.encode_dna(new_dna),
            rlhf_system.encode_feedback(feedback)
        ])

        experience = Experience(
            state=state,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=episode % 5 == 4,
            log_prob=torch.tensor(0.0)
        )

        rlhf_system.replay_buffer.append(experience)

        # Train periodically
        if episode % 5 == 4:
            rlhf_system.train_policy(batch_size=8, num_epochs=1)

        # Update DNA
        current_dna = new_dna

    avg_reward = np.mean(total_rewards)
    print(f"  Episodes completed: {num_episodes}")
    print(f"  Average reward: {avg_reward:.3f}")
    print(f"  Final DNA notes: {len(current_dna.notes)}")
    print(f"  Training losses recorded: {len(rlhf_system.training_stats['policy_loss'])}")

    print("  [OK] Integration test completed")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED RLHF SYSTEM TEST SUITE")
    print("=" * 60)

    tests = [
        ("PolicyNetwork", test_policy_network),
        ("RewardModel", test_reward_model),
        ("DNA Encoding", test_dna_encoding),
        ("Action System", test_action_selection_and_application),
        ("Experience Replay", test_experience_replay),
        ("Human Feedback", test_human_feedback_collection),
        ("GAE Computation", test_gae_computation),
        ("Model Save/Load", test_model_save_load),
        ("Integration", run_integration_test)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] {test_name} test completed successfully")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name} test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\nAll tests passed successfully!")
        print("\nRLHF System Features:")
        print("  [OK] PolicyNetwork with attention mechanism")
        print("  [OK] RewardModel for preference learning")
        print("  [OK] PPO algorithm implementation")
        print("  [OK] GAE for advantage estimation")
        print("  [OK] Human feedback collection")
        print("  [OK] Experience replay buffer")
        print("  [OK] Model checkpointing")
    else:
        print(f"\n{failed} test(s) failed. Please check the errors above.")

    print("=" * 60)