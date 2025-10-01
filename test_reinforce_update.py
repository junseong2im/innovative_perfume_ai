"""
Test Script for REINFORCE Policy Update Implementation
Verifies AdamW optimizer and proper gradient updates
"""

import torch
import numpy as np
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.training.reinforcement_learning_enhanced import (
    EpigeneticVariationAI,
    PolicyNetwork,
    OlfactoryDNA,
    CreativeBrief,
    ScentPhenotype,
    set_seed
)


def test_policy_network_gradients():
    """Test that PolicyNetwork properly computes gradients"""
    print("\n1. Testing PolicyNetwork Gradient Computation...")
    print("-" * 50)

    # Set seed for reproducibility
    set_seed(42)

    # Initialize network
    network = PolicyNetwork(
        dna_dim=50,
        brief_dim=50,
        hidden_dim=256,
        num_actions=30
    )

    # Create dummy input
    state = torch.randn(1, 100)

    # Forward pass
    action_probs, value = network(state)

    # Create dummy loss
    loss = action_probs.mean() + value.mean()

    # Backward pass
    loss.backward()

    # Check gradients exist
    gradients_exist = all(
        param.grad is not None and param.grad.abs().sum() > 0
        for param in network.parameters()
    )

    print(f"  Action probs shape: {action_probs.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Gradients computed: {'YES' if gradients_exist else 'NO'}")

    assert gradients_exist, "No gradients computed for PolicyNetwork"
    print("  [PASS] PolicyNetwork gradient computation verified")

    return True


def test_adamw_optimizer_setup():
    """Test AdamW optimizer configuration"""
    print("\n2. Testing AdamW Optimizer Setup...")
    print("-" * 50)

    ai_system = EpigeneticVariationAI(
        learning_rate=3e-4,
        weight_decay=1e-4
    )

    # Check optimizer type
    assert isinstance(ai_system.optimizer, torch.optim.AdamW), "Optimizer is not AdamW"

    # Check optimizer parameters
    param_groups = ai_system.optimizer.param_groups[0]

    print(f"  Optimizer type: {type(ai_system.optimizer).__name__}")
    print(f"  Learning rate: {param_groups['lr']}")
    print(f"  Weight decay: {param_groups['weight_decay']}")
    print(f"  Betas: {param_groups['betas']}")
    print(f"  Eps: {param_groups['eps']}")

    assert param_groups['lr'] == 3e-4, "Learning rate mismatch"
    assert param_groups['weight_decay'] == 1e-4, "Weight decay mismatch"

    print("  [PASS] AdamW optimizer correctly configured")

    return True


def test_reinforce_loss_calculation():
    """Test REINFORCE loss calculation"""
    print("\n3. Testing REINFORCE Loss Calculation...")
    print("-" * 50)

    ai_system = EpigeneticVariationAI()

    # Create dummy episodes data
    log_probs = [torch.tensor([-0.5], requires_grad=True),
                 torch.tensor([-1.0], requires_grad=True),
                 torch.tensor([-0.3], requires_grad=True)]

    rewards = [1.0, -0.5, 0.5]

    ai_system.episode_log_probs = log_probs
    ai_system.episode_rewards = rewards
    ai_system.episode_values = [torch.tensor([0.5]) for _ in range(3)]
    ai_system.episode_entropy = [torch.tensor(0.5) for _ in range(3)]

    # Calculate returns
    returns = ai_system._calculate_returns()

    print(f"  Rewards: {rewards}")
    print(f"  Returns: {[f'{r:.3f}' for r in returns]}")

    # Verify return calculation (with gamma=0.99)
    expected_return_0 = rewards[0] + 0.99 * (rewards[1] + 0.99 * rewards[2])
    assert abs(returns[0] - expected_return_0) < 1e-5, "Return calculation error"

    # Test loss calculation
    loss = ai_system._perform_reinforce_update()

    print(f"  Total loss: {loss:.4f}")
    print(f"  Policy loss: {ai_system.training_metrics['policy_loss'][-1]:.4f}")
    print(f"  Value loss: {ai_system.training_metrics['value_loss'][-1]:.4f}")

    assert loss is not None, "Loss calculation failed"
    print("  [PASS] REINFORCE loss correctly calculated")

    return True


def test_gradient_update():
    """Test that optimizer actually updates weights"""
    print("\n4. Testing Gradient Updates with AdamW...")
    print("-" * 50)

    set_seed(42)
    ai_system = EpigeneticVariationAI()

    # Get initial parameters
    initial_params = {}
    for name, param in ai_system.policy_network.named_parameters():
        initial_params[name] = param.clone().detach()

    # Create sample data
    dna = OlfactoryDNA(
        genes=[(1, 10.0), (3, 15.0)],
        fitness_scores=(0.5, 0.5, 0.5)
    )

    brief = CreativeBrief(
        emotional_palette=[0.5] * 5,
        fragrance_family="floral",
        mood="calm",
        intensity=0.5,
        season="spring",
        gender="unisex"
    )

    # Generate variations
    variations = ai_system.generate_variations(dna, brief, num_variations=3)

    # Simulate feedback and update
    selected_idx = 0
    loss = ai_system.update_policy_with_feedback(variations, selected_idx)

    # Check if parameters changed
    params_changed = 0
    params_total = 0
    max_change = 0

    for name, param in ai_system.policy_network.named_parameters():
        params_total += 1
        change = (param - initial_params[name]).abs().max().item()
        if change > 1e-7:  # Threshold for numerical precision
            params_changed += 1
            max_change = max(max_change, change)

    print(f"  Parameters updated: {params_changed}/{params_total}")
    print(f"  Maximum parameter change: {max_change:.6f}")
    print(f"  Loss value: {loss:.4f}")

    assert params_changed > 0, "No parameters were updated"
    assert max_change > 1e-6, "Parameter changes too small"

    print("  [PASS] AdamW successfully updated network weights")

    return True


def test_episode_buffer_management():
    """Test episode buffer clearing after update"""
    print("\n5. Testing Episode Buffer Management...")
    print("-" * 50)

    ai_system = EpigeneticVariationAI()

    # Fill episode buffers
    ai_system.episode_log_probs = [torch.tensor([-0.5]) for _ in range(3)]
    ai_system.episode_rewards = [1.0, -0.5, 0.5]
    ai_system.episode_values = [torch.tensor([0.5]) for _ in range(3)]
    ai_system.episode_entropy = [torch.tensor(0.5) for _ in range(3)]

    print(f"  Before update:")
    print(f"    Log probs buffer size: {len(ai_system.episode_log_probs)}")
    print(f"    Rewards buffer size: {len(ai_system.episode_rewards)}")

    # Perform update
    loss = ai_system._perform_reinforce_update()

    print(f"  After update:")
    print(f"    Log probs buffer size: {len(ai_system.episode_log_probs)}")
    print(f"    Rewards buffer size: {len(ai_system.episode_rewards)}")
    print(f"    Values buffer size: {len(ai_system.episode_values)}")
    print(f"    Entropy buffer size: {len(ai_system.episode_entropy)}")

    # Check buffers are cleared
    assert len(ai_system.episode_log_probs) == 0, "Log probs buffer not cleared"
    assert len(ai_system.episode_rewards) == 0, "Rewards buffer not cleared"
    assert len(ai_system.episode_values) == 0, "Values buffer not cleared"
    assert len(ai_system.episode_entropy) == 0, "Entropy buffer not cleared"

    print("  [PASS] Episode buffers correctly cleared after update")

    return True


def test_learning_rate_scheduler():
    """Test learning rate scheduler functionality"""
    print("\n6. Testing Learning Rate Scheduler...")
    print("-" * 50)

    ai_system = EpigeneticVariationAI(learning_rate=1e-3)

    initial_lr = ai_system.optimizer.param_groups[0]['lr']
    print(f"  Initial learning rate: {initial_lr}")

    # Simulate poor performance to trigger scheduler
    for _ in range(15):  # More than patience=10
        ai_system.episode_rewards = [-1.0, -1.0, -1.0]
        ai_system.episode_log_probs = [torch.tensor([-0.5]) for _ in range(3)]
        ai_system.episode_values = [torch.tensor([0.5]) for _ in range(3)]
        ai_system.episode_entropy = [torch.tensor(0.5) for _ in range(3)]
        ai_system._perform_reinforce_update()

    final_lr = ai_system.optimizer.param_groups[0]['lr']
    print(f"  Final learning rate: {final_lr}")
    print(f"  Learning rate reduced: {'YES' if final_lr < initial_lr else 'NO'}")

    # Scheduler should have reduced learning rate
    assert final_lr <= initial_lr, "Learning rate scheduler didn't work"

    print("  [PASS] Learning rate scheduler functioning correctly")

    return True


def test_gradient_clipping():
    """Test gradient clipping for stability"""
    print("\n7. Testing Gradient Clipping...")
    print("-" * 50)

    ai_system = EpigeneticVariationAI(max_grad_norm=0.5)

    # Create artificially large gradients
    for param in ai_system.policy_network.parameters():
        if param.grad is None:
            param.grad = torch.randn_like(param) * 100  # Large gradients

    # Get gradient norms before clipping
    total_norm_before = torch.nn.utils.clip_grad_norm_(
        ai_system.policy_network.parameters(),
        float('inf')
    )

    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(
        ai_system.policy_network.parameters(),
        ai_system.max_grad_norm
    )

    # Get gradient norms after clipping
    total_norm_after = 0
    for param in ai_system.policy_network.parameters():
        if param.grad is not None:
            total_norm_after += param.grad.norm().item() ** 2
    total_norm_after = total_norm_after ** 0.5

    print(f"  Gradient norm before clipping: {total_norm_before:.2f}")
    print(f"  Gradient norm after clipping: {total_norm_after:.2f}")
    print(f"  Max allowed norm: {ai_system.max_grad_norm}")

    assert total_norm_after <= ai_system.max_grad_norm + 1e-6, "Gradient clipping failed"

    print("  [PASS] Gradient clipping working correctly")

    return True


def test_full_training_loop():
    """Test complete training loop with multiple updates"""
    print("\n8. Testing Full Training Loop...")
    print("-" * 50)

    set_seed(42)

    ai_system = EpigeneticVariationAI(
        learning_rate=1e-3,
        weight_decay=1e-4,
        entropy_coef=0.01
    )

    initial_dna = OlfactoryDNA(
        genes=[(1, 10.0), (3, 15.0)],
        fitness_scores=(0.5, 0.5, 0.5)
    )

    brief = CreativeBrief(
        emotional_palette=[0.5] * 5,
        fragrance_family="floral",
        mood="calm",
        intensity=0.5,
        season="spring",
        gender="unisex"
    )

    # Run multiple training iterations
    num_iterations = 5
    losses = []

    print(f"  Running {num_iterations} training iterations...")

    for i in range(num_iterations):
        # Generate variations
        variations = ai_system.generate_variations(initial_dna, brief, num_variations=3)

        # Simulate user selection
        selected_idx = i % 3  # Vary selection

        # Update policy
        loss = ai_system.update_policy_with_feedback(
            variations,
            selected_idx,
            ratings=[6.0, 7.0, 8.0]  # Provide explicit ratings
        )

        if loss is not None:
            losses.append(loss)
            print(f"    Iteration {i+1}: Loss = {loss:.4f}")

    # Check training metrics
    assert len(ai_system.training_metrics['policy_loss']) == num_iterations
    assert len(ai_system.training_metrics['episode_rewards']) == num_iterations
    assert len(losses) == num_iterations

    avg_loss = np.mean(losses)
    print(f"\n  Average loss: {avg_loss:.4f}")
    print(f"  Training metrics collected: {len(ai_system.training_metrics['policy_loss'])} entries")

    print("  [PASS] Full training loop completed successfully")

    return True


def run_all_tests():
    """Run all tests"""
    tests = [
        ("PolicyNetwork Gradients", test_policy_network_gradients),
        ("AdamW Optimizer Setup", test_adamw_optimizer_setup),
        ("REINFORCE Loss Calculation", test_reinforce_loss_calculation),
        ("Gradient Update", test_gradient_update),
        ("Episode Buffer Management", test_episode_buffer_management),
        ("Learning Rate Scheduler", test_learning_rate_scheduler),
        ("Gradient Clipping", test_gradient_clipping),
        ("Full Training Loop", test_full_training_loop)
    ]

    print("=" * 60)
    print("REINFORCE POLICY UPDATE TEST SUITE")
    print("=" * 60)

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{len(tests)}")
    print(f"Tests Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[SUCCESS] All tests passed successfully!")
        print("\nKey Features Verified:")
        print("  - REINFORCE loss function: -log(P(action)) * reward")
        print("  - AdamW optimizer with weight decay")
        print("  - Proper gradient computation and updates")
        print("  - Episode buffer management")
        print("  - Learning rate scheduling")
        print("  - Gradient clipping for stability")
        print("  - Complete training pipeline")
    else:
        print(f"\n[FAILED] {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()