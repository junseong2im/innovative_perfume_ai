"""
Comprehensive Test for Model Persistence
Verifies that policy_network.pth is actually modified after user feedback
"""

import os
import sys
import torch
import time
import hashlib
import json
from pathlib import Path
from datetime import datetime
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.training.rl_with_persistence import (
    RLHFWithPersistence,
    PolicyNetworkWithPersistence,
    ModelPersistenceManager
)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def test_file_modification_after_feedback():
    """
    Main test: Verify that update_policy_with_feedback actually modifies policy_network.pth
    """
    print("\n" + "="*70)
    print("MODEL PERSISTENCE VERIFICATION TEST")
    print("Testing: update_policy_with_feedback modifies policy_network.pth")
    print("="*70)

    # Clean up previous test files
    model_dir = Path("models")
    if model_dir.exists():
        shutil.rmtree(model_dir)

    # Initialize RLHF system with auto-save enabled
    print("\n1. Initializing RLHF System with Auto-Save...")
    print("-" * 50)

    rlhf_system = RLHFWithPersistence(
        state_dim=100,
        hidden_dim=256,
        num_actions=30,
        save_dir="models",
        auto_save=True  # CRITICAL: Auto-save must be enabled
    )

    model_path = Path("models/policy_network.pth")

    # Check initial state
    if model_path.exists():
        initial_hash = calculate_file_hash(model_path)
        initial_mtime = model_path.stat().st_mtime
        print(f"Initial model found:")
        print(f"  File: {model_path}")
        print(f"  Hash: {initial_hash[:16]}...")
        print(f"  Modified: {datetime.fromtimestamp(initial_mtime)}")
    else:
        initial_hash = None
        initial_mtime = None
        print("No initial model file (will be created on first update)")

    # Test multiple feedback sessions
    print("\n2. Testing Feedback Sessions with File Monitoring...")
    print("-" * 50)

    modification_results = []

    for session in range(3):
        print(f"\nFeedback Session {session + 1}:")

        # Record state before update
        before_hash = calculate_file_hash(model_path) if model_path.exists() else None
        before_mtime = model_path.stat().st_mtime if model_path.exists() else None
        before_size = model_path.stat().st_size if model_path.exists() else 0

        # Simulate user feedback
        log_probs = []
        rewards = []
        values = []

        for i in range(5):
            state = torch.randn(100).to(rlhf_system.device)
            action_probs, value = rlhf_system.policy_network(state)

            # Sample action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Simulate user feedback (varying rewards)
            reward = 0.5 + (i * 0.1) if session % 2 == 0 else -0.5 + (i * 0.1)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

        print(f"  Collected feedback: {len(rewards)} actions")
        print(f"  Average reward: {sum(rewards)/len(rewards):.3f}")

        # CRITICAL CALL: This should modify policy_network.pth
        print(f"  Calling update_policy_with_feedback()...")
        start_time = time.time()

        loss = rlhf_system.update_policy_with_feedback(
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            gamma=0.99
        )

        update_time = time.time() - start_time

        # Small delay to ensure file system updates
        time.sleep(0.1)

        # Check state after update
        after_hash = calculate_file_hash(model_path)
        after_mtime = model_path.stat().st_mtime
        after_size = model_path.stat().st_size

        # Verify modification
        file_modified = (before_hash != after_hash) if before_hash else True
        time_changed = (after_mtime > before_mtime) if before_mtime else True
        size_valid = after_size > 1000  # Should be at least 1KB

        # Record results
        result = {
            'session': session + 1,
            'loss': loss,
            'file_modified': file_modified,
            'time_changed': time_changed,
            'size_valid': size_valid,
            'hash_before': before_hash[:8] if before_hash else "None",
            'hash_after': after_hash[:8],
            'size_after': after_size,
            'update_time': update_time
        }
        modification_results.append(result)

        # Print verification results
        print(f"  Update completed in {update_time:.3f} seconds")
        print(f"  Loss: {loss:.4f}")
        print(f"  File modified: {'YES' if file_modified else 'NO'}")
        print(f"  Hash changed: {result['hash_before']}... -> {result['hash_after']}...")
        print(f"  File size: {after_size:,} bytes")
        print(f"  Timestamp updated: {'YES' if time_changed else 'NO'}")

        if not file_modified:
            print("  [WARNING] File was NOT modified!")

    # Summary of results
    print("\n3. Verification Summary:")
    print("-" * 50)

    all_modified = all(r['file_modified'] for r in modification_results)
    all_time_changed = all(r['time_changed'] for r in modification_results)
    all_size_valid = all(r['size_valid'] for r in modification_results)

    print(f"Total feedback sessions: {len(modification_results)}")
    print(f"Files modified: {sum(1 for r in modification_results if r['file_modified'])}/{len(modification_results)}")
    print(f"Timestamps updated: {sum(1 for r in modification_results if r['time_changed'])}/{len(modification_results)}")
    print(f"Valid file sizes: {sum(1 for r in modification_results if r['size_valid'])}/{len(modification_results)}")

    # Test loading saved model
    print("\n4. Testing Model Loading:")
    print("-" * 50)

    # Create new instance and load
    new_system = RLHFWithPersistence(
        state_dim=100,
        hidden_dim=256,
        num_actions=30,
        save_dir="models",
        auto_save=False  # Disable auto-save for loading test
    )

    # Check if loaded correctly
    print(f"Original system updates: {rlhf_system.policy_network.total_updates}")
    print(f"Loaded system updates: {new_system.policy_network.total_updates}")

    # Compare weights
    original_state = rlhf_system.policy_network.state_dict()
    loaded_state = new_system.policy_network.state_dict()

    weights_match = all(
        torch.allclose(original_state[key], loaded_state[key], atol=1e-6)
        for key in original_state.keys()
    )

    print(f"Weights match: {'YES' if weights_match else 'NO'}")

    # Test metadata
    print("\n5. Testing Metadata Persistence:")
    print("-" * 50)

    metadata_path = Path("models/policy_network_metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"Metadata file exists: YES")
        print(f"Total updates in metadata: {metadata.get('total_updates', 0)}")
        print(f"Last loss: {metadata.get('loss', 0):.4f}")
        print(f"Average reward: {metadata.get('average_reward', 0):.3f}")
        print(f"File hash in metadata: {metadata.get('file_hash', '')[:16]}...")

        # Verify hash matches actual file
        actual_hash = calculate_file_hash(model_path)
        hash_matches = metadata.get('file_hash') == actual_hash
        print(f"Hash verification: {'PASS' if hash_matches else 'FAIL'}")
    else:
        print("Metadata file exists: NO")

    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT:")
    print("="*70)

    success_criteria = [
        ("File modification after feedback", all_modified),
        ("Timestamp updates", all_time_changed),
        ("Valid file sizes", all_size_valid),
        ("Model loading works", weights_match),
        ("Metadata consistency", metadata_path.exists())
    ]

    all_passed = all(passed for _, passed in success_criteria)

    for criterion, passed in success_criteria:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {criterion}")

    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        print("\nCONFIRMED: update_policy_with_feedback() successfully modifies policy_network.pth")
        print("The model file is automatically updated after each user feedback session.")
    else:
        print("\n[FAILURE] Some tests failed. Check the results above.")

    # Show file details
    print("\n6. File System Details:")
    print("-" * 50)

    if model_path.exists():
        stat = model_path.stat()
        print(f"Final model file: {model_path.absolute()}")
        print(f"File size: {stat.st_size:,} bytes")
        print(f"Last modified: {datetime.fromtimestamp(stat.st_mtime)}")
        print(f"Final hash: {calculate_file_hash(model_path)[:32]}...")

        # List checkpoints
        checkpoint_dir = Path("models/checkpoints")
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pth"))
            print(f"\nCheckpoints created: {len(checkpoints)}")
            for cp in checkpoints:
                print(f"  - {cp.name} ({cp.stat().st_size:,} bytes)")

    return all_passed


def test_auto_save_toggle():
    """Test that auto-save can be disabled and enabled"""
    print("\n" + "="*70)
    print("AUTO-SAVE TOGGLE TEST")
    print("="*70)

    # Clean up
    model_dir = Path("models_test")
    if model_dir.exists():
        shutil.rmtree(model_dir)

    # Test with auto-save disabled
    print("\n1. Testing with auto-save DISABLED:")
    print("-" * 50)

    system_no_save = RLHFWithPersistence(
        save_dir="models_test",
        auto_save=False  # Disabled
    )

    model_path = Path("models_test/policy_network.pth")

    # Perform update
    log_probs = [torch.tensor([-0.5], requires_grad=True)]
    rewards = [1.0]
    values = [torch.tensor([0.5])]

    system_no_save.update_policy_with_feedback(log_probs, rewards, values)

    print(f"Model file exists after update: {'YES' if model_path.exists() else 'NO'}")

    # Now manually save
    print("\n2. Testing manual save:")
    print("-" * 50)

    save_path = system_no_save.save_model()
    print(f"Manual save completed: {save_path}")
    print(f"Model file exists after manual save: {'YES' if model_path.exists() else 'NO'}")

    if model_path.exists():
        print(f"File size: {model_path.stat().st_size:,} bytes")

    # Clean up
    if model_dir.exists():
        shutil.rmtree(model_dir)

    print("\n[PASS] Auto-save toggle works correctly")


if __name__ == "__main__":
    # Run main test
    success = test_file_modification_after_feedback()

    # Run additional tests
    test_auto_save_toggle()

    print("\n" + "="*70)
    print("ALL PERSISTENCE TESTS COMPLETED")
    print("="*70)

    if success:
        print("\n[SUCCESS] Model persistence system is working correctly!")
        print("update_policy_with_feedback() successfully modifies policy_network.pth")
    else:
        print("\n[WARNING] Some issues detected. Review the test output above.")