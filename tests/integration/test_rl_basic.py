"""
Basic Test for RL Engine - No complex dependencies
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Test basic RL functionality without database imports
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np


class SimpleOlfactoryDNA:
    """Simple test class without database dependencies"""
    def __init__(self, dna_id, notes, genotype):
        self.dna_id = dna_id
        self.notes = notes
        self.genotype = genotype


class SimpleScentPhenotype:
    """Simple phenotype for testing"""
    def __init__(self, phenotype_id, based_on_dna, epigenetic_trigger,
                 variation_applied, recipe_adjusted, description):
        self.phenotype_id = phenotype_id
        self.based_on_dna = based_on_dna
        self.epigenetic_trigger = epigenetic_trigger
        self.variation_applied = variation_applied
        self.recipe_adjusted = recipe_adjusted
        self.description = description


class SimplePolicyNetwork(nn.Module):
    """Simplified Policy Network for testing"""
    def __init__(self, state_dim: int, action_dim: int):
        super(SimplePolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)


class SimpleRLEngine:
    """Simplified RL Engine for testing"""
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = SimplePolicyNetwork(state_dim, action_dim)
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(), lr=learning_rate)

        self.action_space = [
            "amplify_base_note_1", "silence_top_note_1", "add_new_note_rose",
            "add_new_note_vanilla", "shift_to_warm", "shift_to_fresh"
        ]

    def encode_state(self, dna, creative_brief: dict) -> torch.Tensor:
        """Simple state encoding"""
        state_vector = []

        # DNA features
        for i in range(5):
            if i < len(dna.notes):
                state_vector.append(dna.notes[i].get("intensity", 0.5))
            else:
                state_vector.append(0.0)

        # Brief features
        state_vector.extend([
            creative_brief.get("desired_intensity", 0.5),
            creative_brief.get("masculinity", 0.5),
            creative_brief.get("complexity", 0.5)
        ])

        # Pad to state_dim
        while len(state_vector) < self.state_dim:
            state_vector.append(0.0)

        return torch.FloatTensor(state_vector[:self.state_dim]).unsqueeze(0)

    def generate_variations(self, dna, feedback_brief: dict, num_options: int = 3):
        """Generate variations using policy network"""
        import random

        state = self.encode_state(dna, feedback_brief)
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)

        options = []
        saved_actions = []

        for i in range(num_options):
            action = dist.sample()
            action_name = self.action_space[action.item()]

            phenotype = SimpleScentPhenotype(
                phenotype_id=f"pheno_{random.randint(1000, 9999)}_{i}",
                based_on_dna=dna.dna_id,
                epigenetic_trigger=feedback_brief.get('theme', 'N/A'),
                variation_applied=action_name,
                recipe_adjusted=dna.genotype.copy(),
                description=f"Variation based on '{action_name}'"
            )

            options.append({
                "id": phenotype.phenotype_id,
                "phenotype": phenotype,
                "action": action.item(),
                "action_name": action_name,
                "log_prob": dist.log_prob(action)
            })

            saved_actions.append((action, dist.log_prob(action)))

        self.last_state = state
        self.last_saved_actions = saved_actions

        return options


def test_rl_engine():
    """Test basic RL engine functionality"""
    print("=" * 60)
    print("Basic RL Engine Test")
    print("=" * 60)

    # Initialize engine
    engine = SimpleRLEngine(state_dim=10, action_dim=6, learning_rate=0.001)
    print("[OK] RL Engine initialized")

    # Test DNA
    test_dna = SimpleOlfactoryDNA(
        dna_id="test_001",
        notes=[
            {"name": "bergamot", "intensity": 0.8},
            {"name": "jasmine", "intensity": 0.6},
            {"name": "sandalwood", "intensity": 0.4}
        ],
        genotype={"top": ["bergamot"], "heart": ["jasmine"], "base": ["sandalwood"]}
    )
    print("[OK] Test DNA created")

    # Test brief
    feedback_brief = {
        "desired_intensity": 0.7,
        "masculinity": 0.3,
        "complexity": 0.6,
        "theme": "warm and sensual",
        "story": "Evening romance"
    }
    print("[OK] Feedback brief created")

    # Generate variations
    options = engine.generate_variations(
        dna=test_dna,
        feedback_brief=feedback_brief,
        num_options=3
    )

    print(f"[OK] Generated {len(options)} variations:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt['action_name']} (ID: {opt['id']})")

    # Test state encoding
    state = engine.encode_state(test_dna, feedback_brief)
    print(f"[OK] State shape: {state.shape}")

    # Test policy network forward pass
    with torch.no_grad():
        action_probs = engine.policy_network(state)
        print(f"[OK] Action probabilities shape: {action_probs.shape}")
        print(f"  Sum of probabilities: {action_probs.sum().item():.4f}")

    # Test saved state and actions
    assert hasattr(engine, 'last_state')
    assert hasattr(engine, 'last_saved_actions')
    print(f"[OK] State and actions saved for learning")

    return True


def test_policy_network():
    """Test policy network separately"""
    print("\n" + "=" * 60)
    print("Policy Network Test")
    print("=" * 60)

    # Create network
    net = SimplePolicyNetwork(state_dim=10, action_dim=6)
    print(f"[OK] Network created with {sum(p.numel() for p in net.parameters())} parameters")

    # Test forward pass
    test_input = torch.randn(1, 10)
    output = net(test_input)

    assert output.shape == (1, 6)
    assert torch.allclose(output.sum(), torch.tensor(1.0), atol=1e-6)
    print(f"[OK] Forward pass successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output sum: {output.sum().item():.6f}")

    # Test batch processing
    batch_input = torch.randn(32, 10)
    batch_output = net(batch_input)

    assert batch_output.shape == (32, 6)
    print(f"[OK] Batch processing works (batch size: 32)")

    return True


def test_evolution_scenario():
    """Test complete evolution scenario"""
    print("\n" + "=" * 60)
    print("Evolution Scenario Test")
    print("=" * 60)

    # Initialize
    engine = SimpleRLEngine(state_dim=10, action_dim=6)

    # Initial DNA
    initial_dna = SimpleOlfactoryDNA(
        dna_id="evo_001",
        notes=[
            {"name": "lemon", "intensity": 0.9},
            {"name": "rose", "intensity": 0.5},
            {"name": "cedar", "intensity": 0.3}
        ],
        genotype={"composition": ["lemon", "rose", "cedar"]}
    )

    # Evolution requests
    evolution_requests = [
        {"theme": "warmer", "desired_intensity": 0.6},
        {"theme": "fresher", "desired_intensity": 0.8},
        {"theme": "mysterious", "desired_intensity": 0.5}
    ]

    for i, request in enumerate(evolution_requests, 1):
        print(f"\nEvolution {i}: {request['theme']}")

        options = engine.generate_variations(
            dna=initial_dna,
            feedback_brief=request,
            num_options=3
        )

        # Simulate user choice (choose first option)
        chosen = options[0]
        print(f"  Chosen: {chosen['action_name']}")

        # Update DNA for next iteration (simplified)
        initial_dna.notes.append({"name": f"evolved_{i}", "intensity": 0.4})

    print("\n[OK] Evolution scenario completed successfully")
    return True


if __name__ == "__main__":
    try:
        # Run all tests
        test_policy_network()
        test_rl_engine()
        test_evolution_scenario()

        print("\n" + "=" * 60)
        print("[SUCCESS] All Basic RL Tests Passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()