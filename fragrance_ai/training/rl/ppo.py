# fragrance_ai/training/rl/ppo.py
"""
PPO (Proximal Policy Optimization) implementation
Complete Actor-Critic with GAE and clipped objective
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Neural Networks
# ============================================================================

class PolicyNetwork(nn.Module):
    """Actor network for policy"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class ValueNetwork(nn.Module):
    """Critic network for value estimation"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================================
# Rollout Buffer
# ============================================================================

@dataclass
class RolloutBuffer:
    """Buffer for collecting rollout experiences"""

    states: List[torch.Tensor] = None
    actions: List[torch.Tensor] = None
    log_probs: List[torch.Tensor] = None
    rewards: List[float] = None
    values: List[torch.Tensor] = None
    dones: List[bool] = None

    def __post_init__(self):
        if self.states is None:
            self.clear()

    def add(self, state: torch.Tensor, action: torch.Tensor,
            log_prob: torch.Tensor, reward: float,
            value: torch.Tensor, done: bool):
        """Add experience to buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        """Clear buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors"""
        if not self.states:
            return {}

        return {
            'states': torch.cat(self.states),
            'actions': torch.cat(self.actions),
            'log_probs': torch.cat(self.log_probs),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'values': torch.cat(self.values),
            'dones': torch.tensor(self.dones, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.states)


# ============================================================================
# GAE Computation
# ============================================================================

def compute_gae(rewards: torch.Tensor, values: torch.Tensor,
                dones: torch.Tensor, gamma: float = 0.99,
                gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)

    Args:
        rewards: Reward sequence
        values: Value estimates
        dones: Done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)

    last_gae = 0
    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # Temporal difference
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE calculation
        advantages[t] = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        last_gae = advantages[t]

    returns = advantages + values

    # Standardize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


# ============================================================================
# PPO Trainer
# ============================================================================

class PPOTrainer:
    """PPO training algorithm"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        """
        Initialize PPO trainer

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_eps: PPO clipping epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping max norm
            device: Device to use (cpu/cuda)
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        # Buffer
        self.buffer = RolloutBuffer()

        # Tracking
        self.update_count = 0

    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """
        Select action using current policy

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        with torch.no_grad():
            state = state.to(self.device)
            probs = self.policy(state)
            value = self.value(state)

            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state: torch.Tensor, action: int,
                        log_prob: float, reward: float,
                        value: float, done: bool):
        """Store transition in buffer"""
        self.buffer.add(
            state=state,
            action=torch.tensor([action]),
            log_prob=torch.tensor([log_prob]),
            reward=reward,
            value=torch.tensor([value]),
            done=done
        )

    def update(self, batch_size: int = 64, n_epochs: int = 4) -> Dict[str, float]:
        """
        Update policy and value networks using PPO

        Args:
            batch_size: Minibatch size
            n_epochs: Number of optimization epochs

        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) == 0:
            return {}

        # Get data from buffer
        data = self.buffer.get()
        states = data['states'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        rewards = data['rewards'].to(self.device)
        values = data['values'].to(self.device)
        dones = data['dones'].to(self.device)

        # Compute GAE
        advantages, returns = compute_gae(
            rewards, values, dones,
            self.gamma, self.gae_lambda
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        # PPO update loop
        for epoch in range(n_epochs):
            # Create random indices for minibatches
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]

                # Get minibatch
                mb_states = states[batch_indices]
                mb_actions = actions[batch_indices]
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # Forward pass
                probs = self.policy(mb_states)
                values_pred = self.value(mb_states).squeeze()

                # Calculate log probs and entropy
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(mb_actions.squeeze())
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs.squeeze())
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (ensure shapes match)
                value_loss = nn.functional.mse_loss(values_pred.squeeze(), mb_returns)

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()

                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        # Clear buffer after update
        self.buffer.clear()
        self.update_count += 1

        # Calculate average metrics
        num_updates = n_epochs * ((len(states) - 1) // batch_size + 1)
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'update_count': self.update_count
        }

        logger.info(f"PPO update {self.update_count}: "
                   f"policy_loss={metrics['policy_loss']:.4f}, "
                   f"value_loss={metrics['value_loss']:.4f}, "
                   f"entropy={metrics['entropy']:.4f}")

        return metrics

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        logger.info(f"Model loaded from {path}")