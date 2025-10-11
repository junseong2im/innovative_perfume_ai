# fragrance_ai/training/rl/reinforce.py
"""
REINFORCE implementation - Simple policy gradient algorithm
Minimal but complete implementation with proper reward normalization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Policy Network
# ============================================================================

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


# ============================================================================
# REINFORCE Trainer
# ============================================================================

class REINFORCETrainer:
    """REINFORCE training algorithm (minimal implementation)"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        """
        Initialize REINFORCE trainer

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            lr: Learning rate
            gamma: Discount factor
            device: Device to use (cpu/cuda)
        """
        self.gamma = gamma
        self.device = device

        # Policy network
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Episode storage
        self.saved_log_probs = []
        self.rewards = []

        # Tracking
        self.update_count = 0

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        Select action using current policy

        Returns:
            action: Selected action index
            log_prob: Log probability tensor (not detached)
        """
        state = state.to(self.device)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Save for later update
        self.saved_log_probs.append(log_prob)

        return action.item(), log_prob

    def store_reward(self, reward: float):
        """Store reward for current step"""
        self.rewards.append(reward)

    def update(self, normalize_rewards: bool = True) -> Dict[str, float]:
        """
        Update policy using REINFORCE algorithm

        Core formula: loss = -sum(log_prob * reward)

        Args:
            normalize_rewards: Whether to normalize rewards

        Returns:
            Dictionary of training metrics
        """
        if not self.saved_log_probs:
            return {"loss": 0, "avg_reward": 0}

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns
        if normalize_rewards and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE loss: -sum(log_prob * return)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        loss = torch.cat(policy_loss).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Metrics
        avg_reward = np.mean(self.rewards) if self.rewards else 0

        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        self.update_count += 1

        metrics = {
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "update_count": self.update_count
        }

        logger.info(f"REINFORCE update {self.update_count}: "
                   f"loss={metrics['loss']:.4f}, "
                   f"avg_reward={metrics['avg_reward']:.4f}")

        return metrics

    def update_with_feedback(
        self,
        log_probs: List[torch.Tensor],
        rating: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Direct update with feedback (for RLHF integration)

        Args:
            log_probs: Log probabilities of actions
            rating: User rating (1-5), normalized to reward

        Returns:
            Update metrics
        """
        if not log_probs:
            return {"loss": 0, "reward": 0}

        # Reward calculation from rating
        if rating is not None:
            reward = (rating - 3) / 2.0  # Map [1,5] to [-1,1]
        else:
            reward = 1.0

        # REINFORCE loss
        loss = -sum(lp * reward for lp in log_probs)

        # Critical: Call optimizer.step()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        metrics = {
            "loss": loss.item(),
            "reward": reward,
            "algorithm": "REINFORCE",
            "update_count": self.update_count
        }

        logger.info(f"REINFORCE feedback update: "
                   f"rating={rating}, reward={reward:.2f}, "
                   f"loss={metrics['loss']:.4f}")

        return metrics

    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'policy': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']
        logger.info(f"Model loaded from {path}")


# ============================================================================
# Baseline REINFORCE (with value baseline)
# ============================================================================

class BaselineREINFORCE(REINFORCETrainer):
    """REINFORCE with baseline for variance reduction"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cpu'
    ):
        super().__init__(state_dim, action_dim, lr, gamma, device)

        # Value network as baseline
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(device)

        # Add value network to optimizer
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()),
            lr=lr
        )

        self.saved_values = []

    def select_action(self, state: torch.Tensor) -> Tuple[int, float, float]:
        """
        Select action and estimate value

        Returns:
            action: Selected action
            log_prob: Log probability
            value: State value estimate
        """
        action, log_prob = super().select_action(state)

        # Estimate value
        value = self.value_net(state.to(self.device))
        self.saved_values.append(value)

        return action, log_prob, value.item()

    def update(self, normalize_advantages: bool = True) -> Dict[str, float]:
        """
        Update with baseline

        Returns:
            Training metrics
        """
        if not self.saved_log_probs:
            return {"loss": 0, "avg_reward": 0}

        # Calculate returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        values = torch.cat(self.saved_values)

        # Calculate advantages (return - baseline)
        advantages = returns - values.squeeze()

        # Normalize advantages
        if normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss = []
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            policy_loss.append(-log_prob * advantage.detach())

        # Value loss
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = torch.cat(policy_loss).sum() + 0.5 * value_loss

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        avg_reward = np.mean(self.rewards)
        self.saved_log_probs = []
        self.rewards = []
        self.saved_values = []
        self.update_count += 1

        return {
            "loss": loss.item(),
            "policy_loss": torch.cat(policy_loss).sum().item(),
            "value_loss": value_loss.item(),
            "avg_reward": avg_reward,
            "update_count": self.update_count
        }