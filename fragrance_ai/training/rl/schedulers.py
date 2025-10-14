"""
RL Training Schedulers
Entropy annealing, learning rate scheduling, reward normalization
"""

import math
import numpy as np
from typing import Optional, Literal
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Entropy Coefficient Scheduler
# =============================================================================

@dataclass
class EntropySchedulerConfig:
    """Entropy scheduler configuration"""
    initial_entropy: float = 0.01
    final_entropy: float = 0.001
    decay_steps: int = 10000
    schedule_type: Literal["linear", "cosine", "exponential"] = "cosine"


class EntropyScheduler:
    """
    Entropy coefficient scheduler for PPO

    Anneals entropy coefficient from high (exploration) to low (exploitation)
    over the course of training.
    """

    def __init__(self, config: EntropySchedulerConfig):
        self.initial_entropy = config.initial_entropy
        self.final_entropy = config.final_entropy
        self.decay_steps = config.decay_steps
        self.schedule_type = config.schedule_type

        self.current_step = 0

        logger.info(f"EntropyScheduler initialized: {config.schedule_type} "
                   f"from {config.initial_entropy} to {config.final_entropy} "
                   f"over {config.decay_steps} steps")

    def get_entropy_coef(self, step: Optional[int] = None) -> float:
        """
        Get entropy coefficient for current step

        Args:
            step: Current training step (uses internal counter if None)

        Returns:
            Entropy coefficient value
        """
        if step is None:
            step = self.current_step
            self.current_step += 1

        if step >= self.decay_steps:
            return self.final_entropy

        # Calculate progress (0 to 1)
        progress = step / self.decay_steps

        # Apply schedule
        if self.schedule_type == "linear":
            # Linear decay
            entropy_coef = self.initial_entropy - progress * (self.initial_entropy - self.final_entropy)

        elif self.schedule_type == "cosine":
            # Cosine annealing
            # Formula: final + 0.5 * (initial - final) * (1 + cos(π * progress))
            entropy_coef = self.final_entropy + 0.5 * (self.initial_entropy - self.final_entropy) * (1 + math.cos(math.pi * progress))

        elif self.schedule_type == "exponential":
            # Exponential decay
            decay_rate = math.log(self.initial_entropy / self.final_entropy) / self.decay_steps
            entropy_coef = self.initial_entropy * math.exp(-decay_rate * step)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return max(entropy_coef, self.final_entropy)

    def reset(self):
        """Reset scheduler to initial state"""
        self.current_step = 0


# =============================================================================
# Reward Normalizer
# =============================================================================

@dataclass
class RewardNormalizerConfig:
    """Reward normalizer configuration"""
    window_size: int = 1000  # Last 1k steps
    update_mean_std: bool = True
    epsilon: float = 1e-8
    clip_range: float = 10.0


class RewardNormalizer:
    """
    Reward normalizer using rolling window statistics

    Normalizes rewards using mean and std from recent history
    to stabilize training across different reward scales.
    """

    def __init__(self, config: RewardNormalizerConfig):
        self.window_size = config.window_size
        self.update_mean_std = config.update_mean_std
        self.epsilon = config.epsilon
        self.clip_range = config.clip_range

        # Rolling window for rewards
        self.reward_history = deque(maxlen=self.window_size)

        # Running statistics
        self.mean = 0.0
        self.std = 1.0
        self.count = 0

        logger.info(f"RewardNormalizer initialized: window_size={self.window_size}, "
                   f"clip_range={self.clip_range}")

    def normalize(self, reward: float, update: bool = True) -> float:
        """
        Normalize reward using rolling statistics

        Args:
            reward: Raw reward value
            update: Whether to update statistics with this reward

        Returns:
            Normalized reward
        """
        if update and self.update_mean_std:
            # Add to history
            self.reward_history.append(reward)
            self.count += 1

            # Update statistics if we have enough samples
            if len(self.reward_history) >= min(100, self.window_size // 10):
                rewards = np.array(self.reward_history)
                self.mean = np.mean(rewards)
                self.std = np.std(rewards) + self.epsilon

        # Normalize
        normalized = (reward - self.mean) / self.std

        # Clip to prevent extreme values
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return float(normalized)

    def normalize_batch(self, rewards: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalize batch of rewards

        Args:
            rewards: Array of raw rewards
            update: Whether to update statistics

        Returns:
            Array of normalized rewards
        """
        normalized_rewards = []

        for reward in rewards:
            normalized = self.normalize(reward, update=update)
            normalized_rewards.append(normalized)

        return np.array(normalized_rewards)

    def get_statistics(self) -> dict:
        """Get current normalization statistics"""
        return {
            'mean': self.mean,
            'std': self.std,
            'count': self.count,
            'window_size': len(self.reward_history),
            'epsilon': self.epsilon
        }

    def reset(self):
        """Reset normalizer statistics"""
        self.reward_history.clear()
        self.mean = 0.0
        self.std = 1.0
        self.count = 0


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

@dataclass
class LRSchedulerConfig:
    """Learning rate scheduler configuration"""
    initial_lr: float = 3e-4
    final_lr: float = 1e-5
    decay_steps: int = 10000
    schedule_type: Literal["linear", "cosine", "exponential"] = "cosine"


class LearningRateScheduler:
    """
    Learning rate scheduler

    Similar to entropy scheduler but for learning rate.
    """

    def __init__(self, config: LRSchedulerConfig):
        self.initial_lr = config.initial_lr
        self.final_lr = config.final_lr
        self.decay_steps = config.decay_steps
        self.schedule_type = config.schedule_type

        self.current_step = 0

        logger.info(f"LRScheduler initialized: {config.schedule_type} "
                   f"from {config.initial_lr} to {config.final_lr} "
                   f"over {config.decay_steps} steps")

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate for current step"""
        if step is None:
            step = self.current_step
            self.current_step += 1

        if step >= self.decay_steps:
            return self.final_lr

        progress = step / self.decay_steps

        if self.schedule_type == "linear":
            lr = self.initial_lr - progress * (self.initial_lr - self.final_lr)

        elif self.schedule_type == "cosine":
            lr = self.final_lr + 0.5 * (self.initial_lr - self.final_lr) * (1 + math.cos(math.pi * progress))

        elif self.schedule_type == "exponential":
            decay_rate = math.log(self.initial_lr / self.final_lr) / self.decay_steps
            lr = self.initial_lr * math.exp(-decay_rate * step)

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return max(lr, self.final_lr)

    def reset(self):
        """Reset scheduler"""
        self.current_step = 0


# =============================================================================
# Adaptive KL Controller
# =============================================================================

class AdaptiveKLController:
    """
    Adaptive KL divergence controller

    Dynamically adjusts clip_eps or learning rate based on KL divergence
    to maintain stable training.
    """

    def __init__(
        self,
        target_kl: float = 0.01,
        kl_tolerance: float = 0.5,
        adapt_rate: float = 1.5
    ):
        """
        Initialize adaptive KL controller

        Args:
            target_kl: Target KL divergence
            kl_tolerance: Tolerance range (target_kl * kl_tolerance)
            adapt_rate: Adaptation rate multiplier
        """
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.adapt_rate = adapt_rate

        self.current_clip_eps = 0.2  # Initial clip_eps

        logger.info(f"AdaptiveKLController initialized: target_kl={target_kl}")

    def update(self, kl_divergence: float) -> float:
        """
        Update clip_eps based on KL divergence

        Args:
            kl_divergence: Measured KL divergence

        Returns:
            Updated clip_eps value
        """
        if kl_divergence > self.target_kl * (1 + self.kl_tolerance):
            # KL too high - reduce clip_eps (smaller updates)
            self.current_clip_eps /= self.adapt_rate
            logger.info(f"KL too high ({kl_divergence:.4f}), reducing clip_eps to {self.current_clip_eps:.4f}")

        elif kl_divergence < self.target_kl * (1 - self.kl_tolerance):
            # KL too low - increase clip_eps (larger updates)
            self.current_clip_eps *= self.adapt_rate
            logger.info(f"KL too low ({kl_divergence:.4f}), increasing clip_eps to {self.current_clip_eps:.4f}")

        # Clip to reasonable range
        self.current_clip_eps = np.clip(self.current_clip_eps, 0.05, 0.5)

        return self.current_clip_eps

    def get_clip_eps(self) -> float:
        """Get current clip_eps"""
        return self.current_clip_eps


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'EntropyScheduler',
    'EntropySchedulerConfig',
    'RewardNormalizer',
    'RewardNormalizerConfig',
    'LearningRateScheduler',
    'LRSchedulerConfig',
    'AdaptiveKLController'
]
