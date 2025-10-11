# fragrance_ai/training/rl/__init__.py
"""
Reinforcement Learning module for fragrance AI
Provides REINFORCE and PPO implementations
"""

from .reinforce import REINFORCETrainer, BaselineREINFORCE
from .ppo import PPOTrainer, PolicyNetwork, ValueNetwork, RolloutBuffer

from typing import Optional, Dict, Any


def create_rl_trainer(
    algorithm: str,
    state_dim: int,
    action_dim: int,
    **kwargs
) -> Any:
    """
    Factory function to create RL trainer

    Args:
        algorithm: "REINFORCE", "REINFORCE_BASELINE", or "PPO"
        state_dim: State space dimension
        action_dim: Action space dimension
        **kwargs: Additional algorithm-specific parameters

    Returns:
        RL trainer instance
    """
    algorithm = algorithm.upper()

    if algorithm == "REINFORCE":
        return REINFORCETrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=kwargs.get("lr", 3e-4),
            gamma=kwargs.get("gamma", 0.99),
            device=kwargs.get("device", "cpu")
        )

    elif algorithm == "REINFORCE_BASELINE":
        return BaselineREINFORCE(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=kwargs.get("lr", 3e-4),
            gamma=kwargs.get("gamma", 0.99),
            device=kwargs.get("device", "cpu")
        )

    elif algorithm == "PPO":
        return PPOTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=kwargs.get("lr", 3e-4),
            gamma=kwargs.get("gamma", 0.99),
            gae_lambda=kwargs.get("gae_lambda", 0.95),
            clip_eps=kwargs.get("clip_eps", 0.2),
            value_coef=kwargs.get("value_coef", 0.5),
            entropy_coef=kwargs.get("entropy_coef", 0.01),
            max_grad_norm=kwargs.get("max_grad_norm", 0.5),
            device=kwargs.get("device", "cpu")
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def get_default_config(algorithm: str) -> Dict[str, Any]:
    """
    Get default configuration for algorithm

    Args:
        algorithm: Algorithm name

    Returns:
        Default configuration dict
    """
    base_config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "device": "cpu"
    }

    if algorithm.upper() == "PPO":
        base_config.update({
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "batch_size": 64,
            "n_epochs": 4
        })

    return base_config


__all__ = [
    'REINFORCETrainer',
    'BaselineREINFORCE',
    'PPOTrainer',
    'PolicyNetwork',
    'ValueNetwork',
    'RolloutBuffer',
    'create_rl_trainer',
    'get_default_config'
]