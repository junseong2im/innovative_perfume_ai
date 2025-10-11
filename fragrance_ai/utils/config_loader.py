# fragrance_ai/utils/config_loader.py

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Configuration loader with safe defaults"""

    def __init__(self, config_path: str = None):
        """
        Initialize config loader

        Args:
            config_path: Path to config file. If None, uses default path.
        """
        if config_path is None:
            # Try to find config file in standard locations
            possible_paths = [
                Path("configs/rl_config.yaml"),
                Path(__file__).parent.parent.parent / "configs" / "rl_config.yaml",
                Path.home() / ".fragrance_ai" / "config.yaml"
            ]

            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                return self._get_default_config()
        else:
            print("Using default configuration (no config file found)")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "rl": {
                "algorithm": "PPO",
                "state_dim": 20,
                "action_dim": 12,
                "learning_rate": 0.0003,
                "ppo": {
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_epsilon": 0.2,
                    "value_coef": 0.5,
                    "entropy_coef": 0.01,
                    "update_epochs": 4,
                    "minibatch_size": 64,
                    "max_grad_norm": 0.5
                },
                "reinforce": {
                    "baseline": False
                },
                "training": {
                    "max_episodes": 1000,
                    "max_steps_per_episode": 100,
                    "save_frequency": 100,
                    "log_frequency": 10
                },
                "model": {
                    "save_path": "models/rl_model",
                    "load_path": "models/rl_model",
                    "backup_path": "models/backup"
                }
            },
            "ga": {
                "population_size": 100,
                "generations": 200,
                "elite_size": 10,
                "crossover": {
                    "probability": 0.9,
                    "type": "SBX",
                    "eta_c": 20
                },
                "mutation": {
                    "probability": 0.1,
                    "type": "polynomial",
                    "eta_m": 20,
                    "sigma": 0.2,
                    "min_concentration": 0.1
                },
                "constraints": {
                    "ifra_limit": True,
                    "concentration_sum": 100,
                    "min_ingredients": 3,
                    "max_ingredients": 15
                }
            },
            "creativity": {
                "entropy": {
                    "epsilon": 1e-12,
                    "min_probability": 0.001
                },
                "category_weights": {
                    "top_notes": 0.25,
                    "heart_notes": 0.40,
                    "base_notes": 0.35
                },
                "novelty": {
                    "enabled": True,
                    "comparison_pool_size": 50,
                    "distance_threshold": 0.3
                }
            },
            "safety": {
                "nan_handling": {
                    "check_frequency": 10,
                    "replace_value": 0.0
                },
                "gradient": {
                    "clip_norm": 1.0,
                    "check_explosion": True,
                    "max_norm": 10.0
                },
                "validation": {
                    "check_concentrations": True,
                    "check_ifra": True,
                    "check_positive": True
                }
            },
            "logging": {
                "level": "INFO",
                "file": "logs/fragrance_ai.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key_path: Dot-separated path to config value (e.g., "rl.ppo.gamma")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_rl_config(self) -> Dict[str, Any]:
        """Get RL-specific configuration"""
        return self.get("rl", {})

    def get_ga_config(self) -> Dict[str, Any]:
        """Get GA-specific configuration"""
        return self.get("ga", {})

    def get_creativity_config(self) -> Dict[str, Any]:
        """Get creativity-specific configuration"""
        return self.get("creativity", {})

    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety-specific configuration"""
        return self.get("safety", {})

    def save_config(self, path: str = None):
        """Save current configuration to file"""
        save_path = path or self.config_path or "configs/rl_config.yaml"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

        print(f"Configuration saved to {save_path}")


# Global config instance
_config = None


def get_config() -> ConfigLoader:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config(config_path: str = None):
    """Reload configuration from file"""
    global _config
    _config = ConfigLoader(config_path)
    return _config