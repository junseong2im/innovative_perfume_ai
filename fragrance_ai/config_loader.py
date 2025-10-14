"""
Configuration Loader
Load environment-specific configuration from .env files and map to YAML configs
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Load and merge configuration from environment variables and YAML files

    Priority (highest to lowest):
    1. Environment variables
    2. Profile-specific YAML config
    3. Default YAML config
    """

    def __init__(self, env: Optional[str] = None, config_dir: Optional[Path] = None):
        """
        Initialize config loader

        Args:
            env: Environment name (development/staging/production)
                 If None, reads from APP_ENV environment variable
            config_dir: Path to config directory
                       If None, uses ./configs
        """
        self.env = env or os.getenv('APP_ENV', 'development')
        self.config_dir = config_dir or Path(__file__).parent.parent / 'configs'

        # Load .env file for the environment
        self._load_env_file()

        logger.info(f"ConfigLoader initialized: env={self.env}, config_dir={self.config_dir}")

    def _load_env_file(self):
        """Load .env file based on environment"""
        project_root = Path(__file__).parent.parent

        # Try environment-specific .env file first
        env_file = project_root / f".env.{self.env}"
        if env_file.exists():
            logger.info(f"Loading environment file: {env_file}")
            load_dotenv(env_file, override=True)
        else:
            # Fall back to generic .env
            env_file = project_root / ".env"
            if env_file.exists():
                logger.info(f"Loading environment file: {env_file}")
                load_dotenv(env_file, override=True)
            else:
                logger.warning(f"No .env file found for environment: {self.env}")

    def load_llm_config(self) -> Dict[str, Any]:
        """
        Load LLM ensemble configuration

        Returns:
            Merged configuration dict
        """
        yaml_file = self.config_dir / "llm_ensemble.yaml"

        if not yaml_file.exists():
            logger.error(f"LLM config file not found: {yaml_file}")
            return {}

        # Load YAML
        with open(yaml_file, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)

        # Get profile-specific config
        profile_config = full_config.get(self.env, {})

        # Get default config (everything except profiles)
        default_config = {
            k: v for k, v in full_config.items()
            if k not in ['development', 'staging', 'production']
        }

        # Merge: default + profile + env overrides
        merged_config = self._deep_merge(default_config, profile_config)
        merged_config = self._apply_env_overrides(merged_config)

        logger.info(f"Loaded LLM config for profile: {self.env}")
        return merged_config

    def load_rl_config(self) -> Dict[str, Any]:
        """
        Load RL training configuration

        Returns:
            Merged configuration dict
        """
        yaml_file = self.config_dir / "rl_config.yaml"

        if not yaml_file.exists():
            logger.warning(f"RL config file not found: {yaml_file}")
            return self._get_default_rl_config()

        # Load YAML
        with open(yaml_file, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)

        # Get profile-specific config if available
        profile_config = full_config.get(self.env, full_config)

        # Apply environment overrides
        merged_config = self._apply_env_overrides(profile_config)

        logger.info(f"Loaded RL config for profile: {self.env}")
        return merged_config

    def get_database_url(self) -> str:
        """Get database URL from environment"""
        return os.getenv('DATABASE_URL', 'sqlite:///./data/fragrance.db')

    def get_redis_url(self) -> str:
        """Get Redis URL from environment"""
        return os.getenv('REDIS_URL', 'redis://localhost:6379/0')

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration from environment"""
        return {
            'host': os.getenv('API_HOST', '0.0.0.0'),
            'port': int(os.getenv('API_PORT', '8000')),
            'workers': int(os.getenv('API_WORKERS', '4')),
            'reload': os.getenv('API_RELOAD', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO').upper(),
            'debug_mode': os.getenv('DEBUG_MODE', 'false').lower() == 'true',
        }

    def get_worker_config(self, worker_type: str) -> Dict[str, Any]:
        """
        Get worker configuration

        Args:
            worker_type: 'llm' or 'rl'
        """
        if worker_type == 'llm':
            return {
                'concurrency': int(os.getenv('LLM_WORKER_CONCURRENCY', '2')),
                'use_gpu': os.getenv('USE_GPU', 'false').lower() == 'true',
                'cache_dir': os.getenv('HF_HOME', './cache'),
            }
        elif worker_type == 'rl':
            return {
                'concurrency': int(os.getenv('RL_WORKER_CONCURRENCY', '1')),
                'use_gpu': os.getenv('USE_GPU', 'false').lower() == 'true',
                'checkpoint_dir': os.getenv('CHECKPOINT_DIR', './checkpoints'),
                'tensorboard_dir': os.getenv('TENSORBOARD_DIR', './tensorboard'),
            }
        else:
            raise ValueError(f"Unknown worker type: {worker_type}")

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return {
            'prometheus_enabled': os.getenv('PROMETHEUS_ENABLED', 'false').lower() == 'true',
            'prometheus_port': int(os.getenv('PROMETHEUS_PORT', '9090')),
            'grafana_enabled': os.getenv('GRAFANA_ENABLED', 'false').lower() == 'true',
            'grafana_port': int(os.getenv('GRAFANA_PORT', '3000')),
        }

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to config

        Environment variables are mapped as follows:
        - LLM_CONFIG_PROFILE -> profile selection
        - USE_GPU -> performance.use_gpu
        - CACHE_TTL -> cache.ttl_seconds
        - etc.

        Args:
            config: Base configuration

        Returns:
            Configuration with environment overrides
        """
        # GPU setting
        if 'USE_GPU' in os.environ:
            use_gpu = os.getenv('USE_GPU', 'false').lower() == 'true'
            if 'performance' in config:
                config['performance']['use_gpu'] = use_gpu
            if 'use_gpu' in config:
                config['use_gpu'] = use_gpu

        # Cache TTL
        if 'CACHE_TTL' in os.environ:
            cache_ttl = int(os.getenv('CACHE_TTL', '3600'))
            if 'cache' in config:
                config['cache']['ttl_seconds'] = cache_ttl

        # Log level
        if 'LOG_LEVEL' in os.environ:
            log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
            if 'logging' in config:
                config['logging']['log_level'] = log_level

        # Debug mode
        if 'DEBUG_MODE' in os.environ:
            debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
            config['debug_mode'] = debug_mode

        return config

    def _get_default_rl_config(self) -> Dict[str, Any]:
        """Get default RL configuration"""
        return {
            'checkpoint_dir': os.getenv('CHECKPOINT_DIR', './checkpoints'),
            'tensorboard_dir': os.getenv('TENSORBOARD_DIR', './tensorboard'),
            'use_gpu': os.getenv('USE_GPU', 'false').lower() == 'true',
            'n_iterations': 100,
            'n_steps_per_iteration': 2048,
            'n_ppo_epochs': 10,
            'batch_size': 64,
            'learning_rate': 3e-4,
        }


# Singleton instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(env: Optional[str] = None, reload: bool = False) -> ConfigLoader:
    """
    Get singleton ConfigLoader instance

    Args:
        env: Environment name (optional, defaults to APP_ENV)
        reload: Force reload of configuration

    Returns:
        ConfigLoader instance
    """
    global _config_loader

    if _config_loader is None or reload:
        _config_loader = ConfigLoader(env=env)

    return _config_loader


# Convenience functions
def load_llm_config(env: Optional[str] = None) -> Dict[str, Any]:
    """Load LLM configuration"""
    return get_config_loader(env).load_llm_config()


def load_rl_config(env: Optional[str] = None) -> Dict[str, Any]:
    """Load RL configuration"""
    return get_config_loader(env).load_rl_config()


def get_database_url(env: Optional[str] = None) -> str:
    """Get database URL"""
    return get_config_loader(env).get_database_url()


def get_redis_url(env: Optional[str] = None) -> str:
    """Get Redis URL"""
    return get_config_loader(env).get_redis_url()


def get_api_config(env: Optional[str] = None) -> Dict[str, Any]:
    """Get API configuration"""
    return get_config_loader(env).get_api_config()


def get_worker_config(worker_type: str, env: Optional[str] = None) -> Dict[str, Any]:
    """Get worker configuration"""
    return get_config_loader(env).get_worker_config(worker_type)


# Example usage
if __name__ == "__main__":
    import json

    # Test different environments
    for env in ['development', 'staging', 'production']:
        print(f"\n{'='*60}")
        print(f"Environment: {env}")
        print(f"{'='*60}")

        loader = ConfigLoader(env=env)

        # LLM config
        llm_config = loader.load_llm_config()
        print(f"\nLLM Config (profile: {env}):")
        print(f"  use_gpu: {llm_config.get('use_gpu')}")
        print(f"  max_parallel_requests: {llm_config.get('max_parallel_requests')}")
        print(f"  qwen.max_new_tokens: {llm_config.get('qwen', {}).get('max_new_tokens')}")
        print(f"  cache.ttl_seconds: {llm_config.get('cache', {}).get('ttl_seconds')}")

        # API config
        api_config = loader.get_api_config()
        print(f"\nAPI Config:")
        print(f"  port: {api_config['port']}")
        print(f"  workers: {api_config['workers']}")
        print(f"  debug_mode: {api_config['debug_mode']}")

        # Database & Redis
        print(f"\nConnections:")
        print(f"  Database: {loader.get_database_url()}")
        print(f"  Redis: {loader.get_redis_url()}")
