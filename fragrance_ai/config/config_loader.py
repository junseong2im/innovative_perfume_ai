"""
Configuration Loader with Tuning History Tracking
설정 외부화 + 운용 중 파라미터 튜닝 기록
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import copy

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Change Record
# =============================================================================

@dataclass
class ConfigChange:
    """Configuration change record"""
    timestamp: str
    changed_by: str
    config_path: str  # e.g., "rlhf.ppo.entropy_coef.initial"
    old_value: Any
    new_value: Any
    reason: str
    environment: str = "production"  # production, staging, development

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ConfigChange":
        """Load from dictionary"""
        return ConfigChange(**data)


# =============================================================================
# Tuning History
# =============================================================================

class TuningHistory:
    """
    Tuning history tracker

    Records all parameter changes with timestamp, author, and reason.
    Useful for A/B testing analysis and rollback.
    """

    def __init__(self, history_file: str = "configs/tuning_history.json"):
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        self.changes: List[ConfigChange] = []
        self._load_history()

    def _load_history(self):
        """Load tuning history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.changes = [ConfigChange.from_dict(item) for item in data]

                logger.info(f"Loaded {len(self.changes)} config changes from {self.history_file}")
            except Exception as e:
                logger.error(f"Failed to load tuning history: {e}")
                self.changes = []

    def _save_history(self):
        """Save tuning history to file"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                data = [change.to_dict() for change in self.changes]
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.changes)} config changes to {self.history_file}")
        except Exception as e:
            logger.error(f"Failed to save tuning history: {e}")

    def record_change(
        self,
        config_path: str,
        old_value: Any,
        new_value: Any,
        changed_by: str,
        reason: str,
        environment: str = "production"
    ):
        """
        Record configuration change

        Args:
            config_path: Path to changed config (e.g., "rlhf.ppo.learning_rate")
            old_value: Old value
            new_value: New value
            changed_by: Who made the change
            reason: Why the change was made
            environment: Environment where change was made
        """
        change = ConfigChange(
            timestamp=datetime.now().isoformat(),
            changed_by=changed_by,
            config_path=config_path,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            environment=environment
        )

        self.changes.append(change)
        self._save_history()

        logger.info(f"[CONFIG CHANGE] {config_path}: {old_value} → {new_value} (by {changed_by}: {reason})")

    def get_changes_for_path(self, config_path: str) -> List[ConfigChange]:
        """Get all changes for a specific config path"""
        return [change for change in self.changes if change.config_path == config_path]

    def get_changes_by_user(self, changed_by: str) -> List[ConfigChange]:
        """Get all changes by a specific user"""
        return [change for change in self.changes if change.changed_by == changed_by]

    def get_recent_changes(self, n: int = 10) -> List[ConfigChange]:
        """Get N most recent changes"""
        return self.changes[-n:]

    def rollback_to_timestamp(self, timestamp: str) -> Dict[str, Any]:
        """
        Get config state at specific timestamp

        Args:
            timestamp: ISO format timestamp

        Returns:
            Dictionary of config paths and values at that time
        """
        target_time = datetime.fromisoformat(timestamp)
        config_state = {}

        for change in self.changes:
            change_time = datetime.fromisoformat(change.timestamp)

            if change_time <= target_time:
                config_state[change.config_path] = change.new_value

        return config_state


# =============================================================================
# Configuration Loader
# =============================================================================

class ConfigLoader:
    """
    Configuration loader with tuning history tracking

    Loads configuration from YAML files and tracks all changes.
    """

    def __init__(
        self,
        config_file: str = "configs/recommended_params.yaml",
        track_changes: bool = True,
        environment: str = "production"
    ):
        """
        Initialize configuration loader

        Args:
            config_file: Path to config YAML file
            track_changes: Whether to track config changes
            environment: Environment name (production, staging, development)
        """
        self.config_file = Path(config_file)
        self.track_changes = track_changes
        self.environment = environment

        # Load config
        self.config: Dict[str, Any] = self._load_config()

        # Original config (for tracking changes)
        self.original_config = copy.deepcopy(self.config)

        # Tuning history
        self.history = TuningHistory() if track_changes else None

        logger.info(f"ConfigLoader initialized: {config_file}, environment={environment}, track_changes={track_changes}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            logger.info(f"Loaded config from {self.config_file}")
            return config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path

        Args:
            path: Dot-separated path (e.g., "rlhf.ppo.learning_rate")
            default: Default value if path not found

        Returns:
            Configuration value
        """
        keys = path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(
        self,
        path: str,
        value: Any,
        changed_by: str = "system",
        reason: str = "manual_update"
    ):
        """
        Set configuration value

        Args:
            path: Dot-separated path (e.g., "rlhf.ppo.learning_rate")
            value: New value
            changed_by: Who made the change
            reason: Why the change was made
        """
        keys = path.split('.')
        config_dict = self.config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]

        # Get old value
        old_value = config_dict.get(keys[-1])

        # Set new value
        config_dict[keys[-1]] = value

        # Track change
        if self.track_changes and self.history:
            self.history.record_change(
                config_path=path,
                old_value=old_value,
                new_value=value,
                changed_by=changed_by,
                reason=reason,
                environment=self.environment
            )

    def save(self, output_file: Optional[str] = None):
        """
        Save current configuration to YAML file

        Args:
            output_file: Output file path (uses original if None)
        """
        output_path = Path(output_file) if output_file else self.config_file

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            logger.info(f"Saved config to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            raise

    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """
        Get mode-specific configuration

        Args:
            mode: Mode name (fast, balanced, creative)

        Returns:
            Merged configuration with mode overrides
        """
        base_config = copy.deepcopy(self.config)

        # Apply mode overrides
        if 'mode_overrides' in base_config and mode in base_config['mode_overrides']:
            overrides = base_config['mode_overrides'][mode]

            # Deep merge overrides
            base_config = self._deep_merge(base_config, overrides)

        return base_config

    def _deep_merge(self, base: Dict, overrides: Dict) -> Dict:
        """Deep merge override dictionary into base"""
        result = copy.deepcopy(base)

        for key, value in overrides.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
        self.original_config = copy.deepcopy(self.config)
        logger.info("Config reloaded")

    def reset(self):
        """Reset to original configuration"""
        self.config = copy.deepcopy(self.original_config)
        logger.info("Config reset to original")

    def export_json(self, output_file: str):
        """Export configuration as JSON"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported config to JSON: {output_file}")
        except Exception as e:
            logger.error(f"Failed to export config: {e}")
            raise


# =============================================================================
# Global Config Instance
# =============================================================================

_global_config: Optional[ConfigLoader] = None


def get_config(
    config_file: str = "configs/recommended_params.yaml",
    track_changes: bool = True,
    environment: str = "production"
) -> ConfigLoader:
    """
    Get global configuration instance

    Args:
        config_file: Path to config file
        track_changes: Whether to track changes
        environment: Environment name

    Returns:
        ConfigLoader instance
    """
    global _global_config

    if _global_config is None:
        _global_config = ConfigLoader(
            config_file=config_file,
            track_changes=track_changes,
            environment=environment
        )

    return _global_config


# =============================================================================
# CLI Tools
# =============================================================================

def print_config_tree(config: Dict, indent: int = 0):
    """Print configuration tree"""
    for key, value in config.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key}:")
            print_config_tree(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")


def print_tuning_history(history: TuningHistory, n: int = 10):
    """Print recent tuning history"""
    recent = history.get_recent_changes(n)

    print("\n" + "="*80)
    print(f"RECENT CONFIGURATION CHANGES (last {len(recent)})")
    print("="*80 + "\n")

    for change in recent:
        print(f"[{change.timestamp}] {change.config_path}")
        print(f"  Changed by: {change.changed_by}")
        print(f"  Environment: {change.environment}")
        print(f"  Old value: {change.old_value}")
        print(f"  New value: {change.new_value}")
        print(f"  Reason: {change.reason}")
        print()


# =============================================================================
# Export
# =============================================================================

__all__ = [
    'ConfigLoader',
    'TuningHistory',
    'ConfigChange',
    'get_config',
    'print_config_tree',
    'print_tuning_history'
]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Configuration Loader CLI")
    parser.add_argument("--config", default="configs/recommended_params.yaml", help="Config file path")
    parser.add_argument("--get", help="Get config value by path")
    parser.add_argument("--set", nargs=3, metavar=("PATH", "VALUE", "REASON"), help="Set config value")
    parser.add_argument("--changed-by", default="admin", help="Who made the change")
    parser.add_argument("--history", type=int, help="Show N recent changes")
    parser.add_argument("--mode", choices=["fast", "balanced", "creative"], help="Show mode-specific config")
    parser.add_argument("--tree", action="store_true", help="Print full config tree")

    args = parser.parse_args()

    # Initialize config loader
    loader = ConfigLoader(config_file=args.config, track_changes=True)

    if args.get:
        value = loader.get(args.get)
        print(f"{args.get} = {value}")

    elif args.set:
        path, value, reason = args.set

        # Try to parse value as JSON
        try:
            value = json.loads(value)
        except:
            pass  # Keep as string

        loader.set(
            path=path,
            value=value,
            changed_by=args.changed_by,
            reason=reason
        )
        loader.save()
        print(f"✅ Set {path} = {value}")

    elif args.history:
        print_tuning_history(loader.history, args.history)

    elif args.mode:
        mode_config = loader.get_mode_config(args.mode)
        print(f"\n{'='*80}")
        print(f"CONFIG FOR MODE: {args.mode.upper()}")
        print(f"{'='*80}\n")
        print_config_tree(mode_config)

    elif args.tree:
        print("\n" + "="*80)
        print("FULL CONFIGURATION TREE")
        print("="*80 + "\n")
        print_config_tree(loader.config)

    else:
        print("Use --get, --set, --history, --mode, or --tree")
        print("\nExamples:")
        print("  python -m fragrance_ai.config.config_loader --get rlhf.ppo.learning_rate")
        print("  python -m fragrance_ai.config.config_loader --set rlhf.ppo.learning_rate 0.0001 'Experiment XYZ'")
        print("  python -m fragrance_ai.config.config_loader --history 10")
        print("  python -m fragrance_ai.config.config_loader --mode creative")
        print("  python -m fragrance_ai.config.config_loader --tree")
