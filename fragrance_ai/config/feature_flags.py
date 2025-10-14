"""
Feature Flags System
코드 배포와 기능 활성화 분리
"""

import os
import json
import logging
from typing import Dict, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class Environment(Enum):
    """배포 환경"""
    DEV = "dev"
    STG = "stg"
    PROD = "prod"


@dataclass
class FeatureFlag:
    """피처 플래그"""
    name: str
    enabled: bool
    description: str
    rollout_percentage: int = 100  # 0-100%
    environments: list = None  # None = all environments
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        if self.environments is None:
            self.environments = ["dev", "stg", "prod"]
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.updated_at:
            self.updated_at = self.created_at


class FeatureFlagManager:
    """피처 플래그 관리자"""

    def __init__(self, config_path: Optional[str] = None, environment: str = "dev"):
        self.environment = environment
        self.config_path = config_path or self._get_default_config_path()
        self.flags: Dict[str, FeatureFlag] = {}
        self._load_flags()

        logger.info(f"FeatureFlagManager initialized for environment: {self.environment}")

    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로"""
        return f"configs/feature_flags_{self.environment}.json"

    def _load_flags(self):
        """플래그 로드"""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for flag_data in data.get("flags", []):
                    flag = FeatureFlag(**flag_data)
                    self.flags[flag.name] = flag
            logger.info(f"Loaded {len(self.flags)} feature flags from {self.config_path}")
        else:
            logger.warning(f"Feature flags config not found: {self.config_path}")
            self._create_default_flags()

    def _create_default_flags(self):
        """기본 플래그 생성"""
        default_flags = [
            FeatureFlag(
                name="rl_pipeline_enabled",
                enabled=True,
                description="Enable RL (PPO/REINFORCE) pipeline",
                rollout_percentage=100,
                environments=["dev", "stg", "prod"]
            ),
            FeatureFlag(
                name="llm_ensemble_enabled",
                enabled=True,
                description="Enable LLM 3-model ensemble",
                rollout_percentage=100,
                environments=["dev", "stg", "prod"]
            ),
            FeatureFlag(
                name="ppo_algorithm",
                enabled=True,
                description="Use PPO algorithm (False = REINFORCE)",
                rollout_percentage=100,
                environments=["dev", "stg", "prod"]
            ),
            FeatureFlag(
                name="cache_enabled",
                enabled=True,
                description="Enable LLM response caching",
                rollout_percentage=100,
                environments=["dev", "stg", "prod"]
            ),
            FeatureFlag(
                name="circuit_breaker_enabled",
                enabled=True,
                description="Enable circuit breaker for LLM",
                rollout_percentage=100,
                environments=["dev", "stg", "prod"]
            ),
            FeatureFlag(
                name="new_moga_optimizer",
                enabled=False,
                description="Use new MOGA optimizer (experimental)",
                rollout_percentage=0,
                environments=["dev"]
            ),
            FeatureFlag(
                name="advanced_rlhf",
                enabled=False,
                description="Advanced RLHF with reward model",
                rollout_percentage=5,
                environments=["dev", "stg"]
            )
        ]

        for flag in default_flags:
            self.flags[flag.name] = flag

        self.save_flags()
        logger.info(f"Created {len(default_flags)} default feature flags")

    def save_flags(self):
        """플래그 저장"""
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "environment": self.environment,
            "flags": [asdict(flag) for flag in self.flags.values()]
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.flags)} feature flags to {self.config_path}")

    def is_enabled(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """플래그 활성화 여부 확인"""
        if flag_name not in self.flags:
            logger.warning(f"Unknown feature flag: {flag_name}, defaulting to False")
            return False

        flag = self.flags[flag_name]

        # 환경 체크
        if self.environment not in flag.environments:
            return False

        # 기본 활성화 여부
        if not flag.enabled:
            return False

        # Rollout percentage 체크 (user_id 기반 해싱)
        if flag.rollout_percentage < 100:
            if user_id:
                # Consistent hashing for gradual rollout
                hash_value = hash(f"{flag_name}_{user_id}") % 100
                return hash_value < flag.rollout_percentage
            else:
                # No user_id, use random
                import random
                return random.randint(0, 99) < flag.rollout_percentage

        return True

    def enable_flag(self, flag_name: str, rollout_percentage: int = 100):
        """플래그 활성화"""
        if flag_name not in self.flags:
            logger.error(f"Unknown feature flag: {flag_name}")
            return

        self.flags[flag_name].enabled = True
        self.flags[flag_name].rollout_percentage = rollout_percentage
        self.flags[flag_name].updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_flags()

        logger.info(f"Enabled flag: {flag_name} ({rollout_percentage}%)")

    def disable_flag(self, flag_name: str):
        """플래그 비활성화"""
        if flag_name not in self.flags:
            logger.error(f"Unknown feature flag: {flag_name}")
            return

        self.flags[flag_name].enabled = False
        self.flags[flag_name].updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_flags()

        logger.info(f"Disabled flag: {flag_name}")

    def set_rollout_percentage(self, flag_name: str, percentage: int):
        """Rollout 비율 설정 (카나리 배포)"""
        if flag_name not in self.flags:
            logger.error(f"Unknown feature flag: {flag_name}")
            return

        if not 0 <= percentage <= 100:
            logger.error(f"Invalid rollout percentage: {percentage}")
            return

        self.flags[flag_name].rollout_percentage = percentage
        self.flags[flag_name].updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save_flags()

        logger.info(f"Set rollout percentage for {flag_name}: {percentage}%")

    def get_flag_status(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """플래그 상태 조회"""
        if flag_name not in self.flags:
            return None

        flag = self.flags[flag_name]
        return {
            "name": flag.name,
            "enabled": flag.enabled,
            "rollout_percentage": flag.rollout_percentage,
            "environments": flag.environments,
            "description": flag.description,
            "updated_at": flag.updated_at
        }

    def list_all_flags(self) -> Dict[str, Dict[str, Any]]:
        """모든 플래그 목록"""
        return {
            name: self.get_flag_status(name)
            for name in self.flags.keys()
        }


# =============================================================================
# Global Feature Flag Manager
# =============================================================================

# 환경 변수에서 현재 환경 가져오기
CURRENT_ENVIRONMENT = os.getenv("ARTISAN_ENV", "dev")

# Global instance
_feature_flag_manager: Optional[FeatureFlagManager] = None


def get_feature_flag_manager() -> FeatureFlagManager:
    """글로벌 피처 플래그 매니저 반환"""
    global _feature_flag_manager
    if _feature_flag_manager is None:
        _feature_flag_manager = FeatureFlagManager(environment=CURRENT_ENVIRONMENT)
    return _feature_flag_manager


def is_enabled(flag_name: str, user_id: Optional[str] = None) -> bool:
    """피처 플래그 활성화 여부 (편의 함수)"""
    return get_feature_flag_manager().is_enabled(flag_name, user_id)


# =============================================================================
# Decorator for Feature Flags
# =============================================================================

def feature_flag(flag_name: str, fallback: Optional[Callable] = None):
    """
    피처 플래그 데코레이터

    Usage:
        @feature_flag("new_feature", fallback=old_function)
        def new_function():
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # user_id 추출 (kwargs에서)
            user_id = kwargs.get("user_id")

            if is_enabled(flag_name, user_id):
                return func(*args, **kwargs)
            elif fallback:
                return fallback(*args, **kwargs)
            else:
                logger.warning(f"Feature {flag_name} is disabled and no fallback provided")
                return None

        return wrapper
    return decorator


# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("=" * 60)
    print("Feature Flags Demo")
    print("=" * 60)
    print()

    # 1. 매니저 초기화
    manager = FeatureFlagManager(environment="dev")

    # 2. 플래그 확인
    print("1. Check flags:")
    print(f"  RL Pipeline: {manager.is_enabled('rl_pipeline_enabled')}")
    print(f"  LLM Ensemble: {manager.is_enabled('llm_ensemble_enabled')}")
    print(f"  PPO Algorithm: {manager.is_enabled('ppo_algorithm')}")
    print()

    # 3. 카나리 배포 시뮬레이션
    print("2. Canary deployment simulation:")
    manager.set_rollout_percentage("advanced_rlhf", 5)
    print(f"  Advanced RLHF rollout: 5%")

    enabled_count = sum(
        manager.is_enabled("advanced_rlhf", user_id=f"user_{i}")
        for i in range(100)
    )
    print(f"  Enabled for {enabled_count}/100 users (~5%)")
    print()

    # 4. 플래그 토글
    print("3. Toggle flags:")
    manager.disable_flag("new_moga_optimizer")
    print(f"  Disabled new_moga_optimizer")
    manager.enable_flag("new_moga_optimizer", rollout_percentage=25)
    print(f"  Enabled new_moga_optimizer (25% rollout)")
    print()

    # 5. 전체 플래그 목록
    print("4. All flags:")
    for name, status in manager.list_all_flags().items():
        enabled_str = "✓" if status["enabled"] else "✗"
        print(f"  {enabled_str} {name}: {status['rollout_percentage']}%")
    print()

    # 6. Decorator 예시
    print("5. Decorator example:")

    @feature_flag("ppo_algorithm")
    def use_ppo():
        return "Using PPO"

    @feature_flag("ppo_algorithm")
    def use_reinforce():
        return "Using REINFORCE"

    # PPO enabled
    result = use_ppo() if manager.is_enabled("ppo_algorithm") else use_reinforce()
    print(f"  Algorithm: {result}")
    print()

    print("Demo complete!")
