"""
Parameter Loader - 권장 초기값 로드

configs/recommended_params.yaml에서 MOGA/RLHF 파라미터를 로드하고 적용합니다.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MOGA Parameter Classes
# =============================================================================

@dataclass
class MOGANoveltyWeightConfig:
    """MOGA Novelty Weight 설정"""
    base: float = 0.1
    per_hint_increment: float = 0.05
    max: float = 0.5

    def compute_novelty_weight(self, creative_hints: int) -> float:
        """
        Novelty weight 계산

        Formula: novelty_weight = base + per_hint_increment * len(creative_hints)

        Args:
            creative_hints: 창의적 힌트 개수

        Returns:
            novelty_weight (0.0 ~ max)
        """
        weight = self.base + self.per_hint_increment * creative_hints
        return min(weight, self.max)


@dataclass
class MOGAMutationSigmaConfig:
    """MOGA Mutation Sigma 설정"""
    base: float = 0.12
    creative_bonus: float = 0.02
    min: float = 0.05
    max: float = 0.20

    def compute_mutation_sigma(self, mode: str) -> float:
        """
        Mutation sigma 계산

        Formula:
            - Balanced/Fast mode: base
            - Creative mode: base + creative_bonus

        Args:
            mode: fast, balanced, creative

        Returns:
            mutation_sigma
        """
        if mode.lower() == "creative":
            sigma = self.base + self.creative_bonus
        else:
            sigma = self.base

        return max(self.min, min(sigma, self.max))


@dataclass
class MOGAConfig:
    """MOGA 전체 설정"""
    population_size: int = 50
    n_generations: int = 20
    novelty_weight: MOGANoveltyWeightConfig = None
    mutation_sigma: MOGAMutationSigmaConfig = None
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    pareto_size: int = 10

    def __post_init__(self):
        if self.novelty_weight is None:
            self.novelty_weight = MOGANoveltyWeightConfig()
        if self.mutation_sigma is None:
            self.mutation_sigma = MOGAMutationSigmaConfig()


# =============================================================================
# PPO Parameter Classes
# =============================================================================

@dataclass
class PPOEntropyCoefConfig:
    """PPO Entropy Coefficient 설정"""
    initial: float = 0.01
    final: float = 0.001
    decay_schedule: str = "cosine"
    decay_steps: int = 10000

    def compute_entropy_coef(self, current_step: int) -> float:
        """
        Entropy coefficient 계산 (cosine decay)

        Formula: entropy_coef(t) = final + 0.5 * (initial - final) * (1 + cos(π * t / decay_steps))

        Args:
            current_step: 현재 스텝

        Returns:
            entropy_coef
        """
        import math

        if current_step >= self.decay_steps:
            return self.final

        if self.decay_schedule == "cosine":
            progress = current_step / self.decay_steps
            coef = self.final + 0.5 * (self.initial - self.final) * (1 + math.cos(math.pi * progress))
        elif self.decay_schedule == "linear":
            progress = current_step / self.decay_steps
            coef = self.initial - (self.initial - self.final) * progress
        else:
            coef = self.initial

        return coef


@dataclass
class PPOConfig:
    """PPO 전체 설정"""
    clip_eps: float = 0.2
    entropy_coef: PPOEntropyCoefConfig = None
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3.0e-4
    batch_size: int = 64
    n_epochs: int = 10
    gae_lambda: float = 0.95
    gamma: float = 0.99
    n_steps: int = 2048

    def __post_init__(self):
        if self.entropy_coef is None:
            self.entropy_coef = PPOEntropyCoefConfig()


# =============================================================================
# Reward Normalization Config
# =============================================================================

@dataclass
class RewardNormalizationConfig:
    """Reward Normalization 설정"""
    enabled: bool = True
    window_size: int = 1000  # Last 1k steps
    update_mean_std: bool = True
    epsilon: float = 1.0e-8
    clip_range: float = 10.0


# =============================================================================
# Parameter Loader
# =============================================================================

class ParameterLoader:
    """
    권장 파라미터 로더

    configs/recommended_params.yaml에서 설정을 로드합니다.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: YAML 설정 파일 경로 (optional)
        """
        if config_path is None:
            # Default path
            config_path = Path(__file__).parent.parent.parent / "configs" / "recommended_params.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded parameters from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}. Using defaults.")
            return {}

    def get_moga_config(self, mode: str = "balanced") -> MOGAConfig:
        """
        MOGA 설정 조회

        Args:
            mode: fast, balanced, creative

        Returns:
            MOGAConfig
        """
        # Get base config
        base_config = self.config.get("moga", {})

        # Get mode-specific overrides
        mode_overrides = self.config.get("mode_overrides", {}).get(mode, {}).get("moga", {})

        # Merge configs (mode overrides take precedence)
        merged_config = {**base_config, **mode_overrides}

        # Build MOGAConfig
        novelty_weight_config = MOGANoveltyWeightConfig(
            base=merged_config.get("novelty_weight", {}).get("base", 0.1),
            per_hint_increment=merged_config.get("novelty_weight", {}).get("per_hint_increment", 0.05),
            max=merged_config.get("novelty_weight", {}).get("max", 0.5)
        )

        mutation_sigma_config = MOGAMutationSigmaConfig(
            base=merged_config.get("mutation_sigma", {}).get("base", 0.12),
            creative_bonus=merged_config.get("mutation_sigma", {}).get("creative_bonus", 0.02),
            min=merged_config.get("mutation_sigma", {}).get("min", 0.05),
            max=merged_config.get("mutation_sigma", {}).get("max", 0.20)
        )

        return MOGAConfig(
            population_size=merged_config.get("population_size", 50),
            n_generations=merged_config.get("n_generations", 20),
            novelty_weight=novelty_weight_config,
            mutation_sigma=mutation_sigma_config,
            crossover_rate=merged_config.get("crossover_rate", 0.8),
            elite_fraction=merged_config.get("elite_fraction", 0.1),
            pareto_size=merged_config.get("pareto_size", 10)
        )

    def get_ppo_config(self, mode: str = "balanced") -> PPOConfig:
        """
        PPO 설정 조회

        Args:
            mode: fast, balanced, creative

        Returns:
            PPOConfig
        """
        # Get base config
        base_config = self.config.get("rlhf", {}).get("ppo", {})

        # Get mode-specific overrides
        mode_overrides = self.config.get("mode_overrides", {}).get(mode, {}).get("rlhf", {}).get("ppo", {})

        # Merge configs
        merged_config = {**base_config, **mode_overrides}

        # Build PPOConfig
        entropy_coef_config = PPOEntropyCoefConfig(
            initial=merged_config.get("entropy_coef", {}).get("initial", 0.01),
            final=merged_config.get("entropy_coef", {}).get("final", 0.001),
            decay_schedule=merged_config.get("entropy_coef", {}).get("decay_schedule", "cosine"),
            decay_steps=merged_config.get("entropy_coef", {}).get("decay_steps", 10000)
        )

        return PPOConfig(
            clip_eps=merged_config.get("clip_eps", 0.2),
            entropy_coef=entropy_coef_config,
            value_coef=merged_config.get("value_coef", 0.5),
            max_grad_norm=merged_config.get("max_grad_norm", 0.5),
            learning_rate=merged_config.get("learning_rate", 3.0e-4),
            batch_size=merged_config.get("batch_size", 64),
            n_epochs=merged_config.get("n_epochs", 10),
            gae_lambda=merged_config.get("gae_lambda", 0.95),
            gamma=merged_config.get("gamma", 0.99),
            n_steps=merged_config.get("n_steps", 2048)
        )

    def get_reward_normalization_config(self) -> RewardNormalizationConfig:
        """
        Reward Normalization 설정 조회

        Returns:
            RewardNormalizationConfig
        """
        config = self.config.get("reward_normalization", {})

        return RewardNormalizationConfig(
            enabled=config.get("enabled", True),
            window_size=config.get("window_size", 1000),
            update_mean_std=config.get("update_mean_std", True),
            epsilon=config.get("epsilon", 1.0e-8),
            clip_range=config.get("clip_range", 10.0)
        )


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create loader
    loader = ParameterLoader()

    # Test MOGA configs
    print("=== MOGA Configs ===")
    for mode in ["fast", "balanced", "creative"]:
        print(f"\n{mode.upper()} Mode:")
        moga_config = loader.get_moga_config(mode)

        print(f"  Population: {moga_config.population_size}")
        print(f"  Generations: {moga_config.n_generations}")

        # Test novelty weight with different hint counts
        for n_hints in [0, 2, 5]:
            novelty_weight = moga_config.novelty_weight.compute_novelty_weight(n_hints)
            print(f"  Novelty weight ({n_hints} hints): {novelty_weight:.3f}")

        # Test mutation sigma
        mutation_sigma = moga_config.mutation_sigma.compute_mutation_sigma(mode)
        print(f"  Mutation sigma: {mutation_sigma:.3f}")

    # Test PPO configs
    print("\n=== PPO Configs ===")
    for mode in ["fast", "balanced", "creative"]:
        print(f"\n{mode.upper()} Mode:")
        ppo_config = loader.get_ppo_config(mode)

        print(f"  Clip epsilon: {ppo_config.clip_eps}")
        print(f"  Initial entropy: {ppo_config.entropy_coef.initial}")
        print(f"  Final entropy: {ppo_config.entropy_coef.final}")
        print(f"  Value coef: {ppo_config.value_coef}")
        print(f"  Max grad norm: {ppo_config.max_grad_norm}")
        print(f"  Batch size: {ppo_config.batch_size}")

        # Test entropy coef at different steps
        for step in [0, 5000, 10000]:
            entropy_coef = ppo_config.entropy_coef.compute_entropy_coef(step)
            print(f"  Entropy coef at step {step}: {entropy_coef:.4f}")

    # Test reward normalization
    print("\n=== Reward Normalization ===")
    reward_norm_config = loader.get_reward_normalization_config()
    print(f"  Enabled: {reward_norm_config.enabled}")
    print(f"  Window size: {reward_norm_config.window_size} steps")
    print(f"  Update mean/std: {reward_norm_config.update_mean_std}")
    print(f"  Clip range: ±{reward_norm_config.clip_range}")
