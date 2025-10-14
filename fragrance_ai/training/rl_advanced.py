"""
Advanced RLHF/Exploration Features
고급 강화학습 기능: Entropy Annealing, Reward Normalization, Checkpoint & Rollback
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Entropy Annealing Scheduler
# ============================================================================

@dataclass
class EntropyScheduleConfig:
    """Entropy annealing 설정"""
    initial_entropy: float = 0.01  # 초기 entropy 계수 (탐색)
    final_entropy: float = 0.001  # 최종 entropy 계수 (수렴)
    decay_steps: int = 100000  # 감소 스텝 수
    schedule_type: str = "linear"  # linear, cosine, exponential


class EntropyScheduler:
    """
    Entropy Coefficient Annealing Scheduler

    초기에는 높은 entropy로 탐색을 장려하고,
    학습이 진행되면서 점진적으로 감소시켜 수렴을 돕습니다.

    Features:
    - Linear decay (default)
    - Cosine annealing
    - Exponential decay
    """

    def __init__(self, config: EntropyScheduleConfig):
        self.config = config
        self.current_step = 0
        self.current_entropy = config.initial_entropy

        logger.info(
            f"Entropy scheduler initialized: "
            f"{config.initial_entropy} → {config.final_entropy} "
            f"over {config.decay_steps} steps ({config.schedule_type})"
        )

    def step(self) -> float:
        """
        스텝 진행 및 현재 entropy 계수 반환

        Returns:
            Current entropy coefficient
        """
        self.current_step += 1

        # 진행률 계산
        progress = min(self.current_step / self.config.decay_steps, 1.0)

        # 스케줄 타입에 따라 entropy 계산
        if self.config.schedule_type == "linear":
            self.current_entropy = self._linear_decay(progress)
        elif self.config.schedule_type == "cosine":
            self.current_entropy = self._cosine_decay(progress)
        elif self.config.schedule_type == "exponential":
            self.current_entropy = self._exponential_decay(progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

        return self.current_entropy

    def _linear_decay(self, progress: float) -> float:
        """선형 감소"""
        return self.config.initial_entropy - (
            self.config.initial_entropy - self.config.final_entropy
        ) * progress

    def _cosine_decay(self, progress: float) -> float:
        """Cosine annealing"""
        return self.config.final_entropy + (
            self.config.initial_entropy - self.config.final_entropy
        ) * 0.5 * (1 + np.cos(np.pi * progress))

    def _exponential_decay(self, progress: float) -> float:
        """지수 감소"""
        decay_rate = np.log(self.config.final_entropy / self.config.initial_entropy)
        return self.config.initial_entropy * np.exp(decay_rate * progress)

    def get_entropy(self) -> float:
        """현재 entropy 계수 반환"""
        return self.current_entropy

    def get_info(self) -> Dict[str, float]:
        """현재 상태 정보"""
        progress = min(self.current_step / self.config.decay_steps, 1.0)
        return {
            "current_step": self.current_step,
            "current_entropy": self.current_entropy,
            "progress": progress,
            "remaining_steps": max(0, self.config.decay_steps - self.current_step)
        }

    def reset(self):
        """스케줄러 초기화"""
        self.current_step = 0
        self.current_entropy = self.config.initial_entropy


# ============================================================================
# 2. Reward Normalization
# ============================================================================

@dataclass
class RewardNormalizerConfig:
    """Reward normalizer 설정"""
    window_size: int = 1000  # 이동 평균 윈도우 크기
    epsilon: float = 1e-8  # 분산 0 방지
    clip_range: Optional[Tuple[float, float]] = (-10.0, 10.0)  # 정규화 후 클리핑
    update_mean_std: bool = True  # 평균/표준편차 업데이트 여부


class RewardNormalizer:
    """
    Moving Average/Std Reward Normalization

    최근 N개의 보상에 대한 이동 평균과 표준편차를 유지하여
    보상 스케일을 안정화합니다.

    Normalized reward = (reward - mean) / (std + epsilon)

    Features:
    - Rolling window statistics
    - Optional clipping
    - Welford's online algorithm for numerical stability
    """

    def __init__(self, config: RewardNormalizerConfig):
        self.config = config

        # 보상 버퍼 (deque for efficient O(1) append/pop)
        self.reward_buffer: Deque[float] = deque(maxlen=config.window_size)

        # 통계 (Welford's online algorithm)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0  # Sum of squared differences from mean
        self.std = 1.0

        # 통계 추적
        self.stats_history = {
            'raw_rewards': deque(maxlen=1000),
            'normalized_rewards': deque(maxlen=1000),
            'means': deque(maxlen=1000),
            'stds': deque(maxlen=1000)
        }

        logger.info(
            f"Reward normalizer initialized: window_size={config.window_size}, "
            f"clip_range={config.clip_range}"
        )

    def normalize(self, reward: float) -> float:
        """
        보상 정규화

        Args:
            reward: 원본 보상

        Returns:
            Normalized reward
        """
        # 버퍼에 추가
        self.reward_buffer.append(reward)

        # 통계 업데이트 (Welford's algorithm)
        if self.config.update_mean_std:
            self._update_statistics(reward)

        # 정규화
        if self.count < 2:
            # 데이터가 부족하면 원본 반환
            normalized = reward
        else:
            normalized = (reward - self.mean) / (self.std + self.config.epsilon)

        # 클리핑
        if self.config.clip_range is not None:
            clip_min, clip_max = self.config.clip_range
            normalized = np.clip(normalized, clip_min, clip_max)

        # 통계 기록
        self.stats_history['raw_rewards'].append(reward)
        self.stats_history['normalized_rewards'].append(normalized)
        self.stats_history['means'].append(self.mean)
        self.stats_history['stds'].append(self.std)

        return normalized

    def normalize_batch(self, rewards: np.ndarray) -> np.ndarray:
        """
        배치 보상 정규화

        Args:
            rewards: Array of rewards

        Returns:
            Normalized rewards array
        """
        normalized = np.array([self.normalize(r) for r in rewards])
        return normalized

    def _update_statistics(self, reward: float):
        """
        Welford's online algorithm로 평균/분산 업데이트

        This is numerically stable and O(1) per update.
        """
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.m2 += delta * delta2

        # 표준편차 계산
        if self.count > 1:
            variance = self.m2 / (self.count - 1)
            self.std = np.sqrt(variance)
        else:
            self.std = 1.0

    def get_statistics(self) -> Dict[str, float]:
        """현재 통계 반환"""
        # 버퍼 기반 통계 (참고용)
        buffer_mean = np.mean(self.reward_buffer) if self.reward_buffer else 0.0
        buffer_std = np.std(self.reward_buffer) if len(self.reward_buffer) > 1 else 1.0

        return {
            'mean': self.mean,
            'std': self.std,
            'count': self.count,
            'buffer_size': len(self.reward_buffer),
            'buffer_mean': buffer_mean,
            'buffer_std': buffer_std
        }

    def reset(self):
        """정규화기 초기화"""
        self.reward_buffer.clear()
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.std = 1.0
        logger.info("Reward normalizer reset")


# ============================================================================
# 3. Checkpoint & Rollback Manager
# ============================================================================

@dataclass
class CheckpointConfig:
    """Checkpoint manager 설정"""
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 100  # 스텝마다 저장
    max_checkpoints: int = 5  # 유지할 최대 체크포인트 수

    # Rollback 조건
    rollback_on_kl_threshold: float = 0.1  # KL > threshold → rollback
    rollback_on_loss_spike_factor: float = 3.0  # loss > mean * factor → rollback
    rollback_on_reward_drop_factor: float = 0.5  # reward < mean * factor → rollback

    # 메트릭 윈도우
    metric_window_size: int = 10  # 최근 N개 메트릭 추적


@dataclass
class CheckpointMetrics:
    """체크포인트 시점의 메트릭"""
    step: int
    timestamp: str
    policy_loss: float
    value_loss: float
    kl_divergence: float
    mean_reward: float
    explained_variance: float

    # 추가 메트릭
    entropy: float = 0.0
    clip_fraction: float = 0.0


class CheckpointManager:
    """
    Checkpoint & Rollback Manager

    Features:
    - Periodic checkpoint saving (policy, value, optimizer)
    - Performance metric tracking
    - Automatic rollback detection
    - Best checkpoint tracking

    Rollback Conditions:
    1. KL divergence too high (policy changed too much)
    2. Loss spike (sudden increase)
    3. Reward drop (performance degradation)
    """

    def __init__(self, config: CheckpointConfig):
        self.config = config

        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 체크포인트 리스트 (step, filepath, metrics)
        self.checkpoints: List[Tuple[int, Path, CheckpointMetrics]] = []

        # 메트릭 히스토리
        self.metrics_history: Deque[CheckpointMetrics] = deque(
            maxlen=config.metric_window_size
        )

        # Best checkpoint tracking
        self.best_checkpoint_step = -1
        self.best_reward = -float('inf')

        # Rollback 카운터
        self.rollback_count = 0
        self.last_rollback_step = -1

        logger.info(
            f"Checkpoint manager initialized: dir={config.checkpoint_dir}, "
            f"interval={config.save_interval}, max={config.max_checkpoints}"
        )

    def should_save_checkpoint(self, step: int) -> bool:
        """체크포인트 저장 여부 판단"""
        return step > 0 and step % self.config.save_interval == 0

    def save_checkpoint(
        self,
        step: int,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: CheckpointMetrics,
        additional_state: Optional[Dict] = None
    ) -> Path:
        """
        체크포인트 저장

        Args:
            step: Current training step
            network: Policy/value network
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            metrics: Current metrics
            additional_state: Additional state dict

        Returns:
            Path to saved checkpoint
        """
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step{step}_{timestamp}.pth"
        filepath = self.checkpoint_dir / filename

        # 체크포인트 데이터
        checkpoint_data = {
            'step': step,
            'network_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': {
                'step': metrics.step,
                'timestamp': metrics.timestamp,
                'policy_loss': metrics.policy_loss,
                'value_loss': metrics.value_loss,
                'kl_divergence': metrics.kl_divergence,
                'mean_reward': metrics.mean_reward,
                'explained_variance': metrics.explained_variance,
                'entropy': metrics.entropy,
                'clip_fraction': metrics.clip_fraction
            }
        }

        # 추가 상태
        if additional_state:
            checkpoint_data['additional_state'] = additional_state

        # 저장
        torch.save(checkpoint_data, filepath)

        # 리스트에 추가
        self.checkpoints.append((step, filepath, metrics))
        self.metrics_history.append(metrics)

        # Best checkpoint 업데이트
        if metrics.mean_reward > self.best_reward:
            self.best_reward = metrics.mean_reward
            self.best_checkpoint_step = step

            # Best checkpoint 복사
            best_filepath = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint_data, best_filepath)
            logger.info(f"New best checkpoint: step={step}, reward={metrics.mean_reward:.3f}")

        logger.info(f"Checkpoint saved: {filepath}")

        # 오래된 체크포인트 삭제
        self._cleanup_old_checkpoints()

        return filepath

    def should_rollback(self, current_metrics: CheckpointMetrics) -> Tuple[bool, str]:
        """
        롤백 필요 여부 판단

        Args:
            current_metrics: Current training metrics

        Returns:
            (should_rollback, reason)
        """
        if len(self.metrics_history) < 3:
            # 충분한 히스토리가 없으면 롤백 안함
            return False, ""

        # 최근 메트릭 평균
        recent_kl = np.mean([m.kl_divergence for m in self.metrics_history])
        recent_loss = np.mean([m.policy_loss + m.value_loss for m in self.metrics_history])
        recent_reward = np.mean([m.mean_reward for m in self.metrics_history])

        # 조건 1: KL divergence too high
        if current_metrics.kl_divergence > self.config.rollback_on_kl_threshold:
            reason = (
                f"KL divergence too high: {current_metrics.kl_divergence:.4f} "
                f"> {self.config.rollback_on_kl_threshold}"
            )
            return True, reason

        # 조건 2: Loss spike
        current_loss = current_metrics.policy_loss + current_metrics.value_loss
        if current_loss > recent_loss * self.config.rollback_on_loss_spike_factor:
            reason = (
                f"Loss spike detected: {current_loss:.4f} "
                f"> {recent_loss:.4f} * {self.config.rollback_on_loss_spike_factor}"
            )
            return True, reason

        # 조건 3: Reward drop
        if current_metrics.mean_reward < recent_reward * self.config.rollback_on_reward_drop_factor:
            reason = (
                f"Reward drop detected: {current_metrics.mean_reward:.3f} "
                f"< {recent_reward:.3f} * {self.config.rollback_on_reward_drop_factor}"
            )
            return True, reason

        return False, ""

    def rollback(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        reason: str
    ) -> Optional[int]:
        """
        마지막 체크포인트로 롤백

        Args:
            network: Policy/value network
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            reason: Rollback reason

        Returns:
            Rolled back step, or None if no checkpoint available
        """
        if not self.checkpoints:
            logger.warning("No checkpoints available for rollback")
            return None

        # 가장 최근 체크포인트로 롤백
        step, filepath, metrics = self.checkpoints[-1]

        logger.warning(f"Rolling back to step {step}. Reason: {reason}")

        # 체크포인트 로드
        checkpoint_data = torch.load(filepath, map_location=network.parameters().__next__().device, weights_only=False)

        network.load_state_dict(checkpoint_data['network_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        if scheduler and checkpoint_data.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        # 롤백 통계 업데이트
        self.rollback_count += 1
        self.last_rollback_step = step

        logger.info(
            f"Rollback complete: restored to step {step}, "
            f"reward={metrics.mean_reward:.3f}, "
            f"total_rollbacks={self.rollback_count}"
        )

        return step

    def load_best_checkpoint(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    ) -> Optional[int]:
        """Best checkpoint 로드"""
        best_filepath = self.checkpoint_dir / "checkpoint_best.pth"

        if not best_filepath.exists():
            logger.warning("No best checkpoint found")
            return None

        checkpoint_data = torch.load(best_filepath, map_location=network.parameters().__next__().device, weights_only=False)

        network.load_state_dict(checkpoint_data['network_state_dict'])
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

        if scheduler and checkpoint_data.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

        step = checkpoint_data['step']
        logger.info(f"Loaded best checkpoint from step {step}")

        return step

    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 삭제 (최근 N개만 유지)"""
        if len(self.checkpoints) <= self.config.max_checkpoints:
            return

        # 오래된 체크포인트 삭제
        while len(self.checkpoints) > self.config.max_checkpoints:
            step, filepath, _ = self.checkpoints.pop(0)
            if filepath.exists():
                filepath.unlink()
                logger.debug(f"Deleted old checkpoint: {filepath}")

    def get_statistics(self) -> Dict:
        """통계 정보 반환"""
        if not self.metrics_history:
            return {}

        recent_rewards = [m.mean_reward for m in self.metrics_history]
        recent_kls = [m.kl_divergence for m in self.metrics_history]
        recent_losses = [m.policy_loss + m.value_loss for m in self.metrics_history]

        return {
            'num_checkpoints': len(self.checkpoints),
            'best_checkpoint_step': self.best_checkpoint_step,
            'best_reward': self.best_reward,
            'rollback_count': self.rollback_count,
            'last_rollback_step': self.last_rollback_step,
            'recent_mean_reward': np.mean(recent_rewards),
            'recent_std_reward': np.std(recent_rewards),
            'recent_mean_kl': np.mean(recent_kls),
            'recent_mean_loss': np.mean(recent_losses)
        }


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'EntropyScheduleConfig',
    'EntropyScheduler',
    'RewardNormalizerConfig',
    'RewardNormalizer',
    'CheckpointConfig',
    'CheckpointMetrics',
    'CheckpointManager'
]
