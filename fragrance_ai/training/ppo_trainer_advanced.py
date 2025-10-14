"""
Advanced PPO Trainer with:
1. Entropy Annealing
2. Reward Normalization
3. Checkpoint & Rollback

기존 ppo_engine.py를 확장하여 고급 기능 추가
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

# 기존 PPO 컴포넌트 import
from fragrance_ai.training.ppo_engine import (
    ActorCriticNetwork,
    RolloutBuffer,
    FragranceEnvironment
)

# 고급 기능 import
from fragrance_ai.training.rl_advanced import (
    EntropyScheduler,
    EntropyScheduleConfig,
    RewardNormalizer,
    RewardNormalizerConfig,
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetrics
)

logger = logging.getLogger(__name__)


class AdvancedPPOTrainer:
    """
    Advanced PPO Trainer with Enhanced Features

    Features:
    1. **Entropy Annealing**: Linearly decay entropy coefficient from exploration to convergence
    2. **Reward Normalization**: Stabilize reward scale using moving average/std (1k steps)
    3. **Checkpoint & Rollback**: Periodic saving + automatic rollback on bad training signals

    Usage:
        trainer = AdvancedPPOTrainer(
            state_dim=40,
            action_dim=60,
            entropy_config=EntropyScheduleConfig(
                initial_entropy=0.01,
                final_entropy=0.001,
                decay_steps=100000
            ),
            reward_config=RewardNormalizerConfig(window_size=1000),
            checkpoint_config=CheckpointConfig(save_interval=100)
        )

        for iteration in range(n_iterations):
            avg_reward = trainer.collect_rollout(env, n_steps=2048)
            train_stats = trainer.train_step(n_epochs=10)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        # Advanced features
        entropy_config: Optional[EntropyScheduleConfig] = None,
        reward_config: Optional[RewardNormalizerConfig] = None,
        checkpoint_config: Optional[CheckpointConfig] = None
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # 네트워크 초기화
        self.network = ActorCriticNetwork(state_dim, action_dim).to(device)

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

        # 경험 버퍼
        self.buffer = RolloutBuffer(2048, state_dim, device)

        # ====================================================================
        # Feature 1: Entropy Annealing
        # ====================================================================
        if entropy_config is None:
            entropy_config = EntropyScheduleConfig()

        self.entropy_scheduler = EntropyScheduler(entropy_config)
        self.initial_entropy = entropy_config.initial_entropy

        logger.info(
            f"[Entropy Annealing] Enabled: "
            f"{entropy_config.initial_entropy} → {entropy_config.final_entropy} "
            f"over {entropy_config.decay_steps} steps"
        )

        # ====================================================================
        # Feature 2: Reward Normalization
        # ====================================================================
        if reward_config is None:
            reward_config = RewardNormalizerConfig()

        self.reward_normalizer = RewardNormalizer(reward_config)

        logger.info(
            f"[Reward Normalization] Enabled: "
            f"window_size={reward_config.window_size}, "
            f"clip_range={reward_config.clip_range}"
        )

        # ====================================================================
        # Feature 3: Checkpoint & Rollback
        # ====================================================================
        if checkpoint_config is None:
            checkpoint_config = CheckpointConfig()

        self.checkpoint_manager = CheckpointManager(checkpoint_config)

        logger.info(
            f"[Checkpoint & Rollback] Enabled: "
            f"interval={checkpoint_config.save_interval}, "
            f"kl_threshold={checkpoint_config.rollback_on_kl_threshold}"
        )

        # 학습 통계
        self.training_step = 0
        self.episode_count = 0
        self.total_rollouts = 0
        self.reward_history = []  # Track rewards for statistics

    def collect_rollout(self, env, n_steps: int = 2048) -> float:
        """
        환경과 상호작용하여 경험 수집

        **Reward Normalization 적용**

        Args:
            env: Environment
            n_steps: Number of steps to collect

        Returns:
            Average episode reward
        """
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        total_episodes = 0
        total_reward = 0

        for step in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 행동 선택
            with torch.no_grad():
                action, log_prob, value, _ = self.network.get_action_and_value(state_tensor)

            # 환경 스텝
            next_state, raw_reward, done, info = env.step(action)

            # ================================================================
            # Reward Normalization 적용
            # ================================================================
            normalized_reward = self.reward_normalizer.normalize(raw_reward)

            # 버퍼에 저장 (정규화된 보상 사용)
            self.buffer.add(state, action, normalized_reward, value, log_prob, done)

            episode_reward += raw_reward  # 원본 보상으로 통계 기록
            episode_length += 1

            if done:
                total_episodes += 1
                total_reward += episode_reward

                logger.debug(
                    f"Episode {self.episode_count}: "
                    f"reward={episode_reward:.2f}, length={episode_length}"
                )

                state = env.reset()
                episode_reward = 0
                episode_length = 0
                self.episode_count += 1
            else:
                state = next_state

        self.total_rollouts += 1

        avg_episode_reward = total_reward / max(total_episodes, 1)
        self.reward_history.append(avg_episode_reward)  # Track reward
        return avg_episode_reward

    def train_step(self, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """
        PPO 학습 스텝

        **Entropy Annealing 적용**
        **Checkpoint & Rollback 적용**

        Args:
            n_epochs: Number of optimization epochs
            batch_size: Batch size

        Returns:
            Training statistics
        """
        # GAE 계산
        with torch.no_grad():
            last_state = self.buffer.states[self.buffer.pos - 1]
            last_state_tensor = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            _, last_value = self.network(last_state_tensor)
            last_value = last_value.item()

        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

        # Advantages 정규화
        buffer_size = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
        advantages = self.buffer.advantages[:buffer_size]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages[:buffer_size] = advantages

        # ====================================================================
        # Entropy Annealing: 현재 entropy 계수 가져오기
        # ====================================================================
        current_entropy_coef = self.entropy_scheduler.step()

        # PPO 업데이트
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        total_clip_fraction = 0
        update_count = 0

        for epoch in range(n_epochs):
            batch = self.buffer.get_batch(batch_size)

            # Old log probs (behavior policy)
            old_log_probs = batch['log_probs'].detach()

            # Current policy evaluation
            eval_results = self.network.evaluate_actions(batch['states'], batch['actions'])
            new_log_probs = eval_results['log_probs']
            values = eval_results['values']
            entropy = eval_results['entropy']

            # PPO Clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            advantages = batch['advantages']

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (Huber loss for stability)
            value_loss = F.smooth_l1_loss(values, batch['returns'])

            # ================================================================
            # Total loss with annealed entropy
            # ================================================================
            loss = (
                policy_loss +
                self.value_loss_coef * value_loss -
                current_entropy_coef * entropy  # Annealed entropy coefficient
            )

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # KL divergence for early stopping
            with torch.no_grad():
                kl_div = (old_log_probs - new_log_probs).mean().item()
                clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean().item()

            # 통계 업데이트
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl_div += kl_div
            total_clip_fraction += clip_fraction
            update_count += 1

            # KL divergence가 너무 크면 조기 종료
            if kl_div > 0.015:
                logger.info(f"Early stopping at epoch {epoch} due to large KL: {kl_div:.4f}")
                break

        # Learning rate scheduling
        self.lr_scheduler.step()

        # 평균 통계
        avg_policy_loss = total_policy_loss / update_count
        avg_value_loss = total_value_loss / update_count
        avg_entropy = total_entropy / update_count
        avg_kl_div = total_kl_div / update_count
        avg_clip_fraction = total_clip_fraction / update_count

        # Explained variance
        with torch.no_grad():
            returns = self.buffer.returns[:buffer_size]
            values = self.buffer.values[:buffer_size]
            ev = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)

        # 평균 보상 (정규화 전 원본 보상)
        mean_reward = np.mean(self.buffer.rewards[:buffer_size])

        # ====================================================================
        # Checkpoint & Rollback
        # ====================================================================
        self.training_step += 1

        # 현재 메트릭
        current_metrics = CheckpointMetrics(
            step=self.training_step,
            timestamp=datetime.now().isoformat(),
            policy_loss=avg_policy_loss,
            value_loss=avg_value_loss,
            kl_divergence=avg_kl_div,
            mean_reward=mean_reward,
            explained_variance=ev,
            entropy=avg_entropy,
            clip_fraction=avg_clip_fraction
        )

        # 체크포인트 저장
        if self.checkpoint_manager.should_save_checkpoint(self.training_step):
            self.checkpoint_manager.save_checkpoint(
                step=self.training_step,
                network=self.network,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                metrics=current_metrics,
                additional_state={
                    'entropy_scheduler_step': self.entropy_scheduler.current_step,
                    'reward_normalizer_stats': self.reward_normalizer.get_statistics()
                }
            )

        # 롤백 체크
        should_rollback, rollback_reason = self.checkpoint_manager.should_rollback(current_metrics)

        if should_rollback:
            logger.warning(f"[ROLLBACK] Triggered: {rollback_reason}")

            rolled_back_step = self.checkpoint_manager.rollback(
                network=self.network,
                optimizer=self.optimizer,
                scheduler=self.lr_scheduler,
                reason=rollback_reason
            )

            if rolled_back_step is not None:
                logger.info(f"[ROLLBACK] Restored to step {rolled_back_step}")
                # 롤백 후 버퍼 클리어
                self.buffer.clear()

        # 통계 반환
        stats = {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl_div,
            'clip_fraction': avg_clip_fraction,
            'explained_variance': ev,
            'learning_rate': self.lr_scheduler.get_last_lr()[0],
            # 고급 기능 통계
            'current_entropy_coef': current_entropy_coef,
            'entropy_annealing_progress': self.entropy_scheduler.get_info()['progress'],
            'reward_normalizer_mean': self.reward_normalizer.mean,
            'reward_normalizer_std': self.reward_normalizer.std,
            'checkpoint_count': len(self.checkpoint_manager.checkpoints),
            'rollback_count': self.checkpoint_manager.rollback_count
        }

        return stats

    def get_full_statistics(self) -> Dict:
        """전체 통계 반환"""
        entropy_info = self.entropy_scheduler.get_info()
        reward_stats = self.reward_normalizer.get_statistics()
        checkpoint_stats = self.checkpoint_manager.get_statistics()

        return {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'total_rollouts': self.total_rollouts,
            # Reward history
            'rewards': self.reward_history,
            # Entropy annealing
            'entropy': {
                'current': entropy_info['current_entropy'],
                'initial': self.initial_entropy,
                'progress': entropy_info['progress'],
                'remaining_steps': entropy_info['remaining_steps']
            },
            # Reward normalization
            'reward_normalizer': reward_stats,
            # Checkpoint & rollback
            'checkpoint': checkpoint_stats
        }

    def save_full_state(self, filepath: str):
        """전체 학습 상태 저장"""
        full_state = {
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'entropy_scheduler': {
                'current_step': self.entropy_scheduler.current_step,
                'config': {
                    'initial_entropy': self.entropy_scheduler.config.initial_entropy,
                    'final_entropy': self.entropy_scheduler.config.final_entropy,
                    'decay_steps': self.entropy_scheduler.config.decay_steps,
                    'schedule_type': self.entropy_scheduler.config.schedule_type
                }
            },
            'reward_normalizer': self.reward_normalizer.get_statistics(),
            'checkpoint_manager': self.checkpoint_manager.get_statistics()
        }

        torch.save(full_state, filepath)
        logger.info(f"Full training state saved to {filepath}")

    def load_full_state(self, filepath: str):
        """전체 학습 상태 로드"""
        full_state = torch.load(filepath, map_location=self.device)

        self.training_step = full_state['training_step']
        self.episode_count = full_state['episode_count']
        self.network.load_state_dict(full_state['network_state_dict'])
        self.optimizer.load_state_dict(full_state['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(full_state['lr_scheduler_state_dict'])

        # Entropy scheduler state
        self.entropy_scheduler.current_step = full_state['entropy_scheduler']['current_step']

        logger.info(f"Full training state loaded from {filepath}")


# ============================================================================
# 학습 실행 함수
# ============================================================================

def train_advanced_ppo(
    env,
    n_iterations: int = 1000,
    n_steps_per_iteration: int = 2048,
    n_ppo_epochs: int = 10,
    batch_size: int = 64,
    entropy_config: Optional[EntropyScheduleConfig] = None,
    reward_config: Optional[RewardNormalizerConfig] = None,
    checkpoint_config: Optional[CheckpointConfig] = None
):
    """
    Advanced PPO 학습 실행

    Example:
        env = FragranceEnvironment(n_ingredients=20)

        train_advanced_ppo(
            env=env,
            n_iterations=1000,
            entropy_config=EntropyScheduleConfig(
                initial_entropy=0.01,
                final_entropy=0.001,
                decay_steps=100000
            )
        )
    """
    # 트레이너 초기화
    trainer = AdvancedPPOTrainer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        entropy_config=entropy_config,
        reward_config=reward_config,
        checkpoint_config=checkpoint_config
    )

    logger.info("Starting Advanced PPO training...")
    logger.info(f"Iterations: {n_iterations}, Steps/iteration: {n_steps_per_iteration}")

    for iteration in range(n_iterations):
        # 경험 수집
        avg_reward = trainer.collect_rollout(env, n_steps_per_iteration)

        # PPO 업데이트
        train_stats = trainer.train_step(n_ppo_epochs, batch_size)

        # 로깅
        if iteration % 10 == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}:")
            logger.info(f"  Avg Reward: {avg_reward:.3f}")
            logger.info(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
            logger.info(f"  Value Loss: {train_stats['value_loss']:.4f}")
            logger.info(f"  KL Divergence: {train_stats['kl_divergence']:.4f}")
            logger.info(f"  Explained Variance: {train_stats['explained_variance']:.3f}")
            logger.info(f"  [Entropy] Current: {train_stats['current_entropy_coef']:.5f} "
                       f"(Progress: {train_stats['entropy_annealing_progress']:.1%})")
            logger.info(f"  [Reward Norm] Mean: {train_stats['reward_normalizer_mean']:.3f}, "
                       f"Std: {train_stats['reward_normalizer_std']:.3f}")
            logger.info(f"  [Checkpoint] Count: {train_stats['checkpoint_count']}, "
                       f"Rollbacks: {train_stats['rollback_count']}")

        # 전체 통계 출력 (100 iteration마다)
        if iteration % 100 == 0 and iteration > 0:
            full_stats = trainer.get_full_statistics()
            logger.info(f"\n{'='*60}")
            logger.info("Full Statistics:")
            logger.info(f"  Training Step: {full_stats['training_step']}")
            logger.info(f"  Episodes: {full_stats['episode_count']}")
            logger.info(f"  Entropy Progress: {full_stats['entropy']['progress']:.1%}")
            logger.info(f"  Best Reward: {full_stats['checkpoint']['best_reward']:.3f}")
            logger.info(f"  Total Rollbacks: {full_stats['checkpoint']['rollback_count']}")

        # 버퍼 초기화
        trainer.buffer.clear()

    logger.info("\n" + "="*60)
    logger.info("Training complete!")

    # Best checkpoint 로드
    trainer.checkpoint_manager.load_best_checkpoint(
        trainer.network,
        trainer.optimizer,
        trainer.lr_scheduler
    )

    return trainer


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'AdvancedPPOTrainer',
    'train_advanced_ppo'
]
