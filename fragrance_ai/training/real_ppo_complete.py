"""
PPO (Proximal Policy Optimization) - 완전한 구현
실제 강화학습 알고리즘 with RLHF
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from collections import deque
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RolloutBuffer:
    """PPO를 위한 경험 버퍼 - 실제 구현"""

    def __init__(self, buffer_size: int, state_dim: int, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        # 버퍼 초기화
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

    def add(self, state: np.ndarray, action: int, reward: float,
            value: float, log_prob: float, done: bool):
        """경험 추가"""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99,
                                      gae_lambda: float = 0.95):
        """GAE (Generalized Advantage Estimation) 계산 - 실제 공식"""
        last_gae_lam = 0
        buffer_size = self.buffer_size if self.full else self.pos

        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            # TD error: δ = r + γV(s') - V(s)
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]

            # GAE: A = δ + γλA'
            self.advantages[step] = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            last_gae_lam = self.advantages[step]

        # Returns = advantages + values
        buffer_size = self.buffer_size if self.full else self.pos
        self.returns[:buffer_size] = self.advantages[:buffer_size] + self.values[:buffer_size]

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """배치 샘플링"""
        buffer_size = self.buffer_size if self.full else self.pos
        indices = np.random.choice(buffer_size, min(batch_size, buffer_size), replace=False)

        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.LongTensor(self.actions[indices]).to(self.device),
            'returns': torch.FloatTensor(self.returns[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'advantages': torch.FloatTensor(self.advantages[indices]).to(self.device)
        }

    def clear(self):
        """버퍼 초기화"""
        self.pos = 0
        self.full = False


class ActorCriticNetwork(nn.Module):
    """Actor-Critic 신경망 - 실제 아키텍처"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared layers with batch norm and dropout
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Orthogonal initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal initialization - PPO 논문 추천"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value

    def get_action_and_value(self, state: torch.Tensor) -> Tuple[int, float, float, float]:
        """행동 선택 및 가치 예측"""
        action_logits, value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob.item(), value.item(), entropy.item()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """배치 평가 - PPO 업데이트용"""
        action_logits, values = self.forward(states)
        probs = F.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        return {
            'log_probs': log_probs,
            'values': values.squeeze(-1),
            'entropy': entropy
        }


class PPOTrainer:
    """PPO 트레이너 - 완전한 구현"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 네트워크 초기화
        self.network = ActorCriticNetwork(state_dim, action_dim).to(device)

        # Adam optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-5
        )

        # 경험 버퍼
        self.buffer = RolloutBuffer(2048, state_dim, device)

        # 통계 추적
        self.training_stats = {
            'policy_losses': deque(maxlen=100),
            'value_losses': deque(maxlen=100),
            'entropies': deque(maxlen=100),
            'kl_divs': deque(maxlen=100),
            'clip_fractions': deque(maxlen=100),
            'explained_variances': deque(maxlen=100)
        }

        # RLHF 보상 모델
        self.reward_model = self._build_reward_model(state_dim)

    def _build_reward_model(self, state_dim: int) -> nn.Module:
        """인간 피드백 기반 보상 모델"""
        model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        return model

    def train_step(self, n_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """PPO 학습 스텝 - 실제 알고리즘"""

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

            # Total loss
            loss = (
                policy_loss +
                self.value_loss_coef * value_loss -
                self.entropy_coef * entropy
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
                logger.info(f"Early stopping at epoch {epoch} due to large KL divergence: {kl_div:.4f}")
                break

        # Learning rate scheduling
        self.scheduler.step()

        # 통계 기록
        avg_policy_loss = total_policy_loss / update_count
        avg_value_loss = total_value_loss / update_count
        avg_entropy = total_entropy / update_count
        avg_kl_div = total_kl_div / update_count
        avg_clip_fraction = total_clip_fraction / update_count

        self.training_stats['policy_losses'].append(avg_policy_loss)
        self.training_stats['value_losses'].append(avg_value_loss)
        self.training_stats['entropies'].append(avg_entropy)
        self.training_stats['kl_divs'].append(avg_kl_div)
        self.training_stats['clip_fractions'].append(avg_clip_fraction)

        # Explained variance 계산
        with torch.no_grad():
            buffer_size = self.buffer.buffer_size if self.buffer.full else self.buffer.pos
            returns = self.buffer.returns[:buffer_size]
            values = self.buffer.values[:buffer_size]
            ev = 1 - np.var(returns - values) / (np.var(returns) + 1e-8)
            self.training_stats['explained_variances'].append(ev)

        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl_divergence': avg_kl_div,
            'clip_fraction': avg_clip_fraction,
            'explained_variance': ev,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def collect_rollout(self, env, n_steps: int = 2048) -> float:
        """환경과 상호작용하여 경험 수집"""
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 행동 선택
            with torch.no_grad():
                action, log_prob, value, _ = self.network.get_action_and_value(state_tensor)

            # 환경 스텝
            next_state, reward, done, info = env.step(action)

            # RLHF: 인간 피드백으로 보상 조정
            if hasattr(self, 'human_feedback'):
                reward = self.adjust_reward_with_feedback(state, action, reward)

            # 버퍼에 저장
            self.buffer.add(state, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1

            if done:
                state = env.reset()
                logger.info(f"Episode finished. Reward: {episode_reward:.2f}, Length: {episode_length}")
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state

        return episode_reward / max(episode_length, 1)

    def adjust_reward_with_feedback(self, state: np.ndarray, action: int, env_reward: float) -> float:
        """RLHF: 인간 피드백 기반 보상 조정"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            human_preference = self.reward_model(state_tensor).item()

        # 환경 보상과 인간 선호도 결합
        adjusted_reward = 0.7 * env_reward + 0.3 * human_preference
        return adjusted_reward

    def train_reward_model(self, feedback_data: List[Dict]) -> float:
        """인간 피드백으로 보상 모델 학습"""
        if not feedback_data:
            return 0.0

        # 피드백 데이터 준비
        states = torch.FloatTensor([f['state'] for f in feedback_data]).to(self.device)
        ratings = torch.FloatTensor([f['rating'] for f in feedback_data]).to(self.device)

        # 보상 모델 학습
        reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=1e-3)

        for _ in range(100):  # Mini epochs
            predicted_rewards = self.reward_model(states).squeeze()
            loss = F.mse_loss(predicted_rewards, ratings)

            reward_optimizer.zero_grad()
            loss.backward()
            reward_optimizer.step()

        return loss.item()

    def save_checkpoint(self, filepath: str):
        """모델 체크포인트 저장"""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'training_stats': dict(self.training_stats)
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """모델 체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_model_state_dict'])
        logger.info(f"Checkpoint loaded from {filepath}")


class FragranceEnvironment:
    """향수 조합 환경 - 강화학습용"""

    def __init__(self, n_ingredients: int = 20):
        self.n_ingredients = n_ingredients
        self.state_dim = n_ingredients * 2  # 성분 ID + 농도
        self.action_dim = n_ingredients * 3  # 추가/제거/조정
        self.current_formula = None
        self.reset()

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        # 랜덤 초기 포뮬러
        n_initial = np.random.randint(5, 10)
        self.current_formula = {}

        for _ in range(n_initial):
            ing_id = np.random.randint(0, self.n_ingredients)
            concentration = np.random.uniform(0.1, 5.0)
            self.current_formula[ing_id] = concentration

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """현재 상태 벡터화"""
        state = np.zeros(self.state_dim)

        for ing_id, conc in self.current_formula.items():
            state[ing_id * 2] = 1.0  # 성분 존재
            state[ing_id * 2 + 1] = conc / 10.0  # 농도 정규화

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """환경 스텝"""
        # 행동 디코딩
        operation = action // self.n_ingredients
        ingredient_id = action % self.n_ingredients

        if operation == 0:  # 추가
            if ingredient_id not in self.current_formula:
                self.current_formula[ingredient_id] = np.random.uniform(0.5, 3.0)
        elif operation == 1:  # 제거
            if ingredient_id in self.current_formula:
                del self.current_formula[ingredient_id]
        else:  # 조정
            if ingredient_id in self.current_formula:
                self.current_formula[ingredient_id] *= np.random.uniform(0.8, 1.2)
                self.current_formula[ingredient_id] = np.clip(
                    self.current_formula[ingredient_id], 0.1, 10.0
                )

        # 보상 계산
        reward = self._calculate_reward()

        # 종료 조건
        done = len(self.current_formula) > 15 or len(self.current_formula) < 3

        return self._get_state(), reward, done, {}

    def _calculate_reward(self) -> float:
        """보상 함수"""
        reward = 0.0

        # 성분 개수 페널티/보너스
        n_ingredients = len(self.current_formula)
        if 8 <= n_ingredients <= 12:
            reward += 1.0
        else:
            reward -= abs(10 - n_ingredients) * 0.1

        # 농도 합 체크
        total_conc = sum(self.current_formula.values())
        if 15 <= total_conc <= 25:
            reward += 0.5
        else:
            reward -= abs(20 - total_conc) * 0.05

        return reward


# 메인 학습 루프
def main():
    """PPO 학습 실행"""
    # 환경 및 트레이너 초기화
    env = FragranceEnvironment(n_ingredients=20)
    trainer = PPOTrainer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=3e-4,
        clip_epsilon=0.2
    )

    # 학습 파라미터
    n_iterations = 1000
    n_steps_per_iteration = 2048
    n_ppo_epochs = 10

    logger.info("Starting PPO training...")

    for iteration in range(n_iterations):
        # 경험 수집
        avg_reward = trainer.collect_rollout(env, n_steps_per_iteration)

        # PPO 업데이트
        train_stats = trainer.train_step(n_ppo_epochs, batch_size=64)

        # 로깅
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}:")
            logger.info(f"  Average Reward: {avg_reward:.3f}")
            logger.info(f"  Policy Loss: {train_stats['policy_loss']:.4f}")
            logger.info(f"  Value Loss: {train_stats['value_loss']:.4f}")
            logger.info(f"  KL Divergence: {train_stats['kl_divergence']:.4f}")
            logger.info(f"  Explained Variance: {train_stats['explained_variance']:.3f}")

        # 체크포인트 저장
        if iteration % 100 == 0:
            trainer.save_checkpoint(f"ppo_checkpoint_{iteration}.pth")

        # 버퍼 초기화
        trainer.buffer.clear()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()