"""
강화학습 정책 통합 모듈
PPO (Proximal Policy Optimization) + GAE (Generalized Advantage Estimation)
REINFORCE를 fallback으로 지원

사용자 피드백 기반 향수 레시피 최적화
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import deque
import json

logger = logging.getLogger(__name__)


@dataclass
class RolloutBuffer:
    """PPO 롤아웃 버퍼"""
    states: List[np.ndarray]
    actions: List[int]
    log_probs: List[float]
    rewards: List[float]
    values: List[float]
    dones: List[bool]

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def get_batch(self):
        """배치 데이터 반환"""
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'old_log_probs': torch.FloatTensor(self.log_probs),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones)
        }


class PolicyNetwork(nn.Module):
    """정책 네트워크 (Actor)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> Categorical:
        """상태 -> 행동 분포"""
        logits = self.network(state)
        return Categorical(logits=logits)

    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """특정 행동의 log probability"""
        dist = self.forward(state)
        return dist.log_prob(action)

    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """정책 엔트로피"""
        dist = self.forward(state)
        return dist.entropy()


class ValueNetwork(nn.Module):
    """가치 네트워크 (Critic)"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """상태 -> 가치"""
        return self.network(state).squeeze(-1)


class RLPolicy:
    """
    강화학습 정책 통합 인터페이스

    PPO + GAE를 기본으로 사용하며, REINFORCE를 fallback으로 지원
    """

    def __init__(
        self,
        algo: str = "PPO",
        state_dim: int = 256,
        action_dim: int = 16,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr: float = 3e-4,
        device: str = "cuda",
        ckpt_dir: str = "./checkpoints/rl",
        update_epochs: int = 4,
        batch_size: int = 64
    ):
        """
        Args:
            algo: 알고리즘 ("PPO" | "REINFORCE")
            state_dim: 상태 벡터 차원
            action_dim: 행동 개수
            clip_eps: PPO clipping parameter
            value_coef: Value loss 계수
            entropy_coef: Entropy bonus 계수
            max_grad_norm: Gradient clipping
            gamma: Discount factor
            gae_lambda: GAE lambda
            lr: Learning rate
            device: "cuda" | "cpu"
            ckpt_dir: 체크포인트 디렉토리
            update_epochs: PPO 업데이트 epoch 수
            batch_size: 미니배치 크기
        """
        self.algo = algo
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # 네트워크 초기화
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_net = ValueNetwork(state_dim).to(self.device)

        # Optimizer
        if self.algo == "PPO":
            self.optimizer = torch.optim.Adam([
                {'params': self.policy_net.parameters(), 'lr': lr},
                {'params': self.value_net.parameters(), 'lr': lr}
            ])
        else:  # REINFORCE
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # 롤아웃 버퍼
        self.buffer = RolloutBuffer()

        # 통계
        self.episode_rewards = deque(maxlen=100)
        self.update_count = 0

        logger.info(f"RLPolicy initialized: algo={algo}, device={self.device}")

    def select_actions(self, state: np.ndarray, num_options: int) -> Dict[str, Any]:
        """
        상태에서 여러 옵션 생성

        Args:
            state: 상태 벡터 (np.float32[state_dim])
            num_options: 생성할 옵션 개수

        Returns:
            {
                "options": [{"id": str, "action_id": int, "log_prob": float}, ...],
                "saved_actions": [(action_id, log_prob), ...]
            }
        """
        self.policy_net.eval()
        self.value_net.eval()

        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            dist = self.policy_net(state_tensor)

            if self.algo == "PPO":
                value = self.value_net(state_tensor).item()
            else:
                value = None

        options = []
        saved_actions = []

        for i in range(num_options):
            # 행동 샘플링
            action = dist.sample()
            log_prob = dist.log_prob(action).item()
            action_id = action.item()

            option = {
                "id": f"option_{i}",
                "action_id": action_id,
                "log_prob": log_prob
            }

            options.append(option)
            saved_actions.append((action_id, log_prob))

        return {
            "options": options,
            "saved_actions": saved_actions,
            "value": value
        }

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation (GAE) 계산

        Returns:
            advantages, returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        gae = 0
        next_value_tensor = torch.tensor(next_value)

        # 역순으로 계산
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value_tensor
            else:
                next_value_t = values[t + 1]

            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + γ*λ*A_{t+1}
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def update_policy_with_feedback(
        self,
        chosen_id: str,
        options: List[Dict],
        state: np.ndarray,
        saved_actions: List[Tuple[int, float]],
        rating: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        사용자 피드백으로 정책 업데이트

        Args:
            chosen_id: 선택된 옵션 ID
            options: 생성된 옵션 리스트
            state: 상태 벡터
            saved_actions: 저장된 행동 리스트 [(action_id, log_prob), ...]
            rating: 별점 (1-5), 없으면 None

        Returns:
            {
                "loss": float,
                "reward": float,
                "algo": str,
                "clip_frac": float (PPO only),
                "entropy": float (PPO only)
            }
        """
        # Reward 계산
        if rating is not None:
            # 별점 정규화: (rating - 3) / 2
            # 1 -> -1, 3 -> 0, 5 -> 1
            reward = (rating - 3) / 2.0
        else:
            # 선택/비선택
            chosen_idx = next((i for i, opt in enumerate(options) if opt["id"] == chosen_id), None)
            if chosen_idx is None:
                logger.warning(f"Chosen ID {chosen_id} not found in options")
                return {"loss": 0.0, "reward": 0.0, "algo": self.algo}

            reward = 1.0  # 선택된 것

        # 선택된 행동 찾기
        chosen_option = next((opt for opt in options if opt["id"] == chosen_id), None)
        if chosen_option is None:
            return {"loss": 0.0, "reward": 0.0, "algo": self.algo}

        action_id = chosen_option["action_id"]
        old_log_prob = chosen_option["log_prob"]

        if self.algo == "PPO":
            return self._update_ppo(state, action_id, old_log_prob, reward)
        else:  # REINFORCE
            return self._update_reinforce(state, action_id, old_log_prob, reward)

    def _update_ppo(
        self,
        state: np.ndarray,
        action: int,
        old_log_prob: float,
        reward: float
    ) -> Dict[str, Any]:
        """PPO 업데이트"""
        self.policy_net.train()
        self.value_net.train()

        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        old_log_prob_tensor = torch.FloatTensor([old_log_prob]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)

        # Value 계산
        with torch.no_grad():
            old_value = self.value_net(state_tensor)

        # Advantage 계산 (단순 버전: A = R - V)
        advantage = reward_tensor - old_value
        returns = reward_tensor

        # Advantage 정규화
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0

        # 여러 epoch 업데이트
        for epoch in range(self.update_epochs):
            # 새로운 log_prob 계산
            new_log_prob = self.policy_net.get_log_prob(state_tensor, action_tensor)

            # Ratio
            ratio = (new_log_prob - old_log_prob_tensor).exp()

            # Clipped surrogate loss
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantage
            policy_loss = -torch.min(unclipped, clipped).mean()

            # Value loss
            new_value = self.value_net(state_tensor)
            value_loss = 0.5 * (new_value - returns).pow(2).mean()

            # Entropy bonus
            entropy = self.policy_net.get_entropy(state_tensor).mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            # 통계
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

            with torch.no_grad():
                clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()
                total_clip_frac += clip_frac.item()

        self.update_count += 1
        self.episode_rewards.append(reward)

        return {
            "loss": total_policy_loss / self.update_epochs,
            "value_loss": total_value_loss / self.update_epochs,
            "reward": reward,
            "algo": "PPO",
            "clip_frac": total_clip_frac / self.update_epochs,
            "entropy": total_entropy / self.update_epochs,
            "update_count": self.update_count
        }

    def _update_reinforce(
        self,
        state: np.ndarray,
        action: int,
        old_log_prob: float,
        reward: float
    ) -> Dict[str, Any]:
        """REINFORCE 업데이트"""
        self.policy_net.train()

        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)

        # Log probability 계산
        log_prob = self.policy_net.get_log_prob(state_tensor, action_tensor)

        # Loss: -log_prob * reward
        loss = -(log_prob * reward_tensor).mean()

        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.update_count += 1
        self.episode_rewards.append(reward)

        return {
            "loss": loss.item(),
            "reward": reward,
            "algo": "REINFORCE",
            "update_count": self.update_count
        }

    def save(self, suffix: str = ""):
        """모델 저장"""
        save_path = self.ckpt_dir / f"policy_{suffix}.pt" if suffix else self.ckpt_dir / "policy.pt"

        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'config': {
                'algo': self.algo,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'clip_eps': self.clip_eps,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }

        if self.algo == "PPO":
            checkpoint['value_state_dict'] = self.value_net.state_dict()

        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_latest(self) -> bool:
        """최신 모델 로드"""
        policy_path = self.ckpt_dir / "policy.pt"

        if not policy_path.exists():
            logger.warning(f"No checkpoint found at {policy_path}")
            return False

        try:
            checkpoint = torch.load(policy_path, map_location=self.device)

            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            if self.algo == "PPO" and 'value_state_dict' in checkpoint:
                self.value_net.load_state_dict(checkpoint['value_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.update_count = checkpoint.get('update_count', 0)

            logger.info(f"Model loaded from {policy_path}, update_count={self.update_count}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            "algo": self.algo,
            "update_count": self.update_count,
            "avg_reward_100": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "device": str(self.device)
        }


# 전역 인스턴스 (싱글톤)
_global_rl_policy: Optional[RLPolicy] = None


def get_rl_policy(
    algo: str = "PPO",
    state_dim: int = 256,
    action_dim: int = 16,
    **kwargs
) -> RLPolicy:
    """
    전역 RL 정책 인스턴스 반환 (싱글톤)
    """
    global _global_rl_policy

    if _global_rl_policy is None:
        _global_rl_policy = RLPolicy(
            algo=algo,
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs
        )
        # 체크포인트 자동 로드
        _global_rl_policy.load_latest()

    return _global_rl_policy


def reset_rl_policy():
    """전역 인스턴스 리셋 (테스트용)"""
    global _global_rl_policy
    _global_rl_policy = None
