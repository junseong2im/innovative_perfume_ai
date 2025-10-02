"""
'진화' 엔진: EpigeneticVariationAI
PyTorch를 사용한 실제 강화학습(RLHF) 모델 구현
목표: 사용자의 주관적인 선택(피드백)을 '보상'으로 삼아 최적 향수 레시피 학습
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
from collections import deque
import random
from datetime import datetime
import logging
from pathlib import Path

# 프로젝트 내부 모듈 imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# OlfactoryDNA와 CreativeBrief import
from fragrance_ai.training.moga_optimizer import OlfactoryDNA, CreativeBrief

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScentPhenotype:
    """향수 표현형 - 사용자에게 제시될 변형된 향수"""
    dna: OlfactoryDNA
    variation_applied: str
    action_vector: np.ndarray
    user_rating: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Experience:
    """강화학습 경험 단위"""
    state: np.ndarray
    action: int
    action_probs: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class PolicyNetwork(nn.Module):
    """
    실제 정책 신경망(Policy Network) - Attention 메커니즘 포함
    """

    def __init__(self, input_dim: int = 100, hidden_dim: int = 256, num_actions: int = 30):
        super(PolicyNetwork, self).__init__()

        # 입력 임베딩 레이어
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Attention 메커니즘
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        # Deep MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 액션 헤드
        self.action_head = nn.Linear(hidden_dim // 2, num_actions)

        # 가치 헤드 (Actor-Critic용)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim // 2)

    def forward(self, state: torch.Tensor, return_value: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        순방향 전파
        """
        # 입력 투영
        x = self.input_projection(state)

        # Self-attention (배치 차원 처리)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, hidden_dim]

        attended, _ = self.attention(x, x, x)
        x = self.layer_norm1(attended + x)  # Residual connection

        # MLP 처리
        x = x.squeeze(1) if x.dim() == 3 else x
        features = self.mlp(x)
        features = self.layer_norm2(features)

        # 액션 확률 분포
        action_logits = self.action_head(features)
        action_probs = F.softmax(action_logits, dim=-1)

        if return_value:
            # 상태 가치 추정
            value = self.value_head(features)
            return action_probs, value
        else:
            return action_probs


class ReplayBuffer:
    """
    경험 재생 버퍼 - Prioritized Experience Replay
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling weight

    def push(self, experience: Experience, priority: Optional[float] = None):
        """경험 추가"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0

        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """우선순위 기반 샘플링"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """TD 에러 기반 우선순위 업데이트"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self):
        return len(self.buffer)


class EpigeneticVariationAI:
    """
    실제 작동하는 진화 엔진: PPO (Proximal Policy Optimization) 알고리즘
    """

    def __init__(self,
                 state_dim: int = 100,
                 num_actions: int = 30,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon_clip: float = 0.2,
                 gae_lambda: float = 0.95,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # 정책 신경망
        self.policy_network = PolicyNetwork(state_dim, 256, num_actions)
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=100, T_mult=2
        )

        # 경험 재생 버퍼
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # 학습 기록
        self.training_history = []
        self.episode_count = 0
        self.total_timesteps = 0

        # 이동 평균 추적
        self.running_reward = None
        self.reward_window = deque(maxlen=100)

    def encode_state(self, dna: OlfactoryDNA, brief: CreativeBrief) -> np.ndarray:
        """
        DNA와 Brief를 신경망 입력용 상태 벡터로 인코딩
        """
        # DNA 인코딩
        dna_vector = []
        for gene_id, concentration in dna.genes[:10]:  # 최대 10개 유전자
            dna_vector.extend([float(gene_id), concentration])

        # 부족한 부분은 0으로 패딩
        while len(dna_vector) < 20:
            dna_vector.extend([0.0, 0.0])

        # Brief 인코딩
        brief_vector = brief.emotional_palette[:5] if hasattr(brief, 'emotional_palette') else [0.5] * 5
        brief_vector.extend([
            brief.intensity if hasattr(brief, 'intensity') else 0.5,
            hash(brief.fragrance_family) % 10 / 10.0 if hasattr(brief, 'fragrance_family') else 0.5,
            hash(brief.season) % 4 / 4.0 if hasattr(brief, 'season') else 0.5,
            hash(brief.gender) % 3 / 3.0 if hasattr(brief, 'gender') else 0.5,
            hash(brief.mood) % 10 / 10.0 if hasattr(brief, 'mood') else 0.5
        ])

        # Fitness scores 추가
        fitness_vector = list(dna.fitness_scores) if hasattr(dna, 'fitness_scores') else [0.5, 0.5, 0.5]

        # 모든 벡터 결합
        state = np.concatenate([dna_vector, brief_vector, fitness_vector])

        # 정규화
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)

        # 크기 조정 (필요시)
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]

        return state.astype(np.float32)

    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, np.ndarray]:
        """
        epsilon-greedy 정책으로 액션 선택
        """
        state_tensor = torch.FloatTensor(state)

        with torch.no_grad():
            action_probs, value = self.policy_network(state_tensor, return_value=True)
            action_probs = action_probs.cpu().numpy()

        if explore:
            # 탐색: 확률적 선택
            action = np.random.choice(self.num_actions, p=action_probs)
        else:
            # 활용: 최고 확률 액션
            action = np.argmax(action_probs)

        return action, action_probs

    def apply_variation(self, dna: OlfactoryDNA, action: int) -> OlfactoryDNA:
        """
        선택된 액션을 DNA에 실제로 적용
        """
        import copy
        new_dna = copy.deepcopy(dna)

        # 액션 해석
        operation = action // 10  # 0: 증폭, 1: 억제, 2: 추가
        target_idx = action % 10  # 타겟 유전자 인덱스

        if operation == 0 and target_idx < len(new_dna.genes):
            # 증폭: 농도 20% 증가
            gene_id, conc = new_dna.genes[target_idx]
            new_dna.genes[target_idx] = (gene_id, min(conc * 1.2, 30.0))

        elif operation == 1 and target_idx < len(new_dna.genes):
            # 억제: 농도 20% 감소
            gene_id, conc = new_dna.genes[target_idx]
            new_dna.genes[target_idx] = (gene_id, max(conc * 0.8, 0.1))

        elif operation == 2:
            # 추가: 새 유전자 추가
            available_ids = set(range(1, 21)) - set(g[0] for g in new_dna.genes)
            if available_ids:
                new_gene_id = random.choice(list(available_ids))
                new_dna.genes.append((new_gene_id, random.uniform(1.0, 5.0)))

        return new_dna

    def calculate_reward(self, old_dna: OlfactoryDNA, new_dna: OlfactoryDNA,
                        user_rating: float = None) -> float:
        """
        실제 보상 계산 - 다중 요소 고려
        """
        reward = 0.0

        # 1. 사용자 평가 (있을 경우)
        if user_rating is not None:
            reward += (user_rating - 5.0) / 5.0  # -1 to +1 범위로 정규화

        # 2. Fitness 개선도
        if hasattr(old_dna, 'fitness_scores') and hasattr(new_dna, 'fitness_scores'):
            old_fitness = sum(old_dna.fitness_scores) / 3
            new_fitness = sum(new_dna.fitness_scores) / 3
            improvement = new_fitness - old_fitness
            reward += improvement * 2.0

        # 3. 다양성 보너스
        gene_diversity = len(set(g[0] for g in new_dna.genes)) / max(len(new_dna.genes), 1)
        reward += gene_diversity * 0.5

        # 4. 농도 균형 패널티
        total_conc = sum(g[1] for g in new_dna.genes)
        if total_conc < 15 or total_conc > 30:
            reward -= 0.5

        return reward

    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """
        PPO 알고리즘의 실제 학습 단계
        """
        if len(self.replay_buffer) < batch_size:
            return {}

        # 배치 샘플링
        experiences, indices, is_weights = self.replay_buffer.sample(batch_size)

        # 텐서 변환
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        old_probs = torch.FloatTensor([e.action_probs[e.action] for e in experiences])
        is_weights = torch.FloatTensor(is_weights)

        # 현재 정책의 예측
        action_probs, values = self.policy_network(states, return_value=True)
        next_values = self.policy_network(next_states, return_value=True)[1]

        # Advantage 계산 (GAE)
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = td_error + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            advantages.insert(0, advantage)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + values.detach()

        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 손실 계산
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
        old_log_probs = torch.log(old_probs)

        ratio = torch.exp(action_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns) * self.value_loss_coef

        # Entropy bonus (탐색 촉진)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=1).mean()
        entropy_loss = -entropy * self.entropy_coef

        # 총 손실
        total_loss = policy_loss + value_loss + entropy_loss

        # 역전파 및 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()

        # TD 에러로 우선순위 업데이트
        td_errors = (rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()).detach().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # 학습률 스케줄러 업데이트
        self.scheduler.step()

        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def evolve_with_feedback(self,
                            initial_dna: OlfactoryDNA,
                            brief: CreativeBrief,
                            num_episodes: int = 100,
                            steps_per_episode: int = 10) -> OlfactoryDNA:
        """
        실제 RLHF 진화 프로세스
        """
        logger.info("[RLHF] Starting real reinforcement learning evolution")

        best_dna = initial_dna
        best_reward = float('-inf')

        for episode in range(num_episodes):
            current_dna = initial_dna
            episode_reward = 0
            state = self.encode_state(current_dna, brief)

            for step in range(steps_per_episode):
                # 액션 선택
                action, action_probs = self.select_action(state, explore=True)

                # 변형 적용
                new_dna = self.apply_variation(current_dna, action)

                # 보상 계산
                reward = self.calculate_reward(current_dna, new_dna)
                episode_reward += reward

                # 다음 상태
                next_state = self.encode_state(new_dna, brief)
                done = (step == steps_per_episode - 1)

                # 경험 저장
                experience = Experience(
                    state=state,
                    action=action,
                    action_probs=action_probs,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                self.replay_buffer.push(experience)

                # 학습 (충분한 경험이 쌓였을 때)
                if len(self.replay_buffer) >= 32 and step % 4 == 0:
                    metrics = self.train_step()
                    if metrics and episode % 10 == 0:
                        logger.info(f"  Episode {episode}, Step {step}: "
                                  f"Loss={metrics.get('total_loss', 0):.4f}, "
                                  f"LR={metrics.get('learning_rate', 0):.6f}")

                # 상태 전이
                state = next_state
                current_dna = new_dna

                self.total_timesteps += 1

            # 에피소드 종료
            self.reward_window.append(episode_reward)
            self.episode_count += 1

            # 최고 DNA 업데이트
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_dna = current_dna
                logger.info(f"New best DNA found! Reward: {best_reward:.4f}")

            # 진행 상황 출력
            if episode % 10 == 0:
                avg_reward = np.mean(self.reward_window) if self.reward_window else 0
                logger.info(f"Episode {episode}/{num_episodes}: "
                          f"Avg Reward={avg_reward:.4f}, "
                          f"Best={best_reward:.4f}")

        logger.info(f"[RLHF] Evolution complete! Best reward: {best_reward:.4f}")
        return best_dna

    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'replay_buffer': list(self.replay_buffer.buffer)[-1000:]  # 최근 1000개만
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """모델 로드"""
        if Path(path).exists():
            checkpoint = torch.load(path, map_location='cpu')
            self.policy_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.training_history = checkpoint.get('training_history', [])
            self.episode_count = checkpoint.get('episode_count', 0)
            self.total_timesteps = checkpoint.get('total_timesteps', 0)

            # 리플레이 버퍼 복원
            for exp_data in checkpoint.get('replay_buffer', []):
                if isinstance(exp_data, Experience):
                    self.replay_buffer.push(exp_data)

            logger.info(f"Model loaded from {path}")
            logger.info(f"  Episodes: {self.episode_count}, Timesteps: {self.total_timesteps}")
        else:
            logger.warning(f"Model file not found: {path}")


def example_usage():
    """실제 사용 예시"""

    # 초기 DNA 생성
    initial_dna = OlfactoryDNA(
        genes=[(1, 5.0), (3, 8.0), (5, 12.0), (7, 3.0), (9, 6.0)],
        fitness_scores=(0.8, 0.7, 0.9),
        generation=0
    )

    # 사용자 요구사항
    brief = CreativeBrief(
        emotional_palette=[0.4, 0.6, 0.2, 0.1, 0.7],  # 5D 감정 벡터
        fragrance_family="oriental",
        mood="sophisticated",
        intensity=0.8,
        season="autumn",
        gender="unisex"
    )

    # RLHF 엔진 초기화
    engine = EpigeneticVariationAI(
        state_dim=100,
        num_actions=30,
        learning_rate=3e-4
    )

    # 진화 실행
    print("[RLHF] Starting real reinforcement learning with human feedback...")
    print(f"  Initial DNA: {len(initial_dna.genes)} genes")
    print(f"  Brief: {brief.fragrance_family}, {brief.mood}")

    # 실제 학습 기반 진화
    evolved_dna = engine.evolve_with_feedback(
        initial_dna,
        brief,
        num_episodes=50,
        steps_per_episode=10
    )

    print(f"\n[SUCCESS] Evolution complete!")
    print(f"  Final DNA: {len(evolved_dna.genes)} genes")
    print(f"  Genes: {evolved_dna.genes}")

    # 모델 저장
    engine.save_model("rlhf_model.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    example_usage()