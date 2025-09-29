"""
진짜 강화학습 기반 향수 진화 시스템
Reinforcement Learning from Human Feedback (RLHF) 구현
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from collections import deque
import random
from datetime import datetime

@dataclass
class State:
    """향수 상태 표현"""
    dna_features: np.ndarray  # DNA 특징 벡터
    phenotype_features: np.ndarray  # 표현형 특징 벡터
    user_context: np.ndarray  # 사용자 컨텍스트

    def to_tensor(self) -> torch.Tensor:
        """텐서로 변환"""
        combined = np.concatenate([
            self.dna_features,
            self.phenotype_features,
            self.user_context
        ])
        return torch.FloatTensor(combined)

@dataclass
class Action:
    """향수 수정 행동"""
    modification_type: str  # 'amplify', 'silence', 'modulate', 'substitute'
    target_gene: str  # 수정 대상
    modification_strength: float  # 수정 강도 (0-1)

    def to_vector(self) -> np.ndarray:
        """벡터로 변환"""
        type_encoding = {
            'amplify': [1, 0, 0, 0],
            'silence': [0, 1, 0, 0],
            'modulate': [0, 0, 1, 0],
            'substitute': [0, 0, 0, 1]
        }

        gene_encoding = {
            'top': [1, 0, 0],
            'middle': [0, 1, 0],
            'base': [0, 0, 1],
            'all': [1, 1, 1]
        }

        return np.array(
            type_encoding.get(self.modification_type, [0, 0, 0, 0]) +
            gene_encoding.get(self.target_gene, [0, 0, 0]) +
            [self.modification_strength]
        )

@dataclass
class Experience:
    """경험 (상태, 행동, 보상, 다음 상태)"""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool

class PolicyNetwork(nn.Module):
    """정책 네트워크 - 어떤 수정을 할지 결정"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)

        # 출력 헤드들
        self.action_type_head = nn.Linear(128, 4)  # 4가지 수정 타입
        self.target_gene_head = nn.Linear(128, 4)  # top, middle, base, all
        self.strength_head = nn.Linear(128, 1)  # 수정 강도

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """순방향 전파"""
        x = self.activation(self.fc1(state))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        # 각 출력 계산
        action_type = torch.softmax(self.action_type_head(x), dim=-1)
        target_gene = torch.softmax(self.target_gene_head(x), dim=-1)
        strength = torch.sigmoid(self.strength_head(x))

        return action_type, target_gene, strength

class ValueNetwork(nn.Module):
    """가치 네트워크 - 상태의 가치 평가"""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 64)
        self.value_head = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순방향 전파"""
        x = self.activation(self.fc1(state))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        value = self.value_head(x)
        return value

class RewardModel(nn.Module):
    """보상 모델 - 인간 피드백 학습"""

    def __init__(self, state_dim: int, action_dim: int = 8):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.reward_head = nn.Linear(64, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """예측 보상 계산"""
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        reward = torch.tanh(self.reward_head(x))  # -1 ~ 1 범위
        return reward

class FragranceRLHF:
    """향수 강화학습 시스템"""

    def __init__(self, state_dim: int = 100):
        self.state_dim = state_dim

        # 신경망들
        self.policy_net = PolicyNetwork(state_dim)
        self.value_net = ValueNetwork(state_dim)
        self.reward_model = RewardModel(state_dim)

        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.001)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)

        # 경험 버퍼
        self.experience_buffer = deque(maxlen=10000)
        self.human_feedback_buffer = deque(maxlen=5000)

        # 하이퍼파라미터
        self.gamma = 0.99  # 할인율
        self.gae_lambda = 0.95  # GAE 람다
        self.ppo_epsilon = 0.2  # PPO 클리핑
        self.entropy_coef = 0.01  # 엔트로피 보너스

        # 통계
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'reward_loss': [],
            'average_reward': []
        }

    def encode_fragrance_state(self, dna: Any, phenotype: Optional[Any] = None) -> State:
        """향수를 상태로 인코딩"""
        # DNA 특징 추출
        dna_features = []

        # 각 노트의 특징
        for note_type in ['top', 'middle', 'base']:
            genes = dna.genotype.get(note_type, [])
            if genes:
                concentrations = [g.concentration for g in genes]
                expressions = [g.expression_level for g in genes]
                dna_features.extend([
                    np.mean(concentrations),
                    np.std(concentrations),
                    np.mean(expressions),
                    np.std(expressions)
                ])
            else:
                dna_features.extend([0, 0, 0, 0])

        # 표현형 잠재력
        for key in ['longevity', 'sillage', 'complexity', 'balance']:
            dna_features.append(dna.phenotype_potential.get(key, 0.5))

        # 표현형 특징 (있으면)
        phenotype_features = []
        if phenotype:
            for key in ['temperature_sensitivity', 'humidity_response',
                       'temporal_stability', 'emotional_resonance']:
                phenotype_features.append(
                    phenotype.environmental_response.get(key, 0.5)
                )
        else:
            phenotype_features = [0.5] * 4

        # 사용자 컨텍스트 (간단한 예시)
        user_context = [
            random.random(),  # 시간대
            random.random(),  # 계절
            random.random(),  # 선호도
        ]

        # 패딩으로 차원 맞추기
        dna_features = np.array(dna_features)
        phenotype_features = np.array(phenotype_features)
        user_context = np.array(user_context)

        # 고정 크기로 패딩
        dna_features = np.pad(dna_features, (0, 50 - len(dna_features)), 'constant')
        phenotype_features = np.pad(phenotype_features, (0, 30 - len(phenotype_features)), 'constant')
        user_context = np.pad(user_context, (0, 20 - len(user_context)), 'constant')

        return State(dna_features, phenotype_features, user_context)

    def select_action(self, state: State, epsilon: float = 0.1) -> Action:
        """행동 선택 (epsilon-greedy)"""
        if random.random() < epsilon:
            # 탐색: 랜덤 행동
            action_types = ['amplify', 'silence', 'modulate', 'substitute']
            target_genes = ['top', 'middle', 'base', 'all']

            return Action(
                modification_type=random.choice(action_types),
                target_gene=random.choice(target_genes),
                modification_strength=random.random()
            )
        else:
            # 활용: 정책 네트워크 사용
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0)
                action_type_probs, target_gene_probs, strength = self.policy_net(state_tensor)

                # 확률적 샘플링
                action_type_idx = torch.multinomial(action_type_probs, 1).item()
                target_gene_idx = torch.multinomial(target_gene_probs, 1).item()

                action_types = ['amplify', 'silence', 'modulate', 'substitute']
                target_genes = ['top', 'middle', 'base', 'all']

                return Action(
                    modification_type=action_types[action_type_idx],
                    target_gene=target_genes[target_gene_idx],
                    modification_strength=strength.item()
                )

    def calculate_reward(self, state: State, action: Action, human_rating: Optional[float] = None) -> float:
        """보상 계산"""
        if human_rating is not None:
            # 실제 인간 평가 사용 (1-5 -> -1~1로 정규화)
            return (human_rating - 3) / 2
        else:
            # 학습된 보상 모델 사용
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0)
                action_tensor = torch.FloatTensor(action.to_vector()).unsqueeze(0)
                predicted_reward = self.reward_model(state_tensor, action_tensor)
                return predicted_reward.item()

    def store_experience(self, experience: Experience):
        """경험 저장"""
        self.experience_buffer.append(experience)

    def store_human_feedback(self, state: State, action: Action, rating: float):
        """인간 피드백 저장"""
        self.human_feedback_buffer.append((state, action, rating))

    def train_reward_model(self, batch_size: int = 32):
        """보상 모델 학습 (인간 피드백 기반)"""
        if len(self.human_feedback_buffer) < batch_size:
            return

        # 배치 샘플링
        batch = random.sample(self.human_feedback_buffer, batch_size)

        states = torch.stack([item[0].to_tensor() for item in batch])
        actions = torch.stack([torch.FloatTensor(item[1].to_vector()) for item in batch])
        ratings = torch.FloatTensor([(item[2] - 3) / 2 for item in batch])  # 정규화

        # 예측 및 손실 계산
        predicted_rewards = self.reward_model(states, actions).squeeze()
        loss = nn.MSELoss()(predicted_rewards, ratings)

        # 역전파
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()

        self.training_stats['reward_loss'].append(loss.item())

    def compute_gae(self, rewards: List[float], values: List[float], next_values: List[float]) -> List[float]:
        """Generalized Advantage Estimation 계산"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)

        return advantages

    def train_ppo(self, batch_size: int = 64, epochs: int = 4):
        """PPO 알고리즘으로 정책 학습"""
        if len(self.experience_buffer) < batch_size:
            return

        # 배치 샘플링
        batch = random.sample(self.experience_buffer, batch_size)

        states = torch.stack([exp.state.to_tensor() for exp in batch])
        actions = torch.stack([torch.FloatTensor(exp.action.to_vector()) for exp in batch])
        rewards = [exp.reward for exp in batch]

        # 가치 계산
        with torch.no_grad():
            values = self.value_net(states).squeeze().tolist()
            next_states = torch.stack([exp.next_state.to_tensor() for exp in batch])
            next_values = self.value_net(next_states).squeeze().tolist()

        # GAE 계산
        advantages = self.compute_gae(rewards, values, next_values)
        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)

        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 이전 정책의 로그 확률 저장
        with torch.no_grad():
            action_type_probs_old, target_gene_probs_old, strength_old = self.policy_net(states)
            # 간단한 로그 확률 계산 (실제로는 더 복잡)
            old_log_probs = torch.log(action_type_probs_old.mean(dim=1) + 1e-8)

        # PPO 업데이트
        for _ in range(epochs):
            # 정책 네트워크 출력
            action_type_probs, target_gene_probs, strength = self.policy_net(states)
            log_probs = torch.log(action_type_probs.mean(dim=1) + 1e-8)

            # 비율 계산
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO 손실
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 엔트로피 보너스
            entropy = -(action_type_probs * torch.log(action_type_probs + 1e-8)).sum(dim=1).mean()
            policy_loss -= self.entropy_coef * entropy

            # 정책 업데이트
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            # 가치 네트워크 업데이트
            value_pred = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(value_pred, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['average_reward'].append(np.mean(rewards))

    def evolve_fragrance(self, dna: Any, user_feedback: str, rating: float) -> Dict[str, Any]:
        """강화학습으로 향수 진화"""
        # 현재 상태
        current_state = self.encode_fragrance_state(dna)

        # 행동 선택
        action = self.select_action(current_state, epsilon=0.1)

        # 인간 피드백 저장
        self.store_human_feedback(current_state, action, rating)

        # 보상 계산
        reward = self.calculate_reward(current_state, action, rating)

        # 수정 지시 생성
        modification_instructions = {
            'type': action.modification_type,
            'target': action.target_gene,
            'strength': action.modification_strength,
            'user_feedback': user_feedback,
            'reward': reward
        }

        # 다음 상태 시뮬레이션 (실제로는 수정 후 상태)
        next_state = self.encode_fragrance_state(dna)  # 간단한 예시

        # 경험 저장
        experience = Experience(
            state=current_state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=False
        )
        self.store_experience(experience)

        # 모델 학습
        if len(self.experience_buffer) >= 100:
            self.train_reward_model()
            self.train_ppo()

        return modification_instructions

    def get_statistics(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        return {
            'total_experiences': len(self.experience_buffer),
            'total_human_feedbacks': len(self.human_feedback_buffer),
            'avg_policy_loss': np.mean(self.training_stats['policy_loss'][-100:]) if self.training_stats['policy_loss'] else 0,
            'avg_value_loss': np.mean(self.training_stats['value_loss'][-100:]) if self.training_stats['value_loss'] else 0,
            'avg_reward_loss': np.mean(self.training_stats['reward_loss'][-100:]) if self.training_stats['reward_loss'] else 0,
            'avg_reward': np.mean(self.training_stats['average_reward'][-100:]) if self.training_stats['average_reward'] else 0
        }

# 싱글톤 인스턴스
_fragrance_rlhf_instance = None

def get_fragrance_rlhf() -> FragranceRLHF:
    """싱글톤 FragranceRLHF 인스턴스 반환"""
    global _fragrance_rlhf_instance
    if _fragrance_rlhf_instance is None:
        _fragrance_rlhf_instance = FragranceRLHF()
    return _fragrance_rlhf_instance