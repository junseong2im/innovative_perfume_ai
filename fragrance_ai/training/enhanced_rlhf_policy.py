"""
강화된 RLHF Policy Network
사용자 피드백 기반 향수 레시피 최적화
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
import json
import hashlib

@dataclass
class FragranceState:
    """향수 상태 벡터"""
    olfactory_dna: np.ndarray  # OlfactoryRecombinatorAI에서 생성된 DNA
    user_feedback: Dict[str, float]  # 사용자 피드백

    def to_tensor(self) -> torch.Tensor:
        """텐서 변환"""
        # DNA 벡터 (30차원)
        dna_vec = self.olfactory_dna[:30] if len(self.olfactory_dna) > 30 else np.pad(self.olfactory_dna, (0, 30 - len(self.olfactory_dna)))

        # 피드백 벡터 (10차원)
        feedback_vec = np.array([
            self.user_feedback.get('harmony', 0),
            self.user_feedback.get('longevity', 0),
            self.user_feedback.get('sillage', 0),
            self.user_feedback.get('creativity', 0),
            self.user_feedback.get('satisfaction', 0),
            self.user_feedback.get('freshness', 0),
            self.user_feedback.get('elegance', 0),
            self.user_feedback.get('uniqueness', 0),
            self.user_feedback.get('wearability', 0),
            self.user_feedback.get('value', 0)
        ])

        # 결합 (40차원)
        state_vector = np.concatenate([dna_vec, feedback_vec])
        return torch.FloatTensor(state_vector)


@dataclass
class FragranceAction:
    """향수 수정 액션"""
    action_type: str  # 'amplify', 'silence', 'add', 'substitute', 'blend'
    target_notes: List[str]  # 대상 노트들
    intensity: float  # 변경 강도 (0-1)

    @staticmethod
    def from_logits(action_logits: torch.Tensor, note_logits: torch.Tensor, intensity: torch.Tensor) -> 'FragranceAction':
        """로짓에서 액션 생성"""
        action_types = ['amplify', 'silence', 'add', 'substitute', 'blend']
        all_notes = ['bergamot', 'rose', 'musk', 'sandalwood', 'jasmine',
                     'vanilla', 'amber', 'patchouli', 'lavender', 'cedarwood']

        # 액션 타입 선택
        action_idx = torch.argmax(action_logits).item()
        action_type = action_types[action_idx]

        # 노트 선택 (상위 3개)
        top_notes_idx = torch.topk(note_logits, k=3).indices
        target_notes = [all_notes[idx % len(all_notes)] for idx in top_notes_idx]

        return FragranceAction(
            action_type=action_type,
            target_notes=target_notes,
            intensity=intensity.item()
        )


class EnhancedPolicyNetwork(nn.Module):
    """향상된 정책 네트워크"""

    def __init__(self, state_dim: int = 40, hidden_dim: int = 512):
        super().__init__()

        # 인코더 레이어
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1)
        )

        # Attention 메커니즘
        self.attention = nn.MultiheadAttention(256, num_heads=8, batch_first=True)

        # 액션 헤드
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 5)  # 5가지 액션 타입
        )

        # 노트 선택 헤드
        self.note_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 30)  # 30개 노트 후보
        )

        # 강도 헤드
        self.intensity_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1 범위
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """순방향 전파"""
        # 인코딩
        features = self.encoder(state)

        # Self-attention (배치 차원 추가)
        if len(features.shape) == 1:
            features = features.unsqueeze(0).unsqueeze(0)
        elif len(features.shape) == 2:
            features = features.unsqueeze(1)

        attended_features, _ = self.attention(features, features, features)
        attended_features = attended_features.squeeze(1)

        # 액션 예측
        action_logits = self.action_head(attended_features)
        note_logits = self.note_head(attended_features)
        intensity = self.intensity_head(attended_features)

        return action_logits, note_logits, intensity


class RLHFTrainer:
    """RLHF 트레이너"""

    def __init__(self, policy_network: EnhancedPolicyNetwork):
        self.policy = policy_network
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=1e-4, weight_decay=0.01)

        # 경험 버퍼
        self.experience_buffer = deque(maxlen=10000)
        self.feedback_buffer = deque(maxlen=5000)

        # 하이퍼파라미터
        self.gamma = 0.99
        self.ppo_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5

    def calculate_reward(
        self,
        user_rating: float,
        feedback_text: str,
        improvement_metrics: Dict[str, float]
    ) -> float:
        """보상 계산"""
        # 1. 사용자 평점 (0-10 -> -1 to 1)
        rating_reward = (user_rating - 5) / 5

        # 2. 텍스트 피드백 감성 분석 (간단한 키워드 기반)
        positive_keywords = ['love', 'perfect', 'amazing', 'great', 'excellent',
                           'beautiful', 'wonderful', 'fantastic', 'best']
        negative_keywords = ['hate', 'bad', 'terrible', 'awful', 'worst',
                           'horrible', 'disgusting', 'poor', 'disappointing']

        text_reward = 0
        feedback_lower = feedback_text.lower()
        for keyword in positive_keywords:
            if keyword in feedback_lower:
                text_reward += 0.1
        for keyword in negative_keywords:
            if keyword in feedback_lower:
                text_reward -= 0.1
        text_reward = np.clip(text_reward, -0.5, 0.5)

        # 3. 개선 메트릭
        metric_reward = 0
        for metric, value in improvement_metrics.items():
            if value > 0:  # 개선됨
                metric_reward += value * 0.1
        metric_reward = np.clip(metric_reward, -0.3, 0.3)

        # 총 보상 (가중 합)
        total_reward = rating_reward * 0.5 + text_reward * 0.3 + metric_reward * 0.2

        return np.clip(total_reward, -1, 1)

    def update_policy_with_feedback(
        self,
        states: List[FragranceState],
        actions: List[FragranceAction],
        rewards: List[float]
    ):
        """피드백으로 정책 업데이트"""
        if len(states) == 0:
            return

        # 텐서 변환
        state_tensors = torch.stack([s.to_tensor() for s in states])
        reward_tensors = torch.FloatTensor(rewards)

        # 현재 정책의 예측
        action_logits, note_logits, intensities = self.policy(state_tensors)

        # 액션 확률
        action_probs = F.softmax(action_logits, dim=-1)
        note_probs = F.softmax(note_logits, dim=-1)

        # 실제 선택된 액션의 인덱스
        action_types = ['amplify', 'silence', 'add', 'substitute', 'blend']
        action_indices = []
        for action in actions:
            idx = action_types.index(action.action_type)
            action_indices.append(idx)
        action_indices = torch.LongTensor(action_indices)

        # 로그 확률
        log_probs = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)) + 1e-8)

        # PPO 손실
        advantages = reward_tensors  # 간단화: 보상을 직접 advantage로 사용

        # 정책 손실
        policy_loss = -(log_probs.squeeze() * advantages).mean()

        # 엔트로피 보너스 (탐험 촉진)
        action_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        note_entropy = -(note_probs * torch.log(note_probs + 1e-8)).sum(dim=-1).mean()
        entropy_bonus = self.entropy_coef * (action_entropy + note_entropy)

        # 총 손실
        total_loss = policy_loss - entropy_bonus

        # 역전파 및 업데이트
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'entropy': (action_entropy + note_entropy).item() / 2,
            'avg_reward': rewards.mean() if rewards else 0
        }

    def generate_variations(
        self,
        state: FragranceState,
        num_variations: int = 3
    ) -> List[FragranceAction]:
        """여러 변형 생성"""
        variations = []

        state_tensor = state.to_tensor().unsqueeze(0)

        for _ in range(num_variations):
            with torch.no_grad():
                action_logits, note_logits, intensity = self.policy(state_tensor)

                # 샘플링 (탐험을 위해)
                action_probs = F.softmax(action_logits, dim=-1)
                note_probs = F.softmax(note_logits, dim=-1)

                # 확률적 샘플링
                action_dist = torch.distributions.Categorical(action_probs)
                note_dist = torch.distributions.Categorical(note_probs)

                sampled_action = action_dist.sample()
                sampled_notes = note_dist.sample()

                action = FragranceAction.from_logits(
                    action_logits[0],
                    note_logits[0],
                    intensity[0]
                )
                variations.append(action)

        return variations

    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'experience_buffer': list(self.experience_buffer)[-1000:],  # 최근 1000개만
            'feedback_buffer': list(self.feedback_buffer)[-500:]  # 최근 500개만
        }, path)

    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 버퍼 복원
        if 'experience_buffer' in checkpoint:
            self.experience_buffer.extend(checkpoint['experience_buffer'])
        if 'feedback_buffer' in checkpoint:
            self.feedback_buffer.extend(checkpoint['feedback_buffer'])


# 통합 시스템
class FragranceEvolutionSystem:
    """향수 진화 시스템 - MOGA + RLHF 통합"""

    def __init__(self):
        self.policy_network = EnhancedPolicyNetwork()
        self.trainer = RLHFTrainer(self.policy_network)

    def evolve_fragrance(
        self,
        current_dna: np.ndarray,
        user_feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """향수 진화"""

        # 현재 상태 생성
        state = FragranceState(
            olfactory_dna=current_dna,
            user_feedback=user_feedback.get('metrics', {})
        )

        # 변형 생성
        variations = self.trainer.generate_variations(state, num_variations=3)

        # 변형 적용 (시뮬레이션)
        evolved_recipes = []
        for i, action in enumerate(variations):
            # DNA 변형 시뮬레이션
            evolved_dna = current_dna.copy()

            if action.action_type == 'amplify':
                # 특정 노트 증폭
                for note in action.target_notes[:3]:
                    idx = hash(note) % len(evolved_dna)
                    evolved_dna[idx] = min(1.0, evolved_dna[idx] * (1 + action.intensity))

            elif action.action_type == 'silence':
                # 특정 노트 감소
                for note in action.target_notes[:3]:
                    idx = hash(note) % len(evolved_dna)
                    evolved_dna[idx] *= (1 - action.intensity)

            elif action.action_type == 'add':
                # 새 노트 추가
                for note in action.target_notes[:2]:
                    idx = hash(note) % len(evolved_dna)
                    evolved_dna[idx] = action.intensity * 0.5

            elif action.action_type == 'blend':
                # 블렌딩 (평균화)
                indices = [hash(n) % len(evolved_dna) for n in action.target_notes]
                avg_value = np.mean([evolved_dna[i] for i in indices])
                for idx in indices:
                    evolved_dna[idx] = evolved_dna[idx] * 0.7 + avg_value * 0.3

            # 정규화
            evolved_dna = np.clip(evolved_dna, 0, 1)

            evolved_recipes.append({
                'id': f'variation_{i+1}',
                'dna': evolved_dna.tolist(),
                'action': {
                    'type': action.action_type,
                    'targets': action.target_notes,
                    'intensity': action.intensity
                },
                'description': self._generate_description(action)
            })

        return {
            'variations': evolved_recipes,
            'original_dna': current_dna.tolist(),
            'feedback_processed': True
        }

    def _generate_description(self, action: FragranceAction) -> str:
        """액션 설명 생성"""
        descriptions = {
            'amplify': f"Amplified {', '.join(action.target_notes[:2])} notes by {int(action.intensity * 100)}%",
            'silence': f"Reduced {', '.join(action.target_notes[:2])} notes by {int(action.intensity * 100)}%",
            'add': f"Added new {', '.join(action.target_notes[:2])} accents",
            'substitute': f"Substituted with {', '.join(action.target_notes[:2])} notes",
            'blend': f"Blended {', '.join(action.target_notes[:2])} harmoniously"
        }
        return descriptions.get(action.action_type, "Modified fragrance composition")

    def train_on_feedback(
        self,
        session_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """세션 데이터로 학습"""
        states = []
        actions = []
        rewards = []

        for data in session_data:
            # 상태 생성
            state = FragranceState(
                olfactory_dna=np.array(data['dna']),
                user_feedback=data.get('feedback', {})
            )
            states.append(state)

            # 액션 파싱
            action_info = data.get('action', {})
            action = FragranceAction(
                action_type=action_info.get('type', 'amplify'),
                target_notes=action_info.get('targets', []),
                intensity=action_info.get('intensity', 0.5)
            )
            actions.append(action)

            # 보상 계산
            reward = self.trainer.calculate_reward(
                user_rating=data.get('rating', 5),
                feedback_text=data.get('feedback_text', ''),
                improvement_metrics=data.get('improvements', {})
            )
            rewards.append(reward)

        # 정책 업데이트
        if states:
            stats = self.trainer.update_policy_with_feedback(states, actions, rewards)
            return stats

        return {'policy_loss': 0, 'entropy': 0, 'avg_reward': 0}


# 결정적 선택을 위한 유틸리티
class DeterministicSelector:
    def __init__(self, seed=42):
        self.seed = seed
        self.counter = 0

    def array(self, size, context=""):
        """결정적 배열 생성"""
        values = []
        for i in range(size):
            content = f"{self.seed}_{self.counter}_{context}_{i}"
            self.counter += 1
            hash_val = int(hashlib.sha256(content.encode()).hexdigest(), 16)
            values.append((hash_val % 10000) / 10000.0)
        return np.array(values)

# 테스트
if __name__ == "__main__":
    system = FragranceEvolutionSystem()
    selector = DeterministicSelector(seed=42)

    # 테스트 DNA
    test_dna = selector.array(30, "test_dna")

    # 테스트 피드백
    test_feedback = {
        'metrics': {
            'harmony': 0.7,
            'longevity': 0.6,
            'sillage': 0.8,
            'creativity': 0.9
        },
        'rating': 7.5,
        'text': "Love the floral notes but needs more depth"
    }

    # 진화 실행
    result = system.evolve_fragrance(test_dna, test_feedback)

    print("Evolution Result:")
    print(f"Generated {len(result['variations'])} variations")
    for var in result['variations']:
        print(f"  - {var['id']}: {var['description']}")