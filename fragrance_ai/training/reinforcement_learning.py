"""
'진화' 엔진: EpigeneticVariationAI
PyTorch를 사용한 강화학습(RLHF) 모델 구현
목표: 사용자의 주관적인 선택(피드백)을 '보상'으로 삼아, 어떤 종류의 '변형'이 사용자를 만족시키는지 학습
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from collections import deque
import random
from datetime import datetime
import logging

# 프로젝트 내부 모듈 imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# OlfactoryDNA와 CreativeBrief import
from fragrance_ai.training.moga_optimizer import OlfactoryDNA, CreativeBrief

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScentPhenotype:
    """향수 표현형 - 사용자에게 제시될 변형된 향수"""
    dna: OlfactoryDNA
    variation_applied: str  # 적용된 변형 종류
    user_rating: Optional[float] = None  # 사용자 평가 (1-10)


class PolicyNetwork(nn.Module):
    """
    1단계: 정책 신경망(Policy Network) 모델 정의
    torch.nn.Module을 상속받는 간단한 MLP(Multi-Layer Perceptron) 모델

    입력(State): 현재 향수의 OlfactoryDNA 벡터와 사용자의 피드백(CreativeBrief) 벡터를 합친 벡터
    출력(Action): 가능한 모든 '변형' 방법에 대한 확률 분포
    """

    def __init__(self, input_dim: int = 100, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()

        # MLP 레이어들
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)

        # 출력 레이어 - 변형 방법에 대한 확률
        # 변형의 종류: ['Amplify_Note_A', 'Silence_Note_B', 'Add_New_Note_C', ...]
        self.action_head = nn.Linear(128, 30)  # 10개 노트 x 3개 행동 = 30개 액션

        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        마지막 레이어는 Softmax 활성화 함수를 사용하여 각 행동에 대한 확률 값을 출력
        """
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))

        # Softmax를 통한 확률 분포 생성
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs


class EpigeneticVariationAI:
    """
    진화 엔진: 사용자 피드백을 통해 학습하는 강화학습 모델
    REINFORCE 알고리즘을 사용한 정책 경사(Policy Gradient) 구현
    """

    def __init__(self,
                 state_dim: int = 100,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99):

        self.state_dim = state_dim
        self.learning_rate = learning_rate
        self.gamma = gamma  # 할인율

        # 정책 신경망 초기화
        self.policy_network = PolicyNetwork(input_dim=state_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # 경험 버퍼 (에피소드 기록)
        self.episode_log_probs = []
        self.episode_rewards = []

        # 변형 액션 정의
        self.action_space = self._define_action_space()

        # 학습 히스토리
        self.training_history = []

    def _define_action_space(self) -> List[str]:
        """
        가능한 변형 행동들 정의
        출력(Action): 가능한 모든 '변형' 방법
        ['Amplify_Note_A', 'Silence_Note_B', 'Add_New_Note_C', ...]
        """
        actions = []
        note_names = ['Bergamot', 'Lemon', 'Rose', 'Jasmine', 'Sandalwood',
                     'Cedar', 'Vanilla', 'Musk', 'Amber', 'Patchouli']

        for note in note_names:
            actions.append(f"Amplify_{note}")
            actions.append(f"Silence_{note}")
            actions.append(f"Add_{note}")

        return actions

    def encode_state(self, dna: OlfactoryDNA, brief: CreativeBrief) -> torch.Tensor:
        """
        입력(State): 현재 향수의 OlfactoryDNA 벡터와 사용자의 피드백(CreativeBrief) 벡터를 합친 벡터
        """
        # DNA 인코딩 (노트와 농도)
        dna_vector = np.zeros(50)  # 10 노트 x 5 특징
        for i, (note_id, percentage) in enumerate(dna.genes[:10]):
            if i < 10:
                dna_vector[i*5] = note_id / 10.0  # 정규화
                dna_vector[i*5 + 1] = percentage / 30.0  # 정규화
                dna_vector[i*5 + 2] = dna.fitness_scores[0] if dna.fitness_scores else 0
                dna_vector[i*5 + 3] = dna.fitness_scores[1] if dna.fitness_scores else 0
                dna_vector[i*5 + 4] = dna.fitness_scores[2] if dna.fitness_scores else 0

        # CreativeBrief 인코딩
        brief_vector = np.zeros(50)
        brief_vector[:3] = brief.emotional_palette[:3]
        brief_vector[3] = brief.intensity
        # 추가 특징들을 인코딩할 수 있음

        # 상태 벡터 합치기 (concatenate)
        state_vector = np.concatenate([dna_vector, brief_vector])

        return torch.FloatTensor(state_vector).unsqueeze(0)  # 배치 차원 추가

    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """
        2단계: 행동 실행 및 보상 획득
        a. 행동 샘플링: 현재 State를 정책 신경망에 입력하여,
        출력된 확률 분포에 따라 여러 개의 '변형' 행동(Action)을 샘플링
        """
        # 정책 네트워크를 통한 확률 분포 계산
        action_probs = self.policy_network(state)

        # 확률 분포에 따라 행동 샘플링
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()

        # 로그 확률 계산 (나중에 학습에 사용)
        log_prob = action_distribution.log_prob(action)

        return action.item(), log_prob

    def apply_variation(self, dna: OlfactoryDNA, action_idx: int) -> OlfactoryDNA:
        """
        샘플링된 행동을 DNA에 적용하여 변형된 DNA 생성
        """
        # 행동 해석
        action_name = self.action_space[action_idx]
        action_type, note_name = action_name.split('_', 1)

        # DNA 복사
        new_genes = list(dna.genes)

        # 노트 ID 매핑 (간단한 예시)
        note_mapping = {
            'Bergamot': 1, 'Lemon': 2, 'Rose': 3, 'Jasmine': 4,
            'Sandalwood': 5, 'Cedar': 6, 'Vanilla': 7, 'Musk': 8,
            'Amber': 9, 'Patchouli': 10
        }

        note_id = note_mapping.get(note_name, 1)

        if action_type == "Amplify":
            # 해당 노트의 농도 증가
            for i, (nid, percentage) in enumerate(new_genes):
                if nid == note_id:
                    new_genes[i] = (nid, min(percentage * 1.5, 30.0))
                    break

        elif action_type == "Silence":
            # 해당 노트 제거 또는 감소
            for i, (nid, percentage) in enumerate(new_genes):
                if nid == note_id:
                    new_genes[i] = (nid, percentage * 0.3)
                    break

        elif action_type == "Add":
            # 새로운 노트 추가 (빈 슬롯이 있으면)
            added = False
            for i, (nid, percentage) in enumerate(new_genes):
                if percentage < 0.1:  # 거의 사용하지 않는 슬롯
                    new_genes[i] = (note_id, random.uniform(1.0, 5.0))
                    added = True
                    break

            if not added and len(new_genes) < 15:
                # 슬롯 추가
                new_genes.append((note_id, random.uniform(1.0, 5.0)))

        # 새로운 DNA 객체 생성
        return OlfactoryDNA(
            genes=new_genes,
            fitness_scores=dna.fitness_scores  # 일단 동일하게 유지
        )

    def generate_variations(self, dna: OlfactoryDNA, brief: CreativeBrief, num_variations: int = 3) -> List[ScentPhenotype]:
        """
        b. 사용자에게 제시: 이 행동들을 적용하여 생성된 여러 개의 ScentPhenotype 후보 A, B, C를 생성
        """
        variations = []
        state = self.encode_state(dna, brief)

        for _ in range(num_variations):
            # 행동 샘플링
            action_idx, log_prob = self.sample_action(state)

            # 변형 적용
            varied_dna = self.apply_variation(dna, action_idx)

            # ScentPhenotype 생성
            phenotype = ScentPhenotype(
                dna=varied_dna,
                variation_applied=self.action_space[action_idx]
            )

            variations.append(phenotype)

            # 로그 확률 저장 (학습용)
            self.episode_log_probs.append(log_prob)

        return variations

    def update_policy_with_feedback(self, variations: List[ScentPhenotype], selected_idx: int):
        """
        3단계: 정책 업데이트 (학습)
        REINFORCE 알고리즘을 사용하여 정책 신경망의 가중치를 업데이트

        c. 보상 정의: 사용자가 후보 B를 선택하면, 행동 B에 대한 보상(reward)은 +1,
        선택받지 못한 A와 C에 대한 보상은 -1 (또는 0)로 설정
        """

        # 보상 설정
        rewards = []
        for i, phenotype in enumerate(variations):
            if i == selected_idx:
                # 선택된 변형에 대한 긍정적 보상
                reward = 1.0
                logger.info(f"✨ 사용자가 선택한 변형: {phenotype.variation_applied}")
            else:
                # 선택되지 않은 변형에 대한 부정적/중립적 보상
                reward = -0.5  # 또는 0

            rewards.append(reward)
            phenotype.user_rating = reward  # 기록용

        self.episode_rewards.extend(rewards)

        # REINFORCE 알고리즘 적용
        self._update_policy()

    def _update_policy(self):
        """
        REINFORCE 알고리즘을 사용한 정책 업데이트
        핵심 수식: Loss = -log(P(action_chosen)) * reward

        - P(action_chosen): 정책 신경망이 '사용자가 선택한 행동'을 예측했던 확률
        - reward: 위에서 정의한 보상 값(+1 또는 -0.5)

        만약 보상이 긍정적(+1)이면, 손실 함수는 log(P(action_chosen))를 최대화
        만약 보상이 부정적(-1)이면, 손실 함수는 log(P(action_chosen))를 최소화
        """

        if len(self.episode_rewards) == 0:
            return

        # 리턴 계산 (할인된 누적 보상)
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        # 정규화 (안정적인 학습을 위해)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 정책 경사 손실 계산
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            # Loss = -log(P(action)) * G
            # 여기서 G는 해당 행동의 리턴(할인된 누적 보상)
            policy_loss.append(-log_prob * G)

        # 전체 손실
        loss = torch.stack(policy_loss).sum()

        # 역전파 및 가중치 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 학습 기록
        self.training_history.append({
            'loss': loss.item(),
            'mean_reward': np.mean(self.episode_rewards),
            'timestamp': datetime.now().isoformat()
        })

        logger.info(f"📈 정책 업데이트 완료: Loss={loss.item():.4f}, "
                   f"평균 보상={np.mean(self.episode_rewards):.2f}")

        # 에피소드 버퍼 초기화
        self.episode_log_probs = []
        self.episode_rewards = []

    def evolve_with_feedback(self,
                            initial_dna: OlfactoryDNA,
                            brief: CreativeBrief,
                            num_iterations: int = 10) -> OlfactoryDNA:
        """
        사용자 피드백을 통한 진화 시뮬레이션
        실제로는 사용자 인터페이스와 연동되어야 함
        """

        logger.info("🧬 진화 엔진 시작: 사용자 피드백 기반 향수 진화")

        current_dna = initial_dna

        for iteration in range(num_iterations):
            logger.info(f"\n📍 진화 라운드 {iteration + 1}/{num_iterations}")

            # 변형 생성
            variations = self.generate_variations(current_dna, brief, num_variations=3)

            # 사용자 선택 시뮬레이션 (실제로는 UI에서 받아야 함)
            # 여기서는 랜덤하게 선택 (실제 구현시 사용자 입력 필요)
            selected_idx = random.randint(0, len(variations) - 1)

            # 정책 업데이트
            self.update_policy_with_feedback(variations, selected_idx)

            # 선택된 변형을 새로운 현재 DNA로 설정
            current_dna = variations[selected_idx].dna

        logger.info("✨ 진화 완료! 최종 향수 DNA 생성")

        return current_dna

    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
        logger.info(f"💾 모델 저장 완료: {path}")

    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"📂 모델 로드 완료: {path}")


def example_usage():
    """사용 예시"""

    # 초기 DNA 생성 (MOGA 엔진에서 가져올 수 있음)
    initial_dna = OlfactoryDNA(
        genes=[(1, 5.0), (3, 8.0), (5, 12.0), (7, 3.0), (9, 6.0)],
        fitness_scores=(0.8, 0.7, 0.9)
    )

    # 사용자 요구사항
    brief = CreativeBrief(
        emotional_palette=[0.4, 0.6, 0.2],  # 활기, 우아함, 따뜻함
        fragrance_family="oriental",
        mood="sophisticated",
        intensity=0.8,
        season="autumn",
        gender="unisex"
    )

    # 진화 엔진 초기화
    engine = EpigeneticVariationAI(
        state_dim=100,
        learning_rate=0.001,
        gamma=0.99
    )

    # 사용자 피드백 기반 진화 실행
    print("🧬 진화 엔진: 사용자 피드백 기반 향수 진화 시작...")
    evolved_dna = engine.evolve_with_feedback(
        initial_dna=initial_dna,
        brief=brief,
        num_iterations=5
    )

    print("\n✨ 진화 완료!")
    print(f"최종 DNA: {evolved_dna.genes[:5]}")  # 처음 5개 유전자만 출력
    print(f"학습 히스토리 길이: {len(engine.training_history)}")

    # 모델 저장
    engine.save_model("fragrance_rlhf_model.pth")


if __name__ == "__main__":
    example_usage()