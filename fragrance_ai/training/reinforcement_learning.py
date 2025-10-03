# fragrance_ai/training/reinforcement_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from typing import List, Dict, Any

# 프로젝트 내부 모듈 임포트
from fragrance_ai.database.models import OlfactoryDNA, ScentPhenotype
from fragrance_ai.services.orchestrator_service import get_creative_brief  # 가정: creative_brief를 벡터로 변환하는 함수가 필요

# 1. 정책 신경망(Policy Network) 정의
# 이 신경망은 '현재 상태'를 입력받아 '어떤 변형을 할지'에 대한 확률을 출력합니다.
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.network(state)

class RLEngine:
    def __init__(self, state_dim: int, action_dim: int, learning_rate: float = 0.001):
        """
        강화학습 엔진 초기화.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=learning_rate)

        # 가능한 변형 행동(Action) 목록 정의 (예시)
        # 실제 구현에서는 이 목록을 동적으로 관리해야 합니다.
        self.action_space = [
            "amplify_base_note_1", "silence_top_note_1", "add_new_note_rose",
            "add_new_note_vanilla", "shift_to_warm", "shift_to_fresh"
        ]

    def encode_state(self, dna: OlfactoryDNA, creative_brief: dict) -> torch.Tensor:
        """
        DNA와 creative_brief를 정책 신경망이 이해할 수 있는 벡터로 변환합니다.
        """
        # DNA의 특징 추출 (예: 상위 5개 노트의 intensity)
        dna_features = []
        for i in range(5):
            if i < len(dna.notes):
                dna_features.append(dna.notes[i]["intensity"])
            else:
                dna_features.append(0.0)

        # Creative Brief의 특징 추출
        brief_features = [
            creative_brief.get("desired_intensity", 0.5),
            creative_brief.get("masculinity", 0.5),
            creative_brief.get("complexity", 0.5)
        ]

        # 결합하여 상태 벡터 생성
        state_vector = dna_features + brief_features

        # 필요시 패딩
        size = self.state_dim
        padded_vector = state_vector[:size] if len(state_vector) >= size else state_vector + [0.0] * (size - len(state_vector))

        return torch.FloatTensor(padded_vector).unsqueeze(0)

    def generate_variations(self, dna: OlfactoryDNA, feedback_brief: dict, num_options: int = 3) -> List[Dict]:
        """
        정책 신경망을 사용하여 여러 개의 변형(Phenotype) 후보를 생성합니다.
        """
        # 1. 현재 상태를 신경망이 이해할 수 있는 벡터로 변환
        state = self.encode_state(dna, feedback_brief)

        # 2. 정책 신경망으로 행동 확률 분포 계산
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)

        options = []
        saved_actions = []

        # 3. 확률에 따라 여러 개의 행동을 샘플링하고 변형 후보 생성
        for i in range(num_options):
            # 행동 샘플링
            action = dist.sample()
            action_name = self.action_space[action.item()]

            # --- PLACEHOLDER: 이 부분은 실제 레시피를 변형하는 정교한 로직이 필요 ---
            # 지금은 행동 이름에 따라 간단한 설명을 생성하는 것으로 시뮬레이션합니다.
            new_recipe = dna.genotype.copy()
            description = f"Variation based on '{action_name}'. {feedback_brief.get('story', '')}"

            phenotype = ScentPhenotype(
                phenotype_id=f"pheno_{random.randint(1000, 9999)}_{i}",
                based_on_dna=dna.dna_id,
                epigenetic_trigger=feedback_brief.get('theme', 'N/A'),
                variation_applied=action_name,
                recipe_adjusted=new_recipe,
                description=description
            )

            options.append({
                "id": phenotype.phenotype_id,
                "phenotype": phenotype,
                "action": action.item(),
                "action_name": action_name,
                "log_prob": dist.log_prob(action)
            })

            saved_actions.append((action, dist.log_prob(action)))

        # 나중에 학습할 때 사용하기 위해 상태와 행동 정보를 저장
        self.last_state = state
        self.last_saved_actions = saved_actions

        return options

    def update_policy_with_feedback(self, chosen_phenotype_id: str, options: List[Dict], state: torch.Tensor, saved_actions: List[tuple]):
        """
        사용자의 선택을 보상으로 삼아 정책 신경망을 업데이트(학습)합니다.
        """
        # 이 부분은 Step 3.3에서 구현할 예정
        pass

    def save_model(self, path="policy_network.pth"):
        torch.save(self.policy_network.state_dict(), path)

    def load_model(self, path="policy_network.pth"):
        self.policy_network.load_state_dict(torch.load(path))