# fragrance_ai/training/reinforcement_learning_ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from typing import List, Dict, Any, Tuple
from collections import deque

# 프로젝트 내부 모듈 임포트
from fragrance_ai.database.models import OlfactoryDNA, ScentPhenotype
from fragrance_ai.training.reinforcement_learning import PolicyNetwork


class ValueNetwork(nn.Module):
    """PPO의 Critic 네트워크 - 상태의 가치를 예측"""
    def __init__(self, state_dim: int):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)


class PPOMemory:
    """PPO 학습을 위한 경험 버퍼"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def get_batch(self):
        return (
            torch.cat(self.states),
            torch.cat(self.actions),
            torch.cat(self.log_probs),
            torch.tensor(self.rewards, dtype=torch.float32),
            torch.cat(self.values),
            torch.tensor(self.dones, dtype=torch.float32)
        )


class RLEnginePPO:
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 algorithm: str = "PPO"):
        """
        PPO 강화학습 엔진 초기화.

        Args:
            state_dim: 상태 공간 차원
            action_dim: 행동 공간 차원
            learning_rate: 학습률
            gamma: 할인 계수
            epsilon: PPO 클리핑 파라미터
            value_loss_coef: 가치 함수 손실 계수
            entropy_coef: 엔트로피 보너스 계수
            max_grad_norm: 그래디언트 클리핑 값
            algorithm: "PPO" 또는 "REINFORCE"
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm

        # 하이퍼파라미터
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # 네트워크 초기화
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim) if algorithm == "PPO" else None

        # 옵티마이저 설정
        if algorithm == "PPO":
            params = list(self.policy_network.parameters()) + list(self.value_network.parameters())
            self.optimizer = optim.Adam(params, lr=learning_rate)
        else:
            self.optimizer = optim.AdamW(self.policy_network.parameters(), lr=learning_rate)

        # 메모리 버퍼
        self.memory = PPOMemory() if algorithm == "PPO" else None

        # 행동 공간 정의
        self.action_space = [
            "amplify_base_note", "silence_top_note", "add_rose_accord",
            "add_vanilla_accord", "shift_warm_tonality", "shift_fresh_tonality",
            "increase_complexity", "simplify_structure", "enhance_longevity",
            "boost_sillage", "add_woody_base", "add_citrus_top"
        ]

        # 상태/행동 추적용
        self.last_state = None
        self.last_saved_actions = []
        self.last_values = [] if algorithm == "PPO" else None

    def encode_state(self, dna: OlfactoryDNA, creative_brief: dict) -> torch.Tensor:
        """
        DNA와 creative_brief를 정책 신경망이 이해할 수 있는 벡터로 변환합니다.
        """
        # DNA의 특징 추출 (상위 노트들의 intensity)
        dna_features = []
        for i in range(min(10, len(dna.notes))):  # 상위 10개 노트
            if i < len(dna.notes):
                dna_features.append(dna.notes[i].get("intensity", 0.0))
            else:
                dna_features.append(0.0)

        # Creative Brief의 특징 추출
        brief_features = [
            creative_brief.get("desired_intensity", 0.5),
            creative_brief.get("masculinity", 0.5),
            creative_brief.get("complexity", 0.5),
            creative_brief.get("longevity", 0.5),
            creative_brief.get("sillage", 0.5),
            creative_brief.get("warmth", 0.5),
            creative_brief.get("freshness", 0.5),
            creative_brief.get("sweetness", 0.5)
        ]

        # 결합하여 상태 벡터 생성
        state_vector = dna_features + brief_features

        # 패딩 또는 자르기로 크기 맞추기
        if len(state_vector) < self.state_dim:
            state_vector += [0.0] * (self.state_dim - len(state_vector))
        elif len(state_vector) > self.state_dim:
            state_vector = state_vector[:self.state_dim]

        return torch.FloatTensor(state_vector).unsqueeze(0)

    def generate_variations(self, dna: OlfactoryDNA, feedback_brief: dict, num_options: int = 3) -> List[Dict]:
        """
        정책 신경망을 사용하여 여러 개의 변형(Phenotype) 후보를 생성합니다.
        """
        # 1. 현재 상태를 신경망이 이해할 수 있는 벡터로 변환
        state = self.encode_state(dna, feedback_brief)

        # 2. 정책 신경망으로 행동 확률 분포 계산
        action_probs = self.policy_network(state)
        dist = Categorical(action_probs)

        # 3. PPO의 경우 가치 예측
        values = []
        if self.algorithm == "PPO" and self.value_network:
            value = self.value_network(state)
            values = [value] * num_options  # 같은 상태에 대한 가치

        options = []
        saved_actions = []

        # 4. 확률에 따라 여러 개의 행동을 샘플링하고 변형 후보 생성
        for i in range(num_options):
            # 행동 샘플링
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_idx = action.item()
            action_name = self.action_space[action_idx] if action_idx < len(self.action_space) else f"action_{action_idx}"

            # 레시피 변형 시뮬레이션
            new_recipe = self._apply_variation(dna.genotype.copy(), action_name)

            # 변형 설명 생성
            description = self._generate_description(action_name, feedback_brief)

            phenotype = ScentPhenotype(
                phenotype_id=f"pheno_{random.randint(1000, 9999)}_{i}",
                based_on_dna=dna.dna_id,
                epigenetic_trigger=feedback_brief.get('theme', 'evolution'),
                variation_applied=action_name,
                recipe_adjusted=new_recipe,
                description=description
            )

            options.append({
                "id": phenotype.phenotype_id,
                "phenotype": phenotype,
                "action": action_idx,
                "action_name": action_name,
                "log_prob": log_prob
            })

            saved_actions.append((action, log_prob))

        # 나중에 학습할 때 사용하기 위해 상태와 행동 정보를 저장
        self.last_state = state
        self.last_saved_actions = saved_actions
        if self.algorithm == "PPO":
            self.last_values = values

        return options

    def _apply_variation(self, recipe: dict, action_name: str) -> dict:
        """행동에 따른 레시피 변형 적용"""
        # 실제 변형 로직 구현 (예시)
        if "amplify" in action_name:
            # 베이스 노트 증폭
            if "base_notes" in recipe:
                for note in recipe["base_notes"]:
                    note["concentration"] = min(note.get("concentration", 1.0) * 1.2, 10.0)
        elif "silence" in action_name:
            # 탑 노트 감소
            if "top_notes" in recipe:
                for note in recipe["top_notes"]:
                    note["concentration"] = max(note.get("concentration", 1.0) * 0.8, 0.1)
        elif "add_rose" in action_name:
            # 장미 어코드 추가
            if "heart_notes" not in recipe:
                recipe["heart_notes"] = []
            recipe["heart_notes"].append({"ingredient": "Rose Absolute", "concentration": 2.0})
        # ... 더 많은 변형 로직

        return recipe

    def _generate_description(self, action_name: str, feedback_brief: dict) -> str:
        """변형에 대한 설명 생성"""
        theme = feedback_brief.get('theme', 'evolution')
        story = feedback_brief.get('story', '')

        descriptions = {
            "amplify_base_note": f"Enhanced foundation with deeper base notes for {theme}",
            "silence_top_note": f"Softened opening with subtle top notes reflecting {theme}",
            "add_rose_accord": f"Romantic rose accord inspired by {theme}",
            "add_vanilla_accord": f"Warm vanilla embrace evoking {theme}",
            "shift_warm_tonality": f"Warmer interpretation of {theme}",
            "shift_fresh_tonality": f"Fresh perspective on {theme}",
            # ... 더 많은 설명
        }

        base_desc = descriptions.get(action_name, f"Creative variation based on {action_name}")
        return f"{base_desc}. {story}"

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Generalized Advantage Estimation (GAE) 계산"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * 0.95 * (1 - dones[t]) * last_advantage

        return advantages

    def update_policy_with_feedback_ppo(self, chosen_phenotype_id: str, options: List[Dict],
                                        state: torch.Tensor, saved_actions: List[tuple],
                                        values: List[torch.Tensor] = None, rating: float = None):
        """
        PPO 알고리즘을 사용한 정책 업데이트
        """
        # 1) 보상 설계
        if rating is not None:
            base_reward = (rating - 3) / 2.0  # -1.0 ~ +1.0 범위
        else:
            base_reward = 1.0 if any(opt["id"] == chosen_phenotype_id for opt in options) else 0.0

        # 메모리에 경험 저장
        for i, (action, log_prob) in enumerate(saved_actions):
            # 선택된 옵션에 더 높은 보상
            reward = base_reward if options[i]["id"] == chosen_phenotype_id else base_reward * 0.1
            value = values[i] if values else torch.tensor([0.5])
            done = 1.0 if i == len(saved_actions) - 1 else 0.0

            self.memory.add(state, action, log_prob, reward, value, done)

        # 배치 데이터 가져오기
        if len(self.memory.states) < 4:  # 최소 배치 크기
            return {"status": "buffering", "buffer_size": len(self.memory.states)}

        states, actions, old_log_probs, rewards, values, dones = self.memory.get_batch()

        # GAE 계산
        advantages = self.compute_gae(rewards, values.squeeze(), dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 업데이트 (여러 에포크)
        total_loss = 0
        for _ in range(4):  # PPO epochs
            # 현재 정책으로 log_prob 계산
            action_probs = self.policy_network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy().mean()

            # 현재 가치 예측
            new_values = self.value_network(states)

            # PPO 클리핑된 목적 함수
            ratio = (new_log_probs - old_log_probs.squeeze()).exp()
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            # 가치 함수 손실
            value_loss = nn.functional.mse_loss(new_values.squeeze(), rewards + self.gamma * values.squeeze())

            # 총 손실 = 정책 손실 + 가치 손실 - 엔트로피 보너스
            loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

            # 역전파 및 그래디언트 클리핑
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy_network.parameters()) + list(self.value_network.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()

        # 메모리 초기화
        self.memory.clear()

        return {
            "loss": total_loss / 4,
            "reward": float(base_reward),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "algorithm": "PPO"
        }

    def update_policy_with_feedback(self, chosen_phenotype_id: str, options: List[Dict],
                                   state: torch.Tensor = None, saved_actions: List[tuple] = None,
                                   rating: float = None):
        """
        사용자의 선택을 보상으로 삼아 정책 신경망을 업데이트합니다.
        알고리즘 선택에 따라 REINFORCE 또는 PPO를 사용합니다.
        """
        # 저장된 상태와 행동 사용 (인자로 전달되지 않은 경우)
        state = state if state is not None else self.last_state
        saved_actions = saved_actions if saved_actions is not None else self.last_saved_actions

        if self.algorithm == "PPO":
            return self.update_policy_with_feedback_ppo(
                chosen_phenotype_id, options, state, saved_actions,
                self.last_values, rating
            )
        else:
            # REINFORCE 알고리즘 사용
            return self.update_policy_with_feedback_reinforce(
                chosen_phenotype_id, options, state, saved_actions, rating
            )

    def update_policy_with_feedback_reinforce(self, chosen_phenotype_id: str, options: List[Dict],
                                             state: torch.Tensor, saved_actions: List[tuple],
                                             rating: float = None):
        """
        REINFORCE 알고리즘을 사용한 간단한 정책 업데이트
        """
        # 1) 보상 설계
        if rating is not None:
            reward = (rating - 3) / 2.0  # -1.0 ~ +1.0 범위
        else:
            reward = 1.0 if any(opt["id"] == chosen_phenotype_id for opt in options) else 0.0

        # 2) REINFORCE loss 계산
        loss = 0.0
        for (action, log_prob) in saved_actions:
            loss += -log_prob * reward

        # 3) 역전파 및 파라미터 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "reward": float(reward),
            "algorithm": "REINFORCE"
        }

    def save_model(self, path_prefix="rl_model"):
        """모델 저장"""
        torch.save(self.policy_network.state_dict(), f"{path_prefix}_policy.pth")
        if self.value_network:
            torch.save(self.value_network.state_dict(), f"{path_prefix}_value.pth")

    def load_model(self, path_prefix="rl_model"):
        """모델 로드"""
        self.policy_network.load_state_dict(torch.load(f"{path_prefix}_policy.pth"))
        if self.value_network:
            self.value_network.load_state_dict(torch.load(f"{path_prefix}_value.pth"))