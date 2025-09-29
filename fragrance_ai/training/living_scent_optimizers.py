"""
Living Scent Optimizers
살아있는 향수 AI를 위한 최적화 알고리즘 모음

1. AdamW: 뇌 신경망 훈련용 (LinguisticReceptor, CognitiveCore)
2. NSGA-III: 다목적 최적화 (OlfactoryRecombinator)
3. PPO-RLHF: 인간 피드백 강화학습 (EpigeneticVariation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)


# ============================================================================
# 1. AdamW Optimizer - 뇌 신경망 훈련용
# ============================================================================

@dataclass
class AdamWConfig:
    """AdamW 옵티마이저 설정"""
    learning_rate: float = 5e-5
    weight_decay: float = 0.01  # L2 정규화
    betas: Tuple[float, float] = (0.9, 0.999)  # 모멘텀 계수
    eps: float = 1e-8
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0  # Gradient clipping
    scheduler_type: str = "cosine"  # 'cosine', 'linear', 'constant'


class NeuralNetworkOptimizer:
    """
    신경망 훈련용 AdamW 옵티마이저
    LinguisticReceptorAI와 CognitiveCoreAI의 학습에 사용
    """

    def __init__(self, model: nn.Module, config: Optional[AdamWConfig] = None):
        self.model = model
        self.config = config or AdamWConfig()

        # AdamW 옵티마이저 초기화
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay
        )

        # 학습률 스케줄러 설정
        if self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warmup_steps,
                T_mult=2,
                eta_min=self.config.learning_rate * 0.01
            )
        else:
            self.scheduler = None

        self.global_step = 0
        self.training_loss_history = []

    def train_step(self, inputs: torch.Tensor, labels: torch.Tensor, loss_fn: Callable) -> float:
        """
        단일 훈련 스텝 실행

        Args:
            inputs: 입력 텐서
            labels: 정답 레이블
            loss_fn: 손실 함수

        Returns:
            loss: 현재 스텝의 손실값
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping (과도한 기울기 방지)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        # 옵티마이저 스텝
        self.optimizer.step()

        # 스케줄러 스텝
        if self.scheduler:
            self.scheduler.step()

        self.global_step += 1
        self.training_loss_history.append(loss.item())

        return loss.item()

    def get_current_lr(self) -> float:
        """현재 학습률 반환"""
        return self.optimizer.param_groups[0]['lr']

    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        logger.info(f"Checkpoint loaded from {path}")


# ============================================================================
# 2. NSGA-III - 다목적 최적화 (향의 DNA 창조용)
# ============================================================================

@dataclass
class Fragrance:
    """향수 개체 (유전 알고리즘용)"""
    genes: Dict[str, List[float]]  # 향료 유전자
    objectives: Dict[str, float] = None  # 목적 함수 값들
    rank: int = 0  # 파레토 순위
    crowding_distance: float = 0.0  # 밀집도 거리


class NSGAIII:
    """
    NSGA-III (Non-dominated Sorting Genetic Algorithm III)
    다목적 최적화를 위한 진화 알고리즘

    목적:
    1. 조화성 (Harmony): 향료들의 균형
    2. 독창성 (Uniqueness): 기존 향수와의 차별화
    3. 사용자 적합성 (User Fitness): 요구사항 충족도
    """

    def __init__(
        self,
        population_size: int = 100,
        num_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1
    ):
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        # 목적 함수들
        self.objectives = {
            'harmony': self._evaluate_harmony,
            'uniqueness': self._evaluate_uniqueness,
            'user_fitness': self._evaluate_user_fitness
        }

        # 진화 히스토리
        self.evolution_history = []
        self.pareto_front = []

    def _evaluate_harmony(self, fragrance: Fragrance) -> float:
        """
        조화성 평가: 향료들이 얼마나 잘 어울리는지
        """
        harmony_score = 0.0

        # Top, Middle, Base 노트의 균형
        for note_type, ingredients in fragrance.genes.items():
            if ingredients:
                # 농도의 표준편차가 작을수록 균형적
                std_dev = np.std(ingredients)
                harmony_score += 1.0 / (1.0 + std_dev)

        # 노트 간 전환의 부드러움
        if 'top' in fragrance.genes and 'middle' in fragrance.genes:
            transition_smoothness = 1.0 - abs(
                np.mean(fragrance.genes['top']) - np.mean(fragrance.genes['middle'])
            )
            harmony_score += transition_smoothness

        return harmony_score

    def _evaluate_uniqueness(self, fragrance: Fragrance) -> float:
        """
        독창성 평가: 기존 향수들과 얼마나 다른지
        """
        # 간단한 구현: 유전자의 엔트로피 계산
        all_values = []
        for ingredients in fragrance.genes.values():
            all_values.extend(ingredients)

        if not all_values:
            return 0.0

        # 엔트로피 계산
        hist, _ = np.histogram(all_values, bins=10)
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy

    def _evaluate_user_fitness(self, fragrance: Fragrance, user_requirements: Dict = None) -> float:
        """
        사용자 적합성 평가: 요구사항을 얼마나 충족하는지
        """
        if not user_requirements:
            # 기본 평가: 모든 노트가 존재하는지
            score = 0.0
            for note_type in ['top', 'middle', 'base']:
                if note_type in fragrance.genes and fragrance.genes[note_type]:
                    score += 1.0 / 3.0
            return score

        # 사용자 요구사항과의 매칭도 계산
        fitness = 0.0
        for requirement, weight in user_requirements.items():
            if requirement in fragrance.genes:
                fitness += weight
        return fitness

    def initialize_population(self) -> List[Fragrance]:
        """초기 개체군 생성"""
        population = []
        for _ in range(self.population_size):
            genes = {
                'top': [random.random() for _ in range(random.randint(2, 5))],
                'middle': [random.random() for _ in range(random.randint(3, 6))],
                'base': [random.random() for _ in range(random.randint(2, 4))]
            }
            population.append(Fragrance(genes=genes))
        return population

    def evaluate_objectives(self, population: List[Fragrance], user_requirements: Dict = None):
        """모든 개체의 목적 함수 평가"""
        for individual in population:
            individual.objectives = {
                'harmony': self._evaluate_harmony(individual),
                'uniqueness': self._evaluate_uniqueness(individual),
                'user_fitness': self._evaluate_user_fitness(individual, user_requirements)
            }

    def non_dominated_sort(self, population: List[Fragrance]) -> List[List[Fragrance]]:
        """
        비지배 정렬 (Non-dominated Sorting)
        파레토 프론트 찾기
        """
        fronts = [[]]
        dominated_count = {i: 0 for i in range(len(population))}
        dominates_list = {i: [] for i in range(len(population))}

        for i in range(len(population)):
            for j in range(len(population)):
                if i == j:
                    continue

                # i가 j를 지배하는지 확인
                if self._dominates(population[i], population[j]):
                    dominates_list[i].append(j)
                elif self._dominates(population[j], population[i]):
                    dominated_count[i] += 1

            if dominated_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        # 다음 프론트들 찾기
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for individual in fronts[current_front]:
                i = population.index(individual)
                for j in dominates_list[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        population[j].rank = current_front + 1
                        next_front.append(population[j])
            fronts.append(next_front)
            current_front += 1

        return fronts[:-1]  # 마지막 빈 리스트 제거

    def _dominates(self, ind1: Fragrance, ind2: Fragrance) -> bool:
        """ind1이 ind2를 지배하는지 확인"""
        better_in_all = True
        strictly_better_in_one = False

        for obj_name in ind1.objectives:
            val1 = ind1.objectives[obj_name]
            val2 = ind2.objectives[obj_name]

            if val1 < val2:
                better_in_all = False
            elif val1 > val2:
                strictly_better_in_one = True

        return better_in_all and strictly_better_in_one

    def crossover(self, parent1: Fragrance, parent2: Fragrance) -> Tuple[Fragrance, Fragrance]:
        """교차 연산"""
        if random.random() > self.crossover_prob:
            return parent1, parent2

        # 균일 교차 (Uniform Crossover)
        child1_genes = {}
        child2_genes = {}

        for note_type in parent1.genes:
            if random.random() < 0.5:
                child1_genes[note_type] = parent1.genes[note_type].copy()
                child2_genes[note_type] = parent2.genes.get(note_type, []).copy()
            else:
                child1_genes[note_type] = parent2.genes.get(note_type, []).copy()
                child2_genes[note_type] = parent1.genes[note_type].copy()

        return Fragrance(genes=child1_genes), Fragrance(genes=child2_genes)

    def mutate(self, individual: Fragrance):
        """돌연변이 연산"""
        if random.random() < self.mutation_prob:
            # 랜덤하게 선택된 유전자 변형
            note_type = random.choice(list(individual.genes.keys()))
            if individual.genes[note_type]:
                idx = random.randint(0, len(individual.genes[note_type]) - 1)
                individual.genes[note_type][idx] *= random.uniform(0.5, 1.5)
                individual.genes[note_type][idx] = min(1.0, max(0.0, individual.genes[note_type][idx]))

    def optimize(self, user_requirements: Dict = None) -> List[Fragrance]:
        """
        NSGA-III 최적화 실행

        Returns:
            pareto_front: 파레토 최적해 집합
        """
        # 초기 개체군
        population = self.initialize_population()

        for generation in range(self.num_generations):
            # 목적 함수 평가
            self.evaluate_objectives(population, user_requirements)

            # 비지배 정렬
            fronts = self.non_dominated_sort(population)

            # 자식 생성
            offspring = []
            while len(offspring) < self.population_size:
                # 토너먼트 선택
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                # 교차
                child1, child2 = self.crossover(parent1, parent2)

                # 돌연변이
                self.mutate(child1)
                self.mutate(child2)

                offspring.extend([child1, child2])

            # 목적 함수 평가
            self.evaluate_objectives(offspring, user_requirements)

            # 환경 선택
            combined = population + offspring
            fronts = self.non_dominated_sort(combined)

            # 다음 세대 선택
            next_population = []
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                else:
                    # 나머지는 밀집도 기반 선택
                    remaining = self.population_size - len(next_population)
                    self._calculate_crowding_distance(front)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    next_population.extend(front[:remaining])
                    break

            population = next_population

            # 진화 히스토리 저장
            self.evolution_history.append({
                'generation': generation,
                'best_harmony': max(ind.objectives['harmony'] for ind in population),
                'best_uniqueness': max(ind.objectives['uniqueness'] for ind in population),
                'best_user_fitness': max(ind.objectives['user_fitness'] for ind in population)
            })

            logger.info(f"Generation {generation}: Population size = {len(population)}")

        # 최종 파레토 프론트
        self.evaluate_objectives(population, user_requirements)
        fronts = self.non_dominated_sort(population)
        self.pareto_front = fronts[0] if fronts else []

        return self.pareto_front

    def _tournament_select(self, population: List[Fragrance], tournament_size: int = 3) -> Fragrance:
        """토너먼트 선택"""
        tournament = random.sample(population, tournament_size)
        return min(tournament, key=lambda x: x.rank)

    def _calculate_crowding_distance(self, front: List[Fragrance]):
        """밀집도 거리 계산"""
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return

        for ind in front:
            ind.crowding_distance = 0

        num_objectives = len(front[0].objectives)

        for obj_name in front[0].objectives:
            # 목적 함수별 정렬
            front.sort(key=lambda x: x.objectives[obj_name])

            # 경계값은 무한대
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            # 중간 개체들의 거리 계산
            obj_range = front[-1].objectives[obj_name] - front[0].objectives[obj_name]
            if obj_range == 0:
                continue

            for i in range(1, len(front) - 1):
                distance = front[i + 1].objectives[obj_name] - front[i - 1].objectives[obj_name]
                front[i].crowding_distance += distance / obj_range


# ============================================================================
# 3. PPO-RLHF - 인간 피드백 강화학습 (향의 진화용)
# ============================================================================

@dataclass
class Experience:
    """경험 데이터"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class PPO_RLHF:
    """
    PPO (Proximal Policy Optimization) with RLHF
    인간 피드백 기반 강화학습으로 향수 진화 학습
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clipping parameter
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Actor-Critic 네트워크
        self.actor_critic = self._build_actor_critic()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # 경험 버퍼
        self.experience_buffer = deque(maxlen=10000)

        # 인간 피드백 보상 모델
        self.reward_model = self._build_reward_model()
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)

        # 훈련 통계
        self.training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'human_feedback_accuracy': []
        }

    def _build_actor_critic(self) -> nn.Module:
        """Actor-Critic 네트워크 구성"""
        class ActorCritic(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                # 공유 레이어
                self.shared = nn.Sequential(
                    nn.Linear(state_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU()
                )
                # Actor (정책)
                self.actor = nn.Sequential(
                    nn.Linear(256, action_dim),
                    nn.Softmax(dim=-1)
                )
                # Critic (가치 함수)
                self.critic = nn.Linear(256, 1)

            def forward(self, state):
                features = self.shared(state)
                action_probs = self.actor(features)
                value = self.critic(features)
                return action_probs, value

        return ActorCritic(self.state_dim, self.action_dim)

    def _build_reward_model(self) -> nn.Module:
        """인간 피드백 기반 보상 모델"""
        class RewardModel(nn.Module):
            def __init__(self, state_dim, action_dim):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_dim + action_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)  # 보상 스칼라 출력
                )

            def forward(self, state, action):
                # 원-핫 인코딩
                action_one_hot = torch.zeros(action.size(0), action_dim)
                action_one_hot.scatter_(1, action.unsqueeze(1), 1)
                x = torch.cat([state, action_one_hot], dim=1)
                return self.network(x)

        return RewardModel(self.state_dim, self.action_dim)

    def collect_human_feedback(self, state: np.ndarray, action: int, feedback: float):
        """
        인간 피드백 수집

        Args:
            state: 현재 상태 (향수 표현)
            action: 선택된 행동 (변형 종류)
            feedback: 인간 피드백 (-1 ~ 1)
        """
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.LongTensor([action])

        # 보상 모델 훈련
        self.reward_optimizer.zero_grad()
        predicted_reward = self.reward_model(state_tensor.unsqueeze(0), action_tensor)
        loss = nn.MSELoss()(predicted_reward, torch.FloatTensor([[feedback]]))
        loss.backward()
        self.reward_optimizer.step()

        self.training_stats['human_feedback_accuracy'].append(loss.item())

    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """
        행동 선택 (확률적)

        Returns:
            action: 선택된 행동
            log_prob: 로그 확률
            value: 상태 가치
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action_probs, value = self.actor_critic(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def store_experience(self, experience: Experience):
        """경험 저장"""
        self.experience_buffer.append(experience)

    def train(self, batch_size: int = 64, epochs: int = 10):
        """
        PPO 훈련
        """
        if len(self.experience_buffer) < batch_size:
            return

        # 배치 샘플링
        batch = random.sample(self.experience_buffer, batch_size)

        # 텐서 변환
        states = torch.FloatTensor([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])
        next_states = torch.FloatTensor([e.next_state for e in batch])
        dones = torch.FloatTensor([e.done for e in batch])
        old_log_probs = torch.FloatTensor([e.log_prob for e in batch])
        old_values = torch.FloatTensor([e.value for e in batch])

        # Advantage 계산 (GAE)
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze()
            advantages = rewards + self.gamma * next_values * (1 - dones) - old_values
            returns = advantages + old_values

        # PPO 업데이트
        for _ in range(epochs):
            action_probs, values = self.actor_critic(states)
            values = values.squeeze()

            # 정책 손실
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 가치 손실
            value_loss = nn.MSELoss()(values, returns)

            # 엔트로피 보너스 (탐험 촉진)
            entropy = dist.entropy().mean()

            # 전체 손실
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # 역전파
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

            # 통계 기록
            self.training_stats['policy_losses'].append(policy_loss.item())
            self.training_stats['value_losses'].append(value_loss.item())

    def predict_reward(self, state: np.ndarray, action: int) -> float:
        """
        학습된 보상 모델로 보상 예측
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.LongTensor([action])

        with torch.no_grad():
            reward = self.reward_model(state_tensor, action_tensor)

        return reward.item()

    def save_models(self, path: str):
        """모델 저장"""
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        self.training_stats = checkpoint['training_stats']
        logger.info(f"Models loaded from {path}")


# ============================================================================
# 통합 옵티마이저 매니저
# ============================================================================

class LivingScentOptimizerManager:
    """
    Living Scent 시스템의 모든 옵티마이저 관리
    """

    def __init__(self):
        self.optimizers = {}
        self.configs = {}

    def register_neural_optimizer(self, name: str, model: nn.Module, config: AdamWConfig = None):
        """신경망 옵티마이저 등록"""
        self.optimizers[name] = NeuralNetworkOptimizer(model, config)
        self.configs[name] = config or AdamWConfig()
        logger.info(f"Neural optimizer '{name}' registered")

    def register_genetic_optimizer(self, name: str, **kwargs):
        """유전 알고리즘 옵티마이저 등록"""
        self.optimizers[name] = NSGAIII(**kwargs)
        self.configs[name] = kwargs
        logger.info(f"Genetic optimizer '{name}' registered")

    def register_rl_optimizer(self, name: str, state_dim: int, action_dim: int, **kwargs):
        """강화학습 옵티마이저 등록"""
        self.optimizers[name] = PPO_RLHF(state_dim, action_dim, **kwargs)
        self.configs[name] = kwargs
        logger.info(f"RL optimizer '{name}' registered")

    def get_optimizer(self, name: str):
        """옵티마이저 가져오기"""
        return self.optimizers.get(name)

    def train_all(self):
        """모든 옵티마이저 훈련"""
        for name, optimizer in self.optimizers.items():
            logger.info(f"Training optimizer: {name}")
            # 옵티마이저별 훈련 로직 실행
            if isinstance(optimizer, NeuralNetworkOptimizer):
                pass  # 신경망 훈련
            elif isinstance(optimizer, NSGAIII):
                pass  # 유전 알고리즘 실행
            elif isinstance(optimizer, PPO_RLHF):
                pass  # 강화학습 훈련

    def save_all(self, directory: str):
        """모든 옵티마이저 저장"""
        import os
        os.makedirs(directory, exist_ok=True)

        for name, optimizer in self.optimizers.items():
            path = os.path.join(directory, f"{name}.pt")
            if hasattr(optimizer, 'save_checkpoint'):
                optimizer.save_checkpoint(path)
            elif hasattr(optimizer, 'save_models'):
                optimizer.save_models(path)
            logger.info(f"Saved optimizer '{name}' to {path}")


# 싱글톤 인스턴스
_optimizer_manager_instance = None

def get_optimizer_manager() -> LivingScentOptimizerManager:
    """싱글톤 옵티마이저 매니저 반환"""
    global _optimizer_manager_instance
    if _optimizer_manager_instance is None:
        _optimizer_manager_instance = LivingScentOptimizerManager()
    return _optimizer_manager_instance