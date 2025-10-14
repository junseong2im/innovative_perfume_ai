"""
MOGA ↔ RL Hybrid Loop
Exploration(LLM/MOGA)과 Exploitation(RL) 루프를 자동으로 전환
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import torch


class Mode(Enum):
    """하이브리드 모드"""
    EXPLORATION = "exploration"  # MOGA/LLM 기반 탐색
    EXPLOITATION = "exploitation"  # RL 기반 최적화


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    avg_reward: float
    reward_std: float
    diversity_score: float
    convergence_rate: float
    episode_count: int


class HybridController:
    """
    MOGA-RL 하이브리드 컨트롤러
    탐색과 활용을 자동으로 전환
    """

    def __init__(
        self,
        exploration_budget: float = 0.3,  # 30% exploration
        diversity_threshold: float = 0.6,
        plateau_threshold: int = 50,  # 50 에피소드 동안 개선 없으면 plateau
        reward_improvement_threshold: float = 0.05
    ):
        self.exploration_budget = exploration_budget
        self.diversity_threshold = diversity_threshold
        self.plateau_threshold = plateau_threshold
        self.reward_improvement_threshold = reward_improvement_threshold

        # 상태 추적
        self.current_mode = Mode.EXPLORATION
        self.episode_count = 0
        self.exploration_episodes = 0
        self.exploitation_episodes = 0

        # 성능 히스토리
        self.reward_history: List[float] = []
        self.diversity_history: List[float] = []
        self.last_best_reward = -np.inf
        self.episodes_since_improvement = 0

        logger.info(f"HybridController initialized (exploration_budget={exploration_budget})")

    def update_metrics(
        self,
        reward: float,
        diversity: float
    ):
        """
        메트릭 업데이트

        Args:
            reward: 현재 에피소드 보상
            diversity: 현재 다양성 스코어
        """
        self.episode_count += 1
        self.reward_history.append(reward)
        self.diversity_history.append(diversity)

        # Exploration/Exploitation 카운트
        if self.current_mode == Mode.EXPLORATION:
            self.exploration_episodes += 1
        else:
            self.exploitation_episodes += 1

        # 개선 추적
        if reward > self.last_best_reward + self.reward_improvement_threshold:
            self.last_best_reward = reward
            self.episodes_since_improvement = 0
        else:
            self.episodes_since_improvement += 1

    def get_performance_metrics(self, window: int = 100) -> PerformanceMetrics:
        """
        최근 성능 메트릭 계산

        Args:
            window: 평균 계산 윈도우 크기

        Returns:
            PerformanceMetrics object
        """
        recent_rewards = self.reward_history[-window:]
        recent_diversity = self.diversity_history[-window:]

        if not recent_rewards:
            return PerformanceMetrics(
                avg_reward=0.0,
                reward_std=0.0,
                diversity_score=0.0,
                convergence_rate=0.0,
                episode_count=self.episode_count
            )

        avg_reward = np.mean(recent_rewards)
        reward_std = np.std(recent_rewards)
        diversity_score = np.mean(recent_diversity)

        # Convergence rate: 최근 reward의 선형 회귀 기울기
        if len(recent_rewards) > 10:
            x = np.arange(len(recent_rewards))
            convergence_rate = np.polyfit(x, recent_rewards, 1)[0]
        else:
            convergence_rate = 0.0

        return PerformanceMetrics(
            avg_reward=avg_reward,
            reward_std=reward_std,
            diversity_score=diversity_score,
            convergence_rate=convergence_rate,
            episode_count=self.episode_count
        )

    def should_explore(self) -> bool:
        """
        탐색 모드로 전환해야 하는지 판단

        Returns:
            True if should explore
        """
        metrics = self.get_performance_metrics()

        # 1. 초기 학습 단계 (무조건 탐색)
        if self.episode_count < 1000:
            logger.debug("Early stage: explore")
            return True

        # 2. Exploration budget 초과 시 더 이상 탐색하지 않음 (최우선)
        exploration_ratio = self.exploration_episodes / self.episode_count
        if exploration_ratio >= self.exploration_budget:
            logger.debug(f"Exploration budget met ({exploration_ratio:.2f}): no explore")
            return False

        # 3. Exploration budget 미달 시 탐색
        if exploration_ratio < self.exploration_budget * 0.8:  # 80% 미만
            logger.debug(f"Exploration ratio low ({exploration_ratio:.2f}): explore")
            return True

        # 4. 다양성이 낮을 때
        if metrics.diversity_score < self.diversity_threshold:
            logger.debug(f"Low diversity ({metrics.diversity_score:.2f}): explore")
            return True

        # 5. 성능 plateau (단, exploration budget 내에서만)
        if self.episodes_since_improvement > self.plateau_threshold:
            logger.debug(f"Plateau detected ({self.episodes_since_improvement} eps): explore")
            return True

        # 6. Reward 표준편차가 낮을 때 (수렴 징후)
        if metrics.reward_std < 0.5:
            logger.debug(f"Low variance ({metrics.reward_std:.2f}): explore")
            return True

        return False

    def should_exploit(self) -> bool:
        """
        활용 모드로 전환해야 하는지 판단

        Returns:
            True if should exploit
        """
        metrics = self.get_performance_metrics()

        # 1. Exploration budget 초과
        exploration_ratio = self.exploration_episodes / self.episode_count
        if exploration_ratio > self.exploration_budget:
            logger.debug(f"Exploration budget exceeded ({exploration_ratio:.2f}): exploit")
            return True

        # 2. 다양성이 충분할 때
        if metrics.diversity_score > self.diversity_threshold:
            logger.debug(f"High diversity ({metrics.diversity_score:.2f}): exploit")
            return True

        # 3. Reward가 꾸준히 증가 중
        if metrics.convergence_rate > 0.01:
            logger.debug(f"Improving ({metrics.convergence_rate:.4f}): exploit")
            return True

        # 4. Reward 표준편차가 높을 때 (발산 방지)
        if metrics.reward_std > 2.0:
            logger.debug(f"High variance ({metrics.reward_std:.2f}): exploit")
            return True

        return False

    def decide_mode(self) -> Mode:
        """
        현재 상황에 맞는 모드 결정

        Returns:
            Mode.EXPLORATION or Mode.EXPLOITATION
        """
        # Exploration 우선 체크
        if self.should_explore():
            new_mode = Mode.EXPLORATION
        elif self.should_exploit():
            new_mode = Mode.EXPLOITATION
        else:
            # 현재 모드 유지
            new_mode = self.current_mode

        # 모드 변경 로깅
        if new_mode != self.current_mode:
            metrics = self.get_performance_metrics()
            logger.info(
                f"Mode switch: {self.current_mode.value} → {new_mode.value} | "
                f"Reward: {metrics.avg_reward:.2f}±{metrics.reward_std:.2f} | "
                f"Diversity: {metrics.diversity_score:.2f} | "
                f"Episode: {self.episode_count}"
            )
            self.current_mode = new_mode

        return self.current_mode

    def get_epsilon(self) -> float:
        """
        ε-greedy exploration rate 계산

        Returns:
            epsilon value (0.0 ~ 1.0)
        """
        if self.current_mode == Mode.EXPLORATION:
            # Exploration 모드: 높은 epsilon
            base_epsilon = 0.5
        else:
            # Exploitation 모드: 낮은 epsilon
            base_epsilon = 0.1

        # Episode에 따라 감소
        decay_factor = max(0.1, 1.0 - self.episode_count / 10000)
        epsilon = base_epsilon * decay_factor

        return epsilon


class HybridEvolutionEngine:
    """
    MOGA-RL 하이브리드 진화 엔진
    """

    def __init__(self, controller: HybridController):
        self.controller = controller
        logger.info("HybridEvolutionEngine initialized")

    def run_exploration_step(self, population: List) -> Tuple[List, Dict]:
        """
        탐색 스텝 (MOGA/LLM)

        Args:
            population: 현재 개체군

        Returns:
            (new_population, metrics)
        """
        logger.debug("Running EXPLORATION step (MOGA/LLM)")

        # MOGA 연산 (Genetic Algorithm)
        # 1. Selection
        selected = self._tournament_selection(population, k=len(population) // 2)

        # 2. Crossover
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                child1, child2 = self._crossover(selected[i], selected[i + 1])
                offspring.extend([child1, child2])

        # 3. Mutation (높은 mutation rate)
        mutation_rate = 0.3  # Exploration 모드는 높은 mutation
        mutated = [self._mutate(ind, mutation_rate) for ind in offspring]

        # 4. LLM augmentation (선택적으로 LLM으로 새로운 개체 생성)
        if np.random.rand() < 0.2:  # 20% 확률
            llm_generated = self._generate_with_llm()
            mutated.append(llm_generated)

        metrics = {
            "mode": "exploration",
            "population_size": len(mutated),
            "diversity": self._calculate_diversity(mutated)
        }

        return mutated, metrics

    def run_exploitation_step(self, population: List, policy) -> Tuple[List, Dict]:
        """
        활용 스텝 (RL)

        Args:
            population: 현재 개체군
            policy: RL 정책 (PPO)

        Returns:
            (new_population, metrics)
        """
        logger.debug("Running EXPLOITATION step (RL/PPO)")

        # RL 기반 최적화
        # 1. Policy로부터 action 샘플링
        actions = []
        for individual in population:
            state = self._individual_to_state(individual)
            action = policy.select_action(state)
            actions.append(action)

        # 2. Action 적용하여 새로운 개체 생성
        new_population = [
            self._apply_action(ind, act)
            for ind, act in zip(population, actions)
        ]

        # 3. Mutation (낮은 mutation rate)
        mutation_rate = 0.05  # Exploitation 모드는 낮은 mutation
        refined = [self._mutate(ind, mutation_rate) for ind in new_population]

        metrics = {
            "mode": "exploitation",
            "population_size": len(refined),
            "avg_value": np.mean([policy.estimate_value(self._individual_to_state(ind)) for ind in refined])
        }

        return refined, metrics

    # Helper methods (간단한 placeholder 구현)
    def _tournament_selection(self, population: List, k: int) -> List:
        """토너먼트 선택"""
        return np.random.choice(population, size=k, replace=False).tolist()

    def _crossover(self, parent1, parent2):
        """교차"""
        return parent1, parent2  # Placeholder

    def _mutate(self, individual, rate: float):
        """돌연변이"""
        return individual  # Placeholder

    def _generate_with_llm(self):
        """LLM으로 새로운 개체 생성"""
        return {}  # Placeholder

    def _calculate_diversity(self, population: List) -> float:
        """다양성 계산"""
        return np.random.rand()  # Placeholder

    def _individual_to_state(self, individual):
        """개체를 RL 상태로 변환"""
        return torch.zeros(10)  # Placeholder

    def _apply_action(self, individual, action):
        """Action 적용"""
        return individual  # Placeholder


# 사용 예시
if __name__ == "__main__":
    controller = HybridController()
    engine = HybridEvolutionEngine(controller)

    # 시뮬레이션
    population = [{"id": i} for i in range(100)]

    for episode in range(1000):
        # 메트릭 업데이트 (랜덤 시뮬레이션)
        reward = np.random.rand() * 10 + episode * 0.01
        diversity = np.random.rand()
        controller.update_metrics(reward, diversity)

        # 모드 결정
        mode = controller.decide_mode()

        # 해당 모드로 진화 스텝 실행
        if mode == Mode.EXPLORATION:
            population, metrics = engine.run_exploration_step(population)
        else:
            # Exploitation (실제로는 policy 필요)
            population, metrics = engine.run_exploitation_step(population, policy=None)

        if episode % 100 == 0:
            perf = controller.get_performance_metrics()
            print(f"Episode {episode} | Mode: {mode.value} | "
                  f"Reward: {perf.avg_reward:.2f}±{perf.reward_std:.2f} | "
                  f"Diversity: {perf.diversity_score:.2f}")
