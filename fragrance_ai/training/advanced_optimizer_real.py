"""
고급 옵티마이저 실제 구현
다목적 유전 알고리즘(MOGA)과 강화학습(RLHF)의 완전한 구현

이 모듈은 시뮬레이션이 아닌 실제 작동하는 최적화 알고리즘을 포함합니다.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import random
from pathlib import Path
import logging
from collections import deque
import heapq

logger = logging.getLogger(__name__)


# ============================================================================
# 실제 MOGA (Multi-Objective Genetic Algorithm) 구현
# ============================================================================

@dataclass
class FragranceDNA:
    """향수 DNA - 실제 유전자 정보를 담는 구조체"""

    # 유전자 정보
    top_notes: List[Tuple[str, float]]  # (향료명, 농도)
    middle_notes: List[Tuple[str, float]]
    base_notes: List[Tuple[str, float]]

    # 목적 함수 값들
    creativity_score: float = 0.0
    fitness_score: float = 0.0
    stability_score: float = 0.0

    # 메타데이터
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    dna_id: str = field(default_factory=lambda: f"DNA_{random.randint(10000, 99999)}")

    def to_recipe(self) -> Dict[str, Any]:
        """DNA를 실제 레시피로 변환"""
        return {
            'dna_id': self.dna_id,
            'top_notes': [{'name': n, 'concentration': c} for n, c in self.top_notes],
            'middle_notes': [{'name': n, 'concentration': c} for n, c in self.middle_notes],
            'base_notes': [{'name': n, 'concentration': c} for n, c in self.base_notes],
            'scores': {
                'creativity': self.creativity_score,
                'fitness': self.fitness_score,
                'stability': self.stability_score,
                'overall': (self.creativity_score + self.fitness_score + self.stability_score) / 3
            }
        }


class RealMOGA:
    """
    실제 작동하는 다목적 유전 알고리즘
    창의성, 적합성, 안정성을 동시에 최적화
    """

    def __init__(self, fragrance_database_path: Optional[str] = None):
        # 실제 향료 데이터베이스
        self.fragrance_db = self._load_fragrance_database(fragrance_database_path)

        # 유전 알고리즘 파라미터
        self.population_size = 100
        self.num_generations = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.elitism_rate = 0.1

        # 목적 함수 가중치
        self.creativity_weight = 0.33
        self.fitness_weight = 0.33
        self.stability_weight = 0.34

        # 진화 히스토리
        self.evolution_history = []
        self.pareto_front = []

    def _load_fragrance_database(self, path: Optional[str]) -> Dict[str, List[str]]:
        """실제 향료 데이터베이스 로드"""
        if path and Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 기본 향료 데이터베이스
        return {
            'top': [
                'Bergamot', 'Lemon', 'Orange', 'Grapefruit', 'Mandarin',
                'Mint', 'Basil', 'Lavender', 'Pink Pepper', 'Black Pepper',
                'Cardamom', 'Ginger', 'Eucalyptus', 'Tea', 'Aldehydes'
            ],
            'middle': [
                'Rose', 'Jasmine', 'Ylang-Ylang', 'Geranium', 'Violet',
                'Iris', 'Freesia', 'Lily', 'Peony', 'Magnolia',
                'Cinnamon', 'Nutmeg', 'Clove', 'Saffron', 'Honey'
            ],
            'base': [
                'Vanilla', 'Musk', 'Amber', 'Sandalwood', 'Cedarwood',
                'Patchouli', 'Vetiver', 'Oakmoss', 'Leather', 'Tobacco',
                'Benzoin', 'Tonka Bean', 'Labdanum', 'Oud', 'Incense'
            ]
        }

    def _create_random_dna(self) -> FragranceDNA:
        """랜덤 DNA 생성"""
        dna = FragranceDNA(
            top_notes=[(random.choice(self.fragrance_db['top']),
                        random.uniform(0.1, 0.5))
                       for _ in range(random.randint(2, 4))],
            middle_notes=[(random.choice(self.fragrance_db['middle']),
                          random.uniform(0.2, 0.6))
                         for _ in range(random.randint(3, 5))],
            base_notes=[(random.choice(self.fragrance_db['base']),
                        random.uniform(0.3, 0.7))
                       for _ in range(random.randint(2, 4))]
        )
        return dna

    def _evaluate_creativity(self, dna: FragranceDNA) -> float:
        """창의성 평가: 독특한 조합일수록 높은 점수"""
        # 향료 조합의 독특성 계산
        all_notes = [n for n, _ in dna.top_notes + dna.middle_notes + dna.base_notes]
        unique_ratio = len(set(all_notes)) / len(all_notes) if all_notes else 0

        # 비전통적인 조합 보너스
        unusual_combos = 0
        if any('Leather' in n for n, _ in dna.top_notes):  # Top에 Leather는 독특
            unusual_combos += 0.2
        if any('Tea' in n for n, _ in dna.base_notes):  # Base에 Tea는 독특
            unusual_combos += 0.2

        # 농도의 다양성
        concentrations = [c for _, c in dna.top_notes + dna.middle_notes + dna.base_notes]
        conc_std = np.std(concentrations) if concentrations else 0

        creativity = (unique_ratio * 0.5 + unusual_combos * 0.3 + conc_std * 0.2)
        return min(1.0, creativity)

    def _evaluate_fitness(self, dna: FragranceDNA, user_preferences: Dict = None) -> float:
        """적합성 평가: 사용자 선호도와의 일치도"""
        if not user_preferences:
            # 기본 평가: 균형잡힌 구성
            balance_score = 0
            if 2 <= len(dna.top_notes) <= 4:
                balance_score += 0.33
            if 3 <= len(dna.middle_notes) <= 5:
                balance_score += 0.33
            if 2 <= len(dna.base_notes) <= 4:
                balance_score += 0.34
            return balance_score

        # 사용자 선호도 기반 평가
        fitness = 0
        preferred_notes = user_preferences.get('preferred_notes', [])
        for note, _ in dna.top_notes + dna.middle_notes + dna.base_notes:
            if note in preferred_notes:
                fitness += 0.1

        return min(1.0, fitness)

    def _evaluate_stability(self, dna: FragranceDNA) -> float:
        """안정성 평가: 향의 지속성과 조화"""
        # Base notes의 비중 (지속성)
        total_conc = sum(c for _, c in dna.top_notes + dna.middle_notes + dna.base_notes)
        base_conc = sum(c for _, c in dna.base_notes)
        persistence = base_conc / total_conc if total_conc > 0 else 0

        # 노트 간 전환의 부드러움
        transition_smoothness = 0
        if dna.top_notes and dna.middle_notes:
            # Top과 Middle의 농도 차이가 적을수록 부드러운 전환
            top_avg = np.mean([c for _, c in dna.top_notes])
            middle_avg = np.mean([c for _, c in dna.middle_notes])
            transition_smoothness += 1 - abs(top_avg - middle_avg)

        if dna.middle_notes and dna.base_notes:
            middle_avg = np.mean([c for _, c in dna.middle_notes])
            base_avg = np.mean([c for _, c in dna.base_notes])
            transition_smoothness += 1 - abs(middle_avg - base_avg)

        transition_smoothness /= 2

        stability = persistence * 0.6 + transition_smoothness * 0.4
        return min(1.0, stability)

    def _crossover(self, parent1: FragranceDNA, parent2: FragranceDNA) -> FragranceDNA:
        """실제 교차 연산"""
        child = FragranceDNA(
            top_notes=[],
            middle_notes=[],
            base_notes=[],
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.dna_id, parent2.dna_id]
        )

        # 각 노트 타입별로 교차
        for note_type in ['top_notes', 'middle_notes', 'base_notes']:
            p1_notes = getattr(parent1, note_type)
            p2_notes = getattr(parent2, note_type)

            # 균일 교차
            child_notes = []
            for i in range(max(len(p1_notes), len(p2_notes))):
                if random.random() < 0.5:
                    if i < len(p1_notes):
                        child_notes.append(p1_notes[i])
                else:
                    if i < len(p2_notes):
                        child_notes.append(p2_notes[i])

            setattr(child, note_type, child_notes[:5])  # 최대 5개로 제한

        return child

    def _mutate(self, dna: FragranceDNA) -> FragranceDNA:
        """실제 돌연변이 연산"""
        if random.random() < self.mutation_rate:
            # 랜덤하게 한 노트를 변경
            mutation_type = random.choice(['add', 'remove', 'modify'])
            note_type = random.choice(['top_notes', 'middle_notes', 'base_notes'])

            notes = getattr(dna, note_type)

            if mutation_type == 'add' and len(notes) < 5:
                # 새로운 노트 추가
                category = note_type.replace('_notes', '')
                new_note = (random.choice(self.fragrance_db[category]),
                           random.uniform(0.1, 0.7))
                notes.append(new_note)

            elif mutation_type == 'remove' and len(notes) > 1:
                # 랜덤 노트 제거
                notes.pop(random.randint(0, len(notes) - 1))

            elif mutation_type == 'modify' and notes:
                # 농도 변경
                idx = random.randint(0, len(notes) - 1)
                name, conc = notes[idx]
                notes[idx] = (name, min(1.0, max(0.1, conc + random.uniform(-0.2, 0.2))))

        return dna

    def _selection(self, population: List[FragranceDNA]) -> List[FragranceDNA]:
        """토너먼트 선택"""
        selected = []
        tournament_size = 5

        for _ in range(self.population_size):
            tournament = random.sample(population, tournament_size)
            # 종합 점수로 승자 결정
            winner = max(tournament, key=lambda d:
                        d.creativity_score * self.creativity_weight +
                        d.fitness_score * self.fitness_weight +
                        d.stability_score * self.stability_weight)
            selected.append(winner)

        return selected

    def _get_pareto_front(self, population: List[FragranceDNA]) -> List[FragranceDNA]:
        """파레토 프론트 추출"""
        pareto_front = []

        for candidate in population:
            is_dominated = False

            for other in population:
                if other == candidate:
                    continue

                # other가 candidate를 지배하는지 확인
                if (other.creativity_score >= candidate.creativity_score and
                    other.fitness_score >= candidate.fitness_score and
                    other.stability_score >= candidate.stability_score and
                    (other.creativity_score > candidate.creativity_score or
                     other.fitness_score > candidate.fitness_score or
                     other.stability_score > candidate.stability_score)):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(candidate)

        return pareto_front

    def optimize(
        self,
        user_preferences: Dict = None,
        num_generations: Optional[int] = None,
        callbacks: List[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        실제 MOGA 최적화 실행

        Returns:
            최적화된 향수 레시피 리스트
        """
        if num_generations:
            self.num_generations = num_generations

        logger.info(f"Starting MOGA optimization with {self.population_size} population for {self.num_generations} generations")

        # 초기 개체군 생성
        population = [self._create_random_dna() for _ in range(self.population_size)]

        # 진화 루프
        for generation in range(self.num_generations):
            # 평가
            for dna in population:
                dna.creativity_score = self._evaluate_creativity(dna)
                dna.fitness_score = self._evaluate_fitness(dna, user_preferences)
                dna.stability_score = self._evaluate_stability(dna)

            # 엘리트 보존
            population.sort(key=lambda d:
                          d.creativity_score * self.creativity_weight +
                          d.fitness_score * self.fitness_weight +
                          d.stability_score * self.stability_weight,
                          reverse=True)

            elite_size = int(self.population_size * self.elitism_rate)
            elite = population[:elite_size]

            # 선택
            selected = self._selection(population)

            # 교차와 돌연변이
            offspring = elite.copy()  # 엘리트는 그대로 유지

            while len(offspring) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent1, parent2 = random.sample(selected, 2)
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice(selected)

                child = self._mutate(child)
                offspring.append(child)

            population = offspring[:self.population_size]

            # 통계 기록
            best_dna = population[0]
            avg_creativity = np.mean([d.creativity_score for d in population])
            avg_fitness = np.mean([d.fitness_score for d in population])
            avg_stability = np.mean([d.stability_score for d in population])

            self.evolution_history.append({
                'generation': generation,
                'best_overall': (best_dna.creativity_score * self.creativity_weight +
                               best_dna.fitness_score * self.fitness_weight +
                               best_dna.stability_score * self.stability_weight),
                'avg_creativity': avg_creativity,
                'avg_fitness': avg_fitness,
                'avg_stability': avg_stability
            })

            # 콜백 실행
            if callbacks:
                for callback in callbacks:
                    callback(generation, population, self.evolution_history)

            if generation % 10 == 0:
                logger.info(f"Generation {generation}: Best score = {self.evolution_history[-1]['best_overall']:.3f}")

        # 최종 평가
        for dna in population:
            dna.creativity_score = self._evaluate_creativity(dna)
            dna.fitness_score = self._evaluate_fitness(dna, user_preferences)
            dna.stability_score = self._evaluate_stability(dna)

        # 파레토 프론트 추출
        self.pareto_front = self._get_pareto_front(population)

        logger.info(f"Optimization complete. Found {len(self.pareto_front)} Pareto optimal solutions")

        # 상위 10개 반환
        top_solutions = sorted(self.pareto_front,
                             key=lambda d: d.creativity_score * self.creativity_weight +
                                         d.fitness_score * self.fitness_weight +
                                         d.stability_score * self.stability_weight,
                             reverse=True)[:10]

        return [dna.to_recipe() for dna in top_solutions]


# ============================================================================
# 실제 RLHF (Reinforcement Learning from Human Feedback) 구현
# ============================================================================

class FragranceEnvironment:
    """향수 생성 환경"""

    def __init__(self, fragrance_db: Dict[str, List[str]]):
        self.fragrance_db = fragrance_db
        self.current_recipe = None
        self.history = []

    def reset(self) -> np.ndarray:
        """환경 초기화"""
        # 랜덤 초기 레시피
        self.current_recipe = {
            'top': random.sample(self.fragrance_db['top'], 3),
            'middle': random.sample(self.fragrance_db['middle'], 3),
            'base': random.sample(self.fragrance_db['base'], 2)
        }
        self.history = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """현재 상태를 벡터로 변환"""
        # 간단한 원-핫 인코딩
        state = []
        for category in ['top', 'middle', 'base']:
            for note in self.fragrance_db[category]:
                state.append(1.0 if note in self.current_recipe[category] else 0.0)
        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """행동 실행"""
        # 행동 디코딩
        actions = ['add_top', 'remove_top', 'add_middle', 'remove_middle',
                  'add_base', 'remove_base', 'modify_concentration']

        if action >= len(actions):
            action = action % len(actions)

        action_type = actions[action]

        # 행동 실행
        if 'add' in action_type:
            category = action_type.split('_')[1]
            available = [n for n in self.fragrance_db[category]
                        if n not in self.current_recipe[category]]
            if available:
                self.current_recipe[category].append(random.choice(available))

        elif 'remove' in action_type:
            category = action_type.split('_')[1]
            if len(self.current_recipe[category]) > 1:
                self.current_recipe[category].pop(random.randint(0, len(self.current_recipe[category]) - 1))

        # 보상은 나중에 인간 피드백으로
        reward = 0
        done = len(self.history) >= 10  # 10번 수정 후 종료

        self.history.append(action)

        return self._get_state(), reward, done, {'recipe': self.current_recipe}


class RealRLHF:
    """
    실제 작동하는 강화학습 기반 인간 피드백 학습
    """

    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-Network
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

        # 경험 버퍼
        self.memory = deque(maxlen=10000)

        # 하이퍼파라미터
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32

        # 인간 피드백 히스토리
        self.feedback_history = []

    def _build_q_network(self) -> nn.Module:
        """Q-Network 구성"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )

    def act(self, state: np.ndarray) -> int:
        """ε-greedy 정책으로 행동 선택"""
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """경험 저장"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """경험 재생으로 학습"""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.FloatTensor([e[4] for e in batch])

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ε 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _get_feedback_from_database(self, state, action):
        """실제 데이터베이스에서 피드백 조회"""
        import sqlite3
        from pathlib import Path

        db_path = Path(__file__).parent.parent.parent / "data" / "rl_feedback.db"

        if not db_path.exists():
            return None

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 상태를 해시로 변환
        state_hash = hash(state.tobytes() if hasattr(state, 'tobytes') else str(state))

        cursor.execute("""
            SELECT rating FROM feedback
            WHERE state_hash = ? AND action = ?
            ORDER BY timestamp DESC LIMIT 1
        """, (state_hash, action))

        result = cursor.fetchone()
        conn.close()

        return result[0] if result else None

    def incorporate_human_feedback(self, state, action, human_rating):
        """
        인간 피드백을 보상으로 변환하여 학습

        Args:
            state: 상태
            action: 선택된 행동
            human_rating: 인간 평가 (1-5)
        """
        # 평가를 보상으로 변환 (-1 ~ 1)
        reward = (human_rating - 3) / 2

        # 피드백 저장
        self.feedback_history.append({
            'state': state,
            'action': action,
            'rating': human_rating,
            'reward': reward
        })

        # 다음 상태는 현재 상태와 동일 (피드백은 종료 상태)
        self.remember(state, action, reward, state, True)

        # 즉시 학습
        self.replay()

        logger.info(f"Human feedback incorporated: rating={human_rating}, reward={reward:.2f}")

    def train_with_feedback(self, env: FragranceEnvironment, num_episodes: int = 100):
        """
        환경과 상호작용하며 학습
        """
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            while True:
                action = self.act(state)
                next_state, reward, done, info = env.step(action)

                # 실제 인간 피드백 데이터베이스 조회
                human_rating = self._get_feedback_from_database(state, action)
                if human_rating is not None:
                    self.incorporate_human_feedback(state, action, human_rating)
                    reward = (human_rating - 3) / 2

                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    break

                self.replay()

            # 타겟 네트워크 주기적 업데이트
            if episode % 10 == 0:
                self.update_target_network()
                logger.info(f"Episode {episode}: Total reward = {total_reward:.2f}, ε = {self.epsilon:.3f}")


# ============================================================================
# 통합 관리자
# ============================================================================

class RealOptimizerManager:
    """
    실제 작동하는 옵티마이저 관리자
    """

    def __init__(self):
        self.moga = None
        self.rlhf = None
        self.fragrance_env = None

        # 설정 로드
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        """설정 파일 로드"""
        config_path = Path(__file__).parent.parent / 'configs' / 'optimizer_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)

        # 기본 설정
        return {
            'moga': {
                'population_size': 100,
                'num_generations': 50,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7
            },
            'rlhf': {
                'state_dim': 100,
                'action_dim': 10,
                'learning_rate': 0.001
            }
        }

    def initialize_moga(self, fragrance_db_path: Optional[str] = None):
        """MOGA 초기화"""
        self.moga = RealMOGA(fragrance_db_path)
        logger.info("MOGA optimizer initialized")

    def initialize_rlhf(self, state_dim: int = None, action_dim: int = None):
        """RLHF 초기화"""
        state_dim = state_dim or self.config['rlhf']['state_dim']
        action_dim = action_dim or self.config['rlhf']['action_dim']

        self.rlhf = RealRLHF(state_dim, action_dim)

        # 환경도 초기화
        moga = self.moga or RealMOGA()
        self.fragrance_env = FragranceEnvironment(moga.fragrance_db)

        logger.info(f"RLHF optimizer initialized with state_dim={state_dim}, action_dim={action_dim}")

    def optimize_with_moga(
        self,
        user_preferences: Dict = None,
        creativity_weight: float = 0.33,
        fitness_weight: float = 0.33,
        stability_weight: float = 0.34
    ) -> List[Dict[str, Any]]:
        """
        MOGA로 최적화 실행

        Returns:
            최적화된 레시피 리스트
        """
        if not self.moga:
            self.initialize_moga()

        # 가중치 설정
        self.moga.creativity_weight = creativity_weight
        self.moga.fitness_weight = fitness_weight
        self.moga.stability_weight = stability_weight

        # 최적화 실행
        results = self.moga.optimize(user_preferences)

        logger.info(f"MOGA optimization complete. Generated {len(results)} optimal recipes")

        return results

    def train_with_human_feedback(self, num_episodes: int = 100):
        """
        RLHF로 인간 피드백 기반 학습
        """
        if not self.rlhf:
            self.initialize_rlhf()

        if not self.fragrance_env:
            moga = self.moga or RealMOGA()
            self.fragrance_env = FragranceEnvironment(moga.fragrance_db)

        # 학습 실행
        self.rlhf.train_with_feedback(self.fragrance_env, num_episodes)

        logger.info(f"RLHF training complete. Trained for {num_episodes} episodes")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """최적화 통계 반환"""
        stats = {}

        if self.moga and self.moga.evolution_history:
            stats['moga'] = {
                'generations': len(self.moga.evolution_history),
                'pareto_front_size': len(self.moga.pareto_front),
                'final_best_score': self.moga.evolution_history[-1]['best_overall']
            }

        if self.rlhf and self.rlhf.feedback_history:
            ratings = [f['rating'] for f in self.rlhf.feedback_history]
            stats['rlhf'] = {
                'total_feedbacks': len(ratings),
                'average_rating': np.mean(ratings) if ratings else 0,
                'epsilon': self.rlhf.epsilon
            }

        return stats


# 전역 인스턴스
_real_optimizer_manager = None

def get_real_optimizer_manager() -> RealOptimizerManager:
    """싱글톤 옵티마이저 매니저 반환"""
    global _real_optimizer_manager
    if _real_optimizer_manager is None:
        _real_optimizer_manager = RealOptimizerManager()
    return _real_optimizer_manager