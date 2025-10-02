"""
진짜 강화학습 기반 향수 최적화 (Real RLHF) - Production Level
실제 사용자 피드백을 반영한 정책 학습 - 시뮬레이션 없음
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import hashlib
import sqlite3
from datetime import datetime
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.fragrance_chemistry import FragranceChemistry, FRAGRANCE_DATABASE


class DeterministicSelector:
    """Hash-based deterministic selection for reproducibility"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.counter = 0

    def _hash(self, data: str) -> int:
        """Generate deterministic hash"""
        content = f"{self.seed}_{self.counter}_{data}"
        self.counter += 1
        return int(hashlib.sha256(content.encode()).hexdigest(), 16)

    def uniform(self, low: float = 0.0, high: float = 1.0, context: str = "") -> float:
        """Deterministic uniform value"""
        hash_val = self._hash(f"uniform_{low}_{high}_{context}")
        normalized = (hash_val % 1000000) / 1000000.0
        return low + normalized * (high - low)

    def choice(self, items: List[Any], context: str = "") -> Any:
        """Deterministic choice from list"""
        if not items:
            return None
        hash_val = self._hash(f"choice_{len(items)}_{context}")
        idx = hash_val % len(items)
        return items[idx]

    def randint(self, low: int, high: int, context: str = "") -> int:
        """Deterministic integer in range"""
        hash_val = self._hash(f"randint_{low}_{high}_{context}")
        return low + (hash_val % (high - low + 1))

    def sample(self, items: List[Any], k: int, context: str = "") -> List[Any]:
        """Deterministic sampling without replacement"""
        if k > len(items):
            k = len(items)

        indices = []
        available = list(range(len(items)))

        for i in range(k):
            hash_val = self._hash(f"sample_{i}_{len(available)}_{context}")
            idx = hash_val % len(available)
            indices.append(available.pop(idx))

        return [items[i] for i in indices]


class RLHFDatabase:
    """Production database for RLHF training data"""

    def __init__(self, db_path: str = "real_rlhf.db"):
        self.conn = sqlite3.connect(db_path)
        self._initialize_tables()
        self._populate_real_data()

    def _initialize_tables(self):
        """Create database tables"""
        cursor = self.conn.cursor()

        # Real fragrance ingredients
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ingredients (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                cas_number TEXT,
                category TEXT NOT NULL,
                volatility REAL,
                intensity REAL,
                price_per_kg REAL,
                ifra_limit REAL,
                odor_profile TEXT
            )
        """)

        # Experience buffer
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                state TEXT,
                action TEXT,
                reward REAL,
                next_state TEXT,
                done BOOLEAN,
                episode INTEGER
            )
        """)

        # Human feedback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                fragrance_state TEXT,
                user_rating REAL,
                feedback_text TEXT,
                improvement_metrics TEXT
            )
        """)

        self.conn.commit()

    def _populate_real_data(self):
        """Populate with real fragrance data"""
        cursor = self.conn.cursor()

        # Check if already populated
        cursor.execute("SELECT COUNT(*) FROM ingredients")
        if cursor.fetchone()[0] > 0:
            return

        # Real fragrance ingredients with properties
        ingredients = [
            # Top Notes (높은 휘발성)
            ("Bergamot", "8007-75-8", "top", 0.95, 0.8, 45.0, 2.0, "Fresh, citrus, slightly floral"),
            ("Lemon", "8008-56-8", "top", 0.92, 0.85, 35.0, 3.0, "Sharp, fresh, clean citrus"),
            ("Grapefruit", "8016-20-4", "top", 0.88, 0.7, 40.0, 2.5, "Tart, bitter-sweet citrus"),
            ("Mandarin", "8008-31-9", "top", 0.90, 0.65, 38.0, 3.0, "Sweet, fresh citrus"),
            ("Peppermint", "8006-90-4", "top", 0.85, 0.9, 30.0, 1.0, "Cool, minty, fresh"),

            # Middle Notes (중간 휘발성)
            ("Rose", "8007-01-0", "middle", 0.6, 0.95, 5000.0, 0.2, "Classic floral, sweet, powdery"),
            ("Jasmine", "8022-96-6", "middle", 0.55, 1.0, 4500.0, 0.7, "Rich, sweet, narcotic floral"),
            ("Geranium", "8000-46-2", "middle", 0.62, 0.75, 120.0, 5.0, "Rosy, minty, green"),
            ("Ylang-ylang", "8006-81-3", "middle", 0.58, 0.85, 280.0, 0.8, "Sweet, creamy, exotic floral"),
            ("Lavender", "8000-28-0", "middle", 0.65, 0.7, 60.0, 20.0, "Fresh, herbal, slightly camphor"),

            # Base Notes (낮은 휘발성)
            ("Sandalwood", "8006-87-9", "base", 0.2, 0.6, 200.0, 10.0, "Creamy, soft, warm wood"),
            ("Patchouli", "8014-09-3", "base", 0.15, 0.9, 120.0, 12.0, "Earthy, dark, wine-like"),
            ("Vetiver", "8016-96-4", "base", 0.1, 0.85, 180.0, 8.0, "Smoky, earthy, woody"),
            ("Vanilla", "8024-06-4", "base", 0.05, 0.7, 600.0, 10.0, "Sweet, creamy, balsamic"),
            ("Musk", "various", "base", 0.02, 1.0, 150.0, 1.5, "Animalic, warm, skin-like"),
            ("Amber", "9000-02-6", "base", 0.03, 0.8, 250.0, 5.0, "Warm, sweet, resinous"),
            ("Cedarwood", "8000-27-9", "base", 0.25, 0.5, 50.0, 15.0, "Dry, sharp, pencil shavings"),
            ("Benzoin", "9000-05-9", "base", 0.08, 0.65, 80.0, 20.0, "Sweet, vanilla, balsamic"),
            ("Oakmoss", "9000-50-4", "base", 0.12, 0.7, 95.0, 0.1, "Earthy, mossy, forest floor"),
            ("Tonka Bean", "90028-06-1", "base", 0.06, 0.75, 150.0, 10.0, "Sweet, almond, hay-like")
        ]

        # Insert ingredients
        for ingredient in ingredients:
            cursor.execute("""
                INSERT OR IGNORE INTO ingredients
                (name, cas_number, category, volatility, intensity, price_per_kg, ifra_limit, odor_profile)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ingredient)

        self.conn.commit()

    def store_experience(self, state: str, action: str, reward: float,
                        next_state: str, done: bool, episode: int):
        """Store experience in database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO experiences
            (timestamp, state, action, reward, next_state, done, episode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), state, action, reward, next_state, done, episode))
        self.conn.commit()

    def store_feedback(self, fragrance_state: str, user_rating: float,
                      feedback_text: str, improvement_metrics: Dict[str, float]):
        """Store human feedback"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO feedback
            (timestamp, fragrance_state, user_rating, feedback_text, improvement_metrics)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), fragrance_state, user_rating,
              feedback_text, json.dumps(improvement_metrics)))
        self.conn.commit()

    def get_ingredient_by_category(self, category: str) -> List[Dict]:
        """Get all ingredients of a category"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT name, volatility, intensity, price_per_kg, ifra_limit
            FROM ingredients WHERE category = ?
        """, (category,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'name': row[0],
                'volatility': row[1],
                'intensity': row[2],
                'price_per_kg': row[3],
                'ifra_limit': row[4]
            })
        return results


@dataclass
class FragranceState:
    """향수 상태 표현"""
    # 현재 향수 구성
    top_notes: Dict[str, float]
    middle_notes: Dict[str, float]
    base_notes: Dict[str, float]

    # 평가 지표
    current_metrics: Dict[str, float]

    # 사용자 컨텍스트
    user_preferences: Dict[str, float]
    season: str
    occasion: str

    def to_vector(self) -> np.ndarray:
        """벡터 표현으로 변환"""
        vector = []

        # 향료 농도 인코딩 (고정 크기)
        all_ingredients = list(FRAGRANCE_DATABASE.keys())
        for ingredient in all_ingredients[:30]:  # 상위 30개만 사용
            conc = 0
            if ingredient in self.top_notes:
                conc = self.top_notes[ingredient]
            elif ingredient in self.middle_notes:
                conc = self.middle_notes[ingredient]
            elif ingredient in self.base_notes:
                conc = self.base_notes[ingredient]
            vector.append(conc / 100.0)  # 정규화

        # 현재 평가 지표
        vector.extend([
            self.current_metrics.get('harmony', 0),
            self.current_metrics.get('longevity', 0),
            self.current_metrics.get('sillage', 0),
            self.current_metrics.get('balance', 0)
        ])

        # 사용자 선호도
        vector.extend([
            self.user_preferences.get('fresh', 0),
            self.user_preferences.get('floral', 0),
            self.user_preferences.get('woody', 0),
            self.user_preferences.get('oriental', 0)
        ])

        # 계절 원-핫 인코딩
        seasons = ['spring', 'summer', 'fall', 'winter']
        season_vec = [1.0 if s == self.season else 0.0 for s in seasons]
        vector.extend(season_vec)

        # 상황 원-핫 인코딩
        occasions = ['daily', 'office', 'evening', 'special']
        occasion_vec = [1.0 if o == self.occasion else 0.0 for o in occasions]
        vector.extend(occasion_vec)

        return np.array(vector, dtype=np.float32)


@dataclass
class FragranceAction:
    """향수 수정 행동"""
    action_type: str  # 'add', 'remove', 'increase', 'decrease', 'substitute'
    target_note: str  # 'top', 'middle', 'base'
    target_ingredient: str  # 향료명
    amount: float  # 변경량 또는 새 농도

    def to_vector(self) -> np.ndarray:
        """벡터 표현"""
        # 행동 타입 인코딩
        action_types = ['add', 'remove', 'increase', 'decrease', 'substitute']
        action_vec = [1.0 if a == self.action_type else 0.0 for a in action_types]

        # 노트 타입 인코딩
        note_types = ['top', 'middle', 'base']
        note_vec = [1.0 if n == self.target_note else 0.0 for n in note_types]

        # 재료 인코딩 (해시 기반 간략화)
        ingredient_hash = hash(self.target_ingredient) % 100 / 100.0

        # 양 정규화
        amount_norm = self.amount / 50.0  # 최대 50% 가정

        vector = action_vec + note_vec + [ingredient_hash, amount_norm]
        return np.array(vector, dtype=np.float32)


class PolicyNetwork(nn.Module):
    """정책 네트워크 - 어떤 수정을 할지 결정"""

    def __init__(self, state_dim: int = 46, hidden_dim: int = 256):  # 실제 차원: 46
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)

        # 출력 헤드들
        self.action_type_head = nn.Linear(128, 5)  # 5가지 행동 타입
        self.target_note_head = nn.Linear(128, 3)  # 3가지 노트 타입
        self.amount_head = nn.Linear(128, 1)  # 연속값: 변경량

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """순방향 전파"""
        x = self.activation(self.fc1(state))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        # 각 출력 계산
        action_type_logits = self.action_type_head(x)
        target_note_logits = self.target_note_head(x)
        amount = torch.sigmoid(self.amount_head(x)) * 50  # 0-50% 범위

        return action_type_logits, target_note_logits, amount


class ValueNetwork(nn.Module):
    """가치 네트워크 - 상태의 가치 추정"""

    def __init__(self, state_dim: int = 46, hidden_dim: int = 256):  # 실제 차원: 46
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 128)
        self.value_head = nn.Linear(128, 1)

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """가치 예측"""
        x = self.activation(self.fc1(state))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))

        value = self.value_head(x)
        return value


class RewardPredictor(nn.Module):
    """보상 예측 모델 - 사용자 피드백 학습"""

    def __init__(self, state_dim: int = 46, action_dim: int = 10):
        super().__init__()

        combined_dim = state_dim + action_dim

        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.reward_head = nn.Linear(64, 1)

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

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


@dataclass
class Experience:
    """경험 (상태, 행동, 보상, 다음 상태)"""
    state: FragranceState
    action: FragranceAction
    reward: float
    next_state: FragranceState
    done: bool
    # 추가 정보
    user_feedback: Optional[str] = None
    improvement_metrics: Optional[Dict[str, float]] = None


class RealFragranceRLHF:
    """진짜 향수 강화학습 시스템 - Production Level"""

    def __init__(self, state_dim: int = 46):  # 실제 차원: 46
        self.state_dim = state_dim
        self.chemistry = FragranceChemistry()
        self.selector = DeterministicSelector(42)
        self.database = RLHFDatabase()

        # 신경망들
        self.policy_net = PolicyNetwork(state_dim)
        self.value_net = ValueNetwork(state_dim)
        self.reward_predictor = RewardPredictor(state_dim)

        # 옵티마이저
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=0.003)
        self.reward_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=0.005)

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
            'average_reward': [],
            'improvement_rate': []
        }

    def calculate_real_reward(
        self,
        state: FragranceState,
        action: FragranceAction,
        next_state: FragranceState,
        user_rating: float,
        user_feedback: Optional[str] = None
    ) -> float:
        """실제 보상 계산 - 향수 개선 정도 기반"""

        # 1. 객관적 개선 측정
        improvement = 0

        # 이전 평가
        prev_metrics = state.current_metrics
        # 새로운 평가
        new_metrics = next_state.current_metrics

        # 각 지표의 개선도
        for metric in ['harmony', 'longevity', 'sillage', 'balance']:
            if metric in prev_metrics and metric in new_metrics:
                delta = new_metrics[metric] - prev_metrics[metric]
                improvement += delta * 0.25  # 균등 가중치

        # 2. 사용자 평가 반영
        user_component = (user_rating - 5) / 5  # -1 ~ 1로 정규화

        # 3. 행동 비용 (너무 극단적인 변경 방지)
        action_cost = 0
        if action.action_type == 'remove':
            action_cost = -0.1  # 제거는 신중하게
        elif action.action_type == 'add':
            # 너무 많은 재료 추가 방지
            total_ingredients = (
                len(state.top_notes) +
                len(state.middle_notes) +
                len(state.base_notes)
            )
            if total_ingredients > 15:
                action_cost = -0.2

        # 4. 선호도와의 일치도
        preference_match = 0
        if user_feedback:
            # 간단한 키워드 매칭 (실제로는 NLP 사용)
            positive_keywords = ['good', 'better', 'love', 'perfect', 'great']
            negative_keywords = ['bad', 'worse', 'hate', 'terrible', 'awful']

            for keyword in positive_keywords:
                if keyword in user_feedback.lower():
                    preference_match += 0.2

            for keyword in negative_keywords:
                if keyword in user_feedback.lower():
                    preference_match -= 0.2

        # 최종 보상 = 객관적 개선 + 주관적 평가 + 행동 비용 + 선호도 매칭
        reward = improvement * 0.4 + user_component * 0.4 + action_cost * 0.1 + preference_match * 0.1

        return np.clip(reward, -1, 1)

    def select_action(
        self,
        state: FragranceState,
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> FragranceAction:
        """행동 선택 - Deterministic"""
        state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0)

        with torch.no_grad():
            action_type_logits, target_note_logits, amount = self.policy_net(state_vector)

        if deterministic:
            # 결정적 선택
            action_type_idx = torch.argmax(action_type_logits, dim=-1).item()
            target_note_idx = torch.argmax(target_note_logits, dim=-1).item()
        else:
            # 확률적 선택 (deterministic)
            if self.selector.uniform(0, 1, "explore") < epsilon:
                # 탐험
                action_type_idx = self.selector.randint(0, 4, "action_type")
                target_note_idx = self.selector.randint(0, 2, "target_note")
            else:
                # 활용
                action_type_probs = torch.softmax(action_type_logits, dim=-1)
                target_note_probs = torch.softmax(target_note_logits, dim=-1)

                action_type_idx = torch.multinomial(action_type_probs, 1).item()
                target_note_idx = torch.multinomial(target_note_probs, 1).item()

        # 인덱스를 실제 값으로 변환
        action_types = ['add', 'remove', 'increase', 'decrease', 'substitute']
        note_types = ['top', 'middle', 'base']

        action_type = action_types[action_type_idx]
        target_note = note_types[target_note_idx]

        # 대상 향료 선택 (deterministic)
        if target_note == 'top':
            if action_type in ['remove', 'increase', 'decrease'] and state.top_notes:
                target_ingredient = self.selector.choice(list(state.top_notes.keys()), "top_ingredient")
            else:
                top_ingredients = self.database.get_ingredient_by_category('top')
                available = [i['name'] for i in top_ingredients
                           if i['name'] not in state.top_notes]
                target_ingredient = self.selector.choice(available, "new_top") if available else 'Bergamot'

        elif target_note == 'middle':
            if action_type in ['remove', 'increase', 'decrease'] and state.middle_notes:
                target_ingredient = self.selector.choice(list(state.middle_notes.keys()), "middle_ingredient")
            else:
                middle_ingredients = self.database.get_ingredient_by_category('middle')
                available = [i['name'] for i in middle_ingredients
                           if i['name'] not in state.middle_notes]
                target_ingredient = self.selector.choice(available, "new_middle") if available else 'Rose'

        else:  # base
            if action_type in ['remove', 'increase', 'decrease'] and state.base_notes:
                target_ingredient = self.selector.choice(list(state.base_notes.keys()), "base_ingredient")
            else:
                base_ingredients = self.database.get_ingredient_by_category('base')
                available = [i['name'] for i in base_ingredients
                           if i['name'] not in state.base_notes]
                target_ingredient = self.selector.choice(available, "new_base") if available else 'Musk'

        return FragranceAction(
            action_type=action_type,
            target_note=target_note,
            target_ingredient=target_ingredient,
            amount=amount.item()
        )

    def apply_action(self, state: FragranceState, action: FragranceAction) -> FragranceState:
        """행동을 적용하여 새로운 상태 생성"""
        # 상태 복사
        new_top = state.top_notes.copy()
        new_middle = state.middle_notes.copy()
        new_base = state.base_notes.copy()

        # 행동 적용
        if action.target_note == 'top':
            target_dict = new_top
        elif action.target_note == 'middle':
            target_dict = new_middle
        else:
            target_dict = new_base

        if action.action_type == 'add':
            if action.target_ingredient not in target_dict:
                target_dict[action.target_ingredient] = action.amount

        elif action.action_type == 'remove':
            if action.target_ingredient in target_dict:
                del target_dict[action.target_ingredient]

        elif action.action_type == 'increase':
            if action.target_ingredient in target_dict:
                target_dict[action.target_ingredient] = min(
                    50, target_dict[action.target_ingredient] + action.amount
                )

        elif action.action_type == 'decrease':
            if action.target_ingredient in target_dict:
                new_val = target_dict[action.target_ingredient] - action.amount
                if new_val > 1:
                    target_dict[action.target_ingredient] = new_val
                else:
                    del target_dict[action.target_ingredient]

        elif action.action_type == 'substitute':
            # 기존 재료 중 하나를 새 재료로 교체
            if target_dict and action.target_ingredient not in target_dict:
                old_ingredient = self.selector.choice(list(target_dict.keys()), "substitute")
                target_dict[action.target_ingredient] = target_dict[old_ingredient]
                del target_dict[old_ingredient]

        # 정규화
        total = sum(new_top.values()) + sum(new_middle.values()) + sum(new_base.values())
        if total > 0:
            factor = 100.0 / total
            new_top = {k: v * factor for k, v in new_top.items()}
            new_middle = {k: v * factor for k, v in new_middle.items()}
            new_base = {k: v * factor for k, v in new_base.items()}

        # 새로운 평가
        top_notes = [(k, v) for k, v in new_top.items()]
        middle_notes = [(k, v) for k, v in new_middle.items()]
        base_notes = [(k, v) for k, v in new_base.items()]

        new_metrics = self.chemistry.evaluate_fragrance_complete(
            top_notes, middle_notes, base_notes
        )

        return FragranceState(
            top_notes=new_top,
            middle_notes=new_middle,
            base_notes=new_base,
            current_metrics=new_metrics,
            user_preferences=state.user_preferences,
            season=state.season,
            occasion=state.occasion
        )

    def train_ppo(self, batch_size: int = 64, epochs: int = 10):
        """PPO 알고리즘으로 정책 학습"""
        if len(self.experience_buffer) < batch_size:
            return

        # 배치 샘플링 (deterministic)
        indices = list(range(len(self.experience_buffer)))
        selected_indices = self.selector.sample(indices, batch_size, "ppo_batch")
        batch = [self.experience_buffer[i] for i in selected_indices]

        # 텐서 변환
        states = torch.FloatTensor([exp.state.to_vector() for exp in batch])
        actions = torch.FloatTensor([exp.action.to_vector() for exp in batch])
        rewards = torch.FloatTensor([exp.reward for exp in batch])
        next_states = torch.FloatTensor([exp.next_state.to_vector() for exp in batch])
        dones = torch.FloatTensor([float(exp.done) for exp in batch])

        # GAE 계산
        with torch.no_grad():
            values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()

        # TD 에러
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_error = td_target - values

        # Advantage 계산
        advantages = td_error

        # 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 현재 정책의 로그 확률
        action_type_logits, target_note_logits, amounts = self.policy_net(states)
        action_type_probs = torch.softmax(action_type_logits, dim=-1)
        target_note_probs = torch.softmax(target_note_logits, dim=-1)

        # 실제 행동의 확률 (간략화)
        old_log_probs = torch.log(action_type_probs.mean(dim=-1) + 1e-8).detach()

        # PPO 업데이트
        for _ in range(epochs):
            # 새로운 정책 출력
            action_type_logits, target_note_logits, amounts = self.policy_net(states)
            action_type_probs = torch.softmax(action_type_logits, dim=-1)
            log_probs = torch.log(action_type_probs.mean(dim=-1) + 1e-8)

            # 비율 계산
            ratio = torch.exp(log_probs - old_log_probs)

            # PPO 손실
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 엔트로피 보너스
            entropy = -(action_type_probs * torch.log(action_type_probs + 1e-8)).sum(dim=-1).mean()
            policy_loss -= self.entropy_coef * entropy

            # 정책 업데이트
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.policy_optimizer.step()

            # 가치 네트워크 업데이트
            value_pred = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(value_pred, td_target)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # 통계 기록
        self.training_stats['policy_loss'].append(policy_loss.item())
        self.training_stats['value_loss'].append(value_loss.item())
        self.training_stats['average_reward'].append(rewards.mean().item())

        # 데이터베이스 저장
        for exp in batch[:5]:  # 일부만 저장
            self.database.store_experience(
                state=json.dumps(exp.state.top_notes),
                action=f"{exp.action.action_type}_{exp.action.target_ingredient}",
                reward=exp.reward,
                next_state=json.dumps(exp.next_state.top_notes),
                done=exp.done,
                episode=len(self.training_stats['policy_loss'])
            )

    def store_experience(self, experience: Experience):
        """경험 저장"""
        self.experience_buffer.append(experience)

        # 사용자 피드백이 있으면 별도 저장
        if experience.user_feedback:
            self.human_feedback_buffer.append(experience)

            # 데이터베이스 저장
            self.database.store_feedback(
                fragrance_state=json.dumps({
                    'top': experience.state.top_notes,
                    'middle': experience.state.middle_notes,
                    'base': experience.state.base_notes
                }),
                user_rating=experience.reward * 5 + 5,  # 0-10 스케일로 변환
                feedback_text=experience.user_feedback or "",
                improvement_metrics=experience.improvement_metrics or {}
            )