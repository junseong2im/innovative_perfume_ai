"""
실시간 적응형 학습 시스템
Real-time Adaptive Learning System

이 모듈은 사용자 피드백과 실시간 데이터를 기반으로
향수 생성 모델을 지속적으로 개선하는 적응형 학습 시스템을 제공합니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from collections import deque, defaultdict
import threading
import queue
import sqlite3
from datetime import datetime, timedelta
import pickle
import copy

logger = logging.getLogger(__name__)

@dataclass
class AdaptiveLearningConfig:
    """적응형 학습 설정 클래스"""

    # 기본 학습 설정
    base_learning_rate: float = 1e-4
    adaptive_learning_rate: float = 1e-5
    meta_learning_rate: float = 1e-3
    batch_size: int = 32
    buffer_size: int = 10000

    # 피드백 처리 설정
    feedback_weight: float = 2.0
    positive_feedback_bonus: float = 1.5
    negative_feedback_penalty: float = 0.5
    feedback_decay_rate: float = 0.95

    # 온라인 학습 설정
    online_update_frequency: int = 10  # 10개 샘플마다 업데이트
    experience_replay_ratio: float = 0.3
    min_samples_for_update: int = 5

    # 메타 학습 설정
    meta_batch_size: int = 8
    meta_update_steps: int = 5
    task_similarity_threshold: float = 0.7

    # 품질 관리 설정
    quality_threshold: float = 0.6
    max_adaptation_steps: int = 100
    convergence_threshold: float = 1e-6

    # 메모리 관리
    max_experience_buffer_size: int = 50000
    experience_cleanup_interval: int = 1000
    model_checkpoint_interval: int = 500

    # 개인화 설정
    user_profile_dim: int = 128
    preference_decay_factor: float = 0.99
    novelty_bonus: float = 0.1

class ExperienceBuffer:
    """경험 버퍼 - 학습 데이터와 피드백을 저장"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.feedback_scores = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.user_ids = deque(maxlen=max_size)

    def add_experience(self,
                      input_data: Dict[str, torch.Tensor],
                      output_data: Dict[str, torch.Tensor],
                      feedback_score: float,
                      user_id: str = "anonymous"):
        """경험 추가"""
        experience = {
            'input': input_data,
            'output': output_data,
            'feedback': feedback_score,
            'timestamp': datetime.now(),
            'user_id': user_id
        }

        self.experiences.append(experience)
        self.feedback_scores.append(feedback_score)
        self.timestamps.append(datetime.now())
        self.user_ids.append(user_id)

    def sample_batch(self, batch_size: int, priority_sampling: bool = True) -> List[Dict]:
        """배치 샘플링"""
        if len(self.experiences) < batch_size:
            return list(self.experiences)

        if priority_sampling and len(self.feedback_scores) > 0:
            # 피드백 점수 기반 우선순위 샘플링
            feedback_array = np.array(self.feedback_scores)

            # 음수 피드백도 학습에 중요하므로 절댓값 사용
            priorities = np.abs(feedback_array) + 0.1  # 최소 확률 보장
            probabilities = priorities / np.sum(priorities)

            indices = np.random.choice(
                len(self.experiences),
                size=min(batch_size, len(self.experiences)),
                replace=False,
                p=probabilities
            )
        else:
            # 균등 샘플링
            indices = np.random.choice(
                len(self.experiences),
                size=min(batch_size, len(self.experiences)),
                replace=False
            )

        return [self.experiences[i] for i in indices]

    def get_recent_experiences(self, hours: int = 24) -> List[Dict]:
        """최근 경험 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_experiences = []

        for exp, timestamp in zip(self.experiences, self.timestamps):
            if timestamp >= cutoff_time:
                recent_experiences.append(exp)

        return recent_experiences

    def get_user_experiences(self, user_id: str, limit: int = 100) -> List[Dict]:
        """특정 사용자 경험 조회"""
        user_experiences = []

        for exp, uid in zip(self.experiences, self.user_ids):
            if uid == user_id:
                user_experiences.append(exp)
                if len(user_experiences) >= limit:
                    break

        return user_experiences

    def cleanup_old_experiences(self, days: int = 30):
        """오래된 경험 정리"""
        cutoff_time = datetime.now() - timedelta(days=days)

        # 인덱스를 역순으로 처리하여 안전하게 삭제
        indices_to_remove = []
        for i, timestamp in enumerate(self.timestamps):
            if timestamp < cutoff_time:
                indices_to_remove.append(i)

        # 역순으로 삭제
        for i in reversed(indices_to_remove):
            del self.experiences[i]
            del self.feedback_scores[i]
            del self.timestamps[i]
            del self.user_ids[i]

        logger.info(f"{len(indices_to_remove)}개의 오래된 경험을 정리했습니다.")

class UserProfileManager:
    """사용자 프로필 관리자"""

    def __init__(self, profile_dim: int = 128):
        self.profile_dim = profile_dim
        self.user_profiles = {}
        self.user_preferences = defaultdict(lambda: defaultdict(float))
        self.user_feedback_history = defaultdict(list)

    def get_user_profile(self, user_id: str) -> torch.Tensor:
        """사용자 프로필 조회"""
        if user_id not in self.user_profiles:
            # 새 사용자 프로필 초기화
            self.user_profiles[user_id] = torch.zeros(self.profile_dim)

        return self.user_profiles[user_id].clone()

    def update_user_profile(self,
                           user_id: str,
                           fragrance_features: Dict[str, Any],
                           feedback_score: float,
                           learning_rate: float = 0.1):
        """사용자 프로필 업데이트"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = torch.zeros(self.profile_dim)

        # 피드백 히스토리 업데이트
        self.user_feedback_history[user_id].append({
            'features': fragrance_features,
            'feedback': feedback_score,
            'timestamp': datetime.now()
        })

        # 최근 100개 피드백만 유지
        if len(self.user_feedback_history[user_id]) > 100:
            self.user_feedback_history[user_id] = self.user_feedback_history[user_id][-100:]

        # 프로필 업데이트 (간단한 이동 평균)
        fragrance_tensor = self._features_to_tensor(fragrance_features)
        if len(fragrance_tensor) <= self.profile_dim:
            # 패딩 또는 트러케이션
            if len(fragrance_tensor) < self.profile_dim:
                fragrance_tensor = torch.cat([
                    fragrance_tensor,
                    torch.zeros(self.profile_dim - len(fragrance_tensor))
                ])
            else:
                fragrance_tensor = fragrance_tensor[:self.profile_dim]

            # 피드백 가중 업데이트
            weight = learning_rate * feedback_score
            self.user_profiles[user_id] = (
                (1 - abs(weight)) * self.user_profiles[user_id] +
                weight * fragrance_tensor
            )

        # 선호도 업데이트
        self._update_preferences(user_id, fragrance_features, feedback_score)

    def _features_to_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        """특성 딕셔너리를 텐서로 변환"""
        feature_list = []

        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:  # 스칼라
                    feature_list.append(value.unsqueeze(0))
                else:
                    feature_list.append(value.flatten())
            elif isinstance(value, (int, float)):
                feature_list.append(torch.tensor([float(value)]))
            elif isinstance(value, list):
                feature_list.append(torch.tensor(value, dtype=torch.float32))

        if feature_list:
            return torch.cat(feature_list)
        else:
            return torch.zeros(1)

    def _update_preferences(self,
                           user_id: str,
                           features: Dict[str, Any],
                           feedback: float):
        """선호도 업데이트"""
        for key, value in features.items():
            if isinstance(value, (int, float)):
                # 가중 평균으로 선호도 업데이트
                current_pref = self.user_preferences[user_id][key]
                self.user_preferences[user_id][key] = (
                    0.9 * current_pref + 0.1 * feedback * float(value)
                )

    def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """사용자 선호도 조회"""
        return dict(self.user_preferences[user_id])

    def get_similar_users(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """유사한 사용자 찾기"""
        if user_id not in self.user_profiles:
            return []

        target_profile = self.user_profiles[user_id]
        similarities = []

        for other_user_id, other_profile in self.user_profiles.items():
            if other_user_id != user_id:
                # 코사인 유사도 계산
                similarity = torch.cosine_similarity(
                    target_profile.unsqueeze(0),
                    other_profile.unsqueeze(0)
                ).item()
                similarities.append((other_user_id, similarity))

        # 유사도 기준 정렬
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class MetaLearner(nn.Module):
    """메타 학습 모듈 - 빠른 적응을 위한 메타 파라미터 학습"""

    def __init__(self, model_dim: int = 1024, meta_dim: int = 256):
        super().__init__()
        self.model_dim = model_dim
        self.meta_dim = meta_dim

        # 메타 파라미터 네트워크
        self.meta_network = nn.Sequential(
            nn.Linear(model_dim, meta_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(meta_dim * 2, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, model_dim)
        )

        # 적응 속도 조절 네트워크
        self.adaptation_rate_network = nn.Sequential(
            nn.Linear(model_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, 1),
            nn.Sigmoid()
        )

        # 작업 특성 인코더
        self.task_encoder = nn.Sequential(
            nn.Linear(model_dim, meta_dim),
            nn.ReLU(),
            nn.Linear(meta_dim, meta_dim // 2)
        )

    def forward(self, task_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        메타 학습 전방 전파

        Args:
            task_embedding: 작업 임베딩

        Returns:
            meta_params: 메타 파라미터
            adaptation_rate: 적응 속도
        """
        meta_params = self.meta_network(task_embedding)
        adaptation_rate = self.adaptation_rate_network(task_embedding)

        return meta_params, adaptation_rate

    def compute_task_similarity(self,
                               task1_embedding: torch.Tensor,
                               task2_embedding: torch.Tensor) -> torch.Tensor:
        """작업 간 유사도 계산"""
        encoded1 = self.task_encoder(task1_embedding)
        encoded2 = self.task_encoder(task2_embedding)

        return torch.cosine_similarity(encoded1, encoded2, dim=-1)

class AdaptiveLearningSystem:
    """실시간 적응형 학습 시스템"""

    def __init__(self,
                 base_model: nn.Module,
                 config: AdaptiveLearningConfig):
        self.base_model = base_model
        self.config = config

        # 적응형 모델 (베이스 모델의 복사본)
        self.adaptive_model = copy.deepcopy(base_model)

        # 경험 버퍼
        self.experience_buffer = ExperienceBuffer(config.max_experience_buffer_size)

        # 사용자 프로필 관리자
        self.user_manager = UserProfileManager(config.user_profile_dim)

        # 메타 학습 모듈
        self.meta_learner = MetaLearner()

        # 옵티마이저들
        self.base_optimizer = optim.AdamW(
            self.adaptive_model.parameters(),
            lr=config.base_learning_rate,
            weight_decay=0.01
        )

        self.meta_optimizer = optim.AdamW(
            self.meta_learner.parameters(),
            lr=config.meta_learning_rate
        )

        # 학습 상태 추적
        self.update_count = 0
        self.last_meta_update = 0
        self.performance_history = deque(maxlen=1000)

        # 스레드 안전성을 위한 락
        self.update_lock = threading.Lock()

        # 실시간 학습을 위한 큐
        self.feedback_queue = queue.Queue()
        self.is_learning = False

        # 데이터베이스 연결 (피드백 영속성)
        self.db_path = "adaptive_learning.db"
        self._init_database()

        logger.info("적응형 학습 시스템 초기화 완료")

    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                input_data TEXT,
                output_data TEXT,
                feedback_score REAL,
                timestamp DATETIME,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                profile_data TEXT,
                last_updated DATETIME
            )
        ''')

        conn.commit()
        conn.close()

    def add_feedback(self,
                    input_data: Dict[str, torch.Tensor],
                    output_data: Dict[str, torch.Tensor],
                    feedback_score: float,
                    user_id: str = "anonymous"):
        """피드백 추가"""
        # 경험 버퍼에 추가
        self.experience_buffer.add_experience(
            input_data, output_data, feedback_score, user_id
        )

        # 사용자 프로필 업데이트
        fragrance_features = self._extract_fragrance_features(output_data)
        self.user_manager.update_user_profile(
            user_id, fragrance_features, feedback_score
        )

        # 데이터베이스에 저장
        self._save_feedback_to_db(input_data, output_data, feedback_score, user_id)

        # 실시간 학습 큐에 추가
        self.feedback_queue.put({
            'input_data': input_data,
            'output_data': output_data,
            'feedback_score': feedback_score,
            'user_id': user_id
        })

        logger.debug(f"피드백 추가됨: 사용자={user_id}, 점수={feedback_score}")

    def _extract_fragrance_features(self, output_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """출력 데이터에서 향수 특성 추출"""
        features = {}

        for key, value in output_data.items():
            if isinstance(value, torch.Tensor):
                if 'note' in key.lower():
                    # 향료 노트 정보
                    features[f'{key}_mean'] = torch.mean(value).item()
                    features[f'{key}_std'] = torch.std(value).item()
                elif 'concentration' in key.lower():
                    # 농도 정보
                    features[key] = torch.mean(value).item()
                elif 'quality' in key.lower() or 'score' in key.lower():
                    # 품질 점수
                    features[key] = torch.mean(value).item()

        return features

    def _save_feedback_to_db(self,
                            input_data: Dict[str, torch.Tensor],
                            output_data: Dict[str, torch.Tensor],
                            feedback_score: float,
                            user_id: str):
        """피드백을 데이터베이스에 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # 텐서 데이터를 직렬화
            input_serialized = pickle.dumps(input_data)
            output_serialized = pickle.dumps(output_data)

            cursor.execute('''
                INSERT INTO feedback_history
                (user_id, input_data, output_data, feedback_score, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, input_serialized, output_serialized, feedback_score, datetime.now()))

            conn.commit()
        except Exception as e:
            logger.error(f"데이터베이스 저장 오류: {e}")
        finally:
            conn.close()

    def start_adaptive_learning(self):
        """적응형 학습 시작"""
        if self.is_learning:
            logger.warning("적응형 학습이 이미 실행 중입니다.")
            return

        self.is_learning = True
        learning_thread = threading.Thread(target=self._adaptive_learning_loop)
        learning_thread.daemon = True
        learning_thread.start()

        logger.info("적응형 학습 시작됨")

    def stop_adaptive_learning(self):
        """적응형 학습 중지"""
        self.is_learning = False
        logger.info("적응형 학습 중지됨")

    def _adaptive_learning_loop(self):
        """적응형 학습 메인 루프"""
        while self.is_learning:
            try:
                # 피드백 큐에서 데이터 가져오기 (타임아웃 1초)
                feedback_batch = []

                try:
                    # 첫 번째 피드백 대기
                    first_feedback = self.feedback_queue.get(timeout=1.0)
                    feedback_batch.append(first_feedback)

                    # 추가 피드백들을 배치로 수집 (최대 batch_size까지)
                    for _ in range(self.config.batch_size - 1):
                        try:
                            feedback = self.feedback_queue.get_nowait()
                            feedback_batch.append(feedback)
                        except queue.Empty:
                            break

                except queue.Empty:
                    continue

                # 배치가 최소 크기를 만족하면 학습 수행
                if len(feedback_batch) >= self.config.min_samples_for_update:
                    self._perform_adaptive_update(feedback_batch)

            except Exception as e:
                logger.error(f"적응형 학습 루프 오류: {e}")
                time.sleep(1)

    def _perform_adaptive_update(self, feedback_batch: List[Dict]):
        """적응형 업데이트 수행"""
        with self.update_lock:
            try:
                # 온라인 학습
                self._online_update(feedback_batch)

                # 경험 재생
                if np.random.random() < self.config.experience_replay_ratio:
                    self._experience_replay()

                # 메타 학습 (주기적으로)
                if (self.update_count - self.last_meta_update) >= self.config.meta_update_steps:
                    self._meta_learning_update()
                    self.last_meta_update = self.update_count

                self.update_count += 1

                # 주기적 정리
                if self.update_count % self.config.experience_cleanup_interval == 0:
                    self.experience_buffer.cleanup_old_experiences()

                # 모델 체크포인트
                if self.update_count % self.config.model_checkpoint_interval == 0:
                    self._save_checkpoint()

            except Exception as e:
                logger.error(f"적응형 업데이트 오류: {e}")

    def _online_update(self, feedback_batch: List[Dict]):
        """온라인 학습 업데이트"""
        self.adaptive_model.train()

        total_loss = 0.0
        batch_size = len(feedback_batch)

        for feedback_item in feedback_batch:
            input_data = feedback_item['input_data']
            output_data = feedback_item['output_data']
            feedback_score = feedback_item['feedback_score']
            user_id = feedback_item['user_id']

            # 사용자 프로필 획득
            user_profile = self.user_manager.get_user_profile(user_id)

            # 모델 예측
            with torch.no_grad():
                predicted = self.adaptive_model(**input_data)

            # 피드백 기반 손실 계산
            loss = self._compute_feedback_loss(predicted, output_data, feedback_score)

            # 사용자 개인화 손실
            personalization_loss = self._compute_personalization_loss(
                predicted, user_profile, feedback_score
            )

            # 전체 손실
            total_loss += loss + 0.1 * personalization_loss

        # 평균 손실
        avg_loss = total_loss / batch_size

        # 백프로퍼게이션
        self.base_optimizer.zero_grad()
        avg_loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(
            self.adaptive_model.parameters(),
            max_norm=1.0
        )

        self.base_optimizer.step()

        # 성능 히스토리 업데이트
        self.performance_history.append(avg_loss.item())

        logger.debug(f"온라인 업데이트 완료: 손실={avg_loss.item():.4f}")

    def _compute_feedback_loss(self,
                              predicted: Dict[str, torch.Tensor],
                              target: Dict[str, torch.Tensor],
                              feedback_score: float) -> torch.Tensor:
        """피드백 기반 손실 계산"""
        total_loss = torch.tensor(0.0, requires_grad=True)

        # 피드백 가중치 계산
        if feedback_score > 0:
            weight = self.config.positive_feedback_bonus * feedback_score
        else:
            weight = self.config.negative_feedback_penalty * abs(feedback_score)

        for key in predicted:
            if key in target:
                if 'logits' in key:
                    # 분류 손실
                    loss = nn.CrossEntropyLoss()(predicted[key], target[key])
                else:
                    # 회귀 손실
                    loss = nn.MSELoss()(predicted[key], target[key])

                total_loss = total_loss + weight * loss

        return total_loss

    def _compute_personalization_loss(self,
                                     predicted: Dict[str, torch.Tensor],
                                     user_profile: torch.Tensor,
                                     feedback_score: float) -> torch.Tensor:
        """개인화 손실 계산"""
        # 예측 결과를 사용자 프로필과 비교
        if 'last_hidden_state' in predicted:
            hidden_state = predicted['last_hidden_state']
            # 평균 풀링으로 시퀀스 차원 축소
            hidden_mean = torch.mean(hidden_state, dim=1)

            # 사용자 프로필과의 유사도
            similarity = torch.cosine_similarity(
                hidden_mean.mean(dim=0, keepdim=True),
                user_profile.unsqueeze(0)
            )

            # 피드백이 긍정적이면 유사도를 높이고, 부정적이면 낮춘다
            if feedback_score > 0:
                return 1.0 - similarity  # 유사도를 높이는 방향
            else:
                return similarity  # 유사도를 낮추는 방향

        return torch.tensor(0.0)

    def _experience_replay(self):
        """경험 재생"""
        if len(self.experience_buffer.experiences) < self.config.meta_batch_size:
            return

        # 경험 샘플링
        replay_experiences = self.experience_buffer.sample_batch(
            self.config.meta_batch_size,
            priority_sampling=True
        )

        self.adaptive_model.train()
        total_replay_loss = 0.0

        for exp in replay_experiences:
            input_data = exp['input']
            output_data = exp['output']
            feedback_score = exp['feedback']

            # 시간 가중치 적용 (최근 경험일수록 높은 가중치)
            time_diff = datetime.now() - exp['timestamp']
            time_weight = np.exp(-time_diff.total_seconds() / 3600)  # 1시간 반감기

            with torch.no_grad():
                predicted = self.adaptive_model(**input_data)

            loss = self._compute_feedback_loss(predicted, output_data, feedback_score)
            total_replay_loss += time_weight * loss

        # 평균 손실
        avg_replay_loss = total_replay_loss / len(replay_experiences)

        # 백프로퍼게이션
        self.base_optimizer.zero_grad()
        avg_replay_loss.backward()
        self.base_optimizer.step()

        logger.debug(f"경험 재생 완료: 손실={avg_replay_loss.item():.4f}")

    def _meta_learning_update(self):
        """메타 학습 업데이트"""
        if len(self.experience_buffer.experiences) < self.config.meta_batch_size * 2:
            return

        # 다양한 작업 샘플링
        task_samples = self._sample_diverse_tasks()

        self.meta_learner.train()
        meta_loss = 0.0

        for task_data in task_samples:
            # 작업 임베딩 생성
            task_embedding = self._create_task_embedding(task_data)

            # 메타 파라미터 예측
            meta_params, adaptation_rate = self.meta_learner(task_embedding)

            # 빠른 적응 시뮬레이션
            adapted_performance = self._simulate_fast_adaptation(
                task_data, meta_params, adaptation_rate
            )

            # 메타 손실 계산
            target_performance = self._compute_target_performance(task_data)
            meta_loss += nn.MSELoss()(adapted_performance, target_performance)

        # 메타 옵티마이저 업데이트
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        logger.debug(f"메타 학습 완료: 손실={meta_loss.item():.4f}")

    def _sample_diverse_tasks(self) -> List[Dict]:
        """다양한 작업 샘플링"""
        experiences = list(self.experience_buffer.experiences)

        # 사용자별로 그룹화
        user_groups = defaultdict(list)
        for exp in experiences:
            user_groups[exp['user_id']].append(exp)

        # 각 사용자 그룹에서 샘플링
        task_samples = []
        for user_id, user_experiences in user_groups.items():
            if len(user_experiences) >= 2:
                sample = np.random.choice(user_experiences, size=2, replace=False)
                task_samples.append({
                    'user_id': user_id,
                    'experiences': sample.tolist()
                })

        return task_samples[:self.config.meta_batch_size]

    def _create_task_embedding(self, task_data: Dict) -> torch.Tensor:
        """작업 임베딩 생성"""
        user_id = task_data['user_id']
        experiences = task_data['experiences']

        # 사용자 프로필
        user_profile = self.user_manager.get_user_profile(user_id)

        # 경험 특성 추출
        exp_features = []
        for exp in experiences:
            feedback = exp['feedback']
            exp_features.append(torch.tensor([feedback]))

        if exp_features:
            exp_tensor = torch.cat(exp_features)
            exp_mean = torch.mean(exp_tensor)
            exp_std = torch.std(exp_tensor)
        else:
            exp_mean = torch.tensor(0.0)
            exp_std = torch.tensor(1.0)

        # 작업 임베딩 구성
        task_embedding = torch.cat([
            user_profile[:user_profile.size(0)//2],  # 사용자 프로필 일부
            torch.tensor([exp_mean, exp_std, len(experiences)])
        ])

        # 고정 크기로 패딩
        if task_embedding.size(0) < 1024:
            padding = torch.zeros(1024 - task_embedding.size(0))
            task_embedding = torch.cat([task_embedding, padding])
        else:
            task_embedding = task_embedding[:1024]

        return task_embedding

    def _simulate_fast_adaptation(self,
                                 task_data: Dict,
                                 meta_params: torch.Tensor,
                                 adaptation_rate: torch.Tensor) -> torch.Tensor:
        """빠른 적응 시뮬레이션"""
        # 간단한 성능 예측 모델
        user_id = task_data['user_id']
        experiences = task_data['experiences']

        # 과거 성능 기반 예측
        feedback_scores = [exp['feedback'] for exp in experiences]
        avg_feedback = np.mean(feedback_scores)

        # 메타 파라미터와 적응 속도를 고려한 성능 예측
        predicted_performance = torch.tensor(avg_feedback) + 0.1 * torch.tanh(meta_params.mean()) * adaptation_rate

        return predicted_performance.squeeze()

    def _compute_target_performance(self, task_data: Dict) -> torch.Tensor:
        """목표 성능 계산"""
        experiences = task_data['experiences']
        feedback_scores = [exp['feedback'] for exp in experiences]

        # 최근 피드백의 가중 평균
        weights = np.exp(np.arange(len(feedback_scores)))
        target = np.average(feedback_scores, weights=weights)

        return torch.tensor(target)

    def get_personalized_model(self, user_id: str) -> nn.Module:
        """개인화된 모델 반환"""
        # 사용자 프로필을 기반으로 모델 조정
        user_profile = self.user_manager.get_user_profile(user_id)

        # 사용자별 경험 조회
        user_experiences = self.experience_buffer.get_user_experiences(user_id)

        if len(user_experiences) < 5:
            # 경험이 부족한 경우 기본 적응 모델 반환
            return self.adaptive_model

        # 사용자 특화 파인튜닝
        personalized_model = copy.deepcopy(self.adaptive_model)
        self._fine_tune_for_user(personalized_model, user_experiences)

        return personalized_model

    def _fine_tune_for_user(self, model: nn.Module, user_experiences: List[Dict]):
        """사용자별 파인튜닝"""
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=self.config.adaptive_learning_rate)

        for _ in range(5):  # 5회 파인튜닝
            total_loss = 0.0

            for exp in user_experiences[-10:]:  # 최근 10개 경험만 사용
                input_data = exp['input']
                output_data = exp['output']
                feedback_score = exp['feedback']

                predicted = model(**input_data)
                loss = self._compute_feedback_loss(predicted, output_data, feedback_score)
                total_loss += loss

            if len(user_experiences) > 0:
                avg_loss = total_loss / min(len(user_experiences), 10)

                optimizer.zero_grad()
                avg_loss.backward()
                optimizer.step()

    def _save_checkpoint(self):
        """모델 체크포인트 저장"""
        checkpoint = {
            'adaptive_model_state': self.adaptive_model.state_dict(),
            'meta_learner_state': self.meta_learner.state_dict(),
            'base_optimizer_state': self.base_optimizer.state_dict(),
            'meta_optimizer_state': self.meta_optimizer.state_dict(),
            'update_count': self.update_count,
            'config': self.config
        }

        checkpoint_path = f"adaptive_checkpoint_{self.update_count}.pt"
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"체크포인트 저장됨: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path)

        self.adaptive_model.load_state_dict(checkpoint['adaptive_model_state'])
        self.meta_learner.load_state_dict(checkpoint['meta_learner_state'])
        self.base_optimizer.load_state_dict(checkpoint['base_optimizer_state'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state'])
        self.update_count = checkpoint['update_count']

        logger.info(f"체크포인트 로드됨: {checkpoint_path}")

    def get_learning_statistics(self) -> Dict[str, Any]:
        """학습 통계 조회"""
        stats = {
            'update_count': self.update_count,
            'experience_buffer_size': len(self.experience_buffer.experiences),
            'active_users': len(self.user_manager.user_profiles),
            'average_performance': np.mean(self.performance_history) if self.performance_history else 0.0,
            'recent_performance_trend': np.polyfit(
                range(len(self.performance_history)),
                list(self.performance_history),
                1
            )[0] if len(self.performance_history) > 1 else 0.0,
            'is_learning': self.is_learning
        }

        return stats

def create_adaptive_learning_system(base_model: nn.Module,
                                   config_path: Optional[str] = None) -> AdaptiveLearningSystem:
    """적응형 학습 시스템 팩토리"""

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = AdaptiveLearningConfig(**config_dict)
    else:
        config = AdaptiveLearningConfig()

    system = AdaptiveLearningSystem(base_model, config)

    logger.info("적응형 학습 시스템 생성 완료")
    logger.info(f"설정: {config}")

    return system

if __name__ == "__main__":
    # 테스트용 더미 모델
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(100, 50)

        def forward(self, input_ids, **kwargs):
            return {'logits': self.linear(input_ids.float())}

    # 시스템 테스트
    dummy_model = DummyModel()
    adaptive_system = create_adaptive_learning_system(dummy_model)

    # 더미 피드백 추가
    input_data = {'input_ids': torch.randn(1, 100)}
    output_data = {'logits': torch.randn(1, 50)}

    adaptive_system.add_feedback(input_data, output_data, 0.8, "test_user")

    # 적응형 학습 시작
    adaptive_system.start_adaptive_learning()

    # 몇 초 후 중지
    time.sleep(2)
    adaptive_system.stop_adaptive_learning()

    # 통계 출력
    stats = adaptive_system.get_learning_statistics()
    print("학습 통계:", stats)

    print("실시간 적응형 학습 시스템 테스트 완료!")