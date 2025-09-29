"""
Living Scent Training Pipeline
각 AI 에이전트를 위한 통합 훈련 파이프라인

이 모듈은 Living Scent의 각 AI 에이전트를
적절한 옵티마이저와 연결하고 훈련시킵니다.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Any, Optional
import json
import logging
from pathlib import Path
from datetime import datetime

# Living Scent AI 에이전트들
from fragrance_ai.models.living_scent.linguistic_receptor import get_linguistic_receptor
from fragrance_ai.models.living_scent.cognitive_core import get_cognitive_core
from fragrance_ai.models.living_scent.olfactory_recombinator import get_olfactory_recombinator
from fragrance_ai.models.living_scent.epigenetic_variation import get_epigenetic_variation

# 옵티마이저들
from fragrance_ai.training.living_scent_optimizers import (
    NeuralNetworkOptimizer,
    AdamWConfig,
    NSGAIII,
    PPO_RLHF,
    get_optimizer_manager
)

logger = logging.getLogger(__name__)


# ============================================================================
# 1. LinguisticReceptor 훈련 파이프라인 (AdamW)
# ============================================================================

class LinguisticDataset(Dataset):
    """언어 수용체 훈련용 데이터셋"""

    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text': item['text'],
            'intent': item['intent'],
            'keywords': item['keywords']
        }


class LinguisticReceptorTrainer:
    """
    LinguisticReceptorAI 훈련 클래스
    AdamW 옵티마이저로 언어 이해 능력 향상
    """

    def __init__(self, model_path: Optional[str] = None):
        self.receptor = get_linguistic_receptor()

        # 신경망 모델 래퍼 (훈련 가능하도록)
        self.model = self._create_trainable_model()

        # AdamW 옵티마이저 설정
        config = AdamWConfig(
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            scheduler_type="cosine"
        )
        self.optimizer = NeuralNetworkOptimizer(self.model, config)

        # 옵티마이저 매니저에 등록
        manager = get_optimizer_manager()
        manager.register_neural_optimizer("linguistic_receptor", self.model, config)

    def _create_trainable_model(self) -> nn.Module:
        """훈련 가능한 신경망 모델 생성"""
        class LinguisticModel(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=256, output_dim=3):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                self.intent_classifier = nn.Linear(hidden_dim, output_dim)
                self.keyword_extractor = nn.Linear(hidden_dim, 100)  # 100개 키워드 후보

            def forward(self, embeddings):
                features = self.encoder(embeddings)
                intent_logits = self.intent_classifier(features)
                keyword_logits = self.keyword_extractor(features)
                return intent_logits, keyword_logits

        return LinguisticModel()

    def train(self, train_data_path: str, epochs: int = 10, batch_size: int = 32):
        """
        모델 훈련

        Args:
            train_data_path: 훈련 데이터 경로
            epochs: 에포크 수
            batch_size: 배치 크기
        """
        # 데이터 로드
        dataset = LinguisticDataset(train_data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 손실 함수
        intent_loss_fn = nn.CrossEntropyLoss()
        keyword_loss_fn = nn.BCEWithLogitsLoss()

        logger.info(f"Starting LinguisticReceptor training for {epochs} epochs")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                # 텍스트를 임베딩으로 변환 (실제로는 BERT 등 사용)
                embeddings = torch.randn(batch_size, 768)  # 더미 임베딩

                # 정답 레이블
                intent_labels = torch.randint(0, 3, (batch_size,))
                keyword_labels = torch.randint(0, 2, (batch_size, 100)).float()

                # Forward pass
                intent_logits, keyword_logits = self.model(embeddings)

                # 손실 계산
                intent_loss = intent_loss_fn(intent_logits, intent_labels)
                keyword_loss = keyword_loss_fn(keyword_logits, keyword_labels)
                total_loss = intent_loss + 0.5 * keyword_loss

                # 최적화 스텝
                loss = self.optimizer.train_step(
                    embeddings,
                    (intent_labels, keyword_labels),
                    lambda outputs, labels: total_loss
                )

                epoch_loss += loss

            avg_loss = epoch_loss / len(dataloader)
            current_lr = self.optimizer.get_current_lr()

            logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.2e}")

        # 체크포인트 저장
        self.optimizer.save_checkpoint("checkpoints/linguistic_receptor.pt")


# ============================================================================
# 2. OlfactoryRecombinator 훈련 파이프라인 (NSGA-III)
# ============================================================================

class OlfactoryRecombinatorTrainer:
    """
    OlfactoryRecombinatorAI 훈련 클래스
    NSGA-III로 다목적 최적화 수행
    """

    def __init__(self):
        self.recombinator = get_olfactory_recombinator()

        # NSGA-III 옵티마이저 설정
        self.optimizer = NSGAIII(
            population_size=100,
            num_generations=50,
            crossover_prob=0.9,
            mutation_prob=0.1
        )

        # 옵티마이저 매니저에 등록
        manager = get_optimizer_manager()
        manager.register_genetic_optimizer(
            "olfactory_recombinator",
            population_size=100,
            num_generations=50
        )

    def optimize_fragrance_creation(self, user_requirements: Dict[str, float]):
        """
        향수 생성 최적화

        Args:
            user_requirements: 사용자 요구사항
                예: {'fresh': 0.8, 'woody': 0.6, 'lasting': 0.7}

        Returns:
            best_fragrances: 파레토 최적 향수들
        """
        logger.info("Starting NSGA-III optimization for fragrance creation")

        # 최적화 실행
        pareto_front = self.optimizer.optimize(user_requirements)

        logger.info(f"Optimization complete. Found {len(pareto_front)} Pareto optimal solutions")

        # 최적 향수들을 DNA로 변환
        best_fragrances = []
        for fragrance in pareto_front[:5]:  # 상위 5개
            # DNA 생성 (실제 구현에서는 더 정교하게)
            dna_data = {
                'genes': fragrance.genes,
                'harmony': fragrance.objectives['harmony'],
                'uniqueness': fragrance.objectives['uniqueness'],
                'user_fitness': fragrance.objectives['user_fitness']
            }
            best_fragrances.append(dna_data)

        return best_fragrances

    def evaluate_performance(self):
        """최적화 성능 평가"""
        # 진화 히스토리 분석
        history = self.optimizer.evolution_history

        if history:
            final_gen = history[-1]
            logger.info(f"Final generation performance:")
            logger.info(f"  Best Harmony: {final_gen['best_harmony']:.3f}")
            logger.info(f"  Best Uniqueness: {final_gen['best_uniqueness']:.3f}")
            logger.info(f"  Best User Fitness: {final_gen['best_user_fitness']:.3f}")

        return history


# ============================================================================
# 3. EpigeneticVariation 훈련 파이프라인 (PPO-RLHF)
# ============================================================================

class EpigeneticVariationTrainer:
    """
    EpigeneticVariationAI 훈련 클래스
    PPO-RLHF로 인간 피드백 기반 학습
    """

    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.variation_ai = get_epigenetic_variation()

        # PPO-RLHF 옵티마이저 설정
        self.optimizer = PPO_RLHF(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=3e-4,
            gamma=0.99,
            epsilon=0.2
        )

        # 옵티마이저 매니저에 등록
        manager = get_optimizer_manager()
        manager.register_rl_optimizer(
            "epigenetic_variation",
            state_dim=state_dim,
            action_dim=action_dim
        )

        self.state_dim = state_dim
        self.action_dim = action_dim

    def train_with_human_feedback(self, num_episodes: int = 100):
        """
        인간 피드백으로 훈련

        Args:
            num_episodes: 훈련 에피소드 수
        """
        logger.info(f"Starting PPO-RLHF training for {num_episodes} episodes")

        for episode in range(num_episodes):
            # 초기 상태 (향수 표현)
            state = np.random.randn(self.state_dim)
            episode_reward = 0

            for step in range(50):  # 최대 50 스텝
                # 행동 선택
                action, log_prob, value = self.optimizer.select_action(state)

                # 환경 시뮬레이션 (실제로는 사용자 피드백)
                next_state = self._apply_variation(state, action)
                reward = self._simulate_human_feedback(state, action)

                # 경험 저장
                from fragrance_ai.training.living_scent_optimizers import Experience
                exp = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=(step == 49),
                    log_prob=log_prob,
                    value=value
                )
                self.optimizer.store_experience(exp)

                # 인간 피드백 학습
                self.optimizer.collect_human_feedback(state, action, reward)

                state = next_state
                episode_reward += reward

            # PPO 업데이트
            if episode % 10 == 0:
                self.optimizer.train(batch_size=64, epochs=10)

            if episode % 20 == 0:
                logger.info(f"Episode {episode} - Reward: {episode_reward:.2f}")

        # 모델 저장
        self.optimizer.save_models("checkpoints/epigenetic_variation.pt")

    def _apply_variation(self, state: np.ndarray, action: int) -> np.ndarray:
        """행동에 따른 상태 변형 시뮬레이션"""
        # 간단한 변형 시뮬레이션
        variation_vectors = {
            0: np.array([0.1, 0, 0] + [0] * (self.state_dim - 3)),  # 증폭
            1: np.array([-0.1, 0, 0] + [0] * (self.state_dim - 3)),  # 약화
            2: np.array([0, 0.1, 0] + [0] * (self.state_dim - 3)),  # 조절
            # ... 더 많은 변형 타입
        }

        if action < len(variation_vectors):
            variation = variation_vectors[action]
        else:
            variation = np.random.randn(self.state_dim) * 0.05

        return state + variation

    def _simulate_human_feedback(self, state: np.ndarray, action: int) -> float:
        """인간 피드백 시뮬레이션"""
        # 실제로는 사용자 입력을 받아야 함
        # 여기서는 간단한 시뮬레이션
        base_reward = -0.1  # 기본 페널티

        # 특정 조건에서 보상
        if action == 0 and state[0] < 0.5:  # 약한 향을 증폭
            base_reward += 0.5
        elif action == 1 and state[0] > 0.8:  # 강한 향을 약화
            base_reward += 0.5

        # 노이즈 추가 (인간 피드백의 불확실성)
        noise = np.random.normal(0, 0.1)

        return np.clip(base_reward + noise, -1, 1)


# ============================================================================
# 통합 훈련 파이프라인
# ============================================================================

class LivingScentTrainingPipeline:
    """
    전체 Living Scent 시스템 통합 훈련 파이프라인
    """

    def __init__(self):
        self.trainers = {}
        self.training_history = {}

    def setup_trainers(self):
        """모든 트레이너 초기화"""
        logger.info("Setting up Living Scent trainers...")

        # 1. 언어 수용체 트레이너
        self.trainers['linguistic'] = LinguisticReceptorTrainer()

        # 2. 후각 재조합 트레이너
        self.trainers['olfactory'] = OlfactoryRecombinatorTrainer()

        # 3. 후생적 변형 트레이너
        self.trainers['epigenetic'] = EpigeneticVariationTrainer()

        logger.info("All trainers initialized successfully")

    def train_all(self, config: Dict[str, Any]):
        """
        모든 AI 에이전트 훈련

        Args:
            config: 훈련 설정
                {
                    'linguistic': {'data_path': '...', 'epochs': 10},
                    'olfactory': {'user_requirements': {...}},
                    'epigenetic': {'num_episodes': 100}
                }
        """
        logger.info("Starting comprehensive Living Scent training...")

        # 1. LinguisticReceptor 훈련 (AdamW)
        if 'linguistic' in config:
            logger.info("Training LinguisticReceptor with AdamW...")
            self.trainers['linguistic'].train(
                train_data_path=config['linguistic']['data_path'],
                epochs=config['linguistic']['epochs']
            )
            self.training_history['linguistic'] = {
                'completed': True,
                'timestamp': datetime.now().isoformat()
            }

        # 2. OlfactoryRecombinator 최적화 (NSGA-III)
        if 'olfactory' in config:
            logger.info("Optimizing OlfactoryRecombinator with NSGA-III...")
            best_fragrances = self.trainers['olfactory'].optimize_fragrance_creation(
                config['olfactory']['user_requirements']
            )
            self.training_history['olfactory'] = {
                'completed': True,
                'best_fragrances': best_fragrances,
                'timestamp': datetime.now().isoformat()
            }

        # 3. EpigeneticVariation 훈련 (PPO-RLHF)
        if 'epigenetic' in config:
            logger.info("Training EpigeneticVariation with PPO-RLHF...")
            self.trainers['epigenetic'].train_with_human_feedback(
                num_episodes=config['epigenetic']['num_episodes']
            )
            self.training_history['epigenetic'] = {
                'completed': True,
                'timestamp': datetime.now().isoformat()
            }

        logger.info("Living Scent training pipeline completed!")
        return self.training_history

    def evaluate_all(self) -> Dict[str, Any]:
        """모든 모델 평가"""
        evaluation_results = {}

        # 각 트레이너의 성능 평가
        for name, trainer in self.trainers.items():
            if hasattr(trainer, 'evaluate_performance'):
                evaluation_results[name] = trainer.evaluate_performance()

        return evaluation_results

    def save_all_models(self, directory: str):
        """모든 모델 저장"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # 옵티마이저 매니저를 통해 저장
        manager = get_optimizer_manager()
        manager.save_all(directory)

        # 훈련 히스토리 저장
        history_path = path / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)

        logger.info(f"All models and history saved to {directory}")


# ============================================================================
# 훈련 실행 스크립트
# ============================================================================

def main():
    """메인 훈련 실행"""
    # 파이프라인 초기화
    pipeline = LivingScentTrainingPipeline()
    pipeline.setup_trainers()

    # 훈련 설정
    training_config = {
        'linguistic': {
            'data_path': 'data/linguistic_training.json',
            'epochs': 5
        },
        'olfactory': {
            'user_requirements': {
                'fresh': 0.8,
                'woody': 0.6,
                'lasting': 0.7
            }
        },
        'epigenetic': {
            'num_episodes': 50
        }
    }

    # 훈련 실행
    history = pipeline.train_all(training_config)

    # 평가
    evaluation = pipeline.evaluate_all()

    # 모델 저장
    pipeline.save_all_models("models/living_scent/")

    # 결과 출력
    print("\n" + "="*60)
    print("Living Scent Training Complete!")
    print("="*60)
    print("\nTraining History:")
    print(json.dumps(history, indent=2))
    print("\nEvaluation Results:")
    print(json.dumps(evaluation, indent=2, default=str))


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 훈련 실행
    main()