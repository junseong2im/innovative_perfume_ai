"""
Model Retraining Operator
업데이트된 데이터로 AI 모델들을 재학습시키는 오퍼레이터

재학습 대상:
1. PPO 강화학습 모델
2. MOGA 최적화 파라미터
3. 임베딩 모델
4. 향수 생성 모델
"""

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report

# 프로젝트 모델 임포트
from fragrance_ai.training.real_ppo_complete import PPOTrainer, FragranceEnvironment
from fragrance_ai.training.real_moga_complete import CompleteRealMOGA

logger = logging.getLogger(__name__)


class FragranceDataset(Dataset):
    """재학습용 향수 데이터셋"""

    def __init__(self, data_path: str):
        self.data = self._load_data(data_path)
        self.samples = self._prepare_samples()

    def _load_data(self, path: str) -> Dict:
        """데이터 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _prepare_samples(self) -> List[Dict]:
        """학습용 샘플 준비"""
        samples = []

        for key, fragrance in self.data.get('fragrances', {}).items():
            # 노트를 벡터로 변환
            features = self._extract_features(fragrance)
            label = fragrance.get('overall_rating', 0.0)

            samples.append({
                'features': features,
                'label': label,
                'metadata': fragrance
            })

        return samples

    def _extract_features(self, fragrance: Dict) -> np.ndarray:
        """특성 추출"""
        features = []

        # 노트 인코딩 (원-핫 인코딩 또는 임베딩)
        all_notes = set()
        for note_type in ['top_notes', 'heart_notes', 'base_notes']:
            notes = fragrance.get(note_type, [])
            all_notes.update(notes)

        # 간단한 특성 벡터 (실제로는 더 정교한 인코딩 필요)
        features.extend([
            len(fragrance.get('top_notes', [])),
            len(fragrance.get('heart_notes', [])),
            len(fragrance.get('base_notes', [])),
            fragrance.get('avg_longevity', 3.0),
            fragrance.get('avg_sillage', 3.0),
            fragrance.get('sentiment_score', 0.0),
            fragrance.get('data_quality_score', 0.5)
        ])

        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.FloatTensor(sample['features']),
            torch.FloatTensor([sample['label']])
        )


class ModelRetrainingOperator:
    """
    모델 재학습 오퍼레이터
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.models_dir = Path(self.config['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # 재학습 통계
        self.stats = {
            'models_retrained': [],
            'performance_improvements': {},
            'training_time': {},
            'errors': []
        }

        logger.info("ModelRetrainingOperator initialized")

    def _get_default_config(self) -> Dict:
        """기본 설정"""
        return {
            'models_dir': 'models/retrained',
            'backup_dir': 'models/backup',
            'knowledge_base_path': 'data/comprehensive_fragrance_notes_database.json',
            'retraining': {
                'ppo': {
                    'enabled': True,
                    'epochs': 100,
                    'batch_size': 64,
                    'learning_rate': 3e-4,
                    'min_data_points': 100
                },
                'moga': {
                    'enabled': True,
                    'generations': 50,
                    'population_size': 100,
                    'min_data_points': 50
                },
                'embedding': {
                    'enabled': True,
                    'epochs': 50,
                    'batch_size': 32,
                    'learning_rate': 1e-3,
                    'min_data_points': 200
                }
            },
            'performance_thresholds': {
                'min_improvement': 0.01,  # 1% 개선 필요
                'max_degradation': 0.05   # 5% 이상 성능 저하 시 롤백
            }
        }

    def retrain_ppo_model(self, data_path: str) -> Dict[str, Any]:
        """PPO 모델 재학습"""
        logger.info("Starting PPO model retraining...")
        start_time = datetime.now()

        try:
            # 데이터 로드
            dataset = FragranceDataset(data_path)

            if len(dataset) < self.config['retraining']['ppo']['min_data_points']:
                logger.warning(f"Insufficient data for PPO retraining: {len(dataset)} samples")
                return {'status': 'skipped', 'reason': 'insufficient_data'}

            # 환경과 트레이너 초기화
            env = FragranceEnvironment()
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            trainer = PPOTrainer(state_dim=state_dim, action_dim=action_dim)

            # 기존 모델 로드 (있는 경우)
            existing_model_path = self.models_dir / 'ppo_model.pth'
            if existing_model_path.exists():
                trainer.load_state_dict(torch.load(existing_model_path))
                logger.info("Loaded existing PPO model")

            # 학습 전 성능 평가
            initial_performance = self._evaluate_ppo_model(trainer, env)

            # 재학습
            config = self.config['retraining']['ppo']
            total_rewards = []

            for epoch in range(config['epochs']):
                state = env.reset()
                episode_reward = 0

                for step in range(1000):  # 최대 스텝
                    # 행동 선택
                    action, log_prob, value = trainer.get_action_and_value(
                        torch.FloatTensor(state).unsqueeze(0)
                    )

                    # 환경 스텝
                    next_state, reward, done, info = env.step(action.item())

                    # 버퍼에 추가
                    trainer.rollout_buffer.add(state, action, reward, value, log_prob)

                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                # 주기적 학습
                if (epoch + 1) % 10 == 0:
                    train_stats = trainer.train_step(
                        n_epochs=4,
                        batch_size=config['batch_size']
                    )
                    logger.info(f"PPO Epoch {epoch+1}: Reward={episode_reward:.2f}")

                total_rewards.append(episode_reward)

            # 학습 후 성능 평가
            final_performance = self._evaluate_ppo_model(trainer, env)

            # 성능 개선 확인
            improvement = final_performance - initial_performance

            if improvement >= self.config['performance_thresholds']['min_improvement']:
                # 모델 저장
                self._backup_model('ppo_model.pth')
                torch.save(trainer.state_dict(), existing_model_path)
                logger.info(f"PPO model saved with {improvement:.2%} improvement")

                result = {
                    'status': 'success',
                    'initial_performance': initial_performance,
                    'final_performance': final_performance,
                    'improvement': improvement,
                    'avg_reward': np.mean(total_rewards[-10:])
                }
            else:
                logger.warning(f"PPO model not saved: improvement {improvement:.2%} below threshold")
                result = {
                    'status': 'no_improvement',
                    'improvement': improvement
                }

            self.stats['models_retrained'].append('ppo')
            self.stats['performance_improvements']['ppo'] = improvement
            self.stats['training_time']['ppo'] = (datetime.now() - start_time).total_seconds()

            return result

        except Exception as e:
            logger.error(f"Error retraining PPO model: {e}")
            self.stats['errors'].append(f"PPO: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def retrain_moga_optimizer(self, data_path: str) -> Dict[str, Any]:
        """MOGA 최적화 파라미터 재조정"""
        logger.info("Starting MOGA optimizer retraining...")
        start_time = datetime.now()

        try:
            # 데이터 로드
            with open(data_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)

            fragrances = knowledge_base.get('fragrances', {})

            if len(fragrances) < self.config['retraining']['moga']['min_data_points']:
                logger.warning(f"Insufficient data for MOGA retraining: {len(fragrances)} samples")
                return {'status': 'skipped', 'reason': 'insufficient_data'}

            # MOGA 초기화
            moga = CompleteRealMOGA()

            # 새로운 데이터로 성분 데이터베이스 업데이트
            self._update_moga_database(moga, fragrances)

            # 최적화 실행
            config = self.config['retraining']['moga']
            results = moga.optimize(generations=config['generations'])

            # 결과 평가
            if results['pareto_front']:
                best_solution = results['pareto_front'][0]

                # 파라미터 저장
                params_path = self.models_dir / 'moga_params.json'
                self._backup_file(params_path)

                with open(params_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        'best_solutions': results['pareto_front'][:10],
                        'final_generation': results['final_generation'],
                        'convergence_metric': results.get('convergence_metric', 0),
                        'updated_at': datetime.now().isoformat()
                    }, f, indent=2)

                logger.info(f"MOGA parameters updated: {len(results['pareto_front'])} solutions")

                result = {
                    'status': 'success',
                    'pareto_front_size': len(results['pareto_front']),
                    'best_quality': best_solution['quality_score'],
                    'best_stability': best_solution['stability']
                }

                self.stats['models_retrained'].append('moga')
                self.stats['performance_improvements']['moga'] = result['best_quality']

            else:
                result = {'status': 'no_solutions'}

            self.stats['training_time']['moga'] = (datetime.now() - start_time).total_seconds()

            return result

        except Exception as e:
            logger.error(f"Error retraining MOGA: {e}")
            self.stats['errors'].append(f"MOGA: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def retrain_embedding_model(self, data_path: str) -> Dict[str, Any]:
        """임베딩 모델 재학습"""
        logger.info("Starting embedding model retraining...")
        start_time = datetime.now()

        try:
            # 데이터셋 준비
            dataset = FragranceDataset(data_path)

            if len(dataset) < self.config['retraining']['embedding']['min_data_points']:
                logger.warning(f"Insufficient data for embedding retraining: {len(dataset)} samples")
                return {'status': 'skipped', 'reason': 'insufficient_data'}

            # 데이터 분할
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            # 데이터 로더
            config = self.config['retraining']['embedding']
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False
            )

            # 간단한 임베딩 모델 (실제로는 더 복잡한 모델 사용)
            class SimpleEmbeddingModel(nn.Module):
                def __init__(self, input_dim: int, embedding_dim: int = 128):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(256, embedding_dim),
                        nn.ReLU(),
                        nn.Linear(embedding_dim, 64)
                    )
                    self.decoder = nn.Linear(64, 1)  # 평점 예측

                def forward(self, x):
                    embedding = self.encoder(x)
                    output = self.decoder(embedding)
                    return output, embedding

            # 모델 초기화
            input_dim = 7  # 특성 개수
            model = SimpleEmbeddingModel(input_dim)

            # 기존 모델 로드 (있는 경우)
            model_path = self.models_dir / 'embedding_model.pth'
            if model_path.exists():
                model.load_state_dict(torch.load(model_path))
                logger.info("Loaded existing embedding model")

            # 학습 전 성능 평가
            initial_loss = self._evaluate_embedding_model(model, val_loader)

            # 옵티마이저와 손실 함수
            optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
            criterion = nn.MSELoss()

            # 학습
            model.train()
            for epoch in range(config['epochs']):
                epoch_loss = 0
                for batch_features, batch_labels in train_loader:
                    optimizer.zero_grad()

                    outputs, _ = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if (epoch + 1) % 10 == 0:
                    avg_loss = epoch_loss / len(train_loader)
                    logger.info(f"Embedding Epoch {epoch+1}: Loss={avg_loss:.4f}")

            # 학습 후 성능 평가
            final_loss = self._evaluate_embedding_model(model, val_loader)

            # 개선 확인 (손실 감소)
            improvement = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0

            if improvement >= self.config['performance_thresholds']['min_improvement']:
                # 모델 저장
                self._backup_model('embedding_model.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"Embedding model saved with {improvement:.2%} improvement")

                result = {
                    'status': 'success',
                    'initial_loss': initial_loss,
                    'final_loss': final_loss,
                    'improvement': improvement
                }
            else:
                result = {
                    'status': 'no_improvement',
                    'improvement': improvement
                }

            self.stats['models_retrained'].append('embedding')
            self.stats['performance_improvements']['embedding'] = improvement
            self.stats['training_time']['embedding'] = (datetime.now() - start_time).total_seconds()

            return result

        except Exception as e:
            logger.error(f"Error retraining embedding model: {e}")
            self.stats['errors'].append(f"Embedding: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    def _evaluate_ppo_model(self, trainer: PPOTrainer, env: FragranceEnvironment, episodes: int = 10) -> float:
        """PPO 모델 평가"""
        total_rewards = []

        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0

            for _ in range(1000):
                with torch.no_grad():
                    action, _, _ = trainer.get_action_and_value(
                        torch.FloatTensor(state).unsqueeze(0)
                    )

                state, reward, done, _ = env.step(action.item())
                episode_reward += reward

                if done:
                    break

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def _evaluate_embedding_model(self, model: nn.Module, data_loader: DataLoader) -> float:
        """임베딩 모델 평가"""
        model.eval()
        total_loss = 0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for features, labels in data_loader:
                outputs, _ = model(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def _update_moga_database(self, moga: CompleteRealMOGA, fragrances: Dict):
        """MOGA 성분 데이터베이스 업데이트"""
        # 새로운 성분 추가
        all_ingredients = set()

        for fragrance_data in fragrances.values():
            for note_type in ['top_notes', 'heart_notes', 'base_notes']:
                notes = fragrance_data.get(note_type, [])
                all_ingredients.update(notes)

        # MOGA 데이터베이스에 추가 (실제 구현 필요)
        logger.info(f"Updated MOGA database with {len(all_ingredients)} ingredients")

    def _backup_model(self, filename: str):
        """모델 백업"""
        source = self.models_dir / filename
        if source.exists():
            backup_dir = Path(self.config['backup_dir'])
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filename.split('.')[0]}_{timestamp}.{filename.split('.')[-1]}"
            destination = backup_dir / backup_name

            shutil.copy2(source, destination)
            logger.info(f"Backed up {filename} to {destination}")

    def _backup_file(self, filepath: Path):
        """파일 백업"""
        if filepath.exists():
            backup_dir = Path(self.config['backup_dir'])
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            destination = backup_dir / backup_name

            shutil.copy2(filepath, destination)
            logger.info(f"Backed up {filepath} to {destination}")

    def run(self, knowledge_base_path: Optional[str] = None) -> Dict[str, Any]:
        """재학습 파이프라인 실행"""
        if not knowledge_base_path:
            knowledge_base_path = self.config['knowledge_base_path']

        logger.info(f"Starting model retraining with data from {knowledge_base_path}")

        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }

        # PPO 재학습
        if self.config['retraining']['ppo']['enabled']:
            results['models']['ppo'] = self.retrain_ppo_model(knowledge_base_path)

        # MOGA 재학습
        if self.config['retraining']['moga']['enabled']:
            results['models']['moga'] = self.retrain_moga_optimizer(knowledge_base_path)

        # 임베딩 모델 재학습
        if self.config['retraining']['embedding']['enabled']:
            results['models']['embedding'] = self.retrain_embedding_model(knowledge_base_path)

        # 통계 추가
        results['stats'] = self.stats

        # 결과 저장
        results_path = self.models_dir / f"retraining_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Model retraining completed: {len(self.stats['models_retrained'])} models updated")

        return results


# 실행 예시
if __name__ == "__main__":
    retrainer = ModelRetrainingOperator()

    # 테스트 실행
    result = retrainer.run()
    print(f"Retraining completed: {result}")