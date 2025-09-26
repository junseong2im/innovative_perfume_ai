"""
향수 조합 검증 모델 학습 스크립트
ScientificValidator를 위한 딥러닝 모델 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "training"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# 디렉토리 생성
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


class FragranceDataset(Dataset):
    """향수 조합 데이터셋"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: 향수 조합 특징 벡터 [n_samples, n_features]
            labels: 검증 점수 [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ValidationModel(nn.Module):
    """향수 조합 검증을 위한 신경망 모델"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(ValidationModel, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 0-1 범위로 정규화

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) * 10.0  # 0-10 점수로 스케일링


class DataGenerator:
    """학습 데이터 생성기"""

    def __init__(self):
        self.note_categories = {
            'citrus': ['bergamot', 'lemon', 'grapefruit', 'orange', 'lime'],
            'floral': ['rose', 'jasmine', 'iris', 'lily', 'ylang'],
            'woody': ['sandalwood', 'cedar', 'oud', 'pine', 'oak'],
            'oriental': ['vanilla', 'amber', 'incense', 'musk', 'benzoin'],
            'fresh': ['mint', 'cucumber', 'water', 'ozone', 'grass'],
            'spicy': ['cinnamon', 'pepper', 'cardamom', 'ginger', 'clove'],
            'fruity': ['apple', 'peach', 'berry', 'pear', 'plum'],
            'green': ['tea', 'leaf', 'basil', 'vetiver', 'galbanum']
        }

        # 좋은 조합 규칙
        self.harmony_rules = {
            'citrus': ['floral', 'woody', 'fresh'],
            'floral': ['citrus', 'oriental', 'fruity'],
            'woody': ['citrus', 'oriental', 'spicy'],
            'oriental': ['floral', 'woody', 'spicy'],
            'fresh': ['citrus', 'green', 'fruity'],
            'spicy': ['woody', 'oriental', 'citrus'],
            'fruity': ['floral', 'fresh', 'oriental'],
            'green': ['fresh', 'citrus', 'woody']
        }

    def encode_fragrance(self, composition: Dict[str, List[str]]) -> np.ndarray:
        """향수 조합을 특징 벡터로 인코딩"""
        # 간단한 원-핫 인코딩 + 카테고리 비율
        feature_dim = len(self.note_categories) * 10  # 각 카테고리당 10차원
        features = np.zeros(feature_dim)

        # 각 노트 카테고리별 존재 비율 계산
        for i, (category, notes) in enumerate(self.note_categories.items()):
            category_count = 0

            for note_type in ['top', 'heart', 'base']:
                if note_type in composition:
                    for note in composition[note_type]:
                        if any(n in note.lower() for n in notes):
                            category_count += 1

            # 카테고리 특징 설정
            start_idx = i * 10
            features[start_idx:start_idx + 10] = np.random.randn(10) * 0.1  # 노이즈
            features[start_idx] = category_count / 10.0  # 정규화된 카운트

        return features

    def calculate_harmony_score(self, composition: Dict[str, List[str]]) -> float:
        """조화도 점수 계산"""
        score = 5.0  # 기본 점수

        # 카테고리별 노트 분류
        categories_present = set()
        for note_type in ['top', 'heart', 'base']:
            if note_type in composition:
                for note in composition[note_type]:
                    for category, notes in self.note_categories.items():
                        if any(n in note.lower() for n in notes):
                            categories_present.add(category)

        # 조화 규칙에 따른 점수 계산
        for category in categories_present:
            if category in self.harmony_rules:
                harmonious = self.harmony_rules[category]
                for other in categories_present:
                    if other != category and other in harmonious:
                        score += 1.0

        # 다양성 보너스
        diversity_bonus = min(len(categories_present) * 0.5, 2.0)
        score += diversity_bonus

        # 균형 점수 (각 단계에 노트가 있는지)
        balance_score = sum([
            1.0 if 'top' in composition and composition['top'] else 0,
            1.0 if 'heart' in composition and composition['heart'] else 0,
            1.0 if 'base' in composition and composition['base'] else 0
        ]) / 3.0
        score += balance_score

        return min(score, 10.0)  # 최대 10점

    def generate_samples(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """학습용 샘플 생성"""
        features_list = []
        scores_list = []

        for _ in range(n_samples):
            # 무작위 조합 생성
            composition = {
                'top': np.random.choice(
                    [note for notes in self.note_categories.values() for note in notes],
                    size=np.random.randint(1, 4),
                    replace=False
                ).tolist(),
                'heart': np.random.choice(
                    [note for notes in self.note_categories.values() for note in notes],
                    size=np.random.randint(2, 5),
                    replace=False
                ).tolist(),
                'base': np.random.choice(
                    [note for notes in self.note_categories.values() for note in notes],
                    size=np.random.randint(1, 3),
                    replace=False
                ).tolist()
            }

            # 특징 추출 및 점수 계산
            features = self.encode_fragrance(composition)
            score = self.calculate_harmony_score(composition)

            # 노이즈 추가 (현실성 증가)
            score += np.random.normal(0, 0.5)
            score = np.clip(score, 0, 10)

            features_list.append(features)
            scores_list.append(score)

        return np.array(features_list), np.array(scores_list)


class Trainer:
    """모델 학습기"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0

        for batch_features, batch_labels in dataloader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device).unsqueeze(1)

            # Forward pass
            outputs = self.model(batch_features)
            loss = self.criterion(outputs, batch_labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """검증"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device).unsqueeze(1)

                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_labels)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: Path
    ):
        """전체 학습 프로세스"""
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10

        for epoch in range(epochs):
            # 학습
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 검증
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # 학습률 조정
            self.scheduler.step(val_loss)

            # 로깅
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # 모델 저장 (최적 모델)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses
                }

                torch.save(checkpoint, save_path)
                logger.info(f"Model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1

                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        return best_val_loss


def main():
    """메인 학습 함수"""
    logger.info("향수 조합 검증 모델 학습 시작")

    # 설정
    config = {
        'n_samples': 50000,
        'test_size': 0.2,
        'val_size': 0.1,
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'hidden_dims': [256, 128, 64, 32]
    }

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # 데이터 생성
    logger.info("데이터 생성 중...")
    generator = DataGenerator()
    X, y = generator.generate_samples(config['n_samples'])

    # 데이터 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config['test_size'], random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config['val_size'], random_state=42
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 데이터 정규화
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Scaler 저장
    scaler_path = MODEL_DIR / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")

    # 데이터셋 생성
    train_dataset = FragranceDataset(X_train, y_train)
    val_dataset = FragranceDataset(X_val, y_val)
    test_dataset = FragranceDataset(X_test, y_test)

    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # 모델 생성
    input_dim = X_train.shape[1]
    model = ValidationModel(input_dim, config['hidden_dims'])
    logger.info(f"Model created with input_dim: {input_dim}")

    # 학습기 생성
    trainer = Trainer(
        model,
        device,
        config['learning_rate'],
        config['weight_decay']
    )

    # 학습
    checkpoint_path = CHECKPOINT_DIR / f"validator_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    best_val_loss = trainer.train(
        train_loader,
        val_loader,
        config['epochs'],
        checkpoint_path
    )

    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")

    # 최종 모델 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 테스트 평가
    model.eval()
    test_loss = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")

    # 최종 모델 저장 (프로덕션용)
    final_model_path = MODEL_DIR / "validator_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': config['hidden_dims'],
        'config': config,
        'test_loss': test_loss
    }, final_model_path)

    logger.info(f"Final model saved to {final_model_path}")

    # 학습 히스토리 저장
    history_path = MODEL_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'config': config,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss
        }, f, indent=2)

    logger.info(f"Training history saved to {history_path}")

    # 샘플 예측 테스트
    logger.info("\n샘플 예측 테스트:")
    model.eval()
    with torch.no_grad():
        sample_composition = {
            'top': ['bergamot', 'lemon'],
            'heart': ['rose', 'jasmine', 'iris'],
            'base': ['sandalwood', 'musk']
        }

        sample_features = generator.encode_fragrance(sample_composition)
        sample_features = scaler.transform(sample_features.reshape(1, -1))
        sample_tensor = torch.FloatTensor(sample_features).to(device)

        prediction = model(sample_tensor).item()
        expected = generator.calculate_harmony_score(sample_composition)

        logger.info(f"조합: {sample_composition}")
        logger.info(f"예측 점수: {prediction:.2f}")
        logger.info(f"기대 점수: {expected:.2f}")

    logger.info("\n학습 완료!")


if __name__ == "__main__":
    main()