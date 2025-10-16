"""
간단한 향수 추천 신경망 모델 - 실제 학습 가능
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FragranceDataset(Dataset):
    """향수 데이터셋"""

    def __init__(self, data_path: Optional[str] = None):
        """
        데이터셋 초기화
        data_path가 없으면 샘플 데이터 자동 생성
        """
        if data_path and Path(data_path).exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            # 샘플 데이터 생성
            self.data = self._generate_sample_data()

        # 특징 인코딩을 위한 매핑
        self.families = ['fresh', 'floral', 'woody', 'oriental', 'citrus', 'aromatic']
        self.moods = ['romantic', 'energetic', 'elegant', 'mysterious', 'casual', 'sophisticated']
        self.seasons = ['spring', 'summer', 'fall', 'winter', 'all']
        self.intensities = ['light', 'moderate', 'strong']

        # 노트 어휘
        self.note_vocab = self._build_note_vocab()

    def _generate_sample_data(self) -> List[Dict]:
        """학습용 샘플 데이터 생성"""
        sample_data = []

        # 100개의 샘플 향수 데이터 생성
        for i in range(100):
            sample = {
                'id': f'sample_{i}',
                'family': np.random.choice(['fresh', 'floral', 'woody', 'oriental', 'citrus']),
                'mood': np.random.choice(['romantic', 'energetic', 'elegant', 'mysterious', 'casual']),
                'season': np.random.choice(['spring', 'summer', 'fall', 'winter', 'all']),
                'intensity': np.random.choice(['light', 'moderate', 'strong']),
                'top_notes': np.random.choice(['bergamot', 'lemon', 'orange', 'grapefruit'], size=2).tolist(),
                'heart_notes': np.random.choice(['rose', 'jasmine', 'lavender', 'geranium'], size=2).tolist(),
                'base_notes': np.random.choice(['musk', 'amber', 'vanilla', 'sandalwood'], size=2).tolist(),
                'rating': np.random.uniform(3.0, 5.0)  # 타겟 값 (평점)
            }
            sample_data.append(sample)

        return sample_data

    def _build_note_vocab(self) -> Dict[str, int]:
        """노트 어휘 구축"""
        all_notes = set()
        for item in self.data:
            all_notes.update(item.get('top_notes', []))
            all_notes.update(item.get('heart_notes', []))
            all_notes.update(item.get('base_notes', []))

        return {note: idx for idx, note in enumerate(sorted(all_notes))}

    def encode_features(self, item: Dict) -> torch.Tensor:
        """특징 인코딩"""
        features = []

        # One-hot encoding for categorical features
        family_vec = [0] * len(self.families)
        if item['family'] in self.families:
            family_vec[self.families.index(item['family'])] = 1
        features.extend(family_vec)

        mood_vec = [0] * len(self.moods)
        if item['mood'] in self.moods:
            mood_vec[self.moods.index(item['mood'])] = 1
        features.extend(mood_vec)

        season_vec = [0] * len(self.seasons)
        if item['season'] in self.seasons:
            season_vec[self.seasons.index(item['season'])] = 1
        features.extend(season_vec)

        intensity_vec = [0] * len(self.intensities)
        if item['intensity'] in self.intensities:
            intensity_vec[self.intensities.index(item['intensity'])] = 1
        features.extend(intensity_vec)

        # Multi-hot encoding for notes
        note_vec = [0] * len(self.note_vocab)
        for note in item.get('top_notes', []) + item.get('heart_notes', []) + item.get('base_notes', []):
            if note in self.note_vocab:
                note_vec[self.note_vocab[note]] = 1
        features.extend(note_vec)

        return torch.FloatTensor(features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = self.encode_features(item)
        target = torch.FloatTensor([item['rating']])
        return features, target


class SimpleFragranceNN(nn.Module):
    """간단한 향수 추천 신경망"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(SimpleFragranceNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 출력을 0-1 범위로

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) * 5.0  # 0-5 평점 범위로 스케일링


class FragranceModelTrainer:
    """향수 모델 트레이너 - 실제 학습 구현"""

    def __init__(self, model: nn.Module, dataset: FragranceDataset):
        self.model = model
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Optimizer와 Loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # 학습 기록
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader) -> float:
        """한 에폭 학습"""
        self.model.train()
        epoch_loss = 0.0

        for batch_idx, (features, targets) in enumerate(dataloader):
            features, targets = features.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(f'Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

        return epoch_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> float:
        """검증"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, targets in dataloader:
                features, targets = features.to(self.device), targets.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(dataloader)

    def train(self, epochs: int = 50, batch_size: int = 16, val_split: float = 0.2):
        """전체 학습 프로세스"""
        # 데이터 분할
        dataset_size = len(self.dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"Starting training on {self.device}")
        logger.info(f"Train size: {train_size}, Validation size: {val_size}")

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")

            # 학습
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # 검증
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
                logger.info("✓ Best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info("\nTraining completed!")
        return self.train_losses, self.val_losses

    def save_model(self, path: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")

    def predict(self, features: Dict) -> float:
        """예측"""
        self.model.eval()
        with torch.no_grad():
            # 특징 인코딩
            encoded_features = self.dataset.encode_features(features)
            encoded_features = encoded_features.unsqueeze(0).to(self.device)

            # 예측
            output = self.model(encoded_features)
            return output.item()


def main():
    """메인 실행 함수"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # 데이터셋 생성
    dataset = FragranceDataset()

    # 입력 차원 계산
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]

    # 모델 생성
    model = SimpleFragranceNN(input_dim=input_dim, hidden_dims=[128, 64, 32])

    # 트레이너 생성
    trainer = FragranceModelTrainer(model, dataset)

    # 학습
    logger.info("Starting real training...")
    train_losses, val_losses = trainer.train(epochs=30, batch_size=16)

    # 테스트 예측
    test_fragrance = {
        'family': 'fresh',
        'mood': 'energetic',
        'season': 'summer',
        'intensity': 'light',
        'top_notes': ['bergamot', 'lemon'],
        'heart_notes': ['lavender'],
        'base_notes': ['musk']
    }

    prediction = trainer.predict(test_fragrance)
    logger.info(f"\nTest prediction for {test_fragrance['family']} fragrance: {prediction:.2f}/5.0")


if __name__ == "__main__":
    main()