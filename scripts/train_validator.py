"""
딥러닝 검증 모델 훈련 스크립트
- NeuralBlendingPredictor 모델 훈련
- 조화도, 안정성, 지속성 예측 학습
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import argparse
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 설정
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fragrance_ai.models.advanced_blending_ai import NeuralBlendingPredictor, AdvancedBlendingAI


class ValidatorTrainer:
    """검증 모델 훈련기"""

    def __init__(self, config_path: str = "configs/local.json"):
        """훈련기 초기화"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # 경로 설정
        self.model_path = Path(self.config['trained_model_path'])
        self.scaler_path = Path(self.config['scaler_path'])
        self.data_path = Path(self.config['training_data_path'])

        # 디렉토리 생성
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config['deep_learning_validator']

    def _generate_synthetic_data(self, n_samples: int = 1000) -> Dict[str, Any]:
        """합성 훈련 데이터 생성"""
        logger.info(f"Generating {n_samples} synthetic training samples...")

        data = {
            "recipes": []
        }

        # 노트 옵션
        top_notes = ["Bergamot", "Lemon", "Orange", "Grapefruit", "Mint", "Basil"]
        heart_notes = ["Rose", "Jasmine", "Lavender", "Geranium", "Iris", "Violet"]
        base_notes = ["Sandalwood", "Cedar", "Musk", "Amber", "Vanilla", "Patchouli"]

        np.random.seed(42)  # 재현성을 위한 시드

        for i in range(n_samples):
            # 랜덤하게 노트 선택
            n_top = np.random.randint(1, 4)
            n_heart = np.random.randint(1, 4)
            n_base = np.random.randint(1, 4)

            selected_top = np.random.choice(top_notes, n_top, replace=False).tolist()
            selected_heart = np.random.choice(heart_notes, n_heart, replace=False).tolist()
            selected_base = np.random.choice(base_notes, n_base, replace=False).tolist()

            # 퍼센티지 생성 (합이 100이 되도록)
            top_pct = np.random.dirichlet(np.ones(n_top)) * 30
            heart_pct = np.random.dirichlet(np.ones(n_heart)) * 40
            base_pct = np.random.dirichlet(np.ones(n_base)) * 30

            ingredients = []

            for note, pct in zip(selected_top, top_pct):
                ingredients.append({
                    "name": note,
                    "percentage": float(pct),
                    "category": "top"
                })

            for note, pct in zip(selected_heart, heart_pct):
                ingredients.append({
                    "name": note,
                    "percentage": float(pct),
                    "category": "heart"
                })

            for note, pct in zip(selected_base, base_pct):
                ingredients.append({
                    "name": note,
                    "percentage": float(pct),
                    "category": "base"
                })

            # 점수 생성 (노트 수와 밸런스에 기반)
            total_notes = n_top + n_heart + n_base
            balance = 1.0 - abs(n_top - n_heart) * 0.1 - abs(n_heart - n_base) * 0.1

            # 기본 점수
            harmony = min(10, 5 + balance * 3 + np.random.normal(0, 1))
            stability = min(10, 6 + balance * 2 + np.random.normal(0, 0.8))
            longevity = min(10, 7 + (n_base / total_notes) * 3 + np.random.normal(0, 0.7))
            sillage = min(10, 6 + (n_top / total_notes) * 2 + np.random.normal(0, 0.9))

            # 0-10 범위로 클리핑
            harmony = max(0, min(10, harmony))
            stability = max(0, min(10, stability))
            longevity = max(0, min(10, longevity))
            sillage = max(0, min(10, sillage))

            recipe = {
                "id": f"synth_{i:04d}",
                "ingredients": ingredients,
                "scores": {
                    "harmony": harmony,
                    "stability": stability,
                    "longevity": longevity,
                    "sillage": sillage
                }
            }

            data["recipes"].append(recipe)

        return data

    def _prepare_training_data(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 준비"""
        logger.info("Preparing training data...")

        # AdvancedBlendingAI 인스턴스 사용
        blending_ai = AdvancedBlendingAI()

        X_list = []
        y_list = []

        for recipe in data["recipes"]:
            # 특징 추출
            features = blending_ai._encode_ingredient_combination(recipe["ingredients"])
            X_list.append(features)

            # 레이블
            scores = recipe["scores"]
            y = np.array([
                scores["harmony"] / 10.0,
                scores["stability"] / 10.0,
                scores["longevity"] / 10.0,
                scores["sillage"] / 10.0
            ])
            y_list.append(y)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        return X, y

    def train(self, epochs: int = 1000, batch_size: int = 32, learning_rate: float = 0.001):
        """모델 훈련"""
        logger.info("Starting model training...")

        # 데이터 로드 또는 생성
        if self.data_path.exists():
            logger.info(f"Loading training data from {self.data_path}")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        else:
            logger.info("Training data not found. Generating synthetic data...")
            training_data = self._generate_synthetic_data(n_samples=2000)

            # 생성된 데이터 저장
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.data_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Synthetic data saved to {self.data_path}")

        # 데이터 준비
        X, y = self._prepare_training_data(training_data)

        # 스케일링
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 스케일러 저장
        joblib.dump(scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")

        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # 데이터셋과 데이터로더
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 모델 초기화
        input_dim = X_scaled.shape[1]
        model = NeuralBlendingPredictor(input_dim).to(self.device)
        logger.info(f"Model initialized with input dimension: {input_dim}")

        # 옵티마이저와 손실 함수
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # 훈련 루프
        model.train()
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0

        logger.info(f"Training for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_X)

                # 각 헤드별 손실 계산
                loss_harmony = criterion(outputs['harmony'], batch_y[:, 0:1])
                loss_stability = criterion(outputs['stability'], batch_y[:, 1:2])
                loss_longevity = criterion(outputs['longevity'], batch_y[:, 2:3])
                loss_sillage = criterion(outputs['sillage'], batch_y[:, 3:4])

                # 전체 손실
                total_loss = loss_harmony + loss_stability + loss_longevity + loss_sillage

                # Backward pass
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            # 검증
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss_harmony = criterion(val_outputs['harmony'], y_val_tensor[:, 0:1])
                val_loss_stability = criterion(val_outputs['stability'], y_val_tensor[:, 1:2])
                val_loss_longevity = criterion(val_outputs['longevity'], y_val_tensor[:, 2:3])
                val_loss_sillage = criterion(val_outputs['sillage'], y_val_tensor[:, 3:4])
                val_loss = val_loss_harmony + val_loss_stability + val_loss_longevity + val_loss_sillage

            model.train()

            # 진행 상황 출력
            if (epoch + 1) % 100 == 0:
                avg_train_loss = epoch_loss / len(train_loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 조기 종료
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # 최고 모델 저장
                torch.save(model.state_dict(), self.model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to {self.model_path}")

        # 최종 평가
        self._evaluate_model(model, X_val_tensor, y_val_tensor)

    def _evaluate_model(self, model: nn.Module, X_val: torch.Tensor, y_val: torch.Tensor):
        """모델 평가"""
        model.eval()
        with torch.no_grad():
            predictions = model(X_val)

            # 각 점수별 MAE 계산
            harmony_mae = torch.mean(torch.abs(predictions['harmony'] - y_val[:, 0:1])).item() * 10
            stability_mae = torch.mean(torch.abs(predictions['stability'] - y_val[:, 1:2])).item() * 10
            longevity_mae = torch.mean(torch.abs(predictions['longevity'] - y_val[:, 2:3])).item() * 10
            sillage_mae = torch.mean(torch.abs(predictions['sillage'] - y_val[:, 3:4])).item() * 10

        logger.info("\n=== Model Evaluation ===")
        logger.info(f"Harmony MAE: {harmony_mae:.2f}")
        logger.info(f"Stability MAE: {stability_mae:.2f}")
        logger.info(f"Longevity MAE: {longevity_mae:.2f}")
        logger.info(f"Sillage MAE: {sillage_mae:.2f}")
        logger.info(f"Average MAE: {(harmony_mae + stability_mae + longevity_mae + sillage_mae) / 4:.2f}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Train the perfume validation model")
    parser.add_argument("--config", type=str, default="configs/local.json", help="Config file path")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic samples")

    args = parser.parse_args()

    trainer = ValidatorTrainer(config_path=args.config)
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()