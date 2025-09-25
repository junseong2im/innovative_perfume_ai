"""
실제 딥러닝 학습 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fragrance_ai.models.simple_fragrance_model import (
    FragranceDataset,
    SimpleFragranceNN,
    FragranceModelTrainer
)
import torch
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_training_history(train_losses, val_losses, save_path='training_history.png'):
    """학습 히스토리 시각화"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.show()
    logger.info(f"Training history saved to {save_path}")


def test_real_training():
    """실제 학습 테스트"""
    print("\n" + "="*60)
    print("[START] 실제 딥러닝 모델 학습 테스트 시작")
    print("="*60 + "\n")

    # 1. 데이터셋 생성
    print("Step 1: 데이터셋 생성 중...")
    dataset = FragranceDataset()
    print(f"[OK] 데이터셋 크기: {len(dataset)} 샘플")

    # 2. 모델 생성
    print("\nStep 2: 신경망 모델 생성 중...")
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[0]
    model = SimpleFragranceNN(input_dim=input_dim, hidden_dims=[128, 64, 32])

    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] 모델 파라미터: {total_params:,} (학습 가능: {trainable_params:,})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[OK] 사용 디바이스: {device}")

    # 3. 트레이너 생성
    print("\nStep 3: 트레이너 초기화...")
    trainer = FragranceModelTrainer(model, dataset)
    print("[OK] 트레이너 준비 완료")

    # 4. 실제 학습 실행
    print("\nStep 4: 실제 학습 시작...")
    print("-" * 40)

    try:
        train_losses, val_losses = trainer.train(epochs=20, batch_size=8)
        print("\n[DONE] 학습 완료!")

        # 5. 학습 결과 출력
        print("\n학습 결과:")
        print(f"  - 최종 Train Loss: {train_losses[-1]:.4f}")
        print(f"  - 최종 Val Loss: {val_losses[-1]:.4f}")
        print(f"  - Best Val Loss: {min(val_losses):.4f}")

        # 6. 학습 곡선 시각화
        if len(train_losses) > 0:
            plot_training_history(train_losses, val_losses)

        # 7. 모델 저장
        model_path = f"fragrance_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        trainer.save_model(model_path)
        print(f"\n모델 저장: {model_path}")

        # 8. 테스트 예측
        print("\nStep 5: 테스트 예측...")
        test_samples = [
            {
                'family': 'fresh',
                'mood': 'energetic',
                'season': 'summer',
                'intensity': 'light',
                'top_notes': ['bergamot', 'lemon'],
                'heart_notes': ['lavender'],
                'base_notes': ['musk']
            },
            {
                'family': 'oriental',
                'mood': 'mysterious',
                'season': 'winter',
                'intensity': 'strong',
                'top_notes': ['orange'],
                'heart_notes': ['rose', 'jasmine'],
                'base_notes': ['amber', 'vanilla']
            },
            {
                'family': 'floral',
                'mood': 'romantic',
                'season': 'spring',
                'intensity': 'moderate',
                'top_notes': ['grapefruit'],
                'heart_notes': ['rose', 'geranium'],
                'base_notes': ['sandalwood']
            }
        ]

        print("\n예측 결과:")
        print("-" * 40)
        for i, sample in enumerate(test_samples, 1):
            prediction = trainer.predict(sample)
            print(f"\n샘플 {i}: {sample['family']} / {sample['mood']} / {sample['season']}")
            print(f"  예측 평점: {prediction:.2f}/5.0")

        print("\n" + "="*60)
        print("[SUCCESS] 실제 딥러닝 학습 테스트 완료!")
        print("="*60 + "\n")

        return True

    except Exception as e:
        print(f"\n[ERROR] 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_gpu():
    """GPU 사용 가능 여부 확인"""
    print("\nGPU 정보:")
    print("-" * 40)

    if torch.cuda.is_available():
        print("[OK] CUDA 사용 가능")
        print(f"  - GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"  - CUDA 버전: {torch.version.cuda}")
    else:
        print("GPU를 사용할 수 없습니다. CPU로 학습합니다.")
        print("  (GPU를 사용하면 학습 속도가 훨씬 빨라집니다)")


if __name__ == "__main__":
    # GPU 확인
    verify_gpu()

    # 실제 학습 테스트
    success = test_real_training()

    if success:
        print("\n[SUCCESS] 모든 테스트가 성공적으로 완료되었습니다!")
        print("   이제 실제 딥러닝 모델이 학습되고 있습니다.")
    else:
        print("\n[WARNING] 테스트 중 일부 문제가 발생했습니다.")