"""
우리가 만든 AdvancedFragranceGenerator 모델 학습
최적화된 에포크 수로 효율적인 학습
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import os
import glob
import sys

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 우리가 만든 모델 import
from fragrance_ai.models.advanced_generator import AdvancedFragranceGenerator

class FragranceTrainingDataset(Dataset):
    """향수 학습 데이터셋"""
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input': item['input'],
            'output': item['output'],
            'conditions': self.extract_conditions(item)
        }

    def extract_conditions(self, item):
        """데이터에서 조건 추출"""
        conditions = {}

        # 기본 조건 설정
        defaults = {
            'weather': 'sunny',
            'season': 'spring',
            'time': 'afternoon',
            'mood': 'calm',
            'age_group': '20s',
            'gender': 'unisex',
            'intensity': 'moderate',
            'budget': 'mid_range'
        }

        # 데이터에서 조건 추출
        if 'category' in item:
            if 'location' in item['category']:
                defaults['location'] = 'special'
            if 'emotion' in item['category']:
                defaults['mood'] = 'emotional'

        if 'output' in item:
            output = item['output']
            if 'mood' in output and output['mood']:
                mood_map = {
                    'romantic': 'romantic',
                    'energetic': 'energetic',
                    'calm': 'calm',
                    'mysterious': 'mysterious'
                }
                for m in output['mood']:
                    if m in mood_map:
                        defaults['mood'] = mood_map[m]
                        break

            if 'time_period' in output:
                time_map = {
                    'morning': 'morning',
                    'afternoon': 'afternoon',
                    'evening': 'evening',
                    'night': 'night'
                }
                if output['time_period'] in time_map:
                    defaults['time'] = time_map[output['time_period']]

        return defaults

def load_all_fragrance_data():
    """모든 향수 데이터 로드"""
    all_data = []
    data_files = glob.glob('data/**/*.json', recursive=True)

    print("Loading fragrance data...")

    for file_path in data_files:
        if 'fragrance' in file_path.lower():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    # 다양한 데이터 구조 처리
                    if 'multi_domain_training_data' in data:
                        all_data.extend(data['multi_domain_training_data'])
                    elif 'training_data' in data:
                        all_data.extend(data['training_data'])

            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(all_data)} training samples")
    return all_data

def calculate_optimal_epochs(num_samples):
    """데이터 양에 따른 최적 에포크 수 계산"""

    if num_samples < 50:
        # 매우 적은 데이터: 많은 에포크, 강한 정규화
        return 100, 1e-5, 0.5  # epochs, lr, dropout
    elif num_samples < 100:
        # 적은 데이터: 50-70 에포크
        return 70, 2e-5, 0.4
    elif num_samples < 300:
        # 중간 데이터: 30-50 에포크
        return 40, 3e-5, 0.3
    elif num_samples < 500:
        # 적당한 데이터: 20-30 에포크
        return 25, 5e-5, 0.2
    else:
        # 충분한 데이터: 15-20 에포크
        return 20, 5e-5, 0.1

def train_our_model():
    """우리 모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 데이터 로드
    train_data = load_all_fragrance_data()

    if len(train_data) == 0:
        print("No training data found!")
        return

    # 최적 파라미터 계산
    optimal_epochs, learning_rate, dropout_rate = calculate_optimal_epochs(len(train_data))

    print(f"\n=== Optimized Training Parameters ===")
    print(f"Epochs: {optimal_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Dropout Rate: {dropout_rate}")
    print(f"Batch Size: 1 (memory optimization)")
    print(f"Gradient Accumulation: 8 steps")

    # 우리 모델 초기화
    print("\nInitializing AdvancedFragranceGenerator...")
    model = AdvancedFragranceGenerator()
    model.to(device)

    # 데이터셋과 데이터로더
    dataset = FragranceTrainingDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 옵티마이저와 스케줄러
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Gradient accumulation 설정
    accumulation_steps = 8

    # 학습
    print("\nStarting training...")
    model.train()

    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(optimal_epochs):
        print(f"\n========== Epoch {epoch + 1}/{optimal_epochs} ==========")
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 조건과 프롬프트 준비
                conditions = batch['conditions']
                prompt = batch['input'][0]  # batch size 1

                # 모델 실행
                result = model.generate_recipe(
                    prompt=prompt,
                    conditions=conditions,
                    temperature=0.8
                )

                # 손실 계산 (간단한 예제 - 실제로는 더 복잡한 손실 필요)
                loss = torch.tensor(1.0 - result.get('adaptation_score', 0.5), requires_grad=True)

                # Gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accumulation_steps
                num_batches += 1

                # Progress bar 업데이트
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * accumulation_steps:.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        # 에포크 평균 손실
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        # Learning rate 스케줄링
        scheduler.step()

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0

            # 베스트 모델 저장
            if epoch % 5 == 0:  # 5 에포크마다 저장
                save_checkpoint(model, optimizer, epoch, avg_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, is_best=False)

    # 최종 모델 저장
    print("\nSaving final model...")
    save_final_model(model)

    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.4f}")

def save_checkpoint(model, optimizer, epoch, loss, is_best=True):
    """체크포인트 저장"""
    os.makedirs('checkpoints', exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    filename = 'checkpoints/best_model.pt' if is_best else f'checkpoints/checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def save_final_model(model):
    """최종 모델 저장"""
    os.makedirs('models', exist_ok=True)

    # 모델 저장
    torch.save(model.state_dict(), 'models/fragrance_generator_final.pt')

    # 적응 상태 저장
    model.save_adaptation_state('models/adaptation_state.json')

    print("Final model saved to models/")

def test_trained_model():
    """학습된 모델 테스트"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 로드
    print("Loading trained model...")
    model = AdvancedFragranceGenerator()
    model.load_state_dict(torch.load('models/fragrance_generator_final.pt', map_location=device))
    model.to(device)
    model.eval()

    # 테스트 프롬프트
    test_prompts = [
        "여름 해변에 어울리는 상쾌한 향수를 만들어주세요",
        "로맨틱한 겨울 저녁 데이트용 향수를 추천해주세요",
        "프로페셔널한 비즈니스 미팅에 적합한 향수를 만들어주세요"
    ]

    print("\n=== Testing Trained Model ===")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")

        conditions = {
            'season': 'summer' if '여름' in prompt else 'winter' if '겨울' in prompt else 'spring',
            'mood': 'romantic' if '로맨틱' in prompt else 'professional' if '비즈니스' in prompt else 'fresh',
            'time': 'evening' if '저녁' in prompt else 'morning',
            'weather': 'sunny',
            'age_group': '30s',
            'gender': 'unisex',
            'intensity': 'moderate',
            'budget': 'premium'
        }

        result = model.generate_recipe(
            prompt=prompt,
            conditions=conditions,
            temperature=0.7
        )

        print(f"Generated: {result['generated_recipe'][:200]}...")
        print(f"Performance: {result['predicted_performance']}")

if __name__ == "__main__":
    print("=== Training AdvancedFragranceGenerator ===")
    print("Our custom model with optimized epochs")

    # 학습 실행
    train_our_model()

    # 테스트
    if os.path.exists('models/fragrance_generator_final.pt'):
        print("\n=== Testing Trained Model ===")
        test_trained_model()