"""
데이터 분석 및 최적 에포크 수 계산
"""

import json
import os
import glob

def count_all_training_data():
    """모든 학습 데이터 카운트"""
    total_samples = 0
    data_files = glob.glob('data/**/*.json', recursive=True)

    print("=== 향수 데이터 분석 ===\n")

    for file_path in data_files:
        if 'fragrance' in file_path.lower():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    file_samples = 0

                    # 다양한 데이터 구조 처리
                    if 'multi_domain_training_data' in data:
                        file_samples = len(data['multi_domain_training_data'])
                    elif 'training_data' in data:
                        file_samples = len(data['training_data'])
                    elif 'recipes' in data:
                        file_samples = len(data['recipes'])
                    elif 'fragrances' in data:
                        file_samples = len(data['fragrances'])
                    elif 'fragrance_notes' in data:
                        # 각 카테고리의 노트 수 계산
                        for category in data['fragrance_notes'].values():
                            if 'notes' in category:
                                file_samples += len(category['notes'])

                    if file_samples > 0:
                        print(f"[FILE] {os.path.basename(file_path)}: {file_samples} samples")
                        total_samples += file_samples

            except Exception as e:
                print(f"[ERROR] Error reading {file_path}: {e}")

    return total_samples

def calculate_optimal_epochs(total_samples, batch_size=2):
    """최적 에포크 수 계산"""

    print(f"\n=== 학습 파라미터 계산 ===")
    print(f"총 샘플 수: {total_samples}")
    print(f"배치 크기: {batch_size}")

    # 배치당 스텝 수
    steps_per_epoch = total_samples // batch_size
    print(f"에포크당 스텝: {steps_per_epoch}")

    # 최적 총 스텝 수 (경험적 공식)
    # 작은 데이터셋: 샘플당 10-20회 학습
    # 중간 데이터셋: 샘플당 5-10회 학습
    # 큰 데이터셋: 샘플당 3-5회 학습

    if total_samples < 100:
        iterations_per_sample = 20
        min_epochs = 50
        max_epochs = 100
    elif total_samples < 500:
        iterations_per_sample = 10
        min_epochs = 20
        max_epochs = 50
    elif total_samples < 1000:
        iterations_per_sample = 7
        min_epochs = 10
        max_epochs = 30
    else:
        iterations_per_sample = 5
        min_epochs = 5
        max_epochs = 20

    # 최적 에포크 수 계산
    optimal_epochs = (total_samples * iterations_per_sample) // total_samples
    optimal_epochs = max(min_epochs, min(optimal_epochs, max_epochs))

    # 총 학습 스텝
    total_steps = optimal_epochs * steps_per_epoch

    print(f"\n=== 권장 설정 ===")
    print(f"최소 에포크: {min_epochs}")
    print(f"최적 에포크: {optimal_epochs}")
    print(f"최대 에포크: {max_epochs}")
    print(f"총 학습 스텝: {total_steps:,}")

    # 학습 시간 추정 (GPU 없이)
    # CPU에서 배치당 약 2-3초 예상
    estimated_time_per_batch = 2.5  # 초
    total_time_seconds = total_steps * estimated_time_per_batch
    hours = total_time_seconds // 3600
    minutes = (total_time_seconds % 3600) // 60

    print(f"\n=== 예상 학습 시간 (CPU) ===")
    print(f"배치당 시간: ~{estimated_time_per_batch}초")
    print(f"총 예상 시간: {int(hours)}시간 {int(minutes)}분")

    # 효율적인 설정 제안
    print(f"\n=== 효율적인 학습 전략 ===")

    if total_samples < 100:
        print("[STRATEGY] 소규모 데이터셋 전략:")
        print("  1. 데이터 증강 사용 (paraphrasing)")
        print("  2. Dropout 비율 높이기 (0.3-0.5)")
        print("  3. 학습률 낮추기 (1e-5)")
        print("  4. Early stopping 사용")
    elif total_samples < 500:
        print("[STRATEGY] 중규모 데이터셋 전략:")
        print("  1. 적절한 정규화 사용")
        print("  2. Learning rate scheduling")
        print("  3. Validation set 20% 사용")
        print("  4. Gradient accumulation (steps=4)")
    else:
        print("[STRATEGY] 대규모 데이터셋 전략:")
        print("  1. 배치 크기 늘리기 (4-8)")
        print("  2. Mixed precision training")
        print("  3. Gradient checkpointing")
        print("  4. 체크포인트 저장 (5 에포크마다)")

    return optimal_epochs

if __name__ == "__main__":
    # 데이터 분석
    total = count_all_training_data()

    # 최적 에포크 계산
    optimal = calculate_optimal_epochs(total)

    print(f"\n{'='*50}")
    print(f"[RESULT] 최종 권장 사항:")
    print(f"   - 에포크 수: {optimal}")
    print(f"   - 배치 크기: 2 (메모리 제약)")
    print(f"   - 학습률: 5e-5")
    print(f"   - Gradient Accumulation: 4 steps")
    print(f"{'='*50}")