# Advanced RLHF Features

고급 강화학습 기능: Entropy Annealing, Reward Normalization, Checkpoint & Rollback

---

## Overview

이 문서는 PPO (Proximal Policy Optimization) RLHF 학습을 위한 세 가지 고급 기능의 구현을 설명합니다:

1. **Entropy Annealing** - 탐색에서 수렴으로의 점진적 전환
2. **Reward Normalization** - 보상 스케일 안정화
3. **Checkpoint & Rollback** - 자동 롤백으로 학습 안정성 확보

---

## 1. Entropy Annealing

### 개요

Entropy coefficient를 선형적으로 감소시켜 초기 탐색(exploration)에서 후기 수렴(convergence)으로 점진적으로 전환합니다.

### 동기

- **초기 학습**: 높은 entropy로 다양한 행동 탐색
- **후기 학습**: 낮은 entropy로 최적 행동에 수렴
- **문제**: 고정된 entropy는 탐색과 수렴 사이의 균형 최적화 어려움

### 구현

**`fragrance_ai/training/rl_advanced.py`**

```python
from fragrance_ai.training.rl_advanced import (
    EntropyScheduler,
    EntropyScheduleConfig
)

# 설정
config = EntropyScheduleConfig(
    initial_entropy=0.01,    # 초기 값 (탐색)
    final_entropy=0.001,     # 최종 값 (수렴)
    decay_steps=100000,      # 감소 스텝 수
    schedule_type="linear"   # linear, cosine, exponential
)

# 스케줄러 생성
scheduler = EntropyScheduler(config)

# 학습 루프에서 사용
for step in range(n_steps):
    current_entropy = scheduler.step()

    # PPO 업데이트 시 사용
    loss = policy_loss + value_loss - current_entropy * entropy
```

### 지원되는 스케줄 타입

#### 1. Linear Decay (기본)

```
entropy(t) = initial - (initial - final) * (t / total_steps)
```

- 선형적으로 감소
- 예측 가능한 스케줄
- 대부분의 경우에 적합

#### 2. Cosine Annealing

```
entropy(t) = final + (initial - final) * 0.5 * (1 + cos(π * t / total_steps))
```

- 부드러운 감소
- 초기/후기에 느리고 중간에 빠름
- 미세 조정이 필요한 경우 유용

#### 3. Exponential Decay

```
entropy(t) = initial * exp(log(final / initial) * (t / total_steps))
```

- 초기에 빠르게 감소
- 후기에 느리게 감소
- 빠른 수렴이 필요한 경우 유용

### 사용 예시

```python
# Linear decay (추천)
scheduler = EntropyScheduler(EntropyScheduleConfig(
    initial_entropy=0.01,
    final_entropy=0.001,
    decay_steps=100000,
    schedule_type="linear"
))

# 진행 상황 모니터링
info = scheduler.get_info()
print(f"Progress: {info['progress']:.1%}")
print(f"Current entropy: {info['current_entropy']:.5f}")
print(f"Remaining steps: {info['remaining_steps']}")
```

### 모범 사례

1. **초기 entropy**: 0.01 ~ 0.05 (높을수록 더 많은 탐색)
2. **최종 entropy**: 0.001 ~ 0.005 (낮을수록 더 많은 수렴)
3. **Decay steps**: 전체 학습 스텝의 50-80%
4. **Schedule type**: 대부분의 경우 linear가 적합

---

## 2. Reward Normalization

### 개요

최근 N개 스텝의 이동 평균과 표준편차를 사용하여 보상을 정규화하고 스케일을 안정화합니다.

### 동기

- **문제**: 환경마다 보상 스케일이 다름 (0-1, -100-100, etc.)
- **영향**: 보상 스케일이 크면 학습 불안정, 작으면 학습 느림
- **해결**: 정규화로 스케일 표준화 → 안정적 학습

### 구현

**Welford's Online Algorithm 사용**

- O(1) per update (효율적)
- 수치적으로 안정 (큰 값에서도 정확)

```python
from fragrance_ai.training.rl_advanced import (
    RewardNormalizer,
    RewardNormalizerConfig
)

# 설정
config = RewardNormalizerConfig(
    window_size=1000,              # 이동 평균 윈도우 크기
    epsilon=1e-8,                  # 분산 0 방지
    clip_range=(-10.0, 10.0)       # 정규화 후 클리핑
)

# 정규화기 생성
normalizer = RewardNormalizer(config)

# 보상 정규화
raw_reward = env.step(action)
normalized_reward = normalizer.normalize(raw_reward)

# 배치 정규화
rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
normalized = normalizer.normalize_batch(rewards)
```

### 정규화 공식

```
normalized_reward = (reward - mean) / (std + epsilon)
```

선택적 클리핑:
```
clipped = clip(normalized_reward, clip_min, clip_max)
```

### 통계 추적

```python
stats = normalizer.get_statistics()

print(f"Mean: {stats['mean']:.3f}")
print(f"Std: {stats['std']:.3f}")
print(f"Count: {stats['count']}")
print(f"Buffer size: {stats['buffer_size']}")
```

### 모범 사례

1. **Window size**: 1000-5000 (환경에 따라 조정)
   - 작을수록: 빠른 적응, 노이즈에 민감
   - 클수록: 안정적, 느린 적응

2. **Clip range**: (-10, 10) ~ (-5, 5)
   - 극단적인 아웃라이어 제거
   - 학습 안정성 향상

3. **Warm-up**: 초기 N개 샘플은 정규화 안함
   - 통계가 안정될 때까지 대기

---

## 3. Checkpoint & Rollback

### 개요

주기적인 체크포인트 저장과 나쁜 학습 신호 감지 시 자동 롤백으로 학습 안정성을 확보합니다.

### 동기

- **문제**: 학습 중 정책이 갑자기 나빠질 수 있음
  - KL divergence 급증 (정책 변화가 너무 큼)
  - Loss spike (갑작스런 손실 증가)
  - Reward collapse (성능 급격히 하락)

- **해결**: 자동으로 이전 좋은 체크포인트로 롤백

### 구현

```python
from fragrance_ai.training.rl_advanced import (
    CheckpointManager,
    CheckpointConfig,
    CheckpointMetrics
)

# 설정
config = CheckpointConfig(
    checkpoint_dir="checkpoints",
    save_interval=100,             # 100 스텝마다 저장
    max_checkpoints=5,             # 최대 5개 유지

    # 롤백 조건
    rollback_on_kl_threshold=0.1,         # KL > 0.1
    rollback_on_loss_spike_factor=3.0,    # loss > mean * 3
    rollback_on_reward_drop_factor=0.5,   # reward < mean * 0.5

    metric_window_size=10          # 최근 10개 메트릭 추적
)

# 매니저 생성
manager = CheckpointManager(config)

# 학습 루프
for iteration in range(n_iterations):
    # ... PPO 업데이트 ...

    # 현재 메트릭
    metrics = CheckpointMetrics(
        step=iteration,
        timestamp=datetime.now().isoformat(),
        policy_loss=policy_loss,
        value_loss=value_loss,
        kl_divergence=kl_div,
        mean_reward=mean_reward,
        explained_variance=ev
    )

    # 체크포인트 저장
    if manager.should_save_checkpoint(iteration):
        manager.save_checkpoint(
            step=iteration,
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics
        )

    # 롤백 체크
    should_rollback, reason = manager.should_rollback(metrics)

    if should_rollback:
        print(f"ROLLBACK: {reason}")

        rolled_back_step = manager.rollback(
            network=network,
            optimizer=optimizer,
            scheduler=scheduler,
            reason=reason
        )

        print(f"Restored to step {rolled_back_step}")
```

### 롤백 조건

#### 1. KL Divergence Threshold

```python
if current_kl > threshold:
    rollback()
```

정책이 너무 크게 변경되면 롤백합니다.

**기본값**: 0.1 (PPO 논문 권장값: 0.01-0.03)

#### 2. Loss Spike

```python
current_loss = policy_loss + value_loss
mean_loss = mean(recent_losses)

if current_loss > mean_loss * spike_factor:
    rollback()
```

손실이 갑작스럽게 증가하면 롤백합니다.

**기본값**: 3.0 (평균의 3배)

#### 3. Reward Drop

```python
mean_reward_recent = mean(recent_rewards)

if current_reward < mean_reward_recent * drop_factor:
    rollback()
```

성능이 급격히 하락하면 롤백합니다.

**기본값**: 0.5 (평균의 50% 이하)

### Best Checkpoint Tracking

```python
# 최고 성능 체크포인트 자동 추적
# checkpoint_best.pth에 저장

# 최고 체크포인트 로드
manager.load_best_checkpoint(
    network=network,
    optimizer=optimizer,
    scheduler=scheduler
)
```

### 체크포인트 정리

오래된 체크포인트는 자동으로 삭제됩니다:
- `max_checkpoints` 개수만 유지
- 가장 오래된 것부터 삭제
- `checkpoint_best.pth`는 항상 유지

### 통계 조회

```python
stats = manager.get_statistics()

print(f"Checkpoints: {stats['num_checkpoints']}")
print(f"Best step: {stats['best_checkpoint_step']}")
print(f"Best reward: {stats['best_reward']:.3f}")
print(f"Total rollbacks: {stats['rollback_count']}")
print(f"Last rollback: {stats['last_rollback_step']}")
```

---

## 통합 사용법

### Advanced PPO Trainer

세 가지 기능을 모두 통합한 트레이너:

```python
from fragrance_ai.training.ppo_trainer_advanced import (
    AdvancedPPOTrainer,
    train_advanced_ppo
)
from fragrance_ai.training.ppo_engine import FragranceEnvironment

# 환경 생성
env = FragranceEnvironment(n_ingredients=20)

# 트레이너 생성
trainer = AdvancedPPOTrainer(
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    lr=3e-4,

    # Entropy annealing
    entropy_config=EntropyScheduleConfig(
        initial_entropy=0.01,
        final_entropy=0.001,
        decay_steps=100000,
        schedule_type="linear"
    ),

    # Reward normalization
    reward_config=RewardNormalizerConfig(
        window_size=1000,
        clip_range=(-10.0, 10.0)
    ),

    # Checkpoint & rollback
    checkpoint_config=CheckpointConfig(
        checkpoint_dir="checkpoints/ppo_advanced",
        save_interval=100,
        max_checkpoints=5,
        rollback_on_kl_threshold=0.1
    )
)

# 학습 실행
for iteration in range(1000):
    # 경험 수집 (보상 자동 정규화)
    avg_reward = trainer.collect_rollout(env, n_steps=2048)

    # PPO 업데이트 (entropy annealing, checkpoint & rollback 자동)
    train_stats = trainer.train_step(n_epochs=10, batch_size=64)

    # 로깅
    if iteration % 10 == 0:
        print(f"Iteration {iteration}:")
        print(f"  Reward: {avg_reward:.3f}")
        print(f"  Entropy coef: {train_stats['current_entropy_coef']:.5f}")
        print(f"  Reward norm mean: {train_stats['reward_normalizer_mean']:.3f}")
        print(f"  Rollback count: {train_stats['rollback_count']}")
```

### 편의 함수

```python
# 전체 학습 루프 자동 실행
trainer = train_advanced_ppo(
    env=env,
    n_iterations=1000,
    n_steps_per_iteration=2048,
    n_ppo_epochs=10,
    batch_size=64,
    entropy_config=EntropyScheduleConfig(...),
    reward_config=RewardNormalizerConfig(...),
    checkpoint_config=CheckpointConfig(...)
)

# 전체 통계 조회
full_stats = trainer.get_full_statistics()

print(f"Training step: {full_stats['training_step']}")
print(f"Episodes: {full_stats['episode_count']}")
print(f"Entropy progress: {full_stats['entropy']['progress']:.1%}")
print(f"Best reward: {full_stats['checkpoint']['best_reward']:.3f}")
```

---

## 테스트

### 테스트 실행

```bash
# 전체 테스트 (21개)
python -m pytest tests/test_rl_advanced.py -v

# 특정 테스트 클래스
python -m pytest tests/test_rl_advanced.py::TestEntropyScheduler -v
python -m pytest tests/test_rl_advanced.py::TestRewardNormalizer -v
python -m pytest tests/test_rl_advanced.py::TestCheckpointManager -v

# 통합 테스트
python -m pytest tests/test_rl_advanced.py::TestAdvancedPPOIntegration -v
```

### 테스트 커버리지

**Entropy Scheduler (5 tests)**
- Linear/Cosine/Exponential decay
- Info tracking
- Reset

**Reward Normalizer (6 tests)**
- Basic normalization
- Output values
- Clipping
- Welford algorithm stability
- Batch normalization
- Reset

**Checkpoint Manager (7 tests)**
- Checkpoint saving
- Max checkpoints cleanup
- Rollback on KL/loss/reward
- Best checkpoint tracking
- Rollback execution

**Integration (3 tests)**
- Entropy annealing integration
- Reward normalization integration
- Full training loop

**결과**: **21/21 tests passed** ✅

---

## 성능 영향

### Entropy Annealing

- **오버헤드**: 거의 없음 (단순 계산)
- **메모리**: 무시 가능 (~100 bytes)
- **학습 속도**: 약간 향상 (더 효율적인 탐색)

### Reward Normalization

- **오버헤드**: 매우 낮음 (O(1) per reward)
- **메모리**: window_size * 8 bytes (1k window = 8KB)
- **학습 안정성**: 크게 향상

### Checkpoint & Rollback

- **오버헤드**: 저장 시에만 (100 스텝마다)
- **디스크**: checkpoint당 ~50-200MB (네트워크 크기에 따라)
- **학습 안정성**: 크게 향상 (나쁜 학습 방지)

---

## 하이퍼파라미터 튜닝 가이드

### Entropy Annealing

```python
# 빠른 수렴 원하는 경우
EntropyScheduleConfig(
    initial_entropy=0.02,  # 높게 시작
    final_entropy=0.0001,  # 매우 낮게 끝
    decay_steps=50000,     # 빠른 감소
    schedule_type="exponential"
)

# 안정적 학습 원하는 경우
EntropyScheduleConfig(
    initial_entropy=0.01,
    final_entropy=0.005,   # 적당히 낮게
    decay_steps=150000,    # 천천히 감소
    schedule_type="cosine"
)
```

### Reward Normalization

```python
# 노이즈 많은 환경
RewardNormalizerConfig(
    window_size=5000,      # 큰 윈도우
    clip_range=(-5, 5)     # 강한 클리핑
)

# 안정적인 환경
RewardNormalizerConfig(
    window_size=500,       # 작은 윈도우
    clip_range=(-10, 10)   # 약한 클리핑
)
```

### Checkpoint & Rollback

```python
# 보수적 (안정성 우선)
CheckpointConfig(
    save_interval=50,                    # 자주 저장
    max_checkpoints=10,                  # 많이 유지
    rollback_on_kl_threshold=0.05,       # 낮은 임계값
    rollback_on_loss_spike_factor=2.0,   # 민감한 감지
    rollback_on_reward_drop_factor=0.7   # 민감한 감지
)

# 공격적 (학습 속도 우선)
CheckpointConfig(
    save_interval=200,                   # 덜 자주 저장
    max_checkpoints=3,                   # 적게 유지
    rollback_on_kl_threshold=0.15,       # 높은 임계값
    rollback_on_loss_spike_factor=5.0,   # 덜 민감
    rollback_on_reward_drop_factor=0.3   # 덜 민감
)
```

---

## 문제 해결

### Entropy가 너무 빨리 감소

**증상**: 초기부터 탐색 부족

**해결**:
```python
# decay_steps 증가
decay_steps=200000  # 원래 100000에서 2배

# 또는 cosine annealing 사용
schedule_type="cosine"
```

### Reward 정규화 후에도 불안정

**증상**: 학습이 여전히 불안정

**해결**:
```python
# 윈도우 크기 증가
window_size=5000

# 클리핑 범위 축소
clip_range=(-5, 5)
```

### 롤백이 너무 자주 발생

**증상**: 학습이 진행되지 않음

**해결**:
```python
# 임계값 완화
rollback_on_kl_threshold=0.2
rollback_on_loss_spike_factor=5.0
rollback_on_reward_drop_factor=0.3

# 메트릭 윈도우 증가
metric_window_size=20
```

### 롤백이 전혀 발생하지 않음

**증상**: 나쁜 학습 신호를 놓침

**해결**:
```python
# 임계값 강화
rollback_on_kl_threshold=0.05
rollback_on_loss_spike_factor=2.0
rollback_on_reward_drop_factor=0.7
```

---

## 참고 문헌

1. **PPO 논문**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. **Entropy Regularization**: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290)
3. **Welford Algorithm**: [Donald Knuth, The Art of Computer Programming](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
4. **Checkpoint Strategies**: [Uber's Evolution Strategies](https://arxiv.org/abs/1703.03864)

---

## 파일 목록

### 구현
- `fragrance_ai/training/rl_advanced.py` (640 lines) - 3가지 기능 구현
- `fragrance_ai/training/ppo_trainer_advanced.py` (550 lines) - 통합 트레이너

### 테스트
- `tests/test_rl_advanced.py` (680 lines) - 21개 테스트

### 문서
- `docs/RLHF_ADVANCED.md` (이 파일) - 종합 문서

---

## Summary

✅ **Entropy Annealing** - 탐색→수렴 전환으로 학습 효율 향상
✅ **Reward Normalization** - 보상 스케일 안정화로 학습 안정성 향상
✅ **Checkpoint & Rollback** - 자동 롤백으로 나쁜 학습 방지

**결과**: 더 안정적이고 효율적인 RLHF 학습

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
