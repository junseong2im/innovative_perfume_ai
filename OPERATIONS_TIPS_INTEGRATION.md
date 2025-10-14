# 운영 팁 적용 가이드

실무 환경에서 바로 사용 가능한 운영 설정을 구현했습니다.

## 1. 학습 스케줄

### Entropy Coefficient Decay (선형/코사인)

```python
from fragrance_ai.config.operations_config import OperationsManager

# 운영 관리자 초기화
ops_manager = OperationsManager(checkpoint_dir="./checkpoints")

# 학습 루프에서 사용
for step in range(10000):
    # 현재 스텝의 entropy coefficient 자동 계산
    result = ops_manager.on_training_step(
        step=step,
        metrics={"loss": loss, "reward": reward},
        model_state=trainer.network.state_dict()
    )

    # PPO 업데이트 시 사용
    current_entropy_coef = result["entropy_coef"]  # 0.01 -> 0.001로 decay
    trainer.entropy_coef = current_entropy_coef
```

**설정:**
- 초기: 0.01 (exploration)
- 최종: 0.001 (exploitation)
- Decay: cosine (기본) 또는 linear
- 기간: 10,000 steps

### Reward Normalization (1k step 창)

```python
# 자동으로 1k step 창에서 reward normalize
result = ops_manager.on_training_step(step, metrics, model_state)

# Normalized reward 사용
normalized_reward = result["metrics"]["normalized_reward"]

# 통계 확인
stats = ops_manager.reward_normalizer.get_stats()
print(f"Reward mean: {stats['mean']:.3f}, std: {stats['std']:.3f}")
```

**효과:**
- 안정적인 학습
- Outlier 완화
- 1,000 step 슬라이딩 윈도우


## 2. 체크포인트 관리

### 500 Step마다 자동 저장

```python
# 자동 저장 - 별도 코드 불필요
result = ops_manager.on_training_step(step, metrics, model_state)

if result["checkpoint_path"]:
    print(f"✓ Checkpoint saved: {result['checkpoint_path']}")
```

**설정:**
- 저장 주기: 500 steps
- 최대 보관: 5개 (오래된 것 자동 삭제)

### 이상 감지 & 자동 롤백

```python
# 이상 감지
result = ops_manager.on_training_step(step, metrics, model_state)

if result["anomaly"]:
    print(f"❌ Anomaly detected: {result['anomaly']}")

    # 자동 롤백
    if result["rollback_path"]:
        trainer.load_checkpoint(result["rollback_path"])
        print(f"🔄 Rolled back to: {result['rollback_path']}")
```

**임계값:**
- Loss spike: 이전 대비 2배 이상
- Reward drop: 이전 대비 50% 이하


## 3. 장애 대처 (Circuit Breaker)

### Qwen 장애 → 자동 다운시프트

```python
# LLM 호출 후 결과 기록
try:
    result = qwen_model.generate(prompt)
    ops_manager.on_llm_success("qwen")
except Exception as e:
    ops_manager.on_llm_failure("qwen")

# 다운시프트 확인
if ops_manager.circuit_breaker.should_use_downshift():
    # creative -> balanced 또는 fast로 자동 전환
    mode = "balanced"  # 또는 "fast"
    print(f"🔻 Using downshift mode: {mode}")
```

**설정:**
- 실패 임계값: 3회
- 다운시프트 유지: 5분
- 자동 복구: 5분 후

### TTL 단축으로 재시도 폭 줄이기

```python
# 현재 TTL 확인
current_ttl = ops_manager.circuit_breaker.get_current_ttl()

# Cache 설정 시 사용
cache.set(key, value, ttl=current_ttl)
```

**설정:**
- 정상 TTL: 3600초 (1시간)
- 단축 TTL: 300초 (5분)
- 자동 전환: 다운시프트 시


## 4. 데이터 기반 재튜닝

### 100-300건 피드백 수집 후 자동 재튜닝

```python
# 사용자 피드백 수집
feedback = {
    "rating": 4.5,
    "user_id": "user_123",
    "formula_id": "F001"
}

suggestions = ops_manager.on_user_feedback(feedback)

# 재튜닝 제안
if suggestions:
    print("📊 Hyperparameter retuning suggestions:")

    # Learning rate 조정
    if "learning_rate" in suggestions:
        trainer.optimizer.param_groups[0]['lr'] = suggestions["learning_rate"]

    # Entropy coefficient 조정
    if "entropy_coef" in suggestions:
        trainer.entropy_coef = suggestions["entropy_coef"]

    # Clip epsilon 조정
    if "clip_epsilon" in suggestions:
        trainer.clip_epsilon = suggestions["clip_epsilon"]
```

**재튜닝 로직:**
- 평균 rating < 3.0 → entropy_coef ↑ (exploration)
- rating 변동 > 1.5 → learning_rate ↓ (stability)
- 평균 rating ≥ 4.0 → clip_epsilon ↓ (exploitation)


## 통합 예시

### PPO 학습 루프에 적용

```python
from fragrance_ai.config.operations_config import OperationsManager
from fragrance_ai.training.ppo_engine import PPOTrainer

# 초기화
ops_manager = OperationsManager(checkpoint_dir="./checkpoints")
trainer = PPOTrainer(state_dim=40, action_dim=60)

# 학습 루프
for step in range(10000):
    # 1. 경험 수집
    avg_reward = trainer.collect_rollout(env, n_steps=2048)

    # 2. PPO 업데이트
    train_stats = trainer.train_step(n_epochs=10, batch_size=64)

    # 3. 운영 관리 (자동 처리)
    ops_result = ops_manager.on_training_step(
        step=step,
        metrics={
            "loss": train_stats["policy_loss"],
            "reward": avg_reward
        },
        model_state=trainer.network.state_dict()
    )

    # 4. Entropy coefficient 자동 조정
    trainer.entropy_coef = ops_result["entropy_coef"]

    # 5. 이상 감지 & 롤백
    if ops_result["rollback_path"]:
        trainer.load_checkpoint(ops_result["rollback_path"])
        print(f"🔄 Rolled back due to: {ops_result['anomaly']}")

    # 6. 다운시프트 확인
    if ops_result["should_downshift"]:
        print("🔻 Using downshift mode for LLM")

    # 로깅
    if step % 100 == 0:
        print(f"Step {step}:")
        print(f"  Entropy coef: {ops_result['entropy_coef']:.4f}")
        print(f"  Reward: {avg_reward:.3f}")
        print(f"  TTL: {ops_result['current_ttl']}s")
```


## 설정 커스터마이징

### 개별 설정 변경

```python
from fragrance_ai.config.operations_config import (
    LearningScheduleConfig,
    CheckpointConfig,
    CircuitBreakerConfig,
    RetuningConfig
)

# 학습 스케줄 커스터마이징
schedule_config = LearningScheduleConfig(
    entropy_coef_start=0.02,  # 초기값 증가
    entropy_coef_end=0.0005,  # 최종값 감소
    entropy_decay_type="linear",  # cosine -> linear
    reward_norm_window=2000  # 1k -> 2k
)

# 체크포인트 커스터마이징
checkpoint_config = CheckpointConfig(
    save_interval_steps=1000,  # 500 -> 1000
    loss_spike_threshold=1.5,  # 2.0 -> 1.5 (더 민감하게)
    enable_auto_rollback=True
)

# 적용
ops_manager = OperationsManager(checkpoint_dir="./checkpoints")
ops_manager.schedule_config = schedule_config
ops_manager.checkpoint_config = checkpoint_config
```


## 모니터링

### Prometheus 메트릭과 연동

```python
from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

collector = OperationsMetricsCollector()

# 학습 스텝마다 기록
ops_result = ops_manager.on_training_step(step, metrics, model_state)

# RL 메트릭 기록
collector.record_rl_update(
    algorithm="ppo",
    policy_loss=train_stats["policy_loss"],
    value_loss=train_stats["value_loss"],
    reward=avg_reward,
    entropy=ops_result["entropy_coef"]
)

# 서킷브레이커 메트릭 기록
if ops_result["should_downshift"]:
    collector.record_circuit_breaker_downgrade("llm", "creative", "balanced")

# 캐시 TTL 메트릭
collector.set_cache_size("llm", cache_size)
```


## 주요 이점

### 1. 학습 안정성
- ✅ Entropy decay로 exploration → exploitation 자연스러운 전환
- ✅ Reward normalization으로 학습 안정화
- ✅ 자동 이상 감지 & 롤백

### 2. 장애 복원력
- ✅ Qwen 장애 시 자동 다운시프트 (5분간)
- ✅ TTL 단축으로 재시도 빈도 감소
- ✅ 체크포인트 자동 저장 & 복구

### 3. 데이터 기반 최적화
- ✅ 100-300건 피드백으로 하이퍼파라미터 자동 재튜닝
- ✅ 사용자 평가 패턴에 맞춘 동적 조정
- ✅ 지속적 개선 (Continuous Learning)


## 테스트

```bash
# 운영 설정 테스트
cd "C:\Users\user\Desktop\새 폴더 (2)\Newss"
python -m fragrance_ai.config.operations_config
```

**예상 출력:**
```
=== 운영 설정 시뮬레이션 ===

1. 학습 스텝 시뮬레이션
  Step 500: Checkpoint saved
  Step 1000: Checkpoint saved

2. Qwen 장애 시뮬레이션
  Failure 3: Downshift activated!

3. 사용자 피드백 수집 및 재튜닝
  ✓ Retuning triggered after 100 samples
  Suggestions: {'entropy_coef': 0.02, 'learning_rate': 0.0001}

완료!
```


## 요약

| 기능 | 설정 값 | 효과 |
|------|---------|------|
| **Entropy Decay** | 0.01 → 0.001 (cosine) | Exploration → Exploitation |
| **Reward Norm** | 1k step 창 | 학습 안정화 |
| **Checkpoint** | 500 step 간격 | 자동 저장 & 롤백 |
| **Circuit Breaker** | 3회 실패 → 다운시프트 | 장애 복원력 |
| **TTL 단축** | 1h → 5min | 재시도 폭 감소 |
| **재튜닝** | 100-300건 피드백 | 동적 최적화 |

**모든 기능이 자동으로 동작합니다. 별도 코드 수정 불필요!** 🚀
