# Artisan Runbooks

자동화된 장애 대응 절차

---

## 목차

1. [Qwen LLM Failure → Downshift](#1-qwen-llm-failure--downshift)
2. [RL Reward Runaway → Checkpoint Rollback](#2-rl-reward-runaway--checkpoint-rollback)
3. [API High Latency → Scale Up](#3-api-high-latency--scale-up)
4. [Database Connection Failure](#4-database-connection-failure)
5. [Cache Service Failure](#5-cache-service-failure)

---

## 1. Qwen LLM Failure → Downshift

**Runbook ID:** `qwen_failure_downshift`
**Severity:** Sev2
**Automation Level:** 자동 (수동 확인 1단계)
**Expected Recovery Time:** 5분

### 트리거 조건

- Qwen health check 3회 연속 실패
- Qwen API 타임아웃 > 30s
- Qwen 에러율 > 50% (5분 평균)

### 증상

- Creative mode API p95 latency > 10s
- Qwen 모델 응답 없음
- 서킷브레이커 활성화

### 실행 방법

```bash
# Dry run (시뮬레이션)
python -m fragrance_ai.sre.runbooks --execute qwen_failure_downshift --dry-run

# 실제 실행
python -m fragrance_ai.sre.runbooks --execute qwen_failure_downshift
```

### 단계별 절차

#### Step 1: Verify Qwen Failure (자동)
```bash
# Health check
curl http://localhost:8000/health/llm?model=qwen

# Expected: status_code 503 or timeout
```

**확인 사항:**
- [ ] Qwen health check 실패 3회 이상
- [ ] 최근 5분 에러율 > 50%

#### Step 2: Enable Circuit Breaker (자동)
```python
from fragrance_ai.guards.circuit_breaker import get_circuit_breaker

breaker = get_circuit_breaker("qwen")
breaker.force_open()  # 서킷브레이커 강제 활성화
```

**결과:**
- Qwen으로 가는 모든 요청 차단
- 자동으로 대체 경로 활성화

#### Step 3: Downshift Creative → Balanced (자동)
```python
from fragrance_ai.config.feature_flags import get_feature_flag_manager

manager = get_feature_flag_manager()
manager.set_downshift_mode("creative", "balanced")
# Creative 요청을 Balanced 모드로 라우팅
```

**결과:**
- Creative mode 요청이 Balanced mode로 처리됨
- API 지연시간 감소 (15s → 3.2s)

#### Step 4: Check Balanced Health (자동)
```bash
curl http://localhost:8000/health/llm?model=mistral
curl http://localhost:8000/health/llm?model=llama

# Expected: status_code 200
```

**확인 사항:**
- [ ] Mistral 모델 정상
- [ ] Llama 모델 정상
- [ ] Balanced mode p95 < 3.5s

#### Step 5: Notify On-Call (자동)
```bash
# Slack 알림 자동 발송
# PagerDuty alert 자동 발송
```

**알림 내용:**
- Qwen 장애 발생
- 자동으로 Balanced로 다운시프트됨
- 온콜 엔지니어 조치 필요

#### Step 6: Create Incident (자동)
```bash
python -m fragrance_ai.sre.incident_manager --create '{
  "title": "Qwen LLM Failure - Downshifted to Balanced",
  "description": "Qwen model not responding. Auto-downshifted.",
  "severity": "Sev2",
  "components": ["LLM", "Qwen"]
}'
```

**결과:**
- Sev2 사건 생성
- 타임라인 기록 시작

### 수동 복구 절차

#### 온콜 엔지니어 조치:

1. **Qwen 로그 확인**
```bash
kubectl logs -n production deployment/qwen-llm --tail=100
```

2. **근본 원인 파악**
- OOM (Out of Memory)?
- 모델 로딩 실패?
- 네트워크 이슈?

3. **Qwen 재시작**
```bash
kubectl rollout restart deployment/qwen-llm -n production
kubectl rollout status deployment/qwen-llm -n production
```

4. **Health check 확인**
```bash
# 5분 대기 후
curl http://localhost:8000/health/llm?model=qwen
```

5. **Creative mode 복원**
```python
manager.restore_mode("creative")
breaker.close()  # 서킷브레이커 닫기
```

6. **사건 해결**
```bash
python -m fragrance_ai.sre.incident_manager \
  --resolve INC-xxx \
  --resolution "Qwen restarted, Creative mode restored"
```

### 롤백 절차

```python
# 자동 롤백
manager.restore_mode("creative")
breaker.close()
```

### 예방 조치

- [ ] Qwen 메모리 모니터링 강화
- [ ] Qwen 자동 재시작 (liveness probe)
- [ ] Qwen 로드 밸런싱 (여러 인스턴스)

---

## 2. RL Reward Runaway → Checkpoint Rollback

**Runbook ID:** `rl_reward_runaway_rollback`
**Severity:** Sev3
**Automation Level:** 반자동 (수동 확인 2단계)
**Expected Recovery Time:** 15분

### 트리거 조건

- RL 평균 보상 > 100 (정상: 10-20)
- RL 보상 표준편차 > 50
- RL KL divergence > 0.1 (임계값: 0.03)

### 증상

- 학습 불안정
- 보상 값 비정상적으로 높음
- 진화 결과 품질 저하

### 실행 방법

```bash
python -m fragrance_ai.sre.runbooks --execute rl_reward_runaway_rollback
```

### 단계별 절차

#### Step 1: Verify Reward Runaway (자동)
```bash
# 최근 100 에피소드 보상 확인
python -c "
from fragrance_ai.training.rl.ppo import get_trainer
trainer = get_trainer()
stats = trainer.get_statistics()
print(f'Avg Reward: {stats[\"reward_mean\"]}')
print(f'Std Reward: {stats[\"reward_std\"]}')
"
```

**확인 사항:**
- [ ] 평균 보상 > 100
- [ ] 표준편차 > 50

#### Step 2: Stop RL Training (자동)
```bash
# 학습 프로세스 중단
pkill -f "ppo_trainer"
```

**결과:**
- 모든 RL 학습 프로세스 중단
- 추가 보상 폭주 방지

#### Step 3: Find Stable Checkpoint (자동)
```bash
python -c "
from fragrance_ai.training.checkpoint_manager import get_checkpoint_manager

manager = get_checkpoint_manager()
stable_checkpoint = manager.find_stable_checkpoint(
    kl_threshold=0.03,
    reward_min=10.0,
    reward_max=20.0
)
print(f'Found stable checkpoint: {stable_checkpoint}')
"
```

**기준:**
- KL divergence < 0.03
- Reward: 10.0 ~ 20.0
- 최신 안정 체크포인트 선택

#### Step 4: Rollback Checkpoint (수동 확인 필요)
```bash
# 체크포인트 로드
python -c "
from fragrance_ai.training.rl.ppo import get_trainer

trainer = get_trainer()
trainer.load_checkpoint('checkpoint_step_4500.pt')
print('Checkpoint loaded successfully')
"
```

**수동 확인:**
```
⚠️  This will rollback training to checkpoint_step_4500.pt (2 hours ago).
   Current step: 5000
   Rollback to: 4500
   Progress lost: 500 steps (~30 minutes of training)

Proceed with rollback? (y/n):
```

#### Step 5: Verify Rollback (자동)
```bash
# 모델 상태 확인
python -c "
from fragrance_ai.training.rl.ppo import get_trainer

trainer = get_trainer()
print(f'Current step: {trainer.global_step}')
print(f'Expected: 4500')
assert trainer.global_step == 4500
"
```

#### Step 6: Resume Training (자동)
```bash
# 학습률 50% 감소 후 재개
python -c "
from fragrance_ai.training.rl.ppo import get_trainer

trainer = get_trainer()
trainer.set_learning_rate(trainer.learning_rate * 0.5)
print(f'Reduced LR: {trainer.learning_rate}')

trainer.resume_training()
print('Training resumed')
"
```

#### Step 7: Create Incident (자동)
```bash
python -m fragrance_ai.sre.incident_manager --create '{
  "title": "RL Reward Runaway - Checkpoint Rollback",
  "description": "RL training reward runaway. Rolled back to step 4500.",
  "severity": "Sev3",
  "components": ["RL", "PPO"]
}'
```

### 수동 복구 절차

1. **보상 함수 검증**
```python
# 보상 계산 로직 확인
from fragrance_ai.training.rl.reward import calculate_reward

test_state = {...}
reward = calculate_reward(test_state)
print(f'Test reward: {reward}')
# Expected: 10-20 range
```

2. **학습률 조정**
```python
# 더 보수적인 학습률
trainer.set_learning_rate(1e-5)  # Default: 3e-4
```

3. **하이퍼파라미터 재조정**
```python
trainer.set_clip_epsilon(0.1)  # More conservative
trainer.set_entropy_coef(0.01)  # More exploration
```

### 예방 조치

- [ ] 보상 상한선 설정 (max reward = 50)
- [ ] KL divergence 알람 (> 0.05)
- [ ] 자동 체크포인트 검증

---

## 3. API High Latency → Scale Up

**Runbook ID:** `api_high_latency_scaleup`
**Severity:** Sev3
**Automation Level:** 자동
**Expected Recovery Time:** 3분

### 트리거 조건

- API p95 latency > 5s (임계값: 2.5s)
- CPU 사용률 > 85%
- 요청 대기열 > 100

### 실행 방법

```bash
python -m fragrance_ai.sre.runbooks --execute api_high_latency_scaleup
```

### 단계별 절차

#### Step 1: Verify High Latency
```bash
# Prometheus 쿼리
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, rate(api_response_seconds_bucket[5m]))'
```

#### Step 2: Scale Up
```bash
# Kubernetes HPA 수동 스케일
kubectl scale deployment artisan-api --replicas=8 -n production
```

#### Step 3: Verify Improvement
```bash
# 5분 대기 후
curl -G http://prometheus:9090/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, rate(api_response_seconds_bucket[5m]))'
```

---

## 4. Database Connection Failure

**Runbook ID:** `database_connection_failure`
**Severity:** Sev1
**Automation Level:** 반자동
**Expected Recovery Time:** 10분

### 트리거 조건

- Database connection pool exhausted
- Connection timeout > 5s
- 모든 DB 쿼리 실패

### 단계별 절차

1. **Connection Pool 상태 확인**
2. **Database 재시작 (필요 시)**
3. **Connection Pool 재설정**
4. **Read Replica 활성화**

---

## 5. Cache Service Failure

**Runbook ID:** `cache_service_failure`
**Severity:** Sev3
**Automation Level:** 자동
**Expected Recovery Time:** 2분

### 트리거 조건

- Redis connection 실패
- Cache hit rate = 0%

### 단계별 절차

1. **Cache bypass 활성화** (자동)
2. **Redis health check**
3. **Redis 재시작 (필요 시)**
4. **Cache warming**

---

## 런북 테스트

### 월간 런북 드릴

모든 런북은 **월 1회** 테스트 실행:

```bash
# 모든 런북 Dry run
for runbook in qwen_failure_downshift rl_reward_runaway_rollback api_high_latency_scaleup; do
  echo "Testing $runbook..."
  python -m fragrance_ai.sre.runbooks --execute $runbook --dry-run
done
```

### 런북 업데이트 절차

1. 사건 발생 후 런북 효과성 평가
2. 개선 사항 식별
3. 런북 업데이트
4. Dry run 테스트
5. 팀 공유

---

**런북은 살아있는 문서입니다. 지속적으로 업데이트하세요!** 📖
