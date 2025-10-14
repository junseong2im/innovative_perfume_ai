# Fragrance AI - 운영 가이드 (Operations Guide)

**작성일**: 2025-10-13
**버전**: 1.0

---

## 개요

이 문서는 Fragrance AI 시스템의 운영을 위한 완전한 가이드입니다.

### 주요 기능

1. **체크포인트 관리**: 500 step 주기 저장, 손실 급등/보상 급락 시 롤백
2. **모델 핫 리로드**: 새 워커 warm-up → 트래픽 전환 (graceful reload)
3. **운영 대시보드**: LLM Brief, RL Update, API 메트릭 모니터링

---

## 1. 체크포인트 관리 (Checkpoint Manager)

### 기능

정책/가치 네트워크를 주기적으로 저장하고, 이상 징후 발생 시 자동 롤백합니다.

### 설정

```python
from fragrance_ai.training.checkpoint_manager import (
    CheckpointManager,
    RollbackConditions
)

# 체크포인트 매니저 초기화
manager = CheckpointManager(
    checkpoint_dir="checkpoints",      # 저장 디렉토리
    save_interval=500,                  # 500 step마다 저장
    max_checkpoints=5,                  # 최대 5개 유지
    rollback_conditions=RollbackConditions(
        kl_threshold=0.03,              # KL divergence 임계값
        loss_increase_multiplier=2.0,   # 손실 2배 증가 시 롤백
        reward_drop_threshold=0.3       # 보상 30% 하락 시 롤백
    )
)
```

### 사용법

#### 1) 체크포인트 저장

```python
# 학습 루프
for step in range(max_steps):
    # 학습 코드
    loss, reward = train_step(policy_net, value_net, optimizer)

    # 500 step마다 자동 저장
    if manager.should_save(step):
        manager.save_checkpoint(
            step=step,
            policy_net=policy_net,
            value_net=value_net,
            optimizer=optimizer,
            loss=loss,
            reward=reward,
            kl_divergence=compute_kl_divergence(),
            entropy=compute_entropy(),
            clip_fraction=compute_clip_fraction()
        )
```

#### 2) 롤백 체크 및 실행

```python
# 매 step마다 롤백 필요 여부 체크
should_rollback, rollback_ckpt, reason = manager.check_rollback(
    current_step=step,
    current_loss=loss,
    current_reward=reward,
    current_kl_divergence=kl_div
)

if should_rollback:
    logger.warning(f"[롤백 트리거] {reason}")

    # 직전 체크포인트로 롤백
    previous = manager.rollback(
        policy_net=policy_net,
        value_net=value_net,
        optimizer=optimizer
    )

    if previous:
        logger.info(f"롤백 완료: step={previous.step}")
```

### 롤백 트리거 조건

| 조건 | 임계값 | 설명 |
|------|--------|------|
| **KL Divergence** | > 0.03 | 정책이 너무 급격히 변화 |
| **손실 급등** | > 2.0x | 이전 대비 손실이 2배 이상 증가 |
| **보상 급락** | > 30% | 이전 대비 보상이 30% 이상 하락 |

### 파일 구조

```
checkpoints/
├── checkpoint_step_500.pt
├── checkpoint_step_1000.pt
├── checkpoint_step_1500.pt
├── checkpoint_step_2000.pt
├── checkpoint_step_2500.pt
└── checkpoint_history.json   # 메타데이터
```

---

## 2. 모델 핫 리로드 (Model Hot Reload)

### 기능

서비스 중단 없이 모델을 업데이트합니다:
1. 새 워커 생성
2. 새 모델 로드 + warm-up
3. 트래픽 전환 (old → new)
4. Old 워커 graceful shutdown

### 설정

```python
from fragrance_ai.training.model_hot_reload import HotReloadManager

# 모델 로더 함수
def load_model():
    model = load_qwen_model("models/qwen-2.5-7b")
    return model

# Warm-up 함수
def warmup(model):
    # 몇 가지 테스트 추론 실행
    test_input = "Create a floral fragrance"
    _ = model.generate(test_input)

# 핫 리로드 매니저 초기화
reload_manager = HotReloadManager(
    model_loader=load_model,
    warmup_func=warmup,
    warmup_steps=5,        # 5번 warm-up 실행
    drain_timeout=30.0     # 30초 배출 타임아웃
)
```

### 사용법

#### 1) 첫 번째 워커 초기화

```python
# 앱 시작 시 첫 번째 워커 초기화
reload_manager.initialize_first_worker()

# 워커 정보 확인
info = reload_manager.get_current_worker_info()
print(f"Worker ID: {info.worker_id}")
print(f"Status: {info.status.value}")
```

#### 2) 요청 처리

```python
# 요청 처리 함수
def process_request(model, user_text: str):
    return model.generate(user_text)

# API 엔드포인트
@app.post("/dna/create")
async def create_dna(request: DNACreateRequest):
    result = reload_manager.serve(
        process_request,
        user_text=request.user_text
    )
    return result
```

#### 3) 모델 업데이트 (리로드)

```python
# 새 모델로 업데이트
success = reload_manager.reload()

if success:
    logger.info("모델 리로드 완료")
else:
    logger.error("모델 리로드 실패")
```

### 리로드 프로세스

```
[Old Worker (v1)]
       ↓
[New Worker (v2) 생성]
       ↓
[v2 로드 + Warm-up]  ← 5 steps
       ↓
[트래픽 전환]
       ↓
[Old Worker 배출]    ← 진행 중인 요청 완료 대기 (30s)
       ↓
[Old Worker 정지]
       ↓
[완료]
```

### API 사용 예시

```bash
# 모델 리로드 트리거 (FastAPI 엔드포인트)
POST /admin/model/reload
```

```python
@app.post("/admin/model/reload")
async def trigger_reload():
    success = reload_manager.reload()
    return {"success": success}
```

---

## 3. 운영 대시보드 (Operations Dashboard)

### 메트릭 수집

```python
from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

# 컬렉터 초기화
metrics_collector = OperationsMetricsCollector()
```

### 메트릭 종류

#### 1) LLM Brief 메트릭

```python
# LLM Brief 생성 기록
metrics_collector.record_llm_brief(
    mode="creative",              # fast/balanced/creative
    success=True,                 # 성공 여부
    latency_seconds=4.2,          # 지연 시간
    repaired=True,                # 수정 여부
    repair_type="trailing_comma"  # 수정 타입
)

# 또는 context manager 사용
with metrics_collector.track_llm_brief("creative") as tracker:
    brief = generate_brief()
    if repaired:
        tracker.mark_repaired("code_block")
```

**메트릭**:
- `llm_brief_total`: 총 생성 건수 (mode, status)
- `llm_brief_repairs_total`: 수정 건수 (mode, repair_type)
- `llm_brief_latency_seconds`: 지연 (히스토그램, p95/p99)

#### 2) RL Update 메트릭

```python
# RL 업데이트 기록
metrics_collector.record_rl_update(
    algorithm="ppo",
    policy_loss=0.45,
    value_loss=0.32,
    total_loss=0.77,
    reward=25.3,
    entropy=2.1,
    clip_fraction=0.15,
    kl_divergence=0.01
)
```

**메트릭**:
- `rl_update_total`: 총 업데이트 건수
- `rl_loss`: 손실 (policy_loss, value_loss, total_loss)
- `rl_reward`: 평균 보상
- `rl_entropy`: 정책 엔트로피
- `rl_clip_fraction`: Clipping 비율 (PPO)
- `rl_kl_divergence`: KL Divergence

#### 3) API 메트릭

```python
# API 요청 기록
metrics_collector.record_api_request(
    endpoint="/dna/create",
    method="POST",
    status_code=200,
    latency_seconds=3.1
)

# 또는 context manager 사용
with metrics_collector.track_api_request("/dna/create", "POST") as tracker:
    response = handle_request()
    tracker.set_status_code(200)
```

**메트릭**:
- `api_request_total`: 총 요청 건수 (endpoint, method, status)
- `api_request_latency_seconds`: 지연 (히스토그램, p95/p99)

### Grafana 대시보드

#### 설치

```bash
# Grafana 대시보드 임포트
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d @monitoring/grafana_operations_dashboard.json
```

#### 패널 구성

| 패널 | 설명 | 임계값 |
|------|------|--------|
| **LLM Brief 생성 건수** | 모드별 성공/실패 | - |
| **LLM Brief 수정 건수** | 타입별 수정 건수 | - |
| **LLM Brief 지연 (p95)** | 모드별 p95 지연 | fast: 2.5s, balanced: 3.2s, creative: 4.5s |
| **LLM Brief 지연 (p99)** | 모드별 p99 지연 | - |
| **RL Loss** | 정책/가치/총합 손실 | - |
| **RL 평균 보상** | 최근 에피소드 보상 | > 20 (green) |
| **RL 엔트로피** | 정책 엔트로피 | - |
| **RL Clip Fraction** | PPO clipping 비율 | < 0.3 (green) |
| **RL KL Divergence** | 정책 변화 | < 0.03 (green) |
| **API 요청 건수** | 엔드포인트별 성공/에러 | - |
| **API p95 지연** | 엔드포인트별 p95 | < 2.5s (green) |
| **API p99 지연** | 엔드포인트별 p99 | - |
| **시스템 요약 테이블** | 전체 메트릭 요약 | - |

---

## 통합 사용 예시

### FastAPI 앱 통합

```python
from fastapi import FastAPI
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from fragrance_ai.training.checkpoint_manager import CheckpointManager
from fragrance_ai.training.model_hot_reload import HotReloadManager
from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

app = FastAPI(title="Fragrance AI API")

# ========================================
# 1. 체크포인트 매니저 초기화
# ========================================

checkpoint_manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    save_interval=500,
    max_checkpoints=5
)

# ========================================
# 2. 핫 리로드 매니저 초기화
# ========================================

def load_model():
    # 모델 로드 로직
    return model

def warmup(model):
    # Warm-up 로직
    _ = model.generate("test")

reload_manager = HotReloadManager(
    model_loader=load_model,
    warmup_func=warmup,
    warmup_steps=5
)

reload_manager.initialize_first_worker()

# ========================================
# 3. 메트릭 컬렉터 초기화
# ========================================

metrics_collector = OperationsMetricsCollector()

# ========================================
# API 엔드포인트
# ========================================

@app.post("/dna/create")
async def create_dna(request: DNACreateRequest):
    # API 메트릭 추적
    with metrics_collector.track_api_request("/dna/create", "POST") as api_tracker:
        # LLM Brief 생성 추적
        with metrics_collector.track_llm_brief(request.mode) as llm_tracker:
            try:
                # 모델로 추론
                result = reload_manager.serve(
                    generate_brief,
                    user_text=request.user_text,
                    mode=request.mode
                )

                # 성공
                api_tracker.set_status_code(200)
                return result

            except Exception as e:
                # 실패
                llm_tracker.mark_failed()
                api_tracker.set_status_code(500)
                raise

@app.post("/rl/train")
async def train_rl(request: RLTrainRequest):
    # RL 학습
    for step in range(request.max_steps):
        loss, reward, entropy = train_step()

        # RL 메트릭 기록
        metrics_collector.record_rl_update(
            algorithm="ppo",
            policy_loss=loss["policy"],
            value_loss=loss["value"],
            total_loss=loss["total"],
            reward=reward,
            entropy=entropy,
            clip_fraction=0.15
        )

        # 체크포인트 저장
        if checkpoint_manager.should_save(step):
            checkpoint_manager.save_checkpoint(
                step=step,
                policy_net=policy_net,
                value_net=value_net,
                optimizer=optimizer,
                loss=loss["total"],
                reward=reward
            )

        # 롤백 체크
        should_rollback, _, reason = checkpoint_manager.check_rollback(
            current_step=step,
            current_loss=loss["total"],
            current_reward=reward
        )

        if should_rollback:
            logger.warning(f"롤백 트리거: {reason}")
            checkpoint_manager.rollback(policy_net, value_net, optimizer)

@app.post("/admin/model/reload")
async def reload_model():
    success = reload_manager.reload()
    return {"success": success}

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

---

## 운영 시나리오

### 시나리오 1: 손실 급등 감지 → 롤백

```
Step 1000: loss=0.5, reward=20.0  [체크포인트 저장]
Step 1100: loss=0.45, reward=22.0
Step 1200: loss=1.2, reward=18.0  [손실 2배 증가 감지]
              ↓
        [롤백 트리거]
              ↓
Step 1000으로 롤백 (loss=0.5, reward=20.0)
```

**로그**:
```
[WARNING] 롤백 트리거: Loss 급등: 0.50 -> 1.20 (2.40x)
[INFO] 롤백 완료: step=1000, loss=0.5000, reward=20.0000
```

### 시나리오 2: 모델 업데이트 (무중단)

```
현재: Qwen 2.5-7B v1
              ↓
[새 모델 준비: Qwen 2.5-7B v2]
              ↓
[v2 워커 생성 + 로드]
              ↓
[Warm-up 실행 (5 steps)]
              ↓
[트래픽 전환: v1 → v2]
              ↓
[v1 워커 배출 (30s)]
              ↓
[v1 워커 정지]
              ↓
현재: Qwen 2.5-7B v2 (서비스 중단 없음)
```

**로그**:
```
[INFO] === 모델 리로드 시작 ===
[INFO] [worker-1] 로드 + warm-up 시작
[INFO] [worker-1] 로드 + warm-up 완료
[INFO] 트래픽 전환: worker-0 → worker-1
[INFO] [worker-0] 요청 배출 시작 (in-flight: 5)
[INFO] [worker-0] 요청 배출 완료
[INFO] [worker-0] 정지
[INFO] === 모델 리로드 완료 ===
```

### 시나리오 3: 대시보드 알림

**조건**: LLM Brief p95 지연 > 4.5s (creative 모드)

```
Grafana 알림:
  제목: [ALERT] Creative 모드 지연 초과
  내용:
    - 현재 p95: 5.2s
    - 임계값: 4.5s
    - 초과율: +15.6%

  권장 조치:
    1. 모델 상태 확인 (/health/llm?model=qwen)
    2. 시스템 리소스 확인 (메모리/CPU)
    3. 필요 시 다운시프트 트리거
```

---

## 문제 해결 (Troubleshooting)

### 1. 체크포인트 로드 실패

**증상**:
```
[ERROR] 체크포인트 로드 실패: FileNotFoundError
```

**해결**:
1. 체크포인트 경로 확인
2. `checkpoint_history.json` 확인
3. 파일 권한 확인

### 2. 롤백 과다 발생

**증상**: 자주 롤백 트리거

**해결**:
1. 임계값 조정
   ```python
   rollback_conditions=RollbackConditions(
       kl_threshold=0.05,              # 0.03 → 0.05
       loss_increase_multiplier=3.0,   # 2.0 → 3.0
       reward_drop_threshold=0.4       # 0.3 → 0.4
   )
   ```
2. 학습률 낮추기
3. 배치 크기 늘리기

### 3. 핫 리로드 타임아웃

**증상**:
```
[WARNING] [worker-0] 배출 타임아웃 (남은 요청: 3)
```

**해결**:
1. `drain_timeout` 증가
   ```python
   reload_manager = HotReloadManager(
       drain_timeout=60.0  # 30s → 60s
   )
   ```
2. 진행 중인 요청 확인
3. 워커 상태 확인

### 4. 메트릭 수집 안 됨

**증상**: Grafana에 메트릭 표시 안 됨

**해결**:
1. Prometheus 스크래핑 확인
   ```bash
   curl http://localhost:8000/metrics | grep llm_brief
   ```
2. Prometheus 설정 확인 (prometheus.yml)
3. 메트릭 컬렉터 초기화 확인

---

## 배포 체크리스트

### 프로덕션 배포 전

- [ ] 체크포인트 디렉토리 생성 및 권한 설정
- [ ] 롤백 조건 임계값 검증
- [ ] 핫 리로드 warm-up 함수 테스트
- [ ] Prometheus 메트릭 엔드포인트 활성화 (/metrics)
- [ ] Grafana 대시보드 임포트
- [ ] 알림 규칙 설정 (Slack, PagerDuty)
- [ ] 로그 레벨 설정 (INFO 권장)
- [ ] 디스크 공간 확인 (체크포인트 저장용)

### 모니터링 설정

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'fragrance-ai'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 15s
    metrics_path: '/metrics'
```

```yaml
# grafana alerting
alert:
  - name: llm_brief_latency_high
    expr: histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode="creative"}[5m])) > 4.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Creative 모드 지연 초과"
```

---

## 성능 벤치마크

### 체크포인트 저장

| 모델 크기 | 저장 시간 | 디스크 용량 |
|-----------|-----------|-------------|
| 7B (Qwen) | ~15s | ~14GB |
| 8B (Llama) | ~18s | ~16GB |

### 핫 리로드

| 단계 | 소요 시간 |
|------|-----------|
| 모델 로드 | 30-60s |
| Warm-up (5 steps) | 5-10s |
| 트래픽 전환 | < 1s |
| 워커 배출 | 0-30s |
| **총 시간** | **35-100s** |

### 메트릭 오버헤드

| 메트릭 타입 | 오버헤드 |
|-------------|----------|
| Counter | < 0.1ms |
| Histogram | < 0.5ms |
| Gauge | < 0.1ms |
| Summary | < 1ms |

---

## 참고 문서

1. **코드**:
   - `fragrance_ai/training/checkpoint_manager.py`
   - `fragrance_ai/training/model_hot_reload.py`
   - `fragrance_ai/monitoring/operations_metrics.py`

2. **대시보드**:
   - `monitoring/grafana_operations_dashboard.json`

3. **관련 가이드**:
   - `QUALITY_KPI_GUIDE.md` - 품질 KPI 가이드
   - `STABILITY_GUARDS_GUIDE.md` - 안정성 가드 가이드
   - `IMPLEMENTATION_COMPLETE_SUMMARY.md` - 전체 구현 요약

---

**작성자**: Claude Code (Fragrance AI Team)
**문의**: GitHub Issues
