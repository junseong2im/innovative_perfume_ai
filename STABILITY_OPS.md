# Stability & Operations Implementation

안정성/장애 대응 기능 구현 완료 문서

---

## Overview

이 문서는 LLM Ensemble의 안정성 및 운영 기능 구현을 설명합니다:

1. **Health Check Endpoints** - 헬스체크 엔드포인트
2. **Graceful Reload** - 그레이스풀 리로드
3. **Prometheus Metrics + Grafana Dashboard** - 옵스 대시보드

---

## 1. Health Check Endpoints

### 구현 파일

**`fragrance_ai/llm/health_check.py` (344 lines)**
- `LLMHealthChecker` 클래스: 모델 상태 체크
- `ModelHealth` 데이터클래스: 상태 정보 구조
- `HealthStatus` enum: healthy, degraded, unhealthy, unknown
- Kubernetes probes: `check_readiness()`, `check_liveness()`

**`app/routers/llm_health.py` (332 lines)**
- FastAPI 헬스체크 라우터

### API Endpoints

#### 1. GET /health/llm?model={qwen|mistral|llama}&run_inference={true|false}

개별 모델 헬스체크

**Response:**
```json
{
  "model_name": "qwen",
  "status": "healthy",
  "loaded": true,
  "inference_ready": true,
  "last_inference_ms": 45.2,
  "memory_mb": 8192.5,
  "gpu_memory_mb": 6144.0,
  "error_message": null,
  "checked_at": "2025-01-15T10:30:00Z",
  "uptime_seconds": 3600.5
}
```

**Status Codes:**
- `200`: Model is healthy or degraded
- `503`: Model is unhealthy

**Health Status Logic:**
- `unhealthy`: 추론 불가능, 모델 미로딩
- `degraded`: 메모리 >30GB, 추론 >5s
- `healthy`: 정상 작동

#### 2. GET /health/llm/all?run_inference={true|false}

전체 모델 헬스체크

**Response:**
```json
{
  "status": "healthy",
  "healthy_count": 3,
  "degraded_count": 0,
  "unhealthy_count": 0,
  "total_models": 3,
  "models": {
    "qwen": "healthy",
    "mistral": "healthy",
    "llama": "healthy"
  },
  "memory_mb": 8192.5,
  "gpu_available": true,
  "checked_at": "2025-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200`: At least one model is healthy
- `503`: All models are unhealthy

#### 3. GET /health/llm/reload/{model}

모델 리로드 상태 조회

**Response:**
```json
{
  "model": "qwen",
  "state": "warming_up",
  "progress_percent": 45.0,
  "message": "Warmup iteration 3/5",
  "started_at": "2025-01-15T10:25:00Z",
  "completed_at": null,
  "error": null
}
```

**Reload States:**
- `idle`: 리로드 없음
- `warming_up`: 새 모델 준비 중
- `ready`: 준비 완료
- `switching`: 트래픽 전환 중
- `completed`: 완료
- `failed`: 실패

#### 4. GET /health/ready

Kubernetes readiness probe

**Response:**
```json
{
  "status": "ready"
}
```

최소 1개 모델이 healthy면 `200`, 아니면 `503`

#### 5. GET /health/live

Kubernetes liveness probe

**Response:**
```json
{
  "status": "alive"
}
```

시스템이 deadlock 없으면 `200`, 아니면 `503`

#### 6. GET /health/metrics

Prometheus 메트릭 (text format)

**Response:**
```
# HELP llm_requests_total Total number of LLM requests
# TYPE llm_requests_total counter
llm_requests_total{model="qwen",mode="fast",status="success"} 1234
...
```

#### 7. GET /health/metrics/json

디버깅용 JSON 메트릭

**Response:**
```json
{
  "llm_requests_total_model=qwen,mode=fast,status=success": 1234,
  "llm_inference_duration_seconds_sum_model=qwen,mode=fast": 45.2,
  ...
}
```

### 사용 예시

```python
from fragrance_ai.llm.health_check import get_health_checker

# 헬스체커 가져오기
checker = get_health_checker()

# 모델 등록
checker.register_model("qwen", qwen_model_instance)

# 헬스체크 (추론 테스트 없이)
health = checker.check_model_health("qwen", run_inference=False)
print(f"Status: {health.status}, Memory: {health.memory_mb}MB")

# 헬스체크 (추론 테스트 포함)
health = checker.check_model_health("qwen", run_inference=True)
print(f"Latency: {health.last_inference_ms}ms")

# 전체 모델 체크
all_health = checker.check_all_models(run_inference=False)
for model, health in all_health.items():
    print(f"{model}: {health.status}")
```

---

## 2. Graceful Reload

### 구현 파일

**`fragrance_ai/llm/graceful_reload.py` (316 lines)**
- `GracefulReloadManager` 클래스
- `ReloadState` enum
- `ReloadStatus` 데이터클래스

### Features

- **Zero-downtime reload**: 서비스 중단 없이 모델 교체
- **5-phase process**: Load → Warm-up → Switch → Cleanup → Complete
- **Rollback on failure**: 실패 시 이전 모델로 자동 복구
- **Progress tracking**: 0-100% 진행률 추적
- **Thread-safe**: 동시 리로드 방지

### Reload Process

```
Phase 1: Load New Model (10%)
├─ loader_func() 실행
├─ 새 모델 로딩
└─ new_models에 저장

Phase 2: Warm-up (30-80%)
├─ warmup_func() 반복 실행
├─ 추론 테스트 (기본 5회)
└─ 실패 시 rollback

Phase 3: Switch Traffic (80-90%)
├─ active_models 교체
├─ 0.5초 대기 (in-flight requests 완료)
└─ 트래픽 전환 완료

Phase 4: Cleanup (90-100%)
├─ old_models 삭제
├─ 메모리 해제
└─ new_models 정리

Phase 5: Complete (100%)
└─ 상태 업데이트
```

### 사용 예시

```python
from fragrance_ai.llm.graceful_reload import get_reload_manager

manager = get_reload_manager()

# 모델 로더 함수
def load_qwen():
    return AutoModel.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Warm-up 함수
def warmup_qwen(model):
    try:
        result = model.generate("test input")
        return result is not None
    except:
        return False

# 리로드 시작
status = manager.start_reload(
    model_name="qwen",
    loader_func=load_qwen,
    warmup_func=warmup_qwen,
    warmup_iterations=5
)

# 상태 조회 (백그라운드 실행 중)
status = manager.get_reload_status("qwen")
print(f"Progress: {status.progress_percent}%")
print(f"State: {status.state}")
print(f"Message: {status.message}")

# 활성 모델 가져오기
active_model = manager.get_active_model("qwen")
```

### Error Handling

리로드 실패 시:
1. 에러 메시지 기록
2. 이전 모델로 rollback
3. 새 모델 삭제
4. 상태를 `failed`로 설정

```python
# 리로드 실패 시
{
  "state": "failed",
  "progress_percent": 0.0,
  "message": "Reload failed: Model loader returned None",
  "error": "Model loader returned None"
}

# active_models는 이전 모델 유지
```

---

## 3. Prometheus Metrics + Grafana Dashboard

### 구현 파일

**`fragrance_ai/llm/metrics.py` (384 lines)**
- Prometheus 메트릭 정의
- 추적 헬퍼 함수
- 데코레이터

**`grafana/llm_dashboard.json` (700+ lines)**
- Grafana 대시보드 JSON
- 12개 패널 (Request Rate, Latency, Errors, Cache, Memory, etc.)

**`grafana/prometheus.yml`**
- Prometheus 설정 파일

**`grafana/docker-compose.yml`**
- 전체 스택 Docker Compose 설정

**`grafana/README.md`**
- 설치 및 사용 가이드

**`grafana/provisioning/`**
- Grafana 자동 프로비저닝 설정

### Prometheus Metrics

#### Counters
- `llm_requests_total{model, mode, status}` - 총 요청 수
- `llm_errors_total{model, error_type}` - 총 에러 수
- `llm_cache_hits_total` - 캐시 히트 수
- `llm_cache_misses_total` - 캐시 미스 수

#### Histograms
- `llm_inference_duration_seconds{model, mode}` - 추론 지연시간 분포
  - Buckets: 0.1s, 0.5s, 1.0s, 2.0s, 5.0s, 10.0s, 30.0s, ∞

#### Gauges
- `llm_model_loaded{model}` - 모델 로딩 상태 (0 or 1)
- `llm_model_memory_bytes{model, memory_type}` - 메모리 사용량
- `llm_active_requests{model}` - 활성 요청 수
- `llm_requests_per_second{model, mode}` - 초당 요청 수

#### Info
- `llm_ensemble` - 앙상블 정보 (모델 버전 등)

### 메트릭 수집

```python
from fragrance_ai.llm.metrics import (
    track_inference_time,
    record_request,
    record_error,
    record_cache_hit,
    update_model_status
)

# Context manager로 지연시간 추적
with track_inference_time('qwen', 'fast'):
    result = model.generate(input_text)

# 요청 기록
record_request('qwen', 'fast', 'success')

# 에러 기록
record_error('qwen', 'timeout')

# 캐시 히트 기록
record_cache_hit()

# 모델 상태 업데이트
update_model_status('qwen', loaded=True, memory_bytes=8_589_934_592)

# 데코레이터 사용
@track_llm_request('qwen', 'fast')
def infer_brief(text):
    return model.generate(text)
```

### Grafana Dashboard

12개 패널로 구성된 대시보드:

1. **Request Rate (RPS)** - 초당 요청 수 (모델/모드/상태별)
2. **Active Requests** - 현재 처리 중인 요청
3. **Inference Latency (p50, p95, p99)** - 지연시간 백분위수
4. **Error Rate** - 에러 발생률
5. **Cache Hit Rate** - 캐시 적중률 (%)
6. **Total Requests** - 성공/실패 요청 비율
7. **Model Status** - 모델 로딩 상태
8. **RAM Memory Usage** - 메모리 사용량
9. **Request Distribution by Model** - 모델별 요청 분포 (Pie Chart)
10. **Cache Operations** - 캐시 히트/미스 시계열
11. **Average Latency by Mode** - 모드별 평균 지연시간 (Bar Gauge)
12. **Error Types Distribution** - 에러 타입별 분포 (Stacked)

### Quick Start

#### Docker Compose로 전체 스택 실행

```bash
cd grafana/
docker-compose up -d

# 서비스 확인
docker-compose ps

# 로그 확인
docker-compose logs -f
```

접속:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

#### 수동 설치

```bash
# Prometheus 실행
prometheus --config.file=grafana/prometheus.yml

# Grafana 실행
grafana-server

# FastAPI 실행
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

#### 대시보드 Import

1. Grafana UI 접속: http://localhost:3000
2. **+** → **Import**
3. `grafana/llm_dashboard.json` 업로드
4. Prometheus 데이터 소스 선택
5. **Import** 클릭

### Optional Dependencies

`prometheus_client`가 없어도 작동:
- Dummy metrics로 대체
- 에러 발생 없음

설치:
```bash
pip install prometheus-client
```

---

## 통합 테스트

### 헬스체크 테스트

```bash
# 개별 모델 체크
curl http://localhost:8000/health/llm?model=qwen

# 전체 모델 체크
curl http://localhost:8000/health/llm/all

# 추론 테스트 포함
curl http://localhost:8000/health/llm?model=qwen&run_inference=true

# Kubernetes probes
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/live
```

### 메트릭 확인

```bash
# Prometheus 형식
curl http://localhost:8000/health/metrics

# JSON 형식
curl http://localhost:8000/health/metrics/json

# Prometheus UI에서 쿼리
# http://localhost:9090
# Query: rate(llm_requests_total[1m])
```

### 리로드 테스트

```python
from fragrance_ai.llm.graceful_reload import reload_model

# 리로드 시작
status = reload_model(
    model_name="qwen",
    loader_func=load_qwen,
    warmup_func=warmup_qwen,
    warmup_iterations=5
)

# API로 상태 확인
# GET http://localhost:8000/health/llm/reload/qwen
```

---

## Kubernetes Integration

### Deployment with Probes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fragrance-ai-api
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: api
        image: fragrance-ai:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
```

### ServiceMonitor for Prometheus Operator

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fragrance-ai-api
spec:
  selector:
    matchLabels:
      app: fragrance-ai-api
  endpoints:
  - port: api
    path: /health/metrics
    interval: 15s
```

---

## Production Checklist

### Health Checks
- [x] 헬스체크 엔드포인트 구현
- [x] Kubernetes probes 지원
- [x] 모델별 상태 추적
- [x] 메모리 사용량 모니터링
- [x] 추론 테스트 옵션

### Graceful Reload
- [x] Zero-downtime 리로드
- [x] Warm-up 메커니즘
- [x] Rollback on failure
- [x] Progress tracking
- [x] Thread-safe 구현

### Monitoring
- [x] Prometheus 메트릭 정의
- [x] Grafana 대시보드 생성
- [x] Request rate tracking
- [x] Latency percentiles (p50, p95, p99)
- [x] Error rate monitoring
- [x] Cache hit rate
- [x] Memory usage tracking
- [x] Model status gauges

### Documentation
- [x] API 엔드포인트 문서화
- [x] 설치 가이드 작성
- [x] Docker Compose 설정
- [x] Kubernetes 통합 예시
- [x] Troubleshooting 가이드

---

## Files Created

### Core Implementation
1. `fragrance_ai/llm/health_check.py` (344 lines)
2. `fragrance_ai/llm/graceful_reload.py` (316 lines)
3. `fragrance_ai/llm/metrics.py` (384 lines)
4. `app/routers/llm_health.py` (332 lines)

### Monitoring Stack
5. `grafana/llm_dashboard.json` (700+ lines)
6. `grafana/prometheus.yml` (110 lines)
7. `grafana/docker-compose.yml` (140 lines)
8. `grafana/provisioning/datasources/prometheus.yml`
9. `grafana/provisioning/dashboards/dashboards.yml`

### Documentation
10. `grafana/README.md` (comprehensive setup guide)
11. `STABILITY_OPS.md` (this file)

### Modified Files
12. `app/main.py` - Added health router integration

---

## Key Metrics and SLAs

### Latency Targets
- **FAST mode**: p95 < 2.5s
- **CREATIVE mode**: p95 < 4.0s
- **BALANCED mode**: p95 < 3.0s

### Availability Targets
- **Service uptime**: 99.9%
- **Model availability**: At least 1 model healthy at all times
- **Reload downtime**: 0 seconds (zero-downtime reload)

### Performance Targets
- **Request rate**: Support 10+ RPS sustained
- **Cache hit rate**: >80%
- **Error rate**: <1%
- **Memory usage**: <30GB per model (degraded if exceeded)

### Monitoring Targets
- **Metrics collection**: Every 10-15 seconds
- **Dashboard refresh**: Every 10 seconds
- **Retention**: 15 days minimum

---

## Next Steps (Optional)

### Advanced Monitoring
- [ ] Distributed tracing (Jaeger/Zipkin)
- [ ] Custom alerting rules (Alertmanager)
- [ ] Log aggregation (ELK/Loki)
- [ ] APM integration (DataDog/New Relic)

### Advanced Reload
- [ ] Canary deployments (gradual traffic shift)
- [ ] A/B testing (traffic splitting)
- [ ] Blue-green deployments
- [ ] Feature flags for model selection

### Scalability
- [ ] Horizontal pod autoscaling (HPA)
- [ ] Model sharding across replicas
- [ ] Load balancing strategies
- [ ] Distributed caching (Redis)

---

## Summary

모든 안정성/옵스 기능이 성공적으로 구현되었습니다:

✅ **Health Check Endpoints** - 7개 엔드포인트 (LLM 상태, 리로드, K8s probes, 메트릭)
✅ **Graceful Reload** - Zero-downtime 5-phase reload with rollback
✅ **Prometheus + Grafana** - 12-panel dashboard with comprehensive metrics

시스템은 이제 프로덕션 수준의 모니터링과 안정성 기능을 갖추었습니다.

---

**Last Updated**: 2025-01-15
**Version**: 1.0.0
