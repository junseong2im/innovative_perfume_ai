# Artisan Observability Guide

## 개요

Artisan의 관측성(Observability)은 **3가지 핵심 요소**로 구성됩니다:

1. **로그 (Logs)** - 구조화된 JSON 로그
2. **메트릭 (Metrics)** - Prometheus + Grafana
3. **트레이싱 (Tracing)** - OpenTelemetry

---

## 1. 구조화된 JSON 로깅

### 1.1 핵심 로그 키

#### LLM 로그
```json
{
  "timestamp": "2025-10-14T10:30:00.123Z",
  "level": "INFO",
  "component": "LLM",
  "log_type": "llm_brief",
  "mode": "fast",
  "qwen_ok": true,
  "mistral_fix": false,
  "hints": "citrus notes recommended",
  "elapsed_ms": 1847.32,
  "cache_hit": false,
  "brief_style": "fresh",
  "brief_intensity": 0.7
}
```

**핵심 필드:**
- `mode` - fast/balanced/creative
- `qwen_ok` - Qwen 모델 성공 여부
- `mistral_fix` - Mistral 모델 보정 여부
- `hints` - 생성된 힌트 (있는 경우)
- `elapsed_ms` - 전체 소요 시간

#### RL 로그
```json
{
  "timestamp": "2025-10-14T10:30:00.123Z",
  "level": "INFO",
  "component": "RL",
  "log_type": "rl_update",
  "algo": "PPO",
  "loss": 0.002341,
  "reward": 15.67,
  "entropy": 0.0085,
  "clip_frac": 0.12
}
```

**핵심 필드:**
- `algo` - PPO 또는 REINFORCE
- `loss` - 정책 손실
- `reward` - 평균 보상
- `entropy` - 정책 엔트로피
- `clip_frac` - PPO 클리핑 비율

#### API 로그
```json
{
  "timestamp": "2025-10-14T10:30:00.123Z",
  "level": "INFO",
  "component": "API",
  "log_type": "api_request",
  "method": "POST",
  "endpoint": "/api/v1/evolve/options",
  "status_code": 200,
  "latency_ms": 2145.67
}
```

**핵심 필드:**
- `method` - HTTP 메서드
- `endpoint` - API 경로
- `status_code` - HTTP 상태 코드
- `latency_ms` - 응답 지연시간

### 1.2 사용법

```python
from fragrance_ai.observability import llm_logger, rl_logger, orchestrator_logger

# LLM 로그
llm_logger.log_brief(
    user_text="I want a fresh summer fragrance",
    brief={"style": "fresh", "intensity": 0.7},
    model="qwen-2.5-72b",
    mode="fast",
    latency_ms=1847.32,
    qwen_ok=True,
    mistral_fix=False,
    hints="citrus notes recommended"
)

# RL 로그
rl_logger.log_update(
    algorithm="PPO",
    loss=0.002341,
    reward=15.67,
    entropy=0.0085,
    clip_frac=0.12
)

# API 로그
orchestrator_logger.log_api_request(
    method="POST",
    path="/api/v1/evolve/options",
    status_code=200,
    response_time_ms=2145.67
)
```

### 1.3 보안 - 로그 마스킹

**자동으로 마스킹되는 정보:**
- API 키/토큰
- 이메일 주소
- 전화번호
- 신용카드 번호
- AWS 키
- 데이터베이스 자격증명

```python
from fragrance_ai.observability import LogMasker

# 예시
text = "api_key=sk-1234567890abcdef email=user@example.com"
masked = LogMasker.mask_all(text)
# 결과: "api_key=***MASKED*** email=***EMAIL_MASKED***"
```

---

## 2. 메트릭 (Prometheus + Grafana)

### 2.1 Prometheus 메트릭

#### LLM 메트릭
```
# LLM 브리프 생성 지연시간
llm_brief_latency_seconds{mode="fast|balanced|creative"}

# LLM 모델 상태
llm_model_status_total{model="qwen|mistral|llama", status="ok|error|timeout"}

# 캐시 히트율
cache_hits_total{cache_type="llm_brief"}
cache_misses_total{cache_type="llm_brief"}
```

#### RL 메트릭
```
# RL 업데이트 카운트
rl_updates_total{algo="PPO|REINFORCE"}

# RL 보상
rl_reward{algo="PPO|REINFORCE"}

# RL 손실
rl_loss{algo="PPO|REINFORCE"}

# RL 엔트로피
rl_entropy{algo="PPO|REINFORCE"}
```

#### API 메트릭
```
# API 요청 카운트
api_requests_total{method="GET|POST", endpoint="/api/...", status="200|400|500"}

# API 응답 시간 (히스토그램)
api_response_seconds{method="GET|POST", endpoint="/api/..."}

# p95 지연시간
histogram_quantile(0.95, api_response_seconds)
```

#### 서킷브레이커 메트릭
```
# 서킷브레이커 폴백 카운트
circuit_breaker_fallback_total{service="llm", fallback_type="cache|default"}

# 서킷브레이커 다운그레이드 카운트
circuit_breaker_downgrade_total{service="llm", from_tier="creative", to_tier="balanced"}
```

### 2.2 Grafana 대시보드

#### 패널 1: LLM 지연시간 (Mode별)
**쿼리:**
```promql
rate(llm_brief_latency_seconds_sum{mode="fast"}[5m])
/ rate(llm_brief_latency_seconds_count{mode="fast"}[5m])
```

**시각화:** Time Series Graph

#### 패널 2: 캐시 히트율
**쿼리:**
```promql
rate(cache_hits_total[5m])
/ (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))
```

**시각화:** Gauge (0-100%)

#### 패널 3: RL 보상 추세 (알고리즘별)
**쿼리:**
```promql
rl_reward{algo="PPO"}
rl_reward{algo="REINFORCE"}
```

**시각화:** Time Series Graph

#### 패널 4: API p95 지연시간
**쿼리:**
```promql
histogram_quantile(0.95,
  rate(api_response_seconds_bucket[5m])
)
```

**시각화:** Gauge + Threshold (2.5s)

#### 패널 5: 서킷브레이커 전환 카운트
**쿼리:**
```promql
rate(circuit_breaker_fallback_total[5m])
rate(circuit_breaker_downgrade_total[5m])
```

**시각화:** Bar Chart

### 2.3 Grafana + Loki 통합 (JSON 로그)

Loki를 사용하여 JSON 로그를 쿼리:

#### LLM 모드별 지연시간 (JSON 로그 기반)
```logql
{component="LLM", log_type="llm_brief"}
| json
| mode="fast"
| avg_over_time(elapsed_ms) by (mode)
```

#### RL 알고리즘별 보상 (JSON 로그 기반)
```logql
{component="RL", log_type="rl_update"}
| json
| avg(reward) by (algo)
```

#### API 에러율 (JSON 로그 기반)
```logql
sum(rate({component="API", status_code=~"5.."} [5m]))
/ sum(rate({component="API"} [5m]))
```

---

## 3. 트레이싱 (OpenTelemetry)

### 3.1 설정

#### 방법 1: Jaeger 사용
```python
from fragrance_ai.tracing import setup_tracing, TracingConfig

config = TracingConfig(
    service_name="fragrance-ai",
    jaeger_endpoint="localhost:6831",  # Jaeger agent
    sample_rate=1.0  # 100% sampling
)

tracer = setup_tracing(config)
```

#### 방법 2: OTLP 사용 (Grafana Tempo)
```python
config = TracingConfig(
    service_name="fragrance-ai",
    otlp_endpoint="http://localhost:4317",  # Tempo endpoint
    sample_rate=0.1  # 10% sampling
)

tracer = setup_tracing(config)
```

### 3.2 전체 요청 추적

**시나리오:** `/evolve/options` → `/evolve/feedback` 전체 플로우

```python
from fragrance_ai.tracing import artisan_tracer

# Evolution request
with artisan_tracer.trace_evolution(
    experiment_id="exp_123",
    algorithm="PPO",
    num_options=3
):
    # LLM call
    with artisan_tracer.trace_llm_call(
        model="qwen-2.5-72b",
        mode="balanced",
        cache_hit=False
    ):
        brief = llm_service.generate_brief(user_text)

    # RL evolution
    with artisan_tracer.trace_rl_update(
        algorithm="PPO",
        iteration=1
    ):
        options = rl_service.evolve(dna, brief)

# Feedback submission
with artisan_tracer.trace_feedback(
    experiment_id="exp_123",
    chosen_id="pheno_1",
    rating=5
):
    rl_service.submit_feedback(feedback)
```

### 3.3 FastAPI 통합

```python
from fastapi import FastAPI
from fragrance_ai.tracing import create_fastapi_middleware

app = FastAPI()

# Add tracing middleware
app.add_middleware(create_fastapi_middleware())

@app.post("/api/v1/evolve/options")
async def evolve_options(request: EvolutionRequest):
    # 자동으로 트레이스됨
    return evolution_service.create_options(request)
```

### 3.4 추적 가능한 정보

**스팬(Span) 속성:**
- `experiment.id` - 실험 ID
- `evolution.algorithm` - PPO/REINFORCE
- `llm.model` - LLM 모델 이름
- `llm.mode` - fast/balanced/creative
- `llm.latency_ms` - LLM 호출 지연시간
- `rl.iteration` - RL 반복 횟수
- `http.method` - HTTP 메서드
- `http.url` - 요청 URL
- `http.status_code` - HTTP 상태 코드

**트레이스 예시:**
```
[Trace: exp_123]
  └─ evolution.request (2500ms)
      ├─ llm.call (1800ms)
      │   └─ qwen-2.5-72b inference
      ├─ rl.update (600ms)
      │   ├─ policy forward
      │   ├─ loss calculation
      │   └─ optimizer step
      └─ ga.generation (100ms)
          ├─ crossover
          └─ mutation
```

---

## 4. 배포 및 설정

### 4.1 Docker Compose

```yaml
version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources

  # Loki (JSON 로그)
  loki:
    image: grafana/loki:latest
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml

  # Promtail (로그 수집)
  promtail:
    image: grafana/promtail:latest
    volumes:
      - /var/log:/var/log
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml

  # Jaeger (트레이싱)
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "5775:5775/udp"  # Zipkin thrift compact
      - "6831:6831/udp"  # Jaeger agent
      - "6832:6832/udp"  # Jaeger agent
      - "5778:5778"      # Config
      - "16686:16686"    # UI
      - "14268:14268"    # Collector HTTP
      - "14250:14250"    # Collector gRPC

  # Tempo (트레이싱 - Grafana native)
  tempo:
    image: grafana/tempo:latest
    ports:
      - "3200:3200"   # Tempo
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
    command: [ "-config.file=/etc/tempo.yaml" ]
    volumes:
      - ./tempo.yaml:/etc/tempo.yaml

  # Artisan API
  artisan-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - JAEGER_ENDPOINT=jaeger:6831
      - PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
    depends_on:
      - prometheus
      - loki
      - jaeger
```

### 4.2 Prometheus 설정 (`prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fragrance-ai'
    static_configs:
      - targets: ['artisan-api:8000']
    metrics_path: '/metrics'
```

### 4.3 Promtail 설정 (JSON 로그 수집)

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: fragrance-ai-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: fragrance-ai
          __path__: /var/log/fragrance-ai/*.log

    pipeline_stages:
      # JSON 로그 파싱
      - json:
          expressions:
            timestamp: timestamp
            level: level
            component: component
            log_type: log_type
            message: message

      # 타임스탬프 추출
      - timestamp:
          source: timestamp
          format: RFC3339Nano

      # 라벨 추가
      - labels:
          level:
          component:
          log_type:
```

### 4.4 환경 변수

```bash
# Tracing
export JAEGER_ENDPOINT=localhost:6831
export OTLP_ENDPOINT=http://localhost:4317

# Prometheus
export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus

# Artisan
export ARTISAN_ENV=prod
```

---

## 5. 쿼리 예시

### 5.1 Grafana 대시보드 쿼리

#### LLM 모드별 평균 지연시간 (Loki)
```logql
avg_over_time(
  {component="LLM", log_type="llm_brief"}
  | json
  | __error__ = ""
  | unwrap elapsed_ms [5m]
) by (mode)
```

#### RL 알고리즘별 손실 추세 (Loki)
```logql
{component="RL", log_type="rl_update"}
| json
| line_format "{{.algo}}: loss={{.loss}}"
```

#### API 에러율 (Prometheus)
```promql
sum(rate(api_requests_total{status=~"5.."}[5m]))
/ sum(rate(api_requests_total[5m]))
```

#### p95 지연시간 (Prometheus)
```promql
histogram_quantile(0.95,
  sum(rate(api_response_seconds_bucket[5m])) by (le, endpoint)
)
```

### 5.2 알람 규칙 (Prometheus)

```yaml
groups:
  - name: artisan_alerts
    interval: 30s
    rules:
      # p95 지연시간 초과
      - alert: HighAPILatency
        expr: |
          histogram_quantile(0.95,
            rate(api_response_seconds_bucket[5m])
          ) > 2.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API p95 latency exceeded threshold"
          description: "p95 latency is {{ $value }}s (threshold: 2.5s)"

      # 에러율 급증
      - alert: HighErrorRate
        expr: |
          sum(rate(api_requests_total{status=~"5.."}[5m]))
          / sum(rate(api_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High API error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # 서킷브레이커 다운그레이드
      - alert: CircuitBreakerDowngrade
        expr: rate(circuit_breaker_downgrade_total[5m]) > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Circuit breaker downgrade detected"
          description: "LLM service downgraded due to failures"

      # RL 보상 급락
      - alert: RLRewardDrop
        expr: |
          rl_reward < 10.0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "RL reward dropped below threshold"
          description: "Current reward: {{ $value }}"
```

---

## 6. 베스트 프랙티스

### 6.1 로그 레벨

- **DEBUG** - 상세한 디버깅 정보 (dev only)
- **INFO** - 정상 작동 로그 (기본값)
- **WARNING** - 경고 (성능 저하, 재시도)
- **ERROR** - 에러 (복구 가능)
- **CRITICAL** - 치명적 에러 (서비스 중단)

### 6.2 샘플링

**프로덕션 환경:**
- 로그: 100% (모든 로그 수집)
- 메트릭: 100% (모든 메트릭 수집)
- 트레이싱: 1-10% (샘플링으로 부하 감소)

### 6.3 보안

- 모든 로그에 자동 마스킹 적용
- user_id는 SHA256 해시로 익명화
- API 키/비밀번호는 절대 로그에 기록 금지
- PII(개인식별정보)는 자동 마스킹

### 6.4 성능

- 로그는 비동기로 처리
- 메트릭은 인메모리 카운터 사용
- 트레이싱은 샘플링으로 오버헤드 최소화
- Prometheus 스크레이프 간격: 15초

---

## 7. 문제 해결

### 7.1 로그가 Loki에 나타나지 않음

**확인 사항:**
1. Promtail이 실행 중인지 확인
2. 로그 파일 경로가 올바른지 확인
3. JSON 형식이 올바른지 확인
4. Loki URL이 올바른지 확인

```bash
# Promtail 로그 확인
docker logs promtail

# Loki 상태 확인
curl http://localhost:3100/ready
```

### 7.2 메트릭이 Prometheus에 나타나지 않음

**확인 사항:**
1. `/metrics` 엔드포인트 접근 가능 확인
2. Prometheus 스크레이프 설정 확인
3. 메트릭 이름이 올바른지 확인

```bash
# 메트릭 엔드포인트 확인
curl http://localhost:8000/metrics

# Prometheus 타겟 상태 확인
# http://localhost:9090/targets
```

### 7.3 트레이스가 Jaeger에 나타나지 않음

**확인 사항:**
1. Jaeger 엔드포인트 설정 확인
2. OpenTelemetry 설치 확인
3. 트레이서 초기화 확인

```bash
# Jaeger UI 확인
# http://localhost:16686

# 환경 변수 확인
echo $JAEGER_ENDPOINT
```

---

## 8. 요약

| 요소 | 도구 | 포트 | 용도 |
|------|------|------|------|
| **로그** | Loki + Promtail | 3100 | JSON 로그 수집/쿼리 |
| **메트릭** | Prometheus | 9090 | 시계열 메트릭 |
| **대시보드** | Grafana | 3000 | 시각화 |
| **트레이싱** | Jaeger | 16686 | 분산 추적 |
| **트레이싱** | Tempo | 3200 | Grafana native 추적 |

**핵심 로그 키:**
- `llm_brief{mode, qwen_ok, mistral_fix, hints, elapsed_ms}`
- `rl_update{algo, loss, reward, entropy, clip_frac}`
- `api_request{status_code, latency_ms, endpoint}`

**관측성 체크리스트:**
- ✅ 모든 로그는 JSON 형식
- ✅ 민감 정보는 자동 마스킹
- ✅ 메트릭은 Prometheus로 수집
- ✅ Grafana 대시보드로 시각화
- ✅ OpenTelemetry로 전체 요청 추적
- ✅ 알람 규칙 설정 (p95, 에러율)

---

**Artisan은 완전한 관측성을 통해 프로덕션 안정성을 보장합니다!** 📊
