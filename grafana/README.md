# LLM Ensemble Monitoring with Prometheus + Grafana

이 디렉토리에는 LLM Ensemble 모니터링을 위한 Grafana 대시보드 설정이 포함되어 있습니다.

## Dashboard Overview

**llm_dashboard.json**에는 다음 메트릭을 시각화하는 12개의 패널이 포함되어 있습니다:

### 1. Request Rate (RPS)
- 초당 요청 수 (model, mode, status별)
- 시간대별 트래픽 패턴 파악

### 2. Active Requests
- 현재 처리 중인 요청 수
- 모델별 부하 모니터링

### 3. Inference Latency (p50, p95, p99)
- 추론 지연시간 백분위수
- SLA 준수 여부 확인
- 목표: FAST mode p95 < 2.5s, CREATIVE mode p95 < 4.0s

### 4. Error Rate
- 에러 발생률 (model, error_type별)
- timeout, oom, inference_error 등

### 5. Cache Hit Rate
- 캐시 적중률 (%)
- 80% 이상 유지 목표

### 6. Total Requests (Success vs Error)
- 성공/실패 요청 비율
- 전체 시스템 안정성 지표

### 7. Model Status
- 모델 로딩 상태 (Loaded/Not Loaded)
- 각 모델의 가용성 확인

### 8. RAM Memory Usage
- 모델별 메모리 사용량
- OOM 위험 사전 감지

### 9. Request Distribution by Model
- 모델별 요청 분포 (Pie Chart)
- 부하 분산 확인

### 10. Cache Operations
- Cache hits/misses 시계열
- 캐싱 효율성 모니터링

### 11. Average Latency by Mode
- Mode별 평균 지연시간 (Bar Gauge)
- fast/balanced/creative 모드 비교

### 12. Error Types Distribution
- 에러 타입별 분포 (Stacked Time Series)
- 주요 에러 원인 파악

## Setup Instructions

### 1. Prometheus 설치 및 설정

#### Prometheus 설치 (Docker)

```bash
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

#### prometheus.yml 설정

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fragrance-ai'
    static_configs:
      - targets: ['host.docker.internal:8000']
    metrics_path: '/health/metrics'
    scrape_interval: 10s
```

#### Prometheus 실행 (Windows)

```bash
# prometheus.exe 다운로드
# https://prometheus.io/download/

# prometheus.yml 생성 (위 내용)

# Prometheus 실행
prometheus.exe --config.file=prometheus.yml
```

Prometheus UI: http://localhost:9090

### 2. Grafana 설치 및 대시보드 import

#### Grafana 설치 (Docker)

```bash
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana
```

#### Grafana 실행 (Windows)

```bash
# grafana.exe 다운로드
# https://grafana.com/grafana/download

# Grafana 실행
grafana-server.exe
```

Grafana UI: http://localhost:3000 (기본 계정: admin/admin)

### 3. Prometheus 데이터 소스 추가

1. Grafana UI 접속
2. Configuration → Data Sources → Add data source
3. **Prometheus** 선택
4. URL: `http://localhost:9090` (또는 Prometheus 주소)
5. **Save & Test**

### 4. 대시보드 Import

#### 방법 1: JSON 파일 업로드

1. Grafana UI에서 **+** → **Import**
2. **Upload JSON file** 선택
3. `llm_dashboard.json` 파일 업로드
4. Prometheus 데이터 소스 선택
5. **Import** 클릭

#### 방법 2: JSON 복사/붙여넣기

1. Grafana UI에서 **+** → **Import**
2. `llm_dashboard.json` 내용을 복사
3. **Import via panel json** 텍스트 영역에 붙여넣기
4. **Load** → **Import**

### 5. FastAPI 애플리케이션 실행

```bash
# Prometheus 메트릭이 활성화된 상태로 실행
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

메트릭 엔드포인트 확인:
- http://localhost:8000/health/metrics (Prometheus format)
- http://localhost:8000/health/metrics/json (JSON format)

## Dashboard Variables

대시보드는 다음 변수들을 지원합니다:

- **datasource**: Prometheus 데이터 소스 선택
- **model**: 모델 필터 (qwen, mistral, llama) - 다중 선택 가능
- **mode**: 모드 필터 (fast, balanced, creative) - 다중 선택 가능

변수를 사용하여 특정 모델이나 모드에 대한 메트릭을 필터링할 수 있습니다.

## Annotations

대시보드에는 모델 리로드 이벤트를 자동으로 표시하는 annotation이 포함되어 있습니다:

- **Model Reload**: `llm_model_loaded` 메트릭의 변화를 감지하여 모델 리로드 시점을 파란색 선으로 표시

## Alert Rules (Optional)

Grafana에서 알림 규칙을 설정할 수 있습니다:

### High Error Rate Alert

```
alert: HighErrorRate
expr: rate(llm_errors_total[5m]) > 0.1
for: 5m
labels:
  severity: warning
annotations:
  summary: "High error rate detected"
  description: "Error rate is {{ $value }} errors/s for {{ $labels.model }}"
```

### High Latency Alert

```
alert: HighLatency
expr: histogram_quantile(0.95, rate(llm_inference_duration_seconds_bucket[5m])) > 3.0
for: 5m
labels:
  severity: warning
annotations:
  summary: "High latency detected"
  description: "p95 latency is {{ $value }}s for {{ $labels.model }}-{{ $labels.mode }}"
```

### Model Not Loaded Alert

```
alert: ModelNotLoaded
expr: llm_model_loaded == 0
for: 1m
labels:
  severity: critical
annotations:
  summary: "Model not loaded"
  description: "Model {{ $labels.model }} is not loaded"
```

## Metrics Reference

### Counters
- `llm_requests_total{model, mode, status}` - 총 요청 수
- `llm_errors_total{model, error_type}` - 총 에러 수
- `llm_cache_hits_total` - 캐시 히트 수
- `llm_cache_misses_total` - 캐시 미스 수

### Histograms
- `llm_inference_duration_seconds{model, mode}` - 추론 지연시간 분포

### Gauges
- `llm_model_loaded{model}` - 모델 로딩 상태 (0 or 1)
- `llm_model_memory_bytes{model, memory_type}` - 메모리 사용량
- `llm_active_requests{model}` - 활성 요청 수
- `llm_requests_per_second{model, mode}` - 초당 요청 수

## Troubleshooting

### 메트릭이 보이지 않는 경우

1. **Prometheus가 메트릭을 수집하고 있는지 확인**
   ```bash
   # Prometheus UI에서 Targets 확인
   http://localhost:9090/targets

   # Status가 UP인지 확인
   ```

2. **FastAPI 애플리케이션이 실행 중인지 확인**
   ```bash
   curl http://localhost:8000/health/metrics
   ```

3. **prometheus_client가 설치되어 있는지 확인**
   ```bash
   pip install prometheus-client
   ```

### Grafana에서 "No data" 표시되는 경우

1. **데이터 소스 연결 확인**
   - Configuration → Data Sources → Prometheus
   - "Save & Test" 버튼으로 연결 테스트

2. **시간 범위 확인**
   - 대시보드 우측 상단의 시간 범위 선택기 확인
   - "Last 1 hour" 또는 "Last 15 minutes" 선택

3. **쿼리 직접 실행**
   - Explore 메뉴에서 쿼리 직접 실행
   - 예: `llm_requests_total`

### prometheus_client 없을 때

`fragrance_ai/llm/metrics.py`는 prometheus_client가 없어도 작동합니다:
- Dummy metrics로 대체
- 에러 발생 없이 무시

## Production Best Practices

### 1. Retention Policy

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

storage:
  tsdb:
    retention.time: 15d
    retention.size: 50GB
```

### 2. Remote Write (장기 저장)

```yaml
remote_write:
  - url: "https://your-remote-storage/api/v1/write"
    basic_auth:
      username: admin
      password: secret
```

### 3. High Availability

- Prometheus: 다중 인스턴스 + Thanos
- Grafana: PostgreSQL 백엔드 + 다중 인스턴스

### 4. Security

```yaml
# prometheus.yml
global:
  external_labels:
    cluster: 'production'

scrape_configs:
  - job_name: 'fragrance-ai'
    basic_auth:
      username: 'prometheus'
      password: 'secure-password'
    scheme: https
    tls_config:
      ca_file: /path/to/ca.crt
      cert_file: /path/to/client.crt
      key_file: /path/to/client.key
```

## Docker Compose Setup

전체 모니터링 스택을 Docker Compose로 실행:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=15d'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/llm_dashboard.json:/etc/grafana/provisioning/dashboards/llm_dashboard.json
    depends_on:
      - prometheus

  app:
    build: .
    ports:
      - "8000:8000"
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

실행:
```bash
docker-compose up -d
```

## References

- Prometheus Documentation: https://prometheus.io/docs/
- Grafana Documentation: https://grafana.com/docs/
- Prometheus Python Client: https://github.com/prometheus/client_python

## Support

문제가 발생하거나 질문이 있으면 GitHub Issue를 생성해주세요.

---

**Last Updated**: 2025-01-15
**Dashboard Version**: 1.0.0
