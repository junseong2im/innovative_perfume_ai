# Go / No-Go 배포 게이트 가이드

배포 전 자동 체크로 안전성을 보장합니다.

## 기준 (간단 명확)

### ✅ GO (배포 허용)
- **모든 테스트 통과** ✓
- **KPI 달성** ✓

### ⛔ NO-GO (배포 금지)
- **스키마 실패율 > 0%** (무관용)
- **p95 초과 지속** (임계값 초과)


## 체크 항목

### 1. 테스트 (All or Nothing)
```python
test_suites = [
    "tests/test_llm_ensemble_operation.py",  # LLM 앙상블
    "tests/test_ga.py",                      # GA 알고리즘
    "tests/test_ifra.py",                    # IFRA 규제
    "tests/test_end_to_end_evolution.py"     # E2E 진화
]
```
**기준:** 모든 테스트 100% 통과

### 2. KPI 임계값

| 메트릭 | 임계값 | 설명 |
|--------|--------|------|
| **LLM p95 (fast)** | < 2.5s | 빠른 모드 지연 |
| **LLM p95 (balanced)** | < 3.2s | 균형 모드 지연 |
| **LLM p95 (creative)** | < 4.5s | 창의 모드 지연 |
| **API p95** | < 2.5s | API 지연 |
| **API p99** | < 5.0s | API 지연 (99%) |
| **API 에러율** | < 1% | 에러 비율 |
| **스키마 실패율** | = 0% | **무관용** |
| **RL Reward** | > 10.0 | 학습 품질 |
| **RL KL Divergence** | < 0.03 | 정책 안정성 |


## 사용 방법

### CLI 실행

```bash
# Windows
scripts\check_deployment_gate.bat

# Linux/Mac
bash scripts/check_deployment_gate.sh
```

### Python 직접 실행

```python
from fragrance_ai.deployment.go_nogo_gate import GoNoGoGate

# 게이트 초기화
gate = GoNoGoGate(
    prometheus_url="http://localhost:9090"
)

# 평가 실행
result = gate.evaluate()

# 결과 확인
if result.decision == DeploymentDecision.GO:
    print("✅ GO: Safe to deploy")
else:
    print("⛔ NO-GO: Do not deploy")
    for reason in result.reasons:
        print(f"  - {reason}")
```


## CI/CD 통합

### GitHub Actions

```yaml
name: Deployment Gate

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  go-nogo-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-json-report

      - name: Run Go/No-Go Gate
        run: |
          python -m fragrance_ai.deployment.go_nogo_gate \
            --report-file deployment_gate_report.txt \
            --exit-code

      - name: Upload Report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: deployment-gate-report
          path: deployment_gate_report.txt

      - name: Check Result
        run: |
          if [ $? -eq 0 ]; then
            echo "✅ GO: Safe to deploy"
          else
            echo "⛔ NO-GO: Deployment blocked"
            exit 1
          fi
```


## 출력 예시

### GO (배포 허용)

```
============================================================
🚦 GO / NO-GO DEPLOYMENT GATE
============================================================
🧪 Running all test suites...
✓ Test suite passed: tests/test_llm_ensemble_operation.py
✓ Test suite passed: tests/test_ga.py
✓ Test suite passed: tests/test_ifra.py
✓ Test suite passed: tests/test_end_to_end_evolution.py
✓ All tests passed

📊 Checking KPIs...
✓ All KPIs passed

📈 Collecting metrics from Prometheus...

============================================================
✅ DECISION: GO - Safe to deploy
============================================================

GO / NO-GO DEPLOYMENT GATE REPORT
============================================================
Timestamp: 2025-10-14 12:34:56
Decision: GO

✅ SAFE TO DEPLOY

All checks passed:
  ✓ Tests: PASSED
  ✓ KPIs: PASSED
  ✓ Schema: PASSED (0% failure)
  ✓ Latency: PASSED (p95/p99 within limits)

Metrics:
  llm_p95_fast: 2.1s
  llm_p95_balanced: 2.8s
  llm_p95_creative: 4.2s
  api_p95_latency: 2.3s
  api_error_rate: 0.5%
  schema_failure_rate: 0.0%
  rl_reward: 12.5
  rl_kl_divergence: 0.015
============================================================
```

### NO-GO (배포 금지)

```
============================================================
🚦 GO / NO-GO DEPLOYMENT GATE
============================================================
🧪 Running all test suites...
✓ Test suite passed: tests/test_llm_ensemble_operation.py
❌ Test suite failed: tests/test_ga.py
✓ Test suite passed: tests/test_ifra.py
✓ Test suite passed: tests/test_end_to_end_evolution.py

📊 Checking KPIs...
❌ KPI violation: LLM p95 (creative) exceeded: 5.2s > 4.5s
❌ KPI violation: 🚨 Schema failure rate > 0%: 0.02% (ZERO TOLERANCE)

============================================================
⛔ DECISION: NO-GO - Do not deploy
Reasons:
  - ❌ Tests failed
  - LLM p95 (creative) exceeded: 5.2s > 4.5s
  - 🚨 CRITICAL: Schema failure rate > 0% (ZERO TOLERANCE)
============================================================

GO / NO-GO DEPLOYMENT GATE REPORT
============================================================
Timestamp: 2025-10-14 12:45:30
Decision: NO-GO

⛔ DO NOT DEPLOY

Failures detected:
  1. ❌ Tests failed
  2. LLM p95 (creative) exceeded: 5.2s > 4.5s
  3. 🚨 CRITICAL: Schema failure rate > 0% (ZERO TOLERANCE)

Metrics:
  llm_p95_fast: 2.1s
  llm_p95_balanced: 2.8s
  llm_p95_creative: 5.2s  ❌
  api_p95_latency: 2.3s
  schema_failure_rate: 0.02%  ❌
  test_results:
    test_ga.py:
      passed: 9
      failed: 2  ❌
      total: 11
============================================================
```


## 커스터마이징

### KPI 임계값 변경

```python
from fragrance_ai.deployment.go_nogo_gate import GoNoGoGate, KPIThresholds

# 커스텀 임계값
custom_thresholds = KPIThresholds(
    llm_p95_fast=2.0,  # 더 엄격하게
    llm_p95_balanced=3.0,
    llm_p95_creative=4.0,
    api_error_rate=0.005,  # 0.5%
    schema_failure_rate=0.0  # 무관용 유지
)

# 게이트 초기화
gate = GoNoGoGate(kpi_thresholds=custom_thresholds)
result = gate.evaluate()
```

### 테스트 스위트 추가

```python
custom_test_suites = [
    "tests/test_llm_ensemble_operation.py",
    "tests/test_ga.py",
    "tests/test_ifra.py",
    "tests/test_end_to_end_evolution.py",
    "tests/test_quality_kpi.py",  # 추가
    "tests/test_security_privacy.py"  # 추가
]

gate = GoNoGoGate(test_suites=custom_test_suites)
```


## 문제 해결

### NO-GO 시 대응

1. **테스트 실패**
   ```bash
   # 실패한 테스트 재실행
   pytest tests/test_ga.py -v --tb=short

   # 수정 후 다시 체크
   python -m fragrance_ai.deployment.go_nogo_gate
   ```

2. **스키마 실패율 > 0%**
   ```python
   # JSON 스키마 검증 로그 확인
   grep "schema_validation" logs/app.log

   # JSON guard 수정 후 재배포
   # 스키마 실패는 무관용이므로 반드시 수정 필요
   ```

3. **p95 초과**
   ```bash
   # 메트릭 확인
   curl "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket[5m]))"

   # 캐시/최적화 후 재측정
   # 5분 이상 임계값 내로 유지되어야 GO
   ```


## 알림 설정

### Slack 알림

```python
# go_nogo_gate.py에 추가
def send_slack_notification(result: GoNoGoResult):
    """Slack 알림 전송"""
    import requests

    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    if result.decision == DeploymentDecision.GO:
        color = "good"
        title = "✅ GO: Safe to Deploy"
    else:
        color = "danger"
        title = "⛔ NO-GO: Deployment Blocked"

    message = {
        "attachments": [{
            "color": color,
            "title": title,
            "text": "\n".join(result.reasons) if result.reasons else "All checks passed",
            "footer": f"Timestamp: {result.timestamp}"
        }]
    }

    requests.post(webhook_url, json=message)
```


## 요약

| 조건 | 결과 | 설명 |
|------|------|------|
| 모든 테스트 통과 + KPI 달성 | ✅ **GO** | 배포 허용 |
| 테스트 1개라도 실패 | ⛔ **NO-GO** | 수정 후 재시도 |
| 스키마 실패율 > 0% | ⛔ **NO-GO** | **무관용** - 즉시 수정 |
| p95 초과 지속 | ⛔ **NO-GO** | 최적화 후 재시도 |
| KPI 1개라도 위반 | ⛔ **NO-GO** | 원인 파악 & 개선 |

**모든 체크를 자동으로 실행하고 명확한 GO/NO-GO 결정을 제공합니다!** 🚦
