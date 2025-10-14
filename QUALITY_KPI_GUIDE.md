# 품질 KPI 가이드 (Quality KPI Guide)

## 개요

이 문서는 Fragrance AI 시스템의 **최소 품질 기준(KPI)**을 정의하고, 이를 검증하고 모니터링하는 방법을 설명합니다.

프로덕션 환경에서 보장해야 하는 3가지 핵심 KPI:
1. **스키마 준수율 100%** (모든 모드)
2. **API p95 레이턴시**: fast ≤ 2.5s, balanced ≤ 3.2s, creative ≤ 4.5s
3. **RL 50 step 후 학습 효과** (통계적 유의성)

---

## KPI 1: 스키마 준수율 100%

### 정의

LLM 앙상블이 생성하는 모든 CreativeBrief가 정의된 스키마를 100% 준수해야 합니다.

### 스키마 정의

```python
class CreativeBriefSchema(BaseModel):
    style: str              # citrus/floral/woody/oriental/fresh/aquatic/gourmand
    intensity: float        # 0.0 ~ 1.0
    mood: str              # romantic/fresh/elegant/etc.
    season: List[str]      # [spring, summer, fall, winter, all]
    top_notes: List[str]   # 최소 1개
    middle_notes: List[str]  # 최소 1개
    base_notes: List[str]  # 최소 1개
    target_audience: str   # male/female/unisex
```

### 검증 방법

#### 1. 자동화 테스트

```bash
# 스키마 준수율 테스트 실행
pytest tests/test_quality_kpi.py::TestSchemaCompliance -v

# 기대 결과:
# test_schema_compliance_fast_mode PASSED
# test_schema_compliance_balanced_mode PASSED
# test_schema_compliance_creative_mode PASSED (가장 중요!)
# test_schema_compliance_all_modes_combined PASSED
```

**테스트 커버리지**:
- Fast 모드: 30개 입력
- Balanced 모드: 30개 입력
- Creative 모드: 50개 입력 (긴 텍스트, 복잡한 표현)
- 전체 통합: 110개 입력

#### 2. 실시간 모니터링

**Prometheus 메트릭**:
```promql
# 스키마 준수율 (모드별)
schema_compliance_rate{mode="fast"}      # Target: 1.0 (100%)
schema_compliance_rate{mode="balanced"}  # Target: 1.0
schema_compliance_rate{mode="creative"}  # Target: 1.0

# 검증 실패 카운트
sum by (mode) (schema_validation_total{status="failure"})  # Target: 0
```

**Grafana 대시보드**:
- Panel 1: "Schema Compliance Rate" (게이지)
  - 녹색: ≥99%
  - 노란색: 95-99%
  - 빨간색: <95%

### 실패 시 대응

#### 문제 진단

```python
# 실패한 브리프 로그 확인
grep "schema_validation_failure" /var/log/fragrance-ai/llm.log | tail -20

# 실패 원인 분석
{
  "timestamp": "2025-10-12T10:30:00Z",
  "component": "LLM",
  "event": "schema_validation_failure",
  "mode": "creative",
  "error": "ValidationError: intensity must be between 0.0 and 1.0",
  "brief": {"style": "floral", "intensity": 1.5, ...}
}
```

#### 해결 방법

1. **LLM 프롬프트 개선**:
   ```python
   # docs/PROMPT_DESIGN_GUIDE.md 참고
   # 스키마 제약 조건을 시스템 프롬프트에 명시

   SYSTEM_PROMPT = """
   ...
   IMPORTANT: Output JSON must follow this schema strictly:
   - intensity: 0.0 ~ 1.0 (NOT 0 ~ 10 or 0 ~ 100!)
   - style: one of [citrus, floral, woody, oriental, fresh, aquatic, gourmand]
   ...
   """
   ```

2. **후처리 검증 레이어 추가**:
   ```python
   def generate_brief_with_validation(user_text: str, mode: str) -> Dict[str, Any]:
       brief = llm_ensemble.generate(user_text, mode=mode)

       try:
           validated = CreativeBriefSchema(**brief)
           kpi_metrics_collector.record_schema_validation(mode, success=True)
           return validated.dict()
       except ValidationError as e:
           logger.error(f"Schema validation failed: {e}")
           kpi_metrics_collector.record_schema_validation(mode, success=False)

           # 재시도 로직
           brief = retry_with_fixed_prompt(user_text, mode, error=e)
           return brief
   ```

3. **모델 파인튜닝**:
   - 스키마 위반 사례를 학습 데이터에 추가
   - Few-shot 예시 증가

### 성공 기준

- ✅ **Fast 모드**: 100% (30/30)
- ✅ **Balanced 모드**: 100% (30/30)
- ✅ **Creative 모드**: 100% (50/50) ← 가장 중요!
- ✅ **전체**: 100% (110/110)

---

## KPI 2: API p95 레이턴시

### 정의

각 모드별로 p95 레이턴시가 목표치 이하여야 합니다.

| 모드 | p95 목표 | 설명 |
|------|----------|------|
| **Fast** | ≤ 2.5s | 짧은 입력, 빠른 응답 |
| **Balanced** | ≤ 3.2s | 중간 길이, 균형잡힌 품질 |
| **Creative** | ≤ 4.5s | 긴 입력, 높은 품질 |

### 검증 방법

#### 1. 자동화 벤치마크 테스트

```bash
# 레이턴시 벤치마크 테스트 실행
pytest tests/test_quality_kpi.py::TestAPILatency -v

# 기대 출력:
# === Fast Mode Latency ===
#   Mean: 1800ms
#   p50:  1750ms
#   p95:  2350ms (Target: ≤ 2500ms) ✓
#   p99:  2750ms
#
# === Balanced Mode Latency ===
#   Mean: 2300ms
#   p50:  2250ms
#   p95:  3050ms (Target: ≤ 3200ms) ✓
#   p99:  3600ms
#
# === Creative Mode Latency ===
#   Mean: 3200ms
#   p50:  3150ms
#   p95:  4200ms (Target: ≤ 4500ms) ✓
#   p99:  5000ms
```

**테스트 방법**:
- 각 모드별로 100번 호출
- 실제 분포 근사 (정규 분포 + 노이즈)
- p50, p95, p99 계산

#### 2. 실시간 모니터링

**Prometheus 쿼리**:
```promql
# Fast 모드 p95 레이턴시
histogram_quantile(0.95,
  sum(rate(api_request_latency_seconds_bucket{mode="fast"}[5m])) by (le)
)

# Balanced 모드 p95 레이턴시
histogram_quantile(0.95,
  sum(rate(api_request_latency_seconds_bucket{mode="balanced"}[5m])) by (le)
)

# Creative 모드 p95 레이턴시
histogram_quantile(0.95,
  sum(rate(api_request_latency_seconds_bucket{mode="creative"}[5m])) by (le)
)
```

**Grafana 대시보드**:
- Panel 2: "API p95 Latency by Mode" (시계열 그래프)
  - 목표선 표시 (fast: 2.5s, balanced: 3.2s, creative: 4.5s)
  - Alert: p95 > 목표치 + 10%

#### 3. 실제 API 부하 테스트

```bash
# Apache Bench로 부하 테스트
ab -n 1000 -c 10 -p request.json -T application/json \
   http://localhost:8000/dna/create

# 또는 k6 시나리오 테스트
k6 run tests/load/api_latency_test.js
```

### 실패 시 대응

#### 문제 진단

```bash
# 레이턴시 분포 확인
curl -s http://localhost:8000/metrics | grep api_request_latency_seconds_bucket

# 느린 요청 추적
grep "latency_ms" /var/log/fragrance-ai/api.log | \
  awk '$NF > 3000' | tail -20
```

#### 해결 방법

1. **캐시 최적화**:
   ```python
   # TTL 증가
   cache:
     ttl_seconds: 600  # 5분 → 10분

   # 의미 유사성 임계값 완화
   semantic_similarity_threshold: 0.85  # 0.9 → 0.85
   ```

2. **모델 양자화**:
   ```yaml
   # configs/llm_ensemble.yaml
   models:
     qwen:
       quantization: "int8"  # FP16 → INT8 (2배 빠름)
     llama:
       quantization: "int4"  # FP16 → INT4 (4배 빠름, Creative 모드)
   ```

3. **배치 처리**:
   ```python
   # 여러 요청을 배치로 처리
   briefs = llm_ensemble.generate_batch([req1, req2, ...], mode="fast")
   ```

4. **모드 다운시프트**:
   ```python
   # 부하가 높을 때 자동 다운시프트
   if current_load > 0.8 and mode == "creative":
       mode = "balanced"  # Creative → Balanced
   ```

### 성공 기준

- ✅ **Fast 모드 p95**: ≤ 2.5s (2500ms)
- ✅ **Balanced 모드 p95**: ≤ 3.2s (3200ms)
- ✅ **Creative 모드 p95**: ≤ 4.5s (4500ms)

---

## KPI 3: RL 학습 효과 (50 step 후)

### 정의

RL 에이전트가 50 step 학습 후 다음 중 하나 이상이 **통계적으로 유의하게** 개선되어야 합니다:
1. 선호 액션 선택 확률 증가
2. 평균 보상 증가

### 통계적 유의성

- **귀무 가설 (H0)**: 학습 후 개선이 없음 (개선도 ≤ 0)
- **대립 가설 (H1)**: 학습 후 개선이 있음 (개선도 > 0)
- **유의 수준**: p < 0.05 (95% 신뢰도)
- **실험 횟수**: n=30 trials

### 검증 방법

#### 1. 자동화 테스트

```bash
# RL 학습 효과 테스트 실행
pytest tests/test_quality_kpi.py::TestRLLearningEffectiveness -v

# 기대 출력:
# === RL Learning Effectiveness Test ===
# True preferences: [0.15, 0.08, 0.42, 0.25, 0.10]
# Best action: 2 (preference: 0.420)
# Initial policy: [0.20, 0.20, 0.20, 0.20, 0.20]
# Initial prob of best action: 0.200
#
# After 50 steps:
# Final policy: [0.10, 0.05, 0.60, 0.20, 0.05]
# Final prob of best action: 0.600
# Improvement: 0.400 (+200.0%)
#
# === Statistical Significance (n=30 trials) ===
# Mean improvement: 0.385 ± 0.082
# t-statistic: 25.743
# p-value: 0.0001
#
# ✓ RL learning is statistically significant (p < 0.05)
# ✓ Preferred action probability increased by 0.385 on average
```

#### 2. 실시간 모니터링

**Prometheus 메트릭**:
```promql
# 평균 보상 추이 (증가 추세)
rate(rl_episode_reward_sum{algorithm="PPO"}[5m]) /
rate(rl_episode_reward_count{algorithm="PPO"}[5m])

# 정책 엔트로피 (감소 추세 = 수렴)
rl_policy_entropy{algorithm="PPO"}

# 선호 액션 확률 (증가 추세)
rl_preferred_action_probability{algorithm="PPO"}

# 학습 스텝 수
rl_learning_steps_total{algorithm="PPO"}
```

**Grafana 대시보드**:
- Panel 4: "RL Learning - Episode Reward" (시계열)
  - 기대: 우상향 그래프
- Panel 5: "RL Policy Entropy" (시계열)
  - 기대: 우하향 그래프 (탐색 → 수렴)
- Panel 6: "RL Preferred Action Probability" (게이지)
  - Target: > 50% after 50 steps

#### 3. A/B 테스트

```python
# 학습 전 vs 학습 후 비교
def test_rl_ab_comparison():
    # Control: 학습하지 않은 정책
    control_policy = RandomPolicy()
    control_rewards = []
    for _ in range(100):
        reward = env.evaluate(control_policy)
        control_rewards.append(reward)

    # Treatment: 50 step 학습한 정책
    treatment_policy = PPOPolicy()
    treatment_policy.train(n_steps=50)
    treatment_rewards = []
    for _ in range(100):
        reward = env.evaluate(treatment_policy)
        treatment_rewards.append(reward)

    # t-test
    t_stat, p_value = stats.ttest_ind(treatment_rewards, control_rewards, alternative='greater')

    print(f"Control mean: {np.mean(control_rewards):.2f}")
    print(f"Treatment mean: {np.mean(treatment_rewards):.2f}")
    print(f"p-value: {p_value:.4f}")

    assert p_value < 0.05  # 통계적으로 유의
    assert np.mean(treatment_rewards) > np.mean(control_rewards)  # 실제 개선
```

### 실패 시 대응

#### 문제 진단

```python
# 학습 곡선 확인
import matplotlib.pyplot as plt

plt.plot(agent.reward_history)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('RL Learning Curve')
plt.show()

# 문제 패턴:
# 1. 평평한 곡선: 학습이 안됨 → learning rate 조정
# 2. 진동하는 곡선: 불안정 → reward normalization 적용
# 3. 감소하는 곡선: 정책 붕괴 → checkpoint & rollback 사용
```

#### 해결 방법

1. **학습률 조정**:
   ```python
   # 너무 큼 → 불안정
   # 너무 작음 → 학습 느림

   ppo_trainer = AdvancedPPOTrainer(
       learning_rate=3e-4,  # 기본값
       # learning_rate=1e-3,  # 빠른 학습 (위험)
       # learning_rate=1e-4,  # 안정적 학습 (느림)
   )
   ```

2. **보상 정규화 활성화**:
   ```python
   # fragrance_ai/training/rl_advanced.py
   reward_normalizer = RewardNormalizer(
       RewardNormalizerConfig(
           update_mean_std=True,
           epsilon=1e-8
       )
   )
   ```

3. **엔트로피 어닐링 조정**:
   ```python
   # 탐색 기간 연장
   entropy_scheduler = EntropyScheduler(
       EntropySchedulerConfig(
           initial_entropy=0.1,
           final_entropy=0.01,
           decay_steps=200  # 100 → 200 (더 천천히 감소)
       )
   )
   ```

4. **체크포인트 롤백 임계값 조정**:
   ```python
   checkpoint_manager = CheckpointManager(
       CheckpointConfig(
           rollback_on_kl_threshold=0.05,  # 0.03 → 0.05 (더 관대)
           rollback_on_loss_increase=3.0,  # 2.0 → 3.0
       )
   )
   ```

### 성공 기준

- ✅ **선호 액션 확률 증가**: 평균 +0.35 이상 (초기 0.20 → 최종 0.55+)
- ✅ **통계적 유의성**: p < 0.05 (t-test, n=30)
- ✅ **평균 보상 증가**: 최종 > 초기
- ✅ **정책 수렴**: 마지막 10 step 변화량 < 0.05

---

## 통합 모니터링

### Prometheus 설정

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fragrance-ai'
    static_configs:
      - targets: ['localhost:8000']  # API 서버
        labels:
          service: 'api'

      - targets: ['localhost:8001']  # LLM Worker
        labels:
          service: 'llm-worker'

      - targets: ['localhost:8002']  # RL Worker
        labels:
          service: 'rl-worker'

rule_files:
  - 'kpi_alerts.yml'
```

**kpi_alerts.yml**:
```yaml
groups:
  - name: quality_kpi_alerts
    interval: 1m
    rules:
      # KPI 1: 스키마 준수율
      - alert: SchemaComplianceRateLow
        expr: schema_compliance_rate < 0.99
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Schema compliance rate below 99%"
          description: "Mode {{ $labels.mode }}: {{ $value | humanizePercentage }}"

      # KPI 2: API 레이턴시
      - alert: FastModeLatencyHigh
        expr: |
          histogram_quantile(0.95,
            sum(rate(api_request_latency_seconds_bucket{mode="fast"}[5m])) by (le)
          ) > 2.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Fast mode p95 latency exceeds 2.5s"
          description: "Current p95: {{ $value }}s"

      - alert: BalancedModeLatencyHigh
        expr: |
          histogram_quantile(0.95,
            sum(rate(api_request_latency_seconds_bucket{mode="balanced"}[5m])) by (le)
          ) > 3.2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Balanced mode p95 latency exceeds 3.2s"

      - alert: CreativeModeLatencyHigh
        expr: |
          histogram_quantile(0.95,
            sum(rate(api_request_latency_seconds_bucket{mode="creative"}[5m])) by (le)
          ) > 4.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Creative mode p95 latency exceeds 4.5s"

      # KPI 3: RL 학습 효과
      - alert: RLRewardDecreasing
        expr: |
          deriv(
            rate(rl_episode_reward_sum[10m])[5m:]
          ) < 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "RL average reward is decreasing"
          description: "Algorithm {{ $labels.algorithm }}"
```

### Grafana 대시보드 import

```bash
# Grafana 대시보드 import
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana_kpi_dashboard.json

# 또는 UI에서:
# Grafana → Dashboards → Import → Upload JSON file
# → monitoring/grafana_kpi_dashboard.json 선택
```

### API에 메트릭 엔드포인트 추가

**app/main.py**:
```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app
from fragrance_ai.monitoring.kpi_metrics import kpi_metrics_collector

app = FastAPI(title="Fragrance AI API")

# Prometheus 메트릭 엔드포인트
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.post("/dna/create")
async def create_dna(request: DNACreateRequest):
    mode = request.mode or "balanced"

    # 레이턴시 측정
    with kpi_metrics_collector.time_api_request(mode, "/dna/create"):
        # LLM 브리프 생성
        brief = llm_ensemble.generate_brief(request.user_text, mode=mode)

        # 스키마 검증
        try:
            validated = CreativeBriefSchema(**brief)
            kpi_metrics_collector.record_schema_validation(mode, success=True)
        except ValidationError as e:
            kpi_metrics_collector.record_schema_validation(mode, success=False)
            raise HTTPException(status_code=500, detail=str(e))

    return validated.dict()
```

---

## 테스트 실행 가이드

### 전체 KPI 테스트

```bash
# 전체 품질 KPI 테스트 실행
pytest tests/test_quality_kpi.py -v -s

# 기대 출력:
# tests/test_quality_kpi.py::TestSchemaCompliance::test_schema_compliance_fast_mode PASSED
# tests/test_quality_kpi.py::TestSchemaCompliance::test_schema_compliance_balanced_mode PASSED
# tests/test_quality_kpi.py::TestSchemaCompliance::test_schema_compliance_creative_mode PASSED
# tests/test_quality_kpi.py::TestAPILatency::test_api_latency_fast_mode PASSED
# tests/test_quality_kpi.py::TestAPILatency::test_api_latency_balanced_mode PASSED
# tests/test_quality_kpi.py::TestAPILatency::test_api_latency_creative_mode PASSED
# tests/test_quality_kpi.py::TestRLLearningEffectiveness::test_rl_preferred_action_probability_increase PASSED
# tests/test_quality_kpi.py::TestRLLearningEffectiveness::test_rl_average_reward_increase PASSED
# tests/test_quality_kpi.py::TestRLLearningEffectiveness::test_rl_convergence_50_steps PASSED
# tests/test_quality_kpi.py::TestQualityKPISummary::test_all_kpis_summary PASSED
#
# ============================================================
# QUALITY KPI SUMMARY
# ============================================================
#
# [KPI 1] Schema Compliance Rate: 100%
#   ✓ Fast mode: 100%
#   ✓ Balanced mode: 100%
#   ✓ Creative mode: 100%
#
# [KPI 2] API p95 Latency Targets:
#   ✓ Fast mode: ≤ 2.5s
#   ✓ Balanced mode: ≤ 3.2s
#   ✓ Creative mode: ≤ 4.5s
#
# [KPI 3] RL Learning Effectiveness (50 steps):
#   ✓ Preferred action probability increased (p < 0.05)
#   ✓ Average reward increased (p < 0.05)
#
# ============================================================
# ALL KPIs PASSED ✓
# ============================================================
#
# ====================== 10 passed in 45.23s ======================
```

### 특정 KPI만 테스트

```bash
# KPI 1: 스키마 준수율만
pytest tests/test_quality_kpi.py::TestSchemaCompliance -v

# KPI 2: API 레이턴시만
pytest tests/test_quality_kpi.py::TestAPILatency -v

# KPI 3: RL 학습 효과만
pytest tests/test_quality_kpi.py::TestRLLearningEffectiveness -v
```

### CI/CD 통합

**.github/workflows/kpi_check.yml**:
```yaml
name: Quality KPI Check

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 */6 * * *'  # 6시간마다 실행

jobs:
  kpi-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt

      - name: Run KPI tests
        run: |
          pytest tests/test_quality_kpi.py -v --junit-xml=kpi-results.xml

      - name: Publish test results
        uses: EnricoMi/publish-unit-test-result-action@v2
        if: always()
        with:
          files: kpi-results.xml

      - name: Fail if KPIs not met
        if: failure()
        run: |
          echo "::error::Quality KPIs not met! Check test results."
          exit 1
```

---

## 운영 체크리스트

### 일일 체크 (Daily)

- [ ] Grafana 대시보드에서 KPI 상태 확인
  - Schema compliance rate: 모든 모드 ≥99%
  - API p95 latency: 목표치 이하
  - RL learning: 정상 학습 추이

- [ ] 알림 확인
  - Slack/Email 알림 확인
  - 임계값 초과 항목 확인

### 주간 체크 (Weekly)

- [ ] KPI 테스트 실행
  ```bash
  pytest tests/test_quality_kpi.py -v
  ```

- [ ] 트렌드 분석
  - 지난 7일간 레이턴시 추이
  - 스키마 준수율 변화
  - RL 학습 성능 비교

### 월간 체크 (Monthly)

- [ ] KPI 임계값 재검토
  - 실제 사용 패턴 반영
  - 목표치 조정 필요성 검토

- [ ] 부하 테스트
  ```bash
  k6 run tests/load/kpi_load_test.js
  ```

- [ ] A/B 테스트 분석
  - 모드별 성능 비교
  - 사용자 만족도 상관 분석

---

## 참고 문서

1. **테스트 코드**: `tests/test_quality_kpi.py`
2. **메트릭 수집**: `fragrance_ai/monitoring/kpi_metrics.py`
3. **Grafana 대시보드**: `monitoring/grafana_kpi_dashboard.json`
4. **Prometheus 설정**: `monitoring/prometheus.yml`, `monitoring/kpi_alerts.yml`
5. **API 통합 가이드**: `docs/DEPLOYMENT.md`
6. **튜닝 가이드**: `docs/TUNING_GUIDE.md`

---

**작성일**: 2025-10-12
**버전**: 1.0
**작성자**: Claude Code (Fragrance AI Team)
