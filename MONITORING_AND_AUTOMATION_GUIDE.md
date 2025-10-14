# Monitoring & Automation Guide
**Fragrance AI - 고급 모니터링 및 Runbook 자동화**

생성일: 2025-10-14

---

## 1. 추가된 메트릭

### LLM 메트릭
- `llm_brief_elapsed_ms(p95)`: LLM 브리프 생성 시간 (밀리초, p95 계산용)
  - Labels: `mode` (fast/balanced/creative), `model` (qwen/mistral/llama)
- `hybrid_switch_total`: Hybrid 모드 전환 횟수
  - Labels: `from_mode`, `to_mode`
- `hybrid_switch_ratio`: Hybrid 모드 비율
  - Labels: `mode` (exploration/exploitation)

### RL 메트릭
- `rl_reward_ma`: RL 보상 이동 평균
  - Labels: `algorithm` (ppo/reinforce), `window` (100/1000)
- `rl_clip_frac`: RL 클리핑 비율
  - Labels: `algorithm`
- `rl_distill_kl`: RL Distillation KL divergence
  - Labels: `teacher`, `student`

### 규제 메트릭
- `ifra_violations_total`: IFRA 규제 위반 건수
  - Labels: `ingredient`, `violation_type`
- `allergen_hits_total`: 알러젠 검출 건수
  - Labels: `allergen`, `severity`
- `ifra_compliance_rate`: IFRA 준수율
  - Labels: `check_type` (ingredient/concentration/combination)

### Cloud Hub 메트릭
- `ipfs_store_rate`: IPFS 저장 속도 (ops/sec)
  - Labels: `operation` (store/retrieve/pin)
- `ipfs_restore_latency_seconds`: IPFS 복원 레이턴시
  - Labels: `data_type` (feedback/checkpoint/recipe)
- `ipfs_cid_errors_total`: IPFS CID 에러 건수
  - Labels: `error_type`
- `redis_metadata_latency_seconds`: Redis 메타데이터 레이턴시
  - Labels: `operation` (hset/hget/expire)

### 시스템 메트릭
- `system_error_rate`: 시스템 에러율
  - Labels: `component`, `error_type`
- `system_rps`: 시스템 초당 요청 수 (RPS)
  - Labels: `service`
- `worker_vram_bytes`: 워커별 VRAM 사용량 (바이트)
  - Labels: `worker_id`, `model`
- `worker_cpu_percent`: 워커별 CPU 사용률 (%)
  - Labels: `worker_id`

---

## 2. Shadow Evaluation 시스템

### 개요
프로덕션 트래픽의 일부를 Teacher/Student 모델에 그림자 추론으로 흘려 비파괴적으로 성능 비교

### 구현
**파일**: `fragrance_ai/deployment/shadow_evaluation.py`

### 주요 기능

#### 2.1 샘플링
```python
evaluator = ShadowEvaluator(sample_rate=0.05)  # 5% 트래픽 샘플링
```

#### 2.2 병렬 추론
- **Production Model**: 실제 서비스 (항상 실행)
- **Teacher Model**: PPO 정책 (큰 모델, 선택적)
- **Student Model**: Distilled 모델 (작은 모델, 선택적)

```python
metrics = await evaluator.evaluate_request(
    request,
    teacher_model,
    student_model,
    production_model
)
```

#### 2.3 비교 메트릭
- **Latency**: Teacher vs Student vs Production
- **Reward**: 각 모델의 예측 품질
- **KL Divergence**: Teacher-Student 분포 차이
- **Cosine Similarity**: 예측 벡터 유사도
- **Speedup Ratio**: Student / Teacher 속도 비율

### 사용 예시
```bash
# Shadow Evaluation 실행
python fragrance_ai/deployment/shadow_evaluation.py

# 또는 Makefile 사용
make runbook-shadow-eval
```

### 결과 예시
```json
{
  "summary": {
    "total_comparisons": 100,
    "latency": {
      "teacher_avg_ms": 2500,
      "student_avg_ms": 600,
      "speedup_ratio": 4.17
    },
    "reward": {
      "teacher_avg": 22.5,
      "student_avg": 20.8,
      "retention_rate": 0.92
    },
    "kl_divergence": {
      "avg": 0.15,
      "p95": 0.28
    }
  }
}
```

---

## 3. Playbook 자동화 (Makefile)

### 개요
운영 작업을 `make runbook-*` 타겟으로 자동화

### 사용 가능한 Runbook

#### 3.1 Downshift (트래픽 감소)
```bash
make runbook-downshift
# 불안정한 서비스의 트래픽을 50%로 감소
```

**동작**:
1. Nginx 가중치 업데이트 (stable: 50%, canary: 50%)
2. 서비스 부하 감소
3. 모니터링 대기

#### 3.2 Rollback (롤백)
```bash
make runbook-rollback
# 이전 안정 버전으로 롤백
```

**동작**:
1. 이전 Docker 이미지 태그 확인
2. 서비스 롤백
3. 헬스 체크

#### 3.3 Pin Update (모델 버전 고정)
```bash
make runbook-pin-update
# Qwen/Mistral/Llama 모델 버전 고정 및 재시작
```

**동작**:
1. `configs/model_pins.json` 업데이트
   ```json
   {
     "qwen": "2.5.1",
     "mistral": "0.3.1",
     "llama": "3.1.1"
   }
   ```
2. Docker Compose 재시작

#### 3.4 Canary Promote (카나리 승격)
```bash
make runbook-canary-promote
# 카나리를 프로덕션으로 승격
```

**동작**:
1. 카나리 버전 검증
2. Nginx 가중치 100% 전환
3. 구 버전 종료

#### 3.5 Emergency Stop (긴급 중단)
```bash
make runbook-emergency-stop
# 모든 서비스 즉시 중단
```

**동작**:
1. Docker Compose down
2. 모든 컨테이너 종료

#### 3.6 기타 Runbook
```bash
make runbook-restart-workers    # 워커 재시작
make runbook-clear-cache       # Redis 캐시 초기화
make runbook-health-check      # 헬스 체크
make runbook-metrics-check     # 메트릭 확인
```

### 전체 Runbook 목록
```bash
make help
```

---

## 4. Grafana 대시보드 추가

### 권장 패널

#### 4.1 LLM 성능
- **llm_brief_elapsed_ms** (p95): 브리프 생성 시간
- **hybrid_switch_total** (rate): 모드 전환 빈도
- **hybrid_switch_ratio**: 탐색/활용 비율

#### 4.2 RL 훈련
- **rl_reward_ma**: 보상 이동 평균 (100/1000 윈도우)
- **rl_clip_frac**: 클리핑 비율 (0.05~0.25 정상 범위)
- **rl_distill_kl**: Teacher-Student KL divergence

#### 4.3 규제 준수
- **ifra_violations_total** (rate): 규제 위반 추이 (0 목표)
- **allergen_hits_total**: 알러젠 검출 건수
- **ifra_compliance_rate**: 준수율 (>95% 목표)

#### 4.4 Cloud Hub
- **ipfs_store_rate**: IPFS 저장 속도 (ops/sec)
- **ipfs_restore_latency_seconds**: 복원 레이턴시
- **ipfs_cid_errors_total**: CID 에러 건수

#### 4.5 시스템 건강도
- **system_error_rate**: 컴포넌트별 에러율
- **system_rps**: 서비스별 RPS
- **worker_vram_bytes**: 워커별 VRAM 사용량 (GB 단위로 변환)
- **worker_cpu_percent**: 워커별 CPU 사용률

---

## 5. 운영 가이드

### 5.1 정상 범위

| 메트릭 | 정상 범위 | 경고 임계값 | 위험 임계값 |
|--------|----------|-----------|-----------|
| llm_brief_elapsed_ms (p95) | <2000ms | >3000ms | >5000ms |
| rl_reward_ma (100) | 18~25 | <15 | <10 |
| rl_clip_frac | 0.05~0.25 | >0.4 | >0.6 |
| ifra_violations_total | 0 | >1/day | >5/day |
| ipfs_restore_latency | <500ms | >1000ms | >2000ms |
| system_error_rate | <1% | >3% | >5% |
| worker_vram | <18GB | >22GB | >25GB |

### 5.2 경고 대응

#### LLM 레이턴시 증가
```bash
# 1. 워커 재시작
make runbook-restart-workers

# 2. 캐시 확인
make runbook-metrics-check | grep cache_hit_rate

# 3. 트래픽 감소 (필요 시)
make runbook-downshift
```

#### IFRA 규제 위반 발생
```bash
# 1. 위반 로그 확인
docker-compose logs fragrance_ai | grep ifra_violation

# 2. 긴급 중단 (중대 위반 시)
make runbook-emergency-stop

# 3. 규제 검증 강화
# configs/feature_flags_dev.json에서 ifra_strict_mode 활성화
```

#### 워커 VRAM 부족
```bash
# 1. 워커 재시작 (메모리 누수 해결)
make runbook-restart-workers

# 2. 모델 버전 다운그레이드
make runbook-pin-update
# qwen: 32B → 7B로 변경

# 3. 워커 스케일 아웃
docker-compose up -d --scale celery_worker=4
```

### 5.3 Shadow Evaluation 주기

| 단계 | 샘플링 비율 | 실행 주기 |
|------|-----------|----------|
| 개발 | 100% | 코드 변경 시 |
| 스테이징 | 50% | 일 1회 |
| 프로덕션 | 5% | 주 1회 |
| 긴급 검증 | 10% | 장애 후 |

---

## 6. 알림 규칙 (Prometheus)

```yaml
groups:
  - name: fragrance_ai_advanced
    rules:
      - alert: HighLLMLatency
        expr: histogram_quantile(0.95, llm_brief_elapsed_ms) > 3000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "LLM latency p95 > 3s"

      - alert: IFRAViolation
        expr: rate(ifra_violations_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "IFRA regulation violation detected"

      - alert: HighWorkerVRAM
        expr: worker_vram_bytes > 22 * 1024^3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Worker VRAM usage > 22GB"

      - alert: LowRewardMA
        expr: rl_reward_ma{window="100"} < 15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "RL reward moving average < 15"
```

---

## 7. 테스트

### 7.1 메트릭 서버 확인
```bash
# 메트릭 서버 실행 중인지 확인
curl http://localhost:8000/metrics | grep -E "rl_reward_ma|hybrid_switch"

# 출력 예시:
# rl_reward_ma{algorithm="ppo",window="100"} 20.5
# hybrid_switch_total{from_mode="exploration",to_mode="exploitation"} 12
```

### 7.2 Shadow Evaluation 테스트
```bash
# 테스트 실행
python fragrance_ai/deployment/shadow_evaluation.py

# 결과 확인
cat shadow_evaluation_results.json | jq '.summary'
```

### 7.3 Runbook 테스트
```bash
# 헬스 체크
make runbook-health-check

# 메트릭 확인
make runbook-metrics-check

# Shadow Evaluation
make runbook-shadow-eval
```

---

## 8. 파일 위치

```
fragrance_ai/
├── deployment/
│   └── shadow_evaluation.py       # Shadow Evaluation 시스템
├── training/
│   ├── hybrid_loop.py            # Hybrid Loop (메트릭 생성)
│   └── qwen_rlhf.py              # RLHF (메트릭 생성)
scripts/
├── runbook_downshift.py          # Downshift 자동화
└── runbook_pin_update.py         # Pin Update 자동화
Makefile                          # Runbook 진입점
test_metrics_server.py            # 메트릭 생성 서버 (포트 8000)
```

---

## 9. 다음 단계

1. **Grafana 대시보드 커스터마이징**: 새로운 메트릭 패널 추가
2. **알림 규칙 구성**: Prometheus AlertManager 설정
3. **Shadow Evaluation 스케줄링**: Cron job으로 주기적 실행
4. **Runbook 확장**: 추가 운영 작업 자동화

---

*문서 버전: 1.0*
*최종 업데이트: 2025-10-14*
