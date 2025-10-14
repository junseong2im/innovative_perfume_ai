# Artisan SRE (Site Reliability Engineering) Guide

## 개요

Artisan의 SRE 실천은 **3가지 핵심 원칙**으로 구성됩니다:

1. **SLO & Error Budget** - 서비스 품질 목표 및 에러버짓 추적
2. **On-Call & Incident Response** - 장애 대응 및 15분 내 초기 조치
3. **Runbooks & Automation** - 자동화된 장애 대응 절차

---

## 1. SLO (Service Level Objectives)

### 1.1 Artisan SLO 정의

| SLO | 목표 | 측정 기간 | 에러버짓 |
|-----|------|-----------|----------|
| **API Availability** | 99.9% | 30일 | 0.1% (43분/월) |
| **LLM Fast p95** | p95 < 2.5s | 7일 | 5% 초과 허용 |
| **LLM Balanced p95** | p95 < 3.2s | 7일 | 5% 초과 허용 |
| **LLM Creative p95** | p95 ≤ 4.5s | 7일 | 5% 초과 허용 |
| **RL Reward Stability** | reward > 10.0 | 7일 | 1% 위반 허용 |

### 1.2 에러버짓 (Error Budget)

**에러버짓 = (1 - 목표)**

예시:
- API Availability 99.9% → 에러버짓 0.1% = 43분/월
- 43분 이내에서는 장애 허용
- 43분 초과 시 → **신규 기능 배포 중단**

### 1.3 에러버짓 사용 원칙

#### ✅ 에러버짓이 충분한 경우 (> 20%)
- 신규 기능 배포 허용
- 실험적 기능 테스트 가능
- 빠른 반복 개발

#### ⚠️ 에러버짓이 부족한 경우 (< 20%)
- 신규 기능 배포 주의
- 안정성 테스트 강화
- 모니터링 집중

#### 🚨 에러버짓 소진 시 (0%)
- **모든 신규 기능 배포 중단**
- 품질 개선 및 버그 수정만 허용
- Root cause 분석 및 해결 우선
- SLO 복구 후 배포 재개

### 1.4 SLO 추적

```bash
# SLO 리포트 조회
python -m fragrance_ai.sre.slo_tracker --report

# 배포 가능 여부 확인
python -m fragrance_ai.sre.slo_tracker --check
```

**출력 예시:**
```
============================================================
SLO & Error Budget Report
============================================================

✅ API Availability
   Target:              99.90%
   Actual:              99.92%
   Error Budget Used:   20.00%
   Error Budget Left:   80.00%
   Status:              healthy
   Total Requests:      1,234,567
   Failed Requests:     987

⚠️ LLM Creative Mode Latency (p95)
   Target:              95.00%
   Actual:              93.50%
   Error Budget Used:   82.00%
   Error Budget Left:   18.00%
   Status:              warning

============================================================
✅ Status: All SLOs within error budget - Deployments allowed
============================================================
```

---

## 2. 장애 등급 (Severity Levels)

### 2.1 Severity 정의

| 등급 | 설명 | 예시 | 응답 시간 |
|------|------|------|-----------|
| **Sev1** | Critical - 서비스 완전 중단 | API 다운, 데이터 유실 | **15분** |
| **Sev2** | High - 주요 기능 장애 | Qwen LLM 장애, PPO 학습 실패 | **30분** |
| **Sev3** | Medium - 부분적 기능 장애 | 캐시 장애, 일부 기능 느림 | **60분** |
| **Sev4** | Low - 경미한 문제 | 로그 누락, UI 버그 | **4시간** |

### 2.2 Sev1 사건 예시

- API 서버 다운 (5분 이상)
- 데이터베이스 연결 불가
- 모든 LLM 모델 장애
- 데이터 유실 또는 손상
- 보안 침해

### 2.3 Sev2 사건 예시

- Qwen 모델 장애 (다른 모델 정상)
- RL 학습 실패 또는 보상 폭주
- p95 지연시간 2배 초과
- 에러율 > 5%

### 2.4 Sev3 사건 예시

- 캐시 장애 (기능은 동작)
- 서킷브레이커 지속적 활성화
- 단일 엔드포인트 느림
- 로그 수집 실패

---

## 3. 온콜 (On-Call)

### 3.1 온콜 로테이션

- **주중 온콜**: 월~금 (8시간 근무)
- **주말 온콜**: 토~일 (24시간 대기)
- **로테이션 주기**: 1주일
- **핸드오프**: 금요일 17:00

### 3.2 온콜 준비사항

#### 필수 도구
- [ ] Slack 알림 활성화
- [ ] VPN 접근 확인
- [ ] kubectl 설정 확인
- [ ] AWS/GCP 접근 권한
- [ ] Grafana 대시보드 접근
- [ ] Prometheus 접근
- [ ] Incident 관리 도구

#### 필수 지식
- [ ] SLO 및 에러버짓
- [ ] 런북 위치 및 사용법
- [ ] Health check 엔드포인트
- [ ] 배포 롤백 절차
- [ ] 주요 알람 임계값

### 3.3 알람 대응

#### 알람 수신 시 (15분 내)
1. **Acknowledge** - 사건 인지 확인
2. **Assess** - 심각도 평가
3. **Communicate** - Slack에 초기 상황 공유
4. **Investigate** - 근본 원인 조사

#### 알람 예시
```
🚨 ALERT: High API Error Rate
   Current: 2.5% (threshold: 1%)
   Duration: 5 minutes
   Runbook: api_high_error_rate
   Dashboard: http://grafana/d/api
```

---

## 4. 사건 대응 (Incident Response)

### 4.1 사건 대응 흐름

```
1. Detection     → 자동 알람 또는 사용자 리포트
2. Acknowledgment → 15분 내 초기 대응 (Sev1)
3. Investigation → 근본 원인 조사
4. Mitigation    → 임시 해결 (서비스 복구 우선)
5. Resolution    → 근본 원인 해결
6. Closure       → 사건 종료 및 문서화
7. Postmortem    → 블레임리스 포스트모템 (Sev1, Sev2)
```

### 4.2 사건 생성

```bash
# CLI로 사건 생성
python -m fragrance_ai.sre.incident_manager --create '{
  "title": "Qwen LLM Failure",
  "description": "Qwen model not responding to requests",
  "severity": "Sev2",
  "components": ["LLM", "Qwen"]
}'

# 출력:
# 🚨 Incident Created: INC-20251014-103000 [Sev2]
#    Title: Qwen LLM Failure
#    Affected: LLM, Qwen
```

### 4.3 사건 진행

```bash
# 사건 인지
python -m fragrance_ai.sre.incident_manager --ack INC-20251014-103000 --responder alice

# 상태 업데이트
# (Python API 사용)
from fragrance_ai.sre.incident_manager import get_incident_manager, IncidentStatus

manager = get_incident_manager()
manager.update_status(
    "INC-20251014-103000",
    IncidentStatus.INVESTIGATING,
    "Checking Qwen health endpoint",
    "alice"
)
```

### 4.4 사건 종료

```bash
# 근본 원인 설정
manager.set_root_cause(
    "INC-20251014-103000",
    "Qwen model OOM due to memory leak in inference pipeline",
    "alice"
)

# 해결 방법 설정
manager.set_resolution(
    "INC-20251014-103000",
    "Restarted Qwen service with increased memory limit",
    "alice"
)

# 사건 종료
manager.close_incident("INC-20251014-103000", "alice")
```

### 4.5 사건 리포트

```bash
# 사건 리포트 조회
python -m fragrance_ai.sre.incident_manager --report
```

---

## 5. 런북 (Runbooks)

### 5.1 사용 가능한 런북

| 런북 | 설명 | 트리거 | 자동화 |
|------|------|--------|--------|
| **qwen_failure_downshift** | Qwen 장애 시 Balanced/Fast로 다운시프트 | Qwen 3회 연속 실패 | 자동 |
| **rl_reward_runaway_rollback** | RL 보상 폭주 시 체크포인트 롤백 | reward > 100 or std > 50 | 반자동 |
| **api_high_latency_scaleup** | API 지연시간 초과 시 스케일업 | p95 > threshold | 자동 |

### 5.2 런북 실행

```bash
# 런북 목록 조회
python -m fragrance_ai.sre.runbooks --list

# 런북 실행 (Dry run)
python -m fragrance_ai.sre.runbooks --execute qwen_failure_downshift --dry-run

# 런북 실행 (실제)
python -m fragrance_ai.sre.runbooks --execute qwen_failure_downshift
```

### 5.3 런북 예시: Qwen Failure Downshift

**트리거 조건:**
- Qwen health check 3회 연속 실패
- Qwen 호출 타임아웃 (> 30s)
- Qwen 에러율 > 50%

**자동 실행 단계:**
1. Qwen 장애 확인 (3회 health check)
2. 서킷브레이커 활성화
3. Creative → Balanced 다운시프트
4. Balanced 정상 작동 확인
5. 온콜 엔지니어 알림
6. Sev2 사건 생성

**수동 단계:**
- Qwen 재시작 또는 롤백

**예상 복구 시간:** 5분

---

## 6. Health Check

### 6.1 LLM 모델별 Health Check

#### Qwen 상태 확인
```bash
curl http://localhost:8000/health/llm?model=qwen
```

**응답:**
```json
{
  "model_name": "qwen",
  "status": "healthy",
  "loaded": true,
  "inference_ready": true,
  "last_inference_ms": 1847.32,
  "memory_mb": 15234.5,
  "gpu_memory_mb": 8192.0,
  "error_message": null,
  "checked_at": "2025-10-14T10:30:00.123Z",
  "uptime_seconds": 3600.0
}
```

#### 모든 모델 상태 확인
```bash
curl http://localhost:8000/health/llm/all
```

**응답:**
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
  "memory_mb": 45678.5,
  "gpu_available": true
}
```

### 6.2 알람 기준에 Health Check 포함

**Prometheus 알람 규칙:**
```yaml
- alert: QwenModelUnhealthy
  expr: |
    health_llm_qwen_status{status="unhealthy"} == 1
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "Qwen model is unhealthy"
    runbook: "qwen_failure_downshift"
```

---

## 7. 블레임리스 포스트모템 (Blameless Postmortem)

### 7.1 포스트모템 작성 시점

**필수:**
- Sev1 사건
- Sev2 사건

**선택:**
- Sev3 사건 (학습 가치가 큰 경우)
- Near-miss (장애 직전 회피한 경우)

### 7.2 포스트모템 원칙

1. **Blameless** - 개인 비난 금지
2. **Focus on Systems** - 시스템 개선에 집중
3. **Actionable** - 구체적 개선 조치 도출
4. **Transparent** - 전체 팀 공유

### 7.3 포스트모템 템플릿

[POSTMORTEM_TEMPLATE.md](./POSTMORTEM_TEMPLATE.md) 참조

### 7.4 포스트모템 완료

```python
from fragrance_ai.sre.incident_manager import get_incident_manager

manager = get_incident_manager()
manager.complete_postmortem(
    "INC-20251014-103000",
    "https://wiki.company.com/postmortems/2025-10-14-qwen-failure"
)
```

---

## 8. 배포 가드레일 (Deployment Guardrails)

### 8.1 배포 전 체크리스트

```bash
# 1. SLO 확인
python -m fragrance_ai.sre.slo_tracker --check
# → 에러버짓 소진 시 배포 차단

# 2. Go/No-Go 게이트
python -m fragrance_ai.deployment.go_nogo_gate --exit-code
# → 테스트 실패 시 배포 차단

# 3. Critical 테스트
pytest tests/test_llm_ensemble_operation.py -v
pytest tests/test_moga_stability.py -v
pytest tests/test_end_to_end_evolution.py -v
```

### 8.2 에러버짓 소진 시 대응

**자동 차단:**
```bash
$ python -m fragrance_ai.sre.slo_tracker --check
🚨 Deployments blocked due to SLO violations: llm_creative_latency
```

**대응 절차:**
1. 신규 기능 배포 중단
2. 근본 원인 조사
3. 성능 개선 또는 버그 수정
4. SLO 복구 확인
5. 배포 재개

---

## 9. 메트릭 및 대시보드

### 9.1 SRE 대시보드

**Grafana 대시보드:**
- SLO Dashboard: http://grafana/d/slo
- Error Budget Dashboard: http://grafana/d/error-budget
- Incident Dashboard: http://grafana/d/incidents

### 9.2 주요 메트릭

```promql
# API Availability
sum(rate(api_requests_total{status=~"2.."}[30d]))
/ sum(rate(api_requests_total[30d]))

# LLM Creative p95 Latency
histogram_quantile(0.95,
  rate(llm_brief_latency_seconds_bucket{mode="creative"}[7d])
)

# Error Budget Burn Rate
(1 - sum(rate(api_requests_total{status=~"2.."}[1h]))
    / sum(rate(api_requests_total[1h])))
/ (1 - 0.999)
```

---

## 10. 베스트 프랙티스

### 10.1 SLO 관리

- ✅ SLO는 사용자 중심으로 설정
- ✅ 에러버짓은 혁신과 안정성의 균형
- ✅ 주간 SLO 리뷰 미팅
- ✅ 분기별 SLO 재평가

### 10.2 사건 대응

- ✅ 15분 내 초기 대응 (Sev1)
- ✅ 타임라인 상세 기록
- ✅ 사용자 커뮤니케이션 (status page)
- ✅ 포스트모템 블레임리스

### 10.3 런북

- ✅ 자동화 가능한 단계는 자동화
- ✅ 주기적 런북 테스트 (월 1회)
- ✅ 런북 업데이트 (사건 후)
- ✅ 명확한 롤백 절차

### 10.4 온콜

- ✅ 온콜 핸드오프 체크리스트
- ✅ 적절한 알람 임계값
- ✅ 온콜 피로도 모니터링
- ✅ 포스트-온콜 회고

---

## 11. 도구 및 명령어

```bash
# SLO & Error Budget
python -m fragrance_ai.sre.slo_tracker --report
python -m fragrance_ai.sre.slo_tracker --check

# Incident Management
python -m fragrance_ai.sre.incident_manager --report
python -m fragrance_ai.sre.incident_manager --create '...'
python -m fragrance_ai.sre.incident_manager --ack INC-xxx --responder alice

# Runbooks
python -m fragrance_ai.sre.runbooks --list
python -m fragrance_ai.sre.runbooks --execute qwen_failure_downshift

# Health Checks
curl http://localhost:8000/health/llm?model=qwen
curl http://localhost:8000/health/llm/all
curl http://localhost:8000/health/ready
curl http://localhost:8000/health/live
```

---

## 12. 참고 자료

- [Google SRE Book](https://sre.google/books/)
- [POSTMORTEM_TEMPLATE.md](./POSTMORTEM_TEMPLATE.md)
- [RUNBOOKS.md](./RUNBOOKS.md)
- [OBSERVABILITY.md](./OBSERVABILITY.md)
- [OPERATIONS_GUIDE.md](./OPERATIONS_GUIDE.md)

---

**Artisan은 SRE 원칙을 통해 혁신과 안정성의 균형을 달성합니다!** 🎯
