# Canary Deployment System (카나리 배포)
# 완전 자동화된 점진적 배포 시스템

Progressive rollout system with 1% → 5% → 25% → 100% traffic stages, automated validation, and rollback.

---

## ✅ Implementation Complete

완전한 카나리 배포 시스템 구현 완료:

1. ✅ **Progressive Rollout**: 1% → 5% → 25% → 100% 트래픽 단계
2. ✅ **Automated Validation**: 지연시간, 에러율, 스키마, RL 손실 자동 검증
3. ✅ **Traffic Control**: NGINX 가중치 기반 트래픽 라우팅
4. ✅ **LLM Ensemble Testing**: 3-모델 앙상블 실제 부하 테스트
5. ✅ **Automatic Rollback**: 검증 실패 시 자동 롤백
6. ✅ **Metrics Monitoring**: Prometheus 기반 실시간 모니터링
7. ✅ **Smoke Tests**: 런치 직후 5분 수동 검증 자동화 (API + 로그)

---

## 📁 Created Files

### 1. Canary Deployment Orchestrator

**파일**: `scripts/canary_deployment.py` (850+ lines)

완전 자동화된 카나리 배포 오케스트레이션 스크립트.

```bash
# 기본 사용 (1% → 5% → 25% → 100%)
python scripts/canary_deployment.py --version v0.2.0

# 관찰 기간 조정 (30분)
python scripts/canary_deployment.py --version v0.2.0 --observation-period 1800

# 커스텀 단계
python scripts/canary_deployment.py --version v0.2.0 --stages 1,10,50,100

# 드라이 런 (트래픽 변경 없이 테스트)
python scripts/canary_deployment.py --version v0.2.0 --dry-run
```

**주요 클래스**:

1. **TrafficController**: NGINX 업스트림 가중치 제어
2. **MetricsCollector**: Prometheus 메트릭 수집 및 쿼리
3. **CanaryValidator**: 메트릭 검증 및 통과/실패 판정
4. **CanaryDeployment**: 전체 배포 프로세스 조율

**검증 항목** (15분 관찰):

| 메트릭 | 임계값 | 조치 |
|--------|--------|------|
| **p95 지연시간** | fast ≤ 2.5s / balanced ≤ 3.2s / creative ≤ 4.5s | 초과 시 롤백 |
| **에러율** | < 0.5% (5xx 에러) | 초과 시 롤백 |
| **llm_brief 스키마 실패** | 0% (엄격) | 1건이라도 발생 시 롤백 |
| **rl_update 손실 급등** | < 2x 베이스라인 | 초과 시 롤백 |

**출력 예시**:

```
================================================================================
CANARY DEPLOYMENT: v0.2.0
================================================================================

[STAGE 1/4] 1% Canary Traffic
  ✓ Traffic weight updated: 1% canary, 99% production
  ⏳ Observing for 15 minutes...

  Metrics (5 min):
    ├─ p95 latency (fast): 1.8s ✓
    ├─ p95 latency (balanced): 2.9s ✓
    ├─ p95 latency (creative): 3.7s ✓
    ├─ Error rate: 0.12% ✓
    ├─ Schema failures: 0 ✓
    └─ RL loss: 0.45 (baseline: 0.42) ✓

  ✓ Stage 1 validation PASSED
  ➜ Proceeding to Stage 2...

[STAGE 2/4] 5% Canary Traffic
  ✓ Traffic weight updated: 5% canary, 95% production
  ⏳ Observing for 15 minutes...
  ...

[STAGE 4/4] 100% Canary Traffic
  ✓ Traffic weight updated: 100% canary, 0% production
  ⏳ Observing for 15 minutes...

  ✓ Stage 4 validation PASSED

================================================================================
✓ CANARY DEPLOYMENT SUCCESSFUL
================================================================================
Version v0.2.0 is now serving 100% of traffic.

Next steps:
  1. Monitor for 30-60 minutes
  2. Get team approval
  3. Promote to production: docker-compose up -d app
  4. Remove canary infrastructure: docker-compose -f docker-compose.canary.yml down
```

---

### 2. NGINX Traffic Routing Configuration

**파일**:
- `nginx/nginx.canary.conf` - 카나리 지원 NGINX 메인 설정
- `nginx/conf.d/upstream.conf.template` - 가중치 기반 업스트림 템플릿

가중치 기반 트래픽 라우팅으로 production과 canary 간 트래픽 분배.

**Upstream 구조**:

```nginx
upstream app_backend {
    # Production version (stable)
    server app:8000 weight=${PRODUCTION_WEIGHT};  # 99, 95, 75, 0

    # Canary version (new release)
    server app-canary:8001 weight=${CANARY_WEIGHT};  # 1, 5, 25, 100

    keepalive 32;
}
```

**트래픽 분배 예시**:

| 단계 | PRODUCTION_WEIGHT | CANARY_WEIGHT | 설명 |
|------|-------------------|---------------|------|
| Stage 0 | 100 | 0 | 프로덕션만 (카나리 배포 전) |
| Stage 1 | 99 | 1 | 1% 카나리 테스트 |
| Stage 2 | 95 | 5 | 5% 카나리 |
| Stage 3 | 75 | 25 | 25% 카나리 |
| Stage 4 | 0 | 100 | 100% 카나리 (프로모션 준비) |

**특징**:
- ✅ Weighted load balancing
- ✅ Health check for both versions
- ✅ Automatic failover (canary → production on error)
- ✅ Separate logging for canary traffic
- ✅ Version tracking via headers

---

### 3. NGINX Weight Update Script

**파일**: `scripts/update_nginx_weights.sh`

NGINX 업스트림 가중치 동적 업데이트 스크립트.

```bash
# 5% 카나리 트래픽 설정
./scripts/update_nginx_weights.sh 5

# 25% 카나리 트래픽 설정
./scripts/update_nginx_weights.sh 25

# 100% 카나리 (프로모션)
./scripts/update_nginx_weights.sh 100

# 롤백 (0% 카나리)
./scripts/update_nginx_weights.sh 0
```

**동작 과정**:

1. ✅ 입력 검증 (0-100 범위)
2. ✅ PRODUCTION_WEIGHT, CANARY_WEIGHT 계산
3. ✅ upstream.conf.template → upstream.conf 생성 (envsubst)
4. ✅ NGINX 컨테이너에 설정 복사
5. ✅ NGINX 설정 검증 (`nginx -t`)
6. ✅ NGINX 리로드 (`nginx -s reload`)
7. ✅ 검증 및 로깅

**안전장치**:
- NGINX 설정 검증 실패 시 중단
- 리로드 실패 시 이전 설정 유지
- 모든 단계 로깅 및 확인

---

### 4. Canary Docker Compose Configuration

**파일**: `docker-compose.canary.yml`

카나리 배포를 위한 별도 서비스 정의.

```bash
# 카나리 인프라 시작
docker-compose -f docker-compose.production.yml -f docker-compose.canary.yml up -d app-canary

# 카나리 로그 확인
docker-compose -f docker-compose.canary.yml logs -f app-canary

# 카나리 중지
docker-compose -f docker-compose.canary.yml stop app-canary

# 카나리 제거
docker-compose -f docker-compose.canary.yml down
```

**카나리 서비스**:

1. **app-canary**: 카나리 애플리케이션 (port 8001)
   - LLM 앙상블 활성화 (`USE_LLM_ENSEMBLE=true`)
   - 3-모델 앙상블: GPT-3.5-turbo, Claude-3-Haiku, Gemini-Pro
   - Weighted voting
   - Prometheus 메트릭 (canary 라벨)

2. **worker-llm-canary**: 카나리 LLM 워커 (optional)
   - LLM 앙상블 테스트용
   - Profile: `canary-workers`

3. **worker-rl-canary**: 카나리 RL 워커 (optional)
   - RL 학습 로직 변경 테스트용
   - Profile: `canary-workers`

4. **nginx**: 카나리 설정으로 오버라이드
   - `nginx.canary.conf` 사용
   - 가중치 기반 라우팅

**환경 변수**:

```bash
# .env 파일
CANARY_VERSION=v0.2.0                          # 카나리 버전
CANARY_API_PORT=8001                           # 카나리 포트
USE_LLM_ENSEMBLE=true                          # LLM 앙상블 활성화
LLM_ENSEMBLE_MODELS=gpt-3.5-turbo,claude-3-haiku,gemini-pro
LLM_WORKER_CANARY_REPLICAS=1                   # 카나리 LLM 워커 수
RL_WORKER_CANARY_REPLICAS=0                    # 카나리 RL 워커 수 (보통 0)
```

---

### 5. Smoke Test Scripts

**파일**:
- `scripts/smoke_test_api.py` - Python 자동화 스모크 테스트
- `scripts/smoke_test_manual.sh` - Bash 수동 스모크 테스트
- `SMOKE_TEST_GUIDE.md` - 스모크 테스트 가이드

런치 직후 5분 수동 검증 자동화.

```bash
# Python 자동화 테스트 (프로덕션)
python scripts/smoke_test_api.py

# Python 자동화 테스트 (카나리)
python scripts/smoke_test_api.py --canary

# Bash 수동 테스트
./scripts/smoke_test_manual.sh http://localhost:8001 fragrance-ai-app-canary
```

**검증 항목**:

1. **API Smoke Tests**:
   - ✅ `/dna/create` - DNA 생성 및 ID 반환
   - ✅ `/evolve/options` - PPO 알고리즘으로 3개 옵션 생성 (creative 모드)
   - ✅ `/evolve/feedback` - 피드백 제출 및 학습 트리거

2. **로그 검증**:
   - ✅ `llm_brief{mode,...,elapsed_ms}` - LLM 브리프 생성 로그
   - ✅ `rl_update{algo,loss,reward,entropy,clip_frac}` - RL 업데이트 로그

3. **메트릭 검증**:
   - ✅ `/metrics` 엔드포인트에서 llm_brief 메트릭 확인
   - ✅ `/metrics` 엔드포인트에서 rl_update 메트릭 확인

**출력 예시**:

```
================================================================================
SMOKE TEST - 런치 직후 5분 검증
================================================================================
Base URL: http://localhost:8001
Container: fragrance-ai-app-canary
================================================================================

[STEP] Health Check
[✓ PASS] Health check passed

[STEP] API Test - /dna/create
[✓ PASS] DNA created successfully: dna_abc123

[STEP] API Test - /evolve/options
[✓ PASS] Evolution options generated: experiment_id=exp_xyz789
[✓ PASS] First option ID: opt_001

[STEP] API Test - /evolve/feedback
[✓ PASS] Feedback submitted successfully

[STEP] Log Verification - llm_brief metrics
[✓ PASS] Found llm_brief logs with fields: mode, elapsed_ms, duration
[✓ PASS] ✓ All required fields present (mode, timing)

[STEP] Log Verification - rl_update metrics
[✓ PASS] Found rl_update logs with fields: algo, loss, reward, entropy
[✓ PASS] ✓ algo field present
[✓ PASS] ✓ loss field present
[✓ PASS] ✓ reward field present
[✓ PASS] ✓ entropy field present

[STEP] Metrics Endpoint Check
[✓ PASS] ✓ llm_brief metrics present in /metrics
[✓ PASS] ✓ rl_update metrics present in /metrics

================================================================================
SMOKE TEST SUMMARY
================================================================================
Duration: 45.2s

Results:
  ✓ Passed: 12
  ✗ Failed: 0

================================================================================
[✓ PASS] ALL TESTS PASSED - System is healthy ✓
================================================================================
```

---

### 6. Canary Deployment Guide

**파일**: `docs/CANARY_DEPLOYMENT_GUIDE.md` (1000+ lines)

완전한 카나리 배포 가이드 및 트러블슈팅.

**주요 섹션**:

1. **Overview**: 카나리 배포란 무엇이며 왜 사용하는가
2. **Prerequisites**: 필수 구성 요소 및 환경 설정
3. **Architecture**: 트래픽 흐름 및 컴포넌트 역할
4. **Deployment Process**: 단계별 배포 절차
5. **Traffic Stages**: 각 단계별 상세 설명
6. **Metrics Validation**: 메트릭 검증 기준 및 쿼리
7. **Troubleshooting**: 일반적인 문제 및 해결 방법
8. **Rollback Procedures**: 자동/수동 롤백 절차
9. **Promotion to Production**: 프로덕션 승격 프로세스
10. **Best Practices**: 모범 사례 및 권장사항

**트러블슈팅 예시**:

```markdown
### Issue: High Latency in Canary

**Symptoms**:
- p95 latency > threshold
- Validation fails at Stage 1 or 2

**Common Causes**:
1. LLM API Slow
2. Cold Start
3. Resource Constrained
4. Database Queries

**Solutions**:
- Increase resources (CPU/memory)
- Warm up canary with initial requests
- Scale up workers
- Optimize database queries
```

---

## 🚀 Complete Workflow

### Pre-Deployment

```bash
# Step 1: 릴리스 태그 생성
./scripts/create_release.sh v0.2.0

# Step 2: 사전 배포 체크
python scripts/pre_deployment_check.py --version v0.2.0 --strict

# Step 3: 환경 변수 설정
export CANARY_VERSION=v0.2.0
export USE_LLM_ENSEMBLE=true

# Step 4: 카나리 이미지 빌드
docker-compose -f docker-compose.canary.yml build app-canary
```

### Deployment

```bash
# Step 1: 카나리 인프라 시작
docker-compose -f docker-compose.production.yml -f docker-compose.canary.yml up -d app-canary

# Step 2: 카나리 헬스 체크
curl http://localhost:8001/health

# Step 3: 런치 직후 스모크 테스트 (5분)
python scripts/smoke_test_api.py --canary

# Step 4: 자동 카나리 배포 실행
python scripts/canary_deployment.py --version v0.2.0

# Step 5: 모니터링 (별도 터미널)
docker-compose -f docker-compose.canary.yml logs -f app-canary
docker exec fragrance-ai-nginx tail -f /var/log/nginx/canary.log
```

### Post-Deployment (성공 시)

```bash
# Step 1: 추가 모니터링 (30-60분)
# Grafana 대시보드 확인
open http://localhost:3000/d/canary-deployment

# Step 2: 팀 승인 획득
# - 엔지니어링 리드
# - 프로덕트 오너
# - 온콜 엔지니어

# Step 3: 프로덕션 승격
export VERSION=${CANARY_VERSION}
docker tag fragrance-ai-app:${CANARY_VERSION} fragrance-ai-app:${VERSION}
docker-compose -f docker-compose.production.yml up -d app

# Step 4: 카나리 인프라 제거
docker-compose -f docker-compose.canary.yml down

# Step 5: 문서 업데이트
echo "$(date): Promoted ${CANARY_VERSION} to production" >> deployments.log
git add deployments.log
git commit -m "chore: Promote ${CANARY_VERSION} to production"
```

### If Issues Occur (실패 시)

```bash
# 자동 롤백이 실행되거나, 수동 롤백:

# Quick rollback
./scripts/update_nginx_weights.sh 0

# 카나리 중지
docker-compose -f docker-compose.canary.yml stop app-canary

# 로그 수집
docker-compose -f docker-compose.canary.yml logs app-canary > canary_failure_$(date +%Y%m%d_%H%M%S).log

# 근본 원인 분석
grep ERROR canary_failure_*.log

# 스테이징에서 수정 및 재테스트
# ...

# 수정 후 재배포
python scripts/canary_deployment.py --version v0.2.1
```

---

## 📊 Metrics & Thresholds

### Latency Thresholds (p95)

| LLM Mode | Threshold | Validation |
|----------|-----------|------------|
| **fast** | ≤ 2.5s | 빠른 응답 (단순 추천) |
| **balanced** | ≤ 3.2s | 균형 잡힌 품질 (일반 사용) |
| **creative** | ≤ 4.5s | 창의적 생성 (상세 설명) |

**Prometheus Query**:
```promql
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket{
    version="canary",
    mode="fast"
  }[5m])
)
```

### Error Rate Threshold

**Target**: < 0.5% (5xx errors)

**Prometheus Query**:
```promql
rate(http_requests_total{status=~"5..", version="canary"}[5m])
/
rate(http_requests_total{version="canary"}[5m])
```

### Schema Validation

**Target**: 0% failures (zero tolerance)

**Prometheus Query**:
```promql
rate(llm_brief_schema_failures_total{version="canary"}[5m])
```

### RL Training Loss

**Target**: No spikes > 2x baseline

**Prometheus Query**:
```promql
rl_training_loss{version="canary"}
>
(rl_training_loss{version="production"} * 2)
```

---

## 🎯 Traffic Stages

### Stage 1: 1% Canary

- **Duration**: 15 minutes
- **Traffic**: 1% canary, 99% production
- **Expected Requests**: ~10-50
- **Purpose**: Initial smoke test, detect major issues
- **Validation**: All thresholds must pass

### Stage 2: 5% Canary

- **Duration**: 15 minutes
- **Traffic**: 5% canary, 95% production
- **Expected Requests**: ~50-250
- **Purpose**: Moderate testing, statistical significance
- **Validation**: All thresholds + mode-specific latency

### Stage 3: 25% Canary

- **Duration**: 15 minutes
- **Traffic**: 25% canary, 75% production
- **Expected Requests**: ~250-1250
- **Purpose**: High-confidence testing, resource utilization
- **Validation**: All thresholds + performance consistency

### Stage 4: 100% Canary

- **Duration**: Indefinite (until promotion)
- **Traffic**: 100% canary, 0% production
- **Purpose**: Full rollout, final validation before promotion
- **Actions**: Extended monitoring (30-60 min), team approval, promotion

---

## 🔧 Integration with Existing Systems

### Pre-Deployment Checklist (T-1)

카나리 배포 전 필수 체크:

```bash
# 사전 배포 체크 실행
python scripts/pre_deployment_check.py --version ${CANARY_VERSION} --strict

# 5가지 필수 검증:
# 1. ✓ Release tag (vX.Y.Z + model checkpoints)
# 2. ✓ Migrations (DB/Redis schema)
# 3. ✓ Secrets (.env validation)
# 4. ✓ Artifacts (SHA256 hashes)
# 5. ✓ Runbook (procedures documentation)
```

### Runbook Integration

카나리 배포는 `RUNBOOK.md`와 통합:

- **Health Check Procedures**: 카나리 헬스 체크 포함
- **Scaling Procedures**: 카나리 워커 스케일링
- **Rollback Procedures**: 카나리 롤백 절차
- **Monitoring & Alerts**: 카나리 메트릭 대시보드

### Prometheus & Grafana

카나리 메트릭 수집 및 시각화:

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 'fragrance-ai-production'
    static_configs:
      - targets: ['app:8000']
        labels:
          version: 'production'

  - job_name: 'fragrance-ai-canary'
    static_configs:
      - targets: ['app-canary:8001']
        labels:
          version: 'canary'
```

**Grafana Dashboard**: Canary vs Production 비교

- Side-by-side 메트릭 비교
- 실시간 트래픽 분배
- 에러율 및 지연시간 비교
- 리소스 사용량 비교

---

## 🔗 Related Documentation

- **Pre-Deployment Checklist**: `PRE_DEPLOYMENT_CHECKLIST_SUMMARY.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Runbook**: `RUNBOOK.md`
- **Canary Deployment Guide**: `docs/CANARY_DEPLOYMENT_GUIDE.md`
- **Smoke Test Guide**: `SMOKE_TEST_GUIDE.md` ⭐ NEW
- **Docker Compose (Production)**: `docker-compose.production.yml`
- **Docker Compose (Canary)**: `docker-compose.canary.yml`

---

## 📞 Support

**Emergency Contact**: [On-call engineer]

**Documentation**: `/docs`

**Monitoring**:
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090

**Logs**:
```bash
# Application logs
docker-compose -f docker-compose.canary.yml logs -f app-canary

# NGINX canary logs
docker exec fragrance-ai-nginx tail -f /var/log/nginx/canary.log

# All services
docker-compose logs -f
```

---

## ✅ Success Criteria

배포가 성공적으로 완료되었다고 판단하는 기준:

### Automated Validation

```bash
# 모든 단계 통과
python scripts/canary_deployment.py --version v0.2.0
# Exit code: 0

# 카나리 서비스 정상
curl http://localhost:8001/health
# HTTP 200, status: healthy

# 100% 트래픽 처리 중
curl http://localhost:8080/nginx_status | grep active
# active connections: > 0
```

### Manual Validation

- ✅ 모든 4개 단계 통과 (1%, 5%, 25%, 100%)
- ✅ 각 단계에서 15분 관찰 완료
- ✅ 모든 메트릭 임계값 통과
- ✅ 에러율 < 0.5%
- ✅ LLM 앙상블 정상 동작
- ✅ 스키마 검증 실패 0건
- ✅ 추가 모니터링 30-60분 이상
- ✅ 팀 승인 완료
- ✅ 프로덕션 승격 완료

---

## 🎉 Benefits

카나리 배포 시스템의 이점:

1. **위험 최소화**: 1%부터 시작하여 점진적 확대
2. **빠른 피드백**: 15분마다 검증 및 의사결정
3. **자동 복구**: 문제 발생 시 자동 롤백
4. **실제 부하 테스트**: 프로덕션 트래픽으로 검증
5. **LLM 앙상블 검증**: 3-모델 앙상블 실제 성능 확인
6. **팀 신뢰도 향상**: 자동화된 검증으로 배포 신뢰도 증가
7. **다운타임 제로**: 무중단 배포
8. **문제 격리**: 카나리만 영향받음

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Implementation**: Complete ✅
**Maintained by**: DevOps Team
