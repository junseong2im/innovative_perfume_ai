# Pre-Deployment Checklist (T-1)
# 최종 사전 점검 시스템

프로덕션 배포 전 필수 검증 체크리스트 및 자동화 도구

---

## ✅ Implementation Complete

5가지 핵심 검증 항목 모두 구현 완료:

1. ✅ **릴리스 태그**: vX.Y.Z + 모델 체크포인트 tag 포함
2. ✅ **마이그레이션**: DB/Redis 스키마 변화 확인
3. ✅ **시크릿**: .env/KMS 값 검증 (빈값/오타 체크)
4. ✅ **아티팩트 해시**: 모델 가중치/컨테이너 이미지 SHA256 기록
5. ✅ **런북 링크**: 헬스체크/다운시프트/롤백 절차 문서 최신화

---

## 📁 Created Files

### 1. Pre-Deployment Check Script

**파일**: `scripts/pre_deployment_check.py`

완전 자동화된 배포 전 검증 스크립트

```bash
# 기본 사용
python scripts/pre_deployment_check.py --version v0.2.0

# Strict 모드 (warning도 실패로 처리)
python scripts/pre_deployment_check.py --version v0.2.0 --strict

# 결과를 JSON으로 저장
python scripts/pre_deployment_check.py --version v0.2.0 --output results.json
```

**검증 항목**:

1. **Release Tag Verification** (릴리스 태그 검증)
   - ✓ Version format validation (vX.Y.Z)
   - ✓ Git tag exists
   - ✓ Model checkpoint tags present
   - ✓ Current commit identified

2. **Migration Verification** (마이그레이션 검증)
   - ✓ Alembic migration files present
   - ✓ All migrations committed
   - ✓ Schema changes documented
   - ✓ Redis schema compatibility

3. **Secret Validation** (시크릿 검증)
   - ✓ .env file exists
   - ✓ Required variables set
   - ✓ No empty values
   - ✓ No default/example values
   - ✓ SECRET_KEY strength check
   - ✓ No hardcoded secrets in code

4. **Artifact Hash Recording** (아티팩트 해시 기록)
   - ✓ Calculate model file SHA256 hashes
   - ✓ Record Docker image digests
   - ✓ Generate artifact manifest
   - ✓ Save manifest hash

5. **Runbook Verification** (런북 검증)
   - ✓ Runbook file exists
   - ✓ Health check procedures documented
   - ✓ Rollback procedures documented
   - ✓ Scaling procedures documented
   - ✓ Monitoring procedures documented
   - ✓ Document freshness check
   - ✓ No broken links

**출력 예시**:

```
================================================================================
PRE-DEPLOYMENT CHECKLIST (T-1)
================================================================================
Version: v0.2.0
Date: 2025-10-14 15:30:00
================================================================================

================================================================================
CHECK 1: Release Tag Verification
================================================================================
[✓ PASS] Version format valid: v0.2.0
[✓ PASS] Git tag exists: v0.2.0
[✓ PASS] Model checkpoint version tag exists
[✓ PASS] Current commit identified: abc123de

================================================================================
CHECK 2: Migration Verification
================================================================================
[✓ PASS] Migration files present
[✓ PASS] All migrations are committed
[✓ PASS] Schema documentation found
[✓ PASS] No Redis schema changes detected

================================================================================
CHECK 3: Secret Validation
================================================================================
[✓ PASS] Environment file exists: .env
[✓ PASS] ✓ SECRET_KEY
[✓ PASS] ✓ POSTGRES_PASSWORD
[✓ PASS] ✓ DATABASE_URL
[✓ PASS] ✓ REDIS_URL
[✓ PASS] SECRET_KEY length OK: 64 chars

================================================================================
CHECK 4: Artifact Hash Recording
================================================================================
[✓ PASS] Calculated hashes for 5 model files
[✓ PASS] Found digests for 5 container images
[✓ PASS] Artifact manifest saved: releases/manifest_v0.2.0.json
[✓ PASS] Manifest hash saved: releases/manifest_v0.2.0.sha256

================================================================================
CHECK 5: Runbook Verification
================================================================================
[✓ PASS] Runbook found: RUNBOOK.md
[✓ PASS] ✓ Health procedures documented
[✓ PASS] ✓ Rollback procedures documented
[✓ PASS] ✓ Scaling procedures documented
[✓ PASS] ✓ Monitoring procedures documented
[✓ PASS] Runbook is up to date
[✓ PASS] No broken links found

================================================================================
FINAL SUMMARY
================================================================================
✓ PASS release_tag
✓ PASS migrations
✓ PASS secrets
✓ PASS artifacts
✓ PASS runbook

Result: 5/5 checks passed
================================================================================
✓ PRE-DEPLOYMENT CHECK: PASSED
✓ Ready for production deployment!
================================================================================
```

---

### 2. Release Tagging Script

**파일**: `scripts/create_release.sh`

릴리스 태그 자동 생성 및 모델 체크포인트 버전 기록

```bash
# 기본 사용
./scripts/create_release.sh v0.2.0

# 태그를 원격으로 푸시
./scripts/create_release.sh v0.2.0 --push
```

**기능**:

1. **Git Tag 생성**
   - Version format 검증 (vX.Y.Z)
   - Branch 확인 (main/master)
   - Uncommitted changes 확인
   - Annotated tag 생성
   - Release notes 입력

2. **Model Checkpoint Tagging**
   - `checkpoints/` 디렉토리 스캔
   - 모든 .pt, .pth, .bin 파일 해시 계산
   - `VERSION_v0.2.0.txt` 생성
   - SHA256 해시 포함

3. **Artifact Manifest 생성**
   - `releases/manifest_v0.2.0.json` 생성
   - Git commit 정보 기록
   - Docker 이미지 정보 기록
   - Manifest SHA256 계산

**체크포인트 버전 파일 예시**:

```
v0.2.0
Commit: abc123de
Date: 2025-10-14T15:30:00Z

Checkpoint Files:
  model_epoch_100.pt: 3f2a5b9c8d7e1f4a6b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a
  optimizer_state.pt: 4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5
```

**Artifact Manifest 예시**:

```json
{
  "version": "v0.2.0",
  "release_date": "2025-10-14T15:30:00Z",
  "git": {
    "commit": "abc123def456...",
    "branch": "main",
    "tag": "v0.2.0"
  },
  "artifacts": {
    "models": "tagged",
    "checkpoints": "tagged"
  },
  "docker_images": {
    "app": "fragrance-ai-app:v0.2.0",
    "worker-llm": "fragrance-ai-worker-llm:v0.2.0",
    "worker-rl": "fragrance-ai-worker-rl:v0.2.0"
  }
}
```

---

### 3. Operations Runbook

**파일**: `RUNBOOK.md`

완전한 운영 절차서 (헬스체크, 스케일링, 롤백, 인시던트 대응)

**주요 섹션**:

#### 1. Health Check Procedures

```bash
# Quick health check
curl -f http://localhost:8000/health

# Comprehensive check
docker-compose ps
docker-compose exec postgres pg_isready
docker-compose exec redis redis-cli ping
```

**Health Check Schedule**:
- API /health: Every 30s
- Database: Every 60s
- Redis: Every 60s
- Worker queue: Every 5min
- Disk/Memory: Every 5-10min

#### 2. Scaling Procedures

**Upscaling (증설)**:

```bash
# Scale app instances
docker-compose up -d --scale app=5

# Scale LLM workers
docker-compose up -d --scale worker-llm=10

# Scale RL workers
docker-compose up -d --scale worker-rl=4
```

**When to Scale**:
- App: Response time > 500ms, CPU > 70%
- LLM: Queue length > 50, Wait time > 10s
- RL: Training backlog > 10 jobs

**Downscaling (축소)**:

```bash
# Gradual downscale (recommended)
docker-compose up -d --scale app=3
# Wait and monitor
docker-compose up -d --scale app=2
```

#### 3. Rollback Procedures

**Quick Rollback** (2-5 minutes):

```bash
# Set previous version
export VERSION=v0.1.0

# Restart with previous version
docker-compose down
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

**Full Rollback** (10-20 minutes) - with database:

```bash
# Use automated script
./scripts/rollback.sh v0.1.0

# Or manual steps:
# 1. Stop services
# 2. Restore database backup
# 3. Rollback code
# 4. Verify
```

**Rollback Decision Tree**:
- API Errors > 5% → Quick Rollback
- Database Errors → Full Rollback with DB
- Performance 2x slower → Quick Rollback
- Worker Failures > 50% → Quick Rollback (workers only)
- Data Corruption → Full Rollback + Investigation

#### 4. Incident Response

**Severity Levels**:
- **P0 (Critical)**: Service down, data loss - Response: Immediate
- **P1 (High)**: Major feature broken - Response: 15 minutes
- **P2 (Medium)**: Minor feature broken - Response: 4 hours
- **P3 (Low)**: Cosmetic issues - Response: 2 days

**P0 Response** (first 5 minutes):

```bash
# 1. Confirm incident
curl http://localhost:8000/health

# 2. Check status
docker-compose ps

# 3. Notify team

# 4. Start incident log
echo "$(date): Incident - [description]" >> incidents/$(date +%Y%m%d).log
```

**Mitigation Options**:
1. Restart services
2. Rollback to previous version
3. Scale down problematic component

#### 5. Monitoring & Alerts

**Grafana Dashboards**: http://localhost:3000
- System Overview (CPU, Memory, Disk)
- Application Metrics (Requests, Latency, Errors)
- Worker Metrics (Queue, Processing time)
- Database Metrics (Connections, Queries)

**Key Prometheus Queries**:

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Response time (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Worker queue
celery_queue_length

# Database connections
pg_stat_database_numbackends
```

**Alert Rules**:
- High error rate (> 5% for 5min)
- High response time (p95 > 1s for 5min)
- Worker queue backlog (> 100 for 10min)
- Database pool exhausted (> 90% for 5min)

---

## 🚀 Complete Workflow

### Pre-Deployment (T-1)

```bash
# Step 1: Create release tag
./scripts/create_release.sh v0.2.0

# Step 2: Build Docker images
export VERSION=v0.2.0
docker-compose -f docker-compose.production.yml build

# Step 3: Run pre-deployment check
python scripts/pre_deployment_check.py --version v0.2.0 --strict

# Step 4: Review results
cat releases/manifest_v0.2.0.json

# Step 5: Generate release notes (if not auto-generated)
python scripts/generate_release_notes.py --from v0.1.0 --to v0.2.0 --output RELEASE_NOTES_v0.2.0.md
```

### Deployment

```bash
# Step 1: Backup
./scripts/backup.sh

# Step 2: Deploy (rolling update)
./scripts/deploy.sh production v0.2.0 rolling

# Step 3: Verify
curl http://localhost:8000/health
docker-compose ps

# Step 4: Monitor
# Watch Grafana dashboards for 30 minutes
```

### Post-Deployment

```bash
# Step 1: Verify all checks pass
python scripts/pre_deployment_check.py --version v0.2.0

# Step 2: Run smoke tests
./scripts/smoke_test.sh

# Step 3: Check metrics
curl http://localhost:9090/api/v1/query?query=up

# Step 4: Document deployment
echo "$(date): Deployed v0.2.0 successfully" >> deployments.log
```

### If Issues Occur

```bash
# Quick rollback
./scripts/rollback.sh v0.1.0

# Verify rollback
curl http://localhost:8000/health

# Check error rates
docker-compose logs -f --tail=500 | grep ERROR
```

---

## 📋 Checklist Template

### T-1 Pre-Deployment Checklist

**Release Information**:
- [ ] Version: `____________`
- [ ] Target Environment: `____________`
- [ ] Deployment Date/Time: `____________`
- [ ] Deployed By: `____________`

**1. Release Tag** ✓
- [ ] Git tag created: `v________`
- [ ] Model checkpoints tagged
- [ ] Artifact manifest generated
- [ ] Manifest SHA256 recorded

**2. Migrations** ✓
- [ ] Database migrations reviewed
- [ ] Redis schema changes reviewed
- [ ] Migration scripts tested in staging
- [ ] Rollback plan documented
- [ ] No breaking schema changes

**3. Secrets** ✓
- [ ] .env file exists and updated
- [ ] All required variables set
- [ ] No empty values
- [ ] No default/placeholder values
- [ ] SECRET_KEY strength verified (≥ 32 chars)
- [ ] No hardcoded secrets in code

**4. Artifacts** ✓
- [ ] Model weights hashed (SHA256)
- [ ] Container images built and tagged
- [ ] Image digests recorded
- [ ] Artifact manifest created
- [ ] Manifest integrity verified

**5. Runbook** ✓
- [ ] RUNBOOK.md exists and up-to-date
- [ ] Health check procedures documented
- [ ] Rollback procedures documented
- [ ] Scaling procedures documented
- [ ] Monitoring procedures documented
- [ ] Emergency contacts updated
- [ ] No broken links

**6. Testing** ✓
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Smoke tests passing
- [ ] Performance tests passing
- [ ] Security scan completed

**7. Communication** ✓
- [ ] Team notified
- [ ] Stakeholders informed
- [ ] Maintenance window scheduled
- [ ] Status page updated (if applicable)
- [ ] Runbook shared with on-call

**8. Backup** ✓
- [ ] Database backup created
- [ ] Backup verified
- [ ] Backup location documented
- [ ] Retention policy confirmed

**9. Monitoring** ✓
- [ ] Grafana dashboards verified
- [ ] Alert rules reviewed
- [ ] On-call engineer assigned
- [ ] Incident response plan reviewed

**10. Final Verification** ✓
- [ ] Pre-deployment check script passed
- [ ] All items above checked
- [ ] Deployment approved
- [ ] Ready for production

**Sign-off**:
- Engineer: `____________` Date: `____________`
- Reviewer: `____________` Date: `____________`

---

## 🎯 Success Criteria

배포가 성공적으로 완료되었다고 판단하는 기준:

### Automated Checks

```bash
# All checks must pass
python scripts/pre_deployment_check.py --version v0.2.0 --strict
# Exit code: 0

# All services healthy
docker-compose ps | grep Up | wc -l
# Count: 8 (all services)

# API responding
curl -f http://localhost:8000/health
# HTTP 200, status: healthy

# No errors in logs
docker-compose logs --since 10m | grep -i error | wc -l
# Count: 0
```

### Manual Checks

- ✅ All pre-deployment checks passed
- ✅ Services deployed without errors
- ✅ Health endpoints returning 200
- ✅ Database migrations applied successfully
- ✅ No error spikes in metrics
- ✅ Response time within SLA (< 500ms p95)
- ✅ Worker queues processing normally
- ✅ Monitoring dashboards show green
- ✅ Team notified of successful deployment

---

## 🔗 Related Documentation

- **Deployment Guide**: `DEPLOYMENT_GUIDE.md` - Full deployment procedures
- **Runbook**: `RUNBOOK.md` - Operational procedures
- **Release Notes Template**: `.github/RELEASE_TEMPLATE.md`
- **Docker Compose**: `docker-compose.production.yml` - Infrastructure as code

---

## 📞 Support

**Emergency Contact**: [On-call engineer]

**Documentation**: `/docs`

**Monitoring**: http://localhost:3000 (Grafana)

**Metrics**: http://localhost:9090 (Prometheus)

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Maintained by**: DevOps Team
