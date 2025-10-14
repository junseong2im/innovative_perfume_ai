# Canary Deployment Guide
# Fragrance AI - Progressive Rollout System

Complete guide for safe production deployments using canary releases with progressive traffic rollout.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Deployment Process](#deployment-process)
5. [Traffic Stages](#traffic-stages)
6. [Metrics Validation](#metrics-validation)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)
9. [Promotion to Production](#promotion-to-production)
10. [Best Practices](#best-practices)

---

## Overview

### What is Canary Deployment?

Canary deployment is a progressive rollout strategy that minimizes risk by:
- Deploying new version to a small subset of traffic first (1%)
- Validating metrics and behavior at each stage
- Gradually increasing traffic: 1% → 5% → 25% → 100%
- Automatically rolling back if issues are detected

### Why Use Canary Deployment?

- **Risk Mitigation**: Detect issues before they affect all users
- **Gradual Validation**: Observe behavior under real production load
- **Automatic Rollback**: Fail fast and revert to stable version
- **Confidence Building**: Progressive validation at each stage

### Key Features

- **4-Stage Rollout**: 1% → 5% → 25% → 100% traffic
- **15-Minute Observation**: Monitor metrics at each stage
- **Automatic Validation**: Latency, error rate, schema failures, RL loss
- **LLM Ensemble Testing**: Test 3-model ensemble under real load
- **Weighted Routing**: NGINX-based traffic distribution
- **Prometheus Metrics**: Real-time monitoring and alerts

---

## Prerequisites

### Required Components

1. **Production Infrastructure Running**
   ```bash
   docker-compose -f docker-compose.production.yml up -d
   ```

2. **Monitoring Stack Enabled**
   ```bash
   docker-compose -f docker-compose.production.yml --profile monitoring up -d
   ```

3. **Pre-Deployment Checks Passed**
   ```bash
   python scripts/pre_deployment_check.py --version v0.2.0 --strict
   ```

4. **Canary Version Built and Tagged**
   ```bash
   export VERSION=v0.2.0
   docker-compose -f docker-compose.canary.yml build app-canary
   ```

### Environment Configuration

Create or update `.env` file:

```bash
# Canary Configuration
CANARY_VERSION=v0.2.0          # Version to deploy
CANARY_API_PORT=8001            # Canary service port

# LLM Ensemble (enabled for canary)
USE_LLM_ENSEMBLE=true
LLM_ENSEMBLE_MODELS=gpt-3.5-turbo,claude-3-haiku,gemini-pro
LLM_ENSEMBLE_VOTING=weighted

# Worker Scaling (optional)
LLM_WORKER_CANARY_REPLICAS=1    # Canary LLM workers
RL_WORKER_CANARY_REPLICAS=0     # Canary RL workers (usually 0)

# Prometheus Endpoints
PROMETHEUS_URL=http://localhost:9090
```

### Verify Baseline Metrics

Before starting canary deployment, capture baseline metrics:

```bash
# Check production is healthy
curl http://localhost:8000/health

# Get baseline latency (p95)
curl -s 'http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))' | jq

# Get baseline error rate
curl -s 'http://localhost:9090/api/v1/query?query=rate(http_requests_total{status=~"5.."}[5m])' | jq
```

---

## Architecture

### Traffic Flow

```
                    ┌─────────────┐
                    │   NGINX     │
                    │  (weighted) │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         99% traffic              1% traffic
              │                         │
      ┌───────▼────────┐       ┌───────▼────────┐
      │  Production    │       │    Canary      │
      │   (v0.1.0)     │       │   (v0.2.0)     │
      │                │       │  LLM Ensemble  │
      │  app:8000      │       │  app:8001      │
      └────────────────┘       └────────────────┘
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │  PostgreSQL │
                    │    Redis    │
                    └─────────────┘
```

### Component Roles

| Component | Purpose | Canary Behavior |
|-----------|---------|----------------|
| **NGINX** | Traffic distribution | Weighted routing (production + canary) |
| **app-canary** | Canary application | Runs new version with LLM ensemble |
| **worker-llm-canary** | Canary LLM workers | Optional, for testing worker changes |
| **Prometheus** | Metrics collection | Separate labels for production vs canary |
| **Grafana** | Visualization | Dashboards compare both versions |

---

## Deployment Process

### Step 1: Prepare Canary Release

```bash
# 1. Create release tag
./scripts/create_release.sh v0.2.0

# 2. Run pre-deployment checks
python scripts/pre_deployment_check.py --version v0.2.0 --strict

# 3. Set environment variables
export CANARY_VERSION=v0.2.0
export USE_LLM_ENSEMBLE=true

# 4. Build canary images
docker-compose -f docker-compose.canary.yml build app-canary
```

### Step 2: Start Canary Infrastructure

```bash
# Start canary service alongside production
docker-compose -f docker-compose.production.yml -f docker-compose.canary.yml up -d app-canary

# Verify canary is healthy
docker-compose -f docker-compose.canary.yml ps app-canary
curl http://localhost:8001/health
```

Expected output:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "deployment": "canary",
  "llm_ensemble": {
    "enabled": true,
    "models": ["gpt-3.5-turbo", "claude-3-haiku", "gemini-pro"]
  },
  "timestamp": "2025-10-14T12:00:00Z"
}
```

### Step 3: Run Automated Canary Deployment

```bash
# Automated deployment with default stages (1%, 5%, 25%, 100%)
python scripts/canary_deployment.py --version v0.2.0

# With custom observation period (30 minutes per stage)
python scripts/canary_deployment.py --version v0.2.0 --observation-period 1800

# Dry run (no traffic changes)
python scripts/canary_deployment.py --version v0.2.0 --dry-run
```

The script will:
1. ✅ Verify canary service is healthy
2. ✅ Set traffic to 1% canary, observe for 15 minutes
3. ✅ Validate metrics (latency, error rate, schema failures)
4. ✅ Progress to 5% if validation passes
5. ✅ Continue to 25%, then 100%
6. ⚠️ Automatically rollback if any validation fails

### Step 4: Monitor Deployment

Open multiple terminal windows to monitor:

**Terminal 1: Canary deployment output**
```bash
python scripts/canary_deployment.py --version v0.2.0
```

**Terminal 2: Canary logs**
```bash
docker-compose -f docker-compose.canary.yml logs -f app-canary
```

**Terminal 3: NGINX access logs (canary traffic)**
```bash
docker exec fragrance-ai-nginx tail -f /var/log/nginx/canary.log
```

**Terminal 4: Grafana dashboard**
```bash
# Open in browser
open http://localhost:3000/d/canary-deployment
```

---

## Traffic Stages

### Stage 1: 1% Canary Traffic

**Purpose**: Initial smoke test under minimal production load

**Traffic Distribution**:
- Production: 99%
- Canary: 1%

**Duration**: 15 minutes

**Validation Criteria**:
- ✅ Canary service healthy
- ✅ p95 latency within thresholds
- ✅ Error rate < 0.5%
- ✅ No schema validation failures
- ✅ No critical errors in logs

**Expected Traffic**:
- ~10-50 requests (assuming 1000-5000 req/15min)
- Enough to detect major issues
- Minimal user impact if failure

### Stage 2: 5% Canary Traffic

**Purpose**: Moderate testing with more statistical significance

**Traffic Distribution**:
- Production: 95%
- Canary: 5%

**Duration**: 15 minutes

**Validation Criteria**:
- ✅ All Stage 1 criteria
- ✅ LLM ensemble latency acceptable:
  - fast mode: ≤ 2.5s (p95)
  - balanced mode: ≤ 3.2s (p95)
  - creative mode: ≤ 4.5s (p95)
- ✅ llm_brief schema failure rate: 0%
- ✅ No RL training loss spikes (if RL active)

**Expected Traffic**:
- ~50-250 requests
- Statistical significance improves
- Mode-specific validation

### Stage 3: 25% Canary Traffic

**Purpose**: High-confidence testing before full rollout

**Traffic Distribution**:
- Production: 75%
- Canary: 25%

**Duration**: 15 minutes

**Validation Criteria**:
- ✅ All Stage 2 criteria
- ✅ Consistent performance across all modes
- ✅ No degradation in production metrics
- ✅ Resource utilization acceptable
- ✅ No memory leaks or resource exhaustion

**Expected Traffic**:
- ~250-1250 requests
- Detect performance issues
- Resource utilization test

### Stage 4: 100% Canary Traffic

**Purpose**: Full rollout to production

**Traffic Distribution**:
- Production: 0%
- Canary: 100%

**Duration**: Indefinite (until promotion)

**Validation**:
- ✅ All previous criteria
- ✅ Extended monitoring (30-60 minutes)
- ✅ No regressions vs production baseline
- ✅ Team approval for promotion

**Post-100% Actions**:
1. Monitor for 30-60 minutes
2. Get team approval
3. Promote canary to production
4. Update production version tag
5. Remove canary infrastructure

---

## Metrics Validation

### Latency Thresholds

Validated at **p95 (95th percentile)** for each LLM mode:

```python
THRESHOLDS = {
    'fast': {
        'p95_latency': 2.5,      # 2.5 seconds
    },
    'balanced': {
        'p95_latency': 3.2,      # 3.2 seconds
    },
    'creative': {
        'p95_latency': 4.5,      # 4.5 seconds
    },
}
```

**Why p95?**
- Captures typical user experience (95% of requests)
- Filters outliers (network issues, cold starts)
- Industry standard for SLA monitoring

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

**Calculation**:
```
error_rate = (5xx_count / total_requests) * 100
```

**Prometheus Query**:
```promql
rate(http_requests_total{status=~"5..", version="canary"}[5m])
/
rate(http_requests_total{version="canary"}[5m])
```

**Critical Threshold**: > 1% triggers immediate rollback

### Schema Validation

**Target**: 0% failures (strict)

**Validated Fields**:
- `llm_brief`: LLM-generated perfume description
- Required structure, format, length
- JSON schema compliance

**Prometheus Query**:
```promql
rate(llm_brief_schema_failures_total{version="canary"}[5m])
```

**Why Zero Tolerance?**
- Schema failures break downstream processing
- Indicates LLM ensemble issues
- User-facing data corruption

### RL Training Loss

**Target**: No spikes > 2x baseline

**Comparison**:
```
spike_detected = canary_loss > (production_loss * 2.0)
```

**Prometheus Query**:
```promql
rl_training_loss{version="canary"}
>
(rl_training_loss{version="production"} * 2)
```

**Note**: Only validated if RL training is active

---

## Troubleshooting

### Issue: Canary Service Won't Start

**Symptoms**:
```bash
docker-compose ps app-canary
# Status: Exit 1
```

**Diagnosis**:
```bash
# Check logs
docker-compose -f docker-compose.canary.yml logs app-canary

# Common causes:
# - Database connection failure
# - Missing environment variables
# - Port conflict (8001 in use)
# - Image build failure
```

**Solutions**:
```bash
# Fix database connection
docker-compose -f docker-compose.production.yml restart postgres

# Check environment variables
docker-compose -f docker-compose.canary.yml config

# Find process using port 8001
netstat -ano | findstr :8001
taskkill /PID <pid> /F

# Rebuild image
docker-compose -f docker-compose.canary.yml build --no-cache app-canary
```

### Issue: High Latency in Canary

**Symptoms**:
- p95 latency > threshold
- Validation fails at Stage 1 or 2

**Diagnosis**:
```bash
# Check LLM ensemble performance
curl http://localhost:8001/metrics | grep llm_ensemble

# Check resource utilization
docker stats app-canary

# Check external API latency
docker-compose logs app-canary | grep "llm_api_latency"
```

**Common Causes**:
1. **LLM API Slow**: External LLM providers slow
2. **Cold Start**: First requests after deployment
3. **Resource Constrained**: CPU/memory limits
4. **Database Queries**: Slow queries or connection pool exhaustion

**Solutions**:
```bash
# Increase resources
# Edit docker-compose.canary.yml:
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G

# Warm up canary
for i in {1..50}; do
  curl -X POST http://localhost:8001/api/generate \
    -H "Content-Type: application/json" \
    -d '{"mode":"fast","notes":["citrus","floral"]}'
done

# Scale up workers
docker-compose -f docker-compose.canary.yml up -d --scale worker-llm-canary=2
```

### Issue: Error Rate Spike

**Symptoms**:
- Error rate > 0.5%
- 5xx errors in logs

**Diagnosis**:
```bash
# Check error types
docker-compose logs app-canary | grep ERROR

# Check specific error codes
curl http://localhost:8001/metrics | grep http_requests_total

# Check database/redis health
docker-compose ps postgres redis
```

**Common Errors**:
1. **502 Bad Gateway**: Canary service crashed
2. **503 Service Unavailable**: Overloaded
3. **500 Internal Server Error**: Application bug
4. **504 Gateway Timeout**: Request timeout

**Solutions**:
```bash
# Restart canary if crashed
docker-compose -f docker-compose.canary.yml restart app-canary

# Check for application bugs
docker-compose logs app-canary --tail=100 | grep -A 10 "Traceback"

# Increase timeouts (nginx)
# Edit nginx/nginx.canary.conf:
proxy_read_timeout 600s;

# Rollback immediately if critical
./scripts/update_nginx_weights.sh 0
```

### Issue: Schema Validation Failures

**Symptoms**:
- `llm_brief_schema_failure_rate > 0`
- Validation fails immediately

**Diagnosis**:
```bash
# Check LLM outputs
docker-compose logs app-canary | grep llm_brief

# Example failure:
# ERROR: Schema validation failed: Missing required field 'description'
# LLM Output: {"notes": [...], "families": [...]}
```

**Root Causes**:
1. **LLM Ensemble Misconfiguration**: Wrong models or prompts
2. **Model API Changes**: Provider updated response format
3. **Prompt Engineering**: Prompt doesn't enforce schema
4. **Parsing Bug**: Response parsing logic broken

**Solutions**:
```bash
# Check LLM ensemble configuration
curl http://localhost:8001/api/config | jq .llm_ensemble

# Test individual LLM models
curl -X POST http://localhost:8001/api/test-llm \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","prompt":"Generate perfume description"}'

# Rollback and fix in staging
./scripts/update_nginx_weights.sh 0
# Fix prompt engineering
# Re-deploy after validation in staging
```

### Issue: Deployment Stuck at Stage

**Symptoms**:
- Observation period completed
- Metrics pass validation
- Script doesn't progress to next stage

**Diagnosis**:
```bash
# Check script logs
tail -f /tmp/canary_deployment.log

# Check Prometheus connectivity
curl http://localhost:9090/-/healthy

# Check NGINX reload
docker exec fragrance-ai-nginx nginx -t
```

**Solutions**:
```bash
# Restart canary deployment script
# Ctrl+C to cancel, then restart:
python scripts/canary_deployment.py --version v0.2.0 --start-stage 5

# Manual progression (if script fails)
./scripts/update_nginx_weights.sh 5
# Wait 15 minutes, validate metrics manually
./scripts/update_nginx_weights.sh 25
```

---

## Rollback Procedures

### Automatic Rollback

Triggered when validation fails at any stage:

```python
# Automatic rollback conditions
if error_rate > 0.005:  # > 0.5%
    rollback("High error rate")

if p95_latency > threshold:
    rollback("Latency threshold exceeded")

if schema_failures > 0:
    rollback("Schema validation failures")

if rl_loss_spike > 2.0:
    rollback("RL training loss spike")
```

**Rollback Actions**:
1. ⚠️ Alert: "Canary validation failed"
2. 🔄 Reset traffic to 0% canary (100% production)
3. 📊 Capture metrics snapshot
4. 📝 Log failure reason
5. 🛑 Stop canary deployment script

### Manual Rollback

If you need to manually rollback:

```bash
# Quick rollback (reset traffic)
./scripts/update_nginx_weights.sh 0

# Verify traffic reset
docker exec fragrance-ai-nginx cat /etc/nginx/conf.d/upstream.conf

# Stop canary services
docker-compose -f docker-compose.canary.yml stop app-canary

# Capture logs for investigation
docker-compose -f docker-compose.canary.yml logs app-canary > canary_failure_$(date +%Y%m%d_%H%M%S).log

# Remove canary infrastructure (optional)
docker-compose -f docker-compose.canary.yml down
```

### Post-Rollback Actions

1. **Investigate Root Cause**
   ```bash
   # Analyze failure logs
   grep ERROR canary_failure_*.log

   # Check metrics at failure time
   # (Use Grafana to view metrics history)
   ```

2. **Document Incident**
   - Create incident report
   - Record failure reason
   - Document fixes needed

3. **Fix in Staging**
   - Reproduce issue in staging
   - Apply fix
   - Re-test thoroughly

4. **Re-attempt Deployment**
   - After fix validated in staging
   - Create new release version
   - Restart canary deployment

---

## Promotion to Production

After successful 100% canary deployment, promote to production:

### Step 1: Extended Monitoring

Monitor for 30-60 minutes at 100% canary:

```bash
# Check all metrics are stable
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total{version="canary"}[30m])

# Compare to baseline
curl http://localhost:9090/api/v1/query?query=rate(http_requests_total{version="production"}[30m])
```

### Step 2: Team Approval

Get sign-off from:
- ✅ Engineering lead
- ✅ Product owner
- ✅ On-call engineer

### Step 3: Promote Canary Version

```bash
# Update production version
export VERSION=${CANARY_VERSION}

# Tag canary version as production
docker tag fragrance-ai-app:${CANARY_VERSION} fragrance-ai-app:${VERSION}
docker tag fragrance-ai-app:${CANARY_VERSION} fragrance-ai-app:latest

# Update production services
docker-compose -f docker-compose.production.yml up -d app

# Verify production is running new version
curl http://localhost:8000/health
```

### Step 4: Remove Canary Infrastructure

```bash
# Stop canary services
docker-compose -f docker-compose.canary.yml down

# Reset NGINX to production config
docker cp nginx/nginx.prod.conf fragrance-ai-nginx:/etc/nginx/nginx.conf
docker exec fragrance-ai-nginx nginx -s reload

# Clean up canary volumes (optional)
docker volume rm fragrance-ai-app-canary-logs
```

### Step 5: Update Documentation

```bash
# Update CHANGELOG
echo "## [${VERSION}] - $(date +%Y-%m-%d)" >> CHANGELOG.md
echo "### Changed" >> CHANGELOG.md
echo "- Promoted canary ${CANARY_VERSION} to production" >> CHANGELOG.md

# Update deployment log
echo "$(date): Promoted ${CANARY_VERSION} to production after successful canary deployment" >> deployments.log

# Commit changes
git add CHANGELOG.md deployments.log
git commit -m "chore: Promote ${CANARY_VERSION} to production"
git push
```

---

## Best Practices

### 1. Always Run Pre-Deployment Checks

```bash
# Never skip this step
python scripts/pre_deployment_check.py --version ${VERSION} --strict
```

### 2. Test in Staging First

- Deploy to staging environment
- Run full test suite
- Validate LLM ensemble behavior
- Check for regressions

### 3. Monitor Baseline Metrics

Before canary deployment:
- Capture production baseline
- Document current performance
- Set realistic thresholds

### 4. Use Gradual Rollout

Don't skip stages:
- ❌ Don't jump from 1% to 100%
- ✅ Follow 1% → 5% → 25% → 100%

### 5. Communicate with Team

- Notify team before deployment
- Share deployment progress
- Document any issues
- Get approval before promotion

### 6. Keep Canary Infrastructure Lightweight

- Use minimal worker replicas
- Share production database/redis
- Monitor resource usage

### 7. Set Realistic Observation Periods

- Default: 15 minutes per stage
- Increase for critical deployments
- Consider traffic patterns

### 8. Automate Rollback

- Don't wait for manual intervention
- Trust automated validation
- Fail fast and rollback quickly

### 9. Document Everything

- Log all deployment activities
- Capture metrics snapshots
- Document failures and fixes
- Update runbooks

### 10. Review and Iterate

After each deployment:
- Review metrics and logs
- Identify improvement opportunities
- Update thresholds if needed
- Refine validation criteria

---

## Appendix

### Quick Reference Commands

```bash
# Start canary deployment
python scripts/canary_deployment.py --version v0.2.0

# Manual traffic control
./scripts/update_nginx_weights.sh <percentage>

# Check canary health
curl http://localhost:8001/health

# View canary logs
docker-compose -f docker-compose.canary.yml logs -f app-canary

# View NGINX canary logs
docker exec fragrance-ai-nginx tail -f /var/log/nginx/canary.log

# Quick rollback
./scripts/update_nginx_weights.sh 0

# Promote to production
export VERSION=${CANARY_VERSION}
docker-compose -f docker-compose.production.yml up -d app
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CANARY_VERSION` | Canary version tag | - (required) |
| `CANARY_API_PORT` | Canary service port | 8001 |
| `USE_LLM_ENSEMBLE` | Enable LLM ensemble | true |
| `LLM_WORKER_CANARY_REPLICAS` | Canary LLM workers | 1 |
| `RL_WORKER_CANARY_REPLICAS` | Canary RL workers | 0 |
| `PROMETHEUS_URL` | Prometheus endpoint | http://localhost:9090 |

### Useful Links

- **Pre-Deployment Checklist**: `PRE_DEPLOYMENT_CHECKLIST_SUMMARY.md`
- **Runbook**: `RUNBOOK.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Rollback Script**: `scripts/rollback.sh`
- **Monitoring Dashboards**: http://localhost:3000

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Maintained by**: DevOps Team
