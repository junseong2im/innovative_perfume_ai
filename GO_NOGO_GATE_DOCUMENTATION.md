# Go/No-Go Deployment Gate Documentation

**신호등 시스템**: 배포 가능 여부 자동 판단

---

## 1. Overview

The Go/No-Go gate is an automated deployment readiness check that evaluates system health and performance metrics before allowing a production deployment.

### 1.1 Gate Statuses

| Status | Symbol | Meaning | Action |
|--------|--------|---------|--------|
| **GO** | 🟢 | All checks passed | Deployment approved |
| **WARNING** | 🟡 | Some warnings detected | Review before deploying |
| **NO-GO** | 🔴 | Critical issues detected | Deployment blocked |

---

## 2. Go Criteria (녹색 신호등)

All of the following must be true for GO status:

### 2.1 Schema Validation
- **스키마 실패율**: 0% (zero tolerance)
- **Metric**: `llm_schema_fix_count_total`
- **Query**: `sum(rate(llm_schema_fix_count_total[30m])) / sum(rate(llm_brief_total[30m]))`
- **Threshold**: 0.0

### 2.2 API Performance
- **API 에러율**: < 0.5%
- **Metric**: `http_requests_total{status=~"5.."}`
- **Query**: `sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))`
- **Threshold**: 0.005 (0.5%)

### 2.3 Latency (p95 기준)
- **Fast mode**: p95 < 2.5 seconds
- **Balanced mode**: p95 < 3.2 seconds
- **Creative mode**: p95 < 4.5 seconds
- **Metric**: `llm_brief_latency_seconds_bucket`
- **Query**: `histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode="X"}[5m]))`

### 2.4 RL Performance
- **RL reward**: Stable or increasing trend
- **Metric**: `rl_reward_ma{window="100"}`
- **Trend calculation**: Linear regression slope ≥ 0
- **Time window**: Last 2 hours

---

## 3. No-Go Criteria (적색 신호등)

Any of the following will trigger NO-GO status:

### 3.1 Schema Failure
- **스키마 실패**: > 0%
- **Impact**: Indicates LLM output quality issues
- **Action**: Rollback to previous model version

### 3.2 RL Loss Runaway (폭주)
- **Current loss**: > 2.0 OR
- **Loss increase**: > 3x baseline (2-hour average)
- **Metric**: `rl_loss{loss_type="total_loss"}`
- **Action**: Rollback to stable checkpoint

### 3.3 Creative Mode Sustained High Latency
- **Creative p95**: > 4.5s for 5+ minutes
- **Action**: Downshift to Balanced mode or rollback

### 3.4 VRAM Headroom Critical
- **VRAM free**: < 20%
- **Impact**: Risk of OOM errors
- **Action**: Do not scale up, unload models if needed

---

## 4. Usage

### 4.1 Manual Check

```bash
# Run gate check
python scripts/check_deployment_gate.py

# Save report to file
python scripts/check_deployment_gate.py --output gate_report.json

# Exit with error code on NO_GO (for CI/CD)
python scripts/check_deployment_gate.py --strict
```

**Windows**:
```cmd
cd C:\Users\user\Desktop\새 폴더 (2)\Newss
python scripts\check_deployment_gate.py
```

### 4.2 Output Example

```
================================================================================
DEPLOYMENT GATE CHECK
================================================================================

Overall Status: 🟢 GO

✅ All checks passed - Deployment is GO

================================================================================
CHECK DETAILS
================================================================================

🟢 GO (10 checks)
--------------------------------------------------------------------------------
  • API Error Rate
    Error rate: 0.234% (< 0.5%)
    Value: 0.0023 | Threshold: 0.0050

  • Fast p95 Latency
    p95: 2.12s (< 2.5s)
    Value: 2.1200 | Threshold: 2.5000

  • Balanced p95 Latency
    p95: 2.98s (< 3.2s)
    Value: 2.9800 | Threshold: 3.2000

  • Creative p95 Latency
    p95: 4.23s (< 4.5s)
    Value: 4.2300 | Threshold: 4.5000

  • Schema Failure Rate
    Schema failures: 0.000% (= 0%)
    Value: 0.0000 | Threshold: 0.0000

  • RL Reward Trend
    Reward trend: increasing (current: 12.45)
    Value: 12.4500 | Threshold: 0.0000

  • RL Loss Runaway
    Loss stable: 0.4523
    Value: 0.4523 | Threshold: 2.0000

  • Fast Cache Hit Rate
    Hit rate: 67.2% (≥ 60%)
    Value: 0.6720 | Threshold: 0.6000

  • Balanced Cache Hit Rate
    Hit rate: 64.5% (≥ 60%)
    Value: 0.6450 | Threshold: 0.6000

  • CPU Usage
    CPU: 62.3% (< 85%)
    Value: 0.6230 | Threshold: 0.8500

================================================================================
Timestamp: 2025-10-14T15:30:45.123456Z
================================================================================
```

### 4.3 Exit Codes

| Code | Status | Meaning |
|------|--------|---------|
| 0 | GO | All checks passed |
| 1 | WARNING | Some warnings (non-blocking) |
| 2 | NO-GO | Deployment blocked |

---

## 5. CI/CD Integration

### 5.1 GitHub Actions

```yaml
# .github/workflows/deploy.yml

name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  gate-check:
    name: Go/No-Go Gate Check
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install requests

      - name: Run deployment gate check
        run: |
          python scripts/check_deployment_gate.py \
            --strict \
            --output gate_report.json \
            --prometheus-url ${{ secrets.PROMETHEUS_URL }}

      - name: Upload gate report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: gate-report
          path: gate_report.json

  deploy:
    name: Deploy Application
    needs: gate-check
    runs-on: ubuntu-latest
    if: success()  # Only runs if gate check passes
    steps:
      - name: Deploy to production
        run: |
          # Your deployment commands here
          docker-compose -f docker-compose.production.yml up -d
```

### 5.2 GitLab CI

```yaml
# .gitlab-ci.yml

stages:
  - gate-check
  - deploy

gate-check:
  stage: gate-check
  image: python:3.11
  script:
    - pip install requests
    - python scripts/check_deployment_gate.py --strict --output gate_report.json
  artifacts:
    when: always
    paths:
      - gate_report.json
    reports:
      junit: gate_report.json

deploy-production:
  stage: deploy
  dependencies:
    - gate-check
  only:
    - main
  script:
    - docker-compose -f docker-compose.production.yml up -d
```

### 5.3 Jenkins

```groovy
// Jenkinsfile

pipeline {
    agent any

    stages {
        stage('Gate Check') {
            steps {
                script {
                    def gateStatus = sh(
                        script: 'python scripts/check_deployment_gate.py --strict',
                        returnStatus: true
                    )

                    if (gateStatus == 2) {
                        error('Deployment gate check failed: NO-GO')
                    } else if (gateStatus == 1) {
                        input message: 'Warnings detected. Proceed anyway?', ok: 'Deploy'
                    }
                }
            }
        }

        stage('Deploy') {
            when {
                expression { currentBuild.result == null || currentBuild.result == 'SUCCESS' }
            }
            steps {
                sh 'docker-compose -f docker-compose.production.yml up -d'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'gate_report.json', allowEmptyArchive: true
        }
    }
}
```

---

## 6. Grafana Dashboard Integration

### 6.1 Gate Status Panel

Add this panel to your Grafana dashboard:

```json
{
  "id": 23,
  "title": "Deployment Gate Status",
  "type": "stat",
  "description": "Real-time deployment gate status (GO/WARNING/NO-GO)",
  "targets": [
    {
      "expr": "(llm_schema_fix_count_total / llm_brief_total) <= 0 AND \n(sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))) < 0.005 AND \nhistogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode=\"creative\"}[5m])) < 4.5 AND \nrl_loss{loss_type=\"total_loss\"} < 2.0",
      "legendFormat": "Gate Status"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "mappings": [
        {
          "type": "value",
          "options": {
            "0": {
              "text": "NO-GO 🔴",
              "color": "red"
            },
            "1": {
              "text": "GO 🟢",
              "color": "green"
            }
          }
        }
      ],
      "thresholds": {
        "mode": "absolute",
        "steps": [
          {
            "value": null,
            "color": "red"
          },
          {
            "value": 1,
            "color": "green"
          }
        ]
      }
    }
  },
  "options": {
    "graphMode": "none",
    "textMode": "value_and_name",
    "colorMode": "background"
  }
}
```

### 6.2 Individual Check Panels

```json
{
  "id": 24,
  "title": "Gate Check Details",
  "type": "table",
  "targets": [
    {
      "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))",
      "legendFormat": "API Error Rate"
    },
    {
      "expr": "histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode=\"creative\"}[5m]))",
      "legendFormat": "Creative p95"
    },
    {
      "expr": "rl_loss{loss_type=\"total_loss\"}",
      "legendFormat": "RL Loss"
    },
    {
      "expr": "llm_schema_fix_count_total / llm_brief_total",
      "legendFormat": "Schema Failure Rate"
    }
  ],
  "transformations": [
    {
      "id": "organize",
      "options": {
        "excludeByName": {},
        "indexByName": {},
        "renameByName": {
          "Value": "Current Value"
        }
      }
    }
  ]
}
```

---

## 7. Automated Actions on NO-GO

### 7.1 Downshift Strategy

```python
# fragrance_ai/deployment/gate_actions.py

from fragrance_ai.guards.downshift import get_downshift_manager

def handle_nogo_creative_latency():
    """Handle Creative mode p95 > 4.5s"""
    manager = get_downshift_manager()

    # Downshift from Creative to Balanced
    new_mode = manager.apply_downshift('llm', 'creative', 'balanced')

    logger.warning(f"Deployment gate triggered downshift: creative → {new_mode}")

def handle_nogo_rl_loss_runaway():
    """Handle RL loss runaway"""
    from fragrance_ai.training.checkpoint_manager import get_checkpoint_manager

    manager = get_checkpoint_manager()

    # Find stable checkpoint
    stable_checkpoint = manager.find_stable_checkpoint(
        max_kl_divergence=0.03,
        min_reward=5.0,
        max_reward=30.0
    )

    if stable_checkpoint:
        manager.load_checkpoint(stable_checkpoint.checkpoint_path)
        logger.info(f"Rolled back to stable checkpoint: {stable_checkpoint.checkpoint_path}")
```

### 7.2 Auto-Rollback

```bash
# scripts/auto_rollback_on_nogo.sh

#!/bin/bash

# Run gate check
python scripts/check_deployment_gate.py --strict --output gate_report.json
GATE_STATUS=$?

if [ $GATE_STATUS -eq 2 ]; then
    echo "🔴 NO-GO detected - initiating auto-rollback"

    # Rollback to previous version
    docker-compose -f docker-compose.production.yml down
    docker-compose -f docker-compose.production.yml up -d --scale app=2 --scale worker-llm=2

    # Notify team
    curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
      -H 'Content-Type: application/json' \
      -d '{"text":"🔴 Deployment gate NO-GO: Auto-rollback initiated"}'

    exit 1
else
    echo "✅ Gate check passed - deployment can proceed"
    exit 0
fi
```

---

## 8. Troubleshooting

### 8.1 Connection Issues

**Problem**: "Unable to query Prometheus"

**Solution**:
```bash
# Check Prometheus is running
curl http://localhost:9090/-/healthy

# Check Prometheus is accessible
curl "http://localhost:9090/api/v1/query?query=up"

# Override Prometheus URL
python scripts/check_deployment_gate.py --prometheus-url http://custom-prometheus:9090
```

### 8.2 Missing Metrics

**Problem**: "Unable to query [metric name]"

**Solution**:
1. Verify metric is being collected:
   ```bash
   curl "http://localhost:9090/api/v1/query?query=llm_brief_total"
   ```

2. Check metric collection in code:
   ```python
   from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

   collector = OperationsMetricsCollector()
   collector.record_llm_brief(mode='fast', success=True, latency_seconds=2.0)
   ```

3. Verify Prometheus scrape config:
   ```yaml
   # monitoring/prometheus.yml
   scrape_configs:
     - job_name: 'fragrance-ai'
       static_configs:
         - targets: ['app:8000']
   ```

### 8.3 False NO-GO

**Problem**: Gate incorrectly reports NO-GO

**Solution**:
1. Review individual check details in output
2. Verify thresholds are appropriate for your environment
3. Adjust thresholds in script if needed:
   ```python
   THRESHOLDS = {
       "creative_p95_max": 5.0,  # Increase from 4.5s to 5.0s
       # ... other thresholds
   }
   ```

---

## 9. Best Practices

### 9.1 Pre-Deployment Checklist

1. **Run gate check manually before deployment**
   ```bash
   python scripts/check_deployment_gate.py
   ```

2. **Review gate report if WARNING or NO-GO**
   - Identify root cause of issues
   - Fix issues before attempting deployment
   - Re-run gate check to verify

3. **Monitor during deployment**
   - Keep Grafana dashboard open
   - Watch for metric changes
   - Be ready to rollback if needed

### 9.2 Regular Gate Checks

Run gate checks regularly (not just before deployment):

```bash
# Daily gate check (via cron)
0 9 * * * cd /path/to/project && python scripts/check_deployment_gate.py --output /var/log/gate_check_$(date +\%Y\%m\%d).json
```

### 9.3 Alerting on Persistent NO-GO

Set up alerts if gate remains NO-GO for extended period:

```yaml
# Prometheus alert rule
- alert: DeploymentGateNoGo
  expr: deployment_gate_status == 0
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "Deployment gate has been NO-GO for 30+ minutes"
    action: "Investigate and fix issues preventing deployment"
```

---

## 10. Customization

### 10.1 Add Custom Checks

Extend the gate checker with custom checks:

```python
# In check_deployment_gate.py

def check_custom_metric() -> GateCheck:
    """Check custom business metric"""
    query = 'custom_metric_name'
    value = query_prometheus(query)
    threshold = 100.0

    if value is None:
        return GateCheck(
            name="Custom Metric",
            status=GateStatus.WARNING,
            value=None,
            threshold=threshold,
            message="Unable to query custom metric"
        )

    if value >= threshold:
        return GateCheck(
            name="Custom Metric",
            status=GateStatus.GO,
            value=value,
            threshold=threshold,
            message=f"Custom metric: {value:.2f} (≥ {threshold})"
        )
    else:
        return GateCheck(
            name="Custom Metric",
            status=GateStatus.NO_GO,
            value=value,
            threshold=threshold,
            message=f"Custom metric too low: {value:.2f} (threshold: {threshold})"
        )

# Add to run_all_checks()
def run_all_checks():
    # ... existing checks ...
    all_checks.append(check_custom_metric())
    # ...
```

### 10.2 Adjust Thresholds

Edit `THRESHOLDS` dictionary in `check_deployment_gate.py`:

```python
THRESHOLDS = {
    # Adjust based on your production requirements
    "api_error_rate_max": 0.010,        # 1.0% (more lenient)
    "creative_p95_max": 5.0,            # 5s (more lenient)
    "rl_loss_max": 1.5,                 # 1.5 (stricter)
    # ...
}
```

---

## 11. Summary

### Key Features
- ✅ Automated deployment readiness check
- ✅ Traffic light status (GO/WARNING/NO-GO)
- ✅ 10+ comprehensive checks (API, latency, RL, schema, cache, system)
- ✅ CI/CD integration ready
- ✅ JSON report output
- ✅ Grafana dashboard integration

### Decision Matrix

| Condition | Status | Action |
|-----------|--------|--------|
| All checks pass | 🟢 GO | Deploy |
| Non-critical warnings | 🟡 WARNING | Review → Deploy |
| Critical issues | 🔴 NO-GO | Fix → Downshift/Rollback |

### Quick Commands

```bash
# Check gate status
python scripts/check_deployment_gate.py

# CI/CD mode (exit code)
python scripts/check_deployment_gate.py --strict

# Save report
python scripts/check_deployment_gate.py --output gate_report.json
```

---

*Document Version: 1.0*
*Last Updated: 2025-10-14*
*Status: Production Ready*
