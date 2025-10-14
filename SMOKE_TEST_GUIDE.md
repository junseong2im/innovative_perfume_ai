# Smoke Test Guide
# 런치 직후 5분 수동 검증

Quick reference for manual verification after deployment.

---

## Overview

Smoke tests verify that the deployed system is functioning correctly by:
1. Testing critical API endpoints
2. Verifying required logs are present
3. Checking Prometheus metrics

**Duration**: ~5 minutes

---

## Quick Start

### Automated Smoke Test (Python)

```bash
# Test production
python scripts/smoke_test_api.py

# Test canary
python scripts/smoke_test_api.py --canary

# Custom endpoint
python scripts/smoke_test_api.py --base-url http://localhost:8001 --container fragrance-ai-app-canary
```

### Manual Smoke Test (Bash)

```bash
# Test production
./scripts/smoke_test_manual.sh http://localhost:8000

# Test canary
./scripts/smoke_test_manual.sh http://localhost:8001 fragrance-ai-app-canary
```

---

## Test Steps

### 1. Health Check

```bash
curl -s http://localhost:8000/health | jq .
```

**Expected**:
```json
{
  "status": "healthy",
  "version": "0.2.0",
  "services": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

### 2. Create DNA (/dna/create)

```bash
curl -s http://localhost:8000/dna/create \
  -H 'Content-Type: application/json' \
  -d '{
    "brief": {
      "mood": "상큼함",
      "season": ["spring"]
    }
  }' | jq .
```

**Expected**:
```json
{
  "dna_id": "dna_abc123...",
  "brief": {
    "mood": "상큼함",
    "season": ["spring"]
  },
  "created_at": "2025-10-14T12:00:00Z"
}
```

**Extract DNA ID**:
```bash
DNA_ID=$(curl -s http://localhost:8000/dna/create \
  -H 'Content-Type: application/json' \
  -d '{"brief":{"mood":"상큼함","season":["spring"]}}' | jq -r '.dna_id')

echo "DNA ID: $DNA_ID"
```

### 3. Generate Evolution Options (/evolve/options)

```bash
curl -s http://localhost:8000/evolve/options \
  -H 'Content-Type: application/json' \
  -d '{
    "dna_id": "'$DNA_ID'",
    "algorithm": "PPO",
    "num_options": 3,
    "mode": "creative"
  }' | jq .
```

**Expected**:
```json
{
  "experiment_id": "exp_xyz789...",
  "options": [
    {
      "id": "opt_001",
      "formulation": [...],
      "predicted_rating": 4.5
    },
    {
      "id": "opt_002",
      "formulation": [...],
      "predicted_rating": 4.3
    },
    {
      "id": "opt_003",
      "formulation": [...],
      "predicted_rating": 4.1
    }
  ]
}
```

**Extract IDs**:
```bash
RESPONSE=$(curl -s http://localhost:8000/evolve/options \
  -H 'Content-Type: application/json' \
  -d '{"dna_id":"'$DNA_ID'","algorithm":"PPO","num_options":3,"mode":"creative"}')

EXPERIMENT_ID=$(echo "$RESPONSE" | jq -r '.experiment_id')
OPTION_ID=$(echo "$RESPONSE" | jq -r '.options[0].id')

echo "Experiment ID: $EXPERIMENT_ID"
echo "Option ID: $OPTION_ID"
```

### 4. Submit Feedback (/evolve/feedback)

```bash
curl -s http://localhost:8000/evolve/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "experiment_id": "'$EXPERIMENT_ID'",
    "chosen_id": "'$OPTION_ID'",
    "rating": 5
  }' | jq .
```

**Expected**:
```json
{
  "success": true,
  "experiment_id": "exp_xyz789...",
  "feedback_recorded": true,
  "training_triggered": true
}
```

---

## Log Verification

### Check llm_brief Logs

**Required fields**: `mode`, `elapsed_ms`

```bash
# Check Docker logs
docker logs fragrance-ai-app --since 5m | grep llm_brief

# Expected output:
# llm_brief{mode="creative",model="gpt-3.5-turbo",elapsed_ms=1850} Generated brief: ...
# llm_brief{mode="fast",model="claude-3-haiku",elapsed_ms=980} Generated brief: ...
```

**What to look for**:
- ✅ `mode` field present (fast/balanced/creative)
- ✅ `elapsed_ms` or `duration` or `latency` field present
- ✅ Model name present
- ✅ Elapsed time reasonable (< 5000ms)

### Check rl_update Logs

**Required fields**: `algo`, `loss`, `reward`, `entropy`, `clip_frac`

```bash
# Check Docker logs
docker logs fragrance-ai-app --since 5m | grep -E 'rl_update|ppo_update'

# Expected output:
# rl_update{algo="PPO",loss=0.0234,reward=4.5,entropy=0.123,clip_frac=0.15} Training step 1000
# rl_update{algo="PPO",loss=0.0198,reward=4.7,entropy=0.115,clip_frac=0.12} Training step 1001
```

**What to look for**:
- ✅ `algo` field present (PPO/GA/MOGA)
- ✅ `loss` value present and reasonable (< 1.0)
- ✅ `reward` value present
- ✅ `entropy` value present (0-1 range)
- ✅ `clip_frac` value present (PPO specific)

**Note**: RL logs may not appear immediately if:
- Training batch not completed yet
- Async worker processing
- Insufficient feedback collected

---

## Metrics Verification

### Check /metrics Endpoint

```bash
curl -s http://localhost:8000/metrics | grep llm_brief
curl -s http://localhost:8000/metrics | grep rl_update
```

**Expected llm_brief metrics**:
```
# HELP llm_brief_duration_seconds LLM brief generation duration
# TYPE llm_brief_duration_seconds histogram
llm_brief_duration_seconds_bucket{mode="fast",le="1.0"} 45
llm_brief_duration_seconds_bucket{mode="fast",le="2.5"} 89
llm_brief_duration_seconds_sum{mode="fast"} 125.3
llm_brief_duration_seconds_count{mode="fast"} 100

# HELP llm_brief_total Total LLM brief generations
# TYPE llm_brief_total counter
llm_brief_total{mode="fast",status="success"} 95
llm_brief_total{mode="fast",status="error"} 5
```

**Expected rl_update metrics**:
```
# HELP rl_training_loss RL training loss
# TYPE rl_training_loss gauge
rl_training_loss{algo="PPO"} 0.0234

# HELP rl_training_reward RL training reward
# TYPE rl_training_reward gauge
rl_training_reward{algo="PPO"} 4.5

# HELP rl_training_entropy RL training entropy
# TYPE rl_training_entropy gauge
rl_training_entropy{algo="PPO"} 0.123
```

---

## Complete Flow (Bash Script)

```bash
#!/bin/bash
# Complete smoke test flow

BASE_URL="http://localhost:8000"

echo "1. Health check..."
curl -s $BASE_URL/health | jq .
echo ""

echo "2. Creating DNA..."
DNA_RESPONSE=$(curl -s $BASE_URL/dna/create \
  -H 'Content-Type: application/json' \
  -d '{"brief":{"mood":"상큼함","season":["spring"]}}')
echo "$DNA_RESPONSE" | jq .
DNA_ID=$(echo "$DNA_RESPONSE" | jq -r '.dna_id')
echo "DNA ID: $DNA_ID"
echo ""

sleep 2

echo "3. Generating evolution options..."
EVOLVE_RESPONSE=$(curl -s $BASE_URL/evolve/options \
  -H 'Content-Type: application/json' \
  -d '{"dna_id":"'$DNA_ID'","algorithm":"PPO","num_options":3,"mode":"creative"}')
echo "$EVOLVE_RESPONSE" | jq .
EXPERIMENT_ID=$(echo "$EVOLVE_RESPONSE" | jq -r '.experiment_id')
OPTION_ID=$(echo "$EVOLVE_RESPONSE" | jq -r '.options[0].id')
echo "Experiment ID: $EXPERIMENT_ID"
echo "Option ID: $OPTION_ID"
echo ""

sleep 2

echo "4. Submitting feedback..."
FEEDBACK_RESPONSE=$(curl -s $BASE_URL/evolve/feedback \
  -H 'Content-Type: application/json' \
  -d '{"experiment_id":"'$EXPERIMENT_ID'","chosen_id":"'$OPTION_ID'","rating":5}')
echo "$FEEDBACK_RESPONSE" | jq .
echo ""

sleep 3

echo "5. Checking llm_brief logs..."
docker logs fragrance-ai-app --since 5m 2>&1 | grep llm_brief | tail -5
echo ""

echo "6. Checking rl_update logs..."
docker logs fragrance-ai-app --since 5m 2>&1 | grep -E 'rl_update|ppo_update' | tail -5
echo ""

echo "7. Checking metrics..."
echo "llm_brief metrics:"
curl -s $BASE_URL/metrics | grep llm_brief | head -10
echo ""
echo "rl_update metrics:"
curl -s $BASE_URL/metrics | grep rl_update | head -10
echo ""

echo "Smoke test complete!"
```

---

## Integration with Deployment

### During Canary Deployment

Add smoke test to canary deployment workflow:

```python
# In scripts/canary_deployment.py

def validate_canary_stage(percentage: int):
    """Validate canary stage with smoke tests"""

    # Wait for traffic to stabilize
    time.sleep(30)

    # Run smoke tests
    result = subprocess.run(
        ['python', 'scripts/smoke_test_api.py', '--canary'],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        logger.error("Smoke test failed")
        return False

    # Continue with metrics validation
    # ...
```

### Pre-Deployment Checklist

Add to `PRE_DEPLOYMENT_CHECKLIST_SUMMARY.md`:

```markdown
**9. Smoke Test** ✓
- [ ] Smoke test passes in staging
- [ ] All API endpoints responding
- [ ] Required logs present (llm_brief, rl_update)
- [ ] Metrics endpoint accessible
```

### Post-Deployment

Run smoke test immediately after deployment:

```bash
# After deployment
docker-compose up -d

# Wait for services to start
sleep 30

# Run smoke test
python scripts/smoke_test_api.py

# If passed, proceed with monitoring
# If failed, rollback immediately
```

---

## Troubleshooting

### llm_brief Logs Not Found

**Possible causes**:
1. LLM ensemble not enabled
2. No requests processed yet
3. Logging level too high

**Solutions**:
```bash
# Check environment variable
docker exec fragrance-ai-app env | grep USE_LLM_ENSEMBLE

# Check log level
docker exec fragrance-ai-app env | grep LOG_LEVEL

# Send test request
curl -X POST http://localhost:8000/dna/create \
  -H 'Content-Type: application/json' \
  -d '{"brief":{"mood":"test"}}'

# Check logs again
docker logs fragrance-ai-app --since 1m | grep llm_brief
```

### rl_update Logs Not Found

**Possible causes**:
1. RL training not triggered (requires feedback)
2. Training batch not completed
3. Worker not running

**Solutions**:
```bash
# Check if RL workers are running
docker ps | grep worker-rl

# Check Redis queue
docker exec fragrance-ai-redis redis-cli LLEN celery

# Submit feedback to trigger training
curl -X POST http://localhost:8000/evolve/feedback \
  -H 'Content-Type: application/json' \
  -d '{"experiment_id":"exp_test","chosen_id":"opt_1","rating":5}'

# Check worker logs
docker logs fragrance-ai-worker-rl --since 5m
```

### API Endpoints Not Responding

**Possible causes**:
1. Service not started
2. Port not exposed
3. Network issues

**Solutions**:
```bash
# Check service status
docker ps | grep fragrance-ai-app

# Check logs
docker logs fragrance-ai-app --tail 50

# Check port mapping
docker port fragrance-ai-app

# Test from inside container
docker exec fragrance-ai-app curl -s http://localhost:8000/health
```

---

## Success Criteria

Smoke test is successful when:

- ✅ Health check returns 200 OK
- ✅ /dna/create creates DNA with ID
- ✅ /evolve/options generates 3 options
- ✅ /evolve/feedback records feedback
- ✅ llm_brief logs present with mode and elapsed_ms
- ✅ rl_update logs present (or expected to be absent)
- ✅ /metrics endpoint contains llm_brief metrics
- ✅ No critical errors in logs

---

## Related Documentation

- **Canary Deployment Guide**: `docs/CANARY_DEPLOYMENT_GUIDE.md`
- **Pre-Deployment Checklist**: `PRE_DEPLOYMENT_CHECKLIST_SUMMARY.md`
- **Runbook**: `RUNBOOK.md`
- **API Documentation**: http://localhost:8000/docs

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Maintained by**: DevOps Team
