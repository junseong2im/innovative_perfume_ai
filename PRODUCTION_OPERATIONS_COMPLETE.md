# Production Operations - Complete Guide
# 프로덕션 운영 완전 가이드

Complete reference for running Fragrance AI in production with guards, tuning, and monitoring.

---

## ✅ Implementation Complete

프로덕션 운영을 위한 완전한 시스템 구현 완료:

### 1. Deployment Infrastructure
- ✅ **Canary Deployment** (1% → 5% → 25% → 100%)
- ✅ **Traffic Control** (NGINX weighted routing)
- ✅ **Smoke Tests** (API + logs verification)
- ✅ **Pre-Deployment Checks** (T-1 checklist)

### 2. Production Guards ⭐ NEW
- ✅ **JSON Hard Guard** (retry + fallback)
- ✅ **Circuit Breaker** (auto downshift)
- ✅ **Cache LRU+TTL** (10 min expiration)
- ✅ **LLM Health Check** (per-model validation)

### 3. Traffic Tuning (48hr) ⭐ NEW
- ✅ **Traffic Distribution** (fast 60% / balanced 30% / creative 10%)
- ✅ **PPO Schedule** (entropy 0.02 → 0.005, 10k steps)
- ✅ **Reward Normalization** (1k step window)
- ✅ **Checkpoint Strategy** (500 step + auto rollback)

---

## 📋 Quick Reference

### Essential Checks Before Production

```bash
# 1. Pre-Deployment Check (T-1)
python scripts/pre_deployment_check.py --version v0.2.0 --strict

# 2. Production Guards Verification ⭐ NEW
./scripts/verify_production_guards.sh

# 3. Smoke Test
python scripts/smoke_test_api.py

# 4. Canary Deployment
python scripts/canary_deployment.py --version v0.2.0
```

### Post-Deployment Monitoring (First 48 Hours)

```bash
# Monitor traffic distribution
./scripts/monitor_traffic_distribution.sh

# Monitor PPO training
./scripts/monitor_ppo_training.sh

# Check all guards status
./scripts/check_guards_status.sh
```

---

## 🛡️ Production Guards

### 1. JSON Hard Guard

**Purpose**: Protect against LLM JSON parsing failures

**Configuration**: `configs/llm_ensemble.yaml`
```yaml
json_guard:
  enabled: true
  max_retries: 3
  backoff_base: 1.0
  jitter: 0.5
  fallback_enabled: true
```

**Verification**:
```bash
python -c "
from fragrance_ai.config_loader import get_config
config = get_config()
assert config.llm.json_guard_enabled == True
print('✓ JSON Hard Guard: ENABLED')
"
```

**Monitoring**:
```promql
# Fallback usage rate
rate(llm_json_fallback_used_total[5m]) / rate(llm_requests_total[5m])
```

### 2. Circuit Breaker

**Purpose**: Auto downshift on model failures

**Configuration**: `configs/llm_ensemble.yaml`
```yaml
circuit_breaker:
  enabled: true
  failure_threshold: 0.5
  timeout: 60
  auto_downshift: true
```

**Verification**:
```bash
python -c "
from fragrance_ai.llm.circuit_breaker import LLMCircuitBreakerManager
manager = LLMCircuitBreakerManager()
assert all(b.state == 'closed' for b in manager.breakers.values())
print('✓ Circuit Breaker: ALL CLOSED')
"
```

**Monitoring**:
```promql
# Circuit breaker state (0=closed, 1=open)
circuit_breaker_state{model="qwen"}
```

### 3. Cache LRU+TTL

**Purpose**: Memory-efficient caching with freshness guarantee

**Configuration**: `configs/llm_ensemble.yaml`
```yaml
cache:
  enabled: true
  max_size: 1000
  ttl_seconds: 600  # 10 minutes
```

**Verification**:
```bash
# Test TTL expiration
python scripts/test_cache_ttl.py
```

**Monitoring**:
```promql
# Cache hit rate
rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))
```

### 4. LLM Health Check

**Purpose**: Per-model health validation before promotion

**Endpoints**:
```bash
# Check all models
curl "http://localhost:8001/health/llm?model=all"

# Check specific model
curl "http://localhost:8001/health/llm?model=qwen"
curl "http://localhost:8001/health/llm?model=mistral"
curl "http://localhost:8001/health/llm?model=llama"
```

**Pre-Promotion Check**:
```bash
# ALL models must be healthy
HEALTH=$(curl -s "http://localhost:8001/health/llm?model=all")
STATUS=$(echo "$HEALTH" | jq -r '.status')

if [ "$STATUS" == "healthy" ]; then
    echo "✓ Safe to promote"
else
    echo "✗ DO NOT PROMOTE"
    exit 1
fi
```

---

## 🎯 Traffic Tuning (Initial 48 Hours)

### 1. Traffic Distribution

**Target**: fast 60% / balanced 30% / creative 10%

**Configuration**: `configs/traffic_distribution.yaml`
```yaml
traffic_distribution:
  modes:
    fast: 60
    balanced: 30
    creative: 10
```

**Verification**:
```bash
python -c "
from fragrance_ai.routing.traffic_distributor import TrafficDistributor
from fragrance_ai.config_loader import get_config

config = get_config()
distributor = TrafficDistributor(config.traffic_distribution)

actual = distributor.get_current_distribution()
print(f'fast: {actual.get(\"fast\", 0)}% (target: 60%)')
print(f'balanced: {actual.get(\"balanced\", 0)}% (target: 30%)')
print(f'creative: {actual.get(\"creative\", 0)}% (target: 10%)')
"
```

### 2. PPO Schedule

**Target**: Entropy 0.02 → 0.005 over 10k steps (linear)

**Configuration**: `configs/ppo_schedule.yaml`
```yaml
ppo_schedule:
  entropy_coef:
    initial: 0.02
    final: 0.005
    decay_steps: 10000
    decay_strategy: linear
```

**Monitoring**:
```promql
# Current entropy
ppo_entropy_coef

# Training progress
ppo_training_step
```

### 3. Reward Normalization

**Target**: 1k step rolling window

**Configuration**: `configs/reward_normalization.yaml`
```yaml
reward_normalization:
  enabled: true
  window_size: 1000
```

**Monitoring**:
```promql
# Normalization stats
reward_normalizer_mean
reward_normalizer_std
```

### 4. Checkpoint Strategy

**Target**: Save every 500 steps, rollback on spike

**Configuration**: `configs/checkpoint_strategy.yaml`
```yaml
checkpoint_strategy:
  save_frequency: 500
  rollback:
    enabled: true
    loss_spike:
      threshold: 2.0
    reward_drop:
      threshold: 0.5
```

**Monitoring**:
```promql
# Checkpoint saves
rate(checkpoint_saves_total[1h])

# Rollbacks
rate(checkpoint_rollbacks_total[1h])
```

---

## 📊 Complete Deployment Workflow

### Pre-Deployment (T-1)

```bash
# Step 1: Create release
./scripts/create_release.sh v0.2.0

# Step 2: Pre-deployment check
python scripts/pre_deployment_check.py --version v0.2.0 --strict

# Step 3: Verify production guards ⭐ NEW
./scripts/verify_production_guards.sh

# Step 4: Build images
export CANARY_VERSION=v0.2.0
docker-compose -f docker-compose.canary.yml build app-canary
```

### Deployment

```bash
# Step 1: Start canary
docker-compose -f docker-compose.production.yml -f docker-compose.canary.yml up -d app-canary

# Step 2: Smoke test ⭐ NEW
python scripts/smoke_test_api.py --canary

# Step 3: Verify guards ⭐ NEW
./scripts/verify_production_guards.sh --container fragrance-ai-app-canary

# Step 4: LLM health check ⭐ NEW
curl "http://localhost:8001/health/llm?model=all"

# Step 5: Canary deployment
python scripts/canary_deployment.py --version v0.2.0
```

### Post-Deployment (First 48 Hours) ⭐ NEW

```bash
# Hour 0-2: Intensive monitoring
watch -n 60 './scripts/monitor_traffic_distribution.sh'
watch -n 300 './scripts/check_guards_status.sh'

# Hour 2-24: Regular monitoring
# - Check traffic distribution every 6 hours
# - Check PPO training progress every 4 hours
# - Monitor for rollbacks

# Hour 24-48: Stabilization
# - Traffic distribution stable (< 5% deviation)
# - PPO entropy converged (~0.005)
# - 0-1 rollbacks total
# - All guards operational
```

### Promotion

```bash
# Step 1: Verify all guards ⭐ NEW
./scripts/verify_production_guards.sh --container fragrance-ai-app-canary

# Step 2: LLM health check ⭐ NEW
curl "http://localhost:8001/health/llm?model=all"

# Step 3: Check traffic tuning ⭐ NEW
./scripts/check_traffic_tuning_complete.sh

# Step 4: Promote
export VERSION=${CANARY_VERSION}
docker-compose -f docker-compose.production.yml up -d app

# Step 5: Clean up
docker-compose -f docker-compose.canary.yml down
```

---

## 🔧 Verification Scripts

### Verify Production Guards

**파일**: `scripts/verify_production_guards.sh`

```bash
#!/bin/bash
# Verify all production guards are enabled

CONTAINER="${1:-fragrance-ai-app}"

echo "Verifying Production Guards..."
echo "Container: $CONTAINER"
echo "======================================"

# 1. JSON Hard Guard
echo "1. JSON Hard Guard"
docker exec $CONTAINER python -c "
from fragrance_ai.config_loader import get_config
config = get_config()
assert config.llm.json_guard_enabled == True
assert config.llm.max_retries >= 3
print('✓ ENABLED (retries: %d)' % config.llm.max_retries)
"

# 2. Circuit Breaker
echo "2. Circuit Breaker"
docker exec $CONTAINER python -c "
from fragrance_ai.llm.circuit_breaker import LLMCircuitBreakerManager
manager = LLMCircuitBreakerManager()
states = {m: b.state for m, b in manager.breakers.items()}
assert all(s == 'closed' for s in states.values())
print('✓ ALL CLOSED: %s' % states)
"

# 3. Cache LRU+TTL
echo "3. Cache LRU+TTL"
docker exec $CONTAINER python -c "
from fragrance_ai.cache.lru_cache import LRUCacheWithTTL
from fragrance_ai.config_loader import get_config
config = get_config()
assert config.cache.enabled == True
assert config.cache.ttl_seconds >= 600
print('✓ ENABLED (TTL: %ds)' % config.cache.ttl_seconds)
"

# 4. LLM Health Check
echo "4. LLM Health Check"
HEALTH=$(curl -s "http://localhost:8001/health/llm?model=all")
STATUS=$(echo "$HEALTH" | jq -r '.status')

if [ "$STATUS" == "healthy" ]; then
    echo "✓ ALL MODELS HEALTHY"
else
    echo "✗ MODELS UNHEALTHY:"
    echo "$HEALTH" | jq .
    exit 1
fi

echo "======================================"
echo "✓ ALL GUARDS VERIFIED"
```

### Monitor Traffic Distribution

**파일**: `scripts/monitor_traffic_distribution.sh`

```bash
#!/bin/bash
# Monitor actual traffic distribution vs target

echo "Traffic Distribution Monitoring"
echo "======================================"

docker exec fragrance-ai-app python -c "
from fragrance_ai.routing.traffic_distributor import TrafficDistributor
from fragrance_ai.config_loader import get_config

config = get_config()
distributor = TrafficDistributor(config.traffic_distribution)

# Target
target = distributor.mode_weights
print('Target:')
print(f'  fast: {target[\"fast\"]}%')
print(f'  balanced: {target[\"balanced\"]}%')
print(f'  creative: {target[\"creative\"]}%')
print()

# Actual
actual = distributor.get_current_distribution()
print('Actual (last 5 min):')
print(f'  fast: {actual.get(\"fast\", 0)}%')
print(f'  balanced: {actual.get(\"balanced\", 0)}%')
print(f'  creative: {actual.get(\"creative\", 0)}%')
print()

# Deviations
deviations = distributor.check_distribution_deviation()
print('Status:')
for mode, data in deviations.items():
    status = '⚠ ALERT' if data['alert'] else '✓ OK'
    print(f'  {mode}: {data[\"deviation\"]}% deviation {status}')
"

echo "======================================"
```

### Check Traffic Tuning Complete

**파일**: `scripts/check_traffic_tuning_complete.sh`

```bash
#!/bin/bash
# Check if 48-hour traffic tuning is complete

echo "Checking Traffic Tuning Completion..."
echo "======================================"

PASS=0
FAIL=0

# 1. Traffic distribution stable
echo "1. Traffic Distribution"
docker exec fragrance-ai-app python -c "
from fragrance_ai.routing.traffic_distributor import TrafficDistributor
from fragrance_ai.config_loader import get_config

config = get_config()
distributor = TrafficDistributor(config.traffic_distribution)
deviations = distributor.check_distribution_deviation()

all_ok = all(not d['alert'] for d in deviations.values())
if all_ok:
    print('✓ STABLE (< 10% deviation)')
    exit(0)
else:
    print('✗ UNSTABLE (> 10% deviation)')
    exit(1)
" && ((PASS++)) || ((FAIL++))

# 2. PPO entropy converged
echo "2. PPO Entropy Convergence"
docker exec fragrance-ai-app python -c "
from fragrance_ai.monitoring.metrics import MetricsCollector

collector = MetricsCollector()
entropy = collector.get_metric('ppo_entropy_coef')
step = collector.get_metric('ppo_training_step')

if step >= 10000 and entropy <= 0.006:
    print(f'✓ CONVERGED (step={step}, entropy={entropy:.4f})')
    exit(0)
else:
    print(f'✗ NOT CONVERGED (step={step}, entropy={entropy:.4f})')
    exit(1)
" && ((PASS++)) || ((FAIL++))

# 3. Checkpoint stability (< 2 rollbacks)
echo "3. Checkpoint Stability"
docker exec fragrance-ai-app python -c "
from fragrance_ai.training.checkpoint_manager import CheckpointManager

manager = CheckpointManager.get_instance()

if manager.rollback_count <= 1:
    print(f'✓ STABLE ({manager.rollback_count} rollbacks)')
    exit(0)
else:
    print(f'✗ UNSTABLE ({manager.rollback_count} rollbacks)')
    exit(1)
" && ((PASS++)) || ((FAIL++))

echo "======================================"
echo "Results: $PASS/3 checks passed"

if [ $FAIL -eq 0 ]; then
    echo "✓ TRAFFIC TUNING COMPLETE"
    exit 0
else
    echo "✗ TRAFFIC TUNING NOT COMPLETE"
    exit 1
fi
```

---

## 📖 Documentation Index

### Core Documentation
- **CANARY_DEPLOYMENT_SUMMARY.md** - Canary deployment overview
- **docs/CANARY_DEPLOYMENT_GUIDE.md** - Detailed deployment guide
- **SMOKE_TEST_GUIDE.md** - Smoke test procedures

### New Documentation ⭐
- **PRODUCTION_GUARDS_CHECKLIST.md** - Guards verification
- **MIXED_TRAFFIC_TUNING_GUIDE.md** - 48-hour tuning guide
- **PRODUCTION_OPERATIONS_COMPLETE.md** - This document

### Infrastructure
- **PRE_DEPLOYMENT_CHECKLIST_SUMMARY.md** - T-1 checklist
- **RUNBOOK.md** - Operations runbook
- **DEPLOYMENT_GUIDE.md** - General deployment

---

## 🎯 Success Criteria

### Deployment Success

- ✅ All pre-deployment checks passed
- ✅ All production guards enabled and operational
- ✅ Smoke tests passed
- ✅ Canary deployment completed (1% → 100%)
- ✅ LLM health check: all models healthy
- ✅ No critical errors in logs

### 48-Hour Stability

- ✅ Traffic distribution: < 5% deviation from target
- ✅ PPO entropy: converged to ~0.005
- ✅ Checkpoint rollbacks: ≤ 1
- ✅ Error rate: < 0.5%
- ✅ Latency: within thresholds
- ✅ All circuit breakers: closed
- ✅ Cache hit rate: > 70%

---

## 🚨 Emergency Procedures

### Circuit Breaker Open

```bash
# 1. Check which model
curl "http://localhost:8000/health/llm?model=all"

# 2. Check logs
docker logs fragrance-ai-app | grep "circuit_breaker"

# 3. Manual downshift if needed
curl -X POST "http://localhost:8000/admin/set-mode" \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast"}'
```

### High Fallback Rate (> 1%)

```bash
# 1. Check LLM outputs
docker logs fragrance-ai-app | grep "JSON parse error"

# 2. Check schema validation
docker logs fragrance-ai-app | grep "Schema validation failed"

# 3. Review prompts if needed
# Edit prompts in configs/prompts.yaml
```

### Frequent Rollbacks (> 2 in 48hr)

```bash
# 1. Check training logs
docker logs fragrance-ai-worker-rl | grep "rollback"

# 2. Check loss/reward history
curl "http://localhost:9090/api/v1/query?query=ppo_loss"
curl "http://localhost:9090/api/v1/query?query=ppo_reward"

# 3. Pause training if unstable
docker-compose stop worker-rl
```

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Status**: Complete ✅
**Maintained by**: DevOps & ML Engineering Team
