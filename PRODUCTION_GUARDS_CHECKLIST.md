# Production Guards Checklist
# 프로덕션 보호장치 확인 목록

Essential guards and safeguards that MUST be enabled in production.

---

## ✅ Overview

프로덕션 환경에서 반드시 활성화되어야 하는 4가지 핵심 보호장치:

1. ✅ **JSON Hard Guard** - JSON 파싱 실패 시 재시도 + 폴백
2. ✅ **Circuit Breaker** - LLM 모델 실패 시 자동 다운시프트
3. ✅ **Cache LRU+TTL** - 캐시 만료 및 재생성 동작 확인
4. ✅ **LLM Health Check** - 모델별 헬스체크 통과 확인

---

## 1. JSON Hard Guard + Retry + Fallback

### Purpose

LLM 출력이 JSON 스키마를 위반하거나 파싱 실패 시 시스템 보호.

### Configuration

**파일**: `fragrance_ai/llm/guards.py`

```python
class JSONHardGuard:
    """JSON schema validation with retry and fallback"""

    def __init__(self, max_retries: int = 3, backoff_base: float = 1.0):
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.schema_validator = SchemaValidator()

    def parse_with_retry(
        self,
        llm_output: str,
        schema: dict,
        fallback: Optional[dict] = None
    ) -> dict:
        """Parse JSON with exponential backoff + jitter"""

        for attempt in range(self.max_retries):
            try:
                # Parse JSON
                parsed = json.loads(llm_output)

                # Validate schema
                if self.schema_validator.validate(parsed, schema):
                    return parsed
                else:
                    logger.warning(f"Schema validation failed (attempt {attempt+1})")

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error (attempt {attempt+1}): {e}")

            # Exponential backoff with jitter
            if attempt < self.max_retries - 1:
                jitter = random.uniform(0, 0.5)
                sleep_time = (self.backoff_base * (2 ** attempt)) + jitter
                time.sleep(sleep_time)

        # All retries failed - use fallback
        logger.error("All JSON parse attempts failed, using fallback")
        return fallback or self._get_default_fallback()

    def _get_default_fallback(self) -> dict:
        """Default fallback for llm_brief"""
        return {
            "description": "향수 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
            "notes": ["floral", "fresh"],
            "families": ["floral"],
            "intensity": 50,
            "longevity": "medium",
            "error": True
        }
```

### Verification

```bash
# Check if guard is enabled
python -c "
from fragrance_ai.config_loader import get_config
config = get_config()
assert config.llm.json_guard_enabled == True
assert config.llm.max_retries >= 3
print('✓ JSON Hard Guard: ENABLED')
print(f'  Max retries: {config.llm.max_retries}')
print(f'  Backoff base: {config.llm.backoff_base}s')
print(f'  Fallback enabled: {config.llm.fallback_enabled}')
"
```

### Expected Configuration

```yaml
# configs/llm_ensemble.yaml
json_guard:
  enabled: true
  max_retries: 3
  backoff_base: 1.0  # seconds
  jitter: 0.5        # max jitter in seconds
  fallback_enabled: true
  strict_schema: true
```

### Monitoring

**Prometheus Metrics**:
```promql
# JSON parse failures
rate(llm_json_parse_failures_total[5m])

# Retry count
histogram_quantile(0.95, rate(llm_json_retry_count_bucket[5m]))

# Fallback usage
rate(llm_json_fallback_used_total[5m])
```

**Alert Rules**:
```yaml
# Alert if fallback used > 1% of requests
- alert: HighJSONFallbackRate
  expr: rate(llm_json_fallback_used_total[5m]) / rate(llm_requests_total[5m]) > 0.01
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High JSON fallback rate (>1%)"
```

---

## 2. Circuit Breaker + Auto Downshift

### Purpose

LLM 모델 실패율 증가 시 자동으로 더 안정적인 모드로 다운시프트.

### Configuration

**파일**: `fragrance_ai/llm/circuit_breaker.py`

```python
class CircuitBreaker:
    """Circuit breaker with automatic mode downshift"""

    def __init__(
        self,
        failure_threshold: float = 0.5,  # 50% failure rate
        success_threshold: int = 5,      # 5 consecutive successes to recover
        timeout: int = 60,                # 60s timeout
        half_open_requests: int = 3       # 3 requests in half-open state
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_requests = half_open_requests

        self.state = "closed"  # closed, open, half_open
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.request_count = 0

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker"""

        if self.state == "open":
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.timeout:
                logger.info("Circuit breaker: Transitioning to half-open")
                self.state = "half_open"
                self.success_count = 0
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful request"""
        self.failure_count = 0

        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info("Circuit breaker: Recovering to closed state")
                self.state = "closed"
                self.success_count = 0

    def _on_failure(self):
        """Handle failed request"""
        self.failure_count += 1
        self.request_count += 1
        self.last_failure_time = time.time()

        failure_rate = self.failure_count / max(self.request_count, 1)

        if failure_rate > self.failure_threshold:
            logger.warning(f"Circuit breaker: Opening (failure rate: {failure_rate:.2%})")
            self.state = "open"

            # Trigger auto downshift
            self._trigger_downshift()

    def _trigger_downshift(self):
        """Automatically downshift to more stable mode"""
        # creative → balanced → fast
        from fragrance_ai.llm.mode_controller import ModeController

        controller = ModeController()
        current_mode = controller.get_current_mode()

        downshift_map = {
            "creative": "balanced",
            "balanced": "fast",
            "fast": "fast"  # Already at lowest
        }

        new_mode = downshift_map.get(current_mode, "fast")

        if new_mode != current_mode:
            logger.warning(f"Auto downshift: {current_mode} → {new_mode}")
            controller.set_mode(new_mode)

            # Send alert
            self._send_downshift_alert(current_mode, new_mode)


class LLMCircuitBreakerManager:
    """Manage circuit breakers for each LLM model"""

    def __init__(self):
        self.breakers = {
            "qwen": CircuitBreaker(),
            "mistral": CircuitBreaker(),
            "llama": CircuitBreaker()
        }

    def get_breaker(self, model: str) -> CircuitBreaker:
        return self.breakers.get(model)

    def is_available(self, model: str) -> bool:
        """Check if model is available (circuit not open)"""
        breaker = self.get_breaker(model)
        return breaker.state != "open" if breaker else False

    def get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        return [
            model
            for model, breaker in self.breakers.items()
            if breaker.state != "open"
        ]
```

### Verification

```bash
# Check circuit breaker configuration
python -c "
from fragrance_ai.llm.circuit_breaker import LLMCircuitBreakerManager

manager = LLMCircuitBreakerManager()
print('✓ Circuit Breaker: ENABLED')
print(f'  Models monitored: {list(manager.breakers.keys())}')

for model, breaker in manager.breakers.items():
    print(f'  {model}:')
    print(f'    Failure threshold: {breaker.failure_threshold:.0%}')
    print(f'    Timeout: {breaker.timeout}s')
    print(f'    State: {breaker.state}')
"
```

### Expected Configuration

```yaml
# configs/llm_ensemble.yaml
circuit_breaker:
  enabled: true
  failure_threshold: 0.5  # 50%
  success_threshold: 5    # 5 consecutive successes
  timeout: 60             # seconds
  half_open_requests: 3
  auto_downshift: true

  # Per-model overrides
  models:
    qwen:
      failure_threshold: 0.4  # More sensitive for Qwen
    mistral:
      failure_threshold: 0.5
    llama:
      failure_threshold: 0.5
```

### Monitoring

**Prometheus Metrics**:
```promql
# Circuit breaker state (0=closed, 1=open, 2=half_open)
circuit_breaker_state{model="qwen"}

# Failure rate by model
rate(llm_request_failures_total{model="qwen"}[5m]) / rate(llm_requests_total{model="qwen"}[5m])

# Downshift events
rate(llm_mode_downshift_total[1h])
```

**Alert Rules**:
```yaml
# Alert if Qwen circuit breaker opens
- alert: QwenCircuitBreakerOpen
  expr: circuit_breaker_state{model="qwen"} == 1
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Qwen circuit breaker is OPEN"
    description: "Qwen model has high failure rate, auto downshift triggered"
```

---

## 3. Cache LRU + TTL

### Purpose

LRU (Least Recently Used) 캐시로 메모리 제한, TTL (Time To Live)로 신선도 보장.

### Configuration

**파일**: `fragrance_ai/cache/lru_cache.py`

```python
class LRUCacheWithTTL:
    """LRU cache with TTL expiration"""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 600,  # 10 minutes
        cleanup_interval: int = 60
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval

        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.Lock()

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            # Check if key exists
            if key not in self.cache:
                return None

            # Check if expired
            if self._is_expired(key):
                logger.debug(f"Cache miss (expired): {key}")
                del self.cache[key]
                del self.timestamps[key]
                return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)

            logger.debug(f"Cache hit: {key}")
            return self.cache[key]

    def set(self, key: str, value: Any):
        """Set value in cache"""
        with self.lock:
            # Update or add
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                # Check if cache is full
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    logger.debug(f"Cache eviction (LRU): {oldest_key}")
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]

            self.cache[key] = value
            self.timestamps[key] = time.time()
            logger.debug(f"Cache set: {key}")

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        timestamp = self.timestamps.get(key)
        if timestamp is None:
            return True

        age = time.time() - timestamp
        return age > self.ttl_seconds

    def _cleanup_loop(self):
        """Periodically clean up expired entries"""
        while True:
            time.sleep(self.cleanup_interval)
            self._cleanup_expired()

    def _cleanup_expired(self):
        """Remove all expired entries"""
        with self.lock:
            expired_keys = [
                key for key in self.cache.keys()
                if self._is_expired(key)
            ]

            for key in expired_keys:
                logger.debug(f"Cache cleanup (expired): {key}")
                del self.cache[key]
                del self.timestamps[key]

            if expired_keys:
                logger.info(f"Cache cleanup: Removed {len(expired_keys)} expired entries")

    def stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "ttl_seconds": self.ttl_seconds
            }
```

### Verification

```bash
# Check cache configuration
python -c "
from fragrance_ai.cache.lru_cache import LRUCacheWithTTL
from fragrance_ai.config_loader import get_config

config = get_config()
cache = LRUCacheWithTTL(
    max_size=config.cache.max_size,
    ttl_seconds=config.cache.ttl_seconds
)

stats = cache.stats()
print('✓ LRU Cache: ENABLED')
print(f'  Max size: {stats[\"max_size\"]}')
print(f'  TTL: {stats[\"ttl_seconds\"]}s ({stats[\"ttl_seconds\"]//60} minutes)')
print(f'  Current size: {stats[\"size\"]}')
print(f'  Utilization: {stats[\"utilization\"]:.1%}')
"

# Test TTL expiration
python scripts/test_cache_ttl.py
```

### Expected Configuration

```yaml
# configs/llm_ensemble.yaml
cache:
  enabled: true
  type: lru_ttl
  max_size: 1000          # Maximum entries
  ttl_seconds: 600        # 10 minutes
  cleanup_interval: 60    # Cleanup every 60s
  eviction_policy: lru    # Least Recently Used
```

### Test TTL Expiration

**파일**: `scripts/test_cache_ttl.py`

```python
#!/usr/bin/env python3
"""Test cache TTL expiration and regeneration"""

import time
from fragrance_ai.cache.lru_cache import LRUCacheWithTTL

def test_ttl_expiration():
    """Test that cache entries expire after TTL"""

    # Create cache with short TTL for testing
    cache = LRUCacheWithTTL(max_size=100, ttl_seconds=5)

    # Set value
    cache.set("test_key", "test_value")
    print("✓ Set cache entry")

    # Get immediately
    value = cache.get("test_key")
    assert value == "test_value"
    print("✓ Cache hit (fresh)")

    # Wait for TTL to expire
    print("  Waiting 6 seconds for TTL expiration...")
    time.sleep(6)

    # Try to get expired entry
    value = cache.get("test_key")
    assert value is None
    print("✓ Cache miss (expired)")

    # Verify regeneration works
    cache.set("test_key", "new_value")
    value = cache.get("test_key")
    assert value == "new_value"
    print("✓ Cache regeneration successful")

    print("\n✓ All TTL tests passed")

if __name__ == '__main__':
    test_ttl_expiration()
```

### Monitoring

**Prometheus Metrics**:
```promql
# Cache hit rate
rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

# Cache size
cache_size

# Cache evictions
rate(cache_evictions_total[5m])

# TTL expirations
rate(cache_ttl_expirations_total[5m])
```

**Alert Rules**:
```yaml
# Alert if cache hit rate < 70%
- alert: LowCacheHitRate
  expr: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m])) < 0.7
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Low cache hit rate (<70%)"
```

---

## 4. LLM Health Check

### Purpose

각 LLM 모델별 헬스체크 통과 확인 - 승급(카나리→프로덕션) 전 필수.

### Implementation

**파일**: `app/routers/health.py`

```python
from fastapi import APIRouter, Query
from typing import Optional

router = APIRouter()

@router.get("/health/llm")
async def llm_health_check(
    model: Optional[str] = Query(None, regex="^(qwen|mistral|llama|all)$")
):
    """
    LLM model health check

    Query params:
        model: Model to check (qwen, mistral, llama, all)

    Returns:
        200 OK if model is healthy
        503 Service Unavailable if model is unhealthy
    """
    from fragrance_ai.llm.health_check import LLMHealthChecker

    checker = LLMHealthChecker()

    if model == "all" or model is None:
        # Check all models
        results = await checker.check_all_models()

        all_healthy = all(r["status"] == "healthy" for r in results.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "models": results
        }
    else:
        # Check specific model
        result = await checker.check_model(model)

        if result["status"] != "healthy":
            raise HTTPException(
                status_code=503,
                detail=f"Model {model} is unhealthy"
            )

        return result
```

**파일**: `fragrance_ai/llm/health_check.py`

```python
class LLMHealthChecker:
    """Health checker for LLM models"""

    def __init__(self):
        self.circuit_breaker_manager = LLMCircuitBreakerManager()
        self.timeout = 10  # seconds

    async def check_model(self, model: str) -> dict:
        """Check health of specific model"""

        # Check circuit breaker state
        if not self.circuit_breaker_manager.is_available(model):
            return {
                "model": model,
                "status": "unhealthy",
                "reason": "circuit_breaker_open",
                "circuit_state": "open"
            }

        # Perform test inference
        try:
            start_time = time.time()

            result = await self._test_inference(model)

            latency = time.time() - start_time

            if result and latency < self.timeout:
                return {
                    "model": model,
                    "status": "healthy",
                    "latency_ms": int(latency * 1000),
                    "circuit_state": "closed"
                }
            else:
                return {
                    "model": model,
                    "status": "unhealthy",
                    "reason": "timeout" if latency >= self.timeout else "invalid_response",
                    "latency_ms": int(latency * 1000)
                }

        except Exception as e:
            return {
                "model": model,
                "status": "unhealthy",
                "reason": str(e),
                "circuit_state": "error"
            }

    async def check_all_models(self) -> dict:
        """Check health of all models"""
        models = ["qwen", "mistral", "llama"]

        results = {}
        for model in models:
            results[model] = await self.check_model(model)

        return results

    async def _test_inference(self, model: str) -> bool:
        """Perform test inference on model"""
        from fragrance_ai.llm.ensemble import LLMEnsemble

        ensemble = LLMEnsemble()

        # Simple test prompt
        test_prompt = "Generate a brief perfume description."

        try:
            result = await ensemble.generate_single(
                prompt=test_prompt,
                model=model,
                timeout=self.timeout
            )

            # Check if result is valid
            return result is not None and len(result) > 0

        except Exception as e:
            logger.error(f"Test inference failed for {model}: {e}")
            return False
```

### Verification

```bash
# Check all LLM models
curl -s "http://localhost:8000/health/llm?model=all" | jq .

# Expected output (all healthy):
# {
#   "status": "healthy",
#   "models": {
#     "qwen": {
#       "model": "qwen",
#       "status": "healthy",
#       "latency_ms": 850,
#       "circuit_state": "closed"
#     },
#     "mistral": {
#       "model": "mistral",
#       "status": "healthy",
#       "latency_ms": 920,
#       "circuit_state": "closed"
#     },
#     "llama": {
#       "model": "llama",
#       "status": "healthy",
#       "latency_ms": 780,
#       "circuit_state": "closed"
#     }
#   }
# }

# Check specific model (Qwen)
curl -s "http://localhost:8000/health/llm?model=qwen" | jq .

# Check specific model (Mistral)
curl -s "http://localhost:8000/health/llm?model=mistral" | jq .

# Check specific model (Llama)
curl -s "http://localhost:8000/health/llm?model=llama" | jq .
```

### Pre-Promotion Check

Before promoting canary to production:

```bash
# Check ALL models are healthy
HEALTH_CHECK=$(curl -s "http://localhost:8001/health/llm?model=all")

# Parse status
STATUS=$(echo "$HEALTH_CHECK" | jq -r '.status')

if [ "$STATUS" == "healthy" ]; then
    echo "✓ All LLM models healthy - OK to promote"
else
    echo "✗ LLM models unhealthy - DO NOT PROMOTE"
    echo "$HEALTH_CHECK" | jq .
    exit 1
fi
```

### Monitoring

**Prometheus Metrics**:
```promql
# LLM model health status (1=healthy, 0=unhealthy)
llm_health_status{model="qwen"}

# Test inference latency
llm_health_check_duration_seconds{model="qwen"}

# Health check failures
rate(llm_health_check_failures_total{model="qwen"}[5m])
```

**Alert Rules**:
```yaml
# Alert if any model unhealthy
- alert: LLMModelUnhealthy
  expr: llm_health_status == 0
  for: 5m
  labels:
    severity: critical
  annotations:
    summary: "LLM model {{ $labels.model }} is unhealthy"
    description: "Model has been unhealthy for 5 minutes"
```

---

## 📋 Complete Pre-Promotion Checklist

Before promoting canary to production, verify ALL guards:

```bash
#!/bin/bash
# Complete guards verification

echo "Verifying production guards..."
echo "======================================"

# 1. JSON Hard Guard
echo "1. JSON Hard Guard..."
python -c "
from fragrance_ai.config_loader import get_config
config = get_config()
assert config.llm.json_guard_enabled == True
assert config.llm.max_retries >= 3
print('✓ ENABLED')
"

# 2. Circuit Breaker
echo "2. Circuit Breaker..."
python -c "
from fragrance_ai.llm.circuit_breaker import LLMCircuitBreakerManager
manager = LLMCircuitBreakerManager()
assert all(b.state == 'closed' for b in manager.breakers.values())
print('✓ ALL BREAKERS CLOSED')
"

# 3. Cache LRU+TTL
echo "3. Cache LRU+TTL..."
python scripts/test_cache_ttl.py

# 4. LLM Health Check
echo "4. LLM Health Check..."
HEALTH=$(curl -s "http://localhost:8001/health/llm?model=all")
STATUS=$(echo "$HEALTH" | jq -r '.status')

if [ "$STATUS" == "healthy" ]; then
    echo "✓ ALL MODELS HEALTHY"
else
    echo "✗ MODELS UNHEALTHY"
    echo "$HEALTH" | jq .
    exit 1
fi

echo "======================================"
echo "✓ ALL GUARDS VERIFIED"
echo "✓ SAFE TO PROMOTE TO PRODUCTION"
```

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Maintained by**: DevOps & ML Engineering Team
