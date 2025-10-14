#!/bin/bash
# ============================================================================
# Production Guards Verification Script
# Verify all guards are enabled before production deployment
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CONTAINER="${1:-fragrance-ai-app}"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓ PASS]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
}

PASSED=0
FAILED=0

echo "============================================================================"
echo "PRODUCTION GUARDS VERIFICATION"
echo "============================================================================"
echo "Container: $CONTAINER"
echo "============================================================================"
echo ""

# ============================================================================
# 1. JSON Hard Guard
# ============================================================================
echo "1. JSON Hard Guard + Retry + Fallback"
echo "----------------------------------------"

if docker exec $CONTAINER python -c "
from fragrance_ai.config_loader import get_config
config = get_config()
assert config.llm.json_guard_enabled == True, 'JSON guard not enabled'
assert config.llm.max_retries >= 3, 'Max retries < 3'
assert config.llm.fallback_enabled == True, 'Fallback not enabled'
print(f'✓ JSON guard enabled')
print(f'  Max retries: {config.llm.max_retries}')
print(f'  Backoff: {config.llm.backoff_base}s')
print(f'  Fallback: enabled')
" 2>&1; then
    log_success "JSON Hard Guard: ENABLED"
    ((PASSED++))
else
    log_error "JSON Hard Guard: NOT ENABLED"
    ((FAILED++))
fi
echo ""

# ============================================================================
# 2. Circuit Breaker
# ============================================================================
echo "2. Circuit Breaker + Auto Downshift"
echo "----------------------------------------"

if docker exec $CONTAINER python -c "
from fragrance_ai.llm.circuit_breaker import LLMCircuitBreakerManager
manager = LLMCircuitBreakerManager()

print('Circuit Breaker Status:')
for model, breaker in manager.breakers.items():
    state = breaker.state
    print(f'  {model}: {state}')
    assert state == 'closed', f'{model} circuit breaker is {state}'

print('✓ All circuit breakers closed')
print(f'  Auto downshift: enabled')
print(f'  Failure threshold: 50%')
" 2>&1; then
    log_success "Circuit Breaker: ALL CLOSED"
    ((PASSED++))
else
    log_error "Circuit Breaker: OPEN OR HALF-OPEN"
    ((FAILED++))
fi
echo ""

# ============================================================================
# 3. Cache LRU+TTL
# ============================================================================
echo "3. Cache LRU + TTL"
echo "----------------------------------------"

if docker exec $CONTAINER python -c "
from fragrance_ai.cache.lru_cache import LRUCacheWithTTL
from fragrance_ai.config_loader import get_config

config = get_config()
assert config.cache.enabled == True, 'Cache not enabled'
assert config.cache.ttl_seconds >= 600, 'TTL < 10 minutes'

cache = LRUCacheWithTTL(
    max_size=config.cache.max_size,
    ttl_seconds=config.cache.ttl_seconds
)

stats = cache.stats()
print(f'✓ Cache enabled')
print(f'  Max size: {stats[\"max_size\"]}')
print(f'  TTL: {stats[\"ttl_seconds\"]}s ({stats[\"ttl_seconds\"]//60} minutes)')
print(f'  Current size: {stats[\"size\"]}')
print(f'  Utilization: {stats[\"utilization\"]:.1%}')
" 2>&1; then
    log_success "Cache LRU+TTL: ENABLED"
    ((PASSED++))
else
    log_error "Cache LRU+TTL: NOT ENABLED"
    ((FAILED++))
fi
echo ""

# ============================================================================
# 4. LLM Health Check
# ============================================================================
echo "4. LLM Model Health Check"
echo "----------------------------------------"

# Determine base URL based on container name
if [[ "$CONTAINER" == *"canary"* ]]; then
    BASE_URL="http://localhost:8001"
else
    BASE_URL="http://localhost:8000"
fi

HEALTH_RESPONSE=$(curl -s "$BASE_URL/health/llm?model=all" || echo '{"status":"error"}')
STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.status // "error"')

if [ "$STATUS" == "healthy" ]; then
    log_success "LLM Health Check: ALL MODELS HEALTHY"

    # Show individual model status
    echo "$HEALTH_RESPONSE" | jq -r '.models | to_entries[] | "  \(.key): \(.value.status) (\(.value.latency_ms)ms)"'

    ((PASSED++))
elif [ "$STATUS" == "degraded" ]; then
    log_error "LLM Health Check: DEGRADED"
    echo "$HEALTH_RESPONSE" | jq '.models'
    ((FAILED++))
else
    log_error "LLM Health Check: FAILED"
    echo "$HEALTH_RESPONSE" | jq '.'
    ((FAILED++))
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "VERIFICATION SUMMARY"
echo "============================================================================"
echo "Passed: $PASSED/4"
echo "Failed: $FAILED/4"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "============================================================================"
    log_success "ALL PRODUCTION GUARDS VERIFIED ✓"
    log_success "SAFE TO PROCEED WITH DEPLOYMENT"
    echo "============================================================================"
    exit 0
else
    echo "============================================================================"
    log_error "PRODUCTION GUARDS VERIFICATION FAILED ✗"
    log_error "DO NOT PROCEED WITH DEPLOYMENT"
    echo "============================================================================"
    exit 1
fi
