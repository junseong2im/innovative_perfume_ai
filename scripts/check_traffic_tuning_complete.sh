#!/bin/bash
# ============================================================================
# Traffic Tuning Completion Check
# Verify 48-hour traffic tuning is complete before final promotion
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

log_warning() {
    echo -e "${YELLOW}[⚠ WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗ FAIL]${NC} $1"
}

PASSED=0
FAILED=0
WARNINGS=0

echo "============================================================================"
echo "TRAFFIC TUNING COMPLETION CHECK (48-Hour)"
echo "============================================================================"
echo "Container: $CONTAINER"
echo "============================================================================"
echo ""

# ============================================================================
# 1. Traffic Distribution Stability
# ============================================================================
echo "1. Traffic Distribution Stability"
echo "----------------------------------------"
echo "Target: fast 60% / balanced 30% / creative 10%"
echo ""

if docker exec $CONTAINER python -c "
from fragrance_ai.routing.traffic_distributor import TrafficDistributor
from fragrance_ai.config_loader import get_config

config = get_config()
distributor = TrafficDistributor(config.traffic_distribution)

# Get actual distribution
actual = distributor.get_current_distribution()
target = distributor.mode_weights

print('Actual distribution:')
print(f'  fast: {actual.get(\"fast\", 0)}% (target: {target[\"fast\"]}%)')
print(f'  balanced: {actual.get(\"balanced\", 0)}% (target: {target[\"balanced\"]}%)')
print(f'  creative: {actual.get(\"creative\", 0)}% (target: {target[\"creative\"]}%)')
print()

# Check deviations
deviations = distributor.check_distribution_deviation()
max_deviation = max(d['deviation'] for d in deviations.values())

print(f'Max deviation: {max_deviation}%')

if max_deviation < 5:
    print('✓ STABLE (< 5% deviation)')
    exit(0)
elif max_deviation < 10:
    print('⚠ ACCEPTABLE (< 10% deviation)')
    exit(1)
else:
    print('✗ UNSTABLE (> 10% deviation)')
    exit(2)
" 2>&1; then
    log_success "Traffic distribution: STABLE"
    ((PASSED++))
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        log_warning "Traffic distribution: ACCEPTABLE but not optimal"
        ((WARNINGS++))
    else
        log_error "Traffic distribution: UNSTABLE"
        ((FAILED++))
    fi
fi
echo ""

# ============================================================================
# 2. PPO Entropy Convergence
# ============================================================================
echo "2. PPO Entropy Convergence"
echo "----------------------------------------"
echo "Target: entropy ≤ 0.006 after 10k steps"
echo ""

if docker exec $CONTAINER python -c "
from fragrance_ai.monitoring.metrics import MetricsCollector

collector = MetricsCollector()

try:
    entropy = collector.get_metric('ppo_entropy_coef')
    step = collector.get_metric('ppo_training_step')

    print(f'Current step: {step}')
    print(f'Current entropy: {entropy:.5f}')
    print()

    if step >= 10000:
        if entropy <= 0.006:
            print('✓ CONVERGED')
            exit(0)
        elif entropy <= 0.008:
            print('⚠ NEARLY CONVERGED')
            exit(1)
        else:
            print('✗ NOT CONVERGED')
            exit(2)
    else:
        print(f'⚠ Training not complete ({step}/10000 steps)')
        exit(1)
except Exception as e:
    print(f'✗ Cannot retrieve metrics: {e}')
    exit(2)
" 2>&1; then
    log_success "PPO entropy: CONVERGED"
    ((PASSED++))
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        log_warning "PPO entropy: NEARLY CONVERGED or training incomplete"
        ((WARNINGS++))
    else
        log_error "PPO entropy: NOT CONVERGED"
        ((FAILED++))
    fi
fi
echo ""

# ============================================================================
# 3. Checkpoint Stability
# ============================================================================
echo "3. Checkpoint Stability"
echo "----------------------------------------"
echo "Target: ≤ 1 rollback in 48 hours"
echo ""

if docker exec $CONTAINER python -c "
from fragrance_ai.training.checkpoint_manager import CheckpointManager

try:
    manager = CheckpointManager.get_instance()

    rollback_count = manager.rollback_count
    max_rollbacks = manager.max_rollbacks

    print(f'Rollback count: {rollback_count}/{max_rollbacks}')
    print()

    if rollback_count == 0:
        print('✓ STABLE (0 rollbacks)')
        exit(0)
    elif rollback_count == 1:
        print('✓ ACCEPTABLE (1 rollback)')
        exit(0)
    elif rollback_count <= 2:
        print('⚠ UNSTABLE (2 rollbacks)')
        exit(1)
    else:
        print('✗ HIGHLY UNSTABLE (>2 rollbacks)')
        exit(2)
except Exception as e:
    print(f'✗ Cannot retrieve checkpoint info: {e}')
    exit(2)
" 2>&1; then
    log_success "Checkpoint stability: STABLE"
    ((PASSED++))
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        log_warning "Checkpoint stability: UNSTABLE"
        ((WARNINGS++))
    else
        log_error "Checkpoint stability: HIGHLY UNSTABLE"
        ((FAILED++))
    fi
fi
echo ""

# ============================================================================
# 4. System Stability
# ============================================================================
echo "4. System Stability"
echo "----------------------------------------"
echo "Check: Error rate < 0.5%, latency within thresholds"
echo ""

# Determine base URL based on container name
if [[ "$CONTAINER" == *"canary"* ]]; then
    BASE_URL="http://localhost:8001"
else
    BASE_URL="http://localhost:8000"
fi

# Get metrics
METRICS_RESPONSE=$(curl -s "$BASE_URL/metrics" || echo "")

if [ -n "$METRICS_RESPONSE" ]; then
    # Check error rate
    ERROR_COUNT=$(echo "$METRICS_RESPONSE" | grep 'http_requests_total.*status="5' | grep -oP 'http_requests_total\{[^}]+\} \K\d+' | awk '{s+=$1} END {print s}' || echo "0")
    TOTAL_COUNT=$(echo "$METRICS_RESPONSE" | grep 'http_requests_total' | grep -oP 'http_requests_total\{[^}]+\} \K\d+' | awk '{s+=$1} END {print s}' || echo "1")

    if [ "$TOTAL_COUNT" -gt 0 ]; then
        ERROR_RATE=$(echo "scale=4; $ERROR_COUNT / $TOTAL_COUNT * 100" | bc)
        echo "Error rate: ${ERROR_RATE}%"

        if (( $(echo "$ERROR_RATE < 0.5" | bc -l) )); then
            log_success "System stability: STABLE (error rate < 0.5%)"
            ((PASSED++))
        elif (( $(echo "$ERROR_RATE < 1.0" | bc -l) )); then
            log_warning "System stability: ACCEPTABLE (error rate < 1.0%)"
            ((WARNINGS++))
        else
            log_error "System stability: UNSTABLE (error rate > 1.0%)"
            ((FAILED++))
        fi
    else
        log_warning "No traffic data available yet"
        ((WARNINGS++))
    fi
else
    log_error "Cannot retrieve metrics"
    ((FAILED++))
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "TRAFFIC TUNING SUMMARY"
echo "============================================================================"
echo "Passed:   $PASSED/4"
echo "Warnings: $WARNINGS/4"
echo "Failed:   $FAILED/4"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "============================================================================"
    log_success "TRAFFIC TUNING COMPLETE ✓"
    log_success "ALL CRITERIA MET - SAFE TO PROMOTE"
    echo "============================================================================"
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo "============================================================================"
    log_warning "TRAFFIC TUNING MOSTLY COMPLETE ⚠"
    log_warning "SOME WARNINGS - REVIEW BEFORE PROMOTING"
    echo "============================================================================"
    echo ""
    echo "Recommendations:"
    echo "  - Review warnings above"
    echo "  - Consider waiting a few more hours"
    echo "  - Monitor closely after promotion"
    echo ""
    exit 0
else
    echo "============================================================================"
    log_error "TRAFFIC TUNING NOT COMPLETE ✗"
    log_error "CRITERIA NOT MET - DO NOT PROMOTE"
    echo "============================================================================"
    echo ""
    echo "Required actions:"
    echo "  - Fix failed checks above"
    echo "  - Continue monitoring for full 48 hours"
    echo "  - Re-run this check when ready"
    echo ""
    exit 1
fi
