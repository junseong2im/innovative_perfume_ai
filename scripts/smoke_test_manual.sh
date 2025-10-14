#!/bin/bash
# ============================================================================
# Manual Smoke Test Script
# 런치 직후 5분 수동 검증
# ============================================================================
#
# Usage:
#   ./scripts/smoke_test_manual.sh [base_url]
#
# Examples:
#   ./scripts/smoke_test_manual.sh http://localhost:8000
#   ./scripts/smoke_test_manual.sh http://localhost:8001  # Canary
#
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
BASE_URL="${1:-http://localhost:8000}"
TIMEOUT=30
CONTAINER_NAME="${2:-fragrance-ai-app}"

# Helper functions
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

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Test results tracking
PASSED=0
FAILED=0
WARNINGS=0

# Start time
START_TIME=$(date +%s)

echo "============================================================================"
echo "MANUAL SMOKE TEST - 런치 직후 5분 검증"
echo "============================================================================"
echo "Base URL: $BASE_URL"
echo "Container: $CONTAINER_NAME"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"
echo ""

# ============================================================================
# Step 1: Health Check
# ============================================================================
log_step "Step 1: Health Check"

if curl -s -f --max-time $TIMEOUT "$BASE_URL/health" > /dev/null; then
    HEALTH_RESPONSE=$(curl -s --max-time $TIMEOUT "$BASE_URL/health")
    log_success "Health check passed"
    echo "$HEALTH_RESPONSE" | jq '.'
    ((PASSED++))
else
    log_error "Health check failed"
    ((FAILED++))
    echo ""
    exit 1
fi
echo ""

# ============================================================================
# Step 2: API Smoke Test - /dna/create
# ============================================================================
log_step "Step 2: API Smoke Test - /dna/create"

log_info "Creating perfume DNA..."

CREATE_PAYLOAD='{
  "brief": {
    "mood": "상큼함",
    "season": ["spring"]
  }
}'

CREATE_RESPONSE=$(curl -s --max-time $TIMEOUT \
  -X POST "$BASE_URL/dna/create" \
  -H 'Content-Type: application/json' \
  -d "$CREATE_PAYLOAD")

if [ -n "$CREATE_RESPONSE" ]; then
    echo "$CREATE_RESPONSE" | jq '.'

    # Extract DNA ID
    DNA_ID=$(echo "$CREATE_RESPONSE" | jq -r '.dna_id // .id // empty')

    if [ -n "$DNA_ID" ] && [ "$DNA_ID" != "null" ]; then
        log_success "DNA created successfully: $DNA_ID"
        ((PASSED++))
    else
        log_error "DNA ID not found in response"
        ((FAILED++))
        DNA_ID=""
    fi
else
    log_error "/dna/create failed - no response"
    ((FAILED++))
    DNA_ID=""
fi
echo ""

# Wait a moment for processing
sleep 2

# ============================================================================
# Step 3: API Smoke Test - /evolve/options
# ============================================================================
log_step "Step 3: API Smoke Test - /evolve/options"

if [ -n "$DNA_ID" ]; then
    log_info "Generating evolution options (PPO, creative mode)..."

    EVOLVE_PAYLOAD=$(cat <<EOF
{
  "dna_id": "$DNA_ID",
  "algorithm": "PPO",
  "num_options": 3,
  "mode": "creative"
}
EOF
)

    EVOLVE_RESPONSE=$(curl -s --max-time $TIMEOUT \
      -X POST "$BASE_URL/evolve/options" \
      -H 'Content-Type: application/json' \
      -d "$EVOLVE_PAYLOAD")

    if [ -n "$EVOLVE_RESPONSE" ]; then
        echo "$EVOLVE_RESPONSE" | jq '.'

        # Extract experiment ID and first option ID
        EXPERIMENT_ID=$(echo "$EVOLVE_RESPONSE" | jq -r '.experiment_id // .id // empty')
        OPTION_ID=$(echo "$EVOLVE_RESPONSE" | jq -r '.options[0].id // .options[0].option_id // empty')

        if [ -n "$EXPERIMENT_ID" ] && [ "$EXPERIMENT_ID" != "null" ]; then
            log_success "Evolution options generated: experiment_id=$EXPERIMENT_ID"
            ((PASSED++))
        else
            log_error "Experiment ID not found in response"
            ((FAILED++))
            EXPERIMENT_ID=""
        fi

        if [ -n "$OPTION_ID" ] && [ "$OPTION_ID" != "null" ]; then
            log_success "First option ID: $OPTION_ID"
        else
            log_warning "Option ID not found in response"
            ((WARNINGS++))
            OPTION_ID=""
        fi
    else
        log_error "/evolve/options failed - no response"
        ((FAILED++))
        EXPERIMENT_ID=""
        OPTION_ID=""
    fi
else
    log_warning "Skipping /evolve/options (no DNA_ID)"
    ((WARNINGS++))
    EXPERIMENT_ID=""
    OPTION_ID=""
fi
echo ""

# Wait a moment for processing
sleep 2

# ============================================================================
# Step 4: API Smoke Test - /evolve/feedback
# ============================================================================
log_step "Step 4: API Smoke Test - /evolve/feedback"

if [ -n "$EXPERIMENT_ID" ] && [ -n "$OPTION_ID" ]; then
    log_info "Submitting feedback..."

    FEEDBACK_PAYLOAD=$(cat <<EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "chosen_id": "$OPTION_ID",
  "rating": 5
}
EOF
)

    FEEDBACK_RESPONSE=$(curl -s --max-time $TIMEOUT \
      -X POST "$BASE_URL/evolve/feedback" \
      -H 'Content-Type: application/json' \
      -d "$FEEDBACK_PAYLOAD")

    if [ -n "$FEEDBACK_RESPONSE" ]; then
        echo "$FEEDBACK_RESPONSE" | jq '.'

        # Check for success indicator
        SUCCESS=$(echo "$FEEDBACK_RESPONSE" | jq -r '.success // .status // empty')

        if [ "$SUCCESS" == "true" ] || [ "$SUCCESS" == "success" ] || [ "$SUCCESS" == "ok" ]; then
            log_success "Feedback submitted successfully"
            ((PASSED++))
        else
            log_warning "Feedback response unclear: $SUCCESS"
            ((WARNINGS++))
        fi
    else
        log_error "/evolve/feedback failed - no response"
        ((FAILED++))
    fi
else
    log_warning "Skipping /evolve/feedback (no experiment_id or option_id)"
    ((WARNINGS++))
fi
echo ""

# Wait a moment for logs to be written
sleep 3

# ============================================================================
# Step 5: Log Verification - llm_brief metrics
# ============================================================================
log_step "Step 5: Log Verification - llm_brief metrics"

log_info "Checking for llm_brief{mode,...,elapsed_ms} in logs..."

# Try to find llm_brief logs
LLM_BRIEF_LOGS=$(docker logs "$CONTAINER_NAME" --since 5m 2>&1 | grep -i "llm_brief" | tail -20 || true)

if [ -n "$LLM_BRIEF_LOGS" ]; then
    log_success "Found llm_brief logs:"
    echo "----------------------------------------"
    echo "$LLM_BRIEF_LOGS"
    echo "----------------------------------------"

    # Check for required fields
    if echo "$LLM_BRIEF_LOGS" | grep -q "mode"; then
        log_success "✓ mode field present"
    else
        log_warning "⚠ mode field not found"
        ((WARNINGS++))
    fi

    if echo "$LLM_BRIEF_LOGS" | grep -q "elapsed_ms\|duration\|latency"; then
        log_success "✓ elapsed_ms/duration field present"
    else
        log_warning "⚠ elapsed_ms field not found"
        ((WARNINGS++))
    fi

    ((PASSED++))
else
    log_error "No llm_brief logs found"
    log_info "This may indicate:"
    log_info "  1. LLM brief generation not triggered"
    log_info "  2. Logs not yet written"
    log_info "  3. Logging configuration issue"
    ((FAILED++))
fi
echo ""

# ============================================================================
# Step 6: Log Verification - rl_update metrics
# ============================================================================
log_step "Step 6: Log Verification - rl_update metrics"

log_info "Checking for rl_update{algo,loss,reward,entropy,clip_frac} in logs..."

# Try to find rl_update logs
RL_UPDATE_LOGS=$(docker logs "$CONTAINER_NAME" --since 5m 2>&1 | grep -i "rl_update\|rl_training\|ppo_update" | tail -20 || true)

if [ -n "$RL_UPDATE_LOGS" ]; then
    log_success "Found rl_update logs:"
    echo "----------------------------------------"
    echo "$RL_UPDATE_LOGS"
    echo "----------------------------------------"

    # Check for required fields
    REQUIRED_FIELDS=("algo" "loss" "reward" "entropy" "clip_frac")
    for field in "${REQUIRED_FIELDS[@]}"; do
        if echo "$RL_UPDATE_LOGS" | grep -qi "$field"; then
            log_success "✓ $field field present"
        else
            log_warning "⚠ $field field not found"
            ((WARNINGS++))
        fi
    done

    ((PASSED++))
else
    log_warning "No rl_update logs found"
    log_info "This may be expected if:"
    log_info "  1. RL training not triggered yet (requires feedback)"
    log_info "  2. Training batch not completed"
    log_info "  3. Async worker processing"
    ((WARNINGS++))
fi
echo ""

# ============================================================================
# Step 7: Additional Log Checks
# ============================================================================
log_step "Step 7: Additional Log Checks"

log_info "Checking for errors in recent logs..."

# Check for errors in last 5 minutes
ERROR_LOGS=$(docker logs "$CONTAINER_NAME" --since 5m 2>&1 | grep -i "error\|exception\|traceback" | grep -v "ERROR_RATE\|error_rate" | head -10 || true)

if [ -n "$ERROR_LOGS" ]; then
    log_warning "Found errors in logs:"
    echo "----------------------------------------"
    echo "$ERROR_LOGS"
    echo "----------------------------------------"
    ((WARNINGS++))
else
    log_success "No errors found in recent logs"
    ((PASSED++))
fi
echo ""

# ============================================================================
# Step 8: Metrics Endpoint Check
# ============================================================================
log_step "Step 8: Metrics Endpoint Check"

log_info "Checking /metrics endpoint..."

if curl -s -f --max-time $TIMEOUT "$BASE_URL/metrics" > /dev/null; then
    METRICS_RESPONSE=$(curl -s --max-time $TIMEOUT "$BASE_URL/metrics")

    # Check for llm_brief metrics
    if echo "$METRICS_RESPONSE" | grep -q "llm_brief"; then
        log_success "✓ llm_brief metrics present in /metrics"
        ((PASSED++))
    else
        log_warning "⚠ llm_brief metrics not found in /metrics"
        ((WARNINGS++))
    fi

    # Check for rl_update metrics
    if echo "$METRICS_RESPONSE" | grep -q "rl_update\|rl_training"; then
        log_success "✓ rl_update metrics present in /metrics"
        ((PASSED++))
    else
        log_warning "⚠ rl_update metrics not found in /metrics (may not be published yet)"
        ((WARNINGS++))
    fi
else
    log_warning "/metrics endpoint not accessible"
    ((WARNINGS++))
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "============================================================================"
echo "SMOKE TEST SUMMARY"
echo "============================================================================"
echo "Duration: ${DURATION}s"
echo ""
echo "Results:"
echo "  ✓ Passed:   $PASSED"
echo "  ✗ Failed:   $FAILED"
echo "  ⚠ Warnings: $WARNINGS"
echo ""

if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo "============================================================================"
        log_success "ALL TESTS PASSED - System is healthy ✓"
        echo "============================================================================"
        exit 0
    else
        echo "============================================================================"
        log_warning "TESTS PASSED WITH WARNINGS - Review warnings above"
        echo "============================================================================"
        exit 0
    fi
else
    echo "============================================================================"
    log_error "TESTS FAILED - Review failures above"
    echo "============================================================================"
    exit 1
fi
