#!/bin/bash
# 전체 테스트 실행 스크립트
# 유닛 테스트 + 통합 테스트 + 스모크 테스트

set -e  # Exit on error

echo "========================================="
echo "Fragrance AI 전체 테스트 실행"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

FAILED=0

# Function to run test and report result
run_test() {
    TEST_NAME=$1
    TEST_COMMAND=$2

    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}테스트: $TEST_NAME${NC}"
    echo -e "${BLUE}=========================================${NC}"

    if eval "$TEST_COMMAND"; then
        echo -e "${GREEN}✓ PASS${NC}: $TEST_NAME"
        echo ""
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}: $TEST_NAME"
        echo ""
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# ============================================================================
# 유닛 테스트
# ============================================================================

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}1단계: 유닛 테스트${NC}"
echo -e "${YELLOW}============================================${NC}"
echo ""

# LLM ensemble tests
run_test "LLM Ensemble" "pytest -q tests/test_llm_ensemble.py" || true

# MOGA stability tests
run_test "MOGA Stability" "pytest -q tests/test_moga_stability.py" || true

# End-to-end evolution tests
run_test "End-to-End Evolution" "pytest -q tests/test_end_to_end_evolution.py" || true

# RL advanced features tests
run_test "RL Advanced Features" "pytest -q tests/test_rl_advanced.py" || true

# GA tests
run_test "Genetic Algorithm" "pytest -q tests/test_ga.py" || true

# IFRA tests
run_test "IFRA Regulations" "pytest -q tests/test_ifra.py" || true

echo ""

# ============================================================================
# API 스모크 테스트
# ============================================================================

echo -e "${YELLOW}============================================${NC}"
echo -e "${YELLOW}2단계: API 스모크 테스트${NC}"
echo -e "${YELLOW}============================================${NC}"
echo ""

# Check if API is running
if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API 서버가 실행 중입니다${NC}"

    # Run API smoke tests
    run_test "API Smoke Tests" "bash smoke_test_api.sh" || true
else
    echo -e "${YELLOW}⚠ API 서버가 실행되지 않았습니다. API 테스트를 건너뜁니다.${NC}"
    echo -e "${YELLOW}  API를 시작하려면: uvicorn app.main:app --reload${NC}"
    echo ""
fi

echo ""

# ============================================================================
# 테스트 결과 요약
# ============================================================================

echo "========================================="
echo "테스트 결과 요약"
echo "========================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ 모든 테스트 통과!${NC}"
    exit 0
else
    echo -e "${RED}✗ 실패한 테스트: $FAILED${NC}"
    exit 1
fi
