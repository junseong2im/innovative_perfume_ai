#!/bin/bash
# API 스모크 테스트 스크립트
# REINFORCE → PPO 진화 파이프라인 테스트

set -e  # Exit on error

API_URL="${API_URL:-http://localhost:8000}"
TIMEOUT="${TIMEOUT:-30}"

echo "========================================="
echo "API 스모크 테스트 시작"
echo "API URL: $API_URL"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
}

# Function to check if jq is installed
check_jq() {
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}경고: jq가 설치되지 않았습니다. JSON 파싱 없이 진행합니다.${NC}"
        return 1
    fi
    return 0
}

HAS_JQ=$(check_jq && echo 1 || echo 0)

echo "========================================="
echo "테스트 1: 서버 헬스 체크"
echo "========================================="

HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT "$API_URL/health" || echo "000")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -1)
HEALTH_BODY=$(echo "$HEALTH_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ]; then
    print_result 0 "서버 헬스 체크"
    echo "Response: $HEALTH_BODY"
else
    print_result 1 "서버 헬스 체크 (HTTP $HTTP_CODE)"
    echo "Response: $HEALTH_BODY"
fi

echo ""
echo "========================================="
echo "테스트 2: DNA 생성 (POST /dna/create)"
echo "========================================="

CREATE_DNA_PAYLOAD='{
  "brief": {
    "style": "fresh",
    "intensity": 0.7,
    "complexity": 0.5,
    "notes_preference": {
      "citrus": 0.8,
      "fresh": 0.7,
      "floral": 0.3
    },
    "mood": ["상큼함", "활기찬"],
    "season": ["spring", "summer"],
    "product_category": "EDP",
    "target_profile": "daily_fresh"
  }
}'

echo "Payload:"
echo "$CREATE_DNA_PAYLOAD" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )
echo ""

DNA_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
    -X POST "$API_URL/dna/create" \
    -H "Content-Type: application/json" \
    -d "$CREATE_DNA_PAYLOAD" || echo "000")

HTTP_CODE=$(echo "$DNA_RESPONSE" | tail -1)
DNA_BODY=$(echo "$DNA_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    print_result 0 "DNA 생성"

    echo "Response:"
    echo "$DNA_BODY" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )

    if [ "$HAS_JQ" = "1" ]; then
        DNA_ID=$(echo "$DNA_BODY" | jq -r '.dna_id // .id // empty')
        if [ -z "$DNA_ID" ]; then
            echo -e "${YELLOW}경고: DNA ID를 찾을 수 없습니다${NC}"
            DNA_ID="test-dna-001"
        fi
    else
        DNA_ID="test-dna-001"
    fi

    echo "DNA ID: $DNA_ID"
else
    print_result 1 "DNA 생성 (HTTP $HTTP_CODE)"
    echo "Response: $DNA_BODY"
    DNA_ID="test-dna-001"
fi

echo ""
echo "========================================="
echo "테스트 3: 옵션 진화 - REINFORCE (POST /evolve/options)"
echo "========================================="

EVOLVE_OPTIONS_PAYLOAD=$(cat <<EOF
{
  "dna_id": "$DNA_ID",
  "algorithm": "REINFORCE",
  "num_options": 3,
  "parameters": {
    "n_iterations": 50,
    "population_size": 20
  }
}
EOF
)

echo "Payload:"
echo "$EVOLVE_OPTIONS_PAYLOAD" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )
echo ""

EVOLVE_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
    -X POST "$API_URL/evolve/options" \
    -H "Content-Type: application/json" \
    -d "$EVOLVE_OPTIONS_PAYLOAD" || echo "000")

HTTP_CODE=$(echo "$EVOLVE_RESPONSE" | tail -1)
EVOLVE_BODY=$(echo "$EVOLVE_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    print_result 0 "옵션 진화 (REINFORCE)"

    echo "Response:"
    echo "$EVOLVE_BODY" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )

    if [ "$HAS_JQ" = "1" ]; then
        EXPERIMENT_ID=$(echo "$EVOLVE_BODY" | jq -r '.experiment_id // .id // empty')
        OPTION_ID=$(echo "$EVOLVE_BODY" | jq -r '.options[0].id // .options[0].option_id // empty')

        if [ -z "$EXPERIMENT_ID" ]; then
            echo -e "${YELLOW}경고: Experiment ID를 찾을 수 없습니다${NC}"
            EXPERIMENT_ID="test-experiment-001"
        fi

        if [ -z "$OPTION_ID" ]; then
            echo -e "${YELLOW}경고: Option ID를 찾을 수 없습니다${NC}"
            OPTION_ID="option-001"
        fi
    else
        EXPERIMENT_ID="test-experiment-001"
        OPTION_ID="option-001"
    fi

    echo "Experiment ID: $EXPERIMENT_ID"
    echo "Option ID: $OPTION_ID"

    # Check for expected log entries
    if echo "$EVOLVE_BODY" | grep -q "llm_brief\|rl_update"; then
        echo -e "${GREEN}✓${NC} 로그에 llm_brief/rl_update 포함됨"
    else
        echo -e "${YELLOW}⚠${NC} 로그에 llm_brief/rl_update가 없습니다 (선택사항)"
    fi
else
    print_result 1 "옵션 진화 (HTTP $HTTP_CODE)"
    echo "Response: $EVOLVE_BODY"
    EXPERIMENT_ID="test-experiment-001"
    OPTION_ID="option-001"
fi

echo ""
echo "========================================="
echo "테스트 4: 피드백 제출 (POST /evolve/feedback)"
echo "========================================="

FEEDBACK_PAYLOAD=$(cat <<EOF
{
  "experiment_id": "$EXPERIMENT_ID",
  "chosen_id": "$OPTION_ID",
  "rating": 5,
  "feedback_text": "상큼하고 산뜻해서 좋아요!"
}
EOF
)

echo "Payload:"
echo "$FEEDBACK_PAYLOAD" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )
echo ""

FEEDBACK_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
    -X POST "$API_URL/evolve/feedback" \
    -H "Content-Type: application/json" \
    -d "$FEEDBACK_PAYLOAD" || echo "000")

HTTP_CODE=$(echo "$FEEDBACK_RESPONSE" | tail -1)
FEEDBACK_BODY=$(echo "$FEEDBACK_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    print_result 0 "피드백 제출"

    echo "Response:"
    echo "$FEEDBACK_BODY" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )

    # Check for RL update logs
    if echo "$FEEDBACK_BODY" | grep -q "loss\|reward\|entropy\|clip_frac"; then
        echo -e "${GREEN}✓${NC} RL 업데이트 로그 포함 (loss, reward, entropy, clip_frac)"
    else
        echo -e "${YELLOW}⚠${NC} RL 업데이트 로그가 없습니다 (선택사항)"
    fi
else
    print_result 1 "피드백 제출 (HTTP $HTTP_CODE)"
    echo "Response: $FEEDBACK_BODY"
fi

echo ""
echo "========================================="
echo "테스트 5: PPO 재학습 (옵션 진화 - PPO)"
echo "========================================="

PPO_EVOLVE_PAYLOAD=$(cat <<EOF
{
  "dna_id": "$DNA_ID",
  "algorithm": "PPO",
  "num_options": 3,
  "parameters": {
    "n_iterations": 30,
    "n_steps_per_iteration": 512,
    "n_ppo_epochs": 5
  }
}
EOF
)

echo "Payload:"
echo "$PPO_EVOLVE_PAYLOAD" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )
echo ""

PPO_RESPONSE=$(curl -s -w "\n%{http_code}" --max-time $TIMEOUT \
    -X POST "$API_URL/evolve/options" \
    -H "Content-Type: application/json" \
    -d "$PPO_EVOLVE_PAYLOAD" || echo "000")

HTTP_CODE=$(echo "$PPO_RESPONSE" | tail -1)
PPO_BODY=$(echo "$PPO_RESPONSE" | sed '$d')

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    print_result 0 "PPO 재학습"

    echo "Response:"
    echo "$PPO_BODY" | ( [ "$HAS_JQ" = "1" ] && jq '.' || cat )

    # Check for PPO-specific metrics
    if echo "$PPO_BODY" | grep -q "clip_frac\|kl_divergence\|entropy"; then
        echo -e "${GREEN}✓${NC} PPO 메트릭 포함 (clip_frac, kl_divergence, entropy)"
    else
        echo -e "${YELLOW}⚠${NC} PPO 메트릭이 없습니다 (선택사항)"
    fi
else
    print_result 1 "PPO 재학습 (HTTP $HTTP_CODE)"
    echo "Response: $PPO_BODY"
fi

echo ""
echo "========================================="
echo "테스트 결과 요약"
echo "========================================="
echo -e "통과: ${GREEN}$TESTS_PASSED${NC}"
echo -e "실패: ${RED}$TESTS_FAILED${NC}"
echo "총 테스트: $((TESTS_PASSED + TESTS_FAILED))"
echo "========================================="

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}모든 스모크 테스트 통과! ✓${NC}"
    exit 0
else
    echo -e "${RED}일부 테스트 실패 ✗${NC}"
    exit 1
fi
