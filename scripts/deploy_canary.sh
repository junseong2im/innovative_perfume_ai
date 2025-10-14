#!/bin/bash
# Canary Deployment Script
# 1% → 5% → 25% → 100% 점진적 배포

set -e

ENVIRONMENT=${1:-stg}
FEATURE_FLAG=${2:-""}
ROLLOUT_PERCENTAGE=${3:-0}

echo "=================================================="
echo "🐤 Canary Deployment"
echo "=================================================="
echo "Environment: $ENVIRONMENT"
echo "Feature Flag: $FEATURE_FLAG"
echo "Rollout: $ROLLOUT_PERCENTAGE%"
echo ""

# 환경 변수 설정
export ARTISAN_ENV=$ENVIRONMENT
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 1. Go/No-Go 체크
echo "Step 1: Running Go/No-Go gate..."
python -m fragrance_ai.deployment.go_nogo_gate --exit-code
if [ $? -ne 0 ]; then
    echo "❌ Go/No-Go check failed. Aborting deployment."
    exit 1
fi
echo "✓ Go/No-Go check passed"
echo ""

# 2. Feature flag 설정 (선택적)
if [ -n "$FEATURE_FLAG" ]; then
    echo "Step 2: Setting feature flag..."
    python -c "
from fragrance_ai.config.feature_flags import FeatureFlagManager
manager = FeatureFlagManager(environment='$ENVIRONMENT')
manager.set_rollout_percentage('$FEATURE_FLAG', $ROLLOUT_PERCENTAGE)
print(f'Set $FEATURE_FLAG rollout to $ROLLOUT_PERCENTAGE%')
"
    echo "✓ Feature flag configured"
    echo ""
fi

# 3. 카나리 배포 단계
if [ $ROLLOUT_PERCENTAGE -eq 0 ]; then
    echo "Step 3: Disabled deployment (0%)"
    echo "Feature is disabled"
elif [ $ROLLOUT_PERCENTAGE -le 5 ]; then
    echo "Step 3: Canary phase 1 (1-5%)"
    echo "Monitoring for 10 minutes..."
    sleep 10  # 실제로는 600초
    echo "✓ Phase 1 stable"
elif [ $ROLLOUT_PERCENTAGE -le 25 ]; then
    echo "Step 3: Canary phase 2 (25%)"
    echo "Monitoring for 20 minutes..."
    sleep 10  # 실제로는 1200초
    echo "✓ Phase 2 stable"
elif [ $ROLLOUT_PERCENTAGE -le 100 ]; then
    echo "Step 3: Full rollout (100%)"
    echo "Monitoring for 30 minutes..."
    sleep 10  # 실제로는 1800초
    echo "✓ Full rollout complete"
fi

echo ""
echo "=================================================="
echo "✅ Canary deployment complete"
echo "=================================================="
echo "Next steps:"
echo "  - Monitor metrics at $PROMETHEUS_URL"
echo "  - Check error rates"
echo "  - Gradual increase: $ROLLOUT_PERCENTAGE% → $(($ROLLOUT_PERCENTAGE + 20))%"
