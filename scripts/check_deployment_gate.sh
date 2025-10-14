#!/bin/bash
# Go / No-Go 배포 게이트 체크 스크립트

set -e

echo "=================================================="
echo "🚦 Go / No-Go Deployment Gate"
echo "=================================================="
echo ""

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"

# Python 실행
python -m fragrance_ai.deployment.go_nogo_gate \
    --prometheus-url "$PROMETHEUS_URL" \
    --report-file deployment_gate_report.txt \
    --exit-code

# Exit code 처리
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ GO: Safe to deploy"
    exit 0
else
    echo ""
    echo "⛔ NO-GO: Do not deploy"
    echo "Check deployment_gate_report.txt for details"
    exit 1
fi
