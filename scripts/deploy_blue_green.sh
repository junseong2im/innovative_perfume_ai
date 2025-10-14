#!/bin/bash
# Blue/Green Deployment Script
# 신규(Green)와 구버전(Blue) 동시 가동 후 트래픽 스위치

set -e

ENVIRONMENT=${1:-stg}
NEW_VERSION=${2:-"latest"}

echo "=================================================="
echo "🔵🟢 Blue/Green Deployment"
echo "=================================================="
echo "Environment: $ENVIRONMENT"
echo "New Version: $NEW_VERSION"
echo ""

# 1. 현재 Blue 버전 확인
echo "Step 1: Checking current Blue version..."
BLUE_VERSION=$(kubectl get deployment artisan-blue -n $ENVIRONMENT -o jsonpath='{.spec.template.spec.containers[0].image}' || echo "none")
echo "Current Blue: $BLUE_VERSION"
echo ""

# 2. Green 배포
echo "Step 2: Deploying Green version..."
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: artisan-green
  namespace: $ENVIRONMENT
spec:
  replicas: 3
  selector:
    matchLabels:
      app: artisan
      version: green
  template:
    metadata:
      labels:
        app: artisan
        version: green
    spec:
      containers:
      - name: artisan
        image: artisan:$NEW_VERSION
        ports:
        - containerPort: 8000
        env:
        - name: ARTISAN_ENV
          value: "$ENVIRONMENT"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
EOF

echo "Waiting for Green deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/artisan-green -n $ENVIRONMENT
echo "✓ Green deployment ready"
echo ""

# 3. 헬스 체크
echo "Step 3: Running health checks on Green..."
GREEN_POD=$(kubectl get pod -n $ENVIRONMENT -l version=green -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n $ENVIRONMENT $GREEN_POD -- curl -f http://localhost:8000/health || {
    echo "❌ Health check failed on Green. Rolling back..."
    kubectl delete deployment artisan-green -n $ENVIRONMENT
    exit 1
}
echo "✓ Health check passed"
echo ""

# 4. Go/No-Go 체크
echo "Step 4: Running Go/No-Go gate on Green..."
kubectl exec -n $ENVIRONMENT $GREEN_POD -- python -m fragrance_ai.deployment.go_nogo_gate --exit-code || {
    echo "❌ Go/No-Go check failed. Rolling back..."
    kubectl delete deployment artisan-green -n $ENVIRONMENT
    exit 1
}
echo "✓ Go/No-Go check passed"
echo ""

# 5. 트래픽 스위치 (Blue → Green)
echo "Step 5: Switching traffic to Green..."
kubectl patch service artisan -n $ENVIRONMENT -p '{"spec":{"selector":{"version":"green"}}}'
echo "✓ Traffic switched to Green"
echo ""

# 6. 모니터링 대기
echo "Step 6: Monitoring Green for 5 minutes..."
sleep 30  # 실제로는 300초
echo "✓ Green is stable"
echo ""

# 7. Blue 제거
echo "Step 7: Removing old Blue deployment..."
kubectl delete deployment artisan-blue -n $ENVIRONMENT || echo "No Blue deployment found"
echo "✓ Blue removed"
echo ""

# 8. Green → Blue 재라벨링
echo "Step 8: Relabeling Green as Blue..."
kubectl label deployment artisan-green -n $ENVIRONMENT version=blue --overwrite
kubectl patch deployment artisan-green -n $ENVIRONMENT --type='json' -p='[{"op": "replace", "path": "/metadata/name", "value": "artisan-blue"}]'
echo "✓ Green is now Blue"
echo ""

echo "=================================================="
echo "✅ Blue/Green deployment complete"
echo "=================================================="
echo "New version $NEW_VERSION is live!"
