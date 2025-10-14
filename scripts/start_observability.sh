#!/bin/bash
# Start Observability Stack
# Prometheus + Grafana + Loki + Tempo + Jaeger

set -e

echo "=========================================================="
echo "Starting Artisan Observability Stack"
echo "=========================================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Navigate to docker directory
cd "$(dirname "$0")/../docker"

# Create necessary directories
echo "Creating directories..."
mkdir -p prometheus loki promtail tempo alertmanager grafana/datasources logs
echo "✓ Directories created"
echo ""

# Start observability stack
echo "Starting observability services..."
docker-compose -f docker-compose.observability.yml up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."

# Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✓ Prometheus is healthy"
else
    echo "⚠️  Prometheus is not responding"
fi

# Grafana
if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✓ Grafana is healthy"
else
    echo "⚠️  Grafana is not responding"
fi

# Loki
if curl -s http://localhost:3100/ready > /dev/null 2>&1; then
    echo "✓ Loki is healthy"
else
    echo "⚠️  Loki is not responding"
fi

# Jaeger
if curl -s http://localhost:16686 > /dev/null 2>&1; then
    echo "✓ Jaeger is healthy"
else
    echo "⚠️  Jaeger is not responding"
fi

# Tempo
if curl -s http://localhost:3200/ready > /dev/null 2>&1; then
    echo "✓ Tempo is healthy"
else
    echo "⚠️  Tempo is not responding"
fi

echo ""
echo "=========================================================="
echo "✅ Observability Stack Started"
echo "=========================================================="
echo ""
echo "Access URLs:"
echo "  - Prometheus:  http://localhost:9090"
echo "  - Grafana:     http://localhost:3000  (admin/admin)"
echo "  - Loki:        http://localhost:3100"
echo "  - Jaeger UI:   http://localhost:16686"
echo "  - Tempo:       http://localhost:3200"
echo "  - AlertManager: http://localhost:9093"
echo ""
echo "Endpoints for instrumentation:"
echo "  - Prometheus metrics: http://localhost:8000/metrics"
echo "  - Jaeger agent:       localhost:6831 (UDP)"
echo "  - OTLP gRPC:          localhost:4317"
echo "  - OTLP HTTP:          localhost:4318"
echo ""
echo "To stop the stack:"
echo "  docker-compose -f docker-compose.observability.yml down"
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.observability.yml logs -f [service]"
echo ""
