#!/bin/bash
# ============================================================================
# NGINX Weight Update Script
# Updates upstream weights for canary deployment
# ============================================================================
#
# Usage:
#   ./scripts/update_nginx_weights.sh <canary_percentage>
#
# Example:
#   ./scripts/update_nginx_weights.sh 5   # 5% canary, 95% production
#   ./scripts/update_nginx_weights.sh 25  # 25% canary, 75% production
#   ./scripts/update_nginx_weights.sh 100 # 100% canary, 0% production
#
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
if [ $# -lt 1 ]; then
    log_error "Usage: $0 <canary_percentage>"
    echo ""
    echo "Examples:"
    echo "  $0 1    # 1% canary traffic"
    echo "  $0 5    # 5% canary traffic"
    echo "  $0 25   # 25% canary traffic"
    echo "  $0 100  # 100% canary traffic (promote to production)"
    exit 1
fi

CANARY_WEIGHT=$1

# Validate percentage
if ! [[ $CANARY_WEIGHT =~ ^[0-9]+$ ]]; then
    log_error "Canary percentage must be a number"
    exit 1
fi

if [ $CANARY_WEIGHT -lt 0 ] || [ $CANARY_WEIGHT -gt 100 ]; then
    log_error "Canary percentage must be between 0 and 100"
    exit 1
fi

# Calculate production weight
PRODUCTION_WEIGHT=$((100 - CANARY_WEIGHT))

log_info "=========================================="
log_info "Updating NGINX Traffic Weights"
log_info "=========================================="
log_info "Production: ${PRODUCTION_WEIGHT}%"
log_info "Canary: ${CANARY_WEIGHT}%"
log_info "=========================================="

# Find NGINX container
NGINX_CONTAINER=$(docker ps --filter "name=nginx" --format "{{.Names}}" | head -1)

if [ -z "$NGINX_CONTAINER" ]; then
    log_error "NGINX container not found"
    log_info "Start with: docker-compose -f docker-compose.production.yml up -d nginx"
    exit 1
fi

log_success "Found NGINX container: $NGINX_CONTAINER"

# Template file location
TEMPLATE_FILE="nginx/conf.d/upstream.conf.template"
OUTPUT_FILE="nginx/conf.d/upstream.conf"

if [ ! -f "$TEMPLATE_FILE" ]; then
    log_error "Template file not found: $TEMPLATE_FILE"
    exit 1
fi

log_info "Generating upstream configuration from template..."

# Replace variables in template
export PRODUCTION_WEIGHT
export CANARY_WEIGHT
envsubst '${PRODUCTION_WEIGHT} ${CANARY_WEIGHT}' < "$TEMPLATE_FILE" > "$OUTPUT_FILE"

log_success "Generated upstream configuration: $OUTPUT_FILE"

# Show the updated configuration
log_info "Updated upstream configuration:"
echo "----------------------------------------"
grep -A 15 "upstream app_backend" "$OUTPUT_FILE" || true
echo "----------------------------------------"

# Copy to NGINX container
log_info "Copying configuration to NGINX container..."

docker cp "$OUTPUT_FILE" "$NGINX_CONTAINER:/etc/nginx/conf.d/upstream.conf"

log_success "Configuration copied to container"

# Test NGINX configuration
log_info "Testing NGINX configuration..."

if docker exec "$NGINX_CONTAINER" nginx -t 2>&1 | grep -q "successful"; then
    log_success "NGINX configuration is valid"
else
    log_error "NGINX configuration test failed"
    docker exec "$NGINX_CONTAINER" nginx -t
    exit 1
fi

# Reload NGINX
log_info "Reloading NGINX..."

docker exec "$NGINX_CONTAINER" nginx -s reload

if [ $? -eq 0 ]; then
    log_success "NGINX reloaded successfully"
else
    log_error "Failed to reload NGINX"
    exit 1
fi

# Verify reload
sleep 2

if docker exec "$NGINX_CONTAINER" pgrep nginx > /dev/null; then
    log_success "NGINX is running"
else
    log_error "NGINX is not running after reload"
    exit 1
fi

# Summary
echo ""
log_success "=========================================="
log_success "Traffic weights updated successfully!"
log_success "=========================================="
echo ""
log_info "Current traffic distribution:"
echo "  Production: ${PRODUCTION_WEIGHT}%"
echo "  Canary:     ${CANARY_WEIGHT}%"
echo ""
log_info "Monitor traffic with:"
echo "  docker exec $NGINX_CONTAINER tail -f /var/log/nginx/canary.log"
echo ""
log_info "Check NGINX status:"
echo "  curl http://localhost:8080/nginx_status"
echo ""
