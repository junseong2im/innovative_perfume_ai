#!/bin/bash

# ===========================
# Fragrance AI Production Deployment Script
# ===========================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
DEPLOY_DIR="/opt/fragrance-ai"
BACKUP_DIR="/opt/backups/fragrance-ai"
LOG_FILE="/var/log/fragrance-ai-deploy.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a $LOG_FILE
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOG_FILE
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOG_FILE
}

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    error "Please run as root or with sudo"
fi

log "Starting deployment for environment: $ENVIRONMENT"

# 1. Pre-deployment checks
log "Running pre-deployment checks..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker is not installed"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    error "Docker Compose is not installed"
fi

# Check disk space
AVAILABLE_SPACE=$(df -h /opt | awk 'NR==2 {print $4}' | sed 's/G//')
if (( $(echo "$AVAILABLE_SPACE < 10" | bc -l) )); then
    warning "Low disk space: ${AVAILABLE_SPACE}G available"
fi

# 2. Create backup
log "Creating backup..."
mkdir -p $BACKUP_DIR
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
BACKUP_FILE="$BACKUP_DIR/backup_$TIMESTAMP.tar.gz"

if [ -d "$DEPLOY_DIR" ]; then
    tar -czf $BACKUP_FILE \
        --exclude="$DEPLOY_DIR/logs" \
        --exclude="$DEPLOY_DIR/models" \
        --exclude="$DEPLOY_DIR/.git" \
        $DEPLOY_DIR 2>/dev/null || true
    log "Backup created: $BACKUP_FILE"
else
    log "No existing deployment found, skipping backup"
fi

# 3. Pull latest code
log "Pulling latest code..."
if [ ! -d "$DEPLOY_DIR" ]; then
    git clone https://github.com/yourusername/fragrance-ai.git $DEPLOY_DIR
    cd $DEPLOY_DIR
else
    cd $DEPLOY_DIR
    git stash
    git pull origin main
fi

# 4. Copy production configuration
log "Setting up production configuration..."
if [ -f ".env.production" ]; then
    cp .env.production .env
else
    error ".env.production file not found"
fi

# 5. Build and start services
log "Building Docker images..."
docker-compose -f docker-compose.production.yml build

log "Stopping old containers..."
docker-compose -f docker-compose.production.yml down

log "Starting new containers..."
docker-compose -f docker-compose.production.yml up -d

# 6. Run database migrations
log "Running database migrations..."
sleep 10  # Wait for database to be ready
docker-compose -f docker-compose.production.yml exec -T api alembic upgrade head

# 7. Load initial data if needed
if [ "$2" == "--init-data" ]; then
    log "Loading initial data..."
    docker-compose -f docker-compose.production.yml exec -T api python scripts/load_initial_data.py
fi

# 8. Health check
log "Performing health check..."
sleep 15  # Wait for services to fully start

MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8001/health > /dev/null 2>&1; then
        log "Health check passed!"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        warning "Health check failed, retry $RETRY_COUNT/$MAX_RETRIES"
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    error "Health check failed after $MAX_RETRIES attempts"
fi

# 9. Clean up
log "Cleaning up..."
docker system prune -f
find $BACKUP_DIR -name "backup_*.tar.gz" -mtime +7 -delete

# 10. Show status
log "Deployment complete! Showing service status:"
docker-compose -f docker-compose.production.yml ps

echo ""
log "Access points:"
echo "  - API: http://localhost:8001"
echo "  - API Docs: http://localhost:8001/docs"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000"
echo ""
log "To view logs: docker-compose -f docker-compose.production.yml logs -f"