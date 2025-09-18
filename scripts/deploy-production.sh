#!/bin/bash
# ================================================================================
# Fragrance AI - Production Deployment Script
# ================================================================================
# This script handles secure production deployment with health checks
# ================================================================================

set -euo pipefail  # Exit on any error, undefined variable, or pipe failure

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="docker-compose.production.yml"
ENV_FILE=".env.production"
BACKUP_DIR="/opt/fragrance_ai/backups"
LOG_FILE="/var/log/fragrance_ai_deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi

    success "Prerequisites check passed"
}

# Check environment file
check_environment() {
    log "Checking environment configuration..."

    if [[ ! -f "$PROJECT_ROOT/$ENV_FILE" ]]; then
        error "Production environment file not found: $ENV_FILE"
        error "Please copy .env.production.template to $ENV_FILE and configure it"
        exit 1
    fi

    # Source environment file to check variables
    set -a
    source "$PROJECT_ROOT/$ENV_FILE"
    set +a

    # Check critical variables
    local critical_vars=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "SECRET_KEY"
        "GRAFANA_PASSWORD"
    )

    for var in "${critical_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Critical environment variable $var is not set"
            exit 1
        fi

        # Check for default/weak values
        case "$var" in
            "POSTGRES_PASSWORD"|"REDIS_PASSWORD"|"GRAFANA_PASSWORD")
                if [[ "${!var}" == *"password"* ]] || [[ "${!var}" == "admin" ]]; then
                    error "$var appears to use a default/weak value"
                    exit 1
                fi
                ;;
            "SECRET_KEY")
                if [[ ${#SECRET_KEY} -lt 32 ]]; then
                    error "SECRET_KEY should be at least 32 characters long"
                    exit 1
                fi
                ;;
        esac
    done

    success "Environment configuration is valid"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."

    local directories=(
        "/opt/fragrance_ai"
        "/opt/fragrance_ai/postgres_data"
        "/opt/fragrance_ai/redis_data"
        "/opt/fragrance_ai/chroma_data"
        "/opt/fragrance_ai/prometheus_data"
        "/opt/fragrance_ai/grafana_data"
        "$BACKUP_DIR"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/models"
        "$PROJECT_ROOT/checkpoints"
    )

    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            sudo mkdir -p "$dir"
            sudo chown $(id -u):$(id -g) "$dir"
            log "Created directory: $dir"
        fi
    done

    success "Directories created successfully"
}

# Backup existing data
backup_data() {
    if [[ "$1" == "--skip-backup" ]]; then
        warning "Skipping backup as requested"
        return 0
    fi

    log "Creating backup of existing data..."

    local backup_timestamp=$(date +'%Y%m%d_%H%M%S')
    local backup_path="$BACKUP_DIR/backup_$backup_timestamp"

    mkdir -p "$backup_path"

    # Backup database if it exists
    if docker ps | grep -q fragrance_ai_postgres; then
        log "Backing up PostgreSQL database..."
        docker exec fragrance_ai_postgres_prod pg_dump \
            -U fragrance_ai -d fragrance_ai > "$backup_path/database.sql" || true
    fi

    # Backup application data
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        log "Backing up application data..."
        cp -r "$PROJECT_ROOT/data" "$backup_path/" || true
    fi

    success "Backup completed: $backup_path"
}

# Build Docker images
build_images() {
    log "Building Docker images..."

    cd "$PROJECT_ROOT"

    # Build with no cache to ensure latest updates
    docker-compose -f "$COMPOSE_FILE" build \
        --no-cache \
        --parallel \
        --progress=plain \
        fragrance_ai

    success "Docker images built successfully"
}

# Pull latest images
pull_images() {
    log "Pulling latest base images..."

    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" pull \
        postgres \
        redis \
        chroma \
        nginx \
        prometheus \
        grafana

    success "Latest images pulled successfully"
}

# Deploy services
deploy_services() {
    log "Deploying services..."

    cd "$PROJECT_ROOT"

    # Stop existing services gracefully
    if docker-compose -f "$COMPOSE_FILE" ps | grep -q "Up"; then
        log "Stopping existing services..."
        docker-compose -f "$COMPOSE_FILE" down --timeout 30
    fi

    # Start services with rolling update strategy
    log "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d \
        --remove-orphans \
        --force-recreate

    success "Services deployed successfully"
}

# Health checks
health_checks() {
    log "Performing health checks..."

    local max_attempts=30
    local attempt=0

    # Check database
    log "Checking PostgreSQL health..."
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U fragrance_ai -d fragrance_ai &> /dev/null; then
            success "PostgreSQL is healthy"
            break
        fi
        ((attempt++))
        sleep 10
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "PostgreSQL health check failed"
        return 1
    fi

    # Check Redis
    log "Checking Redis health..."
    attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping &> /dev/null; then
            success "Redis is healthy"
            break
        fi
        ((attempt++))
        sleep 5
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "Redis health check failed"
        return 1
    fi

    # Check main application
    log "Checking main application health..."
    attempt=0
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s -m 10 http://localhost:8080/health &> /dev/null; then
            success "Main application is healthy"
            break
        fi
        ((attempt++))
        sleep 10
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "Main application health check failed"
        return 1
    fi

    success "All health checks passed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring and alerting..."

    # Wait for Prometheus to be ready
    sleep 30

    # Check if Prometheus is accessible
    if curl -f -s http://localhost:9090/-/healthy &> /dev/null; then
        success "Prometheus is running"
    else
        warning "Prometheus may not be accessible"
    fi

    # Check if Grafana is accessible
    if curl -f -s http://localhost:3000/api/health &> /dev/null; then
        success "Grafana is running"
    else
        warning "Grafana may not be accessible"
    fi

    success "Monitoring setup completed"
}

# Security checks
security_checks() {
    log "Performing security checks..."

    # Check file permissions
    local sensitive_files=(
        "$PROJECT_ROOT/$ENV_FILE"
        "$PROJECT_ROOT/nginx/ssl"
    )

    for file in "${sensitive_files[@]}"; do
        if [[ -e "$file" ]]; then
            local perms=$(stat -c "%a" "$file")
            if [[ "$perms" != "600" ]] && [[ "$perms" != "700" ]]; then
                warning "File $file has permissive permissions: $perms"
            fi
        fi
    done

    # Check running containers are not running as root
    log "Checking container user privileges..."
    local containers=$(docker-compose -f "$COMPOSE_FILE" ps -q)
    for container in $containers; do
        local user=$(docker exec "$container" whoami 2>/dev/null || echo "unknown")
        if [[ "$user" == "root" ]]; then
            warning "Container $container is running as root"
        fi
    done

    success "Security checks completed"
}

# Cleanup old resources
cleanup() {
    log "Cleaning up old resources..."

    # Remove unused images
    docker image prune -f --filter "until=72h"

    # Remove unused volumes (be careful with this)
    # docker volume prune -f --filter "label!=keep"

    # Clean up old backups (keep last 7 days)
    find "$BACKUP_DIR" -name "backup_*" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true

    success "Cleanup completed"
}

# Display deployment info
show_deployment_info() {
    log "Deployment completed successfully!"
    echo ""
    echo "=================================="
    echo "  Fragrance AI Production Status"
    echo "=================================="
    echo "🌐 Main Application: http://localhost:8080"
    echo "📊 Grafana Dashboard: http://localhost:3000"
    echo "📈 Prometheus Metrics: http://localhost:9090"
    echo "🌸 API Documentation: http://localhost:8080/api/v2/docs"
    echo ""
    echo "Default credentials (CHANGE IMMEDIATELY):"
    echo "  Grafana: admin / (check $ENV_FILE)"
    echo ""
    echo "Services status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    echo "To view logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "To stop: docker-compose -f $COMPOSE_FILE down"
    echo ""
}

# Main deployment function
main() {
    local skip_backup=false
    local skip_build=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-backup)
                skip_backup=true
                shift
                ;;
            --skip-build)
                skip_build=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-backup  Skip data backup"
                echo "  --skip-build   Skip Docker image building"
                echo "  -h, --help     Show this help message"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    log "Starting Fragrance AI production deployment..."

    # Run deployment steps
    check_root
    check_prerequisites
    check_environment
    create_directories

    if [[ "$skip_backup" != true ]]; then
        backup_data
    else
        backup_data --skip-backup
    fi

    pull_images

    if [[ "$skip_build" != true ]]; then
        build_images
    fi

    deploy_services
    health_checks
    setup_monitoring
    security_checks
    cleanup
    show_deployment_info

    success "Production deployment completed successfully!"
}

# Error handling
trap 'error "Deployment failed on line $LINENO. Check logs at $LOG_FILE"' ERR

# Run main function
main "$@"