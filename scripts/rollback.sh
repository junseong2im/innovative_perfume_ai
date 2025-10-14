#!/bin/bash
# ============================================================================
# Rollback Script for Fragrance AI
# ============================================================================
#
# Usage:
#   ./scripts/rollback.sh [version]
#
# Examples:
#   ./scripts/rollback.sh v0.1.0
#   ./scripts/rollback.sh  # Uses last backup
#
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.production.yml"
BACKUP_DIR="${PROJECT_ROOT}/backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

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

prompt_confirm() {
    read -p "$(echo -e "${YELLOW}$1 (y/n):${NC} ")" -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]]
}

find_latest_backup() {
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A $BACKUP_DIR 2>/dev/null)" ]; then
        log_error "No backups found in $BACKUP_DIR"
        return 1
    fi

    LATEST_BACKUP=$(ls -t "$BACKUP_DIR"/backup_*.sql 2>/dev/null | head -1)

    if [ -z "$LATEST_BACKUP" ]; then
        log_error "No backup files found"
        return 1
    fi

    echo "$LATEST_BACKUP"
}

list_available_versions() {
    log_info "Available image versions:"
    docker images | grep fragrance-ai | awk '{print $2}' | sort -u
}

stop_services() {
    log_info "Stopping services..."

    docker-compose -f "$COMPOSE_FILE" down

    log_success "Services stopped"
}

restore_database() {
    local backup_file="$1"

    log_info "Restoring database from: $backup_file"

    # Start only postgres
    docker-compose -f "$COMPOSE_FILE" up -d postgres
    sleep 10

    # Drop existing connections
    log_info "Dropping existing database connections..."
    docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U fragrance_user -d postgres <<EOF
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'fragrance_ai' AND pid <> pg_backend_pid();
EOF

    # Restore database
    log_info "Restoring database..."
    cat "$backup_file" | docker-compose -f "$COMPOSE_FILE" exec -T postgres \
        psql -U fragrance_user -d fragrance_ai

    log_success "Database restored successfully"
}

rollback_to_version() {
    local version="$1"

    log_info "Rolling back to version: $version"

    # Export version
    export VERSION="$version"

    # Start services with previous version
    docker-compose -f "$COMPOSE_FILE" up -d

    log_success "Rollback to version $version completed"
}

verify_rollback() {
    log_info "Verifying rollback..."

    # Wait for services to be ready
    sleep 15

    # Check health endpoint
    if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
        log_success "API health check passed"
    else
        log_error "API health check failed"
        return 1
    fi

    # Check database
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U fragrance_user > /dev/null 2>&1; then
        log_success "Database connection check passed"
    else
        log_error "Database connection check failed"
        return 1
    fi

    log_success "Rollback verification passed"
}

# ----------------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------------

main() {
    local target_version="${1:-}"
    local use_db_backup=false

    log_warning "========================================="
    log_warning "Fragrance AI Rollback"
    log_warning "========================================="

    # If no version specified, show available versions
    if [ -z "$target_version" ]; then
        list_available_versions
        read -p "Enter version to rollback to: " target_version
    fi

    # Confirm rollback
    if ! prompt_confirm "⚠️  Are you sure you want to rollback to version $target_version?"; then
        log_info "Rollback cancelled"
        exit 0
    fi

    # Ask about database rollback
    if prompt_confirm "Do you want to restore database from backup?"; then
        use_db_backup=true

        # Find latest backup
        BACKUP_FILE=$(find_latest_backup)
        if [ $? -ne 0 ]; then
            log_error "Cannot proceed without backup"
            exit 1
        fi

        log_info "Will use backup: $BACKUP_FILE"

        if ! prompt_confirm "Confirm using this backup?"; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi

    # Stop services
    stop_services

    # Restore database if requested
    if [ "$use_db_backup" = true ]; then
        restore_database "$BACKUP_FILE"
    fi

    # Rollback to version
    rollback_to_version "$target_version"

    # Verify rollback
    if verify_rollback; then
        log_success "========================================="
        log_success "Rollback completed successfully!"
        log_success "========================================="

        # Show status
        docker-compose -f "$COMPOSE_FILE" ps

    else
        log_error "========================================="
        log_error "Rollback verification failed!"
        log_error "========================================="
        log_error "Manual intervention required"
        exit 1
    fi
}

# Run main function
main "$@"
