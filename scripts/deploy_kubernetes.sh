#!/bin/bash

# ==============================================================================
# Fragrance AI Kubernetes 배포 스크립트
# Production-ready Kubernetes deployment automation
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
K8S_DIR="$PROJECT_ROOT/kubernetes"

# Configuration
NAMESPACE="fragrance-ai"
ENVIRONMENT="production"
VERSION_TAG="latest"
DRY_RUN=false
FORCE=false
SKIP_BUILD=false
MONITORING=true

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
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

# Help function
show_help() {
    cat << EOF
Fragrance AI Kubernetes Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -n, --namespace NAMESPACE     Kubernetes namespace (default: fragrance-ai)
    -e, --environment ENV         Environment (dev, staging, production)
    -v, --version VERSION         Docker image version/tag
    --dry-run                     Show what would be deployed
    --skip-build                  Skip Docker image building
    --no-monitoring              Skip monitoring stack deployment
    --force                       Force deployment even if validation fails
    -h, --help                    Show this help message

EXAMPLES:
    $0 --environment production --version v1.2.3
    $0 --dry-run --namespace fragrance-ai-staging
    $0 --skip-build --no-monitoring

EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -v|--version)
                VERSION_TAG="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --no-monitoring)
                MONITORING=false
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi

    # Check docker
    if ! command -v docker &> /dev/null && [[ "$SKIP_BUILD" != "true" ]]; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    # Check if namespace exists
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE exists"
    else
        log_info "Namespace $NAMESPACE will be created"
    fi

    log_success "Prerequisites check passed"
}

# Build Docker image
build_image() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_info "Skipping Docker image build"
        return 0
    fi

    log_info "Building Docker image: fragrance-ai:$VERSION_TAG"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would build Docker image"
        return 0
    fi

    cd "$PROJECT_ROOT"
    docker build -t "fragrance-ai:$VERSION_TAG" -f docker/Dockerfile.production .

    # Tag for registry if needed
    if [[ -n "${DOCKER_REGISTRY:-}" ]]; then
        docker tag "fragrance-ai:$VERSION_TAG" "$DOCKER_REGISTRY/fragrance-ai:$VERSION_TAG"
        docker push "$DOCKER_REGISTRY/fragrance-ai:$VERSION_TAG"
    fi

    log_success "Docker image built successfully"
}

# Generate Kubernetes manifests
generate_manifests() {
    log_info "Generating Kubernetes manifests..."

    local temp_dir="/tmp/fragrance-ai-k8s-$$"
    mkdir -p "$temp_dir"

    # Copy base manifests
    cp -r "$K8S_DIR"/* "$temp_dir/"

    # Replace variables in manifests
    find "$temp_dir" -name "*.yaml" -exec sed -i.bak \
        -e "s/{{NAMESPACE}}/$NAMESPACE/g" \
        -e "s/{{ENVIRONMENT}}/$ENVIRONMENT/g" \
        -e "s/{{VERSION_TAG}}/$VERSION_TAG/g" \
        -e "s/{{DOCKER_REGISTRY}}/${DOCKER_REGISTRY:-}/g" {} \;

    # Remove backup files
    find "$temp_dir" -name "*.bak" -delete

    echo "$temp_dir"
}

# Validate manifests
validate_manifests() {
    local manifest_dir="$1"
    log_info "Validating Kubernetes manifests..."

    for file in "$manifest_dir"/*.yaml; do
        if [[ -f "$file" ]]; then
            if ! kubectl apply --dry-run=client -f "$file" &> /dev/null; then
                log_error "Invalid manifest: $(basename "$file")"
                exit 1
            fi
        fi
    done

    log_success "Manifest validation passed"
}

# Create namespace if it doesn't exist
create_namespace() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would create namespace if needed"
        return 0
    fi

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Creating namespace: $NAMESPACE"
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" environment="$ENVIRONMENT"
    fi
}

# Deploy secrets
deploy_secrets() {
    local manifest_dir="$1"
    log_info "Deploying secrets..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy secrets"
        return 0
    fi

    # Check if secrets file exists
    local secrets_file="$manifest_dir/secrets.yaml"
    if [[ ! -f "$secrets_file" ]]; then
        log_warning "No secrets file found, using default secrets from deployment manifest"
        return 0
    fi

    kubectl apply -f "$secrets_file" -n "$NAMESPACE"
    log_success "Secrets deployed"
}

# Deploy ConfigMaps
deploy_configmaps() {
    local manifest_dir="$1"
    log_info "Deploying ConfigMaps..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy ConfigMaps"
        return 0
    fi

    # Apply main deployment which includes ConfigMaps
    kubectl apply -f "$manifest_dir/fragrance-ai-deployment.yaml" -n "$NAMESPACE" --prune --selector="app=fragrance-ai"

    log_success "ConfigMaps deployed"
}

# Deploy storage
deploy_storage() {
    local manifest_dir="$1"
    log_info "Deploying persistent storage..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy storage"
        return 0
    fi

    # Storage is included in the main deployment manifest
    log_success "Storage configuration applied"
}

# Deploy applications
deploy_applications() {
    local manifest_dir="$1"
    log_info "Deploying applications..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy applications"
        return 0
    fi

    # Apply main deployment
    kubectl apply -f "$manifest_dir/fragrance-ai-deployment.yaml" -n "$NAMESPACE"

    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/postgres -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/redis -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/chromadb -n "$NAMESPACE" --timeout=300s
    kubectl rollout status deployment/fragrance-ai-app -n "$NAMESPACE" --timeout=600s

    log_success "Applications deployed successfully"
}

# Deploy monitoring stack
deploy_monitoring() {
    if [[ "$MONITORING" != "true" ]]; then
        log_info "Skipping monitoring stack deployment"
        return 0
    fi

    log_info "Deploying monitoring stack..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy monitoring stack"
        return 0
    fi

    # Deploy Prometheus
    if [[ -f "$K8S_DIR/monitoring/prometheus.yaml" ]]; then
        kubectl apply -f "$K8S_DIR/monitoring/prometheus.yaml" -n "$NAMESPACE"
    fi

    # Deploy Grafana
    if [[ -f "$K8S_DIR/monitoring/grafana.yaml" ]]; then
        kubectl apply -f "$K8S_DIR/monitoring/grafana.yaml" -n "$NAMESPACE"
    fi

    log_success "Monitoring stack deployed"
}

# Perform health checks
health_check() {
    log_info "Performing health checks..."

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would perform health checks"
        return 0
    fi

    local max_attempts=30
    local attempt=1

    # Check if pods are running
    while [[ $attempt -le $max_attempts ]]; do
        local ready_pods=$(kubectl get pods -n "$NAMESPACE" -l app=fragrance-ai-app --field-selector=status.phase=Running | wc -l)
        local total_pods=$(kubectl get pods -n "$NAMESPACE" -l app=fragrance-ai-app | tail -n +2 | wc -l)

        if [[ $ready_pods -gt 0 ]] && [[ $ready_pods -eq $total_pods ]]; then
            log_success "All pods are running"
            break
        fi

        log_info "Waiting for pods to be ready ($attempt/$max_attempts)..."
        sleep 10
        ((attempt++))
    done

    if [[ $attempt -gt $max_attempts ]]; then
        log_error "Health check failed: Pods not ready after $max_attempts attempts"
        if [[ "$FORCE" != "true" ]]; then
            exit 1
        fi
    fi

    # Check service endpoints
    local service_ip=$(kubectl get service fragrance-ai-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    if [[ -n "$service_ip" ]]; then
        log_success "Service is accessible at $service_ip"
    fi

    log_success "Health checks passed"
}

# Display deployment info
show_deployment_info() {
    log_info "Deployment completed successfully!"

    echo
    echo "=============================================="
    echo "Fragrance AI Kubernetes Deployment Summary"
    echo "=============================================="
    echo "Namespace: $NAMESPACE"
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION_TAG"
    echo "Timestamp: $(date)"
    echo "=============================================="

    if [[ "$DRY_RUN" != "true" ]]; then
        # Show pods
        echo "Pods:"
        kubectl get pods -n "$NAMESPACE" -o wide

        echo
        echo "Services:"
        kubectl get services -n "$NAMESPACE"

        echo
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE" 2>/dev/null || echo "No ingress found"

        echo
        echo "Access Information:"
        local external_ip=$(kubectl get service fragrance-ai-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "Pending")
        echo "  External IP: $external_ip"
        echo "  API Port: 8000"
        echo "  Metrics Port: 9090"

        if [[ "$MONITORING" == "true" ]]; then
            echo "  Grafana: Available in monitoring namespace"
            echo "  Prometheus: Available in monitoring namespace"
        fi
    fi

    echo "=============================================="
}

# Cleanup function
cleanup() {
    local manifest_dir="$1"
    if [[ -d "$manifest_dir" ]]; then
        rm -rf "$manifest_dir"
    fi
}

# Main deployment function
main() {
    parse_args "$@"

    log_info "Starting Kubernetes deployment for Fragrance AI"
    log_info "Namespace: $NAMESPACE, Environment: $ENVIRONMENT, Version: $VERSION_TAG"

    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi

    # Pre-deployment checks
    check_prerequisites

    # Build image
    build_image

    # Generate manifests
    local manifest_dir
    manifest_dir=$(generate_manifests)

    # Set up cleanup trap
    trap "cleanup '$manifest_dir'" EXIT

    # Validate manifests
    validate_manifests "$manifest_dir"

    # Deploy components
    create_namespace
    deploy_secrets "$manifest_dir"
    deploy_configmaps "$manifest_dir"
    deploy_storage "$manifest_dir"
    deploy_applications "$manifest_dir"
    deploy_monitoring

    # Post-deployment verification
    health_check

    # Show deployment info
    show_deployment_info

    log_success "Kubernetes deployment completed successfully!"
}

# Run main function
main "$@"