#!/bin/bash

# ==============================================================================
# Fragrance AI 프로덕션 배포 스크립트 v3.0
# Enterprise-grade deployment automation with advanced features
# ==============================================================================

set -euo pipefail

VERSION="3.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
ENVIRONMENT="production"
STRATEGY="rolling"
VERSION_TAG=""
DRY_RUN=false
BACKUP=true
HEALTH_CHECK=true
CLEANUP=false
FORCE=false
CONFIG_FILE=""
ROLLBACK_ON_FAILURE=true
NOTIFICATION_ENABLED=true

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 기본 설정
DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="backups"
DEPLOY_LOG="deploy_$(date +%Y%m%d_%H%M%S).log"

# 도움말 함수
show_help() {
    cat << EOF
Fragrance AI 배포 스크립트

사용법: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENT:
    development    개발 환경 배포 (기본값)
    staging        스테이징 환경 배포
    production     프로덕션 환경 배포

OPTIONS:
    --backup       배포 전 백업 수행
    --no-cache     Docker 이미지 캐시 무시
    --rolling      롤링 업데이트 수행
    --health-check 헬스체크 수행
    --cleanup      이전 이미지/컨테이너 정리
    --help         이 도움말 출력

예시:
    $0 production --backup --health-check
    $0 development --no-cache
    $0 staging --rolling --cleanup

EOF
}

# 인수 파싱
BACKUP=false
NO_CACHE=false
ROLLING=false
HEALTH_CHECK=false
CLEANUP=false

while [[ $# -gt 1 ]]; do
    case $2 in
        --backup)
            BACKUP=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --rolling)
            ROLLING=true
            shift
            ;;
        --health-check)
            HEALTH_CHECK=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "알 수 없는 옵션: $2"
            show_help
            exit 1
            ;;
    esac
done

# 배포 시작
log_info "Fragrance AI 배포 시작 - 환경: ${ENVIRONMENT}"
echo "배포 로그: ${DEPLOY_LOG}"

# 로그 디렉토리 생성
mkdir -p logs

# 모든 출력을 로그 파일에도 기록
exec > >(tee -a "logs/${DEPLOY_LOG}")
exec 2>&1

# 사전 요구사항 확인
check_prerequisites() {
    log_info "사전 요구사항 확인 중..."
    
    # Docker 확인
    if ! command -v docker &> /dev/null; then
        log_error "Docker가 설치되지 않았습니다."
        exit 1
    fi
    
    # Docker Compose 확인
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose가 설치되지 않았습니다."
        exit 1
    fi
    
    # 설정 파일 확인
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        log_error "설정 파일을 찾을 수 없습니다: ${CONFIG_FILE}"
        exit 1
    fi
    
    # 환경 변수 파일 확인
    if [[ ! -f ".env.${ENVIRONMENT}" ]]; then
        log_warning "환경 변수 파일이 없습니다: .env.${ENVIRONMENT}"
        log_info ".env.example을 복사하여 .env.${ENVIRONMENT}을 생성하세요."
    fi
    
    log_success "사전 요구사항 확인 완료"
}

# 환경 변수 로드
load_environment() {
    log_info "환경 변수 로드 중..."
    
    # .env 파일 로드
    if [[ -f ".env.${ENVIRONMENT}" ]]; then
        export $(cat ".env.${ENVIRONMENT}" | grep -v '^#' | xargs)
        log_success "환경 변수 로드 완료: .env.${ENVIRONMENT}"
    elif [[ -f ".env" ]]; then
        export $(cat ".env" | grep -v '^#' | xargs)
        log_success "환경 변수 로드 완료: .env"
    fi
    
    # 필수 환경 변수 확인
    required_vars=("SECRET_KEY" "DATABASE_URL")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            log_error "필수 환경 변수가 설정되지 않았습니다: ${var}"
            exit 1
        fi
    done
}

# 백업 수행
perform_backup() {
    if [[ "${BACKUP}" == true ]]; then
        log_info "백업 수행 중..."
        
        mkdir -p "${BACKUP_DIR}"
        backup_file="${BACKUP_DIR}/backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        
        # 데이터베이스 백업
        if command -v pg_dump &> /dev/null && [[ -n "${DATABASE_URL}" ]]; then
            log_info "PostgreSQL 데이터베이스 백업 중..."
            pg_dump "${DATABASE_URL}" > "${BACKUP_DIR}/db_backup_$(date +%Y%m%d_%H%M%S).sql"
        fi
        
        # 중요 파일들 백업
        tar -czf "${backup_file}" \
            --exclude="node_modules" \
            --exclude="__pycache__" \
            --exclude=".git" \
            --exclude="*.log" \
            data/ models/ checkpoints/ configs/ 2>/dev/null || true
        
        log_success "백업 완료: ${backup_file}"
    fi
}

# Docker 이미지 빌드
build_images() {
    log_info "Docker 이미지 빌드 중..."
    
    if [[ "${NO_CACHE}" == true ]]; then
        log_info "캐시 없이 빌드 수행"
        docker-compose build --no-cache --pull
    else
        docker-compose build --pull
    fi
    
    log_success "Docker 이미지 빌드 완료"
}

# 이전 컨테이너 정리
cleanup_containers() {
    if [[ "${CLEANUP}" == true ]]; then
        log_info "이전 컨테이너 및 이미지 정리 중..."
        
        # 중지된 컨테이너 제거
        docker container prune -f
        
        # 사용하지 않는 이미지 제거
        docker image prune -f
        
        # 사용하지 않는 볼륨 제거 (주의: 데이터 손실 가능)
        if [[ "${ENVIRONMENT}" == "development" ]]; then
            docker volume prune -f
        fi
        
        log_success "정리 완료"
    fi
}

# 롤링 업데이트
rolling_update() {
    if [[ "${ROLLING}" == true ]]; then
        log_info "롤링 업데이트 수행 중..."
        
        # 서비스별로 순차 업데이트
        services=("postgres" "redis" "chroma" "fragrance_ai" "celery_worker")
        
        for service in "${services[@]}"; do
            log_info "서비스 업데이트: ${service}"
            docker-compose up -d --no-deps "${service}"
            
            # 서비스 헬스체크 대기
            sleep 10
            
            if ! docker-compose ps "${service}" | grep -q "Up"; then
                log_error "서비스 업데이트 실패: ${service}"
                exit 1
            fi
            
            log_success "서비스 업데이트 완료: ${service}"
        done
    else
        log_info "표준 배포 수행 중..."
        docker-compose up -d
    fi
}

# 헬스체크 수행
perform_health_check() {
    if [[ "${HEALTH_CHECK}" == true ]]; then
        log_info "헬스체크 수행 중..."
        
        # API 서버 헬스체크
        max_attempts=30
        attempt=1
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f http://localhost:8080/health >/dev/null 2>&1; then
                log_success "API 서버 헬스체크 통과"
                break
            fi
            
            log_info "헬스체크 시도 ${attempt}/${max_attempts}"
            sleep 10
            ((attempt++))
        done
        
        if [[ $attempt -gt $max_attempts ]]; then
            log_error "API 서버 헬스체크 실패"
            exit 1
        fi
        
        # 데이터베이스 헬스체크
        if docker-compose exec -T postgres pg_isready -U fragrance_ai >/dev/null 2>&1; then
            log_success "데이터베이스 헬스체크 통과"
        else
            log_error "데이터베이스 헬스체크 실패"
            exit 1
        fi
        
        # Redis 헬스체크
        if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
            log_success "Redis 헬스체크 통과"
        else
            log_error "Redis 헬스체크 실패"
            exit 1
        fi
    fi
}

# 배포 후 작업
post_deploy() {
    log_info "배포 후 작업 수행 중..."
    
    # 데이터베이스 마이그레이션
    log_info "데이터베이스 마이그레이션 실행..."
    docker-compose exec -T fragrance_ai alembic upgrade head
    
    # 캐시 워밍업 (선택적)
    if [[ "${ENVIRONMENT}" == "production" ]]; then
        log_info "캐시 워밍업 수행..."
        docker-compose exec -T fragrance_ai python scripts/warm_cache.py || true
    fi
    
    log_success "배포 후 작업 완료"
}

# 배포 상태 출력
show_deployment_status() {
    log_info "배포 상태 확인..."
    
    echo "=============================================="
    echo "Fragrance AI 배포 상태"
    echo "=============================================="
    echo "환경: ${ENVIRONMENT}"
    echo "배포 시간: $(date)"
    echo "=============================================="
    
    docker-compose ps
    
    echo "=============================================="
    echo "접속 정보:"
    echo "  API 서버: http://localhost:8080"
    echo "  Grafana: http://localhost:3000"
    echo "  Flower: http://localhost:5555"
    echo "  Prometheus: http://localhost:9090"
    echo "=============================================="
}

# 메인 배포 함수
main() {
    # 도움말 확인
    if [[ "$1" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    log_info "Fragrance AI 배포 스크립트 시작"
    
    # 배포 단계 실행
    check_prerequisites
    load_environment
    perform_backup
    cleanup_containers
    build_images
    rolling_update
    post_deploy
    perform_health_check
    show_deployment_status
    
    log_success "배포 완료! 🎉"
    log_info "배포 로그: logs/${DEPLOY_LOG}"
}

# 신호 처리 (Ctrl+C 등)
trap 'log_error "배포가 중단되었습니다."; exit 1' INT TERM

# 메인 함수 실행
main "$@"