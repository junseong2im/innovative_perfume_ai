# Deployment Guide
# Fragrance AI - Production Deployment & Operations

완전한 배포, 마이그레이션, 롤백 가이드

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dependency Management](#dependency-management)
4. [Container Deployment](#container-deployment)
5. [Scaling](#scaling)
6. [Database Migrations](#database-migrations)
7. [Rollback Procedures](#rollback-procedures)
8. [Monitoring & Health Checks](#monitoring--health-checks)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum (Development/Testing):**
- CPU: 8 cores
- RAM: 16GB
- Disk: 50GB SSD
- OS: Linux (Ubuntu 20.04+, CentOS 8+) or Docker Desktop

**Recommended (Production):**
- CPU: 16+ cores
- RAM: 32GB+
- Disk: 100GB+ SSD (NVMe preferred)
- OS: Linux (Ubuntu 22.04 LTS)

### Software Requirements

```bash
# Docker
docker --version  # >= 24.0.0

# Docker Compose
docker-compose --version  # >= 2.20.0

# Python (for scripts)
python --version  # >= 3.11

# Git
git --version  # >= 2.30

# Optional: NVIDIA Docker (for GPU support)
nvidia-docker --version
```

### Infrastructure Requirements

- **Network**: Stable internet connection for pulling images
- **Firewall**: Ports 80, 443, 8000, 5432, 6379, 9090, 3000
- **DNS**: Configured domain name (for production)
- **SSL**: Valid SSL certificates (for production)
- **Backup**: Automated backup system for database
- **Monitoring**: Prometheus + Grafana setup

---

## Environment Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/fragrance-ai.git
cd fragrance-ai
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` file:

```bash
# Application
APP_ENV=production
VERSION=0.1.0
LOG_LEVEL=INFO

# Database (⚠️ CHANGE IN PRODUCTION)
POSTGRES_DB=fragrance_ai
POSTGRES_USER=fragrance_user
POSTGRES_PASSWORD=CHANGEME_SECURE_PASSWORD_HERE
POSTGRES_PORT=5432

# Redis
REDIS_PORT=6379

# Application Secrets (⚠️ CHANGE IN PRODUCTION)
SECRET_KEY=CHANGEME_RANDOM_SECRET_KEY_HERE

# Security
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_ENABLED=true

# Scaling Configuration
APP_REPLICAS=2
LLM_WORKER_REPLICAS=2
RL_WORKER_REPLICAS=1

# Worker Configuration
LLM_WORKER_CONCURRENCY=2
RL_WORKER_CONCURRENCY=1
LLM_MAX_BATCH_SIZE=8
RL_BATCH_SIZE=64

# GPU Support (set to true if using GPU)
USE_GPU=false

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASSWORD=CHANGEME_GRAFANA_PASSWORD

# Sentry (Optional)
SENTRY_DSN=

# Build Metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD)
```

### 3. Generate Secrets

```bash
# Generate SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate POSTGRES_PASSWORD
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 4. Create Required Directories

```bash
mkdir -p data logs models cache checkpoints tensorboard
mkdir -p monitoring/grafana/dashboards
mkdir -p nginx/ssl
mkdir -p scripts
```

---

## Dependency Management

### Using Locked Dependencies

우리는 `requirements-lock.txt`를 사용하여 해시 검증을 통한 재현 가능한 빌드를 보장합니다.

#### Generate Hash-Locked Requirements

```bash
# Install pip-tools
pip install pip-tools

# Generate locked requirements with hashes
pip-compile --generate-hashes requirements.txt -o requirements-lock.txt

# For specific environments
pip-compile --generate-hashes requirements-prod.txt -o requirements-prod-lock.txt
```

#### Install with Hash Verification

```bash
# Install with hash verification
pip install --require-hashes -r requirements-lock.txt

# This will fail if any package hash doesn't match
```

#### Update Dependencies

```bash
# Update specific package
pip-compile --upgrade-package transformers requirements.txt -o requirements-lock.txt

# Update all packages
pip-compile --upgrade requirements.txt -o requirements-lock.txt

# Review changes
git diff requirements-lock.txt
```

---

## Container Deployment

### Initial Deployment

#### 1. Build Images

```bash
# Build all images
docker-compose -f docker-compose.production.yml build

# Build specific service
docker-compose -f docker-compose.production.yml build app
docker-compose -f docker-compose.production.yml build worker-llm
docker-compose -f docker-compose.production.yml build worker-rl
```

#### 2. Initialize Database

```bash
# Start only postgres
docker-compose -f docker-compose.production.yml up -d postgres

# Wait for postgres to be ready
docker-compose -f docker-compose.production.yml exec postgres pg_isready -U fragrance_user

# Run initial migrations
docker-compose -f docker-compose.production.yml run --rm app alembic upgrade head
```

#### 3. Start All Services

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Verify all services are running
docker-compose -f docker-compose.production.yml ps
```

#### 4. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check logs
docker-compose -f docker-compose.production.yml logs -f --tail=100

# Check worker queues
docker-compose -f docker-compose.production.yml exec redis redis-cli INFO
```

### Update Deployment

#### Zero-Downtime Rolling Update

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Update app instances one by one
docker-compose -f docker-compose.production.yml up -d --no-deps --scale app=3
docker-compose -f docker-compose.production.yml up -d --no-deps --scale app=2

# Update LLM workers
docker-compose -f docker-compose.production.yml up -d --no-deps --build worker-llm

# Update RL workers
docker-compose -f docker-compose.production.yml up -d --no-deps --build worker-rl

# Verify health
curl http://localhost:8000/health
```

#### Full Restart (With Downtime)

```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Pull/Build latest
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml build

# Start all services
docker-compose -f docker-compose.production.yml up -d

# Verify
docker-compose -f docker-compose.production.yml ps
```

---

## Scaling

### Horizontal Scaling

#### Scale Application Servers

```bash
# Scale to 5 app instances
docker-compose -f docker-compose.production.yml up -d --scale app=5

# Verify
docker-compose -f docker-compose.production.yml ps app
```

#### Scale LLM Workers

```bash
# Scale to 3 LLM workers
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=3

# Monitor worker registration
docker-compose -f docker-compose.production.yml logs -f worker-llm
```

#### Scale RL Workers

```bash
# Scale to 2 RL workers
docker-compose -f docker-compose.production.yml up -d --scale worker-rl=2

# Check training status
docker-compose -f docker-compose.production.yml exec worker-rl ps aux | grep python
```

### Vertical Scaling

Edit `docker-compose.production.yml`:

```yaml
worker-llm:
  deploy:
    resources:
      limits:
        cpus: '8.0'      # Increased from 4.0
        memory: 16G      # Increased from 8G
      reservations:
        cpus: '2.0'
        memory: 4G
```

Then restart:

```bash
docker-compose -f docker-compose.production.yml up -d --force-recreate worker-llm
```

### Auto-Scaling (with Docker Swarm)

```bash
# Initialize Swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.production.yml fragrance-ai

# Auto-scale based on CPU
docker service update --replicas=3 fragrance-ai_worker-llm
```

---

## Database Migrations

### Running Migrations

#### Initial Setup

```bash
# Create migration
docker-compose -f docker-compose.production.yml exec app alembic revision --autogenerate -m "Initial schema"

# Review migration file
cat alembic/versions/[revision]_initial_schema.py

# Apply migration
docker-compose -f docker-compose.production.yml exec app alembic upgrade head
```

#### Regular Updates

```bash
# Check current version
docker-compose -f docker-compose.production.yml exec app alembic current

# Show migration history
docker-compose -f docker-compose.production.yml exec app alembic history

# Upgrade to latest
docker-compose -f docker-compose.production.yml exec app alembic upgrade head

# Upgrade to specific version
docker-compose -f docker-compose.production.yml exec app alembic upgrade [revision]
```

### Backup Before Migration

```bash
# Backup database
docker-compose -f docker-compose.production.yml exec postgres pg_dump -U fragrance_user -d fragrance_ai > backup_$(date +%Y%m%d_%H%M%S).sql

# Verify backup
ls -lh backup_*.sql
```

### Migration Rollback

```bash
# Downgrade one version
docker-compose -f docker-compose.production.yml exec app alembic downgrade -1

# Downgrade to specific version
docker-compose -f docker-compose.production.yml exec app alembic downgrade [revision]

# Downgrade to base
docker-compose -f docker-compose.production.yml exec app alembic downgrade base
```

---

## Rollback Procedures

### Quick Rollback (Image Version)

#### 1. Identify Previous Version

```bash
# List image tags
docker images | grep fragrance-ai

# Example output:
# fragrance-ai-app    v0.2.0    abc123    2 hours ago    1.5GB
# fragrance-ai-app    v0.1.0    def456    1 day ago      1.4GB
```

#### 2. Rollback to Previous Version

```bash
# Set VERSION in .env
export VERSION=v0.1.0

# Restart with previous version
docker-compose -f docker-compose.production.yml up -d

# Verify
docker-compose -f docker-compose.production.yml ps
```

### Full Rollback (with Database)

#### 1. Stop Services

```bash
docker-compose -f docker-compose.production.yml down
```

#### 2. Restore Database

```bash
# Start only postgres
docker-compose -f docker-compose.production.yml up -d postgres

# Restore backup
cat backup_[DATE].sql | docker-compose -f docker-compose.production.yml exec -T postgres psql -U fragrance_user -d fragrance_ai

# Verify restoration
docker-compose -f docker-compose.production.yml exec postgres psql -U fragrance_user -d fragrance_ai -c "\dt"
```

#### 3. Rollback Code

```bash
# Revert to previous version
export VERSION=v0.1.0
docker-compose -f docker-compose.production.yml up -d

# Or use git tag
git checkout v0.1.0
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

#### 4. Verify Rollback

```bash
# Health check
curl http://localhost:8000/health

# Check version
curl http://localhost:8000/version

# Monitor logs
docker-compose -f docker-compose.production.yml logs -f --tail=100

# Run smoke tests
./scripts/smoke_test.sh
```

### Rollback Verification Checklist

- [ ] All services are healthy: `docker-compose ps`
- [ ] API responding correctly: `curl http://localhost:8000/health`
- [ ] Database accessible: `docker-compose exec postgres psql -U fragrance_user -d fragrance_ai -c "SELECT 1"`
- [ ] Workers operational: `docker-compose exec redis redis-cli INFO | grep connected_clients`
- [ ] No error spikes: Check logs for ERROR/CRITICAL
- [ ] Metrics normal: Check Grafana dashboards
- [ ] Smoke tests pass: `./scripts/smoke_test.sh`

---

## Monitoring & Health Checks

### Service Health

#### Check All Services

```bash
# Docker health status
docker-compose -f docker-compose.production.yml ps

# Detailed health
docker-compose -f docker-compose.production.yml exec app curl http://localhost:8000/health
```

#### API Health Endpoint

```bash
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-10-14T12:00:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "workers": {
      "llm": 2,
      "rl": 1
    }
  }
}
```

### Worker Monitoring

#### Check LLM Workers

```bash
# Worker processes
docker-compose -f docker-compose.production.yml exec worker-llm ps aux | grep celery

# Queue status
docker-compose -f docker-compose.production.yml exec redis redis-cli LLEN celery

# Worker logs
docker-compose -f docker-compose.production.yml logs -f worker-llm
```

#### Check RL Workers

```bash
# Training status
docker-compose -f docker-compose.production.yml exec worker-rl ls -lh /app/checkpoints/

# Tensorboard
docker-compose -f docker-compose.production.yml exec worker-rl ls -lh /app/tensorboard/

# Worker logs
docker-compose -f docker-compose.production.yml logs -f worker-rl
```

### Metrics & Dashboards

#### Prometheus

```bash
# Access Prometheus
open http://localhost:9090

# Check targets
curl http://localhost:9090/api/v1/targets

# Query metrics
curl 'http://localhost:9090/api/v1/query?query=up'
```

#### Grafana

```bash
# Access Grafana
open http://localhost:3000

# Login: admin / [GRAFANA_PASSWORD from .env]
```

**Key Dashboards:**
- System Overview: CPU, Memory, Disk
- Application Metrics: Requests, Latency, Errors
- Worker Metrics: Queue length, Processing time
- Database Metrics: Connections, Queries, Slow queries

---

## Troubleshooting

### Common Issues

#### 1. Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs [service-name]

# Check resource usage
docker stats

# Restart service
docker-compose -f docker-compose.production.yml restart [service-name]
```

#### 2. Database Connection Failed

```bash
# Check postgres is running
docker-compose -f docker-compose.production.yml ps postgres

# Test connection
docker-compose -f docker-compose.production.yml exec postgres pg_isready -U fragrance_user

# Check credentials
docker-compose -f docker-compose.production.yml exec app env | grep DATABASE_URL
```

#### 3. Workers Not Processing

```bash
# Check Redis
docker-compose -f docker-compose.production.yml exec redis redis-cli ping

# Check queue
docker-compose -f docker-compose.production.yml exec redis redis-cli LLEN celery

# Restart workers
docker-compose -f docker-compose.production.yml restart worker-llm worker-rl
```

#### 4. High Memory Usage

```bash
# Check memory usage
docker stats --no-stream

# If LLM worker using too much memory:
# - Reduce batch size in .env: LLM_MAX_BATCH_SIZE=4
# - Reduce number of workers: docker-compose up -d --scale worker-llm=1
# - Increase memory limit in docker-compose.production.yml
```

#### 5. Disk Space Full

```bash
# Check disk usage
df -h

# Clean up Docker
docker system prune -a --volumes

# Clean old logs
find logs/ -name "*.log" -mtime +7 -delete

# Archive old checkpoints
tar -czf checkpoints_archive.tar.gz checkpoints/
rm -rf checkpoints/*
```

### Emergency Procedures

#### Complete System Restart

```bash
# Stop everything
docker-compose -f docker-compose.production.yml down

# Clean up (⚠️ CAUTION: Removes all data)
docker-compose -f docker-compose.production.yml down -v

# Restart
docker-compose -f docker-compose.production.yml up -d
```

#### Recover from Crash

```bash
# 1. Check system resources
free -h
df -h

# 2. Check Docker
docker info

# 3. Restart Docker daemon
sudo systemctl restart docker

# 4. Restart services
docker-compose -f docker-compose.production.yml up -d
```

---

## Best Practices

### Security

1. **Change all default passwords** in `.env`
2. **Use SSL/TLS** in production (configure nginx with certificates)
3. **Enable rate limiting**: `RATE_LIMIT_ENABLED=true`
4. **Restrict CORS**: Set specific origins in `CORS_ORIGINS`
5. **Regular security updates**: Update base images and dependencies
6. **Backup encryption**: Encrypt database backups
7. **Secret management**: Use Docker secrets or vault

### Performance

1. **Monitor resource usage**: Use Grafana dashboards
2. **Scale proactively**: Before hitting limits
3. **Optimize workers**: Tune concurrency settings
4. **Cache aggressively**: Configure Redis properly
5. **Use connection pooling**: Set `DB_POOL_SIZE` appropriately

### Reliability

1. **Health checks**: All services have health checks
2. **Graceful shutdown**: Handle SIGTERM properly
3. **Retry logic**: Configure retry policies in workers
4. **Circuit breakers**: Protect against cascading failures
5. **Regular backups**: Automated daily backups
6. **Test rollbacks**: Practice rollback procedures regularly

### Operations

1. **Version everything**: Tag images with versions
2. **Document changes**: Use release notes template
3. **Monitor logs**: Centralized logging (ELK/Loki)
4. **Alert on errors**: Configure alerting rules
5. **Capacity planning**: Monitor trends, plan ahead

---

## Support

- **Documentation**: `/docs`
- **Issues**: GitHub Issues
- **Emergency**: [Your emergency contact]
- **Monitoring**: Grafana dashboards

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
