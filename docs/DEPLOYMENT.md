# Deployment Infrastructure Guide

Complete guide for deploying Fragrance AI system with Docker, CI/CD, and environment configuration.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Docker Deployment](#docker-deployment)
3. [Environment Configuration](#environment-configuration)
4. [CI Pipeline](#ci-pipeline)
5. [Production Deployment](#production-deployment)
6. [Monitoring & Observability](#monitoring--observability)

---

## Architecture Overview

### 3-Service Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer (Optional)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         FastAPI Application              │
        │         (docker/Dockerfile.app)          │
        │  • REST API endpoints                    │
        │  • Request routing                       │
        │  • Authentication                        │
        └──────────────┬───────────────────────────┘
                       │
                       │ Redis Queue
                       │
           ┌───────────┴────────────┐
           │                        │
           ▼                        ▼
  ┌────────────────┐      ┌────────────────┐
  │  LLM Worker    │      │  RL Worker     │
  │  (worker-llm)  │      │  (worker-rl)   │
  │                │      │                │
  │  • Qwen       │      │  • PPO Training│
  │  • Mistral    │      │  • Checkpoints │
  │  • Llama      │      │  • Tensorboard │
  └────────────────┘      └────────────────┘
           │                        │
           └───────────┬────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  Redis Cache   │
              │  Message Queue │
              └────────────────┘
```

### Service Responsibilities

| Service | Purpose | Scaling Strategy |
|---------|---------|------------------|
| **app** | API endpoints, request routing | Horizontal (N instances) |
| **worker-llm** | LLM ensemble inference | Horizontal (GPU-bound) |
| **worker-rl** | RL training tasks | Vertical (memory/GPU) |
| **redis** | Queue & cache | Vertical (single instance) |

---

## Docker Deployment

### Quick Start

```bash
# 1. Set environment
export APP_ENV=production

# 2. Start all services
docker-compose -f docker/docker-compose.workers.yml up -d

# 3. Check status
docker-compose -f docker/docker-compose.workers.yml ps

# 4. View logs
docker-compose -f docker/docker-compose.workers.yml logs -f app
```

### Available Docker Images

#### 1. API Service (`Dockerfile.app`)

**Purpose**: FastAPI application server

**Base Image**: `python:3.11-slim`

**Key Dependencies**:
- FastAPI
- Uvicorn (ASGI server)
- SQLAlchemy (database ORM)
- Redis client

**Exposed Ports**: `8000`

**Health Check**: `GET /health` every 30s

**Build**:
```bash
docker build -f docker/Dockerfile.app -t fragrance-ai-app:latest .
```

**Run Standalone**:
```bash
docker run -d \
  --name fragrance-ai-app \
  -p 8000:8000 \
  -e APP_ENV=production \
  -e DATABASE_URL=sqlite:///./data/fragrance.db \
  -e REDIS_URL=redis://redis:6379/0 \
  -v $(pwd)/data:/app/data \
  fragrance-ai-app:latest
```

#### 2. LLM Worker (`Dockerfile.worker-llm`)

**Purpose**: LLM ensemble inference (Qwen, Mistral, Llama)

**Base Image**: `python:3.11-slim`

**Key Dependencies**:
- transformers >= 4.35.0
- torch >= 2.0.0
- accelerate >= 0.24.0
- sentencepiece

**Environment Variables**:
- `WORKER_TYPE=llm`
- `HF_HOME=/app/cache` (Hugging Face cache)
- `USE_GPU=false` (set to true for GPU)

**GPU Support**: Uncomment `deploy.resources.reservations.devices` in docker-compose

**Build**:
```bash
docker build -f docker/Dockerfile.worker-llm -t fragrance-ai-worker-llm:latest .
```

**Run Standalone**:
```bash
docker run -d \
  --name fragrance-ai-worker-llm \
  -e WORKER_TYPE=llm \
  -e REDIS_URL=redis://redis:6379/0 \
  -v $(pwd)/cache:/app/cache \
  fragrance-ai-worker-llm:latest
```

#### 3. RL Worker (`Dockerfile.worker-rl`)

**Purpose**: Reinforcement learning training with PPO

**Base Image**: `python:3.11-slim`

**Key Dependencies**:
- torch >= 2.0.0
- deap >= 1.4.0 (genetic algorithms)
- ray >= 2.7.0 (distributed training)
- tensorboard >= 2.14.0

**Environment Variables**:
- `WORKER_TYPE=rl`
- `CHECKPOINT_DIR=/app/checkpoints`
- `TENSORBOARD_DIR=/app/tensorboard`

**Build**:
```bash
docker build -f docker/Dockerfile.worker-rl -t fragrance-ai-worker-rl:latest .
```

**Run Standalone**:
```bash
docker run -d \
  --name fragrance-ai-worker-rl \
  -e WORKER_TYPE=rl \
  -e CHECKPOINT_DIR=/app/checkpoints \
  -e REDIS_URL=redis://redis:6379/0 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  fragrance-ai-worker-rl:latest
```

### Docker Compose Deployment

#### Basic Deployment (App + Workers + Redis)

```bash
docker-compose -f docker/docker-compose.workers.yml up -d
```

#### With Monitoring (Prometheus + Grafana)

```bash
docker-compose -f docker/docker-compose.workers.yml --profile monitoring up -d
```

#### Scaling Workers

```bash
# Scale LLM workers to 3 instances
docker-compose -f docker/docker-compose.workers.yml up -d --scale worker-llm=3

# Scale RL workers to 2 instances
docker-compose -f docker/docker-compose.workers.yml up -d --scale worker-rl=2
```

#### Production Deployment with GPU

Edit `docker-compose.workers.yml` and uncomment GPU sections:

```yaml
worker-llm:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Then:

```bash
docker-compose -f docker/docker-compose.workers.yml up -d
```

---

## Environment Configuration

### Environment Profiles

Three environment profiles are supported:

| Profile | Purpose | Use Case |
|---------|---------|----------|
| **development** | Local development | CPU-only, minimal resources, debug logging |
| **staging** | Pre-production testing | GPU-enabled, quantization, real workload |
| **production** | Production deployment | Full precision, horizontal scaling, minimal logging |

### Configuration Files

#### 1. `.env.dev` (Development)

```bash
APP_ENV=development
API_WORKERS=2
USE_GPU=false
LOG_LEVEL=DEBUG
LLM_WORKER_CONCURRENCY=1
RL_WORKER_CONCURRENCY=1
DEBUG_MODE=true
```

**Characteristics**:
- SQLite database
- Local Redis
- CPU-only inference
- Verbose logging
- Hot reload enabled

#### 2. `.env.staging` (Staging)

```bash
APP_ENV=staging
API_WORKERS=4
USE_GPU=true
LOG_LEVEL=INFO
LLM_WORKER_CONCURRENCY=2
RL_WORKER_CONCURRENCY=1
PROMETHEUS_ENABLED=true
```

**Characteristics**:
- PostgreSQL database
- GPU inference with quantization
- Monitoring enabled
- Production-like workload

#### 3. `.env.prod` (Production)

```bash
APP_ENV=production
API_WORKERS=8
USE_GPU=true
LOG_LEVEL=WARNING
LLM_WORKER_CONCURRENCY=4
RL_WORKER_CONCURRENCY=2
RATE_LIMIT_ENABLED=true
AUTO_SCALING=true
```

**Characteristics**:
- PostgreSQL with connection pooling
- Full precision GPU inference
- Rate limiting
- Auto-scaling
- Security hardening

### LLM Ensemble Configuration

`configs/llm_ensemble.yaml` contains environment-specific LLM settings:

```yaml
development:
  use_gpu: false
  qwen:
    device_map: "cpu"
    max_new_tokens: 256
  mistral:
    enabled: false  # Save memory
  llama:
    enabled: false

production:
  use_gpu: true
  qwen:
    device_map: "auto"
    max_new_tokens: 1024
  mistral:
    enabled: true
  llama:
    enabled: true
```

### Using Configuration in Code

```python
from fragrance_ai.config_loader import (
    load_llm_config,
    get_api_config,
    get_worker_config,
    get_redis_url
)

# Load LLM config for current environment
llm_config = load_llm_config()

# Get API configuration
api_config = get_api_config()
print(f"API running on port {api_config['port']}")

# Get worker-specific config
llm_worker_config = get_worker_config('llm')
print(f"LLM worker concurrency: {llm_worker_config['concurrency']}")

# Get connection URLs
redis_url = get_redis_url()
```

### Configuration Priority

1. **Environment variables** (highest priority)
2. **Profile-specific YAML config** (e.g., `production:` section)
3. **Default YAML config** (base settings)

Example:

```bash
# Override cache TTL via environment
export CACHE_TTL=7200

# This overrides the YAML setting
python -m fragrance_ai.workers.llm_worker
```

---

## CI Pipeline

### GitHub Actions Workflow

Location: `.github/workflows/ci.yml`

### Pipeline Stages

```
┌────────────┐
│    Lint    │  Ruff linting
└──────┬─────┘
       │
┌──────▼──────┐
│ Type Check  │  mypy static analysis
└──────┬──────┘
       │
┌──────▼──────┐
│    Tests    │  pytest unit tests
└──────┬──────┘
       │
┌──────▼──────┐
│ Smoke Test  │  Small inference test
└──────┬──────┘
       │
┌──────▼──────┐
│Docker Build │  Build all 3 images
└──────┬──────┘
       │
┌──────▼──────┐
│   Summary   │  CI report
└─────────────┘
```

### Running CI Locally

#### 1. Linting with Ruff

```bash
# Install ruff
pip install ruff

# Run linter
ruff check fragrance_ai/ app/ tests/
```

#### 2. Type Checking with mypy

```bash
# Install mypy
pip install mypy

# Run type checker
mypy fragrance_ai/ app/ --ignore-missing-imports --no-strict-optional
```

#### 3. Unit Tests with pytest

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=fragrance_ai --cov-report=term-missing
```

#### 4. Smoke Test

Quick validation that core modules work:

```bash
python -c "
from fragrance_ai.training.rl_advanced import EntropyScheduler
from fragrance_ai.training.ppo_engine import FragranceEnvironment

# Test entropy scheduler
scheduler = EntropyScheduler(EntropyScheduleConfig(
    initial_entropy=0.01,
    final_entropy=0.001,
    decay_steps=100
))
entropy = scheduler.step()
print(f'✓ Entropy scheduler: {entropy:.6f}')

# Test environment
env = FragranceEnvironment(n_ingredients=5)
state = env.reset()
print(f'✓ Environment: state_dim={len(state)}')
"
```

### CI Triggers

- **Push** to `master`, `main`, or `develop` branches
- **Pull Requests** to these branches

### CI Configuration

Edit `.github/workflows/ci.yml` to customize:

```yaml
on:
  push:
    branches: [ master, main, develop ]
  pull_request:
    branches: [ master, main, develop ]
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Update `.env.prod` with production secrets
- [ ] Configure database connection (PostgreSQL recommended)
- [ ] Set up Redis instance (AWS ElastiCache, etc.)
- [ ] Configure monitoring (Prometheus, Grafana)
- [ ] Set up logging aggregation (ELK, Datadog, etc.)
- [ ] Enable TLS/SSL certificates
- [ ] Configure firewall rules
- [ ] Set up backup strategy
- [ ] Configure auto-scaling policies

### Deployment Steps

#### 1. Build Images

```bash
# Build all images
docker build -f docker/Dockerfile.app -t fragrance-ai-app:1.0.0 .
docker build -f docker/Dockerfile.worker-llm -t fragrance-ai-worker-llm:1.0.0 .
docker build -f docker/Dockerfile.worker-rl -t fragrance-ai-worker-rl:1.0.0 .

# Tag for registry
docker tag fragrance-ai-app:1.0.0 registry.example.com/fragrance-ai-app:1.0.0
docker tag fragrance-ai-worker-llm:1.0.0 registry.example.com/fragrance-ai-worker-llm:1.0.0
docker tag fragrance-ai-worker-rl:1.0.0 registry.example.com/fragrance-ai-worker-rl:1.0.0

# Push to registry
docker push registry.example.com/fragrance-ai-app:1.0.0
docker push registry.example.com/fragrance-ai-worker-llm:1.0.0
docker push registry.example.com/fragrance-ai-worker-rl:1.0.0
```

#### 2. Deploy with Docker Compose

```bash
# Set environment
export APP_ENV=production

# Pull latest images
docker-compose -f docker/docker-compose.workers.yml pull

# Start services
docker-compose -f docker/docker-compose.workers.yml up -d

# Verify
docker-compose -f docker/docker-compose.workers.yml ps
```

#### 3. Kubernetes Deployment (Optional)

Convert Docker Compose to Kubernetes manifests:

```bash
# Install kompose
curl -L https://github.com/kubernetes/kompose/releases/download/v1.31.2/kompose-linux-amd64 -o kompose
chmod +x kompose

# Convert
kompose convert -f docker/docker-compose.workers.yml

# Deploy
kubectl apply -f .
```

### Health Checks

All services have health checks:

```bash
# App health check
curl http://localhost:8000/health

# Worker health checks (process check)
docker exec fragrance-ai-worker-llm pgrep -f "worker.*llm"
docker exec fragrance-ai-worker-rl pgrep -f "worker.*rl"

# Redis health check
docker exec fragrance-ai-redis redis-cli ping
```

### Rolling Updates

```bash
# Update app service with zero downtime
docker-compose -f docker/docker-compose.workers.yml up -d --no-deps app

# Update workers
docker-compose -f docker/docker-compose.workers.yml up -d --no-deps worker-llm worker-rl
```

---

## Monitoring & Observability

### Prometheus Metrics

Enable Prometheus monitoring:

```bash
docker-compose -f docker/docker-compose.workers.yml --profile monitoring up -d
```

Access Prometheus: `http://localhost:9090`

### Grafana Dashboards

Access Grafana: `http://localhost:3000`

**Default credentials**: `admin` / `admin`

### Application Metrics

The application exposes metrics at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Key Metrics**:
- `http_requests_total`: Total HTTP requests
- `http_request_duration_seconds`: Request latency
- `llm_inference_duration_seconds`: LLM inference time
- `rl_training_episodes_total`: RL training episodes
- `queue_size`: Redis queue depth

### Logging

Configure structured JSON logging in production:

```bash
# In .env.prod
LOG_FORMAT=json
LOG_LEVEL=WARNING
```

View logs:

```bash
# App logs
docker-compose -f docker/docker-compose.workers.yml logs -f app

# Worker logs
docker-compose -f docker/docker-compose.workers.yml logs -f worker-llm worker-rl

# All logs
docker-compose -f docker/docker-compose.workers.yml logs -f
```

### Tensorboard (RL Training)

Monitor RL training progress:

```bash
# Start Tensorboard
docker run -d \
  -p 6006:6006 \
  -v $(pwd)/tensorboard:/logs \
  tensorflow/tensorflow:latest \
  tensorboard --logdir=/logs --host=0.0.0.0

# Access at http://localhost:6006
```

---

## Troubleshooting

### Common Issues

#### 1. Worker Not Processing Tasks

```bash
# Check Redis connection
docker exec fragrance-ai-app redis-cli -h redis ping

# Check queue size
docker exec fragrance-ai-redis redis-cli llen llm_inference_queue

# Check worker logs
docker logs fragrance-ai-worker-llm
```

#### 2. Out of Memory (GPU)

```bash
# Enable quantization in .env
USE_GPU=true
ENABLE_QUANTIZATION=true

# Or use 4-bit quantization in configs/llm_ensemble.yaml
production:
  qwen:
    load_in_4bit: true
```

#### 3. Slow Inference

```bash
# Check GPU utilization
nvidia-smi

# Scale workers
docker-compose -f docker/docker-compose.workers.yml up -d --scale worker-llm=4

# Enable caching
USE_CACHE=true
CACHE_TTL=7200
```

#### 4. Database Connection Issues

```bash
# Check database URL
docker exec fragrance-ai-app env | grep DATABASE_URL

# Test connection
docker exec fragrance-ai-app python -c "
from fragrance_ai.config_loader import get_database_url
from sqlalchemy import create_engine
engine = create_engine(get_database_url())
conn = engine.connect()
print('✓ Database connected')
"
```

---

## Security Best Practices

1. **Use secrets management**: Don't commit `.env.prod` to Git
2. **Enable TLS**: Use reverse proxy (nginx, Traefik) for HTTPS
3. **Network isolation**: Use Docker networks, firewall rules
4. **API authentication**: Enable API key or OAuth2
5. **Rate limiting**: Prevent abuse with request limits
6. **Update dependencies**: Regularly update Docker images and Python packages
7. **Log monitoring**: Set up alerts for suspicious activity
8. **Backup strategy**: Regular database and checkpoint backups

---

## Summary

**Deployment Infrastructure**:
- ✅ 3 separate Docker images (app, worker-llm, worker-rl)
- ✅ Docker Compose orchestration
- ✅ Environment-specific configuration (dev/staging/prod)
- ✅ CI pipeline (mypy, ruff, pytest, smoke test)
- ✅ Configuration loader with env variable mapping
- ✅ Monitoring with Prometheus & Grafana

**Next Steps**:
1. Customize `.env.prod` for your production environment
2. Set up external database and Redis
3. Configure monitoring and alerting
4. Deploy to your target infrastructure (AWS, GCP, Azure, on-prem)
5. Set up CI/CD automation (GitHub Actions, GitLab CI, Jenkins)

For questions or issues, see the main README or open an issue on GitHub.
