# Multi-stage Docker build for Fragrance AI
# Production-ready with security enhancements

# Base stage with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables for security and performance
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    MALLOC_ARENA_MAX=2 \
    PYTHONHASHSEED=random

# Security labels
LABEL maintainer="Fragrance AI Team" \
      version="2.0.0" \
      description="Advanced Fragrance AI System" \
      security.scan="required"

# Install security updates and system dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    # Build essentials
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    # Database dependencies
    libpq-dev \
    libffi-dev \
    libssl-dev \
    # Security tools
    ca-certificates \
    gnupg \
    # Cleanup tools
    apt-utils \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create non-root user with restricted permissions
RUN groupadd -r appuser --gid=1000 && \
    useradd -r -g appuser --uid=1000 --home-dir=/app --shell=/bin/bash appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy requirements first for better build caching
COPY --chown=appuser:appuser requirements-minimal.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-minimal.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy source code selectively
COPY fragrance_ai/ ./fragrance_ai/
COPY scripts/ ./scripts/
COPY data/initial/ ./data/initial/
COPY migrations/ ./migrations/
COPY *.py ./
COPY *.md ./
COPY docker-compose*.yml ./
COPY .env.example ./.env.example

# Change ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Development command
CMD ["uvicorn", "fragrance_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Set production environment variables
ENV ENVIRONMENT=production \
    WORKERS=4 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=100 \
    TIMEOUT=120 \
    KEEPALIVE=5

# Copy source code selectively for production with proper ownership
COPY --chown=appuser:appuser fragrance_ai/ ./fragrance_ai/
COPY --chown=appuser:appuser scripts/production/ ./scripts/production/
COPY --chown=appuser:appuser data/initial/ ./data/initial/
COPY --chown=appuser:appuser migrations/ ./migrations/
COPY --chown=appuser:appuser setup.py pyproject.toml ./
COPY --chown=appuser:appuser README.md ./
COPY --chown=appuser:appuser docker-compose.prod.yml ./

# Install production dependencies with security scanning
COPY --chown=appuser:appuser requirements-prod.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt && \
    pip install --no-cache-dir -e . && \
    # Security: Remove build dependencies
    pip uninstall -y pip setuptools wheel && \
    # Clean up any temporary files
    find /app -type f -name "*.pyc" -delete && \
    find /app -type d -name "__pycache__" -delete

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/models /app/checkpoints /app/tmp && \
    chmod 750 /app/logs /app/data /app/models /app/checkpoints && \
    chmod 1777 /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set secure file permissions
RUN chmod -R 640 /app/fragrance_ai/ && \
    chmod +x /app/scripts/production/*.sh 2>/dev/null || true

# Expose port (non-privileged)
EXPOSE 8000

# Enhanced health check with timeout and better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f -k --max-time 10 http://localhost:8000/health || exit 1

# Production command with Gunicorn for better performance and security
CMD ["gunicorn", "fragrance_ai.api.main:app", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "4", \
     "--bind", "0.0.0.0:8000", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--timeout", "120", \
     "--keepalive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--preload"]

# Training stage for model training
FROM base as training

# Install training dependencies
COPY requirements.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-train.txt

# Install additional ML libraries
RUN pip install --no-cache-dir \
    tensorboard \
    wandb \
    mlflow \
    optuna \
    ray[tune]

# Copy training-specific files
COPY fragrance_ai/ ./fragrance_ai/
COPY scripts/training/ ./scripts/training/
COPY data/ ./data/
COPY configs/ ./configs/

# Create directories for training
RUN mkdir -p /app/data/training /app/checkpoints /app/tensorboard /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Training command
CMD ["python", "scripts/train_model.py"]

# Evaluation stage for model evaluation
FROM base as evaluation

# Install evaluation dependencies
COPY requirements.txt requirements-eval.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-eval.txt

# Copy evaluation-specific files
COPY fragrance_ai/ ./fragrance_ai/
COPY scripts/evaluation/ ./scripts/evaluation/
COPY data/test/ ./data/test/
COPY tests/ ./tests/

# Create directories for evaluation
RUN mkdir -p /app/data/evaluation /app/evaluation_results /app/logs && \
    chown -R appuser:appuser /app

USER appuser

# Evaluation command
CMD ["python", "scripts/evaluate_model.py"]