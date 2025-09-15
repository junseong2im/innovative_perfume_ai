# Multi-stage Docker build for Fragrance AI

# Base stage with Python and system dependencies
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    wget \
    build-essential \
    libpq-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better build caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

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

# Copy source code selectively for production
COPY fragrance_ai/ ./fragrance_ai/
COPY scripts/production/ ./scripts/production/
COPY data/initial/ ./data/initial/
COPY migrations/ ./migrations/
COPY setup.py pyproject.toml ./
COPY README.md ./
COPY docker-compose.prod.yml ./

# Install production dependencies only
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/checkpoints && \
    chown -R appuser:appuser /app

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "fragrance_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

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