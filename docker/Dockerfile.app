# Dockerfile.app
# FastAPI Application Container

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-api.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install API-specific dependencies (if exists)
RUN if [ -f requirements-api.txt ]; then \
        pip install --no-cache-dir -r requirements-api.txt; \
    fi

# Copy application code
COPY fragrance_ai/ ./fragrance_ai/
COPY app/ ./app/
COPY configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV APP_ENV=production

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
