# 🚀 Fragrance AI - Production Deployment Guide

## 📋 Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- PostgreSQL 16 with pgvector extension
- Redis 7+
- NVIDIA GPU (optional, for AI acceleration)
- Minimum 8GB RAM
- 20GB free disk space

## 🛠️ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/fragrance-ai.git
cd fragrance-ai
```

### 2. Configure Environment
```bash
cp .env.production .env
# Edit .env with your production values
nano .env
```

### 3. Deploy with Docker Compose
```bash
docker-compose -f docker-compose.production.yml up -d
```

### 4. Run Database Migrations
```bash
docker-compose -f docker-compose.production.yml exec api alembic upgrade head
```

### 5. Verify Deployment
```bash
curl http://localhost:8001/health
```

## 📦 Component Overview

| Service | Port | Description |
|---------|------|-------------|
| API | 8001 | Main FastAPI application |
| PostgreSQL | 5432 | Primary database with pgvector |
| Redis | 6379 | Cache and session store |
| Ollama | 11434 | Local LLM service |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Monitoring dashboard |
| Nginx | 80/443 | Load balancer & reverse proxy |

## 🔧 Configuration

### Environment Variables

Key variables in `.env.production`:

- `SECRET_KEY`: **MUST CHANGE** - Random 64-character string
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OLLAMA_BASE_URL`: Ollama service URL
- `DOMAIN_NAME`: Your production domain

### SSL/TLS Setup

1. Obtain SSL certificates (Let's Encrypt recommended)
2. Place certificates in `./ssl/` directory
3. Update nginx.conf with certificate paths

## 🚀 Deployment Steps

### Automated Deployment
```bash
sudo ./deploy.sh production
```

### Manual Deployment

1. **Build Images**
```bash
docker-compose -f docker-compose.production.yml build
```

2. **Start Services**
```bash
docker-compose -f docker-compose.production.yml up -d
```

3. **Check Health**
```bash
docker-compose -f docker-compose.production.yml exec api python -c "
from fragrance_ai.api.health import health_checker
import asyncio
result = asyncio.run(health_checker.perform_all_checks())
print(result)
"
```

## 📊 Monitoring

### Grafana Dashboard
- URL: `http://localhost:3000`
- Default login: admin/admin (change immediately)
- Pre-configured dashboards available

### Prometheus Metrics
- URL: `http://localhost:9090`
- Custom metrics exposed at `/metrics`

### Health Endpoints
- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe
- `/health/detailed` - Comprehensive system check

## 🔄 Scaling

### Horizontal Scaling
```yaml
# docker-compose.production.yml
api:
  deploy:
    replicas: 4  # Increase replicas
```

### Resource Limits
```yaml
resources:
  limits:
    cpus: '4'
    memory: 8G
```

## 🛡️ Security Checklist

- [ ] Change all default passwords
- [ ] Generate new SECRET_KEY
- [ ] Configure firewall rules
- [ ] Enable SSL/TLS
- [ ] Set up rate limiting
- [ ] Configure CORS properly
- [ ] Enable audit logging
- [ ] Regular security updates

## 🔧 Maintenance

### Backup Database
```bash
docker-compose -f docker-compose.production.yml exec postgres \
  pg_dump -U fragrance fragrance_ai > backup_$(date +%Y%m%d).sql
```

### Update Application
```bash
git pull origin main
docker-compose -f docker-compose.production.yml build
docker-compose -f docker-compose.production.yml up -d
```

### View Logs
```bash
# All services
docker-compose -f docker-compose.production.yml logs -f

# Specific service
docker-compose -f docker-compose.production.yml logs -f api
```

### Clean Up
```bash
docker system prune -a --volumes
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   - Solution: Check Python path and module structure
   - Verify: `docker-compose exec api python -c "import fragrance_ai"`

2. **Database Connection Failed**
   - Check: PostgreSQL is running
   - Verify: pgvector extension installed
   - Test: `docker-compose exec postgres psql -U fragrance -c "SELECT version()"`

3. **LLM Service Unavailable**
   - Check: Ollama container is running
   - Pull models: `docker exec ollama ollama pull llama3:8b`

4. **High Memory Usage**
   - Reduce worker count
   - Enable swap memory
   - Use smaller AI models

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
docker-compose -f docker-compose.production.yml up
```

## 📈 Performance Optimization

1. **Database Indexing**
```sql
CREATE INDEX idx_fragrance_embedding ON fragrances USING ivfflat (embedding vector_l2_ops);
```

2. **Redis Caching**
```python
# Configure cache TTL in .env
CACHE_TTL_SECONDS=3600
```

3. **Model Optimization**
- Use quantized models (4-bit/8-bit)
- Enable model caching
- Batch inference requests

## 📝 API Documentation

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`
- OpenAPI Schema: `http://localhost:8001/openapi.json`

## 🆘 Support

- GitHub Issues: [Report bugs](https://github.com/yourusername/fragrance-ai/issues)
- Documentation: [Full docs](https://docs.fragranceai.com)
- Email: support@fragranceai.com

## 📄 License

Proprietary - See LICENSE file

---

**Last Updated**: 2025-01-27
**Version**: 1.0.0