# Operations Runbook
# Fragrance AI - 운영 절차서

프로덕션 환경 운영을 위한 완전한 런북

---

## Table of Contents

1. [Health Check Procedures](#health-check-procedures)
2. [Scaling Procedures](#scaling-procedures)
3. [Rollback Procedures](#rollback-procedures)
4. [Incident Response](#incident-response)
5. [Monitoring & Alerts](#monitoring--alerts)
6. [Maintenance Windows](#maintenance-windows)
7. [Emergency Contacts](#emergency-contacts)

---

## Health Check Procedures

### Quick Health Check

```bash
# API Health
curl -f http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "version": "0.2.0",
#   "timestamp": "2025-10-14T12:00:00Z",
#   "services": {
#     "database": "healthy",
#     "redis": "healthy",
#     "workers": {"llm": 2, "rl": 1}
#   }
# }
```

### Comprehensive Health Check

```bash
# 1. Check all services status
docker-compose -f docker-compose.production.yml ps

# Expected: All services show "Up" status

# 2. Check database
docker-compose -f docker-compose.production.yml exec postgres \
  pg_isready -U fragrance_user -d fragrance_ai

# Expected: "fragrance_ai:5432 - accepting connections"

# 3. Check Redis
docker-compose -f docker-compose.production.yml exec redis redis-cli ping

# Expected: "PONG"

# 4. Check worker queues
docker-compose -f docker-compose.production.yml exec redis redis-cli INFO

# Look for: connected_clients, used_memory

# 5. Check API endpoints
curl -f http://localhost:8000/docs  # Swagger UI (if enabled)
curl -f http://localhost:8000/metrics  # Prometheus metrics
```

### Health Check Schedule

| Check | Frequency | Alert Threshold |
|-------|-----------|----------------|
| API /health endpoint | Every 30s | 2 consecutive failures |
| Database connection | Every 60s | 1 failure |
| Redis connection | Every 60s | 1 failure |
| Worker queue length | Every 5min | > 100 items |
| Disk space | Every 10min | > 85% used |
| Memory usage | Every 5min | > 90% used |

### Health Check Automation

```bash
# Run automated health check
python scripts/pre_deployment_check.py --version $(cat VERSION)

# Monitor continuously
watch -n 30 'curl -s http://localhost:8000/health | jq'
```

---

## Scaling Procedures

### Upscaling (증설)

#### Scale App Instances

```bash
# Current status
docker-compose -f docker-compose.production.yml ps app

# Scale to 5 instances
docker-compose -f docker-compose.production.yml up -d --scale app=5

# Verify
docker-compose -f docker-compose.production.yml ps app

# Check load distribution (via NGINX logs)
docker-compose -f docker-compose.production.yml logs -f nginx
```

**When to Scale App:**
- API response time > 500ms (95th percentile)
- CPU usage > 70% across all instances
- Request queue length > 50
- Expected traffic spike (marketing campaign, etc.)

#### Scale LLM Workers

```bash
# Scale to 10 LLM workers
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=10

# Monitor worker registration
docker-compose -f docker-compose.production.yml logs -f worker-llm | grep "registered"

# Check queue processing rate
docker-compose -f docker-compose.production.yml exec redis redis-cli LLEN celery
```

**When to Scale LLM Workers:**
- LLM queue length > 50
- Average wait time > 10s
- LLM worker CPU > 80%
- Inference requests increasing

#### Scale RL Workers

```bash
# Scale to 4 RL workers
docker-compose -f docker-compose.production.yml up -d --scale worker-rl=4

# Monitor training jobs
docker-compose -f docker-compose.production.yml exec worker-rl ls -lh /app/checkpoints/

# Check GPU utilization (if using GPU)
nvidia-smi
```

**When to Scale RL Workers:**
- Training queue backlog > 10 jobs
- Checkpoint generation delayed
- Model quality degradation detected

### Downscaling (축소)

#### Gradual Downscale

```bash
# Scale down app instances gradually
# From 5 -> 4 -> 3 -> 2
docker-compose -f docker-compose.production.yml up -d --scale app=4

# Wait 5 minutes and monitor
sleep 300

# Check metrics
curl -s http://localhost:8000/metrics | grep http_requests_total

# Continue if metrics are stable
docker-compose -f docker-compose.production.yml up -d --scale app=3
```

#### Emergency Downscale

```bash
# Quick downscale (use only if system overloaded)
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=1 --scale worker-rl=1

# This reduces resource usage immediately
```

**When to Downscale:**
- Off-peak hours (low traffic)
- Resource costs optimization
- After incident resolution
- Normal traffic patterns restored

### Scaling Best Practices

1. **Monitor Before and After**: Always check metrics before and after scaling
2. **Gradual Changes**: Scale in small increments (2x or 0.5x at a time)
3. **Wait Between Changes**: Allow 5-10 minutes for system to stabilize
4. **Document Reasons**: Record why scaling was performed
5. **Review Resource Usage**: Check CPU, memory, disk after scaling

---

## Rollback Procedures

### Quick Rollback (Image Version)

**Scenario**: New version deployed, issues detected

```bash
# 1. Identify previous version
export PREVIOUS_VERSION=v0.1.0

# 2. Stop current services
docker-compose -f docker-compose.production.yml down

# 3. Set version environment variable
export VERSION=$PREVIOUS_VERSION

# 4. Start services with previous version
docker-compose -f docker-compose.production.yml up -d

# 5. Verify rollback
curl http://localhost:8000/health
docker-compose -f docker-compose.production.yml logs -f --tail=100

# 6. Check version
curl http://localhost:8000/version
```

**Expected Duration**: 2-5 minutes

### Full Rollback (with Database)

**Scenario**: Database migration issues, need complete rollback

```bash
# 1. Stop all services
docker-compose -f docker-compose.production.yml down

# 2. Identify backup
ls -lh backups/ | tail -5

# 3. Restore database
BACKUP_FILE="backups/backup_20251014_120000.sql"

# Start only postgres
docker-compose -f docker-compose.production.yml up -d postgres

# Wait for postgres to be ready
sleep 10

# Restore backup
cat $BACKUP_FILE | docker-compose -f docker-compose.production.yml exec -T postgres \
  psql -U fragrance_user -d fragrance_ai

# 4. Verify database restoration
docker-compose -f docker-compose.production.yml exec postgres psql -U fragrance_user -d fragrance_ai -c "\dt"

# 5. Rollback code to previous version
export VERSION=v0.1.0
docker-compose -f docker-compose.production.yml up -d

# 6. Run verification
./scripts/smoke_test.sh
```

**Expected Duration**: 10-20 minutes

### Automated Rollback

```bash
# Use rollback script
./scripts/rollback.sh v0.1.0

# Follow prompts for database restoration
```

### Rollback Decision Tree

```
Issue Detected
    │
    ├─> API Errors (5xx) > 5%
    │   └─> Quick Rollback
    │
    ├─> Database Errors
    │   └─> Full Rollback with DB Restore
    │
    ├─> Performance Degradation (2x slower)
    │   └─> Quick Rollback
    │
    ├─> Worker Failures > 50%
    │   └─> Quick Rollback (workers only)
    │
    └─> Data Corruption
        └─> Full Rollback + Investigation
```

### Post-Rollback Actions

1. ✅ Verify all services healthy
2. ✅ Check error rates (< 1%)
3. ✅ Monitor for 30 minutes
4. ✅ Document incident
5. ✅ Schedule postmortem
6. ✅ Fix issues in staging
7. ✅ Plan re-deployment

---

## Incident Response

### Severity Levels

| Level | Description | Response Time | Escalation |
|-------|-------------|---------------|------------|
| **P0 - Critical** | Service down, data loss | Immediate | All hands |
| **P1 - High** | Major feature broken | 15 minutes | On-call engineer |
| **P2 - Medium** | Minor feature broken | 4 hours | Next business day |
| **P3 - Low** | Cosmetic issues | 2 days | Backlog |

### P0: Critical Incident

**Definition**: Complete service outage, data corruption, security breach

**Immediate Actions** (first 5 minutes):

```bash
# 1. Confirm incident
curl http://localhost:8000/health

# 2. Check service status
docker-compose -f docker-compose.production.yml ps

# 3. Check recent changes
git log -5 --oneline

# 4. Notify team
# Send alert via Slack/PagerDuty

# 5. Start incident log
echo "$(date): Incident detected - [description]" >> incidents/incident_$(date +%Y%m%d).log
```

**Mitigation** (5-15 minutes):

```bash
# Option 1: Restart services
docker-compose -f docker-compose.production.yml restart

# Option 2: Rollback to previous version
./scripts/rollback.sh v0.1.0

# Option 3: Scale down problematic component
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=0
```

**Resolution Checklist**:

- [ ] Service restored
- [ ] Root cause identified
- [ ] Monitoring shows normal behavior
- [ ] Team notified
- [ ] Incident log updated
- [ ] Postmortem scheduled

### P1: High Priority Incident

**Definition**: Major feature unavailable, significant performance degradation

**Response Procedure**:

```bash
# 1. Identify affected component
docker-compose -f docker-compose.production.yml logs -f --tail=500 [service]

# 2. Check metrics
curl http://localhost:9090/api/v1/query?query=up

# 3. Isolate issue
# If specific service:
docker-compose -f docker-compose.production.yml restart [service]

# If specific worker:
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=1
```

### Incident Communication Template

```
🚨 Incident: [Title]

Severity: P[0-3]
Status: [Investigating/Identified/Monitoring/Resolved]
Impact: [Description]
Started: [Time]

Updates:
[HH:MM] - [Update message]

Actions Taken:
- [Action 1]
- [Action 2]

Next Steps:
- [Next action]

ETA: [Estimated resolution time]
```

---

## Monitoring & Alerts

### Grafana Dashboards

**Access**: http://localhost:3000

**Default Dashboards**:

1. **System Overview**
   - CPU, Memory, Disk usage
   - Network I/O
   - Container status

2. **Application Metrics**
   - Request rate
   - Response time (p50, p95, p99)
   - Error rate
   - Active connections

3. **Worker Metrics**
   - Queue length
   - Processing time
   - Task success/failure rate
   - Worker utilization

4. **Database Metrics**
   - Connection pool usage
   - Query performance
   - Slow query log
   - Replication lag (if applicable)

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m])

# Response time (95th percentile)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Worker queue length
celery_queue_length

# Database connections
pg_stat_database_numbackends

# Redis memory usage
redis_memory_used_bytes
```

### Alert Rules

**File**: `monitoring/prometheus/alert_rules.yml`

```yaml
groups:
  - name: fragrance_ai_alerts
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"

      # High response time
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time (p95 > 1s)"

      # Worker queue backlog
      - alert: WorkerQueueBacklog
        expr: celery_queue_length > 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Worker queue backlog > 100"

      # Database connection pool exhausted
      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_database_numbackends / pg_settings_max_connections > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool > 90% utilized"
```

### Log Aggregation

```bash
# View all logs
docker-compose -f docker-compose.production.yml logs -f

# Filter by service
docker-compose -f docker-compose.production.yml logs -f app

# Filter by level
docker-compose -f docker-compose.production.yml logs -f | grep ERROR

# Search for specific pattern
docker-compose -f docker-compose.production.yml logs -f | grep "timeout"

# Save logs to file
docker-compose -f docker-compose.production.yml logs --since 1h > logs/debug_$(date +%Y%m%d_%H%M%S).log
```

---

## Maintenance Windows

### Scheduled Maintenance

**Recommended Window**: Sunday 2:00 AM - 4:00 AM UTC (Lowest traffic)

**Pre-Maintenance Checklist**:

- [ ] Notify users 24 hours in advance
- [ ] Create backup
- [ ] Test maintenance procedure in staging
- [ ] Prepare rollback plan
- [ ] Assign on-call engineer
- [ ] Update status page

**Maintenance Procedure**:

```bash
# 1. Enable maintenance mode
echo "maintenance" > /tmp/maintenance.flag

# Update NGINX to show maintenance page
docker-compose -f docker-compose.production.yml exec nginx nginx -s reload

# 2. Wait for active requests to complete
sleep 60

# 3. Backup database
./scripts/backup.sh

# 4. Perform maintenance
# - Database migrations
# - System updates
# - Configuration changes

# 5. Test
./scripts/smoke_test.sh

# 6. Disable maintenance mode
rm /tmp/maintenance.flag
docker-compose -f docker-compose.production.yml exec nginx nginx -s reload

# 7. Monitor for 30 minutes
watch -n 30 'curl -s http://localhost:8000/health'
```

### Emergency Maintenance

**No advance notice possible**

```bash
# 1. Quick announcement
# Post to status page / Slack

# 2. Execute fix
# ...

# 3. Verify
curl http://localhost:8000/health

# 4. Update status
# Post resolution notice
```

---

## Emergency Contacts

### On-Call Rotation

| Role | Primary | Backup | Phone | Slack |
|------|---------|--------|-------|-------|
| Platform Engineer | TBD | TBD | TBD | @platform |
| Database Admin | TBD | TBD | TBD | @dba |
| ML Engineer | TBD | TBD | TBD | @ml-team |
| DevOps | TBD | TBD | TBD | @devops |

### Escalation Path

```
L1: On-Call Engineer (0-15 min)
    ↓
L2: Team Lead (15-30 min)
    ↓
L3: Engineering Manager (30-60 min)
    ↓
L4: CTO (> 60 min or P0)
```

### External Contacts

- **Cloud Provider Support**: [Support URL/Phone]
- **Database Vendor**: [Support URL/Phone]
- **Monitoring Service**: [Support URL]
- **Security Team**: security@example.com

---

## Appendix

### Quick Reference Commands

```bash
# Health check
curl http://localhost:8000/health

# Service status
docker-compose -f docker-compose.production.yml ps

# Restart service
docker-compose -f docker-compose.production.yml restart [service]

# Scale service
docker-compose -f docker-compose.production.yml up -d --scale [service]=[count]

# View logs
docker-compose -f docker-compose.production.yml logs -f [service]

# Rollback
./scripts/rollback.sh [version]

# Pre-deployment check
python scripts/pre_deployment_check.py --version [version]
```

### Useful Links

- **Documentation**: `/docs`
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Swagger API**: http://localhost:8000/docs
- **GitHub**: [Repository URL]
- **Wiki**: [Wiki URL]
- **Status Page**: [Status Page URL]

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Maintained by**: DevOps Team
