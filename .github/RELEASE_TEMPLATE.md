# Release Notes Template
# Version: [VERSION]
# Date: [DATE]

## 🎯 Summary

<!-- One-paragraph summary of this release -->

[Brief description of what's in this release]

---

## 📦 Changes

### ✨ Features

- **[Feature Name]**: [Description]
  - Impact: [High/Medium/Low]
  - Components: [app/worker-llm/worker-rl/database]

### 🐛 Bug Fixes

- **[Bug Fix]**: [Description]
  - Issue: [Link to issue]
  - Impact: [Description]

### ⚡ Performance Improvements

- **[Performance Change]**: [Description]
  - Metric: [What improved and by how much]
  - Impact: [High/Medium/Low]

### 🔧 Infrastructure

- **[Infrastructure Change]**: [Description]
  - Components: [Affected services]
  - Migration Required: [Yes/No]

### 📚 Documentation

- **[Documentation Update]**: [Description]

### 🔒 Security

- **[Security Fix]**: [Description]
  - Severity: [Critical/High/Medium/Low]
  - CVE: [If applicable]

---

## 🚀 Deployment

### Prerequisites

- [ ] Review all changes and test in staging
- [ ] Backup database
- [ ] Check resource availability
- [ ] Review configuration changes
- [ ] Notify stakeholders

### Migration Steps

#### 1. Database Migrations

```bash
# Run database migrations
docker-compose -f docker-compose.production.yml exec app alembic upgrade head

# Verify migration
docker-compose -f docker-compose.production.yml exec app alembic current
```

**Rollback:**
```bash
# Rollback to previous version
docker-compose -f docker-compose.production.yml exec app alembic downgrade -1
```

#### 2. Configuration Changes

**Changed Configuration:**
```yaml
# configs/recommended_params.yaml
[Configuration changes here]
```

**Action Required:**
- [ ] Update `.env` file with new variables
- [ ] Review and update config files
- [ ] Restart affected services

#### 3. Service Updates

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Stop services (rolling restart)
docker-compose -f docker-compose.production.yml up -d --no-deps --build app
docker-compose -f docker-compose.production.yml up -d --no-deps --build worker-llm
docker-compose -f docker-compose.production.yml up -d --no-deps --build worker-rl

# Verify health
docker-compose -f docker-compose.production.yml ps
```

#### 4. Post-Deployment Verification

- [ ] Check service health: `curl http://localhost:8000/health`
- [ ] Monitor logs: `docker-compose logs -f --tail=100`
- [ ] Verify worker queues: Check Redis queue status
- [ ] Run smoke tests: `./scripts/smoke_test.sh`
- [ ] Check metrics: Review Grafana dashboards

---

## 🔄 Rollback Procedures

### Quick Rollback (Recommended)

```bash
# Rollback to previous image version
export VERSION=v[PREVIOUS_VERSION]
docker-compose -f docker-compose.production.yml up -d

# Verify rollback
docker-compose -f docker-compose.production.yml ps
```

### Full Rollback with Database

```bash
# 1. Stop services
docker-compose -f docker-compose.production.yml down

# 2. Restore database backup
docker exec fragrance-ai-postgres psql -U fragrance_user -d fragrance_ai < backup_[PREVIOUS_VERSION].sql

# 3. Revert to previous version
export VERSION=v[PREVIOUS_VERSION]
docker-compose -f docker-compose.production.yml up -d

# 4. Verify
./scripts/smoke_test.sh
```

### Rollback Verification Checklist

- [ ] All services are healthy
- [ ] Database integrity verified
- [ ] Worker queues operational
- [ ] API endpoints responding correctly
- [ ] No error spikes in logs
- [ ] Metrics show normal behavior

---

## 📊 Impact Assessment

### Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Average Response Time | [XXX]ms | [YYY]ms | [+/-Z%] |
| Throughput (req/s) | [XXX] | [YYY] | [+/-Z%] |
| Memory Usage | [XXX]GB | [YYY]GB | [+/-Z%] |
| CPU Usage | [XX]% | [YY]% | [+/-Z%] |

### Service Availability

- **Expected Downtime**: [None/X minutes]
- **Affected Services**: [List services]
- **Maintenance Window**: [Time range]

### Breaking Changes

⚠️ **ATTENTION**: This release contains breaking changes

- **[Breaking Change 1]**
  - What changed: [Description]
  - Action required: [What users need to do]
  - Migration guide: [Link or instructions]

---

## 🧪 Testing

### Test Coverage

- Unit Tests: ✅ Passing (XX% coverage)
- Integration Tests: ✅ Passing
- End-to-End Tests: ✅ Passing
- Performance Tests: ✅ Passing
- Security Scan: ✅ No critical issues

### Test Scenarios Verified

- [ ] LLM ensemble inference working correctly
- [ ] RL training pipeline functional
- [ ] Database operations (CRUD)
- [ ] Cache invalidation
- [ ] Worker scaling (LLM, RL)
- [ ] Health checks responding
- [ ] Metrics collection
- [ ] Error handling and recovery

---

## 🔗 Dependencies

### Updated Dependencies

| Package | Previous | New | Notes |
|---------|----------|-----|-------|
| [package-name] | [X.Y.Z] | [X.Y.Z] | [Reason for update] |

### Security Updates

- **[Package Name]**: Fixed vulnerability [CVE-XXXX-XXXXX]
  - Severity: [Critical/High/Medium/Low]
  - Impact: [Description]

---

## 📝 Notes

### Known Issues

- **[Issue Description]**
  - Workaround: [If available]
  - Tracking: [Issue link]

### Deprecated Features

⚠️ The following features are deprecated and will be removed in [Version]:

- **[Feature Name]**: [Reason for deprecation]
  - Alternative: [What to use instead]
  - Removal date: [Version/Date]

### Future Work

- [Planned improvement]
- [Future feature]

---

## 👥 Contributors

- [@username]: [Contribution description]
- [@username]: [Contribution description]

---

## 📞 Support

- Documentation: [Link to docs]
- Issues: [Link to issue tracker]
- Slack: [Link to Slack channel]
- Email: support@fragrance-ai.example.com

---

## 🔖 Version Information

- **Version**: [VERSION]
- **Release Date**: [DATE]
- **Git Tag**: `v[VERSION]`
- **Git Commit**: `[COMMIT_SHA]`
- **Docker Images**:
  - `fragrance-ai-app:[VERSION]`
  - `fragrance-ai-worker-llm:[VERSION]`
  - `fragrance-ai-worker-rl:[VERSION]`

---

**Full Changelog**: https://github.com/[org]/[repo]/compare/v[PREV_VERSION]...v[VERSION]
