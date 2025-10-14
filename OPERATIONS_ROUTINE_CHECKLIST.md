# 2-Week Operations Routine Checklist

**Team**: Fragrance AI SRE/DevOps
**Goal**: Maintain 99.9% uptime, optimize performance, prevent incidents

---

## Daily Routine (10 minutes)

**Time**: Every morning at 09:00
**Duration**: ~10 minutes
**Owner**: On-call engineer

### 1. KPI Dashboard Review

**Grafana URL**: http://localhost:3000/d/operations

#### 1.1 API Performance (p95 Latency)

- [ ] **Fast mode**: p95 < 2.5 seconds
- [ ] **Balanced mode**: p95 < 3.2 seconds
- [ ] **Creative mode**: p95 < 4.5 seconds

**Action if exceeded**:
```bash
# Check if downshift occurred
curl http://localhost:8001/admin/downshift/stats

# Check app/worker health
docker-compose -f docker-compose.production.yml ps

# Scale up if needed
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=4
```

#### 1.2 Error Rate

- [ ] **4xx errors**: < 5% of total requests
- [ ] **5xx errors**: < 1% of total requests

**Query:**
```promql
# 4xx rate
sum(rate(http_requests_total{status=~"4.."}[5m]))
/
sum(rate(http_requests_total[5m]))

# 5xx rate
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))
```

**Action if exceeded**:
- Check application logs: `docker-compose logs -f app --tail=100`
- Review Sentry errors (if configured)
- Check for recent deployments

#### 1.3 RL Update Trends

- [ ] **Average reward**: Moving average stable or increasing
- [ ] **Training loss**: Decreasing or stable (< 0.5)
- [ ] **Entropy**: 0.5 ~ 2.0 (exploration vs exploitation balance)

**Query:**
```promql
# Reward trend (10-episode moving average)
rl_reward_ma{window="10"}

# Loss trend
rate(rl_training_loss_total[1h])

# Entropy
rl_entropy{algorithm="PPO"}
```

**Action if anomalies**:
- Diverging loss → Check for NaN, reduce learning rate
- Reward plateau → Review hyperparameters, increase entropy coefficient
- Zero entropy → Model converged too fast, increase exploration

#### 1.4 System Health

- [ ] **CPU usage**: < 70% (app, workers)
- [ ] **Memory usage**: < 80% (all services)
- [ ] **GPU VRAM**: ≥ 20% free
- [ ] **Disk usage**: < 80%

**Commands:**
```bash
# System resources
docker stats --no-stream

# GPU VRAM
python scripts/monitor_gpu_vram.py --once

# Disk usage
df -h
```

### 2. Alert Review

- [ ] Review overnight alerts (Grafana/Prometheus)
- [ ] Check if any alerts are still firing
- [ ] Verify auto-remediation actions worked

### 3. Daily Report Template

```markdown
# Daily Operations Report - [Date]

## KPI Summary
- Fast p95: [X.X]s ✅/❌
- Balanced p95: [X.X]s ✅/❌
- Creative p95: [X.X]s ✅/❌
- Error rate: [X.X]% ✅/❌
- RL reward MA: [X.XX] ✅/❌

## Incidents
- [None / List incidents]

## Actions Taken
- [None / List actions]

## Notes
- [Any observations]
```

---

## Weekly Routine (30 minutes)

**Time**: Every Friday at 14:00
**Duration**: ~30 minutes
**Owner**: ML Engineer + SRE

### 1. Hyperparameter Review

#### 1.1 RL Training Hyperparameters

**Review Metrics:**

| Parameter | Current Value | Last Week | Trend | Action |
|-----------|---------------|-----------|-------|--------|
| Learning Rate | 5e-5 | - | - | Monitor |
| Entropy Coefficient | 0.01 | - | - | Monitor |
| Clip Epsilon | 0.2 | - | - | Monitor |
| Reward MA (100-ep) | - | - | - | Monitor |
| Reward Std Dev | - | - | - | Monitor |

**Query:**
```promql
# Reward statistics
avg_over_time(rl_reward_ma{window="100"}[7d])
stddev_over_time(rl_reward_ma{window="100"}[7d])

# Entropy over time
avg_over_time(rl_entropy[7d])
```

**Decision Matrix:**

| Condition | Action |
|-----------|--------|
| Reward increasing steadily | ✅ No change |
| Reward plateau > 2 weeks | Increase learning rate by 20% OR increase entropy coef |
| High reward variance (std > 0.3) | Decrease learning rate by 30% |
| Entropy < 0.3 for 3+ days | Increase entropy coef from 0.01 → 0.02 |
| Training loss > 1.0 consistently | Reduce learning rate, check for NaN |

#### 1.2 Reward Normalization

- [ ] Check reward distribution (histogram)
- [ ] Verify clipping is effective (-1 to +1 range)
- [ ] Review advantage normalization parameters

**Query:**
```promql
# Reward distribution
histogram_quantile(0.50, rl_reward_bucket)  # Median
histogram_quantile(0.95, rl_reward_bucket)  # P95
```

#### 1.3 Cache Performance

- [ ] **Fast mode hit rate**: ≥ 60%
- [ ] **Balanced mode hit rate**: ≥ 60%
- [ ] **Creative mode hit rate**: (no target)

**Query:**
```promql
cache_hit_rate{mode="fast"}
cache_hit_rate{mode="balanced"}
```

**Action if below 60%:**
```bash
# Increase TTL
curl -X POST http://localhost:8001/admin/cache/ttl \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast", "ttl_seconds": 600}'
```

### 2. Model Update Planning

#### 2.1 Model Performance Review

- [ ] Review model quality metrics (user ratings, acceptance rate)
- [ ] Check for model drift (input distribution changes)
- [ ] Evaluate need for fine-tuning or retraining

#### 2.2 Model Update Schedule

| Model | Current Version | Next Update | Reason |
|-------|----------------|-------------|--------|
| Qwen 32B | v1.0 | - | Stable |
| Mistral 7B | v1.0 | - | Stable |
| Llama3 8B | v1.0 | - | Stable |
| Policy Network | - | - | Continuous learning |

#### 2.3 Checkpoint Management

- [ ] Review checkpoint growth (disk usage)
- [ ] Archive old checkpoints (> 30 days)
- [ ] Verify backup integrity

**Commands:**
```bash
# Check checkpoint disk usage
du -sh ./checkpoints/*

# Archive old checkpoints
find ./checkpoints -mtime +30 -type f -name "*.pt" -exec tar -czf checkpoints_archive_$(date +%Y%m%d).tar.gz {} +

# Verify baseline integrity
python scripts/verify_model_integrity.py
```

### 3. Capacity Planning

#### 3.1 Resource Trend Analysis

- [ ] CPU usage trend (7-day average)
- [ ] Memory usage trend
- [ ] GPU VRAM usage trend
- [ ] Disk I/O trend
- [ ] Network bandwidth usage

**Query:**
```promql
# 7-day CPU trend
avg_over_time(container_cpu_usage_seconds_total[7d])

# 7-day memory trend
avg_over_time(container_memory_usage_bytes[7d])
```

#### 3.2 Scaling Forecast

**Traffic Growth:**
- Current QPS (queries per second): _____
- Week-over-week growth: _____%
- Estimated capacity needed in 4 weeks: _____

**Action:**
- If growth > 20%/week: Plan hardware upgrade
- If growth > 50%/week: Implement aggressive autoscaling

### 4. Security and Compliance

- [ ] Review access logs for anomalies
- [ ] Check SSL certificate expiration (< 30 days remaining)
- [ ] Verify PII masking is effective (spot check logs)
- [ ] Review dependency vulnerabilities (`pip list --outdated`)

### 5. Weekly Report Template

```markdown
# Weekly Operations Report - Week of [Date]

## Hyperparameter Review
### RL Training
- Reward MA (100-ep): [X.XX] (trend: ↑/↓/→)
- Entropy: [X.XX] (target: 0.5-2.0)
- Loss: [X.XX] (trend: ↑/↓)
- **Action**: [None / Adjust learning rate / Increase entropy]

### Cache Performance
- Fast hit rate: [XX]% (target: ≥60%)
- Balanced hit rate: [XX]% (target: ≥60%)
- **Action**: [None / Increase TTL]

## Model Update Planning
- [ ] Models stable, no updates needed
- [ ] Plan to update [model name] on [date]

## Capacity Planning
- Traffic growth: [XX]% WoW
- Resource utilization: [CPU/Memory/GPU status]
- **Action**: [None / Scale up / Plan upgrade]

## Security
- [ ] No security issues detected
- [ ] SSL cert expires on [date] (renewal planned)
- [ ] Dependencies up to date

## Next Week Priorities
1. [Priority 1]
2. [Priority 2]
3. [Priority 3]
```

---

## Incident Review (Blameless Postmortem)

**Trigger**: After any P0/P1 incident
**Duration**: 30-60 minutes
**Attendees**: SRE, ML Engineer, relevant stakeholders

### 1. Incident Summary

**Template:**

```markdown
# Incident Postmortem: [Incident Title]

**Date**: [YYYY-MM-DD]
**Duration**: [Start time] - [End time] (X hours Y minutes)
**Severity**: P0 / P1 / P2
**Affected Services**: [List services]
**User Impact**: [Description]

---

## Timeline

| Time | Event |
|------|-------|
| HH:MM | Incident started (first alert) |
| HH:MM | On-call engineer notified |
| HH:MM | Incident commander assigned |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |
| HH:MM | Incident closed |

---

## Root Cause Analysis

### What Happened?
[Detailed description of what went wrong]

### Why Did It Happen?
[Root cause - technical and process failures]

### Contributing Factors
- [Factor 1]
- [Factor 2]
- [Factor 3]

---

## Impact Assessment

### Metrics
- Requests affected: [number]
- Error rate: [percentage]
- Users impacted: [number]
- Revenue impact: $[amount] (if applicable)
- Downtime: [duration]

### Customer Impact
- [Description of user experience during incident]
- [Number of support tickets filed]

---

## Resolution Steps

### What We Did to Fix It
1. [Step 1]
2. [Step 2]
3. [Step 3]

### Why This Fixed It
[Explanation]

---

## Action Items (JIRA Tickets)

### Immediate (this week)
- [ ] **[JIRA-001]** [Action item 1] - Owner: [Name] - Due: [Date]
- [ ] **[JIRA-002]** [Action item 2] - Owner: [Name] - Due: [Date]

### Short-term (2-4 weeks)
- [ ] **[JIRA-003]** [Action item 3] - Owner: [Name] - Due: [Date]
- [ ] **[JIRA-004]** [Action item 4] - Owner: [Name] - Due: [Date]

### Long-term (1-3 months)
- [ ] **[JIRA-005]** [Action item 5] - Owner: [Name] - Due: [Date]

---

## Prevention Measures

### Technical Improvements
- [Improvement 1]
- [Improvement 2]

### Process Improvements
- [Improvement 1]
- [Improvement 2]

### Monitoring and Alerting
- [New alert 1]
- [New dashboard 2]

---

## Lessons Learned

### What Went Well
- [Positive 1]
- [Positive 2]

### What Went Poorly
- [Negative 1]
- [Negative 2]

### Lucky Breaks
- [Lucky break 1]

---

## Follow-up

- [ ] Share postmortem with team
- [ ] Update runbooks with new procedures
- [ ] Schedule follow-up review in 30 days to verify action items completed
```

### 2. Action Item Tracking

**JIRA Template:**

```
Title: [Postmortem Action Item] - [Short Description]

Type: Task / Bug / Improvement
Priority: Blocker / Critical / Major / Minor
Assignee: [Name]
Due Date: [Date]
Labels: postmortem, incident-[YYYY-MM-DD]

Description:
Context: This action item is from the postmortem of incident "[Incident Title]" on [Date]

Action: [Detailed description of what needs to be done]

Acceptance Criteria:
- [ ] [Criterion 1]
- [ ] [Criterion 2]
- [ ] [Criterion 3]

Related Postmortem: [Link to postmortem document]
```

### 3. Blameless Culture Guidelines

**DO:**
- Focus on system and process failures, not individual mistakes
- Ask "how" and "why" questions, not "who"
- Treat failures as learning opportunities
- Celebrate transparency and honesty
- Reward reporting of near-misses

**DON'T:**
- Blame individuals for incidents
- Punish people for making honest mistakes
- Hide or downplay incidents
- Skip postmortems for "small" incidents
- Leave action items untracked or incomplete

---

## Monthly Review (60 minutes)

**Time**: First Monday of each month at 10:00
**Duration**: ~60 minutes
**Attendees**: Full team

### 1. KPI Trends (30-day review)

- [ ] Average p95 latency by mode (30-day trend)
- [ ] Error rate trend
- [ ] Availability / uptime percentage
- [ ] RL model performance (reward trend)
- [ ] User satisfaction metrics

### 2. Incident Summary

- [ ] Total incidents: P0/P1/P2 count
- [ ] Mean time to detect (MTTD)
- [ ] Mean time to resolve (MTTR)
- [ ] Incident categories (most common causes)

### 3. Action Item Completion Rate

- [ ] Percentage of postmortem action items completed on time
- [ ] Outstanding action items (aging report)

### 4. Strategic Planning

- [ ] Upcoming model updates or migrations
- [ ] Infrastructure improvements needed
- [ ] New features requiring operations support
- [ ] Capacity planning for next quarter

---

## Emergency Contacts

| Role | Name | Phone | Email | Slack |
|------|------|-------|-------|-------|
| On-call Engineer (Primary) | - | - | - | @oncall-primary |
| On-call Engineer (Secondary) | - | - | - | @oncall-secondary |
| Incident Commander | - | - | - | @incident-commander |
| ML Lead | - | - | - | @ml-lead |
| SRE Lead | - | - | - | @sre-lead |

---

## Quick Reference

### Common Commands

```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# Scale services
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=4

# View logs
docker-compose -f docker-compose.production.yml logs -f app

# Check GPU VRAM
python scripts/monitor_gpu_vram.py --once

# Execute runbook
python -c "from fragrance_ai.sre.runbooks import get_runbook_manager; \
           m = get_runbook_manager(); \
           m.execute_runbook('llm_failure_response')"

# Check cache stats
curl http://localhost:8001/admin/cache/stats

# Trigger manual rollback
curl -X POST http://localhost:8001/admin/checkpoint/rollback
```

### Dashboards

- **Grafana Operations**: http://localhost:3000/d/operations
- **Prometheus**: http://localhost:9090
- **API Health**: http://localhost:8001/health
- **Downshift Status**: http://localhost:8001/admin/downshift/stats

---

*Document Version: 1.0*
*Last Updated: 2025-10-14*
