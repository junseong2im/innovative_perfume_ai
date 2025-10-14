# Autoscaling Rules

**Services**: app / worker-llm / worker-rl
**Goal**: Maintain performance while optimizing resource usage

---

## 1. Service Scaling Rules

### 1.1 App (FastAPI API Layer)

**Trigger Metrics:**

| Metric | Scale Up Threshold | Scale Down Threshold | Cooldown |
|--------|-------------------|---------------------|----------|
| CPU Usage | >70% for 3 minutes | <30% for 10 minutes | 5 minutes |
| Memory Usage | >80% for 3 minutes | <40% for 10 minutes | 5 minutes |
| Request Queue | >100 requests | <20 requests | 3 minutes |
| Response Time p95 | >2 seconds for 5 minutes | <0.5 seconds for 15 minutes | 5 minutes |
| Error Rate | >5% for 2 minutes | - (scale up only) | - |

**Scaling Parameters:**
```yaml
min_replicas: 2
max_replicas: 10
scale_up_increment: 1
scale_down_increment: 1
target_cpu_utilization: 60%
target_memory_utilization: 70%
```

**Command:**
```bash
# Docker Compose
docker-compose -f docker-compose.production.yml up -d --scale app=5

# Kubernetes HPA
kubectl autoscale deployment app \
  --min=2 --max=10 \
  --cpu-percent=60 \
  --memory-percent=70
```

---

### 1.2 Worker-LLM (LLM Inference)

**Trigger Metrics:**

| Metric | Scale Up Threshold | Scale Down Threshold | Cooldown |
|--------|-------------------|---------------------|----------|
| Queue Length | >50 tasks | <10 tasks for 15 minutes | 10 minutes |
| LLM Latency p95 | >4 seconds for 5 minutes | <2 seconds for 20 minutes | 10 minutes |
| GPU Utilization | >85% for 5 minutes | <40% for 20 minutes | 10 minutes |
| VRAM Usage | >80% for 5 minutes | <60% for 20 minutes | 10 minutes |
| Task Failure Rate | >10% for 3 minutes | - (scale up only) | - |

**Scaling Parameters:**
```yaml
min_replicas: 2
max_replicas: 8
scale_up_increment: 1
scale_down_increment: 1
warmup_time: 180 seconds  # Model loading time
gpu_check: required  # Ensure GPU availability before scaling
```

**GPU Placement Strategy:**
- **0-2 replicas**: Use GPU 0
- **3-4 replicas**: Use GPU 0, GPU 1
- **5+ replicas**: Distribute across all available GPUs

**Command:**
```bash
# Docker Compose (CPU)
docker-compose -f docker-compose.production.yml up -d --scale worker-llm=3

# Docker Compose (GPU) - with nvidia-docker runtime
USE_GPU=true docker-compose -f docker-compose.production.yml up -d --scale worker-llm=2

# Kubernetes HPA (custom metrics)
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: worker-llm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: worker-llm
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: llm_queue_length
      target:
        type: AverageValue
        averageValue: "30"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 600
      policies:
      - type: Pods
        value: 1
        periodSeconds: 180
EOF
```

---

### 1.3 Worker-RL (RL Training)

**Trigger Metrics:**

| Metric | Scale Up Threshold | Scale Down Threshold | Cooldown |
|--------|-------------------|---------------------|----------|
| Training Queue | >20 experiments | <5 experiments for 30 minutes | 15 minutes |
| Episode Duration | >5 minutes for 10 minutes | <2 minutes for 30 minutes | 15 minutes |
| GPU Utilization | >90% for 10 minutes | <50% for 30 minutes | 20 minutes |
| Checkpoint Frequency | >10/hour | <2/hour for 1 hour | 20 minutes |

**Scaling Parameters:**
```yaml
min_replicas: 1
max_replicas: 4
scale_up_increment: 1
scale_down_increment: 1
warmup_time: 300 seconds  # Checkpoint loading time
checkpoint_sync: required  # Ensure checkpoint consistency
```

**Command:**
```bash
# Docker Compose
docker-compose -f docker-compose.production.yml up -d --scale worker-rl=2

# Kubernetes HPA
kubectl autoscale deployment worker-rl \
  --min=1 --max=4 \
  --cpu-percent=80
```

---

## 2. GPU/VRAM Management

### 2.1 Multi-Model Loading Strategy

**Goal**: Load all 3 models (Qwen, Mistral, Llama) with ≥20% VRAM headroom

**GPU Configuration:**

```python
# Single GPU (≥40GB VRAM required)
GPU 0:
  - Qwen 32B (4-bit): ~16GB
  - Mistral 7B (4-bit): ~4GB
  - Llama3 8B (4-bit): ~5GB
  Total: ~25GB
  Headroom: 15GB (37.5% on 40GB GPU) ✅

# Multi-GPU (24GB VRAM each)
GPU 0:
  - Qwen 32B (4-bit): ~16GB
  Headroom: 8GB (33%) ✅

GPU 1:
  - Mistral 7B (4-bit): ~4GB
  - Llama3 8B (4-bit): ~5GB
  Total: ~9GB
  Headroom: 15GB (62.5%) ✅
```

**Monitoring Command:**
```bash
# Check VRAM headroom
python scripts/monitor_gpu_vram.py --once

# Continuous monitoring
python scripts/monitor_gpu_vram.py
```

**Auto-Unload Policy:**

```python
# fragrance_ai/gpu/vram_manager.py
class VRAMManager:
    def __init__(self):
        self.headroom_threshold = 0.20  # 20%
        self.warning_threshold = 0.25   # 25%

    def check_and_unload(self):
        """Check VRAM usage and unload models if needed"""
        gpus = get_gpu_info()

        for gpu in gpus:
            free_percent = gpu['memory_free_percent'] / 100

            if free_percent < self.headroom_threshold:
                # Critical: Unload least-used model
                self.unload_least_used_model(gpu['index'])
            elif free_percent < self.warning_threshold:
                # Warning: Log only
                logger.warning(f"GPU {gpu['index']}: Low VRAM ({free_percent*100:.1f}% free)")
```

---

## 3. Cache Hit Rate Monitoring

### 3.1 Target Metrics

| Mode | Cache Hit Rate Target | Current Baseline | Action if Below |
|------|----------------------|------------------|-----------------|
| Fast | ≥60% | - | Increase TTL to 10 minutes |
| Balanced | ≥60% | - | Increase TTL to 20 minutes |
| Creative | No target (dynamic) | - | Keep TTL at 30 minutes |

**Rationale**: Creative mode generates unique results, so cache hit rate is naturally lower.

### 3.2 Cache Metrics

**Prometheus Metrics:**
```python
# fragrance_ai/monitoring/operations_metrics.py

# Cache hit rate by mode
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate by mode',
    ['mode']
)

# Cache operations
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['mode']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['mode']
)

# TTL expiration
cache_ttl_expired_total = Counter(
    'cache_ttl_expired_total',
    'Cache entries expired due to TTL',
    ['mode']
)
```

**Recording:**
```python
# In your cache lookup function
def get_from_cache(key: str, mode: str) -> Optional[Any]:
    cached = redis_client.get(key)

    if cached:
        cache_hits_total.labels(mode=mode).inc()
        hit_count = cache_hits_total.labels(mode=mode)._value.get()
        miss_count = cache_misses_total.labels(mode=mode)._value.get()

        total = hit_count + miss_count
        if total > 0:
            hit_rate = hit_count / total
            cache_hit_rate.labels(mode=mode).set(hit_rate)

        return json.loads(cached)
    else:
        cache_misses_total.labels(mode=mode).inc()
        return None
```

### 3.3 Grafana Dashboard Panel

```json
{
  "id": 22,
  "title": "Cache Hit Rate by Mode (Target: ≥60% for Fast/Balanced)",
  "type": "timeseries",
  "targets": [
    {
      "expr": "cache_hit_rate{mode=\"fast\"}",
      "legendFormat": "Fast"
    },
    {
      "expr": "cache_hit_rate{mode=\"balanced\"}",
      "legendFormat": "Balanced"
    },
    {
      "expr": "cache_hit_rate{mode=\"creative\"}",
      "legendFormat": "Creative (no target)"
    }
  ],
  "fieldConfig": {
    "defaults": {
      "unit": "percentunit",
      "min": 0,
      "max": 1
    },
    "overrides": [
      {
        "matcher": {"id": "byName", "options": "Fast"},
        "properties": [
          {
            "id": "thresholds",
            "value": {
              "mode": "absolute",
              "steps": [
                {"value": null, "color": "red"},
                {"value": 0.60, "color": "green"}
              ]
            }
          }
        ]
      }
    ]
  }
}
```

### 3.4 Auto-Tuning TTL

```python
# scripts/auto_tune_cache_ttl.py
"""
Auto-tune cache TTL based on hit rate
"""

def auto_tune_ttl(mode: str):
    """Adjust TTL to maintain 60% hit rate"""
    hit_rate = get_cache_hit_rate(mode)
    current_ttl = get_current_ttl(mode)

    if hit_rate < 0.60:
        # Increase TTL by 50%
        new_ttl = int(current_ttl * 1.5)
        logger.info(f"Increasing {mode} TTL: {current_ttl}s → {new_ttl}s (hit rate: {hit_rate*100:.1f}%)")
        set_ttl(mode, new_ttl)

    elif hit_rate > 0.75 and current_ttl > 300:
        # Decrease TTL to free memory (if hit rate is very high)
        new_ttl = int(current_ttl * 0.8)
        logger.info(f"Decreasing {mode} TTL: {current_ttl}s → {new_ttl}s (hit rate: {hit_rate*100:.1f}%)")
        set_ttl(mode, new_ttl)
```

---

## 4. Scaling Decision Matrix

### 4.1 Combined Conditions

| Condition | App | Worker-LLM | Worker-RL | Priority |
|-----------|-----|------------|-----------|----------|
| High traffic + High latency | Scale up | Scale up | - | P0 |
| Queue buildup | - | Scale up | Scale up | P1 |
| Low utilization (all services) | Scale down | Scale down | Scale down | P3 |
| GPU VRAM < 20% | - | Do not scale up | Do not scale up | P0 |
| Error rate spike | Scale up | Scale up | - | P0 |
| Off-peak hours (01:00-06:00) | Scale down to min | Scale down to min | Scale to 1 | P3 |

### 4.2 Scaling Sequence

**Scale Up:**
1. Check GPU VRAM availability (worker-llm only)
2. Verify resource limits not exceeded
3. Trigger scale up
4. Wait for warmup period (3-5 minutes)
5. Verify health checks pass
6. Route traffic to new instance

**Scale Down:**
1. Check if below threshold for cooldown period
2. Drain connections gracefully (60 second timeout)
3. Wait for active tasks to complete
4. Terminate instance
5. Verify remaining instances healthy

---

## 5. Monitoring and Alerts

### 5.1 Prometheus Alerts

```yaml
# /monitoring/prometheus_alerts.yml

groups:
- name: autoscaling_alerts
  interval: 30s
  rules:

  # App scaling alerts
  - alert: AppHighCPU
    expr: avg(rate(container_cpu_usage_seconds_total{name="fragrance-ai-app"}[5m])) > 0.70
    for: 3m
    labels:
      severity: warning
      service: app
    annotations:
      summary: "App CPU usage above 70%"
      action: "Consider scaling up app replicas"

  # Worker-LLM queue alerts
  - alert: LLMQueueBuildup
    expr: llm_queue_length > 50
    for: 2m
    labels:
      severity: warning
      service: worker-llm
    annotations:
      summary: "LLM task queue > 50"
      action: "Scale up worker-llm replicas"

  # VRAM headroom alerts
  - alert: VRAMHeadroomLow
    expr: (nvidia_gpu_memory_free_bytes / nvidia_gpu_memory_total_bytes) < 0.20
    for: 5m
    labels:
      severity: critical
      service: worker-llm
    annotations:
      summary: "GPU VRAM headroom < 20%"
      action: "Do NOT scale up. Consider unloading models."

  # Cache hit rate alerts
  - alert: CacheHitRateLow
    expr: cache_hit_rate{mode!="creative"} < 0.60
    for: 10m
    labels:
      severity: info
      service: cache
    annotations:
      summary: "Cache hit rate < 60% for {{$labels.mode}} mode"
      action: "Consider increasing cache TTL"
```

### 5.2 Grafana Alerts

Configure Grafana to send alerts to:
- **Slack**: `#fragrance-ai-alerts`
- **PagerDuty**: For P0/P1 alerts
- **Email**: `oncall@example.com`

---

## 6. Manual Scaling Commands

### 6.1 Quick Reference

```bash
# Scale up all services
docker-compose -f docker-compose.production.yml up -d \
  --scale app=5 \
  --scale worker-llm=4 \
  --scale worker-rl=2

# Scale down to minimum
docker-compose -f docker-compose.production.yml up -d \
  --scale app=2 \
  --scale worker-llm=2 \
  --scale worker-rl=1

# Check current replicas
docker-compose -f docker-compose.production.yml ps

# View resource usage
docker stats

# View logs for specific service
docker-compose -f docker-compose.production.yml logs -f worker-llm
```

### 6.2 Kubernetes (if migrating)

```bash
# Set autoscaling
kubectl autoscale deployment app --min=2 --max=10 --cpu-percent=60
kubectl autoscale deployment worker-llm --min=2 --max=8 --cpu-percent=70
kubectl autoscale deployment worker-rl --min=1 --max=4 --cpu-percent=80

# View HPA status
kubectl get hpa

# Manually scale
kubectl scale deployment worker-llm --replicas=5

# View pod status
kubectl get pods -l app=worker-llm
```

---

## 7. Cost Optimization

### 7.1 Scaling Schedule

```python
# scripts/scheduled_scaling.py
"""
Scheduled scaling based on traffic patterns
"""

SCALING_SCHEDULE = {
    # Peak hours (09:00-18:00): High replicas
    "peak": {
        "hours": range(9, 18),
        "app": 5,
        "worker_llm": 4,
        "worker_rl": 2
    },

    # Off-peak (01:00-06:00): Minimum replicas
    "off_peak": {
        "hours": range(1, 6),
        "app": 2,
        "worker_llm": 2,
        "worker_rl": 1
    },

    # Normal (other hours): Standard replicas
    "normal": {
        "app": 3,
        "worker_llm": 3,
        "worker_rl": 1
    }
}
```

---

## 8. Testing Autoscaling

### 8.1 Load Test

```bash
# Use Apache Bench to simulate load
ab -n 10000 -c 100 http://localhost:8000/api/v1/dna/create

# Use Locust for more realistic load
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### 8.2 Verify Scaling Behavior

```bash
# Monitor scaling in real-time
watch -n 2 'docker-compose -f docker-compose.production.yml ps'

# Check resource usage
docker stats --no-stream

# View scaling decisions in logs
docker-compose -f docker-compose.production.yml logs -f | grep -i "scale"
```

---

*Document Version: 1.0*
*Last Updated: 2025-10-14*
