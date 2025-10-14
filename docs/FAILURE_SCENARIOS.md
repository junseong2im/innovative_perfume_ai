# Failure Scenario Handbook

Comprehensive guide for handling failures in the Fragrance AI system with automatic recovery strategies.

## Table of Contents

1. [Overview](#overview)
2. [Model Failure Scenarios](#model-failure-scenarios)
3. [Automatic Downshift Strategies](#automatic-downshift-strategies)
4. [Cache Management](#cache-management)
5. [Queue Management](#queue-management)
6. [Database Failures](#database-failures)
7. [Monitoring & Alerting](#monitoring--alerting)
8. [Recovery Procedures](#recovery-procedures)

---

## Overview

### Failure Categories

| Category | Severity | Auto-Recovery | Manual Intervention |
|----------|----------|---------------|---------------------|
| **Model Failure** | High | ✅ Yes (downshift) | If all models fail |
| **Cache Miss** | Low | ✅ Yes (fallback to compute) | No |
| **Queue Overflow** | Medium | ✅ Yes (reject new) | Scale workers |
| **Database Error** | High | ⚠️ Partial (read-only mode) | Yes |
| **Worker Crash** | Medium | ✅ Yes (restart) | If persistent |
| **OOM (GPU)** | High | ✅ Yes (enable quantization) | If recurring |

### Recovery Strategy Principles

1. **Graceful Degradation**: Reduce quality before rejecting requests
2. **Automatic Downshift**: Fall back to simpler modes
3. **Fast Failure Detection**: Health checks every 10-30 seconds
4. **Transparent to Users**: Return partial results with warnings
5. **Self-Healing**: Automatic retry and recovery

---

## Model Failure Scenarios

### Scenario 1: Qwen Model Down

**Impact**: 🔴 Critical - Qwen is the primary LLM for all modes

**Detection**:
```python
try:
    qwen_output = qwen_model.generate(prompt, timeout=10)
except (TimeoutError, ModelLoadError, OOMError) as e:
    logger.error(f"Qwen model failure: {e}")
    # Trigger recovery
```

**Automatic Recovery Strategy**:

1. **Immediate Fallback**: Use cached similar requests (semantic search)
2. **Retry with Reduced Context**: Shorten prompt, lower max_tokens
3. **Fallback to Mock**: Return default brief with warning

**Implementation**:

```python
class QwenFailureHandler:
    def __init__(self):
        self.failure_count = 0
        self.last_failure_time = None
        self.recovery_strategies = [
            self.strategy_cache_lookup,
            self.strategy_retry_reduced_context,
            self.strategy_mock_output
        ]

    def handle_failure(self, user_text: str, mode: str) -> CreativeBrief:
        self.failure_count += 1
        self.last_failure_time = time.time()

        # Try recovery strategies in order
        for strategy in self.recovery_strategies:
            try:
                result = strategy(user_text, mode)
                if result:
                    logger.info(f"Qwen failure recovered via {strategy.__name__}")
                    return result
            except Exception as e:
                logger.warning(f"Recovery strategy {strategy.__name__} failed: {e}")
                continue

        # All strategies failed
        raise ServiceUnavailableError("Qwen model unavailable, no recovery possible")

    def strategy_cache_lookup(self, user_text: str, mode: str) -> Optional[CreativeBrief]:
        """Strategy 1: Search cache for similar requests"""
        # Semantic similarity search
        similar_requests = cache.search_similar(user_text, threshold=0.8, limit=1)
        if similar_requests:
            logger.info(f"Cache hit for similar request (similarity: {similar_requests[0].score:.2f})")
            return similar_requests[0].brief
        return None

    def strategy_retry_reduced_context(self, user_text: str, mode: str) -> Optional[CreativeBrief]:
        """Strategy 2: Retry with reduced context"""
        # Reduce max_tokens by 50%, increase timeout
        try:
            result = qwen_model.generate(
                user_text,
                max_tokens=256,  # Reduced from 512
                timeout=30,      # Increased from 10
                temperature=0.5  # More deterministic
            )
            return result
        except Exception:
            return None

    def strategy_mock_output(self, user_text: str, mode: str) -> CreativeBrief:
        """Strategy 3: Return default mock output with warning"""
        # Extract basic keywords (rule-based)
        keywords = extract_keywords(user_text)

        # Generate basic brief from keywords
        brief = generate_fallback_brief(keywords)
        brief.metadata = {
            "warning": "Generated from fallback due to model failure",
            "reliability": "low",
            "model": "rule-based-fallback"
        }
        return brief
```

**Monitoring**:
```python
# Alert if Qwen failure rate > 10% over 5 minutes
if qwen_failure_rate > 0.1:
    alert("Qwen model degraded", severity="high")
```

### Scenario 2: Mistral Model Down

**Impact**: 🟡 Medium - Validation step, can be skipped temporarily

**Detection**:
```python
try:
    validation = mistral_model.validate(brief, timeout=5)
except (TimeoutError, ModelLoadError) as e:
    logger.warning(f"Mistral validation failed: {e}")
    # Continue without validation
```

**Automatic Recovery Strategy**:

1. **Skip Validation**: Return Qwen output without validation (with warning)
2. **Rule-Based Validation**: Use simple rule-based checks
3. **Retry Later**: Queue validation task for later

**Implementation**:

```python
class MistralFailureHandler:
    def handle_failure(self, brief: CreativeBrief) -> ValidationReport:
        logger.warning("Mistral unavailable, using rule-based validation")

        # Strategy 1: Rule-based validation
        validation = self.rule_based_validation(brief)

        # Strategy 2: If critical errors, reject
        if validation.has_critical_errors():
            raise ValidationError("Brief has critical errors, cannot proceed without Mistral")

        # Strategy 3: Return with warning
        validation.warnings.append("Validated with rule-based fallback (Mistral unavailable)")
        return validation

    def rule_based_validation(self, brief: CreativeBrief) -> ValidationReport:
        """Simple rule-based validation"""
        errors = []
        warnings = []

        # Check intensity range
        if not (0.0 <= brief.intensity <= 1.0):
            errors.append(f"intensity out of range: {brief.intensity}")

        # Check complexity range
        if not (0.0 <= brief.complexity <= 1.0):
            errors.append(f"complexity out of range: {brief.complexity}")

        # Check notes_preference
        for note, value in brief.notes_preference.items():
            if not (0.0 <= value <= 1.0):
                errors.append(f"{note} preference out of range: {value}")

        # Check product category
        valid_categories = ["EDP", "EDT", "Cologne", "Body Spray"]
        if brief.product_category not in valid_categories:
            errors.append(f"Invalid product_category: {brief.product_category}")

        # Warning if validation incomplete
        warnings.append("Rule-based validation only (IFRA check skipped)")

        return ValidationReport(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            ifra_compliance=None  # Unknown without Mistral
        )
```

### Scenario 3: Llama Model Down (Creative Mode)

**Impact**: 🟢 Low - Only affects creative hints in creative mode

**Detection**:
```python
try:
    hints = llama_model.generate_hints(brief, timeout=10)
except Exception as e:
    logger.info(f"Llama creative hints failed: {e}")
    # Continue without hints
```

**Automatic Recovery Strategy**:

1. **Skip Creative Hints**: Return brief without hints
2. **Template-Based Hints**: Use pre-written hint templates

**Implementation**:

```python
class LlamaFailureHandler:
    def handle_failure(self, brief: CreativeBrief) -> List[str]:
        logger.info("Llama unavailable, using template hints")

        # Return template-based hints based on style
        return self.template_hints(brief.style)

    def template_hints(self, style: str) -> List[str]:
        """Template-based hints by style"""
        templates = {
            "fresh": [
                "Crisp morning air with a hint of citrus",
                "Ocean breeze on a summer day",
                "Dewdrops on fresh green leaves"
            ],
            "floral": [
                "A garden in full bloom at dawn",
                "Petals dancing in the spring breeze",
                "The elegance of white flowers under moonlight"
            ],
            "woody": [
                "Ancient forest after rainfall",
                "Warm cedar and sandalwood embrace",
                "The comfort of a fireside evening"
            ],
            "oriental": [
                "Spices and silk along the ancient trade routes",
                "Golden amber warmed by desert sun",
                "Mystical incense in a temple at dusk"
            ]
        }
        return templates.get(style, templates["fresh"])
```

### Scenario 4: All Models Down

**Impact**: 🔴🔴 Critical - Complete service failure

**Detection**:
```python
if all_models_down():
    logger.critical("All LLM models unavailable")
    # Enter emergency mode
```

**Automatic Recovery Strategy**:

1. **Emergency Read-Only Mode**: Serve from cache only
2. **Static Fallback**: Return pre-computed popular briefs
3. **Reject New Requests**: Return 503 Service Unavailable

**Implementation**:

```python
class EmergencyMode:
    def __init__(self):
        self.active = False
        self.cache_only = True

    def activate(self):
        self.active = True
        logger.critical("Emergency mode activated - cache only")
        alert("All LLM models down - emergency mode", severity="critical")

    def handle_request(self, user_text: str, mode: str) -> CreativeBrief:
        if not self.active:
            raise RuntimeError("Emergency mode not active")

        # Try cache first
        cached = cache.get(user_text)
        if cached:
            return cached

        # Try popular briefs
        popular = self.get_popular_brief(mode)
        if popular:
            popular.metadata["warning"] = "Using popular brief template (models unavailable)"
            return popular

        # Reject request
        raise ServiceUnavailableError(
            "Service temporarily unavailable. Please try again later.",
            retry_after=300  # 5 minutes
        )

    def get_popular_brief(self, mode: str) -> Optional[CreativeBrief]:
        """Return most popular pre-computed brief"""
        popular_briefs = {
            "fast": CreativeBrief(style="fresh", intensity=0.7, ...),
            "balanced": CreativeBrief(style="floral", intensity=0.6, ...),
            "creative": CreativeBrief(style="oriental", intensity=0.8, ...)
        }
        return popular_briefs.get(mode)
```

---

## Automatic Downshift Strategies

### Mode Downshift Decision Tree

```
Request (mode: creative)
    │
    ├─ All models healthy? → Creative Mode (Qwen + Mistral + Llama)
    │
    ├─ Llama down? → Balanced Mode (Qwen + Mistral)
    │                ↓
    │                └─ Mistral down? → Fast Mode (Qwen only)
    │                                   ↓
    │                                   └─ Qwen down? → Cache/Fallback
    │
    └─ All down? → Emergency Mode (Cache only / Reject)
```

### Implementation

```python
class ModeDownshiftManager:
    def __init__(self):
        self.model_health = {
            "qwen": True,
            "mistral": True,
            "llama": True
        }
        self.downshift_count = 0

    def select_mode(self, requested_mode: str, user_text: str) -> Tuple[str, bool]:
        """
        Select actual execution mode based on model health

        Returns:
            (selected_mode, is_downshifted)
        """
        # Check model health
        self.update_model_health()

        # Fast mode requested - only needs Qwen
        if requested_mode == "fast":
            if self.model_health["qwen"]:
                return "fast", False
            else:
                # Qwen down - cache/fallback
                return "fallback", True

        # Balanced mode requested - needs Qwen + Mistral
        if requested_mode == "balanced":
            if self.model_health["qwen"] and self.model_health["mistral"]:
                return "balanced", False
            elif self.model_health["qwen"]:
                # Mistral down - downshift to fast
                logger.warning("Balanced→Fast downshift (Mistral unavailable)")
                self.downshift_count += 1
                return "fast", True
            else:
                # Qwen down - fallback
                return "fallback", True

        # Creative mode requested - needs all 3 models
        if requested_mode == "creative":
            if all(self.model_health.values()):
                return "creative", False
            elif self.model_health["qwen"] and self.model_health["mistral"]:
                # Llama down - downshift to balanced
                logger.warning("Creative→Balanced downshift (Llama unavailable)")
                self.downshift_count += 1
                return "balanced", True
            elif self.model_health["qwen"]:
                # Mistral/Llama down - downshift to fast
                logger.warning("Creative→Fast downshift (Mistral/Llama unavailable)")
                self.downshift_count += 1
                return "fast", True
            else:
                # Qwen down - fallback
                return "fallback", True

        return "fallback", True

    def update_model_health(self):
        """Check health of all models"""
        for model_name in ["qwen", "mistral", "llama"]:
            try:
                health = self.check_model_health(model_name)
                self.model_health[model_name] = health
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {e}")
                self.model_health[model_name] = False

    def check_model_health(self, model_name: str) -> bool:
        """Ping model with simple request"""
        try:
            model = get_model(model_name)
            response = model.generate("test", max_tokens=10, timeout=5)
            return response is not None
        except Exception:
            return False
```

### Downshift Notification

```python
def notify_downshift(requested_mode: str, actual_mode: str, reason: str):
    """Notify user of mode downshift"""
    message = {
        "warning": f"Downshifted from {requested_mode} to {actual_mode}",
        "reason": reason,
        "impact": get_mode_comparison(requested_mode, actual_mode)
    }
    return message

def get_mode_comparison(from_mode: str, to_mode: str) -> dict:
    """Compare modes"""
    impact = {
        "creative→balanced": {
            "latency": "improved (faster)",
            "quality": "slightly reduced",
            "features_lost": ["creative_hints"]
        },
        "creative→fast": {
            "latency": "significantly improved",
            "quality": "moderately reduced",
            "features_lost": ["creative_hints", "validation"]
        },
        "balanced→fast": {
            "latency": "improved",
            "quality": "slightly reduced",
            "features_lost": ["validation"]
        }
    }
    return impact.get(f"{from_mode}→{to_mode}", {})
```

---

## Cache Management

### Cache Failure Recovery

**Scenario**: Redis cache unavailable

**Impact**: 🟡 Medium - Increased latency, higher compute load

**Recovery Strategy**:

```python
class CacheFailureHandler:
    def __init__(self):
        self.fallback_cache = {}  # In-memory LRU cache
        self.cache_down = False

    def get(self, key: str) -> Optional[Any]:
        """Get from cache with fallback"""
        try:
            # Try Redis first
            value = redis_client.get(key)
            self.cache_down = False
            return value
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self.cache_down = True

            # Fallback to in-memory cache
            return self.fallback_cache.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set in cache with fallback"""
        try:
            redis_client.setex(key, ttl, value)
            self.cache_down = False
        except Exception as e:
            logger.warning(f"Redis unavailable: {e}")
            self.cache_down = True

            # Fallback to in-memory cache
            self.fallback_cache[key] = value

            # Limit in-memory cache size
            if len(self.fallback_cache) > 100:
                # Remove oldest entries
                self.fallback_cache.pop(list(self.fallback_cache.keys())[0])
```

### Cache TTL Adjustment

**Scenario**: High load, need to reduce compute

**Strategy**: Increase cache TTL to serve more from cache

```python
class AdaptiveCacheTTL:
    def __init__(self):
        self.base_ttl = 3600  # 1 hour
        self.current_ttl = self.base_ttl
        self.max_ttl = 86400  # 24 hours
        self.min_ttl = 600    # 10 minutes

    def adjust_ttl(self, load_factor: float):
        """
        Adjust cache TTL based on system load

        Args:
            load_factor: 0.0-1.0 (0=idle, 1=overloaded)
        """
        if load_factor > 0.8:
            # High load - increase TTL to serve more from cache
            self.current_ttl = min(self.current_ttl * 2, self.max_ttl)
            logger.info(f"Cache TTL increased to {self.current_ttl}s (load: {load_factor:.2f})")

        elif load_factor < 0.3:
            # Low load - decrease TTL to serve fresher results
            self.current_ttl = max(self.current_ttl // 2, self.min_ttl)
            logger.info(f"Cache TTL decreased to {self.current_ttl}s (load: {load_factor:.2f})")

        return self.current_ttl

    def get_load_factor(self) -> float:
        """Calculate current system load"""
        queue_size = redis_client.llen("llm_inference_queue")
        max_queue_size = 100

        # Load factor based on queue depth
        load_factor = min(queue_size / max_queue_size, 1.0)
        return load_factor
```

**Usage**:

```python
# Adjust TTL every 60 seconds
cache_manager = AdaptiveCacheTTL()

@schedule.every(60).seconds
def adjust_cache_ttl():
    load = cache_manager.get_load_factor()
    ttl = cache_manager.adjust_ttl(load)

    # Log change
    if ttl != cache_manager.base_ttl:
        logger.info(f"Cache TTL adjusted: {ttl}s (load: {load:.2f})")
```

---

## Queue Management

### Queue Overflow

**Scenario**: Worker queue full (> 500 tasks)

**Impact**: 🟡 Medium - New requests rejected

**Recovery Strategy**:

```python
class QueueOverflowHandler:
    def __init__(self):
        self.max_queue_size = 500
        self.reject_count = 0

    def enqueue_task(self, task: dict) -> bool:
        """
        Enqueue task with overflow handling

        Returns:
            True if enqueued, False if rejected
        """
        queue_size = redis_client.llen("llm_inference_queue")

        if queue_size >= self.max_queue_size:
            self.reject_count += 1
            logger.warning(f"Queue overflow - rejecting task (queue size: {queue_size})")

            # Alert if rejection rate high
            if self.reject_count % 10 == 0:
                alert(f"Queue overflow - {self.reject_count} tasks rejected", severity="high")

            return False

        # Enqueue task
        redis_client.lpush("llm_inference_queue", json.dumps(task))
        return True

    def handle_rejection(self) -> dict:
        """Return response for rejected request"""
        return {
            "error": "Service busy - queue full",
            "retry_after": 30,  # Suggest retry after 30 seconds
            "queue_size": redis_client.llen("llm_inference_queue"),
            "suggestion": "Try again later or use fast mode"
        }
```

### Worker Auto-Scaling

**Scenario**: Queue growing, need more workers

**Recovery Strategy**:

```bash
# Monitor queue size
QUEUE_SIZE=$(redis-cli llen llm_inference_queue)

if [ $QUEUE_SIZE -gt 100 ]; then
    echo "Queue size: $QUEUE_SIZE - scaling workers"

    # Scale LLM workers
    docker-compose up -d --scale worker-llm=4

    # Alert
    echo "Scaled LLM workers to 4 instances"
fi
```

---

## Database Failures

### Database Unavailable

**Scenario**: PostgreSQL connection lost

**Impact**: 🔴 High - Cannot persist results

**Recovery Strategy**:

```python
class DatabaseFailureHandler:
    def __init__(self):
        self.db_down = False
        self.pending_writes = []

    def write_with_fallback(self, data: dict):
        """Write to database with fallback"""
        try:
            db.session.add(data)
            db.session.commit()
            self.db_down = False

            # Flush pending writes if DB recovered
            if self.pending_writes:
                self.flush_pending_writes()

        except Exception as e:
            logger.error(f"Database write failed: {e}")
            self.db_down = True

            # Queue write for later
            self.pending_writes.append(data)

            # Alert if too many pending writes
            if len(self.pending_writes) > 100:
                alert(f"Database down - {len(self.pending_writes)} pending writes", severity="critical")

    def flush_pending_writes(self):
        """Flush pending writes when DB recovers"""
        logger.info(f"Flushing {len(self.pending_writes)} pending writes")

        for data in self.pending_writes:
            try:
                db.session.add(data)
                db.session.commit()
            except Exception as e:
                logger.error(f"Pending write failed: {e}")

        self.pending_writes.clear()
```

---

## Monitoring & Alerting

### Health Check Endpoints

```python
@app.get("/health")
def health_check():
    """Overall system health"""
    return {
        "status": "healthy",
        "models": {
            "qwen": check_model_health("qwen"),
            "mistral": check_model_health("mistral"),
            "llama": check_model_health("llama")
        },
        "cache": check_redis_health(),
        "database": check_db_health(),
        "queue_size": redis_client.llen("llm_inference_queue"),
        "workers": {
            "llm": count_active_workers("llm"),
            "rl": count_active_workers("rl")
        }
    }

@app.get("/health/models/{model_name}")
def model_health(model_name: str):
    """Individual model health"""
    try:
        model = get_model(model_name)
        response = model.generate("test", max_tokens=10, timeout=5)
        latency = response.latency

        return {
            "status": "healthy",
            "latency_ms": latency * 1000,
            "last_check": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }
```

### Alert Rules

```yaml
# Prometheus alert rules
groups:
  - name: fragrance_ai_alerts
    interval: 30s
    rules:
      # Model failure
      - alert: ModelDown
        expr: model_health_status == 0
        for: 1m
        labels:
          severity: high
        annotations:
          summary: "Model {{ $labels.model }} is down"

      # High queue depth
      - alert: QueueOverload
        expr: queue_size > 100
        for: 5m
        labels:
          severity: medium
        annotations:
          summary: "Queue size {{ $value }} - consider scaling workers"

      # High error rate
      - alert: HighErrorRate
        expr: rate(http_errors_total[5m]) > 0.1
        for: 5m
        labels:
          severity: high
        annotations:
          summary: "Error rate {{ $value }} over 10%"

      # Cache miss rate
      - alert: HighCacheMissRate
        expr: rate(cache_misses[5m]) / rate(cache_requests[5m]) > 0.8
        for: 10m
        labels:
          severity: low
        annotations:
          summary: "Cache miss rate {{ $value }} - consider increasing TTL"
```

---

## Recovery Procedures

### Manual Recovery Steps

#### 1. Restart Failed Worker

```bash
# Check worker status
docker-compose ps

# Restart LLM worker
docker-compose restart worker-llm

# Check logs
docker-compose logs -f worker-llm
```

#### 2. Clear Stuck Queue

```bash
# Check queue size
redis-cli llen llm_inference_queue

# Clear queue (CAREFUL!)
redis-cli del llm_inference_queue

# Or move to backup
redis-cli rename llm_inference_queue llm_inference_queue_backup
```

#### 3. Rebuild Model Cache

```bash
# Clear model cache
rm -rf /app/cache/*

# Restart worker (will re-download models)
docker-compose restart worker-llm
```

#### 4. Database Recovery

```bash
# Check database connection
docker exec fragrance-ai-app python -c "
from fragrance_ai.config_loader import get_database_url
from sqlalchemy import create_engine
engine = create_engine(get_database_url())
conn = engine.connect()
print('✓ Database connected')
"

# Restore from backup
pg_restore -d fragrance_prod backup.dump
```

---

## Summary

**Key Takeaways**:

1. **Graceful Degradation**: Automatic downshift (creative→balanced→fast→fallback)
2. **Cache as Safety Net**: Serve from cache when models unavailable
3. **Fast Failure Detection**: Health checks every 10-30 seconds
4. **Auto-Recovery**: Retry with reduced context, fallback strategies
5. **Monitoring**: Prometheus alerts for all failure scenarios
6. **Manual Procedures**: Clear recovery steps for operators

**Failure Priority**:
- 🔴 Critical: Qwen down, all models down, database down
- 🟡 Medium: Mistral down, queue overflow, cache miss
- 🟢 Low: Llama down (creative mode only)

For more information:
- Prompt optimization: `docs/PROMPT_DESIGN_GUIDE.md`
- Parameter tuning: `docs/TUNING_GUIDE.md`
- Deployment: `docs/DEPLOYMENT.md`
