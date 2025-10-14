# Mixed Traffic Tuning Guide
# 혼합 트래픽 튜닝 가이드 (초기 48시간)

Recommended settings for the first 48 hours after production deployment.

---

## ✅ Overview

프로덕션 배포 후 초기 48시간 동안의 권장 설정:

1. **트래픽 비율**: fast 60% / balanced 30% / creative 10%
2. **PPO 스케줄**: entropy_coef 선형 감소 (0.02 → 0.005, 1만 step)
3. **Reward Normalization**: 최근 1k step 윈도우 유지
4. **Checkpoint 전략**: 500 step마다 저장, 급등/급락 시 즉시 롤백

---

## 1. Traffic Distribution (트래픽 비율)

### Rationale

- **fast 60%**: 빠른 응답으로 대부분의 사용자 만족 + 시스템 부하 최소화
- **balanced 30%**: 품질과 속도 균형으로 중요 요청 처리
- **creative 10%**: 복잡한 요청 및 실험, 3-모델 앙상블 성능 검증

### Configuration

**파일**: `configs/traffic_distribution.yaml`

```yaml
# Initial 48-hour traffic distribution
traffic_distribution:
  # Mode percentages (must sum to 100)
  modes:
    fast: 60        # 60%
    balanced: 30    # 30%
    creative: 10    # 10%

  # Routing strategy
  routing:
    strategy: weighted_random  # weighted_random | round_robin | user_preference
    sticky_session: true        # Same user gets same mode (cache for 1 hour)
    fallback_mode: fast         # Fallback if preferred mode fails

  # Load balancing
  load_balancing:
    algorithm: least_connections
    health_check_interval: 30s
    max_queue_size: 100

  # Monitoring
  monitoring:
    report_interval: 60s        # Report distribution every 60s
    alert_deviation: 0.1        # Alert if deviation > 10% from target
```

### Implementation

**파일**: `fragrance_ai/routing/traffic_distributor.py`

```python
class TrafficDistributor:
    """Distribute traffic across modes based on configured ratios"""

    def __init__(self, config: dict):
        self.mode_weights = config["modes"]  # {fast: 60, balanced: 30, creative: 10}
        self.sticky_session = config["routing"]["sticky_session"]
        self.fallback_mode = config["routing"]["fallback_mode"]

        # Session cache for sticky sessions
        self.session_cache = {}

    def select_mode(self, user_id: Optional[str] = None) -> str:
        """Select mode based on traffic distribution"""

        # Check sticky session
        if self.sticky_session and user_id:
            cached_mode = self.session_cache.get(user_id)
            if cached_mode:
                return cached_mode

        # Weighted random selection
        modes = list(self.mode_weights.keys())
        weights = list(self.mode_weights.values())

        selected_mode = random.choices(modes, weights=weights)[0]

        # Cache for sticky session
        if self.sticky_session and user_id:
            self.session_cache[user_id] = selected_mode

        return selected_mode

    def get_current_distribution(self) -> dict:
        """Get actual traffic distribution from metrics"""
        from fragrance_ai.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        # Get request counts by mode
        mode_counts = collector.get_request_counts_by_mode(window="5m")

        total = sum(mode_counts.values())
        if total == 0:
            return self.mode_weights  # Return target if no traffic yet

        # Calculate percentages
        actual_distribution = {
            mode: int((count / total) * 100)
            for mode, count in mode_counts.items()
        }

        return actual_distribution

    def check_distribution_deviation(self) -> dict:
        """Check if actual distribution deviates from target"""
        actual = self.get_current_distribution()
        target = self.mode_weights

        deviations = {}
        for mode in target.keys():
            actual_pct = actual.get(mode, 0)
            target_pct = target[mode]
            deviation = abs(actual_pct - target_pct)
            deviations[mode] = {
                "actual": actual_pct,
                "target": target_pct,
                "deviation": deviation,
                "alert": deviation > 10  # Alert if > 10% deviation
            }

        return deviations
```

### Verification

```bash
# Check traffic distribution
python -c "
from fragrance_ai.routing.traffic_distributor import TrafficDistributor
from fragrance_ai.config_loader import get_config

config = get_config()
distributor = TrafficDistributor(config.traffic_distribution)

print('Traffic Distribution:')
for mode, weight in distributor.mode_weights.items():
    print(f'  {mode}: {weight}%')

# Check actual distribution
actual = distributor.get_current_distribution()
print('\nActual Distribution (last 5 min):')
for mode, pct in actual.items():
    print(f'  {mode}: {pct}%')

# Check deviations
deviations = distributor.check_distribution_deviation()
print('\nDeviations:')
for mode, data in deviations.items():
    alert = '⚠ ALERT' if data['alert'] else '✓ OK'
    print(f'  {mode}: {data[\"actual\"]}% (target: {data[\"target\"]}%) {alert}')
"
```

### Monitoring

**Prometheus Metrics**:
```promql
# Actual traffic distribution
rate(http_requests_total{mode="fast"}[5m]) / rate(http_requests_total[5m]) * 100
rate(http_requests_total{mode="balanced"}[5m]) / rate(http_requests_total[5m]) * 100
rate(http_requests_total{mode="creative"}[5m]) / rate(http_requests_total[5m]) * 100

# Deviation from target
abs(
  rate(http_requests_total{mode="fast"}[5m]) / rate(http_requests_total[5m]) * 100
  - 60
)
```

**Grafana Dashboard**:
```json
{
  "title": "Traffic Distribution",
  "panels": [
    {
      "title": "Target vs Actual",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(http_requests_total{mode=\"fast\"}[5m]) / rate(http_requests_total[5m]) * 100",
          "legendFormat": "fast (actual)"
        },
        {
          "expr": "60",
          "legendFormat": "fast (target)"
        }
      ]
    }
  ]
}
```

---

## 2. PPO Schedule (엔트로피 선형 감소)

### Rationale

엔트로피 계수를 점진적으로 감소시켜:
- **초기 (0.02)**: 높은 탐색으로 다양한 조합 시도
- **후기 (0.005)**: 낮은 탐색으로 최적 조합에 수렴

### Configuration

**파일**: `configs/ppo_schedule.yaml`

```yaml
# PPO training schedule for first 48 hours
ppo_schedule:
  # Entropy coefficient linear decay
  entropy_coef:
    initial: 0.02       # High exploration
    final: 0.005        # Low exploration
    decay_steps: 10000  # 10k training steps
    decay_strategy: linear

  # Learning rate schedule
  learning_rate:
    initial: 0.0003
    final: 0.0001
    decay_steps: 10000
    decay_strategy: cosine

  # Other hyperparameters
  clip_range: 0.2       # PPO clip range
  gamma: 0.99           # Discount factor
  lambda_: 0.95         # GAE lambda
  batch_size: 64        # Training batch size
  n_epochs: 4           # Epochs per update

  # Training frequency
  update_frequency: 500  # Update every 500 steps
  save_frequency: 500    # Save checkpoint every 500 steps
```

### Implementation

**파일**: `fragrance_ai/training/ppo_scheduler.py`

```python
class PPOScheduler:
    """Linear scheduler for PPO hyperparameters"""

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        total_steps: int,
        strategy: str = "linear"
    ):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.strategy = strategy
        self.current_step = 0

    def get_value(self, step: Optional[int] = None) -> float:
        """Get scheduled value for current step"""
        if step is None:
            step = self.current_step

        # Clamp to total_steps
        progress = min(step / self.total_steps, 1.0)

        if self.strategy == "linear":
            value = self.initial_value + (self.final_value - self.initial_value) * progress
        elif self.strategy == "cosine":
            value = self.final_value + 0.5 * (self.initial_value - self.final_value) * (
                1 + np.cos(np.pi * progress)
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return value

    def step(self):
        """Increment step counter"""
        self.current_step += 1


class PPOTrainerWithSchedule:
    """PPO trainer with entropy coefficient scheduling"""

    def __init__(self, config: dict):
        self.config = config

        # Initialize schedulers
        self.entropy_scheduler = PPOScheduler(
            initial_value=config["entropy_coef"]["initial"],
            final_value=config["entropy_coef"]["final"],
            total_steps=config["entropy_coef"]["decay_steps"],
            strategy=config["entropy_coef"]["decay_strategy"]
        )

        self.lr_scheduler = PPOScheduler(
            initial_value=config["learning_rate"]["initial"],
            final_value=config["learning_rate"]["final"],
            total_steps=config["learning_rate"]["decay_steps"],
            strategy=config["learning_rate"]["decay_strategy"]
        )

        self.training_step = 0

    def train_step(self, batch: dict) -> dict:
        """Single training step with scheduled hyperparameters"""

        # Get current hyperparameters
        entropy_coef = self.entropy_scheduler.get_value()
        learning_rate = self.lr_scheduler.get_value()

        # Update PPO agent
        loss_info = self.ppo_agent.update(
            batch=batch,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate
        )

        # Log scheduled values
        loss_info.update({
            "entropy_coef": entropy_coef,
            "learning_rate": learning_rate,
            "training_step": self.training_step
        })

        # Increment schedulers
        self.entropy_scheduler.step()
        self.lr_scheduler.step()
        self.training_step += 1

        return loss_info
```

### Verification

```bash
# Visualize entropy schedule
python -c "
import numpy as np
import matplotlib.pyplot as plt
from fragrance_ai.training.ppo_scheduler import PPOScheduler

scheduler = PPOScheduler(
    initial_value=0.02,
    final_value=0.005,
    total_steps=10000,
    strategy='linear'
)

steps = np.arange(0, 10000, 100)
values = [scheduler.get_value(step) for step in steps]

plt.figure(figsize=(10, 6))
plt.plot(steps, values)
plt.xlabel('Training Steps')
plt.ylabel('Entropy Coefficient')
plt.title('PPO Entropy Schedule (Linear Decay)')
plt.grid(True)
plt.savefig('ppo_schedule.png')
print('✓ Schedule visualization saved to ppo_schedule.png')

# Print key milestones
milestones = [0, 2500, 5000, 7500, 10000]
print('\nEntropy Coefficient Milestones:')
for step in milestones:
    value = scheduler.get_value(step)
    print(f'  Step {step:5d}: {value:.4f}')
"
```

### Monitoring

**Prometheus Metrics**:
```promql
# Entropy coefficient over time
ppo_entropy_coef

# Learning rate over time
ppo_learning_rate

# Training step
ppo_training_step
```

**Alert Rules**:
```yaml
# Alert if entropy not decreasing
- alert: PPOEntropyNotDecreasing
  expr: ppo_entropy_coef offset 1h >= ppo_entropy_coef
  for: 30m
  labels:
    severity: warning
  annotations:
    summary: "PPO entropy coefficient not decreasing"
```

---

## 3. Reward Normalization (최근 1k step 윈도우)

### Rationale

보상 정규화로 학습 안정성 확보:
- 최근 1000 step의 평균/표준편차로 정규화
- 보상 스케일 변화에 robust
- 학습 속도 향상

### Configuration

**파일**: `configs/reward_normalization.yaml`

```yaml
# Reward normalization settings
reward_normalization:
  enabled: true
  window_size: 1000    # Recent 1000 steps
  epsilon: 1e-8        # Numerical stability
  clip_range: 10.0     # Clip normalized rewards to [-10, 10]
  update_frequency: 1  # Update stats every step

  # Bootstrap with initial data
  bootstrap_size: 100  # Minimum samples before normalization
  bootstrap_mean: 3.0  # Initial mean estimate
  bootstrap_std: 1.0   # Initial std estimate
```

### Implementation

**파일**: `fragrance_ai/training/reward_normalizer.py`

```python
class RewardNormalizer:
    """Running reward normalizer with fixed window"""

    def __init__(
        self,
        window_size: int = 1000,
        epsilon: float = 1e-8,
        clip_range: float = 10.0
    ):
        self.window_size = window_size
        self.epsilon = epsilon
        self.clip_range = clip_range

        # Rolling window of rewards
        self.reward_buffer = deque(maxlen=window_size)

        # Running statistics
        self.mean = 0.0
        self.std = 1.0
        self.count = 0

    def update(self, reward: float):
        """Add new reward and update statistics"""
        self.reward_buffer.append(reward)
        self.count += 1

        # Recompute statistics from buffer
        if len(self.reward_buffer) >= 10:  # Minimum samples
            self.mean = np.mean(self.reward_buffer)
            self.std = np.std(self.reward_buffer) + self.epsilon

    def normalize(self, reward: float) -> float:
        """Normalize reward using current statistics"""
        if self.count < 10:
            return reward  # Don't normalize until enough samples

        # Standardize
        normalized = (reward - self.mean) / self.std

        # Clip
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

    def normalize_batch(self, rewards: np.ndarray) -> np.ndarray:
        """Normalize batch of rewards"""
        if self.count < 10:
            return rewards

        # Standardize
        normalized = (rewards - self.mean) / self.std

        # Clip
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized

    def get_stats(self) -> dict:
        """Get current normalization statistics"""
        return {
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "window_size": len(self.reward_buffer)
        }
```

### Verification

```bash
# Test reward normalization
python -c "
import numpy as np
from fragrance_ai.training.reward_normalizer import RewardNormalizer

normalizer = RewardNormalizer(window_size=1000)

# Simulate rewards with drift
print('Simulating 2000 steps with reward drift...')
for step in range(2000):
    # Reward drifts from 3.0 to 5.0
    base_reward = 3.0 + (2.0 * step / 2000)
    reward = base_reward + np.random.normal(0, 0.5)

    normalizer.update(reward)

    if step % 500 == 0:
        stats = normalizer.get_stats()
        print(f'  Step {step:4d}: mean={stats[\"mean\"]:.3f}, std={stats[\"std\"]:.3f}')

# Test normalization
print('\nNormalization examples:')
test_rewards = [2.0, 3.5, 5.0, 6.5]
for reward in test_rewards:
    normalized = normalizer.normalize(reward)
    print(f'  Reward {reward:.1f} → Normalized {normalized:.3f}')

print('\n✓ Reward normalization test passed')
"
```

### Monitoring

**Prometheus Metrics**:
```promql
# Reward statistics
reward_normalizer_mean
reward_normalizer_std

# Normalized rewards
histogram_quantile(0.5, rate(reward_normalized_bucket[5m]))  # Median
histogram_quantile(0.95, rate(reward_normalized_bucket[5m]))  # 95th percentile

# Raw rewards (for comparison)
histogram_quantile(0.5, rate(reward_raw_bucket[5m]))
```

---

## 4. Checkpoint Strategy (500 step + 급등/급락 롤백)

### Rationale

자주 체크포인트를 저장하여:
- 모델 손실 시 복구 가능
- 성능 급락 시 즉시 롤백
- 최적 모델 선택 가능

### Configuration

**파일**: `configs/checkpoint_strategy.yaml`

```yaml
# Checkpoint saving and rollback strategy
checkpoint_strategy:
  # Save frequency
  save_frequency: 500  # Save every 500 steps
  max_checkpoints: 20  # Keep last 20 checkpoints

  # Auto-rollback conditions
  rollback:
    enabled: true

    # Loss spike detection
    loss_spike:
      threshold: 2.0     # 2x baseline loss
      window: 100        # Check last 100 steps
      baseline_window: 1000  # Baseline from last 1000 steps

    # Reward drop detection
    reward_drop:
      threshold: 0.5     # 50% drop from baseline
      window: 100
      baseline_window: 1000

    # Auto-rollback strategy
    rollback_steps: 1    # Rollback to 1 checkpoint ago (500 steps)
    max_rollbacks: 3     # Max 3 rollbacks per session

  # Best model tracking
  best_model:
    metric: reward       # Track by reward | loss
    save_best: true      # Save best model separately
```

### Implementation

**파일**: `fragrance_ai/training/checkpoint_manager.py`

```python
class CheckpointManager:
    """Manage checkpoints with auto-rollback on performance degradation"""

    def __init__(self, config: dict, checkpoint_dir: str = "checkpoints"):
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Checkpoint history
        self.checkpoints = []  # List of (step, path, metrics)
        self.max_checkpoints = config["max_checkpoints"]

        # Rollback tracking
        self.rollback_count = 0
        self.max_rollbacks = config["rollback"]["max_rollbacks"]

        # Performance history
        self.loss_history = deque(maxlen=config["rollback"]["loss_spike"]["baseline_window"])
        self.reward_history = deque(maxlen=config["rollback"]["reward_drop"]["baseline_window"])

        # Best model
        self.best_reward = -float('inf')
        self.best_checkpoint = None

    def save_checkpoint(self, step: int, model, metrics: dict):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"

        torch.save({
            "step": step,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": time.time()
        }, checkpoint_path)

        # Add to history
        self.checkpoints.append((step, checkpoint_path, metrics))

        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_step, old_path, _ = self.checkpoints.pop(0)
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")

        # Check if best model
        if metrics.get("reward", -float('inf')) > self.best_reward:
            self.best_reward = metrics["reward"]
            self.best_checkpoint = checkpoint_path
            self._save_best_model(checkpoint_path)

        logger.info(f"Saved checkpoint at step {step}")

    def check_and_rollback(self, current_metrics: dict) -> bool:
        """Check if rollback is needed"""
        if not self.config["rollback"]["enabled"]:
            return False

        # Check rollback limit
        if self.rollback_count >= self.max_rollbacks:
            logger.warning(f"Max rollbacks reached ({self.max_rollbacks})")
            return False

        # Update history
        self.loss_history.append(current_metrics["loss"])
        self.reward_history.append(current_metrics["reward"])

        # Check loss spike
        if self._detect_loss_spike():
            logger.warning("Loss spike detected - rolling back")
            self._perform_rollback()
            return True

        # Check reward drop
        if self._detect_reward_drop():
            logger.warning("Reward drop detected - rolling back")
            self._perform_rollback()
            return True

        return False

    def _detect_loss_spike(self) -> bool:
        """Detect if loss has spiked"""
        config = self.config["rollback"]["loss_spike"]

        if len(self.loss_history) < config["baseline_window"]:
            return False

        # Baseline: mean of all history
        baseline_loss = np.mean(list(self.loss_history)[:-config["window"]])

        # Recent: mean of last window
        recent_loss = np.mean(list(self.loss_history)[-config["window"]:])

        # Check spike
        spike = recent_loss > baseline_loss * config["threshold"]

        if spike:
            logger.warning(
                f"Loss spike: {recent_loss:.4f} > {baseline_loss:.4f} * {config['threshold']}"
            )

        return spike

    def _detect_reward_drop(self) -> bool:
        """Detect if reward has dropped"""
        config = self.config["rollback"]["reward_drop"]

        if len(self.reward_history) < config["baseline_window"]:
            return False

        # Baseline: mean of all history
        baseline_reward = np.mean(list(self.reward_history)[:-config["window"]])

        # Recent: mean of last window
        recent_reward = np.mean(list(self.reward_history)[-config["window"]:])

        # Check drop
        drop = recent_reward < baseline_reward * (1 - config["threshold"])

        if drop:
            logger.warning(
                f"Reward drop: {recent_reward:.4f} < {baseline_reward:.4f} * (1 - {config['threshold']})"
            )

        return drop

    def _perform_rollback(self):
        """Rollback to previous checkpoint"""
        rollback_steps = self.config["rollback"]["rollback_steps"]

        if len(self.checkpoints) <= rollback_steps:
            logger.error("Not enough checkpoints to rollback")
            return

        # Get rollback checkpoint
        rollback_step, rollback_path, rollback_metrics = self.checkpoints[-(rollback_steps + 1)]

        # Load checkpoint
        checkpoint = torch.load(rollback_path)

        logger.info(f"Rolling back to step {rollback_step}")
        logger.info(f"Rollback metrics: loss={rollback_metrics['loss']:.4f}, reward={rollback_metrics['reward']:.4f}")

        # Increment rollback count
        self.rollback_count += 1

        return checkpoint

    def _save_best_model(self, checkpoint_path: Path):
        """Save best model separately"""
        best_path = self.checkpoint_dir / "best_model.pt"
        shutil.copy(checkpoint_path, best_path)
        logger.info(f"Saved new best model: reward={self.best_reward:.4f}")
```

### Verification

```bash
# Test checkpoint and rollback
python scripts/test_checkpoint_rollback.py
```

**파일**: `scripts/test_checkpoint_rollback.py`

```python
#!/usr/bin/env python3
"""Test checkpoint rollback on loss spike"""

import numpy as np
from fragrance_ai.training.checkpoint_manager import CheckpointManager
from fragrance_ai.config_loader import get_config

def test_checkpoint_rollback():
    """Simulate training with loss spike and rollback"""

    config = get_config().checkpoint_strategy
    manager = CheckpointManager(config)

    print("Simulating training with loss spike...")
    print("=" * 60)

    # Normal training
    for step in range(0, 2000, 500):
        loss = 0.5 + np.random.normal(0, 0.1)
        reward = 4.0 + np.random.normal(0, 0.5)

        metrics = {"loss": loss, "reward": reward}

        manager.save_checkpoint(step, model=None, metrics=metrics)
        manager.loss_history.append(loss)
        manager.reward_history.append(reward)

        print(f"Step {step:4d}: loss={loss:.4f}, reward={reward:.4f} ✓")

    # Introduce loss spike at step 2500
    print("\nIntroducing loss spike...")
    spike_loss = 2.0  # 4x baseline
    spike_reward = 2.0  # Drop
    spike_metrics = {"loss": spike_loss, "reward": spike_reward}

    manager.save_checkpoint(2500, model=None, metrics=spike_metrics)

    # Check rollback
    should_rollback = manager.check_and_rollback(spike_metrics)

    if should_rollback:
        print(f"✓ Rollback triggered (loss spike: {spike_loss:.4f})")
        print(f"  Rolling back to step 2000")
    else:
        print(f"✗ Rollback NOT triggered (unexpected)")

    print("\n✓ Checkpoint rollback test completed")

if __name__ == '__main__':
    test_checkpoint_rollback()
```

### Monitoring

**Prometheus Metrics**:
```promql
# Checkpoint saves
rate(checkpoint_saves_total[1h])

# Rollbacks
rate(checkpoint_rollbacks_total[1h])

# Best model updates
changes(best_model_reward[1h])

# Loss/reward for rollback detection
ppo_loss
ppo_reward
```

**Alert Rules**:
```yaml
# Alert on rollback
- alert: CheckpointRollback
  expr: increase(checkpoint_rollbacks_total[5m]) > 0
  labels:
    severity: warning
  annotations:
    summary: "Checkpoint rollback occurred"
    description: "Training rolled back due to loss spike or reward drop"

# Alert on frequent rollbacks
- alert: FrequentRollbacks
  expr: rate(checkpoint_rollbacks_total[1h]) > 0.01
  for: 1h
  labels:
    severity: critical
  annotations:
    summary: "Frequent checkpoint rollbacks"
    description: "Multiple rollbacks in 1 hour - investigate training instability"
```

---

## 📋 Complete 48-Hour Monitoring Checklist

```bash
#!/bin/bash
# Monitor mixed traffic tuning for first 48 hours

echo "Monitoring Mixed Traffic Tuning..."
echo "======================================"

# 1. Traffic Distribution
echo "1. Traffic Distribution"
python -c "
from fragrance_ai.routing.traffic_distributor import TrafficDistributor
from fragrance_ai.config_loader import get_config

config = get_config()
distributor = TrafficDistributor(config.traffic_distribution)

actual = distributor.get_current_distribution()
target = distributor.mode_weights

print('  Target: fast=60%, balanced=30%, creative=10%')
print(f'  Actual: fast={actual.get(\"fast\", 0)}%, balanced={actual.get(\"balanced\", 0)}%, creative={actual.get(\"creative\", 0)}%')

deviations = distributor.check_distribution_deviation()
for mode, data in deviations.items():
    if data['alert']:
        print(f'  ⚠ {mode}: {data[\"deviation\"]}% deviation from target')
"

# 2. PPO Schedule
echo "2. PPO Entropy Schedule"
python -c "
from fragrance_ai.training.ppo_scheduler import PPOScheduler
from fragrance_ai.monitoring.metrics import MetricsCollector

collector = MetricsCollector()
current_entropy = collector.get_metric('ppo_entropy_coef')
current_step = collector.get_metric('ppo_training_step')

print(f'  Current step: {current_step}')
print(f'  Current entropy: {current_entropy:.4f}')
print(f'  Target: 0.02 → 0.005 (linear, 10k steps)')
"

# 3. Reward Normalization
echo "3. Reward Normalization"
python -c "
from fragrance_ai.training.reward_normalizer import RewardNormalizer

# Assuming normalizer is singleton
normalizer = RewardNormalizer.get_instance()
stats = normalizer.get_stats()

print(f'  Mean: {stats[\"mean\"]:.3f}')
print(f'  Std: {stats[\"std\"]:.3f}')
print(f'  Window: {stats[\"window_size\"]}/1000')
"

# 4. Checkpoint Status
echo "4. Checkpoint Status"
python -c "
from fragrance_ai.training.checkpoint_manager import CheckpointManager

manager = CheckpointManager.get_instance()

print(f'  Checkpoints saved: {len(manager.checkpoints)}')
print(f'  Best reward: {manager.best_reward:.4f}')
print(f'  Rollback count: {manager.rollback_count}/{manager.max_rollbacks}')
"

echo "======================================"
echo "✓ Monitoring complete"
```

---

## 📊 Expected Metrics After 48 Hours

### Traffic Distribution

- ✅ fast: 58-62% (target: 60%)
- ✅ balanced: 28-32% (target: 30%)
- ✅ creative: 8-12% (target: 10%)
- ✅ Deviation < 5% from target

### PPO Training

- ✅ ~10k training steps completed
- ✅ Entropy coefficient: ~0.005 (converged)
- ✅ Loss: Stable or decreasing
- ✅ Reward: Stable or increasing
- ✅ 0-1 rollbacks (ideally 0)

### System Stability

- ✅ Uptime: 100%
- ✅ Error rate: < 0.5%
- ✅ Latency: Within thresholds
- ✅ No circuit breakers open
- ✅ Cache hit rate: > 70%

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
**Maintained by**: ML Engineering Team
