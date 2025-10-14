# Model & Algorithm Operations Guide

모델/알고리즘 운영: 안정화 스케줄, 자동 롤백, A/B 테스트, 데이터 큐레이션

---

## Overview

This guide covers the complete operational infrastructure for RL model training and optimization in the Artisan system.

**Key Components:**
- ✅ **Entropy Annealing** - PPO entropy coefficient scheduling (exploration → exploitation)
- ✅ **Reward Normalization** - Rolling window statistics (last 1k steps)
- ✅ **Automatic Rollback** - Checkpoint management with anomaly detection
- ✅ **A/B Testing** - Creative vs Balanced mode traffic splitting
- ✅ **Data Curation** - Feedback quality filtering (outlier removal, spam blocking)
- ✅ **Config Management** - Externalized configs with tuning history tracking

---

## Table of Contents

1. [Stabilization Schedules](#stabilization-schedules)
2. [Checkpoint & Rollback](#checkpoint--rollback)
3. [A/B Testing](#ab-testing)
4. [Data Curation](#data-curation)
5. [Configuration Management](#configuration-management)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Troubleshooting](#troubleshooting)

---

## Stabilization Schedules

### Entropy Coefficient Annealing

**Purpose:** Gradually transition from exploration (high entropy) to exploitation (low entropy) during training.

**Implementation:** `fragrance_ai/training/rl/schedulers.py`

#### Configuration

```yaml
# configs/recommended_params.yaml
rl_advanced:
  entropy_scheduler:
    enabled: true
    schedule_type: "cosine"  # linear, cosine, exponential
    initial_entropy: 0.01    # Start with 1% entropy
    final_entropy: 0.001     # End with 0.1% entropy
    decay_steps: 10000       # Anneal over 10k steps
```

#### Usage

```python
from fragrance_ai.training.rl.schedulers import EntropyScheduler, EntropySchedulerConfig

# Initialize scheduler
config = EntropySchedulerConfig(
    initial_entropy=0.01,
    final_entropy=0.001,
    decay_steps=10000,
    schedule_type="cosine"
)

scheduler = EntropyScheduler(config)

# Use in training loop
for step in range(training_steps):
    entropy_coef = scheduler.get_entropy_coef(step)

    # Use entropy_coef in PPO update
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

#### Visualization

```
Entropy Coefficient (Cosine Schedule)
|
| 0.010 ●━━━━╮
|            ╰━━━╮
| 0.005        ╰━━━╮
|                 ╰━━━╮
| 0.001             ╰━━━━●
└────────────────────────────> steps
    0              10000
```

### Reward Normalization

**Purpose:** Normalize rewards using rolling statistics to stabilize training across different reward scales.

**Implementation:** `fragrance_ai/training/rl/schedulers.py`

#### Configuration

```yaml
# configs/recommended_params.yaml
reward_normalization:
  enabled: true
  window_size: 1000       # Last 1k steps
  update_mean_std: true
  epsilon: 1.0e-8
  clip_range: 10.0        # Clip to [-10, 10]
```

#### Usage

```python
from fragrance_ai.training.rl.schedulers import RewardNormalizer, RewardNormalizerConfig

# Initialize normalizer
config = RewardNormalizerConfig(
    window_size=1000,
    clip_range=10.0
)

normalizer = RewardNormalizer(config)

# Normalize rewards in training loop
raw_reward = 15.3
normalized_reward = normalizer.normalize(raw_reward, update=True)

# Get statistics
stats = normalizer.get_statistics()
print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
```

#### Benefits

- **Stable Training**: Prevents reward scale issues
- **Faster Convergence**: Consistent gradient magnitudes
- **Outlier Handling**: Clips extreme values

---

## Checkpoint & Rollback

### Automatic Checkpoint Management

**Purpose:** Periodically save model checkpoints and automatically rollback on training instability.

**Implementation:** `fragrance_ai/training/checkpoint_manager.py` (already exists)

#### Configuration

```yaml
# configs/recommended_params.yaml
rl_advanced:
  checkpoint_manager:
    enabled: true
    save_interval: 100       # Save every 100 steps
    max_checkpoints: 5       # Keep last 5 checkpoints

    # Rollback triggers
    rollback_on_kl_threshold: 0.03        # KL divergence > 0.03
    rollback_on_loss_increase: 2.0        # Loss increases 2x
    rollback_on_reward_drop: 0.3          # Reward drops 30%
```

#### Usage

```python
from fragrance_ai.training.checkpoint_manager import CheckpointManager, RollbackConditions

# Initialize manager
manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    save_interval=100,
    max_checkpoints=5,
    rollback_conditions=RollbackConditions(
        kl_threshold=0.03,
        loss_increase_multiplier=2.0,
        reward_drop_threshold=0.3
    )
)

# Training loop
for step in range(training_steps):
    # ... training ...

    # Save checkpoint periodically
    if manager.should_save(step):
        manager.save_checkpoint(
            step=step,
            policy_net=policy,
            value_net=value,
            optimizer=optimizer,
            loss=current_loss,
            reward=current_reward,
            kl_divergence=current_kl
        )

    # Check for rollback
    should_rollback, rollback_ckpt, reason = manager.check_rollback(
        current_step=step,
        current_loss=current_loss,
        current_reward=current_reward,
        current_kl_divergence=current_kl
    )

    if should_rollback:
        logger.warning(f"[ROLLBACK] {reason}")
        manager.rollback(policy, value, optimizer)
```

#### Rollback Triggers

| Condition | Threshold | Action |
|-----------|-----------|--------|
| **KL Divergence** | > 0.03 | Immediate rollback |
| **Loss Spike** | 2x increase | Rollback to last stable |
| **Reward Drop** | 30% decrease | Rollback to last stable |

---

## A/B Testing

### Traffic Splitting

**Purpose:** Compare performance of Creative (exploration) vs Balanced (baseline) modes.

**Implementation:** `fragrance_ai/training/ab_testing.py`

#### Configuration

```python
from fragrance_ai.training.ab_testing import ABTestManager, TrafficSplitConfig, Variant

# Configure traffic split
traffic_split = TrafficSplitConfig(
    control_ratio=0.5,      # 50% Balanced (control)
    treatment_a_ratio=0.4,  # 40% Creative (treatment A)
    treatment_b_ratio=0.1   # 10% Fast (treatment B)
)

# Initialize A/B test
ab_test = ABTestManager(
    experiment_id="creative_vs_balanced_2025Q1",
    traffic_split=traffic_split,
    results_dir="ab_test_results",
    min_sample_size=100
)
```

#### Usage

```python
# Assign user to variant
user_id = "user_12345"
variant = ab_test.assign_variant(user_id)  # Consistent hashing

# Handle request based on variant
if variant == Variant.CONTROL:
    mode = "balanced"
elif variant == Variant.TREATMENT_A:
    mode = "creative"
else:
    mode = "fast"

# ... generate formula ...

# Record metrics
ab_test.record_request(
    variant=variant,
    success=True,
    latency=3.2,
    user_rating=4.5,
    reward=15.3,
    ifra_compliant=True,
    entropy=0.008,
    kl_divergence=0.012,
    converted=True
)
```

#### Analysis

```python
# Check if ready for analysis
if ab_test.is_ready_for_analysis():
    result = ab_test.analyze_results()

    print(f"Winner: {result.winner}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Improvement: {result.improvement_percentage:.1f}%")
    print(f"Is Significant: {result.is_significant}")

# Print summary
ab_test.print_summary()
```

**Output Example:**

```
================================================================================
A/B Test Summary: creative_vs_balanced_2025Q1
================================================================================

[Control - Balanced Mode]
  Total Requests: 512
  Success Rate: 96.48%
  Avg Latency: 3.215s
  Avg User Rating: 4.12/5.0
  Avg Reward: 14.583
  IFRA Compliance: 98.24%

[Treatment A - Creative Mode]
  Total Requests: 408
  Success Rate: 94.85%
  Avg Latency: 4.387s
  Avg User Rating: 4.38/5.0
  Avg Reward: 16.721
  IFRA Compliance: 96.08%
  Avg Entropy: 0.0092
  Avg KL Divergence: 0.0145

[Statistical Significance]
  Is Significant: True
  P-value: 0.0234
  Winner: treatment_a
  Improvement: 6.3%
================================================================================
```

---

## Data Curation

### Feedback Quality Filtering

**Purpose:** Filter out low-quality feedback samples (outliers, spam, non-compliant).

**Implementation:** `fragrance_ai/training/data_curation.py`

#### Configuration

```python
from fragrance_ai.training.data_curation import DataCurator, QualityThresholds, FeedbackSample

# Configure quality thresholds
thresholds = QualityThresholds(
    min_reward=-10.0,
    max_reward=100.0,
    min_rating=0.0,
    max_rating=5.0,
    z_score_threshold=3.0,                 # 3 standard deviations
    max_duplicate_submissions=5,           # Max 5 submissions per user
    duplicate_window_minutes=60,           # in 60 minutes
    min_response_time_seconds=2.0,         # At least 2 seconds
    max_response_time_seconds=600.0,       # Max 10 minutes
    require_ifra_compliance=True
)

# Initialize curator
curator = DataCurator(thresholds=thresholds)
```

#### Usage

```python
# Create feedback sample
sample = FeedbackSample(
    sample_id="sample_001",
    user_id="user_12345",
    timestamp=datetime.now().isoformat(),
    reward=15.3,
    user_rating=4.5,
    response_time_seconds=45.2,
    ifra_compliant=True,
    experiment_id="exp_001",
    mode="creative"
)

# Filter sample
is_accepted, rejection_reason = curator.filter_sample(sample)

if is_accepted:
    # Use sample for training
    train_on_sample(sample)
else:
    logger.warning(f"Sample rejected: {rejection_reason}")
```

#### Quality Report

```python
from fragrance_ai.training.data_curation import generate_quality_report

# Generate report
report = generate_quality_report(curator)
print(report)
```

**Output Example:**

```
================================================================================
DATA CURATION QUALITY REPORT
================================================================================

Total Samples: 1000
Accepted: 847 (84.7%)
Rejected: 153 (15.3%)

Rejection Reasons:
  - response_too_fast: 45 (29.4%)
  - spam_too_many_submissions: 38 (24.8%)
  - outlier_reward_z_score: 32 (20.9%)
  - ifra_non_compliant: 23 (15.0%)
  - reward_out_of_range: 15 (9.8%)

Reward Statistics:
  Mean: 14.732
  Std: 3.891
  Min: -2.150
  Max: 28.430
  Median: 14.520

Rating Statistics:
  Mean: 4.18
  Std: 0.63
  Min: 2.50
  Max: 5.00
  Median: 4.20

================================================================================
```

---

## Configuration Management

### Externalized Configs with Tuning History

**Purpose:** Manage operational parameters outside code with full audit trail.

**Implementation:** `fragrance_ai/config/config_loader.py`

#### Configuration File

All operational parameters are in `configs/recommended_params.yaml`:

```yaml
# configs/recommended_params.yaml

moga:
  population_size: 50
  n_generations: 20
  novelty_weight:
    base: 0.1
    per_hint_increment: 0.05

rlhf:
  ppo:
    clip_eps: 0.2
    entropy_coef:
      initial: 0.01
      final: 0.001
      decay_schedule: "cosine"
      decay_steps: 10000
    learning_rate: 3.0e-4
    batch_size: 64

reward_normalization:
  enabled: true
  window_size: 1000
  clip_range: 10.0

mode_overrides:
  creative:
    moga:
      population_size: 70
    rlhf:
      ppo:
        entropy_coef:
          initial: 0.015  # More exploration
```

#### Usage

```python
from fragrance_ai.config.config_loader import ConfigLoader

# Load config
loader = ConfigLoader(
    config_file="configs/recommended_params.yaml",
    track_changes=True,
    environment="production"
)

# Get values
learning_rate = loader.get("rlhf.ppo.learning_rate")  # 3e-4
batch_size = loader.get("rlhf.ppo.batch_size")  # 64

# Get mode-specific config
creative_config = loader.get_mode_config("creative")
# Automatically merges mode_overrides.creative into base config
```

#### Update Config

```python
# Update parameter
loader.set(
    path="rlhf.ppo.learning_rate",
    value=0.0001,
    changed_by="john_doe",
    reason="Experiment ABC: reduce learning rate for stability"
)

# Save to file
loader.save()
```

#### Tuning History

All changes are tracked:

```python
from fragrance_ai.config.config_loader import print_tuning_history

# View recent changes
print_tuning_history(loader.history, n=10)
```

**Output Example:**

```
================================================================================
RECENT CONFIGURATION CHANGES (last 3)
================================================================================

[2025-10-14T15:23:45] rlhf.ppo.learning_rate
  Changed by: john_doe
  Environment: production
  Old value: 0.0003
  New value: 0.0001
  Reason: Experiment ABC: reduce learning rate for stability

[2025-10-14T14:10:22] rlhf.ppo.entropy_coef.initial
  Changed by: alice_smith
  Environment: production
  Old value: 0.01
  New value: 0.015
  Reason: Increase exploration for creative mode A/B test

[2025-10-14T10:05:11] reward_normalization.window_size
  Changed by: bob_jones
  Environment: production
  Old value: 500
  New value: 1000
  Reason: Increase window size for more stable normalization
```

#### CLI Tools

```bash
# Get config value
python -m fragrance_ai.config.config_loader --get rlhf.ppo.learning_rate

# Set config value
python -m fragrance_ai.config.config_loader --set rlhf.ppo.learning_rate 0.0001 "Experiment XYZ" --changed-by john_doe

# View tuning history
python -m fragrance_ai.config.config_loader --history 10

# View mode-specific config
python -m fragrance_ai.config.config_loader --mode creative

# View full config tree
python -m fragrance_ai.config.config_loader --tree
```

---

## Monitoring & Metrics

### Prometheus Metrics

Key operational metrics to monitor:

```promql
# Entropy coefficient (should decrease over time)
rl_entropy_coefficient

# Reward normalization statistics
rl_reward_normalizer_mean
rl_reward_normalizer_std

# Checkpoint operations
rl_checkpoint_saved_total
rl_checkpoint_rollback_total

# A/B test metrics
ab_test_requests_total{variant="control"}
ab_test_requests_total{variant="treatment_a"}
ab_test_success_rate{variant="control"}
ab_test_success_rate{variant="treatment_a"}

# Data curation metrics
data_curation_total_samples
data_curation_accepted_samples
data_curation_rejected_samples{reason="spam"}
data_curation_rejected_samples{reason="outlier"}
```

### Grafana Dashboard

Create dashboard with panels for:

1. **Entropy Annealing**: Line graph of entropy coefficient over time
2. **Reward Distribution**: Histogram of normalized rewards
3. **Checkpoint Health**: Rate of rollbacks, latest checkpoint age
4. **A/B Test Performance**: Success rate, latency, user rating by variant
5. **Data Quality**: Acceptance rate, rejection reasons breakdown
6. **Config Changes**: Timeline of parameter tuning

---

## Troubleshooting

### Entropy Not Decreasing

**Symptom:** Entropy coefficient stays high, model keeps exploring.

**Possible Causes:**
- Scheduler not initialized correctly
- `decay_steps` too large
- Entropy updates not applied in training loop

**Solution:**
```python
# Verify scheduler is working
entropy_coef = scheduler.get_entropy_coef(current_step)
print(f"Step {current_step}: entropy_coef={entropy_coef:.6f}")

# Check if applied in loss
loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
```

### Frequent Rollbacks

**Symptom:** Checkpoint manager triggers rollback every few steps.

**Possible Causes:**
- Thresholds too strict
- Learning rate too high
- Reward function unstable

**Solution:**
```python
# Relax rollback conditions
rollback_conditions = RollbackConditions(
    kl_threshold=0.05,          # Increase from 0.03
    loss_increase_multiplier=3.0,  # Increase from 2.0
    reward_drop_threshold=0.5    # Increase from 0.3
)

# Or reduce learning rate
loader.set("rlhf.ppo.learning_rate", 1e-4, changed_by="admin", reason="Reduce LR for stability")
```

### A/B Test Not Significant

**Symptom:** P-value > 0.05, no clear winner.

**Possible Causes:**
- Insufficient sample size
- Effect size too small
- High variance in metrics

**Solution:**
```python
# Check sample sizes
print(f"Control: {ab_test.control_metrics.total_requests}")
print(f"Treatment A: {ab_test.treatment_a_metrics.total_requests}")

# Continue collecting data
# Minimum recommended: 100+ samples per variant
```

### High Data Rejection Rate

**Symptom:** > 50% of feedback samples rejected.

**Possible Causes:**
- Thresholds too strict
- User behavior issues (bots, spam)
- IFRA compliance problems

**Solution:**
```python
# Review rejection reasons
stats = curator.get_statistics()
print(stats['rejection_reasons'])

# Adjust thresholds if needed
thresholds.min_response_time_seconds = 1.0  # Reduce from 2.0
thresholds.max_duplicate_submissions = 10  # Increase from 5
```

---

## Best Practices

1. **Gradual Tuning**: Make small changes, monitor for 24 hours before next change
2. **Document Changes**: Always provide clear reasons in tuning history
3. **A/B Test First**: Test major parameter changes with A/B testing before full rollout
4. **Monitor Rollbacks**: Frequent rollbacks indicate instability, investigate root cause
5. **Regular Audits**: Review tuning history weekly, identify trends
6. **Backup Configs**: Keep versioned backups of config files in git

---

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original PPO algorithm
- [Entropy Regularization](https://arxiv.org/abs/1805.00909) - Soft Actor-Critic
- [A/B Testing Best Practices](https://exp-platform.com/Documents/2013-02-WEBKDD-ExP-PitfallsInOnlineControlledExperiments.pdf)
- `configs/recommended_params.yaml` - Full parameter reference
- `docs/TUNING_GUIDE.md` - Detailed parameter tuning guide

---

**Last Updated:** 2025-10-14
**Version:** 1.0.0
