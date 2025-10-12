# 🧪 Fragrance AI Test Suite

Comprehensive test suite for the Fragrance AI system with observability, metrics, and extensive validation.

## 📊 Test Overview

| Test Suite | Description | Test Count | Duration |
|------------|-------------|------------|----------|
| **test_ga.py** | Genetic Algorithm (MOGA) stability tests with 100k mutations | 15+ | ~30s |
| **test_rl.py** | Reinforcement Learning with fake users (50 steps) | 12+ | ~20s |
| **test_ifra.py** | IFRA compliance, boundary conditions, epsilon smoothing | 18+ | ~5s |
| **test_api.py** | API integration with 200 complete flows | 10+ | ~60s |

## 🚀 Quick Start

### Run All Tests
```bash
# Using Python directly
python run_tests.py all

# Using pytest
pytest tests/ -v

# Quick tests only (skip slow tests)
python run_tests.py all --quick
```

### Run Specific Test Suites
```bash
# Genetic Algorithm tests
python run_tests.py ga

# Reinforcement Learning tests
python run_tests.py rl

# IFRA compliance tests
python run_tests.py ifra

# API integration tests
python run_tests.py api

# Unit tests only
python run_tests.py unit

# Integration tests only
python run_tests.py integration
```

### Run with Coverage
```bash
# Generate HTML coverage report
python run_tests.py all --coverage

# View coverage report
# Opens in htmlcov/index.html
```

## 📁 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── test_ga.py              # GA/MOGA tests (100k mutations)
├── test_rl.py              # RL policy tests with fake users
├── test_ifra.py            # IFRA compliance tests
├── test_api.py             # API integration tests (200 flows)
└── README.md               # This file
```

## 🧬 GA Tests (test_ga.py)

### Key Tests

1. **test_mutation_100k_iterations**
   - Runs 100,000 mutations
   - Validates: No negatives, sum=100%, IFRA compliance
   - Expected: 0 violations

2. **test_exponential_mutation_positivity**
   - Tests c' = c * exp(N(0,σ)) guarantees positivity
   - 1,000 iterations with extreme cases

3. **test_ifra_clipping_convergence**
   - Tests IFRA limit enforcement with iterative renormalization
   - Validates convergence within 10 iterations

4. **test_minimum_concentration_filtering**
   - Tests c_min = 0.1% filtering
   - Validates renormalization after filtering

5. **test_entropy_calculation**
   - Tests entropy with edge cases (zeros, tiny values, single ingredient)
   - Validates no NaN/Inf, bounds [0,1]

### Logged Metrics
```json
{
  "component": "GA",
  "generation": 0,
  "population_size": 100000,
  "violation_rate": 0.0,
  "novelty": 0.85,
  "cost_norm": 125.3,
  "f_total": 0.72,
  "pareto_size": 15
}
```

## 🤖 RL Tests (test_rl.py)

### Fake User Types

| User Type | Behavior | Purpose |
|-----------|----------|---------|
| **Random** | Random choices/ratings | Baseline |
| **Consistent** | Prefers specific actions | Policy learning |
| **Improving** | Ratings improve over time | Reward signal |
| **Critical** | Low→High ratings | Adaptation |
| **Generous** | Always high ratings | Positive reinforcement |

### Key Tests

1. **test_reinforce_with_fake_users**
   - 50 steps with 3 fake user types
   - Validates reward improvement trend
   - Checks policy updates

2. **test_ppo_with_fake_users**
   - 50 steps with PPO algorithm
   - Validates clip_fraction, entropy, value_loss
   - Tests advantage estimation

3. **test_policy_distribution_change**
   - Validates policy learns from consistent rewards
   - Checks KL divergence > 0.01

4. **test_reward_normalization**
   - Tests (rating-3)/2 mapping [1,5]→[-1,1]
   - Validates all boundary cases

### Logged Metrics
```json
{
  "component": "RL",
  "algorithm": "PPO",
  "loss": 0.0125,
  "reward": 0.5,
  "entropy": 2.1,
  "accept_prob": 0.92,
  "clip_frac": 0.15,
  "value_loss": 0.008,
  "policy_loss": 0.004
}
```

## 🧪 IFRA Tests (test_ifra.py)

### Key Tests

1. **test_violation_detection**
   - Tests detection of IFRA limit violations
   - Validates penalty calculation

2. **test_boundary_conditions**
   - Tests exact limits, ε over/under
   - Validates epsilon smoothing (1e-10 handling)

3. **test_cumulative_penalty**
   - Tests penalty formula:
     - Prohibited: 100 × concentration
     - Restricted: 10 × (1 + excess_ratio)²
   - Validates cumulative sum

4. **test_allergen_threshold**
   - Tests EU allergen declaration (>10 ppm)
   - Validates concentration × product_conc calculation

5. **test_epsilon_smoothing**
   - Tests handling of tiny values (1e-10)
   - Validates no NaN/Inf in calculations

### Example Violations
```python
# Bergamot at 5% (limit 2%)
penalty = 10 * (1 + 3/2)² = 62.5

# Musk Xylene at 2% (prohibited)
penalty = 100 * 2 = 200

# Total penalty = 262.5
```

## 🌐 API Tests (test_api.py)

### Test Flow
```
1. POST /dna/create → dna_id
2. POST /evolve/options → experiment_id + 3 options
3. POST /evolve/feedback → rating → RL update
4. GET /experiments/{id} → status
5. Repeat 200 times
```

### Key Tests

1. **test_complete_flow_200_responses**
   - Complete flow with 200 user interactions
   - Alternates PPO/REINFORCE algorithms
   - Validates learning signal (reward ↑)
   - Expected: <10 errors, avg time <1s

2. **test_error_handling**
   - 400: Bad request
   - 404: DNA/Experiment not found
   - 422: Validation error
   - 500: Internal error

3. **test_concurrent_requests**
   - 20 concurrent requests
   - Expected: >90% success rate

### Logged Metrics
```json
{
  "component": "Orchestrator",
  "experiment_id": "exp_xyz789",
  "user_id_hash": "a1b2c3d4",
  "action": "feedback_processed",
  "timing_ms": 125.3,
  "success": true
}
```

## 📈 Observability

### JSON Logging
All tests emit structured JSON logs:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "logger": "fragrance_ai.ga",
  "message": "GA generation completed",
  "component": "GA",
  "generation": 5,
  "violation_rate": 0.02,
  "f_total": 0.85
}
```

### Prometheus Metrics
Access metrics at `/metrics` endpoint:

```
# GA Metrics
fragrance_ga_generations_total 150
fragrance_ga_violation_rate 0.02
fragrance_ga_fitness 0.85

# RL Metrics
fragrance_rl_updates_total{algorithm="PPO"} 1000
fragrance_rl_reward{algorithm="PPO"} 0.65
fragrance_rl_loss{algorithm="PPO"} 0.012

# API Metrics
fragrance_api_requests_total{method="POST",endpoint="/dna/create",status="201"} 500
fragrance_api_response_seconds_sum 125.5
```

## 🎯 Test Markers

Use pytest markers to filter tests:

```bash
# Skip slow tests (100k iterations)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only GA tests
pytest -m ga

# Run only RL tests
pytest -m rl

# Run only benchmarks
pytest -m benchmark
```

## 🔧 Configuration

### pytest.ini
```ini
[tool:pytest]
markers =
    slow: 100k+ iterations
    integration: API integration tests
    unit: Unit tests
    ga: Genetic Algorithm tests
    rl: Reinforcement Learning tests
    ifra: IFRA compliance tests
    benchmark: Performance benchmarks
```

### conftest.py
Shared fixtures:
- `sample_brief`: Creative brief for testing
- `sample_recipe`: Recipe for IFRA testing
- `temp_dir`: Temporary directory
- `track_test_performance`: Auto-tracks slow tests (>2s)

## 📊 Expected Results

### GA Tests
```
✓ 100,000 mutations: 0 negatives, 0 sum errors
✓ IFRA clipping: converges in <10 iterations
✓ Entropy: handles edge cases, no NaN/Inf
✓ Performance: <100ms average per mutation
```

### RL Tests
```
✓ 50 steps with fake users: reward trend ↑
✓ Policy distribution change: KL div > 0.01
✓ Reward normalization: [1,5] → [-1,1]
✓ PPO clip fraction: 0.1-0.3
```

### IFRA Tests
```
✓ Boundary conditions: exact limits handled
✓ Epsilon smoothing: no NaN/Inf with 1e-10
✓ Cumulative penalty: formula verified
✓ Allergen thresholds: >10 ppm declared
```

### API Tests
```
✓ 200 complete flows: <10 errors
✓ Average response time: <1000ms
✓ Learning signal: rating 2.5→4.0
✓ Concurrent load: >90% success
```

## 🐛 Debugging

### Verbose Output
```bash
# Show all output including prints
pytest tests/ -v -s

# Show only failures
pytest tests/ --tb=short

# Stop on first failure
pytest tests/ -x
```

### Debug Specific Test
```bash
# Run single test with full output
pytest tests/test_ga.py::TestGAMutations::test_mutation_100k_iterations -v -s

# Run single test class
pytest tests/test_rl.py::TestRLAlgorithms -v
```

### Check Coverage
```bash
# Generate coverage report
pytest --cov=fragrance_ai --cov-report=html tests/

# View line-by-line coverage
open htmlcov/index.html
```

## 📝 Adding New Tests

### 1. Create Test File
```python
# tests/test_myfeature.py
import pytest

class TestMyFeature:
    def setup_method(self):
        """Setup before each test"""
        pass

    def test_basic_functionality(self):
        """Test basic feature"""
        assert True
```

### 2. Add Markers
```python
@pytest.mark.unit
@pytest.mark.myfeature
def test_something():
    pass
```

### 3. Add to pytest.ini
```ini
markers =
    myfeature: marks tests for my feature
```

### 4. Add Logging
```python
from fragrance_ai.observability import get_logger

logger = get_logger("test_myfeature")
logger.info("Test completed", metric=value)
```

## 🚨 Common Issues

### Issue: Tests fail with import errors
**Solution**: Ensure project is in PYTHONPATH
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Slow test warnings
**Solution**: Mark slow tests
```python
@pytest.mark.slow
def test_100k_iterations():
    pass
```

### Issue: API tests timeout
**Solution**: Increase timeout or reduce iterations
```python
client.post("/api/endpoint", timeout=30.0)
```

## 📚 References

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Prometheus Python Client](https://github.com/prometheus/client_python)

## ✅ Test Checklist

Before committing:
- [ ] All tests pass
- [ ] Coverage >80%
- [ ] No slow tests without markers
- [ ] Observability logs present
- [ ] API tests include error cases
- [ ] Documentation updated