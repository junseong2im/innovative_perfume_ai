# 스모크 & 회귀 테스트 가이드

전체 시스템의 스모크 테스트 및 회귀 테스트 문서입니다.

## 목차

1. [개요](#개요)
2. [유닛/통합 테스트](#유닛통합-테스트)
3. [API 스모크 테스트](#api-스모크-테스트)
4. [JSON 로깅](#json-로깅)
5. [테스트 실행 방법](#테스트-실행-방법)

---

## 개요

### 테스트 구조

```
tests/
├── test_llm_ensemble.py          # LLM 앙상블 테스트
├── test_moga_stability.py        # MOGA 안정성 테스트 (신규)
├── test_end_to_end_evolution.py  # E2E 진화 테스트 (신규)
├── test_rl_advanced.py           # RL 고급 기능 테스트
├── test_ga.py                    # GA 테스트
└── test_ifra.py                  # IFRA 규제 테스트

smoke_test_api.sh                 # API 스모크 테스트 스크립트 (신규)
run_tests.sh                      # 전체 테스트 실행 스크립트 (신규)
```

### 테스트 범위

| 카테고리 | 테스트 파일 | 테스트 수 | 목적 |
|----------|-------------|-----------|------|
| **LLM** | test_llm_ensemble.py | 10+ | 앙상블 추론, 프롬프트, 캐싱 |
| **MOGA** | test_moga_stability.py | 10+ | 안정성, 수렴, 제약조건 |
| **E2E** | test_end_to_end_evolution.py | 8+ | 전체 파이프라인 통합 |
| **RL** | test_rl_advanced.py | 21+ | Entropy, Reward, Checkpoint |
| **GA** | test_ga.py | 15+ | 돌연변이, 교차, 선택 |
| **IFRA** | test_ifra.py | 10+ | 규제 준수, 알레르겐 |
| **API** | smoke_test_api.sh | 5 | 엔드포인트, 응답 형식 |

---

## 유닛/통합 테스트

### 1. LLM Ensemble 테스트

**파일**: `tests/test_llm_ensemble.py`

**실행**:
```bash
pytest -q tests/test_llm_ensemble.py
```

**테스트 항목**:
- LLM 모델 로딩 (Qwen, Mistral, Llama)
- Brief 생성 (fast/balanced/creative 모드)
- 프롬프트 템플릿 검증
- 캐싱 메커니즘
- 한글/영어 입력 처리
- 모드별 지연 시간

**예상 로그** (JSON):
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "level": "INFO",
  "component": "LLM",
  "message": "LLM brief generated",
  "model": "qwen",
  "mode": "balanced",
  "latency_ms": 2500.0,
  "cache_hit": false,
  "brief_style": "fresh",
  "brief_intensity": 0.7,
  "brief_complexity": 0.5
}
```

### 2. MOGA Stability 테스트

**파일**: `tests/test_moga_stability.py` (신규)

**실행**:
```bash
pytest -q tests/test_moga_stability.py
```

**테스트 항목**:
- ✅ MOGA 수렴성 테스트
- ✅ 다양성 유지 테스트
- ✅ 제약조건 만족 테스트
- ✅ 재현성 테스트 (같은 seed = 같은 결과)
- ✅ 돌연변이 안정성 테스트
- ✅ 교차 안정성 테스트
- ✅ 개체군 크기 영향 테스트
- ✅ 100k 반복 안정성 테스트
- ✅ API 호환성 테스트
- ✅ 출력 형식 테스트

**핵심 검증**:
```python
def test_moga_convergence(self):
    """MOGA 수렴성 테스트"""
    optimizer = MOGAOptimizer(
        brief=brief,
        n_generations=50
    )
    result = optimizer.optimize()

    # Check convergence
    assert len(result['pareto_front']) > 0
    assert len(result['history']) == 50

    # Check improvement
    first_gen_fitness = result['history'][0]['best_fitness']
    last_gen_fitness = result['history'][-1]['best_fitness']

    # At least one objective should improve
    assert improved
```

### 3. End-to-End Evolution 테스트

**파일**: `tests/test_end_to_end_evolution.py` (신규)

**실행**:
```bash
pytest -q tests/test_end_to_end_evolution.py
```

**테스트 항목**:
- ✅ Brief → Formulation 전체 파이프라인
- ✅ REINFORCE → PPO 진화 파이프라인
- ✅ 피드백 루프 수렴 테스트
- ✅ 다목적 최적화 트레이드오프
- ✅ 제약조건 만족 파이프라인
- ✅ 전체 API 워크플로우 시뮬레이션
- ✅ RL과 GA 결과 일관성 테스트
- ✅ 이전 버전 호환성 테스트

**전체 파이프라인 흐름**:
```python
def test_full_api_workflow(self):
    """전체 API 워크플로우 시뮬레이션"""
    # Step 1: Create DNA (brief)
    brief = CreativeBrief(...)
    dna_id = "test-dna-001"

    # Step 2: Generate options with REINFORCE
    optimizer = MOGAOptimizer(brief=brief)
    result = optimizer.optimize()
    options = result['pareto_front'][:3]

    # Step 3: User chooses and provides feedback
    chosen_option = options[0]
    rating = 5

    # Step 4: Refine with PPO (background)
    # (Simulated async workflow)
```

---

## API 스모크 테스트

### API 테스트 스크립트

**파일**: `smoke_test_api.sh` (신규)

**실행**:
```bash
bash smoke_test_api.sh
```

### 테스트 시나리오

#### 1. 서버 헬스 체크

```bash
curl -s http://localhost:8000/health | jq .
```

**예상 응답**:
```json
{
  "status": "healthy",
  "models": {
    "qwen": true,
    "mistral": true,
    "llama": true
  },
  "cache": true,
  "database": true
}
```

#### 2. DNA 생성 (POST /dna/create)

```bash
curl -s -X POST http://localhost:8000/dna/create \
  -H 'Content-Type: application/json' \
  -d '{
    "brief": {
      "style": "fresh",
      "intensity": 0.7,
      "complexity": 0.5,
      "notes_preference": {
        "citrus": 0.8,
        "fresh": 0.7,
        "floral": 0.3
      },
      "mood": ["상큼함", "활기찬"],
      "season": ["spring", "summer"]
    }
  }' | jq .
```

**예상 응답**:
```json
{
  "dna_id": "dna-001",
  "brief": {
    "style": "fresh",
    "intensity": 0.7,
    "complexity": 0.5,
    ...
  },
  "created_at": "2025-01-15T10:30:45Z"
}
```

**예상 로그** (JSON):
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "level": "INFO",
  "component": "LLM",
  "message": "LLM brief generated",
  "model": "qwen",
  "mode": "balanced",
  "latency_ms": 2500.0,
  "brief_style": "fresh",
  "brief_intensity": 0.7
}
```

#### 3. 옵션 진화 - REINFORCE (POST /evolve/options)

```bash
curl -s -X POST http://localhost:8000/evolve/options \
  -H 'Content-Type: application/json' \
  -d '{
    "dna_id": "<위의 dna_id>",
    "algorithm": "REINFORCE",
    "num_options": 3,
    "parameters": {
      "n_iterations": 50,
      "population_size": 20
    }
  }' | jq .
```

**예상 응답**:
```json
{
  "experiment_id": "exp-001",
  "options": [
    {
      "option_id": "opt-001",
      "formulation": [0.1, 0.2, 0.3, ...],
      "fitness": [0.85, 0.78],
      "characteristics": {
        "intensity": 0.7,
        "longevity": "medium"
      }
    },
    ...
  ],
  "algorithm": "REINFORCE",
  "timestamp": "2025-01-15T10:31:00Z"
}
```

**예상 로그** (JSON):
```json
{
  "timestamp": "2025-01-15T10:31:00Z",
  "level": "INFO",
  "component": "RL",
  "message": "RL update completed",
  "algorithm": "REINFORCE",
  "iteration": 50,
  "loss": 0.25,
  "reward": 15.3,
  "entropy": 0.008,
  "accept_prob": 0.65
}
```

#### 4. 피드백 제출 (POST /evolve/feedback)

```bash
curl -s -X POST http://localhost:8000/evolve/feedback \
  -H 'Content-Type: application/json' \
  -d '{
    "experiment_id": "<위의 experiment_id>",
    "chosen_id": "<option_id>",
    "rating": 5,
    "feedback_text": "상큼하고 산뜻해서 좋아요!"
  }' | jq .
```

**예상 응답**:
```json
{
  "success": true,
  "experiment_id": "exp-001",
  "chosen_option": "opt-001",
  "rating": 5,
  "next_iteration": {
    "algorithm": "PPO",
    "scheduled": true
  }
}
```

**예상 로그** (JSON - PPO 업데이트):
```json
{
  "timestamp": "2025-01-15T10:32:00Z",
  "level": "INFO",
  "component": "RL",
  "message": "RL update completed",
  "algorithm": "PPO",
  "iteration": 10,
  "loss": 0.18,
  "reward": 18.5,
  "entropy": 0.006,
  "clip_frac": 0.12,
  "value_loss": 0.10,
  "policy_loss": 0.08
}
```

#### 5. PPO 재학습 (POST /evolve/options)

```bash
curl -s -X POST http://localhost:8000/evolve/options \
  -H 'Content-Type: application/json' \
  -d '{
    "dna_id": "<dna_id>",
    "algorithm": "PPO",
    "num_options": 3,
    "parameters": {
      "n_iterations": 30,
      "n_steps_per_iteration": 512,
      "n_ppo_epochs": 5
    }
  }' | jq .
```

**예상 응답**:
```json
{
  "experiment_id": "exp-002",
  "options": [
    ...
  ],
  "algorithm": "PPO",
  "improvements": {
    "reward_increase": 1.15,
    "convergence": "stable"
  },
  "ppo_metrics": {
    "clip_frac": 0.12,
    "kl_divergence": 0.015,
    "entropy": 0.006
  }
}
```

---

## JSON 로깅

### 로깅 구조

**파일**: `fragrance_ai/observability.py`

### 주요 로거

#### 1. LLMLogger (신규 추가)

```python
from fragrance_ai.observability import llm_logger

# LLM brief 생성 로깅
llm_logger.log_brief(
    user_text="상큼한 레몬향",
    brief={
        'style': 'fresh',
        'intensity': 0.7,
        'complexity': 0.5,
        'notes_preference': {'citrus': 0.9}
    },
    model='qwen',
    mode='fast',
    latency_ms=2500.0,
    cache_hit=False
)
```

**출력**:
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "level": "INFO",
  "logger": "fragrance_ai.llm",
  "message": "LLM brief generated",
  "component": "LLM",
  "model": "qwen",
  "mode": "fast",
  "latency_ms": 2500.0,
  "cache_hit": false,
  "user_text_length": 13,
  "brief_style": "fresh",
  "brief_intensity": 0.7,
  "brief_complexity": 0.5
}
```

#### 2. RLLogger (기존)

```python
from fragrance_ai.observability import rl_logger

# RL 업데이트 로깅
rl_logger.log_update(
    algorithm='PPO',
    loss=0.25,
    reward=15.3,
    entropy=0.008,
    accept_prob=None,
    clip_frac=0.12,
    value_loss=0.15,
    policy_loss=0.10
)
```

**출력**:
```json
{
  "timestamp": "2025-01-15T10:31:00Z",
  "level": "INFO",
  "logger": "fragrance_ai.rl",
  "message": "RL update completed",
  "component": "RL",
  "algorithm": "PPO",
  "loss": 0.25,
  "reward": 15.3,
  "entropy": 0.008,
  "clip_frac": 0.12,
  "value_loss": 0.15,
  "policy_loss": 0.10
}
```

#### 3. GALogger (기존)

```python
from fragrance_ai.observability import ga_logger

# GA 세대 로깅
ga_logger.log_generation(
    generation=50,
    population_size=50,
    violation_rate=0.02,
    novelty=0.45,
    cost_norm=12.5,
    f_total=0.88,
    pareto_size=8
)
```

**출력**:
```json
{
  "timestamp": "2025-01-15T10:31:30Z",
  "level": "INFO",
  "logger": "fragrance_ai.ga",
  "message": "GA generation completed",
  "component": "GA",
  "generation": 50,
  "population_size": 50,
  "violation_rate": 0.02,
  "novelty": 0.45,
  "cost_norm": 12.5,
  "f_total": 0.88,
  "pareto_size": 8
}
```

### JSON 로그 필터링

**LLM brief 로그만 보기**:
```bash
# 로그 파일에서
cat logs/app.log | grep '"component":"LLM"' | jq .

# 실시간
tail -f logs/app.log | grep '"component":"LLM"' | jq .
```

**RL update 로그만 보기**:
```bash
cat logs/app.log | grep '"component":"RL"' | jq .
```

**특정 메트릭 추출**:
```bash
# Reward 추출
cat logs/app.log | grep '"component":"RL"' | jq '.reward'

# Loss, reward, entropy 추출
cat logs/app.log | grep '"component":"RL"' | jq '{loss, reward, entropy, clip_frac}'
```

---

## 테스트 실행 방법

### 전체 테스트 실행

```bash
# 모든 테스트 실행 (유닛 + API 스모크)
bash run_tests.sh
```

### 개별 테스트 실행

```bash
# LLM ensemble
pytest -q tests/test_llm_ensemble.py

# MOGA stability
pytest -q tests/test_moga_stability.py

# End-to-end evolution
pytest -q tests/test_end_to_end_evolution.py

# RL advanced
pytest -q tests/test_rl_advanced.py

# GA
pytest -q tests/test_ga.py

# IFRA
pytest -q tests/test_ifra.py
```

### API 스모크 테스트만 실행

```bash
# 1. API 서버 시작 (별도 터미널)
uvicorn app.main:app --reload

# 2. 스모크 테스트 실행
bash smoke_test_api.sh
```

### 특정 테스트만 실행

```bash
# MOGA 수렴성 테스트만
pytest tests/test_moga_stability.py::TestMOGAStability::test_moga_convergence -v

# E2E 전체 워크플로우 테스트만
pytest tests/test_end_to_end_evolution.py::TestEndToEndEvolution::test_full_api_workflow -v

# RL 100k 안정성 테스트만
pytest tests/test_moga_stability.py::TestMOGAStability::test_moga_100k_iterations_stability -v
```

### 상세 출력으로 실행

```bash
# 상세 출력 + 실시간 로그
pytest tests/test_moga_stability.py -v -s

# 실패 시 즉시 중단
pytest tests/ -x

# 최대 3개 실패까지만
pytest tests/ --maxfail=3
```

---

## 예상 결과

### 유닛 테스트 결과

```
tests/test_llm_ensemble.py ............          [100%]
tests/test_moga_stability.py ..........          [100%]
tests/test_end_to_end_evolution.py ........      [100%]
tests/test_rl_advanced.py .....................   [100%]
tests/test_ga.py ...............                  [100%]
tests/test_ifra.py ..........                     [100%]

========== 70+ passed in 150.25s ==========
```

### API 스모크 테스트 결과

```
=========================================
API 스모크 테스트 시작
API URL: http://localhost:8000
=========================================

=========================================
테스트 1: 서버 헬스 체크
=========================================
✓ PASS: 서버 헬스 체크

=========================================
테스트 2: DNA 생성 (POST /dna/create)
=========================================
✓ PASS: DNA 생성
DNA ID: dna-abc123

=========================================
테스트 3: 옵션 진화 - REINFORCE (POST /evolve/options)
=========================================
✓ PASS: 옵션 진화 (REINFORCE)
Experiment ID: exp-xyz789
✓ 로그에 llm_brief/rl_update 포함됨

=========================================
테스트 4: 피드백 제출 (POST /evolve/feedback)
=========================================
✓ PASS: 피드백 제출
✓ RL 업데이트 로그 포함 (loss, reward, entropy, clip_frac)

=========================================
테스트 5: PPO 재학습 (옵션 진화 - PPO)
=========================================
✓ PASS: PPO 재학습
✓ PPO 메트릭 포함 (clip_frac, kl_divergence, entropy)

=========================================
테스트 결과 요약
=========================================
통과: 5
실패: 0
총 테스트: 5
=========================================
✓ 모든 스모크 테스트 통과! ✓
```

---

## 문제 해결

### API 서버가 시작되지 않음

```bash
# 서버 시작
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### jq가 설치되지 않음

```bash
# Windows (Git Bash)
# jq 없이도 스크립트 실행 가능 (자동 감지)

# Linux
sudo apt-get install jq

# macOS
brew install jq
```

### 테스트 실패 디버깅

```bash
# 상세 출력 + 즉시 중단
pytest tests/test_moga_stability.py -v -s -x

# 특정 테스트만 + 로그 출력
pytest tests/test_moga_stability.py::TestMOGAStability::test_moga_convergence -v -s --log-cli-level=DEBUG
```

---

## 요약

### 생성된 파일

1. **테스트 파일** (신규):
   - `tests/test_moga_stability.py` (10+ tests)
   - `tests/test_end_to_end_evolution.py` (8+ tests)

2. **스크립트** (신규):
   - `smoke_test_api.sh` (API 스모크 테스트)
   - `run_tests.sh` (전체 테스트 실행)

3. **로깅** (업데이트):
   - `fragrance_ai/observability.py` (LLMLogger 추가)

4. **문서** (신규):
   - `SMOKE_AND_REGRESSION_TESTS.md` (이 파일)

### 테스트 범위

✅ **유닛 테스트**: 70+ tests
✅ **API 스모크 테스트**: 5 tests
✅ **JSON 로깅**: llm_brief, rl_update 포함
✅ **REINFORCE → PPO 진화**: 전체 파이프라인 검증

### 실행 방법

```bash
# 전체 테스트
bash run_tests.sh

# 유닛 테스트만
pytest -q tests/test_moga_stability.py
pytest -q tests/test_end_to_end_evolution.py

# API 스모크 테스트만
bash smoke_test_api.sh
```

모든 테스트가 준비되었습니다! 🚀
