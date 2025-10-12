# 코드베이스 정리 계획 (Codebase Cleanup Plan)

## 문제점
기능 업데이트할 때마다 새 파일을 만들어서 중복 파일이 다수 존재합니다.

## 중복 파일 분석

### 1. RL/RLHF 관련 (5개 파일 → 2개로 통합)

**현재:**
- ✅ `reinforcement_learning.py` (사용 중)
- ✅ `reinforcement_learning_ppo.py` (사용 중)
- ❌ `reinforcement_learning_enhanced.py` (미사용)
- ✅ `rlhf_complete.py` (사용 중 - 메인)
- ❌ `rlhf_enhanced.py` (미사용)
- ❌ `real_rlhf.py` (미사용)
- ❌ `enhanced_rlhf_policy.py` (미사용)

**제안:**
- `rlhf_complete.py`를 `rlhf_engine.py`로 rename (명확한 이름)
- 미사용 파일 3개 삭제:
  - reinforcement_learning_enhanced.py
  - rlhf_enhanced.py
  - real_rlhf.py
  - enhanced_rlhf_policy.py

### 2. MOGA 관련 (2개 파일 - 유지)

**현재:**
- ✅ `moga_optimizer.py` (living_scent_orchestrator 사용)
- ✅ `moga_optimizer_stable.py` (test_ga.py 100k 테스트 사용)

**제안:**
- **둘 다 유지** - 각각 다른 용도:
  - `moga_optimizer.py`: Production GA
  - `moga_optimizer_stable.py`: 안정화된 수치 검증용 (100k 테스트)

### 3. Orchestrator 관련

**현재:**
- ✅ `orchestrator.py` (LLMOrchestrator)
- ✅ `artisan_orchestrator.py` (사용 중)
- ❌ `artisan_orchestrator_enhanced.py` (미사용)
- ✅ `living_scent_orchestrator.py` (메인 - __init__ export)
- ✅ `living_scent_orchestrator_evolution.py` (사용 중)
- ✅ `rlhf_orchestrator.py` (사용 중)
- ✅ `ai_perfumer_orchestrator.py` (사용 중)
- ✅ `customer_service_orchestrator.py` (사용 중)

**제안:**
- 미사용 파일 1개 삭제: `artisan_orchestrator_enhanced.py`

## 정리 작업 요약

### 삭제 대상 파일 (총 5개)
```bash
fragrance_ai/training/reinforcement_learning_enhanced.py
fragrance_ai/training/rlhf_enhanced.py
fragrance_ai/training/real_rlhf.py
fragrance_ai/training/enhanced_rlhf_policy.py
fragrance_ai/orchestrator/artisan_orchestrator_enhanced.py
```

### Rename 작업 (선택)
```bash
# 명확한 이름으로 변경
rlhf_complete.py → rlhf_engine.py
# import 경로 업데이트 필요:
# - fragrance_ai/orchestrator/rlhf_orchestrator.py
# - test_rlhf_complete.py
```

## 향후 개발 가이드라인

### ❌ 하지 말아야 할 것:
- 기능 업데이트할 때마다 새 파일 생성 (`_enhanced`, `_complete`, `_v2` 등)
- 기존 파일을 남겨두고 새 파일에 복사-수정

### ✅ 해야 할 것:
- 기존 파일을 **직접 수정**
- Git history로 변경 추적 (rollback 가능)
- Breaking change가 필요하면:
  1. 기존 API 유지하면서 내부 구현만 개선
  2. 정말 필요한 경우에만 새 파일 생성 후 **기존 파일 삭제**
  3. Migration guide 작성

## 실행 명령어

```bash
# 1. 미사용 파일 삭제
git rm fragrance_ai/training/reinforcement_learning_enhanced.py
git rm fragrance_ai/training/rlhf_enhanced.py
git rm fragrance_ai/training/real_rlhf.py
git rm fragrance_ai/training/enhanced_rlhf_policy.py
git rm fragrance_ai/orchestrator/artisan_orchestrator_enhanced.py

# 2. (선택) Rename
git mv fragrance_ai/training/rlhf_complete.py fragrance_ai/training/rlhf_engine.py

# 3. Import 경로 업데이트 (rename한 경우)
# - rlhf_orchestrator.py
# - test_rlhf_complete.py

# 4. Commit
git commit -m "chore: Remove duplicate and unused files

- Remove 5 unused _enhanced/_real files
- Keep production files only
- Improve codebase clarity

Deleted files:
- reinforcement_learning_enhanced.py (unused)
- rlhf_enhanced.py (unused)
- real_rlhf.py (unused)
- enhanced_rlhf_policy.py (unused)
- artisan_orchestrator_enhanced.py (unused)
"
```

## 파일 용도 정리 (최종 상태)

### Training (`fragrance_ai/training/`)
- `reinforcement_learning.py` - Base RL engine (REINFORCE)
- `reinforcement_learning_ppo.py` - PPO extension
- `rlhf_engine.py` (or `rlhf_complete.py`) - Complete RLHF with PPO + REINFORCE
- `moga_optimizer.py` - Production GA
- `moga_optimizer_stable.py` - Stable GA for 100k stress tests

### Orchestrator (`fragrance_ai/orchestrator/`)
- `orchestrator.py` - Base LLM orchestrator
- `living_scent_orchestrator.py` - Main RLHF orchestrator (exported)
- `living_scent_orchestrator_evolution.py` - Evolution extension
- `rlhf_orchestrator.py` - RLHF-specific orchestrator
- `artisan_orchestrator.py` - Artisan workflow
- `ai_perfumer_orchestrator.py` - AI perfumer workflow
- `customer_service_orchestrator.py` - Customer service

## 검증 체크리스트

- [ ] 삭제 후 모든 import 에러 없는지 확인
- [ ] 테스트 실행: `pytest tests/`
- [ ] API 서버 실행: `uvicorn app.main:app`
- [ ] 100k GA 테스트: `pytest tests/test_ga.py -v`
