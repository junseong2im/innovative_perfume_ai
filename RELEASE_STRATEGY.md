# Artisan 릴리스 전략

## 1. 브랜칭 전략 & 품질 게이트

### 1.1 Trunk-Based Development (권장)

**핵심 원칙:**
- 작은 PR 단위 (< 400 lines)
- 하루 여러 번 머지
- 긴 브랜치는 지양 (수명 < 2일)
- Feature flags로 미완성 기능 숨김

```
main (protected)
  ├── feature/small-change-1 (1일) → PR → merge
  ├── feature/small-change-2 (1일) → PR → merge
  └── hotfix/critical-bug (긴급) → PR → merge
```

**브랜치 명명 규칙:**
```
feature/brief-description    # 새로운 기능
bugfix/brief-description     # 버그 수정
hotfix/brief-description     # 긴급 패치
refactor/brief-description   # 리팩토링
docs/brief-description       # 문서 업데이트
```

### 1.2 필수 품질 게이트 (Quality Gates)

**PR마다 필수 통과:**

#### 1) 유닛/통합 테스트 (100% 통과)
```bash
# 전체 테스트 실행
pytest tests/ -v --tb=short --maxfail=5

# Critical Artisan 테스트 (필수)
pytest tests/test_llm_ensemble_operation.py -v
pytest tests/test_moga_stability.py -v
pytest tests/test_end_to_end_evolution.py -v
```

**실패 시 머지 불가 - 예외 없음**

#### 2) 정적 분석 (Static Analysis)
```bash
# Ruff (linting)
ruff check fragrance_ai/ app/ tests/

# mypy (type checking)
mypy fragrance_ai/ app/ --ignore-missing-imports
```

**에러 발생 시 머지 불가**

#### 3) 보안 스캔 (Security Scanning)
```bash
# pip-audit (취약점 검사)
pip-audit -r requirements.txt --desc

# SBOM 생성 (Software Bill of Materials)
pip install cyclonedx-bom
cyclonedx-py requirements -r -o sbom.json
```

**Critical 취약점 발견 시 머지 불가**

#### 4) 부하 스모크 테스트 (Load Smoke Test)
```bash
# 간단한 RPS 테스트로 p95 지연 확인
python scripts/load_smoke_test.py --rps 10 --duration 30 --p95-threshold 2500

# 결과 예시:
# Total Requests:  300
# Successful:      298 (99.3%)
# p95 Latency:     1847.32 ms
# ✅ PASSED: p95 latency within threshold
```

**p95 임계값 초과 시 머지 불가**

### 1.3 GitHub Actions CI/CD Pipeline

**모든 PR에서 자동 실행:**

```yaml
# .github/workflows/ci.yml

jobs:
  lint:          # Ruff + mypy
  test:          # Unit tests (pytest)
  artisan-critical-tests:  # LLM Ensemble, MOGA, E2E
  security:      # pip-audit, SBOM
  load-smoke-test:  # RPS 기반 p95 체크
  smoke-test:    # 샘플 추론
  docker-build:  # Docker 이미지 빌드
  summary:       # 전체 결과 집계
```

**모든 job이 성공해야 PR 머지 가능**

### 1.4 릴리스 태깅 & 체크포인트

**SemVer + 모델 스냅샷:**

```bash
# 릴리스 태그 생성 (자동으로 모델 체크포인트 스냅샷 포함)
python scripts/release_tag.py \
  --version v2.1.0 \
  --notes "LLM 앙상블 추가, PPO 알고리즘 개선" \
  --checkpoint-dir ./checkpoints \
  --push

# 결과:
# Version:         v2.1.0
# Model Snapshot:  model-20251014-abc12345
# Checkpoint Hash: a3f5b8c9d1e2f4a6...
# Git Tag:         v2.1.0 (pushed to origin)
```

**릴리스 히스토리 조회:**
```bash
python scripts/release_tag.py --list

# 출력:
# Version: v2.1.0
#   Model Snapshot: model-20251014-abc12345
#   Checkpoint Hash: a3f5b8c9...
#   Timestamp: 2025-10-14T15:30:00
#
# Version: v2.0.3
#   Model Snapshot: model-20251001-def67890
#   Checkpoint Hash: b2c4d6e8...
#   Timestamp: 2025-10-01T10:15:00
```

### 1.5 Artisan에 적용

**필수 테스트를 PR 머지 전 필수로:**
- `test_llm_ensemble_operation.py` - 3모델 앙상블 테스트
- `test_moga_stability.py` - MOGA 안정성 (10k 반복)
- `test_end_to_end_evolution.py` - DNA → 진화 → 피드백 전체 플로우

**CI 파이프라인에서 자동 실행:**
```yaml
# .github/workflows/ci.yml (발췌)

artisan-critical-tests:
  name: Critical Tests (LLM Ensemble, MOGA Stability, E2E)
  runs-on: ubuntu-latest
  steps:
    - name: Run test_llm_ensemble.py
      run: pytest tests/test_llm_ensemble_operation.py -v --maxfail=1
      timeout-minutes: 10

    - name: Run test_moga_stability.py
      run: pytest tests/test_moga_stability.py -v --maxfail=1
      timeout-minutes: 15

    - name: Run test_end_to_end_evolution.py
      run: pytest tests/test_end_to_end_evolution.py -v --maxfail=1
      timeout-minutes: 20
```

### 1.6 PR 체크리스트

**PR 생성 시 반드시 확인:**

```markdown
## Quality Gates Checklist

### 필수 테스트 (Mandatory)
- [ ] All unit tests pass
- [ ] Critical Artisan tests pass (LLM Ensemble, MOGA, E2E)

### 정적 분석 (Static Analysis)
- [ ] Ruff linting pass
- [ ] mypy type checking pass

### 보안 스캔 (Security)
- [ ] pip-audit security scan pass
- [ ] No critical vulnerabilities

### 성능 테스트 (Performance)
- [ ] Load smoke test pass
- [ ] p95 latency within threshold

### 코드 품질 (Code Quality)
- [ ] Code follows style guide
- [ ] Comments for complex logic
- [ ] Error handling implemented
```

**템플릿 위치:** `.github/pull_request_template.md`

---

## 2. 환경 3단계

### 배포 파이프라인
```
dev → stg → prod
```

### 환경별 특징

| 환경 | 용도 | 데이터 | 배포 주기 |
|------|------|--------|-----------|
| **dev** | 개발/테스트 | Mock/샘플 | 수시 |
| **stg** | 스테이징/QA | 익명화 실데이터 | 주 2-3회 |
| **prod** | 프로덕션 | 실데이터 | 주 1-2회 (Release Train) |

### 환경 설정

```python
# configs/environment_config.py 사용
from configs.environment_config import get_config

config = get_config()  # ARTISAN_ENV 환경 변수 기반

print(f"Environment: {config.env}")
print(f"Database: {config.database.host}")
print(f"LLM Endpoint: {config.llm.qwen_endpoint}")
```

### 시크릿 분리

```bash
# Dev
export DB_PASSWORD_DEV="dev_password"

# Staging
export DB_PASSWORD_STG=$(vault read -field=password secret/artisan/stg/db)

# Production
export DB_PASSWORD_PROD=$(vault read -field=password secret/artisan/prod/db)
```


## 3. 점진적 배포

### 3.1 카나리 배포 (Canary)

**전략:** 1% → 5% → 25% → 100%

```bash
# Phase 1: 1% 트래픽
bash scripts/deploy_canary.sh stg advanced_rlhf 1

# 10분 모니터링 후...

# Phase 2: 5% 트래픽
bash scripts/deploy_canary.sh stg advanced_rlhf 5

# 20분 모니터링 후...

# Phase 3: 25% 트래픽
bash scripts/deploy_canary.sh stg advanced_rlhf 25

# 30분 모니터링 후...

# Phase 4: 100% 전체 배포
bash scripts/deploy_canary.sh stg advanced_rlhf 100
```

**모니터링 지표:**
- 에러율 < 1%
- p95 지연 < 임계값
- 스키마 실패율 = 0%

**롤백 조건:**
- 에러율 급증 (> 2%)
- p95 초과 (> 1.2x baseline)
- 사용자 피드백 급락 (< 3.0)

### 3.2 블루/그린 배포 (Blue/Green)

**전략:** 신규(Green)와 구버전(Blue) 동시 가동

```bash
# Green 배포 및 트래픽 스위치
bash scripts/deploy_blue_green.sh stg v2.1.0

# 단계:
# 1. Green 배포 (신규 버전)
# 2. Health check
# 3. Go/No-Go 체크
# 4. 트래픽 전환 (Blue → Green)
# 5. 모니터링 (5분)
# 6. Blue 제거
```

**장점:**
- 빠른 롤백 (트래픽만 다시 Blue로)
- Zero downtime
- 안전한 데이터베이스 마이그레이션

### 3.3 피처 플래그 (Feature Flags)

**Artisan 적용 사례:**

```python
from fragrance_ai.config.feature_flags import is_enabled

# RL 파이프라인 토글
if is_enabled("rl_pipeline_enabled", user_id=user_id):
    result = rl_pipeline.evolve(dna, brief)
else:
    result = legacy_ga.optimize(dna, brief)

# PPO vs REINFORCE 전환
algorithm = "PPO" if is_enabled("ppo_algorithm") else "REINFORCE"
trainer = get_trainer(algorithm=algorithm)

# LLM 앙상블 토글
if is_enabled("llm_ensemble_enabled"):
    brief = llm_ensemble.generate(prompt)
else:
    brief = single_llm.generate(prompt)
```

**주요 피처 플래그:**

| 플래그 | 설명 | 기본값 |
|--------|------|--------|
| `rl_pipeline_enabled` | RL 파이프라인 활성화 | ✓ |
| `llm_ensemble_enabled` | 3모델 앙상블 | ✓ |
| `ppo_algorithm` | PPO 사용 (False=REINFORCE) | ✓ |
| `cache_enabled` | LLM 캐시 | ✓ |
| `circuit_breaker_enabled` | 서킷브레이커 | ✓ |
| `new_moga_optimizer` | 신규 MOGA (실험) | ✗ |
| `advanced_rlhf` | 고급 RLHF | ✗ (dev only) |


## 4. 릴리스 기차 (Release Train)

### 고정 주기 배포

**주기:** 매주 화요일 & 금요일 오전 10시

```
월: 개발
화: 🚂 Release Train #1 (dev → stg)
수: 개발
목: QA/테스트
금: 🚂 Release Train #2 (stg → prod)
```

### 릴리스 프로세스

#### 화요일 (STG 배포)
```bash
# 1. Feature freeze (월요일 18:00)
git checkout main
git tag release-candidate-$(date +%Y%m%d)

# 2. Go/No-Go 체크 (화요일 09:00)
python -m fragrance_ai.deployment.go_nogo_gate --exit-code

# 3. STG 배포 (화요일 10:00)
ARTISAN_ENV=stg bash scripts/deploy_blue_green.sh stg release-candidate-20251014

# 4. 스모크 테스트
pytest tests/smoke_test.py --env=stg

# 5. 회귀 테스트 (화~목)
pytest tests/ --env=stg --regression
```

#### 금요일 (PROD 배포)
```bash
# 1. STG 검증 완료 확인
# 2. Go/No-Go 체크
python -m fragrance_ai.deployment.go_nogo_gate --exit-code --prometheus-url=http://stg-prometheus:9090

# 3. PROD 카나리 배포 (10:00)
bash scripts/deploy_canary.sh prod rl_pipeline_enabled 5

# 4. 단계별 증가 (10:30, 11:00, 11:30)
bash scripts/deploy_canary.sh prod rl_pipeline_enabled 25
bash scripts/deploy_canary.sh prod rl_pipeline_enabled 50
bash scripts/deploy_canary.sh prod rl_pipeline_enabled 100

# 5. 모니터링 (금요일 오후)
# 6. 주말 온콜 대기
```

### 긴급 패치 (Hotfix)

**프로세스:**
1. `hotfix/` 브랜치 생성
2. 수정 및 테스트
3. Go/No-Go 체크
4. PROD 직접 배포 (카나리 스킵 가능)
5. Main 브랜치 머지

```bash
# Hotfix 브랜치
git checkout -b hotfix/critical-bug-fix main

# 수정 및 커밋
git commit -m "hotfix: Fix critical bug"

# Go/No-Go
python -m fragrance_ai.deployment.go_nogo_gate --exit-code

# PROD 배포 (긴급)
ARTISAN_ENV=prod bash scripts/deploy_blue_green.sh prod hotfix-$(date +%Y%m%d-%H%M)

# Main 머지
git checkout main
git merge --no-ff hotfix/critical-bug-fix
git push origin main
```


## 5. 마이그레이션 관리

### 데이터베이스 마이그레이션

**도구:** Alembic (SQLAlchemy)

```bash
# 마이그레이션 생성
alembic revision --autogenerate -m "Add user_feedback table"

# Dev 적용
ARTISAN_ENV=dev alembic upgrade head

# Stg 적용 (릴리스 기차)
ARTISAN_ENV=stg alembic upgrade head

# Prod 적용 (릴리스 기차)
ARTISAN_ENV=prod alembic upgrade head

# 롤백
ARTISAN_ENV=prod alembic downgrade -1
```

### 마이그레이션 원칙

1. **하위 호환성 유지**
   ```python
   # ❌ BAD: 컬럼 삭제 (즉시)
   op.drop_column('users', 'old_field')

   # ✓ GOOD: 2단계로 진행
   # Step 1: NULL 허용으로 변경 (v2.1.0)
   op.alter_column('users', 'old_field', nullable=True)

   # Step 2: 실제 삭제 (v2.2.0, 2주 후)
   op.drop_column('users', 'old_field')
   ```

2. **Dual-write 패턴**
   ```python
   # 마이그레이션 중 신구 필드 동시 쓰기
   def save_user(user_data):
       db.execute(
           "INSERT INTO users (name, new_field, old_field) "
           "VALUES (:name, :new, :old)",
           name=user_data["name"],
           new=compute_new_field(user_data),
           old=compute_old_field(user_data)  # 호환성
       )
   ```

3. **블루/그린과 결합**
   ```bash
   # Step 1: 신규 컬럼 추가 (Blue 버전)
   alembic upgrade head

   # Step 2: Green 배포 (신규 컬럼 사용)
   bash scripts/deploy_blue_green.sh prod v2.1.0

   # Step 3: Blue 제거 후 구 컬럼 삭제 (다음 릴리스)
   ```


## 6. 실전 예시

### 예시 1: 신규 RLHF 기능 배포

```bash
# Week 1: Dev 개발
ARTISAN_ENV=dev python -m fragrance_ai.config.feature_flags \
    --enable advanced_rlhf --rollout 100

# Week 2 (화): STG 배포 (카나리)
bash scripts/deploy_canary.sh stg advanced_rlhf 5
# 모니터링...
bash scripts/deploy_canary.sh stg advanced_rlhf 25
# 모니터링...
bash scripts/deploy_canary.sh stg advanced_rlhf 100

# Week 2 (금): PROD 배포 (카나리)
bash scripts/deploy_canary.sh prod advanced_rlhf 1
# 금요일 오후 모니터링...

# Week 3 (월-목): 단계적 증가
bash scripts/deploy_canary.sh prod advanced_rlhf 5   # 월
bash scripts/deploy_canary.sh prod advanced_rlhf 25  # 화
bash scripts/deploy_canary.sh prod advanced_rlhf 50  # 수
bash scripts/deploy_canary.sh prod advanced_rlhf 100 # 목

# 완료!
```

### 예시 2: PPO → REINFORCE 전환

```python
# 피처 플래그로 즉시 전환 (코드 배포 불필요)
from fragrance_ai.config.feature_flags import FeatureFlagManager

manager = FeatureFlagManager(environment="prod")

# REINFORCE로 전환 (5% 트래픽)
manager.disable_flag("ppo_algorithm")  # False = REINFORCE
manager.set_rollout_percentage("ppo_algorithm", 5)

# 모니터링 후 단계적 증가
manager.set_rollout_percentage("ppo_algorithm", 25)
manager.set_rollout_percentage("ppo_algorithm", 100)

# 문제 발생 시 즉시 롤백
manager.enable_flag("ppo_algorithm")  # True = PPO
```


## 7. 체크리스트

### 배포 전
- [ ] 모든 테스트 통과 (79개)
- [ ] Go/No-Go 게이트 통과
- [ ] Feature flags 설정 확인
- [ ] 마이그레이션 스크립트 준비
- [ ] 롤백 계획 수립
- [ ] 모니터링 대시보드 확인

### 배포 중
- [ ] 카나리/블루그린 단계별 실행
- [ ] 각 단계마다 메트릭 확인
- [ ] 에러율/지연 모니터링
- [ ] 사용자 피드백 모니터링
- [ ] 온콜 엔지니어 대기

### 배포 후
- [ ] 스모크 테스트 실행
- [ ] 회귀 테스트 실행
- [ ] 주요 지표 24시간 모니터링
- [ ] 사용자 피드백 100건 이상 수집
- [ ] 릴리스 노트 작성
- [ ] 포스트모템 (문제 발생 시)


## 8. 도구 및 명령어

```bash
# Quality Gates
pytest tests/ -v                               # 전체 테스트
ruff check fragrance_ai/ app/ tests/          # Linting
mypy fragrance_ai/ app/                       # Type checking
pip-audit -r requirements.txt --desc          # Security scan
python scripts/load_smoke_test.py            # Load smoke test

# Release tagging
python scripts/release_tag.py --version v2.1.0 --notes "Release notes" --push
python scripts/release_tag.py --list         # List all releases

# Feature flags
python -m fragrance_ai.config.feature_flags

# Environment config
python -m configs.environment_config

# Canary deployment
bash scripts/deploy_canary.sh <env> <flag> <percentage>

# Blue/Green deployment
bash scripts/deploy_blue_green.sh <env> <version>

# Go/No-Go gate
python -m fragrance_ai.deployment.go_nogo_gate --exit-code

# Migrations
alembic revision -m "message"
alembic upgrade head
alembic downgrade -1
```


## 요약

| 전략 | 사용 시기 | 장점 | 단점 |
|------|-----------|------|------|
| **카나리** | 신규 기능, 점진적 | 안전, 롤백 쉬움 | 시간 소요 |
| **블루/그린** | 메이저 업데이트 | Zero downtime | 리소스 2배 |
| **피처 플래그** | A/B 테스트, 토글 | 즉시 전환 | 코드 복잡도 |
| **릴리스 기차** | 정기 배포 | 예측 가능 | 유연성 낮음 |

**Artisan은 세 가지 전략을 모두 활용하여 안전하고 빠른 배포를 실현합니다!** 🚀
