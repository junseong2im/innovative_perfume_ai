# Pull Request

## Description
<!-- 이 PR이 무엇을 변경하는지 간단히 설명해주세요 -->



## Type of Change
<!-- 해당하는 항목에 [x] 표시 -->

- [ ] 🐛 Bug fix (버그 수정)
- [ ] ✨ New feature (새로운 기능)
- [ ] 🔧 Refactoring (기능 변경 없이 코드 개선)
- [ ] 📝 Documentation (문서 업데이트)
- [ ] 🎨 UI/UX (사용자 인터페이스 개선)
- [ ] ⚡ Performance (성능 개선)
- [ ] 🔒 Security (보안 개선)
- [ ] 🧪 Tests (테스트 추가/수정)
- [ ] 🔨 Build/CI (빌드 또는 CI 설정 변경)

## Related Issue
<!-- 관련 이슈가 있으면 링크를 추가해주세요 -->
Closes #

## Changes
<!-- 주요 변경 사항을 나열해주세요 -->

-
-
-

## Testing
<!-- 어떻게 테스트했는지 설명해주세요 -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

### Test Commands
```bash
# 테스트 실행 명령어를 여기에 작성
pytest tests/
```

## Quality Gates Checklist
<!-- 모든 항목이 체크되어야 PR 머지가 가능합니다 -->

### 필수 테스트 (Mandatory)
- [ ] **All unit tests pass** (`pytest tests/`)
- [ ] **Critical Artisan tests pass**:
  - [ ] `test_llm_ensemble_operation.py` - LLM 앙상블 테스트
  - [ ] `test_moga_stability.py` - MOGA 안정성 테스트
  - [ ] `test_end_to_end_evolution.py` - E2E 진화 테스트

### 정적 분석 (Static Analysis)
- [ ] **Ruff linting pass** (`ruff check fragrance_ai/ app/ tests/`)
- [ ] **mypy type checking pass** (`mypy fragrance_ai/ app/`)

### 보안 스캔 (Security)
- [ ] **pip-audit security scan pass** (`pip-audit -r requirements.txt`)
- [ ] No critical vulnerabilities introduced
- [ ] SBOM generated (if dependencies changed)

### 성능 테스트 (Performance)
- [ ] **Load smoke test pass** (`python scripts/load_smoke_test.py`)
- [ ] p95 latency within threshold (< 2.5s)
- [ ] No performance regression

### 코드 품질 (Code Quality)
- [ ] Code follows project style guide
- [ ] Comments added for complex logic
- [ ] No console.log or debug prints left
- [ ] Error handling implemented
- [ ] Edge cases considered

### 문서 (Documentation)
- [ ] README updated (if needed)
- [ ] API documentation updated (if API changed)
- [ ] Release notes added (for significant changes)

## Deployment Considerations
<!-- 배포 시 고려사항이 있으면 작성해주세요 -->

- [ ] Database migrations required? (If yes, include Alembic scripts)
- [ ] Environment variables changed? (If yes, document in `.env.example`)
- [ ] Feature flags needed? (If yes, update `feature_flags.py`)
- [ ] Breaking changes? (If yes, document migration path)

### Rollback Plan
<!-- 문제 발생 시 롤백 계획을 작성해주세요 -->


## Screenshots (if applicable)
<!-- UI 변경이 있으면 스크린샷을 추가해주세요 -->


## Additional Notes
<!-- 리뷰어가 알아야 할 추가 정보 -->


---

## Reviewer Checklist
<!-- 리뷰어가 확인할 항목 -->

- [ ] Code is clear and maintainable
- [ ] Tests cover critical paths
- [ ] No security vulnerabilities
- [ ] Documentation is adequate
- [ ] Performance impact acceptable
- [ ] Deployment plan is sound

---

**Before merging:**
1. ✅ All CI checks pass
2. ✅ At least 1 approval from code owner
3. ✅ All conversations resolved
4. ✅ Branch is up-to-date with `main`
