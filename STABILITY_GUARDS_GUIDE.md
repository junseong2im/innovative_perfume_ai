# 안정성 가드 시스템 (Stability Guards System)

## 개요

Fragrance AI의 안정성 가드 시스템은 프로덕션 환경에서 서비스의 안정성과 보안을 보장합니다.

### 5가지 핵심 가드

1. **JSON 하드 가드** - LLM 출력 파싱 보장 (재시도 + 리페어 + 폴백)
2. **헬스체크** - 모델 상태 모니터링 (로딩/가용/메모리)
3. **자동 다운시프트** - 장애 시 모드 전환 (creative→balanced→fast)
4. **로그 마스킹** - PII 및 민감 정보 자동 마스킹
5. **모델 해시 검증** - SHA256 해시 기반 무결성 검증

---

## 1. JSON 하드 가드

### 개요

LLM 출력의 JSON 파싱을 보장하는 다층 방어 시스템:
1. **재시도** (백오프 + jitter)
2. **미니 리페어** (일반적인 JSON 오류 자동 수정)
3. **기본 브리프 폴백**

### 파일

**`fragrance_ai/guards/json_guard.py`**

### 사용법

#### 기본 사용

```python
from fragrance_ai.guards.json_guard import JSONGuard, JSONGuardConfig

# Create guard
guard = JSONGuard(JSONGuardConfig(
    max_retries=3,
    initial_backoff=0.5,
    enable_repair=True,
    enable_fallback=True
))

# Parse LLM output
raw_output = llm.generate(prompt)
brief = guard.parse_llm_output(raw_output)
```

#### 재시도 with 생성 함수

```python
def generate_brief():
    """LLM 생성 함수"""
    return llm.generate(prompt)

# Parse with retry
brief = guard.parse_llm_output(
    raw_output="invalid json",
    generator_func=generate_brief
)
```

#### 커스텀 폴백

```python
# Parse with custom fallback
brief = guard.parse_with_guard(
    json_string=raw_output,
    fallback_value={
        "style": "neutral",
        "intensity": 0.5,
        # ...
    }
)
```

### 지원하는 JSON 리페어

1. **코드 블록 제거**: \`\`\`json ... \`\`\`
2. **트레일링 쉼표 제거**: `{"key": "value",}`
3. **단일 따옴표 → 이중 따옴표**: `{'key': 'value'}`
4. **불완전한 JSON 완성**: 누락된 `}`, `]` 추가
5. **JSON 추출**: 텍스트 내 JSON 객체 추출

### 설정

```python
@dataclass
class JSONGuardConfig:
    max_retries: int = 3                # 재시도 횟수
    initial_backoff: float = 0.5        # 초기 백오프 (초)
    max_backoff: float = 5.0            # 최대 백오프 (초)
    backoff_multiplier: float = 2.0     # 백오프 배율
    jitter: bool = True                 # 지터 사용 (±25%)
    enable_repair: bool = True          # 리페어 활성화
    enable_fallback: bool = True        # 폴백 활성화
```

### 예시

```python
# Test: Trailing comma
invalid_json = '{"style": "fresh", "intensity": 0.7,}'
result = guard.parse_with_guard(invalid_json)
# Result: {"style": "fresh", "intensity": 0.7}

# Test: Code block
code_block_json = '''```json
{"style": "floral", "intensity": 0.8}
```'''
result = guard.parse_with_guard(code_block_json)
# Result: {"style": "floral", "intensity": 0.8}

# Test: Incomplete JSON
incomplete = '{"style": "woody", "intensity": 0.6'
result = guard.parse_with_guard(incomplete)
# Result: {"style": "woody", "intensity": 0.6} (with closing brace added)
```

---

## 2. 헬스체크

### 개요

각 LLM 모델(Qwen/Mistral/Llama)의 상태를 실시간으로 모니터링:
- **로딩 상태**: loading/ready/error/unavailable
- **가용성**: 사용 가능 여부
- **메모리 사용량**: MB 및 %

### 파일

- **`fragrance_ai/guards/health_check.py`** - 헬스체커 코어
- **`fragrance_ai/api/health_endpoints.py`** - FastAPI 엔드포인트

### 사용법

#### 1. 헬스체커 초기화

```python
from fragrance_ai.guards.health_check import HealthChecker

# Create health checker
health_checker = HealthChecker()

# Register models (after loading)
health_checker.register_model("qwen", qwen_model)
health_checker.register_model("mistral", mistral_model)
health_checker.register_model("llama", llama_model)
```

#### 2. 모델 헬스 체크

```python
# Check specific model
qwen_health = health_checker.check_model_health("qwen")

print(f"Status: {qwen_health.status.value}")
print(f"Available: {qwen_health.available}")
print(f"Memory: {qwen_health.memory_mb:.0f}MB ({qwen_health.memory_percent:.1f}%)")
```

#### 3. 전체 시스템 헬스 체크

```python
# Check all models
system_health = health_checker.check_all_models()

print(f"Health: {system_health.health_status.value}")
print(f"Available: {system_health.available_models}/{system_health.total_models}")
print(f"System Memory: {system_health.system_memory_percent:.1f}%")
print(f"System CPU: {system_health.system_cpu_percent:.1f}%")
```

#### 4. 헬스 요약

```python
# Get summary
summary = health_checker.get_health_summary()
print(summary)

# Output:
# Health Status: HEALTHY
# Available Models: 3/3
# System Memory: 45.2%
# System CPU: 20.5%
#
# Models:
#   [OK] qwen: ready (Memory: 5120MB, 12.5%)
#   [OK] mistral: ready (Memory: 4800MB, 11.7%)
#   [OK] llama: ready (Memory: 6200MB, 15.1%)
```

### API 엔드포인트

#### 통합 (FastAPI)

```python
from fastapi import FastAPI
from fragrance_ai.api.health_endpoints import health_router, init_health_checker

app = FastAPI(title="Fragrance AI API")

# Initialize health checker with models
health_checker = HealthChecker()
health_checker.register_model("qwen", qwen_model)
health_checker.register_model("mistral", mistral_model)
health_checker.register_model("llama", llama_model)

init_health_checker(health_checker)

# Include health router
app.include_router(health_router)
```

#### 엔드포인트 목록

```bash
# 1. 기본 헬스체크
GET /health
Response: {"status": "healthy", "service": "fragrance-ai"}

# 2. 모든 LLM 모델 헬스
GET /health/llm
Response: {
  "health_status": "healthy",
  "models": {
    "qwen": {...},
    "mistral": {...},
    "llama": {...}
  },
  "system_memory_percent": 45.2,
  "system_cpu_percent": 20.5,
  "available_models": 3,
  "total_models": 3
}

# 3. 특정 모델 헬스
GET /health/llm?model=qwen
Response: {
  "model_name": "qwen",
  "status": "ready",
  "available": true,
  "memory_mb": 5120.5,
  "memory_percent": 12.5,
  "timestamp": 1697123456.789
}

# 4. 헬스 요약
GET /health/llm/summary
Response: {
  "summary": "Health Status: HEALTHY\n...",
  "health_status": "healthy",
  "available_models": 3,
  "total_models": 3
}

# 5. Kubernetes Readiness (모든 모델 ready)
GET /health/readiness
Response: {"ready": true, "available_models": 3, "total_models": 3}

# 6. Kubernetes Liveness (서비스 alive)
GET /health/liveness
Response: {"alive": true, "service": "fragrance-ai"}
```

#### 사용 예시

```bash
# Check all models
curl http://localhost:8000/health/llm | jq .

# Check Qwen only
curl http://localhost:8000/health/llm?model=qwen | jq .

# Readiness probe (Kubernetes)
curl http://localhost:8000/health/readiness
```

---

## 3. 자동 다운시프트

### 개요

장애 시 자동으로 더 빠른 모드로 전환하여 서비스 가용성 보장:

**다운시프트 계층**:
```
creative (Llama 3-8B)
    ↓ (장애 발생)
balanced (Mistral 7B)
    ↓ (장애 발생)
fast (Qwen 2.5-7B)
```

### 파일

**`fragrance_ai/guards/downshift.py`**

### 다운시프트 트리거

1. **모델 사용 불가**: 모델 로드 실패 또는 오류
2. **높은 레이턴시**: 목표치 초과 (creative: 6s, balanced: 4.3s, fast: 3.3s)
3. **높은 에러율**: 최근 10개 요청 중 30% 이상 실패
4. **메모리 압박**: 12GB 또는 85% 이상 사용

### 사용법

#### 1. 다운시프트 관리자 초기화

```python
from fragrance_ai.guards.downshift import DownshiftManager, DownshiftConfig, Mode

# Create manager
manager = DownshiftManager(DownshiftConfig(
    creative_latency_threshold=6.0,      # 4.5s target + 33% margin
    balanced_latency_threshold=4.3,      # 3.2s target + 33% margin
    fast_latency_threshold=3.3,          # 2.5s target + 33% margin
    error_rate_threshold=0.3,            # 30%
    error_window_size=10,                # Last 10 requests
    memory_threshold_mb=12000,           # 12GB
    downshift_cooldown_seconds=60,       # 1 minute cooldown
    auto_recovery_enabled=True,
    recovery_wait_minutes=5
))
```

#### 2. 다운시프트 필요 여부 체크

```python
# Check if downshift needed
should_downshift, reason = manager.should_downshift(
    mode=Mode.CREATIVE,
    latency_ms=7000,  # 7 seconds (high latency)
    error=False,
    memory_mb=None,
    model_unavailable=False
)

if should_downshift:
    print(f"Downshift needed: {reason.value}")
```

#### 3. 다운시프트 실행

```python
if should_downshift:
    new_mode = manager.downshift(
        current_mode=Mode.CREATIVE,
        reason=reason,
        metadata={"latency_ms": 7000}
    )
    print(f"Downshifted to: {new_mode.value}")  # "balanced"
```

#### 4. 자동 복구

```python
# Attempt recovery (after cooldown period)
recovered_mode = manager.attempt_recovery(Mode.BALANCED)

if recovered_mode:
    print(f"Recovered to: {recovered_mode.value}")  # "creative"
```

#### 5. 현재 상태 조회

```python
# Get current mode
current_mode = manager.get_current_mode()
print(f"Current mode: {current_mode.value}")

# Get current model
current_model = manager.get_current_model()
print(f"Current model: {current_model}")  # "mistral", "llama", or "qwen"

# Get status
status = manager.get_status()
print(json.dumps(status, indent=2))
```

### LLM 앙상블 통합

```python
class LLMEnsemble:
    def __init__(self):
        self.downshift_manager = DownshiftManager()
        # ...

    def generate_brief(self, user_text: str, mode: str = "balanced"):
        """브리프 생성 with 자동 다운시프트"""
        start_time = time.time()

        # Get current mode (may be downshifted)
        current_mode = self.downshift_manager.get_current_mode()

        try:
            # Generate
            brief = self._generate(user_text, current_mode.value)

            # Check latency
            latency_ms = (time.time() - start_time) * 1000

            should_downshift, reason = self.downshift_manager.should_downshift(
                mode=current_mode,
                latency_ms=latency_ms,
                error=False
            )

            if should_downshift:
                self.downshift_manager.downshift(current_mode, reason)

            return brief

        except Exception as e:
            logger.error(f"Generation failed: {e}")

            # Check for downshift
            should_downshift, reason = self.downshift_manager.should_downshift(
                mode=current_mode,
                error=True,
                model_unavailable=True
            )

            if should_downshift:
                new_mode = self.downshift_manager.downshift(current_mode, reason)
                # Retry with downshifted mode
                return self._generate(user_text, new_mode.value)

            raise
```

### 설정

```python
@dataclass
class DownshiftConfig:
    # Latency thresholds (seconds)
    creative_latency_threshold: float = 6.0
    balanced_latency_threshold: float = 4.3
    fast_latency_threshold: float = 3.3

    # Error rate thresholds
    error_rate_threshold: float = 0.3  # 30%
    error_window_size: int = 10        # Last 10 requests

    # Memory thresholds
    memory_threshold_mb: float = 12000      # 12GB
    memory_threshold_percent: float = 85.0  # 85%

    # Cooldown (prevent rapid downshifts)
    downshift_cooldown_seconds: int = 60  # 1 minute

    # Auto-recovery
    auto_recovery_enabled: bool = True
    recovery_wait_minutes: int = 5
```

---

## 4. 로그 마스킹

### 개요

로그에서 개인 정보(PII)와 민감 정보를 자동으로 마스킹:
- API keys
- Email addresses
- Phone numbers
- Credit card numbers
- IP addresses
- Passwords
- JWT tokens

### 파일

**`fragrance_ai/guards/log_masking.py`**

### 사용법

#### 1. 로그 마스커 초기화

```python
from fragrance_ai.guards.log_masking import LogMasker

# Create masker (with all default rules)
masker = LogMasker(enable_all_rules=True)
```

#### 2. 텍스트 마스킹

```python
# Mask text
text = "User email: john.doe@example.com, API key: sk_test_abc123def456"
masked = masker.mask(text)
# Result: "User email: ***EMAIL***, API key: ***API_KEY***"
```

#### 3. 딕셔너리 마스킹

```python
# Mask dictionary
data = {
    "user_email": "admin@company.com",
    "api_key": "sk_live_abcdef123456789",
    "phone": "555-1234-5678"
}

masked_data = masker.mask_dict(data)
# Result: {
#     "user_email": "***EMAIL***",
#     "api_key": "***API_KEY***",
#     "phone": "***PHONE***"
# }
```

#### 4. 커스텀 규칙 추가

```python
import re

# Add custom masking rule
masker.add_rule(
    name="custom_id",
    pattern=re.compile(r'ID-\d{8}'),
    replacement="***ID***",
    description="Custom ID format"
)
```

### 로거 통합

#### MaskingLogHandler 사용

```python
import logging
from fragrance_ai.guards.log_masking import MaskingLogHandler, LogMasker

# Create masker
masker = LogMasker()

# Create logger
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)

# Create masking handler
console_handler = logging.StreamHandler()
masking_handler = MaskingLogHandler(console_handler, masker)
logger.addHandler(masking_handler)

# Use logger (all messages automatically masked)
logger.info("User logged in: email=user@example.com, ip=192.168.1.1")
# Output: "User logged in: email=***EMAIL***, ip=***IP***"

logger.info("API request with key: api_key=sk_test_abc123")
# Output: "API request with key: api_key=***API_KEY***"
```

#### 앱 전체 적용

```python
# app/logging_config.py

import logging.config
from fragrance_ai.guards.log_masking import MaskingLogHandler, LogMasker

def setup_logging():
    """로깅 설정 (마스킹 포함)"""

    # Create masker
    masker = LogMasker(enable_all_rules=True)

    # Get root logger
    root_logger = logging.getLogger()

    # Replace all handlers with masking handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        masking_handler = MaskingLogHandler(handler, masker)
        root_logger.addHandler(masking_handler)

# Call on app startup
setup_logging()
```

### 지원하는 마스킹 규칙

| 규칙 | 패턴 | 대체 문자열 |
|------|------|-------------|
| API Key | `sk_test_*`, `api_key=*` | `***API_KEY***` |
| Email | `user@domain.com` | `***EMAIL***` |
| Phone | `+1-555-123-4567`, `010-1234-5678` | `***PHONE***` |
| Credit Card | `4532-1234-5678-9010` | `***CARD***` |
| IP Address | `192.168.1.1` | `***IP***` |
| Password | `password="secret"` | `password=***PASSWORD***` |
| JWT Token | `eyJhbG...` | `***JWT***` |
| Bearer Token | `Bearer abc123...` | `Bearer ***TOKEN***` |
| SSH Key | `-----BEGIN PRIVATE KEY-----` | `***SSH_PRIVATE_KEY***` |
| Korean RRN | `901234-1234567` | `***RRN***` |
| SSN | `123-45-6789` | `***SSN***` |

---

## 5. 모델 해시 검증

### 개요

SHA256 해시를 사용한 모델 파일 무결성 검증:
- 모델 파일이 변조되지 않았는지 확인
- 신뢰할 수 있는 해시와 비교
- 검증 실패 시 경고

### 파일

**`fragrance_ai/guards/model_verification.py`**

### 사용법

#### 1. 모델 검증기 초기화

```python
from fragrance_ai.guards.model_verification import ModelVerifier

# Create verifier
verifier = ModelVerifier(trusted_hashes_path="model_hashes.json")
```

#### 2. 신뢰할 수 있는 해시 등록

```python
# Add trusted hashes
verifier.add_trusted_hash(
    "qwen-2.5-7b",
    "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2"
)

verifier.add_trusted_hash(
    "mistral-7b",
    "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2a3"
)

# Save to file
verifier.save_trusted_hashes("model_hashes.json")
```

#### 3. 모델 검증

```python
# Verify single model
hash_info = verifier.verify_model(
    model_name="qwen-2.5-7b",
    model_path="models/qwen-2.5-7b.bin"
)

print(f"Verified: {hash_info.verified}")
print(f"Hash: {hash_info.sha256_hash}")
print(f"Size: {hash_info.file_size_bytes / (1024**3):.2f} GB")
```

#### 4. 모든 모델 검증

```python
# Verify all models
model_paths = {
    "qwen-2.5-7b": "models/qwen-2.5-7b.bin",
    "mistral-7b": "models/mistral-7b.bin",
    "llama-3-8b": "models/llama-3-8b.bin"
}

results = verifier.verify_all_models(model_paths)

for model_name, hash_info in results.items():
    if hash_info.verified:
        print(f"[OK] {model_name}")
    else:
        print(f"[FAIL] {model_name}: {hash_info.error_message}")
```

#### 5. 검증 리포트 생성

```python
# Generate report
report = verifier.generate_hash_report()
print(report)

# Output:
# ================================================================================
# Model Hash Verification Report
# ================================================================================
#
# [OK] qwen-2.5-7b
#   Path:      models/qwen-2.5-7b.bin
#   Size:      6.50 GB
#   Hash:      a1b2c3d4e5f6g7h8...
#   Verified:  2025-10-12T10:30:45Z
#
# [FAIL] mistral-7b
#   Path:      models/mistral-7b.bin
#   Size:      6.80 GB
#   Hash:      xxxxxxxxxxxx...
#   Verified:  2025-10-12T10:30:46Z
#   Error:     Hash mismatch
#
# ================================================================================
```

### 앱 시작 시 검증

```python
# app/startup.py

from fragrance_ai.guards.model_verification import ModelVerifier

def verify_models_on_startup():
    """앱 시작 시 모델 검증"""

    verifier = ModelVerifier(trusted_hashes_path="model_hashes.json")

    model_paths = {
        "qwen-2.5-7b": "models/qwen-2.5-7b.bin",
        "mistral-7b": "models/mistral-7b.bin",
        "llama-3-8b": "models/llama-3-8b.bin"
    }

    logger.info("Verifying model integrity...")
    results = verifier.verify_all_models(model_paths)

    # Check all verified
    all_verified = all(info.verified for info in results.values())

    if not all_verified:
        # Generate report
        report = verifier.generate_hash_report()
        logger.error(f"Model verification failed:\n{report}")

        # Optionally: halt startup
        raise RuntimeError("Model verification failed. Halting startup.")

    logger.info("All models verified successfully")

# Call on app startup
verify_models_on_startup()
```

### Quick Hash Tool

```python
# Compute hash of a file
from fragrance_ai.guards.model_verification import quick_hash

hash_value = quick_hash("models/qwen-2.5-7b.bin")
print(f"SHA256: {hash_value}")
```

---

## 통합 예시

### 전체 가드 적용

```python
# app/main.py

from fastapi import FastAPI
import logging
from fragrance_ai.guards.json_guard import JSONGuard
from fragrance_ai.guards.health_check import HealthChecker
from fragrance_ai.guards.downshift import DownshiftManager
from fragrance_ai.guards.log_masking import MaskingLogHandler, LogMasker
from fragrance_ai.guards.model_verification import ModelVerifier
from fragrance_ai.api.health_endpoints import health_router, init_health_checker

app = FastAPI(title="Fragrance AI API")

# ============================================================================
# 1. 로그 마스킹 설정
# ============================================================================

def setup_logging():
    """로깅 설정 (마스킹 포함)"""
    masker = LogMasker(enable_all_rules=True)
    root_logger = logging.getLogger()

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        masking_handler = MaskingLogHandler(handler, masker)
        root_logger.addHandler(masking_handler)

setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# 2. 모델 해시 검증
# ============================================================================

def verify_models():
    """모델 무결성 검증"""
    verifier = ModelVerifier(trusted_hashes_path="model_hashes.json")

    model_paths = {
        "qwen-2.5-7b": "models/qwen-2.5-7b.bin",
        "mistral-7b": "models/mistral-7b.bin",
        "llama-3-8b": "models/llama-3-8b.bin"
    }

    results = verifier.verify_all_models(model_paths)

    if not all(info.verified for info in results.values()):
        report = verifier.generate_hash_report()
        logger.error(f"Model verification failed:\n{report}")
        raise RuntimeError("Model verification failed")

    logger.info("All models verified successfully")

verify_models()

# ============================================================================
# 3. 헬스체크 설정
# ============================================================================

health_checker = HealthChecker()

# Register models (after loading)
# health_checker.register_model("qwen", qwen_model)
# health_checker.register_model("mistral", mistral_model)
# health_checker.register_model("llama", llama_model)

init_health_checker(health_checker)
app.include_router(health_router)

# ============================================================================
# 4. 다운시프트 관리자 설정
# ============================================================================

downshift_manager = DownshiftManager()

# ============================================================================
# 5. JSON 가드 설정
# ============================================================================

json_guard = JSONGuard()

# ============================================================================
# API 엔드포인트 (모든 가드 적용)
# ============================================================================

@app.post("/dna/create")
async def create_dna(request: DNACreateRequest):
    """DNA 생성 (모든 가드 적용)"""

    # Get current mode (downshifted if needed)
    current_mode = downshift_manager.get_current_mode()

    try:
        # Generate with LLM
        raw_output = llm_ensemble.generate(request.user_text, mode=current_mode.value)

        # Parse with JSON guard (retry + repair + fallback)
        brief = json_guard.parse_llm_output(raw_output)

        return brief

    except Exception as e:
        logger.error(f"DNA creation failed: {e}")

        # Check for downshift
        should_downshift, reason = downshift_manager.should_downshift(
            mode=current_mode,
            error=True
        )

        if should_downshift:
            downshift_manager.downshift(current_mode, reason)

        raise
```

---

## 모니터링

### Prometheus 메트릭

```python
# fragrance_ai/monitoring/stability_metrics.py

from prometheus_client import Counter, Histogram, Gauge

# JSON Guard metrics
json_guard_repairs = Counter(
    'json_guard_repairs_total',
    'Total number of JSON repairs',
    ['repair_type']
)

json_guard_fallbacks = Counter(
    'json_guard_fallbacks_total',
    'Total number of fallback uses'
)

# Downshift metrics
downshift_events = Counter(
    'downshift_events_total',
    'Total number of downshift events',
    ['from_mode', 'to_mode', 'reason']
)

# Health metrics (already defined in kpi_metrics.py)
# - model_status
# - model_memory_usage

# Model verification metrics
model_verification_status = Gauge(
    'model_verification_status',
    'Model verification status (1=verified, 0=failed)',
    ['model_name']
)
```

### Grafana 대시보드

추가 패널:
- JSON Guard 리페어 횟수
- JSON Guard 폴백 사용률
- 다운시프트 이벤트 타임라인
- 모델 검증 상태

---

## 테스트

### 테스트 파일

```bash
# JSON Guard 테스트
pytest tests/test_json_guard.py

# Health Check 테스트
pytest tests/test_health_check.py

# Downshift 테스트
pytest tests/test_downshift.py

# Log Masking 테스트
pytest tests/test_log_masking.py

# Model Verification 테스트
pytest tests/test_model_verification.py
```

---

## 요약

### 5가지 안정성 가드

✅ **JSON 하드 가드** - 재시도(3회, 백오프+jitter) → 리페어(6가지 규칙) → 폴백

✅ **헬스체크** - /health/llm?model=qwen|mistral|llama → 로딩/가용/메모리 OK

✅ **자동 다운시프트** - creative→balanced→fast (레이턴시/에러율/메모리 기반)

✅ **로그 마스킹** - PII/Key 자동 마스킹 (11가지 규칙)

✅ **모델 해시 검증** - SHA256 해시 검증 OK

### 프로덕션 배포 체크리스트

- [ ] JSON 가드 설정 (max_retries, backoff)
- [ ] 신뢰 해시 파일 생성 (`model_hashes.json`)
- [ ] 모델 검증 통합 (앱 시작 시)
- [ ] 헬스체크 엔드포인트 활성화
- [ ] 로그 마스킹 설정 (모든 로거)
- [ ] 다운시프트 설정 (임계값)
- [ ] Prometheus 메트릭 노출
- [ ] Grafana 대시보드 import

---

**작성일**: 2025-10-12
**버전**: 1.0
**작성자**: Claude Code (Fragrance AI Team)
