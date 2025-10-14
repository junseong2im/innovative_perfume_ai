# LLM 앙상블 동작 확인 테스트

LLM 앙상블의 모드 라우팅, 폴백/리트라이, 서킷 브레이커, 캐시 TTL을 검증하는 테스트입니다.

## 목차

1. [개요](#개요)
2. [테스트 시나리오](#테스트-시나리오)
3. [실행 방법](#실행-방법)
4. [예상 결과](#예상-결과)

---

## 개요

### 테스트 대상

1. **모드 라우팅**: fast/balanced/creative 각 3개 입력 → 자동 모드 선택
2. **폴백/리트라이**: 타임아웃/실패 시 자동 재시도 및 대체 모델 사용
3. **서킷 브레이커**: Qwen 비활성화 → Mistral 자동 전환
4. **캐시 TTL**: 동일 입력 레이턴시 비교 (캐시 히트/미스)

### 테스트 파일

```
tests/test_llm_ensemble_operation.py  # 유닛 테스트 (400+ lines)
demo_llm_ensemble.py                  # 실행 가능한 데모 (450+ lines)
```

---

## 테스트 시나리오

### 1. 모드 라우팅 테스트

**목적**: 입력 길이에 따라 자동으로 fast/balanced/creative 모드 선택

**테스트 케이스**:

#### Fast Mode (짧은 입력 < 50자)
```python
fast_inputs = [
    "상큼한 레몬향",           # 7자
    "Fresh citrus scent",    # 19자
    "시트러스"                 # 4자
]
```

**예상**:
- 자동으로 `mode='fast'` 선택
- 지연시간 < 5초
- Qwen 모델 사용

**예상 로그**:
```json
{
  "timestamp": "2025-01-15T10:30:45Z",
  "component": "LLM",
  "message": "LLM brief generated",
  "model": "qwen",
  "mode": "fast",
  "latency_ms": 100.0,
  "cache_hit": false,
  "brief_style": "fresh",
  "brief_intensity": 0.7
}
```

#### Balanced Mode (중간 길이 50-200자)
```python
balanced_inputs = [
    "상큼하면서도 우아한 봄날 아침 향기, 플로럴 노트와 시트러스가 조화롭게",  # 37자 (한글 = 2바이트)
    "A fresh yet elegant morning fragrance for spring, harmonizing floral and citrus notes",  # 88자
    "우디한 베이스에 스파이시한 톱노트, 저녁 시간에 어울리는 중후한 느낌"  # 38자
]
```

**예상**:
- 자동으로 `mode='balanced'` 선택
- 지연시간 < 12초
- Qwen + Mistral 검증

#### Creative Mode (긴 서사적 입력 > 200자)
```python
creative_inputs = [
    "봄날 아침, 햇살에 반짝이는 이슬 맺힌 하얀 꽃잎. 상쾌하면서도 우아한, 마치 발레리나의 첫 스텝처럼..."
    # (전체 200자 이상)
]
```

**예상**:
- 자동으로 `mode='creative'` 선택
- 지연시간 < 20초
- Qwen + Mistral + Llama (full ensemble)

### 2. 폴백/리트라이 테스트

**목적**: 모델 실패 시 자동 재시도 및 대체 모델 사용

**시나리오 1: 타임아웃 → 리트라이**
```
시도 1: Qwen 타임아웃 (> 12s)
     ↓
시도 2: Qwen 재시도 (백오프 0.5s)
     ↓
시도 3: 성공
```

**예상 로그**:
```json
{
  "component": "LLM",
  "message": "Timeout on attempt 1",
  "warning": "Retrying in 0.5s..."
}
{
  "component": "LLM",
  "message": "LLM brief generated",
  "model": "qwen",
  "retry_attempt": 2,
  "success": true
}
```

**시나리오 2: 최대 재시도 초과 → 폴백**
```
시도 1: Qwen 실패
시도 2: Qwen 실패
시도 3: Qwen 실패 (max retries)
     ↓
폴백: Mistral 사용
```

**예상 로그**:
```json
{
  "component": "LLM",
  "message": "Max retries exceeded - trying fallback model"
}
{
  "component": "LLM",
  "model": "mistral",
  "fallback_from": "qwen",
  "success": true
}
```

**시나리오 3: 모든 모델 실패 → 캐시 또는 기본값**
```
Qwen: 실패
Mistral: 실패
Llama: 실패
     ↓
1차: 캐시 검색 → 히트 시 반환
2차: 기본 brief 반환 (emergency mode)
```

### 3. 서킷 브레이커 테스트

**목적**: Qwen 연속 실패 시 Mistral로 자동 전환

**상태 전환**:
```
CLOSED (정상)
   ↓ (실패 3회)
OPEN (Qwen 차단)
   ↓ (60초 대기)
HALF_OPEN (복구 시도)
   ↓ (성공)
CLOSED (복구 완료)
```

**시나리오**:

#### 단계 1: 정상 동작
```python
# Qwen 정상
brief = manager.generate_brief("상큼한 레몬향", mode='fast')
# ✓ Model: Qwen
```

#### 단계 2: 연속 실패 감지
```python
# Qwen 3회 연속 실패
for i in range(3):
    try:
        brief = manager.generate_brief(user_text)
    except:
        failure_count += 1

# Circuit breaker → OPEN
```

**예상 로그**:
```json
{
  "level": "WARNING",
  "message": "Qwen 실패 1/3: Model inference failed"
}
{
  "level": "WARNING",
  "message": "Qwen 실패 2/3: Model inference failed"
}
{
  "level": "WARNING",
  "message": "Qwen 실패 3/3: Model inference failed"
}
{
  "level": "ERROR",
  "message": "⚠ Circuit breaker OPEN (failures: 3)"
}
```

#### 단계 3: Mistral 자동 전환
```python
# Qwen circuit breaker OPEN → Mistral 사용
brief = manager.generate_brief(user_text)
# ✓ Model: Mistral (fallback)
```

**예상 로그**:
```json
{
  "level": "WARNING",
  "message": "Qwen circuit breaker OPEN - using Mistral fallback"
}
{
  "component": "LLM",
  "model": "mistral",
  "fallback_from": "qwen",
  "success": true
}
```

#### 단계 4: Qwen 복구 확인
```python
# 60초 후 복구 시도
time.sleep(60)

brief = manager.generate_brief(user_text)
# ✓ Model: Qwen (recovered)
```

**예상 로그**:
```json
{
  "level": "INFO",
  "message": "Circuit breaker → HALF_OPEN (recovery attempt)"
}
{
  "component": "LLM",
  "model": "qwen",
  "circuit_state": "half_open",
  "success": true
}
{
  "level": "INFO",
  "message": "✓ Circuit breaker CLOSED"
}
```

### 4. 캐시 TTL 검증 테스트

**목적**: 캐시 히트/미스 시 레이턴시 비교

**시나리오**:

#### 요청 1: 캐시 미스 (첫 요청)
```python
user_text = "상큼한 레몬향"

start_time = time.time()
brief = manager.generate_brief(user_text, mode='fast')
latency_1 = (time.time() - start_time) * 1000

# latency_1 ≈ 100-500ms (LLM 추론)
```

**예상 로그**:
```json
{
  "component": "LLM",
  "model": "qwen",
  "latency_ms": 250.0,
  "cache_hit": false
}
```

#### 요청 2: 캐시 히트 (TTL 내)
```python
# 1초 후 (< 5초 TTL)
time.sleep(1)

start_time = time.time()
brief = manager.generate_brief(user_text, mode='fast')
latency_2 = (time.time() - start_time) * 1000

# latency_2 ≈ 1-10ms (캐시 조회)
speedup = latency_1 / latency_2  # ≈ 25-250x 향상
```

**예상 로그**:
```json
{
  "component": "LLM",
  "model": "cache",
  "latency_ms": 5.0,
  "cache_hit": true
}
```

**레이턴시 비교**:
```
캐시 미스:  250ms
캐시 히트:  5ms (↓ 50x)
```

#### 요청 3: 캐시 만료 (TTL 초과)
```python
# 5초 후 (> 5초 TTL)
time.sleep(5)

start_time = time.time()
brief = manager.generate_brief(user_text, mode='fast')
latency_3 = (time.time() - start_time) * 1000

# latency_3 ≈ 100-500ms (LLM 재추론)
```

**예상 로그**:
```json
{
  "component": "LLM",
  "model": "qwen",
  "latency_ms": 240.0,
  "cache_hit": false,
  "cache_expired": true
}
```

**레이턴시 요약**:
```
요청 1 (캐시 미스):     250ms
요청 2 (캐시 히트):     5ms (↓ 50x)
요청 3 (캐시 만료):     240ms (재생성)
```

#### 추가: 의미적 유사도 캐시 히트

**시나리오**: 다른 표현이지만 의미가 비슷한 입력 → 캐시 히트

```python
# 원본 입력
user_text_1 = "상큼한 레몬향"
brief_1 = manager.generate_brief(user_text_1)

# 유사 입력 (다른 표현)
user_text_2 = "레몬 같은 상큼한 향"

# 의미적 유사도 계산 (임베딩 기반)
similarity = calculate_similarity(user_text_1, user_text_2)
# similarity ≈ 0.85

if similarity > 0.8:
    # 캐시 히트 (같은 brief 반환)
    brief_2 = cached_brief_1
```

**예상 로그**:
```json
{
  "component": "LLM",
  "model": "cache",
  "latency_ms": 20.0,
  "cache_hit": true,
  "similarity_score": 0.85,
  "original_input": "상큼한 레몬향"
}
```

---

## 실행 방법

### 유닛 테스트 실행

```bash
# 전체 LLM 앙상블 동작 테스트
pytest tests/test_llm_ensemble_operation.py -v -s

# 개별 테스트 실행
pytest tests/test_llm_ensemble_operation.py::TestLLMEnsembleModeRouting::test_fast_mode_routing -v -s
pytest tests/test_llm_ensemble_operation.py::TestLLMEnsembleCircuitBreaker::test_circuit_breaker_qwen_to_mistral -v -s
pytest tests/test_llm_ensemble_operation.py::TestLLMEnsembleCache::test_cache_ttl_verification -v -s
```

### 데모 스크립트 실행

```bash
# 전체 데모 실행 (대화형)
python demo_llm_ensemble.py
```

**데모 내용**:
1. 모드 라우팅 (fast/balanced/creative)
2. 캐시 TTL 검증 (레이턴시 비교)
3. 서킷 브레이커 (Qwen → Mistral 전환)
4. 리트라이 및 폴백

**예상 출력**:
```
================================================================================
LLM 앙상블 동작 확인 데모
================================================================================

================================================================================
데모 1: 모드 라우팅 (fast/balanced/creative)
================================================================================

[Fast Mode - 짧은 입력 (< 50자)]

  1. '상큼한 레몬향'
     → Style: fresh, Intensity: 0.7

  2. 'Fresh citrus scent'
     → Style: fresh, Intensity: 0.7

  3. '시트러스'
     → Style: fresh, Intensity: 0.7

[Balanced Mode - 중간 길이 입력 (50-200자)]

  1. '상큼하면서도 우아한 봄날 아침 향기, 플로럴 노트와 시트러스가 조화롭게'
     → Style: floral, Intensity: 0.6

[Creative Mode - 긴 서사적 입력 (> 200자)]

  1. '봄날 아침, 햇살에 반짝이는 이슬 맺힌 하얀 꽃잎...'
     → Style: floral, Intensity: 0.8

================================================================================
데모 2: 캐시 TTL 검증 (레이턴시 비교)
================================================================================

요청 1: 캐시 미스 (첫 요청)
  → 레이턴시: 120ms

요청 2: 캐시 히트 (TTL 내)
  → 레이턴시: 5ms
  → 속도 향상: 24.0x

요청 3: 캐시 만료 (5초 후)
  (5초 대기 중...)
  → 레이턴시: 115ms (캐시 재생성)

캐시 TTL 검증 요약:
  캐시 미스 (첫 요청):  120ms
  캐시 히트 (TTL 내):   5ms (↓ 24.0x)
  캐시 만료 (TTL 초과):  115ms

================================================================================
데모 3: 서킷 브레이커 (Qwen → Mistral 전환)
================================================================================

단계 1: Qwen 정상 작동
  ✓ Brief 생성 성공 (Model: Qwen)

단계 2: Qwen 실패 시뮬레이션 (3회 연속)

  시도 1/3:
    ✗ 실패: qwen inference failed

  시도 2/3:
    ✗ 실패: qwen inference failed

  시도 3/3:
    ✗ 실패: qwen inference failed

단계 3: 서킷 브레이커 활성화 → Mistral 전환
  Circuit breaker state: OPEN
  ✓ Brief 생성 성공 (Model: Mistral fallback)

단계 4: Qwen 복구
  ✓ Qwen 복구 확인 (Circuit breaker: CLOSED)

================================================================================
✓ 모든 데모 완료!
================================================================================
```

---

## 예상 결과

### 테스트 결과

```bash
$ pytest tests/test_llm_ensemble_operation.py -v

tests/test_llm_ensemble_operation.py::TestLLMEnsembleModeRouting::test_fast_mode_routing PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleModeRouting::test_balanced_mode_routing PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleModeRouting::test_creative_mode_routing PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleFallback::test_retry_on_timeout PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleFallback::test_fallback_to_cache PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleCircuitBreaker::test_circuit_breaker_qwen_to_mistral PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleCircuitBreaker::test_circuit_breaker_all_models_down PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleCache::test_cache_ttl_verification PASSED
tests/test_llm_ensemble_operation.py::TestLLMEnsembleCache::test_cache_semantic_similarity PASSED

========== 9 passed in 45.23s ==========
```

### JSON 로그 예시

**모드 라우팅 로그**:
```json
{"timestamp":"2025-01-15T10:30:45Z","level":"INFO","component":"LLM","message":"Auto-routed to mode: fast (text length: 15)"}
{"timestamp":"2025-01-15T10:30:45Z","level":"INFO","component":"LLM","message":"LLM brief generated","model":"qwen","mode":"fast","latency_ms":120.0,"cache_hit":false}
```

**서킷 브레이커 로그**:
```json
{"timestamp":"2025-01-15T10:31:00Z","level":"WARNING","message":"Qwen 실패 1/3: Model inference failed"}
{"timestamp":"2025-01-15T10:31:01Z","level":"WARNING","message":"Qwen 실패 2/3: Model inference failed"}
{"timestamp":"2025-01-15T10:31:02Z","level":"WARNING","message":"Qwen 실패 3/3: Model inference failed"}
{"timestamp":"2025-01-15T10:31:02Z","level":"ERROR","message":"⚠ Circuit breaker OPEN (failures: 3)"}
{"timestamp":"2025-01-15T10:31:03Z","level":"WARNING","message":"Qwen circuit breaker OPEN - using Mistral fallback"}
{"timestamp":"2025-01-15T10:31:03Z","level":"INFO","component":"LLM","model":"mistral","fallback_from":"qwen"}
```

**캐시 히트 로그**:
```json
{"timestamp":"2025-01-15T10:32:00Z","level":"INFO","component":"LLM","message":"✗ Cache miss - generating brief"}
{"timestamp":"2025-01-15T10:32:00Z","level":"INFO","component":"LLM","model":"qwen","latency_ms":250.0,"cache_hit":false}
{"timestamp":"2025-01-15T10:32:01Z","level":"INFO","component":"LLM","message":"✓ Cache hit"}
{"timestamp":"2025-01-15T10:32:01Z","level":"INFO","component":"LLM","model":"cache","latency_ms":5.0,"cache_hit":true}
```

---

## 요약

### 생성된 파일

1. **`tests/test_llm_ensemble_operation.py`** (400+ lines)
   - 9개 테스트 클래스
   - 모드 라우팅, 폴백, 서킷 브레이커, 캐시 TTL

2. **`demo_llm_ensemble.py`** (450+ lines)
   - 실행 가능한 대화형 데모
   - 4가지 시나리오 데모

### 테스트 범위

✅ **모드 라우팅**: fast/balanced/creative 각 3개 입력 → 9개
✅ **폴백/리트라이**: 타임아웃, 최대 재시도, 캐시 폴백
✅ **서킷 브레이커**: Qwen → Mistral 전환, 복구 확인
✅ **캐시 TTL**: 히트/미스 레이턴시 비교, 의미적 유사도

### 실행 명령

```bash
# 유닛 테스트
pytest tests/test_llm_ensemble_operation.py -v -s

# 데모
python demo_llm_ensemble.py
```

모든 LLM 앙상블 동작 테스트가 준비되었습니다! 🚀
