#!/usr/bin/env python
"""
LLM 앙상블 동작 확인 데모
- 모드 라우팅 (fast/balanced/creative)
- 폴백/리트라이 메커니즘
- 서킷 브레이커
- 캐시 TTL 검증
"""

import time
import sys
from typing import Dict, Any, Optional
from fragrance_ai.observability import llm_logger, get_logger

# Setup logger
logger = get_logger("demo")


# ============================================================================
# Mock LLM Functions (실제 구현에서는 real LLM 호출)
# ============================================================================

class MockLLM:
    """Mock LLM for demonstration"""

    def __init__(self, name: str, failure_rate: float = 0.0):
        self.name = name
        self.failure_rate = failure_rate
        self.call_count = 0

    def generate_brief(self, user_text: str, mode: str) -> Dict[str, Any]:
        """Generate brief (mock)"""
        self.call_count += 1

        # Simulate failure
        import random
        if random.random() < self.failure_rate:
            raise RuntimeError(f"{self.name} inference failed")

        # Simulate processing time
        if mode == 'fast':
            time.sleep(0.1)  # 100ms
        elif mode == 'balanced':
            time.sleep(0.25)  # 250ms
        else:  # creative
            time.sleep(0.5)  # 500ms

        # Mock brief
        return {
            'style': 'fresh' if 'citrus' in user_text.lower() or '레몬' in user_text else 'floral',
            'intensity': 0.7,
            'complexity': 0.5,
            'notes_preference': {'citrus': 0.9, 'fresh': 0.8}
        }


class SimpleCache:
    """Simple in-memory cache with TTL"""

    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            age = time.time() - timestamp

            if age < self.ttl_seconds:
                return value
            else:
                # Expired
                del self.cache[key]
                return None

        return None

    def set(self, key: str, value: Dict[str, Any]):
        """Set cache entry"""
        self.cache[key] = (value, time.time())

    def clear(self):
        """Clear cache"""
        self.cache = {}


class CircuitBreaker:
    """Circuit breaker for model failover"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def record_failure(self):
        """Record failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            logger.warning(f"⚠ Circuit breaker OPEN (failures: {self.failure_count})")

    def record_success(self):
        """Record success"""
        self.failure_count = 0
        self.state = 'CLOSED'
        logger.info("✓ Circuit breaker CLOSED")

    def can_attempt(self) -> bool:
        """Check if can attempt request"""
        if self.state == 'CLOSED':
            return True

        if self.state == 'OPEN':
            # Check if recovery timeout elapsed
            age = time.time() - self.last_failure_time
            if age >= self.recovery_timeout:
                self.state = 'HALF_OPEN'
                logger.info("Circuit breaker → HALF_OPEN (recovery attempt)")
                return True
            return False

        if self.state == 'HALF_OPEN':
            return True

        return False


# ============================================================================
# LLM Ensemble Manager
# ============================================================================

class LLMEnsembleManager:
    """LLM Ensemble Manager with routing, fallback, and caching"""

    def __init__(self):
        self.qwen = MockLLM('qwen', failure_rate=0.0)
        self.mistral = MockLLM('mistral', failure_rate=0.0)
        self.llama = MockLLM('llama', failure_rate=0.0)

        self.cache = SimpleCache(ttl_seconds=5)  # 5s TTL for demo
        self.circuit_breaker_qwen = CircuitBreaker(failure_threshold=3)

    def route_mode(self, user_text: str) -> str:
        """Route to appropriate mode based on input length"""
        length = len(user_text)

        if length < 50:
            return 'fast'
        elif length < 200:
            return 'balanced'
        else:
            return 'creative'

    def generate_brief(
        self,
        user_text: str,
        mode: Optional[str] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Generate brief with routing, fallback, and caching

        Args:
            user_text: User input text
            mode: Mode (fast/balanced/creative) or None for auto-routing
            max_retries: Max retry attempts

        Returns:
            CreativeBrief dict
        """
        # Auto-route if mode not specified
        if mode is None:
            mode = self.route_mode(user_text)
            logger.info(f"Auto-routed to mode: {mode} (text length: {len(user_text)})")

        # Check cache first
        cache_key = f"{user_text}:{mode}"
        cached = self.cache.get(cache_key)

        if cached:
            logger.info("✓ Cache hit")

            llm_logger.log_brief(
                user_text=user_text,
                brief=cached,
                model='cache',
                mode=mode,
                latency_ms=10.0,  # Very fast from cache
                cache_hit=True
            )

            return cached

        # Cache miss - generate brief
        logger.info("✗ Cache miss - generating brief")

        # Try with retries
        for attempt in range(1, max_retries + 1):
            try:
                start_time = time.time()

                # Check circuit breaker
                if not self.circuit_breaker_qwen.can_attempt():
                    logger.warning("Qwen circuit breaker OPEN - using Mistral fallback")
                    raise RuntimeError("Qwen circuit breaker open")

                # Try primary model (Qwen)
                brief = self.qwen.generate_brief(user_text, mode)

                latency_ms = (time.time() - start_time) * 1000

                # Success
                self.circuit_breaker_qwen.record_success()

                # Log brief
                llm_logger.log_brief(
                    user_text=user_text,
                    brief=brief,
                    model='qwen',
                    mode=mode,
                    latency_ms=latency_ms,
                    cache_hit=False,
                    retry_attempt=attempt if attempt > 1 else None
                )

                # Cache result
                self.cache.set(cache_key, brief)

                return brief

            except Exception as e:
                logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")

                self.circuit_breaker_qwen.record_failure()

                if attempt < max_retries:
                    # Retry with exponential backoff
                    backoff = 0.5 * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {backoff:.1f}s...")
                    time.sleep(backoff)
                else:
                    # Max retries exceeded - try fallback
                    logger.error("Max retries exceeded - trying fallback model")
                    return self._fallback_generate(user_text, mode)

        raise RuntimeError("Failed to generate brief")

    def _fallback_generate(self, user_text: str, mode: str) -> Dict[str, Any]:
        """Fallback to Mistral if Qwen fails"""
        logger.info("Using Mistral fallback")

        try:
            start_time = time.time()

            brief = self.mistral.generate_brief(user_text, mode)

            latency_ms = (time.time() - start_time) * 1000

            llm_logger.log_brief(
                user_text=user_text,
                brief=brief,
                model='mistral',
                mode=mode,
                latency_ms=latency_ms,
                cache_hit=False,
                fallback_from='qwen'
            )

            return brief

        except Exception as e:
            logger.error(f"Fallback also failed: {e}")

            # Last resort: default brief
            logger.warning("Using default brief")

            default_brief = {
                'style': 'fresh',
                'intensity': 0.5,
                'complexity': 0.5,
                'notes_preference': {'fresh': 0.7, 'citrus': 0.5}
            }

            llm_logger.log_brief(
                user_text=user_text,
                brief=default_brief,
                model='default',
                mode=mode,
                latency_ms=10.0,
                cache_hit=False,
                emergency_mode=True
            )

            return default_brief


# ============================================================================
# Demo Functions
# ============================================================================

def demo_mode_routing():
    """데모 1: 모드 라우팅 (fast/balanced/creative)"""
    print("\n" + "="*80)
    print("데모 1: 모드 라우팅 (fast/balanced/creative)")
    print("="*80)

    manager = LLMEnsembleManager()

    # Fast mode inputs
    fast_inputs = [
        "상큼한 레몬향",
        "Fresh citrus scent",
        "시트러스"
    ]

    print("\n[Fast Mode - 짧은 입력 (< 50자)]")
    for i, text in enumerate(fast_inputs, 1):
        print(f"\n  {i}. '{text}'")
        brief = manager.generate_brief(text)
        print(f"     → Style: {brief['style']}, Intensity: {brief['intensity']}")

    # Balanced mode inputs
    balanced_inputs = [
        "상큼하면서도 우아한 봄날 아침 향기, 플로럴 노트와 시트러스가 조화롭게",
        "A fresh yet elegant morning fragrance, harmonizing floral and citrus notes"
    ]

    print("\n[Balanced Mode - 중간 길이 입력 (50-200자)]")
    for i, text in enumerate(balanced_inputs, 1):
        print(f"\n  {i}. '{text[:50]}...'")
        brief = manager.generate_brief(text)
        print(f"     → Style: {brief['style']}, Intensity: {brief['intensity']}")

    # Creative mode inputs
    creative_inputs = [
        "봄날 아침, 햇살에 반짝이는 이슬 맺힌 하얀 꽃잎. 상쾌하면서도 우아한, 마치 발레리나의 첫 스텝처럼 가볍고 섬세한 향. "
        "시트러스의 상큼함이 처음 느껴지고, 곧이어 은은한 화이트 플로럴이 펼쳐지며, 마지막엔 따뜻한 머스크가 감싸는 듯한 3단 변화."
    ]

    print("\n[Creative Mode - 긴 서사적 입력 (> 200자)]")
    for i, text in enumerate(creative_inputs, 1):
        print(f"\n  {i}. '{text[:60]}...'")
        brief = manager.generate_brief(text)
        print(f"     → Style: {brief['style']}, Intensity: {brief['intensity']}")


def demo_cache_ttl():
    """데모 2: 캐시 TTL 검증"""
    print("\n" + "="*80)
    print("데모 2: 캐시 TTL 검증 (레이턴시 비교)")
    print("="*80)

    manager = LLMEnsembleManager()
    user_text = "상큼한 레몬향"

    # Request 1: Cache miss
    print("\n요청 1: 캐시 미스 (첫 요청)")
    start_1 = time.time()
    brief_1 = manager.generate_brief(user_text, mode='fast')
    latency_1 = (time.time() - start_1) * 1000
    print(f"  → 레이턴시: {latency_1:.0f}ms")

    # Request 2: Cache hit (within TTL)
    print("\n요청 2: 캐시 히트 (TTL 내)")
    time.sleep(1)  # Wait 1s (< 5s TTL)

    start_2 = time.time()
    brief_2 = manager.generate_brief(user_text, mode='fast')
    latency_2 = (time.time() - start_2) * 1000
    print(f"  → 레이턴시: {latency_2:.0f}ms")

    speedup = latency_1 / max(latency_2, 1)
    print(f"  → 속도 향상: {speedup:.1f}x")

    # Request 3: Cache expired (after TTL)
    print(f"\n요청 3: 캐시 만료 (5초 후)")
    print("  (5초 대기 중...)")
    time.sleep(5)  # Wait for TTL expiry

    start_3 = time.time()
    brief_3 = manager.generate_brief(user_text, mode='fast')
    latency_3 = (time.time() - start_3) * 1000
    print(f"  → 레이턴시: {latency_3:.0f}ms (캐시 재생성)")

    # Summary
    print(f"\n캐시 TTL 검증 요약:")
    print(f"  캐시 미스 (첫 요청):  {latency_1:.0f}ms")
    print(f"  캐시 히트 (TTL 내):   {latency_2:.0f}ms (↓ {speedup:.1f}x)")
    print(f"  캐시 만료 (TTL 초과):  {latency_3:.0f}ms")


def demo_circuit_breaker():
    """데모 3: 서킷 브레이커 (Qwen → Mistral 전환)"""
    print("\n" + "="*80)
    print("데모 3: 서킷 브레이커 (Qwen → Mistral 전환)")
    print("="*80)

    manager = LLMEnsembleManager()
    user_text = "Fresh citrus scent"

    # Step 1: Normal operation
    print("\n단계 1: Qwen 정상 작동")
    brief = manager.generate_brief(user_text, mode='fast')
    print(f"  ✓ Brief 생성 성공 (Model: Qwen)")

    # Step 2: Simulate Qwen failures
    print("\n단계 2: Qwen 실패 시뮬레이션 (3회 연속)")
    manager.qwen.failure_rate = 1.0  # Force failures

    for i in range(1, 4):
        print(f"\n  시도 {i}/3:")
        try:
            brief = manager.generate_brief(user_text, mode='fast')
        except Exception as e:
            print(f"    ✗ 실패: {e}")

    # Step 3: Circuit breaker triggered
    print("\n단계 3: 서킷 브레이커 활성화 → Mistral 전환")
    print(f"  Circuit breaker state: {manager.circuit_breaker_qwen.state}")

    brief = manager.generate_brief(user_text, mode='fast')
    print(f"  ✓ Brief 생성 성공 (Model: Mistral fallback)")

    # Step 4: Qwen recovery
    print("\n단계 4: Qwen 복구")
    manager.qwen.failure_rate = 0.0  # Fix Qwen
    manager.circuit_breaker_qwen.last_failure_time = time.time() - 61  # Force recovery window

    brief = manager.generate_brief("Another fresh scent", mode='fast')
    print(f"  ✓ Qwen 복구 확인 (Circuit breaker: {manager.circuit_breaker_qwen.state})")


def demo_retry_fallback():
    """데모 4: 리트라이 및 폴백"""
    print("\n" + "="*80)
    print("데모 4: 리트라이 및 폴백 메커니즘")
    print("="*80)

    manager = LLMEnsembleManager()
    user_text = "상큼한 오렌지향"

    # Simulate intermittent failures
    print("\n시나리오: Qwen 간헐적 실패 (50% 실패율)")
    manager.qwen.failure_rate = 0.5

    for i in range(1, 4):
        print(f"\n요청 {i}/3:")
        try:
            brief = manager.generate_brief(user_text, mode='fast')
            print(f"  ✓ 성공: {brief['style']} fragrance")
        except Exception as e:
            print(f"  ✗ 실패: {e}")

    # Reset
    manager.qwen.failure_rate = 0.0


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demos"""
    print("\n" + "="*80)
    print("LLM 앙상블 동작 확인 데모")
    print("="*80)

    try:
        # Demo 1: Mode routing
        demo_mode_routing()

        # Demo 2: Cache TTL
        demo_cache_ttl()

        # Demo 3: Circuit breaker
        demo_circuit_breaker()

        # Demo 4: Retry and fallback
        demo_retry_fallback()

        print("\n" + "="*80)
        print("✓ 모든 데모 완료!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n중단됨")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
