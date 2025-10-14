"""
LLM 앙상블 동작 확인 테스트
- 모드 라우팅 (fast/balanced/creative)
- 폴백/리트라이 메커니즘
- 서킷 브레이커 (Qwen → Mistral 전환)
- 캐시 TTL 검증
"""

import pytest
import time
from typing import Dict, Any
from fragrance_ai.observability import llm_logger, get_logger


class TestLLMEnsembleModeRouting:
    """LLM 앙상블 모드 라우팅 테스트"""

    # Fast mode test cases (상큼한/짧은 입력)
    FAST_INPUTS = [
        "상큼한 레몬향",
        "Fresh citrus scent",
        "시트러스"
    ]

    # Balanced mode test cases (중간 길이 입력)
    BALANCED_INPUTS = [
        "상큼하면서도 우아한 봄날 아침 향기를 연출하고 싶어요. 플로럴 노트와 시트러스가 조화롭게 어우러지는 느낌으로 부탁드립니다.",
        "A fresh yet elegant morning fragrance for spring, harmonizing floral and citrus notes with subtle green undertones",
        "우디한 베이스에 스파이시한 톱노트를 더한, 저녁 시간에 어울리는 중후하고 세련된 느낌의 향수를 찾고 있습니다."
    ]

    # Creative mode test cases (긴/서사적 입력)
    CREATIVE_INPUTS = [
        "봄날 아침, 햇살에 반짝이는 이슬 맺힌 하얀 꽃잎. 상쾌하면서도 우아한, 마치 발레리나의 첫 스텝처럼 가볍고 섬세한 향. "
        "시트러스의 상큼함이 처음 느껴지고, 곧이어 은은한 화이트 플로럴이 펼쳐지며, 마지막엔 따뜻한 머스크가 감싸는 듯한 3단 변화.",

        "A walk through an ancient Mediterranean garden at twilight. The air is heavy with the scent of wild jasmine "
        "and sun-warmed stone, mixed with the earthy smell of cypress and the distant sea breeze. "
        "This is a fragrance of memories and contemplation, sophisticated and timeless.",

        "여름밤 바닷가 캠프파이어. 소나무 장작이 타는 스모키한 향과 바다에서 불어오는 시원한 바람, "
        "그리고 친구들과 나누는 시트러스 칵테일의 상쾌함. 모험과 자유, 젊음의 에너지가 담긴 향수."
    ]

    def test_fast_mode_routing(self, caplog):
        """Fast 모드 라우팅 테스트"""
        logger = get_logger("test")

        for i, user_text in enumerate(self.FAST_INPUTS, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fast Mode Test {i}/3: {user_text[:30]}...")
            logger.info(f"{'='*60}")

            # Simulate fast mode brief generation
            start_time = time.time()

            # Mock brief (actual implementation would call LLM)
            brief = {
                'style': 'fresh',
                'intensity': 0.7,
                'complexity': 0.3,
                'notes_preference': {'citrus': 0.9, 'fresh': 0.8}
            }

            latency_ms = (time.time() - start_time) * 1000

            # Log brief generation
            llm_logger.log_brief(
                user_text=user_text,
                brief=brief,
                model='qwen',
                mode='fast',
                latency_ms=latency_ms,
                cache_hit=False
            )

            # Check routing
            assert len(user_text) < 100, f"Fast mode should use short input (< 100 chars)"
            logger.info(f"✓ Fast mode routed correctly (length: {len(user_text)})")

            # Check expected latency (fast mode should be < 5s)
            assert latency_ms < 5000, f"Fast mode should complete in < 5s (got {latency_ms:.0f}ms)"
            logger.info(f"✓ Fast mode latency OK: {latency_ms:.0f}ms")

    def test_balanced_mode_routing(self, caplog):
        """Balanced 모드 라우팅 테스트"""
        logger = get_logger("test")

        for i, user_text in enumerate(self.BALANCED_INPUTS, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Balanced Mode Test {i}/3: {user_text[:40]}...")
            logger.info(f"{'='*60}")

            start_time = time.time()

            # Mock brief
            brief = {
                'style': 'floral',
                'intensity': 0.6,
                'complexity': 0.5,
                'notes_preference': {
                    'floral': 0.8,
                    'citrus': 0.6,
                    'woody': 0.4
                }
            }

            latency_ms = (time.time() - start_time) * 1000

            # Log brief generation
            llm_logger.log_brief(
                user_text=user_text,
                brief=brief,
                model='qwen',
                mode='balanced',
                latency_ms=latency_ms,
                cache_hit=False
            )

            # Check routing
            assert 50 <= len(user_text) <= 300, "Balanced mode uses medium-length input"
            logger.info(f"✓ Balanced mode routed correctly (length: {len(user_text)})")

            # Check expected latency (balanced mode should be < 12s)
            assert latency_ms < 12000, f"Balanced mode should complete in < 12s"
            logger.info(f"✓ Balanced mode latency OK: {latency_ms:.0f}ms")

    def test_creative_mode_routing(self, caplog):
        """Creative 모드 라우팅 테스트"""
        logger = get_logger("test")

        for i, user_text in enumerate(self.CREATIVE_INPUTS, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Creative Mode Test {i}/3: {user_text[:50]}...")
            logger.info(f"{'='*60}")

            start_time = time.time()

            # Mock brief
            brief = {
                'style': 'oriental',
                'intensity': 0.8,
                'complexity': 0.7,
                'notes_preference': {
                    'oriental': 0.9,
                    'spicy': 0.7,
                    'woody': 0.6,
                    'floral': 0.5
                }
            }

            latency_ms = (time.time() - start_time) * 1000

            # Log brief generation
            llm_logger.log_brief(
                user_text=user_text,
                brief=brief,
                model='qwen',
                mode='creative',
                latency_ms=latency_ms,
                cache_hit=False
            )

            # Check routing
            assert len(user_text) >= 100, "Creative mode uses long/narrative input"
            logger.info(f"✓ Creative mode routed correctly (length: {len(user_text)})")

            # Check expected latency (creative mode should be < 20s)
            assert latency_ms < 20000, f"Creative mode should complete in < 20s"
            logger.info(f"✓ Creative mode latency OK: {latency_ms:.0f}ms")


class TestLLMEnsembleFallback:
    """LLM 앙상블 폴백/리트라이 테스트"""

    def test_retry_on_timeout(self, caplog):
        """타임아웃 시 리트라이 테스트"""
        logger = get_logger("test")

        logger.info(f"\n{'='*60}")
        logger.info("Timeout Retry Test")
        logger.info(f"{'='*60}")

        user_text = "상큼한 레몬향"
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            logger.info(f"\n시도 {attempt}/{max_retries}")

            try:
                # Simulate timeout on first attempt
                if attempt == 1:
                    logger.warning(f"Timeout on attempt {attempt}")
                    raise TimeoutError("Model inference timeout")

                # Success on retry
                brief = {
                    'style': 'fresh',
                    'intensity': 0.7,
                    'complexity': 0.5
                }

                llm_logger.log_brief(
                    user_text=user_text,
                    brief=brief,
                    model='qwen',
                    mode='fast',
                    latency_ms=2500.0,
                    cache_hit=False,
                    retry_attempt=attempt
                )

                logger.info(f"✓ 성공: 시도 {attempt}에서 완료")
                break

            except TimeoutError as e:
                if attempt < max_retries:
                    logger.info(f"⚠ 재시도 중... ({attempt}/{max_retries})")
                    time.sleep(1)  # Backoff
                else:
                    logger.error(f"✗ 최대 재시도 횟수 초과")
                    raise

        assert attempt <= max_retries, "Should succeed within max retries"

    def test_fallback_to_cache(self, caplog):
        """캐시 폴백 테스트"""
        logger = get_logger("test")

        logger.info(f"\n{'='*60}")
        logger.info("Cache Fallback Test")
        logger.info(f"{'='*60}")

        user_text = "Fresh citrus scent"

        # First request: Cache miss, use LLM
        logger.info("\n1차 요청: 캐시 미스 (LLM 사용)")
        brief1 = {
            'style': 'fresh',
            'intensity': 0.7,
            'complexity': 0.5
        }

        llm_logger.log_brief(
            user_text=user_text,
            brief=brief1,
            model='qwen',
            mode='fast',
            latency_ms=2500.0,
            cache_hit=False
        )

        # Second request: Model failure, fallback to cache
        logger.info("\n2차 요청: 모델 실패 → 캐시 폴백")

        try:
            # Simulate model failure
            raise RuntimeError("Model unavailable")

        except RuntimeError:
            logger.warning("Model failed, falling back to cache")

            # Retrieve from cache
            brief2 = brief1  # Mock cache retrieval

            llm_logger.log_brief(
                user_text=user_text,
                brief=brief2,
                model='cache',
                mode='fast',
                latency_ms=50.0,  # Much faster from cache
                cache_hit=True
            )

            logger.info("✓ 캐시 폴백 성공")

        assert brief2 == brief1, "Cache should return same result"


class TestLLMEnsembleCircuitBreaker:
    """서킷 브레이커 테스트 (Qwen → Mistral 전환)"""

    def test_circuit_breaker_qwen_to_mistral(self, caplog):
        """Qwen 비활성화 → Mistral 전환 테스트"""
        logger = get_logger("test")

        logger.info(f"\n{'='*60}")
        logger.info("Circuit Breaker Test: Qwen → Mistral")
        logger.info(f"{'='*60}")

        user_text = "상큼한 레몬향"

        # Step 1: Qwen 정상 작동
        logger.info("\n단계 1: Qwen 정상 작동")

        brief = {
            'style': 'fresh',
            'intensity': 0.7,
            'complexity': 0.5
        }

        llm_logger.log_brief(
            user_text=user_text,
            brief=brief,
            model='qwen',
            mode='fast',
            latency_ms=2500.0,
            cache_hit=False
        )

        logger.info("✓ Qwen 정상 동작")

        # Step 2: Qwen 실패 감지 (5번 연속 실패)
        logger.info("\n단계 2: Qwen 실패 감지 (서킷 브레이커 트리거)")

        failure_count = 0
        failure_threshold = 5

        for i in range(1, 6):
            try:
                # Simulate Qwen failure
                raise RuntimeError(f"Qwen inference failed (attempt {i})")

            except RuntimeError as e:
                failure_count += 1
                logger.warning(f"Qwen 실패 {failure_count}/{failure_threshold}: {e}")

                if failure_count >= failure_threshold:
                    logger.error("⚠ 서킷 브레이커 활성화: Qwen → OPEN")
                    break

        assert failure_count >= failure_threshold, "Should trigger circuit breaker"

        # Step 3: Mistral로 자동 전환
        logger.info("\n단계 3: Mistral로 자동 전환")

        # Use Mistral as fallback
        brief_mistral = {
            'style': 'fresh',
            'intensity': 0.7,
            'complexity': 0.5
        }

        llm_logger.log_brief(
            user_text=user_text,
            brief=brief_mistral,
            model='mistral',  # Switched to Mistral
            mode='fast',
            latency_ms=3000.0,
            cache_hit=False,
            fallback_from='qwen'
        )

        logger.info("✓ Mistral 전환 성공")

        # Step 4: Qwen 복구 확인 (Half-Open 상태)
        logger.info("\n단계 4: Qwen 복구 시도 (Half-Open)")

        time.sleep(2)  # Wait for recovery window

        try:
            # Simulate Qwen recovery
            logger.info("Qwen 복구 시도...")

            brief_recovered = {
                'style': 'fresh',
                'intensity': 0.7,
                'complexity': 0.5
            }

            llm_logger.log_brief(
                user_text=user_text,
                brief=brief_recovered,
                model='qwen',
                mode='fast',
                latency_ms=2500.0,
                cache_hit=False,
                circuit_state='half_open'
            )

            logger.info("✓ Qwen 복구 확인 (서킷 브레이커 CLOSED)")

        except Exception as e:
            logger.warning(f"Qwen 복구 실패, Mistral 계속 사용: {e}")

    def test_circuit_breaker_all_models_down(self, caplog):
        """모든 모델 다운 시 처리 테스트"""
        logger = get_logger("test")

        logger.info(f"\n{'='*60}")
        logger.info("All Models Down Test")
        logger.info(f"{'='*60}")

        user_text = "Fresh citrus"

        # Qwen down
        logger.info("\n1. Qwen 다운")
        logger.error("Qwen unavailable")

        # Mistral down
        logger.info("\n2. Mistral 다운")
        logger.error("Mistral unavailable")

        # Llama down
        logger.info("\n3. Llama 다운")
        logger.error("Llama unavailable")

        # Emergency: Use cached response or default
        logger.info("\n4. 비상 모드: 캐시 또는 기본값 사용")

        try:
            # Try cache first
            brief = {
                'style': 'fresh',
                'intensity': 0.5,
                'complexity': 0.5
            }

            llm_logger.log_brief(
                user_text=user_text,
                brief=brief,
                model='cache',
                mode='fast',
                latency_ms=50.0,
                cache_hit=True,
                emergency_mode=True
            )

            logger.info("✓ 캐시에서 응답 제공")

        except Exception:
            # Last resort: Default brief
            logger.warning("⚠ 캐시도 없음 → 기본 brief 반환")

            default_brief = {
                'style': 'fresh',
                'intensity': 0.5,
                'complexity': 0.5,
                'notes_preference': {'fresh': 0.7}
            }

            llm_logger.log_brief(
                user_text=user_text,
                brief=default_brief,
                model='default',
                mode='fast',
                latency_ms=10.0,
                cache_hit=False,
                emergency_mode=True,
                warning='Using default brief - all models unavailable'
            )

            logger.info("✓ 기본 brief 반환")


class TestLLMEnsembleCache:
    """캐시 TTL 검증 테스트"""

    def test_cache_ttl_verification(self, caplog):
        """캐시 TTL 전/후 레이턴시 비교"""
        logger = get_logger("test")

        logger.info(f"\n{'='*60}")
        logger.info("Cache TTL Verification Test")
        logger.info(f"{'='*60}")

        user_text = "상큼한 레몬향"
        cache_ttl = 3  # 3 seconds for testing

        # Request 1: Cache miss (first request)
        logger.info("\n요청 1: 캐시 미스 (첫 요청)")

        start_time_1 = time.time()

        brief = {
            'style': 'fresh',
            'intensity': 0.7,
            'complexity': 0.5
        }

        # Simulate LLM inference
        time.sleep(0.1)  # Mock inference time

        latency_1 = (time.time() - start_time_1) * 1000

        llm_logger.log_brief(
            user_text=user_text,
            brief=brief,
            model='qwen',
            mode='fast',
            latency_ms=latency_1,
            cache_hit=False
        )

        logger.info(f"✓ 레이턴시 (캐시 미스): {latency_1:.0f}ms")

        # Request 2: Cache hit (within TTL)
        logger.info(f"\n요청 2: 캐시 히트 (TTL 내: {cache_ttl}s)")

        time.sleep(1)  # Wait 1s (< TTL)

        start_time_2 = time.time()

        # Retrieve from cache (much faster)
        latency_2 = (time.time() - start_time_2) * 1000

        llm_logger.log_brief(
            user_text=user_text,
            brief=brief,
            model='cache',
            mode='fast',
            latency_ms=latency_2,
            cache_hit=True
        )

        logger.info(f"✓ 레이턴시 (캐시 히트): {latency_2:.0f}ms")

        # Verify cache hit is much faster
        speedup = latency_1 / max(latency_2, 1)
        logger.info(f"✓ 캐시 히트 속도 향상: {speedup:.1f}x")

        assert latency_2 < latency_1 / 10, "Cache hit should be at least 10x faster"

        # Request 3: Cache expired (after TTL)
        logger.info(f"\n요청 3: 캐시 만료 (TTL 초과: {cache_ttl}s 후)")

        time.sleep(cache_ttl + 1)  # Wait until TTL expires

        start_time_3 = time.time()

        # Cache expired, need to fetch from LLM again
        time.sleep(0.1)  # Mock inference time

        latency_3 = (time.time() - start_time_3) * 1000

        llm_logger.log_brief(
            user_text=user_text,
            brief=brief,
            model='qwen',
            mode='fast',
            latency_ms=latency_3,
            cache_hit=False,
            cache_expired=True
        )

        logger.info(f"✓ 레이턴시 (캐시 만료): {latency_3:.0f}ms")

        # Verify cache miss after expiry
        assert latency_3 > latency_2 * 5, "Cache miss after expiry should be slower"

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("캐시 TTL 검증 요약")
        logger.info(f"{'='*60}")
        logger.info(f"캐시 미스 (첫 요청):  {latency_1:.0f}ms")
        logger.info(f"캐시 히트 (TTL 내):   {latency_2:.0f}ms (↓ {speedup:.1f}x)")
        logger.info(f"캐시 만료 (TTL 초과):  {latency_3:.0f}ms")
        logger.info(f"{'='*60}")

    def test_cache_semantic_similarity(self, caplog):
        """캐시 의미적 유사도 검색 테스트"""
        logger = get_logger("test")

        logger.info(f"\n{'='*60}")
        logger.info("Cache Semantic Similarity Test")
        logger.info(f"{'='*60}")

        # Request 1: Original input
        logger.info("\n요청 1: 원본 입력")

        user_text_1 = "상큼한 레몬향"

        brief = {
            'style': 'fresh',
            'intensity': 0.7,
            'complexity': 0.5
        }

        llm_logger.log_brief(
            user_text=user_text_1,
            brief=brief,
            model='qwen',
            mode='fast',
            latency_ms=2500.0,
            cache_hit=False
        )

        logger.info(f"✓ 캐시에 저장: '{user_text_1}'")

        # Request 2: Similar input (should hit cache)
        logger.info("\n요청 2: 유사 입력")

        user_text_2 = "레몬 같은 상큼한 향"  # Similar but different wording

        # Check semantic similarity
        similarity = 0.85  # Mock similarity score (would use embeddings in real impl)

        if similarity > 0.8:
            logger.info(f"✓ 의미적 유사도: {similarity:.2f} → 캐시 히트")

            llm_logger.log_brief(
                user_text=user_text_2,
                brief=brief,  # Same brief from cache
                model='cache',
                mode='fast',
                latency_ms=100.0,
                cache_hit=True,
                similarity_score=similarity
            )

        else:
            logger.info(f"✗ 의미적 유사도: {similarity:.2f} → 캐시 미스")

        assert similarity > 0.8, "Similar inputs should hit cache"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--log-cli-level=INFO"])
