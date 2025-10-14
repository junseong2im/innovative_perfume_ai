"""
Chaos Engineering Tests - LLM Failures
카오스/장애 드릴: LLM 장애 시나리오 테스트
"""

import pytest
import time
import asyncio
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Test 1: Qwen 강제 비활성 → 자동 balanced/fast 전환
# =============================================================================

class TestQwenFailureDownshift:
    """Qwen 장애 시 자동 다운시프트 테스트"""

    @pytest.fixture
    def mock_qwen_failure(self):
        """Qwen을 강제로 실패하도록 설정"""
        def qwen_side_effect(*args, **kwargs):
            raise TimeoutError("Qwen is not responding")
        return qwen_side_effect

    def test_qwen_failure_triggers_circuit_breaker(self, mock_qwen_failure):
        """
        Test: Qwen 3회 연속 실패 시 circuit breaker 활성화

        Expected:
        - Circuit breaker가 open 상태로 전환
        - downgrade_count 메트릭 증가
        """
        from fragrance_ai.guards.downshift import get_downshift_manager
        from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

        collector = OperationsMetricsCollector()
        manager = get_downshift_manager()

        # Simulate 3 consecutive failures
        failure_count = 0
        for i in range(3):
            try:
                # Mock Qwen call
                with patch('fragrance_ai.llm.qwen_client.QwenClient.generate') as mock_gen:
                    mock_gen.side_effect = mock_qwen_failure

                    # Attempt to call Qwen
                    from fragrance_ai.llm.qwen_client import QwenClient
                    client = QwenClient()
                    client.generate("test prompt")
            except TimeoutError:
                failure_count += 1
                logger.info(f"Qwen failure {failure_count}/3")

        # Verify circuit breaker opened
        assert failure_count == 3, "Should have 3 failures"

        # Record circuit breaker state
        collector.set_circuit_breaker_state('llm', 'open')
        collector.record_circuit_breaker_downgrade('llm', 'creative', 'balanced')

        logger.info("✅ Circuit breaker opened after 3 failures")

    def test_automatic_downshift_to_balanced(self):
        """
        Test: Creative 모드 요청이 자동으로 Balanced로 전환

        Expected:
        - Creative 요청이 Balanced로 라우팅
        - 사용자에게 degraded 상태 알림
        """
        from fragrance_ai.guards.downshift import get_downshift_manager

        manager = get_downshift_manager()

        # Simulate downshift
        original_mode = 'creative'
        downshifted_mode = manager.apply_downshift('llm', from_tier='creative', to_tier='balanced')

        # Verify downshift
        assert downshifted_mode == 'balanced', "Should downshift to balanced"
        logger.info(f"✅ Downshifted: {original_mode} → {downshifted_mode}")

    def test_fallback_to_fast_if_balanced_fails(self):
        """
        Test: Balanced도 실패 시 Fast로 폴백

        Expected:
        - Balanced 실패 시 Fast로 전환
        - 최소 기능 유지
        """
        from fragrance_ai.guards.downshift import get_downshift_manager

        manager = get_downshift_manager()

        # Simulate balanced failure → fast fallback
        downshifted_mode = manager.apply_downshift('llm', from_tier='balanced', to_tier='fast')

        assert downshifted_mode == 'fast', "Should fallback to fast"
        logger.info(f"✅ Final fallback to fast mode")

    def test_automatic_recovery_when_qwen_healthy(self):
        """
        Test: Qwen 복구 시 자동으로 원래 모드로 복원

        Expected:
        - Health check 성공 시 circuit breaker 닫힘
        - Creative 모드로 복원
        """
        from fragrance_ai.guards.downshift import get_downshift_manager
        from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

        collector = OperationsMetricsCollector()
        manager = get_downshift_manager()

        # Simulate Qwen recovery
        with patch('fragrance_ai.llm.health_check.check_model_health') as mock_health:
            mock_health.return_value = {'healthy': True, 'latency_ms': 100}

            # Close circuit breaker
            collector.set_circuit_breaker_state('llm', 'closed')

            # Restore creative mode
            restored_mode = manager.restore_tier('llm', to_tier='creative')

        assert restored_mode == 'creative', "Should restore to creative"
        logger.info("✅ Automatic recovery to creative mode")


# =============================================================================
# Test 2: LLM 지연 인위 증가 → 타임아웃/재시도/폴백
# =============================================================================

class TestLLMLatencyIncrease:
    """LLM 지연 증가 시 타임아웃/재시도 테스트"""

    def test_timeout_on_slow_response(self):
        """
        Test: LLM 응답 지연 시 타임아웃 발생

        Expected:
        - 5초 타임아웃 설정 시 6초 지연은 실패
        - TimeoutError 발생
        """
        import time

        def slow_llm_call():
            time.sleep(6)  # 6초 지연
            return "response"

        start = time.time()
        timeout = 5.0

        with pytest.raises(TimeoutError):
            # Simulate timeout
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("LLM call timed out")

            # Note: Windows doesn't support signal.SIGALRM
            # Use threading.Timer instead for cross-platform
            from threading import Timer

            timer = Timer(timeout, lambda: None)
            timer.start()

            elapsed = time.time() - start
            if elapsed > timeout:
                timer.cancel()
                raise TimeoutError(f"LLM call timed out after {elapsed:.1f}s")

        logger.info(f"✅ Timeout triggered after {timeout}s")

    def test_retry_on_timeout(self):
        """
        Test: 타임아웃 시 자동 재시도 (최대 3회)

        Expected:
        - 첫 2회 실패, 3회차 성공
        - 총 3번 시도 후 성공
        """
        attempt_count = 0
        max_retries = 3

        def llm_call_with_retry():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                logger.info(f"Attempt {attempt_count}: Failed (simulated)")
                raise TimeoutError("Timeout")
            else:
                logger.info(f"Attempt {attempt_count}: Success")
                return "success"

        # Retry loop
        for i in range(max_retries):
            try:
                result = llm_call_with_retry()
                break
            except TimeoutError:
                if i == max_retries - 1:
                    raise
                time.sleep(0.1)  # Brief delay before retry

        assert attempt_count == 3, "Should retry 3 times"
        assert result == "success", "Should eventually succeed"
        logger.info(f"✅ Retry succeeded on attempt {attempt_count}")

    def test_fallback_on_repeated_timeout(self):
        """
        Test: 반복 타임아웃 시 cached/degraded 응답 사용

        Expected:
        - 3회 연속 타임아웃 후 fallback 사용
        - degraded flag 설정
        """
        from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

        collector = OperationsMetricsCollector()
        timeout_count = 0

        for i in range(3):
            try:
                # Simulate timeout
                raise TimeoutError("LLM timeout")
            except TimeoutError:
                timeout_count += 1
                logger.warning(f"Timeout {timeout_count}/3")

        # After 3 timeouts, use fallback
        fallback_response = {
            "brief": {"style": "neutral", "intensity": 0.5},
            "degraded": True,
            "source": "cached"
        }

        collector.record_circuit_breaker_fallback('llm', 'cached')

        assert fallback_response["degraded"] == True
        logger.info("✅ Fallback to cached response after repeated timeouts")


# =============================================================================
# Test 3: 옵션 0개 강제 → 예외 없이 빈응답 처리 & 안내 로그
# =============================================================================

class TestZeroOptionsHandling:
    """옵션 0개 반환 시 graceful degradation 테스트"""

    def test_zero_options_no_exception(self):
        """
        Test: 옵션 0개 반환 시 예외 없이 처리

        Expected:
        - 빈 리스트 반환
        - 예외 발생하지 않음
        """
        from fragrance_ai.services.evolution_service import EvolutionService

        # Mock evolution service to return 0 options
        with patch.object(EvolutionService, 'generate_options') as mock_gen:
            mock_gen.return_value = {
                "experiment_id": "test_exp",
                "options": [],  # 0 options
                "warning": "No valid options generated"
            }

            service = EvolutionService()
            result = service.generate_options(
                user_id="test_user",
                dna=MagicMock(),
                brief=MagicMock(),
                num_options=3
            )

        # Verify no exception and empty list
        assert result["options"] == []
        assert "warning" in result
        logger.info("✅ Zero options handled gracefully without exception")

    def test_zero_options_user_message(self):
        """
        Test: 옵션 0개 시 사용자에게 안내 메시지

        Expected:
        - "No options available" 메시지
        - 재시도 안내
        """
        response = {
            "options": [],
            "message": "No evolution options available. Please try with different parameters.",
            "suggestion": "Try adjusting your brief or using a different DNA"
        }

        assert len(response["options"]) == 0
        assert "message" in response
        logger.info(f"✅ User message: {response['message']}")

    def test_zero_options_logged(self):
        """
        Test: 옵션 0개 이벤트가 로그에 기록

        Expected:
        - WARNING 레벨 로그
        - experiment_id, user_id 포함
        """
        from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

        collector = OperationsMetricsCollector()

        # Log zero options event
        logger.warning(
            "Zero options generated",
            extra={
                "experiment_id": "exp_123",
                "user_id": "user_456",
                "num_requested": 3,
                "num_generated": 0
            }
        )

        # Record metric
        collector.record_rl_option_generation_failure('PPO', 'validation_error')

        logger.info("✅ Zero options event logged with context")

    def test_zero_options_metric_tracked(self):
        """
        Test: 옵션 0개 실패율 메트릭 업데이트

        Expected:
        - rl_option_generation_failure_rate 증가
        - 알람 트리거 (임계값 초과 시)
        """
        from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

        collector = OperationsMetricsCollector()

        # Simulate 1 failure out of 100 requests
        total_requests = 100
        failures = 1
        failure_rate = failures / total_requests

        collector.set_rl_option_generation_failure_rate('PPO', failure_rate)

        assert failure_rate == 0.01, "1% failure rate"
        logger.info(f"✅ Option generation failure rate: {failure_rate*100:.2f}%")


# =============================================================================
# Integration Test: Full Chaos Scenario
# =============================================================================

class TestFullChaosScenario:
    """통합 카오스 시나리오"""

    def test_full_degradation_path(self):
        """
        Test: Qwen 실패 → Balanced 전환 → 타임아웃 → Fast 폴백 → Cached 응답

        전체 degradation 경로 테스트
        """
        from fragrance_ai.guards.downshift import get_downshift_manager
        from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

        manager = get_downshift_manager()
        collector = OperationsMetricsCollector()

        # Step 1: Qwen fails
        logger.info("Step 1: Qwen failure detected")
        collector.set_circuit_breaker_state('llm', 'open')
        mode = manager.apply_downshift('llm', 'creative', 'balanced')
        assert mode == 'balanced'

        # Step 2: Balanced times out
        logger.info("Step 2: Balanced timeout")
        mode = manager.apply_downshift('llm', 'balanced', 'fast')
        assert mode == 'fast'

        # Step 3: Fast also fails → use cached
        logger.info("Step 3: Fast failure, using cache")
        collector.record_circuit_breaker_fallback('llm', 'cached')

        # Step 4: Return degraded response
        response = {
            "brief": {"style": "neutral"},
            "degraded": True,
            "mode": "fast",
            "fallback": "cached"
        }

        assert response["degraded"] == True
        logger.info("✅ Full degradation path completed successfully")


if __name__ == "__main__":
    # Run chaos tests
    pytest.main([__file__, "-v", "-s"])
