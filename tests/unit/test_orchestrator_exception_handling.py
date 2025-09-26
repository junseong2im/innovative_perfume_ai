"""
단위 테스트: 오케스트레이터 예외 처리 검증
unittest.mock의 @patch를 사용하여 도구 실패를 시뮬레이션하고
적절한 예외 처리 및 대체 응답이 반환되는지 검증
"""

import pytest
import asyncio
from unittest.mock import patch, Mock, AsyncMock, MagicMock, call
from datetime import datetime
import json
from typing import Dict, Any

from fragrance_ai.orchestrator.artisan_orchestrator import (
    ArtisanOrchestrator,
    ConversationContext,
    ArtisanResponse,
    ToolExecutionResult,
    ToolStatus
)


class TestOrchestratorExceptionHandling:
    """오케스트레이터 예외 처리 단위 테스트"""

    @pytest.fixture
    def orchestrator(self):
        """테스트용 오케스트레이터 인스턴스"""
        return ArtisanOrchestrator()

    @pytest.fixture
    def context(self):
        """테스트용 대화 컨텍스트"""
        return ConversationContext(
            user_id="test_user",
            conversation_id="test_conv",
            history=[]
        )

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search')
    async def test_search_tool_failure_with_fallback(self, mock_search, orchestrator):
        """
        검색 도구 실패 시 폴백 메커니즘이 작동하는지 검증
        """
        # 검색 도구가 예외를 발생시키도록 설정
        mock_search.side_effect = ConnectionError("Database connection failed")

        # 폴백이 대체 응답을 반환하도록 설정
        with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
            mock_fallback.return_value = ToolExecutionResult(
                tool="hybrid_search",
                status=ToolStatus.PARTIAL,
                result={"results": ["fallback_result"], "method": "cache"},
                fallback_used=True
            )

            # 도구 실행
            result = await orchestrator._execute_single_tool_with_resilience({
                "tool": "hybrid_search",
                "params": {"query": "test query"}
            })

            # 검증
            assert mock_search.called  # 검색 도구가 호출됨
            assert mock_fallback.called  # 폴백이 호출됨
            assert result.status == ToolStatus.PARTIAL
            assert result.fallback_used is True
            assert "fallback_result" in result.result["results"]

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.create_recipe')
    async def test_generator_failure_with_retry(self, mock_generator, orchestrator):
        """
        생성 도구 실패 시 재시도 메커니즘이 작동하는지 검증
        """
        # 처음 2번은 실패, 3번째는 성공
        mock_generator.side_effect = [
            ValueError("Invalid parameters"),
            ConnectionError("Service unavailable"),
            {"name": "Success Recipe", "description": "Generated after retries"}
        ]

        orchestrator.max_retries = 3

        # 도구 실행
        result = await orchestrator._execute_single_tool_with_resilience({
            "tool": "recipe_generator",
            "params": {"description": "test"}
        })

        # 검증
        assert mock_generator.call_count == 3  # 3번 시도
        assert result.status == ToolStatus.SUCCESS
        assert result.result["name"] == "Success Recipe"
        assert result.retry_count == 2  # 0-indexed, 3번째 시도에서 성공

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.validate_composition')
    async def test_validator_timeout_handling(self, mock_validator, orchestrator):
        """
        검증 도구 타임아웃 시 적절히 처리되는지 검증
        """
        # 검증 도구가 오래 걸리도록 설정
        async def slow_validator(*args, **kwargs):
            await asyncio.sleep(10)  # 10초 대기
            return {"valid": True}

        mock_validator.side_effect = slow_validator
        orchestrator.timeout_seconds = 0.5  # 0.5초 타임아웃

        # 폴백 설정
        with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
            mock_fallback.return_value = ToolExecutionResult(
                tool="scientific_validator",
                status=ToolStatus.PARTIAL,
                result={"valid": True, "score": 7.0, "method": "rule_based"},
                fallback_used=True
            )

            # 도구 실행
            result = await orchestrator._execute_single_tool_with_resilience({
                "tool": "scientific_validator",
                "params": {}
            })

            # 검증
            assert mock_fallback.called  # 타임아웃 후 폴백 호출
            assert result.status in [ToolStatus.PARTIAL, ToolStatus.FAILED]

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.query_knowledge_base')
    async def test_knowledge_tool_complete_failure(self, mock_knowledge, orchestrator):
        """
        지식 도구 완전 실패 시 에러 응답이 반환되는지 검증
        """
        # 지식 도구가 항상 실패하도록 설정
        mock_knowledge.side_effect = Exception("Knowledge base corrupted")

        orchestrator.max_retries = 2

        # 폴백도 실패하도록 설정
        with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
            mock_fallback.return_value = ToolExecutionResult(
                tool="perfumer_knowledge",
                status=ToolStatus.FAILED,
                error="All fallbacks exhausted",
                fallback_used=True
            )

            # 도구 실행
            result = await orchestrator._execute_single_tool_with_resilience({
                "tool": "perfumer_knowledge",
                "params": {"query": "test"}
            })

            # 검증
            assert result.status == ToolStatus.FAILED
            assert result.error is not None
            assert "All fallbacks exhausted" in result.error

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search')
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.create_recipe')
    async def test_multiple_tool_failures_in_sequence(self, mock_generator, mock_search, orchestrator):
        """
        여러 도구가 연속으로 실패할 때 적절히 처리되는지 검증
        """
        # 모든 도구가 실패하도록 설정
        mock_search.side_effect = ConnectionError("Search failed")
        mock_generator.side_effect = ValueError("Generation failed")

        # 폴백 설정
        orchestrator.tool_priorities = {
            "hybrid_search": ["search_fallback"],
            "recipe_generator": ["simple_generator"]
        }

        # 각 도구에 대한 폴백 반환값 설정
        fallback_results = {
            "hybrid_search": ToolExecutionResult(
                tool="hybrid_search",
                status=ToolStatus.PARTIAL,
                result={"results": [], "method": "fallback"},
                fallback_used=True
            ),
            "recipe_generator": ToolExecutionResult(
                tool="recipe_generator",
                status=ToolStatus.PARTIAL,
                result={"name": "Template Recipe", "method": "template"},
                fallback_used=True
            )
        }

        with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
            mock_fallback.side_effect = lambda tool, params: fallback_results.get(tool)

            # 도구 계획 실행
            plan = [
                {"tool": "hybrid_search", "params": {"query": "test"}},
                {"tool": "recipe_generator", "params": {"description": "test"}}
            ]

            results = await orchestrator._execute_tools(plan)

            # 검증
            assert len(results) == 2
            assert all(r.status == ToolStatus.PARTIAL for r in results)
            assert all(r.fallback_used for r in results)

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search')
    async def test_circuit_breaker_activation(self, mock_search, orchestrator):
        """
        Circuit Breaker가 반복된 실패 후 활성화되는지 검증
        """
        # 검색이 항상 실패하도록 설정
        mock_search.side_effect = ConnectionError("Persistent failure")

        # 폴백도 실패하도록 설정
        with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
            mock_fallback.return_value = ToolExecutionResult(
                tool="hybrid_search",
                status=ToolStatus.FAILED,
                error="Fallback failed",
                fallback_used=True
            )

            # 5번 연속 실행
            for i in range(5):
                result = await orchestrator._execute_single_tool_with_resilience({
                    "tool": "hybrid_search",
                    "params": {"query": f"test_{i}"}
                })
                assert result.status == ToolStatus.FAILED

            # Circuit breaker 상태 확인
            breaker = orchestrator.circuit_breakers.get("hybrid_search")
            assert breaker is not None
            assert breaker.is_open is True
            assert breaker.failures >= 3

            # Circuit breaker가 열린 후 추가 호출
            result = await orchestrator._execute_single_tool_with_resilience({
                "tool": "hybrid_search",
                "params": {"query": "should_skip"}
            })

            # 실제 도구는 호출되지 않고 바로 폴백으로 감
            assert mock_search.call_count < 10  # Circuit breaker가 작동하여 호출 제한

    @pytest.mark.asyncio
    async def test_process_message_with_all_tools_failing(self, orchestrator, context):
        """
        모든 도구가 실패해도 사용자에게 적절한 응답을 반환하는지 검증
        """
        # 모든 도구를 실패하도록 패치
        with patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search') as mock_search, \
             patch('fragrance_ai.orchestrator.artisan_orchestrator.create_recipe') as mock_generator, \
             patch('fragrance_ai.orchestrator.artisan_orchestrator.validate_composition') as mock_validator, \
             patch('fragrance_ai.orchestrator.artisan_orchestrator.query_knowledge_base') as mock_knowledge:

            # 모든 도구가 예외를 발생시키도록 설정
            mock_search.side_effect = Exception("Search failed")
            mock_generator.side_effect = Exception("Generation failed")
            mock_validator.side_effect = Exception("Validation failed")
            mock_knowledge.side_effect = Exception("Knowledge failed")

            # 폴백도 모두 실패하도록 설정
            with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
                mock_fallback.return_value = ToolExecutionResult(
                    tool="any_tool",
                    status=ToolStatus.FAILED,
                    error="All systems down",
                    fallback_used=True
                )

                # 메시지 처리
                response = await orchestrator.process_message(
                    "향수를 만들어주세요",
                    context
                )

                # 검증: 모든 도구가 실패해도 응답이 반환됨
                assert response is not None
                assert isinstance(response, ArtisanResponse)
                assert response.message is not None
                assert "죄송" in response.message or "오류" in response.message
                assert response.recipe_summary is None  # 레시피는 생성되지 않음

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search')
    async def test_partial_success_aggregation(self, mock_search, orchestrator):
        """
        일부 도구만 성공할 때 부분 결과가 집계되는지 검증
        """
        # 검색은 성공
        mock_search.return_value = {
            "results": [
                {"name": "Rose Garden", "notes": ["Rose", "Jasmine"]},
                {"name": "Summer Breeze", "notes": ["Citrus", "Mint"]}
            ]
        }

        # 생성기는 실패
        with patch('fragrance_ai.orchestrator.artisan_orchestrator.create_recipe') as mock_generator:
            mock_generator.side_effect = ValueError("Generation failed")

            # 생성기 폴백 설정
            with patch.object(orchestrator, '_execute_fallback') as mock_fallback:
                def fallback_logic(tool, params):
                    if tool == "recipe_generator":
                        return ToolExecutionResult(
                            tool="recipe_generator",
                            status=ToolStatus.PARTIAL,
                            result={"name": "Basic Template", "method": "fallback"},
                            fallback_used=True
                        )
                    return ToolExecutionResult(
                        tool=tool,
                        status=ToolStatus.FAILED,
                        error="No fallback"
                    )

                mock_fallback.side_effect = fallback_logic

                # 도구 계획 실행
                plan = [
                    {"tool": "hybrid_search", "params": {"query": "rose"}},
                    {"tool": "recipe_generator", "params": {"description": "rose perfume"}}
                ]

                results = await orchestrator._execute_tools(plan)

                # 검증
                assert len(results) == 2
                assert results[0].status == ToolStatus.SUCCESS  # 검색 성공
                assert results[0].result["results"][0]["name"] == "Rose Garden"
                assert results[1].status == ToolStatus.PARTIAL  # 생성기 폴백
                assert results[1].fallback_used is True

    @pytest.mark.asyncio
    async def test_graceful_error_message_formatting(self, orchestrator, context):
        """
        에러 발생 시 사용자 친화적인 메시지가 생성되는지 검증
        """
        # 모든 도구 실패 시뮬레이션
        with patch.object(orchestrator, '_execute_tools') as mock_execute:
            mock_execute.return_value = [
                ToolExecutionResult(
                    tool="hybrid_search",
                    status=ToolStatus.FAILED,
                    error="Database connection failed"
                ),
                ToolExecutionResult(
                    tool="recipe_generator",
                    status=ToolStatus.FAILED,
                    error="Model not loaded"
                )
            ]

            # 응답 합성 로직 테스트
            response = await orchestrator._synthesize_response(
                message="향수를 만들어주세요",
                intent={"type": "create_perfume", "confidence": 0.9},
                tool_results=mock_execute.return_value,
                context=context
            )

            # 검증
            assert response is not None
            assert response.message is not None
            # 기술적 에러 메시지가 아닌 사용자 친화적 메시지
            assert "Database connection" not in response.message
            assert "Model not loaded" not in response.message
            # 대신 일반적인 사과 메시지 포함
            assert any(word in response.message for word in ["죄송", "다시", "문제"])


class TestExceptionPropagation:
    """예외 전파 및 처리 테스트"""

    @pytest.mark.asyncio
    @patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search')
    async def test_exception_logging(self, mock_search):
        """
        예외 발생 시 적절히 로깅되는지 검증
        """
        orchestrator = ArtisanOrchestrator()
        mock_search.side_effect = RuntimeError("Critical error")

        with patch('fragrance_ai.orchestrator.artisan_orchestrator.logger') as mock_logger:
            # 도구 실행
            try:
                await orchestrator._execute_tool_internal("hybrid_search", {})
            except RuntimeError:
                pass  # 예외는 예상됨

            # 로깅 검증
            mock_logger.error.assert_called()
            error_message = mock_logger.error.call_args[0][0]
            assert "hybrid_search" in error_message
            assert "Critical error" in error_message

    @pytest.mark.asyncio
    async def test_exception_context_preservation(self):
        """
        예외 컨텍스트가 보존되는지 검증
        """
        orchestrator = ArtisanOrchestrator()

        # 커스텀 예외
        class CustomToolError(Exception):
            def __init__(self, message, context):
                super().__init__(message)
                self.context = context

        with patch('fragrance_ai.orchestrator.artisan_orchestrator.hybrid_search') as mock_search:
            error_context = {"tool": "search", "user": "test", "query": "rose"}
            mock_search.side_effect = CustomToolError("Search failed", error_context)

            # 도구 실행
            result = await orchestrator._execute_single_tool_with_resilience({
                "tool": "hybrid_search",
                "params": {"query": "rose"}
            })

            # 검증
            assert result.status == ToolStatus.FAILED
            assert result.error is not None
            assert "Search failed" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])