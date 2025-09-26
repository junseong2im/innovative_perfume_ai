"""
Comprehensive Exception Handling Tests for Orchestrators
Tests all failure modes, recovery mechanisms, and alternative response strategies
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json
from typing import Dict, Any, List

from fragrance_ai.orchestrator.artisan_orchestrator import (
    ArtisanOrchestrator,
    ToolExecutionResult,
    ToolStatus,
    CircuitBreakerState
)


class TestComprehensiveExceptionHandling:
    """Comprehensive tests for orchestrator exception handling"""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance with test configuration"""
        orchestrator = ArtisanOrchestrator()

        # Configure for testing
        orchestrator.max_retries = 2
        orchestrator.timeout_seconds = 1
        orchestrator.enable_partial_results = True

        return orchestrator

    @pytest.mark.asyncio
    async def test_tool_failure_with_successful_fallback(self, orchestrator):
        """Test that when a tool fails, the fallback mechanism provides alternative response"""

        # Make primary tool fail
        async def failing_search(*args, **kwargs):
            raise ConnectionError("Database connection failed")

        # Make fallback succeed
        async def successful_fallback(*args, **kwargs):
            return {"results": ["fallback_result"], "method": "cache"}

        # Patch tools
        orchestrator.tools["hybrid_search"] = failing_search
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="hybrid_search",
            status=ToolStatus.PARTIAL,
            result={"results": ["fallback_result"], "method": "cache"},
            fallback_used=True
        ))

        # Execute
        plan = [{"tool": "hybrid_search", "params": {"query": "test"}}]
        results = await orchestrator._execute_tools(plan)

        # Verify fallback was used
        assert len(results) == 1
        assert results[0].status == ToolStatus.PARTIAL
        assert results[0].fallback_used is True
        assert "fallback_result" in results[0].result["results"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, orchestrator):
        """Test that circuit breaker opens after repeated failures"""

        failure_count = 0

        async def failing_tool(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise ConnectionError(f"Failure {failure_count}")

        orchestrator.tools["hybrid_search"] = failing_tool
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="hybrid_search",
            status=ToolStatus.FAILED,
            error="Fallback also failed"
        ))

        # Execute multiple times to trigger circuit breaker
        for i in range(5):
            plan = [{"tool": "hybrid_search", "params": {}}]
            await orchestrator._execute_tools(plan)

        # Check circuit breaker state
        breaker = orchestrator.circuit_breakers.get("hybrid_search")
        assert breaker is not None
        assert breaker.is_open is True
        assert breaker.failures >= 3

    @pytest.mark.asyncio
    async def test_partial_success_aggregation(self, orchestrator):
        """Test that partial successes are properly aggregated"""

        async def search_succeeds(*args, **kwargs):
            return {"results": ["item1", "item2"]}

        async def generator_fails(*args, **kwargs):
            raise ValueError("Generation failed")

        async def validator_succeeds(*args, **kwargs):
            return {"valid": True, "score": 8.5}

        orchestrator.tools["hybrid_search"] = search_succeeds
        orchestrator.tools["recipe_generator"] = generator_fails
        orchestrator.tools["scientific_validator"] = validator_succeeds

        # Set up fallback for generator
        orchestrator._execute_fallback = AsyncMock(side_effect=lambda tool, params:
            ToolExecutionResult(
                tool=tool,
                status=ToolStatus.PARTIAL,
                result={"name": "Fallback Recipe", "method": "template"},
                fallback_used=True
            ) if tool == "recipe_generator" else ToolExecutionResult(
                tool=tool,
                status=ToolStatus.FAILED,
                error="No fallback"
            )
        )

        # Execute plan with mixed success/failure
        plan = [
            {"tool": "hybrid_search", "params": {}},
            {"tool": "recipe_generator", "params": {}},
            {"tool": "scientific_validator", "params": {}}
        ]

        results = await orchestrator._execute_tools(plan)

        # Verify partial results
        assert len(results) == 3
        assert results[0].status == ToolStatus.SUCCESS
        assert results[1].status == ToolStatus.PARTIAL
        assert results[1].fallback_used is True
        assert results[2].status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_timeout_handling_with_partial_results(self, orchestrator):
        """Test that timeouts return partial results when available"""

        partial_data = []

        async def slow_tool(*args, **kwargs):
            partial_data.append("phase1")
            await asyncio.sleep(0.5)
            partial_data.append("phase2")
            await asyncio.sleep(2)  # Will timeout
            partial_data.append("phase3")
            return {"complete": True}

        orchestrator.tools["hybrid_search"] = slow_tool
        orchestrator.timeout_seconds = 1

        # Set up fallback to return partial data
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="hybrid_search",
            status=ToolStatus.PARTIAL,
            result={"partial_data": partial_data.copy(), "complete": False},
            fallback_used=True
        ))

        plan = [{"tool": "hybrid_search", "params": {}}]
        results = await orchestrator._execute_tools(plan)

        # Should have partial data before timeout
        assert len(partial_data) >= 1
        assert len(partial_data) < 3  # Should not complete all phases
        assert results[0].status in [ToolStatus.PARTIAL, ToolStatus.FAILED, ToolStatus.TIMEOUT]

    @pytest.mark.asyncio
    async def test_dependency_chain_with_failure_recovery(self, orchestrator):
        """Test handling of dependent tools when one fails"""

        execution_log = []

        async def tool_a(*args, **kwargs):
            execution_log.append("A")
            return {"data": "from_A"}

        async def tool_b(*args, **kwargs):
            execution_log.append("B")
            if "from_A" not in str(kwargs.get("context", {})):
                raise ValueError("Missing dependency from A")
            return {"data": "from_B"}

        async def tool_c(*args, **kwargs):
            execution_log.append("C")
            # C can work independently
            return {"data": "from_C", "independent": True}

        orchestrator.tools["tool_a"] = tool_a
        orchestrator.tools["tool_b"] = tool_b
        orchestrator.tools["tool_c"] = tool_c

        # No fallback for B
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="tool_b",
            status=ToolStatus.FAILED,
            error="No fallback available"
        ))

        # Plan with dependencies
        plan = [
            {"tool": "tool_a", "params": {}, "parallel": False},
            {"tool": "tool_b", "params": {"context": {"from_A": True}},
             "depends_on": ["tool_a"], "parallel": False},
            {"tool": "tool_c", "params": {}, "parallel": False}  # Independent
        ]

        results = await orchestrator._execute_tools(plan)

        # Verify execution
        assert "A" in execution_log
        assert "B" in execution_log
        assert "C" in execution_log
        assert results[0].status == ToolStatus.SUCCESS
        assert results[2].status == ToolStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_graceful_degradation_through_service_levels(self, orchestrator):
        """Test degradation from premium to basic service levels"""

        service_attempts = []

        async def premium_service(*args, **kwargs):
            service_attempts.append("premium")
            raise Exception("Premium unavailable")

        async def standard_service(*args, **kwargs):
            service_attempts.append("standard")
            raise Exception("Standard unavailable")

        async def basic_service(*args, **kwargs):
            service_attempts.append("basic")
            return {"level": "basic", "data": "limited"}

        orchestrator.tools["premium"] = premium_service
        orchestrator.tools["standard"] = standard_service
        orchestrator.tools["basic"] = basic_service

        # Configure fallback chain
        call_count = 0
        async def smart_fallback(tool, params):
            nonlocal call_count
            call_count += 1

            if tool == "premium" and call_count == 1:
                # First fallback to standard
                try:
                    result = await standard_service()
                    return ToolExecutionResult(tool=tool, status=ToolStatus.SUCCESS, result=result)
                except:
                    pass

            if call_count <= 2:
                # Second fallback to basic
                try:
                    result = await basic_service()
                    return ToolExecutionResult(
                        tool=tool,
                        status=ToolStatus.PARTIAL,
                        result=result,
                        fallback_used=True
                    )
                except:
                    pass

            return ToolExecutionResult(tool=tool, status=ToolStatus.FAILED, error="All levels failed")

        orchestrator._execute_fallback = smart_fallback

        # Execute
        plan = [{"tool": "premium", "params": {}}]
        results = await orchestrator._execute_tools(plan)

        # Should degrade to basic
        assert results[0].status == ToolStatus.PARTIAL
        assert results[0].result["level"] == "basic"
        assert "basic" in service_attempts

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self, orchestrator):
        """Test retry mechanism with exponential backoff"""

        attempt_times = []
        attempt_count = 0

        async def flaky_tool(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            attempt_times.append(datetime.utcnow())

            if attempt_count < 3:
                raise ConnectionError(f"Attempt {attempt_count} failed")

            return {"success": True, "attempts": attempt_count}

        orchestrator.tools["hybrid_search"] = flaky_tool
        orchestrator.max_retries = 3

        # No fallback needed - will succeed on retry
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="hybrid_search",
            status=ToolStatus.FAILED,
            error="No fallback"
        ))

        plan = [{"tool": "hybrid_search", "params": {}}]
        results = await orchestrator._execute_tools(plan)

        # Should succeed after retries
        assert results[0].status == ToolStatus.SUCCESS
        assert results[0].result["attempts"] == 3
        assert results[0].retry_count > 0

    @pytest.mark.asyncio
    async def test_parallel_execution_with_isolated_failures(self, orchestrator):
        """Test that failures in parallel execution don't affect other tools"""

        results_dict = {"success": [], "failure": []}

        async def tool_factory(tool_id):
            await asyncio.sleep(0.1)

            if tool_id % 2 == 0:
                results_dict["success"].append(tool_id)
                return {"id": tool_id, "status": "success"}
            else:
                results_dict["failure"].append(tool_id)
                raise Exception(f"Tool {tool_id} failed")

        # Set up multiple tools
        for i in range(6):
            orchestrator.tools[f"tool_{i}"] = lambda *a, id=i, **k: tool_factory(id)

        # Simple fallback for failed tools
        orchestrator._execute_fallback = AsyncMock(side_effect=lambda tool, params:
            ToolExecutionResult(
                tool=tool,
                status=ToolStatus.PARTIAL,
                result={"fallback": True},
                fallback_used=True
            )
        )

        # Create parallel plan
        plan = [
            {"tool": f"tool_{i}", "params": {}, "parallel": True}
            for i in range(6)
        ]

        results = await orchestrator._execute_tools(plan)

        # Verify isolation - successes and failures handled independently
        assert len(results) == 6
        success_count = sum(1 for r in results if r.status == ToolStatus.SUCCESS)
        fallback_count = sum(1 for r in results if r.fallback_used)

        assert success_count == 3  # Even numbered tools succeed
        assert fallback_count == 3  # Odd numbered tools use fallback

    @pytest.mark.asyncio
    async def test_critical_tool_failure_stops_execution(self, orchestrator):
        """Test that critical tool failure stops further execution"""

        execution_log = []

        async def tool_1(*args, **kwargs):
            execution_log.append("tool_1")
            return {"success": True}

        async def critical_tool(*args, **kwargs):
            execution_log.append("critical")
            raise Exception("Critical failure")

        async def tool_3(*args, **kwargs):
            execution_log.append("tool_3")
            return {"success": True}

        orchestrator.tools["tool_1"] = tool_1
        orchestrator.tools["critical"] = critical_tool
        orchestrator.tools["tool_3"] = tool_3

        # No fallback for critical
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="critical",
            status=ToolStatus.FAILED,
            error="Critical tool has no fallback"
        ))

        # Plan with critical tool
        plan = [
            {"tool": "tool_1", "params": {}, "parallel": False},
            {"tool": "critical", "params": {}, "critical": True, "parallel": False},
            {"tool": "tool_3", "params": {}, "parallel": False}
        ]

        results = await orchestrator._execute_tools(plan)

        # tool_3 should not execute after critical failure
        assert "tool_1" in execution_log
        assert "critical" in execution_log
        assert "tool_3" not in execution_log
        assert len(results) == 2  # Only first two tools attempted

    @pytest.mark.asyncio
    async def test_alternative_response_generation_on_all_tools_failure(self, orchestrator):
        """Test that orchestrator generates meaningful response even when all tools fail"""

        # Make all tools fail
        async def failing_tool(*args, **kwargs):
            raise Exception("Tool unavailable")

        orchestrator.tools = {
            "hybrid_search": failing_tool,
            "recipe_generator": failing_tool,
            "scientific_validator": failing_tool,
            "perfumer_knowledge": failing_tool
        }

        # Fallbacks also fail
        orchestrator._execute_fallback = AsyncMock(return_value=ToolExecutionResult(
            tool="any",
            status=ToolStatus.FAILED,
            error="All systems down"
        ))

        # Test orchestrate method
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate = AsyncMock(return_value=json.dumps({
            "tools": ["hybrid_search", "recipe_generator"],
            "reasoning": "Search and generate"
        }))

        # Execute
        result = await orchestrator.orchestrate(
            "Create a summer perfume",
            user_id="test_user"
        )

        # Should return error message, not crash
        assert result is not None
        assert result.get("status") == "error" or "error" in result.get("message", "").lower()
        assert result.get("message") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])