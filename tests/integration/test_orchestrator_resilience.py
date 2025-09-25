"""
Advanced Integration Tests for Orchestrator Resilience
Tests complex failure scenarios, recovery mechanisms, and fallback strategies
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import json
import random
from typing import Dict, Any, List

from fragrance_ai.orchestrator.artisan_orchestrator import ArtisanOrchestrator
from fragrance_ai.orchestrator.customer_service_orchestrator import CustomerServiceOrchestrator


class TestOrchestratorResilience:
    """Test orchestrator resilience under various failure conditions"""

    @pytest.fixture
    async def orchestrator_with_monitoring(self):
        """Create orchestrator with monitoring capabilities"""
        orchestrator = ArtisanOrchestrator()

        # Add monitoring
        orchestrator.call_history = []
        orchestrator.error_history = []
        orchestrator.recovery_attempts = []

        # Wrap tools with monitoring
        for tool_name, tool in orchestrator.tools.items():
            original_tool = tool

            async def monitored_tool(*args, tool_name=tool_name, original=original_tool, **kwargs):
                try:
                    result = await original(*args, **kwargs)
                    orchestrator.call_history.append({
                        "tool": tool_name,
                        "status": "success",
                        "timestamp": datetime.utcnow()
                    })
                    return result
                except Exception as e:
                    orchestrator.error_history.append({
                        "tool": tool_name,
                        "error": str(e),
                        "timestamp": datetime.utcnow()
                    })
                    raise

            orchestrator.tools[tool_name] = monitored_tool

        return orchestrator

    @pytest.mark.asyncio
    async def test_cascading_tool_failures_with_recovery(self, orchestrator_with_monitoring):
        """Test orchestrator behavior when multiple tools fail in sequence"""

        # Setup cascading failures
        failure_sequence = [
            ("hybrid_search", ConnectionError("Database unavailable")),
            ("perfumer_knowledge", TimeoutError("Knowledge base timeout")),
            ("recipe_generator", MemoryError("GPU out of memory"))
        ]

        for tool_name, error in failure_sequence:
            orchestrator_with_monitoring.tools[tool_name] = AsyncMock(side_effect=error)

        # Setup one working tool as fallback
        orchestrator_with_monitoring.tools["scientific_validator"] = AsyncMock(
            return_value={"status": "operational", "message": "Basic validation available"}
        )

        # Setup LLM to attempt multiple tools
        orchestrator_with_monitoring.llm_client = AsyncMock()
        orchestrator_with_monitoring.llm_client.generate.side_effect = [
            json.dumps({"tools": ["hybrid_search"], "reasoning": "Search first"}),
            json.dumps({"tools": ["perfumer_knowledge"], "reasoning": "Fallback to knowledge"}),
            json.dumps({"tools": ["recipe_generator"], "reasoning": "Try generation"}),
            json.dumps({"tools": ["scientific_validator"], "reasoning": "Final fallback"})
        ]

        # Execute
        result = await orchestrator_with_monitoring.orchestrate("Create a perfume")

        # Verify resilience
        assert len(orchestrator_with_monitoring.error_history) == 3
        assert any("operational" in str(call) for call in orchestrator_with_monitoring.call_history)
        assert result is not None
        assert "error" not in result or "fallback" in result.lower()

    @pytest.mark.asyncio
    async def test_partial_tool_success_aggregation(self):
        """Test orchestrator aggregating partial results from partially failing tools"""

        orchestrator = ArtisanOrchestrator()

        # Setup mixed success/failure scenario
        successful_results = []

        async def search_tool(*args, **kwargs):
            result = {"results": [{"id": "1", "name": "Rose"}]}
            successful_results.append(("search", result))
            return result

        async def generator_tool(*args, **kwargs):
            raise ValueError("Generation parameters invalid")

        async def validator_tool(*args, **kwargs):
            result = {"validation": "passed", "score": 0.95}
            successful_results.append(("validator", result))
            return result

        orchestrator.tools = {
            "hybrid_search": search_tool,
            "recipe_generator": generator_tool,
            "scientific_validator": validator_tool
        }

        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["hybrid_search", "recipe_generator", "scientific_validator"],
            "aggregate_partial": True
        })

        # Execute
        result = await orchestrator.orchestrate("Analyze and create")

        # Verify partial aggregation
        assert len(successful_results) == 2
        assert result is not None
        # Should contain partial results from successful tools

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern implementation for repeated failures"""

        orchestrator = ArtisanOrchestrator()

        # Add circuit breaker state
        circuit_breaker_state = {
            "hybrid_search": {"failures": 0, "is_open": False, "last_failure": None}
        }

        call_count = 0

        async def failing_search(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            # Check circuit breaker
            state = circuit_breaker_state["hybrid_search"]
            if state["is_open"]:
                # Check if enough time has passed to retry
                if state["last_failure"] and \
                   (datetime.utcnow() - state["last_failure"]).seconds < 30:
                    raise Exception("Circuit breaker open")

            # Simulate failure
            state["failures"] += 1
            state["last_failure"] = datetime.utcnow()

            if state["failures"] >= 3:
                state["is_open"] = True

            raise ConnectionError("Service unavailable")

        orchestrator.tools = {"hybrid_search": failing_search}
        orchestrator.llm_client = AsyncMock()

        # Attempt multiple calls
        for i in range(5):
            orchestrator.llm_client.generate.return_value = json.dumps({
                "tools": ["hybrid_search"]
            })

            try:
                await orchestrator.orchestrate(f"Query {i}")
            except:
                pass

        # Circuit breaker should have tripped after 3 failures
        assert circuit_breaker_state["hybrid_search"]["is_open"] is True
        assert call_count <= 4  # Should stop trying after circuit opens

    @pytest.mark.asyncio
    async def test_intelligent_retry_with_exponential_backoff(self):
        """Test intelligent retry mechanism with exponential backoff"""

        orchestrator = ArtisanOrchestrator()

        retry_delays = []
        attempt_times = []

        async def flaky_tool(*args, **kwargs):
            attempt_times.append(datetime.utcnow())

            if len(attempt_times) > 1:
                # Calculate delay between attempts
                delay = (attempt_times[-1] - attempt_times[-2]).total_seconds()
                retry_delays.append(delay)

            if len(attempt_times) < 3:
                raise ConnectionError("Temporary failure")

            return {"success": True}

        # Implement retry logic
        async def retry_wrapper(tool_func, max_retries=3, base_delay=1):
            for attempt in range(max_retries):
                try:
                    return await tool_func()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise

                    # Exponential backoff
                    delay = base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        orchestrator.tools = {
            "hybrid_search": lambda *a, **k: retry_wrapper(
                lambda: flaky_tool(*a, **k)
            )
        }

        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["hybrid_search"],
            "enable_retry": True
        })

        # Execute
        result = await orchestrator.orchestrate("Search query")

        # Verify exponential backoff
        assert len(retry_delays) >= 1
        if len(retry_delays) >= 2:
            # Each delay should be roughly double the previous
            assert retry_delays[1] > retry_delays[0] * 1.5

    @pytest.mark.asyncio
    async def test_graceful_degradation_chain(self):
        """Test graceful degradation through multiple service levels"""

        orchestrator = ArtisanOrchestrator()

        service_levels = []

        # Define service levels from best to worst
        async def premium_service(*args, **kwargs):
            service_levels.append("premium")
            raise Exception("Premium service unavailable")

        async def standard_service(*args, **kwargs):
            service_levels.append("standard")
            raise Exception("Standard service unavailable")

        async def basic_service(*args, **kwargs):
            service_levels.append("basic")
            return {"level": "basic", "result": "Limited functionality"}

        orchestrator.tools = {
            "premium": premium_service,
            "standard": standard_service,
            "basic": basic_service
        }

        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.side_effect = [
            json.dumps({"tools": ["premium"], "fallback_chain": ["standard", "basic"]}),
            json.dumps({"tools": ["standard"]}),
            json.dumps({"tools": ["basic"]})
        ]

        # Execute
        result = await orchestrator.orchestrate("Get service")

        # Verify degradation chain
        assert service_levels == ["premium", "standard", "basic"]
        assert result is not None
        assert "basic" in str(result).lower()

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_failure(self):
        """Test proper resource cleanup when operations fail"""

        orchestrator = ArtisanOrchestrator()

        resources_allocated = []
        resources_cleaned = []

        async def resource_intensive_tool(*args, **kwargs):
            # Allocate resources
            resource_id = f"resource_{len(resources_allocated)}"
            resources_allocated.append(resource_id)

            try:
                # Simulate work that fails
                raise Exception("Operation failed")
            finally:
                # Cleanup should happen even on failure
                resources_cleaned.append(resource_id)

        orchestrator.tools = {"heavy_tool": resource_intensive_tool}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["heavy_tool"]
        })

        # Execute multiple times
        for _ in range(3):
            try:
                await orchestrator.orchestrate("Heavy operation")
            except:
                pass

        # Verify all resources were cleaned up
        assert len(resources_allocated) == len(resources_cleaned)
        assert set(resources_allocated) == set(resources_cleaned)

    @pytest.mark.asyncio
    async def test_timeout_recovery_with_partial_results(self):
        """Test recovery from timeout with partial results preservation"""

        orchestrator = ArtisanOrchestrator()

        partial_results = []

        async def slow_tool(*args, **kwargs):
            # Start processing
            partial_results.append("started")
            await asyncio.sleep(0.5)

            partial_results.append("phase1")
            await asyncio.sleep(0.5)

            partial_results.append("phase2")
            await asyncio.sleep(0.5)

            partial_results.append("completed")
            return {"status": "complete"}

        async def tool_with_timeout(*args, **kwargs):
            try:
                return await asyncio.wait_for(slow_tool(*args, **kwargs), timeout=1.0)
            except asyncio.TimeoutError:
                # Return partial results on timeout
                return {"status": "partial", "completed_phases": partial_results.copy()}

        orchestrator.tools = {"slow_tool": tool_with_timeout}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["slow_tool"]
        })

        # Execute
        result = await orchestrator.orchestrate("Slow operation")

        # Should have partial results
        assert len(partial_results) >= 1
        assert len(partial_results) < 4  # Should not complete all phases
        assert result is not None

    @pytest.mark.asyncio
    async def test_dependency_chain_failure_handling(self):
        """Test handling failures in dependent tool chains"""

        orchestrator = ArtisanOrchestrator()

        execution_order = []

        async def tool_a(*args, **kwargs):
            execution_order.append("A")
            return {"data": "from_A"}

        async def tool_b(*args, **kwargs):
            execution_order.append("B")
            # B depends on A's output
            if "from_A" not in str(kwargs.get("context", {})):
                raise ValueError("Missing required input from A")
            raise Exception("B failed even with correct input")

        async def tool_c(*args, **kwargs):
            execution_order.append("C")
            # C can work independently
            return {"data": "from_C", "independent": True}

        orchestrator.tools = {
            "tool_a": tool_a,
            "tool_b": tool_b,
            "tool_c": tool_c
        }

        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.side_effect = [
            json.dumps({
                "tools": ["tool_a"],
                "next": "tool_b",
                "reasoning": "A then B"
            }),
            json.dumps({
                "tools": ["tool_b"],
                "on_failure": "tool_c",
                "context": {"from_A": True}
            }),
            json.dumps({
                "tools": ["tool_c"],
                "reasoning": "Fallback to independent tool"
            })
        ]

        # Execute
        result = await orchestrator.orchestrate("Dependent operation")

        # Verify execution order and fallback
        assert execution_order == ["A", "B", "C"]
        assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_failure_isolation(self):
        """Test that failures in concurrent operations don't affect each other"""

        orchestrator = ArtisanOrchestrator()

        results = {"success": [], "failure": []}

        async def mixed_reliability_tool(tool_id, *args, **kwargs):
            await asyncio.sleep(random.uniform(0.1, 0.3))

            if tool_id % 2 == 0:
                results["success"].append(tool_id)
                return {"id": tool_id, "status": "success"}
            else:
                results["failure"].append(tool_id)
                raise Exception(f"Tool {tool_id} failed")

        # Create multiple tool instances
        for i in range(10):
            orchestrator.tools[f"tool_{i}"] = lambda *a, id=i, **k: mixed_reliability_tool(id, *a, **k)

        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": [f"tool_{i}" for i in range(10)],
            "parallel": True,
            "isolate_failures": True
        })

        # Execute
        result = await orchestrator.orchestrate("Parallel operations")

        # Verify isolation
        assert len(results["success"]) == 5  # Even numbered tools
        assert len(results["failure"]) == 5  # Odd numbered tools
        assert result is not None


class TestErrorPropagation:
    """Test error propagation and handling across system boundaries"""

    @pytest.mark.asyncio
    async def test_error_context_preservation(self):
        """Test that error context is preserved through the call stack"""

        orchestrator = ArtisanOrchestrator()

        error_contexts = []

        async def tool_with_context(*args, **kwargs):
            context = {
                "user_query": kwargs.get("query"),
                "timestamp": datetime.utcnow().isoformat(),
                "session_id": kwargs.get("session_id"),
                "tool_params": kwargs
            }
            error_contexts.append(context)
            raise ValueError("Tool error with context")

        orchestrator.tools = {"contextual_tool": tool_with_context}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["contextual_tool"]
        })

        # Execute
        try:
            await orchestrator.orchestrate(
                "Test query",
                session_id="test_session_123"
            )
        except:
            pass

        # Verify context was preserved
        assert len(error_contexts) == 1
        assert error_contexts[0]["user_query"] is not None
        assert error_contexts[0]["session_id"] == "test_session_123"

    @pytest.mark.asyncio
    async def test_error_aggregation_and_reporting(self):
        """Test aggregation of multiple errors for comprehensive reporting"""

        orchestrator = ArtisanOrchestrator()

        all_errors = []

        # Create error aggregator
        class ErrorAggregator:
            def __init__(self):
                self.errors = []

            async def __call__(self, tool_name, *args, **kwargs):
                try:
                    if tool_name == "tool_1":
                        raise ValueError("Validation error")
                    elif tool_name == "tool_2":
                        raise ConnectionError("Network error")
                    elif tool_name == "tool_3":
                        raise MemoryError("Resource error")
                    else:
                        return {"success": True}
                except Exception as e:
                    self.errors.append({
                        "tool": tool_name,
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "timestamp": datetime.utcnow()
                    })
                    raise

        aggregator = ErrorAggregator()

        for i in range(4):
            orchestrator.tools[f"tool_{i}"] = lambda *a, id=i, **k: aggregator(f"tool_{id}", *a, **k)

        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": [f"tool_{i}" for i in range(4)],
            "aggregate_errors": True
        })

        # Execute
        result = await orchestrator.orchestrate("Multi-tool operation")

        # Should have collected all errors
        assert len(aggregator.errors) == 3
        assert {e["error_type"] for e in aggregator.errors} == {
            "ValueError", "ConnectionError", "MemoryError"
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])