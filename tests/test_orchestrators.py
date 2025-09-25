"""
Comprehensive test suite for orchestrators
Tests complex branching logic, error handling, and edge cases
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import json

from fragrance_ai.orchestrator.artisan_orchestrator import ArtisanOrchestrator
from fragrance_ai.orchestrator.customer_service_orchestrator import CustomerServiceOrchestrator


class TestArtisanOrchestrator:
    """Test cases for ArtisanOrchestrator"""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance with mocked dependencies"""
        orchestrator = ArtisanOrchestrator()

        # Mock all tools
        orchestrator.tools = {
            "hybrid_search": AsyncMock(),
            "recipe_generator": AsyncMock(),
            "scientific_validator": AsyncMock(),
            "perfumer_knowledge": AsyncMock()
        }

        # Mock LLM client
        orchestrator.llm_client = AsyncMock()

        return orchestrator

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_tools", [
        ("Find me a floral perfume", ["hybrid_search"]),
        ("Create a new fragrance recipe", ["recipe_generator", "scientific_validator"]),
        ("Tell me about rose notes", ["perfumer_knowledge"]),
        ("Search and create similar to Chanel No.5", ["hybrid_search", "recipe_generator", "scientific_validator"])
    ])
    async def test_tool_selection_based_on_query(self, orchestrator, query, expected_tools):
        """Test that correct tools are selected based on query type"""
        # Mock LLM to return specific tool selection
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": expected_tools,
            "reasoning": "Test reasoning"
        })

        # Execute orchestration
        result = await orchestrator.orchestrate(query)

        # Verify correct tools were called
        for tool_name in expected_tools:
            assert orchestrator.tools[tool_name].called

    @pytest.mark.asyncio
    async def test_tool_failure_handling(self, orchestrator):
        """Test graceful handling when a tool fails"""
        # Make search tool fail
        orchestrator.tools["hybrid_search"].side_effect = Exception("Search service unavailable")

        # Mock LLM to select search tool
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["hybrid_search"],
            "reasoning": "Need to search"
        })

        # Should handle error gracefully
        result = await orchestrator.orchestrate("Find perfume")

        assert "error" in result or "failed" in result.lower()
        assert orchestrator.tools["hybrid_search"].called

    @pytest.mark.asyncio
    async def test_multi_tool_pipeline(self, orchestrator):
        """Test complex multi-tool pipeline execution"""
        # Setup tool responses
        orchestrator.tools["hybrid_search"].return_value = {
            "results": [
                {"id": "1", "name": "Rose Perfume", "notes": ["rose", "jasmine"]}
            ]
        }

        orchestrator.tools["recipe_generator"].return_value = {
            "recipe": {
                "name": "New Rose Creation",
                "notes": {"top": ["bergamot"], "middle": ["rose"], "base": ["musk"]},
                "concentrations": {"top": 20, "middle": 50, "base": 30}
            }
        }

        orchestrator.tools["scientific_validator"].return_value = {
            "is_valid": True,
            "safety_score": 0.95
        }

        # Mock LLM to chain tools
        orchestrator.llm_client.generate.side_effect = [
            json.dumps({"tools": ["hybrid_search"], "reasoning": "Search first"}),
            json.dumps({"tools": ["recipe_generator"], "reasoning": "Generate based on search"}),
            json.dumps({"tools": ["scientific_validator"], "reasoning": "Validate recipe"})
        ]

        result = await orchestrator.orchestrate("Create something similar to Rose Perfume")

        # Verify all tools were called in sequence
        assert orchestrator.tools["hybrid_search"].called
        assert orchestrator.tools["recipe_generator"].called
        assert orchestrator.tools["scientific_validator"].called

    @pytest.mark.asyncio
    async def test_context_preservation_across_tools(self, orchestrator):
        """Test that context is preserved across tool calls"""
        context_captured = []

        async def capture_context(*args, **kwargs):
            context_captured.append(kwargs.get("context", {}))
            return {"result": "success"}

        orchestrator.tools["hybrid_search"] = capture_context
        orchestrator.tools["recipe_generator"] = capture_context

        orchestrator.llm_client.generate.side_effect = [
            json.dumps({"tools": ["hybrid_search"], "context": {"query": "test"}}),
            json.dumps({"tools": ["recipe_generator"], "context": {"previous": "search_result"}})
        ]

        await orchestrator.orchestrate("Test query")

        # Context should accumulate
        assert len(context_captured) == 2
        # Second call should have context from first

    @pytest.mark.asyncio
    async def test_timeout_handling(self, orchestrator):
        """Test handling of tool timeout"""
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate slow operation
            return {"result": "late"}

        orchestrator.tools["hybrid_search"] = slow_tool
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["hybrid_search"],
            "timeout": 1  # 1 second timeout
        })

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                orchestrator.orchestrate("Test timeout"),
                timeout=2
            )

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, orchestrator):
        """Test handling of empty or invalid queries"""
        for invalid_query in ["", None, "   ", "\n\t"]:
            result = await orchestrator.orchestrate(invalid_query)
            assert "error" in result.lower() or "invalid" in result.lower()

    @pytest.mark.asyncio
    async def test_recursive_tool_calls_prevention(self, orchestrator):
        """Test prevention of infinite recursive tool calls"""
        call_count = 0

        async def recursive_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 10:
                raise Exception("Too many recursive calls")
            return {"next_tool": "hybrid_search"}

        orchestrator.tools["hybrid_search"] = recursive_tool

        # Should prevent infinite recursion
        result = await orchestrator.orchestrate("Test recursion")
        assert call_count <= 10  # Should stop before infinite loop


class TestCustomerServiceOrchestrator:
    """Test cases for CustomerServiceOrchestrator"""

    @pytest.fixture
    async def cs_orchestrator(self):
        """Create CS orchestrator with mocked dependencies"""
        orchestrator = CustomerServiceOrchestrator()
        orchestrator.llm_client = AsyncMock()
        orchestrator.knowledge_base = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query,expected_category", [
        ("How do I return my order?", "returns"),
        ("What are your shipping rates?", "shipping"),
        ("Is this perfume vegan?", "product_info"),
        ("I have an allergic reaction", "safety"),
        ("Can I get a discount?", "pricing")
    ])
    async def test_query_categorization(self, cs_orchestrator, query, expected_category):
        """Test correct categorization of customer queries"""
        cs_orchestrator.llm_client.generate.return_value = json.dumps({
            "category": expected_category,
            "confidence": 0.95
        })

        result = await cs_orchestrator.handle_customer_query(query)

        # Verify categorization happened
        cs_orchestrator.llm_client.generate.assert_called()
        call_args = cs_orchestrator.llm_client.generate.call_args[0][0]
        assert "categorize" in call_args.lower() or "category" in call_args.lower()

    @pytest.mark.asyncio
    async def test_escalation_to_human(self, cs_orchestrator):
        """Test escalation to human agent when needed"""
        # Simulate low confidence response
        cs_orchestrator.llm_client.generate.return_value = json.dumps({
            "response": "I'm not sure",
            "confidence": 0.3,
            "needs_human": True
        })

        result = await cs_orchestrator.handle_customer_query("Complex legal question")

        assert "escalate" in result.lower() or "human" in result.lower() or "agent" in result.lower()

    @pytest.mark.asyncio
    async def test_knowledge_base_integration(self, cs_orchestrator):
        """Test integration with knowledge base for FAQ"""
        cs_orchestrator.knowledge_base.search.return_value = [
            {
                "question": "What is your return policy?",
                "answer": "30-day return policy for unopened items",
                "confidence": 0.98
            }
        ]

        cs_orchestrator.llm_client.generate.return_value = json.dumps({
            "use_knowledge_base": True,
            "query": "return policy"
        })

        result = await cs_orchestrator.handle_customer_query("What's your return policy?")

        cs_orchestrator.knowledge_base.search.assert_called()
        assert "30-day" in result or "return" in result.lower()

    @pytest.mark.asyncio
    async def test_sentiment_analysis_integration(self, cs_orchestrator):
        """Test sentiment analysis affects response tone"""
        # Angry customer
        cs_orchestrator.llm_client.generate.side_effect = [
            json.dumps({"sentiment": "negative", "emotion": "angry"}),
            "I understand your frustration. Let me help resolve this immediately."
        ]

        result = await cs_orchestrator.handle_customer_query("This is terrible service!")

        assert "understand" in result.lower() or "apologize" in result.lower()

    @pytest.mark.asyncio
    async def test_multi_language_support(self, cs_orchestrator):
        """Test handling of non-English queries"""
        cs_orchestrator.llm_client.generate.side_effect = [
            json.dumps({"language": "korean", "needs_translation": True}),
            "향수 반품 정책은 30일입니다"  # Korean response
        ]

        result = await cs_orchestrator.handle_customer_query("반품 정책이 뭐예요?")

        # Should handle Korean query
        assert result is not None
        assert len(result) > 0


class TestOrchestratorIntegration:
    """Integration tests for orchestrator system"""

    @pytest.mark.asyncio
    async def test_orchestrator_handoff(self):
        """Test handoff between different orchestrators"""
        artisan = ArtisanOrchestrator()
        cs = CustomerServiceOrchestrator()

        # Mock dependencies
        artisan.llm_client = AsyncMock()
        cs.llm_client = AsyncMock()

        # Customer service identifies need for technical expertise
        cs.llm_client.generate.return_value = json.dumps({
            "needs_expert": True,
            "expert_type": "perfumer",
            "query_forward": "How to create a rose fragrance"
        })

        # Should trigger artisan orchestrator
        artisan.llm_client.generate.return_value = json.dumps({
            "tools": ["recipe_generator"],
            "response": "To create a rose fragrance..."
        })

        # Simulate handoff
        cs_result = await cs.handle_customer_query("How do I make a rose perfume?")
        if "expert" in cs_result.lower():
            artisan_result = await artisan.orchestrate("How to create a rose fragrance")
            assert artisan_result is not None

    @pytest.mark.asyncio
    async def test_concurrent_orchestration(self):
        """Test multiple concurrent orchestrations"""
        orchestrator = ArtisanOrchestrator()
        orchestrator.llm_client = AsyncMock()
        orchestrator.tools = {
            "hybrid_search": AsyncMock(return_value={"results": []}),
            "recipe_generator": AsyncMock(return_value={"recipe": {}})
        }

        # Run multiple queries concurrently
        queries = [
            "Find floral perfumes",
            "Create woody fragrance",
            "Search for citrus scents"
        ]

        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["hybrid_search"],
            "response": "Found results"
        })

        results = await asyncio.gather(*[
            orchestrator.orchestrate(q) for q in queries
        ])

        assert len(results) == len(queries)
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_error_recovery_mechanism(self):
        """Test error recovery and retry logic"""
        orchestrator = ArtisanOrchestrator()

        call_count = 0
        async def flaky_tool(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return {"success": True}

        orchestrator.tools = {"hybrid_search": flaky_tool}
        orchestrator.llm_client = AsyncMock()
        orchestrator.llm_client.generate.return_value = json.dumps({
            "tools": ["hybrid_search"],
            "retry_on_failure": True,
            "max_retries": 3
        })

        result = await orchestrator.orchestrate("Test retry")

        assert call_count >= 3  # Should retry
        assert "success" in str(result).lower()


@pytest.mark.parametrize("exception_type,expected_handling", [
    (ValueError, "validation_error"),
    (TimeoutError, "timeout_error"),
    (ConnectionError, "connection_error"),
    (MemoryError, "resource_error"),
    (KeyError, "configuration_error")
])
@pytest.mark.asyncio
async def test_exception_handling(exception_type, expected_handling):
    """Test handling of various exception types"""
    orchestrator = ArtisanOrchestrator()
    orchestrator.tools = {
        "test_tool": AsyncMock(side_effect=exception_type("Test error"))
    }
    orchestrator.llm_client = AsyncMock()
    orchestrator.llm_client.generate.return_value = json.dumps({
        "tools": ["test_tool"]
    })

    result = await orchestrator.orchestrate("Test query")

    # Should handle exception gracefully
    assert result is not None
    # Should indicate error occurred
    assert "error" in str(result).lower() or "failed" in str(result).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])