"""
Comprehensive edge case testing with parametrization
Tests boundary conditions, invalid inputs, and extreme scenarios
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import json
import random
import string
from datetime import datetime, timedelta

from fragrance_ai.tools.unified_tools import (
    UnifiedSearchTool,
    UnifiedValidatorTool,
    UnifiedGeneratorTool,
    UnifiedKnowledgeTool
)


class TestSearchEdgeCases:
    """Edge case tests for search functionality"""

    @pytest.fixture
    async def search_tool(self):
        tool = UnifiedSearchTool()
        tool.vector_store = AsyncMock()
        tool.embedding_model = AsyncMock()
        tool._initialized = True
        return tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "",  # Empty string
        " " * 1000,  # Only spaces
        "a",  # Single character
        "a" * 10000,  # Very long query
        "🌹🌺🌸",  # Only emojis
        "<script>alert('xss')</script>",  # XSS attempt
        "'; DROP TABLE fragrances; --",  # SQL injection attempt
        "\n\r\t",  # Only whitespace
        "\\x00\\x01\\x02",  # Binary characters
        None,  # None value
    ])
    async def test_invalid_search_queries(self, search_tool, query):
        """Test handling of various invalid search queries"""
        search_tool.embedding_model.encode.return_value = [[0.1, 0.2]]
        search_tool.vector_store.search.return_value = []

        # Should handle gracefully without crashing
        if query is None:
            with pytest.raises((TypeError, AttributeError)):
                await search_tool.execute(query)
        else:
            result = await search_tool.execute(query)
            assert result is not None
            assert isinstance(result, list)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [
        -1,  # Negative
        0,  # Zero
        1,  # Minimum valid
        10000,  # Very large
        float('inf'),  # Infinity
        "not_a_number",  # Invalid type
        None,  # None
        1.5,  # Float
    ])
    async def test_top_k_boundary_conditions(self, search_tool, top_k):
        """Test various top_k parameter values"""
        search_tool.embedding_model.encode.return_value = [[0.1, 0.2]]
        search_tool.vector_store.search.return_value = [
            {"id": str(i), "score": 0.9 - i * 0.1}
            for i in range(100)
        ]

        if isinstance(top_k, (int, float)) and top_k > 0 and top_k != float('inf'):
            result = await search_tool.execute("test", top_k=int(top_k))
            assert len(result) <= min(int(top_k), 100)
        else:
            # Should either handle gracefully or raise appropriate error
            try:
                result = await search_tool.execute("test", top_k=top_k)
                assert result is not None
            except (TypeError, ValueError):
                pass  # Expected for invalid types

    @pytest.mark.asyncio
    @pytest.mark.parametrize("filters", [
        {},  # Empty filters
        {"non_existent_field": "value"},  # Invalid field
        {"price": "not_a_number"},  # Invalid value type
        {"price": [-100, -50]},  # Negative price range
        {"categories": ["a" * 1000]},  # Very long category name
        {"nested": {"deeply": {"nested": {"filter": "value"}}}},  # Deep nesting
        None,  # None filters
        {"injection": "'; DROP TABLE --"},  # Injection attempt in filter
    ])
    async def test_search_filter_edge_cases(self, search_tool, filters):
        """Test various filter configurations"""
        search_tool.embedding_model.encode.return_value = [[0.1, 0.2]]
        search_tool.vector_store.search.return_value = []

        result = await search_tool.execute("test", filters=filters)
        assert result is not None
        assert isinstance(result, list)


class TestValidatorEdgeCases:
    """Edge case tests for validation functionality"""

    @pytest.fixture
    def validator_tool(self):
        return UnifiedValidatorTool()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("recipe", [
        {},  # Empty recipe
        {"notes": {}},  # Empty notes
        {"concentrations": {}},  # Empty concentrations
        {"notes": {"top": ["a" * 1000]}},  # Very long ingredient name
        {"notes": {"top": [""] * 100}},  # Many empty ingredients
        {"concentrations": {"top": -10}},  # Negative concentration
        {"concentrations": {"top": 200}},  # Over 100% concentration
        {"concentrations": {"top": "not_a_number"}},  # Invalid type
        {"notes": None},  # None notes
        {"notes": {"invalid_note_type": ["ingredient"]}},  # Invalid note type
    ])
    async def test_invalid_recipe_validation(self, validator_tool, recipe):
        """Test validation of various invalid recipes"""
        result = await validator_tool.execute(recipe)

        assert result is not None
        assert "is_valid" in result
        # Most of these should be invalid
        if recipe and "notes" in recipe and "concentrations" in recipe:
            # Might be valid depending on values
            pass
        else:
            assert result["is_valid"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("total_concentration", [
        0,  # Zero total
        50,  # Under 95%
        95,  # Minimum valid
        100,  # Perfect
        105,  # Maximum valid
        150,  # Over maximum
        1000,  # Way over
    ])
    async def test_concentration_boundaries(self, validator_tool, total_concentration):
        """Test concentration boundary values"""
        # Create recipe with specific total concentration
        if total_concentration <= 0:
            concentrations = {"top": 0, "middle": 0, "base": 0}
        else:
            # Distribute concentration
            top = min(30, total_concentration * 0.2)
            middle = min(60, total_concentration * 0.5)
            base = total_concentration - top - middle
            concentrations = {"top": top, "middle": middle, "base": base}

        recipe = {
            "notes": {
                "top": ["bergamot", "lemon"],
                "middle": ["rose", "jasmine", "ylang"],
                "base": ["sandalwood", "musk"]
            },
            "concentrations": concentrations
        }

        result = await validator_tool.execute(recipe)

        if 95 <= total_concentration <= 105:
            # Might be valid if individual concentrations are also valid
            pass
        else:
            assert result["is_valid"] is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ingredient_count", [
        0,  # No ingredients
        1,  # Too few
        5,  # Minimum valid
        15,  # Normal
        30,  # Maximum valid
        100,  # Too many
        1000,  # Way too many
    ])
    async def test_ingredient_count_boundaries(self, validator_tool, ingredient_count):
        """Test ingredient count boundaries"""
        # Generate ingredients
        if ingredient_count == 0:
            notes = {}
        else:
            # Distribute ingredients across note types
            top_count = max(1, ingredient_count // 3)
            middle_count = max(1, ingredient_count // 3)
            base_count = ingredient_count - top_count - middle_count

            notes = {
                "top": [f"ing_{i}" for i in range(top_count)],
                "middle": [f"ing_{i}" for i in range(middle_count)],
                "base": [f"ing_{i}" for i in range(base_count)]
            }

        recipe = {
            "notes": notes,
            "concentrations": {"top": 20, "middle": 50, "base": 30}
        }

        result = await validator_tool.execute(recipe)

        if 5 <= ingredient_count <= 30:
            # Should be valid
            concentration_check = next(
                v for v in result["validations"]
                if v["check"] == "ingredient_count"
            )
            assert concentration_check["passed"] is True
        else:
            assert result["is_valid"] is False


class TestGeneratorEdgeCases:
    """Edge case tests for generation functionality"""

    @pytest.fixture
    async def generator_tool(self):
        tool = UnifiedGeneratorTool()
        tool.generator = AsyncMock()
        tool._initialized = True
        return tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("request", [
        {},  # Empty request
        {"fragrance_family": ""},  # Empty family
        {"fragrance_family": "non_existent_family"},  # Invalid family
        {"mood": "😀😁😂"},  # Emoji mood
        {"intensity": "super_ultra_extreme"},  # Invalid intensity
        {"gender": "invalid_gender"},  # Invalid gender
        {"season": "not_a_season"},  # Invalid season
        {"budget_range": -1000},  # Negative budget
        {"budget_range": "free"},  # Invalid budget type
        {"complexity": 100},  # Invalid complexity value
    ])
    async def test_invalid_generation_requests(self, generator_tool, request):
        """Test generation with various invalid requests"""
        generator_tool.generator.generate.return_value = {
            "notes": {"top": ["a"], "middle": ["b"], "base": ["c"]},
            "concentrations": {"top": 20, "middle": 50, "base": 30}
        }

        result = await generator_tool.execute(request, validate=False)

        assert result is not None
        # Should either use defaults or handle gracefully
        assert "error" in result or "notes" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_count", [0, 1, 2, 3, 4, 5])
    async def test_regeneration_attempts(self, generator_tool, failure_count):
        """Test regeneration when validation fails multiple times"""
        valid_recipe = {
            "notes": {"top": ["a", "b"], "middle": ["c", "d", "e"], "base": ["f", "g"]},
            "concentrations": {"top": 20, "middle": 50, "base": 30}
        }

        invalid_recipe = {
            "notes": {"top": ["a"]},
            "concentrations": {"top": 100}
        }

        # Generate invalid recipes first, then valid
        recipes = [invalid_recipe] * failure_count + [valid_recipe]
        generator_tool.generator.generate.side_effect = recipes

        result = await generator_tool.execute(
            {"fragrance_family": "floral"},
            validate=True
        )

        assert result is not None
        if failure_count < 3:  # Max attempts is 3
            assert "validation" in result
        else:
            # Should return last attempt even if invalid
            assert "warning" in result.lower() or result.get("validation", {}).get("is_valid") is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrent_requests", [1, 5, 10, 50, 100])
    async def test_concurrent_generation_load(self, generator_tool, concurrent_requests):
        """Test system under concurrent generation load"""
        generator_tool.generator.generate.return_value = {
            "notes": {"top": ["a"], "middle": ["b"], "base": ["c"]},
            "concentrations": {"top": 20, "middle": 50, "base": 30}
        }

        # Generate many requests concurrently
        requests = [
            {"fragrance_family": f"family_{i}", "mood": f"mood_{i}"}
            for i in range(concurrent_requests)
        ]

        results = await asyncio.gather(*[
            generator_tool.execute(req, validate=False)
            for req in requests
        ])

        assert len(results) == concurrent_requests
        assert all("notes" in r or "error" in r for r in results)


class TestKnowledgeEdgeCases:
    """Edge case tests for knowledge base functionality"""

    @pytest.fixture
    async def knowledge_tool(self):
        tool = UnifiedKnowledgeTool()
        tool.rag_system = AsyncMock()
        tool.ollama_client = AsyncMock()
        tool._initialized = True
        return tool

    @pytest.mark.asyncio
    @pytest.mark.parametrize("query", [
        "",  # Empty query
        "?" * 1000,  # Only question marks
        "a" * 10000,  # Very long query
        "\x00\x01\x02",  # Binary characters
        "SELECT * FROM knowledge",  # SQL-like query
        "<script>alert('xss')</script>",  # XSS attempt
    ])
    async def test_invalid_knowledge_queries(self, knowledge_tool, query):
        """Test knowledge base with invalid queries"""
        knowledge_tool.rag_system.retrieve.return_value = []
        knowledge_tool.ollama_client.generate.return_value = "No information found"

        result = await knowledge_tool.execute(query)

        assert result is not None
        assert "answer" in result or "error" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context_size", [0, 1, 10, 100, 1000])
    async def test_varying_context_sizes(self, knowledge_tool, context_size):
        """Test with different amounts of context"""
        # Generate context of specific size
        context = [
            {"text": f"Context item {i}", "source": f"Source {i}", "score": 0.9 - i * 0.001}
            for i in range(context_size)
        ]

        knowledge_tool.rag_system.retrieve.return_value = context
        knowledge_tool.ollama_client.generate.return_value = f"Answer based on {context_size} items"

        result = await knowledge_tool.execute("Test query")

        assert result is not None
        assert "answer" in result

        if context_size > 0:
            assert len(result.get("citations", [])) > 0


class TestCachingEdgeCases:
    """Test caching behavior under edge conditions"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cache_size", [0, 1, 100, 10000])
    async def test_cache_memory_limits(self, cache_size):
        """Test cache behavior with different memory limits"""
        knowledge_tool = UnifiedKnowledgeTool()
        knowledge_tool.rag_system = AsyncMock()
        knowledge_tool.ollama_client = AsyncMock()
        knowledge_tool._initialized = True

        # Generate many unique queries
        queries = [f"query_{i}" for i in range(cache_size)]

        for query in queries:
            knowledge_tool.rag_system.retrieve.return_value = [{"text": query}]
            knowledge_tool.ollama_client.generate.return_value = f"Answer for {query}"
            await knowledge_tool.execute(query)

        # Check cache size doesn't grow unbounded
        assert len(knowledge_tool._knowledge_cache) <= 10000  # Reasonable limit

    @pytest.mark.asyncio
    async def test_cache_key_collisions(self):
        """Test cache with potentially colliding keys"""
        knowledge_tool = UnifiedKnowledgeTool()
        knowledge_tool.rag_system = AsyncMock()
        knowledge_tool.ollama_client = AsyncMock()
        knowledge_tool._initialized = True

        # Create queries that might have similar hashes
        queries = [
            ("query", "category1", True),
            ("query", "category1", False),
            ("query", "category2", True),
            ("query", None, True),
        ]

        for q, cat, citations in queries:
            knowledge_tool.rag_system.retrieve.return_value = [{"text": f"{q}-{cat}-{citations}"}]
            knowledge_tool.ollama_client.generate.return_value = f"Answer: {q}-{cat}-{citations}"
            result = await knowledge_tool.execute(q, category=cat, include_citations=citations)
            assert result["answer"] == f"Answer: {q}-{cat}-{citations}"


@pytest.mark.parametrize("data_type,value", [
    ("string", ""),
    ("string", "a" * 1000000),  # 1MB string
    ("number", float('inf')),
    ("number", float('-inf')),
    ("number", float('nan')),
    ("list", [] * 1000),
    ("list", [[]] * 1000),  # Nested empty lists
    ("dict", {}),
    ("dict", {"key": None}),
    ("special", ...),  # Ellipsis
])
@pytest.mark.asyncio
async def test_data_type_edge_cases(data_type, value):
    """Test tools with various edge case data types"""
    search_tool = UnifiedSearchTool()
    search_tool.vector_store = AsyncMock()
    search_tool.embedding_model = AsyncMock()
    search_tool._initialized = True

    # Should handle without crashing
    try:
        if data_type == "string" and isinstance(value, str):
            await search_tool.execute(value)
        elif data_type == "list":
            search_tool.embedding_model.encode.return_value = value
            await search_tool.execute("test")
        elif data_type == "dict":
            await search_tool.execute("test", filters=value)
    except (TypeError, ValueError, AttributeError):
        pass  # Expected for some edge cases


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])