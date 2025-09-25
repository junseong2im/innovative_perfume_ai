"""
Unit tests for unified tools
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from fragrance_ai.tools.unified_tools import (
    UnifiedSearchTool,
    UnifiedKnowledgeTool,
    UnifiedValidatorTool,
    UnifiedGeneratorTool
)


class TestUnifiedSearchTool:
    """Test cases for UnifiedSearchTool"""

    @pytest.fixture
    async def search_tool(self):
        """Create search tool instance"""
        tool = UnifiedSearchTool()
        return tool

    @pytest.mark.asyncio
    async def test_vector_search(self, search_tool):
        """Test vector search functionality"""
        with patch.object(search_tool, 'initialize', new=AsyncMock()):
            with patch.object(search_tool, 'vector_store') as mock_store:
                with patch.object(search_tool, 'embedding_model') as mock_embed:
                    # Setup mocks
                    mock_embed.encode = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
                    mock_store.search = AsyncMock(return_value=[
                        {"id": "1", "name": "Rose Perfume", "score": 0.95},
                        {"id": "2", "name": "Jasmine Scent", "score": 0.85}
                    ])

                    # Execute search
                    results = await search_tool.execute(
                        query="floral fragrance",
                        search_type="vector",
                        top_k=5
                    )

                    # Assertions
                    assert len(results) == 2
                    assert results[0]["name"] == "Rose Perfume"
                    assert results[0]["score"] == 0.95
                    mock_embed.encode.assert_called_once_with(["floral fragrance"])

    @pytest.mark.asyncio
    async def test_hybrid_search_deduplication(self, search_tool):
        """Test hybrid search with deduplication"""
        with patch.object(search_tool, 'initialize', new=AsyncMock()):
            # Mock duplicate results from different sources
            vector_results = [
                {"id": "1", "name": "Rose", "score": 0.9},
                {"id": "2", "name": "Jasmine", "score": 0.8}
            ]
            keyword_results = [
                {"id": "1", "name": "Rose", "score": 0.7},
                {"id": "3", "name": "Lavender", "score": 0.6}
            ]

            merged = search_tool._merge_results(vector_results, keyword_results)

            # Should have 3 unique results
            assert len(merged) == 3
            # Rose should have boosted score from being in both
            rose_result = next(r for r in merged if r["id"] == "1")
            assert rose_result["score"] > 0.9  # Boosted from combination

    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_tool):
        """Test error handling in search"""
        with patch.object(search_tool, 'initialize', AsyncMock(side_effect=Exception("DB Error"))):
            results = await search_tool.execute("test query")
            assert results == []  # Should return empty list on error


class TestUnifiedValidatorTool:
    """Test cases for UnifiedValidatorTool"""

    @pytest.fixture
    def validator_tool(self):
        """Create validator tool instance"""
        return UnifiedValidatorTool()

    @pytest.mark.asyncio
    async def test_valid_recipe_validation(self, validator_tool):
        """Test validation of valid recipe"""
        valid_recipe = {
            "notes": {
                "top": ["bergamot", "lemon"],
                "middle": ["rose", "jasmine", "ylang"],
                "base": ["sandalwood", "musk"]
            },
            "concentrations": {
                "top": 25,
                "middle": 45,
                "base": 30
            }
        }

        result = await validator_tool.execute(valid_recipe)

        assert result["is_valid"] is True
        assert result["validation_level"] == "standard"
        assert all(v["passed"] for v in result["validations"][:2])

    @pytest.mark.asyncio
    async def test_invalid_concentration_validation(self, validator_tool):
        """Test validation of recipe with invalid concentrations"""
        invalid_recipe = {
            "notes": {
                "top": ["bergamot"],
                "middle": ["rose"],
                "base": ["sandalwood"]
            },
            "concentrations": {
                "top": 5,  # Too low (min is 15)
                "middle": 80,  # Too high (max is 60)
                "base": 15  # Too low (min is 20)
            }
        }

        result = await validator_tool.execute(invalid_recipe)

        assert result["is_valid"] is False
        concentration_check = next(
            v for v in result["validations"]
            if v["check"] == "concentration_validation"
        )
        assert not concentration_check["passed"]
        assert len(concentration_check["issues"]) > 0

    @pytest.mark.asyncio
    async def test_ingredient_count_validation(self, validator_tool):
        """Test validation of ingredient count"""
        too_few_ingredients = {
            "notes": {
                "top": ["bergamot"],
                "middle": ["rose"],
                "base": ["musk"]
            },
            "concentrations": {"top": 20, "middle": 50, "base": 30}
        }

        result = await validator_tool.execute(too_few_ingredients)

        ingredient_check = next(
            v for v in result["validations"]
            if v["check"] == "ingredient_count"
        )
        assert not ingredient_check["passed"]
        assert "Too few ingredients" in ingredient_check["issues"][0]

    @pytest.mark.asyncio
    async def test_strict_validation_level(self, validator_tool):
        """Test strict validation level"""
        recipe = {
            "notes": {"top": ["a", "b"], "middle": ["c", "d", "e"], "base": ["f", "g"]},
            "concentrations": {"top": 20, "middle": 50, "base": 30}
        }

        result = await validator_tool.execute(recipe, validation_level="strict")

        # Should include IFRA compliance check in strict mode
        ifra_check = next(
            (v for v in result["validations"] if v["check"] == "ifra_compliance"),
            None
        )
        assert ifra_check is not None


class TestUnifiedKnowledgeTool:
    """Test cases for UnifiedKnowledgeTool"""

    @pytest.fixture
    async def knowledge_tool(self):
        """Create knowledge tool instance"""
        tool = UnifiedKnowledgeTool()
        return tool

    @pytest.mark.asyncio
    async def test_knowledge_query_with_cache(self, knowledge_tool):
        """Test knowledge query with caching"""
        with patch.object(knowledge_tool, 'initialize', new=AsyncMock()):
            with patch.object(knowledge_tool, 'rag_system') as mock_rag:
                with patch.object(knowledge_tool, 'ollama_client') as mock_llm:
                    # Setup mocks
                    mock_rag.retrieve = AsyncMock(return_value=[
                        {"text": "Rose is a floral note", "source": "Book1", "score": 0.9}
                    ])
                    mock_llm.generate = AsyncMock(return_value="Rose is a classic floral note.")

                    # First query
                    result1 = await knowledge_tool.execute("What is rose?")
                    assert result1["answer"] == "Rose is a classic floral note."
                    assert mock_rag.retrieve.call_count == 1

                    # Second identical query should use cache
                    result2 = await knowledge_tool.execute("What is rose?")
                    assert result2["answer"] == "Rose is a classic floral note."
                    assert mock_rag.retrieve.call_count == 1  # Not called again

    @pytest.mark.asyncio
    async def test_knowledge_with_citations(self, knowledge_tool):
        """Test knowledge query with citations"""
        with patch.object(knowledge_tool, 'initialize', new=AsyncMock()):
            with patch.object(knowledge_tool, 'rag_system') as mock_rag:
                with patch.object(knowledge_tool, 'ollama_client') as mock_llm:
                    mock_rag.retrieve = AsyncMock(return_value=[
                        {"text": "Text 1", "source": "Source A", "score": 0.95},
                        {"text": "Text 2", "source": "Source B", "score": 0.85}
                    ])
                    mock_llm.generate = AsyncMock(return_value="Answer")

                    result = await knowledge_tool.execute(
                        "Query",
                        include_citations=True
                    )

                    assert "citations" in result
                    assert len(result["citations"]) == 2
                    assert result["citations"][0]["source"] == "Source A"
                    assert result["citations"][0]["relevance"] == 0.95


class TestUnifiedGeneratorTool:
    """Test cases for UnifiedGeneratorTool"""

    @pytest.fixture
    async def generator_tool(self):
        """Create generator tool instance"""
        tool = UnifiedGeneratorTool()
        return tool

    @pytest.mark.asyncio
    async def test_basic_generation(self, generator_tool):
        """Test basic recipe generation"""
        with patch.object(generator_tool, 'initialize', new=AsyncMock()):
            with patch.object(generator_tool, 'generator') as mock_gen:
                mock_recipe = {
                    "notes": {"top": ["a"], "middle": ["b", "c"], "base": ["d"]},
                    "concentrations": {"top": 20, "middle": 50, "base": 30}
                }
                mock_gen.generate = AsyncMock(return_value=mock_recipe)

                result = await generator_tool.execute(
                    {"fragrance_family": "floral", "mood": "romantic"},
                    validate=False
                )

                assert result["notes"] == mock_recipe["notes"]
                assert "metadata" in result
                assert result["metadata"]["validated"] is False

    @pytest.mark.asyncio
    async def test_generation_with_validation(self, generator_tool):
        """Test generation with validation"""
        with patch.object(generator_tool, 'initialize', new=AsyncMock()):
            with patch.object(generator_tool, 'generator') as mock_gen:
                with patch.object(generator_tool.validator, 'execute', new=AsyncMock()) as mock_val:
                    mock_recipe = {
                        "notes": {"top": ["a", "b"], "middle": ["c", "d", "e"], "base": ["f", "g"]},
                        "concentrations": {"top": 20, "middle": 50, "base": 30}
                    }
                    mock_gen.generate = AsyncMock(return_value=mock_recipe)
                    mock_val.return_value = {"is_valid": True, "validations": []}

                    result = await generator_tool.execute(
                        {"fragrance_family": "woody"},
                        validate=True
                    )

                    assert "validation" in result
                    assert result["validation"]["is_valid"] is True
                    mock_val.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_with_enhancement(self, generator_tool):
        """Test generation with enhancements"""
        with patch.object(generator_tool, 'initialize', new=AsyncMock()):
            with patch.object(generator_tool, 'generator') as mock_gen:
                mock_recipe = {
                    "notes": {"top": ["a"], "middle": ["b", "c"], "base": ["d"]},
                    "concentrations": {"top": 20, "middle": 50, "base": 30}
                }
                mock_gen.generate = AsyncMock(return_value=mock_recipe)

                result = await generator_tool.execute(
                    {
                        "fragrance_family": "oriental",
                        "korean_region": "Jeju",
                        "budget_range": "luxury"
                    },
                    validate=False,
                    enhance=True
                )

                assert "korean_elements" in result
                assert result["korean_elements"]["region"] == "Jeju"
                assert "estimated_cost" in result
                assert result["estimated_cost"]["per_100ml"] == 1500  # Luxury range
                assert "complexity_score" in result

    @pytest.mark.asyncio
    async def test_regeneration_on_invalid(self, generator_tool):
        """Test regeneration when validation fails"""
        with patch.object(generator_tool, 'initialize', new=AsyncMock()):
            with patch.object(generator_tool, 'generator') as mock_gen:
                with patch.object(generator_tool.validator, 'execute', new=AsyncMock()) as mock_val:
                    # First recipe invalid, second valid
                    invalid_recipe = {"notes": {"top": ["a"]}, "concentrations": {"top": 100}}
                    valid_recipe = {
                        "notes": {"top": ["a", "b"], "middle": ["c", "d", "e"], "base": ["f", "g"]},
                        "concentrations": {"top": 20, "middle": 50, "base": 30}
                    }

                    mock_gen.generate = AsyncMock(side_effect=[invalid_recipe, valid_recipe])
                    mock_val.side_effect = [
                        {"is_valid": False, "validations": []},
                        {"is_valid": True, "validations": []}
                    ]

                    result = await generator_tool.execute(
                        {"fragrance_family": "fresh"},
                        validate=True
                    )

                    assert mock_gen.generate.call_count == 2  # Called twice
                    assert result["validation"]["is_valid"] is True


@pytest.mark.asyncio
async def test_integration_search_to_generate():
    """Integration test: Search then generate based on results"""
    search_tool = UnifiedSearchTool()
    generator_tool = UnifiedGeneratorTool()

    with patch.object(search_tool, 'initialize', new=AsyncMock()):
        with patch.object(generator_tool, 'initialize', new=AsyncMock()):
            # Mock search results
            with patch.object(search_tool, 'vector_store') as mock_store:
                with patch.object(search_tool, 'embedding_model') as mock_embed:
                    mock_embed.encode = AsyncMock(return_value=[[0.1, 0.2]])
                    mock_store.search = AsyncMock(return_value=[
                        {"id": "1", "fragrance_family": "floral", "mood": "romantic"}
                    ])

                    # Search for similar fragrances
                    search_results = await search_tool.execute("romantic perfume")

                    # Use search results to generate new recipe
                    if search_results:
                        base_recipe = search_results[0]
                        with patch.object(generator_tool, 'generator') as mock_gen:
                            mock_gen.generate = AsyncMock(return_value={
                                "notes": {"top": ["rose"], "middle": ["jasmine"], "base": ["musk"]},
                                "concentrations": {"top": 20, "middle": 50, "base": 30}
                            })

                            new_recipe = await generator_tool.execute(
                                {
                                    "fragrance_family": base_recipe.get("fragrance_family"),
                                    "mood": base_recipe.get("mood")
                                },
                                validate=False
                            )

                            assert new_recipe is not None
                            assert "notes" in new_recipe


if __name__ == "__main__":
    pytest.main([__file__, "-v"])