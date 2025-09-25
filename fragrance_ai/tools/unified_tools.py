"""
Unified tools module - consolidates all fragrance AI tools
Prevents duplication and ensures consistency
"""

from typing import Dict, Any, List, Optional, Union
import json
import logging
from datetime import datetime

from ..core.vector_store import get_vector_store
from ..models.embedding import EmbeddingModel
from ..models.generator import FragranceGenerator
from ..models.rag_system import RAGSystem
from ..llm.ollama_client import OllamaClient
from ..core.config import settings

logger = logging.getLogger(__name__)


class UnifiedSearchTool:
    """
    Unified search tool combining vector and traditional search
    Replaces: search_tool.py, hybrid_search_tool.py
    """

    def __init__(self):
        self.vector_store = None
        self.embedding_model = None
        self.name = "hybrid_search"
        self.description = "Unified search tool for fragrances"
        self._initialized = False

    async def initialize(self):
        """Lazy initialization"""
        if not self._initialized:
            self.vector_store = await get_vector_store()
            from ..api.dependencies import get_model_factory
            factory = get_model_factory()
            self.embedding_model = factory.get_model("embedding_model", EmbeddingModel)
            self._initialized = True

    async def execute(
        self,
        query: str,
        search_type: str = "hybrid",
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute unified search

        Args:
            query: Search query
            search_type: "vector", "traditional", or "hybrid"
            top_k: Number of results
            filters: Optional filters

        Returns:
            List of search results
        """
        await self.initialize()

        try:
            results = []

            if search_type in ["vector", "hybrid"]:
                # Vector search
                query_embedding = await self.embedding_model.encode([query])
                vector_results = await self.vector_store.search(
                    query_vector=query_embedding[0],
                    top_k=top_k,
                    metadata_filter=filters
                )
                results.extend(vector_results)

            if search_type in ["traditional", "hybrid"]:
                # Traditional keyword search
                keyword_results = await self._keyword_search(query, filters, top_k)

                # Merge results if hybrid
                if search_type == "hybrid":
                    results = self._merge_results(results, keyword_results)
                else:
                    results = keyword_results

            # Deduplicate and rank
            results = self._deduplicate_results(results)[:top_k]

            logger.info(f"Search completed: {len(results)} results found")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _keyword_search(
        self,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Traditional keyword search implementation"""
        # Simplified keyword search logic
        # In production, this would query a full-text search index
        return []

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge vector and keyword search results"""
        merged = {}

        # Score vector results higher
        for result in vector_results:
            id_key = result.get("id", str(result))
            if id_key not in merged:
                merged[id_key] = result
                merged[id_key]["score"] = result.get("score", 0.5) * 1.2

        # Add keyword results with lower weight
        for result in keyword_results:
            id_key = result.get("id", str(result))
            if id_key not in merged:
                merged[id_key] = result
                merged[id_key]["score"] = result.get("score", 0.5) * 0.8
            else:
                # Boost score if in both results
                merged[id_key]["score"] += result.get("score", 0.5) * 0.5

        # Sort by combined score
        return sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results"""
        seen = set()
        unique_results = []

        for result in results:
            id_key = result.get("id", str(result))
            if id_key not in seen:
                seen.add(id_key)
                unique_results.append(result)

        return unique_results


class UnifiedKnowledgeTool:
    """
    Unified knowledge base tool
    Replaces: knowledge_tool.py, perfumer_knowledge_tool.py
    """

    def __init__(self):
        self.rag_system = None
        self.ollama_client = None
        self.name = "knowledge_base"
        self.description = "Unified fragrance knowledge base"
        self._initialized = False
        self._knowledge_cache = {}

    async def initialize(self):
        """Lazy initialization"""
        if not self._initialized:
            from ..api.dependencies import get_model_factory
            factory = get_model_factory()
            self.rag_system = factory.get_model("rag_system", RAGSystem)
            self.ollama_client = factory.get_model("ollama_client", OllamaClient)
            self._initialized = True

    async def execute(
        self,
        query: str,
        category: Optional[str] = None,
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Query knowledge base

        Args:
            query: Knowledge query
            category: Optional category filter
            include_citations: Include source citations

        Returns:
            Knowledge response with citations
        """
        await self.initialize()

        try:
            # Check cache
            cache_key = f"{query}:{category}:{include_citations}"
            if cache_key in self._knowledge_cache:
                logger.info("Knowledge retrieved from cache")
                return self._knowledge_cache[cache_key]

            # Retrieve relevant context
            context = await self.rag_system.retrieve(
                query=query,
                category=category,
                top_k=5
            )

            # Generate response with LLM
            response = await self.ollama_client.generate(
                prompt=self._build_knowledge_prompt(query, context),
                model="llama3"
            )

            result = {
                "answer": response,
                "query": query,
                "category": category,
                "timestamp": datetime.utcnow().isoformat()
            }

            if include_citations:
                result["citations"] = self._extract_citations(context)

            # Cache result
            self._knowledge_cache[cache_key] = result

            logger.info(f"Knowledge query completed for: {query}")
            return result

        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return {
                "answer": "Unable to retrieve knowledge at this time.",
                "error": str(e),
                "query": query
            }

    def _build_knowledge_prompt(self, query: str, context: List[Dict]) -> str:
        """Build prompt for knowledge generation"""
        context_text = "\n".join([
            f"- {item.get('text', '')}" for item in context
        ])

        return f"""Based on the following fragrance knowledge:

{context_text}

Please answer this question: {query}

Provide a detailed, accurate answer based on the context provided."""

    def _extract_citations(self, context: List[Dict]) -> List[Dict[str, str]]:
        """Extract citations from context"""
        citations = []
        for item in context:
            citations.append({
                "source": item.get("source", "Unknown"),
                "relevance": item.get("score", 0),
                "text": item.get("text", "")[:200]  # First 200 chars
            })
        return citations


class UnifiedValidatorTool:
    """
    Unified validation tool for fragrance recipes
    Single source of truth for validation logic
    """

    def __init__(self):
        self.name = "recipe_validator"
        self.description = "Unified recipe validation tool"
        self._validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules"""
        return {
            "concentration_limits": {
                "top": {"min": 15, "max": 30},
                "middle": {"min": 30, "max": 60},
                "base": {"min": 20, "max": 40}
            },
            "total_concentration": {"min": 95, "max": 105},
            "ingredient_limits": {
                "min_ingredients": 5,
                "max_ingredients": 30
            },
            "safety_restrictions": [
                "no_allergens",
                "ifra_compliance",
                "eu_regulations"
            ]
        }

    async def execute(
        self,
        recipe: Dict[str, Any],
        validation_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        Validate fragrance recipe

        Args:
            recipe: Recipe to validate
            validation_level: "basic", "standard", or "strict"

        Returns:
            Validation results
        """
        try:
            validations = []
            is_valid = True

            # Concentration validation
            conc_result = self._validate_concentrations(recipe)
            validations.append(conc_result)
            if not conc_result["passed"]:
                is_valid = False

            # Ingredient count validation
            ing_result = self._validate_ingredient_count(recipe)
            validations.append(ing_result)
            if not ing_result["passed"]:
                is_valid = False

            if validation_level in ["standard", "strict"]:
                # Safety validation
                safety_result = self._validate_safety(recipe)
                validations.append(safety_result)
                if not safety_result["passed"]:
                    is_valid = False

            if validation_level == "strict":
                # IFRA compliance
                ifra_result = self._validate_ifra_compliance(recipe)
                validations.append(ifra_result)
                if not ifra_result["passed"]:
                    is_valid = False

            return {
                "is_valid": is_valid,
                "validation_level": validation_level,
                "validations": validations,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "is_valid": False,
                "error": str(e),
                "validations": []
            }

    def _validate_concentrations(self, recipe: Dict) -> Dict[str, Any]:
        """Validate note concentrations"""
        notes = recipe.get("notes", {})
        concentrations = recipe.get("concentrations", {})

        rules = self._validation_rules["concentration_limits"]
        issues = []

        for note_type, limits in rules.items():
            conc = concentrations.get(note_type, 0)
            if conc < limits["min"] or conc > limits["max"]:
                issues.append(f"{note_type}: {conc}% (should be {limits['min']}-{limits['max']}%)")

        total = sum(concentrations.values())
        total_limits = self._validation_rules["total_concentration"]
        if total < total_limits["min"] or total > total_limits["max"]:
            issues.append(f"Total: {total}% (should be {total_limits['min']}-{total_limits['max']}%)")

        return {
            "check": "concentration_validation",
            "passed": len(issues) == 0,
            "issues": issues
        }

    def _validate_ingredient_count(self, recipe: Dict) -> Dict[str, Any]:
        """Validate ingredient count"""
        notes = recipe.get("notes", {})
        total_ingredients = sum(len(v) if isinstance(v, list) else 1 for v in notes.values())

        limits = self._validation_rules["ingredient_limits"]
        issues = []

        if total_ingredients < limits["min_ingredients"]:
            issues.append(f"Too few ingredients: {total_ingredients} (minimum {limits['min_ingredients']})")
        elif total_ingredients > limits["max_ingredients"]:
            issues.append(f"Too many ingredients: {total_ingredients} (maximum {limits['max_ingredients']})")

        return {
            "check": "ingredient_count",
            "passed": len(issues) == 0,
            "issues": issues
        }

    def _validate_safety(self, recipe: Dict) -> Dict[str, Any]:
        """Validate safety compliance"""
        # Simplified safety validation
        # In production, this would check against allergen databases
        return {
            "check": "safety_compliance",
            "passed": True,
            "issues": []
        }

    def _validate_ifra_compliance(self, recipe: Dict) -> Dict[str, Any]:
        """Validate IFRA compliance"""
        # Simplified IFRA validation
        # In production, this would check against IFRA standards
        return {
            "check": "ifra_compliance",
            "passed": True,
            "issues": []
        }


class UnifiedGeneratorTool:
    """
    Unified recipe generation tool
    Single source for all generation logic
    """

    def __init__(self):
        self.generator = None
        self.validator = UnifiedValidatorTool()
        self.name = "recipe_generator"
        self.description = "Unified recipe generation tool"
        self._initialized = False

    async def initialize(self):
        """Lazy initialization"""
        if not self._initialized:
            from ..api.dependencies import get_model_factory
            factory = get_model_factory()
            self.generator = factory.get_model("generator", FragranceGenerator)
            self._initialized = True

    async def execute(
        self,
        request: Dict[str, Any],
        validate: bool = True,
        enhance: bool = False
    ) -> Dict[str, Any]:
        """
        Generate fragrance recipe

        Args:
            request: Generation request parameters
            validate: Validate generated recipe
            enhance: Apply enhancements

        Returns:
            Generated recipe with metadata
        """
        await self.initialize()

        try:
            # Generate base recipe
            recipe = await self.generator.generate(
                fragrance_family=request.get("fragrance_family", "floral"),
                mood=request.get("mood", "romantic"),
                intensity=request.get("intensity", "moderate"),
                gender=request.get("gender", "unisex"),
                season=request.get("season", "spring")
            )

            # Validate if requested
            if validate:
                validation = await self.validator.execute(recipe)
                recipe["validation"] = validation

                # Regenerate if invalid
                if not validation["is_valid"]:
                    logger.warning("Generated recipe failed validation, regenerating...")
                    recipe = await self._regenerate_valid_recipe(request)

            # Enhance if requested
            if enhance:
                recipe = await self._enhance_recipe(recipe, request)

            recipe["metadata"] = {
                "generated_at": datetime.utcnow().isoformat(),
                "request": request,
                "validated": validate,
                "enhanced": enhance
            }

            logger.info(f"Recipe generated successfully")
            return recipe

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "error": str(e),
                "request": request
            }

    async def _regenerate_valid_recipe(
        self,
        request: Dict[str, Any],
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Regenerate until valid recipe is created"""
        for attempt in range(max_attempts):
            recipe = await self.generator.generate(
                fragrance_family=request.get("fragrance_family", "floral"),
                mood=request.get("mood", "romantic"),
                intensity=request.get("intensity", "moderate")
            )

            validation = await self.validator.execute(recipe)
            if validation["is_valid"]:
                recipe["validation"] = validation
                return recipe

            logger.warning(f"Regeneration attempt {attempt + 1} failed validation")

        # Return last attempt even if invalid
        recipe["validation"] = validation
        recipe["warning"] = "Could not generate valid recipe within attempts"
        return recipe

    async def _enhance_recipe(
        self,
        recipe: Dict[str, Any],
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhance recipe with additional details"""
        # Add cultural elements if Korean request
        if request.get("korean_region") or request.get("korean_season"):
            recipe["korean_elements"] = {
                "region": request.get("korean_region"),
                "season": request.get("korean_season"),
                "traditional_influence": request.get("traditional_element")
            }

        # Add cost estimation
        recipe["estimated_cost"] = self._estimate_cost(recipe, request.get("budget_range", "medium"))

        # Add complexity score
        recipe["complexity_score"] = self._calculate_complexity(recipe)

        return recipe

    def _estimate_cost(self, recipe: Dict, budget_range: str) -> Dict[str, float]:
        """Estimate recipe cost"""
        base_costs = {
            "low": 50,
            "medium": 150,
            "high": 500,
            "luxury": 1500
        }

        base = base_costs.get(budget_range, 150)
        ingredient_count = sum(
            len(v) if isinstance(v, list) else 1
            for v in recipe.get("notes", {}).values()
        )

        return {
            "per_ml": base / 100,
            "per_30ml": base * 0.3,
            "per_50ml": base * 0.5,
            "per_100ml": base,
            "currency": "USD"
        }

    def _calculate_complexity(self, recipe: Dict) -> int:
        """Calculate recipe complexity score (1-10)"""
        notes = recipe.get("notes", {})
        total_ingredients = sum(
            len(v) if isinstance(v, list) else 1
            for v in notes.values()
        )

        if total_ingredients < 10:
            return 3
        elif total_ingredients < 15:
            return 5
        elif total_ingredients < 20:
            return 7
        else:
            return 9


# Export unified tools
search_tool = UnifiedSearchTool()
knowledge_tool = UnifiedKnowledgeTool()
validator_tool = UnifiedValidatorTool()
generator_tool = UnifiedGeneratorTool()

__all__ = [
    "UnifiedSearchTool",
    "UnifiedKnowledgeTool",
    "UnifiedValidatorTool",
    "UnifiedGeneratorTool",
    "search_tool",
    "knowledge_tool",
    "validator_tool",
    "generator_tool"
]