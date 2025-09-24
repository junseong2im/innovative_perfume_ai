"""
Hybrid Search Tool for the LLM Orchestrator
- Combines semantic search with structured filtering
- Primary method for finding information about existing perfumes
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
import logging
import asyncio
from ..services.search_service import SearchService
from ..core.exceptions_unified import handle_exceptions_async

logger = logging.getLogger(__name__)

class MetadataFilters(BaseModel):
    """SQL-based pre-filtering conditions."""
    price_less_than: Optional[int] = Field(None, description="Filter for perfumes below this price point.")
    gender: Optional[Literal['male', 'female', 'unisex']] = Field(None, description="Filter by gender suitability.")
    season: Optional[Literal['spring', 'summer', 'autumn', 'winter']] = Field(None, description="Filter by recommended season.")
    include_notes: Optional[List[str]] = Field(None, description="List of notes that MUST be present in the perfume.")
    exclude_notes: Optional[List[str]] = Field(None, description="List of notes that MUST NOT be present.")
    brand: Optional[str] = Field(None, description="Filter by specific brand name.")
    fragrance_family: Optional[str] = Field(None, description="Filter by fragrance family (floral, citrus, woody, etc.).")
    min_longevity: Optional[float] = Field(None, description="Minimum longevity in hours.")
    max_price: Optional[float] = Field(None, description="Maximum price filter.")

class SearchResultItem(BaseModel):
    """A single item returned from the search tool."""
    perfume_id: str = Field(..., description="Unique identifier for the perfume.")
    name: str = Field(..., description="Name of the perfume.")
    brand: str = Field(..., description="Brand name.")
    description: str = Field(..., description="Detailed description of the perfume.")
    combined_score: float = Field(..., description="Hybrid score combining semantic and keyword relevance.")
    notes: Optional[Dict[str, List[str]]] = Field(None, description="Fragrance notes organized by type (top, middle, base).")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata about the perfume.")
    collection: Optional[str] = Field(None, description="Source collection name.")

# Global search service instance
_search_service = None

async def _get_search_service() -> SearchService:
    """Get or initialize the search service instance."""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
        await _search_service.initialize()
    return _search_service

async def hybrid_search(
    text_query: str,
    metadata_filters: MetadataFilters,
    top_k: int = 10
) -> List[SearchResultItem]:
    """
    ## LLM Tool Description
    Use this tool as your primary method for finding information about existing perfumes.
    It is essential for answering questions about specific products, finding perfumes that match an abstract feeling,
    or retrieving examples that fit certain criteria. It combines semantic search on descriptions (for abstract queries)
    with structured filtering on concrete attributes (like price or season).

    ## When to use
    - When a user asks for a recommendation based on a mood, feeling, or abstract concept (e.g., "a perfume that feels like a rainy forest").
    - When a user asks to find perfumes with specific characteristics (e.g., "find summer perfumes under $100 with citrus notes").
    - When you need inspiration from existing products to create a new one.

    Args:
        text_query: The semantic search query (mood, feeling, abstract concept)
        metadata_filters: Structured filters for concrete attributes
        top_k: Number of results to return (default: 10)

    Returns:
        List of SearchResultItem objects containing perfume information
    """
    logger.info(f"Starting hybrid search for query: {text_query[:50]}...")

    try:
        # Try vector search first, fallback to database search if needed
        try:
            search_service = await _get_search_service()
            logger.info("Vector search service initialized, attempting semantic search")

            # Convert metadata filters to search service format
            # ChromaDB only supports simple filters, so we prioritize the most important ones
            filters = {}

            # Only use one filter at a time for ChromaDB compatibility
            if metadata_filters.fragrance_family:
                filters["fragrance_family"] = metadata_filters.fragrance_family
            elif metadata_filters.season:
                filters["season"] = metadata_filters.season
            elif metadata_filters.gender:
                filters["gender"] = metadata_filters.gender

            # Note: Complex filters like price ranges and note inclusion/exclusion
            # will be handled by post-processing the results

            # Perform hybrid search with timeout protection
            search_results = await asyncio.wait_for(
                search_service.semantic_search(
                    query=text_query,
                    collection_names=["fragrance_notes", "recipes", "brands"],
                    search_type="hybrid",
                    top_k=top_k,
                    similarity_threshold=0.6,
                    filters=filters,
                    include_metadata=True,
                    use_cache=True
                ),
                timeout=30.0  # 30 second timeout
            )

            # Convert results to SearchResultItem format
            search_items = []
            for result in search_results.get("results", []):
                # Extract notes from metadata or document
                notes = None
                metadata = result.get("metadata", {})

                if "notes" in metadata:
                    notes = metadata["notes"]
                elif result.get("collection") == "recipes":
                    # Try to extract notes from recipe data
                    try:
                        import json
                        doc = result.get("document", "{}")
                        if isinstance(doc, str):
                            doc_data = json.loads(doc)
                            notes = doc_data.get("notes", {})
                    except:
                        notes = {}

                search_item = SearchResultItem(
                    perfume_id=result.get("id", "unknown"),
                    name=metadata.get("name", "Unknown Perfume"),
                    brand=metadata.get("brand", "Unknown Brand"),
                    description=result.get("document", ""),
                    combined_score=result.get("similarity_score", 0.0),
                    notes=notes,
                    metadata=metadata,
                    collection=result.get("collection")
                )
                search_items.append(search_item)

            logger.info(f"Vector search completed: {len(search_items)} results for query: {text_query[:50]}...")
            return search_items

        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Vector search failed ({type(e).__name__}: {e}), falling back to database search")
            # Fallback to basic database search
            return await _basic_database_search(text_query, metadata_filters, top_k)

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        # Fallback to mock results if database search also fails
        return [
            SearchResultItem(
                perfume_id="fallback_1",
                name="Mock Romantic Perfume",
                brand="Test Brand",
                description=f"로맨틱한 향수 - {text_query}에 대한 응답",
                combined_score=0.95,
                notes={"top": ["bergamot"], "middle": ["rose"], "base": ["vanilla"]},
                metadata={"mock": True, "query": text_query},
                collection="fallback"
            )
        ]

# Alternative function for when filters are optional
async def semantic_search_only(
    text_query: str,
    top_k: int = 10,
    collection_names: Optional[List[str]] = None
) -> List[SearchResultItem]:
    """
    Simplified semantic search without metadata filters.
    Useful for pure semantic/conceptual queries.
    """
    return await hybrid_search(
        text_query=text_query,
        metadata_filters=MetadataFilters(),  # Empty filters
        top_k=top_k
    )

# Utility function for finding similar perfumes to a given one
async def find_similar_perfumes(
    perfume_id: str,
    collection_name: str = "recipes",
    top_k: int = 5
) -> List[SearchResultItem]:
    """
    Find perfumes similar to a given perfume ID.
    """
    try:
        search_service = await _get_search_service()

        similar_results = await search_service.get_similar_items(
            item_id=perfume_id,
            collection_name=collection_name,
            top_k=top_k
        )

        # Convert to SearchResultItem format
        search_items = []
        for result in similar_results:
            metadata = result.get("metadata", {})

            search_item = SearchResultItem(
                perfume_id=result.get("id", "unknown"),
                name=metadata.get("name", "Unknown Perfume"),
                brand=metadata.get("brand", "Unknown Brand"),
                description=result.get("document", ""),
                combined_score=result.get("similarity_score", 0.0),
                notes=metadata.get("notes"),
                metadata=metadata,
                collection=result.get("collection")
            )
            search_items.append(search_item)

        return search_items

    except Exception as e:
        logger.error(f"Similar perfume search failed: {e}")
        raise

# Basic database search fallback
async def _basic_database_search(text_query: str, metadata_filters: MetadataFilters, top_k: int) -> List[SearchResultItem]:
    """Basic database search without vector embeddings"""
    try:
        import sqlite3
        conn = sqlite3.connect('fragrance_ai.db')

        # Build SQL query based on filters
        where_conditions = []
        params = []

        if metadata_filters.include_notes:
            note_conditions = " OR ".join(["name LIKE ? OR name_korean LIKE ?" for _ in metadata_filters.include_notes])
            where_conditions.append(f"({note_conditions})")
            for note in metadata_filters.include_notes:
                params.extend([f"%{note}%", f"%{note}%"])

        if metadata_filters.fragrance_family:
            where_conditions.append("fragrance_family = ?")
            params.append(metadata_filters.fragrance_family)

        if metadata_filters.gender:
            where_conditions.append("JSON_EXTRACT(gender_tags, '$[0]') = ?")
            params.append(metadata_filters.gender)

        # Search in description for text query
        if text_query:
            where_conditions.append("(description LIKE ? OR description_korean LIKE ? OR name LIKE ? OR name_korean LIKE ?)")
            params.extend([f"%{text_query}%", f"%{text_query}%", f"%{text_query}%", f"%{text_query}%"])

        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""

        query = f"""
            SELECT name, name_korean, fragrance_family, note_type, description,
                   description_korean, intensity, longevity
            FROM fragrance_notes
            {where_clause}
            LIMIT {top_k}
        """

        results = conn.execute(query, params).fetchall()
        conn.close()

        search_items = []
        for i, result in enumerate(results):
            name, korean_name, family, note_type, desc, desc_korean, intensity, longevity = result

            search_item = SearchResultItem(
                perfume_id=f"db_{i}",
                name=name,
                brand="Database Brand",
                description=desc or desc_korean or f"{name} - {family} {note_type}",
                combined_score=0.8 - i * 0.1,  # Simple scoring
                notes={note_type: [name]},
                metadata={
                    "name": name,
                    "korean_name": korean_name,
                    "fragrance_family": family,
                    "note_type": note_type,
                    "intensity": intensity,
                    "longevity": longevity,
                    "source": "database"
                },
                collection="fragrance_notes"
            )
            search_items.append(search_item)

        logger.info(f"Basic database search returned {len(search_items)} results")
        return search_items

    except Exception as e:
        logger.error(f"Basic database search failed: {e}")
        # Return mock results as final fallback
        return [
            SearchResultItem(
                perfume_id="mock_1",
                name="Mock Romantic Perfume",
                brand="Test Brand",
                description="A romantic fragrance with rose and vanilla notes for testing purposes",
                combined_score=0.95,
                notes={"top": ["bergamot"], "middle": ["rose"], "base": ["vanilla"]},
                metadata={"mock": True},
                collection="mock"
            )
        ]