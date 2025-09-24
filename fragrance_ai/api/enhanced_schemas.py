"""
Enhanced API Schemas with Comprehensive Documentation
Detailed Pydantic models with examples and validation
"""

from typing import Dict, List, Optional, Union, Any, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator


class FragranceFamily(str, Enum):
    """Fragrance family categories"""
    CITRUS = "citrus"
    FLORAL = "floral"
    ORIENTAL = "oriental"
    WOODY = "woody"
    FRESH = "fresh"
    FOUGERE = "fougere"
    CHYPRE = "chypre"
    GOURMAND = "gourmand"


class Season(str, Enum):
    """Seasonal preferences"""
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"
    ALL_SEASON = "all_season"


class Gender(str, Enum):
    """Gender targeting"""
    MASCULINE = "masculine"
    FEMININE = "feminine"
    UNISEX = "unisex"


class Mood(str, Enum):
    """Mood categories"""
    ROMANTIC = "romantic"
    ENERGETIC = "energetic"
    RELAXING = "relaxing"
    MYSTERIOUS = "mysterious"
    FRESH = "fresh"
    SOPHISTICATED = "sophisticated"
    PLAYFUL = "playful"
    ELEGANT = "elegant"


class Intensity(str, Enum):
    """Fragrance intensity levels"""
    LIGHT = "light"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class NotePosition(str, Enum):
    """Note position in fragrance pyramid"""
    TOP = "top"
    HEART = "heart"
    BASE = "base"


# Base Models

class FragranceNote(BaseModel):
    """Individual fragrance note"""
    name: str = Field(
        ...,
        description="Name of the fragrance note",
        example="Rose Bulgarian",
        min_length=1,
        max_length=100
    )
    percentage: float = Field(
        ...,
        description="Percentage of this note in the blend",
        example=15.5,
        ge=0.0,
        le=100.0
    )
    position: NotePosition = Field(
        ...,
        description="Position in fragrance pyramid (top, heart, base)",
        example="heart"
    )
    supplier: Optional[str] = Field(
        None,
        description="Supplier of the ingredient",
        example="Givaudan",
        max_length=50
    )
    cost_per_kg: Optional[float] = Field(
        None,
        description="Cost per kilogram in USD",
        example=1250.00,
        ge=0.0
    )
    sustainability_score: Optional[float] = Field(
        None,
        description="Sustainability score (0-10)",
        example=8.5,
        ge=0.0,
        le=10.0
    )

    class Config:
        schema_extra = {
            "example": {
                "name": "Rose Bulgarian",
                "percentage": 15.5,
                "position": "heart",
                "supplier": "Givaudan",
                "cost_per_kg": 1250.00,
                "sustainability_score": 8.5
            }
        }


class FragranceProfile(BaseModel):
    """Complete fragrance profile"""
    notes: Dict[str, List[str]] = Field(
        ...,
        description="Notes organized by position (top, heart, base)",
        example={
            "top": ["bergamot", "lemon", "pink pepper"],
            "heart": ["rose", "jasmine", "lily of the valley"],
            "base": ["sandalwood", "white musk", "amber"]
        }
    )
    family: FragranceFamily = Field(
        ...,
        description="Primary fragrance family",
        example="floral"
    )
    sub_families: List[str] = Field(
        default=[],
        description="Secondary fragrance families",
        example=["rose", "white_floral"],
        max_items=5
    )
    intensity: Intensity = Field(
        ...,
        description="Overall intensity level",
        example="moderate"
    )
    longevity_hours: float = Field(
        ...,
        description="Expected longevity in hours",
        example=6.5,
        ge=0.5,
        le=24.0
    )
    projection_rating: float = Field(
        ...,
        description="Projection strength (1-10)",
        example=7.2,
        ge=1.0,
        le=10.0
    )

    class Config:
        schema_extra = {
            "example": {
                "notes": {
                    "top": ["bergamot", "lemon", "pink pepper"],
                    "heart": ["rose", "jasmine", "lily of the valley"],
                    "base": ["sandalwood", "white musk", "amber"]
                },
                "family": "floral",
                "sub_families": ["rose", "white_floral"],
                "intensity": "moderate",
                "longevity_hours": 6.5,
                "projection_rating": 7.2
            }
        }


# Search Schemas

class SearchFilters(BaseModel):
    """Advanced search filters"""
    families: Optional[List[FragranceFamily]] = Field(
        None,
        description="Filter by fragrance families",
        example=["floral", "oriental"],
        max_items=10
    )
    seasons: Optional[List[Season]] = Field(
        None,
        description="Filter by suitable seasons",
        example=["spring", "summer"],
        max_items=4
    )
    gender: Optional[Gender] = Field(
        None,
        description="Filter by gender targeting",
        example="feminine"
    )
    mood: Optional[List[Mood]] = Field(
        None,
        description="Filter by mood categories",
        example=["romantic", "elegant"],
        max_items=5
    )
    intensity: Optional[List[Intensity]] = Field(
        None,
        description="Filter by intensity levels",
        example=["moderate", "strong"],
        max_items=4
    )
    price_range: Optional[List[float]] = Field(
        None,
        description="Price range filter [min, max] in USD",
        example=[50.0, 200.0],
        min_items=2,
        max_items=2
    )
    longevity_min: Optional[float] = Field(
        None,
        description="Minimum longevity in hours",
        example=4.0,
        ge=0.5,
        le=24.0
    )
    projection_min: Optional[float] = Field(
        None,
        description="Minimum projection rating",
        example=5.0,
        ge=1.0,
        le=10.0
    )
    brands: Optional[List[str]] = Field(
        None,
        description="Filter by brand names",
        example=["Chanel", "Dior", "Tom Ford"],
        max_items=20
    )
    release_year_range: Optional[List[int]] = Field(
        None,
        description="Release year range [start, end]",
        example=[2000, 2024],
        min_items=2,
        max_items=2
    )
    availability: Optional[bool] = Field(
        None,
        description="Filter by current availability",
        example=True
    )

    @field_validator('price_range')
    def validate_price_range(cls, v):
        if v and len(v) == 2 and v[0] > v[1]:
            raise ValueError('Minimum price must be less than maximum price')
        return v

    @field_validator('release_year_range')
    def validate_year_range(cls, v):
        if v and len(v) == 2:
            if v[0] > v[1]:
                raise ValueError('Start year must be less than end year')
            if v[0] < 1900 or v[1] > 2030:
                raise ValueError('Years must be between 1900 and 2030')
        return v

    class Config:
        schema_extra = {
            "example": {
                "families": ["floral", "oriental"],
                "seasons": ["spring", "summer"],
                "gender": "feminine",
                "mood": ["romantic", "elegant"],
                "intensity": ["moderate"],
                "price_range": [50.0, 200.0],
                "longevity_min": 4.0,
                "projection_min": 6.0,
                "brands": ["Chanel", "Dior"],
                "release_year_range": [2010, 2024],
                "availability": True
            }
        }


class SemanticSearchRequest(BaseModel):
    """Semantic search request"""
    query: str = Field(
        ...,
        description="Natural language search query describing the desired fragrance",
        example="romantic rose fragrance perfect for date nights in spring",
        min_length=3,
        max_length=500
    )
    top_k: int = Field(
        default=10,
        description="Number of results to return",
        example=10,
        ge=1,
        le=100
    )
    filters: Optional[SearchFilters] = Field(
        None,
        description="Additional filters to apply to search results"
    )
    search_type: Literal["similarity", "hybrid", "keyword"] = Field(
        default="similarity",
        description="Type of search algorithm to use",
        example="similarity"
    )
    include_similar: bool = Field(
        default=True,
        description="Include similar fragrances in results",
        example=True
    )
    min_similarity_score: float = Field(
        default=0.5,
        description="Minimum similarity score for results",
        example=0.7,
        ge=0.0,
        le=1.0
    )
    explain_results: bool = Field(
        default=False,
        description="Include explanation of why each result was selected",
        example=False
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "romantic rose fragrance perfect for date nights in spring",
                "top_k": 10,
                "filters": {
                    "families": ["floral"],
                    "seasons": ["spring"],
                    "mood": ["romantic"],
                    "price_range": [50.0, 300.0]
                },
                "search_type": "similarity",
                "include_similar": True,
                "min_similarity_score": 0.7,
                "explain_results": True
            }
        }


class FragranceResult(BaseModel):
    """Individual fragrance search result"""
    id: str = Field(
        ...,
        description="Unique fragrance identifier",
        example="fragrance_rose_garden_001"
    )
    name: str = Field(
        ...,
        description="Fragrance name",
        example="Rose Garden Elegance"
    )
    brand: str = Field(
        ...,
        description="Brand name",
        example="Maison Lumière"
    )
    description: str = Field(
        ...,
        description="Detailed fragrance description",
        example="An elegant rose-centered fragrance that captures the essence of a blooming garden at dawn"
    )
    profile: FragranceProfile = Field(
        ...,
        description="Complete fragrance profile"
    )
    similarity_score: float = Field(
        ...,
        description="Similarity score to search query (0-1)",
        example=0.92,
        ge=0.0,
        le=1.0
    )
    price: Optional[float] = Field(
        None,
        description="Price in USD",
        example=150.00,
        ge=0.0
    )
    availability: bool = Field(
        ...,
        description="Current availability status",
        example=True
    )
    release_year: Optional[int] = Field(
        None,
        description="Year of release",
        example=2023,
        ge=1900,
        le=2030
    )
    rating: Optional[float] = Field(
        None,
        description="Average user rating (1-10)",
        example=8.7,
        ge=1.0,
        le=10.0
    )
    review_count: Optional[int] = Field(
        None,
        description="Number of user reviews",
        example=247,
        ge=0
    )
    explanation: Optional[str] = Field(
        None,
        description="Explanation of why this result matches the query",
        example="This fragrance matches your search for romantic rose fragrances with its prominent Bulgarian rose heart and elegant white floral accompaniments"
    )
    image_url: Optional[str] = Field(
        None,
        description="URL to fragrance bottle image",
        example="https://cdn.fragranceai.com/images/rose_garden_elegance.jpg"
    )
    created_at: datetime = Field(
        ...,
        description="When this fragrance was added to our database",
        example="2024-01-15T10:30:00Z"
    )

    class Config:
        schema_extra = {
            "example": {
                "id": "fragrance_rose_garden_001",
                "name": "Rose Garden Elegance",
                "brand": "Maison Lumière",
                "description": "An elegant rose-centered fragrance that captures the essence of a blooming garden at dawn",
                "profile": {
                    "notes": {
                        "top": ["bergamot", "pink pepper", "mandarin"],
                        "heart": ["rose bulgarian", "peony", "lily of the valley"],
                        "base": ["white musk", "cedar", "sandalwood"]
                    },
                    "family": "floral",
                    "sub_families": ["rose", "white_floral"],
                    "intensity": "moderate",
                    "longevity_hours": 6.5,
                    "projection_rating": 7.2
                },
                "similarity_score": 0.92,
                "price": 150.00,
                "availability": True,
                "release_year": 2023,
                "rating": 8.7,
                "review_count": 247,
                "explanation": "This fragrance matches your search with its romantic rose composition",
                "image_url": "https://cdn.fragranceai.com/images/rose_garden_elegance.jpg",
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class SearchResponse(BaseModel):
    """Semantic search response"""
    results: List[FragranceResult] = Field(
        ...,
        description="List of matching fragrances"
    )
    total_count: int = Field(
        ...,
        description="Total number of matching fragrances (before pagination)",
        example=156
    )
    query_embedding_time_ms: float = Field(
        ...,
        description="Time taken to generate query embedding",
        example=12.5
    )
    search_time_ms: float = Field(
        ...,
        description="Time taken to perform vector search",
        example=8.3
    )
    total_time_ms: float = Field(
        ...,
        description="Total processing time",
        example=20.8
    )
    search_metadata: Dict[str, Any] = Field(
        default={},
        description="Additional search metadata",
        example={
            "model_version": "v3.2.1",
            "index_size": 50000,
            "search_algorithm": "HNSW"
        }
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier for tracking",
        example="req_search_1701234567890"
    )

    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "fragrance_rose_garden_001",
                        "name": "Rose Garden Elegance",
                        "brand": "Maison Lumière",
                        "similarity_score": 0.92,
                        "price": 150.00
                    }
                ],
                "total_count": 156,
                "query_embedding_time_ms": 12.5,
                "search_time_ms": 8.3,
                "total_time_ms": 20.8,
                "search_metadata": {
                    "model_version": "v3.2.1",
                    "index_size": 50000
                },
                "request_id": "req_search_1701234567890"
            }
        }


# Generation Schemas

class GenerationRequest(BaseModel):
    """AI fragrance generation request"""
    mood: Optional[Mood] = Field(
        None,
        description="Desired mood for the fragrance",
        example="romantic"
    )
    season: Optional[Season] = Field(
        None,
        description="Target season for the fragrance",
        example="spring"
    )
    gender: Optional[Gender] = Field(
        None,
        description="Gender targeting",
        example="feminine"
    )
    family: Optional[FragranceFamily] = Field(
        None,
        description="Desired fragrance family",
        example="floral"
    )
    intensity: Optional[Intensity] = Field(
        None,
        description="Desired intensity level",
        example="moderate"
    )
    inspiration: Optional[str] = Field(
        None,
        description="Creative inspiration or theme",
        example="A walk through a blooming cherry blossom garden",
        max_length=500
    )
    target_price: Optional[float] = Field(
        None,
        description="Target price range in USD",
        example=120.00,
        ge=10.0,
        le=1000.0
    )
    sustainability_preference: Optional[float] = Field(
        None,
        description="Sustainability preference (0-10, 10 being most sustainable)",
        example=8.0,
        ge=0.0,
        le=10.0
    )
    complexity: Literal["simple", "medium", "complex"] = Field(
        default="medium",
        description="Desired complexity level",
        example="medium"
    )
    creativity: float = Field(
        default=0.7,
        description="Creativity level (0-1, higher = more creative)",
        example=0.7,
        ge=0.0,
        le=1.0
    )
    must_include_notes: Optional[List[str]] = Field(
        None,
        description="Notes that must be included in the recipe",
        example=["rose", "vanilla"],
        max_items=10
    )
    exclude_notes: Optional[List[str]] = Field(
        None,
        description="Notes to avoid in the recipe",
        example=["patchouli", "oud"],
        max_items=20
    )

    class Config:
        schema_extra = {
            "example": {
                "mood": "romantic",
                "season": "spring",
                "gender": "feminine",
                "family": "floral",
                "intensity": "moderate",
                "inspiration": "A walk through a blooming cherry blossom garden",
                "target_price": 120.00,
                "sustainability_preference": 8.0,
                "complexity": "medium",
                "creativity": 0.7,
                "must_include_notes": ["rose", "cherry blossom"],
                "exclude_notes": ["patchouli", "tobacco"]
            }
        }


class GeneratedRecipe(BaseModel):
    """AI-generated fragrance recipe"""
    id: str = Field(
        ...,
        description="Unique recipe identifier",
        example="recipe_cherry_bloom_001"
    )
    name: str = Field(
        ...,
        description="Generated fragrance name",
        example="Cherry Blossom Dream"
    )
    description: str = Field(
        ...,
        description="AI-generated description",
        example="A delicate floral fragrance capturing the ephemeral beauty of cherry blossoms in full bloom"
    )
    ingredients: List[FragranceNote] = Field(
        ...,
        description="Detailed ingredient list with percentages"
    )
    total_percentage: float = Field(
        ...,
        description="Total percentage (should be 100.0)",
        example=100.0
    )
    estimated_cost: float = Field(
        ...,
        description="Estimated production cost in USD",
        example=85.50
    )
    sustainability_score: float = Field(
        ...,
        description="Overall sustainability score (0-10)",
        example=7.8,
        ge=0.0,
        le=10.0
    )
    complexity_score: float = Field(
        ...,
        description="Recipe complexity score (0-10)",
        example=6.2,
        ge=0.0,
        le=10.0
    )
    predicted_profile: FragranceProfile = Field(
        ...,
        description="Predicted fragrance profile"
    )
    confidence_score: float = Field(
        ...,
        description="AI confidence in the recipe (0-1)",
        example=0.89,
        ge=0.0,
        le=1.0
    )
    alternative_variations: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Alternative ingredient variations",
        max_items=3
    )
    production_notes: Optional[str] = Field(
        None,
        description="Special production considerations",
        example="Allow 24h maceration for optimal blending"
    )
    created_at: datetime = Field(
        ...,
        description="Recipe creation timestamp",
        example="2024-01-15T14:22:33Z"
    )

    @field_validator('total_percentage')
    def validate_total_percentage(cls, v):
        if abs(v - 100.0) > 0.1:
            raise ValueError('Total percentage must equal 100.0')
        return v

    class Config:
        schema_extra = {
            "example": {
                "id": "recipe_cherry_bloom_001",
                "name": "Cherry Blossom Dream",
                "description": "A delicate floral fragrance capturing cherry blossoms",
                "ingredients": [
                    {
                        "name": "Cherry Blossom Accord",
                        "percentage": 30.0,
                        "position": "heart",
                        "supplier": "IFF"
                    },
                    {
                        "name": "Bergamot FCF",
                        "percentage": 20.0,
                        "position": "top",
                        "supplier": "Givaudan"
                    }
                ],
                "total_percentage": 100.0,
                "estimated_cost": 85.50,
                "sustainability_score": 7.8,
                "complexity_score": 6.2,
                "confidence_score": 0.89,
                "created_at": "2024-01-15T14:22:33Z"
            }
        }


class GenerationResponse(BaseModel):
    """AI generation response"""
    recipe: GeneratedRecipe = Field(
        ...,
        description="Generated fragrance recipe"
    )
    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate recipe",
        example=1250.5
    )
    model_version: str = Field(
        ...,
        description="AI model version used",
        example="fragrance_gen_v3.2.1"
    )
    tokens_used: int = Field(
        ...,
        description="Number of tokens consumed",
        example=2847
    )
    request_id: str = Field(
        ...,
        description="Unique request identifier",
        example="req_gen_1701234567890"
    )

    class Config:
        schema_extra = {
            "example": {
                "recipe": {
                    "id": "recipe_cherry_bloom_001",
                    "name": "Cherry Blossom Dream",
                    "description": "A delicate floral fragrance",
                    "estimated_cost": 85.50,
                    "confidence_score": 0.89
                },
                "generation_time_ms": 1250.5,
                "model_version": "fragrance_gen_v3.2.1",
                "tokens_used": 2847,
                "request_id": "req_gen_1701234567890"
            }
        }


# Error Schemas

class ValidationError(BaseModel):
    """Validation error details"""
    field: str = Field(
        ...,
        description="Field that failed validation",
        example="query"
    )
    message: str = Field(
        ...,
        description="Validation error message",
        example="Query must be between 3 and 500 characters"
    )
    invalid_value: Any = Field(
        ...,
        description="The invalid value that was provided",
        example="hi"
    )


class APIError(BaseModel):
    """Standard API error response"""
    error: str = Field(
        ...,
        description="Error message",
        example="Validation failed"
    )
    code: str = Field(
        ...,
        description="Error code",
        example="VALIDATION_ERROR"
    )
    message: Optional[str] = Field(
        None,
        description="Detailed error message",
        example="The request contains invalid parameters"
    )
    details: Optional[List[ValidationError]] = Field(
        None,
        description="Detailed validation errors"
    )
    request_id: str = Field(
        ...,
        description="Request identifier for tracking",
        example="req_error_1701234567890"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )
    suggestion: Optional[str] = Field(
        None,
        description="Suggested solution",
        example="Please provide a query between 3 and 500 characters"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "Validation failed",
                "code": "VALIDATION_ERROR",
                "message": "The request contains invalid parameters",
                "details": [
                    {
                        "field": "query",
                        "message": "Query must be between 3 and 500 characters",
                        "invalid_value": "hi"
                    }
                ],
                "request_id": "req_error_1701234567890",
                "timestamp": "2024-01-15T10:30:00Z",
                "suggestion": "Please provide a query between 3 and 500 characters"
            }
        }


def get_enhanced_openapi_schema(app):
    """Enhanced OpenAPI schema with additional metadata"""
    from fastapi.openapi.utils import get_openapi

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Fragrance AI API",
        version="2.0.0",
        description="Advanced AI-powered fragrance recommendation and generation system",
        routes=app.routes,
    )

    # Add enhanced metadata
    openapi_schema["info"]["contact"] = {
        "name": "Fragrance AI Team",
        "email": "junseong2im@gmail.com"
    }

    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
        "url": "https://example.com/license"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Export all schemas
__all__ = [
    "FragranceFamily", "Season", "Gender", "Mood", "Intensity", "NotePosition",
    "FragranceNote", "FragranceProfile", "SearchFilters", "SemanticSearchRequest",
    "FragranceResult", "SearchResponse", "GenerationRequest", "GeneratedRecipe",
    "GenerationResponse", "ValidationError", "APIError", "get_enhanced_openapi_schema"
]