"""Search-related schemas for Fragrance AI API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    name: str
    description: str
    fragrance_family: str
    similarity_score: float = Field(..., ge=0, le=1)
    metadata: Optional[Dict[str, Any]] = None
    ingredients: Optional[List[str]] = None
    mood_tags: Optional[List[str]] = None
    season_tags: Optional[List[str]] = None


class SemanticSearchRequest(BaseModel):
    """Semantic search request schema."""
    query: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(default=10, ge=1, le=100)
    fragrance_families: Optional[List[str]] = None
    mood: Optional[str] = None
    season: Optional[str] = None
    min_similarity: float = Field(default=0.5, ge=0, le=1)
    search_type: str = Field(default="similarity")  # similarity, hybrid, exact
    filters: Optional[Dict[str, Any]] = None


class SemanticSearchResponse(BaseModel):
    """Semantic search response schema."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    search_type: str
    metadata: Optional[Dict[str, Any]] = None


class BatchGenerationRequest(BaseModel):
    """Batch generation request schema."""
    recipes: List[Dict[str, Any]]
    batch_options: Optional[Dict[str, Any]] = None
    parallel_processing: bool = Field(default=True)
    max_parallel: int = Field(default=5, ge=1, le=20)


class BatchGenerationResponse(BaseModel):
    """Batch generation response schema."""
    batch_id: str
    total_recipes: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    processing_time_ms: float
    errors: Optional[List[str]] = None


class SystemStatus(BaseModel):
    """System status information."""
    status: str  # healthy, degraded, error
    version: str
    uptime_seconds: float
    database_status: str
    cache_status: str
    model_status: Dict[str, str]
    active_connections: int
    requests_per_minute: float
    average_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    request_id: Optional[str] = None
    error_code: Optional[str] = None