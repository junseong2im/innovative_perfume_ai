"""Schemas package for Fragrance AI API."""

from .recipe_schemas import (
    UserRole,
    FragranceNoteInfo,
    FragranceNoteCustomer,
    FragranceNoteAdmin,
    RecipeIngredientCustomer,
    RecipeIngredientAdmin,
    FragranceProfileCustomer,
    FragranceRecipeAdmin,
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    RecipeListResponse,
    UserAuthRequest,
    UserAuthResponse,
    AdminStats,
)

from .search_schemas import (
    SearchResult,
    SemanticSearchRequest,
    SemanticSearchResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    SystemStatus,
    ErrorResponse,
)

__all__ = [
    'UserRole',
    'FragranceNoteInfo',
    'FragranceNoteCustomer',
    'FragranceNoteAdmin',
    'RecipeIngredientCustomer',
    'RecipeIngredientAdmin',
    'FragranceProfileCustomer',
    'FragranceRecipeAdmin',
    'RecipeGenerationRequest',
    'RecipeGenerationResponse',
    'RecipeListResponse',
    'UserAuthRequest',
    'UserAuthResponse',
    'AdminStats',
    'SearchResult',
    'SemanticSearchRequest',
    'SemanticSearchResponse',
    'BatchGenerationRequest',
    'BatchGenerationResponse',
    'SystemStatus',
    'ErrorResponse',
]