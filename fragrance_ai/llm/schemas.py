# fragrance_ai/llm/schemas.py
"""
LLM Ensemble Schemas
Pydantic models for LLM input/output validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Literal, Optional


class CreativeBrief(BaseModel):
    """
    Creative brief schema for LLM ensemble
    All fields validated and normalized by Mistral validator
    """

    language: Literal["ko", "en"] = "ko"
    mood: List[str] = Field(default_factory=list, description="Mood keywords")
    season: List[Literal["spring", "summer", "autumn", "winter"]] = Field(
        default_factory=list,
        description="Target seasons"
    )
    notes_preference: Dict[str, float] = Field(
        default_factory=dict,
        description="Note type preferences (0-1 range, sum <= 1)"
    )
    forbidden_ingredients: List[str] = Field(
        default_factory=list,
        description="Ingredients to avoid"
    )
    budget_tier: Literal["low", "mid", "high"] = "mid"
    target_profile: Literal["daily_fresh", "evening", "luxury", "sport", "signature"] = "daily_fresh"
    constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Numeric constraints, e.g., max_allergens_ppm"
    )
    product_category: Optional[Literal["EDP", "EDT", "PARFUM"]] = "EDP"
    creative_hints: List[str] = Field(
        default_factory=list,
        description="Creative hints from Llama (max 8, length 2-48)"
    )

    @field_validator("notes_preference")
    @classmethod
    def validate_notes_preference(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Clip values to [0,1] and ensure sum <= 1"""
        # Clip to [0,1]
        clipped = {k: max(0.0, min(1.0, val)) for k, val in v.items()}

        # Check sum
        total = sum(clipped.values())
        if total > 1.0:
            # Normalize
            clipped = {k: val / total for k, val in clipped.items()}

        return clipped

    @field_validator("creative_hints")
    @classmethod
    def validate_creative_hints(cls, v: List[str]) -> List[str]:
        """Limit to max 8 hints, each 2-48 chars"""
        valid_hints = []
        for hint in v[:8]:  # Max 8
            hint = hint.strip()
            if 2 <= len(hint) <= 48:
                valid_hints.append(hint)
        return valid_hints

    @field_validator("constraints")
    @classmethod
    def validate_constraints(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure all constraint values are numeric"""
        return {k: float(val) for k, val in v.items()}

    class Config:
        json_schema_extra = {
            "example": {
                "language": "ko",
                "mood": ["fresh", "romantic", "elegant"],
                "season": ["spring", "summer"],
                "notes_preference": {
                    "citrus": 0.3,
                    "floral": 0.4,
                    "woody": 0.2
                },
                "forbidden_ingredients": ["oakmoss", "musk ketone"],
                "budget_tier": "mid",
                "target_profile": "daily_fresh",
                "constraints": {
                    "max_allergens_ppm": 500,
                    "min_longevity_hours": 4
                },
                "product_category": "EDT",
                "creative_hints": [
                    "morning dew",
                    "spring garden",
                    "clean linen"
                ]
            }
        }


class LLMMode(BaseModel):
    """LLM routing mode"""
    mode: Literal["fast", "balanced", "creative"] = "balanced"

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "balanced"
            }
        }


class LLMRequest(BaseModel):
    """Request to LLM ensemble"""
    user_text: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[Literal["fast", "balanced", "creative"]] = None  # Auto-detect if None
    language: Literal["ko", "en"] = "ko"

    class Config:
        json_schema_extra = {
            "example": {
                "user_text": "상큼하고 가벼운 여름용 향수를 만들어주세요",
                "mode": "balanced",
                "language": "ko"
            }
        }


class LLMResponse(BaseModel):
    """Response from LLM ensemble"""
    brief: CreativeBrief
    mode_used: Literal["fast", "balanced", "creative"]
    processing_time_ms: float
    models_used: List[str]
    cached: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "brief": {
                    "language": "ko",
                    "mood": ["fresh", "light"],
                    "season": ["summer"]
                },
                "mode_used": "balanced",
                "processing_time_ms": 1234.5,
                "models_used": ["Qwen2.5-7B", "Mistral-7B"],
                "cached": False
            }
        }


# Default fallback brief when all LLMs fail
DEFAULT_BRIEF = CreativeBrief(
    language="ko",
    mood=["fresh", "clean"],
    season=["spring"],
    notes_preference={"citrus": 0.3, "floral": 0.4, "woody": 0.3},
    forbidden_ingredients=[],
    budget_tier="mid",
    target_profile="daily_fresh",
    constraints={"max_allergens_ppm": 500},
    product_category="EDT",
    creative_hints=[]
)


__all__ = [
    "CreativeBrief",
    "LLMMode",
    "LLMRequest",
    "LLMResponse",
    "DEFAULT_BRIEF"
]
