# fragrance_ai/schemas/domain_models.py

from pydantic import BaseModel, Field, validator, model_validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np


# ============================================================================
# Enums for Type Safety
# ============================================================================

class NoteCategory(str, Enum):
    """Fragrance note categories"""
    TOP = "top"
    HEART = "heart"  # Also known as middle/mid
    BASE = "base"
    MODIFIER = "modifier"


class ProductCategory(str, Enum):
    """Product categories for IFRA limits"""
    EAU_DE_PARFUM = "eau_de_parfum"
    EAU_DE_TOILETTE = "eau_de_toilette"
    COLOGNE = "cologne"
    BODY_LOTION = "body_lotion"
    SOAP = "soap"
    CANDLE = "candle"
    ROOM_SPRAY = "room_spray"


class ConcentrationType(str, Enum):
    """Types of fragrance concentration"""
    PARFUM = "parfum"  # 20-40%
    EAU_DE_PARFUM = "eau_de_parfum"  # 15-20%
    EAU_DE_TOILETTE = "eau_de_toilette"  # 5-15%
    EAU_DE_COLOGNE = "eau_de_cologne"  # 2-5%
    EAU_FRAICHE = "eau_fraiche"  # 1-3%


# ============================================================================
# Base Models
# ============================================================================

class IngredientBase(BaseModel):
    """Base ingredient model with validation"""

    ingredient_id: str = Field(..., description="Unique identifier for ingredient")
    name: str = Field(..., min_length=1, max_length=100)
    cas_number: Optional[str] = Field(None, pattern=r"^\d{2,7}-\d{2}-\d$")
    category: NoteCategory
    concentration: float = Field(..., ge=0.0, le=100.0)  # Percentage

    @validator('concentration')
    def validate_concentration(cls, v):
        """Ensure concentration is within valid range"""
        if v < 0.0 or v > 100.0:
            raise ValueError(f"Concentration must be between 0 and 100, got {v}")
        return round(v, 4)  # Round to 4 decimal places


class Ingredient(IngredientBase):
    """Extended ingredient with additional properties"""

    ifra_limit: Optional[float] = Field(None, ge=0.0, le=100.0)
    cost_per_kg: Optional[float] = Field(None, ge=0.0)
    density: Optional[float] = Field(None, ge=0.1, le=2.0)  # g/ml
    volatility: Optional[float] = Field(None, ge=0.0, le=1.0)
    odor_threshold: Optional[float] = Field(None, ge=0.0)  # ppm
    description: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "ingredient_id": "ing_001",
                "name": "Bergamot Oil",
                "cas_number": "8007-75-8",
                "category": "top",
                "concentration": 15.5,
                "ifra_limit": 2.0,
                "cost_per_kg": 85.0,
                "density": 0.875,
                "volatility": 0.9,
                "odor_threshold": 0.05,
                "description": "Fresh, citrusy with slight floral undertones"
            }
        }


# ============================================================================
# Olfactory DNA Model
# ============================================================================

class OlfactoryDNA(BaseModel):
    """Core fragrance DNA with genetic information"""

    dna_id: str = Field(..., description="Unique DNA identifier")
    genotype: Dict[str, Any] = Field(..., description="Genetic recipe structure")
    ingredients: List[Ingredient] = Field(..., min_items=1)
    total_concentration: Optional[float] = Field(None, ge=0.0, le=100.0)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    generation: int = Field(default=0, ge=0)
    parent_dna_ids: List[str] = Field(default_factory=list)

    # Computed properties
    category_balance: Optional[Dict[str, float]] = None
    complexity_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_and_normalize(self):
        """Normalize concentrations to sum to 100% and compute properties"""
        ingredients = self.ingredients

        if not ingredients:
            return self

        # Calculate total
        total = sum(ing.concentration for ing in ingredients)

        # Apply minimum effective concentration
        MIN_CONCENTRATION = 0.1
        filtered_ingredients = [
            ing for ing in ingredients
            if ing.concentration >= MIN_CONCENTRATION
        ]

        if not filtered_ingredients:
            raise ValueError("No ingredients above minimum concentration threshold")

        # Normalize to 100%
        if total > 0 and abs(total - 100.0) > 0.01:
            for ing in filtered_ingredients:
                ing.concentration = (ing.concentration / total) * 100.0

        self.ingredients = filtered_ingredients
        self.total_concentration = sum(ing.concentration for ing in filtered_ingredients)

        # Calculate category balance
        category_totals = {'top': 0.0, 'heart': 0.0, 'base': 0.0}
        for ing in filtered_ingredients:
            if ing.category == NoteCategory.TOP:
                category_totals['top'] += ing.concentration
            elif ing.category == NoteCategory.HEART:
                category_totals['heart'] += ing.concentration
            elif ing.category == NoteCategory.BASE:
                category_totals['base'] += ing.concentration

        self.category_balance = category_totals

        # Calculate complexity score (Shannon entropy normalized)
        concentrations = [ing.concentration for ing in filtered_ingredients]
        if concentrations:
            # Normalize to probabilities
            probs = np.array(concentrations) / sum(concentrations)
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(concentrations))
            self.complexity_score = entropy / max_entropy if max_entropy > 0 else 0.0

        return self

    def is_balanced(self) -> bool:
        """Check if fragrance has good category balance"""
        if not self.category_balance:
            return False

        # Ideal ranges: Top: 20-30%, Heart: 30-50%, Base: 30-50%
        return (20 <= self.category_balance['top'] <= 30 and
                30 <= self.category_balance['heart'] <= 50 and
                30 <= self.category_balance['base'] <= 50)


# ============================================================================
# Scent Phenotype Model
# ============================================================================

class ScentPhenotype(BaseModel):
    """Expressed variation of DNA based on environmental triggers"""

    phenotype_id: str = Field(..., description="Unique phenotype identifier")
    based_on_dna: str = Field(..., description="Reference to parent DNA")

    # Epigenetic factors
    epigenetic_trigger: str = Field(..., description="Environmental trigger")
    variation_applied: str = Field(..., description="Type of variation")

    # Adjusted recipe
    adjusted_ingredients: List[Ingredient]
    adjustment_factor: Dict[str, float] = Field(default_factory=dict)

    # Properties
    description: str
    estimated_performance: Optional[Dict[str, float]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('adjusted_ingredients')
    def validate_adjusted_ingredients(cls, v):
        """Ensure adjusted ingredients maintain valid concentrations"""
        total = sum(ing.concentration for ing in v)
        if abs(total - 100.0) > 0.1:
            # Auto-normalize if close
            if 90 < total < 110:
                for ing in v:
                    ing.concentration = (ing.concentration / total) * 100.0
            else:
                raise ValueError(f"Total concentration {total}% is not close to 100%")
        return v


# ============================================================================
# Creative Brief Model
# ============================================================================

class CreativeBrief(BaseModel):
    """User requirements and creative direction"""

    brief_id: Optional[str] = None
    user_id: str

    # Creative direction
    theme: str = Field(..., min_length=1, max_length=200)
    story: Optional[str] = Field(None, max_length=1000)
    mood_keywords: List[str] = Field(default_factory=list, max_items=10)

    # Technical requirements
    target_category: ProductCategory
    concentration_type: ConcentrationType

    # Desired characteristics (0-1 scale)
    desired_intensity: float = Field(0.5, ge=0.0, le=1.0)
    masculinity: float = Field(0.5, ge=0.0, le=1.0)  # 0=feminine, 1=masculine
    complexity: float = Field(0.5, ge=0.0, le=1.0)
    longevity: float = Field(0.5, ge=0.0, le=1.0)
    sillage: float = Field(0.5, ge=0.0, le=1.0)  # Projection/trail
    warmth: float = Field(0.5, ge=0.0, le=1.0)  # 0=cool, 1=warm
    sweetness: float = Field(0.5, ge=0.0, le=1.0)
    freshness: float = Field(0.5, ge=0.0, le=1.0)

    # Constraints
    max_cost_per_kg: Optional[float] = Field(None, ge=0.0)
    excluded_ingredients: List[str] = Field(default_factory=list)
    required_ingredients: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_123",
                "theme": "Modern Rose Garden",
                "story": "A contemporary interpretation of classic rose",
                "mood_keywords": ["elegant", "fresh", "sophisticated"],
                "target_category": "eau_de_parfum",
                "concentration_type": "eau_de_parfum",
                "desired_intensity": 0.7,
                "masculinity": 0.3,
                "complexity": 0.6,
                "warmth": 0.4,
                "max_cost_per_kg": 500.0
            }
        }


# ============================================================================
# Constraint Pack Model
# ============================================================================

class IFRALimit(BaseModel):
    """IFRA regulation limit for an ingredient"""

    ingredient_id: str
    product_category: ProductCategory
    max_concentration: float = Field(..., ge=0.0, le=100.0)
    restriction_type: str = Field(default="standard")  # standard, prohibited, restricted
    notes: Optional[str] = None

    class Config:
        frozen = True  # Make immutable


class SolubilityRule(BaseModel):
    """Solubility constraints between ingredients"""

    ingredient1_id: str
    ingredient2_id: str
    compatibility_score: float = Field(..., ge=-1.0, le=1.0)  # -1=incompatible, 1=excellent
    max_combined_concentration: Optional[float] = Field(None, ge=0.0, le=100.0)
    notes: Optional[str] = None


class ConstraintPack(BaseModel):
    """Complete set of constraints for formulation"""

    pack_id: str = Field(default="default")
    ifra_limits: List[IFRALimit] = Field(default_factory=list)
    solubility_rules: List[SolubilityRule] = Field(default_factory=list)

    # Global constraints
    min_ingredients: int = Field(3, ge=1, le=50)
    max_ingredients: int = Field(15, ge=1, le=50)
    min_concentration_per_ingredient: float = Field(0.1, ge=0.0, le=10.0)
    max_concentration_per_ingredient: float = Field(50.0, ge=1.0, le=100.0)

    # Category balance constraints
    category_ranges: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "top": {"min": 15.0, "max": 35.0},
            "heart": {"min": 30.0, "max": 55.0},
            "base": {"min": 25.0, "max": 50.0}
        }
    )

    @validator('max_ingredients')
    def validate_ingredient_range(cls, v, values):
        """Ensure max >= min"""
        min_ing = values.get('min_ingredients', 1)
        if v < min_ing:
            raise ValueError(f"max_ingredients ({v}) must be >= min_ingredients ({min_ing})")
        return v

    def get_ifra_limit(self, ingredient_id: str, product_category: ProductCategory) -> Optional[float]:
        """Get IFRA limit for specific ingredient and product category"""
        for limit in self.ifra_limits:
            if limit.ingredient_id == ingredient_id and limit.product_category == product_category:
                return limit.max_concentration
        return None

    def check_solubility(self, ing1_id: str, ing2_id: str) -> Optional[float]:
        """Check solubility compatibility between two ingredients"""
        for rule in self.solubility_rules:
            if (rule.ingredient1_id == ing1_id and rule.ingredient2_id == ing2_id) or \
               (rule.ingredient1_id == ing2_id and rule.ingredient2_id == ing1_id):
                return rule.compatibility_score
        return None


# ============================================================================
# Validation Response Models
# ============================================================================

class ValidationViolation(BaseModel):
    """Single validation violation"""

    violation_type: str  # ifra, solubility, balance, concentration
    severity: str  # error, warning, info
    ingredient_id: Optional[str] = None
    message: str
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    suggestion: Optional[str] = None


class ValidationResult(BaseModel):
    """Complete validation result for a formulation"""

    is_valid: bool
    violations: List[ValidationViolation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    score: float = Field(1.0, ge=0.0, le=1.0)  # Overall compliance score

    def has_errors(self) -> bool:
        """Check if there are any error-level violations"""
        return any(v.severity == "error" for v in self.violations)

    def get_errors(self) -> List[ValidationViolation]:
        """Get only error-level violations"""
        return [v for v in self.violations if v.severity == "error"]


# ============================================================================
# User Interaction Models
# ============================================================================

class UserChoice(BaseModel):
    """User's selection from presented options"""

    session_id: str
    user_id: str
    dna_id: str
    phenotype_id: str
    brief_id: str

    # Choice details
    chosen_option_id: str
    presented_options: List[str]  # All option IDs that were shown
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    feedback_text: Optional[str] = None

    # Context
    iteration_number: int = Field(1, ge=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Computed preference signals
    preference_vector: Optional[List[float]] = None


class LearningHistory(BaseModel):
    """Training history for ML models"""

    history_id: str
    user_id: str

    entries: List[UserChoice] = Field(default_factory=list)

    # Aggregated metrics
    total_interactions: int = Field(0, ge=0)
    average_rating: Optional[float] = None
    preference_evolution: Optional[List[Dict[str, float]]] = None

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def add_choice(self, choice: UserChoice):
        """Add a new choice and update metrics"""
        self.entries.append(choice)
        self.total_interactions += 1

        # Update average rating
        ratings = [e.rating for e in self.entries if e.rating is not None]
        if ratings:
            self.average_rating = sum(ratings) / len(ratings)

        self.updated_at = datetime.utcnow()


# ============================================================================
# Export all models
# ============================================================================

__all__ = [
    # Enums
    'NoteCategory',
    'ProductCategory',
    'ConcentrationType',

    # Core models
    'Ingredient',
    'IngredientBase',
    'OlfactoryDNA',
    'ScentPhenotype',
    'CreativeBrief',

    # Constraints
    'IFRALimit',
    'SolubilityRule',
    'ConstraintPack',

    # Validation
    'ValidationViolation',
    'ValidationResult',

    # User interaction
    'UserChoice',
    'LearningHistory'
]