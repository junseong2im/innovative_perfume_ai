# fragrance_ai/schemas/models.py
"""
Pydantic Models with Comprehensive Validation
Enforces business rules: sum=100%, no negatives, minimum concentrations
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from enum import Enum
import math


# ============================================================================
# Constants
# ============================================================================

MIN_CONCENTRATION = 0.1  # Minimum effective concentration (%)
SUM_TOLERANCE = 0.01  # Tolerance for sum validation (0.01%)
MAX_INGREDIENTS = 50  # Maximum number of ingredients in a formula


# ============================================================================
# Enums
# ============================================================================

class FormulaType(str, Enum):
    """Types of fragrance formulas"""
    FRAGRANCE = "fragrance"
    COMPOUND = "compound"
    BASE = "base"
    ACCORD = "accord"
    DILUTION = "dilution"


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"  # Enforce all rules
    RELAXED = "relaxed"  # Allow minor deviations
    DEVELOPMENT = "development"  # More permissive for R&D


# ============================================================================
# Component Models
# ============================================================================

class FormulaIngredient(BaseModel):
    """Single ingredient in a formula with validation"""

    name: str = Field(..., min_length=1, max_length=100)
    cas_number: Optional[str] = Field(None, pattern=r'^\d{2,7}-\d{2}-\d$')
    percentage: float = Field(..., ge=0, le=100)
    grams: Optional[float] = Field(None, ge=0)
    milliliters: Optional[float] = Field(None, ge=0)
    material_type: Optional[str] = None
    ifra_limit: Optional[float] = Field(None, ge=0, le=100)
    cost_per_kg: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = None

    @field_validator('percentage')
    def validate_percentage(cls, v: float) -> float:
        """Ensure percentage is non-negative"""
        if v < 0:
            raise ValueError(f"Percentage cannot be negative: {v}")
        if v > 100:
            raise ValueError(f"Percentage cannot exceed 100: {v}")
        return round(v, 4)  # Round to 4 decimal places

    @field_validator('cas_number')
    def validate_cas(cls, v: Optional[str]) -> Optional[str]:
        """Validate CAS number format and checksum if provided"""
        if v is None:
            return None

        # CAS format: XXXXXX-XX-X
        parts = v.split('-')
        if len(parts) != 3:
            raise ValueError(f"Invalid CAS format: {v}")

        # Validate checksum
        try:
            digits = parts[0] + parts[1]
            check_digit = int(parts[2])

            # CAS checksum algorithm
            total = sum((i + 1) * int(d) for i, d in enumerate(reversed(digits)))
            calculated_check = total % 10

            if calculated_check != check_digit:
                raise ValueError(f"Invalid CAS checksum: {v}")
        except (ValueError, IndexError):
            raise ValueError(f"Invalid CAS number: {v}")

        return v


class FragranceFormula(BaseModel):
    """Complete fragrance formula with validation"""

    name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(default="1.0", pattern=r'^\d+\.\d+(\.\d+)?$')
    type: FormulaType = FormulaType.FRAGRANCE
    ingredients: List[FormulaIngredient] = Field(..., min_length=1, max_length=MAX_INGREDIENTS)
    batch_size_g: float = Field(100.0, gt=0)
    validation_level: ValidationLevel = ValidationLevel.STRICT

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: Optional[str] = None
    target_cost_per_kg: Optional[float] = Field(None, ge=0)

    # Calculated fields
    total_percentage: Optional[float] = None
    total_cost: Optional[float] = None
    ingredient_count: Optional[int] = None

    @field_validator('ingredients')
    def validate_unique_ingredients(cls, v: List[FormulaIngredient]) -> List[FormulaIngredient]:
        """Ensure ingredient names are unique"""
        names = [ing.name for ing in v]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate ingredients: {set(duplicates)}")
        return v

    @model_validator(mode='after')
    def validate_formula(self) -> 'FragranceFormula':
        """Comprehensive formula validation"""

        # Skip validation in development mode
        if self.validation_level == ValidationLevel.DEVELOPMENT:
            return self

        # 1. Check percentage sum
        total = sum(ing.percentage for ing in self.ingredients)
        self.total_percentage = round(total, 4)

        if self.validation_level == ValidationLevel.STRICT:
            if abs(total - 100.0) > SUM_TOLERANCE:
                raise ValueError(f"Percentages sum to {total:.4f}%, must be 100% ±{SUM_TOLERANCE}%")
        elif self.validation_level == ValidationLevel.RELAXED:
            if abs(total - 100.0) > 1.0:  # Allow 1% deviation
                raise ValueError(f"Percentages sum to {total:.4f}%, exceeds 1% tolerance")

        # 2. Remove ingredients below minimum concentration (if strict)
        if self.validation_level == ValidationLevel.STRICT:
            filtered = [ing for ing in self.ingredients if ing.percentage >= MIN_CONCENTRATION]
            if len(filtered) < len(self.ingredients):
                removed = len(self.ingredients) - len(filtered)
                # Re-normalize after filtering
                total_filtered = sum(ing.percentage for ing in filtered)
                if total_filtered > 0:
                    for ing in filtered:
                        ing.percentage = (ing.percentage / total_filtered) * 100
                self.ingredients = filtered

        # 3. Calculate total cost if prices available
        if all(ing.cost_per_kg is not None for ing in self.ingredients):
            self.total_cost = sum(
                (ing.percentage / 100) * ing.cost_per_kg
                for ing in self.ingredients
            )

        # 4. Update ingredient count
        self.ingredient_count = len(self.ingredients)

        return self

    def normalize(self) -> 'FragranceFormula':
        """Normalize percentages to sum to 100%"""
        total = sum(ing.percentage for ing in self.ingredients)
        if total > 0:
            for ing in self.ingredients:
                ing.percentage = (ing.percentage / total) * 100
        self.total_percentage = 100.0
        return self

    def apply_minimum_concentration(self, min_conc: float = MIN_CONCENTRATION) -> 'FragranceFormula':
        """Remove ingredients below minimum and renormalize"""
        filtered = [ing for ing in self.ingredients if ing.percentage >= min_conc]
        if filtered:
            self.ingredients = filtered
            self.normalize()
        return self


# ============================================================================
# Batch Production Models
# ============================================================================

class BatchProduction(BaseModel):
    """Production batch with mass conservation validation"""

    batch_id: str = Field(..., min_length=1)
    formula: FragranceFormula
    target_weight_g: float = Field(..., gt=0)
    actual_weights: Dict[str, float] = Field(default_factory=dict)

    # Production metadata
    production_date: datetime = Field(default_factory=datetime.utcnow)
    operator: Optional[str] = None
    equipment: Optional[str] = None
    ambient_temp_c: Optional[float] = Field(None, ge=0, le=50)

    # Validation results
    mass_balance: Optional[Dict[str, Any]] = None
    yield_percent: Optional[float] = None
    deviations: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_production(self) -> 'BatchProduction':
        """Validate production batch"""

        if not self.actual_weights:
            return self

        # 1. Check all ingredients are weighed
        formula_ingredients = {ing.name for ing in self.formula.ingredients}
        actual_ingredients = set(self.actual_weights.keys())

        missing = formula_ingredients - actual_ingredients
        if missing:
            self.deviations.append(f"Missing weights for: {missing}")

        extra = actual_ingredients - formula_ingredients
        if extra:
            self.deviations.append(f"Extra ingredients weighed: {extra}")

        # 2. Calculate theoretical weights
        theoretical_weights = {}
        for ing in self.formula.ingredients:
            theoretical_weights[ing.name] = (ing.percentage / 100) * self.target_weight_g

        # 3. Check mass balance
        total_theoretical = sum(theoretical_weights.values())
        total_actual = sum(self.actual_weights.values())

        deviation_g = abs(total_actual - total_theoretical)
        deviation_pct = (deviation_g / total_theoretical * 100) if total_theoretical > 0 else 100

        self.mass_balance = {
            "theoretical_total_g": round(total_theoretical, 2),
            "actual_total_g": round(total_actual, 2),
            "deviation_g": round(deviation_g, 2),
            "deviation_percent": round(deviation_pct, 2),
            "within_tolerance": deviation_pct <= 1.0  # 1% tolerance
        }

        if not self.mass_balance["within_tolerance"]:
            self.deviations.append(
                f"Mass balance deviation: {deviation_pct:.2f}% (limit: 1%)"
            )

        # 4. Calculate yield
        self.yield_percent = (total_actual / total_theoretical * 100) if total_theoretical > 0 else 0

        # 5. Check individual ingredient deviations
        for ing_name, theoretical in theoretical_weights.items():
            if ing_name in self.actual_weights:
                actual = self.actual_weights[ing_name]
                ing_deviation = abs(actual - theoretical)
                ing_deviation_pct = (ing_deviation / theoretical * 100) if theoretical > 0 else 100

                if ing_deviation_pct > 5.0:  # 5% tolerance per ingredient
                    self.deviations.append(
                        f"{ing_name}: {ing_deviation_pct:.1f}% deviation"
                    )

        return self


# ============================================================================
# Stability and Compliance Models
# ============================================================================

class StabilityData(BaseModel):
    """Stability test data with validation"""

    formula: FragranceFormula
    test_conditions: str = Field(..., min_length=1)
    time_point_days: int = Field(..., ge=0)

    # Measurements
    color_change: Optional[str] = None
    odor_change: Optional[str] = None
    ph_value: Optional[float] = Field(None, ge=0, le=14)
    specific_gravity: Optional[float] = Field(None, ge=0.5, le=2.0)
    refractive_index: Optional[float] = Field(None, ge=1.0, le=2.0)

    # Component stability
    component_changes: Dict[str, float] = Field(default_factory=dict)

    # Validation
    stable: bool = True
    issues: List[str] = Field(default_factory=list)

    @field_validator('ph_value')
    def validate_ph(cls, v: Optional[float]) -> Optional[float]:
        """Validate pH is in reasonable range"""
        if v is not None and (v < 2 or v > 12):
            raise ValueError(f"Unusual pH value: {v}")
        return v

    @model_validator(mode='after')
    def check_stability(self) -> 'StabilityData':
        """Check for stability issues"""

        # Check component changes
        for ingredient, change_pct in self.component_changes.items():
            if abs(change_pct) > 10:  # 10% change threshold
                self.stable = False
                self.issues.append(f"{ingredient}: {change_pct:+.1f}% change")

        # Check physical changes
        if self.color_change and self.color_change.lower() in ['significant', 'major', 'dark']:
            self.stable = False
            self.issues.append(f"Color change: {self.color_change}")

        if self.odor_change and self.odor_change.lower() in ['off', 'rancid', 'degraded']:
            self.stable = False
            self.issues.append(f"Odor change: {self.odor_change}")

        return self


class ComplianceCheck(BaseModel):
    """Regulatory compliance validation"""

    formula: FragranceFormula
    region: str = Field("EU", pattern=r'^(EU|US|JAPAN|CHINA|GLOBAL)$')
    product_category: str

    # IFRA compliance
    ifra_violations: List[Dict[str, Any]] = Field(default_factory=list)
    ifra_compliant: bool = True

    # Allergen declaration
    allergens_to_declare: List[str] = Field(default_factory=list)

    # Restricted materials
    restricted_materials: List[str] = Field(default_factory=list)
    prohibited_materials: List[str] = Field(default_factory=list)

    # Overall compliance
    compliant: bool = True
    compliance_notes: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_compliance(self) -> 'ComplianceCheck':
        """Perform compliance checks"""

        # Check each ingredient against IFRA limits
        for ing in self.formula.ingredients:
            if ing.ifra_limit is not None and ing.percentage > ing.ifra_limit:
                violation = {
                    "ingredient": ing.name,
                    "percentage": ing.percentage,
                    "limit": ing.ifra_limit,
                    "excess": ing.percentage - ing.ifra_limit
                }
                self.ifra_violations.append(violation)
                self.ifra_compliant = False
                self.compliance_notes.append(
                    f"{ing.name}: {ing.percentage:.2f}% exceeds IFRA limit {ing.ifra_limit:.2f}%"
                )

        # Update overall compliance
        self.compliant = (
            self.ifra_compliant and
            len(self.prohibited_materials) == 0
        )

        if not self.compliant:
            if self.prohibited_materials:
                self.compliance_notes.append(
                    f"Contains prohibited materials: {', '.join(self.prohibited_materials)}"
                )

        return self


# ============================================================================
# Quality Control Models
# ============================================================================

class QualityControl(BaseModel):
    """QC test results with validation"""

    batch: BatchProduction
    test_date: datetime = Field(default_factory=datetime.utcnow)
    tester: str = Field(..., min_length=1)

    # Organoleptic evaluation
    appearance: str = Field(..., min_length=1)
    color: str = Field(..., min_length=1)
    odor_description: str = Field(..., min_length=1)
    odor_strength: int = Field(..., ge=1, le=10)

    # Physical parameters
    specific_gravity: float = Field(..., ge=0.5, le=2.0)
    refractive_index: float = Field(..., ge=1.0, le=2.0)
    flash_point_c: Optional[float] = Field(None, ge=0, le=200)

    # Chromatography
    gc_peaks: Optional[int] = Field(None, ge=0)
    main_components_match: bool = True

    # Results
    passed: bool = True
    rejection_reasons: List[str] = Field(default_factory=list)
    corrective_actions: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def evaluate_quality(self) -> 'QualityControl':
        """Evaluate QC results"""

        # Check specific gravity range (typical: 0.85-1.05 for fragrances)
        if not (0.85 <= self.specific_gravity <= 1.05):
            self.passed = False
            self.rejection_reasons.append(
                f"Specific gravity {self.specific_gravity:.3f} out of range"
            )
            self.corrective_actions.append("Verify formulation and re-blend")

        # Check refractive index range (typical: 1.45-1.55 for fragrances)
        if not (1.45 <= self.refractive_index <= 1.55):
            self.passed = False
            self.rejection_reasons.append(
                f"Refractive index {self.refractive_index:.4f} out of range"
            )
            self.corrective_actions.append("Check raw material quality")

        # Check component matching
        if not self.main_components_match:
            self.passed = False
            self.rejection_reasons.append("GC profile does not match standard")
            self.corrective_actions.append("Investigate formulation deviation")

        # Check odor strength
        if self.odor_strength < 3:
            self.rejection_reasons.append("Weak odor strength")
            self.corrective_actions.append("Verify concentration and aging")

        return self


# ============================================================================
# Helper Functions
# ============================================================================

def validate_formula_sum(
    ingredients: List[Dict[str, float]],
    tolerance: float = SUM_TOLERANCE
) -> Dict[str, Any]:
    """
    Validate that formula percentages sum to 100%

    Args:
        ingredients: List of dicts with 'percentage' key
        tolerance: Acceptable deviation from 100%

    Returns:
        Validation result dictionary
    """
    total = sum(ing.get('percentage', 0) for ing in ingredients)
    deviation = abs(total - 100.0)
    is_valid = deviation <= tolerance

    return {
        "valid": is_valid,
        "total": total,
        "deviation": deviation,
        "message": "OK" if is_valid else f"Sum is {total:.2f}%, deviation {deviation:.2f}%"
    }


def apply_minimum_concentration_filter(
    ingredients: List[Dict[str, Any]],
    min_concentration: float = MIN_CONCENTRATION,
    renormalize: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter out ingredients below minimum concentration

    Args:
        ingredients: List of ingredient dictionaries
        min_concentration: Minimum percentage threshold
        renormalize: Whether to renormalize to 100% after filtering

    Returns:
        Filtered (and optionally renormalized) ingredient list
    """
    # Filter
    filtered = [
        ing for ing in ingredients
        if ing.get('percentage', 0) >= min_concentration
    ]

    if not filtered:
        return []

    # Renormalize if requested
    if renormalize:
        total = sum(ing['percentage'] for ing in filtered)
        if total > 0:
            for ing in filtered:
                ing['percentage'] = (ing['percentage'] / total) * 100.0

    return filtered


def check_no_negatives(values: List[float]) -> bool:
    """Check that all values are non-negative"""
    return all(v >= 0 for v in values)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Models
    'FormulaIngredient',
    'FragranceFormula',
    'BatchProduction',
    'StabilityData',
    'ComplianceCheck',
    'QualityControl',

    # Enums
    'FormulaType',
    'ValidationLevel',

    # Constants
    'MIN_CONCENTRATION',
    'SUM_TOLERANCE',
    'MAX_INGREDIENTS',

    # Helper functions
    'validate_formula_sum',
    'apply_minimum_concentration_filter',
    'check_no_negatives'
]