# fragrance_ai/regulations/ifra_rules.py

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from fragrance_ai.schemas.domain_models import (
    Ingredient, OlfactoryDNA, ProductCategory,
    ValidationViolation, ValidationResult, IFRALimit
)


# ============================================================================
# IFRA Categories and Limits
# ============================================================================

class IFRACategory(Enum):
    """IFRA product application categories (49th Amendment)"""
    # Category 1: Products applied to the lips
    CAT_1 = "1"
    # Category 2: Products applied to the axillae
    CAT_2 = "2"
    # Category 3: Products applied to the face/body using fingertips
    CAT_3 = "3"
    # Category 4: Fine fragrance products
    CAT_4 = "4"
    # Category 5A: Products applied to the face and body using the hands (palms), primarily leave-on
    CAT_5A = "5A"
    # Category 5B: Products applied to the face and body using the hands (palms), primarily rinse-off
    CAT_5B = "5B"
    # Category 5C: Products applied to the hair with some hand contact
    CAT_5C = "5C"
    # Category 5D: Products not intended for skin contact, minimal or insignificant transfer to skin
    CAT_5D = "5D"
    # Category 6: Products with oral and lip exposure
    CAT_6 = "6"
    # Category 7A: Rinse-off products applied to the hair with some hand contact
    CAT_7A = "7A"
    # Category 7B: Leave-on products applied to the hair with some hand contact
    CAT_7B = "7B"
    # Category 8: Products with significant ano-genital exposure
    CAT_8 = "8"
    # Category 9: Rinse-off products with body and hand exposure
    CAT_9 = "9"
    # Category 10A: Household care products with mostly hand contact
    CAT_10A = "10A"
    # Category 10B: Household care products with mostly no skin contact
    CAT_10B = "10B"
    # Category 11A: Products with intended skin contact but minimal transfer of fragrance to skin from inert substrate
    CAT_11A = "11A"
    # Category 11B: Products not intended for skin contact, minimal or insignificant transfer to skin
    CAT_11B = "11B"
    # Category 12: Products not intended for direct skin contact, minimal or insignificant transfer to skin
    CAT_12 = "12"


# Map product categories to IFRA categories
PRODUCT_TO_IFRA_MAPPING = {
    ProductCategory.EAU_DE_PARFUM: IFRACategory.CAT_4,
    ProductCategory.EAU_DE_TOILETTE: IFRACategory.CAT_4,
    ProductCategory.COLOGNE: IFRACategory.CAT_4,
    ProductCategory.BODY_LOTION: IFRACategory.CAT_5A,
    ProductCategory.SOAP: IFRACategory.CAT_9,
    ProductCategory.CANDLE: IFRACategory.CAT_12,
    ProductCategory.ROOM_SPRAY: IFRACategory.CAT_11B,
}


@dataclass
class IFRAIngredientLimit:
    """IFRA limit for a specific ingredient"""
    ingredient_name: str
    cas_number: Optional[str]
    category_limits: Dict[IFRACategory, float]  # % limits per category
    restriction_type: str = "standard"  # standard, prohibited, restricted
    notes: Optional[str] = None


class IFRADatabase:
    """Database of IFRA regulations and limits"""

    def __init__(self, data_path: Optional[str] = None):
        """Initialize IFRA database with regulations"""
        self.limits_db: Dict[str, IFRAIngredientLimit] = {}
        self._load_default_limits()

        if data_path:
            self._load_custom_limits(data_path)

    def _load_default_limits(self):
        """Load default IFRA limits for common ingredients"""
        # Example IFRA limits (simplified - real database would be much larger)
        default_limits = [
            IFRAIngredientLimit(
                ingredient_name="Bergamot Oil",
                cas_number="8007-75-8",
                category_limits={
                    IFRACategory.CAT_1: 0.0,  # Prohibited on lips
                    IFRACategory.CAT_2: 0.2,  # Very low for underarms
                    IFRACategory.CAT_3: 0.4,
                    IFRACategory.CAT_4: 2.0,  # Fine fragrance limit
                    IFRACategory.CAT_5A: 0.8,
                    IFRACategory.CAT_9: 2.0,
                    IFRACategory.CAT_12: 100.0,  # No skin contact
                },
                restriction_type="restricted",
                notes="Phototoxic - contains bergapten"
            ),
            IFRAIngredientLimit(
                ingredient_name="Oakmoss Absolute",
                cas_number="9000-50-4",
                category_limits={
                    IFRACategory.CAT_1: 0.0,
                    IFRACategory.CAT_2: 0.04,
                    IFRACategory.CAT_3: 0.04,
                    IFRACategory.CAT_4: 0.1,
                    IFRACategory.CAT_5A: 0.1,
                    IFRACategory.CAT_9: 0.1,
                    IFRACategory.CAT_12: 100.0,
                },
                restriction_type="restricted",
                notes="Sensitizer - atranol and chloroatranol content"
            ),
            IFRAIngredientLimit(
                ingredient_name="Jasmine Absolute",
                cas_number="8022-96-6",
                category_limits={
                    IFRACategory.CAT_4: 0.7,
                    IFRACategory.CAT_5A: 0.2,
                    IFRACategory.CAT_9: 1.0,
                    IFRACategory.CAT_12: 100.0,
                },
                restriction_type="standard"
            ),
            IFRAIngredientLimit(
                ingredient_name="Ylang Ylang Oil",
                cas_number="8006-81-3",
                category_limits={
                    IFRACategory.CAT_4: 0.8,
                    IFRACategory.CAT_5A: 0.3,
                    IFRACategory.CAT_9: 1.2,
                    IFRACategory.CAT_12: 100.0,
                },
                restriction_type="standard"
            ),
            IFRAIngredientLimit(
                ingredient_name="Cinnamon Bark Oil",
                cas_number="8015-91-6",
                category_limits={
                    IFRACategory.CAT_1: 0.0,
                    IFRACategory.CAT_4: 0.05,
                    IFRACategory.CAT_5A: 0.02,
                    IFRACategory.CAT_9: 0.2,
                    IFRACategory.CAT_12: 100.0,
                },
                restriction_type="restricted",
                notes="Strong sensitizer"
            ),
        ]

        for limit in default_limits:
            self.limits_db[limit.ingredient_name.lower()] = limit

    def _load_custom_limits(self, data_path: str):
        """Load custom IFRA limits from JSON file"""
        path = Path(data_path)
        if path.exists():
            with open(path, 'r') as f:
                custom_data = json.load(f)
                for item in custom_data:
                    limit = IFRAIngredientLimit(
                        ingredient_name=item['name'],
                        cas_number=item.get('cas'),
                        category_limits={
                            IFRACategory(k): v
                            for k, v in item['limits'].items()
                        },
                        restriction_type=item.get('type', 'standard'),
                        notes=item.get('notes')
                    )
                    self.limits_db[limit.ingredient_name.lower()] = limit

    def get_limit(self, ingredient_name: str, product_category: ProductCategory) -> Optional[float]:
        """Get IFRA limit for ingredient in specific product category"""
        ing_lower = ingredient_name.lower()
        if ing_lower not in self.limits_db:
            return None  # No restriction if not in database

        limit_data = self.limits_db[ing_lower]
        ifra_category = PRODUCT_TO_IFRA_MAPPING.get(product_category, IFRACategory.CAT_4)

        return limit_data.category_limits.get(ifra_category, 100.0)

    def is_prohibited(self, ingredient_name: str, product_category: ProductCategory) -> bool:
        """Check if ingredient is prohibited in product category"""
        limit = self.get_limit(ingredient_name, product_category)
        return limit == 0.0 if limit is not None else False


class IFRAValidator:
    """Validator for IFRA compliance"""

    def __init__(self, database: Optional[IFRADatabase] = None):
        """Initialize validator with IFRA database"""
        self.database = database or IFRADatabase()

    def check_ifra_violations(
        self,
        recipe: OlfactoryDNA,
        product_category: ProductCategory
    ) -> List[ValidationViolation]:
        """Check recipe for IFRA violations"""
        violations = []

        for ingredient in recipe.ingredients:
            # Get IFRA limit for this ingredient
            limit = self.database.get_limit(ingredient.name, product_category)

            if limit is None:
                continue  # No restriction

            # Check if prohibited
            if limit == 0.0:
                violations.append(
                    ValidationViolation(
                        violation_type="ifra_prohibited",
                        severity="error",
                        ingredient_id=ingredient.ingredient_id,
                        message=f"{ingredient.name} is prohibited in {product_category.value}",
                        current_value=ingredient.concentration,
                        limit_value=0.0,
                        suggestion="Remove this ingredient or choose different product category"
                    )
                )
            # Check if exceeds limit
            elif ingredient.concentration > limit:
                violations.append(
                    ValidationViolation(
                        violation_type="ifra_exceeded",
                        severity="error",
                        ingredient_id=ingredient.ingredient_id,
                        message=f"{ingredient.name} exceeds IFRA limit in {product_category.value}",
                        current_value=ingredient.concentration,
                        limit_value=limit,
                        suggestion=f"Reduce concentration to {limit}% or below"
                    )
                )
            # Warning if close to limit (>80% of limit)
            elif ingredient.concentration > limit * 0.8:
                violations.append(
                    ValidationViolation(
                        violation_type="ifra_warning",
                        severity="warning",
                        ingredient_id=ingredient.ingredient_id,
                        message=f"{ingredient.name} is close to IFRA limit",
                        current_value=ingredient.concentration,
                        limit_value=limit,
                        suggestion="Consider reducing concentration for safety margin"
                    )
                )

        return violations

    def apply_ifra_clipping(
        self,
        recipe: OlfactoryDNA,
        product_category: ProductCategory,
        auto_normalize: bool = True
    ) -> Tuple[OlfactoryDNA, List[str]]:
        """
        Apply IFRA clipping to ensure compliance

        Args:
            recipe: Original recipe
            product_category: Target product category
            auto_normalize: Whether to renormalize after clipping

        Returns:
            Tuple of (clipped_recipe, list_of_changes)
        """
        changes = []
        clipped_ingredients = []

        for ingredient in recipe.ingredients:
            clipped_ing = ingredient.model_copy(deep=True)
            limit = self.database.get_limit(ingredient.name, product_category)

            if limit is not None and limit < ingredient.concentration:
                if limit == 0.0:
                    # Prohibited - remove
                    changes.append(f"Removed prohibited {ingredient.name}")
                    continue
                else:
                    # Clip to limit
                    old_conc = clipped_ing.concentration
                    clipped_ing.concentration = limit
                    changes.append(
                        f"Clipped {ingredient.name} from {old_conc:.2f}% to {limit:.2f}%"
                    )

            clipped_ingredients.append(clipped_ing)

        # Renormalize if requested
        if auto_normalize and clipped_ingredients:
            total = sum(ing.concentration for ing in clipped_ingredients)
            if total > 0 and abs(total - 100.0) > 0.01:
                for ing in clipped_ingredients:
                    ing.concentration = (ing.concentration / total) * 100.0
                changes.append(f"Renormalized total from {total:.2f}% to 100%")

        # Create new DNA with clipped ingredients
        clipped_recipe = recipe.model_copy(deep=True)
        clipped_recipe.ingredients = clipped_ingredients

        return clipped_recipe, changes

    def validate_complete(
        self,
        recipe: OlfactoryDNA,
        product_category: ProductCategory,
        check_balance: bool = True
    ) -> ValidationResult:
        """
        Complete validation including IFRA, balance, and concentration

        Args:
            recipe: Recipe to validate
            product_category: Target product category
            check_balance: Whether to check category balance

        Returns:
            Complete validation result
        """
        violations = []
        warnings = []

        # Check IFRA compliance
        ifra_violations = self.check_ifra_violations(recipe, product_category)
        violations.extend(ifra_violations)

        # Check total concentration
        total_conc = sum(ing.concentration for ing in recipe.ingredients)
        if abs(total_conc - 100.0) > 0.1:
            violations.append(
                ValidationViolation(
                    violation_type="concentration_sum",
                    severity="error",
                    message=f"Total concentration {total_conc:.2f}% != 100%",
                    current_value=total_conc,
                    limit_value=100.0,
                    suggestion="Renormalize concentrations to sum to 100%"
                )
            )

        # Check category balance if requested
        if check_balance and recipe.category_balance:
            balance = recipe.category_balance

            # Check if balanced according to traditional perfumery rules
            if balance['top'] < 15:
                warnings.append(f"Low top notes ({balance['top']:.1f}%)")
            elif balance['top'] > 35:
                warnings.append(f"High top notes ({balance['top']:.1f}%)")

            if balance['heart'] < 25:
                warnings.append(f"Low heart notes ({balance['heart']:.1f}%)")
            elif balance['heart'] > 55:
                warnings.append(f"High heart notes ({balance['heart']:.1f}%)")

            if balance['base'] < 20:
                warnings.append(f"Low base notes ({balance['base']:.1f}%)")
            elif balance['base'] > 50:
                warnings.append(f"High base notes ({balance['base']:.1f}%)")

        # Calculate compliance score
        error_count = sum(1 for v in violations if v.severity == "error")
        warning_count = sum(1 for v in violations if v.severity == "warning")
        score = max(0.0, 1.0 - (error_count * 0.2) - (warning_count * 0.05))

        return ValidationResult(
            is_valid=error_count == 0,
            violations=violations,
            warnings=warnings,
            score=score
        )


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_ifra_penalty(
    recipe: OlfactoryDNA,
    product_category: ProductCategory,
    validator: Optional[IFRAValidator] = None
) -> float:
    """
    Calculate penalty score for IFRA violations

    Returns:
        Penalty score (0 = compliant, higher = more violations)
    """
    if validator is None:
        validator = IFRAValidator()

    violations = validator.check_ifra_violations(recipe, product_category)

    penalty = 0.0
    for violation in violations:
        if violation.severity == "error":
            if violation.violation_type == "ifra_prohibited":
                penalty += 10.0  # Heavy penalty for prohibited ingredients
            elif violation.violation_type == "ifra_exceeded":
                # Penalty proportional to how much limit is exceeded
                if violation.current_value and violation.limit_value:
                    excess_ratio = (violation.current_value - violation.limit_value) / violation.limit_value
                    penalty += 1.0 + excess_ratio * 5.0
        elif violation.severity == "warning":
            penalty += 0.1  # Small penalty for warnings

    return penalty


def ensure_ifra_compliance(
    recipe: OlfactoryDNA,
    product_category: ProductCategory,
    max_iterations: int = 5
) -> OlfactoryDNA:
    """
    Iteratively ensure IFRA compliance through clipping and normalization

    Args:
        recipe: Original recipe
        product_category: Target product category
        max_iterations: Maximum iterations to try

    Returns:
        IFRA-compliant recipe
    """
    validator = IFRAValidator()
    current_recipe = recipe.model_copy(deep=True)

    for iteration in range(max_iterations):
        # Check current compliance
        result = validator.validate_complete(current_recipe, product_category)

        if result.is_valid:
            return current_recipe

        # Apply clipping
        current_recipe, changes = validator.apply_ifra_clipping(
            current_recipe,
            product_category,
            auto_normalize=True
        )

        # Check if we're making progress
        if not changes:
            break

    return current_recipe


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'IFRACategory',
    'IFRAIngredientLimit',
    'IFRADatabase',
    'IFRAValidator',
    'calculate_ifra_penalty',
    'ensure_ifra_compliance',
    'PRODUCT_TO_IFRA_MAPPING'
]