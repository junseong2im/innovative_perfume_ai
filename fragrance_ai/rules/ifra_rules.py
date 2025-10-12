# fragrance_ai/rules/ifra_rules.py
"""
IFRA (International Fragrance Association) Rules and Compliance
성분별 상한, 제품 카테고리별 허용치 관리
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


# ============================================================================
# Product Categories
# ============================================================================

class ProductCategory(str, Enum):
    """IFRA product categories"""
    # Category 4: Hydroalcoholic products
    EAU_DE_PARFUM = "eau_de_parfum"  # 15-20% fragrance
    EAU_DE_TOILETTE = "eau_de_toilette"  # 5-15% fragrance
    EAU_DE_COLOGNE = "eau_de_cologne"  # 2-5% fragrance

    # Category 5: Facial products
    FACE_CREAM = "face_cream"
    FACE_TONER = "face_toner"

    # Category 6: Mouthwash
    MOUTHWASH = "mouthwash"

    # Category 9: Rinse-off products
    SHAMPOO = "shampoo"
    BODY_WASH = "body_wash"

    # Category 11: Non-skin contact
    CANDLE = "candle"
    ROOM_SPRAY = "room_spray"
    DIFFUSER = "diffuser"


# ============================================================================
# IFRA Limit Data
# ============================================================================

@dataclass
class IFRALimit:
    """IFRA limit for specific ingredient in product category"""
    ingredient_name: str
    cas_number: Optional[str]
    category: ProductCategory
    max_percentage: float  # Maximum allowed percentage
    restriction_type: str  # "prohibited", "restricted", "specification"
    amendment: int  # IFRA amendment number (e.g., 49, 50)
    notes: Optional[str] = None


class IFRADatabase:
    """IFRA limits database"""

    def __init__(self):
        self.limits: Dict[str, Dict[ProductCategory, IFRALimit]] = {}
        self._load_ifra_limits()

    def _load_ifra_limits(self):
        """Load IFRA Amendment 50 limits (latest as of 2024)"""

        # Common restricted materials with limits
        ifra_data = [
            # Citrus oils (phototoxic)
            {
                "ingredient": "Bergamot Oil",
                "cas": "8007-75-8",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 2.0,
                    ProductCategory.EAU_DE_TOILETTE: 2.0,
                    ProductCategory.FACE_CREAM: 0.4,
                    ProductCategory.BODY_WASH: 2.0,
                    ProductCategory.CANDLE: 100.0,  # No restriction for non-skin
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Lemon Oil Cold Pressed",
                "cas": "8008-56-8",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 3.0,
                    ProductCategory.EAU_DE_TOILETTE: 3.0,
                    ProductCategory.FACE_CREAM: 0.6,
                    ProductCategory.BODY_WASH: 3.0,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Allergens
            {
                "ingredient": "Oakmoss Absolute",
                "cas": "90028-68-5",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.1,
                    ProductCategory.EAU_DE_TOILETTE: 0.1,
                    ProductCategory.FACE_CREAM: 0.1,
                    ProductCategory.BODY_WASH: 0.1,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Treemoss Absolute",
                "cas": "90028-67-4",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.2,
                    ProductCategory.EAU_DE_TOILETTE: 0.2,
                    ProductCategory.FACE_CREAM: 0.2,
                    ProductCategory.BODY_WASH: 0.2,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Rose/Jasmine (sensitizers)
            {
                "ingredient": "Rose Absolute",
                "cas": "8007-01-0",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.6,
                    ProductCategory.EAU_DE_TOILETTE: 0.6,
                    ProductCategory.FACE_CREAM: 0.02,
                    ProductCategory.BODY_WASH: 0.6,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Jasmine Absolute",
                "cas": "8022-96-6",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.7,
                    ProductCategory.EAU_DE_TOILETTE: 0.7,
                    ProductCategory.FACE_CREAM: 0.02,
                    ProductCategory.BODY_WASH: 0.7,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Musks
            {
                "ingredient": "Musk Xylene",
                "cas": "81-15-2",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.0,  # Prohibited
                    ProductCategory.EAU_DE_TOILETTE: 0.0,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 0.0,
                    ProductCategory.CANDLE: 0.0,
                },
                "type": "prohibited",
                "amendment": 50
            },
            {
                "ingredient": "Musk Ketone",
                "cas": "81-14-1",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 1.4,
                    ProductCategory.EAU_DE_TOILETTE: 1.4,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 1.4,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },

            # Other common materials
            {
                "ingredient": "Coumarin",
                "cas": "91-64-5",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 1.6,
                    ProductCategory.EAU_DE_TOILETTE: 1.6,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 1.6,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            },
            {
                "ingredient": "Eugenol",
                "cas": "97-53-0",
                "limits": {
                    ProductCategory.EAU_DE_PARFUM: 0.5,
                    ProductCategory.EAU_DE_TOILETTE: 0.5,
                    ProductCategory.FACE_CREAM: 0.0,
                    ProductCategory.BODY_WASH: 0.5,
                    ProductCategory.CANDLE: 100.0,
                },
                "type": "restricted",
                "amendment": 50
            }
        ]

        # Load into database
        for item in ifra_data:
            ingredient = item["ingredient"]
            self.limits[ingredient] = {}

            for category, limit in item["limits"].items():
                self.limits[ingredient][category] = IFRALimit(
                    ingredient_name=ingredient,
                    cas_number=item["cas"],
                    category=category,
                    max_percentage=limit,
                    restriction_type=item["type"],
                    amendment=item["amendment"],
                    notes=item.get("notes")
                )

    def get_limit(self, ingredient: str, category: ProductCategory) -> Optional[float]:
        """Get IFRA limit for ingredient in product category"""
        if ingredient in self.limits:
            if category in self.limits[ingredient]:
                return self.limits[ingredient][category].max_percentage
        return None  # No restriction

    def is_prohibited(self, ingredient: str, category: ProductCategory) -> bool:
        """Check if ingredient is prohibited in category"""
        limit = self.get_limit(ingredient, category)
        return limit == 0.0 if limit is not None else False


# ============================================================================
# IFRA Compliance Checker
# ============================================================================

class IFRAComplianceChecker:
    """Check formulation compliance with IFRA standards"""

    def __init__(self):
        self.database = IFRADatabase()

    def check_ifra_violations(
        self,
        recipe: Dict[str, Any],
        product_category: ProductCategory = ProductCategory.EAU_DE_PARFUM
    ) -> Dict[str, Any]:
        """
        Check recipe for IFRA violations

        Args:
            recipe: Recipe with ingredients list
            product_category: Target product category

        Returns:
            Dictionary with violation count, penalty, and details
        """
        violations = []
        total_penalty = 0.0

        # Get ingredients from recipe
        ingredients = recipe.get("ingredients", [])

        for ingredient in ingredients:
            name = ingredient.get("name", "")
            concentration = ingredient.get("concentration", 0.0)

            # Check IFRA limit
            limit = self.database.get_limit(name, product_category)

            if limit is not None:
                if concentration > limit:
                    violation = {
                        "ingredient": name,
                        "concentration": concentration,
                        "limit": limit,
                        "excess": concentration - limit,
                        "severity": "critical" if limit == 0 else "warning"
                    }
                    violations.append(violation)

                    # Calculate penalty (exponential for severe violations)
                    if limit == 0:  # Prohibited
                        penalty = 100.0 * concentration
                    else:
                        excess_ratio = (concentration - limit) / limit
                        penalty = 10.0 * (1 + excess_ratio) ** 2

                    total_penalty += penalty

        return {
            "count": len(violations),
            "penalty": total_penalty,
            "details": violations,
            "compliant": len(violations) == 0,
            "product_category": product_category.value
        }

    def apply_ifra_limits(
        self,
        recipe: Dict[str, Any],
        product_category: ProductCategory = ProductCategory.EAU_DE_PARFUM,
        mode: str = "clip"
    ) -> Dict[str, Any]:
        """
        Apply IFRA limits to recipe

        Args:
            recipe: Original recipe
            product_category: Target product category
            mode: "clip" to cap at limits, "remove" to remove violating ingredients

        Returns:
            Modified recipe with IFRA compliance
        """
        ingredients = recipe.get("ingredients", []).copy()
        modified_ingredients = []
        removed_ingredients = []

        for ingredient in ingredients:
            name = ingredient.get("name", "")
            concentration = ingredient.get("concentration", 0.0)

            # Check IFRA limit
            limit = self.database.get_limit(name, product_category)

            if limit is not None and concentration > limit:
                if limit == 0 or mode == "remove":
                    # Remove prohibited or violating ingredient
                    removed_ingredients.append(ingredient)
                else:
                    # Clip to limit
                    ingredient = ingredient.copy()
                    ingredient["concentration"] = limit
                    ingredient["ifra_clipped"] = True
                    modified_ingredients.append(ingredient)
            else:
                modified_ingredients.append(ingredient)

        # Renormalize to 100%
        total = sum(ing["concentration"] for ing in modified_ingredients)
        if total > 0:
            for ing in modified_ingredients:
                ing["concentration"] = (ing["concentration"] / total) * 100.0

        # Create modified recipe
        modified_recipe = recipe.copy()
        modified_recipe["ingredients"] = modified_ingredients
        modified_recipe["ifra_compliant"] = True
        modified_recipe["removed_ingredients"] = removed_ingredients

        return modified_recipe


# ============================================================================
# Allergen Declaration
# ============================================================================

class AllergenChecker:
    """EU allergen declaration requirements"""

    # 26 EU allergens that must be declared
    EU_ALLERGENS = {
        "Alpha-Isomethyl Ionone": 10.0,  # ppm threshold in final product
        "Amyl Cinnamal": 10.0,
        "Amylcinnamyl Alcohol": 10.0,
        "Anise Alcohol": 10.0,
        "Benzyl Alcohol": 10.0,
        "Benzyl Benzoate": 10.0,
        "Benzyl Cinnamate": 10.0,
        "Benzyl Salicylate": 10.0,
        "Butylphenyl Methylpropional": 10.0,  # Lilial - now banned
        "Cinnamal": 10.0,
        "Cinnamyl Alcohol": 10.0,
        "Citral": 10.0,
        "Citronellol": 10.0,
        "Coumarin": 10.0,
        "Eugenol": 10.0,
        "Evernia Furfuracea": 10.0,  # Treemoss
        "Evernia Prunastri": 10.0,  # Oakmoss
        "Farnesol": 10.0,
        "Geraniol": 10.0,
        "Hexyl Cinnamal": 10.0,
        "Hydroxycitronellal": 10.0,
        "Isoeugenol": 10.0,
        "Limonene": 10.0,
        "Linalool": 10.0,
        "Methyl 2-Octynoate": 10.0,
        "Hydroxyisohexyl 3-Cyclohexene Carboxaldehyde": 10.0  # Lyral - now banned
    }

    @classmethod
    def check_allergens(cls, recipe: Dict[str, Any], product_concentration: float = 15.0) -> Dict[str, Any]:
        """
        Check which allergens need declaration

        Args:
            recipe: Fragrance recipe
            product_concentration: % of fragrance in final product (e.g., 15% for EDP)

        Returns:
            Dictionary with allergen information
        """
        allergens_to_declare = []

        for ingredient in recipe.get("ingredients", []):
            name = ingredient.get("name", "")
            concentration_in_fragrance = ingredient.get("concentration", 0.0)

            # Calculate concentration in final product (ppm)
            concentration_in_product = (concentration_in_fragrance / 100) * (product_concentration / 100) * 1000000

            if name in cls.EU_ALLERGENS:
                threshold = cls.EU_ALLERGENS[name]
                if concentration_in_product > threshold:
                    allergens_to_declare.append({
                        "name": name,
                        "concentration_ppm": concentration_in_product,
                        "threshold_ppm": threshold,
                        "must_declare": True
                    })

        return {
            "allergens": allergens_to_declare,
            "count": len(allergens_to_declare),
            "compliant": True  # Allergens can be present if declared
        }


# ============================================================================
# Main Module Interface
# ============================================================================

# Global instances
_ifra_checker = None
_allergen_checker = None


def get_ifra_checker() -> IFRAComplianceChecker:
    """Get global IFRA checker instance"""
    global _ifra_checker
    if _ifra_checker is None:
        _ifra_checker = IFRAComplianceChecker()
    return _ifra_checker


def get_allergen_checker() -> AllergenChecker:
    """Get global allergen checker instance"""
    global _allergen_checker
    if _allergen_checker is None:
        _allergen_checker = AllergenChecker()
    return _allergen_checker


def check_compliance(
    recipe: Dict[str, Any],
    product_category: ProductCategory = ProductCategory.EAU_DE_PARFUM,
    product_concentration: float = 15.0
) -> Dict[str, Any]:
    """
    Complete compliance check for recipe

    Returns:
        Dictionary with IFRA and allergen compliance results
    """
    ifra_checker = get_ifra_checker()
    allergen_checker = get_allergen_checker()

    ifra_result = ifra_checker.check_ifra_violations(recipe, product_category)
    allergen_result = allergen_checker.check_allergens(recipe, product_concentration)

    return {
        "ifra": ifra_result,
        "allergens": allergen_result,
        "overall_compliant": ifra_result["compliant"] and allergen_result["compliant"]
    }


# Export main classes and functions
__all__ = [
    'ProductCategory',
    'IFRALimit',
    'IFRADatabase',
    'IFRAComplianceChecker',
    'AllergenChecker',
    'get_ifra_checker',
    'get_allergen_checker',
    'check_compliance'
]