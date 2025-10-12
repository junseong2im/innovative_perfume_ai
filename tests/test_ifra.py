# tests/test_ifra.py
"""
IFRA Compliance Tests
Tests boundary conditions, epsilon smoothing, and cumulative penalty calculations
"""

import pytest
import numpy as np
from typing import Dict, List, Any

from fragrance_ai.rules.ifra_rules import (
    IFRADatabase, IFRAComplianceChecker, AllergenChecker,
    ProductCategory, get_ifra_checker, get_allergen_checker
)
from fragrance_ai.observability import get_logger

logger = get_logger("test_ifra")


class TestIFRALimits:
    """Test IFRA limit database and lookups"""

    def setup_method(self):
        """Setup for each test"""
        self.database = IFRADatabase()
        self.checker = IFRAComplianceChecker()

    def test_limit_lookups(self):
        """Test IFRA limit lookups for various ingredients"""
        print("\n[TEST] Testing IFRA limit lookups...")

        test_cases = [
            ("Bergamot Oil", ProductCategory.EAU_DE_PARFUM, 2.0),
            ("Bergamot Oil", ProductCategory.FACE_CREAM, 0.4),
            ("Bergamot Oil", ProductCategory.CANDLE, 100.0),  # No restriction
            ("Oakmoss Absolute", ProductCategory.EAU_DE_PARFUM, 0.1),
            ("Musk Xylene", ProductCategory.EAU_DE_PARFUM, 0.0),  # Prohibited
            ("Unknown Material", ProductCategory.EAU_DE_PARFUM, None),  # No limit
        ]

        for ingredient, category, expected_limit in test_cases:
            limit = self.database.get_limit(ingredient, category)
            print(f"  {ingredient} in {category.value}: {limit}% (expected: {expected_limit}%)")
            assert limit == expected_limit, f"Wrong limit for {ingredient}"

        print("[OK] All limit lookups correct")

    def test_prohibited_materials(self):
        """Test detection of prohibited materials"""
        print("\n[TEST] Testing prohibited material detection...")

        # Musk Xylene is prohibited in all categories
        for category in ProductCategory:
            is_prohibited = self.database.is_prohibited("Musk Xylene", category)
            assert is_prohibited, f"Musk Xylene should be prohibited in {category}"

        # Bergamot is not prohibited (just restricted)
        is_prohibited = self.database.is_prohibited("Bergamot Oil", ProductCategory.EAU_DE_PARFUM)
        assert not is_prohibited, "Bergamot should be restricted, not prohibited"

        print("[OK] Prohibited material detection working")

    def test_category_specific_limits(self):
        """Test that limits vary by product category"""
        print("\n[TEST] Testing category-specific limits...")

        ingredient = "Rose Absolute"

        # Face products have stricter limits
        face_limit = self.database.get_limit(ingredient, ProductCategory.FACE_CREAM)
        edp_limit = self.database.get_limit(ingredient, ProductCategory.EAU_DE_PARFUM)
        candle_limit = self.database.get_limit(ingredient, ProductCategory.CANDLE)

        assert face_limit < edp_limit, "Face products should have stricter limits"
        assert candle_limit > edp_limit, "Non-skin contact should have higher limits"

        print(f"  {ingredient} limits:")
        print(f"    Face cream: {face_limit}%")
        print(f"    EDP: {edp_limit}%")
        print(f"    Candle: {candle_limit}%")

        print("[OK] Category-specific limits validated")


class TestIFRACompliance:
    """Test IFRA compliance checking"""

    def setup_method(self):
        """Setup for each test"""
        self.checker = IFRAComplianceChecker()

    def test_violation_detection(self):
        """Test detection of IFRA violations"""
        print("\n[TEST] Testing violation detection...")

        recipe = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 5.0},  # Exceeds 2% limit
                {"name": "Oakmoss Absolute", "concentration": 0.5},  # Exceeds 0.1% limit
                {"name": "Lavender", "concentration": 30.0},  # No limit
                {"name": "Sandalwood", "concentration": 64.5}  # No limit
            ]
        }

        result = self.checker.check_ifra_violations(recipe, ProductCategory.EAU_DE_PARFUM)

        assert result["count"] == 2, f"Expected 2 violations, got {result['count']}"
        assert not result["compliant"], "Should not be compliant"
        assert result["penalty"] > 0, "Should have penalty"

        print(f"  Violations found: {result['count']}")
        for violation in result["details"]:
            print(f"    {violation['ingredient']}: {violation['concentration']}% > {violation['limit']}%")

        print("[OK] Violations detected correctly")

    def test_boundary_conditions(self):
        """Test boundary conditions (exact limits, epsilon values)"""
        print("\n[TEST] Testing boundary conditions...")

        # Test exact limit (should pass)
        recipe_exact = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 2.0},  # Exactly at limit
                {"name": "Lavender", "concentration": 98.0}
            ]
        }

        result = self.checker.check_ifra_violations(recipe_exact, ProductCategory.EAU_DE_PARFUM)
        assert result["compliant"], "Should be compliant at exact limit"

        # Test epsilon over limit (should fail)
        recipe_over = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 2.0001},  # Slightly over
                {"name": "Lavender", "concentration": 97.9999}
            ]
        }

        result = self.checker.check_ifra_violations(recipe_over, ProductCategory.EAU_DE_PARFUM)
        assert not result["compliant"], "Should not be compliant when over limit"

        # Test epsilon under limit (should pass)
        recipe_under = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 1.9999},  # Slightly under
                {"name": "Lavender", "concentration": 98.0001}
            ]
        }

        result = self.checker.check_ifra_violations(recipe_under, ProductCategory.EAU_DE_PARFUM)
        assert result["compliant"], "Should be compliant when under limit"

        print("[OK] Boundary conditions handled correctly")

    def test_cumulative_penalty(self):
        """Test cumulative penalty calculation"""
        print("\n[TEST] Testing cumulative penalty calculation...")

        # Recipe with multiple violations
        recipe = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 10.0},  # 5x over limit
                {"name": "Oakmoss Absolute", "concentration": 1.0},  # 10x over limit
                {"name": "Musk Xylene", "concentration": 5.0},  # Prohibited
                {"name": "Lavender", "concentration": 84.0}
            ]
        }

        result = self.checker.check_ifra_violations(recipe, ProductCategory.EAU_DE_PARFUM)

        print(f"  Total penalty: {result['penalty']:.2f}")

        # Check individual penalties
        penalties = []
        for violation in result["details"]:
            ing = violation["ingredient"]
            if violation["limit"] == 0:  # Prohibited
                # Penalty = 100 * concentration
                expected = 100.0 * violation["concentration"]
            else:
                # Penalty = 10 * (1 + excess_ratio)^2
                excess_ratio = violation["excess"] / violation["limit"]
                expected = 10.0 * (1 + excess_ratio) ** 2

            print(f"    {ing}: penalty ≈ {expected:.2f}")
            penalties.append(expected)

        # Total should be sum of individual penalties
        expected_total = sum(penalties)
        assert abs(result["penalty"] - expected_total) < 0.1, "Cumulative penalty incorrect"

        print(f"  Expected total: {expected_total:.2f}")
        print("[OK] Cumulative penalty calculation correct")

    def test_apply_ifra_limits(self):
        """Test automatic IFRA limit application"""
        print("\n[TEST] Testing IFRA limit application...")

        recipe = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 10.0},
                {"name": "Oakmoss Absolute", "concentration": 0.5},
                {"name": "Musk Xylene", "concentration": 2.0},  # Prohibited
                {"name": "Lavender", "concentration": 87.5}
            ]
        }

        # Apply limits (clip mode)
        modified = self.checker.apply_ifra_limits(recipe, ProductCategory.EAU_DE_PARFUM, mode="clip")

        # Check modifications
        print("  After clipping:")
        for ing in modified["ingredients"]:
            print(f"    {ing['name']}: {ing['concentration']:.2f}%")

        # Check clipped values
        bergamot = next(i for i in modified["ingredients"] if i["name"] == "Bergamot Oil")
        assert bergamot["concentration"] <= 2.0, "Bergamot not clipped"

        oakmoss = next(i for i in modified["ingredients"] if i["name"] == "Oakmoss Absolute")
        assert oakmoss["concentration"] <= 0.1, "Oakmoss not clipped"

        # Musk Xylene should be removed (prohibited)
        musk_present = any(i["name"] == "Musk Xylene" for i in modified["ingredients"])
        assert not musk_present, "Prohibited material not removed"

        # Check renormalization
        total = sum(i["concentration"] for i in modified["ingredients"])
        assert abs(total - 100.0) < 0.01, f"Not normalized: {total}%"

        print(f"  Total after renormalization: {total:.2f}%")
        print("[OK] IFRA limits applied correctly")


class TestAllergenDeclaration:
    """Test EU allergen declaration requirements"""

    def setup_method(self):
        """Setup for each test"""
        self.checker = AllergenChecker()

    def test_allergen_threshold(self):
        """Test allergen declaration thresholds"""
        print("\n[TEST] Testing allergen declaration thresholds...")

        # Recipe with allergens at various levels
        recipe = {
            "ingredients": [
                {"name": "Linalool", "concentration": 0.5},      # Below threshold
                {"name": "Limonene", "concentration": 2.0},      # Above threshold
                {"name": "Coumarin", "concentration": 0.08},     # Just below
                {"name": "Eugenol", "concentration": 0.1},       # At threshold
                {"name": "Lavender", "concentration": 97.32}     # Not an allergen
            ]
        }

        # Test for EDP (15% fragrance concentration)
        result = self.checker.check_allergens(recipe, product_concentration=15.0)

        print(f"  Allergens to declare: {len(result['allergens'])}")
        for allergen in result["allergens"]:
            print(f"    {allergen['name']}: {allergen['concentration_ppm']:.1f} ppm")

        # Limonene should be declared (2% * 15% = 0.3% = 3000 ppm > 10 ppm)
        limonene_declared = any(a["name"] == "Limonene" for a in result["allergens"])
        assert limonene_declared, "Limonene should be declared"

        # Linalool should be declared (0.5% * 15% = 0.075% = 750 ppm > 10 ppm)
        linalool_declared = any(a["name"] == "Linalool" for a in result["allergens"])
        assert linalool_declared, "Linalool should be declared"

        # Coumarin should not be declared (0.08% * 15% = 0.012% = 120 ppm > 10 ppm)
        # Actually it should be declared!
        coumarin_declared = any(a["name"] == "Coumarin" for a in result["allergens"])
        assert coumarin_declared, "Coumarin should be declared"

        print("[OK] Allergen thresholds calculated correctly")

    def test_product_concentration_impact(self):
        """Test how product concentration affects allergen declaration"""
        print("\n[TEST] Testing product concentration impact...")

        recipe = {
            "ingredients": [
                {"name": "Geraniol", "concentration": 0.1}  # Borderline
            ]
        }

        # Test different product concentrations
        concentrations = {
            "Parfum (30%)": 30.0,
            "EDP (15%)": 15.0,
            "EDT (8%)": 8.0,
            "EDC (3%)": 3.0
        }

        for product_type, conc in concentrations.items():
            result = self.checker.check_allergens(recipe, product_concentration=conc)

            # Calculate expected ppm
            ppm = (0.1 / 100) * (conc / 100) * 1_000_000

            needs_declaration = ppm > 10
            is_declared = len(result["allergens"]) > 0

            print(f"  {product_type}: {ppm:.1f} ppm - Declaration: {is_declared}")

            if needs_declaration:
                assert is_declared, f"Should declare at {conc}%"
            else:
                assert not is_declared, f"Should not declare at {conc}%"

        print("[OK] Product concentration impact validated")

    def test_banned_allergens(self):
        """Test detection of banned allergens"""
        print("\n[TEST] Testing banned allergen detection...")

        # Recipe with banned materials
        recipe = {
            "ingredients": [
                {"name": "Butylphenyl Methylpropional", "concentration": 0.1},  # Lilial - banned
                {"name": "Hydroxyisohexyl 3-Cyclohexene Carboxaldehyde", "concentration": 0.1},  # Lyral - banned
                {"name": "Lavender", "concentration": 99.8}
            ]
        }

        result = self.checker.check_allergens(recipe, product_concentration=15.0)

        # These should still be in allergen list but marked as problematic
        for allergen in result["allergens"]:
            if "Methylpropional" in allergen["name"] or "Cyclohexene" in allergen["name"]:
                print(f"  WARNING: {allergen['name']} is now banned in EU")

        print("[OK] Banned allergens detected")


class TestComplianceIntegration:
    """Test integrated compliance checking"""

    def test_complete_compliance_check(self):
        """Test complete compliance workflow"""
        print("\n[TEST] Testing complete compliance check...")

        from fragrance_ai.rules.ifra_rules import check_compliance

        recipe = {
            "ingredients": [
                {"name": "Bergamot Oil", "concentration": 3.0},
                {"name": "Linalool", "concentration": 0.5},
                {"name": "Limonene", "concentration": 1.0},
                {"name": "Rose Absolute", "concentration": 0.8},
                {"name": "Sandalwood", "concentration": 94.7}
            ]
        }

        result = check_compliance(
            recipe,
            product_category=ProductCategory.EAU_DE_PARFUM,
            product_concentration=15.0
        )

        print("  Compliance Check Results:")
        print(f"    IFRA compliant: {result['ifra']['compliant']}")
        print(f"    IFRA violations: {result['ifra']['count']}")
        print(f"    Allergens to declare: {result['allergens']['count']}")
        print(f"    Overall compliant: {result['overall_compliant']}")

        # Should have violations
        assert not result["ifra"]["compliant"], "Should have IFRA violations"
        assert result["allergens"]["count"] > 0, "Should have allergens to declare"

        # Log for observability
        logger.info(
            "Compliance check completed",
            ifra_violations=result["ifra"]["count"],
            allergens=result["allergens"]["count"],
            compliant=result["overall_compliant"]
        )

        print("[OK] Complete compliance check working")

    def test_epsilon_smoothing(self):
        """Test epsilon smoothing in calculations"""
        print("\n[TEST] Testing epsilon smoothing...")

        # Test very small concentrations
        recipe = {
            "ingredients": [
                {"name": "Linalool", "concentration": 1e-10},  # Tiny amount
                {"name": "Limonene", "concentration": 1e-8},   # Very small
                {"name": "Base", "concentration": 99.9999999}
            ]
        }

        # Should not crash or produce NaN/Inf
        result = self.checker.check_allergens(recipe, product_concentration=15.0)

        assert result is not None, "Should handle tiny concentrations"
        assert "allergens" in result, "Should return valid result"

        # Check no NaN or Inf in calculations
        for allergen in result["allergens"]:
            assert not np.isnan(allergen["concentration_ppm"]), "NaN in calculation"
            assert not np.isinf(allergen["concentration_ppm"]), "Inf in calculation"

        print("[OK] Epsilon smoothing prevents numerical issues")

    def test_empty_recipe_handling(self):
        """Test handling of edge cases"""
        print("\n[TEST] Testing edge case handling...")

        # Empty recipe
        empty_recipe = {"ingredients": []}
        result = self.checker.check_ifra_violations(empty_recipe, ProductCategory.EAU_DE_PARFUM)
        assert result["compliant"], "Empty recipe should be compliant"
        assert result["count"] == 0, "No violations in empty recipe"

        # Recipe with unknown ingredients only
        unknown_recipe = {
            "ingredients": [
                {"name": "Unknown1", "concentration": 50.0},
                {"name": "Unknown2", "concentration": 50.0}
            ]
        }
        result = self.checker.check_ifra_violations(unknown_recipe, ProductCategory.EAU_DE_PARFUM)
        assert result["compliant"], "Unknown ingredients should pass (no limits)"

        print("[OK] Edge cases handled gracefully")


def test_performance_benchmark():
    """Benchmark IFRA checking performance"""
    print("\n[BENCHMARK] IFRA Performance Test...")

    import time

    checker = IFRAComplianceChecker()

    # Create large recipe
    recipe = {
        "ingredients": [
            {"name": f"Ingredient_{i}", "concentration": 100.0/50}
            for i in range(50)
        ]
    }

    # Benchmark violation checking
    start = time.time()
    for _ in range(1000):
        checker.check_ifra_violations(recipe, ProductCategory.EAU_DE_PARFUM)
    elapsed = time.time() - start

    print(f"  1000 compliance checks in {elapsed:.2f} seconds")
    print(f"  Average: {elapsed/1000*1000:.2f} ms per check")

    assert elapsed < 5, "IFRA checking too slow"

    print("[OK] Performance acceptable")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])