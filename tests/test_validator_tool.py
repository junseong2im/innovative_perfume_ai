"""
Unit tests for ValidatorTool
Tests validation against perfume blending rules
"""

import unittest
import json
import os
import sys
from pathlib import Path
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.tools.validator_tool import (
    ScientificValidator,
    NotesComposition,
    ValidationResult,
    validate_composition,
    get_validator
)


class TestValidatorTool(unittest.TestCase):
    """Test suite for ValidatorTool functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.test_rules_path = Path(__file__).parent.parent / "assets" / "perfume_blending_rules.json"
        cls.validator = None

    def setUp(self):
        """Set up each test"""
        self.validator = ScientificValidator()

        # Load blending rules
        if self.test_rules_path.exists():
            with open(self.test_rules_path, 'r') as f:
                self.rules = json.load(f)['blending_rules']
        else:
            self.rules = None

    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        validator = ScientificValidator()
        self.assertIsNotNone(validator)
        self.assertIsNotNone(validator.device)

    def test_valid_composition(self):
        """Test validation of a valid composition"""
        composition = NotesComposition(
            top_notes=[
                {"bergamot": 10.0},
                {"lemon": 5.0}
            ],
            heart_notes=[
                {"jasmine": 20.0},
                {"rose": 15.0}
            ],
            base_notes=[
                {"sandalwood": 25.0},
                {"musk": 20.0}
            ],
            total_ingredients=6
        )

        result = self.validator.validate(composition)

        self.assertIsInstance(result, ValidationResult)
        self.assertIsInstance(result.is_valid, bool)
        self.assertGreaterEqual(result.harmony_score, 0)
        self.assertLessEqual(result.harmony_score, 10)
        self.assertGreaterEqual(result.stability_score, 0)
        self.assertLessEqual(result.stability_score, 10)
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 1)

    def test_invalid_composition_too_few_notes(self):
        """Test validation fails for too few notes"""
        composition = NotesComposition(
            top_notes=[{"bergamot": 10.0}],  # Only 1 top note
            heart_notes=[],  # No heart notes
            base_notes=[{"musk": 80.0}],
            total_ingredients=2
        )

        result = self.validator.validate(composition)

        self.assertIsInstance(result, ValidationResult)
        # Should have low scores due to poor balance
        self.assertLess(result.overall_score, 7.0)
        self.assertGreater(len(result.key_risks), 0)

    def test_composition_balance_check(self):
        """Test that validator checks note balance"""
        # Unbalanced composition - too much base
        composition = NotesComposition(
            top_notes=[{"lemon": 5.0}],
            heart_notes=[{"rose": 10.0}],
            base_notes=[
                {"sandalwood": 40.0},
                {"amber": 30.0},
                {"musk": 15.0}
            ],
            total_ingredients=5
        )

        result = self.validator.validate(composition)

        # Should detect imbalance
        self.assertLess(result.stability_score, 8.0)

    def test_rule_compliance(self):
        """Test validation against blending rules"""
        if not self.rules:
            self.skipTest("Blending rules not available")

        # Test max percentage rule
        composition = NotesComposition(
            top_notes=[{"bergamot": 50.0}],  # Single note too high
            heart_notes=[{"jasmine": 30.0}],
            base_notes=[{"musk": 25.0}],
            total_ingredients=3
        )

        result = self.validator.validate(composition)

        # Should have warnings about concentration
        self.assertIsNotNone(result.scientific_notes)

    def test_scientific_score_calculation(self):
        """Test that scientific scores are calculated correctly"""
        composition = NotesComposition(
            top_notes=[
                {"bergamot": 8.0},
                {"lemon": 7.0}
            ],
            heart_notes=[
                {"jasmine": 15.0},
                {"rose": 10.0},
                {"ylang": 10.0}
            ],
            base_notes=[
                {"sandalwood": 20.0},
                {"amber": 15.0},
                {"musk": 15.0}
            ],
            total_ingredients=8
        )

        result = self.validator.validate(composition)

        # Check all scores are present and valid
        self.assertIsNotNone(result.harmony_score)
        self.assertIsNotNone(result.stability_score)
        self.assertIsNotNone(result.longevity_score)
        self.assertIsNotNone(result.sillage_score)
        self.assertIsNotNone(result.overall_score)

        # Overall should be average of other scores
        expected_overall = (
            result.harmony_score +
            result.stability_score +
            result.longevity_score +
            result.sillage_score
        ) / 4

        self.assertAlmostEqual(result.overall_score, expected_overall, places=1)

    def test_risk_detection(self):
        """Test that validator detects risks properly"""
        # Poor composition
        composition = NotesComposition(
            top_notes=[],
            heart_notes=[{"rose": 100.0}],  # Only one note at 100%
            base_notes=[],
            total_ingredients=1
        )

        result = self.validator.validate(composition)

        # Should detect multiple risks
        self.assertGreater(len(result.key_risks), 0)
        self.assertGreater(len(result.suggestions), 0)
        self.assertFalse(result.is_valid)

    def test_suggestion_generation(self):
        """Test that validator provides helpful suggestions"""
        # Composition with low longevity
        composition = NotesComposition(
            top_notes=[
                {"lemon": 20.0},
                {"bergamot": 20.0}
            ],
            heart_notes=[{"rose": 30.0}],
            base_notes=[{"light_musk": 30.0}],  # Light base = poor longevity
            total_ingredients=4
        )

        result = self.validator.validate(composition)

        # Should suggest improvements
        self.assertGreater(len(result.suggestions), 0)
        # Check for longevity-related suggestion
        longevity_suggestion_found = any(
            "베이스" in s or "지속" in s or "픽서티브" in s
            for s in result.suggestions
        )
        self.assertTrue(longevity_suggestion_found or result.longevity_score > 7)

    def test_fallback_validation(self):
        """Test fallback validation when model unavailable"""
        with patch.object(self.validator, 'model', None):
            composition = NotesComposition(
                top_notes=[{"bergamot": 15.0}],
                heart_notes=[{"jasmine": 35.0}],
                base_notes=[{"sandalwood": 30.0}],
                total_ingredients=3
            )

            result = self.validator.validate(composition)

            # Should still return valid result
            self.assertIsInstance(result, ValidationResult)
            # Confidence should be lower for fallback
            self.assertLess(result.confidence, 0.6)

    def test_async_validation(self):
        """Test async validation function"""
        composition = NotesComposition(
            top_notes=[{"bergamot": 15.0}],
            heart_notes=[{"jasmine": 35.0}],
            base_notes=[{"sandalwood": 30.0}],
            total_ingredients=3
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(validate_composition(composition))
        loop.close()

        self.assertIsInstance(result, ValidationResult)

    def test_singleton_validator(self):
        """Test that get_validator returns singleton"""
        validator1 = get_validator()
        validator2 = get_validator()

        self.assertIs(validator1, validator2)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Empty composition
        composition = NotesComposition(
            top_notes=[],
            heart_notes=[],
            base_notes=[],
            total_ingredients=0
        )

        result = self.validator.validate(composition)
        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_valid)

        # Very large composition
        composition = NotesComposition(
            top_notes=[{f"note_{i}": 1.0} for i in range(10)],
            heart_notes=[{f"note_{i}": 1.0} for i in range(10, 20)],
            base_notes=[{f"note_{i}": 1.0} for i in range(20, 30)],
            total_ingredients=30
        )

        result = self.validator.validate(composition)
        self.assertIsInstance(result, ValidationResult)

    def test_scientific_notes_generation(self):
        """Test that scientific notes are generated properly"""
        composition = NotesComposition(
            top_notes=[{"bergamot": 15.0}],
            heart_notes=[{"jasmine": 35.0}],
            base_notes=[{"sandalwood": 30.0}],
            total_ingredients=3
        )

        result = self.validator.validate(composition)

        self.assertIsNotNone(result.scientific_notes)
        self.assertGreater(len(result.scientific_notes), 0)
        # Should contain score information
        self.assertIn("/10", result.scientific_notes)


class TestValidatorRuleCompliance(unittest.TestCase):
    """Test compliance with specific blending rules"""

    def setUp(self):
        """Load rules and create validator"""
        self.validator = ScientificValidator()
        rules_path = Path(__file__).parent.parent / "assets" / "perfume_blending_rules.json"

        if rules_path.exists():
            with open(rules_path, 'r') as f:
                self.rules = json.load(f)['blending_rules']
        else:
            self.rules = None

    def test_percentage_limits(self):
        """Test total percentage validation"""
        if not self.rules:
            self.skipTest("Rules not available")

        # Over 100% total
        composition = NotesComposition(
            top_notes=[{"bergamot": 40.0}],
            heart_notes=[{"jasmine": 40.0}],
            base_notes=[{"sandalwood": 40.0}],
            total_ingredients=3
        )

        result = self.validator.validate(composition)

        # Should detect the issue
        self.assertLess(result.overall_score, 8.0)

    def test_note_count_limits(self):
        """Test note count validation against rules"""
        if not self.rules:
            self.skipTest("Rules not available")

        max_top = self.rules['note_balance']['top_notes']['max_count']

        # Too many top notes
        composition = NotesComposition(
            top_notes=[{f"note_{i}": 2.0} for i in range(max_top + 2)],
            heart_notes=[{"jasmine": 30.0}],
            base_notes=[{"sandalwood": 30.0}],
            total_ingredients=max_top + 4
        )

        result = self.validator.validate(composition)

        # Should have lower harmony score
        self.assertLess(result.harmony_score, 9.0)


if __name__ == '__main__':
    unittest.main()