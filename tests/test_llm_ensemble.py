# tests/test_llm_ensemble.py
"""
Comprehensive tests for LLM Ensemble
Tests all 3 modes (fast/balanced/creative) with validation, fallback, and caching
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

from fragrance_ai.llm import build_brief
from fragrance_ai.llm.schemas import CreativeBrief, DEFAULT_BRIEF
from fragrance_ai.llm.llm_router import route_mode, detect_language
from fragrance_ai.llm.qwen_client import QwenClient
from fragrance_ai.llm.mistral_validator import validate_and_patch
from fragrance_ai.llm.llama_hints import LlamaHintsGenerator
from fragrance_ai.llm.brief_mapper import map_llm_brief_to_domain, extract_moga_constraints

logger = logging.getLogger(__name__)


# ============================================================================
# Test Data
# ============================================================================

KOREAN_INPUT_SHORT = "시트러스 30% 우디 50% 비율 정확하게"  # Has % and 비율/정확 = 2+ fast keywords
KOREAN_INPUT_MEDIUM = "여름에 사용할 상쾌한 향수를 만들고 싶어요. 레몬과 베르가못 향이 강했으면 좋겠습니다."
KOREAN_INPUT_LONG = """
여름날 아침 정원에서의 추억을 담은 향수를 만들고 싶어요.
어린 시절 할머니 댁 뒤뜰에서 맡던 재스민 향기와 레몬 나무의 상쾌한 냄새,
그리고 따뜻한 햇살 아래 느껴지던 편안함을 향수로 표현하고 싶습니다.
몽환적이고 시적인 분위기를 원합니다.
"""

ENGLISH_INPUT = "I want a fresh citrus perfume for daily use in summer, with bergamot and lemon notes"


# ============================================================================
# Test: Schema Validation
# ============================================================================

class TestSchemaValidation:
    """Test CreativeBrief schema validation"""

    def test_valid_brief(self):
        """Valid brief should pass"""
        brief = CreativeBrief(
            language="ko",
            mood=["fresh", "energetic"],
            season=["summer"],
            notes_preference={"citrus": 0.4, "floral": 0.3, "woody": 0.3},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )
        assert brief.language == "ko"
        assert len(brief.mood) == 2
        assert sum(brief.notes_preference.values()) == pytest.approx(1.0, rel=0.01)

    def test_notes_preference_normalization(self):
        """notes_preference should normalize if sum > 1"""
        brief = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 0.6, "floral": 0.5, "woody": 0.4},  # Sum = 1.5
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )
        # Should be normalized to sum = 1.0
        total = sum(brief.notes_preference.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_notes_preference_clipping(self):
        """notes_preference values should clip to [0, 1]"""
        brief = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 1.5, "floral": -0.2},  # Out of bounds
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )
        # Values should be clipped to [0, 1]
        for value in brief.notes_preference.values():
            assert 0.0 <= value <= 1.0

    def test_creative_hints_validation(self):
        """creative_hints should validate length and count"""
        brief = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP",
            creative_hints=["a" * 60, "b", "valid hint", "x" * 50] + ["hint"] * 10
        )
        # Should have max 8 hints, each 2-48 chars
        assert len(brief.creative_hints) <= 8
        for hint in brief.creative_hints:
            assert 2 <= len(hint) <= 48

    def test_default_brief(self):
        """DEFAULT_BRIEF should be valid"""
        assert DEFAULT_BRIEF.language == "ko"
        assert DEFAULT_BRIEF.product_category in ["EDP", "EDT", "PARFUM"]
        assert DEFAULT_BRIEF.budget_tier == "mid"


# ============================================================================
# Test: LLM Router
# ============================================================================

class TestLLMRouter:
    """Test mode routing and language detection"""

    def test_route_fast_mode(self):
        """Short input with numeric keywords → fast"""
        mode = route_mode(KOREAN_INPUT_SHORT)
        assert mode == "fast"

    def test_route_balanced_mode(self):
        """Medium input without creative keywords → balanced"""
        mode = route_mode(KOREAN_INPUT_MEDIUM)
        assert mode == "balanced"

    def test_route_creative_mode(self):
        """Long input with creative keywords → creative"""
        mode = route_mode(KOREAN_INPUT_LONG)
        assert mode == "creative"

    def test_detect_korean(self):
        """Korean text should detect as 'ko'"""
        lang = detect_language(KOREAN_INPUT_SHORT)
        assert lang == "ko"

    def test_detect_english(self):
        """English text should detect as 'en'"""
        lang = detect_language(ENGLISH_INPUT)
        assert lang == "en"


# ============================================================================
# Test: Mistral Validator
# ============================================================================

class TestMistralValidator:
    """Test schema validation and patching"""

    def test_validate_and_patch_defaults(self):
        """Should set defaults for missing fields"""
        brief = CreativeBrief(
            language="ko",
            mood=[],  # Empty
            season=[],  # Empty
            notes_preference={},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category=None  # Missing
        )

        patched = validate_and_patch(brief)

        # Should have defaults
        assert patched.product_category == "EDP"
        assert len(patched.mood) > 0  # Default mood
        assert len(patched.season) > 0  # Default season

    def test_validate_notes_preference_sum(self):
        """Should normalize notes_preference if sum > 1"""
        brief = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 0.8, "floral": 0.8},  # Sum = 1.6
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )

        patched = validate_and_patch(brief)
        total = sum(patched.notes_preference.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_validate_constraints_default(self):
        """Should add default max_allergens_ppm to constraints"""
        brief = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP",
            constraints={}  # Empty constraints
        )

        patched = validate_and_patch(brief)
        # Should add default max_allergens_ppm
        assert "max_allergens_ppm" in patched.constraints
        assert patched.constraints["max_allergens_ppm"] == 500.0


# ============================================================================
# Test: build_brief with Mocks
# ============================================================================

class TestBuildBriefMocked:
    """Test build_brief function with mocked LLM clients"""

    @patch('fragrance_ai.llm.qwen_client.get_qwen_client')
    @patch('fragrance_ai.llm.llama_hints.get_llama_generator')
    def test_fast_mode_qwen_only(self, mock_llama, mock_qwen):
        """Fast mode should use Qwen only"""
        # Mock Qwen client
        mock_qwen_client = Mock()
        mock_qwen_client.infer_brief.return_value = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 1.0},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )
        mock_qwen.return_value = mock_qwen_client

        # Call build_brief with fast mode
        brief = build_brief(KOREAN_INPUT_SHORT, mode="fast", use_cache=False)

        # Assertions
        assert brief is not None
        mock_qwen_client.infer_brief.assert_called_once()
        mock_llama.assert_not_called()  # Llama should NOT be called in fast mode

    @patch('fragrance_ai.llm.qwen_client.get_qwen_client')
    @patch('fragrance_ai.llm.llama_hints.get_llama_generator')
    def test_balanced_mode_qwen_mistral(self, mock_llama, mock_qwen):
        """Balanced mode should use Qwen + Mistral validation"""
        # Mock Qwen client
        mock_qwen_client = Mock()
        mock_qwen_client.infer_brief.return_value = CreativeBrief(
            language="ko",
            mood=[],  # Empty, to be filled by Mistral
            season=["summer"],
            notes_preference={"citrus": 1.5, "woody": 0.8},  # Sum > 1, to be normalized
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category=None  # Missing, to be filled by Mistral
        )
        mock_qwen.return_value = mock_qwen_client

        # Call build_brief with balanced mode
        brief = build_brief(KOREAN_INPUT_MEDIUM, mode="balanced", use_cache=False)

        # Assertions
        assert brief is not None
        mock_qwen_client.infer_brief.assert_called_once()
        assert brief.product_category == "EDP"  # Filled by Mistral
        assert len(brief.mood) > 0  # Filled by Mistral with default
        # notes_preference should be normalized
        total = sum(brief.notes_preference.values())
        assert total == pytest.approx(1.0, rel=0.01)
        mock_llama.assert_not_called()  # Llama should NOT be called in balanced mode

    @patch('fragrance_ai.llm.qwen_client.get_qwen_client')
    @patch('fragrance_ai.llm.llama_hints.get_llama_generator')
    def test_creative_mode_all_three(self, mock_llama, mock_qwen):
        """Creative mode should use Qwen + Mistral + Llama"""
        # Mock Qwen client
        mock_qwen_client = Mock()
        mock_qwen_client.infer_brief.return_value = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 1.0},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP",
            creative_hints=[]  # Empty, to be filled by Llama
        )
        mock_qwen.return_value = mock_qwen_client

        # Mock Llama generator
        mock_llama_gen = Mock()
        mock_llama_gen.generate_hints.return_value = ["garden", "sunshine", "nostalgia"]
        mock_llama.return_value = mock_llama_gen

        # Call build_brief with creative mode
        brief = build_brief(KOREAN_INPUT_LONG, mode="creative", use_cache=False)

        # Assertions
        assert brief is not None
        mock_qwen_client.infer_brief.assert_called_once()
        mock_llama_gen.generate_hints.assert_called_once()
        assert len(brief.creative_hints) >= 3  # Should have Llama hints

    @patch('fragrance_ai.llm.qwen_client.get_qwen_client')
    def test_fallback_to_default(self, mock_qwen):
        """Should fallback to DEFAULT_BRIEF if Qwen fails"""
        # Mock Qwen client to raise exception
        mock_qwen_client = Mock()
        mock_qwen_client.infer_brief.side_effect = Exception("Qwen failed")
        mock_qwen.return_value = mock_qwen_client

        # Call build_brief
        brief = build_brief(KOREAN_INPUT_SHORT, mode="fast", use_cache=False, retry=0)

        # Should return DEFAULT_BRIEF
        assert brief.language == DEFAULT_BRIEF.language
        assert brief.product_category == DEFAULT_BRIEF.product_category

    @patch('fragrance_ai.llm.qwen_client.get_qwen_client')
    def test_caching(self, mock_qwen):
        """Should cache and reuse results"""
        # Mock Qwen client
        mock_qwen_client = Mock()
        mock_qwen_client.infer_brief.return_value = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 1.0},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )
        mock_qwen.return_value = mock_qwen_client

        # First call
        brief1 = build_brief("test input", mode="fast", use_cache=True)

        # Second call with same input
        brief2 = build_brief("test input", mode="fast", use_cache=True)

        # Qwen should only be called once (second call uses cache)
        assert mock_qwen_client.infer_brief.call_count == 1
        assert brief1.language == brief2.language


# ============================================================================
# Test: Brief Mapper
# ============================================================================

class TestBriefMapper:
    """Test LLM brief → domain brief mapping"""

    def test_map_llm_to_domain_basic(self):
        """Should map LLM brief to domain brief"""
        llm_brief = CreativeBrief(
            language="ko",
            mood=["fresh", "clean"],
            season=["summer"],
            notes_preference={"citrus": 0.5, "floral": 0.5},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP"
        )

        domain_brief = map_llm_brief_to_domain(
            llm_brief,
            user_id="test_user",
            original_text=KOREAN_INPUT_MEDIUM
        )

        # Verify mapping
        assert domain_brief.user_id == "test_user"
        assert len(domain_brief.mood_keywords) > 0
        assert 0.0 <= domain_brief.freshness <= 1.0
        assert 0.0 <= domain_brief.warmth <= 1.0

    def test_extract_moga_constraints(self):
        """Should extract MOGA constraints from LLM brief"""
        llm_brief = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={"citrus": 0.7, "woody": 0.3},
            forbidden_ingredients=["patchouli", "oakmoss"],
            budget_tier="high",
            target_profile="luxury",
            product_category="EDP",
            creative_hints=["elegant", "sophisticated", "timeless"]
        )

        domain_brief = map_llm_brief_to_domain(llm_brief, user_id="test_user")
        constraints = extract_moga_constraints(llm_brief, domain_brief)

        # Verify constraints
        assert 'notes_preference' in constraints
        assert 'forbidden_ingredients' in constraints
        assert len(constraints['forbidden_ingredients']) == 2
        assert constraints['max_cost'] > 100  # high budget

        # Verify novelty_weight boost from creative hints
        assert 'novelty_weight' in constraints
        expected_novelty = 0.2 + 0.05 * len(llm_brief.creative_hints)
        assert constraints['novelty_weight'] == pytest.approx(expected_novelty, rel=0.01)
        assert 'creative_hints' in constraints

    def test_novelty_weight_calculation(self):
        """Should calculate novelty_weight correctly"""
        # 0 hints
        llm_brief_0 = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP",
            creative_hints=[]
        )
        domain_brief_0 = map_llm_brief_to_domain(llm_brief_0, user_id="test")
        constraints_0 = extract_moga_constraints(llm_brief_0, domain_brief_0)
        assert 'novelty_weight' not in constraints_0  # No hints, no novelty boost

        # 5 hints
        llm_brief_5 = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP",
            creative_hints=["hint1", "hint2", "hint3", "hint4", "hint5"]
        )
        domain_brief_5 = map_llm_brief_to_domain(llm_brief_5, user_id="test")
        constraints_5 = extract_moga_constraints(llm_brief_5, domain_brief_5)
        expected = 0.2 + 0.05 * 5  # 0.45
        assert constraints_5['novelty_weight'] == pytest.approx(expected, rel=0.01)

        # 8 hints (max)
        llm_brief_8 = CreativeBrief(
            language="ko",
            mood=["fresh"],
            season=["summer"],
            notes_preference={},
            budget_tier="mid",
            target_profile="daily_fresh",
            product_category="EDP",
            creative_hints=["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8"]
        )
        domain_brief_8 = map_llm_brief_to_domain(llm_brief_8, user_id="test")
        constraints_8 = extract_moga_constraints(llm_brief_8, domain_brief_8)
        expected = 0.2 + 0.05 * 8  # 0.6
        assert constraints_8['novelty_weight'] == pytest.approx(expected, rel=0.01)


# ============================================================================
# Test: End-to-End (Smoke Test with Mocks)
# ============================================================================

class TestEndToEnd:
    """End-to-end smoke tests"""

    @patch('fragrance_ai.llm.qwen_client.get_qwen_client')
    @patch('fragrance_ai.llm.llama_hints.get_llama_generator')
    def test_full_pipeline_mocked(self, mock_llama, mock_qwen):
        """Full pipeline: User input → LLM brief → Domain brief → MOGA constraints"""
        # Mock Qwen
        mock_qwen_client = Mock()
        mock_qwen_client.infer_brief.return_value = CreativeBrief(
            language="ko",
            mood=["romantic", "warm"],
            season=["autumn"],
            notes_preference={"floral": 0.4, "woody": 0.4, "vanilla": 0.2},
            forbidden_ingredients=["musk"],
            budget_tier="high",
            target_profile="evening",
            product_category="EDP",
            creative_hints=["sunset", "romance", "mystery"]
        )
        mock_qwen.return_value = mock_qwen_client

        # Mock Llama
        mock_llama_gen = Mock()
        mock_llama_gen.generate_hints.return_value = ["velvet", "candlelight"]
        mock_llama.return_value = mock_llama_gen

        # Step 1: LLM brief generation (creative mode)
        user_input = "저녁에 사용할 로맨틱한 향수를 만들고 싶어요. 플로랄과 우디 계열이 좋습니다."
        llm_brief = build_brief(user_input, mode="creative", use_cache=False)

        # Step 2: Map to domain brief
        domain_brief = map_llm_brief_to_domain(llm_brief, user_id="test_user")

        # Step 3: Extract MOGA constraints
        constraints = extract_moga_constraints(llm_brief, domain_brief)

        # Assertions
        assert llm_brief is not None
        assert domain_brief is not None
        assert constraints is not None

        # Verify LLM brief
        assert "romantic" in llm_brief.mood
        assert llm_brief.target_profile == "evening"
        assert len(llm_brief.creative_hints) >= 3  # Original 3 + Llama 2 = 5

        # Verify domain brief
        assert domain_brief.user_id == "test_user"
        assert domain_brief.warmth > 0.5  # Evening profile

        # Verify MOGA constraints
        assert 'notes_preference' in constraints
        assert 'forbidden_ingredients' in constraints
        assert 'musk' in constraints['forbidden_ingredients']
        assert 'novelty_weight' in constraints
        assert constraints['novelty_weight'] > 0.2  # Boosted by creative hints
        assert 'fragrance_family' in constraints
        assert constraints['mood'] == 'romantic'


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
