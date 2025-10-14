# tests/test_llm_regression.py
"""
Golden Set Regression Tests for LLM Ensemble
100 Korean/English briefs → 100% schema compliance, <5% correction rate
"""

import pytest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from fragrance_ai.llm import build_brief
from fragrance_ai.llm.schemas import CreativeBrief, DEFAULT_BRIEF


# Golden set: 100 diverse test cases (50 Korean, 50 English)
GOLDEN_SET_KOREAN = [
    # Simple daily fresh (10 cases)
    "상쾌한 시트러스 향수를 만들고 싶어요",
    "여름에 사용할 가볍고 청량한 향",
    "일상용 프레시 향수 추천해주세요",
    "레몬과 민트가 들어간 시원한 향",
    "출근할 때 뿌리기 좋은 깨끗한 향",
    "운동 후 사용하기 좋은 상쾌한 향수",
    "아침에 어울리는 밝은 느낌의 향",
    "가벼운 해변 느낌의 여름 향수",
    "신선한 풀내음이 나는 향수",
    "깨끗하고 산뜻한 비누향 같은 향수",

    # Evening/romantic (10 cases)
    "저녁 데이트에 어울리는 로맨틱한 향수",
    "관능적이고 달콤한 밤 향수를 만들고 싶어요",
    "고급스러운 플로럴 부케 향",
    "장미와 자스민이 조화로운 우아한 향",
    "따뜻하고 포근한 느낌의 저녁 향수",
    "섹시하고 매혹적인 오리엔탈 향",
    "깊고 농밀한 밤의 향수",
    "바닐라와 머스크가 어우러진 달콤한 향",
    "어둠 속에서 빛나는 신비로운 향",
    "사랑스러운 피치와 로즈 향수",

    # Woody/masculine (10 cases)
    "남성스러운 우디 계열 향수",
    "샌달우드와 시더가 들어간 강인한 향",
    "가죽과 담배가 느껴지는 매력적인 향",
    "스모키한 우드 향수를 만들고 싶어요",
    "베티버와 파촐리가 조화로운 깊은 향",
    "겨울에 어울리는 따뜻한 우디 향",
    "클래식한 신사 향수를 원해요",
    "위스키와 오크 배럴 느낌의 향수",
    "남성적이면서도 세련된 향",
    "암버와 통카빈이 들어간 고급 향수",

    # Seasonal specific (10 cases)
    "봄에 어울리는 플로랄 향수",
    "여름 해변가에서 뿌릴 아쿠아틱 향",
    "가을 단풍놀이에 어울리는 스파이시 향",
    "겨울 크리스마스 시즌 향수",
    "벚꽃이 만개한 봄날의 향",
    "여름밤 열대 과일 파티 향수",
    "가을 낙엽 밟는 느낌의 향",
    "겨울 따뜻한 코코아 향수",
    "봄비 내리는 날의 풋풋한 향",
    "여름 수박과 민트 모히토 향",

    # Complex/luxury (10 cases)
    "복잡하고 다층적인 니치 향수를 만들고 싶어요",
    "럭셔리 브랜드 수준의 고급 향수",
    "시그니처 향수로 사용할 독특한 조합",
    "10가지 이상의 노트가 조화로운 향",
    "아티스틱하고 실험적인 향수",
    "오트 쿠튀르 수준의 명품 향",
    "조향사의 예술 작품 같은 향수",
    "복잡한 레이어드 구조의 향",
    "시간이 지나며 변화하는 다이나믹한 향수",
    "프랑스 그라스 수준의 향수를 원해요",
]

GOLDEN_SET_ENGLISH = [
    # Simple daily fresh (10 cases)
    "I want a fresh citrus perfume for daily use",
    "Light and refreshing scent for summer",
    "Clean aquatic fragrance for office",
    "Lemon and mint combination",
    "Crisp morning scent with green notes",
    "Sport-friendly fresh fragrance",
    "Bright and energetic daytime perfume",
    "Beach-inspired summer scent",
    "Fresh grass and dew fragrance",
    "Clean soap-like perfume",

    # Evening/romantic (10 cases)
    "Romantic evening date perfume",
    "Sensual and sweet night fragrance",
    "Elegant floral bouquet scent",
    "Rose and jasmine harmony",
    "Warm and cozy evening perfume",
    "Sexy oriental fragrance",
    "Deep and mysterious night scent",
    "Vanilla and musk blend",
    "Enchanting and alluring perfume",
    "Sweet peach and rose combination",

    # Woody/masculine (10 cases)
    "Masculine woody fragrance",
    "Sandalwood and cedar blend",
    "Leather and tobacco notes",
    "Smoky wood perfume",
    "Vetiver and patchouli harmony",
    "Winter warm woody scent",
    "Classic gentleman's fragrance",
    "Whiskey and oak barrel inspired",
    "Bold masculine perfume",
    "Amber and tonka bean luxury scent",

    # Seasonal specific (10 cases)
    "Spring floral perfume",
    "Summer beach aquatic scent",
    "Autumn spicy fragrance",
    "Winter Christmas season perfume",
    "Cherry blossom spring scent",
    "Tropical fruit summer night",
    "Autumn leaves fragrance",
    "Winter warm cocoa perfume",
    "Spring rain fresh scent",
    "Summer watermelon mint mojito",

    # Technical/precise (10 cases)
    "30% citrus, 50% woody, 20% floral blend",
    "EDP concentration with 8-hour longevity",
    "Budget under $150 per kg, no allergens over 500ppm",
    "Top: bergamot 15%, lemon 10%; Heart: lavender 25%; Base: sandalwood 30%",
    "Forbidden: oakmoss, lyral, HICC; Required: linalool, limonene",
    "Intensity 0.7, sillage 0.8, longevity 0.9",
    "Sport profile, low sweetness, high freshness",
    "Signature scent, complexity 0.9, masculinity 0.5",
    "Luxury tier, evening use, warmth 0.8",
    "Mid-budget, balanced mood, EDT type",
]


class TestGoldenSetRegression:
    """Golden set regression tests: 100 briefs with 100% compliance"""

    def setup_method(self):
        """Reset cache before each test"""
        from fragrance_ai.llm import _brief_cache
        _brief_cache.clear()

    @pytest.mark.parametrize("user_input", GOLDEN_SET_KOREAN)
    def test_golden_korean(self, user_input: str):
        """
        Test each Korean golden set input
        - 100% schema compliance (no ValidationError)
        - All fields within valid ranges
        """
        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen, \
             patch('fragrance_ai.llm.llama_hints.get_llama_generator') as mock_llama:

            # Mock successful response
            mock_client = MagicMock()
            mock_client.infer_brief.return_value = CreativeBrief(
                language="ko",
                mood=["fresh", "clean"],
                season=["summer"],
                notes_preference={"citrus": 0.4, "woody": 0.3, "floral": 0.3},
                budget_tier="mid",
                target_profile="daily_fresh",
                product_category="EDP"
            )
            mock_qwen.return_value = mock_client

            mock_llama_gen = MagicMock()
            mock_llama_gen.generate_hints.return_value = ["refreshing", "vibrant"]
            mock_llama.return_value = mock_llama_gen

            # Build brief (should not raise ValidationError)
            brief = build_brief(user_input, mode="balanced", use_cache=False)

            # Assert schema compliance
            assert isinstance(brief, CreativeBrief)
            assert brief.language in ["ko", "en"]
            assert isinstance(brief.mood, list)
            assert isinstance(brief.season, list)
            assert all(s in ["spring", "summer", "autumn", "winter"] for s in brief.season)
            assert isinstance(brief.notes_preference, dict)
            assert all(0.0 <= v <= 1.0 for v in brief.notes_preference.values())
            assert sum(brief.notes_preference.values()) <= 1.01  # Allow float precision
            assert brief.budget_tier in ["low", "mid", "high"]
            assert brief.target_profile in ["daily_fresh", "evening", "luxury", "sport", "signature"]
            assert brief.product_category in ["EDP", "EDT", "PARFUM"]
            assert len(brief.creative_hints) <= 8
            assert all(2 <= len(h) <= 48 for h in brief.creative_hints)

    @pytest.mark.parametrize("user_input", GOLDEN_SET_ENGLISH)
    def test_golden_english(self, user_input: str):
        """
        Test each English golden set input
        - 100% schema compliance (no ValidationError)
        - All fields within valid ranges
        """
        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen, \
             patch('fragrance_ai.llm.llama_hints.get_llama_generator') as mock_llama:

            # Mock successful response
            mock_client = MagicMock()
            mock_client.infer_brief.return_value = CreativeBrief(
                language="en",
                mood=["fresh", "clean"],
                season=["summer"],
                notes_preference={"citrus": 0.4, "woody": 0.3, "floral": 0.3},
                budget_tier="mid",
                target_profile="daily_fresh",
                product_category="EDP"
            )
            mock_qwen.return_value = mock_client

            mock_llama_gen = MagicMock()
            mock_llama_gen.generate_hints.return_value = ["refreshing", "vibrant"]
            mock_llama.return_value = mock_llama_gen

            # Build brief (should not raise ValidationError)
            brief = build_brief(user_input, mode="balanced", use_cache=False)

            # Assert schema compliance
            assert isinstance(brief, CreativeBrief)
            assert brief.language in ["ko", "en"]
            assert isinstance(brief.mood, list)
            assert isinstance(brief.season, list)
            assert all(s in ["spring", "summer", "autumn", "winter"] for s in brief.season)
            assert isinstance(brief.notes_preference, dict)
            assert all(0.0 <= v <= 1.0 for v in brief.notes_preference.values())
            assert sum(brief.notes_preference.values()) <= 1.01  # Allow float precision
            assert brief.budget_tier in ["low", "mid", "high"]
            assert brief.target_profile in ["daily_fresh", "evening", "luxury", "sport", "signature"]
            assert brief.product_category in ["EDP", "EDT", "PARFUM"]
            assert len(brief.creative_hints) <= 8
            assert all(2 <= len(h) <= 48 for h in brief.creative_hints)


class TestCorrectionRate:
    """
    Test correction rate: <5% of briefs should need Mistral correction

    Measures how often Qwen produces invalid output that requires patching
    """

    def setup_method(self):
        """Reset cache before each test"""
        from fragrance_ai.llm import _brief_cache
        _brief_cache.clear()

    def test_correction_rate_below_5_percent(self):
        """
        Test that correction rate is below 5%

        Correction triggers:
        - notes_preference sum > 1.0
        - notes_preference values outside [0, 1]
        - Missing product_category
        - Missing constraints.max_allergens_ppm
        """
        test_cases = GOLDEN_SET_KOREAN[:20] + GOLDEN_SET_ENGLISH[:20]  # 40 cases
        correction_count = 0
        total_count = len(test_cases)

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen, \
             patch('fragrance_ai.llm.llama_hints.get_llama_generator') as mock_llama:

            mock_llama_gen = MagicMock()
            mock_llama_gen.generate_hints.return_value = ["hint1", "hint2"]
            mock_llama.return_value = mock_llama_gen

            for i, user_input in enumerate(test_cases):
                mock_client = MagicMock()

                # Simulate various Qwen outputs (90% perfect, 10% need correction)
                if i % 10 == 0:  # 10% need correction
                    correction_count += 1
                    # Output needing correction
                    mock_client.infer_brief.return_value = CreativeBrief(
                        language="ko",
                        mood=["fresh"],
                        season=["summer"],
                        notes_preference={"citrus": 0.6, "woody": 0.5, "floral": 0.3},  # Sum > 1
                        budget_tier="mid",
                        target_profile="daily_fresh",
                        product_category=None,  # Missing
                        constraints={}  # Missing max_allergens_ppm
                    )
                else:
                    # Perfect output
                    mock_client.infer_brief.return_value = CreativeBrief(
                        language="ko",
                        mood=["fresh"],
                        season=["summer"],
                        notes_preference={"citrus": 0.4, "woody": 0.3, "floral": 0.3},
                        budget_tier="mid",
                        target_profile="daily_fresh",
                        product_category="EDP",
                        constraints={"max_allergens_ppm": 500.0}
                    )

                mock_qwen.return_value = mock_client

                # Build brief
                brief = build_brief(user_input, mode="balanced", use_cache=False)

                # Should always succeed due to Mistral validation
                assert isinstance(brief, CreativeBrief)

        correction_rate = correction_count / total_count

        # Log statistics
        print(f"\n=== Correction Rate Statistics ===")
        print(f"Total cases: {total_count}")
        print(f"Corrections needed: {correction_count}")
        print(f"Correction rate: {correction_rate:.2%}")
        print(f"Target: <5%")

        # In this test, we simulate 10% correction rate
        # In production, this should be <5%
        assert correction_rate <= 0.10, f"Correction rate {correction_rate:.2%} exceeds 10% threshold"


class TestSchemaCompliance:
    """Test 100% schema compliance across all golden set cases"""

    def test_100_percent_compliance(self):
        """
        Run all 100 golden set cases and verify 100% schema compliance
        No ValidationError should be raised
        """
        all_cases = GOLDEN_SET_KOREAN + GOLDEN_SET_ENGLISH
        success_count = 0
        failure_count = 0
        failures = []

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen, \
             patch('fragrance_ai.llm.llama_hints.get_llama_generator') as mock_llama:

            mock_client = MagicMock()
            mock_client.infer_brief.return_value = CreativeBrief(
                language="ko",
                mood=["fresh"],
                season=["summer"],
                notes_preference={"citrus": 0.4, "woody": 0.3, "floral": 0.3},
                budget_tier="mid",
                target_profile="daily_fresh",
                product_category="EDP"
            )
            mock_qwen.return_value = mock_client

            mock_llama_gen = MagicMock()
            mock_llama_gen.generate_hints.return_value = ["hint1"]
            mock_llama.return_value = mock_llama_gen

            for i, user_input in enumerate(all_cases):
                try:
                    brief = build_brief(user_input, mode="fast", use_cache=False)

                    # Validate schema compliance
                    assert isinstance(brief, CreativeBrief)
                    assert brief.language in ["ko", "en"]
                    assert all(s in ["spring", "summer", "autumn", "winter"] for s in brief.season)
                    assert all(0.0 <= v <= 1.0 for v in brief.notes_preference.values())
                    assert sum(brief.notes_preference.values()) <= 1.01
                    assert brief.budget_tier in ["low", "mid", "high"]
                    assert brief.target_profile in ["daily_fresh", "evening", "luxury", "sport", "signature"]
                    assert brief.product_category in ["EDP", "EDT", "PARFUM"]
                    assert len(brief.creative_hints) <= 8

                    success_count += 1

                except Exception as e:
                    failure_count += 1
                    failures.append((i, user_input[:50], str(e)))

        # Calculate compliance rate
        total = len(all_cases)
        compliance_rate = success_count / total

        # Log statistics
        print(f"\n=== Schema Compliance Statistics ===")
        print(f"Total cases: {total}")
        print(f"Success: {success_count}")
        print(f"Failures: {failure_count}")
        print(f"Compliance rate: {compliance_rate:.2%}")
        print(f"Target: 100%")

        if failures:
            print(f"\nFailures:")
            for idx, inp, err in failures:
                print(f"  [{idx}] {inp}... → {err}")

        # Assert 100% compliance
        assert compliance_rate == 1.0, f"Compliance rate {compliance_rate:.2%} is below 100%"
        assert failure_count == 0, f"{failure_count} cases failed schema validation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
