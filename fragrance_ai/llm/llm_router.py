# fragrance_ai/llm/llm_router.py
"""
LLM Router
Determines which LLM mode to use based on user input characteristics
"""

import re
from typing import Literal
import logging

logger = logging.getLogger(__name__)


# Keywords for mode detection
FAST_KEYWORDS = [
    "%", "비율", "정량", "정확", "수치", "농도", "퍼센트",
    "exact", "precise", "ratio", "percentage", "concentration"
]

CREATIVE_KEYWORDS = [
    "서사", "스토리", "시적", "몽환", "이미지", "상상", "감정", "느낌",
    "분위기", "무드", "예술적", "영감", "추상적", "은유", "시나리오",
    "narrative", "story", "poetic", "dreamy", "imagery", "imagination",
    "emotion", "feeling", "atmosphere", "mood", "artistic", "inspiration"
]


def route_mode(user_text: str) -> Literal["fast", "balanced", "creative"]:
    """
    Determine LLM mode based on user input

    Args:
        user_text: User input text

    Returns:
        Mode: "fast", "balanced", or "creative"

    Rules:
        - fast: Short + has numeric/percentage keywords
        - creative: Long + has creative/emotion keywords
        - balanced: Default for everything else
    """
    text_lower = user_text.lower()
    text_len = len(user_text)

    # Count keyword matches
    fast_count = sum(1 for kw in FAST_KEYWORDS if kw in text_lower)
    creative_count = sum(1 for kw in CREATIVE_KEYWORDS if kw in text_lower)

    # Rule 1: Fast mode - short + numeric
    if text_len < 100 and fast_count >= 2:
        logger.info(f"Routing to FAST mode: len={text_len}, fast_kw={fast_count}")
        return "fast"

    # Rule 2: Creative mode - long + many creative keywords
    if text_len > 200 or creative_count >= 3:
        logger.info(f"Routing to CREATIVE mode: len={text_len}, creative_kw={creative_count}")
        return "creative"

    # Rule 3: Creative mode - moderate length + some creative keywords
    if text_len > 100 and creative_count >= 2:
        logger.info(f"Routing to CREATIVE mode: len={text_len}, creative_kw={creative_count}")
        return "creative"

    # Default: Balanced mode
    logger.info(f"Routing to BALANCED mode: len={text_len}, fast_kw={fast_count}, creative_kw={creative_count}")
    return "balanced"


def detect_language(user_text: str) -> Literal["ko", "en"]:
    """
    Detect language from user text

    Args:
        user_text: User input text

    Returns:
        Language code: "ko" or "en"
    """
    # Check for Korean characters
    korean_pattern = re.compile(r'[가-힣]')
    if korean_pattern.search(user_text):
        return "ko"

    return "en"


def extract_numeric_hints(user_text: str) -> dict:
    """
    Extract numeric constraints from user text

    Args:
        user_text: User input text

    Returns:
        Dict of extracted numeric constraints

    Examples:
        "시트러스 30%" -> {"citrus_pct": 30}
        "알레르겐 500ppm 이하" -> {"max_allergens_ppm": 500}
    """
    constraints = {}

    # Pattern: number + % (percentage)
    pct_matches = re.findall(r'(\d+)\s*%', user_text)
    if pct_matches:
        constraints["percentage_hint"] = [int(m) for m in pct_matches]

    # Pattern: number + ppm
    ppm_matches = re.findall(r'(\d+)\s*ppm', user_text.lower())
    if ppm_matches:
        constraints["max_allergens_ppm"] = int(ppm_matches[0])

    # Pattern: longevity hours (지속 시간)
    hour_matches = re.findall(r'(\d+)\s*시간|(\d+)\s*hours?', user_text.lower())
    if hour_matches:
        hours = [int(h) for group in hour_matches for h in group if h]
        if hours:
            constraints["min_longevity_hours"] = hours[0]

    return constraints


__all__ = [
    "route_mode",
    "detect_language",
    "extract_numeric_hints",
    "FAST_KEYWORDS",
    "CREATIVE_KEYWORDS"
]
