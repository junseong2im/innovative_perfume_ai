# fragrance_ai/llm/brief_mapper.py
"""
LLM CreativeBrief Mapper
Converts LLM's CreativeBrief to domain_models.CreativeBrief for MOGA/RLHF pipeline
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from fragrance_ai.llm.schemas import CreativeBrief as LLMCreativeBrief
from fragrance_ai.schemas.domain_models import (
    CreativeBrief as DomainCreativeBrief,
    ProductCategory,
    ConcentrationType
)

logger = logging.getLogger(__name__)


def map_llm_brief_to_domain(
    llm_brief: LLMCreativeBrief,
    user_id: str,
    original_text: str = ""
) -> DomainCreativeBrief:
    """
    Map LLM's CreativeBrief to domain_models.CreativeBrief

    Args:
        llm_brief: CreativeBrief from LLM ensemble
        user_id: User identifier
        original_text: Original user input text

    Returns:
        Domain CreativeBrief for MOGA/RLHF pipeline
    """

    # Product category mapping
    product_category_map = {
        "EDP": ProductCategory.EAU_DE_PARFUM,
        "EDT": ProductCategory.EAU_DE_TOILETTE,
        "PARFUM": ProductCategory.EAU_DE_PARFUM,  # Fallback to EDP
    }

    concentration_type_map = {
        "EDP": ConcentrationType.EAU_DE_PARFUM,
        "EDT": ConcentrationType.EAU_DE_TOILETTE,
        "PARFUM": ConcentrationType.PARFUM,
    }

    # Target profile to numeric scales mapping
    profile_characteristics = {
        "daily_fresh": {
            "desired_intensity": 0.5,
            "masculinity": 0.5,
            "complexity": 0.4,
            "longevity": 0.5,
            "sillage": 0.5,
            "warmth": 0.3,
            "sweetness": 0.4,
            "freshness": 0.9
        },
        "evening": {
            "desired_intensity": 0.8,
            "masculinity": 0.5,
            "complexity": 0.7,
            "longevity": 0.8,
            "sillage": 0.8,
            "warmth": 0.7,
            "sweetness": 0.6,
            "freshness": 0.3
        },
        "luxury": {
            "desired_intensity": 0.9,
            "masculinity": 0.5,
            "complexity": 0.9,
            "longevity": 0.9,
            "sillage": 0.7,
            "warmth": 0.6,
            "sweetness": 0.5,
            "freshness": 0.4
        },
        "sport": {
            "desired_intensity": 0.6,
            "masculinity": 0.7,
            "complexity": 0.3,
            "longevity": 0.6,
            "sillage": 0.6,
            "warmth": 0.2,
            "sweetness": 0.2,
            "freshness": 1.0
        },
        "signature": {
            "desired_intensity": 0.8,
            "masculinity": 0.5,
            "complexity": 0.8,
            "longevity": 0.9,
            "sillage": 0.7,
            "warmth": 0.5,
            "sweetness": 0.5,
            "freshness": 0.5
        }
    }

    # Get base characteristics from target profile
    base_chars = profile_characteristics.get(
        llm_brief.target_profile,
        profile_characteristics["daily_fresh"]
    )

    # Adjust based on season
    season_adjustments = _get_season_adjustments(llm_brief.season)

    # Adjust based on mood
    mood_adjustments = _get_mood_adjustments(llm_brief.mood)

    # Combine adjustments
    characteristics = base_chars.copy()
    for key in characteristics:
        characteristics[key] = min(1.0, max(0.0,
            characteristics[key] +
            season_adjustments.get(key, 0.0) +
            mood_adjustments.get(key, 0.0)
        ))

    # Budget to max cost mapping
    budget_to_cost = {
        "low": 50.0,
        "mid": 150.0,
        "high": 500.0
    }

    # Generate theme from mood and season
    theme = _generate_theme(llm_brief.mood, llm_brief.season, llm_brief.creative_hints)

    # Generate story from creative hints
    story = _generate_story(llm_brief.creative_hints, original_text)

    # Map product category
    target_category = product_category_map.get(
        llm_brief.product_category,
        ProductCategory.EAU_DE_PARFUM
    )

    concentration_type = concentration_type_map.get(
        llm_brief.product_category,
        ConcentrationType.EAU_DE_PARFUM
    )

    # Create domain brief
    domain_brief = DomainCreativeBrief(
        user_id=user_id,
        theme=theme,
        story=story,
        mood_keywords=llm_brief.mood[:10],  # Max 10
        target_category=target_category,
        concentration_type=concentration_type,
        desired_intensity=characteristics["desired_intensity"],
        masculinity=characteristics["masculinity"],
        complexity=characteristics["complexity"],
        longevity=characteristics["longevity"],
        sillage=characteristics["sillage"],
        warmth=characteristics["warmth"],
        sweetness=characteristics["sweetness"],
        freshness=characteristics["freshness"],
        max_cost_per_kg=budget_to_cost.get(llm_brief.budget_tier, 150.0),
        excluded_ingredients=llm_brief.forbidden_ingredients,
        required_ingredients=[]  # Could be extracted from notes_preference
    )

    logger.info(f"Mapped LLM brief to domain brief: theme='{theme}', profile={llm_brief.target_profile}")

    return domain_brief


def _get_season_adjustments(seasons: List[str]) -> Dict[str, float]:
    """Get characteristic adjustments based on seasons"""
    adjustments = {
        "desired_intensity": 0.0,
        "warmth": 0.0,
        "freshness": 0.0,
        "sweetness": 0.0,
    }

    if not seasons:
        return adjustments

    # Season influence
    if "spring" in seasons:
        adjustments["freshness"] += 0.1
        adjustments["warmth"] -= 0.1
    if "summer" in seasons:
        adjustments["freshness"] += 0.2
        adjustments["warmth"] -= 0.2
        adjustments["desired_intensity"] -= 0.1
    if "autumn" in seasons:
        adjustments["warmth"] += 0.1
        adjustments["sweetness"] += 0.1
    if "winter" in seasons:
        adjustments["warmth"] += 0.2
        adjustments["desired_intensity"] += 0.1

    return adjustments


def _get_mood_adjustments(moods: List[str]) -> Dict[str, float]:
    """Get characteristic adjustments based on moods"""
    adjustments = {
        "masculinity": 0.0,
        "complexity": 0.0,
        "sweetness": 0.0,
        "warmth": 0.0,
        "freshness": 0.0,
    }

    if not moods:
        return adjustments

    # Mood influence
    mood_str = " ".join(moods).lower()

    if any(m in mood_str for m in ["romantic", "sensual", "intimate"]):
        adjustments["sweetness"] += 0.2
        adjustments["warmth"] += 0.1
        adjustments["complexity"] += 0.1

    if any(m in mood_str for m in ["fresh", "clean", "light"]):
        adjustments["freshness"] += 0.2
        adjustments["warmth"] -= 0.1

    if any(m in mood_str for m in ["masculine", "strong", "bold"]):
        adjustments["masculinity"] += 0.2

    if any(m in mood_str for m in ["feminine", "delicate", "soft"]):
        adjustments["masculinity"] -= 0.2
        adjustments["sweetness"] += 0.1

    if any(m in mood_str for m in ["complex", "sophisticated", "mysterious"]):
        adjustments["complexity"] += 0.2

    return adjustments


def _generate_theme(moods: List[str], seasons: List[str], hints: List[str]) -> str:
    """Generate theme string from moods, seasons, and hints"""
    parts = []

    # Add primary mood
    if moods:
        parts.append(moods[0].title())

    # Add season context
    if seasons:
        parts.append(seasons[0].title())

    # Add creative hint
    if hints:
        parts.append(hints[0])

    # Fallback
    if not parts:
        parts = ["Fresh", "Daily"]

    return " ".join(parts[:3])  # Max 3 parts


def _generate_story(hints: List[str], original_text: str = "") -> str:
    """Generate story from creative hints and original text"""
    if original_text and len(original_text) > 20:
        # Use original text as story if substantial
        return original_text[:1000]

    if hints:
        # Combine hints into story
        return f"A fragrance inspired by: {', '.join(hints[:5])}"

    return "A uniquely crafted fragrance"


def extract_moga_constraints(
    llm_brief: LLMCreativeBrief,
    domain_brief: DomainCreativeBrief
) -> Dict[str, Any]:
    """
    Extract MOGA-specific constraints from both briefs

    Args:
        llm_brief: LLM CreativeBrief with notes_preference
        domain_brief: Domain CreativeBrief

    Returns:
        Constraints dict for MOGA optimizer
    """
    constraints = {
        'max_ingredients': 15,
        'min_ingredients': 5,
        'max_cost': domain_brief.max_cost_per_kg or 150.0,
        'min_quality': 80,
    }

    # Add notes preference as weight constraints
    if llm_brief.notes_preference:
        constraints['notes_preference'] = llm_brief.notes_preference

    # Add excluded ingredients
    if domain_brief.excluded_ingredients:
        constraints['forbidden_ingredients'] = domain_brief.excluded_ingredients

    # Add required ingredients
    if domain_brief.required_ingredients:
        constraints['required_ingredients'] = domain_brief.required_ingredients

    # Add custom constraints from LLM
    if llm_brief.constraints:
        for key, value in llm_brief.constraints.items():
            constraints[key] = value

    # Map domain characteristics to MOGA parameters
    if domain_brief.freshness > 0.7:
        constraints['fragrance_family'] = 'citrus'
    elif domain_brief.warmth > 0.7:
        constraints['fragrance_family'] = 'woody'
    elif domain_brief.sweetness > 0.7:
        constraints['fragrance_family'] = 'floral'
    else:
        constraints['fragrance_family'] = 'fresh'

    # Map mood to MOGA mood
    mood_keywords = " ".join(domain_brief.mood_keywords).lower()
    if any(m in mood_keywords for m in ['romantic', 'sensual']):
        constraints['mood'] = 'romantic'
    elif any(m in mood_keywords for m in ['energetic', 'vibrant', 'sport']):
        constraints['mood'] = 'energetic'
    else:
        constraints['mood'] = 'balanced'

    # Creative hints influence: Increase novelty weight
    if llm_brief.creative_hints:
        # Base novelty weight + k * hint_count
        k = 0.05  # Novelty boost per hint
        novelty_boost = k * len(llm_brief.creative_hints)
        constraints['novelty_weight'] = 0.2 + novelty_boost  # Base 0.2
        constraints['creative_hints'] = llm_brief.creative_hints
        logger.info(f"MOGA novelty weight boosted by {novelty_boost:.3f} ({len(llm_brief.creative_hints)} hints)")

    return constraints


__all__ = [
    "map_llm_brief_to_domain",
    "extract_moga_constraints"
]
