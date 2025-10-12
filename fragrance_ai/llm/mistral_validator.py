# fragrance_ai/llm/mistral_validator.py
"""
Mistral Validator
Validates and patches CreativeBrief with schema corrections
"""

import logging
from typing import Dict, List
from .schemas import CreativeBrief

logger = logging.getLogger(__name__)


def validate_and_patch(brief: CreativeBrief) -> CreativeBrief:
    """
    Validate and patch CreativeBrief

    Args:
        brief: Input CreativeBrief from Qwen

    Returns:
        Validated and patched CreativeBrief

    Validation rules:
        1. notes_preference: Clip to [0,1], normalize if sum > 1
        2. product_category: Default to "EDP" if missing
        3. constraints.max_allergens_ppm: Default to 500 if missing
        4. Fix typos/case in season and target_profile
        5. Fill missing fields with defaults
    """
    logger.info(f"Validating CreativeBrief from Qwen")

    # Create mutable copy
    data = brief.model_dump()

    # Rule 1: Notes preference validation
    data = _validate_notes_preference(data)

    # Rule 2: Product category default
    if not data.get("product_category"):
        data["product_category"] = "EDP"
        logger.info(f"Set default product_category: EDP")

    # Rule 3: Constraints defaults
    data = _validate_constraints(data)

    # Rule 4: Fix typos and case
    data = _fix_typos_and_case(data)

    # Rule 5: Fill missing fields
    data = _fill_missing_fields(data)

    # Create validated brief
    try:
        validated_brief = CreativeBrief(**data)
        logger.info(f"Validation successful")
        return validated_brief
    except Exception as e:
        logger.error(f"Validation failed: {e}, using original brief")
        return brief


def _validate_notes_preference(data: Dict) -> Dict:
    """Validate and normalize notes_preference"""
    notes = data.get("notes_preference", {})

    if not notes:
        return data

    # Clip values to [0, 1]
    clipped = {k: max(0.0, min(1.0, v)) for k, v in notes.items()}

    # Check sum
    total = sum(clipped.values())
    if total > 1.0:
        # Normalize
        clipped = {k: v / total for k, v in clipped.items()}
        logger.info(f"Normalized notes_preference (sum was {total:.3f})")

    data["notes_preference"] = clipped
    return data


def _validate_constraints(data: Dict) -> Dict:
    """Validate and set defaults for constraints"""
    constraints = data.get("constraints", {})

    # Default: max_allergens_ppm = 500
    if "max_allergens_ppm" not in constraints:
        constraints["max_allergens_ppm"] = 500.0
        logger.info(f"Set default max_allergens_ppm: 500")

    # Ensure all values are float
    constraints = {k: float(v) for k, v in constraints.items()}

    data["constraints"] = constraints
    return data


def _fix_typos_and_case(data: Dict) -> Dict:
    """Fix common typos and case issues"""

    # Season typos
    season_map = {
        "spring": "spring",
        "summer": "summer",
        "autumn": "autumn",
        "fall": "autumn",  # Alias
        "winter": "winter"
    }

    if "season" in data and data["season"]:
        fixed_seasons = []
        for s in data["season"]:
            s_lower = s.lower().strip()
            if s_lower in season_map:
                fixed_seasons.append(season_map[s_lower])
            else:
                logger.warning(f"Unknown season: {s}, skipping")

        data["season"] = list(set(fixed_seasons))  # Remove duplicates

    # Product category typos
    category_map = {
        "edp": "EDP",
        "eau de parfum": "EDP",
        "edt": "EDT",
        "eau de toilette": "EDT",
        "parfum": "PARFUM",
        "perfume": "PARFUM"
    }

    if "product_category" in data and data["product_category"]:
        cat = data["product_category"]
        cat_lower = cat.lower().strip()
        if cat_lower in category_map:
            data["product_category"] = category_map[cat_lower]
        elif cat.upper() in ["EDP", "EDT", "PARFUM"]:
            data["product_category"] = cat.upper()

    # Target profile typos
    profile_map = {
        "daily": "daily_fresh",
        "daily_fresh": "daily_fresh",
        "evening": "evening",
        "luxury": "luxury",
        "sport": "sport",
        "signature": "signature"
    }

    if "target_profile" in data and data["target_profile"]:
        prof = data["target_profile"]
        prof_lower = prof.lower().strip()
        if prof_lower in profile_map:
            data["target_profile"] = profile_map[prof_lower]

    return data


def _fill_missing_fields(data: Dict) -> Dict:
    """Fill missing fields with sensible defaults"""

    # Default mood if empty
    if not data.get("mood"):
        data["mood"] = ["fresh", "clean"]
        logger.info(f"Set default mood: fresh, clean")

    # Default season if empty
    if not data.get("season"):
        data["season"] = ["spring"]
        logger.info(f"Set default season: spring")

    # Default budget_tier if missing
    if not data.get("budget_tier"):
        data["budget_tier"] = "mid"

    # Default target_profile if missing
    if not data.get("target_profile"):
        data["target_profile"] = "daily_fresh"

    # Ensure creative_hints is a list
    if "creative_hints" not in data or not isinstance(data["creative_hints"], list):
        data["creative_hints"] = []

    # Ensure forbidden_ingredients is a list
    if "forbidden_ingredients" not in data or not isinstance(data["forbidden_ingredients"], list):
        data["forbidden_ingredients"] = []

    return data


# Mistral-based validation (optional - can be used for complex validation)
def validate_with_mistral_llm(brief: CreativeBrief, mistral_client=None) -> CreativeBrief:
    """
    Optional: Use Mistral LLM for validation (not implemented yet)

    This function is a placeholder for using Mistral LLM to validate
    and correct the brief. For now, we use rule-based validation.

    Args:
        brief: Input brief
        mistral_client: Optional Mistral client

    Returns:
        Validated brief
    """
    # TODO: Implement Mistral LLM-based validation if needed
    logger.info(f"Mistral LLM validation not implemented, using rule-based validation")
    return validate_and_patch(brief)


__all__ = [
    "validate_and_patch",
    "validate_with_mistral_llm"
]
