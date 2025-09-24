"""
Scientific Validator Tool for the LLM Orchestrator
- Provides scientific analysis of fragrance compositions
- Evaluates harmony, stability, longevity and provides suggestions
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import math
import asyncio
from ..core.exceptions_unified import handle_exceptions_async

logger = logging.getLogger(__name__)

class NotesComposition(BaseModel):
    """Defines the note structure of a fragrance recipe."""
    top: List[str] = Field(..., description="List of top notes.")
    middle: List[str] = Field(..., description="List of middle (heart) notes.")
    base: List[str] = Field(..., description="List of base notes.")

class ValidationResult(BaseModel):
    """The scientific assessment of a fragrance composition."""
    harmony_score: float = Field(..., description="Overall blend harmony (0.0 to 1.0). Higher is better.")
    stability_score: float = Field(..., description="Chemical stability of the blend (0.0 to 1.0).")
    longevity_hours: float = Field(..., description="Estimated longevity in hours.")
    key_risks: List[str] = Field(..., description="Potential issues, e.g., 'Geosmin may overpower delicate florals'.")
    suggestions: List[str] = Field(..., description="Actionable suggestions for improvement, e.g., 'Consider adding Iris as a bridge note'.")
    sillage_rating: Optional[str] = Field(None, description="Estimated sillage strength (weak, moderate, strong).")
    complexity_rating: Optional[str] = Field(None, description="Complexity level (simple, balanced, complex).")
    balance_analysis: Optional[Dict[str, float]] = Field(None, description="Note type balance percentages.")

# 향료 노트의 화학적 특성과 상호작용 데이터베이스
FRAGRANCE_DATABASE = {
    # Citrus family
    "bergamot": {"family": "citrus", "volatility": 0.9, "intensity": 7, "longevity": 2, "harmony_with": ["lavender", "rose", "jasmine"], "conflicts_with": ["patchouli"]},
    "lemon": {"family": "citrus", "volatility": 0.95, "intensity": 8, "longevity": 1.5, "harmony_with": ["mint", "basil", "thyme"], "conflicts_with": ["heavy_musks"]},
    "orange": {"family": "citrus", "volatility": 0.85, "intensity": 6, "longevity": 2.5, "harmony_with": ["cinnamon", "clove", "vanilla"], "conflicts_with": []},
    "grapefruit": {"family": "citrus", "volatility": 0.9, "intensity": 7.5, "longevity": 2, "harmony_with": ["rosemary", "juniper"], "conflicts_with": ["sweet_florals"]},

    # Floral family
    "rose": {"family": "floral", "volatility": 0.6, "intensity": 8, "longevity": 6, "harmony_with": ["sandalwood", "vanilla", "patchouli"], "conflicts_with": ["eucalyptus"]},
    "jasmine": {"family": "floral", "volatility": 0.5, "intensity": 9, "longevity": 8, "harmony_with": ["amber", "musk", "benzoin"], "conflicts_with": ["sharp_citrus"]},
    "lavender": {"family": "floral", "volatility": 0.7, "intensity": 6, "longevity": 4, "harmony_with": ["bergamot", "rosemary", "cedar"], "conflicts_with": ["heavy_orientals"]},
    "ylang ylang": {"family": "floral", "volatility": 0.4, "intensity": 9, "longevity": 7, "harmony_with": ["coconut", "vanilla", "sandalwood"], "conflicts_with": ["green_notes"]},

    # Woody family
    "sandalwood": {"family": "woody", "volatility": 0.2, "intensity": 7, "longevity": 12, "harmony_with": ["rose", "jasmine", "vanilla"], "conflicts_with": []},
    "cedar": {"family": "woody", "volatility": 0.3, "intensity": 6, "longevity": 10, "harmony_with": ["lavender", "juniper", "iris"], "conflicts_with": ["sweet_fruits"]},
    "oakmoss": {"family": "woody", "volatility": 0.1, "intensity": 8, "longevity": 16, "harmony_with": ["bergamot", "rose", "patchouli"], "conflicts_with": ["light_florals"]},
    "patchouli": {"family": "woody", "volatility": 0.15, "intensity": 9, "longevity": 14, "harmony_with": ["rose", "vanilla", "amber"], "conflicts_with": ["citrus", "light_florals"]},

    # Oriental/Amber family
    "vanilla": {"family": "oriental", "volatility": 0.25, "intensity": 8, "longevity": 10, "harmony_with": ["sandalwood", "amber", "musk"], "conflicts_with": ["green_notes", "aquatic"]},
    "amber": {"family": "oriental", "volatility": 0.2, "intensity": 7, "longevity": 12, "harmony_with": ["vanilla", "benzoin", "labdanum"], "conflicts_with": ["fresh_citrus"]},
    "benzoin": {"family": "oriental", "volatility": 0.18, "intensity": 6, "longevity": 11, "harmony_with": ["vanilla", "cinnamon", "rose"], "conflicts_with": ["aquatic", "ozonic"]},
    "labdanum": {"family": "oriental", "volatility": 0.15, "intensity": 8, "longevity": 14, "harmony_with": ["amber", "rose", "oakmoss"], "conflicts_with": ["light_citrus"]},

    # Fresh/Aquatic family
    "sea breeze": {"family": "aquatic", "volatility": 0.8, "intensity": 5, "longevity": 3, "harmony_with": ["cucumber", "mint", "lime"], "conflicts_with": ["heavy_orientals", "animalic"]},
    "ozone": {"family": "aquatic", "volatility": 0.85, "intensity": 4, "longevity": 2.5, "harmony_with": ["white_tea", "lily"], "conflicts_with": ["vanilla", "amber"]},

    # Herbal/Green family
    "mint": {"family": "herbal", "volatility": 0.9, "intensity": 8, "longevity": 2, "harmony_with": ["basil", "eucalyptus", "lemon"], "conflicts_with": ["heavy_florals"]},
    "basil": {"family": "herbal", "volatility": 0.8, "intensity": 7, "longevity": 3, "harmony_with": ["tomato_leaf", "green_pepper"], "conflicts_with": ["sweet_orientals"]},
    "eucalyptus": {"family": "herbal", "volatility": 0.85, "intensity": 9, "longevity": 3, "harmony_with": ["mint", "pine"], "conflicts_with": ["rose", "jasmine"]},

    # Spice family
    "cinnamon": {"family": "spice", "volatility": 0.6, "intensity": 8, "longevity": 6, "harmony_with": ["vanilla", "orange", "clove"], "conflicts_with": ["delicate_florals"]},
    "clove": {"family": "spice", "volatility": 0.5, "intensity": 9, "longevity": 8, "harmony_with": ["orange", "rose", "carnation"], "conflicts_with": ["light_citrus"]},
    "cardamom": {"family": "spice", "volatility": 0.7, "intensity": 6, "longevity": 4, "harmony_with": ["rose", "saffron", "oudh"], "conflicts_with": ["aquatic"]},

    # Musk family
    "white musk": {"family": "musk", "volatility": 0.1, "intensity": 5, "longevity": 15, "harmony_with": ["clean_cotton", "lily"], "conflicts_with": ["animalic_notes"]},
    "red musk": {"family": "musk", "volatility": 0.12, "intensity": 7, "longevity": 12, "harmony_with": ["rose", "patchouli"], "conflicts_with": ["fresh_citrus"]},

    # Korean traditional notes - Royal/Palace Culture
    "agarwood": {"family": "oriental", "volatility": 0.05, "intensity": 10, "longevity": 20, "harmony_with": ["sandalwood", "rose", "saffron"], "conflicts_with": ["light_citrus", "aquatic"]},
    "ambergris": {"family": "animalic", "volatility": 0.08, "intensity": 9, "longevity": 18, "harmony_with": ["agarwood", "sandalwood", "rose"], "conflicts_with": ["fresh_citrus", "light_florals"]},
    "korean_pine": {"family": "coniferous", "volatility": 0.4, "intensity": 7, "longevity": 8, "harmony_with": ["cedar", "agarwood", "incense"], "conflicts_with": ["sweet_florals", "gourmand"]},
    "korean_cedar": {"family": "woody", "volatility": 0.3, "intensity": 8, "longevity": 12, "harmony_with": ["agarwood", "sandalwood", "frankincense"], "conflicts_with": ["light_citrus", "aquatic"]},
    "frankincense": {"family": "resinous", "volatility": 0.2, "intensity": 8, "longevity": 14, "harmony_with": ["agarwood", "myrrh", "sandalwood"], "conflicts_with": ["citrus", "aquatic"]},
    "myrrh": {"family": "resinous", "volatility": 0.15, "intensity": 7, "longevity": 12, "harmony_with": ["frankincense", "agarwood", "labdanum"], "conflicts_with": ["fresh_notes"]},
    "korean_cinnamon": {"family": "spice", "volatility": 0.5, "intensity": 9, "longevity": 8, "harmony_with": ["agarwood", "clove", "cardamom"], "conflicts_with": ["light_florals", "aquatic"]},
    "korean_clove": {"family": "spice", "volatility": 0.45, "intensity": 9, "longevity": 9, "harmony_with": ["agarwood", "cinnamon", "orange"], "conflicts_with": ["delicate_florals"]},
    "imperial_musk": {"family": "musk", "volatility": 0.1, "intensity": 8, "longevity": 16, "harmony_with": ["agarwood", "ambergris", "rose"], "conflicts_with": ["light_citrus"]},
    "dragon_blood": {"family": "resinous", "volatility": 0.12, "intensity": 9, "longevity": 15, "harmony_with": ["agarwood", "frankincense", "patchouli"], "conflicts_with": ["fresh_notes", "aquatic"]},

    # Traditional Korean seasonal/cultural notes
    "pine needle": {"family": "coniferous", "volatility": 0.7, "intensity": 6, "longevity": 5, "harmony_with": ["cedar", "juniper"], "conflicts_with": ["sweet_florals"]},
    "bamboo": {"family": "green", "volatility": 0.6, "intensity": 4, "longevity": 4, "harmony_with": ["green_tea", "cucumber"], "conflicts_with": ["heavy_orientals"]},
    "ginseng": {"family": "herbal", "volatility": 0.4, "intensity": 6, "longevity": 7, "harmony_with": ["ginger", "white_tea"], "conflicts_with": ["sweet_fruits"]},
    "korean_pear": {"family": "fruity", "volatility": 0.8, "intensity": 5, "longevity": 3, "harmony_with": ["white_tea", "peony"], "conflicts_with": ["heavy_spices"]},
    "plum_blossom": {"family": "floral", "volatility": 0.6, "intensity": 5, "longevity": 5, "harmony_with": ["white_tea", "bamboo"], "conflicts_with": ["heavy_spices"]},
    "korean_tea": {"family": "herbal", "volatility": 0.5, "intensity": 4, "longevity": 6, "harmony_with": ["bamboo", "ginseng", "plum_blossom"], "conflicts_with": ["animalic"]}
}

def normalize_note_name(note: str) -> str:
    """Normalize note names for database lookup."""
    note = note.lower().strip()
    # Handle common variations including Korean traditional royal notes
    mapping = {
        # Basic notes
        "베르가못": "bergamot",
        "레몬": "lemon",
        "오렌지": "orange",
        "자몽": "grapefruit",
        "장미": "rose",
        "재스민": "jasmine",
        "라벤더": "lavender",
        "일랑일랑": "ylang ylang",
        "샌달우드": "sandalwood",
        "백단향": "sandalwood",  # Korean traditional name
        "시더": "cedar",
        "오크모스": "oakmoss",
        "패출리": "patchouli",
        "바닐라": "vanilla",
        "앰버": "amber",
        "벤조인": "benzoin",
        "민트": "mint",
        "바질": "basil",
        "유칼립투스": "eucalyptus",
        "계피": "cinnamon",
        "정향": "clove",
        "카다몬": "cardamom",
        "화이트 머스크": "white musk",
        "레드 머스크": "red musk",

        # Korean royal/palace culture notes
        "침향": "agarwood",
        "아가우드": "agarwood",
        "용연향": "ambergris",
        "앰버그리스": "ambergris",
        "소나무": "korean_pine",
        "한국소나무": "korean_pine",
        "한국삼나무": "korean_cedar",
        "유향": "frankincense",
        "프랑킨센스": "frankincense",
        "몰약": "myrrh",
        "미르": "myrrh",
        "한국계피": "korean_cinnamon",
        "한국정향": "korean_clove",
        "황실머스크": "imperial_musk",
        "제왕머스크": "imperial_musk",
        "용혈": "dragon_blood",
        "드래곤블러드": "dragon_blood",

        # Traditional seasonal/cultural notes
        "솔잎": "pine needle",
        "대나무": "bamboo",
        "인삼": "ginseng",
        "한국배": "korean_pear",
        "매화": "plum_blossom",
        "매화꽃": "plum_blossom",
        "차": "korean_tea",
        "한국차": "korean_tea",
        "녹차": "korean_tea"
    }
    return mapping.get(note, note)

def get_note_properties(note: str) -> Dict:
    """Get properties for a note from the database."""
    normalized_note = normalize_note_name(note)
    return FRAGRANCE_DATABASE.get(normalized_note, {
        "family": "unknown",
        "volatility": 0.5,
        "intensity": 5,
        "longevity": 5,
        "harmony_with": [],
        "conflicts_with": []
    })

async def validate_composition(composition: NotesComposition) -> ValidationResult:
    """
    ## LLM Tool Description
    Use this tool ONLY AFTER you have created a new, hypothetical fragrance recipe.
    This tool acts as your lab assistant, providing a scientific analysis of your creation.
    It does NOT search for existing perfumes. It evaluates the chemical and artistic viability of a note combination.

    ## When to use
    - After generating a draft of a new perfume recipe to check if it's well-balanced and stable.
    - When you need to improve a recipe and want scientific suggestions on which notes to add or remove to increase harmony.
    - To estimate the longevity of a new creation.

    Args:
        composition: The fragrance composition with top, middle, and base notes

    Returns:
        ValidationResult with harmony score, stability, longevity estimate, risks, and suggestions
    """
    try:
        # Analyze note structure and balance
        total_notes = len(composition.top) + len(composition.middle) + len(composition.base)

        if total_notes == 0:
            return ValidationResult(
                harmony_score=0.0,
                stability_score=0.0,
                longevity_hours=0.0,
                key_risks=["No notes provided"],
                suggestions=["Add at least one note from each category (top, middle, base)"]
            )

        # Calculate note type balance
        top_ratio = len(composition.top) / total_notes
        middle_ratio = len(composition.middle) / total_notes
        base_ratio = len(composition.base) / total_notes

        balance_analysis = {
            "top_notes": round(top_ratio * 100, 1),
            "middle_notes": round(middle_ratio * 100, 1),
            "base_notes": round(base_ratio * 100, 1)
        }

        # Analyze harmony between notes
        harmony_score = await _calculate_harmony_score(composition)

        # Analyze chemical stability
        stability_score = await _calculate_stability_score(composition)

        # Estimate longevity
        longevity_hours = await _estimate_longevity(composition)

        # Identify potential risks
        key_risks = await _identify_risks(composition)

        # Generate improvement suggestions
        suggestions = await _generate_suggestions(composition, harmony_score, stability_score, balance_analysis)

        # Determine sillage rating
        sillage_rating = await _calculate_sillage(composition)

        # Determine complexity rating
        complexity_rating = await _calculate_complexity(composition)

        result = ValidationResult(
            harmony_score=harmony_score,
            stability_score=stability_score,
            longevity_hours=longevity_hours,
            key_risks=key_risks,
            suggestions=suggestions,
            sillage_rating=sillage_rating,
            complexity_rating=complexity_rating,
            balance_analysis=balance_analysis
        )

        logger.info(f"Composition validation completed - Harmony: {harmony_score:.2f}, Stability: {stability_score:.2f}")
        return result

    except Exception as e:
        logger.error(f"Composition validation failed: {e}")
        raise

async def _calculate_harmony_score(composition: NotesComposition) -> float:
    """Calculate overall harmony score based on note compatibility."""
    all_notes = composition.top + composition.middle + composition.base

    if len(all_notes) < 2:
        return 0.5  # Neutral score for single note

    harmony_points = 0
    total_comparisons = 0

    # Check each pair of notes for compatibility
    for i, note1 in enumerate(all_notes):
        props1 = get_note_properties(note1)

        for j, note2 in enumerate(all_notes[i+1:], i+1):
            props2 = get_note_properties(note2)
            total_comparisons += 1

            # Check if notes harmonize well
            if normalize_note_name(note2) in props1.get("harmony_with", []):
                harmony_points += 2
            elif normalize_note_name(note2) in props1.get("conflicts_with", []):
                harmony_points -= 2
            elif props1["family"] == props2["family"]:
                harmony_points += 1  # Same family usually harmonizes
            else:
                # Check intensity compatibility
                intensity_diff = abs(props1["intensity"] - props2["intensity"])
                if intensity_diff <= 2:
                    harmony_points += 0.5
                elif intensity_diff >= 6:
                    harmony_points -= 1

    if total_comparisons == 0:
        return 0.5

    # Normalize to 0-1 scale
    raw_score = harmony_points / (total_comparisons * 2)  # Max possible is 2 per comparison
    return max(0.0, min(1.0, 0.5 + raw_score))  # Center around 0.5

async def _calculate_stability_score(composition: NotesComposition) -> float:
    """Calculate chemical stability based on volatility patterns."""
    # Check if volatility decreases from top to base (ideal pyramid structure)
    top_volatility = sum(get_note_properties(note)["volatility"] for note in composition.top) / max(len(composition.top), 1)
    middle_volatility = sum(get_note_properties(note)["volatility"] for note in composition.middle) / max(len(composition.middle), 1)
    base_volatility = sum(get_note_properties(note)["volatility"] for note in composition.base) / max(len(composition.base), 1)

    # Ideal: top > middle > base volatility
    stability_score = 0.8  # Base score

    if top_volatility > middle_volatility:
        stability_score += 0.1
    else:
        stability_score -= 0.15

    if middle_volatility > base_volatility:
        stability_score += 0.1
    else:
        stability_score -= 0.15

    # Check for extreme volatility differences that might cause instability
    volatilities = [top_volatility, middle_volatility, base_volatility]
    volatility_range = max(volatilities) - min(volatilities)

    if volatility_range > 0.7:
        stability_score -= 0.1  # Too much range can cause instability

    return max(0.0, min(1.0, stability_score))

async def _estimate_longevity(composition: NotesComposition) -> float:
    """Estimate overall longevity in hours."""
    if not composition.base:
        return 2.0  # Very short without base notes

    # Base notes primarily determine longevity
    base_longevity = sum(get_note_properties(note)["longevity"] for note in composition.base) / len(composition.base)

    # Middle notes contribute to longevity
    middle_longevity = sum(get_note_properties(note)["longevity"] for note in composition.middle) / max(len(composition.middle), 1)

    # Weighted average favoring base notes
    estimated_longevity = (base_longevity * 0.7) + (middle_longevity * 0.3)

    # Apply modifier based on composition balance
    total_notes = len(composition.top) + len(composition.middle) + len(composition.base)
    base_ratio = len(composition.base) / total_notes

    if base_ratio < 0.2:  # Too few base notes
        estimated_longevity *= 0.7
    elif base_ratio > 0.5:  # Many base notes
        estimated_longevity *= 1.2

    return round(estimated_longevity, 1)

async def _identify_risks(composition: NotesComposition) -> List[str]:
    """Identify potential composition risks."""
    risks = []
    all_notes = composition.top + composition.middle + composition.base

    # Check for conflicting notes
    for i, note1 in enumerate(all_notes):
        props1 = get_note_properties(note1)
        conflicts = props1.get("conflicts_with", [])

        for note2 in all_notes[i+1:]:
            if normalize_note_name(note2) in conflicts:
                risks.append(f"{note1.title()} may conflict with {note2.title()}")

    # Check balance issues
    total_notes = len(all_notes)
    top_ratio = len(composition.top) / total_notes
    middle_ratio = len(composition.middle) / total_notes
    base_ratio = len(composition.base) / total_notes

    if top_ratio > 0.6:
        risks.append("Too many top notes may cause the fragrance to be fleeting")
    if base_ratio < 0.15:
        risks.append("Insufficient base notes may result in poor longevity")
    if middle_ratio < 0.2:
        risks.append("Lack of heart notes may create a disconnected scent progression")

    # Check for overpowering notes
    high_intensity_notes = [note for note in all_notes if get_note_properties(note)["intensity"] >= 8]
    if len(high_intensity_notes) > total_notes * 0.4:
        risks.append("Multiple high-intensity notes may create an overwhelming blend")

    return risks

async def _generate_suggestions(composition: NotesComposition, harmony_score: float, stability_score: float, balance_analysis: Dict[str, float]) -> List[str]:
    """Generate improvement suggestions."""
    suggestions = []

    # Harmony improvement suggestions
    if harmony_score < 0.6:
        suggestions.append("Consider adding bridge notes like iris or white tea to improve harmony")
        suggestions.append("Review conflicting note combinations and consider alternatives")

    # Stability improvement suggestions
    if stability_score < 0.7:
        suggestions.append("Ensure volatility decreases from top to base notes for better stability")
        if len(composition.base) == 0:
            suggestions.append("Add base notes like sandalwood, vanilla, or musk for stability")

    # Balance suggestions
    if balance_analysis["base_notes"] < 20:
        suggestions.append("Increase base note proportion to at least 20% for better longevity")
    if balance_analysis["top_notes"] > 50:
        suggestions.append("Reduce top note proportion to avoid overwhelming initial impression")
    if balance_analysis["middle_notes"] < 25:
        suggestions.append("Add more heart notes to create a smoother scent transition")

    # Family-specific suggestions
    families = [get_note_properties(note)["family"] for note in composition.top + composition.middle + composition.base]
    family_counts = {}
    for family in families:
        family_counts[family] = family_counts.get(family, 0) + 1

    # If only one family, suggest diversity
    if len(set(families)) == 1:
        suggestions.append("Consider adding notes from different fragrance families for complexity")

    # Specific note suggestions based on what's missing
    has_floral = any(get_note_properties(note)["family"] == "floral" for note in composition.middle)
    has_woody = any(get_note_properties(note)["family"] == "woody" for note in composition.base)

    if not has_floral and len(composition.middle) < 3:
        suggestions.append("Consider adding a floral heart note like rose or jasmine")
    if not has_woody and len(composition.base) < 2:
        suggestions.append("Consider adding a woody base note like sandalwood or cedar")

    return suggestions[:5]  # Limit to top 5 suggestions

async def _calculate_sillage(composition: NotesComposition) -> str:
    """Calculate sillage (projection) rating."""
    all_notes = composition.top + composition.middle + composition.base
    avg_intensity = sum(get_note_properties(note)["intensity"] for note in all_notes) / len(all_notes)

    if avg_intensity < 5:
        return "weak"
    elif avg_intensity < 7:
        return "moderate"
    else:
        return "strong"

async def _calculate_complexity(composition: NotesComposition) -> str:
    """Calculate complexity rating."""
    total_notes = len(composition.top) + len(composition.middle) + len(composition.base)
    families = set(get_note_properties(note)["family"] for note in composition.top + composition.middle + composition.base)

    complexity_score = total_notes + len(families) * 2

    if complexity_score < 8:
        return "simple"
    elif complexity_score < 15:
        return "balanced"
    else:
        return "complex"