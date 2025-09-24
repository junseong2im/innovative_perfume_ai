"""
Perfumer Knowledge Tool for the LLM Orchestrator
- Accesses deep, encyclopedic knowledge of master perfumery
- Provides structured information about perfumer styles and accord formulas
"""

from pydantic import BaseModel, Field
from typing import Literal, Union, Dict, List, Optional
import logging
from ..core.exceptions_unified import handle_exceptions_async

logger = logging.getLogger(__name__)

class PerfumerStyleResponse(BaseModel):
    """Structured data for a master perfumer's style."""
    name: str = Field(..., description="Full name of the perfumer")
    style_characteristics: List[str] = Field(..., description="Key characteristics of their style")
    favorite_materials: List[str] = Field(..., description="Preferred fragrance materials")
    innovation_techniques: List[str] = Field(..., description="Signature techniques or innovations")
    signature_creations: Optional[List[str]] = Field(None, description="Notable perfume creations")
    philosophy: Optional[str] = Field(None, description="Perfumer's artistic philosophy")
    era: Optional[str] = Field(None, description="Time period of peak creativity")
    school: Optional[str] = Field(None, description="Perfumery school or tradition")

class AccordFormulaResponse(BaseModel):
    """Structured data for a famous accord."""
    name: str = Field(..., description="Name of the accord")
    ingredients: Dict[str, float] = Field(..., description="Ingredients with percentages")
    olfactory_effect: str = Field(..., description="Scent description and effect")
    creator: str = Field(..., description="Original creator or perfumer")
    historical_period: Optional[str] = Field(None, description="When it was created")
    variations: Optional[List[str]] = Field(None, description="Common variations")
    usage_notes: Optional[str] = Field(None, description="How to use this accord")

# Master Perfumer Knowledge Database
PERFUMER_DATABASE = {
    "jean-claude ellena": {
        "name": "Jean-Claude Ellena",
        "style_characteristics": [
            "Minimalist approach with precise ingredient selection",
            "Transparent and airy compositions",
            "Emphasis on subtlety over power",
            "Innovative use of synthetic molecules",
            "Water-inspired freshness"
        ],
        "favorite_materials": [
            "Hedione (methyl dihydrojasmonate)",
            "Iso E Super",
            "Ambroxan",
            "Calone (watermelon ketone)",
            "Ginger",
            "Cardamom",
            "White tea"
        ],
        "innovation_techniques": [
            "Overdosing single molecules",
            "Creating 'transparent' florals",
            "Water accord development",
            "Minimalist layering",
            "Synthetic-natural hybridization"
        ],
        "signature_creations": [
            "Un Jardin sur le Toit",
            "Terre d'Hermès",
            "Kelly Calèche",
            "Vanille Galante"
        ],
        "philosophy": "Perfume should be like a whisper, not a shout. The art is in what you leave out, not what you put in.",
        "era": "1990s-2010s",
        "school": "French minimalist"
    },
    "francis kurkdjian": {
        "name": "Francis Kurkdjian",
        "style_characteristics": [
            "Sophisticated and luxurious compositions",
            "Perfect technical execution",
            "Oriental-floral mastery",
            "Radiant and luminous effects",
            "Modern interpretation of classics"
        ],
        "favorite_materials": [
            "Rose",
            "Saffron",
            "Ambergris",
            "Cedar",
            "Jasmine",
            "Amber",
            "Cashmere wood"
        ],
        "innovation_techniques": [
            "Radiance amplification",
            "Molecular gastronomy approach",
            "Olfactory architecture",
            "Luminosity creation",
            "Emotional storytelling"
        ],
        "signature_creations": [
            "Maison Francis Kurkdjian Baccarat Rouge 540",
            "Le Labo Rose 31",
            "Elie Saab Le Parfum",
            "Narciso Rodriguez For Her"
        ],
        "philosophy": "Perfume is about creating emotion and memory. Each fragrance should tell a story.",
        "era": "2000s-present",
        "school": "French haute parfumerie"
    },
    "serge lutens": {
        "name": "Serge Lutens",
        "style_characteristics": [
            "Dark and mysterious compositions",
            "Oriental complexity",
            "Artistic and avant-garde approach",
            "Theatrical intensity",
            "Uncompromising vision"
        ],
        "favorite_materials": [
            "Incense",
            "Spices (cinnamon, clove)",
            "Dark fruits",
            "Precious woods",
            "Amber",
            "Rose",
            "Dried fruits"
        ],
        "innovation_techniques": [
            "Extreme contrasts",
            "Storytelling through scent",
            "Artistic interpretation",
            "Cultural fusion",
            "Sensory provocation"
        ],
        "signature_creations": [
            "Chergui",
            "Ambre Sultan",
            "La Fille de Berlin",
            "Tubéreuse Criminelle"
        ],
        "philosophy": "Perfume should disturb, seduce, and create desire. It's not about comfort, it's about passion.",
        "era": "1990s-present",
        "school": "Artistic avant-garde"
    },
    "edmond roudnitska": {
        "name": "Edmond Roudnitska",
        "style_characteristics": [
            "Mathematical precision in composition",
            "Classical French elegance",
            "Aldehydic mastery",
            "Perfect balance and harmony",
            "Timeless sophistication"
        ],
        "favorite_materials": [
            "Aldehydes",
            "Rose",
            "Jasmine",
            "Sandalwood",
            "Oakmoss",
            "Bergamot",
            "Ylang-ylang"
        ],
        "innovation_techniques": [
            "Aldehydic overdosing",
            "Classical pyramid structure",
            "Mathematical proportions",
            "Harmonic resonance",
            "Timeless elegance"
        ],
        "signature_creations": [
            "Diorissimo",
            "Eau Sauvage",
            "Femme by Rochas",
            "Le Parfum de Thérèse"
        ],
        "philosophy": "A perfumer is a poet who writes in odors. Each creation must be perfect in its harmony.",
        "era": "1940s-1980s",
        "school": "Classical French"
    }
}

# Famous Accord Database
ACCORD_DATABASE = {
    "chypre": {
        "name": "Chypre",
        "ingredients": {
            "bergamot": 30.0,
            "oakmoss": 25.0,
            "labdanum": 20.0,
            "patchouli": 15.0,
            "rose": 10.0
        },
        "olfactory_effect": "A sophisticated, mossy-woody accord with citrus freshness and earthy depth. Creates an elegant, timeless base.",
        "creator": "François Coty",
        "historical_period": "1917",
        "variations": [
            "Modern Chypre (synthetic oakmoss)",
            "Fruity Chypre (added fruits)",
            "Floral Chypre (enhanced florals)",
            "Leather Chypre (added leather notes)"
        ],
        "usage_notes": "Foundation accord for sophisticated perfumes. Adjust bergamot for brightness, oakmoss for earthiness."
    },
    "fougère": {
        "name": "Fougère",
        "ingredients": {
            "lavender": 40.0,
            "geranium": 20.0,
            "oakmoss": 20.0,
            "coumarin": 15.0,
            "bergamot": 5.0
        },
        "olfactory_effect": "Fresh, herbaceous accord with aromatic lavender, hay-like coumarin, and mossy base. Masculine and timeless.",
        "creator": "Paul Parquet",
        "historical_period": "1882",
        "variations": [
            "Modern Fougère (synthetic coumarin)",
            "Aromatic Fougère (added herbs)",
            "Woody Fougère (cedar, sandalwood)",
            "Fresh Fougère (aquatic notes)"
        ],
        "usage_notes": "Classic masculine foundation. Balance lavender with geranium, oakmoss provides depth."
    },
    "aldehydic floral": {
        "name": "Aldehydic Floral",
        "ingredients": {
            "aldehydes_c10_c12": 25.0,
            "rose": 20.0,
            "jasmine": 20.0,
            "ylang_ylang": 15.0,
            "sandalwood": 10.0,
            "musk": 10.0
        },
        "olfactory_effect": "Sparkling, radiant floral with soapy-waxy aldehydes creating lift and brightness. Sophisticated and timeless.",
        "creator": "Ernest Beaux",
        "historical_period": "1920s",
        "variations": [
            "Soft Aldehydic (reduced aldehydes)",
            "Powdery Aldehydic (added iris)",
            "Modern Aldehydic (synthetic aldehydes)",
            "Fruity Aldehydic (added fruits)"
        ],
        "usage_notes": "Iconic feminine accord. Aldehydes must be balanced carefully - too much creates harshness."
    },
    "amber oriental": {
        "name": "Amber Oriental",
        "ingredients": {
            "labdanum": 30.0,
            "benzoin": 25.0,
            "vanilla": 20.0,
            "sandalwood": 15.0,
            "cinnamon": 10.0
        },
        "olfactory_effect": "Warm, resinous, and sweet accord with powdery-balsamic character. Sensual and enveloping.",
        "creator": "Jacques Guerlain",
        "historical_period": "1920s",
        "variations": [
            "Vanilla Amber (enhanced vanilla)",
            "Spicy Amber (added spices)",
            "Floral Amber (added florals)",
            "Woody Amber (added woods)"
        ],
        "usage_notes": "Base for oriental fragrances. Adjust vanilla for sweetness, spices for warmth."
    },
    "white floral": {
        "name": "White Floral",
        "ingredients": {
            "jasmine": 35.0,
            "tuberose": 25.0,
            "gardenia": 20.0,
            "ylang_ylang": 15.0,
            "white_musk": 5.0
        },
        "olfactory_effect": "Intoxicating, narcotic floral blend with creamy, indolic character. Sensual and feminine.",
        "creator": "Traditional blend",
        "historical_period": "Classical perfumery",
        "variations": [
            "Clean White Floral (reduced indoles)",
            "Tropical White Floral (coconut, frangipani)",
            "Aldehydic White Floral (added aldehydes)",
            "Green White Floral (added green notes)"
        ],
        "usage_notes": "Heart of feminine fragrances. Balance indolic richness with fresh top notes."
    },
    "aquatic marine": {
        "name": "Aquatic Marine",
        "ingredients": {
            "calone": 30.0,
            "marine_algae": 25.0,
            "sea_salt": 20.0,
            "ambergris": 15.0,
            "white_musk": 10.0
        },
        "olfactory_effect": "Fresh, oceanic accord evoking sea breeze and coastal air. Modern and invigorating.",
        "creator": "Olivier Cresp",
        "historical_period": "1990s",
        "variations": [
            "Ozonic Aquatic (added ozone notes)",
            "Mineral Aquatic (salt, stones)",
            "Tropical Aquatic (coconut water)",
            "Clean Aquatic (soap notes)"
        ],
        "usage_notes": "Modern fresh accord. Calone provides watermelon-like freshness, use sparingly."
    }
}

async def query_knowledge_base(
    query_type: Literal["perfumer_style", "accord_formula"],
    name: str
) -> Union[PerfumerStyleResponse, AccordFormulaResponse]:
    """
    ## LLM Tool Description
    Use this tool to access a deep, encyclopedic knowledge base of master perfumery.
    It provides structured information about the styles of legendary perfumers or the formulas of iconic accords.

    ## When to use
    - When a user asks to create a perfume "in the style of" a specific perfumer (e.g., "Jean-Claude Ellena").
    - When you need to understand the composition of a historical fragrance accord (e.g., "What is in the classic 'Chypre' accord?").
    - To retrieve the signature characteristics or favorite ingredients of a master perfumer for inspiration.

    Args:
        query_type: Type of query - either "perfumer_style" or "accord_formula"
        name: Name of the perfumer or accord to query

    Returns:
        PerfumerStyleResponse for perfumer queries or AccordFormulaResponse for accord queries
    """
    try:
        normalized_name = name.lower().strip()

        if query_type == "perfumer_style":
            # Search for perfumer in database
            perfumer_data = PERFUMER_DATABASE.get(normalized_name)

            if not perfumer_data:
                # Try partial matching
                for key in PERFUMER_DATABASE.keys():
                    if normalized_name in key or key in normalized_name:
                        perfumer_data = PERFUMER_DATABASE[key]
                        break

            if not perfumer_data:
                # Return a default response for unknown perfumers
                logger.warning(f"Perfumer '{name}' not found in database")
                return PerfumerStyleResponse(
                    name=name,
                    style_characteristics=["Information not available"],
                    favorite_materials=["Unknown"],
                    innovation_techniques=["Data not available"],
                    signature_creations=[],
                    philosophy="Information not available in knowledge base",
                    era="Unknown",
                    school="Unknown"
                )

            return PerfumerStyleResponse(**perfumer_data)

        elif query_type == "accord_formula":
            # Search for accord in database
            accord_data = ACCORD_DATABASE.get(normalized_name)

            if not accord_data:
                # Try partial matching
                for key in ACCORD_DATABASE.keys():
                    if normalized_name in key or key in normalized_name:
                        accord_data = ACCORD_DATABASE[key]
                        break

            if not accord_data:
                # Return a default response for unknown accords
                logger.warning(f"Accord '{name}' not found in database")
                return AccordFormulaResponse(
                    name=name,
                    ingredients={"unknown": 100.0},
                    olfactory_effect="Information not available",
                    creator="Unknown",
                    historical_period="Unknown",
                    variations=[],
                    usage_notes="Data not available in knowledge base"
                )

            return AccordFormulaResponse(**accord_data)

        else:
            raise ValueError(f"Invalid query_type: {query_type}")

    except Exception as e:
        logger.error(f"Knowledge base query failed: {e}")
        raise

# Helper function to list available perfumers
async def list_available_perfumers() -> List[str]:
    """Get a list of all available perfumers in the knowledge base."""
    return [data["name"] for data in PERFUMER_DATABASE.values()]

# Helper function to list available accords
async def list_available_accords() -> List[str]:
    """Get a list of all available accords in the knowledge base."""
    return [data["name"] for data in ACCORD_DATABASE.values()]

# Helper function to search for perfumers by characteristics
async def search_perfumers_by_style(characteristic: str) -> List[str]:
    """Search for perfumers who have a specific style characteristic."""
    matching_perfumers = []

    characteristic_lower = characteristic.lower()

    for perfumer_key, perfumer_data in PERFUMER_DATABASE.items():
        # Check style characteristics
        for style_char in perfumer_data["style_characteristics"]:
            if characteristic_lower in style_char.lower():
                matching_perfumers.append(perfumer_data["name"])
                break

        # Check philosophy
        if characteristic_lower in perfumer_data.get("philosophy", "").lower():
            if perfumer_data["name"] not in matching_perfumers:
                matching_perfumers.append(perfumer_data["name"])

    return matching_perfumers

# Helper function to search accords by ingredient
async def search_accords_by_ingredient(ingredient: str) -> List[str]:
    """Search for accords that contain a specific ingredient."""
    matching_accords = []
    ingredient_lower = ingredient.lower()

    for accord_key, accord_data in ACCORD_DATABASE.items():
        for ingredient_name in accord_data["ingredients"].keys():
            if ingredient_lower in ingredient_name.lower():
                matching_accords.append(accord_data["name"])
                break

    return matching_accords