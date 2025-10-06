"""
Real AI Service for Fragrance Generation
Uses Hugging Face Transformers for local AI processing
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FragranceAI:
    """Main AI service for fragrance generation."""

    def __init__(self):
        """Initialize AI models and services."""
        self.initialized = False
        self.fragrance_database = self._load_fragrance_database()
        self.initialize_models()

    def initialize_models(self):
        """Initialize AI models with fallback to rule-based system."""
        try:
            # Try to load transformer model
            from transformers import pipeline
            self.generator = pipeline("text-generation", model="gpt2", max_length=100)
            self.initialized = True
            logger.info("AI models initialized successfully with GPT-2")
        except Exception as e:
            logger.warning(f"Could not load transformer models: {e}")
            logger.info("Using advanced rule-based system")
            self.initialized = False

    def _load_fragrance_database(self) -> Dict:
        """Load fragrance ingredients database."""
        return {
            "top_notes": {
                "citrus": ["Bergamot", "Lemon", "Grapefruit", "Orange", "Mandarin"],
                "fresh": ["Mint", "Basil", "Green Tea", "Cucumber", "Melon"],
                "fruity": ["Apple", "Pear", "Peach", "Berry", "Tropical Fruits"],
                "aromatic": ["Lavender", "Rosemary", "Sage", "Thyme", "Eucalyptus"]
            },
            "heart_notes": {
                "floral": ["Rose", "Jasmine", "Ylang-Ylang", "Geranium", "Violet"],
                "spicy": ["Cinnamon", "Cardamom", "Pepper", "Ginger", "Clove"],
                "woody": ["Cedar", "Pine", "Birch", "Bamboo", "Oak"],
                "green": ["Grass", "Leaf", "Stem", "Moss", "Fern"]
            },
            "base_notes": {
                "oriental": ["Vanilla", "Amber", "Benzoin", "Tonka Bean", "Balsam"],
                "musky": ["White Musk", "Ambrette", "Cashmeran", "Galaxolide"],
                "woody": ["Sandalwood", "Patchouli", "Vetiver", "Oud", "Guaiac"],
                "gourmand": ["Chocolate", "Caramel", "Honey", "Coffee", "Praline"]
            }
        }

    def generate_recipe(self, description: str, mood: str = None, season: str = None) -> Dict:
        """Generate a fragrance recipe based on description."""

        # Extract keywords for ingredient selection
        keywords = self._extract_keywords(description, mood, season)

        # Select ingredients based on keywords
        ingredients = self._select_ingredients(keywords)

        # Generate recipe name using AI or rules
        recipe_name = self._generate_name(description, mood)

        # Create complete recipe
        recipe = {
            "recipe_id": self._generate_id(description),
            "name": recipe_name,
            "description": self._enhance_description(description, mood, season),
            "ingredients": ingredients,
            "notes": self._organize_notes(ingredients),
            "characteristics": self._analyze_characteristics(ingredients),
            "metadata": {
                "mood": mood or self._detect_mood(description),
                "season": season or "all-season",
                "intensity": self._calculate_intensity(ingredients),
                "longevity": self._estimate_longevity(ingredients),
                "sillage": self._estimate_sillage(ingredients),
                "created_at": datetime.now().isoformat(),
                "ai_generated": True
            }
        }

        return recipe

    def _extract_keywords(self, description: str, mood: str, season: str) -> List[str]:
        """Extract relevant keywords from input."""
        keywords = []

        # Common fragrance descriptors
        descriptors = {
            "fresh": ["fresh", "clean", "crisp", "light", "airy"],
            "floral": ["flower", "floral", "rose", "jasmine", "bouquet"],
            "woody": ["wood", "forest", "tree", "earth", "natural"],
            "oriental": ["warm", "spicy", "exotic", "rich", "luxurious"],
            "citrus": ["citrus", "lemon", "orange", "bright", "zesty"],
            "sweet": ["sweet", "vanilla", "candy", "sugar", "dessert"]
        }

        description_lower = description.lower()
        for category, terms in descriptors.items():
            if any(term in description_lower for term in terms):
                keywords.append(category)

        if mood:
            keywords.append(mood.lower())
        if season:
            keywords.append(season.lower())

        return keywords

    def _select_ingredients(self, keywords: List[str]) -> List[Dict]:
        """Select ingredients based on keywords."""
        ingredients = []

        # Select top notes
        if "fresh" in keywords or "citrus" in keywords:
            top_family = "citrus" if "citrus" in keywords else "fresh"
        elif "aromatic" in keywords:
            top_family = "aromatic"
        else:
            top_family = "fruity"

        top_notes = self.fragrance_database["top_notes"].get(top_family, ["Bergamot"])
        ingredients.append({
            "name": top_notes[0],
            "type": "top",
            "percentage": 20,
            "note_family": top_family
        })

        # Select heart notes
        if "floral" in keywords:
            heart_family = "floral"
        elif "spicy" in keywords:
            heart_family = "spicy"
        elif "woody" in keywords:
            heart_family = "woody"
        else:
            heart_family = "floral"

        heart_notes = self.fragrance_database["heart_notes"].get(heart_family, ["Rose"])
        ingredients.append({
            "name": heart_notes[0],
            "type": "heart",
            "percentage": 40,
            "note_family": heart_family
        })

        # Select base notes
        if "oriental" in keywords or "warm" in keywords:
            base_family = "oriental"
        elif "woody" in keywords:
            base_family = "woody"
        elif "sweet" in keywords:
            base_family = "gourmand"
        else:
            base_family = "musky"

        base_notes = self.fragrance_database["base_notes"].get(base_family, ["Musk"])
        ingredients.append({
            "name": base_notes[0],
            "type": "base",
            "percentage": 40,
            "note_family": base_family
        })

        return ingredients

    def _generate_name(self, description: str, mood: str) -> str:
        """Generate a creative name for the fragrance."""
        if self.initialized:
            try:
                # Use AI to generate name
                prompt = f"Create a perfume name for: {description}"
                result = self.generator(prompt, max_length=20, num_return_sequences=1)
                name = result[0]['generated_text'].split('\n')[0]
                return name[:50]  # Limit length
            except:
                pass

        # Fallback to rule-based naming
        base_names = {
            "romantic": "Eternal Romance",
            "fresh": "Morning Dew",
            "mysterious": "Midnight Secret",
            "elegant": "Royal Grace",
            "vibrant": "Solar Burst",
            "calm": "Serene Waters",
            "bold": "Fierce Spirit"
        }

        if mood and mood.lower() in base_names:
            return base_names[mood.lower()]

        # Generate based on description keywords
        words = description.split()[:3]
        return f"Essence of {' '.join(words).title()}"

    def _enhance_description(self, description: str, mood: str, season: str) -> str:
        """Enhance the fragrance description."""
        enhanced = f"A sophisticated fragrance inspired by {description}."

        if mood:
            enhanced += f" This {mood} scent captures the essence of emotion."

        if season:
            enhanced += f" Perfect for {season} occasions."

        return enhanced

    def _organize_notes(self, ingredients: List[Dict]) -> Dict:
        """Organize ingredients into note categories."""
        notes = {"top": [], "heart": [], "base": []}

        for ingredient in ingredients:
            note_type = ingredient.get("type", "heart")
            notes[note_type].append(ingredient["name"])

        return notes

    def _analyze_characteristics(self, ingredients: List[Dict]) -> Dict:
        """Analyze fragrance characteristics based on ingredients."""
        characteristics = {
            "fragrance_wheel": [],
            "accords": [],
            "personality": ""
        }

        # Determine fragrance wheel position
        families = [ing.get("note_family", "") for ing in ingredients]
        characteristics["fragrance_wheel"] = list(set(families))

        # Determine accords
        if "floral" in families and "woody" in families:
            characteristics["accords"].append("Floral Woody")
        if "citrus" in families and "aromatic" in families:
            characteristics["accords"].append("Citrus Aromatic")
        if "oriental" in families:
            characteristics["accords"].append("Oriental Spicy")

        # Determine personality
        if "floral" in families:
            characteristics["personality"] = "Romantic and feminine"
        elif "woody" in families:
            characteristics["personality"] = "Sophisticated and grounded"
        elif "fresh" in families or "citrus" in families:
            characteristics["personality"] = "Energetic and uplifting"
        else:
            characteristics["personality"] = "Unique and memorable"

        return characteristics

    def _detect_mood(self, description: str) -> str:
        """Detect mood from description."""
        mood_keywords = {
            "romantic": ["love", "romantic", "passion", "heart"],
            "fresh": ["fresh", "clean", "bright", "morning"],
            "mysterious": ["mystery", "night", "dark", "secret"],
            "elegant": ["elegant", "luxury", "sophisticated", "refined"],
            "playful": ["fun", "playful", "happy", "joy"]
        }

        description_lower = description.lower()
        for mood, keywords in mood_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return mood

        return "versatile"

    def _calculate_intensity(self, ingredients: List[Dict]) -> str:
        """Calculate fragrance intensity."""
        total_percentage = sum(ing.get("percentage", 0) for ing in ingredients)

        if total_percentage < 70:
            return "light"
        elif total_percentage < 90:
            return "moderate"
        else:
            return "intense"

    def _estimate_longevity(self, ingredients: List[Dict]) -> str:
        """Estimate fragrance longevity."""
        base_notes = [ing for ing in ingredients if ing.get("type") == "base"]

        if len(base_notes) > 0 and base_notes[0].get("percentage", 0) > 35:
            return "8-12 hours"
        else:
            return "4-6 hours"

    def _estimate_sillage(self, ingredients: List[Dict]) -> str:
        """Estimate fragrance sillage (projection)."""
        intensity = self._calculate_intensity(ingredients)

        if intensity == "intense":
            return "strong"
        elif intensity == "moderate":
            return "moderate"
        else:
            return "intimate"

    def _generate_id(self, description: str) -> str:
        """Generate unique recipe ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        hash_part = hashlib.md5(description.encode()).hexdigest()[:6]
        return f"recipe_{timestamp}_{hash_part}"

    def chat_response(self, message: str) -> Dict:
        """Generate AI chat response for fragrance queries."""

        # Analyze message intent
        intent = self._analyze_intent(message)

        if intent == "create_fragrance":
            recipe = self.generate_recipe(message)
            response = f"I've created a beautiful fragrance based on your description. {recipe['name']} features {recipe['notes']['top'][0]} top notes, {recipe['notes']['heart'][0]} heart notes, and {recipe['notes']['base'][0]} base notes."
            return {"response": response, "recipe": recipe}

        elif intent == "fragrance_info":
            response = self._provide_fragrance_info(message)
            return {"response": response, "recipe": None}

        else:
            response = "I'm here to help you create your perfect fragrance. Tell me about the scent you imagine - the mood, the memories, or the feelings you want to capture."
            return {"response": response, "recipe": None}

    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message."""
        create_keywords = ["create", "make", "design", "want", "need", "looking for"]
        info_keywords = ["what", "how", "tell me", "explain", "difference"]

        message_lower = message.lower()

        if any(keyword in message_lower for keyword in create_keywords):
            return "create_fragrance"
        elif any(keyword in message_lower for keyword in info_keywords):
            return "fragrance_info"
        else:
            return "general"

    def _provide_fragrance_info(self, message: str) -> str:
        """Provide information about fragrances."""
        info_database = {
            "notes": "Fragrances are composed of three layers: top notes (first impression, 5-15 minutes), heart notes (main body, 2-4 hours), and base notes (foundation, 4+ hours).",
            "families": "Main fragrance families include Floral, Oriental, Woody, and Fresh. Each has unique characteristics and emotional associations.",
            "concentration": "Perfume concentrations vary: Parfum (20-30%), Eau de Parfum (15-20%), Eau de Toilette (5-15%), and Eau de Cologne (2-4%).",
            "default": "Fragrance is an art form that combines science and creativity. Each scent tells a unique story through carefully selected ingredients."
        }

        for key, info in info_database.items():
            if key in message.lower():
                return info

        return info_database["default"]


# Global AI instance
ai_service = FragranceAI()