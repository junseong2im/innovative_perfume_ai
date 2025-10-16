"""
Trained Fragrance AI Model Loader
Loads and uses the real trained Transformer model for recipe generation
"""

import torch
import torch.nn as nn
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class FragranceTransformer(nn.Module):
    """Trained fragrance generation model (Transformer architecture)"""

    def __init__(self, num_ingredients=12, hidden_dim=512, num_layers=6, num_heads=8):
        super().__init__()

        self.num_ingredients = num_ingredients
        self.embedding = nn.Embedding(num_ingredients, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_ingredients),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Forward pass
        x: (batch, seq_len) - ingredient indices
        returns: (batch, num_ingredients) - probability distribution
        """
        embedded = self.embedding(x)  # (batch, seq_len, hidden_dim)
        transformed = self.transformer(embedded)  # (batch, seq_len, hidden_dim)
        output = self.output_layer(transformed[:, -1, :])  # (batch, num_ingredients)
        return output


class TrainedFragranceAI:
    """
    Wrapper for the trained Fragrance Transformer model
    Provides high-level API for recipe generation
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        """
        Initialize the trained model

        Args:
            model_path: Path to the .pth checkpoint (default: auto-detect latest)
            device: Device to run on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.model = None
        self.ingredients = []
        self.ingredient_map = {}

        # Auto-detect model path if not provided
        if model_path is None:
            model_path = self._find_latest_model()

        self.model_path = model_path

        # Load ingredient database
        self._load_ingredients()

        # Load trained model
        self._load_model()

        logger.info(f"Loaded trained Fragrance AI model from {model_path}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        logger.info(f"Working with {len(self.ingredients)} ingredients")

    def _find_latest_model(self) -> str:
        """Find the latest trained model file"""
        project_root = Path(__file__).parent.parent.parent

        # Look for fragrance_transformer_*.pth files
        model_files = list(project_root.glob("fragrance_transformer_*.pth"))

        if not model_files:
            raise FileNotFoundError("No trained model found. Please run train_real_fragrance.py first")

        # Get the latest one by modification time
        latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
        return str(latest_model)

    def _load_ingredients(self):
        """Load ingredient database"""
        db_path = Path(__file__).parent.parent.parent / "data" / "fragrance_stable.db"

        if not db_path.exists():
            logger.warning(f"Ingredient database not found at {db_path}")
            # Use fallback ingredients
            self.ingredients = [
                (0, 'Bergamot', 'top', 0.9, 85, 2.0, 0.8),
                (1, 'Lemon', 'top', 0.95, 65, 3.0, 0.9),
                (2, 'Orange', 'top', 0.85, 50, 2.5, 0.7),
                (3, 'Grapefruit', 'top', 0.88, 70, 2.0, 0.75),
                (4, 'Rose', 'heart', 0.5, 5000, 0.5, 0.95),
                (5, 'Jasmine', 'heart', 0.45, 8000, 0.4, 0.98),
                (6, 'Lavender', 'heart', 0.6, 120, 5.0, 0.85),
                (7, 'Geranium', 'heart', 0.55, 180, 3.0, 0.8),
                (8, 'Sandalwood', 'base', 0.2, 2500, 2.0, 0.9),
                (9, 'Vanilla', 'base', 0.15, 600, 3.0, 0.92),
                (10, 'Musk', 'base', 0.1, 3000, 1.5, 0.88),
                (11, 'Amber', 'base', 0.18, 800, 2.5, 0.87),
            ]
        else:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ingredients")
            self.ingredients = cursor.fetchall()
            conn.close()

        # Create ingredient mapping
        self.ingredient_map = {i: ing for i, ing in enumerate(self.ingredients)}

    def _load_model(self):
        """Load the trained model checkpoint"""
        logger.info(f"Loading model from {self.model_path}")

        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Create model with correct architecture
        self.model = FragranceTransformer(
            num_ingredients=len(self.ingredients),
            hidden_dim=512,
            num_layers=6,
            num_heads=8
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"Model loaded successfully (trained for {checkpoint['epoch']} epochs)")

    def generate_recipe(
        self,
        brief: Dict[str, Any],
        num_ingredients: int = 8,
        temperature: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate a fragrance recipe using the trained AI model

        Args:
            brief: Creative brief with style, intensity, etc.
            num_ingredients: Number of ingredients to generate
            temperature: Sampling temperature (higher = more creative)

        Returns:
            Dict with recipe and metadata
        """
        logger.info(f"[TRAINED AI] Generating recipe with trained Transformer model")
        logger.info(f"[TRAINED AI] Brief: {brief}")

        # Start with base notes based on brief
        style = brief.get("style", "fresh")

        # Initial seed ingredients based on style
        if style == "fresh" or "citrus" in str(brief).lower():
            # Start with citrus top notes
            seed_indices = [0, 1]  # Bergamot, Lemon
        elif style == "floral":
            seed_indices = [4, 5]  # Rose, Jasmine
        elif style == "woody":
            seed_indices = [8, 10]  # Sandalwood, Musk
        else:
            seed_indices = [0, 4]  # Bergamot, Rose (balanced)

        # Generate recipe using trained model
        recipe_indices = []
        current_sequence = seed_indices.copy()

        with torch.no_grad():
            for _ in range(num_ingredients - len(seed_indices)):
                # Convert to tensor
                seq_tensor = torch.tensor([current_sequence], dtype=torch.long).to(self.device)

                # Get prediction from model
                probs = self.model(seq_tensor)[0]  # (num_ingredients,)

                # Apply temperature
                if temperature > 0:
                    probs = torch.pow(probs, 1.0 / temperature)
                    probs = probs / probs.sum()

                # Sample from distribution
                next_idx = torch.multinomial(probs, 1).item()

                # Avoid repeats
                if next_idx not in recipe_indices:
                    recipe_indices.append(next_idx)
                    current_sequence.append(next_idx)

                    # Keep sequence length manageable
                    if len(current_sequence) > 5:
                        current_sequence = current_sequence[-5:]

        # Combine seed + generated
        final_indices = seed_indices + recipe_indices

        # Calculate concentrations using AI-based weighting
        concentrations = self._calculate_concentrations(final_indices, brief)

        # Build recipe
        recipe_ingredients = []
        total_cost = 0.0

        for idx, concentration in zip(final_indices, concentrations):
            ing_data = self.ingredient_map[idx]

            ingredient = {
                "name": ing_data[1],  # name
                "concentration": round(concentration, 2),
                "category": ing_data[2],  # category (top/heart/base)
                "cost_per_kg": ing_data[4] if len(ing_data) > 4 else 0,  # price_per_kg
                "ifra_limit": ing_data[5] if len(ing_data) > 5 else None,
                "odor_strength": ing_data[6] if len(ing_data) > 6 else 0.5,
            }

            recipe_ingredients.append(ingredient)
            total_cost += (concentration / 100) * ingredient["cost_per_kg"]

        logger.info(f"[TRAINED AI] Generated recipe with {len(recipe_ingredients)} ingredients")

        return {
            "ingredients": recipe_ingredients,
            "total_cost_per_kg": round(total_cost, 2),
            "ai_method": "trained_transformer",
            "model_path": self.model_path,
            "temperature": temperature
        }

    def _calculate_concentrations(
        self,
        ingredient_indices: List[int],
        brief: Dict[str, Any]
    ) -> List[float]:
        """
        Calculate ingredient concentrations based on categories and brief
        Uses perfume pyramid rules (30% top, 40% heart, 30% base)
        """
        categories = [self.ingredient_map[idx][2] for idx in ingredient_indices]

        # Count ingredients by category
        top_count = categories.count('top')
        heart_count = categories.count('heart')
        base_count = categories.count('base')

        # Target percentages (adjustable based on brief intensity)
        intensity = brief.get("intensity", 0.7)

        top_target = 30.0 * (1.0 + (1.0 - intensity) * 0.2)  # More top if lighter
        heart_target = 40.0
        base_target = 100.0 - top_target - heart_target

        # Distribute concentrations
        concentrations = []

        for category in categories:
            if category == 'top':
                conc = top_target / max(top_count, 1)
            elif category == 'heart':
                conc = heart_target / max(heart_count, 1)
            else:  # base
                conc = base_target / max(base_count, 1)

            concentrations.append(conc)

        # Normalize to 100%
        total = sum(concentrations)
        concentrations = [c * 100.0 / total for c in concentrations]

        return concentrations


# Global singleton
_trained_ai_instance = None


def get_trained_ai(device: str = "cpu") -> TrainedFragranceAI:
    """
    Get or create the global trained AI instance

    Args:
        device: Device to run on ("cpu" or "cuda")

    Returns:
        TrainedFragranceAI instance
    """
    global _trained_ai_instance

    if _trained_ai_instance is None:
        _trained_ai_instance = TrainedFragranceAI(device=device)

    return _trained_ai_instance
