# fragrance_ai/services/evolution_service.py
"""
Evolution Service - Orchestrates RLHF-based fragrance evolution
Manages the complete generate → feedback → update flow
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

import torch

from fragrance_ai.training.rl import create_rl_trainer
from fragrance_ai.schemas.domain_models import (
    OlfactoryDNA, ScentPhenotype, CreativeBrief,
    Ingredient, NoteCategory
)

logger = logging.getLogger(__name__)


class EvolutionService:
    """
    Service for managing fragrance evolution with RLHF
    """

    def __init__(
        self,
        algorithm: str = "PPO",
        state_dim: int = 20,
        action_dim: int = 12,
        **rl_config
    ):
        """
        Initialize evolution service

        Args:
            algorithm: RL algorithm to use ("REINFORCE" or "PPO")
            state_dim: State representation dimension
            action_dim: Number of possible variations
            **rl_config: Additional RL configuration
        """
        self.algorithm = algorithm
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create RL trainer
        self.trainer = create_rl_trainer(
            algorithm=algorithm,
            state_dim=state_dim,
            action_dim=action_dim,
            **rl_config
        )

        # Session management
        self.sessions = {}  # experiment_id -> session_data

        # Action definitions
        self.action_map = {
            0: ("amplify_base", "Amplify base notes by 20%"),
            1: ("reduce_base", "Reduce base notes by 20%"),
            2: ("amplify_heart", "Amplify heart notes by 15%"),
            3: ("reduce_heart", "Reduce heart notes by 15%"),
            4: ("amplify_top", "Amplify top notes by 15%"),
            5: ("reduce_top", "Reduce top notes by 15%"),
            6: ("add_warmth", "Enhance warm notes (vanilla, amber, musk)"),
            7: ("add_freshness", "Enhance fresh notes (citrus, green)"),
            8: ("increase_complexity", "Add complexity with subtle accords"),
            9: ("simplify", "Simplify by focusing on key notes"),
            10: ("boost_longevity", "Increase base note proportion for longevity"),
            11: ("lighten_sillage", "Reduce sillage with lighter molecules")
        }

        logger.info(f"Initialized EvolutionService with {algorithm}")

    def _generate_experiment_id(self, user_id: str) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.utcnow().isoformat()
        hash_input = f"{user_id}_{timestamp}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _encode_state(
        self,
        dna: OlfactoryDNA,
        brief: CreativeBrief
    ) -> torch.Tensor:
        """
        Encode DNA and brief into state vector

        Args:
            dna: Current fragrance DNA
            brief: Creative requirements

        Returns:
            State tensor [1, state_dim]
        """
        state = []

        # DNA composition (3 dims)
        top_conc = sum(ing.concentration for ing in dna.ingredients
                      if ing.category == NoteCategory.TOP)
        heart_conc = sum(ing.concentration for ing in dna.ingredients
                        if ing.category == NoteCategory.HEART)
        base_conc = sum(ing.concentration for ing in dna.ingredients
                       if ing.category == NoteCategory.BASE)

        state.extend([top_conc / 100, heart_conc / 100, base_conc / 100])

        # DNA complexity (1 dim)
        state.append(len(dna.ingredients) / 20)  # Normalize by typical max

        # Brief requirements (8 dims)
        state.extend([
            brief.desired_intensity,
            brief.masculinity,
            brief.complexity,
            brief.longevity,
            brief.sillage,
            brief.warmth,
            brief.freshness,
            brief.sweetness
        ])

        # Pad or truncate to state_dim
        if len(state) < self.state_dim:
            state.extend([0.0] * (self.state_dim - len(state)))
        else:
            state = state[:self.state_dim]

        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def _apply_variation(
        self,
        dna: OlfactoryDNA,
        action_idx: int
    ) -> ScentPhenotype:
        """
        Apply variation action to DNA

        Args:
            dna: Original DNA
            action_idx: Action index

        Returns:
            Modified phenotype
        """
        action_name, description = self.action_map.get(
            action_idx % len(self.action_map),
            ("default", "Default variation")
        )

        # Copy ingredients
        modified_ingredients = [ing.model_copy() for ing in dna.ingredients]

        # Apply action-specific modifications
        if action_name == "amplify_base":
            for ing in modified_ingredients:
                if ing.category == NoteCategory.BASE:
                    ing.concentration *= 1.2

        elif action_name == "reduce_base":
            for ing in modified_ingredients:
                if ing.category == NoteCategory.BASE:
                    ing.concentration *= 0.8

        elif action_name == "amplify_heart":
            for ing in modified_ingredients:
                if ing.category == NoteCategory.HEART:
                    ing.concentration *= 1.15

        elif action_name == "reduce_heart":
            for ing in modified_ingredients:
                if ing.category == NoteCategory.HEART:
                    ing.concentration *= 0.85

        elif action_name == "amplify_top":
            for ing in modified_ingredients:
                if ing.category == NoteCategory.TOP:
                    ing.concentration *= 1.15

        elif action_name == "reduce_top":
            for ing in modified_ingredients:
                if ing.category == NoteCategory.TOP:
                    ing.concentration *= 0.85

        elif action_name == "add_warmth":
            for ing in modified_ingredients:
                warm_notes = ['vanilla', 'amber', 'musk', 'sandalwood', 'tonka']
                if any(note in ing.name.lower() for note in warm_notes):
                    ing.concentration *= 1.3

        elif action_name == "add_freshness":
            for ing in modified_ingredients:
                fresh_notes = ['citrus', 'lemon', 'bergamot', 'green', 'mint']
                if any(note in ing.name.lower() for note in fresh_notes):
                    ing.concentration *= 1.3

        elif action_name == "boost_longevity":
            # Increase base, decrease top
            for ing in modified_ingredients:
                if ing.category == NoteCategory.BASE:
                    ing.concentration *= 1.25
                elif ing.category == NoteCategory.TOP:
                    ing.concentration *= 0.9

        # Normalize to 100%
        total = sum(ing.concentration for ing in modified_ingredients)
        if total > 0:
            for ing in modified_ingredients:
                ing.concentration = (ing.concentration / total) * 100

        # Create phenotype
        phenotype = ScentPhenotype(
            phenotype_id=f"pheno_{datetime.utcnow().timestamp():.0f}_{action_idx}",
            based_on_dna=dna.dna_id,
            epigenetic_trigger="RLHF evolution",
            variation_applied=action_name,
            adjusted_ingredients=modified_ingredients,
            adjustment_factor={action_name: 1.0},
            description=description
        )

        return phenotype

    def generate_options(
        self,
        user_id: str,
        dna: OlfactoryDNA,
        brief: CreativeBrief,
        num_options: int = 3
    ) -> Dict[str, Any]:
        """
        Generate variation options for user

        Args:
            user_id: User identifier
            dna: Current fragrance DNA
            brief: Creative brief
            num_options: Number of options to generate

        Returns:
            Response with experiment_id and options
        """
        # Create experiment ID
        experiment_id = self._generate_experiment_id(user_id)

        # Initialize session
        session = {
            "user_id": user_id,
            "experiment_id": experiment_id,
            "dna_id": dna.dna_id,
            "brief_id": brief.brief_id,
            "iteration": 0,
            "created_at": datetime.utcnow().isoformat()
        }

        # Encode state
        state = self._encode_state(dna, brief)

        # Generate options
        options = []
        log_probs = []

        if self.algorithm == "PPO":
            # For PPO, also get values
            values = []
            for i in range(num_options):
                action, log_prob, value = self.trainer.select_action(state)
                phenotype = self._apply_variation(dna, action)

                options.append({
                    "id": phenotype.phenotype_id,
                    "action_idx": action,
                    "action_name": self.action_map[action % len(self.action_map)][0],
                    "description": phenotype.description,
                    "phenotype": phenotype.model_dump()
                })

                log_probs.append(log_prob)  # Keep the actual tensor
                values.append(value)

            session["values"] = values

        else:  # REINFORCE
            for i in range(num_options):
                action, log_prob = self.trainer.select_action(state)
                phenotype = self._apply_variation(dna, action)

                options.append({
                    "id": phenotype.phenotype_id,
                    "action_idx": action,
                    "action_name": self.action_map[action % len(self.action_map)][0],
                    "description": phenotype.description,
                    "phenotype": phenotype.model_dump()
                })

                log_probs.append(log_prob)  # Keep the actual tensor

        # Store session data
        session["state"] = state
        session["options"] = options
        session["log_probs"] = log_probs

        self.sessions[experiment_id] = session

        # Log event
        logger.info(json.dumps({
            "event": "options_generated",
            "experiment_id": experiment_id,
            "user_id": hashlib.sha256(user_id.encode()).hexdigest()[:8],
            "num_options": len(options),
            "algorithm": self.algorithm,
            "actions": [opt["action_name"] for opt in options]
        }))

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "options": [
                {
                    "id": opt["id"],
                    "action": opt["action_name"],
                    "description": opt["description"]
                }
                for opt in options
            ]
        }

    def process_feedback(
        self,
        experiment_id: str,
        chosen_id: str,
        rating: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback and update policy

        Args:
            experiment_id: Experiment identifier
            chosen_id: ID of chosen option
            rating: Optional rating (1-5)

        Returns:
            Update results
        """
        if experiment_id not in self.sessions:
            return {
                "status": "error",
                "message": "Invalid experiment_id"
            }

        session = self.sessions[experiment_id]

        # Find chosen option index
        chosen_idx = None
        for i, opt in enumerate(session["options"]):
            if opt["id"] == chosen_id:
                chosen_idx = i
                break

        if chosen_idx is None:
            return {
                "status": "error",
                "message": "Invalid chosen_id"
            }

        # Update based on algorithm
        if self.algorithm == "PPO":
            # Store experience in buffer
            reward = (rating - 3) / 2.0 if rating else 1.0

            # Get log_prob and convert if needed
            log_prob_val = session["log_probs"][chosen_idx]
            if isinstance(log_prob_val, torch.Tensor):
                log_prob_val = log_prob_val.item()

            # For PPO, store transition
            self.trainer.store_transition(
                state=session["state"],
                action=session["options"][chosen_idx]["action_idx"],
                log_prob=log_prob_val,
                reward=reward,
                value=session.get("values", [0])[chosen_idx],
                done=True
            )

            # Update policy (may buffer or update immediately)
            metrics = self.trainer.update()

            if not metrics:
                metrics = {
                    "status": "buffering",
                    "buffer_size": len(self.trainer.buffer),
                    "reward": reward
                }

        else:  # REINFORCE
            # Direct update with feedback
            metrics = self.trainer.update_with_feedback(
                log_probs=[session["log_probs"][chosen_idx]],
                rating=rating
            )

        # Update session
        session["iteration"] += 1
        session["last_feedback"] = {
            "chosen_id": chosen_id,
            "rating": rating,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Log event
        logger.info(json.dumps({
            "event": "feedback_processed",
            "experiment_id": experiment_id,
            "iteration": session["iteration"],
            "chosen_action": session["options"][chosen_idx]["action_name"],
            "rating": rating,
            "algorithm": self.algorithm,
            **metrics
        }))

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "iteration": session["iteration"],
            "metrics": metrics
        }

    def get_session_info(self, experiment_id: str) -> Optional[Dict]:
        """Get session information"""
        return self.sessions.get(experiment_id)

    def end_session(self, experiment_id: str) -> Dict[str, Any]:
        """End session and clean up"""
        if experiment_id not in self.sessions:
            return {
                "status": "error",
                "message": "Invalid experiment_id"
            }

        session = self.sessions[experiment_id]
        del self.sessions[experiment_id]

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "total_iterations": session["iteration"],
            "algorithm": self.algorithm
        }


# ============================================================================
# Singleton instance
# ============================================================================

_evolution_service = None


def get_evolution_service(**kwargs) -> EvolutionService:
    """Get or create evolution service singleton"""
    global _evolution_service
    if _evolution_service is None:
        _evolution_service = EvolutionService(**kwargs)
    return _evolution_service