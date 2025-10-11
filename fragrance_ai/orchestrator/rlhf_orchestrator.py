# fragrance_ai/orchestrator/rlhf_orchestrator.py
"""
RLHF Orchestrator - Complete integration with living scent system
Handles generate_variations -> user_feedback -> policy_update flow
"""

import json
import torch
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fragrance_ai.training.rlhf_complete import RLHFEngine
from fragrance_ai.schemas.domain_models import (
    OlfactoryDNA, ScentPhenotype, CreativeBrief, UserChoice
)
from fragrance_ai.data.dataset_manager import DatasetManager

logger = logging.getLogger(__name__)


class RLHFOrchestrator:
    """
    Orchestrator for RLHF-based fragrance evolution
    Manages the complete feedback loop
    """

    def __init__(
        self,
        state_dim: int = 20,
        action_dim: int = 12,
        algorithm: str = "PPO",
        dataset_manager: Optional[DatasetManager] = None,
        **rlhf_kwargs
    ):
        """
        Initialize RLHF orchestrator

        Args:
            state_dim: Dimension of state representation
            action_dim: Number of possible variation actions
            algorithm: "REINFORCE" or "PPO"
            dataset_manager: Optional dataset manager for logging
            **rlhf_kwargs: Additional RLHF parameters
        """
        # Initialize RLHF engine
        self.rl_engine = RLHFEngine(
            state_dim=state_dim,
            action_dim=action_dim,
            algorithm=algorithm,
            **rlhf_kwargs
        )

        # Dataset management
        self.dataset_manager = dataset_manager or DatasetManager()

        # Session management
        self.active_sessions = {}  # user_id -> session_data

        # Action mapping
        self.action_names = [
            "amplify_base", "reduce_base", "amplify_heart", "reduce_heart",
            "amplify_top", "reduce_top", "add_warmth", "add_freshness",
            "increase_complexity", "simplify", "boost_longevity", "lighten_sillage"
        ]

        logger.info(f"Initialized RLHF orchestrator with {algorithm}")

    def encode_state(
        self,
        dna: OlfactoryDNA,
        brief: CreativeBrief
    ) -> torch.Tensor:
        """
        Encode DNA and brief into state vector

        Args:
            dna: Current fragrance DNA
            brief: Creative brief with requirements

        Returns:
            State tensor
        """
        state_vector = []

        # DNA features
        if dna.category_balance:
            state_vector.extend([
                dna.category_balance.get('top', 0) / 100,
                dna.category_balance.get('heart', 0) / 100,
                dna.category_balance.get('base', 0) / 100
            ])

        # Add complexity score
        state_vector.append(dna.complexity_score or 0.5)

        # Brief features
        state_vector.extend([
            brief.desired_intensity,
            brief.masculinity,
            brief.complexity,
            brief.longevity,
            brief.sillage,
            brief.warmth,
            brief.freshness,
            brief.sweetness
        ])

        # Pad to state dimension
        while len(state_vector) < self.rl_engine.agent.state_dim:
            state_vector.append(0.0)

        return torch.tensor(state_vector[:self.rl_engine.agent.state_dim]).unsqueeze(0).float()

    def apply_action(
        self,
        dna: OlfactoryDNA,
        action_idx: int
    ) -> ScentPhenotype:
        """
        Apply variation action to DNA

        Args:
            dna: Original DNA
            action_idx: Action index to apply

        Returns:
            New phenotype with variation
        """
        action_name = self.action_names[action_idx % len(self.action_names)]

        # Copy ingredients for modification
        adjusted_ingredients = [ing.model_copy() for ing in dna.ingredients]

        # Apply action-specific modifications
        if "amplify_base" in action_name:
            for ing in adjusted_ingredients:
                if ing.category.value == "base":
                    ing.concentration *= 1.2
        elif "reduce_base" in action_name:
            for ing in adjusted_ingredients:
                if ing.category.value == "base":
                    ing.concentration *= 0.8
        elif "amplify_heart" in action_name:
            for ing in adjusted_ingredients:
                if ing.category.value == "heart":
                    ing.concentration *= 1.15
        elif "amplify_top" in action_name:
            for ing in adjusted_ingredients:
                if ing.category.value == "top":
                    ing.concentration *= 1.15
        elif "add_warmth" in action_name:
            # Boost warm notes (vanilla, amber, musk)
            for ing in adjusted_ingredients:
                if any(warm in ing.name.lower() for warm in ['vanilla', 'amber', 'musk']):
                    ing.concentration *= 1.3
        elif "add_freshness" in action_name:
            # Boost fresh notes (citrus, green)
            for ing in adjusted_ingredients:
                if any(fresh in ing.name.lower() for fresh in ['citrus', 'lemon', 'bergamot', 'green']):
                    ing.concentration *= 1.3

        # Normalize to 100%
        total = sum(ing.concentration for ing in adjusted_ingredients)
        if total > 0:
            for ing in adjusted_ingredients:
                ing.concentration = (ing.concentration / total) * 100

        # Create phenotype
        phenotype = ScentPhenotype(
            phenotype_id=f"pheno_{datetime.utcnow().timestamp():.0f}_{action_idx}",
            based_on_dna=dna.dna_id,
            epigenetic_trigger="RLHF variation",
            variation_applied=action_name,
            adjusted_ingredients=adjusted_ingredients,
            adjustment_factor={action_name: 1.0},
            description=f"Variation: {action_name.replace('_', ' ').title()}"
        )

        return phenotype

    def generate_variations(
        self,
        user_id: str,
        dna: OlfactoryDNA,
        brief: CreativeBrief,
        num_options: int = 3
    ) -> Dict[str, Any]:
        """
        Generate variation options using RLHF

        Args:
            user_id: User identifier
            dna: Current DNA
            brief: Creative requirements
            num_options: Number of variations to generate

        Returns:
            Response with variation options
        """
        # Start experiment if new session
        if user_id not in self.active_sessions:
            exp_id = self.dataset_manager.start_experiment(
                user_id=user_id,
                algorithm=self.rl_engine.algorithm,
                hyperparameters={"num_options": num_options}
            )
            self.active_sessions[user_id] = {
                "experiment_id": exp_id,
                "iteration": 0
            }

        session = self.active_sessions[user_id]
        session["iteration"] += 1

        # Encode state
        state = self.encode_state(dna, brief)

        # Generate actions using RL engine
        options = []
        saved_actions = []

        # Get action probabilities
        if hasattr(self.rl_engine.agent, 'policy_net'):
            with torch.no_grad():
                probs = self.rl_engine.agent.policy_net(state)
                dist = torch.distributions.Categorical(probs)

                # Sample actions
                for i in range(num_options):
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                    # Apply action to create phenotype
                    phenotype = self.apply_action(dna, action.item())

                    option = {
                        "id": phenotype.phenotype_id,
                        "phenotype": phenotype,
                        "action": action.item(),
                        "action_name": self.action_names[action.item() % len(self.action_names)],
                        "log_prob": log_prob.item(),
                        "description": phenotype.description
                    }
                    options.append(option)
                    saved_actions.append((action, log_prob))

        # Store in session for later update
        session["last_state"] = state
        session["last_saved_actions"] = saved_actions
        session["last_options"] = options
        session["dna_id"] = dna.dna_id
        session["brief_id"] = brief.brief_id

        # Save to RL engine
        self.rl_engine.agent.last_state = state
        self.rl_engine.agent.last_saved_actions = saved_actions

        # Log generation
        logger.info(json.dumps({
            "event": "variations_generated",
            "user_id": self.dataset_manager.hash_user_id(user_id),
            "experiment_id": session["experiment_id"],
            "iteration": session["iteration"],
            "num_options": len(options),
            "actions": [opt["action_name"] for opt in options]
        }))

        return {
            "status": "success",
            "session_id": session["experiment_id"],
            "iteration": session["iteration"],
            "options": [
                {
                    "id": opt["id"],
                    "action": opt["action_name"],
                    "description": opt["description"],
                    "preview": {
                        "top_notes": [
                            ing.name for ing in opt["phenotype"].adjusted_ingredients
                            if ing.category.value == "top"
                        ],
                        "heart_notes": [
                            ing.name for ing in opt["phenotype"].adjusted_ingredients
                            if ing.category.value == "heart"
                        ],
                        "base_notes": [
                            ing.name for ing in opt["phenotype"].adjusted_ingredients
                            if ing.category.value == "base"
                        ]
                    }
                }
                for opt in options
            ]
        }

    def process_feedback(
        self,
        user_id: str,
        chosen_phenotype_id: str,
        rating: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback and update policy

        Args:
            user_id: User identifier
            chosen_phenotype_id: ID of chosen phenotype
            rating: Optional rating (1-5)

        Returns:
            Update results
        """
        if user_id not in self.active_sessions:
            return {"status": "error", "message": "No active session"}

        session = self.active_sessions[user_id]

        # Update policy with feedback
        update_result = self.rl_engine.update_policy_with_feedback(
            chosen_id=chosen_phenotype_id,
            options=session.get("last_options", []),
            state=session.get("last_state"),
            saved_actions=session.get("last_saved_actions"),
            rating=rating
        )

        # Log interaction
        choice = UserChoice(
            session_id=session["experiment_id"],
            user_id=user_id,
            dna_id=session.get("dna_id", "unknown"),
            phenotype_id=chosen_phenotype_id,
            brief_id=session.get("brief_id", "unknown"),
            chosen_option_id=chosen_phenotype_id,
            presented_options=[opt["id"] for opt in session.get("last_options", [])],
            rating=rating,
            iteration_number=session["iteration"]
        )

        # Convert state to list for storage
        state_vector = session["last_state"].squeeze().tolist() if session.get("last_state") is not None else None

        self.dataset_manager.log_interaction(
            choice=choice,
            state_vector=state_vector,
            reward=update_result.get("reward")
        )

        # Log training step
        metrics = {
            "loss": update_result.get("loss", 0),
            "reward": update_result.get("reward", 0),
            "entropy": update_result.get("entropy", 0),
            "policy_loss": update_result.get("policy_loss", 0),
            "value_loss": update_result.get("value_loss", 0)
        }

        self.dataset_manager.log_training_step(
            experiment_id=session["experiment_id"],
            step=session["iteration"],
            metrics=metrics
        )

        # Structured logging
        logger.info(json.dumps({
            "event": "policy_updated",
            "user_id": self.dataset_manager.hash_user_id(user_id),
            "experiment_id": session["experiment_id"],
            "iteration": session["iteration"],
            "chosen_id": chosen_phenotype_id,
            "rating": rating,
            **metrics
        }))

        return {
            "status": "success",
            "update_result": update_result,
            "session": {
                "experiment_id": session["experiment_id"],
                "iteration": session["iteration"]
            }
        }

    def end_session(self, user_id: str) -> Dict[str, Any]:
        """
        End user session and finalize experiment

        Args:
            user_id: User identifier

        Returns:
            Session summary
        """
        if user_id not in self.active_sessions:
            return {"status": "error", "message": "No active session"}

        session = self.active_sessions[user_id]

        # End experiment
        self.dataset_manager.end_experiment(
            experiment_id=session["experiment_id"],
            status="completed"
        )

        # Get statistics
        stats = self.dataset_manager.calculate_statistics(user_id)

        # Clean up
        del self.active_sessions[user_id]

        return {
            "status": "success",
            "experiment_id": session["experiment_id"],
            "total_iterations": session["iteration"],
            "statistics": stats
        }


# ============================================================================
# Export
# ============================================================================

__all__ = ['RLHFOrchestrator']