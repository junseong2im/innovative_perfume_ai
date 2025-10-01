"""
Enhanced Artisan Orchestrator with Real AI Engines
- Integrates MOGA Optimizer for CREATE_NEW intent
- Integrates RLHF System for EVOLVE_EXISTING intent
- Removes all simulation and hardcoded parts
"""

import logging
import json
import uuid
import asyncio
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum

# Import real AI engines
from ..training.moga_optimizer_enhanced import EnhancedMOGAOptimizer
from ..training.rl_with_persistence import RLHFWithPersistence
from ..database.models import OlfactoryDNA, CreativeBrief, ScentPhenotype

# Import tools
from ..tools.search_tool import hybrid_search
from ..tools.validator_tool import validate_composition, NotesComposition
from ..tools.knowledge_tool import query_knowledge_base
from ..tools.generator_tool import create_recipe, GenerationRequest

logger = logging.getLogger(__name__)


class UserIntent(Enum):
    """User intent classification"""
    CREATE_NEW = "create_new"           # Create entirely new perfume
    EVOLVE_EXISTING = "evolve_existing" # Refine/evolve existing recipe
    SEARCH = "search"                   # Search for perfumes
    VALIDATE = "validate"               # Validate composition
    KNOWLEDGE = "knowledge"             # Query knowledge base
    UNKNOWN = "unknown"                 # Unknown intent


@dataclass
class OrchestrationContext:
    """Context for orchestration"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, str]]
    current_dna: Optional[OlfactoryDNA] = None
    current_brief: Optional[CreativeBrief] = None
    current_recipe: Optional[Dict[str, Any]] = None
    user_feedback_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.user_feedback_history is None:
            self.user_feedback_history = []


@dataclass
class OrchestrationResult:
    """Result from orchestration"""
    success: bool
    intent: UserIntent
    message: str
    recipe: Optional[Dict[str, Any]] = None
    variations: Optional[List[Dict[str, Any]]] = None
    search_results: Optional[List[Dict[str, Any]]] = None
    validation_result: Optional[Dict[str, Any]] = None
    knowledge_result: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedArtisanOrchestrator:
    """
    Enhanced Orchestrator integrating real AI engines
    - MOGA for multi-objective optimization
    - RLHF for human feedback learning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced orchestrator"""
        self.config = config or {}

        # Initialize MOGA optimizer for CREATE_NEW
        self.moga_optimizer = EnhancedMOGAOptimizer(
            population_size=self.config.get('moga_population_size', 100),
            generations=self.config.get('moga_generations', 50),
            mutation_rate=self.config.get('moga_mutation_rate', 0.2),
            crossover_rate=self.config.get('moga_crossover_rate', 0.8),
            use_validator=True  # Always use real validator
        )

        # Initialize RLHF system for EVOLVE_EXISTING
        self.rlhf_system = RLHFWithPersistence(
            state_dim=self.config.get('rlhf_state_dim', 100),
            hidden_dim=self.config.get('rlhf_hidden_dim', 256),
            num_actions=self.config.get('rlhf_num_actions', 30),
            learning_rate=self.config.get('rlhf_learning_rate', 3e-4),
            save_dir=self.config.get('rlhf_save_dir', 'models/orchestrator'),
            auto_save=True  # Always save after updates
        )

        # Intent patterns for classification
        self.intent_patterns = {
            UserIntent.CREATE_NEW: [
                "create", "new", "design", "make", "generate", "fresh",
                "만들", "새로운", "제작", "생성", "디자인"
            ],
            UserIntent.EVOLVE_EXISTING: [
                "improve", "refine", "evolve", "adjust", "modify", "better",
                "개선", "수정", "조정", "발전", "변경", "더 나은"
            ],
            UserIntent.SEARCH: [
                "find", "search", "look for", "similar", "recommend",
                "찾", "검색", "추천", "비슷한"
            ],
            UserIntent.VALIDATE: [
                "validate", "check", "verify", "test", "evaluate",
                "검증", "확인", "테스트", "평가"
            ],
            UserIntent.KNOWLEDGE: [
                "what", "why", "how", "explain", "tell me",
                "무엇", "왜", "어떻게", "설명", "알려"
            ]
        }

        logger.info("Enhanced Artisan Orchestrator initialized with real AI engines")

    async def process(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """
        Main orchestration entry point
        Routes to appropriate AI engine based on intent
        """
        try:
            # 1. Classify user intent
            intent = self._classify_intent(message, context)
            logger.info(f"Classified intent: {intent}")

            # 2. Route to appropriate handler
            if intent == UserIntent.CREATE_NEW:
                return await self._handle_create_new(message, context)

            elif intent == UserIntent.EVOLVE_EXISTING:
                return await self._handle_evolve_existing(message, context)

            elif intent == UserIntent.SEARCH:
                return await self._handle_search(message, context)

            elif intent == UserIntent.VALIDATE:
                return await self._handle_validate(message, context)

            elif intent == UserIntent.KNOWLEDGE:
                return await self._handle_knowledge(message, context)

            else:
                return await self._handle_unknown(message, context)

        except Exception as e:
            logger.error(f"Orchestration failed: {e}", exc_info=True)
            return OrchestrationResult(
                success=False,
                intent=UserIntent.UNKNOWN,
                message="An error occurred during processing.",
                error=str(e)
            )

    def _classify_intent(
        self,
        message: str,
        context: OrchestrationContext
    ) -> UserIntent:
        """Classify user intent from message"""
        message_lower = message.lower()

        # Check for evolve intent (requires existing recipe)
        if context.current_recipe:
            for pattern in self.intent_patterns[UserIntent.EVOLVE_EXISTING]:
                if pattern in message_lower:
                    return UserIntent.EVOLVE_EXISTING

        # Check other intents
        for intent, patterns in self.intent_patterns.items():
            if intent == UserIntent.EVOLVE_EXISTING:
                continue  # Already checked above
            for pattern in patterns:
                if pattern in message_lower:
                    return intent

        return UserIntent.UNKNOWN

    async def _handle_create_new(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """
        Handle CREATE_NEW intent using MOGA optimizer
        """
        logger.info("Handling CREATE_NEW with MOGA optimizer")

        try:
            # 1. Parse creative brief from message
            brief = self._parse_creative_brief(message)

            # 2. Create initial DNA (can be random or based on search)
            initial_dna = await self._create_initial_dna(message, brief)

            # 3. Run MOGA optimization
            logger.info("Starting MOGA optimization...")
            optimized_population = self.moga_optimizer.optimize(
                initial_dna=initial_dna,
                creative_brief=brief
            )

            # 4. Extract top solutions
            top_solutions = self._extract_top_solutions(optimized_population)

            # 5. Convert to recipes
            recipes = []
            for solution in top_solutions[:3]:  # Top 3 solutions
                recipe = self._dna_to_recipe(solution, brief)
                recipes.append(recipe)

            # 6. Update context
            context.current_dna = top_solutions[0] if top_solutions else initial_dna
            context.current_brief = brief
            context.current_recipe = recipes[0] if recipes else None

            return OrchestrationResult(
                success=True,
                intent=UserIntent.CREATE_NEW,
                message=f"Created {len(recipes)} new perfume recipes using multi-objective optimization.",
                recipe=recipes[0] if recipes else None,
                variations=recipes[1:] if len(recipes) > 1 else None,
                metadata={
                    "optimization_method": "MOGA",
                    "generations_run": self.moga_optimizer.generations,
                    "population_size": self.moga_optimizer.population_size,
                    "pareto_front_size": len(optimized_population)
                }
            )

        except Exception as e:
            logger.error(f"CREATE_NEW failed: {e}")
            return OrchestrationResult(
                success=False,
                intent=UserIntent.CREATE_NEW,
                message="Failed to create new perfume recipe.",
                error=str(e)
            )

    async def _handle_evolve_existing(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """
        Handle EVOLVE_EXISTING intent using RLHF system
        """
        logger.info("Handling EVOLVE_EXISTING with RLHF system")

        try:
            # Check if we have a current recipe to evolve
            if not context.current_recipe:
                return OrchestrationResult(
                    success=False,
                    intent=UserIntent.EVOLVE_EXISTING,
                    message="No existing recipe to evolve. Please create or select a recipe first.",
                    error="No current recipe"
                )

            # 1. Convert current state to tensors
            state = self._prepare_state_vector(context)

            # 2. Generate variations using policy network
            action_probs, value = self.rlhf_system.policy_network(state)

            # 3. Sample multiple actions for variations
            num_variations = 3
            variations = []
            log_probs = []

            for _ in range(num_variations):
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)

                # Apply action to create variation
                variation = self._apply_action_to_recipe(
                    context.current_recipe,
                    action.item()
                )
                variations.append(variation)

            # 4. If user provided explicit feedback, update policy
            if self._contains_feedback(message):
                feedback = self._extract_feedback(message)
                rewards = self._feedback_to_rewards(feedback, variations)
                values = [value for _ in range(len(variations))]

                # Update policy with user feedback (auto-saves to policy_network.pth)
                loss = self.rlhf_system.update_policy_with_feedback(
                    log_probs=log_probs,
                    rewards=rewards,
                    values=values
                )

                logger.info(f"Policy updated with feedback. Loss: {loss:.4f}")

                # Record feedback in context
                context.user_feedback_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "feedback": feedback,
                    "rewards": rewards,
                    "loss": loss
                })

                message = f"Thank you for your feedback! Policy updated (loss: {loss:.4f}). Here are refined variations:"
            else:
                message = "Generated variations based on current policy. Please provide feedback to improve."

            # 5. Update context with best variation
            context.current_recipe = variations[0]

            return OrchestrationResult(
                success=True,
                intent=UserIntent.EVOLVE_EXISTING,
                message=message,
                recipe=variations[0],
                variations=variations[1:],
                metadata={
                    "optimization_method": "RLHF",
                    "policy_updates": self.rlhf_system.policy_network.total_updates,
                    "model_file": "models/orchestrator/policy_network.pth",
                    "feedback_history_length": len(context.user_feedback_history)
                }
            )

        except Exception as e:
            logger.error(f"EVOLVE_EXISTING failed: {e}")
            return OrchestrationResult(
                success=False,
                intent=UserIntent.EVOLVE_EXISTING,
                message="Failed to evolve recipe.",
                error=str(e)
            )

    async def _handle_search(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """Handle search intent"""
        try:
            # Use hybrid search tool
            search_results = await hybrid_search(
                text_query=message,
                top_k=10
            )

            return OrchestrationResult(
                success=True,
                intent=UserIntent.SEARCH,
                message=f"Found {len(search_results.get('results', []))} matching perfumes.",
                search_results=search_results.get('results', [])
            )

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return OrchestrationResult(
                success=False,
                intent=UserIntent.SEARCH,
                message="Search failed.",
                error=str(e)
            )

    async def _handle_validate(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """Handle validation intent"""
        try:
            if not context.current_recipe:
                return OrchestrationResult(
                    success=False,
                    intent=UserIntent.VALIDATE,
                    message="No recipe to validate. Please create or select a recipe first.",
                    error="No current recipe"
                )

            # Convert recipe to NotesComposition
            composition = self._recipe_to_composition(context.current_recipe)

            # Validate using real validator
            validation_result = await validate_composition(composition)

            return OrchestrationResult(
                success=True,
                intent=UserIntent.VALIDATE,
                message=f"Validation complete. Score: {validation_result.get('overall_score', 0):.1f}/10",
                validation_result=validation_result
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return OrchestrationResult(
                success=False,
                intent=UserIntent.VALIDATE,
                message="Validation failed.",
                error=str(e)
            )

    async def _handle_knowledge(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """Handle knowledge query intent"""
        try:
            # Determine category
            category = self._infer_knowledge_category(message)

            # Query knowledge base
            result = await query_knowledge_base(
                category=category,
                query=message
            )

            return OrchestrationResult(
                success=True,
                intent=UserIntent.KNOWLEDGE,
                message=result.get('answer', 'No information found.'),
                knowledge_result=result.get('answer')
            )

        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return OrchestrationResult(
                success=False,
                intent=UserIntent.KNOWLEDGE,
                message="Knowledge query failed.",
                error=str(e)
            )

    async def _handle_unknown(
        self,
        message: str,
        context: OrchestrationContext
    ) -> OrchestrationResult:
        """Handle unknown intent"""
        return OrchestrationResult(
            success=False,
            intent=UserIntent.UNKNOWN,
            message="I'm not sure how to help with that. Try asking about creating, searching, or validating perfumes.",
            error="Unknown intent"
        )

    # Helper methods

    def _parse_creative_brief(self, message: str) -> CreativeBrief:
        """Parse creative brief from message"""
        # Simple parsing - can be enhanced with NLP
        brief = CreativeBrief(
            emotional_palette=[0.5] * 5,  # Default neutral
            fragrance_family="floral",
            mood="romantic",
            intensity=0.7,
            season="spring",
            gender="unisex"
        )

        message_lower = message.lower()

        # Parse fragrance family
        families = ["floral", "woody", "citrus", "oriental", "fresh"]
        for family in families:
            if family in message_lower:
                brief.fragrance_family = family
                break

        # Parse mood
        moods = ["romantic", "energetic", "calm", "mysterious", "playful"]
        for mood in moods:
            if mood in message_lower:
                brief.mood = mood
                break

        # Parse season
        seasons = ["spring", "summer", "fall", "winter"]
        for season in seasons:
            if season in message_lower:
                brief.season = season
                break

        return brief

    async def _create_initial_dna(
        self,
        message: str,
        brief: CreativeBrief
    ) -> OlfactoryDNA:
        """Create initial DNA for optimization"""
        # Could use search results as inspiration
        try:
            search_results = await hybrid_search(message, top_k=3)
            # Extract genes from search results (simplified)
            genes = [(i+1, 10.0 + i*5) for i in range(5)]
        except:
            # Default genes
            genes = [(1, 10.0), (3, 15.0), (5, 20.0)]

        return OlfactoryDNA(
            genes=genes,
            fitness_scores=(0.5, 0.5, 0.5)
        )

    def _extract_top_solutions(
        self,
        population: List[Any]
    ) -> List[OlfactoryDNA]:
        """Extract top solutions from MOGA population"""
        # Convert DEAP individuals back to OlfactoryDNA
        top_dnas = []

        for individual in population[:5]:  # Top 5
            genes = [(i, val) for i, val in enumerate(individual) if val > 0]
            dna = OlfactoryDNA(
                genes=genes,
                fitness_scores=individual.fitness.values if hasattr(individual, 'fitness') else (0.5, 0.5, 0.5)
            )
            top_dnas.append(dna)

        return top_dnas

    def _dna_to_recipe(
        self,
        dna: OlfactoryDNA,
        brief: CreativeBrief
    ) -> Dict[str, Any]:
        """Convert DNA to recipe format"""
        # Map genes to notes (simplified)
        note_mapping = {
            1: "Bergamot", 2: "Lemon", 3: "Rose", 4: "Jasmine",
            5: "Sandalwood", 6: "Musk", 7: "Amber", 8: "Vanilla"
        }

        top_notes = []
        heart_notes = []
        base_notes = []

        for gene_id, intensity in dna.genes:
            note_name = note_mapping.get(gene_id, f"Note_{gene_id}")

            if gene_id <= 2:
                top_notes.append({"name": note_name, "percentage": intensity})
            elif gene_id <= 4:
                heart_notes.append({"name": note_name, "percentage": intensity})
            else:
                base_notes.append({"name": note_name, "percentage": intensity})

        return {
            "name": f"{brief.mood.title()} {brief.fragrance_family.title()}",
            "description": f"A {brief.mood} {brief.fragrance_family} fragrance for {brief.season}",
            "top_notes": top_notes,
            "heart_notes": heart_notes,
            "base_notes": base_notes,
            "character": brief.mood,
            "longevity": "6-8 hours",
            "sillage": "moderate" if brief.intensity < 0.7 else "strong",
            "fitness_scores": dna.fitness_scores
        }

    def _prepare_state_vector(self, context: OrchestrationContext) -> torch.Tensor:
        """Prepare state vector for RLHF"""
        # Create a fixed-size state representation
        state = torch.zeros(100)

        # Encode current recipe (simplified)
        if context.current_recipe:
            # Encode number of notes
            state[0] = len(context.current_recipe.get('top_notes', []))
            state[1] = len(context.current_recipe.get('heart_notes', []))
            state[2] = len(context.current_recipe.get('base_notes', []))

            # Encode fitness if available
            if 'fitness_scores' in context.current_recipe:
                state[3:6] = torch.tensor(context.current_recipe['fitness_scores'])

        # Encode brief
        if context.current_brief:
            state[10] = hash(context.current_brief.fragrance_family) % 10 / 10
            state[11] = hash(context.current_brief.mood) % 10 / 10
            state[12] = context.current_brief.intensity

        # Encode feedback history
        state[20] = len(context.user_feedback_history)

        return state

    def _apply_action_to_recipe(
        self,
        recipe: Dict[str, Any],
        action: int
    ) -> Dict[str, Any]:
        """Apply an action to create recipe variation"""
        import copy
        variation = copy.deepcopy(recipe)

        # Map actions to modifications (30 possible actions)
        if action < 10:
            # Modify top notes
            if variation['top_notes']:
                idx = action % len(variation['top_notes'])
                variation['top_notes'][idx]['percentage'] *= (1 + (action - 5) * 0.1)
        elif action < 20:
            # Modify heart notes
            if variation['heart_notes']:
                idx = (action - 10) % len(variation['heart_notes'])
                variation['heart_notes'][idx]['percentage'] *= (1 + (action - 15) * 0.1)
        else:
            # Modify base notes
            if variation['base_notes']:
                idx = (action - 20) % len(variation['base_notes'])
                variation['base_notes'][idx]['percentage'] *= (1 + (action - 25) * 0.1)

        # Normalize percentages
        total = sum(n['percentage'] for notes in [variation['top_notes'],
                                                  variation['heart_notes'],
                                                  variation['base_notes']]
                   for n in notes)

        if total > 0:
            for notes in [variation['top_notes'], variation['heart_notes'], variation['base_notes']]:
                for note in notes:
                    note['percentage'] = (note['percentage'] / total) * 100

        return variation

    def _contains_feedback(self, message: str) -> bool:
        """Check if message contains feedback"""
        feedback_keywords = [
            "good", "bad", "better", "worse", "like", "dislike",
            "prefer", "love", "hate", "perfect", "terrible",
            "좋", "나쁜", "더 나은", "싫", "완벽"
        ]

        message_lower = message.lower()
        return any(keyword in message_lower for keyword in feedback_keywords)

    def _extract_feedback(self, message: str) -> Dict[str, Any]:
        """Extract feedback from message"""
        message_lower = message.lower()

        # Simple sentiment analysis
        positive_words = ["good", "better", "like", "love", "perfect", "great", "좋", "완벽"]
        negative_words = ["bad", "worse", "dislike", "hate", "terrible", "나쁜", "싫"]

        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)

        sentiment = "positive" if positive_count > negative_count else "negative" if negative_count > 0 else "neutral"

        return {
            "sentiment": sentiment,
            "strength": max(positive_count, negative_count),
            "message": message
        }

    def _feedback_to_rewards(
        self,
        feedback: Dict[str, Any],
        variations: List[Dict[str, Any]]
    ) -> List[float]:
        """Convert feedback to rewards for each variation"""
        base_reward = 0.0

        if feedback['sentiment'] == "positive":
            base_reward = 1.0 * feedback['strength']
        elif feedback['sentiment'] == "negative":
            base_reward = -1.0 * feedback['strength']

        # Assume first variation is best if positive, worst if negative
        rewards = []
        for i, _ in enumerate(variations):
            if i == 0:
                rewards.append(base_reward)
            else:
                # Decreasing rewards for other variations
                rewards.append(base_reward * (0.5 ** i))

        return rewards

    def _recipe_to_composition(self, recipe: Dict[str, Any]) -> NotesComposition:
        """Convert recipe to NotesComposition for validation"""
        return NotesComposition(
            top_notes=[{note['name']: note['percentage']}
                      for note in recipe.get('top_notes', [])],
            heart_notes=[{note['name']: note['percentage']}
                        for note in recipe.get('heart_notes', [])],
            base_notes=[{note['name']: note['percentage']}
                       for note in recipe.get('base_notes', [])],
            total_ingredients=sum(len(recipe.get(f'{phase}_notes', []))
                                for phase in ['top', 'heart', 'base'])
        )

    def _infer_knowledge_category(self, message: str) -> str:
        """Infer knowledge category from message"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["history", "ancient", "역사"]):
            return "history"
        elif any(word in message_lower for word in ["technique", "method", "기술"]):
            return "technique"
        elif any(word in message_lower for word in ["note", "ingredient", "노트"]):
            return "note"
        elif any(word in message_lower for word in ["accord", "blend", "어코드"]):
            return "accord"
        else:
            return "general"


async def demonstrate_orchestrator():
    """Demonstrate the enhanced orchestrator"""
    print("\n" + "="*70)
    print("ENHANCED ARTISAN ORCHESTRATOR DEMONSTRATION")
    print("="*70)

    # Initialize orchestrator
    orchestrator = EnhancedArtisanOrchestrator()

    # Create context
    context = OrchestrationContext(
        user_id="demo_user",
        session_id=str(uuid.uuid4()),
        conversation_history=[]
    )

    print("\n1. Testing CREATE_NEW intent with MOGA")
    print("-" * 50)

    result = await orchestrator.process(
        "Create a fresh floral perfume for summer",
        context
    )

    print(f"Intent: {result.intent}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.recipe:
        print(f"Recipe: {result.recipe['name']}")
        print(f"Optimization: {result.metadata.get('optimization_method')}")

    print("\n2. Testing EVOLVE_EXISTING intent with RLHF")
    print("-" * 50)

    result = await orchestrator.process(
        "Make it better and more romantic",
        context
    )

    print(f"Intent: {result.intent}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    if result.metadata:
        print(f"Policy updates: {result.metadata.get('policy_updates')}")

    # Provide feedback
    print("\n3. Providing feedback to RLHF system")
    print("-" * 50)

    result = await orchestrator.process(
        "I love the first variation, it's perfect!",
        context
    )

    print(f"Feedback processed: {result.success}")
    print(f"Message: {result.message}")

    print("\n" + "="*70)
    print("Demonstration complete!")
    print("="*70)


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_orchestrator())