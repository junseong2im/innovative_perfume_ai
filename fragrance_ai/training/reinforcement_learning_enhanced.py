"""
Enhanced Reinforcement Learning with Real REINFORCE Algorithm
Complete implementation with AdamW optimizer and proper gradient updates
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
from collections import deque
import random
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OlfactoryDNA:
    """Fragrance DNA representation"""
    genes: List[Tuple[int, float]]  # [(note_id, percentage), ...]
    fitness_scores: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CreativeBrief:
    """User's creative requirements"""
    emotional_palette: List[float]  # Emotion vector
    fragrance_family: str
    mood: str
    intensity: float
    season: str
    gender: str


@dataclass
class ScentPhenotype:
    """Fragrance phenotype for user evaluation"""
    dna: OlfactoryDNA
    variation_applied: str
    user_rating: Optional[float] = None
    log_prob: Optional[torch.Tensor] = None  # Store for REINFORCE


class PolicyNetwork(nn.Module):
    """
    Enhanced Policy Network with proper architecture
    Input: DNA vector + Creative Brief vector
    Output: Action probability distribution
    """

    def __init__(self,
                 dna_dim: int = 50,
                 brief_dim: int = 50,
                 hidden_dim: int = 256,
                 num_actions: int = 30,
                 dropout_rate: float = 0.1):
        super(PolicyNetwork, self).__init__()

        input_dim = dna_dim + brief_dim

        # Feature extraction with batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Main network with residual connections
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer for action probabilities
        self.action_head = nn.Linear(hidden_dim // 2, num_actions)

        # Optional: Value head for Actor-Critic (future enhancement)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        Returns both action probabilities and state value
        """
        # Handle batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # Skip batch norm if batch size is 1
        if state.size(0) == 1:
            x = state
        else:
            x = self.input_bn(state)

        # First layer with residual
        identity = x
        x = F.relu(self.fc1(x))
        if x.size(0) > 1:
            x = self.bn1(x)
        x = self.dropout1(x)

        # Second layer
        x = F.relu(self.fc2(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout2(x)

        # Third layer
        x = F.relu(self.fc3(x))
        if x.size(0) > 1:
            x = self.bn3(x)
        x = self.dropout3(x)

        # Output heads
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        value = self.value_head(x)

        return action_probs, value


class EpigeneticVariationAI:
    """
    Enhanced RLHF Engine with proper REINFORCE implementation
    Uses AdamW optimizer and real gradient updates
    """

    def __init__(self,
                 state_dim: int = 100,
                 num_actions: int = 30,
                 learning_rate: float = 3e-4,
                 weight_decay: float = 1e-4,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = None):

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing EpigeneticVariationAI on {self.device}")

        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Initialize policy network
        self.policy_network = PolicyNetwork(
            dna_dim=50,
            brief_dim=50,
            hidden_dim=256,
            num_actions=num_actions
        ).to(self.device)

        # AdamW optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10
        )

        # Episode storage
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_entropy = []

        # Action space definition
        self.action_space = self._define_action_space()

        # Training metrics
        self.training_metrics = {
            'episode_rewards': [],
            'policy_loss': [],
            'value_loss': [],
            'total_loss': [],
            'entropy': [],
            'learning_rate': []
        }

        # Experience replay buffer (optional enhancement)
        self.replay_buffer = deque(maxlen=1000)

    def _define_action_space(self) -> List[str]:
        """Define comprehensive action space"""
        actions = []
        note_names = ['Bergamot', 'Lemon', 'Rose', 'Jasmine', 'Sandalwood',
                     'Cedar', 'Vanilla', 'Musk', 'Amber', 'Patchouli']

        for note in note_names:
            actions.append(f"Amplify_{note}")
            actions.append(f"Reduce_{note}")
            actions.append(f"Add_{note}")

        return actions

    def encode_state(self, dna: OlfactoryDNA, brief: CreativeBrief) -> torch.Tensor:
        """
        Encode DNA and brief into state vector
        Enhanced encoding with better normalization
        """
        # DNA encoding (50 dimensions)
        dna_vector = np.zeros(50)
        for i, (note_id, percentage) in enumerate(dna.genes[:10]):
            if i < 10:
                # Note features
                dna_vector[i*5] = note_id / 10.0  # Normalized note ID
                dna_vector[i*5 + 1] = percentage / 30.0  # Normalized percentage
                dna_vector[i*5 + 2] = np.tanh(percentage / 10.0)  # Tanh normalization

                # Fitness scores
                if dna.fitness_scores:
                    dna_vector[i*5 + 3] = dna.fitness_scores[0] if len(dna.fitness_scores) > 0 else 0
                    dna_vector[i*5 + 4] = dna.fitness_scores[1] if len(dna.fitness_scores) > 1 else 0

        # Creative brief encoding (50 dimensions)
        brief_vector = np.zeros(50)

        # Emotional palette (first 10 dimensions)
        for i, value in enumerate(brief.emotional_palette[:10]):
            brief_vector[i] = value

        # Categorical features (one-hot encoding)
        brief_vector[10] = brief.intensity

        # Family encoding (11-20)
        families = ['floral', 'woody', 'fresh', 'oriental', 'citrus']
        if brief.fragrance_family in families:
            brief_vector[11 + families.index(brief.fragrance_family)] = 1.0

        # Mood encoding (21-30)
        moods = ['romantic', 'energetic', 'calm', 'mysterious', 'fresh']
        if brief.mood in moods:
            brief_vector[21 + moods.index(brief.mood)] = 1.0

        # Season encoding (31-35)
        seasons = ['spring', 'summer', 'fall', 'winter', 'all']
        if brief.season in seasons:
            brief_vector[31 + seasons.index(brief.season)] = 1.0

        # Gender encoding (36-38)
        genders = ['masculine', 'feminine', 'unisex']
        if brief.gender in genders:
            brief_vector[36 + genders.index(brief.gender)] = 1.0

        # Concatenate vectors
        state_vector = np.concatenate([dna_vector, brief_vector])

        # Convert to tensor and move to device
        return torch.FloatTensor(state_vector).to(self.device)

    def sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy network
        Returns action index, log probability, value estimate, and entropy
        """
        # Get action probabilities and value
        action_probs, value = self.policy_network(state)

        # Create categorical distribution
        distribution = torch.distributions.Categorical(action_probs)

        # Sample action
        action = distribution.sample()

        # Calculate log probability
        log_prob = distribution.log_prob(action)

        # Calculate entropy for exploration bonus
        entropy = distribution.entropy()

        return action.item(), log_prob, value, entropy

    def apply_variation(self, dna: OlfactoryDNA, action_idx: int) -> OlfactoryDNA:
        """Apply selected action to create DNA variation"""
        action_name = self.action_space[action_idx]
        action_type, note_name = action_name.split('_', 1)

        # Clone genes
        new_genes = list(dna.genes)

        # Note mapping
        note_mapping = {
            'Bergamot': 1, 'Lemon': 2, 'Rose': 3, 'Jasmine': 4,
            'Sandalwood': 5, 'Cedar': 6, 'Vanilla': 7, 'Musk': 8,
            'Amber': 9, 'Patchouli': 10
        }

        note_id = note_mapping.get(note_name, 1)

        if action_type == "Amplify":
            # Increase note concentration
            modified = False
            for i, (nid, percentage) in enumerate(new_genes):
                if nid == note_id:
                    new_genes[i] = (nid, min(percentage * 1.5, 30.0))
                    modified = True
                    break

            if not modified:
                new_genes.append((note_id, 5.0))

        elif action_type == "Reduce":
            # Decrease note concentration
            for i, (nid, percentage) in enumerate(new_genes):
                if nid == note_id:
                    new_genes[i] = (nid, max(percentage * 0.5, 0.1))
                    break

        elif action_type == "Add":
            # Add new note if not present
            if not any(nid == note_id for nid, _ in new_genes):
                if len(new_genes) < 15:
                    new_genes.append((note_id, random.uniform(2.0, 8.0)))

        # Create new DNA
        return OlfactoryDNA(
            genes=new_genes[:15],  # Limit to 15 notes
            fitness_scores=dna.fitness_scores,
            metadata={**dna.metadata, 'last_action': action_name}
        )

    def generate_variations(self, dna: OlfactoryDNA, brief: CreativeBrief,
                           num_variations: int = 3) -> List[ScentPhenotype]:
        """
        Generate variations for user evaluation
        Store log probabilities for later training
        """
        variations = []
        state = self.encode_state(dna, brief)

        for _ in range(num_variations):
            # Sample action from policy
            action_idx, log_prob, value, entropy = self.sample_action(state)

            # Apply variation
            varied_dna = self.apply_variation(dna, action_idx)

            # Create phenotype with stored log_prob
            phenotype = ScentPhenotype(
                dna=varied_dna,
                variation_applied=self.action_space[action_idx],
                log_prob=log_prob
            )

            variations.append(phenotype)

            # Store for training
            self.episode_log_probs.append(log_prob)
            self.episode_values.append(value)
            self.episode_entropy.append(entropy)

        return variations

    def update_policy_with_feedback(self, variations: List[ScentPhenotype],
                                   selected_idx: int,
                                   ratings: Optional[List[float]] = None):
        """
        Update policy using REINFORCE algorithm with real gradients

        Key Implementation:
        1. Calculate rewards based on user selection
        2. Compute REINFORCE loss: -log(P(action)) * reward
        3. Use AdamW optimizer for gradient update
        """

        # Calculate rewards
        rewards = []
        for i, phenotype in enumerate(variations):
            if i == selected_idx:
                # Positive reward for selected variation
                reward = 1.0
                logger.info(f"User selected: {phenotype.variation_applied}")
            else:
                # Negative reward for unselected variations
                reward = -0.5

            # Optional: Use explicit ratings if provided
            if ratings and i < len(ratings):
                reward = (ratings[i] - 5.0) / 5.0  # Normalize to [-1, 1]

            rewards.append(reward)
            phenotype.user_rating = reward

        self.episode_rewards.extend(rewards)

        # Perform policy update
        loss = self._perform_reinforce_update()

        return loss

    def _perform_reinforce_update(self):
        """
        Perform actual REINFORCE policy update with AdamW optimizer

        Core REINFORCE Algorithm:
        Loss = -Σ[log(P(a_t|s_t)) * G_t]
        where G_t is the discounted return from time t
        """

        if len(self.episode_rewards) == 0:
            return None

        # Calculate discounted returns
        returns = self._calculate_returns()

        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns for stable training
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate REINFORCE loss
        policy_loss = []
        value_loss = []

        for log_prob, value, G in zip(self.episode_log_probs, self.episode_values, returns):
            # REINFORCE loss: -log(P(action)) * advantage
            # Using advantage (G - V) instead of just G for variance reduction
            advantage = G - value.detach()
            policy_loss.append(-log_prob * advantage)

            # Value loss for critic (optional)
            value_loss.append(F.mse_loss(value.squeeze(), G))

        # Calculate entropy bonus for exploration
        entropy_bonus = torch.stack(self.episode_entropy).mean() if self.episode_entropy else 0

        # Combine losses
        policy_loss = torch.stack(policy_loss).mean()
        value_loss = torch.stack(value_loss).mean() if value_loss else torch.tensor(0.0)

        # Total loss with entropy regularization
        total_loss = (
            policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * entropy_bonus
        )

        # Perform gradient update with AdamW
        self.optimizer.zero_grad()  # Clear previous gradients
        total_loss.backward()       # Compute gradients

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            self.max_grad_norm
        )

        self.optimizer.step()        # Update weights

        # Log training metrics
        self.training_metrics['policy_loss'].append(policy_loss.item())
        self.training_metrics['value_loss'].append(value_loss.item() if value_loss != 0 else 0)
        self.training_metrics['total_loss'].append(total_loss.item())
        self.training_metrics['entropy'].append(entropy_bonus.item() if entropy_bonus != 0 else 0)
        self.training_metrics['episode_rewards'].append(np.mean(self.episode_rewards))
        self.training_metrics['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

        logger.info(
            f"Policy Update: "
            f"Loss={total_loss.item():.4f}, "
            f"Policy={policy_loss.item():.4f}, "
            f"Value={value_loss.item() if value_loss != 0 else 0:.4f}, "
            f"Reward={np.mean(self.episode_rewards):.3f}"
        )

        # Update learning rate scheduler
        self.scheduler.step(np.mean(self.episode_rewards))

        # Clear episode buffers
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []
        self.episode_entropy = []

        return total_loss.item()

    def _calculate_returns(self) -> List[float]:
        """Calculate discounted returns for REINFORCE"""
        returns = []
        G = 0

        for reward in reversed(self.episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        return returns

    def train_episode(self, initial_dna: OlfactoryDNA, brief: CreativeBrief,
                     num_rounds: int = 10) -> Dict[str, Any]:
        """
        Train for one complete episode with simulated user feedback
        """
        current_dna = initial_dna
        episode_metrics = {
            'total_reward': 0,
            'losses': [],
            'selected_actions': []
        }

        for round_idx in range(num_rounds):
            # Generate variations
            variations = self.generate_variations(current_dna, brief, num_variations=3)

            # Simulate user selection (in production, this would be real user input)
            selected_idx = self._simulate_user_selection(variations)

            # Update policy with feedback
            loss = self.update_policy_with_feedback(variations, selected_idx)

            if loss is not None:
                episode_metrics['losses'].append(loss)

            episode_metrics['selected_actions'].append(variations[selected_idx].variation_applied)
            episode_metrics['total_reward'] += (1.0 if selected_idx == 0 else -0.5)

            # Update current DNA
            current_dna = variations[selected_idx].dna

        return episode_metrics

    def _simulate_user_selection(self, variations: List[ScentPhenotype]) -> int:
        """Simulate user selection for training (replace with real user input in production)"""
        # Simple heuristic: prefer variations with balanced notes
        scores = []
        for phenotype in variations:
            total_percentage = sum(p for _, p in phenotype.dna.genes)
            balance_score = 1.0 / (1.0 + abs(total_percentage - 20.0))
            scores.append(balance_score)

        # Add some randomness
        scores = [s + random.random() * 0.2 for s in scores]

        return scores.index(max(scores))

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_metrics': self.training_metrics,
            'episode': len(self.training_metrics['episode_rewards'])
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        logger.info(f"Checkpoint loaded from {filepath}")


def train_rlhf_system():
    """Complete training example with real REINFORCE updates"""

    # Set random seed
    set_seed(42)

    # Initialize system
    ai_system = EpigeneticVariationAI(
        learning_rate=3e-4,
        weight_decay=1e-4,
        gamma=0.99,
        entropy_coef=0.01
    )

    # Create sample DNA and brief
    initial_dna = OlfactoryDNA(
        genes=[(1, 10.0), (3, 15.0), (5, 8.0)],
        fitness_scores=(0.7, 0.6, 0.8)
    )

    brief = CreativeBrief(
        emotional_palette=[0.7, 0.3, 0.5, 0.8, 0.6],
        fragrance_family="floral",
        mood="romantic",
        intensity=0.7,
        season="spring",
        gender="feminine"
    )

    print("\n" + "="*60)
    print("REINFORCE Training with AdamW Optimizer")
    print("="*60)

    # Training loop
    num_episodes = 20
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")

        # Train one episode
        metrics = ai_system.train_episode(initial_dna, brief, num_rounds=5)

        print(f"  Total Reward: {metrics['total_reward']:.2f}")
        if metrics['losses']:
            print(f"  Average Loss: {np.mean(metrics['losses']):.4f}")
        print(f"  Selected Actions: {metrics['selected_actions'][:3]}...")

        # Save checkpoint periodically
        if (episode + 1) % 10 == 0:
            ai_system.save_checkpoint(f"rlhf_checkpoint_ep{episode+1}.pth")

    # Display final training metrics
    print("\n" + "="*60)
    print("Training Complete - Final Metrics")
    print("="*60)

    if ai_system.training_metrics['policy_loss']:
        print(f"Final Policy Loss: {ai_system.training_metrics['policy_loss'][-1]:.4f}")
        print(f"Final Learning Rate: {ai_system.training_metrics['learning_rate'][-1]:.6f}")
        print(f"Average Reward (last 10): {np.mean(ai_system.training_metrics['episode_rewards'][-10:]):.3f}")

    return ai_system


if __name__ == "__main__":
    # Run training
    trained_system = train_rlhf_system()