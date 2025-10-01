"""
Enhanced RLHF (Reinforcement Learning from Human Feedback) Implementation
Real working Policy Network with advanced features for fragrance AI
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import json
from collections import deque
import random
from datetime import datetime
import logging
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of fragrance modification actions"""
    AMPLIFY = "amplify"      # Increase note concentration
    REDUCE = "reduce"        # Decrease note concentration
    ADD = "add"              # Add new note
    REMOVE = "remove"        # Remove note
    BALANCE = "balance"      # Adjust ratio between notes
    TRANSFORM = "transform"  # Replace with similar note


@dataclass
class FragranceDNA:
    """Enhanced DNA representation for fragrances"""
    notes: List[Tuple[int, float]]  # [(note_id, percentage), ...]
    emotional_profile: List[float]  # Emotional vector
    fitness_scores: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # stability, suitability, creativity
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """Structured user feedback for RLHF"""
    rating: float  # 0-10 scale
    preferences: Dict[str, float]  # Detailed preferences
    comparison: Optional[int] = None  # Preference between options (-1, 0, 1)
    text_feedback: Optional[str] = None  # Natural language feedback
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: Optional[torch.Tensor] = None


class PolicyNetwork(nn.Module):
    """
    Advanced Policy Network for RLHF

    Architecture:
    - Input: DNA vector + Feedback vector concatenated
    - Hidden: Multi-layer MLP with skip connections and attention
    - Output: Probability distribution over transformation actions
    """

    def __init__(self,
                 dna_dim: int = 75,      # DNA encoding dimension
                 feedback_dim: int = 25,  # Feedback encoding dimension
                 hidden_dim: int = 256,
                 num_actions: int = 60,   # Total possible actions
                 dropout_rate: float = 0.1):
        """
        Initialize Policy Network

        Args:
            dna_dim: Dimension of DNA encoding
            feedback_dim: Dimension of feedback encoding
            hidden_dim: Hidden layer dimension
            num_actions: Number of possible actions
            dropout_rate: Dropout probability
        """
        super(PolicyNetwork, self).__init__()

        self.input_dim = dna_dim + feedback_dim

        # Feature extraction layers
        self.dna_encoder = nn.Sequential(
            nn.Linear(dna_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.feedback_encoder = nn.Sequential(
            nn.Linear(feedback_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Main processing layers with residual connections
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)

        # Attention mechanism for action selection
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=dropout_rate
        )

        # Action output heads
        self.action_logits = nn.Linear(hidden_dim // 2, num_actions)
        self.value_head = nn.Linear(hidden_dim // 2, 1)  # For Actor-Critic

        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/He initialization for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, dna_input: torch.Tensor, feedback_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network

        Args:
            dna_input: DNA encoding tensor [batch_size, dna_dim]
            feedback_input: Feedback encoding tensor [batch_size, feedback_dim]

        Returns:
            action_probs: Probability distribution over actions [batch_size, num_actions]
            value: State value estimate [batch_size, 1]
        """
        # Encode inputs separately
        dna_features = self.dna_encoder(dna_input)
        feedback_features = self.feedback_encoder(feedback_input)

        # Concatenate encoded features
        x = torch.cat([dna_features, feedback_features], dim=-1)

        # Process through main layers with residual connections
        residual = x
        x = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x = x + residual  # Residual connection

        residual = x
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))
        x = x + residual  # Residual connection

        x = self.dropout(F.relu(self.ln3(self.fc3(x))))

        # Self-attention for action relevance
        x = x.unsqueeze(0)  # Add sequence dimension for attention
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(0)  # Remove sequence dimension

        # Generate action probabilities and value estimate
        action_logits = self.action_logits(x)
        action_probs = F.softmax(action_logits, dim=-1)

        value = self.value_head(x)

        return action_probs, value


class RewardModel(nn.Module):
    """
    Learned reward model from human feedback
    Predicts reward based on state-action pairs
    """

    def __init__(self, state_dim: int = 100, action_dim: int = 60, hidden_dim: int = 128):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, 1)  # Single reward value

        self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict reward for state-action pair"""
        x = torch.cat([state, action], dim=-1)

        x = self.dropout(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout(F.relu(self.ln2(self.fc2(x))))

        reward = torch.tanh(self.fc3(x))  # Normalize reward to [-1, 1]

        return reward


class EnhancedRLHFSystem:
    """
    Complete RLHF system with PPO (Proximal Policy Optimization)
    """

    def __init__(self,
                 dna_dim: int = 75,
                 feedback_dim: int = 25,
                 hidden_dim: int = 256,
                 num_actions: int = 60,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        """
        Initialize RLHF system with PPO algorithm

        Args:
            dna_dim: DNA encoding dimension
            feedback_dim: Feedback encoding dimension
            hidden_dim: Hidden layer size
            num_actions: Number of possible actions
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = PolicyNetwork(
            dna_dim=dna_dim,
            feedback_dim=feedback_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        ).to(self.device)

        self.reward_model = RewardModel(
            state_dim=dna_dim + feedback_dim,
            action_dim=num_actions
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.AdamW(
            self.policy_net.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        self.reward_optimizer = optim.AdamW(
            self.reward_model.parameters(),
            lr=learning_rate * 2  # Higher LR for reward model
        )

        # PPO parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.feedback_buffer = deque(maxlen=5000)

        # Action space definition
        self.action_space = self._define_action_space()

        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'reward_loss': [],
            'average_reward': [],
            'entropy': []
        }

    def _define_action_space(self) -> List[Dict[str, Any]]:
        """
        Define comprehensive action space for fragrance modification

        Returns:
            List of action definitions
        """
        actions = []

        # Common fragrance notes
        notes = [
            'Bergamot', 'Lemon', 'Orange', 'Grapefruit', 'Lime',
            'Rose', 'Jasmine', 'Lavender', 'Ylang-ylang', 'Geranium',
            'Sandalwood', 'Cedarwood', 'Patchouli', 'Vetiver', 'Oakmoss',
            'Vanilla', 'Amber', 'Musk', 'Benzoin', 'Tonka'
        ]

        # Generate actions for each note
        for i, note in enumerate(notes[:10]):  # Limit to 10 notes for simplicity
            for action_type in ActionType:
                actions.append({
                    'type': action_type,
                    'note': note,
                    'note_id': i + 1,
                    'description': f"{action_type.value} {note}"
                })

        return actions

    def encode_dna(self, dna: FragranceDNA) -> torch.Tensor:
        """
        Encode DNA into tensor representation

        Args:
            dna: FragranceDNA object

        Returns:
            Encoded DNA tensor
        """
        encoding = np.zeros(75)

        # Encode notes (first 50 dimensions)
        for i, (note_id, percentage) in enumerate(dna.notes[:10]):
            if i < 10:
                encoding[i * 5] = note_id / 20.0  # Normalize note ID
                encoding[i * 5 + 1] = percentage / 100.0  # Normalize percentage
                encoding[i * 5 + 2] = np.log1p(percentage)  # Log scale
                # Volatility and intensity (placeholder)
                encoding[i * 5 + 3] = random.random()
                encoding[i * 5 + 4] = random.random()

        # Encode emotional profile (next 15 dimensions)
        for i, value in enumerate(dna.emotional_profile[:15]):
            encoding[50 + i] = value

        # Encode fitness scores (last 10 dimensions)
        encoding[65:68] = dna.fitness_scores

        # Add noise for exploration
        encoding[68:75] = np.random.randn(7) * 0.01

        return torch.FloatTensor(encoding).to(self.device)

    def encode_feedback(self, feedback: UserFeedback) -> torch.Tensor:
        """
        Encode user feedback into tensor representation

        Args:
            feedback: UserFeedback object

        Returns:
            Encoded feedback tensor
        """
        encoding = np.zeros(25)

        # Encode rating
        encoding[0] = feedback.rating / 10.0

        # Encode preferences
        pref_keys = ['sweetness', 'freshness', 'intensity', 'longevity', 'sillage']
        for i, key in enumerate(pref_keys):
            if key in feedback.preferences:
                encoding[i + 1] = feedback.preferences[key]

        # Encode comparison result
        if feedback.comparison is not None:
            encoding[6] = (feedback.comparison + 1) / 2.0  # Normalize to [0, 1]

        # Text sentiment (placeholder - would use NLP in production)
        encoding[7:10] = np.random.randn(3) * 0.1

        # Temporal features
        hour = feedback.timestamp.hour / 24.0
        day = feedback.timestamp.weekday() / 7.0
        encoding[10] = hour
        encoding[11] = day

        # Padding with noise
        encoding[12:25] = np.random.randn(13) * 0.01

        return torch.FloatTensor(encoding).to(self.device)

    def select_action(self, dna: FragranceDNA, feedback: UserFeedback,
                     epsilon: float = 0.1) -> Tuple[int, Dict[str, Any]]:
        """
        Select action using current policy with epsilon-greedy exploration

        Args:
            dna: Current fragrance DNA
            feedback: User feedback
            epsilon: Exploration rate

        Returns:
            action_idx: Selected action index
            action_info: Dictionary with action details
        """
        # Encode inputs
        dna_tensor = self.encode_dna(dna).unsqueeze(0)
        feedback_tensor = self.encode_feedback(feedback).unsqueeze(0)

        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action_idx = random.randrange(len(self.action_space))
        else:
            with torch.no_grad():
                action_probs, _ = self.policy_net(dna_tensor, feedback_tensor)

                # Sample from probability distribution
                dist = torch.distributions.Categorical(action_probs)
                action_idx = dist.sample().item()

        action_info = self.action_space[action_idx].copy()
        action_info['index'] = action_idx

        return action_idx, action_info

    def apply_action(self, dna: FragranceDNA, action_info: Dict[str, Any]) -> FragranceDNA:
        """
        Apply selected action to DNA to create variation

        Args:
            dna: Original DNA
            action_info: Action details

        Returns:
            Modified DNA
        """
        new_notes = list(dna.notes)
        action_type = action_info['type']
        note_id = action_info['note_id']

        if action_type == ActionType.AMPLIFY:
            # Increase concentration of target note
            for i, (nid, pct) in enumerate(new_notes):
                if nid == note_id:
                    new_notes[i] = (nid, min(pct * 1.5, 30.0))
                    break
            else:
                # Add if not present
                new_notes.append((note_id, 5.0))

        elif action_type == ActionType.REDUCE:
            # Decrease concentration
            for i, (nid, pct) in enumerate(new_notes):
                if nid == note_id:
                    new_notes[i] = (nid, max(pct * 0.5, 0.1))
                    break

        elif action_type == ActionType.ADD:
            # Add new note
            if not any(nid == note_id for nid, _ in new_notes):
                new_notes.append((note_id, random.uniform(2.0, 8.0)))

        elif action_type == ActionType.REMOVE:
            # Remove note
            new_notes = [(nid, pct) for nid, pct in new_notes if nid != note_id]

        elif action_type == ActionType.BALANCE:
            # Rebalance all notes
            total = sum(pct for _, pct in new_notes)
            if total > 0:
                new_notes = [(nid, (pct / total) * 20.0) for nid, pct in new_notes]

        elif action_type == ActionType.TRANSFORM:
            # Replace with similar note
            for i, (nid, pct) in enumerate(new_notes):
                if nid == note_id:
                    new_note_id = (note_id % 10) + 1  # Simple transformation
                    new_notes[i] = (new_note_id, pct)
                    break

        # Create new DNA with modifications
        new_dna = FragranceDNA(
            notes=new_notes[:15],  # Limit to 15 notes
            emotional_profile=dna.emotional_profile,
            fitness_scores=dna.fitness_scores,
            metadata={**dna.metadata, 'last_action': action_info['description']}
        )

        return new_dna

    def compute_gae(self, rewards: List[float], values: List[float],
                   dones: List[bool]) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE)

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of episode termination flags

        Returns:
            advantages: GAE advantages tensor
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(self.device)

    def train_policy(self, batch_size: int = 32, num_epochs: int = 4):
        """
        Train policy using PPO with collected experiences

        Args:
            batch_size: Batch size for training
            num_epochs: Number of epochs per update
        """
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, batch_size)

        # Prepare batch tensors
        states = torch.stack([e.state for e in batch])
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        old_log_probs = torch.stack([e.log_prob for e in batch])

        # Multiple epochs of optimization
        for epoch in range(num_epochs):
            # Forward pass
            # Split states back into DNA and feedback components
            dna_states = states[:, :75]
            feedback_states = states[:, 75:]

            action_probs, values = self.policy_net(dna_states, feedback_states)

            # Calculate current log probabilities
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)

            # Calculate ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs.detach())

            # Calculate advantages (simplified without GAE for this example)
            advantages = rewards - values.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(), rewards)

            # Entropy bonus for exploration
            entropy = dist.entropy().mean()

            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            # Optimization step
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()

            # Log statistics
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy'].append(entropy.item())

    def train_reward_model(self, batch_size: int = 16):
        """
        Train reward model from human feedback comparisons

        Args:
            batch_size: Batch size for training
        """
        if len(self.feedback_buffer) < batch_size:
            return

        # Sample feedback pairs for comparison learning
        batch = random.sample(self.feedback_buffer, batch_size)

        total_loss = 0
        for feedback_pair in batch:
            state1, action1, reward1 = feedback_pair['option1']
            state2, action2, reward2 = feedback_pair['option2']
            preference = feedback_pair['preference']  # -1, 0, or 1

            # Predict rewards
            pred_reward1 = self.reward_model(state1, action1)
            pred_reward2 = self.reward_model(state2, action2)

            # Bradley-Terry model for preference learning
            if preference != 0:
                # One option is preferred
                if preference > 0:
                    # Option 1 preferred
                    loss = -torch.log(torch.sigmoid(pred_reward1 - pred_reward2))
                else:
                    # Option 2 preferred
                    loss = -torch.log(torch.sigmoid(pred_reward2 - pred_reward1))
            else:
                # Equal preference
                loss = F.mse_loss(pred_reward1, pred_reward2)

            total_loss += loss

        # Optimization step
        avg_loss = total_loss / batch_size
        self.reward_optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.max_grad_norm)
        self.reward_optimizer.step()

        self.training_stats['reward_loss'].append(avg_loss.item())

    def collect_human_feedback(self, dna1: FragranceDNA, dna2: FragranceDNA,
                              preference: int, rating1: float, rating2: float) -> None:
        """
        Collect human feedback for reward model training

        Args:
            dna1: First fragrance option
            dna2: Second fragrance option
            preference: User preference (-1, 0, 1)
            rating1: Rating for first option
            rating2: Rating for second option
        """
        # Create feedback entries
        feedback1 = UserFeedback(
            rating=rating1,
            preferences={'overall': rating1 / 10.0},
            comparison=preference
        )

        feedback2 = UserFeedback(
            rating=rating2,
            preferences={'overall': rating2 / 10.0},
            comparison=-preference
        )

        # Encode states
        state1 = torch.cat([
            self.encode_dna(dna1),
            self.encode_feedback(feedback1)
        ])

        state2 = torch.cat([
            self.encode_dna(dna2),
            self.encode_feedback(feedback2)
        ])

        # Store in feedback buffer
        self.feedback_buffer.append({
            'option1': (state1, torch.zeros(60).to(self.device), rating1),
            'option2': (state2, torch.zeros(60).to(self.device), rating2),
            'preference': preference
        })

    def save_model(self, path: str):
        """Save model checkpoints"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'reward_state_dict': self.reward_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoints"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.reward_model.load_state_dict(checkpoint['reward_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        logger.info(f"Model loaded from {path}")


def example_training_loop():
    """Example training loop for RLHF system"""

    # Initialize system
    rlhf_system = EnhancedRLHFSystem()

    # Create sample DNA and feedback
    sample_dna = FragranceDNA(
        notes=[(1, 10.0), (3, 15.0), (5, 8.0), (7, 12.0)],
        emotional_profile=[0.7, 0.3, 0.5, 0.8, 0.6],
        fitness_scores=(0.8, 0.7, 0.9)
    )

    sample_feedback = UserFeedback(
        rating=7.5,
        preferences={
            'sweetness': 0.6,
            'freshness': 0.8,
            'intensity': 0.7,
            'longevity': 0.5,
            'sillage': 0.6
        }
    )

    print("Starting RLHF Training Loop...")
    print("-" * 50)

    # Training episodes
    num_episodes = 100
    for episode in range(num_episodes):

        # Select action
        action_idx, action_info = rlhf_system.select_action(
            sample_dna,
            sample_feedback,
            epsilon=max(0.1, 1.0 - episode / 100)  # Decay exploration
        )

        # Apply action to get new DNA
        new_dna = rlhf_system.apply_action(sample_dna, action_info)

        # Simulate user feedback (in real scenario, this would be actual human feedback)
        simulated_rating = random.uniform(5, 9)
        reward = (simulated_rating - 7.0) / 2.0  # Normalize to [-1, 1]

        # Store experience
        state = torch.cat([
            rlhf_system.encode_dna(sample_dna),
            rlhf_system.encode_feedback(sample_feedback)
        ])

        next_state = torch.cat([
            rlhf_system.encode_dna(new_dna),
            rlhf_system.encode_feedback(sample_feedback)
        ])

        experience = Experience(
            state=state,
            action=action_idx,
            reward=reward,
            next_state=next_state,
            done=False,
            log_prob=torch.tensor(0.0)  # Placeholder
        )

        rlhf_system.replay_buffer.append(experience)

        # Train policy every 10 episodes
        if episode % 10 == 0 and episode > 0:
            rlhf_system.train_policy(batch_size=16, num_epochs=2)

            if episode % 20 == 0:
                avg_reward = np.mean([e.reward for e in list(rlhf_system.replay_buffer)[-100:]])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, "
                      f"Action = {action_info['description']}")

        # Update sample DNA for next iteration
        sample_dna = new_dna

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)

    # Save trained model
    rlhf_system.save_model("rlhf_model_checkpoint.pth")

    return rlhf_system


if __name__ == "__main__":
    # Run example training
    trained_system = example_training_loop()

    # Display training statistics
    print("\nTraining Statistics:")
    for key, values in trained_system.training_stats.items():
        if values:
            print(f"  {key}: Mean = {np.mean(values[-100:]):.4f}")