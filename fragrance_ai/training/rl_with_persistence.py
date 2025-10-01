"""
Enhanced Reinforcement Learning with Model Persistence
Automatic saving of PolicyNetwork after each update with verification
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata"""
    epoch: int
    total_updates: int
    timestamp: str
    loss: float
    average_reward: float
    learning_rate: float
    file_hash: str
    training_metrics: Dict[str, List[float]] = field(default_factory=dict)


class PolicyNetworkWithPersistence(nn.Module):
    """
    Policy Network with built-in persistence capabilities
    """

    def __init__(self,
                 state_dim: int = 100,
                 hidden_dim: int = 256,
                 num_actions: int = 30,
                 dropout_rate: float = 0.1):
        super(PolicyNetworkWithPersistence, self).__init__()

        # Network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.ln3 = nn.LayerNorm(hidden_dim // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Action and value heads
        self.action_head = nn.Linear(hidden_dim // 2, num_actions)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Metadata
        self.creation_time = datetime.now().isoformat()
        self.total_updates = 0
        self.config = {
            'state_dim': state_dim,
            'hidden_dim': hidden_dim,
            'num_actions': num_actions,
            'dropout_rate': dropout_rate
        }

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        # Ensure batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # Process through layers
        x = self.dropout1(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout2(F.relu(self.ln2(self.fc2(x))))
        x = self.dropout3(F.relu(self.ln3(self.fc3(x))))

        # Generate outputs
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.value_head(x)

        return action_probs, value


class ModelPersistenceManager:
    """
    Manages saving, loading, and versioning of PolicyNetwork models
    """

    def __init__(self,
                 save_dir: str = "models",
                 model_name: str = "policy_network",
                 max_checkpoints: int = 5,
                 auto_save: bool = True):
        """
        Initialize persistence manager

        Args:
            save_dir: Directory to save models
            model_name: Base name for model files
            max_checkpoints: Maximum number of checkpoints to keep
            auto_save: Whether to auto-save after updates
        """
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.auto_save = auto_save

        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.main_model_path = self.save_dir / f"{model_name}.pth"
        self.metadata_path = self.save_dir / f"{model_name}_metadata.json"
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)

        # Track modifications
        self.last_save_time = None
        self.last_file_hash = None
        self.save_count = 0

        logger.info(f"ModelPersistenceManager initialized at {self.save_dir}")

    def save_model(self,
                   model: nn.Module,
                   optimizer: optim.Optimizer,
                   training_metrics: Dict[str, Any],
                   epoch: int = 0,
                   loss: float = 0.0,
                   force_checkpoint: bool = False) -> str:
        """
        Save model with metadata and verification

        Args:
            model: The PolicyNetwork to save
            optimizer: The optimizer state to save
            training_metrics: Training metrics to save
            epoch: Current epoch/episode
            loss: Current loss value
            force_checkpoint: Force creation of a checkpoint

        Returns:
            Path to saved model file
        """
        # Prepare checkpoint data
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': getattr(model, 'config', {}),
            'epoch': epoch,
            'total_updates': getattr(model, 'total_updates', 0),
            'timestamp': datetime.now().isoformat(),
            'training_metrics': training_metrics,
            'loss': loss,
            'learning_rate': optimizer.param_groups[0]['lr']
        }

        # Save main model
        torch.save(checkpoint, self.main_model_path)
        self.save_count += 1

        # Calculate file hash for verification
        file_hash = self._calculate_file_hash(self.main_model_path)

        # Create metadata
        metadata = ModelCheckpoint(
            epoch=epoch,
            total_updates=checkpoint['total_updates'],
            timestamp=checkpoint['timestamp'],
            loss=loss,
            average_reward=training_metrics.get('average_reward', 0.0),
            learning_rate=checkpoint['learning_rate'],
            file_hash=file_hash,
            training_metrics=training_metrics
        )

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Verify save
        self._verify_save(self.main_model_path, file_hash)

        # Create checkpoint if needed
        if force_checkpoint or (self.save_count % 10 == 0):
            checkpoint_path = self._create_checkpoint(checkpoint, metadata)
            self._cleanup_old_checkpoints()
        else:
            checkpoint_path = None

        # Log save details
        logger.info(
            f"Model saved: {self.main_model_path} "
            f"(Hash: {file_hash[:8]}..., "
            f"Updates: {checkpoint['total_updates']}, "
            f"Loss: {loss:.4f})"
        )

        # Update tracking
        self.last_save_time = datetime.now()
        self.last_file_hash = file_hash

        return str(self.main_model_path)

    def load_model(self,
                   model: nn.Module,
                   optimizer: Optional[optim.Optimizer] = None,
                   load_path: Optional[str] = None,
                   device: str = 'cpu') -> Dict[str, Any]:
        """
        Load model from file with verification

        Args:
            model: The PolicyNetwork to load weights into
            optimizer: Optional optimizer to restore state
            load_path: Path to model file (uses default if None)
            device: Device to load model to

        Returns:
            Dictionary with loaded checkpoint data
        """
        if load_path is None:
            load_path = self.main_model_path

        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(f"No saved model found at {load_path}")
            return {}

        # Load checkpoint (weights_only=False for compatibility with saved training metrics)
        checkpoint = torch.load(load_path, map_location=device, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Update model metadata
        if hasattr(model, 'total_updates'):
            model.total_updates = checkpoint.get('total_updates', 0)

        # Load and verify metadata
        metadata = self._load_metadata()

        # Verify file integrity
        current_hash = self._calculate_file_hash(load_path)
        if metadata and metadata.get('file_hash') != current_hash:
            logger.warning("File hash mismatch! Model may have been modified externally.")

        logger.info(
            f"Model loaded from {load_path} "
            f"(Epoch: {checkpoint.get('epoch', 0)}, "
            f"Updates: {checkpoint.get('total_updates', 0)}, "
            f"Loss: {checkpoint.get('loss', 0):.4f})"
        )

        return checkpoint

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for verification"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _verify_save(self, file_path: Path, expected_hash: str):
        """Verify that file was saved correctly"""
        if not file_path.exists():
            raise FileNotFoundError(f"Save verification failed: {file_path} not found")

        actual_hash = self._calculate_file_hash(file_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Save verification failed: Hash mismatch")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 1000:  # Minimum expected size
            raise ValueError(f"Save verification failed: File too small ({file_size} bytes)")

        logger.debug(f"Save verified: {file_path} ({file_size} bytes)")

    def _create_checkpoint(self,
                          checkpoint_data: Dict,
                          metadata: ModelCheckpoint) -> str:
        """Create a checkpoint with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{self.model_name}_checkpoint_{timestamp}.pth"
        checkpoint_path = self.checkpoints_dir / checkpoint_name

        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)

        # Save checkpoint metadata
        meta_path = checkpoint_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        logger.info(f"Checkpoint created: {checkpoint_path}")
        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only max_checkpoints"""
        checkpoints = sorted(
            self.checkpoints_dir.glob(f"{self.model_name}_checkpoint_*.pth"),
            key=lambda p: p.stat().st_mtime
        )

        if len(checkpoints) > self.max_checkpoints:
            for checkpoint in checkpoints[:-self.max_checkpoints]:
                checkpoint.unlink()
                # Also remove metadata
                meta_path = checkpoint.with_suffix('.json')
                if meta_path.exists():
                    meta_path.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")

    def _load_metadata(self) -> Optional[Dict]:
        """Load metadata file"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return None

    def get_modification_info(self) -> Dict[str, Any]:
        """Get information about model modifications"""
        info = {
            'main_model_exists': self.main_model_path.exists(),
            'last_save_time': self.last_save_time.isoformat() if self.last_save_time else None,
            'save_count': self.save_count,
            'last_file_hash': self.last_file_hash[:8] if self.last_file_hash else None
        }

        if self.main_model_path.exists():
            stat = self.main_model_path.stat()
            info['file_size'] = stat.st_size
            info['last_modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        metadata = self._load_metadata()
        if metadata:
            info['total_updates'] = metadata.get('total_updates', 0)
            info['last_loss'] = metadata.get('loss', 0)
            info['last_reward'] = metadata.get('average_reward', 0)

        return info


class RLHFWithPersistence:
    """
    Complete RLHF system with automatic model persistence
    """

    def __init__(self,
                 state_dim: int = 100,
                 hidden_dim: int = 256,
                 num_actions: int = 30,
                 learning_rate: float = 3e-4,
                 save_dir: str = "models",
                 auto_save: bool = True):
        """
        Initialize RLHF system with persistence

        Args:
            state_dim: State vector dimension
            hidden_dim: Hidden layer size
            num_actions: Number of possible actions
            learning_rate: Learning rate for AdamW
            save_dir: Directory for saving models
            auto_save: Auto-save after each update
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize network
        self.policy_network = PolicyNetworkWithPersistence(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Initialize persistence manager
        self.persistence_manager = ModelPersistenceManager(
            save_dir=save_dir,
            model_name="policy_network",
            auto_save=auto_save
        )

        # Training metrics
        self.training_metrics = {
            'losses': [],
            'rewards': [],
            'average_reward': 0.0,
            'total_episodes': 0
        }

        # Episode buffers
        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_values = []

        # Load existing model if available
        self.load_model()

        logger.info(f"RLHFWithPersistence initialized on {self.device}")

    def update_policy_with_feedback(self,
                                   log_probs: List[torch.Tensor],
                                   rewards: List[float],
                                   values: List[torch.Tensor],
                                   gamma: float = 0.99) -> float:
        """
        Update policy using REINFORCE and automatically save

        Args:
            log_probs: Log probabilities of selected actions
            rewards: Rewards received
            values: Value estimates
            gamma: Discount factor

        Returns:
            Loss value
        """
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate loss
        policy_loss = []
        value_loss = []

        for log_prob, value, G in zip(log_probs, values, returns):
            advantage = G - value.detach()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.mse_loss(value.squeeze(), G))

        # Total loss
        total_loss = torch.stack(policy_loss).mean()
        if value_loss:
            total_loss += 0.5 * torch.stack(value_loss).mean()

        # Perform gradient update
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.optimizer.step()

        # Update metrics
        self.policy_network.total_updates += 1
        self.training_metrics['losses'].append(total_loss.item())
        self.training_metrics['rewards'].extend(rewards)
        self.training_metrics['average_reward'] = np.mean(rewards)
        self.training_metrics['total_episodes'] += 1

        # AUTO-SAVE MODEL AFTER UPDATE
        if self.persistence_manager.auto_save:
            save_path = self.persistence_manager.save_model(
                model=self.policy_network,
                optimizer=self.optimizer,
                training_metrics=self.training_metrics,
                epoch=self.training_metrics['total_episodes'],
                loss=total_loss.item(),
                force_checkpoint=(self.policy_network.total_updates % 50 == 0)
            )

            # Verify file was modified
            self._verify_file_updated(save_path)

        logger.info(
            f"Policy updated (Update #{self.policy_network.total_updates}): "
            f"Loss={total_loss.item():.4f}, "
            f"Avg Reward={np.mean(rewards):.3f}"
        )

        return total_loss.item()

    def _verify_file_updated(self, file_path: str):
        """Verify that the file was actually updated"""
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Verification failed: {file_path} does not exist!")
            return

        # Check modification time
        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
        time_diff = (datetime.now() - mod_time).total_seconds()

        if time_diff > 5:  # More than 5 seconds old
            logger.warning(f"File may not have been updated: Last modified {time_diff:.1f} seconds ago")
        else:
            logger.info(f"File update verified: Modified {time_diff:.1f} seconds ago")

    def save_model(self, force_checkpoint: bool = False) -> str:
        """Manually save the model"""
        return self.persistence_manager.save_model(
            model=self.policy_network,
            optimizer=self.optimizer,
            training_metrics=self.training_metrics,
            epoch=self.training_metrics['total_episodes'],
            loss=self.training_metrics['losses'][-1] if self.training_metrics['losses'] else 0,
            force_checkpoint=force_checkpoint
        )

    def load_model(self, load_path: Optional[str] = None):
        """Load model from file"""
        checkpoint = self.persistence_manager.load_model(
            model=self.policy_network,
            optimizer=self.optimizer,
            load_path=load_path,
            device=self.device
        )

        # Restore training metrics if available
        if 'training_metrics' in checkpoint:
            self.training_metrics.update(checkpoint['training_metrics'])

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            'total_updates': self.policy_network.total_updates,
            'total_episodes': self.training_metrics['total_episodes'],
            'average_reward': self.training_metrics['average_reward'],
            'last_loss': self.training_metrics['losses'][-1] if self.training_metrics['losses'] else None,
            'persistence_info': self.persistence_manager.get_modification_info()
        }


def demonstration_with_feedback():
    """
    Demonstration of model persistence with user feedback
    """
    print("\n" + "="*60)
    print("RLHF Model Persistence Demonstration")
    print("="*60)

    # Initialize system
    rlhf_system = RLHFWithPersistence(
        state_dim=100,
        hidden_dim=256,
        num_actions=30,
        save_dir="models",
        auto_save=True  # Enable auto-save
    )

    print("\n1. Initial Model State:")
    print("-" * 40)
    info = rlhf_system.get_model_info()
    print(f"Total Updates: {info['total_updates']}")
    print(f"File Exists: {info['persistence_info']['main_model_exists']}")
    print(f"Last Save: {info['persistence_info']['last_save_time']}")

    # Simulate multiple feedback sessions
    print("\n2. Simulating User Feedback Sessions:")
    print("-" * 40)

    for session in range(3):
        print(f"\nSession {session + 1}:")

        # Simulate collecting feedback
        num_actions = 5
        log_probs = []
        rewards = []
        values = []

        for _ in range(num_actions):
            # Simulate action selection
            state = torch.randn(100).to(rlhf_system.device)
            action_probs, value = rlhf_system.policy_network(state)

            # Simulate sampling action
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Simulate user feedback
            reward = np.random.uniform(-1, 1)

            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)

        # Update policy with feedback (AUTOMATICALLY SAVES)
        loss = rlhf_system.update_policy_with_feedback(
            log_probs=log_probs,
            rewards=rewards,
            values=values
        )

        # Check file modification
        info = rlhf_system.get_model_info()
        print(f"  Loss: {loss:.4f}")
        print(f"  Total Updates: {info['total_updates']}")
        print(f"  Last Modified: {info['persistence_info']['last_modified']}")
        print(f"  File Hash: {info['persistence_info']['last_file_hash']}...")

    # Verify persistence
    print("\n3. Verifying Model Persistence:")
    print("-" * 40)

    # Create new system and load saved model
    new_system = RLHFWithPersistence(
        state_dim=100,
        hidden_dim=256,
        num_actions=30,
        save_dir="models",
        auto_save=True
    )

    new_info = new_system.get_model_info()
    print(f"Loaded Model - Total Updates: {new_info['total_updates']}")
    print(f"Loaded Model - Total Episodes: {new_info['total_episodes']}")
    print(f"Loaded Model - Average Reward: {new_info['average_reward']:.3f}")

    # Verify weights are identical
    original_state = rlhf_system.policy_network.state_dict()
    loaded_state = new_system.policy_network.state_dict()

    weights_match = all(
        torch.allclose(original_state[key], loaded_state[key])
        for key in original_state.keys()
    )

    print(f"Weights Match: {'YES' if weights_match else 'NO'}")

    # Show file details
    print("\n4. File System Verification:")
    print("-" * 40)
    model_path = Path("models/policy_network.pth")
    if model_path.exists():
        stat = model_path.stat()
        print(f"File Path: {model_path}")
        print(f"File Size: {stat.st_size:,} bytes")
        print(f"Last Modified: {datetime.fromtimestamp(stat.st_mtime)}")

        # Show checkpoints
        checkpoints = list(Path("models/checkpoints").glob("*.pth"))
        print(f"Checkpoints Saved: {len(checkpoints)}")
        for cp in checkpoints[-3:]:  # Show last 3
            print(f"  - {cp.name}")

    print("\n" + "="*60)
    print("Demonstration Complete!")
    print("="*60)

    return rlhf_system


if __name__ == "__main__":
    # Run demonstration
    system = demonstration_with_feedback()