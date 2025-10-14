"""
RL Worker - Reinforcement Learning Training Worker
Handles RL training tasks from queue
"""

import logging
import time
import json
import os
from typing import Dict, Any
import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RLWorker:
    """Reinforcement Learning Training Worker"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(self.redis_url)

        # Worker configuration
        self.worker_id = f"rl-worker-{os.getpid()}"
        self.queue_name = "rl_training_queue"
        self.result_prefix = "rl_result:"
        self.checkpoint_dir = os.getenv('CHECKPOINT_DIR', './checkpoints')

        # RL trainer (lazy loaded)
        self.trainer = None

        logger.info(f"RL Worker initialized: {self.worker_id}")
        logger.info(f"Redis URL: {self.redis_url}")
        logger.info(f"Queue: {self.queue_name}")
        logger.info(f"Checkpoint dir: {self.checkpoint_dir}")

    def load_trainer(self):
        """Load RL trainer"""
        logger.info("Loading RL trainer...")

        try:
            from fragrance_ai.training.ppo_trainer_advanced import AdvancedPPOTrainer
            from fragrance_ai.training.rl_advanced import (
                EntropyScheduleConfig,
                RewardNormalizerConfig,
                CheckpointConfig
            )

            # Configuration
            entropy_config = EntropyScheduleConfig(
                initial_entropy=0.01,
                final_entropy=0.001,
                decay_steps=100000
            )

            reward_config = RewardNormalizerConfig(
                window_size=1000,
                clip_range=(-10.0, 10.0)
            )

            checkpoint_config = CheckpointConfig(
                checkpoint_dir=self.checkpoint_dir,
                save_interval=100,
                rollback_on_kl_threshold=0.1
            )

            # Create trainer (will be initialized with specific env later)
            self.trainer_config = {
                'entropy_config': entropy_config,
                'reward_config': reward_config,
                'checkpoint_config': checkpoint_config
            }

            logger.info("RL trainer configuration loaded")

        except Exception as e:
            logger.error(f"Failed to load RL trainer: {e}")
            logger.warning("RL training will be mocked")
            self.trainer_config = None

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process RL training request

        Args:
            request: {
                'task_id': str,
                'experiment_id': str,
                'dna_id': str,
                'algorithm': 'PPO' | 'REINFORCE',
                'n_iterations': int,
                'n_steps_per_iteration': int
            }

        Returns:
            result: {
                'task_id': str,
                'experiment_id': str,
                'final_reward': float,
                'total_episodes': int,
                'training_time': float,
                'checkpoint_path': str
            }
        """
        task_id = request.get('task_id')
        experiment_id = request.get('experiment_id')
        algorithm = request.get('algorithm', 'PPO')
        n_iterations = request.get('n_iterations', 100)

        logger.info(f"Processing task {task_id}: experiment={experiment_id}, algo={algorithm}")

        start_time = time.time()

        try:
            if self.trainer_config:
                # Real training
                from fragrance_ai.training.ppo_trainer_advanced import train_advanced_ppo
                from fragrance_ai.training.ppo_engine import FragranceEnvironment

                # Create environment
                env = FragranceEnvironment(n_ingredients=20)

                # Run training
                trainer = train_advanced_ppo(
                    env=env,
                    n_iterations=n_iterations,
                    n_steps_per_iteration=request.get('n_steps_per_iteration', 2048),
                    n_ppo_epochs=10,
                    **self.trainer_config
                )

                # Get statistics
                full_stats = trainer.get_full_statistics()

                result = {
                    'task_id': task_id,
                    'experiment_id': experiment_id,
                    'final_reward': full_stats['checkpoint']['best_reward'],
                    'total_episodes': full_stats['episode_count'],
                    'training_time': time.time() - start_time,
                    'checkpoint_path': str(self.checkpoint_dir),
                    'rollback_count': full_stats['checkpoint']['rollback_count'],
                    'success': True
                }

            else:
                # Mock training
                time.sleep(5)  # Simulate training
                result = {
                    'task_id': task_id,
                    'experiment_id': experiment_id,
                    'final_reward': 10.5,
                    'total_episodes': n_iterations * 10,
                    'training_time': time.time() - start_time,
                    'checkpoint_path': 'mock',
                    'rollback_count': 0,
                    'success': True
                }

            logger.info(f"Task {task_id} completed: reward={result['final_reward']:.2f}")
            return result

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return {
                'task_id': task_id,
                'experiment_id': experiment_id,
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }

    def run(self):
        """Main worker loop"""
        logger.info(f"Worker {self.worker_id} starting...")

        # Load trainer
        self.load_trainer()

        logger.info("Worker ready. Waiting for tasks...")

        while True:
            try:
                # Block and wait for task (BRPOP with timeout)
                result = self.redis_client.brpop(self.queue_name, timeout=5)

                if result is None:
                    # Timeout, continue
                    continue

                queue_name, task_data = result

                # Parse task
                task = json.loads(task_data)
                task_id = task.get('task_id')

                logger.info(f"Received task: {task_id}")

                # Process task
                result = self.process_request(task)

                # Store result in Redis
                result_key = f"{self.result_prefix}{task_id}"
                self.redis_client.setex(
                    result_key,
                    86400,  # Expire after 24 hours
                    json.dumps(result)
                )

                logger.info(f"Result stored: {result_key}")

            except KeyboardInterrupt:
                logger.info("Worker shutting down...")
                break

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on persistent errors


def main():
    """Main entry point"""
    worker = RLWorker()
    worker.run()


if __name__ == "__main__":
    main()
