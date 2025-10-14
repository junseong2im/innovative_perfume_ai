"""
LLM Worker - Ensemble Inference Worker
Handles LLM inference requests from queue
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


class LLMWorker:
    """LLM Ensemble Inference Worker"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.from_url(self.redis_url)

        # Worker configuration
        self.worker_id = f"llm-worker-{os.getpid()}"
        self.queue_name = "llm_inference_queue"
        self.result_prefix = "llm_result:"

        # LLM models (lazy loaded)
        self.models = {}

        logger.info(f"LLM Worker initialized: {self.worker_id}")
        logger.info(f"Redis URL: {self.redis_url}")
        logger.info(f"Queue: {self.queue_name}")

    def load_models(self):
        """Load LLM models (Qwen, Mistral, Llama)"""
        logger.info("Loading LLM models...")

        try:
            # Import LLM modules
            from fragrance_ai.llm import get_llm_ensemble

            # Get ensemble
            self.ensemble = get_llm_ensemble()

            logger.info("LLM models loaded successfully")
            logger.info(f"Available models: qwen, mistral, llama")

        except Exception as e:
            logger.error(f"Failed to load LLM models: {e}")
            # Use mock for testing
            logger.warning("Using mock LLM for testing")
            self.ensemble = None

    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process LLM inference request

        Args:
            request: {
                'task_id': str,
                'user_text': str,
                'mode': 'fast' | 'balanced' | 'creative',
                'use_cache': bool
            }

        Returns:
            result: {
                'task_id': str,
                'brief': CreativeBrief dict,
                'processing_time': float,
                'model_used': str
            }
        """
        task_id = request.get('task_id')
        user_text = request.get('user_text')
        mode = request.get('mode', 'balanced')
        use_cache = request.get('use_cache', True)

        logger.info(f"Processing task {task_id}: mode={mode}, text_len={len(user_text)}")

        start_time = time.time()

        try:
            if self.ensemble:
                # Real inference
                from fragrance_ai.llm import build_brief

                brief = build_brief(
                    user_text=user_text,
                    mode=mode,
                    use_cache=use_cache
                )

                result = {
                    'task_id': task_id,
                    'brief': brief.model_dump() if hasattr(brief, 'model_dump') else brief.__dict__,
                    'processing_time': time.time() - start_time,
                    'model_used': mode,
                    'success': True
                }

            else:
                # Mock inference
                result = {
                    'task_id': task_id,
                    'brief': {
                        'style': 'fresh',
                        'intensity': 0.7,
                        'complexity': 0.5,
                        'notes_preference': {
                            'citrus': 0.8,
                            'floral': 0.3,
                            'woody': 0.4
                        }
                    },
                    'processing_time': time.time() - start_time,
                    'model_used': 'mock',
                    'success': True
                }

            logger.info(f"Task {task_id} completed in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return {
                'task_id': task_id,
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }

    def run(self):
        """Main worker loop"""
        logger.info(f"Worker {self.worker_id} starting...")

        # Load models
        self.load_models()

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
                    3600,  # Expire after 1 hour
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
    worker = LLMWorker()
    worker.run()


if __name__ == "__main__":
    main()
