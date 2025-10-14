"""
Qwen RLHF (Reinforcement Learning from Human Feedback) System
Qwen2.5 API를 내부 Fine-tuner와 연결하여 사용자 피드백을 RLHF로 직접 반영
"""

import json
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import redis.asyncio as redis
from loguru import logger


@dataclass
class UserFeedback:
    """사용자 피드백 데이터 구조"""
    recipe_id: str
    user_rating: float  # 1.0 ~ 5.0
    feedback_text: str
    preferred_notes: List[str]
    disliked_aspects: List[str]
    timestamp: str
    user_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserFeedback':
        return cls(**data)


@dataclass
class RLHFSample:
    """RLHF 학습 샘플"""
    prompt: str
    chosen_response: str  # 선호된 레시피
    rejected_response: str  # 거부된 레시피
    reward_score: float


class RewardModel(nn.Module):
    """
    Reward Model for RLHF
    사용자 피드백을 reward score로 변환
    """

    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Recipe embeddings [batch_size, hidden_dim]
        Returns:
            reward_scores: [batch_size, 1]
        """
        return self.encoder(embeddings)


class QwenRLHFTrainer:
    """
    Qwen2.5 RLHF Trainer
    사용자 피드백을 받아 Qwen 모델을 fine-tuning
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct",
        redis_url: str = "redis://localhost:6379",
        feedback_queue: str = "rlhf:feedback:queue",
        batch_size: int = 4,
        learning_rate: float = 1e-5,
        lora_r: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.redis_url = redis_url
        self.feedback_queue = feedback_queue
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device

        # Redis 연결
        self.redis_client = None

        # 모델 초기화 (lazy loading)
        self.tokenizer = None
        self.model = None
        self.reward_model = None

        # LoRA 설정
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        logger.info(f"QwenRLHFTrainer initialized for {model_name}")

    async def connect_redis(self):
        """Redis 연결"""
        if not self.redis_client:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info("Connected to Redis")

    async def disconnect_redis(self):
        """Redis 연결 종료"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")

    def load_models(self):
        """Qwen 모델 및 Reward Model 로드"""
        logger.info("Loading Qwen model...")

        # Tokenizer 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Base 모델 로드 (8-bit quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )

        # LoRA 적용
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

        # Reward Model 초기화
        self.reward_model = RewardModel().to(self.device)

        logger.info("Models loaded successfully")

    async def collect_feedback(self, timeout: int = 60) -> List[UserFeedback]:
        """
        Redis Queue에서 사용자 피드백 수집

        Args:
            timeout: 대기 시간 (초)

        Returns:
            List of UserFeedback objects
        """
        await self.connect_redis()

        feedbacks = []
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout:
            # Redis Stream에서 피드백 읽기
            result = await self.redis_client.xread(
                {self.feedback_queue: '$'},
                count=10,
                block=5000  # 5초 대기
            )

            if result:
                for stream_name, messages in result:
                    for message_id, data in messages:
                        try:
                            feedback = UserFeedback.from_dict(
                                {k.decode(): v.decode() for k, v in data.items()}
                            )
                            feedbacks.append(feedback)
                            logger.debug(f"Collected feedback: {feedback.recipe_id}")
                        except Exception as e:
                            logger.error(f"Failed to parse feedback: {e}")

            # 충분한 피드백이 모이면 종료
            if len(feedbacks) >= self.batch_size:
                break

        logger.info(f"Collected {len(feedbacks)} feedbacks")
        return feedbacks

    def compute_reward(self, feedback: UserFeedback) -> float:
        """
        사용자 피드백을 reward score로 변환

        Args:
            feedback: UserFeedback object

        Returns:
            reward_score: -1.0 ~ 1.0
        """
        # Rating 정규화 (1~5 → -1~1)
        normalized_rating = (feedback.user_rating - 3.0) / 2.0

        # 텍스트 감정 분석 보정 (간단한 키워드 기반)
        sentiment_bonus = 0.0
        positive_keywords = ['좋', '향기롭', '만족', '훌륭', '완벽', '사랑']
        negative_keywords = ['나쁨', '싫', '불만', '실망', '별로', '최악']

        text = feedback.feedback_text.lower()
        for keyword in positive_keywords:
            if keyword in text:
                sentiment_bonus += 0.1
        for keyword in negative_keywords:
            if keyword in text:
                sentiment_bonus -= 0.1

        # 최종 reward
        reward = np.clip(normalized_rating + sentiment_bonus, -1.0, 1.0)
        return float(reward)

    def create_rlhf_samples(
        self,
        feedbacks: List[UserFeedback]
    ) -> List[RLHFSample]:
        """
        피드백을 RLHF 학습 샘플로 변환

        Args:
            feedbacks: List of UserFeedback

        Returns:
            List of RLHFSample
        """
        samples = []

        # 피드백을 긍정/부정으로 분류
        positive_feedbacks = [f for f in feedbacks if f.user_rating >= 4.0]
        negative_feedbacks = [f for f in feedbacks if f.user_rating < 3.0]

        # Pairwise comparison 샘플 생성
        for pos_fb in positive_feedbacks:
            for neg_fb in negative_feedbacks:
                sample = RLHFSample(
                    prompt=f"다음 조건에 맞는 향수를 만들어주세요: {pos_fb.preferred_notes}",
                    chosen_response=f"Recipe ID: {pos_fb.recipe_id} (선호됨)",
                    rejected_response=f"Recipe ID: {neg_fb.recipe_id} (거부됨)",
                    reward_score=self.compute_reward(pos_fb)
                )
                samples.append(sample)

        logger.info(f"Created {len(samples)} RLHF samples")
        return samples

    async def train_step(self, samples: List[RLHFSample]) -> Dict[str, float]:
        """
        RLHF 학습 스텝 실행

        Args:
            samples: List of RLHFSample

        Returns:
            Training metrics
        """
        if not self.model:
            self.load_models()

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate
        )

        total_loss = 0.0
        for sample in samples:
            # Tokenize
            chosen_inputs = self.tokenizer(
                sample.prompt + sample.chosen_response,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            rejected_inputs = self.tokenizer(
                sample.prompt + sample.rejected_response,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            # Forward pass
            chosen_outputs = self.model(**chosen_inputs, labels=chosen_inputs.input_ids)
            rejected_outputs = self.model(**rejected_inputs, labels=rejected_inputs.input_ids)

            # Compute loss (DPO-style)
            chosen_logprobs = -chosen_outputs.loss
            rejected_logprobs = -rejected_outputs.loss

            # Bradley-Terry preference loss
            loss = -torch.log(
                torch.sigmoid(chosen_logprobs - rejected_logprobs)
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(samples)
        logger.info(f"Training step completed. Avg loss: {avg_loss:.4f}")

        return {
            "loss": avg_loss,
            "samples": len(samples)
        }

    async def run_training_loop(self, max_iterations: int = 1000):
        """
        RLHF 학습 루프 실행

        Args:
            max_iterations: 최대 반복 횟수
        """
        logger.info(f"Starting RLHF training loop (max_iterations={max_iterations})")

        for iteration in range(max_iterations):
            # 1. 피드백 수집
            feedbacks = await self.collect_feedback(timeout=60)

            if len(feedbacks) < self.batch_size:
                logger.warning(f"Not enough feedbacks ({len(feedbacks)}), skipping iteration")
                await asyncio.sleep(10)
                continue

            # 2. RLHF 샘플 생성
            samples = self.create_rlhf_samples(feedbacks)

            if not samples:
                logger.warning("No valid samples created, skipping iteration")
                continue

            # 3. 학습 스텝 실행
            metrics = await self.train_step(samples)

            # 4. 체크포인트 저장 (100 iteration마다)
            if iteration % 100 == 0:
                checkpoint_path = f"./checkpoints/qwen_rlhf_iter_{iteration}.pt"
                self.model.save_pretrained(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

            # 5. 메트릭 로깅
            logger.info(
                f"Iteration {iteration}/{max_iterations} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Samples: {metrics['samples']}"
            )

            await asyncio.sleep(1)

        logger.info("RLHF training loop completed")


# CLI 진입점
async def main():
    """메인 함수"""
    trainer = QwenRLHFTrainer(
        model_name="Qwen/Qwen2.5-7B-Instruct",  # 테스트용 작은 모델
        redis_url="redis://localhost:6379",
        batch_size=4
    )

    # 학습 루프 실행
    await trainer.run_training_loop(max_iterations=100)


if __name__ == "__main__":
    asyncio.run(main())
