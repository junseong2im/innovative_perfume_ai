"""
운영 설정 (Operations Config)
실무 환경에서 바로 적용 가능한 운영 팁 구현
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# 1. 학습 스케줄 설정
# =============================================================================

@dataclass
class LearningScheduleConfig:
    """학습 스케줄 설정"""

    # Entropy coefficient decay
    entropy_coef_start: float = 0.01  # 초기 엔트로피 계수
    entropy_coef_end: float = 0.001   # 최종 엔트로피 계수
    entropy_decay_type: str = "cosine"  # linear or cosine
    entropy_decay_steps: int = 10000   # Decay 기간

    # Reward normalization
    reward_norm_window: int = 1000  # 1k step 창
    reward_norm_epsilon: float = 1e-8

    # Learning rate decay
    lr_decay_type: str = "cosine"
    lr_warmup_steps: int = 100


class EntropyCoefficientScheduler:
    """Entropy coefficient scheduler with linear/cosine decay"""

    def __init__(self, config: LearningScheduleConfig):
        self.config = config
        self.current_step = 0

    def get_entropy_coef(self, step: Optional[int] = None) -> float:
        """현재 스텝의 entropy coefficient 반환"""
        if step is None:
            step = self.current_step
        else:
            self.current_step = step

        if step >= self.config.entropy_decay_steps:
            return self.config.entropy_coef_end

        progress = step / self.config.entropy_decay_steps

        if self.config.entropy_decay_type == "linear":
            # Linear decay
            coef = self.config.entropy_coef_start - \
                   (self.config.entropy_coef_start - self.config.entropy_coef_end) * progress
        else:
            # Cosine decay
            coef = self.config.entropy_coef_end + \
                   0.5 * (self.config.entropy_coef_start - self.config.entropy_coef_end) * \
                   (1 + np.cos(np.pi * progress))

        return coef

    def step(self) -> float:
        """스텝 진행 및 현재 coefficient 반환"""
        self.current_step += 1
        return self.get_entropy_coef()


class RewardNormalizer:
    """Reward normalization with sliding window"""

    def __init__(self, window_size: int = 1000, epsilon: float = 1e-8):
        self.window_size = window_size
        self.epsilon = epsilon
        self.rewards = deque(maxlen=window_size)

    def normalize(self, reward: float) -> float:
        """Reward 정규화"""
        # 윈도우에 추가
        self.rewards.append(reward)

        # 통계 계산
        if len(self.rewards) < 2:
            return reward

        mean = np.mean(self.rewards)
        std = np.std(self.rewards)

        # 정규화
        normalized = (reward - mean) / (std + self.epsilon)
        return normalized

    def get_stats(self) -> Dict[str, float]:
        """현재 통계 반환"""
        if len(self.rewards) < 2:
            return {"mean": 0.0, "std": 1.0, "count": len(self.rewards)}

        return {
            "mean": float(np.mean(self.rewards)),
            "std": float(np.std(self.rewards)),
            "count": len(self.rewards)
        }


# =============================================================================
# 2. 체크포인트 관리
# =============================================================================

@dataclass
class CheckpointConfig:
    """체크포인트 설정"""

    # 저장 주기
    save_interval_steps: int = 500  # 500 step마다 저장
    max_checkpoints: int = 5  # 최대 보관 개수

    # 이상 감지 임계값
    loss_spike_threshold: float = 2.0  # 이전 대비 2배 이상
    reward_drop_threshold: float = 0.5  # 이전 대비 50% 이하

    # 롤백 설정
    enable_auto_rollback: bool = True
    rollback_lookback: int = 3  # 최근 3개 체크포인트 중 선택


class CheckpointManager:
    """체크포인트 관리 with 자동 롤백"""

    def __init__(self, checkpoint_dir: str, config: CheckpointConfig):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # 이력 추적
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.loss_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=20)

    def should_save(self, current_step: int) -> bool:
        """저장 필요 여부 확인"""
        return current_step % self.config.save_interval_steps == 0

    def save_checkpoint(
        self,
        step: int,
        model_state: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> str:
        """체크포인트 저장"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pth"

        checkpoint_data = {
            "step": step,
            "model_state": model_state,
            "metrics": metrics,
            "timestamp": str(Path(__file__).stat().st_mtime)
        }

        import torch
        torch.save(checkpoint_data, checkpoint_path)

        # 이력 추가
        self.checkpoint_history.append({
            "step": step,
            "path": str(checkpoint_path),
            "loss": metrics.get("loss", 0.0),
            "reward": metrics.get("reward", 0.0)
        })

        # 메트릭 이력 추가
        self.loss_history.append(metrics.get("loss", 0.0))
        self.reward_history.append(metrics.get("reward", 0.0))

        # 오래된 체크포인트 정리
        self._cleanup_old_checkpoints()

        logger.info(f"✓ Checkpoint saved: {checkpoint_path} (step={step})")
        return str(checkpoint_path)

    def detect_anomaly(self, current_metrics: Dict[str, float]) -> Optional[str]:
        """이상 감지 - 급격한 loss/reward 변화"""
        if len(self.loss_history) < 3 or len(self.reward_history) < 3:
            return None

        current_loss = current_metrics.get("loss", 0.0)
        current_reward = current_metrics.get("reward", 0.0)

        # 최근 평균
        recent_loss_avg = np.mean(list(self.loss_history)[-5:])
        recent_reward_avg = np.mean(list(self.reward_history)[-5:])

        # Loss spike 감지
        if current_loss > recent_loss_avg * self.config.loss_spike_threshold:
            return f"loss_spike: {current_loss:.4f} > {recent_loss_avg:.4f} * {self.config.loss_spike_threshold}"

        # Reward drop 감지
        if current_reward < recent_reward_avg * self.config.reward_drop_threshold:
            return f"reward_drop: {current_reward:.4f} < {recent_reward_avg:.4f} * {self.config.reward_drop_threshold}"

        return None

    def rollback_to_best(self) -> Optional[str]:
        """최적 체크포인트로 롤백"""
        if len(self.checkpoint_history) < 2:
            logger.warning("Not enough checkpoints for rollback")
            return None

        # 최근 N개 중 최고 reward 선택
        recent_checkpoints = self.checkpoint_history[-self.config.rollback_lookback:]
        best_checkpoint = max(recent_checkpoints, key=lambda x: x["reward"])

        logger.warning(f"🔄 Rolling back to step {best_checkpoint['step']} (reward={best_checkpoint['reward']:.3f})")
        return best_checkpoint["path"]

    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 삭제"""
        if len(self.checkpoint_history) > self.config.max_checkpoints:
            # 오래된 것 삭제
            old_checkpoint = self.checkpoint_history.pop(0)
            old_path = Path(old_checkpoint["path"])
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Deleted old checkpoint: {old_path}")


# =============================================================================
# 3. 장애 대처 (Circuit Breaker)
# =============================================================================

@dataclass
class CircuitBreakerConfig:
    """서킷브레이커 설정"""

    # Qwen 장애 감지
    qwen_failure_threshold: int = 3  # 3번 실패 시 다운시프트
    qwen_timeout_seconds: float = 5.0

    # 다운시프트 전략
    enable_auto_downshift: bool = True
    downshift_duration_seconds: int = 300  # 5분간 다운시프트 유지

    # TTL 단축
    enable_ttl_reduction: bool = True
    normal_ttl_seconds: int = 3600  # 정상: 1시간
    reduced_ttl_seconds: int = 300   # 단축: 5분


class CircuitBreaker:
    """서킷브레이커 with 자동 다운시프트"""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config

        # 실패 카운터
        self.qwen_failures = 0
        self.qwen_last_failure_time = 0

        # 다운시프트 상태
        self.is_downshifted = False
        self.downshift_start_time = 0

    def record_qwen_failure(self):
        """Qwen 실패 기록"""
        import time
        self.qwen_failures += 1
        self.qwen_last_failure_time = time.time()

        logger.warning(f"⚠️ Qwen failure recorded: {self.qwen_failures}/{self.config.qwen_failure_threshold}")

        # 임계값 도달 시 다운시프트
        if self.qwen_failures >= self.config.qwen_failure_threshold:
            if self.config.enable_auto_downshift:
                self._trigger_downshift()

    def record_qwen_success(self):
        """Qwen 성공 기록 - 실패 카운터 리셋"""
        self.qwen_failures = max(0, self.qwen_failures - 1)

    def _trigger_downshift(self):
        """다운시프트 트리거"""
        import time
        self.is_downshifted = True
        self.downshift_start_time = time.time()

        logger.warning("🔻 Auto downshift triggered: creative -> balanced/fast")

        # 메트릭 기록
        try:
            from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector
            collector = OperationsMetricsCollector()
            collector.record_circuit_breaker_downgrade("llm", "creative", "balanced")
        except Exception as e:
            logger.debug(f"Failed to record downshift metric: {e}")

    def should_use_downshift(self) -> bool:
        """다운시프트 사용 여부"""
        if not self.is_downshifted:
            return False

        import time
        elapsed = time.time() - self.downshift_start_time

        # 다운시프트 기간 종료
        if elapsed > self.config.downshift_duration_seconds:
            self.is_downshifted = False
            self.qwen_failures = 0
            logger.info("✓ Downshift period ended, returning to normal")
            return False

        return True

    def get_current_ttl(self) -> int:
        """현재 TTL 반환"""
        if self.config.enable_ttl_reduction and self.is_downshifted:
            return self.config.reduced_ttl_seconds
        return self.config.normal_ttl_seconds


# =============================================================================
# 4. 데이터 기반 재튜닝
# =============================================================================

@dataclass
class RetuningConfig:
    """재튜닝 설정"""

    # 피드백 샘플 임계값
    min_samples_for_retune: int = 100
    optimal_samples_for_retune: int = 300

    # 재튜닝 대상
    retune_learning_rate: bool = True
    retune_entropy_coef: bool = True
    retune_clip_epsilon: bool = True


class HyperparameterRetuner:
    """데이터 기반 하이퍼파라미터 재튜닝"""

    def __init__(self, config: RetuningConfig):
        self.config = config
        self.feedback_samples: List[Dict[str, Any]] = []

    def add_feedback(self, feedback: Dict[str, Any]):
        """피드백 추가"""
        self.feedback_samples.append(feedback)
        logger.debug(f"Feedback added: {len(self.feedback_samples)}/{self.config.optimal_samples_for_retune}")

    def should_retune(self) -> bool:
        """재튜닝 필요 여부"""
        return len(self.feedback_samples) >= self.config.min_samples_for_retune

    def suggest_hyperparameters(self) -> Dict[str, float]:
        """피드백 기반 하이퍼파라미터 추천"""
        if not self.should_retune():
            logger.warning(f"Not enough samples for retuning: {len(self.feedback_samples)}")
            return {}

        # 피드백 분석
        ratings = [f.get("rating", 0.0) for f in self.feedback_samples]
        avg_rating = np.mean(ratings)
        rating_std = np.std(ratings)

        suggestions = {}

        # 평균 rating이 낮으면 exploration 증가 (entropy_coef 증가)
        if avg_rating < 3.0 and self.config.retune_entropy_coef:
            suggestions["entropy_coef"] = 0.02  # 기본 0.01 -> 0.02
            logger.info("📈 Suggesting higher entropy_coef for more exploration")

        # Rating 변동이 크면 learning rate 감소
        if rating_std > 1.5 and self.config.retune_learning_rate:
            suggestions["learning_rate"] = 1e-4  # 기본 3e-4 -> 1e-4
            logger.info("📉 Suggesting lower learning_rate for stability")

        # 평균 rating이 높으면 exploitation 증가 (clip_epsilon 감소)
        if avg_rating >= 4.0 and self.config.retune_clip_epsilon:
            suggestions["clip_epsilon"] = 0.1  # 기본 0.2 -> 0.1
            logger.info("🎯 Suggesting lower clip_epsilon for exploitation")

        logger.info(f"Retuning suggestions based on {len(self.feedback_samples)} samples:")
        for key, value in suggestions.items():
            logger.info(f"  {key}: {value}")

        return suggestions

    def clear_feedback(self):
        """피드백 샘플 초기화"""
        self.feedback_samples.clear()
        logger.info("Feedback samples cleared")


# =============================================================================
# 통합 운영 설정
# =============================================================================

class OperationsManager:
    """통합 운영 관리자"""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        enable_all_features: bool = True
    ):
        # 설정 초기화
        self.schedule_config = LearningScheduleConfig()
        self.checkpoint_config = CheckpointConfig()
        self.circuit_breaker_config = CircuitBreakerConfig()
        self.retuning_config = RetuningConfig()

        # 컴포넌트 초기화
        self.entropy_scheduler = EntropyCoefficientScheduler(self.schedule_config)
        self.reward_normalizer = RewardNormalizer(
            window_size=self.schedule_config.reward_norm_window
        )
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, self.checkpoint_config)
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_config)
        self.retuner = HyperparameterRetuner(self.retuning_config)

        self.enabled = enable_all_features

    def on_training_step(
        self,
        step: int,
        metrics: Dict[str, float],
        model_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """학습 스텝마다 호출"""
        if not self.enabled:
            return {}

        # 1. Entropy coefficient 업데이트
        current_entropy_coef = self.entropy_scheduler.get_entropy_coef(step)

        # 2. Reward normalization
        if "reward" in metrics:
            normalized_reward = self.reward_normalizer.normalize(metrics["reward"])
            metrics["normalized_reward"] = normalized_reward

        # 3. 체크포인트 저장
        checkpoint_path = None
        if self.checkpoint_manager.should_save(step) and model_state:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(step, model_state, metrics)

        # 4. 이상 감지 및 롤백
        anomaly = self.checkpoint_manager.detect_anomaly(metrics)
        rollback_path = None
        if anomaly and self.checkpoint_config.enable_auto_rollback:
            logger.error(f"❌ Anomaly detected: {anomaly}")
            rollback_path = self.checkpoint_manager.rollback_to_best()

        return {
            "entropy_coef": current_entropy_coef,
            "checkpoint_path": checkpoint_path,
            "anomaly": anomaly,
            "rollback_path": rollback_path,
            "should_downshift": self.circuit_breaker.should_use_downshift(),
            "current_ttl": self.circuit_breaker.get_current_ttl()
        }

    def on_llm_failure(self, model: str):
        """LLM 실패 시 호출"""
        if model == "qwen":
            self.circuit_breaker.record_qwen_failure()

    def on_llm_success(self, model: str):
        """LLM 성공 시 호출"""
        if model == "qwen":
            self.circuit_breaker.record_qwen_success()

    def on_user_feedback(self, feedback: Dict[str, Any]):
        """사용자 피드백 수집"""
        self.retuner.add_feedback(feedback)

        # 재튜닝 제안
        if self.retuner.should_retune():
            suggestions = self.retuner.suggest_hyperparameters()
            return suggestions
        return None


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== 운영 설정 시뮬레이션 ===\n")

    # 운영 관리자 초기화
    ops_manager = OperationsManager(checkpoint_dir="./test_checkpoints")

    # 1. 학습 스텝 시뮬레이션
    print("1. 학습 스텝 시뮬레이션")
    for step in range(0, 1500, 100):
        metrics = {
            "loss": 0.5 - step * 0.0001 + np.random.randn() * 0.01,
            "reward": 10.0 + step * 0.01 + np.random.randn() * 0.5
        }

        model_state = {"dummy": "state"}  # 실제로는 torch 모델 state

        result = ops_manager.on_training_step(step, metrics, model_state)

        if result.get("checkpoint_path"):
            print(f"  Step {step}: Checkpoint saved")
        if result.get("anomaly"):
            print(f"  Step {step}: ⚠️ Anomaly - {result['anomaly']}")

    # 2. Qwen 장애 시뮬레이션
    print("\n2. Qwen 장애 시뮬레이션")
    for i in range(5):
        ops_manager.on_llm_failure("qwen")
        if ops_manager.circuit_breaker.should_use_downshift():
            print(f"  Failure {i+1}: Downshift activated!")
            break

    # 3. 사용자 피드백 수집
    print("\n3. 사용자 피드백 수집 및 재튜닝")
    for i in range(150):
        feedback = {
            "rating": np.random.uniform(2.5, 4.5),
            "user_id": f"user_{i}"
        }
        suggestions = ops_manager.on_user_feedback(feedback)

        if suggestions:
            print(f"  ✓ Retuning triggered after {i+1} samples")
            print(f"  Suggestions: {suggestions}")
            break

    print("\n완료!")
