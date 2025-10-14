"""
체크포인트 매니저 (Checkpoint Manager)

정책/가치 네트워크를 주기적으로 저장하고, 손실 급등/보상 급락 시 롤백합니다.
"""

import torch
import os
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 체크포인트 메타데이터
# =============================================================================

@dataclass
class CheckpointMetadata:
    """체크포인트 메타데이터"""
    step: int
    timestamp: str
    loss: float
    reward: float
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    clip_fraction: Optional[float] = None
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "CheckpointMetadata":
        """딕셔너리에서 로드"""
        return CheckpointMetadata(**data)


# =============================================================================
# 롤백 트리거 조건
# =============================================================================

@dataclass
class RollbackConditions:
    """롤백 트리거 조건"""
    kl_threshold: float = 0.03  # KL divergence 임계값
    loss_increase_multiplier: float = 2.0  # 손실 증가 배수
    reward_drop_threshold: float = 0.3  # 보상 하락 임계값 (30%)

    def should_rollback(
        self,
        current: CheckpointMetadata,
        previous: CheckpointMetadata
    ) -> tuple[bool, Optional[str]]:
        """
        롤백 필요 여부 판단

        Args:
            current: 현재 체크포인트 메타데이터
            previous: 이전 체크포인트 메타데이터

        Returns:
            (should_rollback, reason)
        """
        # 1. KL divergence 체크
        if current.kl_divergence and current.kl_divergence > self.kl_threshold:
            return True, f"KL divergence {current.kl_divergence:.4f} > {self.kl_threshold}"

        # 2. 손실 급등 체크
        if current.loss > previous.loss * self.loss_increase_multiplier:
            return True, f"Loss 급등: {previous.loss:.4f} -> {current.loss:.4f} ({current.loss/previous.loss:.2f}x)"

        # 3. 보상 급락 체크
        if previous.reward > 0:
            reward_drop = (previous.reward - current.reward) / abs(previous.reward)
            if reward_drop > self.reward_drop_threshold:
                return True, f"Reward 급락: {previous.reward:.4f} -> {current.reward:.4f} (-{reward_drop*100:.1f}%)"

        return False, None


# =============================================================================
# 체크포인트 매니저
# =============================================================================

class CheckpointManager:
    """
    체크포인트 매니저

    N step마다 정책/가치 네트워크를 저장하고, 손실 급등/보상 급락 시 롤백합니다.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 500,
        max_checkpoints: int = 5,
        rollback_conditions: Optional[RollbackConditions] = None
    ):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            save_interval: 저장 주기 (steps)
            max_checkpoints: 최대 보관 체크포인트 수
            rollback_conditions: 롤백 조건
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.rollback_conditions = rollback_conditions or RollbackConditions()

        # 체크포인트 히스토리
        self.checkpoints: List[CheckpointMetadata] = []
        self._load_checkpoint_history()

        logger.info(f"체크포인트 매니저 초기화: {checkpoint_dir}, 주기={save_interval}, 최대={max_checkpoints}")

    def _load_checkpoint_history(self):
        """체크포인트 히스토리 로드"""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        if history_path.exists():
            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.checkpoints = [CheckpointMetadata.from_dict(item) for item in data]
                logger.info(f"체크포인트 히스토리 로드: {len(self.checkpoints)}개")
            except Exception as e:
                logger.error(f"체크포인트 히스토리 로드 실패: {e}")
                self.checkpoints = []

    def _save_checkpoint_history(self):
        """체크포인트 히스토리 저장"""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        try:
            with open(history_path, 'w', encoding='utf-8') as f:
                data = [ckpt.to_dict() for ckpt in self.checkpoints]
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"체크포인트 히스토리 저장 실패: {e}")

    def should_save(self, step: int) -> bool:
        """
        체크포인트 저장 여부 판단

        Args:
            step: 현재 스텝

        Returns:
            저장 필요 여부
        """
        return step > 0 and step % self.save_interval == 0

    def save_checkpoint(
        self,
        step: int,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss: float = 0.0,
        reward: float = 0.0,
        kl_divergence: Optional[float] = None,
        entropy: Optional[float] = None,
        clip_fraction: Optional[float] = None
    ) -> CheckpointMetadata:
        """
        체크포인트 저장

        Args:
            step: 현재 스텝
            policy_net: 정책 네트워크
            value_net: 가치 네트워크
            optimizer: 옵티마이저 (optional)
            loss: 현재 손실
            reward: 현재 평균 보상
            kl_divergence: KL divergence (optional)
            entropy: 엔트로피 (optional)
            clip_fraction: Clipping fraction (optional)

        Returns:
            CheckpointMetadata
        """
        # 체크포인트 경로
        checkpoint_name = f"checkpoint_step_{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        # 모델 저장
        checkpoint = {
            "step": step,
            "policy_net_state_dict": policy_net.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
            "loss": loss,
            "reward": reward,
            "kl_divergence": kl_divergence,
            "entropy": entropy,
            "clip_fraction": clip_fraction,
            "timestamp": datetime.now().isoformat()
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"체크포인트 저장: step={step}, loss={loss:.4f}, reward={reward:.4f}")
        except Exception as e:
            logger.error(f"체크포인트 저장 실패: {e}")
            raise

        # 메타데이터 생성
        metadata = CheckpointMetadata(
            step=step,
            timestamp=checkpoint["timestamp"],
            loss=loss,
            reward=reward,
            kl_divergence=kl_divergence,
            entropy=entropy,
            clip_fraction=clip_fraction,
            checkpoint_path=str(checkpoint_path)
        )

        # 히스토리에 추가
        self.checkpoints.append(metadata)

        # 오래된 체크포인트 삭제
        self._cleanup_old_checkpoints()

        # 히스토리 저장
        self._save_checkpoint_history()

        return metadata

    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 삭제 (최대 N개 유지)"""
        if len(self.checkpoints) > self.max_checkpoints:
            # 삭제할 체크포인트들
            to_delete = self.checkpoints[:-self.max_checkpoints]

            for ckpt in to_delete:
                if ckpt.checkpoint_path and os.path.exists(ckpt.checkpoint_path):
                    try:
                        os.remove(ckpt.checkpoint_path)
                        logger.info(f"오래된 체크포인트 삭제: {ckpt.checkpoint_path}")
                    except Exception as e:
                        logger.error(f"체크포인트 삭제 실패: {e}")

            # 히스토리에서 제거
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]

    def check_rollback(
        self,
        current_step: int,
        current_loss: float,
        current_reward: float,
        current_kl_divergence: Optional[float] = None
    ) -> tuple[bool, Optional[CheckpointMetadata], Optional[str]]:
        """
        롤백 필요 여부 체크

        Args:
            current_step: 현재 스텝
            current_loss: 현재 손실
            current_reward: 현재 보상
            current_kl_divergence: 현재 KL divergence (optional)

        Returns:
            (should_rollback, rollback_checkpoint, reason)
        """
        if len(self.checkpoints) < 2:
            return False, None, None

        # 현재 메타데이터
        current = CheckpointMetadata(
            step=current_step,
            timestamp=datetime.now().isoformat(),
            loss=current_loss,
            reward=current_reward,
            kl_divergence=current_kl_divergence
        )

        # 이전 체크포인트
        previous = self.checkpoints[-1]

        # 롤백 판단
        should_rollback, reason = self.rollback_conditions.should_rollback(current, previous)

        if should_rollback:
            logger.warning(f"[롤백 트리거] {reason}")
            return True, previous, reason

        return False, None, None

    def load_checkpoint(
        self,
        checkpoint_metadata: CheckpointMetadata,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        체크포인트 로드

        Args:
            checkpoint_metadata: 체크포인트 메타데이터
            policy_net: 정책 네트워크
            value_net: 가치 네트워크
            optimizer: 옵티마이저 (optional)

        Returns:
            체크포인트 데이터
        """
        checkpoint_path = checkpoint_metadata.checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"체크포인트 파일 없음: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 네트워크 로드
            policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            value_net.load_state_dict(checkpoint["value_net_state_dict"])

            # 옵티마이저 로드
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            logger.info(f"체크포인트 로드: step={checkpoint['step']}, loss={checkpoint['loss']:.4f}")

            return checkpoint

        except Exception as e:
            logger.error(f"체크포인트 로드 실패: {e}")
            raise

    def rollback(
        self,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Optional[CheckpointMetadata]:
        """
        직전 체크포인트로 롤백

        Args:
            policy_net: 정책 네트워크
            value_net: 가치 네트워크
            optimizer: 옵티마이저 (optional)

        Returns:
            롤백된 체크포인트 메타데이터 (없으면 None)
        """
        if not self.checkpoints:
            logger.warning("롤백 불가: 체크포인트 없음")
            return None

        # 직전 체크포인트
        previous = self.checkpoints[-1]

        try:
            # 체크포인트 로드
            self.load_checkpoint(previous, policy_net, value_net, optimizer)

            logger.info(f"[롤백 완료] step={previous.step}, loss={previous.loss:.4f}, reward={previous.reward:.4f}")

            return previous

        except Exception as e:
            logger.error(f"롤백 실패: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """최신 체크포인트 조회"""
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """체크포인트 요약"""
        return {
            "total_checkpoints": len(self.checkpoints),
            "latest_step": self.checkpoints[-1].step if self.checkpoints else None,
            "latest_loss": self.checkpoints[-1].loss if self.checkpoints else None,
            "latest_reward": self.checkpoints[-1].reward if self.checkpoints else None,
            "checkpoint_dir": str(self.checkpoint_dir),
            "save_interval": self.save_interval,
            "max_checkpoints": self.max_checkpoints
        }


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 더미 네트워크
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 10)

    policy_net = DummyNet()
    value_net = DummyNet()
    optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

    # 체크포인트 매니저 초기화
    manager = CheckpointManager(
        checkpoint_dir="test_checkpoints",
        save_interval=500,
        max_checkpoints=5,
        rollback_conditions=RollbackConditions(
            kl_threshold=0.03,
            loss_increase_multiplier=2.0,
            reward_drop_threshold=0.3
        )
    )

    # 시뮬레이션: 학습 루프
    print("\n=== 학습 시뮬레이션 ===\n")

    for step in range(0, 2500, 500):
        loss = 1.0 - step / 5000 + (0.1 if step == 2000 else 0)  # step 2000에서 손실 증가
        reward = step / 100

        # 체크포인트 저장 여부
        if manager.should_save(step):
            metadata = manager.save_checkpoint(
                step=step,
                policy_net=policy_net,
                value_net=value_net,
                optimizer=optimizer,
                loss=loss,
                reward=reward,
                kl_divergence=0.01,
                entropy=2.5,
                clip_fraction=0.15
            )

        # 롤백 체크
        should_rollback, rollback_ckpt, reason = manager.check_rollback(
            current_step=step,
            current_loss=loss,
            current_reward=reward,
            current_kl_divergence=0.01
        )

        if should_rollback:
            print(f"\n[ROLLBACK] Step {step}: {reason}")
            manager.rollback(policy_net, value_net, optimizer)

    # 체크포인트 요약
    print("\n=== 체크포인트 요약 ===")
    summary = manager.get_checkpoint_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 최신 체크포인트
    print("\n=== 최신 체크포인트 ===")
    latest = manager.get_latest_checkpoint()
    if latest:
        print(f"  Step: {latest.step}")
        print(f"  Loss: {latest.loss:.4f}")
        print(f"  Reward: {latest.reward:.4f}")
        print(f"  Path: {latest.checkpoint_path}")

    # Cleanup
    import shutil
    shutil.rmtree("test_checkpoints", ignore_errors=True)
    print("\n테스트 완료")
