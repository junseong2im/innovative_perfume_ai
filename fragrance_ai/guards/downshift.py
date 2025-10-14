"""
Automatic Downshift - 장애 시 자동 모드 다운시프트

creative → balanced → fast

장애 시 자동으로 더 빠른 모드로 전환하여 서비스 가용성 보장
"""

import logging
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Mode(Enum):
    """LLM 모드"""
    CREATIVE = "creative"
    BALANCED = "balanced"
    FAST = "fast"


class DownshiftReason(Enum):
    """다운시프트 사유"""
    MODEL_UNAVAILABLE = "model_unavailable"
    HIGH_LATENCY = "high_latency"
    ERROR_RATE_HIGH = "error_rate_high"
    MEMORY_PRESSURE = "memory_pressure"
    MANUAL = "manual"


@dataclass
class DownshiftEvent:
    """다운시프트 이벤트"""
    from_mode: Mode
    to_mode: Mode
    reason: DownshiftReason
    timestamp: datetime
    metadata: Dict[str, Any]

    def __str__(self) -> str:
        return (
            f"Downshift: {self.from_mode.value} -> {self.to_mode.value} "
            f"(reason: {self.reason.value}, time: {self.timestamp.isoformat()})"
        )


@dataclass
class DownshiftConfig:
    """다운시프트 설정"""
    # Latency thresholds (seconds)
    creative_latency_threshold: float = 6.0  # 4.5s target + 33% margin
    balanced_latency_threshold: float = 4.3  # 3.2s target + 33% margin
    fast_latency_threshold: float = 3.3  # 2.5s target + 33% margin

    # Error rate thresholds
    error_rate_threshold: float = 0.3  # 30%
    error_window_size: int = 10  # Last 10 requests

    # Memory thresholds
    memory_threshold_mb: float = 12000  # 12GB
    memory_threshold_percent: float = 85.0  # 85%

    # Cooldown (prevent rapid downshifts)
    downshift_cooldown_seconds: int = 60  # 1 minute

    # Auto-recovery
    auto_recovery_enabled: bool = True
    recovery_wait_minutes: int = 5


class DownshiftManager:
    """
    다운시프트 관리자

    장애 상황에서 자동으로 모드를 다운시프트하여
    서비스 가용성을 보장합니다.

    Downshift hierarchy:
    creative (Llama) → balanced (Mistral) → fast (Qwen)
    """

    # Mode hierarchy (highest to lowest)
    MODE_HIERARCHY = [Mode.CREATIVE, Mode.BALANCED, Mode.FAST]

    # Mode to model mapping
    MODE_TO_MODEL = {
        Mode.CREATIVE: "llama",
        Mode.BALANCED: "mistral",
        Mode.FAST: "qwen"
    }

    def __init__(self, config: Optional[DownshiftConfig] = None):
        self.config = config or DownshiftConfig()
        self.current_mode = Mode.BALANCED  # Default mode
        self.downshift_history: List[DownshiftEvent] = []
        self.last_downshift_time: Optional[datetime] = None
        self.error_history: List[bool] = []  # True = error, False = success

    def should_downshift(
        self,
        mode: Mode,
        latency_ms: Optional[float] = None,
        error: Optional[bool] = None,
        memory_mb: Optional[float] = None,
        model_unavailable: bool = False
    ) -> tuple[bool, Optional[DownshiftReason]]:
        """
        다운시프트가 필요한지 판단

        Args:
            mode: 현재 모드
            latency_ms: 레이턴시 (밀리초)
            error: 에러 발생 여부
            memory_mb: 메모리 사용량 (MB)
            model_unavailable: 모델 사용 불가 여부

        Returns:
            (should_downshift: bool, reason: DownshiftReason)
        """
        # Check cooldown
        if self.last_downshift_time:
            elapsed = (datetime.now() - self.last_downshift_time).total_seconds()
            if elapsed < self.config.downshift_cooldown_seconds:
                logger.debug(f"Downshift cooldown active ({elapsed:.1f}s/{self.config.downshift_cooldown_seconds}s)")
                return False, None

        # Reason 1: Model unavailable
        if model_unavailable:
            return True, DownshiftReason.MODEL_UNAVAILABLE

        # Reason 2: High latency
        if latency_ms is not None:
            latency_s = latency_ms / 1000
            threshold = self._get_latency_threshold(mode)
            if latency_s > threshold:
                logger.warning(f"High latency detected: {latency_s:.2f}s > {threshold:.2f}s")
                return True, DownshiftReason.HIGH_LATENCY

        # Reason 3: High error rate
        if error is not None:
            self.error_history.append(error)
            # Keep only last N errors
            if len(self.error_history) > self.config.error_window_size:
                self.error_history.pop(0)

            if len(self.error_history) >= self.config.error_window_size:
                error_rate = sum(self.error_history) / len(self.error_history)
                if error_rate > self.config.error_rate_threshold:
                    logger.warning(f"High error rate detected: {error_rate:.1%}")
                    return True, DownshiftReason.ERROR_RATE_HIGH

        # Reason 4: Memory pressure
        if memory_mb is not None:
            if memory_mb > self.config.memory_threshold_mb:
                logger.warning(f"Memory pressure detected: {memory_mb:.0f}MB > {self.config.memory_threshold_mb:.0f}MB")
                return True, DownshiftReason.MEMORY_PRESSURE

        return False, None

    def downshift(
        self,
        current_mode: Mode,
        reason: DownshiftReason,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Mode]:
        """
        다운시프트 실행

        Args:
            current_mode: 현재 모드
            reason: 다운시프트 사유
            metadata: 추가 메타데이터

        Returns:
            새로운 모드 or None (이미 최하위 모드)
        """
        # Get next lower mode
        new_mode = self._get_next_lower_mode(current_mode)

        if new_mode is None:
            logger.warning(f"Already at lowest mode: {current_mode.value}")
            return None

        # Record downshift event
        event = DownshiftEvent(
            from_mode=current_mode,
            to_mode=new_mode,
            reason=reason,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.downshift_history.append(event)
        self.last_downshift_time = event.timestamp
        self.current_mode = new_mode

        logger.warning(str(event))

        return new_mode

    def attempt_recovery(self, current_mode: Mode) -> Optional[Mode]:
        """
        자동 복구 시도 (모드 업그레이드)

        Args:
            current_mode: 현재 모드

        Returns:
            복구된 모드 or None
        """
        if not self.config.auto_recovery_enabled:
            return None

        # Check if enough time has passed since last downshift
        if self.last_downshift_time:
            elapsed = datetime.now() - self.last_downshift_time
            if elapsed < timedelta(minutes=self.config.recovery_wait_minutes):
                return None

        # Get next higher mode
        new_mode = self._get_next_higher_mode(current_mode)

        if new_mode is None:
            # Already at highest mode
            return None

        logger.info(f"Attempting recovery: {current_mode.value} -> {new_mode.value}")
        self.current_mode = new_mode

        return new_mode

    def _get_next_lower_mode(self, current_mode: Mode) -> Optional[Mode]:
        """다음 하위 모드 조회"""
        try:
            current_index = self.MODE_HIERARCHY.index(current_mode)
            if current_index < len(self.MODE_HIERARCHY) - 1:
                return self.MODE_HIERARCHY[current_index + 1]
        except ValueError:
            pass
        return None

    def _get_next_higher_mode(self, current_mode: Mode) -> Optional[Mode]:
        """다음 상위 모드 조회"""
        try:
            current_index = self.MODE_HIERARCHY.index(current_mode)
            if current_index > 0:
                return self.MODE_HIERARCHY[current_index - 1]
        except ValueError:
            pass
        return None

    def _get_latency_threshold(self, mode: Mode) -> float:
        """모드별 레이턴시 임계값 조회"""
        thresholds = {
            Mode.CREATIVE: self.config.creative_latency_threshold,
            Mode.BALANCED: self.config.balanced_latency_threshold,
            Mode.FAST: self.config.fast_latency_threshold
        }
        return thresholds.get(mode, 5.0)

    def get_current_mode(self) -> Mode:
        """현재 모드 조회"""
        return self.current_mode

    def get_current_model(self) -> str:
        """현재 모델 조회"""
        return self.MODE_TO_MODEL[self.current_mode]

    def get_downshift_history(self, limit: int = 10) -> List[DownshiftEvent]:
        """다운시프트 히스토리 조회"""
        return self.downshift_history[-limit:]

    def reset_error_history(self):
        """에러 히스토리 리셋"""
        self.error_history.clear()

    def get_status(self) -> Dict[str, Any]:
        """상태 조회"""
        return {
            "current_mode": self.current_mode.value,
            "current_model": self.get_current_model(),
            "last_downshift": self.last_downshift_time.isoformat() if self.last_downshift_time else None,
            "downshift_count": len(self.downshift_history),
            "error_rate": sum(self.error_history) / len(self.error_history) if self.error_history else 0.0,
            "recent_downshifts": [
                {
                    "from": event.from_mode.value,
                    "to": event.to_mode.value,
                    "reason": event.reason.value,
                    "time": event.timestamp.isoformat()
                }
                for event in self.get_downshift_history(5)
            ]
        }


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create downshift manager
    manager = DownshiftManager(DownshiftConfig(
        creative_latency_threshold=6.0,
        balanced_latency_threshold=4.3,
        fast_latency_threshold=3.3,
        error_rate_threshold=0.3,
        downshift_cooldown_seconds=5  # Short cooldown for demo
    ))

    print("=== Initial State ===")
    print(f"Current mode: {manager.get_current_mode().value}")
    print(f"Current model: {manager.get_current_model()}\n")

    # Scenario 1: High latency in creative mode
    print("=== Scenario 1: High Latency in Creative Mode ===")
    should_downshift, reason = manager.should_downshift(
        mode=Mode.CREATIVE,
        latency_ms=7000  # 7 seconds
    )
    print(f"Should downshift: {should_downshift}, Reason: {reason}")

    if should_downshift:
        new_mode = manager.downshift(Mode.CREATIVE, reason, metadata={"latency_ms": 7000})
        print(f"Downshifted to: {new_mode.value if new_mode else 'None'}\n")

    # Scenario 2: High error rate
    print("=== Scenario 2: High Error Rate ===")
    import time
    time.sleep(6)  # Wait for cooldown

    for i in range(10):
        manager.should_downshift(mode=Mode.BALANCED, error=(i < 4))  # 40% error rate

    should_downshift, reason = manager.should_downshift(mode=Mode.BALANCED, error=True)
    print(f"Should downshift: {should_downshift}, Reason: {reason}")

    if should_downshift:
        new_mode = manager.downshift(Mode.BALANCED, reason)
        print(f"Downshifted to: {new_mode.value if new_mode else 'None'}\n")

    # Scenario 3: Model unavailable
    print("=== Scenario 3: Model Unavailable ===")
    time.sleep(6)  # Wait for cooldown

    should_downshift, reason = manager.should_downshift(
        mode=Mode.FAST,
        model_unavailable=True
    )
    print(f"Should downshift: {should_downshift}, Reason: {reason}")

    if should_downshift:
        new_mode = manager.downshift(Mode.FAST, reason)
        print(f"Downshifted to: {new_mode.value if new_mode else 'Already at lowest mode'}\n")

    # Status
    print("=== Final Status ===")
    import json
    print(json.dumps(manager.get_status(), indent=2))
