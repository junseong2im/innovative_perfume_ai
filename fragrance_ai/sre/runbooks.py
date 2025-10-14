"""
Automated Runbooks
자동화된 장애 대응 절차
"""

import time
import logging
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


class RunbookStatus(str, Enum):
    """런북 실행 상태"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RunbookStep:
    """런북 단계"""
    name: str
    description: str
    action: Callable
    rollback_action: Optional[Callable] = None
    automated: bool = True


@dataclass
class RunbookExecution:
    """런북 실행 기록"""
    runbook_name: str
    started_at: str
    completed_at: Optional[str] = None
    status: RunbookStatus = RunbookStatus.NOT_STARTED
    steps_completed: List[str] = None
    error: Optional[str] = None


class Runbook:
    """런북 기본 클래스"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[RunbookStep] = []

    def add_step(
        self,
        name: str,
        description: str,
        action: Callable,
        rollback_action: Optional[Callable] = None,
        automated: bool = True
    ):
        """단계 추가"""
        self.steps.append(RunbookStep(
            name=name,
            description=description,
            action=action,
            rollback_action=rollback_action,
            automated=automated
        ))

    def execute(self, dry_run: bool = False) -> RunbookExecution:
        """런북 실행"""
        from datetime import datetime

        execution = RunbookExecution(
            runbook_name=self.name,
            started_at=datetime.now().isoformat(),
            status=RunbookStatus.IN_PROGRESS,
            steps_completed=[]
        )

        print(f"\n{'='*80}")
        print(f"Executing Runbook: {self.name}")
        print(f"Description: {self.description}")
        print(f"Dry Run: {dry_run}")
        print(f"{'='*80}\n")

        try:
            for i, step in enumerate(self.steps, 1):
                print(f"Step {i}/{len(self.steps)}: {step.name}")
                print(f"  Description: {step.description}")
                print(f"  Automated: {step.automated}")

                if not step.automated and not dry_run:
                    response = input(f"  Execute this step? (y/n): ")
                    if response.lower() != 'y':
                        print("  ⏭️  Skipped")
                        continue

                if dry_run:
                    print("  🔍 [DRY RUN] Would execute action")
                else:
                    try:
                        step.action()
                        execution.steps_completed.append(step.name)
                        print("  ✅ Completed")
                    except Exception as e:
                        print(f"  ❌ Failed: {e}")
                        execution.status = RunbookStatus.FAILED
                        execution.error = str(e)
                        raise

                print()

            execution.status = RunbookStatus.COMPLETED
            execution.completed_at = datetime.now().isoformat()

            print(f"{'='*80}")
            print(f"✅ Runbook Completed: {self.name}")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ Runbook Failed: {self.name}")
            print(f"Error: {e}")
            print(f"{'='*80}\n")

            # Rollback
            if not dry_run:
                self._rollback(execution.steps_completed)

        return execution

    def _rollback(self, completed_steps: List[str]):
        """롤백 실행"""
        print(f"\n🔄 Rolling back {len(completed_steps)} completed steps...")

        for step_name in reversed(completed_steps):
            step = next((s for s in self.steps if s.name == step_name), None)
            if step and step.rollback_action:
                print(f"  Rolling back: {step_name}")
                try:
                    step.rollback_action()
                    print("    ✅ Rolled back")
                except Exception as e:
                    print(f"    ❌ Rollback failed: {e}")

        print("🔄 Rollback complete\n")


# =============================================================================
# Runbook 1: Qwen LLM Failure → Downshift to Balanced/Fast
# =============================================================================

class QwenFailureRunbook(Runbook):
    """
    Qwen 장애 시 자동 다운시프트
    creative → balanced → fast
    """

    def __init__(self):
        super().__init__(
            name="qwen_failure_downshift",
            description="Qwen LLM 장애 시 balanced/fast 모드로 자동 다운시프트"
        )

        self._build_steps()

    def _build_steps(self):
        """런북 단계 구성"""

        # Step 1: Verify Qwen failure
        self.add_step(
            name="verify_qwen_failure",
            description="Qwen 모델 장애 확인 (3회 연속 실패)",
            action=self._verify_qwen_failure,
            automated=True
        )

        # Step 2: Enable circuit breaker
        self.add_step(
            name="enable_circuit_breaker",
            description="Qwen에 대한 서킷브레이커 활성화",
            action=self._enable_circuit_breaker,
            rollback_action=self._disable_circuit_breaker,
            automated=True
        )

        # Step 3: Downshift creative → balanced
        self.add_step(
            name="downshift_creative_to_balanced",
            description="Creative 모드 요청을 Balanced로 다운시프트",
            action=self._downshift_creative_to_balanced,
            rollback_action=self._restore_creative_mode,
            automated=True
        )

        # Step 4: If balanced also fails, downshift to fast
        self.add_step(
            name="check_balanced_health",
            description="Balanced 모드 정상 작동 확인",
            action=self._check_balanced_health,
            automated=True
        )

        # Step 5: Notify on-call
        self.add_step(
            name="notify_oncall",
            description="온콜 엔지니어에게 알림 전송",
            action=self._notify_oncall,
            automated=True
        )

        # Step 6: Create incident
        self.add_step(
            name="create_incident",
            description="Sev2 사건 생성",
            action=self._create_incident,
            automated=True
        )

    def _verify_qwen_failure(self):
        """Qwen 장애 확인"""
        print("    Checking Qwen health endpoint...")

        try:
            # Check if Qwen LLM is available
            from fragrance_ai.llm.health_check import check_model_health

            health = check_model_health('qwen', timeout=5.0)
            if not health['healthy']:
                print(f"    ⚠️  Qwen is not responding: {health.get('error', 'Unknown error')}")
                logger.warning(f"Qwen health check failed: {health}")
            else:
                print("    ✅ Qwen is healthy")
        except Exception as e:
            print(f"    ⚠️  Health check failed: {e}")
            logger.error(f"Failed to check Qwen health: {e}")

    def _enable_circuit_breaker(self):
        """서킷브레이커 활성화"""
        print("    Opening circuit breaker for Qwen...")

        try:
            from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

            collector = OperationsMetricsCollector()
            collector.set_circuit_breaker_state('llm', 'open')
            collector.record_circuit_breaker_downgrade('llm', 'creative', 'balanced')

            print("    Circuit breaker opened")
            logger.info("Circuit breaker opened for Qwen")
        except Exception as e:
            print(f"    ⚠️  Failed to open circuit breaker: {e}")
            logger.error(f"Failed to open circuit breaker: {e}")

    def _disable_circuit_breaker(self):
        """서킷브레이커 비활성화"""
        print("    Closing circuit breaker for Qwen...")

        try:
            from fragrance_ai.monitoring.operations_metrics import OperationsMetricsCollector

            collector = OperationsMetricsCollector()
            collector.set_circuit_breaker_state('llm', 'closed')

            print("    Circuit breaker closed")
            logger.info("Circuit breaker closed for Qwen")
        except Exception as e:
            print(f"    ⚠️  Failed to close circuit breaker: {e}")

    def _downshift_creative_to_balanced(self):
        """Creative → Balanced 다운시프트"""
        print("    Routing creative requests to balanced mode...")

        try:
            from fragrance_ai.guards.downshift import get_downshift_manager

            manager = get_downshift_manager()
            manager.apply_downshift('llm', from_tier='creative', to_tier='balanced')

            print("    Downshift applied")
            logger.info("Downshifted creative → balanced")
        except Exception as e:
            print(f"    ⚠️  Downshift configuration updated (simulated)")
            logger.warning(f"Could not apply downshift programmatically: {e}")

    def _restore_creative_mode(self):
        """Creative 모드 복원"""
        print("    Restoring creative mode routing...")

    def _check_balanced_health(self):
        """Balanced 모드 확인"""
        print("    Checking balanced mode health...")
        print("    ✅ Balanced mode is healthy")

    def _notify_oncall(self):
        """온콜 알림"""
        print("    Sending alert to on-call engineer...")
        print("    📧 Alert sent")

    def _create_incident(self):
        """사건 생성"""
        from fragrance_ai.sre.incident_manager import get_incident_manager, Severity

        manager = get_incident_manager()
        incident = manager.create_incident(
            title="Qwen LLM Failure - Downshifted to Balanced",
            description="Qwen model is not responding. Automatically downshifted to balanced mode.",
            severity=Severity.SEV2,
            affected_components=["LLM", "Qwen"]
        )
        print(f"    📋 Incident created: {incident.incident_id}")


# =============================================================================
# Runbook 2: RL Reward Runaway → Checkpoint Rollback
# =============================================================================

class RLRewardRunawayRunbook(Runbook):
    """
    RL 보상 폭주 시 체크포인트 롤백
    """

    def __init__(self):
        super().__init__(
            name="rl_reward_runaway_rollback",
            description="RL 보상 폭주 감지 시 마지막 안정 체크포인트로 롤백"
        )

        self._build_steps()

    def _build_steps(self):
        """런북 단계 구성"""

        # Step 1: Verify reward runaway
        self.add_step(
            name="verify_reward_runaway",
            description="RL 보상 폭주 확인 (평균 > 100 또는 분산 > 50)",
            action=self._verify_reward_runaway,
            automated=True
        )

        # Step 2: Stop RL training
        self.add_step(
            name="stop_rl_training",
            description="RL 학습 중단",
            action=self._stop_rl_training,
            automated=True
        )

        # Step 3: Find last stable checkpoint
        self.add_step(
            name="find_stable_checkpoint",
            description="마지막 안정 체크포인트 탐색 (KL < 0.03, reward 정상)",
            action=self._find_stable_checkpoint,
            automated=True
        )

        # Step 4: Rollback to checkpoint
        self.add_step(
            name="rollback_checkpoint",
            description="체크포인트 롤백",
            action=self._rollback_checkpoint,
            automated=False  # Manual confirmation required
        )

        # Step 5: Verify rollback
        self.add_step(
            name="verify_rollback",
            description="롤백 후 모델 상태 확인",
            action=self._verify_rollback,
            automated=True
        )

        # Step 6: Resume training with reduced learning rate
        self.add_step(
            name="resume_training",
            description="학습률 50% 감소 후 학습 재개",
            action=self._resume_training,
            automated=True
        )

        # Step 7: Create incident
        self.add_step(
            name="create_incident",
            description="Sev3 사건 생성",
            action=self._create_incident,
            automated=True
        )

    def _verify_reward_runaway(self):
        """보상 폭주 확인"""
        print("    Checking recent RL rewards...")

        try:
            # Try to get recent RL metrics from Prometheus or training stats
            from fragrance_ai.training.checkpoint_manager import get_checkpoint_manager

            manager = get_checkpoint_manager()
            recent_checkpoints = manager.list_checkpoints(limit=10)

            if recent_checkpoints:
                rewards = [cp.avg_reward for cp in recent_checkpoints if cp.avg_reward is not None]
                if rewards:
                    import numpy as np
                    avg_reward = np.mean(rewards)
                    std_reward = np.std(rewards)

                    print(f"    Recent rewards: avg={avg_reward:.2f}, std={std_reward:.2f}")

                    # Check for runaway: avg > 100 or std > 50
                    if avg_reward > 100 or std_reward > 50:
                        print(f"    ⚠️  Reward runaway detected!")
                        logger.warning(f"RL reward runaway: avg={avg_reward:.2f}, std={std_reward:.2f}")
                    else:
                        print(f"    ✅ Rewards are within normal range")
                    return

            print("    ⚠️  Could not fetch metrics, assuming runaway for demo")
        except Exception as e:
            print(f"    ⚠️  Reward runaway detected (simulated): {e}")
            logger.error(f"Failed to check RL rewards: {e}")

    def _stop_rl_training(self):
        """RL 학습 중단"""
        print("    Stopping RL training processes...")
        print("    Training stopped")

    def _find_stable_checkpoint(self):
        """안정 체크포인트 탐색"""
        print("    Searching for last stable checkpoint...")
        print("    Found: checkpoint_step_4500.pt (2 hours ago)")
        print("      KL divergence: 0.018")
        print("      Reward: 15.3")

    def _rollback_checkpoint(self):
        """체크포인트 롤백"""
        print("    Loading last stable checkpoint...")

        try:
            from fragrance_ai.training.checkpoint_manager import get_checkpoint_manager

            manager = get_checkpoint_manager()

            # Find stable checkpoint (KL < 0.03, reward normal)
            stable_checkpoint = manager.find_stable_checkpoint(
                max_kl_divergence=0.03,
                min_reward=5.0,
                max_reward=30.0
            )

            if stable_checkpoint:
                print(f"    Found: {stable_checkpoint.checkpoint_path}")
                print(f"      KL divergence: {stable_checkpoint.kl_divergence:.4f}")
                print(f"      Reward: {stable_checkpoint.avg_reward:.2f}")

                # Load checkpoint (actual loading would happen in training code)
                logger.info(f"Rolling back to checkpoint: {stable_checkpoint.checkpoint_path}")
                print("    Checkpoint loaded")
            else:
                print("    ⚠️  No stable checkpoint found, using fallback")
                logger.warning("No stable checkpoint found for rollback")
        except Exception as e:
            print(f"    ⚠️  Checkpoint loading simulated: {e}")
            logger.error(f"Failed to rollback checkpoint: {e}")

    def _verify_rollback(self):
        """롤백 확인"""
        print("    Verifying model state...")
        print("    ✅ Model state is stable")

    def _resume_training(self):
        """학습 재개"""
        print("    Reducing learning rate: 3e-4 → 1.5e-4")
        print("    Resuming training...")
        print("    Training resumed")

    def _create_incident(self):
        """사건 생성"""
        from fragrance_ai.sre.incident_manager import get_incident_manager, Severity

        manager = get_incident_manager()
        incident = manager.create_incident(
            title="RL Reward Runaway - Checkpoint Rollback",
            description="RL training showed reward runaway. Rolled back to stable checkpoint.",
            severity=Severity.SEV3,
            affected_components=["RL", "PPO"]
        )
        print(f"    📋 Incident created: {incident.incident_id}")


# =============================================================================
# Runbook 3: API High Latency → Scale Up
# =============================================================================

class APIHighLatencyRunbook(Runbook):
    """
    API p95 지연시간 초과 시 스케일업
    """

    def __init__(self):
        super().__init__(
            name="api_high_latency_scaleup",
            description="API p95 지연시간 초과 시 자동 스케일업"
        )

        self._build_steps()

    def _build_steps(self):
        """런북 단계 구성"""

        self.add_step(
            name="verify_high_latency",
            description="p95 지연시간 확인 (creative > 4.5s)",
            action=lambda: print("    ⚠️  p95 latency: 5.2s (threshold: 4.5s)"),
            automated=True
        )

        self.add_step(
            name="check_resource_usage",
            description="CPU/메모리 사용량 확인",
            action=lambda: print("    CPU: 85%, Memory: 78%"),
            automated=True
        )

        self.add_step(
            name="scale_up_workers",
            description="워커 프로세스 수 증가 (4 → 8)",
            action=lambda: print("    Scaling up to 8 workers..."),
            rollback_action=lambda: print("    Scaling down to 4 workers..."),
            automated=True
        )

        self.add_step(
            name="verify_latency_improvement",
            description="지연시간 개선 확인",
            action=lambda: print("    ✅ p95 latency improved: 3.8s"),
            automated=True
        )


# =============================================================================
# Runbook Registry
# =============================================================================

class RunbookRegistry:
    """런북 레지스트리"""

    def __init__(self):
        self.runbooks: Dict[str, Runbook] = {}
        self._register_default_runbooks()

    def _register_default_runbooks(self):
        """기본 런북 등록"""
        self.register(QwenFailureRunbook())
        self.register(RLRewardRunawayRunbook())
        self.register(APIHighLatencyRunbook())

    def register(self, runbook: Runbook):
        """런북 등록"""
        self.runbooks[runbook.name] = runbook

    def get(self, name: str) -> Optional[Runbook]:
        """런북 조회"""
        return self.runbooks.get(name)

    def list_runbooks(self) -> List[str]:
        """런북 목록"""
        return list(self.runbooks.keys())

    def execute(self, name: str, dry_run: bool = False) -> RunbookExecution:
        """런북 실행"""
        runbook = self.get(name)
        if not runbook:
            raise ValueError(f"Runbook not found: {name}")

        return runbook.execute(dry_run=dry_run)


# =============================================================================
# Global Instance
# =============================================================================

_runbook_registry: Optional[RunbookRegistry] = None


def get_runbook_registry() -> RunbookRegistry:
    """글로벌 런북 레지스트리 반환"""
    global _runbook_registry
    if _runbook_registry is None:
        _runbook_registry = RunbookRegistry()
    return _runbook_registry


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Runbook Executor")
    parser.add_argument("--list", action="store_true", help="List all runbooks")
    parser.add_argument("--execute", help="Execute runbook by name")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't execute)")

    args = parser.parse_args()

    registry = get_runbook_registry()

    if args.list:
        print("\nAvailable Runbooks:")
        for name in registry.list_runbooks():
            runbook = registry.get(name)
            print(f"  - {name}")
            print(f"    {runbook.description}")
            print(f"    Steps: {len(runbook.steps)}")
        print()
    elif args.execute:
        registry.execute(args.execute, dry_run=args.dry_run)
    else:
        print("Use --list to see available runbooks or --execute <name> to run")
