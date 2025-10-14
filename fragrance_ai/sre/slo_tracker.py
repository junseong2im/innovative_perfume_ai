"""
SLO (Service Level Objective) & Error Budget Tracking
에러버짓 소진 시 신규 기능 중단, 품질 개선 우선
"""

import time
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class SLOStatus(str, Enum):
    """SLO 상태"""
    HEALTHY = "healthy"  # 에러버짓 충분
    WARNING = "warning"  # 에러버짓 < 20%
    CRITICAL = "critical"  # 에러버짓 소진
    EXCEEDED = "exceeded"  # SLO 위반


@dataclass
class SLODefinition:
    """SLO 정의"""
    name: str
    description: str
    target: float  # 목표 (예: 0.999 = 99.9%)
    measurement_window: int  # 측정 기간 (초)
    error_budget: float  # 에러버짓 (1 - target)

    def __post_init__(self):
        if self.error_budget is None:
            self.error_budget = 1.0 - self.target


@dataclass
class SLOMetrics:
    """SLO 메트릭"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    actual_availability: float
    error_budget_consumed: float  # 0.0 ~ 1.0
    error_budget_remaining: float
    status: SLOStatus
    timestamp: str


class SLOTracker:
    """SLO 추적 및 에러버짓 관리"""

    # Artisan SLO 정의
    SLOS = {
        "api_availability": SLODefinition(
            name="API Availability",
            description="API 가용성 (모든 엔드포인트)",
            target=0.999,  # 99.9%
            measurement_window=30 * 24 * 3600,  # 30 days
            error_budget=0.001  # 0.1%
        ),
        "llm_fast_latency": SLODefinition(
            name="LLM Fast Mode Latency (p95)",
            description="LLM Fast 모드 p95 지연시간",
            target=0.95,  # p95 < 2.5s 달성률 95%
            measurement_window=7 * 24 * 3600,  # 7 days
            error_budget=0.05
        ),
        "llm_balanced_latency": SLODefinition(
            name="LLM Balanced Mode Latency (p95)",
            description="LLM Balanced 모드 p95 지연시간",
            target=0.95,  # p95 < 3.2s 달성률 95%
            measurement_window=7 * 24 * 3600,
            error_budget=0.05
        ),
        "llm_creative_latency": SLODefinition(
            name="LLM Creative Mode Latency (p95)",
            description="LLM Creative 모드 p95 지연시간 ≤ 4.5s",
            target=0.95,  # p95 < 4.5s 달성률 95%
            measurement_window=7 * 24 * 3600,
            error_budget=0.05
        ),
        "rl_reward_stability": SLODefinition(
            name="RL Reward Stability",
            description="RL 보상 안정성 (최소 임계값 유지)",
            target=0.99,  # 99% 시간 동안 reward > 10.0
            measurement_window=7 * 24 * 3600,
            error_budget=0.01
        ),
    }

    # p95 latency thresholds (milliseconds)
    LATENCY_THRESHOLDS = {
        "fast": 2500,
        "balanced": 3200,
        "creative": 4500
    }

    def __init__(self, data_file: str = "slo_metrics.json"):
        self.data_file = Path(data_file)
        self.metrics_history: Dict[str, List[Dict]] = {}
        self._load_history()

    def _load_history(self):
        """메트릭 히스토리 로드"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                self.metrics_history = json.load(f)
        else:
            self.metrics_history = {slo: [] for slo in self.SLOS.keys()}

    def _save_history(self):
        """메트릭 히스토리 저장"""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def record_api_request(self, success: bool):
        """API 요청 기록"""
        self._record_slo_event("api_availability", success)

    def record_llm_latency(self, mode: str, latency_ms: float):
        """LLM 지연시간 기록"""
        threshold = self.LATENCY_THRESHOLDS.get(mode)
        if threshold is None:
            return

        slo_key = f"llm_{mode}_latency"
        success = latency_ms <= threshold

        self._record_slo_event(slo_key, success)

    def record_rl_reward(self, reward: float):
        """RL 보상 기록"""
        success = reward >= 10.0
        self._record_slo_event("rl_reward_stability", success)

    def _record_slo_event(self, slo_key: str, success: bool):
        """SLO 이벤트 기록"""
        if slo_key not in self.metrics_history:
            self.metrics_history[slo_key] = []

        event = {
            "timestamp": datetime.now().isoformat(),
            "success": success
        }

        self.metrics_history[slo_key].append(event)

        # Keep only recent events within measurement window
        slo = self.SLOS[slo_key]
        cutoff_time = datetime.now() - timedelta(seconds=slo.measurement_window)

        self.metrics_history[slo_key] = [
            e for e in self.metrics_history[slo_key]
            if datetime.fromisoformat(e["timestamp"]) > cutoff_time
        ]

        self._save_history()

    def get_slo_metrics(self, slo_key: str) -> SLOMetrics:
        """SLO 메트릭 조회"""
        if slo_key not in self.SLOS:
            raise ValueError(f"Unknown SLO: {slo_key}")

        slo = self.SLOS[slo_key]
        events = self.metrics_history.get(slo_key, [])

        if not events:
            return SLOMetrics(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                actual_availability=0.0,
                error_budget_consumed=0.0,
                error_budget_remaining=1.0,
                status=SLOStatus.HEALTHY,
                timestamp=datetime.now().isoformat()
            )

        # Calculate metrics
        total = len(events)
        successful = sum(1 for e in events if e["success"])
        failed = total - successful

        actual_availability = successful / total if total > 0 else 0.0

        # Error budget calculation
        # Error budget = (1 - target) = allowed failure rate
        # Consumed = (actual_failures / total) / error_budget
        actual_error_rate = failed / total if total > 0 else 0.0
        error_budget_consumed = actual_error_rate / slo.error_budget if slo.error_budget > 0 else 0.0
        error_budget_remaining = max(0.0, 1.0 - error_budget_consumed)

        # Determine status
        if error_budget_consumed >= 1.0:
            status = SLOStatus.CRITICAL if actual_availability < slo.target else SLOStatus.EXCEEDED
        elif error_budget_remaining < 0.2:  # < 20% remaining
            status = SLOStatus.WARNING
        else:
            status = SLOStatus.HEALTHY

        return SLOMetrics(
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            actual_availability=actual_availability,
            error_budget_consumed=error_budget_consumed,
            error_budget_remaining=error_budget_remaining,
            status=status,
            timestamp=datetime.now().isoformat()
        )

    def get_all_slo_metrics(self) -> Dict[str, SLOMetrics]:
        """모든 SLO 메트릭 조회"""
        return {
            slo_key: self.get_slo_metrics(slo_key)
            for slo_key in self.SLOS.keys()
        }

    def is_error_budget_exhausted(self, slo_key: str) -> bool:
        """에러버짓 소진 여부"""
        metrics = self.get_slo_metrics(slo_key)
        return metrics.status in (SLOStatus.CRITICAL, SLOStatus.EXCEEDED)

    def should_block_new_features(self) -> Tuple[bool, List[str]]:
        """
        신규 기능 배포 차단 여부

        Returns:
            (should_block, violated_slos)
        """
        violated_slos = []

        for slo_key in self.SLOS.keys():
            if self.is_error_budget_exhausted(slo_key):
                violated_slos.append(slo_key)

        should_block = len(violated_slos) > 0

        return should_block, violated_slos

    def get_error_budget_report(self) -> Dict[str, Dict]:
        """에러버짓 리포트 생성"""
        report = {}

        for slo_key, slo in self.SLOS.items():
            metrics = self.get_slo_metrics(slo_key)

            report[slo_key] = {
                "slo_name": slo.name,
                "description": slo.description,
                "target": f"{slo.target * 100:.2f}%",
                "actual": f"{metrics.actual_availability * 100:.2f}%",
                "error_budget_consumed": f"{metrics.error_budget_consumed * 100:.2f}%",
                "error_budget_remaining": f"{metrics.error_budget_remaining * 100:.2f}%",
                "status": metrics.status,
                "total_requests": metrics.total_requests,
                "failed_requests": metrics.failed_requests,
                "measurement_window_days": slo.measurement_window / (24 * 3600)
            }

        return report

    def print_error_budget_report(self):
        """에러버짓 리포트 출력"""
        report = self.get_error_budget_report()

        print("=" * 80)
        print("SLO & Error Budget Report")
        print("=" * 80)
        print()

        for slo_key, data in report.items():
            status_emoji = {
                SLOStatus.HEALTHY: "✅",
                SLOStatus.WARNING: "⚠️",
                SLOStatus.CRITICAL: "🔴",
                SLOStatus.EXCEEDED: "❌"
            }.get(data["status"], "❓")

            print(f"{status_emoji} {data['slo_name']}")
            print(f"   Description:         {data['description']}")
            print(f"   Target:              {data['target']}")
            print(f"   Actual:              {data['actual']}")
            print(f"   Error Budget Used:   {data['error_budget_consumed']}")
            print(f"   Error Budget Left:   {data['error_budget_remaining']}")
            print(f"   Status:              {data['status']}")
            print(f"   Total Requests:      {data['total_requests']}")
            print(f"   Failed Requests:     {data['failed_requests']}")
            print(f"   Window:              {data['measurement_window_days']} days")
            print()

        # Check if new features should be blocked
        should_block, violated_slos = self.should_block_new_features()

        if should_block:
            print("=" * 80)
            print("🚨 ALERT: Error Budget Exhausted - New Features Blocked")
            print("=" * 80)
            print()
            print("Violated SLOs:")
            for slo in violated_slos:
                print(f"  - {self.SLOS[slo].name}")
            print()
            print("Action Required:")
            print("  1. Stop all new feature deployments")
            print("  2. Focus on quality improvements and bug fixes")
            print("  3. Investigate root causes of SLO violations")
            print("  4. Deploy fixes to restore SLO compliance")
            print()
        else:
            print("=" * 80)
            print("✅ Status: All SLOs within error budget - Deployments allowed")
            print("=" * 80)
            print()


# =============================================================================
# Global Instance
# =============================================================================

_slo_tracker: Optional[SLOTracker] = None


def get_slo_tracker() -> SLOTracker:
    """글로벌 SLO 트래커 반환"""
    global _slo_tracker
    if _slo_tracker is None:
        _slo_tracker = SLOTracker()
    return _slo_tracker


# =============================================================================
# Convenience Functions
# =============================================================================

def record_api_request(success: bool):
    """API 요청 기록"""
    get_slo_tracker().record_api_request(success)


def record_llm_latency(mode: str, latency_ms: float):
    """LLM 지연시간 기록"""
    get_slo_tracker().record_llm_latency(mode, latency_ms)


def record_rl_reward(reward: float):
    """RL 보상 기록"""
    get_slo_tracker().record_rl_reward(reward)


def check_error_budget() -> Tuple[bool, List[str]]:
    """에러버짓 확인"""
    return get_slo_tracker().should_block_new_features()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SLO & Error Budget Tracker")
    parser.add_argument("--report", action="store_true", help="Show error budget report")
    parser.add_argument("--check", action="store_true", help="Check if deployments should be blocked")

    args = parser.parse_args()

    tracker = get_slo_tracker()

    if args.report:
        tracker.print_error_budget_report()
    elif args.check:
        should_block, violated = tracker.should_block_new_features()
        if should_block:
            print(f"🚨 Deployments blocked due to SLO violations: {', '.join(violated)}")
            exit(1)
        else:
            print("✅ Deployments allowed - All SLOs within error budget")
            exit(0)
    else:
        # Default: show report
        tracker.print_error_budget_report()
