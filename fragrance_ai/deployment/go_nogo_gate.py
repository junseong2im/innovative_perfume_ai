"""
Go / No-Go 배포 게이트
배포 전 자동 체크로 안전성 보장
"""

import logging
import subprocess
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class DeploymentDecision(Enum):
    """배포 결정"""
    GO = "GO"
    NO_GO = "NO_GO"


@dataclass
class GoNoGoResult:
    """Go/No-Go 결과"""
    decision: DeploymentDecision
    reasons: List[str]
    metrics: Dict[str, any]
    timestamp: str


# =============================================================================
# 1. 테스트 실행 및 결과 수집
# =============================================================================

class TestRunner:
    """테스트 실행기"""

    def __init__(self, test_suites: List[str]):
        self.test_suites = test_suites

    def run_all_tests(self) -> Tuple[bool, Dict[str, any]]:
        """모든 테스트 실행"""
        logger.info("🧪 Running all test suites...")

        all_passed = True
        results = {}

        for suite in self.test_suites:
            passed, stats = self._run_test_suite(suite)
            results[suite] = stats

            if not passed:
                all_passed = False
                logger.error(f"❌ Test suite failed: {suite}")
            else:
                logger.info(f"✓ Test suite passed: {suite}")

        return all_passed, results

    def _run_test_suite(self, suite: str) -> Tuple[bool, Dict[str, any]]:
        """개별 테스트 스위트 실행"""
        try:
            # pytest 실행
            result = subprocess.run(
                ["pytest", suite, "-v", "--tb=short", "--json-report", "--json-report-file=.test_report.json"],
                capture_output=True,
                text=True,
                timeout=300
            )

            # 결과 파싱
            report_path = Path(".test_report.json")
            if report_path.exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)

                stats = {
                    "passed": report.get("summary", {}).get("passed", 0),
                    "failed": report.get("summary", {}).get("failed", 0),
                    "total": report.get("summary", {}).get("total", 0),
                    "duration": report.get("duration", 0.0)
                }

                report_path.unlink()  # Clean up
                return stats["failed"] == 0, stats

            # Fallback: return code 기반
            return result.returncode == 0, {
                "passed": "unknown",
                "failed": 0 if result.returncode == 0 else 1,
                "total": 1,
                "duration": 0.0
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Test suite timed out: {suite}")
            return False, {"failed": 1, "reason": "timeout"}
        except Exception as e:
            logger.error(f"Test suite error: {suite} - {e}")
            return False, {"failed": 1, "reason": str(e)}


# =============================================================================
# 2. KPI 메트릭 체크
# =============================================================================

@dataclass
class KPIThresholds:
    """KPI 임계값"""

    # LLM 메트릭
    llm_p95_fast: float = 2.5  # seconds
    llm_p95_balanced: float = 3.2  # seconds
    llm_p95_creative: float = 4.5  # seconds

    # API 메트릭
    api_p95_latency: float = 2.5  # seconds
    api_p99_latency: float = 5.0  # seconds
    api_error_rate: float = 0.01  # 1%

    # 스키마 검증
    schema_failure_rate: float = 0.0  # 0% (무관용)

    # RL 메트릭
    rl_reward_min: float = 10.0
    rl_kl_divergence_max: float = 0.03


class KPIChecker:
    """KPI 체크"""

    def __init__(self, thresholds: KPIThresholds):
        self.thresholds = thresholds

    def check_all_kpis(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """모든 KPI 체크"""
        logger.info("📊 Checking KPIs...")

        violations = []

        # LLM p95 체크
        if metrics.get("llm_p95_fast", 0) > self.thresholds.llm_p95_fast:
            violations.append(
                f"LLM p95 (fast) exceeded: {metrics['llm_p95_fast']:.2f}s > {self.thresholds.llm_p95_fast}s"
            )

        if metrics.get("llm_p95_balanced", 0) > self.thresholds.llm_p95_balanced:
            violations.append(
                f"LLM p95 (balanced) exceeded: {metrics['llm_p95_balanced']:.2f}s > {self.thresholds.llm_p95_balanced}s"
            )

        if metrics.get("llm_p95_creative", 0) > self.thresholds.llm_p95_creative:
            violations.append(
                f"LLM p95 (creative) exceeded: {metrics['llm_p95_creative']:.2f}s > {self.thresholds.llm_p95_creative}s"
            )

        # API 지연 체크
        if metrics.get("api_p95_latency", 0) > self.thresholds.api_p95_latency:
            violations.append(
                f"API p95 latency exceeded: {metrics['api_p95_latency']:.2f}s > {self.thresholds.api_p95_latency}s"
            )

        if metrics.get("api_p99_latency", 0) > self.thresholds.api_p99_latency:
            violations.append(
                f"API p99 latency exceeded: {metrics['api_p99_latency']:.2f}s > {self.thresholds.api_p99_latency}s"
            )

        # API 에러율 체크
        if metrics.get("api_error_rate", 0) > self.thresholds.api_error_rate:
            violations.append(
                f"API error rate exceeded: {metrics['api_error_rate']:.2%} > {self.thresholds.api_error_rate:.2%}"
            )

        # 스키마 실패율 체크 (무관용)
        if metrics.get("schema_failure_rate", 0) > self.thresholds.schema_failure_rate:
            violations.append(
                f"🚨 Schema failure rate > 0%: {metrics['schema_failure_rate']:.2%} (ZERO TOLERANCE)"
            )

        # RL 메트릭 체크
        if metrics.get("rl_reward", 0) < self.thresholds.rl_reward_min:
            violations.append(
                f"RL reward too low: {metrics['rl_reward']:.2f} < {self.thresholds.rl_reward_min}"
            )

        if metrics.get("rl_kl_divergence", 0) > self.thresholds.rl_kl_divergence_max:
            violations.append(
                f"RL KL divergence too high: {metrics['rl_kl_divergence']:.4f} > {self.thresholds.rl_kl_divergence_max}"
            )

        if violations:
            for v in violations:
                logger.error(f"❌ KPI violation: {v}")
            return False, violations
        else:
            logger.info("✓ All KPIs passed")
            return True, []


# =============================================================================
# 3. 메트릭 수집 (Prometheus 쿼리)
# =============================================================================

class MetricsCollector:
    """Prometheus 메트릭 수집"""

    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url

    def collect_metrics(self) -> Dict[str, float]:
        """메트릭 수집"""
        logger.info("📈 Collecting metrics from Prometheus...")

        metrics = {}

        # LLM p95 지연 (fast/balanced/creative)
        metrics["llm_p95_fast"] = self._query_prometheus(
            'histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode="fast"}[5m]))'
        )
        metrics["llm_p95_balanced"] = self._query_prometheus(
            'histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode="balanced"}[5m]))'
        )
        metrics["llm_p95_creative"] = self._query_prometheus(
            'histogram_quantile(0.95, rate(llm_brief_latency_seconds_bucket{mode="creative"}[5m]))'
        )

        # API 지연
        metrics["api_p95_latency"] = self._query_prometheus(
            'histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))'
        )
        metrics["api_p99_latency"] = self._query_prometheus(
            'histogram_quantile(0.99, rate(api_request_latency_seconds_bucket[5m]))'
        )

        # API 에러율
        total_requests = self._query_prometheus('sum(rate(api_request_total[5m]))')
        error_requests = self._query_prometheus('sum(rate(api_request_total{status=~"4xx|5xx"}[5m]))')
        metrics["api_error_rate"] = error_requests / total_requests if total_requests > 0 else 0

        # 스키마 실패율
        total_briefs = self._query_prometheus('sum(rate(llm_brief_total[5m]))')
        failed_briefs = self._query_prometheus('sum(rate(llm_brief_total{status="failure"}[5m]))')
        metrics["schema_failure_rate"] = failed_briefs / total_briefs if total_briefs > 0 else 0

        # RL 메트릭
        metrics["rl_reward"] = self._query_prometheus('rl_reward')
        metrics["rl_kl_divergence"] = self._query_prometheus('rl_kl_divergence')

        return metrics

    def _query_prometheus(self, query: str) -> float:
        """Prometheus 쿼리 (실제 구현 시 requests 사용)"""
        # Mock implementation - 실제로는 requests.get() 사용
        try:
            import requests
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query},
                timeout=5
            )
            data = response.json()
            if data.get("status") == "success":
                result = data.get("data", {}).get("result", [])
                if result:
                    return float(result[0]["value"][1])
        except Exception as e:
            logger.debug(f"Prometheus query failed: {query} - {e}")

        # Fallback to mock data
        return 0.0


# =============================================================================
# 4. Go / No-Go 게이트
# =============================================================================

class GoNoGoGate:
    """Go / No-Go 배포 게이트"""

    def __init__(
        self,
        test_suites: Optional[List[str]] = None,
        kpi_thresholds: Optional[KPIThresholds] = None,
        prometheus_url: str = "http://localhost:9090"
    ):
        # 기본 테스트 스위트
        if test_suites is None:
            test_suites = [
                "tests/test_llm_ensemble_operation.py",
                "tests/test_ga.py",
                "tests/test_ifra.py",
                "tests/test_end_to_end_evolution.py"
            ]

        self.test_runner = TestRunner(test_suites)
        self.kpi_checker = KPIChecker(kpi_thresholds or KPIThresholds())
        self.metrics_collector = MetricsCollector(prometheus_url)

    def evaluate(self) -> GoNoGoResult:
        """Go / No-Go 평가"""
        logger.info("=" * 60)
        logger.info("🚦 GO / NO-GO DEPLOYMENT GATE")
        logger.info("=" * 60)

        reasons = []
        metrics = {}
        decision = DeploymentDecision.GO

        # 1. 테스트 실행
        tests_passed, test_results = self.test_runner.run_all_tests()
        metrics["test_results"] = test_results

        if not tests_passed:
            decision = DeploymentDecision.NO_GO
            reasons.append("❌ Tests failed")
            logger.error("NO-GO: Tests failed")
        else:
            logger.info("✓ All tests passed")

        # 2. KPI 체크
        current_metrics = self.metrics_collector.collect_metrics()
        metrics.update(current_metrics)

        kpis_passed, kpi_violations = self.kpi_checker.check_all_kpis(current_metrics)

        if not kpis_passed:
            decision = DeploymentDecision.NO_GO
            reasons.extend(kpi_violations)
            logger.error("NO-GO: KPI violations detected")
        else:
            logger.info("✓ All KPIs passed")

        # 3. 특별 체크: 스키마 실패율 무관용
        if current_metrics.get("schema_failure_rate", 0) > 0:
            decision = DeploymentDecision.NO_GO
            reasons.append("🚨 CRITICAL: Schema failure rate > 0% (ZERO TOLERANCE)")
            logger.error("NO-GO: Schema failures detected (zero tolerance)")

        # 4. 특별 체크: p95 지속 초과
        p95_violations = [
            r for r in reasons
            if "p95" in r.lower() or "p99" in r.lower()
        ]
        if p95_violations:
            decision = DeploymentDecision.NO_GO
            logger.error("NO-GO: p95/p99 thresholds exceeded")

        # 결과 요약
        logger.info("=" * 60)
        if decision == DeploymentDecision.GO:
            logger.info("✅ DECISION: GO - Safe to deploy")
        else:
            logger.error("⛔ DECISION: NO-GO - Do not deploy")
            logger.error("Reasons:")
            for reason in reasons:
                logger.error(f"  - {reason}")

        logger.info("=" * 60)

        return GoNoGoResult(
            decision=decision,
            reasons=reasons,
            metrics=metrics,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def generate_report(self, result: GoNoGoResult) -> str:
        """배포 게이트 리포트 생성"""
        report = []
        report.append("=" * 60)
        report.append("GO / NO-GO DEPLOYMENT GATE REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Decision: {result.decision.value}")
        report.append("")

        if result.decision == DeploymentDecision.GO:
            report.append("✅ SAFE TO DEPLOY")
            report.append("")
            report.append("All checks passed:")
            report.append("  ✓ Tests: PASSED")
            report.append("  ✓ KPIs: PASSED")
            report.append("  ✓ Schema: PASSED (0% failure)")
            report.append("  ✓ Latency: PASSED (p95/p99 within limits)")
        else:
            report.append("⛔ DO NOT DEPLOY")
            report.append("")
            report.append("Failures detected:")
            for i, reason in enumerate(result.reasons, 1):
                report.append(f"  {i}. {reason}")

        report.append("")
        report.append("Metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, dict):
                report.append(f"  {key}:")
                for k, v in value.items():
                    report.append(f"    {k}: {v}")
            else:
                report.append(f"  {key}: {value}")

        report.append("=" * 60)

        return "\n".join(report)


# =============================================================================
# CLI 실행
# =============================================================================

def main():
    """CLI 엔트리포인트"""
    import argparse

    parser = argparse.ArgumentParser(description="Go / No-Go Deployment Gate")
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus server URL"
    )
    parser.add_argument(
        "--report-file",
        help="Save report to file"
    )
    parser.add_argument(
        "--exit-code",
        action="store_true",
        help="Exit with code 1 on NO-GO"
    )

    args = parser.parse_args()

    # Logging 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Go/No-Go 게이트 실행
    gate = GoNoGoGate(prometheus_url=args.prometheus_url)
    result = gate.evaluate()

    # 리포트 생성
    report = gate.generate_report(result)
    print(report)

    # 파일 저장
    if args.report_file:
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.report_file}")

    # Exit code
    if args.exit_code and result.decision == DeploymentDecision.NO_GO:
        return 1

    return 0


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.exit(main())
