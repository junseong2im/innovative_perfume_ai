"""
Go / No-Go 게이트 시뮬레이션
GO와 NO-GO 시나리오 데모
"""

from fragrance_ai.deployment.go_nogo_gate import (
    GoNoGoGate,
    KPIThresholds,
    DeploymentDecision,
    GoNoGoResult
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def simulate_go_scenario():
    """GO 시나리오: 모든 체크 통과"""
    print("\n" + "=" * 60)
    print("시나리오 1: GO (모든 체크 통과)")
    print("=" * 60)

    # Mock metrics - 모두 정상
    mock_metrics = {
        "llm_p95_fast": 2.1,
        "llm_p95_balanced": 2.8,
        "llm_p95_creative": 4.2,
        "api_p95_latency": 2.3,
        "api_p99_latency": 4.5,
        "api_error_rate": 0.005,  # 0.5%
        "schema_failure_rate": 0.0,  # Perfect!
        "rl_reward": 12.5,
        "rl_kl_divergence": 0.015
    }

    # 결과 생성
    result = GoNoGoResult(
        decision=DeploymentDecision.GO,
        reasons=[],
        metrics=mock_metrics,
        timestamp="2025-10-14 12:34:56"
    )

    # 리포트 출력
    gate = GoNoGoGate()
    report = gate.generate_report(result)
    print(report)


def simulate_nogo_scenario_tests():
    """NO-GO 시나리오 1: 테스트 실패"""
    print("\n" + "=" * 60)
    print("시나리오 2: NO-GO (테스트 실패)")
    print("=" * 60)

    mock_metrics = {
        "llm_p95_fast": 2.1,
        "llm_p95_balanced": 2.8,
        "llm_p95_creative": 4.2,
        "api_p95_latency": 2.3,
        "api_p99_latency": 4.5,
        "api_error_rate": 0.005,
        "schema_failure_rate": 0.0,
        "rl_reward": 12.5,
        "rl_kl_divergence": 0.015,
        "test_results": {
            "test_llm_ensemble_operation.py": {"passed": 9, "failed": 0, "total": 9},
            "test_ga.py": {"passed": 9, "failed": 2, "total": 11},  # Failed!
            "test_ifra.py": {"passed": 45, "failed": 0, "total": 45},
            "test_end_to_end_evolution.py": {"passed": 9, "failed": 0, "total": 9}
        }
    }

    result = GoNoGoResult(
        decision=DeploymentDecision.NO_GO,
        reasons=["❌ Tests failed: test_ga.py (9/11 passed, 2 failed)"],
        metrics=mock_metrics,
        timestamp="2025-10-14 12:45:30"
    )

    gate = GoNoGoGate()
    report = gate.generate_report(result)
    print(report)


def simulate_nogo_scenario_schema():
    """NO-GO 시나리오 2: 스키마 실패 (무관용)"""
    print("\n" + "=" * 60)
    print("시나리오 3: NO-GO (스키마 실패 - 무관용)")
    print("=" * 60)

    mock_metrics = {
        "llm_p95_fast": 2.1,
        "llm_p95_balanced": 2.8,
        "llm_p95_creative": 4.2,
        "api_p95_latency": 2.3,
        "api_p99_latency": 4.5,
        "api_error_rate": 0.005,
        "schema_failure_rate": 0.02,  # 0.02% - CRITICAL!
        "rl_reward": 12.5,
        "rl_kl_divergence": 0.015
    }

    result = GoNoGoResult(
        decision=DeploymentDecision.NO_GO,
        reasons=[
            "🚨 CRITICAL: Schema failure rate > 0% (ZERO TOLERANCE)",
            "Schema failures detected: 0.02%"
        ],
        metrics=mock_metrics,
        timestamp="2025-10-14 13:15:42"
    )

    gate = GoNoGoGate()
    report = gate.generate_report(result)
    print(report)


def simulate_nogo_scenario_p95():
    """NO-GO 시나리오 3: p95 초과 지속"""
    print("\n" + "=" * 60)
    print("시나리오 4: NO-GO (p95 초과 지속)")
    print("=" * 60)

    mock_metrics = {
        "llm_p95_fast": 2.1,
        "llm_p95_balanced": 2.8,
        "llm_p95_creative": 5.2,  # Exceeded!
        "api_p95_latency": 2.8,  # Exceeded!
        "api_p99_latency": 5.5,  # Exceeded!
        "api_error_rate": 0.005,
        "schema_failure_rate": 0.0,
        "rl_reward": 12.5,
        "rl_kl_divergence": 0.015
    }

    result = GoNoGoResult(
        decision=DeploymentDecision.NO_GO,
        reasons=[
            "LLM p95 (creative) exceeded: 5.2s > 4.5s",
            "API p95 latency exceeded: 2.8s > 2.5s",
            "API p99 latency exceeded: 5.5s > 5.0s"
        ],
        metrics=mock_metrics,
        timestamp="2025-10-14 13:30:15"
    )

    gate = GoNoGoGate()
    report = gate.generate_report(result)
    print(report)


def simulate_nogo_scenario_multiple():
    """NO-GO 시나리오 4: 복합 실패"""
    print("\n" + "=" * 60)
    print("시나리오 5: NO-GO (복합 실패)")
    print("=" * 60)

    mock_metrics = {
        "llm_p95_fast": 2.1,
        "llm_p95_balanced": 2.8,
        "llm_p95_creative": 5.2,  # Exceeded!
        "api_p95_latency": 2.3,
        "api_p99_latency": 4.5,
        "api_error_rate": 0.015,  # Exceeded (1.5%)
        "schema_failure_rate": 0.01,  # CRITICAL!
        "rl_reward": 8.5,  # Too low!
        "rl_kl_divergence": 0.035,  # Too high!
        "test_results": {
            "test_llm_ensemble_operation.py": {"passed": 9, "failed": 0, "total": 9},
            "test_ga.py": {"passed": 9, "failed": 2, "total": 11},  # Failed!
            "test_ifra.py": {"passed": 43, "failed": 2, "total": 45},  # Failed!
            "test_end_to_end_evolution.py": {"passed": 9, "failed": 0, "total": 9}
        }
    }

    result = GoNoGoResult(
        decision=DeploymentDecision.NO_GO,
        reasons=[
            "❌ Tests failed: test_ga.py, test_ifra.py",
            "LLM p95 (creative) exceeded: 5.2s > 4.5s",
            "API error rate exceeded: 1.5% > 1.0%",
            "🚨 CRITICAL: Schema failure rate > 0% (ZERO TOLERANCE)",
            "RL reward too low: 8.5 < 10.0",
            "RL KL divergence too high: 0.035 > 0.03"
        ],
        metrics=mock_metrics,
        timestamp="2025-10-14 14:00:00"
    )

    gate = GoNoGoGate()
    report = gate.generate_report(result)
    print(report)


def main():
    """시뮬레이션 실행"""
    print("\n")
    print("*" * 60)
    print("Go / No-Go 배포 게이트 시뮬레이션")
    print("*" * 60)

    # 1. GO 시나리오
    simulate_go_scenario()

    # 2. NO-GO 시나리오들
    simulate_nogo_scenario_tests()
    simulate_nogo_scenario_schema()
    simulate_nogo_scenario_p95()
    simulate_nogo_scenario_multiple()

    print("\n" + "=" * 60)
    print("시뮬레이션 완료")
    print("=" * 60)
    print("")
    print("📝 요약:")
    print("  ✅ GO: 모든 테스트 통과 + KPI 달성")
    print("  ⛔ NO-GO: 스키마 실패 >0% OR p95 초과 지속")
    print("")


if __name__ == "__main__":
    main()
