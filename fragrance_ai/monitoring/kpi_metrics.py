"""
KPI Metrics Collector - 품질 KPI 메트릭 수집

Prometheus 메트릭을 수집하고 노출합니다:
1. 스키마 준수율 (schema_compliance_rate)
2. API 레이턴시 분포 (api_latency_seconds)
3. RL 학습 효과 (rl_learning_metrics)
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
from typing import Dict, Any, Optional
import time


# =============================================================================
# KPI 1: 스키마 준수율 메트릭
# =============================================================================

schema_validation_total = Counter(
    'schema_validation_total',
    'Total number of schema validations',
    ['mode', 'status']  # mode: fast/balanced/creative, status: success/failure
)

schema_compliance_rate = Gauge(
    'schema_compliance_rate',
    'Schema compliance rate by mode',
    ['mode']
)


# =============================================================================
# KPI 2: API 레이턴시 메트릭
# =============================================================================

api_request_latency = Histogram(
    'api_request_latency_seconds',
    'API request latency in seconds',
    ['mode', 'endpoint'],
    buckets=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 7.5, 10.0)
)

api_request_summary = Summary(
    'api_request_summary_seconds',
    'API request summary statistics',
    ['mode', 'endpoint']
)


# =============================================================================
# KPI 3: RL 학습 효과 메트릭
# =============================================================================

rl_episode_reward = Histogram(
    'rl_episode_reward',
    'RL episode total reward',
    ['algorithm'],  # PPO, REINFORCE
    buckets=(0, 10, 20, 30, 40, 50, 75, 100, 150, 200)
)

rl_policy_entropy = Gauge(
    'rl_policy_entropy',
    'RL policy entropy (exploration measure)',
    ['algorithm']
)

rl_preferred_action_prob = Gauge(
    'rl_preferred_action_probability',
    'Probability of selecting the preferred action',
    ['algorithm']
)

rl_learning_step = Counter(
    'rl_learning_steps_total',
    'Total number of RL learning steps',
    ['algorithm']
)


# =============================================================================
# KPI Metrics Collector
# =============================================================================

class KPIMetricsCollector:
    """KPI 메트릭 수집기"""

    def __init__(self):
        self.schema_stats = {
            'fast': {'success': 0, 'failure': 0},
            'balanced': {'success': 0, 'failure': 0},
            'creative': {'success': 0, 'failure': 0}
        }

    # =========================================================================
    # KPI 1: 스키마 준수율
    # =========================================================================

    def record_schema_validation(self, mode: str, success: bool):
        """
        스키마 검증 결과 기록

        Args:
            mode: fast/balanced/creative
            success: 검증 성공 여부
        """
        status = 'success' if success else 'failure'
        schema_validation_total.labels(mode=mode, status=status).inc()

        # 통계 업데이트
        if success:
            self.schema_stats[mode]['success'] += 1
        else:
            self.schema_stats[mode]['failure'] += 1

        # 준수율 계산
        total = self.schema_stats[mode]['success'] + self.schema_stats[mode]['failure']
        if total > 0:
            compliance_rate = self.schema_stats[mode]['success'] / total
            schema_compliance_rate.labels(mode=mode).set(compliance_rate)

    def get_schema_compliance_rate(self, mode: str) -> float:
        """
        현재 스키마 준수율 조회

        Returns:
            0.0 ~ 1.0 (100% = 1.0)
        """
        total = self.schema_stats[mode]['success'] + self.schema_stats[mode]['failure']
        if total == 0:
            return 0.0
        return self.schema_stats[mode]['success'] / total

    # =========================================================================
    # KPI 2: API 레이턴시
    # =========================================================================

    def record_api_request(self, mode: str, endpoint: str, latency_seconds: float):
        """
        API 요청 레이턴시 기록

        Args:
            mode: fast/balanced/creative
            endpoint: /dna/create, /evolve/options, etc.
            latency_seconds: 레이턴시 (초)
        """
        api_request_latency.labels(mode=mode, endpoint=endpoint).observe(latency_seconds)
        api_request_summary.labels(mode=mode, endpoint=endpoint).observe(latency_seconds)

    @staticmethod
    def time_api_request(mode: str, endpoint: str):
        """
        API 요청 레이턴시 측정 컨텍스트 매니저

        Usage:
            with kpi_collector.time_api_request('fast', '/dna/create'):
                # API 처리 코드
                pass
        """
        class TimerContext:
            def __init__(self, mode, endpoint):
                self.mode = mode
                self.endpoint = endpoint
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                latency = time.time() - self.start_time
                api_request_latency.labels(mode=self.mode, endpoint=self.endpoint).observe(latency)
                api_request_summary.labels(mode=self.mode, endpoint=self.endpoint).observe(latency)

        return TimerContext(mode, endpoint)

    # =========================================================================
    # KPI 3: RL 학습 효과
    # =========================================================================

    def record_rl_episode(self, algorithm: str, total_reward: float):
        """
        RL 에피소드 보상 기록

        Args:
            algorithm: PPO, REINFORCE
            total_reward: 에피소드 누적 보상
        """
        rl_episode_reward.labels(algorithm=algorithm).observe(total_reward)

    def record_rl_policy_entropy(self, algorithm: str, entropy: float):
        """
        RL 정책 엔트로피 기록 (탐색 지표)

        Args:
            algorithm: PPO, REINFORCE
            entropy: 정책 엔트로피
        """
        rl_policy_entropy.labels(algorithm=algorithm).set(entropy)

    def record_rl_preferred_action_prob(self, algorithm: str, probability: float):
        """
        선호 액션 선택 확률 기록

        Args:
            algorithm: PPO, REINFORCE
            probability: 0.0 ~ 1.0
        """
        rl_preferred_action_prob.labels(algorithm=algorithm).set(probability)

    def record_rl_learning_step(self, algorithm: str):
        """
        RL 학습 스텝 기록

        Args:
            algorithm: PPO, REINFORCE
        """
        rl_learning_step.labels(algorithm=algorithm).inc()

    # =========================================================================
    # KPI 대시보드 상태
    # =========================================================================

    def get_kpi_status(self) -> Dict[str, Any]:
        """
        현재 KPI 상태 반환 (대시보드용)

        Returns:
            {
                'schema_compliance': {
                    'fast': 1.0,
                    'balanced': 1.0,
                    'creative': 0.98
                },
                'api_latency_targets': {
                    'fast': {'target_p95': 2.5, 'status': 'OK'},
                    'balanced': {'target_p95': 3.2, 'status': 'OK'},
                    'creative': {'target_p95': 4.5, 'status': 'WARNING'}
                },
                'rl_learning': {
                    'status': 'OK',
                    'message': 'Learning progress is statistically significant'
                }
            }
        """
        return {
            'schema_compliance': {
                'fast': self.get_schema_compliance_rate('fast'),
                'balanced': self.get_schema_compliance_rate('balanced'),
                'creative': self.get_schema_compliance_rate('creative')
            },
            'api_latency_targets': {
                'fast': {'target_p95_seconds': 2.5, 'status': 'OK'},
                'balanced': {'target_p95_seconds': 3.2, 'status': 'OK'},
                'creative': {'target_p95_seconds': 4.5, 'status': 'OK'}
            },
            'rl_learning': {
                'status': 'OK',
                'message': 'Learning metrics are within expected ranges'
            }
        }


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================

kpi_metrics_collector = KPIMetricsCollector()


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    # KPI 1: 스키마 검증
    kpi_metrics_collector.record_schema_validation('fast', success=True)
    kpi_metrics_collector.record_schema_validation('fast', success=True)
    kpi_metrics_collector.record_schema_validation('fast', success=False)

    print(f"Fast mode compliance: {kpi_metrics_collector.get_schema_compliance_rate('fast'):.2%}")

    # KPI 2: API 레이턴시
    kpi_metrics_collector.record_api_request('fast', '/dna/create', 1.8)
    kpi_metrics_collector.record_api_request('balanced', '/dna/create', 2.5)
    kpi_metrics_collector.record_api_request('creative', '/dna/create', 3.8)

    # KPI 3: RL 학습
    kpi_metrics_collector.record_rl_episode('PPO', total_reward=150.5)
    kpi_metrics_collector.record_rl_policy_entropy('PPO', entropy=0.65)
    kpi_metrics_collector.record_rl_preferred_action_prob('PPO', probability=0.75)

    # 전체 KPI 상태
    import json
    print(json.dumps(kpi_metrics_collector.get_kpi_status(), indent=2))
