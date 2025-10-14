"""
운영 대시보드 메트릭 (Operations Metrics)

- LLM Brief: 모드/수정건수/지연
- RL Update: loss/reward/entropy/clip_frac
- API: p95/p99 지연
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
from typing import Dict, Any, Optional
import time
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Brief 메트릭
# =============================================================================

# LLM Brief 생성 카운터
llm_brief_total = Counter(
    'llm_brief_total',
    'LLM Brief 생성 총 건수',
    ['mode', 'status']  # fast/balanced/creative, success/failure
)

# LLM Brief 수정 카운터
llm_brief_repairs_total = Counter(
    'llm_brief_repairs_total',
    'LLM Brief JSON 수정 총 건수',
    ['mode', 'repair_type']  # code_block/trailing_comma/single_quote/incomplete/escape/extract
)

# LLM 모델별 상세 지표
llm_model_status = Counter(
    'llm_model_status_total',
    'LLM 모델별 성공/실패 건수',
    ['model', 'status']  # qwen/mistral/llama, ok/error/timeout
)

# LLM 힌트 제공 카운터
llm_hints_total = Counter(
    'llm_hints_total',
    'LLM 힌트 제공 건수',
    ['mode', 'hint_type']  # creative_hint/quality_hint/compliance_hint
)

# LLM Mistral Schema Fix 카운터 (명시적)
llm_schema_fix_count = Counter(
    'llm_schema_fix_count_total',
    'Mistral로 JSON 스키마 수정한 건수',
    ['mode', 'original_model']  # qwen/llama로 생성 후 mistral로 수정
)

# LLM Creative Hints 길이 게이지
llm_creative_hints_len = Gauge(
    'llm_creative_hints_len',
    'Creative Hints 텍스트 길이 (과도/부족 검출용)',
    ['mode']
)

# LLM Brief 생성 지연 (히스토그램)
llm_brief_latency = Histogram(
    'llm_brief_latency_seconds',
    'LLM Brief 생성 지연 (초)',
    ['mode'],
    buckets=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
)

# LLM Brief 생성 지연 (서머리 - p95/p99 자동 계산)
llm_brief_latency_summary = Summary(
    'llm_brief_latency_summary_seconds',
    'LLM Brief 생성 지연 서머리 (p95, p99)',
    ['mode']
)


# =============================================================================
# RL Update 메트릭
# =============================================================================

# RL 업데이트 카운터
rl_update_total = Counter(
    'rl_update_total',
    'RL 업데이트 총 건수',
    ['algorithm']  # ppo/reinforce
)

# RL 손실 (Gauge)
rl_loss = Gauge(
    'rl_loss',
    'RL 손실 (최근 업데이트)',
    ['algorithm', 'loss_type']  # policy_loss/value_loss/total_loss
)

# RL 보상 (Gauge)
rl_reward = Gauge(
    'rl_reward',
    'RL 평균 보상 (최근 에피소드)',
    ['algorithm']
)

# RL 엔트로피 (Gauge)
rl_entropy = Gauge(
    'rl_entropy',
    'RL 정책 엔트로피 (최근 업데이트)',
    ['algorithm']
)

# RL Clip Fraction (Gauge)
rl_clip_fraction = Gauge(
    'rl_clip_fraction',
    'RL Clipping 비율 (PPO)',
    ['algorithm']
)

# RL KL Divergence (Gauge)
rl_kl_divergence = Gauge(
    'rl_kl_divergence',
    'RL KL Divergence (정책 변화)',
    ['algorithm']
)

# RL Reward 이동평균 (Gauge)
rl_reward_ma = Gauge(
    'rl_reward_ma',
    'RL 보상 이동평균 (최근 100 에피소드)',
    ['algorithm', 'window']  # window: 10/50/100
)

# RL 옵션 생성 실패율 (Gauge)
rl_option_generation_failure_rate = Gauge(
    'rl_option_generation_failure_rate',
    'RL 옵션 생성 실패율 (0% 유지 목표)',
    ['algorithm']
)

# RL 옵션 생성 실패 카운터
rl_option_generation_failures_total = Counter(
    'rl_option_generation_failures_total',
    'RL 옵션 생성 실패 총 건수',
    ['algorithm', 'error_type']  # validation_error/timeout/exception
)


# =============================================================================
# API 메트릭
# =============================================================================

# API 요청 카운터
api_request_total = Counter(
    'api_request_total',
    'API 요청 총 건수',
    ['endpoint', 'method', 'status']  # /dna/create, POST, 200/400/500
)

# API 요청 지연 (히스토그램)
api_request_latency = Histogram(
    'api_request_latency_seconds',
    'API 요청 지연 (초)',
    ['endpoint', 'method'],
    buckets=[0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
)

# API 요청 지연 (서머리 - p95/p99)
api_request_latency_summary = Summary(
    'api_request_latency_summary_seconds',
    'API 요청 지연 서머리 (p95, p99)',
    ['endpoint', 'method']
)


# =============================================================================
# 서킷브레이커 메트릭
# =============================================================================

# 서킷브레이커 폴백 카운터
circuit_breaker_fallback_total = Counter(
    'circuit_breaker_fallback_total',
    '서킷브레이커 폴백 전환 건수',
    ['service', 'fallback_type']  # llm/api/rl, degraded/cached/mock
)

# 서킷브레이커 다운그레이드 카운터
circuit_breaker_downgrade_total = Counter(
    'circuit_breaker_downgrade_total',
    '서킷브레이커 다운그레이드 건수',
    ['service', 'from_tier', 'to_tier']  # creative->balanced, balanced->fast
)

# 서킷브레이커 상태 게이지
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    '서킷브레이커 상태 (0=closed, 1=half_open, 2=open)',
    ['service']
)


# =============================================================================
# 캐시 메트릭 (확장)
# =============================================================================

# 캐시 히트 카운터
cache_hits_total = Counter(
    'cache_hits_total',
    '캐시 히트 총 건수',
    ['mode', 'cache_type']  # fast/balanced/creative, llm/perception/ensemble
)

# 캐시 미스 카운터
cache_misses_total = Counter(
    'cache_misses_total',
    '캐시 미스 총 건수',
    ['mode', 'cache_type']
)

# 캐시 히트율 게이지 (목표: ≥60% for fast/balanced)
cache_hit_rate = Gauge(
    'cache_hit_rate',
    '캐시 히트율 (fast/balanced 목표: ≥60%)',
    ['mode', 'cache_type']
)

# 캐시 TTL 만료 카운터
cache_ttl_expired_total = Counter(
    'cache_ttl_expired_total',
    '캐시 TTL 만료 건수',
    ['mode', 'cache_type']  # fast/balanced/creative, llm/perception/ensemble
)

# 캐시 크기 게이지
cache_size = Gauge(
    'cache_size',
    '현재 캐시 크기',
    ['cache_type']
)


# =============================================================================
# 운영 메트릭 컬렉터
# =============================================================================

class OperationsMetricsCollector:
    """
    운영 메트릭 수집기

    LLM Brief, RL Update, API 메트릭을 수집합니다.
    """

    def __init__(self):
        logger.info("운영 메트릭 컬렉터 초기화")

    # =========================================================================
    # LLM Brief 메트릭
    # =========================================================================

    def record_llm_brief(
        self,
        mode: str,
        success: bool,
        latency_seconds: float,
        repaired: bool = False,
        repair_type: Optional[str] = None,
        qwen_ok: Optional[bool] = None,
        mistral_fix: Optional[bool] = None,
        hints: Optional[str] = None,
        original_model: Optional[str] = None,
        hints_text_len: Optional[int] = None
    ):
        """
        LLM Brief 생성 기록

        Args:
            mode: fast/balanced/creative
            success: 성공 여부
            latency_seconds: 지연 (초)
            repaired: 수정 여부
            repair_type: 수정 타입 (code_block, trailing_comma, 등)
            qwen_ok: Qwen 성공 여부 (optional)
            mistral_fix: Mistral로 수정했는지 여부 (optional)
            hints: 제공된 힌트 타입 (optional)
            original_model: 원본 모델 (mistral_fix=True일 때 필요)
            hints_text_len: Creative hints 텍스트 길이 (optional)
        """
        # 카운터
        status = 'success' if success else 'failure'
        llm_brief_total.labels(mode=mode, status=status).inc()

        # 수정 카운터
        if repaired and repair_type:
            llm_brief_repairs_total.labels(mode=mode, repair_type=repair_type).inc()

        # 모델별 상세 지표
        if qwen_ok is not None:
            llm_model_status.labels(model='qwen', status='ok' if qwen_ok else 'error').inc()
        if mistral_fix:
            llm_model_status.labels(model='mistral', status='fix').inc()
            # Schema fix count (명시적)
            original = original_model or 'unknown'
            llm_schema_fix_count.labels(mode=mode, original_model=original).inc()

        # 힌트 제공
        if hints:
            llm_hints_total.labels(mode=mode, hint_type=hints).inc()

        # Creative hints 길이 (과도/부족 검출)
        if hints_text_len is not None:
            llm_creative_hints_len.labels(mode=mode).set(hints_text_len)

        # 지연
        llm_brief_latency.labels(mode=mode).observe(latency_seconds)
        llm_brief_latency_summary.labels(mode=mode).observe(latency_seconds)

        logger.debug(f"LLM Brief 기록: mode={mode}, success={success}, latency={latency_seconds:.3f}s")

    # =========================================================================
    # RL Update 메트릭
    # =========================================================================

    def record_rl_update(
        self,
        algorithm: str,
        policy_loss: float,
        value_loss: Optional[float] = None,
        total_loss: Optional[float] = None,
        reward: Optional[float] = None,
        entropy: Optional[float] = None,
        clip_fraction: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        reward_ma_10: Optional[float] = None,
        reward_ma_50: Optional[float] = None,
        reward_ma_100: Optional[float] = None
    ):
        """
        RL 업데이트 기록

        Args:
            algorithm: ppo/reinforce
            policy_loss: 정책 손실
            value_loss: 가치 손실 (optional)
            total_loss: 총 손실 (optional)
            reward: 평균 보상 (optional)
            entropy: 엔트로피 (optional)
            clip_fraction: Clipping 비율 (optional, PPO only)
            kl_divergence: KL Divergence (optional)
            reward_ma_10: 10 에피소드 이동평균 (optional)
            reward_ma_50: 50 에피소드 이동평균 (optional)
            reward_ma_100: 100 에피소드 이동평균 (optional)
        """
        # 카운터
        rl_update_total.labels(algorithm=algorithm).inc()

        # 손실
        rl_loss.labels(algorithm=algorithm, loss_type='policy_loss').set(policy_loss)
        if value_loss is not None:
            rl_loss.labels(algorithm=algorithm, loss_type='value_loss').set(value_loss)
        if total_loss is not None:
            rl_loss.labels(algorithm=algorithm, loss_type='total_loss').set(total_loss)

        # 보상
        if reward is not None:
            rl_reward.labels(algorithm=algorithm).set(reward)

        # 보상 이동평균
        if reward_ma_10 is not None:
            rl_reward_ma.labels(algorithm=algorithm, window='10').set(reward_ma_10)
        if reward_ma_50 is not None:
            rl_reward_ma.labels(algorithm=algorithm, window='50').set(reward_ma_50)
        if reward_ma_100 is not None:
            rl_reward_ma.labels(algorithm=algorithm, window='100').set(reward_ma_100)

        # 엔트로피
        if entropy is not None:
            rl_entropy.labels(algorithm=algorithm).set(entropy)

        # Clip fraction (PPO)
        if clip_fraction is not None:
            rl_clip_fraction.labels(algorithm=algorithm).set(clip_fraction)

        # KL divergence
        if kl_divergence is not None:
            rl_kl_divergence.labels(algorithm=algorithm).set(kl_divergence)

        logger.debug(f"RL Update 기록: algorithm={algorithm}, policy_loss={policy_loss:.4f}, reward={reward}")

    def record_rl_option_generation_failure(
        self,
        algorithm: str,
        error_type: str
    ):
        """
        RL 옵션 생성 실패 기록

        Args:
            algorithm: ppo/reinforce
            error_type: validation_error/timeout/exception
        """
        rl_option_generation_failures_total.labels(
            algorithm=algorithm,
            error_type=error_type
        ).inc()
        logger.warning(f"RL 옵션 생성 실패: algorithm={algorithm}, error_type={error_type}")

    def set_rl_option_generation_failure_rate(
        self,
        algorithm: str,
        failure_rate: float
    ):
        """
        RL 옵션 생성 실패율 설정 (0% 유지 목표)

        Args:
            algorithm: ppo/reinforce
            failure_rate: 0.0 ~ 1.0 (0% ~ 100%)
        """
        rl_option_generation_failure_rate.labels(algorithm=algorithm).set(failure_rate)
        if failure_rate > 0.0:
            logger.warning(f"RL 옵션 생성 실패율 상승: algorithm={algorithm}, rate={failure_rate*100:.2f}%")

    # =========================================================================
    # API 메트릭
    # =========================================================================

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_seconds: float
    ):
        """
        API 요청 기록

        Args:
            endpoint: /dna/create, /moga/optimize, 등
            method: GET/POST/PUT/DELETE
            status_code: 200/400/500
            latency_seconds: 지연 (초)
        """
        # 카운터
        status = f"{status_code // 100}xx"  # 2xx, 4xx, 5xx
        api_request_total.labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).inc()

        # 지연
        api_request_latency.labels(
            endpoint=endpoint,
            method=method
        ).observe(latency_seconds)

        api_request_latency_summary.labels(
            endpoint=endpoint,
            method=method
        ).observe(latency_seconds)

        logger.debug(f"API 요청 기록: {method} {endpoint} {status_code} {latency_seconds:.3f}s")

    # =========================================================================
    # 서킷브레이커 메트릭
    # =========================================================================

    def record_circuit_breaker_fallback(
        self,
        service: str,
        fallback_type: str
    ):
        """
        서킷브레이커 폴백 기록

        Args:
            service: llm/api/rl
            fallback_type: degraded/cached/mock
        """
        circuit_breaker_fallback_total.labels(
            service=service,
            fallback_type=fallback_type
        ).inc()
        logger.debug(f"서킷브레이커 폴백: {service} -> {fallback_type}")

    def record_circuit_breaker_downgrade(
        self,
        service: str,
        from_tier: str,
        to_tier: str
    ):
        """
        서킷브레이커 다운그레이드 기록

        Args:
            service: llm/api/rl
            from_tier: creative/balanced
            to_tier: balanced/fast
        """
        circuit_breaker_downgrade_total.labels(
            service=service,
            from_tier=from_tier,
            to_tier=to_tier
        ).inc()
        logger.info(f"서킷브레이커 다운그레이드: {service} {from_tier}->{to_tier}")

    def set_circuit_breaker_state(
        self,
        service: str,
        state: str
    ):
        """
        서킷브레이커 상태 설정

        Args:
            service: llm/api/rl
            state: closed/half_open/open
        """
        state_value = {'closed': 0, 'half_open': 1, 'open': 2}.get(state, 0)
        circuit_breaker_state.labels(service=service).set(state_value)
        logger.debug(f"서킷브레이커 상태: {service} = {state}")

    # =========================================================================
    # 캐시 메트릭
    # =========================================================================

    def record_cache_hit(
        self,
        mode: str,
        cache_type: str = 'llm'
    ):
        """
        캐시 히트 기록

        Args:
            mode: fast/balanced/creative
            cache_type: llm/perception/ensemble
        """
        cache_hits_total.labels(mode=mode, cache_type=cache_type).inc()

        # Calculate and update hit rate
        self._update_cache_hit_rate(mode, cache_type)

        logger.debug(f"캐시 히트: mode={mode}, type={cache_type}")

    def record_cache_miss(
        self,
        mode: str,
        cache_type: str = 'llm'
    ):
        """
        캐시 미스 기록

        Args:
            mode: fast/balanced/creative
            cache_type: llm/perception/ensemble
        """
        cache_misses_total.labels(mode=mode, cache_type=cache_type).inc()

        # Calculate and update hit rate
        self._update_cache_hit_rate(mode, cache_type)

        logger.debug(f"캐시 미스: mode={mode}, type={cache_type}")

    def _update_cache_hit_rate(
        self,
        mode: str,
        cache_type: str
    ):
        """
        캐시 히트율 계산 및 업데이트 (내부 메서드)

        Args:
            mode: fast/balanced/creative
            cache_type: llm/perception/ensemble
        """
        try:
            # Get current counts
            hits = cache_hits_total.labels(mode=mode, cache_type=cache_type)._value.get()
            misses = cache_misses_total.labels(mode=mode, cache_type=cache_type)._value.get()

            total = hits + misses
            if total > 0:
                hit_rate = hits / total
                cache_hit_rate.labels(mode=mode, cache_type=cache_type).set(hit_rate)

                # Log warning if below 60% target (except creative mode)
                if mode != 'creative' and hit_rate < 0.60 and total >= 100:
                    logger.warning(
                        f"캐시 히트율 낮음: mode={mode}, type={cache_type}, "
                        f"hit_rate={hit_rate*100:.1f}% (목표: ≥60%)"
                    )
        except Exception as e:
            logger.error(f"캐시 히트율 계산 오류: {e}")

    def record_cache_ttl_expired(
        self,
        mode: str,
        cache_type: str = 'llm'
    ):
        """
        캐시 TTL 만료 기록

        Args:
            mode: fast/balanced/creative
            cache_type: llm/perception/ensemble
        """
        cache_ttl_expired_total.labels(mode=mode, cache_type=cache_type).inc()
        logger.debug(f"캐시 TTL 만료: mode={mode}, type={cache_type}")

    def set_cache_size(
        self,
        cache_type: str,
        size: int
    ):
        """
        캐시 크기 설정

        Args:
            cache_type: llm/perception/ensemble
            size: 캐시 항목 수
        """
        cache_size.labels(cache_type=cache_type).set(size)

    def get_cache_hit_rate(
        self,
        mode: str,
        cache_type: str = 'llm'
    ) -> float:
        """
        현재 캐시 히트율 조회

        Args:
            mode: fast/balanced/creative
            cache_type: llm/perception/ensemble

        Returns:
            Hit rate (0.0 ~ 1.0)
        """
        try:
            hits = cache_hits_total.labels(mode=mode, cache_type=cache_type)._value.get()
            misses = cache_misses_total.labels(mode=mode, cache_type=cache_type)._value.get()

            total = hits + misses
            if total > 0:
                return hits / total
            return 0.0
        except Exception as e:
            logger.error(f"캐시 히트율 조회 오류: {e}")
            return 0.0

    # =========================================================================
    # 컨텍스트 매니저 (자동 지연 측정)
    # =========================================================================

    def track_llm_brief(self, mode: str):
        """
        LLM Brief 생성 추적 (context manager)

        Example:
            with collector.track_llm_brief("creative"):
                brief = generate_brief()
        """
        return _LLMBriefTracker(self, mode)

    def track_api_request(self, endpoint: str, method: str):
        """
        API 요청 추적 (context manager)

        Example:
            with collector.track_api_request("/dna/create", "POST"):
                response = handle_request()
        """
        return _APIRequestTracker(self, endpoint, method)


# =============================================================================
# Context Manager Helpers
# =============================================================================

class _LLMBriefTracker:
    """LLM Brief 생성 추적 헬퍼"""

    def __init__(self, collector: OperationsMetricsCollector, mode: str):
        self.collector = collector
        self.mode = mode
        self.start_time = None
        self.success = True
        self.repaired = False
        self.repair_type = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        success = exc_type is None and self.success

        self.collector.record_llm_brief(
            mode=self.mode,
            success=success,
            latency_seconds=latency,
            repaired=self.repaired,
            repair_type=self.repair_type
        )

    def mark_repaired(self, repair_type: str):
        """수정 마킹"""
        self.repaired = True
        self.repair_type = repair_type

    def mark_failed(self):
        """실패 마킹"""
        self.success = False


class _APIRequestTracker:
    """API 요청 추적 헬퍼"""

    def __init__(self, collector: OperationsMetricsCollector, endpoint: str, method: str):
        self.collector = collector
        self.endpoint = endpoint
        self.method = method
        self.start_time = None
        self.status_code = 200

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time

        # 예외 발생 시 500
        if exc_type is not None:
            self.status_code = 500

        self.collector.record_api_request(
            endpoint=self.endpoint,
            method=self.method,
            status_code=self.status_code,
            latency_seconds=latency
        )

    def set_status_code(self, code: int):
        """상태 코드 설정"""
        self.status_code = code


# =============================================================================
# 사용 예시
# =============================================================================

if __name__ == "__main__":
    import logging
    from prometheus_client import generate_latest

    logging.basicConfig(level=logging.INFO)

    # 컬렉터 초기화
    collector = OperationsMetricsCollector()

    print("=== 운영 메트릭 시뮬레이션 ===\n")

    # 1. LLM Brief 생성
    print("1. LLM Brief 생성")
    for mode in ["fast", "balanced", "creative"]:
        for i in range(5):
            success = i < 4  # 4/5 성공
            latency = {"fast": 2.0, "balanced": 3.0, "creative": 4.5}[mode]
            repaired = (i == 2)  # 3번째 요청은 수정됨

            collector.record_llm_brief(
                mode=mode,
                success=success,
                latency_seconds=latency + i*0.1,
                repaired=repaired,
                repair_type="trailing_comma" if repaired else None
            )

    # 2. RL Update
    print("2. RL Update")
    for step in range(10):
        collector.record_rl_update(
            algorithm="ppo",
            policy_loss=0.5 - step*0.02,
            value_loss=0.3 - step*0.01,
            total_loss=0.8 - step*0.03,
            reward=10.0 + step*2.0,
            entropy=2.5 - step*0.1,
            clip_fraction=0.15,
            kl_divergence=0.01
        )

    # 3. API 요청
    print("3. API 요청")
    endpoints = ["/dna/create", "/moga/optimize", "/rl/train"]
    for endpoint in endpoints:
        for i in range(10):
            status_code = 200 if i < 9 else 500  # 9/10 성공
            latency = 2.5 + i*0.1

            collector.record_api_request(
                endpoint=endpoint,
                method="POST",
                status_code=status_code,
                latency_seconds=latency
            )

    # 4. Context Manager 예시
    print("\n4. Context Manager 예시")

    with collector.track_llm_brief("creative") as tracker:
        time.sleep(0.1)  # 시뮬레이션
        tracker.mark_repaired("code_block")

    with collector.track_api_request("/dna/create", "POST") as tracker:
        time.sleep(0.05)  # 시뮬레이션
        tracker.set_status_code(200)

    # 5. 메트릭 출력
    print("\n=== Prometheus 메트릭 출력 ===\n")
    metrics = generate_latest().decode('utf-8')

    # LLM Brief 메트릭만 출력
    for line in metrics.split('\n'):
        if 'llm_brief' in line or 'rl_' in line or 'api_request' in line:
            if not line.startswith('#'):
                print(line)

    print("\n완료")
