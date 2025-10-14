"""
Shadow Evaluation System
프로덕션 트래픽의 일부를 Teacher/Student 모델에 그림자 추론으로 흘려 비파괴적 비교
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from loguru import logger
import torch


class ModelType(Enum):
    """모델 타입"""
    TEACHER = "teacher"  # PPO 정책 (큰 모델)
    STUDENT = "student"  # Distilled 모델 (작은 모델)
    PRODUCTION = "production"  # 현재 프로덕션 모델


@dataclass
class ShadowRequest:
    """그림자 평가 요청"""
    request_id: str
    user_input: Dict[str, Any]
    sample_rate: float  # 샘플링 비율 (0.0 ~ 1.0)
    timestamp: float


@dataclass
class ShadowResponse:
    """그림자 평가 응답"""
    request_id: str
    model_type: ModelType
    prediction: Dict[str, Any]
    latency_ms: float
    reward_score: Optional[float] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ComparisonMetrics:
    """비교 메트릭"""
    request_id: str
    teacher_latency_ms: float
    student_latency_ms: float
    production_latency_ms: float
    teacher_reward: float
    student_reward: float
    production_reward: float
    kl_divergence: float
    cosine_similarity: float
    speedup_ratio: float  # student / teacher


class ShadowEvaluator:
    """
    Shadow Evaluation 시스템
    프로덕션 트래픽을 Teacher/Student에 복제하여 성능 비교
    """

    def __init__(
        self,
        sample_rate: float = 0.05,  # 5% 샘플링
        enable_teacher: bool = True,
        enable_student: bool = True,
        log_results: bool = True
    ):
        self.sample_rate = sample_rate
        self.enable_teacher = enable_teacher
        self.enable_student = enable_student
        self.log_results = log_results

        # 메트릭 저장
        self.comparison_results: List[ComparisonMetrics] = []

        logger.info(
            f"ShadowEvaluator initialized (sample_rate={sample_rate}, "
            f"teacher={enable_teacher}, student={enable_student})"
        )

    def should_sample(self) -> bool:
        """
        샘플링 여부 결정

        Returns:
            True if should sample
        """
        return np.random.rand() < self.sample_rate

    async def evaluate_request(
        self,
        request: ShadowRequest,
        teacher_model: Any,
        student_model: Any,
        production_model: Any
    ) -> Optional[ComparisonMetrics]:
        """
        요청을 Teacher/Student/Production 모델에 전달하여 평가

        Args:
            request: 그림자 평가 요청
            teacher_model: Teacher 모델
            student_model: Student 모델
            production_model: Production 모델

        Returns:
            ComparisonMetrics or None
        """
        if not self.should_sample():
            return None

        logger.debug(f"Shadow evaluating request {request.request_id}")

        # 비동기 병렬 추론
        tasks = []

        # Production (항상 실행)
        tasks.append(self._run_inference(
            request, production_model, ModelType.PRODUCTION
        ))

        # Teacher (선택적)
        if self.enable_teacher:
            tasks.append(self._run_inference(
                request, teacher_model, ModelType.TEACHER
            ))

        # Student (선택적)
        if self.enable_student:
            tasks.append(self._run_inference(
                request, student_model, ModelType.STUDENT
            ))

        # 병렬 실행
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 응답 처리
        production_resp = responses[0]
        teacher_resp = responses[1] if self.enable_teacher else None
        student_resp = responses[2] if self.enable_student else None

        # 에러 처리
        if isinstance(production_resp, Exception):
            logger.error(f"Production inference failed: {production_resp}")
            return None

        # 메트릭 계산
        metrics = self._compute_metrics(
            request.request_id,
            production_resp,
            teacher_resp,
            student_resp
        )

        if metrics and self.log_results:
            self._log_comparison(metrics)

        self.comparison_results.append(metrics)

        return metrics

    async def _run_inference(
        self,
        request: ShadowRequest,
        model: Any,
        model_type: ModelType
    ) -> ShadowResponse:
        """
        모델 추론 실행

        Args:
            request: 요청
            model: 모델
            model_type: 모델 타입

        Returns:
            ShadowResponse
        """
        start_time = time.time()

        try:
            # 실제 모델 추론 (Mock)
            await asyncio.sleep(0.01)  # 시뮬레이션

            # Mock prediction
            prediction = {
                "recipe_id": f"recipe_{np.random.randint(1000, 9999)}",
                "ingredients": {
                    "citrus": np.random.uniform(5, 15),
                    "woody": np.random.uniform(10, 20),
                    "floral": np.random.uniform(3, 10)
                },
                "confidence": np.random.uniform(0.7, 1.0)
            }

            # Mock reward score
            if model_type == ModelType.TEACHER:
                reward_score = np.random.uniform(20, 25)
            elif model_type == ModelType.STUDENT:
                reward_score = np.random.uniform(18, 24)
            else:  # PRODUCTION
                reward_score = np.random.uniform(19, 24)

            latency_ms = (time.time() - start_time) * 1000

            return ShadowResponse(
                request_id=request.request_id,
                model_type=model_type,
                prediction=prediction,
                latency_ms=latency_ms,
                reward_score=reward_score
            )

        except Exception as e:
            logger.error(f"{model_type.value} inference error: {e}")
            raise

    def _compute_metrics(
        self,
        request_id: str,
        production_resp: ShadowResponse,
        teacher_resp: Optional[ShadowResponse],
        student_resp: Optional[ShadowResponse]
    ) -> Optional[ComparisonMetrics]:
        """
        비교 메트릭 계산

        Args:
            request_id: 요청 ID
            production_resp: Production 응답
            teacher_resp: Teacher 응답
            student_resp: Student 응답

        Returns:
            ComparisonMetrics or None
        """
        if not teacher_resp or not student_resp:
            return None

        # KL divergence 계산 (간단한 시뮬레이션)
        teacher_logits = torch.randn(10)
        student_logits = teacher_logits + torch.randn(10) * 0.5

        teacher_probs = torch.softmax(teacher_logits, dim=0)
        student_log_probs = torch.log_softmax(student_logits, dim=0)

        kl_div = torch.nn.functional.kl_div(
            student_log_probs, teacher_probs, reduction='sum'
        ).item()

        # Cosine similarity 계산
        teacher_vec = teacher_logits / teacher_logits.norm()
        student_vec = student_logits / student_logits.norm()
        cosine_sim = torch.dot(teacher_vec, student_vec).item()

        # Speedup ratio
        speedup_ratio = teacher_resp.latency_ms / student_resp.latency_ms

        metrics = ComparisonMetrics(
            request_id=request_id,
            teacher_latency_ms=teacher_resp.latency_ms,
            student_latency_ms=student_resp.latency_ms,
            production_latency_ms=production_resp.latency_ms,
            teacher_reward=teacher_resp.reward_score,
            student_reward=student_resp.reward_score,
            production_reward=production_resp.reward_score,
            kl_divergence=kl_div,
            cosine_similarity=cosine_sim,
            speedup_ratio=speedup_ratio
        )

        return metrics

    def _log_comparison(self, metrics: ComparisonMetrics):
        """
        비교 결과 로깅

        Args:
            metrics: 비교 메트릭
        """
        logger.info(
            f"[Shadow Eval] {metrics.request_id} | "
            f"Latency (T/S/P): {metrics.teacher_latency_ms:.1f}ms / "
            f"{metrics.student_latency_ms:.1f}ms / {metrics.production_latency_ms:.1f}ms | "
            f"Reward (T/S/P): {metrics.teacher_reward:.2f} / "
            f"{metrics.student_reward:.2f} / {metrics.production_reward:.2f} | "
            f"KL: {metrics.kl_divergence:.4f} | "
            f"Speedup: {metrics.speedup_ratio:.2f}x"
        )

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        전체 통계 요약

        Returns:
            Dict with summary statistics
        """
        if not self.comparison_results:
            return {}

        # NumPy 배열로 변환
        teacher_latencies = [m.teacher_latency_ms for m in self.comparison_results]
        student_latencies = [m.student_latency_ms for m in self.comparison_results]
        teacher_rewards = [m.teacher_reward for m in self.comparison_results]
        student_rewards = [m.student_reward for m in self.comparison_results]
        kl_divergences = [m.kl_divergence for m in self.comparison_results]
        speedup_ratios = [m.speedup_ratio for m in self.comparison_results]

        summary = {
            "total_comparisons": len(self.comparison_results),
            "latency": {
                "teacher_avg_ms": np.mean(teacher_latencies),
                "student_avg_ms": np.mean(student_latencies),
                "teacher_p95_ms": np.percentile(teacher_latencies, 95),
                "student_p95_ms": np.percentile(student_latencies, 95),
            },
            "reward": {
                "teacher_avg": np.mean(teacher_rewards),
                "student_avg": np.mean(student_rewards),
                "retention_rate": np.mean(student_rewards) / np.mean(teacher_rewards),
            },
            "kl_divergence": {
                "avg": np.mean(kl_divergences),
                "p95": np.percentile(kl_divergences, 95),
            },
            "speedup": {
                "avg": np.mean(speedup_ratios),
                "min": np.min(speedup_ratios),
                "max": np.max(speedup_ratios),
            }
        }

        return summary

    def export_results(self, filepath: str):
        """
        결과를 JSON 파일로 내보내기

        Args:
            filepath: 저장할 파일 경로
        """
        import json

        summary = self.get_summary_stats()
        results = {
            "summary": summary,
            "details": [asdict(m) for m in self.comparison_results]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Shadow evaluation results exported to {filepath}")


# CLI 진입점
async def main():
    """메인 함수"""
    evaluator = ShadowEvaluator(sample_rate=1.0)  # 테스트용 100% 샘플링

    # Mock 모델
    teacher_model = None
    student_model = None
    production_model = None

    # 100개 요청 시뮬레이션
    for i in range(100):
        request = ShadowRequest(
            request_id=f"req_{i:04d}",
            user_input={"prompt": f"test prompt {i}"},
            sample_rate=1.0,
            timestamp=time.time()
        )

        metrics = await evaluator.evaluate_request(
            request,
            teacher_model,
            student_model,
            production_model
        )

        await asyncio.sleep(0.1)  # 100ms 간격

    # 결과 요약
    summary = evaluator.get_summary_stats()
    print("\n=== Shadow Evaluation Summary ===")
    import json
    print(json.dumps(summary, indent=2))

    # 결과 저장
    evaluator.export_results("shadow_evaluation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
