#!/usr/bin/env python3
"""
Fragrance AI 시스템 성능 벤치마크 스크립트
과학적이고 정확한 성능 측정을 위한 종합적인 벤치마크 도구
"""

import asyncio
import time
import statistics
import numpy as np
from pathlib import Path
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import argparse
import psutil
import torch
import gc
from dataclasses import dataclass
from contextlib import contextmanager
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fragrance_ai.core.vector_store import VectorStore
from fragrance_ai.models.embedding import FragranceEmbedding
from fragrance_ai.models.generator import FragranceGenerator
from fragrance_ai.services.search_service import SearchService
from fragrance_ai.services.generation_service import GenerationService
from fragrance_ai.evaluation.mathematical_metrics import MathematicalMetrics
from fragrance_ai.utils.data_loader import DatasetLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""
    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    sample_size: int
    confidence_interval: Tuple[float, float]

class PerformanceBenchmark:
    """성능 벤치마크 클래스"""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 벤치마크 결과 저장
        self.results = {}
        
        # 시스템 정보
        self.system_info = self._collect_system_info()
        
        # 메트릭 계산기
        self.metrics_calculator = MathematicalMetrics(confidence_level=0.95)
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """시스템 정보 수집"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": sys.version,
            "torch_version": torch.__version__ if torch.cuda.is_available() else "CPU",
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "timestamp": datetime.now().isoformat()
        }
    
    @contextmanager
    def _monitor_resources(self):
        """리소스 사용량 모니터링"""
        process = psutil.Process()
        
        # 시작 상태
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        start_cpu_time = process.cpu_times()
        start_time = time.time()
        
        # GC 실행
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield
        
        # 종료 상태
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        end_cpu_time = process.cpu_times()
        
        # CPU 사용률 계산
        elapsed_time = end_time - start_time
        cpu_time_used = (end_cpu_time.user - start_cpu_time.user) + (end_cpu_time.system - start_cpu_time.system)
        cpu_usage = (cpu_time_used / elapsed_time) * 100 if elapsed_time > 0 else 0
        
        self._current_memory_usage = end_memory - start_memory
        self._current_cpu_usage = cpu_usage
    
    async def benchmark_embedding_model(
        self, 
        model_path: str = None,
        sample_sizes: List[int] = [1, 10, 50, 100, 500],
        iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """임베딩 모델 성능 벤치마크"""
        logger.info("Starting embedding model benchmark...")
        
        # 모델 초기화
        embedding_model = FragranceEmbedding()
        if model_path:
            embedding_model.load_model(model_path)
        else:
            await embedding_model.initialize()
        
        results = {}
        
        for sample_size in sample_sizes:
            logger.info(f"Benchmarking embedding with sample_size={sample_size}")
            
            # 테스트 데이터 생성
            test_texts = [
                f"상큼한 시트러스 향수 {i}번째 테스트" 
                for i in range(sample_size)
            ]
            
            # 성능 측정
            times = []
            errors = 0
            
            for iteration in range(iterations):
                with self._monitor_resources():
                    start_time = time.perf_counter()
                    
                    try:
                        if sample_size == 1:
                            # 단일 인코딩
                            _ = await embedding_model.encode_query(test_texts[0])
                        else:
                            # 배치 인코딩
                            _ = await embedding_model.encode_batch(test_texts)
                        
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                        
                    except Exception as e:
                        logger.error(f"Error in embedding benchmark: {e}")
                        errors += 1
                        times.append(float('inf'))
            
            # 통계 계산
            valid_times = [t for t in times if t != float('inf')]
            
            if valid_times:
                result = self._calculate_benchmark_stats(
                    name=f"embedding_batch_{sample_size}",
                    times=valid_times,
                    sample_size=sample_size,
                    errors=errors,
                    total_iterations=iterations
                )
                results[f"batch_{sample_size}"] = result
        
        return results
    
    async def benchmark_generation_model(
        self,
        model_path: str = None,
        generation_types: List[str] = ["basic_recipe", "detailed_recipe"],
        iterations: int = 5
    ) -> Dict[str, BenchmarkResult]:
        """생성 모델 성능 벤치마크"""
        logger.info("Starting generation model benchmark...")
        
        # 서비스 초기화
        generation_service = GenerationService()
        await generation_service.initialize()
        
        results = {}
        
        test_requests = {
            "basic_recipe": {
                "fragrance_family": "floral",
                "mood": "romantic",
                "intensity": "moderate",
                "generation_type": "basic_recipe"
            },
            "detailed_recipe": {
                "fragrance_family": "woody",
                "mood": "sophisticated", 
                "intensity": "strong",
                "generation_type": "detailed_recipe"
            }
        }
        
        for gen_type in generation_types:
            logger.info(f"Benchmarking generation type: {gen_type}")
            
            times = []
            errors = 0
            quality_scores = []
            
            for iteration in range(iterations):
                with self._monitor_resources():
                    start_time = time.perf_counter()
                    
                    try:
                        result = await generation_service.generate_recipe(
                            request_data=test_requests[gen_type],
                            use_cache=False  # 캐시 사용하지 않음
                        )
                        
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                        
                        # 품질 점수 수집
                        quality_scores.append(result.get("quality_score", 0))
                        
                    except Exception as e:
                        logger.error(f"Error in generation benchmark: {e}")
                        errors += 1
                        times.append(float('inf'))
            
            # 통계 계산
            valid_times = [t for t in times if t != float('inf')]
            
            if valid_times:
                result = self._calculate_benchmark_stats(
                    name=f"generation_{gen_type}",
                    times=valid_times,
                    sample_size=1,  # 한 번에 하나씩 생성
                    errors=errors,
                    total_iterations=iterations
                )
                
                # 품질 점수 메타데이터 추가
                if quality_scores:
                    result.metadata = {
                        "avg_quality_score": statistics.mean(quality_scores),
                        "std_quality_score": statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0
                    }
                
                results[gen_type] = result
        
        return results
    
    async def benchmark_search_service(
        self,
        sample_queries: List[str] = None,
        iterations: int = 10
    ) -> Dict[str, BenchmarkResult]:
        """검색 서비스 성능 벤치마크"""
        logger.info("Starting search service benchmark...")
        
        # 서비스 초기화
        search_service = SearchService()
        await search_service.initialize()
        
        # 기본 쿼리 설정
        if not sample_queries:
            sample_queries = [
                "상큼한 시트러스 향수",
                "로맨틱한 플로럴 향수", 
                "깊은 우디 향수",
                "신선한 봄 향수",
                "세련된 저녁 향수"
            ]
        
        results = {}
        
        # 단일 쿼리 테스트
        logger.info("Benchmarking single query search...")
        times = []
        errors = 0
        result_counts = []
        
        for iteration in range(iterations):
            query = sample_queries[iteration % len(sample_queries)]
            
            with self._monitor_resources():
                start_time = time.perf_counter()
                
                try:
                    result = await search_service.semantic_search(
                        query=query,
                        top_k=10,
                        use_cache=False
                    )
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                    result_counts.append(len(result.get("results", [])))
                    
                except Exception as e:
                    logger.error(f"Error in search benchmark: {e}")
                    errors += 1
                    times.append(float('inf'))
        
        # 통계 계산
        valid_times = [t for t in times if t != float('inf')]
        
        if valid_times:
            result = self._calculate_benchmark_stats(
                name="search_single_query",
                times=valid_times,
                sample_size=1,
                errors=errors,
                total_iterations=iterations
            )
            
            result.metadata = {
                "avg_results_returned": statistics.mean(result_counts) if result_counts else 0
            }
            
            results["single_query"] = result
        
        return results
    
    async def benchmark_end_to_end(
        self,
        scenarios: List[Dict[str, Any]] = None,
        iterations: int = 3
    ) -> Dict[str, BenchmarkResult]:
        """End-to-end 성능 벤치마크"""
        logger.info("Starting end-to-end benchmark...")
        
        # 서비스 초기화
        search_service = SearchService()
        generation_service = GenerationService()
        await search_service.initialize()
        await generation_service.initialize()
        
        # 기본 시나리오 설정
        if not scenarios:
            scenarios = [
                {
                    "search_query": "봄에 어울리는 상큼한 향수",
                    "generation_params": {
                        "fragrance_family": "citrus",
                        "mood": "fresh",
                        "intensity": "light"
                    }
                }
            ]
        
        results = {}
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Benchmarking E2E scenario {i+1}")
            
            times = []
            errors = 0
            
            for iteration in range(iterations):
                with self._monitor_resources():
                    start_time = time.perf_counter()
                    
                    try:
                        # 1. 검색
                        search_result = await search_service.semantic_search(
                            query=scenario["search_query"],
                            top_k=3,
                            use_cache=False
                        )
                        
                        # 2. 생성
                        generation_result = await generation_service.generate_recipe(
                            request_data=scenario["generation_params"],
                            use_cache=False
                        )
                        
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                        
                    except Exception as e:
                        logger.error(f"Error in E2E benchmark: {e}")
                        errors += 1
                        times.append(float('inf'))
            
            # 통계 계산
            valid_times = [t for t in times if t != float('inf')]
            
            if valid_times:
                result = self._calculate_benchmark_stats(
                    name=f"e2e_scenario_{i+1}",
                    times=valid_times,
                    sample_size=1,
                    errors=errors,
                    total_iterations=iterations
                )
                
                results[f"scenario_{i+1}"] = result
        
        return results
    
    def _calculate_benchmark_stats(
        self,
        name: str,
        times: List[float],
        sample_size: int,
        errors: int,
        total_iterations: int
    ) -> BenchmarkResult:
        """벤치마크 통계 계산"""
        if not times:
            # 모든 시도가 실패한 경우
            return BenchmarkResult(
                name=name,
                mean_time=float('inf'),
                std_time=0,
                min_time=float('inf'),
                max_time=float('inf'),
                p95_time=float('inf'),
                p99_time=float('inf'),
                throughput=0,
                memory_usage_mb=getattr(self, '_current_memory_usage', 0),
                cpu_usage_percent=getattr(self, '_current_cpu_usage', 0),
                success_rate=0,
                error_count=errors,
                sample_size=sample_size,
                confidence_interval=(float('inf'), float('inf'))
            )
        
        # 기본 통계
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        min_time = min(times)
        max_time = max(times)
        
        # 백분위수 계산
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        # 처리량 계산 (초당 처리 건수)
        throughput = sample_size / mean_time if mean_time > 0 else 0
        
        # 성공률 계산
        success_rate = len(times) / total_iterations
        
        # 신뢰구간 계산 (95%)
        if len(times) > 1:
            confidence_interval = self._calculate_confidence_interval(times, 0.95)
        else:
            confidence_interval = (mean_time, mean_time)
        
        return BenchmarkResult(
            name=name,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            p95_time=p95_time,
            p99_time=p99_time,
            throughput=throughput,
            memory_usage_mb=getattr(self, '_current_memory_usage', 0),
            cpu_usage_percent=getattr(self, '_current_cpu_usage', 0),
            success_rate=success_rate,
            error_count=errors,
            sample_size=sample_size,
            confidence_interval=confidence_interval
        )
    
    def _calculate_confidence_interval(
        self, 
        data: List[float], 
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """신뢰구간 계산"""
        n = len(data)
        mean = statistics.mean(data)
        std_err = statistics.stdev(data) / (n ** 0.5)
        
        # t-분포 임계값 (근사값)
        alpha = 1 - confidence_level
        if n > 30:
            t_critical = 1.96  # 정규분포 근사
        else:
            # 간단한 t-분포 근사값들
            t_values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                       10: 2.228, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}
            t_critical = t_values.get(n, 2.0)
        
        margin_of_error = t_critical * std_err
        
        return (mean - margin_of_error, mean + margin_of_error)
    
    def generate_report(self, all_results: Dict[str, Dict[str, BenchmarkResult]]) -> str:
        """벤치마크 리포트 생성"""
        report_lines = [
            "# Fragrance AI 성능 벤치마크 리포트",
            f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 시스템 정보",
            f"- CPU: {self.system_info['cpu_count']} cores",
            f"- 메모리: {self.system_info['memory_total_gb']:.1f} GB", 
            f"- GPU: {self.system_info['gpu_name']}",
            f"- CUDA: {self.system_info['cuda_available']}",
            f"- Python: {self.system_info['python_version']}",
            f"- PyTorch: {self.system_info['torch_version']}",
            "",
            "## 벤치마크 결과",
            ""
        ]
        
        for category, results in all_results.items():
            report_lines.append(f"### {category.replace('_', ' ').title()}")
            report_lines.append("")
            
            # 테이블 헤더
            report_lines.extend([
                "| 테스트 | 평균 시간(s) | 표준편차(s) | P95(s) | 처리량(/s) | 성공률(%) | 메모리(MB) |",
                "|--------|-------------|------------|---------|-----------|----------|----------|"
            ])
            
            for test_name, result in results.items():
                report_lines.append(
                    f"| {test_name} | {result.mean_time:.3f} | {result.std_time:.3f} | "
                    f"{result.p95_time:.3f} | {result.throughput:.2f} | "
                    f"{result.success_rate*100:.1f} | {result.memory_usage_mb:.1f} |"
                )
            
            report_lines.append("")
        
        # 성능 권장사항
        report_lines.extend([
            "## 성능 분석 및 권장사항",
            ""
        ])
        
        # 임베딩 모델 분석
        if "embedding" in all_results:
            embedding_results = all_results["embedding"]
            batch_1_time = embedding_results.get("batch_1", BenchmarkResult("", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0, 0))).mean_time
            batch_100_time = embedding_results.get("batch_100", BenchmarkResult("", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (0, 0))).mean_time
            
            if batch_1_time > 0 and batch_100_time > 0:
                efficiency_ratio = (batch_1_time * 100) / batch_100_time
                report_lines.append(f"- **임베딩 배치 효율성**: {efficiency_ratio:.1f}x (배치 처리가 {efficiency_ratio:.1f}배 더 효율적)")
                
                if efficiency_ratio < 5:
                    report_lines.append("  - ⚠️ 배치 처리 효율성이 낮습니다. GPU 활용을 확인해보세요.")
                elif efficiency_ratio > 20:
                    report_lines.append("  - ✅ 우수한 배치 처리 효율성을 보입니다.")
        
        # 생성 모델 분석
        if "generation" in all_results:
            generation_results = all_results["generation"]
            for test_name, result in generation_results.items():
                if result.mean_time > 10:
                    report_lines.append(f"- ⚠️ {test_name} 생성 시간이 긴 편입니다 ({result.mean_time:.1f}s). 모델 최적화를 고려해보세요.")
                elif result.mean_time < 3:
                    report_lines.append(f"- ✅ {test_name} 생성 성능이 우수합니다 ({result.mean_time:.1f}s).")
        
        return "\n".join(report_lines)
    
    def save_results(self, all_results: Dict[str, Dict[str, BenchmarkResult]]):
        """결과를 파일로 저장"""
        # JSON 형태로 저장
        json_data = {
            "system_info": self.system_info,
            "results": {}
        }
        
        for category, results in all_results.items():
            json_data["results"][category] = {}
            for test_name, result in results.items():
                json_data["results"][category][test_name] = {
                    "mean_time": result.mean_time,
                    "std_time": result.std_time,
                    "min_time": result.min_time,
                    "max_time": result.max_time,
                    "p95_time": result.p95_time,
                    "p99_time": result.p99_time,
                    "throughput": result.throughput,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_usage_percent": result.cpu_usage_percent,
                    "success_rate": result.success_rate,
                    "error_count": result.error_count,
                    "sample_size": result.sample_size,
                    "confidence_interval": result.confidence_interval
                }
        
        # JSON 파일 저장
        json_file = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 리포트 저장
        report = self.generate_report(all_results)
        report_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"결과 저장 완료:")
        logger.info(f"  - JSON: {json_file}")
        logger.info(f"  - Report: {report_file}")

async def main():
    parser = argparse.ArgumentParser(description="Fragrance AI 성능 벤치마크")
    parser.add_argument("--output-dir", type=str, default="./benchmark_results", 
                       help="결과 저장 디렉토리")
    parser.add_argument("--iterations", type=int, default=10, 
                       help="반복 횟수")
    parser.add_argument("--skip-embedding", action="store_true", 
                       help="임베딩 벤치마크 건너뛰기")
    parser.add_argument("--skip-generation", action="store_true", 
                       help="생성 벤치마크 건너뛰기")
    parser.add_argument("--skip-search", action="store_true", 
                       help="검색 벤치마크 건너뛰기")
    parser.add_argument("--skip-e2e", action="store_true", 
                       help="E2E 벤치마크 건너뛰기")
    
    args = parser.parse_args()
    
    # 벤치마크 인스턴스 생성
    benchmark = PerformanceBenchmark(args.output_dir)
    all_results = {}
    
    logger.info("Fragrance AI 성능 벤치마크 시작...")
    logger.info(f"시스템 정보: {benchmark.system_info}")
    
    # 임베딩 모델 벤치마크
    if not args.skip_embedding:
        try:
            embedding_results = await benchmark.benchmark_embedding_model(
                iterations=args.iterations
            )
            all_results["embedding"] = embedding_results
            logger.info("✅ 임베딩 벤치마크 완료")
        except Exception as e:
            logger.error(f"❌ 임베딩 벤치마크 실패: {e}")
    
    # 생성 모델 벤치마크
    if not args.skip_generation:
        try:
            generation_results = await benchmark.benchmark_generation_model(
                iterations=max(1, args.iterations // 2)  # 생성은 시간이 오래 걸리므로 절반
            )
            all_results["generation"] = generation_results
            logger.info("✅ 생성 벤치마크 완료")
        except Exception as e:
            logger.error(f"❌ 생성 벤치마크 실패: {e}")
    
    # 검색 서비스 벤치마크
    if not args.skip_search:
        try:
            search_results = await benchmark.benchmark_search_service(
                iterations=args.iterations
            )
            all_results["search"] = search_results
            logger.info("✅ 검색 벤치마크 완료")
        except Exception as e:
            logger.error(f"❌ 검색 벤치마크 실패: {e}")
    
    # End-to-End 벤치마크
    if not args.skip_e2e:
        try:
            e2e_results = await benchmark.benchmark_end_to_end(
                iterations=max(1, args.iterations // 3)  # E2E는 더 적은 반복
            )
            all_results["end_to_end"] = e2e_results
            logger.info("✅ E2E 벤치마크 완료")
        except Exception as e:
            logger.error(f"❌ E2E 벤치마크 실패: {e}")
    
    # 결과 저장 및 리포트 생성
    if all_results:
        benchmark.save_results(all_results)
        logger.info("🎉 모든 벤치마크 완료!")
    else:
        logger.warning("⚠️ 벤치마크 결과가 없습니다.")

if __name__ == "__main__":
    asyncio.run(main())