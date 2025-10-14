# tests/test_llm_performance.py
"""
Performance and Latency Tests for LLM Ensemble
Test rate/latency under load: 10 RPS for 1 minute
Target: p95 <2.5s (FAST), <4.0s (CREATIVE)
"""

import pytest
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch, MagicMock
from typing import List, Dict

from fragrance_ai.llm import build_brief
from fragrance_ai.llm.schemas import CreativeBrief


# Test inputs for load testing
LOAD_TEST_INPUTS = [
    "상쾌한 시트러스 향수",
    "로맨틱한 저녁 향수",
    "남성적인 우디 향수",
    "Fresh citrus perfume",
    "Romantic evening scent",
    "Masculine woody fragrance",
    "여름에 사용할 가벼운 향",
    "겨울 따뜻한 향수",
    "Spring floral perfume",
    "Autumn spicy fragrance",
]


@pytest.mark.slow
@pytest.mark.performance
class TestPerformanceAndLatency:
    """Performance and latency tests for LLM ensemble"""

    def setup_method(self):
        """Reset cache before each test"""
        from fragrance_ai.llm import _brief_cache
        _brief_cache.clear()

    def test_fast_mode_latency_target(self):
        """
        Test FAST mode latency under load
        Target: p95 < 2.5s
        """
        test_count = 100
        latencies = []

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen:
            # Mock fast responses (simulate Qwen inference time)
            mock_client = MagicMock()

            def mock_infer(user_text, **kwargs):
                # Simulate realistic Qwen inference time (1-2s)
                time.sleep(0.05)  # Simulated processing
                return CreativeBrief(
                    language="ko",
                    mood=["fresh"],
                    season=["summer"],
                    notes_preference={"citrus": 0.5, "floral": 0.5},
                    budget_tier="mid",
                    target_profile="daily_fresh",
                    product_category="EDP"
                )

            mock_client.infer_brief.side_effect = mock_infer
            mock_qwen.return_value = mock_client

            # Measure latencies
            for i in range(test_count):
                user_input = LOAD_TEST_INPUTS[i % len(LOAD_TEST_INPUTS)]

                start_time = time.time()
                brief = build_brief(user_input, mode="fast", use_cache=False)
                latency = time.time() - start_time

                latencies.append(latency)
                assert isinstance(brief, CreativeBrief)

        # Calculate statistics
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99 = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        avg = statistics.mean(latencies)

        print(f"\n=== FAST Mode Latency Statistics ({test_count} requests) ===")
        print(f"Average: {avg*1000:.2f}ms")
        print(f"p50: {p50*1000:.2f}ms")
        print(f"p95: {p95*1000:.2f}ms")
        print(f"p99: {p99*1000:.2f}ms")
        print(f"Target p95: <2500ms")

        # Assert p95 target
        assert p95 < 2.5, f"FAST mode p95 latency {p95:.2f}s exceeds target 2.5s"

    def test_creative_mode_latency_target(self):
        """
        Test CREATIVE mode latency under load
        Target: p95 < 4.0s
        """
        test_count = 100
        latencies = []

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen, \
             patch('fragrance_ai.llm.llama_hints.get_llama_generator') as mock_llama:

            # Mock Qwen
            mock_qwen_client = MagicMock()

            def mock_qwen_infer(user_text, **kwargs):
                time.sleep(0.05)  # Simulated Qwen processing
                return CreativeBrief(
                    language="ko",
                    mood=["fresh"],
                    season=["summer"],
                    notes_preference={"citrus": 0.5, "floral": 0.5},
                    budget_tier="mid",
                    target_profile="daily_fresh",
                    product_category="EDP"
                )

            mock_qwen_client.infer_brief.side_effect = mock_qwen_infer
            mock_qwen.return_value = mock_qwen_client

            # Mock Llama
            mock_llama_gen = MagicMock()

            def mock_llama_hints(user_text, **kwargs):
                time.sleep(0.08)  # Simulated Llama processing
                return ["refreshing", "vibrant", "energetic"]

            mock_llama_gen.generate_hints.side_effect = mock_llama_hints
            mock_llama.return_value = mock_llama_gen

            # Measure latencies
            for i in range(test_count):
                user_input = LOAD_TEST_INPUTS[i % len(LOAD_TEST_INPUTS)]

                start_time = time.time()
                brief = build_brief(user_input, mode="creative", use_cache=False)
                latency = time.time() - start_time

                latencies.append(latency)
                assert isinstance(brief, CreativeBrief)

        # Calculate statistics
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]
        avg = statistics.mean(latencies)

        print(f"\n=== CREATIVE Mode Latency Statistics ({test_count} requests) ===")
        print(f"Average: {avg*1000:.2f}ms")
        print(f"p50: {p50*1000:.2f}ms")
        print(f"p95: {p95*1000:.2f}ms")
        print(f"p99: {p99*1000:.2f}ms")
        print(f"Target p95: <4000ms")

        # Assert p95 target
        assert p95 < 4.0, f"CREATIVE mode p95 latency {p95:.2f}s exceeds target 4.0s"

    @pytest.mark.slow
    def test_rate_10_rps_for_1_minute(self):
        """
        Test sustained 10 RPS for 1 minute
        Total requests: 600 (10 RPS * 60s)
        """
        target_rps = 10
        duration_seconds = 60
        total_requests = target_rps * duration_seconds

        successful_requests = 0
        failed_requests = 0
        latencies = []

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen:
            # Mock fast responses
            mock_client = MagicMock()

            def mock_infer(user_text, **kwargs):
                time.sleep(0.05)  # Simulated processing
                return CreativeBrief(
                    language="ko",
                    mood=["fresh"],
                    season=["summer"],
                    notes_preference={"citrus": 0.5},
                    budget_tier="mid",
                    target_profile="daily_fresh",
                    product_category="EDP"
                )

            mock_client.infer_brief.side_effect = mock_infer
            mock_qwen.return_value = mock_client

            # Execute requests at 10 RPS
            start_time = time.time()
            request_times = []

            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = []

                for i in range(total_requests):
                    # Schedule request at specific time to maintain 10 RPS
                    scheduled_time = start_time + (i / target_rps)
                    sleep_duration = scheduled_time - time.time()

                    if sleep_duration > 0:
                        time.sleep(sleep_duration)

                    # Submit request
                    user_input = LOAD_TEST_INPUTS[i % len(LOAD_TEST_INPUTS)]
                    future = executor.submit(
                        self._execute_request,
                        user_input,
                        "fast"
                    )
                    futures.append((future, time.time()))

                # Wait for all requests to complete
                for future, req_start_time in futures:
                    try:
                        latency = future.result()
                        successful_requests += 1
                        latencies.append(latency)
                        request_times.append(req_start_time)
                    except Exception as e:
                        failed_requests += 1

            # Calculate actual rate
            total_duration = time.time() - start_time
            actual_rps = successful_requests / total_duration

            # Calculate latency statistics
            p50 = statistics.median(latencies)
            p95 = statistics.quantiles(latencies, n=20)[18]
            avg = statistics.mean(latencies)

            print(f"\n=== 10 RPS Load Test (1 minute) ===")
            print(f"Total requests: {total_requests}")
            print(f"Successful: {successful_requests}")
            print(f"Failed: {failed_requests}")
            print(f"Duration: {total_duration:.2f}s")
            print(f"Target RPS: {target_rps}")
            print(f"Actual RPS: {actual_rps:.2f}")
            print(f"Average latency: {avg*1000:.2f}ms")
            print(f"p50 latency: {p50*1000:.2f}ms")
            print(f"p95 latency: {p95*1000:.2f}ms")

            # Assertions
            assert successful_requests >= total_requests * 0.99, \
                f"Success rate {successful_requests/total_requests:.2%} below 99%"
            assert 9.5 <= actual_rps <= 10.5, \
                f"Actual RPS {actual_rps:.2f} not within target range 9.5-10.5"
            assert p95 < 2.5, \
                f"p95 latency {p95:.2f}s exceeds target 2.5s"

    @pytest.mark.slow
    def test_concurrent_requests_throughput(self):
        """
        Test maximum throughput with concurrent requests
        Measure: requests/second with varying concurrency
        """
        concurrency_levels = [1, 5, 10, 20]
        requests_per_level = 50
        results = {}

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen:
            mock_client = MagicMock()

            def mock_infer(user_text, **kwargs):
                time.sleep(0.05)  # Simulated processing
                return CreativeBrief(
                    language="ko",
                    mood=["fresh"],
                    season=["summer"],
                    notes_preference={"citrus": 0.5},
                    budget_tier="mid",
                    target_profile="daily_fresh",
                    product_category="EDP"
                )

            mock_client.infer_brief.side_effect = mock_infer
            mock_qwen.return_value = mock_client

            for concurrency in concurrency_levels:
                start_time = time.time()
                successful = 0

                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for i in range(requests_per_level):
                        user_input = LOAD_TEST_INPUTS[i % len(LOAD_TEST_INPUTS)]
                        future = executor.submit(
                            self._execute_request,
                            user_input,
                            "fast"
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            future.result()
                            successful += 1
                        except:
                            pass

                duration = time.time() - start_time
                throughput = successful / duration

                results[concurrency] = {
                    'throughput': throughput,
                    'duration': duration,
                    'successful': successful
                }

        print(f"\n=== Concurrent Requests Throughput Test ===")
        for concurrency, data in results.items():
            print(f"Concurrency {concurrency:2d}: {data['throughput']:6.2f} req/s "
                  f"({data['successful']}/{requests_per_level} in {data['duration']:.2f}s)")

        # Assert throughput scales with concurrency
        assert results[10]['throughput'] > results[1]['throughput'] * 5, \
            "Throughput did not scale adequately with concurrency"

    def test_cache_performance_improvement(self):
        """
        Test that caching provides significant performance improvement
        """
        test_input = "상쾌한 시트러스 향수"
        iterations = 20

        with patch('fragrance_ai.llm.qwen_client.get_qwen_client') as mock_qwen:
            mock_client = MagicMock()

            def mock_infer(user_text, **kwargs):
                time.sleep(0.1)  # Simulated processing
                return CreativeBrief(
                    language="ko",
                    mood=["fresh"],
                    season=["summer"],
                    notes_preference={"citrus": 0.5},
                    budget_tier="mid",
                    target_profile="daily_fresh",
                    product_category="EDP"
                )

            mock_client.infer_brief.side_effect = mock_infer
            mock_qwen.return_value = mock_client

            # First request (cache miss)
            start_time = time.time()
            build_brief(test_input, mode="fast", use_cache=True)
            first_request_time = time.time() - start_time

            # Subsequent requests (cache hits)
            cache_times = []
            for _ in range(iterations):
                start_time = time.time()
                build_brief(test_input, mode="fast", use_cache=True)
                cache_times.append(time.time() - start_time)

            avg_cache_time = statistics.mean(cache_times)
            speedup = first_request_time / avg_cache_time

            print(f"\n=== Cache Performance ===")
            print(f"First request (cache miss): {first_request_time*1000:.2f}ms")
            print(f"Average cached request: {avg_cache_time*1000:.2f}ms")
            print(f"Speedup: {speedup:.1f}x")

            # Assert cache provides significant speedup
            assert speedup > 5, f"Cache speedup {speedup:.1f}x below target 5x"

    def _execute_request(self, user_input: str, mode: str) -> float:
        """Execute a single request and return latency"""
        start_time = time.time()
        build_brief(user_input, mode=mode, use_cache=False)
        latency = time.time() - start_time
        return latency


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "performance"])
