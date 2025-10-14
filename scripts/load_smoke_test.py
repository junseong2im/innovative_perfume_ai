#!/usr/bin/env python3
"""
Load Smoke Test
간단한 RPS 기반 부하 테스트로 p95 지연 확인
"""

import sys
import time
import argparse
import statistics
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


class LoadTester:
    """간단한 부하 테스터"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.latencies: List[float] = []
        self.errors: List[str] = []

    def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> float:
        """단일 요청 실행 및 지연 측정"""
        url = f"{self.base_url}{endpoint}"
        start = time.time()

        try:
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            latency_ms = (time.time() - start) * 1000

            if response.status_code >= 400:
                self.errors.append(f"HTTP {response.status_code}: {endpoint}")
            else:
                self.latencies.append(latency_ms)

            return latency_ms

        except Exception as e:
            self.errors.append(f"Exception: {str(e)}")
            return -1

    def run_load_test(self, rps: int, duration: int):
        """부하 테스트 실행"""
        print(f"Starting load test: {rps} RPS for {duration} seconds")
        print(f"Target: {self.base_url}")
        print("")

        # Test endpoints
        test_cases = [
            {
                "endpoint": "/health",
                "method": "GET",
                "weight": 0.3  # 30% of traffic
            },
            {
                "endpoint": "/api/v1/dna/create",
                "method": "POST",
                "data": {
                    "brief": {
                        "mood": ["fresh"],
                        "season": ["spring"],
                        "intensity": 0.5
                    },
                    "mode": "fast"
                },
                "weight": 0.5  # 50% of traffic
            },
            {
                "endpoint": "/api/v1/models/status",
                "method": "GET",
                "weight": 0.2  # 20% of traffic
            }
        ]

        total_requests = rps * duration
        interval = 1.0 / rps  # seconds between requests

        print(f"Total requests to send: {total_requests}")
        print(f"Request interval: {interval:.3f}s")
        print("")

        # Execute requests
        with ThreadPoolExecutor(max_workers=min(rps * 2, 50)) as executor:
            futures = []
            start_time = time.time()

            for i in range(total_requests):
                # Select endpoint based on weight
                rand = (i % 10) / 10.0
                cumulative = 0.0
                selected_test = test_cases[0]

                for test in test_cases:
                    cumulative += test["weight"]
                    if rand < cumulative:
                        selected_test = test
                        break

                # Submit request
                future = executor.submit(
                    self.make_request,
                    selected_test["endpoint"],
                    selected_test["method"],
                    selected_test.get("data")
                )
                futures.append(future)

                # Wait to maintain RPS
                elapsed = time.time() - start_time
                expected_time = (i + 1) * interval
                sleep_time = expected_time - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

            # Wait for all requests to complete
            print("Waiting for all requests to complete...")
            for future in as_completed(futures):
                future.result()  # This will raise exceptions if any occurred

        print("")
        print("Load test completed")
        print("")

    def calculate_percentiles(self) -> Dict[str, float]:
        """백분위수 계산"""
        if not self.latencies:
            return {}

        sorted_latencies = sorted(self.latencies)
        n = len(sorted_latencies)

        return {
            "p50": sorted_latencies[int(n * 0.50)],
            "p75": sorted_latencies[int(n * 0.75)],
            "p90": sorted_latencies[int(n * 0.90)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)] if n >= 100 else sorted_latencies[-1],
            "min": sorted_latencies[0],
            "max": sorted_latencies[-1],
            "mean": statistics.mean(sorted_latencies),
            "stdev": statistics.stdev(sorted_latencies) if n > 1 else 0.0
        }

    def print_results(self):
        """결과 출력"""
        print("=" * 60)
        print("Load Test Results")
        print("=" * 60)
        print()

        total = len(self.latencies) + len(self.errors)
        success_rate = (len(self.latencies) / total * 100) if total > 0 else 0

        print(f"Total Requests:  {total}")
        print(f"Successful:      {len(self.latencies)} ({success_rate:.1f}%)")
        print(f"Failed:          {len(self.errors)}")
        print()

        if self.errors:
            print("Errors:")
            for error in self.errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more errors")
            print()

        if self.latencies:
            percentiles = self.calculate_percentiles()
            print("Latency Statistics (milliseconds):")
            print(f"  Min:    {percentiles['min']:8.2f} ms")
            print(f"  Mean:   {percentiles['mean']:8.2f} ms")
            print(f"  Median: {percentiles['p50']:8.2f} ms")
            print(f"  p75:    {percentiles['p75']:8.2f} ms")
            print(f"  p90:    {percentiles['p90']:8.2f} ms")
            print(f"  p95:    {percentiles['p95']:8.2f} ms")
            print(f"  p99:    {percentiles['p99']:8.2f} ms")
            print(f"  Max:    {percentiles['max']:8.2f} ms")
            print(f"  StdDev: {percentiles['stdev']:8.2f} ms")
            print()

            return percentiles
        else:
            print("No successful requests to calculate latency statistics")
            return {}

    def check_threshold(self, p95_threshold: float) -> bool:
        """p95 임계값 확인"""
        if not self.latencies:
            print("❌ FAILED: No successful requests")
            return False

        percentiles = self.calculate_percentiles()
        p95 = percentiles.get("p95", float("inf"))

        print("=" * 60)
        print("Threshold Check")
        print("=" * 60)
        print(f"p95 Latency:  {p95:.2f} ms")
        print(f"Threshold:    {p95_threshold:.2f} ms")
        print()

        if p95 <= p95_threshold:
            print(f"✅ PASSED: p95 latency ({p95:.2f} ms) is within threshold ({p95_threshold:.2f} ms)")
            return True
        else:
            print(f"❌ FAILED: p95 latency ({p95:.2f} ms) exceeds threshold ({p95_threshold:.2f} ms)")
            return False


def main():
    parser = argparse.ArgumentParser(description="Load Smoke Test for Artisan API")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of API")
    parser.add_argument("--rps", type=int, default=10, help="Requests per second")
    parser.add_argument("--duration", type=int, default=30, help="Test duration in seconds")
    parser.add_argument("--p95-threshold", type=float, default=2500.0, help="p95 latency threshold in ms")

    args = parser.parse_args()

    # Create tester
    tester = LoadTester(base_url=args.base_url)

    # Check health before starting
    print("Checking API health...")
    try:
        response = requests.get(f"{args.base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            sys.exit(1)
        print("✅ API is healthy")
        print()
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(1)

    # Run load test
    try:
        tester.run_load_test(rps=args.rps, duration=args.duration)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print results
    tester.print_results()

    # Check threshold
    passed = tester.check_threshold(p95_threshold=args.p95_threshold)

    if passed:
        print()
        print("=" * 60)
        print("✅ LOAD SMOKE TEST PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print()
        print("=" * 60)
        print("❌ LOAD SMOKE TEST FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
