#!/usr/bin/env python3
"""
Smoke Test Script - API 및 로그 검증
런치 직후 5분 수동 검증 자동화

Usage:
    python scripts/smoke_test_api.py --base-url http://localhost:8000
    python scripts/smoke_test_api.py --base-url http://localhost:8001 --canary
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


# Colors for output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


@dataclass
class TestResult:
    """Test result container"""
    name: str
    passed: bool
    message: str
    details: Optional[Dict] = None


class SmokeTestRunner:
    """Automated smoke test runner"""

    def __init__(self, base_url: str, container_name: str = "fragrance-ai-app", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.container_name = container_name
        self.timeout = timeout
        self.results: List[TestResult] = []

        # Store IDs for chained tests
        self.dna_id: Optional[str] = None
        self.experiment_id: Optional[str] = None
        self.option_id: Optional[str] = None

    def log_info(self, message: str):
        print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")

    def log_success(self, message: str):
        print(f"{Colors.GREEN}[✓ PASS]{Colors.NC} {message}")

    def log_warning(self, message: str):
        print(f"{Colors.YELLOW}[⚠ WARN]{Colors.NC} {message}")

    def log_error(self, message: str):
        print(f"{Colors.RED}[✗ FAIL]{Colors.NC} {message}")

    def log_step(self, message: str):
        print(f"{Colors.CYAN}[STEP]{Colors.NC} {message}")

    def test_health_check(self) -> TestResult:
        """Test health check endpoint"""
        self.log_step("Health Check")

        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            self.log_success("Health check passed")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            return TestResult(
                name="health_check",
                passed=True,
                message="Health check successful",
                details=data
            )
        except Exception as e:
            self.log_error(f"Health check failed: {e}")
            return TestResult(
                name="health_check",
                passed=False,
                message=f"Health check failed: {e}"
            )

    def test_dna_create(self) -> TestResult:
        """Test /dna/create endpoint"""
        self.log_step("API Test - /dna/create")

        payload = {
            "brief": {
                "mood": "상큼함",
                "season": ["spring"]
            }
        }

        try:
            self.log_info("Creating perfume DNA...")
            response = requests.post(
                f"{self.base_url}/dna/create",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Extract DNA ID
            self.dna_id = data.get('dna_id') or data.get('id')

            if self.dna_id:
                self.log_success(f"DNA created successfully: {self.dna_id}")
                return TestResult(
                    name="dna_create",
                    passed=True,
                    message=f"DNA created: {self.dna_id}",
                    details=data
                )
            else:
                self.log_error("DNA ID not found in response")
                return TestResult(
                    name="dna_create",
                    passed=False,
                    message="DNA ID not found in response",
                    details=data
                )
        except Exception as e:
            self.log_error(f"/dna/create failed: {e}")
            return TestResult(
                name="dna_create",
                passed=False,
                message=f"/dna/create failed: {e}"
            )

    def test_evolve_options(self) -> TestResult:
        """Test /evolve/options endpoint"""
        self.log_step("API Test - /evolve/options")

        if not self.dna_id:
            self.log_warning("Skipping /evolve/options (no DNA_ID)")
            return TestResult(
                name="evolve_options",
                passed=False,
                message="Skipped (no DNA_ID)"
            )

        payload = {
            "dna_id": self.dna_id,
            "algorithm": "PPO",
            "num_options": 3,
            "mode": "creative"
        }

        try:
            self.log_info("Generating evolution options (PPO, creative mode)...")
            response = requests.post(
                f"{self.base_url}/evolve/options",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Extract experiment ID and option ID
            self.experiment_id = data.get('experiment_id') or data.get('id')
            options = data.get('options', [])
            if options:
                self.option_id = options[0].get('id') or options[0].get('option_id')

            if self.experiment_id:
                self.log_success(f"Evolution options generated: {self.experiment_id}")
                if self.option_id:
                    self.log_success(f"First option ID: {self.option_id}")

                return TestResult(
                    name="evolve_options",
                    passed=True,
                    message=f"Evolution options generated: {self.experiment_id}",
                    details=data
                )
            else:
                self.log_error("Experiment ID not found in response")
                return TestResult(
                    name="evolve_options",
                    passed=False,
                    message="Experiment ID not found",
                    details=data
                )
        except Exception as e:
            self.log_error(f"/evolve/options failed: {e}")
            return TestResult(
                name="evolve_options",
                passed=False,
                message=f"/evolve/options failed: {e}"
            )

    def test_evolve_feedback(self) -> TestResult:
        """Test /evolve/feedback endpoint"""
        self.log_step("API Test - /evolve/feedback")

        if not self.experiment_id or not self.option_id:
            self.log_warning("Skipping /evolve/feedback (no experiment_id or option_id)")
            return TestResult(
                name="evolve_feedback",
                passed=False,
                message="Skipped (no experiment_id or option_id)"
            )

        payload = {
            "experiment_id": self.experiment_id,
            "chosen_id": self.option_id,
            "rating": 5
        }

        try:
            self.log_info("Submitting feedback...")
            response = requests.post(
                f"{self.base_url}/evolve/feedback",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Check for success indicator
            success = data.get('success') or data.get('status')

            if success in [True, 'true', 'success', 'ok']:
                self.log_success("Feedback submitted successfully")
                return TestResult(
                    name="evolve_feedback",
                    passed=True,
                    message="Feedback submitted",
                    details=data
                )
            else:
                self.log_warning(f"Feedback response unclear: {success}")
                return TestResult(
                    name="evolve_feedback",
                    passed=True,  # Don't fail on unclear response
                    message=f"Response: {success}",
                    details=data
                )
        except Exception as e:
            self.log_error(f"/evolve/feedback failed: {e}")
            return TestResult(
                name="evolve_feedback",
                passed=False,
                message=f"/evolve/feedback failed: {e}"
            )

    def check_logs(self, pattern: str, required_fields: List[str]) -> Tuple[bool, List[str]]:
        """Check Docker logs for pattern and required fields"""
        try:
            # Get logs from last 5 minutes
            result = subprocess.run(
                ['docker', 'logs', self.container_name, '--since', '5m'],
                capture_output=True,
                text=True,
                timeout=30
            )

            logs = result.stdout + result.stderr

            # Find lines matching pattern
            matching_lines = [line for line in logs.split('\n') if pattern.lower() in line.lower()]

            if not matching_lines:
                return False, []

            # Check for required fields
            found_fields = []
            for field in required_fields:
                if any(field.lower() in line.lower() for line in matching_lines):
                    found_fields.append(field)

            return True, found_fields
        except Exception as e:
            self.log_warning(f"Error checking logs: {e}")
            return False, []

    def test_llm_brief_logs(self) -> TestResult:
        """Verify llm_brief logs are present"""
        self.log_step("Log Verification - llm_brief metrics")

        self.log_info("Checking for llm_brief{mode,...,elapsed_ms} in logs...")

        # Wait for logs to be written
        time.sleep(3)

        required_fields = ['mode', 'elapsed_ms', 'duration', 'latency']
        found, found_fields = self.check_logs('llm_brief', required_fields)

        if found:
            self.log_success(f"Found llm_brief logs with fields: {', '.join(found_fields)}")

            # Check if essential fields are present
            has_mode = any('mode' in f.lower() for f in found_fields)
            has_timing = any(f in found_fields for f in ['elapsed_ms', 'duration', 'latency'])

            if has_mode and has_timing:
                self.log_success("✓ All required fields present (mode, timing)")
                return TestResult(
                    name="llm_brief_logs",
                    passed=True,
                    message="llm_brief logs verified",
                    details={'found_fields': found_fields}
                )
            else:
                missing = []
                if not has_mode:
                    missing.append('mode')
                if not has_timing:
                    missing.append('elapsed_ms/duration')

                self.log_warning(f"⚠ Missing fields: {', '.join(missing)}")
                return TestResult(
                    name="llm_brief_logs",
                    passed=False,
                    message=f"Missing fields: {', '.join(missing)}",
                    details={'found_fields': found_fields}
                )
        else:
            self.log_error("No llm_brief logs found")
            self.log_info("This may indicate:")
            self.log_info("  1. LLM brief generation not triggered")
            self.log_info("  2. Logs not yet written")
            self.log_info("  3. Logging configuration issue")
            return TestResult(
                name="llm_brief_logs",
                passed=False,
                message="No llm_brief logs found"
            )

    def test_rl_update_logs(self) -> TestResult:
        """Verify rl_update logs are present"""
        self.log_step("Log Verification - rl_update metrics")

        self.log_info("Checking for rl_update{algo,loss,reward,entropy,clip_frac} in logs...")

        required_fields = ['algo', 'loss', 'reward', 'entropy', 'clip_frac']
        patterns = ['rl_update', 'rl_training', 'ppo_update']

        found = False
        found_fields = []

        for pattern in patterns:
            found, found_fields = self.check_logs(pattern, required_fields)
            if found:
                break

        if found:
            self.log_success(f"Found rl_update logs with fields: {', '.join(found_fields)}")

            # Check which fields are present
            for field in required_fields:
                if field in found_fields:
                    self.log_success(f"✓ {field} field present")
                else:
                    self.log_warning(f"⚠ {field} field not found")

            # Consider it a pass if at least 3/5 fields are present
            if len(found_fields) >= 3:
                return TestResult(
                    name="rl_update_logs",
                    passed=True,
                    message=f"rl_update logs verified ({len(found_fields)}/5 fields)",
                    details={'found_fields': found_fields}
                )
            else:
                return TestResult(
                    name="rl_update_logs",
                    passed=False,
                    message=f"Insufficient fields ({len(found_fields)}/5)",
                    details={'found_fields': found_fields}
                )
        else:
            self.log_warning("No rl_update logs found")
            self.log_info("This may be expected if:")
            self.log_info("  1. RL training not triggered yet (requires feedback)")
            self.log_info("  2. Training batch not completed")
            self.log_info("  3. Async worker processing")
            return TestResult(
                name="rl_update_logs",
                passed=False,  # Mark as fail but not critical
                message="No rl_update logs found (may be expected)"
            )

    def test_metrics_endpoint(self) -> TestResult:
        """Test /metrics endpoint for required metrics"""
        self.log_step("Metrics Endpoint Check")

        try:
            self.log_info("Checking /metrics endpoint...")
            response = requests.get(
                f"{self.base_url}/metrics",
                timeout=self.timeout
            )
            response.raise_for_status()

            metrics_text = response.text

            # Check for llm_brief metrics
            has_llm_brief = 'llm_brief' in metrics_text

            # Check for rl_update metrics
            has_rl_update = 'rl_update' in metrics_text or 'rl_training' in metrics_text

            if has_llm_brief:
                self.log_success("✓ llm_brief metrics present in /metrics")
            else:
                self.log_warning("⚠ llm_brief metrics not found in /metrics")

            if has_rl_update:
                self.log_success("✓ rl_update metrics present in /metrics")
            else:
                self.log_warning("⚠ rl_update metrics not found in /metrics (may not be published yet)")

            return TestResult(
                name="metrics_endpoint",
                passed=has_llm_brief,  # Only require llm_brief for pass
                message="Metrics endpoint verified",
                details={
                    'has_llm_brief': has_llm_brief,
                    'has_rl_update': has_rl_update
                }
            )
        except Exception as e:
            self.log_warning(f"/metrics endpoint not accessible: {e}")
            return TestResult(
                name="metrics_endpoint",
                passed=False,
                message=f"/metrics endpoint error: {e}"
            )

    def run_all_tests(self) -> bool:
        """Run all smoke tests"""
        print("=" * 80)
        print("SMOKE TEST - 런치 직후 5분 검증")
        print("=" * 80)
        print(f"Base URL: {self.base_url}")
        print(f"Container: {self.container_name}")
        print(f"Timeout: {self.timeout}s")
        print("=" * 80)
        print()

        start_time = time.time()

        # Run API tests
        self.results.append(self.test_health_check())
        print()

        time.sleep(2)
        self.results.append(self.test_dna_create())
        print()

        time.sleep(2)
        self.results.append(self.test_evolve_options())
        print()

        time.sleep(2)
        self.results.append(self.test_evolve_feedback())
        print()

        # Wait for logs to be written
        time.sleep(3)

        # Run log verification
        self.results.append(self.test_llm_brief_logs())
        print()

        self.results.append(self.test_rl_update_logs())
        print()

        # Run metrics check
        self.results.append(self.test_metrics_endpoint())
        print()

        # Calculate duration
        duration = time.time() - start_time

        # Print summary
        self.print_summary(duration)

        # Determine overall pass/fail
        critical_tests = ['health_check', 'dna_create', 'llm_brief_logs']
        critical_failures = [r for r in self.results if r.name in critical_tests and not r.passed]

        return len(critical_failures) == 0

    def print_summary(self, duration: float):
        """Print test summary"""
        print("=" * 80)
        print("SMOKE TEST SUMMARY")
        print("=" * 80)
        print(f"Duration: {duration:.1f}s")
        print()

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print("Results:")
        print(f"  ✓ Passed: {passed}")
        print(f"  ✗ Failed: {failed}")
        print()

        print("Test Details:")
        for result in self.results:
            status = f"{Colors.GREEN}✓ PASS{Colors.NC}" if result.passed else f"{Colors.RED}✗ FAIL{Colors.NC}"
            print(f"  {status} {result.name}: {result.message}")
        print()

        if failed == 0:
            print("=" * 80)
            self.log_success("ALL TESTS PASSED - System is healthy ✓")
            print("=" * 80)
        else:
            print("=" * 80)
            self.log_error(f"TESTS FAILED - {failed} test(s) failed")
            print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Smoke test for Fragrance AI API and logs'
    )
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000',
        help='Base URL for API (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--container',
        default='fragrance-ai-app',
        help='Docker container name (default: fragrance-ai-app)'
    )
    parser.add_argument(
        '--canary',
        action='store_true',
        help='Test canary deployment (port 8001, container app-canary)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Request timeout in seconds (default: 30)'
    )

    args = parser.parse_args()

    # Adjust for canary if specified
    if args.canary:
        args.base_url = 'http://localhost:8001'
        args.container = 'fragrance-ai-app-canary'

    # Run tests
    runner = SmokeTestRunner(
        base_url=args.base_url,
        container_name=args.container,
        timeout=args.timeout
    )

    success = runner.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
