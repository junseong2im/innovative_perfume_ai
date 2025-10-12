#!/usr/bin/env python
# run_tests.py
"""
Comprehensive test runner for Fragrance AI
Runs all tests with proper configuration and reporting
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_suite="all", verbose=False, coverage=False, markers=None):
    """
    Run test suite

    Args:
        test_suite: Which tests to run (all, unit, integration, ga, rl, ifra, api)
        verbose: Verbose output
        coverage: Generate coverage report
        markers: Additional pytest markers
    """

    # Base pytest command
    cmd = ["pytest"]

    # Select test files based on suite
    if test_suite == "all":
        cmd.append("tests/")
    elif test_suite == "unit":
        cmd.extend(["tests/test_ga.py", "tests/test_rl.py", "tests/test_ifra.py"])
    elif test_suite == "integration":
        cmd.append("tests/test_api.py")
    elif test_suite == "ga":
        cmd.append("tests/test_ga.py")
    elif test_suite == "rl":
        cmd.append("tests/test_rl.py")
    elif test_suite == "ifra":
        cmd.append("tests/test_ifra.py")
    elif test_suite == "api":
        cmd.append("tests/test_api.py")
    else:
        print(f"Unknown test suite: {test_suite}")
        return 1

    # Add verbosity
    if verbose:
        cmd.append("-v")
        cmd.append("-s")  # Show print statements

    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=fragrance_ai",
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Add markers
    if markers:
        cmd.extend(["-m", markers])

    # Add other useful options
    cmd.extend([
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "-ra",  # Show all test results
    ])

    # Print command
    print("="*70)
    print("RUNNING TESTS")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print("="*70)
    print()

    # Run tests
    result = subprocess.run(cmd)

    return result.returncode


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Fragrance AI tests")

    parser.add_argument(
        "suite",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "ga", "rl", "ifra", "api"],
        help="Test suite to run"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "-m", "--markers",
        type=str,
        help="Pytest markers (e.g., 'not slow')"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick tests only (skip slow tests)"
    )

    args = parser.parse_args()

    # Set markers for quick mode
    if args.quick:
        args.markers = "not slow"

    # Run tests
    exit_code = run_tests(
        test_suite=args.suite,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()