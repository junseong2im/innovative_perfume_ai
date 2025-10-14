#!/usr/bin/env python3
# scripts/check_licenses.py
"""
CLI Script to check model licenses during build
빌드 시 자동 라이선스 체크 스크립트
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fragrance_ai.security.license_checker import (
    LicenseChecker,
    KNOWN_LICENSES
)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check LLM model licenses for compliance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all default models
  python scripts/check_licenses.py

  # Check specific models
  python scripts/check_licenses.py --models Qwen/Qwen2.5-7B-Instruct

  # Generate markdown report
  python scripts/check_licenses.py --output LICENSE_REPORT.md

  # Check with custom MAU limit
  python scripts/check_licenses.py --monthly-users 1000000000
        """
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ],
        help="Model names to check (default: all ensemble models)"
    )
    parser.add_argument(
        "--commercial",
        action="store_true",
        default=True,
        help="Check for commercial use compliance (default: True)"
    )
    parser.add_argument(
        "--monthly-users",
        type=int,
        help="Monthly active users (for Llama restrictions)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit with error if warnings found"
    )
    parser.add_argument(
        "--list-known",
        action="store_true",
        help="List all known licenses and exit"
    )

    args = parser.parse_args()

    # List known licenses
    if args.list_known:
        print("\n" + "=" * 80)
        print("KNOWN MODEL LICENSES")
        print("=" * 80)
        for model_name, license_info in KNOWN_LICENSES.items():
            print(f"\n{model_name}")
            print(f"  License: {license_info.license_type.value}")
            print(f"  Commercial: {'Yes' if license_info.commercial_use else 'No'}")
            print(f"  URL: {license_info.source_url}")
        print("\n" + "=" * 80)
        return 0

    # Run license checks
    print("\n[*] Checking model licenses...")
    print(f"    Commercial use: {args.commercial}")
    if args.monthly_users:
        print(f"    Monthly users: {args.monthly_users:,}")

    checker = LicenseChecker()
    results = checker.check_all_models(
        model_names=args.models,
        commercial_use=args.commercial,
        monthly_users=args.monthly_users
    )

    # Print summary
    checker.print_license_summary(results)

    # Generate report if requested
    if args.output:
        print(f"\n[*] Generating license report...")
        report = checker.generate_license_report(results, args.output)
        print(f"    Report saved to: {args.output}")

    # Check compliance
    all_compliant = all(r.compliant for r in results.values())
    has_warnings = any(r.warnings for r in results.values())

    if not all_compliant:
        print("\n[FAIL] LICENSE CHECK FAILED: Some models are not compliant")
        return 1

    if has_warnings and args.fail_on_warning:
        print("\n[WARN] LICENSE CHECK WARNING: Warnings found (--fail-on-warning enabled)")
        return 1

    print("\n[PASS] LICENSE CHECK PASSED: All models are compliant")
    return 0


if __name__ == "__main__":
    exit(main())
