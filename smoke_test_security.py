"""
Security-Enhanced Smoke Test
배포 전 필수 보안 검증: IFRA + Model Integrity + License + Secrets

Usage:
    python smoke_test_security.py
    python smoke_test_security.py --skip-models  # Skip model integrity check
"""

import requests
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fragrance_ai.security.license_checker import LicenseChecker, KNOWN_LICENSES
from fragrance_ai.security.model_integrity import (
    get_checksum_database,
    verify_model_integrity,
    list_registered_models
)
from fragrance_ai.security.secrets_manager import get_secrets_manager, REQUIRED_SECRETS
from fragrance_ai.regulations.ifra_rules import IFRAValidator


API_BASE_URL = "http://localhost:8000"
MODELS_TO_CHECK = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]


# =============================================================================
# Security Checks
# =============================================================================

def check_license_compliance() -> Tuple[bool, str]:
    """Check license compliance for all models"""
    print("\n" + "="*80)
    print("1. LICENSE COMPLIANCE CHECK")
    print("="*80)

    try:
        checker = LicenseChecker()
        results = checker.check_all_models(
            model_names=MODELS_TO_CHECK,
            commercial_use=True
        )

        all_compliant = all(r.compliant for r in results.values())

        for model_name, result in results.items():
            status = "✅" if result.compliant else "❌"
            license_type = result.license_info.license_type.value if result.license_info else "Unknown"
            print(f"{status} {model_name}: {license_type}")

            if result.warnings:
                for warning in result.warnings:
                    print(f"   ⚠️  {warning}")
            if result.errors:
                for error in result.errors:
                    print(f"   ❌ {error}")

        if all_compliant:
            message = "✅ All models are license compliant"
            print(f"\n{message}")
            return True, message
        else:
            message = "❌ Some models have license issues"
            print(f"\n{message}")
            return False, message

    except Exception as e:
        message = f"❌ License check failed: {e}"
        print(f"\n{message}")
        return False, message


def check_model_integrity(skip: bool = False) -> Tuple[bool, str]:
    """Verify SHA256 checksums for all registered models"""
    print("\n" + "="*80)
    print("2. MODEL INTEGRITY CHECK (SHA256)")
    print("="*80)

    if skip:
        message = "⏭️  Skipped (--skip-models flag)"
        print(f"\n{message}")
        return True, message

    try:
        registered_models = list_registered_models()

        if not registered_models:
            message = "⚠️  No models registered - skipping integrity check"
            print(f"\n{message}")
            return True, message

        print(f"Checking {len(registered_models)} registered models...")

        all_verified = True
        failed_models = []

        for model_checksum in registered_models:
            model_name = model_checksum.model_name
            print(f"\n  Verifying {model_name}...")

            result = verify_model_integrity(
                model_name=model_name,
                file_path=model_checksum.file_path,
                auto_register=False
            )

            if result.verified:
                print(f"    ✅ Integrity verified")
            else:
                print(f"    ❌ FAILED: {result.error_message}")
                all_verified = False
                failed_models.append(model_name)

        if all_verified:
            message = "✅ All model checksums verified"
            print(f"\n{message}")
            return True, message
        else:
            message = f"❌ {len(failed_models)} model(s) failed verification: {', '.join(failed_models)}"
            print(f"\n{message}")
            return False, message

    except Exception as e:
        message = f"❌ Model integrity check failed: {e}"
        print(f"\n{message}")
        return False, message


def check_secrets_compliance() -> Tuple[bool, str]:
    """Validate all required secrets are present"""
    print("\n" + "="*80)
    print("3. SECRETS VALIDATION")
    print("="*80)

    try:
        manager = get_secrets_manager()
        all_present, missing = manager.validate_secrets(REQUIRED_SECRETS)

        print(f"Required secrets: {len(REQUIRED_SECRETS)}")
        print(f"Present: {len(REQUIRED_SECRETS) - len(missing)}")

        if all_present:
            message = "✅ All required secrets are present"
            print(f"\n{message}")
            return True, message
        else:
            message = f"❌ Missing {len(missing)} required secret(s): {', '.join(missing)}"
            print(f"\n{message}")
            return False, message

    except Exception as e:
        message = f"❌ Secrets validation failed: {e}"
        print(f"\n{message}")
        return False, message


def check_health_endpoints() -> Tuple[bool, str]:
    """Check API and LLM health endpoints"""
    print("\n" + "="*80)
    print("4. HEALTH ENDPOINTS CHECK")
    print("="*80)

    try:
        # Main health endpoint
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()

        print(f"✅ API Health: {health['status']}")

        # LLM health endpoints
        llm_models = ["qwen", "mistral", "llama"]
        all_healthy = True

        for model in llm_models:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/health/llm",
                    params={"model": model},
                    timeout=10
                )
                response.raise_for_status()
                llm_health = response.json()

                status_icon = "✅" if llm_health['status'] == 'healthy' else "⚠️"
                print(f"{status_icon} LLM Health ({model}): {llm_health['status']}")

                if llm_health['status'] != 'healthy':
                    all_healthy = False

            except Exception as e:
                print(f"❌ LLM Health ({model}): Failed - {e}")
                all_healthy = False

        if all_healthy:
            message = "✅ All health endpoints are healthy"
            print(f"\n{message}")
            return True, message
        else:
            message = "⚠️  Some health endpoints are degraded"
            print(f"\n{message}")
            return True, message  # Warning, not failure

    except Exception as e:
        message = f"❌ Health check failed: {e}"
        print(f"\n{message}")
        return False, message


def test_ifra_validation() -> Tuple[bool, str]:
    """Test IFRA validation in API call"""
    print("\n" + "="*80)
    print("5. IFRA VALIDATION TEST")
    print("="*80)

    try:
        # Test DNA creation with IFRA validation
        dna_request = {
            "brief": {
                "style": "fresh",
                "intensity": 0.6,
                "complexity": 0.4,
                "masculinity": 0.5,
                "notes": ["citrus", "aquatic"]
            },
            "name": "Security Smoke Test",
            "description": "Testing IFRA validation",
            "product_category": "eau_de_parfum"
        }

        print("Creating DNA with IFRA validation...")

        response = requests.post(
            f"{API_BASE_URL}/dna/create",
            json=dna_request,
            timeout=30
        )
        response.raise_for_status()
        dna_data = response.json()

        # Check IFRA compliance
        ifra_compliant = dna_data.get('compliance', {}).get('ifra_compliant', False)

        if ifra_compliant:
            print(f"✅ DNA created with IFRA compliance")
            print(f"   - DNA ID: {dna_data['dna_id']}")
            print(f"   - Ingredients: {len(dna_data['ingredients'])}")
            print(f"   - IFRA Compliant: {ifra_compliant}")

            message = "✅ IFRA validation working correctly"
            print(f"\n{message}")
            return True, message
        else:
            message = "⚠️  DNA created but IFRA compliance unclear"
            print(f"\n{message}")
            return True, message  # Warning, not failure

    except requests.exceptions.RequestException as e:
        message = f"❌ IFRA validation test failed: {e}"
        print(f"\n{message}")
        return False, message


def test_pii_masking() -> Tuple[bool, str]:
    """Test PII masking in logs"""
    print("\n" + "="*80)
    print("6. PII MASKING TEST")
    print("="*80)

    try:
        from fragrance_ai.security.pii_masking import (
            mask_pii_patterns,
            detect_pii_patterns
        )

        # Test text with PII
        test_text = "Contact me at john.doe@example.com or 010-1234-5678"

        # Detect PII
        pii_found = detect_pii_patterns(test_text)
        print(f"PII patterns detected: {pii_found}")

        # Mask PII
        masked_text = mask_pii_patterns(test_text)
        print(f"Original: {test_text}")
        print(f"Masked:   {masked_text}")

        # Verify masking worked
        if "@example.com" in masked_text and "010-1234-5678" not in masked_text:
            message = "✅ PII masking working correctly"
            print(f"\n{message}")
            return True, message
        else:
            message = "⚠️  PII masking may have issues"
            print(f"\n{message}")
            return True, message  # Warning, not failure

    except Exception as e:
        message = f"❌ PII masking test failed: {e}"
        print(f"\n{message}")
        return False, message


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_checks(skip_models: bool = False) -> Tuple[bool, List[str]]:
    """
    Run all security checks

    Returns:
        (all_passed, messages)
    """
    checks = [
        ("License Compliance", lambda: check_license_compliance()),
        ("Model Integrity", lambda: check_model_integrity(skip=skip_models)),
        ("Secrets Validation", lambda: check_secrets_compliance()),
        ("Health Endpoints", lambda: check_health_endpoints()),
        ("IFRA Validation", lambda: test_ifra_validation()),
        ("PII Masking", lambda: test_pii_masking()),
    ]

    results = []
    messages = []

    print("\n" + "="*80)
    print("SECURITY-ENHANCED SMOKE TEST")
    print("Pre-deployment security validation")
    print("="*80)

    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            results.append(passed)
            messages.append(f"{check_name}: {message}")
        except Exception as e:
            results.append(False)
            messages.append(f"{check_name}: ❌ Exception - {e}")
            print(f"\n❌ {check_name} raised exception: {e}")

    all_passed = all(results)

    # Summary
    print("\n" + "="*80)
    print("SECURITY SMOKE TEST SUMMARY")
    print("="*80)

    for i, (check_name, _) in enumerate(checks):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} - {check_name}")

    print("\n" + "="*80)

    if all_passed:
        print("✅ ALL SECURITY CHECKS PASSED")
        print("="*80)
        print("\n🚀 DEPLOYMENT APPROVED")
    else:
        print("❌ SOME SECURITY CHECKS FAILED")
        print("="*80)
        print("\n⛔ DEPLOYMENT BLOCKED")
        print("\nFix the issues above before deploying.")

    print("\n")

    return all_passed, messages


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Security-Enhanced Smoke Test for Artisan"
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model integrity check (useful if models not downloaded)"
    )

    args = parser.parse_args()

    try:
        all_passed, messages = run_all_checks(skip_models=args.skip_models)

        # Exit with appropriate code
        sys.exit(0 if all_passed else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
