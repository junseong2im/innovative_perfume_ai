#!/usr/bin/env python3
"""
Security Compliance Verification Script
SBOM + License + Model Integrity + Secrets Validation

Usage:
    python scripts/verify_security_compliance.py --all
    python scripts/verify_security_compliance.py --licenses --models
    python scripts/verify_security_compliance.py --secrets
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fragrance_ai.security.license_checker import LicenseChecker, KNOWN_LICENSES
from fragrance_ai.security.model_integrity import (
    get_checksum_database,
    verify_model_integrity,
    list_registered_models
)
from fragrance_ai.security.secrets_manager import get_secrets_manager, REQUIRED_SECRETS


# =============================================================================
# Configuration
# =============================================================================

ARTISAN_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",           # Creative mode
    "mistralai/Mistral-7B-Instruct-v0.3", # Balanced mode
    "meta-llama/Meta-Llama-3-8B-Instruct" # Fast mode
]

# Model file paths (example - adjust to your setup)
MODEL_PATHS = {
    "qwen": "models/Qwen2.5-7B-Instruct",
    "mistral": "models/Mistral-7B-Instruct-v0.3",
    "llama": "models/Meta-Llama-3-8B-Instruct"
}


# =============================================================================
# License Compliance Check
# =============================================================================

def check_license_compliance(
    commercial_use: bool = True,
    monthly_users: int = None
) -> Tuple[bool, Dict]:
    """
    Check license compliance for all models

    Returns:
        (all_compliant, results_dict)
    """
    print("\n" + "="*80)
    print("LICENSE COMPLIANCE CHECK")
    print("="*80)

    checker = LicenseChecker()
    results = checker.check_all_models(
        model_names=ARTISAN_MODELS,
        commercial_use=commercial_use,
        monthly_users=monthly_users
    )

    checker.print_license_summary(results)

    all_compliant = all(r.compliant for r in results.values())

    if all_compliant:
        print("\n✅ All models are license compliant")
    else:
        print("\n❌ Some models have license issues")

    return all_compliant, results


# =============================================================================
# Model Integrity Check
# =============================================================================

def check_model_integrity() -> Tuple[bool, List[str]]:
    """
    Verify SHA256 checksums for all registered models

    Returns:
        (all_verified, failed_models)
    """
    print("\n" + "="*80)
    print("MODEL INTEGRITY CHECK (SHA256)")
    print("="*80)

    db = get_checksum_database()
    registered_models = list_registered_models()

    if not registered_models:
        print("\n⚠️  No models registered in checksum database")
        print("   Register models using:")
        print("   python -m fragrance_ai.security.model_integrity --register <name> <path> <license>")
        return True, []  # No models to verify = pass

    print(f"\nRegistered models: {len(registered_models)}")

    all_verified = True
    failed_models = []

    for model_checksum in registered_models:
        model_name = model_checksum.model_name
        file_path = model_checksum.file_path

        print(f"\nVerifying {model_name}...")
        print(f"  Path: {file_path}")
        print(f"  Expected SHA256: {model_checksum.sha256[:16]}...")

        result = verify_model_integrity(
            model_name=model_name,
            file_path=file_path,
            auto_register=False
        )

        if result.verified:
            print(f"  ✅ Integrity verified")
        else:
            print(f"  ❌ Integrity check FAILED")
            print(f"     {result.error_message}")
            all_verified = False
            failed_models.append(model_name)

    print("\n" + "-"*80)
    if all_verified:
        print("✅ All model checksums verified")
    else:
        print(f"❌ {len(failed_models)} model(s) failed verification:")
        for model in failed_models:
            print(f"   - {model}")

    return all_verified, failed_models


# =============================================================================
# SBOM Generation
# =============================================================================

def generate_sbom(output_path: str = "sbom.json") -> bool:
    """
    Generate Software Bill of Materials

    Returns:
        success
    """
    print("\n" + "="*80)
    print("SBOM GENERATION")
    print("="*80)

    try:
        from fragrance_ai.security.model_integrity import get_checksum_database

        db = get_checksum_database()
        registered_models = list_registered_models()

        if not registered_models:
            print("\n⚠️  No models registered - SBOM will be empty")

        # Generate SBOM
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
                "component": {
                    "type": "application",
                    "name": "Artisan Fragrance AI",
                    "version": "1.0.0",
                    "description": "AI-powered fragrance composition system"
                }
            },
            "components": []
        }

        # Add models
        for model_checksum in registered_models:
            # Get license info
            license_info = None
            for model_id, license_data in KNOWN_LICENSES.items():
                if model_checksum.model_name in model_id or model_id in model_checksum.model_name:
                    license_info = license_data
                    break

            component = {
                "type": "machine-learning-model",
                "name": model_checksum.model_name,
                "version": model_checksum.version or "1.0",
                "hashes": [
                    {
                        "alg": "SHA-256",
                        "content": model_checksum.sha256
                    }
                ],
                "properties": [
                    {
                        "name": "file_size_bytes",
                        "value": str(model_checksum.file_size_bytes)
                    },
                    {
                        "name": "last_verified",
                        "value": model_checksum.last_verified
                    }
                ]
            }

            if license_info:
                component["licenses"] = [
                    {
                        "license": {
                            "id": license_info.license_type.value,
                            "url": license_info.license_url
                        }
                    }
                ]

            if model_checksum.source:
                component["externalReferences"] = [
                    {
                        "type": "distribution",
                        "url": model_checksum.source
                    }
                ]

            sbom["components"].append(component)

        # Save SBOM
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sbom, f, indent=2, ensure_ascii=False)

        print(f"\n✅ SBOM generated: {output_path}")
        print(f"   Format: CycloneDX 1.4")
        print(f"   Components: {len(sbom['components'])}")

        return True

    except Exception as e:
        print(f"\n❌ SBOM generation failed: {e}")
        return False


# =============================================================================
# Secrets Validation
# =============================================================================

def check_secrets_compliance() -> Tuple[bool, List[str]]:
    """
    Validate all required secrets are present

    Returns:
        (all_present, missing_secrets)
    """
    print("\n" + "="*80)
    print("SECRETS VALIDATION")
    print("="*80)

    manager = get_secrets_manager()
    all_present, missing = manager.validate_secrets(REQUIRED_SECRETS)

    print(f"\nRequired secrets: {len(REQUIRED_SECRETS)}")
    print(f"Present: {len(REQUIRED_SECRETS) - len(missing)}")

    if all_present:
        print("\n✅ All required secrets are present")
    else:
        print(f"\n❌ Missing {len(missing)} required secret(s):")
        for secret_name in missing:
            print(f"   - {secret_name}")
        print("\nSet missing secrets using:")
        print("  - Environment variables")
        print("  - AWS Secrets Manager")
        print("  - KMS encryption")

    return all_present, missing


# =============================================================================
# Full Compliance Report
# =============================================================================

def generate_compliance_report(
    license_results: Dict,
    model_integrity_passed: bool,
    failed_models: List[str],
    secrets_passed: bool,
    missing_secrets: List[str],
    output_path: str = "compliance_report.md"
):
    """Generate comprehensive compliance report"""
    from datetime import datetime

    lines = []
    lines.append("# Security Compliance Report")
    lines.append(f"\nGenerated: {datetime.utcnow().isoformat()}")
    lines.append("\n---\n")

    # Summary
    lines.append("## Summary\n")

    license_passed = all(r.compliant for r in license_results.values())

    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    lines.append(f"| License Compliance | {'✅ PASS' if license_passed else '❌ FAIL'} |")
    lines.append(f"| Model Integrity | {'✅ PASS' if model_integrity_passed else '❌ FAIL'} |")
    lines.append(f"| Secrets Validation | {'✅ PASS' if secrets_passed else '❌ FAIL'} |")

    # Overall status
    overall_passed = license_passed and model_integrity_passed and secrets_passed

    lines.append(f"\n**Overall Status:** {'✅ COMPLIANT' if overall_passed else '❌ NON-COMPLIANT'}\n")

    # Detailed results
    lines.append("\n---\n")
    lines.append("## Detailed Results\n")

    # License compliance
    lines.append("### License Compliance\n")
    for model_name, result in license_results.items():
        status = "✅" if result.compliant else "❌"
        license_type = result.license_info.license_type.value if result.license_info else "Unknown"
        lines.append(f"{status} **{model_name}** - {license_type}")

        if result.warnings:
            for warning in result.warnings:
                lines.append(f"  - ⚠️  {warning}")
        if result.errors:
            for error in result.errors:
                lines.append(f"  - ❌ {error}")
        lines.append("")

    # Model integrity
    lines.append("\n### Model Integrity\n")
    if model_integrity_passed:
        lines.append("✅ All model checksums verified\n")
    else:
        lines.append(f"❌ {len(failed_models)} model(s) failed verification:\n")
        for model in failed_models:
            lines.append(f"- {model}\n")

    # Secrets
    lines.append("\n### Secrets Validation\n")
    if secrets_passed:
        lines.append("✅ All required secrets present\n")
    else:
        lines.append(f"❌ {len(missing_secrets)} missing secret(s):\n")
        for secret in missing_secrets:
            lines.append(f"- {secret}\n")

    # Recommendations
    if not overall_passed:
        lines.append("\n---\n")
        lines.append("## Recommendations\n")

        if not license_passed:
            lines.append("- Review and address license compliance issues\n")
        if not model_integrity_passed:
            lines.append("- Re-download or re-verify failed model files\n")
            lines.append("- Update checksum database if models were intentionally updated\n")
        if not secrets_passed:
            lines.append("- Set missing secrets in environment or secrets manager\n")

    report = "\n".join(lines)

    # Save report
    Path(output_path).write_text(report, encoding='utf-8')
    print(f"\n📄 Compliance report saved: {output_path}")

    return report


# =============================================================================
# Main CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Artisan Security Compliance Verification"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all security checks"
    )
    parser.add_argument(
        "--licenses",
        action="store_true",
        help="Check license compliance"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Verify model integrity (SHA256)"
    )
    parser.add_argument(
        "--secrets",
        action="store_true",
        help="Validate required secrets"
    )
    parser.add_argument(
        "--sbom",
        help="Generate SBOM (path)",
        metavar="OUTPUT_PATH"
    )
    parser.add_argument(
        "--report",
        help="Generate compliance report (path)",
        metavar="OUTPUT_PATH"
    )
    parser.add_argument(
        "--commercial",
        action="store_true",
        default=True,
        help="Check for commercial use (default: True)"
    )
    parser.add_argument(
        "--monthly-users",
        type=int,
        help="Monthly active users (for Llama restrictions)"
    )

    args = parser.parse_args()

    # If no specific check, run all
    if not any([args.licenses, args.models, args.secrets, args.sbom]):
        args.all = True

    exit_code = 0
    license_results = {}
    model_integrity_passed = True
    failed_models = []
    secrets_passed = True
    missing_secrets = []

    print("\n" + "="*80)
    print("ARTISAN SECURITY COMPLIANCE VERIFICATION")
    print("="*80)

    try:
        # License compliance
        if args.all or args.licenses:
            passed, license_results = check_license_compliance(
                commercial_use=args.commercial,
                monthly_users=args.monthly_users
            )
            if not passed:
                exit_code = 1

        # Model integrity
        if args.all or args.models:
            model_integrity_passed, failed_models = check_model_integrity()
            if not model_integrity_passed:
                exit_code = 1

        # Secrets validation
        if args.all or args.secrets:
            secrets_passed, missing_secrets = check_secrets_compliance()
            if not secrets_passed:
                exit_code = 1

        # SBOM generation
        if args.sbom or args.all:
            sbom_path = args.sbom or "sbom.json"
            sbom_success = generate_sbom(sbom_path)
            if not sbom_success:
                exit_code = 1

        # Compliance report
        if args.report or (args.all and license_results):
            report_path = args.report or "compliance_report.md"
            generate_compliance_report(
                license_results=license_results,
                model_integrity_passed=model_integrity_passed,
                failed_models=failed_models,
                secrets_passed=secrets_passed,
                missing_secrets=missing_secrets,
                output_path=report_path
            )

        # Final summary
        print("\n" + "="*80)
        if exit_code == 0:
            print("✅ SECURITY COMPLIANCE: PASS")
        else:
            print("❌ SECURITY COMPLIANCE: FAIL")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
