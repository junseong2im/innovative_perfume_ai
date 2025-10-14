# fragrance_ai/security/license_checker.py
"""
License Verification for LLM Models
Qwen/Mistral/Llama 라이선스·배포 조건 검증
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# License Types
# ============================================================================

class LicenseType(str, Enum):
    """Known license types"""
    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    LLAMA_2 = "Llama-2-Community"
    LLAMA_3 = "Llama-3-Community"
    MISTRAL_APACHE = "Mistral-Apache-2.0"
    QWEN_TONGYI = "Qwen-Tongyi"
    UNKNOWN = "Unknown"


# ============================================================================
# License Information
# ============================================================================

@dataclass
class LicenseInfo:
    """License information for a model"""
    model_name: str
    license_type: LicenseType
    commercial_use: bool
    attribution_required: bool
    modification_allowed: bool
    distribution_allowed: bool
    restrictions: List[str]
    source_url: str
    license_url: Optional[str] = None
    notes: Optional[str] = None


# Known model licenses
KNOWN_LICENSES: Dict[str, LicenseInfo] = {
    "Qwen/Qwen2.5-7B-Instruct": LicenseInfo(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        license_type=LicenseType.QWEN_TONGYI,
        commercial_use=True,
        attribution_required=True,
        modification_allowed=True,
        distribution_allowed=True,
        restrictions=[
            "Must provide attribution to Alibaba Cloud",
            "Cannot use for illegal purposes",
            "Cannot harm Alibaba Cloud's reputation"
        ],
        source_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
        license_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE",
        notes="Tongyi Qianwen License Agreement - free for commercial and research use"
    ),

    "mistralai/Mistral-7B-Instruct-v0.3": LicenseInfo(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        license_type=LicenseType.APACHE_2_0,
        commercial_use=True,
        attribution_required=True,
        modification_allowed=True,
        distribution_allowed=True,
        restrictions=[
            "Must include Apache 2.0 license notice",
            "State changes if modified",
            "No trademark use without permission"
        ],
        source_url="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        notes="Apache 2.0 - permissive license for commercial use"
    ),

    "meta-llama/Meta-Llama-3-8B-Instruct": LicenseInfo(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        license_type=LicenseType.LLAMA_3,
        commercial_use=True,
        attribution_required=True,
        modification_allowed=True,
        distribution_allowed=True,
        restrictions=[
            "Monthly active users > 700M require Meta license",
            "Cannot use to train other LLMs",
            "Must include Meta attribution",
            "Comply with Meta's Acceptable Use Policy"
        ],
        source_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct",
        license_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct/blob/main/LICENSE",
        notes="Llama 3 Community License - free for most commercial use"
    ),

    "meta-llama/Llama-2-7b-chat-hf": LicenseInfo(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        license_type=LicenseType.LLAMA_2,
        commercial_use=True,
        attribution_required=True,
        modification_allowed=True,
        distribution_allowed=True,
        restrictions=[
            "Monthly active users > 700M require Meta license",
            "Cannot use to improve other LLMs",
            "Comply with Acceptable Use Policy"
        ],
        source_url="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
        license_url="https://ai.meta.com/llama/license/",
        notes="Llama 2 Community License"
    )
}


# ============================================================================
# License Checker
# ============================================================================

@dataclass
class LicenseCheckResult:
    """Result of license verification"""
    model_name: str
    compliant: bool
    license_info: Optional[LicenseInfo]
    warnings: List[str]
    errors: List[str]
    checked_at: str


class LicenseChecker:
    """Verify license compliance for models"""

    def __init__(self, licenses_db: Optional[Dict[str, LicenseInfo]] = None):
        """
        Initialize license checker

        Args:
            licenses_db: Custom license database (uses KNOWN_LICENSES if None)
        """
        self.licenses_db = licenses_db or KNOWN_LICENSES

    def check_model_license(
        self,
        model_name: str,
        commercial_use: bool = True,
        monthly_users: Optional[int] = None
    ) -> LicenseCheckResult:
        """
        Check license compliance for model

        Args:
            model_name: Model identifier
            commercial_use: Whether using for commercial purposes
            monthly_users: Monthly active users (for Llama restrictions)

        Returns:
            LicenseCheckResult
        """
        from datetime import datetime

        warnings = []
        errors = []

        # Look up license info
        license_info = self.licenses_db.get(model_name)

        if license_info is None:
            errors.append(f"License information not found for model: {model_name}")
            return LicenseCheckResult(
                model_name=model_name,
                compliant=False,
                license_info=None,
                warnings=warnings,
                errors=errors,
                checked_at=datetime.utcnow().isoformat()
            )

        # Check commercial use
        if commercial_use and not license_info.commercial_use:
            errors.append(
                f"{model_name} does not allow commercial use "
                f"(License: {license_info.license_type.value})"
            )

        # Check Llama-specific restrictions
        if license_info.license_type in [LicenseType.LLAMA_2, LicenseType.LLAMA_3]:
            if monthly_users and monthly_users > 700_000_000:
                warnings.append(
                    f"Monthly active users ({monthly_users:,}) exceeds 700M - "
                    f"Meta license required for {model_name}"
                )

        # Check restrictions
        if license_info.restrictions:
            for restriction in license_info.restrictions:
                warnings.append(f"[{model_name}] {restriction}")

        # Compliance determination
        compliant = len(errors) == 0

        return LicenseCheckResult(
            model_name=model_name,
            compliant=compliant,
            license_info=license_info,
            warnings=warnings,
            errors=errors,
            checked_at=datetime.utcnow().isoformat()
        )

    def check_all_models(
        self,
        model_names: List[str],
        commercial_use: bool = True,
        monthly_users: Optional[int] = None
    ) -> Dict[str, LicenseCheckResult]:
        """
        Check licenses for multiple models

        Args:
            model_names: List of model identifiers
            commercial_use: Commercial use flag
            monthly_users: Monthly active users

        Returns:
            Dictionary of {model_name: LicenseCheckResult}
        """
        results = {}

        for model_name in model_names:
            result = self.check_model_license(
                model_name=model_name,
                commercial_use=commercial_use,
                monthly_users=monthly_users
            )
            results[model_name] = result

        return results

    def print_license_summary(self, results: Dict[str, LicenseCheckResult]):
        """Print human-readable license summary"""
        print("\n" + "=" * 80)
        print("LICENSE COMPLIANCE SUMMARY")
        print("=" * 80)

        for model_name, result in results.items():
            status = "[PASS] COMPLIANT" if result.compliant else "[FAIL] NON-COMPLIANT"
            print(f"\n{status}: {model_name}")

            if result.license_info:
                info = result.license_info
                print(f"  License: {info.license_type.value}")
                print(f"  Commercial Use: {'Yes' if info.commercial_use else 'No'}")
                print(f"  Attribution Required: {'Yes' if info.attribution_required else 'No'}")
                print(f"  Source: {info.source_url}")

                if info.license_url:
                    print(f"  License URL: {info.license_url}")

            # Print warnings
            if result.warnings:
                print("  Warnings:")
                for warning in result.warnings:
                    print(f"    [WARN] {warning}")

            # Print errors
            if result.errors:
                print("  Errors:")
                for error in result.errors:
                    print(f"    [ERROR] {error}")

        print("\n" + "=" * 80)

    def generate_license_report(
        self,
        results: Dict[str, LicenseCheckResult],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed license report

        Args:
            results: License check results
            output_path: Path to save report (optional)

        Returns:
            Report as markdown string
        """
        from datetime import datetime

        lines = []
        lines.append("# Model License Compliance Report")
        lines.append(f"\nGenerated: {datetime.utcnow().isoformat()}")
        lines.append("\n---\n")

        # Summary table
        lines.append("## Summary\n")
        lines.append("| Model | License | Commercial | Compliant |")
        lines.append("|-------|---------|------------|-----------|")

        for model_name, result in results.items():
            license_type = result.license_info.license_type.value if result.license_info else "Unknown"
            commercial = "✓" if result.license_info and result.license_info.commercial_use else "✗"
            compliant = "✓" if result.compliant else "✗"
            lines.append(f"| {model_name} | {license_type} | {commercial} | {compliant} |")

        # Detailed information
        lines.append("\n---\n")
        lines.append("## Detailed Information\n")

        for model_name, result in results.items():
            lines.append(f"\n### {model_name}\n")

            if result.license_info:
                info = result.license_info
                lines.append(f"**License Type:** {info.license_type.value}\n")
                lines.append(f"**Commercial Use:** {'Yes' if info.commercial_use else 'No'}\n")
                lines.append(f"**Attribution Required:** {'Yes' if info.attribution_required else 'No'}\n")
                lines.append(f"**Modification Allowed:** {'Yes' if info.modification_allowed else 'No'}\n")
                lines.append(f"**Distribution Allowed:** {'Yes' if info.distribution_allowed else 'No'}\n")
                lines.append(f"\n**Source:** [{info.source_url}]({info.source_url})\n")

                if info.license_url:
                    lines.append(f"**License:** [{info.license_url}]({info.license_url})\n")

                if info.restrictions:
                    lines.append("\n**Restrictions:**\n")
                    for restriction in info.restrictions:
                        lines.append(f"- {restriction}\n")

                if info.notes:
                    lines.append(f"\n**Notes:** {info.notes}\n")

            if result.warnings:
                lines.append("\n**Warnings:**\n")
                for warning in result.warnings:
                    lines.append(f"- WARNING: {warning}\n")

            if result.errors:
                lines.append("\n**Errors:**\n")
                for error in result.errors:
                    lines.append(f"- ERROR: {error}\n")

        report = "\n".join(lines)

        # Save to file if path provided
        if output_path:
            Path(output_path).write_text(report, encoding='utf-8')
            logger.info(f"License report saved to {output_path}")

        return report


# ============================================================================
# CLI Tool
# ============================================================================

def check_licenses_cli():
    """CLI tool to check model licenses"""
    import argparse

    parser = argparse.ArgumentParser(description="Check LLM model licenses")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ],
        help="Model names to check"
    )
    parser.add_argument(
        "--commercial",
        action="store_true",
        default=True,
        help="Check for commercial use"
    )
    parser.add_argument(
        "--monthly-users",
        type=int,
        help="Monthly active users (for Llama restrictions)"
    )
    parser.add_argument(
        "--output",
        help="Output path for markdown report"
    )

    args = parser.parse_args()

    # Run checks
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
        report = checker.generate_license_report(results, args.output)
        print(f"\nReport saved to: {args.output}")

    # Exit with error if any non-compliant
    if any(not r.compliant for r in results.values()):
        exit(1)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'LicenseType',
    'LicenseInfo',
    'KNOWN_LICENSES',
    'LicenseCheckResult',
    'LicenseChecker',
    'check_licenses_cli'
]
