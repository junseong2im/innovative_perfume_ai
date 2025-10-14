#!/usr/bin/env python3
"""
Pre-Deployment Checklist (T-1)
프로덕션 배포 전 최종 검증 체크리스트

Usage:
    python scripts/pre_deployment_check.py --version v0.2.0
    python scripts/pre_deployment_check.py --version v0.2.0 --strict
"""

import argparse
import json
import hashlib
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import yaml


# =============================================================================
# Configuration
# =============================================================================

REQUIRED_ENV_VARS = [
    "SECRET_KEY",
    "POSTGRES_PASSWORD",
    "DATABASE_URL",
    "REDIS_URL"
]

OPTIONAL_ENV_VARS = [
    "SENTRY_DSN",
    "GRAFANA_PASSWORD",
    "CORS_ORIGINS"
]

MODEL_PATHS = [
    "models/",
    "checkpoints/"
]

CONTAINER_SERVICES = [
    "app",
    "worker-llm",
    "worker-rl",
    "postgres",
    "redis"
]

# Colors
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


# =============================================================================
# Helper Functions
# =============================================================================

def log_info(msg: str):
    """Info log"""
    print(f"{BLUE}[INFO]{NC} {msg}")


def log_success(msg: str):
    """Success log"""
    print(f"{GREEN}[✓ PASS]{NC} {msg}")


def log_warning(msg: str):
    """Warning log"""
    print(f"{YELLOW}[⚠ WARN]{NC} {msg}")


def log_error(msg: str):
    """Error log"""
    print(f"{RED}[✗ FAIL]{NC} {msg}")


def run_command(cmd: List[str], capture_output=True) -> Tuple[int, str, str]:
    """Run shell command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timeout"
    except Exception as e:
        return 1, "", str(e)


def calculate_file_hash(file_path: Path) -> Optional[str]:
    """Calculate SHA256 hash of file"""
    if not file_path.exists():
        return None

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# =============================================================================
# Check 1: Release Tag Verification
# =============================================================================

def check_release_tag(version: str) -> Tuple[bool, str]:
    """
    Check 1: 릴리스 태그 검증
    - Git tag exists
    - Tag format is valid (vX.Y.Z)
    - Model checkpoints are tagged
    """
    log_info("=" * 80)
    log_info("CHECK 1: Release Tag Verification")
    log_info("=" * 80)

    checks_passed = []
    checks_failed = []

    # 1.1 Validate version format
    version_pattern = r'^v\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$'
    if not re.match(version_pattern, version):
        msg = f"Invalid version format: {version}. Expected: vX.Y.Z or vX.Y.Z-suffix"
        log_error(msg)
        checks_failed.append(msg)
    else:
        log_success(f"Version format valid: {version}")
        checks_passed.append("version_format")

    # 1.2 Check if git tag exists
    returncode, stdout, stderr = run_command(['git', 'tag', '-l', version])
    if returncode == 0 and version in stdout:
        log_success(f"Git tag exists: {version}")
        checks_passed.append("git_tag_exists")
    else:
        msg = f"Git tag does not exist: {version}"
        log_warning(msg)
        log_info("  Create tag with: git tag -a {version} -m 'Release {version}'")
        checks_failed.append(msg)

    # 1.3 Check model checkpoint tags
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
        if checkpoint_files:
            log_info(f"Found {len(checkpoint_files)} checkpoint files")

            # Check if checkpoints have version tags
            version_tag_file = checkpoint_dir / f"VERSION_{version}.txt"
            if version_tag_file.exists():
                log_success(f"Model checkpoint version tag exists: {version_tag_file}")
                checks_passed.append("checkpoint_tag")
            else:
                msg = f"Model checkpoint version tag missing: {version_tag_file}"
                log_warning(msg)
                log_info(f"  Create with: echo '{version}' > {version_tag_file}")
                checks_failed.append(msg)
        else:
            log_info("No checkpoint files found (may be intentional)")
    else:
        log_info("No checkpoints directory (may be intentional)")

    # 1.4 Check git commit is tagged
    returncode, stdout, stderr = run_command(['git', 'rev-parse', 'HEAD'])
    if returncode == 0:
        commit_hash = stdout.strip()[:8]
        log_info(f"Current commit: {commit_hash}")
        checks_passed.append("commit_identified")

    # Summary
    log_info("")
    if checks_failed:
        log_error(f"Release tag verification: {len(checks_failed)} issue(s) found")
        return False, "\n".join(checks_failed)
    else:
        log_success(f"Release tag verification: All checks passed ({len(checks_passed)} checks)")
        return True, "OK"


# =============================================================================
# Check 2: Migration Verification
# =============================================================================

def check_migrations() -> Tuple[bool, str]:
    """
    Check 2: 마이그레이션 검증
    - Database schema changes
    - Redis schema changes
    - Alembic migration status
    """
    log_info("")
    log_info("=" * 80)
    log_info("CHECK 2: Migration Verification")
    log_info("=" * 80)

    checks_passed = []
    checks_failed = []

    # 2.1 Check alembic migrations directory
    alembic_dir = Path("alembic/versions")
    if alembic_dir.exists():
        migration_files = list(alembic_dir.glob("*.py"))
        log_info(f"Found {len(migration_files)} migration files")

        # Check if there are pending migrations
        # (This would require database connection in real scenario)
        log_info("Migration files present")
        checks_passed.append("migrations_exist")

        # Check for unreviewd migrations (by checking git status)
        returncode, stdout, stderr = run_command(['git', 'status', '--porcelain', 'alembic/versions'])
        if stdout.strip():
            msg = "Uncommitted migration files detected"
            log_warning(msg)
            log_info(f"  Files: {stdout.strip()}")
            checks_failed.append(msg)
        else:
            log_success("All migrations are committed")
            checks_passed.append("migrations_committed")
    else:
        log_info("No alembic migrations directory found")

    # 2.2 Check for schema documentation
    schema_docs = [
        Path("docs/database_schema.md"),
        Path("README.md"),
        Path("DEPLOYMENT_GUIDE.md")
    ]

    schema_documented = False
    for doc_path in schema_docs:
        if doc_path.exists():
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'schema' in content.lower() or 'migration' in content.lower():
                    log_success(f"Schema documentation found: {doc_path}")
                    schema_documented = True
                    checks_passed.append(f"schema_doc_{doc_path.name}")
                    break

    if not schema_documented:
        msg = "Schema changes not documented"
        log_warning(msg)
        log_info("  Add schema change notes to DEPLOYMENT_GUIDE.md")
        checks_failed.append(msg)

    # 2.3 Check Redis schema changes
    # Look for redis-related files that changed
    returncode, stdout, stderr = run_command(['git', 'diff', '--name-only', 'HEAD~5..HEAD'])
    if returncode == 0:
        redis_files = [f for f in stdout.split('\n') if 'redis' in f.lower() or 'cache' in f.lower()]
        if redis_files:
            log_warning(f"Redis-related files changed: {len(redis_files)} files")
            for f in redis_files[:5]:
                log_info(f"  - {f}")
            log_info("  Review Redis schema compatibility")
        else:
            log_success("No Redis schema changes detected")
            checks_passed.append("redis_schema")

    # Summary
    log_info("")
    if checks_failed:
        log_error(f"Migration verification: {len(checks_failed)} issue(s) found")
        return False, "\n".join(checks_failed)
    else:
        log_success(f"Migration verification: All checks passed ({len(checks_passed)} checks)")
        return True, "OK"


# =============================================================================
# Check 3: Secret Validation
# =============================================================================

def check_secrets(env_file: str = ".env") -> Tuple[bool, str]:
    """
    Check 3: 시크릿 검증
    - .env file exists
    - Required variables are set
    - No empty values
    - No default/example values
    """
    log_info("")
    log_info("=" * 80)
    log_info("CHECK 3: Secret Validation")
    log_info("=" * 80)

    checks_passed = []
    checks_failed = []

    # 3.1 Check .env file exists
    env_path = Path(env_file)
    if not env_path.exists():
        msg = f"Environment file not found: {env_file}"
        log_error(msg)
        checks_failed.append(msg)
        return False, msg

    log_success(f"Environment file exists: {env_file}")
    checks_passed.append("env_file_exists")

    # 3.2 Load environment variables
    env_vars = {}
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()

    log_info(f"Loaded {len(env_vars)} environment variables")

    # 3.3 Check required variables
    for var in REQUIRED_ENV_VARS:
        if var not in env_vars:
            msg = f"Required variable missing: {var}"
            log_error(msg)
            checks_failed.append(msg)
        elif not env_vars[var]:
            msg = f"Required variable is empty: {var}"
            log_error(msg)
            checks_failed.append(msg)
        elif any(default in env_vars[var].lower() for default in ['changeme', 'example', 'your_', 'placeholder']):
            msg = f"Variable has default/example value: {var}"
            log_error(msg)
            log_info(f"  Current value: {env_vars[var][:20]}...")
            checks_failed.append(msg)
        else:
            log_success(f"✓ {var}")
            checks_passed.append(f"secret_{var}")

    # 3.4 Check optional variables
    log_info("")
    log_info("Optional variables:")
    for var in OPTIONAL_ENV_VARS:
        if var in env_vars and env_vars[var]:
            log_info(f"  ✓ {var} (set)")
        else:
            log_info(f"  - {var} (not set)")

    # 3.5 Check SECRET_KEY strength
    if 'SECRET_KEY' in env_vars:
        secret_key = env_vars['SECRET_KEY']
        if len(secret_key) < 32:
            msg = f"SECRET_KEY too short: {len(secret_key)} chars (minimum 32)"
            log_error(msg)
            checks_failed.append(msg)
        else:
            log_success(f"SECRET_KEY length OK: {len(secret_key)} chars")
            checks_passed.append("secret_key_length")

    # 3.6 Check for hardcoded secrets in code
    returncode, stdout, stderr = run_command(['git', 'grep', '-i', '-E', 'password.*=|secret.*=|api_key.*='])
    if returncode == 0 and stdout:
        suspicious_lines = [line for line in stdout.split('\n') if line and 'env' not in line.lower()]
        if suspicious_lines:
            msg = f"Potential hardcoded secrets found: {len(suspicious_lines)} occurrences"
            log_warning(msg)
            for line in suspicious_lines[:3]:
                log_info(f"  {line[:80]}")
            checks_failed.append(msg)

    # Summary
    log_info("")
    if checks_failed:
        log_error(f"Secret validation: {len(checks_failed)} issue(s) found")
        return False, "\n".join(checks_failed)
    else:
        log_success(f"Secret validation: All checks passed ({len(checks_passed)} checks)")
        return True, "OK"


# =============================================================================
# Check 4: Artifact Hash Recording
# =============================================================================

def check_artifacts(version: str) -> Tuple[bool, str]:
    """
    Check 4: 아티팩트 해시 기록
    - Model weights SHA256
    - Container images SHA256
    - Record to manifest file
    """
    log_info("")
    log_info("=" * 80)
    log_info("CHECK 4: Artifact Hash Recording")
    log_info("=" * 80)

    checks_passed = []
    checks_failed = []
    artifacts = {}

    # 4.1 Calculate model hashes
    log_info("Calculating model file hashes...")
    model_hashes = {}

    for model_dir in MODEL_PATHS:
        model_path = Path(model_dir)
        if model_path.exists():
            model_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth")) + list(model_path.glob("*.bin"))
            for model_file in model_files:
                if model_file.stat().st_size > 0:
                    file_hash = calculate_file_hash(model_file)
                    if file_hash:
                        model_hashes[str(model_file)] = file_hash
                        log_info(f"  {model_file.name}: {file_hash[:16]}...")

    if model_hashes:
        log_success(f"Calculated hashes for {len(model_hashes)} model files")
        artifacts['models'] = model_hashes
        checks_passed.append("model_hashes")
    else:
        log_info("No model files found to hash")

    # 4.2 Get Docker image digests
    log_info("")
    log_info("Getting container image digests...")
    image_digests = {}

    for service in CONTAINER_SERVICES:
        image_name = f"fragrance-ai-{service}:{version}"
        returncode, stdout, stderr = run_command(['docker', 'images', '--digests', '--format', '{{.Repository}}:{{.Tag}} {{.Digest}}'])

        if returncode == 0:
            for line in stdout.split('\n'):
                if image_name in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        digest = parts[1]
                        image_digests[image_name] = digest
                        log_info(f"  {image_name}: {digest[:30]}...")

    if image_digests:
        log_success(f"Found digests for {len(image_digests)} container images")
        artifacts['containers'] = image_digests
        checks_passed.append("container_digests")
    else:
        msg = "No container image digests found. Build images first."
        log_warning(msg)
        checks_failed.append(msg)

    # 4.3 Record to manifest file
    manifest_dir = Path("releases")
    manifest_dir.mkdir(exist_ok=True)
    manifest_file = manifest_dir / f"manifest_{version}.json"

    manifest = {
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "git_commit": run_command(['git', 'rev-parse', 'HEAD'])[1].strip(),
        "artifacts": artifacts
    }

    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    log_success(f"Artifact manifest saved: {manifest_file}")
    checks_passed.append("manifest_created")

    # 4.4 Verify manifest integrity
    manifest_hash = calculate_file_hash(manifest_file)
    log_info(f"Manifest hash (SHA256): {manifest_hash}")

    # Save manifest hash
    hash_file = manifest_dir / f"manifest_{version}.sha256"
    with open(hash_file, 'w') as f:
        f.write(f"{manifest_hash}  {manifest_file.name}\n")

    log_success(f"Manifest hash saved: {hash_file}")
    checks_passed.append("manifest_hash")

    # Summary
    log_info("")
    if checks_failed:
        log_error(f"Artifact recording: {len(checks_failed)} issue(s) found")
        return False, "\n".join(checks_failed)
    else:
        log_success(f"Artifact recording: All checks passed ({len(checks_passed)} checks)")
        return True, "OK"


# =============================================================================
# Check 5: Runbook Verification
# =============================================================================

def check_runbook() -> Tuple[bool, str]:
    """
    Check 5: 런북 검증
    - Runbook exists
    - Health check procedures documented
    - Rollback procedures documented
    - Recent updates
    """
    log_info("")
    log_info("=" * 80)
    log_info("CHECK 5: Runbook Verification")
    log_info("=" * 80)

    checks_passed = []
    checks_failed = []

    # 5.1 Check runbook files exist
    runbook_files = [
        Path("DEPLOYMENT_GUIDE.md"),
        Path("RUNBOOK.md"),
        Path("docs/OPERATIONS.md")
    ]

    runbook_found = None
    for runbook_path in runbook_files:
        if runbook_path.exists():
            runbook_found = runbook_path
            log_success(f"Runbook found: {runbook_path}")
            checks_passed.append(f"runbook_{runbook_path.name}")
            break

    if not runbook_found:
        msg = "No runbook file found"
        log_error(msg)
        log_info("  Expected files: DEPLOYMENT_GUIDE.md, RUNBOOK.md, or docs/OPERATIONS.md")
        checks_failed.append(msg)
        return False, msg

    # 5.2 Check runbook content
    with open(runbook_found, 'r', encoding='utf-8') as f:
        content = f.read().lower()

    required_sections = {
        'health': ['health check', 'healthcheck', '/health'],
        'rollback': ['rollback', 'revert', 'downgrade'],
        'scaling': ['scale', 'scaling', 'replicas'],
        'monitoring': ['monitor', 'metrics', 'grafana', 'prometheus']
    }

    for section, keywords in required_sections.items():
        if any(keyword in content for keyword in keywords):
            log_success(f"✓ {section.capitalize()} procedures documented")
            checks_passed.append(f"runbook_section_{section}")
        else:
            msg = f"{section.capitalize()} procedures not documented"
            log_warning(msg)
            checks_failed.append(msg)

    # 5.3 Check runbook freshness
    mtime = runbook_found.stat().st_mtime
    days_old = (datetime.now().timestamp() - mtime) / 86400

    if days_old > 30:
        msg = f"Runbook is {int(days_old)} days old (last updated: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')})"
        log_warning(msg)
        log_info("  Consider updating runbook with latest procedures")
        checks_failed.append(msg)
    else:
        log_success(f"Runbook is up to date (last updated: {datetime.fromtimestamp(mtime).strftime('%Y-%m-%d')})")
        checks_passed.append("runbook_fresh")

    # 5.4 Check for broken links (markdown links)
    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = re.findall(link_pattern, content)

    broken_links = []
    for link_text, link_url in links:
        if link_url.startswith('http'):
            continue  # Skip external links

        link_path = runbook_found.parent / link_url
        if not link_path.exists():
            broken_links.append(f"{link_text} -> {link_url}")

    if broken_links:
        msg = f"Broken links found: {len(broken_links)}"
        log_warning(msg)
        for link in broken_links[:3]:
            log_info(f"  - {link}")
        checks_failed.append(msg)
    else:
        log_success("No broken links found")
        checks_passed.append("no_broken_links")

    # Summary
    log_info("")
    if checks_failed:
        log_error(f"Runbook verification: {len(checks_failed)} issue(s) found")
        return False, "\n".join(checks_failed)
    else:
        log_success(f"Runbook verification: All checks passed ({len(checks_passed)} checks)")
        return True, "OK"


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Pre-deployment checklist for Fragrance AI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--version',
        required=True,
        help='Release version (e.g., v0.2.0)'
    )
    parser.add_argument(
        '--env-file',
        default='.env',
        help='Environment file to validate (default: .env)'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on warnings'
    )
    parser.add_argument(
        '--output',
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Print header
    print("")
    print("=" * 80)
    print("PRE-DEPLOYMENT CHECKLIST (T-1)")
    print("=" * 80)
    print(f"Version: {args.version}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("")

    # Run all checks
    results = {}
    all_passed = True

    # Check 1: Release Tag
    passed, message = check_release_tag(args.version)
    results['release_tag'] = {'passed': passed, 'message': message}
    if not passed:
        all_passed = False

    # Check 2: Migrations
    passed, message = check_migrations()
    results['migrations'] = {'passed': passed, 'message': message}
    if not passed:
        all_passed = False

    # Check 3: Secrets
    passed, message = check_secrets(args.env_file)
    results['secrets'] = {'passed': passed, 'message': message}
    if not passed:
        all_passed = False

    # Check 4: Artifacts
    passed, message = check_artifacts(args.version)
    results['artifacts'] = {'passed': passed, 'message': message}
    if not passed and args.strict:
        all_passed = False

    # Check 5: Runbook
    passed, message = check_runbook()
    results['runbook'] = {'passed': passed, 'message': message}
    if not passed and args.strict:
        all_passed = False

    # Final summary
    print("")
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for r in results.values() if r['passed'])
    total_count = len(results)

    for check_name, result in results.items():
        status = f"{GREEN}✓ PASS{NC}" if result['passed'] else f"{RED}✗ FAIL{NC}"
        print(f"{status} {check_name}")

    print("")
    print(f"Result: {passed_count}/{total_count} checks passed")

    # Save results to file
    if args.output:
        output_data = {
            'version': args.version,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'passed': passed_count,
                'total': total_count,
                'all_passed': all_passed
            }
        }

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        log_info(f"Results saved to: {args.output}")

    print("=" * 80)

    # Exit with appropriate code
    if all_passed:
        print(f"{GREEN}✓ PRE-DEPLOYMENT CHECK: PASSED{NC}")
        print(f"{GREEN}✓ Ready for production deployment!{NC}")
        print("")
        sys.exit(0)
    else:
        print(f"{RED}✗ PRE-DEPLOYMENT CHECK: FAILED{NC}")
        print(f"{RED}✗ Fix issues before deploying to production{NC}")
        print("")
        sys.exit(1)


if __name__ == '__main__':
    main()
