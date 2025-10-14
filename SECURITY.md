# Security & Privacy Documentation

보안 및 프라이버시 구현 문서

---

## Overview

This document describes the security and privacy features implemented in the Fragrance AI system, including:

1. **PII Masking & Privacy Protection** - 입력/응답 PII 마스킹
2. **Model File Integrity Verification** - 모델 파일/가중치 무결성 검증
3. **License Compliance Checking** - 라이선스 준수 확인

---

## 1. PII Masking & Privacy Protection

### Features

- **Automatic PII Detection**: Detects emails, phone numbers, credit cards, SSN, IP addresses, URLs
- **Configurable Privacy Levels**: NONE, HASH_ONLY, SAMPLED, OPT_IN, FULL
- **Smart Masking**: Masks sensitive patterns while preserving context
- **Sampling**: Log only N% of requests (default: 1%)
- **Opt-in Mechanism**: Users can explicitly consent to logging
- **Audit Logging**: Track all privacy-related access events

### Usage

```python
from fragrance_ai.security import configure_privacy, LogLevel, sanitize_for_logging

# Configure privacy settings
configure_privacy(
    log_level=LogLevel.HASH_ONLY,  # Only log hashes
    sampling_rate=0.01,  # Sample 1% of requests
    mask_patterns=True   # Auto-mask PII patterns
)

# Sanitize user input for logging
user_input = "Contact me at user@example.com or 010-1234-5678"
sanitized = sanitize_for_logging(
    user_text=user_input,
    user_opted_in=False  # User has not opted in
)

print(sanitized)
# Output: {
#     "user_text_logged": False,
#     "user_text_hash": "a3f8b2c1...",
#     "text_length": 48,
#     "contains_pii": True,
#     "pii_detected": {"email": 1, "phone_kr": 1}
# }
```

### Privacy Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `NONE` | No user data logged | Maximum privacy |
| `HASH_ONLY` | Only hash of input logged | Production (recommended) |
| `SAMPLED` | Sample N% of requests | Monitoring with privacy |
| `OPT_IN` | Log only if user opts in | User-controlled logging |
| `FULL` | Full logging | Development only |

### PII Patterns Detected

- **Email**: `user@example.com`
- **Korean Phone**: `010-1234-5678`, `02-987-6543`
- **International Phone**: `+82-10-1234-5678`
- **Korean SSN**: `123456-1234567`
- **Credit Card**: `1234-5678-9012-3456`
- **IP Address**: `192.168.1.1`
- **URLs**: `https://example.com/path`

---

## 2. Model File Integrity Verification

### Features

- **SHA256 Checksums**: Verify model file integrity using SHA256 hashes
- **Checksum Database**: Persistent storage of trusted model checksums
- **Auto-registration**: Automatically register new models
- **Batch Verification**: Verify all files in a directory
- **Load-time Validation**: Verify models before loading

### Usage

#### Register Model Checksum

```python
from fragrance_ai.security import register_model_checksum

# Register a trusted model checksum
register_model_checksum(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    file_path="/path/to/model/pytorch_model.bin",
    version="2.5",
    source="HuggingFace"
)
```

#### Verify Model Integrity

```python
from fragrance_ai.security import verify_model_integrity

# Verify model before loading
result = verify_model_integrity(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    file_path="/path/to/model/pytorch_model.bin",
    auto_register=True  # Register if not in database
)

if result.verified:
    print(f"✓ Model verified: {result.actual_sha256[:16]}...")
    # Safe to load model
else:
    print(f"✗ Verification failed: {result.error_message}")
    # Do NOT load model
```

#### Verify Directory

```python
from fragrance_ai.security import verify_model_directory

# Verify all model files in directory
results = verify_model_directory(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    directory="/path/to/model/",
    extensions=['.bin', '.safetensors'],
    auto_register=True
)

# Check results
for file_path, result in results.items():
    print(f"{result.verified} - {file_path}")
```

### Checksum Database

Checksums are stored in `model_checksums.json`:

```json
{
  "Qwen/Qwen2.5-7B-Instruct:/path/to/model.bin": {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "file_path": "/path/to/model.bin",
    "sha256": "a3f8b2c1d4e5f6...",
    "file_size_bytes": 14212345678,
    "last_verified": "2025-01-15T10:30:00Z",
    "version": "2.5",
    "source": "HuggingFace"
  }
}
```

---

## 3. License Compliance Checking

### Features

- **Known License Database**: Pre-configured licenses for Qwen, Mistral, Llama
- **Compliance Verification**: Check commercial use, user limits, restrictions
- **Automated Checks**: Run during build/deployment
- **Markdown Reports**: Generate detailed compliance reports
- **CLI Tool**: Command-line license checker

### Supported Models

| Model | License | Commercial Use | Notes |
|-------|---------|----------------|-------|
| **Qwen/Qwen2.5-7B-Instruct** | Tongyi Qianwen | ✓ Yes | Attribution required |
| **mistralai/Mistral-7B-Instruct-v0.3** | Apache 2.0 | ✓ Yes | Include license notice |
| **meta-llama/Meta-Llama-3-8B-Instruct** | Llama 3 Community | ✓ Yes* | *<700M MAU |

### Usage

#### Check Licenses Programmatically

```python
from fragrance_ai.security import LicenseChecker

# Create checker
checker = LicenseChecker()

# Check single model
result = checker.check_model_license(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    commercial_use=True,
    monthly_users=1_000_000
)

if result.compliant:
    print("✓ License compliant")
else:
    print(f"✗ License issues: {result.errors}")

# Check all models
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

results = checker.check_all_models(
    model_names=models,
    commercial_use=True
)

# Generate report
report = checker.generate_license_report(
    results,
    output_path="LICENSE_REPORT.md"
)
```

#### CLI Tool

```bash
# Check all default models
python scripts/check_licenses.py

# Check specific models
python scripts/check_licenses.py --models Qwen/Qwen2.5-7B-Instruct

# Generate report
python scripts/check_licenses.py --output LICENSE_REPORT.md

# Check with MAU limit
python scripts/check_licenses.py --monthly-users 1000000000

# List known licenses
python scripts/check_licenses.py --list-known

# Fail on warnings
python scripts/check_licenses.py --fail-on-warning
```

#### Build Integration

Add to your build script or CI/CD pipeline:

```bash
# Check licenses before build
python scripts/check_licenses.py || exit 1

# Generate license report
python scripts/check_licenses.py --output docs/LICENSE_REPORT.md
```

### License Details

#### Qwen (Tongyi Qianwen License)
- ✓ Commercial use allowed
- ✓ Modification allowed
- ✓ Distribution allowed
- ⚠️ Attribution required
- ⚠️ Cannot harm Alibaba Cloud's reputation

#### Mistral (Apache 2.0)
- ✓ Commercial use allowed
- ✓ Modification allowed
- ✓ Distribution allowed
- ⚠️ Must include license notice
- ⚠️ State changes if modified

#### Llama 3 (Meta Community License)
- ✓ Commercial use allowed*
- ✓ Modification allowed
- ✓ Distribution allowed
- ⚠️ *Monthly active users > 700M require Meta license
- ⚠️ Cannot use to train other LLMs
- ⚠️ Must comply with Acceptable Use Policy

---

## Best Practices

### 1. Production Privacy Settings

```python
# Recommended for production
configure_privacy(
    log_level=LogLevel.HASH_ONLY,  # Only hashes
    sampling_rate=0.001,  # 0.1% sampling
    mask_patterns=True,   # Auto-mask PII
    allow_opt_in=True     # Allow user consent
)
```

### 2. Model Loading with Integrity Check

```python
from fragrance_ai.security import verify_model_integrity
from transformers import AutoModel

def load_model_safely(model_name, model_path):
    # Verify integrity first
    result = verify_model_integrity(
        model_name=model_name,
        file_path=model_path
    )

    if not result.verified:
        raise RuntimeError(f"Model integrity check failed: {result.error_message}")

    # Safe to load
    return AutoModel.from_pretrained(model_path)
```

### 3. License Check in CI/CD

```yaml
# .github/workflows/build.yml
steps:
  - name: Check Model Licenses
    run: python scripts/check_licenses.py --fail-on-warning

  - name: Generate License Report
    run: python scripts/check_licenses.py --output docs/LICENSE_REPORT.md
```

---

## Testing

Run security tests:

```bash
# Run all security tests
pytest tests/test_security_privacy.py -v

# Run specific test class
pytest tests/test_security_privacy.py::TestPIIMasking -v

# Run with coverage
pytest tests/test_security_privacy.py --cov=fragrance_ai.security
```

Test Results: **23/23 tests passed** ✓

---

## API Reference

### PII Masking

```python
from fragrance_ai.security import (
    configure_privacy,
    get_privacy_settings,
    detect_pii_patterns,
    mask_pii_patterns,
    hash_text,
    sanitize_for_logging
)
```

### Model Integrity

```python
from fragrance_ai.security import (
    calculate_file_sha256,
    verify_model_integrity,
    verify_model_directory,
    register_model_checksum,
    get_checksum_database
)
```

### License Checking

```python
from fragrance_ai.security import (
    LicenseChecker,
    KNOWN_LICENSES,
    check_licenses_cli
)
```

---

## Compliance

This implementation complies with:

- **GDPR**: PII masking, user consent (opt-in), data minimization
- **CCPA**: Privacy controls, user rights
- **Model Licenses**: Qwen (Tongyi), Mistral (Apache 2.0), Llama 3 (Community)

---

---

## 4. Secrets Management

### Features

- **Multi-Provider Support**: AWS Secrets Manager, AWS KMS, GCP Secret Manager, Azure Key Vault, HashiCorp Vault
- **No Secrets in Code**: All secrets managed externally
- **Validation**: Automated verification of required secrets
- **Fallback**: Graceful degradation if secrets are unavailable

### Required Secrets

```bash
# Database
DB_PASSWORD_PROD
DB_PASSWORD_STG

# LLM API Keys
QWEN_API_KEY
MISTRAL_API_KEY
LLAMA_API_KEY

# AWS (if used)
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

# Monitoring
GRAFANA_API_KEY

# Encryption
JWT_SECRET_KEY
ENCRYPTION_KEY
```

### Configuration

**Environment Variables (.env):**

```bash
SECRETS_PROVIDER=env
QWEN_API_KEY=sk-...
```

**AWS Secrets Manager:**

```bash
SECRETS_PROVIDER=aws_secrets_manager
AWS_REGION=us-east-1
```

**AWS KMS (Encrypted):**

```bash
SECRETS_PROVIDER=aws_kms
AWS_REGION=us-east-1
KMS_KEY_ID=alias/artisan-prod
QWEN_API_KEY_ENCRYPTED=AQICAHh...  # base64-encoded ciphertext
```

### Usage

```python
from fragrance_ai.security.secrets_manager import get_secret

# Get secret (provider-agnostic)
api_key = get_secret("QWEN_API_KEY")
```

### Validation

```bash
# Validate all required secrets
python -m fragrance_ai.security.secrets_manager --validate
```

---

## 5. IFRA Compliance

### Features

- **Automatic IFRA Validation**: All formulas checked against IFRA limits
- **49 IFRA Categories**: Complete coverage of product categories
- **Automatic Clipping**: Violations automatically corrected
- **Monitoring**: Real-time IFRA violation tracking

### Usage

```python
from fragrance_ai.regulations.ifra_rules import IFRAValidator, ensure_ifra_compliance

# Validate formula
validator = IFRAValidator()
result = validator.validate_complete(
    recipe=olfactory_dna,
    product_category=ProductCategory.EAU_DE_PARFUM
)

# Auto-correct violations
compliant_recipe = ensure_ifra_compliance(
    recipe=olfactory_dna,
    product_category=ProductCategory.EAU_DE_PARFUM
)
```

### Testing

```bash
pytest tests/test_ifra.py -v
```

---

## 6. Security Monitoring

### Prometheus Metrics

Security metrics exposed at `/metrics`:

```promql
# Model integrity check failures
security_model_integrity_check_failed_total

# License compliance issues
security_license_compliance_failed_total

# Missing required secrets
security_missing_secrets_total

# IFRA violations
security_ifra_violations_total

# PII detected in logs
security_pii_detected_total

# Unauthorized access attempts
security_unauthorized_access_total

# Security scan failures
security_scan_failed_total
```

### Alerts

Security alerts configured in `docker/prometheus/alerts.yml`:

| Alert | Severity | Threshold |
|-------|----------|-----------|
| ModelIntegrityCheckFailed | Critical | > 0 failures |
| LicenseComplianceIssue | Critical | > 0 issues |
| MissingRequiredSecrets | Critical | > 0 missing |
| IFRAViolationDetected | Warning | > 0 violations/5m |
| PIIDetectedInLogs | Warning | > 5 detections/5m |
| UnauthorizedAccessAttempt | Warning | > 10 attempts/5m |
| SecurityScanFailed | Critical | > 0 failures |

---

## 7. Security Compliance Verification

### Comprehensive Security Check

Run before every deployment:

```bash
# Full security compliance check
python scripts/verify_security_compliance.py --all --report compliance_report.md
```

**Checks performed:**
- ✅ License compliance (Qwen, Mistral, Llama)
- ✅ Model integrity (SHA256 verification)
- ✅ Secrets validation (all required secrets present)
- ✅ SBOM generation (CycloneDX format)

### Security Smoke Test

```bash
# Security-enhanced smoke test
python smoke_test_security.py
```

**Checks performed:**
- ✅ API health
- ✅ LLM health (qwen, mistral, llama)
- ✅ IFRA validation in API endpoints
- ✅ PII masking in logs
- ✅ Model integrity verification
- ✅ License compliance
- ✅ Secrets validation

### SBOM Generation

```bash
# Generate Software Bill of Materials
python scripts/verify_security_compliance.py --sbom sbom.json
```

**SBOM Format:** CycloneDX 1.4

**Contents:**
- Model names and versions
- SHA256 hashes
- License information
- File sizes
- Last verification timestamps

---

## 8. Deployment Checklist

### Pre-Deployment Security Checklist

See `PRODUCTION_CHECKLIST.md` for complete checklist. Key security items:

**Secrets Management:**
- [ ] Configure secrets manager (AWS/GCP/Azure/Vault)
- [ ] Validate all required secrets
- [ ] Set SECRETS_PROVIDER environment variable
- [ ] Verify no secrets in git repository

**Model Security:**
- [ ] Register all models with SHA256 checksums
- [ ] Verify model integrity
- [ ] Check license compliance
- [ ] Generate SBOM

**Privacy & Compliance:**
- [ ] Configure PII masking (HASH_ONLY for production)
- [ ] Enable IFRA validation
- [ ] Test IFRA compliance
- [ ] Configure audit logging

**Security Scanning:**
- [ ] Run comprehensive security check
- [ ] Run security smoke test
- [ ] Check for vulnerable dependencies
- [ ] Verify no exposed secrets

---

## 9. Incident Response

### Security Incident Severity

| Severity | Response Time | Examples |
|----------|---------------|----------|
| **Sev1** | 15 minutes | Data breach, secrets exposed |
| **Sev2** | 30 minutes | Model tampering, license violation |
| **Sev3** | 60 minutes | IFRA violations, PII leak |
| **Sev4** | 4 hours | Low-risk security issues |

### Runbooks

Automated incident response procedures available in `docs/RUNBOOKS.md`.

### Blameless Postmortems

See `docs/POSTMORTEM_TEMPLATE.md` for structured incident analysis.

---

## 10. Vulnerability Reporting

### Reporting Process

If you discover a security vulnerability:

1. **Do NOT** open a public GitHub issue
2. Email: security@artisan.example.com
3. Include:
   - Description of vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (optional)

### Response Timeline

- **Initial response**: Within 48 hours
- **Triage & assessment**: Within 1 week
- **Fix & disclosure**: Coordinated with reporter

---

## CI/CD Integration

### GitHub Actions Security Workflow

```yaml
# .github/workflows/security.yml
name: Security Checks

on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: License Compliance
        run: python scripts/verify_security_compliance.py --licenses

      - name: Secrets Validation
        run: python -m fragrance_ai.security.secrets_manager --validate

      - name: Model Integrity
        run: python -m fragrance_ai.security.model_integrity --verify-all

      - name: IFRA Tests
        run: pytest tests/test_ifra.py -v

      - name: Security Smoke Test
        run: python smoke_test_security.py

      - name: SBOM Generation
        run: python scripts/verify_security_compliance.py --sbom sbom.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: sbom.json
```

---

## Contact

For security concerns or questions:
- Create an issue on GitHub
- Email: security@artisan.example.com

For license questions:
- Qwen: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Mistral: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- Llama 3: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

---

## References

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [IFRA Standards](https://ifrafragrance.org/)
- [CycloneDX SBOM](https://cyclonedx.org/)
- [AWS Secrets Manager](https://aws.amazon.com/secrets-manager/)
- [HashiCorp Vault](https://www.vaultproject.io/)

---

**Last Updated:** 2025-10-14
**Version:** 2.0.0 - Enhanced with comprehensive security infrastructure
