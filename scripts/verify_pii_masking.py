"""
PII Masking Verification Script
샘플 로그를 생성하고 PII 마스킹 검증
"""

import re
import json
import logging
import hashlib
from typing import List, Dict, Any
from datetime import datetime

# =============================================================================
# Inline LogMasker Implementation (from fragrance_ai.observability)
# =============================================================================

class LogMasker:
    """Mask sensitive information in logs"""

    API_KEY_PATTERNS = [
        r'(?i)(api[_-]?key|apikey|api[_-]?token)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
        r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{20,})',
        r'(?i)(token\s*[:=]\s*)["\']?([a-zA-Z0-9_\-\.]{20,})["\']?',
    ]

    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    PHONE_PATTERN = r'(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'
    CREDIT_CARD_PATTERN = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
    AWS_ACCESS_KEY = r'(?i)(aws[_-]?access[_-]?key[_-]?id)\s*[:=]\s*["\']?([A-Z0-9]{20})["\']?'
    AWS_SECRET_KEY = r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[:=]\s*["\']?([A-Za-z0-9/+=]{40})["\']?'
    DB_URL_PATTERN = r'(?i)(postgres|mysql|mongodb):\/\/([^:]+):([^@]+)@([^\/]+)'

    @classmethod
    def mask_api_keys(cls, text: str) -> str:
        for pattern in cls.API_KEY_PATTERNS:
            text = re.sub(pattern, r'\1=***MASKED***', text)
        text = re.sub(cls.AWS_ACCESS_KEY, r'\1=***MASKED***', text)
        text = re.sub(cls.AWS_SECRET_KEY, r'\1=***MASKED***', text)
        return text

    @classmethod
    def mask_pii(cls, text: str) -> str:
        text = re.sub(cls.EMAIL_PATTERN, '***EMAIL_MASKED***', text)
        text = re.sub(cls.PHONE_PATTERN, '***PHONE_MASKED***', text)
        text = re.sub(cls.CREDIT_CARD_PATTERN, '***CARD_MASKED***', text)
        return text

    @classmethod
    def mask_db_credentials(cls, text: str) -> str:
        return re.sub(cls.DB_URL_PATTERN, r'\1://***USER***:***PASS***@\4', text)

    @classmethod
    def hash_user_id(cls, user_id: str) -> str:
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    @classmethod
    def mask_all(cls, text: str) -> str:
        text = cls.mask_api_keys(text)
        text = cls.mask_pii(text)
        text = cls.mask_db_credentials(text)
        return text

    @classmethod
    def sanitize_log_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in data.items():
            if key in ('user_id', 'userId', 'user'):
                sanitized[f"{key}_hash"] = cls.hash_user_id(str(value))
                continue
            if key in ('api_key', 'apiKey', 'token', 'password', 'secret', 'auth',
                      'aws_access_key_id', 'aws_secret_access_key'):
                sanitized[key] = '***MASKED***'
                continue
            if isinstance(value, dict):
                sanitized[key] = cls.sanitize_log_data(value)
            elif isinstance(value, str):
                sanitized[key] = cls.mask_all(value)
            else:
                sanitized[key] = value
        return sanitized


# =============================================================================
# Sample Data with PII
# =============================================================================

SAMPLE_DATA_WITH_PII = [
    # Email addresses
    {
        "user_id": "user_12345",
        "email": "john.doe@example.com",
        "message": "User john.smith@company.co.kr requested fragrance"
    },
    # Phone numbers
    {
        "user_id": "user_67890",
        "phone": "+82-10-1234-5678",
        "message": "Contact: 010-9876-5432 for delivery"
    },
    # API keys and tokens (EXAMPLE/DUMMY DATA ONLY)
    {
        "api_key": "sk_test_EXAMPLE_KEY_123456789012345678901234",
        "token": "Bearer EXAMPLE_JWT_TOKEN_eyJhbGciOiJIUzI1NiJ9",
        "message": "API key: test_dummy_key_abc123def456ghi789"
    },
    # Credit card numbers
    {
        "payment": "4532-1234-5678-9010",
        "message": "Card ending in 9010"
    },
    # AWS credentials
    {
        "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
        "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        "message": "AWS credentials configured"
    },
    # Database URLs
    {
        "db_url": "postgresql://admin:secretpassword@db.example.com:5432/mydb",
        "message": "Connected to database"
    },
    # Mixed PII
    {
        "user_id": "user_mixed",
        "email": "alice@test.com",
        "phone": "555-123-4567",
        "message": "User alice@test.com called from 555-123-4567 with API key sk_test_abc123"
    }
]


# =============================================================================
# PII Detection Patterns
# =============================================================================

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}',
    "phone_kr": r'\+?82-?10-?\d{4}-?\d{4}',
    "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    "api_key": r'(?i)(api[_-]?key|apikey|token)\s*[:=]\s*["\']?([a-zA-Z0-9_\-\.]{20,})["\']?',
    "bearer_token": r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{20,})',
    "aws_key": r'(?i)(AKIA[0-9A-Z]{16})',
    "aws_secret": r'(?i)([A-Za-z0-9/+=]{40})',
    "db_password": r'(?i)(postgres|mysql|mongodb):\/\/([^:]+):([^@]+)@'
}


# =============================================================================
# Verification Functions
# =============================================================================

def detect_pii_in_text(text: str) -> List[Dict[str, Any]]:
    """텍스트에서 PII 감지"""
    found_pii = []

    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            found_pii.append({
                "type": pii_type,
                "value": match.group(0),
                "start": match.start(),
                "end": match.end()
            })

    return found_pii


def verify_masking_works(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """마스킹이 제대로 작동하는지 검증"""
    # Convert to JSON string
    original_str = json.dumps(sample_data, ensure_ascii=False)

    # Apply masking
    masked_data = LogMasker.sanitize_log_data(sample_data.copy())
    masked_str = json.dumps(masked_data, ensure_ascii=False)

    # Also mask the string representation
    masked_str_direct = LogMasker.mask_all(original_str)

    # Detect PII in original
    original_pii = detect_pii_in_text(original_str)

    # Detect PII in masked (should be 0 or minimal)
    masked_pii_dict = detect_pii_in_text(masked_str)
    masked_pii_str = detect_pii_in_text(masked_str_direct)

    return {
        "original": original_str,
        "masked_dict": masked_str,
        "masked_str": masked_str_direct,
        "original_pii_count": len(original_pii),
        "masked_pii_count_dict": len(masked_pii_dict),
        "masked_pii_count_str": len(masked_pii_str),
        "original_pii": original_pii,
        "masked_pii_dict": masked_pii_dict,
        "masked_pii_str": masked_pii_str,
        "masking_effective": len(masked_pii_dict) == 0 and len(masked_pii_str) == 0
    }


def generate_sample_logs() -> List[str]:
    """샘플 로그 생성"""
    logs = []

    for i, sample in enumerate(SAMPLE_DATA_WITH_PII):
        # Create log entry with masked data
        masked_sample = LogMasker.sanitize_log_data(sample.copy())

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "INFO",
            "message": f"Sample log {i+1}",
            "data": masked_sample
        }

        logs.append(json.dumps(log_entry, ensure_ascii=False))

    return logs


# =============================================================================
# Verification Report
# =============================================================================

def generate_verification_report() -> Dict[str, Any]:
    """PII 마스킹 검증 보고서 생성"""
    print("=" * 80)
    print("PII Masking Verification Report")
    print("=" * 80)
    print()

    results = []
    total_pii_found_original = 0
    total_pii_found_masked = 0

    for i, sample in enumerate(SAMPLE_DATA_WITH_PII, 1):
        print(f"Test Case {i}:")
        print("-" * 80)

        result = verify_masking_works(sample)
        results.append(result)

        total_pii_found_original += result["original_pii_count"]
        total_pii_found_masked += result["masked_pii_count_dict"]

        # Print original (truncated)
        print(f"Original (first 150 chars):")
        print(f"  {result['original'][:150]}...")
        print()

        # Print masked
        print(f"Masked (first 150 chars):")
        print(f"  {result['masked_dict'][:150]}...")
        print()

        # Print PII found
        print(f"PII in original: {result['original_pii_count']}")
        if result['original_pii']:
            for pii in result['original_pii'][:3]:  # Show first 3
                print(f"  - {pii['type']}: {pii['value'][:30]}...")

        print(f"PII in masked: {result['masked_pii_count_dict']}")
        if result['masked_pii_dict']:
            print("  [WARNING] PII still detected after masking!")
            for pii in result['masked_pii_dict']:
                print(f"    - {pii['type']}: {pii['value']}")

        # Status
        if result["masking_effective"]:
            print("[PASS] Masking EFFECTIVE")
        else:
            print("[FAIL] Masking FAILED")

        print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total test cases: {len(SAMPLE_DATA_WITH_PII)}")
    print(f"PII instances in original data: {total_pii_found_original}")
    print(f"PII instances in masked data: {total_pii_found_masked}")
    print()

    effective_count = sum(1 for r in results if r["masking_effective"])
    print(f"Effective masking: {effective_count}/{len(results)}")
    print(f"Success rate: {effective_count/len(results)*100:.1f}%")
    print()

    if total_pii_found_masked == 0:
        print("[PASS] ALL PII SUCCESSFULLY MASKED")
    else:
        print(f"[WARNING] {total_pii_found_masked} PII instances still present after masking")

    print()

    return {
        "total_tests": len(SAMPLE_DATA_WITH_PII),
        "pii_original": total_pii_found_original,
        "pii_masked": total_pii_found_masked,
        "effective_count": effective_count,
        "success_rate": effective_count / len(results),
        "results": results
    }


# =============================================================================
# Specific Pattern Tests
# =============================================================================

def test_specific_patterns():
    """특정 패턴별 마스킹 테스트"""
    print("=" * 80)
    print("Specific Pattern Tests")
    print("=" * 80)
    print()

    test_cases = [
        ("Email", "Contact me at john.doe@example.com for details"),
        ("Phone", "Call me at +82-10-1234-5678 or 010-9876-5432"),
        ("API Key", "Use API key: sk_live_1234567890abcdefghijk"),
        ("Credit Card", "Payment with card 4532-1234-5678-9010"),
        ("AWS Key", "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"),
        ("DB URL", "postgres://admin:secret123@db.example.com:5432/mydb"),
    ]

    for name, text in test_cases:
        print(f"Test: {name}")
        print(f"  Original: {text}")

        masked = LogMasker.mask_all(text)
        print(f"  Masked:   {masked}")

        # Check if original sensitive data is still present
        pii_found = detect_pii_in_text(masked)
        if pii_found:
            print(f"  [WARNING] {len(pii_found)} PII still detected")
        else:
            print(f"  [OK]")

        print()


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PII MASKING VERIFICATION")
    print("=" * 80)
    print()

    # Test 1: Specific patterns
    test_specific_patterns()

    # Test 2: Full verification
    report = generate_verification_report()

    # Test 3: Generate sample logs
    print("=" * 80)
    print("Sample Logs Generation")
    print("=" * 80)
    print()

    logs = generate_sample_logs()
    print(f"Generated {len(logs)} sample logs")
    print()

    # Show first log
    if logs:
        print("First log (masked):")
        print(logs[0])
        print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)
    print()

    if report["success_rate"] == 1.0 and report["pii_masked"] == 0:
        print("[PASS] PII MASKING VERIFICATION PASSED")
        print()
        print("All sensitive data is properly masked:")
        print("  * Email addresses -> ***EMAIL_MASKED***")
        print("  * Phone numbers -> ***PHONE_MASKED***")
        print("  * Credit cards -> ***CARD_MASKED***")
        print("  * API keys/tokens -> ***MASKED***")
        print("  * User IDs -> hashed")
        exit(0)
    else:
        print("[FAIL] PII MASKING VERIFICATION FAILED")
        print()
        print(f"Issues found:")
        print(f"  * Success rate: {report['success_rate']*100:.1f}%")
        print(f"  * PII still present: {report['pii_masked']} instances")
        print()
        print("Review the logs above for details.")
        exit(1)
