# tests/test_security_privacy.py
"""
Security and Privacy Tests
PII masking, model integrity, and license compliance testing
"""

import pytest
import tempfile
import hashlib
from pathlib import Path
from fragrance_ai.security import (
    # PII Masking
    LogLevel,
    configure_privacy,
    get_privacy_settings,
    detect_pii_patterns,
    mask_pii_patterns,
    hash_text,
    sanitize_for_logging,
    # Model Integrity
    calculate_file_sha256,
    verify_model_integrity,
    get_checksum_database,
    # License Checking
    LicenseChecker,
    KNOWN_LICENSES
)


class TestPIIMasking:
    """Test PII detection and masking"""

    def test_detect_email_patterns(self):
        """Test email detection"""
        text = "Contact me at user@example.com or admin@test.org"
        pii = detect_pii_patterns(text)

        assert "email" in pii
        assert pii["email"] == 2

    def test_detect_korean_phone(self):
        """Test Korean phone number detection"""
        text = "전화: 010-1234-5678 또는 02-987-6543"
        pii = detect_pii_patterns(text)

        assert "phone_kr" in pii
        assert pii["phone_kr"] == 2

    def test_detect_credit_card(self):
        """Test credit card detection"""
        text = "Card: 1234-5678-9012-3456"
        pii = detect_pii_patterns(text)

        assert "credit_card" in pii

    def test_mask_email(self):
        """Test email masking"""
        text = "Email: user@example.com"
        masked = mask_pii_patterns(text)

        assert "user@example.com" not in masked
        assert "us*****@example.com" in masked

    def test_mask_phone(self):
        """Test phone number masking"""
        text = "전화: 010-1234-5678"
        masked = mask_pii_patterns(text)

        assert "1234" not in masked
        assert "***" in masked

    def test_hash_text_sha256(self):
        """Test SHA256 hashing"""
        text = "sensitive data"
        hash_result = hash_text(text, algorithm="sha256", output_length=16)

        assert len(hash_result) == 16
        assert hash_result == hash_text(text, algorithm="sha256", output_length=16)  # Deterministic

    def test_hash_text_with_salt(self):
        """Test hashing with salt"""
        text = "data"
        hash1 = hash_text(text, salt="salt1")
        hash2 = hash_text(text, salt="salt2")

        assert hash1 != hash2  # Different salts produce different hashes


class TestPrivacySettings:
    """Test privacy configuration"""

    def test_default_privacy_settings(self):
        """Test default settings"""
        configure_privacy(log_level=LogLevel.HASH_ONLY)
        settings = get_privacy_settings()

        assert settings.log_level == LogLevel.HASH_ONLY
        assert settings.sampling_rate == 0.01

    def test_configure_sampling_rate(self):
        """Test sampling rate configuration"""
        configure_privacy(log_level=LogLevel.SAMPLED, sampling_rate=0.05)
        settings = get_privacy_settings()

        assert settings.sampling_rate == 0.05

    def test_sanitize_for_logging_hash_only(self):
        """Test HASH_ONLY logging level"""
        configure_privacy(log_level=LogLevel.HASH_ONLY)

        result = sanitize_for_logging("사용자 입력 텍스트", user_opted_in=False)

        assert not result["user_text_logged"]
        assert "user_text_hash" in result
        assert result["text_length"] == len("사용자 입력 텍스트")

    def test_sanitize_for_logging_opt_in(self):
        """Test OPT_IN logging level"""
        configure_privacy(log_level=LogLevel.OPT_IN, mask_patterns=True)

        # Without opt-in
        result1 = sanitize_for_logging(
            "test@example.com",
            user_opted_in=False
        )
        assert not result1["user_text_logged"]

        # With opt-in
        result2 = sanitize_for_logging(
            "test@example.com",
            user_opted_in=True
        )
        assert result2["user_text_logged"]
        assert "test@example.com" not in result2.get("user_text", "")  # Masked

    def test_pii_detection_in_sanitize(self):
        """Test that PII is detected during sanitization"""
        result = sanitize_for_logging(
            "Contact: user@example.com, 010-1234-5678",
            user_opted_in=False
        )

        assert result["contains_pii"]
        assert "pii_detected" in result
        assert "email" in result["pii_detected"]
        assert "phone_kr" in result["pii_detected"]


class TestModelIntegrity:
    """Test model file integrity verification"""

    def test_calculate_sha256(self):
        """Test SHA256 calculation for file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            hash_result = calculate_file_sha256(temp_path)
            assert len(hash_result) == 64  # SHA256 produces 64-char hex
            assert all(c in '0123456789abcdef' for c in hash_result)
        finally:
            Path(temp_path).unlink()

    def test_verify_model_integrity_success(self):
        """Test successful integrity verification"""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("model weights")
            temp_path = f.name

        try:
            # Calculate expected checksum
            expected = calculate_file_sha256(temp_path)

            # Verify
            result = verify_model_integrity(
                model_name="test_model",
                file_path=temp_path,
                expected_checksum=expected
            )

            assert result.verified
            assert result.actual_sha256 == expected
        finally:
            Path(temp_path).unlink()

    def test_verify_model_integrity_mismatch(self):
        """Test checksum mismatch detection"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("model weights")
            temp_path = f.name

        try:
            # Use wrong checksum
            wrong_checksum = "0" * 64

            result = verify_model_integrity(
                model_name="test_model",
                file_path=temp_path,
                expected_checksum=wrong_checksum
            )

            assert not result.verified
            assert result.error_message == "Checksum mismatch"
        finally:
            Path(temp_path).unlink()

    def test_verify_missing_file(self):
        """Test verification of missing file"""
        result = verify_model_integrity(
            model_name="test_model",
            file_path="/nonexistent/path/model.bin"
        )

        assert not result.verified
        assert "File not found" in result.error_message


class TestLicenseChecking:
    """Test license compliance checking"""

    def test_known_licenses_loaded(self):
        """Test that known licenses are available"""
        assert len(KNOWN_LICENSES) > 0
        assert "Qwen/Qwen2.5-7B-Instruct" in KNOWN_LICENSES
        assert "mistralai/Mistral-7B-Instruct-v0.3" in KNOWN_LICENSES
        assert "meta-llama/Meta-Llama-3-8B-Instruct" in KNOWN_LICENSES

    def test_qwen_license_details(self):
        """Test Qwen license details"""
        qwen_license = KNOWN_LICENSES["Qwen/Qwen2.5-7B-Instruct"]

        assert qwen_license.commercial_use
        assert qwen_license.attribution_required
        assert qwen_license.modification_allowed
        assert len(qwen_license.restrictions) > 0

    def test_license_checker_compliant_model(self):
        """Test license check for compliant model"""
        checker = LicenseChecker()
        result = checker.check_model_license(
            "Qwen/Qwen2.5-7B-Instruct",
            commercial_use=True
        )

        assert result.compliant
        assert result.license_info is not None
        assert len(result.warnings) > 0  # Has restrictions
        assert len(result.errors) == 0

    def test_license_checker_llama_user_limit(self):
        """Test Llama license with user limit warning"""
        checker = LicenseChecker()
        result = checker.check_model_license(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            commercial_use=True,
            monthly_users=800_000_000  # Above 700M limit
        )

        assert result.compliant  # Still compliant, just a warning
        assert any("700M" in w for w in result.warnings)

    def test_license_checker_multiple_models(self):
        """Test checking multiple models"""
        checker = LicenseChecker()
        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct"
        ]

        results = checker.check_all_models(models, commercial_use=True)

        assert len(results) == 3
        assert all(r.compliant for r in results.values())

    def test_license_checker_unknown_model(self):
        """Test unknown model handling"""
        checker = LicenseChecker()
        result = checker.check_model_license(
            "unknown/model",
            commercial_use=True
        )

        assert not result.compliant
        assert result.license_info is None
        assert len(result.errors) > 0
        assert "not found" in result.errors[0]

    def test_license_report_generation(self):
        """Test license report generation"""
        checker = LicenseChecker()
        results = checker.check_all_models(
            ["Qwen/Qwen2.5-7B-Instruct"],
            commercial_use=True
        )

        report = checker.generate_license_report(results)

        assert "# Model License Compliance Report" in report
        assert "Qwen/Qwen2.5-7B-Instruct" in report
        assert "Qwen-Tongyi" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
