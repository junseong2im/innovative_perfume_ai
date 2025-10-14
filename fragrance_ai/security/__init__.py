# fragrance_ai/security/__init__.py
"""
Security and Privacy Module
보안 및 프라이버시 관리
"""

from .pii_masking import (
    LogLevel,
    PrivacySettings,
    configure_privacy,
    get_privacy_settings,
    detect_pii_patterns,
    mask_pii_patterns,
    hash_text,
    hash_with_prefix,
    sanitize_for_logging,
    PrivacyAuditLog,
    get_audit_log
)

from .model_integrity import (
    ModelChecksum,
    ChecksumDatabase,
    get_checksum_database,
    calculate_file_sha256,
    verify_model_integrity,
    verify_model_directory,
    register_model_checksum,
    list_registered_models
)

from .license_checker import (
    LicenseType,
    LicenseInfo,
    KNOWN_LICENSES,
    LicenseCheckResult,
    LicenseChecker,
    check_licenses_cli
)

__all__ = [
    # PII Masking
    'LogLevel',
    'PrivacySettings',
    'configure_privacy',
    'get_privacy_settings',
    'detect_pii_patterns',
    'mask_pii_patterns',
    'hash_text',
    'hash_with_prefix',
    'sanitize_for_logging',
    'PrivacyAuditLog',
    'get_audit_log',

    # Model Integrity
    'ModelChecksum',
    'ChecksumDatabase',
    'get_checksum_database',
    'calculate_file_sha256',
    'verify_model_integrity',
    'verify_model_directory',
    'register_model_checksum',
    'list_registered_models',

    # License Checking
    'LicenseType',
    'LicenseInfo',
    'KNOWN_LICENSES',
    'LicenseCheckResult',
    'LicenseChecker',
    'check_licenses_cli'
]
