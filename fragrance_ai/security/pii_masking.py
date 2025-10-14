# fragrance_ai/security/pii_masking.py
"""
PII Masking and Privacy Protection Module
로그에 사용자 텍스트 저장 금지 - 해시/샘플링/옵트인 구현
"""

import re
import hashlib
import random
from typing import Optional, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Privacy Settings
# ============================================================================

class LogLevel(str, Enum):
    """Privacy-aware logging levels"""
    NONE = "none"           # No user data logged at all
    HASH_ONLY = "hash"      # Only hash of input logged
    SAMPLED = "sampled"     # Sample N% of requests (default)
    OPT_IN = "opt_in"       # Log only if user explicitly opts in
    FULL = "full"           # Full logging (for development only)


@dataclass
class PrivacySettings:
    """Privacy configuration"""
    log_level: LogLevel = LogLevel.HASH_ONLY
    sampling_rate: float = 0.01  # 1% of requests
    hash_algorithm: str = "sha256"
    truncate_length: int = 50  # Truncate to N chars if logged
    mask_patterns: bool = True  # Mask PII patterns (email, phone, etc.)
    allow_opt_in: bool = True  # Allow users to opt-in for logging


# Global privacy settings
_privacy_settings = PrivacySettings()


def configure_privacy(
    log_level: LogLevel = LogLevel.HASH_ONLY,
    sampling_rate: float = 0.01,
    hash_algorithm: str = "sha256",
    truncate_length: int = 50,
    mask_patterns: bool = True,
    allow_opt_in: bool = True
):
    """
    Configure global privacy settings

    Args:
        log_level: Privacy level for logging
        sampling_rate: Percentage of requests to log (0.0-1.0)
        hash_algorithm: Hash algorithm (sha256, sha512, md5)
        truncate_length: Max characters to log
        mask_patterns: Auto-mask PII patterns
        allow_opt_in: Allow opt-in for detailed logging
    """
    global _privacy_settings
    _privacy_settings = PrivacySettings(
        log_level=log_level,
        sampling_rate=sampling_rate,
        hash_algorithm=hash_algorithm,
        truncate_length=truncate_length,
        mask_patterns=mask_patterns,
        allow_opt_in=allow_opt_in
    )


def get_privacy_settings() -> PrivacySettings:
    """Get current privacy settings"""
    return _privacy_settings


# ============================================================================
# PII Pattern Detection and Masking
# ============================================================================

# Common PII patterns
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone_kr": re.compile(r'\b0\d{1,2}-?\d{3,4}-?\d{4}\b'),  # Korean phone
    "phone_intl": re.compile(r'\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
    "ssn_kr": re.compile(r'\b\d{6}-?\d{7}\b'),  # Korean SSN
    "credit_card": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "url": re.compile(r'https?://[^\s]+'),
}


def detect_pii_patterns(text: str) -> Dict[str, int]:
    """
    Detect PII patterns in text

    Args:
        text: Input text to scan

    Returns:
        Dictionary of pattern types and counts found
    """
    findings = {}

    for pattern_name, pattern_regex in PII_PATTERNS.items():
        matches = pattern_regex.findall(text)
        if matches:
            findings[pattern_name] = len(matches)

    return findings


def mask_pii_patterns(text: str, mask_char: str = "*") -> str:
    """
    Mask PII patterns in text

    Args:
        text: Input text
        mask_char: Character to use for masking

    Returns:
        Text with PII patterns masked
    """
    masked = text

    # Email: show first 2 chars + domain
    masked = PII_PATTERNS["email"].sub(
        lambda m: f"{m.group()[:2]}{'*' * 5}@{m.group().split('@')[1]}",
        masked
    )

    # Phone: mask middle digits
    masked = PII_PATTERNS["phone_kr"].sub(
        lambda m: f"{m.group()[:3]}***{m.group()[-4:]}",
        masked
    )
    masked = PII_PATTERNS["phone_intl"].sub(
        lambda m: f"+{'*' * 8}{m.group()[-4:]}",
        masked
    )

    # SSN: completely mask
    masked = PII_PATTERNS["ssn_kr"].sub("******-*******", masked)

    # Credit card: mask middle digits
    masked = PII_PATTERNS["credit_card"].sub(
        lambda m: f"****-****-****-{m.group()[-4:]}",
        masked
    )

    # IP: mask last octet
    masked = PII_PATTERNS["ip_address"].sub(
        lambda m: f"{'.'.join(m.group().split('.')[:-1])}.***",
        masked
    )

    # URLs: show domain only
    masked = PII_PATTERNS["url"].sub(
        lambda m: f"{m.group()[:m.group().find('/', 8)]}/***/",
        masked
    )

    return masked


# ============================================================================
# Hashing Utilities
# ============================================================================

def hash_text(
    text: str,
    algorithm: str = "sha256",
    salt: Optional[str] = None,
    output_length: int = 16
) -> str:
    """
    Hash text using specified algorithm

    Args:
        text: Input text to hash
        algorithm: Hash algorithm (sha256, sha512, md5)
        salt: Optional salt for hashing
        output_length: Length of output hash (truncated)

    Returns:
        Hex hash string
    """
    if salt:
        text = f"{salt}{text}"

    if algorithm == "sha256":
        hash_obj = hashlib.sha256(text.encode())
    elif algorithm == "sha512":
        hash_obj = hashlib.sha512(text.encode())
    elif algorithm == "md5":
        hash_obj = hashlib.md5(text.encode())
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    return hash_obj.hexdigest()[:output_length]


def hash_with_prefix(text: str, prefix_length: int = 3) -> str:
    """
    Hash text but keep first N characters as prefix

    Args:
        text: Input text
        prefix_length: Characters to keep as prefix

    Returns:
        Prefix + hash
    """
    prefix = text[:prefix_length] if len(text) >= prefix_length else text
    hash_suffix = hash_text(text, output_length=12)
    return f"{prefix}...{hash_suffix}"


# ============================================================================
# Sampling Logic
# ============================================================================

def should_sample() -> bool:
    """
    Determine if current request should be sampled for logging

    Returns:
        True if should log, False otherwise
    """
    settings = get_privacy_settings()
    return random.random() < settings.sampling_rate


# ============================================================================
# Privacy-Aware Logging Wrapper
# ============================================================================

def sanitize_for_logging(
    user_text: str,
    user_opted_in: bool = False,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Sanitize user input for privacy-safe logging

    Args:
        user_text: User input text
        user_opted_in: Whether user explicitly opted in for logging
        context: Additional context metadata

    Returns:
        Dictionary with sanitized data safe for logging
    """
    settings = get_privacy_settings()

    result = {
        "user_text_logged": False,
        "privacy_level": settings.log_level.value,
        "text_length": len(user_text),
        "language_detected": _detect_language_simple(user_text)
    }

    # Add context if provided
    if context:
        result.update(context)

    # Handle different privacy levels
    if settings.log_level == LogLevel.NONE:
        # No user data logged at all
        result["user_text_hash"] = hash_text(user_text, output_length=8)

    elif settings.log_level == LogLevel.HASH_ONLY:
        # Only hash
        result["user_text_hash"] = hash_text(
            user_text,
            algorithm=settings.hash_algorithm,
            output_length=16
        )
        result["text_preview"] = hash_with_prefix(user_text, prefix_length=3)

    elif settings.log_level == LogLevel.SAMPLED:
        # Sample N% of requests
        if should_sample():
            result["user_text_logged"] = True
            result["sampled"] = True

            if settings.mask_patterns:
                result["user_text"] = mask_pii_patterns(user_text)
            else:
                result["user_text"] = user_text[:settings.truncate_length]
        else:
            result["user_text_hash"] = hash_text(user_text, output_length=12)
            result["sampled"] = False

    elif settings.log_level == LogLevel.OPT_IN:
        # Log only if user opted in
        if user_opted_in and settings.allow_opt_in:
            result["user_text_logged"] = True
            result["opted_in"] = True

            if settings.mask_patterns:
                result["user_text"] = mask_pii_patterns(user_text)
            else:
                result["user_text"] = user_text[:settings.truncate_length]
        else:
            result["user_text_hash"] = hash_text(user_text, output_length=12)
            result["opted_in"] = False

    elif settings.log_level == LogLevel.FULL:
        # Full logging (development only - NOT for production)
        result["user_text_logged"] = True
        result["user_text"] = user_text
        result["warning"] = "FULL logging enabled - disable in production!"

    # Detect PII patterns
    pii_detected = detect_pii_patterns(user_text)
    if pii_detected:
        result["pii_detected"] = pii_detected
        result["contains_pii"] = True
    else:
        result["contains_pii"] = False

    return result


def _detect_language_simple(text: str) -> str:
    """Simple language detection"""
    # Korean characters
    if re.search(r'[가-힣]', text):
        return "ko"
    # English
    elif re.search(r'[a-zA-Z]', text):
        return "en"
    else:
        return "unknown"


# ============================================================================
# Audit Log
# ============================================================================

class PrivacyAuditLog:
    """Track privacy-related events for audit purposes"""

    def __init__(self):
        self.events = []
        self.max_events = 1000

    def log_access(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        request_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log privacy-related access event

        Args:
            event_type: Type of event (access, export, delete, etc.)
            user_id: User identifier (hashed)
            request_hash: Hash of request
            metadata: Additional metadata
        """
        from datetime import datetime

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest()[:16] if user_id else None,
            "request_hash": request_hash,
            "metadata": metadata or {}
        }

        self.events.append(event)

        # LRU-style eviction
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def get_recent_events(self, n: int = 100) -> list:
        """Get N most recent events"""
        return self.events[-n:]

    def clear(self):
        """Clear audit log"""
        self.events = []


# Global audit log instance
_audit_log = PrivacyAuditLog()


def get_audit_log() -> PrivacyAuditLog:
    """Get global audit log instance"""
    return _audit_log


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'LogLevel',
    'PrivacySettings',
    'configure_privacy',
    'get_privacy_settings',
    'detect_pii_patterns',
    'mask_pii_patterns',
    'hash_text',
    'hash_with_prefix',
    'should_sample',
    'sanitize_for_logging',
    'PrivacyAuditLog',
    'get_audit_log'
]
