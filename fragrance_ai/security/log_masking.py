"""
Log Masking Utility
민감 정보(user_id, prompt_text 등)를 로그에서 마스킹
"""

import re
import hashlib
from typing import Any, Dict, Union
from loguru import logger


# 마스킹할 필드 목록
SENSITIVE_FIELDS = [
    "user_id",
    "user_email",
    "email",
    "prompt",
    "prompt_text",
    "query",
    "password",
    "api_key",
    "token",
    "access_token",
    "refresh_token",
    "secret",
    "private_key",
    "credit_card",
    "ssn",
    "phone",
    "address"
]

# 마스킹 패턴 (정규식)
EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
PHONE_PATTERN = re.compile(r'\d{3}[-.\s]?\d{3,4}[-.\s]?\d{4}')
CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')


def hash_value(value: str, prefix: str = "") -> str:
    """
    값을 SHA256 해시로 변환 (일부만 표시)

    Args:
        value: 원본 값
        prefix: 접두사

    Returns:
        해시된 값
    """
    if not value:
        return "[EMPTY]"

    hashed = hashlib.sha256(value.encode()).hexdigest()
    return f"{prefix}{hashed[:8]}***"


def mask_string(value: str, show_chars: int = 3) -> str:
    """
    문자열 마스킹 (앞/뒤 일부만 표시)

    Args:
        value: 원본 문자열
        show_chars: 표시할 문자 수

    Returns:
        마스킹된 문자열
    """
    if not value or len(value) <= show_chars * 2:
        return "***"

    return f"{value[:show_chars]}***{value[-show_chars:]}"


def mask_email(email: str) -> str:
    """
    이메일 마스킹

    Args:
        email: 이메일 주소

    Returns:
        마스킹된 이메일
    """
    if '@' not in email:
        return mask_string(email)

    local, domain = email.split('@', 1)
    masked_local = mask_string(local, show_chars=2)

    return f"{masked_local}@{domain}"


def mask_phone(phone: str) -> str:
    """
    전화번호 마스킹

    Args:
        phone: 전화번호

    Returns:
        마스킹된 전화번호
    """
    # 숫자만 추출
    digits = re.sub(r'\D', '', phone)

    if len(digits) <= 4:
        return "***"

    # 뒤 4자리만 표시
    return f"***-***-{digits[-4:]}"


def mask_text(text: str, max_length: int = 50) -> str:
    """
    긴 텍스트 마스킹 (프롬프트 등)

    Args:
        text: 원본 텍스트
        max_length: 표시할 최대 길이

    Returns:
        마스킹된 텍스트
    """
    if not text:
        return "[EMPTY]"

    if len(text) <= max_length:
        return f"{text[:20]}...[{len(text)} chars]"

    return f"{text[:20]}...[{len(text)} chars total, truncated]"


def mask_dict(data: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """
    딕셔너리 내 민감 정보 마스킹

    Args:
        data: 원본 딕셔너리
        deep: 중첩된 딕셔너리도 마스킹할지 여부

    Returns:
        마스킹된 딕셔너리
    """
    masked = {}

    for key, value in data.items():
        key_lower = key.lower()

        # 민감 필드 확인
        if any(field in key_lower for field in SENSITIVE_FIELDS):
            if isinstance(value, str):
                # 필드 타입에 따라 다른 마스킹 적용
                if 'email' in key_lower:
                    masked[key] = mask_email(value)
                elif 'phone' in key_lower:
                    masked[key] = mask_phone(value)
                elif 'user_id' in key_lower or 'id' in key_lower:
                    masked[key] = hash_value(value, prefix="user_")
                elif any(text_field in key_lower for text_field in ['prompt', 'query', 'text', 'message']):
                    masked[key] = mask_text(value)
                elif any(secret_field in key_lower for secret_field in ['password', 'token', 'key', 'secret']):
                    masked[key] = "***REDACTED***"
                else:
                    masked[key] = mask_string(value)
            else:
                masked[key] = f"[{type(value).__name__}]"
        else:
            # 민감 필드가 아니면 그대로 복사
            if deep and isinstance(value, dict):
                masked[key] = mask_dict(value, deep=True)
            elif deep and isinstance(value, list):
                masked[key] = [
                    mask_dict(item, deep=True) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                masked[key] = value

    return masked


def mask_text_patterns(text: str) -> str:
    """
    텍스트 내 민감 패턴 마스킹 (이메일, 전화번호, 카드번호 등)

    Args:
        text: 원본 텍스트

    Returns:
        마스킹된 텍스트
    """
    # 이메일 마스킹
    text = EMAIL_PATTERN.sub(lambda m: mask_email(m.group()), text)

    # 전화번호 마스킹
    text = PHONE_PATTERN.sub(lambda m: mask_phone(m.group()), text)

    # 카드번호 마스킹
    text = CREDIT_CARD_PATTERN.sub(lambda m: "****-****-****-" + m.group()[-4:], text)

    return text


def safe_log(message: str, data: Union[Dict, Any] = None, level: str = "info"):
    """
    안전한 로깅 (민감 정보 자동 마스킹)

    Args:
        message: 로그 메시지
        data: 추가 데이터
        level: 로그 레벨
    """
    # 메시지 내 패턴 마스킹
    masked_message = mask_text_patterns(message)

    if data:
        if isinstance(data, dict):
            masked_data = mask_dict(data)
        elif isinstance(data, str):
            masked_data = mask_text_patterns(data)
        else:
            masked_data = data

        log_func = getattr(logger, level, logger.info)
        log_func(f"{masked_message} | Data: {masked_data}")
    else:
        log_func = getattr(logger, level, logger.info)
        log_func(masked_message)


# 사용 예시
if __name__ == "__main__":
    # 테스트 데이터
    test_data = {
        "user_id": "user_12345",
        "user_email": "john.doe@example.com",
        "prompt_text": "I want to create a romantic fragrance with rose and vanilla notes for my girlfriend's birthday",
        "recipe_name": "Romantic Rose",
        "api_key": "sk_live_1234567890abcdef",
        "metadata": {
            "phone": "010-1234-5678",
            "address": "123 Main St, Seoul"
        }
    }

    print("=== Original Data ===")
    print(test_data)

    print("\n=== Masked Data ===")
    masked = mask_dict(test_data)
    print(masked)

    print("\n=== Safe Log Example ===")
    safe_log(
        "User request received",
        test_data,
        level="info"
    )

    print("\n=== Text Pattern Masking ===")
    sample_text = "Contact me at john.doe@example.com or 010-1234-5678. My card is 1234-5678-9012-3456."
    masked_text = mask_text_patterns(sample_text)
    print(f"Original: {sample_text}")
    print(f"Masked: {masked_text}")
