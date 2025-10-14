"""
Log Masking - PII 및 민감 정보 마스킹

로그에서 개인 정보(PII)와 API 키 등 민감 정보를 자동으로 마스킹
"""

import re
import logging
from typing import Dict, List, Pattern, Callable
from dataclasses import dataclass


@dataclass
class MaskingRule:
    """마스킹 규칙"""
    name: str
    pattern: Pattern
    replacement: str
    description: str


class LogMasker:
    """
    로그 마스커

    민감 정보를 자동으로 마스킹:
    - API keys
    - Email addresses
    - Phone numbers
    - Credit card numbers
    - IP addresses
    - Passwords
    - JWT tokens
    - Custom patterns
    """

    def __init__(self, enable_all_rules: bool = True):
        self.rules: List[MaskingRule] = []
        self.enabled = True

        if enable_all_rules:
            self._register_default_rules()

    def _register_default_rules(self):
        """기본 마스킹 규칙 등록"""

        # 1. API Keys (generic pattern)
        self.add_rule(
            name="api_key",
            pattern=re.compile(
                r'\b[A-Za-z0-9]{20,}(?:[_-][A-Za-z0-9]+)*\b'
                r'|'
                r'\b(?:api[_-]?key|apikey|api-key|access[_-]?token|secret[_-]?key)["\']?\s*[:=]\s*["\']?([A-Za-z0-9_-]{20,})',
                re.IGNORECASE
            ),
            replacement="***API_KEY***",
            description="API keys and access tokens"
        )

        # 2. Email addresses
        self.add_rule(
            name="email",
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            replacement="***EMAIL***",
            description="Email addresses"
        )

        # 3. Phone numbers (various formats)
        self.add_rule(
            name="phone",
            pattern=re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
                r'|'
                r'\b\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b'  # Korean phone numbers
            ),
            replacement="***PHONE***",
            description="Phone numbers"
        )

        # 4. Credit card numbers
        self.add_rule(
            name="credit_card",
            pattern=re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            replacement="***CARD***",
            description="Credit card numbers"
        )

        # 5. IP addresses
        self.add_rule(
            name="ip_address",
            pattern=re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            replacement="***IP***",
            description="IP addresses"
        )

        # 6. Passwords in logs
        self.add_rule(
            name="password",
            pattern=re.compile(
                r'(?:password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']+)',
                re.IGNORECASE
            ),
            replacement=r'password=***PASSWORD***',
            description="Passwords"
        )

        # 7. JWT tokens
        self.add_rule(
            name="jwt_token",
            pattern=re.compile(
                r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'
            ),
            replacement="***JWT***",
            description="JWT tokens"
        )

        # 8. Bearer tokens
        self.add_rule(
            name="bearer_token",
            pattern=re.compile(
                r'Bearer\s+[A-Za-z0-9_-]{20,}',
                re.IGNORECASE
            ),
            replacement="Bearer ***TOKEN***",
            description="Bearer tokens"
        )

        # 9. SSH Private Keys
        self.add_rule(
            name="ssh_key",
            pattern=re.compile(
                r'-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----'
            ),
            replacement="***SSH_PRIVATE_KEY***",
            description="SSH private keys"
        )

        # 10. Korean resident registration numbers (주민등록번호)
        self.add_rule(
            name="rrn_korea",
            pattern=re.compile(r'\b\d{6}[-\s]?[1-4]\d{6}\b'),
            replacement="***RRN***",
            description="Korean resident registration numbers"
        )

        # 11. Social Security Numbers (SSN)
        self.add_rule(
            name="ssn",
            pattern=re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            replacement="***SSN***",
            description="Social Security Numbers"
        )

    def add_rule(
        self,
        name: str,
        pattern: Pattern,
        replacement: str,
        description: str = ""
    ):
        """
        커스텀 마스킹 규칙 추가

        Args:
            name: 규칙 이름
            pattern: 정규식 패턴
            replacement: 대체 문자열
            description: 설명
        """
        rule = MaskingRule(
            name=name,
            pattern=pattern,
            replacement=replacement,
            description=description
        )
        self.rules.append(rule)

    def mask(self, text: str) -> str:
        """
        텍스트 마스킹

        Args:
            text: 원본 텍스트

        Returns:
            마스킹된 텍스트
        """
        if not self.enabled or not text:
            return text

        masked_text = text

        for rule in self.rules:
            try:
                masked_text = rule.pattern.sub(rule.replacement, masked_text)
            except Exception as e:
                # Log masking should never break the application
                logging.debug(f"Masking rule '{rule.name}' failed: {e}")

        return masked_text

    def mask_dict(self, data: Dict) -> Dict:
        """
        딕셔너리 내 모든 문자열 마스킹

        Args:
            data: 원본 딕셔너리

        Returns:
            마스킹된 딕셔너리
        """
        if not isinstance(data, dict):
            return data

        masked_data = {}

        for key, value in data.items():
            if isinstance(value, str):
                masked_data[key] = self.mask(value)
            elif isinstance(value, dict):
                masked_data[key] = self.mask_dict(value)
            elif isinstance(value, list):
                masked_data[key] = [
                    self.mask(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                masked_data[key] = value

        return masked_data

    def disable(self):
        """마스킹 비활성화"""
        self.enabled = False

    def enable(self):
        """마스킹 활성화"""
        self.enabled = True

    def get_rules(self) -> List[str]:
        """등록된 규칙 목록 조회"""
        return [f"{rule.name}: {rule.description}" for rule in self.rules]


# =============================================================================
# Logging Handler with Masking
# =============================================================================

class MaskingLogHandler(logging.Handler):
    """
    마스킹 로그 핸들러

    모든 로그 메시지를 자동으로 마스킹
    """

    def __init__(self, base_handler: logging.Handler, masker: LogMasker):
        super().__init__()
        self.base_handler = base_handler
        self.masker = masker

    def emit(self, record: logging.LogRecord):
        """로그 레코드 emit (마스킹 적용)"""
        try:
            # Mask log message
            if hasattr(record, 'msg'):
                record.msg = self.masker.mask(str(record.msg))

            # Mask args
            if hasattr(record, 'args') and record.args:
                record.args = tuple(
                    self.masker.mask(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )

            # Delegate to base handler
            self.base_handler.emit(record)

        except Exception:
            self.handleError(record)


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    # Create masker
    masker = LogMasker()

    print("=== Registered Masking Rules ===")
    for rule in masker.get_rules():
        print(f"  - {rule}")
    print()

    # Test masking
    test_cases = [
        # API Key
        "API_KEY=sk_test_abc123def456ghi789jkl012mno345pqr678",

        # Email
        "User email: john.doe@example.com",

        # Phone
        "Contact: +1-555-123-4567 or (555) 987-6543",

        # Credit Card
        "Card: 4532-1234-5678-9010",

        # IP Address
        "Server IP: 192.168.1.100",

        # Password
        'password="super_secret_password_123"',

        # JWT Token
        "Authorization: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",

        # Korean RRN
        "주민등록번호: 901234-1234567",

        # Multiple sensitive data
        "User: john@example.com, Phone: 555-1234, API Key: abc123def456ghi789jkl012"
    ]

    print("=== Test Cases ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Original:  {test_case}")
        print(f"  Masked:    {masker.mask(test_case)}")

    # Test dictionary masking
    print("\n=== Dictionary Masking ===")
    data = {
        "user_email": "admin@company.com",
        "api_key": "sk_live_abcdef123456789",
        "phone": "555-1234-5678",
        "settings": {
            "password": "secret123",
            "backup_ip": "10.0.0.1"
        }
    }

    print("Original:")
    import json
    print(json.dumps(data, indent=2))

    masked_data = masker.mask_dict(data)
    print("\nMasked:")
    print(json.dumps(masked_data, indent=2))

    # Test with logger
    print("\n=== Logger with Masking ===")
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # Create masking handler
    console_handler = logging.StreamHandler()
    masking_handler = MaskingLogHandler(console_handler, masker)
    logger.addHandler(masking_handler)

    logger.info("User logged in: email=user@example.com, ip=192.168.1.1")
    logger.info("API request with key: api_key=sk_test_abc123")
    logger.warning("Failed login attempt from 10.20.30.40 for admin@company.com")
