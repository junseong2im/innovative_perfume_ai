"""
Enterprise Security Middleware
Advanced multi-layered security protection system
"""

import time
import json
import re
import ipaddress
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from urllib.parse import unquote
import base64
import hashlib

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from .security import (
    security_manager, SecurityContext, ThreatType, SecurityLevel,
    rate_limiter, security_monitor, api_key_manager
)
from .production_logging import get_logger, LogCategory
from .config import settings

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """통합 보안 미들웨어"""

    def __init__(
        self,
        app,
        rate_limit_per_minute: int = 60,
        max_request_size: int = 1024 * 1024,  # 1MB
        blocked_patterns: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.rate_limit_per_minute = rate_limit_per_minute
        self.max_request_size = max_request_size
        self.blocked_patterns = self._compile_patterns(blocked_patterns or [])

        # Rate limiting storage
        self.request_counts = defaultdict(deque)
        self.blocked_ips: Set[str] = set()

        # Security patterns
        self.sql_injection_patterns = self._compile_sql_patterns()
        self.xss_patterns = self._compile_xss_patterns()

    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """패턴 컴파일"""
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _compile_sql_patterns(self) -> List[re.Pattern]:
        """SQL Injection 패턴"""
        sql_patterns = [
            r"(\\'|\\\")",  # Escaped quotes
            r"(\\x27|\\x22)",  # Hex encoded quotes
            r"(union.*select)",  # Union select
            r"(select.*from)",  # Select from
            r"(insert.*into)",  # Insert into
            r"(delete.*from)",  # Delete from
            r"(update.*set)",  # Update set
            r"(drop.*table)",  # Drop table
            r"(exec\s*\()",  # Exec
            r"(script.*>)",  # Script tags
            r"(javascript:)",  # JavaScript protocol
            r"(vbscript:)",  # VBScript protocol
        ]
        return self._compile_patterns(sql_patterns)

    def _compile_xss_patterns(self) -> List[re.Pattern]:
        """XSS 패턴"""
        xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(<iframe[^>]*>.*?</iframe>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            r"(<img[^>]*onerror)",
            r"(<svg[^>]*onload)",
            r"(eval\s*\()",
            r"(alert\s*\()",
            r"(document\.cookie)",
            r"(window\.location)",
        ]
        return self._compile_patterns(xss_patterns)

    async def dispatch(self, request: Request, call_next):
        """메인 보안 체크 로직"""
        client_ip = self._get_client_ip(request)

        # 1. IP 차단 확인
        if client_ip in self.blocked_ips:
            return JSONResponse(
                status_code=403,
                content={"error": "IP blocked", "code": "SECURITY_IP_BLOCKED"}
            )

        # 2. Rate limiting
        if not self._check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "code": "SECURITY_RATE_LIMIT",
                    "retry_after": 60
                }
            )

        # 3. Request size check
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Request too large",
                    "code": "SECURITY_REQUEST_TOO_LARGE"
                }
            )

        # 4. Malicious pattern detection
        security_check = await self._check_security_patterns(request)
        if not security_check["safe"]:
            logger.warning(f"Security threat detected from {client_ip}: {security_check['threat_type']}")
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Malicious request detected",
                    "code": f"SECURITY_{security_check['threat_type'].upper()}",
                    "details": security_check['details']
                }
            )

        # 5. 보안 헤더 추가
        response = await call_next(request)
        self._add_security_headers(response)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출"""
        # X-Forwarded-For 헤더 확인 (프록시 환경)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        # X-Real-IP 헤더 확인
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        # 직접 연결
        return request.client.host if request.client else "unknown"

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Rate limiting 체크"""
        now = time.time()
        minute_ago = now - 60

        # 1분 이전 요청 제거
        request_times = self.request_counts[client_ip]
        while request_times and request_times[0] < minute_ago:
            request_times.popleft()

        # 현재 요청 추가
        request_times.append(now)

        # Rate limit 확인
        if len(request_times) > self.rate_limit_per_minute:
            # IP 차단 (10분)
            self.blocked_ips.add(client_ip)
            # 10분 후 자동 해제 (실제로는 별도 스케줄러 필요)
            return False

        return True

    async def _check_security_patterns(self, request: Request) -> Dict[str, any]:
        """보안 패턴 체크"""
        # URL 검사
        url_check = self._check_patterns(str(request.url), "URL")
        if not url_check["safe"]:
            return url_check

        # Query parameters 검사
        for key, value in request.query_params.items():
            param_check = self._check_patterns(f"{key}={value}", "QUERY_PARAM")
            if not param_check["safe"]:
                return param_check

        # Headers 검사 (특정 헤더만)
        dangerous_headers = ["user-agent", "referer", "x-forwarded-for"]
        for header in dangerous_headers:
            header_value = request.headers.get(header, "")
            header_check = self._check_patterns(header_value, "HEADER")
            if not header_check["safe"]:
                return header_check

        # Body 검사 (JSON인 경우)
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.body()
                if body:
                    body_str = body.decode("utf-8")
                    body_check = self._check_patterns(body_str, "BODY")
                    if not body_check["safe"]:
                        return body_check
            except Exception:
                pass  # Body 읽기 실패는 무시

        return {"safe": True}

    def _check_patterns(self, text: str, location: str) -> Dict[str, any]:
        """패턴 매칭 검사"""
        # SQL Injection 검사
        for pattern in self.sql_injection_patterns:
            if pattern.search(text):
                return {
                    "safe": False,
                    "threat_type": "sql_injection",
                    "details": f"Detected in {location}"
                }

        # XSS 검사
        for pattern in self.xss_patterns:
            if pattern.search(text):
                return {
                    "safe": False,
                    "threat_type": "xss",
                    "details": f"Detected in {location}"
                }

        # 차단된 패턴 검사
        for pattern in self.blocked_patterns:
            if pattern.search(text):
                return {
                    "safe": False,
                    "threat_type": "blocked_pattern",
                    "details": f"Detected in {location}"
                }

        return {"safe": True}

    def _add_security_headers(self, response: Response):
        """보안 헤더 추가"""
        security_headers = {
            # XSS 보호
            "X-XSS-Protection": "1; mode=block",

            # Content Type Sniffing 방지
            "X-Content-Type-Options": "nosniff",

            # Clickjacking 방지
            "X-Frame-Options": "DENY",

            # HSTS (HTTPS 전용)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",

            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'"
            ),

            # Referrer Policy
            "Referrer-Policy": "strict-origin-when-cross-origin",

            # Permissions Policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=(), "
                "vibrate=(), "
                "fullscreen=()"
            )
        }

        for header, value in security_headers.items():
            response.headers[header] = value


class InputValidator:
    """입력 검증 클래스"""

    @staticmethod
    def validate_text_input(text: str, max_length: int = 1000) -> Dict[str, any]:
        """텍스트 입력 검증"""
        if not isinstance(text, str):
            return {"valid": False, "error": "Input must be a string"}

        if len(text) > max_length:
            return {"valid": False, "error": f"Input too long (max {max_length} characters)"}

        if len(text.strip()) == 0:
            return {"valid": False, "error": "Input cannot be empty"}

        # 기본 HTML 태그 검사
        html_pattern = re.compile(r'<[^>]+>', re.IGNORECASE)
        if html_pattern.search(text):
            return {"valid": False, "error": "HTML tags not allowed"}

        return {"valid": True}

    @staticmethod
    def validate_numeric_input(value: any, min_val: float = None, max_val: float = None) -> Dict[str, any]:
        """숫자 입력 검증"""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return {"valid": False, "error": "Invalid numeric value"}

        if min_val is not None and num_value < min_val:
            return {"valid": False, "error": f"Value must be >= {min_val}"}

        if max_val is not None and num_value > max_val:
            return {"valid": False, "error": f"Value must be <= {max_val}"}

        return {"valid": True, "value": num_value}

    @staticmethod
    def sanitize_text(text: str) -> str:
        """텍스트 새니타이제이션"""
        if not isinstance(text, str):
            return ""

        # HTML 태그 제거
        html_pattern = re.compile(r'<[^>]+>')
        text = html_pattern.sub('', text)

        # 특수 문자 이스케이프
        escape_chars = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }

        for char, escaped in escape_chars.items():
            text = text.replace(char, escaped)

        return text.strip()


# 보안 검증 데코레이터
def validate_input(validation_rules: Dict[str, Dict]):
    """입력 검증 데코레이터"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 요청 객체에서 데이터 추출
            for param_name, rules in validation_rules.items():
                if param_name in kwargs:
                    value = kwargs[param_name]

                    if rules.get("type") == "text":
                        result = InputValidator.validate_text_input(
                            value,
                            max_length=rules.get("max_length", 1000)
                        )
                        if not result["valid"]:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid {param_name}: {result['error']}"
                            )

                        # 새니타이즈된 값으로 교체
                        kwargs[param_name] = InputValidator.sanitize_text(value)

                    elif rules.get("type") == "numeric":
                        result = InputValidator.validate_numeric_input(
                            value,
                            min_val=rules.get("min"),
                            max_val=rules.get("max")
                        )
                        if not result["valid"]:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Invalid {param_name}: {result['error']}"
                            )

                        kwargs[param_name] = result["value"]

            return await func(*args, **kwargs)
        return wrapper
    return decorator