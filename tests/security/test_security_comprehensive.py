"""
Comprehensive Security Testing Suite
Tests all security components and attack vectors
"""

import pytest
import asyncio
import time
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
import jwt

from fragrance_ai.core.security import (
    SecurityManager, EncryptionManager, ApiKeyManager, RateLimiter,
    SecurityMonitor, PasswordValidator, AccessLevel, SecurityLevel
)
from fragrance_ai.core.security_middleware import SecurityMiddleware, SecurityConfig
from fragrance_ai.core.config import settings


class TestSecurityManager:
    """Test SecurityManager functionality"""

    @pytest.fixture
    def security_manager(self):
        return SecurityManager()

    @pytest.fixture
    def encryption_manager(self):
        return EncryptionManager()

    def test_encryption_decryption(self, encryption_manager):
        """Test data encryption and decryption"""
        original_data = "sensitive information 🔐"

        # Encrypt
        encrypted = encryption_manager.encrypt(original_data)
        assert encrypted != original_data
        assert len(encrypted) > len(original_data)

        # Decrypt
        decrypted = encryption_manager.decrypt(encrypted)
        assert decrypted == original_data

    def test_password_hashing(self, encryption_manager):
        """Test password hashing and verification"""
        password = "SuperSecure123!@#"

        # Hash password
        hashed = encryption_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are typically 60 chars

        # Verify correct password
        assert encryption_manager.verify_password(password, hashed)

        # Verify incorrect password
        assert not encryption_manager.verify_password("wrong_password", hashed)

    def test_secure_token_generation(self, encryption_manager):
        """Test secure token generation"""
        token1 = encryption_manager.generate_secure_token(32)
        token2 = encryption_manager.generate_secure_token(32)

        assert len(token1) > 40  # URL-safe base64 encoding
        assert len(token2) > 40
        assert token1 != token2  # Should be unique


class TestPasswordValidator:
    """Test password validation"""

    @pytest.fixture
    def validator(self):
        return PasswordValidator()

    def test_strong_password_validation(self, validator):
        """Test strong password validation"""
        strong_passwords = [
            "MyVerySecure123!Password",
            "Tr0ub4dor&3Security",
            "C0mpl3x!P@ssw0rd#2024"
        ]

        for password in strong_passwords:
            is_valid, errors = validator.validate_password(password)
            assert is_valid, f"Password '{password}' should be valid. Errors: {errors}"

    def test_weak_password_rejection(self, validator):
        """Test weak password rejection"""
        weak_passwords = [
            "password",           # Common weak password
            "123456789",         # Only numbers
            "short",             # Too short
            "nouppercase123!",   # No uppercase
            "NOLOWERCASE123!",   # No lowercase
            "NoNumbers!",        # No numbers
            "NoSpecialChars123", # No special characters
            "aaaa1234!",         # Repeating characters
        ]

        for password in weak_passwords:
            is_valid, errors = validator.validate_password(password)
            assert not is_valid, f"Password '{password}' should be invalid"
            assert len(errors) > 0

    def test_password_generation(self, validator):
        """Test secure password generation"""
        generated = validator.generate_secure_password(16)

        assert len(generated) == 16
        is_valid, errors = validator.validate_password(generated)
        assert is_valid, f"Generated password should be valid. Errors: {errors}"


class TestApiKeyManager:
    """Test API key management"""

    @pytest.fixture
    def api_manager(self):
        return ApiKeyManager()

    def test_api_key_creation(self, api_manager):
        """Test API key creation"""
        api_key = api_manager.create_api_key(
            name="Test Key",
            user_id="test_user",
            access_level=AccessLevel.AUTHENTICATED,
            permissions=["read", "write"],
            rate_limit=1000
        )

        assert api_key.startswith("sk_")  # Authenticated key prefix
        assert len(api_key) > 40

    def test_api_key_validation(self, api_manager):
        """Test API key validation"""
        # Create key
        api_key = api_manager.create_api_key(
            name="Test Key",
            user_id="test_user",
            access_level=AccessLevel.AUTHENTICATED,
            permissions=["read"],
            rate_limit=100
        )

        # Validate valid key
        key_info = api_manager.validate_api_key(api_key)
        assert key_info is not None
        assert key_info.user_id == "test_user"
        assert key_info.is_active

        # Validate invalid key
        invalid_key_info = api_manager.validate_api_key("invalid_key")
        assert invalid_key_info is None

    def test_api_key_revocation(self, api_manager):
        """Test API key revocation"""
        # Create key
        api_key = api_manager.create_api_key(
            name="Test Key",
            user_id="test_user",
            access_level=AccessLevel.AUTHENTICATED,
            permissions=["read"],
            rate_limit=100
        )

        # Get key info before revocation
        key_info = api_manager.validate_api_key(api_key)
        key_id = key_info.key_id

        # Revoke key
        revoked = api_manager.revoke_api_key(key_id)
        assert revoked

        # Validate revoked key should fail
        key_info_after = api_manager.validate_api_key(api_key)
        assert key_info_after is None


class TestRateLimiter:
    """Test rate limiting functionality"""

    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter()

    def test_rate_limiting_allows_normal_requests(self, rate_limiter):
        """Test that normal request rates are allowed"""
        user_id = "test_user"

        # Should allow normal requests
        for i in range(10):
            allowed, info = rate_limiter.is_allowed(user_id, 60, 1000)
            assert allowed
            time.sleep(0.01)  # Small delay

    def test_rate_limiting_blocks_excessive_requests(self, rate_limiter):
        """Test that excessive requests are blocked"""
        user_id = "test_user_excessive"

        # Exceed per-minute limit
        for i in range(65):  # Default is 60 per minute
            allowed, info = rate_limiter.is_allowed(user_id, 60, 1000)
            if i < 60:
                assert allowed
            else:
                assert not allowed
                assert "Rate limit exceeded" in info["reason"]

    def test_rate_limiting_statistics(self, rate_limiter):
        """Test rate limiting statistics"""
        user_id = "test_user_stats"

        # Make some requests
        for i in range(5):
            rate_limiter.is_allowed(user_id, 60, 1000)

        # Get statistics
        stats = rate_limiter.get_stats(user_id)
        assert stats["requests_last_minute"] == 5
        assert not stats["is_blocked"]


class TestSecurityMonitor:
    """Test security monitoring"""

    @pytest.fixture
    def monitor(self):
        return SecurityMonitor()

    def test_security_event_logging(self, monitor):
        """Test security event logging"""
        monitor.log_security_event(
            event_type="test_event",
            severity=SecurityLevel.HIGH,
            source_ip="192.168.1.100",
            user_id="test_user",
            details={"test": "data"}
        )

        assert len(monitor.events) == 1
        event = monitor.events[0]
        assert event.event_type == "test_event"
        assert event.severity == SecurityLevel.HIGH
        assert event.source_ip == "192.168.1.100"

    def test_suspicious_content_detection(self, monitor):
        """Test suspicious content detection"""
        malicious_contents = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "$(cat /etc/passwd)",
            "javascript:alert(1)"
        ]

        for content in malicious_contents:
            findings = monitor.check_suspicious_content(content, "192.168.1.1")
            assert len(findings) > 0, f"Should detect threat in: {content}"

    def test_request_pattern_analysis(self, monitor):
        """Test request pattern analysis"""
        # Test suspicious patterns
        analysis = monitor.analyze_request_pattern(
            ip="192.168.1.1",
            endpoint="/admin/config",
            user_agent=""  # Empty user agent is suspicious
        )

        assert analysis["is_suspicious"]
        assert "suspicious_user_agent" in analysis["reasons"]

    def test_security_summary(self, monitor):
        """Test security summary generation"""
        # Generate some events
        for i in range(5):
            monitor.log_security_event(
                event_type=f"test_event_{i}",
                severity=SecurityLevel.MEDIUM,
                source_ip=f"192.168.1.{i}",
                user_id=f"user_{i}"
            )

        summary = monitor.get_security_summary(24)
        assert summary["total_events"] == 5
        assert len(summary["unique_ips"]) == 5
        assert len(summary["affected_users"]) == 5


class TestSecurityMiddleware:
    """Test security middleware"""

    @pytest.fixture
    def app(self):
        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        @app.post("/test")
        async def test_post_endpoint(request: Request):
            body = await request.body()
            return {"received": body.decode() if body else ""}

        # Add security middleware
        config = SecurityConfig(
            enable_rate_limiting=True,
            enable_content_scanning=True,
            enable_ip_filtering=True,
            max_request_size=1024
        )
        app.add_middleware(SecurityMiddleware, config=config)

        return app

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    def test_normal_request_allowed(self, client):
        """Test that normal requests are allowed"""
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {"message": "success"}

    def test_security_headers_added(self, client):
        """Test that security headers are added"""
        response = client.get("/test")

        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]

        for header in security_headers:
            assert header in response.headers

    def test_sql_injection_blocked(self, client):
        """Test that SQL injection attempts are blocked"""
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users",
            "' AND (SELECT COUNT(*) FROM users) > 0 --"
        ]

        for payload in malicious_payloads:
            response = client.get(f"/test?query={payload}")
            assert response.status_code == 403

    def test_xss_blocked(self, client):
        """Test that XSS attempts are blocked"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<iframe src=javascript:alert(1)></iframe>"
        ]

        for payload in xss_payloads:
            response = client.post("/test", json={"data": payload})
            assert response.status_code == 403

    def test_oversized_request_blocked(self, client):
        """Test that oversized requests are blocked"""
        large_data = "x" * 2048  # Larger than 1024 byte limit

        response = client.post("/test", json={"data": large_data})
        assert response.status_code == 403

    def test_path_traversal_blocked(self, client):
        """Test that path traversal attempts are blocked"""
        traversal_payloads = [
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "....//....//....//etc//passwd"
        ]

        for payload in traversal_payloads:
            response = client.get(f"/test?file={payload}")
            assert response.status_code == 403


class TestIntegratedSecurity:
    """Test integrated security scenarios"""

    @pytest.fixture
    def security_system(self):
        """Complete security system setup"""
        return {
            'manager': SecurityManager(),
            'monitor': SecurityMonitor(),
            'rate_limiter': RateLimiter(),
            'api_manager': ApiKeyManager()
        }

    @pytest.mark.asyncio
    async def test_complete_attack_detection(self, security_system):
        """Test complete attack detection workflow"""
        context = SecurityContext(
            ip_address="192.168.1.100",
            user_agent="Malicious Bot 1.0",
            request_path="/admin",
            method="POST"
        )

        # Simulate malicious payload
        malicious_payload = {
            "query": "'; DROP TABLE users; --",
            "script": "<script>alert('xss')</script>",
            "file": "../../../etc/passwd"
        }

        # Validate request
        is_valid = await security_system['manager'].validate_request(
            context, malicious_payload
        )

        assert not is_valid  # Should block malicious request

    @pytest.mark.asyncio
    async def test_legitimate_user_workflow(self, security_system):
        """Test legitimate user workflow"""
        # Create API key for legitimate user
        api_key = security_system['api_manager'].create_api_key(
            name="Legitimate User",
            user_id="user123",
            access_level=AccessLevel.AUTHENTICATED,
            permissions=["read", "write"],
            rate_limit=1000
        )

        # Create session
        session_id = security_system['manager'].create_session("user123", 30)

        # Create context
        context = SecurityContext(
            user_id="user123",
            session_id=session_id,
            ip_address="192.168.1.50",
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            request_path="/api/search",
            method="GET"
        )

        # Validate legitimate request
        legitimate_payload = {"query": "rose fragrance"}

        is_valid = await security_system['manager'].validate_request(
            context, legitimate_payload
        )

        assert is_valid  # Should allow legitimate request

    def test_security_performance_under_load(self, security_system):
        """Test security system performance under load"""
        import threading
        import time

        results = []

        def simulate_request():
            start_time = time.time()

            # Simulate validation
            context = SecurityContext(
                ip_address="192.168.1.200",
                user_agent="Load Test Client",
                request_path="/api/test",
                method="GET"
            )

            # Check rate limiting
            allowed, _ = security_system['rate_limiter'].is_allowed("load_test_user")

            # Check for suspicious content
            findings = security_system['monitor'].check_suspicious_content(
                "normal search query", "192.168.1.200"
            )

            end_time = time.time()
            results.append(end_time - start_time)

        # Simulate 100 concurrent requests
        threads = []
        for _ in range(100):
            thread = threading.Thread(target=simulate_request)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check performance
        avg_time = sum(results) / len(results)
        max_time = max(results)

        assert avg_time < 0.01  # Should process in under 10ms on average
        assert max_time < 0.05  # Should never take more than 50ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])