"""
보안 테스트
SQL Injection, XSS, Rate Limiting 등 보안 기능 테스트
"""

import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch

from fragrance_ai.api.main import app
from fragrance_ai.core.security_middleware import InputValidator


@pytest.fixture
def client():
    """테스트 클라이언트"""
    return TestClient(app)


class TestSecurityMiddleware:
    """보안 미들웨어 테스트"""

    def test_sql_injection_detection(self, client):
        """SQL Injection 탐지 테스트"""
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'/**/OR/**/1=1#",
            "UNION SELECT * FROM users",
            "1; DELETE FROM users WHERE 1=1"
        ]

        for payload in malicious_payloads:
            response = client.post(
                "/api/v2/semantic-search",
                json={"query": payload, "top_k": 5}
            )
            assert response.status_code == 400
            data = response.json()
            assert "SECURITY_SQL_INJECTION" in data.get("code", "")

    def test_xss_detection(self, client):
        """XSS 탐지 테스트"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "<iframe src=javascript:alert('xss')></iframe>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>"
        ]

        for payload in xss_payloads:
            response = client.post(
                "/api/v2/semantic-search",
                json={"query": payload, "top_k": 5}
            )
            assert response.status_code == 400
            data = response.json()
            assert "SECURITY_XSS" in data.get("code", "")

    def test_rate_limiting(self, client):
        """Rate Limiting 테스트"""
        # 짧은 시간에 많은 요청 보내기
        responses = []
        for i in range(105):  # 분당 100개 제한 초과
            response = client.post(
                "/api/v2/semantic-search",
                json={"query": f"test query {i}", "top_k": 5}
            )
            responses.append(response.status_code)

        # 일부 요청이 429 (Too Many Requests)로 거부되어야 함
        rate_limited_responses = [code for code in responses if code == 429]
        assert len(rate_limited_responses) > 0

    def test_request_size_limit(self, client):
        """요청 크기 제한 테스트"""
        # 매우 큰 요청 생성 (2MB 초과)
        large_query = "a" * (3 * 1024 * 1024)  # 3MB

        response = client.post(
            "/api/v2/semantic-search",
            json={"query": large_query, "top_k": 5}
        )

        assert response.status_code == 413  # Request Entity Too Large
        data = response.json()
        assert "SECURITY_REQUEST_TOO_LARGE" in data.get("code", "")

    def test_security_headers(self, client):
        """보안 헤더 테스트"""
        response = client.get("/health")

        # 필수 보안 헤더 확인
        security_headers = [
            "X-XSS-Protection",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy"
        ]

        for header in security_headers:
            assert header in response.headers

        # 특정 헤더 값 확인
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]

    def test_malicious_user_agent(self, client):
        """악성 User-Agent 탐지 테스트"""
        malicious_user_agents = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "sqlmap/1.0",
        ]

        for user_agent in malicious_user_agents:
            response = client.get(
                "/health",
                headers={"User-Agent": user_agent}
            )
            # User-Agent 내 악성 패턴이 탐지되어야 함
            assert response.status_code in [400, 403]


class TestInputValidator:
    """입력 검증기 테스트"""

    def test_text_validation_success(self):
        """정상 텍스트 검증 테스트"""
        result = InputValidator.validate_text_input("정상적인 텍스트 입력")
        assert result["valid"] is True

    def test_text_validation_too_long(self):
        """너무 긴 텍스트 검증 테스트"""
        long_text = "a" * 1001
        result = InputValidator.validate_text_input(long_text, max_length=1000)
        assert result["valid"] is False
        assert "too long" in result["error"]

    def test_text_validation_empty(self):
        """빈 텍스트 검증 테스트"""
        result = InputValidator.validate_text_input("   ")
        assert result["valid"] is False
        assert "empty" in result["error"]

    def test_text_validation_html_tags(self):
        """HTML 태그 포함 텍스트 검증 테스트"""
        html_text = "Hello <script>alert('xss')</script> World"
        result = InputValidator.validate_text_input(html_text)
        assert result["valid"] is False
        assert "HTML tags" in result["error"]

    def test_numeric_validation_success(self):
        """정상 숫자 검증 테스트"""
        result = InputValidator.validate_numeric_input(5.5, min_val=0, max_val=10)
        assert result["valid"] is True
        assert result["value"] == 5.5

    def test_numeric_validation_out_of_range(self):
        """범위 초과 숫자 검증 테스트"""
        result = InputValidator.validate_numeric_input(15, min_val=0, max_val=10)
        assert result["valid"] is False
        assert "must be <=" in result["error"]

    def test_numeric_validation_invalid_type(self):
        """잘못된 타입 숫자 검증 테스트"""
        result = InputValidator.validate_numeric_input("not_a_number")
        assert result["valid"] is False
        assert "Invalid numeric" in result["error"]

    def test_text_sanitization(self):
        """텍스트 새니타이제이션 테스트"""
        malicious_text = "<script>alert('xss')</script> & \"quotes\" 'test'"
        sanitized = InputValidator.sanitize_text(malicious_text)

        assert "<script>" not in sanitized
        assert "&amp;" in sanitized
        assert "&quot;" in sanitized
        assert "&#x27;" in sanitized


class TestAPIInputValidation:
    """API 입력 검증 테스트"""

    def test_semantic_search_input_validation(self, client):
        """의미 검색 입력 검증 테스트"""
        # 잘못된 top_k 값
        response = client.post(
            "/api/v2/semantic-search",
            json={"query": "테스트", "top_k": -5}
        )
        assert response.status_code == 400

        # 너무 긴 쿼리
        long_query = "a" * 501  # 500자 제한 초과
        response = client.post(
            "/api/v2/semantic-search",
            json={"query": long_query, "top_k": 5}
        )
        assert response.status_code == 400

        # 잘못된 similarity 값
        response = client.post(
            "/api/v2/semantic-search",
            json={"query": "테스트", "top_k": 5, "min_similarity": 1.5}
        )
        assert response.status_code == 400

    @patch('fragrance_ai.api.main.app.state.embedding_model')
    @patch('fragrance_ai.api.main.perform_single_search')
    def test_valid_input_processing(self, mock_search, mock_model, client):
        """유효한 입력 처리 테스트"""
        # Mock 설정
        mock_model.encode_async.return_value.embeddings = [[0.1] * 384]
        mock_search.return_value = []

        response = client.post(
            "/api/v2/semantic-search",
            json={
                "query": "정상적인 향수 검색 쿼리",
                "top_k": 10,
                "min_similarity": 0.7
            }
        )

        assert response.status_code == 200


class TestSecurityBypass:
    """보안 우회 시도 테스트"""

    def test_encoded_payload_detection(self, client):
        """인코딩된 악성 페이로드 탐지 테스트"""
        # URL 인코딩된 스크립트
        encoded_payloads = [
            "%3Cscript%3Ealert%28%27xss%27%29%3C%2Fscript%3E",
            "%27%3B%20DROP%20TABLE%20users%3B%20--",
        ]

        for payload in encoded_payloads:
            response = client.get(f"/health?test={payload}")
            # URL 디코딩 후 탐지되어야 함
            assert response.status_code in [400, 403]

    def test_case_variation_attacks(self, client):
        """대소문자 변형 공격 테스트"""
        case_variants = [
            "<ScRiPt>alert('xss')</ScRiPt>",
            "uNiOn SeLeCt * FrOm users",
        ]

        for payload in case_variants:
            response = client.post(
                "/api/v2/semantic-search",
                json={"query": payload, "top_k": 5}
            )
            assert response.status_code == 400

    def test_comment_based_injection(self, client):
        """주석 기반 Injection 테스트"""
        comment_payloads = [
            "admin'/**/OR/**/1=1#",
            "1'/**/UNION/**/SELECT/**/null--",
        ]

        for payload in comment_payloads:
            response = client.post(
                "/api/v2/semantic-search",
                json={"query": payload, "top_k": 5}
            )
            assert response.status_code == 400


@pytest.mark.security
class TestSecurityCompliance:
    """보안 규정 준수 테스트"""

    def test_no_sensitive_data_in_logs(self, client):
        """로그에 민감한 데이터 노출 방지 테스트"""
        # 실제 로그 확인은 더 복잡한 구현 필요
        # 여기서는 기본적인 응답 확인만
        response = client.post(
            "/api/v2/semantic-search",
            json={"query": "password123", "top_k": 5}
        )

        # 응답에 민감한 정보가 노출되지 않아야 함
        response_text = response.text.lower()
        assert "password" not in response_text

    def test_error_information_disclosure(self, client):
        """에러 정보 노출 방지 테스트"""
        # 잘못된 요청으로 에러 유발
        response = client.post(
            "/api/v2/semantic-search",
            json={"invalid": "request"}
        )

        # 에러 응답에 시스템 정보가 노출되지 않아야 함
        data = response.json()
        error_message = str(data).lower()

        # 노출되면 안 되는 정보들
        sensitive_info = [
            "traceback",
            "file path",
            "database",
            "internal",
            "exception"
        ]

        for info in sensitive_info:
            assert info not in error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])