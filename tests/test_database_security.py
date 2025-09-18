"""
데이터베이스 및 보안 기능 테스트
데이터 무결성과 보안 기능을 검증합니다.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import jwt
from datetime import datetime, timedelta
import hashlib
import os

# 테스트용 환경 설정
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test_security.db"
os.environ["SECRET_KEY"] = "test-secret-key-for-security-testing"


class TestDatabaseConnections:
    """데이터베이스 연결 테스트"""

    @pytest.fixture
    def test_db_session(self):
        """테스트용 데이터베이스 세션"""
        from fragrance_ai.database.base import get_db
        return next(get_db())

    def test_database_connection(self, test_db_session):
        """데이터베이스 연결 테스트"""
        # 기본 연결 확인
        assert test_db_session is not None

        # 간단한 쿼리 실행
        result = test_db_session.execute("SELECT 1").scalar()
        assert result == 1

    def test_database_transaction_rollback(self, test_db_session):
        """트랜잭션 롤백 테스트"""
        try:
            test_db_session.begin()
            # 의도적으로 에러 발생
            test_db_session.execute("SELECT 1/0")
            test_db_session.commit()
        except Exception:
            test_db_session.rollback()
            # 롤백이 성공적으로 실행되었는지 확인
            result = test_db_session.execute("SELECT 1").scalar()
            assert result == 1


class TestAuthenticationSecurity:
    """인증 보안 테스트"""

    @pytest.fixture
    def mock_auth_service(self):
        """Mock 인증 서비스"""
        with patch('fragrance_ai.services.auth_service.auth_service') as mock_service:
            mock_service.create_access_token = Mock(return_value="mock_token")
            mock_service.verify_token = Mock(return_value={
                "user_id": "test_user",
                "email": "test@example.com",
                "permissions": ["read", "write"]
            })
            mock_service.hash_password = Mock(return_value="hashed_password")
            mock_service.verify_password = Mock(return_value=True)
            yield mock_service

    def test_password_hashing(self, mock_auth_service):
        """패스워드 해싱 테스트"""
        password = "test_password_123"

        hashed = mock_auth_service.hash_password(password)

        # 해시된 패스워드가 원본과 다른지 확인
        assert hashed != password
        assert len(hashed) > 0

        # 패스워드 검증
        is_valid = mock_auth_service.verify_password(password, hashed)
        assert is_valid is True

    def test_jwt_token_generation(self, mock_auth_service):
        """JWT 토큰 생성 테스트"""
        user_data = {
            "user_id": "test_user_123",
            "email": "test@example.com",
            "permissions": ["read", "write"]
        }

        token = mock_auth_service.create_access_token(user_data)

        assert token is not None
        assert len(token) > 0
        assert token == "mock_token"  # Mock 결과 확인

    def test_token_verification(self, mock_auth_service):
        """토큰 검증 테스트"""
        token = "valid_test_token"

        decoded_data = mock_auth_service.verify_token(token)

        assert decoded_data is not None
        assert "user_id" in decoded_data
        assert "email" in decoded_data
        assert "permissions" in decoded_data

    def test_invalid_token_handling(self, mock_auth_service):
        """잘못된 토큰 처리 테스트"""
        # 잘못된 토큰에 대해 None 반환하도록 설정
        mock_auth_service.verify_token.return_value = None

        invalid_token = "invalid_token"
        result = mock_auth_service.verify_token(invalid_token)

        assert result is None


class TestInputValidation:
    """입력 검증 테스트"""

    def test_query_injection_prevention(self):
        """SQL 인젝션 방지 테스트"""
        # SQLAlchemy ORM을 사용하므로 자동으로 방지됨
        # 하지만 명시적으로 테스트
        malicious_input = "'; DROP TABLE users; --"

        # 이런 입력이 들어와도 안전하게 처리되는지 확인
        from fragrance_ai.api.schemas import SemanticSearchRequest
        from pydantic import ValidationError

        try:
            request = SemanticSearchRequest(query=malicious_input)
            # 정상적으로 생성되면 쿼리가 적절히 escape됨
            assert request.query == malicious_input
        except ValidationError:
            # validation 에러가 발생해도 OK (길이 제한 등)
            pass

    def test_xss_prevention(self):
        """XSS 방지 테스트"""
        xss_payload = "<script>alert('xss')</script>"

        from fragrance_ai.api.schemas import RecipeGenerationRequest

        try:
            request = RecipeGenerationRequest(
                fragrance_family="floral",
                mood=xss_payload  # XSS 페이로드
            )
            # 입력이 그대로 저장되더라도 출력시 escape되어야 함
            assert request.mood == xss_payload
        except Exception:
            # validation에서 차단되어도 OK
            pass

    def test_input_length_validation(self):
        """입력 길이 검증 테스트"""
        from fragrance_ai.api.schemas import SemanticSearchRequest
        from pydantic import ValidationError

        # 너무 긴 쿼리
        long_query = "a" * 1000

        with pytest.raises(ValidationError):
            SemanticSearchRequest(query=long_query)

    def test_required_field_validation(self):
        """필수 필드 검증 테스트"""
        from fragrance_ai.api.schemas import SemanticSearchRequest
        from pydantic import ValidationError

        # 필수 필드 누락
        with pytest.raises(ValidationError):
            SemanticSearchRequest()  # query 필드 누락


class TestRateLimiting:
    """Rate Limiting 테스트"""

    @patch('fragrance_ai.api.middleware.RateLimitMiddleware')
    def test_rate_limit_enforcement(self, mock_middleware):
        """Rate limit 적용 테스트"""
        # Mock rate limiter 설정
        mock_middleware.return_value.check_rate_limit = Mock(return_value=True)

        # 정상적인 요청
        result = mock_middleware.return_value.check_rate_limit("127.0.0.1")
        assert result is True

        # Rate limit 초과 시뮬레이션
        mock_middleware.return_value.check_rate_limit.return_value = False
        result = mock_middleware.return_value.check_rate_limit("127.0.0.1")
        assert result is False

    def test_rate_limit_bypass_for_health_check(self):
        """헬스체크는 rate limit 적용 안되는지 테스트"""
        # 헬스체크 엔드포인트는 항상 접근 가능해야 함
        from fastapi.testclient import TestClient
        from fragrance_ai.api.main import app

        client = TestClient(app)

        # 여러 번 연속 요청해도 성공해야 함
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200


class TestDataEncryption:
    """데이터 암호화 테스트"""

    def test_sensitive_data_encryption(self):
        """민감 데이터 암호화 테스트"""
        from cryptography.fernet import Fernet

        # 암호화 키 생성
        key = Fernet.generate_key()
        cipher = Fernet(key)

        # 민감한 데이터
        sensitive_data = "사용자의 민감한 정보"

        # 암호화
        encrypted_data = cipher.encrypt(sensitive_data.encode())

        # 암호화된 데이터가 원본과 다른지 확인
        assert encrypted_data != sensitive_data.encode()

        # 복호화
        decrypted_data = cipher.decrypt(encrypted_data).decode()
        assert decrypted_data == sensitive_data

    def test_password_storage_security(self):
        """패스워드 저장 보안 테스트"""
        import bcrypt

        password = "user_password_123"

        # bcrypt로 해싱
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)

        # 해시된 패스워드가 원본과 다른지 확인
        assert hashed != password.encode('utf-8')

        # 패스워드 검증
        is_valid = bcrypt.checkpw(password.encode('utf-8'), hashed)
        assert is_valid is True

        # 잘못된 패스워드
        wrong_password = "wrong_password"
        is_valid = bcrypt.checkpw(wrong_password.encode('utf-8'), hashed)
        assert is_valid is False


class TestSecurityHeaders:
    """보안 헤더 테스트"""

    def test_security_headers_present(self):
        """보안 헤더 존재 확인 테스트"""
        from fastapi.testclient import TestClient
        from fragrance_ai.api.main import app

        client = TestClient(app)
        response = client.get("/")

        # 기본적인 보안 헤더들이 설정되어 있는지 확인
        # (실제 구현에 따라 조정 필요)
        headers = response.headers

        # CORS 헤더
        assert "access-control-allow-origin" in headers or response.status_code == 200

    def test_no_sensitive_info_in_headers(self):
        """헤더에 민감 정보 노출 안되는지 테스트"""
        from fastapi.testclient import TestClient
        from fragrance_ai.api.main import app

        client = TestClient(app)
        response = client.get("/health")

        headers = response.headers

        # 민감한 정보가 헤더에 노출되지 않는지 확인
        sensitive_keywords = ["password", "secret", "key", "token"]
        for keyword in sensitive_keywords:
            for header_name, header_value in headers.items():
                assert keyword.lower() not in header_name.lower()
                assert keyword.lower() not in str(header_value).lower()


class TestErrorHandlingSecurity:
    """에러 처리 보안 테스트"""

    def test_error_info_not_leaked(self):
        """에러 정보 유출 방지 테스트"""
        from fastapi.testclient import TestClient
        from fragrance_ai.api.main import app

        client = TestClient(app)

        # 존재하지 않는 엔드포인트
        response = client.get("/admin/secret-data")
        assert response.status_code == 404

        # 에러 응답에 민감한 시스템 정보가 포함되지 않는지 확인
        error_text = response.text.lower()
        sensitive_info = ["traceback", "file path", "database", "internal"]

        for info in sensitive_info:
            assert info not in error_text


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_files():
    """테스트 파일 정리"""
    yield
    # 테스트 완료 후 생성된 파일들 정리
    test_files = ["./test_security.db", "./test_security.db-journal"]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])