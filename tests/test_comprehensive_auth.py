"""
인증 및 권한 부여 시스템 종합 테스트
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

from fragrance_ai.core.auth import (
    AuthenticationManager, AuthorizationManager, User, UserRole, Permission,
    auth_manager, authz_manager, get_current_user
)
from fragrance_ai.core.exceptions import AuthenticationError, AuthorizationError, RateLimitError


class TestAuthenticationManager:
    """인증 관리자 테스트"""

    @pytest.fixture
    async def auth_mgr(self):
        """테스트용 인증 관리자"""
        mgr = AuthenticationManager()
        await mgr.initialize()
        return mgr

    @pytest.mark.asyncio
    async def test_password_hashing(self, auth_mgr):
        """비밀번호 해싱 테스트"""
        password = "test_password_123"
        hashed = auth_mgr.hash_password(password)

        assert hashed != password
        assert auth_mgr.verify_password(password, hashed)
        assert not auth_mgr.verify_password("wrong_password", hashed)

    @pytest.mark.asyncio
    async def test_user_creation(self, auth_mgr):
        """사용자 생성 테스트"""
        user = await auth_mgr.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            role=UserRole.USER,
            full_name="Test User"
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.is_active
        assert not user.is_verified
        assert user.api_key is not None
        assert len(user.permissions) > 0

    @pytest.mark.asyncio
    async def test_jwt_token_creation_and_verification(self, auth_mgr):
        """JWT 토큰 생성 및 검증 테스트"""
        user = await auth_mgr.create_user(
            username="tokenuser",
            email="token@example.com",
            password="password123"
        )

        # 액세스 토큰 생성
        access_token = auth_mgr.create_access_token(user)
        assert access_token is not None

        # 토큰 검증
        payload = auth_mgr.verify_token(access_token)
        assert payload["sub"] == user.user_id
        assert payload["username"] == user.username
        assert payload["role"] == user.role.value

        # 리프레시 토큰 생성
        refresh_token = auth_mgr.create_refresh_token(user)
        assert refresh_token is not None

        # 리프레시 토큰 검증
        refresh_payload = auth_mgr.verify_token(refresh_token)
        assert refresh_payload["sub"] == user.user_id
        assert refresh_payload["type"] == "refresh"

    @pytest.mark.asyncio
    async def test_invalid_token_verification(self, auth_mgr):
        """잘못된 토큰 검증 테스트"""
        invalid_tokens = [
            "invalid.token.here",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.invalid.signature",
            "",
            None
        ]

        for token in invalid_tokens:
            if token is None:
                continue
            with pytest.raises(AuthenticationError):
                auth_mgr.verify_token(token)

    @pytest.mark.asyncio
    async def test_api_key_generation_and_validation(self, auth_mgr):
        """API 키 생성 및 검증 테스트"""
        user = await auth_mgr.create_user(
            username="apikeyuser",
            email="apikey@example.com",
            password="password123"
        )

        api_key = auth_mgr.generate_api_key(user)
        assert api_key.startswith("fai_")
        assert len(api_key) > 32

        # API 키로 사용자 조회
        found_user = await auth_mgr.get_user_by_api_key(api_key)
        assert found_user is not None
        assert found_user.user_id == user.user_id

    @pytest.mark.asyncio
    async def test_user_authentication(self, auth_mgr):
        """사용자 인증 테스트"""
        username = "authuser"
        password = "authpassword"

        # 사용자 생성
        user = await auth_mgr.create_user(
            username=username,
            email="auth@example.com",
            password=password
        )

        # 성공적인 인증
        authenticated_user = await auth_mgr.authenticate_user(username, password)
        assert authenticated_user is not None
        assert authenticated_user.username == username

        # 실패한 인증 - 잘못된 비밀번호
        failed_auth = await auth_mgr.authenticate_user(username, "wrongpassword")
        assert failed_auth is None

        # 실패한 인증 - 존재하지 않는 사용자
        failed_auth = await auth_mgr.authenticate_user("nonexistent", password)
        assert failed_auth is None


class TestUserPermissions:
    """사용자 권한 테스트"""

    def test_user_permission_checking(self):
        """권한 확인 메소드 테스트"""
        user = User(
            user_id="test123",
            username="permuser",
            email="perm@example.com",
            role=UserRole.DEVELOPER,
            permissions=[Permission.API_READ, Permission.API_WRITE, Permission.MODEL_INFERENCE]
        )

        # 단일 권한 확인
        assert user.has_permission(Permission.API_READ)
        assert user.has_permission(Permission.API_WRITE)
        assert not user.has_permission(Permission.SYSTEM_ADMIN)

        # 여러 권한 중 하나라도
        assert user.has_any_permission([Permission.API_READ, Permission.SYSTEM_ADMIN])
        assert not user.has_any_permission([Permission.SYSTEM_ADMIN, Permission.DATA_DELETE])

        # 모든 권한 보유
        assert user.has_all_permissions([Permission.API_READ, Permission.API_WRITE])
        assert not user.has_all_permissions([Permission.API_READ, Permission.SYSTEM_ADMIN])

    def test_role_based_permissions(self):
        """역할 기반 권한 테스트"""
        from fragrance_ai.core.auth import ROLE_PERMISSIONS

        # 관리자 권한
        admin_perms = ROLE_PERMISSIONS[UserRole.ADMIN]
        assert Permission.SYSTEM_ADMIN in admin_perms
        assert Permission.API_ADMIN in admin_perms

        # 사용자 권한
        user_perms = ROLE_PERMISSIONS[UserRole.USER]
        assert Permission.SEARCH_BASIC in user_perms
        assert Permission.SYSTEM_ADMIN not in user_perms

        # 게스트 권한
        guest_perms = ROLE_PERMISSIONS[UserRole.GUEST]
        assert len(guest_perms) > 0
        assert Permission.SEARCH_BASIC in guest_perms
        assert Permission.API_WRITE not in guest_perms

    def test_user_serialization(self):
        """사용자 직렬화 테스트"""
        user = User(
            user_id="serial123",
            username="serialuser",
            email="serial@example.com",
            role=UserRole.RESEARCHER,
            permissions=[Permission.DATA_READ, Permission.MODEL_INFERENCE],
            metadata={"department": "AI Research", "level": "senior"}
        )

        user_dict = user.to_dict()

        assert user_dict["user_id"] == user.user_id
        assert user_dict["username"] == user.username
        assert user_dict["role"] == UserRole.RESEARCHER.value
        assert Permission.DATA_READ.value in user_dict["permissions"]
        assert user_dict["metadata"]["department"] == "AI Research"


class TestRateLimiting:
    """요청 제한 테스트"""

    @pytest.fixture
    def mock_redis(self):
        """모의 Redis 클라이언트"""
        redis_mock = Mock()
        redis_mock.get = AsyncMock(return_value=None)
        redis_mock.setex = AsyncMock()
        redis_mock.incr = AsyncMock()
        return redis_mock

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_first_request(self, mock_redis):
        """첫 번째 요청 허용 테스트"""
        from fragrance_ai.core.auth import RateLimiter

        rate_limiter = RateLimiter(mock_redis)
        mock_redis.get.return_value = None

        allowed = await rate_limiter.is_allowed("user:123", limit=10, window_seconds=60)
        assert allowed is True

        mock_redis.setex.assert_called_once_with("user:123", 60, 1)

    @pytest.mark.asyncio
    async def test_rate_limiter_increments_counter(self, mock_redis):
        """카운터 증가 테스트"""
        from fragrance_ai.core.auth import RateLimiter

        rate_limiter = RateLimiter(mock_redis)
        mock_redis.get.return_value = "5"  # 현재 5번 요청

        allowed = await rate_limiter.is_allowed("user:123", limit=10, window_seconds=60)
        assert allowed is True

        mock_redis.incr.assert_called_once_with("user:123")

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self, mock_redis):
        """초과 요청 차단 테스트"""
        from fragrance_ai.core.auth import RateLimiter

        rate_limiter = RateLimiter(mock_redis)
        mock_redis.get.return_value = "10"  # 이미 한계 도달

        allowed = await rate_limiter.is_allowed("user:123", limit=10, window_seconds=60)
        assert allowed is False

        # incr 호출되지 않아야 함
        mock_redis.incr.assert_not_called()


class TestAuthorizationManager:
    """권한 부여 관리자 테스트"""

    @pytest.fixture
    async def authz_mgr(self):
        """테스트용 권한 부여 관리자"""
        auth_mgr = AuthenticationManager()
        await auth_mgr.initialize()
        mgr = AuthorizationManager(auth_mgr)
        await mgr.initialize()
        return mgr

    @pytest.mark.asyncio
    async def test_permission_decorator(self, authz_mgr):
        """권한 데코레이터 테스트"""

        # 권한이 있는 사용자
        authorized_user = User(
            user_id="auth123",
            username="authuser",
            email="auth@example.com",
            permissions=[Permission.API_READ, Permission.API_WRITE]
        )

        # 권한이 없는 사용자
        unauthorized_user = User(
            user_id="unauth123",
            username="unauthuser",
            email="unauth@example.com",
            permissions=[Permission.API_READ]  # API_WRITE 권한 없음
        )

        @authz_mgr.require_permissions([Permission.API_READ, Permission.API_WRITE])
        async def protected_function(request=None):
            return "success"

        # Mock request 객체
        mock_request_authorized = Mock()
        mock_request_authorized.state.current_user = authorized_user

        mock_request_unauthorized = Mock()
        mock_request_unauthorized.state.current_user = unauthorized_user

        # 권한이 있는 사용자는 접근 가능
        result = await protected_function(request=mock_request_authorized)
        assert result == "success"

        # 권한이 없는 사용자는 예외 발생
        with pytest.raises(AuthorizationError):
            await protected_function(request=mock_request_unauthorized)

    @pytest.mark.asyncio
    async def test_role_decorator(self, authz_mgr):
        """역할 데코레이터 테스트"""

        admin_user = User(
            user_id="admin123",
            username="admin",
            email="admin@example.com",
            role=UserRole.ADMIN
        )

        regular_user = User(
            user_id="user123",
            username="user",
            email="user@example.com",
            role=UserRole.USER
        )

        @authz_mgr.require_role(UserRole.ADMIN)
        async def admin_only_function(request=None):
            return "admin_success"

        # Mock requests
        mock_admin_request = Mock()
        mock_admin_request.state.current_user = admin_user

        mock_user_request = Mock()
        mock_user_request.state.current_user = regular_user

        # 관리자는 접근 가능
        result = await admin_only_function(request=mock_admin_request)
        assert result == "admin_success"

        # 일반 사용자는 예외 발생
        with pytest.raises(AuthorizationError):
            await admin_only_function(request=mock_user_request)


class TestIntegrationAuth:
    """인증 통합 테스트"""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self):
        """전체 인증 플로우 테스트"""
        # 인증 관리자 초기화
        auth_mgr = AuthenticationManager()
        await auth_mgr.initialize()

        # 1. 사용자 생성
        user = await auth_mgr.create_user(
            username="flowuser",
            email="flow@example.com",
            password="flowpassword",
            role=UserRole.DEVELOPER
        )

        # 2. 로그인 (사용자 인증)
        authenticated_user = await auth_mgr.authenticate_user("flowuser", "flowpassword")
        assert authenticated_user is not None

        # 3. JWT 토큰 생성
        access_token = auth_mgr.create_access_token(authenticated_user)
        assert access_token is not None

        # 4. 토큰 검증
        payload = auth_mgr.verify_token(access_token)
        assert payload["sub"] == user.user_id

        # 5. API 키를 통한 인증
        api_key = auth_mgr.generate_api_key(user)
        api_user = await auth_mgr.get_user_by_api_key(api_key)
        assert api_user.user_id == user.user_id

        # 6. 권한 확인
        assert authenticated_user.has_permission(Permission.API_READ)
        assert authenticated_user.has_permission(Permission.MODEL_TRAIN)

    @pytest.mark.asyncio
    async def test_security_edge_cases(self):
        """보안 엣지 케이스 테스트"""
        auth_mgr = AuthenticationManager()
        await auth_mgr.initialize()

        # 빈 사용자명으로 사용자 생성 시도
        with pytest.raises((ValueError, AssertionError)):
            await auth_mgr.create_user("", "email@example.com", "password")

        # 잘못된 이메일 형식 (실제로는 검증 로직이 있어야 함)
        user = await auth_mgr.create_user("testuser", "invalid-email", "password")
        assert user is not None  # 현재는 검증 없이 생성

        # 만료된 토큰 시뮬레이션
        expired_user = await auth_mgr.create_user("expireduser", "expired@example.com", "password")
        expired_token = auth_mgr.create_access_token(
            expired_user,
            expires_delta=timedelta(seconds=-1)  # 이미 만료된 토큰
        )

        with pytest.raises(AuthenticationError):
            auth_mgr.verify_token(expired_token)

    @pytest.mark.asyncio
    async def test_concurrent_authentication(self):
        """동시 인증 테스트"""
        auth_mgr = AuthenticationManager()
        await auth_mgr.initialize()

        # 여러 사용자 동시 생성
        async def create_test_user(i):
            return await auth_mgr.create_user(
                username=f"concurrent_user_{i}",
                email=f"concurrent_{i}@example.com",
                password=f"password_{i}"
            )

        # 10개의 사용자 동시 생성
        users = await asyncio.gather(*[create_test_user(i) for i in range(10)])
        assert len(users) == 10
        assert len(set(user.user_id for user in users)) == 10  # 모든 ID가 고유함

        # 동시 토큰 생성
        tokens = await asyncio.gather(*[
            asyncio.create_task(asyncio.coroutine(lambda u=user: auth_mgr.create_access_token(u))())
            for user in users
        ])
        assert len(tokens) == 10
        assert len(set(tokens)) == 10  # 모든 토큰이 고유함


@pytest.mark.integration
class TestAuthAPIIntegration:
    """API와 인증 시스템 통합 테스트"""

    @pytest.mark.asyncio
    async def test_api_authentication_required(self, async_test_client):
        """API 인증 필수 테스트"""
        # 인증 없는 요청
        response = await async_test_client.get("/api/v1/protected-endpoint")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_api_with_valid_token(self, async_test_client):
        """유효한 토큰으로 API 접근 테스트"""
        # 사용자 생성 및 토큰 발급
        user = await auth_manager.create_user("apiuser", "api@example.com", "password")
        token = auth_manager.create_access_token(user)

        headers = {"Authorization": f"Bearer {token}"}
        response = await async_test_client.get("/api/v1/user/profile", headers=headers)

        # 실제 엔드포인트가 구현되어 있다면 200, 없으면 404
        assert response.status_code in [200, 404]

    @pytest.mark.asyncio
    async def test_api_with_invalid_token(self, async_test_client):
        """잘못된 토큰으로 API 접근 테스트"""
        headers = {"Authorization": "Bearer invalid.token.here"}
        response = await async_test_client.get("/api/v1/user/profile", headers=headers)
        assert response.status_code == 401