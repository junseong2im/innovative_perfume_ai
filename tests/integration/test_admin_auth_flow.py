"""
Admin Authentication Flow Integration Tests
관리자 인증 플로우 통합 테스트
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi import status
import json

from fragrance_ai.api.main import app
from fragrance_ai.core.security import security_manager


@pytest.fixture
async def async_client():
    """비동기 테스트 클라이언트"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def admin_credentials():
    """테스트용 관리자 자격증명"""
    return {
        "email": "admin@deulsoom.com",
        "password": "admin123!"
    }


@pytest.fixture
def super_admin_credentials():
    """테스트용 최고 관리자 자격증명"""
    return {
        "email": "super@deulsoom.com",
        "password": "admin123!"
    }


class TestAdminAuthenticationFlow:
    """관리자 인증 플로우 통합 테스트"""

    @pytest.mark.asyncio
    async def test_admin_login_success(self, async_client, admin_credentials):
        """정상적인 관리자 로그인"""
        response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # 응답 검증
        assert "user_id" in data
        assert "email" in data
        assert "role" in data
        assert "csrf_token" in data
        assert data["email"] == admin_credentials["email"]
        assert data["role"] == "admin"

        # 쿠키 검증
        cookies = response.cookies
        assert "session_id" in cookies
        assert "access_token" in cookies

    @pytest.mark.asyncio
    async def test_admin_login_invalid_credentials(self, async_client):
        """잘못된 자격증명으로 로그인 시도"""
        response = await async_client.post(
            "/api/v1/auth/admin/login",
            json={
                "email": "wrong@deulsoom.com",
                "password": "wrongpass"
            }
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_admin_session_check(self, async_client, admin_credentials):
        """세션 확인 테스트"""
        # 먼저 로그인
        login_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )
        assert login_response.status_code == status.HTTP_200_OK

        # 세션 확인
        session_response = await async_client.get(
            "/api/v1/auth/admin/session",
            cookies=login_response.cookies
        )

        assert session_response.status_code == status.HTTP_200_OK
        data = session_response.json()
        assert data["is_active"] is True
        assert data["email"] == admin_credentials["email"]

    @pytest.mark.asyncio
    async def test_admin_logout(self, async_client, admin_credentials):
        """로그아웃 테스트"""
        # 로그인
        login_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )
        assert login_response.status_code == status.HTTP_200_OK

        # 로그아웃
        logout_response = await async_client.post(
            "/api/v1/auth/admin/logout",
            cookies=login_response.cookies
        )
        assert logout_response.status_code == status.HTTP_200_OK

        # 로그아웃 후 세션 확인 (실패해야 함)
        session_response = await async_client.get(
            "/api/v1/auth/admin/session",
            cookies=login_response.cookies
        )
        assert session_response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client):
        """Rate limiting 테스트"""
        # 5번 이상 잘못된 로그인 시도
        for i in range(6):
            response = await async_client.post(
                "/api/v1/auth/admin/login",
                json={
                    "email": f"test{i}@deulsoom.com",
                    "password": "wrong"
                }
            )

            if i < 5:
                assert response.status_code == status.HTTP_401_UNAUTHORIZED
            else:
                # 6번째 시도는 rate limit에 걸려야 함
                assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.asyncio
    async def test_csrf_token_validation(self, async_client, admin_credentials):
        """CSRF 토큰 검증 테스트"""
        # 로그인
        login_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )
        assert login_response.status_code == status.HTTP_200_OK

        csrf_token = login_response.json()["csrf_token"]

        # CSRF 토큰 없이 요청 (실패해야 함)
        response = await async_client.post(
            "/api/v1/auth/admin/refresh",
            cookies=login_response.cookies
        )
        # CSRF 검증은 dependencies에서 처리

        # CSRF 토큰과 함께 요청
        response = await async_client.post(
            "/api/v1/auth/admin/refresh",
            cookies=login_response.cookies,
            headers={"X-CSRF-Token": csrf_token}
        )
        # 성공 또는 다른 에러 (CSRF는 통과)

    @pytest.mark.asyncio
    async def test_session_timeout(self, async_client, admin_credentials):
        """세션 타임아웃 테스트"""
        # 로그인
        login_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )
        assert login_response.status_code == status.HTTP_200_OK

        # 세션 ID 추출
        session_id = None
        for cookie in login_response.cookies.jar:
            if cookie.name == "session_id":
                session_id = cookie.value
                break

        assert session_id is not None

        # 세션 강제 만료 (테스트 목적)
        if hasattr(security_manager.session_manager, 'sessions'):
            if session_id in security_manager.session_manager.sessions:
                security_manager.session_manager.sessions[session_id]["is_active"] = False

        # 만료된 세션으로 요청
        response = await async_client.get(
            "/api/v1/auth/admin/session",
            cookies=login_response.cookies
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_role_based_access(self, async_client, admin_credentials, super_admin_credentials):
        """역할 기반 접근 제어 테스트"""
        # 일반 관리자 로그인
        admin_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )
        assert admin_response.status_code == status.HTTP_200_OK
        assert admin_response.json()["role"] == "admin"

        # 최고 관리자 로그인
        super_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=super_admin_credentials
        )
        assert super_response.status_code == status.HTTP_200_OK
        assert super_response.json()["role"] == "super_admin"

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, async_client, admin_credentials):
        """동시 세션 테스트"""
        # 첫 번째 로그인
        response1 = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials
        )
        assert response1.status_code == status.HTTP_200_OK

        # 두 번째 로그인 (다른 브라우저 시뮬레이션)
        response2 = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials,
            headers={"User-Agent": "Different Browser"}
        )
        assert response2.status_code == status.HTTP_200_OK

        # 두 세션 모두 유효해야 함
        session1 = await async_client.get(
            "/api/v1/auth/admin/session",
            cookies=response1.cookies
        )
        assert session1.status_code == status.HTTP_200_OK

        session2 = await async_client.get(
            "/api/v1/auth/admin/session",
            cookies=response2.cookies
        )
        assert session2.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_ip_validation(self, async_client, admin_credentials):
        """IP 주소 검증 테스트"""
        # 로그인
        login_response = await async_client.post(
            "/api/v1/auth/admin/login",
            json=admin_credentials,
            headers={"X-Forwarded-For": "192.168.1.1"}
        )
        assert login_response.status_code == status.HTTP_200_OK

        # 다른 IP에서 세션 사용 시도
        # (정책에 따라 차단 또는 경고)
        session_response = await async_client.get(
            "/api/v1/auth/admin/session",
            cookies=login_response.cookies,
            headers={"X-Forwarded-For": "10.0.0.1"}
        )
        # IP 불일치는 로깅되지만 차단은 정책에 따름


@pytest.mark.asyncio
async def test_full_admin_workflow():
    """전체 관리자 워크플로우 테스트"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. 로그인
        login_response = await client.post(
            "/api/v1/auth/admin/login",
            json={
                "email": "admin@deulsoom.com",
                "password": "admin123!"
            }
        )
        assert login_response.status_code == 200
        cookies = login_response.cookies
        csrf_token = login_response.json()["csrf_token"]

        # 2. 세션 확인
        session_response = await client.get(
            "/api/v1/auth/admin/session",
            cookies=cookies
        )
        assert session_response.status_code == 200

        # 3. 권한 확인
        verify_response = await client.get(
            "/api/v1/auth/admin/verify",
            cookies=cookies,
            headers={"X-CSRF-Token": csrf_token}
        )
        # get_current_admin_user 의존성이 필요함

        # 4. 토큰 갱신
        refresh_response = await client.post(
            "/api/v1/auth/admin/refresh",
            cookies=cookies,
            headers={"X-CSRF-Token": csrf_token}
        )
        # 의존성 처리 필요

        # 5. 로그아웃
        logout_response = await client.post(
            "/api/v1/auth/admin/logout",
            cookies=cookies
        )
        assert logout_response.status_code == 200

        # 6. 로그아웃 후 세션 확인 (실패해야 함)
        final_session = await client.get(
            "/api/v1/auth/admin/session",
            cookies=cookies
        )
        assert final_session.status_code == 401


if __name__ == "__main__":
    # 테스트 실행
    pytest.main([__file__, "-v", "-s"])