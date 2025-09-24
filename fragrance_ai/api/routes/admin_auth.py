"""
Admin Authentication Routes
관리자 전용 세션 기반 인증 라우터
"""

from fastapi import APIRouter, HTTPException, Depends, Response, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import secrets

from ...core.security import (
    security_manager,
    AccessLevel,
    TokenType
)
from ...core.logging_config import get_logger
from ..dependencies import get_current_admin_user

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/auth/admin",
    tags=["Admin Authentication"]
)


class AdminLoginRequest(BaseModel):
    """관리자 로그인 요청"""
    email: EmailStr
    password: str


class AdminLoginResponse(BaseModel):
    """관리자 로그인 응답"""
    user_id: str
    email: str
    role: str
    csrf_token: str
    message: str


class AdminSessionResponse(BaseModel):
    """관리자 세션 정보"""
    user_id: str
    email: str
    username: str
    role: str
    login_time: str
    is_active: bool


# 임시 관리자 계정 데이터 (프로덕션에서는 DB 사용)
ADMIN_ACCOUNTS = {
    "admin@deulsoom.com": {
        "password_hash": "$2b$12$KIXxPfJHgFqQlW1ZQqhFH.0Y1Z5TtRHXw5xwfvJQBmF6JHtGq9Zte",  # "admin123!"
        "role": AccessLevel.ADMIN,
        "user_id": "admin_001",
        "username": "시스템 관리자"
    },
    "super@deulsoom.com": {
        "password_hash": "$2b$12$KIXxPfJHgFqQlW1ZQqhFH.0Y1Z5TtRHXw5xwfvJQBmF6JHtGq9Zte",  # "admin123!"
        "role": AccessLevel.SUPER_ADMIN,
        "user_id": "super_001",
        "username": "최고 관리자"
    }
}


@router.post("/login", response_model=AdminLoginResponse)
async def admin_login(
    request: Request,
    response: Response,
    login_data: AdminLoginRequest
):
    """
    관리자 로그인

    - 이메일과 비밀번호로 인증
    - 성공 시 HttpOnly 쿠키로 세션 ID 발급
    - CSRF 토큰 반환
    """
    try:
        # IP 주소 추출
        client_ip = request.client.host if request.client else "unknown"

        # Rate limiting 확인
        is_allowed, rate_info = security_manager.check_rate_limit(
            identifier=f"admin_login:{client_ip}",
            limit_per_minute=5,  # 관리자 로그인은 엄격하게 제한
            access_level=AccessLevel.PUBLIC
        )

        if not is_allowed:
            logger.warning(f"Admin login rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"너무 많은 로그인 시도입니다. {rate_info.get('reset_time_minute', 60)}초 후 다시 시도해주세요."
            )

        # 계정 확인
        if login_data.email not in ADMIN_ACCOUNTS:
            # 보안상 구체적인 에러 메시지 제공하지 않음
            logger.warning(f"Failed admin login attempt for email: {login_data.email} from IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다"
            )

        admin = ADMIN_ACCOUNTS[login_data.email]

        # 비밀번호 검증 (실제로는 bcrypt 사용)
        # 여기서는 데모용으로 간단히 처리
        if not security_manager.encryption_manager.verify_password(
            login_data.password,
            admin["password_hash"]
        ):
            logger.warning(f"Invalid password for admin: {login_data.email} from IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="이메일 또는 비밀번호가 올바르지 않습니다"
            )

        # 세션 생성
        user_agent = request.headers.get("User-Agent", "Unknown")
        session_id = security_manager.create_session(
            user_id=admin["user_id"],
            ip_address=client_ip,
            user_agent=user_agent,
            additional_data={
                "email": login_data.email,
                "role": admin["role"].value,
                "username": admin["username"]
            }
        )

        # CSRF 토큰 생성
        csrf_token = security_manager.generate_csrf_token(session_id)

        # JWT 토큰 생성 (세션과 별도로, 추가 검증용)
        access_token = security_manager.create_jwt_token(
            user_id=admin["user_id"],
            token_type=TokenType.ACCESS,
            additional_claims={
                "email": login_data.email,
                "access_level": admin["role"].value,
                "session_id": session_id,
                "ip_address": client_ip
            }
        )

        # HttpOnly 쿠키로 세션 설정
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,  # JavaScript에서 접근 불가
            secure=True,     # HTTPS에서만 전송 (프로덕션)
            samesite="strict",  # CSRF 공격 방어
            max_age=7200,    # 2시간
            path="/"
        )

        # 액세스 토큰도 HttpOnly 쿠키로 설정 (이중 보안)
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=900,  # 15분
            path="/"
        )

        # 보안 이벤트 로깅
        security_manager.log_security_event(
            event_type="admin_login_success",
            severity="medium",
            source_ip=client_ip,
            user_id=admin["user_id"],
            details={
                "email": login_data.email,
                "role": admin["role"].value
            }
        )

        logger.info(f"Admin login successful: {login_data.email} from IP: {client_ip}")

        return AdminLoginResponse(
            user_id=admin["user_id"],
            email=login_data.email,
            role=admin["role"].value,
            csrf_token=csrf_token,
            message="로그인 성공"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="로그인 처리 중 오류가 발생했습니다"
        )


@router.get("/session", response_model=AdminSessionResponse)
async def get_admin_session(
    request: Request
):
    """
    현재 관리자 세션 확인

    - HttpOnly 쿠키에서 세션 ID 확인
    - 세션 유효성 검증
    - 세션 정보 반환
    """
    try:
        # 쿠키에서 세션 ID 추출
        session_id = request.cookies.get("session_id")
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="세션이 없습니다"
            )

        # 세션 정보 조회
        session_data = security_manager.get_session(session_id)
        if not session_data or not session_data.get("is_active"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 세션입니다"
            )

        # IP 주소 검증 (선택적)
        client_ip = request.client.host if request.client else "unknown"
        if session_data.get("ip_address") != client_ip:
            logger.warning(f"IP mismatch for session {session_id}: {session_data.get('ip_address')} != {client_ip}")
            # 정책에 따라 차단 또는 경고만

        return AdminSessionResponse(
            user_id=session_data["user_id"],
            email=session_data.get("email", ""),
            username=session_data.get("username", ""),
            role=session_data.get("role", "admin"),
            login_time=session_data.get("created_at", ""),
            is_active=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session check error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="세션 확인 중 오류가 발생했습니다"
        )


@router.post("/logout")
async def admin_logout(
    request: Request,
    response: Response
):
    """
    관리자 로그아웃

    - 세션 무효화
    - 쿠키 삭제
    """
    try:
        # 쿠키에서 세션 ID 추출
        session_id = request.cookies.get("session_id")

        if session_id:
            # 세션 무효화
            security_manager.session_manager.invalidate_session(session_id)

            # 보안 이벤트 로깅
            client_ip = request.client.host if request.client else "unknown"
            security_manager.log_security_event(
                event_type="admin_logout",
                severity="low",
                source_ip=client_ip,
                user_id=None,
                details={
                    "session_id": session_id[:8] + "..."  # 일부만 로깅
                }
            )

        # 쿠키 삭제
        response.delete_cookie(key="session_id", path="/")
        response.delete_cookie(key="access_token", path="/")

        return {"message": "로그아웃 되었습니다"}

    except Exception as e:
        logger.error(f"Logout error: {e}")
        # 로그아웃은 항상 성공으로 처리
        return {"message": "로그아웃 되었습니다"}


@router.post("/refresh")
async def refresh_admin_token(
    request: Request,
    response: Response,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
):
    """
    관리자 토큰 갱신

    - 현재 세션 확인
    - 새 액세스 토큰 발급
    """
    try:
        # 새 액세스 토큰 생성
        new_token = security_manager.create_jwt_token(
            user_id=current_admin["user_id"],
            token_type=TokenType.ACCESS,
            additional_claims={
                "email": current_admin.get("email"),
                "access_level": current_admin.get("access_level"),
                "session_id": current_admin.get("session_id"),
                "ip_address": current_admin.get("ip_address")
            }
        )

        # 새 토큰으로 쿠키 업데이트
        response.set_cookie(
            key="access_token",
            value=new_token,
            httponly=True,
            secure=True,
            samesite="strict",
            max_age=900,  # 15분
            path="/"
        )

        return {"message": "토큰이 갱신되었습니다"}

    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="토큰 갱신 중 오류가 발생했습니다"
        )


@router.get("/verify")
async def verify_admin_access(
    current_admin: Dict[str, Any] = Depends(get_current_admin_user)
):
    """
    관리자 권한 확인

    - 현재 인증 상태 확인
    - 권한 레벨 반환
    """
    return {
        "authenticated": True,
        "user_id": current_admin["user_id"],
        "access_level": current_admin["access_level"],
        "permissions": current_admin.get("permissions", [])
    }