"""
인증 및 권한 관리 미들웨어
"""

from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import jwt
from datetime import datetime, timedelta
import bcrypt
from sqlalchemy.orm import Session

from fragrance_ai.core.config import get_settings
from fragrance_ai.database.base import get_db
from fragrance_ai.api.schemas.recipe_schemas import UserRole

settings = get_settings()
security = HTTPBearer()

# 테스트용 사용자 데이터 (실제로는 데이터베이스에서)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": bcrypt.hashpw("admin123!".encode('utf-8'), bcrypt.gensalt()),
        "role": UserRole.ADMIN
    },
    "customer": {
        "username": "customer",
        "hashed_password": bcrypt.hashpw("customer123".encode('utf-8'), bcrypt.gensalt()),
        "role": UserRole.CUSTOMER
    }
}

class AuthService:
    """인증 서비스"""

    @staticmethod
    def verify_password(plain_password: str, hashed_password: bytes) -> bool:
        """비밀번호 검증"""
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

    @staticmethod
    def create_access_token(username: str, role: UserRole) -> str:
        """JWT 토큰 생성"""
        expire = datetime.utcnow() + timedelta(hours=24)
        payload = {
            "sub": username,
            "role": role.value,
            "exp": expire
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")

    @staticmethod
    def verify_token(token: str) -> dict:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            return payload
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 토큰입니다",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def authenticate_user(username: str, password: str) -> Optional[dict]:
        """사용자 인증"""
        user = USERS_DB.get(username)
        if not user:
            return None

        if not AuthService.verify_password(password, user["hashed_password"]):
            return None

        return {
            "username": user["username"],
            "role": user["role"]
        }

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """현재 인증된 사용자 정보 반환"""
    token = credentials.credentials
    payload = AuthService.verify_token(token)

    username = payload.get("sub")
    role = payload.get("role")

    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="사용자 정보를 찾을 수 없습니다",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "username": username,
        "role": UserRole(role)
    }

async def get_current_admin(current_user: dict = Depends(get_current_user)):
    """관리자 권한 확인"""
    if current_user["role"] not in [UserRole.ADMIN, UserRole.SUPER_ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="관리자 권한이 필요합니다"
        )
    return current_user

async def get_optional_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    """선택적 사용자 인증 (토큰 없어도 허용)"""
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        payload = AuthService.verify_token(token)
        username = payload.get("sub")
        role = payload.get("role")

        return {
            "username": username,
            "role": UserRole(role)
        }
    except HTTPException:
        return None