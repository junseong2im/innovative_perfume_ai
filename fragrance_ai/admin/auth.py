"""
관리자 인증 및 권한 관리
"""

from typing import Dict, Optional, Any, List
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta

from ..core.config import settings
from ..core.exceptions_advanced import AuthenticationError, AuthorizationError, ErrorCode

security = HTTPBearer()

# 관리자 역할 정의
ADMIN_ROLES = {
    "super_admin": {
        "name": "Super Administrator",
        "permissions": ["*"]  # 모든 권한
    },
    "admin": {
        "name": "Administrator",
        "permissions": [
            "dashboard.view",
            "users.view", "users.edit", "users.suspend",
            "system.monitor", "system.logs",
            "billing.view", "billing.edit"
        ]
    },
    "moderator": {
        "name": "Moderator",
        "permissions": [
            "dashboard.view",
            "users.view", "users.suspend",
            "system.monitor"
        ]
    },
    "analyst": {
        "name": "Analyst",
        "permissions": [
            "dashboard.view",
            "users.view",
            "system.monitor"
        ]
    }
}

async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """현재 관리자 사용자 정보 조회"""

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate admin credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])

        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception

        # 관리자 권한 확인
        user_role = payload.get("role")
        if user_role not in ADMIN_ROLES:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient admin privileges"
            )

        return {
            "id": user_id,
            "email": payload.get("email"),
            "role": user_role,
            "permissions": ADMIN_ROLES[user_role]["permissions"]
        }

    except JWTError:
        raise credentials_exception

def require_admin_access(
    current_admin: Dict[str, Any] = Depends(get_current_admin)
) -> Dict[str, Any]:
    """관리자 접근 권한 필요"""
    return current_admin

def require_permission(permission: str):
    """특정 권한 필요"""
    def permission_checker(
        current_admin: Dict[str, Any] = Depends(get_current_admin)
    ) -> Dict[str, Any]:
        admin_permissions = current_admin.get("permissions", [])

        # super_admin은 모든 권한
        if "*" in admin_permissions:
            return current_admin

        # 특정 권한 확인
        if permission not in admin_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission}"
            )

        return current_admin

    return permission_checker