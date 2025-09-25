"""
Secure admin authentication with server-side sessions
"""

from typing import Dict, Optional, Any
from fastapi import Depends, HTTPException, Request, Response, status, Cookie
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import bcrypt
from datetime import datetime
import re

from ..core.session_manager import get_session_manager
from ..core.config import settings
from ..core.exceptions_advanced import AuthenticationError, AuthorizationError


# Admin user store (in production, this should be in database)
ADMIN_USERS = {
    "admin@fragrance.ai": {
        "password_hash": bcrypt.hashpw(settings.admin_password.encode(), bcrypt.gensalt()),
        "role": "super_admin",
        "name": "Super Administrator",
        "mfa_secret": None  # For future MFA implementation
    }
}

# Role permissions
ADMIN_ROLES = {
    "super_admin": {
        "name": "Super Administrator",
        "permissions": ["*"]
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
    }
}


class SecureAdminAuth:
    """Secure admin authentication handler"""

    def __init__(self):
        self.session_manager = get_session_manager()
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
        self.failed_attempts = {}  # Track failed login attempts

    def _get_client_info(self, request: Request) -> tuple[str, str]:
        """Extract client IP and user agent"""
        # Get real IP from proxy headers if available
        ip_address = (
            request.headers.get("X-Real-IP") or
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.client.host
        )
        user_agent = request.headers.get("User-Agent", "")
        return ip_address, user_agent

    def _check_rate_limit(self, email: str, ip: str) -> None:
        """Check login rate limiting"""
        key = f"{email}:{ip}"
        now = datetime.utcnow()

        if key in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[key]
            time_diff = (now - last_attempt).total_seconds()

            # Reset if lockout period has passed
            if time_diff > self.lockout_duration:
                del self.failed_attempts[key]
            elif attempts >= self.max_login_attempts:
                remaining = self.lockout_duration - time_diff
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Account locked. Try again in {int(remaining)} seconds."
                )

    def _record_failed_attempt(self, email: str, ip: str) -> None:
        """Record failed login attempt"""
        key = f"{email}:{ip}"
        now = datetime.utcnow()

        if key in self.failed_attempts:
            attempts, _ = self.failed_attempts[key]
            self.failed_attempts[key] = (attempts + 1, now)
        else:
            self.failed_attempts[key] = (1, now)

    def _clear_failed_attempts(self, email: str, ip: str) -> None:
        """Clear failed attempts after successful login"""
        key = f"{email}:{ip}"
        if key in self.failed_attempts:
            del self.failed_attempts[key]

    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements"""
        if len(password) < 12:
            return False
        if not re.search(r"[A-Z]", password):
            return False
        if not re.search(r"[a-z]", password):
            return False
        if not re.search(r"\d", password):
            return False
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            return False
        return True

    async def login(
        self,
        request: Request,
        response: Response,
        email: str,
        password: str
    ) -> Dict[str, Any]:
        """
        Authenticate admin and create secure session
        """
        ip_address, user_agent = self._get_client_info(request)

        # Check rate limiting
        self._check_rate_limit(email, ip_address)

        # Validate user exists
        user_data = ADMIN_USERS.get(email)
        if not user_data:
            self._record_failed_attempt(email, ip_address)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Verify password
        if not bcrypt.checkpw(password.encode(), user_data["password_hash"]):
            self._record_failed_attempt(email, ip_address)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Clear failed attempts
        self._clear_failed_attempts(email, ip_address)

        # Create session
        session_id = self.session_manager.create_admin_session(
            user_id=email,
            email=email,
            role=user_data["role"],
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Set secure cookie
        response.set_cookie(
            key="admin_session_id",
            value=session_id,
            max_age=1800,  # 30 minutes
            httponly=True,  # Prevent XSS
            secure=True,  # HTTPS only
            samesite="strict",  # CSRF protection
            path="/api/admin"  # Restrict to admin routes
        )

        # Rotate session ID for security
        new_session_id = self.session_manager.rotate_session_id(session_id)
        if new_session_id:
            response.set_cookie(
                key="admin_session_id",
                value=new_session_id,
                max_age=1800,
                httponly=True,
                secure=True,
                samesite="strict",
                path="/api/admin"
            )

        return {
            "success": True,
            "user": {
                "email": email,
                "role": user_data["role"],
                "name": user_data["name"]
            }
        }

    async def logout(
        self,
        request: Request,
        response: Response,
        session_id: Optional[str] = Cookie(None, alias="admin_session_id")
    ) -> Dict[str, Any]:
        """Logout admin and invalidate session"""
        if session_id:
            self.session_manager.invalidate_session(session_id)

        # Clear cookie
        response.delete_cookie(
            key="admin_session_id",
            path="/api/admin"
        )

        return {"success": True, "message": "Logged out successfully"}

    async def get_current_admin(
        self,
        request: Request,
        session_id: Optional[str] = Cookie(None, alias="admin_session_id")
    ) -> Dict[str, Any]:
        """Get current authenticated admin from session"""
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No active session"
            )

        ip_address, user_agent = self._get_client_info(request)

        # Validate session
        session_data = self.session_manager.validate_admin_session(
            session_id,
            ip_address,
            user_agent
        )

        if not session_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session"
            )

        # Get role permissions
        role = session_data.get("role")
        permissions = ADMIN_ROLES.get(role, {}).get("permissions", [])

        return {
            "id": session_data["user_id"],
            "email": session_data["email"],
            "role": role,
            "permissions": permissions,
            "session_created": session_data["created_at"],
            "last_activity": session_data["last_activity"]
        }

    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        async def permission_checker(
            admin: Dict[str, Any] = Depends(self.get_current_admin)
        ) -> Dict[str, Any]:
            admin_permissions = admin.get("permissions", [])

            # Super admin has all permissions
            if "*" in admin_permissions:
                return admin

            # Check specific permission
            if permission not in admin_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {permission}"
                )

            return admin

        return permission_checker

    async def change_password(
        self,
        request: Request,
        current_admin: Dict[str, Any],
        old_password: str,
        new_password: str
    ) -> Dict[str, Any]:
        """Change admin password"""
        email = current_admin["email"]
        user_data = ADMIN_USERS.get(email)

        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify old password
        if not bcrypt.checkpw(old_password.encode(), user_data["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid current password"
            )

        # Validate new password strength
        if not self._validate_password_strength(new_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password does not meet security requirements"
            )

        # Update password
        ADMIN_USERS[email]["password_hash"] = bcrypt.hashpw(
            new_password.encode(),
            bcrypt.gensalt()
        )

        # Invalidate all sessions for security
        self.session_manager.invalidate_all_user_sessions(email)

        return {"success": True, "message": "Password changed successfully"}


# Singleton instance
_admin_auth = None

def get_admin_auth() -> SecureAdminAuth:
    """Get singleton admin auth instance"""
    global _admin_auth
    if _admin_auth is None:
        _admin_auth = SecureAdminAuth()
    return _admin_auth