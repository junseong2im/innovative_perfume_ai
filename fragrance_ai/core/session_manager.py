"""
Secure server-side session management
"""

import secrets
import json
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from redis import Redis
import hashlib
import hmac

from .config import settings


class SessionManager:
    """
    Secure session management with Redis backend
    Implements OWASP best practices for session security
    """

    def __init__(self):
        self.redis_client = Redis.from_url(
            settings.redis_url,
            decode_responses=True
        )
        self.session_ttl = 3600  # 1 hour default
        self.admin_session_ttl = 1800  # 30 minutes for admin sessions

    def _generate_session_id(self) -> str:
        """Generate cryptographically secure session ID"""
        return secrets.token_urlsafe(32)

    def _hash_session_id(self, session_id: str) -> str:
        """Hash session ID for storage"""
        return hashlib.sha256(session_id.encode()).hexdigest()

    def create_admin_session(
        self,
        user_id: str,
        email: str,
        role: str,
        ip_address: str,
        user_agent: str
    ) -> str:
        """Create secure admin session"""
        session_id = self._generate_session_id()
        hashed_id = self._hash_session_id(session_id)

        session_data = {
            "user_id": user_id,
            "email": email,
            "role": role,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "is_admin": True
        }

        # Store in Redis with TTL
        self.redis_client.setex(
            f"admin_session:{hashed_id}",
            self.admin_session_ttl,
            json.dumps(session_data)
        )

        # Track active admin sessions
        self.redis_client.sadd(f"active_admin_sessions:{user_id}", hashed_id)

        return session_id

    def validate_admin_session(
        self,
        session_id: str,
        ip_address: str,
        user_agent: str
    ) -> Optional[Dict[str, Any]]:
        """
        Validate admin session with security checks
        """
        if not session_id:
            return None

        hashed_id = self._hash_session_id(session_id)
        session_data = self.redis_client.get(f"admin_session:{hashed_id}")

        if not session_data:
            return None

        data = json.loads(session_data)

        # Verify IP address and user agent haven't changed (session fixation protection)
        if data.get("ip_address") != ip_address:
            # Log suspicious activity
            self._log_security_event("IP_MISMATCH", data["user_id"], ip_address)
            self.invalidate_session(session_id)
            return None

        if data.get("user_agent") != user_agent:
            # Log suspicious activity
            self._log_security_event("USER_AGENT_MISMATCH", data["user_id"], user_agent)
            self.invalidate_session(session_id)
            return None

        # Update last activity
        data["last_activity"] = datetime.utcnow().isoformat()
        self.redis_client.setex(
            f"admin_session:{hashed_id}",
            self.admin_session_ttl,
            json.dumps(data)
        )

        return data

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        hashed_id = self._hash_session_id(session_id)

        # Get session data before deletion
        session_data = self.redis_client.get(f"admin_session:{hashed_id}")
        if session_data:
            data = json.loads(session_data)
            user_id = data.get("user_id")

            # Remove from active sessions set
            if user_id:
                self.redis_client.srem(f"active_admin_sessions:{user_id}", hashed_id)

        # Delete session
        return self.redis_client.delete(f"admin_session:{hashed_id}") > 0

    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a user (useful for password change, logout everywhere)"""
        session_ids = self.redis_client.smembers(f"active_admin_sessions:{user_id}")
        count = 0

        for hashed_id in session_ids:
            if self.redis_client.delete(f"admin_session:{hashed_id}"):
                count += 1

        self.redis_client.delete(f"active_admin_sessions:{user_id}")
        return count

    def rotate_session_id(self, old_session_id: str) -> Optional[str]:
        """
        Rotate session ID (prevent session fixation attacks)
        Called after successful login
        """
        hashed_old_id = self._hash_session_id(old_session_id)
        session_data = self.redis_client.get(f"admin_session:{hashed_old_id}")

        if not session_data:
            return None

        # Generate new session ID
        new_session_id = self._generate_session_id()
        hashed_new_id = self._hash_session_id(new_session_id)

        # Transfer session data
        data = json.loads(session_data)
        data["rotated_at"] = datetime.utcnow().isoformat()

        # Create new session
        self.redis_client.setex(
            f"admin_session:{hashed_new_id}",
            self.admin_session_ttl,
            json.dumps(data)
        )

        # Update active sessions set
        user_id = data.get("user_id")
        if user_id:
            self.redis_client.srem(f"active_admin_sessions:{user_id}", hashed_old_id)
            self.redis_client.sadd(f"active_admin_sessions:{user_id}", hashed_new_id)

        # Delete old session
        self.redis_client.delete(f"admin_session:{hashed_old_id}")

        return new_session_id

    def get_active_admin_count(self) -> int:
        """Get count of active admin sessions"""
        pattern = "admin_session:*"
        return len(list(self.redis_client.scan_iter(pattern)))

    def _log_security_event(self, event_type: str, user_id: str, details: str):
        """Log security events for audit"""
        event = {
            "type": event_type,
            "user_id": user_id,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Store in Redis list with TTL
        key = f"security_events:{datetime.utcnow().strftime('%Y%m%d')}"
        self.redis_client.lpush(key, json.dumps(event))
        self.redis_client.expire(key, 86400 * 30)  # Keep for 30 days

    def cleanup_expired_sessions(self):
        """Cleanup expired sessions (called by scheduled job)"""
        # Redis automatically handles expiry with TTL
        # This method is for additional cleanup if needed
        pass


# Singleton instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get singleton session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager