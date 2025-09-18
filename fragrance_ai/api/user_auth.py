"""
사용자 인증 및 권한 관리 API 엔드포인트
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Form, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, validator
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import secrets
import uuid

from ..database import get_db
from ..core.logging_config import get_logger
from ..core.exceptions_advanced import AuthenticationError, AuthorizationError
from ..models.user import User, UserStatus, UserRole as UserRoleEnum, AuthProvider
from ..services.auth_service import auth_service, UserRegistrationData, AuthResult
from ..api.dependencies import get_current_user, require_admin

logger = get_logger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Pydantic 모델들
class UserRegistrationRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    username: Optional[str] = None

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('비밀번호는 최소 8자 이상이어야 합니다')
        return v

    @validator('username')
    def validate_username(cls, v):
        if v and len(v) < 3:
            raise ValueError('사용자명은 최소 3자 이상이어야 합니다')
        return v

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    remember_me: bool = False

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]

class UserResponse(BaseModel):
    id: str
    email: str
    username: Optional[str]
    full_name: Optional[str]
    role: str
    status: str
    is_email_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime]

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('새 비밀번호는 최소 8자 이상이어야 합니다')
        return v

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirmRequest(BaseModel):
    token: str
    new_password: str

class APIKeyRequest(BaseModel):
    name: str
    scopes: List[str] = []
    expires_in_days: Optional[int] = None

class APIKeyResponse(BaseModel):
    key_id: str
    name: str
    key: str  # 생성 시에만 반환
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]

class UserPermissionRequest(BaseModel):
    permission_code: str
    granted: bool = True
    expires_at: Optional[datetime] = None

# 인증 엔드포인트들

@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    request: UserRegistrationRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """사용자 등록"""
    try:
        # auth_service 초기화 확인
        if not hasattr(auth_service, 'redis_client'):
            await auth_service.initialize()

        # 등록 데이터 준비
        registration_data = UserRegistrationData(
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            username=request.username
        )

        # 사용자 등록
        result = await auth_service.register_user(registration_data, db)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.message
            )

        logger.info(f"User registered successfully: {request.email}")

        # 백그라운드에서 이메일 인증 메일 발송 (구현 필요)
        # background_tasks.add_task(send_verification_email, result.user.email)

        return {
            "message": "사용자가 성공적으로 등록되었습니다. 이메일 인증을 확인해주세요.",
            "user_id": str(result.user.id),
            "email": result.user.email
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="사용자 등록 중 오류가 발생했습니다"
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(
    request: LoginRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """사용자 로그인"""
    try:
        # auth_service 초기화 확인
        if not hasattr(auth_service, 'redis_client'):
            await auth_service.initialize()

        # 사용자 인증
        result = await auth_service.authenticate_user(
            email=request.email,
            password=request.password,
            request=http_request,
            db=db
        )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.message,
                headers={"WWW-Authenticate": "Bearer"},
            )

        # 사용자 정보
        user_info = {
            "id": str(result.user.id),
            "email": result.user.email,
            "username": result.user.username,
            "full_name": result.user.full_name,
            "role": result.user.role.value,
            "status": result.user.status.value,
            "is_email_verified": result.user.is_email_verified
        }

        logger.info(f"User logged in successfully: {request.email}")

        return TokenResponse(
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=3600,  # 1시간
            user=user_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="로그인 처리 중 오류가 발생했습니다"
        )

@router.post("/login/oauth2", response_model=TokenResponse)
async def oauth_login(
    provider: str,
    oauth_data: Dict[str, Any],
    http_request: Request,
    db: Session = Depends(get_db)
):
    """OAuth 로그인"""
    try:
        # auth_service 초기화 확인
        if not hasattr(auth_service, 'redis_client'):
            await auth_service.initialize()

        # 지원되는 OAuth 제공자 확인
        try:
            auth_provider = AuthProvider(provider.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"지원되지 않는 OAuth 제공자입니다: {provider}"
            )

        # OAuth 인증
        result = await auth_service.authenticate_oauth_user(
            provider=auth_provider,
            oauth_data=oauth_data,
            request=http_request,
            db=db
        )

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=result.message
            )

        # 사용자 정보
        user_info = {
            "id": str(result.user.id),
            "email": result.user.email,
            "username": result.user.username,
            "full_name": result.user.full_name,
            "role": result.user.role.value,
            "status": result.user.status.value,
            "is_email_verified": result.user.is_email_verified
        }

        logger.info(f"OAuth user logged in successfully: {result.user.email} via {provider}")

        return TokenResponse(
            access_token=result.access_token,
            refresh_token=result.refresh_token,
            expires_in=3600,
            user=user_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAuth 로그인 처리 중 오류가 발생했습니다"
        )

@router.post("/refresh")
async def refresh_token(
    refresh_token: str = Form(...),
    db: Session = Depends(get_db)
):
    """토큰 갱신"""
    try:
        # 리프레시 토큰으로 세션 찾기
        from ..models.user import UserSession
        session = db.query(UserSession).filter(
            UserSession.refresh_token == refresh_token,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow()
        ).first()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="유효하지 않은 리프레시 토큰입니다"
            )

        # 새 액세스 토큰 생성
        user = session.user
        access_token = auth_service._create_access_token({
            "sub": str(user.id),
            "email": user.email,
            "role": user.role.value
        })

        # 세션 업데이트
        session.last_used_at = datetime.utcnow()
        db.commit()

        logger.info(f"Token refreshed for user: {user.email}")

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="토큰 갱신 중 오류가 발생했습니다"
        )

@router.post("/logout")
async def logout_user(
    refresh_token: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """사용자 로그아웃"""
    try:
        from ..models.user import UserSession

        # 모든 세션 비활성화 또는 특정 세션만 비활성화
        sessions_query = db.query(UserSession).filter(
            UserSession.user_id == uuid.UUID(current_user["user_id"]),
            UserSession.is_active == True
        )

        if refresh_token:
            sessions_query = sessions_query.filter(UserSession.refresh_token == refresh_token)

        sessions = sessions_query.all()
        for session in sessions:
            session.is_active = False

        db.commit()

        logger.info(f"User logged out: {current_user.get('email')}")

        return {"message": "성공적으로 로그아웃되었습니다"}

    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="로그아웃 처리 중 오류가 발생했습니다"
        )

# 사용자 계정 관리

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """현재 사용자 정보 조회"""
    try:
        user = db.query(User).filter(User.id == uuid.UUID(current_user["user_id"])).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다"
            )

        return UserResponse(
            id=str(user.id),
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            role=user.role.value,
            status=user.status.value,
            is_email_verified=user.is_email_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="사용자 정보 조회 중 오류가 발생했습니다"
        )

@router.put("/me")
async def update_user_profile(
    full_name: Optional[str] = Form(None),
    username: Optional[str] = Form(None),
    bio: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """사용자 프로필 업데이트"""
    try:
        user = db.query(User).filter(User.id == uuid.UUID(current_user["user_id"])).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다"
            )

        # 사용자명 중복 확인
        if username and username != user.username:
            existing_user = db.query(User).filter(User.username == username).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="이미 사용 중인 사용자명입니다"
                )

        # 프로필 업데이트
        if full_name is not None:
            user.full_name = full_name
        if username is not None:
            user.username = username
        if bio is not None:
            user.bio = bio
        if location is not None:
            user.location = location
        if website is not None:
            user.website = website

        user.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f"User profile updated: {user.email}")

        return {"message": "프로필이 성공적으로 업데이트되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="프로필 업데이트 중 오류가 발생했습니다"
        )

@router.post("/change-password")
async def change_password(
    request: PasswordChangeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """비밀번호 변경"""
    try:
        user = db.query(User).filter(User.id == uuid.UUID(current_user["user_id"])).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다"
            )

        # 현재 비밀번호 확인
        if not auth_service._verify_password(request.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="현재 비밀번호가 올바르지 않습니다"
            )

        # 새 비밀번호 설정
        user.hashed_password = auth_service._hash_password(request.new_password)
        user.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f"Password changed for user: {user.email}")

        return {"message": "비밀번호가 성공적으로 변경되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="비밀번호 변경 중 오류가 발생했습니다"
        )

# API 키 관리

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """API 키 생성"""
    try:
        user = db.query(User).filter(User.id == uuid.UUID(current_user["user_id"])).first()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다"
            )

        # API 키 생성
        full_key, api_key = await auth_service.create_api_key(
            user=user,
            name=request.name,
            scopes=request.scopes,
            db=db,
            expires_in_days=request.expires_in_days
        )

        logger.info(f"API key created for user: {user.email}")

        return APIKeyResponse(
            key_id=api_key.key_id,
            name=api_key.name,
            key=full_key,
            scopes=api_key.scopes,
            created_at=api_key.created_at,
            expires_at=api_key.expires_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API 키 생성 중 오류가 발생했습니다"
        )

@router.get("/api-keys")
async def list_api_keys(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """사용자 API 키 목록 조회"""
    try:
        from ..models.user import ApiKey

        api_keys = db.query(ApiKey).filter(
            ApiKey.user_id == uuid.UUID(current_user["user_id"]),
            ApiKey.is_active == True
        ).all()

        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "scopes": key.scopes,
                "created_at": key.created_at,
                "last_used_at": key.last_used_at,
                "expires_at": key.expires_at,
                "usage_count": key.usage_count
            }
            for key in api_keys
        ]

    except Exception as e:
        logger.error(f"API key listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API 키 목록 조회 중 오류가 발생했습니다"
        )

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """API 키 비활성화"""
    try:
        from ..models.user import ApiKey

        api_key = db.query(ApiKey).filter(
            ApiKey.key_id == key_id,
            ApiKey.user_id == uuid.UUID(current_user["user_id"])
        ).first()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API 키를 찾을 수 없습니다"
            )

        api_key.is_active = False
        db.commit()

        logger.info(f"API key revoked: {key_id}")

        return {"message": "API 키가 성공적으로 비활성화되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API key revocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API 키 비활성화 중 오류가 발생했습니다"
        )

# 관리자 전용 엔드포인트들

@router.get("/admin/users")
async def list_users(
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None,
    current_admin: Dict[str, Any] = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """사용자 목록 조회 (관리자 전용)"""
    try:
        query = db.query(User)

        # 상태 필터링
        if status_filter:
            try:
                status_enum = UserStatus(status_filter)
                query = query.filter(User.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"유효하지 않은 상태 값입니다: {status_filter}"
                )

        users = query.offset(skip).limit(limit).all()
        total = query.count()

        return {
            "users": [
                {
                    "id": str(user.id),
                    "email": user.email,
                    "username": user.username,
                    "full_name": user.full_name,
                    "role": user.role.value,
                    "status": user.status.value,
                    "is_email_verified": user.is_email_verified,
                    "created_at": user.created_at,
                    "last_login_at": user.last_login_at
                }
                for user in users
            ],
            "total": total,
            "skip": skip,
            "limit": limit
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User listing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="사용자 목록 조회 중 오류가 발생했습니다"
        )

@router.put("/admin/users/{user_id}/status")
async def update_user_status(
    user_id: str,
    new_status: str,
    current_admin: Dict[str, Any] = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """사용자 상태 업데이트 (관리자 전용)"""
    try:
        # 상태 검증
        try:
            status_enum = UserStatus(new_status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"유효하지 않은 상태 값입니다: {new_status}"
            )

        # 사용자 찾기
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="사용자를 찾을 수 없습니다"
            )

        # 상태 업데이트
        user.status = status_enum
        user.updated_at = datetime.utcnow()
        db.commit()

        logger.info(f"User status updated by admin {current_admin.get('email')}: {user.email} -> {new_status}")

        return {"message": f"사용자 상태가 {new_status}로 변경되었습니다"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User status update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="사용자 상태 업데이트 중 오류가 발생했습니다"
        )