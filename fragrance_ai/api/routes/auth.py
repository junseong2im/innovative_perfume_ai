"""
인증 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.orm import Session

from fragrance_ai.api.schemas.recipe_schemas import (
    UserAuthRequest,
    UserAuthResponse,
    UserRole,
    AdminStats
)
from fragrance_ai.api.middleware.auth_middleware import AuthService, get_current_admin
from fragrance_ai.database.base import get_db

router = APIRouter(prefix="/auth", tags=["인증"])

@router.post("/login", response_model=UserAuthResponse)
async def login(request: UserAuthRequest):
    """사용자 로그인"""
    user = AuthService.authenticate_user(request.username, request.password)

    if not user:
        return UserAuthResponse(
            success=False,
            message="잘못된 사용자명 또는 비밀번호입니다"
        )

    access_token = AuthService.create_access_token(
        username=user["username"],
        role=user["role"]
    )

    return UserAuthResponse(
        success=True,
        access_token=access_token,
        user_role=user["role"],
        message=f"{user['role'].value} 권한으로 로그인되었습니다"
    )

@router.get("/admin/stats", response_model=AdminStats)
async def get_admin_stats(
    current_admin: dict = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    """관리자용 통계 정보"""
    try:
        # 향료 노트 수 조회
        from fragrance_ai.repositories.fragrance_note_repository import FragranceNoteRepository
        note_repo = FragranceNoteRepository(db)
        total_notes = note_repo.count_all()

        # 더미 데이터 (실제로는 데이터베이스에서 조회)
        stats = AdminStats(
            total_recipes=45,
            total_notes=total_notes,
            popular_families={
                "floral": 15,
                "citrus": 12,
                "woody": 10,
                "oriental": 8
            },
            recent_activity=[
                {
                    "action": "recipe_generated",
                    "user": "customer01",
                    "timestamp": "2024-01-15 14:30:00",
                    "details": "로맨틱 플로럴 향수 생성"
                },
                {
                    "action": "note_added",
                    "user": current_admin["username"],
                    "timestamp": "2024-01-15 13:45:00",
                    "details": "제주 감귤 노트 추가"
                }
            ],
            cost_analysis={
                "average_cost_per_recipe": 85.50,
                "most_expensive_ingredient": 2.80,
                "total_inventory_value": 12450.00
            }
        )

        return stats

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/verify")
async def verify_token(current_user: dict = Depends(get_current_admin)):
    """토큰 검증"""
    return {
        "valid": True,
        "user": current_user["username"],
        "role": current_user["role"]
    }