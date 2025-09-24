"""
고객용 공개 레시피 API 엔드포인트
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional
from sqlalchemy.orm import Session

from fragrance_ai.api.schemas.recipe_schemas import (
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    RecipeListResponse,
    FragranceProfileCustomer,
    UserRole
)
from fragrance_ai.api.middleware.auth_middleware import get_optional_user
from fragrance_ai.services.generation_service import GenerationService
from fragrance_ai.api.dependencies import get_generation_service
from fragrance_ai.database.base import get_db
import uuid
import time

router = APIRouter(prefix="/public/recipes", tags=["공개 레시피"])

@router.post("/generate", response_model=RecipeGenerationResponse)
async def generate_public_recipe(
    request: RecipeGenerationRequest,
    generation_service: GenerationService = Depends(get_generation_service),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    공개 레시피 생성 - 고객은 향수 설명만, 관리자는 상세 레시피도 함께 제공
    """
    try:
        generation_id = str(uuid.uuid4())

        # 사용자 역할 확인
        user_role = current_user["role"] if current_user else UserRole.CUSTOMER
        is_admin = user_role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]

        # 실제 생성 서비스 호출
        request_data = {
            "fragrance_family": request.fragrance_family,
            "mood": request.mood,
            "intensity": request.intensity,
            "gender": request.gender or "unisex",
            "season": request.season or "spring",
            "korean_region": request.korean_region,
            "korean_season": request.korean_season,
            "traditional_element": request.traditional_element,
            "budget_range": request.budget_range or "medium",
            "complexity": request.complexity or "medium",
            "unique_request": request.unique_request
        }

        # 생성 서비스를 통해 레시피 생성
        generation_result = await generation_service.generate_recipe(request_data)
        recipe_data = generation_result.get("recipe", {})

        # 고객용 향수 프로필 생성 (항상 제공)
        customer_info = FragranceProfileCustomer(
            name=recipe_data.get("name", "맞춤형 향수"),
            description=_generate_customer_description(request, recipe_data),
            story=_generate_customer_story(request, recipe_data),
            fragrance_family=request.fragrance_family,
            mood=request.mood,
            season=request.season or "사계절",
            gender=request.gender or "남녀공용",
            intensity=request.intensity,
            longevity=_get_longevity_description(request.intensity),
            sillage=_get_sillage_description(request.mood),
            top_notes_description=_get_notes_description("top", request),
            middle_notes_description=_get_notes_description("middle", request),
            base_notes_description=_get_notes_description("base", request),
            occasion=_get_occasion_suggestions(request),
            time_of_day=_get_time_suggestions(request),
            korean_inspiration=request.traditional_element,
            regional_character=request.korean_region
        )

        # 관리자용 상세 레시피 (관리자만)
        admin_recipe = None
        if is_admin:
            from fragrance_ai.api.schemas.recipe_schemas import (
                FragranceRecipeAdmin,
                RecipeIngredientAdmin,
                FragranceNoteAdmin
            )

            # 상세 레시피 생성 (실제로는 데이터베이스에서 조회)
            admin_recipe = _generate_admin_recipe(generation_id, request, recipe_data, current_user)

        # 응답 메시지
        if is_admin:
            message = "관리자용 상세 레시피와 고객용 정보가 모두 생성되었습니다."
        else:
            message = "고객님만을 위한 특별한 향수 정보가 준비되었습니다."

        return RecipeGenerationResponse(
            success=True,
            message=message,
            generation_id=generation_id,
            customer_info=customer_info,
            admin_recipe=admin_recipe
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"레시피 생성 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("/popular", response_model=RecipeListResponse)
async def get_popular_recipes(
    page: int = 1,
    page_size: int = 10,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """인기 레시피 목록 조회"""
    try:
        # 사용자 역할 확인
        user_role = current_user["role"] if current_user else UserRole.CUSTOMER
        is_admin = user_role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]

        # 더미 데이터 (실제로는 데이터베이스에서 조회)
        popular_recipes = [
            {
                "id": "recipe_001",
                "name": "봄날의 로맨스",
                "description": "벚꽃이 피는 봄날을 연상시키는 로맨틱한 플로럴 향수",
                "fragrance_family": "floral",
                "mood": "romantic",
                "popularity_score": 95,
                "created_at": "2024-01-15T10:30:00Z",
                "admin_only": not is_admin  # 관리자가 아니면 상세 정보 숨김
            },
            {
                "id": "recipe_002",
                "name": "제주 감귤 바람",
                "description": "제주의 싱그러운 감귤향이 담긴 상큼한 시트러스 향수",
                "fragrance_family": "citrus",
                "mood": "fresh",
                "popularity_score": 88,
                "created_at": "2024-01-14T14:20:00Z",
                "admin_only": not is_admin
            }
        ]

        # 페이지네이션 적용
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_recipes = popular_recipes[start_idx:end_idx]

        return RecipeListResponse(
            recipes=paginated_recipes,
            total_count=len(popular_recipes),
            page=page,
            page_size=page_size
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"인기 레시피 조회 중 오류가 발생했습니다: {str(e)}"
        )

def _generate_customer_description(request: RecipeGenerationRequest, recipe_data: dict) -> str:
    """고객용 향수 설명 생성"""
    base_desc = f"{request.mood} 무드의 {request.fragrance_family} 계열 향수입니다."

    if request.korean_region:
        base_desc += f" {request.korean_region} 지역의 특색을 담아냈습니다."

    if request.traditional_element:
        base_desc += f" {request.traditional_element}의 전통적 아름다움이 느껴집니다."

    return base_desc

def _generate_customer_story(request: RecipeGenerationRequest, recipe_data: dict) -> str:
    """고객용 향수 스토리 생성"""
    stories = {
        "romantic": "사랑하는 사람과의 특별한 순간을 위해 탄생한 향수입니다.",
        "fresh": "상쾌한 아침 공기처럼 활력을 주는 향수입니다.",
        "elegant": "우아하고 세련된 당신의 품격을 드러내는 향수입니다.",
        "mysterious": "신비로운 매력을 발산하는 독특한 향수입니다."
    }
    return stories.get(request.mood, "당신만을 위해 특별히 조제된 향수입니다.")

def _get_longevity_description(intensity: str) -> str:
    """강도에 따른 지속성 설명"""
    longevity_map = {
        "light": "4-6시간",
        "moderate": "6-8시간",
        "strong": "8-12시간"
    }
    return longevity_map.get(intensity, "6-8시간")

def _get_sillage_description(mood: str) -> str:
    """무드에 따른 확산성 설명"""
    sillage_map = {
        "romantic": "은은하게 퍼짐",
        "fresh": "적당히 퍼짐",
        "elegant": "우아하게 퍼짐",
        "mysterious": "강하게 퍼짐"
    }
    return sillage_map.get(mood, "적당히 퍼짐")

def _get_notes_description(note_type: str, request: RecipeGenerationRequest) -> str:
    """노트 타입별 설명 생성"""
    descriptions = {
        "top": {
            "floral": "싱그러운 꽃봉오리의 첫 향",
            "citrus": "상큼한 과즙의 톡톡 터지는 향",
            "woody": "신선한 나무 껍질의 깔끔한 향"
        },
        "middle": {
            "floral": "만개한 꽃들의 풍성한 향",
            "citrus": "달콤한 과육의 부드러운 향",
            "woody": "따뜻한 나무의 포근한 향"
        },
        "base": {
            "floral": "꽃잎이 스며든 깊은 향",
            "citrus": "과일껍질의 깊이 있는 향",
            "woody": "나무 심재의 웅장한 향"
        }
    }

    family_desc = descriptions.get(note_type, {})
    return family_desc.get(request.fragrance_family, f"{note_type} 노트의 조화로운 향")

def _get_occasion_suggestions(request: RecipeGenerationRequest) -> List[str]:
    """상황별 사용 추천"""
    occasion_map = {
        "romantic": ["데이트", "기념일", "로맨틱 디너"],
        "fresh": ["일상", "업무", "스포츠"],
        "elegant": ["파티", "비즈니스", "특별한 모임"],
        "mysterious": ["나이트라이프", "특별한 밤", "파티"]
    }
    return occasion_map.get(request.mood, ["일상", "특별한 날"])

def _get_time_suggestions(request: RecipeGenerationRequest) -> List[str]:
    """시간대별 사용 추천"""
    time_map = {
        "romantic": ["저녁", "밤"],
        "fresh": ["아침", "오후"],
        "elegant": ["오후", "저녁"],
        "mysterious": ["밤", "늦은 밤"]
    }
    return time_map.get(request.mood, ["오후", "저녁"])

def _generate_admin_recipe(generation_id: str, request: RecipeGenerationRequest, recipe_data: dict, current_user: dict):
    """관리자용 상세 레시피 생성"""
    from fragrance_ai.api.schemas.recipe_schemas import (
        FragranceRecipeAdmin,
        RecipeIngredientAdmin,
        FragranceNoteAdmin
    )

    # 더미 상세 레시피 데이터 (실제로는 데이터베이스에서 조회)
    ingredients = [
        RecipeIngredientAdmin(
            note=FragranceNoteAdmin(
                name="Bulgarian Rose",
                name_korean="불가리안 로즈",
                fragrance_family=request.fragrance_family,
                note_type="middle",
                description="진한 장미 향",
                description_korean="불가리아산 프리미엄 장미 에센스",
                origin="불가리아",
                extraction_method="증류법",
                intensity=8.5,
                longevity=7.0,
                sillage=6.5,
                price_per_ml=2.50,
                supplier="Rose Valley Bulgaria",
                grade="A+",
                mood_tags=[request.mood],
                season_tags=[request.season] if request.season else [],
                gender_tags=[request.gender] if request.gender else []
            ),
            percentage=25.0,
            ml_amount=8.0,
            role="heart",
            order=1,
            notes="핵심 노트로 사용"
        )
    ]

    return FragranceRecipeAdmin(
        recipe_id=generation_id,
        name=recipe_data.get("name", "맞춤형 향수"),
        description="고객 요청에 따른 맞춤형 향수 레시피",
        ingredients=ingredients,
        total_volume=30.0,
        alcohol_percentage=70.0,
        maturation_time=21,
        total_cost=95.50,
        cost_per_ml=3.18,
        suggested_price=280.0,
        mixing_instructions=[
            "1. 베이스 노트부터 차례로 첨가",
            "2. 각 단계마다 5분간 숙성",
            "3. 최종 혼합 후 21일간 숙성"
        ],
        storage_instructions="직사광선을 피하고 15-20°C에서 보관",
        safety_notes=[
            "피부 접촉 시 즉시 세척",
            "환기가 잘 되는 곳에서 작업"
        ],
        expected_color="연한 황금색",
        expected_clarity="투명",
        created_at=time.time(),
        created_by=current_user["username"] if current_user else "system",
        version="1.0"
    )