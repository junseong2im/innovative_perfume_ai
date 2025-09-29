from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from typing import List, Optional, Dict, Any
import logging
import time
import uuid
import asyncio
from datetime import datetime
from pydantic import BaseModel, Field

from ..schemas import (
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    FragranceRecipe,
    QualityScores,
    FragranceComposition
)
from ..schemas.recipe_schemas import (
    RecipeGenerationRequest as NewRecipeRequest,
    RecipeGenerationResponse as NewRecipeResponse,
    FragranceProfileCustomer,
    FragranceRecipeAdmin,
    RecipeIngredientCustomer,
    RecipeIngredientAdmin,
    FragranceNoteCustomer,
    FragranceNoteAdmin,
    UserRole
)
from ...services.generation_service import GenerationService
from ..dependencies import get_generation_service
from ..middleware.auth_middleware import get_optional_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recipe", response_model=NewRecipeResponse)
async def generate_recipe(
    request: NewRecipeRequest,
    background_tasks: BackgroundTasks,
    generation_service: GenerationService = Depends(get_generation_service),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """향수 레시피 생성 - 역할 기반 접근 제어"""
    try:
        start_time = time.time()
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
        
        # 생성 결과에서 레시피 정보 추출
        recipe_data = generation_result.get("recipe", {})

        # 관리자용 상세 레시피 생성 (관리자만)
        admin_recipe = None
        if is_admin:
            # 향료 재료 상세 정보 생성
            ingredients = []
            for note_type, note_list in recipe_data.get("notes", {}).items():
                if isinstance(note_list, list):
                    for i, note_name in enumerate(note_list):
                        ingredients.append(RecipeIngredientAdmin(
                            note=FragranceNoteAdmin(
                                name=note_name,
                                name_korean=f"{note_name} (한국어)",
                                fragrance_family=recipe_data.get("fragrance_family", request.fragrance_family),
                                note_type=note_type,
                                description=f"{note_name}의 독특한 향",
                                description_korean=f"{note_name}의 한국어 설명",
                                origin="프랑스/이탈리아",
                                extraction_method="증류법",
                                intensity=7.5 + (hash(note_name) % 3) * 0.5,
                                longevity=6.0 + (hash(note_name) % 4) * 0.5,
                                sillage=5.5 + (hash(note_name) % 5) * 0.5,
                                price_per_ml=0.50 + (hash(note_name) % 20) * 0.1,
                                supplier="Premium Fragrances Ltd",
                                grade="A+",
                                mood_tags=[request.mood],
                                season_tags=[request.season] if request.season else [],
                                gender_tags=[request.gender] if request.gender else []
                            ),
                            percentage=15.0 + (hash(note_name) % 10),
                            ml_amount=5.0 + (hash(note_name) % 15),
                            role=note_type,
                            order=i + 1,
                            notes=f"{note_name} 사용 시 주의사항"
                        ))

            admin_recipe = FragranceRecipeAdmin(
                recipe_id=generation_id,
                name=recipe_data.get("name", "생성된 향수"),
                description=recipe_data.get("description", "AI가 생성한 향수 레시피"),
                ingredients=ingredients,
                total_volume=100.0,
                alcohol_percentage=70.0,
                maturation_time=30,
                total_cost=125.50,
                cost_per_ml=1.26,
                suggested_price=450.0,
                mixing_instructions=[
                    "1. 알코올에 베이스 노트부터 차례로 첨가",
                    "2. 중간 노트 추가 후 10분간 숙성",
                    "3. 탑 노트를 마지막에 추가",
                    "4. 30일간 어두운 곳에서 숙성"
                ],
                storage_instructions="직사광선을 피하고 서늘한 곳에 보관",
                safety_notes=[
                    "피부에 직접 접촉 시 즉시 세척",
                    "임산부 사용 금지",
                    "어린이 손에 닿지 않는 곳에 보관"
                ],
                expected_color="연한 황금색",
                expected_clarity="투명",
                created_at=time.time(),
                created_by=current_user["username"] if current_user else "system",
                version="1.0"
            )

        # 고객용 향수 프로필 생성
        customer_info = FragranceProfileCustomer(
            name=recipe_data.get("name", "생성된 향수"),
            description=recipe_data.get("description", "AI가 창조한 독특한 향수"),
            story=recipe_data.get("story", "이 향수는 당신만을 위해 특별히 제조된 향수입니다. 섬세한 향료들이 조화롭게 어우러져 독특하고 매력적인 향을 만들어냅니다."),
            fragrance_family=request.fragrance_family,
            mood=request.mood,
            season=request.season or "사계절",
            gender=request.gender or "남녀공용",
            intensity=request.intensity,
            longevity="6-8시간",
            sillage="적당함",
            top_notes_description="상쾌하고 활기찬 첫 인상을 주는 탑 노트",
            middle_notes_description="풍부하고 조화로운 중심 향을 이루는 미들 노트",
            base_notes_description="깊이 있고 지속적인 여운을 남기는 베이스 노트",
            occasion=["데이트", "파티", "일상"],
            time_of_day=["오후", "저녁"],
            korean_inspiration=request.traditional_element,
            regional_character=request.korean_region
        )
        
        # 응답 메시지 생성
        if is_admin:
            message = "관리자용 상세 레시피가 생성되었습니다. 비용 정보와 제조 지침을 확인하세요."
        else:
            message = "고객님만을 위한 특별한 향수가 완성되었습니다. 향수 정보를 확인해보세요."

        return NewRecipeResponse(
            success=True,
            message=message,
            generation_id=generation_id,
            customer_info=customer_info,
            admin_recipe=admin_recipe
        )
        
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchGenerationResponse)
async def batch_generate_recipes(
    request: BatchGenerationRequest,
    background_tasks: BackgroundTasks,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """배치 레시피 생성"""
    try:
        start_time = time.time()
        
        if len(request.prompts) > 20:
            raise HTTPException(status_code=400, detail="배치 크기가 너무 큽니다 (최대 20개)")
        
        # 실제 배치 생성 서비스 사용
        batch_requests = []
        for prompt in request.prompts:
            batch_requests.append({
                "prompt": prompt,
                "recipe_type": str(request.recipe_type.value) if hasattr(request.recipe_type, 'value') else str(request.recipe_type),
                "fragrance_family": getattr(request, 'fragrance_family', 'floral'),
                "mood": getattr(request, 'mood', 'romantic'),
                "intensity": getattr(request, 'intensity', 'moderate'),
                "gender": getattr(request, 'gender', 'unisex'),
                "season": getattr(request, 'season', 'spring')
            })
        
        # 배치 생성 서비스 호출
        batch_results = await generation_service.batch_generate(batch_requests)
        
        # 결과 변환
        recipes = []
        failed_count = 0
        total_quality = 0.0
        
        for i, result in enumerate(batch_results.get("results", [])):
            if result.get("success", False):
                recipe_data = result.get("recipe", {})
                recipe = FragranceRecipe(
                    name=recipe_data.get("name", f"생성된 향수 #{i+1}"),
                    concept=recipe_data.get("concept", recipe_data.get("description", "AI가 생성한 향수")),
                    mood=recipe_data.get("mood", batch_requests[i]["mood"]),
                    season=recipe_data.get("season", batch_requests[i]["season"]),
                    composition=FragranceComposition(
                        top_notes={"percentage": 30, "ingredients": recipe_data.get("notes", {}).get("top", [])},
                        heart_notes={"percentage": 40, "ingredients": recipe_data.get("notes", {}).get("middle", [])},
                        base_notes={"percentage": 30, "ingredients": recipe_data.get("notes", {}).get("base", [])}
                    ),
                    longevity=recipe_data.get("longevity", "6-8시간"),
                    sillage=recipe_data.get("sillage", "보통"),
                    story=recipe_data.get("story", recipe_data.get("description", "AI가 창조한 독특한 향수 레시피")),
                    raw_text=str(recipe_data),
                    generated_at=time.time(),
                    recipe_type=request.recipe_type
                )
                recipes.append(recipe)
                total_quality += result.get("quality_score", 75.0)
            else:
                failed_count += 1
        
        average_quality = (total_quality / len(recipes)) / 100.0 if recipes else 0.0
        
        return BatchGenerationResponse(
            recipes=recipes,
            total_recipes=len(recipes),
            average_quality=average_quality,
            generation_time=time.time() - start_time,
            failed_generations=failed_count
        )
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/custom-blend")
async def generate_custom_blend(
    notes: List[str],
    mood: str,
    intensity: int = 5,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """커스텀 블렌드 생성"""
    try:
        if len(notes) > 15:
            raise HTTPException(status_code=400, detail="노트는 최대 15개까지 가능합니다")
        
        # 실제 커스텀 블렌드 생성
        custom_request = {
            "notes": notes,
            "mood": mood,
            "intensity": intensity,
            "generation_type": "custom_blend"
        }
        
        # 생성 서비스를 통해 커스텀 블렌드 생성
        blend_result = await generation_service.generate_recipe(custom_request)
        
        # 비율 계산 (균등 분배 기본, AI 추천 사용 가능)
        total_notes = len(notes)
        base_ratio = 100 / total_notes
        recommended_ratios = {}
        
        # 노트 강도에 따른 비율 조정
        for i, note in enumerate(notes):
            # 강도에 따른 비율 조정 (특정 노트는 더 많이, 일부는 적게)
            if i < total_notes // 3:  # 첫 번째 1/3은 더 많이
                ratio = base_ratio * (1.0 + (intensity / 20))
            elif i > (total_notes * 2) // 3:  # 마지막 1/3은 적게
                ratio = base_ratio * (1.0 - (intensity / 30))
            else:  # 중간은 기본
                ratio = base_ratio
            recommended_ratios[note] = round(ratio, 2)
        
        # 비율 정규화 (100%가 되도록)
        total_ratio = sum(recommended_ratios.values())
        for note in recommended_ratios:
            recommended_ratios[note] = round((recommended_ratios[note] / total_ratio) * 100, 2)
        
        # 블렌딩 지침 생성
        blending_instructions = blend_result.get("recipe", {}).get("blending_notes", 
            f"지정된 {len(notes)}개 노트를 {mood} 무드로 블렌딩합니다. 강도 {intensity}/10으로 조절하세요.")
        
        # 예상 비용 계산 (노트 개수와 강도에 따른 기본 비용)
        base_cost_per_note = 15.0  # 기본 노트당 비용
        intensity_multiplier = 1.0 + (intensity / 20)  # 강도에 따른 배수
        estimated_cost = len(notes) * base_cost_per_note * intensity_multiplier
        
        return {
            "blend_id": str(uuid.uuid4()),
            "notes": notes,
            "mood": mood,
            "intensity": intensity,
            "recommended_ratios": recommended_ratios,
            "blending_instructions": blending_instructions,
            "estimated_cost": round(estimated_cost, 2),
            "generated_recipe": blend_result.get("recipe", {})
        }
        
    except Exception as e:
        logger.error(f"Custom blend generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mood-board")
async def generate_mood_board(
    concept: str,
    target_mood: str,
    color_palette: List[str] = None,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """향기 무드보드 생성"""
    try:
        # 실제 무드보드 생성
        mood_request = {
            "prompt": f"{concept} 컨셉으로 {target_mood} 무드의 향수 무드보드 생성",
            "concept": concept,
            "target_mood": target_mood,
            "color_palette": color_palette,
            "generation_type": "mood_board"
        }
        
        # 생성 서비스를 통해 무드보드 생성
        mood_result = await generation_service.generate_recipe(mood_request)
        
        # 생성된 결과에서 무드보드 요소 추출
        recipe_data = mood_result.get("recipe", {})
        
        # 색상 팔레트 처리
        colors = color_palette or [
            "#E6E6FA", "#DDA0DD", "#9370DB",  # 라벤더 톤
            "#FFB6C1", "#FFC0CB", "#FF69B4",  # 핀크 톤  
            "#F0E68C", "#FFFF99", "#FFD700"   # 옘로우 톤
        ][:6]  # 최대 6개 색상
        
        # 텍스처와 영감 추출
        textures = recipe_data.get("textures", ["silk", "velvet", "mist", "cotton", "satin"])
        inspirations = recipe_data.get("inspirations", ["sunrise", "ocean breeze", "garden walk", "moonlight", "forest rain"])
        
        # 향료 추천 생성
        fragrance_recommendations = []
        notes = recipe_data.get("notes", {})
        
        # 노트별로 추천 생성
        for note_type, note_list in notes.items():
            if isinstance(note_list, list) and note_list:
                for note in note_list[:2]:  # 각 노트 타입에서 2개씩
                    role_map = {
                        "top": "uplifting opener",
                        "middle": "harmonious heart", 
                        "base": "lasting foundation"
                    }
                    fragrance_recommendations.append({
                        "note": note,
                        "intensity": 6 + (hash(note) % 4),  # 6-9 사이의 강도
                        "role": role_map.get(note_type, "supporting note")
                    })
        
        # 기본 추천이 없으면 기본값
        if not fragrance_recommendations:
            fragrance_recommendations = [
                {"note": "라벤더", "intensity": 7, "role": "calming base"},
                {"note": "베르가못", "intensity": 8, "role": "uplifting top"},
                {"note": "새니달우드", "intensity": 6, "role": "warm heart"}
            ]
        
        # 스토리 요소 추출
        story_elements = recipe_data.get("story_elements", [
            "아침 정원의 이슬",
            "따스한 햇살", 
            "부드러운 바람",
            "꽃쟎이의 속삭임",
            "연인들의 만남"
        ])
        
        return {
            "mood_board_id": str(uuid.uuid4()),
            "concept": concept,
            "target_mood": target_mood,
            "visual_elements": {
                "colors": colors,
                "textures": textures[:5],  # 최대 5개
                "inspirations": inspirations[:5]  # 최대 5개
            },
            "fragrance_recommendations": fragrance_recommendations[:6],  # 최대 6개
            "story_elements": story_elements[:5],  # 최대 5개
            "generated_recipe": recipe_data,
            "creation_notes": recipe_data.get("story", f"{concept} 컨셉의 {target_mood} 무드를 표현하는 예술적 무드보드")
        }
        
    except Exception as e:
        logger.error(f"Mood board generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Artisan Chat 엔드포인트 - 새로운 대화형 인터페이스
class ChatRequest(BaseModel):
    """채팅 요청"""
    message: str = Field(..., description="사용자 메시지")
    conversation_id: Optional[str] = Field(None, description="대화 ID")
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="대화 기록")


class ChatResponse(BaseModel):
    """채팅 응답"""
    response: str = Field(..., description="AI 응답")
    recipe_preview: Optional[Dict[str, Any]] = Field(None, description="생성된 레시피 미리보기")
    suggestions: Optional[List[str]] = Field(None, description="추가 제안사항")
    conversation_id: str = Field(..., description="대화 ID")
    request_id: str = Field(..., description="요청 ID")


@router.post("/chat", response_model=ChatResponse)
async def chat_with_artisan(
    request: ChatRequest,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Artisan AI와 대화하기 - 향수 전문 AI 챗봇

    - 향수 레시피 생성
    - 향수 지식 답변
    - 기존 향수 검색
    - 조합 검증
    """
    try:
        # Artisan Orchestrator 임포트
        from ...orchestrator.artisan_orchestrator import process_chat_message

        # 사용자 ID 확인
        user_id = current_user.get("id") if current_user else f"guest_{uuid.uuid4().hex[:8]}"

        # 대화 ID 확인 또는 생성
        conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

        # 대화 기록 준비
        history = request.history or []

        logger.info(f"Processing chat request for user {user_id}, conversation {conversation_id}")

        # Artisan 오케스트레이터 호출
        artisan_response = await process_chat_message(
            message=request.message,
            user_id=user_id,
            conversation_id=conversation_id,
            history=history
        )

        # 레시피 미리보기 준비 (IP 보호 - 요약만 제공)
        recipe_preview = None
        if artisan_response.recipe_summary:
            recipe_preview = {
                "name": artisan_response.recipe_summary.get("name"),
                "description": artisan_response.recipe_summary.get("description"),
                "key_notes": artisan_response.recipe_summary.get("key_notes"),
                "character": artisan_response.recipe_summary.get("character"),
                "occasions": artisan_response.recipe_summary.get("occasions", [])
            }

        return ChatResponse(
            response=artisan_response.message,
            recipe_preview=recipe_preview,
            suggestions=artisan_response.suggestions,
            conversation_id=conversation_id,
            request_id=artisan_response.request_id
        )

    except ImportError as e:
        logger.error(f"Failed to import Artisan orchestrator: {e}")
        # 폴백 응답
        return ChatResponse(
            response="죄송합니다. AI 시스템이 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해주세요.",
            conversation_id=request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}",
            request_id=f"req_{uuid.uuid4().hex[:8]}"
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"대화 처리 중 오류가 발생했습니다: {str(e)}"
        )


# AI Perfumer (공감각 조향사) 엔드포인트
class AIPerfumerRequest(BaseModel):
    """AI 조향사 요청 모델"""
    message: str = Field(..., description="사용자 메시지")
    context: List[str] = Field(default_factory=list, description="대화 맥락")
    session_id: Optional[str] = Field(None, description="세션 ID")


class AIPerfumerResponse(BaseModel):
    """AI 조향사 응답 모델"""
    response: str = Field(..., description="AI 응답")
    fragrance: Optional[Dict[str, Any]] = Field(None, description="생성된 향수")
    timestamp: str = Field(..., description="응답 시간")
    session_id: str = Field(..., description="세션 ID")


@router.post("/ai-perfumer/chat", response_model=AIPerfumerResponse)
async def ai_perfumer_chat(
    request: AIPerfumerRequest,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    AI 조향사와 대화 (공감각적 향수 창조)

    세상의 모든 개념을 향으로 번역하는 예술가와 대화합니다.
    추상적 개념, 감정, 기억을 향수로 변환합니다.
    """
    try:
        # AI Perfumer 오케스트레이터 임포트
        from ...orchestrator.ai_perfumer_orchestrator import get_ai_perfumer_orchestrator

        # 오케스트레이터 가져오기
        orchestrator = get_ai_perfumer_orchestrator()

        # 세션 관리
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

        # 대화 처리
        response = orchestrator.generate_response(
            message=request.message,
            context=request.context
        )

        # 충분한 대화 후 향수 생성
        fragrance = None
        if len(request.context) >= 2:
            full_context = ' '.join(request.context + [request.message])
            fragrance_data = orchestrator.execute_creative_process(full_context)
            fragrance = fragrance_data

        return AIPerfumerResponse(
            response=response,
            fragrance=fragrance,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"AI Perfumer chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))