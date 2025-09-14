from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List
import logging
import time
import uuid
import asyncio

from ..schemas import (
    RecipeGenerationRequest,
    RecipeGenerationResponse,
    BatchGenerationRequest,
    BatchGenerationResponse,
    FragranceRecipe,
    QualityScores,
    FragranceComposition
)
from ...services.generation_service import GenerationService
from ..dependencies import get_generation_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/recipe", response_model=RecipeGenerationResponse)
async def generate_recipe(
    request: RecipeGenerationRequest,
    background_tasks: BackgroundTasks,
    generation_service: GenerationService = Depends(get_generation_service)
):
    """향수 레시피 생성"""
    try:
        start_time = time.time()
        
        # 실제 생성 서비스 호출
        request_data = {
            "fragrance_family": getattr(request, 'fragrance_family', 'floral'),
            "mood": getattr(request, 'mood', 'romantic'),
            "intensity": getattr(request, 'intensity', 'moderate'),
            "gender": getattr(request, 'gender', 'unisex'),
            "season": getattr(request, 'season', 'spring'),
            "generation_type": str(request.recipe_type.value) if hasattr(request.recipe_type, 'value') else str(request.recipe_type)
        }
        
        # 생성 서비스를 통해 레시피 생성
        generation_result = await generation_service.generate_recipe(request_data)
        
        # 생성 결과에서 레시피 정보 추출
        recipe_data = generation_result.get("recipe", {})
        
        # FragranceRecipe 객체 생성
        fragrance_recipe = FragranceRecipe(
            name=recipe_data.get("name", "생성된 향수"),
            concept=recipe_data.get("concept", recipe_data.get("description", "AI가 생성한 향수")),
            mood=recipe_data.get("mood", request_data["mood"]),
            season=recipe_data.get("season", request_data["season"]),
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
        
        # 품질 점수 (생성 결과에서 추출 또는 기본값)
        quality_score = generation_result.get("quality_score", 75.0)
        quality_scores = QualityScores(
            completeness=min(1.0, quality_score / 100.0),
            coherence=min(1.0, (quality_score + 5) / 100.0),
            creativity=min(1.0, (quality_score - 5) / 100.0),
            technical_accuracy=min(1.0, quality_score / 100.0),
            overall=min(1.0, quality_score / 100.0)
        )
        
        return RecipeGenerationResponse(
            recipe=fragrance_recipe,
            quality_scores=quality_scores,
            generation_time=time.time() - start_time,
            prompt=request.prompt
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