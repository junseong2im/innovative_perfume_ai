"""
AI 향수 생성 라우터 - DEAP MOGA + RLHF 통합
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional, List
import logging
import time
import uuid
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pydantic import BaseModel, Field
from fragrance_ai.training.olfactory_recombinator_deap import OlfactoryRecombinatorAI, CreativeBrief
from fragrance_ai.training.enhanced_rlhf_policy import FragranceEvolutionSystem, FragranceState
from fragrance_ai.tools.validator_tool import validate_composition, NotesComposition

logger = logging.getLogger(__name__)
router = APIRouter()

# 글로벌 인스턴스 (싱글톤)
recombinator = OlfactoryRecombinatorAI()
evolution_system = FragranceEvolutionSystem()


class AIGenerationRequest(BaseModel):
    """AI 향수 생성 요청"""
    description: str = Field(..., description="향수 설명 텍스트")
    emotional_keywords: List[str] = Field(default=[], description="감정 키워드")
    fragrance_family: str = Field(default="floral", description="향수 계열")
    season: str = Field(default="spring", description="계절")
    occasion: str = Field(default="daily", description="상황")
    intensity: float = Field(default=0.5, ge=0, le=1, description="강도")
    avoid_notes: List[str] = Field(default=[], description="피해야 할 노트")


class AIGenerationResponse(BaseModel):
    """AI 향수 생성 응답"""
    recipe: Dict[str, Any]
    dna: List[float]
    evaluation: Dict[str, float]
    generation_id: str
    processing_time: float
    ai_method: str


class AIEvolutionRequest(BaseModel):
    """AI 향수 진화 요청"""
    current_dna: List[float] = Field(..., description="현재 DNA")
    user_rating: float = Field(..., ge=0, le=10, description="사용자 평점")
    feedback_text: str = Field(default="", description="피드백 텍스트")
    metrics: Dict[str, float] = Field(default={}, description="평가 메트릭")


class AIEvolutionResponse(BaseModel):
    """AI 향수 진화 응답"""
    variations: List[Dict[str, Any]]
    original_dna: List[float]
    evolution_id: str
    processing_time: float


@router.post("/generate/moga", response_model=AIGenerationResponse)
async def generate_with_moga(request: AIGenerationRequest):
    """DEAP MOGA를 사용한 향수 생성"""
    try:
        start_time = time.time()
        generation_id = str(uuid.uuid4())

        logger.info(f"MOGA Generation started: {generation_id}")

        # CreativeBrief 생성
        emotional_profile = {}
        for keyword in request.emotional_keywords:
            emotional_profile[keyword] = np.random.uniform(0.5, 1.0)

        brief = CreativeBrief(
            emotional_profile=emotional_profile,
            fragrance_family=request.fragrance_family,
            season=request.season,
            occasion=request.occasion,
            intensity=request.intensity,
            keywords=request.emotional_keywords,
            avoid_notes=request.avoid_notes
        )

        # MOGA 최적화 실행
        result = recombinator.generate_olfactory_dna(
            brief=brief,
            population_size=50,
            generations=20,
            verbose=False
        )

        # 레시피 포맷팅
        recipe = result['recipe']

        # ValidatorTool을 사용한 과학적 검증
        validation_result = None
        try:
            composition = NotesComposition(
                top_notes=[{note: pct} for note, pct in recipe['top'].items()],
                heart_notes=[{note: pct} for note, pct in recipe['middle'].items()],
                base_notes=[{note: pct} for note, pct in recipe['base'].items()],
                total_ingredients=len(recipe['top']) + len(recipe['middle']) + len(recipe['base'])
            )
            validation_result = await validate_composition(composition)
            logger.info(f"Validation result: {validation_result.overall_score}/10")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")

        formatted_recipe = {
            "name": f"MOGA-{generation_id[:8]}",
            "description": f"AI-generated fragrance based on: {request.description}",
            "composition": {
                "top_notes": [
                    {"name": note, "percentage": round(pct, 2)}
                    for note, pct in recipe['top'].items()
                ],
                "heart_notes": [
                    {"name": note, "percentage": round(pct, 2)}
                    for note, pct in recipe['middle'].items()
                ],
                "base_notes": [
                    {"name": note, "percentage": round(pct, 2)}
                    for note, pct in recipe['base'].items()
                ]
            },
            "characteristics": {
                "intensity": request.intensity,
                "season": request.season,
                "occasion": request.occasion,
                "family": request.fragrance_family
            },
            "validation": {
                "is_valid": validation_result.is_valid if validation_result else True,
                "overall_score": validation_result.overall_score if validation_result else 0,
                "harmony_score": validation_result.harmony_score if validation_result else 0,
                "stability_score": validation_result.stability_score if validation_result else 0,
                "scientific_notes": validation_result.scientific_notes if validation_result else ""
            } if validation_result else None
        }

        processing_time = time.time() - start_time

        return AIGenerationResponse(
            recipe=formatted_recipe,
            dna=result['olfactory_dna'],
            evaluation=result['evaluation'],
            generation_id=generation_id,
            processing_time=processing_time,
            ai_method="DEAP_MOGA"
        )

    except Exception as e:
        logger.error(f"MOGA generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve/rlhf", response_model=AIEvolutionResponse)
async def evolve_with_rlhf(request: AIEvolutionRequest):
    """RLHF를 사용한 향수 진화"""
    try:
        start_time = time.time()
        evolution_id = str(uuid.uuid4())

        logger.info(f"RLHF Evolution started: {evolution_id}")

        # DNA 배열 변환
        current_dna = np.array(request.current_dna)

        # 사용자 피드백 준비
        user_feedback = {
            'metrics': request.metrics,
            'rating': request.user_rating,
            'text': request.feedback_text
        }

        # RLHF 진화 실행
        result = evolution_system.evolve_fragrance(
            current_dna=current_dna,
            user_feedback=user_feedback
        )

        # 학습 (피드백이 충분할 때)
        if request.user_rating != 5.0:  # 중립이 아닌 경우만
            session_data = [{
                'dna': request.current_dna,
                'feedback': request.metrics,
                'rating': request.user_rating,
                'feedback_text': request.feedback_text,
                'action': {
                    'type': 'user_feedback',
                    'targets': [],
                    'intensity': 0.5
                },
                'improvements': {}
            }]

            training_stats = evolution_system.train_on_feedback(session_data)
            logger.info(f"RLHF Training stats: {training_stats}")

        processing_time = time.time() - start_time

        return AIEvolutionResponse(
            variations=result['variations'],
            original_dna=result['original_dna'],
            evolution_id=evolution_id,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"RLHF evolution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/hybrid", response_model=Dict[str, Any])
async def generate_hybrid(request: AIGenerationRequest):
    """MOGA + RLHF 하이브리드 생성"""
    try:
        start_time = time.time()
        generation_id = str(uuid.uuid4())

        logger.info(f"Hybrid Generation started: {generation_id}")

        # 1단계: MOGA로 초기 DNA 생성
        emotional_profile = {
            keyword: np.random.uniform(0.6, 1.0)
            for keyword in request.emotional_keywords
        }

        brief = CreativeBrief(
            emotional_profile=emotional_profile,
            fragrance_family=request.fragrance_family,
            season=request.season,
            occasion=request.occasion,
            intensity=request.intensity,
            keywords=request.emotional_keywords,
            avoid_notes=request.avoid_notes
        )

        moga_result = recombinator.generate_olfactory_dna(
            brief=brief,
            population_size=30,
            generations=10,
            verbose=False
        )

        # 2단계: RLHF로 개선
        initial_dna = np.array(moga_result['olfactory_dna'])

        # 가상의 사용자 피드백 (초기값)
        initial_feedback = {
            'metrics': moga_result['evaluation'],
            'rating': 7.0,  # 초기 평점
            'text': request.description
        }

        evolution_result = evolution_system.evolve_fragrance(
            current_dna=initial_dna,
            user_feedback=initial_feedback
        )

        # 최종 레시피 선택 (첫 번째 변형)
        best_variation = evolution_result['variations'][0]
        final_dna = np.array(best_variation['dna'])

        # DNA를 레시피로 변환
        final_recipe = recombinator._individual_to_recipe(final_dna)

        # 포맷팅
        formatted_recipe = {
            "name": f"Hybrid-{generation_id[:8]}",
            "description": f"AI-optimized fragrance: {request.description}",
            "composition": {
                "top_notes": [
                    {"name": note, "percentage": round(pct, 2)}
                    for note, pct in final_recipe['top'].items()
                ],
                "heart_notes": [
                    {"name": note, "percentage": round(pct, 2)}
                    for note, pct in final_recipe['middle'].items()
                ],
                "base_notes": [
                    {"name": note, "percentage": round(pct, 2)}
                    for note, pct in final_recipe['base'].items()
                ]
            },
            "characteristics": {
                "intensity": request.intensity,
                "season": request.season,
                "occasion": request.occasion,
                "family": request.fragrance_family
            },
            "ai_optimization": {
                "moga_fitness": moga_result['fitness_values'],
                "rlhf_variations": len(evolution_result['variations']),
                "method": "MOGA+RLHF"
            }
        }

        processing_time = time.time() - start_time

        return {
            "recipe": formatted_recipe,
            "dna": final_dna.tolist(),
            "generation_id": generation_id,
            "processing_time": processing_time,
            "ai_method": "Hybrid_MOGA_RLHF",
            "optimization_steps": {
                "moga_generations": 10,
                "rlhf_variations": len(evolution_result['variations'])
            }
        }

    except Exception as e:
        logger.error(f"Hybrid generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/status")
async def get_ai_status():
    """AI 시스템 상태 확인"""
    return {
        "status": "operational",
        "engines": {
            "moga": "DEAP-based Multi-Objective Genetic Algorithm",
            "rlhf": "PyTorch-based Reinforcement Learning from Human Feedback",
            "hybrid": "MOGA+RLHF Combined System"
        },
        "capabilities": [
            "DNA-based fragrance generation",
            "Multi-objective optimization",
            "User feedback learning",
            "Recipe evolution"
        ],
        "version": "2.0.0"
    }