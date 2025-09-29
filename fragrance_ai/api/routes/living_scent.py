"""
Living Scent API Routes
살아있는 향수 DNA 생성 및 진화 API
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import logging

from fragrance_ai.database.base import get_db
from fragrance_ai.orchestrator.living_scent_orchestrator import get_living_scent_orchestrator
from fragrance_ai.database.living_scent_models import (
    OlfactoryDNAModel,
    ScentPhenotypeModel,
    UserInteractionModel
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/living-scent", tags=["Living Scent"])


# Pydantic 모델들
class LivingScentRequest(BaseModel):
    """Living Scent 생성 요청"""
    user_input: str = Field(..., description="사용자의 자연어 입력")
    user_id: Optional[str] = Field(None, description="사용자 ID")
    existing_dna_id: Optional[str] = Field(None, description="진화시킬 기존 DNA ID")

    class Config:
        json_schema_extra = {
            "example": {
                "user_input": "옛날 할머니 댁 다락방에서 나던 낡고 포근한 느낌의 냄새를 만들어줘",
                "user_id": "user123"
            }
        }


class LivingScentResponse(BaseModel):
    """Living Scent 응답"""
    success: bool
    intent: str
    result: Dict[str, Any]
    metadata: Dict[str, Any]
    message: Optional[str] = None


class DNAEvolutionRequest(BaseModel):
    """DNA 진화 요청"""
    dna_id: str = Field(..., description="진화시킬 DNA ID")
    feedback: str = Field(..., description="사용자 피드백")
    user_id: Optional[str] = Field(None, description="사용자 ID")

    class Config:
        json_schema_extra = {
            "example": {
                "dna_id": "DNA_ABC123DEF456",
                "feedback": "더 스모키하고 강렬하게 만들어줘",
                "user_id": "user123"
            }
        }


class RatingRequest(BaseModel):
    """평점 요청"""
    phenotype_id: Optional[str] = Field(None, description="표현형 ID")
    dna_id: Optional[str] = Field(None, description="DNA ID")
    rating: float = Field(..., ge=1, le=5, description="평점 (1-5)")
    feedback: Optional[str] = Field(None, description="텍스트 피드백")
    user_id: str = Field(..., description="사용자 ID")


# API 엔드포인트들

@router.post("/create", response_model=LivingScentResponse)
async def create_living_scent(
    request: LivingScentRequest,
    db: Session = Depends(get_db)
):
    """
    새로운 Living Scent DNA를 생성하거나 기존 DNA를 진화시킵니다.

    - **user_input**: 자연어로 된 향수 설명
    - **user_id**: (선택) 사용자 식별자
    - **existing_dna_id**: (선택) 진화시킬 기존 DNA ID
    """
    try:
        orchestrator = get_living_scent_orchestrator(db)
        result = orchestrator.process_user_input(
            user_input=request.user_input,
            user_id=request.user_id,
            existing_dna_id=request.existing_dna_id
        )

        if result['success']:
            return LivingScentResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result.get('message', 'Processing failed'))

    except Exception as e:
        logger.error(f"Error creating living scent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evolve", response_model=LivingScentResponse)
async def evolve_dna(
    request: DNAEvolutionRequest,
    db: Session = Depends(get_db)
):
    """
    기존 DNA를 사용자 피드백에 따라 진화시킵니다.

    - **dna_id**: 진화시킬 DNA의 ID
    - **feedback**: 변형을 위한 피드백 (예: "더 강하게", "스모키하게")
    - **user_id**: (선택) 사용자 식별자
    """
    try:
        orchestrator = get_living_scent_orchestrator(db)

        # 피드백을 포함한 입력 생성
        evolved_input = f"기존 향수를 {request.feedback}"

        result = orchestrator.process_user_input(
            user_input=evolved_input,
            user_id=request.user_id,
            existing_dna_id=request.dna_id
        )

        if result['success']:
            return LivingScentResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result.get('message', 'Evolution failed'))

    except Exception as e:
        logger.error(f"Error evolving DNA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dna/{dna_id}")
async def get_dna_details(
    dna_id: str,
    db: Session = Depends(get_db)
):
    """
    특정 DNA의 상세 정보를 조회합니다.

    - **dna_id**: 조회할 DNA ID
    """
    try:
        orchestrator = get_living_scent_orchestrator(db)
        dna_info = orchestrator.get_dna_info(dna_id)

        if dna_info:
            return dna_info
        else:
            # DB에서 직접 조회
            dna_model = db.query(OlfactoryDNAModel).filter_by(dna_id=dna_id).first()
            if dna_model:
                return {
                    "dna_id": dna_model.dna_id,
                    "lineage": dna_model.lineage,
                    "genotype": dna_model.genotype,
                    "phenotype_potential": dna_model.phenotype_potential,
                    "story": dna_model.story,
                    "generation": dna_model.generation,
                    "fitness_score": dna_model.fitness_score,
                    "total_phenotypes": dna_model.total_phenotypes,
                    "average_rating": dna_model.average_rating,
                    "created_at": str(dna_model.created_at)
                }
            else:
                raise HTTPException(status_code=404, detail="DNA not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DNA details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phenotype/{phenotype_id}")
async def get_phenotype_details(
    phenotype_id: str,
    db: Session = Depends(get_db)
):
    """
    특정 표현형의 상세 정보를 조회합니다.

    - **phenotype_id**: 조회할 표현형 ID
    """
    try:
        orchestrator = get_living_scent_orchestrator(db)
        phenotype_info = orchestrator.get_phenotype_info(phenotype_id)

        if phenotype_info:
            return phenotype_info
        else:
            # DB에서 직접 조회
            pheno_model = db.query(ScentPhenotypeModel).filter_by(
                phenotype_id=phenotype_id
            ).first()
            if pheno_model:
                return {
                    "phenotype_id": pheno_model.phenotype_id,
                    "based_on_dna": pheno_model.based_on_dna,
                    "recipe": pheno_model.recipe,
                    "description": pheno_model.description,
                    "epigenetic_trigger": pheno_model.epigenetic_trigger,
                    "environmental_response": pheno_model.environmental_response,
                    "evolution_path": pheno_model.evolution_path,
                    "user_rating": pheno_model.user_rating,
                    "created_at": str(pheno_model.created_at)
                }
            else:
                raise HTTPException(status_code=404, detail="Phenotype not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting phenotype details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evolution-tree/{dna_id}")
async def get_evolution_tree(
    dna_id: str,
    db: Session = Depends(get_db)
):
    """
    특정 DNA의 진화 트리를 조회합니다.
    모든 파생된 표현형과 그들의 관계를 보여줍니다.

    - **dna_id**: 루트 DNA ID
    """
    try:
        orchestrator = get_living_scent_orchestrator(db)
        tree = orchestrator.get_evolution_tree(dna_id)

        # DB에서 추가 정보 가져오기
        if db:
            phenotypes = db.query(ScentPhenotypeModel).filter_by(
                based_on_dna=dna_id
            ).all()

            tree['phenotypes'] = [
                {
                    "id": p.phenotype_id,
                    "trigger": p.epigenetic_trigger,
                    "rating": p.user_rating,
                    "created_at": str(p.created_at)
                }
                for p in phenotypes
            ]

        return tree

    except Exception as e:
        logger.error(f"Error getting evolution tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rate")
async def rate_fragrance(
    request: RatingRequest,
    db: Session = Depends(get_db)
):
    """
    DNA 또는 표현형에 평점을 매깁니다.

    - **phenotype_id**: (선택) 평가할 표현형 ID
    - **dna_id**: (선택) 평가할 DNA ID
    - **rating**: 1-5 사이의 평점
    - **feedback**: (선택) 텍스트 피드백
    - **user_id**: 사용자 ID
    """
    try:
        if not request.phenotype_id and not request.dna_id:
            raise HTTPException(status_code=400, detail="Either phenotype_id or dna_id is required")

        # 표현형 평가
        if request.phenotype_id:
            phenotype = db.query(ScentPhenotypeModel).filter_by(
                phenotype_id=request.phenotype_id
            ).first()
            if not phenotype:
                raise HTTPException(status_code=404, detail="Phenotype not found")

            phenotype.user_rating = request.rating
            db.commit()

            # 상호작용 기록
            interaction = UserInteractionModel(
                user_id=request.user_id,
                phenotype_id=request.phenotype_id,
                interaction_type="rate",
                interaction_data={
                    "rating": request.rating,
                    "feedback": request.feedback
                },
                feedback_text=request.feedback,
                satisfaction_score=request.rating
            )
            db.add(interaction)
            db.commit()

        # DNA 평가
        if request.dna_id:
            dna = db.query(OlfactoryDNAModel).filter_by(dna_id=request.dna_id).first()
            if not dna:
                raise HTTPException(status_code=404, detail="DNA not found")

            # 평균 평점 업데이트
            current_total = dna.average_rating * dna.usage_count
            dna.usage_count += 1
            dna.average_rating = (current_total + request.rating) / dna.usage_count
            db.commit()

            # 상호작용 기록
            interaction = UserInteractionModel(
                user_id=request.user_id,
                dna_id=request.dna_id,
                interaction_type="rate",
                interaction_data={
                    "rating": request.rating,
                    "feedback": request.feedback
                },
                feedback_text=request.feedback,
                satisfaction_score=request.rating
            )
            db.add(interaction)
            db.commit()

        return {"success": True, "message": "Rating saved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving rating: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}/history")
async def get_user_history(
    user_id: str,
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    특정 사용자의 Living Scent 생성/진화 히스토리를 조회합니다.

    - **user_id**: 사용자 ID
    - **limit**: 반환할 최대 항목 수 (기본값: 10)
    """
    try:
        # DNA 히스토리
        dnas = db.query(OlfactoryDNAModel).filter_by(
            created_by_user_id=user_id
        ).order_by(OlfactoryDNAModel.created_at.desc()).limit(limit).all()

        # 표현형 히스토리
        phenotypes = db.query(ScentPhenotypeModel).filter_by(
            created_by_user_id=user_id
        ).order_by(ScentPhenotypeModel.created_at.desc()).limit(limit).all()

        return {
            "user_id": user_id,
            "dna_creations": [
                {
                    "dna_id": d.dna_id,
                    "story": d.story,
                    "generation": d.generation,
                    "created_at": str(d.created_at)
                }
                for d in dnas
            ],
            "phenotype_evolutions": [
                {
                    "phenotype_id": p.phenotype_id,
                    "based_on_dna": p.based_on_dna,
                    "description": p.description,
                    "created_at": str(p.created_at)
                }
                for p in phenotypes
            ]
        }

    except Exception as e:
        logger.error(f"Error getting user history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trending")
async def get_trending_dna(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    인기 있는 DNA들을 조회합니다.
    평점과 사용 횟수를 기준으로 정렬됩니다.

    - **limit**: 반환할 최대 항목 수 (기본값: 10)
    """
    try:
        trending = db.query(OlfactoryDNAModel).filter(
            OlfactoryDNAModel.is_public == True,
            OlfactoryDNAModel.is_active == True
        ).order_by(
            (OlfactoryDNAModel.average_rating * OlfactoryDNAModel.usage_count).desc()
        ).limit(limit).all()

        return {
            "trending_dna": [
                {
                    "dna_id": d.dna_id,
                    "story": d.story,
                    "generation": d.generation,
                    "average_rating": d.average_rating,
                    "usage_count": d.usage_count,
                    "total_phenotypes": d.total_phenotypes
                }
                for d in trending
            ]
        }

    except Exception as e:
        logger.error(f"Error getting trending DNA: {e}")
        raise HTTPException(status_code=500, detail=str(e))