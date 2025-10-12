# app/main.py
"""
Main FastAPI Application
Handles DNA creation, evolution with RLHF, and experiment tracking
"""

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import traceback
import uuid
import hashlib

# Import fragrance AI modules
from fragrance_ai.schemas.domain_models import (
    OlfactoryDNA, CreativeBrief, ScentPhenotype,
    Ingredient, NoteCategory
)
from fragrance_ai.schemas.models import (
    FragranceFormula, FormulaIngredient, ValidationLevel,
    ComplianceCheck, FormulaType
)
from fragrance_ai.services.evolution_service import get_evolution_service
from fragrance_ai.rules.ifra_rules import (
    get_ifra_checker, get_allergen_checker, ProductCategory
)
from fragrance_ai.eval.objectives import TotalObjective, OptimizationProfile
from fragrance_ai.utils.units import UnitConverter, MassConservationChecker
from fragrance_ai.observability import (
    orchestrator_logger, metrics_collector, log_timing, get_logger
)

# ============================================================================
# Setup Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI App Configuration
# ============================================================================

app = FastAPI(
    title="Fragrance AI API",
    description="AI-powered fragrance creation with RLHF evolution",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class DNACreateRequest(BaseModel):
    """Request model for DNA creation"""
    brief: Dict[str, Any] = Field(..., description="Creative brief with requirements")
    name: Optional[str] = Field(None, description="Name for the DNA")
    description: Optional[str] = Field(None, description="Description")
    target_cost_per_kg: Optional[float] = Field(None, ge=0)
    product_category: str = Field("eau_de_parfum", description="Product category for IFRA")
    validation_level: ValidationLevel = ValidationLevel.STRICT

    class Config:
        json_schema_extra = {
            "example": {
                "brief": {
                    "style": "fresh",
                    "intensity": 0.7,
                    "complexity": 0.5,
                    "masculinity": 0.6,
                    "season": "summer",
                    "notes": ["citrus", "woody", "aquatic"]
                },
                "name": "Summer Breeze",
                "product_category": "eau_de_toilette"
            }
        }


class DNACreateResponse(BaseModel):
    """Response model for DNA creation"""
    dna_id: str
    name: str
    description: Optional[str]
    ingredients: List[Dict[str, Any]]
    total_cost_per_kg: Optional[float]
    compliance: Dict[str, Any]
    created_at: str

    class Config:
        json_schema_extra = {
            "example": {
                "dna_id": "dna_abc123",
                "name": "Summer Breeze",
                "ingredients": [
                    {"name": "Bergamot", "percentage": 15.0, "category": "top"},
                    {"name": "Marine Accord", "percentage": 10.0, "category": "heart"}
                ],
                "total_cost_per_kg": 125.50,
                "compliance": {"ifra_compliant": True, "allergens": []},
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class EvolveOptionsRequest(BaseModel):
    """Request model for evolution options"""
    dna_id: str = Field(..., description="DNA identifier")
    brief: Dict[str, Any] = Field(..., description="Creative brief")
    num_options: int = Field(3, ge=1, le=10, description="Number of variations")
    optimization_profile: OptimizationProfile = OptimizationProfile.COMMERCIAL
    algorithm: str = Field("PPO", pattern="^(PPO|REINFORCE)$")

    class Config:
        json_schema_extra = {
            "example": {
                "dna_id": "dna_abc123",
                "brief": {
                    "style": "fresh",
                    "intensity": 0.7
                },
                "num_options": 3,
                "optimization_profile": "commercial",
                "algorithm": "PPO"
            }
        }


class EvolveOptionsResponse(BaseModel):
    """Response model for evolution options"""
    experiment_id: str
    options: List[Dict[str, Any]]
    optimization_scores: Optional[Dict[str, float]]
    created_at: str

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "exp_xyz789",
                "options": [
                    {
                        "id": "opt_1",
                        "action": "amplify_top",
                        "description": "Amplify top notes by 15%",
                        "preview": {"Bergamot": 17.25, "Lemon": 11.5}
                    }
                ],
                "optimization_scores": {
                    "creativity": 0.7,
                    "fitness": 0.8,
                    "stability": 0.6
                },
                "created_at": "2024-01-15T10:35:00Z"
            }
        }


class EvolveFeedbackRequest(BaseModel):
    """Request model for evolution feedback"""
    experiment_id: str = Field(..., description="Experiment identifier")
    chosen_id: str = Field(..., description="Chosen option ID")
    rating: Optional[float] = Field(None, ge=1, le=5, description="User rating 1-5")
    notes: Optional[str] = Field(None, description="Additional feedback notes")

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "exp_xyz789",
                "chosen_id": "opt_1",
                "rating": 4,
                "notes": "Good balance, slightly too strong on citrus"
            }
        }


class EvolveFeedbackResponse(BaseModel):
    """Response model for evolution feedback"""
    status: str
    experiment_id: str
    iteration: int
    metrics: Dict[str, float]
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "experiment_id": "exp_xyz789",
                "iteration": 1,
                "metrics": {"loss": 0.125, "reward": 0.5},
                "message": "Feedback processed and policy updated"
            }
        }


class ExperimentStatusResponse(BaseModel):
    """Response model for experiment status"""
    experiment_id: str
    status: str
    created_at: str
    iterations: int
    last_feedback: Optional[Dict[str, Any]]
    dna_id: str
    algorithm: str

    class Config:
        json_schema_extra = {
            "example": {
                "experiment_id": "exp_xyz789",
                "status": "active",
                "created_at": "2024-01-15T10:35:00Z",
                "iterations": 2,
                "last_feedback": {
                    "chosen_id": "opt_1",
                    "rating": 4,
                    "timestamp": "2024-01-15T10:40:00Z"
                },
                "dna_id": "dna_abc123",
                "algorithm": "PPO"
            }
        }


# ============================================================================
# Error Response Model
# ============================================================================

class ErrorResponse(BaseModel):
    """Consistent error response format"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    class Config:
        json_schema_extra = {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid request parameters",
                "details": {"field": "rating", "issue": "Must be between 1 and 5"},
                "timestamp": "2024-01-15T10:45:00Z"
            }
        }


# ============================================================================
# In-Memory Storage (for demo - use database in production)
# ============================================================================

DNA_STORAGE: Dict[str, Dict[str, Any]] = {}
EXPERIMENT_STORAGE: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def generate_dna_id(name: str) -> str:
    """Generate unique DNA ID"""
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{name}_{timestamp}"
    return f"dna_{hashlib.sha256(hash_input.encode()).hexdigest()[:12]}"


def create_dna_from_brief(brief: Dict[str, Any], name: str) -> OlfactoryDNA:
    """
    Create initial DNA from creative brief

    This is a simplified version - in production, you'd use
    more sophisticated generation based on the brief
    """
    # Mock ingredient selection based on brief
    ingredients = []

    # Top notes (30%)
    if "citrus" in str(brief).lower() or brief.get("style") == "fresh":
        ingredients.append(Ingredient(
            id=1,
            name="Bergamot",
            cas_number="8007-75-8",
            concentration=15.0,
            category=NoteCategory.TOP,
            cost_per_kg=80.0,
            ifra_limit=2.0
        ))
        ingredients.append(Ingredient(
            id=2,
            name="Lemon",
            cas_number="8008-56-8",
            concentration=10.0,
            category=NoteCategory.TOP,
            cost_per_kg=60.0,
            ifra_limit=3.0
        ))
        ingredients.append(Ingredient(
            id=3,
            name="Grapefruit",
            cas_number="8016-20-4",
            concentration=5.0,
            category=NoteCategory.TOP,
            cost_per_kg=70.0
        ))

    # Heart notes (40%)
    if "floral" in str(brief).lower():
        ingredients.append(Ingredient(
            id=4,
            name="Rose Absolute",
            cas_number="8007-01-0",
            concentration=20.0,
            category=NoteCategory.HEART,
            cost_per_kg=3000.0,
            ifra_limit=0.6
        ))
    else:
        ingredients.append(Ingredient(
            id=5,
            name="Lavender",
            cas_number="8000-28-0",
            concentration=20.0,
            category=NoteCategory.HEART,
            cost_per_kg=120.0
        ))

    ingredients.append(Ingredient(
        id=6,
        name="Geranium",
        cas_number="8000-46-2",
        concentration=20.0,
        category=NoteCategory.HEART,
        cost_per_kg=180.0
    ))

    # Base notes (30%)
    if "woody" in str(brief).lower() or brief.get("masculinity", 0) > 0.5:
        ingredients.append(Ingredient(
            id=7,
            name="Sandalwood",
            cas_number="8006-87-9",
            concentration=15.0,
            category=NoteCategory.BASE,
            cost_per_kg=400.0
        ))
        ingredients.append(Ingredient(
            id=8,
            name="Cedarwood",
            cas_number="8000-27-9",
            concentration=15.0,
            category=NoteCategory.BASE,
            cost_per_kg=80.0
        ))
    else:
        ingredients.append(Ingredient(
            id=9,
            name="Vanilla",
            cas_number="8024-06-4",
            concentration=15.0,
            category=NoteCategory.BASE,
            cost_per_kg=200.0
        ))
        ingredients.append(Ingredient(
            id=10,
            name="Musk",
            cas_number="541-91-3",
            concentration=15.0,
            category=NoteCategory.BASE,
            cost_per_kg=1500.0
        ))

    # Create DNA
    dna_id = generate_dna_id(name)

    dna = OlfactoryDNA(
        dna_id=dna_id,
        name=name,
        ingredients=ingredients,
        generation=1,
        parent_ids=[],
        mutations=[],
        created_at=datetime.utcnow()
    )

    # Auto-normalize will ensure sum = 100%
    return dna


def check_compliance(
    ingredients: List[Dict[str, Any]],
    product_category: ProductCategory
) -> Dict[str, Any]:
    """Check IFRA and allergen compliance"""

    # Convert to recipe format for checkers
    recipe = {
        "ingredients": [
            {
                "name": ing["name"],
                "concentration": ing["concentration"]
            }
            for ing in ingredients
        ]
    }

    # IFRA check
    ifra_checker = get_ifra_checker()
    ifra_result = ifra_checker.check_ifra_violations(recipe, product_category)

    # Allergen check
    allergen_checker = get_allergen_checker()
    allergen_result = allergen_checker.check_allergens(recipe, 15.0)  # Assume 15% fragrance

    return {
        "ifra_compliant": ifra_result["compliant"],
        "ifra_violations": ifra_result["details"],
        "allergens_to_declare": allergen_result["allergens"],
        "overall_compliant": ifra_result["compliant"]
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "Fragrance AI API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check if services are available
        evolution_service = get_evolution_service()
        ifra_checker = get_ifra_checker()

        return {
            "status": "healthy",
            "services": {
                "evolution": "available",
                "ifra": "available",
                "storage": "in-memory"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get Prometheus metrics

    Returns metrics in Prometheus format for monitoring
    """
    from fastapi.responses import Response

    metrics_data = metrics_collector.get_metrics()
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4"
    )


@app.post(
    "/dna/create",
    response_model=DNACreateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["DNA"]
)
async def create_dna(request: DNACreateRequest):
    """
    Create initial DNA from creative brief

    - Generates fragrance DNA based on brief requirements
    - Checks IFRA compliance
    - Returns DNA with compliance status
    """
    try:
        # Validate request
        if not request.brief:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="INVALID_BRIEF",
                    message="Brief cannot be empty"
                ).model_dump()
            )

        # Create name if not provided
        name = request.name or f"Formula_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # Generate DNA from brief
        dna = create_dna_from_brief(request.brief, name)

        # Convert to dict format for storage and response
        ingredients_dict = []
        total_cost = 0.0

        for ing in dna.ingredients:
            ing_dict = {
                "id": ing.id,
                "name": ing.name,
                "cas_number": ing.cas_number,
                "concentration": ing.concentration,
                "percentage": ing.concentration,  # Same as concentration
                "category": ing.category.value,
                "cost_per_kg": ing.cost_per_kg,
                "ifra_limit": ing.ifra_limit
            }
            ingredients_dict.append(ing_dict)

            if ing.cost_per_kg:
                total_cost += (ing.concentration / 100) * ing.cost_per_kg

        # Check compliance
        product_category = ProductCategory(request.product_category)
        compliance = check_compliance(ingredients_dict, product_category)

        # Store DNA
        dna_data = {
            "dna_id": dna.dna_id,
            "name": dna.name,
            "description": request.description,
            "ingredients": ingredients_dict,
            "total_cost_per_kg": total_cost if total_cost > 0 else None,
            "compliance": compliance,
            "brief": request.brief,
            "created_at": datetime.utcnow().isoformat(),
            "product_category": request.product_category,
            "validation_level": request.validation_level.value
        }
        DNA_STORAGE[dna.dna_id] = dna_data

        logger.info(f"Created DNA: {dna.dna_id} with {len(ingredients_dict)} ingredients")

        # Return response
        return DNACreateResponse(
            dna_id=dna.dna_id,
            name=dna.name,
            description=request.description,
            ingredients=ingredients_dict,
            total_cost_per_kg=total_cost if total_cost > 0 else None,
            compliance=compliance,
            created_at=dna_data["created_at"]
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=ErrorResponse(
                error="VALIDATION_ERROR",
                message="Invalid request data",
                details={"validation_errors": e.errors()}
            ).model_dump()
        )
    except Exception as e:
        logger.error(f"Error creating DNA: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="INTERNAL_ERROR",
                message=f"Failed to create DNA: {str(e)}"
            ).model_dump()
        )


@app.post(
    "/evolve/options",
    response_model=EvolveOptionsResponse,
    tags=["Evolution"]
)
async def generate_evolution_options(request: EvolveOptionsRequest):
    """
    Generate evolution options from DNA and brief

    - Uses RLHF to generate variations
    - Returns N candidate options
    - Creates experiment session for tracking
    """
    try:
        # Check if DNA exists
        if request.dna_id not in DNA_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="DNA_NOT_FOUND",
                    message=f"DNA {request.dna_id} not found"
                ).model_dump()
            )

        # Get DNA data
        dna_data = DNA_STORAGE[request.dna_id]

        # Create OlfactoryDNA object for evolution service
        ingredients = []
        for ing_dict in dna_data["ingredients"]:
            ingredients.append(Ingredient(
                id=ing_dict["id"],
                name=ing_dict["name"],
                cas_number=ing_dict.get("cas_number"),
                concentration=ing_dict["concentration"],
                category=NoteCategory(ing_dict["category"]),
                cost_per_kg=ing_dict.get("cost_per_kg"),
                ifra_limit=ing_dict.get("ifra_limit")
            ))

        dna = OlfactoryDNA(
            dna_id=request.dna_id,
            name=dna_data["name"],
            ingredients=ingredients
        )

        # Create CreativeBrief object
        brief = CreativeBrief(
            brief_id=f"brief_{uuid.uuid4().hex[:8]}",
            user_id="api_user",  # Would come from auth in production
            **request.brief
        )

        # Get evolution service with specified algorithm
        evolution_service = get_evolution_service(algorithm=request.algorithm)

        # Generate options
        result = evolution_service.generate_options(
            user_id="api_user",
            dna=dna,
            brief=brief,
            num_options=request.num_options
        )

        # Calculate optimization scores if requested
        optimization_scores = None
        if request.optimization_profile:
            evaluator = TotalObjective(request.optimization_profile)

            # Convert first option to formula for evaluation
            if result["options"]:
                first_option = result["options"][0]
                # Get phenotype from session storage
                session = evolution_service.get_session_info(result["experiment_id"])
                if session and session["options"]:
                    phenotype_data = session["options"][0]["phenotype"]
                    formula = [
                        (ing["id"], ing["concentration"])
                        for ing in phenotype_data["adjusted_ingredients"]
                    ]
                    scores = evaluator.evaluate(formula)
                    optimization_scores = {
                        "creativity": scores["creativity"],
                        "fitness": scores["fitness"],
                        "stability": scores["stability"],
                        "total": scores["total"]
                    }

        # Store experiment
        experiment_data = {
            "experiment_id": result["experiment_id"],
            "dna_id": request.dna_id,
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "iterations": 0,
            "algorithm": request.algorithm,
            "options": result["options"],
            "optimization_profile": request.optimization_profile.value
        }
        EXPERIMENT_STORAGE[result["experiment_id"]] = experiment_data

        # Format options for response
        formatted_options = []
        for opt in result["options"]:
            # Get preview of changes
            preview = {}
            session = evolution_service.get_session_info(result["experiment_id"])
            if session and session["options"]:
                for session_opt in session["options"]:
                    if session_opt["id"] == opt["id"]:
                        phenotype = session_opt["phenotype"]
                        for ing in phenotype["adjusted_ingredients"][:3]:  # Top 3 changes
                            preview[ing["name"]] = round(ing["concentration"], 2)
                        break

            formatted_options.append({
                "id": opt["id"],
                "action": opt["action"],
                "description": opt["description"],
                "preview": preview
            })

        logger.info(f"Generated {len(formatted_options)} evolution options for experiment {result['experiment_id']}")

        return EvolveOptionsResponse(
            experiment_id=result["experiment_id"],
            options=formatted_options,
            optimization_scores=optimization_scores,
            created_at=experiment_data["created_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating evolution options: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="EVOLUTION_ERROR",
                message=f"Failed to generate options: {str(e)}"
            ).model_dump()
        )


@app.post(
    "/evolve/feedback",
    response_model=EvolveFeedbackResponse,
    tags=["Evolution"]
)
async def process_evolution_feedback(request: EvolveFeedbackRequest):
    """
    Process user feedback for RL update

    - Updates RL policy based on choice and rating
    - Advances experiment iteration
    - Returns update metrics
    """
    try:
        # Check if experiment exists
        if request.experiment_id not in EXPERIMENT_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="EXPERIMENT_NOT_FOUND",
                    message=f"Experiment {request.experiment_id} not found"
                ).model_dump()
            )

        # Get experiment data
        experiment = EXPERIMENT_STORAGE[request.experiment_id]

        # Check if experiment is active
        if experiment.get("status") != "active":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="EXPERIMENT_INACTIVE",
                    message=f"Experiment {request.experiment_id} is not active"
                ).model_dump()
            )

        # Validate chosen_id exists in options
        valid_ids = [opt["id"] for opt in experiment["options"]]
        if request.chosen_id not in valid_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="INVALID_OPTION",
                    message=f"Option {request.chosen_id} not found in experiment"
                ).model_dump()
            )

        # Get evolution service
        algorithm = experiment.get("algorithm", "PPO")
        evolution_service = get_evolution_service(algorithm=algorithm)

        # Process feedback
        result = evolution_service.process_feedback(
            experiment_id=request.experiment_id,
            chosen_id=request.chosen_id,
            rating=request.rating
        )

        # Update experiment storage
        experiment["iterations"] += 1
        experiment["last_feedback"] = {
            "chosen_id": request.chosen_id,
            "rating": request.rating,
            "notes": request.notes,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Log feedback
        logger.info(
            f"Processed feedback for experiment {request.experiment_id}: "
            f"chosen={request.chosen_id}, rating={request.rating}"
        )

        return EvolveFeedbackResponse(
            status="success",
            experiment_id=request.experiment_id,
            iteration=result.get("iteration", experiment["iterations"]),
            metrics=result.get("metrics", {}),
            message="Feedback processed and policy updated"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="FEEDBACK_ERROR",
                message=f"Failed to process feedback: {str(e)}"
            ).model_dump()
        )


@app.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentStatusResponse,
    tags=["Experiments"]
)
async def get_experiment_status(experiment_id: str):
    """
    Get experiment status and logs

    - Returns current experiment state
    - Includes iteration count and last feedback
    - Shows associated DNA and algorithm
    """
    try:
        # Check if experiment exists
        if experiment_id not in EXPERIMENT_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="EXPERIMENT_NOT_FOUND",
                    message=f"Experiment {experiment_id} not found"
                ).model_dump()
            )

        # Get experiment data
        experiment = EXPERIMENT_STORAGE[experiment_id]

        return ExperimentStatusResponse(
            experiment_id=experiment_id,
            status=experiment.get("status", "unknown"),
            created_at=experiment.get("created_at", ""),
            iterations=experiment.get("iterations", 0),
            last_feedback=experiment.get("last_feedback"),
            dna_id=experiment.get("dna_id", ""),
            algorithm=experiment.get("algorithm", "PPO")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting experiment status: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="STATUS_ERROR",
                message=f"Failed to get experiment status: {str(e)}"
            ).model_dump()
        )


@app.delete(
    "/experiments/{experiment_id}",
    tags=["Experiments"]
)
async def end_experiment(experiment_id: str):
    """
    End an experiment session

    - Marks experiment as completed
    - Cleans up evolution service session
    - Returns final statistics
    """
    try:
        # Check if experiment exists
        if experiment_id not in EXPERIMENT_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="EXPERIMENT_NOT_FOUND",
                    message=f"Experiment {experiment_id} not found"
                ).model_dump()
            )

        # Get experiment data
        experiment = EXPERIMENT_STORAGE[experiment_id]

        # End evolution service session
        algorithm = experiment.get("algorithm", "PPO")
        evolution_service = get_evolution_service(algorithm=algorithm)
        result = evolution_service.end_session(experiment_id)

        # Update experiment status
        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.utcnow().isoformat()

        logger.info(f"Ended experiment {experiment_id} after {experiment['iterations']} iterations")

        return {
            "status": "success",
            "experiment_id": experiment_id,
            "total_iterations": experiment["iterations"],
            "completed_at": experiment["completed_at"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending experiment: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="END_ERROR",
                message=f"Failed to end experiment: {str(e)}"
            ).model_dump()
        )


@app.get(
    "/dna/{dna_id}",
    tags=["DNA"]
)
async def get_dna(dna_id: str):
    """
    Get DNA by ID

    - Returns complete DNA information
    - Includes ingredients and compliance status
    """
    try:
        # Check if DNA exists
        if dna_id not in DNA_STORAGE:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=ErrorResponse(
                    error="DNA_NOT_FOUND",
                    message=f"DNA {dna_id} not found"
                ).model_dump()
            )

        return DNA_STORAGE[dna_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DNA: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="GET_ERROR",
                message=f"Failed to get DNA: {str(e)}"
            ).model_dump()
        )


@app.get(
    "/dna",
    tags=["DNA"]
)
async def list_dnas(limit: int = 10, offset: int = 0):
    """
    List all DNAs with pagination

    - Returns list of DNA summaries
    - Supports pagination with limit and offset
    """
    try:
        # Get all DNA IDs
        all_dna_ids = list(DNA_STORAGE.keys())

        # Apply pagination
        paginated_ids = all_dna_ids[offset:offset + limit]

        # Get DNA summaries
        dnas = []
        for dna_id in paginated_ids:
            dna_data = DNA_STORAGE[dna_id]
            dnas.append({
                "dna_id": dna_id,
                "name": dna_data["name"],
                "created_at": dna_data["created_at"],
                "ingredient_count": len(dna_data["ingredients"]),
                "compliant": dna_data["compliance"]["overall_compliant"]
            })

        return {
            "dnas": dnas,
            "total": len(all_dna_ids),
            "limit": limit,
            "offset": offset
        }

    except Exception as e:
        logger.error(f"Error listing DNAs: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="LIST_ERROR",
                message=f"Failed to list DNAs: {str(e)}"
            ).model_dump()
        )


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="VALIDATION_ERROR",
            message="Request validation failed",
            details={"validation_errors": exc.errors()}
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    # If detail is already ErrorResponse format, use it
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    # Otherwise, create ErrorResponse
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTP_ERROR",
            message=str(exc.detail)
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"type": type(exc).__name__, "message": str(exc)}
        ).model_dump()
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Fragrance AI API v2.0.0")

    # Initialize services
    try:
        evolution_service = get_evolution_service()
        logger.info("Evolution service initialized")

        ifra_checker = get_ifra_checker()
        logger.info("IFRA compliance checker initialized")

        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Fragrance AI API")

    # Save any pending data (in production, persist to database)
    logger.info(f"Saved {len(DNA_STORAGE)} DNAs and {len(EXPERIMENT_STORAGE)} experiments")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )