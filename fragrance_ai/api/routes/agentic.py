"""
Agentic API Routes - LLM Orchestrator Endpoints
- Provides unified API endpoints for the agentic fragrance system
- Routes all requests through the LLM Orchestrator
- Implements the wedding gift scenario and other specialized use cases
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
import time

from ...services.orchestrator_service import get_orchestrator_service, OrchestratorService
from ..middleware.auth_middleware import get_optional_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agentic", tags=["Agentic Fragrance System"])

# Request/Response Models
class FragranceRequestModel(BaseModel):
    """General fragrance request model."""
    query: str = Field(..., description="Natural language request for fragrance assistance")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context or preferences")
    language: Optional[str] = Field("auto", description="Response language preference (auto, korean, english)")


class RecommendationRequest(BaseModel):
    """Request for personalized perfume recommendations."""
    mood: str = Field(..., description="Desired mood or feeling")
    season: Optional[str] = Field(None, description="Season preference")
    gender: Optional[str] = Field(None, description="Gender preference")
    price_range: Optional[str] = Field(None, description="Price range preference")
    preferred_notes: Optional[List[str]] = Field(None, description="Preferred fragrance notes")
    occasion: Optional[str] = Field(None, description="Intended occasion or use")

class CustomFragranceRequest(BaseModel):
    """Request for custom fragrance creation."""
    concept: str = Field(..., description="Main concept or inspiration")
    inspiration: Optional[str] = Field(None, description="Specific inspiration source")
    perfumer_style: Optional[str] = Field(None, description="Preferred perfumer style")
    target_audience: Optional[str] = Field(None, description="Target audience")
    complexity: Optional[str] = Field("moderate", description="Desired complexity level")

class KnowledgeRequest(BaseModel):
    """Request for perfumery knowledge."""
    query_type: str = Field(..., description="Type of knowledge query (perfumer_style, accord_formula)")
    subject: str = Field(..., description="Name of perfumer or accord to query")

class AgenticResponse(BaseModel):
    """Unified response model for agentic system."""
    success: bool
    message: str
    request_id: str
    execution_time: float
    tools_used: List[str]
    complexity: str
    metadata: Dict[str, Any]

# Main Endpoints

@router.post("/request", response_model=AgenticResponse)
async def process_fragrance_request(
    request: FragranceRequestModel,
    background_tasks: BackgroundTasks,
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Universal endpoint for all fragrance-related requests.
    Routes through the LLM Orchestrator for agentic processing.
    """
    try:
        start_time = time.time()

        # Add user context if available
        user_context = request.context or {}
        if current_user:
            user_context.update({
                "user_id": current_user.get("user_id"),
                "user_role": current_user.get("role"),
                "user_preferences": current_user.get("preferences", {})
            })

        # Process through orchestrator
        result = await orchestrator_service.process_fragrance_request(
            user_query=request.query,
            user_context=user_context
        )

        return AgenticResponse(
            success=result["success"],
            message=result["message"],
            request_id=result["request_id"],
            execution_time=result["execution_time"],
            tools_used=result.get("tools_used", []),
            complexity=result.get("complexity", "unknown"),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Fragrance request processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations", response_model=AgenticResponse)
async def get_perfume_recommendations(
    request: RecommendationRequest,
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Get personalized perfume recommendations through the agentic system.
    """
    try:
        result = await orchestrator_service.get_perfume_recommendations(
            mood=request.mood,
            season=request.season,
            gender=request.gender,
            price_range=request.price_range,
            preferred_notes=request.preferred_notes
        )

        return AgenticResponse(
            success=result["success"],
            message=result["message"],
            request_id=result["request_id"],
            execution_time=result["execution_time"],
            tools_used=result.get("tools_used", []),
            complexity=result.get("complexity", "moderate"),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Recommendation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-custom", response_model=AgenticResponse)
async def create_custom_fragrance(
    request: CustomFragranceRequest,
    background_tasks: BackgroundTasks,
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Create a custom fragrance based on concept and inspiration.
    """
    try:
        result = await orchestrator_service.create_custom_fragrance(
            concept=request.concept,
            inspiration=request.inspiration,
            perfumer_style=request.perfumer_style,
            target_audience=request.target_audience
        )

        return AgenticResponse(
            success=result["success"],
            message=result["message"],
            request_id=result["request_id"],
            execution_time=result["execution_time"],
            tools_used=result.get("tools_used", []),
            complexity=result.get("complexity", "complex"),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Custom fragrance creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/perfumer-style", response_model=AgenticResponse)
async def analyze_perfumer_style(
    perfumer_name: str,
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service)
):
    """
    Analyze and explain a master perfumer's style.
    """
    try:
        result = await orchestrator_service.analyze_perfumer_style(perfumer_name)

        return AgenticResponse(
            success=result["success"],
            message=result["message"],
            request_id=result["request_id"],
            execution_time=result["execution_time"],
            tools_used=result.get("tools_used", []),
            complexity=result.get("complexity", "simple"),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Perfumer style analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/knowledge/accord", response_model=AgenticResponse)
async def explain_fragrance_accord(
    accord_name: str,
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service)
):
    """
    Explain a classic fragrance accord and its composition.
    """
    try:
        result = await orchestrator_service.explain_fragrance_accord(accord_name)

        return AgenticResponse(
            success=result["success"],
            message=result["message"],
            request_id=result["request_id"],
            execution_time=result["execution_time"],
            tools_used=result.get("tools_used", []),
            complexity=result.get("complexity", "simple"),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Accord explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat-style endpoint for conversational interactions
@router.post("/chat")
async def chat_with_artisan(
    message: str,
    session_id: Optional[str] = None,
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Chat-style interface with Artisan, the AI Perfumer.
    Maintains conversation context and provides natural language interaction.
    """
    try:
        # Create context for chat interaction
        chat_context = {
            "interaction_type": "chat",
            "session_id": session_id
        }

        if current_user:
            chat_context.update({
                "user_id": current_user.get("user_id"),
                "user_name": current_user.get("username")
            })

        result = await orchestrator_service.process_fragrance_request(
            user_query=message,
            user_context=chat_context
        )

        return {
            "response": result["message"],
            "request_id": result["request_id"],
            "session_id": chat_context.get("session_id"),
            "tools_used": result.get("tools_used", []),
            "execution_time": result["execution_time"]
        }

    except Exception as e:
        logger.error(f"Chat interaction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System status and monitoring endpoints
@router.get("/status")
async def get_system_status(
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service)
):
    """
    Get the current status of the agentic system.
    """
    try:
        status = await orchestrator_service.get_service_status()
        return {
            "status": "operational",
            "system_info": status,
            "timestamp": time.time()
        }

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@router.get("/tools")
async def list_available_tools(
    orchestrator_service: OrchestratorService = Depends(get_orchestrator_service)
):
    """
    List all available tools in the agentic system.
    """
    try:
        status = await orchestrator_service.get_service_status()
        return {
            "available_tools": status.get("available_tools", []),
            "tool_descriptions": {
                "hybrid_search": "Search existing perfumes with semantic and metadata filtering",
                "validate_composition": "Scientific analysis of fragrance compositions",
                "query_knowledge_base": "Access master perfumer styles and classic accords"
            }
        }

    except Exception as e:
        logger.error(f"Tool listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

