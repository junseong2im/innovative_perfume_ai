"""
Orchestrator Service - Integration layer between FastAPI and LLM Orchestrator
- Provides service-level interface for the agentic system
- Handles request preprocessing and response formatting
- Manages orchestrator lifecycle and dependencies
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..orchestrator.orchestrator import LLMOrchestrator, OrchestratorResponse
from ..core.exceptions_unified import handle_exceptions_async
from ..core.config import settings

logger = logging.getLogger(__name__)

class OrchestratorService:
    """
    Service layer for the LLM Orchestrator.
    Provides high-level interface for agentic fragrance operations.
    """

    def __init__(self):
        self.orchestrator = LLMOrchestrator()
        self.initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the orchestrator service and all dependencies."""
        async with self._lock:
            if not self.initialized:
                try:
                    # Initialize any additional dependencies here
                    # The individual tools handle their own initialization
                    self.initialized = True
                    logger.info("OrchestratorService initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize OrchestratorService: {e}")
                    raise

    async def process_fragrance_request(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a fragrance-related request through the agentic system.

        Args:
            user_query: The user's natural language request
            user_context: Additional context (user preferences, session data, etc.)

        Returns:
            Formatted response with fragrance recommendations or creations
        """
        if not self.initialized:
            await self.initialize()

        try:
            # Preprocess the request
            processed_context = await self._preprocess_request(user_query, user_context)

            # Route through the orchestrator
            orchestrator_response = await self.orchestrator.process_request(
                user_query=user_query,
                context=processed_context
            )

            # Format response for API consumption
            formatted_response = await self._format_response(orchestrator_response)

            return formatted_response

        except Exception as e:
            logger.error(f"Fragrance request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "향수 요청 처리 중 오류가 발생했습니다. 다시 시도해 주세요."
            }


    async def get_perfume_recommendations(
        self,
        mood: str,
        season: Optional[str] = None,
        gender: Optional[str] = None,
        price_range: Optional[str] = None,
        preferred_notes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get personalized perfume recommendations.
        """
        # Construct recommendation query
        query_parts = [f"{mood} 무드의 향수를 추천해 주세요"]

        if season:
            query_parts.append(f"{season} 계절에 어울리는")
        if gender:
            query_parts.append(f"{gender}용")
        if price_range:
            query_parts.append(f"{price_range} 가격대의")
        if preferred_notes:
            query_parts.append(f"다음 노트들이 포함된: {', '.join(preferred_notes)}")

        recommendation_query = " ".join(query_parts)

        recommendation_context = {
            "scenario_type": "recommendation",
            "mood": mood,
            "season": season,
            "gender": gender,
            "price_range": price_range,
            "preferred_notes": preferred_notes
        }

        return await self.process_fragrance_request(recommendation_query, recommendation_context)

    async def create_custom_fragrance(
        self,
        concept: str,
        inspiration: Optional[str] = None,
        perfumer_style: Optional[str] = None,
        target_audience: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a custom fragrance based on concept and inspiration.
        """
        creation_query = f"{concept} 컨셉의 향수를 창조해 주세요"

        if inspiration:
            creation_query += f". 영감: {inspiration}"
        if perfumer_style:
            creation_query += f". {perfumer_style} 스타일로"
        if target_audience:
            creation_query += f". {target_audience}를 대상으로"

        creation_context = {
            "scenario_type": "custom_creation",
            "concept": concept,
            "inspiration": inspiration,
            "perfumer_style": perfumer_style,
            "target_audience": target_audience
        }

        return await self.process_fragrance_request(creation_query, creation_context)

    async def analyze_perfumer_style(self, perfumer_name: str) -> Dict[str, Any]:
        """
        Analyze and explain a master perfumer's style.
        """
        style_query = f"{perfumer_name}의 조향 스타일에 대해 자세히 설명해 주세요"

        style_context = {
            "scenario_type": "style_analysis",
            "perfumer_name": perfumer_name
        }

        return await self.process_fragrance_request(style_query, style_context)

    async def explain_fragrance_accord(self, accord_name: str) -> Dict[str, Any]:
        """
        Explain a classic fragrance accord and its composition.
        """
        accord_query = f"{accord_name} 조화의 구성과 특징에 대해 설명해 주세요"

        accord_context = {
            "scenario_type": "accord_explanation",
            "accord_name": accord_name
        }

        return await self.process_fragrance_request(accord_query, accord_context)

    async def _preprocess_request(
        self,
        user_query: str,
        user_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Preprocess the user request and context.
        """
        processed_context = user_context or {}

        # Add system context
        processed_context.update({
            "timestamp": datetime.utcnow().isoformat(),
            "system_version": "2.0-agentic",
            "language": "korean" if self._is_korean_query(user_query) else "english"
        })

        # Extract any implicit preferences from the query
        implicit_prefs = await self._extract_implicit_preferences(user_query)
        processed_context.update(implicit_prefs)

        return processed_context

    async def _format_response(self, orchestrator_response: OrchestratorResponse) -> Dict[str, Any]:
        """
        Format the orchestrator response for API consumption.
        """
        if not orchestrator_response.success:
            return {
                "success": False,
                "message": orchestrator_response.response,
                "error": orchestrator_response.metadata.get("error"),
                "request_id": orchestrator_response.request_id
            }

        # Extract tool results for additional information
        tool_results = {}
        for tool_exec in orchestrator_response.tool_executions:
            if tool_exec.success:
                tool_results[tool_exec.tool_name] = {
                    "execution_time": tool_exec.execution_time,
                    "result_summary": self._summarize_tool_result(tool_exec.result)
                }

        return {
            "success": True,
            "message": orchestrator_response.response,
            "request_id": orchestrator_response.request_id,
            "execution_time": orchestrator_response.total_execution_time,
            "tools_used": orchestrator_response.metadata.get("tools_used", []),
            "complexity": orchestrator_response.execution_plan.complexity_level if orchestrator_response.execution_plan else "unknown",
            "tool_results": tool_results,
            "metadata": {
                "session_id": orchestrator_response.metadata.get("session_id"),
                "timestamp": orchestrator_response.metadata.get("timestamp"),
                "iterations": orchestrator_response.metadata.get("iterations", 0)
            }
        }

    def _is_korean_query(self, query: str) -> bool:
        """Check if the query contains Korean characters."""
        return any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in query)

    async def _extract_implicit_preferences(self, query: str) -> Dict[str, Any]:
        """Extract implicit preferences from the user query."""
        preferences = {}

        query_lower = query.lower()

        # Seasonal preferences
        if any(season in query_lower for season in ["봄", "spring"]):
            preferences["season_preference"] = "spring"
        elif any(season in query_lower for season in ["여름", "summer"]):
            preferences["season_preference"] = "summer"
        elif any(season in query_lower for season in ["가을", "autumn", "fall"]):
            preferences["season_preference"] = "autumn"
        elif any(season in query_lower for season in ["겨울", "winter"]):
            preferences["season_preference"] = "winter"

        # Gender preferences
        if any(gender in query_lower for gender in ["남성", "men", "masculine"]):
            preferences["gender_preference"] = "masculine"
        elif any(gender in query_lower for gender in ["여성", "women", "feminine"]):
            preferences["gender_preference"] = "feminine"

        # Intensity preferences
        if any(intensity in query_lower for intensity in ["가벼운", "light", "subtle"]):
            preferences["intensity_preference"] = "light"
        elif any(intensity in query_lower for intensity in ["강한", "strong", "bold"]):
            preferences["intensity_preference"] = "strong"

        # Occasion preferences
        if any(occasion in query_lower for occasion in ["결혼", "wedding", "marriage"]):
            preferences["occasion"] = "wedding"
        elif any(occasion in query_lower for occasion in ["데이트", "date", "romantic"]):
            preferences["occasion"] = "date"
        elif any(occasion in query_lower for occasion in ["직장", "office", "work"]):
            preferences["occasion"] = "office"

        return preferences

    def _summarize_tool_result(self, result) -> str:
        """Create a brief summary of tool execution results."""
        if isinstance(result, list):
            return f"{len(result)} items returned"
        elif hasattr(result, 'harmony_score'):
            return f"Harmony: {result.harmony_score:.2f}, Stability: {result.stability_score:.2f}"
        elif hasattr(result, 'name'):
            return f"Knowledge about: {result.name}"
        else:
            return "Result returned successfully"

    async def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the orchestrator service."""
        return {
            "service_name": "OrchestratorService",
            "initialized": self.initialized,
            "orchestrator_session": self.orchestrator.session_id if self.orchestrator else None,
            "conversation_history_length": len(self.orchestrator.conversation_history) if self.orchestrator else 0,
            "available_tools": list(self.orchestrator.tool_registry.keys()) if self.orchestrator else [],
            "system_version": "2.0-agentic"
        }

# Global service instance
_orchestrator_service = None

async def get_orchestrator_service() -> OrchestratorService:
    """Dependency function for FastAPI injection."""
    global _orchestrator_service
    if _orchestrator_service is None:
        _orchestrator_service = OrchestratorService()
        await _orchestrator_service.initialize()
    return _orchestrator_service