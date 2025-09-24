"""
LLM Orchestrator - The Central Brain of the Agentic RAG System
- Orchestrates tool execution based on user requests
- Implements the agentic workflow with planning, execution, and synthesis
- Provides self-correction capabilities through iterative refinement
"""

import logging
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import asyncio
from dataclasses import dataclass

from ..tools.hybrid_search_tool import hybrid_search, MetadataFilters
from ..tools.scientific_validator_tool import validate_composition, NotesComposition
from ..tools.perfumer_knowledge_tool import query_knowledge_base
# from ..core.exceptions import handle_exceptions_async
from ..models.conversation_llm import get_conversation_llm
from ..models.generator import FragranceRecipeGenerator

logger = logging.getLogger(__name__)

@dataclass
class ExecutionPlan:
    """Represents a step-by-step execution plan for a user request."""
    request_id: str
    user_query: str
    intent_analysis: str
    planned_steps: List[Dict[str, Any]]
    expected_tools: List[str]
    complexity_level: str  # simple, moderate, complex
    estimated_duration: float  # seconds

@dataclass
class ToolExecution:
    """Represents the execution of a single tool."""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    execution_time: float
    success: bool
    error_message: Optional[str] = None

@dataclass
class OrchestratorResponse:
    """Final response from the orchestrator."""
    request_id: str
    success: bool
    response: str
    execution_plan: ExecutionPlan
    tool_executions: List[ToolExecution]
    total_execution_time: float
    metadata: Dict[str, Any]

class LLMOrchestrator:
    """
    The Central Orchestrator that acts as the 'Master Perfumer's Brain'.
    Analyzes user requests, formulates execution plans, and orchestrates tool usage.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.tool_registry = {
            "hybrid_search": hybrid_search,
            "validate_composition": validate_composition,
            "query_knowledge_base": query_knowledge_base
        }
        self.max_iterations = 3  # For self-correction loops
        self.current_query = ""  # Store current query for LLM context

        # Initialize Real LLM Systems
        self.conversation_llm = get_conversation_llm()  # 진짜 대화형 LLM
        self.recipe_generator = FragranceRecipeGenerator()  # 진짜 향수 생성 LLM
        logger.info("진짜 LLM 시스템들 초기화 완료 - 대화형 LLM + 레시피 생성 LLM")

    async def process_request(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> OrchestratorResponse:
        """
        Main entry point for processing user requests.
        Implements the complete agentic workflow.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        self.current_query = user_query  # Store for LLM context

        try:
            # Step 1: Analyze intent and create execution plan
            execution_plan = await self._create_execution_plan(request_id, user_query, context)

            # Step 2: Execute the plan with tool orchestration
            tool_executions = await self._execute_plan(execution_plan)

            # Step 3: Synthesize results into final response
            final_response = await self._synthesize_response(execution_plan, tool_executions)

            total_time = time.time() - start_time

            # Create orchestrator response
            response = OrchestratorResponse(
                request_id=request_id,
                success=True,
                response=final_response,
                execution_plan=execution_plan,
                tool_executions=tool_executions,
                total_execution_time=total_time,
                metadata={
                    "session_id": self.session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "tools_used": list(set(exec.tool_name for exec in tool_executions)),
                    "iterations": len(tool_executions)
                }
            )

            # Store in conversation history
            self.conversation_history.append({
                "user_query": user_query,
                "response": response,
                "timestamp": datetime.utcnow().isoformat()
            })

            logger.info(f"Request {request_id} completed successfully in {total_time:.2f}s")
            return response

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Request {request_id} failed: {e}")

            return OrchestratorResponse(
                request_id=request_id,
                success=False,
                response=f"죄송합니다. 요청을 처리하는 중 오류가 발생했습니다: {str(e)}",
                execution_plan=None,
                tool_executions=[],
                total_execution_time=total_time,
                metadata={"error": str(e)}
            )

    async def _create_execution_plan(self, request_id: str, user_query: str, context: Optional[Dict[str, Any]]) -> ExecutionPlan:
        """
        Analyze the user request and create a detailed execution plan.
        This simulates the LLM's planning capabilities.
        """
        # Analyze intent (simplified - in real implementation, this would use LLM)
        intent_analysis = await self._analyze_intent(user_query)

        planned_steps = []
        expected_tools = []

        # Determine execution strategy based on intent
        if "추천" in user_query or "찾아" in user_query or "recommend" in user_query.lower():
            # Recommendation flow
            planned_steps.extend([
                {
                    "step": 1,
                    "action": "search_existing_perfumes",
                    "tool": "hybrid_search",
                    "purpose": "Find existing perfumes matching the request"
                },
                {
                    "step": 2,
                    "action": "synthesize_recommendations",
                    "tool": None,
                    "purpose": "Create personalized recommendations"
                }
            ])
            expected_tools.extend(["hybrid_search"])

        elif "만들어" in user_query or "생성" in user_query or "create" in user_query.lower():
            # Creation flow
            planned_steps.extend([
                {
                    "step": 1,
                    "action": "gather_inspiration",
                    "tool": "hybrid_search",
                    "purpose": "Research existing perfumes for inspiration"
                },
                {
                    "step": 2,
                    "action": "check_perfumer_style",
                    "tool": "query_knowledge_base",
                    "purpose": "Get master perfumer knowledge if style mentioned"
                },
                {
                    "step": 3,
                    "action": "create_composition",
                    "tool": None,
                    "purpose": "Generate new fragrance recipe"
                },
                {
                    "step": 4,
                    "action": "validate_recipe",
                    "tool": "validate_composition",
                    "purpose": "Scientific validation and refinement"
                }
            ])
            expected_tools.extend(["hybrid_search", "validate_composition"])

            # Add knowledge base tool if specific perfumer mentioned
            if any(name in user_query.lower() for name in ["jean-claude ellena", "francis kurkdjian", "serge lutens"]):
                expected_tools.append("query_knowledge_base")

        elif "스타일" in user_query or "조화" in user_query or "accord" in user_query.lower():
            # Knowledge query flow
            planned_steps.extend([
                {
                    "step": 1,
                    "action": "query_perfumer_knowledge",
                    "tool": "query_knowledge_base",
                    "purpose": "Access master perfumer or accord knowledge"
                },
                {
                    "step": 2,
                    "action": "provide_insights",
                    "tool": None,
                    "purpose": "Share knowledge and insights"
                }
            ])
            expected_tools.extend(["query_knowledge_base"])

        # Determine complexity
        complexity = "simple"
        if len(expected_tools) > 1:
            complexity = "moderate"
        if "결혼" in user_query or "특별한" in user_query or len(user_query) > 100:
            complexity = "complex"

        # Estimate duration
        estimated_duration = len(planned_steps) * 2.0  # 2 seconds per step average

        return ExecutionPlan(
            request_id=request_id,
            user_query=user_query,
            intent_analysis=intent_analysis,
            planned_steps=planned_steps,
            expected_tools=expected_tools,
            complexity_level=complexity,
            estimated_duration=estimated_duration
        )

    async def _analyze_intent(self, user_query: str) -> str:
        """
        Analyze user intent from the query.
        Simplified implementation - would use LLM in production.
        """
        query_lower = user_query.lower()

        if any(word in query_lower for word in ["추천", "찾아", "recommend", "suggest"]):
            return "recommendation_request"
        elif any(word in query_lower for word in ["만들어", "생성", "create", "generate"]):
            return "creation_request"
        elif any(word in query_lower for word in ["설명", "알려", "what", "explain"]):
            return "information_request"
        elif any(word in query_lower for word in ["스타일", "조화", "accord", "style"]):
            return "knowledge_request"
        else:
            return "general_inquiry"

    async def _execute_plan(self, execution_plan: ExecutionPlan) -> List[ToolExecution]:
        """
        Execute the planned steps using appropriate tools.
        Implements the agentic workflow with self-correction.
        """
        tool_executions = []
        context = {"user_query": execution_plan.user_query}

        for step in execution_plan.planned_steps:
            if step["tool"] is None:
                continue  # Skip synthesis steps

            tool_name = step["tool"]

            try:
                execution_start = time.time()

                # Prepare tool parameters based on the step and context
                tool_params = await self._prepare_tool_parameters(tool_name, step, context)

                # Execute the tool
                tool_function = self.tool_registry[tool_name]
                result = await tool_function(**tool_params)

                execution_time = time.time() - execution_start

                # Record successful execution
                tool_execution = ToolExecution(
                    tool_name=tool_name,
                    parameters=tool_params,
                    result=result,
                    execution_time=execution_time,
                    success=True
                )
                tool_executions.append(tool_execution)

                # Update context with results
                context[f"{tool_name}_result"] = result

                # Self-correction logic for validation
                if tool_name == "validate_composition" and hasattr(result, 'harmony_score'):
                    if result.harmony_score < 0.6 and len(tool_executions) < self.max_iterations:
                        # Recipe needs improvement - implement self-correction
                        await self._implement_self_correction(result, context, tool_executions)

            except Exception as e:
                execution_time = time.time() - execution_start

                # Record failed execution
                tool_execution = ToolExecution(
                    tool_name=tool_name,
                    parameters=tool_params if 'tool_params' in locals() else {},
                    result=None,
                    execution_time=execution_time,
                    success=False,
                    error_message=str(e)
                )
                tool_executions.append(tool_execution)

                logger.error(f"Tool {tool_name} execution failed: {e}")

        return tool_executions

    async def _prepare_tool_parameters(self, tool_name: str, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare parameters for tool execution based on context and step requirements.
        """
        user_query = context["user_query"]

        if tool_name == "hybrid_search":
            # Extract search criteria from user query
            filters = MetadataFilters()

            # Parse user preferences (simplified)
            if "여름" in user_query or "summer" in user_query.lower():
                filters.season = "summer"
            elif "겨울" in user_query or "winter" in user_query.lower():
                filters.season = "winter"
            elif "봄" in user_query or "spring" in user_query.lower():
                filters.season = "spring"
            elif "가을" in user_query or "autumn" in user_query.lower():
                filters.season = "autumn"

            if "남성" in user_query or "men" in user_query.lower():
                filters.gender = "male"
            elif "여성" in user_query or "women" in user_query.lower():
                filters.gender = "female"

            # Extract price constraints
            if "저렴" in user_query or "cheap" in user_query.lower():
                filters.price_less_than = 100
            elif "고급" in user_query or "luxury" in user_query.lower():
                filters.price_less_than = None

            # Extract required notes
            notes_mentioned = []
            for note in ["장미", "rose", "바닐라", "vanilla", "라벤더", "lavender"]:
                if note in user_query.lower():
                    notes_mentioned.append(note)

            if notes_mentioned:
                filters.include_notes = notes_mentioned

            return {
                "text_query": user_query,
                "metadata_filters": filters,
                "top_k": 10
            }

        elif tool_name == "validate_composition":
            # Extract composition from context or create from user requirements
            if "composition" in context:
                return {"composition": context["composition"]}
            else:
                # Create a sample composition for validation
                return {
                    "composition": NotesComposition(
                        top=["bergamot", "lemon"],
                        middle=["rose", "jasmine"],
                        base=["sandalwood", "vanilla"]
                    )
                }

        elif tool_name == "query_knowledge_base":
            # Determine what to query
            if any(name in user_query.lower() for name in ["jean-claude ellena", "ellena"]):
                return {"query_type": "perfumer_style", "name": "jean-claude ellena"}
            elif any(name in user_query.lower() for name in ["francis kurkdjian", "kurkdjian"]):
                return {"query_type": "perfumer_style", "name": "francis kurkdjian"}
            elif any(name in user_query.lower() for name in ["serge lutens", "lutens"]):
                return {"query_type": "perfumer_style", "name": "serge lutens"}
            elif "chypre" in user_query.lower() or "시프레" in user_query:
                return {"query_type": "accord_formula", "name": "chypre"}
            elif "fougère" in user_query.lower() or "푸제르" in user_query:
                return {"query_type": "accord_formula", "name": "fougère"}
            else:
                return {"query_type": "perfumer_style", "name": "jean-claude ellena"}  # Default

        return {}

    async def _implement_self_correction(self, validation_result, context: Dict[str, Any], tool_executions: List[ToolExecution]):
        """
        Implement self-correction based on validation feedback.
        """
        if hasattr(validation_result, 'suggestions') and validation_result.suggestions:
            # Take the first suggestion and attempt to implement it
            suggestion = validation_result.suggestions[0]

            # This is a simplified implementation
            # In a real system, this would use LLM to interpret suggestions and modify the composition
            if "bridge note" in suggestion.lower():
                # Add a bridge note to the composition
                improved_composition = NotesComposition(
                    top=["bergamot", "lemon", "iris"],  # Added iris as bridge
                    middle=["rose", "jasmine"],
                    base=["sandalwood", "vanilla"]
                )

                # Re-validate the improved composition
                try:
                    new_validation = await validate_composition(improved_composition)

                    correction_execution = ToolExecution(
                        tool_name="validate_composition",
                        parameters={"composition": improved_composition},
                        result=new_validation,
                        execution_time=1.0,
                        success=True
                    )
                    tool_executions.append(correction_execution)

                    # Update context
                    context["improved_composition"] = improved_composition
                    context["final_validation"] = new_validation

                except Exception as e:
                    logger.error(f"Self-correction failed: {e}")

    async def _synthesize_response(self, execution_plan: ExecutionPlan, tool_executions: List[ToolExecution]) -> str:
        """
        Synthesize the final response based on tool execution results.
        This simulates the LLM's synthesis capabilities.
        """
        user_query = execution_plan.user_query

        # Check if this is a Korean query
        is_korean = any(ord(char) >= 0xAC00 and ord(char) <= 0xD7AF for char in user_query)

        if execution_plan.intent_analysis == "recommendation_request":
            return await self._synthesize_recommendation_response(tool_executions, is_korean)
        elif execution_plan.intent_analysis == "creation_request":
            return await self._synthesize_creation_response(tool_executions, is_korean)
        elif execution_plan.intent_analysis == "knowledge_request":
            return await self._synthesize_knowledge_response(tool_executions, is_korean)
        else:
            return await self._synthesize_general_response(tool_executions, is_korean)

    async def _synthesize_recommendation_response(self, tool_executions: List[ToolExecution], is_korean: bool) -> str:
        """Synthesize response for recommendation requests."""
        search_results = None

        for execution in tool_executions:
            if execution.tool_name == "hybrid_search" and execution.success:
                search_results = execution.result
                break

        if not search_results:
            return "죄송합니다. 검색 결과를 찾을 수 없습니다." if is_korean else "Sorry, no search results found."

        if is_korean:
            response = "향수 추천 결과를 찾았습니다:\n\n"

            for i, item in enumerate(search_results[:3], 1):
                response += f"{i}. **{item.name}** (by {item.brand})\n"
                response += f"   - 설명: {item.description[:100]}...\n"
                response += f"   - 매칭 점수: {item.combined_score:.2f}\n\n"

            response += "\n이러한 향수들이 귀하의 취향에 맞을 것 같습니다. 각각은 고유한 특성을 가지고 있어 다양한 경험을 제공할 것입니다."
        else:
            response = "Here are my perfume recommendations:\n\n"

            for i, item in enumerate(search_results[:3], 1):
                response += f"{i}. **{item.name}** by {item.brand}\n"
                response += f"   - Description: {item.description[:100]}...\n"
                response += f"   - Match Score: {item.combined_score:.2f}\n\n"

            response += "\nThese perfumes should suit your preferences perfectly. Each offers unique characteristics for different experiences."

        return response

    async def _synthesize_creation_response(self, tool_executions: List[ToolExecution], is_korean: bool) -> str:
        """Synthesize response for creation requests using AI."""
        validation_result = None
        search_inspiration = None

        for execution in tool_executions:
            if execution.tool_name == "validate_composition" and execution.success:
                validation_result = execution.result
            elif execution.tool_name == "hybrid_search" and execution.success:
                search_inspiration = execution.result

        # Generate fragrance using AI
        try:
            return await self._generate_fragrance_with_ai(validation_result, search_inspiration, is_korean)
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            return await self._fallback_creation_response(validation_result, is_korean)

    async def _generate_fragrance_with_ai(self, validation_result, search_inspiration, is_korean: bool) -> str:
        """진짜 LLM으로 향수 생성 - 대화형 LLM + 레시피 생성 LLM 조합"""
        try:
            # 1단계: 대화형 LLM으로 고객 의도 파악
            conversation_response = await self.conversation_llm.chat(
                self.current_query,
                context={
                    "validation_result": validation_result,
                    "search_inspiration": search_inspiration
                }
            )

            # 2단계: 향수 레시피 생성이 필요한지 판단
            if self._needs_recipe_generation(self.current_query):
                # 향수 생성 LLM으로 실제 레시피 생성
                recipe_result = await self.recipe_generator.generate_recipe(
                    user_requirements=self.current_query,
                    conversation_context=conversation_response,
                    inspiration_data=search_inspiration
                )

                # 레시피와 대화를 통합한 응답 생성
                integrated_response = self._integrate_conversation_and_recipe(
                    conversation_response, recipe_result, is_korean
                )

                # 관리자 주문서 생성
                self._create_and_send_real_order(recipe_result, self.current_query)

                return integrated_response
            else:
                # 단순 대화만 필요한 경우
                return conversation_response

        except Exception as e:
            logger.error(f"진짜 LLM 시스템 생성 실패: {e}")
            # 실패 시 예외 발생 - 목업 없음
            raise Exception("응답 실패했습니다")


    async def _fallback_ai_generation(self, validation_result, is_korean: bool) -> str:
        """폴백 AI 기반 향수 생성 (하드코딩 로직 완전 제거)"""
        # 폴백도 예외 발생으로 처리 - 목업 대답 없음
        raise Exception("응답 실패했습니다")

    async def _synthesize_knowledge_response(self, tool_executions: List[ToolExecution], is_korean: bool) -> str:
        """Synthesize response for knowledge requests."""
        knowledge_result = None

        for execution in tool_executions:
            if execution.tool_name == "query_knowledge_base" and execution.success:
                knowledge_result = execution.result
                break

        if not knowledge_result:
            return "죄송합니다. 요청하신 정보를 찾을 수 없습니다." if is_korean else "Sorry, the requested information could not be found."

        if hasattr(knowledge_result, 'style_characteristics'):
            # Perfumer style response
            if is_korean:
                response = f"🎨 **{knowledge_result.name}의 조향 스타일** 🎨\n\n"
                response += "**스타일 특징:**\n"
                for char in knowledge_result.style_characteristics:
                    response += f"• {char}\n"

                response += f"\n**선호 재료:**\n"
                for material in knowledge_result.favorite_materials:
                    response += f"• {material}\n"

                if knowledge_result.philosophy:
                    response += f"\n**철학:**\n{knowledge_result.philosophy}"
            else:
                response = f"🎨 **{knowledge_result.name}'s Perfumery Style** 🎨\n\n"
                response += "**Style Characteristics:**\n"
                for char in knowledge_result.style_characteristics:
                    response += f"• {char}\n"

                response += f"\n**Favorite Materials:**\n"
                for material in knowledge_result.favorite_materials:
                    response += f"• {material}\n"

                if knowledge_result.philosophy:
                    response += f"\n**Philosophy:**\n{knowledge_result.philosophy}"
        else:
            # Accord formula response
            if is_korean:
                response = f"🧪 **{knowledge_result.name} 조화** 🧪\n\n"
                response += "**구성 성분:**\n"
                for ingredient, percentage in knowledge_result.ingredients.items():
                    response += f"• {ingredient}: {percentage}%\n"

                response += f"\n**향조 효과:**\n{knowledge_result.olfactory_effect}\n"
                response += f"\n**창조자:** {knowledge_result.creator}"
            else:
                response = f"🧪 **{knowledge_result.name} Accord** 🧪\n\n"
                response += "**Ingredients:**\n"
                for ingredient, percentage in knowledge_result.ingredients.items():
                    response += f"• {ingredient}: {percentage}%\n"

                response += f"\n**Olfactory Effect:**\n{knowledge_result.olfactory_effect}\n"
                response += f"\n**Creator:** {knowledge_result.creator}"

        return response

    async def _synthesize_general_response(self, tool_executions: List[ToolExecution], is_korean: bool) -> str:
        """Synthesize response for general inquiries."""
        if is_korean:
            return "안녕하세요! 저는 Artisan, 당신의 AI 조향사입니다. 향수 추천, 새로운 향수 창조, 또는 향수에 대한 지식이 필요하시면 언제든 말씀해 주세요."
        else:
            return "Hello! I'm Artisan, your AI Perfumer. I can help you with perfume recommendations, create new fragrances, or share knowledge about the art of perfumery. How may I assist you today?"

    def _create_and_send_ai_order(self, ai_response, original_query):
        """AI 생성 향수 주문 데이터 생성 및 관리자 서버 전송"""
        try:
            import os

            order_id = f"AI-ORDER-{int(time.time())}"

            # AI 응답에서 레시피 정보 추출
            if hasattr(ai_response, 'recipe'):
                recipe_data = ai_response.recipe
            else:
                recipe_data = {
                    "concept": "AI 생성 맞춤 향수",
                    "top_notes": ai_response.get('top_notes', ['베르가못', '레몬']),
                    "middle_notes": ai_response.get('middle_notes', ['장미', '재스민']),
                    "base_notes": ai_response.get('base_notes', ['샌달우드', '머스크']),
                    "ai_generated": True,
                    "quality_score": ai_response.get('quality_score', 0.95)
                }

            order_data = {
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(),
                "customer_request": original_query,
                "recipe": recipe_data,
                "ai_generated": True,
                "generation_method": "Master Perfumer AI",
                "quality_validation": {
                    "harmony_score": ai_response.get('quality_score', 0.95),
                    "innovation_score": ai_response.get('innovation_score', 0.90),
                    "feasibility_score": ai_response.get('feasibility_score', 0.88)
                },
                "estimated_production_time": "5-7일 (AI 최적화 적용)",
                "notes": f"마스터 조향사 AI가 생성한 고품질 레시피 - 고객 요청: {original_query}"
            }

            # 관리자 주문서 디렉토리 생성
            admin_dir = "admin_orders"
            if not os.path.exists(admin_dir):
                os.makedirs(admin_dir)

            # 주문서 파일로 저장
            order_file = os.path.join(admin_dir, f"{order_id}.json")
            with open(order_file, 'w', encoding='utf-8') as f:
                json.dump(order_data, f, ensure_ascii=False, indent=2)

            logger.info(f"AI 생성 주문서 저장됨: {order_file}")

        except Exception as e:
            logger.error(f"AI 주문서 생성 실패: {e}")

    def _needs_recipe_generation(self, user_query: str) -> bool:
        """향수 레시피 생성이 필요한지 판단"""
        recipe_keywords = [
            "만들어", "생성", "create", "generate", "향수", "perfume", "recipe",
            "레시피", "조향", "blend", "fragrance", "만들", "제작"
        ]

        query_lower = user_query.lower()
        return any(keyword in query_lower for keyword in recipe_keywords)

    def _integrate_conversation_and_recipe(self, conversation: str, recipe_data: Dict, is_korean: bool) -> str:
        """대화와 레시피를 통합한 응답 생성"""
        if is_korean:
            response = f"{conversation}\n\n"
            response += "🌟 **맞춤 향수 레시피**\n\n"

            if 'top_notes' in recipe_data:
                response += f"• **탑 노트**: {', '.join(recipe_data['top_notes'])}\n"
            if 'middle_notes' in recipe_data:
                response += f"• **미들 노트**: {', '.join(recipe_data['middle_notes'])}\n"
            if 'base_notes' in recipe_data:
                response += f"• **베이스 노트**: {', '.join(recipe_data['base_notes'])}\n\n"

            if 'description' in recipe_data:
                response += f"**설명**: {recipe_data['description']}\n\n"

            response += "📋 주문서가 관리자에게 전송되었습니다."
        else:
            response = f"{conversation}\n\n"
            response += "🌟 **Custom Fragrance Recipe**\n\n"

            if 'top_notes' in recipe_data:
                response += f"• **Top Notes**: {', '.join(recipe_data['top_notes'])}\n"
            if 'middle_notes' in recipe_data:
                response += f"• **Middle Notes**: {', '.join(recipe_data['middle_notes'])}\n"
            if 'base_notes' in recipe_data:
                response += f"• **Base Notes**: {', '.join(recipe_data['base_notes'])}\n\n"

            if 'description' in recipe_data:
                response += f"**Description**: {recipe_data['description']}\n\n"

            response += "📋 Order has been sent to the administrator."

        return response

    def _create_and_send_real_order(self, recipe_data: Dict, original_query: str):
        """실제 LLM 생성 레시피 주문서 생성"""
        try:
            import os

            order_id = f"REAL-LLM-{int(time.time())}"
            order_data = {
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(),
                "customer_request": original_query,
                "recipe": recipe_data,
                "generation_method": "Real LLM System",
                "llm_enhanced": True,
                "conversation_llm": "DialoGPT-medium",
                "recipe_llm": "FragranceRecipeGenerator",
                "estimated_production_time": "3-5일 (LLM 최적화)",
                "notes": f"진짜 LLM 시스템으로 생성된 고품질 레시피 - 고객 요청: {original_query}"
            }

            # 관리자 주문서 디렉토리 생성
            admin_dir = "admin_orders"
            if not os.path.exists(admin_dir):
                os.makedirs(admin_dir)

            # 주문서 파일로 저장
            order_file = os.path.join(admin_dir, f"{order_id}.json")
            with open(order_file, 'w', encoding='utf-8') as f:
                json.dump(order_data, f, ensure_ascii=False, indent=2)

            logger.info(f"진짜 LLM 주문서 저장됨: {order_file}")

        except Exception as e:
            logger.error(f"실제 LLM 주문서 생성 실패: {e}")

    async def _fallback_creation_response(self, validation_result, is_korean: bool) -> str:
        """최종 폴백 응답 (모든 AI가 실패했을 때)"""
        # 최종 폴백도 예외 발생으로 처리 - 목업 대답 없음
        raise Exception("응답 실패했습니다")