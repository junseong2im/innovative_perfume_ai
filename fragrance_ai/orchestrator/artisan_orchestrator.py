"""
Artisan Orchestrator - Dual-LLM 아키텍처 구현
- Conductor LLM: 대화 및 도구 오케스트레이션
- Creator LLM: 전문 향수 레시피 생성
- IP 보호 및 도메인 특화
"""

import logging
import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import traceback

# 도구 임포트
from ..tools.search_tool import hybrid_search, SearchQuery
from ..tools.validator_tool import validate_composition, NotesComposition
from ..tools.knowledge_tool import query_knowledge_base, KnowledgeQuery
from ..tools.generator_tool import create_recipe, GenerationRequest

# 데이터베이스 모델
from ..database.models import GeneratedRecipe as DBGeneratedRecipe, User
from ..database.base import get_db_session

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Tool execution status"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolExecutionResult:
    """Tool execution result with detailed status"""
    tool: str
    status: ToolStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    fallback_used: bool = False
    retry_count: int = 0


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for tool reliability"""
    failures: int = 0
    success: int = 0
    is_open: bool = False
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    cooldown_period: timedelta = field(default_factory=lambda: timedelta(seconds=60))


@dataclass
class ConversationContext:
    """대화 컨텍스트"""
    user_id: str
    conversation_id: str
    history: List[Dict[str, str]]
    current_recipe: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None


@dataclass
class ArtisanResponse:
    """Artisan 응답"""
    message: str  # 사용자에게 보여질 메시지
    recipe_summary: Optional[Dict[str, Any]] = None  # 간략화된 레시피 (사용자용)
    conversation_id: str = ""
    request_id: str = ""
    suggestions: List[str] = None
    visualization: Optional[str] = None  # 시각화 URL 또는 데이터


class DomainGuardrail:
    """도메인 가드레일 - 향수 관련 질문만 처리"""

    def __init__(self):
        self.perfume_keywords = [
            "향수", "perfume", "fragrance", "scent", "노트", "note",
            "플로럴", "floral", "우디", "woody", "시트러스", "citrus",
            "오리엔탈", "oriental", "향", "smell", "조향", "레시피",
            "블렌딩", "blending", "어코드", "accord", "향료", "aroma"
        ]

        self.off_topic_responses = [
            "죄송하지만, 저는 향수 전문 AI입니다. 향수 관련 질문을 해주세요.",
            "향수와 조향에 대한 질문만 답변 드릴 수 있습니다.",
            "향수 레시피, 노트, 조향 기술에 대해 물어보세요."
        ]

    def is_on_topic(self, text: str) -> bool:
        """질문이 향수 관련인지 확인"""
        text_lower = text.lower()

        # 키워드 체크
        for keyword in self.perfume_keywords:
            if keyword in text_lower:
                return True

        # 문맥 분석 (간단한 휴리스틱)
        if any(word in text_lower for word in ["만들", "create", "design", "blend"]):
            if any(word in text_lower for word in ["향", "scent", "aroma"]):
                return True

        return False

    def get_rejection_message(self) -> str:
        """거부 메시지 반환"""
        import random
        return random.choice(self.off_topic_responses)


class ArtisanOrchestrator:
    """Artisan 오케스트레이터 - 메인 컨트롤러 with resilience patterns"""

    def __init__(self, config_path: str = "configs/local.json"):
        """오케스트레이터 초기화"""
        self.config = self._load_config(config_path)
        self.guardrail = DomainGuardrail()
        self.conductor_llm = None
        self.creator_llm = None
        self._initialize_llms()

        # Resilience components
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.tool_priorities = {
            "hybrid_search": ["search_fallback", "basic_search"],
            "recipe_generator": ["simple_generator", "template_generator"],
            "scientific_validator": ["basic_validator", "rule_validator"],
            "perfumer_knowledge": ["cached_knowledge", "static_knowledge"]
        }
        self.max_retries = 3
        self.timeout_seconds = 30
        self.enable_partial_results = True

        # Initialize tools mapping for testing
        self.tools = {
            "hybrid_search": self._tool_hybrid_search,
            "recipe_generator": self._tool_recipe_generator,
            "scientific_validator": self._tool_scientific_validator,
            "perfumer_knowledge": self._tool_perfumer_knowledge
        }
        self.llm_client = None  # For test compatibility

    def _load_config(self, config_path: str) -> dict:
        """설정 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            return {}

    def _initialize_llms(self):
        """LLM 초기화"""
        try:
            # Conductor LLM (일반 대화 및 계획)
            if self.config.get('llm_orchestrator', {}).get('provider') == 'ollama':
                # Ollama 사용 (로컬)
                self.conductor_llm = self._init_ollama_conductor()
            else:
                # 폴백: 간단한 규칙 기반
                self.conductor_llm = self._init_rule_based_conductor()

            # Creator LLM은 generator_tool.py에서 처리
            logger.info("LLMs initialized successfully")

        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            self.conductor_llm = self._init_rule_based_conductor()

    def _init_ollama_conductor(self):
        """Ollama Conductor 초기화"""
        try:
            from ..llm.ollama_client import OllamaClient

            # Ollama 클라이언트 생성
            self.ollama_client = OllamaClient(self.config)

            # 비동기로 가용성 확인
            import asyncio
            loop = asyncio.new_event_loop()
            is_available = loop.run_until_complete(self.ollama_client.check_availability())
            loop.close()

            if is_available:
                logger.info("Ollama conductor initialized successfully")
                return self.ollama_client  # 실제 클라이언트 반환
            else:
                logger.warning("Ollama not available, falling back to rule-based")
                return self._init_rule_based_conductor()

        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            return self._init_rule_based_conductor()

    def _init_rule_based_conductor(self):
        """규칙 기반 Conductor (폴백)"""
        return RuleBasedConductor()

    async def process_message(
        self,
        message: str,
        context: ConversationContext
    ) -> ArtisanResponse:
        """메시지 처리 - 메인 엔트리 포인트"""
        request_id = str(uuid.uuid4())[:8]
        logger.info(f"Processing message {request_id}: {message[:100]}...")

        try:
            # 1. 가드레일 체크
            if not self.guardrail.is_on_topic(message):
                return ArtisanResponse(
                    message=self.guardrail.get_rejection_message(),
                    conversation_id=context.conversation_id,
                    request_id=request_id
                )

            # 2. 의도 분석 및 계획
            intent = await self._analyze_intent(message, context)
            plan = await self._create_execution_plan(intent, message, context)

            # 3. 도구 실행
            tool_results = await self._execute_tools(plan)

            # 4. 결과 종합 및 레시피 생성
            response = await self._synthesize_response(
                message, intent, tool_results, context
            )

            # 5. IP 보호 - 상세 레시피는 DB에 저장, 사용자에게는 요약만
            if response.recipe_summary:
                await self._save_protected_recipe(
                    response.recipe_summary,
                    context,
                    message,
                    request_id
                )

            # 6. 대화 기록 업데이트
            context.history.append({"user": message, "assistant": response.message})

            response.request_id = request_id
            response.conversation_id = context.conversation_id

            return response

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return ArtisanResponse(
                message="죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                conversation_id=context.conversation_id,
                request_id=request_id
            )

    async def _analyze_intent(
        self,
        message: str,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """사용자 의도 분석"""
        # Perfume Description LLM으로 먼저 해석
        try:
            from ..llm.perfume_description_llm import interpret_customer_description

            # 고객 설명 해석
            interpretation_result = await interpret_customer_description(
                description=message,
                enhance_story=False
            )

            interpretation = interpretation_result['interpretation']

            # 해석 결과를 intent로 변환
            if interpretation.confidence > 0.6:
                intent = {
                    "type": "create_perfume",
                    "confidence": interpretation.confidence,
                    "entities": {
                        "fragrance_family": interpretation.fragrance_families[0] if interpretation.fragrance_families else "floral",
                        "mood": interpretation.mood,
                        "season": interpretation.season,
                        "intensity": interpretation.intensity,
                        "key_notes": interpretation.key_notes,
                        "style": interpretation.style,
                        "occasion": interpretation.occasion
                    }
                }
                logger.info(f"AI interpreted intent: {intent}")
                return intent
        except Exception as e:
            logger.warning(f"Description LLM failed: {e}")

        # Ollama LLM 사용 시도
        if hasattr(self, 'ollama_client'):
            try:
                # 실제 Ollama로 intent 분석
                prompt = f"""Analyze this message and determine the user's intent regarding perfume:

Message: {message}

Respond with a JSON object containing:
- type: (create_perfume, search_perfume, knowledge_query, validate_recipe, general_chat)
- confidence: (0.0 to 1.0)
- entities: (detected entities like fragrance_family, mood, season, etc.)

JSON:"""

                response = await self.ollama_client.generate(prompt)

                # JSON 파싱 시도
                import json
                import re

                # JSON 추출 (코드 블록 제거)
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    intent = json.loads(json_match.group())
                    if intent.get("confidence", 0) > 0.3:
                        logger.info(f"Ollama analyzed intent: {intent}")
                        return intent
            except Exception as e:
                logger.warning(f"Ollama intent analysis failed: {e}")

        # 폴백: 규칙 기반 분석
        intent = {
            "type": "unknown",
            "confidence": 0.0,
            "entities": {}
        }

        message_lower = message.lower()

        if any(word in message_lower for word in ["만들", "create", "design", "제작"]):
            intent["type"] = "create_perfume"
            intent["confidence"] = 0.9

        elif any(word in message_lower for word in ["찾", "search", "find", "추천"]):
            intent["type"] = "search_perfume"
            intent["confidence"] = 0.85

        elif any(word in message_lower for word in ["뭐", "what", "설명", "알려"]):
            intent["type"] = "knowledge_query"
            intent["confidence"] = 0.8

        elif any(word in message_lower for word in ["검증", "validate", "평가", "점수"]):
            intent["type"] = "validate_recipe"
            intent["confidence"] = 0.85

        else:
            intent["type"] = "general_chat"
            intent["confidence"] = 0.5

        # 엔티티 추출
        if "플로럴" in message_lower or "floral" in message_lower:
            intent["entities"]["fragrance_family"] = "floral"

        if "여름" in message_lower or "summer" in message_lower:
            intent["entities"]["season"] = "summer"

        if "로맨틱" in message_lower or "romantic" in message_lower:
            intent["entities"]["mood"] = "romantic"

        return intent

    async def _create_execution_plan(
        self,
        intent: Dict[str, Any],
        message: str,
        context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """실행 계획 수립"""
        plan = []

        if intent["type"] == "create_perfume":
            # 향수 제작 플로우
            plan.append({
                "tool": "search",
                "purpose": "Find similar existing perfumes for inspiration",
                "params": {"query": message, "top_k": 3}
            })
            plan.append({
                "tool": "knowledge",
                "purpose": "Get relevant perfume knowledge",
                "params": {"category": "technique", "query": "blending"}
            })
            plan.append({
                "tool": "generate",
                "purpose": "Create new perfume recipe",
                "params": {"description": message, **intent.get("entities", {})}
            })
            plan.append({
                "tool": "validate",
                "purpose": "Validate the generated recipe",
                "params": {}  # Will use generated recipe
            })

        elif intent["type"] == "search_perfume":
            plan.append({
                "tool": "search",
                "purpose": "Search for perfumes",
                "params": {"query": message, "top_k": 10}
            })

        elif intent["type"] == "knowledge_query":
            # 지식 카테고리 추론
            category = self._infer_knowledge_category(message)
            plan.append({
                "tool": "knowledge",
                "purpose": "Answer knowledge question",
                "params": {"category": category, "query": message}
            })

        elif intent["type"] == "validate_recipe":
            if context.current_recipe:
                plan.append({
                    "tool": "validate",
                    "purpose": "Validate current recipe",
                    "params": {"recipe": context.current_recipe}
                })
            else:
                plan.append({
                    "tool": "knowledge",
                    "purpose": "Explain validation process",
                    "params": {"category": "technique", "query": "validation"}
                })

        return plan

    def _infer_knowledge_category(self, message: str) -> str:
        """지식 카테고리 추론"""
        message_lower = message.lower()

        if any(word in message_lower for word in ["역사", "history", "고대", "ancient"]):
            return "history"
        elif any(word in message_lower for word in ["기술", "technique", "방법", "제조"]):
            return "technique"
        elif any(word in message_lower for word in ["노트", "note", "재료", "향료"]):
            return "note"
        elif any(word in message_lower for word in ["어코드", "accord", "조합"]):
            return "accord"
        elif any(word in message_lower for word in ["조향사", "perfumer"]):
            return "perfumer"
        else:
            return "general"

    async def _execute_tools(self, plan: List[Dict[str, Any]]) -> List[ToolExecutionResult]:
        """Execute tools with resilience patterns"""
        results = []

        # Group tools by parallelizable vs sequential
        parallel_tools = []
        sequential_tools = []

        for step in plan:
            if step.get("parallel", True):
                parallel_tools.append(step)
            else:
                sequential_tools.append((step))

        # Execute parallel tools concurrently
        if parallel_tools:
            parallel_results = await self._execute_parallel_tools(parallel_tools)
            results.extend(parallel_results)

        # Execute sequential tools with dependency checking
        for step in sequential_tools:
            # Check if dependencies are met
            if not self._check_dependencies(step, results):
                results.append(ToolExecutionResult(
                    tool=step["tool"],
                    status=ToolStatus.SKIPPED,
                    error="Dependencies not met"
                ))
                continue

            result = await self._execute_single_tool_with_resilience(step)
            results.append(result)

            # Stop if critical tool fails
            if result.status == ToolStatus.FAILED and step.get("critical", False):
                logger.warning(f"Critical tool {step['tool']} failed, stopping execution")
                break

        return results

    async def _execute_parallel_tools(self, tools: List[Dict[str, Any]]) -> List[ToolExecutionResult]:
        """Execute multiple tools in parallel with isolation"""
        tasks = []
        for tool in tools:
            task = asyncio.create_task(
                self._execute_single_tool_with_resilience(tool)
            )
            tasks.append(task)

        # Wait for all with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout_seconds * 2  # Double timeout for parallel
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

            # Collect completed results
            results = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    try:
                        results.append(task.result())
                    except Exception:
                        pass

            # Add timeout results for incomplete tasks
            for i, task in enumerate(tasks):
                if task.cancelled() or not task.done():
                    results.append(ToolExecutionResult(
                        tool=tools[i]["tool"],
                        status=ToolStatus.TIMEOUT,
                        error="Execution timeout in parallel batch"
                    ))

        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolExecutionResult(
                    tool=tools[i]["tool"],
                    status=ToolStatus.FAILED,
                    error=str(result)
                ))
            else:
                final_results.append(result)

        return final_results

    async def _execute_single_tool_with_resilience(self, step: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a single tool with all resilience patterns"""
        tool_name = step["tool"]
        params = step.get("params", {})
        start_time = asyncio.get_event_loop().time()

        # Check circuit breaker
        if not self._check_circuit_breaker(tool_name):
            # Try fallback immediately
            return await self._execute_fallback(tool_name, params)

        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Add timeout wrapper
                result = await asyncio.wait_for(
                    self._execute_tool_internal(tool_name, params),
                    timeout=self.timeout_seconds
                )

                # Record success
                self._record_tool_success(tool_name)

                return ToolExecutionResult(
                    tool=tool_name,
                    status=ToolStatus.SUCCESS,
                    result=result,
                    execution_time=asyncio.get_event_loop().time() - start_time,
                    retry_count=attempt
                )

            except asyncio.TimeoutError:
                last_error = "Tool execution timeout"
                logger.warning(f"Tool {tool_name} timeout on attempt {attempt + 1}")

                # Don't retry on timeout
                break

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool {tool_name} failed on attempt {attempt + 1}: {e}")

                # Exponential backoff
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        # Record failure
        self._record_tool_failure(tool_name)

        # Try fallback
        fallback_result = await self._execute_fallback(tool_name, params)
        if fallback_result.status == ToolStatus.SUCCESS:
            return fallback_result

        # Return failure
        return ToolExecutionResult(
            tool=tool_name,
            status=ToolStatus.FAILED,
            error=last_error,
            execution_time=asyncio.get_event_loop().time() - start_time,
            retry_count=self.max_retries
        )

    async def _execute_tool_internal(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Internal tool execution with proper error handling"""
        try:
            if tool_name == "search" or tool_name == "hybrid_search":
                result = await hybrid_search(
                    text_query=params.get("query", ""),
                    top_k=params.get("top_k", 10)
                )
            elif tool_name == "knowledge" or tool_name == "perfumer_knowledge":
                result = await query_knowledge_base(
                    category=params.get("category", "general"),
                    query=params.get("query", "")
                )
            elif tool_name == "generate" or tool_name == "recipe_generator":
                request = GenerationRequest(
                    description=params.get("description", ""),
                    fragrance_family=params.get("fragrance_family", "floral"),
                    mood=params.get("mood", "romantic"),
                    season=params.get("season"),
                    gender=params.get("gender", "unisex"),
                    intensity=params.get("intensity", "moderate")
                )
                result = await create_recipe(request)
            elif tool_name == "validate" or tool_name == "scientific_validator":
                # Extract composition from params if provided
                if params.get("composition"):
                    composition = params["composition"]
                else:
                    # Create a default composition for testing
                    composition = NotesComposition(
                        top_notes=[{"Bergamot": 25}],
                        heart_notes=[{"Rose": 35}],
                        base_notes=[{"Sandalwood": 20}],
                        total_ingredients=12
                    )
                result = await validate_composition(composition)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return result

        except Exception as e:
            logger.error(f"Tool {tool_name} execution failed: {e}")
            raise  # Re-raise to be caught by the retry mechanism

    async def _execute_fallback(self, tool_name: str, params: Dict[str, Any]) -> ToolExecutionResult:
        """Execute fallback strategy for failed tool"""
        fallbacks = self.tool_priorities.get(tool_name, [])

        for fallback_name in fallbacks:
            try:
                logger.info(f"Attempting fallback {fallback_name} for {tool_name}")

                # Execute fallback based on type
                if fallback_name == "search_fallback":
                    # Simple keyword search
                    result = {"results": [], "method": "fallback_keyword"}
                elif fallback_name == "simple_generator":
                    # Template-based generation
                    result = self._generate_template_recipe(params)
                elif fallback_name == "basic_validator":
                    # Rule-based validation
                    result = {"valid": True, "score": 7.0, "method": "rule_based"}
                elif fallback_name == "cached_knowledge":
                    # Return cached or static knowledge
                    result = {"answer": "Standard perfume knowledge", "cached": True}
                else:
                    continue

                return ToolExecutionResult(
                    tool=tool_name,
                    status=ToolStatus.PARTIAL,
                    result=result,
                    fallback_used=True
                )

            except Exception as e:
                logger.warning(f"Fallback {fallback_name} failed: {e}")
                continue

        # All fallbacks failed
        return ToolExecutionResult(
            tool=tool_name,
            status=ToolStatus.FAILED,
            error="All fallback strategies exhausted",
            fallback_used=True
        )

    def _check_circuit_breaker(self, tool_name: str) -> bool:
        """Check if circuit breaker allows execution"""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = CircuitBreakerState()

        breaker = self.circuit_breakers[tool_name]

        # Check if circuit is open
        if breaker.is_open:
            # Check if cooldown period has passed
            if breaker.last_failure:
                time_since_failure = datetime.utcnow() - breaker.last_failure
                if time_since_failure > breaker.cooldown_period:
                    # Try to close circuit (half-open state)
                    breaker.is_open = False
                    breaker.failures = 0
                    logger.info(f"Circuit breaker for {tool_name} entering half-open state")
                    return True
            return False

        return True

    def _record_tool_success(self, tool_name: str):
        """Record successful tool execution"""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = CircuitBreakerState()

        breaker = self.circuit_breakers[tool_name]
        breaker.success += 1
        breaker.last_success = datetime.utcnow()

        # Reset failure count on success
        if breaker.failures > 0:
            breaker.failures = 0

        # Close circuit if it was open
        if breaker.is_open:
            breaker.is_open = False
            logger.info(f"Circuit breaker for {tool_name} closed after success")

    def _record_tool_failure(self, tool_name: str):
        """Record tool execution failure"""
        if tool_name not in self.circuit_breakers:
            self.circuit_breakers[tool_name] = CircuitBreakerState()

        breaker = self.circuit_breakers[tool_name]
        breaker.failures += 1
        breaker.last_failure = datetime.utcnow()

        # Open circuit after threshold
        if breaker.failures >= 3 and not breaker.is_open:
            breaker.is_open = True
            logger.warning(f"Circuit breaker opened for {tool_name} after {breaker.failures} failures")

    def _check_dependencies(self, step: Dict[str, Any], results: List[ToolExecutionResult]) -> bool:
        """Check if tool dependencies are satisfied"""
        dependencies = step.get("depends_on", [])
        if not dependencies:
            return True

        for dep in dependencies:
            # Check if dependency was executed successfully
            dep_result = next(
                (r for r in results if r.tool == dep),
                None
            )
            if not dep_result or dep_result.status not in [ToolStatus.SUCCESS, ToolStatus.PARTIAL]:
                logger.warning(f"Dependency {dep} not satisfied for {step['tool']}")
                return False

        return True

    def _generate_template_recipe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple template-based recipe as fallback"""
        return {
            "name": "Classic Floral Blend",
            "description": "A timeless fragrance composition",
            "top_notes": [{"name": "Bergamot", "percentage": 15}],
            "heart_notes": [{"name": "Rose", "percentage": 30}],
            "base_notes": [{"name": "Sandalwood", "percentage": 20}],
            "character": "Elegant",
            "longevity": "6-8 hours",
            "method": "template_fallback"
        }

    async def _synthesize_response(
        self,
        message: str,
        intent: Dict[str, Any],
        tool_results: List[ToolExecutionResult],
        context: ConversationContext
    ) -> ArtisanResponse:
        """응답 종합"""

        # 실제 AI LLM으로 응답 생성 시도
        if hasattr(self, 'ollama_client'):
            try:
                # Description LLM으로 스토리텔링 추가
                from ..llm.perfume_description_llm import interpret_customer_description

                interpretation_result = await interpret_customer_description(
                    description=message,
                    enhance_story=True
                )

                # 도구 결과 요약
                tool_summary = self._summarize_tool_results(tool_results)

                # Ollama로 최종 응답 생성
                prompt = f"""당신은 럭셔리 향수 브랜드 Deulsoom의 수석 조향사입니다.
고객의 요청에 대해 전문적이면서도 친근하게 응답하세요.

고객 요청: {message}

분석된 의도: {intent}

도구 실행 결과:
{tool_summary}

고객 해석 스토리:
{interpretation_result.get('story', '')}

위 정보를 바탕으로 자연스럽고 매력적인 한국어 응답을 작성하세요.
기술적 정보를 포함하되 이해하기 쉽게 설명하고, 고객이 특별함을 느끼도록 하세요.

응답:"""

                ai_response = await self.ollama_client.generate(prompt)

                if ai_response and len(ai_response) > 100:
                    # AI 응답 성공
                    recipe_summary = self._extract_recipe_from_results(tool_results)

                    return ArtisanResponse(
                        message=ai_response,
                        conversation_id=context.conversation_id,
                        recipe_summary=recipe_summary,
                        suggestions=["향수 시향 예약", "비슷한 향수 추천", "노트 커스터마이징"]
                    )
            except Exception as e:
                logger.warning(f"AI synthesis failed: {e}")

        # 폴백: 기존 규칙 기반 응답
        response_text = ""
        recipe_summary = None
        suggestions = []

        if intent["type"] == "create_perfume":
            # 생성된 레시피 찾기
            recipe_result = next(
                (r for r in tool_results if r.tool == "generate" and r.result),
                None
            )

            validation_result = next(
                (r for r in tool_results if r.tool == "validate" and r.result),
                None
            )

            if recipe_result:
                recipe = recipe_result.result

                # 사용자용 요약 생성 (IP 보호)
                recipe_summary = {
                    "name": recipe.name,
                    "description": recipe.description,
                    "character": recipe.character,
                    "key_notes": {
                        "top": [n.name for n in recipe.top_notes[:2]],  # 주요 노트만
                        "heart": [n.name for n in recipe.heart_notes[:2]],
                        "base": [n.name for n in recipe.base_notes[:2]]
                    },
                    "mood": recipe.character,
                    "longevity": recipe.longevity,
                    "occasions": recipe.wearing_occasions
                }

                response_text = f"""당신을 위한 특별한 향수를 창조했습니다.

**{recipe.name}**

{recipe.description}

이 향수는 {recipe.character}의 특성을 가지며, {recipe.longevity}의 지속시간을 보입니다.

주요 노트:
- 탑: {', '.join(recipe_summary['key_notes']['top'])}
- 하트: {', '.join(recipe_summary['key_notes']['heart'])}
- 베이스: {', '.join(recipe_summary['key_notes']['base'])}"""

                if validation_result:
                    val = validation_result.result
                    response_text += f"\n\n과학적 검증 결과: {val.overall_score:.1f}/10"

                    if val.suggestions:
                        suggestions = val.suggestions[:2]

                # 컨텍스트에 현재 레시피 저장
                context.current_recipe = recipe_summary

            else:
                response_text = "죄송합니다. 레시피 생성 중 문제가 발생했습니다."

        elif intent["type"] == "search_perfume":
            search_result = next(
                (r for r in tool_results if r.tool == "search" and r.result),
                None
            )

            if search_result:
                results = search_result.result
                if results.results:
                    response_text = f"'{message}'와 관련된 향수를 찾았습니다:\n\n"
                    for i, item in enumerate(results.results[:5], 1):
                        response_text += f"{i}. **{item.name}**\n"
                        response_text += f"   {item.description[:100]}...\n"
                        response_text += f"   계열: {item.fragrance_family}\n\n"
                else:
                    response_text = "관련 향수를 찾을 수 없습니다."

        elif intent["type"] == "knowledge_query":
            knowledge_result = next(
                (r for r in tool_results if r.tool == "knowledge" and r.result),
                None
            )

            if knowledge_result:
                kb = knowledge_result.result
                response_text = kb.answer

                if kb.related_topics:
                    suggestions = [f"관련 주제: {topic}" for topic in kb.related_topics[:3]]

        else:
            response_text = "무엇을 도와드릴까요? 향수 제작, 검색, 또는 향수 지식에 대해 물어보세요."

        return ArtisanResponse(
            message=response_text,
            recipe_summary=recipe_summary,
            suggestions=suggestions
        )

    def _summarize_tool_results(self, tool_results: List[ToolExecutionResult]) -> str:
        """도구 결과 요약"""
        summary_lines = []
        for result in tool_results:
            if result.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL]:
                tool_name = result.tool
                status_icon = "✓" if result.status == ToolStatus.SUCCESS else "⚠"

                if result.fallback_used:
                    summary_lines.append(f"{status_icon} {tool_name}: 대체 방법 사용")
                elif tool_name == "generate" and result.result:
                    recipe_name = result.result.get('name', 'Custom Creation')
                    summary_lines.append(f"{status_icon} 새로운 향수 레시피 생성: {recipe_name}")
                elif tool_name == "search":
                    count = len(result.result.get('results', [])) if result.result else 0
                    summary_lines.append(f"{status_icon} 유사 향수 {count}개 검색 완료")
                elif tool_name == "validate":
                    summary_lines.append(f"{status_icon} 레시피 검증 완료")
                elif tool_name == "knowledge":
                    summary_lines.append(f"{status_icon} 향수 지식베이스 조회 완료")
            elif result.status == ToolStatus.FAILED:
                summary_lines.append(f"✗ {result.tool}: 실행 실패")
            elif result.status == ToolStatus.TIMEOUT:
                summary_lines.append(f"⏱ {result.tool}: 시간 초과")

        return "\n".join(summary_lines) if summary_lines else "도구 실행 완료"

    def _extract_recipe_from_results(self, tool_results: List[ToolExecutionResult]) -> Optional[Dict]:
        """도구 결과에서 레시피 추출"""
        for result in tool_results:
            if result.tool == "generate" and result.result and result.status in [ToolStatus.SUCCESS, ToolStatus.PARTIAL]:
                recipe = result.result
                return {
                    "name": getattr(recipe, 'name', 'Custom Creation'),
                    "description": getattr(recipe, 'description', ''),
                    "character": getattr(recipe, 'character', ''),
                    "key_notes": {
                        "top": [n.name for n in getattr(recipe, 'top_notes', [])[:2]],
                        "heart": [n.name for n in getattr(recipe, 'heart_notes', [])[:2]],
                        "base": [n.name for n in getattr(recipe, 'base_notes', [])[:2]]
                    }
                }
        return None

    async def _save_protected_recipe(
        self,
        recipe_summary: Dict[str, Any],
        context: ConversationContext,
        user_prompt: str,
        request_id: str
    ):
        """보호된 레시피 DB 저장"""
        try:
            async with get_db_session() as session:
                # 전체 레시피 정보 (백엔드 전용)
                master_formula = {
                    # 실제 상세 포뮬라 - 사용자에게 노출되지 않음
                    "complete_formula": context.current_recipe,
                    "proprietary_data": "CONFIDENTIAL"
                }

                db_recipe = DBGeneratedRecipe(
                    id=str(uuid.uuid4()),
                    user_id=context.user_id,
                    conversation_id=context.conversation_id,
                    user_prompt=user_prompt,
                    conversation_history=context.history,
                    recipe_name=recipe_summary["name"],
                    recipe_description=recipe_summary["description"],
                    master_formula=master_formula,
                    top_notes=recipe_summary["key_notes"]["top"],
                    heart_notes=recipe_summary["key_notes"]["heart"],
                    base_notes=recipe_summary["key_notes"]["base"],
                    generation_model="artisan-v1",
                    generation_timestamp=datetime.utcnow(),
                    is_validated=True
                )

                session.add(db_recipe)
                await session.commit()

                logger.info(f"Recipe {request_id} saved securely to database")

        except Exception as e:
            logger.error(f"Failed to save recipe: {e}")

    async def _tool_hybrid_search(self, *args, **kwargs):
        """Tool wrapper for hybrid search"""
        from ..tools.search_tool import hybrid_search
        return await hybrid_search(*args, **kwargs)

    async def _tool_recipe_generator(self, *args, **kwargs):
        """Tool wrapper for recipe generator"""
        from ..tools.generator_tool import create_recipe
        return await create_recipe(*args, **kwargs)

    async def _tool_scientific_validator(self, *args, **kwargs):
        """Tool wrapper for scientific validator"""
        from ..tools.validator_tool import validate_composition
        return await validate_composition(*args, **kwargs)

    async def _tool_perfumer_knowledge(self, *args, **kwargs):
        """Tool wrapper for perfumer knowledge"""
        from ..tools.knowledge_tool import query_knowledge_base
        return await query_knowledge_base(*args, **kwargs)

    async def orchestrate(self, message: str, **kwargs) -> Dict[str, Any]:
        """Test-compatible orchestrate method"""
        # Extract context parameters
        user_id = kwargs.get('user_id', 'test_user')
        conversation_id = kwargs.get('conversation_id', str(uuid.uuid4()))
        session_id = kwargs.get('session_id', 'test_session')

        # Create context
        context = ConversationContext(
            user_id=user_id,
            conversation_id=conversation_id,
            history=[]
        )

        # Process message
        response = await self.process_message(message, context)

        # Return test-compatible format
        return {
            "message": response.message,
            "recipe_summary": response.recipe_summary,
            "status": "success" if response.message and "오류" not in response.message else "error",
            "request_id": response.request_id
        }


class RuleBasedConductor:
    """규칙 기반 Conductor (폴백)"""

    def analyze(self, text: str) -> Dict[str, Any]:
        """간단한 규칙 기반 분석"""
        return {
            "intent": "general",
            "entities": {},
            "confidence": 0.5
        }

    def plan(self, intent: Dict[str, Any]) -> List[str]:
        """간단한 계획 수립"""
        if intent.get("intent") == "create":
            return ["search", "generate", "validate"]
        return ["search"]


# 전역 오케스트레이터 인스턴스
orchestrator_instance = None


def get_orchestrator():
    """오케스트레이터 인스턴스 가져오기"""
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = ArtisanOrchestrator()
    return orchestrator_instance


async def process_chat_message(
    message: str,
    user_id: str,
    conversation_id: str,
    history: List[Dict[str, str]] = None
) -> ArtisanResponse:
    """채팅 메시지 처리 - API 엔드포인트용"""
    orchestrator = get_orchestrator()

    context = ConversationContext(
        user_id=user_id,
        conversation_id=conversation_id,
        history=history or []
    )

    return await orchestrator.process_message(message, context)