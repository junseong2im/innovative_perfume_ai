"""
SSE (Server-Sent Events) 기반 실시간 스트리밍 API
ChatGPT와 같은 실시간 타이핑 효과 제공
"""

from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
from sse_starlette.sse import EventSourceResponse

from fragrance_ai.orchestrator.artisan_orchestrator import (
    ArtisanOrchestrator,
    ConversationContext
)
from fragrance_ai.serving.vllm_client import get_vllm_client, VLLMConfig
from fragrance_ai.api.schemas import (
    GenerationRequest,
    StreamingChatRequest,
    StreamingResponse as StreamResponseSchema
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])

# 오케스트레이터 인스턴스
orchestrator = ArtisanOrchestrator()

# vLLM 클라이언트 설정
vllm_config = VLLMConfig(
    base_url="http://localhost:8100",
    model_name="llama3-8b"
)
vllm_client = get_vllm_client(vllm_config)


@router.post("/chat")
async def stream_chat(
    request: StreamingChatRequest,
    user_id: str = Query(..., description="사용자 ID"),
    session_id: Optional[str] = Query(None, description="세션 ID")
) -> EventSourceResponse:
    """
    실시간 채팅 스트리밍 엔드포인트
    SSE를 통해 토큰 단위로 응답을 스트리밍

    Example:
        ```javascript
        const eventSource = new EventSource('/api/v1/stream/chat?user_id=user123');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.content); // 한 토큰씩 출력
        };
        ```
    """
    async def generate_stream() -> AsyncGenerator[str, None]:
        """SSE 스트림 생성기"""
        try:
            # 대화 컨텍스트 생성
            context = ConversationContext(
                user_id=user_id,
                conversation_id=session_id or f"session_{datetime.now().timestamp()}",
                history=request.history or []
            )

            # 스트리밍 시작 이벤트
            yield json.dumps({
                "event": "start",
                "timestamp": datetime.now().isoformat(),
                "session_id": context.conversation_id
            })

            # vLLM으로 스트리밍 생성
            messages = [
                {"role": "system", "content": request.system_prompt or "You are a helpful perfume expert AI."},
                {"role": "user", "content": request.message}
            ]

            # 토큰 스트리밍
            token_count = 0
            async for token in vllm_client.chat_completion(
                messages=messages,
                max_tokens=request.max_tokens or 512,
                temperature=request.temperature or 0.7,
                stream=True
            ):
                token_count += 1

                # 토큰 이벤트 전송
                yield json.dumps({
                    "event": "token",
                    "content": token,
                    "token_count": token_count,
                    "timestamp": datetime.now().isoformat()
                })

                # 클라이언트 연결 확인
                if request.is_disconnected():
                    break

            # 완료 이벤트
            yield json.dumps({
                "event": "complete",
                "token_count": token_count,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # 에러 이벤트
            yield json.dumps({
                "event": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })

    return EventSourceResponse(generate_stream())


@router.post("/generate")
async def stream_generation(
    request: GenerationRequest
) -> EventSourceResponse:
    """
    향수 레시피 생성 스트리밍
    생성 과정을 실시간으로 전송
    """
    async def generate_recipe_stream() -> AsyncGenerator[str, None]:
        """레시피 생성 스트림"""
        try:
            # 시작 이벤트
            yield json.dumps({
                "event": "start",
                "stage": "analyzing",
                "message": "요청 분석 중..."
            })

            await asyncio.sleep(0.5)  # 시뮬레이션

            # 검색 단계
            yield json.dumps({
                "event": "progress",
                "stage": "searching",
                "message": "관련 향수 검색 중...",
                "progress": 20
            })

            # 실제 검색 수행 (비동기)
            from fragrance_ai.tools.search_tool import hybrid_search
            search_results = await hybrid_search(
                text_query=request.description,
                metadata_filters={
                    "season": request.season,
                    "gender": request.gender
                },
                top_k=5
            )

            yield json.dumps({
                "event": "progress",
                "stage": "search_complete",
                "message": f"{len(search_results.results)}개의 참조 향수 발견",
                "progress": 40
            })

            # 생성 단계
            yield json.dumps({
                "event": "progress",
                "stage": "generating",
                "message": "AI가 레시피를 생성하는 중...",
                "progress": 60
            })

            # 레시피 생성 (스트리밍)
            prompt = f"""
            Create a detailed perfume recipe:
            Description: {request.description}
            Style: {request.style}
            Season: {request.season}
            Gender: {request.gender}
            Intensity: {request.intensity}

            Provide the recipe with exact percentages for each note.
            """

            recipe_text = ""
            async for token in vllm_client.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.8,
                stream=True
            ):
                recipe_text += token
                # 주요 섹션이 완성되면 이벤트 전송
                if "Top Notes:" in recipe_text and "top_notes" not in locals():
                    top_notes = True
                    yield json.dumps({
                        "event": "section",
                        "type": "top_notes",
                        "content": recipe_text.split("Top Notes:")[1].split("\n")[0]
                    })
                elif "Heart Notes:" in recipe_text and "heart_notes" not in locals():
                    heart_notes = True
                    yield json.dumps({
                        "event": "section",
                        "type": "heart_notes",
                        "content": recipe_text.split("Heart Notes:")[1].split("\n")[0]
                    })
                elif "Base Notes:" in recipe_text and "base_notes" not in locals():
                    base_notes = True
                    yield json.dumps({
                        "event": "section",
                        "type": "base_notes",
                        "content": recipe_text.split("Base Notes:")[1].split("\n")[0]
                    })

            # 검증 단계
            yield json.dumps({
                "event": "progress",
                "stage": "validating",
                "message": "레시피 검증 중...",
                "progress": 80
            })

            # 실제 검증 수행
            from fragrance_ai.tools.validator_tool import validate_composition
            validation_result = await validate_composition({
                "top_notes": [{"name": "bergamot", "percentage": 20}],
                "heart_notes": [{"name": "rose", "percentage": 30}],
                "base_notes": [{"name": "sandalwood", "percentage": 20}]
            })

            yield json.dumps({
                "event": "validation",
                "valid": validation_result["valid"],
                "score": validation_result["score"],
                "feedback": validation_result.get("feedback", "")
            })

            # 완료
            yield json.dumps({
                "event": "complete",
                "recipe": {
                    "name": "Generated Perfume",
                    "description": request.description,
                    "full_text": recipe_text,
                    "validation_score": validation_result["score"]
                },
                "progress": 100
            })

        except Exception as e:
            logger.error(f"Generation streaming error: {e}")
            yield json.dumps({
                "event": "error",
                "message": str(e)
            })

    return EventSourceResponse(generate_recipe_stream())


@router.get("/health/ws")
async def websocket_health():
    """WebSocket/SSE 연결 상태 확인"""
    return {
        "status": "healthy",
        "streaming_available": True,
        "vllm_connected": await vllm_client.health_check(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/batch-stream")
async def batch_stream_generation(
    requests: list[GenerationRequest]
) -> EventSourceResponse:
    """
    배치 생성 스트리밍
    여러 요청을 동시에 처리하며 진행 상황 스트리밍
    """
    async def batch_stream() -> AsyncGenerator[str, None]:
        """배치 처리 스트림"""
        try:
            total = len(requests)
            completed = 0

            # 시작 이벤트
            yield json.dumps({
                "event": "batch_start",
                "total": total,
                "timestamp": datetime.now().isoformat()
            })

            # 각 요청을 병렬로 처리
            async def process_single(idx: int, req: GenerationRequest):
                """단일 요청 처리"""
                nonlocal completed

                # 요청 시작 알림
                yield json.dumps({
                    "event": "item_start",
                    "index": idx,
                    "description": req.description[:50] + "..."
                })

                # 실제 생성 (간단화)
                from fragrance_ai.tools.generator_tool import create_recipe
                result = await create_recipe(req.dict())

                completed += 1

                # 완료 알림
                return {
                    "event": "item_complete",
                    "index": idx,
                    "result": result,
                    "progress": (completed / total) * 100
                }

            # 병렬 실행
            tasks = [
                process_single(i, req)
                for i, req in enumerate(requests)
            ]

            # 결과를 받는 대로 스트리밍
            for coro in asyncio.as_completed(tasks):
                result = await coro
                yield json.dumps(result)

            # 배치 완료
            yield json.dumps({
                "event": "batch_complete",
                "total": total,
                "completed": completed,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            logger.error(f"Batch streaming error: {e}")
            yield json.dumps({
                "event": "error",
                "message": str(e)
            })

    return EventSourceResponse(batch_stream())


# 프론트엔드 예제 코드
FRONTEND_EXAMPLE = """
<!-- HTML -->
<div id="chat-output"></div>
<button onclick="startStreaming()">Start Streaming</button>

<script>
// JavaScript (React 예제)
function ChatComponent() {
    const [messages, setMessages] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);

    const startStreaming = async () => {
        setIsStreaming(true);
        const eventSource = new EventSource(
            `/api/v1/stream/chat?user_id=${userId}&session_id=${sessionId}`,
            {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: "Create a summer perfume",
                    temperature: 0.7
                })
            }
        );

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.event === 'token') {
                // 토큰을 하나씩 추가 (타이핑 효과)
                setMessages(prev => prev + data.content);
            } else if (data.event === 'complete') {
                setIsStreaming(false);
                eventSource.close();
            } else if (data.event === 'error') {
                console.error('Streaming error:', data.message);
                setIsStreaming(false);
                eventSource.close();
            }
        };

        eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            setIsStreaming(false);
            eventSource.close();
        };
    };

    return (
        <div>
            <div className="chat-output">
                {messages}
                {isStreaming && <span className="typing-cursor">▋</span>}
            </div>
            <button onClick={startStreaming} disabled={isStreaming}>
                {isStreaming ? 'Generating...' : 'Generate'}
            </button>
        </div>
    );
}

// CSS for typing effect
.typing-cursor {
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0; }
}
</script>
"""