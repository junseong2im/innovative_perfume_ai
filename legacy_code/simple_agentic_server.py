"""
Simple Agentic API Server for website integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fragrance_ai.services.orchestrator_service import OrchestratorService
    AGENTIC_AVAILABLE = True
    print("SUCCESS: Agentic system imported successfully")
except Exception as e:
    print(f"WARNING: Agentic system not available: {e}")
    AGENTIC_AVAILABLE = False

app = FastAPI(title="Simple Agentic Fragrance API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response 모델
class AgenticRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AgenticResponse(BaseModel):
    success: bool
    message: str
    request_id: str
    execution_time: float
    tools_used: List[str]
    complexity: str
    metadata: Dict[str, Any]

# Global orchestrator service
orchestrator_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize agentic system on startup"""
    global orchestrator_service
    if AGENTIC_AVAILABLE:
        try:
            orchestrator_service = OrchestratorService()
            await orchestrator_service.initialize()
            print("SUCCESS: Agentic orchestrator initialized successfully")
        except Exception as e:
            print(f"ERROR: Failed to initialize orchestrator: {e}")
            orchestrator_service = None

@app.get("/")
async def root():
    return {
        "message": "Simple Agentic Fragrance API",
        "status": "operational",
        "agentic_available": AGENTIC_AVAILABLE and orchestrator_service is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agentic_system": AGENTIC_AVAILABLE and orchestrator_service is not None
    }

@app.get("/api/v1/health")
async def health_check_v1():
    return {
        "status": "healthy",
        "agentic_system": AGENTIC_AVAILABLE and orchestrator_service is not None,
        "version": "v1"
    }

@app.post("/api/v2/agentic/request")
async def process_agentic_request(request: AgenticRequest):
    """Process general agentic requests"""
    if not AGENTIC_AVAILABLE or not orchestrator_service:
        # Mock response when agentic system is not available
        return {
            "success": True,
            "message": f"목업 응답: {request.query}에 대한 향수를 생성하고 있습니다. 로맨틱하고 우아한 향수가 완성되었습니다.",
            "request_id": f"mock_{int(asyncio.get_event_loop().time())}",
            "execution_time": 2.5,
            "tools_used": ["hybrid_search", "validate_composition"],
            "complexity": "moderate",
            "metadata": {
                "mock": True,
                "timestamp": int(asyncio.get_event_loop().time())
            }
        }

    try:
        result = await orchestrator_service.process_fragrance_request(
            user_query=request.query,
            user_context=request.context or {}
        )

        return {
            "success": result["success"],
            "message": result["message"],
            "request_id": result["request_id"],
            "execution_time": result["execution_time"],
            "tools_used": result.get("tools_used", []),
            "complexity": result.get("complexity", "unknown"),
            "metadata": result.get("metadata", {})
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/agentic/chat")
async def agentic_chat(request: ChatRequest):
    """Chat interface with Artisan AI"""
    if not AGENTIC_AVAILABLE or not orchestrator_service:
        # Mock chat response
        return {
            "response": f"안녕하세요! 목업 AI 향수 아티스트입니다. '{request.message}'에 대해 답변드리겠습니다. 어떤 향수를 원하시는지 더 자세히 말씀해주세요!",
            "request_id": f"chat_mock_{int(asyncio.get_event_loop().time())}",
            "session_id": request.session_id or f"session_{int(asyncio.get_event_loop().time())}",
            "tools_used": [],
            "execution_time": 1.0
        }

    try:
        # Create context for chat interaction
        chat_context = {
            "interaction_type": "chat",
            "session_id": request.session_id,
            "message_count": request.context.get("message_count", 0) if request.context else 0
        }

        if request.context:
            chat_context.update(request.context)

        result = await orchestrator_service.process_fragrance_request(
            user_query=request.message,
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/agentic/status")
async def get_agentic_status():
    """Get agentic system status"""
    if not AGENTIC_AVAILABLE or not orchestrator_service:
        return {
            "status": "mock_mode",
            "agentic_available": False,
            "message": "Agentic system not available, using mock responses"
        }

    try:
        status = await orchestrator_service.get_service_status()
        return {
            "status": "operational",
            "system_info": status,
            "agentic_available": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "agentic_available": False
        }

if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Agentic API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)