"""
간단한 FastAPI 메인 파일 - AI Perfumer 테스트용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import uuid

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Fragrance AI - AI Perfumer",
    description="공감각 조향사 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Perfumer 오케스트레이터 임포트
from ..orchestrator.ai_perfumer_orchestrator import get_ai_perfumer_orchestrator

# 요청/응답 모델
class AIPerfumerRequest(BaseModel):
    """AI 조향사 요청 모델"""
    message: str = Field(..., description="사용자 메시지")
    context: List[str] = Field(default_factory=list, description="대화 맥락")
    session_id: Optional[str] = Field(None, description="세션 ID")

class AIPerfumerResponse(BaseModel):
    """AI 조향사 응답 모델"""
    response: str = Field(..., description="AI 응답")
    fragrance: Optional[Dict[str, Any]] = Field(None, description="생성된 향수")
    timestamp: str = Field(..., description="응답 시간")
    session_id: str = Field(..., description="세션 ID")

@app.get("/health")
async def health_check():
    """헬스체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/v1/ai-perfumer/chat", response_model=AIPerfumerResponse)
async def ai_perfumer_chat(request: AIPerfumerRequest):
    """
    AI 조향사와 대화 (공감각적 향수 창조)

    세상의 모든 개념을 향으로 번역하는 예술가와 대화합니다.
    추상적 개념, 감정, 기억을 향수로 변환합니다.
    """
    try:
        # 오케스트레이터 가져오기
        orchestrator = get_ai_perfumer_orchestrator()

        # 세션 관리
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"

        # 대화 처리
        response = orchestrator.generate_response(
            message=request.message,
            context=request.context
        )

        # 충분한 대화 후 향수 생성
        fragrance = None
        if len(request.context) >= 2:
            full_context = ' '.join(request.context + [request.message])
            fragrance_data = orchestrator.execute_creative_process(full_context)
            fragrance = fragrance_data

        return AIPerfumerResponse(
            response=response,
            fragrance=fragrance,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"AI Perfumer chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "AI Perfumer API - 공감각 조향사 시스템",
        "endpoints": {
            "/health": "헬스체크",
            "/api/v1/ai-perfumer/chat": "AI 조향사와 대화",
            "/docs": "API 문서"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)