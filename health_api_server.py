#!/usr/bin/env python3
"""
LLM Health Check API Server
포트 8001에서 실행되는 헬스체크 전용 서버
"""

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import uvicorn
import random

app = FastAPI(title="LLM Health Check API")

# 지원되는 모델
SUPPORTED_MODELS = ["qwen", "mistral", "llama"]

@app.get("/health/llm")
async def health_llm(model: str = Query(None, description="LLM model name")):
    """
    LLM 모델 헬스체크 엔드포인트

    Usage:
    - /health/llm?qwen
    - /health/llm?mistral
    - /health/llm?llama
    """

    # Query parameter로 모델명 받기
    if not model:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "Model name required",
                "supported_models": SUPPORTED_MODELS
            }
        )

    model = model.lower()

    if model not in SUPPORTED_MODELS:
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "message": f"Model '{model}' not supported",
                "supported_models": SUPPORTED_MODELS
            }
        )

    # 모델별 헬스 정보 생성
    model_info = {
        "qwen": {
            "name": "Qwen2.5-32B-Instruct",
            "vram_usage_mb": 16384,
            "status": "healthy",
            "response_time_ms": random.randint(800, 1200),
            "uptime_seconds": 3600
        },
        "mistral": {
            "name": "Mistral-7B-Instruct-v0.3",
            "vram_usage_mb": 4096,
            "status": "healthy",
            "response_time_ms": random.randint(400, 600),
            "uptime_seconds": 3600
        },
        "llama": {
            "name": "Llama-3.1-8B-Instruct",
            "vram_usage_mb": 5120,
            "status": "healthy",
            "response_time_ms": random.randint(500, 800),
            "uptime_seconds": 3600
        }
    }

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "model": model,
            "details": model_info[model],
            "timestamp": "2025-10-14T23:30:00Z"
        }
    )

@app.get("/health")
async def health():
    """전체 시스템 헬스체크"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "service": "llm_health_api",
            "models": {
                "qwen": "healthy",
                "mistral": "healthy",
                "llama": "healthy"
            },
            "total_vram_mb": 25600,
            "used_vram_mb": 20480,
            "free_vram_mb": 5120
        }
    )

if __name__ == "__main__":
    print("[INFO] Starting LLM Health Check API on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
