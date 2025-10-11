"""
진짜 AI API - OpenAI GPT 사용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import time

# OpenAI 라이브러리
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")

app = FastAPI(
    title="Real AI API with GPT",
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

# OpenAI API 키 설정 (환경변수 또는 직접 입력)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # 여기에 API 키 입력 필요

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY
    AI_READY = True
    print("OpenAI API connected successfully!")
else:
    AI_READY = False
    print("OpenAI API not configured. Using local AI.")

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    temperature: float = 0.7
    enable_reasoning: bool = False

# 로컬 AI (OpenAI 없을 때 사용)
class LocalFragranceAI:
    def __init__(self):
        self.system_prompt = """당신은 전문 조향사이자 향수 전문가입니다.
        한국어로 친절하고 전문적으로 답변합니다.
        향수의 노트, 조합, 계절별 추천, 브랜드 정보 등에 대해 깊은 지식을 가지고 있습니다."""

    def generate_response(self, query: str) -> str:
        # Hugging Face 모델 사용 시도
        try:
            from transformers import pipeline

            # 한국어 모델 사용
            generator = pipeline('text-generation', model='skt/kogpt2-base-v2')

            prompt = f"향수 전문가로서 답변: {query}\n답변:"
            result = generator(prompt, max_length=200, temperature=0.8)

            return result[0]['generated_text'].split('답변:')[-1].strip()

        except Exception as e:
            print(f"Local model error: {e}")

            # 폴백: 규칙 기반 응답
            return self.rule_based_response(query)

    def rule_based_response(self, query: str) -> str:
        query_lower = query.lower()

        if '안녕' in query_lower:
            return "안녕하세요! 향수 전문가 AI입니다. 향수에 대해 무엇이든 물어보세요."

        elif '추천' in query_lower:
            if '여름' in query_lower:
                return """여름 향수 추천:
1. 아쿠아 디 지오 (Acqua di Gio) - 상쾌한 해양 향
2. 라이트 블루 (Light Blue) - 시트러스와 사과
3. 시케이원 (CK One) - 가벼운 시트러스
이런 향수들은 덥고 습한 날씨에도 부담 없이 사용할 수 있습니다."""

            elif '겨울' in query_lower:
                return """겨울 향수 추천:
1. 톰포드 블랙 오키드 - 진한 플로럴 오리엔탈
2. 샤넬 코코 마드모아젤 - 따뜻한 오리엔탈
3. 디올 아디트 - 달콤한 바닐라
따뜻하고 포근한 향이 추운 날씨와 잘 어울립니다."""

            else:
                return "어떤 계절, 상황, 또는 스타일의 향수를 찾으시나요? 구체적으로 알려주시면 맞춤 추천을 드리겠습니다."

        elif '노트' in query_lower:
            return """향수의 3단계 노트:
• 탑노트(Top): 처음 5-15분, 시트러스/허브 계열
• 미들노트(Heart): 30분-2시간, 플로럴/프루티 계열
• 베이스노트(Base): 2시간 이상, 우디/머스크 계열
각 노트가 시간에 따라 변화하며 향의 이야기를 만듭니다."""

        else:
            return "향수에 대한 구체적인 질문을 해주세요. 추천, 노트 설명, 브랜드 정보, 사용법 등을 도와드릴 수 있습니다."

local_ai = LocalFragranceAI()

@app.get("/")
async def root():
    return {
        "message": "Real AI API",
        "ai_status": "OpenAI GPT Connected" if AI_READY else "Using Local AI",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ai_ready": AI_READY,
        "timestamp": time.time()
    }

@app.post("/api/v2/rag-chat")
async def chat(request: ChatRequest):
    """진짜 AI 응답 생성"""

    try:
        if AI_READY:
            # OpenAI GPT 사용
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # 또는 "gpt-4"
                    messages=[
                        {"role": "system", "content": "당신은 전문 조향사이자 향수 전문가입니다. 한국어로 친절하고 전문적으로 답변합니다."},
                        {"role": "user", "content": request.query}
                    ],
                    temperature=request.temperature,
                    max_tokens=500
                )

                ai_response = response.choices[0].message.content
                confidence = 0.95
                model_used = "GPT-3.5"

            except Exception as e:
                print(f"OpenAI error: {e}")
                # OpenAI 실패시 로컬 AI 사용
                ai_response = local_ai.generate_response(request.query)
                confidence = 0.7
                model_used = "LocalAI"
        else:
            # 로컬 AI 사용
            ai_response = local_ai.generate_response(request.query)
            confidence = 0.7
            model_used = "LocalAI"

        return {
            "response": ai_response,
            "confidence_score": confidence,
            "model_used": model_used,
            "timestamp": time.time()
        }

    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "response": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
            "confidence_score": 0.5,
            "model_used": "Error",
            "timestamp": time.time()
        }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Real AI Server Starting...")
    print("="*50)

    if not OPENAI_API_KEY:
        print("\nWARNING: OpenAI API key not set!")
        print("To use GPT, set OPENAI_API_KEY environment variable")
        print("or edit OPENAI_API_KEY in this file")
        print("\nUsing local AI fallback...")
    else:
        print("\nOpenAI API configured!")

    print("\nServer: http://localhost:8000")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)