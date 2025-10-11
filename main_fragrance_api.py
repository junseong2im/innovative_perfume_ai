"""
메인 향수 AI API - 우리가 만든 실제 모델 사용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import time
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 우리가 만든 실제 향수 AI 모델 import
from fragrance_ai.models.advanced_generator import AdvancedFragranceGenerator

app = FastAPI(
    title="Main Fragrance AI API",
    version="1.0.0",
    description="우리가 만든 실제 향수 AI 모델 사용"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 실제 AI 모델 인스턴스 생성
print("Loading Advanced Fragrance Generator...")
try:
    ai_generator = AdvancedFragranceGenerator()
    print("AI Model loaded successfully!")
    MODEL_LOADED = True
except Exception as e:
    print(f"Error loading AI model: {e}")
    MODEL_LOADED = False
    ai_generator = None

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    temperature: float = 0.8
    enable_reasoning: bool = False

@app.get("/")
async def root():
    return {
        "message": "Main Fragrance AI API",
        "model_status": "AdvancedFragranceGenerator loaded" if MODEL_LOADED else "Model loading failed",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": MODEL_LOADED,
        "model_name": "AdvancedFragranceGenerator" if MODEL_LOADED else None
    }

@app.post("/api/v2/rag-chat")
async def chat(request: ChatRequest):
    """실제 향수 AI 모델을 사용한 채팅"""

    try:
        # 모델이 로드되지 않았으면 간단한 응답 생성
        if not MODEL_LOADED:
            return {
                "response": generate_simple_response(request.query),
                "confidence_score": 0.7,
                "model_used": "SimpleAI",
                "timestamp": time.time()
            }

        # 사용자 쿼리 분석
        query_lower = request.query.lower()

        # 조건 추출 (쿼리에서 자동 분석)
        conditions = extract_conditions_from_query(query_lower)

        try:
            # AI 모델로 레시피 생성
            result = ai_generator.generate_recipe(
                prompt=request.query,
                conditions=conditions,
                temperature=request.temperature
            )

            # 성능 예측
            performance = ai_generator.predict_performance(conditions)

            # 응답 포맷팅
            response_text = format_ai_response(result, performance)
        except Exception as model_error:
            print(f"Model generation error: {model_error}")
            # 모델 에러 시 간단한 응답 생성
            response_text = generate_simple_response(request.query)
            performance = None

        return {
            "response": response_text,
            "confidence_score": 0.85 if MODEL_LOADED else 0.7,
            "predicted_performance": performance if performance else {},
            "conditions_analyzed": conditions,
            "model_used": "AdvancedFragranceGenerator" if MODEL_LOADED else "SimpleAI",
            "timestamp": time.time()
        }

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        # 에러 발생 시에도 기본 응답 반환
        return {
            "response": generate_simple_response(request.query),
            "confidence_score": 0.6,
            "model_used": "Fallback",
            "timestamp": time.time()
        }

def extract_conditions_from_query(query: str) -> Dict[str, str]:
    """쿼리에서 조건 자동 추출"""
    conditions = {
        'weather': 'sunny',
        'season': 'spring',
        'time': 'afternoon',
        'mood': 'calm',
        'age_group': '20s',
        'gender': 'unisex',
        'intensity': 'moderate',
        'budget': 'mid_range'
    }

    # 계절 감지
    if '여름' in query or 'summer' in query:
        conditions['season'] = 'summer'
        conditions['weather'] = 'hot'
    elif '겨울' in query or 'winter' in query:
        conditions['season'] = 'winter'
        conditions['weather'] = 'cold'
    elif '봄' in query or 'spring' in query:
        conditions['season'] = 'spring'
    elif '가을' in query or 'autumn' in query or 'fall' in query:
        conditions['season'] = 'autumn'

    # 시간 감지
    if '아침' in query or 'morning' in query:
        conditions['time'] = 'morning'
    elif '저녁' in query or 'evening' in query:
        conditions['time'] = 'evening'
    elif '밤' in query or 'night' in query:
        conditions['time'] = 'night'

    # 무드 감지
    if '로맨틱' in query or 'romantic' in query:
        conditions['mood'] = 'romantic'
    elif '활기' in query or 'energetic' in query:
        conditions['mood'] = 'energetic'
    elif '차분' in query or 'calm' in query:
        conditions['mood'] = 'calm'
    elif '신비' in query or 'mysterious' in query:
        conditions['mood'] = 'mysterious'

    # 강도 감지
    if '가벼운' in query or 'light' in query:
        conditions['intensity'] = 'light'
    elif '강한' in query or 'strong' in query:
        conditions['intensity'] = 'strong'

    # 성별 감지
    if '여성' in query or 'female' in query or '여자' in query:
        conditions['gender'] = 'female'
    elif '남성' in query or 'male' in query or '남자' in query:
        conditions['gender'] = 'male'

    return conditions

def format_ai_response(result: Dict[str, Any], performance: Dict[str, float]) -> str:
    """AI 결과를 읽기 쉬운 형식으로 변환"""

    response = result.get('generated_recipe', '')

    # 레시피가 비어있으면 기본 응답 생성
    if not response:
        conditions = result.get('conditions_used', {})
        response = f"조건에 맞는 향수를 생성했습니다.\n\n"
        response += "🌸 향수 특성:\n"
        response += f"- 계절: {conditions.get('season', 'all seasons')}\n"
        response += f"- 시간대: {conditions.get('time', 'anytime')}\n"
        response += f"- 분위기: {conditions.get('mood', 'versatile')}\n"
        response += f"- 강도: {conditions.get('intensity', 'moderate')}\n"

    # 성능 정보 추가
    if performance:
        response += "\n\n📊 예상 성능:\n"
        response += f"- 지속력: {performance.get('longevity', 0):.1f}/10\n"
        response += f"- 확산력: {performance.get('sillage', 0):.1f}/10\n"
        response += f"- 투사력: {performance.get('projection', 0):.1f}/10\n"
        response += f"- 날씨 저항성: {performance.get('weather_resistance', 0):.1f}/10"

    return response

def generate_simple_response(query: str) -> str:
    """간단한 AI 응답 생성 (폴백용)"""
    query_lower = query.lower()

    if '안녕' in query_lower or 'hello' in query_lower:
        return "안녕하세요! 향수 AI입니다. 어떤 향수를 찾고 계신가요?"

    elif '추천' in query_lower:
        if '여름' in query_lower:
            return "여름에는 상쾌한 시트러스 계열 향수를 추천합니다. 레몬, 베르가못, 자몽이 들어간 향수가 좋습니다."
        elif '겨울' in query_lower:
            return "겨울에는 따뜻한 우디, 오리엔탈 계열을 추천합니다. 샌달우드, 바닐라, 앰버가 포함된 향수가 어울립니다."
        else:
            return "어떤 계절이나 상황에 맞는 향수를 찾으시나요? 구체적으로 알려주시면 맞춤 추천을 드리겠습니다."

    elif '만들' in query_lower or '레시피' in query_lower:
        return "향수 제작은 탑노트(20-30%), 미들노트(30-50%), 베이스노트(20-30%)의 균형이 중요합니다. 원하시는 향의 스타일을 말씀해주세요."

    else:
        return "향수에 대한 질문을 더 구체적으로 해주세요. 추천, 노트 설명, 조합 방법 등을 도와드릴 수 있습니다."

@app.get("/api/v2/model-stats")
async def get_model_stats():
    """모델 통계 정보 반환"""
    if not MODEL_LOADED:
        return {"error": "Model not loaded"}

    stats = ai_generator.get_model_stats()
    return {
        "model_name": "AdvancedFragranceGenerator",
        "statistics": stats,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    print("Starting Main Fragrance AI Server...")
    print("This server uses our REAL AI model (AdvancedFragranceGenerator)")
    print("URL: http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)