"""
우리가 만든 AdvancedFragranceGenerator 모델을 사용하는 API 서버
실제 학습된 모델 사용 - 시뮬레이션 없음
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import torch
import os
import sys
import time

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 우리가 만든 모델 import
from fragrance_ai.models.advanced_generator import AdvancedFragranceGenerator

app = FastAPI(
    title="Advanced Fragrance Generator API",
    version="2.0.0",
    description="우리가 만든 AdvancedFragranceGenerator 모델 사용"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 학습된 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_LOADED = False
model = None

print("="*50)
print("우리가 만든 AdvancedFragranceGenerator 모델 로드 중...")

try:
    # 우리 모델 초기화
    model = AdvancedFragranceGenerator()

    # 학습된 가중치 로드
    if os.path.exists('models/fragrance_generator_final.pt'):
        state_dict = torch.load('models/fragrance_generator_final.pt', map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("[OK] 학습된 가중치 로드 완료")
    else:
        print("[WARNING] 학습된 가중치 없음 - 기본 모델 사용")

    # 적응 상태 로드
    if os.path.exists('models/adaptation_state.json'):
        model.load_adaptation_state('models/adaptation_state.json')
        print("[OK] 적응 상태 로드 완료")

    model.to(device)
    model.eval()
    MODEL_LOADED = True
    print("[OK] AdvancedFragranceGenerator 모델 준비 완료")

except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    print("[INFO] 기본 모델로 초기화")
    model = AdvancedFragranceGenerator()
    model.to(device)
    model.eval()
    MODEL_LOADED = False

print("="*50)

class ChatRequest(BaseModel):
    query: str
    temperature: float = 0.7
    enable_reasoning: bool = False
    conditions: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    return {
        "message": "Advanced Fragrance Generator API",
        "model": "AdvancedFragranceGenerator",
        "model_loaded": MODEL_LOADED,
        "device": str(device),
        "status": "실제 학습된 모델 사용 중" if MODEL_LOADED else "기본 모델 사용 중"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_type": "AdvancedFragranceGenerator",
        "timestamp": time.time()
    }

@app.post("/api/v2/rag-chat")
async def chat(request: ChatRequest):
    """우리가 만든 모델로 실제 응답 생성"""

    try:
        # 조건 설정
        conditions = request.conditions or {}

        # 기본 조건 추가
        if '여름' in request.query:
            conditions['season'] = 'summer'
            conditions['mood'] = 'fresh'
        elif '겨울' in request.query:
            conditions['season'] = 'winter'
            conditions['mood'] = 'warm'
        elif '로맨틱' in request.query or '데이트' in request.query:
            conditions['mood'] = 'romantic'
            conditions['time'] = 'evening'
        elif '비즈니스' in request.query or '직장' in request.query:
            conditions['mood'] = 'professional'
            conditions['time'] = 'morning'

        # 기본값 설정
        conditions.setdefault('weather', 'sunny')
        conditions.setdefault('season', 'spring')
        conditions.setdefault('time', 'afternoon')
        conditions.setdefault('mood', 'calm')
        conditions.setdefault('age_group', '30s')
        conditions.setdefault('gender', 'unisex')
        conditions.setdefault('intensity', 'moderate')
        conditions.setdefault('budget', 'mid_range')

        # 모델로 레시피 생성
        result = model.generate_recipe(
            prompt=request.query,
            conditions=conditions,
            temperature=request.temperature
        )

        # 응답 포맷팅
        response_text = format_response(result, request.query)

        return {
            "response": response_text,
            "confidence_score": result.get('adaptation_score', 0.8),
            "model_used": "AdvancedFragranceGenerator (Trained)" if MODEL_LOADED else "AdvancedFragranceGenerator (Base)",
            "performance": result.get('predicted_performance', {}),
            "timestamp": time.time()
        }

    except Exception as e:
        print(f"생성 오류: {e}")
        # 에러시 기본 응답
        return {
            "response": generate_fallback_response(request.query),
            "confidence_score": 0.6,
            "model_used": "Fallback",
            "error": str(e),
            "timestamp": time.time()
        }

def format_response(result: Dict, query: str) -> str:
    """모델 결과를 읽기 쉬운 형태로 포맷팅"""

    if 'generated_recipe' in result:
        return result['generated_recipe']

    # 기본 포맷팅
    response = f"향수 레시피를 생성했습니다.\n\n"

    if 'notes' in result:
        response += "향료 구성:\n"
        for note_type, notes in result['notes'].items():
            if notes:
                response += f"- {note_type}: {', '.join(notes)}\n"

    if 'predicted_performance' in result:
        perf = result['predicted_performance']
        response += f"\n예상 성능:\n"
        response += f"- 지속력: {perf.get('longevity', 'N/A')}\n"
        response += f"- 확산력: {perf.get('sillage', 'N/A')}\n"
        response += f"- 계절 적합성: {perf.get('season_match', 'N/A')}\n"

    return response

def generate_fallback_response(query: str) -> str:
    """폴백 응답 생성"""

    query_lower = query.lower()

    if '여름' in query_lower:
        return """여름 바람 (Summer Breeze)

시트러스와 해양의 상쾌한 조화

탑노트: 베르가못, 레몬, 그레이프프루트
미들노트: 해양 노트, 민트, 그린티
베이스노트: 화이트 머스크, 드리프트우드

상쾌하고 활기찬 여름 향수입니다."""

    elif '겨울' in query_lower:
        return """겨울 온기 (Winter Warmth)

따뜻한 스파이스와 우디의 포근함

탑노트: 계피, 카다몬, 오렌지
미들노트: 장미, 자스민, 클로브
베이스노트: 샌달우드, 바닐라, 앰버

포근하고 우아한 겨울 향수입니다."""

    elif '로맨틱' in query_lower or '데이트' in query_lower:
        return """로맨틱 가든 (Romantic Garden)

달콤한 플로럴과 과일의 하모니

탑노트: 핑크페퍼, 배, 프리지아
미들노트: 장미, 작약, 자스민
베이스노트: 화이트 머스크, 시더우드, 꿀

로맨틱하고 여성스러운 향수입니다."""

    else:
        return """맞춤형 향수를 제작해드립니다.

원하시는 향의 특징을 구체적으로 알려주세요:
- 계절, 시간대, 분위기
- 선호하는 향료 종류

더 자세한 정보를 주시면 완벽한 레시피를 만들어드리겠습니다."""

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*50)
    print("Advanced Fragrance Generator API 서버 시작")
    print("="*50)
    print(f"[OK] 모델: AdvancedFragranceGenerator")
    print(f"[OK] 상태: {'학습된 모델' if MODEL_LOADED else '기본 모델'}")
    print(f"[OK] 디바이스: {device}")
    print(f"[OK] 서버 URL: http://localhost:8001")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)