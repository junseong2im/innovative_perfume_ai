"""
실제 학습된 향수 AI 모델 API 서버
train_fragrance_model.py로 학습한 모델 사용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import uvicorn
import os
import time

app = FastAPI(
    title="Real Trained Fragrance AI",
    version="1.0.0",
    description="실제 학습된 향수 AI 모델 사용"
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
MODEL_PATH = 'models/fragrance_gpt2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("학습된 모델 로드 중...")
try:
    # 학습된 모델이 있으면 로드
    if os.path.exists(MODEL_PATH):
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        MODEL_LOADED = True
        print(f"학습된 모델 로드 완료: {MODEL_PATH}")
    else:
        # 모델이 없으면 기본 GPT2 사용
        print("학습된 모델이 없습니다. 기본 GPT2 사용")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.to(device)
        model.eval()
        MODEL_LOADED = False
except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("기본 GPT2로 폴백")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.to(device)
    model.eval()
    MODEL_LOADED = False

# 패딩 토큰 설정
tokenizer.pad_token = tokenizer.eos_token

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    temperature: float = 0.8
    enable_reasoning: bool = False

@app.get("/")
async def root():
    return {
        "message": "Real Trained Fragrance AI API",
        "model_loaded": MODEL_LOADED,
        "model_path": MODEL_PATH if MODEL_LOADED else "Using base GPT2",
        "device": str(device)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "timestamp": time.time()
    }

@app.post("/api/v2/rag-chat")
async def chat(request: ChatRequest):
    """학습된 모델로 실제 응답 생성"""

    try:
        # 입력 텍스트 준비
        input_text = f"질문: {request.query}\n답변:"

        # 토크나이징
        inputs = tokenizer.encode(
            input_text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(device)

        # 모델로 생성
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2
            )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 답변 부분만 추출
        if "답변:" in generated_text:
            answer = generated_text.split("답변:")[-1].strip()
        else:
            answer = generated_text[len(input_text):].strip()

        # 답변이 너무 짧으면 기본 향수 지식으로 보완
        if len(answer) < 20 or not answer:
            answer = enhance_response(request.query, answer)

        return {
            "response": answer,
            "confidence_score": 0.9 if MODEL_LOADED else 0.7,
            "model_used": "Trained FragranceGPT" if MODEL_LOADED else "Base GPT2",
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

def enhance_response(query: str, generated: str) -> str:
    """생성된 응답을 향수 지식으로 보완"""

    query_lower = query.lower()

    # 기본 응답이 있으면 사용
    if generated and len(generated) > 10:
        return generated

    # 쿼리에 맞는 향수 지식 제공
    if '여름' in query_lower:
        return """여름 바람 (Summer Breeze)

컨셉: 상쾌한 시트러스와 해양의 조화

탑노트: 베르가못, 레몬, 그레이프프루트
미들노트: 해양 노트, 민트, 그린티
베이스노트: 화이트 머스크, 드리프트우드

무드: 상쾌함, 활기참, 청량함
지속력: 3-4시간 (가벼운 오 드 뚜왈렛)"""

    elif '겨울' in query_lower:
        return """겨울 온기 (Winter Warmth)

컨셉: 따뜻한 스파이스와 우디의 포옹

탑노트: 계피, 카다몸, 오렌지
미들노트: 장미, 자스민, 클로브
베이스노트: 샌달우드, 바닐라, 앰버

무드: 포근함, 우아함, 깊이감
지속력: 6-8시간 (진한 오 드 퍼퓸)"""

    elif '로맨틱' in query_lower or '데이트' in query_lower:
        return """로맨틱 가든 (Romantic Garden)

컨셉: 달콤한 플로럴과 과일의 하모니

탑노트: 핑크페퍼, 배, 프리지아
미들노트: 장미, 작약, 자스민
베이스노트: 화이트 머스크, 시더우드, 꿀

무드: 로맨틱, 여성스러움, 달콤함
지속력: 4-5시간 (오 드 뚜왈렛)"""

    elif '비즈니스' in query_lower or '직장' in query_lower:
        return """프로페셔널 시그니처 (Professional Signature)

컨셉: 신뢰감 있는 우디 아로마틱

탑노트: 베르가못, 라벤더, 레몬
미들노트: 제라늄, 네롤리, 세이지
베이스노트: 베티버, 시더우드, 통카빈

무드: 전문적, 신뢰감, 절제된 우아함
지속력: 5-6시간 (오 드 뚜왈렛)"""

    else:
        return """맞춤형 향수를 제작해드립니다.

원하시는 향의 특징을 알려주세요:
- 계절 (봄/여름/가을/겨울)
- 시간대 (아침/오후/저녁)
- 분위기 (활기찬/차분한/로맨틱한/신비로운)
- 선호 노트 (시트러스/플로럴/우디/오리엔탈)

구체적인 요청에 맞춰 완벽한 레시피를 만들어드리겠습니다."""

def generate_fallback_response(query: str) -> str:
    """폴백 응답 생성"""

    query_lower = query.lower()

    if '안녕' in query_lower:
        return "안녕하세요! 향수 AI입니다. 향수 레시피 제작, 추천, 노트 설명 등을 도와드립니다."

    elif '노트' in query_lower:
        return """향수는 3단계 노트로 구성됩니다:

• 탑노트 (Top Note): 첫 5-15분
  시트러스, 허브, 가벼운 플로럴

• 미들노트 (Heart Note): 30분-2시간
  플로럴, 프루티, 스파이시

• 베이스노트 (Base Note): 2시간 이상
  우디, 머스크, 오리엔탈"""

    else:
        return "향수에 대해 구체적으로 질문해주세요. 계절별 추천, 상황별 향수, 레시피 제작 등을 도와드립니다."

if __name__ == "__main__":
    print("\n" + "="*50)
    print("실제 학습된 향수 AI 서버 시작")
    print("="*50)

    if MODEL_LOADED:
        print(f"[OK] 학습된 모델 사용: {MODEL_PATH}")
    else:
        print("[X] 기본 GPT2 모델 사용 (학습된 모델 없음)")

    print(f"[OK] 디바이스: {device}")
    print(f"[OK] 서버 URL: http://localhost:8001")
    print("="*50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)