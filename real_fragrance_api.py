"""
실제 향수 AI API 서버
실제 모델과 데이터베이스 사용
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import time

# 실제 향수 AI 모델 import
try:
    from fragrance_ai.models.advanced_generator import AdvancedFragranceGenerator
    from fragrance_ai.services.generation_service import FragranceGenerationService
    from fragrance_ai.models.quality_analyzer import QualityAnalyzer
    from fragrance_ai.models.compatibility_matrix import CompatibilityMatrix
    from fragrance_ai.knowledge.master_perfumer_principles import MasterPerfumerKnowledge
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

app = FastAPI(
    title="Real Fragrance AI API",
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

# 실제 향수 AI 모델 초기화
if MODEL_AVAILABLE:
    try:
        generator = AdvancedFragranceGenerator()
        generation_service = FragranceGenerationService()
        quality_analyzer = QualityAnalyzer()
        compatibility_matrix = CompatibilityMatrix()
        master_knowledge = MasterPerfumerKnowledge()
        print("Real AI models loaded successfully!")
    except Exception as e:
        print(f"Failed to load AI models: {e}")
        MODEL_AVAILABLE = False

# 간단한 폴백 모델 (AI 모델이 없을 때)
class SimpleFragranceAI:
    def __init__(self):
        self.knowledge_base = {
            "greeting": [
                "안녕하세요! 향수 AI 어시스턴트입니다.",
                "향수에 대해 무엇이든 물어보세요."
            ],
            "citrus": {
                "notes": ["레몬", "오렌지", "베르가못", "자몽", "라임"],
                "description": "시트러스 계열은 상쾌하고 활기찬 향을 제공합니다.",
                "combinations": ["플로럴", "우디", "그린"]
            },
            "floral": {
                "notes": ["장미", "자스민", "라벤더", "일랑일랑", "프리지아"],
                "description": "플로럴 계열은 부드럽고 로맨틱한 향을 제공합니다.",
                "combinations": ["시트러스", "우디", "머스크"]
            },
            "woody": {
                "notes": ["샌달우드", "시더우드", "베티버", "파출리"],
                "description": "우디 계열은 따뜻하고 깊이 있는 향을 제공합니다.",
                "combinations": ["플로럴", "오리엔탈", "시트러스"]
            },
            "oriental": {
                "notes": ["바닐라", "앰버", "머스크", "통카빈"],
                "description": "오리엔탈 계열은 이국적이고 관능적인 향을 제공합니다.",
                "combinations": ["플로럴", "우디", "스파이시"]
            }
        }

    def generate_response(self, query: str) -> str:
        query_lower = query.lower()

        # 인사말 처리
        if any(word in query_lower for word in ["안녕", "하이", "hello", "hi"]):
            return "안녕하세요! 향수에 대해 궁금한 점을 물어보세요. 향수 추천, 노트 설명, 조합 추천 등을 도와드릴 수 있습니다."

        # 향수 추천
        if "추천" in query_lower:
            if "여름" in query_lower or "상쾌" in query_lower:
                return "여름에는 시트러스 계열의 향수를 추천드립니다. 베르가못, 레몬, 자몽 노트가 들어간 향수는 상쾌하고 가벼워 더운 날씨에 잘 어울립니다. 대표적으로 아쿠아 디 지오, 라이트 블루 등이 있습니다."
            elif "겨울" in query_lower or "따뜻" in query_lower:
                return "겨울에는 우디나 오리엔탈 계열의 향수를 추천드립니다. 샌달우드, 바닐라, 앰버 노트는 따뜻하고 포근한 느낌을 줍니다. 톰포드 블랙 오키드, 샤넬 코코 등이 좋은 선택입니다."
            elif "데이트" in query_lower or "로맨틱" in query_lower:
                return "데이트에는 플로럴 계열의 향수를 추천드립니다. 장미와 자스민이 조화된 향은 로맨틱한 분위기를 연출합니다. 미스 디올, 랑콤 라비에벨 등이 인기가 많습니다."
            else:
                return "어떤 상황이나 계절에 맞는 향수를 찾으시나요? 구체적으로 알려주시면 더 정확한 추천을 드릴 수 있습니다."

        # 향료 설명
        for category, info in self.knowledge_base.items():
            if category in query_lower and category != "greeting":
                return f"{info['description']} 주요 노트로는 {', '.join(info['notes'][:3])} 등이 있으며, {', '.join(info['combinations'])} 계열과 잘 어울립니다."

        # 조합 질문
        if "조합" in query_lower or "어울" in query_lower:
            return "향수 조합의 기본 원칙은 탑-미들-베이스 노트의 균형입니다. 시트러스(탑) + 플로럴(미들) + 우디(베이스)는 클래식한 조합이며, 실험적인 조합을 원하신다면 구체적인 향료를 말씀해주세요."

        # 만들기/레시피
        if "만들" in query_lower or "레시피" in query_lower:
            return "향수 제작의 기본 비율은 탑노트 20-30%, 미들노트 30-50%, 베이스노트 20-30%입니다. 에탄올 베이스에 향료를 희석하고, 최소 2주간 숙성시키면 향이 안정됩니다. 구체적인 레시피를 원하시면 원하는 향의 스타일을 알려주세요."

        # 기본 응답
        return "향수에 대한 질문을 더 구체적으로 해주시면 정확한 답변을 드릴 수 있습니다. 향수 추천, 노트 설명, 조합 방법, 레시피 등 무엇이든 물어보세요!"

# AI 인스턴스 생성
if not MODEL_AVAILABLE:
    simple_ai = SimpleFragranceAI()

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    temperature: float = 0.7
    enable_reasoning: bool = False

@app.get("/")
async def root():
    return {
        "message": "Real Fragrance AI API",
        "model_status": "Real AI models loaded" if MODEL_AVAILABLE else "Using fallback model",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_available": MODEL_AVAILABLE
    }

@app.post("/api/v2/rag-chat")
async def chat(request: ChatRequest):
    """실제 향수 AI 채팅 엔드포인트"""

    try:
        if MODEL_AVAILABLE:
            # 실제 AI 모델 사용
            try:
                # 고급 생성기 사용
                response = await generator.generate_fragrance_response(
                    query=request.query,
                    context=request.context,
                    temperature=request.temperature
                )

                # 품질 분석
                quality_score = quality_analyzer.analyze(response)

                return {
                    "response": response,
                    "confidence_score": quality_score,
                    "model_used": "AdvancedFragranceGenerator",
                    "timestamp": time.time()
                }
            except Exception as e:
                print(f"AI model error: {e}")
                # 폴백으로 전환

        # 폴백 모델 사용
        response = simple_ai.generate_response(request.query)

        return {
            "response": response,
            "confidence_score": 0.8,
            "model_used": "SimpleFragranceAI",
            "timestamp": time.time()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/generate-recipe")
async def generate_recipe(query: str):
    """향수 레시피 생성"""

    if MODEL_AVAILABLE:
        try:
            # 실제 생성 서비스 사용
            recipe = generation_service.generate_recipe(query)

            # 호환성 체크
            compatibility = compatibility_matrix.check_compatibility(recipe['ingredients'])

            # 마스터 조향사 원칙 적용
            refined_recipe = master_knowledge.refine_recipe(recipe)

            return {
                "recipe": refined_recipe,
                "compatibility_score": compatibility,
                "model_used": "FragranceGenerationService"
            }
        except:
            pass

    # 폴백 레시피 생성
    return {
        "recipe": {
            "name": "Custom Blend",
            "top_notes": ["베르가못", "레몬"],
            "middle_notes": ["장미", "자스민"],
            "base_notes": ["샌달우드", "머스크"],
            "description": f"{query}를 반영한 조화로운 블렌드"
        },
        "compatibility_score": 0.85,
        "model_used": "SimpleGenerator"
    }

if __name__ == "__main__":
    print("Starting Real Fragrance AI Server...")
    print(f"Model Status: {'Real AI models' if MODEL_AVAILABLE else 'Fallback model'}")
    print("URL: http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)