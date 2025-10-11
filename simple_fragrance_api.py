"""
간단한 향수 AI API 서버
테스트용 최소 구현
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import random
import time

app = FastAPI(
    title="Fragrance AI Test API",
    version="1.0.0",
    description="향수 AI 테스트용 간단한 API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델
class RecipeRequest(BaseModel):
    description: str
    base_notes: List[str] = []
    complexity_level: int = 5
    quantity_ml: int = 50

class SearchRequest(BaseModel):
    query: str
    search_type: str = "semantic"
    top_k: int = 5
    min_similarity: float = 0.5
    enable_reranking: bool = True
    use_cache: bool = True
    collections: List[str] = []

class ChatRequest(BaseModel):
    query: str
    context: Optional[str] = None
    temperature: float = 0.7
    enable_reasoning: bool = False

# 샘플 향수 데이터베이스
FRAGRANCE_DB = {
    "citrus": ["레몬", "오렌지", "베르가못", "자몽", "라임", "유자"],
    "floral": ["장미", "자스민", "라벤더", "일랑일랑", "프리지아", "피오니"],
    "woody": ["샌달우드", "시더우드", "베티버", "파출리", "오크모스"],
    "oriental": ["바닐라", "앰버", "머스크", "통카빈", "벤조인"],
    "fresh": ["민트", "바질", "그린티", "오이", "알로에"],
    "fruity": ["피치", "애플", "베리", "멜론", "파인애플", "코코넛"]
}

PERFUME_BRANDS = [
    "샤넬 No.5", "디올 미스 디올", "조말론 잉글리시 페어",
    "톰포드 블랙 오키드", "크리드 아벤투스", "르라보 상탈 33",
    "바이레도 집시 워터", "메종 마르지엘라 레플리카 비치 워크"
]

@app.get("/")
async def root():
    return {
        "message": "🌸 Fragrance AI Test API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": {
            "embedding_model": True,
            "rag_system": True,
            "cache_manager": True,
            "performance_optimizer": True,
            "batch_processors": True
        },
        "all_systems_ready": True,
        "cache_stats": {
            "hit_rate": 0.85,
            "cache_size": 1024
        }
    }

@app.post("/api/v2/public/recipes/generate")
async def generate_recipe(request: RecipeRequest):
    """향수 레시피 생성"""

    # 복잡도에 따라 재료 수 결정
    num_ingredients = min(3 + request.complexity_level // 2, 8)

    # 랜덤 재료 선택
    all_notes = []
    for category in FRAGRANCE_DB.values():
        all_notes.extend(category)

    selected_ingredients = random.sample(all_notes, num_ingredients)

    # 비율 계산
    total = 100
    percentages = []
    for i in range(num_ingredients - 1):
        p = random.randint(5, total - (num_ingredients - i - 1) * 5)
        percentages.append(p)
        total -= p
    percentages.append(total)
    random.shuffle(percentages)

    # 노트 분류
    top_notes = selected_ingredients[:num_ingredients//3]
    middle_notes = selected_ingredients[num_ingredients//3:2*num_ingredients//3]
    base_notes = selected_ingredients[2*num_ingredients//3:]

    recipe = {
        "name": f"AI 블렌드 #{random.randint(1000, 9999)}",
        "description": f"{request.description}를 표현한 독특한 조합",
        "ingredients": [
            {"name": ing, "percentage": perc}
            for ing, perc in zip(selected_ingredients, percentages)
        ],
        "top_notes": top_notes,
        "middle_notes": middle_notes,
        "base_notes": base_notes + request.base_notes,
        "complexity": request.complexity_level,
        "volume_ml": request.quantity_ml
    }

    return {
        "success": True,
        "recipe": recipe,
        "generated_at": time.time()
    }

@app.post("/api/v2/semantic-search")
async def semantic_search(request: SearchRequest):
    """시맨틱 검색"""

    # 간단한 키워드 매칭 시뮬레이션
    results = []
    query_lower = request.query.lower()

    # 향수 브랜드 검색
    for brand in PERFUME_BRANDS:
        if any(word in brand.lower() for word in query_lower.split()):
            similarity = random.uniform(0.7, 0.95)
            results.append({
                "id": f"doc_{len(results)}",
                "document": f"{brand} - 프리미엄 향수",
                "similarity": similarity,
                "metadata": {"type": "perfume", "brand": brand},
                "collection": "perfumes",
                "rank": len(results) + 1
            })

    # 노트 검색
    for category, notes in FRAGRANCE_DB.items():
        if category in query_lower or any(note in query_lower for note in notes):
            for note in notes[:3]:  # 각 카테고리에서 최대 3개
                similarity = random.uniform(0.6, 0.9)
                results.append({
                    "id": f"doc_{len(results)}",
                    "document": f"{note} - {category} 계열 향료",
                    "similarity": similarity,
                    "metadata": {"type": "note", "category": category},
                    "collection": "ingredients",
                    "rank": len(results) + 1
                })

    # top_k로 제한
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:request.top_k]

    return {
        "results": results,
        "total_results": len(results),
        "query": request.query,
        "search_time": random.uniform(0.1, 0.3),
        "cached": request.use_cache and random.choice([True, False]),
        "reranked": request.enable_reranking
    }

@app.post("/api/v2/rag-chat")
async def rag_chat(request: ChatRequest):
    """RAG 기반 채팅"""

    query_lower = request.query.lower()

    # 간단한 응답 생성
    response_templates = {
        "추천": "고객님의 취향을 고려할 때, {perfume}을(를) 추천드립니다. 이 향수는 {notes}의 조화로운 블렌드로 {occasion}에 적합합니다.",
        "레시피": "{base_note}를 베이스로 {top_note}와 {middle_note}를 조합하면 훌륭한 향을 만들 수 있습니다. 비율은 30:40:30이 이상적입니다.",
        "호환성": "{note1}과(와) {note2}는 excellent한 조합입니다. 두 향료는 서로의 특성을 보완하며 조화로운 향을 만들어냅니다.",
        "일반": "향수는 개인의 체온과 피부 pH에 따라 다르게 발향됩니다. 시향 후 최소 30분은 기다려보시는 것을 권장합니다."
    }

    # 키워드에 따른 응답 선택
    if "추천" in query_lower or "비슷한" in query_lower:
        perfume = random.choice(PERFUME_BRANDS)
        notes = ", ".join(random.sample([n for cat in FRAGRANCE_DB.values() for n in cat], 3))
        occasion = random.choice(["데일리", "특별한 날", "비즈니스", "로맨틱한 저녁"])
        response = response_templates["추천"].format(
            perfume=perfume, notes=notes, occasion=occasion
        )
    elif "레시피" in query_lower or "만들" in query_lower:
        base = random.choice(FRAGRANCE_DB["woody"])
        top = random.choice(FRAGRANCE_DB["citrus"])
        middle = random.choice(FRAGRANCE_DB["floral"])
        response = response_templates["레시피"].format(
            base_note=base, top_note=top, middle_note=middle
        )
    elif "호환" in query_lower or "조합" in query_lower:
        note1 = random.choice(FRAGRANCE_DB["floral"])
        note2 = random.choice(FRAGRANCE_DB["woody"])
        response = response_templates["호환성"].format(note1=note1, note2=note2)
    else:
        response = response_templates["일반"]

    # 추가 정보 생성
    reasoning_steps = []
    if request.enable_reasoning:
        reasoning_steps = [
            "사용자 질문 분석 및 의도 파악",
            "관련 향수 데이터베이스 검색",
            "유사도 기반 매칭 수행",
            "최적 답변 생성"
        ]

    source_docs = [
        f"향수 가이드: {random.choice(['조향 기초', '향료 배합법', '향수 역사'])}"
        for _ in range(3)
    ]

    return {
        "response": response,
        "confidence_score": random.uniform(0.85, 0.95),
        "source_documents": source_docs,
        "reasoning_steps": reasoning_steps,
        "retrieval_info": {
            "retrieval_time": random.uniform(0.1, 0.3),
            "documents_retrieved": len(source_docs),
            "avg_similarity": random.uniform(0.7, 0.9)
        },
        "response_time": random.uniform(0.5, 1.5),
        "timestamp": time.time()
    }

@app.get("/api/v2/performance")
async def get_performance():
    """성능 메트릭"""
    return {
        "timestamp": time.time(),
        "performance": {
            "avg_response_time": random.uniform(100, 300),
            "requests_per_second": random.randint(10, 50),
            "cache_hit_rate": random.uniform(0.7, 0.9)
        },
        "cache": {
            "hit_rate": random.uniform(0.8, 0.95),
            "cache_size": random.randint(500, 2000),
            "hot_keys": [
                "citrus_perfume",
                "floral_blend",
                "summer_fragrance"
            ]
        },
        "models": {
            "knowledge_base": {
                "total_documents": 10000,
                "indexed": True,
                "last_update": time.time() - 3600
            }
        },
        "system_info": {
            "version": "1.0.0",
            "uptime": random.randint(1000, 10000)
        }
    }

@app.post("/api/v2/auth/login")
async def login():
    """인증 테스트"""
    return {
        "access_token": "test_token_" + str(random.randint(1000, 9999)),
        "token_type": "bearer",
        "expires_in": 3600
    }

if __name__ == "__main__":
    print("Fragrance AI Test Server Starting...")
    print("URL: http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)