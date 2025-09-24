"""
관리자 주문서 조회 서버
고객이 주문한 향수 레시피를 관리자가 조회할 수 있는 API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import glob
from datetime import datetime
from typing import List, Dict, Any

app = FastAPI(title="향수 주문 관리 시스템", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ADMIN_ORDERS_DIR = "admin_orders"

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"message": "향수 주문 관리 시스템이 실행 중입니다", "status": "running"}

@app.get("/api/admin/orders")
async def get_all_orders():
    """모든 주문서 조회"""
    try:
        if not os.path.exists(ADMIN_ORDERS_DIR):
            return {"orders": [], "total": 0}

        orders = []
        order_files = glob.glob(os.path.join(ADMIN_ORDERS_DIR, "*.json"))

        for order_file in order_files:
            with open(order_file, 'r', encoding='utf-8') as f:
                order_data = json.load(f)
                orders.append(order_data)

        # 최신 주문순으로 정렬
        orders.sort(key=lambda x: x['timestamp'], reverse=True)

        return {
            "orders": orders,
            "total": len(orders),
            "message": f"총 {len(orders)}개의 주문서를 찾았습니다"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"주문서 조회 실패: {str(e)}")

@app.get("/api/admin/orders/{order_id}")
async def get_order_by_id(order_id: str):
    """특정 주문서 상세 조회"""
    try:
        order_file = os.path.join(ADMIN_ORDERS_DIR, f"{order_id}.json")

        if not os.path.exists(order_file):
            raise HTTPException(status_code=404, detail="주문서를 찾을 수 없습니다")

        with open(order_file, 'r', encoding='utf-8') as f:
            order_data = json.load(f)

        return {
            "order": order_data,
            "message": "주문서 조회 성공"
        }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="주문서를 찾을 수 없습니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"주문서 조회 실패: {str(e)}")

@app.get("/api/admin/orders/search/{keyword}")
async def search_orders(keyword: str):
    """주문서 검색 (고객 요청 내용 기준)"""
    try:
        if not os.path.exists(ADMIN_ORDERS_DIR):
            return {"orders": [], "total": 0}

        matching_orders = []
        order_files = glob.glob(os.path.join(ADMIN_ORDERS_DIR, "*.json"))

        for order_file in order_files:
            with open(order_file, 'r', encoding='utf-8') as f:
                order_data = json.load(f)

                # 고객 요청 내용이나 컨셉에서 키워드 검색
                if (keyword.lower() in order_data.get('customer_request', '').lower() or
                    keyword.lower() in order_data.get('recipe', {}).get('concept', '').lower()):
                    matching_orders.append(order_data)

        # 최신 주문순으로 정렬
        matching_orders.sort(key=lambda x: x['timestamp'], reverse=True)

        return {
            "orders": matching_orders,
            "total": len(matching_orders),
            "keyword": keyword,
            "message": f"'{keyword}' 검색 결과 {len(matching_orders)}개 발견"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")

@app.get("/api/admin/stats")
async def get_order_statistics():
    """주문 통계 정보"""
    try:
        if not os.path.exists(ADMIN_ORDERS_DIR):
            return {"stats": {}, "message": "주문 데이터가 없습니다"}

        order_files = glob.glob(os.path.join(ADMIN_ORDERS_DIR, "*.json"))

        stats = {
            "총_주문수": len(order_files),
            "오늘_주문수": 0,
            "향조별_통계": {},
            "최근_주문": None
        }

        today = datetime.now().date()
        latest_order = None

        for order_file in order_files:
            with open(order_file, 'r', encoding='utf-8') as f:
                order_data = json.load(f)

                # 오늘 주문 카운트
                order_date = datetime.fromisoformat(order_data['timestamp']).date()
                if order_date == today:
                    stats["오늘_주문수"] += 1

                # 향조별 통계
                fragrance_family = order_data.get('recipe', {}).get('fragrance_family', '기타')
                if fragrance_family in stats["향조별_통계"]:
                    stats["향조별_통계"][fragrance_family] += 1
                else:
                    stats["향조별_통계"][fragrance_family] = 1

                # 최신 주문 찾기
                if latest_order is None or order_data['timestamp'] > latest_order['timestamp']:
                    latest_order = order_data

        stats["최근_주문"] = latest_order

        return {
            "stats": stats,
            "message": "통계 조회 성공"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("관리자 서버 시작 중...")
    print("주문서 조회: http://localhost:8001/api/admin/orders")
    print("API 문서: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001)