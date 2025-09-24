"""
Customer Service API Routes
- 고객 서비스 전용 API
- 주문, 배송, 반품 등 일반 CS
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import logging
from datetime import datetime

from ...orchestrator.customer_service_orchestrator import (
    get_customer_service_orchestrator,
    CustomerServiceTools
)
from ...core.auth import get_current_user_optional

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/customer-service", tags=["Customer Service"])


# Request/Response Models
class CustomerQuery(BaseModel):
    """고객 문의 모델"""
    message: str = Field(..., description="고객 문의 내용")
    session_id: Optional[str] = Field(None, description="세션 ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")


class CustomerResponse(BaseModel):
    """고객 응답 모델"""
    success: bool
    response: str
    session_id: str
    category: Optional[Dict[str, Any]] = None
    sentiment: Optional[Dict[str, Any]] = None
    ticket_id: Optional[str] = None
    requires_human: bool = False
    redirect: Optional[str] = None
    timestamp: str


class OrderStatusQuery(BaseModel):
    """주문 상태 조회"""
    order_id: str = Field(..., description="주문 번호")


class ReturnRequest(BaseModel):
    """반품 요청"""
    order_id: str = Field(..., description="주문 번호")
    reason: str = Field(..., description="반품 사유")
    items: Optional[List[str]] = Field(None, description="반품 품목")


class FeedbackRequest(BaseModel):
    """피드백 요청"""
    session_id: str = Field(..., description="세션 ID")
    rating: int = Field(..., ge=1, le=5, description="평점 (1-5)")
    comment: Optional[str] = Field(None, description="코멘트")


@router.post("/chat", response_model=CustomerResponse)
async def customer_chat(
    query: CustomerQuery,
    background_tasks: BackgroundTasks,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    고객 서비스 채팅
    
    - 일반 고객 문의 처리
    - 주문, 배송, 반품 등
    - 향수 관련은 /artisan으로 리다이렉트
    """
    try:
        orchestrator = get_customer_service_orchestrator()
        
        # 사용자 ID 추출
        user_id = None
        if current_user:
            user_id = current_user.get('user_id')
            
        # 고객 요청 처리
        result = await orchestrator.handle_customer_request(
            message=query.message,
            user_id=user_id,
            session_id=query.session_id,
            metadata=query.metadata
        )
        
        # 백그라운드 작업: 분석 로깅
        if result.get('requires_human'):
            background_tasks.add_task(
                log_escalation,
                session_id=result.get('session_id'),
                reason="High priority or complex query"
            )
            
        return CustomerResponse(**result)
        
    except Exception as e:
        logger.error(f"Customer chat error: {e}")
        raise HTTPException(status_code=500, detail="Customer service temporarily unavailable")


@router.post("/order/status")
async def check_order_status(
    query: OrderStatusQuery,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    주문 상태 확인
    """
    try:
        result = await CustomerServiceTools.check_order_status(query.order_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return {
            "success": True,
            "order_id": query.order_id,
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order status check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check order status")


@router.post("/return/initiate")
async def initiate_return(
    request: ReturnRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    반품 시작
    """
    try:
        if not current_user:
            raise HTTPException(status_code=401, detail="Authentication required for returns")
            
        result = await CustomerServiceTools.initiate_return(
            order_id=request.order_id,
            reason=request.reason
        )
        
        return {
            "success": True,
            "return_data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Return initiation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate return")


@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    고객 피드백 제출
    """
    try:
        orchestrator = get_customer_service_orchestrator()
        
        result = await orchestrator.handle_feedback(
            session_id=feedback.session_id,
            rating=feedback.rating,
            comment=feedback.comment
        )
        
        # 백그라운드: 피드백 분석
        background_tasks.add_task(
            analyze_feedback,
            session_id=feedback.session_id,
            rating=feedback.rating
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.post("/escalate")
async def escalate_to_human(
    session_id: str,
    reason: Optional[str] = "Customer requested human agent"
):
    """
    사람 상담원으로 에스켈레이션
    """
    try:
        orchestrator = get_customer_service_orchestrator()
        
        result = await orchestrator.escalate_to_human(
            session_id=session_id,
            reason=reason
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Escalation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to escalate to human agent")


@router.get("/ticket/{ticket_id}")
async def get_ticket_status(
    ticket_id: str,
    current_user: dict = Depends(get_current_user_optional)
):
    """
    티켓 상태 확인
    """
    try:
        orchestrator = get_customer_service_orchestrator()
        result = await orchestrator.get_ticket_status(ticket_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return {
            "success": True,
            "ticket": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ticket status check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check ticket status")


@router.post("/shipping/calculate")
async def calculate_shipping(
    destination: str,
    method: str = "standard",
    weight: Optional[float] = None
):
    """
    배송비 계산
    """
    try:
        result = await CustomerServiceTools.calculate_shipping(
            destination=destination,
            method=method
        )
        
        return {
            "success": True,
            "shipping": result
        }
        
    except Exception as e:
        logger.error(f"Shipping calculation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate shipping")


@router.get("/faq")
async def get_frequently_asked_questions():
    """
    자주 묻는 질문
    """
    return {
        "success": True,
        "categories": [
            {
                "category": "Orders",
                "questions": [
                    {
                        "q": "How can I track my order?",
                        "a": "You can track your order using the tracking number sent to your email or through your account dashboard."
                    },
                    {
                        "q": "Can I cancel my order?",
                        "a": "Orders can be cancelled within 24 hours of placement if they haven't been shipped yet."
                    }
                ]
            },
            {
                "category": "Shipping",
                "questions": [
                    {
                        "q": "What are the shipping options?",
                        "a": "We offer standard (5-7 days), express (2-3 days), and overnight shipping."
                    },
                    {
                        "q": "Do you ship internationally?",
                        "a": "Yes, we ship to select countries. International shipping takes 10-15 business days."
                    }
                ]
            },
            {
                "category": "Returns",
                "questions": [
                    {
                        "q": "What is your return policy?",
                        "a": "We accept returns within 30 days for unopened products in original packaging."
                    },
                    {
                        "q": "How long does a refund take?",
                        "a": "Refunds are processed within 5-7 business days after we receive the returned item."
                    }
                ]
            }
        ]
    }


# Background tasks
async def log_escalation(session_id: str, reason: str):
    """에스켈레이션 로깅"""
    logger.info(f"Escalation logged - Session: {session_id}, Reason: {reason}")
    # 실제로는 DB에 저장하고 상담원에게 알림


async def analyze_feedback(session_id: str, rating: int):
    """피드백 분석"""
    logger.info(f"Feedback analysis - Session: {session_id}, Rating: {rating}")
    # 실제로는 피드백 통계 업데이트, ML 모델 학습 등


@router.get("/health")
async def health_check():
    """
    고객 서비스 상태 체크
    """
    orchestrator = get_customer_service_orchestrator()
    llm_available = await orchestrator.llm.check_availability()
    
    return {
        "service": "customer-service",
        "status": "healthy",
        "llm_available": llm_available,
        "timestamp": datetime.utcnow().isoformat()
    }
