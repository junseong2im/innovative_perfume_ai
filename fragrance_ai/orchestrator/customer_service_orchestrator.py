"""
Customer Service Orchestrator
- 고객 서비스 전용 오케스트레이터
- 향수와 분리된 일반 CS 처리
- 주문, 배송, 반품 등 일반 문의 처리
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from ..llm.customer_service_client import CustomerServiceLLM, get_customer_service_llm
from ..database.models import User, Order, Ticket
from ..core.auth import get_current_user

logger = logging.getLogger(__name__)


class CustomerServiceOrchestrator:
    """고객 서비스 오케스트레이터"""
    
    def __init__(self):
        """초기화"""
        self.llm = get_customer_service_llm()
        self.conversation_cache = {}  # user_id -> conversation history
        self.ticket_cache = {}  # ticket_id -> ticket info
        
    async def initialize(self):
        """비동기 초기화"""
        # LLM 사용 가능 여부 확인
        is_available = await self.llm.check_availability()
        if not is_available:
            logger.warning("Customer Service LLM not available, using rule-based fallback")
            
    async def handle_customer_request(
        self,
        message: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """고객 요청 처리"""
        try:
            # 세션 ID 생성 (필요시)
            if not session_id:
                session_id = f"cs_{user_id or 'anon'}_{datetime.now().timestamp()}"
                
            # 대화 히스토리 가져오기
            conversation_history = self._get_conversation_history(session_id)
            
            # 컨텍스트 구성 (DB에서 사용자/주문 정보 조회)
            context = await self._build_context(user_id, metadata)
            
            # 문의 분류
            category = await self.llm.categorize_query(message)
            
            # 특별한 처리가 필요한 경우 체크
            if category['category'] == 'product' and '향수 제작' in message:
                return self._redirect_to_artisan()
                
            # LLM으로 응답 생성
            response = await self.llm.handle_customer_query(
                query=message,
                context=context,
                conversation_history=conversation_history
            )
            
            # 감정 분석
            sentiment = await self.llm.analyze_sentiment(message)
            
            # 대화 히스토리 업데이트
            self._update_conversation_history(
                session_id,
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            )
            
            # 긴급한 경우 티켓 생성
            ticket_id = None
            if sentiment['priority'] in ['urgent', 'high']:
                ticket_id = await self._create_support_ticket(
                    user_id=user_id,
                    message=message,
                    category=category['category'],
                    priority=sentiment['priority']
                )
                
            return {
                'success': True,
                'response': response,
                'session_id': session_id,
                'category': category,
                'sentiment': sentiment,
                'ticket_id': ticket_id,
                'timestamp': datetime.utcnow().isoformat(),
                'requires_human': category.get('requires_human', False) or sentiment['priority'] == 'urgent'
            }
            
        except Exception as e:
            logger.error(f"Customer request handling failed: {e}")
            return {
                'success': False,
                'response': "I apologize for the technical difficulty. Please try again or contact support@deulsoom.com",
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def _redirect_to_artisan(self) -> Dict[str, Any]:
        """향수 제작 채팅으로 리다이렉트"""
        return {
            'success': True,
            'response': "For fragrance creation and personalized perfume recipes, please visit our Artisan AI Perfumer chat. Would you like me to redirect you there?",
            'redirect': '/artisan',
            'category': {'category': 'product', 'subcategory': 'fragrance_creation'},
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def _build_context(self, user_id: Optional[str], metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """컨텍스트 구성"""
        context = {}
        
        if user_id:
            context['user_id'] = user_id
            # 실제로는 DB에서 조회
            # user = await get_user_by_id(user_id)
            # if user:
            #     context['user_name'] = user.name
            #     context['user_tier'] = user.tier
            
        if metadata:
            if 'order_id' in metadata:
                context['order_id'] = metadata['order_id']
                # order = await get_order_by_id(metadata['order_id'])
                # if order:
                #     context['order_status'] = order.status
                #     context['order_date'] = order.created_at
                    
            if 'product_id' in metadata:
                context['product_id'] = metadata['product_id']
                
        return context
    
    def _get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """대화 히스토리 가져오기"""
        return self.conversation_cache.get(session_id, [])
    
    def _update_conversation_history(
        self,
        session_id: str,
        user_message: Dict[str, str],
        assistant_message: Dict[str, str]
    ):
        """대화 히스토리 업데이트"""
        if session_id not in self.conversation_cache:
            self.conversation_cache[session_id] = []
            
        history = self.conversation_cache[session_id]
        history.append(user_message)
        history.append(assistant_message)
        
        # 최대 20개 메시지만 유지
        if len(history) > 20:
            self.conversation_cache[session_id] = history[-20:]
            
    async def _create_support_ticket(
        self,
        user_id: Optional[str],
        message: str,
        category: str,
        priority: str
    ) -> str:
        """지원 티켓 생성"""
        try:
            import uuid
            ticket_id = f"TICKET-{str(uuid.uuid4())[:8].upper()}"
            
            # 실제로는 DB에 저장
            # ticket = Ticket(
            #     id=ticket_id,
            #     user_id=user_id,
            #     message=message,
            #     category=category,
            #     priority=priority,
            #     status='open'
            # )
            # await save_ticket(ticket)
            
            self.ticket_cache[ticket_id] = {
                'user_id': user_id,
                'message': message,
                'category': category,
                'priority': priority,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'open'
            }
            
            logger.info(f"Support ticket created: {ticket_id} with priority: {priority}")
            return ticket_id
            
        except Exception as e:
            logger.error(f"Ticket creation failed: {e}")
            return None
    
    async def get_ticket_status(self, ticket_id: str) -> Dict[str, Any]:
        """티켓 상태 조회"""
        if ticket_id in self.ticket_cache:
            return self.ticket_cache[ticket_id]
            
        # 실제로는 DB에서 조회
        # ticket = await get_ticket_by_id(ticket_id)
        # if ticket:
        #     return ticket.to_dict()
            
        return {'error': 'Ticket not found'}
    
    async def escalate_to_human(
        self,
        session_id: str,
        reason: str
    ) -> Dict[str, Any]:
        """사람 상담원으로 에스켈레이션"""
        # 실제로는 상담원 큐에 추가하고 알림 송출
        logger.info(f"Escalating session {session_id} to human agent. Reason: {reason}")
        
        return {
            'success': True,
            'message': "I'm connecting you with a human agent who can better assist you. Please wait a moment.",
            'estimated_wait': "2-3 minutes",
            'queue_position': 3  # 예시
        }
    
    async def handle_feedback(
        self,
        session_id: str,
        rating: int,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """고객 피드백 처리"""
        try:
            # 피드백 저장 (실제로는 DB에)
            feedback_data = {
                'session_id': session_id,
                'rating': rating,
                'comment': comment,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # 대화 히스토리와 함께 저장
            if session_id in self.conversation_cache:
                feedback_data['conversation'] = self.conversation_cache[session_id]
                
            logger.info(f"Feedback received for session {session_id}: {rating}/5")
            
            response_message = "Thank you for your feedback!"
            if rating <= 2:
                response_message += " We're sorry your experience wasn't satisfactory. We'll use your feedback to improve."
            elif rating >= 4:
                response_message += " We're glad we could help you today!"
                
            return {
                'success': True,
                'message': response_message
            }
            
        except Exception as e:
            logger.error(f"Feedback handling failed: {e}")
            return {
                'success': False,
                'message': "Thank you for your feedback. We've recorded it for review."
            }


class CustomerServiceTools:
    """고객 서비스 도구 모음"""
    
    @staticmethod
    async def check_order_status(order_id: str) -> Dict[str, Any]:
        """주문 상태 확인 - 실제 데이터베이스 조회"""
        import sqlite3
        import os
        from pathlib import Path

        # 실제 데이터베이스 연결
        db_path = Path(__file__).parent.parent.parent / "data" / "orders.db"

        # DB가 없으면 생성
        if not db_path.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 주문 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    status TEXT,
                    tracking_number TEXT,
                    delivered_date TEXT,
                    expected_delivery TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 초기 데이터 삽입
            cursor.executemany("""
                INSERT OR IGNORE INTO orders (order_id, status, tracking_number, delivered_date, expected_delivery)
                VALUES (?, ?, ?, ?, ?)
            """, [
                ("ORD-001", "Delivered", "1Z999AA10123456784", "2024-01-15", None),
                ("ORD-002", "In Transit", "1Z999AA10123456785", None, "2024-01-20"),
                ("ORD-003", "Processing", None, None, "2024-01-25"),
                ("ORD-004", "Shipped", "1Z999AA10123456786", None, "2024-01-22")
            ])
            conn.commit()
        else:
            conn = sqlite3.connect(db_path)

        # 실제 조회
        cursor = conn.cursor()
        cursor.execute("""
            SELECT status, tracking_number, delivered_date, expected_delivery
            FROM orders WHERE order_id = ?
        """, (order_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            result = {"status": row[0]}
            if row[1]:
                result["tracking_number"] = row[1]
            if row[2]:
                result["delivered_date"] = row[2]
            if row[3]:
                result["expected_delivery"] = row[3]
            return result

        return {"error": "Order not found"}
    
    @staticmethod
    async def initiate_return(order_id: str, reason: str) -> Dict[str, Any]:
        """반품 시작"""
        # 실제로는 반품 프로세스 시작
        return {
            "return_id": f"RET-{order_id}",
            "status": "Return label created",
            "instructions": "Please print the return label and attach it to your package."
        }
    
    @staticmethod
    async def calculate_shipping(destination: str, method: str) -> Dict[str, Any]:
        """배송비 계산"""
        rates = {
            "standard": 5.99,
            "express": 15.99,
            "overnight": 29.99
        }
        
        return {
            "destination": destination,
            "method": method,
            "cost": rates.get(method, 5.99),
            "estimated_days": {"standard": "5-7", "express": "2-3", "overnight": "1"}.get(method, "5-7")
        }


# 전역 오케스트레이터 인스턴스
_orchestrator_instance = None

def get_customer_service_orchestrator() -> CustomerServiceOrchestrator:
    """고객 서비스 오케스트레이터 싱글톤"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = CustomerServiceOrchestrator()
    return _orchestrator_instance
