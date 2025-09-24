"""
Customer Service LLM Client
- 고객 서비스 전용 LLM
- 일반적인 고객 문의 처리
- 주문, 배송, 반품 등 처리
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
import aiohttp
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class CustomerServiceLLM:
    """고객 서비스 전용 LLM 클라이언트"""
    
    def __init__(self, config: Optional[dict] = None):
        """초기화"""
        self.config = config or self._load_config()
        self.provider = self.config.get('provider', 'ollama')
        self.model_name = self.config.get('model_name_or_path', 'mistral:7b-instruct-q4_K_M')
        self.api_base = self.config.get('api_base', 'http://localhost:11434')
        self.session = None
        
        # 고객 서비스 전용 시스템 프롬프트
        self.system_prompt = """You are a professional customer service representative for Deulsoom, a luxury fragrance brand. 
You help customers with:
- Order inquiries and tracking
- Product information and recommendations  
- Shipping and delivery questions
- Returns and exchanges
- Account and payment issues
- General support

Be helpful, professional, and empathetic. If asked about fragrance creation or recipes, 
politely redirect them to our Artisan AI Perfumer chat for specialized assistance.

Always maintain a luxury brand voice - sophisticated yet approachable."""
        
    def _load_config(self) -> dict:
        """설정 로드"""
        try:
            import json
            with open('configs/local.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('llm_customer_service', {
                'provider': 'ollama',
                'model_name_or_path': 'mistral:7b-instruct-q4_K_M',
                'api_base': 'http://localhost:11434'
            })
        except:
            return {
                'provider': 'ollama',
                'model_name_or_path': 'mistral:7b-instruct-q4_K_M',
                'api_base': 'http://localhost:11434'
            }
    
    async def _ensure_session(self):
        """세션 확인"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def close(self):
        """세션 종료"""
        if self.session:
            await self.session.close()
            self.session = None
            
    async def check_availability(self) -> bool:
        """모델 사용 가능 여부 확인"""
        try:
            await self._ensure_session()
            
            if self.provider == 'ollama':
                async with self.session.get(f"{self.api_base}/api/tags") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get('models', [])
                        # 모델이 설치되어 있는지 확인
                        for model in models:
                            if self.model_name in model.get('name', ''):
                                logger.info(f"Customer service model {self.model_name} is available")
                                return True
                        logger.warning(f"Model {self.model_name} not found. Available models: {[m['name'] for m in models]}")
                        return False
            return False
                        
        except Exception as e:
            logger.error(f"Availability check failed: {e}")
            return False
    
    async def handle_customer_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """고객 문의 처리"""
        try:
            await self._ensure_session()
            
            # 대화 히스토리 구성
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # 컨텍스트 추가 (주문 정보, 사용자 정보 등)
            if context:
                context_str = self._format_context(context)
                messages.append({
                    "role": "system",
                    "content": f"Current context:\n{context_str}"
                })
            
            # 이전 대화 추가
            if conversation_history:
                messages.extend(conversation_history[-10:])  # 최근 10개만
                
            # 현재 질문 추가
            messages.append({"role": "user", "content": query})
            
            # LLM 호출
            if self.provider == 'ollama':
                return await self._ollama_chat(messages)
            else:
                # Fallback to rule-based
                return self._rule_based_response(query, context)
                
        except Exception as e:
            logger.error(f"Customer query handling failed: {e}")
            return self._get_fallback_response()
    
    async def _ollama_chat(self, messages: List[Dict[str, str]]) -> str:
        """Ollama로 채팅"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }
            
            async with self.session.post(
                f"{self.api_base}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('message', {}).get('content', self._get_fallback_response())
                else:
                    logger.error(f"Ollama chat failed with status {resp.status}")
                    return self._get_fallback_response()
                    
        except asyncio.TimeoutError:
            logger.error("Ollama chat timeout")
            return "I apologize for the delay. Our system is currently processing many requests. Please try again in a moment."
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return self._get_fallback_response()
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트 포맷팅"""
        parts = []
        
        if 'user_id' in context:
            parts.append(f"Customer ID: {context['user_id']}")
        if 'order_id' in context:
            parts.append(f"Order ID: {context['order_id']}")
        if 'order_status' in context:
            parts.append(f"Order Status: {context['order_status']}")
        if 'products' in context:
            parts.append(f"Products: {', '.join(context['products'])}")
        if 'issue_type' in context:
            parts.append(f"Issue Type: {context['issue_type']}")
            
        return "\n".join(parts)
    
    def _rule_based_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """규칙 기반 응답 (폴백)"""
        query_lower = query.lower()
        
        # 주문 관련
        if any(word in query_lower for word in ['order', '주문', '구매']):
            if 'track' in query_lower or '추적' in query_lower or '배송' in query_lower:
                if context and 'order_id' in context:
                    return f"Your order #{context['order_id']} is currently {context.get('order_status', 'being processed')}. You can track your shipment using the tracking number sent to your email."
                return "To track your order, please provide your order number or check the confirmation email we sent you."
            elif 'cancel' in query_lower or '취소' in query_lower:
                return "To cancel an order, please contact us within 24 hours of placing it. Orders that have been shipped cannot be cancelled but can be returned."
            else:
                return "I can help you with your order. Could you please provide your order number or describe what you need assistance with?"
        
        # 배송 관련
        elif any(word in query_lower for word in ['shipping', '배송', 'delivery']):
            return "We offer standard shipping (5-7 business days) and express shipping (2-3 business days). Free shipping on orders over $100. International shipping is available to select countries."
        
        # 반품/교환
        elif any(word in query_lower for word in ['return', '반품', 'exchange', '교환']):
            return "We accept returns within 30 days of purchase for unopened products. Please initiate a return request through your account or contact our support team with your order number."
        
        # 결제 관련
        elif any(word in query_lower for word in ['payment', '결제', 'card', '카드']):
            return "We accept all major credit cards, PayPal, and Apple Pay. Your payment information is securely processed and never stored on our servers."
        
        # 제품 정보
        elif any(word in query_lower for word in ['product', '제품', 'fragrance', '향수']):
            if 'recommend' in query_lower or '추천' in query_lower:
                return "For personalized fragrance recommendations, I'd be happy to connect you with our Artisan AI Perfumer. Would you like me to transfer you to our specialized fragrance consultation service?"
            return "Our luxury fragrances are crafted with the finest ingredients. Each scent tells a unique story. What type of fragrance are you interested in?"
        
        # 계정 관련
        elif any(word in query_lower for word in ['account', '계정', 'password', '비밀번호']):
            return "For account security, please visit your account settings or use the 'Forgot Password' link on the login page. If you need further assistance, our support team can help verify your identity."
        
        # 일반 인사
        elif any(word in query_lower for word in ['hello', '안녕', 'hi', 'help', '도움']):
            return "Hello! Welcome to Deulsoom customer service. I'm here to help with orders, shipping, returns, or any questions about our luxury fragrances. How may I assist you today?"
        
        # 기본 응답
        else:
            return "I'm here to help with any questions about orders, shipping, products, or account issues. Could you please tell me more about what you need assistance with?"
    
    def _get_fallback_response(self) -> str:
        """최종 폴백 응답"""
        return "I apologize for the inconvenience. Our system is currently experiencing high demand. Please try again shortly, or contact our support team directly at support@deulsoom.com for immediate assistance."
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """감정 분석"""
        try:
            # 간단한 규칙 기반 감정 분석
            negative_words = ['angry', '화가', 'disappointed', '실망', 'terrible', '최악', 'bad', '나쁜']
            positive_words = ['happy', '행복', 'satisfied', '만족', 'great', '좋은', 'excellent', '훌륭']
            
            text_lower = text.lower()
            
            negative_count = sum(1 for word in negative_words if word in text_lower)
            positive_count = sum(1 for word in positive_words if word in text_lower)
            
            if negative_count > positive_count:
                sentiment = 'negative'
                priority = 'high'
            elif positive_count > negative_count:
                sentiment = 'positive'
                priority = 'normal'
            else:
                sentiment = 'neutral'
                priority = 'normal'
                
            # 긴급 키워드 체크
            urgent_words = ['urgent', '긴급', 'immediately', '즉시', 'asap']
            if any(word in text_lower for word in urgent_words):
                priority = 'urgent'
                
            return {
                'sentiment': sentiment,
                'priority': priority,
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'priority': 'normal', 'confidence': 0.5}
    
    async def categorize_query(self, query: str) -> Dict[str, Any]:
        """문의 분류"""
        query_lower = query.lower()
        
        categories = {
            'order': ['order', '주문', 'purchase', '구매', 'buy'],
            'shipping': ['ship', '배송', 'delivery', 'track', '추적'],
            'return': ['return', '반품', 'exchange', '교환', 'refund', '환불'],
            'product': ['product', '제품', 'fragrance', '향수', 'scent', '향'],
            'account': ['account', '계정', 'login', '로그인', 'password', '비밀번호'],
            'payment': ['payment', '결제', 'card', '카드', 'charge', '청구'],
            'general': []
        }
        
        detected_category = 'general'
        confidence = 0.5
        
        for category, keywords in categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_category = category
                confidence = 0.9
                break
                
        # 하위 카테고리 감지
        subcategory = None
        if detected_category == 'order':
            if 'track' in query_lower or '추적' in query_lower:
                subcategory = 'tracking'
            elif 'cancel' in query_lower or '취소' in query_lower:
                subcategory = 'cancellation'
            elif 'modify' in query_lower or '변경' in query_lower:
                subcategory = 'modification'
                
        return {
            'category': detected_category,
            'subcategory': subcategory,
            'confidence': confidence,
            'requires_human': confidence < 0.7
        }


# 전역 인스턴스
_customer_service_instance = None

def get_customer_service_llm() -> CustomerServiceLLM:
    """고객 서비스 LLM 싱글톤"""
    global _customer_service_instance
    if _customer_service_instance is None:
        _customer_service_instance = CustomerServiceLLM()
    return _customer_service_instance


async def handle_customer_support(
    query: str,
    user_id: Optional[str] = None,
    order_id: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """고객 지원 처리 메인 함수"""
    llm = get_customer_service_llm()
    
    # 컨텍스트 구성
    context = {}
    if user_id:
        context['user_id'] = user_id
    if order_id:
        context['order_id'] = order_id
        # 실제로는 DB에서 주문 정보 조회
        context['order_status'] = 'In Transit'
        
    # 감정 분석
    sentiment = await llm.analyze_sentiment(query)
    
    # 카테고리 분류
    category = await llm.categorize_query(query)
    
    # LLM 응답 생성
    response = await llm.handle_customer_query(
        query=query,
        context=context,
        conversation_history=conversation_history
    )
    
    return {
        'response': response,
        'sentiment': sentiment,
        'category': category,
        'timestamp': datetime.utcnow().isoformat(),
        'requires_escalation': sentiment['priority'] == 'urgent' or category.get('requires_human', False)
    }
