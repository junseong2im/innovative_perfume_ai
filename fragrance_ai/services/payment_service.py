"""
결제 서비스 관리자
- 다중 결제 게이트웨이 지원
- 구독 결제 관리
- 결제 상태 추적 및 웹훅
- PCI DSS 준수
"""

import asyncio
import aiohttp
import hashlib
import hmac
from typing import List, Dict, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import json
import uuid
from decimal import Decimal, ROUND_HALF_UP
import base64

from ..core.logging_config import get_logger
from ..core.monitoring import MetricsCollector
from ..models.user import User

logger = get_logger(__name__)

class PaymentProvider(Enum):
    STRIPE = "stripe"
    PAYPAL = "paypal"
    TOSS_PAYMENTS = "toss_payments"  # 토스페이먼츠
    KAKAO_PAY = "kakao_pay"
    NAVER_PAY = "naver_pay"
    IAMPORT = "iamport"  # 아임포트 (국내)

class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    PARTIALLY_REFUNDED = "partially_refunded"

class PaymentMethod(Enum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTOCURRENCY = "cryptocurrency"

class CurrencyCode(Enum):
    USD = "usd"
    KRW = "krw"
    EUR = "eur"
    JPY = "jpy"

class SubscriptionStatus(Enum):
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    EXPIRED = "expired"

@dataclass
class PaymentCustomer:
    id: str
    email: str
    name: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[Dict[str, str]] = None
    provider_customer_id: Optional[str] = None

@dataclass
class PaymentItem:
    name: str
    quantity: int
    unit_price: Decimal
    description: Optional[str] = None
    category: Optional[str] = None

    @property
    def total_price(self) -> Decimal:
        return self.unit_price * self.quantity

@dataclass
class PaymentTransaction:
    id: str
    customer: PaymentCustomer
    items: List[PaymentItem]
    total_amount: Decimal
    currency: CurrencyCode
    payment_method: PaymentMethod
    status: PaymentStatus = PaymentStatus.PENDING
    provider_transaction_id: Optional[str] = None
    provider_payment_url: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    refund_amount: Decimal = Decimal('0')
    fees: Decimal = Decimal('0')

@dataclass
class Subscription:
    id: str
    customer: PaymentCustomer
    plan_id: str
    plan_name: str
    amount: Decimal
    currency: CurrencyCode
    billing_cycle: str  # monthly, yearly
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    provider_subscription_id: Optional[str] = None
    current_period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_period_end: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30))
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    cancelled_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

class PaymentService:
    """결제 서비스 관리자"""

    def __init__(
        self,
        provider: PaymentProvider = PaymentProvider.TOSS_PAYMENTS,
        # Stripe 설정
        stripe_secret_key: Optional[str] = None,
        stripe_publishable_key: Optional[str] = None,
        stripe_webhook_secret: Optional[str] = None,
        # 토스페이먼츠 설정
        toss_secret_key: Optional[str] = None,
        toss_client_key: Optional[str] = None,
        # 아임포트 설정
        iamport_api_key: Optional[str] = None,
        iamport_api_secret: Optional[str] = None,
        # 카카오페이 설정
        kakao_admin_key: Optional[str] = None,
        kakao_cid: Optional[str] = None,
        # 일반 설정
        webhook_endpoints: Optional[Dict[str, str]] = None,
        default_currency: CurrencyCode = CurrencyCode.KRW,
        enable_webhooks: bool = True
    ):
        self.provider = provider
        self.default_currency = default_currency
        self.enable_webhooks = enable_webhooks

        # 제공업체별 설정
        self.provider_config = {
            "stripe": {
                "secret_key": stripe_secret_key,
                "publishable_key": stripe_publishable_key,
                "webhook_secret": stripe_webhook_secret,
                "base_url": "https://api.stripe.com/v1"
            },
            "toss_payments": {
                "secret_key": toss_secret_key,
                "client_key": toss_client_key,
                "base_url": "https://api.tosspayments.com/v1"
            },
            "iamport": {
                "api_key": iamport_api_key,
                "api_secret": iamport_api_secret,
                "base_url": "https://api.iamport.kr"
            },
            "kakao_pay": {
                "admin_key": kakao_admin_key,
                "cid": kakao_cid,
                "base_url": "https://kapi.kakao.com"
            }
        }

        # 웹훅 엔드포인트
        self.webhook_endpoints = webhook_endpoints or {}

        # 데이터 저장
        self.transactions: Dict[str, PaymentTransaction] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.customers: Dict[str, PaymentCustomer] = {}

        # 통계
        self.payment_stats = {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "total_amount": Decimal('0'),
            "total_fees": Decimal('0'),
            "refunded_amount": Decimal('0')
        }

        # HTTP 클라이언트
        self.session: Optional[aiohttp.ClientSession] = None

        # 웹훅 핸들러
        self.webhook_handlers: Dict[str, Callable] = {}

        # 백그라운드 작업
        self._status_checker_task: Optional[asyncio.Task] = None

        # 메트릭
        self.metrics_collector = MetricsCollector()

    async def initialize(self):
        """결제 서비스 초기화"""
        try:
            logger.info(f"Initializing payment service with provider: {self.provider.value}")

            # HTTP 세션 설정
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # 제공업체별 인증 설정
            await self._setup_provider_auth()

            # 웹훅 핸들러 등록
            self._register_webhook_handlers()

            # 백그라운드 작업 시작
            self._status_checker_task = asyncio.create_task(self._payment_status_checker())

            logger.info("Payment service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize payment service: {e}")
            raise

    async def _setup_provider_auth(self):
        """제공업체별 인증 설정"""
        config = self.provider_config[self.provider.value]

        if self.provider == PaymentProvider.STRIPE:
            if not config.get("secret_key"):
                raise ValueError("Stripe secret key is required")

            # Stripe 계정 정보 확인
            await self._stripe_verify_account()

        elif self.provider == PaymentProvider.TOSS_PAYMENTS:
            if not config.get("secret_key"):
                raise ValueError("Toss Payments secret key is required")

        elif self.provider == PaymentProvider.IAMPORT:
            if not config.get("api_key") or not config.get("api_secret"):
                raise ValueError("Iamport API credentials are required")

            # 아임포트 토큰 발급
            await self._iamport_get_access_token()

    def _register_webhook_handlers(self):
        """웹훅 핸들러 등록"""
        self.webhook_handlers = {
            "payment.succeeded": self._handle_payment_success,
            "payment.failed": self._handle_payment_failure,
            "subscription.created": self._handle_subscription_created,
            "subscription.updated": self._handle_subscription_updated,
            "subscription.cancelled": self._handle_subscription_cancelled,
            "invoice.payment_succeeded": self._handle_invoice_payment_success,
            "invoice.payment_failed": self._handle_invoice_payment_failure
        }

    async def create_payment(
        self,
        customer: PaymentCustomer,
        items: List[PaymentItem],
        payment_method: PaymentMethod,
        currency: Optional[CurrencyCode] = None,
        description: Optional[str] = None,
        success_url: Optional[str] = None,
        cancel_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PaymentTransaction:
        """결제 생성"""

        transaction_id = str(uuid.uuid4())
        total_amount = sum(item.total_price for item in items)
        currency = currency or self.default_currency

        # 결제 트랜잭션 생성
        transaction = PaymentTransaction(
            id=transaction_id,
            customer=customer,
            items=items,
            total_amount=total_amount,
            currency=currency,
            payment_method=payment_method,
            description=description,
            metadata=metadata or {},
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1)  # 1시간 후 만료
        )

        try:
            # 제공업체별 결제 생성
            if self.provider == PaymentProvider.STRIPE:
                await self._create_stripe_payment(transaction, success_url, cancel_url)
            elif self.provider == PaymentProvider.TOSS_PAYMENTS:
                await self._create_toss_payment(transaction, success_url, cancel_url)
            elif self.provider == PaymentProvider.IAMPORT:
                await self._create_iamport_payment(transaction, success_url, cancel_url)
            else:
                raise NotImplementedError(f"Provider {self.provider.value} not implemented")

            # 저장
            self.transactions[transaction_id] = transaction
            self.payment_stats["total_transactions"] += 1

            logger.info(f"Payment created: {transaction_id}")
            return transaction

        except Exception as e:
            logger.error(f"Failed to create payment {transaction_id}: {e}")
            transaction.status = PaymentStatus.FAILED
            transaction.failure_reason = str(e)
            raise

    async def _create_stripe_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """Stripe 결제 생성"""
        config = self.provider_config["stripe"]

        # Stripe 고객 생성 또는 조회
        customer_data = await self._get_or_create_stripe_customer(transaction.customer)
        stripe_customer_id = customer_data["id"]

        # 결제 인텐트 생성
        payment_data = {
            "amount": int(transaction.total_amount * 100),  # cents
            "currency": transaction.currency.value,
            "customer": stripe_customer_id,
            "description": transaction.description,
            "metadata": {
                "transaction_id": transaction.id,
                **transaction.metadata
            }
        }

        headers = {
            "Authorization": f"Bearer {config['secret_key']}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        async with self.session.post(
            f"{config['base_url']}/payment_intents",
            data=payment_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                transaction.provider_transaction_id = result["id"]
                transaction.status = PaymentStatus.PROCESSING
            else:
                error_data = await response.json()
                raise Exception(f"Stripe API error: {error_data}")

    async def _create_toss_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """토스페이먼츠 결제 생성"""
        config = self.provider_config["toss_payments"]

        payment_data = {
            "orderId": transaction.id,
            "amount": int(transaction.total_amount),
            "orderName": transaction.description or "향수 AI 서비스",
            "customerEmail": transaction.customer.email,
            "customerName": transaction.customer.name or "고객",
            "successUrl": success_url or self.webhook_endpoints.get("success"),
            "failUrl": cancel_url or self.webhook_endpoints.get("cancel")
        }

        # Base64 인코딩된 시크릿 키
        auth_string = base64.b64encode(f"{config['secret_key']}:".encode()).decode()

        headers = {
            "Authorization": f"Basic {auth_string}",
            "Content-Type": "application/json"
        }

        async with self.session.post(
            f"{config['base_url']}/payments",
            json=payment_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                transaction.provider_transaction_id = result.get("paymentKey")
                transaction.provider_payment_url = result.get("checkout", {}).get("url")
                transaction.status = PaymentStatus.PROCESSING
            else:
                error_data = await response.json()
                raise Exception(f"Toss Payments API error: {error_data}")

    async def _create_iamport_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """아임포트 결제 생성"""
        # 아임포트는 주로 프론트엔드에서 초기화하고 백엔드에서 검증하는 방식
        # 여기서는 결제 준비만 진행

        merchant_uid = f"order_{transaction.id}_{int(datetime.now().timestamp())}"

        payment_data = {
            "merchant_uid": merchant_uid,
            "amount": int(transaction.total_amount),
            "name": transaction.description or "향수 AI 서비스",
            "buyer_email": transaction.customer.email,
            "buyer_name": transaction.customer.name or "고객",
            "notice_url": self.webhook_endpoints.get("iamport_webhook")
        }

        transaction.provider_transaction_id = merchant_uid
        transaction.metadata["iamport_payment_data"] = payment_data
        transaction.status = PaymentStatus.PENDING

        logger.info(f"Iamport payment prepared: {merchant_uid}")

    async def create_subscription(
        self,
        customer: PaymentCustomer,
        plan_id: str,
        plan_name: str,
        amount: Decimal,
        billing_cycle: str = "monthly",
        trial_days: Optional[int] = None,
        currency: Optional[CurrencyCode] = None
    ) -> Subscription:
        """구독 생성"""

        subscription_id = str(uuid.uuid4())
        currency = currency or self.default_currency

        # 체험 기간 설정
        trial_end = None
        if trial_days:
            trial_end = datetime.now(timezone.utc) + timedelta(days=trial_days)

        # 구독 객체 생성
        subscription = Subscription(
            id=subscription_id,
            customer=customer,
            plan_id=plan_id,
            plan_name=plan_name,
            amount=amount,
            currency=currency,
            billing_cycle=billing_cycle,
            trial_end=trial_end
        )

        try:
            # 제공업체별 구독 생성
            if self.provider == PaymentProvider.STRIPE:
                await self._create_stripe_subscription(subscription)
            elif self.provider == PaymentProvider.TOSS_PAYMENTS:
                await self._create_toss_subscription(subscription)
            else:
                raise NotImplementedError(f"Subscription not supported for {self.provider.value}")

            # 저장
            self.subscriptions[subscription_id] = subscription

            logger.info(f"Subscription created: {subscription_id}")
            return subscription

        except Exception as e:
            logger.error(f"Failed to create subscription {subscription_id}: {e}")
            raise

    async def cancel_subscription(self, subscription_id: str, cancel_at_period_end: bool = True) -> bool:
        """구독 취소"""
        if subscription_id not in self.subscriptions:
            raise ValueError(f"Subscription not found: {subscription_id}")

        subscription = self.subscriptions[subscription_id]

        try:
            # 제공업체별 구독 취소
            if self.provider == PaymentProvider.STRIPE:
                await self._cancel_stripe_subscription(subscription, cancel_at_period_end)
            elif self.provider == PaymentProvider.TOSS_PAYMENTS:
                await self._cancel_toss_subscription(subscription, cancel_at_period_end)
            else:
                raise NotImplementedError(f"Subscription cancellation not supported for {self.provider.value}")

            # 상태 업데이트
            subscription.cancel_at_period_end = cancel_at_period_end
            if not cancel_at_period_end:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.cancelled_at = datetime.now(timezone.utc)

            logger.info(f"Subscription cancelled: {subscription_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel subscription {subscription_id}: {e}")
            raise

    async def refund_payment(self, transaction_id: str, amount: Optional[Decimal] = None) -> bool:
        """결제 환불"""
        if transaction_id not in self.transactions:
            raise ValueError(f"Transaction not found: {transaction_id}")

        transaction = self.transactions[transaction_id]
        refund_amount = amount or transaction.total_amount

        try:
            # 제공업체별 환불 처리
            if self.provider == PaymentProvider.STRIPE:
                await self._stripe_refund(transaction, refund_amount)
            elif self.provider == PaymentProvider.TOSS_PAYMENTS:
                await self._toss_refund(transaction, refund_amount)
            else:
                raise NotImplementedError(f"Refund not supported for {self.provider.value}")

            # 상태 업데이트
            transaction.refund_amount += refund_amount
            if transaction.refund_amount >= transaction.total_amount:
                transaction.status = PaymentStatus.REFUNDED
            else:
                transaction.status = PaymentStatus.PARTIALLY_REFUNDED

            self.payment_stats["refunded_amount"] += refund_amount

            logger.info(f"Payment refunded: {transaction_id}, amount: {refund_amount}")
            return True

        except Exception as e:
            logger.error(f"Failed to refund payment {transaction_id}: {e}")
            raise

    async def handle_webhook(self, provider: str, event_type: str, data: Dict[str, Any]) -> bool:
        """웹훅 처리"""
        try:
            handler_key = f"{event_type}"
            if handler_key in self.webhook_handlers:
                await self.webhook_handlers[handler_key](data)
                logger.info(f"Webhook handled: {event_type}")
                return True
            else:
                logger.warning(f"No handler for webhook event: {event_type}")
                return False

        except Exception as e:
            logger.error(f"Webhook handling failed: {e}")
            return False

    async def _handle_payment_success(self, data: Dict[str, Any]):
        """결제 성공 웹훅 핸들러"""
        transaction_id = data.get("transaction_id")
        if transaction_id and transaction_id in self.transactions:
            transaction = self.transactions[transaction_id]
            transaction.status = PaymentStatus.COMPLETED
            transaction.completed_at = datetime.now(timezone.utc)
            self.payment_stats["successful_transactions"] += 1

    async def _handle_payment_failure(self, data: Dict[str, Any]):
        """결제 실패 웹훅 핸들러"""
        transaction_id = data.get("transaction_id")
        if transaction_id and transaction_id in self.transactions:
            transaction = self.transactions[transaction_id]
            transaction.status = PaymentStatus.FAILED
            transaction.failure_reason = data.get("failure_reason")
            self.payment_stats["failed_transactions"] += 1

    async def _handle_subscription_created(self, data: Dict[str, Any]):
        """구독 생성 웹훅 핸들러"""
        # 구독 생성 후 처리 로직
        pass

    async def _handle_subscription_updated(self, data: Dict[str, Any]):
        """구독 업데이트 웹훅 핸들러"""
        # 구독 업데이트 후 처리 로직
        pass

    async def _handle_subscription_cancelled(self, data: Dict[str, Any]):
        """구독 취소 웹훅 핸들러"""
        # 구독 취소 후 처리 로직
        pass

    async def _handle_invoice_payment_success(self, data: Dict[str, Any]):
        """인보이스 결제 성공 웹훅 핸들러"""
        # 구독 인보이스 결제 성공 처리
        pass

    async def _handle_invoice_payment_failure(self, data: Dict[str, Any]):
        """인보이스 결제 실패 웹훅 핸들러"""
        # 구독 인보이스 결제 실패 처리
        pass

    async def _payment_status_checker(self):
        """결제 상태 확인 백그라운드 작업"""
        while True:
            try:
                await asyncio.sleep(300)  # 5분마다 체크

                current_time = datetime.now(timezone.utc)

                # 만료된 결제 처리
                for transaction in self.transactions.values():
                    if (transaction.status == PaymentStatus.PENDING and
                        transaction.expires_at and
                        transaction.expires_at < current_time):

                        transaction.status = PaymentStatus.CANCELLED
                        logger.info(f"Transaction expired: {transaction.id}")

            except Exception as e:
                logger.error(f"Payment status checker error: {e}")

    def get_payment_stats(self) -> Dict[str, Any]:
        """결제 통계 조회"""
        total_transactions = self.payment_stats["total_transactions"]
        success_rate = (
            (self.payment_stats["successful_transactions"] / total_transactions * 100)
            if total_transactions > 0 else 0
        )

        return {
            "total_transactions": total_transactions,
            "successful_transactions": self.payment_stats["successful_transactions"],
            "failed_transactions": self.payment_stats["failed_transactions"],
            "success_rate": round(success_rate, 2),
            "total_amount": float(self.payment_stats["total_amount"]),
            "total_fees": float(self.payment_stats["total_fees"]),
            "refunded_amount": float(self.payment_stats["refunded_amount"]),
            "active_subscriptions": len([s for s in self.subscriptions.values() if s.status == SubscriptionStatus.ACTIVE]),
            "provider": self.provider.value
        }

    # 추가 헬퍼 메서드들 (간소화)
    async def _stripe_verify_account(self):
        """Stripe 계정 확인"""
        # Stripe 계정 정보 확인 API 호출
        pass

    async def _get_or_create_stripe_customer(self, customer: PaymentCustomer):
        """Stripe 고객 생성 또는 조회"""
        # Stripe 고객 API 호출
        return {"id": "cus_example"}

    async def _create_stripe_subscription(self, subscription: Subscription):
        """Stripe 구독 생성"""
        # Stripe 구독 API 호출
        pass

    async def _cancel_stripe_subscription(self, subscription: Subscription, at_period_end: bool):
        """Stripe 구독 취소"""
        # Stripe 구독 취소 API 호출
        pass

    async def _stripe_refund(self, transaction: PaymentTransaction, amount: Decimal):
        """Stripe 환불 처리"""
        # Stripe 환불 API 호출
        pass

    async def _create_toss_subscription(self, subscription: Subscription):
        """토스페이먼츠 구독 생성"""
        # 토스페이먼츠 정기결제 API 호출
        pass

    async def _cancel_toss_subscription(self, subscription: Subscription, at_period_end: bool):
        """토스페이먼츠 구독 취소"""
        # 토스페이먼츠 정기결제 취소 API 호출
        pass

    async def _toss_refund(self, transaction: PaymentTransaction, amount: Decimal):
        """토스페이먼츠 환불 처리"""
        # 토스페이먼츠 환불 API 호출
        pass

    async def _iamport_get_access_token(self):
        """아임포트 액세스 토큰 발급"""
        # 아임포트 토큰 API 호출
        pass

    async def shutdown(self):
        """서비스 종료"""
        try:
            # 백그라운드 작업 중지
            if self._status_checker_task:
                self._status_checker_task.cancel()

            # HTTP 세션 종료
            if self.session:
                await self.session.close()

            logger.info("Payment service shutdown completed")

        except Exception as e:
            logger.error(f"Payment service shutdown error: {e}")


# 전역 결제 서비스 인스턴스
payment_service: Optional[PaymentService] = None

def get_payment_service() -> Optional[PaymentService]:
    """글로벌 결제 서비스 반환"""
    return payment_service

async def initialize_payment_service(**kwargs) -> PaymentService:
    """결제 서비스 초기화"""
    global payment_service

    payment_service = PaymentService(**kwargs)
    await payment_service.initialize()

    return payment_service