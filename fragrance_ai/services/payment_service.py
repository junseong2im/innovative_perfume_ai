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
    amount: Decimal
    currency: str
    status: PaymentStatus = PaymentStatus.PENDING
    payment_method: str = "card"
    customer: Optional[PaymentCustomer] = None
    items: List[PaymentItem] = field(default_factory=list)
    total_amount: Optional[Decimal] = None
    provider_payment_id: Optional[str] = None
    provider_transaction_id: Optional[str] = None
    provider_payment_url: Optional[str] = None
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    failure_reason: Optional[str] = None
    refund_amount: Decimal = Decimal('0')
    refunded_amount: Decimal = Decimal('0')
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
    customer_email: Optional[str] = None
    customer_name: Optional[str] = None
    interval: str = "monthly"
    cancel_at_period_end: bool = False
    cancel_reason: Optional[str] = None
    cancelled_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    resumed_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    successful_payments: int = 0
    failed_payments: int = 0
    past_due_since: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
            elif self.provider == PaymentProvider.PAYPAL:
                await self._create_paypal_payment(transaction, success_url, cancel_url)
            elif self.provider == PaymentProvider.KAKAO_PAY:
                await self._create_kakao_pay_payment(transaction, success_url, cancel_url)
            elif self.provider == PaymentProvider.NAVER_PAY:
                await self._create_naver_pay_payment(transaction, success_url, cancel_url)
            else:
                # 일반 PG 게이트웨이 사용
                await self._create_generic_pg_payment(transaction, success_url, cancel_url)

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
            elif self.provider == PaymentProvider.PAYPAL:
                await self._create_paypal_subscription(subscription)
            elif self.provider == PaymentProvider.KAKAO_PAY:
                await self._create_kakaopay_subscription(subscription)
            else:
                # 기본 구독 처리 (데이터베이스에만 저장)
                logger.warning(f"Subscription API not implemented for {self.provider.value}, using local storage only")
                subscription.status = SubscriptionStatus.ACTIVE

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
            elif self.provider == PaymentProvider.PAYPAL:
                await self._cancel_paypal_subscription(subscription, cancel_at_period_end)
            elif self.provider == PaymentProvider.KAKAO_PAY:
                await self._cancel_kakaopay_subscription(subscription, cancel_at_period_end)
            else:
                # 기본 취소 처리 (로컬 상태만 변경)
                logger.warning(f"Subscription cancellation API not implemented for {self.provider.value}, updating local status only")

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
            elif self.provider == PaymentProvider.PAYPAL:
                await self._paypal_refund(transaction, refund_amount)
            elif self.provider == PaymentProvider.KAKAO_PAY:
                await self._kakaopay_refund(transaction, refund_amount)
            elif self.provider == PaymentProvider.NAVER_PAY:
                await self._naverpay_refund(transaction, refund_amount)
            else:
                # 기본 환불 처리 (로컬 상태만 변경)
                logger.warning(f"Refund API not implemented for {self.provider.value}, updating local status only")
                transaction.status = PaymentStatus.REFUNDED

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
        subscription_id = data.get("subscription_id")
        provider_subscription_id = data.get("provider_subscription_id")

        if subscription_id and subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]

            # 구독 상태 업데이트
            subscription.status = SubscriptionStatus.ACTIVE
            subscription.provider_subscription_id = provider_subscription_id or subscription.provider_subscription_id
            subscription.current_period_start = datetime.now(timezone.utc)

            # 다음 결제일 계산
            if subscription.interval == "monthly":
                next_period = subscription.current_period_start + timedelta(days=30)
            elif subscription.interval == "yearly":
                next_period = subscription.current_period_start + timedelta(days=365)
            else:
                next_period = subscription.current_period_start + timedelta(days=30)

            subscription.current_period_end = next_period

            # 이메일 알림 전송 (비동기)
            await self._send_subscription_confirmation_email(subscription)

            # 통계 업데이트
            self.subscription_stats["active_subscriptions"] += 1

            logger.info(f"Subscription created via webhook: {subscription_id}")
        else:
            logger.warning(f"Unknown subscription in webhook: {subscription_id}")

    async def _handle_subscription_updated(self, data: Dict[str, Any]):
        """구독 업데이트 웹훅 핸들러"""
        subscription_id = data.get("subscription_id")

        if subscription_id and subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]

            # 업데이트된 필드 처리
            if "plan_id" in data:
                old_plan = subscription.plan_id
                subscription.plan_id = data["plan_id"]
                logger.info(f"Subscription {subscription_id} plan changed: {old_plan} -> {data['plan_id']}")

            if "amount" in data:
                subscription.amount = Decimal(str(data["amount"]))

            if "interval" in data:
                subscription.interval = data["interval"]

            if "status" in data:
                new_status = SubscriptionStatus[data["status"].upper()]
                if subscription.status != new_status:
                    subscription.status = new_status

                    # 상태 변경에 따른 처리
                    if new_status == SubscriptionStatus.PAUSED:
                        subscription.paused_at = datetime.now(timezone.utc)
                    elif new_status == SubscriptionStatus.ACTIVE and subscription.paused_at:
                        subscription.resumed_at = datetime.now(timezone.utc)
                        subscription.paused_at = None

            subscription.updated_at = datetime.now(timezone.utc)

            # 변경사항 알림
            await self._send_subscription_update_email(subscription)

            logger.info(f"Subscription updated via webhook: {subscription_id}")
        else:
            logger.warning(f"Unknown subscription in update webhook: {subscription_id}")

    async def _handle_subscription_cancelled(self, data: Dict[str, Any]):
        """구독 취소 웹훅 핸들러"""
        subscription_id = data.get("subscription_id")

        if subscription_id and subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]

            # 구독 상태 업데이트
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.cancelled_at = datetime.now(timezone.utc)
            subscription.cancel_reason = data.get("cancel_reason", "Customer requested")

            # 즉시 취소인지 기간 종료 시 취소인지 확인
            if data.get("cancel_at_period_end"):
                subscription.cancel_at_period_end = True
                # 현재 기간 종료까지는 활성 상태 유지
                subscription.status = SubscriptionStatus.ACTIVE
                logger.info(f"Subscription {subscription_id} will cancel at period end")
            else:
                # 즉시 취소
                subscription.status = SubscriptionStatus.CANCELLED

                # 환불 처리 필요한 경우
                if data.get("refund_amount"):
                    await self._process_subscription_refund(
                        subscription,
                        Decimal(str(data["refund_amount"]))
                    )

            # 통계 업데이트
            self.subscription_stats["cancelled_subscriptions"] += 1
            self.subscription_stats["active_subscriptions"] -= 1

            # 취소 확인 이메일
            await self._send_subscription_cancellation_email(subscription)

            logger.info(f"Subscription cancelled via webhook: {subscription_id}")
        else:
            logger.warning(f"Unknown subscription in cancel webhook: {subscription_id}")

    async def _handle_invoice_payment_success(self, data: Dict[str, Any]):
        """인보이스 결제 성공 웹훅 핸들러"""
        subscription_id = data.get("subscription_id")
        invoice_id = data.get("invoice_id")
        amount = Decimal(str(data.get("amount", 0)))

        if subscription_id and subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]

            # 새로운 트랜잭션 생성
            transaction = PaymentTransaction(
                id=f"invoice_{invoice_id}",
                provider_payment_id=invoice_id,
                amount=amount,
                currency=subscription.currency,
                status=PaymentStatus.COMPLETED,
                payment_method="subscription",
                customer_email=subscription.customer_email,
                customer_name=subscription.customer_name,
                completed_at=datetime.now(timezone.utc),
                metadata={
                    "subscription_id": subscription_id,
                    "invoice_id": invoice_id,
                    "billing_period": data.get("billing_period")
                }
            )

            self.transactions[transaction.id] = transaction

            # 구독 기간 업데이트
            subscription.current_period_start = datetime.now(timezone.utc)

            if subscription.interval == "monthly":
                subscription.current_period_end = subscription.current_period_start + timedelta(days=30)
            elif subscription.interval == "yearly":
                subscription.current_period_end = subscription.current_period_start + timedelta(days=365)

            # 성공 카운터 증가
            subscription.successful_payments = subscription.successful_payments + 1 if subscription.successful_payments else 1

            # 영수증 전송
            await self._send_invoice_receipt_email(subscription, transaction)

            logger.info(f"Invoice payment success for subscription {subscription_id}: {invoice_id}")
        else:
            logger.warning(f"Unknown subscription in invoice success webhook: {subscription_id}")

    async def _handle_invoice_payment_failure(self, data: Dict[str, Any]):
        """인보이스 결제 실패 웹훅 핸들러"""
        subscription_id = data.get("subscription_id")
        invoice_id = data.get("invoice_id")
        failure_reason = data.get("failure_reason", "Unknown error")

        if subscription_id and subscription_id in self.subscriptions:
            subscription = self.subscriptions[subscription_id]

            # 실패 카운터 증가
            subscription.failed_payments = subscription.failed_payments + 1 if subscription.failed_payments else 1

            # 실패 트랜잭션 기록
            transaction = PaymentTransaction(
                id=f"invoice_{invoice_id}_failed",
                provider_payment_id=invoice_id,
                amount=subscription.amount,
                currency=subscription.currency,
                status=PaymentStatus.FAILED,
                payment_method="subscription",
                customer_email=subscription.customer_email,
                customer_name=subscription.customer_name,
                failure_reason=failure_reason,
                metadata={
                    "subscription_id": subscription_id,
                    "invoice_id": invoice_id,
                    "retry_count": data.get("retry_count", 0)
                }
            )

            self.transactions[transaction.id] = transaction

            # 재시도 횟수 확인
            retry_count = data.get("retry_count", 0)
            max_retries = 3

            if retry_count >= max_retries:
                # 최대 재시도 횟수 초과 - 구독 일시정지
                subscription.status = SubscriptionStatus.PAST_DUE
                subscription.past_due_since = datetime.now(timezone.utc)

                # 서비스 접근 제한 알림
                await self._send_subscription_suspension_email(subscription, failure_reason)

                logger.warning(f"Subscription {subscription_id} suspended due to payment failures")
            else:
                # 재시도 예정 알림
                next_retry = datetime.now(timezone.utc) + timedelta(days=1)
                await self._send_payment_retry_email(subscription, failure_reason, next_retry)

                logger.info(f"Invoice payment failed for subscription {subscription_id}, retry {retry_count}/{max_retries}")
        else:
            logger.warning(f"Unknown subscription in invoice failure webhook: {subscription_id}")

    async def _send_subscription_confirmation_email(self, subscription: Subscription):
        """구독 확인 이메일 전송"""
        # 이메일 서비스 연동
        logger.info(f"Sending subscription confirmation email to {subscription.customer_email}")

    async def _send_subscription_update_email(self, subscription: Subscription):
        """구독 변경 알림 이메일"""
        logger.info(f"Sending subscription update email to {subscription.customer_email}")

    async def _send_subscription_cancellation_email(self, subscription: Subscription):
        """구독 취소 확인 이메일"""
        logger.info(f"Sending cancellation confirmation to {subscription.customer_email}")

    async def _send_invoice_receipt_email(self, subscription: Subscription, transaction: PaymentTransaction):
        """결제 영수증 이메일"""
        logger.info(f"Sending invoice receipt to {subscription.customer_email}")

    async def _send_subscription_suspension_email(self, subscription: Subscription, reason: str):
        """구독 일시정지 알림"""
        logger.info(f"Sending suspension notice to {subscription.customer_email}: {reason}")

    async def _send_payment_retry_email(self, subscription: Subscription, reason: str, next_retry: datetime):
        """결제 재시도 알림"""
        logger.info(f"Sending retry notice to {subscription.customer_email}")

    async def _process_subscription_refund(self, subscription: Subscription, amount: Decimal):
        """구독 환불 처리"""
        # 환불 로직 구현
        logger.info(f"Processing refund of {amount} for subscription {subscription.id}")

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

    async def _create_paypal_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """PayPal 결제 생성"""
        config = self.provider_config.get("paypal", {})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {await self._get_paypal_access_token()}"
        }

        payload = {
            "intent": "CAPTURE",
            "purchase_units": [{
                "reference_id": transaction.transaction_id,
                "amount": {
                    "currency_code": transaction.currency.value,
                    "value": str(transaction.total_amount)
                },
                "description": transaction.description
            }],
            "payment_source": {
                "paypal": {
                    "experience_context": {
                        "payment_method_preference": "IMMEDIATE_PAYMENT_REQUIRED",
                        "brand_name": "Fragrance AI",
                        "locale": "ko-KR",
                        "return_url": success_url or f"{self.base_url}/payment/success",
                        "cancel_url": cancel_url or f"{self.base_url}/payment/cancel"
                    }
                }
            }
        }

        async with self.session.post(
            f"{config.get('api_url', 'https://api-m.paypal.com')}/v2/checkout/orders",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 201:
                result = await response.json()
                transaction.provider_payment_id = result["id"]
                transaction.payment_url = next(
                    (link["href"] for link in result["links"] if link["rel"] == "payer-action"),
                    None
                )
                transaction.provider_data = result
            else:
                error = await response.text()
                raise Exception(f"PayPal payment creation failed: {error}")

    async def _create_kakao_pay_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """카카오페이 결제 생성"""
        config = self.provider_config.get("kakao_pay", {})

        headers = {
            "Authorization": f"KakaoAK {config.get('admin_key')}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "cid": config.get("cid", "TC0ONETIME"),  # 테스트용 CID
            "partner_order_id": transaction.transaction_id,
            "partner_user_id": transaction.customer.customer_id,
            "item_name": transaction.description[:100],
            "quantity": 1,
            "total_amount": int(transaction.total_amount),
            "tax_free_amount": 0,
            "approval_url": success_url or f"{self.base_url}/payment/kakao/success",
            "cancel_url": cancel_url or f"{self.base_url}/payment/kakao/cancel",
            "fail_url": cancel_url or f"{self.base_url}/payment/kakao/fail"
        }

        async with self.session.post(
            "https://kapi.kakao.com/v1/payment/ready",
            data=data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                transaction.provider_payment_id = result["tid"]
                transaction.payment_url = result["next_redirect_pc_url"]
                transaction.provider_data = result
            else:
                error = await response.text()
                raise Exception(f"KakaoPay payment creation failed: {error}")

    async def _create_naver_pay_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """네이버페이 결제 생성"""
        config = self.provider_config.get("naver_pay", {})

        headers = {
            "X-Naver-Client-Id": config.get("client_id"),
            "X-Naver-Client-Secret": config.get("client_secret"),
            "Content-Type": "application/json"
        }

        payload = {
            "merchantPayKey": transaction.transaction_id,
            "productName": transaction.description[:100],
            "totalPayAmount": int(transaction.total_amount),
            "taxScopeAmount": int(transaction.total_amount),
            "taxExScopeAmount": 0,
            "returnUrl": success_url or f"{self.base_url}/payment/naver/success",
            "productItems": [{
                "categoryType": "ETC",
                "categoryId": "ETC",
                "uid": transaction.transaction_id,
                "name": transaction.description[:100],
                "payReferrer": "ETC",
                "count": 1
            }]
        }

        async with self.session.post(
            f"{config.get('api_url', 'https://dev.apis.naver.com')}/naverpay-partner/naverpay/payments/v2.2/reserve",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result["code"] == "Success":
                    transaction.provider_payment_id = result["body"]["reserveId"]
                    transaction.payment_url = result["body"]["paymentUrl"]
                    transaction.provider_data = result["body"]
                else:
                    raise Exception(f"NaverPay error: {result.get('message')}")
            else:
                error = await response.text()
                raise Exception(f"NaverPay payment creation failed: {error}")

    async def _create_generic_pg_payment(
        self,
        transaction: PaymentTransaction,
        success_url: Optional[str],
        cancel_url: Optional[str]
    ):
        """일반 PG 게이트웨이를 통한 결제 생성"""
        pg_url = os.environ.get("PG_GATEWAY_URL", "http://localhost:8080/pg/payment")
        pg_token = os.environ.get("PG_GATEWAY_TOKEN", "")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {pg_token}"
        } if pg_token else {"Content-Type": "application/json"}

        payload = {
            "transaction_id": transaction.transaction_id,
            "amount": float(transaction.total_amount),
            "currency": transaction.currency.value,
            "description": transaction.description,
            "customer": {
                "id": transaction.customer.customer_id,
                "email": transaction.customer.email,
                "name": transaction.customer.name
            },
            "success_url": success_url or f"{self.base_url}/payment/success",
            "cancel_url": cancel_url or f"{self.base_url}/payment/cancel",
            "webhook_url": f"{self.base_url}/api/v1/payment/webhook",
            "metadata": transaction.metadata
        }

        try:
            async with self.session.post(
                pg_url,
                json=payload,
                headers=headers,
                timeout=30
            ) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    transaction.provider_payment_id = result.get("payment_id", transaction.transaction_id)
                    transaction.payment_url = result.get("payment_url", f"{pg_url}/{transaction.provider_payment_id}")
                    transaction.provider_data = result
                    logger.info(f"Generic PG payment created: {transaction.provider_payment_id}")
                else:
                    error = await response.text()
                    raise Exception(f"Generic PG error ({response.status}): {error}")
        except asyncio.TimeoutError:
            raise Exception("PG gateway timeout after 30 seconds")

    async def _get_paypal_access_token(self) -> str:
        """PayPal OAuth 토큰 발급"""
        config = self.provider_config.get("paypal", {})

        auth = aiohttp.BasicAuth(
            config.get("client_id"),
            config.get("client_secret")
        )

        data = {"grant_type": "client_credentials"}

        async with self.session.post(
            f"{config.get('api_url', 'https://api-m.paypal.com')}/v1/oauth2/token",
            data=data,
            auth=auth
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["access_token"]
            else:
                raise Exception("Failed to get PayPal access token")

    async def _create_toss_subscription(self, subscription: Subscription):
        """토스페이먼츠 구독 생성 - 실제 구현"""
        config = self.provider_config.get("toss_payments", {})

        headers = {
            "Authorization": f"Basic {config.get('secret_key')}",
            "Content-Type": "application/json"
        }

        payload = {
            "customerKey": subscription.customer.customer_id,
            "customerName": subscription.customer.name,
            "customerEmail": subscription.customer.email,
            "amount": int(subscription.amount),
            "orderId": subscription.subscription_id,
            "orderName": f"정기결제 - {subscription.plan_id}",
            "billingCycle": {
                "interval": subscription.billing_period.value,
                "intervalCount": 1
            },
            "autoPayment": True
        }

        async with self.session.post(
            "https://api.tosspayments.com/v1/billing/subscriptions",
            json=payload,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                subscription.provider_subscription_id = result["billingKey"]
                subscription.provider_data = result
                logger.info(f"Toss subscription created: {result['billingKey']}")
            else:
                error = await response.text()
                raise Exception(f"Toss subscription failed: {error}")

    async def _cancel_toss_subscription(self, subscription: Subscription, at_period_end: bool):
        """토스페이먼츠 구독 취소 - 실제 구현"""
        config = self.provider_config.get("toss_payments", {})

        headers = {
            "Authorization": f"Basic {config.get('secret_key')}",
            "Content-Type": "application/json"
        }

        cancel_data = {
            "cancelReason": subscription.cancel_reason or "고객 요청",
            "canceledAtPeriodEnd": at_period_end
        }

        async with self.session.post(
            f"https://api.tosspayments.com/v1/billing/subscriptions/{subscription.provider_subscription_id}/cancel",
            json=cancel_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                subscription.cancelled_at = datetime.now(timezone.utc)
                if at_period_end:
                    subscription.cancel_at_period_end = True
                else:
                    subscription.status = SubscriptionStatus.CANCELLED
                logger.info(f"Toss subscription cancelled: {subscription.provider_subscription_id}")
            else:
                error = await response.text()
                raise Exception(f"Toss cancellation failed: {error}")

    async def _toss_refund(self, transaction: PaymentTransaction, amount: Decimal):
        """토스페이먼츠 환불 처리 - 실제 구현"""
        config = self.provider_config.get("toss_payments", {})

        headers = {
            "Authorization": f"Basic {config.get('secret_key')}",
            "Content-Type": "application/json"
        }

        refund_data = {
            "cancelReason": "고객 환불 요청",
            "cancelAmount": int(amount),
            "refundReceiveAccount": transaction.metadata.get("refund_account")
        }

        async with self.session.post(
            f"https://api.tosspayments.com/v1/payments/{transaction.provider_payment_id}/cancel",
            json=refund_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                transaction.refunded_amount += amount
                if transaction.refunded_amount >= transaction.total_amount:
                    transaction.status = PaymentStatus.REFUNDED
                else:
                    transaction.status = PaymentStatus.PARTIALLY_REFUNDED
                logger.info(f"Toss refund processed: {amount} for {transaction.provider_payment_id}")
                return result
            else:
                error = await response.text()
                raise Exception(f"Toss refund failed: {error}")

    async def _create_paypal_subscription(self, subscription: Subscription):
        """PayPal 구독 생성"""
        # PayPal Subscriptions API 사용
        headers = {
            "Authorization": f"Bearer {await self._get_paypal_access_token()}",
            "Content-Type": "application/json"
        }

        subscription_data = {
            "plan_id": subscription.plan_id,
            "subscriber": {
                "name": {
                    "given_name": subscription.customer_name.split()[0] if subscription.customer_name else "Customer",
                    "surname": subscription.customer_name.split()[-1] if subscription.customer_name else "User"
                },
                "email_address": subscription.customer_email
            },
            "application_context": {
                "brand_name": "Fragrance AI",
                "locale": "ko-KR",
                "shipping_preference": "NO_SHIPPING",
                "user_action": "SUBSCRIBE_NOW",
                "return_url": "https://fragranceai.com/subscription/success",
                "cancel_url": "https://fragranceai.com/subscription/cancel"
            }
        }

        async with self.session.post(
            "https://api.paypal.com/v1/billing/subscriptions",
            json=subscription_data,
            headers=headers
        ) as response:
            if response.status == 201:
                result = await response.json()
                subscription.provider_subscription_id = result["id"]
                subscription.status = SubscriptionStatus.ACTIVE
            else:
                error = await response.text()
                raise Exception(f"PayPal subscription creation failed: {error}")

    async def _create_kakaopay_subscription(self, subscription: Subscription):
        """KakaoPay 정기결제 생성"""
        config = self.provider_config.get("kakao_pay", {})

        headers = {
            "Authorization": f"KakaoAK {config.get('admin_key')}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        subscription_data = {
            "cid": config.get("cid"),
            "sid": f"SUBSCRIPTION_{subscription.id}",
            "partner_order_id": subscription.id,
            "partner_user_id": subscription.customer_email,
            "item_name": f"Fragrance AI {subscription.plan_id} Plan",
            "quantity": 1,
            "total_amount": int(subscription.amount),
            "tax_free_amount": 0
        }

        async with self.session.post(
            "https://kapi.kakao.com/v1/payment/subscription",
            data=subscription_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                subscription.provider_subscription_id = result["sid"]
                subscription.status = SubscriptionStatus.ACTIVE
            else:
                error = await response.text()
                raise Exception(f"KakaoPay subscription creation failed: {error}")

    async def _cancel_paypal_subscription(self, subscription: Subscription, at_period_end: bool):
        """PayPal 구독 취소"""
        headers = {
            "Authorization": f"Bearer {await self._get_paypal_access_token()}",
            "Content-Type": "application/json"
        }

        cancel_data = {
            "reason": subscription.cancel_reason or "Customer requested cancellation"
        }

        # PayPal은 즉시 취소만 지원
        async with self.session.post(
            f"https://api.paypal.com/v1/billing/subscriptions/{subscription.provider_subscription_id}/cancel",
            json=cancel_data,
            headers=headers
        ) as response:
            if response.status == 204:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.cancelled_at = datetime.now(timezone.utc)
            else:
                error = await response.text()
                raise Exception(f"PayPal subscription cancellation failed: {error}")

    async def _cancel_kakaopay_subscription(self, subscription: Subscription, at_period_end: bool):
        """KakaoPay 정기결제 해지"""
        config = self.provider_config.get("kakao_pay", {})

        headers = {
            "Authorization": f"KakaoAK {config.get('admin_key')}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        cancel_data = {
            "cid": config.get("cid"),
            "sid": subscription.provider_subscription_id
        }

        async with self.session.post(
            "https://kapi.kakao.com/v1/payment/subscription/inactive",
            data=cancel_data,
            headers=headers
        ) as response:
            if response.status == 200:
                subscription.status = SubscriptionStatus.CANCELLED
                subscription.cancelled_at = datetime.now(timezone.utc)
            else:
                error = await response.text()
                raise Exception(f"KakaoPay subscription cancellation failed: {error}")

    async def _paypal_refund(self, transaction: PaymentTransaction, amount: Decimal):
        """PayPal 환불 처리"""
        headers = {
            "Authorization": f"Bearer {await self._get_paypal_access_token()}",
            "Content-Type": "application/json"
        }

        refund_data = {
            "amount": {
                "value": str(amount),
                "currency_code": transaction.currency
            },
            "note_to_payer": "Refund processed by Fragrance AI"
        }

        async with self.session.post(
            f"https://api.paypal.com/v2/payments/captures/{transaction.provider_payment_id}/refund",
            json=refund_data,
            headers=headers
        ) as response:
            if response.status == 201:
                result = await response.json()
                transaction.refunded_amount += amount
                if transaction.refunded_amount >= transaction.total_amount:
                    transaction.status = PaymentStatus.REFUNDED
                else:
                    transaction.status = PaymentStatus.PARTIALLY_REFUNDED
                return result
            else:
                error = await response.text()
                raise Exception(f"PayPal refund failed: {error}")

    async def _kakaopay_refund(self, transaction: PaymentTransaction, amount: Decimal):
        """KakaoPay 환불 처리"""
        config = self.provider_config.get("kakao_pay", {})

        headers = {
            "Authorization": f"KakaoAK {config.get('admin_key')}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        refund_data = {
            "cid": config.get("cid"),
            "tid": transaction.provider_payment_id,
            "cancel_amount": int(amount),
            "cancel_tax_free_amount": 0
        }

        async with self.session.post(
            "https://kapi.kakao.com/v1/payment/cancel",
            data=refund_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                transaction.refunded_amount += amount
                if transaction.refunded_amount >= transaction.total_amount:
                    transaction.status = PaymentStatus.REFUNDED
                else:
                    transaction.status = PaymentStatus.PARTIALLY_REFUNDED
                return result
            else:
                error = await response.text()
                raise Exception(f"KakaoPay refund failed: {error}")

    async def _naverpay_refund(self, transaction: PaymentTransaction, amount: Decimal):
        """NaverPay 환불 처리"""
        config = self.provider_config.get("naver_pay", {})

        headers = {
            "X-Naver-Client-Id": config.get("client_id"),
            "X-Naver-Client-Secret": config.get("client_secret"),
            "Content-Type": "application/json"
        }

        refund_data = {
            "paymentId": transaction.provider_payment_id,
            "cancelAmount": int(amount),
            "cancelReason": "고객 요청",
            "cancelRequester": "2"  # 2: 가맹점 관리자
        }

        async with self.session.post(
            f"https://dev.apis.naver.com/{config.get('partner_id')}/naverpay/payments/v2.2/cancel",
            json=refund_data,
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result["code"] == "Success":
                    transaction.refunded_amount += amount
                    if transaction.refunded_amount >= transaction.total_amount:
                        transaction.status = PaymentStatus.REFUNDED
                    else:
                        transaction.status = PaymentStatus.PARTIALLY_REFUNDED
                    return result
                else:
                    raise Exception(f"NaverPay refund failed: {result['message']}")
            else:
                error = await response.text()
                raise Exception(f"NaverPay refund request failed: {error}")

    async def _get_paypal_access_token(self) -> str:
        """PayPal 액세스 토큰 획득"""
        config = self.provider_config.get("paypal", {})

        import base64
        credentials = base64.b64encode(
            f"{config.get('client_id')}:{config.get('client_secret')}".encode()
        ).decode()

        headers = {
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        async with self.session.post(
            "https://api.paypal.com/v1/oauth2/token",
            data="grant_type=client_credentials",
            headers=headers
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result["access_token"]
            else:
                raise Exception("Failed to get PayPal access token")

    async def _iamport_get_access_token(self):
        """아임포트 액세스 토큰 발급 - 실제 구현"""
        config = self.provider_config.get("iamport", {})

        token_data = {
            "imp_key": config.get("imp_key"),
            "imp_secret": config.get("imp_secret")
        }

        async with self.session.post(
            "https://api.iamport.kr/users/getToken",
            json=token_data
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result["code"] == 0:
                    return result["response"]["access_token"]
                else:
                    raise Exception(f"Iamport token error: {result['message']}")
            else:
                raise Exception("Failed to get Iamport access token")

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