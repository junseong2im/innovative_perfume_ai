"""
Fragrance AI 비즈니스 로직 및 과금 시스템
사용량 기반 요금 정책, 구독 관리, 결제 처리
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import uuid
from dataclasses import dataclass, asdict
import asyncio

from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Integer, DateTime, Decimal as SQLDecimal, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from ..core.exceptions_advanced import BusinessLogicError, ErrorCode
from ..core.config import settings

Base = declarative_base()

class SubscriptionTier(str, Enum):
    """구독 계층"""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class UsageType(str, Enum):
    """사용량 타입"""
    API_CALL = "api_call"
    FRAGRANCE_GENERATION = "fragrance_generation"
    SEMANTIC_SEARCH = "semantic_search"
    RAG_QUERY = "rag_query"
    PREMIUM_FEATURE = "premium_feature"
    STORAGE_GB = "storage_gb"
    EMBEDDING_GENERATION = "embedding_generation"

class BillingCycle(str, Enum):
    """청구 주기"""
    MONTHLY = "monthly"
    YEARLY = "yearly"
    PAY_AS_YOU_GO = "pay_as_you_go"

class PaymentStatus(str, Enum):
    """결제 상태"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"

# ==============================================================================
# 데이터베이스 모델
# ==============================================================================

class Subscription(Base):
    """구독 정보"""
    __tablename__ = "subscriptions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    tier = Column(String, nullable=False)
    billing_cycle = Column(String, nullable=False)
    status = Column(String, nullable=False)
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False)
    trial_end = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계 설정
    usage_records = relationship("UsageRecord", back_populates="subscription")
    invoices = relationship("Invoice", back_populates="subscription")

class UsageRecord(Base):
    """사용량 기록"""
    __tablename__ = "usage_records"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String, ForeignKey("subscriptions.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    usage_type = Column(String, nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(SQLDecimal(10, 4), nullable=False)
    total_cost = Column(SQLDecimal(10, 2), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata = Column(Text, nullable=True)  # JSON으로 저장

    # 관계 설정
    subscription = relationship("Subscription", back_populates="usage_records")

class Invoice(Base):
    """청구서"""
    __tablename__ = "invoices"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String, ForeignKey("subscriptions.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(SQLDecimal(10, 2), nullable=False)
    tax_amount = Column(SQLDecimal(10, 2), default=0)
    total_amount = Column(SQLDecimal(10, 2), nullable=False)
    currency = Column(String, default="USD")
    status = Column(String, nullable=False)
    billing_period_start = Column(DateTime, nullable=False)
    billing_period_end = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=False)
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    invoice_items = Column(Text, nullable=True)  # JSON으로 저장

    # 관계 설정
    subscription = relationship("Subscription", back_populates="invoices")

class PaymentTransaction(Base):
    """결제 트랜잭션"""
    __tablename__ = "payment_transactions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String, ForeignKey("invoices.id"), nullable=True)
    user_id = Column(String, nullable=False, index=True)
    amount = Column(SQLDecimal(10, 2), nullable=False)
    currency = Column(String, default="USD")
    payment_method = Column(String, nullable=False)
    payment_provider = Column(String, nullable=False)
    provider_transaction_id = Column(String, nullable=True)
    status = Column(String, nullable=False)
    failure_reason = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

# ==============================================================================
# 요금 정책 정의
# ==============================================================================

@dataclass
class PricingRule:
    """요금 규칙"""
    usage_type: UsageType
    unit_price: Decimal
    included_quota: int = 0
    overage_price: Optional[Decimal] = None

@dataclass
class SubscriptionPlan:
    """구독 계획"""
    tier: SubscriptionTier
    name: str
    monthly_price: Decimal
    yearly_price: Decimal
    features: List[str]
    pricing_rules: List[PricingRule]
    limits: Dict[str, int]

class PricingEngine:
    """요금 계산 엔진"""

    def __init__(self):
        self.plans = self._initialize_plans()

    def _initialize_plans(self) -> Dict[SubscriptionTier, SubscriptionPlan]:
        """구독 계획 초기화"""
        return {
            SubscriptionTier.FREE: SubscriptionPlan(
                tier=SubscriptionTier.FREE,
                name="Free Tier",
                monthly_price=Decimal("0.00"),
                yearly_price=Decimal("0.00"),
                features=[
                    "Basic fragrance search",
                    "Limited AI recommendations",
                    "Community support"
                ],
                pricing_rules=[
                    PricingRule(UsageType.API_CALL, Decimal("0.00"), included_quota=1000),
                    PricingRule(UsageType.FRAGRANCE_GENERATION, Decimal("0.00"), included_quota=10),
                    PricingRule(UsageType.SEMANTIC_SEARCH, Decimal("0.00"), included_quota=100)
                ],
                limits={
                    "max_api_calls_per_day": 100,
                    "max_fragrance_generations_per_day": 5,
                    "storage_gb": 1
                }
            ),

            SubscriptionTier.BASIC: SubscriptionPlan(
                tier=SubscriptionTier.BASIC,
                name="Basic Plan",
                monthly_price=Decimal("29.99"),
                yearly_price=Decimal("299.99"),
                features=[
                    "Advanced fragrance search",
                    "AI-powered recommendations",
                    "Basic analytics",
                    "Email support",
                    "API access"
                ],
                pricing_rules=[
                    PricingRule(UsageType.API_CALL, Decimal("0.01"), included_quota=10000, overage_price=Decimal("0.005")),
                    PricingRule(UsageType.FRAGRANCE_GENERATION, Decimal("0.50"), included_quota=100, overage_price=Decimal("0.25")),
                    PricingRule(UsageType.SEMANTIC_SEARCH, Decimal("0.02"), included_quota=5000, overage_price=Decimal("0.01")),
                    PricingRule(UsageType.RAG_QUERY, Decimal("0.03"), included_quota=1000, overage_price=Decimal("0.02"))
                ],
                limits={
                    "max_api_calls_per_day": 1000,
                    "max_fragrance_generations_per_day": 50,
                    "storage_gb": 10
                }
            ),

            SubscriptionTier.PRO: SubscriptionPlan(
                tier=SubscriptionTier.PRO,
                name="Professional Plan",
                monthly_price=Decimal("99.99"),
                yearly_price=Decimal("999.99"),
                features=[
                    "All Basic features",
                    "Advanced AI models",
                    "Custom fragrance profiles",
                    "Advanced analytics",
                    "Priority support",
                    "API rate limiting removal"
                ],
                pricing_rules=[
                    PricingRule(UsageType.API_CALL, Decimal("0.005"), included_quota=50000, overage_price=Decimal("0.002")),
                    PricingRule(UsageType.FRAGRANCE_GENERATION, Decimal("0.30"), included_quota=500, overage_price=Decimal("0.15")),
                    PricingRule(UsageType.SEMANTIC_SEARCH, Decimal("0.015"), included_quota=20000, overage_price=Decimal("0.008")),
                    PricingRule(UsageType.RAG_QUERY, Decimal("0.02"), included_quota=5000, overage_price=Decimal("0.01")),
                    PricingRule(UsageType.PREMIUM_FEATURE, Decimal("0.10"), included_quota=1000, overage_price=Decimal("0.05"))
                ],
                limits={
                    "max_api_calls_per_day": 10000,
                    "max_fragrance_generations_per_day": 200,
                    "storage_gb": 100
                }
            ),

            SubscriptionTier.ENTERPRISE: SubscriptionPlan(
                tier=SubscriptionTier.ENTERPRISE,
                name="Enterprise Plan",
                monthly_price=Decimal("499.99"),
                yearly_price=Decimal("4999.99"),
                features=[
                    "All Pro features",
                    "Custom AI model training",
                    "White-label solutions",
                    "Dedicated support",
                    "SLA guarantees",
                    "Custom integrations"
                ],
                pricing_rules=[
                    PricingRule(UsageType.API_CALL, Decimal("0.002"), included_quota=500000, overage_price=Decimal("0.001")),
                    PricingRule(UsageType.FRAGRANCE_GENERATION, Decimal("0.20"), included_quota=5000, overage_price=Decimal("0.10")),
                    PricingRule(UsageType.SEMANTIC_SEARCH, Decimal("0.01"), included_quota=100000, overage_price=Decimal("0.005")),
                    PricingRule(UsageType.RAG_QUERY, Decimal("0.015"), included_quota=50000, overage_price=Decimal("0.008")),
                    PricingRule(UsageType.PREMIUM_FEATURE, Decimal("0.05"), included_quota=10000, overage_price=Decimal("0.025"))
                ],
                limits={
                    "max_api_calls_per_day": 100000,
                    "max_fragrance_generations_per_day": 1000,
                    "storage_gb": 1000
                }
            )
        }

    def get_plan(self, tier: SubscriptionTier) -> SubscriptionPlan:
        """구독 계획 조회"""
        return self.plans.get(tier)

    def calculate_usage_cost(
        self,
        tier: SubscriptionTier,
        usage_type: UsageType,
        quantity: int,
        current_usage: int = 0
    ) -> Dict[str, Any]:
        """사용량 비용 계산"""
        plan = self.get_plan(tier)
        if not plan:
            raise BusinessLogicError(
                message=f"Unknown subscription tier: {tier}",
                error_code=ErrorCode.RESOURCE_NOT_FOUND
            )

        # 해당 사용량 타입의 요금 규칙 찾기
        pricing_rule = None
        for rule in plan.pricing_rules:
            if rule.usage_type == usage_type:
                pricing_rule = rule
                break

        if not pricing_rule:
            # 기본 요금 적용
            return {
                "base_cost": Decimal("0.00"),
                "overage_cost": Decimal("0.00"),
                "total_cost": Decimal("0.00"),
                "included_in_plan": True
            }

        # 할당량 확인
        total_usage = current_usage + quantity
        included_usage = min(total_usage, pricing_rule.included_quota)
        overage_usage = max(0, total_usage - pricing_rule.included_quota)
        overage_from_this_request = max(0, quantity - max(0, pricing_rule.included_quota - current_usage))

        # 비용 계산
        base_cost = Decimal("0.00")
        if current_usage < pricing_rule.included_quota:
            billable_base_usage = min(quantity, pricing_rule.included_quota - current_usage)
            base_cost = billable_base_usage * pricing_rule.unit_price

        overage_cost = Decimal("0.00")
        if overage_from_this_request > 0 and pricing_rule.overage_price:
            overage_cost = overage_from_this_request * pricing_rule.overage_price

        total_cost = base_cost + overage_cost

        return {
            "base_cost": base_cost,
            "overage_cost": overage_cost,
            "total_cost": total_cost,
            "included_usage": included_usage,
            "overage_usage": overage_usage,
            "included_in_plan": total_cost == Decimal("0.00")
        }

# ==============================================================================
# 구독 관리자
# ==============================================================================

class SubscriptionManager:
    """구독 관리자"""

    def __init__(self, db_session, pricing_engine: PricingEngine):
        self.db = db_session
        self.pricing_engine = pricing_engine

    async def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        trial_days: int = 14
    ) -> Subscription:
        """새 구독 생성"""

        # 기존 활성 구독 확인
        existing_subscription = self.db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active"
        ).first()

        if existing_subscription:
            raise BusinessLogicError(
                message="User already has an active subscription",
                error_code=ErrorCode.RESOURCE_ALREADY_EXISTS,
                details={"existing_subscription_id": existing_subscription.id}
            )

        # 구독 기간 설정
        now = datetime.utcnow()
        if billing_cycle == BillingCycle.MONTHLY:
            period_end = now + timedelta(days=30)
        elif billing_cycle == BillingCycle.YEARLY:
            period_end = now + timedelta(days=365)
        else:
            period_end = now + timedelta(days=30)  # 기본값

        trial_end = now + timedelta(days=trial_days) if trial_days > 0 else None

        # 구독 생성
        subscription = Subscription(
            user_id=user_id,
            tier=tier.value,
            billing_cycle=billing_cycle.value,
            status="active",
            current_period_start=now,
            current_period_end=period_end,
            trial_end=trial_end
        )

        self.db.add(subscription)
        self.db.commit()
        self.db.refresh(subscription)

        return subscription

    async def record_usage(
        self,
        subscription_id: str,
        usage_type: UsageType,
        quantity: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UsageRecord:
        """사용량 기록"""

        # 구독 정보 조회
        subscription = self.db.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()

        if not subscription:
            raise BusinessLogicError(
                message="Subscription not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND
            )

        # 구독 상태 확인
        if subscription.status != "active":
            raise BusinessLogicError(
                message="Subscription is not active",
                error_code=ErrorCode.OPERATION_NOT_ALLOWED,
                details={"subscription_status": subscription.status}
            )

        # 현재 사용량 조회
        current_period_start = subscription.current_period_start
        current_usage = self.db.query(UsageRecord).filter(
            UsageRecord.subscription_id == subscription_id,
            UsageRecord.usage_type == usage_type.value,
            UsageRecord.timestamp >= current_period_start
        ).count()

        # 비용 계산
        tier = SubscriptionTier(subscription.tier)
        cost_breakdown = self.pricing_engine.calculate_usage_cost(
            tier, usage_type, quantity, current_usage
        )

        # 사용량 기록 생성
        usage_record = UsageRecord(
            subscription_id=subscription_id,
            user_id=subscription.user_id,
            usage_type=usage_type.value,
            quantity=quantity,
            unit_price=cost_breakdown["total_cost"] / quantity if quantity > 0 else Decimal("0.00"),
            total_cost=cost_breakdown["total_cost"],
            metadata=json.dumps(metadata) if metadata else None
        )

        self.db.add(usage_record)
        self.db.commit()
        self.db.refresh(usage_record)

        return usage_record

    async def check_usage_limits(
        self,
        subscription_id: str,
        usage_type: UsageType,
        requested_quantity: int = 1
    ) -> Dict[str, Any]:
        """사용량 제한 확인"""

        subscription = self.db.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()

        if not subscription:
            raise BusinessLogicError(
                message="Subscription not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND
            )

        tier = SubscriptionTier(subscription.tier)
        plan = self.pricing_engine.get_plan(tier)

        # 일일 제한 확인
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_usage = self.db.query(UsageRecord).filter(
            UsageRecord.subscription_id == subscription_id,
            UsageRecord.usage_type == usage_type.value,
            UsageRecord.timestamp >= today_start
        ).count()

        daily_limit_key = f"max_{usage_type.value}s_per_day"
        daily_limit = plan.limits.get(daily_limit_key, float('inf'))

        can_proceed = (today_usage + requested_quantity) <= daily_limit

        return {
            "can_proceed": can_proceed,
            "current_usage": today_usage,
            "daily_limit": daily_limit,
            "remaining": max(0, daily_limit - today_usage),
            "period": "daily"
        }

# ==============================================================================
# 과금 서비스
# ==============================================================================

class BillingService:
    """과금 서비스"""

    def __init__(self, db_session):
        self.db = db_session
        self.pricing_engine = PricingEngine()
        self.subscription_manager = SubscriptionManager(db_session, self.pricing_engine)

    async def process_usage(
        self,
        user_id: str,
        usage_type: UsageType,
        quantity: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """사용량 처리"""

        # 활성 구독 조회
        subscription = self.db.query(Subscription).filter(
            Subscription.user_id == user_id,
            Subscription.status == "active"
        ).first()

        if not subscription:
            # 무료 사용자를 위한 기본 구독 생성
            subscription = await self.subscription_manager.create_subscription(
                user_id=user_id,
                tier=SubscriptionTier.FREE,
                trial_days=0
            )

        # 사용량 제한 확인
        limit_check = await self.subscription_manager.check_usage_limits(
            subscription.id, usage_type, quantity
        )

        if not limit_check["can_proceed"]:
            raise BusinessLogicError(
                message="Usage limit exceeded",
                error_code=ErrorCode.QUOTA_EXCEEDED,
                details=limit_check,
                user_message=f"{usage_type.value} 일일 사용 한도를 초과했습니다."
            )

        # 사용량 기록
        usage_record = await self.subscription_manager.record_usage(
            subscription.id, usage_type, quantity, metadata
        )

        return {
            "usage_record_id": usage_record.id,
            "cost": float(usage_record.total_cost),
            "subscription_tier": subscription.tier,
            "remaining_quota": limit_check["remaining"] - quantity
        }

    async def generate_invoice(self, subscription_id: str, period_start: datetime, period_end: datetime) -> Invoice:
        """청구서 생성"""

        subscription = self.db.query(Subscription).filter(
            Subscription.id == subscription_id
        ).first()

        if not subscription:
            raise BusinessLogicError(
                message="Subscription not found",
                error_code=ErrorCode.RESOURCE_NOT_FOUND
            )

        # 해당 기간의 사용량 조회
        usage_records = self.db.query(UsageRecord).filter(
            UsageRecord.subscription_id == subscription_id,
            UsageRecord.timestamp >= period_start,
            UsageRecord.timestamp < period_end
        ).all()

        # 구독료 계산
        plan = self.pricing_engine.get_plan(SubscriptionTier(subscription.tier))
        if subscription.billing_cycle == BillingCycle.MONTHLY.value:
            base_amount = plan.monthly_price
        else:
            base_amount = plan.yearly_price

        # 사용량 비용 합계
        usage_amount = sum(record.total_cost for record in usage_records)
        total_amount = base_amount + usage_amount

        # 세금 계산 (간단한 예시: 10%)
        tax_amount = total_amount * Decimal("0.10")

        # 청구서 생성
        invoice = Invoice(
            subscription_id=subscription_id,
            user_id=subscription.user_id,
            amount=total_amount,
            tax_amount=tax_amount,
            total_amount=total_amount + tax_amount,
            status=PaymentStatus.PENDING.value,
            billing_period_start=period_start,
            billing_period_end=period_end,
            due_date=period_end + timedelta(days=7)
        )

        self.db.add(invoice)
        self.db.commit()
        self.db.refresh(invoice)

        return invoice

# ==============================================================================
# 사용량 추적 데코레이터
# ==============================================================================

def track_usage(usage_type: UsageType, quantity: int = 1):
    """사용량 추적 데코레이터"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 요청에서 사용자 ID 추출 (실제 구현에서는 인증 시스템과 연동)
            user_id = kwargs.get('current_user', {}).get('id')

            if user_id:
                # 비즈니스 로직 실행 전 사용량 확인
                billing_service = BillingService(db_session=None)  # 실제 DB 세션 주입 필요
                try:
                    usage_result = await billing_service.process_usage(
                        user_id=user_id,
                        usage_type=usage_type,
                        quantity=quantity
                    )

                    # 원본 함수 실행
                    result = await func(*args, **kwargs)

                    # 결과에 사용량 정보 추가
                    if isinstance(result, dict):
                        result['usage_info'] = usage_result

                    return result

                except BusinessLogicError as e:
                    if e.error_code == ErrorCode.QUOTA_EXCEEDED:
                        # 할당량 초과 에러 처리
                        raise e
                    # 기타 에러는 원본 함수 실행
                    return await func(*args, **kwargs)
            else:
                # 인증되지 않은 사용자는 그냥 실행
                return await func(*args, **kwargs)

        return wrapper
    return decorator