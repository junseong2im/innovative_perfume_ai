"""
외부 서비스 API 엔드포인트
- 이메일 서비스 API
- SMS 서비스 API
- 결제 서비스 API
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from pydantic import BaseModel, EmailStr, validator
from decimal import Decimal
from datetime import datetime

from ..services.email_service import get_email_service, EmailPriority, EmailAttachment
from ..services.sms_service import get_sms_service, SMSType
from ..services.payment_service import (
    get_payment_service, PaymentCustomer, PaymentItem, PaymentMethod,
    CurrencyCode, PaymentStatus, SubscriptionStatus
)
from ..api.dependencies import get_current_user, require_admin, require_permission
from ..core.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/external", tags=["External Services"])

# Pydantic 모델들

# 이메일 관련 모델
class SendEmailRequest(BaseModel):
    recipients: List[EmailStr]
    subject: str
    html_content: Optional[str] = None
    text_content: Optional[str] = None
    priority: str = "normal"

    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "normal", "high", "urgent"]
        if v.lower() not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v.lower()

class SendTemplateEmailRequest(BaseModel):
    recipients: List[EmailStr]
    template_name: str
    template_variables: Dict[str, Any]
    priority: str = "normal"

class EmailStatusResponse(BaseModel):
    email_id: str
    status: str
    sent_at: Optional[datetime]
    delivered_at: Optional[datetime]
    error_message: Optional[str]

# SMS 관련 모델
class SendSMSRequest(BaseModel):
    recipients: List[str]
    content: str
    sms_type: str = "sms"

    @validator('sms_type')
    def validate_sms_type(cls, v):
        valid_types = ["sms", "lms", "mms"]
        if v.lower() not in valid_types:
            raise ValueError(f"SMS type must be one of: {valid_types}")
        return v.lower()

class SendTemplateSMSRequest(BaseModel):
    recipients: List[str]
    template_name: str
    template_variables: Dict[str, Any]

class SendVerificationSMSRequest(BaseModel):
    phone_number: str
    code: str
    app_name: str = "Fragrance AI"
    expiry_minutes: int = 5

# 결제 관련 모델
class PaymentItemRequest(BaseModel):
    name: str
    quantity: int
    unit_price: Decimal
    description: Optional[str] = None

class CreatePaymentRequest(BaseModel):
    customer_email: EmailStr
    customer_name: Optional[str] = None
    items: List[PaymentItemRequest]
    payment_method: str = "card"
    currency: str = "krw"
    description: Optional[str] = None
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class CreateSubscriptionRequest(BaseModel):
    customer_email: EmailStr
    customer_name: Optional[str] = None
    plan_id: str
    plan_name: str
    amount: Decimal
    billing_cycle: str = "monthly"
    trial_days: Optional[int] = None
    currency: str = "krw"

class RefundRequest(BaseModel):
    amount: Optional[Decimal] = None
    reason: Optional[str] = None

# 이메일 서비스 API

@router.post("/email/send")
async def send_email(
    request: SendEmailRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permission("email.send"))
):
    """이메일 전송"""
    try:
        email_service = get_email_service()
        if not email_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email service not available"
            )

        # 우선순위 매핑
        priority_map = {
            "low": EmailPriority.LOW,
            "normal": EmailPriority.NORMAL,
            "high": EmailPriority.HIGH,
            "urgent": EmailPriority.URGENT
        }

        email_id = await email_service.send_email(
            recipients=request.recipients,
            subject=request.subject,
            html_content=request.html_content,
            text_content=request.text_content,
            priority=priority_map[request.priority]
        )

        return {
            "email_id": email_id,
            "status": "queued",
            "recipients_count": len(request.recipients)
        }

    except Exception as e:
        logger.error(f"Email send failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send email"
        )

@router.post("/email/send-template")
async def send_template_email(
    request: SendTemplateEmailRequest,
    current_user: Dict[str, Any] = Depends(require_permission("email.send"))
):
    """템플릿 이메일 전송"""
    try:
        email_service = get_email_service()
        if not email_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email service not available"
            )

        priority_map = {
            "low": EmailPriority.LOW,
            "normal": EmailPriority.NORMAL,
            "high": EmailPriority.HIGH,
            "urgent": EmailPriority.URGENT
        }

        email_id = await email_service.send_template_email(
            recipients=request.recipients,
            template_name=request.template_name,
            template_variables=request.template_variables,
            priority=priority_map[request.priority]
        )

        return {
            "email_id": email_id,
            "status": "queued",
            "template": request.template_name,
            "recipients_count": len(request.recipients)
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Template email send failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send template email"
        )

@router.get("/email/status/{email_id}")
async def get_email_status(
    email_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("email.view"))
) -> EmailStatusResponse:
    """이메일 상태 조회"""
    try:
        email_service = get_email_service()
        if not email_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email service not available"
            )

        email_status = email_service.get_email_status(email_id)
        if not email_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Email not found"
            )

        return EmailStatusResponse(
            email_id=email_id,
            status=email_status.value,
            sent_at=None,  # 실제 데이터에서 가져와야 함
            delivered_at=None,
            error_message=None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check email status"
        )

@router.get("/email/stats")
async def get_email_stats(
    current_user: Dict[str, Any] = Depends(require_permission("email.stats"))
):
    """이메일 통계 조회"""
    try:
        email_service = get_email_service()
        if not email_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email service not available"
            )

        stats = email_service.get_delivery_stats()
        return stats

    except Exception as e:
        logger.error(f"Email stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get email statistics"
        )

# SMS 서비스 API

@router.post("/sms/send")
async def send_sms(
    request: SendSMSRequest,
    current_user: Dict[str, Any] = Depends(require_permission("sms.send"))
):
    """SMS 전송"""
    try:
        sms_service = get_sms_service()
        if not sms_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SMS service not available"
            )

        sms_type_map = {
            "sms": SMSType.SMS,
            "lms": SMSType.LMS,
            "mms": SMSType.MMS
        }

        sms_id = await sms_service.send_sms(
            recipients=request.recipients,
            content=request.content,
            sms_type=sms_type_map[request.sms_type]
        )

        return {
            "sms_id": sms_id,
            "status": "queued",
            "recipients_count": len(request.recipients),
            "sms_type": request.sms_type
        }

    except Exception as e:
        logger.error(f"SMS send failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send SMS"
        )

@router.post("/sms/send-template")
async def send_template_sms(
    request: SendTemplateSMSRequest,
    current_user: Dict[str, Any] = Depends(require_permission("sms.send"))
):
    """템플릿 SMS 전송"""
    try:
        sms_service = get_sms_service()
        if not sms_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SMS service not available"
            )

        sms_id = await sms_service.send_template_sms(
            recipients=request.recipients,
            template_name=request.template_name,
            template_variables=request.template_variables
        )

        return {
            "sms_id": sms_id,
            "status": "queued",
            "template": request.template_name,
            "recipients_count": len(request.recipients)
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Template SMS send failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send template SMS"
        )

@router.post("/sms/send-verification")
async def send_verification_sms(
    request: SendVerificationSMSRequest,
    current_user: Dict[str, Any] = Depends(require_permission("sms.verification"))
):
    """인증 SMS 전송"""
    try:
        sms_service = get_sms_service()
        if not sms_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SMS service not available"
            )

        sms_id = await sms_service.send_verification_sms(
            phone_number=request.phone_number,
            code=request.code,
            app_name=request.app_name,
            expiry_minutes=request.expiry_minutes
        )

        return {
            "sms_id": sms_id,
            "status": "queued",
            "phone_number": request.phone_number,
            "expires_in_minutes": request.expiry_minutes
        }

    except Exception as e:
        logger.error(f"Verification SMS send failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification SMS"
        )

@router.get("/sms/stats")
async def get_sms_stats(
    current_user: Dict[str, Any] = Depends(require_permission("sms.stats"))
):
    """SMS 통계 조회"""
    try:
        sms_service = get_sms_service()
        if not sms_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SMS service not available"
            )

        stats = sms_service.get_delivery_stats()
        return stats

    except Exception as e:
        logger.error(f"SMS stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get SMS statistics"
        )

# 결제 서비스 API

@router.post("/payment/create")
async def create_payment(
    request: CreatePaymentRequest,
    current_user: Dict[str, Any] = Depends(require_permission("payment.create"))
):
    """결제 생성"""
    try:
        payment_service = get_payment_service()
        if not payment_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Payment service not available"
            )

        # 고객 정보 생성
        customer = PaymentCustomer(
            id=str(current_user.get("user_id")),
            email=request.customer_email,
            name=request.customer_name
        )

        # 결제 항목 변환
        items = [
            PaymentItem(
                name=item.name,
                quantity=item.quantity,
                unit_price=item.unit_price,
                description=item.description
            )
            for item in request.items
        ]

        # 결제 방법 매핑
        payment_method_map = {
            "card": PaymentMethod.CARD,
            "bank_transfer": PaymentMethod.BANK_TRANSFER,
            "digital_wallet": PaymentMethod.DIGITAL_WALLET
        }

        # 통화 매핑
        currency_map = {
            "usd": CurrencyCode.USD,
            "krw": CurrencyCode.KRW,
            "eur": CurrencyCode.EUR,
            "jpy": CurrencyCode.JPY
        }

        transaction = await payment_service.create_payment(
            customer=customer,
            items=items,
            payment_method=payment_method_map[request.payment_method],
            currency=currency_map[request.currency],
            description=request.description,
            success_url=request.success_url,
            cancel_url=request.cancel_url
        )

        return {
            "transaction_id": transaction.id,
            "status": transaction.status.value,
            "total_amount": float(transaction.total_amount),
            "currency": transaction.currency.value,
            "payment_url": transaction.provider_payment_url,
            "expires_at": transaction.expires_at.isoformat() if transaction.expires_at else None
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Payment creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment"
        )

@router.post("/payment/subscription")
async def create_subscription(
    request: CreateSubscriptionRequest,
    current_user: Dict[str, Any] = Depends(require_permission("subscription.create"))
):
    """구독 생성"""
    try:
        payment_service = get_payment_service()
        if not payment_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Payment service not available"
            )

        # 고객 정보 생성
        customer = PaymentCustomer(
            id=str(current_user.get("user_id")),
            email=request.customer_email,
            name=request.customer_name
        )

        # 통화 매핑
        currency_map = {
            "usd": CurrencyCode.USD,
            "krw": CurrencyCode.KRW,
            "eur": CurrencyCode.EUR,
            "jpy": CurrencyCode.JPY
        }

        subscription = await payment_service.create_subscription(
            customer=customer,
            plan_id=request.plan_id,
            plan_name=request.plan_name,
            amount=request.amount,
            billing_cycle=request.billing_cycle,
            trial_days=request.trial_days,
            currency=currency_map[request.currency]
        )

        return {
            "subscription_id": subscription.id,
            "status": subscription.status.value,
            "plan_id": subscription.plan_id,
            "plan_name": subscription.plan_name,
            "amount": float(subscription.amount),
            "billing_cycle": subscription.billing_cycle,
            "current_period_end": subscription.current_period_end.isoformat(),
            "trial_end": subscription.trial_end.isoformat() if subscription.trial_end else None
        }

    except Exception as e:
        logger.error(f"Subscription creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription"
        )

@router.post("/payment/{transaction_id}/refund")
async def refund_payment(
    transaction_id: str,
    request: RefundRequest,
    current_user: Dict[str, Any] = Depends(require_permission("payment.refund"))
):
    """결제 환불"""
    try:
        payment_service = get_payment_service()
        if not payment_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Payment service not available"
            )

        success = await payment_service.refund_payment(
            transaction_id=transaction_id,
            amount=request.amount
        )

        if success:
            return {
                "transaction_id": transaction_id,
                "status": "refund_processed",
                "refund_amount": float(request.amount) if request.amount else None,
                "reason": request.reason
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Refund failed"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Refund failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process refund"
        )

@router.delete("/payment/subscription/{subscription_id}")
async def cancel_subscription(
    subscription_id: str,
    cancel_at_period_end: bool = True,
    current_user: Dict[str, Any] = Depends(require_permission("subscription.cancel"))
):
    """구독 취소"""
    try:
        payment_service = get_payment_service()
        if not payment_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Payment service not available"
            )

        success = await payment_service.cancel_subscription(
            subscription_id=subscription_id,
            cancel_at_period_end=cancel_at_period_end
        )

        if success:
            return {
                "subscription_id": subscription_id,
                "status": "cancelled" if not cancel_at_period_end else "cancel_at_period_end",
                "cancel_at_period_end": cancel_at_period_end
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Subscription cancellation failed"
            )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Subscription cancellation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )

@router.get("/payment/stats")
async def get_payment_stats(
    current_admin: Dict[str, Any] = Depends(require_admin)
):
    """결제 통계 조회 (관리자 전용)"""
    try:
        payment_service = get_payment_service()
        if not payment_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Payment service not available"
            )

        stats = payment_service.get_payment_stats()
        return stats

    except Exception as e:
        logger.error(f"Payment stats retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get payment statistics"
        )

# 전체 서비스 상태 확인

@router.get("/services/status")
async def get_services_status(
    current_admin: Dict[str, Any] = Depends(require_admin)
):
    """모든 외부 서비스 상태 확인"""
    try:
        services_status = {
            "email_service": {
                "available": get_email_service() is not None,
                "stats": get_email_service().get_delivery_stats() if get_email_service() else None
            },
            "sms_service": {
                "available": get_sms_service() is not None,
                "stats": get_sms_service().get_delivery_stats() if get_sms_service() else None
            },
            "payment_service": {
                "available": get_payment_service() is not None,
                "stats": get_payment_service().get_payment_stats() if get_payment_service() else None
            }
        }

        return services_status

    except Exception as e:
        logger.error(f"Service status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check service status"
        )